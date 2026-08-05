"""Gemini native API wire backend (httpx).

POSTs the Google Generative Language API directly —
``{root}/v1beta/models/{model}:generateContent``, or
``:streamGenerateContent?alt=sse`` for streams — authenticating with
``x-goog-api-key``.

Why native rather than an Anthropic-compat shim: **Google does not serve an
Anthropic-Messages surface.** ``POST /v1beta/anthropic/messages`` returns a bare
404 for every credential (no key, invalid key, OAuth bearer, with and without
``anthropic-version``). Google's only documented compatibility layer is
OpenAI-shaped and authenticates with ``Authorization: Bearer``, which would
break the ``x-goog-api-key`` credential contract in
``openspec/specs/provider-configuration/spec.md``. So this wire speaks the
native API and does the Anthropic->Gemini translation here.

Translation (Anthropic request shape in, normalized response out):

=========================  =======================================
Anthropic ``LlmRequest``   Gemini native
=========================  =======================================
``messages[].role``        ``contents[].role`` (``assistant``->``model``)
``content[] text``         ``parts[].text``
``content[] tool_use``     ``parts[].functionCall``
``content[] tool_result``  ``parts[].functionResponse``
``system``                 ``systemInstruction.parts[].text``
``tools``                  ``tools[0].functionDeclarations``
``temperature``            ``generationConfig.temperature``
``max_tokens``             ``generationConfig.maxOutputTokens``
=========================  =======================================

Three shape mismatches are handled explicitly:

- **Gemini function calls carry no id.** Anthropic pairs ``tool_use`` with
  ``tool_result`` by ``id``; Gemini pairs ``functionCall`` with
  ``functionResponse`` by ``name``. Emitted ids are synthesized
  (``gemini-call-<n>-<name>``) and resolved back to a name on the return leg
  from the conversation itself (:func:`_tool_name_index`), falling back to
  decoding the synthesized id.
- **Function-declaration schemas are an OpenAPI subset.** ``input_schema`` is
  run through :func:`_sanitize_schema`: unsupported JSON-Schema keywords
  (``$schema``, ``additionalProperties``, ``allOf``, ``$ref``, ...) are dropped
  and ``type`` is upper-cased to the proto enum spelling, so a tool that
  validates under Anthropic is not rejected by Google.
- **Thinking parts are not answer text.** Parts flagged ``thought: true`` are
  skipped, mirroring the Anthropic wire, which never surfaces reasoning as text.

``max_tokens`` is defended (default 4096) so no request is unbounded. NOTE: on
the thinking models (``gemini-2.5-*``) ``maxOutputTokens`` also covers thinking
tokens, so a small per-stage budget can be spent before any visible text is
produced — size ``BotPolicy`` stage budgets accordingly.

Error envelopes map Google's ``{"error": {"code", "message", "status"}}`` shape
into the owned taxonomy (429/5xx classification) with the server's ``message``
as the exception text — the credential only ever travels in a request header,
so it cannot leak into an error.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import quote

import httpx

from ._http import make_client, map_request_error
from ._retry import with_retries
from .errors import (
    TransportProtocolError,
    TransportRateLimit,
    TransportServerError,
    TransportStatusError,
)
from .events import StreamDone, StreamEvent, TextDelta
from .sse import iter_sse_json
from .types import LlmRequest, LlmResponse, TextBlock, ToolUseBlock, Usage

_DEFAULT_ROOT = "https://generativelanguage.googleapis.com"
_API_VERSION = "v1beta"
_AUTH_HEADER = "x-goog-api-key"
_DEFAULT_MAX_TOKENS = 4096  # never issue an unbounded request

# Stored bases that collapse to the host root before the native path is
# appended. ``/v1beta/anthropic`` was the original c0036 preset default; saved
# profiles still carry it, so it is normalized rather than rejected.
_STRIPPABLE_SUFFIXES = ("/v1beta/anthropic", "/v1beta/openai", "/v1beta", "/v1")

_CALL_ID_RE = re.compile(r"^gemini-call-\d+-(?P<name>.+)$")


def _host_root(base_url: str | None) -> str:
    """Normalize a stored base URL to the host root (never a doubled API path)."""
    base = (base_url or _DEFAULT_ROOT).strip().rstrip("/")
    stripped = True
    while stripped:
        stripped = False
        for suffix in _STRIPPABLE_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)].rstrip("/")
                stripped = True
                break
    return base or _DEFAULT_ROOT


def _model_path(model: str) -> str:
    """``gemini-2.5-flash`` / ``models/gemini-2.5-flash`` -> a single path segment."""
    name = (model or "").strip()
    if name.startswith("models/"):
        name = name[len("models/") :]
    return quote(name, safe="")


def _google_status_error(status_code: int, body: str | None) -> TransportStatusError:
    """Map a Google ``{"error": {"code", "message", "status"}}`` envelope to the taxonomy."""
    envelope_code = status_code
    message = ""
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        # Some Google endpoints wrap the envelope in a single-element array.
        if isinstance(payload, list) and payload:
            payload = payload[0]
        error = payload.get("error") if isinstance(payload, Mapping) else None
        if isinstance(error, Mapping):
            code = error.get("code")
            if isinstance(code, int):
                envelope_code = code
            msg = error.get("message")
            if isinstance(msg, str):
                message = msg
    if envelope_code == 429:
        return TransportRateLimit(envelope_code, message, body=body)
    if envelope_code >= 500:
        return TransportServerError(envelope_code, message, body=body)
    return TransportStatusError(envelope_code, message, body=body)


# -- Anthropic -> Gemini request translation -----------------------------------


def _system_text(system: str | list[dict] | None) -> str | None:
    """Flatten an Anthropic ``system`` (str OR cache_control block array) to text.

    Gemini has no ``cache_control`` concept on ``systemInstruction``, so block
    prefixes collapse to one string and the cache breakpoint is dropped.
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system or None
    parts = [
        b.get("text", "") for b in system if isinstance(b, Mapping) and b.get("type") == "text"
    ]
    joined = "\n".join(p for p in parts if p)
    return joined or None


# The Schema subset Gemini accepts inside a functionDeclaration. Anything else
# ($schema, additionalProperties, $ref, allOf, oneOf, not, ...) is dropped.
_SCHEMA_KEYS = frozenset(
    {
        "type",
        "format",
        "title",
        "description",
        "nullable",
        "enum",
        "items",
        "properties",
        "required",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "default",
        "example",
        "anyOf",
        "propertyOrdering",
    }
)
_JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object", "null"})


def _schema_type(value: Any) -> tuple[str | None, bool]:
    """JSON-Schema ``type`` -> ``(proto enum spelling, nullable)``.

    Accepts the union spelling (``["string", "null"]``) that JSON Schema allows
    but Gemini does not, folding the null arm into ``nullable``.
    """
    if isinstance(value, str):
        return (value.upper() if value.lower() in _JSON_TYPES else None), False
    if isinstance(value, list):
        nullable = any(isinstance(v, str) and v.lower() == "null" for v in value)
        for v in value:
            if isinstance(v, str) and v.lower() in _JSON_TYPES and v.lower() != "null":
                return v.upper(), nullable
        return None, nullable
    return None, False


def _sanitize_schema(node: Any) -> dict[str, Any]:
    """Reduce a JSON Schema to the OpenAPI subset Gemini accepts."""
    if not isinstance(node, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "const":
            out["enum"] = [value]  # Gemini has no ``const``; a 1-value enum is equivalent
            continue
        if key not in _SCHEMA_KEYS:
            continue
        if key == "type":
            type_value, nullable = _schema_type(value)
            if type_value:
                out["type"] = type_value
            if nullable:
                out["nullable"] = True
        elif key == "properties" and isinstance(value, Mapping):
            out["properties"] = {str(k): _sanitize_schema(v) for k, v in value.items()}
        elif key == "items":
            out["items"] = _sanitize_schema(value)
        elif key == "anyOf" and isinstance(value, list):
            out["anyOf"] = [_sanitize_schema(v) for v in value]
        else:
            out[key] = value
    return out


def _to_function_declarations(tools: list[dict] | None) -> list[dict]:
    decls: list[dict] = []
    for tool in tools or []:
        if not isinstance(tool, Mapping):
            continue
        name = tool.get("name") or ""
        if not name:
            continue
        decl: dict[str, Any] = {"name": str(name)}
        description = tool.get("description")
        if description:
            decl["description"] = description
        parameters = _sanitize_schema(tool.get("input_schema"))
        # Gemini rejects an OBJECT declaration with no properties, so a
        # parameterless tool simply omits ``parameters``.
        if parameters.get("properties"):
            parameters.setdefault("type", "OBJECT")
            decl["parameters"] = parameters
        decls.append(decl)
    return decls


def _tool_name_index(messages: list[dict] | None) -> dict[str, str]:
    """Map every ``tool_use`` id in the conversation to its function name.

    Gemini's ``functionResponse`` pairs by name, so the id the engine echoes
    back in a ``tool_result`` has to be resolved to one.
    """
    index: dict[str, str] = {}
    for msg in messages or []:
        if not isinstance(msg, Mapping):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                block_id, block_name = block.get("id"), block.get("name")
                if block_id and block_name:
                    index[str(block_id)] = str(block_name)
    return index


def _resolve_tool_name(tool_use_id: Any, index: dict[str, str]) -> str:
    tid = str(tool_use_id or "")
    if tid in index:
        return index[tid]
    match = _CALL_ID_RE.match(tid)  # our synthesized id carries the name
    return match.group("name") if match else tid


def _function_response_payload(content: Any) -> dict[str, Any]:
    """``functionResponse.response`` must be a JSON object — wrap anything else."""
    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, list):
        texts = [
            b.get("text", "") for b in content if isinstance(b, Mapping) and b.get("type") == "text"
        ]
        if any(texts):
            return {"result": "\n".join(t for t in texts if t)}
        return {"result": json.dumps(content)}
    if isinstance(content, str):
        return {"result": content}
    return {"result": "" if content is None else json.dumps(content)}


def _to_contents(messages: list[dict] | None, index: dict[str, str]) -> list[dict]:
    contents: list[dict] = []

    def _append(role: str, parts: list[dict]) -> None:
        if not parts:
            return
        # Gemini expects alternating turns; merging consecutive same-role parts
        # keeps a tool_result that follows a user message from splitting a turn.
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for msg in messages or []:
        if not isinstance(msg, Mapping):
            continue
        role = "model" if msg.get("role") == "assistant" else "user"
        content = msg.get("content", "")

        if isinstance(content, str):
            if content:
                _append(role, [{"text": content}])
            continue
        if not isinstance(content, list):
            continue

        parts: list[dict] = []
        results: list[dict] = []
        for block in content:
            if not isinstance(block, Mapping):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    parts.append({"text": text})
            elif btype == "tool_use":
                parts.append(
                    {
                        "functionCall": {
                            "name": str(block.get("name") or ""),
                            "args": block.get("input") or {},
                        }
                    }
                )
            elif btype == "tool_result":
                results.append(
                    {
                        "functionResponse": {
                            "name": _resolve_tool_name(block.get("tool_use_id"), index),
                            "response": _function_response_payload(block.get("content")),
                        }
                    }
                )

        # Tool results are always a user turn, even when Anthropic batched them
        # into the same message as text (matching the OpenAI wire's ordering).
        if results:
            _append("user", results)
        if parts:
            _append(role, parts)

    return contents


# -- Gemini -> normalized response translation ---------------------------------


def _blocks_from_parts(parts: Any) -> tuple[list[Any], bool]:
    """``candidate.content.parts[]`` -> content blocks + whether a call was seen."""
    blocks: list[Any] = []
    buffer: list[str] = []
    saw_call = False

    def _flush() -> None:
        if buffer:
            joined = "".join(buffer)
            buffer.clear()
            if joined:
                blocks.append(TextBlock(text=joined))

    for i, part in enumerate(parts or []):
        if not isinstance(part, Mapping):
            continue
        if part.get("thought"):
            continue  # thinking summary is never answer text
        call = part.get("functionCall")
        if isinstance(call, Mapping):
            _flush()
            saw_call = True
            name = str(call.get("name") or "")
            args = call.get("args")
            blocks.append(
                ToolUseBlock(
                    id=f"gemini-call-{i}-{name}",
                    name=name,
                    input=dict(args) if isinstance(args, Mapping) else {},
                )
            )
            continue
        text = part.get("text")
        if isinstance(text, str) and text:
            buffer.append(text)

    _flush()
    return blocks, saw_call


def _stop_reason(finish_reason: Any, *, saw_tool_call: bool) -> str:
    """Normalize to our ``{end_turn, tool_use, max_tokens}``.

    Gemini reports ``STOP`` even when the candidate is a function call, so a
    seen call decides ``tool_use`` — the cite/filter stages depend on it.
    """
    if saw_tool_call:
        return "tool_use"
    if isinstance(finish_reason, str) and finish_reason.upper() == "MAX_TOKENS":
        return "max_tokens"
    return "end_turn"


def _usage_from(raw: Any) -> Usage:
    if not isinstance(raw, Mapping):
        return Usage()
    output = raw.get("candidatesTokenCount") or 0
    thoughts = raw.get("thoughtsTokenCount")
    if isinstance(thoughts, int):
        output += thoughts  # thinking tokens bill as output; keep the rollup honest
    cached = raw.get("cachedContentTokenCount")
    return Usage(
        input_tokens=raw.get("promptTokenCount") or 0,
        output_tokens=output,
        cache_read_input_tokens=cached if isinstance(cached, int) else None,
        # Gemini's explicit caching is a separate API with no creation counter
        # on this response, so the field stays unreported rather than zeroed.
        cache_creation_input_tokens=None,
    )


# -- the backend ---------------------------------------------------------------


class GeminiWireTransport:
    """Native Gemini ``generateContent`` transport authenticating with ``x-goog-api-key``."""

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._root = _host_root(base_url)
        self._api_key = api_key or "missing-key"
        self._timeout = timeout
        self._max_retries = max_retries
        self._client, self._owns_client = make_client(timeout, http_client)

    @property
    def _headers(self) -> dict[str, str]:
        return {_AUTH_HEADER: self._api_key, "content-type": "application/json"}

    def _url(self, model: str, *, stream: bool) -> str:
        method = "streamGenerateContent" if stream else "generateContent"
        url = f"{self._root}/{_API_VERSION}/models/{_model_path(model)}:{method}"
        return f"{url}?alt=sse" if stream else url

    def _status_error(self, status_code: int, body: str) -> TransportStatusError:
        return _google_status_error(status_code, body)

    def _build_body(self, req: LlmRequest) -> dict:
        body: dict[str, Any] = {
            "contents": _to_contents(req.messages, _tool_name_index(req.messages))
        }

        system = _system_text(req.system)
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}

        decls = _to_function_declarations(req.tools)
        if decls:
            body["tools"] = [{"functionDeclarations": decls}]
            # cite/filter dispatch: AUTO keeps tool use available on every stage
            # that declares tools.
            body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        generation: dict[str, Any] = {
            "maxOutputTokens": (
                req.max_tokens if req.max_tokens is not None else _DEFAULT_MAX_TOKENS
            )
        }
        if req.temperature is not None:
            generation["temperature"] = req.temperature
        body["generationConfig"] = generation
        return body

    async def complete(self, req: LlmRequest) -> LlmResponse:
        url = self._url(req.model, stream=False)
        body = self._build_body(req)

        async def _do() -> Any:
            try:
                resp = await self._client.post(url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                raise map_request_error(exc) from exc
            if resp.status_code >= 400:
                raise self._status_error(resp.status_code, resp.text)
            return resp.json()

        data = await with_retries(_do, max_retries=self._max_retries)
        if not isinstance(data, Mapping):
            raise TransportProtocolError("invalid_response_type")

        usage = _usage_from(data.get("usageMetadata"))
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            # A safety/recitation block returns no candidate. Surface an empty
            # answer rather than inventing text — the filter stage refuses on
            # missing citations, which is the correct grounded outcome.
            return LlmResponse(usage=usage)

        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            raise TransportProtocolError("invalid_candidate")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else None
        blocks, saw_call = _blocks_from_parts(parts)
        return LlmResponse(
            content=blocks,
            stop_reason=_stop_reason(candidate.get("finishReason"), saw_tool_call=saw_call),
            usage=usage,
        )

    def stream(self, req: LlmRequest) -> _GeminiStream:
        return _GeminiStream(self, self._url(req.model, stream=True), self._build_body(req))


class _GeminiStream:
    """Async ctx manager: ``streamGenerateContent`` SSE reassembled to a ``LlmResponse``."""

    def __init__(self, transport: GeminiWireTransport, url: str, body: dict) -> None:
        self._t = transport
        self._url = url
        self._body = body
        self._cm: Any = None
        self._response: httpx.Response | None = None
        self._text = ""
        self._calls: list[dict] = []
        self._finish_reason: Any = None
        self._usage = Usage()
        self._drained = False

    async def __aenter__(self) -> _GeminiStream:
        # Retry the connection/first-response ONLY — never a stream mid-flight.
        async def _open():
            cm = self._t._client.stream(
                "POST", self._url, json=self._body, headers=self._t._headers
            )
            try:
                resp = await cm.__aenter__()
            except httpx.HTTPError as exc:
                raise map_request_error(exc) from exc
            if resp.status_code >= 400:
                body = await resp.aread()
                await cm.__aexit__(None, None, None)
                raise self._t._status_error(resp.status_code, body.decode("utf-8", "replace"))
            return cm, resp

        self._cm, self._response = await with_retries(_open, max_retries=self._t._max_retries)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._cm is not None:
            with contextlib.suppress(Exception):  # best-effort cleanup
                await self._cm.__aexit__(*exc if any(exc) else (None, None, None))

    async def _consume(self) -> AsyncIterator[str]:
        assert self._response is not None
        async for data in iter_sse_json(self._response):
            usage = data.get("usageMetadata")
            if usage is not None:
                self._usage = _usage_from(usage)
            candidates = data.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                continue
            candidate = candidates[0]
            if not isinstance(candidate, Mapping):
                continue
            if candidate.get("finishReason"):
                self._finish_reason = candidate["finishReason"]
            content = candidate.get("content")
            parts = content.get("parts") if isinstance(content, Mapping) else None
            for part in parts or []:
                if not isinstance(part, Mapping):
                    continue
                if part.get("thought"):
                    continue
                call = part.get("functionCall")
                if isinstance(call, Mapping):
                    args = call.get("args")
                    self._calls.append(
                        {
                            "name": str(call.get("name") or ""),
                            "args": dict(args) if isinstance(args, Mapping) else {},
                        }
                    )
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    self._text += text
                    yield text
        self._drained = True

    @property
    async def text_stream(self) -> AsyncIterator[str]:
        async for delta in self._consume():
            yield delta

    async def get_final_message(self) -> LlmResponse:
        if not self._drained:
            async for _ in self._consume():
                pass
        blocks: list[Any] = []
        if self._text:
            blocks.append(TextBlock(text=self._text))
        for i, call in enumerate(self._calls):
            blocks.append(
                ToolUseBlock(
                    id=f"gemini-call-{i}-{call['name']}", name=call["name"], input=call["args"]
                )
            )
        return LlmResponse(
            content=blocks,
            stop_reason=_stop_reason(self._finish_reason, saw_tool_call=bool(self._calls)),
            usage=self._usage,
        )

    async def events(self) -> AsyncIterator[StreamEvent]:
        async for delta in self._consume():
            yield TextDelta(delta)
        yield StreamDone(await self.get_final_message())
