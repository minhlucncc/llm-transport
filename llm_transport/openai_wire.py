"""OpenAI chat-completions wire backend (httpx).

POSTs ``{base_url}/chat/completions`` with ``Authorization: Bearer``. The
Anthropic↔OpenAI translation (messages, tools, finish_reason, tolerant tool-arg
parsing, ``<think>`` stripping) is ported from the proven
``rag_core/llm/client.py`` adapter; only the httpx SSE transport underneath is
new. ``base_url`` is used as-is plus ``/chat/completions`` — matching how the
OpenAI SDK appended to a configured ``base_url`` (so an NCC gateway without
``/v1`` keeps working).
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from ._http import make_client, map_request_error, status_error
from ._retry import with_retries
from .errors import TransportProtocolError
from .events import StreamDone, StreamEvent, TextDelta
from .sse import iter_sse_json
from .thinkfilter import ThinkTagFilter, strip_think
from .types import LlmRequest, LlmResponse, TextBlock, ToolUseBlock, Usage


def _response_mapping(value: Any, *, stream: bool = False) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TransportProtocolError("invalid_chunk_type" if stream else "invalid_response_type")
    return value


def _response_choices(value: Mapping[str, Any]) -> Sequence[Any]:
    if "choices" not in value:
        raise TransportProtocolError("missing_choices")
    choices = value["choices"]
    if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes, bytearray)):
        raise TransportProtocolError("invalid_choices")
    return choices


# ── Anthropic → OpenAI translation (ported from rag_core/llm/client.py) ─────────


def _system_text(system: str | list[dict] | None) -> str | None:
    """Flatten an Anthropic ``system`` (str OR cache_control block array) to text.

    The OpenAI wire has no cache_control concept, so block prefixes collapse to
    a single system string — the cache breakpoint is simply ignored here.
    """
    if system is None:
        return None
    if isinstance(system, str):
        return system
    parts = [b.get("text", "") for b in system if isinstance(b, dict) and b.get("type") == "text"]
    joined = "\n".join(p for p in parts if p)
    return joined or None


def _to_openai_messages(system: str | None, messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )
            elif btype == "tool_result":
                tc = block.get("content", "")
                if not isinstance(tc, str):
                    tc = json.dumps(tc)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": tc,
                    }
                )

        if role == "assistant":
            entry: dict[str, Any] = {"role": "assistant"}
            entry["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        else:
            out.extend(tool_results)
            if text_parts or not tool_results:
                out.append({"role": role, "content": "\n".join(text_parts)})

    return out


def _to_openai_tools(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _tool_args(raw: Any) -> dict[str, Any]:
    """Tool-call ``arguments`` → dict; some providers send a dict, not JSON text."""
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stop_reason(finish_reason: str | None) -> str:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _blocks_from_message(message: dict) -> list[Any]:
    """OpenAI choice ``message`` dict → Anthropic-shaped content blocks."""
    blocks: list[Any] = []
    text = message.get("content")
    if text:
        text = strip_think(text)
    if text:
        blocks.append(TextBlock(text=text))
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = _tool_args(fn.get("arguments"))
        blocks.append(ToolUseBlock(id=tc.get("id") or "", name=fn.get("name") or "", input=args))
    return blocks


def _usage_from(usage: dict | None) -> Usage:
    if not usage:
        return Usage()
    return Usage(
        input_tokens=usage.get("prompt_tokens") or 0,
        output_tokens=usage.get("completion_tokens") or 0,
    )


# ── the backend ────────────────────────────────────────────────────────────────


class OpenAIWireTransport:
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        base = (base_url or "https://api.openai.com/v1").rstrip("/")
        self._url = f"{base}/chat/completions"
        self._api_key = api_key or "missing-key"
        self._timeout = timeout
        self._max_retries = max_retries
        self._client, self._owns_client = make_client(timeout, http_client)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_body(self, req: LlmRequest, *, stream: bool) -> dict:
        body: dict[str, Any] = {
            "model": req.model,
            "messages": _to_openai_messages(_system_text(req.system), req.messages),
        }
        tools = _to_openai_tools(req.tools)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.max_tokens is not None:
            body["max_tokens"] = req.max_tokens
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, req: LlmRequest) -> LlmResponse:
        body = self._build_body(req, stream=False)

        async def _do() -> dict:
            try:
                resp = await self._client.post(self._url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                raise map_request_error(exc) from exc
            if resp.status_code >= 400:
                raise status_error(resp.status_code, resp.text)
            return resp.json()

        data = _response_mapping(await with_retries(_do, max_retries=self._max_retries))
        choices = _response_choices(data)
        if not choices:
            return LlmResponse()
        choice = _response_mapping(choices[0])
        message = choice.get("message")
        if not isinstance(message, Mapping):
            raise TransportProtocolError("missing_message")
        return LlmResponse(
            content=_blocks_from_message(dict(message)),
            stop_reason=_stop_reason(choice.get("finish_reason")),
            usage=_usage_from(data.get("usage")),
        )

    def stream(self, req: LlmRequest) -> _OpenAIStream:
        return _OpenAIStream(self, self._build_body(req, stream=True))


class _OpenAIStream:
    """Async ctx manager: SSE over chat-completions, reassembled to a ``LlmResponse``."""

    def __init__(self, transport: OpenAIWireTransport, body: dict) -> None:
        self._t = transport
        self._body = body
        self._cm: Any = None
        self._response: httpx.Response | None = None
        self._text = ""
        self._tool_calls: dict[int, dict] = {}
        self._finish_reason: str | None = None
        self._usage = Usage()
        self._drained = False
        self._think = ThinkTagFilter()

    async def __aenter__(self) -> _OpenAIStream:
        # Retry the connection/first-response ONLY — never a stream mid-flight.
        async def _open():
            cm = self._t._client.stream(
                "POST", self._t._url, json=self._body, headers=self._t._headers
            )
            try:
                resp = await cm.__aenter__()
            except httpx.HTTPError as exc:
                raise map_request_error(exc) from exc
            if resp.status_code >= 400:
                body = await resp.aread()
                await cm.__aexit__(None, None, None)
                raise status_error(resp.status_code, body.decode("utf-8", "replace"))
            return cm, resp

        self._cm, self._response = await with_retries(_open, max_retries=self._t._max_retries)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._cm is not None:
            with contextlib.suppress(Exception):  # best-effort cleanup
                await self._cm.__aexit__(*exc if any(exc) else (None, None, None))

    async def _consume(self) -> AsyncIterator[str]:
        assert self._response is not None
        async for data in iter_sse_json(self._response, include_non_objects=True):
            data = _response_mapping(data, stream=True)
            usage = data.get("usage")
            if usage is not None:
                self._usage = _usage_from(usage)
            choices = _response_choices(data)
            if not choices:
                continue
            choice = _response_mapping(choices[0])
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if delta is None:
                continue
            delta = _response_mapping(delta, stream=True)
            for tc in delta.get("tool_calls") or []:
                tc = _response_mapping(tc, stream=True)
                slot = self._tool_calls.setdefault(
                    tc.get("index", 0), {"id": "", "name": "", "arguments": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function")
                if fn is not None:
                    fn = _response_mapping(fn, stream=True)
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    args = fn.get("arguments")
                    if args:
                        if isinstance(args, dict):
                            slot["arguments"] = args
                        elif isinstance(slot["arguments"], str):
                            slot["arguments"] += args
            text = delta.get("content")
            if text:
                out = self._think.feed(text)
                if out:
                    self._text += out
                    yield out
        tail = self._think.flush()
        if tail:
            self._text += tail
            yield tail
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
        for _, slot in sorted(self._tool_calls.items()):
            blocks.append(
                ToolUseBlock(id=slot["id"], name=slot["name"], input=_tool_args(slot["arguments"]))
            )
        return LlmResponse(
            content=blocks, stop_reason=_stop_reason(self._finish_reason), usage=self._usage
        )

    async def events(self) -> AsyncIterator[StreamEvent]:
        async for delta in self._consume():
            yield TextDelta(delta)
        yield StreamDone(await self.get_final_message())
