"""Anthropic Messages wire backend (httpx).

POSTs ``{base_url}/v1/messages`` with ``x-api-key`` + ``anthropic-version``
(matching what ``AsyncAnthropic(api_key=...)`` sends — the production default).
The auth header name is a constructor parameter (default ``"x-api-key"``) so
per-wire kinds can authenticate differently (D3 — the gemini wire passes
``"x-goog-api-key"``) without special-casing here.
``messages``/``system``/``tools`` are already Anthropic-shaped, so the request
body is a near pass-through, including ``system`` ``cache_control`` breakpoints.

Responses (and the SSE stream) are assembled into our normalized
:class:`~llm_transport.types.LlmResponse`, preserving the prompt-cache usage
fields (``cache_read_input_tokens`` / ``cache_creation_input_tokens``) that the
interpreter's usage rollup depends on. No ``<think>`` filtering on this wire —
the Anthropic wire never surfaces reasoning in text blocks.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from ._http import make_client, map_request_error, status_error
from ._retry import with_retries
from .errors import TransportStatusError
from .events import StreamDone, StreamEvent, TextDelta
from .sse import iter_sse_json
from .types import LlmRequest, LlmResponse, TextBlock, ToolUseBlock, Usage

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096  # Anthropic requires max_tokens; defend if a caller omits it.
_AI_BOX_HOST = "api.ai-box.vn"


def _stop_reason(reason: str | None) -> str:
    """Normalize an Anthropic stop_reason to our {end_turn, tool_use, max_tokens}."""
    if reason in ("tool_use", "max_tokens"):
        return reason
    return "end_turn"  # end_turn, stop_sequence, None → end_turn


def _usage_from(raw: dict | None) -> Usage:
    if not raw:
        return Usage()
    return Usage(
        input_tokens=raw.get("input_tokens") or 0,
        output_tokens=raw.get("output_tokens") or 0,
        cache_read_input_tokens=raw.get("cache_read_input_tokens"),
        cache_creation_input_tokens=raw.get("cache_creation_input_tokens"),
    )


def _blocks_from_content(content: list[dict] | None) -> list[Any]:
    blocks: list[Any] = []
    for block in content or []:
        btype = block.get("type")
        if btype == "text":
            blocks.append(TextBlock(text=block.get("text", "")))
        elif btype == "tool_use":
            blocks.append(
                ToolUseBlock(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    input=block.get("input") or {},
                )
            )
        # other block types (e.g. thinking) never surface as text on this wire
    return blocks


class AnthropicWireTransport:
    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
        auth_header: str = "x-api-key",
    ) -> None:
        base = (base_url or "https://api.anthropic.com").rstrip("/")
        self._url = f"{base}/v1/messages"
        self._uses_ai_box_bearer_compatibility = (
            urlparse(base).hostname or ""
        ).lower() == _AI_BOX_HOST
        self._api_key = api_key
        self._auth_header = auth_header
        self._timeout = timeout
        self._max_retries = max_retries
        self._client, self._owns_client = make_client(timeout, http_client)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {
            self._auth_header: self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        if self._uses_ai_box_bearer_compatibility:
            # AI Box serves Anthropic Messages but authenticates its tenant
            # token through the same Bearer convention as its OpenAI route.
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _status_error(self, status_code: int, body: str) -> TransportStatusError:
        """Map a non-2xx response to the owned taxonomy. Overridable per wire."""
        return status_error(status_code, body)

    def _build_body(self, req: LlmRequest, *, stream: bool) -> dict:
        body: dict[str, Any] = {
            "model": req.model,
            "max_tokens": req.max_tokens if req.max_tokens is not None else _DEFAULT_MAX_TOKENS,
            "messages": req.messages,
        }
        if req.system is not None:
            body["system"] = req.system  # str OR cache_control block array — pass through
        if req.tools:
            body["tools"] = req.tools
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if stream:
            body["stream"] = True
        return body

    async def complete(self, req: LlmRequest) -> LlmResponse:
        body = self._build_body(req, stream=False)

        async def _do() -> dict:
            try:
                resp = await self._client.post(self._url, json=body, headers=self._headers)
            except httpx.HTTPError as exc:
                raise map_request_error(exc) from exc
            if resp.status_code >= 400:
                raise self._status_error(resp.status_code, resp.text)
            return resp.json()

        data = await with_retries(_do, max_retries=self._max_retries)
        return LlmResponse(
            content=_blocks_from_content(data.get("content")),
            stop_reason=_stop_reason(data.get("stop_reason")),
            usage=_usage_from(data.get("usage")),
        )

    def stream(self, req: LlmRequest) -> _AnthropicStream:
        return _AnthropicStream(self, self._build_body(req, stream=True))


class _AnthropicStream:
    """Async ctx manager: Anthropic SSE messages protocol → ``LlmResponse``."""

    def __init__(self, transport: AnthropicWireTransport, body: dict) -> None:
        self._t = transport
        self._body = body
        self._cm: Any = None
        self._response: httpx.Response | None = None
        # index → {"type", "text"} | {"type", "id", "name", "json"}
        self._blocks: dict[int, dict] = {}
        self._stop_reason: str | None = None
        self._usage = Usage()
        self._drained = False

    async def __aenter__(self) -> _AnthropicStream:
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
                raise self._t._status_error(resp.status_code, body.decode("utf-8", "replace"))
            return cm, resp

        self._cm, self._response = await with_retries(_open, max_retries=self._t._max_retries)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._cm is not None:
            with contextlib.suppress(Exception):  # best-effort cleanup
                await self._cm.__aexit__(*exc if any(exc) else (None, None, None))

    def _merge_usage(self, raw: dict | None) -> None:
        """Fold an event's ``usage`` into the running total.

        ``message_start`` carries input + cache fields (and an initial
        output_tokens); ``message_delta`` carries the final cumulative
        output_tokens. We keep the cache/input fields from the first sighting
        and let later output_tokens override.
        """
        if not raw:
            return
        if "input_tokens" in raw:
            self._usage.input_tokens = raw.get("input_tokens") or 0
        if raw.get("cache_read_input_tokens") is not None:
            self._usage.cache_read_input_tokens = raw["cache_read_input_tokens"]
        if raw.get("cache_creation_input_tokens") is not None:
            self._usage.cache_creation_input_tokens = raw["cache_creation_input_tokens"]
        if "output_tokens" in raw:
            self._usage.output_tokens = raw.get("output_tokens") or 0

    async def _consume(self) -> AsyncIterator[str]:
        assert self._response is not None
        async for data in iter_sse_json(self._response):
            etype = data.get("type")
            if etype == "message_start":
                self._merge_usage((data.get("message") or {}).get("usage"))
            elif etype == "content_block_start":
                idx = data.get("index", 0)
                cb = data.get("content_block") or {}
                if cb.get("type") == "text":
                    self._blocks[idx] = {"type": "text", "text": ""}
                elif cb.get("type") == "tool_use":
                    self._blocks[idx] = {
                        "type": "tool_use",
                        "id": cb.get("id", ""),
                        "name": cb.get("name", ""),
                        "json": "",
                    }
            elif etype == "content_block_delta":
                idx = data.get("index", 0)
                delta = data.get("delta") or {}
                slot = self._blocks.get(idx)
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if slot is not None:
                        slot["text"] = slot.get("text", "") + text
                    if text:
                        yield text
                elif delta.get("type") == "input_json_delta" and slot is not None:
                    slot["json"] = slot.get("json", "") + delta.get("partial_json", "")
            elif etype == "message_delta":
                d = data.get("delta") or {}
                if d.get("stop_reason"):
                    self._stop_reason = d["stop_reason"]
                self._merge_usage(data.get("usage"))
            # content_block_stop / message_stop need no handling
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
        for _, slot in sorted(self._blocks.items()):
            if slot["type"] == "text":
                if slot["text"]:
                    blocks.append(TextBlock(text=slot["text"]))
            else:
                try:
                    parsed = json.loads(slot["json"] or "{}")
                except json.JSONDecodeError:
                    parsed = {}
                blocks.append(
                    ToolUseBlock(
                        id=slot["id"],
                        name=slot["name"],
                        input=parsed if isinstance(parsed, dict) else {},
                    )
                )
        return LlmResponse(
            content=blocks, stop_reason=_stop_reason(self._stop_reason), usage=self._usage
        )

    async def events(self) -> AsyncIterator[StreamEvent]:
        async for delta in self._consume():
            yield TextDelta(delta)
        yield StreamDone(await self.get_final_message())
