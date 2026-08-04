"""Gemini wire backend tests over ``httpx.MockTransport``.

Mirrors ``test_anthropic_wire.py``: the gemini wire speaks the Anthropic
Messages shape to ``{host_root}/v1beta/anthropic/messages`` and authenticates
with ``x-goog-api-key`` — never ``x-api-key`` (D3). A stored base ending in
``/v1beta/anthropic`` is normalized to the host root so the path is never
doubled (D4, delta scenarios 'Preset-default Base URL is normalized and probed
exactly once' / 'Root-entered Base URL is probed at the anthropic-compat path').
Google ``{"error": {...}}`` envelopes map into the owned error taxonomy without
leaking the key, and probe requests stay bounded (``max_tokens=1``).

RED: no ``llm_transport/gemini_wire.py`` exists yet — collection fails on the
module import; ``build_transport`` has no gemini dispatch either.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_transport.errors import (
    TransportRateLimit,
    TransportServerError,
    TransportStatusError,
)
from llm_transport.factory import build_transport
from llm_transport.gemini_wire import GeminiWireTransport
from llm_transport.types import LlmRequest

_PRESET_BASE = "https://generativelanguage.googleapis.com/v1beta/anthropic"


def _transport(handler, **kw) -> GeminiWireTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiWireTransport(
        base_url=_PRESET_BASE, api_key=kw.pop("api_key", "gemini-key"), http_client=client, **kw
    )


def _req(**kw) -> LlmRequest:
    kw.setdefault("model", "m")
    kw.setdefault("messages", [{"role": "user", "content": "q"}])
    return LlmRequest(**kw)


# ── complete(): auth ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_sends_x_goog_api_key_and_never_x_api_key():
    """The gemini kind authenticates with ``x-goog-api-key`` only (D3/D5)."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={"content": [], "stop_reason": "end_turn", "usage": {}})

    await _transport(handler).complete(_req())
    assert seen["headers"]["x-goog-api-key"] == "gemini-key"
    assert "x-api-key" not in seen["headers"]


# ── complete(): wire shape ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_request_path_and_anthropic_messages_shape():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    r = await _transport(handler).complete(_req(system="s", max_tokens=64, temperature=0))
    assert seen["url"] == "https://generativelanguage.googleapis.com/v1beta/anthropic/messages"
    assert seen["body"]["model"] == "m"
    assert seen["body"]["messages"] == [{"role": "user", "content": "q"}]
    assert seen["body"]["system"] == "s"
    assert seen["body"]["max_tokens"] == 64
    assert seen["body"]["temperature"] == 0
    assert r.content[0].text == "answer"
    assert r.usage.input_tokens == 5 and r.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_complete_forwards_tools_and_tool_choice():
    """cite/filter dispatch: ``tools`` forwarded and ``tool_choice`` set (D1)."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [], "stop_reason": "end_turn", "usage": {}})

    tools = [{"name": "read", "description": "", "input_schema": {"type": "object"}}]
    await _transport(handler).complete(_req(tools=tools))
    assert seen["body"]["tools"] == tools
    assert seen["body"]["tool_choice"] == {"type": "auto"}


@pytest.mark.asyncio
async def test_complete_preserves_tool_use_stop_reason():
    """A ``tool_use`` response round-trips — the cite/filter stages depend on it."""

    def handler(_):
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        )

    r = await _transport(handler).complete(
        _req(tools=[{"name": "search", "description": "", "input_schema": {}}])
    )
    assert r.stop_reason == "tool_use"
    assert r.content[1].type == "tool_use" and r.content[1].name == "search"
    assert r.content[1].input == {"q": "x"}


# ── base URL normalization (D4) ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "base_url, want",
    [
        # preset default — stored /v1beta/anthropic suffix normalized to host root
        (
            "https://generativelanguage.googleapis.com/v1beta/anthropic",
            "https://generativelanguage.googleapis.com/v1beta/anthropic/messages",
        ),
        (
            "https://generativelanguage.googleapis.com/v1beta/anthropic/",
            "https://generativelanguage.googleapis.com/v1beta/anthropic/messages",
        ),
        # root-entered — anthropic-compat path appended exactly once
        (
            "https://generativelanguage.googleapis.com",
            "https://generativelanguage.googleapis.com/v1beta/anthropic/messages",
        ),
        (
            "https://generativelanguage.googleapis.com/",
            "https://generativelanguage.googleapis.com/v1beta/anthropic/messages",
        ),
    ],
)
@pytest.mark.asyncio
async def test_base_url_normalized_to_host_root_never_doubled(base_url, want):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"content": [], "stop_reason": "end_turn", "usage": {}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await GeminiWireTransport(base_url=base_url, api_key="gemini-key", http_client=client).complete(
        _req()
    )
    assert seen["url"] == want
    assert "/v1beta/anthropic/v1beta/anthropic" not in seen["url"]


# ── errors ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_maps_google_error_envelope_without_leaking_key():
    key = "sk-gemini-secret-8f3a"

    def handler(_):
        # Google error envelope: {"error": {"code", "message", "status"}}
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "API key not valid. Please pass a valid API key.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )

    with pytest.raises(TransportStatusError) as ei:
        await _transport(handler, api_key=key, max_retries=0).complete(_req())
    exc = ei.value
    assert exc.status_code == 400
    assert "API key not valid" in str(exc)  # Google message extracted into the envelope
    assert key not in str(exc)
    assert key not in (exc.body or "")


@pytest.mark.asyncio
async def test_complete_429_raises_rate_limit():
    def handler(_):
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded",
                    "status": "RESOURCE_EXHAUSTED",
                }
            },
        )

    with pytest.raises(TransportRateLimit):
        await _transport(handler, max_retries=0).complete(_req())


@pytest.mark.asyncio
async def test_complete_500_raises_server_error():
    def handler(_):
        return httpx.Response(
            500,
            json={"error": {"code": 500, "message": "Internal error", "status": "INTERNAL"}},
        )

    with pytest.raises(TransportServerError):
        await _transport(handler, max_retries=0).complete(_req())


# ── probe boundedness ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_probe_request_is_bounded_max_tokens_one():
    """The probe-shaped request is one bounded Anthropic-Messages request."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [], "stop_reason": "end_turn", "usage": {}})

    await _transport(handler).complete(_req(max_tokens=1))
    assert seen["body"]["max_tokens"] == 1


@pytest.mark.asyncio
async def test_complete_defaults_bounded_max_tokens_when_omitted():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [], "stop_reason": "end_turn", "usage": {}})

    await _transport(handler).complete(_req())  # no max_tokens
    assert seen["body"]["max_tokens"] == 4096  # never an unbounded request


# ── factory dispatch ───────────────────────────────────────────────────────────


def test_build_transport_dispatches_gemini_kind():
    t = build_transport(
        kind="gemini-compatible",
        base_url=_PRESET_BASE,
        api_key="gemini-key",
    )
    assert isinstance(t, GeminiWireTransport)
