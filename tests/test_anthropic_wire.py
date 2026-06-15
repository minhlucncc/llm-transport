"""Anthropic wire backend tests over ``httpx.MockTransport``.

Covers request shaping (x-api-key, cache_control system passthrough), response
parsing, stop_reason normalization, the SSE messages protocol, and — the key
risk — prompt-cache usage round-tripping through both the non-stream and stream
paths.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_transport.anthropic_wire import AnthropicWireTransport
from llm_transport.errors import TransportRateLimit
from llm_transport.events import StreamDone, TextDelta
from llm_transport.types import LlmRequest


def _sse(events: list[dict]) -> bytes:
    out = []
    for e in events:
        out.append(f"event: {e['type']}\n")
        out.append(f"data: {json.dumps(e)}\n\n")
    return "".join(out).encode("utf-8")


def _transport(handler, **kw) -> AnthropicWireTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return AnthropicWireTransport(base_url="https://api.anthropic.com", api_key="k", http_client=client, **kw)


def _req(**kw) -> LlmRequest:
    kw.setdefault("model", "m")
    kw.setdefault("messages", [{"role": "user", "content": "q"}])
    return LlmRequest(**kw)


# ── complete() ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_request_shape_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["x_api_key"] = request.headers.get("x-api-key")
        seen["version"] = request.headers.get("anthropic-version")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "t9", "name": "read", "input": {"a": 1}},
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 11, "output_tokens": 4},
            },
        )

    t = _transport(handler)
    r = await t.complete(_req(system="s", tools=[{"name": "read", "description": "", "input_schema": {}}], max_tokens=64, temperature=0))
    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["x_api_key"] == "k" and seen["version"] == "2023-06-01"
    assert seen["body"]["max_tokens"] == 64 and seen["body"]["system"] == "s"
    assert seen["body"]["messages"] == [{"role": "user", "content": "q"}]
    assert r.stop_reason == "tool_use"
    assert r.content[0].text == "answer"
    assert r.content[1].type == "tool_use" and r.content[1].name == "read" and r.content[1].input == {"a": 1}
    assert r.usage.input_tokens == 11 and r.usage.output_tokens == 4


@pytest.mark.asyncio
async def test_complete_preserves_cache_usage_fields():
    def handler(_):
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 50}})

    r = await _transport(handler).complete(_req())
    assert r.usage.cache_read_input_tokens == 100
    assert r.usage.cache_creation_input_tokens == 50


@pytest.mark.asyncio
async def test_complete_passes_system_cache_control_blocks_through():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {}})

    system = [
        {"type": "text", "text": "STABLE", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "VOLATILE"},
    ]
    await _transport(handler).complete(_req(system=system))
    assert seen["body"]["system"] == system  # untouched, cache_control intact


@pytest.mark.asyncio
async def test_complete_defaults_max_tokens_when_omitted():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"content": [], "stop_reason": "end_turn", "usage": {}})

    await _transport(handler).complete(_req())  # no max_tokens
    assert seen["body"]["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_complete_stop_reason_normalization():
    def handler(_):
        return httpx.Response(200, json={"content": [{"type": "text", "text": "x"}], "stop_reason": "stop_sequence", "usage": {}})

    r = await _transport(handler).complete(_req())
    assert r.stop_reason == "end_turn"  # stop_sequence → end_turn


# ── stream() ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_text_and_cache_usage_roundtrip():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 20, "output_tokens": 1, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 10}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]
    acc = ""
    async with _transport(_stream_handler(events)).stream(_req(max_tokens=10)) as st:
        async for d in st.text_stream:
            acc += d
        final = await st.get_final_message()
    assert acc == "Hello"
    assert final.content[0].text == "Hello" and final.stop_reason == "end_turn"
    assert final.usage.input_tokens == 20 and final.usage.output_tokens == 7
    # cache fields survived the stream (risk #1 guard)
    assert final.usage.cache_read_input_tokens == 200
    assert final.usage.cache_creation_input_tokens == 10


@pytest.mark.asyncio
async def test_stream_reassembles_tool_use_from_input_json_delta():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "tu1", "name": "search", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"q":'}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ' "hi"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
    ]
    async with _transport(_stream_handler(events)).stream(_req(max_tokens=10)) as st:
        async for _ in st.text_stream:
            pass
        final = await st.get_final_message()
    assert final.stop_reason == "tool_use"
    assert final.content[0].type == "tool_use"
    assert final.content[0].id == "tu1" and final.content[0].name == "search"
    assert final.content[0].input == {"q": "hi"}


@pytest.mark.asyncio
async def test_stream_mixed_text_then_tool_use_block_order():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me check"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu2", "name": "lookup", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 12}},
    ]
    async with _transport(_stream_handler(events)).stream(_req(max_tokens=20)) as st:
        async for _ in st.text_stream:
            pass
        final = await st.get_final_message()
    assert [b.type for b in final.content] == ["text", "tool_use"]
    assert final.content[0].text == "Let me check"
    assert final.content[1].name == "lookup" and final.content[1].input == {}


@pytest.mark.asyncio
async def test_stream_events_surface():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 2, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
    ]
    seen = []
    async with _transport(_stream_handler(events)).stream(_req(max_tokens=10)) as st:
        async for ev in st.events():
            seen.append(ev)
    assert isinstance(seen[0], TextDelta) and seen[0].text == "Hi"
    assert isinstance(seen[-1], StreamDone) and seen[-1].response.content[0].text == "Hi"


# ── errors ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_429_raises_rate_limit():
    def handler(_):
        return httpx.Response(429, text="slow down")

    with pytest.raises(TransportRateLimit):
        await _transport(handler, max_retries=0).complete(_req())


def _stream_handler(events):
    def handler(_request):
        return httpx.Response(200, content=_sse(events), headers={"content-type": "text/event-stream"})

    return handler
