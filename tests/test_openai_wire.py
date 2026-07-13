"""OpenAI wire backend tests — same scenarios as the rag-core parity oracle,
exercised at the httpx layer via ``httpx.MockTransport``.
"""

from __future__ import annotations

import json

import httpx
import pytest

from llm_transport.errors import TransportProtocolError, TransportRateLimit, TransportServerError
from llm_transport.openai_wire import (
    OpenAIWireTransport,
    _to_openai_messages,
    _to_openai_tools,
)
from llm_transport.types import LlmRequest


def _sse(chunks: list[dict]) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    body += "data: [DONE]\n\n"
    return body.encode("utf-8")


def _transport(handler, **kw) -> OpenAIWireTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAIWireTransport(base_url="https://x/v1", api_key="k", http_client=client, **kw)


def _req(**kw) -> LlmRequest:
    kw.setdefault("model", "m")
    kw.setdefault("messages", [{"role": "user", "content": "q"}])
    return LlmRequest(**kw)


# ── translation parity ──────────────────────────────────────────────────────────


def test_to_openai_messages_roundtrips_tool_use_and_result():
    out = _to_openai_messages(
        "SYS",
        [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "res"}],
            },
        ],
    )
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2]["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": "res"}


def test_to_openai_tools_maps_input_schema():
    tools = _to_openai_tools(
        [{"name": "read", "description": "d", "input_schema": {"type": "object"}}]
    )
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"] == {"type": "object"}
    assert _to_openai_tools(None) is None and _to_openai_tools([]) is None


# ── complete() ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_shapes_response_and_sends_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "answer",
                            "tool_calls": [
                                {"id": "t9", "function": {"name": "read", "arguments": '{"a": 1}'}}
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    t = _transport(handler)
    r = await t.complete(
        _req(
            system="s",
            tools=[{"name": "read", "description": "", "input_schema": {}}],
            temperature=0,
            max_tokens=50,
        )
    )
    assert seen["url"] == "https://x/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["tool_choice"] == "auto"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "s"}
    assert r.stop_reason == "tool_use"
    assert r.content[0].type == "text" and r.content[0].text == "answer"
    assert (
        r.content[1].type == "tool_use"
        and r.content[1].name == "read"
        and r.content[1].input == {"a": 1}
    )
    assert r.usage.input_tokens == 7 and r.usage.output_tokens == 3


@pytest.mark.asyncio
async def test_complete_maps_length_to_max_tokens():
    def handler(_):
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "length", "message": {"content": "cut"}}]}
        )

    r = await _transport(handler).complete(_req(max_tokens=5))
    assert r.stop_reason == "max_tokens"
    assert r.usage.input_tokens == 0 and r.usage.output_tokens == 0


@pytest.mark.asyncio
async def test_complete_empty_choices_returns_empty():
    def handler(_):
        return httpx.Response(200, json={"choices": []})

    r = await _transport(handler).complete(_req())
    assert r.content == [] and r.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_complete_tool_args_already_dict():
    def handler(_):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {"id": "t1", "function": {"name": "read", "arguments": {"a": 1}}}
                            ]
                        },
                    }
                ]
            },
        )

    r = await _transport(handler).complete(_req())
    assert r.content[0].type == "tool_use" and r.content[0].input == {"a": 1}


@pytest.mark.asyncio
async def test_complete_strips_inlined_think():
    def handler(_):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "<think>reason\n</think>\n\n\nOK"},
                    }
                ]
            },
        )

    r = await _transport(handler).complete(_req(max_tokens=50))
    assert r.content[0].text == "OK"


@pytest.mark.asyncio
async def test_complete_unterminated_think_drops_to_empty():
    def handler(_):
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "length", "message": {"content": "<think>let me"}}]
            },
        )

    r = await _transport(handler).complete(_req())
    assert r.content == []


@pytest.mark.asyncio
async def test_complete_rejects_a_string_body_without_exposing_it():
    def handler(_):
        return httpx.Response(200, json="gateway diagnostic: bearer secret-value")

    with pytest.raises(TransportProtocolError, match="invalid_response_type") as exc_info:
        await _transport(handler).complete(_req())

    assert "secret-value" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_rejects_string_chunk_without_exposing_it():
    def handler(_):
        return httpx.Response(
            200,
            content=b'data: "gateway diagnostic: bearer secret-value"\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    async with _transport(handler).stream(_req()) as stream:
        with pytest.raises(TransportProtocolError, match="invalid_chunk_type") as exc_info:
            async for _ in stream.text_stream:
                pass

    assert "secret-value" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_complete_flattens_system_blocks():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        )

    system = [
        {"type": "text", "text": "STABLE", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "VOLATILE"},
    ]
    await _transport(handler).complete(_req(system=system))
    assert seen["body"]["messages"][0] == {"role": "system", "content": "STABLE\nVOLATILE"}


# ── stream() ─────────────────────────────────────────────────────────────────


def _stream_handler(chunks):
    def handler(_request):
        return httpx.Response(
            200, content=_sse(chunks), headers={"content-type": "text/event-stream"}
        )

    return handler


@pytest.mark.asyncio
async def test_stream_accumulates_text_and_usage():
    chunks = [
        {"choices": [{"finish_reason": None, "delta": {"content": "Hel"}}]},
        {"choices": [{"finish_reason": None, "delta": {"content": "lo"}}]},
        {
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
            "choices": [{"finish_reason": "stop", "delta": {"content": None}}],
        },
    ]
    acc = ""
    async with _transport(_stream_handler(chunks)).stream(_req(max_tokens=10)) as st:
        async for d in st.text_stream:
            acc += d
        final = await st.get_final_message()
    assert acc == "Hello"
    assert final.content[0].text == "Hello" and final.stop_reason == "end_turn"
    assert final.usage.input_tokens == 5 and final.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_stream_reassembles_tool_calls():
    chunks = [
        {
            "choices": [
                {
                    "finish_reason": None,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc1",
                                "function": {"name": "search", "arguments": '{"q":'},
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "finish_reason": None,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"hi"}'}}]},
                }
            ]
        },
        {"choices": [{"finish_reason": "tool_calls", "delta": {"content": None}}]},
    ]
    async with _transport(_stream_handler(chunks)).stream(_req(max_tokens=10)) as st:
        async for _ in st.text_stream:
            pass
        final = await st.get_final_message()
    assert final.stop_reason == "tool_use"
    assert final.content[0].name == "search" and final.content[0].input == {"q": "hi"}


@pytest.mark.asyncio
async def test_stream_think_split_across_chunks():
    chunks = [
        {"choices": [{"finish_reason": None, "delta": {"content": "<th"}}]},
        {"choices": [{"finish_reason": None, "delta": {"content": "ink>reasoning "}}]},
        {"choices": [{"finish_reason": None, "delta": {"content": "here</thi"}}]},
        {"choices": [{"finish_reason": None, "delta": {"content": "nk>\n\nXin chào"}}]},
        {"choices": [{"finish_reason": None, "delta": {"content": "!"}}]},
        {"choices": [{"finish_reason": "stop", "delta": {"content": None}}]},
    ]
    acc = ""
    async with _transport(_stream_handler(chunks)).stream(_req(max_tokens=50)) as st:
        async for d in st.text_stream:
            acc += d
        final = await st.get_final_message()
    assert acc == "Xin chào!" and final.content[0].text == "Xin chào!"


@pytest.mark.asyncio
async def test_stream_events_surface():
    from llm_transport.events import StreamDone, TextDelta

    chunks = [
        {"choices": [{"finish_reason": None, "delta": {"content": "Hi"}}]},
        {"choices": [{"finish_reason": "stop", "delta": {"content": None}}]},
    ]
    seen = []
    async with _transport(_stream_handler(chunks)).stream(_req(max_tokens=10)) as st:
        async for ev in st.events():
            seen.append(ev)
    assert isinstance(seen[0], TextDelta) and seen[0].text == "Hi"
    assert isinstance(seen[-1], StreamDone) and seen[-1].response.content[0].text == "Hi"


# ── error mapping + retry ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_complete_429_raises_rate_limit():
    def handler(_):
        return httpx.Response(429, text="slow down")

    with pytest.raises(TransportRateLimit) as ei:
        await _transport(handler, max_retries=0).complete(_req())
    assert ei.value.status_code == 429


@pytest.mark.asyncio
async def test_complete_retries_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(_):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        )

    r = await _transport(handler, max_retries=2).complete(_req())
    assert calls["n"] == 2 and r.content[0].text == "ok"


@pytest.mark.asyncio
async def test_complete_500_exhausts_retries_and_raises():
    def handler(_):
        return httpx.Response(500, text="boom")

    with pytest.raises(TransportServerError):
        await _transport(handler, max_retries=1).complete(_req())


# ── full production object graph: TransportClient over the wire backend ─────────


@pytest.mark.asyncio
async def test_transport_client_over_openai_wire_messages_surface():
    from llm_transport import TransportClient

    def handler(request):
        if json.loads(request.content).get("stream"):
            return httpx.Response(
                200,
                content=_sse(
                    [
                        {"choices": [{"finish_reason": None, "delta": {"content": "hi"}}]},
                        {"choices": [{"finish_reason": "stop", "delta": {"content": None}}]},
                    ]
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = TransportClient(_transport(handler))

    # .messages.create — the legacy seam, with an interpreter-only kwarg dropped
    r = await client.messages.create(
        model="m", messages=[{"role": "user", "content": "q"}], max_tokens=5, count_usage=True
    )
    assert r.content[0].text == "done" and r.usage.input_tokens == 1

    # .messages.stream — the legacy ctx-manager seam
    acc = ""
    async with client.messages.stream(
        model="m", messages=[{"role": "user", "content": "q"}], max_tokens=5
    ) as st:
        async for d in st.text_stream:
            acc += d
        final = await st.get_final_message()
    assert acc == "hi" and final.content[0].text == "hi"
