"""Gemini native wire backend tests over ``httpx.MockTransport``.

The gemini wire speaks the **native** Google Generative Language API —
``{root}/v1beta/models/{model}:generateContent`` — authenticating with
``x-goog-api-key``, never ``x-api-key``. Google serves no Anthropic-Messages
surface (``/v1beta/anthropic/messages`` returns a bare 404 for every
credential), so the wire translates Anthropic-shaped requests into
``contents``/``systemInstruction``/``functionDeclarations`` here and normalizes
``candidates`` back into our content blocks.

No env, no real network, no API keys — every case drives an injected
``httpx.MockTransport``.
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

_ROOT = "https://generativelanguage.googleapis.com"
# The base URL saved by profiles created under the original c0036 preset.
_LEGACY_ANTHROPIC_BASE = f"{_ROOT}/v1beta/anthropic"


def _transport(handler, base_url: str = _ROOT, **kw) -> GeminiWireTransport:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiWireTransport(
        base_url=base_url, api_key=kw.pop("api_key", "gemini-key"), http_client=client, **kw
    )


def _req(**kw) -> LlmRequest:
    kw.setdefault("model", "gemini-2.5-flash")
    kw.setdefault("messages", [{"role": "user", "content": "q"}])
    return LlmRequest(**kw)


def _ok(parts=None, *, finish="STOP", usage=None):
    """A minimal successful ``generateContent`` response."""
    return httpx.Response(
        200,
        json={
            "candidates": [
                {"content": {"role": "model", "parts": parts or []}, "finishReason": finish}
            ],
            "usageMetadata": usage or {},
        },
    )


def _sse(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)


# -- auth ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_sends_x_goog_api_key_and_never_x_api_key():
    """The gemini kind authenticates with ``x-goog-api-key`` only."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return _ok()

    await _transport(handler).complete(_req())
    assert seen["headers"]["x-goog-api-key"] == "gemini-key"
    assert "x-api-key" not in seen["headers"]
    assert "authorization" not in seen["headers"]


# -- URL construction ----------------------------------------------------------


@pytest.mark.parametrize(
    "base_url",
    [
        _ROOT,
        f"{_ROOT}/",
        f"{_ROOT}/v1beta",
        _LEGACY_ANTHROPIC_BASE,  # migrated from the original c0036 preset
        f"{_LEGACY_ANTHROPIC_BASE}/",
        f"{_ROOT}/v1beta/openai",
    ],
)
@pytest.mark.asyncio
async def test_base_url_normalized_to_host_root_never_doubled(base_url):
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return _ok()

    await _transport(handler, base_url=base_url).complete(_req())
    assert seen["url"] == f"{_ROOT}/v1beta/models/gemini-2.5-flash:generateContent"
    assert "/v1beta/v1beta" not in seen["url"]
    assert "/anthropic" not in seen["url"]


@pytest.mark.asyncio
async def test_model_path_strips_models_prefix():
    """``models/gemini-2.5-pro`` and ``gemini-2.5-pro`` address the same endpoint."""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return _ok()

    await _transport(handler).complete(_req(model="models/gemini-2.5-pro"))
    assert seen["url"] == f"{_ROOT}/v1beta/models/gemini-2.5-pro:generateContent"


@pytest.mark.asyncio
async def test_custom_gateway_root_is_preserved():
    """A self-hosted proxy root keeps its path prefix."""
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return _ok()

    await _transport(handler, base_url="https://proxy.internal/gemini/v1beta").complete(_req())
    assert (
        seen["url"]
        == "https://proxy.internal/gemini/v1beta/models/gemini-2.5-flash:generateContent"
    )


# -- request translation -------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_translates_system_contents_and_generation_config():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok(
            [{"text": "answer"}],
            usage={"promptTokenCount": 5, "candidatesTokenCount": 2},
        )

    r = await _transport(handler).complete(_req(system="s", max_tokens=64, temperature=0))
    body = seen["body"]
    assert body["contents"] == [{"role": "user", "parts": [{"text": "q"}]}]
    assert body["systemInstruction"] == {"parts": [{"text": "s"}]}
    assert body["generationConfig"] == {"maxOutputTokens": 64, "temperature": 0}
    assert r.content[0].text == "answer"
    assert r.usage.input_tokens == 5 and r.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_system_block_array_flattens_to_system_instruction():
    """A cache_control block prefix collapses to one systemInstruction string."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    system = [
        {"type": "text", "text": "first"},
        {"type": "text", "text": "second", "cache_control": {"type": "ephemeral"}},
    ]
    await _transport(handler).complete(_req(system=system))
    assert seen["body"]["systemInstruction"] == {"parts": [{"text": "first\nsecond"}]}


@pytest.mark.asyncio
async def test_assistant_role_maps_to_model_and_same_role_turns_merge():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    messages = [
        {"role": "user", "content": "one"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
    ]
    await _transport(handler).complete(_req(messages=messages))
    assert seen["body"]["contents"] == [
        {"role": "user", "parts": [{"text": "one"}, {"text": "two"}]},
        {"role": "model", "parts": [{"text": "reply"}]},
    ]


@pytest.mark.asyncio
async def test_tools_become_function_declarations_with_auto_tool_config():
    """cite/filter dispatch: declared tools stay callable via AUTO mode."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    tools = [
        {
            "name": "retrieve_kb",
            "description": "search",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]
    await _transport(handler).complete(_req(tools=tools))
    assert seen["body"]["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "retrieve_kb",
                    "description": "search",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {"q": {"type": "STRING"}},
                        "required": ["q"],
                    },
                }
            ]
        }
    ]
    assert seen["body"]["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}


@pytest.mark.asyncio
async def test_function_declaration_schema_is_sanitized_to_the_gemini_subset():
    """Unsupported JSON-Schema keywords are dropped rather than 400-ing the call."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    tools = [
        {
            "name": "t",
            "description": "",
            "input_schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "mode": {"const": "fast"},
                    "limit": {"type": ["integer", "null"], "minimum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["mode"],
            },
        }
    ]
    await _transport(handler).complete(_req(tools=tools))
    params = seen["body"]["tools"][0]["functionDeclarations"][0]["parameters"]
    assert "$schema" not in params and "additionalProperties" not in params
    assert params["type"] == "OBJECT"
    assert params["properties"]["mode"] == {"enum": ["fast"]}
    assert params["properties"]["limit"] == {"type": "INTEGER", "nullable": True, "minimum": 1}
    assert params["properties"]["tags"] == {"type": "ARRAY", "items": {"type": "STRING"}}


@pytest.mark.asyncio
async def test_parameterless_tool_omits_parameters():
    """Gemini rejects an OBJECT declaration with no properties."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    await _transport(handler).complete(
        _req(tools=[{"name": "ping", "description": "", "input_schema": {"type": "object"}}])
    )
    decl = seen["body"]["tools"][0]["functionDeclarations"][0]
    assert decl == {"name": "ping"}


@pytest.mark.asyncio
async def test_tool_result_becomes_function_response_named_from_the_conversation():
    """Gemini pairs calls by name, so the echoed tool_use_id resolves to one."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_42", "name": "retrieve_kb", "input": {"q": "x"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_42", "content": "chunk text"}
            ],
        },
    ]
    await _transport(handler).complete(_req(messages=messages))
    contents = seen["body"]["contents"]
    assert contents[1] == {
        "role": "model",
        "parts": [{"functionCall": {"name": "retrieve_kb", "args": {"q": "x"}}}],
    }
    assert contents[2] == {
        "role": "user",
        "parts": [
            {"functionResponse": {"name": "retrieve_kb", "response": {"result": "chunk text"}}}
        ],
    }


@pytest.mark.asyncio
async def test_tool_result_name_falls_back_to_the_synthesized_id():
    """A round-trip of our own emitted id still resolves without the assistant turn."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "gemini-call-0-retrieve_kb",
                    "content": {"hits": 2},
                }
            ],
        }
    ]
    await _transport(handler).complete(_req(messages=messages))
    call = seen["body"]["contents"][0]["parts"][0]["functionResponse"]
    assert call["name"] == "retrieve_kb"
    assert call["response"] == {"hits": 2}


# -- response translation ------------------------------------------------------


@pytest.mark.asyncio
async def test_function_call_response_maps_to_tool_use_stop_reason():
    """Gemini reports STOP for a call; the wire must still say ``tool_use``."""

    def handler(_):
        return _ok(
            [
                {"text": "checking"},
                {"functionCall": {"name": "search", "args": {"q": "x"}}},
            ],
            finish="STOP",
        )

    r = await _transport(handler).complete(
        _req(tools=[{"name": "search", "description": "", "input_schema": {}}])
    )
    assert r.stop_reason == "tool_use"
    assert r.content[0].type == "text" and r.content[0].text == "checking"
    assert r.content[1].type == "tool_use" and r.content[1].name == "search"
    assert r.content[1].input == {"q": "x"}
    assert r.content[1].id  # an id is synthesized so tool_result can pair back


@pytest.mark.asyncio
async def test_adjacent_text_parts_merge_into_one_block():
    def handler(_):
        return _ok([{"text": "hello "}, {"text": "world"}])

    r = await _transport(handler).complete(_req())
    assert len(r.content) == 1
    assert r.content[0].text == "hello world"


@pytest.mark.asyncio
async def test_thought_parts_never_surface_as_answer_text():
    def handler(_):
        return _ok([{"text": "reasoning", "thought": True}, {"text": "answer"}])

    r = await _transport(handler).complete(_req())
    assert [b.text for b in r.content] == ["answer"]


@pytest.mark.asyncio
async def test_max_tokens_finish_reason_maps_through():
    def handler(_):
        return _ok([{"text": "trunc"}], finish="MAX_TOKENS")

    r = await _transport(handler).complete(_req())
    assert r.stop_reason == "max_tokens"


@pytest.mark.asyncio
async def test_safety_block_with_no_candidates_returns_empty_answer():
    """No candidate must not crash — an ungrounded empty answer is refused downstream."""

    def handler(_):
        return httpx.Response(
            200,
            json={
                "promptFeedback": {"blockReason": "SAFETY"},
                "usageMetadata": {"promptTokenCount": 9},
            },
        )

    r = await _transport(handler).complete(_req())
    assert r.content == []
    assert r.stop_reason == "end_turn"
    assert r.usage.input_tokens == 9


@pytest.mark.asyncio
async def test_usage_folds_thinking_tokens_and_maps_cached_prompt_tokens():
    def handler(_):
        return _ok(
            [{"text": "a"}],
            usage={
                "promptTokenCount": 100,
                "candidatesTokenCount": 20,
                "thoughtsTokenCount": 30,
                "cachedContentTokenCount": 60,
            },
        )

    r = await _transport(handler).complete(_req())
    assert r.usage.input_tokens == 100
    assert r.usage.output_tokens == 50  # 20 visible + 30 thinking, both billed as output
    assert r.usage.cache_read_input_tokens == 60
    assert r.usage.cache_creation_input_tokens is None


# -- errors --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_complete_maps_google_error_envelope_without_leaking_key():
    key = "sk-gemini-secret-8f3a"

    def handler(_):
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
    assert "API key not valid" in str(exc)
    assert key not in str(exc)
    assert key not in (exc.body or "")


@pytest.mark.asyncio
async def test_array_wrapped_error_envelope_is_mapped():
    """The live endpoint wraps the envelope in a single-element array."""

    def handler(_):
        return httpx.Response(
            400,
            json=[
                {"error": {"code": 400, "message": "Please pass a valid API key", "status": "X"}}
            ],
        )

    with pytest.raises(TransportStatusError) as ei:
        await _transport(handler, max_retries=0).complete(_req())
    assert "Please pass a valid API key" in str(ei.value)


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
            500, json={"error": {"code": 500, "message": "Internal error", "status": "INTERNAL"}}
        )

    with pytest.raises(TransportServerError):
        await _transport(handler, max_retries=0).complete(_req())


# -- boundedness ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_request_is_bounded_max_output_tokens_one():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    await _transport(handler).complete(_req(max_tokens=1))
    assert seen["body"]["generationConfig"]["maxOutputTokens"] == 1


@pytest.mark.asyncio
async def test_complete_defaults_bounded_max_output_tokens_when_omitted():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return _ok()

    await _transport(handler).complete(_req())
    assert seen["body"]["generationConfig"]["maxOutputTokens"] == 4096


# -- streaming -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_text_deltas_and_assembles_final_message():
    def handler(request):
        assert str(request.url).endswith(":streamGenerateContent?alt=sse")
        return httpx.Response(
            200,
            text=_sse(
                {"candidates": [{"content": {"parts": [{"text": "he"}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "llo"}]}}]},
                {
                    "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2},
                },
            ),
            headers={"content-type": "text/event-stream"},
        )

    deltas = []
    async with _transport(handler).stream(_req()) as session:
        async for chunk in session.text_stream:
            deltas.append(chunk)
        final = await session.get_final_message()

    assert deltas == ["he", "llo"]
    assert final.content[0].text == "hello"
    assert final.stop_reason == "end_turn"
    assert final.usage.input_tokens == 3 and final.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_stream_reassembles_function_calls_as_tool_use():
    def handler(_):
        return httpx.Response(
            200,
            text=_sse(
                {"candidates": [{"content": {"parts": [{"text": "looking"}]}}]},
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"functionCall": {"name": "search", "args": {"q": "x"}}}]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                },
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with _transport(handler).stream(_req()) as session:
        final = await session.get_final_message()

    assert final.stop_reason == "tool_use"
    assert final.content[0].text == "looking"
    assert final.content[1].name == "search" and final.content[1].input == {"q": "x"}


@pytest.mark.asyncio
async def test_stream_skips_thought_parts():
    def handler(_):
        return httpx.Response(
            200,
            text=_sse(
                {"candidates": [{"content": {"parts": [{"text": "hidden", "thought": True}]}}]},
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "shown"}]}, "finishReason": "STOP"}
                    ]
                },
            ),
            headers={"content-type": "text/event-stream"},
        )

    deltas = []
    async with _transport(handler).stream(_req()) as session:
        async for chunk in session.text_stream:
            deltas.append(chunk)
    assert deltas == ["shown"]


@pytest.mark.asyncio
async def test_stream_events_terminate_with_stream_done():
    from llm_transport.events import StreamDone, TextDelta

    def handler(_):
        return httpx.Response(
            200,
            text=_sse(
                {"candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}]},
            ),
            headers={"content-type": "text/event-stream"},
        )

    events = []
    async with _transport(handler).stream(_req()) as session:
        async for event in session.events():
            events.append(event)

    assert isinstance(events[0], TextDelta) and events[0].text == "hi"
    assert isinstance(events[-1], StreamDone)
    assert events[-1].response.content[0].text == "hi"


@pytest.mark.asyncio
async def test_stream_error_status_maps_google_envelope():
    def handler(_):
        return httpx.Response(
            429,
            json={"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}},
        )

    with pytest.raises(TransportRateLimit):
        async with _transport(handler, max_retries=0).stream(_req()):
            pass


# -- factory dispatch ----------------------------------------------------------


def test_build_transport_dispatches_gemini_kind():
    t = build_transport(kind="gemini-compatible", base_url=_ROOT, api_key="gemini-key")
    assert isinstance(t, GeminiWireTransport)
