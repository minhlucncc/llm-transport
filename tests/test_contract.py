"""Contract smoke tests: the shim presents the legacy ``.messages`` surface, and
our normalized types are duck-compatible with what call sites read off an
``anthropic.types.Message``.

No wire backend needed — a hand-rolled fake ``Transport`` exercises the contract.
"""

from __future__ import annotations

import pytest

from llm_transport import (
    LlmRequest,
    LlmResponse,
    TextBlock,
    ToolUseBlock,
    TransportClient,
    Usage,
    is_retryable,
)
from llm_transport.errors import (
    TransportConnectionError,
    TransportRateLimit,
    TransportServerError,
    TransportStatusError,
    TransportTimeout,
)
from llm_transport.events import StreamDone, TextDelta

# ── a fake transport that records the request and replays a canned response ────


class _FakeStream:
    def __init__(self, deltas, response):
        self._deltas = deltas
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    @property
    async def text_stream(self):
        for d in self._deltas:
            yield d

    async def get_final_message(self):
        return self._response

    async def events(self):
        for d in self._deltas:
            yield TextDelta(d)
        yield StreamDone(self._response)


class _FakeTransport:
    def __init__(self, response, deltas=()):
        self._response = response
        self._deltas = list(deltas)
        self.last_request: LlmRequest | None = None

    async def complete(self, req: LlmRequest) -> LlmResponse:
        self.last_request = req
        return self._response

    def stream(self, req: LlmRequest):
        self.last_request = req
        return _FakeStream(self._deltas, self._response)


# ── create() ───────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_maps_kwargs_to_request_and_returns_response():
    resp = LlmResponse(
        content=[
            TextBlock(text="hi"),
            ToolUseBlock(id="t1", name="search", input={"q": "x"}),
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=7, output_tokens=3, cache_read_input_tokens=5),
    )
    t = _FakeTransport(resp)
    client = TransportClient(t)

    r = await client.messages.create(
        model="m",
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        tools=[{"name": "search", "description": "", "input_schema": {}}],
        temperature=0,
        max_tokens=50,
        count_usage=True,  # interpreter-only kwarg — must be dropped, not crash
    )

    # request was built from kwargs, unknown kwarg dropped
    assert t.last_request.model == "m"
    assert t.last_request.system == "sys"
    assert t.last_request.max_tokens == 50
    # response is duck-compatible with anthropic Message
    assert r.stop_reason == "tool_use"
    assert r.content[0].type == "text" and r.content[0].text == "hi"
    assert r.content[1].type == "tool_use" and r.content[1].name == "search"
    assert r.content[1].input == {"q": "x"}
    assert r.usage.input_tokens == 7 and r.usage.output_tokens == 3
    assert r.usage.cache_read_input_tokens == 5


# ── stream() ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stream_legacy_surface():
    resp = LlmResponse(content=[TextBlock(text="Hello")], usage=Usage(input_tokens=5, output_tokens=2))
    client = TransportClient(_FakeTransport(resp, deltas=["Hel", "lo"]))

    acc = ""
    async with client.messages.stream(model="m", messages=[{"role": "user", "content": "q"}], max_tokens=10) as st:
        async for d in st.text_stream:
            acc += d
        final = await st.get_final_message()
    assert acc == "Hello"
    assert final.content[0].text == "Hello"
    assert final.usage.input_tokens == 5


@pytest.mark.asyncio
async def test_stream_events_surface_terminates_with_done():
    resp = LlmResponse(content=[TextBlock(text="Hello")])
    client = TransportClient(_FakeTransport(resp, deltas=["Hel", "lo"]))

    seen = []
    async with client.messages.stream(model="m", messages=[], max_tokens=10) as st:
        async for ev in st.events():
            seen.append(ev)
    assert [e.text for e in seen if isinstance(e, TextDelta)] == ["Hel", "lo"]
    assert isinstance(seen[-1], StreamDone)
    assert seen[-1].response.content[0].text == "Hello"


# ── error taxonomy ──────────────────────────────────────────────────────────────


def test_is_retryable_classification():
    assert is_retryable(TransportTimeout())
    assert is_retryable(TransportConnectionError())
    assert is_retryable(TransportRateLimit(429))
    assert is_retryable(TransportServerError(503))
    assert is_retryable(TransportStatusError(500))
    assert not is_retryable(TransportStatusError(400))
    assert not is_retryable(TransportStatusError(401))
    assert not is_retryable(ValueError("nope"))
