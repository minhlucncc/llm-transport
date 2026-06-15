"""The transport contract: ``Transport`` + ``StreamSession`` Protocols.

A backend implements ``complete`` (one non-streaming call) and ``stream`` (an
async context manager yielding deltas). The ``StreamSession`` deliberately
exposes BOTH surfaces:

- ``text_stream`` + ``get_final_message`` — the legacy Anthropic-``messages``
  shape, so :mod:`llm_transport.compat` can present it unchanged to existing
  call sites.
- ``events()`` — the richer typed stream the new ReactLoop consumes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from .events import StreamEvent
from .types import LlmRequest, LlmResponse


@runtime_checkable
class StreamSession(Protocol):
    """Async context manager over one streaming model call."""

    async def __aenter__(self) -> StreamSession: ...

    async def __aexit__(self, *exc: Any) -> None: ...

    @property
    def text_stream(self) -> AsyncIterator[str]:
        """Yield answer-text deltas (post ``<think>`` filter)."""
        ...

    async def get_final_message(self) -> LlmResponse:
        """The assembled response (text + reassembled tool calls + usage)."""
        ...

    def events(self) -> AsyncIterator[StreamEvent]:
        """Typed event stream: ``TextDelta`` … terminated by ``StreamDone``."""
        ...


@runtime_checkable
class Transport(Protocol):
    """A provider-neutral LLM transport. Backends speak a specific wire underneath."""

    async def complete(self, req: LlmRequest) -> LlmResponse: ...

    def stream(self, req: LlmRequest) -> StreamSession: ...
