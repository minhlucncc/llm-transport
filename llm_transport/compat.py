"""Compatibility shim: present a ``Transport`` as the legacy ``.messages`` surface.

Existing call sites do ``await client.messages.create(model=..., messages=...)``
and ``async with client.messages.stream(...) as st: ...``. ``TransportClient``
wraps a :class:`~llm_transport.base.Transport` so those sites work unchanged —
the returned :class:`~llm_transport.types.LlmResponse` is duck-compatible with
``anthropic.types.Message`` (same ``.content``/``.stop_reason``/``.usage`` shape).

This is what makes the cutover incremental: flip ``LLM_TRANSPORT_BACKEND=wire``
and every ``messages.*`` caller keeps running with no edits.
"""

from __future__ import annotations

from typing import Any

from .base import StreamSession, Transport
from .types import LlmRequest, LlmResponse

# The kwargs every call site passes to messages.create/stream. Anything else
# (e.g. an interpreter-only ``count_usage`` flag) is dropped here, mirroring the
# old adapter's ``**_: Any`` tolerance.
_REQUEST_FIELDS = frozenset(
    {"model", "messages", "system", "tools", "temperature", "max_tokens"}
)


def _request_from_kwargs(kwargs: dict[str, Any]) -> LlmRequest:
    return LlmRequest(**{k: v for k, v in kwargs.items() if k in _REQUEST_FIELDS})


class MessagesFacade:
    """The ``.messages`` attribute: ``create`` (await) + ``stream`` (async ctx)."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport

    async def create(self, **kwargs: Any) -> LlmResponse:
        return await self._transport.complete(_request_from_kwargs(kwargs))

    def stream(self, **kwargs: Any) -> StreamSession:
        return self._transport.stream(_request_from_kwargs(kwargs))


class TransportClient:
    """Legacy-shaped client: ``client.messages.create/.stream`` over a Transport."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self.messages = MessagesFacade(transport)

    @property
    def transport(self) -> Transport:
        """The underlying transport (for call sites migrating off the shim)."""
        return self._transport
