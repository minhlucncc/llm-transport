"""Stream events emitted by a :class:`StreamSession`.

The legacy surface (``text_stream`` yielding ``str`` deltas + ``get_final_message``)
is preserved for the compat shim. ``events()`` is the richer surface the new
ReactLoop consumes: a single typed stream of deltas terminated by a ``StreamDone``
carrying the fully assembled :class:`~llm_transport.types.LlmResponse`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import LlmResponse


@dataclass(frozen=True)
class TextDelta:
    """A chunk of answer text (already past the ``<think>`` filter, if any)."""

    text: str


@dataclass(frozen=True)
class StreamDone:
    """Terminal event: the assembled response (text + reassembled tool calls + usage)."""

    response: LlmResponse


StreamEvent = TextDelta | StreamDone
