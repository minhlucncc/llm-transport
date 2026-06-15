"""Normalized, provider-neutral request/response types — OUR own, not ``anthropic.types``.

The canonical request keeps the **Anthropic block shape** for ``messages``,
``system`` and ``tools`` (``{"type": "text"|"tool_use"|"tool_result", ...}``,
``{"name", "description", "input_schema"}``), because every call site in the
engine already emits that shape. So the Anthropic wire backend is a near
pass-through and only the OpenAI wire translates.

The response block/usage field names mirror exactly what the interpreter reads
via ``getattr`` today (``.content`` → ``.type``/``.text`` or ``.id``/``.name``/
``.input``; ``.stop_reason``; ``.usage.input_tokens``/``.output_tokens``/
``.cache_read_input_tokens``/``.cache_creation_input_tokens``). That duck
compatibility is what lets the compat shim drop in with zero call-site edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── response content blocks (attribute-only; never isinstance-checked) ─────────


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


ContentBlock = TextBlock | ToolUseBlock


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    # Prompt-cache accounting. ``None`` when the provider/wire does not report it
    # (the OpenAI wire never does) — kept optional so cache metrics never get
    # silently zeroed into the rollup.
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass
class LlmResponse:
    content: list[ContentBlock] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens
    usage: Usage = field(default_factory=Usage)


# ── request ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LlmRequest:
    """A single model call. ``messages``/``system``/``tools`` are Anthropic-shaped.

    ``system`` may be a plain string OR a block array carrying ``cache_control``
    breakpoints (the prompt-cache prefix the interpreter builds). Backends honor
    or translate it as their wire requires.
    """

    model: str
    messages: list[dict[str, Any]]
    system: str | list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    temperature: float | None = None
    max_tokens: int | None = None
