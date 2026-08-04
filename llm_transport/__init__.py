"""llm_transport — a provider-neutral LLM transport we own end-to-end.

Our own normalized request/response/stream contract over native httpx wire
backends (Anthropic + OpenAI). No vendor SDK, no DB, no ``rag_core`` import —
``rag_core.llm`` wires config + provider resolution into :func:`build_transport`
and presents the legacy ``.messages`` surface via :class:`TransportClient`.
"""

from .base import StreamSession, Transport
from .compat import MessagesFacade, TransportClient
from .errors import (
    TransportConnectionError,
    TransportError,
    TransportRateLimit,
    TransportServerError,
    TransportStatusError,
    TransportTimeout,
    is_retryable,
)
from .events import StreamDone, StreamEvent, TextDelta
from .factory import ANTHROPIC_COMPATIBLE, GEMINI_COMPATIBLE, OPENAI_COMPATIBLE, build_transport
from .types import ContentBlock, LlmRequest, LlmResponse, TextBlock, ToolUseBlock, Usage

__all__ = [
    # contract
    "Transport",
    "StreamSession",
    "LlmRequest",
    "LlmResponse",
    "TextBlock",
    "ToolUseBlock",
    "ContentBlock",
    "Usage",
    # stream events
    "TextDelta",
    "StreamDone",
    "StreamEvent",
    # shim
    "TransportClient",
    "MessagesFacade",
    # factory
    "build_transport",
    "ANTHROPIC_COMPATIBLE",
    "OPENAI_COMPATIBLE",
    "GEMINI_COMPATIBLE",
    # errors
    "TransportError",
    "TransportTimeout",
    "TransportConnectionError",
    "TransportStatusError",
    "TransportRateLimit",
    "TransportServerError",
    "is_retryable",
]
