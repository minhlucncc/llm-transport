"""``build_transport`` — pick a wire backend from a resolved provider profile.

``kind`` uses the same string values ``rag_core.llm.provider`` resolves
(``"anthropic-compatible"`` / ``"openai-compatible"``); the constants are
re-declared here so this package never imports ``rag_core`` (keeping it pure and
free of a circular dependency).
"""

from __future__ import annotations

from .base import Transport

ANTHROPIC_COMPATIBLE = "anthropic-compatible"
OPENAI_COMPATIBLE = "openai-compatible"


def build_transport(
    *,
    kind: str,
    base_url: str | None,
    api_key: str,
    timeout: float = 120.0,
    max_retries: int = 2,
) -> Transport:
    """Construct the wire transport for a provider ``kind``.

    Anthropic-compatible is the default/fallback (matches ``rag_core`` provider
    resolution, where an unknown/blank kind falls back to the Anthropic wire).
    """
    if kind == OPENAI_COMPATIBLE:
        from .openai_wire import OpenAIWireTransport

        return OpenAIWireTransport(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    from .anthropic_wire import AnthropicWireTransport

    return AnthropicWireTransport(
        base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries
    )
