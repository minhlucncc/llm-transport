"""Gemini Anthropic-compat wire backend (httpx).

A narrow adapter over the Anthropic Messages wire: the same request/response
shape (``messages``/``system``/``tools`` pass-through, ``max_tokens``
defended) against Google's Anthropic-compatible endpoint
``{root}/v1beta/anthropic/messages``, authenticating with ``x-goog-api-key``
via the parent wire's per-wire auth-header name (D3). ``tool_choice`` is
pinned to ``{"type": "auto"}`` when tools are present so the cite/filter
stages dispatch deterministically (D1).

Base URL normalization (D4): a stored base ending in ``/v1beta/anthropic``
(the preset default) is normalized to the host root before use so the probe
and runtime never form a doubled ``/v1beta/anthropic/v1beta/anthropic`` path;
a root-entered base gets the compat path appended exactly once.

Error envelopes map Google's ``{"error": {"code", "message", "status"}}``
shape into the owned taxonomy (429/5xx classification) with the server's
``message`` as the exception text — the credential never leaves the request
header, so it cannot leak into errors.
"""

from __future__ import annotations

import json

import httpx

from .anthropic_wire import AnthropicWireTransport
from .errors import TransportRateLimit, TransportServerError, TransportStatusError
from .types import LlmRequest

_ANTHROPIC_COMPAT_PATH = "/v1beta/anthropic"
_PRESET_BASE = "https://generativelanguage.googleapis.com/v1beta/anthropic"


def _host_root(base_url: str | None) -> str:
    """Normalize a stored base URL to the host root (never a doubled compat path)."""
    base = (base_url or _PRESET_BASE).rstrip("/")
    if base.endswith(_ANTHROPIC_COMPAT_PATH):
        base = base[: -len(_ANTHROPIC_COMPAT_PATH)].rstrip("/")
    return base


def _google_status_error(status_code: int, body: str | None) -> TransportStatusError:
    """Map a Google ``{"error": {"code", "message", "status"}}`` envelope to the taxonomy."""
    envelope_code = status_code
    message = ""
    if body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            code = error.get("code")
            if isinstance(code, int):
                envelope_code = code
            msg = error.get("message")
            if isinstance(msg, str):
                message = msg
    if envelope_code == 429:
        return TransportRateLimit(envelope_code, message, body=body)
    if envelope_code >= 500:
        return TransportServerError(envelope_code, message, body=body)
    return TransportStatusError(envelope_code, message, body=body)


class GeminiWireTransport(AnthropicWireTransport):
    """Anthropic-Messages-shaped requests to ``{root}/v1beta/anthropic/messages``.

    Inherits the wire mechanics (body build, response parsing, SSE stream)
    from the Anthropic wire; only the endpoint path, the auth header name
    (``x-goog-api-key``), the ``tool_choice`` pin, and the Google
    error-envelope mapping differ.
    """

    def __init__(
        self,
        *,
        base_url: str | None,
        api_key: str,
        timeout: float = 120.0,
        max_retries: int = 2,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        root = _host_root(base_url)
        super().__init__(
            base_url=root,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            auth_header="x-goog-api-key",
        )
        self._url = f"{root}{_ANTHROPIC_COMPAT_PATH}/messages"

    def _build_body(self, req: LlmRequest, *, stream: bool) -> dict:
        body = super()._build_body(req, stream=stream)
        if req.tools:
            # cite/filter dispatch: pin tool use so every stage with tools
            # issues tool calls over the Gemini wire (D1).
            body["tool_choice"] = {"type": "auto"}
        return body

    def _status_error(self, status_code: int, body: str) -> TransportStatusError:
        return _google_status_error(status_code, body)
