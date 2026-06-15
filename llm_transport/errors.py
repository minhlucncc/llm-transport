"""Owned exception taxonomy — replaces the SDK exception classes.

The legacy retry-classifier (``worker-retrieve`` ``_is_retryable_error``) keyed
off ``anthropic`` exception types + ``httpx`` errors. Once the SDK is gone, the
wire backends raise these instead, and :func:`is_retryable` is the single
classifier the retry wrapper and the worker share.
"""

from __future__ import annotations


class TransportError(Exception):
    """Base for every error a transport backend raises."""


class TransportTimeout(TransportError):
    """The request exceeded the configured timeout (connect or read)."""


class TransportConnectionError(TransportError):
    """The connection could not be established or was dropped."""


class TransportStatusError(TransportError):
    """The provider returned a non-2xx HTTP status."""

    def __init__(self, status_code: int, message: str = "", *, body: str | None = None) -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body


class TransportRateLimit(TransportStatusError):
    """HTTP 429 — provider rate limit."""


class TransportServerError(TransportStatusError):
    """HTTP 5xx — provider-side error."""


def is_retryable(exc: Exception) -> bool:
    """Whether a transport error is worth retrying.

    Retryable: timeouts, connection drops, 429s, and 5xx. A 4xx other than 429
    is a caller error (bad request, auth) and must not be retried.
    """
    if isinstance(exc, (TransportTimeout, TransportConnectionError)):
        return True
    if isinstance(exc, TransportStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False
