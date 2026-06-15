"""Shared httpx plumbing for the wire backends: client construction + error mapping."""

from __future__ import annotations

import httpx

from .errors import (
    TransportConnectionError,
    TransportError,
    TransportRateLimit,
    TransportServerError,
    TransportStatusError,
    TransportTimeout,
)


def make_client(
    timeout: float, http_client: httpx.AsyncClient | None
) -> tuple[httpx.AsyncClient, bool]:
    """Return ``(client, owned)``. An injected client (tests) is not owned/closed."""
    if http_client is not None:
        return http_client, False
    return httpx.AsyncClient(timeout=timeout), True


def status_error(status_code: int, body: str | None) -> TransportStatusError:
    """Map an HTTP error status to the owned taxonomy."""
    if status_code == 429:
        return TransportRateLimit(status_code, body=body)
    if status_code >= 500:
        return TransportServerError(status_code, body=body)
    return TransportStatusError(status_code, body=body)


def map_request_error(exc: httpx.HTTPError) -> TransportError:
    """Map an httpx transport-level error to the owned taxonomy."""
    if isinstance(exc, httpx.TimeoutException):
        return TransportTimeout(str(exc))
    # ConnectError, ReadError, NetworkError, ProtocolError, …
    return TransportConnectionError(str(exc))
