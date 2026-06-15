"""Bounded retry/backoff — replaces the SDK-provided ``max_retries``.

Retries only :func:`llm_transport.errors.is_retryable` failures (timeouts,
connection drops, 429, 5xx) with exponential backoff + jitter. For streams this
guards the connection/first-byte phase ONLY — a stream that has already emitted
a delta is never replayed (see the wire backends' ``__aenter__``).
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .errors import is_retryable

T = TypeVar("T")

_BASE_DELAY = 0.5
_MAX_DELAY = 8.0


async def with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float = _BASE_DELAY,
) -> T:
    """Run ``fn``, retrying retryable transport errors up to ``max_retries`` times."""
    attempt = 0
    while True:
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 — re-raised unless retryable
            if attempt >= max_retries or not is_retryable(exc):
                raise
            delay = min(base_delay * (2**attempt), _MAX_DELAY)
            delay += random.uniform(0, base_delay)  # jitter to de-sync retries
            await asyncio.sleep(delay)
            attempt += 1
