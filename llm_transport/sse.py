"""Minimal SSE parser over an httpx streaming response.

Both wire backends emit ``data: {json}`` lines (the Anthropic wire also emits
``event:`` lines, but the event ``type`` is duplicated inside the JSON payload,
so backends can rely on the data alone). ``[DONE]`` terminates an OpenAI stream;
malformed/keepalive lines are skipped.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

_DATA_PREFIX = "data:"


async def iter_sse_json(
    response: httpx.Response, *, include_non_objects: bool = False
) -> AsyncIterator[dict[str, Any] | Any]:
    """Yield parsed SSE JSON, optionally preserving non-object values for validation.

    Legacy consumers retain the historical object-only behavior. A wire adapter
    that validates its provider contract can opt in so a JSON string is rejected
    as a protocol failure instead of being silently skipped.
    """
    async for raw in response.aiter_lines():
        line = raw.strip()
        if not line or not line.startswith(_DATA_PREFIX):
            continue
        payload = line[len(_DATA_PREFIX) :].strip()
        if not payload or payload == "[DONE]":
            if payload == "[DONE]":
                return
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) or include_non_objects:
            yield obj
