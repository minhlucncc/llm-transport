"""``<think>…</think>`` reasoning stripping for OpenAI-compatible providers.

Reasoning models behind OpenAI-compatible gateways (MiniMax, DeepSeek-R1 via
some proxies, Qwen) emit chain-of-thought inside ``content`` instead of a
separate field. The Anthropic wire never surfaces thinking in text blocks, so
the OpenAI wire strips it to stay surface-faithful. Ported verbatim from the
proven ``rag_core/llm/client.py`` logic (the golden-tested behavior).
"""

from __future__ import annotations

import re

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def strip_think(text: str) -> str:
    """Drop inlined ``<think>…</think>`` reasoning from a complete response.

    An unterminated block (max_tokens cut mid-thought) is all reasoning —
    dropped too.
    """
    if _THINK_OPEN not in text:
        return text
    return _THINK_RE.sub("", text).lstrip("\n")


class ThinkTagFilter:
    """Streaming counterpart of :func:`strip_think` for ``text_stream`` deltas.

    Suppresses a response-leading ``<think>…</think>`` block without leaking
    partial tag fragments across chunk boundaries; everything after the close
    tag (or any response that doesn't open with a think tag) passes through.
    """

    def __init__(self) -> None:
        self._state = "start"  # start → think → pass
        self._buf = ""

    def feed(self, delta: str) -> str:
        if self._state == "pass":
            return delta
        self._buf += delta
        if self._state == "start":
            probe = self._buf.lstrip()
            if not probe:
                return ""
            if probe.startswith(_THINK_OPEN):
                self._state = "think"
            elif _THINK_OPEN.startswith(probe):
                return ""  # could still become an opening tag; keep buffering
            else:
                self._state = "pass"
                out, self._buf = self._buf, ""
                return out
        end = self._buf.find(_THINK_CLOSE)
        if end == -1:
            return ""
        out = self._buf[end + len(_THINK_CLOSE):].lstrip("\n")
        self._state = "pass"
        self._buf = ""
        return out

    def flush(self) -> str:
        """End of stream: an unresolved start-buffer is real text; an
        unterminated think block is reasoning and stays dropped."""
        if self._state == "start":
            out, self._buf = self._buf, ""
            self._state = "pass"
            return out
        return ""
