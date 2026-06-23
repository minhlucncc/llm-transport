# llm-transport

A provider-neutral LLM transport for Python: **our own** normalized request / response / stream
contract over native `httpx` wire backends — **no vendor SDK**. Two wires ship today:

- **Anthropic Messages** (`anthropic-compatible`) — near pass-through to `{base_url}/v1/messages`,
  preserving prompt-caching (`cache_control`) and cache-token usage end-to-end.
- **OpenAI chat-completions** (`openai-compatible`) — translates the Anthropic-shaped request to/from
  `{base_url}/chat/completions` (messages, tools, `tool_choice`, `finish_reason`) and strips inline
  `<think>` reasoning. MiniMax / NCC gateways are configured as this kind.

The request stays Anthropic-shaped (canonical block shape); only the OpenAI wire translates. The
package is duck-compatible with `anthropic.types.Message` but **never imports** `anthropic`,
`openai`, or any agent framework — only `httpx`.

## Why

This exists to give the retrieval runtime one stable, owned contract across providers without taking a
hard dependency on any vendor SDK. `errors.py` / `_retry.py` deliberately re-implement the error
taxonomy and backoff that an SDK would otherwise provide, so a provider swap is a `kind` + `base_url`
change, not a code change.

## Install

Standalone:

```bash
pip install httpx        # the only runtime dependency
# then add this package (path/git install) — name: llm-transport
```

Inside the `funix-mezon-bot` monorepo it is a `uv` workspace member (mounted as a git submodule at
`packages/llm-transport`), referenced by name from `rag-core` and `agent-core`:

```toml
# rag-core / agent-core pyproject.toml
dependencies = ["llm-transport", ...]   # resolved via [tool.uv.sources] { workspace = true }
```

Requires Python >= 3.11.

## Quick start

```python
from llm_transport import build_transport, LlmRequest

transport = build_transport(
    kind="openai-compatible",          # or "anthropic-compatible" (the fallback)
    base_url="https://api.example-gateway.com",
    api_key="...",
    timeout=60.0,
    max_retries=2,
)

req = LlmRequest(
    model="MiniMax-M2.7",              # model id is ALWAYS per-call, never hardcoded here
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=1024,
)

resp = await transport.complete(req)   # -> LlmResponse (TextBlock / ToolUseBlock, Usage)

async with transport.stream(req) as session:
    async for event in session.events():   # typed events
        ...
    final = await session.get_final_message()
```

A `compat` facade (`TransportClient` / `MessagesFacade`) provides the legacy
`client.messages.create(...)` / `.stream(...)` surface over any `Transport`, for drop-in cutover from
the vendor SDK.

## Layout (`llm_transport/`, flat — no `src/`)

| Path | What |
|---|---|
| `base.py` | The contract: `Transport` + `StreamSession` `Protocol`s (`complete(req)` / `stream(req)`). |
| `types.py` | Owned normalized dataclasses: frozen `LlmRequest`, `LlmResponse`, `TextBlock`, `ToolUseBlock`, `Usage` (incl. cache-token fields). |
| `factory.py` | `build_transport(*, kind, base_url, api_key, timeout, max_retries)`. |
| `anthropic_wire.py` | `AnthropicWireTransport` → `/v1/messages`, `x-api-key` + `anthropic-version`. |
| `openai_wire.py` | `OpenAIWireTransport` → `/chat/completions`, `Bearer`; Anthropic↔OpenAI translation + `<think>` stripping. |
| `compat.py` | Legacy `messages.create/stream(...)` facade over any `Transport`. |
| `errors.py` · `_retry.py` · `_http.py` · `sse.py` · `thinkfilter.py` | Owned error taxonomy + `is_retryable`; backoff retry; httpx plumbing; SSE parser; `<think>` filter. |

## Gotchas

- **`base_url` handling differs per wire.** The Anthropic wire appends `/v1/messages`; the OpenAI wire
  appends only `/chat/completions` to `base_url` as-is (so gateways without a `/v1` prefix work).
- **`<think>` stripping is OpenAI-wire-only** (MiniMax / DeepSeek-R1 / Qwen emit inline CoT); the
  Anthropic wire never surfaces reasoning as text.
- **Cache usage** (`cache_read/creation_input_tokens`) is preserved end-to-end on the Anthropic wire;
  the OpenAI wire leaves those `None`.
- **Streams are never replayed once data has flowed** — retries cover connect / first-byte only.
- `compat.py` silently drops unknown kwargs (filters to a fixed set).

## Build & test

```bash
python -m pytest -q       # tests/: test_anthropic_wire, test_openai_wire, test_contract
ruff check .
pyright
```

Tests need **no env, no network, no API keys** — everything uses `httpx.MockTransport` via an injected
`http_client` or hand-rolled fakes (`asyncio_mode = "auto"`).

## License

See [LICENSE](LICENSE).
