# CLAUDE.md — llm-transport (Python / uv)

Provider-neutral LLM transport: our **own** normalized request/response/stream
contract over native `httpx` wire backends — **no vendor SDK**. Two wires:
Anthropic Messages (near pass-through) and OpenAI chat-completions (does the
translation). MiniMax / NCC gateways are configured as the OpenAI-compatible kind.

**Specs:** openspec/specs/retrieval-runtime/spec.md
**Rationale:** docs/design/0005-retrieval-runtime.md
**Workflow / invariants:** /CLAUDE.md · /DEVELOPMENT.md

## Layout (`llm_transport/`, flat — no `src/`)
| Path | What |
|---|---|
| `base.py` | The contract: `Transport` + `StreamSession` `Protocol`s. `complete(req)` / `stream(req)`. `StreamSession` exposes both the legacy Anthropic shape (`text_stream`, `get_final_message`) and typed `events()`. |
| `types.py` | Owned normalized dataclasses: frozen `LlmRequest`, `LlmResponse`, `TextBlock`, `ToolUseBlock`, `Usage` (incl. cache-token fields). Request keeps Anthropic block shape. Duck-compatible with `anthropic.types.Message` but **not** imported from `anthropic`. |
| `factory.py` | `build_transport(*, kind, base_url, api_key, timeout, max_retries)`. Kinds `anthropic-compatible` / `openai-compatible`; Anthropic is the fallback. Re-declares kind strings to avoid importing `rag_core`. |
| `anthropic_wire.py` | `AnthropicWireTransport` → `{base_url}/v1/messages`, `x-api-key` + `anthropic-version`. Near pass-through; preserves `system` cache_control + cache usage end-to-end. |
| `openai_wire.py` | `OpenAIWireTransport` → `{base_url}/chat/completions`, `Bearer`. Anthropic↔OpenAI translation (messages/tools/tool_choice/finish_reason); applies `<think>` stripping. |
| `compat.py` | `TransportClient`/`MessagesFacade`: legacy `client.messages.create/stream(...)` surface over any `Transport` for drop-in cutover. |
| `errors.py` · `_retry.py` · `_http.py` · `sse.py` · `thinkfilter.py` | Owned error taxonomy + `is_retryable`; backoff retry; httpx plumbing; SSE parser; `<think>` filter. |

## Commands
```bash
uv --directory packages/llm-transport run python -m pytest -q
uv --directory packages/llm-transport run ruff check .
uv --directory packages/llm-transport run pyright
```

## Tests
`tests/` (`test_anthropic_wire`/`test_openai_wire`/`test_contract`, no `conftest.py`).
**No env / no real network / no API keys** — everything uses `httpx.MockTransport` via an injected
`http_client` (the `make_client` "not owned" path) or hand-rolled fakes. `asyncio_mode = "auto"`.

## Benchmarks
No dedicated suite; the wires are smoke-checked by `benchmarks/gates/minimax-smoke.sh` and
`benchmarks/gates/provider-env-check.sh`, and exercised end-to-end via the retrieval/skill suites
through `rag_core.llm`. Free ladder: `bash benchmarks/ci-free-gates.sh`.

## Invariants & patterns (module-specific)
- **No vendor SDK** — only `httpx`. No `anthropic`, no `openai`, **no LangChain**. `errors.py`/`_retry.py`
  exist precisely to replace SDK exceptions + SDK retry.
- **No `rag_core` / no DB import** — `factory.py` re-declares kind strings rather than importing them
  (one-way dependency: `rag_core` wires config in, never the reverse).
- **Model id is per-call** via `LlmRequest.model`, set into the body by both wires — never inlined or
  hardcoded here. The per-stage policy concern lives in the consumer (`rag_core.llm`); there is **no
  `BotPolicy` in this package**.
- **Request stays Anthropic-shaped** (canonical block shape); only the OpenAI wire translates.
- **Streams are never replayed once data has flowed** — retries cover connect/first-byte only.

## Where to look first
`__init__.py` (public API) → `base.py` (contract) → `types.py` → `factory.py` / `compat.py`, then the
two wire files. Gotchas:
- `base_url` handling **differs per wire**: Anthropic appends `/v1/messages`; OpenAI appends only
  `/chat/completions` to base_url as-is (so gateways without `/v1` work). MiniMax = `openai-compatible` + its gateway base_url.
- **`<think>` stripping is OpenAI-wire-only** (MiniMax/DeepSeek-R1/Qwen emit inline CoT); the Anthropic
  wire never surfaces reasoning as text.
- **Cache usage** (`cache_read/creation_input_tokens`) is preserved end-to-end on the Anthropic wire
  (tested as "risk #1"); OpenAI wire leaves them `None`.
- `compat.py` **silently drops** unknown kwargs (e.g. `count_usage`) — filters to a fixed set.
- No README; the design rationale lives in module docstrings (esp. `__init__.py`, `types.py`, `compat.py`).
