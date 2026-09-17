# CLAUDE.md

## Project Overview

**llmigrate** is an open-source Python library for cross-model session migration in LLM conversations. When switching models mid-conversation — for cost optimization, capability routing, context window management, failover, or multi-agent handoff — the conversation history often needs transformation to work well with the new model. llmigrate provides a single `transfer()` call with pluggable strategies that handle this transformation.

### Design Philosophy

- **Messages in, messages out.** The core operation transforms a conversation history using a specified strategy.
- **Single entry point.** `llmigrate.transfer()` is the primary API. Strategy selection is a parameter, not a function choice.
- **Provider-agnostic.** Works with any LLM provider. Model-assisted strategies accept a generic `generate` callable rather than depending on any SDK.
- **Canonical internal representation.** The library defines its own `Message` type for internal processing. Adapters convert to/from provider-native formats (OpenAI, Anthropic, etc.). Users can pass OpenAI-format dicts directly and the library auto-detects the format.
- **Zero required dependencies.** Core functionality has no external dependencies beyond the Python standard library.

## Build & Run

```bash
pip install -e ".[dev]"     # install with dev dependencies
pytest tests/ -v            # run all tests (20 tests, no external services needed)
```

## Repository Layout

```
src/llmigrate/
  __init__.py           # Public API — transfer() entry point
  types.py              # Core types: Message, Role, Strategy, TransferResult
  strategies/           # Migration strategy implementations
    __init__.py          # Strategy registry
    raw.py               # Pass conversation unchanged (T0)
    keep_last.py         # Keep system + last N turns (T1)
    token_budget.py      # Capacity-based truncation (T1b)
    summarize.py         # Summarize older history, keep recent tail (T2)
    capsule.py           # Extract structured state capsule (T3)
    audit.py             # Append verification instructions (T4)
  adapters/             # Format converters
    __init__.py
    openai.py            # OpenAI message format <-> canonical
    detect.py            # Auto-detect input format
tests/
  test_transfer.py       # Tests for transfer() and all strategies
  test_adapters.py       # Tests for format adapters
```

## Public API

### `llmigrate.transfer(messages, strategy, *, generate=None, **params) -> TransferResult`

Single entry point for all migrations.

- `messages`: `list[dict]` (OpenAI format) or `list[Message]` (canonical)
- `strategy`: string name — `"raw"`, `"keep_last"`, `"token_budget"`, `"summarize"`, `"capsule"`, `"audit"`
- `generate`: `Callable[[list[dict]], str]` — required for `summarize` and `capsule` strategies, ignored by others
- `**params`: strategy-specific parameters (see below)

Returns a `TransferResult` containing:
- `messages`: `list[Message]` — the transformed conversation
- `strategy`: which strategy was applied
- `metadata`: dict with transformation details (original_count, strategy-specific info)

### Strategy Parameters

| Strategy | Parameters | Notes |
|---|---|---|
| `raw` | (none) | Identity transform |
| `keep_last` | `n: int = 5` | Number of recent non-system messages to keep |
| `token_budget` | `max_tokens: int = 4096`, `tokenizer: Callable = default` | Custom tokenizer via `tokenizer` kwarg |
| `summarize` | `tail: int = 3`, `generate: Callable` | Without `generate`, falls back to truncation with notice |
| `capsule` | `generate: Callable`, `schema: dict = DEFAULT_SCHEMA` | Custom capsule schema via `schema` kwarg |
| `audit` | `instruction: str = DEFAULT_INSTRUCTION` | Custom audit instruction text |

### Types

- `Message(role: Role, content: str, metadata: dict)` — canonical message
- `Role` — enum: `SYSTEM`, `USER`, `ASSISTANT`, `TOOL_CALL`, `TOOL_RESULT`
- `Strategy` — enum: `RAW`, `KEEP_LAST`, `TOKEN_BUDGET`, `SUMMARIZE`, `CAPSULE`, `AUDIT`
- `TransferResult(messages, strategy, metadata)` — transform output

## Key Architecture Decisions

- **Internal canonical representation** rather than assuming any single provider format. `Message` is a simple dataclass with role, content, and a metadata dict for tool calls and other provider-specific fields. Adapters handle format conversion at the boundary.
- **Strategy registry** (`STRATEGY_REGISTRY` in `strategies/__init__.py`) maps string names to transform functions, enabling config-driven usage and easy extension.
- **Each strategy is a module** with a `transform(messages: list[Message], **params) -> TransferResult` function. No class hierarchy — just functions.
- **Auto-detection** of input format in `adapters/detect.py`. Currently supports OpenAI dicts and canonical Messages. More adapters (Anthropic, LangChain, etc.) can be added without changing the core.
- **System messages are always preserved** by truncation strategies (keep_last, token_budget).
- **Model-assisted strategies** (`summarize`, `capsule`) accept a `generate` callable with signature `Callable[[list[dict]], str]`. This decouples the library from any provider SDK. The callable receives OpenAI-format messages and returns a string.
- **Metadata on synthetic messages**: messages created by the library (summaries, capsules, audit instructions) carry `metadata["llmigrate_synthetic"] = True` so downstream code can distinguish them.

## Current State (v0.1.0 — Scaffolding)

This is an initial scaffolding with working but minimal implementations. All strategies work end-to-end but have known gaps marked with TODO comments:

### Known TODOs

1. **Turn boundary alignment** — `keep_last` and `token_budget` currently count individual messages, not user/assistant turn pairs. They should never split a turn pair or orphan a tool result from its tool call.
2. **Tool call/result integrity** — truncation strategies must handle the case where dropping a message would orphan a tool_call or tool_result.
3. **Async support** — `summarize` and `capsule` should accept async `generate` callables. Consider `atransfer()` or auto-detecting coroutine functions.
4. **Anthropic adapter** — convert to/from Anthropic's message format (separate system param, content blocks, tool_use/tool_result block types).
5. **Summarization prompt quality** — the summarization and capsule extraction prompts are basic placeholders. They need careful prompt engineering.
6. **Heuristic capsule extraction** — the fallback when no `generate` is provided is a stub. Implement actual heuristic extraction (keyword extraction, last-N distillation, etc.).
7. **Output format convenience** — users who pass OpenAI dicts probably want OpenAI dicts back. Consider adding a convenience method on `TransferResult` (e.g., `result.to_openai()`) or a parameter on `transfer()`.
8. **Composability** — strategies compose naturally via chaining (`audit(summarize(messages))` works), but a config-driven pipeline API might be useful.
9. **Token counting** — the default tokenizer is a rough heuristic (~4 chars/token). Consider optional tiktoken integration or a better heuristic.
10. **Validation** — no input validation yet (empty messages, messages without content, etc.).

### Open Design Questions

These were identified during brainstorming but not yet resolved:

1. **Should `capsule` return structured data or messages?** Currently returns messages (capsule serialized as text). An `extract_capsule() -> dict` variant could be useful for users who want the structured data directly.
2. **The `generate` callable signature** — currently `Callable[[list[dict]], str]`. Should the library be able to pass generation parameters (temperature, max_tokens) to control the summarization call? If so, the signature needs kwargs support.
3. **Message format normalization** — should the library handle cross-provider format conversion as a standalone utility (e.g., `llmigrate.normalize(messages, from_format="anthropic", to_format="openai")`)? Or is that scope creep?
4. **Metadata preservation** — when truncating or summarizing, what metadata from dropped messages should be preserved in the result metadata? Token counts? Model IDs? Timestamps?
5. **The `transfer()` signature** — the current signature works but may need refinement. The brainstorming considered a more explicit form: `transfer(state=current_state, source_model=src, target_model=tgt, config=cnf)`. The `source_model`/`target_model` parameters might be useful for strategies that behave differently based on the models involved (e.g., adjusting token budget for the target's context window).

## Use Cases

These are the real-world scenarios llmigrate is designed to serve:

1. **Cost optimization** — start on an expensive model for hard reasoning, hand off to a cheaper model (with summarized context) for ongoing Q&A.
2. **Context window management** — compress a conversation approaching the context limit to fit within budget.
3. **Multi-agent handoff** — research agent hands off to writing agent with structured capsule of findings.
4. **Model fallback/migration** — provider down or switching providers mid-conversation. Transfer with context verification via audit strategy.
5. **Escalation in support bots** — tier-1 (small model) escalates to tier-2 (large model) with appropriate context.
6. **Capability routing** — start with a fast model, escalate to a stronger one when complexity increases.

## Mapping to route-and-adapt

For reference, here is how llmigrate strategies correspond to route-and-adapt's transfer mechanisms:

| llmigrate | route-and-adapt | Action space |
|---|---|---|
| `raw` | `raw` | T0 — no transformation |
| `keep_last` | `keep_last_n` | T1 — turn-based truncation |
| `token_budget` | `token_budget` | T1b — capacity-based truncation |
| `summarize` | `simple_summary` | T2 — abstractive summary + tail |
| `capsule` | `structured_capsule` | T3 — structured state extraction |
| `audit` | `audit_and_repair` | T4 — verification instruction |

The route-and-adapt implementations are tightly coupled to its trajectory/experiment infrastructure. llmigrate reimplements the core transformation logic with a focus on standalone usability.

## Testing

20 tests covering:
- All six strategies via the `transfer()` entry point
- Input format auto-detection (OpenAI dicts and canonical Messages)
- OpenAI adapter roundtrip
- Tool call preservation through adapter conversion
- Edge cases (empty content, budget overflow, unknown strategy)
- Input immutability

All tests run locally with no external services.

## Adding a New Strategy

1. Create a new module in `src/llmigrate/strategies/` with a `transform(messages: list[Message], **params) -> TransferResult` function.
2. Register it in `strategies/__init__.py` by adding to `STRATEGY_REGISTRY`.
3. Add tests in `tests/test_transfer.py`.

## Adding a New Adapter

1. Create a new module in `src/llmigrate/adapters/` with `from_<format>()` and `to_<format>()` functions.
2. Add the format to `adapters/detect.py:auto_convert()` for auto-detection.
3. Export from `adapters/__init__.py`.
4. Add tests in `tests/test_adapters.py`.
