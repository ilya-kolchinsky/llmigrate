# CLAUDE.md

## Project Overview

**llmigrate** is an open-source Python library for cross-model session migration in LLM conversations. When switching models mid-conversation — for cost optimization, capability routing, context window management, failover, or multi-agent handoff — the conversation history often needs transformation to work well with the new model. llmigrate provides a single `transfer()` call with pluggable strategies that handle this transformation.

### Design Philosophy

- **Messages in, messages out.** The core operation transforms a conversation history using a specified strategy.
- **Single entry point.** `llmigrate.transfer()` is the primary API. Strategy selection is a parameter, not a function choice.
- **Provider-agnostic.** Works with any LLM provider. Model-assisted strategies accept a generic `generate` callable rather than depending on any SDK. OpenAI-compatible self-hosted servers (vLLM, LocalAI, LM Studio, Ollama's OpenAI-compat mode, etc.) work out of the box, since they speak the same wire format as OpenAI.
- **Canonical internal representation.** The library defines its own `Message` type for internal processing. Adapters convert to/from provider-native formats (OpenAI, Anthropic). Users can pass OpenAI- or Anthropic-format dicts directly and the library auto-detects the format.
- **Protected content is a framework concern, not a per-strategy one.** No strategy — truncation, summarization, or selection — may drop or dilute the system prompt or the first user message (e.g. a task description). This is enforced centrally, not left to each strategy to remember.
- **Zero required dependencies.** Core functionality has no external dependencies beyond the Python standard library. `tiktoken` and `openai` are optional extras.

## Build & Run

```bash
pip install -e ".[dev]"              # install with dev dependencies
pip install -e ".[dev,tiktoken]"     # + accurate token counting
pip install -e ".[dev,openai]"       # + OpenAI-compatible generate() builders
pytest tests/ -v                     # run all tests (no external services needed)
mypy src/llmigrate                   # strict type checking
ruff check src/llmigrate tests       # linting
```

## Repository Layout

```
src/llmigrate/
  __init__.py            # Public API — transfer() entry point
  types.py               # Core types: Message, Role, Strategy, TransferResult
  pinning.py             # Framework-level protected content (split_pinned)
  alternation.py         # Framework-level role-alternation enforcement
  turns.py               # Turn-boundary grouping shared by truncation strategies
  tokens.py              # Shared token estimation (tiktoken if installed, else heuristic)
  models.py              # Best-effort model -> context-window lookup table
  selectors.py           # Pluggable selectors for selective_history (Priority, Relevance)
  summarizers.py         # Pluggable Summarizer abstraction with cost/latency accounting
  generators.py          # generate() builders for OpenAI-compatible endpoints (incl. vLLM)
  _util.py               # call_generate() — sync/async generate() dispatch
  strategies/            # Migration strategy implementations
    __init__.py            # Strategy registry
    raw.py                  # Pass conversation unchanged (T0)
    keep_last.py            # Turn-based truncation (T1)
    token_budget.py         # Capacity-based truncation (T1b)
    summarize.py            # Summarize older history, keep recent tail (T2)
    capsule.py               # Extract structured state capsule (T3)
    audit.py                 # Append verification instructions (T4)
    selective_history.py     # Verbatim, budget-constrained event selection
    summary_tail.py          # Summarized prefix + verbatim tail, with cost/latency accounting
  adapters/               # Format converters
    __init__.py
    openai.py               # OpenAI message format <-> canonical
    anthropic.py            # Anthropic message format <-> canonical
    detect.py               # Auto-detect input format
tests/
  test_transfer.py         # Tests for transfer() and the original six strategies
  test_pinning.py           # Registry-wide invariants: protected content, role alternation
  test_selective_history.py
  test_summary_tail.py
  test_adapters.py          # Format adapter tests (OpenAI, Anthropic, auto-detect)
  test_generators.py        # OpenAI-compatible generate() builders (mocked client)
```

## Public API

### `llmigrate.transfer(messages, strategy, *, generate=None, generate_kwargs=None, source_model=None, target_model=None, enforce_alternation=True, **params) -> TransferResult`

Single entry point for all migrations.

- `messages`: `list[dict]` (OpenAI or Anthropic format, auto-detected) or `list[Message]` (canonical)
- `strategy`: string name — `"raw"`, `"keep_last"`, `"token_budget"`, `"summarize"`, `"capsule"`, `"audit"`, `"selective_history"`, `"summary_tail"`
- `generate`: `Callable[[list[dict]], str]` — required for `summarize`/`capsule` (and `summary_tail` if no `summarizer` given). May be sync or async — async callables are auto-detected and awaited.
- `generate_kwargs`: dict forwarded to `generate` on every call (e.g. `{"temperature": 0.2}`), kept separate from strategy params so the two namespaces never collide.
- `source_model` / `target_model`: optional model names. Recorded in `result.metadata`; `token_budget`/`summary_tail`/`selective_history` use `target_model` to size a default token budget from a built-in context-window table (see `models.py`); `audit` uses `source_model` to customize its instruction.
- `enforce_alternation`: default `True`. Merges consecutive same-role messages in the output (needed for providers like Anthropic that reject non-alternating turns). Skipped for `raw`, whose contract is "unchanged".
- `**params`: strategy-specific parameters (see below). All strategies also accept `pin_first_user: bool = True` (see Protected Content below).

Returns a `TransferResult` containing:
- `messages`: `list[Message]` — the transformed conversation
- `strategy`: which strategy was applied
- `metadata`: dict with transformation details (original_count, strategy-specific info)
- `.to_openai()` / `.to_anthropic()`: convert the result back to a provider-native format

### Strategy Parameters

| Strategy | Parameters | Notes |
|---|---|---|
| `raw` | (none) | Identity transform. Not subject to alternation enforcement. |
| `keep_last` | `n: int = 5` | Number of recent **turns** to keep (a turn = one user message plus everything up to the next user message). |
| `token_budget` | `max_tokens: int` (defaults from `target_model`'s context window, else 4096), `tokenizer: Callable` | Keeps whole turns from the end that fit the budget. |
| `summarize` | `tail: int = 3`, `generate`, `generate_kwargs`, `max_summary_tokens: int | None` | Without `generate`, falls back to truncation with notice. Records `latency_ms`. |
| `capsule` | `generate`, `generate_kwargs`, `schema: dict = DEFAULT_SCHEMA` | Populates `result.metadata["capsule_data"]` (parsed structured fields) in addition to the message. Heuristic fallback is rule-based, not a stub. |
| `audit` | `instruction: str = DEFAULT_INSTRUCTION` | Default instruction interpolates `source_model`/`target_model` when given. |
| `selective_history` | `budget: int` (required), `selector: Selector` (required), `always_keep: set[str] | None` | Never rewrites content; selects verbatim events by score within budget. See Selective Retention below. |
| `summary_tail` | `total_budget: int`, `summary_budget: int`, `tail_budget: int | None`, `summarizer: Summarizer | Callable` | Summarizes the prefix, keeps a turn-aligned verbatim tail. See Summary + Tail below. |

### Types

- `Message(role: Role, content: str, metadata: dict)` — canonical message
- `Role` — enum: `SYSTEM`, `USER`, `ASSISTANT`, `TOOL_CALL`, `TOOL_RESULT`
- `Strategy` — enum: `RAW`, `KEEP_LAST`, `TOKEN_BUDGET`, `SUMMARIZE`, `CAPSULE`, `AUDIT`, `SELECTIVE_HISTORY`, `SUMMARY_TAIL`
- `TransferResult(messages, strategy, metadata)` — transform output, with `.to_openai()`/`.to_anthropic()`
- `Selector` protocol + `PrioritySelector`, `RelevanceSelector` (`selectors.py`)
- `Summarizer` protocol + `SummarizerResult`, `TokenUsage`, `GenerateSummarizer` (`summarizers.py`)

## Key Architecture Decisions

- **Internal canonical representation** rather than assuming any single provider format. `Message` is a simple dataclass with role, content, and a metadata dict for tool calls and other provider-specific fields. Adapters handle format conversion at the boundary.
- **Strategy registry** (`STRATEGY_REGISTRY` in `strategies/__init__.py`) maps string names to transform functions, enabling config-driven usage and easy extension.
- **Each strategy is a module** with a `transform(messages: list[Message], **params) -> TransferResult` function. No class hierarchy — just functions. Pluggable components (`Selector`, `Summarizer`) are passed in as **parameter values**, not strategy classes — this keeps the "single entry point, strategy is a string" design intact for `selective_history`/`summary_tail` too.
- **Auto-detection** of input format in `adapters/detect.py`, covering OpenAI dicts, Anthropic dicts (detected via content-block lists), and canonical Messages.
- **Protected content is centralized** (`pinning.py`): every strategy that can drop or rewrite content calls `split_pinned()` instead of its own ad hoc filtering. Pinned = all `SYSTEM` messages + the first `USER` message (by default) + anything explicitly tagged `metadata["pinned"] = True`. This exists because truncation/summarization can otherwise silently drop or dilute critical context — e.g. a SWE-bench task description in the first user message, after which the receiving model has no way to succeed. Enforcement is verified by a registry-wide parametrized test (`tests/test_pinning.py::test_preserves_task_marker`), following the pattern used by the sibling `route-and-adapt` research project rather than a structural type-level guarantee.
- **Role-alternation is enforced centrally** (`alternation.py`), inside `transfer()` itself, for every strategy except `raw`. Many providers (Anthropic) reject consecutive same-role messages, which pinning can incidentally produce (e.g. a pinned first-user task immediately followed by a synthetic summary, both `USER`). The generic merge (`enforce_alternation`) fixes this and, as a side effect, guarantees pinned content survives verbatim inside the merged message even if a model-generated summary doesn't reproduce it.
- **Turn-boundary grouping** (`turns.py`): `group_into_turns()` groups a `USER` message with everything up to the next `USER` message, so truncation strategies (`keep_last`, `token_budget`, `summary_tail`'s tail) operate on whole turns and never split a turn pair or orphan a `TOOL_RESULT` from its `TOOL_CALL`. `selective_history` is the exception — see below.
- **Model-assisted strategies** (`summarize`, `capsule`, `summary_tail`) accept a `generate` callable with signature `Callable[[list[dict]], str]`, sync or async (auto-detected via `_util.call_generate`). This decouples the library from any provider SDK. `generators.py` provides ready-made `generate` builders for OpenAI-compatible endpoints (OpenAI itself, vLLM, LocalAI, etc.) for convenience — optional, requires `pip install llmigrate[openai]`.
- **Cost/latency accounting** (`summarizers.py`): `summary_tail` (and `summarize`, for `latency_ms`) records wall-clock latency and estimated token usage for every model call. A bare `generate` callable is auto-wrapped in `GenerateSummarizer`, which can only report latency/token estimates; implement the `Summarizer` protocol directly against your provider client for real cost/model reporting.
- **Metadata on synthetic messages**: messages created by the library (summaries, capsules, audit instructions) carry `metadata["llmigrate_synthetic"] = True`. `summary_tail` additionally tags `metadata["segment"]` (`"summary_prefix"` / `"verbatim_tail"`) so the summarized/verbatim boundary is explicit in the output.

## Protected Content (framework-level)

Every strategy that can remove or rewrite content calls `pinning.split_pinned()`, which protects (verbatim, always emitted, never fed to a summarizer):
- all `SYSTEM` messages
- the first `USER` message (disable via `pin_first_user=False`)
- any message with `metadata["pinned"] = True`

This is what keeps a SWE-bench-style task description alive through `keep_last(n=0)`, an aggressive `token_budget`, or a `summarize`/`capsule` call that would otherwise paraphrase it away.

## Selective Retention (`selective_history`)

Chooses a **verbatim** subset of events (no rewriting) that fits a token `budget`, ranked by a pluggable `Selector` rather than recency alone:
- `PrioritySelector(priorities={...}, recency_tiebreak=True)` — deterministic, category-based (default categories: `user_instruction`, `assistant`, `tool_call`, `tool_result`, overridable via `metadata["category"]`).
- `RelevanceSelector(embed=..., query=...)` — cosine similarity against a pluggable embedding function; llmigrate binds to no specific embedding provider.
- `always_keep: set[str]` forces retention of matching categories (mandatory, most-recent-first if they alone exceed the budget).
- Output is re-sorted into original chronological order; metadata includes `selected_event_ids`/`dropped_event_ids` (0-indexed positions in the input list for that call), `original_tokens`, `transferred_tokens`, `selector`, and per-candidate scores.
- **Limitation**: operates at message granularity per its spec and does *not* guarantee `TOOL_CALL`/`TOOL_RESULT` pairing the way turn-based strategies do — put both categories in `always_keep` if that matters.

## Summary + Tail (`summary_tail`)

Combines a summarized prefix with a verbatim, turn-aligned tail:
- The tail is the largest recent contiguous run of whole turns that fits `tail_budget` (reuses `turns.group_into_turns` + the same greedy-from-the-end logic as `token_budget`).
- If the whole (non-pinned) history already fits `tail_budget`, no summarizer call is made.
- Otherwise the prefix is summarized under `summary_budget` via a `Summarizer` (or a bare `generate`, auto-wrapped); an oversized summary is truncated to fit, flagged via `metadata["summary_truncated"]`.
- `summary_budget + (tail_budget or total_budget - summary_budget)` must not exceed `total_budget` (`ValueError` otherwise).

## Testing

Covers, across `tests/*.py`:
- All eight strategies via the `transfer()` entry point, including validation errors and edge cases (empty content, budget overflow, unknown strategy, negative/zero params)
- Registry-wide invariants (`test_pinning.py`): every strategy preserves pinned task content under tight parameters; every strategy except `raw` produces alternation-safe output
- Turn/tool-call integrity for `keep_last`/`token_budget`
- `generate_kwargs` passthrough and async `generate` auto-detection
- `source_model`/`target_model` behavior (token budget auto-sizing, audit instruction interpolation)
- `capsule_data` metadata parsing (both the `generate` and heuristic paths)
- `selective_history` (both selectors, budget enforcement, chronological output, `always_keep`)
- `summary_tail` (verbatim-if-fits shortcut, budget validation, truncation, custom `Summarizer` cost reporting)
- OpenAI and Anthropic adapter roundtrips (incl. tool calls/results), format auto-detection
- `generators.py`'s OpenAI-compatible `generate` builders, via an injected fake client (no real `openai` package or network access required)

All tests run locally with no external services.

## Adding a New Strategy

1. Create a new module in `src/llmigrate/strategies/` with a `transform(messages: list[Message], **params) -> TransferResult` function.
2. If the strategy can drop or rewrite content, call `pinning.split_pinned()` at the top and never touch the pinned half.
3. Register it in `strategies/__init__.py` by adding to `STRATEGY_REGISTRY` and add a `Strategy` enum value in `types.py`.
4. Add tests, including a `TIGHT_PARAMS` entry in `tests/test_pinning.py` so the registry-wide invariant tests cover it.

## Adding a New Adapter

1. Create a new module in `src/llmigrate/adapters/` with `from_<format>()` and `to_<format>()` functions.
2. Add detection logic to `adapters/detect.py:auto_convert()`.
3. Export from `adapters/__init__.py`.
4. Add tests in `tests/test_adapters.py`.

## Use Cases

These are the real-world scenarios llmigrate is designed to serve:

1. **Cost optimization** — start on an expensive model for hard reasoning, hand off to a cheaper model (with summarized context) for ongoing Q&A.
2. **Context window management** — compress a conversation approaching the context limit to fit within budget.
3. **Multi-agent handoff** — research agent hands off to writing agent with structured capsule of findings.
4. **Model fallback/migration** — provider down or switching providers mid-conversation. Transfer with context verification via audit strategy.
5. **Escalation in support bots** — tier-1 (small model) escalates to tier-2 (large model) with appropriate context.
6. **Capability routing** — start with a fast model, escalate to a stronger one when complexity increases.
7. **Long-running coding/agentic tasks** — SWE-bench-style sessions where the task description must survive arbitrarily aggressive context compression (this is why protected content is a framework-level guarantee, not opt-in).
8. **Self-hosted/local models** — routing to or from a model served via vLLM, LocalAI, or another OpenAI-compatible endpoint, using `generators.openai_compatible_generate()`.

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
| `summary_tail` | generalizes `simple_summary` | independently configurable prefix/tail budgets, pluggable `Summarizer` with cost/latency accounting |
| `selective_history` | *(no equivalent)* | new capability — verbatim, score-ranked event selection |

route-and-adapt's `handoffs/turn_slicing.py:split_immutable_and_trajectory()` (system + first-user-event protection, enforced via a registry-wide invariant test) and its role-alternation re-alignment helpers were the direct inspiration for llmigrate's `pinning.py`/`alternation.py`. Its cost/latency reporting shape (`latency_ms` + token usage, with dollar cost left to the caller since llmigrate has no pricing table) is mirrored in `summarizers.py`.

The route-and-adapt implementations are tightly coupled to its trajectory/experiment infrastructure. llmigrate reimplements the core transformation logic with a focus on standalone usability.
