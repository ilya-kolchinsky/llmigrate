# CLAUDE.md

## Project Overview

**llmigrate** is an open-source Python library for cross-model session migration in LLM conversations. When switching models mid-conversation — for cost optimization, capability routing, context window management, failover, or multi-agent handoff — the conversation history often needs transformation to work well with the new model. llmigrate provides a single `migrate()` call with composable strategies that handle this transformation.

### Design Philosophy

- **Messages in, messages out.** The core operation transforms a conversation history using a specified strategy.
- **Single entry point.** `llmigrate.migrate()` is the primary API. Strategy selection is a parameter, not a function choice.
- **Composable strategies.** Strategies are categorized into three concerns — selection, transformation, and validation — and can be freely composed. The user declares *what* they want; the library imposes the correct execution order.
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
  __init__.py            # Public API — migrate() entry point + pipeline execution
  types.py               # Core types: Message, Role, Strategy, StrategyCategory, MigrationResult
  pinning.py             # Framework-level protected content (split_pinned)
  alternation.py         # Framework-level role-alternation enforcement
  turns.py               # Turn-boundary grouping shared by truncation strategies
  tokens.py              # Shared token estimation (tiktoken if installed, else heuristic)
  models.py              # Best-effort model -> context-window lookup table
  selectors.py           # Pluggable selectors for selective_history (Priority, Relevance)
  summarizers.py         # Pluggable Summarizer abstraction with cost/latency accounting
  generators.py          # generate() builders for OpenAI-compatible endpoints (incl. vLLM)
  _util.py               # call_generate() — sync/async generate() dispatch
  strategies/            # Strategy implementations
    __init__.py            # Strategy + category registries, select/compress/adapt registries
    raw.py                  # Pass conversation unchanged
    keep_last.py            # Turn-based selection (select + transform)
    token_budget.py         # Capacity-based selection (select + transform)
    selective_history.py    # Verbatim, score-ranked event selection (select + transform)
    summarize.py            # Summarize dropped messages (compress + transform)
    structured_state.py     # Extract structured state (compress + transform)
    audit.py                # Append verification instructions (adapt + transform)
  adapters/               # Format converters
    __init__.py
    openai.py               # OpenAI message format <-> canonical
    anthropic.py            # Anthropic message format <-> canonical
    detect.py               # Auto-detect input format
tests/
  test_transfer.py         # Tests for migrate() and single-strategy use
  test_composition.py      # Tests for the composable pipeline (strategies= parameter)
  test_pinning.py          # Registry-wide invariants: protected content, role alternation
  test_selective_history.py
  test_adapters.py         # Format adapter tests (OpenAI, Anthropic, auto-detect)
  test_generators.py       # OpenAI-compatible generate() builders (mocked client)
```

## Public API

### `llmigrate.migrate(messages, strategy=None, *, strategies=None, generate=None, generate_kwargs=None, source_model=None, target_model=None, target_format=None, enforce_alternation=True, **params) -> MigrationResult`

Single entry point for all migrations.

- `messages`: `list[dict]` (OpenAI or Anthropic format, auto-detected) or `list[Message]` (canonical)
- `strategy`: string name for a single strategy — `"raw"`, `"keep_last"`, `"token_budget"`, `"summarize"`, `"structured_state"`, `"audit"`, `"selective_history"`. Sugar for `strategies=["X"]`.
- `strategies`: list of strategy names to compose. At most one from each category (selection, transformation, validation). User can pass them in any order; the library sorts internally.
- `generate`: `Callable[[list[dict]], str]` — required for `summarize`/`structured_state` when no `summarizer` is given. May be sync or async — async callables are auto-detected and awaited.
- `generate_kwargs`: dict forwarded to `generate` on every call (e.g. `{"temperature": 0.2}`), kept separate from strategy params so the two namespaces never collide.
- `source_model` / `target_model`: optional model names. Recorded in `result.metadata`; `token_budget`/`selective_history` use `target_model` to size a default token budget from a built-in context-window table (see `models.py`); `audit` uses `source_model` to customize its instruction.
- `target_format`: `"openai"` or `"anthropic"`. Controls the wire format of the output messages. When omitted, defaults to the detected format of the input (or `"openai"` if canonical `Message` objects are passed).
- `enforce_alternation`: default `True`. Merges consecutive same-role messages in the output (needed for providers like Anthropic that reject non-alternating turns). Skipped for `raw`, whose contract is "unchanged".
- `**params`: strategy-specific parameters (see below). All strategies also accept `pin_first_user: bool = True` (see Protected Content below).

Returns a `MigrationResult` containing:
- `messages`: `list[dict]` — the transformed conversation in wire format, ready for the target provider's API. For OpenAI format, system messages are included in the list. For Anthropic format, system messages are extracted into `system`.
- `strategies`: list of `Strategy` enums that were applied
- `metadata`: dict with transformation details (original_count, strategy-specific info). When using `strategies=`, metadata from each step is namespaced under `"selection"`, `"transformation"`, and `"validation"` keys. When using `strategy=`, metadata is flat.
- `format`: the wire format of the output (`"openai"` or `"anthropic"`)
- `system`: `str | None` — the system prompt content (populated for Anthropic format, `None` for OpenAI format)

### Strategy Categories

Strategies are organized into three categories. At most one strategy from each category can be used in a single `migrate()` call.

| Category | Strategies | Purpose |
|---|---|---|
| **Selection** | `keep_last`, `token_budget`, `selective_history` | Choose which messages survive verbatim |
| **Transformation** | `summarize`, `structured_state` | Compress dropped messages into something shorter |
| **Validation** | `audit` | Prepare output for the target model |

When composing, the execution order is always: selection → transformation → validation, regardless of the order passed to `strategies=`.

### Strategy Parameters

| Strategy | Category | Parameters | Notes |
|---|---|---|---|
| `raw` | (none) | (none) | Identity transform. Not subject to alternation enforcement. |
| `keep_last` | Selection | `n: int = 5` | Number of recent **turns** to keep (a turn = one user message plus everything up to the next user message). |
| `token_budget` | Selection | `max_tokens: int` (defaults from `target_model`'s context window, else 4096), `tokenizer: Callable` | Keeps whole turns from the end that fit the budget. |
| `selective_history` | Selection | `budget: int` (required), `selector: Selector` (required), `always_keep: set[str] \| None` | Selects verbatim events by score within budget. See Selective Retention below. |
| `summarize` | Transformation | `generate`, `generate_kwargs`, `max_summary_tokens: int \| None`, `summarizer: Summarizer` | Summarizes the dropped messages. Without `generate`/`summarizer`, falls back to a truncation notice. Supports the `Summarizer` protocol for cost/latency accounting. |
| `structured_state` | Transformation | `generate`, `generate_kwargs`, `schema: dict = DEFAULT_STATE_SCHEMA` | Extracts structured state from the dropped messages. Heuristic fallback is rule-based, not a stub. |
| `audit` | Validation | `instruction: str = DEFAULT_INSTRUCTION` | Appends verification instructions. Default instruction interpolates `source_model`/`target_model` when given. |

### Composition Examples

```python
# Single strategy (simple form)
migrate(messages, strategy="keep_last", n=3)

# Equivalent using strategies=
migrate(messages, strategies=["keep_last"], n=3)

# Selection + transformation: keep last 2 turns, summarize the rest
migrate(messages, strategies=["keep_last", "summarize"], n=2, generate=fn)

# Selection + validation: keep within budget, append audit
migrate(messages, strategies=["token_budget", "audit"], max_tokens=4096)

# All three: select, summarize dropped, audit the result
migrate(messages, strategies=["keep_last", "summarize", "audit"], n=3, generate=fn)

# Order doesn't matter — the library sorts internally
migrate(messages, strategies=["audit", "summarize", "keep_last"], n=3, generate=fn)
```

### Types

- `Message(role: Role, content: str, metadata: dict)` — canonical message
- `Role` — enum: `SYSTEM`, `USER`, `ASSISTANT`, `TOOL_CALL`, `TOOL_RESULT`
- `Strategy` — enum: `RAW`, `KEEP_LAST`, `TOKEN_BUDGET`, `SUMMARIZE`, `STRUCTURED_STATE`, `AUDIT`, `SELECTIVE_HISTORY`
- `StrategyCategory` — enum: `SELECTION`, `TRANSFORMATION`, `VALIDATION`
- `MigrationResult(messages, strategies, metadata, format, system)` — migration output with wire-format messages
- `SelectionResult(kept, dropped, metadata)` — internal result from selection step
- `CompressionResult(messages, metadata)` — internal result from transformation step
- `ValidationResult(messages, metadata)` — internal result from validation step
- `Selector` protocol + `PrioritySelector`, `RelevanceSelector` (`selectors.py`)
- `Summarizer` protocol + `SummarizerResult`, `TokenUsage`, `GenerateSummarizer` (`summarizers.py`)

## Key Architecture Decisions

- **Composable strategy pipeline.** Strategies fall into three categories — selection, transformation, validation — each targeting a distinct concern. Users compose by listing strategies (in any order); the library validates the composition (at most one per category) and executes in the canonical order: selection → transformation → validation. When using `strategy=` (single), the legacy `transform()` path runs directly; when using `strategies=`, the pipeline executes each step via the category-specific `select()`/`compress()`/`validate()` functions.
- **Internal canonical representation** rather than assuming any single provider format. `Message` is a simple dataclass with role, content, and a metadata dict for tool calls and other provider-specific fields. Adapters handle format conversion at the boundary.
- **Strategy registry** (`STRATEGY_REGISTRY` in `strategies/__init__.py`) maps string names to transform functions. Separate `SELECT_REGISTRY`, `COMPRESS_REGISTRY`, and `VALIDATE_REGISTRY` map strategy names to their category-specific functions for pipeline use. `STRATEGY_CATEGORIES` maps each strategy name to its `StrategyCategory`.
- **Each strategy is a module** with a `transform(messages, **params) -> MigrationResult` function for standalone use, plus a category-specific function (`select()`, `compress()`, or `validate()`) for pipeline composition. No class hierarchy — just functions. Pluggable components (`Selector`, `Summarizer`) are passed in as **parameter values**, not strategy classes.
- **Auto-detection** of input format in `adapters/detect.py`, covering OpenAI dicts, Anthropic dicts (detected via content-block lists), and canonical Messages.
- **Protected content is centralized** (`pinning.py`): every strategy that can drop or rewrite content calls `split_pinned()` (in standalone mode) or has pinning done centrally by the pipeline. Pinned = all `SYSTEM` messages + the first `USER` message (by default) + anything explicitly tagged `metadata["pinned"] = True`. Enforcement is verified by a registry-wide parametrized test (`tests/test_pinning.py::test_preserves_task_marker`).
- **Role-alternation is enforced centrally** (`alternation.py`), inside `migrate()` itself, for every strategy except `raw`. Many providers (Anthropic) reject consecutive same-role messages, which pinning can incidentally produce (e.g. a pinned first-user task immediately followed by a synthetic summary, both `USER`). The generic merge (`enforce_alternation`) fixes this.
- **Turn-boundary grouping** (`turns.py`): `group_into_turns()` groups a `USER` message with everything up to the next `USER` message, so selection strategies (`keep_last`, `token_budget`) operate on whole turns and never split a turn pair or orphan a `TOOL_RESULT` from its `TOOL_CALL`. `selective_history` is the exception — see below.
- **Model-assisted strategies** (`summarize`, `structured_state`) accept a `generate` callable with signature `Callable[[list[dict]], str]`, sync or async (auto-detected via `_util.call_generate`). `summarize` also accepts a `Summarizer` protocol object (or a bare `generate`, auto-wrapped via `as_summarizer`) for cost/latency accounting. `generators.py` provides ready-made `generate` builders for OpenAI-compatible endpoints — optional, requires `pip install llmigrate[openai]`.
- **Cost/latency accounting** (`summarizers.py`): `summarize` records wall-clock latency and estimated token usage for every model call. A bare `generate` callable is auto-wrapped in `GenerateSummarizer`, which can only report latency/token estimates; implement the `Summarizer` protocol directly against your provider client for real cost/model reporting.
- **Wire-format output**: `migrate()` always returns messages as provider-native dicts (OpenAI or Anthropic format), never canonical `Message` objects. The `target_format` parameter controls the output format, defaulting to the detected input format (or `"openai"` when canonical `Message` objects are passed). The canonical `Message` type is purely an internal implementation detail.
- **Metadata on synthetic messages**: messages created by the library (summaries, structured state extractions, audit instructions) carry `llmigrate["llmigrate_synthetic"] = True` in the wire-format dict. The `"llmigrate"` key namespaces library metadata away from provider-specific fields; extra keys are silently ignored by all major providers.

## Protected Content (framework-level)

Every strategy that can remove or rewrite content calls `pinning.split_pinned()`, which protects (verbatim, always emitted, never fed to a summarizer):
- all `SYSTEM` messages
- the first `USER` message (disable via `pin_first_user=False`)
- any message with `metadata["pinned"] = True`

This is what keeps a SWE-bench-style task description alive through `keep_last(n=0)`, an aggressive `token_budget`, or a `summarize`/`structured_state` call that would otherwise paraphrase it away.

## Selective Retention (`selective_history`)

Chooses a **verbatim** subset of events (no rewriting) that fits a token `budget`, ranked by a pluggable `Selector` rather than recency alone:
- `PrioritySelector(priorities={...}, recency_tiebreak=True)` — deterministic, category-based (default categories: `user_instruction`, `assistant`, `tool_call`, `tool_result`, overridable via `metadata["category"]`).
- `RelevanceSelector(embed=..., query=...)` — cosine similarity against a pluggable embedding function; llmigrate binds to no specific embedding provider.
- `always_keep: set[str]` forces retention of matching categories (mandatory, most-recent-first if they alone exceed the budget).
- Output is re-sorted into original chronological order; metadata includes `selected_event_ids`/`dropped_event_ids` (0-indexed positions in the input list for that call), `original_tokens`, `transferred_tokens`, `selector`, and per-candidate scores.
- **Limitation**: operates at message granularity per its spec and does *not* guarantee `TOOL_CALL`/`TOOL_RESULT` pairing the way turn-based strategies do — put both categories in `always_keep` if that matters.

## Testing

Covers, across `tests/*.py`:
- All seven strategies via the `migrate()` entry point, including validation errors and edge cases (empty content, budget overflow, unknown strategy, negative/zero params)
- Strategy composition via the `strategies=` parameter: multi-strategy pipelines, order invariance, category validation, metadata namespacing
- Registry-wide invariants (`test_pinning.py`): every strategy preserves pinned task content under tight parameters; every strategy except `raw` produces alternation-safe output
- Turn/tool-call integrity for `keep_last`/`token_budget`
- `generate_kwargs` passthrough and async `generate` auto-detection
- `source_model`/`target_model` behavior (token budget auto-sizing, audit instruction interpolation)
- `state_data` metadata parsing (both the `generate` and heuristic paths)
- `selective_history` (both selectors, budget enforcement, chronological output, `always_keep`)
- Composition: selection+transformation (skips transformation when nothing dropped), selection+validation, all three categories, custom `Summarizer` cost reporting, summary truncation
- OpenAI and Anthropic adapter roundtrips (incl. tool calls/results), format auto-detection
- `generators.py`'s OpenAI-compatible `generate` builders, via an injected fake client (no real `openai` package or network access required)

All tests run locally with no external services.

## Adding a New Strategy

1. Create a new module in `src/llmigrate/strategies/` with:
   - A `transform(messages: list[Message], **params) -> MigrationResult` function for standalone use.
   - A category-specific function: `select()` for selection, `compress()` for transformation, or `validate()` for validation.
2. If the strategy can drop or rewrite content, call `pinning.split_pinned()` at the top of `transform()` and never touch the pinned half. The pipeline handles pinning centrally for the category-specific functions.
3. Register it in `strategies/__init__.py`: add to `STRATEGY_REGISTRY`, `STRATEGY_CATEGORIES`, and the appropriate category registry (`SELECT_REGISTRY`, `COMPRESS_REGISTRY`, or `VALIDATE_REGISTRY`).
4. Add a `Strategy` enum value in `types.py`.
5. Add tests, including a `TIGHT_PARAMS` entry in `tests/test_pinning.py` so the registry-wide invariant tests cover it.

## Adding a New Adapter

1. Create a new module in `src/llmigrate/adapters/` with `from_<format>()` and `to_<format>()` functions.
2. Add detection logic to `adapters/detect.py:auto_convert()`.
3. Export from `adapters/__init__.py`.
4. Add tests in `tests/test_adapters.py`.

## Use Cases

These are the real-world scenarios llmigrate is designed to serve:

1. **Cost optimization** — start on an expensive model for hard reasoning, hand off to a cheaper model (with summarized context) for ongoing Q&A. Use `strategies=["keep_last", "summarize"]` to keep recent turns and summarize older context.
2. **Context window management** — compress a conversation approaching the context limit to fit within budget. Use `strategies=["token_budget", "summarize"]` for automatic budget-aware compression.
3. **Multi-agent handoff** — research agent hands off to writing agent with structured state of findings. Use `strategy="structured_state"` or compose with `strategies=["selective_history", "structured_state"]` to extract state from only the most relevant dropped events.
4. **Model fallback/migration** — provider down or switching providers mid-conversation. Use `strategies=["keep_last", "audit"]` to migrate with context verification.
5. **Escalation in support bots** — tier-1 (small model) escalates to tier-2 (large model) with appropriate context.
6. **Capability routing** — start with a fast model, escalate to a stronger one when complexity increases.
7. **Long-running coding/agentic tasks** — SWE-bench-style sessions where the task description must survive arbitrarily aggressive context compression (this is why protected content is a framework-level guarantee, not opt-in).
8. **Self-hosted/local models** — routing to or from a model served via vLLM, LocalAI, or another OpenAI-compatible endpoint, using `generators.openai_compatible_generate()`.

## Mapping to route-and-adapt

For reference, here is how llmigrate strategies correspond to route-and-adapt's migration mechanisms:

| llmigrate | route-and-adapt | Action space |
|---|---|---|
| `raw` | `raw` | T0 — no transformation |
| `keep_last` | `keep_last_n` | T1 — turn-based truncation |
| `token_budget` | `token_budget` | T1b — capacity-based truncation |
| `summarize` | `simple_summary` | T2 — abstractive summary |
| `structured_state` | `structured_capsule` | T3 — structured state extraction |
| `audit` | `audit_and_repair` | T4 — verification instruction |
| `selective_history` | *(no equivalent)* | new capability — verbatim, score-ranked event selection |
| `["token_budget", "summarize"]` | generalizes `simple_summary` | replaces the old `summary_tail` strategy; composable budget-aware compression |

route-and-adapt's `handoffs/turn_slicing.py:split_immutable_and_trajectory()` (system + first-user-event protection, enforced via a registry-wide invariant test) and its role-alternation re-alignment helpers were the direct inspiration for llmigrate's `pinning.py`/`alternation.py`. Its cost/latency reporting shape (`latency_ms` + token usage, with dollar cost left to the caller since llmigrate has no pricing table) is mirrored in `summarizers.py`.

The route-and-adapt implementations are tightly coupled to its trajectory/experiment infrastructure. llmigrate reimplements the core transformation logic with a focus on standalone usability.
