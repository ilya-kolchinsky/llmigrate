# llmigrate API Reference

This document is the rigorous reference for llmigrate: the `migrate()` entry point, every strategy and its parameters, the core types, and the supporting abstractions (pinning, alternation, selectors, summarizers, token estimation, model context windows, `generate()` builders, and format adapters). For the pitch and a quick start, see the [README](../README.md).

## Table of Contents

- [`migrate()`](#migrate)
- [Core Types](#core-types)
- [Strategies](#strategies)
  - [`raw`](#raw)
  - [`keep_last`](#keep_last)
  - [`token_budget`](#token_budget)
  - [`summarize`](#summarize)
  - [`structured_state`](#structured_state)
  - [`audit`](#audit)
  - [`selective_history`](#selective_history)
- [Strategy Composition](#strategy-composition)
- [Protected Content (pinning)](#protected-content-pinning)
- [Role Alternation](#role-alternation)
- [Turn Grouping](#turn-grouping)
- [Selectors](#selectors)
- [Summarizers](#summarizers)
- [Token Estimation](#token-estimation)
- [Model Context Windows](#model-context-windows)
- [`generate()` Callables](#generate-callables)
- [Format Adapters](#format-adapters)
- [Errors & Validation](#errors--validation)
- [Adding a New Strategy](#adding-a-new-strategy)

## `migrate()`

```python
llmigrate.migrate(
    messages: list[dict] | list[Message],
    strategy: str | None = None,
    *,
    strategies: list[str] | None = None,
    generate: Callable[..., str] | None = None,
    generate_kwargs: dict[str, Any] | None = None,
    source_model: str | None = None,
    target_model: str | None = None,
    target_format: str | None = None,
    enforce_alternation: bool = True,
    **params: Any,
) -> MigrationResult
```

The single entry point for every migration. `migrate()`:

1. Validates `messages` (must be a `list`; every element must be a `dict` with a `"role"` key, or a `Message`).
2. Auto-detects the input format (OpenAI dict, Anthropic dict, or canonical `Message`) and converts to canonical `Message` objects — see [Format Adapters](#format-adapters).
3. Validates that every message's `content` is a `str`.
4. Routes to the appropriate execution path:
   - `strategy="X"` (single): looks up `X` in the strategy registry and calls its `transform()` function directly, producing flat metadata.
   - `strategies=[...]` (pipeline): validates the composition (at most one per category), then executes the pipeline in canonical order: selection → transformation → validation. Metadata is namespaced by category.
   - Neither specified (or `strategy="raw"` / `strategies=[]`): identity transform.
5. Records `source_model`/`target_model` into `result.metadata` if not already set by the strategy.
6. Unless the strategy is `raw`, merges consecutive same-role messages in the output — see [Role Alternation](#role-alternation). Controlled by `enforce_alternation`.
7. Converts canonical `Message` objects to wire-format dicts (OpenAI or Anthropic) based on `target_format`.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[dict] \| list[Message]` | required | Conversation history. OpenAI-format dicts, Anthropic-format dicts, or canonical `Message` objects — auto-detected, may not be mixed within a single call except uniformly as `Message`. |
| `strategy` | `str \| None` | `None` | Single strategy name — `"raw"`, `"keep_last"`, `"token_budget"`, `"summarize"`, `"structured_state"`, `"audit"`, `"selective_history"`. Sugar for `strategies=["X"]` but with flat (not namespaced) metadata. Mutually exclusive with `strategies`. |
| `strategies` | `list[str] \| None` | `None` | List of strategy names to compose, in any order. At most one from each category (selection, transformation, validation). The library sorts internally: selection → transformation → validation. Mutually exclusive with `strategy`. |
| `generate` | `Callable[[list[dict]], str] \| None` | `None` | Model call used by `summarize` and `structured_state`. May be a sync or async callable — async is auto-detected and awaited via `asyncio.run`. Forwarded into `params["generate"]`. |
| `generate_kwargs` | `dict \| None` | `None` | Extra keyword arguments (e.g. `{"temperature": 0.2}`) passed to `generate` (or to the auto-wrapped `Summarizer`) on every call. Kept as a separate namespace from strategy params so the two can never collide — e.g. a strategy param named `temperature` would otherwise be ambiguous. |
| `source_model` | `str \| None` | `None` | Name of the model the conversation started on. Recorded in `result.metadata["source_model"]`; interpolated into `audit`'s default instruction. |
| `target_model` | `str \| None` | `None` | Name of the model the conversation is moving to. Recorded in `result.metadata["target_model"]`; used by `token_budget` and `selective_history` to size a default token budget and to pick a model-appropriate tokenizer — see [Model Context Windows](#model-context-windows) and [Token Estimation](#token-estimation). |
| `target_format` | `str \| None` | `None` | `"openai"` or `"anthropic"`. Controls the wire format of the output messages. When omitted, defaults to the detected format of the input (or `"openai"` if canonical `Message` objects are passed). |
| `enforce_alternation` | `bool` | `True` | Merge consecutive same-role messages in the result. Always skipped for `strategy="raw"` regardless of this flag, since `raw`'s contract is "unchanged". |
| `**params` | `Any` | — | Strategy-specific parameters. See each strategy's table below. Every strategy that can drop or rewrite content also accepts `pin_first_user: bool = True` (see [Protected Content](#protected-content-pinning)). |

### Returns

A `MigrationResult` (see [Core Types](#core-types)).

### Raises

- `ValueError` — unknown `strategy` name, or a strategy-specific validation failure (see each strategy's table).
- `ValueError` — both `strategy` and `strategies` specified.
- `ValueError` — `strategies` contains duplicate categories (e.g. two selection strategies).
- `ValueError` — `"raw"` combined with other strategies in `strategies`.
- `TypeError` — `messages` is not a `list`, or an element is neither a `dict` nor a `Message`.
- `ValueError` — a `dict` element is missing the `"role"` key, or any message's `content` is not a `str` after conversion.
- `ValueError` — unknown `target_format`.

## Core Types

### `Message`

```python
@dataclass
class Message:
    role: Role
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

The canonical, provider-agnostic message representation. `to_dict()` / `from_dict()` round-trip through `{"role": ..., "content": ..., "metadata": ...}` (metadata omitted when empty). This type is purely internal — `migrate()` always returns wire-format dicts, not `Message` objects.

Metadata keys used across the library:

| Key | Set by | Meaning |
|---|---|---|
| `pinned` | caller | Forces this message to be treated as protected content by `split_pinned()`, regardless of role or position. |
| `category` | caller | Overrides the default role-derived category used by `selective_history`'s selectors (see [Selectors](#selectors)). |
| `llmigrate_synthetic` | library | Set on any message the library generates (summaries, structured state extractions, audit instructions). Also propagated onto a merged message by `enforce_alternation` if either half was synthetic. |
| `tool_calls`, `tool_call_id`, `name` | OpenAI adapter | Preserved OpenAI tool-call fields, round-tripped by `to_openai()`. |
| `anthropic_content`, `tool_call_id` | Anthropic adapter | Preserved Anthropic content-block payloads, round-tripped by `to_anthropic()`. |

### `Role`

```python
class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
```

### `Strategy`

```python
class Strategy(Enum):
    RAW = "raw"
    KEEP_LAST = "keep_last"
    TOKEN_BUDGET = "token_budget"
    SUMMARIZE = "summarize"
    STRUCTURED_STATE = "structured_state"
    AUDIT = "audit"
    SELECTIVE_HISTORY = "selective_history"
```

### `StrategyCategory`

```python
class StrategyCategory(Enum):
    SELECTION = "selection"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
```

Strategies are organized into categories. At most one strategy from each category can be used in a single `migrate()` call.

| Category | Strategies | Purpose |
|---|---|---|
| **Selection** | `keep_last`, `token_budget`, `selective_history` | Choose which messages survive verbatim |
| **Transformation** | `summarize`, `structured_state` | Compress dropped messages into something shorter |
| **Validation** | `audit` | Prepare output for the target model |

### `MigrationResult`

```python
@dataclass
class MigrationResult:
    messages: list[dict[str, Any]]
    strategies: list[Strategy]
    metadata: dict[str, Any] = field(default_factory=dict)
    format: str | None = None
    system: str | None = None
```

- `.messages` — the transformed conversation in wire format (OpenAI or Anthropic dicts), ready to pass directly to the target provider's API. For OpenAI format, system messages are included in the list. For Anthropic format, system messages are extracted into `.system`.
- `.strategies` — list of `Strategy` enums that were applied.
- `.metadata` — transformation details. When using `strategy=` (single), metadata is flat. When using `strategies=` (pipeline), metadata from each step is namespaced under `"selection"`, `"transformation"`, and `"validation"` keys.
- `.format` — the wire format of the output (`"openai"` or `"anthropic"`).
- `.system` — the system prompt content (populated for Anthropic format, `None` for OpenAI format).
- `.original_count` — `int(metadata.get("original_count", 0))`, the length of the input.
- `.transferred_count` — `len(messages)`, the length of the output.

Every strategy sets `metadata["original_count"]`; strategy-specific keys are documented per strategy below.

### Pipeline Result Types

These types are used internally by the composable pipeline and by category-specific strategy functions:

```python
@dataclass
class SelectionResult:
    kept: list[Message]
    dropped: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class CompressionResult:
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)
```

## Strategies

Each strategy module exposes:
- A `transform(messages, **params) -> MigrationResult` function for standalone use via `strategy=`.
- A category-specific function for pipeline composition via `strategies=`: `select()` for selection, `compress()` for transformation, or `validate()` for validation.

### `raw`

Identity transform. Returns the input unchanged, including its exact message list — not even role-alternation merging is applied (`migrate()` special-cases `raw` out of that step regardless of `enforce_alternation`).

| Parameter | Type | Default | Notes |
|---|---|---|---|
| *(none)* | | | |

**Metadata:** `original_count`.

**When to use:** provider/protocol change only, no content transformation — e.g. moving between two deployments of the same model family where context size isn't a concern.

### `keep_last`

**Category:** Selection.

Turn-based truncation: keeps pinned content plus the last `n` turns, dropping everything older. A "turn" is a `USER` message plus everything up to (not including) the next `USER` message — see [Turn Grouping](#turn-grouping).

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `n` | `int` | `5` | Number of most recent turns to keep. `0` keeps only pinned content. Raises `ValueError` if negative. |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |

**Behavior:** splits pinned vs. rest via `split_pinned()`, groups `rest` into turns, keeps the last `n` turn-groups verbatim, drops the rest. Output is `pinned + kept_turns`.

**Metadata:** `original_count`, `n`, `dropped_count` (messages dropped, not turns).

**When to use:** capacity routing where a fixed number of recent exchanges is "enough" — e.g. tier-1→tier-2 escalation in a support bot, where only the last few turns are relevant to the immediate question.

### `token_budget`

**Category:** Selection.

Capacity-based truncation: keeps as many whole recent turns as fit within a token budget, instead of a fixed turn count.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `max_tokens` | `int` | `target_model`'s context window × 0.9 if `target_model` is known (see [Model Context Windows](#model-context-windows)), else `4096` | Total token budget, including pinned content. Raises `ValueError` if ≤ 0. |
| `tokenizer` | `Callable[[str], int] \| None` | `default_tokenizer(target_model)` | Custom token counter; see [Token Estimation](#token-estimation). |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |

**Behavior:** splits pinned vs. rest, computes `budget = max_tokens - tokens(pinned)` (floored at 0), groups `rest` into turns, and greedily accumulates whole turns from the most recent backwards until the next turn would exceed `budget`. Never splits a turn.

**Metadata:** `original_count`, `max_tokens` (the resolved value), `estimated_tokens_used`, `dropped_count`.

**When to use:** context-window management when migrating to a model with a smaller (or larger) window than the source — the budget adapts to the target automatically via `target_model`.

### `summarize`

**Category:** Transformation.

Compresses dropped messages into one synthetic summary message. When used alone via `strategy="summarize"`, summarizes all non-pinned messages. When composed with a selection strategy via `strategies=`, summarizes only the messages the selection step dropped.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `generate` | `Callable[[list[dict]], str] \| None` | `None` | If omitted, falls back to a truncation notice (`"[Prior conversation (N messages) omitted for brevity]"`) instead of an actual summary — no model call is made. |
| `generate_kwargs` | `dict \| None` | `None` | Forwarded to `generate`. |
| `max_summary_tokens` | `int \| None` | `None` | If set and the summary exceeds it, the summary text is hard-truncated (character-based, ~4 chars/token) and suffixed with `" [truncated]"`. |
| `summarizer` | `Summarizer \| None` | `None` | Preferred over a bare `generate` for cost/latency accounting. See [Summarizers](#summarizers). |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |

**Behavior (standalone — `strategy="summarize"`):** splits pinned vs. rest. Summarizes all of `rest` via `generate`, or falls back to a truncation notice if `generate` is `None`. The summary is emitted as a single synthetic `USER` message (`llmigrate["llmigrate_synthetic"] = True`). Output is `pinned + [summary]`.

**Behavior (composed — `strategies=["keep_last", "summarize"]`):** receives the `dropped` messages from the selection step. If nothing was dropped, the transformation step is skipped entirely (no model call). Otherwise summarizes the dropped messages. Output is `pinned + [summary] + kept`.

**Metadata:** `original_count` (standalone only), `summarized` (`bool`), `summarized_count` (messages fed to the summarizer), `latency_ms` (only if `generate`/`summarizer` was called), `summary_truncated` (only if truncation occurred). When a `Summarizer` is used: additionally `cost`, `model`, `token_usage` (if the `Summarizer` reports them).

**When to use:** cost optimization handoffs where a cheaper model only needs the gist of the earlier conversation. Compose with a selection strategy to keep recent turns verbatim: `strategies=["keep_last", "summarize"]`.

### `structured_state`

**Category:** Transformation.

Extracts structured state (objective, progress, key facts, ...) instead of a prose summary — designed for agent-to-agent handoff rather than continuing a chat.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `generate` | `Callable[[list[dict]], str] \| None` | `None` | If omitted, falls back to a rule-based (non-model) heuristic extraction — not a stub; see below. |
| `generate_kwargs` | `dict \| None` | `None` | Forwarded to `generate`. |
| `schema` | `dict[str, str]` | `DEFAULT_STATE_SCHEMA` (see below) | Maps field name → a natural-language description of what that field should contain. Field order is preserved in the output. |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |

Default schema (`llmigrate.strategies.structured_state.DEFAULT_STATE_SCHEMA`):

```python
{
    "objective": "The main goal or task being worked on",
    "completed": "What has been accomplished so far",
    "observations": "Key facts, findings, or constraints discovered",
    "open_questions": "Unresolved questions or blockers",
    "next_steps": "What should happen next",
}
```

**Behavior:** splits pinned vs. rest. With `generate`, prompts it to produce a response using `## <field name>` headings matching `schema`'s keys, on the non-pinned messages. Without `generate`, runs a rule-based heuristic instead: `objective` = the pinned task (or first user message); `completed` = a digest of all-but-the-last assistant message; `observations` = the last assistant message (truncated); `open_questions` = the last user message if it looks like a question; `next_steps` = a fixed placeholder. Either way, the response text is parsed back into `metadata["state_data"]` (`dict[str, str]`) by matching `## <field>` headings — the same parser handles both the model and heuristic paths since both use the heading format. The structured state is emitted as a single synthetic `USER` message. Output is `pinned + [state_msg]` — **all other non-pinned history is dropped**. Compose with a selection strategy to keep a verbatim tail.

**Metadata:** `original_count` (standalone only), `schema_fields` (list of keys), `state_data` (parsed `dict[str, str]`), `latency_ms` (only if `generate` was called).

**When to use:** multi-agent handoff, e.g. a research agent passing structured findings to a writing agent, where the receiving agent needs organized state rather than a linear transcript.

### `audit`

**Category:** Validation.

Appends a verification instruction for the receiving model, without removing or rewriting any existing content.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `instruction` | `str` | `DEFAULT_AUDIT_INSTRUCTION_TEMPLATE` interpolated with `source_model` | Full override of the appended instruction text. |
| `source_model` | `str \| None` | `None` | When set (and `instruction` is not given), interpolated into the default instruction's source-model clause. |
| `target_model` | `str \| None` | `None` | Recorded in metadata only; does not affect the instruction text. |

Default instruction template:

```
You are continuing a conversation{source_clause}. Before proceeding, review the
conversation history above and verify that the assumptions, facts, and
reasoning are sound. If you find any issues, flag them before continuing.
Then proceed with addressing the user's request.
```

where `{source_clause}` is `" that was started with a different model ({source_model})"` if `source_model` is given, else `" that was started with a different model"`.

**Behavior:** appends one synthetic `USER` message (`llmigrate = {"llmigrate_synthetic": True, "audit_instruction": True}`) to the (unmodified) input. Does not call `split_pinned()` — nothing is dropped or rewritten, so there is nothing to protect.

**Metadata:** `original_count` (standalone only), `source_model` (if given), `target_model` (if given).

**When to use:** model fallback/failover, where you want the new model to double-check assumptions made by the model that started the conversation before continuing — e.g. after a provider outage forces an unplanned mid-conversation switch.

### `selective_history`

**Category:** Selection.

Chooses a **verbatim** subset of events that fits a token budget, ranked by a pluggable [`Selector`](#selectors) — never rewrites content, unlike `summarize`/`structured_state`.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `budget` | `int` | required | Token budget, including pinned content. Raises `ValueError` if not provided or negative. |
| `selector` | `Selector` | required | Scores candidate (non-pinned) messages; higher score = retained first. Raises `ValueError` if not provided. See [Selectors](#selectors). |
| `always_keep` | `set[str] \| None` | `None` | Set of categories (see [`category_of`](#selectors)) that are mandatorily retained ahead of `selector`'s ranking. If the mandatory set alone exceeds the remaining budget after pinning, the most-recent-first subset that fits is kept. |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |
| `tokenizer` | `Callable[[str], int] \| None` | `default_tokenizer(target_model)` | See [Token Estimation](#token-estimation). |
| `target_model` | `str \| None` | `None` | Used only to select a default tokenizer. |

**Behavior:**
1. Splits pinned vs. rest via `split_pinned()`; `remaining_budget = budget - tokens(pinned)`.
2. If `always_keep` is set, partitions `rest` into `forced` (matching categories) and `candidates` (the remainder). `forced` is then trimmed to fit `remaining_budget`, most-recent-first, if it doesn't fit whole.
3. Scores `candidates` with `selector.score(candidates)`, ranks by score descending, and greedily selects into the leftover budget (an item that doesn't fit is skipped, not swapped for a smaller one later in the ranking).
4. Recombines `forced + selected`, restoring original chronological order within `rest`.

**Limitation:** operates at message granularity — it does not guarantee `TOOL_CALL`/`TOOL_RESULT` pairing the way turn-based strategies do. Put both categories in `always_keep` if pairing matters.

**Metadata:** `original_count` (standalone only), `selected_event_ids` / `dropped_event_ids` (0-indexed positions in the *input* list, sorted ascending), `original_tokens`, `transferred_tokens`, `selector` (its `repr()`), `scores` (`dict[int, float]` mapping input index → score, for scored/candidate messages only — pinned and mandatorily-forced-but-not-scored messages are absent).

**When to use:** long agentic sessions where recency isn't the right retention signal — e.g. keeping every tool result (`always_keep={"tool_result"}`) while ranking assistant chatter by a `PrioritySelector`, or ranking by embedding similarity to the current task with a `RelevanceSelector`.

## Strategy Composition

Strategies can be composed via `strategies=` to combine concerns:

```python
# Selection + transformation: keep last 2 turns, summarize the rest
migrate(messages, strategies=["keep_last", "summarize"], n=2, generate=fn)

# Selection + validation: keep within budget, append audit
migrate(messages, strategies=["token_budget", "audit"], max_tokens=4096)

# All three: select, summarize dropped, audit the result
migrate(messages, strategies=["keep_last", "summarize", "audit"], n=3, generate=fn)

# Order doesn't matter — the library sorts internally
migrate(messages, strategies=["audit", "summarize", "keep_last"], n=3, generate=fn)
```

**Execution order** is always: selection → transformation → validation, regardless of the order passed to `strategies=`.

**Pipeline execution:**
1. **Pinning**: `split_pinned(messages)` → `(pinned, rest)` — done once, centrally.
2. **Selection** (if any): `select(rest, ...)` → `SelectionResult(kept, dropped)`. If no selection strategy: `kept = []`, `dropped = rest`.
3. **Transformation** (if any): `compress(dropped, ...)` → `CompressionResult(messages)`. Skipped if `dropped` is empty (nothing to compress). If no transformation strategy: dropped content is silently discarded.
4. **Assembly**: `pinned + transformed + kept`.
5. **Validation** (if any): `validate(assembled, ...)` → `ValidationResult(messages)`.
6. **Alternation enforcement** (unless `raw`).
7. **Format conversion** to wire-format dicts.

**Metadata namespacing**: when using `strategies=`, metadata from each step is stored under `"selection"`, `"transformation"`, and `"validation"` keys. When using `strategy=` (single), metadata is flat.

**Constraints**: at most one strategy per category. `"raw"` cannot be combined with other strategies.

## Protected Content (pinning)

`src/llmigrate/pinning.py` — `split_pinned(messages, *, pin_first_user=True) -> (pinned, rest)`.

Every strategy that can drop or rewrite content calls `split_pinned()` instead of implementing its own filtering. A message is pinned if **any** of the following hold:

- its role is `SYSTEM`;
- it carries `metadata["pinned"] = True` (works regardless of role or position — the escape hatch for protecting arbitrary messages, e.g. a prior migration's structured state);
- it is the **first** `USER` message in the list, and `pin_first_user` is `True` (the default).

Pinned messages are always emitted verbatim, in their original order, and are never truncated, summarized, or dropped by any strategy. `raw` and `audit` don't call `split_pinned()` at all, since neither drops nor rewrites content in the first place. In the pipeline (`strategies=`), pinning is done once centrally before any step executes.

Disable first-user pinning per call with `pin_first_user=False`; this is forwarded through `migrate(**params)` like any other strategy parameter.

This is the mechanism that keeps a SWE-bench-style task description alive through `keep_last(n=0)`, an aggressive `token_budget`, or a `summarize`/`structured_state` call that would otherwise paraphrase it away. It is verified by a registry-wide parametrized test (`tests/test_pinning.py::test_preserves_task_marker`) that runs every strategy under deliberately tight parameters.

## Role Alternation

`src/llmigrate/alternation.py` — `enforce_alternation(messages) -> list[Message]`, applied by `migrate()` itself (not by individual strategies) whenever `enforce_alternation=True` (the default) and the strategy is not `raw`.

Merges consecutive messages that share the same non-`SYSTEM` role into a single message: content is joined with `"\n\n"`, metadata is merged (later message's keys win on conflict), and `metadata["llmigrate_synthetic"]` is set on the merged message if either side had it set.

This exists because pinning can incidentally produce non-alternating output — e.g. a pinned first-`USER` task immediately followed by a synthetic `USER`-role summary — which providers like Anthropic reject outright. Doing this centrally, rather than in each strategy, also guarantees pinned content survives verbatim *inside* the merged message even when a model-generated summary doesn't reproduce it.

## Turn Grouping

`src/llmigrate/turns.py` — `group_into_turns(messages) -> list[list[Message]]`.

A turn starts at a `USER` message and includes everything up to (not including) the next `USER` message — so an assistant reply plus any interleaved `TOOL_CALL`/`TOOL_RESULT` messages stay together. A leading run of non-`USER` messages (if any) forms its own group. Used by `keep_last` and `token_budget` so truncation never splits a turn or orphans a `TOOL_RESULT` from its `TOOL_CALL`. Not used by `selective_history` (message-granularity by design).

## Selectors

`src/llmigrate/selectors.py` — pluggable ranking used by `selective_history`'s `selector` parameter.

```python
class Selector(Protocol):
    def score(self, messages: list[Message]) -> list[float]: ...
```

`score()` receives the *candidate* messages (non-pinned, minus anything already claimed by `always_keep`) and must return one float per message, in the same order; higher scores are retained first.

### `category_of(message) -> str`

The retention category used by `PrioritySelector` and by `selective_history`'s `always_keep`: `metadata["category"]` if the caller set one, else a default derived from role — `USER → "user_instruction"`, `ASSISTANT → "assistant"`, `TOOL_CALL → "tool_call"`, `TOOL_RESULT → "tool_result"`, `SYSTEM → "system"`.

### `PrioritySelector`

```python
PrioritySelector(priorities: dict[str, float], recency_tiebreak: bool = True)
```

Deterministic, category-based scoring: each message scores `priorities.get(category_of(msg), 0.0)`. Categories absent from `priorities` score `0`. If `recency_tiebreak` (default `True`), a small index-scaled epsilon (`(i / n) * 1e-6`) is added so more recent messages outrank equal-priority older ones without changing cross-category ordering.

```python
from llmigrate.selectors import PrioritySelector

selector = PrioritySelector(priorities={"user_instruction": 100, "tool_result": 80, "assistant": 20})
```

### `RelevanceSelector`

```python
RelevanceSelector(
    embed: Callable[[list[str]], list[list[float]]],
    query: str | Callable[[list[Message]], str] | None = None,
)
```

Scores by cosine similarity to a query embedding. `embed` takes a list of strings and returns one equal-length vector per string — llmigrate binds to no specific embedding provider, so any function (OpenAI embeddings, a local sentence-transformer, etc.) can be plugged in. `query`:
- a `str` — used as-is;
- a callable `(messages) -> str` — resolved against the candidate list at scoring time;
- `None` (default) — resolved to `messages[0].content` (the first candidate message).

```python
from llmigrate.selectors import RelevanceSelector

selector = RelevanceSelector(embed=my_embed_fn, query="fix the failing test")
```

## Summarizers

`src/llmigrate/summarizers.py` — pluggable summarization with cost/latency accounting, used by `summarize`'s `summarizer` parameter.

```python
@dataclass
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None

@dataclass
class SummarizerResult:
    text: str
    latency_ms: float | None = None
    token_usage: TokenUsage | None = None
    cost: float | None = None
    model: str | None = None

class Summarizer(Protocol):
    def summarize(self, messages: list[dict[str, Any]]) -> SummarizerResult: ...
```

Implement `Summarizer` directly against your provider client (e.g. wrapping the OpenAI SDK's usage-reporting response) to get real `cost`/`model` reporting — a bare `generate` callable can't self-report either.

### `GenerateSummarizer`

```python
GenerateSummarizer(generate: Callable[..., str], generate_kwargs: dict[str, Any] | None = None)
```

Wraps a plain `generate` callable (sync or async, dispatched via `_util.call_generate`) as a `Summarizer`. Measures wall-clock `latency_ms` and estimates `token_usage` via the char-based heuristic (see [Token Estimation](#token-estimation)) regardless of whether `tiktoken` is installed; `cost` and `model` are always `None`.

### `as_summarizer(summarizer, generate_kwargs=None) -> Summarizer`

Returns `summarizer` unchanged if it already has a `.summarize()` method, otherwise wraps it (a bare `generate` callable) in `GenerateSummarizer`. Raises `ValueError` if `summarizer` is `None`.

## Token Estimation

`src/llmigrate/tokens.py`.

```python
default_tokenizer(target_model: str | None = None) -> Callable[[str], int]
estimate_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int
```

`default_tokenizer()` returns a `tiktoken`-backed counter if the optional `tiktoken` package is installed (using `tiktoken.encoding_for_model(target_model)` when that model name is recognized, else `cl100k_base`), otherwise a character heuristic (`max(1, len(text) // 4)`, i.e. ~4 characters per token). `estimate_tokens()` applies a given tokenizer (or the heuristic, if none given) to a string. Any strategy accepting a `tokenizer` parameter takes a plain `Callable[[str], int]`, so a custom tokenizer (e.g. a provider-specific one) can always be substituted.

## Model Context Windows

`src/llmigrate/models.py`.

```python
context_window_for(model: str | None) -> int | None
register_model_context_window(name_or_prefix: str, tokens: int) -> None
```

`context_window_for()` does a best-effort prefix match against a built-in table of common models (GPT-4/4o/4.1, o1/o3, Claude 3/3.5/3.7/4 families, Gemini 1.5/2.0, Llama 3/3.1, Mixtral) plus any names registered via `register_model_context_window()` (checked first, so custom registrations can override the built-in table). Returns `None` for unrecognized names — including self-hosted/custom checkpoint names served via vLLM/LocalAI — in which case `token_budget` falls back to `4096` tokens unless `max_tokens` is given explicitly.

This table is a best-effort convenience, not a source of truth, and may go stale as providers change their lineups. Register your own model explicitly rather than relying on prefix-matching luck:

```python
import llmigrate
llmigrate.register_model_context_window("my-finetuned-llama", 32_768)
```

`token_budget` uses 90% of the looked-up window as its default `max_tokens` (leaving headroom for the target model's own response).

## `generate()` Callables

Model-assisted strategies (`summarize`, `structured_state`) accept a `generate` callable with signature `Callable[[list[dict]], str]` (sync or async — auto-detected via `inspect.iscoroutinefunction` and dispatched by `_util.call_generate`). This is the mechanism that keeps llmigrate decoupled from any specific provider SDK: `generate` receives the OpenAI-style message list llmigrate builds internally for the summarization/extraction prompt, and returns the completion text.

```python
result = llmigrate.migrate(
    messages,
    strategy="summarize",
    generate=my_generate_fn,       # Callable[[list[dict]], str], sync or async
    generate_kwargs={"temperature": 0.2},
)
```

Calling an async `generate` from inside an already-running event loop raises `RuntimeError` (`asyncio.run` cannot be nested) — call `migrate()` from synchronous code, or drive it via `asyncio.to_thread`/an executor if you're already inside an event loop.

### `generators.py` — OpenAI-compatible builders

`src/llmigrate/generators.py` provides ready-made `generate` callables for OpenAI itself and any OpenAI-compatible Chat Completions endpoint (vLLM, LocalAI, LM Studio, Ollama's OpenAI-compat mode, Together, Groq, Fireworks, ...). Requires `pip install llmigrate[openai]` unless you supply your own `client`.

```python
openai_compatible_generate(
    base_url: str,
    model: str,
    api_key: str | None = None,
    client: Any = None,
    **default_kwargs: Any,
) -> Callable[..., str]

async_openai_compatible_generate(
    base_url: str,
    model: str,
    api_key: str | None = None,
    client: Any = None,
    **default_kwargs: Any,
) -> Callable[..., Awaitable[str]]
```

`default_kwargs` (e.g. `temperature`) apply to every call and are overridden per-call by `migrate()`'s `generate_kwargs`. Pass `client` to inject an already-configured (or fake/test) `OpenAI`/`AsyncOpenAI`-compatible client instead of constructing one from `base_url`/`api_key`.

```python
from llmigrate.generators import openai_compatible_generate

generate = openai_compatible_generate(
    base_url="http://localhost:8000/v1",  # e.g. a local vLLM server
    model="meta-llama/Llama-3-70b-instruct",
)
result = llmigrate.migrate(messages, strategy="summarize", generate=generate)
```

## Format Adapters

`src/llmigrate/adapters/` — convert between provider-native formats and canonical `Message`s. `migrate()` calls `auto_convert()` on input and converts back to wire format on output via `to_openai()` or `to_anthropic()`, controlled by `target_format`.

### Auto-detection (`adapters/detect.py`)

`auto_convert(messages) -> list[Message]` dispatches based on the first element:
- `Message` → returned as-is (all elements are assumed to already be canonical).
- `dict` whose `content` is a list containing a dict with `"type"` in `{"text", "tool_use", "tool_result", "image"}` → treated as Anthropic format (`from_anthropic`).
- any other `dict` → treated as OpenAI format (`from_openai`), which also covers OpenAI-compatible self-hosted servers, since they speak the same wire format.
- anything else → `TypeError`.

Detection only inspects the *first* message, so a single call must use a consistent format throughout.

### OpenAI (`adapters/openai.py`)

`from_openai(messages: list[dict]) -> list[Message]` / `to_openai(messages: list[Message]) -> list[dict]`.

Role mapping: `system`/`developer → SYSTEM`, `user → USER`, `assistant → ASSISTANT`, `tool`/`function → TOOL_RESULT`; a message with `tool_calls` present is reclassified as `TOOL_CALL` regardless of its stated role. `tool_calls`, `tool_call_id`, and `name` are round-tripped via `metadata`. `content: None` is normalized to `""`. An unrecognized `role` string raises `ValueError`.

### Anthropic (`adapters/anthropic.py`)

`from_anthropic(messages: list[dict], system: str | None = None) -> list[Message]` / `to_anthropic(messages: list[Message]) -> dict` (returns `{"system": str | None, "messages": [...]}`).

`system`, if given, is prepended as a `SYSTEM` message. String content maps directly to `USER`/`ASSISTANT`. Content-block lists are inspected: blocks with `type == "tool_use"` classify the message as `TOOL_CALL` (converted into OpenAI-shaped `tool_calls` metadata for canonical storage); blocks with `type == "tool_result"` classify it as `TOOL_RESULT` (`tool_call_id` taken from `tool_use_id`); otherwise text blocks are concatenated and the role falls back to `USER`/`ASSISTANT`. The original block list is preserved in `metadata["anthropic_content"]` so `to_anthropic()` can round-trip it exactly rather than reconstructing lossily.

## Errors & Validation

Summary of every validation error surfaced by `migrate()` or a strategy's `transform()`:

| Condition | Exception |
|---|---|
| `messages` is not a `list` | `TypeError` |
| a `messages` element is neither `dict` nor `Message` | `TypeError` |
| a `dict` element has no `"role"` key | `ValueError` |
| a message's `content` is not a `str` | `ValueError` |
| unrecognized OpenAI `role` string | `ValueError` (from `from_openai`) |
| unsupported first-element type during format detection | `TypeError` (from `auto_convert`) |
| unknown `strategy` name | `ValueError`, message lists available strategies |
| unknown strategy name in `strategies` list | `ValueError` |
| both `strategy` and `strategies` specified | `ValueError` |
| more than one strategy from the same category in `strategies` | `ValueError` |
| `"raw"` combined with other strategies in `strategies` | `ValueError` |
| unknown `target_format` | `ValueError` |
| `keep_last`: `n < 0` | `ValueError` |
| `token_budget`: `max_tokens <= 0` | `ValueError` |
| `selective_history`: missing `budget` or `budget < 0` | `ValueError` |
| `selective_history`: missing `selector` | `ValueError` |
| async `generate` called from within a running event loop | `RuntimeError` |

## Adding a New Strategy

1. Create a new module in `src/llmigrate/strategies/` with:
   - A `transform(messages: list[Message], **params) -> MigrationResult` function for standalone use.
   - A category-specific function: `select()` for selection, `compress()` for transformation, or `validate()` for validation.
2. If the strategy can drop or rewrite content, call `pinning.split_pinned()` at the top of `transform()` and never touch the pinned half. The pipeline handles pinning centrally for the category-specific functions.
3. Register it in `strategies/__init__.py`: add to `STRATEGY_REGISTRY`, `STRATEGY_CATEGORIES`, and the appropriate category registry (`SELECT_REGISTRY`, `COMPRESS_REGISTRY`, or `VALIDATE_REGISTRY`).
4. Add a `Strategy` enum value in `types.py`.
5. Add tests, including a `TIGHT_PARAMS` entry in `tests/test_pinning.py` so the registry-wide invariant tests cover it.
