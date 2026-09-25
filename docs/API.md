# llmigrate API Reference

This document is the reference for llmigrate's `migrate()` and `async_migrate()` entry points, strategies, core types, and supporting abstractions (pinning, alternation, selectors, summarizers, token estimation, model context windows, `generate()` builders, and format adapters). For the pitch and a quick start, see the [README](../README.md).

## Table of Contents

- [`migrate()`](#migrate)
- [`async_migrate()`](#async_migrate)
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
- [Format Adapters](API-details.md#format-adapters)
- [Errors & Validation](API-details.md#errors--validation)
- [Adding a New Strategy](API-details.md#adding-a-new-strategy)

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
    input_format: str | None = None,
    target_format: str | None = None,
    system: str | None = None,
    enforce_alternation: bool = True,
    **params: Any,
) -> MigrationResult
```

The synchronous entry point for every migration. `migrate()`:

1. Validates `messages` (must be a `list`; elements are provider dictionaries or canonical `Message` objects).
2. Auto-detects the input format (OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, Gemini Interactions, or canonical `Message`) unless `input_format` is provided, then converts to canonical `Message` objects — see [Format Adapters](API-details.md#format-adapters).
3. Converts supported text content to the canonical string representation. Unsupported item kinds and content blocks raise `ValueError`; see [Supported Data](API-details.md#supported-data-and-conversion-limits).
4. Routes to the appropriate execution path:
   - `strategy="X"` (single): looks up `X` in the strategy registry and calls its `transform()` function directly, producing flat metadata.
   - `strategies=[...]` (pipeline): validates the composition (at most one per category), then executes the pipeline in canonical order: selection → transformation → validation. Metadata is namespaced by category.
   - Neither specified (or `strategy="raw"` / `strategies=[]`): identity transform.
5. Records `source_model`/`target_model` into `result.metadata` if not already set by the strategy.
6. Unless the strategy is `raw`, enforces role alternation where the target requires it while preserving tool IDs and provider content blocks — see [Role Alternation](#role-alternation). Controlled by `enforce_alternation`.
7. Converts canonical `Message` objects to the selected provider's wire-format items. Top-level system instructions are returned separately where required.

### `async_migrate()`

`async_migrate()` has the same arguments and result as `migrate()`, and is the entry point for code already running in an event loop, including async web servers, agent runtimes, and notebooks. It awaits async `generate` callables and async summarizers directly. Synchronous generators and summarizers run in a worker thread so they do not block the event loop.

```python
result = await llmigrate.async_migrate(
    messages,
    strategy="summarize",
    generate=async_generate,
)
```

Use `migrate()` from synchronous code. If its model-assisted strategy receives an async callback, it runs that callback with `asyncio.run`; calling this path from a thread that already has a running event loop raises `RuntimeError`. In that case, call `async_migrate()` instead.

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[dict] \| list[Message]` | required | Conversation history in one supported provider format or canonical `Message` objects — auto-detected; provider formats cannot be mixed, and dictionaries cannot be mixed with `Message` objects. |
| `strategy` | `str \| None` | `None` | Single strategy name — `"raw"`, `"keep_last"`, `"token_budget"`, `"summarize"`, `"structured_state"`, `"audit"`, `"selective_history"`. Sugar for `strategies=["X"]` but with flat (not namespaced) metadata. Mutually exclusive with `strategies`. |
| `strategies` | `list[str] \| None` | `None` | List of strategy names to compose, in any order. At most one from each category (selection, transformation, validation). The library sorts internally: selection → transformation → validation. Mutually exclusive with `strategy`. |
| `generate` | `Callable[[list[dict]], str] \| None` | `None` | Model call used by `summarize` and `structured_state`. May be sync or async. `migrate()` drives async callbacks with `asyncio.run` when no event loop is active; use `async_migrate()` from async code. Forwarded into `params["generate"]`. |
| `generate_kwargs` | `dict \| None` | `None` | Extra keyword arguments (e.g. `{"temperature": 0.2}`) passed to `generate` (or to the auto-wrapped `Summarizer`) on every call. Kept as a separate namespace from strategy params so the two can never collide — e.g. a strategy param named `temperature` would otherwise be ambiguous. |
| `source_model` | `str \| None` | `None` | Name of the model the conversation started on. Recorded in `result.metadata["source_model"]`; interpolated into `audit`'s default instruction. |
| `target_model` | `str \| None` | `None` | Name of the model the conversation is moving to. Recorded in `result.metadata["target_model"]`; used by `token_budget` and `selective_history` to size a default token budget and to pick a model-appropriate tokenizer — see [Model Context Windows](#model-context-windows) and [Token Estimation](#token-estimation). |
| `input_format` | `str \| None` | `None` | Explicit input format: `"openai"`, `"openai_responses"`, `"anthropic"`, `"gemini_interactions"`, or `"canonical"`. Useful when the message shape is ambiguous. Canonical input still requires `Message` objects. |
| `target_format` | `str \| None` | `None` | `"openai"`, `"openai_responses"`, `"anthropic"`, or `"gemini_interactions"`. Controls the output wire format. When omitted, defaults to the detected input format (or `"openai"` for canonical `Message` objects). |
| `system` | `str \| None` | `None` | Optional top-level system instruction. It is incorporated before strategies run. Output is in `MigrationResult.system` for Anthropic, OpenAI Responses (`instructions`), and Gemini Interactions (`system_instruction`). Do not pass a prompt already present in `messages` a second time. |
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
- `ValueError` — a provider dictionary is missing its required role/type fields, or its content cannot be represented as supported text.
- `ValueError` — unknown `target_format`.
- `ValueError` — unknown `input_format`, or an explicit input format incompatible with the supplied message representation.
- `TypeError` — `messages` mixes provider dictionaries and `Message` objects.

## Core Types

### `Message`

```python
@dataclass
class Message:
    role: Role
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

The canonical, provider-agnostic message representation. `to_dict()` / `from_dict()` round-trip through `{"role": ..., "content": ..., "metadata": ...}` (metadata omitted when empty). This type is used internally; `migrate()` and `async_migrate()` return provider-format items, not canonical `Message` objects.

Metadata keys used across the library:

| Key | Set by | Meaning |
|---|---|---|
| `pinned` | caller | Forces this message to be treated as protected content by `split_pinned()`, regardless of role or position. |
| `category` | caller | Overrides the default role-derived category used by `selective_history`'s selectors (see [Selectors](#selectors)). |
| `llmigrate_synthetic` | library | Set on any message the library generates (summaries, structured state extractions, audit instructions). Also propagated onto a merged message by `enforce_alternation` if either half was synthetic. |
| `tool_calls`, `tool_call_id`, `name` | OpenAI and tool-capable adapters | Canonical tool-call/result fields used across formats. |
| `anthropic_content` | Anthropic adapter | Preserved Anthropic content-block payloads for safe same-format conversion. |
| `responses_item` and related keys | OpenAI Responses adapter | Original Responses item snapshots used to preserve unchanged provider fields. |
| `gemini_interactions_step` and related keys | Gemini Interactions adapter | Original Interactions step snapshots used to preserve unchanged provider fields. |

### `Role`

```python
class Role(Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
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

- `.messages` — the transformed conversation in the selected provider format, including llmigrate's per-item metadata sidecar for inspection. Use `.provider_messages` when sending the items to a provider API.
- `.provider_messages` — wire-format messages, Responses items, or Interactions steps with the `"llmigrate"` sidecar removed. OpenAI Chat Completions includes system/developer messages in this list. For Anthropic, Responses, and Gemini Interactions, pass `.system` separately using the provider's corresponding system argument.
- `.strategies` — list of `Strategy` enums that were applied.
- `.metadata` — transformation details. When using `strategy=` (single), metadata is flat. When using `strategies=` (pipeline), metadata from each step is namespaced under `"selection"`, `"transformation"`, and `"validation"` keys.
- `.format` — the output format: `"openai"`, `"openai_responses"`, `"anthropic"`, or `"gemini_interactions"`.
- `.system` — the top-level system instruction for Anthropic, OpenAI Responses, and Gemini Interactions; `None` for OpenAI Chat Completions.
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

**Behavior:** splits pinned vs. rest, computes `budget = max_tokens - estimated_tokens(pinned)` (floored at 0), groups `rest` into turns, and greedily accumulates whole turns from the most recent backwards until the next turn would exceed `budget`. Estimates include message roles, text, tool names/arguments and IDs, small framing overhead, and markers for media blocks. Never splits a turn.

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
1. Splits pinned vs. rest via `split_pinned()`; `remaining_budget = budget - estimated_tokens(pinned)`.
2. Groups messages connected by tool-call IDs into indivisible events; other messages are single-message events.
3. If `always_keep` is set, any event containing a message in a forced category is forced as a whole. If the forced events do not fit, the most recent events that fit are kept.
4. Scores candidate messages with `selector.score(candidates)`. Each event receives its highest member score; events are ranked by that score and greedily selected into the remaining budget (an event that does not fit is skipped, not swapped for smaller events later in the ranking).
5. Recombines forced and selected events, restoring original chronological order within `rest`.

**Tool-event safety:** a function call and its known result(s) are kept or dropped together, including parallel calls represented in one message. This prevents `selective_history` from knowingly emitting an orphaned call or result. Results without a matching call remain standalone events because the missing call is not present in the input.

**Metadata:** `original_count` (standalone only), `selected_event_ids` / `dropped_event_ids` (0-indexed message positions in the *input* list, sorted ascending; members of a tool event appear together), `original_tokens`, `transferred_tokens`, `selector` (its `repr()`), `scores` (`dict[int, float]` mapping input index → score, for scored/candidate messages only — pinned and mandatorily-forced-but-not-scored messages are absent).

**When to use:** long agentic sessions where recency isn't the right retention signal — e.g. forcing events that contain tool results with `always_keep={"tool_result"}` while ranking the remaining events by a `PrioritySelector`, or ranking by embedding similarity to the current task with a `RelevanceSelector`.

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

- its role is `SYSTEM` or `DEVELOPER`;
- it carries `metadata["pinned"] = True` (works regardless of role or position — the escape hatch for protecting arbitrary messages, e.g. a prior migration's structured state);
- it is the **first** `USER` message in the list, and `pin_first_user` is `True` (the default).

Pinned messages are always emitted verbatim, in their original order, and are never truncated, summarized, or dropped by any strategy. `raw` and `audit` don't call `split_pinned()` at all, since neither drops nor rewrites content in the first place. In the pipeline (`strategies=`), pinning is done once centrally before any step executes.

Disable first-user pinning per call with `pin_first_user=False`; this is forwarded through `migrate(**params)` like any other strategy parameter.

This is the mechanism that keeps a SWE-bench-style task description alive through `keep_last(n=0)`, an aggressive `token_budget`, or a `summarize`/`structured_state` call that would otherwise paraphrase it away. It is verified by a registry-wide parametrized test (`tests/test_pinning.py::test_preserves_task_marker`) that runs every strategy under deliberately tight parameters.

## Role Alternation

`src/llmigrate/alternation.py` — `enforce_alternation(messages) -> list[Message]`, applied by `migrate()` itself (not by individual strategies) whenever `enforce_alternation=True` (the default) and the strategy is not `raw`.

Merges consecutive plain-text messages that share the same role, except `SYSTEM`, `DEVELOPER`, `TOOL_CALL`, and `TOOL_RESULT`: content is joined with `"\n\n"`, metadata is merged (later message's keys win on conflict), and `metadata["llmigrate_synthetic"]` is set on the merged message if either side had it set. Provider content-block messages are left intact so adapters can preserve their blocks and tool IDs while formatting the target request.

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

class AsyncSummarizer(Protocol):
    async def summarize(self, messages: list[dict[str, Any]]) -> SummarizerResult: ...
```

Implement `Summarizer` or `AsyncSummarizer` directly against your provider client (for example, wrapping its usage-reporting response) to report actual `cost`, `model`, and token usage. An async summarizer is awaited by `async_migrate()`; the sync API accepts a synchronous `Summarizer`.

### `GenerateSummarizer`

```python
GenerateSummarizer(generate: Callable[..., str], generate_kwargs: dict[str, Any] | None = None)
```

Wraps a plain `generate` callable as a `Summarizer`. The synchronous API dispatches it with `_util.call_generate`; `async_migrate()` uses the async path. It measures wall-clock `latency_ms` and estimates `token_usage` via the char-based heuristic (see [Token Estimation](#token-estimation)) regardless of whether `tiktoken` is installed; `cost` and `model` are always `None`.

### `as_summarizer(summarizer, generate_kwargs=None) -> Summarizer`

Returns `summarizer` unchanged if it already has a `.summarize()` method, otherwise wraps it (a bare `generate` callable) in `GenerateSummarizer`. Raises `ValueError` if `summarizer` is `None`.

## Token Estimation

`src/llmigrate/tokens.py`.

```python
default_tokenizer(target_model: str | None = None) -> Callable[[str], int]
estimate_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int
```

`default_tokenizer()` returns a `tiktoken`-backed counter if the optional `tiktoken` package is installed (using `tiktoken.encoding_for_model(target_model)` when that model name is recognized, else `cl100k_base`), otherwise a character heuristic (`max(1, len(text) // 4)`, i.e. ~4 characters per token). `estimate_tokens()` applies a given tokenizer (or the heuristic, if none given) to a string. Budget strategies apply that counter to message text and tool payloads, with a small framing allowance and markers for media blocks. Image/audio/document token costs and provider-specific message/tool schemas are not knowable from text alone, so these remain estimates; leave additional room or supply a provider-aware custom tokenizer where needed. Any strategy accepting a `tokenizer` parameter takes a plain `Callable[[str], int]`, so a custom tokenizer (e.g. a provider-specific one) can always be substituted.

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

Model-assisted strategies (`summarize`, `structured_state`) accept a `generate` callable with signature `Callable[[list[dict]], str]` (sync or async). `migrate()` is for synchronous callers and runs async callbacks only when it can create an event loop. `async_migrate()` awaits async callbacks directly and sends synchronous callbacks to a worker thread. This keeps llmigrate decoupled from provider SDKs: `generate` receives the OpenAI-style message list built for the summarization/extraction prompt and returns completion text.

```python
result = llmigrate.migrate(
    messages,
    strategy="summarize",
    generate=my_generate_fn,       # synchronous callable, or async outside a running loop
    generate_kwargs={"temperature": 0.2},
)
```

Inside an async agent or server, use `await llmigrate.async_migrate(...)` with an async generator or summarizer. Calling `migrate()` with an async callback from within an already-running event loop raises `RuntimeError`; the sync entry point cannot nest event loops.

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

Provider adapter details, validation errors, and extension guidance are in the [extended API reference](API-details.md).
