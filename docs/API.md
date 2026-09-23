# llmigrate API Reference

This document is the rigorous reference for llmigrate: the `transfer()` entry point, every strategy and its parameters, the core types, and the supporting abstractions (pinning, alternation, selectors, summarizers, token estimation, model context windows, `generate()` builders, and format adapters). For the pitch and a quick start, see the [README](../README.md).

## Table of Contents

- [`transfer()`](#transfer)
- [Core Types](#core-types)
- [Strategies](#strategies)
  - [`raw`](#raw)
  - [`keep_last`](#keep_last)
  - [`token_budget`](#token_budget)
  - [`summarize`](#summarize)
  - [`capsule`](#capsule)
  - [`audit`](#audit)
  - [`selective_history`](#selective_history)
  - [`summary_tail`](#summary_tail)
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

## `transfer()`

```python
llmigrate.transfer(
    messages: list[dict] | list[Message],
    strategy: str = "raw",
    *,
    generate: Callable[..., str] | None = None,
    generate_kwargs: dict[str, Any] | None = None,
    source_model: str | None = None,
    target_model: str | None = None,
    enforce_alternation: bool = True,
    **params: Any,
) -> TransferResult
```

The single entry point for every migration. `transfer()`:

1. Validates `messages` (must be a `list`; every element must be a `dict` with a `"role"` key, or a `Message`).
2. Auto-detects the input format (OpenAI dict, Anthropic dict, or canonical `Message`) and converts to canonical `Message` objects — see [Format Adapters](#format-adapters).
3. Validates that every message's `content` is a `str`.
4. Looks up `strategy` in the strategy registry and calls its `transform(messages, **params)` function, forwarding `generate`, `generate_kwargs`, `source_model`, and `target_model` into `params` when given.
5. Records `source_model`/`target_model` into `result.metadata` if not already set by the strategy.
6. Unless `strategy == "raw"` and unless `enforce_alternation=False`, merges consecutive same-role messages in the output — see [Role Alternation](#role-alternation).

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `messages` | `list[dict] \| list[Message]` | required | Conversation history. OpenAI-format dicts, Anthropic-format dicts, or canonical `Message` objects — auto-detected, may not be mixed within a single call except uniformly as `Message`. |
| `strategy` | `str` | `"raw"` | One of `"raw"`, `"keep_last"`, `"token_budget"`, `"summarize"`, `"capsule"`, `"audit"`, `"selective_history"`, `"summary_tail"`. Unknown names raise `ValueError` listing the available strategies. |
| `generate` | `Callable[[list[dict]], str] \| None` | `None` | Model call used by `summarize`, `capsule`, and (as a fallback for `summarizer`) `summary_tail`. May be a sync or async callable — async is auto-detected and awaited via `asyncio.run`. Forwarded into `params["generate"]`. |
| `generate_kwargs` | `dict \| None` | `None` | Extra keyword arguments (e.g. `{"temperature": 0.2}`) passed to `generate` (or to the auto-wrapped `Summarizer`) on every call. Kept as a separate namespace from strategy params so the two can never collide — e.g. a strategy param named `temperature` would otherwise be ambiguous. |
| `source_model` | `str \| None` | `None` | Name of the model the conversation started on. Recorded in `result.metadata["source_model"]`; interpolated into `audit`'s default instruction. |
| `target_model` | `str \| None` | `None` | Name of the model the conversation is moving to. Recorded in `result.metadata["target_model"]`; used by `token_budget`, `summary_tail`, and `selective_history` to size a default token budget and to pick a model-appropriate tokenizer — see [Model Context Windows](#model-context-windows) and [Token Estimation](#token-estimation). |
| `enforce_alternation` | `bool` | `True` | Merge consecutive same-role messages in the result. Always skipped for `strategy="raw"` regardless of this flag, since `raw`'s contract is "unchanged". |
| `**params` | `Any` | — | Strategy-specific parameters. See each strategy's table below. Every strategy that can drop or rewrite content also accepts `pin_first_user: bool = True` (see [Protected Content](#protected-content-pinning)). |

### Returns

A `TransferResult` (see [Core Types](#core-types)).

### Raises

- `ValueError` — unknown `strategy` name, or a strategy-specific validation failure (see each strategy's table).
- `TypeError` — `messages` is not a `list`, or an element is neither a `dict` nor a `Message`.
- `ValueError` — a `dict` element is missing the `"role"` key, or any message's `content` is not a `str` after conversion.

## Core Types

### `Message`

```python
@dataclass
class Message:
    role: Role
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

The canonical, provider-agnostic message representation. `to_dict()` / `from_dict()` round-trip through `{"role": ..., "content": ..., "metadata": ...}` (metadata omitted when empty).

Metadata keys used across the library:

| Key | Set by | Meaning |
|---|---|---|
| `pinned` | caller | Forces this message to be treated as protected content by `split_pinned()`, regardless of role or position. |
| `category` | caller | Overrides the default role-derived category used by `selective_history`'s selectors (see [Selectors](#selectors)). |
| `llmigrate_synthetic` | library | Set on any message the library generates (summaries, capsules, audit instructions). Also propagated onto a merged message by `enforce_alternation` if either half was synthetic. |
| `segment` | `summary_tail` | `"summary_prefix"` or `"verbatim_tail"` — marks which half of a `summary_tail` output a message belongs to. |
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
    CAPSULE = "capsule"
    AUDIT = "audit"
    SELECTIVE_HISTORY = "selective_history"
    SUMMARY_TAIL = "summary_tail"
```

### `TransferResult`

```python
@dataclass
class TransferResult:
    messages: list[Message]
    strategy: Strategy
    metadata: dict[str, Any] = field(default_factory=dict)
```

- `.original_count` — `int(metadata.get("original_count", 0))`, the length of the input.
- `.transferred_count` — `len(messages)`, the length of the output.
- `.to_openai() -> list[dict]` — converts `messages` to OpenAI-format dicts.
- `.to_anthropic() -> dict` — converts `messages` to Anthropic's `{"system": str | None, "messages": [...]}` shape.

Every strategy sets `metadata["original_count"]`; strategy-specific keys are documented per strategy below.

## Strategies

Each strategy is a `transform(messages: list[Message], **params) -> TransferResult` function registered in `STRATEGY_REGISTRY` (`src/llmigrate/strategies/__init__.py`), keyed by the strings below.

### `raw`

Identity transform. Returns the input unchanged, including its exact message list — not even role-alternation merging is applied (`transfer()` special-cases `raw` out of that step regardless of `enforce_alternation`).

| Parameter | Type | Default | Notes |
|---|---|---|---|
| *(none)* | | | |

**Metadata:** `original_count`.

**When to use:** provider/protocol change only, no content transformation — e.g. moving between two deployments of the same model family where context size isn't a concern.

### `keep_last`

Turn-based truncation: keeps pinned content plus the last `n` turns, dropping everything older. A "turn" is a `USER` message plus everything up to (not including) the next `USER` message — see [Turn Grouping](#turn-grouping).

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `n` | `int` | `5` | Number of most recent turns to keep. `0` keeps only pinned content. Raises `ValueError` if negative. |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |

**Behavior:** splits pinned vs. rest via `split_pinned()`, groups `rest` into turns, keeps the last `n` turn-groups verbatim, drops the rest. Output is `pinned + kept_turns`.

**Metadata:** `original_count`, `n`, `dropped_count` (messages dropped, not turns).

**When to use:** capacity routing where a fixed number of recent exchanges is "enough" — e.g. tier-1→tier-2 escalation in a support bot, where only the last few turns are relevant to the immediate question.

### `token_budget`

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

Compresses older history into one synthetic summary message, keeping the most recent `tail` turns verbatim (unlike other strategies, `tail` here counts individual messages taken from the end of `rest`, not turn-groups — see caveat below).

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `tail` | `int` | `3` | Number of most recent non-pinned messages to keep verbatim after the summary. If `len(rest) <= tail`, the whole input is returned unchanged and no model call is made. |
| `generate` | `Callable[[list[dict]], str] \| None` | `None` | If omitted, falls back to a truncation notice (`"[Prior conversation (N messages) omitted for brevity]"`) instead of an actual summary — no model call is made. |
| `generate_kwargs` | `dict \| None` | `None` | Forwarded to `generate`. |
| `max_summary_tokens` | `int \| None` | `None` | If set and the summary exceeds it, the summary text is hard-truncated (character-based, ~4 chars/token) and suffixed with `" [truncated]"`. |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |

**Behavior:** splits pinned vs. rest. If `rest` already fits within `tail`, returns the input unchanged (`metadata["summarized"] = False`). Otherwise summarizes `rest[:-tail]` (or all of `rest` if `tail == 0`) via `generate`, or falls back to a truncation notice if `generate` is `None`. The summary is emitted as a single synthetic `USER` message (`metadata["llmigrate_synthetic"] = True`), followed by `rest[-tail:]` verbatim. Output is `pinned + [summary] + tail_msgs`.

**Note:** `summarize`'s `tail` operates on raw messages, not turn-groups (unlike `keep_last`/`token_budget`/`summary_tail`'s tail) — it can split a turn if `tail` falls mid-turn. Use `summary_tail` instead when turn-aligned tails matter.

**Metadata:** `original_count`, `summarized` (`bool`), `summarized_count` (messages fed to the summarizer), `tail`, `latency_ms` (only if `generate` was called), `summary_truncated` (only if truncation occurred).

**When to use:** cost optimization handoffs where a cheaper model only needs the gist of the earlier conversation plus the live thread — e.g. moving from an expensive reasoning model to a cheap Q&A model once the hard part is solved.

### `capsule`

Extracts a structured state capsule (objective, progress, key facts, ...) instead of a prose summary — designed for agent-to-agent handoff rather than continuing a chat.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `generate` | `Callable[[list[dict]], str] \| None` | `None` | If omitted, falls back to a rule-based (non-model) heuristic extraction — not a stub; see below. |
| `generate_kwargs` | `dict \| None` | `None` | Forwarded to `generate`. |
| `schema` | `dict[str, str]` | `DEFAULT_CAPSULE_SCHEMA` (see below) | Maps field name → a natural-language description of what that field should contain. Field order is preserved in the output. |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |

Default schema (`llmigrate.strategies.capsule.DEFAULT_CAPSULE_SCHEMA`):

```python
{
    "objective": "The main goal or task being worked on",
    "completed": "What has been accomplished so far",
    "observations": "Key facts, findings, or constraints discovered",
    "open_questions": "Unresolved questions or blockers",
    "next_steps": "What should happen next",
}
```

**Behavior:** splits pinned vs. rest. With `generate`, prompts it to produce a response using `## <field name>` headings matching `schema`'s keys, on the non-pinned messages. Without `generate`, runs a rule-based heuristic instead: `objective` = the pinned task (or first user message); `completed` = a digest of all-but-the-last assistant message; `observations` = the last assistant message (truncated); `open_questions` = the last user message if it looks like a question; `next_steps` = a fixed placeholder. Either way, the response text is parsed back into `metadata["capsule_data"]` (`dict[str, str]`) by matching `## <field>` headings — the same parser handles both the model and heuristic paths since both use the heading format. The capsule is emitted as a single synthetic `USER` message. Output is `pinned + [capsule_msg]` — **all other non-pinned history is dropped**, unlike `summarize`/`summary_tail`, which keep a verbatim tail.

**Metadata:** `original_count`, `schema_fields` (list of keys), `capsule_data` (parsed `dict[str, str]`), `latency_ms` (only if `generate` was called).

**When to use:** multi-agent handoff, e.g. a research agent passing structured findings to a writing agent, where the receiving agent needs organized state rather than a linear transcript.

### `audit`

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

**Behavior:** appends one synthetic `USER` message (`metadata = {"llmigrate_synthetic": True, "audit_instruction": True}`) to the (unmodified) input. Does not call `split_pinned()` — nothing is dropped or rewritten, so there is nothing to protect.

**Metadata:** `original_count`, `source_model` (if given), `target_model` (if given).

**When to use:** model fallback/failover, where you want the new model to double-check assumptions made by the model that started the conversation before continuing — e.g. after a provider outage forces an unplanned mid-conversation switch.

### `selective_history`

Chooses a **verbatim** subset of events that fits a token budget, ranked by a pluggable [`Selector`](#selectors) — never rewrites content, unlike `summarize`/`capsule`/`summary_tail`.

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

**Metadata:** `original_count`, `selected_event_ids` / `dropped_event_ids` (0-indexed positions in the *input* list, sorted ascending), `original_tokens`, `transferred_tokens`, `selector` (its `repr()`), `scores` (`dict[int, float]` mapping input index → score, for scored/candidate messages only — pinned and mandatorily-forced-but-not-scored messages are absent).

**When to use:** long agentic sessions where recency isn't the right retention signal — e.g. keeping every tool result (`always_keep={"tool_result"}`) while ranking assistant chatter by a `PrioritySelector`, or ranking by embedding similarity to the current task with a `RelevanceSelector`.

### `summary_tail`

Combines a summarized prefix with a verbatim, turn-aligned tail — the generalization of `summarize` with independently sized, budget-driven prefix/tail and pluggable cost/latency accounting.

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `total_budget` | `int` | required | Raises `ValueError` if not provided. |
| `summary_budget` | `int` | required | Token budget for the summary text. Raises `ValueError` if not provided or negative. |
| `tail_budget` | `int \| None` | `total_budget - summary_budget` | Token budget for the verbatim tail. Raises `ValueError` if negative, or if `summary_budget + tail_budget > total_budget`. |
| `summarizer` | `Summarizer \| None` | `None` | Preferred over `generate` if both given. See [Summarizers](#summarizers). |
| `generate` | `Callable[[list[dict]], str] \| None` | `None` | Used (auto-wrapped in `GenerateSummarizer`) if `summarizer` is not given. One of `summarizer`/`generate` is required *only if* the (non-pinned) history doesn't already fit `tail_budget` — see below. |
| `generate_kwargs` | `dict \| None` | `None` | Forwarded when auto-wrapping a bare `generate`. |
| `pin_first_user` | `bool` | `True` | See [Protected Content](#protected-content-pinning). |
| `tokenizer` | `Callable[[str], int] \| None` | `default_tokenizer(target_model)` | See [Token Estimation](#token-estimation). |
| `target_model` | `str \| None` | `None` | Used to select a default tokenizer (does **not** auto-size `total_budget`, unlike `token_budget`). |

**Behavior:**
1. Splits pinned vs. rest, groups `rest` into turns.
2. Greedily accumulates whole turns from the end into the tail while they fit `tail_budget`.
3. Whatever precedes the tail (`prefix_msgs`) is summarized — but **only if `prefix_msgs` is non-empty**. If the whole (non-pinned) history already fits in `tail_budget`, `prefix_msgs` is empty and **no summarizer/`generate` call is made at all** (so `summarizer`/`generate` become optional in that case).
4. If a summary is produced and exceeds `summary_budget`, it's hard-truncated (~4 chars/token) with `" [truncated]"` appended, and `metadata["summary_truncated"] = True`.
5. Tail messages are re-emitted with `metadata["segment"] = "verbatim_tail"` added (preserving existing metadata); the summary message carries `metadata["segment"] = "summary_prefix"`. Output is `pinned + [summary?] + tagged_tail`.

**Metadata:** `original_count`, `prefix_event_ids` / `tail_event_ids` (0-indexed input positions), `summary_tokens`, `tail_tokens`, `total_transferred_tokens`, `summarizer` (its `repr()`, or `None` if no summarization occurred), `summary_truncated` (only if truncated), and — only if a summarizer call was made — `latency_ms`, `token_usage` (a `TokenUsage`), `cost`, `model` (the latter two only if the `Summarizer` implementation reports them; a bare `generate` reports `None` for both).

**When to use:** context-window management with finer control than `token_budget` alone — e.g. guaranteeing a minimum verbatim tail size regardless of how much history precedes it, while still tracking summarization cost/latency for observability.

## Protected Content (pinning)

`src/llmigrate/pinning.py` — `split_pinned(messages, *, pin_first_user=True) -> (pinned, rest)`.

Every strategy that can drop or rewrite content calls `split_pinned()` instead of implementing its own filtering. A message is pinned if **any** of the following hold:

- its role is `SYSTEM`;
- it carries `metadata["pinned"] = True` (works regardless of role or position — the escape hatch for protecting arbitrary messages, e.g. a prior transfer's capsule);
- it is the **first** `USER` message in the list, and `pin_first_user` is `True` (the default).

Pinned messages are always emitted verbatim, in their original order, and are never truncated, summarized, or dropped by any strategy. `raw` and `audit` don't call `split_pinned()` at all, since neither drops nor rewrites content in the first place.

Disable first-user pinning per call with `pin_first_user=False`; this is forwarded through `transfer(**params)` like any other strategy parameter.

This is the mechanism that keeps a SWE-bench-style task description alive through `keep_last(n=0)`, an aggressive `token_budget`, or a `summarize`/`capsule` call that would otherwise paraphrase it away. It is verified by a registry-wide parametrized test (`tests/test_pinning.py::test_preserves_task_marker`) that runs every strategy under deliberately tight parameters.

## Role Alternation

`src/llmigrate/alternation.py` — `enforce_alternation(messages) -> list[Message]`, applied by `transfer()` itself (not by individual strategies) whenever `enforce_alternation=True` (the default) and `strategy != "raw"`.

Merges consecutive messages that share the same non-`SYSTEM` role into a single message: content is joined with `"\n\n"`, metadata is merged (later message's keys win on conflict), and `metadata["llmigrate_synthetic"]` is set on the merged message if either side had it set.

This exists because pinning can incidentally produce non-alternating output — e.g. a pinned first-`USER` task immediately followed by a synthetic `USER`-role summary — which providers like Anthropic reject outright. Doing this centrally, rather than in each strategy, also guarantees pinned content survives verbatim *inside* the merged message even when a model-generated summary doesn't reproduce it.

## Turn Grouping

`src/llmigrate/turns.py` — `group_into_turns(messages) -> list[list[Message]]`.

A turn starts at a `USER` message and includes everything up to (not including) the next `USER` message — so an assistant reply plus any interleaved `TOOL_CALL`/`TOOL_RESULT` messages stay together. A leading run of non-`USER` messages (if any) forms its own group. Used by `keep_last`, `token_budget`, and `summary_tail`'s tail so truncation never splits a turn or orphans a `TOOL_RESULT` from its `TOOL_CALL`. Not used by `summarize` (whose `tail` is message-count-based — see its caveat) or `selective_history` (message-granularity by design).

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

`src/llmigrate/summarizers.py` — pluggable summarization with cost/latency accounting, used by `summary_tail`'s `summarizer` parameter (and available generally — `summarize` reports `latency_ms` but does not accept a `Summarizer`, only a bare `generate`).

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

Used internally by `summary_tail`: returns `summarizer` unchanged if it already has a `.summarize()` method, otherwise wraps it (a bare `generate` callable) in `GenerateSummarizer`. Raises `ValueError` if `summarizer` is `None` (i.e. neither `summarizer` nor `generate` was supplied) — except that `summary_tail` only calls this when there's actually a prefix to summarize, so omitting both is valid whenever the tail alone covers the whole history.

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

Model-assisted strategies (`summarize`, `capsule`, `summary_tail`) accept a `generate` callable with signature `Callable[[list[dict]], str]` (sync or async — auto-detected via `inspect.iscoroutinefunction` and dispatched by `_util.call_generate`). This is the mechanism that keeps llmigrate decoupled from any specific provider SDK: `generate` receives the OpenAI-style message list llmigrate builds internally for the summarization/extraction prompt, and returns the completion text.

```python
result = llmigrate.transfer(
    messages,
    strategy="summarize",
    generate=my_generate_fn,       # Callable[[list[dict]], str], sync or async
    generate_kwargs={"temperature": 0.2},
    tail=3,
)
```

Calling an async `generate` from inside an already-running event loop raises `RuntimeError` (`asyncio.run` cannot be nested) — call `transfer()` from synchronous code, or drive it via `asyncio.to_thread`/an executor if you're already inside an event loop.

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

`default_kwargs` (e.g. `temperature`) apply to every call and are overridden per-call by `transfer()`'s `generate_kwargs`. Pass `client` to inject an already-configured (or fake/test) `OpenAI`/`AsyncOpenAI`-compatible client instead of constructing one from `base_url`/`api_key`.

```python
from llmigrate.generators import openai_compatible_generate

generate = openai_compatible_generate(
    base_url="http://localhost:8000/v1",  # e.g. a local vLLM server
    model="meta-llama/Llama-3-70b-instruct",
)
result = llmigrate.transfer(messages, strategy="summarize", generate=generate)
```

## Format Adapters

`src/llmigrate/adapters/` — convert between provider-native formats and canonical `Message`s. `transfer()` calls `auto_convert()` before dispatching to a strategy; you generally don't need to call adapters directly except via `TransferResult.to_openai()`/`.to_anthropic()`.

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

Summary of every validation error surfaced by `transfer()` or a strategy's `transform()`:

| Condition | Exception |
|---|---|
| `messages` is not a `list` | `TypeError` |
| a `messages` element is neither `dict` nor `Message` | `TypeError` |
| a `dict` element has no `"role"` key | `ValueError` |
| a message's `content` is not a `str` | `ValueError` |
| unrecognized OpenAI `role` string | `ValueError` (from `from_openai`) |
| unsupported first-element type during format detection | `TypeError` (from `auto_convert`) |
| unknown `strategy` name | `ValueError`, message lists available strategies |
| `keep_last`: `n < 0` | `ValueError` |
| `token_budget`: `max_tokens <= 0` | `ValueError` |
| `selective_history`: missing `budget` or `budget < 0` | `ValueError` |
| `selective_history`: missing `selector` | `ValueError` |
| `summary_tail`: missing `total_budget` or `summary_budget` | `ValueError` |
| `summary_tail`: `summary_budget < 0` or `tail_budget < 0` | `ValueError` |
| `summary_tail`: `summary_budget + tail_budget > total_budget` | `ValueError` |
| `summary_tail`: prefix needs summarizing but neither `summarizer` nor `generate` given | `ValueError` (from `as_summarizer`) |
| async `generate` called from within a running event loop | `RuntimeError` |

## Adding a New Strategy

1. Create a new module in `src/llmigrate/strategies/` with a `transform(messages: list[Message], **params) -> TransferResult` function.
2. If the strategy can drop or rewrite content, call `pinning.split_pinned()` at the top and never touch the pinned half.
3. Register it in `strategies/__init__.py` by adding to `STRATEGY_REGISTRY`, and add a corresponding `Strategy` enum value in `types.py`.
4. Add tests, including a `TIGHT_PARAMS` entry in `tests/test_pinning.py` so the registry-wide invariant tests cover it.
