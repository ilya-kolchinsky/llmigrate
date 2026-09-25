"""SelectiveHistory strategy — verbatim, budget-constrained event selection.

Unlike summarization or structured state extraction, this strategy never rewrites
selected content: it chooses a subset of the original events (ranked by a
pluggable Selector, not just recency) that fits a token budget, and returns
them verbatim, in their original chronological order.

Note: unlike keep_last/token_budget, this strategy selects at message
granularity and does not guarantee tool_call/tool_result pairing — callers
who need that should put both categories in `always_keep`.
"""

from __future__ import annotations

from typing import Any

from llmigrate.pinning import split_pinned
from llmigrate.selectors import Selector, category_of
from llmigrate.tokens import default_tokenizer, estimate_message_tokens
from llmigrate.types import Message, MigrationResult, SelectionResult, Strategy


def select(rest: list[Message], **params: Any) -> SelectionResult:
    if "budget" not in params:
        raise ValueError("selective_history requires a 'budget' parameter (int, tokens)")
    budget: int = params["budget"]
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    if "selector" not in params:
        raise ValueError("selective_history requires a 'selector' parameter")
    selector: Selector = params["selector"]

    always_keep: set[str] | None = params.get("always_keep")
    tokenizer = params.get("tokenizer") or default_tokenizer(params.get("target_model"))
    pinned_tokens: int = params.get("_pinned_tokens", 0)

    remaining_budget = max(0, budget - pinned_tokens)

    forced: list[Message] = []
    candidates = list(rest)
    if always_keep:
        forced = [m for m in rest if category_of(m) in always_keep]
        forced_ids = {id(m) for m in forced}
        candidates = [m for m in rest if id(m) not in forced_ids]

        kept_forced: list[Message] = []
        used = 0
        for m in reversed(forced):
            cost = estimate_message_tokens(m, tokenizer)
            if used + cost > remaining_budget:
                continue
            kept_forced.append(m)
            used += cost
        forced = kept_forced
        remaining_budget -= used

    scores = selector.score(candidates) if candidates else []
    score_by_id = {id(m): s for m, s in zip(candidates, scores)}
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)

    selected: list[Message] = []
    used = 0
    for m, _score in ranked:
        cost = estimate_message_tokens(m, tokenizer)
        if used + cost > remaining_budget:
            continue
        selected.append(m)
        used += cost

    kept_ids = {id(m) for m in (forced + selected)}
    kept = [m for m in rest if id(m) in kept_ids]
    dropped = [m for m in rest if id(m) not in kept_ids]

    return SelectionResult(
        kept=kept,
        dropped=dropped,
        metadata={
            "selector": repr(selector),
            "scores": score_by_id,
        },
    )


def transform(messages: list[Message], **params: Any) -> MigrationResult:
    pin_first_user: bool = params.get("pin_first_user", True)
    tokenizer = params.get("tokenizer") or default_tokenizer(params.get("target_model"))

    index_of = {id(m): i for i, m in enumerate(messages)}
    original_tokens = sum(estimate_message_tokens(m, tokenizer) for m in messages)

    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)
    pinned_tokens = sum(estimate_message_tokens(m, tokenizer) for m in pinned)

    result = select(rest, _pinned_tokens=pinned_tokens, **params)

    result_messages = pinned + result.kept
    selected_ids = sorted(index_of[id(m)] for m in result_messages)
    dropped_ids = sorted(set(index_of.values()) - set(selected_ids))
    transferred_tokens = sum(estimate_message_tokens(m, tokenizer) for m in result_messages)

    score_by_id = result.metadata.get("scores", {})
    scores_by_index = {index_of[mid]: score for mid, score in score_by_id.items()}

    return MigrationResult(
        messages=result_messages,
        strategies=[Strategy.SELECTIVE_HISTORY],
        metadata={
            "original_count": len(messages),
            "selected_event_ids": selected_ids,
            "dropped_event_ids": dropped_ids,
            "original_tokens": original_tokens,
            "transferred_tokens": transferred_tokens,
            "selector": result.metadata.get("selector", ""),
            "scores": scores_by_index,
        },
    )
