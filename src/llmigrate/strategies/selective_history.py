"""SelectiveHistory strategy — verbatim, budget-constrained event selection.

Unlike summarization or capsule extraction, this strategy never rewrites
selected content: it chooses a subset of the original events (ranked by a
pluggable Selector, not just recency) that fits a token budget, and returns
them verbatim, in their original chronological order.

Note: unlike keep_last/token_budget/summary_tail, this strategy selects at
message granularity and does not guarantee tool_call/tool_result pairing —
callers who need that should put both categories in `always_keep`.
"""

from __future__ import annotations

from typing import Any

from llmigrate.pinning import split_pinned
from llmigrate.selectors import Selector, category_of
from llmigrate.tokens import default_tokenizer, estimate_tokens
from llmigrate.types import Message, Strategy, TransferResult


def transform(messages: list[Message], **params: Any) -> TransferResult:
    if "budget" not in params:
        raise ValueError("selective_history requires a 'budget' parameter (int, tokens)")
    budget: int = params["budget"]
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")

    if "selector" not in params:
        raise ValueError("selective_history requires a 'selector' parameter")
    selector: Selector = params["selector"]

    always_keep: set[str] | None = params.get("always_keep")
    pin_first_user: bool = params.get("pin_first_user", True)
    tokenizer = params.get("tokenizer") or default_tokenizer(params.get("target_model"))

    index_of = {id(m): i for i, m in enumerate(messages)}
    original_tokens = sum(estimate_tokens(m.content, tokenizer) for m in messages)

    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)
    remaining_budget = max(0, budget - sum(estimate_tokens(m.content, tokenizer) for m in pinned))

    forced: list[Message] = []
    candidates = list(rest)
    if always_keep:
        forced = [m for m in rest if category_of(m) in always_keep]
        forced_ids = {id(m) for m in forced}
        candidates = [m for m in rest if id(m) not in forced_ids]

        # Mandatory retention: if `forced` alone exceeds the budget, keep as
        # many as fit, most-recent-first.
        kept_forced: list[Message] = []
        used = 0
        for m in reversed(forced):
            cost = estimate_tokens(m.content, tokenizer)
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
        cost = estimate_tokens(m.content, tokenizer)
        if used + cost > remaining_budget:
            continue
        selected.append(m)
        used += cost

    kept_ids = {id(m) for m in (forced + selected)}
    kept_rest = [m for m in rest if id(m) in kept_ids]  # restores chronological order

    result_messages = pinned + kept_rest
    selected_ids = sorted(index_of[id(m)] for m in result_messages)
    dropped_ids = sorted(set(index_of.values()) - set(selected_ids))
    transferred_tokens = sum(estimate_tokens(m.content, tokenizer) for m in result_messages)

    return TransferResult(
        messages=result_messages,
        strategy=Strategy.SELECTIVE_HISTORY,
        metadata={
            "original_count": len(messages),
            "selected_event_ids": selected_ids,
            "dropped_event_ids": dropped_ids,
            "original_tokens": original_tokens,
            "transferred_tokens": transferred_tokens,
            "selector": repr(selector),
            "scores": {index_of[mid]: score for mid, score in score_by_id.items()},
        },
    )
