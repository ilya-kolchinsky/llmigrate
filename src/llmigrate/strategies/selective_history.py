"""SelectiveHistory strategy — verbatim, budget-constrained event selection.

Unlike summarization or structured state extraction, this strategy never rewrites
selected content: it chooses a subset of the original events (ranked by a
pluggable Selector, not just recency) that fits a token budget, and returns
them verbatim, in their original chronological order.

Messages that share tool-call IDs are selected as one event, so a call and its
known results cannot be separated by the selector. Other messages remain
independent events.
"""

from __future__ import annotations

from typing import Any

from llmigrate.pinning import split_pinned
from llmigrate.selectors import Selector, category_of
from llmigrate.tokens import default_tokenizer, estimate_message_tokens
from llmigrate.turns import group_tool_events
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

    events = group_tool_events(rest)
    forced_events: list[list[Message]] = []
    candidate_events = list(events)
    if always_keep:
        forced_events = [
            event for event in events if any(category_of(m) in always_keep for m in event)
        ]
        forced_event_ids = {id(event) for event in forced_events}
        candidate_events = [event for event in events if id(event) not in forced_event_ids]

        kept_forced_events: list[list[Message]] = []
        used = 0
        for event in reversed(forced_events):
            cost = sum(estimate_message_tokens(m, tokenizer) for m in event)
            if used + cost > remaining_budget:
                continue
            kept_forced_events.append(event)
            used += cost
        forced_events = kept_forced_events
        remaining_budget -= used

    candidates = [message for event in candidate_events for message in event]
    scores = selector.score(candidates) if candidates else []
    if len(scores) != len(candidates):
        raise ValueError("selector.score() must return one score per candidate message")
    score_by_id = {id(m): s for m, s in zip(candidates, scores)}
    event_scores = [
        max(score_by_id[id(message)] for message in event)
        for event in candidate_events
    ]
    ranked = sorted(
        zip(candidate_events, event_scores), key=lambda pair: pair[1], reverse=True
    )

    selected_events: list[list[Message]] = []
    used = 0
    for event, _score in ranked:
        cost = sum(estimate_message_tokens(m, tokenizer) for m in event)
        if used + cost > remaining_budget:
            continue
        selected_events.append(event)
        used += cost

    kept_ids = {
        id(message)
        for event in (forced_events + selected_events)
        for message in event
    }
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
