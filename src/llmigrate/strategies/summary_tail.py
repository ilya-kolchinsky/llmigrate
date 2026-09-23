"""SummaryTail strategy — summarize older history, keep the tail verbatim.

Combines compressed long-term history with verbatim recent context: the tail
is the largest recent, turn-aligned, contiguous section of history that fits
`tail_budget` (no reordering, whole turns only); everything before it is
summarized under `summary_budget`. If the whole (non-pinned) history already
fits in `tail_budget`, no summarizer call is made at all.
"""

from __future__ import annotations

from typing import Any

from llmigrate.pinning import split_pinned
from llmigrate.summarizers import as_summarizer
from llmigrate.tokens import default_tokenizer, estimate_tokens
from llmigrate.turns import group_into_turns
from llmigrate.types import Message, Role, Strategy, TransferResult

_TRUNCATION_CHARS_PER_TOKEN = 4


def transform(messages: list[Message], **params: Any) -> TransferResult:
    if "total_budget" not in params:
        raise ValueError("summary_tail requires a 'total_budget' parameter (int, tokens)")
    total_budget: int = params["total_budget"]

    if "summary_budget" not in params:
        raise ValueError("summary_tail requires a 'summary_budget' parameter (int, tokens)")
    summary_budget: int = params["summary_budget"]

    tail_budget: int = params.get("tail_budget", total_budget - summary_budget)
    if summary_budget < 0 or tail_budget < 0:
        raise ValueError("summary_budget and tail_budget must be non-negative")
    if summary_budget + tail_budget > total_budget:
        raise ValueError(
            f"summary_budget ({summary_budget}) + tail_budget ({tail_budget}) "
            f"exceeds total_budget ({total_budget})"
        )

    generate_kwargs: dict[str, Any] = params.get("generate_kwargs") or {}
    pin_first_user: bool = params.get("pin_first_user", True)
    tokenizer = params.get("tokenizer") or default_tokenizer(params.get("target_model"))

    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)
    index_of = {id(m): i for i, m in enumerate(messages)}

    turns = group_into_turns(rest)
    tail_turns: list[list[Message]] = []
    used = 0
    for turn in reversed(turns):
        cost = sum(estimate_tokens(m.content, tokenizer) for m in turn)
        if used + cost > tail_budget:
            break
        tail_turns.append(turn)
        used += cost
    tail_turns.reverse()
    tail_msgs = [m for turn in tail_turns for m in turn]
    tail_tokens = used

    prefix_msgs = rest[: len(rest) - len(tail_msgs)]

    summary_msg: Message | None = None
    summarizer_result = None
    summarizer_repr: str | None = None
    truncated = False
    if prefix_msgs:
        summarizer = as_summarizer(
            params.get("summarizer") or params.get("generate"), generate_kwargs
        )
        summarizer_repr = repr(summarizer)
        prompt = _build_summary_prompt(prefix_msgs)
        summarizer_result = summarizer.summarize(prompt)
        summary_text = summarizer_result.text
        if estimate_tokens(summary_text, tokenizer) > summary_budget:
            max_chars = max(0, summary_budget * _TRUNCATION_CHARS_PER_TOKEN)
            summary_text = summary_text[:max_chars].rstrip() + " [truncated]"
            truncated = True
        summary_msg = Message(
            role=Role.USER,
            content=summary_text,
            metadata={
                "llmigrate_synthetic": True,
                "segment": "summary_prefix",
                "summarized_count": len(prefix_msgs),
                **({"summary_truncated": True} if truncated else {}),
            },
        )

    tagged_tail = [
        Message(role=m.role, content=m.content, metadata={**m.metadata, "segment": "verbatim_tail"})
        for m in tail_msgs
    ]

    result_messages = pinned + ([summary_msg] if summary_msg else []) + tagged_tail

    metadata: dict[str, Any] = {
        "original_count": len(messages),
        "prefix_event_ids": [index_of[id(m)] for m in prefix_msgs],
        "tail_event_ids": [index_of[id(m)] for m in tail_msgs],
        "summary_tokens": estimate_tokens(summary_msg.content, tokenizer) if summary_msg else 0,
        "tail_tokens": tail_tokens,
        "total_transferred_tokens": sum(
            estimate_tokens(m.content, tokenizer) for m in result_messages
        ),
        "summarizer": summarizer_repr,
    }
    if truncated:
        metadata["summary_truncated"] = True
    if summarizer_result is not None:
        if summarizer_result.latency_ms is not None:
            metadata["latency_ms"] = summarizer_result.latency_ms
        if summarizer_result.token_usage is not None:
            metadata["token_usage"] = summarizer_result.token_usage
        if summarizer_result.cost is not None:
            metadata["cost"] = summarizer_result.cost
        if summarizer_result.model is not None:
            metadata["model"] = summarizer_result.model

    return TransferResult(
        messages=result_messages,
        strategy=Strategy.SUMMARY_TAIL,
        metadata=metadata,
    )


def _build_summary_prompt(messages: list[Message]) -> list[dict[str, Any]]:
    conversation_text = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
    return [
        {
            "role": "user",
            "content": (
                "Summarize the older portion of this conversation for a model "
                "that will continue it, given the verbatim recent messages "
                "separately. Preserve key facts, decisions, and open threads; do "
                "not omit specific names, numbers, file paths, or error "
                "messages.\n\n"
                f"Conversation:\n{conversation_text}"
            ),
        }
    ]
