"""Token-budget strategy — keep as many recent messages as fit within a token limit."""

from __future__ import annotations

from typing import Any, Callable

from llmigrate.types import Message, Role, Strategy, TransferResult


def _default_token_estimate(text: str) -> int:
    """Approximate token count. ~4 characters per token is a rough heuristic."""
    return max(1, len(text) // 4)


def transform(messages: list[Message], **params: Any) -> TransferResult:
    max_tokens: int = params.get("max_tokens", 4096)
    tokenizer: Callable[[str], int] = params.get("tokenizer", _default_token_estimate)

    system_msgs = [m for m in messages if m.role == Role.SYSTEM]
    non_system = [m for m in messages if m.role != Role.SYSTEM]

    budget = max_tokens - sum(tokenizer(m.content) for m in system_msgs)
    budget = max(0, budget)

    # Add messages from the end until budget is exhausted
    # TODO: Align to turn boundaries and tool-call integrity
    kept: list[Message] = []
    used = 0
    for msg in reversed(non_system):
        cost = tokenizer(msg.content)
        if used + cost > budget:
            break
        kept.append(msg)
        used += cost
    kept.reverse()

    return TransferResult(
        messages=system_msgs + kept,
        strategy=Strategy.TOKEN_BUDGET,
        metadata={
            "original_count": len(messages),
            "max_tokens": max_tokens,
            "estimated_tokens_used": used,
        },
    )
