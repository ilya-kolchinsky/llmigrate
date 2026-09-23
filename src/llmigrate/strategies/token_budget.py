"""Token-budget strategy — keep as many recent turns as fit within a token limit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llmigrate.models import context_window_for
from llmigrate.pinning import split_pinned
from llmigrate.tokens import default_tokenizer, estimate_tokens
from llmigrate.turns import group_into_turns
from llmigrate.types import Message, Strategy, TransferResult

_DEFAULT_MAX_TOKENS = 4096
_TARGET_MODEL_BUDGET_FRACTION = 0.9


def transform(messages: list[Message], **params: Any) -> TransferResult:
    target_model: str | None = params.get("target_model")
    pin_first_user: bool = params.get("pin_first_user", True)

    if "max_tokens" in params:
        max_tokens: int = params["max_tokens"]
    else:
        window = context_window_for(target_model)
        max_tokens = int(window * _TARGET_MODEL_BUDGET_FRACTION) if window else _DEFAULT_MAX_TOKENS
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")

    tokenizer: Callable[[str], int] = params.get("tokenizer") or default_tokenizer(target_model)

    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)
    pinned_tokens = sum(estimate_tokens(m.content, tokenizer) for m in pinned)
    budget = max(0, max_tokens - pinned_tokens)

    turns = group_into_turns(rest)
    kept_turns: list[list[Message]] = []
    used = 0
    for turn in reversed(turns):
        cost = sum(estimate_tokens(m.content, tokenizer) for m in turn)
        if used + cost > budget:
            break
        kept_turns.append(turn)
        used += cost
    kept_turns.reverse()
    kept = [msg for turn in kept_turns for msg in turn]

    dropped_count = sum(len(t) for t in turns) - len(kept)

    return TransferResult(
        messages=pinned + kept,
        strategy=Strategy.TOKEN_BUDGET,
        metadata={
            "original_count": len(messages),
            "max_tokens": max_tokens,
            "estimated_tokens_used": pinned_tokens + used,
            "dropped_count": dropped_count,
        },
    )
