"""Token-budget strategy — keep as many recent turns as fit within a token limit."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llmigrate.models import context_window_for
from llmigrate.pinning import split_pinned
from llmigrate.tokens import default_tokenizer, estimate_tokens
from llmigrate.turns import group_into_turns
from llmigrate.types import Message, MigrationResult, SelectionResult, Strategy

_DEFAULT_MAX_TOKENS = 4096
_TARGET_MODEL_BUDGET_FRACTION = 0.9


def _resolve_budget(params: dict[str, Any]) -> int:
    target_model: str | None = params.get("target_model")
    if "max_tokens" in params:
        max_tokens: int = params["max_tokens"]
    else:
        window = context_window_for(target_model)
        max_tokens = int(window * _TARGET_MODEL_BUDGET_FRACTION) if window else _DEFAULT_MAX_TOKENS
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    return max_tokens


def select(rest: list[Message], **params: Any) -> SelectionResult:
    max_tokens: int = _resolve_budget(params)
    tokenizer: Callable[[str], int] = params.get("tokenizer") or default_tokenizer(
        params.get("target_model")
    )
    pinned_tokens: int = params.get("_pinned_tokens", 0)

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

    dropped_start = len(turns) - len(kept_turns)
    dropped = [msg for turn in turns[:dropped_start] for msg in turn]

    return SelectionResult(
        kept=kept,
        dropped=dropped,
        metadata={
            "max_tokens": max_tokens,
            "estimated_tokens_used": pinned_tokens + used,
            "dropped_count": len(dropped),
        },
    )


def transform(messages: list[Message], **params: Any) -> MigrationResult:
    pin_first_user: bool = params.get("pin_first_user", True)
    tokenizer: Callable[[str], int] = params.get("tokenizer") or default_tokenizer(
        params.get("target_model")
    )

    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)
    pinned_tokens = sum(estimate_tokens(m.content, tokenizer) for m in pinned)

    result = select(rest, _pinned_tokens=pinned_tokens, **params)

    return MigrationResult(
        messages=pinned + result.kept,
        strategies=[Strategy.TOKEN_BUDGET],
        metadata={"original_count": len(messages), **result.metadata},
    )
