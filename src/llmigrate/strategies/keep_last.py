"""Keep-last strategy — retain pinned content + the last N interaction turns."""

from __future__ import annotations

from typing import Any

from llmigrate.pinning import split_pinned
from llmigrate.turns import group_into_turns
from llmigrate.types import Message, MigrationResult, SelectionResult, Strategy


def select(rest: list[Message], **params: Any) -> SelectionResult:
    n: int = params.get("n", 5)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")

    turns = group_into_turns(rest)
    kept_turns = turns[-n:] if n > 0 else []
    kept = [msg for turn in kept_turns for msg in turn]

    dropped_start = len(turns) - len(kept_turns)
    dropped = [msg for turn in turns[:dropped_start] for msg in turn]

    return SelectionResult(
        kept=kept,
        dropped=dropped,
        metadata={"n": n, "dropped_count": len(dropped)},
    )


def transform(messages: list[Message], **params: Any) -> MigrationResult:
    pin_first_user: bool = params.get("pin_first_user", True)
    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)

    result = select(rest, **params)

    return MigrationResult(
        messages=pinned + result.kept,
        strategies=[Strategy.KEEP_LAST],
        metadata={"original_count": len(messages), **result.metadata},
    )
