"""Keep-last strategy — retain pinned content + the last N interaction turns."""

from __future__ import annotations

from typing import Any

from llmigrate.pinning import split_pinned
from llmigrate.turns import group_into_turns
from llmigrate.types import Message, Strategy, TransferResult


def transform(messages: list[Message], **params: Any) -> TransferResult:
    n: int = params.get("n", 5)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    pin_first_user: bool = params.get("pin_first_user", True)

    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)
    turns = group_into_turns(rest)
    kept_turns = turns[-n:] if n > 0 else []
    kept = [msg for turn in kept_turns for msg in turn]
    dropped_count = sum(len(t) for t in turns) - len(kept)

    return TransferResult(
        messages=pinned + kept,
        strategy=Strategy.KEEP_LAST,
        metadata={"original_count": len(messages), "n": n, "dropped_count": dropped_count},
    )
