"""Keep-last strategy — retain system messages + last N interaction turns."""

from __future__ import annotations

from typing import Any

from llmigrate.types import Message, Role, Strategy, TransferResult


def transform(messages: list[Message], **params: Any) -> TransferResult:
    n: int = params.get("n", 5)

    system_msgs = [m for m in messages if m.role == Role.SYSTEM]
    non_system = [m for m in messages if m.role != Role.SYSTEM]

    # Keep last n messages from the non-system portion
    # TODO: Align to turn boundaries (never split a user/assistant pair)
    # TODO: Handle tool call/result integrity (never orphan a tool result)
    kept = non_system[-n:] if len(non_system) > n else non_system

    return TransferResult(
        messages=system_msgs + kept,
        strategy=Strategy.KEEP_LAST,
        metadata={"original_count": len(messages), "n": n},
    )
