"""Audit strategy — append verification instructions for the receiving model."""

from __future__ import annotations

from typing import Any

from llmigrate.types import Message, Role, Strategy, TransferResult

DEFAULT_AUDIT_INSTRUCTION = (
    "You are continuing a conversation that was started with a different model. "
    "Before proceeding, review the conversation history above and verify that "
    "the assumptions, facts, and reasoning are sound. If you find any issues, "
    "flag them before continuing. Then proceed with addressing the user's request."
)


def transform(messages: list[Message], **params: Any) -> TransferResult:
    instruction: str = params.get("instruction", DEFAULT_AUDIT_INSTRUCTION)

    result_messages = list(messages)

    audit_msg = Message(
        role=Role.USER,
        content=instruction,
        metadata={"llmigrate_synthetic": True, "audit_instruction": True},
    )
    result_messages.append(audit_msg)

    return TransferResult(
        messages=result_messages,
        strategy=Strategy.AUDIT,
        metadata={"original_count": len(messages)},
    )
