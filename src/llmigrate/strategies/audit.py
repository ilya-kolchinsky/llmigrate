"""Audit strategy — append verification instructions for the receiving model."""

from __future__ import annotations

from typing import Any

from llmigrate.types import Message, MigrationResult, Role, Strategy, ValidationResult

DEFAULT_AUDIT_INSTRUCTION_TEMPLATE = (
    "You are continuing a conversation{source_clause}. "
    "Before proceeding, review the conversation history above and verify that "
    "the assumptions, facts, and reasoning are sound. If you find any issues, "
    "flag them before continuing. Then proceed with addressing the user's request."
)


def validate(messages: list[Message], **params: Any) -> ValidationResult:
    source_model: str | None = params.get("source_model")
    target_model: str | None = params.get("target_model")

    if "instruction" in params:
        instruction: str = params["instruction"]
    else:
        source_clause = (
            f" that was started with a different model ({source_model})"
            if source_model
            else " that was started with a different model"
        )
        instruction = DEFAULT_AUDIT_INSTRUCTION_TEMPLATE.format(source_clause=source_clause)

    audit_msg = Message(
        role=Role.USER,
        content=instruction,
        metadata={"llmigrate_synthetic": True, "audit_instruction": True},
    )

    metadata: dict[str, Any] = {}
    if source_model:
        metadata["source_model"] = source_model
    if target_model:
        metadata["target_model"] = target_model

    return ValidationResult(
        messages=list(messages) + [audit_msg],
        metadata=metadata,
    )


def transform(messages: list[Message], **params: Any) -> MigrationResult:
    result = validate(messages, **params)

    return MigrationResult(
        messages=result.messages,
        strategies=[Strategy.AUDIT],
        metadata={"original_count": len(messages), **result.metadata},
    )
