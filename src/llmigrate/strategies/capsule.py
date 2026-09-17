"""Capsule strategy — extract structured state from conversation history."""

from __future__ import annotations

from typing import Any, Callable

from llmigrate.types import Message, Role, Strategy, TransferResult

DEFAULT_CAPSULE_SCHEMA = {
    "objective": "The main goal or task being worked on",
    "completed": "What has been accomplished so far",
    "observations": "Key facts, findings, or constraints discovered",
    "open_questions": "Unresolved questions or blockers",
    "next_steps": "What should happen next",
}


def transform(messages: list[Message], **params: Any) -> TransferResult:
    generate: Callable[..., str] | None = params.get("generate")
    schema: dict[str, str] = params.get("schema", DEFAULT_CAPSULE_SCHEMA)

    system_msgs = [m for m in messages if m.role == Role.SYSTEM]
    non_system = [m for m in messages if m.role != Role.SYSTEM]

    if generate is None:
        # Fallback: heuristic extraction
        # TODO: Implement heuristic capsule extraction
        capsule_text = _heuristic_capsule(non_system, schema)
    else:
        # TODO: Handle async generate callables
        extraction_prompt = _build_extraction_prompt(non_system, schema)
        capsule_text = generate(extraction_prompt)

    capsule_msg = Message(
        role=Role.USER,
        content=capsule_text,
        metadata={
            "llmigrate_synthetic": True,
            "capsule_schema": list(schema.keys()),
        },
    )

    return TransferResult(
        messages=system_msgs + [capsule_msg],
        strategy=Strategy.CAPSULE,
        metadata={
            "original_count": len(messages),
            "schema_fields": list(schema.keys()),
        },
    )


def _heuristic_capsule(messages: list[Message], schema: dict[str, str]) -> str:
    """Best-effort capsule without a model call."""
    lines = ["[Session state transferred from previous model]", ""]
    for field_name, description in schema.items():
        lines.append(f"## {field_name.replace('_', ' ').title()}")
        lines.append(f"({description})")
        lines.append("[Could not extract automatically — no generate function provided]")
        lines.append("")
    return "\n".join(lines)


def _build_extraction_prompt(
    messages: list[Message], schema: dict[str, str]
) -> list[dict[str, Any]]:
    """Build the prompt that asks a model to extract a structured capsule."""
    conversation_text = "\n".join(
        f"{m.role.value}: {m.content}" for m in messages
    )
    schema_text = "\n".join(
        f"- **{k}**: {v}" for k, v in schema.items()
    )
    return [
        {
            "role": "user",
            "content": (
                "Extract a structured state capsule from the following conversation. "
                "For each field below, provide the relevant information:\n\n"
                f"{schema_text}\n\n"
                "Format your response with a heading for each field followed by the "
                "extracted content.\n\n"
                f"Conversation:\n{conversation_text}"
            ),
        }
    ]
