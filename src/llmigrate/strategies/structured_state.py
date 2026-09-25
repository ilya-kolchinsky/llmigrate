"""Structured state strategy — extract structured state from conversation history."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from llmigrate._util import call_generate, message_to_text
from llmigrate.pinning import split_pinned
from llmigrate.types import CompressionResult, Message, MigrationResult, Role, Strategy

DEFAULT_STATE_SCHEMA = {
    "objective": "The main goal or task being worked on",
    "completed": "What has been accomplished so far",
    "observations": "Key facts, findings, or constraints discovered",
    "open_questions": "Unresolved questions or blockers",
    "next_steps": "What should happen next",
}

_HEADING_RE = re.compile(r"^##\s*(.+?)\s*$", re.MULTILINE)


def compress(dropped: list[Message], **params: Any) -> CompressionResult:
    generate: Callable[..., str] | None = params.get("generate")
    generate_kwargs: dict[str, Any] = params.get("generate_kwargs") or {}
    schema: dict[str, str] = params.get("schema", DEFAULT_STATE_SCHEMA)
    pinned: list[Message] = params.get("_pinned", [])

    latency_ms: float | None = None
    if generate is None:
        state_text = _heuristic_state(pinned, dropped, schema)
    else:
        extraction_prompt = _build_extraction_prompt(dropped, schema)
        start = time.monotonic()
        state_text = call_generate(generate, extraction_prompt, **generate_kwargs)
        latency_ms = (time.monotonic() - start) * 1000

    state_data = _parse_state_response(state_text, schema)

    state_msg = Message(
        role=Role.USER,
        content=state_text,
        metadata={
            "llmigrate_synthetic": True,
            "state_schema": list(schema.keys()),
        },
    )

    metadata: dict[str, Any] = {
        "schema_fields": list(schema.keys()),
        "state_data": state_data,
    }
    if latency_ms is not None:
        metadata["latency_ms"] = latency_ms

    return CompressionResult(messages=[state_msg], metadata=metadata)


def transform(messages: list[Message], **params: Any) -> MigrationResult:
    pin_first_user: bool = params.get("pin_first_user", True)
    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)

    result = compress(rest, _pinned=pinned, **params)

    return MigrationResult(
        messages=pinned + result.messages,
        strategies=[Strategy.STRUCTURED_STATE],
        metadata={"original_count": len(messages), **result.metadata},
    )


def _heuristic_state(
    pinned: list[Message], rest: list[Message], schema: dict[str, str]
) -> str:
    user_msgs = [m for m in rest if m.role == Role.USER]
    assistant_msgs = [
        m for m in rest if m.role in (Role.ASSISTANT, Role.TOOL_CALL, Role.TOOL_RESULT)
    ]
    pinned_task = next(
        (m.content or message_to_text(m) for m in pinned if m.role == Role.USER), None
    )

    values: dict[str, str] = {}
    if "objective" in schema:
        objective = pinned_task or (
            user_msgs[0].content or message_to_text(user_msgs[0]) if user_msgs else ""
        )
        values["objective"] = objective or "(no objective found)"
    if "completed" in schema:
        values["completed"] = (
            "; ".join(message_to_text(m)[:200] for m in assistant_msgs[:-1])
            or "(nothing completed yet)"
        )
    if "observations" in schema:
        values["observations"] = (
            message_to_text(assistant_msgs[-1])[:400]
            if assistant_msgs
            else "(no observations)"
        )
    if "open_questions" in schema:
        last_user = (
            user_msgs[-1].content or message_to_text(user_msgs[-1]) if user_msgs else ""
        )
        values["open_questions"] = (
            last_user if last_user.strip().endswith("?") else "(none identified)"
        )
    if "next_steps" in schema:
        values["next_steps"] = "(continue from the most recent exchange above)"
    for field_name in schema:
        values.setdefault(field_name, "(not available without a generate function)")

    lines = ["[Session state transferred from previous model]", ""]
    for field_name in schema:
        lines.append(f"## {field_name.replace('_', ' ').title()}")
        lines.append(values[field_name])
        lines.append("")
    return "\n".join(lines)


def _parse_state_response(text: str, schema: dict[str, str]) -> dict[str, str]:
    matches = list(_HEADING_RE.finditer(text))
    normalized = {k.replace("_", " ").lower(): k for k in schema}
    result: dict[str, str] = {}
    for i, match in enumerate(matches):
        field = normalized.get(match.group(1).strip().lower())
        if field is None:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result[field] = text[start:end].strip()
    return result


def _build_extraction_prompt(
    messages: list[Message], schema: dict[str, str]
) -> list[dict[str, Any]]:
    conversation_text = "\n".join(message_to_text(message) for message in messages)
    schema_text = "\n".join(
        f"- **{k}**: {v}" for k, v in schema.items()
    )
    return [
        {
            "role": "user",
            "content": (
                "Extract structured state from the conversation below, "
                "so another model can continue the work with no other context. "
                "For each field, use a '## <field name>' heading exactly as given, "
                "followed by its content on the next line(s). Be concrete: preserve "
                "specific names, file paths, numbers, and error messages verbatim — "
                "do not paraphrase them away.\n\n"
                f"Fields:\n{schema_text}\n\n"
                f"Conversation:\n{conversation_text}"
            ),
        }
    ]
