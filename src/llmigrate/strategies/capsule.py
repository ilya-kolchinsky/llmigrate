"""Capsule strategy — extract structured state from conversation history."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Any

from llmigrate._util import call_generate
from llmigrate.pinning import split_pinned
from llmigrate.types import Message, Role, Strategy, TransferResult

DEFAULT_CAPSULE_SCHEMA = {
    "objective": "The main goal or task being worked on",
    "completed": "What has been accomplished so far",
    "observations": "Key facts, findings, or constraints discovered",
    "open_questions": "Unresolved questions or blockers",
    "next_steps": "What should happen next",
}

_HEADING_RE = re.compile(r"^##\s*(.+?)\s*$", re.MULTILINE)


def transform(messages: list[Message], **params: Any) -> TransferResult:
    generate: Callable[..., str] | None = params.get("generate")
    generate_kwargs: dict[str, Any] = params.get("generate_kwargs") or {}
    schema: dict[str, str] = params.get("schema", DEFAULT_CAPSULE_SCHEMA)
    pin_first_user: bool = params.get("pin_first_user", True)

    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)

    latency_ms: float | None = None
    if generate is None:
        # Fallback: heuristic extraction
        capsule_text = _heuristic_capsule(pinned, rest, schema)
    else:
        extraction_prompt = _build_extraction_prompt(rest, schema)
        start = time.monotonic()
        capsule_text = call_generate(generate, extraction_prompt, **generate_kwargs)
        latency_ms = (time.monotonic() - start) * 1000

    capsule_data = _parse_capsule_response(capsule_text, schema)

    capsule_msg = Message(
        role=Role.USER,
        content=capsule_text,
        metadata={
            "llmigrate_synthetic": True,
            "capsule_schema": list(schema.keys()),
        },
    )

    metadata: dict[str, Any] = {
        "original_count": len(messages),
        "schema_fields": list(schema.keys()),
        "capsule_data": capsule_data,
    }
    if latency_ms is not None:
        metadata["latency_ms"] = latency_ms

    return TransferResult(
        messages=pinned + [capsule_msg],
        strategy=Strategy.CAPSULE,
        metadata=metadata,
    )


def _heuristic_capsule(
    pinned: list[Message], rest: list[Message], schema: dict[str, str]
) -> str:
    """Best-effort, rule-based capsule extraction — no model call.

    objective: the pinned task content, or the first user message otherwise.
    completed/observations: distilled from assistant messages.
    open_questions: the last user message, if it reads like a question.
    next_steps: a generic placeholder when nothing else is inferable.
    """
    user_msgs = [m for m in rest if m.role == Role.USER]
    assistant_msgs = [m for m in rest if m.role == Role.ASSISTANT]
    pinned_task = next((m.content for m in pinned if m.role == Role.USER), None)

    values: dict[str, str] = {}
    if "objective" in schema:
        objective = pinned_task or (user_msgs[0].content if user_msgs else "")
        values["objective"] = objective or "(no objective found)"
    if "completed" in schema:
        values["completed"] = (
            "; ".join(m.content[:200] for m in assistant_msgs[:-1]) or "(nothing completed yet)"
        )
    if "observations" in schema:
        values["observations"] = (
            assistant_msgs[-1].content[:400] if assistant_msgs else "(no observations)"
        )
    if "open_questions" in schema:
        last_user = user_msgs[-1].content if user_msgs else ""
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


def _parse_capsule_response(text: str, schema: dict[str, str]) -> dict[str, str]:
    """Parse a '## <field>' formatted response into {field: text}, best-effort.
    Works for both the heuristic and generate()-based output, since both use
    the same heading format."""
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
                "Extract a structured state capsule from the conversation below, "
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
