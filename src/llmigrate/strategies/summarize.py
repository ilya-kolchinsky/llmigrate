"""Summarize strategy — compress older history, keep recent tail."""

from __future__ import annotations

from typing import Any, Callable

from llmigrate.types import Message, Role, Strategy, TransferResult


def transform(messages: list[Message], **params: Any) -> TransferResult:
    tail: int = params.get("tail", 3)
    generate: Callable[..., str] | None = params.get("generate")

    system_msgs = [m for m in messages if m.role == Role.SYSTEM]
    non_system = [m for m in messages if m.role != Role.SYSTEM]

    if len(non_system) <= tail:
        return TransferResult(
            messages=list(messages),
            strategy=Strategy.SUMMARIZE,
            metadata={"original_count": len(messages), "summarized": False},
        )

    to_summarize = non_system[:-tail] if tail > 0 else non_system
    tail_msgs = non_system[-tail:] if tail > 0 else []

    if generate is None:
        # Fallback: simple truncation with a notice
        summary_text = (
            f"[Prior conversation ({len(to_summarize)} messages) omitted for brevity]"
        )
    else:
        # TODO: Build a proper summarization prompt
        # TODO: Handle async generate callables
        summary_prompt = _build_summary_prompt(to_summarize)
        summary_text = generate(summary_prompt)

    summary_msg = Message(
        role=Role.USER,
        content=summary_text,
        metadata={"llmigrate_synthetic": True, "summarized_count": len(to_summarize)},
    )

    return TransferResult(
        messages=system_msgs + [summary_msg] + tail_msgs,
        strategy=Strategy.SUMMARIZE,
        metadata={
            "original_count": len(messages),
            "summarized": generate is not None,
            "summarized_count": len(to_summarize),
            "tail": tail,
        },
    )


def _build_summary_prompt(messages: list[Message]) -> list[dict[str, Any]]:
    """Build the prompt that asks a model to summarize the conversation."""
    conversation_text = "\n".join(
        f"{m.role.value}: {m.content}" for m in messages
    )
    return [
        {
            "role": "user",
            "content": (
                "Summarize the following conversation concisely, preserving all key "
                "facts, decisions, and context needed to continue the discussion.\n\n"
                f"{conversation_text}"
            ),
        }
    ]
