"""Summarize strategy — compress older history, keep recent tail."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from llmigrate._util import call_generate
from llmigrate.pinning import split_pinned
from llmigrate.tokens import estimate_tokens
from llmigrate.types import Message, Role, Strategy, TransferResult

_TRUNCATION_CHARS_PER_TOKEN = 4


def transform(messages: list[Message], **params: Any) -> TransferResult:
    tail: int = params.get("tail", 3)
    generate: Callable[..., str] | None = params.get("generate")
    generate_kwargs: dict[str, Any] = params.get("generate_kwargs") or {}
    max_summary_tokens: int | None = params.get("max_summary_tokens")
    pin_first_user: bool = params.get("pin_first_user", True)

    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)

    if len(rest) <= tail:
        return TransferResult(
            messages=list(messages),
            strategy=Strategy.SUMMARIZE,
            metadata={"original_count": len(messages), "summarized": False},
        )

    to_summarize = rest[:-tail] if tail > 0 else rest
    tail_msgs = rest[-tail:] if tail > 0 else []

    latency_ms: float | None = None
    truncated = False
    if generate is None:
        # Fallback: simple truncation with a notice
        summary_text = (
            f"[Prior conversation ({len(to_summarize)} messages) omitted for brevity]"
        )
    else:
        summary_prompt = _build_summary_prompt(to_summarize)
        start = time.monotonic()
        summary_text = call_generate(generate, summary_prompt, **generate_kwargs)
        latency_ms = (time.monotonic() - start) * 1000
        if max_summary_tokens is not None:
            summary_text, truncated = _truncate_to_budget(summary_text, max_summary_tokens)

    summary_msg = Message(
        role=Role.USER,
        content=summary_text,
        metadata={"llmigrate_synthetic": True, "summarized_count": len(to_summarize)},
    )

    metadata: dict[str, Any] = {
        "original_count": len(messages),
        "summarized": generate is not None,
        "summarized_count": len(to_summarize),
        "tail": tail,
    }
    if latency_ms is not None:
        metadata["latency_ms"] = latency_ms
    if truncated:
        metadata["summary_truncated"] = True

    return TransferResult(
        messages=pinned + [summary_msg] + tail_msgs,
        strategy=Strategy.SUMMARIZE,
        metadata=metadata,
    )


def _truncate_to_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    if estimate_tokens(text) <= max_tokens:
        return text, False
    max_chars = max(0, max_tokens * _TRUNCATION_CHARS_PER_TOKEN)
    return text[:max_chars].rstrip() + " [truncated]", True


def _build_summary_prompt(messages: list[Message]) -> list[dict[str, Any]]:
    """Build the prompt that asks a model to summarize the conversation."""
    conversation_text = "\n".join(
        f"{m.role.value}: {m.content}" for m in messages
    )
    return [
        {
            "role": "user",
            "content": (
                "Summarize the conversation below for a model that will continue "
                "it. Preserve, as concise bullet points:\n"
                "- Key facts and constraints established so far\n"
                "- Decisions that were made and why\n"
                "- Any open questions or unresolved threads\n"
                "Do not add commentary or restate these instructions. Do not omit "
                "specific names, numbers, file paths, or error messages.\n\n"
                f"Conversation:\n{conversation_text}"
            ),
        }
    ]
