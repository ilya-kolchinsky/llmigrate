"""Summarize strategy — compress conversation history into a summary message."""

from __future__ import annotations

from typing import Any

from llmigrate.pinning import split_pinned
from llmigrate.summarizers import as_summarizer
from llmigrate.tokens import estimate_tokens
from llmigrate.types import CompressionResult, Message, MigrationResult, Role, Strategy

_TRUNCATION_CHARS_PER_TOKEN = 4


def compress(dropped: list[Message], **params: Any) -> CompressionResult:
    generate = params.get("generate")
    generate_kwargs: dict[str, Any] = params.get("generate_kwargs") or {}
    max_summary_tokens: int | None = params.get("max_summary_tokens")
    summarizer_obj = params.get("summarizer")

    if not dropped:
        return CompressionResult(messages=[], metadata={"summarized": False})

    truncated = False
    summarizer_result = None
    summarizer_repr: str | None = None

    if generate is None and summarizer_obj is None:
        summary_text = (
            f"[Prior conversation ({len(dropped)} messages) omitted for brevity]"
        )
        metadata: dict[str, Any] = {
            "summarized": False,
            "summarized_count": len(dropped),
        }
    else:
        summarizer = as_summarizer(summarizer_obj or generate, generate_kwargs)
        summarizer_repr = repr(summarizer)
        prompt = _build_summary_prompt(dropped)
        summarizer_result = summarizer.summarize(prompt)
        summary_text = summarizer_result.text

        if max_summary_tokens is not None:
            summary_text, truncated = _truncate_to_budget(summary_text, max_summary_tokens)

        metadata = {
            "summarized": True,
            "summarized_count": len(dropped),
            "summarizer": summarizer_repr,
        }
        if summarizer_result.latency_ms is not None:
            metadata["latency_ms"] = summarizer_result.latency_ms
        if summarizer_result.token_usage is not None:
            metadata["token_usage"] = summarizer_result.token_usage
        if summarizer_result.cost is not None:
            metadata["cost"] = summarizer_result.cost
        if summarizer_result.model is not None:
            metadata["model"] = summarizer_result.model
        if truncated:
            metadata["summary_truncated"] = True

    summary_msg = Message(
        role=Role.USER,
        content=summary_text,
        metadata={
            "llmigrate_synthetic": True,
            "summarized_count": len(dropped),
            **({"summary_truncated": True} if truncated else {}),
        },
    )

    return CompressionResult(messages=[summary_msg], metadata=metadata)


def transform(messages: list[Message], **params: Any) -> MigrationResult:
    pin_first_user: bool = params.get("pin_first_user", True)
    pinned, rest = split_pinned(messages, pin_first_user=pin_first_user)

    result = compress(rest, **params)

    return MigrationResult(
        messages=pinned + result.messages,
        strategies=[Strategy.SUMMARIZE],
        metadata={"original_count": len(messages), **result.metadata},
    )


def _truncate_to_budget(text: str, max_tokens: int) -> tuple[str, bool]:
    if estimate_tokens(text) <= max_tokens:
        return text, False
    max_chars = max(0, max_tokens * _TRUNCATION_CHARS_PER_TOKEN)
    return text[:max_chars].rstrip() + " [truncated]", True


def _build_summary_prompt(messages: list[Message]) -> list[dict[str, Any]]:
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
