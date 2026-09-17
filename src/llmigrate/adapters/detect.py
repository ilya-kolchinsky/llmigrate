"""Auto-detect message format and convert to canonical."""

from __future__ import annotations

from typing import Any

from llmigrate.adapters.openai import from_openai
from llmigrate.types import Message


def auto_convert(messages: list[dict[str, Any]] | list[Message]) -> list[Message]:
    """Convert messages to canonical format, auto-detecting the input format.

    Accepts:
    - list[Message]: returned as-is
    - list[dict]: treated as OpenAI format (the most common)
    """
    if not messages:
        return []

    first = messages[0]

    if isinstance(first, Message):
        return list(messages)  # type: ignore[arg-type]

    if isinstance(first, dict):
        return from_openai(messages)  # type: ignore[arg-type]

    raise TypeError(
        f"Unsupported message format: {type(first).__name__}. "
        "Expected list[dict] (OpenAI format) or list[Message]."
    )
