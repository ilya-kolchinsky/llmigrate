"""Auto-detect message format and convert to canonical."""

from __future__ import annotations

from typing import Any

from llmigrate.adapters.anthropic import from_anthropic
from llmigrate.adapters.openai import from_openai
from llmigrate.types import Message

_ANTHROPIC_BLOCK_TYPES = {"text", "tool_use", "tool_result", "image"}


def _looks_like_anthropic(messages: list[dict[str, Any]]) -> bool:
    """Anthropic messages use content-block lists; OpenAI messages use plain
    string content (or a null/omitted content alongside tool_calls)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") in _ANTHROPIC_BLOCK_TYPES for b in content
        ):
            return True
    return False


def auto_convert(messages: list[dict[str, Any]] | list[Message]) -> list[Message]:
    """Convert messages to canonical format, auto-detecting the input format.

    Accepts:
    - list[Message]: returned as-is
    - list[dict] with content-block lists (Anthropic's signature): from_anthropic
    - list[dict] with string content (OpenAI format, the most common,
      including OpenAI-compatible endpoints like vLLM): from_openai
    """
    if not messages:
        return []

    first = messages[0]

    if isinstance(first, Message):
        return list(messages)  # type: ignore[arg-type]

    if isinstance(first, dict):
        if _looks_like_anthropic(messages):  # type: ignore[arg-type]
            return from_anthropic(messages)  # type: ignore[arg-type]
        return from_openai(messages)  # type: ignore[arg-type]

    raise TypeError(
        f"Unsupported message format: {type(first).__name__}. "
        "Expected list[dict] (OpenAI/Anthropic format) or list[Message]."
    )
