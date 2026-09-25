"""Auto-detect message format and convert to canonical."""

from __future__ import annotations

from typing import Any

from llmigrate.adapters.anthropic import from_anthropic
from llmigrate.adapters.gemini_interactions import from_gemini_interactions
from llmigrate.adapters.openai import from_openai
from llmigrate.adapters.openai_responses import from_openai_responses
from llmigrate.types import Message

_ANTHROPIC_BLOCK_TYPES = {
    "text",
    "tool_use",
    "tool_result",
    "image",
    "thinking",
    "redacted_thinking",
    "document",
    "server_tool_use",
    "web_search_tool_use",
    "web_search_tool_result",
    "code_execution_tool_use",
    "code_execution_tool_result",
}
_ANTHROPIC_UNIQUE_BLOCK_TYPES = _ANTHROPIC_BLOCK_TYPES - {"text"}
_OPENAI_UNIQUE_BLOCK_TYPES = {
    "image_url",
    "input_text",
    "input_image",
    "input_audio",
    "output_text",
    "audio",
    "refusal",
}

OPENAI = "openai"
OPENAI_RESPONSES = "openai_responses"
ANTHROPIC = "anthropic"
GEMINI_INTERACTIONS = "gemini_interactions"
CANONICAL = "canonical"

_INPUT_FORMATS = {
    OPENAI,
    OPENAI_RESPONSES,
    ANTHROPIC,
    GEMINI_INTERACTIONS,
    CANONICAL,
}
_GEMINI_STEP_TYPES = {"user_input", "model_output", "function_result"}
_RESPONSES_ITEM_TYPES = {"message", "function_call", "function_call_output"}


def _looks_like_anthropic(messages: list[dict[str, Any]]) -> bool:
    """Distinguish provider block signatures while retaining the text-only default."""
    saw_anthropic_text = False
    saw_openai_unique = False
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        block_types = {
            block.get("type") for block in content if isinstance(block, dict)
        }
        if block_types & _ANTHROPIC_UNIQUE_BLOCK_TYPES:
            return True
        if block_types & _OPENAI_UNIQUE_BLOCK_TYPES:
            saw_openai_unique = True
            continue
        if "text" in block_types:
            saw_anthropic_text = True
    # A list of plain text blocks is ambiguous between the two APIs. Keep the
    # historical Anthropic preference and let callers set input_format when
    # the source is an OpenAI text-part list.
    return saw_anthropic_text and not saw_openai_unique


def detect_format(messages: list[dict[str, Any]] | list[Message]) -> str:
    """Detect the format of the input messages.

    Returns a supported provider format or ``"canonical"``.
    """
    if not messages:
        return OPENAI
    first = messages[0]
    if isinstance(first, Message):
        return CANONICAL
    if isinstance(first, dict):
        item_types = {item.get("type") for item in messages if isinstance(item, dict)}
        if item_types & _GEMINI_STEP_TYPES:
            return GEMINI_INTERACTIONS
        if "function_call" in item_types:
            function_calls = [
                item for item in messages
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]
            if any("id" in item and "call_id" not in item for item in function_calls):
                return GEMINI_INTERACTIONS
        if item_types & _RESPONSES_ITEM_TYPES:
            return OPENAI_RESPONSES
        if _looks_like_anthropic(messages):  # type: ignore[arg-type]
            return ANTHROPIC
        return OPENAI
    return OPENAI


def auto_convert(
    messages: list[dict[str, Any]] | list[Message], *, input_format: str | None = None
) -> list[Message]:
    """Convert messages to canonical format, auto-detecting the input format.

    Accepts:
    - list[Message]: returned as-is
    - list[dict] with content-block lists (Anthropic's signature): from_anthropic
    - list[dict] with string content (OpenAI format, the most common,
      including OpenAI-compatible endpoints like vLLM): from_openai
    """
    if input_format not in {None, *_INPUT_FORMATS}:
        raise ValueError(
            f"Unknown input_format: {input_format!r}. Must be one of "
            f"{', '.join(sorted(_INPUT_FORMATS))}."
        )

    if not messages:
        return []

    first = messages[0]

    if isinstance(first, Message):
        if input_format not in {None, CANONICAL}:
            raise ValueError(
                "input_format must be 'canonical' when messages contain Message objects"
            )
        return list(messages)  # type: ignore[arg-type]

    if isinstance(first, dict):
        if input_format == CANONICAL:
            raise ValueError("input_format='canonical' requires llmigrate.Message objects")
        resolved_format = input_format or detect_format(messages)
        if resolved_format == ANTHROPIC:
            return from_anthropic(messages)  # type: ignore[arg-type]
        if resolved_format == OPENAI_RESPONSES:
            return from_openai_responses(messages)  # type: ignore[arg-type]
        if resolved_format == GEMINI_INTERACTIONS:
            return from_gemini_interactions(messages)  # type: ignore[arg-type]
        return from_openai(messages)  # type: ignore[arg-type]

    raise TypeError(
        f"Unsupported message format: {type(first).__name__}. "
        "Expected list[dict] (OpenAI/Anthropic format) or list[Message]."
    )
