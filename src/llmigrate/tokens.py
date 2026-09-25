"""Shared token estimation utilities used by budget-based strategies."""

from __future__ import annotations

from collections.abc import Callable

from llmigrate._util import message_to_text
from llmigrate.types import Message


def _char_heuristic(text: str) -> int:
    """Approximate token count. ~4 characters per token is a rough heuristic."""
    return max(1, len(text) // 4)


def default_tokenizer(target_model: str | None = None) -> Callable[[str], int]:
    """Best tokenizer available: tiktoken if installed, else a char heuristic.

    tiktoken is an optional dependency (`pip install llmigrate[tiktoken]`); core
    llmigrate has no required dependencies.
    """
    try:
        import tiktoken
    except ImportError:
        return _char_heuristic

    try:
        encoding = tiktoken.encoding_for_model(target_model) if target_model else None
    except KeyError:
        encoding = None
    if encoding is None:
        encoding = tiktoken.get_encoding("cl100k_base")

    def _tiktoken_count(text: str) -> int:
        return max(1, len(encoding.encode(text)))

    return _tiktoken_count


def estimate_tokens(text: str, tokenizer: Callable[[str], int] | None = None) -> int:
    return (tokenizer or _char_heuristic)(text)


def estimate_message_tokens(
    message: Message, tokenizer: Callable[[str], int] | None = None
) -> int:
    """Estimate text, tool payload, media markers, and per-message framing."""
    return estimate_tokens(message_to_text(message), tokenizer) + 4
