"""llmigrate — Cross-model session migration for LLM conversations."""

from __future__ import annotations

from typing import Any, Callable

from llmigrate.adapters.detect import auto_convert
from llmigrate.strategies import STRATEGY_REGISTRY
from llmigrate.types import Message, Role, Strategy, TransferResult

__all__ = [
    "Message",
    "Role",
    "Strategy",
    "TransferResult",
    "transfer",
]


def transfer(
    messages: list[dict[str, Any]] | list[Message],
    strategy: str = "raw",
    *,
    generate: Callable[..., str] | None = None,
    **params: Any,
) -> TransferResult:
    """Transfer a conversation session using the specified strategy.

    Args:
        messages: Conversation history in any supported format (OpenAI-format
                  dicts or canonical Message objects).
        strategy: Migration strategy name (raw, keep_last, token_budget,
                  summarize, capsule, audit).
        generate: Callable for model-assisted strategies. Signature:
                  (messages: list[dict]) -> str.
        **params: Strategy-specific parameters (e.g., n=5 for keep_last).

    Returns:
        TransferResult with transformed messages (canonical Message objects)
        and metadata. Use adapters (e.g., adapters.to_openai()) to convert
        the result to a provider-native format.
    """
    if strategy not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy: {strategy!r}. Available: {available}")

    canonical = auto_convert(messages)

    if generate is not None:
        params["generate"] = generate

    transform_fn = STRATEGY_REGISTRY[strategy]
    return transform_fn(canonical, **params)  # type: ignore[operator]
