"""llmigrate — Cross-model session migration for LLM conversations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from llmigrate.adapters.anthropic import to_anthropic as _to_anthropic
from llmigrate.adapters.detect import ANTHROPIC as _ANTHROPIC
from llmigrate.adapters.detect import CANONICAL as _CANONICAL
from llmigrate.adapters.detect import OPENAI as _OPENAI
from llmigrate.adapters.detect import auto_convert, detect_format
from llmigrate.adapters.openai import to_openai as _to_openai
from llmigrate.alternation import enforce_alternation as _enforce_alternation
from llmigrate.generators import async_openai_compatible_generate, openai_compatible_generate
from llmigrate.models import register_model_context_window
from llmigrate.selectors import PrioritySelector, RelevanceSelector, Selector
from llmigrate.strategies import STRATEGY_REGISTRY
from llmigrate.summarizers import GenerateSummarizer, Summarizer, SummarizerResult, TokenUsage
from llmigrate.types import Message, Role, Strategy, TransferResult

__all__ = [
    "GenerateSummarizer",
    "Message",
    "PrioritySelector",
    "RelevanceSelector",
    "Role",
    "Selector",
    "Strategy",
    "Summarizer",
    "SummarizerResult",
    "TokenUsage",
    "TransferResult",
    "async_openai_compatible_generate",
    "openai_compatible_generate",
    "register_model_context_window",
    "transfer",
]

# Strategies whose contract is "pass through unchanged" — alternation
# enforcement is skipped for them so it can't mask input issues or violate
# the strategy's own spec.
_NO_ALTERNATION_ENFORCEMENT = {"raw"}


_VALID_FORMATS = {_OPENAI, _ANTHROPIC}


def transfer(
    messages: list[dict[str, Any]] | list[Message],
    strategy: str = "raw",
    *,
    generate: Callable[..., str] | None = None,
    generate_kwargs: dict[str, Any] | None = None,
    source_model: str | None = None,
    target_model: str | None = None,
    target_format: str | None = None,
    enforce_alternation: bool = True,
    **params: Any,
) -> TransferResult:
    """Transfer a conversation session using the specified strategy.

    Args:
        messages: Conversation history in any supported format (OpenAI-format
                  or Anthropic-format dicts, or canonical Message objects).
        strategy: Migration strategy name (raw, keep_last, token_budget,
                  summarize, capsule, audit, selective_history, summary_tail).
        generate: Callable for model-assisted strategies. Signature:
                  (messages: list[dict]) -> str. May be async.
        generate_kwargs: Extra kwargs (e.g. temperature) forwarded to `generate`
                  on every call, kept separate from strategy params.
        source_model: Optional model name the conversation started on. Recorded
                  in metadata; used by `audit` to customize its instruction.
        target_model: Optional model name the conversation is moving to. Used
                  by `token_budget`/`summary_tail`/`selective_history` to
                  size defaults from a built-in context-window table, and
                  recorded in metadata.
        target_format: Wire format for the output messages (``"openai"`` or
                  ``"anthropic"``). When omitted, defaults to the detected
                  format of the input (or ``"openai"`` if canonical Message
                  objects are passed).
        enforce_alternation: Merge consecutive same-role messages in the output
                  (default True) so results are safe for providers that reject
                  non-alternating turns (e.g. Anthropic). Skipped for `raw`.
        **params: Strategy-specific parameters (e.g. n=5 for keep_last).

    Returns:
        TransferResult with transformed messages as wire-format dicts,
        ready to pass directly to the target provider's API.
    """
    if strategy not in STRATEGY_REGISTRY:
        available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unknown strategy: {strategy!r}. Available: {available}")

    _validate_messages_input(messages)

    detected = detect_format(messages)
    resolved_format = target_format or (detected if detected != _CANONICAL else _OPENAI)
    if resolved_format not in _VALID_FORMATS:
        raise ValueError(
            f"Unknown target_format: {resolved_format!r}. Must be 'openai' or 'anthropic'."
        )

    canonical = _validate_canonical(auto_convert(messages))

    if generate is not None:
        params["generate"] = generate
    if generate_kwargs is not None:
        params["generate_kwargs"] = generate_kwargs
    if source_model is not None:
        params["source_model"] = source_model
    if target_model is not None:
        params["target_model"] = target_model

    transform_fn = STRATEGY_REGISTRY[strategy]
    result = cast(TransferResult, transform_fn(canonical, **params))  # type: ignore[operator]

    if source_model is not None:
        result.metadata.setdefault("source_model", source_model)
    if target_model is not None:
        result.metadata.setdefault("target_model", target_model)

    if enforce_alternation and strategy not in _NO_ALTERNATION_ENFORCEMENT:
        result.messages = _enforce_alternation(
            cast(list[Message], result.messages)
        )

    # Convert to wire format
    canonical_out = cast(list[Message], result.messages)
    if resolved_format == _ANTHROPIC:
        anthropic_out = _to_anthropic(canonical_out, include_metadata=True)
        result.messages = anthropic_out["messages"]
        result.system = anthropic_out["system"]
    else:
        result.messages = _to_openai(canonical_out, include_metadata=True)
        result.system = None
    result.format = resolved_format

    return result


def _validate_messages_input(messages: Any) -> None:
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    for i, msg in enumerate(messages):
        if isinstance(msg, Message):
            continue
        if isinstance(msg, dict):
            if "role" not in msg:
                raise ValueError(f"messages[{i}] is missing required key 'role'")
            continue
        raise TypeError(
            f"messages[{i}] has unsupported type {type(msg).__name__}; "
            "expected dict or llmigrate.Message"
        )


def _validate_canonical(messages: list[Message]) -> list[Message]:
    for i, msg in enumerate(messages):
        if not isinstance(msg.content, str):
            raise ValueError(
                f"messages[{i}].content must be a string, got {type(msg.content).__name__}"
            )
    return messages
