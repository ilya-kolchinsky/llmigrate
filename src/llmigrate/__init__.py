"""llmigrate — Cross-model session migration for LLM conversations."""

from __future__ import annotations

from collections import Counter
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
from llmigrate.pinning import split_pinned
from llmigrate.selectors import PrioritySelector, RelevanceSelector, Selector
from llmigrate.strategies import (
    COMPRESS_REGISTRY,
    SELECT_REGISTRY,
    STRATEGY_CATEGORIES,
    STRATEGY_REGISTRY,
    VALIDATE_REGISTRY,
)
from llmigrate.summarizers import GenerateSummarizer, Summarizer, SummarizerResult, TokenUsage
from llmigrate.tokens import default_tokenizer, estimate_message_tokens
from llmigrate.types import (
    CompressionResult,
    Message,
    MigrationResult,
    Role,
    SelectionResult,
    Strategy,
    StrategyCategory,
    ValidationResult,
)

__all__ = [
    "ValidationResult",
    "GenerateSummarizer",
    "Message",
    "PrioritySelector",
    "RelevanceSelector",
    "Role",
    "SelectionResult",
    "Selector",
    "Strategy",
    "StrategyCategory",
    "Summarizer",
    "SummarizerResult",
    "TokenUsage",
    "MigrationResult",
    "CompressionResult",
    "async_openai_compatible_generate",
    "openai_compatible_generate",
    "register_model_context_window",
    "migrate",
]

_VALID_FORMATS = {_OPENAI, _ANTHROPIC}


def migrate(
    messages: list[dict[str, Any]] | list[Message],
    strategy: str | None = None,
    *,
    strategies: list[str] | None = None,
    generate: Callable[..., str] | None = None,
    generate_kwargs: dict[str, Any] | None = None,
    source_model: str | None = None,
    target_model: str | None = None,
    input_format: str | None = None,
    target_format: str | None = None,
    system: str | None = None,
    enforce_alternation: bool = True,
    **params: Any,
) -> MigrationResult:
    resolved = _resolve_strategies(strategy, strategies)

    _validate_messages_input(messages)

    detected = input_format or detect_format(messages)
    if detected not in {_OPENAI, _ANTHROPIC, _CANONICAL}:
        raise ValueError(
            f"Unknown input_format: {detected!r}. Must be 'openai', 'anthropic', or 'canonical'."
        )
    resolved_format = target_format or (detected if detected != _CANONICAL else _OPENAI)
    if resolved_format not in _VALID_FORMATS:
        raise ValueError(
            f"Unknown target_format: {resolved_format!r}. Must be 'openai' or 'anthropic'."
        )

    canonical = auto_convert(messages, input_format=input_format)
    if system is not None:
        canonical = [Message(role=Role.SYSTEM, content=system), *canonical]
    canonical = _validate_canonical(canonical)

    if generate is not None:
        params["generate"] = generate
    if generate_kwargs is not None:
        params["generate_kwargs"] = generate_kwargs
    if source_model is not None:
        params["source_model"] = source_model
    if target_model is not None:
        params["target_model"] = target_model

    if not resolved:
        transform_fn = STRATEGY_REGISTRY["raw"]
        result = cast(MigrationResult, transform_fn(canonical, **params))  # type: ignore[operator]
    elif strategy is not None:
        transform_fn = STRATEGY_REGISTRY[resolved[0]]
        result = cast(MigrationResult, transform_fn(canonical, **params))  # type: ignore[operator]
    else:
        result = _execute_pipeline(canonical, resolved, params)

    if source_model is not None:
        result.metadata.setdefault("source_model", source_model)
    if target_model is not None:
        result.metadata.setdefault("target_model", target_model)

    if enforce_alternation and resolved:
        result.messages = _enforce_alternation(
            cast(list[Message], result.messages)
        )

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


def _resolve_strategies(
    strategy: str | None, strategies: list[str] | None
) -> list[str]:
    if strategy is not None and strategies is not None:
        raise ValueError("Cannot specify both 'strategy' and 'strategies'")

    if strategy is not None:
        if strategy == "raw":
            return []
        if strategy not in STRATEGY_REGISTRY:
            available = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
            raise ValueError(f"Unknown strategy: {strategy!r}. Available: {available}")
        return [strategy]

    if strategies is not None:
        all_known = set(STRATEGY_REGISTRY.keys())
        for s in strategies:
            if s not in all_known:
                available = ", ".join(sorted(all_known - {"raw"}))
                raise ValueError(f"Unknown strategy: {s!r}. Available: {available}")
        if "raw" in strategies:
            if len(strategies) > 1:
                raise ValueError("'raw' cannot be combined with other strategies")
            return []
        _validate_composition(strategies)
        return list(strategies)

    return []


def _validate_composition(strategies: list[str]) -> None:
    category_counts: Counter[StrategyCategory] = Counter()
    for s in strategies:
        cat = STRATEGY_CATEGORIES.get(s)
        if cat is not None:
            category_counts[cat] += 1

    for cat, count in category_counts.items():
        if count > 1:
            offenders = [s for s in strategies if STRATEGY_CATEGORIES.get(s) == cat]
            raise ValueError(
                f"At most one {cat.value} strategy allowed, got: {offenders}"
            )


def _execute_pipeline(
    canonical: list[Message],
    strategy_names: list[str],
    params: dict[str, Any],
) -> MigrationResult:
    pin_first_user: bool = params.get("pin_first_user", True)
    pinned, rest = split_pinned(canonical, pin_first_user=pin_first_user)

    by_category: dict[StrategyCategory, str] = {}
    for s in strategy_names:
        cat = STRATEGY_CATEGORIES[s]
        by_category[cat] = s

    applied: list[Strategy] = []
    metadata: dict[str, Any] = {"original_count": len(canonical)}

    selection_name = by_category.get(StrategyCategory.SELECTION)
    transform_name = by_category.get(StrategyCategory.TRANSFORMATION)
    if selection_name:
        select_fn = SELECT_REGISTRY[selection_name]
        tokenizer = params.get("tokenizer") or default_tokenizer(params.get("target_model"))
        pinned_tokens = sum(estimate_message_tokens(m, tokenizer) for m in pinned)
        sel_result = cast(
            SelectionResult,
            select_fn(rest, _pinned_tokens=pinned_tokens, **params),  # type: ignore[operator]
        )
        kept = sel_result.kept
        dropped = sel_result.dropped
        applied.append(Strategy(selection_name))
        metadata["selection"] = sel_result.metadata
    else:
        # With no selector, a transformation consumes the full history. A
        # validation-only pipeline (for example strategies=["audit"]) must
        # instead preserve the full history and pass it to validation.
        kept = [] if transform_name else list(rest)
        dropped = list(rest) if transform_name else []

    if transform_name:
        compress_fn = COMPRESS_REGISTRY[transform_name]
        compress_params = dict(params)
        if transform_name == "structured_state":
            compress_params["_pinned"] = pinned
        transformed = cast(
            CompressionResult,
            compress_fn(dropped, **compress_params),  # type: ignore[operator]
        )
        transform_msgs = transformed.messages
        applied.append(Strategy(transform_name))
        metadata["transformation"] = transformed.metadata
    else:
        transform_msgs = []

    if selection_name is None and transform_name is None:
        # Validation-only pipelines must not reorder explicitly pinned messages.
        assembled = list(canonical)
    else:
        assembled = pinned + transform_msgs + kept

    validate_name = by_category.get(StrategyCategory.VALIDATION)
    if validate_name:
        validate_fn = VALIDATE_REGISTRY[validate_name]
        validated = cast(
            ValidationResult,
            validate_fn(assembled, **params),  # type: ignore[operator]
        )
        assembled = validated.messages
        applied.append(Strategy(validate_name))
        metadata["validation"] = validated.metadata

    return MigrationResult(
        messages=assembled,
        strategies=applied,
        metadata=metadata,
    )


def _validate_messages_input(messages: Any) -> None:
    if not isinstance(messages, list):
        raise TypeError(f"messages must be a list, got {type(messages).__name__}")
    representation: str | None = None
    for i, msg in enumerate(messages):
        if isinstance(msg, Message):
            if representation == "dict":
                raise TypeError("messages cannot mix dicts and llmigrate.Message objects")
            representation = "message"
            continue
        if isinstance(msg, dict):
            if representation == "message":
                raise TypeError("messages cannot mix dicts and llmigrate.Message objects")
            representation = "dict"
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
