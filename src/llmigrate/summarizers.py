"""Pluggable summarizer abstraction with cost/latency accounting, used by the
summary_tail strategy (and available to summarize's callers) for
model-assisted prefix compression.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, cast

from llmigrate._util import call_generate
from llmigrate.tokens import estimate_tokens


@dataclass
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class SummarizerResult:
    text: str
    latency_ms: float | None = None
    token_usage: TokenUsage | None = None
    cost: float | None = None
    model: str | None = None


class Summarizer(Protocol):
    def summarize(self, messages: list[dict[str, Any]]) -> SummarizerResult: ...


class GenerateSummarizer:
    """Wraps a plain `generate` callable (sync or async) as a Summarizer.

    Measures wall-clock latency and estimates token_usage via the char-based
    heuristic; cost/model are left None since a bare callable can't self-report
    them — implement Summarizer directly against your provider client for real
    cost accounting.
    """

    def __init__(
        self, generate: Callable[..., str], generate_kwargs: dict[str, Any] | None = None
    ):
        self.generate = generate
        self.generate_kwargs = generate_kwargs or {}

    def summarize(self, messages: list[dict[str, Any]]) -> SummarizerResult:
        input_tokens = sum(estimate_tokens(m.get("content") or "") for m in messages)
        start = time.monotonic()
        text = call_generate(self.generate, messages, **self.generate_kwargs)
        latency_ms = (time.monotonic() - start) * 1000
        return SummarizerResult(
            text=text,
            latency_ms=latency_ms,
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=estimate_tokens(text)),
        )

    def __repr__(self) -> str:
        name = getattr(self.generate, "__name__", repr(self.generate))
        return f"GenerateSummarizer({name})"


def as_summarizer(
    summarizer: Any, generate_kwargs: dict[str, Any] | None = None
) -> Summarizer:
    """Return `summarizer` as-is if it already implements Summarizer, else wrap
    a plain `generate` callable in GenerateSummarizer."""
    if summarizer is None:
        raise ValueError("a 'summarizer' or 'generate' callable is required")
    if hasattr(summarizer, "summarize"):
        return cast(Summarizer, summarizer)
    return GenerateSummarizer(summarizer, generate_kwargs=generate_kwargs)
