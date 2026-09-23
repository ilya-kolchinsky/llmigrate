"""Pluggable event-retention selectors, used as parameter values by the
selective_history strategy (llmigrate.transfer(..., strategy="selective_history",
selector=PrioritySelector(...))).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from llmigrate.types import Message, Role

_DEFAULT_CATEGORIES: dict[Role, str] = {
    Role.USER: "user_instruction",
    Role.ASSISTANT: "assistant",
    Role.TOOL_CALL: "tool_call",
    Role.TOOL_RESULT: "tool_result",
    Role.SYSTEM: "system",
}


def category_of(message: Message) -> str:
    """The retention category of a message: metadata["category"] if set, else a
    default derived from its role."""
    return str(
        message.metadata.get("category", _DEFAULT_CATEGORIES.get(message.role, message.role.value))
    )


class Selector(Protocol):
    def score(self, messages: list[Message]) -> list[float]: ...


class PrioritySelector:
    """Deterministic selection based on event category, e.g.:

        PrioritySelector(priorities={
            "user_instruction": 100,
            "tool_result": 80,
            "assistant": 20,
        })

    Events with higher priority are retained first. Categories not present in
    `priorities` score 0. recency_tiebreak (default True) adds a small
    index-scaled epsilon so more recent events outrank equal-priority older ones.
    """

    def __init__(self, priorities: dict[str, float], recency_tiebreak: bool = True):
        self.priorities = priorities
        self.recency_tiebreak = recency_tiebreak

    def score(self, messages: list[Message]) -> list[float]:
        n = len(messages)
        scores = []
        for i, msg in enumerate(messages):
            base = self.priorities.get(category_of(msg), 0.0)
            tiebreak = (i / n) * 1e-6 if self.recency_tiebreak and n > 0 else 0.0
            scores.append(base + tiebreak)
        return scores

    def __repr__(self) -> str:
        return f"PrioritySelector(priorities={self.priorities!r})"


class RelevanceSelector:
    """Select events by semantic relevance to a task/query, via a pluggable
    embedding function — llmigrate binds to no specific embedding provider.

        RelevanceSelector(embed=my_embed_fn, query="fix the failing test")

    `embed` takes a list of strings and returns a list of equal-length
    embedding vectors. If `query` is omitted (or a callable), it's resolved
    against the candidate list passed to score() — by default, the first
    message in that list.
    """

    def __init__(
        self,
        embed: Callable[[list[str]], list[list[float]]],
        query: str | Callable[[list[Message]], str] | None = None,
    ):
        self.embed = embed
        self.query = query

    def _resolve_query(self, messages: list[Message]) -> str:
        if callable(self.query):
            return self.query(messages)
        if isinstance(self.query, str):
            return self.query
        return messages[0].content if messages else ""

    def score(self, messages: list[Message]) -> list[float]:
        if not messages:
            return []
        query_text = self._resolve_query(messages)
        vectors = self.embed([query_text, *[m.content for m in messages]])
        query_vec, *msg_vecs = vectors
        return [_cosine_similarity(query_vec, v) for v in msg_vecs]

    def __repr__(self) -> str:
        return "RelevanceSelector(...)"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot: float = sum(x * y for x, y in zip(a, b))
    norm_a: float = sum(x * x for x in a) ** 0.5
    norm_b: float = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
