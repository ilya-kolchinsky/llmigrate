"""Core types for llmigrate.

The canonical message representation and transfer result types.
These are the only types in the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


@dataclass
class Message:
    """Canonical message representation.

    Provider-agnostic internal format. Adapters convert to/from provider-native
    formats (OpenAI, Anthropic, plain text, framework types).
    """

    role: Role
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            role=Role(d["role"]),
            content=d["content"],
            metadata=d.get("metadata", {}),
        )


class Strategy(Enum):
    RAW = "raw"
    KEEP_LAST = "keep_last"
    TOKEN_BUDGET = "token_budget"
    SUMMARIZE = "summarize"
    CAPSULE = "capsule"
    AUDIT = "audit"
    SELECTIVE_HISTORY = "selective_history"
    SUMMARY_TAIL = "summary_tail"


@dataclass
class TransferResult:
    """Result of a transfer operation.

    Contains the transformed messages ready for the target model,
    plus metadata about the transformation that was applied.
    """

    messages: list[Message]
    strategy: Strategy
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def original_count(self) -> int:
        return int(self.metadata.get("original_count", 0))

    @property
    def transferred_count(self) -> int:
        return len(self.messages)

    def to_openai(self) -> list[dict[str, Any]]:
        """Convert the transferred messages to OpenAI-format dicts."""
        from llmigrate.adapters.openai import to_openai

        return to_openai(self.messages)

    def to_anthropic(self) -> dict[str, Any]:
        """Convert the transferred messages to Anthropic's {"system", "messages"} shape."""
        from llmigrate.adapters.anthropic import to_anthropic

        return to_anthropic(self.messages)
