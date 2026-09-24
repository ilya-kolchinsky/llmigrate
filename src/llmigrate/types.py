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

    Contains the transformed messages in wire format (OpenAI or Anthropic
    dicts), ready to pass directly to the target provider's API.

    Attributes:
        messages: Conversation messages in the target wire format.
            For OpenAI format, system messages are included in the list.
            For Anthropic format, system messages are extracted into ``system``.
        strategy: Which strategy was applied.
        metadata: Transformation details (original_count, strategy-specific info).
        format: Wire format of the output (``"openai"`` or ``"anthropic"``).
        system: System prompt content, populated for Anthropic format
            (where the system message is a separate API parameter).
            ``None`` for OpenAI format.
    """

    messages: list[Message] | list[dict[str, Any]]
    strategy: Strategy
    metadata: dict[str, Any] = field(default_factory=dict)
    format: str | None = None
    system: str | None = None

    @property
    def original_count(self) -> int:
        return int(self.metadata.get("original_count", 0))

    @property
    def transferred_count(self) -> int:
        return len(self.messages)
