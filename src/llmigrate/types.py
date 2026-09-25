"""Core types for llmigrate.

The canonical message representation and migration result types.
These are the only types in the public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
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
    STRUCTURED_STATE = "structured_state"
    AUDIT = "audit"
    SELECTIVE_HISTORY = "selective_history"


class StrategyCategory(Enum):
    SELECTION = "selection"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"


@dataclass
class SelectionResult:
    kept: list[Message]
    dropped: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    messages: list[Message]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationResult:
    """Result of a migrate operation.

    ``messages`` contains transformed provider-format items with llmigrate's
    metadata sidecar. Use ``provider_messages`` when sending them to a provider
    API; pass ``system`` separately for formats with a top-level instruction.
    """

    messages: list[Message] | list[dict[str, Any]]
    strategies: list[Strategy]
    metadata: dict[str, Any] = field(default_factory=dict)
    format: str | None = None
    system: str | None = None

    @property
    def original_count(self) -> int:
        return int(self.metadata.get("original_count", 0))

    @property
    def transferred_count(self) -> int:
        return len(self.messages)

    @property
    def provider_messages(self) -> list[dict[str, Any]]:
        """Return wire-format messages without llmigrate's private metadata.

        ``messages`` retains per-item llmigrate metadata for inspection and
        backward compatibility. Provider request schemas generally do not
        accept that extension, so use this property when forwarding the result
        directly to an API. Pass ``system`` as Anthropic's ``system`` argument,
        OpenAI Responses' ``instructions``, or Gemini Interactions'
        ``system_instruction`` when applicable.
        """
        provider_messages: list[dict[str, Any]] = []
        for message in self.messages:
            if not isinstance(message, dict):
                raise TypeError(
                    "provider_messages is available only after wire-format conversion"
                )
            provider_messages.append(
                {key: value for key, value in message.items() if key != "llmigrate"}
            )
        return provider_messages
