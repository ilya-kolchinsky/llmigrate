"""Raw strategy — pass conversation unchanged."""

from __future__ import annotations

from typing import Any

from llmigrate.types import Message, MigrationResult, Strategy


def transform(messages: list[Message], **params: Any) -> MigrationResult:
    return MigrationResult(
        messages=list(messages),
        strategies=[Strategy.RAW],
        metadata={"original_count": len(messages)},
    )
