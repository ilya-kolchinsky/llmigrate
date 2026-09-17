"""Raw strategy — pass conversation unchanged."""

from __future__ import annotations

from typing import Any

from llmigrate.types import Message, Strategy, TransferResult


def transform(messages: list[Message], **params: Any) -> TransferResult:
    return TransferResult(
        messages=list(messages),
        strategy=Strategy.RAW,
        metadata={"original_count": len(messages)},
    )
