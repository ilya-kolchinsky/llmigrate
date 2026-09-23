"""OpenAI message format adapter.

Converts between OpenAI's list[dict] format and llmigrate's canonical Messages.
This is the most common format and serves as the default.
"""

from __future__ import annotations

from typing import Any

from llmigrate.types import Message, Role

_ROLE_MAP = {
    "system": Role.SYSTEM,
    "developer": Role.SYSTEM,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL_RESULT,
    "function": Role.TOOL_RESULT,
}

_REVERSE_ROLE_MAP = {
    Role.SYSTEM: "system",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL_CALL: "assistant",
    Role.TOOL_RESULT: "tool",
}


def from_openai(messages: list[dict[str, Any]]) -> list[Message]:
    """Convert OpenAI-format messages to canonical Messages."""
    result: list[Message] = []
    for msg in messages:
        role_str = msg.get("role", "user")
        if role_str not in _ROLE_MAP:
            raise ValueError(f"Unknown OpenAI role: {role_str!r}")
        role = _ROLE_MAP[role_str]

        content = msg.get("content", "")
        if content is None:
            content = ""

        metadata: dict[str, Any] = {}

        # Preserve tool call information
        if "tool_calls" in msg:
            metadata["tool_calls"] = msg["tool_calls"]
            role = Role.TOOL_CALL

        if "tool_call_id" in msg:
            metadata["tool_call_id"] = msg["tool_call_id"]

        if "name" in msg:
            metadata["name"] = msg["name"]

        result.append(Message(role=role, content=content, metadata=metadata))
    return result


def to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert canonical Messages to OpenAI-format dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        d: dict[str, Any] = {
            "role": _REVERSE_ROLE_MAP.get(msg.role, "user"),
            "content": msg.content,
        }

        if "tool_calls" in msg.metadata:
            d["tool_calls"] = msg.metadata["tool_calls"]

        if "tool_call_id" in msg.metadata:
            d["tool_call_id"] = msg.metadata["tool_call_id"]

        if "name" in msg.metadata:
            d["name"] = msg.metadata["name"]

        result.append(d)
    return result
