"""Anthropic message format adapter.

Converts between Anthropic's Messages API format (content blocks, separate
`system` parameter) and llmigrate's canonical Messages. OpenAI-compatible
self-hosted servers (vLLM, LocalAI, etc.) use the OpenAI wire format, not
this one — see adapters/openai.py for those.
"""

from __future__ import annotations

import json
from typing import Any

from llmigrate.types import Message, Role


def from_anthropic(messages: list[dict[str, Any]], system: str | None = None) -> list[Message]:
    """Convert Anthropic-format messages (+ optional top-level system) to canonical Messages."""
    result: list[Message] = []
    if system:
        result.append(Message(role=Role.SYSTEM, content=system))

    for msg in messages:
        role_str = msg.get("role", "user")
        content = msg.get("content", "")
        metadata: dict[str, Any] = {}

        if isinstance(content, str):
            text = content
            role = Role.ASSISTANT if role_str == "assistant" else Role.USER
        else:
            blocks = content or []
            metadata["anthropic_content"] = blocks
            text = "\n".join(
                b.get("text", "") for b in blocks if b.get("type") == "text" and b.get("text")
            )

            tool_use_blocks = [b for b in blocks if b.get("type") == "tool_use"]
            tool_result_blocks = [b for b in blocks if b.get("type") == "tool_result"]

            if tool_use_blocks:
                role = Role.TOOL_CALL
                metadata["tool_calls"] = [
                    {
                        "id": b.get("id"),
                        "type": "function",
                        "function": {
                            "name": b.get("name"),
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    }
                    for b in tool_use_blocks
                ]
            elif tool_result_blocks:
                role = Role.TOOL_RESULT
                first = tool_result_blocks[0]
                metadata["tool_call_id"] = first.get("tool_use_id")
                result_content = first.get("content", "")
                if isinstance(result_content, list):
                    result_content = "\n".join(
                        b.get("text", "") for b in result_content if isinstance(b, dict)
                    )
                text = text or result_content
            else:
                role = Role.ASSISTANT if role_str == "assistant" else Role.USER

        result.append(Message(role=role, content=text, metadata=metadata))
    return result


def to_anthropic(messages: list[Message]) -> dict[str, Any]:
    """Convert canonical Messages to Anthropic's {"system": ..., "messages": [...]} shape."""
    system_parts = [m.content for m in messages if m.role == Role.SYSTEM]
    system = "\n\n".join(system_parts) if system_parts else None

    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == Role.SYSTEM:
            continue

        if "anthropic_content" in msg.metadata:
            role_str = "assistant" if msg.role in (Role.ASSISTANT, Role.TOOL_CALL) else "user"
            result.append({"role": role_str, "content": msg.metadata["anthropic_content"]})
            continue

        if msg.role == Role.TOOL_CALL:
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for call in msg.metadata.get("tool_calls", []):
                function = call.get("function", {})
                try:
                    tool_input = json.loads(function.get("arguments") or "{}")
                except (TypeError, ValueError):
                    tool_input = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id"),
                        "name": function.get("name"),
                        "input": tool_input,
                    }
                )
            result.append({"role": "assistant", "content": blocks})
        elif msg.role == Role.TOOL_RESULT:
            result.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.metadata.get("tool_call_id"),
                            "content": msg.content,
                        }
                    ],
                }
            )
        else:
            role_str = "assistant" if msg.role == Role.ASSISTANT else "user"
            result.append({"role": role_str, "content": msg.content})

    return {"system": system, "messages": result}
