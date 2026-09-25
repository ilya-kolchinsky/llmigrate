"""OpenAI message format adapter.

Converts between OpenAI's list[dict] format and llmigrate's canonical Messages.
This is the most common format and serves as the default.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from llmigrate.types import Message, Role

_ROLE_MAP = {
    "system": Role.SYSTEM,
    "developer": Role.DEVELOPER,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
    "tool": Role.TOOL_RESULT,
    "function": Role.TOOL_RESULT,
}

_REVERSE_ROLE_MAP = {
    Role.SYSTEM: "system",
    Role.DEVELOPER: "developer",
    Role.USER: "user",
    Role.ASSISTANT: "assistant",
    Role.TOOL_CALL: "assistant",
    Role.TOOL_RESULT: "tool",
}

_PROVIDER_META_KEYS = frozenset(
    {
        "tool_calls",
        "function_call",
        "tool_call_id",
        "tool_results",
        "name",
        "anthropic_content",
        "anthropic_text",
        "anthropic_extra",
        "anthropic_role",
        "openai_content",
        "openai_text",
        "openai_extra",
        "openai_legacy_function",
        "anthropic_tool_calls_snapshot",
        "anthropic_tool_results_snapshot",
    }
)

_TEXT_PART_TYPES = {"text", "input_text", "output_text"}


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") in _TEXT_PART_TYPES
            and isinstance(part.get("text"), str)
        )
    return ""


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
        llmigrate_metadata = msg.get("llmigrate")
        if isinstance(llmigrate_metadata, dict):
            metadata.update(deepcopy(llmigrate_metadata))

        if isinstance(content, list):
            metadata["openai_content"] = deepcopy(content)
            text = _text_from_content(content)
            metadata["openai_text"] = text
            content = text

        # Preserve tool call information
        if "tool_calls" in msg:
            metadata["tool_calls"] = deepcopy(msg["tool_calls"])
            role = Role.TOOL_CALL

        if "function_call" in msg:
            metadata["function_call"] = deepcopy(msg["function_call"])
            role = Role.TOOL_CALL

        if "tool_call_id" in msg:
            metadata["tool_call_id"] = msg["tool_call_id"]

        if "name" in msg:
            metadata["name"] = msg["name"]

        if role_str == "function":
            metadata["openai_legacy_function"] = True

        extras = {
            key: value
            for key, value in msg.items()
            if key
            not in {
                "role",
                "content",
                "tool_calls",
                "function_call",
                "tool_call_id",
                "name",
                "llmigrate",
            }
        }
        if extras:
            metadata["openai_extra"] = deepcopy(extras)

        result.append(Message(role=role, content=content, metadata=metadata))
    return result


def to_openai(
    messages: list[Message], *, include_metadata: bool = False
) -> list[dict[str, Any]]:
    """Convert canonical Messages to OpenAI-format dicts."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        anthropic_content = msg.metadata.get("anthropic_content")
        if isinstance(anthropic_content, list):
            unsupported = [
                block.get("type", "<unknown>") if isinstance(block, dict) else "<unknown>"
                for block in anthropic_content
                if not isinstance(block, dict)
                or block.get("type") not in {"text", "tool_use", "tool_result"}
            ]
            if unsupported:
                raise ValueError(
                    "Anthropic content blocks cannot be converted to the OpenAI "
                    "Chat Completions format automatically (unsupported blocks: "
                    f"{', '.join(map(str, unsupported))}). Convert them explicitly "
                    "before calling migrate()."
                )

        role = _REVERSE_ROLE_MAP.get(msg.role, "user")
        original_content = msg.metadata.get("openai_content")
        original_text = msg.metadata.get("openai_text")
        content_unchanged = original_content is not None and msg.content == original_text

        content: Any = msg.content
        if content_unchanged:
            content = original_content
        elif isinstance(original_content, list):
            restored: list[Any] = []
            inserted_text = False
            for part in original_content:
                if isinstance(part, dict) and part.get("type") in _TEXT_PART_TYPES:
                    if not inserted_text and msg.content:
                        restored.append({"type": part["type"], "text": msg.content})
                        inserted_text = True
                else:
                    restored.append(part)
            if msg.content and not inserted_text:
                restored.insert(0, {"type": "text", "text": msg.content})
            content = restored or ""

        base: dict[str, Any] = {"role": role, "content": content}
        if msg.metadata.get("openai_legacy_function") and msg.role == Role.TOOL_RESULT:
            base["role"] = "function"
            base.pop("tool_call_id", None)

        if "tool_calls" in msg.metadata:
            base["tool_calls"] = msg.metadata["tool_calls"]
        if "function_call" in msg.metadata:
            base["function_call"] = msg.metadata["function_call"]

        if "tool_call_id" in msg.metadata:
            base["tool_call_id"] = msg.metadata["tool_call_id"]

        if "name" in msg.metadata:
            base["name"] = msg.metadata["name"]

        output_messages: list[dict[str, Any]] = []
        tool_results = msg.metadata.get("tool_results")
        if msg.role == Role.TOOL_RESULT and isinstance(tool_results, list) and tool_results:
            snapshot = msg.metadata.get("anthropic_tool_results_snapshot")
            results_unchanged = tool_results == snapshot
            original_text = msg.metadata.get("anthropic_text")
            content_unchanged = original_text is None or msg.content == original_text
            if not results_unchanged and len(tool_results) > 1:
                raise ValueError(
                    "A multi-result Anthropic message was edited and cannot be split "
                    "back into OpenAI tool messages unambiguously."
                )
            if not content_unchanged and len(tool_results) > 1:
                raise ValueError(
                    "A multi-result Anthropic message was rewritten and cannot be split "
                    "back into OpenAI tool messages unambiguously."
                )
            for tool_result in tool_results:
                tool_call_id = tool_result.get("tool_call_id")
                if not tool_call_id:
                    raise ValueError("OpenAI tool messages require a tool_call_id")
                raw_content = tool_result.get("raw_content", tool_result.get("content", ""))
                if not _is_openai_tool_content(raw_content):
                    raise ValueError(
                        "Anthropic tool-result content contains blocks unsupported by the "
                        "OpenAI Chat Completions message format. Convert those blocks "
                        "explicitly before calling migrate()."
                    )
                result_content = (
                    msg.content
                    if not content_unchanged and len(tool_results) == 1
                    else tool_result.get("content", "")
                )
                output_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result_content,
                    }
                )
        else:
            if (
                msg.role == Role.TOOL_RESULT
                and not msg.metadata.get("tool_call_id")
                and not msg.metadata.get("openai_legacy_function")
            ):
                raise ValueError("OpenAI tool messages require a tool_call_id")
            output_messages.append(base)

        if include_metadata:
            llm_meta = {
                k: v for k, v in msg.metadata.items() if k not in _PROVIDER_META_KEYS
            }
            if llm_meta:
                for output_message in output_messages:
                    output_message["llmigrate"] = llm_meta

        result.extend(output_messages)
    return result


def _is_openai_tool_content(content: Any) -> bool:
    """OpenAI Chat Completions tool messages accept text content only."""
    if isinstance(content, str):
        return True
    return isinstance(content, list) and all(
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        for block in content
    )
