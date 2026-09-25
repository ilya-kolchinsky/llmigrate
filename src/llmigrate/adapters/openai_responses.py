"""Text and function-call adapter for OpenAI Responses input items."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from llmigrate.types import Message, Role

_ROLE_MAP = {
    "system": Role.SYSTEM,
    "developer": Role.DEVELOPER,
    "user": Role.USER,
    "assistant": Role.ASSISTANT,
}
_TEXT_PART_TYPES = {"input_text", "output_text", "text"}
_PROVIDER_META_KEYS = frozenset(
    {
        "tool_calls",
        "function_call",
        "tool_call_id",
        "tool_results",
        "name",
        "responses_item",
        "responses_item_id",
        "responses_status",
        "responses_text",
        "responses_output",
        "responses_call_snapshot",
        "openai_responses_extra",
        "openai_responses_role",
        "openai_content",
        "openai_text",
        "openai_extra",
        "openai_legacy_function",
        "anthropic_content",
        "anthropic_text",
        "anthropic_extra",
        "anthropic_role",
        "gemini_interactions_step",
        "gemini_interactions_call_snapshot",
        "gemini_interactions_text",
        "gemini_interactions_extra",
    }
)


def _read_llmigrate_metadata(item: dict[str, Any], metadata: dict[str, Any]) -> None:
    llmigrate_metadata = item.get("llmigrate")
    if isinstance(llmigrate_metadata, dict):
        metadata.update(deepcopy(llmigrate_metadata))


def _add_llmigrate_metadata(
    item: dict[str, Any], metadata: dict[str, Any], *, include_metadata: bool
) -> None:
    item.pop("llmigrate", None)
    if not include_metadata:
        return
    llm_meta = {key: value for key, value in metadata.items() if key not in _PROVIDER_META_KEYS}
    if llm_meta:
        item["llmigrate"] = llm_meta


def _parts_text(value: Any, *, item_type: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError(f"OpenAI Responses {item_type} content must be text or text parts")
    parts: list[str] = []
    for part in value:
        if not isinstance(part, dict) or part.get("type") not in _TEXT_PART_TYPES:
            kind = part.get("type", "<unknown>") if isinstance(part, dict) else "<unknown>"
            raise ValueError(
                "llmigrate currently supports text-only OpenAI Responses content; "
                f"unsupported content part {kind!r}. Convert or remove it explicitly."
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("OpenAI Responses text content parts require a string 'text'")
        parts.append(text)
    return "\n".join(parts)


def from_openai_responses(items: list[dict[str, Any]]) -> list[Message]:
    """Convert materialized Responses input/output items to canonical messages.

    Supports text message items and function-call/function-call-output items.
    Server-side conversation IDs and other item kinds are not portable history.
    """
    result: list[Message] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"Responses items[{index}] must be a dictionary")
        item_type = item.get("type")

        if item_type in {"function_call", "tool_call"}:
            call_id = item.get("call_id")
            name = item.get("name")
            arguments = item.get("arguments", "{}")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("OpenAI Responses function_call items require 'call_id'")
            if not isinstance(name, str) or not name:
                raise ValueError("OpenAI Responses function_call items require 'name'")
            arguments_text = (
                arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
            )
            metadata = {
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments_text},
                            }
                        ],
                        "responses_item": deepcopy(item),
                        "responses_call_snapshot": {
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments_text,
                        },
                    }
            _read_llmigrate_metadata(item, metadata)
            result.append(Message(role=Role.TOOL_CALL, content="", metadata=metadata))
            continue

        if item_type in {"function_call_output", "tool_result"}:
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("OpenAI Responses function_call_output items require 'call_id'")
            output = item.get("output", "")
            if not isinstance(output, str):
                raise ValueError("Text-only OpenAI Responses tool outputs must be strings")
            metadata = {
                        "tool_call_id": call_id,
                        "responses_item": deepcopy(item),
                        "responses_output": output,
                    }
            _read_llmigrate_metadata(item, metadata)
            result.append(Message(role=Role.TOOL_RESULT, content=output, metadata=metadata))
            continue

        if item_type not in {None, "message"}:
            raise ValueError(
                "llmigrate supports OpenAI Responses text messages and function-call "
                f"items only; unsupported item type {item_type!r}."
            )

        role_name = item.get("role")
        if role_name not in _ROLE_MAP:
            raise ValueError(f"Unknown OpenAI Responses message role: {role_name!r}")
        content = _parts_text(item.get("content", ""), item_type="message")
        extras = {
            key: deepcopy(value)
            for key, value in item.items()
            if key not in {"type", "role", "content", "id", "status", "llmigrate"}
        }
        metadata: dict[str, Any] = {
            "responses_item": deepcopy(item),
            "responses_text": content,
            "openai_responses_role": role_name,
        }
        if "id" in item:
            metadata["responses_item_id"] = item["id"]
        if "status" in item:
            metadata["responses_status"] = item["status"]
        if extras:
            metadata["openai_responses_extra"] = extras
        _read_llmigrate_metadata(item, metadata)
        result.append(Message(role=_ROLE_MAP[role_name], content=content, metadata=metadata))
    return result


def _render_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function", {})
    call_id = call.get("id") or call.get("call_id")
    name = function.get("name") if isinstance(function, dict) else None
    arguments = function.get("arguments", "{}") if isinstance(function, dict) else None
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("OpenAI Responses function calls require a call ID")
    if not isinstance(name, str) or not name:
        raise ValueError("OpenAI Responses function calls require a function name")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def to_openai_responses(
    messages: list[Message], *, include_metadata: bool = False
) -> dict[str, Any]:
    """Convert canonical messages to Responses API ``input`` items.

    System messages are returned separately as ``instructions``.
    """
    instructions = "\n\n".join(
        message.content for message in messages if message.role == Role.SYSTEM
    )
    items: list[dict[str, Any]] = []
    for message in messages:
        if message.role == Role.SYSTEM:
            continue
        if message.role == Role.TOOL_CALL:
            calls = message.metadata.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError("OpenAI Responses tool-call messages require tool_calls")
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("OpenAI Responses tool_calls entries must be objects")
                rendered = _render_tool_call(call)
                original = message.metadata.get("responses_item")
                snapshot = message.metadata.get("responses_call_snapshot")
                unchanged = (
                    isinstance(original, dict)
                    and isinstance(snapshot, dict)
                    and rendered["call_id"] == snapshot.get("call_id")
                    and rendered["name"] == snapshot.get("name")
                    and rendered["arguments"] == snapshot.get("arguments")
                )
                item = deepcopy(original) if unchanged else rendered
                _add_llmigrate_metadata(item, message.metadata, include_metadata=include_metadata)
                items.append(item)
            continue
        if message.role == Role.TOOL_RESULT:
            results = message.metadata.get("tool_results")
            if isinstance(results, list) and results:
                for tool_result in results:
                    if not isinstance(tool_result, dict):
                        raise ValueError("OpenAI Responses tool_results entries must be objects")
                    call_id = tool_result.get("tool_call_id") or tool_result.get("call_id")
                    output = tool_result.get("content", message.content)
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError("OpenAI Responses tool outputs require a call ID")
                    if not isinstance(output, str):
                        raise ValueError("Text-only OpenAI Responses tool outputs must be strings")
                    item = {"type": "function_call_output", "call_id": call_id, "output": output}
                    _add_llmigrate_metadata(
                        item, message.metadata, include_metadata=include_metadata
                    )
                    items.append(item)
            else:
                call_id = message.metadata.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("OpenAI Responses tool outputs require a call ID")
                original = message.metadata.get("responses_item")
                unchanged = (
                    isinstance(original, dict)
                    and original.get("type") == "function_call_output"
                    and original.get("call_id") == call_id
                    and message.content == message.metadata.get("responses_output")
                )
                item = (
                    deepcopy(original)
                    if unchanged
                    else {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": message.content,
                    }
                )
                _add_llmigrate_metadata(item, message.metadata, include_metadata=include_metadata)
                items.append(item)
            continue

        role_name = message.role.value
        if role_name not in _ROLE_MAP:
            raise ValueError(f"OpenAI Responses cannot represent role {role_name!r}")
        original = message.metadata.get("responses_item")
        unchanged = (
            isinstance(original, dict)
            and original.get("role") == role_name
            and message.content == message.metadata.get("responses_text")
        )
        if unchanged:
            item = deepcopy(original)
        else:
            item = {
                "type": "message",
                "role": role_name,
                "content": [{"type": "input_text", "text": message.content}],
            }
            extras = message.metadata.get("openai_responses_extra")
            if isinstance(extras, dict):
                item.update(deepcopy(extras))
        _add_llmigrate_metadata(item, message.metadata, include_metadata=include_metadata)
        items.append(item)

    return {"instructions": instructions or None, "input": items}
