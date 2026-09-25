"""Anthropic Messages API adapter.

Converts between Anthropic's content-block messages and llmigrate's canonical
messages. Untouched provider blocks are retained verbatim. When a strategy
changes message text, non-text blocks are kept and the updated text is emitted
as a new text block.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from llmigrate.types import Message, Role

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
        "openai_content",
        "openai_text",
        "openai_extra",
        "openai_legacy_function",
        "anthropic_tool_calls_snapshot",
        "anthropic_tool_results_snapshot",
        "anthropic_role",
    }
)
_SYSTEM_ROLES = (Role.SYSTEM, Role.DEVELOPER)
_OPENAI_TEXT_PART_TYPES = {"text", "input_text", "output_text"}


def _anthropic_role(role: Role) -> str:
    return "assistant" if role in (Role.ASSISTANT, Role.TOOL_CALL) else "user"


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def from_anthropic(messages: list[dict[str, Any]], system: str | None = None) -> list[Message]:
    """Convert Anthropic messages and their separate system prompt to canonical messages."""
    result: list[Message] = []
    if system:
        result.append(Message(role=Role.SYSTEM, content=system))

    for msg in messages:
        role_str = msg.get("role", "user")
        if role_str not in {"user", "assistant"}:
            raise ValueError(f"Unknown Anthropic role: {role_str!r}")
        content = msg.get("content", "")
        metadata: dict[str, Any] = {}
        llmigrate_metadata = msg.get("llmigrate")
        if isinstance(llmigrate_metadata, dict):
            metadata.update(deepcopy(llmigrate_metadata))

        if isinstance(content, str):
            text = content
            role = Role.ASSISTANT if role_str == "assistant" else Role.USER
            metadata["anthropic_role"] = role_str
        else:
            blocks = deepcopy(content) if isinstance(content, list) else []
            metadata["anthropic_content"] = blocks
            metadata["anthropic_role"] = role_str
            text_parts = [
                block["text"]
                for block in blocks
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            tool_use_blocks = [
                block for block in blocks
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
            tool_result_blocks = [
                block for block in blocks
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
            if tool_use_blocks and tool_result_blocks:
                raise ValueError(
                    "An Anthropic message cannot contain both tool_use and tool_result "
                    "blocks in this canonical representation."
                )

            if tool_use_blocks:
                role = Role.TOOL_CALL
                metadata["tool_calls"] = [
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                    for block in tool_use_blocks
                ]
                metadata["anthropic_tool_calls_snapshot"] = deepcopy(
                    metadata["tool_calls"]
                )
                text = "\n".join(text_parts)
            elif tool_result_blocks:
                role = Role.TOOL_RESULT
                tool_results = [
                    {
                        "tool_call_id": block.get("tool_use_id"),
                        "content": _tool_result_text(block.get("content", "")),
                        "raw_content": block.get("content", ""),
                    }
                    for block in tool_result_blocks
                ]
                metadata["tool_results"] = tool_results
                metadata["anthropic_tool_results_snapshot"] = deepcopy(tool_results)
                if len(tool_results) == 1:
                    metadata["tool_call_id"] = tool_results[0]["tool_call_id"]
                result_texts = [item["content"] for item in tool_results if item["content"]]
                text = "\n".join([*text_parts, *result_texts])
            else:
                role = Role.ASSISTANT if role_str == "assistant" else Role.USER
                text = "\n".join(text_parts)

            metadata["anthropic_text"] = text
            extras = {
                key: deepcopy(value)
                for key, value in msg.items()
                if key not in {"role", "content", "llmigrate"}
            }
            if extras:
                metadata["anthropic_extra"] = extras

        result.append(Message(role=role, content=text, metadata=metadata))
    return result


def _text_block(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}] if text else []


def _updated_tool_content(content: Any, text: str) -> Any:
    """Replace tool-result text while retaining nested media blocks."""
    if isinstance(content, str):
        return text
    if not isinstance(content, list):
        return text

    result: list[Any] = []
    inserted_text = False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            if text and not inserted_text:
                result.append({"type": "text", "text": text})
                inserted_text = True
        else:
            result.append(block)
    if text and not inserted_text:
        result.insert(0, {"type": "text", "text": text})
    return result


def _render_tool_use(call: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise ValueError("Anthropic tool_use blocks require tool-call objects")
    function = call.get("function", {})
    if not isinstance(function, dict):
        raise ValueError("Anthropic tool_use blocks require function objects")
    if not call.get("id") or not isinstance(function.get("name"), str):
        raise ValueError("Anthropic tool_use blocks require a tool-call ID and function name")
    arguments = function.get("arguments") or "{}"
    if isinstance(arguments, dict):
        tool_input = arguments
    else:
        try:
            tool_input = json.loads(arguments)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Tool call {call.get('id')!r} has invalid JSON arguments and cannot "
                "be converted to Anthropic format."
            ) from exc
    if not isinstance(tool_input, dict):
        raise ValueError(
            f"Tool call {call.get('id')!r} arguments must decode to a JSON object."
        )
    return {
        "type": "tool_use",
        "id": call.get("id"),
        "name": function.get("name"),
        "input": tool_input,
    }


def _changed_content_blocks(msg: Message) -> list[dict[str, Any]]:
    """Rebuild edited content while retaining supported blocks in their order."""
    original = msg.metadata.get("anthropic_content")
    original_blocks = original if isinstance(original, list) else []
    if any(
        isinstance(block, dict)
        and block.get("type") in {"thinking", "redacted_thinking"}
        for block in original_blocks
    ):
        raise ValueError(
            "This Anthropic message contains signed thinking blocks that cannot be "
            "safely rewritten. Preserve the message unchanged or remove those blocks "
            "explicitly before calling migrate()."
        )

    if msg.role == Role.TOOL_CALL:
        calls = msg.metadata.get("tool_calls", [])
        if not isinstance(calls, list) or not calls:
            raise ValueError("Anthropic assistant tool calls require at least one tool call")
        rendered_calls = [_render_tool_use(call) for call in calls]
        return _replace_blocks(original_blocks, msg.content, "tool_use", rendered_calls)

    if msg.role == Role.TOOL_RESULT:
        tool_results = msg.metadata.get("tool_results")
        if isinstance(tool_results, list) and tool_results:
            snapshot = msg.metadata.get("anthropic_tool_results_snapshot")
            content_changed = msg.content != msg.metadata.get("anthropic_text")
            results_changed = tool_results != snapshot
            if content_changed and len(tool_results) > 1:
                raise ValueError(
                    "A multi-result Anthropic message was rewritten and cannot be split "
                    "back into tool_result blocks unambiguously."
                )
            rendered_results = []
            for item in tool_results:
                if not isinstance(item, dict):
                    raise ValueError("Anthropic tool_result metadata must contain objects")
                result_content = item.get("raw_content", item.get("content", ""))
                if content_changed and len(tool_results) == 1:
                    result_content = _updated_tool_content(result_content, msg.content)
                elif results_changed:
                    result_content = _updated_tool_content(
                        result_content, item.get("content", "")
                    )
                rendered_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": item.get("tool_call_id"),
                        "content": result_content,
                    }
                )
        else:
            rendered_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.metadata.get("tool_call_id"),
                    "content": msg.content,
                }
            ]
        return _replace_blocks(original_blocks, "", "tool_result", rendered_results)

    return _replace_blocks(original_blocks, msg.content)


def _replace_blocks(
    original: list[Any],
    text: str,
    structured_type: str | None = None,
    replacements: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Replace text/structured blocks without reordering opaque content blocks."""
    replacement_blocks = replacements or []
    replacement_index = 0
    text_inserted = False
    result: list[dict[str, Any]] = []

    for block in original:
        if not isinstance(block, dict):
            raise ValueError("Anthropic content blocks must be dictionaries")
        block_type = block.get("type")
        if not isinstance(block_type, str):
            raise ValueError("Anthropic content blocks require a string 'type'")
        if block_type == "text":
            if not text_inserted and text:
                result.append({"type": "text", "text": text})
                text_inserted = True
            continue
        if structured_type and block_type == structured_type:
            if replacement_index < len(replacement_blocks):
                result.append(replacement_blocks[replacement_index])
                replacement_index += 1
            continue
        if block_type != structured_type:
            result.append(block)

    if text and not text_inserted:
        result.insert(0, {"type": "text", "text": text})
    result.extend(replacement_blocks[replacement_index:])
    return result


def _as_blocks(content: Any) -> list[Any]:
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str):
        return _text_block(content)
    return []


def _append_message(result: list[dict[str, Any]], message: dict[str, Any]) -> None:
    """Anthropic requires alternating user/assistant messages; preserve blocks when joining."""
    if result and result[-1]["role"] == message["role"]:
        result[-1]["content"] = _as_blocks(result[-1].get("content")) + _as_blocks(
            message.get("content")
        )
        if "llmigrate" in message:
            previous = result[-1].setdefault("llmigrate", {})
            previous.update(message["llmigrate"])
        return
    result.append(message)


def to_anthropic(
    messages: list[Message], *, include_metadata: bool = False
) -> dict[str, Any]:
    """Convert canonical messages to Anthropic's separate system/messages shape."""
    system_parts = [m.content for m in messages if m.role in _SYSTEM_ROLES]
    system = "\n\n".join(system_parts) if system_parts else None

    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role in _SYSTEM_ROLES:
            continue

        if "function_call" in msg.metadata or msg.metadata.get("openai_legacy_function"):
            raise ValueError(
                "Legacy OpenAI function-call messages cannot be converted to Anthropic "
                "tool_use/tool_result blocks without a tool-call ID. Convert them to "
                "tool_calls first."
            )

        original_blocks = msg.metadata.get("anthropic_content")
        unchanged = (
            isinstance(original_blocks, list)
            and msg.content == msg.metadata.get("anthropic_text")
            and _anthropic_role(msg.role) == msg.metadata.get("anthropic_role")
            and msg.metadata.get("tool_calls")
            == msg.metadata.get("anthropic_tool_calls_snapshot")
            and msg.metadata.get("tool_results")
            == msg.metadata.get("anthropic_tool_results_snapshot")
        )

        openai_blocks = msg.metadata.get("openai_content")
        if isinstance(openai_blocks, list) and any(
            not isinstance(block, dict)
            or block.get("type") not in _OPENAI_TEXT_PART_TYPES
            for block in openai_blocks
        ):
            raise ValueError(
                "OpenAI multimodal content cannot be converted to Anthropic automatically; "
                "convert the content blocks before calling migrate()."
            )

        if unchanged:
            role_str = _anthropic_role(msg.role)
            d: dict[str, Any] = {"role": role_str, "content": original_blocks}
        elif msg.role == Role.TOOL_CALL:
            d = {"role": "assistant", "content": _changed_content_blocks(msg)}
        elif msg.role == Role.TOOL_RESULT:
            tool_results = msg.metadata.get("tool_results", [])
            ids = (
                [item.get("tool_call_id") for item in tool_results]
                if isinstance(tool_results, list) and tool_results
                else [msg.metadata.get("tool_call_id")]
            )
            if any(not tool_call_id for tool_call_id in ids):
                raise ValueError("Anthropic tool_result blocks require a tool-call ID")
            d = {"role": "user", "content": _changed_content_blocks(msg)}
        else:
            role_str = _anthropic_role(msg.role)
            content = _changed_content_blocks(msg)
            # Anthropic accepts either a string or a block list. Keep the
            # established string form for ordinary text messages when there
            # were no Anthropic blocks whose order or opaque payloads need
            # preserving. This also covers canonical and OpenAI text input.
            if not isinstance(original_blocks, list):
                content = msg.content
            d = {"role": role_str, "content": content}

        if include_metadata:
            llm_meta = {
                key: value
                for key, value in msg.metadata.items()
                if key not in _PROVIDER_META_KEYS
            }
            if llm_meta:
                d["llmigrate"] = llm_meta

        _append_message(result, d)

    return {"system": system, "messages": result}
