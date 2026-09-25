"""Text and function-call adapter for Gemini Interactions steps."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from llmigrate.types import Message, Role

_TEXT_PART_TYPES = {"text", "input_text", "output_text"}
_PROVIDER_META_KEYS = frozenset(
    {
        "tool_calls",
        "function_call",
        "tool_call_id",
        "tool_results",
        "name",
        "gemini_interactions_step",
        "gemini_interactions_text",
        "gemini_interactions_extra",
        "gemini_interactions_call_snapshot",
        "openai_content",
        "openai_text",
        "openai_extra",
        "openai_legacy_function",
        "anthropic_content",
        "anthropic_text",
        "anthropic_extra",
        "anthropic_role",
        "responses_item",
        "responses_item_id",
        "responses_status",
        "responses_text",
        "responses_output",
        "responses_call_snapshot",
        "openai_responses_extra",
        "openai_responses_role",
    }
)


def _read_llmigrate_metadata(step: dict[str, Any], metadata: dict[str, Any]) -> None:
    llmigrate_metadata = step.get("llmigrate")
    if isinstance(llmigrate_metadata, dict):
        metadata.update(deepcopy(llmigrate_metadata))


def _add_llmigrate_metadata(
    step: dict[str, Any], metadata: dict[str, Any], *, include_metadata: bool
) -> None:
    step.pop("llmigrate", None)
    if not include_metadata:
        return
    llm_meta = {key: value for key, value in metadata.items() if key not in _PROVIDER_META_KEYS}
    if llm_meta:
        step["llmigrate"] = llm_meta


def _text(value: Any, *, location: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError(f"Gemini Interactions {location} must contain text")
    parts: list[str] = []
    for part in value:
        if not isinstance(part, dict) or part.get("type") not in _TEXT_PART_TYPES:
            kind = part.get("type", "<unknown>") if isinstance(part, dict) else "<unknown>"
            raise ValueError(
                "llmigrate currently supports text-only Gemini Interactions content; "
                f"unsupported content part {kind!r}. Convert or remove it explicitly."
            )
        text = part.get("text")
        if not isinstance(text, str):
            raise ValueError("Gemini Interactions text parts require a string 'text'")
        parts.append(text)
    return "\n".join(parts)


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _text(value, location="function_result")
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def from_gemini_interactions(steps: list[dict[str, Any]]) -> list[Message]:
    """Convert materialized Gemini Interactions steps to canonical messages.

    Supports user/model text and client-side function-call/result steps. Server
    interaction IDs and other step types are not portable conversation history.
    """
    result: list[Message] = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise TypeError(f"Gemini Interactions steps[{index}] must be a dictionary")
        step_type = step.get("type")
        if step_type == "user_input":
            content = _text(step.get("content", []), location="user_input")
            metadata = {
                "gemini_interactions_step": deepcopy(step),
                "gemini_interactions_text": content,
            }
            _read_llmigrate_metadata(step, metadata)
            result.append(
                Message(
                    role=Role.USER,
                    content=content,
                    metadata=metadata,
                )
            )
        elif step_type == "model_output":
            content = _text(step.get("content", []), location="model_output")
            metadata = {
                "gemini_interactions_step": deepcopy(step),
                "gemini_interactions_text": content,
            }
            _read_llmigrate_metadata(step, metadata)
            result.append(
                Message(
                    role=Role.ASSISTANT,
                    content=content,
                    metadata=metadata,
                )
            )
        elif step_type == "function_call":
            call_id = step.get("id") or step.get("call_id")
            name = step.get("name")
            arguments = step.get("arguments", {})
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Gemini Interactions function_call steps require 'id'")
            if not isinstance(name, str) or not name:
                raise ValueError("Gemini Interactions function_call steps require 'name'")
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
                "gemini_interactions_step": deepcopy(step),
                "gemini_interactions_call_snapshot": {
                    "id": call_id,
                    "name": name,
                    "arguments": arguments_text,
                },
            }
            _read_llmigrate_metadata(step, metadata)
            result.append(Message(role=Role.TOOL_CALL, content="", metadata=metadata))
        elif step_type == "function_result":
            call_id = step.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ValueError("Gemini Interactions function_result steps require 'call_id'")
            text = _result_text(step.get("result", ""))
            metadata = {
                "tool_call_id": call_id,
                "tool_results": [
                    {
                        "tool_call_id": call_id,
                        "content": text,
                        "raw_content": deepcopy(step.get("result", "")),
                        "name": step.get("name"),
                        "is_error": step.get("is_error"),
                    }
                ],
                "gemini_interactions_step": deepcopy(step),
                "gemini_interactions_text": text,
            }
            _read_llmigrate_metadata(step, metadata)
            result.append(Message(role=Role.TOOL_RESULT, content=text, metadata=metadata))
        else:
            raise ValueError(
                "llmigrate supports Gemini Interactions user_input/model_output text "
                "and function_call/function_result steps only; "
                f"unsupported step type {step_type!r}."
            )
    return result


def _tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function", {})
    call_id = call.get("id") or call.get("call_id")
    name = function.get("name") if isinstance(function, dict) else None
    arguments = function.get("arguments", {}) if isinstance(function, dict) else None
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("Gemini Interactions function calls require a call ID")
    if not isinstance(name, str) or not name:
        raise ValueError("Gemini Interactions function calls require a function name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError as exc:
            raise ValueError(
                f"Tool call {call_id!r} has invalid JSON arguments and cannot be "
                "converted to Gemini Interactions."
            ) from exc
    if not isinstance(arguments, dict):
        raise ValueError("Gemini Interactions function-call arguments must be a JSON object")
    return {"type": "function_call", "id": call_id, "name": name, "arguments": arguments}


def to_gemini_interactions(
    messages: list[Message], *, include_metadata: bool = False
) -> dict[str, Any]:
    """Convert canonical messages to Gemini Interactions ``input`` steps.

    System and developer messages are returned separately as
    ``system_instruction``.
    """
    system_parts = [
        message.content
        for message in messages
        if message.role in (Role.SYSTEM, Role.DEVELOPER)
    ]
    system_instruction = "\n\n".join(system_parts) or None
    steps: list[dict[str, Any]] = []
    for message in messages:
        if message.role in (Role.SYSTEM, Role.DEVELOPER):
            continue
        if message.role == Role.TOOL_CALL:
            calls = message.metadata.get("tool_calls")
            if not isinstance(calls, list) or not calls:
                raise ValueError("Gemini Interactions tool-call messages require tool_calls")
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError("Gemini Interactions tool_calls entries must be objects")
                rendered = _tool_call(call)
                original = message.metadata.get("gemini_interactions_step")
                snapshot = message.metadata.get("gemini_interactions_call_snapshot")
                function = call.get("function", {})
                arguments = function.get("arguments", "{}") if isinstance(function, dict) else "{}"
                arguments_text = (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False)
                )
                unchanged = (
                    isinstance(original, dict)
                    and isinstance(snapshot, dict)
                    and rendered["id"] == snapshot.get("id")
                    and rendered["name"] == snapshot.get("name")
                    and arguments_text == snapshot.get("arguments")
                )
                step = deepcopy(original) if unchanged else rendered
                _add_llmigrate_metadata(
                    step, message.metadata, include_metadata=include_metadata
                )
                steps.append(step)
            continue
        if message.role == Role.TOOL_RESULT:
            original = message.metadata.get("gemini_interactions_step")
            if (
                isinstance(original, dict)
                and original.get("type") == "function_result"
                and message.content == message.metadata.get("gemini_interactions_text")
                and message.metadata.get("tool_call_id") == original.get("call_id")
            ):
                step = deepcopy(original)
                _add_llmigrate_metadata(
                    step, message.metadata, include_metadata=include_metadata
                )
                steps.append(step)
                continue
            results = message.metadata.get("tool_results")
            if isinstance(results, list) and results:
                for tool_result in results:
                    if not isinstance(tool_result, dict):
                        raise ValueError("Gemini Interactions tool_results entries must be objects")
                    call_id = tool_result.get("tool_call_id") or tool_result.get("call_id")
                    if not isinstance(call_id, str) or not call_id:
                        raise ValueError("Gemini Interactions function results require a call ID")
                    output = tool_result.get("content", message.content)
                    if (
                        len(results) == 1
                        and message.content != message.metadata.get("gemini_interactions_text")
                    ):
                        output = message.content
                    step: dict[str, Any] = {
                        "type": "function_result",
                        "call_id": call_id,
                        "result": output if isinstance(output, str) else json.dumps(output),
                    }
                    if isinstance(tool_result.get("name"), str):
                        step["name"] = tool_result["name"]
                    if isinstance(tool_result.get("is_error"), bool):
                        step["is_error"] = tool_result["is_error"]
                    _add_llmigrate_metadata(
                        step, message.metadata, include_metadata=include_metadata
                    )
                    steps.append(step)
            else:
                call_id = message.metadata.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError("Gemini Interactions function results require a call ID")
                step = {
                    "type": "function_result",
                    "call_id": call_id,
                    "result": message.content,
                }
                _add_llmigrate_metadata(
                    step, message.metadata, include_metadata=include_metadata
                )
                steps.append(step)
            continue

        step_type = "user_input" if message.role == Role.USER else "model_output"
        original = message.metadata.get("gemini_interactions_step")
        if (
            isinstance(original, dict)
            and original.get("type") == step_type
            and message.content == message.metadata.get("gemini_interactions_text")
        ):
            step = deepcopy(original)
        else:
            step = {
                "type": step_type,
                "content": [{"type": "text", "text": message.content}],
            }
            extras = message.metadata.get("gemini_interactions_extra")
            if isinstance(extras, dict):
                step.update(deepcopy(extras))
        _add_llmigrate_metadata(step, message.metadata, include_metadata=include_metadata)
        steps.append(step)

    return {"system_instruction": system_instruction, "input": steps}
