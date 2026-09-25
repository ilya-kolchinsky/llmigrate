"""Tests for text and function-call support in Gemini Interactions histories."""

from __future__ import annotations

import pytest

import llmigrate
from llmigrate.adapters.gemini_interactions import (
    from_gemini_interactions,
    to_gemini_interactions,
)
from llmigrate.types import Role


def test_interaction_text_and_function_history_round_trips():
    steps = [
        {"type": "user_input", "content": [{"type": "text", "text": "Check Paris."}]},
        {"type": "function_call", "id": "call_1", "name": "weather", "arguments": {"city": "Paris"}},
        {
            "type": "function_result",
            "call_id": "call_1",
            "name": "weather",
            "result": [{"type": "text", "text": "Sunny."}],
        },
        {"type": "model_output", "content": [{"type": "text", "text": "It is sunny."}]},
    ]

    result = llmigrate.migrate(steps, strategy="raw")

    assert result.format == "gemini_interactions"
    assert result.provider_messages == steps


def test_interaction_adapter_normalizes_function_call_ids():
    steps = [
        {"type": "function_call", "id": "call_2", "name": "search", "arguments": {"q": "x"}},
        {"type": "function_result", "call_id": "call_2", "result": "Found."},
    ]

    canonical = from_gemini_interactions(steps)
    assert canonical[0].role == Role.TOOL_CALL
    assert canonical[1].role == Role.TOOL_RESULT
    assert canonical[1].metadata["tool_call_id"] == "call_2"

    result = llmigrate.migrate(steps, strategy="raw", target_format="openai_responses")
    assert result.provider_messages[0]["call_id"] == "call_2"
    assert result.provider_messages[1]["call_id"] == "call_2"


def test_interaction_system_instruction_is_separate():
    result = llmigrate.migrate(
        [{"type": "user_input", "content": [{"type": "text", "text": "Hello."}]}],
        strategy="raw",
        system="Be helpful.",
    )

    assert result.system == "Be helpful."
    assert result.provider_messages == [
        {"type": "user_input", "content": [{"type": "text", "text": "Hello."}]}
    ]


def test_interaction_text_content_rejects_multimodal_parts():
    with pytest.raises(ValueError, match="text-only Gemini Interactions"):
        from_gemini_interactions(
            [
                {
                    "type": "user_input",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image", "uri": "gs://bucket/image.png"},
                    ],
                }
            ]
        )


def test_interaction_unknown_step_fails_closed():
    with pytest.raises(ValueError, match="unsupported step type 'thought'"):
        from_gemini_interactions([{"type": "thought", "content": "private"}])


def test_interaction_tool_result_requires_call_id():
    with pytest.raises(ValueError, match="require 'call_id'"):
        from_gemini_interactions([{"type": "function_result", "result": "Done."}])


def test_interaction_tool_arguments_must_be_json_object():
    with pytest.raises(ValueError, match="must be a JSON object"):
        to_gemini_interactions(
            [
                from_gemini_interactions(
                    [{"type": "function_call", "id": "call_3", "name": "run", "arguments": []}]
                )[0]
            ]
        )


def test_interaction_function_steps_preserve_unchanged_provider_fields():
    steps = [
        {
            "type": "function_call",
            "id": "call_4",
            "name": "lookup",
            "arguments": {"key": "value"},
            "signature": "provider-signature",
        },
        {
            "type": "function_result",
            "call_id": "call_4",
            "name": "lookup",
            "result": [{"type": "text", "text": "Found it."}],
            "is_error": False,
        },
    ]

    assert to_gemini_interactions(from_gemini_interactions(steps))["input"] == steps


def test_interaction_adapter_preserves_llmigrate_metadata_for_all_step_types():
    steps = [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "Keep this."}],
            "llmigrate": {"pinned": True},
        },
        {
            "type": "function_call",
            "id": "call_5",
            "name": "lookup",
            "arguments": {},
            "llmigrate": {"category": "important"},
        },
        {
            "type": "function_result",
            "call_id": "call_5",
            "result": "Done.",
            "llmigrate": {"pinned": True},
        },
    ]

    result = llmigrate.migrate(steps, strategy="raw")

    assert [step.get("llmigrate") for step in result.messages] == [
        {"pinned": True},
        {"category": "important"},
        {"pinned": True},
    ]
    assert all("llmigrate" not in step for step in result.provider_messages)
