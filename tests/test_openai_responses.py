"""Tests for text and function-call support in OpenAI Responses histories."""

from __future__ import annotations

import pytest

import llmigrate
from llmigrate.adapters.openai_responses import from_openai_responses, to_openai_responses
from llmigrate.types import Message, Role


def test_responses_text_and_tool_history_round_trips():
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Find it."}]},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Searching."}]},
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "search",
            "arguments": '{"query":"llmigrate"}',
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "Found it."},
    ]

    result = llmigrate.migrate(items, strategy="raw")

    assert result.format == "openai_responses"
    assert result.provider_messages == items
    assert result.system is None


def test_responses_adapter_normalizes_tool_call_ids_for_cross_provider_migration():
    items = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Search."}]},
        {"type": "function_call", "call_id": "call_2", "name": "search", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_2", "output": "Done."},
    ]

    canonical = from_openai_responses(items)
    assert canonical[1].role == Role.TOOL_CALL
    assert canonical[2].role == Role.TOOL_RESULT
    assert canonical[2].metadata["tool_call_id"] == "call_2"

    result = llmigrate.migrate(items, strategy="raw", target_format="anthropic")
    assert result.provider_messages[1]["content"][0]["id"] == "call_2"
    assert result.provider_messages[2]["content"][0]["tool_use_id"] == "call_2"


def test_responses_system_instruction_is_separate():
    result = llmigrate.migrate(
        [{"type": "message", "role": "user", "content": "Hello."}],
        strategy="raw",
        input_format="openai_responses",
        system="Be concise.",
    )

    assert result.system == "Be concise."
    assert result.provider_messages == [
        {"type": "message", "role": "user", "content": "Hello."}
    ]


def test_responses_text_content_rejects_multimodal_parts():
    with pytest.raises(ValueError, match="text-only OpenAI Responses"):
        from_openai_responses(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": "https://example.test/a.png"}],
                }
            ]
        )


def test_responses_unknown_item_kind_fails_closed():
    with pytest.raises(ValueError, match="unsupported item type 'reasoning'"):
        from_openai_responses([{"type": "reasoning", "summary": []}])


def test_response_item_round_trip_keeps_message_role_and_item_id_when_unchanged():
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Answer."}],
    }
    assert to_openai_responses(from_openai_responses([item]))["input"] == [item]


def test_responses_tool_items_round_trip_unchanged_provider_fields():
    items = [
        {
            "type": "function_call",
            "id": "fc_item_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": '{ "key": "value" }',
            "status": "completed",
        },
        {
            "type": "function_call_output",
            "id": "fco_item_1",
            "call_id": "call_1",
            "output": "Found it.",
        },
    ]

    assert to_openai_responses(from_openai_responses(items))["input"] == items


def test_responses_adapter_preserves_llmigrate_metadata_for_all_item_types():
    items = [
        {
            "type": "message",
            "role": "user",
            "content": "Keep this.",
            "llmigrate": {"pinned": True},
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "lookup",
            "arguments": "{}",
            "llmigrate": {"category": "important"},
        },
        {
            "type": "function_call_output",
            "call_id": "call_2",
            "output": "Done.",
            "llmigrate": {"pinned": True},
        },
    ]

    result = llmigrate.migrate(items, strategy="raw")

    assert [item.get("llmigrate") for item in result.messages] == [
        {"pinned": True},
        {"category": "important"},
        {"pinned": True},
    ]
    assert all("llmigrate" not in item for item in result.provider_messages)


def test_response_output_requires_call_ids_for_tool_results():
    with pytest.raises(ValueError, match="require 'call_id'"):
        from_openai_responses([{"type": "function_call_output", "output": "Done."}])


def test_response_canonical_text_message_can_be_rendered():
    converted = to_openai_responses([Message(role=Role.USER, content="Hello.")])
    assert converted["input"] == [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello."}]}
    ]
