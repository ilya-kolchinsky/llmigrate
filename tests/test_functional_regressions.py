"""Regression coverage for provider-safe context migration."""

from __future__ import annotations

import pytest

import llmigrate
from llmigrate.adapters.anthropic import from_anthropic, to_anthropic
from llmigrate.types import Message, Role


def test_validation_only_pipeline_preserves_full_history_and_provider_payload():
    messages = [
        {"role": "system", "content": "System instructions."},
        {"role": "developer", "content": "Use concise answers."},
        {"role": "user", "content": "Start the task."},
        {"role": "assistant", "content": "I will begin."},
    ]

    result = llmigrate.migrate(messages, strategies=["audit"])

    assert result.provider_messages[: len(messages)] == messages
    assert result.messages[-1]["llmigrate"]["audit_instruction"] is True
    assert "llmigrate" not in result.provider_messages[-1]
    # Accessing provider_messages must not strip metadata from the inspectable result.
    assert result.messages[-1]["llmigrate"]["audit_instruction"] is True


def test_explicit_openai_format_disambiguates_text_content_parts():
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

    result = llmigrate.migrate(
        messages,
        strategy="raw",
        input_format="openai",
        target_format="openai",
    )

    assert result.format == "openai"
    assert result.provider_messages == messages


def test_developer_messages_survive_aggressive_selection():
    result = llmigrate.migrate(
        [
            {"role": "developer", "content": "Never reveal secrets."},
            {"role": "user", "content": "Help me."},
            {"role": "assistant", "content": "Sure."},
        ],
        strategy="keep_last",
        n=0,
    )

    assert result.provider_messages[0] == {
        "role": "developer",
        "content": "Never reveal secrets.",
    }


def test_parallel_openai_tool_results_keep_their_ids_in_pipeline():
    messages = [
        {"role": "user", "content": "Check both cities."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_paris", "type": "function", "function": {"name": "weather"}},
                {"id": "call_rome", "type": "function", "function": {"name": "weather"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_paris", "content": "Sunny."},
        {"role": "tool", "tool_call_id": "call_rome", "content": "Rainy."},
        {"role": "assistant", "content": "Paris is sunny; Rome is rainy."},
    ]

    result = llmigrate.migrate(messages, strategy="keep_last", n=1)
    tool_results = [message for message in result.provider_messages if message["role"] == "tool"]

    assert [message["tool_call_id"] for message in tool_results] == ["call_paris", "call_rome"]
    assert [message["content"] for message in tool_results] == ["Sunny.", "Rainy."]


def test_anthropic_parallel_tool_results_expand_to_openai_messages():
    messages = [
        {"role": "user", "content": "Check both cities."},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_paris",
                    "name": "weather",
                    "input": {"city": "Paris"},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_rome",
                    "name": "weather",
                    "input": {"city": "Rome"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_paris", "content": "Sunny."},
                {"type": "tool_result", "tool_use_id": "toolu_rome", "content": "Rainy."},
            ],
        },
    ]

    result = llmigrate.migrate(
        messages,
        strategy="raw",
        input_format="anthropic",
        target_format="openai",
    )
    tool_results = [message for message in result.provider_messages if message["role"] == "tool"]

    assert [message["tool_call_id"] for message in tool_results] == ["toolu_paris", "toolu_rome"]
    assert [message["content"] for message in tool_results] == ["Sunny.", "Rainy."]


def test_cross_provider_multimodal_content_fails_instead_of_disappearing():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
            ],
        }
    ]

    with pytest.raises(ValueError, match="OpenAI multimodal content"):
        llmigrate.migrate(
            messages,
            strategy="raw",
            input_format="openai",
            target_format="anthropic",
        )


def test_same_provider_anthropic_image_blocks_survive_roundtrip():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this."},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "encoded"},
                },
            ],
        }
    ]

    result = llmigrate.migrate(messages, strategy="raw", target_format="anthropic")

    assert result.provider_messages == messages


def test_rewriting_signed_anthropic_thinking_blocks_is_rejected():
    canonical = from_anthropic(
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "private reasoning", "signature": "sig"},
                    {"type": "text", "text": "Public answer."},
                ],
            }
        ]
    )
    canonical[0].content = "Rewritten answer."

    with pytest.raises(ValueError, match="signed thinking blocks"):
        to_anthropic(canonical)


def test_changed_anthropic_string_content_remains_a_string():
    canonical = from_anthropic([{"role": "user", "content": "Before."}])
    canonical[0].content = "After."

    result = to_anthropic(canonical)

    assert result["messages"] == [{"role": "user", "content": "After."}]


def test_system_argument_is_returned_separately_for_anthropic():
    result = llmigrate.migrate(
        [{"role": "user", "content": "Hello."}],
        strategy="raw",
        target_format="anthropic",
        system="Be precise.",
    )

    assert result.system == "Be precise."
    assert result.provider_messages == [{"role": "user", "content": "Hello."}]


def test_summary_prompt_includes_tool_call_arguments_and_result_ids():
    messages = [
        {"role": "user", "content": "Find the weather."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather_7",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather_7", "content": "Snowing."},
    ]
    prompts: list[str] = []

    def generate(prompt):
        prompts.append(prompt[0]["content"])
        return "Weather lookup completed."

    llmigrate.migrate(messages, strategy="summarize", generate=generate)

    assert "call_weather_7" in prompts[0]
    assert "get_weather" in prompts[0]
    assert '"city":"Oslo"' in prompts[0]
    assert "Snowing." in prompts[0]


def test_message_token_estimate_accounts_for_tool_payload():
    plain = Message(role=Role.ASSISTANT, content="")
    with_tool_call = Message(
        role=Role.TOOL_CALL,
        content="",
        metadata={
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search",
                        "arguments": '{"query":"long search payload"}',
                    },
                }
            ]
        },
    )

    from llmigrate.tokens import estimate_message_tokens

    assert estimate_message_tokens(with_tool_call, tokenizer=len) > estimate_message_tokens(
        plain, tokenizer=len
    )
