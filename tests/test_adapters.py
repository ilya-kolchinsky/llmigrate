"""Tests for format adapters."""

from __future__ import annotations

import pytest

from llmigrate.adapters.anthropic import from_anthropic, to_anthropic
from llmigrate.adapters.detect import auto_convert
from llmigrate.adapters.openai import from_openai, to_openai
from llmigrate.types import Message, Role


class TestOpenAIAdapter:
    def test_roundtrip(self):
        original = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        canonical = from_openai(original)
        back = to_openai(canonical)
        assert back == original

    def test_handles_none_content(self):
        messages = [{"role": "assistant", "content": None}]
        canonical = from_openai(messages)
        assert canonical[0].content == ""

    def test_preserves_tool_calls(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "function": {"name": "get_weather"}}],
            }
        ]
        canonical = from_openai(messages)
        assert canonical[0].role == Role.TOOL_CALL
        assert "tool_calls" in canonical[0].metadata

        back = to_openai(canonical)
        assert "tool_calls" in back[0]

    def test_raises_on_unknown_role(self):
        with pytest.raises(ValueError, match="Unknown OpenAI role"):
            from_openai([{"role": "narrator", "content": "hi"}])


class TestAnthropicAdapter:
    def test_roundtrip_with_system(self):
        canonical = from_anthropic(
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            system="Be helpful.",
        )
        assert canonical[0].role == Role.SYSTEM
        assert canonical[0].content == "Be helpful."

        back = to_anthropic(canonical)
        assert back["system"] == "Be helpful."
        assert back["messages"] == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

    def test_preserves_tool_use(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            }
        ]
        canonical = from_anthropic(messages)
        assert canonical[0].role == Role.TOOL_CALL
        assert canonical[0].metadata["tool_calls"][0]["function"]["name"] == "get_weather"

        back = to_anthropic(canonical)
        assert back["messages"][0]["content"] == messages[0]["content"]

    def test_preserves_tool_result(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "Sunny, 20C"},
                ],
            }
        ]
        canonical = from_anthropic(messages)
        assert canonical[0].role == Role.TOOL_RESULT
        assert canonical[0].content == "Sunny, 20C"
        assert canonical[0].metadata["tool_call_id"] == "toolu_1"

    def test_auto_detects_anthropic_format(self):
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
        ]
        result = auto_convert(messages)
        assert result[0].role == Role.ASSISTANT
        assert result[0].content == "hi"


class TestAutoDetect:
    def test_detects_dicts(self):
        result = auto_convert([{"role": "user", "content": "hi"}])
        assert isinstance(result[0], Message)

    def test_passes_through_messages(self):
        msgs = [Message(role=Role.USER, content="hi")]
        result = auto_convert(msgs)
        assert result[0] is msgs[0]

    def test_empty_list(self):
        assert auto_convert([]) == []
