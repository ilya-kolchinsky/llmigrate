"""Tests for format adapters."""

from __future__ import annotations

from llmigrate.adapters.openai import from_openai, to_openai
from llmigrate.adapters.detect import auto_convert
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
