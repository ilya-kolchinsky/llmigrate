"""Basic tests for the transfer() entry point and strategies."""

from __future__ import annotations

import pytest

import llmigrate
from llmigrate.types import Role


SAMPLE_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "Thanks"},
    {"role": "assistant", "content": "You're welcome!"},
]


class TestRawStrategy:
    def test_preserves_all_messages(self):
        result = llmigrate.transfer(SAMPLE_MESSAGES, strategy="raw")
        assert len(result.messages) == len(SAMPLE_MESSAGES)
        assert result.strategy == llmigrate.Strategy.RAW

    def test_does_not_mutate_input(self):
        original = [dict(m) for m in SAMPLE_MESSAGES]
        llmigrate.transfer(SAMPLE_MESSAGES, strategy="raw")
        assert SAMPLE_MESSAGES == original


class TestKeepLastStrategy:
    def test_keeps_system_plus_last_n(self):
        result = llmigrate.transfer(SAMPLE_MESSAGES, strategy="keep_last", n=2)
        assert result.messages[0].role == Role.SYSTEM
        assert len(result.messages) == 3  # 1 system + 2 kept

    def test_keeps_all_when_n_exceeds_length(self):
        result = llmigrate.transfer(SAMPLE_MESSAGES, strategy="keep_last", n=100)
        assert len(result.messages) == len(SAMPLE_MESSAGES)


class TestTokenBudgetStrategy:
    def test_respects_budget(self):
        result = llmigrate.transfer(
            SAMPLE_MESSAGES, strategy="token_budget", max_tokens=5
        )
        assert len(result.messages) < len(SAMPLE_MESSAGES)
        assert result.messages[0].role == Role.SYSTEM

    def test_large_budget_keeps_all(self):
        result = llmigrate.transfer(
            SAMPLE_MESSAGES, strategy="token_budget", max_tokens=100000
        )
        assert len(result.messages) == len(SAMPLE_MESSAGES)


class TestSummarizeStrategy:
    def test_without_generate_uses_fallback(self):
        result = llmigrate.transfer(SAMPLE_MESSAGES, strategy="summarize", tail=2)
        assert result.metadata["summarized"] is False
        assert any("omitted" in m.content.lower() for m in result.messages)

    def test_with_generate(self):
        def mock_generate(messages):
            return "Summary: the user asked about math."

        result = llmigrate.transfer(
            SAMPLE_MESSAGES, strategy="summarize", generate=mock_generate, tail=2
        )
        assert result.metadata["summarized"] is True
        assert any("math" in m.content for m in result.messages)


class TestCapsuleStrategy:
    def test_without_generate_uses_heuristic(self):
        result = llmigrate.transfer(SAMPLE_MESSAGES, strategy="capsule")
        assert result.strategy == llmigrate.Strategy.CAPSULE
        assert len(result.messages) >= 1

    def test_with_generate(self):
        def mock_generate(messages):
            return "Objective: answer math questions\nCompleted: answered 2+2=4"

        result = llmigrate.transfer(
            SAMPLE_MESSAGES, strategy="capsule", generate=mock_generate
        )
        assert any("math" in m.content for m in result.messages)


class TestAuditStrategy:
    def test_appends_audit_message(self):
        result = llmigrate.transfer(SAMPLE_MESSAGES, strategy="audit")
        assert len(result.messages) == len(SAMPLE_MESSAGES) + 1
        assert result.messages[-1].metadata.get("audit_instruction") is True

    def test_custom_instruction(self):
        result = llmigrate.transfer(
            SAMPLE_MESSAGES,
            strategy="audit",
            instruction="Check everything twice.",
        )
        assert "twice" in result.messages[-1].content


class TestUnknownStrategy:
    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            llmigrate.transfer(SAMPLE_MESSAGES, strategy="nonexistent")


class TestCanonicalInput:
    def test_accepts_message_objects(self):
        canonical = [
            llmigrate.Message(role=Role.USER, content="Hello"),
            llmigrate.Message(role=Role.ASSISTANT, content="Hi"),
        ]
        result = llmigrate.transfer(canonical, strategy="raw")
        assert len(result.messages) == 2
