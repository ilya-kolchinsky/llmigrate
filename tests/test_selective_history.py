"""Tests for the selective_history strategy."""

from __future__ import annotations

import pytest

import llmigrate
from llmigrate.selectors import PrioritySelector, RelevanceSelector
from llmigrate.types import Role

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Fix the bug in payments.py"},
    {"role": "assistant", "content": "Sure, looking into it."},
    {"role": "assistant", "content": "I ran the test suite and found the failure."},
    {"role": "user", "content": "Here's the traceback: KeyError on line 42"},
    {"role": "assistant", "content": "I see the issue, it's a missing default value."},
]


class TestRequiredParams:
    def test_requires_budget(self):
        with pytest.raises(ValueError, match="budget"):
            llmigrate.migrate(
                MESSAGES, strategy="selective_history", selector=PrioritySelector({})
            )

    def test_requires_selector(self):
        with pytest.raises(ValueError, match="selector"):
            llmigrate.migrate(MESSAGES, strategy="selective_history", budget=1000)

    def test_rejects_negative_budget(self):
        with pytest.raises(ValueError, match="non-negative"):
            llmigrate.migrate(
                MESSAGES, strategy="selective_history", budget=-1, selector=PrioritySelector({})
            )


class TestPrioritySelector:
    def test_content_is_verbatim(self):
        selector = PrioritySelector(priorities={"user_instruction": 100, "assistant": 10})
        result = llmigrate.migrate(
            MESSAGES, strategy="selective_history", budget=10_000, selector=selector
        )
        original_texts = {m["content"] for m in MESSAGES}
        for m in result.messages:
            for part in m["content"].split("\n\n"):
                assert part in original_texts

    def test_output_is_chronological(self):
        selector = PrioritySelector(priorities={"user_instruction": 100, "assistant": 10})
        result = llmigrate.migrate(
            MESSAGES, strategy="selective_history", budget=10_000, selector=selector
        )
        selected_ids = result.metadata["selected_event_ids"]
        assert selected_ids == sorted(selected_ids)

    def test_respects_tight_budget(self):
        selector = PrioritySelector(priorities={"user_instruction": 100, "assistant": 10})
        result = llmigrate.migrate(
            MESSAGES, strategy="selective_history", budget=1, selector=selector
        )
        assert result.messages[0]["role"] == "system"

    def test_always_keep_forces_category(self):
        selector = PrioritySelector(priorities={"assistant": 100})
        result = llmigrate.migrate(
            MESSAGES,
            strategy="selective_history",
            budget=10_000,
            selector=selector,
            always_keep={"user_instruction"},
        )
        user_texts = {m["content"] for m in MESSAGES if m["role"] == "user"}
        result_texts = {part for m in result.messages for part in m["content"].split("\n\n")}
        assert user_texts <= result_texts

    def test_metadata_shape(self):
        selector = PrioritySelector(priorities={"user_instruction": 100})
        result = llmigrate.migrate(
            MESSAGES, strategy="selective_history", budget=10_000, selector=selector
        )
        for key in (
            "selected_event_ids",
            "dropped_event_ids",
            "original_tokens",
            "transferred_tokens",
            "selector",
        ):
            assert key in result.metadata


class TestRelevanceSelector:
    def test_scores_by_cosine_similarity(self):
        def embed(texts):
            vocab = ["bug", "traceback", "keyerror", "test"]
            return [[1.0 if word in t.lower() else 0.0 for word in vocab] for t in texts]

        selector = RelevanceSelector(embed=embed, query="traceback keyerror")
        result = llmigrate.migrate(
            MESSAGES, strategy="selective_history", budget=10_000, selector=selector
        )
        assert any("KeyError" in m["content"] for m in result.messages)


def test_tool_call_and_matching_result_are_dropped_atomically_when_pair_does_not_fit():
    class PreferCall:
        def score(self, messages):
            return [100.0 if message.role == Role.TOOL_CALL else 0.0 for message in messages]

    messages = [
        {"role": "user", "content": "Find the weather."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny."},
        {"role": "assistant", "content": "The weather is sunny."},
    ]

    result = llmigrate.migrate(
        messages,
        strategy="selective_history",
        budget=8,
        selector=PreferCall(),
        tokenizer=lambda _text: 0,
    )

    assert not any(message.get("tool_calls") for message in result.provider_messages)
    assert not any(message.get("role") == "tool" for message in result.provider_messages)
    assert any(message.get("content") == "The weather is sunny." for message in result.provider_messages)


def test_selective_history_can_select_tool_call_and_result_as_one_event():
    class PreferCall:
        def score(self, messages):
            return [100.0 if message.role == Role.TOOL_CALL else 0.0 for message in messages]

    messages = [
        {"role": "user", "content": "Find the weather."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_weather",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_weather", "content": "Sunny."},
        {"role": "assistant", "content": "The weather is sunny."},
    ]

    result = llmigrate.migrate(
        messages,
        strategy="selective_history",
        budget=12,
        selector=PreferCall(),
        tokenizer=lambda _text: 0,
    )

    assert any(message.get("tool_calls") for message in result.provider_messages)
    assert any(message.get("role") == "tool" for message in result.provider_messages)
    assert not any(
        message.get("content") == "The weather is sunny."
        for message in result.provider_messages
    )
