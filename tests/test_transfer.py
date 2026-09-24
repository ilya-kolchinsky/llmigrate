"""Basic tests for the migrate() entry point and strategies."""

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

TOOL_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "function": {"name": "get_weather", "arguments": "{}"}}],
    },
    {"role": "tool", "content": "Sunny, 20C", "tool_call_id": "call_1"},
    {"role": "assistant", "content": "It's sunny and 20C in Paris."},
    {"role": "user", "content": "And in London?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_2", "function": {"name": "get_weather", "arguments": "{}"}}],
    },
    {"role": "tool", "content": "Rainy, 12C", "tool_call_id": "call_2"},
    {"role": "assistant", "content": "It's rainy and 12C in London."},
]


def _tool_ids(messages):
    call_ids = {call["id"] for m in messages for call in m.get("tool_calls", [])}
    result_ids = {m["tool_call_id"] for m in messages if "tool_call_id" in m}
    return call_ids, result_ids


class TestRawStrategy:
    def test_preserves_all_messages(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="raw")
        assert len(result.messages) == len(SAMPLE_MESSAGES)
        assert result.strategies == [llmigrate.Strategy.RAW]

    def test_does_not_mutate_input(self):
        original = [dict(m) for m in SAMPLE_MESSAGES]
        llmigrate.migrate(SAMPLE_MESSAGES, strategy="raw")
        assert SAMPLE_MESSAGES == original

    def test_does_not_enforce_alternation(self):
        consecutive_user = [
            {"role": "user", "content": "one"},
            {"role": "user", "content": "two"},
        ]
        result = llmigrate.migrate(consecutive_user, strategy="raw")
        assert len(result.messages) == 2


class TestKeepLastStrategy:
    def test_keeps_last_n_turns(self):
        result = llmigrate.migrate(
            SAMPLE_MESSAGES, strategy="keep_last", n=2, pin_first_user=False
        )
        assert result.messages[0]["role"] == "system"
        # system + last 2 turns (4 messages: "What is 2+2?"/"4", "Thanks"/"You're welcome!")
        assert len(result.messages) == 5

    def test_n_zero_keeps_only_pinned(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="keep_last", n=0)
        assert [m["role"] for m in result.messages] == ["system", "user"]
        assert result.messages[1]["content"] == "Hello"

    def test_keeps_all_when_n_exceeds_length(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="keep_last", n=100)
        assert len(result.messages) == len(SAMPLE_MESSAGES)

    def test_never_orphans_tool_result(self):
        result = llmigrate.migrate(TOOL_MESSAGES, strategy="keep_last", n=1)
        call_ids, result_ids = _tool_ids(result.messages)
        assert result_ids <= call_ids

    def test_rejects_negative_n(self):
        with pytest.raises(ValueError, match="non-negative"):
            llmigrate.migrate(SAMPLE_MESSAGES, strategy="keep_last", n=-1)


class TestTokenBudgetStrategy:
    def test_respects_budget(self):
        result = llmigrate.migrate(
            SAMPLE_MESSAGES, strategy="token_budget", max_tokens=5, pin_first_user=False
        )
        assert len(result.messages) < len(SAMPLE_MESSAGES)
        assert result.messages[0]["role"] == "system"

    def test_large_budget_keeps_all(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="token_budget", max_tokens=100000)
        assert len(result.messages) == len(SAMPLE_MESSAGES)

    def test_never_orphans_tool_result(self):
        result = llmigrate.migrate(TOOL_MESSAGES, strategy="token_budget", max_tokens=5)
        call_ids, result_ids = _tool_ids(result.messages)
        assert result_ids <= call_ids

    def test_target_model_sizes_default_budget(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="token_budget", target_model="gpt-4o")
        assert result.metadata["max_tokens"] == int(128_000 * 0.9)

    def test_rejects_non_positive_max_tokens(self):
        with pytest.raises(ValueError, match="positive"):
            llmigrate.migrate(SAMPLE_MESSAGES, strategy="token_budget", max_tokens=0)


class TestSummarizeStrategy:
    def test_without_generate_uses_fallback(self):
        result = llmigrate.migrate(
            SAMPLE_MESSAGES, strategy="summarize", pin_first_user=False
        )
        assert result.metadata["summarized"] is False
        assert any("omitted" in m["content"].lower() for m in result.messages)

    def test_with_generate(self):
        def mock_generate(messages):
            return "Summary: the user asked about math."

        result = llmigrate.migrate(
            SAMPLE_MESSAGES, strategy="summarize", generate=mock_generate
        )
        assert result.metadata["summarized"] is True
        assert any("math" in m["content"] for m in result.messages)
        assert "latency_ms" in result.metadata

    def test_generate_kwargs_passthrough(self):
        captured = {}

        def mock_generate(messages, **kwargs):
            captured.update(kwargs)
            return "ok"

        llmigrate.migrate(
            SAMPLE_MESSAGES,
            strategy="summarize",
            generate=mock_generate,
            generate_kwargs={"temperature": 0.2},
        )
        assert captured == {"temperature": 0.2}

    def test_async_generate(self):
        async def mock_generate(messages):
            return "async summary"

        result = llmigrate.migrate(
            SAMPLE_MESSAGES, strategy="summarize", generate=mock_generate
        )
        assert any("async summary" in m["content"] for m in result.messages)

    def test_pinned_task_not_fed_to_summarizer(self):
        captured_prompts = []

        def mock_generate(messages):
            captured_prompts.append(messages)
            return "summary text"

        llmigrate.migrate(SAMPLE_MESSAGES, strategy="summarize", generate=mock_generate)
        prompt_text = captured_prompts[0][0]["content"]
        assert "Hello" not in prompt_text


class TestStructuredStateStrategy:
    def test_without_generate_uses_heuristic(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="structured_state")
        assert result.strategies == [llmigrate.Strategy.STRUCTURED_STATE]
        assert len(result.messages) >= 1
        assert "Hello" in result.metadata["state_data"]["objective"]

    def test_with_generate(self):
        def mock_generate(messages):
            return "## Objective\nanswer math questions\n\n## Completed\nanswered 2+2=4"

        result = llmigrate.migrate(
            SAMPLE_MESSAGES, strategy="structured_state", generate=mock_generate
        )
        assert any("math" in m["content"] for m in result.messages)
        assert result.metadata["state_data"]["objective"] == "answer math questions"
        assert result.metadata["state_data"]["completed"] == "answered 2+2=4"


class TestAuditStrategy:
    def test_appends_audit_message(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="audit")
        assert len(result.messages) == len(SAMPLE_MESSAGES) + 1
        assert result.messages[-1].get("llmigrate", {}).get("audit_instruction") is True

    def test_custom_instruction(self):
        result = llmigrate.migrate(
            SAMPLE_MESSAGES,
            strategy="audit",
            instruction="Check everything twice.",
        )
        assert "twice" in result.messages[-1]["content"]

    def test_source_model_interpolated(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="audit", source_model="gpt-4o")
        assert "gpt-4o" in result.messages[-1]["content"]
        assert result.metadata["source_model"] == "gpt-4o"


class TestUnknownStrategy:
    def test_raises_on_unknown(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            llmigrate.migrate(SAMPLE_MESSAGES, strategy="nonexistent")


class TestCanonicalInput:
    def test_accepts_message_objects(self):
        canonical = [
            llmigrate.Message(role=Role.USER, content="Hello"),
            llmigrate.Message(role=Role.ASSISTANT, content="Hi"),
        ]
        result = llmigrate.migrate(canonical, strategy="raw")
        assert len(result.messages) == 2


class TestTargetFormat:
    def test_defaults_to_source_format(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="raw")
        assert result.format == "openai"
        assert result.system is None
        assert isinstance(result.messages[0], dict)
        assert "role" in result.messages[0]

    def test_openai_to_anthropic(self):
        result = llmigrate.migrate(SAMPLE_MESSAGES, strategy="raw", target_format="anthropic")
        assert result.format == "anthropic"
        assert result.system == "You are a helpful assistant."
        assert all(m["role"] != "system" for m in result.messages)
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "Hello"

    def test_anthropic_to_openai(self):
        anthropic_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        result = llmigrate.migrate(anthropic_messages, strategy="raw", target_format="openai")
        assert result.format == "openai"
        assert result.system is None
        assert result.messages[0]["role"] == "user"
        assert result.messages[0]["content"] == "Hello"

    def test_anthropic_input_defaults_to_anthropic_output(self):
        anthropic_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        result = llmigrate.migrate(anthropic_messages, strategy="raw")
        assert result.format == "anthropic"

    def test_canonical_input_defaults_to_openai(self):
        canonical = [
            llmigrate.Message(role=Role.USER, content="Hello"),
            llmigrate.Message(role=Role.ASSISTANT, content="Hi"),
        ]
        result = llmigrate.migrate(canonical, strategy="raw")
        assert result.format == "openai"

    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="Unknown target_format"):
            llmigrate.migrate(SAMPLE_MESSAGES, strategy="raw", target_format="gemini")

    def test_format_with_strategy(self):
        result = llmigrate.migrate(
            SAMPLE_MESSAGES, strategy="keep_last", n=2, target_format="anthropic"
        )
        assert result.format == "anthropic"
        assert result.system is not None
        assert all(m["role"] in ("user", "assistant") for m in result.messages)

    def test_anthropic_format_system_is_none_when_no_system(self):
        no_system = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = llmigrate.migrate(no_system, strategy="raw", target_format="anthropic")
        assert result.system is None

    def test_llmigrate_metadata_in_output(self):
        def mock_generate(messages):
            return "Summary text."

        result = llmigrate.migrate(
            SAMPLE_MESSAGES, strategy="summarize", generate=mock_generate
        )
        synthetic_msgs = [
            m for m in result.messages
            if m.get("llmigrate", {}).get("llmigrate_synthetic")
        ]
        assert len(synthetic_msgs) > 0


class TestValidation:
    def test_rejects_non_list(self):
        with pytest.raises(TypeError, match="must be a list"):
            llmigrate.migrate("not a list", strategy="raw")

    def test_rejects_missing_role(self):
        with pytest.raises(ValueError, match="role"):
            llmigrate.migrate([{"content": "hi"}], strategy="raw")

    def test_rejects_unknown_role(self):
        with pytest.raises(ValueError, match="Unknown OpenAI role"):
            llmigrate.migrate([{"role": "narrator", "content": "hi"}], strategy="raw")

    def test_empty_messages_is_valid(self):
        result = llmigrate.migrate([], strategy="raw")
        assert result.messages == []
