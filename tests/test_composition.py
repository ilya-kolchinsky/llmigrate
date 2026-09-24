"""Tests for the composable strategy pipeline (strategies= parameter)."""

from __future__ import annotations

import pytest

import llmigrate
from llmigrate.selectors import PrioritySelector
from llmigrate.summarizers import SummarizerResult

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Let's plan the migration."},
    {"role": "assistant", "content": "Sure, first we inventory the schemas."},
    {"role": "user", "content": "Done, there are 12 tables."},
    {"role": "assistant", "content": "Great, next we write the migration scripts."},
    {"role": "user", "content": "I've written scripts for 8 of them."},
    {"role": "assistant", "content": "Let's finish the remaining 4 and then test."},
]


def _mock_generate(messages):
    return "Summary of the migration planning discussion."


class TestValidation:
    def test_rejects_both_strategy_and_strategies(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            llmigrate.migrate(MESSAGES, strategy="keep_last", strategies=["keep_last"])

    def test_rejects_unknown_strategy_in_list(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            llmigrate.migrate(MESSAGES, strategies=["nonexistent"])

    def test_rejects_duplicate_category(self):
        with pytest.raises(ValueError, match="At most one selection"):
            llmigrate.migrate(MESSAGES, strategies=["keep_last", "token_budget"])

    def test_rejects_raw_in_composition(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            llmigrate.migrate(MESSAGES, strategies=["raw", "keep_last"])

    def test_empty_strategies_is_raw(self):
        result = llmigrate.migrate(MESSAGES, strategies=[])
        assert result.strategies == [llmigrate.Strategy.RAW]
        assert len(result.messages) == len(MESSAGES)

    def test_single_strategy_in_list(self):
        result = llmigrate.migrate(MESSAGES, strategies=["keep_last"], n=2)
        assert result.strategies == [llmigrate.Strategy.KEEP_LAST]
        assert len(result.messages) < len(MESSAGES)


class TestOrderInvariance:
    def test_selection_then_validation(self):
        r1 = llmigrate.migrate(MESSAGES, strategies=["keep_last", "audit"], n=2)
        r2 = llmigrate.migrate(MESSAGES, strategies=["audit", "keep_last"], n=2)
        assert len(r1.messages) == len(r2.messages)
        assert r1.strategies == r2.strategies

    def test_all_three_categories(self):
        r1 = llmigrate.migrate(
            MESSAGES,
            strategies=["audit", "summarize", "keep_last"],
            n=2,
            generate=_mock_generate,
        )
        r2 = llmigrate.migrate(
            MESSAGES,
            strategies=["keep_last", "summarize", "audit"],
            n=2,
            generate=_mock_generate,
        )
        assert r1.strategies == r2.strategies
        assert len(r1.messages) == len(r2.messages)


class TestSelectionPlusTransformation:
    def test_keep_last_plus_summarize(self):
        result = llmigrate.migrate(
            MESSAGES,
            strategies=["keep_last", "summarize"],
            n=1,
            generate=_mock_generate,
        )
        assert llmigrate.Strategy.KEEP_LAST in result.strategies
        assert llmigrate.Strategy.SUMMARIZE in result.strategies
        assert any("Summary" in m["content"] for m in result.messages)

    def test_token_budget_plus_summarize(self):
        result = llmigrate.migrate(
            MESSAGES,
            strategies=["token_budget", "summarize"],
            max_tokens=10,
            generate=_mock_generate,
        )
        assert llmigrate.Strategy.TOKEN_BUDGET in result.strategies
        assert llmigrate.Strategy.SUMMARIZE in result.strategies
        assert any("Summary" in m["content"] for m in result.messages)

    def test_skips_transformation_when_nothing_dropped(self):
        called = []

        def generate(messages):
            called.append(True)
            return "should not be called"

        result = llmigrate.migrate(
            MESSAGES,
            strategies=["keep_last", "summarize"],
            n=100,
            generate=generate,
        )
        assert called == []
        for m in MESSAGES:
            assert any(m["content"] in out["content"] for out in result.messages)

    def test_selective_history_plus_structured_state(self):
        selector = PrioritySelector(priorities={"user_instruction": 100})
        result = llmigrate.migrate(
            MESSAGES,
            strategies=["selective_history", "structured_state"],
            budget=10_000,
            selector=selector,
        )
        assert llmigrate.Strategy.SELECTIVE_HISTORY in result.strategies
        assert llmigrate.Strategy.STRUCTURED_STATE in result.strategies

    def test_custom_summarizer_reports_cost(self):
        class FakeSummarizer:
            def summarize(self, messages):
                return SummarizerResult(text="custom summary", cost=0.002, model="my-model")

            def __repr__(self):
                return "FakeSummarizer()"

        result = llmigrate.migrate(
            MESSAGES,
            strategies=["token_budget", "summarize"],
            max_tokens=10,
            summarizer=FakeSummarizer(),
        )
        assert result.metadata["transformation"]["cost"] == 0.002
        assert result.metadata["transformation"]["model"] == "my-model"

    def test_truncates_oversized_summary(self):
        def long_generate(messages):
            return "x" * 10_000

        result = llmigrate.migrate(
            MESSAGES,
            strategies=["token_budget", "summarize"],
            max_tokens=10,
            generate=long_generate,
            max_summary_tokens=5,
        )
        assert result.metadata["transformation"].get("summary_truncated") is True


class TestSelectionPlusAdaptation:
    def test_keep_last_plus_audit(self):
        result = llmigrate.migrate(
            MESSAGES, strategies=["keep_last", "audit"], n=1
        )
        assert llmigrate.Strategy.KEEP_LAST in result.strategies
        assert llmigrate.Strategy.AUDIT in result.strategies
        assert result.messages[-1].get("llmigrate", {}).get("audit_instruction") is True

    def test_audit_source_model_in_metadata(self):
        result = llmigrate.migrate(
            MESSAGES,
            strategies=["keep_last", "audit"],
            n=1,
            source_model="gpt-4o",
        )
        assert "gpt-4o" in result.messages[-1]["content"]


class TestPipelineMetadata:
    def test_namespaced_metadata(self):
        result = llmigrate.migrate(
            MESSAGES,
            strategies=["keep_last", "summarize"],
            n=1,
            generate=_mock_generate,
        )
        assert "original_count" in result.metadata
        assert "selection" in result.metadata
        assert "transformation" in result.metadata

    def test_validation_metadata(self):
        result = llmigrate.migrate(
            MESSAGES,
            strategies=["audit"],
            source_model="gpt-4o",
        )
        assert "validation" in result.metadata


class TestPinningInPipeline:
    MARKER = "TASK-MARKER: implement the fix"

    FIXTURE = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": MARKER},
        {"role": "assistant", "content": "Sure, let's start."},
        {"role": "user", "content": "Here's more context."},
        {"role": "assistant", "content": "Got it."},
        {"role": "user", "content": "One more thing."},
        {"role": "assistant", "content": "Noted."},
    ]

    def test_preserves_task_marker_in_composition(self):
        result = llmigrate.migrate(
            self.FIXTURE,
            strategies=["token_budget", "summarize"],
            max_tokens=1,
            generate=_mock_generate,
        )
        assert any(self.MARKER in m["content"] for m in result.messages)

    def test_alternation_enforced_in_pipeline(self):
        result = llmigrate.migrate(
            self.FIXTURE,
            strategies=["keep_last", "summarize"],
            n=0,
            generate=_mock_generate,
        )
        roles = [m["role"] for m in result.messages if m["role"] != "system"]
        for a, b in zip(roles, roles[1:]):
            assert a != b
