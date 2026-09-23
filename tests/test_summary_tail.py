"""Tests for the summary_tail strategy."""

from __future__ import annotations

import pytest

import llmigrate
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
    def test_requires_total_budget(self):
        with pytest.raises(ValueError, match="total_budget"):
            llmigrate.transfer(MESSAGES, strategy="summary_tail", summary_budget=10)

    def test_requires_summary_budget(self):
        with pytest.raises(ValueError, match="summary_budget"):
            llmigrate.transfer(MESSAGES, strategy="summary_tail", total_budget=100)

    def test_rejects_budgets_exceeding_total(self):
        with pytest.raises(ValueError, match="exceeds total_budget"):
            llmigrate.transfer(
                MESSAGES,
                strategy="summary_tail",
                total_budget=100,
                summary_budget=60,
                tail_budget=60,
            )

    def test_tail_budget_defaults_to_remainder(self):
        result = llmigrate.transfer(
            MESSAGES,
            strategy="summary_tail",
            total_budget=10_000,
            summary_budget=1_000,
            generate=_mock_generate,
        )
        assert result.strategy == llmigrate.Strategy.SUMMARY_TAIL


class TestVerbatimIfFits:
    def test_skips_summarizer_when_everything_fits(self):
        called = []

        def generate(messages):
            called.append(messages)
            return "should not be called"

        result = llmigrate.transfer(
            MESSAGES,
            strategy="summary_tail",
            total_budget=10_000,
            summary_budget=1_000,
            generate=generate,
        )
        assert called == []
        assert result.metadata["prefix_event_ids"] == []
        for m in MESSAGES:
            assert any(m["content"] in out.content for out in result.messages)


class TestSummaryPrefixPlusTail:
    def test_requires_summarizer_when_prefix_nonempty(self):
        with pytest.raises(ValueError, match="summarizer"):
            llmigrate.transfer(
                MESSAGES, strategy="summary_tail", total_budget=10, summary_budget=5, tail_budget=5
            )

    def test_segments_are_tagged(self):
        result = llmigrate.transfer(
            MESSAGES,
            strategy="summary_tail",
            total_budget=10_000,
            summary_budget=1_000,
            tail_budget=10,
            generate=_mock_generate,
        )
        segments = {m.metadata.get("segment") for m in result.messages if "segment" in m.metadata}
        assert "summary_prefix" in segments or "verbatim_tail" in segments

    def test_metadata_shape(self):
        result = llmigrate.transfer(
            MESSAGES,
            strategy="summary_tail",
            total_budget=10_000,
            summary_budget=1_000,
            tail_budget=10,
            generate=_mock_generate,
        )
        for key in (
            "prefix_event_ids",
            "tail_event_ids",
            "summary_tokens",
            "tail_tokens",
            "total_transferred_tokens",
            "summarizer",
        ):
            assert key in result.metadata

    def test_custom_summarizer_reports_cost(self):
        class FakeSummarizer:
            def summarize(self, messages):
                return SummarizerResult(text="custom summary", cost=0.002, model="my-model")

            def __repr__(self):
                return "FakeSummarizer()"

        result = llmigrate.transfer(
            MESSAGES,
            strategy="summary_tail",
            total_budget=10_000,
            summary_budget=1_000,
            tail_budget=10,
            summarizer=FakeSummarizer(),
        )
        assert result.metadata["cost"] == 0.002
        assert result.metadata["model"] == "my-model"

    def test_truncates_oversized_summary(self):
        def long_generate(messages):
            return "x" * 10_000

        result = llmigrate.transfer(
            MESSAGES,
            strategy="summary_tail",
            total_budget=10_000,
            summary_budget=5,
            tail_budget=10,
            generate=long_generate,
        )
        assert result.metadata.get("summary_truncated") is True
