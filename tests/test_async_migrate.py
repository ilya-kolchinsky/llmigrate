"""Tests for async-native migration and model-assisted strategies."""

from __future__ import annotations

import threading

import pytest

import llmigrate
from llmigrate.summarizers import SummarizerResult, TokenUsage


async def test_async_migrate_awaits_async_generate_inside_event_loop():
    async def generate(messages, **kwargs):
        return "Async summary."

    result = await llmigrate.async_migrate(
        [
            {"role": "user", "content": "Keep this task."},
            {"role": "assistant", "content": "Old answer."},
            {"role": "user", "content": "Follow-up?"},
            {"role": "assistant", "content": "Latest answer."},
        ],
        strategies=["keep_last", "summarize", "audit"],
        n=1,
        generate=generate,
    )

    assert any("Async summary." in message["content"] for message in result.provider_messages)
    assert result.strategies == [
        llmigrate.Strategy.KEEP_LAST,
        llmigrate.Strategy.SUMMARIZE,
        llmigrate.Strategy.AUDIT,
    ]


async def test_async_migrate_supports_async_summarizer_protocol():
    class MyAsyncSummarizer:
        async def summarize(self, messages):
            return SummarizerResult(
                text="Custom async summary.",
                latency_ms=2.5,
                token_usage=TokenUsage(input_tokens=10, output_tokens=4),
                model="test-model",
            )

    result = await llmigrate.async_migrate(
        [
            {"role": "user", "content": "Task."},
            {"role": "assistant", "content": "Prior work."},
            {"role": "user", "content": "Continue."},
        ],
        strategy="summarize",
        summarizer=MyAsyncSummarizer(),
    )

    assert any("Custom async summary." in message["content"] for message in result.provider_messages)
    assert result.metadata["model"] == "test-model"
    assert result.metadata["token_usage"].output_tokens == 4


async def test_async_migrate_awaits_async_structured_state_generator():
    async def generate(messages, **kwargs):
        return "## Objective\nFinish the task\n## Next Steps\nContinue"

    result = await llmigrate.async_migrate(
        [
            {"role": "user", "content": "Original task."},
            {"role": "assistant", "content": "Some history."},
        ],
        strategy="structured_state",
        generate=generate,
    )

    assert result.metadata["state_data"]["objective"] == "Finish the task"
    assert result.metadata["state_data"]["next_steps"] == "Continue"


async def test_async_migrate_runs_sync_generate_in_worker_thread():
    event_loop_thread = threading.get_ident()

    def generate(messages):
        return f"generated on thread {threading.get_ident()}"

    result = await llmigrate.async_migrate(
        [
            {"role": "user", "content": "Task."},
            {"role": "assistant", "content": "Prior work."},
        ],
        strategy="summarize",
        generate=generate,
    )

    summary = next(
        message["content"]
        for message in result.provider_messages
        if "generated on thread" in message["content"]
    )
    assert int(summary.rsplit(" ", 1)[1]) != event_loop_thread


async def test_sync_migrate_gives_actionable_error_inside_running_loop():
    async def generate(messages):
        return "Summary."

    with pytest.raises(RuntimeError, match=r"use async_migrate\(\) instead"):
        llmigrate.migrate(
            [
                {"role": "user", "content": "Task."},
                {"role": "assistant", "content": "Prior."},
            ],
            strategy="summarize",
            generate=generate,
        )


async def test_sync_migrate_rejects_async_summarizer_with_actionable_error():
    class MyAsyncSummarizer:
        async def summarize(self, messages):
            return SummarizerResult(text="Summary.")

    with pytest.raises(RuntimeError, match=r"use async_migrate\(\)"):
        llmigrate.migrate(
            [
                {"role": "user", "content": "Task."},
                {"role": "assistant", "content": "Prior."},
            ],
            strategy="summarize",
            summarizer=MyAsyncSummarizer(),
        )
