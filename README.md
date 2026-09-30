# llmigrate

[![CI](https://github.com/ilya-kolchinsky/llmigrate/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ilya-kolchinsky/llmigrate/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/llmigrate)](https://pypi.org/project/llmigrate/)

## Why llmigrate?

When you switch models mid-conversation, it can seem natural to send the receiving model the full transcript as-is. But the transcript labels previous replies only as `assistant`; it does not say which model wrote them. The receiving model may treat the previous model’s claims, mistakes, or commitments as its own earlier replies and build on them as if they were established facts. If the previous model is much weaker or belongs to a very different family, its replies can seriously mislead the receiving model or give it context far outside the kinds of assistant histories it usually sees—in other words, out of distribution.

llmigrate makes the handoff an explicit step: choose what history to preserve, what to condense, and which provider format to send. It helps shape context for the receiving model, but cannot guarantee that model will interpret the migrated history correctly.

## Migration techniques

- Preserve the conversation as-is when only a provider format change is needed.
- Keep recent complete turns, optionally within an estimated token budget.
- Select relevant history while keeping linked tool calls and results together.
- Replace older history with a generated summary or extract structured state such as goals, progress, observations, and next steps.
- Ask the receiving model to check inherited assumptions before continuing. These techniques can be combined into a migration pipeline.

**Formats:** OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Gemini Interactions. OpenAI-compatible servers using the Chat Completions format are supported too.

## Does migration strategy matter?

We ran 5,758 mid-conversation handoffs on [SWE-bench Verified](https://www.swebench.com) — real GitHub issues solved by autonomous coding agents — across three open-weight models (Qwen 3.5 9B, Qwen 3.6 27B, Devstral Small 2 24B). The results show that *how* you transfer context matters more than *how much* you transfer.

**Cross-family handoffs hurt most with raw transfer.** When handing off between model families (Qwen → Devstral), sending the full conversation as-is was the worst-performing strategy at 41.1% task success. The source model's reasoning patterns appear to actively confuse the cross-family recipient. Strategies that restructure or selectively filter the history recovered up to 60% — a 19 percentage-point improvement over raw.

<p align="center">
  <img src="benchmarks/figures/cross_family_transfer.png" width="680" alt="Cross-family model handoff: raw transfer is the worst strategy at 41.1%, while selective_history (priority) reaches 60.0%">
</p>

**Less context, better outcomes.** The Pareto frontier runs through `selective_history` and `structured_state` — they sit in the upper left (less context, higher quality). Raw transfer sits in the lower right (maximum context, no quality gain). Structured strategies used 60–80% less context than raw while matching or exceeding its task success rate.

<p align="center">
  <img src="benchmarks/figures/context_vs_quality.png" width="680" alt="Context size vs. quality tradeoff: selective_history and structured_state achieve higher quality with less context than raw">
</p>

Full results, methodology, and additional figures are in the [benchmark report](benchmarks/BENCHMARKS.md).

## Install

Requires Python 3.10 or newer.

```sh
pip install llmigrate
```

## Quick start

```python
import llmigrate

messages = [
    {"role": "system", "content": "You are a concise writing assistant."},
    {"role": "user", "content": "Write an update about our API migration."},
    {"role": "assistant", "content": "What progress should it mention?"},
    {"role": "user", "content": "The API is complete and all tests pass."},
    {"role": "assistant", "content": "The API is complete, and all tests pass."},
    {"role": "user", "content": "Make that sound more upbeat."},
]

result = llmigrate.migrate(
    messages,
    strategy="keep_last",
    n=1,
    target_format="openai_responses",
)

# Use these fields to build the receiving provider's request.
print(result.provider_messages)  # OpenAI Responses `input`
print(result.system)             # OpenAI Responses `instructions`
```

This example only transforms local data; it makes no model request. The [Getting Started guide](https://github.com/ilya-kolchinsky/llmigrate/blob/main/docs/GETTING-STARTED.md) shows how to choose a strategy, handle each provider's system prompt, and add an optional model callback.

## What it handles

llmigrate migrates **text conversations and structured tool-call/result records**. It does not convert image, audio, video, or document payloads. Some untouched OpenAI and Anthropic blocks can pass through in same-format raw migrations; that does not make them safe for content-changing strategies or cross-format conversion. Unsupported content is rejected where the adapter cannot represent it safely. See [supported data and conversion limits](https://github.com/ilya-kolchinsky/llmigrate/blob/main/docs/API-details.md#supported-data-and-conversion-limits).

The system prompt and first user message are protected by default. `selective_history` keeps linked tool calls and known results together. These features preserve transcript structure; they do not guarantee that a different model will interpret the history the same way.

## Documentation

- [Getting Started](https://github.com/ilya-kolchinsky/llmigrate/blob/main/docs/GETTING-STARTED.md) — installation, first migration, provider integration, and common choices.
- [API reference](https://github.com/ilya-kolchinsky/llmigrate/blob/main/docs/API.md) — entry points, strategies, parameters, and extension points.
- [Format adapters and limitations](https://github.com/ilya-kolchinsky/llmigrate/blob/main/docs/API-details.md#format-adapters) — accepted input and output shapes.
- [Release guide](RELEASING.md) — maintainer steps for preparing and publishing verified releases.

## Development checks

Install the `dev` extra from the repository checkout to get the test, lint, type-checking, and packaging tools. Then run these commands from the repository root:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check src tests
python -m mypy src/llmigrate
python -m build
python -m twine check dist/*
```

The CI workflow tests on Python 3.10, runs lint and type checks on Python 3.14, and builds and smoke-tests the wheel and source distribution on Python 3.14.

## License

MIT
