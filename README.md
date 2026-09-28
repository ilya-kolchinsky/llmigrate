# llmigrate

**Prepare a conversation for its next model.** llmigrate reshapes text chat history when you route work to another model, recover from a provider change, or hand a task from one agent to another.

It offers synchronous and asynchronous APIs, several history-selection and compression strategies, and adapters for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Gemini Interactions.

[![CI](https://github.com/ilya-kolchinsky/llmigrate/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ilya-kolchinsky/llmigrate/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/llmigrate)](https://pypi.org/project/llmigrate/)

## Install

llmigrate requires Python 3.10 or newer. Install the latest release from PyPI in a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install llmigrate
```

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install llmigrate
```

On Windows, replace `3.11` with the version you have installed if it is 3.10 or newer. On macOS/Linux, check `python3 --version` and use a `python3` command for version 3.10 or newer.

The core library has no runtime dependencies and does not contact a model. Optional extras add an OpenAI-compatible `generate` helper or improve token estimates for supported models:

```powershell
.\.venv\Scripts\python.exe -m pip install "llmigrate[openai]"
.\.venv\Scripts\python.exe -m pip install "llmigrate[tiktoken]"
```

```sh
.venv/bin/python -m pip install 'llmigrate[openai]'
.venv/bin/python -m pip install 'llmigrate[tiktoken]'
```

`openai` adds a ready-made OpenAI-compatible `generate` helper. `tiktoken` improves token estimates for supported models. Both extras are optional.

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
