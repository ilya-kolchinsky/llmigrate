# llmigrate

**Prepare a conversation for its next model.** llmigrate reshapes text chat history when you route work to another model, recover from a provider change, or hand a task from one agent to another.

It offers synchronous and asynchronous APIs, several history-selection and compression strategies, and adapters for OpenAI Chat Completions, OpenAI Responses, Anthropic Messages, and Gemini Interactions.

## Install

llmigrate is not published on PyPI yet. Install the current source checkout with Python 3.10 or newer:

```powershell
git clone https://github.com/ilya-kolchinsky/llmigrate.git
cd llmigrate
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

On macOS or Linux:

```sh
git clone https://github.com/ilya-kolchinsky/llmigrate.git
cd llmigrate
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

On Windows, replace `3.11` with the version you have installed if it is 3.10 or newer. On macOS/Linux, check `python3 --version` and use a `python3` command for version 3.10 or newer.

The core library has no runtime dependencies and does not contact a model. Install optional extras from the repository directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[openai]"
.\.venv\Scripts\python.exe -m pip install -e ".[tiktoken]"
```

```sh
.venv/bin/python -m pip install -e '.[openai]'
.venv/bin/python -m pip install -e '.[tiktoken]'
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

This example only transforms local data; it makes no model request. The [Getting Started guide](docs/GETTING-STARTED.md) shows how to choose a strategy, handle each provider's system prompt, and add an optional model callback.

## What it handles

llmigrate migrates **text conversations and structured tool-call/result records**. It does not convert image, audio, video, or document payloads. Some untouched OpenAI and Anthropic blocks can pass through in same-format raw migrations; that does not make them safe for content-changing strategies or cross-format conversion. Unsupported content is rejected where the adapter cannot represent it safely. See [supported data and conversion limits](docs/API-details.md#supported-data-and-conversion-limits).

The system prompt and first user message are protected by default. `selective_history` keeps linked tool calls and known results together. These features preserve transcript structure; they do not guarantee that a different model will interpret the history the same way.

## Documentation

- [Getting Started](docs/GETTING-STARTED.md) — installation, first migration, provider integration, and common choices.
- [API reference](docs/API.md) — entry points, strategies, parameters, and extension points.
- [Format adapters and limitations](docs/API-details.md#format-adapters) — accepted input and output shapes.

## License

MIT
