# Getting Started

This guide gets a text conversation ready for a receiving model. The first example runs entirely locally: llmigrate does not need an API key and does not make model calls unless you provide a model callback or call a provider yourself.

## 1. Install from source

llmigrate requires Python 3.10 or newer. A PyPI release is not available yet, so clone the repository and install it in a virtual environment. Follow the Windows PowerShell or macOS/Linux commands in the [README installation section](../README.md#install). The commands use the environment's Python directly, so activation is optional.

To add the optional OpenAI-compatible `generate` helper, install the `openai` extra from the repository directory:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[openai]"
```

```sh
.venv/bin/python -m pip install -e '.[openai]'
```

## 2. Run a first migration

Start with a provider-native list of messages. This example keeps the latest turn, protects the system prompt and first user message by default, and formats the result for OpenAI Responses:

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

print(result.provider_messages)  # send as the Responses API `input`
print(result.system)             # send as the Responses API `instructions`
```

`result.provider_messages` removes llmigrate's metadata sidecars. Use this property when building a provider request; `.messages` is intended for inspecting the migration result. The original system instruction is returned separately in `.system` for OpenAI Responses, Gemini Interactions, and Anthropic. OpenAI Chat Completions keeps the system message in `provider_messages`.

`keep_last` counts complete turns. The first user message is pinned by default, so it can remain alongside the latest turn. If that first message is only a greeting or another message you do not want protected, use `pin_first_user=False`. With provider dictionaries, an explicitly pinned message uses the `llmigrate` sidecar, for example `"llmigrate": {"pinned": True}`. See [protected content](API.md#protected-content-pinning) for the exact rules.

## 3. Choose a strategy

| If you need to… | Start with… | Notes |
|---|---|---|
| Preserve the conversation as-is | `raw` (the default) | Converts supported formats but does not shorten the history. |
| Keep only recent context | `keep_last` | Works without a model call; set `n` to the number of turns. |
| Fit a token budget | `token_budget` | Uses an estimate by default; the [`tiktoken` extra](../README.md#install) improves counts for supported models. |
| Keep important messages and tool events | `selective_history` | Requires a `Selector`; tool calls stay grouped with their known results. |
| Compress earlier context into a summary | `summarize` | Requires your `generate` callback, so it makes a provider call if that callback does. |
| Produce explicit handoff state | `structured_state` | Also uses `generate` to create the state. |

Begin with one strategy. Pipelines can combine one selection, one transformation, and the `audit` validation step; the [strategy reference](API.md#strategies) describes composition and parameters.

For example, a `PrioritySelector` ranks user instructions and tool results above assistant messages. The budget includes pinned content:

```python
selector = llmigrate.PrioritySelector(
    priorities={"user_instruction": 100, "tool_result": 80, "assistant": 20}
)
selected = llmigrate.migrate(
    messages,
    strategy="selective_history",
    budget=2_000,
    selector=selector,
)
```

## 4. Send the result to a provider

Set `target_format` to the receiver's format. Pass `result.provider_messages` to the provider's corresponding message or input parameter, and pass `result.system` separately when the format uses a top-level system instruction.

| Target format | Message/input parameter | System instruction |
|---|---|---|
| `openai` | Chat Completions `messages` | Included in the messages list |
| `openai_responses` | Responses `input` | `instructions` |
| `anthropic` | Messages API `messages` | `system` |
| `gemini_interactions` | Interactions `input` | `system_instruction` |

For example, this OpenAI Chat Completions call performs inference and requires the optional SDK plus valid credentials. It is deliberately separate from the local migration example above:

```python
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from the environment
chat_result = llmigrate.migrate(
    messages,
    strategy="keep_last",
    n=1,
    target_format="openai",
)
response = client.chat.completions.create(
    model="YOUR_MODEL_ID",
    messages=chat_result.provider_messages,
)
print(response.choices[0].message.content)
```

Replace `YOUR_MODEL_ID` with a model available to your account. This provider call may incur charges. llmigrate itself neither chooses the target model nor sends this request.

For Anthropic, Responses, and Gemini Interactions, keep `result.system` in the provider's separate system-instruction parameter; do not add a duplicate system message to the input. The earlier migration example prepares the Responses fields without calling the model. Provider SDKs and request objects differ, so use each provider's API documentation for the final request call.

## 5. Use async migration in an async agent

When your application already has an event loop, use `async_migrate()`. This example only reshapes the messages locally:

```python
import asyncio
import llmigrate

async def main():
    result = await llmigrate.async_migrate(
        messages,
        strategy="keep_last",
        n=1,
        target_format="openai_responses",
    )
    print(result.provider_messages)

asyncio.run(main())
```

`async_migrate()` awaits async callbacks and runs synchronous callbacks in a worker thread. For `summarize` or `structured_state`, give it an async `generate` callback to keep provider I/O async as well. See [`async_migrate()`](API.md#async_migrate).

## 6. Add summarization when you want model-assisted compression

Summarization is opt-in. If your application already has a model client, wrap it in a function that accepts a list of chat messages and returns text, then pass that function as `generate`:

```python
def generate(summary_messages):
    # Demo response only. Replace this with your provider call and return its text.
    return "The API migration is complete and all tests pass."

result = llmigrate.migrate(
    messages,
    strategy="summarize",
    generate=generate,
)
```

Or use the optional `openai_compatible_generate` helper with OpenAI or a compatible Chat Completions endpoint. This example makes a model request when `migrate()` runs; provide a valid model and credentials first:

```python
import os
import llmigrate

generate = llmigrate.openai_compatible_generate(
    base_url="https://api.openai.com/v1",
    model="YOUR_MODEL_ID",
    api_key=os.environ["OPENAI_API_KEY"],
)
result = llmigrate.migrate(messages, strategy="summarize", generate=generate)
```

This helper targets the Chat Completions interface. For other providers or APIs, use your own callback. llmigrate does not provide API keys, select a model, or make hidden provider requests.

## Supported data and limitations

llmigrate operates on text conversations and structured tool-call/result records. It does not convert images, audio, video, or documents. Some untouched OpenAI and Anthropic blocks can pass through same-format raw migrations, but this is not multimodal support. Content-changing strategies can omit media, and cross-format conversion rejects unsupported payloads instead of silently converting them. OpenAI Responses and Gemini Interactions adapters support text and function-call/result items. See [supported data and conversion limits](API-details.md#supported-data-and-conversion-limits).

Format adapters normalize transcript structure; they do not make provider-specific semantics equivalent. In particular, Gemini Interactions histories containing thought steps fail closed because exact step replay may be required. See [adapter details](API-details.md#format-adapters).

## If something goes wrong

- **`ModuleNotFoundError: llmigrate`** — run the script with the Python executable from the virtual environment where you installed the checkout.
- **Unsupported item/content error** — check the supported-data section above. Convert or remove unsupported content explicitly before migration.
- **Ambiguous message format** — specify `input_format`, such as `input_format="openai_responses"`.
- **Already inside an event loop** — await `async_migrate()` rather than calling synchronous model-assisted `migrate()`.
- **The first user message is not the task** — set `pin_first_user=False` or mark the intended task message as pinned.

The full [API reference](API.md) and [adapter reference](API-details.md) document all parameters, return fields, and edge cases.
