# Extended API Reference

Provider adapter details, data limits, validation errors, and extension guidance. The main [API reference](API.md) describes strategies and the `migrate()` / `async_migrate()` entry points.

## Format Adapters

The adapters in `src/llmigrate/adapters/` translate provider-native history into canonical `Message` objects and convert the result back to the requested target format. Set `input_format` when an abbreviated or ambiguous history cannot be reliably identified.

Supported format names are:

| `input_format` / `target_format` | Provider representation |
|---|---|
| `openai` | OpenAI Chat Completions messages; also OpenAI-compatible endpoints |
| `openai_responses` | Materialized OpenAI Responses input/output items |
| `anthropic` | Anthropic Messages entries; its top-level system prompt uses `system=` |
| `gemini_interactions` | Materialized Gemini Interactions steps |
| `canonical` | `list[llmigrate.Message]` input only |

`target_format="canonical"` is not supported; migrations return provider-ready data. Empty input defaults to `openai` unless a format is specified.

### Auto-detection (`adapters/detect.py`)

`auto_convert(messages, input_format=None) -> list[Message]` detects canonical `Message` objects directly. Provider dictionary arrays are recognized by their item signatures: Gemini `user_input` / `model_output` / `function_result` steps, Responses `message` / `function_call` / `function_call_output` items, Anthropic content blocks, then OpenAI Chat Completions as the common default.

The `function_call` item is shared by Gemini and Responses. Their identifiers differ in the normal shapes (`id` for Gemini, `call_id` for Responses); specify `input_format` when working with customized payloads or single items that do not carry the usual distinguishing field. A single call must contain one provider format, not a mixture.

### OpenAI Chat Completions (`adapters/openai.py`)

Roles map to the canonical `SYSTEM`, `DEVELOPER`, `USER`, `ASSISTANT`, `TOOL_CALL`, and `TOOL_RESULT` roles. Assistant `tool_calls` and legacy `function_call` fields are preserved in canonical metadata; tool results retain their `tool_call_id`. OpenAI-compatible servers use the same message shape.

The adapter preserves some content blocks for same-format pass-through. The presence of raw data in an unchanged message does not mean strategies can interpret or translate that data; see [Supported Data and Conversion Limits](#supported-data-and-conversion-limits).

### Anthropic Messages (`adapters/anthropic.py`)

```python
result = llmigrate.migrate(
    anthropic_messages,
    input_format="anthropic",
    target_format="anthropic",
    system=anthropic_system,
)
request = {"system": result.system, "messages": result.provider_messages}
```

`tool_use` blocks map to canonical tool calls and `tool_result` blocks preserve their `tool_use_id`. The system prompt is returned separately as `MigrationResult.system`. Untouched content blocks can be retained in same-format conversions; rewritten signed thinking blocks and unsupported cross-provider blocks fail with `ValueError`.

### OpenAI Responses (`adapters/openai_responses.py`; [API reference](https://platform.openai.com/docs/api-reference/responses))

Input and output are materialized item lists. The adapter supports text `message` items with `system`, `developer`, `user`, or `assistant` roles, plus `function_call` and `function_call_output` items:

```python
history = [
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Find the release date."}]},
    {"type": "function_call", "call_id": "call_1", "name": "search", "arguments": "{\"q\": \"release date\"}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "Released in 2025."},
]

result = llmigrate.migrate(history, target_format="openai_responses")
request = {"instructions": result.system, "input": result.provider_messages}
```

System messages are emitted through `result.system` as the Responses API `instructions` value. Tool-call arguments are JSON text; text tool outputs are strings. The adapter rejects non-text content parts, refusal and reasoning items, and other item types instead of discarding them. A server-side conversation or response ID is not a substitute for a materialized history and is not migrated.

### Gemini Interactions (`adapters/gemini_interactions.py`; [overview](https://ai.google.dev/gemini-api/docs/interactions-overview))

Input and output are materialized Interactions steps. Supported steps are text `user_input` / `model_output` and client function `function_call` / `function_result`:

```python
steps = [
    {"type": "user_input", "content": [{"type": "text", "text": "Check the weather in Paris."}]},
    {"type": "function_call", "id": "call_1", "name": "weather", "arguments": {"city": "Paris"}},
    {"type": "function_result", "call_id": "call_1", "name": "weather", "result": "Sunny."},
    {"type": "model_output", "content": [{"type": "text", "text": "It is sunny."}]},
]

result = llmigrate.migrate(steps, target_format="gemini_interactions")
request = {"system_instruction": result.system, "input": result.provider_messages}
```

Function arguments must be JSON objects when converting to Gemini. Text or JSON function results can be represented; multimodal results cannot. The adapter rejects `thought`, built-in/server tool steps, and other unsupported step types. Gemini's [stateless function-calling flow](https://ai.google.dev/gemini-api/docs/function-calling) can require thought steps to be replayed exactly; those provider-specific steps are not portable context for cross-model migration, so such histories currently fail closed. An interaction ID or `previous_interaction_id` is server-side state, not a materialized history.

## Supported Data and Conversion Limits

llmigrate's strategies operate on a canonical message with a string `content`. It can transform textual conversation history and preserve structured function-call/result records whose IDs link the calls to their results. It does not provide multimodal understanding or media conversion.

| Data | Behavior |
|---|---|
| Text messages | Supported across the four provider formats |
| Function calls and results | Supported when IDs and required names/arguments are present; selection keeps known call/result groups together |
| JSON function arguments/results | Preserved as structured data in raw same-format migrations; normalized as JSON text when a target format requires text |
| Images, audio, video, documents, and other media | Not transformed; new Responses and Gemini adapters reject them. Existing OpenAI/Anthropic adapters may pass through some unchanged source blocks in same-format paths |
| Provider reasoning, thought, and server-managed state | Not treated as portable conversation text; unsupported item types raise `ValueError` |

Content-changing strategies may summarize or omit the dropped part of the history, and token estimates cannot account accurately for provider-specific framing or media. Preserve extra context budget and preprocess content explicitly when a session contains unsupported data. Cross-format migration is limited to the data represented by the canonical model; provider-specific semantics cannot be reconstructed automatically.

## Errors & Validation

| Condition | Exception |
|---|---|
| `messages` is not a `list` | `TypeError` |
| An element is neither a provider dictionary nor a `Message` | `TypeError` |
| A provider item has an unsupported or malformed role/type | `ValueError` |
| A Responses/Gemini item contains an unsupported content part or step type | `ValueError` |
| A tool result lacks its call identifier, or a tool call lacks required fields | `ValueError` |
| `messages` mixes provider dictionaries and `Message` objects | `TypeError` |
| Provider formats are mixed in one input list | `ValueError` or provider validation error |
| Unknown `strategy` or a strategy-specific invalid parameter | `ValueError` |
| Both `strategy` and `strategies` are specified, or categories are duplicated | `ValueError` |
| Unknown `target_format` or `input_format` | `ValueError` |
| An async `generate` callback is passed to `migrate()` inside a running event loop | `RuntimeError`; use `await async_migrate(...)` |

## Adding a New Strategy

1. Create a module in `src/llmigrate/strategies/` with a `transform(messages, **params) -> MigrationResult` function and the category-specific `select()`, `compress()`, or `validate()` function.
2. For transformations that can drop or rewrite content, use `pinning.split_pinned()` at the start of standalone `transform()`. The pipeline handles pinning centrally.
3. Register the strategy in `strategies/__init__.py`: `STRATEGY_REGISTRY`, `STRATEGY_CATEGORIES`, and its category registry (`SELECT_REGISTRY`, `COMPRESS_REGISTRY`, or `VALIDATE_REGISTRY`).
4. Add a `Strategy` enum value in `types.py`.
5. Add tests, including a `TIGHT_PARAMS` entry in `tests/test_pinning.py` so registry-wide invariant tests cover it.
