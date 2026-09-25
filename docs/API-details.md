# Extended API Reference

Provider-specific adapter details, validation errors, and extension guidance.

## Format Adapters

`src/llmigrate/adapters/` — convert between provider-native formats and canonical `Message`s. `migrate()` calls `auto_convert()` on input and converts back to wire format on output via `to_openai()` or `to_anthropic()`, controlled by `target_format`. If content-part arrays make the source ambiguous, set `input_format` explicitly.

### Auto-detection (`adapters/detect.py`)

`auto_convert(messages, input_format=None) -> list[Message]` dispatches based on the first element unless `input_format` is supplied:
- `Message` → returned as-is (all elements are assumed to already be canonical).
- `dict` whose `content` contains Anthropic-specific blocks such as `tool_use`, `tool_result`, or `image` → treated as Anthropic format (`from_anthropic`).
- any other `dict` → treated as OpenAI format (`from_openai`), which also covers OpenAI-compatible self-hosted servers, since they speak the same wire format.
- anything else → `TypeError`.

Detection inspects content blocks across the message list, so a single call must use a consistent format throughout.

### OpenAI (`adapters/openai.py`)

`from_openai(messages: list[dict]) -> list[Message]` / `to_openai(messages: list[Message]) -> list[dict]`.

Role mapping: `system → SYSTEM`, `developer → DEVELOPER`, `user → USER`, `assistant → ASSISTANT`, `tool`/`function → TOOL_RESULT`; a message with `tool_calls` or legacy `function_call` is reclassified as `TOOL_CALL`. Supported tool fields are round-tripped via `metadata`. `content: None` is normalized to `""`. An unrecognized `role` string raises `ValueError`.

### Anthropic (`adapters/anthropic.py`)

`from_anthropic(messages: list[dict], system: str | None = None) -> list[Message]` / `to_anthropic(messages: list[Message]) -> dict` (returns `{"system": str | None, "messages": [...]}`).

`system`, if given, is prepended as a `SYSTEM` message. String content maps directly to `USER`/`ASSISTANT`. Content-block lists are inspected: blocks with `type == "tool_use"` classify the message as `TOOL_CALL` (converted into OpenAI-shaped `tool_calls` metadata for canonical storage); blocks with `type == "tool_result"` classify it as `TOOL_RESULT` and preserve every `tool_use_id`; otherwise text blocks are concatenated and the role falls back to `USER`/`ASSISTANT`. Untouched provider blocks are preserved. Cross-provider conversions that cannot represent multimodal blocks raise `ValueError` instead of discarding them; Anthropic signed thinking blocks cannot be rewritten safely.

## Errors & Validation

Summary of every validation error surfaced by `migrate()` or a strategy's `transform()`:

| Condition | Exception |
|---|---|
| `messages` is not a `list` | `TypeError` |
| a `messages` element is neither `dict` nor `Message` | `TypeError` |
| a `dict` element has no `"role"` key | `ValueError` |
| a message's `content` is not a `str` | `ValueError` |
| unrecognized OpenAI `role` string | `ValueError` (from `from_openai`) |
| unrecognized Anthropic `role` string | `ValueError` (from `from_anthropic`) |
| unsupported first-element type during format detection | `TypeError` (from `auto_convert`) |
| unknown `strategy` name | `ValueError`, message lists available strategies |
| unknown strategy name in `strategies` list | `ValueError` |
| both `strategy` and `strategies` specified | `ValueError` |
| more than one strategy from the same category in `strategies` | `ValueError` |
| `"raw"` combined with other strategies in `strategies` | `ValueError` |
| unknown `target_format` | `ValueError` |
| unknown `input_format` or input-format/message mismatch | `ValueError` |
| mixed dictionary and `Message` inputs | `TypeError` |
| provider conversion cannot represent a multimodal block, or a tool message is missing required identifiers | `ValueError` |
| `keep_last`: `n < 0` | `ValueError` |
| `token_budget`: `max_tokens <= 0` | `ValueError` |
| `selective_history`: missing `budget` or `budget < 0` | `ValueError` |
| `selective_history`: missing `selector` | `ValueError` |
| async `generate` called from within a running event loop | `RuntimeError` |

## Adding a New Strategy

1. Create a new module in `src/llmigrate/strategies/` with:
   - A `transform(messages: list[Message], **params) -> MigrationResult` function for standalone use.
   - A category-specific function: `select()` for selection, `compress()` for transformation, or `validate()` for validation.
2. If the strategy can drop or rewrite content, call `pinning.split_pinned()` at the top of `transform()` and never touch the pinned half. The pipeline handles pinning centrally for the category-specific functions.
3. Register it in `strategies/__init__.py`: add to `STRATEGY_REGISTRY`, `STRATEGY_CATEGORIES`, and the appropriate category registry (`SELECT_REGISTRY`, `COMPRESS_REGISTRY`, or `VALIDATE_REGISTRY`).
4. Add a `Strategy` enum value in `types.py`.
5. Add tests, including a `TIGHT_PARAMS` entry in `tests/test_pinning.py` so the registry-wide invariant tests cover it.

