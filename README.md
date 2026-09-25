# llmigrate

Cross-model session migration for LLM conversations.

Model routing is now routine: you start on a big reasoning model and hand off to a cheap one once the hard part is done, you fail over when a provider has an outage, you escalate from a tier-1 to a tier-2 support model, or you pass a research agent's findings to a writing agent. In every one of these cases, the conversation history that made sense for the old model doesn't automatically make sense for the new one — it may blow the new model's context window, bury the task under stale exploration, or omit the structure a receiving agent needs to pick up the work. Naively forwarding the raw transcript is often the wrong move, and hand-rolling the fix for every call site is how context gets silently dropped.

**llmigrate** gives you `migrate()` and `async_migrate()`: pass a conversation and a strategy, and get back history shaped for the next model — truncated, summarized, distilled into structured state, or filtered down to the highest-value events. Whatever strategy you pick, one guarantee never moves: your system prompt and the task the user actually asked for are never dropped or paraphrased away, even under the most aggressive compression.

## Why llmigrate

- **One call, swappable strategies.** `migrate(messages, strategy="...")` — going from "keep the last few turns" to "summarize everything but the tail" to "extract a structured handoff state" is a one-line change, not a rewrite.
- **Protected content, enforced centrally.** No strategy — truncation, summarization, or selection — can drop or dilute the system prompt or the first user message. This is checked by a framework-level guarantee and a registry-wide test, not left to each strategy's discretion. See [Protected Content](docs/API.md#protected-content-pinning).
- **Provider-agnostic at the transcript layer.** Pass OpenAI Chat Completions messages, OpenAI Responses items, Anthropic Messages, Gemini Interactions steps, or canonical `Message` objects. Model-assisted strategies take a plain `generate` callable rather than depending on a provider SDK.
- **Zero required dependencies.** Core functionality is pure standard library. `tiktoken` (accurate token counts) and `openai` (ready-made `generate` builders) are optional extras.
- **Built for the messy cases**, not just clean chat transcripts: turn-boundary grouping keeps tool calls paired with their results, role-alternation is fixed up automatically for providers that reject consecutive same-role turns, and every model call reports latency/token accounting so migrations stay observable.

## Installation

```bash
pip install llmigrate
```

## Quick Start

```python
import llmigrate

# Your conversation history (provider-native format or llmigrate's canonical format)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum computing."},
    {"role": "assistant", "content": "Quantum computing uses qubits..."},
    {"role": "user", "content": "How does entanglement work?"},
    {"role": "assistant", "content": "Entanglement is a phenomenon..."},
    {"role": "user", "content": "Can you give me a practical example?"},
]

# Migrate the session to a new model, keeping the last 3 turns
result = llmigrate.migrate(messages, strategy="keep_last", n=3)

# provider_messages removes llmigrate's per-message metadata before sending
response = target_client.chat(messages=result.provider_messages)
```

For Anthropic, pass `result.system` as the separate `system` argument alongside
`result.provider_messages`. Cross-provider image, audio, document, and other
provider-specific blocks are not silently discarded: convert them explicitly
or keep the target format compatible with the source content.

For OpenAI Responses and Gemini Interactions, pass `result.provider_messages`
as `input`; pass `result.system` as `instructions` or `system_instruction`,
respectively.

### Async agent workflows

Use `async_migrate()` when your agent runtime already has an event loop. Async
model callbacks are awaited directly, and synchronous callbacks run in a worker
thread:

```python
result = await llmigrate.async_migrate(
    messages,
    strategies=["selective_history", "summarize"],
    budget=8_000,
    selector=my_selector,
    generate=async_generate,
)
```

See the [API reference](docs/API.md#async_migrate) for async summarizers,
callback behavior, and parameter details.

## Supported Data

llmigrate transforms **text conversations and structured tool-call/result
records**. It does not understand or convert image, audio, video, or document
payloads. Existing OpenAI and Anthropic adapters can preserve some untouched
provider blocks in same-format raw migrations, but that is pass-through, not
multimodal migration support. Content-changing strategies may omit media from
compressed history, and cross-format conversions can reject unsupported
payloads. OpenAI Responses and Gemini Interactions adapters currently accept
text and function-call/result items only. See [the format and limitation
details](docs/API-details.md#supported-data-and-conversion-limits).

Swap `strategy="keep_last"` for `"summarize"`, `"structured_state"`, `"token_budget"`, `"selective_history"`, `"audit"`, or `"raw"` to change how the handoff is shaped — see the [strategy reference](docs/API.md#strategies) for what each one does and which use cases it fits.

Strategies can be composed — keep the last 2 turns verbatim, summarize the rest, and append a verification instruction:

```python
result = llmigrate.migrate(
    messages,
    strategies=["keep_last", "summarize", "audit"],
    n=2,
    generate=my_generate_fn,
)
```

## Protected Content

The system prompt and the first user message (e.g. a task description) are always preserved verbatim, never fed to a summarizer, and never dropped — regardless of how aggressive the transformation is:

```python
# Even with n=0, the task description survives.
result = llmigrate.migrate(messages, strategy="keep_last", n=0)
```

Disable this for a specific call with `pin_first_user=False`, or protect an arbitrary message yourself with `metadata["pinned"] = True`. Full semantics: [Protected Content](docs/API.md#protected-content-pinning).

## Documentation

The [API reference](docs/API.md) covers `migrate()` and `async_migrate()`, every strategy and its parameters/metadata, the pluggable `Selector`/`Summarizer` abstractions, format adapters, and the framework-level pinning/alternation guarantees. [Adapter details](docs/API-details.md) describe supported data shapes and conversion limits.

## License

MIT
