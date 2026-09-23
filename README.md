# llmigrate

Cross-model session migration for LLM conversations.

When you switch models mid-conversation — for cost optimization, capability routing, context window management, or failover — the conversation history needs to be transformed to work well with the new model. **llmigrate** provides a single `transfer()` call that handles this transformation using pluggable strategies.

No strategy will ever drop or dilute your system prompt or the first user message (e.g. a task description) — that's enforced at the framework level, not left up to each strategy. See [Protected Content](#protected-content) below.

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
result = llmigrate.transfer(messages, strategy="keep_last", n=3)

# result.messages is ready to send to the target model
response = target_client.chat(messages=result.messages)
```

## Strategies

| Strategy | Description | Requires model call |
|---|---|---|
| `raw` | Pass conversation unchanged | No |
| `keep_last` | Keep pinned content + last N turns | No |
| `token_budget` | Keep as many recent turns as fit within a token budget | No |
| `summarize` | Summarize older history, keep recent tail | Yes |
| `capsule` | Extract structured state (objective, progress, key facts) | Yes |
| `audit` | Append verification instructions for the receiving model | No |
| `selective_history` | Verbatim, budget-constrained selection of events, ranked by a pluggable selector | No |
| `summary_tail` | Summarized prefix + verbatim, turn-aligned tail, with cost/latency accounting | Yes |

## Protected Content

Every strategy that can drop or rewrite content is built on a shared, framework-level guarantee: the system prompt and the first user message (e.g. a task description) are always preserved verbatim, never fed to a summarizer, and never dropped — regardless of how aggressive the truncation is:

```python
# Even with n=0, the task description survives.
result = llmigrate.transfer(messages, strategy="keep_last", n=0)
```

Disable this for a specific call with `pin_first_user=False`, or protect an arbitrary message yourself with `metadata["pinned"] = True`.

## Model-Assisted Strategies

Strategies that need a model call (`summarize`, `capsule`, `summary_tail`) accept a `generate` callable, which may be sync or async:

```python
result = llmigrate.transfer(
    messages,
    strategy="summarize",
    generate=my_generate_fn,       # Callable[[list[dict]], str], sync or async
    generate_kwargs={"temperature": 0.2},
    tail=3,
)
```

The `generate` function takes a list of messages and returns a string completion. This keeps llmigrate decoupled from any specific provider SDK.

### OpenAI-compatible endpoints (including vLLM, LocalAI, ...)

For OpenAI itself or any OpenAI-compatible server, `generators.py` builds a ready-made `generate` callable (requires `pip install llmigrate[openai]`):

```python
from llmigrate.generators import openai_compatible_generate

generate = openai_compatible_generate(
    base_url="http://localhost:8000/v1",  # e.g. a local vLLM server
    model="meta-llama/Llama-3-70b-instruct",
)
result = llmigrate.transfer(messages, strategy="summarize", generate=generate)
```

An async variant, `async_openai_compatible_generate`, is also available.

## Selective Retention and Summary+Tail

`selective_history` never rewrites content — it selects a verbatim subset of events, ranked by a pluggable selector, that fits a token budget:

```python
from llmigrate.selectors import PrioritySelector

result = llmigrate.transfer(
    messages,
    strategy="selective_history",
    budget=4000,
    selector=PrioritySelector(priorities={"user_instruction": 100, "tool_result": 80, "assistant": 20}),
    always_keep={"tool_result"},
)
```

`summary_tail` summarizes older history while keeping the most recent turns verbatim, with cost/latency accounting for the summarization call:

```python
result = llmigrate.transfer(
    messages,
    strategy="summary_tail",
    total_budget=8000,
    summary_budget=1000,
    generate=my_generate_fn,
)
print(result.metadata["latency_ms"], result.metadata["token_usage"])
```

## Format Support

Pass OpenAI-format dicts, Anthropic-format dicts (auto-detected), or llmigrate's canonical `Message` objects — `transfer()` figures out which. Convert the result back with `result.to_openai()` or `result.to_anthropic()`.

## License

MIT
