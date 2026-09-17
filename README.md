# llmigrate

Cross-model session migration for LLM conversations.

When you switch models mid-conversation — for cost optimization, capability routing, context window management, or failover — the conversation history needs to be transformed to work well with the new model. **llmigrate** provides a single `transfer()` call that handles this transformation using pluggable strategies.

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
| `keep_last` | Keep system messages + last N turns | No |
| `token_budget` | Keep as many recent turns as fit within a token budget | No |
| `summarize` | Summarize older history, keep recent tail | Yes |
| `capsule` | Extract structured state (objective, progress, key facts) | Yes |
| `audit` | Append verification instructions for the receiving model | No |

## Model-Assisted Strategies

Strategies that need a model call (`summarize`, `capsule`) accept a `generate` callable:

```python
result = llmigrate.transfer(
    messages,
    strategy="summarize",
    generate=my_generate_fn,  # Callable[[list[dict]], str]
    tail=3,
)
```

The `generate` function takes a list of messages and returns a string completion. This keeps llmigrate decoupled from any specific provider SDK.

## License

MIT
