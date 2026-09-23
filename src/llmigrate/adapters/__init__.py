"""Format adapters — convert between provider-native and canonical formats.

Each adapter implements two functions:
    from_native(messages) -> list[Message]    # provider format -> canonical
    to_native(messages) -> list[dict/etc.]    # canonical -> provider format

OpenAI-compatible self-hosted servers (vLLM, LocalAI, LM Studio, Ollama's
OpenAI-compat mode, etc.) speak the same wire format as OpenAI, so the OpenAI
adapter below already covers them — no separate adapter is needed.
"""

from llmigrate.adapters.anthropic import (
    from_anthropic,
    to_anthropic,
)
from llmigrate.adapters.openai import (
    from_openai,
    to_openai,
)

__all__ = [
    "from_anthropic",
    "from_openai",
    "to_anthropic",
    "to_openai",
]
