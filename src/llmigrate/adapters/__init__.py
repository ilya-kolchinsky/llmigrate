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
from llmigrate.adapters.gemini_interactions import (
    from_gemini_interactions,
    to_gemini_interactions,
)
from llmigrate.adapters.openai import (
    from_openai,
    to_openai,
)
from llmigrate.adapters.openai_responses import (
    from_openai_responses,
    to_openai_responses,
)

__all__ = [
    "from_anthropic",
    "from_gemini_interactions",
    "from_openai",
    "from_openai_responses",
    "to_anthropic",
    "to_gemini_interactions",
    "to_openai",
    "to_openai_responses",
]
