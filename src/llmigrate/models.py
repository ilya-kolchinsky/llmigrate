"""Best-effort model -> context-window lookup.

This table is approximate and may go stale as providers update their model
lineups — treat it as a convenience default, not a source of truth. Unknown or
self-hosted model names (e.g. a custom checkpoint served via vLLM) return
None; callers should pass max_tokens explicitly in that case, or register the
model's context window with register_model_context_window().
"""

from __future__ import annotations

_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o3": 200_000,
    "claude-opus-4": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-haiku-4": 200_000,
    "claude-3-7-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-5-haiku": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-2.0": 1_000_000,
    "llama-3.1": 128_000,
    "llama-3": 8_192,
    "mixtral": 32_768,
}

_custom_context_windows: dict[str, int] = {}


def register_model_context_window(name_or_prefix: str, tokens: int) -> None:
    """Teach context_window_for() about a model it doesn't already know — e.g.
    a self-hosted model served via vLLM/LocalAI under a custom name."""
    _custom_context_windows[name_or_prefix] = tokens


def context_window_for(model: str | None) -> int | None:
    """Best-effort context window lookup by prefix match. None if unknown."""
    if not model:
        return None

    for table in (_custom_context_windows, _CONTEXT_WINDOWS):
        for prefix, size in table.items():
            if model.startswith(prefix):
                return size
    return None
