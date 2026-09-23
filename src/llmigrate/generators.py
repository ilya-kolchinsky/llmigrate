"""Convenience `generate` callables for OpenAI-compatible endpoints — OpenAI
itself, and self-hosted/alternative servers that speak the same Chat
Completions wire format (vLLM, LocalAI, LM Studio, Ollama's OpenAI-compat
mode, Together, Groq, Fireworks, ...).

Requires the optional `openai` package unless you pass your own `client`:
    pip install llmigrate[openai]
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any


def openai_compatible_generate(
    base_url: str,
    model: str,
    api_key: str | None = None,
    client: Any = None,
    **default_kwargs: Any,
) -> Callable[..., str]:
    """Build a sync `generate` callable pointed at any OpenAI-compatible endpoint.

    `default_kwargs` (e.g. temperature) are applied to every call and
    overridden by transfer()'s `generate_kwargs` on a per-call basis.
    """
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai_compatible_generate requires the 'openai' package: "
                "pip install llmigrate[openai]"
            ) from e
        client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def generate(messages: list[dict[str, Any]], **kwargs: Any) -> str:
        response = client.chat.completions.create(
            model=model, messages=messages, **{**default_kwargs, **kwargs}
        )
        return response.choices[0].message.content or ""

    return generate


def async_openai_compatible_generate(
    base_url: str,
    model: str,
    api_key: str | None = None,
    client: Any = None,
    **default_kwargs: Any,
) -> Callable[..., Awaitable[str]]:
    """Async counterpart of openai_compatible_generate — plugs directly into
    transfer()'s automatic async-`generate` detection."""
    if client is None:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "async_openai_compatible_generate requires the 'openai' package: "
                "pip install llmigrate[openai]"
            ) from e
        client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed")

    async def generate(messages: list[dict[str, Any]], **kwargs: Any) -> str:
        response = await client.chat.completions.create(
            model=model, messages=messages, **{**default_kwargs, **kwargs}
        )
        return response.choices[0].message.content or ""

    return generate
