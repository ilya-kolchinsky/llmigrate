"""Small internal helpers shared across strategies."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, cast


def call_generate(generate: Callable[..., Any], prompt: Any, **kwargs: Any) -> str:
    """Call a `generate` callable, auto-detecting and awaiting async callables."""
    if inspect.iscoroutinefunction(generate):
        try:
            return cast(str, asyncio.run(generate(prompt, **kwargs)))
        except RuntimeError as e:
            raise RuntimeError(
                "generate is async but transfer() was called from within a "
                "running event loop; call transfer() from synchronous code."
            ) from e
    return cast(str, generate(prompt, **kwargs))
