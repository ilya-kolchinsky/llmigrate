"""Small internal helpers shared across strategies."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any, cast

from llmigrate.types import Message


def message_to_text(message: Message) -> str:
    """Render canonical message text plus tool and media context for prompts."""
    parts = [message.content] if message.content else []
    tool_calls = message.metadata.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            if not isinstance(function, dict):
                function = {}
            arguments = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, default=str)
            parts.append(
                f"tool_call id={call.get('id', '')} name={function.get('name', '')} "
                f"arguments={arguments}"
            )

    function_call = message.metadata.get("function_call")
    if isinstance(function_call, dict):
        parts.append(
            f"function_call name={function_call.get('name', '')} "
            f"arguments={function_call.get('arguments', '')}"
        )

    tool_results = message.metadata.get("tool_results")
    if isinstance(tool_results, list):
        ids = [
            str(item["tool_call_id"])
            for item in tool_results
            if isinstance(item, dict) and item.get("tool_call_id") is not None
        ]
        if ids:
            parts.append(f"tool_result ids={', '.join(ids)}")
    elif message.metadata.get("tool_call_id") is not None:
        parts.append(f"tool_result id={message.metadata['tool_call_id']}")

    provider_blocks = message.metadata.get("openai_content")
    if provider_blocks is None:
        provider_blocks = message.metadata.get("anthropic_content")
    parts.extend(_media_markers(provider_blocks))

    body = "\n".join(parts) or "[no text content]"
    return f"{message.role.value}: {body}"


def _media_markers(value: Any) -> list[str]:
    markers: list[str] = []
    if isinstance(value, dict):
        block_type = value.get("type")
        if isinstance(block_type, str) and block_type not in {
            "text",
            "input_text",
            "output_text",
            "tool_use",
            "tool_result",
        }:
            markers.append(f"[{block_type} content]")
        for key, child in value.items():
            if key not in {"source", "data", "url"}:
                markers.extend(_media_markers(child))
    elif isinstance(value, list):
        for child in value:
            markers.extend(_media_markers(child))
    return markers


def call_generate(generate: Callable[..., Any], prompt: Any, **kwargs: Any) -> str:
    """Call a sync or async callable from synchronous code."""
    result = generate(prompt, **kwargs)
    if not inspect.isawaitable(result):
        return cast(str, result)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        async def await_result() -> Any:
            return await result

        return cast(str, asyncio.run(await_result()))
    if inspect.iscoroutine(result):
        result.close()
    raise RuntimeError(
        "generate returned an awaitable but migrate() was called from within a "
        "running event loop; use async_migrate() instead."
    )


async def call_generate_async(
    generate: Callable[..., Any], prompt: Any, **kwargs: Any
) -> str:
    """Call sync or async generation without blocking the running event loop."""
    if inspect.iscoroutinefunction(generate) or inspect.iscoroutinefunction(
        getattr(generate, "__call__", None)
    ):
        result = generate(prompt, **kwargs)
    else:
        result = await asyncio.to_thread(generate, prompt, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return cast(str, result)
