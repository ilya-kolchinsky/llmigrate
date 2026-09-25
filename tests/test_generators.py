"""Tests for OpenAI-compatible generate() convenience builders.

These inject a fake client, so they exercise the real logic without requiring
the optional `openai` package or any network access.
"""

from __future__ import annotations

import builtins

import pytest

from llmigrate.generators import async_openai_compatible_generate, openai_compatible_generate


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse("mock response")


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeCompletions()})()


class _FakeAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse("mock async response")


class _FakeAsyncClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _FakeAsyncCompletions()})()


def test_sync_generate_calls_client_and_extracts_content():
    client = _FakeClient()
    generate = openai_compatible_generate(
        base_url="http://localhost:8000/v1", model="my-model", client=client, temperature=0.1
    )
    result = generate([{"role": "user", "content": "hi"}])
    assert result == "mock response"
    call = client.chat.completions.calls[0]
    assert call["model"] == "my-model"
    assert call["temperature"] == 0.1
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_sync_generate_per_call_kwargs_override_defaults():
    client = _FakeClient()
    generate = openai_compatible_generate(
        base_url="http://localhost:8000/v1", model="my-model", client=client, temperature=0.1
    )
    generate([{"role": "user", "content": "hi"}], temperature=0.9)
    assert client.chat.completions.calls[0]["temperature"] == 0.9


async def test_async_generate_calls_client_and_extracts_content():
    client = _FakeAsyncClient()
    generate = async_openai_compatible_generate(
        base_url="http://localhost:8000/v1", model="my-model", client=client
    )
    result = await generate([{"role": "user", "content": "hi"}])
    assert result == "mock async response"


def test_missing_openai_package_raises_helpful_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("no module named openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"python -m pip install openai"):
        openai_compatible_generate(base_url="http://localhost:8000/v1", model="my-model")
