"""Registry-wide invariant tests: no strategy may drop the protected content
(system + first user message), and no strategy except raw should produce
consecutive same-role output (which providers like Anthropic reject)."""

from __future__ import annotations

import pytest

import llmigrate
from llmigrate.alternation import enforce_alternation
from llmigrate.pinning import split_pinned
from llmigrate.selectors import PrioritySelector
from llmigrate.strategies import STRATEGY_REGISTRY
from llmigrate.types import Message, Role

MARKER = "TASK-MARKER-b17b3f: implement the fix in foo.py"

FIXTURE = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": MARKER},
    {"role": "assistant", "content": "Sure, let's start."},
    {"role": "user", "content": "Here's more context."},
    {"role": "assistant", "content": "Got it."},
    {"role": "user", "content": "One more thing."},
    {"role": "assistant", "content": "Noted."},
]


def _mock_generate(messages):
    return "a model-generated summary that does not repeat the task"


TIGHT_PARAMS: dict[str, dict] = {
    "raw": {},
    "keep_last": {"n": 0},
    "token_budget": {"max_tokens": 1},
    "summarize": {"generate": _mock_generate},
    "structured_state": {},
    "audit": {},
    "selective_history": {"budget": 0, "selector": PrioritySelector({})},
}


@pytest.mark.parametrize("strategy", sorted(STRATEGY_REGISTRY.keys()))
def test_preserves_task_marker(strategy):
    result = llmigrate.migrate(FIXTURE, strategy=strategy, **TIGHT_PARAMS[strategy])
    assert any(MARKER in m["content"] for m in result.messages), (
        f"{strategy} dropped or diluted the pinned task content"
    )


@pytest.mark.parametrize("strategy", sorted(s for s in STRATEGY_REGISTRY if s != "raw"))
def test_no_consecutive_same_role_output(strategy):
    result = llmigrate.migrate(FIXTURE, strategy=strategy, **TIGHT_PARAMS[strategy])
    roles = [m["role"] for m in result.messages if m["role"] != "system"]
    for a, b in zip(roles, roles[1:]):
        assert a != b, f"{strategy} produced consecutive {a} messages"


class TestSplitPinned:
    def test_pins_system_and_first_user(self):
        messages = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="task"),
            Message(role=Role.ASSISTANT, content="reply"),
            Message(role=Role.USER, content="followup"),
        ]
        pinned, rest = split_pinned(messages)
        assert [m.content for m in pinned] == ["sys", "task"]
        assert [m.content for m in rest] == ["reply", "followup"]

    def test_pin_first_user_false(self):
        messages = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="task"),
        ]
        pinned, rest = split_pinned(messages, pin_first_user=False)
        assert [m.content for m in pinned] == ["sys"]
        assert [m.content for m in rest] == ["task"]

    def test_explicit_metadata_pin(self):
        messages = [
            Message(role=Role.USER, content="a"),
            Message(role=Role.ASSISTANT, content="b", metadata={"pinned": True}),
        ]
        pinned, rest = split_pinned(messages)
        assert [m.content for m in pinned] == ["a", "b"]
        assert rest == []


class TestEnforceAlternation:
    def test_merges_consecutive_same_role(self):
        messages = [
            Message(role=Role.USER, content="one"),
            Message(role=Role.USER, content="two"),
            Message(role=Role.ASSISTANT, content="three"),
        ]
        merged = enforce_alternation(messages)
        assert len(merged) == 2
        assert merged[0].content == "one\n\ntwo"

    def test_preserves_alternating_input(self):
        messages = [Message(role=Role.USER, content="a"), Message(role=Role.ASSISTANT, content="b")]
        assert enforce_alternation(messages) == messages

    def test_system_messages_never_merge(self):
        messages = [
            Message(role=Role.SYSTEM, content="sys1"),
            Message(role=Role.SYSTEM, content="sys2"),
        ]
        assert enforce_alternation(messages) == messages
