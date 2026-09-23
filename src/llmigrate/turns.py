"""Turn-boundary grouping shared by truncation strategies.

A "turn" starts at a USER message and includes everything up to (not
including) the next USER message, so an assistant reply plus any interleaved
TOOL_CALL/TOOL_RESULT messages stay together. Truncation strategies that
operate on whole turns therefore never split a turn pair or orphan a tool
result from its tool call.
"""

from __future__ import annotations

from llmigrate.types import Message, Role


def group_into_turns(messages: list[Message]) -> list[list[Message]]:
    """Group messages into turns. A leading run before the first USER message
    (if any) forms its own group."""
    groups: list[list[Message]] = []
    for msg in messages:
        if msg.role == Role.USER or not groups:
            groups.append([msg])
        else:
            groups[-1].append(msg)
    return groups
