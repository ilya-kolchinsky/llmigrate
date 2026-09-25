"""Turn-boundary grouping shared by truncation strategies.

A "turn" starts at a USER message and includes everything up to (not
including) the next USER message, so an assistant reply plus any interleaved
TOOL_CALL/TOOL_RESULT messages stay together. Truncation strategies that
operate on whole turns therefore never split a turn pair or orphan a tool
result from its tool call.
"""

from __future__ import annotations

from collections import defaultdict

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


def _tool_call_ids(message: Message) -> set[str]:
    """Return normalized call IDs referenced by a canonical message."""
    ids: set[str] = set()
    calls = message.metadata.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict):
                call_id = call.get("id") or call.get("call_id")
                if call_id is not None:
                    ids.add(str(call_id))

    function_call = message.metadata.get("function_call")
    if isinstance(function_call, dict):
        call_id = function_call.get("id") or function_call.get("call_id")
        if call_id is not None:
            ids.add(str(call_id))

    call_id = message.metadata.get("tool_call_id")
    if call_id is not None:
        ids.add(str(call_id))

    results = message.metadata.get("tool_results")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict):
                call_id = result.get("tool_call_id") or result.get("call_id")
                if call_id is not None:
                    ids.add(str(call_id))
    return ids


def group_tool_events(messages: list[Message]) -> list[list[Message]]:
    """Group messages connected by tool-call IDs into indivisible events.

    Ordinary messages remain individual events. Parallel calls recorded in one
    message, their matching results, and any result message containing several
    results are joined into one event. Event groups retain input order.
    """
    if not messages:
        return []

    parent = list(range(len(messages)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    first_for_id: dict[str, int] = {}
    for index, message in enumerate(messages):
        ids = _tool_call_ids(message)
        if message.role not in (Role.TOOL_CALL, Role.TOOL_RESULT) and not ids:
            continue
        for call_id in ids:
            previous = first_for_id.setdefault(call_id, index)
            union(previous, index)

    grouped: dict[int, list[Message]] = defaultdict(list)
    order: list[int] = []
    for index, message in enumerate(messages):
        root = find(index)
        if root not in grouped:
            order.append(root)
        grouped[root].append(message)
    return [grouped[root] for root in order]
