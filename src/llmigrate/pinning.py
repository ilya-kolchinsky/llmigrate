"""Framework-level protected content — messages no strategy may drop or rewrite.

Truncation and summarization strategies must never lose or dilute critical
context (e.g. a task description in the first user message — see e.g. the
SWE-bench use case, where losing it leaves the receiving model unable to
succeed). This module is the single place that decides what is protected;
every strategy that removes or rewrites content calls it instead of
reimplementing its own ad hoc filtering.
"""

from __future__ import annotations

from llmigrate.types import Message, Role


def split_pinned(
    messages: list[Message], *, pin_first_user: bool = True
) -> tuple[list[Message], list[Message]]:
    """Split messages into (pinned, rest).

    Pinned messages are never dropped, rewritten, or fed through a
    summarizer by any strategy — they are always emitted verbatim. A message is
    pinned if any of the following hold:
    - its role is SYSTEM or DEVELOPER
    - it is the first USER message, and pin_first_user is True (the default)
    - it carries metadata["pinned"] = True (an explicit escape hatch for callers
      who want to protect something else, e.g. a prior transfer's structured state)
    """
    pinned: list[Message] = []
    rest: list[Message] = []
    first_user_seen = False

    for msg in messages:
        is_pinned = (
            msg.role in (Role.SYSTEM, Role.DEVELOPER)
            or msg.metadata.get("pinned") is True
            or (pin_first_user and not first_user_seen and msg.role == Role.USER)
        )
        if msg.role == Role.USER:
            first_user_seen = True

        (pinned if is_pinned else rest).append(msg)

    return pinned, rest
