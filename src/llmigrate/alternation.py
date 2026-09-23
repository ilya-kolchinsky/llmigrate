"""Framework-level role-alternation enforcement, applied centrally in transfer().

Many providers (notably Anthropic) reject a message list with consecutive
same-role turns. Truncation/summarization strategies can incidentally produce
these — e.g. a pinned first-user message immediately followed by a synthetic
USER-role summary. Merging generically here, rather than special-casing each
strategy, also guarantees pinned content survives verbatim inside the merged
message even if a model-generated summary doesn't reproduce it.
"""

from __future__ import annotations

from llmigrate.types import Message, Role


def enforce_alternation(messages: list[Message]) -> list[Message]:
    """Merge consecutive same-role, non-SYSTEM messages into one."""
    if not messages:
        return []

    result: list[Message] = [messages[0]]
    for msg in messages[1:]:
        prev = result[-1]
        if msg.role != Role.SYSTEM and msg.role == prev.role:
            merged_metadata = {**prev.metadata, **msg.metadata}
            if prev.metadata.get("llmigrate_synthetic") or msg.metadata.get("llmigrate_synthetic"):
                merged_metadata["llmigrate_synthetic"] = True
            result[-1] = Message(
                role=prev.role,
                content=f"{prev.content}\n\n{msg.content}",
                metadata=merged_metadata,
            )
        else:
            result.append(msg)
    return result
