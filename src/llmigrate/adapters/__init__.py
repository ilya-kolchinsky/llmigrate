"""Format adapters — convert between provider-native and canonical formats.

Each adapter implements two functions:
    from_native(messages) -> list[Message]    # provider format -> canonical
    to_native(messages) -> list[dict/etc.]    # canonical -> provider format
"""

from llmigrate.adapters.openai import (
    from_openai,
    to_openai,
)

__all__ = [
    "from_openai",
    "to_openai",
]
