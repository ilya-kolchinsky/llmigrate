"""Migration strategies.

Each strategy implements the transform() protocol:
    transform(messages, **params) -> TransferResult
"""

from llmigrate.strategies.audit import transform as audit
from llmigrate.strategies.capsule import transform as capsule
from llmigrate.strategies.keep_last import transform as keep_last
from llmigrate.strategies.raw import transform as raw
from llmigrate.strategies.summarize import transform as summarize
from llmigrate.strategies.token_budget import transform as token_budget

STRATEGY_REGISTRY: dict[str, object] = {
    "raw": raw,
    "keep_last": keep_last,
    "token_budget": token_budget,
    "summarize": summarize,
    "capsule": capsule,
    "audit": audit,
}

__all__ = [
    "STRATEGY_REGISTRY",
    "audit",
    "capsule",
    "keep_last",
    "raw",
    "summarize",
    "token_budget",
]
