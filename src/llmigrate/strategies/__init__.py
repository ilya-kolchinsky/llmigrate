"""Migration strategies.

Each strategy implements the transform() protocol:
    transform(messages, **params) -> TransferResult
"""

from llmigrate.strategies.audit import transform as audit
from llmigrate.strategies.capsule import transform as capsule
from llmigrate.strategies.keep_last import transform as keep_last
from llmigrate.strategies.raw import transform as raw
from llmigrate.strategies.selective_history import transform as selective_history
from llmigrate.strategies.summarize import transform as summarize
from llmigrate.strategies.summary_tail import transform as summary_tail
from llmigrate.strategies.token_budget import transform as token_budget

STRATEGY_REGISTRY: dict[str, object] = {
    "raw": raw,
    "keep_last": keep_last,
    "token_budget": token_budget,
    "summarize": summarize,
    "capsule": capsule,
    "audit": audit,
    "selective_history": selective_history,
    "summary_tail": summary_tail,
}

__all__ = [
    "STRATEGY_REGISTRY",
    "audit",
    "capsule",
    "keep_last",
    "raw",
    "selective_history",
    "summarize",
    "summary_tail",
    "token_budget",
]
