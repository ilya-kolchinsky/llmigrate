"""Migration strategies.

Each strategy module exposes category-specific functions:
- Selection: select(rest, **params) -> SelectionResult
- Transformation: compress(dropped, **params) -> CompressionResult
- Validation: validate(messages, **params) -> ValidationResult

Legacy transform() entry points are kept for single-strategy use via
the `strategy=` parameter on migrate().
"""

from llmigrate.strategies.audit import transform as audit
from llmigrate.strategies.audit import validate as validate_audit
from llmigrate.strategies.keep_last import select as select_keep_last
from llmigrate.strategies.keep_last import transform as keep_last
from llmigrate.strategies.raw import transform as raw
from llmigrate.strategies.selective_history import select as select_selective_history
from llmigrate.strategies.selective_history import transform as selective_history
from llmigrate.strategies.structured_state import compress as compress_structured_state
from llmigrate.strategies.structured_state import compress_async as compress_structured_state_async
from llmigrate.strategies.structured_state import transform as structured_state
from llmigrate.strategies.summarize import compress as compress_summarize
from llmigrate.strategies.summarize import compress_async as compress_summarize_async
from llmigrate.strategies.summarize import transform as summarize
from llmigrate.strategies.token_budget import select as select_token_budget
from llmigrate.strategies.token_budget import transform as token_budget
from llmigrate.types import StrategyCategory

STRATEGY_REGISTRY: dict[str, object] = {
    "raw": raw,
    "keep_last": keep_last,
    "token_budget": token_budget,
    "summarize": summarize,
    "structured_state": structured_state,
    "audit": audit,
    "selective_history": selective_history,
}

STRATEGY_CATEGORIES: dict[str, StrategyCategory] = {
    "keep_last": StrategyCategory.SELECTION,
    "token_budget": StrategyCategory.SELECTION,
    "selective_history": StrategyCategory.SELECTION,
    "summarize": StrategyCategory.TRANSFORMATION,
    "structured_state": StrategyCategory.TRANSFORMATION,
    "audit": StrategyCategory.VALIDATION,
}

SELECT_REGISTRY: dict[str, object] = {
    "keep_last": select_keep_last,
    "token_budget": select_token_budget,
    "selective_history": select_selective_history,
}

COMPRESS_REGISTRY: dict[str, object] = {
    "summarize": compress_summarize,
    "structured_state": compress_structured_state,
}

ASYNC_COMPRESS_REGISTRY: dict[str, object] = {
    "summarize": compress_summarize_async,
    "structured_state": compress_structured_state_async,
}

VALIDATE_REGISTRY: dict[str, object] = {
    "audit": validate_audit,
}

__all__ = [
    "VALIDATE_REGISTRY",
    "ASYNC_COMPRESS_REGISTRY",
    "COMPRESS_REGISTRY",
    "SELECT_REGISTRY",
    "STRATEGY_CATEGORIES",
    "STRATEGY_REGISTRY",
    "audit",
    "structured_state",
    "keep_last",
    "raw",
    "selective_history",
    "summarize",
    "token_budget",
]
