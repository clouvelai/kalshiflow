"""
Versioned prompts for V3 agentic research pipeline.

Each version is a separate module that exports a prompt template.
"""

from .event_context_v1 import EVENT_CONTEXT_PROMPT_V1
from .event_context_v2 import EVENT_CONTEXT_PROMPT_V2
from .market_eval_v1 import MARKET_EVAL_PROMPT_V1
from .market_eval_v2 import MARKET_EVAL_PROMPT_V2

__all__ = [
    "EVENT_CONTEXT_PROMPT_V1",
    "EVENT_CONTEXT_PROMPT_V2",
    "MARKET_EVAL_PROMPT_V1",
    "MARKET_EVAL_PROMPT_V2",
]
