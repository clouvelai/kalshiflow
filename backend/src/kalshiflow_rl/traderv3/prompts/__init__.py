"""
V3 Agentic Research Prompts - Versioned Prompt Management.

This module provides versioned prompts for the event-first research pipeline.
Prompts are organized by phase and version to enable A/B testing and rollback.

Directory Structure:
    prompts/
        __init__.py          # This file - exports active versions
        current.py           # Points to active versions (v1 or v2)
        versions/
            __init__.py
            event_context_v1.py   # Phase 2+3 prompt v1
            event_context_v2.py   # Phase 2+3 prompt v2 (improved)
            market_eval_v1.py     # Phase 5 prompt v1
            market_eval_v2.py     # Phase 5 prompt v2 (improved)
        schemas/
            __init__.py
            full_context_schema.py      # FullContextOutput schema
            market_assessment_schema.py # SingleMarketAssessment schema

Usage:
    from traderv3.prompts import get_event_context_prompt, get_market_eval_prompt
    from traderv3.prompts.schemas import FullContextOutput, SingleMarketAssessment
"""

from .current import (
    get_event_context_prompt,
    get_market_eval_prompt,
    ACTIVE_EVENT_CONTEXT_VERSION,
    ACTIVE_MARKET_EVAL_VERSION,
)

from .schemas import (
    FullContextOutput,
    SemanticRoleOutput,
    SingleMarketAssessment,
    BatchMarketAssessmentOutput,
)

__all__ = [
    # Prompt getters
    "get_event_context_prompt",
    "get_market_eval_prompt",
    # Version tracking
    "ACTIVE_EVENT_CONTEXT_VERSION",
    "ACTIVE_MARKET_EVAL_VERSION",
    # Schemas
    "FullContextOutput",
    "SemanticRoleOutput",
    "SingleMarketAssessment",
    "BatchMarketAssessmentOutput",
]
