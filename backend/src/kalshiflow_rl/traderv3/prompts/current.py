"""
Current Active Prompt Versions.

This module points to the active versions of each prompt.
Change the version here to switch between v1 (baseline), v2 (improved), v3 (base rate anchoring).

Configuration:
    ACTIVE_EVENT_CONTEXT_VERSION: "v1" or "v2"
    ACTIVE_MARKET_EVAL_VERSION: "v1", "v2", "v3", or "v4"

Rollback Instructions:
    To rollback to v1/v2, change the ACTIVE_*_VERSION constants
    and restart the service.

A/B Testing:
    For A/B testing, you can use environment variables:
    - PROMPT_EVENT_CONTEXT_VERSION=v1
    - PROMPT_MARKET_EVAL_VERSION=v2

Version History:
    v1: Baseline prompts
    v2: Added calibration checks, evidence citations, mutual exclusivity
    v3: Added base rate anchoring with explicit adjustment fields and math validation
    v4: Clean calibration-focused (8 fields vs 17), free-form reasoning, no enforcement
"""

import os
from typing import Callable
from langchain_core.prompts import ChatPromptTemplate

from .versions.event_context_v1 import EVENT_CONTEXT_PROMPT_V1
from .versions.event_context_v2 import EVENT_CONTEXT_PROMPT_V2
from .versions.market_eval_v1 import MARKET_EVAL_PROMPT_V1
from .versions.market_eval_v2 import MARKET_EVAL_PROMPT_V2
from .versions.market_eval_v3 import MARKET_EVAL_PROMPT_V3
from .versions.market_eval_v4 import MARKET_EVAL_PROMPT_V4

# === Active Version Configuration ===
# Change these to switch versions. Environment variables override defaults.

ACTIVE_EVENT_CONTEXT_VERSION = os.getenv("PROMPT_EVENT_CONTEXT_VERSION", "v2")
ACTIVE_MARKET_EVAL_VERSION = os.getenv("PROMPT_MARKET_EVAL_VERSION", "v4")

# === Version Maps ===

EVENT_CONTEXT_PROMPTS = {
    "v1": EVENT_CONTEXT_PROMPT_V1,
    "v2": EVENT_CONTEXT_PROMPT_V2,
}

MARKET_EVAL_PROMPTS = {
    "v1": MARKET_EVAL_PROMPT_V1,
    "v2": MARKET_EVAL_PROMPT_V2,
    "v3": MARKET_EVAL_PROMPT_V3,
    "v4": MARKET_EVAL_PROMPT_V4,
}


def get_event_context_prompt(current_date: str, version: str = None) -> ChatPromptTemplate:
    """
    Get the active event context prompt.

    Args:
        current_date: Current date string for prompt injection
        version: Optional version override (defaults to ACTIVE_EVENT_CONTEXT_VERSION)

    Returns:
        ChatPromptTemplate for Phase 2+3 extraction

    Raises:
        ValueError: If requested version doesn't exist
    """
    version = version or ACTIVE_EVENT_CONTEXT_VERSION
    if version not in EVENT_CONTEXT_PROMPTS:
        raise ValueError(f"Unknown event context prompt version: {version}. Available: {list(EVENT_CONTEXT_PROMPTS.keys())}")

    prompt_getter = EVENT_CONTEXT_PROMPTS[version]
    return prompt_getter(current_date)


def get_market_eval_prompt(current_date: str, version: str = None) -> ChatPromptTemplate:
    """
    Get the active market evaluation prompt.

    Args:
        current_date: Current date string for prompt injection
        version: Optional version override (defaults to ACTIVE_MARKET_EVAL_VERSION)

    Returns:
        ChatPromptTemplate for Phase 5 market evaluation

    Raises:
        ValueError: If requested version doesn't exist
    """
    version = version or ACTIVE_MARKET_EVAL_VERSION
    if version not in MARKET_EVAL_PROMPTS:
        raise ValueError(f"Unknown market eval prompt version: {version}. Available: {list(MARKET_EVAL_PROMPTS.keys())}")

    prompt_getter = MARKET_EVAL_PROMPTS[version]
    return prompt_getter(current_date)


def get_prompt_versions() -> dict:
    """
    Get information about available and active prompt versions.

    Returns:
        Dict with version info for monitoring/logging
    """
    return {
        "event_context": {
            "active": ACTIVE_EVENT_CONTEXT_VERSION,
            "available": list(EVENT_CONTEXT_PROMPTS.keys()),
        },
        "market_eval": {
            "active": ACTIVE_MARKET_EVAL_VERSION,
            "available": list(MARKET_EVAL_PROMPTS.keys()),
        },
    }
