"""
Centralized LLM model configuration for mentions system and all V3 consumers.

Tier system (configured via V3Config or env vars):
- captain: Main Captain agent (Sonnet default)
- subagent: EventUnderstanding (Haiku default)
- utility: Mentions sim/extraction (Gemini Flash default)
- embedding: Vector store (text-embedding-3-small default)

Supports: Anthropic (Haiku/Sonnet), Google (Gemini), OpenAI (GPT-4o-mini)
"""

import logging
import os
from typing import Literal, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.mentions_models")

# =============================================================================
# MODEL TIER DEFAULTS — set by configure() from V3Config at startup
# =============================================================================

_captain_model: str = "claude-sonnet-4-20250514"
_subagent_model: str = "claude-haiku-4-5-20251001"
_utility_model: str = "gemini-2.0-flash"
_embedding_model: str = "text-embedding-3-small"
_configured: bool = False

# Legacy env var overrides for simulation/extraction (deprecated)
_simulation_model_override: Optional[str] = None
_extraction_model_override: Optional[str] = None


def configure(config) -> None:
    """Initialize model tiers from V3Config. Call once at startup.

    Args:
        config: V3Config instance with model_captain, model_subagent, etc.
    """
    global _captain_model, _subagent_model, _utility_model, _embedding_model
    global _simulation_model_override, _extraction_model_override, _configured

    _captain_model = getattr(config, "model_captain", _captain_model)
    _subagent_model = getattr(config, "model_subagent", _subagent_model)
    _utility_model = getattr(config, "model_utility", _utility_model)
    _embedding_model = getattr(config, "model_embedding", _embedding_model)

    # Check for deprecated per-consumer env var overrides
    sim_override = os.environ.get("MENTIONS_SIMULATION_MODEL")
    if sim_override:
        logger.warning(
            f"MENTIONS_SIMULATION_MODEL is deprecated, use V3_MODEL_UTILITY instead. "
            f"Using override: {sim_override}"
        )
        _simulation_model_override = sim_override

    ext_override = os.environ.get("MENTIONS_EXTRACTION_MODEL")
    if ext_override:
        logger.warning(
            f"MENTIONS_EXTRACTION_MODEL is deprecated, use V3_MODEL_UTILITY instead. "
            f"Using override: {ext_override}"
        )
        _extraction_model_override = ext_override

    _configured = True
    logger.info(
        f"Model tiers configured: captain={_captain_model}, subagent={_subagent_model}, "
        f"utility={_utility_model}, embedding={_embedding_model}"
    )


def get_captain_model() -> str:
    """Return the configured captain-tier model identifier."""
    return _captain_model


def get_subagent_model() -> str:
    """Return the configured subagent-tier model identifier."""
    return _subagent_model


def get_utility_model() -> str:
    """Return the configured utility-tier model identifier."""
    return _utility_model


def get_embedding_model() -> str:
    """Return the configured embedding-tier model identifier."""
    return _embedding_model


# =============================================================================
# LEGACY API — simulation/extraction LLM factories
# =============================================================================

# Default models: use utility tier, allow legacy env var overrides
DEFAULT_SIMULATION_MODEL = os.getenv("MENTIONS_SIMULATION_MODEL", "gemini-2.0-flash")
DEFAULT_EXTRACTION_MODEL = os.getenv("MENTIONS_EXTRACTION_MODEL", "gemini-2.0-flash")

# Supported model identifiers
ModelType = Literal[
    # Gemini (cheapest for high-volume)
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    # Anthropic (good for structured extraction)
    "haiku",
    "sonnet",
    # OpenAI
    "gpt-4o-mini",
    "gpt-4o",
]


def get_simulation_llm(
    model: Optional[str] = None,
    temperature: float = 0.8,
    max_tokens: int = 4000,
):
    """Get LLM for transcript simulation (high-volume, creative).

    Args:
        model: Model identifier (default: utility tier or MENTIONS_SIMULATION_MODEL override)
        temperature: Creativity (0.8 recommended for natural variation)
        max_tokens: Max output tokens

    Returns:
        LangChain chat model instance
    """
    model = model or _simulation_model_override or _utility_model
    return _create_llm(model, temperature, max_tokens)


def get_extraction_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
):
    """Get LLM for structured extraction (precise, deterministic).

    Args:
        model: Model identifier (default: utility tier or MENTIONS_EXTRACTION_MODEL override)
        temperature: Creativity (0.0 recommended for extraction)
        max_tokens: Max output tokens

    Returns:
        LangChain chat model instance
    """
    model = model or _extraction_model_override or _utility_model
    return _create_llm(model, temperature, max_tokens)


def _create_llm(model: str, temperature: float, max_tokens: int):
    """Create LLM instance based on model identifier."""

    # Gemini models
    if model.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    # Anthropic models
    elif model in ("haiku", "sonnet") or model.startswith("claude"):
        from langchain_anthropic import ChatAnthropic

        model_id = {
            "haiku": "claude-haiku-4-5-20251001",
            "sonnet": "claude-sonnet-4-20250514",
        }.get(model, model)

        return ChatAnthropic(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # OpenAI models
    elif model.startswith("gpt"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    else:
        # Default to Gemini 2.0 Flash
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
            max_output_tokens=max_tokens,
        )


# =============================================================================
# COST REFERENCE (per 1M tokens, as of early 2025)
# =============================================================================
#
# | Model              | Input   | Output  | Best For                    |
# |--------------------|---------|---------|------------------------------|
# | gemini-2.0-flash   | $0.10   | $0.40   | High-volume simulation       |
# | gemini-1.5-flash   | $0.075  | $0.30   | Budget simulation            |
# | gpt-4o-mini        | $0.15   | $0.60   | Balanced cost/quality        |
# | claude-haiku-4.5   | $0.80   | $4.00   | Structured extraction        |
# | claude-sonnet-4    | $3.00   | $15.00  | Complex reasoning            |
# | gpt-4o             | $2.50   | $10.00  | Complex reasoning            |
#
# =============================================================================
