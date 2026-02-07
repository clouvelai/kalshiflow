"""
Centralized LLM model configuration for mentions system.

Change these defaults to swap providers/models across all mentions tools.
Supports: Anthropic (Haiku/Sonnet), Google (Gemini), OpenAI (GPT-4o-mini)
"""

import os
from typing import Literal, Optional

# =============================================================================
# MODEL CONFIGURATION - Change these to swap providers
# =============================================================================

# Default models (override via env vars)
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
        model: Model identifier (default: DEFAULT_SIMULATION_MODEL)
        temperature: Creativity (0.8 recommended for natural variation)
        max_tokens: Max output tokens

    Returns:
        LangChain chat model instance
    """
    model = model or DEFAULT_SIMULATION_MODEL
    return _create_llm(model, temperature, max_tokens)


def get_extraction_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 500,
):
    """Get LLM for structured extraction (precise, deterministic).

    Args:
        model: Model identifier (default: DEFAULT_EXTRACTION_MODEL)
        temperature: Creativity (0.0 recommended for extraction)
        max_tokens: Max output tokens

    Returns:
        LangChain chat model instance
    """
    model = model or DEFAULT_EXTRACTION_MODEL
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
