"""
NLP Extraction Module for Kalshi Market Signal Extraction.

This module provides a langextract-based extraction pipeline for processing
text content (Reddit posts, news articles, etc.) into structured market signals.

Key Components:
- KalshiExtractor: Wrapper around langextract with merged event-specific specs
- EventConfig: Per-event extraction configuration from Supabase
- ExtractionRow: Individual extraction result ready for Supabase insert
- ExampleManager: Manages global + event-specific extraction examples

Pipeline Flow:
    Reddit Post / News Article
        ↓
    KalshiExtractor.extract(text, event_configs)
        ↓
    langextract (Gemini) → ExtractionResult
        ↓
    Parse → ExtractionRow[] → Supabase extractions table
        ↓
    Supabase Realtime → PriceImpactAgent → Frontend WebSocket
"""

from .kalshi_extractor import (
    KalshiExtractor,
    EventConfig,
    ExtractionRow,
)

__all__ = [
    "KalshiExtractor",
    "EventConfig",
    "ExtractionRow",
]
