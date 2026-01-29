"""
NLP Pipeline Module for Entity Extraction and Sentiment Analysis.

This module provides a spaCy-based NLP pipeline backed by a Knowledge Base (KB)
for entity extraction, disambiguation, and sentiment analysis.

Key Components:
- KalshiKnowledgeBase: Wrapper around InMemoryLookupKB for market entities
- create_hybrid_entity_pipeline: Factory for unified NLP pipeline
- SentimentTask: Custom spacy-llm task for batched sentiment scoring
- RelatedEntityStore: Storage for non-market entities (second-hand signals)

Pipeline Flow:
    Reddit Post
        ↓
    [sentencizer] → [EntityRuler: MARKET_ENTITY] → [LLM NER: PERSON/ORG/GPE/EVENT]
        ↓
    [EntityLinker: link MARKET_ENTITY → KB]
        ↓
    [LLM Sentiment: score ALL entities in one batched call]
        ↓
    Output: doc.ents with kb_id_, sentiment, entity_type
"""

from .knowledge_base import (
    KalshiKnowledgeBase,
    EntityMetadata,
    get_kalshi_knowledge_base,
    set_kalshi_knowledge_base,
)
from .extensions import (
    register_custom_extensions,
    EntityExtensions,
)
from .pipeline import (
    create_hybrid_entity_pipeline,
    build_patterns_from_kb,
    get_shared_vocab,
)
from .sentiment_task import (
    BatchedSentimentTask,
)
from .entity_store import (
    RelatedEntityStore,
    RelatedEntity,
)
from .market_impact_reasoner import (
    MarketImpactReasoner,
    MarketInfo,
    should_analyze_for_market_impact,
    analyze_market_impact,
)
from .llm_entity_extractor import (
    LLMEntityExtractor,
)

__all__ = [
    # Knowledge Base
    "KalshiKnowledgeBase",
    "EntityMetadata",
    "get_kalshi_knowledge_base",
    "set_kalshi_knowledge_base",
    # Extensions
    "register_custom_extensions",
    "EntityExtensions",
    # Pipeline
    "create_hybrid_entity_pipeline",
    "build_patterns_from_kb",
    "get_shared_vocab",
    # Sentiment
    "BatchedSentimentTask",
    # Related Entities
    "RelatedEntityStore",
    "RelatedEntity",
    # Market Impact Reasoning
    "MarketImpactReasoner",
    "MarketInfo",
    "should_analyze_for_market_impact",
    "analyze_market_impact",
    # LLM Entity Extraction
    "LLMEntityExtractor",
]
