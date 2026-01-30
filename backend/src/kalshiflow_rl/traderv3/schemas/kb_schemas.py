"""
Knowledge Base Schema Models.

Defines data structures for the entity accumulation and knowledge base system:
1. ObjectiveEntity - Kalshi event entity with keyword matching rules
2. EntityMention - Single mention of an entity from any source
3. AccumulatedEntitySignal - Aggregated signal over sliding window
4. EntitySignalSummary - Deep agent-facing summary
5. EntityRelation - Extracted relation between two entities
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class EntityCategory(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    OBJECTIVE = "objective"  # Kalshi-derived event entities


class RelationLabel(str, Enum):
    """Domain-specific relation labels for political/event prediction markets."""
    SUPPORTS = "SUPPORTS"      # Subject endorses, defends, or promotes object
    OPPOSES = "OPPOSES"        # Subject criticizes, blocks, or works against object
    CAUSES = "CAUSES"          # Subject's action/event leads to object's outcome
    AFFECTED_BY = "AFFECTED_BY"  # Subject is impacted by object's actions
    MEMBER_OF = "MEMBER_OF"    # Subject belongs to or is part of object


@dataclass
class EntityRelation:
    """Extracted relation between two named entities.

    Represents a directional relationship discovered via LLM relation
    extraction (REL). Used to build causal chains and enrich entity
    signals for the deep agent.

    Example:
        subject="Donald Trump", relation="SUPPORTS", object="Pam Bondi"
        subject="ICE Enforcement", relation="CAUSES", object="Government Shutdown"
    """

    subject_entity_id: str
    subject_name: str
    relation: str  # One of RelationLabel values
    object_entity_id: str
    object_name: str
    confidence: float = 0.7  # 0.0 to 1.0
    source_post_id: str = ""
    context_snippet: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ObjectiveEntity:
    """Kalshi event entity with keyword matching rules.

    Represents an outcome-type event (not tied to a specific person) that
    can be matched via keyword and category overlap. Generated during
    EntityMarketIndex refresh for events classified as outcome-type.

    Example:
        entity_id="government_shutdown"
        canonical_name="Government Shutdown"
        event_ticker="KXGOVSHUT"
        keywords=["shutdown", "ice", "immigration", "funding bill", "cr"]
        related_entities=["donald trump", "mike johnson", "congress"]
        categories=["government", "immigration", "budget"]
    """

    entity_id: str
    canonical_name: str
    event_ticker: str
    market_tickers: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    related_entities: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectiveEntity":
        return cls(
            entity_id=data.get("entity_id", ""),
            canonical_name=data.get("canonical_name", ""),
            event_ticker=data.get("event_ticker", ""),
            market_tickers=data.get("market_tickers", []),
            keywords=data.get("keywords", []),
            related_entities=data.get("related_entities", []),
            categories=data.get("categories", []),
            updated_at=data.get("updated_at", time.time()),
        )


@dataclass
class ObjectiveEntityMatch:
    """Result of matching text against an objective entity's keywords."""

    objective_entity: ObjectiveEntity
    hit_score: int  # Combined score from keyword + entity + category matches
    matched_keywords: List[str] = field(default_factory=list)
    matched_entities: List[str] = field(default_factory=list)
    matched_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.objective_entity.entity_id,
            "canonical_name": self.objective_entity.canonical_name,
            "event_ticker": self.objective_entity.event_ticker,
            "hit_score": self.hit_score,
            "matched_keywords": self.matched_keywords,
            "matched_entities": self.matched_entities,
            "matched_categories": self.matched_categories,
        }


@dataclass
class EntityMention:
    """Single mention of an entity from any source.

    Created each time an entity is extracted from a document (Reddit post,
    comment batch, news article). Fed into the EntityAccumulator to build
    up signal strength over time.
    """

    entity_id: str
    source_type: str  # "reddit_post", "reddit_comments", "news"
    source_post_id: str = ""
    sentiment_score: int = 0  # -100 to +100
    confidence: float = 0.5
    categories: List[str] = field(default_factory=list)
    reddit_score: int = 0
    reddit_comments: int = 0
    context_snippet: str = ""
    source_domain: str = "reddit.com"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccumulatedEntitySignal:
    """Aggregated signal for an entity over a sliding window.

    Maintained by the EntityAccumulator. When signal_strength crosses
    the configured threshold, a PriceImpactSignal is emitted.
    """

    entity_id: str
    canonical_name: str
    entity_category: str  # person | organization | objective
    mention_count: int = 0
    unique_sources: int = 0
    weighted_sentiment: float = 0.0  # Reddit-score-weighted average
    signal_strength: float = 0.0  # 0.0-1.0 composite score
    max_reddit_score: int = 0
    total_reddit_comments: int = 0
    source_types: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    linked_market_tickers: List[str] = field(default_factory=list)
    latest_context: str = ""
    first_mention_at: float = 0.0
    last_mention_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EntitySignalSummary:
    """Deep agent-facing summary of an accumulated entity signal.

    This is what the deep agent sees when calling get_entity_signals().
    Provides a concise view of building narratives for trading decisions.
    """

    entity_id: str
    canonical_name: str
    entity_category: str
    mention_count: int
    unique_sources: int
    weighted_sentiment: float
    signal_strength: float
    max_reddit_score: int
    total_reddit_comments: int
    linked_market_tickers: List[str]
    categories: List[str]
    latest_context: str
    window_hours: float
    relations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
