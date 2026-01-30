"""
Entity-Based Trading System Data Models.

Defines the core data structures for:
1. Reddit Entity Signal - Raw sentiment per entity from Reddit posts
2. Entity-to-Market Mapping - How entities map to Kalshi markets
3. Price Impact Signal - Transformed sentiment for specific markets

Key Insight: Sentiment ≠ Price Impact
- Negative sentiment about "Pam Bondi" = +impact on "Bondi OUT" market
- The transformation depends on market type (OUT, WIN, CONFIRM, etc.)
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================================
# Price Impact Transformation Rules
# ============================================================================

IMPACT_RULES: Dict[str, Dict[str, str]] = {
    "OUT": {
        "transform": "invert",
        "description": "OUT markets benefit from negative sentiment (scandal makes OUT more likely)",
    },
    "WIN": {
        "transform": "preserve",
        "description": "WIN markets follow sentiment direction",
    },
    "CONFIRM": {
        "transform": "preserve",
        "description": "CONFIRM markets follow sentiment direction",
    },
    "NOMINEE": {
        "transform": "preserve",
        "description": "NOMINEE markets follow sentiment direction",
    },
    "PRESIDENT": {
        "transform": "preserve",
        "description": "President markets follow sentiment direction",
    },
    "DEFAULT": {
        "transform": "preserve",
        "description": "Default: preserve sentiment direction",
    },
}


def compute_price_impact(sentiment: int, market_type: str) -> int:
    """
    Transform entity sentiment into market-specific price impact.

    Args:
        sentiment: Entity sentiment score (-100 to +100)
        market_type: Market type (OUT, WIN, CONFIRM, NOMINEE, etc.)

    Returns:
        Price impact score (-100 to +100)

    Examples:
        - Bondi scandal (-98 sentiment) + "BONDI-OUT" market → +98 impact
        - Hegseth praise (+70 sentiment) + "HEGSETH-CONFIRM" market → +70 impact
    """
    rule = IMPACT_RULES.get(market_type.upper(), IMPACT_RULES["DEFAULT"])
    if rule["transform"] == "invert":
        return -sentiment
    return sentiment


def normalize_entity_id(name: str) -> str:
    """
    Normalize an entity name to a consistent ID format.

    Args:
        name: Entity name (e.g., "Pam Bondi", "Pete Hegseth")

    Returns:
        Normalized ID (e.g., "pam_bondi", "pete_hegseth")
    """
    # Lowercase, replace spaces with underscores, remove special chars
    normalized = name.lower().strip()
    normalized = re.sub(r"[^\w\s]", "", normalized)  # Remove punctuation
    normalized = re.sub(r"\s+", "_", normalized)  # Spaces to underscores
    return normalized


# ============================================================================
# Reddit Entity Signal Models
# ============================================================================


@dataclass(frozen=True)
class LLMExtractedEntity:
    """
    An entity extracted by the LLM entity extractor.

    Combines entity extraction and sentiment scoring in a single LLM call.
    This replaces the separate spaCy NER + batched sentiment approach.
    """

    name: str  # Canonical name: "Alex Pretti"
    entity_type: str  # PERSON, ORG, GPE, EVENT, POLICY, NORP
    sentiment: int  # -100 to +100
    confidence: str  # "low", "medium", "high"
    market_tickers: Tuple[str, ...] = ()  # LLM-matched market tickers (from extract_with_markets)
    context: str = ""  # Per-entity summary from LLM

    @property
    def name_normalized(self) -> str:
        """Return normalized entity ID."""
        return normalize_entity_id(self.name)

    @property
    def confidence_float(self) -> float:
        """Convert confidence label to float."""
        confidence_map = {"low": 0.5, "medium": 0.7, "high": 0.9}
        return confidence_map.get(self.confidence.lower(), 0.5)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "market_tickers": list(self.market_tickers),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMExtractedEntity":
        return cls(
            name=data.get("name", ""),
            entity_type=data.get("type", data.get("entity_type", "UNKNOWN")),
            sentiment=int(data.get("sentiment", 0)),
            confidence=data.get("confidence", "low"),
            market_tickers=tuple(data.get("market_tickers", [])),
            context=data.get("context", ""),
        )


@dataclass(frozen=True)
class ExtractedEntity:
    """
    A single entity extracted from text with sentiment analysis.

    This is immutable (frozen) to ensure consistency in the pipeline.

    Sentiment uses a 5-point scale (-2 to +2) which is then mapped to
    impact scores for trading decisions:
        -2 (strongly negative) -> -75
        -1 (mildly negative)   -> -40
         0 (neutral)           ->   0
        +1 (mildly positive)   -> +40
        +2 (strongly positive) -> +75
    """

    entity_id: str  # Normalized: "pam_bondi"
    canonical_name: str  # Display: "Pam Bondi"
    entity_type: str  # "person", "org", "position"
    sentiment_score: int  # Mapped value: -75, -40, 0, 40, 75
    confidence: float  # From LLM: 0.5, 0.7, 0.9
    context_snippet: str = ""  # Relevant text snippet
    sentiment_category: Optional[int] = None  # Original 5-point scale: -2 to +2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedEntity":
        return cls(
            entity_id=data.get("entity_id", ""),
            canonical_name=data.get("canonical_name", ""),
            entity_type=data.get("entity_type", "unknown"),
            sentiment_score=int(data.get("sentiment_score", 0)),
            confidence=float(data.get("confidence", 0.5)),
            context_snippet=data.get("context_snippet", ""),
            sentiment_category=data.get("sentiment_category"),
        )


@dataclass
class RedditEntitySignal:
    """
    A Reddit post with all extracted entities and their sentiments.

    This is the primary output of the Reddit Entity Agent.
    """

    signal_id: str
    post_id: str
    subreddit: str
    title: str
    url: str
    author: str
    score: int
    num_comments: int
    post_created_utc: float
    entities: List[ExtractedEntity]
    aggregate_sentiment: int  # Average sentiment across entities
    created_at: float
    expires_at: float  # TTL for cache expiration

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "post_id": self.post_id,
            "subreddit": self.subreddit,
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "score": self.score,
            "num_comments": self.num_comments,
            "post_created_utc": self.post_created_utc,
            "entities": [e.to_dict() for e in self.entities],
            "aggregate_sentiment": self.aggregate_sentiment,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedditEntitySignal":
        entities = [
            ExtractedEntity.from_dict(e) for e in data.get("entities", [])
        ]
        return cls(
            signal_id=data.get("signal_id", str(uuid.uuid4())),
            post_id=data.get("post_id", ""),
            subreddit=data.get("subreddit", ""),
            title=data.get("title", ""),
            url=data.get("url", ""),
            author=data.get("author", ""),
            score=int(data.get("score", 0)),
            num_comments=int(data.get("num_comments", 0)),
            post_created_utc=float(data.get("post_created_utc", 0)),
            entities=entities,
            aggregate_sentiment=int(data.get("aggregate_sentiment", 0)),
            created_at=float(data.get("created_at", time.time())),
            expires_at=float(data.get("expires_at", time.time() + 3600)),
        )


# ============================================================================
# Entity-to-Market Mapping Models
# ============================================================================


@dataclass
class MarketMapping:
    """
    Maps an entity to a specific Kalshi market.

    Used by the Entity-Market Index to track which markets
    are affected by which entities.
    """

    market_ticker: str  # "KXBONDIOUT-25FEB01"
    event_ticker: str  # "KXBONDIOUT"
    market_type: str  # "OUT", "WIN", "CONFIRM", "NOMINEE"
    yes_sub_title: str  # From Kalshi API: "Pam Bondi"
    confidence: float  # Mapping confidence (0.0 to 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalEntity:
    """
    Single source of truth for an entity.

    This represents a canonical entity record with all aliases mapping to it.
    The entity index uses these to match Reddit mentions to market entities.
    """

    entity_id: str  # "donald_trump"
    canonical_name: str  # "Donald Trump"
    entity_type: str  # "person" | "organization" | "position" | "outcome"
    aliases: set  # {"trump", "donald", "donald j trump", "djt", "potus"}
    markets: List[MarketMapping]  # All linked markets
    created_at: float
    last_seen_at: float
    llm_aliases: Set[str] = field(default_factory=set)  # LLM-generated aliases (e.g., "government shutdown")

    # Aggregated signals (populated at runtime)
    reddit_mentions: int = 0
    aggregate_sentiment: float = 0.0
    last_reddit_signal: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": list(self.aliases),
            "llm_aliases": list(self.llm_aliases),
            "markets": [m.to_dict() for m in self.markets],
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "reddit_mentions": self.reddit_mentions,
            "aggregate_sentiment": self.aggregate_sentiment,
            "last_reddit_signal": self.last_reddit_signal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CanonicalEntity":
        return cls(
            entity_id=data.get("entity_id", ""),
            canonical_name=data.get("canonical_name", ""),
            entity_type=data.get("entity_type", "person"),
            aliases=set(data.get("aliases", [])),
            markets=[MarketMapping(**m) for m in data.get("markets", [])],
            created_at=data.get("created_at", time.time()),
            last_seen_at=data.get("last_seen_at", time.time()),
            llm_aliases=set(data.get("llm_aliases", [])),
            reddit_mentions=data.get("reddit_mentions", 0),
            aggregate_sentiment=data.get("aggregate_sentiment", 0.0),
            last_reddit_signal=data.get("last_reddit_signal"),
        )

    @property
    def market_count(self) -> int:
        return len(self.markets)

    @property
    def alias_count(self) -> int:
        return len(self.aliases)


@dataclass
class EntityMarketEntry:
    """
    An entry in the Entity-Market Index.

    Maps a single entity to all relevant markets.
    """

    entity_id: str  # Normalized: "pam_bondi"
    canonical_name: str  # Display: "Pam Bondi"
    markets: List[MarketMapping]  # All markets for this entity
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "markets": [m.to_dict() for m in self.markets],
            "last_updated": self.last_updated,
        }

    @property
    def market_count(self) -> int:
        return len(self.markets)


# ============================================================================
# Price Impact Signal Model
# ============================================================================


@dataclass(frozen=True)
class PriceImpactSignal:
    """
    A trading signal with transformed price impact for a specific market.

    This is the final output consumed by the Deep Agent for trading decisions.
    The transformation from sentiment_score to price_impact_score accounts
    for market type (e.g., OUT markets invert sentiment).
    """

    signal_id: str
    market_ticker: str
    entity_id: str
    entity_name: str

    # Scores
    sentiment_score: int  # Original: -98 (bad for entity)
    price_impact_score: int  # Transformed: +98 (for OUT market)
    confidence: float  # 0.0 to 1.0

    # Market context
    market_type: str  # "OUT", "WIN", "CONFIRM", etc.
    event_ticker: str
    transformation_logic: str  # "OUT market: inverted sentiment"

    # Source tracking
    source_post_id: str
    source_subreddit: str
    reddit_entity_id: Optional[str] = None

    # Timestamps
    created_at: float = field(default_factory=time.time)
    source_created_at: Optional[float] = None  # Original Reddit post creation time

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriceImpactSignal":
        source_created_at = data.get("source_created_at")
        if source_created_at is not None:
            source_created_at = float(source_created_at)
        return cls(
            signal_id=data.get("signal_id", str(uuid.uuid4())),
            market_ticker=data.get("market_ticker", ""),
            entity_id=data.get("entity_id", ""),
            entity_name=data.get("entity_name", ""),
            sentiment_score=int(data.get("sentiment_score", 0)),
            price_impact_score=int(data.get("price_impact_score", 0)),
            confidence=float(data.get("confidence", 0.5)),
            market_type=data.get("market_type", ""),
            event_ticker=data.get("event_ticker", ""),
            transformation_logic=data.get("transformation_logic", ""),
            source_post_id=data.get("source_post_id", ""),
            source_subreddit=data.get("source_subreddit", ""),
            reddit_entity_id=data.get("reddit_entity_id"),
            created_at=float(data.get("created_at", time.time())),
            source_created_at=source_created_at,
        )

    @classmethod
    def from_entity_and_mapping(
        cls,
        entity: ExtractedEntity,
        mapping: MarketMapping,
        signal: RedditEntitySignal,
        reddit_entity_db_id: Optional[str] = None,
    ) -> "PriceImpactSignal":
        """
        Create a PriceImpactSignal from entity extraction and market mapping.

        This is the main factory method for creating price impact signals.
        """
        price_impact = compute_price_impact(entity.sentiment_score, mapping.market_type)
        rule = IMPACT_RULES.get(mapping.market_type.upper(), IMPACT_RULES["DEFAULT"])

        return cls(
            signal_id=str(uuid.uuid4()),
            market_ticker=mapping.market_ticker,
            entity_id=entity.entity_id,
            entity_name=entity.canonical_name,
            sentiment_score=entity.sentiment_score,
            price_impact_score=price_impact,
            confidence=min(entity.confidence, mapping.confidence),
            market_type=mapping.market_type,
            event_ticker=mapping.event_ticker,
            transformation_logic=f"{mapping.market_type} market: {rule['description']}",
            source_post_id=signal.post_id,
            source_subreddit=signal.subreddit,
            reddit_entity_id=reddit_entity_db_id,
            created_at=time.time(),
            source_created_at=signal.post_created_utc,  # Original Reddit timestamp
        )

    @property
    def is_strong_signal(self) -> bool:
        """Check if this is a strong signal (|impact| > 50, confidence > 0.7)."""
        return abs(self.price_impact_score) > 50 and self.confidence > 0.7

    @property
    def suggested_side(self) -> str:
        """Suggest trade side based on price impact."""
        return "yes" if self.price_impact_score > 0 else "no"


# ============================================================================
# Market Impact Reasoning Model (Indirect Effects)
# ============================================================================


@dataclass
class MarketImpactResult:
    """
    Result of LLM reasoning about which markets are affected by news content.

    This captures INDIRECT market impacts that entity-sentiment mapping misses.
    For example: "ICE shooting in Minnesota" → government shutdown market.

    The reasoner analyzes news content and determines which markets are affected,
    even when there's no direct entity → market title match.
    """

    market_ticker: str  # "KXGOVSHUT-26JAN31"
    market_title: str  # "Government shutdown on Saturday?"
    impact_direction: str  # "bullish" or "bearish" (on YES price)
    impact_magnitude: int  # -2 to +2 (same as sentiment scale)
    confidence: str  # "low", "medium", "high"
    reasoning: str  # LLM's causal chain explanation

    # Source tracking
    source_post_id: str = ""
    source_content_summary: str = ""  # Brief summary of triggering content

    # Timestamp
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_ticker": self.market_ticker,
            "market_title": self.market_title,
            "impact_direction": self.impact_direction,
            "impact_magnitude": self.impact_magnitude,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "source_post_id": self.source_post_id,
            "source_content_summary": self.source_content_summary,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketImpactResult":
        return cls(
            market_ticker=data.get("market_ticker", ""),
            market_title=data.get("market_title", ""),
            impact_direction=data.get("impact_direction", "neutral"),
            impact_magnitude=int(data.get("impact_magnitude", 0)),
            confidence=data.get("confidence", "low"),
            reasoning=data.get("reasoning", ""),
            source_post_id=data.get("source_post_id", ""),
            source_content_summary=data.get("source_content_summary", ""),
            created_at=float(data.get("created_at", time.time())),
        )

    @property
    def price_impact_score(self) -> int:
        """Convert magnitude (-2 to +2) to price impact score (-75 to +75)."""
        magnitude_map = {-2: -75, -1: -40, 0: 0, 1: 40, 2: 75}
        base_score = magnitude_map.get(self.impact_magnitude, 0)
        # Bearish = negative impact, bullish = positive
        if self.impact_direction == "bearish":
            return -abs(base_score)
        return abs(base_score)

    @property
    def confidence_float(self) -> float:
        """Convert confidence label to float (0.5, 0.7, 0.9)."""
        confidence_map = {"low": 0.5, "medium": 0.7, "high": 0.9}
        return confidence_map.get(self.confidence.lower(), 0.5)

    @property
    def suggested_side(self) -> str:
        """Suggest trade side based on impact direction."""
        return "yes" if self.impact_direction == "bullish" else "no"


# ============================================================================
# Related Entity Model (Second-Hand Signals)
# ============================================================================


@dataclass
class RelatedEntitySignal:
    """
    A non-market entity discovered via general NER.

    These entities are not directly linked to Kalshi markets but provide
    contextual signals for second-hand analysis. For example:
    - "Putin meets Xi" → geopolitical context for Taiwan markets
    - "Fed Chair Powell speaks" → economic context for various markets

    Used by DeepAgent for contextual queries.
    """

    entity_text: str  # Original text: "Putin"
    entity_type: str  # PERSON, ORG, GPE, EVENT
    normalized_id: str  # Normalized: "vladimir_putin"
    sentiment_score: int  # -100 to +100
    confidence: float = 1.0

    # Source context
    source_post_id: str = ""
    source_subreddit: str = ""
    context_snippet: str = ""

    # Co-occurrence with market entities
    co_occurring_market_entities: List[str] = field(default_factory=list)

    # Timestamp
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type,
            "normalized_id": self.normalized_id,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "source_post_id": self.source_post_id,
            "source_subreddit": self.source_subreddit,
            "context_snippet": self.context_snippet,
            "co_occurring_market_entities": self.co_occurring_market_entities,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelatedEntitySignal":
        return cls(
            entity_text=data.get("entity_text", ""),
            entity_type=data.get("entity_type", ""),
            normalized_id=data.get("normalized_id", ""),
            sentiment_score=int(data.get("sentiment_score", 0)),
            confidence=float(data.get("confidence", 1.0)),
            source_post_id=data.get("source_post_id", ""),
            source_subreddit=data.get("source_subreddit", ""),
            context_snippet=data.get("context_snippet", ""),
            co_occurring_market_entities=data.get("co_occurring_market_entities", []),
            created_at=float(data.get("created_at", time.time())),
        )
