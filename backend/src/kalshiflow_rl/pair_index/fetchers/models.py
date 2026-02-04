"""
Standalone data models for the pair index system.

Zero traderv3 imports. These models represent normalized venue data
and match candidates flowing through the pipeline.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, FrozenSet, List, Optional


@dataclass
class NormalizedMarket:
    """A normalized market from either venue."""
    venue: str                                  # "kalshi" | "polymarket"
    event_id: str                               # kalshi event_ticker or poly event slug
    market_id: str                              # kalshi ticker or poly condition_id
    question: str                               # market title / question
    close_time: Optional[datetime] = None
    is_active: bool = True
    # Venue-specific IDs
    kalshi_ticker: Optional[str] = None
    kalshi_event_ticker: Optional[str] = None
    poly_condition_id: Optional[str] = None
    poly_token_id_yes: Optional[str] = None
    poly_token_id_no: Optional[str] = None
    # Text features (populated by matcher)
    normalized_text: str = ""
    token_set: FrozenSet[str] = frozenset()
    entities: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)


@dataclass
class NormalizedEvent:
    """A normalized event from either venue."""
    venue: str                                  # "kalshi" | "polymarket"
    event_id: str                               # kalshi event_ticker or poly slug
    title: str
    category: str
    markets: List[NormalizedMarket] = field(default_factory=list)
    close_time: Optional[datetime] = None
    mutually_exclusive: bool = False
    market_count: int = 0
    volume_24h: int = 0
    # Text features (populated by matcher)
    normalized_title: str = ""
    token_set: FrozenSet[str] = frozenset()
    entities: List[str] = field(default_factory=list)
    # Raw API data (not serialized)
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class MatchCandidate:
    """A matched market pair ready for persistence."""
    kalshi_market: NormalizedMarket
    poly_market: NormalizedMarket
    kalshi_event: NormalizedEvent
    poly_event: NormalizedEvent
    event_text_score: float
    market_text_score: float
    combined_score: float
    match_method: str                           # "text", "entity", "llm"
    match_tier: int                             # 1 or 2
    match_signals: List[str] = field(default_factory=list)
    llm_reasoning: Optional[str] = None

    @property
    def kalshi_ticker(self) -> str:
        return self.kalshi_market.kalshi_ticker or self.kalshi_market.market_id

    @property
    def kalshi_event_ticker(self) -> str:
        return self.kalshi_market.kalshi_event_ticker or self.kalshi_event.event_id

    @property
    def poly_condition_id(self) -> str:
        return self.poly_market.poly_condition_id or self.poly_market.market_id

    @property
    def poly_token_id_yes(self) -> str:
        return self.poly_market.poly_token_id_yes or ""

    @property
    def poly_token_id_no(self) -> Optional[str]:
        return self.poly_market.poly_token_id_no

    @property
    def question(self) -> str:
        return self.kalshi_market.question or self.poly_market.question
