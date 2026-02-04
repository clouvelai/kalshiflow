"""Arbitrage event dataclasses for cross-venue spread monitoring."""

from dataclasses import dataclass, field
from typing import Optional
import time

from .types import EventType


@dataclass
class PolyPriceEvent:
    """Polymarket price update for a paired market."""
    pair_id: str
    kalshi_ticker: str
    poly_token_id: str
    poly_yes_cents: int  # 0-100
    poly_no_cents: int   # 0-100
    source: str = "ws"  # "ws" or "api"
    latency_ms: Optional[float] = None  # processing/RTT latency
    event_type: EventType = EventType.POLY_PRICE_UPDATE
    timestamp: float = field(default_factory=time.time)


@dataclass
class SpreadUpdateEvent:
    """Spread recalculation for a paired market."""
    pair_id: str
    kalshi_ticker: str
    kalshi_yes_bid: Optional[int] = None  # cents
    kalshi_yes_ask: Optional[int] = None  # cents
    kalshi_yes_mid: Optional[int] = None  # cents
    poly_yes_cents: Optional[int] = None  # cents
    poly_no_cents: Optional[int] = None   # cents
    spread_cents: Optional[int] = None    # kalshi_mid - poly_yes
    question: str = ""
    event_type: EventType = EventType.SPREAD_UPDATE
    timestamp: float = field(default_factory=time.time)


@dataclass
class SpreadTradeExecutedEvent:
    """Arb trade executed on the hot path."""
    pair_id: str
    kalshi_ticker: str
    side: str          # "yes" or "no"
    action: str        # "buy" or "sell"
    contracts: int
    price_cents: int
    spread_at_entry: int
    kalshi_mid: int
    poly_mid: int
    kalshi_order_id: Optional[str] = None
    reasoning: str = ""
    event_type: EventType = EventType.SPREAD_TRADE_EXECUTED
    timestamp: float = field(default_factory=time.time)


@dataclass
class KalshiApiPriceEvent:
    """Kalshi REST API price update (orderbook BBO via polling)."""
    pair_id: str
    kalshi_ticker: str
    yes_bid: Optional[int] = None  # cents
    yes_ask: Optional[int] = None  # cents
    yes_mid: Optional[int] = None  # cents
    latency_ms: Optional[float] = None
    event_type: EventType = EventType.KALSHI_API_PRICE_UPDATE
    timestamp: float = field(default_factory=time.time)


@dataclass
class PairMatchedEvent:
    """New cross-venue pair discovered and matched."""
    pair_id: str
    kalshi_ticker: str
    poly_condition_id: str
    poly_token_id_yes: str
    question: str
    match_method: str = "manual"
    match_confidence: float = 1.0
    event_type: EventType = EventType.PAIR_MATCHED
    timestamp: float = field(default_factory=time.time)
