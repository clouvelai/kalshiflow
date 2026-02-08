"""
Trade Flow and TMO event dataclasses for TRADER V3.

Contains events related to market microstructure tracking and trade flow analysis.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import EventType


@dataclass
class TradeFlowMarketUpdateEvent:
    """
    Event data for trade flow market state updates.

    Emitted by MarketStateAgent when a tracked market's trade state changes.
    Used for real-time UI updates showing trade direction and price movement.

    Attributes:
        event_type: Always TRADE_FLOW_MARKET_UPDATE
        market_ticker: Market ticker for this update
        state: Dictionary containing trade flow state (yes_trades, no_trades, etc.)
        timestamp: When the update was generated
    """
    event_type: EventType = EventType.TRADE_FLOW_MARKET_UPDATE
    market_ticker: str = ""
    state: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.state is None:
            self.state = {}


@dataclass
class TradeFlowTradeArrivedEvent:
    """
    Event data for individual trades arriving for tracking.

    Emitted by MarketStateAgent for every trade in a tracked market.
    Used for real-time UI pulse/glow animations on trade arrival.

    Attributes:
        event_type: Always TRADE_FLOW_TRADE_ARRIVED
        market_ticker: Market ticker where trade occurred
        event_ticker: Event ticker this market belongs to
        side: Trade side ("yes" or "no")
        count: Number of contracts in this trade
        price_cents: Trade price in cents
        timestamp: When the trade was received
    """
    event_type: EventType = EventType.TRADE_FLOW_TRADE_ARRIVED
    market_ticker: str = ""
    event_ticker: str = ""
    side: str = ""
    count: int = 0
    price_cents: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class TMOFetchedEvent:
    """
    Event data for True Market Open (TMO) price fetches.

    Emitted by TrueMarketOpenFetcher when the true market open price
    is successfully retrieved from the Kalshi candlestick API.

    The TMO is the first candlestick's yes_bid.open price, representing
    the actual market opening price (not just the first trade we observe).

    Attributes:
        event_type: Always TMO_FETCHED
        market_ticker: Market ticker for this TMO
        true_market_open: YES price in cents at market open
        open_ts: Unix timestamp when the market opened
        timestamp: When this event was created
    """
    event_type: EventType = EventType.TMO_FETCHED
    market_ticker: str = ""
    true_market_open: int = 0  # YES price in cents at market open
    open_ts: int = 0           # Unix timestamp when market opened
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
