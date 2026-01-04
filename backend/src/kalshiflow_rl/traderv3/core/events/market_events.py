"""
Market-related event dataclasses for TRADER V3.

Contains events related to market data, positions, tickers, and order fills.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import EventType


@dataclass
class MarketEvent:
    """
    Event data for market updates.

    Represents orderbook snapshots and deltas from Kalshi WebSocket.
    These events drive the core trading logic and market monitoring.

    Attributes:
        event_type: Type of market event (SNAPSHOT or DELTA)
        market_ticker: Kalshi market identifier
        sequence_number: Orderbook sequence for consistency
        timestamp_ms: Kalshi timestamp in milliseconds
        received_at: Local timestamp when event was received
        metadata: Additional data (orderbook levels, etc.)
    """
    event_type: EventType
    market_ticker: str
    sequence_number: int
    timestamp_ms: int
    received_at: float

    # Optional additional data
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MarketPositionEvent:
    """
    Event data for real-time position updates from Kalshi WebSocket.

    Emitted when the market_positions WebSocket channel sends an update.
    All monetary values are in cents (converted from Kalshi centi-cents).

    Attributes:
        event_type: Always MARKET_POSITION_UPDATE
        market_ticker: Market ticker for this position
        position_data: Position details (position, market_exposure, realized_pnl, etc.)
        timestamp: When the update was received
    """
    event_type: EventType = EventType.MARKET_POSITION_UPDATE
    market_ticker: str = ""
    position_data: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0


@dataclass
class MarketTickerEvent:
    """
    Event data for real-time market price updates from Kalshi ticker WebSocket.

    Emitted when the ticker WebSocket channel sends a price update.
    All price values are in cents (1-99 for prediction markets).

    Attributes:
        event_type: Always MARKET_TICKER_UPDATE
        market_ticker: Market ticker for this price update
        price_data: Price details (last_price, yes_bid, yes_ask, etc.)
        timestamp: When the update was received
    """
    event_type: EventType = EventType.MARKET_TICKER_UPDATE
    market_ticker: str = ""
    price_data: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.price_data is None:
            self.price_data = {}


@dataclass
class OrderFillEvent:
    """
    Event data for real-time order fill notifications from Kalshi fill WebSocket.

    Emitted when the fill WebSocket channel sends a notification that one of
    our orders has been filled (partially or fully).

    All price values are in cents (1-99 for prediction markets).

    Attributes:
        event_type: Always ORDER_FILL
        trade_id: Unique identifier for this fill
        order_id: Associated order UUID
        market_ticker: Market ticker where fill occurred
        is_taker: Whether we were the taker (True) or maker (False)
        side: "yes" or "no"
        action: "buy" or "sell"
        price_cents: Fill price in cents (1-99)
        count: Number of contracts filled
        post_position: Our position after the fill
        fill_timestamp: Unix timestamp when fill occurred (seconds)
        timestamp: When the event was received locally
    """
    event_type: EventType = EventType.ORDER_FILL
    trade_id: str = ""
    order_id: str = ""
    market_ticker: str = ""
    is_taker: bool = False
    side: str = ""
    action: str = ""
    price_cents: int = 0
    count: int = 0
    post_position: int = 0
    fill_timestamp: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
