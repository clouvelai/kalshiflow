"""
Market lifecycle event dataclasses for TRADER V3.

Contains events related to market creation, tracking, and determination.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .types import EventType


@dataclass
class MarketLifecycleEvent:
    """
    Event data for market lifecycle events from Kalshi WebSocket.

    Emitted when LifecycleIntegration receives an event from the
    market_lifecycle_v2 channel. EventLifecycleService subscribes to
    these events for market discovery.

    Lifecycle Event Types:
        - created: Market initialized (triggers REST lookup + category filter)
        - activated: Market becomes tradeable
        - deactivated: Trading paused
        - close_date_updated: Settlement time modified
        - determined: Outcome resolved (triggers orderbook unsubscription)
        - settled: Positions liquidated

    Attributes:
        event_type: Always MARKET_LIFECYCLE_EVENT
        lifecycle_event_type: Type of lifecycle event (created, determined, etc.)
        market_ticker: Market ticker for this event
        payload: Full event data including timestamps and metadata
        timestamp: When the event was received locally
    """
    event_type: EventType = EventType.MARKET_LIFECYCLE_EVENT
    lifecycle_event_type: str = ""  # "created", "determined", "settled", etc.
    market_ticker: str = ""
    payload: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.payload is None:
            self.payload = {}


@dataclass
class MarketTrackedEvent:
    """
    Event data for when a market is added to tracking.

    Emitted by EventLifecycleService after a market passes category filtering
    and is successfully added to TrackedMarketsState. Downstream services
    can use this to trigger orderbook subscription.

    Attributes:
        event_type: Always MARKET_TRACKED
        market_ticker: Market ticker that was tracked
        category: Market category (e.g., "Sports", "Crypto")
        market_info: Full market info from REST API
        timestamp: When the market was tracked
    """
    event_type: EventType = EventType.MARKET_TRACKED
    market_ticker: str = ""
    category: str = ""
    market_info: Optional[Dict[str, Any]] = None
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
        if self.market_info is None:
            self.market_info = {}


@dataclass
class MarketDeterminedEvent:
    """
    Event data for when a market outcome is determined.

    Emitted by EventLifecycleService when a tracked market receives
    a 'determined' lifecycle event. Signals that orderbook subscription
    should be stopped for this market.

    Attributes:
        event_type: Always MARKET_DETERMINED
        market_ticker: Market ticker that was determined
        result: Market result if available
        determined_ts: Kalshi timestamp when determined (seconds)
        timestamp: When the event was processed locally
    """
    event_type: EventType = EventType.MARKET_DETERMINED
    market_ticker: str = ""
    result: str = ""
    determined_ts: int = 0
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()
