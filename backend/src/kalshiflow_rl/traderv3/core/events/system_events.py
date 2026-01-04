"""
System event dataclasses for TRADER V3.

Contains events related to system activity, trader status, and state transitions.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .types import EventType


@dataclass
class StateTransitionEvent:
    """Event data for state machine transitions."""
    event_type: EventType
    from_state: str
    to_state: str
    context: str
    timestamp: float

    # Optional additional data
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TraderStatusEvent:
    """Event data for trader status updates."""
    event_type: EventType
    state: str
    metrics: Dict[str, Any]
    health: str
    timestamp: float

    # Optional additional data
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemActivityEvent:
    """
    Unified event for all system activity console messages.

    This event type consolidates all console messaging into a single
    pattern, providing clean, informative updates without emoji spam.
    It's designed for both human operators and log analysis tools.

    Activity Types:
        - "state_transition": State machine state changes
        - "sync": Kalshi API synchronization events
        - "health_check": Component health status updates
        - "operation": Trade execution and order management
        - "connection": WebSocket connection events

    Attributes:
        event_type: Always SYSTEM_ACTIVITY
        activity_type: Category of activity
        message: Clean informative text (no emojis)
        metadata: Rich contextual data for the activity
        timestamp: When the activity occurred
    """
    event_type: EventType = EventType.SYSTEM_ACTIVITY
    activity_type: str = ""  # "state_transition", "sync", "health_check", "operation"
    message: str = ""  # Clean informative text (no emojis)
    metadata: Optional[Dict[str, Any]] = None  # Rich contextual data
    timestamp: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class PublicTradeEvent:
    """
    Event data for public trades from Kalshi.

    Used for trade monitoring and RLM strategy analysis.

    Attributes:
        event_type: Always PUBLIC_TRADE_RECEIVED
        market_ticker: Market ticker where trade occurred
        timestamp_ms: Trade timestamp in milliseconds
        side: Taker side ("yes" or "no")
        price_cents: Trade price in cents (0-100)
        count: Number of contracts traded
        received_at: Local timestamp when event was received
    """
    event_type: EventType = EventType.PUBLIC_TRADE_RECEIVED
    market_ticker: str = ""
    timestamp_ms: int = 0
    side: str = ""
    price_cents: int = 0
    count: int = 0
    received_at: float = 0.0

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.received_at == 0.0:
            self.received_at = time.time()
