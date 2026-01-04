"""
Event types for TRADER V3.

Defines all event types that can flow through the event bus.
Each event type has specific data structures and subscribers.
"""

from enum import Enum


class EventType(Enum):
    """
    Event types for TRADER V3.

    Defines all event types that can flow through the event bus.
    Each event type has specific data structures and subscribers.
    """
    # Orderbook events (from existing system)
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_DELTA = "orderbook_delta"
    SETTLEMENT = "settlement"

    # V3 specific events
    STATE_TRANSITION = "state_transition"
    TRADER_STATUS = "trader_status"
    CONNECTION_STATUS = "connection_status"
    SYSTEM_ACTIVITY = "system_activity"  # Unified console messaging

    # Whale detection events
    PUBLIC_TRADE_RECEIVED = "public_trade_received"
    WHALE_QUEUE_UPDATED = "whale_queue_updated"

    # Real-time position updates (from WebSocket)
    MARKET_POSITION_UPDATE = "market_position_update"

    # Real-time market price updates (from ticker WebSocket)
    MARKET_TICKER_UPDATE = "market_ticker_update"

    # Real-time order fill notifications (from fill WebSocket)
    ORDER_FILL = "order_fill"

    # Event Lifecycle Discovery events
    MARKET_LIFECYCLE_EVENT = "market_lifecycle_event"  # Raw lifecycle events from Kalshi
    MARKET_TRACKED = "market_tracked"                   # Market added to tracking
    MARKET_DETERMINED = "market_determined"             # Market outcome resolved

    # RLM (Reverse Line Movement) events
    RLM_MARKET_UPDATE = "rlm_market_update"            # RLM state changed for a market
    RLM_TRADE_ARRIVED = "rlm_trade_arrived"            # New trade arrived for RLM tracking

    # True Market Open (TMO) events
    TMO_FETCHED = "tmo_fetched"                        # True market open price fetched from candlestick API
