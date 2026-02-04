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

    # V3 specific events
    STATE_TRANSITION = "state_transition"
    TRADER_STATUS = "trader_status"
    SYSTEM_ACTIVITY = "system_activity"  # Unified console messaging

    # Public trade events
    PUBLIC_TRADE_RECEIVED = "public_trade_received"

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

    # Trade Flow events (market microstructure tracking)
    TRADE_FLOW_MARKET_UPDATE = "trade_flow_market_update"    # Trade flow state changed for a market
    TRADE_FLOW_TRADE_ARRIVED = "trade_flow_trade_arrived"    # New trade arrived for tracking

    # True Market Open (TMO) events
    TMO_FETCHED = "tmo_fetched"                        # True market open price fetched from candlestick API

    # Cross-venue arbitrage events
    POLY_PRICE_UPDATE = "poly_price_update"            # Polymarket price update for a paired market
    SPREAD_UPDATE = "spread_update"                    # Spread recalculation for a paired market
    SPREAD_ALERT = "spread_alert"                      # Spread exceeds threshold
    SPREAD_TRADE_EXECUTED = "spread_trade_executed"    # Arb trade executed on hot path
    PAIR_MATCHED = "pair_matched"                      # New cross-venue pair discovered
    KALSHI_API_PRICE_UPDATE = "kalshi_api_price_update"  # Kalshi REST API price poll result

    # Event Codex (background data enrichment)
    EVENT_CODEX_SNAPSHOT = "event_codex_snapshot"        # Full codex snapshot with candle data

