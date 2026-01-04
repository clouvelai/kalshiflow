"""
Event dataclasses for TRADER V3 EventBus.

This module exports all event types and dataclasses used throughout
the V3 trading system. Events are organized by category:

- types: EventType enum defining all event types
- market_events: Market data, positions, tickers, and fills
- lifecycle_events: Market creation, tracking, and determination
- rlm_events: Reverse Line Movement and True Market Open
- system_events: System activity and status

Usage:
    from kalshiflow_rl.traderv3.core.events import (
        EventType,
        MarketEvent,
        OrderFillEvent,
        SystemActivityEvent,
    )
"""

from .types import EventType

from .market_events import (
    MarketEvent,
    MarketPositionEvent,
    MarketTickerEvent,
    OrderFillEvent,
)

from .lifecycle_events import (
    MarketLifecycleEvent,
    MarketTrackedEvent,
    MarketDeterminedEvent,
)

from .rlm_events import (
    RLMMarketUpdateEvent,
    RLMTradeArrivedEvent,
    TMOFetchedEvent,
)

from .system_events import (
    StateTransitionEvent,
    TraderStatusEvent,
    SystemActivityEvent,
    PublicTradeEvent,
)

__all__ = [
    # Types
    "EventType",
    # Market events
    "MarketEvent",
    "MarketPositionEvent",
    "MarketTickerEvent",
    "OrderFillEvent",
    # Lifecycle events
    "MarketLifecycleEvent",
    "MarketTrackedEvent",
    "MarketDeterminedEvent",
    # RLM events
    "RLMMarketUpdateEvent",
    "RLMTradeArrivedEvent",
    "TMOFetchedEvent",
    # System events
    "StateTransitionEvent",
    "TraderStatusEvent",
    "SystemActivityEvent",
    "PublicTradeEvent",
]
