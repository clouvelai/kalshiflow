"""
Event Bus for TRADER V3 - Central Event Distribution System.

This module implements a robust publish-subscribe event bus that serves as
the nervous system of the V3 trader, enabling loose coupling between components
while maintaining high-performance event distribution.

Purpose:
    The EventBus enables components to communicate without direct dependencies,
    following the observer pattern. It ensures events are processed asynchronously
    without blocking the publisher, critical for real-time trading systems.

Key Responsibilities:
    1. **Event Distribution** - Routes events to interested subscribers
    2. **Async Processing** - Non-blocking event queue with background processing
    3. **Error Isolation** - Subscriber errors don't affect other subscribers
    4. **Performance Monitoring** - Tracks event throughput and errors
    5. **Type Safety** - Strongly-typed events with dataclasses
    6. **Circuit Breaking** - Protects against cascading failures
    7. **Batch Draining** - Processes events in batches to handle bursty traffic

Event Types:
    - ORDERBOOK_SNAPSHOT/DELTA: Market data updates from Kalshi
    - STATE_TRANSITION: State machine state changes
    - TRADER_STATUS: System health and metrics updates
    - SYSTEM_ACTIVITY: Unified console messaging events
    - CONNECTION_STATUS: WebSocket connection state changes

Architecture Position:
    The EventBus is a core V3 component used by:
    - V3Coordinator: Publishes status and state events
    - V3StateMachine: Publishes state transition events
    - V3OrderbookIntegration: Publishes market data events
    - V3WebSocketManager: Subscribes to events for client broadcast
    - Future: Actor service will subscribe to market events

Design Principles:
    - **Non-blocking**: Publishers never wait for subscribers
    - **Error Isolation**: One bad subscriber can't break others
    - **Type Safety**: All events are strongly typed dataclasses
    - **Observable**: Comprehensive metrics and health monitoring
    - **Scalable**: Queue-based with configurable capacity

Performance Characteristics:
    - Queue capacity: 10,000 events (configurable via QUEUE_CAPACITY)
    - Batch drain: Up to 500 events per processing cycle (configurable via BATCH_SIZE)
    - Ticker coalescing: Duplicate market_ticker_update events within a batch
      are collapsed to keep only the latest per market ticker
    - Critical events (trades, fills, state transitions): concurrent subscriber
      notification via asyncio.gather for lowest latency
    - Non-critical events: sequential subscriber notification to reduce task overhead
    - Processing timeout: 5 seconds per batch
    - Callback timeout: 5 seconds per subscriber group (critical path only)
    - Circuit breaker: Triggers at 100 callback errors
    - Drop reporting: Aggregated every 30 seconds to avoid log spam
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List, Callable, Any, Optional

from .events import (
    EventType,
    MarketEvent,
    StateTransitionEvent,
    TraderStatusEvent,
    SystemActivityEvent,
    PublicTradeEvent,
    MarketPositionEvent,
    MarketTickerEvent,
    OrderFillEvent,
    MarketLifecycleEvent,
    MarketTrackedEvent,
    MarketDeterminedEvent,
    TradeFlowMarketUpdateEvent,
    TradeFlowTradeArrivedEvent,
    TMOFetchedEvent,
)

logger = logging.getLogger("kalshiflow_rl.traderv3.event_bus")

# ============================================================
# Tunable constants
# ============================================================

# Maximum number of events the queue can hold before dropping.
# 50K provides ~7+ hours of headroom given the pre-filter reduces
# inbound trade volume by ~80% and steady-state production is ~10 ev/s.
QUEUE_CAPACITY = 50_000

# How many queued events to drain per processing cycle.  Keeps the
# consumer from falling behind during bursty traffic while still
# yielding to the event loop between batches.
BATCH_SIZE = 500

# How often (seconds) to log aggregated drop statistics.
# Prevents per-event warning spam during sustained back-pressure.
DROP_REPORT_INTERVAL = 30.0

# Event types that use concurrent subscriber notification (asyncio.gather).
# All other event types use sequential notification to reduce task overhead.
# Only latency-sensitive events that benefit from parallelism should be here.
CRITICAL_EVENT_TYPES = frozenset({
    "public_trade_received",
    "order_fill",
    "state_transition",
})


class EventBus:
    """
    Async event bus for TRADER V3 - the system's nervous system.

    Implements a high-performance publish-subscribe pattern with
    async processing, error isolation, and comprehensive monitoring.
    All events flow through a queue for non-blocking operation.

    Core Features:
        - **Non-blocking emission**: Publishers never wait
        - **Async processing**: Background task processes events in batches
        - **Error isolation**: Subscriber errors are contained
        - **Performance monitoring**: Tracks throughput, errors, and drops
        - **Circuit breaker**: Protects against failure cascades
        - **Type-safe events**: Strongly typed event dataclasses
        - **Burst tolerance**: 10k queue + batch drain handles WebSocket bursts

    Key Attributes:
        _subscribers: Dict mapping event types to callback lists
        _event_queue: Async queue for event processing (max QUEUE_CAPACITY)
        _processing_task: Background task processing events
        _running: Whether the bus is operational
        _events_emitted: Counter of events published
        _events_processed: Counter of events delivered
        _events_dropped: Counter of events dropped due to full queue
        _drops_by_type: Per-EventType drop counters for diagnostics
        _callback_errors: Counter of subscriber errors

    Thread Safety:
        Designed for single event loop operation. All methods
        should be called from the same asyncio event loop.

    Usage Pattern:
        ```python
        bus = EventBus()
        await bus.start()

        # Subscribe to events
        bus.subscribe(EventType.STATE_TRANSITION, my_callback)

        # Emit events (non-blocking)
        await bus.emit_state_transition("idle", "ready", "System started")

        await bus.stop()
        ```
    """

    def __init__(self):
        """Initialize event bus with 10k queue and batch drain support."""
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_CAPACITY)
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._shutdown_requested = False

        # Performance monitoring
        self._events_emitted = 0
        self._events_processed = 0
        self._events_coalesced = 0
        self._callback_errors = 0
        self._last_error: Optional[str] = None
        self._started_at: Optional[float] = None

        # Drop tracking -- aggregated to avoid per-event log spam
        self._events_dropped = 0
        self._drops_by_type: Dict[str, int] = defaultdict(int)
        self._last_drop_report_time: float = 0.0

        logger.info(
            "TRADER V3 EventBus initialized "
            f"(queue_capacity={QUEUE_CAPACITY}, batch_size={BATCH_SIZE})"
        )
    
    async def start(self) -> None:
        """Start the event bus processing loop."""
        if self._running:
            logger.warning("EventBus is already running")
            return
        
        logger.info("Starting TRADER V3 EventBus...")
        self._running = True
        self._shutdown_requested = False
        self._started_at = time.time()
        
        # Start async processing loop
        self._processing_task = asyncio.create_task(self._process_events())
        
        logger.info("✅ TRADER V3 EventBus started")
    
    async def stop(self) -> None:
        """Stop the event bus and cleanup."""
        if not self._running:
            return
        
        logger.info("Stopping TRADER V3 EventBus...")
        self._shutdown_requested = True
        self._running = False
        
        # Wait for processing to complete
        if self._processing_task:
            try:
                await asyncio.wait_for(self._processing_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("EventBus processing task timeout during shutdown")
                self._processing_task.cancel()
        
        # Clear subscribers
        self._subscribers.clear()
        
        # Log final drop report if there were any drops
        if self._events_dropped > 0:
            self._log_drop_report()

        logger.info(
            f"TRADER V3 EventBus stopped. "
            f"emitted={self._events_emitted}, "
            f"processed={self._events_processed}, "
            f"dropped={self._events_dropped}"
        )
    
    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Async callback function to handle events
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        logger.info(f"Subscriber added for {event_type.value}: {callback.__name__}")
    
    def unsubscribe(self, event_type: EventType, callback: Callable) -> None:
        """
        Unsubscribe from events.
        
        Args:
            event_type: Event type to unsubscribe from  
            callback: Callback to remove
        """
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
                logger.info(f"Subscriber removed for {event_type.value}: {callback.__name__}")
            except ValueError:
                logger.warning(f"Callback not found in subscribers: {callback.__name__}")
    
    async def emit_market_event(
        self,
        event_type: EventType,
        market_ticker: str,
        sequence_number: int,
        timestamp_ms: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Emit a market event (non-blocking).
        
        Args:
            event_type: Type of event
            market_ticker: Market that triggered event
            sequence_number: Sequence number of update
            timestamp_ms: Timestamp of update
            metadata: Optional additional data
            
        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False
        
        event = MarketEvent(
            event_type=event_type,
            market_ticker=market_ticker,
            sequence_number=sequence_number,
            timestamp_ms=timestamp_ms,
            received_at=time.time(),
            metadata=metadata
        )
        
        return await self._queue_event(event)
    
    async def emit_state_transition(
        self,
        from_state: str,
        to_state: str,
        context: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Emit a state transition event (non-blocking).
        
        Args:
            from_state: Previous state
            to_state: New state
            context: Human-readable description of transition
            metadata: Optional additional data
            
        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            logger.warning(f"EventBus not running, cannot emit state transition: {from_state} → {to_state}")
            return False
        
        event = StateTransitionEvent(
            event_type=EventType.STATE_TRANSITION,
            from_state=from_state,
            to_state=to_state,
            context=context,
            timestamp=time.time(),
            metadata=metadata
        )
        
        logger.debug(f"Emitting state transition: {from_state} → {to_state}")
        return await self._queue_event(event)
    
    async def emit_trader_status(
        self,
        state: str,
        metrics: Dict[str, Any],
        health: str = "healthy",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Emit a trader status event (non-blocking).
        
        Args:
            state: Current trader state
            metrics: Performance metrics
            health: Health status ("healthy", "degraded", "unhealthy")
            metadata: Optional additional data
            
        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False
        
        event = TraderStatusEvent(
            event_type=EventType.TRADER_STATUS,
            state=state,
            metrics=metrics,
            health=health,
            timestamp=time.time(),
            metadata=metadata
        )
        
        return await self._queue_event(event)
    
    async def emit_system_activity(
        self,
        activity_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        severity: Optional[str] = None
    ) -> bool:
        """
        Emit a unified system activity event for console messaging.
        
        Args:
            activity_type: Type of activity ("state_transition", "sync", "health_check", etc.)
            message: Clean informative message text (no emojis)
            metadata: Optional contextual data for the activity
            severity: Optional severity level ("error", "warning", "info", "success")
            
        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False
        
        # Include severity in metadata if provided
        if metadata is None:
            metadata = {}
        if severity:
            metadata["severity"] = severity
        
        event = SystemActivityEvent(
            event_type=EventType.SYSTEM_ACTIVITY,
            activity_type=activity_type,
            message=message,
            metadata=metadata,
            timestamp=time.time()
        )
        
        return await self._queue_event(event)
    
    async def emit_orderbook_snapshot(
        self,
        market_ticker: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Emit an orderbook snapshot event (instance method for V3).
        
        Args:
            market_ticker: Market that had snapshot
            metadata: Metadata containing sequence_number, timestamp_ms, etc.
            
        Returns:
            True if event was queued successfully
        """
        sequence_number = metadata.get("sequence_number", 0) if metadata else 0
        timestamp_ms = metadata.get("timestamp_ms", int(time.time() * 1000)) if metadata else int(time.time() * 1000)
        
        return await self.emit_market_event(
            event_type=EventType.ORDERBOOK_SNAPSHOT,
            market_ticker=market_ticker,
            sequence_number=sequence_number,
            timestamp_ms=timestamp_ms,
            metadata=metadata
        )
    
    async def emit_orderbook_delta(
        self,
        market_ticker: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Emit an orderbook delta event (instance method for V3).
        
        Args:
            market_ticker: Market that had delta
            metadata: Metadata containing sequence_number, timestamp_ms, etc.
            
        Returns:
            True if event was queued successfully
        """
        sequence_number = metadata.get("sequence_number", 0) if metadata else 0
        timestamp_ms = metadata.get("timestamp_ms", int(time.time() * 1000)) if metadata else int(time.time() * 1000)
        
        return await self.emit_market_event(
            event_type=EventType.ORDERBOOK_DELTA,
            market_ticker=market_ticker,
            sequence_number=sequence_number,
            timestamp_ms=timestamp_ms,
            metadata=metadata
        )
    
    async def subscribe_to_orderbook_snapshot(self, callback: Callable) -> None:
        """
        Subscribe to orderbook snapshot events.
        
        Args:
            callback: Async function(market_ticker: str, metadata: Dict) to call on snapshot
        """
        # Ensure the event type exists in subscribers
        if EventType.ORDERBOOK_SNAPSHOT not in self._subscribers:
            self._subscribers[EventType.ORDERBOOK_SNAPSHOT] = []
        
        # Store the callback directly without a wrapper
        # The _notify_subscribers method will handle the parameter extraction
        self._subscribers[EventType.ORDERBOOK_SNAPSHOT].append(callback)
        logger.debug(f"Added orderbook snapshot subscriber: {callback.__name__}")
    
    async def subscribe_to_orderbook_delta(self, callback: Callable) -> None:
        """
        Subscribe to orderbook delta events.

        Args:
            callback: Async function(market_ticker: str, metadata: Dict) to call on delta
        """
        # Ensure the event type exists in subscribers
        if EventType.ORDERBOOK_DELTA not in self._subscribers:
            self._subscribers[EventType.ORDERBOOK_DELTA] = []

        # Store the callback directly without a wrapper
        # The _notify_subscribers method will handle the parameter extraction
        self._subscribers[EventType.ORDERBOOK_DELTA].append(callback)
        logger.debug(f"Added orderbook delta subscriber: {callback.__name__}")

    async def emit_public_trade(self, trade_data: Dict[str, Any]) -> bool:
        """
        Emit a public trade event (non-blocking).

        Args:
            trade_data: Trade data containing market_ticker, timestamp_ms,
                       taker_side, yes_price/no_price, count

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        # Determine price in cents based on taker side
        side = trade_data.get("taker_side", "unknown")
        if side == "yes":
            price_cents = trade_data.get("yes_price", 0)
        else:
            price_cents = trade_data.get("no_price", 0)

        event = PublicTradeEvent(
            event_type=EventType.PUBLIC_TRADE_RECEIVED,
            market_ticker=trade_data.get("market_ticker", ""),
            timestamp_ms=trade_data.get("timestamp_ms", 0),
            side=side,
            price_cents=price_cents,
            count=trade_data.get("count", 0),
            received_at=time.time(),
        )

        return await self._queue_event(event)

    async def subscribe_to_public_trade(self, callback: Callable) -> None:
        """
        Subscribe to public trade events.

        Args:
            callback: Async function(trade_event: PublicTradeEvent) to call on trade
        """
        if EventType.PUBLIC_TRADE_RECEIVED not in self._subscribers:
            self._subscribers[EventType.PUBLIC_TRADE_RECEIVED] = []

        self._subscribers[EventType.PUBLIC_TRADE_RECEIVED].append(callback)
        logger.debug(f"Added public trade subscriber: {callback.__name__}")

    async def emit_market_position_update(
        self,
        ticker: str,
        position_data: Dict[str, Any]
    ) -> bool:
        """
        Emit a market position update event (non-blocking).

        Called by PositionListener when receiving real-time position updates
        from Kalshi WebSocket market_positions channel.

        Args:
            ticker: Market ticker for this position
            position_data: Position details (position, market_exposure, realized_pnl, etc.)

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = MarketPositionEvent(
            event_type=EventType.MARKET_POSITION_UPDATE,
            market_ticker=ticker,
            position_data=position_data,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def emit_market_ticker_update(
        self,
        ticker: str,
        price_data: Dict[str, Any]
    ) -> bool:
        """
        Emit a market ticker update event (non-blocking).

        Called by MarketTickerListener when receiving real-time price updates
        from Kalshi WebSocket ticker channel.

        Args:
            ticker: Market ticker for this price update
            price_data: Price details (last_price, yes_bid, yes_ask, volume, etc.)

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = MarketTickerEvent(
            event_type=EventType.MARKET_TICKER_UPDATE,
            market_ticker=ticker,
            price_data=price_data,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def subscribe_to_market_ticker(self, callback: Callable) -> None:
        """
        Subscribe to market ticker update events.

        Args:
            callback: Async callback function(MarketTickerEvent)
        """
        # subscribe() is synchronous, no await needed
        self.subscribe(EventType.MARKET_TICKER_UPDATE, callback)

    async def subscribe_to_market_position(self, callback: Callable) -> None:
        """
        Subscribe to market position update events.

        Args:
            callback: Async function(event: MarketPositionEvent) to call on update
        """
        if EventType.MARKET_POSITION_UPDATE not in self._subscribers:
            self._subscribers[EventType.MARKET_POSITION_UPDATE] = []

        self._subscribers[EventType.MARKET_POSITION_UPDATE].append(callback)
        logger.debug(f"Added market position subscriber: {callback.__name__}")

    async def emit_order_fill(
        self,
        trade_id: str,
        order_id: str,
        market_ticker: str,
        is_taker: bool,
        side: str,
        action: str,
        price_cents: int,
        count: int,
        post_position: int,
        fill_timestamp: int,
    ) -> bool:
        """
        Emit an order fill event (non-blocking).

        Called by FillListener when receiving real-time fill notifications
        from Kalshi WebSocket fill channel.

        Args:
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

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = OrderFillEvent(
            event_type=EventType.ORDER_FILL,
            trade_id=trade_id,
            order_id=order_id,
            market_ticker=market_ticker,
            is_taker=is_taker,
            side=side,
            action=action,
            price_cents=price_cents,
            count=count,
            post_position=post_position,
            fill_timestamp=fill_timestamp,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def subscribe_to_order_fill(self, callback: Callable) -> None:
        """
        Subscribe to order fill events.

        Args:
            callback: Async function(event: OrderFillEvent) to call on fill
        """
        if EventType.ORDER_FILL not in self._subscribers:
            self._subscribers[EventType.ORDER_FILL] = []

        self._subscribers[EventType.ORDER_FILL].append(callback)
        logger.debug(f"Added order fill subscriber: {callback.__name__}")

    # ============================================================
    # Event Lifecycle Discovery Methods
    # ============================================================

    async def emit_market_lifecycle(
        self,
        event_type: str,
        market_ticker: str,
        payload: Dict[str, Any],
    ) -> bool:
        """
        Emit a market lifecycle event (non-blocking).

        Called by V3LifecycleIntegration when receiving events from
        the market_lifecycle_v2 WebSocket channel.

        Args:
            event_type: Lifecycle event type (created, determined, settled, etc.)
            market_ticker: Market ticker for this event
            payload: Full event payload from Kalshi

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = MarketLifecycleEvent(
            event_type=EventType.MARKET_LIFECYCLE_EVENT,
            lifecycle_event_type=event_type,
            market_ticker=market_ticker,
            payload=payload,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def emit_market_tracked(
        self,
        market_ticker: str,
        category: str,
        market_info: Dict[str, Any],
    ) -> bool:
        """
        Emit a market tracked event (non-blocking).

        Called by EventLifecycleService when a market passes filtering
        and is added to tracking. Signals that orderbook subscription
        should be started.

        Args:
            market_ticker: Market ticker that was tracked
            category: Market category
            market_info: Full market info from REST API

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = MarketTrackedEvent(
            event_type=EventType.MARKET_TRACKED,
            market_ticker=market_ticker,
            category=category,
            market_info=market_info,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def emit_market_determined(
        self,
        market_ticker: str,
        result: str = "",
        determined_ts: int = 0,
    ) -> bool:
        """
        Emit a market determined event (non-blocking).

        Called by EventLifecycleService when a tracked market's outcome
        is resolved. Signals that orderbook subscription should be stopped.

        Args:
            market_ticker: Market ticker that was determined
            result: Market result if available
            determined_ts: Kalshi timestamp when determined (seconds)

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = MarketDeterminedEvent(
            event_type=EventType.MARKET_DETERMINED,
            market_ticker=market_ticker,
            result=result,
            determined_ts=determined_ts,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def subscribe_to_market_lifecycle(self, callback: Callable) -> None:
        """
        Subscribe to market lifecycle events.

        Args:
            callback: Async function(event: MarketLifecycleEvent) to call on event
        """
        if EventType.MARKET_LIFECYCLE_EVENT not in self._subscribers:
            self._subscribers[EventType.MARKET_LIFECYCLE_EVENT] = []

        self._subscribers[EventType.MARKET_LIFECYCLE_EVENT].append(callback)
        logger.debug(f"Added market lifecycle subscriber: {callback.__name__}")

    async def subscribe_to_market_tracked(self, callback: Callable) -> None:
        """
        Subscribe to market tracked events.

        Args:
            callback: Async function(event: MarketTrackedEvent) to call when market tracked
        """
        if EventType.MARKET_TRACKED not in self._subscribers:
            self._subscribers[EventType.MARKET_TRACKED] = []

        self._subscribers[EventType.MARKET_TRACKED].append(callback)
        logger.debug(f"Added market tracked subscriber: {callback.__name__}")

    async def subscribe_to_market_determined(self, callback: Callable) -> None:
        """
        Subscribe to market determined events.

        Args:
            callback: Async function(event: MarketDeterminedEvent) to call when market determined
        """
        if EventType.MARKET_DETERMINED not in self._subscribers:
            self._subscribers[EventType.MARKET_DETERMINED] = []

        self._subscribers[EventType.MARKET_DETERMINED].append(callback)
        logger.debug(f"Added market determined subscriber: {callback.__name__}")

    # ============================================================
    # Trade Flow Event Methods
    # ============================================================

    async def emit_trade_flow_market_update(
        self,
        market_ticker: str,
        state: Dict[str, Any],
    ) -> bool:
        """
        Emit a trade flow market state update event (non-blocking).

        Called by MarketStateAgent when a tracked market's trade state changes.
        Used for real-time UI updates showing trade direction and price movement.

        Args:
            market_ticker: Market ticker for this update
            state: Dictionary containing trade flow state (yes_trades, no_trades, etc.)

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = TradeFlowMarketUpdateEvent(
            event_type=EventType.TRADE_FLOW_MARKET_UPDATE,
            market_ticker=market_ticker,
            state=state,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def emit_trade_flow_trade_arrived(
        self,
        market_ticker: str,
        side: str,
        count: int,
        yes_price: int,
        event_ticker: str = "",
    ) -> bool:
        """
        Emit a trade flow trade arrived event (non-blocking).

        Called by MarketStateAgent for every trade in a tracked market.
        Used for real-time UI pulse/glow animations on trade arrival.

        Args:
            market_ticker: Market ticker where trade occurred
            side: Trade side ("yes" or "no")
            count: Number of contracts in this trade
            yes_price: YES price in cents
            event_ticker: Event ticker this market belongs to

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = TradeFlowTradeArrivedEvent(
            event_type=EventType.TRADE_FLOW_TRADE_ARRIVED,
            market_ticker=market_ticker,
            event_ticker=event_ticker,
            side=side,
            count=count,
            price_cents=yes_price,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def subscribe_to_trade_flow_market_update(self, callback: Callable) -> None:
        """
        Subscribe to trade flow market update events.

        Args:
            callback: Async function(event: TradeFlowMarketUpdateEvent) to call on update
        """
        if EventType.TRADE_FLOW_MARKET_UPDATE not in self._subscribers:
            self._subscribers[EventType.TRADE_FLOW_MARKET_UPDATE] = []

        self._subscribers[EventType.TRADE_FLOW_MARKET_UPDATE].append(callback)
        logger.debug(f"Added trade flow market update subscriber: {callback.__name__}")

    async def subscribe_to_trade_flow_trade_arrived(self, callback: Callable) -> None:
        """
        Subscribe to trade flow trade arrived events.

        Args:
            callback: Async function(event: TradeFlowTradeArrivedEvent) to call on trade
        """
        if EventType.TRADE_FLOW_TRADE_ARRIVED not in self._subscribers:
            self._subscribers[EventType.TRADE_FLOW_TRADE_ARRIVED] = []

        self._subscribers[EventType.TRADE_FLOW_TRADE_ARRIVED].append(callback)
        logger.debug(f"Added trade flow trade arrived subscriber: {callback.__name__}")

    # ============================================================
    # True Market Open (TMO) Event Methods
    # ============================================================

    async def emit_tmo_fetched(
        self,
        market_ticker: str,
        true_market_open: int,
        open_ts: int = 0,
    ) -> bool:
        """
        Emit a True Market Open fetched event (non-blocking).

        Called by TrueMarketOpenFetcher when the true market open price
        is successfully retrieved from the Kalshi candlestick API.

        Args:
            market_ticker: Market ticker for this TMO
            true_market_open: YES price in cents at market open
            open_ts: Unix timestamp when the market opened

        Returns:
            True if event was queued, False if queue full
        """
        if not self._running:
            return False

        event = TMOFetchedEvent(
            event_type=EventType.TMO_FETCHED,
            market_ticker=market_ticker,
            true_market_open=true_market_open,
            open_ts=open_ts,
            timestamp=time.time(),
        )

        return await self._queue_event(event)

    async def subscribe_to_tmo_fetched(self, callback: Callable) -> None:
        """
        Subscribe to TMO fetched events.

        Args:
            callback: Async function(event: TMOFetchedEvent) to call when TMO fetched
        """
        if EventType.TMO_FETCHED not in self._subscribers:
            self._subscribers[EventType.TMO_FETCHED] = []

        self._subscribers[EventType.TMO_FETCHED].append(callback)
        logger.debug(f"Added TMO fetched subscriber: {callback.__name__}")

    async def _queue_event(self, event: Any) -> bool:
        """
        Queue an event for processing.

        On queue-full, the event is dropped and the drop is recorded by
        event type. Drop statistics are logged periodically (every
        DROP_REPORT_INTERVAL seconds) instead of per-event to prevent
        log spam during sustained back-pressure.

        Args:
            event: Event object to queue

        Returns:
            True if event was queued, False if queue full
        """
        try:
            # Non-blocking queue put
            self._event_queue.put_nowait(event)
            self._events_emitted += 1
            return True

        except asyncio.QueueFull:
            # Track drop by event type
            event_type_value = event.event_type.value if hasattr(event, "event_type") else "unknown"
            self._events_dropped += 1
            self._drops_by_type[event_type_value] += 1

            # Periodic aggregated drop report (avoids per-event log spam)
            now = time.time()
            if now - self._last_drop_report_time >= DROP_REPORT_INTERVAL:
                self._log_drop_report()
                self._last_drop_report_time = now

            return False
        except Exception as e:
            logger.error(f"Error queuing event: {e}")
            return False

    def _log_drop_report(self) -> None:
        """
        Log aggregated drop statistics and reset per-type counters.

        Called at most every DROP_REPORT_INTERVAL seconds from _queue_event
        to provide actionable diagnostics without flooding logs.
        """
        if not self._drops_by_type:
            return

        breakdown = ", ".join(
            f"{etype}={count}" for etype, count in sorted(
                self._drops_by_type.items(), key=lambda x: x[1], reverse=True
            )
        )
        logger.warning(
            f"EventBus dropped {self._events_dropped} events total "
            f"(queue_capacity={QUEUE_CAPACITY}). "
            f"Recent breakdown: {breakdown}. "
            f"Queue size: {self._event_queue.qsize()}/{QUEUE_CAPACITY}"
        )
        # Reset per-type counters after reporting (total stays cumulative)
        self._drops_by_type.clear()
    
    async def _process_events(self) -> None:
        """
        Main event processing loop (runs in background).

        This is the heart of the event bus - a background task that
        continuously pulls events from the queue and distributes them
        to subscribers. It provides error isolation and timeout protection.

        Processing Flow:
            1. Wait for first event from queue (1s timeout)
            2. Drain up to BATCH_SIZE events from the queue (non-blocking)
            3. Process all drained events sequentially
            4. Yield to event loop between batches
            5. Repeat

        Batch Draining:
            After receiving the first event via blocking get(), the loop
            drains up to (BATCH_SIZE - 1) additional events using
            get_nowait(). This lets the consumer keep pace with bursty
            WebSocket traffic (e.g. 30-40 orderbook deltas arriving in
            the same millisecond) without falling behind.

        Error Handling:
            - Subscriber errors are logged but don't stop processing
            - Queue timeouts just continue the loop
            - CancelledError stops the loop gracefully

        Performance:
            - Batch drain keeps consumer from falling behind bursts
            - Concurrent subscriber notification for parallelism
            - 5-second timeout on subscriber batch completion
        """
        logger.info("TRADER V3 event processing loop started")

        while not self._shutdown_requested:
            try:
                # Block-wait for the first event (1s timeout)
                try:
                    first_event = await asyncio.wait_for(
                        self._event_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Drain up to BATCH_SIZE events from the queue (non-blocking).
                batch = [first_event]
                for _ in range(BATCH_SIZE - 1):
                    try:
                        batch.append(self._event_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                # Coalesce market_ticker_update events: keep only the
                # latest update per market ticker to reduce subscriber load.
                batch = self._coalesce_ticker_updates(batch)

                # Process each event in the batch
                for event in batch:
                    try:
                        logger.debug(
                            f"Processing event: "
                            f"{event.event_type.value if hasattr(event, 'event_type') else type(event).__name__}"
                        )
                        await self._notify_subscribers(event)
                    except Exception as e:
                        logger.error(f"Error notifying subscribers for event: {e}")
                    finally:
                        self._event_queue.task_done()
                        self._events_processed += 1

            except asyncio.CancelledError:
                logger.info("Event processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error

        logger.info("TRADER V3 event processing loop stopped")
    
    def _coalesce_ticker_updates(self, batch: list) -> list:
        """
        Coalesce market_ticker_update events within a batch.

        When multiple ticker updates arrive for the same market in one batch,
        only the LATEST update (last in the batch) is kept. All earlier
        updates for that ticker are dropped since only the most recent
        price data matters. Other event types pass through unchanged.

        This dramatically reduces subscriber processing when 107+ markets
        each produce frequent ticker updates.

        Args:
            batch: List of events drained from the queue

        Returns:
            Filtered batch with deduplicated ticker updates
        """
        # Fast path: nothing to coalesce in small batches
        if len(batch) <= 1:
            return batch

        # Walk the batch in reverse to find the LAST ticker update per market
        seen_tickers: set = set()
        coalesced: list = []
        coalesced_count = 0

        for event in reversed(batch):
            if (
                hasattr(event, "event_type")
                and event.event_type == EventType.MARKET_TICKER_UPDATE
                and hasattr(event, "market_ticker")
            ):
                if event.market_ticker in seen_tickers:
                    # Duplicate ticker update -- drop it but still mark task_done
                    self._event_queue.task_done()
                    self._events_coalesced += 1
                    coalesced_count += 1
                    continue
                seen_tickers.add(event.market_ticker)

            coalesced.append(event)

        # Reverse back to original order
        coalesced.reverse()

        if coalesced_count > 0:
            logger.debug(
                f"Coalesced {coalesced_count} duplicate ticker updates "
                f"({len(batch)} -> {len(coalesced)} events)"
            )

        return coalesced

    async def _notify_subscribers(self, event: Any) -> None:
        """
        Notify all subscribers of an event with error isolation.

        Distributes an event to all registered subscribers for that
        event type. For critical event types (trades, fills, state transitions),
        subscribers are called concurrently via asyncio.gather for lowest
        latency. For all other event types, subscribers are called sequentially
        to avoid the overhead of creating N async tasks per event.

        Special Handling:
            - MarketEvent: Extracts market_ticker and metadata parameters
            - Other events: Passes full event object to callback

        Args:
            event: Event object to distribute to subscribers

        Implementation Notes:
            - Critical events: concurrent tasks with 5-second timeout
            - Non-critical events: sequential calls with per-subscriber error isolation
            - Logs but doesn't fail on individual callback errors
        """
        # Copy subscriber list to prevent mutation during iteration
        subscribers = list(self._subscribers.get(event.event_type, []))

        if not subscribers:
            return

        logger.debug(f"Notifying {len(subscribers)} subscribers for {event.event_type.value}")

        # Determine call args based on event type
        is_orderbook = isinstance(event, MarketEvent) and event.event_type in [
            EventType.ORDERBOOK_SNAPSHOT, EventType.ORDERBOOK_DELTA
        ]

        event_type_value = event.event_type.value if hasattr(event, "event_type") else ""
        use_concurrent = event_type_value in CRITICAL_EVENT_TYPES

        if use_concurrent:
            # Critical path: concurrent notification for lowest latency
            tasks = []
            for callback in subscribers:
                if is_orderbook:
                    task = asyncio.create_task(
                        self._safe_call_subscriber(callback, event.market_ticker, event.metadata or {})
                    )
                else:
                    task = asyncio.create_task(
                        self._safe_call_subscriber(callback, event)
                    )
                tasks.append(task)

            if tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Subscriber callbacks timeout for {event_type_value}")
        else:
            # Non-critical path: sequential notification to reduce task overhead
            for callback in subscribers:
                if is_orderbook:
                    await self._safe_call_subscriber(callback, event.market_ticker, event.metadata or {})
                else:
                    await self._safe_call_subscriber(callback, event)
    
    async def _safe_call_subscriber(self, callback: Callable, *args) -> None:
        """
        Safely call a subscriber callback with error isolation.
        
        Args:
            callback: Subscriber callback to call
            *args: Variable arguments to pass to the callback
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
                
        except Exception as e:
            self._callback_errors += 1
            self._last_error = str(e)
            logger.error(f"Error in subscriber callback {callback.__name__}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics including drop metrics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "events_emitted": self._events_emitted,
            "events_processed": self._events_processed,
            "events_coalesced": self._events_coalesced,
            "events_dropped": self._events_dropped,
            "callback_errors": self._callback_errors,
            "last_error": self._last_error,
            "queue_size": self._event_queue.qsize(),
            "queue_capacity": QUEUE_CAPACITY,
            "subscriber_count": sum(len(subs) for subs in self._subscribers.values()),
            "uptime_seconds": uptime,
            "events_per_second": self._events_processed / max(uptime, 1),
        }
    
    def is_healthy(self) -> bool:
        """
        Check if event bus is healthy.
        
        Health is determined by:
            1. Bus is running (_running = True)
            2. Processing task is active and not done
            3. Callback errors below threshold (< 100)
        
        Returns:
            True if all health checks pass, False otherwise
        
        Usage:
            Used by V3Coordinator health monitoring to detect
            event bus failures and trigger recovery if needed.
        """
        if not self._running:
            return False
        
        # Check if processing task is running
        if self._processing_task is None or self._processing_task.done():
            return False
        
        # Check for excessive callback errors (indicates problems)
        if self._callback_errors > 100:
            return False
        
        return True
    
    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information.

        Returns:
            Dictionary with health status and operational details
        """
        stats = self.get_stats()
        return {
            "running": self._running,
            "processing_task_active": (
                self._processing_task is not None and not self._processing_task.done()
            ),
            "events_emitted": stats.get("events_emitted", 0),
            "events_processed": stats.get("events_processed", 0),
            "events_dropped": stats.get("events_dropped", 0),
            "events_coalesced": stats.get("events_coalesced", 0),
            "queue_size": stats.get("queue_size", 0),
            "queue_capacity": stats.get("queue_capacity", QUEUE_CAPACITY),
            "subscriber_count": stats.get("subscriber_count", 0),
            "callback_errors": stats.get("callback_errors", 0),
            "last_error": stats.get("last_error"),
            "uptime_seconds": stats.get("uptime_seconds", 0),
        }
    
    def get_last_activity_time(self) -> Optional[float]:
        """
        Get last event processed time.
        
        Returns:
            Timestamp when event bus started (as proxy for last activity),
            or None if not started
        """
        return self._started_at