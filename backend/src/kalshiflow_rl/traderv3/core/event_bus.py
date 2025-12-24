"""
Event Bus for TRADER V3.

Provides a clean event-driven architecture for the V3 trader system.
Ported from trading/event_bus.py with V3-specific enhancements.

Features:
- Non-blocking event emission (never blocks publisher)
- Async callback execution with error isolation
- State machine event support for V3 trader
- Performance monitoring and circuit breaker
- Clean dependency injection (no global singletons in V3)
"""

import asyncio
import logging
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.traderv3.event_bus")


class EventType(Enum):
    """Event types for TRADER V3."""
    # Orderbook events (from existing system)
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_DELTA = "orderbook_delta"
    SETTLEMENT = "settlement"
    
    # V3 specific events
    STATE_TRANSITION = "state_transition"
    TRADER_STATUS = "trader_status"
    CONNECTION_STATUS = "connection_status"


@dataclass
class MarketEvent:
    """Event data for market updates."""
    event_type: EventType
    market_ticker: str
    sequence_number: int
    timestamp_ms: int
    received_at: float
    
    # Optional additional data
    metadata: Optional[Dict[str, Any]] = None


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


class EventBus:
    """
    Simple async event bus for TRADER V3.
    
    Features:
    - Non-blocking event emission (never blocks publisher)
    - Async callback execution with error isolation
    - Subscription management with cleanup
    - Performance monitoring and circuit breaker
    - Support for both market and state machine events
    """
    
    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._shutdown_requested = False
        
        # Performance monitoring
        self._events_emitted = 0
        self._events_processed = 0
        self._callback_errors = 0
        self._last_error: Optional[str] = None
        self._started_at: Optional[float] = None
        
        logger.info("TRADER V3 EventBus initialized")
    
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
        
        logger.info(f"✅ TRADER V3 EventBus stopped. Events emitted: {self._events_emitted}, processed: {self._events_processed}")
    
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
    
    async def _queue_event(self, event: Any) -> bool:
        """
        Queue an event for processing.
        
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
            logger.warning(f"Event queue full, dropping event of type {event.event_type.value}")
            return False
        except Exception as e:
            logger.error(f"Error queuing event: {e}")
            return False
    
    async def _process_events(self) -> None:
        """Main event processing loop (runs in background)."""
        logger.info("TRADER V3 event processing loop started")
        
        while not self._shutdown_requested:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Debug log event processing
                logger.debug(f"Processing event: {event.event_type.value if hasattr(event, 'event_type') else type(event).__name__}")
                
                # Process event by notifying all subscribers
                await self._notify_subscribers(event)
                
                # Mark task as done
                self._event_queue.task_done()
                self._events_processed += 1
                
            except asyncio.CancelledError:
                logger.info("Event processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(0.1)  # Brief pause on error
        
        logger.info("TRADER V3 event processing loop stopped")
    
    async def publish(self, event: Any) -> bool:
        """
        Publish an event to the event bus.
        
        This is a convenience wrapper that routes to the appropriate emit method
        based on event type.
        
        Args:
            event: Event to publish (TraderStatusEvent, StateTransitionEvent, or MarketEvent)
            
        Returns:
            True if event was published successfully
        """
        if isinstance(event, TraderStatusEvent):
            return await self.emit_trader_status(
                state=event.state,
                metrics=event.metrics,
                health=event.health
            )
        elif isinstance(event, StateTransitionEvent):
            return await self.emit_state_transition(
                from_state=event.from_state,
                to_state=event.to_state,
                context=event.context,
                metadata=event.metadata
            )
        elif isinstance(event, MarketEvent):
            if event.event_type == EventType.ORDERBOOK_SNAPSHOT:
                return await self.emit_orderbook_snapshot(
                    market_ticker=event.market_ticker,
                    snapshot=event.metadata.get('snapshot', {})
                )
            elif event.event_type == EventType.ORDERBOOK_DELTA:
                return await self.emit_orderbook_delta(
                    market_ticker=event.market_ticker,
                    delta=event.metadata.get('delta', {})
                )
        
        logger.warning(f"Unknown event type for publish: {type(event)}")
        return False
    
    async def _notify_subscribers(self, event: Any) -> None:
        """
        Notify all subscribers of an event with error isolation.
        
        Args:
            event: Event to send to subscribers
        """
        subscribers = self._subscribers.get(event.event_type, [])
        
        if not subscribers:
            logger.debug(f"No subscribers for event type: {event.event_type.value}")
            return
        
        logger.debug(f"Notifying {len(subscribers)} subscribers for {event.event_type.value}")
        
        # Call all subscribers concurrently with error isolation
        tasks = []
        for callback in subscribers:
            # For orderbook events, extract the parameters for callbacks
            if isinstance(event, MarketEvent) and event.event_type in [EventType.ORDERBOOK_SNAPSHOT, EventType.ORDERBOOK_DELTA]:
                # Call with extracted parameters (market_ticker, metadata)
                task = asyncio.create_task(self._safe_call_subscriber(callback, event.market_ticker, event.metadata or {}))
            else:
                # Call with the full event object
                task = asyncio.create_task(self._safe_call_subscriber(callback, event))
            tasks.append(task)
        
        # Wait for all callbacks to complete (with timeout)
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Subscriber callbacks timeout for {event.event_type.value}")
    
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
        """Get event bus statistics."""
        uptime = time.time() - self._started_at if self._started_at else 0
        
        return {
            "running": self._running,
            "events_emitted": self._events_emitted,
            "events_processed": self._events_processed,
            "callback_errors": self._callback_errors,
            "last_error": self._last_error,
            "queue_size": self._event_queue.qsize(),
            "subscriber_count": sum(len(subs) for subs in self._subscribers.values()),
            "uptime_seconds": uptime,
            "events_per_second": self._events_processed / max(uptime, 1)
        }
    
    def is_healthy(self) -> bool:
        """
        Check if event bus is healthy.
        
        Returns:
            True if running and processing events normally
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
            "queue_size": stats.get("queue_size", 0),
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


# Convenience functions for V3 events

async def emit_orderbook_snapshot(
    event_bus: EventBus,
    market_ticker: str,
    sequence_number: int,
    timestamp_ms: int,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Emit an orderbook snapshot event.
    
    Args:
        event_bus: EventBus instance to use
        market_ticker: Market that had snapshot
        sequence_number: Sequence number
        timestamp_ms: Timestamp
        metadata: Optional additional data
        
    Returns:
        True if event was queued successfully
    """
    return await event_bus.emit_market_event(
        event_type=EventType.ORDERBOOK_SNAPSHOT,
        market_ticker=market_ticker,
        sequence_number=sequence_number,
        timestamp_ms=timestamp_ms,
        metadata=metadata
    )


async def emit_orderbook_delta(
    event_bus: EventBus,
    market_ticker: str,
    sequence_number: int,
    timestamp_ms: int,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Emit an orderbook delta event.
    
    Args:
        event_bus: EventBus instance to use
        market_ticker: Market that had delta
        sequence_number: Sequence number
        timestamp_ms: Timestamp
        metadata: Optional additional data
        
    Returns:
        True if event was queued successfully
    """
    return await event_bus.emit_market_event(
        event_type=EventType.ORDERBOOK_DELTA,
        market_ticker=market_ticker,
        sequence_number=sequence_number,
        timestamp_ms=timestamp_ms,
        metadata=metadata
    )


async def emit_state_transition(
    event_bus: EventBus,
    from_state: str,
    to_state: str,
    context: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Emit a state transition event.
    
    Args:
        event_bus: EventBus instance to use
        from_state: Previous state
        to_state: New state
        context: Human-readable description
        metadata: Optional additional data
        
    Returns:
        True if event was queued successfully
    """
    return await event_bus.emit_state_transition(
        from_state=from_state,
        to_state=to_state,
        context=context,
        metadata=metadata
    )


async def emit_trader_status(
    event_bus: EventBus,
    state: str,
    metrics: Dict[str, Any],
    health: str = "healthy",
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Emit a trader status event.
    
    Args:
        event_bus: EventBus instance to use
        state: Current trader state
        metrics: Performance metrics
        health: Health status
        metadata: Optional additional data
        
    Returns:
        True if event was queued successfully
    """
    return await event_bus.emit_trader_status(
        state=state,
        metrics=metrics,
        health=health,
        metadata=metadata
    )