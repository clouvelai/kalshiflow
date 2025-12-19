"""
Event Bus for Kalshi Trading Actor MVP.

Provides a clean event-driven architecture to break circular dependencies
between OrderbookClient and ActorService. Uses async callbacks and proper
error isolation to ensure WebSocket performance is never impacted.
"""

import asyncio
import logging
import time
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.event_bus")


class EventType(Enum):
    """Event types for the trading system."""
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_DELTA = "orderbook_delta"


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


class EventBus:
    """
    Simple async event bus for breaking circular dependencies.
    
    Features:
    - Non-blocking event emission (never blocks publisher)
    - Async callback execution with error isolation
    - Subscription management with cleanup
    - Performance monitoring and circuit breaker
    - Global singleton pattern for clean architecture
    """
    
    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._event_queue: asyncio.Queue[MarketEvent] = asyncio.Queue(maxsize=1000)
        self._processing_task: Optional[asyncio.Task] = None
        self._running = False
        self._shutdown_requested = False
        
        # Performance monitoring
        self._events_emitted = 0
        self._events_processed = 0
        self._callback_errors = 0
        self._last_error: Optional[str] = None
        self._started_at: Optional[float] = None
        
        logger.info("EventBus initialized")
    
    async def start(self) -> None:
        """Start the event bus processing loop."""
        if self._running:
            logger.warning("EventBus is already running")
            return
        
        logger.info("Starting EventBus...")
        self._running = True
        self._shutdown_requested = False
        self._started_at = time.time()
        
        # Start async processing loop
        self._processing_task = asyncio.create_task(self._process_events())
        
        logger.info("✅ EventBus started")
    
    async def stop(self) -> None:
        """Stop the event bus and cleanup."""
        if not self._running:
            return
        
        logger.info("Stopping EventBus...")
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
        
        logger.info(f"✅ EventBus stopped. Events emitted: {self._events_emitted}, processed: {self._events_processed}")
    
    def subscribe(self, event_type: EventType, callback: Callable[[MarketEvent], None]) -> None:
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
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[MarketEvent], None]) -> None:
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
    
    async def emit(
        self,
        event_type: EventType,
        market_ticker: str,
        sequence_number: int,
        timestamp_ms: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Emit an event (non-blocking).
        
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
        
        try:
            # Non-blocking queue put
            self._event_queue.put_nowait(event)
            self._events_emitted += 1
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping {event_type.value} event for {market_ticker}")
            return False
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
            return False
    
    async def _process_events(self) -> None:
        """Main event processing loop (runs in background)."""
        logger.info("Event processing loop started")
        
        while not self._shutdown_requested:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
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
        
        logger.info("Event processing loop stopped")
    
    async def _notify_subscribers(self, event: MarketEvent) -> None:
        """
        Notify all subscribers of an event with error isolation.
        
        Args:
            event: Event to send to subscribers
        """
        subscribers = self._subscribers.get(event.event_type, [])
        
        if not subscribers:
            return
        
        # Call all subscribers concurrently with error isolation
        tasks = []
        for callback in subscribers:
            task = asyncio.create_task(self._safe_call_subscriber(callback, event))
            tasks.append(task)
        
        # Wait for all callbacks to complete (with timeout)
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Subscriber callbacks timeout for {event.event_type.value}")
    
    async def _safe_call_subscriber(self, callback: Callable, event: MarketEvent) -> None:
        """
        Safely call a subscriber callback with error isolation.
        
        Args:
            callback: Subscriber callback to call
            event: Event data to pass
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)
                
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
        Get detailed health information for initialization tracker.
        
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
    
    def get_last_sync_time(self) -> Optional[float]:
        """
        Get last event processed time.
        
        Returns:
            Timestamp when event bus started (as proxy for last activity),
            or None if not started
        """
        return self._started_at


# Global event bus instance (singleton pattern)
_event_bus: Optional[EventBus] = None
_bus_lock = asyncio.Lock()


async def get_event_bus() -> EventBus:
    """
    Get the global event bus instance.
    
    Returns:
        EventBus instance (creates if not exists)
    """
    global _event_bus
    
    async with _bus_lock:
        if _event_bus is None:
            _event_bus = EventBus()
            await _event_bus.start()
            logger.info("Global EventBus created and started")
        
        return _event_bus


async def shutdown_event_bus() -> None:
    """Shutdown the global event bus."""
    global _event_bus
    
    async with _bus_lock:
        if _event_bus:
            await _event_bus.stop()
            _event_bus = None
            logger.info("Global EventBus shutdown complete")


# Convenience functions for common events

async def emit_orderbook_snapshot(
    market_ticker: str,
    sequence_number: int,
    timestamp_ms: int,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Emit an orderbook snapshot event.
    
    Args:
        market_ticker: Market that had snapshot
        sequence_number: Sequence number
        timestamp_ms: Timestamp
        metadata: Optional additional data
        
    Returns:
        True if event was queued successfully
    """
    bus = await get_event_bus()
    return await bus.emit(
        event_type=EventType.ORDERBOOK_SNAPSHOT,
        market_ticker=market_ticker,
        sequence_number=sequence_number,
        timestamp_ms=timestamp_ms,
        metadata=metadata
    )


async def emit_orderbook_delta(
    market_ticker: str,
    sequence_number: int,
    timestamp_ms: int,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Emit an orderbook delta event.
    
    Args:
        market_ticker: Market that had delta
        sequence_number: Sequence number
        timestamp_ms: Timestamp
        metadata: Optional additional data
        
    Returns:
        True if event was queued successfully
    """
    bus = await get_event_bus()
    return await bus.emit(
        event_type=EventType.ORDERBOOK_DELTA,
        market_ticker=market_ticker,
        sequence_number=sequence_number,
        timestamp_ms=timestamp_ms,
        metadata=metadata
    )