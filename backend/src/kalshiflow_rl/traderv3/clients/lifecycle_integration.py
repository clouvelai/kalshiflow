"""
TRADER V3 Lifecycle Integration.

Integration layer between V3 and the LifecycleClient for market lifecycle monitoring.
Provides clean abstraction for lifecycle event flow and market discovery.

Purpose:
    V3LifecycleIntegration wraps LifecycleClient and integrates it with the V3
    EventBus, enabling downstream services (EventLifecycleService) to react
    to market creation, determination, and settlement events.

Key Responsibilities:
    1. **Client Wrapper** - Manages LifecycleClient lifecycle (start/stop)
    2. **EventBus Integration** - Emits MARKET_LIFECYCLE_EVENT to V3 EventBus
    3. **Metrics Tracking** - Tracks events received, processed, errors
    4. **Health Monitoring** - Reports health status for coordinator

Architecture Position:
    - Coordinator initializes this integration in lifecycle discovery mode
    - EventLifecycleService subscribes to MARKET_LIFECYCLE_EVENT from EventBus
    - Health reported to HealthMonitor for system state management

Design Principles:
    - **Follows V3TradesIntegration pattern**: Same structure and interface
    - **Non-blocking**: Events emitted to EventBus without waiting
    - **Error Isolation**: Individual event errors don't break the stream
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .lifecycle_client import LifecycleClient
from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.lifecycle_integration")


@dataclass
class LifecycleMetrics:
    """Metrics for lifecycle event flow."""
    events_received: int = 0
    events_processed: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    errors: int = 0
    last_event_time: Optional[float] = None


class V3LifecycleIntegration:
    """
    Integration layer for market lifecycle events in TRADER V3.

    Features:
    - Wrapper around LifecycleClient for lifecycle event stream
    - Event bus integration for lifecycle event distribution
    - Metrics tracking for monitoring
    - Clean start/stop lifecycle
    - Emits MARKET_LIFECYCLE_EVENT events for market discovery
    """

    def __init__(
        self,
        lifecycle_client: LifecycleClient,
        event_bus: EventBus,
    ):
        """
        Initialize lifecycle integration.

        Args:
            lifecycle_client: LifecycleClient instance for lifecycle events
            event_bus: V3 Event bus for broadcasting lifecycle events
        """
        self._client = lifecycle_client
        self._event_bus = event_bus

        self._metrics = LifecycleMetrics()
        self._running = False
        self._started_at: Optional[float] = None
        self._connection_established = False
        self._first_event_received = False
        self._connection_established_time: Optional[float] = None
        self._first_event_time: Optional[float] = None

        # Track client task for proper cleanup
        self._client_task: Optional[asyncio.Task] = None

        logger.info("V3 Lifecycle Integration initialized")

    async def _handle_lifecycle_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle incoming lifecycle event from LifecycleClient.

        Emits MARKET_LIFECYCLE_EVENT to EventBus for downstream
        processing (EventLifecycleService).

        Args:
            event_data: Normalized lifecycle event data from LifecycleClient
        """
        if not self._running:
            return

        try:
            # Update metrics
            self._metrics.events_received += 1
            self._metrics.last_event_time = time.time()

            event_type = event_data.get("event_type", "unknown")
            self._metrics.events_by_type[event_type] = self._metrics.events_by_type.get(event_type, 0) + 1

            if not self._first_event_received:
                self._first_event_received = True
                self._first_event_time = time.time()
                logger.info(
                    f"First lifecycle event received: {event_type} for "
                    f"{event_data.get('market_ticker')}"
                )

            # Emit MARKET_LIFECYCLE_EVENT to EventBus
            await self._event_bus.emit_market_lifecycle(
                event_type=event_type,
                market_ticker=event_data.get("market_ticker", ""),
                payload=event_data,
            )

            self._metrics.events_processed += 1

            # Log periodically
            if self._metrics.events_received % 100 == 0:
                logger.debug(
                    f"V3 Lifecycle Integration: {self._metrics.events_received} events received, "
                    f"by_type={self._metrics.events_by_type}"
                )

        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Error handling lifecycle event: {e}", exc_info=True)

    async def start(self) -> None:
        """Start lifecycle integration."""
        if self._running:
            logger.warning("Lifecycle integration is already running")
            return

        logger.info("Starting V3 lifecycle integration")
        self._running = True
        self._started_at = time.time()

        # Set the callback on the client
        self._client.on_lifecycle_callback = self._handle_lifecycle_event

        # Start the lifecycle client in background task
        self._client_task = asyncio.create_task(self._client.start())

        logger.info("V3 Lifecycle Integration started, waiting for connection...")

    async def stop(self) -> None:
        """Stop lifecycle integration."""
        if not self._running:
            return

        logger.info("Stopping V3 lifecycle integration...")
        self._running = False

        # Cancel the client task before stopping the client
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
            try:
                await self._client_task
            except asyncio.CancelledError:
                pass

        # Stop the lifecycle client
        await self._client.stop()

        logger.info(f"V3 Lifecycle Integration stopped - Metrics: {self.get_metrics()}")

    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for lifecycle client to establish connection.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected, False if timeout
        """
        logger.info(f"Waiting up to {timeout}s for lifecycle connection...")

        connected = await self._client.wait_for_connection(timeout=timeout)
        if connected:
            logger.info("Lifecycle client connected to WebSocket")
            self._connection_established = True
            self._connection_established_time = time.time()
            return True
        else:
            logger.error("Lifecycle client connection timeout!")
            return False

    async def wait_for_first_event(self, timeout: float = 60.0) -> bool:
        """
        Wait for first lifecycle event to confirm data flow.

        Note: Lifecycle events may be sparse (markets aren't created constantly),
        so this timeout should be generous.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if event received, False if timeout
        """
        logger.info(f"Waiting up to {timeout}s for first lifecycle event...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._first_event_received:
                logger.info(
                    f"First lifecycle event received after {time.time() - start_time:.1f}s. "
                    f"Total events: {self._metrics.events_received}"
                )
                return True
            await asyncio.sleep(0.5)

        logger.warning(f"No lifecycle event received within {timeout}s (this may be normal)")
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get lifecycle integration metrics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "events_received": self._metrics.events_received,
            "events_processed": self._metrics.events_processed,
            "events_by_type": dict(self._metrics.events_by_type),
            "errors": self._metrics.errors,
            "last_event_time": self._metrics.last_event_time,
            "uptime_seconds": uptime,
            "events_per_minute": (self._metrics.events_received / max(uptime, 1)) * 60,
        }

    def is_healthy(self) -> bool:
        """Check if lifecycle integration is healthy."""
        if not self._running:
            return False

        # Check if the client task is still running (critical for connection monitoring)
        if self._client_task and self._client_task.done():
            # Task has completed - check if it failed with an exception
            try:
                exc = self._client_task.exception()
                if exc:
                    now = time.time()
                    if not hasattr(self, '_last_task_warning') or (now - self._last_task_warning) > 60.0:
                        logger.error(f"Lifecycle client task failed with exception: {exc}")
                        self._last_task_warning = now
                    return False
            except asyncio.CancelledError:
                pass  # Task was cancelled, which is expected during shutdown
            except asyncio.InvalidStateError:
                pass  # Task not yet done, which is fine

        # During initial startup (first 60s), be lenient
        # Lifecycle events are sparse, so we can't require events quickly
        if self._started_at and (time.time() - self._started_at) < 60.0:
            return True

        # If we've received events, trust our tracking
        if self._first_event_received:
            return True

        # Check client health as fallback
        # Client should be healthy even without events (heartbeats keep it alive)
        if self._client and not self._client.is_healthy():
            now = time.time()
            if not hasattr(self, '_last_client_warning') or (now - self._last_client_warning) > 60.0:
                logger.warning("Lifecycle client connection is unhealthy")
                self._last_client_warning = now
            return False

        # If client is healthy but no events, that's OK for lifecycle
        # (events are sparse - markets aren't created constantly)
        return True

    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        metrics = self.get_metrics()
        now = time.time()

        time_since_event = None
        if self._metrics.last_event_time:
            time_since_event = now - self._metrics.last_event_time

        # Get message-based health from client
        ping_health = "unknown"
        last_ping_age = None

        if self._client:
            client_stats = self._client.get_stats()
            last_ping_age = client_stats.get("last_message_age_seconds")

        # Determine health based on message age
        if last_ping_age is not None:
            if last_ping_age < 60:
                ping_health = "healthy"
            elif last_ping_age < 300:
                ping_health = "degraded"
            else:
                ping_health = "unhealthy"

        return {
            "healthy": self.is_healthy(),
            "running": self._running,
            "events_received": metrics["events_received"],
            "events_processed": metrics["events_processed"],
            "events_by_type": metrics["events_by_type"],
            "errors": metrics["errors"],
            "time_since_event": time_since_event,
            "uptime_seconds": metrics["uptime_seconds"],
            "client_connected": self._client.is_healthy() if self._client else False,
            "ping_health": ping_health,
            "last_ping_age_seconds": last_ping_age,
            "connection_established": time.strftime(
                "%H:%M:%S", time.localtime(self._connection_established_time)
            ) if self._connection_established_time else None,
            "first_event_received": time.strftime(
                "%H:%M:%S", time.localtime(self._first_event_time)
            ) if self._first_event_time else None,
        }
