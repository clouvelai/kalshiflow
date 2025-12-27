"""
TRADER V3 Trades Integration.

Integration layer between V3 and the TradesClient for public trades monitoring.
Provides clean abstraction for trade data flow and whale detection.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .trades_client import TradesClient
from ..core.event_bus import EventBus, EventType

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.trades_integration")


@dataclass
class TradesMetrics:
    """Metrics for trades data flow."""
    trades_received: int = 0
    errors: int = 0
    last_trade_time: Optional[float] = None


class V3TradesIntegration:
    """
    Integration layer for public trades data in TRADER V3.

    Features:
    - Wrapper around TradesClient for public trades stream
    - Event bus integration for trade data distribution
    - Metrics tracking for monitoring
    - Clean start/stop lifecycle
    - Emits PUBLIC_TRADE_RECEIVED events for whale detection
    """

    def __init__(
        self,
        trades_client: TradesClient,
        event_bus: EventBus,
    ):
        """
        Initialize trades integration.

        Args:
            trades_client: TradesClient instance for public trades
            event_bus: V3 Event bus for broadcasting trade events
        """
        self._client = trades_client
        self._event_bus = event_bus

        self._metrics = TradesMetrics()
        self._running = False
        self._started_at: Optional[float] = None
        self._connection_established = False
        self._first_trade_received = False
        self._connection_established_time: Optional[float] = None
        self._first_trade_time: Optional[float] = None

        # Track client task for proper cleanup
        self._client_task: Optional[asyncio.Task] = None

        logger.info("V3 Trades Integration initialized")

    async def _handle_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Handle incoming trade from TradesClient.

        Emits PUBLIC_TRADE_RECEIVED event to EventBus for downstream
        processing (whale detection, etc.)

        Args:
            trade_data: Normalized trade data from TradesClient
        """
        if not self._running:
            return

        try:
            # Update metrics
            self._metrics.trades_received += 1
            self._metrics.last_trade_time = time.time()

            if not self._first_trade_received:
                self._first_trade_received = True
                self._first_trade_time = time.time()
                logger.info(f"First trade received: {trade_data.get('market_ticker')}")

            # Emit PUBLIC_TRADE_RECEIVED event
            await self._event_bus.emit_public_trade(trade_data)

            # Log periodically
            if self._metrics.trades_received % 500 == 0:
                logger.debug(
                    f"V3 Trades Integration: {self._metrics.trades_received} trades processed"
                )

        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Error handling trade: {e}", exc_info=True)

    async def start(self) -> None:
        """Start trades integration."""
        if self._running:
            logger.warning("Trades integration is already running")
            return

        logger.info("Starting V3 trades integration")
        self._running = True
        self._started_at = time.time()

        # Set the callback on the client
        self._client.on_trade_callback = self._handle_trade

        # Start the trades client in background task
        self._client_task = asyncio.create_task(self._client.start())

        logger.info("V3 Trades Integration started, waiting for connection...")

    async def stop(self) -> None:
        """Stop trades integration."""
        if not self._running:
            return

        logger.info("Stopping V3 trades integration...")
        self._running = False

        # Cancel the client task before stopping the client
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
            try:
                await self._client_task
            except asyncio.CancelledError:
                pass

        # Stop the trades client
        await self._client.stop()

        logger.info(f"V3 Trades Integration stopped - Metrics: {self.get_metrics()}")

    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for trades client to establish connection.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if connected, False if timeout
        """
        logger.info(f"Waiting up to {timeout}s for trades connection...")

        connected = await self._client.wait_for_connection(timeout=timeout)
        if connected:
            logger.info("Trades client connected to WebSocket")
            self._connection_established = True
            self._connection_established_time = time.time()
            return True
        else:
            logger.error("Trades client connection timeout!")
            return False

    async def wait_for_first_trade(self, timeout: float = 30.0) -> bool:
        """
        Wait for first trade to confirm data flow.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if trade received, False if timeout
        """
        logger.info(f"Waiting up to {timeout}s for first trade...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._first_trade_received:
                logger.info(
                    f"First trade received after {time.time() - start_time:.1f}s. "
                    f"Total trades: {self._metrics.trades_received}"
                )
                return True
            await asyncio.sleep(0.1)

        logger.warning(f"No trade received within {timeout}s")
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get trades integration metrics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "trades_received": self._metrics.trades_received,
            "errors": self._metrics.errors,
            "last_trade_time": self._metrics.last_trade_time,
            "uptime_seconds": uptime,
            "trades_per_second": self._metrics.trades_received / max(uptime, 1),
        }

    def is_healthy(self) -> bool:
        """Check if trades integration is healthy."""
        if not self._running:
            return False

        # During initial startup (first 30s), be lenient
        if self._started_at and (time.time() - self._started_at) < 30.0:
            return True

        # If we've received trades, trust our tracking
        if self._first_trade_received:
            return True

        # Check client health as fallback during startup
        if self._client and not self._client.is_healthy():
            now = time.time()
            if not hasattr(self, '_last_client_warning') or (now - self._last_client_warning) > 30.0:
                logger.warning("Trades client connection is unhealthy during startup")
                self._last_client_warning = now
            return False

        return True

    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        metrics = self.get_metrics()
        now = time.time()

        time_since_trade = None
        if self._metrics.last_trade_time:
            time_since_trade = now - self._metrics.last_trade_time

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
            "trades_received": metrics["trades_received"],
            "errors": metrics["errors"],
            "time_since_trade": time_since_trade,
            "uptime_seconds": metrics["uptime_seconds"],
            "client_connected": self._client.is_healthy() if self._client else False,
            "ping_health": ping_health,
            "last_ping_age_seconds": last_ping_age,
            "connection_established": time.strftime(
                "%H:%M:%S", time.localtime(self._connection_established_time)
            ) if self._connection_established_time else None,
            "first_trade_received": time.strftime(
                "%H:%M:%S", time.localtime(self._first_trade_time)
            ) if self._first_trade_time else None,
        }
