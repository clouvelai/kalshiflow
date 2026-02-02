"""
TRADER V3 Trades Integration.

Integration layer between V3 and the TradesClient for public trades monitoring.
Provides clean abstraction for trade data flow and whale detection.

Purpose:
    Bridges the Kalshi public trades WebSocket stream into the V3 event bus,
    emitting PUBLIC_TRADE_RECEIVED events for downstream consumers.

Key Responsibilities:
    1. **Trade Ingestion** - Receives all public trades from Kalshi WebSocket
    2. **Market Filtering** - Pre-filters trades to tracked markets BEFORE
       they enter the event bus queue, preventing queue saturation
    3. **Event Emission** - Emits PUBLIC_TRADE_RECEIVED for tracked-market trades
    4. **Metrics Tracking** - Tracks received, emitted, and filtered counts

Architecture Position:
    TradesClient (WebSocket) -> V3TradesIntegration (filter) -> EventBus -> TradeFlowService

Design Principles:
    - **Pre-filter at source**: Only tracked-market trades enter the event bus.
      The public trades stream covers hundreds of markets but the deep agent
      and TradeFlowService only care about 10-20 tracked markets. Filtering
      here prevents ~76% of event bus drops that previously saturated the queue.
    - **Set-based lookup**: O(1) market membership check via TrackedMarketsState
    - **Dynamic tracking**: The tracked markets set changes as lifecycle
      discovery adds/removes markets; filtering adapts automatically.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from .trades_client import TradesClient
from ..core.event_bus import EventBus, EventType

if TYPE_CHECKING:
    from ..state.tracked_markets import TrackedMarketsState

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.trades_integration")


@dataclass
class TradesMetrics:
    """Metrics for trades data flow."""
    trades_received: int = 0
    trades_emitted: int = 0
    trades_filtered: int = 0
    errors: int = 0
    last_trade_time: Optional[float] = None


class V3TradesIntegration:
    """
    Integration layer for public trades data in TRADER V3.

    Features:
    - Wrapper around TradesClient for public trades stream
    - Event bus integration for trade data distribution
    - Pre-filters trades to tracked markets before event bus emission
    - Metrics tracking for monitoring
    - Clean start/stop lifecycle
    - Emits PUBLIC_TRADE_RECEIVED events for tracked markets only
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

        # Market filter: when set, only trades for tracked markets are emitted.
        # Injected after construction via set_tracked_markets() because
        # TrackedMarketsState is created in _connect_lifecycle() which runs
        # before _connect_trades() in the coordinator startup sequence.
        self._tracked_markets: Optional['TrackedMarketsState'] = None

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

    def set_tracked_markets(self, tracked_markets: 'TrackedMarketsState') -> None:
        """
        Inject TrackedMarketsState for pre-filtering trades.

        When set, only trades for markets present in TrackedMarketsState
        are emitted to the event bus. This prevents queue saturation from
        the hundreds of untracked markets on the public trades stream.

        Args:
            tracked_markets: TrackedMarketsState instance for market lookup
        """
        self._tracked_markets = tracked_markets
        logger.info("TrackedMarketsState injected - trades will be pre-filtered to tracked markets")

    async def _handle_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Handle incoming trade from TradesClient.

        Pre-filters to tracked markets, then emits PUBLIC_TRADE_RECEIVED
        event to EventBus for downstream processing (TradeFlowService, etc.)

        Filtering rationale:
            The public trades stream covers ALL Kalshi markets (180+), but
            downstream consumers (TradeFlowService, deep agent) only care about
            the 10-20 tracked markets. Pre-filtering here prevents ~76% of
            event bus queue saturation that previously caused ~1,773 drops/min.

        Args:
            trade_data: Normalized trade data from TradesClient
        """
        if not self._running:
            return

        try:
            # Update total received counter (all markets, before filtering)
            self._metrics.trades_received += 1
            self._metrics.last_trade_time = time.time()

            if not self._first_trade_received:
                self._first_trade_received = True
                self._first_trade_time = time.time()
                logger.info(f"First trade received: {trade_data.get('market_ticker')}")

            # Pre-filter: skip trades for markets we are not tracking.
            # This is an O(1) dict lookup on TrackedMarketsState._markets.
            if self._tracked_markets is not None:
                market_ticker = trade_data.get("market_ticker", "")
                if not self._tracked_markets.is_tracked(market_ticker):
                    self._metrics.trades_filtered += 1
                    return

            # Emit PUBLIC_TRADE_RECEIVED event (only for tracked markets)
            await self._event_bus.emit_public_trade(trade_data)
            self._metrics.trades_emitted += 1

            # Log periodically with filter stats
            if self._metrics.trades_emitted % 500 == 0:
                filter_pct = (
                    (self._metrics.trades_filtered / max(self._metrics.trades_received, 1)) * 100
                )
                logger.info(
                    f"Trades: {self._metrics.trades_emitted} emitted, "
                    f"{self._metrics.trades_filtered} filtered ({filter_pct:.0f}% pre-filtered), "
                    f"{self._metrics.trades_received} total received"
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
        filter_pct = (
            (self._metrics.trades_filtered / max(self._metrics.trades_received, 1)) * 100
        )

        return {
            "running": self._running,
            "trades_received": self._metrics.trades_received,
            "trades_emitted": self._metrics.trades_emitted,
            "trades_filtered": self._metrics.trades_filtered,
            "filter_percentage": round(filter_pct, 1),
            "errors": self._metrics.errors,
            "last_trade_time": self._metrics.last_trade_time,
            "uptime_seconds": uptime,
            "trades_per_second": self._metrics.trades_received / max(uptime, 1),
            "emitted_per_second": self._metrics.trades_emitted / max(uptime, 1),
            "market_filter_active": self._tracked_markets is not None,
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
            "trades_emitted": metrics["trades_emitted"],
            "trades_filtered": metrics["trades_filtered"],
            "filter_percentage": metrics["filter_percentage"],
            "market_filter_active": metrics["market_filter_active"],
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
