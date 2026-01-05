"""
TRADER V3 Orderbook Integration.

Simple integration layer between V3 and the existing orderbook client.
Provides clean abstraction for orderbook subscription and data flow.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from ...data.orderbook_client import OrderbookClient
from ...data.orderbook_state import get_shared_orderbook_state
from ...data.orderbook_signals import OrderbookSignalAggregator
from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.orderbook_integration")


@dataclass
class OrderbookMetrics:
    """Metrics for orderbook data flow."""
    snapshots_received: int = 0
    deltas_received: int = 0
    errors: int = 0
    last_snapshot_time: Optional[float] = None
    last_delta_time: Optional[float] = None
    markets_connected: Set[str] = None
    
    def __post_init__(self):
        if self.markets_connected is None:
            self.markets_connected = set()


class V3OrderbookIntegration:
    """
    Integration layer for orderbook data in TRADER V3.
    
    Features:
    - Simple wrapper around existing OrderbookClient
    - Event bus integration for data flow
    - Metrics tracking for monitoring
    - Clean start/stop lifecycle
    """
    
    def __init__(
        self,
        orderbook_client: OrderbookClient,
        event_bus: EventBus,
        market_tickers: List[str],
        signal_aggregator: Optional[OrderbookSignalAggregator] = None,
        enable_signal_aggregation: bool = True,
    ):
        """
        Initialize orderbook integration.

        Args:
            orderbook_client: Existing orderbook client instance
            event_bus: Event bus for broadcasting updates
            market_tickers: List of market tickers to subscribe to
            signal_aggregator: Optional pre-configured signal aggregator
            enable_signal_aggregation: If True (default), auto-creates aggregator on first snapshot
        """
        self._client = orderbook_client
        self._event_bus = event_bus  # V3's internal event bus
        self._market_tickers = list(market_tickers)  # Copy to avoid shared reference with OrderbookClient
        self._signal_aggregator = signal_aggregator
        self._enable_signal_aggregation = enable_signal_aggregation
        self._aggregator_initialized = signal_aggregator is not None

        self._metrics = OrderbookMetrics()
        self._running = False
        self._started_at: Optional[float] = None
        self._connection_established = False
        self._first_snapshot_received = False
        self._connection_established_time: Optional[float] = None
        self._first_snapshot_time: Optional[float] = None

        # Track subscription callbacks for proper cleanup
        self._snapshot_callback = None
        self._delta_callback = None

        # Track client task for proper cleanup
        self._client_task: Optional[asyncio.Task] = None

        # Trading client for REST fallback (injected after construction)
        self._trading_client: Optional[Any] = None

        # Rate limiting for REST refreshes (once per minute per market)
        self._last_rest_refresh: Dict[str, float] = {}

        aggregation_status = "enabled (lazy init)" if enable_signal_aggregation else "disabled"
        if signal_aggregator:
            aggregation_status = "enabled (pre-configured)"
        logger.info(f"V3 Orderbook Integration initialized for {len(market_tickers)} markets "
                    f"(signal aggregation: {aggregation_status})")
    
    
    async def _handle_snapshot_event(self, market_ticker: str, metadata: Dict[str, Any]) -> None:
        """Handle snapshot events from V3 event bus."""
        if not self._running:
            return

        # Check if this is a reconnection (snapshot for a market we already had)
        # If all markets are already connected and we get another snapshot, it's a reconnection
        if len(self._metrics.markets_connected) >= len(self._market_tickers) and market_ticker in self._metrics.markets_connected:
            # This is a reconnection - reset metrics
            logger.info(f"Reconnection detected for {market_ticker} - resetting metrics")
            self._metrics.snapshots_received = 0
            self._metrics.deltas_received = 0
            self._metrics.markets_connected.clear()
            self._connection_established = False
            self._first_snapshot_received = False
            self._connection_established_time = None
            self._first_snapshot_time = None

        # Update local metrics
        self._metrics.snapshots_received += 1
        self._metrics.last_snapshot_time = time.time()

        if not self._first_snapshot_received:
            self._first_snapshot_received = True
            self._first_snapshot_time = time.time()
        # Track market as connected only when we receive its first snapshot
        # This ensures accurate connection state tracking
        self._metrics.markets_connected.add(market_ticker)

        # Lazy initialization of signal aggregator on first snapshot
        # At this point we know the client has a session_id
        if self._enable_signal_aggregation and not self._aggregator_initialized:
            await self._initialize_signal_aggregator()

        # Record snapshot for signal aggregation (10-second buckets)
        if self._signal_aggregator:
            try:
                # Get the current orderbook state snapshot
                orderbook_state = await get_shared_orderbook_state(market_ticker)
                snapshot = await orderbook_state.get_snapshot()
                self._signal_aggregator.record_snapshot(market_ticker, snapshot)
            except Exception as e:
                # Don't let aggregation errors affect main processing
                logger.debug(f"Signal aggregation error for {market_ticker}: {e}")

        # Log periodically
        if self._metrics.snapshots_received % 100 == 0:
            logger.debug(f"V3 received {self._metrics.snapshots_received} snapshots, {self._metrics.deltas_received} deltas")
    
    async def _handle_delta_event(self, market_ticker: str, metadata: Dict[str, Any]) -> None:
        """Handle delta events from V3 event bus."""
        if not self._running:
            return

        # Update local metrics
        self._metrics.deltas_received += 1
        self._metrics.last_delta_time = time.time()

        # Record delta for signal aggregation (activity counting)
        if self._signal_aggregator:
            self._signal_aggregator.record_delta(market_ticker)

    async def _initialize_signal_aggregator(self) -> None:
        """
        Lazy initialization of signal aggregator on first snapshot.

        Called when signal aggregation is enabled but no aggregator was provided.
        Creates an aggregator using the session_id from the orderbook client.
        """
        if self._aggregator_initialized:
            return

        try:
            # Get session_id from the orderbook client
            client_stats = self._client.get_stats()
            session_id = client_stats.get("session_id")

            if not session_id:
                logger.warning("Cannot initialize signal aggregator: no session_id available yet")
                return

            # Create the signal aggregator
            self._signal_aggregator = OrderbookSignalAggregator(
                session_id=session_id,
                bucket_seconds=10,
                flush_interval=10.0,
            )
            await self._signal_aggregator.start()
            self._aggregator_initialized = True

            logger.info(f"✅ Signal aggregator initialized with session_id={session_id}")

        except Exception as e:
            logger.error(f"Failed to initialize signal aggregator: {e}")
            # Don't retry - mark as initialized to prevent repeated failures
            self._aggregator_initialized = True

    async def start(self) -> None:
        """Start orderbook integration."""
        if self._running:
            logger.warning("Orderbook integration is already running")
            return

        logger.info(f"Starting V3 orderbook integration for {len(self._market_tickers)} markets")
        self._running = True
        self._started_at = time.time()

        # Start signal aggregator if configured
        if self._signal_aggregator:
            await self._signal_aggregator.start()
            logger.info("✅ Signal aggregator started")

        # Subscribe to V3 event bus to track snapshots and deltas
        # OrderbookClient now publishes directly to V3 event bus
        # Store callbacks for proper cleanup
        self._snapshot_callback = self._handle_snapshot_event
        self._delta_callback = self._handle_delta_event
        await self._event_bus.subscribe_to_orderbook_snapshot(self._snapshot_callback)
        await self._event_bus.subscribe_to_orderbook_delta(self._delta_callback)
        logger.info("✅ Subscribed to V3 event bus for orderbook events")

        # Start the orderbook client (it will publish to V3 event bus directly)
        # Track the task for proper cleanup
        self._client_task = asyncio.create_task(self._client.start())

        # Don't immediately mark markets as connected - wait for actual connection
        logger.info(f"Waiting for orderbook client connection...")

    async def stop(self) -> None:
        """Stop orderbook integration."""
        if not self._running:
            return

        logger.info("Stopping V3 orderbook integration...")
        self._running = False

        # Stop signal aggregator (flushes remaining buckets)
        if self._signal_aggregator:
            await self._signal_aggregator.stop()
            logger.info("✅ Signal aggregator stopped")

        # Note: Event bus clears all subscribers on stop, so explicit unsubscribe is not needed

        # Cancel the client task before stopping the client
        if self._client_task and not self._client_task.done():
            self._client_task.cancel()
            try:
                await self._client_task
            except asyncio.CancelledError:
                pass

        # Stop the orderbook client
        await self._client.stop()

        self._metrics.markets_connected.clear()

        logger.info(f"✅ V3 Orderbook Integration stopped - Metrics: {self.get_metrics()}")
    
    async def ensure_session_for_recovery(self) -> bool:
        """
        Ensure session is ready for recovery scenarios.
        
        This is called by the coordinator when recovering from ERROR state
        to ensure the session lifecycle is properly managed.
        
        Returns:
            True if session is ready, False if session creation failed
        """
        if not self._running:
            logger.warning("Cannot ensure session - integration not running")
            return False
        
        if hasattr(self._client, '_ensure_session_for_recovery'):
            logger.info("Ensuring session is ready for recovery")
            return await self._client._ensure_session_for_recovery()
        else:
            logger.debug("Client does not support session recovery - assuming ready")
            return True
    
    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for orderbook client to establish connection.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected, False if timeout
        """
        logger.info(f"Waiting up to {timeout}s for orderbook connection...")
        
        # Wait for the orderbook client to connect
        connected = await self._client.wait_for_connection(timeout=timeout)
        if connected:
            logger.info("✅ Orderbook client connected to WebSocket")
            self._connection_established = True
            self._connection_established_time = time.time()
            
            # Note: Markets are tracked as connected only when we receive their snapshots
            # This ensures accurate connection state (handled in _handle_snapshot_event)
            
            return True
        else:
            logger.error("❌ Orderbook client connection timeout!")
            return False
    
    async def wait_for_first_snapshot(self, timeout: float = 10.0) -> bool:
        """
        Wait for first orderbook snapshot to confirm data flow.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if snapshot received, False if timeout
        """
        logger.info(f"Waiting up to {timeout}s for first orderbook snapshot...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Wait for actual metrics update via event bus, not just client's report
            # This ensures markets_connected is properly populated before we proceed
            if self._metrics.markets_connected:
                # We have actual market connections tracked in our metrics
                snapshot_markets = self._client.get_snapshot_received_markets()
                logger.info(
                    f"✅ First snapshot received after {time.time() - start_time:.1f}s. "
                    f"Markets connected: {len(self._metrics.markets_connected)} "
                    f"({', '.join(list(self._metrics.markets_connected)[:5])}{'...' if len(self._metrics.markets_connected) > 5 else ''})"
                )
                if not self._first_snapshot_received:
                    self._first_snapshot_received = True
                    self._first_snapshot_time = time.time()
                return True
            # Also check client tracking as backup
            elif self._client.has_received_snapshots():
                # Client has snapshots but event bus hasn't processed them yet
                # Give it a bit more time for event bus to catch up
                await asyncio.sleep(0.1)
                continue
            await asyncio.sleep(0.1)
        
        logger.warning(f"⚠️ No snapshot received within {timeout}s")
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orderbook integration metrics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        metrics = {
            "running": self._running,
            "markets_connected": len(self._metrics.markets_connected),
            "markets_list": list(self._metrics.markets_connected),
            # Use local session values
            "snapshots_received": self._metrics.snapshots_received,
            "deltas_received": self._metrics.deltas_received,
            "errors": self._metrics.errors,
            "last_snapshot_time": self._metrics.last_snapshot_time,
            "last_delta_time": self._metrics.last_delta_time,
            "uptime_seconds": uptime,
            "snapshots_per_second": self._metrics.snapshots_received / max(uptime, 1),
            "deltas_per_second": self._metrics.deltas_received / max(uptime, 1),
        }

        # Include signal aggregator stats if available
        if self._signal_aggregator:
            metrics["signal_aggregator"] = self._signal_aggregator.get_stats()

        return metrics
    
    def is_healthy(self) -> bool:
        """Check if orderbook integration is healthy."""
        if not self._running:
            return False
        
        # Check if we have active market connections (received at least one snapshot)
        if not self._metrics.markets_connected:
            # During initial startup (first 20s), be lenient
            if self._started_at and (time.time() - self._started_at) < 20.0:
                return True  # Still starting up
            return False
        
        # If we've received snapshots and have connected markets, trust our own tracking
        # This prevents cascade failures from brief client health flickers
        if self._first_snapshot_received and self._metrics.markets_connected:
            # We have data - we're healthy regardless of client's brief health flickers
            return True
        
        # Fallback: Check if we're still in the startup phase
        now = time.time()
        if self._started_at and (now - self._started_at) > 20.0:
            # If we've been running for 20s and never received any snapshots, that's unhealthy
            if not self._first_snapshot_received:
                # Only log warning every 60 seconds to avoid spam
                if not hasattr(self, '_last_health_warning') or (now - self._last_health_warning) > 60.0:
                    logger.warning(f"No orderbook snapshots received after {now - self._started_at:.1f}s - marking unhealthy")
                    self._last_health_warning = now
                return False
        
        # Only check client health during startup (before we have data)
        # Once we have data, we trust our own tracking
        if not self._first_snapshot_received:
            if self._client and not self._client.is_healthy():
                # Only log warning every 30 seconds to avoid spam
                now = time.time()
                if not hasattr(self, '_last_client_warning') or (now - self._last_client_warning) > 30.0:
                    logger.warning("Orderbook client connection is unhealthy during startup")
                    self._last_client_warning = now
                return False
        
        return True
    
    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        metrics = self.get_metrics()
        now = time.time()
        
        time_since_snapshot = None
        if self._metrics.last_snapshot_time:
            time_since_snapshot = now - self._metrics.last_snapshot_time
        
        time_since_delta = None
        if self._metrics.last_delta_time:
            time_since_delta = now - self._metrics.last_delta_time
        
        # Get message-based health from client
        ping_health = "unknown"
        last_ping_age = None
        
        # Get message time for ping/pong from client
        if self._client:
            client_stats = self._client.get_stats()
            last_ping_age = client_stats.get("last_message_age_seconds")
        
        # Determine health based on message age
        if last_ping_age is not None:
            if last_ping_age < 60:  # Less than 1 minute = healthy
                ping_health = "healthy"
            elif last_ping_age < 300:  # 1-5 minutes = degraded
                ping_health = "degraded"
            else:  # More than 5 minutes = unhealthy
                ping_health = "unhealthy"
        
        # Get signal aggregator stats
        signal_stats = None
        if self._signal_aggregator:
            signal_stats = self._signal_aggregator.get_stats()

        return {
            "healthy": self.is_healthy(),
            "running": self._running,
            # Orderbook subscriptions (actual subscribed markets, not universe)
            "subscribed_markets": self._client.get_subscribed_market_count() if self._client else 0,
            # Legacy fields (kept for backwards compatibility, may be removed later)
            "markets_connected": metrics["markets_connected"],
            "snapshots_received": metrics["snapshots_received"],
            "deltas_received": metrics["deltas_received"],
            "errors": metrics["errors"],
            "time_since_snapshot": time_since_snapshot,
            "time_since_delta": time_since_delta,
            "uptime_seconds": metrics["uptime_seconds"],
            "client_connected": self._client.is_healthy() if self._client else False,
            # Ping/heartbeat health
            "ping_health": ping_health,
            "last_ping_age_seconds": last_ping_age,
            # Signal aggregator stats
            "signal_aggregator": {
                "active_buckets": signal_stats["active_buckets"] if signal_stats else 0,
                "signals_flushed": signal_stats["signals_flushed"] if signal_stats else 0,
                "snapshots_processed": signal_stats["snapshots_processed"] if signal_stats else 0,
            } if signal_stats else None,
            # Return timestamp strings for display in console
            "connection_established": time.strftime("%H:%M:%S", time.localtime(self._connection_established_time)) if self._connection_established_time else None,
            "first_snapshot_received": time.strftime("%H:%M:%S", time.localtime(self._first_snapshot_time)) if self._first_snapshot_time else None
        }

    # ============================================================
    # Dynamic Market Subscription (Event Lifecycle Discovery)
    # ============================================================

    async def subscribe_additional_markets(self, tickers: List[str]) -> bool:
        """
        Subscribe to additional markets on the existing connection.

        Used by Event Lifecycle Discovery mode to dynamically add markets
        when they are discovered via lifecycle events.

        Args:
            tickers: List of market tickers to subscribe to

        Returns:
            True if subscription was successful, False otherwise
        """
        if not self._running:
            logger.warning("Cannot subscribe to additional markets - integration not running")
            return False

        if not tickers:
            return True

        # Filter to only new tickers
        new_tickers = [t for t in tickers if t not in self._market_tickers]
        if not new_tickers:
            logger.debug("All requested tickers already subscribed")
            return True

        try:
            # Use the underlying client method
            success = await self._client.subscribe_additional_markets(new_tickers)

            if success:
                # Update our local tracking
                self._market_tickers.extend(new_tickers)
                logger.info(
                    f"Subscribed to {len(new_tickers)} additional markets. "
                    f"Total: {len(self._market_tickers)}"
                )

            return success

        except Exception as e:
            logger.error(f"Error subscribing to additional markets: {e}", exc_info=True)
            self._metrics.errors += 1
            return False

    async def unsubscribe_markets(self, tickers: List[str]) -> bool:
        """
        Unsubscribe from markets on the existing connection.

        Used by Event Lifecycle Discovery mode to remove markets when
        they are determined (outcome resolved).

        Args:
            tickers: List of market tickers to unsubscribe from

        Returns:
            True if unsubscription was successful, False otherwise
        """
        if not self._running:
            logger.warning("Cannot unsubscribe from markets - integration not running")
            return False

        if not tickers:
            return True

        # Filter to only tickers we're subscribed to
        subscribed_tickers = [t for t in tickers if t in self._market_tickers]
        if not subscribed_tickers:
            logger.debug("None of the requested tickers are currently subscribed")
            return True

        try:
            # Use the underlying client method
            success = await self._client.unsubscribe_markets(subscribed_tickers)

            if success:
                # Update our local tracking
                for ticker in subscribed_tickers:
                    if ticker in self._market_tickers:
                        self._market_tickers.remove(ticker)
                    self._metrics.markets_connected.discard(ticker)

                logger.info(
                    f"Unsubscribed from {len(subscribed_tickers)} markets. "
                    f"Remaining: {len(self._market_tickers)}"
                )

            return success

        except Exception as e:
            logger.error(f"Error unsubscribing from markets: {e}", exc_info=True)
            self._metrics.errors += 1
            return False

    def get_subscribed_market_count(self) -> int:
        """
        Get the current number of subscribed markets.

        Returns:
            Number of markets currently subscribed to
        """
        return len(self._market_tickers)

    def get_subscribed_markets(self) -> List[str]:
        """
        Get the list of currently subscribed markets.

        Returns:
            List of market tickers currently subscribed to
        """
        return list(self._market_tickers)

    async def subscribe_market(self, ticker: str) -> bool:
        """
        Subscribe to a single market (wrapper for lifecycle callbacks).

        Args:
            ticker: Market ticker to subscribe to

        Returns:
            True if subscription was successful, False otherwise
        """
        return await self.subscribe_additional_markets([ticker])

    async def unsubscribe_market(self, ticker: str) -> bool:
        """
        Unsubscribe from a single market (wrapper for lifecycle callbacks).

        Args:
            ticker: Market ticker to unsubscribe from

        Returns:
            True if unsubscription was successful, False otherwise
        """
        return await self.unsubscribe_markets([ticker])

    def get_orderbook_signals(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get current orderbook signals for a market.

        Returns the live 10-second bucket data with imbalance ratio,
        delta count, spread OHLC, and large order count.

        Args:
            market_ticker: Market ticker to get signals for

        Returns:
            Signal data dict or None if aggregator not ready or no data
        """
        if self._signal_aggregator is None:
            return None
        return self._signal_aggregator.get_current_bucket_signals(market_ticker)

    def set_trading_client(self, trading_client: Any) -> None:
        """
        Inject trading client for REST API fallback.

        Called after construction to enable REST orderbook refresh
        when WebSocket data is stale.

        Args:
            trading_client: KalshiDemoTradingClient instance with get_orderbook()
        """
        self._trading_client = trading_client
        logger.info("Trading client injected for REST orderbook fallback")

    def is_connection_healthy(self, degraded_threshold: float = 30.0) -> bool:
        """
        Check if WS connection is healthy (receiving messages).

        Args:
            degraded_threshold: Seconds since last WS message to consider degraded

        Returns:
            True if connection is healthy (recent messages), False if degraded
        """
        client_stats = self._client.get_stats()
        last_message_age = client_stats.get("last_message_age_seconds", 0)
        return last_message_age < degraded_threshold

    async def refresh_orderbook_if_stale(
        self,
        market_ticker: str,
        connection_degraded_threshold: float = 30.0,
        min_refresh_interval: float = 60.0
    ) -> bool:
        """
        Refresh orderbook via REST only if WS connection is degraded.

        Only triggers REST if:
        1. Connection is degraded (no messages from ANY market in >30s)
        2. Haven't REST synced this market in the last 60s

        If WS connection is healthy, returns False (no refresh needed - quiet
        markets with no updates are fine, their data is still accurate).

        Args:
            market_ticker: Market to refresh
            connection_degraded_threshold: Seconds since last WS message to consider degraded
            min_refresh_interval: Minimum seconds between REST refreshes per market

        Returns:
            True if REST refresh happened, False otherwise
        """
        from ...data.orderbook_state import get_shared_orderbook_state

        # Check if we have a trading client for REST fallback
        if self._trading_client is None:
            return False

        # Check connection-level health (not per-market staleness)
        client_stats = self._client.get_stats()
        last_message_age = client_stats.get("last_message_age_seconds", 0)

        # If connection is healthy, don't REST - data is fine even if "stale"
        if last_message_age < connection_degraded_threshold:
            logger.debug(
                f"Skipping REST refresh for {market_ticker}: "
                f"WS connection healthy (last msg {last_message_age:.1f}s ago)"
            )
            return False

        # Check rate limit (once per minute per market)
        current_time = time.time()
        last_refresh = self._last_rest_refresh.get(market_ticker, 0)
        if current_time - last_refresh < min_refresh_interval:
            logger.debug(
                f"Skipping REST refresh for {market_ticker}: "
                f"rate limited ({current_time - last_refresh:.0f}s since last refresh)"
            )
            return False

        # Connection is degraded, rate limit passed - do REST refresh
        try:
            state = await get_shared_orderbook_state(market_ticker)
            if state is None:
                logger.warning(f"No orderbook state for {market_ticker}")
                return False

            logger.info(
                f"WS connection degraded ({last_message_age:.1f}s since last msg), "
                f"REST refreshing {market_ticker}..."
            )

            rest_data = await self._trading_client.get_orderbook(market_ticker, depth=5)
            await state.apply_rest_snapshot(rest_data)

            self._last_rest_refresh[market_ticker] = current_time
            self._metrics.rest_refreshes = getattr(self._metrics, 'rest_refreshes', 0) + 1
            logger.info(f"REST orderbook refresh succeeded for {market_ticker}")
            return True

        except Exception as e:
            self._metrics.rest_refresh_failures = getattr(self._metrics, 'rest_refresh_failures', 0) + 1
            logger.warning(f"REST orderbook refresh failed for {market_ticker}: {e}")
            return False