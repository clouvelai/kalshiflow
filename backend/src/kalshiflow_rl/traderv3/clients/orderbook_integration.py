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
from ..core.event_bus import EventBus, EventType, MarketEvent

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
        market_tickers: List[str]
    ):
        """
        Initialize orderbook integration.
        
        Args:
            orderbook_client: Existing orderbook client instance
            event_bus: Event bus for broadcasting updates
            market_tickers: List of market tickers to subscribe to
        """
        self._client = orderbook_client
        self._event_bus = event_bus  # V3's internal event bus
        self._market_tickers = market_tickers
        
        self._metrics = OrderbookMetrics()
        self._running = False
        self._started_at: Optional[float] = None
        self._connection_established = False
        self._first_snapshot_received = False
        self._connection_established_time: Optional[float] = None
        self._first_snapshot_time: Optional[float] = None
        
        # Subscribe to V3 event bus for tracking snapshots
        self._snapshot_subscription = None
        
        logger.info(f"V3 Orderbook Integration initialized for {len(market_tickers)} markets")
    
    async def _handle_snapshot_event(self, market_ticker: str, metadata: Dict[str, Any]) -> None:
        """Handle snapshot events from V3 event bus."""
        if not self._running:
            return
        
        # Update metrics
        self._metrics.snapshots_received += 1
        self._metrics.last_snapshot_time = time.time()
        if not self._first_snapshot_received:
            self._first_snapshot_received = True
            self._first_snapshot_time = time.time()
        # Track market as connected only when we receive its first snapshot
        # This ensures accurate connection state tracking
        self._metrics.markets_connected.add(market_ticker)
        
        # Log periodically
        if self._metrics.snapshots_received % 100 == 0:
            logger.debug(f"V3 received {self._metrics.snapshots_received} snapshots, {self._metrics.deltas_received} deltas")
    
    async def _handle_delta_event(self, market_ticker: str, metadata: Dict[str, Any]) -> None:
        """Handle delta events from V3 event bus."""
        if not self._running:
            return
        
        # Update metrics
        self._metrics.deltas_received += 1
        self._metrics.last_delta_time = time.time()
    
    async def start(self) -> None:
        """Start orderbook integration."""
        if self._running:
            logger.warning("Orderbook integration is already running")
            return
        
        logger.info(f"Starting V3 orderbook integration for {len(self._market_tickers)} markets")
        self._running = True
        self._started_at = time.time()
        
        # Subscribe to V3 event bus to track snapshots and deltas
        # OrderbookClient now publishes directly to V3 event bus
        await self._event_bus.subscribe_to_orderbook_snapshot(self._handle_snapshot_event)
        await self._event_bus.subscribe_to_orderbook_delta(self._handle_delta_event)
        logger.info("✅ Subscribed to V3 event bus for orderbook events")
        
        # Start the orderbook client (it will publish to V3 event bus directly)
        asyncio.create_task(self._client.start())
        
        # Don't immediately mark markets as connected - wait for actual connection
        logger.info(f"Waiting for orderbook client connection...")
    
    async def stop(self) -> None:
        """Stop orderbook integration."""
        if not self._running:
            return
        
        logger.info("Stopping V3 orderbook integration...")
        self._running = False
        
        # No need to unsubscribe - V3 event bus manages its own subscriptions
        
        # Stop the orderbook client
        await self._client.stop()
        
        self._metrics.markets_connected.clear()
        
        logger.info(f"✅ V3 Orderbook Integration stopped - Metrics: {self.get_metrics()}")
    
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
            # Check both our metrics and the client's snapshot tracking
            if self._first_snapshot_received or self._client.has_received_snapshots():
                snapshot_markets = self._client.get_snapshot_received_markets()
                logger.info(
                    f"✅ First snapshot received after {time.time() - start_time:.1f}s. "
                    f"Markets with snapshots: {', '.join(snapshot_markets)}"
                )
                if not self._first_snapshot_received:
                    self._first_snapshot_received = True
                    self._first_snapshot_time = time.time()
                return True
            await asyncio.sleep(0.1)
        
        logger.warning(f"⚠️ No snapshot received within {timeout}s")
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orderbook integration metrics."""
        uptime = time.time() - self._started_at if self._started_at else 0
        
        return {
            "running": self._running,
            "markets_connected": len(self._metrics.markets_connected),
            "markets_list": list(self._metrics.markets_connected),
            "snapshots_received": self._metrics.snapshots_received,
            "deltas_received": self._metrics.deltas_received,
            "errors": self._metrics.errors,
            "last_snapshot_time": self._metrics.last_snapshot_time,
            "last_delta_time": self._metrics.last_delta_time,
            "uptime_seconds": uptime,
            "snapshots_per_second": self._metrics.snapshots_received / max(uptime, 1),
            "deltas_per_second": self._metrics.deltas_received / max(uptime, 1)
        }
    
    def is_healthy(self) -> bool:
        """Check if orderbook integration is healthy."""
        if not self._running:
            return False
        
        # Check if we have active market connections (received at least one snapshot)
        if not self._metrics.markets_connected:
            # During initial startup (first 30s), be lenient
            if self._started_at and (time.time() - self._started_at) < 30.0:
                return True  # Still starting up
            return False
        
        # If we've received snapshots and the underlying client is healthy, we're good
        # Markets might be quiet with no deltas - that's normal for low-volume markets
        if self._first_snapshot_received and self._client and self._client.is_healthy():
            return True
        
        # Fallback: Check if we're still in the startup phase
        now = time.time()
        if self._started_at and (now - self._started_at) > 30.0:
            # If we've been running for 30s and never received any snapshots, that's unhealthy
            if not self._first_snapshot_received:
                logger.warning(f"No orderbook snapshots received after {now - self._started_at:.1f}s - marking unhealthy")
                return False
        
        # If the client connection is broken, we're unhealthy
        if self._client and not self._client.is_healthy():
            logger.warning("Orderbook client connection is unhealthy")
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
        
        return {
            "healthy": self.is_healthy(),
            "running": self._running,
            "markets_connected": metrics["markets_connected"],
            "snapshots_received": metrics["snapshots_received"],
            "deltas_received": metrics["deltas_received"],
            "errors": metrics["errors"],
            "time_since_snapshot": time_since_snapshot,
            "time_since_delta": time_since_delta,
            "uptime_seconds": metrics["uptime_seconds"],
            "client_connected": self._client.is_healthy() if self._client else False,
            # Return timestamp strings for display in console
            "connection_established": time.strftime("%H:%M:%S", time.localtime(self._connection_established_time)) if self._connection_established_time else None,
            "first_snapshot_received": time.strftime("%H:%M:%S", time.localtime(self._first_snapshot_time)) if self._first_snapshot_time else None
        }