"""
API Discovery Syncer Service for Event Lifecycle Discovery Mode.

REST API-based market discovery for already-open markets. Bootstraps lifecycle
mode on startup by fetching status="open" markets filtered by RLM categories.

Pattern: Follows UpcomingMarketsSyncer and TrackedMarketsSyncer for consistency.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..services.event_lifecycle_service import EventLifecycleService
    from ..state.tracked_markets import TrackedMarketsState
    from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.services.api_discovery_syncer")


class ApiDiscoverySyncer:
    """
    REST API sync for already-open markets.

    Bootstraps lifecycle mode on startup by discovering markets that are
    already open when the system starts. This fills the gap where lifecycle
    WebSocket events only notify about new market openings.

    - Initial discovery on startup
    - Periodic refresh every N seconds (configurable, default 300)
    - Filters markets by configured categories (same as EventLifecycleService)
    - Respects capacity limits from TrackedMarketsState
    - Uses EventLifecycleService.track_market_from_api_data() for tracking

    This service provides the "already-open" market bootstrap that lifecycle
    WebSocket events cannot provide.
    """

    def __init__(
        self,
        trading_client: 'V3TradingClientIntegration',
        event_lifecycle_service: 'EventLifecycleService',
        tracked_markets_state: 'TrackedMarketsState',
        event_bus: 'EventBus',
        categories: List[str],
        sync_interval: float = 300.0,
        batch_size: int = 200,
        close_min_minutes: int = 10,
    ):
        """
        Initialize API discovery syncer.

        Args:
            trading_client: Trading client for API calls
            event_lifecycle_service: Lifecycle service for tracking markets
            tracked_markets_state: State to check capacity and existing markets
            event_bus: Event bus for system activity events
            categories: Category list to filter markets (same as lifecycle)
            sync_interval: Seconds between periodic syncs (default 300 = 5 min)
            batch_size: Maximum markets to fetch per API call (default 200)
            close_min_minutes: Skip markets closing within N minutes (default 10)
        """
        self._client = trading_client
        self._lifecycle_service = event_lifecycle_service
        self._tracked_state = tracked_markets_state
        self._event_bus = event_bus
        self._categories = categories
        self._sync_interval = sync_interval
        self._batch_size = batch_size
        self._close_min_minutes = close_min_minutes

        # Syncer state
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._started_at: Optional[float] = None

        # Health metrics
        self._last_sync_time: Optional[float] = None
        self._sync_count: int = 0
        self._markets_discovered: int = 0
        self._markets_tracked: int = 0
        self._sync_errors: int = 0
        self._last_error: Optional[str] = None

        logger.info(
            f"ApiDiscoverySyncer initialized "
            f"(sync_interval={sync_interval}s, batch_size={batch_size}, "
            f"categories={categories}, close_min={close_min_minutes}min)"
        )

    async def start(self) -> None:
        """
        Start the API discovery syncer.

        Performs initial discovery immediately, then starts periodic sync loop.
        """
        if self._running:
            logger.warning("ApiDiscoverySyncer already running")
            return

        self._running = True
        self._started_at = time.time()

        # Initial discovery
        logger.info("Starting ApiDiscoverySyncer - performing initial discovery...")
        discovered = await self._discover_open_markets()

        await self._event_bus.emit_system_activity(
            activity_type="api_discovery",
            message=f"API discovery startup: {discovered} new markets tracked",
            metadata={
                "markets_discovered": discovered,
                "categories": self._categories,
                "severity": "info"
            }
        )

        # Start periodic sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info(f"ApiDiscoverySyncer started (discovered {discovered} markets)")

    async def stop(self) -> None:
        """Stop the API discovery syncer."""
        if not self._running:
            return

        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"ApiDiscoverySyncer stopped "
            f"(syncs: {self._sync_count}, total tracked: {self._markets_tracked})"
        )

    async def _sync_loop(self) -> None:
        """Periodic sync every N seconds."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                await self._discover_open_markets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in API discovery sync loop: {e}")
                self._sync_errors += 1
                self._last_error = str(e)

    async def _discover_open_markets(self) -> int:
        """
        Fetch and track open markets via REST API.

        Returns:
            Number of new markets tracked
        """
        try:
            # Check if trading client is connected before attempting sync
            if not self._client.is_healthy():
                logger.debug("Skipping API discovery - trading client not healthy")
                return 0

            # Check remaining capacity
            remaining = self._tracked_state.capacity_remaining()
            if remaining <= 0:
                logger.debug("Skipping API discovery - at capacity")
                return 0

            # Fetch open markets from trading client via events API
            # Uses cursor pagination to fetch up to remaining capacity
            # Sorted by close_time ascending (soonest first)
            markets = await self._client.get_open_markets(
                categories=self._categories,
                max_markets=remaining,
                close_min_minutes=self._close_min_minutes,
            )

            now = time.time()
            self._last_sync_time = now
            self._sync_count += 1
            self._markets_discovered = len(markets)

            # Track new markets (sorted newest first from get_open_markets)
            tracked_count = 0
            for market in markets:
                # Stop if at capacity
                if self._tracked_state.capacity_remaining() <= 0:
                    logger.debug("API discovery stopped - reached capacity")
                    break

                ticker = market.get("ticker", "")
                if not ticker:
                    continue

                # Skip if already tracked
                if self._tracked_state.is_tracked(ticker):
                    continue

                # Track via EventLifecycleService
                success = await self._lifecycle_service.track_market_from_api_data(market)
                if success:
                    tracked_count += 1

            self._markets_tracked += tracked_count

            if tracked_count > 0:
                logger.info(
                    f"API discovery sync complete: {tracked_count} new markets tracked "
                    f"(found {len(markets)}, sync #{self._sync_count})"
                )
            else:
                logger.debug(
                    f"API discovery sync complete: no new markets "
                    f"(found {len(markets)}, sync #{self._sync_count})"
                )

            return tracked_count

        except Exception as e:
            logger.error(f"API discovery sync failed: {e}")
            self._sync_errors += 1
            self._last_error = str(e)
            return 0

    def get_health_details(self) -> Dict[str, Any]:
        """Get health details for status reporting."""
        now = time.time()
        uptime = now - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "last_sync_time": self._last_sync_time,
            "last_sync_age_seconds": now - self._last_sync_time if self._last_sync_time else None,
            "sync_count": self._sync_count,
            "markets_discovered": self._markets_discovered,
            "markets_tracked_total": self._markets_tracked,
            "sync_errors": self._sync_errors,
            "last_error": self._last_error,
            "sync_interval": self._sync_interval,
            "batch_size": self._batch_size,
            "categories": self._categories,
            "healthy": self._running and (
                self._last_sync_time is None or
                (now - self._last_sync_time) < self._sync_interval * 3
            )
        }

    def is_healthy(self) -> bool:
        """Check if syncer is healthy."""
        if not self._running:
            return False

        # Healthy if running and last sync wasn't too long ago
        if self._last_sync_time:
            age = time.time() - self._last_sync_time
            if age > self._sync_interval * 3:  # 3x interval = unhealthy
                return False

        return True

    @property
    def sync_count(self) -> int:
        """Get number of syncs completed."""
        return self._sync_count

    @property
    def markets_tracked(self) -> int:
        """Get total number of markets tracked via API discovery."""
        return self._markets_tracked

    @property
    def last_sync_time(self) -> Optional[float]:
        """Get timestamp of last successful sync."""
        return self._last_sync_time
