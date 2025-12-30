"""
Upcoming Markets Syncer Service for Event Lifecycle Discovery Mode.

REST API sync for upcoming (unopened) markets that will open within N hours.
Broadcasts to frontend for display in the Activity Feed schedule section.

Pattern: Follows TrackedMarketsSyncer exactly for consistency.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.websocket_manager import V3WebSocketManager
    from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.services.upcoming_markets_syncer")


class UpcomingMarketsSyncer:
    """
    REST API sync for upcoming (unopened) markets.

    - Initial fetch on startup
    - Periodic refresh every N seconds (configurable, default 60)
    - Filters markets opening within hours_ahead (default 4h)
    - Broadcasts to frontend via WebSocket
    - Caches results to avoid unnecessary broadcasts

    This service provides a schedule view of markets about to open
    for the lifecycle discovery Activity Feed panel.
    """

    def __init__(
        self,
        trading_client: 'V3TradingClientIntegration',
        websocket_manager: 'V3WebSocketManager',
        event_bus: 'EventBus',
        sync_interval: float = 60.0,
        hours_ahead: float = 4.0
    ):
        """
        Initialize upcoming markets syncer.

        Args:
            trading_client: Trading client for API calls
            websocket_manager: WebSocket manager for broadcasting
            event_bus: Event bus for system activity events
            sync_interval: Seconds between periodic syncs (default 60)
            hours_ahead: Hours ahead to look for upcoming markets (default 4)
        """
        self._client = trading_client
        self._websocket_manager = websocket_manager
        self._event_bus = event_bus
        self._sync_interval = sync_interval
        self._hours_ahead = hours_ahead

        # Syncer state
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._started_at: Optional[float] = None

        # Cached upcoming markets (to avoid unnecessary broadcasts)
        self._upcoming_markets: List[Dict[str, Any]] = []
        self._markets_version: int = 0

        # Health metrics
        self._last_sync_time: Optional[float] = None
        self._sync_count: int = 0
        self._markets_found: int = 0
        self._sync_errors: int = 0
        self._last_error: Optional[str] = None

        logger.info(
            f"UpcomingMarketsSyncer initialized "
            f"(sync_interval={sync_interval}s, hours_ahead={hours_ahead}h)"
        )

    async def start(self) -> None:
        """
        Start the upcoming markets syncer.

        Performs initial sync immediately, then starts periodic sync loop.
        """
        if self._running:
            logger.warning("UpcomingMarketsSyncer already running")
            return

        self._running = True
        self._started_at = time.time()

        # Initial sync
        logger.info("Starting UpcomingMarketsSyncer - performing initial sync...")
        await self._sync_upcoming_markets()

        # Start periodic sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info("UpcomingMarketsSyncer started")

    async def stop(self) -> None:
        """Stop the upcoming markets syncer."""
        if not self._running:
            return

        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info(f"UpcomingMarketsSyncer stopped (syncs: {self._sync_count})")

    async def _sync_loop(self) -> None:
        """Periodic sync every N seconds."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                await self._sync_upcoming_markets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in upcoming markets sync loop: {e}")
                self._sync_errors += 1
                self._last_error = str(e)

    async def _sync_upcoming_markets(self) -> None:
        """
        Fetch upcoming markets via REST API.

        Broadcasts to frontend if data changed.
        """
        try:
            # Fetch upcoming markets from trading client
            markets = await self._client.get_upcoming_markets(
                limit=100,
                max_open_hours=self._hours_ahead
            )

            now = time.time()
            self._last_sync_time = now
            self._sync_count += 1
            self._markets_found = len(markets)

            # Check if data changed (compare tickers)
            current_tickers = {m["ticker"] for m in self._upcoming_markets}
            new_tickers = {m["ticker"] for m in markets}

            data_changed = (
                current_tickers != new_tickers or
                self._sync_count == 1  # Always broadcast on first sync
            )

            # Update cache
            self._upcoming_markets = markets
            if data_changed:
                self._markets_version += 1

            # Broadcast if data changed
            if data_changed:
                await self._broadcast_upcoming_markets()
                logger.info(
                    f"Upcoming markets sync complete: {len(markets)} markets "
                    f"(sync #{self._sync_count}, version {self._markets_version})"
                )
            else:
                logger.debug(
                    f"Upcoming markets sync complete: {len(markets)} markets "
                    f"(no change, sync #{self._sync_count})"
                )

        except Exception as e:
            logger.error(f"Upcoming markets sync failed: {e}")
            self._sync_errors += 1
            self._last_error = str(e)

    async def _broadcast_upcoming_markets(self) -> None:
        """Broadcast upcoming markets to all connected frontend clients."""
        message = {
            "type": "upcoming_markets",
            "data": {
                "markets": self._upcoming_markets,
                "count": len(self._upcoming_markets),
                "version": self._markets_version,
                "timestamp": time.time(),
                "hours_ahead": self._hours_ahead,
            }
        }

        await self._websocket_manager.broadcast_message(
            message["type"],
            message["data"]
        )

    def get_upcoming_markets(self) -> List[Dict[str, Any]]:
        """
        Get cached upcoming markets.

        Used for sending snapshot to new WebSocket clients on connect.
        """
        return self._upcoming_markets

    def get_snapshot_message(self) -> Dict[str, Any]:
        """
        Get snapshot message for new WebSocket client.

        Returns the same format as broadcast for consistency.
        """
        return {
            "type": "upcoming_markets",
            "data": {
                "markets": self._upcoming_markets,
                "count": len(self._upcoming_markets),
                "version": self._markets_version,
                "timestamp": time.time(),
                "hours_ahead": self._hours_ahead,
            }
        }

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
            "markets_found": self._markets_found,
            "markets_version": self._markets_version,
            "sync_errors": self._sync_errors,
            "last_error": self._last_error,
            "sync_interval": self._sync_interval,
            "hours_ahead": self._hours_ahead,
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
    def markets_found(self) -> int:
        """Get number of markets in last sync."""
        return self._markets_found

    @property
    def last_sync_time(self) -> Optional[float]:
        """Get timestamp of last successful sync."""
        return self._last_sync_time
