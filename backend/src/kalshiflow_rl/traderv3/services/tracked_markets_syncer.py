"""
Tracked Markets Syncer Service for Event Lifecycle Discovery Mode.

REST API sync for tracked market info with periodic refresh.
Provides real-time market data updates for the lifecycle discovery UI grid.

Pattern: Follows MarketPriceSyncer exactly for consistency.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable, TYPE_CHECKING

from ..state.tracked_markets import MarketStatus

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..state.tracked_markets import TrackedMarketsState
    from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.services.tracked_markets_syncer")


class TrackedMarketsSyncer:
    """
    REST API sync for tracked markets info.

    - Initial fetch on startup
    - Periodic refresh every N seconds (configurable)
    - Updates TrackedMarketsState with latest prices, volume, etc.
    - Emits system activity events for sync status

    This service ensures market info is available immediately on startup
    and stays fresh for the lifecycle discovery UI grid.
    """

    def __init__(
        self,
        trading_client: 'V3TradingClientIntegration',
        tracked_markets_state: 'TrackedMarketsState',
        event_bus: 'EventBus',
        sync_interval: float = 30.0,
        on_market_closed: Optional[Callable[[str], Awaitable[None]]] = None
    ):
        """
        Initialize tracked markets syncer.

        Args:
            trading_client: Trading client for API calls
            tracked_markets_state: State container for tracked markets
            event_bus: Event bus for system activity events
            sync_interval: Seconds between periodic syncs (default 30)
            on_market_closed: Callback when a market is detected as closed/settled via API
        """
        self._client = trading_client
        self._state = tracked_markets_state
        self._event_bus = event_bus
        self._sync_interval = sync_interval
        self._on_market_closed = on_market_closed

        # Syncer state
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._started_at: Optional[float] = None

        # Health metrics
        self._last_sync_time: Optional[float] = None
        self._sync_count: int = 0
        self._markets_synced: int = 0
        self._sync_errors: int = 0
        self._last_error: Optional[str] = None

        logger.info(f"TrackedMarketsSyncer initialized (sync_interval={sync_interval}s)")

    async def start(self) -> None:
        """
        Start the tracked markets syncer.

        Performs initial sync immediately, then starts periodic sync loop.
        """
        if self._running:
            logger.warning("TrackedMarketsSyncer already running")
            return

        self._running = True
        self._started_at = time.time()

        # Initial sync
        logger.info("Starting TrackedMarketsSyncer - performing initial sync...")
        await self._sync_market_info()

        # Start periodic sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info("TrackedMarketsSyncer started")

    async def stop(self) -> None:
        """Stop the tracked markets syncer."""
        if not self._running:
            return

        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info(f"TrackedMarketsSyncer stopped (syncs: {self._sync_count})")

    async def _sync_loop(self) -> None:
        """Periodic sync every N seconds."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                await self._sync_market_info()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in tracked markets sync loop: {e}")
                self._sync_errors += 1
                self._last_error = str(e)

    async def _sync_market_info(self) -> None:
        """
        Fetch market info for all tracked markets via REST API.

        Updates TrackedMarketsState with latest prices, volume, etc.
        Handles pagination in batches of 100 (Kalshi API limit).
        """
        try:
            # Get active tickers from tracked markets state
            tickers = self._state.get_active_tickers()
            if not tickers:
                logger.debug("No active tracked markets - skipping sync")
                return

            # Batch tickers in groups of 100 (Kalshi API limit)
            BATCH_SIZE = 100
            logger.debug(f"Syncing market info for {len(tickers)} tracked markets in batches of {BATCH_SIZE}...")

            markets = []
            for i in range(0, len(tickers), BATCH_SIZE):
                batch = tickers[i:i + BATCH_SIZE]
                batch_markets = await self._client.get_markets(tickers=batch)
                markets.extend(batch_markets)
                if len(tickers) > BATCH_SIZE:
                    logger.debug(f"Batch {i // BATCH_SIZE + 1}/{(len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE}: fetched {len(batch_markets)} markets")

            synced_count = 0
            closed_count = 0
            now = time.time()

            for market in markets:
                ticker = market.get("ticker")
                if not ticker:
                    continue

                # Check if market has closed/settled via API status
                api_status = market.get("status", "").lower()
                if api_status in ("closed", "settled", "inactive"):
                    logger.info(f"Market {ticker} detected as {api_status} via API sync")

                    # Update status in tracked markets state
                    new_status = MarketStatus.SETTLED if api_status == "settled" else MarketStatus.DETERMINED
                    await self._state.update_status(ticker, new_status)

                    # Trigger cleanup callback (unsubscribe orderbook)
                    if self._on_market_closed:
                        try:
                            await self._on_market_closed(ticker)
                        except Exception as e:
                            logger.warning(f"Failed to cleanup closed market {ticker}: {e}")

                    # Emit lifecycle event for Activity Feed
                    # Include result (YES/NO winner) when available
                    result = market.get("result", "")
                    await self._event_bus.emit_system_activity(
                        activity_type="lifecycle_event",
                        message=f"Market {ticker} {api_status}",
                        metadata={
                            "event_type": "determined" if api_status != "settled" else "settled",
                            "market_ticker": ticker,
                            "action": api_status,
                            "reason": "api_sync",
                            "result": result,
                        }
                    )

                    closed_count += 1
                    continue  # Skip normal update for closed markets

                # Update tracked market state with latest info
                updated = await self._state.update_market(
                    ticker,
                    price=market.get("last_price", 0),
                    volume=market.get("volume", 0),
                    open_interest=market.get("open_interest", 0),
                    yes_bid=market.get("yes_bid", 0),
                    yes_ask=market.get("yes_ask", 0),
                )

                if updated:
                    synced_count += 1

            # Update health metrics
            self._last_sync_time = now
            self._sync_count += 1
            self._markets_synced = synced_count

            closed_msg = f", {closed_count} closed" if closed_count > 0 else ""
            logger.info(
                f"Tracked markets sync complete: {synced_count}/{len(tickers)} markets{closed_msg} "
                f"(sync #{self._sync_count})"
            )

            # Emit system activity for visibility
            await self._event_bus.emit_system_activity(
                activity_type="sync",
                message=f"Tracked markets synced: {synced_count} markets",
                metadata={
                    "sync_type": "tracked_markets",
                    "markets_synced": synced_count,
                    "sync_count": self._sync_count,
                    "severity": "info"
                }
            )

        except Exception as e:
            logger.error(f"Tracked markets sync failed: {e}")
            self._sync_errors += 1
            self._last_error = str(e)

    async def sync_market_info(self) -> None:
        """
        Public method to trigger a sync manually.

        Useful for forcing a refresh when needed (e.g., after new market tracked).
        """
        await self._sync_market_info()

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
            "markets_synced": self._markets_synced,
            "sync_errors": self._sync_errors,
            "last_error": self._last_error,
            "sync_interval": self._sync_interval,
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
    def markets_synced(self) -> int:
        """Get number of markets in last sync."""
        return self._markets_synced

    @property
    def last_sync_time(self) -> Optional[float]:
        """Get timestamp of last successful sync."""
        return self._last_sync_time
