"""
Market Price Syncer Service for TRADER V3.

REST API sync for market prices on startup and periodic refresh.
Works alongside WebSocket ticker updates to ensure prices are available.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.state_container import V3StateContainer
    from ..core.event_bus import EventBus

from ..core.state_container import MarketPriceData

logger = logging.getLogger("kalshiflow_rl.traderv3.services.market_price_syncer")


class MarketPriceSyncer:
    """
    REST API sync for market prices.

    - Initial fetch on startup (after positions load)
    - Periodic refresh every 30s
    - Updates state_container._market_prices
    - Triggers state broadcast after updating

    This service ensures market prices are available immediately on startup
    and stay fresh even if WebSocket ticker updates are delayed.
    """

    def __init__(
        self,
        trading_client: 'V3TradingClientIntegration',
        state_container: 'V3StateContainer',
        event_bus: 'EventBus',
        sync_interval: float = 30.0
    ):
        """
        Initialize market price syncer.

        Args:
            trading_client: Trading client for API calls
            state_container: State container for storing prices
            event_bus: Event bus for system activity events
            sync_interval: Seconds between periodic syncs (default 30)
        """
        self._client = trading_client
        self._state = state_container
        self._event_bus = event_bus
        self._sync_interval = sync_interval

        # Syncer state
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        self._started_at: Optional[float] = None

        # Health metrics
        self._last_sync_time: Optional[float] = None
        self._sync_count: int = 0
        self._tickers_synced: int = 0
        self._sync_errors: int = 0
        self._last_error: Optional[str] = None

        logger.info(f"MarketPriceSyncer initialized (sync_interval={sync_interval}s)")

    async def start(self) -> None:
        """
        Start the market price syncer.

        Performs initial sync immediately, then starts periodic sync loop.
        """
        if self._running:
            logger.warning("MarketPriceSyncer already running")
            return

        self._running = True
        self._started_at = time.time()

        # Initial sync
        logger.info("Starting MarketPriceSyncer - performing initial sync...")
        await self._sync_market_prices()

        # Start periodic sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info("MarketPriceSyncer started")

    async def stop(self) -> None:
        """Stop the market price syncer."""
        if not self._running:
            return

        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        logger.info(f"MarketPriceSyncer stopped (syncs: {self._sync_count})")

    async def _sync_loop(self) -> None:
        """Periodic sync every N seconds."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                await self._sync_market_prices()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market price sync loop: {e}")
                self._sync_errors += 1
                self._last_error = str(e)

    async def _sync_market_prices(self) -> None:
        """
        Fetch market prices for all position tickers via REST API.

        Updates state_container with MarketPriceData for each ticker.
        Handles pagination in batches of 100 (Kalshi API limit).
        """
        try:
            # Get tickers from current positions
            if not self._state.trading_state:
                logger.debug("No trading state - skipping market price sync")
                return

            tickers = list(self._state.trading_state.positions.keys())
            if not tickers:
                logger.debug("No positions - skipping market price sync")
                return

            # Batch tickers in groups of 100 (Kalshi API limit)
            BATCH_SIZE = 100
            logger.debug(f"Syncing market prices for {len(tickers)} tickers in batches of {BATCH_SIZE}...")

            markets = []
            for i in range(0, len(tickers), BATCH_SIZE):
                batch = tickers[i:i + BATCH_SIZE]
                batch_markets = await self._client.get_markets(tickers=batch)
                markets.extend(batch_markets)
                if len(tickers) > BATCH_SIZE:
                    logger.debug(f"Batch {i // BATCH_SIZE + 1}/{(len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE}: fetched {len(batch_markets)} markets")

            synced_count = 0
            now = time.time()

            # Log first market response for debugging
            if markets:
                sample = markets[0]
                logger.info(f"Sample market API response keys: {list(sample.keys())}")
                logger.info(f"Sample market prices: ticker={sample.get('ticker')}, "
                           f"yes_bid={sample.get('yes_bid')}, yes_ask={sample.get('yes_ask')}, "
                           f"no_bid={sample.get('no_bid')}, no_ask={sample.get('no_ask')}, "
                           f"last_price={sample.get('last_price')}")

            for market in markets:
                ticker = market.get("ticker")
                if not ticker:
                    continue

                # Create MarketPriceData from REST API response
                # REST API includes close_time which WebSocket ticker doesn't have
                price_data = MarketPriceData(
                    ticker=ticker,
                    last_price=market.get("last_price", 0),
                    yes_bid=market.get("yes_bid", 0),
                    yes_ask=market.get("yes_ask", 0),
                    no_bid=market.get("no_bid", 0),
                    no_ask=market.get("no_ask", 0),
                    volume=market.get("volume", 0),
                    open_interest=market.get("open_interest", 0),
                    close_time=market.get("close_time"),  # ISO timestamp from REST
                    timestamp=now,
                )

                # Update state container (don't trigger individual broadcasts)
                self._state._market_prices[ticker] = price_data
                synced_count += 1

            # Increment version once for all updates
            if synced_count > 0:
                self._state._market_prices_version += 1
                self._state._last_update = now

            # Update health metrics
            self._last_sync_time = now
            self._sync_count += 1
            self._tickers_synced = synced_count

            logger.info(
                f"Market price sync complete: {synced_count}/{len(tickers)} tickers "
                f"(sync #{self._sync_count})"
            )

            # Emit system activity for visibility
            await self._event_bus.emit_system_activity(
                activity_type="sync",
                message=f"Market prices synced: {synced_count} tickers",
                metadata={
                    "sync_type": "market_prices",
                    "tickers_synced": synced_count,
                    "sync_count": self._sync_count,
                    "severity": "info"
                }
            )

        except Exception as e:
            logger.error(f"Market price sync failed: {e}")
            self._sync_errors += 1
            self._last_error = str(e)

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
            "tickers_synced": self._tickers_synced,
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
    def tickers_synced(self) -> int:
        """Get number of tickers in last sync."""
        return self._tickers_synced

    @property
    def last_sync_time(self) -> Optional[float]:
        """Get timestamp of last successful sync."""
        return self._last_sync_time
