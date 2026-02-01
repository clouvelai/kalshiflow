"""
True Market Open (TMO) Fetcher Service.

Background service that fetches true market open prices from the Kalshi
candlestick API after markets are tracked via lifecycle discovery.

Purpose:
    The RLM strategy measures price_drop from market open, but our live
    "first observed" price may miss the true market opening. This service
    fetches the actual first candlestick's open price for better accuracy.

Architecture:
    - Subscribes to MARKET_TRACKED events (triggered by lifecycle discovery)
    - Queues markets for TMO fetching
    - Rate-limits to 10 req/s (Kalshi API limit)
    - Broadcasts TMO_FETCHED events for RLMService consumption
    - Updates TrackedMarketsState with TMO data

Rate Limiting:
    Kalshi API limit: 10 requests/second
    600 markets = ~60 seconds to fetch all TMOs after discovery

Error Handling:
    - Retries on transient errors (429, 500, timeout)
    - Max 3 retries per market
    - Marks market as tmo_fetch_failed after max retries
    - Falls back to first observed price in RLM signal detection
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Set
from dataclasses import dataclass

from ..core.event_bus import EventBus, MarketTrackedEvent
from ..state.tracked_markets import TrackedMarketsState

logger = logging.getLogger("kalshiflow_rl.traderv3.services.tmo_fetcher")


@dataclass
class TMOFetchTask:
    """A task to fetch TMO for a market."""
    ticker: str
    market_info: Dict[str, Any]
    attempts: int = 0
    queued_at: float = 0.0

    def __post_init__(self):
        if self.queued_at == 0.0:
            self.queued_at = time.time()


class RateLimiter:
    """
    Token bucket rate limiter with semaphore for concurrent request control.

    Implements 10 requests/second with max 5 concurrent in-flight requests.
    """

    def __init__(self, rate: float = 10.0, max_concurrent: int = 5):
        """
        Initialize rate limiter.

        Args:
            rate: Maximum requests per second
            max_concurrent: Maximum concurrent in-flight requests
        """
        self._rate = rate
        self._tokens = rate
        self._max_tokens = rate
        self._last_refill = time.time()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        # Wait for semaphore (limits concurrent requests)
        await self._semaphore.acquire()

        # Wait for token bucket (limits rate)
        async with self._lock:
            await self._wait_for_token()

    def release(self) -> None:
        """Release the semaphore after request completes."""
        self._semaphore.release()

    async def _wait_for_token(self) -> None:
        """Wait until a token is available in the bucket."""
        while True:
            # Refill tokens based on time elapsed
            now = time.time()
            elapsed = now - self._last_refill
            self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return

            # Wait for next token
            wait_time = (1.0 - self._tokens) / self._rate
            await asyncio.sleep(wait_time)


class TrueMarketOpenFetcher:
    """
    Background service for fetching true market open prices.

    Lifecycle:
        1. start() - Subscribe to events, start background task
        2. _handle_market_tracked() - Queue markets for TMO fetch
        3. _fetch_loop() - Process queue with rate limiting
        4. _fetch_tmo() - Fetch TMO from candlestick API
        5. stop() - Cleanup and shutdown

    Events:
        - Subscribes to: MARKET_TRACKED
        - Emits: TMO_FETCHED
    """

    def __init__(
        self,
        event_bus: EventBus,
        trading_client_integration,  # V3TradingClientIntegration
        tracked_markets: TrackedMarketsState,
        max_retries: int = 3,
        rate_limit: float = 10.0,
    ):
        """
        Initialize TMO fetcher.

        Args:
            event_bus: Event bus for subscribing/emitting events
            trading_client_integration: Trading client for API calls
            tracked_markets: Tracked markets state to update
            max_retries: Maximum retry attempts per market
            rate_limit: Max requests per second (default 10)
        """
        self._event_bus = event_bus
        self._trading_client = trading_client_integration
        self._tracked_markets = tracked_markets
        self._max_retries = max_retries

        # Rate limiter (10 req/s, 5 concurrent)
        self._rate_limiter = RateLimiter(rate=rate_limit, max_concurrent=5)

        # Task queue
        self._queue: asyncio.Queue[TMOFetchTask] = asyncio.Queue()
        self._pending_tickers: Set[str] = set()

        # Background task
        self._fetch_task: Optional[asyncio.Task] = None
        self._running = False

        # Metrics
        self._fetched_count = 0
        self._failed_count = 0
        self._queued_count = 0
        self._started_at: Optional[float] = None

        logger.info(f"TrueMarketOpenFetcher initialized (rate={rate_limit}/s, retries={max_retries})")

    async def start(self) -> None:
        """Start the TMO fetcher service."""
        if self._running:
            logger.warning("TMO fetcher already running")
            return

        logger.info("Starting TrueMarketOpenFetcher...")
        self._running = True
        self._started_at = time.time()

        # Subscribe to MARKET_TRACKED events
        await self._event_bus.subscribe_to_market_tracked(self._handle_market_tracked)

        # Start background fetch loop
        self._fetch_task = asyncio.create_task(self._fetch_loop())

        # Queue existing tracked markets that don't have TMO yet
        await self._queue_existing_markets()

        logger.info("✅ TrueMarketOpenFetcher started")

    async def _queue_existing_markets(self) -> None:
        """Queue all existing tracked markets that don't have TMO yet."""
        markets = self._tracked_markets.get_all()
        queued = 0

        for market in markets:
            ticker = market.ticker
            if not ticker:  # Skip empty tickers
                continue
            if ticker in self._pending_tickers:
                continue
            if market.true_market_open is not None:
                continue
            if getattr(market, 'tmo_fetch_failed', False):
                continue

            task = TMOFetchTask(
                ticker=ticker,
                market_info=market.to_dict(),
            )
            await self._queue.put(task)
            self._pending_tickers.add(ticker)
            self._queued_count += 1
            queued += 1

        if queued > 0:
            logger.info(f"Queued {queued} existing markets for TMO fetch")

    async def stop(self) -> None:
        """Stop the TMO fetcher service."""
        if not self._running:
            return

        logger.info("Stopping TrueMarketOpenFetcher...")
        self._running = False

        # Cancel fetch task
        if self._fetch_task:
            self._fetch_task.cancel()
            try:
                await asyncio.wait_for(self._fetch_task, timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        # Unsubscribe from events
        from ..core.event_bus import EventType
        self._event_bus.unsubscribe(EventType.MARKET_TRACKED, self._handle_market_tracked)

        logger.info(
            f"✅ TrueMarketOpenFetcher stopped - "
            f"fetched={self._fetched_count}, failed={self._failed_count}, queued={self._queued_count}"
        )

    async def _handle_market_tracked(self, event: MarketTrackedEvent) -> None:
        """
        Handle MARKET_TRACKED event - queue market for TMO fetch.

        Args:
            event: Market tracked event with ticker and market_info
        """
        ticker = event.market_ticker

        # Skip if already queued/pending
        if ticker in self._pending_tickers:
            logger.debug(f"TMO already pending for {ticker}")
            return

        # Skip if TMO already fetched or permanently failed
        market = self._tracked_markets.get_market(ticker)
        if market and market.true_market_open is not None:
            logger.debug(f"TMO already fetched for {ticker}")
            return
        if market and getattr(market, 'tmo_fetch_failed', False):
            return

        # Queue for fetch
        task = TMOFetchTask(
            ticker=ticker,
            market_info=event.market_info or {},
        )
        await self._queue.put(task)
        self._pending_tickers.add(ticker)
        self._queued_count += 1

        logger.debug(f"Queued TMO fetch for {ticker} (queue_size={self._queue.qsize()})")

    async def _fetch_loop(self) -> None:
        """Background loop processing TMO fetch queue."""
        logger.info("TMO fetch loop started")

        while self._running:
            try:
                # Get next task from queue (with timeout)
                try:
                    task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Acquire rate limit token
                await self._rate_limiter.acquire()

                try:
                    # Fetch TMO
                    success = await self._fetch_tmo(task)

                    if not success and task.attempts < self._max_retries:
                        # Re-queue for retry
                        task.attempts += 1
                        await self._queue.put(task)
                        logger.debug(f"Retry queued for {task.ticker} (attempt {task.attempts})")
                    elif not success:
                        # Max retries exceeded
                        self._failed_count += 1
                        self._pending_tickers.discard(task.ticker)
                        await self._mark_fetch_failed(task.ticker)
                    else:
                        # Success
                        self._fetched_count += 1
                        self._pending_tickers.discard(task.ticker)

                finally:
                    self._rate_limiter.release()
                    self._queue.task_done()

            except asyncio.CancelledError:
                logger.info("TMO fetch loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in TMO fetch loop: {e}")
                await asyncio.sleep(0.1)

        logger.info("TMO fetch loop stopped")

    async def _fetch_tmo(self, task: TMOFetchTask) -> bool:
        """
        Fetch TMO for a single market.

        Args:
            task: Fetch task with ticker and market_info

        Returns:
            True if fetch succeeded, False otherwise
        """
        ticker = task.ticker

        try:
            # Fetch TMO via trading client
            tmo = await self._trading_client.get_true_market_open(
                ticker=ticker,
                market_info=task.market_info,
            )

            if tmo is None:
                logger.debug(f"No TMO data for {ticker}")
                return False

            # Update TrackedMarketsState
            await self._tracked_markets.update_market(
                ticker,
                true_market_open=tmo,
                tmo_fetched_at=time.time(),
                tmo_fetch_failed=False,
            )

            # Emit TMO_FETCHED event
            open_ts = task.market_info.get("open_ts", 0)
            await self._event_bus.emit_tmo_fetched(
                market_ticker=ticker,
                true_market_open=tmo,
                open_ts=open_ts,
            )

            logger.info(f"TMO fetched for {ticker}: {tmo}c")
            return True

        except Exception as e:
            logger.warning(f"TMO fetch failed for {ticker}: {e}")
            return False

    async def _mark_fetch_failed(self, ticker: str) -> None:
        """Mark a market as TMO fetch failed after max retries."""
        await self._tracked_markets.update_market(
            ticker,
            tmo_fetch_failed=True,
        )
        logger.warning(f"TMO fetch failed after {self._max_retries} retries for {ticker}")

    # ======== Metrics ========

    def get_stats(self) -> Dict[str, Any]:
        """Get TMO fetcher statistics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "fetched": self._fetched_count,
            "failed": self._failed_count,
            "queued_total": self._queued_count,
            "queue_size": self._queue.qsize(),
            "pending": len(self._pending_tickers),
            "uptime_seconds": uptime,
        }

    def is_healthy(self) -> bool:
        """Check if TMO fetcher is healthy."""
        if not self._running:
            return False

        if self._fetch_task and self._fetch_task.done():
            return False

        return True

    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        stats = self.get_stats()
        return {
            "healthy": self.is_healthy(),
            "running": self._running,
            "task_active": self._fetch_task is not None and not self._fetch_task.done(),
            "fetched": stats["fetched"],
            "failed": stats["failed"],
            "queue_size": stats["queue_size"],
            "pending": stats["pending"],
        }
