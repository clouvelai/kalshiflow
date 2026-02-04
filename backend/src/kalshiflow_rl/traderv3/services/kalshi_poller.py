"""
Kalshi REST API price polling service.

Periodically fetches BBO for all active paired markets via a single bulk
GET /markets?tickers=... call, emits KALSHI_API_PRICE_UPDATE events so
SpreadMonitor can use API prices as the primary signal.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..core.event_bus import EventBus
from ..core.events.types import EventType
from ..core.events.arb_events import KalshiApiPriceEvent
from ..services.pair_registry import PairRegistry

logger = logging.getLogger("kalshiflow_rl.traderv3.services.kalshi_poller")


class KalshiPoller:
    """
    Background poller for Kalshi REST API prices.

    Uses a single bulk GET /markets?tickers=... call per cycle (up to 200
    tickers) instead of per-ticker orderbook calls. Extracts yes_bid/yes_ask
    directly from the market objects returned by the API.
    """

    def __init__(
        self,
        trading_client: Any,
        pair_registry: PairRegistry,
        event_bus: EventBus,
        poll_interval: float = 3.0,
    ):
        self._trading_client = trading_client
        self._pair_registry = pair_registry
        self._event_bus = event_bus
        self._poll_interval = poll_interval

        # State
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._poll_count = 0
        self._error_count = 0
        self._consecutive_errors = 0
        self._last_poll: Optional[float] = None
        self._tickers_polled = 0

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            logger.warning("Kalshi poller already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Kalshi REST poller started (interval={self._poll_interval}s, bulk mode)")

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Kalshi REST poller stopped")

    async def _poll_loop(self) -> None:
        """Main polling loop with exponential backoff."""
        while self._running:
            try:
                await self._poll_once()
                self._consecutive_errors = 0
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                self._consecutive_errors = min(self._consecutive_errors + 1, 10)
                backoff = min(self._poll_interval * (2 ** self._consecutive_errors), 60.0)
                logger.warning(f"Kalshi poll error (consecutive={self._consecutive_errors}, backoff={backoff:.1f}s): {e}")
                await asyncio.sleep(backoff)

    async def _poll_once(self) -> None:
        """Execute a single poll cycle: one bulk API call for all tickers."""
        pairs = self._pair_registry.get_all_active()
        if not pairs:
            return

        # Collect unique Kalshi tickers and map ticker -> pair
        ticker_to_pair: Dict[str, Any] = {}
        for pair in pairs:
            if pair.kalshi_ticker not in ticker_to_pair:
                ticker_to_pair[pair.kalshi_ticker] = pair

        tickers = list(ticker_to_pair.keys())
        if not tickers:
            return

        self._poll_count += 1
        self._last_poll = time.time()

        t0 = time.time()
        # Single bulk call: GET /markets?tickers=T1,T2,...&limit=200
        result = await self._trading_client.get_markets(
            tickers=tickers,
            limit=200,
        )
        latency_ms = (time.time() - t0) * 1000

        markets = result.get("markets", [])
        polled = 0

        for market in markets:
            ticker = market.get("ticker")
            if not ticker or ticker not in ticker_to_pair:
                continue

            pair = ticker_to_pair[ticker]

            yes_bid = market.get("yes_bid")
            yes_ask = market.get("yes_ask")

            # Compute midpoint from bid/ask
            if yes_bid is not None and yes_ask is not None:
                # Filter out zero values (no liquidity)
                yes_bid = yes_bid if yes_bid > 0 else None
                yes_ask = yes_ask if yes_ask > 0 else None

            yes_mid = None
            if yes_bid is not None and yes_ask is not None:
                yes_mid = int((yes_bid + yes_ask) / 2)

            event = KalshiApiPriceEvent(
                pair_id=pair.id,
                kalshi_ticker=ticker,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                yes_mid=yes_mid,
                latency_ms=latency_ms,
            )
            await self._event_bus.emit(EventType.KALSHI_API_PRICE_UPDATE, event)
            polled += 1

        self._tickers_polled = polled

        if polled > 0:
            logger.debug(
                f"Kalshi bulk poll: {polled}/{len(tickers)} markets priced "
                f"({latency_ms:.0f}ms)"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get poller status for health/status endpoints."""
        return {
            "running": self._running,
            "poll_count": self._poll_count,
            "error_count": self._error_count,
            "consecutive_errors": self._consecutive_errors,
            "last_poll": self._last_poll,
            "tickers_polled": self._tickers_polled,
            "poll_interval": self._poll_interval,
        }

    def is_healthy(self) -> bool:
        """Health check."""
        return self._running and self._consecutive_errors < 5
