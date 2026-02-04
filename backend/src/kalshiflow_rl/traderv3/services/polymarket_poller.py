"""
Polymarket price polling service.

Periodically fetches midpoint prices for all active paired tokens from the
Polymarket CLOB API, emits POLY_PRICE_UPDATE events, and writes price_ticks
to Supabase.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from ..clients.polymarket_client import PolymarketClient
from ..core.event_bus import EventBus
from ..core.events.types import EventType
from ..core.events.arb_events import PolyPriceEvent
from ..services.pair_registry import PairRegistry

logger = logging.getLogger("kalshiflow_rl.traderv3.services.polymarket_poller")


class PolymarketPoller:
    """
    Background poller for Polymarket prices.

    Polls CLOB /midpoint at a configurable interval, normalizes to cents,
    emits POLY_PRICE_UPDATE events, and writes ticks to Supabase.
    Uses exponential backoff on failures (capped at 40s).
    """

    def __init__(
        self,
        poly_client: PolymarketClient,
        pair_registry: PairRegistry,
        event_bus: EventBus,
        poll_interval: float = 3.0,
        supabase_client: Any = None,
    ):
        self._poly_client = poly_client
        self._pair_registry = pair_registry
        self._event_bus = event_bus
        self._poll_interval = poll_interval
        self._supabase = supabase_client

        # State
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._poll_count = 0
        self._error_count = 0
        self._consecutive_errors = 0
        self._last_poll: Optional[float] = None
        self._last_prices: Dict[str, int] = {}  # token_id -> last price in cents

    async def start(self) -> None:
        """Start the polling loop."""
        if self._running:
            logger.warning("Poller already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._poll_loop())
        logger.info(f"Polymarket poller started (interval={self._poll_interval}s)")

    async def stop(self) -> None:
        """Stop the polling loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Polymarket poller stopped")

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
                backoff = min(self._poll_interval * (2 ** self._consecutive_errors), 40.0)
                logger.warning(f"Poll error (consecutive={self._consecutive_errors}, backoff={backoff:.1f}s): {e}")
                await asyncio.sleep(backoff)

    async def _poll_once(self) -> None:
        """Execute a single poll cycle."""
        token_ids = self._pair_registry.get_poly_token_ids()
        if not token_ids:
            if self._poll_count == 0:
                logger.debug("No Polymarket token IDs to poll (pair registry empty)")
            return

        t0 = time.time()
        prices = await self._poly_client.get_midpoints(token_ids)
        latency_ms = (time.time() - t0) * 1000

        if not prices:
            logger.debug(f"No midpoints returned for {len(token_ids)} tokens")
            return

        self._poll_count += 1
        self._last_poll = time.time()

        for token_id, price_float in prices.items():
            pair = self._pair_registry.get_by_poly(token_id)
            if not pair:
                continue

            yes_cents = max(0, min(100, round(price_float * 100)))
            no_cents = 100 - yes_cents

            event = PolyPriceEvent(
                pair_id=pair.id,
                kalshi_ticker=pair.kalshi_ticker,
                poly_token_id=token_id,
                poly_yes_cents=yes_cents,
                poly_no_cents=no_cents,
                source="api",
                latency_ms=latency_ms,
            )
            await self._event_bus.emit(EventType.POLY_PRICE_UPDATE, event)
            self._last_prices[token_id] = yes_cents

        if self._supabase:
            asyncio.create_task(self._write_ticks(prices))

    async def _write_ticks(self, prices: Dict[str, float]) -> None:
        """Write price ticks to Supabase (best-effort, non-blocking)."""
        try:
            rows = []
            for token_id, price_float in prices.items():
                pair = self._pair_registry.get_by_poly(token_id)
                if not pair:
                    continue

                yes_cents = max(0, min(100, round(price_float * 100)))
                rows.append({
                    "pair_id": pair.id,
                    "poly_yes_cents": yes_cents,
                    "poly_no_cents": 100 - yes_cents,
                })

            if rows:
                self._supabase.table("price_ticks").insert(rows).execute()
        except Exception as e:
            logger.debug(f"Failed to write price_ticks: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get poller status for health/status endpoints."""
        return {
            "running": self._running,
            "poll_count": self._poll_count,
            "error_count": self._error_count,
            "consecutive_errors": self._consecutive_errors,
            "last_poll": self._last_poll,
            "tokens_tracked": len(self._last_prices),
            "poll_interval": self._poll_interval,
        }

    def is_healthy(self) -> bool:
        """Health check - unhealthy if too many consecutive errors."""
        return self._running and self._consecutive_errors < 5
