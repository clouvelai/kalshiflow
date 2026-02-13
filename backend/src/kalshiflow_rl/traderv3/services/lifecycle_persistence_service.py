"""
Lifecycle Persistence Service - Fire-and-forget DB writes for lifecycle state.

Follows the SessionMemoryStore pattern:
- Fire-and-forget writes (don't block event loop)
- Startup recovery from DB
- Flush pending writes on shutdown

Key Responsibilities:
    1. **Fire-and-Forget Writes** - Persist TrackedEvent/TrackedMarket changes without blocking
    2. **Startup Recovery** - Load active events/markets from DB after restarts
    3. **Shutdown Flush** - Await all pending writes before exit
    4. **Deactivation** - Soft-delete events/markets that are settled

Architecture Position:
    Used by:
    - EventLifecycleService: Calls persist_event/persist_market on state changes
    - Coordinator: Calls load_from_db on startup, stop on shutdown
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ...data.database import RLDatabase
    from ..state.tracked_events import TrackedEvent
    from ..state.tracked_markets import TrackedMarket

logger = logging.getLogger("kalshiflow_rl.traderv3.services.lifecycle_persistence_service")


class LifecyclePersistenceService:
    """
    Persists TrackedEventsState and TrackedMarketsState to Supabase.

    Fire-and-forget writes following SessionMemoryStore pattern.
    Provides startup recovery to reload state from DB after restarts.
    """

    def __init__(self, db: "RLDatabase"):
        self._db = db
        self._pending_writes: List[asyncio.Task] = []
        self._write_count = 0
        self._error_count = 0
        self._running = False

    async def load_from_db(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Load persisted events and markets from DB for startup recovery.

        Returns:
            (events_list, markets_list) - lists of dicts from DB
        """
        try:
            events = await self._db.get_active_tracked_events()
            markets = await self._db.get_active_tracked_markets_v2()
            logger.info(f"Loaded {len(events)} events, {len(markets)} markets from DB")
            return events, markets
        except Exception as e:
            logger.error(f"Failed to load from DB: {e}")
            return [], []

    def persist_event(self, event: "TrackedEvent") -> None:
        """Fire-and-forget persist of a TrackedEvent."""
        if not self._running:
            return
        task = asyncio.create_task(self._write_event(event))
        self._pending_writes.append(task)
        task.add_done_callback(
            lambda t: self._pending_writes.remove(t) if t in self._pending_writes else None
        )

    def persist_market(self, market: "TrackedMarket") -> None:
        """Fire-and-forget persist of a TrackedMarket."""
        if not self._running:
            return
        task = asyncio.create_task(self._write_market(market))
        self._pending_writes.append(task)
        task.add_done_callback(
            lambda t: self._pending_writes.remove(t) if t in self._pending_writes else None
        )

    def deactivate_event(self, event_ticker: str) -> None:
        """Fire-and-forget soft-delete of a tracked event."""
        if not self._running:
            return
        task = asyncio.create_task(self._deactivate_event(event_ticker))
        self._pending_writes.append(task)
        task.add_done_callback(
            lambda t: self._pending_writes.remove(t) if t in self._pending_writes else None
        )

    def deactivate_market(self, ticker: str) -> None:
        """Fire-and-forget soft-delete of a tracked market."""
        if not self._running:
            return
        task = asyncio.create_task(self._deactivate_market(ticker))
        self._pending_writes.append(task)
        task.add_done_callback(
            lambda t: self._pending_writes.remove(t) if t in self._pending_writes else None
        )

    async def _write_event(self, event: "TrackedEvent") -> None:
        try:
            data = event.to_dict()
            await self._db.upsert_tracked_event(event.event_ticker, data)
            self._write_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to persist event {event.event_ticker}: {e}")

    async def _write_market(self, market: "TrackedMarket") -> None:
        try:
            data = {
                "event_ticker": market.event_ticker,
                "title": market.title,
                "category": market.category,
                "status": market.status.value,
                "open_ts": market.open_ts,
                "close_ts": market.close_ts,
                "determined_ts": market.determined_ts,
                "settled_ts": market.settled_ts,
                "discovery_source": market.discovery_source,
                "market_info": market.market_info,
            }
            await self._db.upsert_tracked_market_v2(market.ticker, data)
            self._write_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to persist market {market.ticker}: {e}")

    async def _deactivate_event(self, event_ticker: str) -> None:
        try:
            await self._db.deactivate_tracked_event(event_ticker)
            self._write_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to deactivate event {event_ticker}: {e}")

    async def _deactivate_market(self, ticker: str) -> None:
        try:
            await self._db.deactivate_tracked_market_v2(ticker)
            self._write_count += 1
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to deactivate market {ticker}: {e}")

    async def start(self) -> None:
        """Start accepting fire-and-forget writes."""
        self._running = True
        logger.info("LifecyclePersistenceService started")

    async def stop(self) -> None:
        """Flush pending writes on shutdown."""
        self._running = False
        await self.flush()
        logger.info(
            f"LifecyclePersistenceService stopped "
            f"(writes={self._write_count}, errors={self._error_count})"
        )

    async def flush(self) -> None:
        """Await all pending writes."""
        if self._pending_writes:
            logger.info(f"Flushing {len(self._pending_writes)} pending writes...")
            await asyncio.gather(*self._pending_writes, return_exceptions=True)
            self._pending_writes.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "running": self._running,
            "write_count": self._write_count,
            "error_count": self._error_count,
            "pending_writes": len(self._pending_writes),
        }
