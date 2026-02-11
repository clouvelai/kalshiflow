"""
TopVolumeDiscovery - Simple volume-ranked event discovery.

Purpose:
    Discovers the N most active events on Kalshi by total 24h volume,
    loads them into the EventArbIndex for monitoring.

Key Responsibilities:
    1. **Fetch Events** - Paginates GET /events?status=open&with_nested_markets=true
    2. **Rank by Volume** - Sums volume_24h across each event's markets
    3. **Top-N Selection** - Keeps the N highest-volume events (configurable)
    4. **Size Filtering** - Skips events with too many markets (configurable cap)
    5. **Seed Support** - Optionally loads hard-coded event tickers as seeds
    6. **Index Integration** - Feeds events/markets into EventArbIndex

Architecture Position:
    Used by SingleArbCoordinator at startup and via background refresh loop.

Design Principles:
    - **Dead simple**: One API scan, sort, take top N
    - **Non-blocking**: All fetches are async
    - **Idempotent**: Re-discovering an already-loaded event is a no-op
    - **Size-safe**: Caps markets per event to avoid system overload
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from .index import EventArbIndex, EventMeta, MarketMeta

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.discovery")

DEFAULT_EVENT_COUNT = 10
DEFAULT_MAX_MARKETS_PER_EVENT = 50


@dataclass
class DiscoveryStats:
    """Telemetry for discovery operations."""
    total_fetches: int = 0
    total_events_discovered: int = 0
    total_markets_discovered: int = 0
    total_events_skipped_size: int = 0
    total_events_evicted: int = 0
    total_events_scanned: int = 0
    last_fetch_at: float = 0.0
    last_fetch_duration: float = 0.0
    errors: int = 0
    last_error: Optional[str] = None


class TopVolumeDiscovery:
    """
    Discovers the top N events by 24h volume on Kalshi.

    Simple approach:
    1. Paginate GET /events?status=open&with_nested_markets=true
    2. Sum volume_24h across each event's markets
    3. Sort descending, take top N
    4. Load into index

    Usage:
        discovery = TopVolumeDiscovery(index=index, trading_client=client, event_count=10)
        loaded = await discovery.discover()
        await discovery.start()   # background refresh
        await discovery.stop()    # shutdown
    """

    def __init__(
        self,
        index: EventArbIndex,
        trading_client,
        event_count: int = DEFAULT_EVENT_COUNT,
        seed_event_tickers: Optional[List[str]] = None,
        max_markets_per_event: int = DEFAULT_MAX_MARKETS_PER_EVENT,
        refresh_interval: float = 300.0,
        subscribe_callback: Optional[Callable[..., Coroutine]] = None,
        unsubscribe_callback: Optional[Callable[..., Coroutine]] = None,
        broadcast_callback: Optional[Callable[..., Coroutine]] = None,
    ):
        self._index = index
        self._client = trading_client
        self._event_count = event_count
        self._seed_event_tickers = list(seed_event_tickers or [])
        self._max_markets_per_event = max_markets_per_event
        self._refresh_interval = refresh_interval
        self._subscribe_callback = subscribe_callback
        self._unsubscribe_callback = unsubscribe_callback
        self._broadcast_callback = broadcast_callback

        self._known_event_tickers: Set[str] = set()
        self._event_volumes: Dict[str, int] = {}  # event_ticker -> total_volume_24h
        self._stats = DiscoveryStats()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._pending_tasks: set = set()  # Track fire-and-forget tasks for clean shutdown

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def known_events(self) -> Set[str]:
        return set(self._known_event_tickers)

    async def discover(self) -> int:
        """Run one discovery pass: fetch all open events, rank by volume, load top N.

        Returns:
            Number of new events loaded into the index.
        """
        start = time.time()
        total_new = 0

        # Phase 1: Load seed events
        for event_ticker in self._seed_event_tickers:
            if event_ticker in self._known_event_tickers:
                continue
            try:
                loaded = await self._load_event_by_ticker(event_ticker)
                if loaded:
                    total_new += 1
            except Exception as e:
                logger.warning(f"[DISCOVERY] Seed event {event_ticker} failed: {e}")
                self._stats.errors += 1
                self._stats.last_error = str(e)

        # Phase 2: Fetch all open events, rank by volume, load top N
        fetch_succeeded = False
        try:
            all_events = await self._fetch_all_open_events()
            self._stats.total_events_scanned = len(all_events)
            fetch_succeeded = len(all_events) > 0

            # Rank by total volume_24h across markets
            ranked = self._rank_by_volume(all_events)

            # Take top N (excluding already-known and oversized)
            loaded_count = 0
            for event_data, volume in ranked:
                if loaded_count >= self._event_count:
                    break

                event_ticker = event_data.get("event_ticker", "")
                if event_ticker in self._known_event_tickers:
                    loaded_count += 1  # counts toward the N limit
                    continue

                markets = event_data.get("markets", [])
                if len(markets) > self._max_markets_per_event:
                    logger.info(
                        f"[DISCOVERY] Skipping {event_ticker}: {len(markets)} markets "
                        f"exceeds cap of {self._max_markets_per_event}"
                    )
                    self._stats.total_events_skipped_size += 1
                    continue

                if self._load_event_from_data(event_data, volume):
                    total_new += 1
                    loaded_count += 1

        except Exception as e:
            logger.error(f"[DISCOVERY] Failed to fetch/rank events: {e}")
            self._stats.errors += 1
            self._stats.last_error = str(e)

        # Phase 3: Evict events that are no longer open on Kalshi
        if fetch_succeeded:
            fresh_open_tickers = {
                e.get("event_ticker", "") for e in all_events if e.get("event_ticker")
            }
            evicted = await self._evict_stale_events(fresh_open_tickers)
            if evicted > 0:
                self._stats.total_events_evicted += evicted

        duration = time.time() - start
        self._stats.total_fetches += 1
        self._stats.last_fetch_at = time.time()
        self._stats.last_fetch_duration = duration

        logger.info(
            f"[DISCOVERY] Pass complete: {total_new} new events "
            f"({len(self._known_event_tickers)} total, "
            f"scanned={self._stats.total_events_scanned}, "
            f"target={self._event_count}, {duration:.1f}s)"
        )

        # Broadcast full state to frontend
        if self._broadcast_callback:
            try:
                await self._broadcast_callback({
                    "type": "discovery_state",
                    "data": self.get_discovery_snapshot(),
                })
            except Exception:
                pass

        return total_new

    async def start(self) -> None:
        """Start background discovery loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._discovery_loop())
        logger.info(
            f"[DISCOVERY] Background loop started "
            f"(top {self._event_count} events, interval={self._refresh_interval}s)"
        )

    async def stop(self) -> None:
        """Stop background discovery loop and await pending tasks."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Await any in-flight subscribe/broadcast tasks
        if self._pending_tasks:
            await asyncio.gather(*self._pending_tasks, return_exceptions=True)
            self._pending_tasks.clear()
        logger.info("[DISCOVERY] Background loop stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Return discovery stats for health reporting."""
        return {
            "event_count": self._event_count,
            "known_events": len(self._known_event_tickers),
            "max_markets_per_event": self._max_markets_per_event,
            "total_fetches": self._stats.total_fetches,
            "total_events_discovered": self._stats.total_events_discovered,
            "total_markets_discovered": self._stats.total_markets_discovered,
            "total_events_skipped_size": self._stats.total_events_skipped_size,
            "total_events_evicted": self._stats.total_events_evicted,
            "total_events_scanned": self._stats.total_events_scanned,
            "last_fetch_at": self._stats.last_fetch_at,
            "last_fetch_duration": round(self._stats.last_fetch_duration, 2),
            "errors": self._stats.errors,
            "last_error": self._stats.last_error,
            "running": self._running,
        }

    def get_discovery_snapshot(self) -> Dict[str, Any]:
        """Full discovery state for frontend WebSocket.

        Shape:
            {
                "events": [
                    {
                        "event_ticker": "...",
                        "title": "...",
                        "market_count": 5,
                        "volume_24h": 123456,
                        "source": "volume" | "seed",
                    },
                    ...
                ],
                "stats": { ... },
                "timestamp": 1234567890.0,
            }
        """
        events_out = []

        for event_ticker in self._known_event_tickers:
            event = self._index.events.get(event_ticker)
            if not event:
                continue

            events_out.append({
                "event_ticker": event_ticker,
                "title": event.title,
                "series_ticker": event.series_ticker,
                "category": event.category,
                "mutually_exclusive": event.mutually_exclusive,
                "market_count": len(event.markets),
                "volume_24h": self._event_volumes.get(event_ticker, 0),
                "source": "seed" if event_ticker in self._seed_event_tickers else "volume",
            })

        # Sort by volume descending
        events_out.sort(key=lambda e: e["volume_24h"], reverse=True)

        return {
            "events": events_out,
            "event_count": len(events_out),
            "stats": self.get_stats(),
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------ #
    #  Internal                                                           #
    # ------------------------------------------------------------------ #

    async def _discovery_loop(self) -> None:
        """Background loop that periodically refreshes discovery."""
        while self._running:
            try:
                await asyncio.sleep(self._refresh_interval)
                if not self._running:
                    break
                await self.discover()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[DISCOVERY] Loop error: {e}")
                self._stats.errors += 1
                self._stats.last_error = str(e)
                await asyncio.sleep(30.0)

    async def _fetch_all_open_events(self) -> List[Dict[str, Any]]:
        """Paginate through all open events with nested markets."""
        all_events = []
        cursor = None

        while True:
            resp = await self._client.get_events(
                status="open",
                with_nested_markets=True,
                limit=200,
                cursor=cursor,
            )

            events = resp.get("events", [])
            if not events:
                break

            all_events.extend(events)

            next_cursor = resp.get("cursor", "")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

        logger.debug(f"[DISCOVERY] Fetched {len(all_events)} open events")
        return all_events

    def _rank_by_volume(self, events: List[Dict[str, Any]]) -> List[tuple]:
        """Sort events by total volume_24h across their markets (descending).

        Returns list of (event_data, total_volume) tuples.
        """
        scored = []
        for event_data in events:
            markets = event_data.get("markets", [])
            total_volume = sum(m.get("volume_24h", 0) or 0 for m in markets)
            scored.append((event_data, total_volume))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    async def _load_event_by_ticker(self, event_ticker: str) -> bool:
        """Load a single seed event by ticker."""
        meta = await self._index.load_event(event_ticker, self._client)
        if meta is None:
            return False

        if len(meta.markets) > self._max_markets_per_event:
            logger.info(
                f"[DISCOVERY] Seed {event_ticker} skipped: {len(meta.markets)} markets "
                f"exceeds cap of {self._max_markets_per_event}"
            )
            self._stats.total_events_skipped_size += 1
            return False

        self._known_event_tickers.add(event_ticker)
        self._event_volumes[event_ticker] = 0  # seeds don't have pre-computed volume
        self._stats.total_events_discovered += 1
        self._stats.total_markets_discovered += len(meta.markets)

        new_tickers = list(meta.markets.keys())
        if new_tickers and self._subscribe_callback:
            try:
                await self._subscribe_callback(new_tickers)
            except Exception as e:
                logger.warning(f"[DISCOVERY] Subscribe callback failed: {e}")

        if self._broadcast_callback:
            try:
                await self._broadcast_callback({
                    "type": "discovery_update",
                    "data": {
                        "event_ticker": event_ticker,
                        "title": meta.title,
                        "market_count": len(meta.markets),
                        "volume_24h": 0,
                        "source": "seed",
                    },
                })
            except Exception:
                pass

        return True

    def _load_event_from_data(self, event_data: Dict[str, Any], volume: int) -> bool:
        """Load an event from API response data into the index."""
        event_ticker = event_data.get("event_ticker", "")
        if not event_ticker:
            return False

        markets_data = event_data.get("markets", [])
        if not markets_data:
            return False

        meta = EventMeta(
            raw=event_data,
            event_ticker=event_ticker,
            series_ticker=event_data.get("series_ticker", ""),
            title=event_data.get("title", event_ticker),
            category=event_data.get("category", ""),
            mutually_exclusive=event_data.get("mutually_exclusive", False),
            subtitle=event_data.get("sub_title", ""),
            loaded_at=time.time(),
        )

        for market_data in markets_data:
            ticker = market_data.get("ticker", "")
            if not ticker:
                continue
            market_status = market_data.get("status", "open")
            if market_status not in ("open", "active"):
                continue
            meta.markets[ticker] = MarketMeta.from_api(market_data, event_ticker)

        if not meta.markets:
            return False

        # Register in index
        self._index._events[event_ticker] = meta
        for ticker in meta.markets:
            self._index._ticker_to_event[ticker] = event_ticker

        self._known_event_tickers.add(event_ticker)
        self._event_volumes[event_ticker] = volume
        self._stats.total_events_discovered += 1
        self._stats.total_markets_discovered += len(meta.markets)

        logger.info(
            f"[DISCOVERY] Loaded {event_ticker}: {meta.title} "
            f"({len(meta.markets)} markets, vol24h={volume:,})"
        )

        # Fire-and-forget callbacks (tracked for clean shutdown)
        new_tickers = list(meta.markets.keys())
        if new_tickers and self._subscribe_callback:
            task = asyncio.create_task(self._safe_subscribe(new_tickers))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        if self._broadcast_callback:
            task = asyncio.create_task(self._safe_broadcast({
                "type": "discovery_update",
                "data": {
                    "event_ticker": event_ticker,
                    "title": meta.title,
                    "market_count": len(meta.markets),
                    "volume_24h": volume,
                    "source": "volume",
                },
            }))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        return True

    async def _evict_stale_events(self, fresh_open_tickers: Set[str]) -> int:
        """Remove events no longer present in the API's open events list.

        Args:
            fresh_open_tickers: Set of event_tickers from the latest API fetch.

        Returns:
            Number of events evicted.
        """
        stale = self._known_event_tickers - fresh_open_tickers
        if not stale:
            return 0

        all_removed_tickers: List[str] = []
        eviction_info: List[Dict[str, Any]] = []

        for event_ticker in stale:
            # Collect info before removing
            event = self._index._events.get(event_ticker)
            if event:
                eviction_info.append({
                    "event_ticker": event_ticker,
                    "title": event.title,
                    "market_count": len(event.markets),
                })
                all_removed_tickers.extend(event.markets.keys())
                # Remove from index
                for ticker in list(event.markets.keys()):
                    self._index._ticker_to_event.pop(ticker, None)
                del self._index._events[event_ticker]

            self._known_event_tickers.discard(event_ticker)
            self._event_volumes.pop(event_ticker, None)

        logger.info(
            f"[DISCOVERY] Evicted {len(stale)} stale events "
            f"({len(all_removed_tickers)} markets): {sorted(stale)}"
        )

        # Unsubscribe removed market tickers from WS channels
        if all_removed_tickers and self._unsubscribe_callback:
            try:
                await self._unsubscribe_callback(all_removed_tickers)
            except Exception as e:
                logger.warning(f"[DISCOVERY] Unsubscribe callback failed: {e}")

        # Broadcast eviction details so frontend can show transient notice
        if self._broadcast_callback:
            try:
                await self._broadcast_callback({
                    "type": "discovery_eviction",
                    "data": {
                        "evicted": eviction_info,
                        "count": len(stale),
                        "timestamp": time.time(),
                    },
                })
            except Exception:
                pass

            # Broadcast updated event_arb_snapshot so the events Map updates
            try:
                snapshot = self._index.get_snapshot()
                await self._broadcast_callback({
                    "type": "event_arb_snapshot",
                    "data": snapshot,
                })
            except Exception:
                pass

        return len(stale)

    async def _safe_subscribe(self, tickers: List[str]) -> None:
        try:
            await self._subscribe_callback(tickers)
        except Exception as e:
            logger.warning(f"[DISCOVERY] Subscribe callback failed: {e}")

    async def _safe_broadcast(self, message: Dict) -> None:
        try:
            await self._broadcast_callback(message)
        except Exception:
            pass
