"""
Pair Index Service - Single owner of event pair lifecycle.

Owns: event discovery, pairing, price feeds, index.
On startup: fetches top Kalshi events by volume, matches against Polymarket,
registers pairs, subscribes to orderbook + Poly WS, broadcasts index.
Background: re-ranks events, adds new pairs, prunes expired, broadcasts.

OUTPUT: Live price index consumed by trading layer (SpreadMonitor) + UI.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..config.environment import V3Config
from ..core.event_bus import EventBus
from ..core.websocket_manager import V3WebSocketManager
from .pair_registry import PairRegistry, EventGroup
from .pairing_service import NormalizedEvent

logger = logging.getLogger("kalshiflow_rl.traderv3.services.pair_index_service")


class PairIndexService:
    """
    Builds and maintains a persistent, volume-sorted pair index.

    Single authority for the full event pair lifecycle:
    1. Discover top Kalshi events by volume
    2. Match against Polymarket via PairingService
    3. Register pairs, subscribe to price feeds
    4. Maintain and broadcast live index
    """

    def __init__(
        self,
        pairing_service,
        pair_registry: PairRegistry,
        event_bus: EventBus,
        websocket_manager: V3WebSocketManager,
        config: V3Config,
        supabase_client=None,
        orderbook_integration=None,
        poly_ws_client=None,
        spread_monitor=None,
    ):
        self._pairing_service = pairing_service
        self._pair_registry = pair_registry
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._config = config
        self._supabase = supabase_client
        self._orderbook_integration = orderbook_integration
        self._poly_ws_client = poly_ws_client
        self._spread_monitor = spread_monitor

        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        self._last_scan_at: Optional[float] = None
        self._next_scan_at: Optional[float] = None
        self._scan_count = 0
        self._total_new_pairs = 0

        # Cached index snapshot
        self._cached_snapshot: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        """Start the pair index service - drives full discovery + subscription."""
        if self._running:
            return

        self._running = True

        # Drive the initial discovery pipeline
        try:
            await self._initial_discovery()
        except Exception as e:
            logger.error(f"Initial discovery failed: {e}")

        # Broadcast initial index
        await self._broadcast_index()

        # Start background scan loop
        self._scan_task = asyncio.create_task(self._scan_loop())

        logger.info(
            f"PairIndexService started: "
            f"{len(self._pair_registry.get_events_grouped())} events, "
            f"{self._pair_registry.count} pairs"
        )

    async def _initial_discovery(self) -> None:
        """Run the full startup discovery pipeline."""
        # 1. Select top events by volume
        top_events = await self._select_top_events()
        if not top_events:
            logger.warning("No Kalshi events found - index will be empty until next scan")
            return

        event_tickers = [e.event_id for e in top_events]
        logger.info(f"Selected top {len(top_events)} events by volume: {event_tickers[:5]}{'...' if len(event_tickers) > 5 else ''}")

        # 2. Match against Polymarket (LLM finds correct market pairings)
        candidates = await self._pairing_service.match_for_events(
            top_events,
            min_score=self._config.arb_min_pair_confidence,
            use_llm=True,
        )
        if not candidates:
            logger.info("No Poly matches for top events")
            # Still build event groups from Kalshi data for display
            self._build_event_groups_from_kalshi_events(top_events)
            return

        logger.info(f"Matched {len(candidates)} Poly pairs for top events")

        # 3. Activate pairs (DB + registry)
        match_method = "llm_index" if any("llm_matched" in c.match_signals for c in candidates) else "auto_index"
        result = await self._pairing_service.activate_pairs_batch(
            candidates,
            match_method=match_method,
            default_confidence=0.9,
        )
        activated = result.get("activated", 0)
        self._total_new_pairs += activated
        logger.info(f"Activated {activated} pairs")

        # 4. Subscribe new Kalshi tickers to orderbook
        await self._subscribe_kalshi_tickers(candidates)

        # 5. Subscribe new Poly tokens to WS
        await self._subscribe_poly_tokens(candidates)

        # 6. Build event groups with volume metadata
        self._build_event_groups_from_candidates(candidates, top_events)

    async def _select_top_events(self) -> list:
        """Fetch Kalshi events and return top N by volume."""
        all_events = await self._pairing_service.fetch_kalshi_events()
        if not all_events:
            return []

        def event_volume(e: NormalizedEvent) -> int:
            total = 0
            for m in e.raw.get("markets", []):
                total += m.get("volume", 0) or 0
                total += m.get("volume_24h", 0) or 0
            return total

        ranked = sorted(all_events, key=event_volume, reverse=True)
        top = ranked[:self._config.arb_top_n_events]

        # Fallback for demo env where all volumes may be 0:
        # prefer events with more markets (more pairing opportunities)
        if top and event_volume(top[0]) == 0:
            ranked = sorted(all_events, key=lambda e: e.market_count, reverse=True)
            top = ranked[:self._config.arb_top_n_events]
            logger.info("All event volumes are 0 (demo env?) - ranking by market count instead")

        total_volume = sum(event_volume(e) for e in top)
        logger.info(
            f"Fetched {len(all_events)} Kalshi events, selected top {len(top)} "
            f"(total volume: {total_volume:,})"
        )
        return top

    async def _subscribe_kalshi_tickers(self, candidates) -> None:
        """Subscribe candidate Kalshi tickers to orderbook."""
        if not self._orderbook_integration:
            return

        subscribed = 0
        for c in candidates:
            if c.kalshi_ticker:
                try:
                    await self._orderbook_integration.subscribe_market(c.kalshi_ticker)
                    subscribed += 1
                except Exception as e:
                    logger.debug(f"Failed to subscribe {c.kalshi_ticker} to orderbook: {e}")

        if subscribed:
            logger.info(f"Subscribed {subscribed} arb tickers to orderbook")

    async def _subscribe_poly_tokens(self, candidates) -> None:
        """Subscribe candidate Poly tokens to WebSocket client."""
        if not self._poly_ws_client:
            return

        token_ids = []
        for c in candidates:
            if c.poly_token_id_yes:
                token_ids.append(c.poly_token_id_yes)
            if c.poly_token_id_no:
                token_ids.append(c.poly_token_id_no)

        if token_ids:
            try:
                await self._poly_ws_client.subscribe_tokens(token_ids)
                logger.info(f"Subscribed {len(token_ids)} Poly tokens to WS")
            except Exception as e:
                logger.warning(f"Failed to subscribe Poly tokens: {e}")

    def _build_event_groups_from_kalshi_events(self, events) -> None:
        """Build event groups from Kalshi events even without Poly matches."""
        for e in events:
            event_ticker = e.event_id
            volume = 0
            for m in e.raw.get("markets", []):
                volume += m.get("volume", 0) or 0
                volume += m.get("volume_24h", 0) or 0

            self._pair_registry.update_event_metadata(
                event_ticker,
                title=e.title,
                category=e.category,
                volume_24h=volume,
            )
        self._apply_trading_whitelist()

    def _build_event_groups_from_candidates(self, candidates, kalshi_events=None) -> None:
        """Extract volume and metadata from candidates into event groups."""
        event_volumes = {}
        event_metadata = {}

        for c in candidates:
            event_ticker = c.kalshi_event_ticker
            if not event_ticker:
                continue

            kalshi_volume = 0
            if c.kalshi_event and c.kalshi_event.raw:
                for m in c.kalshi_event.raw.get("markets", []):
                    kalshi_volume += m.get("volume", 0) or 0

            poly_volume = 0
            if c.poly_event and c.poly_event.raw:
                poly_raw = c.poly_event.raw
                poly_volume = poly_raw.get("volume", 0) or 0
                if not poly_volume:
                    for m in poly_raw.get("markets", []):
                        poly_volume += int(m.get("volume", 0) or 0)

            combined = kalshi_volume + poly_volume
            if event_ticker not in event_volumes or combined > event_volumes[event_ticker]:
                event_volumes[event_ticker] = combined

            if event_ticker not in event_metadata and c.kalshi_event:
                event_metadata[event_ticker] = {
                    "title": c.kalshi_event.title,
                    "category": c.kalshi_event.category,
                }

        # Also add top events that had no candidates (display only)
        if kalshi_events:
            for e in kalshi_events:
                et = e.event_id
                if et not in event_volumes:
                    vol = 0
                    for m in e.raw.get("markets", []):
                        vol += m.get("volume", 0) or 0
                        vol += m.get("volume_24h", 0) or 0
                    event_volumes[et] = vol
                    event_metadata[et] = {"title": e.title, "category": e.category}

        for event_ticker, volume in event_volumes.items():
            meta = event_metadata.get(event_ticker, {})
            self._pair_registry.update_event_metadata(
                event_ticker,
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                volume_24h=volume,
            )

        self._apply_trading_whitelist()

    def _apply_trading_whitelist(self) -> None:
        """Mark events as tradeable/non-tradeable based on config whitelist."""
        whitelist = self._config.arb_active_event_tickers
        if not whitelist:
            for group in self._pair_registry.get_events_grouped():
                group.is_tradeable = True
            return

        for group in self._pair_registry.get_events_grouped():
            group.is_tradeable = group.kalshi_event_ticker in whitelist

    # ─── Background Scan ────────────────────────────────────────────────

    async def _scan_loop(self) -> None:
        """Background loop: run scan every scan_interval."""
        # Initial delay to let system stabilize
        await asyncio.sleep(10.0)

        while self._running:
            try:
                await self._run_scan()
            except Exception as e:
                logger.error(f"Pair index scan error: {e}")

            self._next_scan_at = time.time() + self._config.arb_scan_interval_seconds
            try:
                await asyncio.sleep(self._config.arb_scan_interval_seconds)
            except asyncio.CancelledError:
                break

    async def _run_scan(self) -> None:
        """Run a single scan cycle: re-rank, match new, subscribe, broadcast."""
        self._scan_count += 1
        self._last_scan_at = time.time()
        logger.info(f"PairIndexService: Starting scan #{self._scan_count}")

        # 1. Re-select top events (volumes may have shifted)
        top_events = await self._select_top_events()
        if not top_events:
            logger.info("PairIndexService: No events found in scan")
            await self._broadcast_index()
            return

        # 2. Match any new events not already paired (LLM finds correct pairings)
        candidates = await self._pairing_service.match_for_events(
            top_events,
            min_score=self._config.arb_min_pair_confidence,
            use_llm=True,
        )

        if candidates:
            # 3. Activate new pairs
            match_method = "llm_index" if any("llm_matched" in c.match_signals for c in candidates) else "auto_index"
            result = await self._pairing_service.activate_pairs_batch(
                candidates,
                match_method=match_method,
                default_confidence=0.9,
            )
            activated = result.get("activated", 0)
            self._total_new_pairs += activated

            # 4. Subscribe new tickers/tokens
            if activated > 0:
                await self._subscribe_kalshi_tickers(candidates)
                await self._subscribe_poly_tokens(candidates)

            logger.info(f"PairIndexService: Scan #{self._scan_count}: {activated} new pairs")
        else:
            logger.info(f"PairIndexService: Scan #{self._scan_count}: no new candidates")

        # 5. Rebuild event groups with updated volume data
        self._build_event_groups_from_candidates(candidates or [], top_events)

        # 6. Broadcast updated index
        await self._broadcast_index()

        logger.info(
            f"PairIndexService: Scan #{self._scan_count} complete: "
            f"{len(self._pair_registry.get_events_grouped())} events, "
            f"{self._pair_registry.count} total pairs"
        )

    # ─── Index Snapshot ─────────────────────────────────────────────────

    def get_index_snapshot(self) -> Dict[str, Any]:
        """Get JSON-serializable index snapshot for WebSocket."""
        events = self._pair_registry.get_events_grouped()

        # Cap at max_events
        events = events[:self._config.arb_max_events]

        event_data = []
        total_pairs = 0

        for group in events:
            pairs_data = []
            tradeable_count = 0
            for pair in group.pairs:
                pair_dict = {
                    "pair_id": pair.id,
                    "kalshi_ticker": pair.kalshi_ticker,
                    "question": pair.question,
                    "match_confidence": pair.match_confidence,
                }
                # Merge live price fields from SpreadMonitor
                if self._spread_monitor:
                    ss = self._spread_monitor.get_spread_state(pair.id)
                    if ss:
                        pair_dict.update(ss.to_dict())
                        if ss.tradeable:
                            tradeable_count += 1
                pairs_data.append(pair_dict)
            total_pairs += len(pairs_data)

            event_data.append({
                "event_ticker": group.kalshi_event_ticker,
                "title": group.title,
                "category": group.category,
                "volume_24h": group.volume_24h,
                "market_count": group.market_count,
                "is_tradeable": group.is_tradeable,
                "tradeable_pair_count": tradeable_count,
                "pairs": pairs_data,
            })

        snapshot = {
            "events": event_data,
            "total_pairs": total_pairs,
            "total_events": len(event_data),
            "last_scan_at": self._last_scan_at,
            "next_scan_at": self._next_scan_at,
        }

        self._cached_snapshot = snapshot
        return snapshot

    async def _broadcast_index(self) -> None:
        """Broadcast pair_index_snapshot to all WebSocket clients."""
        snapshot = self.get_index_snapshot()
        await self._websocket_manager.broadcast_message("pair_index_snapshot", snapshot)

    # ─── Lifecycle ──────────────────────────────────────────────────────

    async def stop(self) -> None:
        """Stop the pair index service."""
        if not self._running:
            return

        self._running = False

        if self._scan_task and not self._scan_task.done():
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"PairIndexService stopped: "
            f"{self._scan_count} scans, "
            f"{self._total_new_pairs} new pairs discovered"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get service status for health endpoints."""
        return {
            "running": self._running,
            "scan_count": self._scan_count,
            "last_scan_at": self._last_scan_at,
            "next_scan_at": self._next_scan_at,
            "total_new_pairs": self._total_new_pairs,
            "total_events": len(self._pair_registry.get_events_grouped()),
            "total_pairs": self._pair_registry.count,
        }

    def is_healthy(self) -> bool:
        """Health check."""
        return self._running
