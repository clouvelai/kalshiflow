"""
Pair Index Service - Manages pair lifecycle using the standalone pair_index builder.

On startup: loads pre-built pairs from Supabase (instant).
Background: refreshes index periodically via PairIndexBuilder.refresh().
Owns: subscriptions (orderbook + Poly WS), index snapshot broadcasting, trading whitelist.

OUTPUT: Live price index consumed by trading layer (SpreadMonitor) + UI.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from ..config.environment import V3Config
from ..core.event_bus import EventBus
from ..core.websocket_manager import V3WebSocketManager
from .pair_registry import PairRegistry, MarketPair

logger = logging.getLogger("kalshiflow_rl.traderv3.services.pair_index_service")


class PairIndexService:
    """
    Manages the pair index lifecycle using the standalone pair_index builder.

    Startup: Pairs are already loaded in PairRegistry from Supabase.
    Background: Calls PairIndexBuilder.refresh() to discover new pairs,
    deactivate closed events, and sync the DB.
    """

    def __init__(
        self,
        pair_registry: PairRegistry,
        event_bus: EventBus,
        websocket_manager: V3WebSocketManager,
        config: V3Config,
        pair_index_builder=None,
        supabase_client=None,
        orderbook_integration=None,
        poly_ws_client=None,
        spread_monitor=None,
    ):
        self._pair_registry = pair_registry
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._config = config
        self._builder = pair_index_builder
        self._supabase = supabase_client
        self._orderbook_integration = orderbook_integration
        self._poly_ws_client = poly_ws_client
        self._spread_monitor = spread_monitor

        self._running = False
        self._initial_discovery_done = False
        self._initial_discovery_event = asyncio.Event()
        self._scan_task: Optional[asyncio.Task] = None
        self._last_scan_at: Optional[float] = None
        self._next_scan_at: Optional[float] = None
        self._scan_count = 0
        self._total_new_pairs = 0
        self._cached_snapshot: Optional[Dict[str, Any]] = None

    async def start(self) -> None:
        """Start the pair index service.

        On startup: subscribes all loaded pairs to orderbook + Poly WS,
        then starts background refresh loop.
        """
        if self._running:
            return

        self._running = True

        # Subscribe all loaded pairs immediately (they're already in the registry)
        async def _run_initial():
            try:
                await self._subscribe_loaded_pairs()
                self._apply_trading_whitelist()
            except Exception as e:
                logger.error(f"Initial pair subscription failed: {e}")
            finally:
                self._initial_discovery_done = True
                self._initial_discovery_event.set()

            await self._broadcast_index()
            logger.info(
                f"PairIndexService ready: "
                f"{len(self._pair_registry.get_events_grouped())} events, "
                f"{self._pair_registry.count} pairs subscribed"
            )

        asyncio.create_task(_run_initial())

        # Start background refresh loop
        self._scan_task = asyncio.create_task(self._refresh_loop())

        logger.info("PairIndexService started (DB-loaded, background refresh enabled)")

    async def wait_for_initial_discovery(self, timeout: float = 30.0) -> bool:
        """Wait until initial pair subscription completes."""
        try:
            await asyncio.wait_for(self._initial_discovery_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Initial pair subscription did not complete within {timeout}s")
            return False

    async def _subscribe_loaded_pairs(self) -> None:
        """Subscribe all pairs already in the registry to price feeds."""
        pairs = self._pair_registry.get_all_active()
        if not pairs:
            logger.info("No pairs loaded from DB - index will be empty until refresh")
            return

        # Subscribe Kalshi tickers to orderbook
        if self._orderbook_integration:
            subscribed = 0
            for pair in pairs:
                if pair.kalshi_ticker:
                    try:
                        await self._orderbook_integration.subscribe_market(pair.kalshi_ticker)
                        subscribed += 1
                    except Exception as e:
                        logger.debug(f"Failed to subscribe {pair.kalshi_ticker}: {e}")
            if subscribed:
                logger.info(f"Subscribed {subscribed} Kalshi tickers to orderbook")

        # Subscribe Poly tokens to WS
        if self._poly_ws_client:
            token_ids = []
            for pair in pairs:
                if pair.poly_token_id_yes:
                    token_ids.append(pair.poly_token_id_yes)
                if pair.poly_token_id_no:
                    token_ids.append(pair.poly_token_id_no)
            if token_ids:
                try:
                    await self._poly_ws_client.subscribe_tokens(token_ids)
                    logger.info(f"Subscribed {len(token_ids)} Poly tokens to WS")
                except Exception as e:
                    logger.warning(f"Failed to subscribe Poly tokens: {e}")

        # Register with SpreadMonitor
        if self._spread_monitor:
            for pair in pairs:
                self._spread_monitor.register_pair(pair)

    # ─── Background Refresh ─────────────────────────────────────────────

    async def _refresh_loop(self) -> None:
        """Background loop: refresh index periodically using the builder."""
        # Wait for initial subscription to complete
        while self._running and not self._initial_discovery_done:
            await asyncio.sleep(1.0)

        if not self._running:
            return

        while self._running:
            interval = self._config.arb_scan_interval_seconds
            self._next_scan_at = time.time() + interval
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

            if not self._running:
                break

            try:
                await self._run_refresh()
            except Exception as e:
                logger.error(f"Pair index refresh error: {e}")

    async def _run_refresh(self) -> None:
        """Run a single refresh cycle using the builder."""
        if not self._builder:
            return

        self._scan_count += 1
        self._last_scan_at = time.time()
        logger.info(f"PairIndexService: Starting refresh #{self._scan_count}")

        result = await self._builder.refresh(no_llm=False)

        # Reload pairs from DB into registry
        if self._supabase and result.upserted > 0:
            before_count = self._pair_registry.count
            await self._pair_registry.load_from_supabase(self._supabase)
            new_count = self._pair_registry.count - before_count

            if new_count > 0:
                self._total_new_pairs += new_count
                # Subscribe new pairs
                await self._subscribe_loaded_pairs()
                logger.info(f"Refresh #{self._scan_count}: {new_count} new pairs loaded")

        self._apply_trading_whitelist()
        await self._broadcast_index()

        logger.info(
            f"PairIndexService: Refresh #{self._scan_count} complete: "
            f"{self._pair_registry.count} pairs, "
            f"builder: {result.total_pairs} matched, "
            f"{result.upserted} upserted, {result.deactivated} deactivated "
            f"({result.duration_seconds:.1f}s)"
        )

    def report_bad_pair(self, kalshi_ticker: str, reason: str = "") -> bool:
        """Report a bad pair (called by orchestrator). Deactivates in DB."""
        if self._builder:
            success = self._builder.report_bad_pair(kalshi_ticker, reason)
            if success:
                # Remove from registry
                pair = self._pair_registry.get_by_kalshi(kalshi_ticker)
                if pair:
                    self._pair_registry.remove_pair(pair.id)
            return success
        return False

    # ─── Trading Whitelist ──────────────────────────────────────────────

    def _apply_trading_whitelist(self) -> None:
        """Mark events as tradeable/non-tradeable based on config whitelist."""
        whitelist = self._config.arb_active_event_tickers
        if not whitelist:
            for group in self._pair_registry.get_events_grouped():
                group.is_tradeable = True
            return
        for group in self._pair_registry.get_events_grouped():
            group.is_tradeable = group.kalshi_event_ticker in whitelist

    # ─── Index Snapshot ─────────────────────────────────────────────────

    def get_index_snapshot(self) -> Dict[str, Any]:
        """Get JSON-serializable index snapshot for WebSocket."""
        events = self._pair_registry.get_events_grouped()
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
            f"{self._scan_count} refreshes, "
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
