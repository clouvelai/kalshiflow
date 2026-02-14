"""MMCoordinator - Wires MMIndex + MMMonitor + QuoteEngine + Admiral.

Startup sequence:
1. Create MMIndex
2. Load events via REST (from mm_event_tickers config)
3. Subscribe market tickers to orderbook WS
4. Create session order group
5. Setup gateway + TradingSession
6. Create QuoteEngine with initial QuoteConfig
7. Setup Admiral tools (MMToolContext)
8. Create MMMonitor
9. Prefetch orderbooks via REST
10. Create MMAttentionRouter + Admiral
11. Start all components
12. Broadcast initial snapshot
"""

import asyncio
import logging
import os
import time
from typing import Dict, Optional

from .index import MMIndex
from .models import QuoteConfig
from .monitor import MMMonitor
from ..single_arb.event_understanding import UnderstandingBuilder
from ..single_arb.mentions_models import configure as configure_models

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.coordinator")

DEFAULT_MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "single_arb", "memory", "data"
)


class MMCoordinator:
    """Coordinator for the market maker system.

    Wires all components together and manages lifecycle.
    """

    def __init__(
        self,
        config,
        event_bus,
        websocket_manager,
        orderbook_integration,
        trading_client=None,
        gateway=None,
    ):
        self._config = config
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._orderbook_integration = orderbook_integration
        self._trading_client = trading_client
        self._gateway = gateway

        self._index: Optional[MMIndex] = None
        self._monitor: Optional[MMMonitor] = None
        self._quote_engine = None
        self._admiral = None
        self._attention_router = None
        self._order_group_id: Optional[str] = None
        self._system_ready = asyncio.Event()
        self._running = False
        self._started_at: Optional[float] = None
        self._search_service = None
        self._memory_store = None
        self._understanding_builder: Optional[UnderstandingBuilder] = None
        self._deferred_init_task: Optional[asyncio.Task] = None
        self._news_impact_task: Optional[asyncio.Task] = None
        self._position_sync_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Initialize and start all MM components."""
        if self._running:
            return

        logger.info("[MM_COORD] Starting market maker system...")
        self._started_at = time.time()

        try:
            # Configure model tiers
            configure_models(self._config)

            # Step 1: Create MMIndex
            self._index = MMIndex(
                fee_per_contract_cents=getattr(self._config, "single_arb_fee_per_contract", 1),
                min_edge_cents=0.5,
            )

            # Step 2: Load events
            event_tickers = self._config.mm_event_tickers
            if not event_tickers:
                logger.warning("[MM_COORD] No mm_event_tickers configured!")
                return

            # Gateway returns Pydantic models; load_event expects dicts.
            # Use MarketDataAdapter for dict-based calls, raw gateway for orders.
            if self._gateway:
                from ..gateway.market_data_adapter import MarketDataAdapter
                data_client = MarketDataAdapter(self._gateway)
            else:
                data_client = self._trading_client
            api_client = self._gateway or self._trading_client

            for et in event_tickers:
                meta = await self._index.load_event(et, data_client)
                if meta:
                    logger.info(f"[MM_COORD] Loaded {et}: {meta.title} ({len(meta.markets)} markets)")

            if not self._index.events:
                logger.error("[MM_COORD] No events loaded, aborting")
                return

            # Step 3: Subscribe to orderbook WS
            for ticker in self._index.market_tickers:
                if self._orderbook_integration:
                    await self._orderbook_integration.subscribe_market(ticker)
                    logger.debug(f"[MM_COORD] Subscribed to orderbook: {ticker}")

            # Step 4: Create order group
            if self._gateway:
                try:
                    og = await self._gateway.create_order_group(contracts_limit=10000)
                    self._order_group_id = og.order_group_id
                    logger.info(f"[MM_COORD] Order group: {self._order_group_id}")
                except Exception as e:
                    logger.warning(f"[MM_COORD] Order group creation failed: {e}")

            # Step 5: Setup session
            from ..agent_tools.session import TradingSession
            session = TradingSession(
                order_group_id=self._order_group_id or "",
            )

            # Step 6: Create QuoteEngine
            from .quote_engine import QuoteEngine
            quote_config = QuoteConfig(
                enabled=True,
                base_spread_cents=self._config.mm_base_spread_cents,
                quote_size=self._config.mm_quote_size,
                skew_factor=self._config.mm_skew_factor,
                max_position=self._config.mm_max_position,
                max_event_exposure=self._config.mm_max_event_exposure,
                refresh_interval=self._config.mm_refresh_interval,
            )

            async def _ws_broadcast(msg_type: str, data: Dict) -> None:
                if self._websocket_manager:
                    await self._websocket_manager.broadcast_message(msg_type, data)

            self._quote_engine = QuoteEngine(
                index=self._index,
                gateway=self._gateway,
                config=quote_config,
                max_drawdown_cents=self._config.mm_max_drawdown_cents,
                ws_broadcast=_ws_broadcast,
            )

            # Step 7: Setup tools
            from .tools import MMToolContext, set_context
            from ..single_arb.memory.session_store import SessionMemoryStore

            # Initialize memory (reuse single_arb memory store)
            self._memory_store = SessionMemoryStore()

            # Initialize search service
            if self._config.tavily_enabled and self._config.tavily_api_key:
                try:
                    from ..single_arb.tavily_service import TavilySearchService
                    self._search_service = TavilySearchService(
                        api_key=self._config.tavily_api_key,
                        search_depth=self._config.tavily_search_depth,
                    )
                except Exception as e:
                    logger.warning(f"[MM_COORD] Tavily init failed: {e}")

            # Create UnderstandingBuilder (sync, no LLM calls yet)
            understanding_cache_dir = os.path.join(DEFAULT_MEMORY_DIR, "understanding")
            self._understanding_builder = UnderstandingBuilder(
                cache_dir=understanding_cache_dir,
                search_service=self._search_service,
            )
            logger.info("[MM_COORD] UnderstandingBuilder created (LLM builds deferred)")

            ctx = MMToolContext(
                gateway=self._gateway,
                index=self._index,
                quote_engine=self._quote_engine,
                memory=self._memory_store,
                search=self._search_service,
                session=session,
                broadcast=_ws_broadcast,
                understanding_builder=self._understanding_builder,
            )
            set_context(ctx)

            # Step 8: Create monitor
            from .attention import MMAttentionRouter
            self._attention_router = MMAttentionRouter()

            self._monitor = MMMonitor(
                index=self._index,
                event_bus=self._event_bus,
                trading_client=api_client,
                config=self._config,
                attention_router=self._attention_router,
                broadcast_callback=_ws_broadcast,
            )

            # Step 9: Prefetch orderbooks
            logger.info("[MM_COORD] Prefetching orderbooks via REST...")
            for ticker in self._index.market_tickers:
                try:
                    ob = await api_client.get_orderbook(ticker)
                    if ob:
                        # Gateway Orderbook model: yes/no are List[List[int]] [[price, qty], ...]
                        self._index.on_orderbook_update(ticker, ob.yes or [], ob.no or [], source="api")
                except Exception as e:
                    logger.warning(f"[MM_COORD] Prefetch failed for {ticker}: {e}")

            logger.info(f"[MM_COORD] Index readiness: {self._index.arb_index.readiness_summary}")

            # Step 10: Create Admiral (if enabled)
            if self._config.mm_admiral_enabled:
                from .admiral import Admiral
                from .context_builder import MMContextBuilder

                context_builder = MMContextBuilder(self._index, self._quote_engine)

                self._admiral = Admiral(
                    context_builder=context_builder,
                    attention_router=self._attention_router,
                    config=self._config,
                    event_callback=_ws_broadcast,
                    system_ready=self._system_ready,
                )

            # Step 11: Start all components
            await self._monitor.start()

            # Get initial balance
            try:
                balance = await self._gateway.get_balance()
                balance_cents = balance.balance if hasattr(balance, 'balance') else 0
                self._quote_engine.start(balance_cents)
            except Exception:
                self._quote_engine.start(0)

            if self._admiral:
                await self._admiral.start()

            # Launch background tasks
            self._deferred_init_task = asyncio.create_task(self._run_deferred_init())
            self._news_impact_task = asyncio.create_task(self._news_impact_tracker_loop())
            self._position_sync_task = asyncio.create_task(self._run_position_sync_loop())

            self._system_ready.set()

            # Step 12: Broadcast initial snapshot + register replay provider
            try:
                snapshot = self._index.get_full_snapshot()
                await _ws_broadcast("mm_snapshot", snapshot)
            except Exception:
                pass

            # Register snapshot provider so new WS clients get MM state on connect
            if self._websocket_manager:
                self._websocket_manager.set_mm_snapshot_provider(
                    lambda: self._index.get_full_snapshot()
                )

            self._running = True
            elapsed = time.time() - self._started_at
            logger.info(
                f"[MM_COORD] Market maker started in {elapsed:.1f}s: "
                f"{len(self._index.events)} events, {len(self._index.market_tickers)} markets"
            )

        except Exception as e:
            logger.error(f"[MM_COORD] Startup failed: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """Stop all MM components."""
        logger.info("[MM_COORD] Stopping market maker...")
        self._running = False

        # Cancel background tasks
        for task in (self._deferred_init_task, self._news_impact_task, self._position_sync_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        if self._admiral:
            await self._admiral.stop()

        if self._quote_engine:
            await self._quote_engine.stop()

        if self._monitor:
            await self._monitor.stop()

        if self._memory_store:
            await self._memory_store.flush()

        # Clean up order group
        if self._order_group_id and self._gateway:
            try:
                await self._gateway.reset_order_group(self._order_group_id)
            except Exception:
                pass

        logger.info("[MM_COORD] Market maker stopped")

    def get_status(self) -> Dict:
        """Get MM system status."""
        status = {
            "running": self._running,
            "started_at": self._started_at,
            "uptime_seconds": time.time() - self._started_at if self._started_at else 0,
            "events": len(self._index.events) if self._index else 0,
            "markets": len(self._index.market_tickers) if self._index else 0,
        }

        if self._quote_engine:
            status["quote_engine"] = {
                "running": self._quote_engine.is_running,
                **self._quote_engine.state.to_dict(),
            }

        if self._admiral:
            status["admiral"] = self._admiral.get_status()

        if self._monitor:
            status["monitor"] = self._monitor.get_stats()

        return status

    # ------------------------------------------------------------------ #
    #  Position + Balance Sync (reconcile with Kalshi API)                #
    # ------------------------------------------------------------------ #

    async def _run_position_sync_loop(self) -> None:
        """Periodically sync balance and positions from Kalshi API.

        Runs every 30 seconds:
        1. Fetch balance → update quote engine
        2. Fetch positions → reconcile with local inventory
        3. Detect settled markets → clean up from index
        4. Broadcast updated state via WS
        """
        logger.info("[POSITION_SYNC] Starting sync loop (30s interval)")
        await asyncio.sleep(5)  # Initial delay to let quotes settle

        while self._running:
            try:
                await self._sync_balance_and_positions()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[POSITION_SYNC] Error: {e}")
                await asyncio.sleep(30)

        logger.info("[POSITION_SYNC] Sync loop stopped")

    async def _sync_balance_and_positions(self) -> None:
        """Single sync pass: balance + positions from Kalshi API."""
        if not self._gateway or not self._index:
            return

        # 1. Sync balance
        try:
            balance = await self._gateway.get_balance()
            balance_cents = balance.balance if hasattr(balance, 'balance') else 0
            if self._quote_engine:
                self._quote_engine.update_balance(balance_cents)
        except Exception as e:
            logger.debug(f"[POSITION_SYNC] Balance fetch failed: {e}")
            balance_cents = None

        # 2. Sync positions for our event tickers
        positions_by_market = {}
        for event_ticker in list(self._index.events.keys()):
            try:
                positions = await self._gateway.get_positions(event_ticker=event_ticker)
                for pos in positions:
                    ticker = getattr(pos, 'ticker', None)
                    if ticker:
                        positions_by_market[ticker] = pos
            except Exception as e:
                logger.debug(f"[POSITION_SYNC] Positions fetch failed for {event_ticker}: {e}")

        # 3. Reconcile positions with local inventory
        reconciled = 0
        for ticker in self._index.market_tickers:
            inv = self._index.get_inventory(ticker)
            api_pos = positions_by_market.get(ticker)
            api_position = getattr(api_pos, 'position', 0) if api_pos else 0
            api_realized = getattr(api_pos, 'realized_pnl', 0) if api_pos else 0

            # Only update if API shows a different position
            if api_position != inv.position:
                logger.info(
                    f"[POSITION_SYNC] Reconciling {ticker}: "
                    f"local={inv.position} → api={api_position}"
                )
                inv.position = api_position
                if api_pos:
                    inv.realized_pnl_cents = api_realized
                reconciled += 1

        if reconciled > 0:
            logger.info(f"[POSITION_SYNC] Reconciled {reconciled} markets")

        # 4. Broadcast balance update
        if self._websocket_manager and balance_cents is not None:
            try:
                total_realized = self._index.total_realized_pnl()
                total_unrealized = self._index.total_unrealized_pnl()
                await self._websocket_manager.broadcast_message("mm_balance_update", {
                    "balance_cents": balance_cents,
                    "total_realized_pnl_cents": round(total_realized, 1),
                    "total_unrealized_pnl_cents": round(total_unrealized, 1),
                    "positions_synced": len(positions_by_market),
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    #  Deferred initialization (background LLM builds)                    #
    # ------------------------------------------------------------------ #

    async def _run_deferred_init(self) -> None:
        """Build event understanding in background for all loaded events."""
        try:
            if self._understanding_builder and self._index:
                logger.info("[DEFERRED_INIT] Building event understanding...")
                built_count = 0
                total = len(self._index.events)
                for i, (event_ticker, event) in enumerate(self._index.events.items(), 1):
                    try:
                        understanding = await self._understanding_builder.build(event)
                        event.understanding = understanding.to_dict()
                        built_count += 1
                        logger.info(f"[DEFERRED_INIT] Built understanding for {event_ticker} ({i}/{total})")
                    except Exception as e:
                        logger.warning(f"[DEFERRED_INIT] Understanding failed for {event_ticker}: {e}")
                logger.info(f"[DEFERRED_INIT] Built {built_count}/{total} understandings")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[DEFERRED_INIT] Failed: {e}")

    # ------------------------------------------------------------------ #
    #  News-price impact tracking                                         #
    # ------------------------------------------------------------------ #

    async def _news_impact_tracker_loop(self) -> None:
        """Backfill news_price_impacts by checking price changes after news events.

        Runs every 30 minutes. For each news memory with a price_snapshot:
        - Creates news_price_impacts rows (one per market) if they don't exist
        - Fills in price_after_1h/4h/24h as enough time passes
        """
        logger.info("[NEWS_IMPACT] Background tracker started (30-min interval)")

        # Immediate first pass on startup
        try:
            await self._backfill_news_impacts()
        except Exception as e:
            logger.warning(f"[NEWS_IMPACT] Startup pass error: {e}")

        while self._running:
            try:
                await asyncio.sleep(1800)  # 30 min
                if not self._running:
                    break
                await self._backfill_news_impacts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[NEWS_IMPACT] Error in tracker loop: {e}")
                await asyncio.sleep(60)

        logger.info("[NEWS_IMPACT] Background tracker stopped")

    async def _backfill_news_impacts(self) -> None:
        """Query recent news memories and create/update news_price_impacts rows."""
        import json
        from datetime import datetime, timezone, timedelta

        try:
            from kalshiflow_rl.data.database import rl_db
            pool = await rl_db.get_pool()
        except Exception as e:
            logger.debug(f"[NEWS_IMPACT] DB not available: {e}")
            return

        async with pool.acquire() as conn:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
            rows = await conn.fetch(
                """
                SELECT id, price_snapshot, created_at, event_tickers
                FROM agent_memories
                WHERE memory_type = 'news'
                  AND price_snapshot IS NOT NULL
                  AND created_at >= $1
                ORDER BY created_at DESC
                """,
                cutoff,
            )

            if not rows:
                return

            created = 0
            updated = 0
            now = time.time()

            for row in rows:
                memory_id = row["id"]
                try:
                    raw_snap = row["price_snapshot"]
                    snapshot = json.loads(raw_snap) if isinstance(raw_snap, str) else raw_snap
                except (json.JSONDecodeError, TypeError):
                    continue

                news_ts = row["created_at"].timestamp() if row["created_at"] else None
                if not news_ts:
                    continue

                age_hours = (now - news_ts) / 3600.0
                event_tickers = row["event_tickers"] or []

                for market_ticker, market_snap in snapshot.items():
                    if market_ticker.startswith("_"):
                        continue
                    if not isinstance(market_snap, dict):
                        continue

                    original_mid = market_snap.get("yes_mid")
                    if original_mid is None:
                        continue

                    existing = await conn.fetchrow(
                        """
                        SELECT id, price_after_1h, price_after_4h, price_after_24h
                        FROM news_price_impacts
                        WHERE news_memory_id = $1 AND market_ticker = $2
                        """,
                        memory_id, market_ticker,
                    )

                    if not existing:
                        await conn.execute(
                            """
                            INSERT INTO news_price_impacts (
                                news_memory_id, market_ticker, event_ticker,
                                price_at_news
                            ) VALUES ($1, $2, $3, $4::jsonb)
                            """,
                            memory_id, market_ticker,
                            event_tickers[0] if event_tickers else None,
                            json.dumps(market_snap),
                        )
                        created += 1
                        existing = {"id": None, "price_after_1h": None, "price_after_4h": None, "price_after_24h": None}

                    updates = {}
                    windows = [
                        (1.0, "price_after_1h", "change_1h_cents"),
                        (4.0, "price_after_4h", "change_4h_cents"),
                        (24.0, "price_after_24h", "change_24h_cents"),
                    ]
                    for min_hours, price_col, change_col in windows:
                        if existing[price_col] is None and age_hours >= min_hours:
                            snap_now = self._get_market_snapshot(market_ticker)
                            if snap_now:
                                mid_now = snap_now.get("yes_mid")
                                change = round(mid_now - original_mid, 2) if mid_now is not None else None
                                updates[price_col] = json.dumps(snap_now)
                                updates[change_col] = change

                    if not updates:
                        continue

                    # Compute magnitude from all available changes
                    all_changes = []
                    for key in ("change_1h_cents", "change_4h_cents", "change_24h_cents"):
                        val = updates.get(key)
                        if val is not None:
                            all_changes.append(abs(val))
                    if existing.get("id"):
                        prev_row = await conn.fetchrow(
                            "SELECT change_1h_cents, change_4h_cents, change_24h_cents FROM news_price_impacts WHERE news_memory_id = $1 AND market_ticker = $2",
                            memory_id, market_ticker,
                        )
                        if prev_row:
                            for key in ("change_1h_cents", "change_4h_cents", "change_24h_cents"):
                                if key not in updates and prev_row[key] is not None:
                                    all_changes.append(abs(prev_row[key]))

                    if all_changes:
                        magnitude = self._classify_magnitude(max(all_changes))
                        updates["magnitude"] = magnitude

                        signal_map = {"large": 1.0, "medium": 0.8, "small": 0.6, "none": 0.3}
                        sq = signal_map.get(magnitude, 0.5)
                        try:
                            await conn.execute(
                                "UPDATE agent_memories SET signal_quality = GREATEST(signal_quality, $1) WHERE id = $2",
                                sq, memory_id,
                            )
                        except Exception as e:
                            logger.debug(f"[NEWS_IMPACT] signal_quality update failed for {memory_id}: {e}")

                    set_clauses = []
                    params = []
                    for i, (col, val) in enumerate(updates.items(), 1):
                        if col in ("price_after_1h", "price_after_4h", "price_after_24h"):
                            set_clauses.append(f"{col} = ${i}::jsonb")
                        else:
                            set_clauses.append(f"{col} = ${i}")
                        params.append(val)

                    params.append(memory_id)
                    params.append(market_ticker)

                    await conn.execute(
                        f"UPDATE news_price_impacts SET {', '.join(set_clauses)} "
                        f"WHERE news_memory_id = ${len(params) - 1} AND market_ticker = ${len(params)}",
                        *params,
                    )
                    updated += 1

            if created or updated:
                logger.info(f"[NEWS_IMPACT] Backfill pass: {created} created, {updated} updated")

    def _get_market_snapshot(self, market_ticker: str) -> Optional[dict]:
        """Get enriched price snapshot for a market from the live index."""
        if not self._index:
            return None
        for event in self._index.events.values():
            market = event.markets.get(market_ticker)
            if market and market.yes_mid is not None:
                return {
                    "yes_bid": market.yes_bid,
                    "yes_ask": market.yes_ask,
                    "yes_mid": market.yes_mid,
                    "spread": market.spread,
                    "volume_5m": market.micro.volume_5m,
                    "book_imbalance": round(market.micro.book_imbalance, 3),
                    "open_interest": getattr(market, "open_interest", None),
                    "ts": time.time(),
                }
        return None

    @staticmethod
    def _classify_magnitude(max_change: float) -> str:
        """Classify price impact magnitude from max absolute change in cents."""
        if max_change < 2:
            return "none"
        if max_change < 5:
            return "small"
        if max_change < 10:
            return "medium"
        return "large"
