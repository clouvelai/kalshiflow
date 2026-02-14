"""
SingleArbCoordinator - Wires EventArbIndex + EventArbMonitor + ArbCaptain.

Startup sequence:
1. Create EventArbIndex
2. Load events via REST
3. Initialize Tavily search service
4. Create UnderstandingBuilder (LLM builds deferred)
5. Subscribe market tickers to orderbook WS
6. Create session order group
7. Setup gateway (KalshiGateway + EventBridge + TradingSession + Sniper)
8. Setup V2 tools (ToolContext with SessionMemoryStore + ContextBuilder)
9. Create EventArbMonitor
9b. Prefetch all orderbooks via REST (populates index before Captain starts)
10. Check exchange status + create Captain
11. Launch deferred init (understanding builds)
12. Start background loops (exchange monitor, order tracker)
13. Broadcast initial snapshot
"""

import asyncio
import logging
import os
import time
from typing import Dict, Optional, Tuple

from .index import EventArbIndex
from .monitor import EventArbMonitor
from .mentions_models import configure as configure_models
from .event_understanding import UnderstandingBuilder

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.coordinator")

# Default memory data directory
DEFAULT_MEMORY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "memory", "data"
)


class SingleArbCoordinator:
    """
    Coordinator for the single-event arb system.

    Wires all components together and manages lifecycle.
    """

    def __init__(
        self,
        config,
        event_bus,
        websocket_manager,
        orderbook_integration,
        trading_client=None,
        health_monitor=None,
        lifecycle_markets=None,
    ):
        self._config = config
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._orderbook_integration = orderbook_integration
        self._trading_client = trading_client
        self._health_monitor = health_monitor
        self._lifecycle_markets = lifecycle_markets  # TrackedMarketsState from lifecycle service

        self._index: Optional[EventArbIndex] = None
        self._monitor: Optional[EventArbMonitor] = None
        self._captain = None
        self._sniper = None  # Sniper execution layer
        self._health_service = None  # AccountHealthService
        self._attention_router = None  # AttentionRouter
        self._auto_actions = None  # AutoActionManager
        self._market_gateway = None  # Prod gateway for hybrid data mode

        # Market maker (QuoteEngine folded into Captain's ecosystem)
        self._mm_index = None       # MMIndex
        self._mm_monitor = None     # MMMonitor
        self._quote_engine = None   # QuoteEngine
        self._mm_attention = None   # MMAttentionRouter
        self._mm_order_group_id: Optional[str] = None
        self._mm_position_sync_task: Optional[asyncio.Task] = None

        self._discovery = None  # SeriesDiscovery
        self._order_group_id: Optional[str] = None
        self._understanding_builder: Optional[UnderstandingBuilder] = None
        self._tavily_budget = None  # TavilyBudgetManager
        self._search_service = None  # TavilySearchService
        self._deferred_init_task: Optional[asyncio.Task] = None
        self._system_ready = asyncio.Event()  # Set when deferred init completes
        self._running = False
        self._started_at: Optional[float] = None

        # Order lifecycle tracking
        self._order_tracker_task: Optional[asyncio.Task] = None
        self._tracked_orders: Dict[str, dict] = {}

        # News-price impact tracking
        self._news_impact_task: Optional[asyncio.Task] = None

        # Decision ledger
        self._decision_ledger = None  # DecisionLedger
        self._decision_backfill_task: Optional[asyncio.Task] = None

        # News ingestion + article analysis
        self._news_ingestion = None  # NewsIngestionService
        self._article_analyzer = None  # ArticleAnalyzer

        # Swing detection + news correlation
        self._swing_detector = None  # SwingDetector
        self._swing_news_loop = None  # SwingNewsLoop

        # Exchange status monitoring
        self._exchange_active: bool = True
        self._exchange_check_interval: float = 60.0  # Check every 60 seconds
        self._exchange_monitor_task: Optional[asyncio.Task] = None
        self._exchange_last_check: Optional[float] = None
        self._exchange_error_count: int = 0
        self._captain_paused_by_exchange: bool = False  # Track if we paused due to exchange

    async def start(self) -> None:
        """Start the single-arb system."""
        if self._running:
            return

        logger.info("Starting single-event arb system...")

        # 0. Configure centralized model tiers (must happen before any LLM consumer)
        configure_models(self._config)

        # 1. Create index
        fee_per_contract = getattr(self._config, "single_arb_fee_per_contract", 1)
        min_edge = getattr(self._config, "single_arb_min_edge_cents", 3.0)
        self._index = EventArbIndex(
            fee_per_contract_cents=fee_per_contract,
            min_edge_cents=min_edge,
        )

        # 2. Setup trading client + market data client
        client = self._get_trading_client()
        if not client:
            logger.error("No trading client available for single-arb REST calls")
            return

        # Create market data client (prod gateway in hybrid mode, demo client otherwise)
        if self._config.hybrid_data_mode:
            from ..gateway import KalshiGateway, MarketDataAdapter

            self._market_gateway = KalshiGateway(
                api_url=self._config.prod_api_url,
                ws_url=self._config.prod_ws_url,
                subaccount=0,  # Read-only, no trading
                api_key_id=self._config.prod_api_key_id,
                private_key_content=self._config.prod_private_key_content,
            )
            await self._market_gateway.connect()
            market_client = MarketDataAdapter(self._market_gateway)
            logger.info(f"[HYBRID] Prod market gateway connected: {self._config.prod_api_url}")
        else:
            self._market_gateway = None
            market_client = client  # existing demo trading client
        self._market_client = market_client

        # 2b. Subscribe to MARKET_TRACKED early (before loading events)
        # This catches any new lifecycle events that arrive during startup
        try:
            await self._event_bus.subscribe_to_market_tracked(self._on_market_tracked)
            logger.info("[LIFECYCLE_BRIDGE] Subscribed to MARKET_TRACKED events")
        except Exception as e:
            logger.warning(f"[LIFECYCLE_BRIDGE] Subscription failed: {e}")

        # 2c. Create discovery container (passive — no REST polling, no background loop)
        from .discovery import TopVolumeDiscovery

        seed_events = getattr(self._config, "discovery_seed_events", [])
        legacy_seeds = getattr(self._config, "single_arb_event_tickers", [])
        all_seeds = list(dict.fromkeys(seed_events + legacy_seeds))

        self._discovery = TopVolumeDiscovery(
            index=self._index,
            trading_client=market_client,
            event_count=9999,  # No limit — lifecycle controls breadth
            seed_event_tickers=all_seeds,
            max_markets_per_event=getattr(self._config, "discovery_max_markets_per_event", 50),
            refresh_interval=getattr(self._config, "discovery_refresh_interval", 300.0),
            subscribe_callback=self._subscribe_new_markets,
            unsubscribe_callback=self._unsubscribe_markets,
            broadcast_callback=self._broadcast,
        )

        # 2d. Load events from lifecycle (sole event source — no REST discovery polling)
        loaded_events = await self._catchup_lifecycle_events(market_client)

        if loaded_events == 0 and not self._index.events:
            logger.warning("No events from lifecycle - single-arb system starting with empty index")

        await self._broadcast_startup(f"Loaded {loaded_events} events from lifecycle ({len(self._index.market_tickers)} markets)", 1, 10)

        # 3. Initialize Tavily search service (sync object creation only)
        if self._config.tavily_enabled and self._config.tavily_api_key:
            from .tavily_budget import TavilyBudgetManager
            from .tavily_service import TavilySearchService
            self._tavily_budget = TavilyBudgetManager(
                monthly_limit=self._config.tavily_monthly_budget,
            )
            self._search_service = TavilySearchService(
                api_key=self._config.tavily_api_key,
                budget_manager=self._tavily_budget,
                search_depth=self._config.tavily_search_depth,
                max_results=self._config.tavily_max_results,
            )
            logger.info(
                f"[TAVILY] Search service initialized "
                f"(depth={self._config.tavily_search_depth}, "
                f"budget={self._config.tavily_monthly_budget})"
            )

        await self._broadcast_startup("Search service ready", 2, 10)

        # 4. Create UnderstandingBuilder (sync, no LLM calls yet)
        understanding_cache_dir = os.path.join(DEFAULT_MEMORY_DIR, "understanding")
        self._understanding_builder = UnderstandingBuilder(
            cache_dir=understanding_cache_dir,
            search_service=self._search_service,
        )
        logger.info("[UNDERSTANDING] Builder created (LLM builds deferred)")

        # 5. Subscribe all market tickers to orderbook WS
        # In hybrid mode, the prod gateway handles all market data — skip legacy integration
        market_tickers = self._index.market_tickers
        if market_tickers and self._orderbook_integration and not self._config.hybrid_data_mode:
            for ticker in market_tickers:
                try:
                    await self._orderbook_integration.subscribe_market(ticker)
                except Exception as e:
                    logger.warning(f"Failed to subscribe {ticker} to orderbook: {e}")

            logger.info(f"Subscribed {len(market_tickers)} markets to orderbook WS")

        await self._broadcast_startup(f"Subscribed {len(market_tickers)} markets to orderbook", 3, 10)

        # 6. Create session order group for the captain
        self._order_group_id = None
        try:
            resp = await client.create_order_group(contracts_limit=10000)
            self._order_group_id = resp.get("order_group_id")
            if self._order_group_id:
                logger.info(f"[SINGLE_ARB] Order group created: {self._order_group_id[:8]}...")
            else:
                logger.warning("[SINGLE_ARB] Order group response missing order_group_id")
        except Exception as e:
            logger.warning(f"[SINGLE_ARB] Order group creation failed: {e}")

        await self._broadcast_startup("Order group created", 4, 10)

        # 7. Setup gateway (KalshiGateway + EventBridge + TradingSession + Sniper + WS)
        order_ttl = getattr(self._config, "single_arb_order_ttl", 60)
        await self._setup_gateway(client, order_ttl)

        await self._broadcast_startup("Gateway connected", 5, 10)

        # 8. Setup V2 tools (ToolContext + SessionMemoryStore + ContextBuilder)
        self._setup_tools(order_ttl)

        await self._broadcast_startup("Captain tools wired", 6, 10)

        # 8b. Create NewsIngestionService + ArticleAnalyzer (optional)
        if self._search_service and self._config.news_ingestion_enabled:
            try:
                from .article_analyzer import ArticleAnalyzer
                from .news_ingestion import NewsIngestionService

                self._article_analyzer = ArticleAnalyzer()

                # Get session memory from tool context
                from .tools import get_context
                ctx = get_context()
                session_memory = ctx.memory if ctx else None

                if session_memory:
                    self._news_ingestion = NewsIngestionService(
                        search_service=self._search_service,
                        memory_store=session_memory,
                        index=self._index,
                        budget_manager=self._tavily_budget,
                        config={
                            "enabled": self._config.news_ingestion_enabled,
                            "max_credits_per_cycle": self._config.news_max_credits_per_cycle,
                            "extract_top_n": self._config.news_extract_top_n,
                        },
                        article_analyzer=self._article_analyzer,
                    )
                    logger.info("[NEWS_INGESTION] Service created (will start after background loops)")
                else:
                    logger.warning("[NEWS_INGESTION] No session memory available, skipping")
            except Exception as e:
                logger.warning(f"[NEWS_INGESTION] Setup failed: {e}")

        # 8c. Create SwingDetector + SwingNewsService + SwingNewsLoop
        if self._config.swing_detection_enabled and self._search_service:
            try:
                from .swing_detector import SwingDetector
                from .swing_news_service import SwingNewsService, SwingNewsLoop
                from .tools import get_context

                ctx = get_context()
                session_memory = ctx.memory if ctx else None

                if session_memory:
                    self._swing_detector = SwingDetector(
                        min_change_cents=self._config.swing_min_change_cents,
                        volume_multiplier=self._config.swing_volume_multiplier,
                        live_window_seconds=self._config.swing_live_window_seconds,
                    )

                    swing_news_service = SwingNewsService(
                        search_service=self._search_service,
                        memory_store=session_memory,
                        index=self._index,
                        budget_manager=self._tavily_budget,
                        article_analyzer=self._article_analyzer,
                    )

                    self._swing_news_loop = SwingNewsLoop(
                        swing_detector=self._swing_detector,
                        swing_news_service=swing_news_service,
                        index=self._index,
                        config=self._config,
                        candle_fetch_callback=self._fetch_all_candlesticks,
                    )

                    # Wire swing_detector into ToolContext
                    if ctx:
                        ctx.swing_detector = self._swing_detector

                    logger.info(
                        f"[SWING] SwingDetector + SwingNewsLoop created "
                        f"(min_change={self._config.swing_min_change_cents}c, "
                        f"volume_mult={self._config.swing_volume_multiplier}x)"
                    )
                else:
                    logger.warning("[SWING] No session memory available, skipping swing detection")
            except Exception as e:
                logger.warning(f"[SWING] Setup failed: {e}")

        # 9. Create monitor
        self._monitor = EventArbMonitor(
            index=self._index,
            event_bus=self._event_bus,
            trading_client=market_client,
            config=self._config,
            broadcast_callback=self._broadcast,
            opportunity_callback=self._on_opportunity,
        )
        await self._monitor.start()

        # 9b. Prefetch all orderbooks via REST so index is ready immediately
        #     (avoids 30-60s wait for WS snapshots on thin/inactive markets)
        await self._broadcast_startup("Fetching orderbook data via REST", 7, 10)
        try:
            prefetched = await self._monitor.prefetch_all_orderbooks()
            logger.info(f"[PREFETCH] {prefetched} markets prefetched, index ready={self._index.is_ready}")
        except Exception as e:
            logger.warning(f"[PREFETCH] Orderbook prefetch failed: {e}")

        # 9c. Create AttentionRouter + AutoActionManager
        try:
            from .attention import AttentionRouter
            from .auto_actions import AutoActionManager

            self._auto_actions = AutoActionManager(
                gateway=self._gateway,
                index=self._index,
                sniper=self._sniper,
                config=self._config,
                broadcast_callback=self._broadcast,
            )
            self._attention_router = AttentionRouter(
                index=self._index,
                config=self._config,
                auto_action_callback=self._auto_actions.on_attention_item,
                broadcast_callback=self._broadcast,
            )
            self._attention_router.subscribe(self._event_bus)
            await self._attention_router.start()

            # Wire attention callback into Sniper (deferred from step 7)
            if self._sniper:
                self._sniper._attention_callback = self._attention_router.inject_item

            # Wire auto_actions into ToolContext
            from .tools import _ctx as tool_ctx
            if tool_ctx:
                tool_ctx.auto_actions = self._auto_actions

            # Wire SwingDetector to EventBus for live BBO updates
            if self._swing_detector:
                from ..core.events.types import EventType

                async def _on_bbo_for_swing(market_ticker: str, metadata: dict) -> None:
                    yes_mid = metadata.get("yes_mid")
                    if yes_mid is None:
                        return
                    event_ticker = self._index.get_event_for_ticker(market_ticker)
                    if not event_ticker:
                        return
                    event = self._index.events.get(event_ticker)
                    market = event.markets.get(market_ticker) if event else None
                    title = market.title if market and market.title else market_ticker

                    swing = self._swing_detector.on_bbo_update(
                        event_ticker=event_ticker,
                        market_ticker=market_ticker,
                        market_title=title,
                        yes_mid=yes_mid,
                    )
                    if swing and self._attention_router:
                        from .models import AttentionItem
                        score = min(95, 65 + swing.change_cents * 3)
                        self._attention_router.inject_item(AttentionItem(
                            event_ticker=event_ticker,
                            market_ticker=market_ticker,
                            category="live_swing",
                            score=score,
                            summary=f"{swing.direction} {swing.change_cents:.0f}c swing on {market_ticker}",
                            data={"swing_source": "live", "change_cents": swing.change_cents},
                        ))

                self._event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, _on_bbo_for_swing)
                self._event_bus.subscribe(EventType.TICKER_UPDATE, _on_bbo_for_swing)
                logger.info("[SWING] SwingDetector wired to EventBus (live BBO tracking)")

            logger.info("[ATTENTION] AttentionRouter + AutoActionManager wired")
        except Exception as e:
            logger.warning(f"[ATTENTION] AttentionRouter setup failed, Captain will use legacy mode: {e}")
            self._attention_router = None
            self._auto_actions = None

        # 9d. Setup QuoteEngine (MM folded into Captain) if enabled
        if self._config.mm_enabled:
            try:
                await self._setup_quote_engine()
                await self._broadcast_startup("QuoteEngine started", 8, 11)
            except Exception as e:
                logger.error(f"[MM] QuoteEngine setup failed: {e}", exc_info=True)

        # 10. Check exchange status before starting Captain
        is_exchange_active, exchange_error = await self._check_exchange_status()
        self._exchange_active = is_exchange_active

        # Broadcast initial exchange status
        await self._broadcast({
            "type": "exchange_status",
            "data": {
                "active": is_exchange_active,
                "error": exchange_error,
                "last_check": self._exchange_last_check,
            },
        })

        exchange_label = "active" if is_exchange_active else "down"
        await self._broadcast_startup(f"Exchange: {exchange_label}", 8, 10)

        if not is_exchange_active:
            logger.warning(f"[SINGLE_ARB] Exchange not active: {exchange_error}")
            logger.warning("[SINGLE_ARB] Captain will NOT start until exchange is available")

        # 10b. Clear stale FAISS session memory for a fresh start
        try:
            from .tools import get_context
            ctx = get_context()
            if ctx and ctx.memory:
                ctx.memory._faiss_store = None
                ctx.memory._faiss_docs = []
                ctx.memory._faiss_ready = False
                ctx.memory._journal.clear()
                logger.info("[CAPTAIN] Cleared session memory (FAISS + journal) for fresh start")
        except Exception as e:
            logger.debug(f"[CAPTAIN] Session memory clear failed: {e}")

        # 11. Create and start Captain (if enabled AND exchange is active)
        captain_enabled = getattr(self._config, "single_arb_captain_enabled", True)
        if captain_enabled:
            try:
                captain_interval = getattr(self._config, "single_arb_captain_interval", 60.0)

                from .captain import ArbCaptain
                from .context_builder import ContextBuilder

                ctx_builder = ContextBuilder(index=self._index, subaccount=self._config.subaccount)
                self._captain = ArbCaptain(
                    context_builder=ctx_builder,
                    attention_router=self._attention_router,
                    config=self._config,
                    cycle_interval=captain_interval,
                    event_callback=self._emit_agent_event,
                    sniper_ref=self._sniper,
                    system_ready=self._system_ready,
                )
                mode = "attention-driven" if self._attention_router else "legacy (fixed-interval)"
                logger.info(f"[CAPTAIN] Using single-agent Captain, mode={mode}")

                if is_exchange_active:
                    await self._captain.start()
                    logger.info("ArbCaptain started")
                else:
                    # Captain created but paused, waiting for exchange
                    self._captain._paused = True
                    self._captain_paused_by_exchange = True
                    await self._captain.start()  # Start loop but paused
                    logger.info("ArbCaptain created but paused (waiting for exchange)")
                    await self._broadcast({
                        "type": "captain_paused",
                        "data": {"paused": True, "reason": "exchange_down"},
                    })
            except Exception as e:
                logger.error(f"Failed to start ArbCaptain: {e}")

        await self._broadcast_startup("Captain started", 9, 10)

        # 12. Launch deferred initialization (understanding builds) in background
        self._deferred_init_task = asyncio.create_task(self._run_deferred_init())
        logger.info("[DEFERRED_INIT] Background initialization launched")

        # 13. Start background loops
        self._exchange_monitor_task = asyncio.create_task(self._exchange_monitor_loop())
        self._order_tracker_task = asyncio.create_task(self._order_tracker_loop())
        self._news_impact_task = asyncio.create_task(self._news_impact_tracker_loop())
        self._decision_backfill_task = asyncio.create_task(self._decision_backfill_loop())

        # Discovery is passive — lifecycle is the sole event source via MARKET_TRACKED
        # No background REST polling (discovery.start() not called)
        if self._health_service:
            await self._health_service.start()
            logger.info("[HEALTH] AccountHealthService background loop started")
        if self._news_ingestion:
            await self._news_ingestion.start()
            logger.info("[NEWS_INGESTION] Background news polling started")
        if self._swing_news_loop:
            await self._swing_news_loop.start()
            logger.info("[SWING] SwingNewsLoop background service started")

        # 14. Broadcast initial snapshot
        await self._broadcast_snapshot()

        # 15. Broadcast gateway config (hybrid mode awareness for frontend)
        await self._broadcast_gateway_config()

        # Register with health monitor
        if self._health_monitor:
            self._health_monitor.register_component(
                "single_arb", self, critical=False
            )

        self._running = True
        self._started_at = time.time()

        logger.info(
            f"[SINGLE_ARB:STARTUP] Single-event arb system started: "
            f"events={loaded_events} markets={len(market_tickers)} captain={captain_enabled}"
        )

    async def stop(self) -> None:
        """Stop the single-arb system."""
        self._running = False

        # Cancel all background tasks (including MM position sync)
        for attr in ("_order_tracker_task", "_exchange_monitor_task",
                      "_news_impact_task", "_deferred_init_task",
                      "_decision_backfill_task", "_mm_position_sync_task"):
            task = getattr(self, attr, None)
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                setattr(self, attr, None)

        # Flush pending memory writes before stopping components
        try:
            from .tools import get_context
            ctx = get_context()
            if ctx and ctx.memory:
                await ctx.memory.flush(timeout=5.0)
        except Exception as e:
            logger.debug(f"Memory flush during shutdown: {e}")

        if self._swing_news_loop:
            await self._swing_news_loop.stop()
        if self._news_ingestion:
            await self._news_ingestion.stop()
        if self._discovery:
            await self._discovery.stop()
        if self._attention_router:
            await self._attention_router.stop()
        if self._health_service:
            await self._health_service.stop()
        # Stop MM components
        if self._quote_engine:
            await self._quote_engine.stop()
            logger.info("[MM] QuoteEngine stopped")
        if self._mm_monitor:
            await self._mm_monitor.stop()
            logger.info("[MM] MMMonitor stopped")

        if self._sniper:
            await self._sniper.stop()
        if self._captain:
            await self._captain.stop()
        if self._monitor:
            await self._monitor.stop()

        # Disconnect gateways
        if getattr(self, "_market_gateway", None):
            try:
                await self._market_gateway.disconnect()
                logger.info("[GATEWAY] Prod market gateway disconnected")
            except Exception as e:
                logger.debug(f"Prod gateway disconnect error: {e}")

        if getattr(self, "_gateway", None):
            try:
                await self._gateway.disconnect()
                logger.info("[GATEWAY] Demo KalshiGateway disconnected")
            except Exception as e:
                logger.debug(f"Gateway disconnect error: {e}")

        # Reset trading session if present
        if getattr(self, "_trading_session", None):
            self._trading_session.reset()

        # Reset order groups (cancels all resting orders in groups)
        client = self._get_trading_client()
        for group_id, label in [
            (getattr(self, "_order_group_id", None), "Captain"),
            (getattr(self, "_mm_order_group_id", None), "MM"),
        ]:
            if group_id:
                try:
                    if client:
                        await client.reset_order_group(group_id)
                        logger.info(f"[SINGLE_ARB] {label} order group reset: {group_id[:8]}...")
                except Exception as e:
                    logger.debug(f"{label} order group reset failed: {e}")

        logger.info("[SINGLE_ARB:SHUTDOWN] Single-event arb system stopped")

    def _get_trading_client(self):
        """Get the underlying trading client for REST calls."""
        if self._trading_client:
            # V3TradingClientIntegration wraps the actual client
            if hasattr(self._trading_client, "_client"):
                return self._trading_client._client
            return self._trading_client
        return None

    # ------------------------------------------------------------------ #
    #  Lifecycle bridge: load lifecycle-discovered events into arb index  #
    # ------------------------------------------------------------------ #

    async def _catchup_lifecycle_events(self, market_client) -> int:
        """Load events already discovered by lifecycle into EventArbIndex.

        This is the sole event source — replaces REST discovery polling.
        Reads unique event_tickers from TrackedMarketsState and loads each
        into the index via REST (for full event structure with markets).

        Returns:
            Number of events loaded.
        """
        if not self._lifecycle_markets:
            logger.warning("[LIFECYCLE_BRIDGE] No lifecycle markets available, falling back to REST discovery")
            return await self._discovery.discover()

        # Get unique event_tickers from lifecycle's tracked markets
        events_by_ticker = self._lifecycle_markets.get_markets_by_event()
        event_tickers = [et for et in events_by_ticker.keys() if et and et != "unknown"]

        if not event_tickers:
            logger.warning("[LIFECYCLE_BRIDGE] Lifecycle has no tracked events yet")
            return 0

        logger.info(f"[LIFECYCLE_BRIDGE] Catching up {len(event_tickers)} events from lifecycle...")

        loaded = 0
        skipped_size = 0
        errors = 0
        max_markets = getattr(self._config, "discovery_max_markets_per_event", 50)

        for event_ticker in event_tickers:
            if event_ticker in self._index.events:
                loaded += 1  # Already loaded
                continue

            try:
                meta = await self._index.load_event(event_ticker, market_client)
                if not meta:
                    errors += 1
                    continue

                if len(meta.markets) > max_markets:
                    self._index._events.pop(event_ticker, None)
                    for t in list(meta.markets.keys()):
                        self._index._ticker_to_event.pop(t, None)
                    skipped_size += 1
                    logger.debug(f"[LIFECYCLE_BRIDGE] Skipped {event_ticker}: {len(meta.markets)} markets > {max_markets}")
                    continue

                # Track in discovery container for stats/snapshots
                if self._discovery:
                    self._discovery.add_external_event(event_ticker)

                loaded += 1
                logger.debug(f"[LIFECYCLE_BRIDGE] Loaded {event_ticker}: {meta.title} ({len(meta.markets)} mkts)")

            except Exception as e:
                errors += 1
                logger.warning(f"[LIFECYCLE_BRIDGE] Error loading {event_ticker}: {e}")

            # Rate limit protection (Kalshi 10 req/s)
            if loaded % 8 == 0 and loaded > 0:
                await asyncio.sleep(0.5)

        logger.info(
            f"[LIFECYCLE_BRIDGE] Catch-up complete: {loaded} events loaded, "
            f"{skipped_size} skipped (too large), {errors} errors "
            f"({len(self._index.market_tickers)} total markets)"
        )

        return loaded

    async def _on_market_tracked(self, event) -> None:
        """Bridge lifecycle-discovered events into EventArbIndex."""
        market_info = event.market_info or {}
        event_ticker = market_info.get("event_ticker", "")
        if not event_ticker or event_ticker in self._index.events:
            return
        # Debounce: schedule deferred load (let sibling markets arrive)
        asyncio.create_task(self._load_lifecycle_event(event_ticker))

    async def _load_lifecycle_event(self, event_ticker: str) -> None:
        """Load a lifecycle event into the arb index after brief debounce."""
        await asyncio.sleep(3.0)  # Let sibling markets arrive
        if event_ticker in self._index.events:
            return
        try:
            client = getattr(self, "_market_client", None) or self._get_trading_client()
            if not client:
                return
            meta = await self._index.load_event(event_ticker, client)
            if not meta:
                return
            if len(meta.markets) > 50:  # Skip oversized
                self._index._events.pop(event_ticker, None)
                return
            if self._discovery:
                self._discovery.add_external_event(event_ticker)
            new_tickers = list(meta.markets.keys())
            if new_tickers:
                await self._subscribe_new_markets(new_tickers)
            await self._broadcast_snapshot()
            logger.info(
                f"[LIFECYCLE_BRIDGE] Loaded {event_ticker}: {meta.title} "
                f"({len(meta.markets)} mkts)"
            )
        except Exception as e:
            logger.warning(f"[LIFECYCLE_BRIDGE] Error loading {event_ticker}: {e}")

    # ------------------------------------------------------------------ #
    #  Discovery: subscribe new markets to WS channels                    #
    # ------------------------------------------------------------------ #

    async def _subscribe_new_markets(self, market_tickers: list) -> None:
        """Subscribe newly discovered market tickers to orderbook WS + gateway channels.

        Also prefetches orderbooks via REST so new markets have data immediately
        instead of waiting 30-60s+ for WS snapshots.
        """
        if not market_tickers:
            return

        # Subscribe to orderbook integration (V3 event bus)
        # In hybrid mode, the prod gateway handles all market data — skip legacy integration
        if self._orderbook_integration and not self._config.hybrid_data_mode:
            for ticker in market_tickers:
                try:
                    await self._orderbook_integration.subscribe_market(ticker)
                except Exception as e:
                    logger.warning(f"[DISCOVERY] Failed to subscribe {ticker} to orderbook: {e}")

        # Subscribe to gateway WS channels (prod in hybrid mode, demo otherwise)
        ws_gateway = getattr(self, "_market_gateway", None) or getattr(self, "_gateway", None)
        if ws_gateway:
            ws_mux = ws_gateway.get_ws()
            for channel in ("orderbook_delta", "ticker", "trade"):
                try:
                    await ws_mux.subscribe_tickers(channel, market_tickers)
                except Exception as e:
                    logger.warning(f"[DISCOVERY] Failed to subscribe {channel}: {e}")

        logger.info(f"[DISCOVERY] Subscribed {len(market_tickers)} new markets to WS channels")

        # Prefetch orderbooks via REST so index has data immediately
        # Use market_client (prod adapter in hybrid mode) for accurate data
        client = getattr(self, "_market_client", None) or self._get_trading_client()
        if client and self._index:
            prefetched = 0
            for ticker in market_tickers:
                try:
                    resp = await client.get_orderbook(ticker, depth=5)
                    if not resp:
                        continue
                    orderbook = resp.get("orderbook", resp)
                    if not orderbook or not isinstance(orderbook, dict):
                        continue

                    yes_levels = orderbook.get("yes") or []
                    no_levels = orderbook.get("no") or []

                    self._index.on_orderbook_update(
                        market_ticker=ticker,
                        yes_levels=yes_levels,
                        no_levels=no_levels,
                        source="api",
                    )
                    prefetched += 1
                except Exception as e:
                    logger.debug(f"[DISCOVERY] Prefetch failed for {ticker}: {e}")

                # Rate limit protection (Kalshi 10 req/s)
                if prefetched % 8 == 0 and prefetched > 0:
                    await asyncio.sleep(0.5)

            if prefetched:
                logger.info(f"[DISCOVERY] Prefetched {prefetched}/{len(market_tickers)} new markets via REST")

    async def _unsubscribe_markets(self, market_tickers: list) -> None:
        """Unsubscribe evicted market tickers from orderbook WS + gateway channels."""
        if not market_tickers:
            return

        if self._orderbook_integration and not self._config.hybrid_data_mode:
            for ticker in market_tickers:
                try:
                    await self._orderbook_integration.unsubscribe_market(ticker)
                except Exception as e:
                    logger.warning(f"[DISCOVERY] Failed to unsubscribe {ticker} from orderbook: {e}")

        ws_gateway = getattr(self, "_market_gateway", None) or getattr(self, "_gateway", None)
        if ws_gateway:
            ws_mux = ws_gateway.get_ws()
            for channel in ("orderbook_delta", "ticker", "trade"):
                try:
                    await ws_mux.unsubscribe_tickers(channel, market_tickers)
                except Exception as e:
                    logger.warning(f"[DISCOVERY] Failed to unsubscribe {channel}: {e}")

        logger.info(f"[DISCOVERY] Unsubscribed {len(market_tickers)} evicted markets from WS channels")

    # ------------------------------------------------------------------ #
    #  Sniper → Captain order registration                                #
    # ------------------------------------------------------------------ #

    def _register_sniper_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        action: str,
        contracts: int,
        price_cents: int,
        ttl_seconds: int,
    ) -> None:
        """Register a sniper order into the Captain's order tracking system.

        Called by Sniper._execute_arb() for each successful leg. This ensures
        the order tracker (_poll_order_statuses) picks up sniper orders for:
        - Status polling and frontend broadcasts
        - Capital release via _release_sniper_capital
        - Trade outcome recording to memory
        """
        from .tools import get_context

        ctx = get_context()
        if not ctx:
            logger.warning(
                f"[SNIPER:REGISTER] ToolContext not ready, order {order_id[:8]}... "
                f"will not be tracked (ticker={ticker})"
            )
            return

        ctx.captain_order_ids.add(order_id)
        self._trading_session.captain_order_ids.add(order_id)
        ctx.order_initial_states[order_id] = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "contracts": contracts,
            "price_cents": price_cents,
            "placed_at": time.time(),
            "ttl_seconds": ttl_seconds,
            "status": "placed",
            "source": "sniper",
        }

        # Record to decision ledger (fire-and-forget)
        if self._decision_ledger:
            event_ticker = self._index.get_event_for_ticker(ticker) if self._index else None
            asyncio.create_task(self._decision_ledger.record_decision(
                order_id=order_id, source="sniper",
                event_ticker=event_ticker, market_ticker=ticker,
                side=side, action=action, contracts=contracts,
                limit_price_cents=price_cents,
            ))

    # ------------------------------------------------------------------ #
    #  Exchange status monitoring                                         #
    # ------------------------------------------------------------------ #

    async def _check_exchange_status(self) -> Tuple[bool, Optional[str]]:
        """
        Check if the exchange is active and available for trading.

        Returns:
            Tuple of (is_active, error_message)
            - (True, None) if exchange is active
            - (False, error_message) if exchange is down or unreachable
        """
        client = self._get_trading_client()
        if not client:
            return False, "No trading client available"

        try:
            status = await client.get_exchange_status()
            self._exchange_last_check = time.time()

            exchange_active = status.get("exchange_active", False)
            trading_active = status.get("trading_active", False)

            if not exchange_active:
                resume_time = status.get("exchange_estimated_resume_time", "unknown")
                return False, f"Exchange inactive (estimated resume: {resume_time})"

            if not trading_active:
                return False, "Trading temporarily disabled"

            self._exchange_error_count = 0
            return True, None

        except Exception as e:
            self._exchange_error_count += 1
            error_str = str(e)

            # Check for 5xx errors indicating server issues
            if "503" in error_str or "502" in error_str or "500" in error_str:
                return False, f"Exchange server error: {error_str}"
            if "5" in error_str and ("error" in error_str.lower() or "status" in error_str.lower()):
                return False, f"Exchange server error: {error_str}"

            # After 3 consecutive errors, treat as exchange down
            if self._exchange_error_count >= 3:
                return False, f"Exchange unreachable after {self._exchange_error_count} attempts: {error_str}"

            # Transient error, don't pause yet
            logger.warning(f"[SINGLE_ARB] Exchange check error (attempt {self._exchange_error_count}): {e}")
            return True, None

    async def _exchange_monitor_loop(self) -> None:
        """
        Background task that monitors exchange status every 60 seconds.

        Pauses Captain if exchange goes down, resumes when it comes back.
        """
        logger.info("[SINGLE_ARB] Exchange monitor started")

        while self._running:
            try:
                await asyncio.sleep(self._exchange_check_interval)

                if not self._running:
                    break

                is_active, error_msg = await self._check_exchange_status()

                # Update state
                was_active = self._exchange_active
                self._exchange_active = is_active

                # Broadcast status to frontend
                await self._broadcast({
                    "type": "exchange_status",
                    "data": {
                        "active": is_active,
                        "error": error_msg,
                        "last_check": self._exchange_last_check,
                    },
                })

                # Handle state transitions
                if was_active and not is_active:
                    # Exchange went down - pause Captain
                    logger.warning(f"[SINGLE_ARB] Exchange DOWN: {error_msg}")
                    if self._captain and not self._captain.is_paused:
                        self._captain.pause()
                        self._captain_paused_by_exchange = True
                        logger.info("[SINGLE_ARB] Captain paused due to exchange down")
                        await self._broadcast({
                            "type": "captain_paused",
                            "data": {"paused": True, "reason": "exchange_down"},
                        })

                elif not was_active and is_active:
                    # Exchange came back up - resume Captain if we paused it
                    logger.info("[SINGLE_ARB] Exchange UP: resuming operations")
                    if self._captain and self._captain_paused_by_exchange:
                        self._captain.resume()
                        self._captain_paused_by_exchange = False
                        logger.info("[SINGLE_ARB] Captain resumed after exchange recovery")
                        await self._broadcast({
                            "type": "captain_paused",
                            "data": {"paused": False, "reason": "exchange_recovered"},
                        })

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SINGLE_ARB] Exchange monitor error: {e}")
                await asyncio.sleep(10.0)  # Brief pause on error

        logger.info("[SINGLE_ARB] Exchange monitor stopped")

    # ------------------------------------------------------------------ #
    #  Order lifecycle tracking                                           #
    # ------------------------------------------------------------------ #

    async def _order_tracker_loop(self) -> None:
        """Background task that polls order statuses every 15 seconds.

        Tracks Captain orders through their lifecycle (placed -> resting -> executed/expired/cancelled)
        and broadcasts status changes to the frontend.
        Also performs periodic cleanup of stale tracked orders and releases sniper capital.
        """
        logger.info("[ORDER_TRACKER] Started (15s interval)")
        cleanup_counter = 0
        event_cleanup_counter = 0

        while self._running:
            try:
                await asyncio.sleep(15)
                if not self._running:
                    break
                await self._poll_order_statuses()

                # Release sniper capital every poll (pure Python, zero I/O)
                if self._sniper:
                    self._release_sniper_capital()

                # Reconcile sniper capital every poll (pure Python, zero I/O)
                if self._sniper and hasattr(self._sniper, "reconcile_capital"):
                    resting_ids = {
                        oid for oid, state in self._tracked_orders.items()
                        if state.get("status") in ("resting", "partial", "placed")
                    }
                    self._sniper.reconcile_capital(resting_ids)

                # Periodic cleanup every ~5 minutes (20 iterations * 15s)
                cleanup_counter += 1
                if cleanup_counter >= 20:
                    cleanup_counter = 0
                    self._cleanup_stale_tracked_orders()
                    from .tools import cleanup_terminal_orders
                    cleanup_terminal_orders(max_age_seconds=86400)

                # Settled event cleanup every ~10 minutes (40 iterations * 15s)
                # Safety net — primary eviction happens via discovery refresh
                event_cleanup_counter += 1
                if event_cleanup_counter >= 40:
                    event_cleanup_counter = 0
                    if self._index:
                        # Collect tickers before cleanup for understanding cache eviction
                        settled_tickers = set()
                        for et, ev in self._index.events.items():
                            if not ev.markets or all(
                                m.status in ("closed", "settled", "finalized")
                                for m in ev.markets.values()
                            ):
                                settled_tickers.add(et)

                        removed = self._index.cleanup_settled_events()
                        if removed:
                            logger.info(f"[ORDER_TRACKER] Cleaned {removed} settled events")
                            # Clean up understanding cache for evicted events
                            if self._understanding_builder and settled_tickers:
                                cleaned = self._understanding_builder.cleanup_cache(settled_tickers)
                                if cleaned:
                                    logger.info(f"[ORDER_TRACKER] Cleaned {cleaned} understanding caches")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[ORDER_TRACKER] Loop error: {e}")
                await asyncio.sleep(5)

        logger.info("[ORDER_TRACKER] Stopped")

    async def _poll_order_statuses(self) -> None:
        """Poll Kalshi API for current order statuses and broadcast changes."""
        from .tools import get_context

        ctx = get_context()
        if not ctx:
            return
        captain_order_ids = ctx.captain_order_ids
        order_initial_states = ctx.order_initial_states

        if not captain_order_ids:
            return

        client = self._get_trading_client()
        if not client:
            return

        try:
            # Fetch all orders (no status filter to catch all states)
            orders_resp = await client.get_orders()
            api_orders = {
                o.get("order_id"): o
                for o in orders_resp.get("orders", [])
            }

            # Fetch fills for our order group
            fills_resp = {}
            if self._order_group_id:
                try:
                    fills_resp = await client.get_fills(order_group_id=self._order_group_id)
                except Exception:
                    pass
            fills_by_order = {}
            for f in fills_resp.get("fills", []):
                oid = f.get("order_id")
                if oid:
                    fills_by_order.setdefault(oid, []).append(f)

            now = time.time()

            for order_id in list(captain_order_ids):
                # Seed from initial state if first seen
                if order_id not in self._tracked_orders:
                    initial = order_initial_states.get(order_id, {})
                    self._tracked_orders[order_id] = {
                        "order_id": order_id,
                        "ticker": initial.get("ticker", ""),
                        "side": initial.get("side", ""),
                        "action": initial.get("action", "buy"),
                        "contracts": initial.get("contracts", 0),
                        "price_cents": initial.get("price_cents", 0),
                        "placed_at": initial.get("placed_at", now),
                        "ttl_seconds": initial.get("ttl_seconds", 60),
                        "status": initial.get("status", "placed"),
                        "fill_count": 0,
                        "remaining_count": initial.get("contracts", 0),
                        "updated_at": now,
                    }

                prev = self._tracked_orders[order_id]
                prev_status = prev["status"]

                # Skip terminal states
                if prev_status in ("executed", "filled", "expired", "canceled", "cancelled"):
                    continue

                api_order = api_orders.get(order_id)
                order_fills = fills_by_order.get(order_id, [])
                total_filled = sum(f.get("count", 0) for f in order_fills)
                original_count = prev["contracts"]

                if api_order:
                    # Order still exists in API
                    api_status = api_order.get("status", "unknown")
                    remaining = api_order.get("remaining_count", original_count - total_filled)

                    if api_status == "executed":
                        new_status = "executed"
                    elif api_status == "resting":
                        new_status = "partial" if total_filled > 0 else "resting"
                    elif api_status == "canceled" or api_status == "cancelled":
                        new_status = "cancelled"
                    else:
                        new_status = api_status
                else:
                    # Order not in API response - determine why
                    remaining = max(0, original_count - total_filled)

                    if total_filled >= original_count:
                        new_status = "executed"
                    elif total_filled > 0:
                        # Partially filled then disappeared (cancelled remainder or expired)
                        new_status = "cancelled"
                    else:
                        # Never filled - check TTL
                        placed_at = prev.get("placed_at", now)
                        ttl = prev.get("ttl_seconds", 60)
                        if now - placed_at > ttl:
                            new_status = "expired"
                        else:
                            new_status = "cancelled"

                # Update tracked state
                self._tracked_orders[order_id].update({
                    "status": new_status,
                    "fill_count": total_filled,
                    "remaining_count": remaining,
                    "updated_at": now,
                })

                # Broadcast if status changed
                if new_status != prev_status:
                    logger.info(
                        f"[ORDER_TRACKER] {order_id[:8]}... {prev_status} -> {new_status} "
                        f"(fills={total_filled}/{original_count})"
                    )
                    await self._broadcast({
                        "type": "order_status_update",
                        "data": {
                            "order_id": order_id,
                            "status": new_status,
                            "previous_status": prev_status,
                            "ticker": prev.get("ticker", ""),
                            "side": prev.get("side", ""),
                            "contracts": original_count,
                            "price_cents": prev.get("price_cents", 0),
                            "fill_count": total_filled,
                            "remaining_count": remaining,
                            "placed_at": prev.get("placed_at"),
                            "updated_at": now,
                        },
                    })

                    # Auto-record trade outcome to memory
                    if new_status in ("executed", "cancelled", "expired"):
                        try:
                            outcome_text = (
                                f"OUTCOME: {prev.get('action', 'buy')} {total_filled}/{original_count} "
                                f"{prev.get('side', '')} {prev.get('ticker', '')} "
                                f"@{prev.get('price_cents', 0)}c -> {new_status}"
                            )
                            asyncio.create_task(ctx.memory.store(
                                content=outcome_text,
                                memory_type="trade_outcome",
                                metadata={
                                    "order_id": order_id,
                                    "ticker": prev.get("ticker", ""),
                                    "trade_result": new_status,
                                    "fill_count": total_filled,
                                },
                            ))
                        except Exception as e:
                            logger.debug(f"[ORDER_TRACKER] Memory store failed: {e}")

        except Exception as e:
            logger.warning(f"[ORDER_TRACKER] Poll error: {e}")

    def _cleanup_stale_tracked_orders(self) -> None:
        """Remove terminal orders older than 24h from _tracked_orders."""
        now = time.time()
        stale = [
            oid for oid, info in self._tracked_orders.items()
            if info.get("status") in ("executed", "filled", "expired", "canceled", "cancelled")
            and now - info.get("placed_at", 0) > 86400
        ]
        for oid in stale:
            del self._tracked_orders[oid]
        if stale:
            logger.info(f"[ORDER_TRACKER:CLEANUP] Removed {len(stale)} stale tracked orders")

    def _release_sniper_capital(self) -> None:
        """Release sniper capital_active for orders in terminal states.

        Scans tracked orders: for orders placed by sniper that have reached terminal
        status (executed, expired, cancelled), release the corresponding capital.
        """
        if not self._sniper:
            return

        sniper_order_ids = getattr(self._sniper._session, "sniper_order_ids", set())
        if not sniper_order_ids:
            return

        released = 0
        for oid in list(self._sniper.state.active_order_ids):
            tracked = self._tracked_orders.get(oid, {})
            status = tracked.get("status", "")
            if status in ("executed", "filled", "expired", "canceled", "cancelled"):
                cost = tracked.get("contracts", 0) * tracked.get("price_cents", 0)
                if cost > 0:
                    self._sniper.state.capital_active = max(0, self._sniper.state.capital_active - cost)
                    released += cost
                self._sniper.state.active_order_ids.discard(oid)

        if released > 0:
            logger.info(f"[SNIPER:CAPITAL_RELEASE] Released {released}c from terminal orders")

    # ------------------------------------------------------------------ #
    #  Decision ledger backfill                                           #
    # ------------------------------------------------------------------ #

    async def _decision_backfill_loop(self) -> None:
        """Background loop that backfills decision outcomes every 60 seconds."""
        await self._system_ready.wait()
        logger.info("[DECISION_LEDGER] Backfill loop started (60s interval)")

        while self._running:
            try:
                await asyncio.sleep(60)
                if not self._running:
                    break
                if self._decision_ledger:
                    await self._decision_ledger.backfill_outcomes(self._tracked_orders)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[DECISION_LEDGER] Backfill loop error: {e}")
                await asyncio.sleep(10)

        logger.info("[DECISION_LEDGER] Backfill loop stopped")

    # ------------------------------------------------------------------ #
    #  News-price impact tracking                                         #
    # ------------------------------------------------------------------ #

    async def _news_impact_tracker_loop(self) -> None:
        """Backfill news_price_impacts by checking price changes after news events.

        Runs every 30 minutes. For each news memory with a price_snapshot:
        - Creates news_price_impacts rows (one per market) if they don't exist
        - Fills in price_after_1h/4h/24h as enough time passes
        - Sets magnitude at each step based on MAX change across filled windows
        """
        await self._system_ready.wait()
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
            # Fetch news memories from last 48h with price_snapshot
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

                # Iterate over market tickers in snapshot (skip _ts metadata key)
                for market_ticker, market_snap in snapshot.items():
                    if market_ticker.startswith("_"):
                        continue
                    if not isinstance(market_snap, dict):
                        continue

                    original_mid = market_snap.get("yes_mid")
                    if original_mid is None:
                        continue

                    # Check if row already exists
                    existing = await conn.fetchrow(
                        """
                        SELECT id, price_after_1h, price_after_4h, price_after_24h
                        FROM news_price_impacts
                        WHERE news_memory_id = $1 AND market_ticker = $2
                        """,
                        memory_id,
                        market_ticker,
                    )

                    if not existing:
                        # CREATE row with price_at_news
                        await conn.execute(
                            """
                            INSERT INTO news_price_impacts (
                                news_memory_id, market_ticker, event_ticker,
                                price_at_news
                            ) VALUES ($1, $2, $3, $4::jsonb)
                            """,
                            memory_id,
                            market_ticker,
                            event_tickers[0] if event_tickers else None,
                            json.dumps(market_snap),
                        )
                        created += 1
                        existing = {"id": None, "price_after_1h": None, "price_after_4h": None, "price_after_24h": None}

                    # Fill windows as time passes
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
                    # Re-fetch to get any changes we just set + existing ones
                    all_changes = []
                    for key in ("change_1h_cents", "change_4h_cents", "change_24h_cents"):
                        val = updates.get(key)
                        if val is not None:
                            all_changes.append(abs(val))
                    # Also check existing row for previously filled windows
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

                        # Write signal_quality back to source memory (feedback loop)
                        # GREATEST ensures quality only goes UP over time
                        signal_map = {"large": 1.0, "medium": 0.8, "small": 0.6, "none": 0.3}
                        sq = signal_map.get(magnitude, 0.5)
                        try:
                            await conn.execute(
                                "UPDATE agent_memories SET signal_quality = GREATEST(signal_quality, $1) WHERE id = $2",
                                sq, memory_id,
                            )
                        except Exception as e:
                            logger.debug(f"[NEWS_IMPACT] signal_quality update failed for {memory_id}: {e}")

                    # Build UPDATE query
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
                    "open_interest": market.open_interest,
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

    # ------------------------------------------------------------------ #
    #  Deferred initialization (background LLM builds)                    #
    # ------------------------------------------------------------------ #

    async def _run_deferred_init(self) -> None:
        """Build event understanding in background."""
        try:
            await self._fetch_all_candlesticks()

            if self._understanding_builder and self._index:
                logger.info("[DEFERRED_INIT] Building event understanding...")
                built_count = 0
                total = len(self._index.events)
                for i, (event_ticker, event) in enumerate(self._index.events.items(), 1):
                    await self._broadcast_startup(f"Building understanding: {event_ticker} ({i}/{total})")
                    try:
                        understanding = await self._understanding_builder.build(event)
                        event.understanding = understanding.to_dict()
                        built_count += 1
                    except Exception as e:
                        logger.warning(f"Understanding failed for {event_ticker}: {e}")
                logger.info(f"[DEFERRED_INIT] Built {built_count}/{len(self._index.events)} understandings")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Deferred init failed: {e}")
        finally:
            self._system_ready.set()
            await self._broadcast_startup("System fully initialized")
            try:
                await self._broadcast_snapshot()
            except Exception:
                pass

    async def _fetch_all_candlesticks(self) -> None:
        """Fetch 7-day hourly candlesticks for all loaded events.

        Uses market_client (prod in hybrid mode) for real volume data.
        """
        if not self._index:
            return

        client = getattr(self, "_market_client", None) or self._get_trading_client()
        if not client:
            logger.warning("[CANDLESTICKS] No trading client for candlestick fetch")
            return

        now = int(time.time())
        start_ts = now - (7 * 24 * 60 * 60)  # 7 days ago
        fetched = 0

        for event_ticker, event in self._index.events.items():
            try:
                series_ticker = event.series_ticker
                if not series_ticker:
                    logger.debug(f"[CANDLESTICKS] No series_ticker for {event_ticker}, skipping")
                    continue

                resp = await client.get_event_candlesticks(
                    series_ticker=series_ticker,
                    event_ticker=event_ticker,
                    start_ts=start_ts,
                    end_ts=now,
                    period_interval=60,  # hourly intervals
                )

                if resp:
                    event.candlesticks = resp
                    event.candlesticks_fetched_at = time.time()
                    fetched += 1
                    total_candles = sum(
                        len(c) for c in resp.get("market_candlesticks", []) if c
                    )
                    logger.debug(
                        f"[CANDLESTICKS] {event_ticker}: {total_candles} candles "
                        f"across {len(resp.get('market_tickers', []))} markets"
                    )

            except Exception as e:
                logger.warning(f"[CANDLESTICKS] Failed for {event_ticker}: {e}")

            await asyncio.sleep(0.3)  # Rate limit protection

        logger.info(f"[CANDLESTICKS] Fetched candlesticks for {fetched}/{len(self._index.events)} events")

    # ------------------------------------------------------------------ #
    #  Gateway + tool setup                                               #
    # ------------------------------------------------------------------ #

    async def _setup_gateway(self, client, order_ttl: int) -> None:
        """Create KalshiGateway, EventBridge, TradingSession, Sniper, and subscribe WS channels.

        In hybrid data mode, two EventBridges are created:
        - Market data (orderbook, ticker, trade) from the prod gateway WS
        - Portfolio events (fill, market_positions) from the demo gateway WS
        """
        from ..gateway import KalshiGateway, GatewayEventBridge
        from ..agent_tools import TradingSession

        # Create demo gateway (always — handles trading + portfolio events)
        self._gateway = KalshiGateway(
            api_url=self._config.api_url,
            ws_url=self._config.ws_url,
            subaccount=self._config.subaccount,
        )
        await self._gateway.connect()
        logger.info(f"[GATEWAY] Demo KalshiGateway connected (subaccount #{self._config.subaccount})")

        # Validate subaccount works before proceeding
        if self._config.subaccount > 0:
            try:
                bal = await self._gateway.get_balance()
                logger.info(f"[GATEWAY] Subaccount #{self._config.subaccount} verified: {bal.balance}c")
            except Exception as e:
                logger.error(f"[GATEWAY] Subaccount #{self._config.subaccount} validation failed: {e}")
                raise

        # Bridge gateway WS events -> EventBus
        if self._market_gateway:
            # HYBRID MODE: split market data (prod) and portfolio events (demo)
            market_bridge = GatewayEventBridge(
                event_bus=self._event_bus,
                ws=self._market_gateway.get_ws(),
                channels=["orderbook_delta", "ticker", "trade"],
            )
            market_bridge.wire()

            trading_bridge = GatewayEventBridge(
                event_bus=self._event_bus,
                ws=self._gateway.get_ws(),
                subaccount=self._config.subaccount,
                channels=["fill", "market_positions"],
            )
            trading_bridge.wire()

            self._event_bridge = market_bridge  # Primary bridge for stats
            self._trading_bridge = trading_bridge
            logger.info("[GATEWAY] Hybrid EventBridges wired (prod=market, demo=portfolio)")
        else:
            # Single bridge (existing behavior)
            self._event_bridge = GatewayEventBridge(
                event_bus=self._event_bus,
                ws=self._gateway.get_ws(),
                subaccount=self._config.subaccount,
            )
            self._event_bridge.wire()
            logger.info("[GATEWAY] EventBridge wired to EventBus")

        # Create shared trading session
        self._trading_session = TradingSession(
            order_group_id=self._order_group_id or "",
            order_ttl=order_ttl,
        )

        # Create Sniper execution layer (if enabled) — always uses demo gateway
        if self._config.sniper_enabled:
            from .sniper import Sniper, SniperConfig
            sniper_config = SniperConfig(
                enabled=self._config.sniper_enabled,
                max_position=self._config.sniper_max_position,
                max_capital=self._config.sniper_max_capital,
                cooldown=self._config.sniper_cooldown,
                max_trades_per_cycle=self._config.sniper_max_trades_per_cycle,
                arb_min_edge=self._config.sniper_arb_min_edge,
                order_ttl=getattr(self._config, "sniper_order_ttl", 30),
                leg_timeout=getattr(self._config, "sniper_leg_timeout", 5.0),
                vpin_reject_threshold=getattr(self._config, "sniper_vpin_reject_threshold", 0.98),
            )
            self._sniper = Sniper(
                gateway=self._gateway,
                index=self._index,
                event_bus=self._event_bus,
                session=self._trading_session,
                config=sniper_config,
                broadcast_callback=self._broadcast,
                order_register_callback=self._register_sniper_order,
                attention_callback=None,  # Wired after AttentionRouter created in step 9c
            )
            await self._sniper.start()
            logger.info("[GATEWAY] Sniper execution layer initialized")

        # Subscribe WS channels for single-arb market tickers
        market_tickers = list(self._index.market_tickers) if self._index else []
        if market_tickers:
            # In hybrid mode, subscribe market channels on PROD gateway
            ws_mux = self._market_gateway.get_ws() if self._market_gateway else self._gateway.get_ws()
            for channel in ("orderbook_delta", "ticker", "trade"):
                await ws_mux.subscribe_tickers(channel, market_tickers)
            logger.info(
                f"[GATEWAY] Subscribed {len(market_tickers)} tickers to "
                f"orderbook_delta+ticker+trade channels"
                f"{' (prod)' if self._market_gateway else ''}"
            )

        logger.info("[GATEWAY] Gateway layer wired")

    def _setup_tools(self, order_ttl: int) -> None:
        """Wire the V2 single-agent tool context (12 tools, 1 ToolContext)."""
        from .tools import ToolContext, set_context
        from .memory.session_store import SessionMemoryStore
        from .memory.vector_store import VectorMemoryService
        from .context_builder import ContextBuilder
        from .decision_ledger import DecisionLedger

        # Build pgvector persistent store (graceful degradation if DB unavailable)
        vector_store = None
        try:
            from kalshiflow_rl.data.database import rl_db
            vector_store = VectorMemoryService(db=rl_db)
            logger.info("[TOOLS] VectorMemoryService initialized (pgvector)")
        except Exception as e:
            logger.warning(f"[TOOLS] VectorMemoryService unavailable, session-only mode: {e}")

        # Build session memory (FAISS session + pgvector persistent)
        session_memory = SessionMemoryStore(vector_store=vector_store)

        # Build context builder
        ctx_builder = ContextBuilder(index=self._index, subaccount=self._config.subaccount)

        # Get sniper config if available
        sniper_config = self._sniper.config if self._sniper else None

        # Create AccountHealthService (background hygiene, no LLM)
        from .account_health import AccountHealthService
        max_drawdown = getattr(self._config, "max_drawdown_pct", 25.0)
        # Create DecisionLedger for order quality tracking
        self._decision_ledger = DecisionLedger(index=self._index, memory_store=session_memory)

        self._health_service = AccountHealthService(
            gateway=self._gateway,
            index=self._index,
            session=self._trading_session,
            order_group_id=self._order_group_id,
            broadcast_callback=self._broadcast,
            max_drawdown_pct=max_drawdown,
            pause_callback=self.pause_captain,
            resume_callback=self.resume_captain,
            memory=session_memory,
            decision_ledger=self._decision_ledger,
        )

        ctx = ToolContext(
            gateway=self._gateway,
            index=self._index,
            memory=session_memory,
            search=self._search_service,
            sniper=self._sniper,
            sniper_config=sniper_config,
            session=self._trading_session,
            context_builder=ctx_builder,
            broadcast=self._broadcast,
            health_service=self._health_service,
            decision_ledger=self._decision_ledger,
        )
        set_context(ctx)
        logger.info("[TOOLS] ToolContext wired (12 tools, single agent)")

    # ------------------------------------------------------------------ #
    #  QuoteEngine setup (MM folded into Captain)                         #
    # ------------------------------------------------------------------ #

    async def _setup_quote_engine(self) -> None:
        """Create and start QuoteEngine, MMIndex, MMMonitor, MMAttentionRouter.

        Called from start() when mm_enabled=True.
        Folds the deterministic QuoteEngine into Captain's ecosystem.
        """
        from ..market_maker.index import MMIndex
        from ..market_maker.monitor import MMMonitor
        from ..market_maker.attention import MMAttentionRouter
        from ..market_maker.quote_engine import QuoteEngine
        from ..market_maker.models import QuoteConfig
        from .tools import get_context

        client = self._get_trading_client()
        if not client:
            logger.warning("[MM] No trading client — skipping QuoteEngine setup")
            return

        # 1. Create MMIndex (wraps its own EventArbIndex — coexists with Captain's index)
        self._mm_index = MMIndex(
            fee_per_contract_cents=getattr(self._config, "single_arb_fee_per_contract", 1),
        )

        # 2. Load MM events via REST
        mm_tickers = self._config.mm_event_tickers
        loaded = 0
        market_client = getattr(self, "_market_client", client)
        for event_ticker in mm_tickers:
            try:
                meta = await self._mm_index.load_event(event_ticker, market_client)
                if meta:
                    loaded += 1
                    logger.info(f"[MM] Loaded {event_ticker}: {len(meta.markets)} markets")
            except Exception as e:
                logger.warning(f"[MM] Failed to load event {event_ticker}: {e}")

        if loaded == 0:
            logger.warning("[MM] No MM events loaded — skipping QuoteEngine")
            return

        # 3. Subscribe MM market tickers to WS channels
        mm_market_tickers = list(self._mm_index.market_tickers)
        if mm_market_tickers:
            ws_mux = self._market_gateway.get_ws() if self._market_gateway else self._gateway.get_ws()
            for channel in ("orderbook_delta", "ticker", "trade"):
                await ws_mux.subscribe_tickers(channel, mm_market_tickers)
            logger.info(f"[MM] Subscribed {len(mm_market_tickers)} MM tickers to WS")

        # 4. Create separate MM order group
        try:
            resp = await client.create_order_group(contracts_limit=10000)
            self._mm_order_group_id = resp.get("order_group_id")
            if self._mm_order_group_id:
                logger.info(f"[MM] Order group created: {self._mm_order_group_id[:8]}...")
                # Register with AccountHealthService as protected
                if self._health_service:
                    self._health_service.register_order_group(self._mm_order_group_id)
        except Exception as e:
            logger.warning(f"[MM] Order group creation failed: {e}")

        # 5. Create QuoteConfig from V3_MM_* config fields
        quote_config = QuoteConfig(
            enabled=True,
            base_spread_cents=self._config.mm_base_spread_cents,
            quote_size=self._config.mm_quote_size,
            skew_factor=self._config.mm_skew_factor,
            max_position=self._config.mm_max_position,
            max_event_exposure=self._config.mm_max_event_exposure,
            refresh_interval=self._config.mm_refresh_interval,
        )

        # 6. Create QuoteEngine
        self._quote_engine = QuoteEngine(
            index=self._mm_index,
            gateway=self._gateway,
            config=quote_config,
            max_drawdown_cents=self._config.mm_max_drawdown_cents,
            order_ttl=getattr(self._config, "single_arb_order_ttl", 60),
            ws_broadcast=self._broadcast_mm,
            order_group_id=self._mm_order_group_id,
        )

        # 7. Create MMAttentionRouter (bridges signals to Captain)
        captain_inject = self._attention_router.inject_item if self._attention_router else None
        self._mm_attention = MMAttentionRouter(
            captain_inject=captain_inject,
        )

        # 8. Create MMMonitor (EventBus → MMIndex bridge)
        self._mm_monitor = MMMonitor(
            index=self._mm_index,
            event_bus=self._event_bus,
            trading_client=market_client,
            config=self._config,
            attention_router=self._mm_attention,
            broadcast_callback=self._broadcast_mm,
            quote_engine=self._quote_engine,
        )

        # 9. Prefetch orderbooks via REST
        for ticker in mm_market_tickers:
            try:
                ob = await market_client.get_orderbook(ticker)
                if ob:
                    self._mm_index.on_orderbook_update(ticker, ob.yes or [], ob.no or [], source="api")
            except Exception:
                pass
        logger.info(f"[MM] Prefetched {len(mm_market_tickers)} orderbooks")

        # 10. Start monitor + quote engine
        await self._mm_monitor.start()
        balance = 0
        try:
            bal_resp = await self._gateway.get_balance()
            balance = bal_resp.balance
        except Exception:
            pass
        self._quote_engine.start(balance_cents=balance)

        # 11. Wire into ToolContext
        ctx = get_context()
        if ctx:
            ctx.quote_engine = self._quote_engine
            ctx.mm_index = self._mm_index
            ctx.quote_config = quote_config

        # 12. Start position sync loop
        self._mm_position_sync_task = asyncio.create_task(self._mm_position_sync_loop())

        logger.info(
            f"[MM] QuoteEngine started: {loaded} events, "
            f"{len(mm_market_tickers)} markets, "
            f"spread={quote_config.base_spread_cents}c, size={quote_config.quote_size}"
        )

    async def _on_mm_fill(
        self,
        market_ticker: str,
        side: str,
        action: str,
        price_cents: int,
        count: int,
        order_id: str = "",
    ) -> None:
        """Handle fill event for MM markets. Routes to QuoteEngine + MMAttentionRouter."""
        if not self._mm_index:
            return
        if market_ticker not in self._mm_index.market_tickers:
            return

        # Forward to QuoteEngine (updates inventory, telemetry, clears filled quote)
        if self._quote_engine:
            self._quote_engine.on_fill(market_ticker, side, action, price_cents, count)

        # Signal to attention router
        if self._mm_attention:
            event_ticker = self._mm_index.get_event_for_ticker(market_ticker) or ""
            inv = self._mm_index.get_inventory(market_ticker)
            self._mm_attention.on_fill(
                event_ticker, market_ticker, side, price_cents, count, inv.position
            )

        logger.info(
            f"[MM_FILL] {market_ticker}: {action} {count} {side} @ {price_cents}c"
        )

    async def _mm_position_sync_loop(self) -> None:
        """Periodically sync MM balance tracking."""
        while self._running:
            try:
                await asyncio.sleep(30)
                if self._quote_engine:
                    try:
                        bal_resp = await self._gateway.get_balance()
                        self._quote_engine.update_balance(bal_resp.balance)
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[MM] Position sync error: {e}")

    async def _broadcast_mm(self, msg_type: str, data: dict) -> None:
        """Broadcast MM-specific messages to frontend."""
        await self._broadcast({
            "type": msg_type,
            "data": data,
        })

    # ------------------------------------------------------------------ #
    #  Broadcast helpers                                                  #
    # ------------------------------------------------------------------ #

    async def _broadcast_startup(self, message: str, step: int = 0, total_steps: int = 0) -> None:
        """Broadcast a startup progress message to frontend."""
        data = {
            "activity_type": "startup_progress",
            "message": message,
        }
        if step:
            data["step"] = step
            data["total_steps"] = total_steps
        await self._broadcast({
            "type": "system_activity",
            "data": data,
        })

    async def _broadcast(self, message: Dict) -> None:
        """Broadcast message to all frontend WebSocket clients."""
        if self._websocket_manager:
            try:
                msg_type = message.get("type", "unknown")
                msg_data = message.get("data", message)
                await self._websocket_manager.broadcast_message(msg_type, msg_data)
            except Exception as e:
                logger.warning(f"[COORDINATOR:BROADCAST_ERROR] {e}")

    async def _broadcast_snapshot(self) -> None:
        """Broadcast full snapshot to frontend."""
        if not self._index:
            return

        snapshot = self._index.get_snapshot()
        await self._broadcast({
            "type": "event_arb_snapshot",
            "data": snapshot,
        })

        # Send order tracker snapshot for reconnecting clients
        if self._tracked_orders:
            await self._broadcast({
                "type": "order_tracker_snapshot",
                "data": {"orders": list(self._tracked_orders.values())},
            })

    def get_gateway_config(self) -> dict:
        """Return gateway source configuration for frontend display."""
        hybrid = self._config.hybrid_data_mode
        if hybrid:
            prod_url = self._config.prod_api_url
            prod_host = prod_url.replace("https://", "").replace("http://", "").split("/")[0]
        else:
            prod_host = None

        demo_url = self._config.api_url or ""
        demo_host = demo_url.replace("https://", "").replace("http://", "").split("/")[0] if demo_url else "demo-api.kalshi.co"

        return {
            "hybrid_mode": hybrid,
            "market_data_source": "prod" if hybrid else "demo",
            "market_data_host": prod_host or demo_host,
            "trading_source": "demo",
            "trading_host": demo_host,
            "subaccount": self._config.subaccount,
        }

    async def _broadcast_gateway_config(self) -> None:
        """Broadcast gateway source configuration so frontend can show data provenance."""
        await self._broadcast({
            "type": "gateway_config",
            "data": self.get_gateway_config(),
        })

    # ------------------------------------------------------------------ #
    #  Opportunity + agent event handlers                                 #
    # ------------------------------------------------------------------ #

    async def _on_opportunity(self, opportunity) -> None:
        """Handle detected arb opportunity.

        Broadcasts to frontend and feeds Sniper for sub-second execution.
        """
        # Broadcast to frontend
        await self._broadcast({
            "type": "arb_opportunity",
            "data": opportunity.to_dict(),
        })

        # Feed Sniper (non-blocking - Sniper checks its own risk gates)
        if self._sniper:
            try:
                await self._sniper.on_arb_opportunity(opportunity)
            except Exception as e:
                logger.warning(f"[SNIPER] Execution error: {e}")

    async def _emit_agent_event(self, event_data: Dict) -> None:
        """Forward agent events to frontend WebSocket."""
        await self._broadcast(event_data)

    # ------------------------------------------------------------------ #
    #  Captain pause/resume interface                                     #
    # ------------------------------------------------------------------ #

    def pause_captain(self) -> None:
        """Pause the Captain after current cycle completes."""
        if self._captain:
            self._captain.pause()

    def resume_captain(self) -> None:
        """Resume Captain cycles."""
        if self._captain:
            self._captain.resume()

    def is_captain_paused(self) -> bool:
        """Check if Captain is paused."""
        return self._captain.is_paused if self._captain else False

    # ------------------------------------------------------------------ #
    #  Snapshot + health check interface                                  #
    # ------------------------------------------------------------------ #

    def get_tavily_budget_status(self) -> Optional[Dict]:
        """Get Tavily budget status for inclusion in trading_state WS broadcasts."""
        if self._tavily_budget:
            return self._tavily_budget.get_budget_status()
        return None

    def get_snapshot(self) -> Optional[Dict]:
        """Get full event arb snapshot for on-connect delivery."""
        if self._index:
            return self._index.get_snapshot()
        return None

    def get_discovery_snapshot(self) -> Optional[Dict]:
        """Get discovery state snapshot for on-connect delivery."""
        if self._discovery:
            return self._discovery.get_discovery_snapshot()
        return None

    def is_healthy(self) -> bool:
        """Health check for health monitor (must be callable method, not property)."""
        return self._running and self._index is not None

    def get_health_details(self) -> Dict:
        """Health details for the health monitor."""
        details = {
            "running": self._running,
            "started_at": self._started_at,
            "exchange_active": self._exchange_active,
            "exchange_last_check": self._exchange_last_check,
            "exchange_error_count": self._exchange_error_count,
        }

        if self._index:
            details["events"] = len(self._index.events)
            details["markets"] = len(self._index.market_tickers)

        if self._monitor:
            details["monitor"] = self._monitor.get_stats()

        if self._captain:
            captain_stats = self._captain.get_stats()
            captain_stats["paused_by_exchange"] = self._captain_paused_by_exchange
            details["captain"] = captain_stats

        if self._sniper:
            details["sniper"] = self._sniper.get_health_details()

        if self._discovery:
            details["discovery"] = self._discovery.get_stats()

        if self._tavily_budget:
            details["tavily_budget"] = self._tavily_budget.get_budget_status()

        if self._health_service:
            details["account_health"] = self._health_service.get_health_status().model_dump()

        if self._news_ingestion:
            details["news_ingestion"] = self._news_ingestion.get_stats()

        if self._swing_news_loop:
            details["swing_detection"] = self._swing_news_loop.get_stats()
        elif self._swing_detector:
            details["swing_detection"] = self._swing_detector.stats

        return details
