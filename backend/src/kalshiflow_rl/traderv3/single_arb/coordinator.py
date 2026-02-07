"""
SingleArbCoordinator - Wires EventArbIndex + EventArbMonitor + ArbCaptain.

Startup sequence:
1. Create EventArbIndex
2. For each hardcoded event ticker: load_event() via REST
3. Subscribe all market tickers to orderbook WS
4. Setup memory store (file + vector) + seed AGENTS.md
5. Create session order group
6. Set tool dependencies
7. Create EventArbMonitor (subscribes to EventBus)
8. Create ArbCaptain with FilesystemBackend for persistent /memories/
9. Broadcast initial snapshot
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from .index import EventArbIndex
from .monitor import EventArbMonitor
from .tools import set_dependencies as set_tool_dependencies
from .mentions_tools import (
    set_mentions_dependencies,
    set_mentions_broadcast_callback,
    restore_mentions_state_from_disk,
    _llm_parse_rules,
)
from .mentions_context import set_context_cache_dir, gather_event_context
from .event_understanding import UnderstandingBuilder, MentionsExtension
from .mentions_semantic import is_wordnet_available

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
    ):
        self._config = config
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._orderbook_integration = orderbook_integration
        self._trading_client = trading_client
        self._health_monitor = health_monitor

        self._index: Optional[EventArbIndex] = None
        self._monitor: Optional[EventArbMonitor] = None
        self._captain = None
        self._memory_store = None

        self._order_group_id: Optional[str] = None
        self._understanding_builder: Optional[UnderstandingBuilder] = None
        self._running = False
        self._started_at: Optional[float] = None

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

        # 1. Create index
        fee_per_contract = getattr(self._config, "single_arb_fee_per_contract", 1)
        min_edge = getattr(self._config, "single_arb_min_edge_cents", 3.0)
        self._index = EventArbIndex(
            fee_per_contract_cents=fee_per_contract,
            min_edge_cents=min_edge,
        )

        # 2. Load events
        event_tickers = getattr(self._config, "single_arb_event_tickers", [])
        if not event_tickers:
            logger.warning("No single_arb_event_tickers configured")
            return

        # Need a trading client for REST calls
        client = self._get_trading_client()
        if not client:
            logger.error("No trading client available for single-arb REST calls")
            return

        loaded_events = 0
        for event_ticker in event_tickers:
            state = None
            for attempt in range(3):  # 3 attempts per event
                state = await self._index.load_event(event_ticker, client)
                if state:
                    loaded_events += 1
                    break
                logger.warning(f"Failed to load event {event_ticker} (attempt {attempt + 1}/3)")
                await asyncio.sleep(1.0)  # 1s backoff between retries

            if not state:
                logger.error(f"Failed to load event after 3 attempts: {event_ticker}")

            # Small delay between events to avoid rate limiting
            await asyncio.sleep(0.5)

        if loaded_events == 0:
            logger.error("No events loaded - single-arb system cannot start")
            return

        # 2a. Build event understanding for all events
        await self._build_all_understanding()

        # 2b. Initialize mentions for applicable events
        mentions_enabled = getattr(self._config, "mentions_enabled", True)
        mentions_prewarm = getattr(self._config, "mentions_prewarm_enabled", False)
        if mentions_enabled:
            await self._initialize_mentions_events()
            # Pre-warm WordNet (downloads on first use if not present)
            self._initialize_wordnet()
            # ALWAYS establish baselines for mentions markets (critical for Captain)
            # This runs blind simulations synchronously on startup (~20s per event)
            await self._establish_mentions_baselines()
            # Pre-warm informed simulations (optional, can be slow)
            if mentions_prewarm:
                await self._prewarm_mentions_simulations()

        # 3. Subscribe all market tickers to orderbook WS
        market_tickers = self._index.market_tickers
        if market_tickers and self._orderbook_integration:
            for ticker in market_tickers:
                try:
                    await self._orderbook_integration.subscribe_market(ticker)
                except Exception as e:
                    logger.warning(f"Failed to subscribe {ticker} to orderbook: {e}")

            logger.info(f"Subscribed {len(market_tickers)} markets to orderbook WS")

        # 4. Setup memory store
        self._setup_memory()

        # 5. Create session order group for the captain
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

        # 6. Set tool dependencies (feature-flagged: new gateway vs legacy)
        order_ttl = getattr(self._config, "single_arb_order_ttl", 60)
        use_new_gateway = getattr(self._config, "use_new_gateway", False)

        if use_new_gateway:
            await self._setup_gateway_tools(client, order_ttl)
        else:
            self._setup_legacy_tools(client, order_ttl)

        # 7. Create monitor
        self._monitor = EventArbMonitor(
            index=self._index,
            event_bus=self._event_bus,
            trading_client=client,
            config=self._config,
            broadcast_callback=self._broadcast,
            opportunity_callback=self._on_opportunity,
        )
        await self._monitor.start()

        # 8. Check exchange status before starting Captain
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

        if not is_exchange_active:
            logger.warning(f"[SINGLE_ARB] Exchange not active: {exchange_error}")
            logger.warning("[SINGLE_ARB] Captain will NOT start until exchange is available")

        # 9. Create and start Captain (if enabled AND exchange is active)
        captain_enabled = getattr(self._config, "single_arb_captain_enabled", True)
        if captain_enabled:
            try:
                from .captain import ArbCaptain
                captain_interval = getattr(self._config, "single_arb_captain_interval", 60.0)

                # Build tool_overrides for new gateway path
                tool_overrides = None
                if use_new_gateway:
                    tool_overrides = self._build_gateway_tool_overrides()

                captain_kwargs = dict(
                    cycle_interval=captain_interval,
                    event_callback=self._emit_agent_event,
                    memory_data_dir=DEFAULT_MEMORY_DIR,
                    tool_overrides=tool_overrides,
                )
                if use_new_gateway:
                    captain_kwargs["index"] = self._index
                    captain_kwargs["gateway"] = getattr(self, "_gateway", None)

                self._captain = ArbCaptain(**captain_kwargs)

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

        # 10. Start exchange status monitor
        self._exchange_monitor_task = asyncio.create_task(self._exchange_monitor_loop())

        # 11. Broadcast initial snapshot
        await self._broadcast_snapshot()

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

        # Stop exchange monitor
        if self._exchange_monitor_task:
            self._exchange_monitor_task.cancel()
            try:
                await self._exchange_monitor_task
            except asyncio.CancelledError:
                pass
            self._exchange_monitor_task = None

        if self._captain:
            await self._captain.stop()
        if self._monitor:
            await self._monitor.stop()

        # Disconnect gateway if present
        if getattr(self, "_gateway", None):
            try:
                await self._gateway.disconnect()
                logger.info("[GATEWAY] KalshiGateway disconnected")
            except Exception as e:
                logger.debug(f"Gateway disconnect error: {e}")

        # Reset trading session if present
        if getattr(self, "_trading_session", None):
            self._trading_session.reset()

        # Reset order group (cancels all resting orders in group)
        if getattr(self, "_order_group_id", None):
            try:
                client = self._get_trading_client()
                if client:
                    await client.reset_order_group(self._order_group_id)
                    logger.info(f"[SINGLE_ARB] Order group reset: {self._order_group_id[:8]}...")
            except Exception as e:
                logger.debug(f"Order group reset failed: {e}")

        logger.info("[SINGLE_ARB:SHUTDOWN] Single-event arb system stopped")

    def _get_trading_client(self):
        """Get the underlying trading client for REST calls."""
        if self._trading_client:
            # V3TradingClientIntegration wraps the actual client
            if hasattr(self._trading_client, "_client"):
                return self._trading_client._client
            return self._trading_client
        return None

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

    async def _build_all_understanding(self) -> None:
        """Build EventUnderstanding for all loaded events.

        Uses UnderstandingBuilder with Wikipedia context, LLM synthesis,
        and registered extensions. Cached to disk with 4-hour TTL.
        """
        if not self._index:
            return

        # Create builder with cache dir and register extensions
        understanding_cache_dir = os.path.join(DEFAULT_MEMORY_DIR, "understanding")
        self._understanding_builder = UnderstandingBuilder(cache_dir=understanding_cache_dir)
        self._understanding_builder.register_extension(MentionsExtension())

        logger.info("[UNDERSTANDING] Building event understanding...")
        built_count = 0

        for event_ticker, event in self._index.events.items():
            try:
                understanding = await self._understanding_builder.build(event)
                event.understanding = understanding.to_dict()
                built_count += 1
                logger.debug(
                    f"[UNDERSTANDING] Built for {event_ticker}: "
                    f"participants={len(understanding.participants)} "
                    f"factors={len(understanding.key_factors)}"
                )
            except Exception as e:
                logger.warning(f"[UNDERSTANDING] Failed for {event_ticker}: {e}")

        logger.info(f"[UNDERSTANDING] Built {built_count}/{len(self._index.events)} event understandings")

    def _initialize_wordnet(self) -> None:
        """Initialize WordNet for semantic analysis (downloads on first use)."""
        try:
            if is_wordnet_available():
                logger.info("[SINGLE_ARB] WordNet available for semantic analysis")
            else:
                logger.warning("[SINGLE_ARB] WordNet not available - semantic features disabled")
        except Exception as e:
            logger.warning(f"[SINGLE_ARB] WordNet initialization failed: {e}")

    def _is_mentions_event(self, event) -> bool:
        """Check if event is a mentions market based on ticker pattern or category."""
        if not event:
            return False
        ticker = event.event_ticker.upper()
        title = event.title.upper() if event.title else ""
        category = event.category.lower() if event.category else ""

        # Check ticker patterns
        mentions_patterns = ["MENTION", "KXFEDMENTION", "KXNBAMENTION", "KXNFLMENTION"]
        for pattern in mentions_patterns:
            if pattern in ticker:
                return True

        # Check title for "mention" keyword
        if "MENTION" in title:
            return True

        # Check category
        if "mention" in category:
            return True

        return False

    async def _initialize_mentions_events(self) -> None:
        """Initialize mentions_data for all mentions events.

        Parses settlement rules into LexemePackLite for each mentions market.
        """
        if not self._index:
            return

        mentions_count = 0
        for event_ticker, event in self._index.events.items():
            if not self._is_mentions_event(event):
                continue

            mentions_count += 1
            logger.info(f"[MENTIONS] Detected mentions event: {event_ticker}")

            # Initialize mentions_data if empty
            if not event.mentions_data:
                event.mentions_data = {
                    "current_count": 0,
                    "evidence": [],
                    "sources_scanned": [],
                    "last_scan_ts": None,
                }

            # Parse rules for each market in the event
            for market_ticker, market in event.markets.items():
                # Get rules_primary from raw API response
                rules_text = market.raw.get("rules_primary", "")
                if not rules_text:
                    logger.warning(f"[MENTIONS] No rules_primary for {market_ticker}")
                    continue

                # Parse rules into LexemePackLite
                try:
                    lexeme_pack = await _llm_parse_rules(rules_text, market.title)
                    lexeme_pack.market_ticker = market_ticker

                    # Store in event.mentions_data
                    event.mentions_data["lexeme_pack"] = lexeme_pack.to_dict()
                    logger.info(
                        f"[MENTIONS] Parsed rules for {market_ticker}: "
                        f"entity='{lexeme_pack.entity}', "
                        f"forms={len(lexeme_pack.accepted_forms)}"
                    )
                except Exception as e:
                    logger.error(f"[MENTIONS] Failed to parse rules for {market_ticker}: {e}")

        if mentions_count > 0:
            logger.info(f"[MENTIONS] Initialized {mentions_count} mentions events")

    async def _establish_mentions_baselines(self) -> None:
        """Establish baseline probability estimates for all mentions events.

        Runs SYNCHRONOUS blind simulations on startup for any mentions events
        that don't already have baseline_estimates. This is critical for the
        Captain to make informed decisions - without baselines, edge calculations
        are impossible.

        Takes ~20s per event (10 simulations x ~2s each). This is intentionally
        blocking because correctness > speed at startup.
        """
        if not self._index:
            return

        # Find mentions events without baselines
        events_needing_baseline = []
        for event_ticker, event in self._index.events.items():
            if not self._is_mentions_event(event):
                continue
            if not event.mentions_data:
                continue

            # Check if baseline already exists
            baseline = event.mentions_data.get("baseline_estimates")
            if not baseline:
                events_needing_baseline.append((event_ticker, event))

        if not events_needing_baseline:
            logger.info("[MENTIONS] All mentions events already have baselines")
            return

        logger.info(
            f"[MENTIONS] Establishing baselines for {len(events_needing_baseline)} events "
            f"(~{len(events_needing_baseline) * 20}s)..."
        )

        for event_ticker, event in events_needing_baseline:
            try:
                # Get terms from lexeme pack
                lexeme_pack = event.mentions_data.get("lexeme_pack", {})
                entity = lexeme_pack.get("entity", "")
                terms = lexeme_pack.get("accepted_forms", [entity]) if entity else []

                if not terms:
                    logger.warning(f"[MENTIONS] No terms for {event_ticker}, skipping baseline")
                    continue

                # Run BLIND simulation (baseline)
                from .mentions_simulator import run_mentions_simulation

                logger.info(f"[MENTIONS] Running baseline simulation for {event_ticker}...")
                result = await run_mentions_simulation(
                    event_ticker=event_ticker,
                    event_title=event.title,
                    mention_terms=terms[:5],  # Limit to top 5 terms
                    n_simulations=10,  # Full baseline needs 10 simulations
                    cache_dir=os.path.join(DEFAULT_MEMORY_DIR, "simulations"),
                    mode="blind",  # BLIND = baseline
                )

                # Store as baseline (first blind run = stable baseline)
                estimates = result.get("estimates", {})
                if estimates:
                    event.mentions_data["baseline_estimates"] = estimates
                    event.mentions_data["last_simulation_ts"] = time.time()
                    event.mentions_data["simulation_mode"] = "blind"
                    logger.info(
                        f"[MENTIONS] Baseline established for {event_ticker}: "
                        f"{len(estimates)} terms, P(entity)={estimates.get(entity, {}).get('probability', 'N/A')}"
                    )
                else:
                    logger.warning(f"[MENTIONS] No estimates returned for {event_ticker}")

            except Exception as e:
                logger.error(f"[MENTIONS] Baseline failed for {event_ticker}: {e}")

    async def _prewarm_mentions_simulations(self) -> None:
        """Pre-warm INFORMED simulations for mentions events.

        Runs context-aware LLM roleplay simulations in the background to cache
        probability estimates for all mentions terms. This is optional and
        separate from baseline establishment.
        """
        if not self._index:
            return

        # Find all mentions events
        mentions_events = [
            (ticker, event)
            for ticker, event in self._index.events.items()
            if self._is_mentions_event(event) and event.mentions_data
        ]

        if not mentions_events:
            logger.info("[MENTIONS] No mentions events to pre-warm")
            return

        logger.info(f"[MENTIONS] Pre-warming simulations for {len(mentions_events)} events...")

        for event_ticker, event in mentions_events:
            try:
                # Get terms from lexeme pack
                lexeme_pack = event.mentions_data.get("lexeme_pack", {})
                entity = lexeme_pack.get("entity", "")
                terms = lexeme_pack.get("accepted_forms", [entity]) if entity else []

                if not terms:
                    continue

                # Run simulation (will cache results)
                from .mentions_simulator import run_mentions_simulation

                result = await run_mentions_simulation(
                    event_ticker=event_ticker,
                    event_title=event.title,
                    mention_terms=terms[:5],  # Limit to top 5 terms
                    n_simulations=5,  # Quick pre-warm with 5 simulations
                    cache_dir=os.path.join(DEFAULT_MEMORY_DIR, "simulations"),
                )

                # Store estimates
                event.mentions_data["simulation_estimates"] = result["estimates"]
                logger.info(
                    f"[MENTIONS] Pre-warmed {event_ticker}: "
                    f"{len(result['estimates'])} terms simulated"
                )

            except Exception as e:
                logger.warning(f"[MENTIONS] Pre-warm failed for {event_ticker}: {e}")

    def _setup_legacy_tools(self, client, order_ttl: int) -> None:
        """Wire the legacy tool layer (module globals)."""
        set_tool_dependencies(
            index=self._index,
            trading_client=client,
            file_store=self._memory_store,
            config=self._config,
            order_group_id=self._order_group_id,
            order_ttl=order_ttl,
            broadcast_callback=self._broadcast,
            understanding_builder=self._understanding_builder,
        )

        # Mentions tool dependencies
        set_mentions_dependencies(
            index=self._index,
            file_store=self._memory_store,
            config=self._config,
            mentions_data_dir=DEFAULT_MEMORY_DIR,
        )
        set_mentions_broadcast_callback(self._broadcast)
        set_context_cache_dir(DEFAULT_MEMORY_DIR)
        restore_mentions_state_from_disk()
        logger.info("[SINGLE_ARB] Legacy tool layer wired")

    async def _setup_gateway_tools(self, client, order_ttl: int) -> None:
        """Wire the new gateway + agent_tools layer."""
        from ..gateway import KalshiGateway, GatewayEventBridge
        from ..agent_tools import (
            captain_tools,
            commando_tools,
            mentions_tools as new_mentions_tools,
            CaptainToolContext,
            CommandoToolContext,
            MentionsToolContext,
            TradingSession,
        )

        # Create gateway (auth handled internally by GatewayAuth)
        self._gateway = KalshiGateway(
            api_url=self._config.api_url,
            ws_url=self._config.ws_url,
        )
        await self._gateway.connect()
        logger.info("[GATEWAY] KalshiGateway connected")

        # Bridge gateway WS events → EventBus
        self._event_bridge = GatewayEventBridge(
            event_bus=self._event_bus,
            ws=self._gateway.get_ws(),
        )
        self._event_bridge.wire()
        logger.info("[GATEWAY] EventBridge wired to EventBus")

        # Create shared trading session
        self._trading_session = TradingSession(
            order_group_id=self._order_group_id or "",
            order_ttl=order_ttl,
        )

        # Get the file store from the dual memory store
        file_store = self._memory_store

        # Wire Captain tools
        captain_ctx = CaptainToolContext(
            gateway=self._gateway,
            index=self._index,
            file_store=file_store,
            session=self._trading_session,
            understanding_builder=self._understanding_builder,
            broadcast_callback=self._broadcast,
        )
        captain_tools.set_context(captain_ctx)

        # Wire Commando tools
        commando_ctx = CommandoToolContext(
            gateway=self._gateway,
            index=self._index,
            file_store=file_store,
            session=self._trading_session,
            event_bus=self._event_bus,
            broadcast_callback=self._broadcast,
        )
        commando_tools.set_context(commando_ctx)

        # Wire Mentions tools (Phase 1: still uses old system underneath)
        mentions_ctx = MentionsToolContext(
            index=self._index,
            file_store=file_store,
            config=self._config,
            mentions_data_dir=DEFAULT_MEMORY_DIR,
            broadcast_callback=self._broadcast,
        )
        new_mentions_tools.set_context(mentions_ctx)

        # Also set legacy mentions deps (Phase 1 re-exports need them)
        set_mentions_dependencies(
            index=self._index,
            file_store=self._memory_store,
            config=self._config,
            mentions_data_dir=DEFAULT_MEMORY_DIR,
        )
        set_mentions_broadcast_callback(self._broadcast)
        set_context_cache_dir(DEFAULT_MEMORY_DIR)
        restore_mentions_state_from_disk()

        logger.info("[GATEWAY] New agent_tools layer wired (captain + commando + mentions)")

    def _build_gateway_tool_overrides(self) -> Dict:
        """Build tool override lists from the new agent_tools module."""
        from ..agent_tools import captain_tools, commando_tools, mentions_tools as new_mentions

        # Import the self-improvement tools from legacy (not yet in agent_tools)
        from .tools import report_issue, get_issues

        return {
            "captain": [
                captain_tools.get_events_summary,
                captain_tools.get_event_snapshot,
                captain_tools.get_market_orderbook,
                captain_tools.get_trade_history,
                captain_tools.get_positions,
                captain_tools.get_balance,
                captain_tools.update_understanding,
                report_issue,
                get_issues,
            ],
            "commando": [
                commando_tools.place_order,
                commando_tools.execute_arb,
                commando_tools.cancel_order,
                commando_tools.get_resting_orders,
                commando_tools.get_market_orderbook,
                commando_tools.get_recent_trades,
                commando_tools.get_balance,
                commando_tools.get_positions,
                commando_tools.record_learning,
            ],
            "mentions": [
                new_mentions.get_mentions_status,
                new_mentions.compute_edge,
                new_mentions.simulate_probability,
                new_mentions.trigger_simulation,
                new_mentions.get_event_context,
                new_mentions.get_mention_context,
                new_mentions.query_wordnet,
                new_mentions.get_mentions_rules,
                new_mentions.get_mentions_summary,
                commando_tools.record_learning,
            ],
        }

    def _setup_memory(self) -> None:
        """Setup dual memory store (file + vector) and seed memory files."""
        from .memory import FileMemoryStore, DualMemoryStore, VectorMemoryService

        file_store = FileMemoryStore(data_dir=DEFAULT_MEMORY_DIR)

        vector_store = None
        try:
            from kalshiflow_rl.data.database import rl_db
            vector_store = VectorMemoryService(db=rl_db)
            logger.info("VectorMemoryService initialized (pgvector)")
        except Exception as e:
            logger.warning(f"VectorMemoryService unavailable, file-only mode: {e}")

        self._memory_store = DualMemoryStore(
            file_store=file_store,
            vector_store=vector_store,
        )
        logger.info(
            f"DualMemoryStore initialized (vector={'yes' if vector_store else 'no'})"
        )

        # Seed AGENTS.md if it doesn't exist (loaded into Captain system prompt each cycle)
        agents_md_path = os.path.join(DEFAULT_MEMORY_DIR, "AGENTS.md")
        if not os.path.exists(agents_md_path):
            with open(agents_md_path, "w") as f:
                f.write(
                    "# Trading Learnings\n"
                    "\n"
                    "## Subagents\n"
                    "- trade_commando: execution (orders, queue management)\n"
                    "- mentions_specialist: 'will X say Y?' edge detection\n"
                    "\n"
                    "## Strategy Notes\n"
                    "- Arb: sum YES asks < 100 - fees → buy all YES\n"
                    "- Arb: sum YES bids > 100 + fees → buy all NO\n"
                    "- Mentions: simulation P vs market P → trade the gap\n"
                    "- Fee ~7c per contract. Start 5-10 contracts.\n"
                    "\n"
                    "## Patterns Observed\n"
                    "(Record patterns as discovered)\n"
                    "\n"
                    "## Mistakes & Lessons\n"
                    "(Record mistakes and learnings)\n"
                )
            logger.info("Created seed AGENTS.md")

        # Seed SIGNALS.md (auto-computed microstructure intel, Captain can annotate)
        signals_md_path = os.path.join(DEFAULT_MEMORY_DIR, "SIGNALS.md")
        if not os.path.exists(signals_md_path):
            with open(signals_md_path, "w") as f:
                f.write(
                    "# Microstructure Signals\n"
                    "\n"
                    "## Automated Signals (in market data)\n"
                    "- `micro.whale_trade_count`: Trades >= 100 contracts\n"
                    "- `micro.book_imbalance`: (bid_depth - ask_depth) / total\n"
                    "- `micro.buy_sell_ratio`: 5-min buy/sell flow\n"
                    "- `micro.rapid_sequence_count`: Sub-100ms trade bursts\n"
                    "- `micro.consistent_size_ratio`: 1.0 = all same size (bot)\n"
                    "\n"
                    "## Domain Signals\n"
                    "- Sports: speaker personas from Wikipedia, high variance\n"
                    "- Mentions: baseline from blind simulation, speaker style matters\n"
                    "- Check participants for related-person mentions\n"
                    "\n"
                    "## Observations\n"
                    "(Add microstructure observations here)\n"
                )
            logger.info("Created seed SIGNALS.md")

        # Seed PLAYBOOK.md (active strategies, in-flight plans)
        playbook_md_path = os.path.join(DEFAULT_MEMORY_DIR, "PLAYBOOK.md")
        if not os.path.exists(playbook_md_path):
            with open(playbook_md_path, "w") as f:
                f.write(
                    "# Active Playbook\n"
                    "\n"
                    "## Active Strategies\n"
                    "(Record strategies currently in play)\n"
                    "\n"
                    "## Exit Watchlist\n"
                    "(Positions to exit and target prices)\n"
                    "\n"
                    "## Event Understanding\n"
                    "- Built at startup, cached 4 hours\n"
                    "- Call update_understanding(ticker) if data feels stale\n"
                    "- time_to_close_hours < 1 = execution window, act decisively\n"
                    "- Domain affects uncertainty: sports > politics > corporate\n"
                    "\n"
                    "## Multi-Cycle Plans\n"
                    "(Record plans spanning multiple cycles)\n"
                )
            logger.info("Created seed PLAYBOOK.md")

    async def _broadcast(self, message: Dict) -> None:
        """Broadcast message to all frontend WebSocket clients."""
        if self._websocket_manager:
            try:
                msg_type = message.get("type", "unknown")
                msg_data = message.get("data", message)
                await self._websocket_manager.broadcast_message(msg_type, msg_data)
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")

    async def _broadcast_snapshot(self) -> None:
        """Broadcast full snapshot to frontend."""
        if not self._index:
            return

        snapshot = self._index.get_snapshot()
        await self._broadcast({
            "type": "event_arb_snapshot",
            "data": snapshot,
        })

    async def _on_opportunity(self, opportunity) -> None:
        """Handle detected arb opportunity."""
        # Broadcast to frontend
        await self._broadcast({
            "type": "arb_opportunity",
            "data": opportunity.to_dict(),
        })

    async def _emit_agent_event(self, event_data: Dict) -> None:
        """Forward agent events to frontend WebSocket."""
        await self._broadcast(event_data)

    # Captain pause/resume interface
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

    # Health check interface
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

        return details
