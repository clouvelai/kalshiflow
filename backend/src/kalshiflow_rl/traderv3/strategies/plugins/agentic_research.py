"""
Agentic Research Strategy Plugin for TRADER V3.

This plugin uses AI agents (LangChain + OpenAI) to research prediction markets
and generate trading decisions based on detected mispricing opportunities.

Purpose:
    Uses EVENT-FIRST research architecture to understand events holistically
    before evaluating individual markets. Combines web search and LLM reasoning
    with first-principles thinking to identify key drivers of outcomes.

Key Responsibilities:
    1. **Event-First Research** - Groups markets by event for holistic analysis
    2. **Key Driver Identification** - LLM identifies what determines YES vs NO
    3. **Evidence Gathering** - Targeted web search for key driver data
    4. **Batch Market Evaluation** - Evaluate all markets with shared event context
    5. **Trade Generation** - Create TradingDecision when confident mispricing detected
    6. **Microstructure Context** - Accumulates trade flow and orderbook signals

Architecture Position:
    - Registered with StrategyRegistry as "agentic_research"
    - Subscribes to MARKET_TRACKED, PUBLIC_TRADE_RECEIVED, TMO_FETCHED events
    - Uses TrackedMarketsState for market discovery
    - Groups markets by event_ticker for event-first research
    - Uses EventResearchService for 6-phase research pipeline
    - Builds MicrostructureContext for LLM consumption
    - Creates TradingDecision objects with strategy_id="agentic_research"

Research Flow:
    1. EVENT DISCOVERY: Group tracked markets by event_ticker
    2. EVENT RESEARCH: "What is this event about?"
    3. KEY DRIVER IDENTIFICATION: "What single factor determines YES vs NO?"
    4. EVIDENCE GATHERING: Targeted search for key driver data
    5. MARKET EVALUATION: Batch assess all markets with shared event context
    6. TRADE DECISIONS: Execute on mispriced markets

Design Principles:
    - **Event-first**: Research events holistically, not markets individually
    - **First-principles**: Identify key drivers, not just search for news
    - **Non-blocking**: Research runs asynchronously
    - **Config-driven**: All parameters from YAML config
    - **Multi-strategy safe**: Tracks its own positions separately
    - **Graceful degradation**: Works even without orderbook data

Dependencies:
    - langchain_openai: OpenAI ChatGPT integration
    - langchain_community: DuckDuckGo web search
    - OPENAI_API_KEY environment variable required
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass

from ..protocol import Strategy, StrategyContext
from ..registry import StrategyRegistry
from ...core.events import EventType
from ...core.event_bus import PublicTradeEvent, TMOFetchedEvent
from ...services.agentic_research_service import (
    AgenticResearchService,
    ResearchAssessment,
)
from ...services.event_research_service import EventResearchService
from ...state.microstructure_context import MicrostructureContext, TradeFlowState
from ...state.event_research_context import (
    EventResearchContext,
    EventResearchResult,
    MarketAssessment as EventMarketAssessment,
    Confidence,
)
from ...state.order_context import OrderbookContext
from ...state.tracked_markets import TrackedMarket
from ....data.orderbook_state import get_shared_orderbook_state

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.plugins.agentic_research")


@StrategyRegistry.register("agentic_research")
class AgenticResearchStrategy:
    """Strategy that uses AI research to find mispriced markets."""

    name = "agentic_research"
    display_name = "Agentic Research"
    subscribed_events = {
        EventType.MARKET_TRACKED,         # New market added to tracking
        EventType.PUBLIC_TRADE_RECEIVED,  # Accumulate trade flow per market
        EventType.TMO_FETCHED,            # True market open prices
    }
    
    def __init__(self):
        self._context: Optional[StrategyContext] = None
        self._research_service: Optional[AgenticResearchService] = None
        self._event_research_service: Optional[EventResearchService] = None
        self._running = False
        self._evaluation_task: Optional[asyncio.Task] = None

        # Async lock for trade flow processing
        self._trade_flow_lock = asyncio.Lock()

        # State
        self._markets_researched: Set[str] = set()
        self._events_researched: Set[str] = set()  # Event tickers already researched
        self._event_research_cache: Dict[str, EventResearchResult] = {}  # Cached results
        self._signals_detected = 0
        self._orders_placed = 0
        self._last_signal_at: Optional[float] = None

        # Trade flow accumulation per market (for microstructure context)
        self._trade_flow: Dict[str, TradeFlowState] = {}
        self._trades_processed = 0
        self._trades_filtered = 0

        # Config (will be loaded from YAML)
        self._min_mispricing = 0.10  # 10% mispricing required
        self._min_confidence = 0.70  # 70% confidence required
        self._max_positions = 10  # Max concurrent positions
        self._contracts_per_trade = 25  # Position size
        self._evaluation_interval = 30.0  # Check every 30 seconds

        # Time-to-settlement filter
        self._min_hours_to_settlement = 1.0  # Skip markets settling too soon
        self._max_days_to_settlement = 30  # Skip markets settling too far out

        # Category filter
        self._researchable_categories: list = []  # Empty = all categories

        # Event-first research config
        self._use_event_first = True  # Use event-first research (vs market-by-market)
        self._event_cache_ttl_seconds = 600.0  # 10 minute event research cache
        self._min_markets_for_event_research = 1  # Min markets in event to use event-first

        # Research service config
        self._openai_model = "gpt-4o"
        self._openai_temperature = 0.3
        self._web_search_enabled = True
        self._cache_ttl_seconds = 300.0
        self._max_concurrent_research = 3

        # Decision tracking for iteration
        self._decision_log: List[Dict] = []  # Track all decisions with reasoning
        self._calibration_log: List[Dict] = []  # Track price estimation accuracy

        # Stats
        self._stats = {
            "markets_researched": 0,
            "events_researched": 0,
            "assessments_completed": 0,
            "signals_detected": 0,
            "orders_placed": 0,
            "trades_skipped_threshold": 0,
            "trades_skipped_position_limit": 0,
            "trades_processed": 0,
            "trades_filtered": 0,
            "markets_with_trade_flow": 0,
            "tmo_updates_received": 0,
            "calibration_total_error": 0.0,  # Sum of |estimated - actual| prices
            "calibration_count": 0,  # Number of price guesses
        }
    
    async def start(self, context: StrategyContext) -> None:
        """Start the strategy."""
        self._context = context
        
        # Load config from context if available
        if context.config:
            params = context.config.params or {}
            self._min_mispricing = params.get("min_mispricing", self._min_mispricing)
            self._min_confidence = params.get("min_confidence", self._min_confidence)
            self._max_positions = params.get("max_positions", self._max_positions)
            self._contracts_per_trade = params.get("contracts_per_trade", self._contracts_per_trade)
            self._evaluation_interval = params.get("evaluation_interval_seconds", self._evaluation_interval)

            # Time-to-settlement filter params
            self._min_hours_to_settlement = params.get("min_hours_to_settlement", self._min_hours_to_settlement)
            self._max_days_to_settlement = params.get("max_days_to_settlement", self._max_days_to_settlement)

            # Category filter
            self._researchable_categories = params.get("researchable_categories", self._researchable_categories)

            # Event-first research params
            self._use_event_first = params.get("use_event_first", self._use_event_first)
            self._event_cache_ttl_seconds = params.get("event_cache_ttl_seconds", self._event_cache_ttl_seconds)
            self._min_markets_for_event_research = params.get("min_markets_for_event_research", self._min_markets_for_event_research)

            # Research service params
            self._openai_model = params.get("openai_model", self._openai_model)
            self._openai_temperature = params.get("openai_temperature", self._openai_temperature)
            self._web_search_enabled = params.get("web_search_enabled", self._web_search_enabled)
            self._cache_ttl_seconds = params.get("cache_ttl_seconds", self._cache_ttl_seconds)
            self._max_concurrent_research = params.get("max_concurrent_research", self._max_concurrent_research)

        # Initialize event research service (primary for event-first approach)
        if self._use_event_first:
            try:
                # Get trading client from context for event fetching
                trading_client = None
                if context.trading_service and hasattr(context.trading_service, '_client'):
                    trading_client = context.trading_service._client

                self._event_research_service = EventResearchService(
                    trading_client=trading_client,
                    openai_api_key=None,  # Will load from env
                    openai_model=self._openai_model,
                    openai_temperature=self._openai_temperature,
                    web_search_enabled=self._web_search_enabled,
                    min_mispricing_threshold=self._min_mispricing,
                )
                logger.info("EventResearchService initialized for event-first research")
            except Exception as e:
                logger.error(f"Failed to initialize event research service: {e}")
                logger.info("Falling back to market-by-market research")
                self._use_event_first = False

        # Initialize market research service (fallback or for single-market events)
        try:
            self._research_service = AgenticResearchService(
                openai_api_key=None,  # Will load from env
                openai_model=self._openai_model,
                openai_temperature=self._openai_temperature,
                web_search_enabled=self._web_search_enabled,
                cache_ttl_seconds=self._cache_ttl_seconds,
                max_concurrent_research=self._max_concurrent_research,
            )
            await self._research_service.start()
        except Exception as e:
            logger.error(f"Failed to initialize research service: {e}")
            raise
        
        # Subscribe to events
        if context.event_bus:
            context.event_bus.subscribe(
                EventType.MARKET_TRACKED,
                self._on_market_tracked,
            )

            # Subscribe to public trade events for trade flow accumulation
            await context.event_bus.subscribe_to_public_trade(self._on_public_trade)
            logger.info("Subscribed to PUBLIC_TRADE_RECEIVED events for trade flow")

            # Subscribe to TMO fetched events for true market open prices
            await context.event_bus.subscribe_to_tmo_fetched(self._on_tmo_fetched)
            logger.info("Subscribed to TMO_FETCHED events for price improvement")

        # Start periodic evaluation loop
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())

        logger.info(
            f"AgenticResearchStrategy started "
            f"(event_first={self._use_event_first}, "
            f"min_mispricing={self._min_mispricing}, "
            f"min_confidence={self._min_confidence}, "
            f"max_positions={self._max_positions})"
        )
    
    async def stop(self) -> None:
        """Stop the strategy."""
        self._running = False

        # Cancel evaluation loop
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass

        # Stop research service
        if self._research_service:
            await self._research_service.stop()

        # Unsubscribe from events
        if self._context and self._context.event_bus:
            self._context.event_bus.unsubscribe(
                EventType.MARKET_TRACKED,
                self._on_market_tracked,
            )
            self._context.event_bus.unsubscribe(
                EventType.PUBLIC_TRADE_RECEIVED,
                self._on_public_trade,
            )
            self._context.event_bus.unsubscribe(
                EventType.TMO_FETCHED,
                self._on_tmo_fetched,
            )

        # Clear trade flow state
        self._trade_flow.clear()

        logger.info("AgenticResearchStrategy stopped")
    
    def is_healthy(self) -> bool:
        """Check if strategy is healthy."""
        return (
            self._running
            and self._research_service is not None
            and self._context is not None
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics."""
        research_stats = (
            self._research_service.get_stats()
            if self._research_service
            else {}
        )

        event_research_stats = (
            self._event_research_service.get_stats()
            if self._event_research_service
            else {}
        )

        # Calculate trade flow summary
        total_trades_accumulated = sum(
            tf.total_trades for tf in self._trade_flow.values()
        )
        markets_with_flow = len([
            tf for tf in self._trade_flow.values() if tf.total_trades > 0
        ])

        # Calibration summary
        calibration_avg_error = (
            self._stats["calibration_total_error"] / self._stats["calibration_count"]
            if self._stats["calibration_count"] > 0 else 0.0
        )

        return {
            "name": self.name,
            "display_name": self.display_name,
            "running": self._running,
            "use_event_first": self._use_event_first,
            "signals_detected": self._signals_detected,
            "orders_placed": self._orders_placed,
            "last_signal_at": self._last_signal_at,
            "markets_researched": len(self._markets_researched),
            "events_researched": len(self._events_researched),
            "research_stats": research_stats,
            "event_research_stats": event_research_stats,
            # Trade flow metrics
            "trade_flow": {
                "trades_processed": self._trades_processed,
                "trades_filtered": self._trades_filtered,
                "markets_tracked": len(self._trade_flow),
                "markets_with_flow": markets_with_flow,
                "total_trades_accumulated": total_trades_accumulated,
            },
            # Calibration metrics (how well does LLM guess market prices?)
            "calibration": {
                "samples": self._stats["calibration_count"],
                "avg_price_error_cents": round(calibration_avg_error, 1),
            },
            # Recent decisions for debugging
            "recent_decisions": self._decision_log[-10:] if self._decision_log else [],
            **self._stats,
        }
    
    async def _on_market_tracked(self, event: Any) -> None:
        """Handle new market being tracked."""
        # Event will contain market ticker, but we'll use periodic evaluation
        # to check all tracked markets. This is just for logging.
        if hasattr(event, 'market_ticker'):
            logger.debug(f"Market tracked event: {event.market_ticker}")

    async def _on_public_trade(self, trade_event: PublicTradeEvent) -> None:
        """
        Handle public trade events to accumulate trade flow per market.

        Only processes trades for tracked markets to avoid memory bloat.
        Updates TradeFlowState which is used to build MicrostructureContext.
        Uses async lock to prevent race conditions in trade flow updates.
        """
        if not self._running or not self._context:
            return

        market_ticker = trade_event.market_ticker

        # Filter: only process trades for tracked markets
        if not self._context.tracked_markets:
            return
        if not self._context.tracked_markets.is_tracked(market_ticker):
            self._trades_filtered += 1
            return

        async with self._trade_flow_lock:
            self._trades_processed += 1

            # Get or create trade flow state for this market
            if market_ticker not in self._trade_flow:
                self._trade_flow[market_ticker] = TradeFlowState()
                self._stats["markets_with_trade_flow"] = len(self._trade_flow)
                logger.debug(f"Created trade flow state for {market_ticker}")

            # Update trade flow from the trade event
            flow_state = self._trade_flow[market_ticker]
            flow_state.update_from_trade(
                side=trade_event.side,
                price_cents=trade_event.price_cents,
                timestamp=time.time(),
            )

            self._stats["trades_processed"] = self._trades_processed
            self._stats["trades_filtered"] = self._trades_filtered

    async def _on_tmo_fetched(self, event: TMOFetchedEvent) -> None:
        """
        Handle True Market Open fetched events.

        Updates the TradeFlowState with the accurate market open price
        from the candlestick API, which is more reliable than first observed price.
        """
        if not self._running:
            return

        market_ticker = event.market_ticker
        tmo_price = event.true_market_open

        # Get or create trade flow state for this market
        if market_ticker not in self._trade_flow:
            self._trade_flow[market_ticker] = TradeFlowState()
            logger.debug(f"Created trade flow state for {market_ticker} (from TMO)")

        flow_state = self._trade_flow[market_ticker]
        old_tmo = flow_state.tmo_price
        flow_state.tmo_price = tmo_price

        self._stats["tmo_updates_received"] = self._stats.get("tmo_updates_received", 0) + 1

        if old_tmo is None:
            improvement = ""
            if flow_state.first_yes_price is not None:
                diff = flow_state.first_yes_price - tmo_price
                if diff != 0:
                    improvement = f" (diff from first observed: {diff:+d}c)"
            logger.info(f"TMO set for {market_ticker}: {tmo_price}c{improvement}")
        else:
            logger.debug(f"TMO updated for {market_ticker}: {old_tmo}c -> {tmo_price}c")

    async def _build_microstructure_context(
        self,
        market_ticker: str,
    ) -> Optional[MicrostructureContext]:
        """
        Build MicrostructureContext for a market from available data sources.

        Combines:
        - Trade flow state (from PUBLIC_TRADE_RECEIVED accumulation)
        - Orderbook context (from shared orderbook state)
        - Orderbook signals (from signal aggregator)
        - Position info (from state container)

        Args:
            market_ticker: Market to build context for

        Returns:
            MicrostructureContext or None if insufficient data
        """
        if not self._context:
            return None

        # === TRADE FLOW ===
        trade_flow = self._trade_flow.get(market_ticker)

        # === ORDERBOOK CONTEXT ===
        orderbook_context: Optional[OrderbookContext] = None
        try:
            orderbook_state = await asyncio.wait_for(
                get_shared_orderbook_state(market_ticker),
                timeout=2.0,
            )
            snapshot = await orderbook_state.get_snapshot()

            if snapshot:
                orderbook_context = OrderbookContext.from_orderbook_snapshot(
                    snapshot,
                    tight_spread=2,
                    normal_spread=5,
                )
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"Could not get orderbook for {market_ticker}: {e}")
            # Continue without orderbook data - graceful degradation

        # === ORDERBOOK SIGNALS (from aggregator buckets) ===
        orderbook_signals: Optional[Dict[str, Any]] = None
        if self._context.orderbook_integration:
            try:
                orderbook_signals = self._context.orderbook_integration.get_orderbook_signals(
                    market_ticker
                )
            except Exception as e:
                logger.debug(f"Could not get orderbook signals for {market_ticker}: {e}")

        # === POSITION INFO ===
        position_count: Optional[int] = None
        position_side: Optional[str] = None
        unrealized_pnl: Optional[int] = None

        if self._context.state_container:
            trading_state = self._context.state_container.trading_state
            if trading_state and trading_state.positions:
                market_position = trading_state.positions.get(market_ticker)
                if market_position:
                    position_count = abs(market_position.get("position", 0))
                    # Determine side from position sign or stored side
                    pos_value = market_position.get("position", 0)
                    if pos_value > 0:
                        position_side = "yes"
                    elif pos_value < 0:
                        position_side = "no"
                    # Calculate unrealized P&L if available
                    market_exposure = market_position.get("market_exposure", 0)
                    total_traded = market_position.get("total_traded", 0)
                    if market_exposure and total_traded:
                        unrealized_pnl = market_exposure - total_traded

        # Build the context
        context = MicrostructureContext.from_components(
            trade_flow=trade_flow,
            orderbook_context=orderbook_context,
            orderbook_signals=orderbook_signals,
            position_count=position_count,
            position_side=position_side,
            unrealized_pnl=unrealized_pnl,
        )

        # Log if we have meaningful data
        if not context.is_empty:
            logger.debug(
                f"Built microstructure context for {market_ticker}: "
                f"trades={context.total_trades}, yes_ratio={context.yes_ratio:.2f}, "
                f"spread={context.no_spread}c"
            )

        return context
    
    async def _evaluation_loop(self) -> None:
        """Periodic loop to evaluate tracked markets."""
        while self._running:
            try:
                await asyncio.sleep(self._evaluation_interval)
                await self._evaluate_tracked_markets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in evaluation loop: {e}", exc_info=True)
    
    async def _evaluate_tracked_markets(self) -> None:
        """
        Evaluate all tracked markets for research/trading opportunities.

        Uses event-first research when enabled:
        1. Group markets by event_ticker
        2. Research each event holistically
        3. Evaluate all markets in event with shared context
        4. Generate trades for mispriced markets

        Falls back to market-by-market research when:
        - event-first is disabled
        - event has fewer markets than threshold
        - event research service unavailable
        """
        if not self._context or not self._context.tracked_markets:
            return

        # Get active tracked markets
        tracked_markets = self._context.tracked_markets.get_active()
        if not tracked_markets:
            return

        # Use event-first research if enabled and service available
        if self._use_event_first and self._event_research_service:
            await self._evaluate_by_event(tracked_markets)
        else:
            await self._evaluate_by_market(tracked_markets)

    async def _evaluate_by_event(self, tracked_markets: list) -> None:
        """
        Event-first evaluation: group markets by event and research holistically.

        This is the primary research path when event-first is enabled.
        """
        # Step 1: Group markets by event_ticker
        event_groups: Dict[str, list] = {}
        for market in tracked_markets:
            # Skip if not researchable
            if not self._is_researchable(market):
                continue

            event_ticker = market.event_ticker or market.ticker  # Fallback to market ticker
            if event_ticker not in event_groups:
                event_groups[event_ticker] = []
            event_groups[event_ticker].append(market)

        logger.debug(f"Grouped {len(tracked_markets)} markets into {len(event_groups)} events")

        # Step 2: Process each event
        for event_ticker, markets in event_groups.items():
            # Skip if event was recently researched
            if event_ticker in self._events_researched:
                # Check if we have cached results
                cached_result = self._get_cached_event_result(event_ticker)
                if cached_result and cached_result.success:
                    # Re-check assessments for trading opportunities
                    for assessment in cached_result.assessments:
                        await self._check_and_trade_from_event_assessment(
                            assessment,
                            event_ticker=event_ticker,
                        )
                continue

            # Skip events with too few markets (use market-by-market instead)
            if len(markets) < self._min_markets_for_event_research:
                logger.debug(
                    f"Event {event_ticker} has {len(markets)} markets < "
                    f"{self._min_markets_for_event_research}, using market-by-market"
                )
                await self._evaluate_by_market(markets)
                continue

            # Step 3: Build microstructure context for all markets in event
            microstructure: Dict[str, MicrostructureContext] = {}
            for market in markets:
                ctx = await self._build_microstructure_context(market.ticker)
                if ctx:
                    microstructure[market.ticker] = ctx

            # Step 4: Research the event
            try:
                logger.info(
                    f"Starting event-first research for {event_ticker} "
                    f"({len(markets)} markets)"
                )

                result = await self._event_research_service.research_event(
                    event_ticker=event_ticker,
                    markets=markets,
                    microstructure=microstructure,
                )

                # Cache the result
                self._event_research_cache[event_ticker] = result
                self._events_researched.add(event_ticker)
                self._stats["events_researched"] += 1

                # Mark all markets as researched
                for market in markets:
                    self._markets_researched.add(market.ticker)
                    self._stats["markets_researched"] += 1

                if result.success:
                    logger.info(
                        f"Event research complete for {event_ticker}: "
                        f"{result.markets_evaluated} markets, "
                        f"{result.markets_with_edge} with edge"
                    )

                    # Step 5: Broadcast research results to frontend
                    if self._context.websocket_manager:
                        await self._context.websocket_manager.broadcast_event_research(
                            event_ticker, result
                        )

                    # Step 6: Process assessments for trading
                    for assessment in result.assessments:
                        await self._check_and_trade_from_event_assessment(
                            assessment,
                            event_ticker=event_ticker,
                        )
                else:
                    logger.warning(
                        f"Event research failed for {event_ticker}: "
                        f"{result.error_message}"
                    )

            except Exception as e:
                logger.error(f"Event research failed for {event_ticker}: {e}", exc_info=True)
                # Fall back to market-by-market for this event's markets
                await self._evaluate_by_market(markets)

    async def _evaluate_by_market(self, tracked_markets: list) -> None:
        """
        Market-by-market evaluation (fallback or when event-first disabled).

        This is the legacy research path that evaluates each market individually.
        """
        if not self._research_service:
            return

        for market in tracked_markets:
            # Skip if already researched recently
            if market.ticker in self._markets_researched:
                # Check if assessment is still valid
                assessment = await self._research_service.get_assessment(
                    market.ticker,
                    wait_seconds=0.0,
                )
                if assessment:
                    # Re-evaluate for trading opportunity
                    await self._check_and_trade(market.ticker, assessment)
                continue

            # Check if market is researchable
            if not self._is_researchable(market):
                continue

            # Build microstructure context before research (non-blocking)
            microstructure_context = await self._build_microstructure_context(market.ticker)

            # Trigger research (async, non-blocking) with microstructure context
            try:
                await self._research_service.research_market(
                    market_ticker=market.ticker,
                    market_title=market.title,
                    market_category=market.category,
                    current_price_cents=market.price,
                    hours_to_close=(market.close_ts - time.time()) / 3600 if market.close_ts else 24.0,
                    microstructure_context=microstructure_context,
                )

                self._markets_researched.add(market.ticker)
                self._stats["markets_researched"] += 1
            except Exception as e:
                logger.error(f"Failed to trigger research for {market.ticker}: {e}")

    def _get_cached_event_result(self, event_ticker: str) -> Optional[EventResearchResult]:
        """Get cached event research result if still valid."""
        result = self._event_research_cache.get(event_ticker)
        if not result:
            return None

        # Check TTL
        age = time.time() - result.event_context.researched_at
        if age > self._event_cache_ttl_seconds:
            # Expired, remove from cache
            self._event_research_cache.pop(event_ticker, None)
            self._events_researched.discard(event_ticker)
            return None

        return result

    async def _check_and_trade_from_event_assessment(
        self,
        assessment: EventMarketAssessment,
        event_ticker: Optional[str] = None,
    ) -> None:
        """
        Check event-based market assessment and generate trade if warranted.

        Similar to _check_and_trade but uses EventMarketAssessment structure.

        Args:
            assessment: Market assessment from event research
            event_ticker: Event ticker for grouping (persisted for calibration queries)
        """
        if not self._context:
            return

        # Increment assessments completed at start (for tracking all evaluations)
        self._stats["assessments_completed"] += 1

        # Log the assessment being evaluated
        logger.info(f"[RESEARCH] Market: {assessment.market_ticker}")
        logger.info(
            f"[RESEARCH] AI Prob: {assessment.evidence_probability:.1%} | "
            f"Market Prob: {assessment.market_probability:.1%} | "
            f"Edge: {assessment.mispricing_magnitude:+.1%}"
        )
        logger.info(
            f"[RESEARCH] Confidence: {assessment.confidence.value} | "
            f"Recommendation: {assessment.recommendation}"
        )

        # Helper to log decision
        def log_decision(action: str, reason: str, traded: bool = False, price: int = 0):
            decision_entry = {
                "timestamp": time.time(),
                "market_ticker": assessment.market_ticker,
                "strategy_id": self.name,
                "ai_probability": assessment.evidence_probability,
                "market_probability": assessment.market_probability,
                "edge": assessment.mispricing_magnitude,
                "confidence": assessment.confidence.value,
                "recommendation": assessment.recommendation,
                "reasoning": assessment.edge_explanation[:200] if assessment.edge_explanation else "",
                "action": action,
                "reason": reason,
                "traded": traded,
                "price": price,
            }
            self._decision_log.append(decision_entry)
            # Keep only last 100 decisions
            if len(self._decision_log) > 100:
                self._decision_log = self._decision_log[-100:]

        # Log what WOULD have been filtered (but we're letting it rip for now)
        abs_mispricing = abs(assessment.mispricing_magnitude)

        # Log if this would have been a HOLD (but still trade)
        if assessment.recommendation == "HOLD":
            logger.info(f"[NOTE] Would have been HOLD, but trading anyway (edge={assessment.mispricing_magnitude:+.1%})")

        # Log if below threshold (but still trade)
        if abs_mispricing < self._min_mispricing:
            logger.info(
                f"[NOTE] Below threshold: {abs_mispricing:.1%} < {self._min_mispricing:.1%} "
                f"(trading anyway)"
            )

        # Still skip LOW confidence - that's a real quality filter
        if assessment.confidence == Confidence.LOW:
            self._stats["trades_skipped_threshold"] += 1
            logger.info(f"[DECISION] SKIP: Low confidence")
            log_decision("SKIP", "Low confidence")
            return

        # For trading decision, use confidence level directly
        confidence = 0.80 if assessment.confidence == Confidence.HIGH else 0.70

        # Check position limits
        if not self._can_open_position(assessment.market_ticker):
            self._stats["trades_skipped_position_limit"] += 1
            logger.info(f"[DECISION] SKIP: Position limit reached")
            log_decision("SKIP", "Position limit reached")
            return

        # Determine side
        if assessment.recommendation == "BUY_YES":
            side = "yes"
        elif assessment.recommendation == "BUY_NO":
            side = "no"
        elif assessment.recommendation == "HOLD":
            # HOLD means small edge - pick side based on mispricing direction
            side = "yes" if assessment.mispricing_magnitude > 0 else "no"
            logger.info(f"[NOTE] HOLD → trading {side.upper()} (edge={assessment.mispricing_magnitude:+.1%})")
        else:
            log_decision("SKIP", "Invalid recommendation")
            return

        # Get current orderbook price
        ob_context = await self._get_orderbook(assessment.market_ticker)
        if not ob_context:
            # Fallback to market price from tracked markets
            if self._context.tracked_markets:
                market = self._context.tracked_markets.get_market(assessment.market_ticker)
                if market and market.price:
                    price = market.price if side == "yes" else (100 - market.price)
                else:
                    log_decision("SKIP", "No price available")
                    return
            else:
                log_decision("SKIP", "No tracked markets")
                return
        else:
            # Use orderbook best ask for buying, with fallback to tracked market price
            if side == "yes":
                if ob_context.yes_best_ask is not None:
                    price = ob_context.yes_best_ask
                elif self._context.tracked_markets:
                    # Fallback to tracked market price (same as ob_context=None case)
                    market = self._context.tracked_markets.get_market(assessment.market_ticker)
                    if market and market.price:
                        price = market.price
                        log_decision("FALLBACK", f"Using tracked market price {price}c (empty orderbook)")
                    else:
                        log_decision("SKIP", f"No price source for {assessment.market_ticker}")
                        return
                else:
                    log_decision("SKIP", f"No price source for {assessment.market_ticker}")
                    return
            else:  # NO side
                if ob_context.no_best_ask is not None:
                    price = ob_context.no_best_ask
                elif self._context.tracked_markets:
                    market = self._context.tracked_markets.get_market(assessment.market_ticker)
                    if market and market.price:
                        price = 100 - market.price  # Convert YES→NO
                        log_decision("FALLBACK", f"Using tracked market price {price}c (empty orderbook)")
                    else:
                        log_decision("SKIP", f"No price source for {assessment.market_ticker}")
                        return
                else:
                    log_decision("SKIP", f"No price source for {assessment.market_ticker}")
                    return

        # Create trading decision
        from ...services.trading_decision_service import TradingDecision

        # Get research metadata from cached event result
        research_duration = None
        llm_calls = None
        sources_checked = None
        evidence_reliability = None
        key_driver = None
        key_evidence = None
        if event_ticker:
            cached_result = self._event_research_cache.get(event_ticker)
            if cached_result and cached_result.success:
                research_duration = cached_result.event_context.research_duration_seconds
                llm_calls = cached_result.event_context.llm_calls_made
                sources_checked = cached_result.event_context.evidence.sources_checked
                evidence_reliability = cached_result.event_context.evidence.reliability.value
                # Audit trail: key reasoning components
                key_driver = cached_result.event_context.driver_analysis.primary_driver
                key_evidence = cached_result.event_context.evidence.key_evidence[:3]  # Top 3

        decision = TradingDecision(
            action="buy",
            market=assessment.market_ticker,
            side=side,
            quantity=self._contracts_per_trade,
            price=price,
            reason=f"Event research: {assessment.edge_explanation[:150]}..." if assessment.edge_explanation else "Event research signal",
            confidence=confidence,
            strategy_id=self.name,
            signal_params={
                "evidence_probability": assessment.evidence_probability,
                "market_probability": assessment.market_probability,
                "mispricing_magnitude": assessment.mispricing_magnitude,
                "confidence": assessment.confidence.value,  # For DB calibration column
                "event_ticker": event_ticker,  # For DB event grouping column
                "driver_application": assessment.driver_application,
                "specific_question": assessment.specific_question,
                # Price calibration (blind estimation tracking)
                "price_guess_cents": assessment.price_guess_cents,
                "price_guess_error_cents": assessment.price_guess_error_cents,
                # Research metadata (for performance correlation)
                "research_duration_seconds": research_duration,
                "llm_calls": llm_calls,
                "sources_checked": sources_checked,
                "evidence_reliability": evidence_reliability,
                # Audit trail: reasoning and evidence for post-hoc analysis
                "key_driver": key_driver,
                "key_evidence": key_evidence,
                "edge_explanation": assessment.edge_explanation,  # Full LLM reasoning
            },
        )

        # Execute via trading service
        logger.info(f"[DECISION] TRADE: {side.upper()} {self._contracts_per_trade}x @ {price}c")

        if self._context.trading_service:
            success = await self._context.trading_service.execute_decision(decision)
            if success:
                self._orders_placed += 1
                self._signals_detected += 1
                self._last_signal_at = time.time()
                self._stats["orders_placed"] += 1
                self._stats["signals_detected"] += 1

                # Log successful trade
                log_decision(f"TRADE_{side.upper()}", f"Order placed @ {price}c", traded=True, price=price)

                logger.info(
                    f"[ORDER PLACED] {decision.market} "
                    f"{decision.side} {decision.quantity}x @ {decision.price}c "
                    f"(AI={assessment.evidence_probability:.1%}, "
                    f"Mkt={assessment.market_probability:.1%}, "
                    f"Edge={assessment.mispricing_magnitude:+.1%})"
                )
            else:
                log_decision("TRADE_FAILED", "Trading service returned False")
                logger.warning(f"[ORDER FAILED] {decision.market}: Trading service returned False")
    
    def _is_researchable(self, market) -> bool:
        """Check if market is worth researching.

        Applies filters based on config:
        - Title must exist
        - Skip spread markets
        - Time-to-settlement within bounds
        - Category filter (if configured)
        """
        # Simple filters for now
        if not market.title:
            return False

        # Skip spread markets
        if "spread" in market.title.lower():
            return False

        # Time-to-settlement filter
        if market.close_ts:
            hours_to_close = (market.close_ts - time.time()) / 3600

            # Skip markets settling too soon
            if hours_to_close < self._min_hours_to_settlement:
                logger.debug(
                    f"Skipping {market.ticker}: settles in {hours_to_close:.1f}h < "
                    f"{self._min_hours_to_settlement}h minimum"
                )
                return False

            # Skip markets settling too far out
            max_hours = self._max_days_to_settlement * 24
            if hours_to_close > max_hours:
                logger.debug(
                    f"Skipping {market.ticker}: settles in {hours_to_close/24:.1f}d > "
                    f"{self._max_days_to_settlement}d maximum"
                )
                return False

        # Category filter (if configured)
        if self._researchable_categories:
            market_category = (market.category or "").lower()
            allowed_categories = [c.lower() for c in self._researchable_categories]
            if market_category not in allowed_categories:
                logger.debug(
                    f"Skipping {market.ticker}: category '{market.category}' not in "
                    f"allowed list {self._researchable_categories}"
                )
                return False

        return True
    
    async def _check_and_trade(
        self,
        market_ticker: str,
        assessment: Optional[ResearchAssessment] = None,
    ) -> None:
        """Check assessment and generate trade if warranted."""
        if not self._research_service or not self._context:
            return
        
        # Get assessment if not provided
        if not assessment:
            assessment = await self._research_service.get_assessment(
                market_ticker,
                wait_seconds=2.0,
            )
        
        if not assessment:
            return  # Research not complete yet

        # Increment assessments completed at start (for tracking all evaluations)
        self._stats["assessments_completed"] += 1

        # Check if recommendation is actionable
        if assessment.recommendation == "HOLD":
            return
        
        # Check thresholds
        abs_mispricing = abs(assessment.mispricing_magnitude)
        if abs_mispricing < self._min_mispricing:
            self._stats["trades_skipped_threshold"] += 1
            return
        
        if assessment.confidence < self._min_confidence:
            self._stats["trades_skipped_threshold"] += 1
            return
        
        # Check position limits
        if not self._can_open_position(market_ticker):
            self._stats["trades_skipped_position_limit"] += 1
            return
        
        # Generate trading decision
        decision = await self._create_trading_decision(assessment)
        if not decision:
            return
        
        # Execute via trading service
        if self._context.trading_service:
            success = await self._context.trading_service.execute_decision(decision)
            if success:
                self._orders_placed += 1
                self._signals_detected += 1
                self._last_signal_at = time.time()
                self._stats["orders_placed"] += 1
                self._stats["signals_detected"] += 1

                logger.info(
                    f"Agentic research trade executed: {decision.market} "
                    f"{decision.side} {decision.quantity}x @ {decision.price}c "
                    f"(agent_prob={assessment.agent_probability:.2f}, "
                    f"market_prob={assessment.market_price_probability:.2f}, "
                    f"mispricing={assessment.mispricing_magnitude:+.2f})"
                )
    
    def _can_open_position(self, market_ticker: str) -> bool:
        """Check if we can open a new position.

        For paper trading MVP: counts ALL positions (not strategy-specific)
        since Kalshi API doesn't return strategy_id on positions.
        """
        if not self._context or not self._context.state_container:
            return False

        trading_state = self._context.state_container.trading_state
        if not trading_state or not trading_state.positions:
            # No positions exist, we can open one
            return True

        # Check if already in this specific market (any strategy)
        if market_ticker in trading_state.positions:
            position = trading_state.positions[market_ticker]
            # Has actual position (not just zero)
            pos_value = position.get("position", 0)
            if pos_value != 0:
                logger.debug(f"Already have position in {market_ticker}: {pos_value}")
                return False

        # Count ALL active positions (multi-strategy isolation not yet supported)
        active_position_count = 0
        for ticker, position in trading_state.positions.items():
            pos_value = position.get("position", 0)
            if pos_value != 0:
                active_position_count += 1

        can_open = active_position_count < self._max_positions
        if not can_open:
            logger.debug(f"Position limit reached: {active_position_count}/{self._max_positions}")
        return can_open

    def _has_active_position(self, market_ticker: str) -> bool:
        """Check if we have an active position in a market."""
        if not self._context or not self._context.state_container:
            return False

        trading_state = self._context.state_container.trading_state
        if not trading_state or not trading_state.positions:
            return False

        return market_ticker in trading_state.positions
    
    async def _create_trading_decision(
        self,
        assessment: ResearchAssessment,
    ) -> Optional[Any]:  # TradingDecision
        """Create trading decision from assessment."""
        if not self._context:
            return None
        
        # Determine side and quantity
        if assessment.recommendation == "BUY_YES":
            side = "yes"
        elif assessment.recommendation == "BUY_NO":
            side = "no"
        else:
            return None
        
        # Get current orderbook price
        ob_context = await self._get_orderbook(assessment.market_ticker)
        if not ob_context:
            # Fallback to market price from tracked markets
            if self._context.tracked_markets:
                market = self._context.tracked_markets.get_market(assessment.market_ticker)
                if market and market.price:
                    # market.price is YES price, convert to NO if needed
                    price = market.price if side == "yes" else (100 - market.price)
                else:
                    return None
            else:
                return None
        else:
            # Use orderbook best ask for buying, with fallback to tracked market price
            if side == "yes":
                if ob_context.yes_best_ask is not None:
                    price = ob_context.yes_best_ask
                elif self._context.tracked_markets:
                    # Fallback to tracked market price (same as ob_context=None case)
                    market = self._context.tracked_markets.get_market(assessment.market_ticker)
                    if market and market.price:
                        price = market.price
                    else:
                        return None
                else:
                    return None
            else:  # NO side
                if ob_context.no_best_ask is not None:
                    price = ob_context.no_best_ask
                elif self._context.tracked_markets:
                    market = self._context.tracked_markets.get_market(assessment.market_ticker)
                    if market and market.price:
                        price = 100 - market.price  # Convert YES→NO
                    else:
                        return None
                else:
                    return None

        # Create decision
        from ...services.trading_decision_service import TradingDecision
        
        return TradingDecision(
            action="buy",
            market=assessment.market_ticker,
            side=side,
            quantity=self._contracts_per_trade,
            price=price,
            reason=f"Agentic research: {assessment.reasoning[:150]}...",
            confidence=assessment.confidence,
            strategy_id=self.name,
            signal_params={
                "agent_probability": assessment.agent_probability,
                "market_probability": assessment.market_price_probability,
                "mispricing_magnitude": assessment.mispricing_magnitude,
                "key_facts": assessment.key_facts,
                "sources": assessment.sources,
                "research_duration_seconds": assessment.research_duration_seconds,
            },
        )
    
    async def _get_orderbook(self, market_ticker: str):
        """Get orderbook context for market (supports both YES and NO sides)."""
        if not self._context or not self._context.orderbook_integration:
            return None
        
        try:
            # Use shared orderbook state
            orderbook_state = await asyncio.wait_for(
                get_shared_orderbook_state(market_ticker),
                timeout=2.0
            )
            snapshot = await orderbook_state.get_snapshot()
            
            # Convert to OrderbookContext format (has both YES and NO sides)
            from ...state.order_context import OrderbookContext
            return OrderbookContext.from_orderbook_snapshot(
                snapshot,
                tight_spread=2,
                normal_spread=5,
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.debug(f"Could not get orderbook for {market_ticker}: {e}")
            return None

    def get_decision_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent decision history for frontend display."""
        current_time = time.time()
        decisions = list(self._decision_log)
        decisions.reverse()  # Newest first
        result = []
        for d in decisions[:limit]:
            entry = dict(d)
            entry["age_seconds"] = int(current_time - d.get("timestamp", current_time))
            # Ensure market_ticker is present (backwards compatibility)
            if "market_ticker" not in entry and "market" in entry:
                entry["market_ticker"] = entry["market"]
            result.append(entry)
        return result