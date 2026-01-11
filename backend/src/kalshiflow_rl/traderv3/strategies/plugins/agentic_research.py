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
import json
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
from ...services.order_context_service import get_order_context_service
from ...services.execution_agent import (
    ExecutionAgent,
    MarketAssessmentToolResult,
    MarketMicrostructureToolResult,
    PositionsAndOrdersToolResult,
    ActionType,
)
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
        self._started_at: Optional[float] = None  # For uptime tracking
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
        self._max_positions = 300  # Max concurrent positions (match YAML default)
        self._max_positions_per_event = 10  # Max positions per event (concentration limit)
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
        
        # Execution agent config
        self._execution_agent: Optional[ExecutionAgent] = None
        self._execution_agent_model = "gpt-4o-mini"
        self._execution_agent_temperature = 0.3
        self._execution_shadow_mode = False  # Execution enabled - places orders via TradingDecisionService
        self._execution_interval = 30.0  # Fast loop: every 30s
        self._research_refresh_interval = 900.0  # Slow loop: every 15 minutes
        self._execution_loop_task: Optional[asyncio.Task] = None
        self._execution_candidate_limit = 20  # Max markets to evaluate per execution cycle

        # Decision tracking for iteration
        self._decision_log: List[Dict] = []  # Track all decisions with reasoning
        self._calibration_log: List[Dict] = []  # Track price estimation accuracy

        # Session tracking for decision persistence
        self._session_id = f"agentic_{int(time.time())}"

        # Stats
        self._stats = {
            "markets_researched": 0,
            "events_researched": 0,
            "assessments_completed": 0,
            "signals_detected": 0,
            "orders_placed": 0,
            "trades_skipped_threshold": 0,
            "trades_skipped_position_limit": 0,
            "trades_skipped_event_limit": 0,
            "trades_processed": 0,
            "trades_filtered": 0,
            "markets_with_trade_flow": 0,
            "tmo_updates_received": 0,
            "calibration_total_error": 0.0,  # Sum of |estimated - actual| prices
            "calibration_count": 0,  # Number of price guesses
            # Granular threshold skip tracking (all count toward trades_skipped_threshold)
            "skip_hold_recommendation": 0,  # Edge < 5% - HOLD recommendation
            "skip_below_threshold": 0,  # Edge < min_mispricing threshold
            "skip_low_confidence": 0,  # LLM confidence = LOW
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
            self._max_positions_per_event = params.get("max_positions_per_event", self._max_positions_per_event)
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
            
            # Execution agent params
            self._execution_agent_model = params.get("execution_agent_model", self._execution_agent_model)
            self._execution_agent_temperature = params.get("execution_agent_temperature", self._execution_agent_temperature)
            self._execution_shadow_mode = params.get("execution_shadow_mode", self._execution_shadow_mode)
            self._execution_interval = params.get("execution_interval_seconds", self._execution_interval)
            self._research_refresh_interval = params.get("research_refresh_interval_seconds", self._research_refresh_interval)
            self._execution_candidate_limit = params.get("execution_candidate_limit", self._execution_candidate_limit)
        
        # Initialize execution agent (before loops start)
        try:
            self._execution_agent = ExecutionAgent(
                openai_api_key=None,  # Load from env
                model=self._execution_agent_model,
                temperature=self._execution_agent_temperature,
                shadow_mode=self._execution_shadow_mode,
                # Position sizing constraints (passed to prompt)
                default_contracts_per_trade=self._contracts_per_trade,
                max_position_per_market=self._contracts_per_trade * 10,  # 10x default size max per market
                max_total_positions=self._max_positions,
                max_positions_per_event=self._max_positions_per_event,
                # Event bus for activity feed
                event_bus=self._context.event_bus if self._context else None,
            )
            logger.info(
                f"ExecutionAgent initialized (model={self._execution_agent_model}, "
                f"shadow_mode={self._execution_shadow_mode}, "
                f"default_qty={self._contracts_per_trade})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize execution agent: {e}")
            raise

        # Initialize event research service (primary for event-first approach)
        if self._use_event_first:
            try:
                # Get trading client from context for event fetching
                # Use trading_client_integration which has get_event() for fetching nested markets
                trading_client = context.trading_client_integration

                self._event_research_service = EventResearchService(
                    trading_client=trading_client,
                    openai_api_key=None,  # Will load from env
                    openai_model=self._openai_model,
                    openai_temperature=self._openai_temperature,
                    web_search_enabled=self._web_search_enabled,
                    min_mispricing_threshold=self._min_mispricing,
                    cache_ttl_seconds=self._cache_ttl_seconds,  # Semantic frame cache TTL
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

        # Start periodic evaluation loop (slow: research refresh)
        self._running = True
        self._started_at = time.time()
        self._evaluation_task = asyncio.create_task(self._research_refresh_loop())
        
        # Start fast execution loop (fast: trade decisions)
        self._execution_loop_task = asyncio.create_task(self._execution_loop())

        logger.info(
            f"AgenticResearchStrategy started "
            f"(event_first={self._use_event_first}, "
            f"execution_agent={self._execution_agent is not None}, "
            f"execution_shadow_mode={self._execution_shadow_mode}, "
            f"execution_interval={self._execution_interval}s, "
            f"research_refresh_interval={self._research_refresh_interval}s)"
        )
    
    async def stop(self) -> None:
        """Stop the strategy."""
        self._running = False

        # Cancel evaluation loops
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
        
        if self._execution_loop_task:
            self._execution_loop_task.cancel()
            try:
                await self._execution_loop_task
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
            and self._execution_agent is not None
            and self._context is not None
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics.

        Returns stats in the format expected by StrategyCoordinator.get_panel_data():
        - Standard fields: running, signals_detected, signals_executed, signals_skipped, etc.
        - skip_breakdown: Strategy-specific skip reasons for UI display
        - Additional agentic-research-specific metrics
        """
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

        # Calculate uptime
        uptime_seconds = 0
        if hasattr(self, '_started_at') and self._started_at:
            uptime_seconds = time.time() - self._started_at

        # Calculate signals_skipped (sum of all skip reasons)
        signals_skipped = (
            self._stats["trades_skipped_threshold"] +
            self._stats["trades_skipped_position_limit"] +
            self._stats["trades_skipped_event_limit"]
        )

        # Cache stats from event research service
        cache_hits = event_research_stats.get("cache_hits", 0)
        cache_misses = event_research_stats.get("cache_misses", 0)
        cache_hit_rate = event_research_stats.get("cache_hit_rate", 0.0)

        return {
            # Standard fields expected by coordinator
            "name": self.name,
            "display_name": self.display_name,
            "running": self._running,
            "uptime_seconds": uptime_seconds,
            "signals_detected": self._stats["assessments_completed"],  # All assessments evaluated
            "signals_executed": self._orders_placed,
            "signals_skipped": signals_skipped,
            "last_signal_at": self._last_signal_at,

            # Agentic research skip breakdown (different from RLM/ODMR)
            # These are the reasons trades were NOT executed
            "skip_breakdown": {
                "threshold": self._stats["trades_skipped_threshold"],  # Total: edge threshold + low confidence
                "position_limit": self._stats["trades_skipped_position_limit"],  # Global position limit reached
                "event_limit": self._stats["trades_skipped_event_limit"],  # Per-event concentration limit
                # Granular threshold breakdown (helps diagnose why trades are being skipped)
                "hold_recommendation": self._stats["skip_hold_recommendation"],  # Edge < 5%
                "below_edge_threshold": self._stats["skip_below_threshold"],  # Edge < min_mispricing
                "low_confidence": self._stats["skip_low_confidence"],  # LLM confidence = LOW
            },

            # Agentic-research-specific metrics for extended UI display
            "agentic_metrics": {
                "events_researched": len(self._events_researched),
                "markets_researched": len(self._markets_researched),
                "assessments_completed": self._stats["assessments_completed"],
                "orders_placed": self._orders_placed,
                # Cache performance
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": cache_hit_rate,
                # LLM calibration (price estimation accuracy)
                "calibration_samples": self._stats["calibration_count"],
                "calibration_avg_error_cents": round(calibration_avg_error, 1),
            },

            # Keep legacy fields for backwards compatibility
            "use_event_first": self._use_event_first,
            "orders_placed": self._orders_placed,
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
            # Execution agent stats
            "execution_agent": {
                "shadow_mode": self._execution_shadow_mode,
                "model": self._execution_agent_model,
                "execution_interval": self._execution_interval,
                "decisions_logged": len(self._execution_agent._decision_log) if self._execution_agent else 0,
                "recent_decisions": (
                    self._execution_agent.get_decision_history(limit=10)
                    if self._execution_agent else []
                ),
            },
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
    
    async def _research_refresh_loop(self) -> None:
        """Slow loop: Refresh event research and market assessments."""
        while self._running:
            try:
                await self._refresh_research()  # Run research first, then sleep
                await asyncio.sleep(self._research_refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in research refresh loop: {e}", exc_info=True)
    
    async def _execution_loop(self) -> None:
        """Fast loop: Make execution decisions using cached assessments + live market data."""
        while self._running:
            try:
                await asyncio.sleep(self._execution_interval)
                await self._run_execution_agent()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in execution loop: {e}", exc_info=True)
    
    async def _refresh_research(self) -> None:
        """
        Slow loop: Refresh event research and market assessments (every 15 minutes).
        
        This replaces the old _evaluate_tracked_markets() which did both
        research AND execution. Now this only does research; execution is handled
        by the fast execution loop.
        """
        if not self._context or not self._context.tracked_markets:
            return

        # Get active tracked markets
        tracked_markets = self._context.tracked_markets.get_active()
        if not tracked_markets:
            return

        # Use event-first research if enabled and service available
        if self._use_event_first and self._event_research_service:
            await self._evaluate_by_event_research_only(tracked_markets)
        else:
            await self._evaluate_by_market(tracked_markets)
    
    async def _evaluate_by_event_research_only(self, tracked_markets: list) -> None:
        """
        Event-first research only (no execution).

        Research events and cache assessments for the execution agent to use.
        Execution decisions are made separately by the fast execution loop.
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
            # Skip if event was recently researched (within TTL)
            # Execution agent will handle trade decisions using cached assessments
            if event_ticker in self._events_researched:
                cached_result = self._get_cached_event_result(event_ticker)
                if cached_result:
                    # Still valid - skip research refresh
                    continue
                else:
                    # Expired - clear from cache so we re-research
                    self._events_researched.discard(event_ticker)

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
                    event_bus=self._context.event_bus if self._context else None,
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

                    # Step 6: Assessments are cached - execution agent will use them
                    # (No direct execution here - that's now in the fast execution loop)
                else:
                    logger.warning(
                        f"Event research failed for {event_ticker}: "
                        f"{result.error_message}"
                    )

            except Exception as e:
                logger.error(f"Event research failed for {event_ticker}: {e}", exc_info=True)
                # Fall back to market-by-market for this event's markets
                await self._evaluate_by_market(markets)
    
    async def _run_execution_agent(self) -> None:
        """
        Fast execution loop: Make trade decisions using cached assessments + live market data.

        This runs every 30 seconds and uses the execution agent to decide whether
        to trade based on current microstructure, positions, and cached research.
        """
        if not self._execution_agent or not self._context:
            return

        # Get candidate markets from cached assessments
        candidate_markets = self._get_candidate_markets()

        if not candidate_markets:
            logger.debug("[EXECUTION_AGENT] No candidate markets found")
            return

        logger.debug(f"[EXECUTION_AGENT] Evaluating {len(candidate_markets)} candidate markets")

        # Process each candidate market
        for market_ticker, event_ticker in candidate_markets:
            try:
                # Build tool functions for this market
                async def get_assessment(ticker: str) -> MarketAssessmentToolResult:
                    return await self._get_assessment_tool(ticker)

                async def get_microstructure(ticker: str) -> MarketMicrostructureToolResult:
                    return await self._get_microstructure_tool(ticker)

                async def get_positions(ticker: str, _evt=event_ticker) -> PositionsAndOrdersToolResult:
                    return await self._get_positions_tool(ticker, _evt)

                async def execute_action(action) -> None:
                    await self._execute_action_tool(action)

                # Let execution agent decide
                plan = await self._execution_agent.decide_actions(
                    market_ticker=market_ticker,
                    get_assessment_fn=get_assessment,
                    get_microstructure_fn=get_microstructure,
                    get_positions_fn=get_positions,
                    execute_fn=execute_action,
                )

                if plan:
                    # Count as an assessment completed (evaluation happened)
                    self._stats["assessments_completed"] += 1

                    # Track skip reasons based on plan outcome
                    actions = plan.actions
                    if not actions:
                        # No action taken - this is a HOLD decision
                        self._stats["trades_skipped_threshold"] += 1
                        self._stats["skip_hold_recommendation"] += 1
                    else:
                        # Check if actions were executed
                        executed_any = False
                        for action in actions:
                            if action.type == ActionType.PLACE_ORDER:
                                # Order was placed (we track this in _execute_action_tool)
                                executed_any = True

                        if not executed_any:
                            # Actions planned but not executed (e.g., shadow mode)
                            pass  # Don't count as skip - it's by design

                    logger.info(
                        f"[EXECUTION_AGENT] {market_ticker}: {len(plan.actions)} actions planned "
                        f"(EV={plan.expected_value_cents}c)"
                    )

            except Exception as e:
                logger.error(
                    f"[EXECUTION_AGENT] Failed to process {market_ticker}: {e}",
                    exc_info=True
                )
    
    def _get_candidate_markets(self) -> List[tuple]:
        """
        Get candidate markets for execution agent to evaluate.
        
        Returns list of (market_ticker, event_ticker) tuples.
        """
        candidates = []
        current_time = time.time()
        
        # Get markets from cached event research results
        for event_ticker, result in self._event_research_cache.items():
            if not result or not result.success:
                continue
            
            # Check if cache is still valid
            age = current_time - result.event_context.researched_at
            if age > self._event_cache_ttl_seconds:
                continue
            
            # Add all markets from this event
            for assessment in result.assessments:
                candidates.append((assessment.market_ticker, event_ticker))
        
        # Also include markets with existing positions (for order management)
        if self._context and self._context.state_container:
            trading_state = self._context.state_container.trading_state
            if trading_state and trading_state.positions:
                for market_ticker in trading_state.positions.keys():
                    # Get event ticker from tracked markets
                    if self._context.tracked_markets:
                        market = self._context.tracked_markets.get_market(market_ticker)
                        if market:
                            event_ticker = market.event_ticker or market_ticker
                            if (market_ticker, event_ticker) not in candidates:
                                candidates.append((market_ticker, event_ticker))
        
        # Limit candidates to avoid excessive token usage
        return candidates[:self._execution_candidate_limit]
    
    async def _get_assessment_tool(self, market_ticker: str) -> MarketAssessmentToolResult:
        """Tool: Get cached market assessment."""
        # Find assessment in cached event results
        assessment = None
        event_ticker = None
        assessment_time = 0.0
        
        for event_tick, result in self._event_research_cache.items():
            if not result or not result.success:
                continue
            for assm in result.assessments:
                if assm.market_ticker == market_ticker:
                    assessment = assm
                    event_ticker = event_tick
                    assessment_time = result.event_context.researched_at
                    break
            if assessment:
                break
        
        if not assessment:
            # No assessment found - return default
            return MarketAssessmentToolResult(
                prob_yes=0.5,
                confidence="low",
                assessment_time=0.0,
                age_seconds=999999.0,
                market_probability_at_assessment=0.5,
                mispricing_at_assessment=0.0,
                thesis="No assessment available",
                key_evidence=[],
                what_would_change_mind="",
                event_ticker=None,
                resolution_criteria=None,
            )
        
        age_seconds = time.time() - assessment_time
        
        # Get event context for additional metadata
        resolution_criteria = None
        if event_ticker:
            cached_result = self._event_research_cache.get(event_ticker)
            if cached_result and cached_result.event_context and cached_result.event_context.context:
                resolution_criteria = cached_result.event_context.context.resolution_criteria
        
        return MarketAssessmentToolResult(
            prob_yes=assessment.evidence_probability,
            confidence=assessment.confidence.value if hasattr(assessment.confidence, 'value') else str(assessment.confidence),
            assessment_time=assessment_time,
            age_seconds=age_seconds,
            market_probability_at_assessment=assessment.market_probability,
            mispricing_at_assessment=assessment.mispricing_magnitude,
            thesis=assessment.edge_explanation or "",
            key_evidence=getattr(assessment, 'evidence_cited', []) or [],
            what_would_change_mind=getattr(assessment, 'what_would_change_mind', "") or "",
            event_ticker=event_ticker,
            resolution_criteria=resolution_criteria,
        )
    
    async def _get_microstructure_tool(self, market_ticker: str) -> MarketMicrostructureToolResult:
        """Tool: Get live market microstructure (execution-quality)."""
        # Build full microstructure context
        micro_ctx = await self._build_microstructure_context(market_ticker)
        
        # Get orderbook snapshot for raw bid/ask data
        yes_bid = None
        yes_ask = None
        no_bid = None
        no_ask = None
        yes_bid_size = None
        yes_ask_size = None
        no_bid_size = None
        no_ask_size = None
        orderbook_age = 999.0
        
        try:
            orderbook_state = await asyncio.wait_for(
                get_shared_orderbook_state(market_ticker),
                timeout=2.0
            )
            snapshot = await orderbook_state.get_snapshot()
            
            if snapshot:
                # Extract YES side
                yes_bids = snapshot.get("yes_bids", {})
                yes_asks = snapshot.get("yes_asks", {})
                if yes_bids:
                    yes_bid = max(yes_bids.keys())
                    yes_bid_size = yes_bids.get(yes_bid, 0)
                if yes_asks:
                    yes_ask = min(yes_asks.keys())
                    yes_ask_size = yes_asks.get(yes_ask, 0)
                
                # Extract NO side (or derive from YES)
                no_bids = snapshot.get("no_bids", {})
                no_asks = snapshot.get("no_asks", {})
                if no_bids:
                    no_bid = max(no_bids.keys())
                    no_bid_size = no_bids.get(no_bid, 0)
                if no_asks:
                    no_ask = min(no_asks.keys())
                    no_ask_size = no_asks.get(no_ask, 0)
                elif yes_bid is not None and yes_ask is not None:
                    # Derive NO prices from YES
                    no_bid = 100 - yes_ask
                    no_ask = 100 - yes_bid
                
                # Get orderbook age
                last_update = snapshot.get("last_update_time", 0)
                if last_update:
                    orderbook_age = (time.time() * 1000 - last_update) / 1000.0
        except Exception as e:
            logger.debug(f"Could not get orderbook for {market_ticker}: {e}")
        
        # Calculate spreads
        yes_spread = (yes_ask - yes_bid) if (yes_bid is not None and yes_ask is not None) else None
        no_spread = (no_ask - no_bid) if (no_bid is not None and no_ask is not None) else None
        
        # Get recent trade info from trade flow
        trade_flow = self._trade_flow.get(market_ticker)
        recent_trade_count = trade_flow.total_trades if trade_flow else 0
        recent_yes_trades = trade_flow.yes_trades if trade_flow else 0
        recent_no_trades = trade_flow.no_trades if trade_flow else 0
        last_trade_price = trade_flow.last_yes_price if trade_flow else None
        last_trade_time = trade_flow.last_trade_at if trade_flow else None
        
        # Calculate price change (simple: from first to last)
        price_change = 0
        if trade_flow and trade_flow.first_yes_price and trade_flow.last_yes_price:
            price_change = trade_flow.first_yes_price - trade_flow.last_yes_price
        
        # Determine staleness
        trade_flow_age = micro_ctx.trade_flow_age_seconds if micro_ctx else 999.0
        is_stale = orderbook_age > 30.0 or trade_flow_age > 120.0
        
        return MarketMicrostructureToolResult(
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            yes_spread=yes_spread,
            no_spread=no_spread,
            yes_bid_size=yes_bid_size,
            yes_ask_size=yes_ask_size,
            no_bid_size=no_bid_size,
            no_ask_size=no_ask_size,
            last_trade_price=last_trade_price,
            last_trade_time=last_trade_time,
            recent_trade_count=recent_trade_count,
            recent_yes_trades=recent_yes_trades,
            recent_no_trades=recent_no_trades,
            price_change_last_5min=price_change,  # Simplified - could improve with time window
            orderbook_age_seconds=orderbook_age,
            trade_flow_age_seconds=trade_flow_age,
            is_stale=is_stale,
        )
    
    async def _get_positions_tool(
        self,
        market_ticker: str,
        event_ticker: Optional[str]
    ) -> PositionsAndOrdersToolResult:
        """Tool: Get current positions and open orders."""
        position_side = None
        position_size = 0
        avg_entry = None
        unrealized_pnl = None
        open_orders = []
        event_exposure_count = 0
        total_event_exposure = 0
        
        if self._context and self._context.state_container:
            trading_state = self._context.state_container.trading_state
            
            # Get position in this market
            if trading_state and trading_state.positions:
                market_pos = trading_state.positions.get(market_ticker, {})
                pos_value = market_pos.get("position", 0)
                if pos_value != 0:
                    position_size = abs(pos_value)
                    position_side = "yes" if pos_value > 0 else "no"
                    # Try to compute avg entry (simplified)
                    total_traded = market_pos.get("total_traded", 0)
                    if total_traded and position_size:
                        avg_entry = int(total_traded / position_size)
                    # Unrealized P&L
                    market_exposure = market_pos.get("market_exposure", 0)
                    if market_exposure and total_traded:
                        unrealized_pnl = market_exposure - total_traded
            
            # Get open orders in this market
            if trading_state and trading_state.orders:
                for order_id, order_data in trading_state.orders.items():
                    if order_data.get("ticker") == market_ticker:
                        placed_at = order_data.get("placed_at", time.time())
                        open_orders.append({
                            "order_id": order_id,
                            "side": order_data.get("side", "unknown"),
                            "qty": order_data.get("count", 0),
                            "price": order_data.get("price", 0),
                            "age_seconds": time.time() - placed_at,
                        })
            
            # Count event exposure
            if event_ticker and trading_state and trading_state.positions:
                if self._context.tracked_markets:
                    for ticker, pos in trading_state.positions.items():
                        market = self._context.tracked_markets.get_market(ticker)
                        if market and market.event_ticker == event_ticker:
                            pos_value = pos.get("position", 0)
                            if pos_value != 0:
                                event_exposure_count += 1
                                market_exposure = pos.get("market_exposure", 0)
                                total_event_exposure += market_exposure
        
        return PositionsAndOrdersToolResult(
            current_position_side=position_side,
            current_position_size=position_size,
            avg_entry_price=avg_entry,
            unrealized_pnl_cents=unrealized_pnl,
            open_orders=open_orders,
            event_exposure_count=event_exposure_count,
            total_event_exposure_cents=total_event_exposure,
        )
    
    async def _execute_action_tool(self, action) -> None:
        """Tool: Execute an action (place order). Raises exception on failure."""
        if not self._context or not self._context.trading_service:
            raise RuntimeError("No trading service available")
        
        if action.type != ActionType.PLACE_ORDER:
            raise ValueError(f"Action type {action.type} not yet implemented")
        
        if not action.market_ticker or not action.side or not action.qty or not action.limit_price:
            raise ValueError("Invalid action: missing required fields")
        
        # Create TradingDecision
        from ...services.trading_decision_service import TradingDecision
        
        decision = TradingDecision(
            action="buy",
            market=action.market_ticker,
            side=action.side,
            quantity=action.qty,
            price=action.limit_price,
            reason=f"Execution agent: {action.reason}",
            confidence=0.75,  # Default confidence (could be improved)
            strategy_id=self.name,  # Important: keep attribution to agentic_research
            signal_params={
                "execution_agent": True,
                "action_reason": action.reason,
            },
        )
        
        # Execute via trading service
        success = await self._context.trading_service.execute_decision(decision)
        
        if success:
            self._orders_placed += 1
            self._signals_detected += 1
            self._last_signal_at = time.time()
            self._stats["orders_placed"] += 1
            self._stats["signals_detected"] += 1
            logger.info(
                f"[EXECUTION_AGENT] Order placed: {action.side.upper()} "
                f"{action.qty}x {action.market_ticker} @ {action.limit_price}c"
            )
        else:
            # Raise exception so execution agent can track failure
            raise RuntimeError(f"Trading service returned False for {action.market_ticker}")
        
        # Persist execution decision for analysis (fire-and-forget)
        try:
            await self._persist_execution_decision(action, decision, success)
        except Exception as e:
            logger.warning(f"Failed to persist execution decision: {e}")
    
    async def _persist_execution_decision(
        self,
        action,
        decision: Any,  # TradingDecision
        success: bool,
    ) -> None:
        """Persist execution agent decision to database for analysis."""
        try:
            from ...services.order_context_service import get_order_context_service
            import json
            
            order_context_service = get_order_context_service()
            db_pool = order_context_service.db_pool
            
            if not db_pool:
                return
            
            # Get assessment for context
            assessment_result = await self._get_assessment_tool(action.market_ticker)
            
            async with db_pool.acquire() as conn:
                # Use existing research_decisions table
                await conn.execute(
                    """
                    INSERT INTO research_decisions (
                        session_id, strategy_id, market_ticker, event_ticker,
                        action, reason, traded,
                        ai_probability, market_probability, edge, confidence, recommendation,
                        edge_explanation, key_driver, key_evidence
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                    )
                    """,
                    self._session_id,
                    self.name,
                    action.market_ticker,
                    assessment_result.event_ticker,
                    f"EXECUTION_AGENT_{action.type.value.upper()}",
                    action.reason,
                    success,
                    assessment_result.prob_yes,
                    assessment_result.market_probability_at_assessment,
                    assessment_result.mispricing_at_assessment,
                    assessment_result.confidence,
                    f"{action.side.upper()}" if action.side else "HOLD",
                    f"Execution agent decision: {action.reason}",
                    None,  # key_driver
                    json.dumps(assessment_result.key_evidence) if assessment_result.key_evidence else None,
                )
        except Exception as e:
            logger.warning(f"Failed to persist execution decision: {e}")

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
                # Legacy market-by-market path - assessments are cached for execution agent
                # Execution agent will handle trade decisions in fast loop
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

    # NOTE: _check_and_trade_from_event_assessment() has been removed.
    # This legacy execution path was replaced by the execution agent in _run_execution_agent().
    # See git history for the original implementation if needed for reference.
    
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
    
    # NOTE: _check_and_trade() has been removed.
    # This legacy execution path was replaced by the execution agent in _run_execution_agent().
    # See git history for the original implementation if needed for reference.
    
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

        # For paper trading: allow adding to existing positions
        # (In production, would want to limit max position per market)
        if market_ticker in trading_state.positions:
            position = trading_state.positions[market_ticker]
            pos_value = position.get("position", 0)
            if pos_value != 0:
                logger.debug(f"Already have position in {market_ticker}: {pos_value} (allowing add)")
                # Allow adding to position - no longer blocking

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

    def _count_event_positions(self, event_ticker: str) -> int:
        """Count current positions in markets from this event.

        Used to enforce per-event concentration limits and avoid
        over-exposure to a single event's correlated markets.

        Args:
            event_ticker: Event ticker to check positions for

        Returns:
            Number of active positions in markets from this event
        """
        if not self._context or not self._context.state_container:
            return 0

        trading_state = self._context.state_container.trading_state
        if not trading_state or not trading_state.positions:
            return 0

        # Get tracked markets for event ticker lookup
        tracked = self._context.tracked_markets
        if not tracked:
            return 0

        # Count positions in markets with matching event_ticker
        count = 0
        for market_ticker, position in trading_state.positions.items():
            market = tracked.get_market(market_ticker)
            if market and market.event_ticker == event_ticker:
                if position.get("position", 0) != 0:
                    count += 1

        return count

    def _has_active_position(self, market_ticker: str) -> bool:
        """Check if we have an active position in a market."""
        if not self._context or not self._context.state_container:
            return False

        trading_state = self._context.state_container.trading_state
        if not trading_state or not trading_state.positions:
            return False

        return market_ticker in trading_state.positions
    
    # NOTE: _create_trading_decision() has been removed.
    # This was only used by the deprecated _check_and_trade() path.
    # The execution agent now creates TradingDecision objects directly in _execute_action_tool().

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
        """Get recent decision history for frontend display.

        Pulls decisions from the execution agent and transforms them into the
        format expected by the TradingStrategiesPanel frontend component.

        Expected frontend format:
        - market_ticker: str
        - action: str (e.g., "executed", "skipped", "hold")
        - ai_probability: float (0-1)
        - edge: float (signed, e.g., 0.15 for 15% edge)
        - confidence: str ("high", "medium", "low")
        - age_seconds: int
        - strategy_id: str (for filtering in multi-strategy view)
        """
        current_time = time.time()
        result = []

        # Get decisions from execution agent
        if self._execution_agent:
            agent_decisions = self._execution_agent.get_decision_history(limit=limit)

            for d in agent_decisions:
                # Extract data from the nested structure
                plan = d.get("plan", {})
                tool_snapshots = d.get("tool_snapshots", {})
                assessment = tool_snapshots.get("assessment", {})
                execution_results = d.get("execution_results", [])

                # Determine action type based on plan actions
                actions = plan.get("actions", [])
                if not actions:
                    action = "hold"
                elif any(r.get("executed") for r in execution_results):
                    action = "executed"
                elif any(r.get("error") == "shadow_mode" for r in execution_results):
                    action = "shadow_mode"
                else:
                    # Actions were planned but failed or not executed
                    first_action = actions[0] if actions else {}
                    action_type = first_action.get("type", "hold")
                    if action_type == "hold":
                        action = "hold"
                    else:
                        action = "planned"

                # Calculate edge from assessment data
                prob_yes = assessment.get("prob_yes", 0.5)
                market_prob = assessment.get("market_probability_at_assessment", 0.5)
                edge = abs(prob_yes - market_prob) if prob_yes and market_prob else 0

                # Get action details for display
                action_details = []
                for a in actions:
                    side = a.get("side", "")
                    qty = a.get("qty", 0)
                    price = a.get("limit_price", 0)
                    if side and qty:
                        action_details.append(f"{side.upper()} {qty}@{price}c")

                entry = {
                    "market_ticker": d.get("market_ticker", ""),
                    "action": action,
                    "ai_probability": prob_yes,
                    "edge": edge,
                    "confidence": assessment.get("confidence", "unknown"),
                    "age_seconds": int(current_time - d.get("timestamp", current_time)),
                    "strategy_id": self.name,
                    # Additional details for expanded view
                    "trade_rationale": plan.get("trade_rationale", ""),
                    "expected_value_cents": plan.get("expected_value_cents"),
                    "risk_notes": plan.get("risk_notes", ""),
                    "action_details": action_details,
                    "actions_count": len(actions),
                }
                result.append(entry)

        # Also include any decisions from the strategy's own log (legacy)
        for d in list(self._decision_log)[-limit:]:
            entry = dict(d)
            entry["age_seconds"] = int(current_time - d.get("timestamp", current_time))
            entry["strategy_id"] = self.name
            if "market_ticker" not in entry and "market" in entry:
                entry["market_ticker"] = entry["market"]
            result.append(entry)

        # Sort by timestamp (newest first) and limit
        result.sort(key=lambda x: x.get("age_seconds", 9999))
        return result[:limit]

    async def _persist_decision(
        self,
        market_ticker: str,
        event_ticker: Optional[str],
        action: str,
        reason: str,
        traded: bool,
        ai_probability: Optional[float],
        market_probability: Optional[float],
        edge: Optional[float],
        confidence: Optional[str],
        recommendation: Optional[str],
        price_guess_cents: Optional[int],
        price_guess_error_cents: Optional[int],
        edge_explanation: Optional[str],
        key_driver: Optional[str],
        key_evidence: Optional[List[str]],
        entry_price_cents: Optional[int],
        order_id: Optional[str],
        # v2 calibration fields
        evidence_cited: Optional[List[str]] = None,
        what_would_change_mind: Optional[str] = None,
        assumption_flags: Optional[List[str]] = None,
        calibration_notes: Optional[str] = None,
        evidence_quality: Optional[str] = None,
    ) -> None:
        """
        Persist decision to research_decisions table for offline analysis.

        This enables post-hoc analysis of ALL decisions (traded AND skipped),
        allowing us to:
        - Understand why edges weren't taken
        - Optimize thresholds based on outcome data
        - Attribute performance to decision quality
        - Reconstruct the agent's reasoning chain

        Args:
            market_ticker: Market being evaluated
            event_ticker: Event grouping (for concentration analysis)
            action: Decision action (TRADE_YES, TRADE_NO, SKIP, etc.)
            reason: Full reason for decision (not truncated)
            traded: Whether a trade was executed
            ai_probability: LLM's YES probability estimate
            market_probability: Actual market price
            edge: Mispricing magnitude
            confidence: high/medium/low
            recommendation: BUY_YES/BUY_NO/HOLD
            price_guess_cents: LLM's blind price estimate
            price_guess_error_cents: Guess error (guess - actual)
            edge_explanation: Full LLM reasoning text
            key_driver: Primary driver from event research
            key_evidence: Top evidence points
            entry_price_cents: Trade entry price (if traded)
            order_id: Order ID (if traded)
            evidence_cited: Which evidence points support this estimate
            what_would_change_mind: What would most change this estimate
            assumption_flags: Assumptions made due to missing info
            calibration_notes: Notes on confidence calibration
            evidence_quality: Quality rating: high, medium, low
        """
        order_context_service = get_order_context_service()
        db_pool = order_context_service.db_pool

        if not db_pool:
            logger.debug("No DB pool available for decision persistence")
            return

        try:
            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO research_decisions (
                        session_id, strategy_id, market_ticker, event_ticker,
                        action, reason, traded,
                        ai_probability, market_probability, edge, confidence, recommendation,
                        price_guess_cents, price_guess_error_cents,
                        edge_explanation, key_driver, key_evidence,
                        entry_price_cents, order_id,
                        evidence_cited, what_would_change_mind, assumption_flags,
                        calibration_notes, evidence_quality
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19,
                        $20, $21, $22, $23, $24
                    )
                    """,
                    self._session_id,
                    self.name,
                    market_ticker,
                    event_ticker,
                    action,
                    reason,
                    traded,
                    ai_probability,
                    market_probability,
                    edge,
                    confidence,
                    recommendation,
                    price_guess_cents,
                    price_guess_error_cents,
                    edge_explanation,
                    key_driver,
                    json.dumps(key_evidence) if key_evidence else None,
                    entry_price_cents,
                    order_id,
                    json.dumps(evidence_cited) if evidence_cited else None,
                    what_would_change_mind,
                    json.dumps(assumption_flags) if assumption_flags else None,
                    calibration_notes,
                    evidence_quality,
                )
                logger.debug(f"Persisted decision for {market_ticker}: {action}")
        except Exception as e:
            logger.warning(f"Failed to persist decision to DB: {e}")