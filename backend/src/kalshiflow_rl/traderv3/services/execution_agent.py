"""
Execution Agent - LLM-powered trade decision and execution.

This module provides a lightweight LLM agent that makes intelligent trade
decisions using current market microstructure, positions, and cached research
assessments.

Architecture:
    - Runs on fast cadence (every 30s) to make execution decisions
    - Uses tools to gather: cached assessments, live microstructure, positions
    - Produces structured trade actions (place, cancel, replace, hold)
    - Executes only via TradingDecisionService (centralized safety)

Design Principles:
    - No hard-coded trading rules (beyond basic safety invariants)
    - Profit-focused reasoning using executable prices
    - Full decision logging for iteration/learning
    - Shadow mode support for safe rollout
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ..state.event_research_context import EventMarketAssessment, EventResearchResult
    from ..state.microstructure_context import MicrostructureContext
    from ..state.order_context import OrderbookContext
    from ..services.trading_decision_service import TradingDecisionService
    from ..core.state_container import V3StateContainer

logger = logging.getLogger("kalshiflow_rl.traderv3.services.execution_agent")


class ActionType(str, Enum):
    """Types of actions the execution agent can take."""
    PLACE_ORDER = "place_order"
    CANCEL_ORDER = "cancel_order"
    REPLACE_ORDER = "replace_order"
    HOLD = "hold"


class ExecutionAction(BaseModel):
    """Single action to execute."""
    type: ActionType = Field(description="Type of action")
    market_ticker: Optional[str] = Field(None, description="Market ticker for this action")
    side: Optional[str] = Field(None, description="'yes' or 'no' for place_order actions")
    qty: Optional[int] = Field(None, description="Quantity in contracts")
    limit_price: Optional[int] = Field(None, description="Limit price in cents")
    order_id: Optional[str] = Field(None, description="Order ID for cancel/replace actions")
    reason: str = Field("", description="Brief reason for this action")


class ExecutionPlan(BaseModel):
    """Complete execution plan from the agent."""
    actions: List[ExecutionAction] = Field(default_factory=list, description="List of actions to execute")
    trade_rationale: str = Field("", description="Why these actions are optimal now")
    expected_value_cents: Optional[float] = Field(None, description="Expected value at executable price")
    risk_notes: str = Field("", description="Risks: liquidity, staleness, correlation, tail risks")
    data_quality: str = Field("", description="Assessment of tool data completeness/quality")
    alternatives_considered: str = Field("", description="Other options considered (wait, different size, etc.)")


@dataclass
class MarketAssessmentToolResult:
    """Result from get_market_assessment tool."""
    prob_yes: float
    confidence: str  # "high", "medium", "low"
    assessment_time: float
    age_seconds: float
    market_probability_at_assessment: float
    mispricing_at_assessment: float
    thesis: str
    key_evidence: List[str]
    what_would_change_mind: str
    event_ticker: Optional[str]
    resolution_criteria: Optional[str]


@dataclass
class MarketMicrostructureToolResult:
    """Result from get_market_microstructure tool."""
    # Executable prices
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    
    # Spreads
    yes_spread: Optional[int]
    no_spread: Optional[int]
    
    # Depth at top-of-book
    yes_bid_size: Optional[int]
    yes_ask_size: Optional[int]
    no_bid_size: Optional[int]
    no_ask_size: Optional[int]
    
    # Recent trades
    last_trade_price: Optional[int]
    last_trade_time: Optional[float]
    recent_trade_count: int
    recent_yes_trades: int
    recent_no_trades: int
    
    # Movement
    price_change_last_5min: int
    
    # Data quality
    orderbook_age_seconds: float
    trade_flow_age_seconds: float
    is_stale: bool


@dataclass
class PositionsAndOrdersToolResult:
    """Result from get_positions_and_orders tool."""
    current_position_side: Optional[str]  # "yes", "no", or None
    current_position_size: int
    avg_entry_price: Optional[int]
    unrealized_pnl_cents: Optional[int]
    open_orders: List[Dict[str, Any]]  # [{order_id, side, qty, price, age_seconds}]
    event_exposure_count: int  # Number of positions in same event
    total_event_exposure_cents: int  # Total notional in same event


class ExecutionAgent:
    """
    LLM-powered execution agent for making intelligent trade decisions.
    
    This agent runs on a fast cadence (every 30s) and uses tools to gather
    current market state, positions, and cached research to make optimal
    execution decisions.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        shadow_mode: bool = True,
        # Position sizing constraints
        default_contracts_per_trade: int = 10,
        max_position_per_market: int = 100,
        max_total_positions: int = 1000,
        max_positions_per_event: int = 100,
        # Event bus for activity feed
        event_bus=None,
    ):
        """
        Initialize execution agent.

        Args:
            openai_api_key: OpenAI API key (loads from env if None)
            model: Model to use (default: gpt-4o-mini for cost efficiency)
            temperature: Temperature for LLM (default: 0.3 for consistent reasoning)
            shadow_mode: If True, log decisions but don't execute trades
            default_contracts_per_trade: Default quantity for new orders
            max_position_per_market: Max contracts allowed per market
            max_total_positions: Max total positions across all markets
            max_positions_per_event: Max positions in correlated markets (same event)
            event_bus: Optional EventBus for emitting decision activity to frontend
        """
        import os
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var.")

        self._model = model
        self._temperature = temperature
        self._shadow_mode = shadow_mode

        # Position sizing constraints (passed to prompt)
        self._default_contracts_per_trade = default_contracts_per_trade
        self._max_position_per_market = max_position_per_market
        self._max_total_positions = max_total_positions
        self._max_positions_per_event = max_positions_per_event

        self._llm = ChatOpenAI(
            model=self._model,
            temperature=self._temperature,
            api_key=self._api_key,
        )

        # Decision log for observability
        self._decision_log: List[Dict[str, Any]] = []

        # Event bus for activity feed
        self._event_bus = event_bus

        logger.info(
            f"ExecutionAgent initialized (model={model}, shadow_mode={shadow_mode}, "
            f"default_qty={default_contracts_per_trade}, max_per_market={max_position_per_market})"
        )

    async def _emit_decision_activity(
        self,
        market_ticker: str,
        plan: ExecutionPlan,
        execution_results: List[Dict[str, Any]],
        assessment: MarketAssessmentToolResult,
    ) -> None:
        """
        Emit execution decision activity to frontend via event bus.

        Args:
            market_ticker: Market evaluated
            plan: The execution plan from LLM
            execution_results: Results of action execution attempts
            assessment: The market assessment used for decision
        """
        if not self._event_bus:
            return

        try:
            # Determine decision outcome
            actions_count = len(plan.actions)
            executed_count = sum(1 for r in execution_results if r.get("executed"))
            shadow_count = sum(1 for r in execution_results if r.get("error") == "shadow_mode")

            if actions_count == 0:
                decision_type = "hold"
                message = f"{market_ticker}: Hold (no action needed)"
            elif executed_count > 0:
                # Format action details
                action_desc = []
                for action in plan.actions:
                    if action.type == ActionType.PLACE_ORDER:
                        action_desc.append(f"{action.side.upper()} {action.qty}@{action.limit_price}c")
                    elif action.type == ActionType.HOLD:
                        action_desc.append("HOLD")
                action_str = ", ".join(action_desc) if action_desc else "action"
                decision_type = "executed"
                message = f"{market_ticker}: Executed {action_str}"
            elif shadow_count > 0:
                action_desc = []
                for action in plan.actions:
                    if action.type == ActionType.PLACE_ORDER:
                        action_desc.append(f"{action.side.upper()} {action.qty}@{action.limit_price}c")
                action_str = ", ".join(action_desc) if action_desc else "action"
                decision_type = "shadow"
                message = f"{market_ticker}: Shadow {action_str}"
            else:
                decision_type = "planned"
                message = f"{market_ticker}: {actions_count} actions planned"

            await self._event_bus.emit_system_activity(
                activity_type="execution_decision",
                message=message,
                metadata={
                    "market_ticker": market_ticker,
                    "decision_type": decision_type,
                    "actions_count": actions_count,
                    "executed_count": executed_count,
                    "shadow_mode": self._shadow_mode,
                    "expected_value_cents": plan.expected_value_cents,
                    "trade_rationale": plan.trade_rationale[:100] if plan.trade_rationale else "",
                    "ai_probability": assessment.prob_yes,
                    "confidence": assessment.confidence,
                    "assessment_age_seconds": round(assessment.age_seconds, 1),
                },
            )
        except Exception as e:
            # Don't let activity emission failures break execution
            logger.debug(f"Failed to emit decision activity: {e}")

    async def decide_actions(
        self,
        market_ticker: str,
        get_assessment_fn,
        get_microstructure_fn,
        get_positions_fn,
        execute_fn,
    ) -> Optional[ExecutionPlan]:
        """
        Make execution decision for a market using tool functions.
        
        Args:
            market_ticker: Market to evaluate
            get_assessment_fn: Callable returning MarketAssessmentToolResult
            get_microstructure_fn: Callable returning MarketMicrostructureToolResult
            get_positions_fn: Callable returning PositionsAndOrdersToolResult
            execute_fn: Callable to execute actions (will be no-op in shadow mode)
            
        Returns:
            ExecutionPlan if decision made, None if skipped/error
        """
        try:
            # Gather tool data
            assessment = await get_assessment_fn(market_ticker)
            microstructure = await get_microstructure_fn(market_ticker)
            positions = await get_positions_fn(market_ticker)
            
            # Build prompt with tool data
            prompt = self._build_decision_prompt(
                market_ticker=market_ticker,
                assessment=assessment,
                microstructure=microstructure,
                positions=positions,
            )
            
            # Get structured output from LLM
            chain = prompt | self._llm.with_structured_output(ExecutionPlan)
            plan = await chain.ainvoke({})
            
            # Log decision
            decision_log_entry = {
                "timestamp": time.time(),
                "market_ticker": market_ticker,
                "plan": plan.model_dump(),
                "tool_snapshots": {
                    "assessment": {
                        "prob_yes": assessment.prob_yes,
                        "confidence": assessment.confidence,
                        "age_seconds": assessment.age_seconds,
                    },
                    "microstructure": {
                        "yes_bid": microstructure.yes_bid,
                        "yes_ask": microstructure.yes_ask,
                        "no_bid": microstructure.no_bid,
                        "no_ask": microstructure.no_ask,
                        "is_stale": microstructure.is_stale,
                    },
                    "positions": {
                        "position_size": positions.current_position_size,
                        "position_side": positions.current_position_side,
                        "open_orders_count": len(positions.open_orders),
                    },
                },
            }
            self._decision_log.append(decision_log_entry)
            if len(self._decision_log) > 1000:
                self._decision_log = self._decision_log[-1000:]
            
            # Execute actions (if not shadow mode) and track results
            execution_results = []
            if not self._shadow_mode:
                for action in plan.actions:
                    if action.type == ActionType.PLACE_ORDER:
                        try:
                            await execute_fn(action)
                            execution_results.append({
                                "action": action.model_dump(),
                                "executed": True,
                                "error": None,
                            })
                        except Exception as e:
                            execution_results.append({
                                "action": action.model_dump(),
                                "executed": False,
                                "error": str(e),
                            })
                    else:
                        execution_results.append({
                            "action": action.model_dump(),
                            "executed": False,
                            "error": "Action type not yet implemented",
                        })
            else:
                # Shadow mode: log what would have been executed
                execution_results = [
                    {"action": a.model_dump(), "executed": False, "error": "shadow_mode"}
                    for a in plan.actions
                ]
            
            # Add execution results to decision log
            decision_log_entry["execution_results"] = execution_results
            
            logger.info(
                f"[EXECUTION_AGENT] {market_ticker}: {len(plan.actions)} actions, "
                f"EV={plan.expected_value_cents}c (shadow={self._shadow_mode}, "
                f"executed={sum(1 for r in execution_results if r.get('executed'))})"
            )

            # Emit decision to activity feed
            await self._emit_decision_activity(
                market_ticker=market_ticker,
                plan=plan,
                execution_results=execution_results,
                assessment=assessment,
            )

            return plan
            
        except Exception as e:
            logger.error(f"Execution agent decision failed for {market_ticker}: {e}", exc_info=True)
            return None
    
    def _build_decision_prompt(
        self,
        market_ticker: str,
        assessment: MarketAssessmentToolResult,
        microstructure: MarketMicrostructureToolResult,
        positions: PositionsAndOrdersToolResult,
    ) -> ChatPromptTemplate:
        """Build prompt for execution decision."""

        # Calculate position headroom
        current_position = positions.current_position_size
        max_additional = self._max_position_per_market - current_position
        event_positions_remaining = self._max_positions_per_event - positions.event_exposure_count

        # Format assessment data
        assessment_text = f"""
MARKET ASSESSMENT (from research, age: {assessment.age_seconds:.0f}s):
- Probability (YES): {assessment.prob_yes:.1%}
- Confidence: {assessment.confidence}
- Edge at assessment time: {assessment.mispricing_at_assessment:+.1%}
- Thesis: {assessment.thesis}
- Key Evidence: {', '.join(assessment.key_evidence[:3])}
- Would change mind if: {assessment.what_would_change_mind}
"""

        # Format microstructure data
        microstructure_text = f"""
CURRENT MARKET STATE:
- YES: bid={microstructure.yes_bid}c, ask={microstructure.yes_ask}c, spread={microstructure.yes_spread or 'N/A'}c
- NO: bid={microstructure.no_bid}c, ask={microstructure.no_ask}c, spread={microstructure.no_spread or 'N/A'}c
- Depth (YES): bid={microstructure.yes_bid_size or 0}, ask={microstructure.yes_ask_size or 0}
- Depth (NO): bid={microstructure.no_bid_size or 0}, ask={microstructure.no_ask_size or 0}
- Recent trades: {microstructure.recent_trade_count} total ({microstructure.recent_yes_trades} YES, {microstructure.recent_no_trades} NO)
- Price change (5min): {microstructure.price_change_last_5min:+d}c
- Data freshness: orderbook={microstructure.orderbook_age_seconds:.0f}s, trades={microstructure.trade_flow_age_seconds:.0f}s
- Stale: {microstructure.is_stale}
"""

        # Format positions data with constraints
        positions_text = f"""
CURRENT POSITIONS & ORDERS:
- Position: {positions.current_position_size} contracts {positions.current_position_side or 'none'}
- Entry price: {positions.avg_entry_price or 'N/A'}c
- Unrealized P&L: {positions.unrealized_pnl_cents or 0}c
- Open orders: {len(positions.open_orders)}
- Event exposure: {positions.event_exposure_count} positions, {positions.total_event_exposure_cents}c notional

POSITION CONSTRAINTS (must be respected):
- Default trade size: {self._default_contracts_per_trade} contracts
- Max position per market: {self._max_position_per_market} contracts (can add up to {max(0, max_additional)} more)
- Max positions per event: {self._max_positions_per_event} (can open {max(0, event_positions_remaining)} more in this event)
"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an execution trader. Your job is to maximize risk-adjusted expected value using the provided tool data.

CORE PRINCIPLES:
1. Compute expected value at EXECUTABLE PRICES (use ask for buying, bid for selling)
2. Consider spread, slippage, and data staleness as real costs
3. Do not trade based on stale data unless you explicitly justify why it's still acceptable
4. Consider portfolio exposure and correlation risks
5. Prefer fewer, higher-quality trades over churn
6. If edge exists but conditions are poor (wide spread, stale data, high correlation), consider waiting

POSITION SIZING RULES (mandatory):
- Use the default trade size unless there's a specific reason to deviate
- NEVER exceed the max position per market constraint
- NEVER exceed the max positions per event constraint (correlated markets)
- Consider reducing size if: liquidity is thin, spread is wide, or confidence is low

OUTPUT REQUIREMENTS:
- Provide clear rationale referencing microstructure + portfolio
- Compute expected value at executable price (not mid)
- List risks explicitly (liquidity, staleness, correlation, tail)
- Consider alternatives (wait, smaller size, better price)"""),

            ("user", f"""Market: {market_ticker}

{assessment_text}

{microstructure_text}

{positions_text}

DECISION TASK:
Given this market assessment, current trading state, and portfolio context, decide what actions to take.

You can:
- place_order: Buy YES or NO at a specific limit price
- hold: Wait for better conditions
- cancel_order: Cancel an existing open order
- replace_order: Modify an existing order's price/quantity

For each action, provide:
- type: action type
- market_ticker: market (for place orders)
- side: "yes" or "no" (for place orders)
- qty: number of contracts (use default size unless justified, respect max constraints)
- limit_price: price in cents (for place orders)
- reason: brief explanation

Your plan should include:
- trade_rationale: Why these actions are optimal now
- expected_value_cents: Expected value at executable price
- risk_notes: Explicit risks
- data_quality: Assessment of tool data
- alternatives_considered: What else you considered""")
        ])

        return prompt
    
    def get_decision_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decision history for debugging."""
        return self._decision_log[-limit:]
    
    def set_shadow_mode(self, shadow_mode: bool) -> None:
        """Enable/disable shadow mode."""
        self._shadow_mode = shadow_mode
        logger.info(f"ExecutionAgent shadow_mode={shadow_mode}")
