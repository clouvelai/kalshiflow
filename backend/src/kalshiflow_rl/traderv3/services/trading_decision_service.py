"""
Trading Decision Service for V3 trader.

Handles trading logic and decision-making for the V3 trading system.
This service is designed to be the single point for all trading decisions.

Key Responsibilities:
    1. Evaluate market conditions and generate trading decisions
    2. Execute trades through the trading client
    3. Support HOLD and RLM_NO strategies

Design Principles:
    - Single point for all trading decisions
    - Strategy pattern for easy extension
    - Event-driven updates via EventBus
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass, field

from ..state.trading_attachment import TrackedMarketOrder
from ..state.order_context import (
    StagedOrderContext,
    OrderbookSnapshot,
    PositionContext,
    MarketContext,
    SpreadTier,
)
from .order_context_service import get_order_context_service

from ...data.orderbook_state import get_shared_orderbook_state

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..clients.orderbook_integration import V3OrderbookIntegration
    from ..core.state_container import V3StateContainer
    from ..core.event_bus import EventBus
    from ..config.environment import V3Config
    from kalshiflow_rl.data.models import OrderbookSnapshot

logger = logging.getLogger("kalshiflow_rl.traderv3.services.trading_decision")


class TradingStrategy(Enum):
    """Available trading strategies."""
    HOLD = "hold"  # Never trade (safe default)
    RLM_NO = "rlm_no"  # Reverse Line Movement NO (validated +17.38% edge)


@dataclass
class TradingDecision:
    """
    Represents a trading decision.

    Attributes:
        action: "buy", "sell", or "hold"
        market: Market ticker
        side: "yes" or "no"
        quantity: Number of contracts
        price: Limit price in cents (None for market order)
        reason: Human-readable explanation
        confidence: Confidence score (0.0 - 1.0)
        strategy: DEPRECATED - Use strategy_id instead. Kept for backward compatibility.
        strategy_id: String identifier for the strategy (e.g., "rlm_no", "s013").
                    This is the preferred field for strategy attribution.
        signal_params: Strategy-specific parameters for quant analysis
    """
    action: str  # "buy", "sell", "hold"
    market: str
    side: str  # "yes" or "no"
    quantity: int
    price: Optional[int] = None  # If None, use market order
    reason: str = ""
    confidence: float = 0.0
    strategy: TradingStrategy = TradingStrategy.HOLD  # DEPRECATED: Use strategy_id
    strategy_id: str = ""  # New: String identifier for plugin system
    signal_params: Dict[str, Any] = field(default_factory=dict)  # Strategy-specific params for quant analysis

    def __post_init__(self):
        """Set strategy_id from strategy enum if not provided."""
        if not self.strategy_id and self.strategy != TradingStrategy.HOLD:
            self.strategy_id = self.strategy.value


class TradingDecisionService:
    """
    Service for making trading decisions in the V3 trader.

    Responsibilities:
    - Evaluate market conditions
    - Generate trading decisions
    - Execute trades through trading client
    - Track decision history
    - Support HOLD and RLM_NO strategies
    """

    def __init__(
        self,
        trading_client: Optional['V3TradingClientIntegration'],
        state_container: 'V3StateContainer',
        event_bus: 'EventBus',
        strategy: TradingStrategy = TradingStrategy.HOLD,
        config: Optional['V3Config'] = None,
        orderbook_integration: Optional['V3OrderbookIntegration'] = None
    ):
        """
        Initialize trading decision service.

        Args:
            trading_client: Trading client integration for order execution
            state_container: State container for accessing system state
            event_bus: Event bus for emitting trading events
            strategy: Trading strategy to use
            config: Optional V3 config for balance protection settings
            orderbook_integration: Optional orderbook integration for session_id and orderbook access
        """
        self._trading_client = trading_client
        self._state_container = state_container
        self._event_bus = event_bus
        self._strategy = strategy
        self._orderbook_integration = orderbook_integration

        # Balance protection (from config or default $100.00)
        self._min_trader_cash = config.min_trader_cash if config else 10000

        # Decision tracking
        self._decision_count = 0
        self._trade_count = 0
        self._last_decision: Optional[TradingDecision] = None

        # Stats tracking
        self._decision_stats = {
            "low_balance": 0  # Trades skipped due to insufficient balance
        }

        logger.info(f"Trading Decision Service initialized with strategy: {strategy.value}")

    async def evaluate_market(
        self, 
        market: str, 
        orderbook: Optional['OrderbookSnapshot'] = None
    ) -> Optional[TradingDecision]:
        """
        Evaluate a market and generate a trading decision.
        
        Args:
            market: Market ticker to evaluate
            orderbook: Current orderbook snapshot (optional)
        
        Returns:
            Trading decision or None if no action needed
        """
        self._decision_count += 1
        
        # Get current positions and orders
        trading_state = self._state_container.trading_state
        positions = trading_state.positions if trading_state else {}
        orders = trading_state.orders if trading_state else {}

        # Check if we already have exposure in this market
        has_position = market in positions
        # orders is Dict[order_id, order_data] where order_data has 'ticker' key
        has_orders = any(o.get("ticker") == market for o in orders.values())
        
        if has_orders:
            logger.debug(f"Skipping {market} - has open orders")
            return None
        
        # Apply strategy
        decision = None

        if self._strategy == TradingStrategy.HOLD:
            # Never trade
            decision = TradingDecision(
                action="hold",
                market=market,
                side="",
                quantity=0,
                reason="HOLD strategy - no trading",
                strategy=self._strategy
            )

        elif self._strategy == TradingStrategy.RLM_NO:
            # RLM_NO is handled by RLMNoStrategy plugin via event-driven architecture
            # This is a fallback for direct market-based calls
            decision = TradingDecision(
                action="hold",
                market=market,
                side="",
                quantity=0,
                reason="RLM_NO uses event-driven strategy plugin",
                strategy=self._strategy
            )
        
        # Store decision
        if decision and decision.action != "hold":
            self._last_decision = decision
            logger.info(
                f"Trading decision for {market}: {decision.action} "
                f"{decision.quantity} {decision.side} @ {decision.price or 'market'} "
                f"(reason: {decision.reason})"
            )
        
        return decision

    async def execute_decision(self, decision: TradingDecision) -> bool:
        """
        Execute a trading decision.

        Args:
            decision: Trading decision to execute

        Returns:
            True if execution successful, False otherwise
        """
        if not self._trading_client:
            logger.error("No trading client configured - cannot execute trades")
            return False

        if decision.action == "hold":
            return True  # No action needed

        # Check minimum balance protection (only for buy actions)
        if decision.action == "buy" and self._min_trader_cash > 0:
            trading_state = self._state_container.trading_state
            balance = trading_state.balance if trading_state else 0
            if balance < self._min_trader_cash:
                self._decision_stats["low_balance"] += 1
                logger.warning(
                    f"Skipping trade: balance ${balance/100:.2f} < minimum ${self._min_trader_cash/100:.2f} "
                    f"({decision.market} {decision.side} {decision.quantity}x @ {decision.price}c)"
                )
                # Emit activity event for frontend visibility
                await self._event_bus.emit_system_activity(
                    activity_type="low_balance_skip",
                    message=f"Trade skipped: balance ${balance/100:.2f} < ${self._min_trader_cash/100:.2f} minimum",
                    metadata={
                        "market": decision.market,
                        "side": decision.side,
                        "quantity": decision.quantity,
                        "price": decision.price,
                        "balance_cents": balance,
                        "min_required_cents": self._min_trader_cash,
                        "skip_count": self._decision_stats["low_balance"]
                    }
                )
                return False

        try:
            # Emit decision event
            # Use strategy_id if available, fall back to strategy enum for backward compat
            strategy_name = decision.strategy_id or decision.strategy.value
            await self._event_bus.emit_system_activity(
                activity_type="trading_decision",
                message=f"Executing {decision.action} {decision.quantity} {decision.side} on {decision.market}",
                metadata={
                    "market": decision.market,
                    "action": decision.action,
                    "side": decision.side,
                    "quantity": decision.quantity,
                    "price": decision.price,
                    "strategy": strategy_name,
                    "strategy_id": decision.strategy_id,  # New field
                }
            )
            
            # Execute based on action type
            if decision.action == "buy":
                success = await self._execute_buy(decision)
            elif decision.action == "sell":
                success = await self._execute_sell(decision)
            else:
                logger.warning(f"Unknown action: {decision.action}")
                return False
            
            if success:
                self._trade_count += 1
                logger.info(f"Successfully executed trade #{self._trade_count}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="trading_error",
                message=f"Failed to execute trade: {str(e)}",
                metadata={"error": str(e), "decision": decision.__dict__}
            )
            return False

    async def _stage_order_context(
        self,
        order_id: str,
        decision: TradingDecision,
        signal_id: str,
    ) -> None:
        """
        Stage order context for quant analysis.

        Context is staged in memory when order is placed,
        then persisted to DB when fill is confirmed.

        Args:
            order_id: Kalshi order ID
            decision: Trading decision with order details
            signal_id: Unique signal identifier
        """
        try:
            # Get session_id from orderbook integration
            session_id = None
            if self._orderbook_integration and hasattr(self._orderbook_integration, '_client'):
                client_stats = self._orderbook_integration._client.get_stats()
                session_id = client_stats.get("session_id")
                if session_id is not None:
                    session_id = str(session_id)

            # Get orderbook snapshot for this market
            orderbook_snapshot = OrderbookSnapshot()
            try:
                orderbook_state = await asyncio.wait_for(
                    get_shared_orderbook_state(decision.market),
                    timeout=1.0  # Short timeout since we already have the order
                )
                snapshot = await orderbook_state.get_snapshot()
                orderbook_snapshot = OrderbookSnapshot.from_orderbook_state(snapshot)
            except (asyncio.TimeoutError, Exception) as e:
                logger.debug(f"Could not capture orderbook for context: {e}")
                # Continue with empty orderbook - don't fail context staging

            # Get market context from tracked markets
            market_context = MarketContext()
            if self._state_container._tracked_markets:
                tracked_market = self._state_container._tracked_markets.get_market(decision.market)
                if tracked_market:
                    market_context.market_category = tracked_market.category
                    market_context.market_close_ts = tracked_market.close_ts
                    if tracked_market.close_ts:
                        market_context.hours_to_settlement = (tracked_market.close_ts - time.time()) / 3600
                    market_context.trades_in_market = decision.signal_params.get("total_trades", 0)

            # Get position context
            position_context = PositionContext()
            trading_state = self._state_container.trading_state
            if trading_state:
                position_context.balance_cents = trading_state.balance or 0
                position_context.open_position_count = len(trading_state.positions) if trading_state.positions else 0

                # Check existing position in this market
                existing_pos = trading_state.positions.get(decision.market, {}) if trading_state.positions else {}
                if existing_pos:
                    position_context.existing_position_count = abs(existing_pos.get("position", 0))
                    # Determine side from position count sign
                    pos_count = existing_pos.get("position", 0)
                    if pos_count > 0:
                        position_context.existing_position_side = "yes"
                    elif pos_count < 0:
                        position_context.existing_position_side = "no"
                    position_context.is_reentry = True
                    position_context.entry_number = decision.signal_params.get("signal_trigger_count", 1)

            # Calculate NO price at signal for edge calculation
            no_price_at_signal = None
            last_yes_price = decision.signal_params.get("last_yes_price")
            if last_yes_price is not None:
                no_price_at_signal = 100 - last_yes_price

            # Create staged context
            # Use strategy_id if available, fall back to strategy enum for backward compat
            strategy_name = decision.strategy_id or decision.strategy.value
            context = StagedOrderContext(
                order_id=order_id,
                market_ticker=decision.market,
                session_id=session_id,
                strategy=strategy_name,
                signal_id=signal_id,
                signal_detected_at=decision.signal_params.get("signal_detected_at", time.time()),
                signal_params=decision.signal_params,
                market=market_context,
                no_price_at_signal=no_price_at_signal,
                orderbook=orderbook_snapshot,
                position=position_context,
                action=decision.action,
                side=decision.side,
                order_price_cents=decision.price,
                order_quantity=decision.quantity,
                order_type="limit",
                placed_at=time.time(),
                strategy_version="1.0",
            )

            # Stage the context
            order_context_service = get_order_context_service()
            order_context_service.stage_context(context)

            logger.debug(
                f"Staged order context: order_id={order_id[:8]}..., "
                f"strategy={decision.strategy.value}, ticker={decision.market}, "
                f"session={session_id}, ob_captured={orderbook_snapshot.best_bid_cents is not None}"
            )

        except Exception as e:
            logger.warning(f"Failed to stage order context: {e}")
            # Don't fail the trade if context staging fails

    async def _execute_buy(self, decision: TradingDecision) -> bool:
        """
        Execute a buy order through the trading client.

        Args:
            decision: Trading decision with buy details

        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            # Get order group ID for portfolio limits
            order_group_id = self._trading_client.get_order_group_id()

            # Place the order
            response = await self._trading_client.place_order(
                ticker=decision.market,
                action="buy",
                side=decision.side,
                count=decision.quantity,
                price=decision.price,
                order_type="limit",
                order_group_id=order_group_id
            )

            order_id = response.get("order", {}).get("order_id", "unknown")

            # Calculate order cost and record cash flow
            order_cost_cents = decision.quantity * (decision.price or 0)
            self._state_container.record_order_fill(order_cost_cents, decision.quantity)

            logger.info(
                f"BUY order placed: {decision.quantity} {decision.side} {decision.market} "
                f"@ {decision.price}c = {order_cost_cents}¢ (order_id: {order_id[:8]}...)"
            )

            # Track order in trading attachment for tracked markets
            # Status is "resting" since API confirmed the order is on the book
            signal_id = f"{decision.reason}:{decision.market}:{int(time.time() * 1000)}"
            strategy_id = decision.strategy_id or decision.strategy.value
            await self._state_container.update_order_in_attachment(
                ticker=decision.market,
                order_id=order_id,
                order_data=TrackedMarketOrder(
                    order_id=order_id,
                    signal_id=signal_id,
                    action="buy",
                    side=decision.side,
                    count=decision.quantity,
                    price=decision.price or 0,
                    status="resting",  # Order confirmed on book
                    placed_at=time.time(),
                    strategy_id=strategy_id,
                )
            )

            # Stage order context for quant analysis (persisted on fill)
            await self._stage_order_context(order_id, decision, signal_id)

            # Emit order_placed activity for frontend visibility
            strategy_name = decision.strategy_id or decision.strategy.value
            await self._event_bus.emit_system_activity(
                activity_type="order_placed",
                message=f"Order placed: BUY {decision.quantity} {decision.side.upper()} @ {decision.price}c",
                metadata={
                    "order_id": order_id,
                    "ticker": decision.market,
                    "action": "buy",
                    "side": decision.side,
                    "count": decision.quantity,
                    "price_cents": decision.price,
                    "cost_cents": order_cost_cents,
                    "strategy": strategy_name,
                    "strategy_id": decision.strategy_id,
                }
            )

            return True

        except Exception as e:
            logger.error(f"BUY order failed: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="trading_error",
                message=f"BUY order failed: {str(e)}",
                metadata={
                    "market": decision.market,
                    "side": decision.side,
                    "quantity": decision.quantity,
                    "price": decision.price,
                    "error": str(e)
                }
            )
            return False

    async def _execute_sell(self, decision: TradingDecision) -> bool:
        """
        Execute a sell order through the trading client.

        Args:
            decision: Trading decision with sell details

        Returns:
            True if order placed successfully, False otherwise
        """
        try:
            # Get order group ID for portfolio limits
            order_group_id = self._trading_client.get_order_group_id()

            # Place the order
            response = await self._trading_client.place_order(
                ticker=decision.market,
                action="sell",
                side=decision.side,
                count=decision.quantity,
                price=decision.price,
                order_type="limit",
                order_group_id=order_group_id
            )

            order_id = response.get("order", {}).get("order_id", "unknown")

            logger.info(
                f"SELL order placed: {decision.quantity} {decision.side} {decision.market} "
                f"@ {decision.price}c (order_id: {order_id[:8]}...)"
            )

            # Track order in trading attachment for tracked markets
            # Status is "resting" since API confirmed the order is on the book
            signal_id = f"{decision.reason}:{decision.market}:{int(time.time() * 1000)}"
            strategy_id = decision.strategy_id or decision.strategy.value
            await self._state_container.update_order_in_attachment(
                ticker=decision.market,
                order_id=order_id,
                order_data=TrackedMarketOrder(
                    order_id=order_id,
                    signal_id=signal_id,
                    action="sell",
                    side=decision.side,
                    count=decision.quantity,
                    price=decision.price or 0,
                    status="resting",  # Order confirmed on book
                    placed_at=time.time(),
                    strategy_id=strategy_id,
                )
            )

            # Stage order context for quant analysis (persisted on fill)
            await self._stage_order_context(order_id, decision, signal_id)

            # Emit order_placed activity for frontend visibility
            strategy_name = decision.strategy_id or decision.strategy.value
            await self._event_bus.emit_system_activity(
                activity_type="order_placed",
                message=f"Order placed: SELL {decision.quantity} {decision.side.upper()} @ {decision.price}c",
                metadata={
                    "order_id": order_id,
                    "ticker": decision.market,
                    "action": "sell",
                    "side": decision.side,
                    "count": decision.quantity,
                    "price_cents": decision.price,
                    "strategy": strategy_name,
                    "strategy_id": decision.strategy_id,
                }
            )

            return True

        except Exception as e:
            logger.error(f"SELL order failed: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="trading_error",
                message=f"SELL order failed: {str(e)}",
                metadata={
                    "market": decision.market,
                    "side": decision.side,
                    "quantity": decision.quantity,
                    "price": decision.price,
                    "error": str(e)
                }
            )
            return False
    
    def set_strategy(self, strategy: TradingStrategy) -> None:
        """Change the trading strategy."""
        logger.info(f"Changing strategy from {self._strategy.value} to {strategy.value}")
        self._strategy = strategy
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "strategy": self._strategy.value,
            "decision_count": self._decision_count,
            "trade_count": self._trade_count,
            "last_decision": self._last_decision.__dict__ if self._last_decision else None,
            "low_balance_skips": self._decision_stats["low_balance"]
        }

    def get_decision_stats(self) -> Dict[str, int]:
        """Get decision statistics (for WebSocket heartbeat)."""
        return self._decision_stats.copy()

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return True  # Simple health check for now