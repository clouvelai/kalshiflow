"""
Trading Decision Service for V3 trader.

Handles trading logic and decision-making for the V3 trading system.
This service is designed to be the single point for all trading decisions,
supporting both hardcoded strategies and RL model integration.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.state_container import V3StateContainer
    from ..core.event_bus import EventBus
    from kalshiflow_rl.data.models import OrderbookSnapshot

logger = logging.getLogger("kalshiflow_rl.traderv3.services.trading_decision")


class TradingStrategy(Enum):
    """Available trading strategies."""
    HOLD = "hold"  # Never trade (safe default)
    PAPER_TEST = "paper_test"  # Simple test trades for paper mode
    RL_MODEL = "rl_model"  # Use trained RL model
    CUSTOM = "custom"  # Custom strategy implementation


@dataclass
class TradingDecision:
    """Represents a trading decision."""
    action: str  # "buy", "sell", "hold"
    market: str
    side: str  # "yes" or "no"
    quantity: int
    price: Optional[int] = None  # If None, use market order
    reason: str = ""
    confidence: float = 0.0
    strategy: TradingStrategy = TradingStrategy.HOLD


class TradingDecisionService:
    """
    Service for making trading decisions in the V3 trader.
    
    Responsibilities:
    - Evaluate market conditions
    - Generate trading decisions
    - Execute trades through trading client
    - Track decision history
    - Support multiple strategies (HOLD, RL, custom)
    """
    
    def __init__(
        self,
        trading_client: Optional['V3TradingClientIntegration'],
        state_container: 'V3StateContainer',
        event_bus: 'EventBus',
        strategy: TradingStrategy = TradingStrategy.HOLD
    ):
        """
        Initialize trading decision service.
        
        Args:
            trading_client: Trading client integration for order execution
            state_container: State container for accessing system state
            event_bus: Event bus for emitting trading events
            strategy: Trading strategy to use
        """
        self._trading_client = trading_client
        self._state_container = state_container
        self._event_bus = event_bus
        self._strategy = strategy
        
        # Decision tracking
        self._decision_count = 0
        self._trade_count = 0
        self._last_decision: Optional[TradingDecision] = None
        
        # RL model (loaded lazily if needed)
        self._rl_model = None
        self._rl_model_loaded = False
        
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
        open_orders = trading_state.open_orders if trading_state else []
        
        # Check if we already have exposure in this market
        has_position = market in positions
        has_orders = any(o.ticker == market for o in open_orders)
        
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
        
        elif self._strategy == TradingStrategy.PAPER_TEST:
            # Simple test strategy for paper mode
            decision = await self._paper_test_strategy(
                market, orderbook, has_position
            )
        
        elif self._strategy == TradingStrategy.RL_MODEL:
            # Use RL model for decision
            decision = await self._rl_model_strategy(
                market, orderbook, has_position
            )
        
        elif self._strategy == TradingStrategy.CUSTOM:
            # Custom strategy implementation
            decision = await self._custom_strategy(
                market, orderbook, has_position
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
    
    async def _paper_test_strategy(
        self,
        market: str,
        orderbook: Optional['OrderbookSnapshot'],
        has_position: bool
    ) -> TradingDecision:
        """
        Simple test strategy for paper trading validation.
        
        Buys 5 contracts at mid-price if no position,
        sells if we have a position.
        """
        if not orderbook:
            return TradingDecision(
                action="hold",
                market=market,
                side="",
                quantity=0,
                reason="No orderbook data",
                strategy=self._strategy
            )
        
        # Calculate mid price
        best_bid = orderbook.yes[0][0] if orderbook.yes else 0
        best_ask = orderbook.no[0][0] if orderbook.no else 100
        mid_price = (best_bid + best_ask) // 2
        
        if has_position:
            # Sell position
            return TradingDecision(
                action="sell",
                market=market,
                side="yes",  # Assuming we bought yes
                quantity=5,
                price=mid_price,
                reason="Paper test: closing position",
                confidence=0.5,
                strategy=self._strategy
            )
        else:
            # Buy if price is reasonable
            if 20 <= mid_price <= 80:
                return TradingDecision(
                    action="buy",
                    market=market,
                    side="yes" if mid_price < 50 else "no",
                    quantity=5,
                    price=mid_price,
                    reason=f"Paper test: mid price {mid_price}",
                    confidence=0.5,
                    strategy=self._strategy
                )
        
        return TradingDecision(
            action="hold",
            market=market,
            side="",
            quantity=0,
            reason="Paper test: conditions not met",
            strategy=self._strategy
        )
    
    async def _rl_model_strategy(
        self,
        market: str,
        orderbook: Optional['OrderbookSnapshot'],
        has_position: bool
    ) -> TradingDecision:
        """
        Use RL model for trading decision.
        """
        # TODO: Implement RL model integration
        # For now, return hold
        return TradingDecision(
            action="hold",
            market=market,
            side="",
            quantity=0,
            reason="RL model not yet integrated",
            strategy=self._strategy
        )
    
    async def _custom_strategy(
        self,
        market: str,
        orderbook: Optional['OrderbookSnapshot'],
        has_position: bool
    ) -> TradingDecision:
        """
        Custom trading strategy implementation.
        Override this method for custom strategies.
        """
        return TradingDecision(
            action="hold",
            market=market,
            side="",
            quantity=0,
            reason="Custom strategy not implemented",
            strategy=self._strategy
        )
    
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
        
        try:
            # Emit decision event
            await self._event_bus.emit_system_activity(
                activity_type="trading_decision",
                message=f"Executing {decision.action} {decision.quantity} {decision.side} on {decision.market}",
                metadata={
                    "market": decision.market,
                    "action": decision.action,
                    "side": decision.side,
                    "quantity": decision.quantity,
                    "price": decision.price,
                    "strategy": decision.strategy.value
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
    
    async def _execute_buy(self, decision: TradingDecision) -> bool:
        """Execute a buy order."""
        # TODO: Implement actual order placement through trading client
        logger.info(f"Would execute BUY: {decision}")
        return True  # Placeholder
    
    async def _execute_sell(self, decision: TradingDecision) -> bool:
        """Execute a sell order."""
        # TODO: Implement actual order placement through trading client
        logger.info(f"Would execute SELL: {decision}")
        return True  # Placeholder
    
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
            "rl_model_loaded": self._rl_model_loaded
        }
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return True  # Simple health check for now