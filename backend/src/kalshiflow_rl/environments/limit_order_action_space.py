"""
Limit Order Action Space for the Kalshi Flow RL Trading Subsystem.

This module implements a 5-action space that works with Kalshi's limit-only order system
using the OrderManager abstraction. The agent outputs simple intents (0-4) and the
OrderManager handles all order complexity including pricing, cancellation, and state management.

Key Design Principles:
- Agent outputs simple trading intents (HOLD, BUY_YES, SELL_YES, BUY_NO, SELL_NO)
- OrderManager translates intents into proper limit orders with pricing
- Stateful order management is abstracted away from RL agent
- Clean separation between strategy (agent) and execution (OrderManager)

Action Space:
0: HOLD - No action
1: BUY_YES_LIMIT - Place/maintain buy order for YES contracts  
2: SELL_YES_LIMIT - Place/maintain sell order for YES contracts
3: BUY_NO_LIMIT - Place/maintain buy order for NO contracts
4: SELL_NO_LIMIT - Place/maintain sell order for NO contracts

The OrderManager handles:
- Cancelling conflicting orders when switching sides
- Pricing limit orders based on current market state
- Amending orders when prices change
- Tracking order state for observation features
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import IntEnum
import logging

import numpy as np
from gymnasium import spaces

from ..trading.order_manager import OrderManager, OrderSide, ContractSide
from ..data.orderbook_state import OrderbookState

logger = logging.getLogger("kalshiflow_rl.environments.limit_order_action_space")


class ActionType(IntEnum):
    """
    Action types for trading integration compatibility.
    
    This enum provides compatibility with the existing TradingSession integration
    by mapping to LimitOrderActions. Used primarily by the trading integration module.
    """
    HOLD = 0
    BUY_YES = 1  # Maps to BUY_YES_LIMIT
    SELL_YES = 2  # Maps to SELL_YES_LIMIT
    BUY_NO = 3  # Maps to BUY_NO_LIMIT
    SELL_NO = 4  # Maps to SELL_NO_LIMIT
    CLOSE_POSITION = 5  # Special action for closing positions


class LimitOrderActions(IntEnum):
    """
    Enumeration of the 5 limit order actions available to the agent.
    
    These actions represent trading intents that the OrderManager translates
    into appropriate limit orders with proper pricing and state management.
    """
    HOLD = 0              # No action - maintain current orders
    BUY_YES_LIMIT = 1     # Place/maintain buy order for YES contracts
    SELL_YES_LIMIT = 2    # Place/maintain sell order for YES contracts
    BUY_NO_LIMIT = 3      # Place/maintain buy order for NO contracts
    SELL_NO_LIMIT = 4     # Place/maintain sell order for NO contracts


@dataclass
class ActionExecutionResult:
    """Result of executing an action through the OrderManager."""
    action_taken: LimitOrderActions
    order_placed: bool
    order_cancelled: bool
    order_amended: bool
    order_id: Optional[str]
    error_message: Optional[str] = None
    
    def was_successful(self) -> bool:
        """Check if action was executed successfully."""
        return self.error_message is None


class LimitOrderActionSpace:
    """
    Limit order action space for Kalshi trading with OrderManager integration.
    
    This class implements a 5-action space designed to work with Kalshi's limit-only
    order system. The agent outputs simple trading intents, and the OrderManager
    handles all order complexity including pricing, cancellation, and state tracking.
    
    Features:
    - 5 discrete actions: HOLD + 4 limit order intents
    - OrderManager handles all order complexity
    - Automatic order cancellation when switching sides
    - Order amendment when market prices change significantly
    - Fixed 10 contract size for all trades
    - Market-agnostic pricing strategies
    
    Usage:
        action_space = LimitOrderActionSpace(order_manager, contract_size=10)
        result = await action_space.execute_action(action, ticker, orderbook)
    """
    
    def __init__(
        self, 
        order_manager: OrderManager,
        contract_size: int = 10,
        pricing_strategy: str = "aggressive"
    ):
        """
        Initialize the limit order action space.
        
        Args:
            order_manager: OrderManager instance for execution
            contract_size: Fixed number of contracts per trade
            pricing_strategy: Default pricing strategy ("aggressive", "passive", "mid")
        """
        self.order_manager = order_manager
        self.contract_size = contract_size
        self.pricing_strategy = pricing_strategy
        
        self.action_descriptions = {
            LimitOrderActions.HOLD: "No action - maintain current orders",
            LimitOrderActions.BUY_YES_LIMIT: "Place/maintain buy order for YES contracts",
            LimitOrderActions.SELL_YES_LIMIT: "Place/maintain sell order for YES contracts",
            LimitOrderActions.BUY_NO_LIMIT: "Place/maintain buy order for NO contracts", 
            LimitOrderActions.SELL_NO_LIMIT: "Place/maintain sell order for NO contracts"
        }
        
        logger.info(f"LimitOrderActionSpace initialized with {contract_size} contract size")
    
    def get_gym_space(self) -> spaces.Discrete:
        """
        Get the Gymnasium space representation.
        
        Returns:
            spaces.Discrete(5): Discrete action space with 5 actions
        """
        return spaces.Discrete(5)
    
    async def execute_action(
        self,
        action: int,
        ticker: str,
        orderbook: OrderbookState
    ) -> ActionExecutionResult:
        """
        Execute a trading action through the OrderManager.
        
        This method translates high-level agent actions into specific order
        management operations, handling all the complexity of limit orders.
        
        Args:
            action: Integer action from 0-4
            ticker: Market ticker for the order
            orderbook: Current orderbook state for pricing
            
        Returns:
            ActionExecutionResult with details of what was executed
        """
        try:
            if not (0 <= action <= 4):
                return ActionExecutionResult(
                    action_taken=LimitOrderActions(action),
                    order_placed=False,
                    order_cancelled=False,
                    order_amended=False,
                    order_id=None,
                    error_message=f"Invalid action: {action}"
                )
            
            action_enum = LimitOrderActions(action)
            
            # Handle HOLD action
            if action_enum == LimitOrderActions.HOLD:
                return await self._execute_hold_action(ticker, orderbook)
            
            # Handle trading actions
            elif action_enum == LimitOrderActions.BUY_YES_LIMIT:
                return await self._execute_buy_action(ticker, ContractSide.YES, orderbook)
            
            elif action_enum == LimitOrderActions.SELL_YES_LIMIT:
                return await self._execute_sell_action(ticker, ContractSide.YES, orderbook)
            
            elif action_enum == LimitOrderActions.BUY_NO_LIMIT:
                return await self._execute_buy_action(ticker, ContractSide.NO, orderbook)
            
            elif action_enum == LimitOrderActions.SELL_NO_LIMIT:
                return await self._execute_sell_action(ticker, ContractSide.NO, orderbook)
            
            else:
                return ActionExecutionResult(
                    action_taken=action_enum,
                    order_placed=False,
                    order_cancelled=False,
                    order_amended=False,
                    order_id=None,
                    error_message=f"Unhandled action: {action_enum}"
                )
                
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return ActionExecutionResult(
                action_taken=LimitOrderActions(action) if 0 <= action <= 4 else LimitOrderActions.HOLD,
                order_placed=False,
                order_cancelled=False,
                order_amended=False,
                order_id=None,
                error_message=str(e)
            )
    
    async def _execute_hold_action(
        self,
        ticker: str,
        orderbook: OrderbookState
    ) -> ActionExecutionResult:
        """
        Execute HOLD action - maintain current orders, possibly amend if needed.
        
        For HOLD actions, we check if existing orders need price adjustment
        based on market movement, but don't place new orders.
        """
        open_orders = self.order_manager.get_open_orders(ticker)
        
        amended_count = 0
        last_order_id = None
        
        # Check if any existing orders need price adjustment
        for order in open_orders:
            # Calculate what the current optimal price should be
            optimal_price = self.order_manager._calculate_limit_price(
                order.side, order.contract_side, orderbook, self.pricing_strategy
            )
            
            # Amend if price has moved significantly (>1 cent difference)
            if abs(optimal_price - order.limit_price) > 1:
                success = await self.order_manager.amend_order(order.order_id, optimal_price, orderbook)
                if success:
                    amended_count += 1
                    last_order_id = order.order_id
                    logger.debug(f"Amended order {order.order_id}: {order.limit_price}¢ → {optimal_price}¢")
        
        return ActionExecutionResult(
            action_taken=LimitOrderActions.HOLD,
            order_placed=False,
            order_cancelled=False,
            order_amended=amended_count > 0,
            order_id=last_order_id
        )
    
    async def _execute_buy_action(
        self,
        ticker: str,
        contract_side: ContractSide,
        orderbook: OrderbookState
    ) -> ActionExecutionResult:
        """
        Execute a buy action - place buy order, cancelling conflicting sells.
        
        This method implements the "one order per market" rule by cancelling
        any conflicting sell orders before placing the new buy order.
        """
        # Cancel conflicting orders (any sell orders)
        open_orders = self.order_manager.get_open_orders(ticker)
        conflicting_orders = [
            order for order in open_orders 
            if order.side == OrderSide.SELL or order.contract_side != contract_side
        ]
        
        cancelled_count = 0
        for order in conflicting_orders:
            success = await self.order_manager.cancel_order(order.order_id)
            if success:
                cancelled_count += 1
        
        # Check if we already have a compatible buy order
        existing_buy_orders = [
            order for order in open_orders
            if order.side == OrderSide.BUY and order.contract_side == contract_side
        ]
        
        if existing_buy_orders:
            # Already have a buy order - just amend if needed
            existing_order = existing_buy_orders[0]
            optimal_price = self.order_manager._calculate_limit_price(
                OrderSide.BUY, contract_side, orderbook, self.pricing_strategy
            )
            
            if abs(optimal_price - existing_order.limit_price) > 1:
                success = await self.order_manager.amend_order(
                    existing_order.order_id, optimal_price, orderbook
                )
                return ActionExecutionResult(
                    action_taken=LimitOrderActions.BUY_YES_LIMIT if contract_side == ContractSide.YES 
                                else LimitOrderActions.BUY_NO_LIMIT,
                    order_placed=False,
                    order_cancelled=cancelled_count > 0,
                    order_amended=success,
                    order_id=existing_order.order_id
                )
            else:
                # Order is fine as-is
                return ActionExecutionResult(
                    action_taken=LimitOrderActions.BUY_YES_LIMIT if contract_side == ContractSide.YES 
                                else LimitOrderActions.BUY_NO_LIMIT,
                    order_placed=False,
                    order_cancelled=cancelled_count > 0,
                    order_amended=False,
                    order_id=existing_order.order_id
                )
        
        # Place new buy order
        order = await self.order_manager.place_order(
            ticker=ticker,
            side=OrderSide.BUY,
            contract_side=contract_side,
            quantity=self.contract_size,
            orderbook=orderbook,
            pricing_strategy=self.pricing_strategy
        )
        
        if order:
            return ActionExecutionResult(
                action_taken=LimitOrderActions.BUY_YES_LIMIT if contract_side == ContractSide.YES 
                            else LimitOrderActions.BUY_NO_LIMIT,
                order_placed=True,
                order_cancelled=cancelled_count > 0,
                order_amended=False,
                order_id=order.order_id
            )
        else:
            return ActionExecutionResult(
                action_taken=LimitOrderActions.BUY_YES_LIMIT if contract_side == ContractSide.YES 
                            else LimitOrderActions.BUY_NO_LIMIT,
                order_placed=False,
                order_cancelled=cancelled_count > 0,
                order_amended=False,
                order_id=None,
                error_message="Failed to place buy order"
            )
    
    async def _execute_sell_action(
        self,
        ticker: str,
        contract_side: ContractSide,
        orderbook: OrderbookState
    ) -> ActionExecutionResult:
        """
        Execute a sell action - place sell order, cancelling conflicting buys.
        """
        # Cancel conflicting orders (any buy orders)
        open_orders = self.order_manager.get_open_orders(ticker)
        conflicting_orders = [
            order for order in open_orders 
            if order.side == OrderSide.BUY or order.contract_side != contract_side
        ]
        
        cancelled_count = 0
        for order in conflicting_orders:
            success = await self.order_manager.cancel_order(order.order_id)
            if success:
                cancelled_count += 1
        
        # Check if we already have a compatible sell order
        existing_sell_orders = [
            order for order in open_orders
            if order.side == OrderSide.SELL and order.contract_side == contract_side
        ]
        
        if existing_sell_orders:
            # Already have a sell order - just amend if needed
            existing_order = existing_sell_orders[0]
            optimal_price = self.order_manager._calculate_limit_price(
                OrderSide.SELL, contract_side, orderbook, self.pricing_strategy
            )
            
            if abs(optimal_price - existing_order.limit_price) > 1:
                success = await self.order_manager.amend_order(
                    existing_order.order_id, optimal_price, orderbook
                )
                return ActionExecutionResult(
                    action_taken=LimitOrderActions.SELL_YES_LIMIT if contract_side == ContractSide.YES 
                                else LimitOrderActions.SELL_NO_LIMIT,
                    order_placed=False,
                    order_cancelled=cancelled_count > 0,
                    order_amended=success,
                    order_id=existing_order.order_id
                )
            else:
                # Order is fine as-is
                return ActionExecutionResult(
                    action_taken=LimitOrderActions.SELL_YES_LIMIT if contract_side == ContractSide.YES 
                                else LimitOrderActions.SELL_NO_LIMIT,
                    order_placed=False,
                    order_cancelled=cancelled_count > 0,
                    order_amended=False,
                    order_id=existing_order.order_id
                )
        
        # Place new sell order
        order = await self.order_manager.place_order(
            ticker=ticker,
            side=OrderSide.SELL,
            contract_side=contract_side,
            quantity=self.contract_size,
            orderbook=orderbook,
            pricing_strategy=self.pricing_strategy
        )
        
        if order:
            return ActionExecutionResult(
                action_taken=LimitOrderActions.SELL_YES_LIMIT if contract_side == ContractSide.YES 
                            else LimitOrderActions.SELL_NO_LIMIT,
                order_placed=True,
                order_cancelled=cancelled_count > 0,
                order_amended=False,
                order_id=order.order_id
            )
        else:
            return ActionExecutionResult(
                action_taken=LimitOrderActions.SELL_YES_LIMIT if contract_side == ContractSide.YES 
                            else LimitOrderActions.SELL_NO_LIMIT,
                order_placed=False,
                order_cancelled=cancelled_count > 0,
                order_amended=False,
                order_id=None,
                error_message="Failed to place sell order"
            )
    
    def get_action_info(self, action: int) -> Dict[str, Any]:
        """
        Get information about a specific action.
        
        Args:
            action: Integer action from 0-4
            
        Returns:
            Dictionary with action metadata
        """
        if not (0 <= action <= 4):
            raise ValueError(f"Action must be in range [0, 4], got {action}")
        
        action_enum = LimitOrderActions(action)
        
        return {
            'action_id': action,
            'action_name': action_enum.name,
            'description': self.action_descriptions[action_enum],
            'is_trade': action != LimitOrderActions.HOLD,
            'contract_size': self.contract_size if action != LimitOrderActions.HOLD else 0,
            'order_type': 'limit' if action != LimitOrderActions.HOLD else None,
            'requires_order_management': True
        }
    
    def get_all_actions_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all available actions.
        
        Returns:
            Dictionary mapping action IDs to their info
        """
        return {i: self.get_action_info(i) for i in range(5)}
    
    def validate_action(
        self,
        action: int,
        ticker: str,
        orderbook: OrderbookState,
        current_position: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Validate if an action can be executed given current state.
        
        Args:
            action: Integer action to validate
            ticker: Market ticker
            orderbook: Current orderbook state
            current_position: Current position state (optional)
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not (0 <= action <= 4):
            return False, f"Action {action} not in valid range [0, 4]"
        
        # HOLD is always valid
        if action == LimitOrderActions.HOLD:
            return True, "HOLD action is always valid"
        
        # Check orderbook has valid prices
        if not orderbook.yes_bids or not orderbook.yes_asks:
            return False, f"Orderbook for {ticker} has no valid prices"
        
        # Check for reasonable spread (not too wide)
        best_bid = max(orderbook.yes_bids.keys())
        best_ask = min(orderbook.yes_asks.keys()) 
        spread = best_ask - best_bid
        
        if spread > 50:  # More than 50¢ spread
            return False, f"Spread too wide ({spread}¢) for safe order placement"
        
        # For now, all other checks pass
        # Future enhancements could check:
        # - Position limits
        # - Risk management constraints  
        # - Market liquidity depth
        # - Account balance for new positions
        
        return True, "Action is valid"
    
    async def cancel_all_orders(self, ticker: str) -> int:
        """
        Cancel all orders for a specific ticker.
        
        Args:
            ticker: Market ticker
            
        Returns:
            Number of orders cancelled
        """
        return await self.order_manager.cancel_all_orders(ticker)
    
    def get_pricing_strategy(self) -> str:
        """Get the current pricing strategy."""
        return self.pricing_strategy
    
    def set_pricing_strategy(self, strategy: str) -> None:
        """
        Set the pricing strategy for future orders.
        
        Args:
            strategy: Pricing strategy ("aggressive", "passive", "mid")
        """
        valid_strategies = ["aggressive", "passive", "mid"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid pricing strategy: {strategy}. Must be one of {valid_strategies}")
        
        self.pricing_strategy = strategy
        logger.info(f"Pricing strategy updated to: {strategy}")
    
    def get_contract_size(self) -> int:
        """Get the fixed contract size."""
        return self.contract_size
    
    def get_action_mask(
        self,
        ticker: str,
        orderbook: OrderbookState,
        current_position: Optional[Dict[str, Any]] = None,
        cash_balance: float = 0.0
    ) -> np.ndarray:
        """
        Get action mask indicating which actions are valid.
        
        Returns a boolean array where True indicates the action is valid.
        This can be used by RL agents to avoid taking invalid actions.
        
        Args:
            ticker: Market ticker
            orderbook: Current orderbook state
            current_position: Current position state (optional)
            cash_balance: Available cash balance
            
        Returns:
            Boolean array of shape (5,) indicating valid actions
        """
        mask = np.ones(5, dtype=bool)  # Start with all actions valid
        
        for action_id in range(5):
            is_valid, _ = self.validate_action(
                action_id, ticker, orderbook, current_position
            )
            mask[action_id] = is_valid
            
            # Additional check for cash availability
            if action_id in [1, 2, 3, 4]:  # Trading actions
                if not self._check_cash_availability(cash_balance, orderbook):
                    mask[action_id] = False
        
        return mask
    
    def _check_cash_availability(self, cash_balance: float, orderbook: OrderbookState) -> bool:
        """
        Check if there's enough cash for a potential trade.
        
        Args:
            cash_balance: Available cash
            orderbook: Current orderbook state
            
        Returns:
            True if cash is sufficient for a trade
        """
        if not orderbook.yes_bids or not orderbook.yes_asks:
            return False
            
        # Estimate worst-case cost (buying at highest ask)
        max_ask = max(max(orderbook.yes_asks.keys()), max(orderbook.no_asks.keys()) if orderbook.no_asks else 0)
        estimated_cost = self.contract_size * max_ask / 100.0
        
        return cash_balance >= estimated_cost
    
    def suggest_best_action(
        self,
        ticker: str,
        orderbook: OrderbookState,
        target_position: Optional[int] = None,
        current_position: int = 0
    ) -> Tuple[int, str]:
        """
        Suggest the best action to achieve a target position.
        
        This is a utility function for testing and debugging, not for
        production RL use (agents should learn their own strategies).
        
        Args:
            ticker: Market ticker
            orderbook: Current orderbook state
            target_position: Desired position (positive=YES, negative=NO, 0=flat)
            current_position: Current position
            
        Returns:
            Tuple of (action_id, reasoning)
        """
        if target_position is None:
            return 0, "No target specified, suggest HOLD"
        
        position_diff = target_position - current_position
        
        if abs(position_diff) < self.contract_size:
            return 0, f"Position difference ({position_diff}) less than contract size"
        
        if position_diff > 0:  # Need more YES position
            if target_position > 0:
                return 1, f"Buy YES to reach target position {target_position}"
            else:
                return 4, f"Sell NO to reduce negative position toward {target_position}"
        
        else:  # Need more NO position or less YES
            if target_position < 0:
                return 3, f"Buy NO to reach target position {target_position}"
            else:
                return 2, f"Sell YES to reduce positive position toward {target_position}"
    
    def get_action_space_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the action space.
        
        Returns:
            Dictionary with action space metadata and documentation
        """
        return {
            "action_space_type": "LimitOrderActionSpace",
            "action_count": 5,
            "contract_size": self.contract_size,
            "pricing_strategy": self.pricing_strategy,
            "actions": self.get_all_actions_info(),
            "constraints": {
                "order_management": "One order per market rule",
                "order_type": "Limit orders only (Kalshi constraint)",
                "automatic_cancellation": "Conflicting orders automatically cancelled",
                "price_amendment": "Orders amended when prices move >1 cent"
            },
            "features": {
                "action_masking": True,
                "action_validation": True,
                "order_state_tracking": True,
                "automatic_pricing": True,
                "position_awareness": True
            },
            "kalshi_compliance": {
                "limit_orders_only": True,
                "position_convention": "+YES/-NO contracts",
                "pricing_format": "Integer cents (1-99)",
                "websocket_integration": "Fill tracking ready"
            }
        }