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

from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field
from enum import IntEnum
import logging

import numpy as np
from gymnasium import spaces

from ..data.orderbook_state import OrderbookState

# DEPRECATED: /trading/ module removed, training system is broken
# These imports will fail - training requires reimplementation
try:
    from ..trading.order_manager import OrderManager, OrderSide, ContractSide
except ImportError:
    OrderManager = None
    OrderSide = None
    ContractSide = None

logger = logging.getLogger("kalshiflow_rl.environments.limit_order_action_space")


@dataclass
class PositionConfig:
    """Central configuration for position sizing.
    
    EVOLUTION STRATEGY:
    - Phase 1: sizes=[20] -> 5 actions (1 HOLD + 4 trades)
    - Phase 2: sizes=[10, 50] -> 9 actions (1 HOLD + 8 trades)
    - Phase 3: sizes=[5, 10, 20, 50, 100] -> 21 actions (full granularity)
    """
    # START WITH SINGLE SIZE FOR SIMPLICITY (5 total actions)
    sizes: List[int] = field(default_factory=lambda: [5])  # Phase 1: Single size
    # sizes: List[int] = field(default_factory=lambda: [10, 50])  # Phase 2: Two sizes
    # sizes: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])  # Phase 3: Full
    
    max_position_per_market: int = 500
    max_position_value: int = 50000  # $500 in cents
    max_portfolio_concentration: float = 0.20  # 20% max
    min_cash_buffer: int = 5000  # $50 minimum reserve


class PositionSizeValidator:
    """Validator for position sizing constraints."""
    
    def __init__(self, config: PositionConfig):
        self.config = config
    
    def validate_action(self, action: int, cash: int, current_position: int, orderbook: Dict) -> bool:
        """Validate if action is executable given constraints."""
        if action == 0:
            return True  # HOLD always valid
            
        base_action, size_index = decode_action(action, len(self.config.sizes))
        size = self.config.sizes[size_index]
        
        # Validate size constraints
        price = self._get_execution_price(base_action, orderbook)
        position_value = size * price
        
        checks = [
            abs(current_position + size) <= self.config.max_position_per_market,
            position_value <= self.config.max_position_value,
            position_value <= cash * self.config.max_portfolio_concentration,
            cash - position_value >= self.config.min_cash_buffer
        ]
        
        return all(checks)
    
    def _get_execution_price(self, base_action: int, orderbook: Dict) -> int:
        """Get estimated execution price for action."""
        # Simplified price estimation - can be enhanced
        if not orderbook or not hasattr(orderbook, 'yes_bids') or not hasattr(orderbook, 'yes_asks'):
            return 50  # Default mid price
        
        if base_action in [1, 3]:  # BUY actions
            return min(orderbook.yes_asks.keys()) if orderbook.yes_asks else 50
        else:  # SELL actions  
            return max(orderbook.yes_bids.keys()) if orderbook.yes_bids else 50


def decode_action(action: int, num_sizes: int = 1) -> Tuple[int, int]:
    """Decode single action into base action and size index.
    
    Flexible decoding that adapts to the number of position sizes configured.
    
    Args:
        action: The raw action from the model (0 to N)
        num_sizes: Number of position sizes available (from PositionConfig)
    
    Returns:
        base_action: 0=HOLD, 1=BUY_YES, 2=SELL_YES, 3=BUY_NO, 4=SELL_NO
        size_index: Index into the sizes array
    """
    if action == 0:
        return 0, 0  # HOLD, no size
    
    if num_sizes == 1:
        # Phase 1: Single size, actions 1-4 map directly to trading intents
        return action, 0  # Always use first (only) size
    else:
        # Phase 2+: Multiple sizes
        adjusted = action - 1  # Now 0-based for trading actions
        base_action = (adjusted // num_sizes) + 1  # Which trading intent
        size_index = adjusted % num_sizes  # Which size
        return base_action, size_index


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
        position_config: Optional[PositionConfig] = None,
        pricing_strategy: str = "aggressive"
    ):
        """
        Initialize the limit order action space.
        
        Args:
            order_manager: OrderManager instance for execution
            position_config: Position sizing configuration (defaults to PositionConfig())
            pricing_strategy: Default pricing strategy ("aggressive", "passive", "mid")
        """
        self.order_manager = order_manager
        self.position_config = position_config or PositionConfig()
        self.position_sizes = self.position_config.sizes
        self.position_validator = PositionSizeValidator(self.position_config)
        self.pricing_strategy = pricing_strategy
        
        self.action_descriptions = {
            LimitOrderActions.HOLD: "No action - maintain current orders",
            LimitOrderActions.BUY_YES_LIMIT: "Place/maintain buy order for YES contracts",
            LimitOrderActions.SELL_YES_LIMIT: "Place/maintain sell order for YES contracts",
            LimitOrderActions.BUY_NO_LIMIT: "Place/maintain buy order for NO contracts", 
            LimitOrderActions.SELL_NO_LIMIT: "Place/maintain sell order for NO contracts"
        }
        
        logger.info(f"LimitOrderActionSpace initialized with position sizes: {self.position_sizes}")
    
    def get_gym_space(self) -> spaces.Discrete:
        """
        Get the Gymnasium space representation.
        
        Dynamically calculates space size based on configured position sizes:
        - 1 size: 5 actions (1 HOLD + 4 trading directions)
        - 2 sizes: 9 actions (1 HOLD + 8 trading combinations)
        - 5 sizes: 21 actions (1 HOLD + 20 trading combinations)
        
        Returns:
            spaces.Discrete(n): Where n = 1 + (4 * num_sizes)
        """
        num_actions = 1 + (4 * len(self.position_sizes))  # HOLD + (4 directions × sizes)
        return spaces.Discrete(num_actions)
    
    def execute_action_sync(
        self,
        action: int,
        ticker: str,
        orderbook: OrderbookState
    ) -> ActionExecutionResult:
        """
        Execute a trading action synchronously (for training environments).
        
        This is a synchronous wrapper around execute_action for use in
        training environments where async operations are not needed.
        
        Args:
            action: Integer action from 0-4
            ticker: Market ticker for the order
            orderbook: Current orderbook state for pricing
            
        Returns:
            ActionExecutionResult with details of what was executed
        """
        import asyncio
        
        # For training environments, execute actions properly regardless of async context
        try:
            # Check if this is a SimulatedOrderManager (training) vs real OrderManager
            if hasattr(self.order_manager, '__class__') and 'SimulatedOrderManager' in str(self.order_manager.__class__):
                # For SimulatedOrderManager, we can safely execute synchronously
                # even in async context since it's just pure Python simulation
                return self._execute_action_sync_simulated(action, ticker, orderbook)
            else:
                # For real OrderManager, handle async properly
                try:
                    current_loop = asyncio.get_running_loop()
                    # Real OrderManager in async context - might not be safe
                    logger.debug(f"Real OrderManager in async context, might skip execution for action {action}")
                    return ActionExecutionResult(
                        action_taken=LimitOrderActions(decode_action(action, len(self.position_sizes))[0]) if 0 <= action <= 4 else LimitOrderActions.HOLD,
                        order_placed=False,
                        order_cancelled=False,
                        order_amended=False,
                        order_id=None,
                        error_message="Skipped real order execution in async context"
                    )
                except RuntimeError:
                    # No running loop, create new one for real OrderManager
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(self.execute_action(action, ticker, orderbook))
                    finally:
                        loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous action execution: {e}")
            return ActionExecutionResult(
                action_taken=LimitOrderActions(action) if 0 <= action <= 4 else LimitOrderActions.HOLD,
                order_placed=False,
                order_cancelled=False,
                order_amended=False,
                order_id=None,
                error_message=str(e)
            )
    
    def _execute_action_sync_simulated(
        self,
        action: int,
        ticker: str,
        orderbook: OrderbookState
    ) -> ActionExecutionResult:
        """
        Execute action synchronously for SimulatedOrderManager.
        
        This bypasses async/await and directly calls the simulated order operations
        since they don't actually need to be async. Now supports 21 actions with position sizing.
        """
        try:
            if not (0 <= action <= 4):
                return ActionExecutionResult(
                    action_taken=LimitOrderActions.HOLD,
                    order_placed=False,
                    order_cancelled=False,
                    order_amended=False,
                    order_id=None,
                    error_message=f"Invalid action: {action}. Must be 0-20."
                )
            
            # Decode action into base action and size
            base_action, size_index = decode_action(action, len(self.position_sizes))
            
            # Handle HOLD action
            if base_action == 0:
                return self._execute_hold_action_sync(ticker, orderbook)
            
            # Get position size for trading actions
            position_size = self.position_sizes[size_index]
            
            # Handle trading actions with variable position size
            if base_action == 1:  # BUY_YES
                return self._execute_buy_action_sync(ticker, ContractSide.YES, orderbook, position_size)
            
            elif base_action == 2:  # SELL_YES
                return self._execute_sell_action_sync(ticker, ContractSide.YES, orderbook, position_size)
            
            elif base_action == 3:  # BUY_NO
                return self._execute_buy_action_sync(ticker, ContractSide.NO, orderbook, position_size)
            
            elif base_action == 4:  # SELL_NO
                return self._execute_sell_action_sync(ticker, ContractSide.NO, orderbook, position_size)
            
            else:
                return ActionExecutionResult(
                    action_taken=LimitOrderActions.HOLD,
                    order_placed=False,
                    order_cancelled=False,
                    order_amended=False,
                    order_id=None,
                    error_message=f"Unhandled base action: {base_action}"
                )
                
        except Exception as e:
            logger.error(f"Error executing simulated action {action}: {e}")
            return ActionExecutionResult(
                action_taken=LimitOrderActions.HOLD,
                order_placed=False,
                order_cancelled=False,
                order_amended=False,
                order_id=None,
                error_message=str(e)
            )

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
        Now supports 21 actions with variable position sizing.
        
        Args:
            action: Integer action from 0-20
            ticker: Market ticker for the order
            orderbook: Current orderbook state for pricing
            
        Returns:
            ActionExecutionResult with details of what was executed
        """
        try:
            if not (0 <= action <= 4):
                return ActionExecutionResult(
                    action_taken=LimitOrderActions.HOLD,
                    order_placed=False,
                    order_cancelled=False,
                    order_amended=False,
                    order_id=None,
                    error_message=f"Invalid action: {action}. Must be 0-20."
                )
            
            # Decode action into base action and size
            base_action, size_index = decode_action(action, len(self.position_sizes))
            
            # Handle HOLD action
            if base_action == 0:
                return await self._execute_hold_action(ticker, orderbook)
            
            # Get position size for trading actions
            position_size = self.position_sizes[size_index]
            
            # Handle trading actions with variable position size
            if base_action == 1:  # BUY_YES
                return await self._execute_buy_action(ticker, ContractSide.YES, orderbook, position_size)
            
            elif base_action == 2:  # SELL_YES
                return await self._execute_sell_action(ticker, ContractSide.YES, orderbook, position_size)
            
            elif base_action == 3:  # BUY_NO
                return await self._execute_buy_action(ticker, ContractSide.NO, orderbook, position_size)
            
            elif base_action == 4:  # SELL_NO
                return await self._execute_sell_action(ticker, ContractSide.NO, orderbook, position_size)
            
            else:
                return ActionExecutionResult(
                    action_taken=LimitOrderActions.HOLD,
                    order_placed=False,
                    order_cancelled=False,
                    order_amended=False,
                    order_id=None,
                    error_message=f"Unhandled base action: {base_action}"
                )
                
        except Exception as e:
            logger.error(f"Error executing action {action}: {e}")
            return ActionExecutionResult(
                action_taken=LimitOrderActions.HOLD,
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
        orderbook: OrderbookState,
        position_size: Optional[int] = None
    ) -> ActionExecutionResult:
        """
        Execute a buy action - place buy order, cancelling conflicting sells.
        
        This method implements the "one order per market" rule by cancelling
        any conflicting sell orders before placing the new buy order.
        
        Args:
            position_size: Number of contracts to trade (if None, uses first position size from config)
        """
        # Use provided position_size or fallback to contract_size for backward compatibility
        quantity = position_size if position_size is not None else self.position_sizes[0]
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
            quantity=quantity,
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
        orderbook: OrderbookState,
        position_size: Optional[int] = None
    ) -> ActionExecutionResult:
        """
        Execute a sell action - place sell order, cancelling conflicting buys.
        
        Args:
            position_size: Number of contracts to trade (if None, uses first position size from config)
        """
        # Use provided position_size or fallback to contract_size for backward compatibility
        quantity = position_size if position_size is not None else self.position_sizes[0]
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
            quantity=quantity,
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
            'contract_size': self.position_sizes[0] if action != LimitOrderActions.HOLD else 0,
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
        """Get the default contract size.
        
        Deprecated: Use get_position_sizes() for multi-size support.
        Returns first size for backward compatibility.
        """
        if len(self.position_sizes) > 1:
            logger.warning("get_contract_size() called with multiple sizes configured. "
                          "Use get_position_sizes() or get_position_size_for_action().")
        return self.position_sizes[0]
    
    def get_position_sizes(self) -> List[int]:
        """Get all configured position sizes."""
        return self.position_sizes
    
    def get_position_size_for_action(self, action: int) -> int:
        """Get the position size for a specific action."""
        if action == 0:  # HOLD
            return 0
        _, size_index = decode_action(action, len(self.position_sizes))
        return self.position_sizes[size_index]
    
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
        estimated_cost = self.position_sizes[0] * max_ask / 100.0
        
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
        
        if abs(position_diff) < self.position_sizes[0]:
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
            "contract_size": self.position_sizes[0],
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
    
    def _execute_hold_action_sync(
        self,
        ticker: str,
        orderbook: OrderbookState
    ) -> ActionExecutionResult:
        """Synchronous version of _execute_hold_action for SimulatedOrderManager."""
        # For now, just return no action taken (can be enhanced later)
        return ActionExecutionResult(
            action_taken=LimitOrderActions.HOLD,
            order_placed=False,
            order_cancelled=False,
            order_amended=False,
            order_id=None
        )
    
    def _execute_buy_action_sync(
        self,
        ticker: str,
        contract_side: ContractSide,
        orderbook: OrderbookState,
        position_size: Optional[int] = None
    ) -> ActionExecutionResult:
        """Synchronous version of _execute_buy_action for SimulatedOrderManager."""
        try:
            # Use provided position_size or fallback to contract_size for backward compatibility
            quantity = position_size if position_size is not None else self.position_sizes[0]
            # Calculate limit price using SimulatedOrderManager's method
            limit_price = self.order_manager._calculate_limit_price(
                OrderSide.BUY, contract_side, orderbook, self.pricing_strategy
            )
            
            # Create order info directly (bypassing async place_order)
            # DEPRECATED: This will fail - training system is broken
            try:
                from ..trading.order_manager import OrderInfo, OrderStatus
            except ImportError:
                raise RuntimeError("Training system is broken - /trading/ module was removed")
            import time

            order = OrderInfo(
                order_id=self.order_manager._generate_order_id(),
                ticker=ticker,
                side=OrderSide.BUY,
                contract_side=contract_side,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                placed_at=time.time()
            )
            
            # Check if order can fill immediately
            can_fill = self.order_manager._can_fill_immediately(order, orderbook)
            
            if can_fill:
                # Execute immediate fill
                fill_price = self.order_manager._get_fill_price(order, orderbook)
                self.order_manager._process_fill(order, fill_price)
                logger.debug(f"Sync buy order filled immediately: {order.order_id} at {fill_price}¢")
            else:
                # Add to open orders
                self.order_manager.open_orders[order.order_id] = order
                logger.debug(f"Sync buy order placed: {order.order_id} at {limit_price}¢")
            
            return ActionExecutionResult(
                action_taken=LimitOrderActions.BUY_YES_LIMIT if contract_side == ContractSide.YES 
                            else LimitOrderActions.BUY_NO_LIMIT,
                order_placed=True,
                order_cancelled=False,
                order_amended=False,
                order_id=order.order_id
            )
            
        except Exception as e:
            logger.error(f"Error in sync buy action: {e}")
            return ActionExecutionResult(
                action_taken=LimitOrderActions.BUY_YES_LIMIT if contract_side == ContractSide.YES 
                            else LimitOrderActions.BUY_NO_LIMIT,
                order_placed=False,
                order_cancelled=False,
                order_amended=False,
                order_id=None,
                error_message=str(e)
            )
    
    def _execute_sell_action_sync(
        self,
        ticker: str,
        contract_side: ContractSide,
        orderbook: OrderbookState,
        position_size: Optional[int] = None
    ) -> ActionExecutionResult:
        """Synchronous version of _execute_sell_action for SimulatedOrderManager."""
        try:
            # Use provided position_size or fallback to contract_size for backward compatibility
            quantity = position_size if position_size is not None else self.position_sizes[0]
            # Calculate limit price using SimulatedOrderManager's method
            limit_price = self.order_manager._calculate_limit_price(
                OrderSide.SELL, contract_side, orderbook, self.pricing_strategy
            )
            
            # Create order info directly (bypassing async place_order)
            # DEPRECATED: This will fail - training system is broken
            try:
                from ..trading.order_manager import OrderInfo, OrderStatus
            except ImportError:
                raise RuntimeError("Training system is broken - /trading/ module was removed")
            import time

            order = OrderInfo(
                order_id=self.order_manager._generate_order_id(),
                ticker=ticker,
                side=OrderSide.SELL,
                contract_side=contract_side,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                placed_at=time.time()
            )
            
            # Check if order can fill immediately
            can_fill = self.order_manager._can_fill_immediately(order, orderbook)
            
            if can_fill:
                # Execute immediate fill
                fill_price = self.order_manager._get_fill_price(order, orderbook)
                self.order_manager._process_fill(order, fill_price)
                logger.debug(f"Sync sell order filled immediately: {order.order_id} at {fill_price}¢")
            else:
                # Add to open orders
                self.order_manager.open_orders[order.order_id] = order
                logger.debug(f"Sync sell order placed: {order.order_id} at {limit_price}¢")
            
            return ActionExecutionResult(
                action_taken=LimitOrderActions.SELL_YES_LIMIT if contract_side == ContractSide.YES 
                            else LimitOrderActions.SELL_NO_LIMIT,
                order_placed=True,
                order_cancelled=False,
                order_amended=False,
                order_id=order.order_id
            )
            
        except Exception as e:
            logger.error(f"Error in sync sell action: {e}")
            return ActionExecutionResult(
                action_taken=LimitOrderActions.SELL_YES_LIMIT if contract_side == ContractSide.YES 
                            else LimitOrderActions.SELL_NO_LIMIT,
                order_placed=False,
                order_cancelled=False,
                order_amended=False,
                order_id=None,
                error_message=str(e)
            )