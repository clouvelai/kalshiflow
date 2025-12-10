"""
Primitive action space for the Kalshi Flow RL Trading Subsystem.

This module implements a simplified 5-action space that enables truly stateless operation
by only using immediate market orders. The agent learns to discover complex trading strategies
through these primitive building blocks.

Key Design Principles:
- NO WAIT actions (no limit orders) - only immediate market orders
- Truly stateless - no pending orders to track
- Simple enough for V1.0 deployment
- Fixed 10 contract sizing
- Compatible with single-market observation space
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from gymnasium import spaces


class PrimitiveActions(IntEnum):
    """
    Enumeration of the 5 primitive actions available to the agent.
    
    These actions are stateless and execute immediately, making the system
    much simpler to deploy and manage compared to stateful limit order tracking.
    """
    HOLD = 0           # No action - maintain current positions
    BUY_YES_NOW = 1    # Market buy YES contracts immediately  
    SELL_YES_NOW = 2   # Market sell YES contracts immediately
    BUY_NO_NOW = 3     # Market buy NO contracts immediately
    SELL_NO_NOW = 4    # Market sell NO contracts immediately


@dataclass
class DecodedAction:
    """Represents a decoded action ready for execution."""
    ticker: str
    action: str        # 'buy' or 'sell'
    side: str         # 'yes' or 'no'
    count: int        # Always 10 for fixed sizing
    type: str         # Always 'market' for immediate execution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API calls."""
        return {
            'ticker': self.ticker,
            'action': self.action,
            'side': self.side,
            'count': self.count,
            'type': self.type
        }


class PrimitiveActionSpace:
    """
    Primitive action space for single-market Kalshi trading.
    
    This class implements a simplified 5-action space designed for truly stateless
    operation. All actions are immediate market orders, removing the complexity
    of tracking pending limit orders across episodes.
    
    Features:
    - 5 discrete actions: HOLD + 4 immediate market orders
    - Fixed 10 contract size for all trades
    - Stateless operation - no order tracking required
    - Single market focus for strategy discovery
    - Compatible with Kalshi API order format
    
    Usage:
        action_space = PrimitiveActionSpace()
        gym_space = action_space.get_gym_space()  # spaces.Discrete(5)
        order = action_space.decode_action(2, "MARKET-123")  # Sell YES
    """
    
    FIXED_CONTRACT_SIZE = 10  # Fixed number of contracts for all trades
    
    def __init__(self):
        """Initialize the primitive action space."""
        self.action_descriptions = {
            PrimitiveActions.HOLD: "No action - maintain current positions",
            PrimitiveActions.BUY_YES_NOW: "Market buy YES contracts immediately",
            PrimitiveActions.SELL_YES_NOW: "Market sell YES contracts immediately", 
            PrimitiveActions.BUY_NO_NOW: "Market buy NO contracts immediately",
            PrimitiveActions.SELL_NO_NOW: "Market sell NO contracts immediately"
        }
    
    def get_gym_space(self) -> spaces.Discrete:
        """
        Get the Gymnasium space representation.
        
        Returns:
            spaces.Discrete(5): Discrete action space with 5 actions
        """
        return spaces.Discrete(5)
    
    def decode_action(
        self, 
        action: int, 
        market_ticker: str,
        validate: bool = True
    ) -> Optional[DecodedAction]:
        """
        Decode an integer action into a Kalshi order format.
        
        Args:
            action: Integer action from 0-4
            market_ticker: Market ticker for the order (e.g., "MARKET-123")
            validate: Whether to validate the action (for future use)
            
        Returns:
            DecodedAction object if action != HOLD, None for HOLD
            
        Raises:
            ValueError: If action is not in valid range
        """
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer, got {type(action)}")
        
        if not (0 <= action <= 4):
            raise ValueError(f"Action must be in range [0, 4], got {action}")
        
        action_enum = PrimitiveActions(action)
        
        # HOLD action returns None (no order to execute)
        if action_enum == PrimitiveActions.HOLD:
            return None
        
        # Decode action to order parameters
        if action_enum == PrimitiveActions.BUY_YES_NOW:
            return DecodedAction(
                ticker=market_ticker,
                action='buy',
                side='yes',
                count=self.FIXED_CONTRACT_SIZE,
                type='market'
            )
        elif action_enum == PrimitiveActions.SELL_YES_NOW:
            return DecodedAction(
                ticker=market_ticker,
                action='sell',
                side='yes',
                count=self.FIXED_CONTRACT_SIZE,
                type='market'
            )
        elif action_enum == PrimitiveActions.BUY_NO_NOW:
            return DecodedAction(
                ticker=market_ticker,
                action='buy',
                side='no',
                count=self.FIXED_CONTRACT_SIZE,
                type='market'
            )
        elif action_enum == PrimitiveActions.SELL_NO_NOW:
            return DecodedAction(
                ticker=market_ticker,
                action='sell',
                side='no',
                count=self.FIXED_CONTRACT_SIZE,
                type='market'
            )
        
        # Should never reach here due to enum validation
        raise ValueError(f"Unexpected action: {action_enum}")
    
    def validate_action(
        self, 
        action: int, 
        current_positions: Optional[Dict[str, Any]] = None,
        market_state: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str]:
        """
        Validate if an action can be executed given current state.
        
        This is a placeholder for future position limit and liquidity checks.
        Currently returns True for all actions since we use fixed small sizes.
        
        Args:
            action: Integer action to validate
            current_positions: Current position state (future use)
            market_state: Current market liquidity state (future use)
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not (0 <= action <= 4):
            return False, f"Action {action} not in valid range [0, 4]"
        
        # HOLD is always valid
        if action == PrimitiveActions.HOLD:
            return True, "HOLD action is always valid"
        
        # For now, all market orders are considered valid due to small fixed size
        # Future improvements could check:
        # - Position limits
        # - Market liquidity
        # - Risk management constraints
        return True, "Market order with fixed small size is valid"
    
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
        
        action_enum = PrimitiveActions(action)
        
        return {
            'action_id': action,
            'action_name': action_enum.name,
            'description': self.action_descriptions[action_enum],
            'is_trade': action != PrimitiveActions.HOLD,
            'contract_size': self.FIXED_CONTRACT_SIZE if action != PrimitiveActions.HOLD else 0,
            'order_type': 'market' if action != PrimitiveActions.HOLD else None
        }
    
    def get_all_actions_info(self) -> Dict[int, Dict[str, Any]]:
        """
        Get information about all available actions.
        
        Returns:
            Dictionary mapping action IDs to their info
        """
        return {i: self.get_action_info(i) for i in range(5)}


# Global instance for easy import
primitive_action_space = PrimitiveActionSpace()