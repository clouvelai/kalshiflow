"""
Action space definition for Kalshi RL Trading Subsystem.

Provides multi-market action space with position sizing, risk management,
and action encoding/decoding for reinforcement learning agents.

CRITICAL ARCHITECTURAL REQUIREMENT:
- Actions must support variable number of markets (1 to N)
- Position limits enforced per market and globally
- Risk management prevents excessive correlation exposure
- Actions must be interpretable and debuggable
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
import gymnasium as gym
from gymnasium import spaces

logger = logging.getLogger("kalshiflow_rl.action_space")

# Action space constants
MAX_MARKETS = 10  # Maximum number of markets supported
MAX_POSITION_SIZE = 1000  # Maximum position size per market
MAX_TOTAL_PORTFOLIO_VALUE = 10000  # Maximum total portfolio exposure


class ActionType(Enum):
    """Types of actions available in the trading environment."""
    HOLD = 0
    BUY_YES = 1
    SELL_YES = 2
    BUY_NO = 3
    SELL_NO = 4
    CLOSE_POSITION = 5


class PositionSizing(Enum):
    """Position sizing strategies."""
    FIXED_SMALL = 0    # 10 contracts
    FIXED_MEDIUM = 1   # 50 contracts
    FIXED_LARGE = 2    # 100 contracts
    PROPORTIONAL = 3   # Based on confidence/signal strength
    RISK_ADJUSTED = 4  # Based on volatility and correlation


class ActionConfig:
    """Configuration for action space."""
    
    def __init__(self):
        self.max_markets = MAX_MARKETS
        self.max_position_size = MAX_POSITION_SIZE
        self.max_total_exposure = MAX_TOTAL_PORTFOLIO_VALUE
        
        # Action constraints
        self.allow_short_selling = True  # Allow negative positions
        self.enforce_position_limits = True
        self.enforce_portfolio_limits = True
        self.min_order_size = 1  # Minimum order size
        self.max_order_size = 200  # Maximum single order size
        
        # Risk management
        self.max_correlation_exposure = 0.8  # Max exposure to correlated markets
        self.daily_loss_limit = 1000  # Daily loss limit in dollars
        self.position_concentration_limit = 0.5  # Max % of portfolio in single position
        
        # Market making / liquidity parameters
        self.enable_market_making = False  # Not implemented in MVP
        self.enable_arbitrage = False  # Not implemented in MVP
        
        # Action encoding
        self.use_discrete_actions = True  # Discrete vs continuous action space
        self.discrete_action_count = len(ActionType) * len(PositionSizing)  # Total discrete actions per market
        self.market_selection_method = "all_markets"  # "single_market", "subset", "all_markets"


def create_action_space(
    market_count: int,
    config: Optional[ActionConfig] = None
) -> gym.Space:
    """
    Create Gymnasium action space for multi-market RL environment.
    
    Args:
        market_count: Number of markets to trade
        config: Action configuration
        
    Returns:
        Gymnasium action space (Discrete or Box)
    """
    if config is None:
        config = ActionConfig()
    
    if market_count > config.max_markets:
        raise ValueError(f"Market count {market_count} exceeds maximum {config.max_markets}")
    
    if config.use_discrete_actions:
        # Discrete action space: for each market, select (ActionType, PositionSizing)
        # Plus a global "number of markets to act on" dimension
        if config.market_selection_method == "all_markets":
            # Action for each market independently
            action_space = spaces.MultiDiscrete([config.discrete_action_count] * market_count)
        elif config.market_selection_method == "single_market":
            # Select one market + one action for that market
            action_space = spaces.MultiDiscrete([market_count, config.discrete_action_count])
        else:
            # Default: act on all markets
            action_space = spaces.MultiDiscrete([config.discrete_action_count] * market_count)
    else:
        # Continuous action space: for each market, output (action_type_probability, position_size)
        # Each market gets: [action_type_probs (6), position_size (1)] = 7 values
        continuous_dim = market_count * 7
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(continuous_dim,),
            dtype=np.float32
        )
    
    logger.info(f"Created action space for {market_count} markets: {action_space}")
    return action_space


def encode_action(
    market_actions: Dict[str, Dict[str, Any]],
    market_tickers: List[str],
    config: Optional[ActionConfig] = None
) -> np.ndarray:
    """
    Encode human-readable actions to numerical action vector.
    
    Args:
        market_actions: Dict mapping market_ticker -> action details
            Format: {market: {"action_type": ActionType, "position_sizing": PositionSizing, "quantity": int}}
        market_tickers: Ordered list of market tickers
        config: Action configuration
        
    Returns:
        Encoded action vector for RL agent
    """
    if config is None:
        config = ActionConfig()
    
    if config.use_discrete_actions:
        actions = []
        for ticker in market_tickers[:config.max_markets]:
            if ticker in market_actions:
                action_info = market_actions[ticker]
                action_type = action_info.get("action_type", ActionType.HOLD)
                position_sizing = action_info.get("position_sizing", PositionSizing.FIXED_SMALL)
                
                # Encode as single discrete value: action_type * num_position_types + position_sizing
                encoded_action = action_type.value * len(PositionSizing) + position_sizing.value
                actions.append(encoded_action)
            else:
                # Default to HOLD with small sizing
                actions.append(ActionType.HOLD.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value)
        
        return np.array(actions, dtype=np.int32)
    else:
        # Continuous encoding not implemented in MVP
        raise NotImplementedError("Continuous action space not implemented in MVP")


def decode_action(
    action_vector: np.ndarray,
    market_tickers: List[str],
    config: Optional[ActionConfig] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Decode numerical action vector to human-readable actions.
    
    Args:
        action_vector: Numerical action from RL agent
        market_tickers: Ordered list of market tickers
        config: Action configuration
        
    Returns:
        Dict mapping market_ticker -> action details
    """
    if config is None:
        config = ActionConfig()
    
    if config.use_discrete_actions:
        market_actions = {}
        
        for idx, ticker in enumerate(market_tickers):
            if idx >= len(action_vector) or idx >= config.max_markets:
                break
                
            encoded_action = int(action_vector[idx])
            
            # Decode: action_type = encoded // num_position_types, position_sizing = encoded % num_position_types
            num_position_types = len(PositionSizing)
            action_type_val = encoded_action // num_position_types
            position_sizing_val = encoded_action % num_position_types
            
            # Clamp to valid ranges
            action_type_val = max(0, min(action_type_val, len(ActionType) - 1))
            position_sizing_val = max(0, min(position_sizing_val, len(PositionSizing) - 1))
            
            action_type = ActionType(action_type_val)
            position_sizing = PositionSizing(position_sizing_val)
            
            # Convert position sizing to actual quantity
            quantity = _position_sizing_to_quantity(position_sizing, config)
            
            market_actions[ticker] = {
                "action_type": action_type,
                "position_sizing": position_sizing,
                "quantity": quantity,
                "side": _action_type_to_side(action_type),
                "price": None  # Will be determined by execution engine
            }
        
        return market_actions
    else:
        raise NotImplementedError("Continuous action space not implemented in MVP")


def validate_action(
    action_vector: np.ndarray,
    market_tickers: List[str],
    current_positions: Dict[str, Dict[str, float]],
    current_portfolio_value: float,
    config: Optional[ActionConfig] = None
) -> Tuple[bool, List[str]]:
    """
    Validate action against risk management constraints.
    
    Args:
        action_vector: Proposed action
        market_tickers: Market tickers
        current_positions: Current positions per market
        current_portfolio_value: Current portfolio value
        config: Action configuration
        
    Returns:
        (is_valid, list_of_violation_reasons)
    """
    if config is None:
        config = ActionConfig()
    
    violations = []
    
    try:
        # Decode action for validation
        market_actions = decode_action(action_vector, market_tickers, config)
        
        # Calculate total exposure after all actions
        total_position_value_after = 0.0
        total_position_count = 0
        
        for ticker, action_info in market_actions.items():
            action_type = action_info["action_type"]
            quantity = action_info["quantity"]
            current_pos = current_positions.get(ticker, {})
            
            # Skip validation for HOLD actions
            if action_type == ActionType.HOLD:
                continue
            
            # Check individual order size limits
            if quantity < config.min_order_size:
                violations.append(f"Order size {quantity} below minimum {config.min_order_size} for {ticker}")
            
            if quantity > config.max_order_size:
                violations.append(f"Order size {quantity} above maximum {config.max_order_size} for {ticker}")
            
            # Check position limits per market
            current_yes_pos = current_pos.get("position_yes", 0.0)
            current_no_pos = current_pos.get("position_no", 0.0)
            
            new_yes_pos, new_no_pos = _calculate_new_position(
                current_yes_pos, current_no_pos, action_type, quantity
            )
            
            if config.enforce_position_limits:
                if abs(new_yes_pos) > config.max_position_size:
                    violations.append(f"Yes position {new_yes_pos} exceeds limit {config.max_position_size} for {ticker}")
                
                if abs(new_no_pos) > config.max_position_size:
                    violations.append(f"No position {new_no_pos} exceeds limit {config.max_position_size} for {ticker}")
            
            # Estimate exposure for the new position sizes
            # This is the total value of positions AFTER the trade for this market
            estimated_position_value = (abs(new_yes_pos) + abs(new_no_pos)) * 0.50  # Convert to dollars
            total_position_value_after += estimated_position_value
            
            if abs(new_yes_pos) > 0.1 or abs(new_no_pos) > 0.1:
                total_position_count += 1
        
        # Also add existing positions from other markets
        for ticker in market_tickers:
            if ticker not in market_actions:
                current_pos = current_positions.get(ticker, {})
                existing_value = (abs(current_pos.get("position_yes", 0)) + 
                                abs(current_pos.get("position_no", 0))) * 0.50
                total_position_value_after += existing_value
        
        # Check portfolio-level constraints
        if config.enforce_portfolio_limits:
            # Check if total position value exceeds limit
            # This should be the total value locked up in positions
            if total_position_value_after > config.max_total_exposure:
                violations.append(f"Total position exposure {total_position_value_after:.2f} exceeds limit {config.max_total_exposure}")
            
            # Check concentration limit
            if total_position_count > 0:
                max_single_exposure = max([
                    (abs(current_positions.get(ticker, {}).get("position_yes", 0)) + 
                     abs(current_positions.get(ticker, {}).get("position_no", 0))) * 50
                    for ticker in market_tickers
                ])
                
                concentration_ratio = max_single_exposure / max(total_position_value_after, 1.0)
                if concentration_ratio > config.position_concentration_limit:
                    violations.append(f"Position concentration {concentration_ratio:.2f} exceeds limit {config.position_concentration_limit}")
        
        return len(violations) == 0, violations
        
    except Exception as e:
        violations.append(f"Action validation error: {str(e)}")
        return False, violations


def _position_sizing_to_quantity(position_sizing: PositionSizing, config: ActionConfig) -> int:
    """Convert position sizing enum to actual quantity."""
    if position_sizing == PositionSizing.FIXED_SMALL:
        return 10
    elif position_sizing == PositionSizing.FIXED_MEDIUM:
        return 50
    elif position_sizing == PositionSizing.FIXED_LARGE:
        return 100
    elif position_sizing == PositionSizing.PROPORTIONAL:
        return 75  # Default proportional size
    elif position_sizing == PositionSizing.RISK_ADJUSTED:
        return 25  # Conservative risk-adjusted size
    else:
        return config.min_order_size


def _action_type_to_side(action_type: ActionType) -> Optional[str]:
    """Convert action type to trading side."""
    if action_type in [ActionType.BUY_YES, ActionType.SELL_YES]:
        return "yes"
    elif action_type in [ActionType.BUY_NO, ActionType.SELL_NO]:
        return "no"
    else:
        return None


def _calculate_new_position(
    current_yes_pos: float,
    current_no_pos: float,
    action_type: ActionType,
    quantity: int
) -> Tuple[float, float]:
    """Calculate new position after applying action."""
    new_yes_pos = current_yes_pos
    new_no_pos = current_no_pos
    
    if action_type == ActionType.BUY_YES:
        new_yes_pos += quantity
    elif action_type == ActionType.SELL_YES:
        new_yes_pos -= quantity
    elif action_type == ActionType.BUY_NO:
        new_no_pos += quantity
    elif action_type == ActionType.SELL_NO:
        new_no_pos -= quantity
    elif action_type == ActionType.CLOSE_POSITION:
        new_yes_pos = 0.0
        new_no_pos = 0.0
    
    return new_yes_pos, new_no_pos


def create_random_action(
    market_tickers: List[str],
    config: Optional[ActionConfig] = None,
    action_probability: float = 0.1
) -> np.ndarray:
    """
    Create a random action for testing/exploration.
    
    Args:
        market_tickers: List of market tickers
        config: Action configuration
        action_probability: Probability of taking non-HOLD action per market
        
    Returns:
        Random action vector
    """
    if config is None:
        config = ActionConfig()
    
    np.random.seed()  # Ensure randomness
    
    if config.use_discrete_actions:
        actions = []
        
        for _ in market_tickers[:config.max_markets]:
            if np.random.random() < action_probability:
                # Random action
                action_type_val = np.random.randint(0, len(ActionType))
                position_sizing_val = np.random.randint(0, len(PositionSizing))
                encoded_action = action_type_val * len(PositionSizing) + position_sizing_val
                actions.append(encoded_action)
            else:
                # HOLD action
                actions.append(ActionType.HOLD.value * len(PositionSizing) + PositionSizing.FIXED_SMALL.value)
        
        return np.array(actions, dtype=np.int32)
    else:
        raise NotImplementedError("Continuous action space not implemented")


def get_action_description(
    action_vector: np.ndarray,
    market_tickers: List[str],
    config: Optional[ActionConfig] = None
) -> str:
    """
    Get human-readable description of an action.
    
    Args:
        action_vector: Action to describe
        market_tickers: Market tickers
        config: Action configuration
        
    Returns:
        Human-readable action description
    """
    if config is None:
        config = ActionConfig()
    
    try:
        market_actions = decode_action(action_vector, market_tickers, config)
        
        descriptions = []
        for ticker, action_info in market_actions.items():
            action_type = action_info["action_type"]
            quantity = action_info["quantity"]
            
            if action_type == ActionType.HOLD:
                continue  # Skip HOLD actions for brevity
            
            action_desc = f"{ticker}: {action_type.name} {quantity}"
            descriptions.append(action_desc)
        
        if not descriptions:
            return "HOLD all positions"
        
        return " | ".join(descriptions)
        
    except Exception as e:
        return f"Error describing action: {str(e)}"


# Create default action config for easy import
default_action_config = ActionConfig()