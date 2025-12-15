"""
Market-agnostic Kalshi RL environment using pre-loaded session data.

This module implements the core RL environment that operates on pre-loaded session data
without database dependencies or async issues. Episodes are generated from historical
session data with guaranteed data continuity.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .session_data_loader import SessionDataPoint, MarketSessionView
from .feature_extractors import build_observation_from_session_data
from .limit_order_action_space import LimitOrderActionSpace
from ..trading.unified_metrics import UnifiedRewardCalculator
from ..trading.order_manager import SimulatedOrderManager
from ..data.orderbook_state import OrderbookState

logger = logging.getLogger("kalshiflow_rl.environments.market_agnostic_env")


def convert_session_data_to_orderbook(market_data: Dict[str, Any], market_ticker: str) -> OrderbookState:
    """
    Convert SessionDataPoint.markets_data to OrderbookState for OrderManager integration.
    
    This is the KEY conversion layer that connects SessionDataLoader → OrderManager.
    
    Args:
        market_data: Dictionary from SessionDataPoint.markets_data[ticker]
        market_ticker: Market ticker for the OrderbookState
        
    Returns:
        OrderbookState instance ready for OrderManager operations
    """
    orderbook = OrderbookState(market_ticker)
    orderbook.apply_snapshot(market_data)
    return orderbook


@dataclass
class EnvConfig:
    """Configuration for environment initialization."""
    max_markets: int = 1     # Single market training (universal strategy across all markets)
    temporal_features: bool = True  # Include time gap and activity analysis
    cash_start: int = 10000  # Starting cash per episode in cents


class MarketAgnosticKalshiEnv(gym.Env):
    """
    Market-agnostic Kalshi RL environment using single-market session views.
    
    This environment operates exclusively on MarketSessionView data, which provides
    an efficient pre-filtered view of a single market from a larger session.
    No runtime market selection or filtering is needed.
    
    Key features:
    - Single-market focus via MarketSessionView
    - No market selection logic (handled by CurriculumService upstream)
    - Market-agnostic feature extraction (model never sees tickers)
    - Unified position tracking matching Kalshi API conventions
    - Primitive action space enabling strategy discovery
    - Simple reward = portfolio value change only
    
    Args:
        market_view: Pre-filtered single-market view from SessionData
        config: Environment configuration
    """
    
    # Observation space dimension (calculated from feature extractors)
    # 1 market × 21 market features + 14 temporal + 12 portfolio + 5 order = 52
    OBSERVATION_DIM = 52
    
    def __init__(
        self,
        market_view: MarketSessionView,
        config: Optional[EnvConfig] = None
    ):
        super().__init__()
        
        self.market_view = market_view
        self.config = config or EnvConfig()
        
        # Validate market view
        if not self.market_view or self.market_view.get_episode_length() < 3:
            raise ValueError("Market view must have at least 3 data points")
        logger.info(f"MarketAgnosticKalshiEnv initialized with market {self.market_view.target_market}, session {self.market_view.session_id}, {self.market_view.get_episode_length()} steps")
        
        # Episode state
        self.current_step: int = 0
        self.current_market: str = self.market_view.target_market  # Pre-selected from view
        self.episode_length: int = self.market_view.get_episode_length()
        
        # Core components (initialized fresh on each reset)
        self._is_reset: bool = False  # Track if environment has been reset
        self.reward_calculator: Optional[UnifiedRewardCalculator] = None
        self.order_manager: Optional[SimulatedOrderManager] = None
        self.action_space_handler: Optional[LimitOrderActionSpace] = None
        
        # Observation history for temporal features
        self.observation_history: List[SessionDataPoint] = []
        
        # Define gym spaces using calculated observation dimension
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.OBSERVATION_DIM,),
            dtype=np.float32
        )
        
        # 21 discrete actions: HOLD (1) + trading actions with 5 position sizes (4×5=20)
        self.action_space = spaces.Discrete(21)
        
        logger.info(f"Environment initialized: obs_space={self.observation_space.shape}, action_space={self.action_space.n}")
        
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode using the market view.
        
        Since we're using MarketSessionView, the market is pre-selected and
        no market selection logic is needed.
        """
        super().reset(seed=seed)
        
        # Market is pre-selected in the view - no selection needed!
        # This is the key simplification from using MarketSessionView
        
        # Reset episode state
        self.current_step = 0
        self.episode_length = self.market_view.get_episode_length()
        self.observation_history = []
        self._is_reset = True  # Mark environment as reset
        
        # Initialize fresh components for this episode
        self.order_manager = SimulatedOrderManager(
            initial_cash=self.config.cash_start  # Now in cents
        )
        self.action_space_handler = LimitOrderActionSpace(
            order_manager=self.order_manager,
            contract_size=10  # Fixed contract size
        )
        
        # Build initial observation
        observation = self._build_observation()
        
        info = {
            "session_id": self.market_view.session_id,
            "market_ticker": self.current_market,
            "episode_length": self.episode_length,
            "initial_cash": self.config.cash_start,
            "coverage_pct": 100.0  # MarketSessionView always has 100% coverage for its market
        }
        
        logger.info(f"Reset episode: session={self.market_view.session_id}, market={self.current_market}, length={self.episode_length}")
        
        return observation, info
    
    def step(self, action: Union[np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step with the given action.
        
        Args:
            action: Integer action (0-4) from limit order action space
            
        Returns:
            observation: Market-agnostic feature vector (52 features)
            reward: Portfolio value change from previous step in cents
            terminated: Episode finished naturally (end of session OR balance <= 0)
            truncated: Episode cut short due to limits  
            info: Additional episode information
        """
        if not self._is_reset:
            raise ValueError("Environment not properly reset")
        
        # Get current portfolio value before action
        prev_portfolio_value = self.order_manager.get_portfolio_value_cents(
            self._get_current_market_prices()
        )
        
        # Get current session data point
        current_data = self.market_view.get_timestep_data(self.current_step)
        if current_data is None:
            # End of session
            terminated = True
            reward = 0.0
            observation = self._build_observation()
        else:
            # Execute action if orderbook data is available for current market
            if self.current_market in current_data.markets_data:
                orderbook = convert_session_data_to_orderbook(
                    current_data.markets_data[self.current_market],
                    self.current_market
                )
                
                # Execute action using action space handler (no async needed)
                try:
                    action_result = self.action_space_handler.execute_action_sync(
                        action, self.current_market, orderbook
                    )
                    # Fix #3: Process action results for better debugging and feedback
                    if action_result is None:
                        logger.debug(f"Invalid action {action} at step {self.current_step}")
                    elif hasattr(action_result, 'status'):
                        if action_result.status == 'success':
                            if hasattr(action_result, 'order') and action_result.order:
                                order = action_result.order
                                logger.debug(
                                    f"Action {action} executed: {order.side.value if hasattr(order, 'side') else 'N/A'} "
                                    f"{order.quantity if hasattr(order, 'quantity') else 0} contracts"
                                )
                        elif action_result.status == 'hold':
                            logger.debug(f"Action {action}: HOLD - no order placed")
                        else:
                            logger.debug(f"Action {action} status: {action_result.status}")
                    else:
                        logger.debug(f"Action {action} result received")
                except Exception as e:
                    logger.warning(f"Action execution failed: {e}")
                    
            # Update observation history
            self.observation_history.append(current_data)
            
            # Advance step
            self.current_step += 1
            
            # Build next observation
            observation = self._build_observation()
            
            # Calculate reward using portfolio value change
            new_portfolio_value = self.order_manager.get_portfolio_value_cents(
                self._get_current_market_prices()
            )
            # Simple reward: portfolio value change in cents
            reward = float(new_portfolio_value - prev_portfolio_value)
            
            # Check termination conditions
            terminated = (
                self.current_step >= self.episode_length or  # End of session
                new_portfolio_value <= 0  # Bankruptcy
            )
        
        truncated = False  # No truncation conditions for now
        
        position_info = self.order_manager.get_position_info()
        current_position = position_info.get(self.current_market, {}).get('position', 0)
        
        info = {
            "step": self.current_step,
            "portfolio_value": self.order_manager.get_portfolio_value_cents(self._get_current_market_prices()),
            "cash_balance": self.order_manager.get_cash_balance_cents(),
            "position": current_position,
            "market_ticker": self.current_market,
            "session_id": self.market_view.session_id,
            "episode_progress": self.current_step / self.episode_length if self.episode_length > 0 else 0.0
        }
        
        return observation, reward, terminated, truncated, info
    
    def set_market_view(self, market_view: MarketSessionView) -> None:
        """
        Manually set a new market view for curriculum learning.
        
        This allows switching between markets without recreating the environment.
        
        Args:
            market_view: New market view to use for next episode
        """
        if not market_view or market_view.get_episode_length() < 3:
            raise ValueError("Market view must have at least 3 data points")
        
        self.market_view = market_view
        self.current_market = market_view.target_market
        self.episode_length = market_view.get_episode_length()
        
        logger.info(f"Market view updated to {market_view.target_market} from session {market_view.session_id} for curriculum learning")
    
    def _build_observation(self) -> np.ndarray:
        """
        Build market-agnostic observation from current session data.
        
        Uses shared feature extractors to ensure consistency between
        training and inference pipelines.
        """
        if self.current_market is None:
            logger.warning("Building observation with no current market - returning zeros")
            return np.zeros(self.OBSERVATION_DIM, dtype=np.float32)
        
        # Get current data point
        current_data = self.market_view.get_timestep_data(self.current_step)
        if current_data is None:
            logger.warning(f"No session data for step {self.current_step} - returning zeros")
            return np.zeros(self.OBSERVATION_DIM, dtype=np.float32)
            
        # Get current market prices for portfolio features
        current_prices = self._get_current_market_prices()
        
        # Extract position data for portfolio features
        # Get position data from OrderManager in UnifiedPositionTracker format
        position_data = self.order_manager.get_position_info()
        
        # Get portfolio value and cash balance
        portfolio_value = self.order_manager.get_portfolio_value_cents(current_prices)
        cash_balance = self.order_manager.get_cash_balance_cents()
        
        # Build observation using shared feature extractor
        observation = build_observation_from_session_data(
            session_data=current_data,
            historical_data=self.observation_history[-10:],  # Last 10 for temporal features
            position_data=position_data,
            portfolio_value=portfolio_value,
            cash_balance=cash_balance,
            max_markets=self.config.max_markets,
            order_features=None  # TODO: Add order features in future
        )
        
        observation = observation.astype(np.float32)
        
        # Validate observation dimensions
        if observation.shape[0] != self.OBSERVATION_DIM:
            logger.error(
                f"Observation dimension mismatch: expected {self.OBSERVATION_DIM}, "
                f"got {observation.shape[0]}. Observation shape: {observation.shape}"
            )
            # Return properly sized zero observation as fallback
            return np.zeros(self.OBSERVATION_DIM, dtype=np.float32)
        
        # Additional validation checks
        if np.any(np.isnan(observation)):
            logger.warning("Observation contains NaN values - replacing with zeros")
            observation = np.nan_to_num(observation, nan=0.0)
        
        if np.any(np.isinf(observation)):
            logger.warning("Observation contains infinite values - clipping")
            observation = np.clip(observation, -10.0, 10.0)
        
        return observation
    
    def _get_current_market_prices(self) -> Dict[str, Dict[str, float]]:
        """
        Extract current market prices for portfolio value calculation.
        
        Returns:
            Dict mapping market ticker to {"bid": float, "ask": float} in cents
            Falls back to mid prices if bid/ask not available
        """
        if (self.current_market is None or
            self.current_step >= len(self.market_view.data_points)):
            return {}
        
        current_data = self.market_view.data_points[self.current_step]
        prices = {}
        
        # Try to get bid/ask from orderbook data first
        if self.current_market in current_data.markets_data:
            market_data = current_data.markets_data[self.current_market]
            yes_bids = market_data.get('yes_bids', {})
            yes_asks = market_data.get('yes_asks', {})
            
            # Extract best bid and ask prices
            best_bid = None
            best_ask = None
            
            if yes_bids:
                try:
                    best_bid = max(map(int, yes_bids.keys()))
                except (ValueError, TypeError):
                    pass
            
            if yes_asks:
                try:
                    best_ask = min(map(int, yes_asks.keys()))
                except (ValueError, TypeError):
                    pass
            
            # Use bid/ask if both available
            if best_bid is not None and best_ask is not None:
                prices[self.current_market] = {
                    "bid": float(best_bid),
                    "ask": float(best_ask)
                }
                return prices
        
        # Fall back to mid prices if bid/ask not available
        if self.current_market in current_data.mid_prices:
            yes_mid, no_mid = current_data.mid_prices[self.current_market]
            if yes_mid is not None and no_mid is not None:
                # Convert Decimal to float and ensure in cents
                # For backward compatibility, create bid/ask from mid with minimal spread
                yes_mid_cents = float(yes_mid)
                prices[self.current_market] = {
                    "bid": yes_mid_cents - 0.5,  # Subtract minimal spread
                    "ask": yes_mid_cents + 0.5   # Add minimal spread
                }
        
        return prices
    
    def close(self) -> None:
        """Clean up resources."""
        self.reward_calculator = None
        self.order_manager = None
        self.action_space_handler = None
        self.observation_history = []
        logger.info("Environment closed and resources cleaned up")