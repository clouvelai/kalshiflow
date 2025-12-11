"""
Market-agnostic Kalshi RL environment using pre-loaded session data.

This module implements the core RL environment that operates on pre-loaded session data
without database dependencies or async issues. Episodes are generated from historical
session data with guaranteed data continuity.
"""

import logging
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .session_data_loader import SessionData, SessionDataPoint
from .feature_extractors import build_observation_from_session_data
from .limit_order_action_space import LimitOrderActionSpace
from ..trading.unified_metrics import UnifiedPositionTracker, UnifiedRewardCalculator
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
    Market-agnostic Kalshi RL environment using pre-loaded session data.
    
    This environment operates on pre-loaded session data without database dependencies.
    It generates episodes from session data without exposing market tickers or 
    market-specific metadata to the model. The agent learns universal trading
    strategies that work across all Kalshi markets.
    
    Key features:
    - Session-based episodes with guaranteed data continuity
    - Market-agnostic feature extraction (model never sees tickers)
    - Unified position tracking matching Kalshi API conventions
    - Primitive action space enabling strategy discovery
    - Simple reward = portfolio value change only
    - No database dependencies (data pre-loaded)
    
    Args:
        session_data: Pre-loaded session data for episodes
        config: Environment configuration
    """
    
    # Observation space dimension (calculated from feature extractors)
    # 1 market × 21 market features + 14 temporal + 12 portfolio + 5 order = 52
    OBSERVATION_DIM = 52
    
    def __init__(
        self,
        session_data: SessionData,
        config: Optional[EnvConfig] = None
    ):
        super().__init__()
        
        self.session_data = session_data
        self.config = config or EnvConfig()
        
        # Validate session data
        if not self.session_data or self.session_data.get_episode_length() < 3:
            raise ValueError("Session data must have at least 3 data points")
        logger.info(f"MarketAgnosticKalshiEnv initialized with session {self.session_data.session_id}, {self.session_data.get_episode_length()} steps")
        
        # Episode state
        self.current_step: int = 0
        self.current_market: Optional[str] = None
        self.episode_length: int = self.session_data.get_episode_length()
        
        # Core components (initialized fresh on each reset)
        self.position_tracker: Optional[UnifiedPositionTracker] = None
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
        
        # 5 discrete actions: HOLD, BUY_YES_LIMIT, SELL_YES_LIMIT, BUY_NO_LIMIT, SELL_NO_LIMIT
        self.action_space = spaces.Discrete(5)
        
        logger.info(f"Environment initialized: obs_space={self.observation_space.shape}, action_space={self.action_space.n}")
        
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode using pre-loaded session data.
        
        Initializes fresh components and selects the most active market for single-market training.
        """
        super().reset(seed=seed)
        
        # Select most active market for single-market training
        self.current_market = self._select_most_active_market(self.session_data)
        if not self.current_market:
            raise ValueError(f"No valid markets found in session {self.session_data.session_id}")
        
        # Reset episode state
        self.current_step = 0
        self.episode_length = self.session_data.get_episode_length()
        self.observation_history = []
        
        # Initialize fresh components for this episode
        self.position_tracker = UnifiedPositionTracker(initial_cash=self.config.cash_start)
        self.reward_calculator = UnifiedRewardCalculator()
        self.order_manager = SimulatedOrderManager(initial_cash=self.config.cash_start / 100.0)  # Convert cents to dollars
        self.action_space_handler = LimitOrderActionSpace(
            order_manager=self.order_manager,
            contract_size=10  # Fixed contract size
        )
        
        # Build initial observation
        observation = self._build_observation()
        
        info = {
            "session_id": self.session_data.session_id,
            "market_ticker": self.current_market,
            "episode_length": self.episode_length,
            "markets_available": len(self.session_data.data_points[0].markets_data) if self.session_data.data_points else 0,
            "initial_cash": self.config.cash_start
        }
        
        logger.info(f"Reset episode: session={self.session_data.session_id}, market={self.current_market}, length={self.episode_length}")
        
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
        if self.current_market is None:
            raise ValueError("Environment not properly reset")
        
        # Get current portfolio value before action
        prev_portfolio_value = self.position_tracker.get_total_portfolio_value(
            self._get_current_market_prices()
        )
        
        # Get current session data point
        current_data = self.session_data.get_timestep_data(self.current_step)
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
                    if action_result is None:
                        logger.debug(f"Invalid action {action} at step {self.current_step}")
                except Exception as e:
                    logger.warning(f"Action execution failed: {e}")
                    
            # Update observation history
            self.observation_history.append(current_data)
            
            # Advance step
            self.current_step += 1
            
            # Build next observation
            observation = self._build_observation()
            
            # Calculate reward using portfolio value change
            new_portfolio_value = self.position_tracker.get_total_portfolio_value(
                self._get_current_market_prices()
            )
            reward = self.reward_calculator.calculate_reward(
                prev_portfolio_value, new_portfolio_value
            )
            
            # Check termination conditions
            terminated = (
                self.current_step >= self.episode_length or  # End of session
                new_portfolio_value <= 0  # Bankruptcy
            )
        
        truncated = False  # No truncation conditions for now
        
        info = {
            "step": self.current_step,
            "portfolio_value": self.position_tracker.get_total_portfolio_value(self._get_current_market_prices()),
            "cash_balance": self.position_tracker.cash_balance,
            "position": self.position_tracker.positions.get(self.current_market, {}).get('position', 0),
            "market_ticker": self.current_market,
            "session_id": self.session_data.session_id,
            "episode_progress": self.current_step / self.episode_length if self.episode_length > 0 else 0.0
        }
        
        return observation, reward, terminated, truncated, info
    
    def set_session_data(self, session_data: SessionData) -> None:
        """
        Manually set the session data for curriculum learning.
        
        Args:
            session_data: New session data to use for next episode
        """
        if not session_data or session_data.get_episode_length() < 3:
            raise ValueError("Session data must have at least 3 data points")
        
        self.session_data = session_data
        self.episode_length = session_data.get_episode_length()
        logger.info(f"Session data set to {session_data.session_id} for curriculum learning")
    
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
        current_data = self.session_data.get_timestep_data(self.current_step)
        if current_data is None:
            logger.warning(f"No session data for step {self.current_step} - returning zeros")
            return np.zeros(self.OBSERVATION_DIM, dtype=np.float32)
            
        # Get current market prices for portfolio features
        current_prices = self._get_current_market_prices()
        
        # Extract position data for portfolio features
        # Position data should map ticker to position info dict
        position_data = {}
        if self.current_market in self.position_tracker.positions:
            position_data[self.current_market] = self.position_tracker.positions[self.current_market]
        
        # Get portfolio value and cash balance
        portfolio_value = self.position_tracker.get_total_portfolio_value(current_prices)
        cash_balance = self.position_tracker.cash_balance
        
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
    
    def _select_most_active_market(self, session_data: SessionData) -> Optional[str]:
        """
        Select the most active market by total volume for single-market training.
        
        Args:
            session_data: Loaded session data
            
        Returns:
            Market ticker with highest total volume, or None if no markets
        """
        if not session_data.data_points:
            return None
            
        # Count total activity across all data points for each market
        market_activity = {}
        for data_point in session_data.data_points:
            for market_ticker, market_data in data_point.markets_data.items():
                if market_ticker not in market_activity:
                    market_activity[market_ticker] = 0
                # Use total orderbook depth as activity metric
                total_depth = 0
                if 'yes_bids' in market_data:
                    total_depth += sum(market_data['yes_bids'].values())
                if 'yes_asks' in market_data:
                    total_depth += sum(market_data['yes_asks'].values())
                if 'no_bids' in market_data:
                    total_depth += sum(market_data['no_bids'].values())
                if 'no_asks' in market_data:
                    total_depth += sum(market_data['no_asks'].values())
                market_activity[market_ticker] += total_depth
        
        if not market_activity:
            return None
            
        # Return market with highest total activity
        most_active_market = max(market_activity.items(), key=lambda x: x[1])[0]
        logger.info(f"Selected most active market: {most_active_market} (activity: {market_activity[most_active_market]})")
        return most_active_market
    
    def _get_current_market_prices(self) -> Dict[str, Tuple[float, float]]:
        """
        Extract current market prices for portfolio value calculation.
        
        Returns:
            Dict mapping market ticker to (yes_mid_price, no_mid_price) in cents
        """
        if (self.current_market is None or
            self.current_step >= len(self.session_data.data_points)):
            return {}
        
        current_data = self.session_data.data_points[self.current_step]
        prices = {}
        
        if self.current_market in current_data.mid_prices:
            yes_mid, no_mid = current_data.mid_prices[self.current_market]
            if yes_mid is not None and no_mid is not None:
                # Convert Decimal to float and ensure in cents
                prices[self.current_market] = (float(yes_mid), float(no_mid))
        
        return prices
    
    def close(self) -> None:
        """Clean up resources."""
        self.position_tracker = None
        self.reward_calculator = None
        self.order_manager = None
        self.action_space_handler = None
        self.observation_history = []
        logger.info("Environment closed and resources cleaned up")