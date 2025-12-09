"""
KalshiTradingEnv - Multi-market Gymnasium environment for Kalshi RL Trading Subsystem.

CRITICAL ARCHITECTURAL REQUIREMENTS:
1. This environment ONLY replays historical data (NO live WebSocket connections)
2. It does NOT use async/await (Gymnasium environments are synchronous)
3. ALL data must be preloaded - NO DB queries during step() or reset()
4. Must use IDENTICAL observation builder as actor (from observation_space.py)
5. Supports multiple markets simultaneously
6. NO blocking operations during training episodes
"""

import asyncio
import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List, Optional, Tuple, Union, Iterator
from datetime import datetime, timedelta
import random
import time
from copy import deepcopy

from .observation_space import (
    build_observation_from_orderbook, 
    create_observation_space, 
    ObservationConfig,
    get_observation_feature_names
)
from .action_space import (
    create_action_space,
    decode_action,
    validate_action,
    get_action_description,
    ActionConfig,
    ActionType
)
from .historical_data_loader import (
    HistoricalDataLoader,
    DataLoadConfig,
    HistoricalDataPoint
)
from ..config import config

logger = logging.getLogger("kalshiflow_rl.kalshi_env")


class KalshiTradingEnv(gym.Env):
    """
    Multi-market Kalshi trading environment for reinforcement learning.
    
    This environment:
    - Loads historical orderbook data during initialization
    - Provides multi-market observations and actions
    - Simulates realistic trading with position tracking
    - Calculates rewards based on market performance
    - Supports variable episode lengths and market configurations
    
    CRITICAL: Uses same observation builder as inference actor for consistency.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 4
    }
    
    def __init__(
        self,
        market_tickers: Optional[List[str]] = None,
        observation_config: Optional[ObservationConfig] = None,
        action_config: Optional[ActionConfig] = None,
        data_config: Optional[DataLoadConfig] = None,
        reward_config: Optional[Dict[str, Any]] = None,
        episode_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Kalshi trading environment.
        
        Args:
            market_tickers: List of market tickers to trade (uses config default if None)
            observation_config: Observation space configuration
            action_config: Action space configuration  
            data_config: Historical data loading configuration
            reward_config: Reward calculation configuration
            episode_config: Episode management configuration
        """
        super().__init__()
        
        # Configuration
        self.market_tickers = market_tickers or config.RL_MARKET_TICKERS
        self.observation_config = observation_config or ObservationConfig()
        self.action_config = action_config or ActionConfig()
        self.data_config = data_config or DataLoadConfig()
        self.reward_config = reward_config or self._default_reward_config()
        
        # Merge episode config with defaults
        default_episode_config = self._default_episode_config()
        if episode_config:
            default_episode_config.update(episode_config)
        self.episode_config = default_episode_config
        
        # Validate market count
        if len(self.market_tickers) > self.observation_config.max_markets:
            raise ValueError(f"Too many markets: {len(self.market_tickers)} > {self.observation_config.max_markets}")
        
        # Create observation and action spaces
        self.observation_space = create_observation_space(self.observation_config)
        self.action_space = create_action_space(len(self.market_tickers), self.action_config)
        
        # Environment state
        self.current_step = 0
        self.episode_length = 0
        self.historical_data: Dict[str, List[HistoricalDataPoint]] = {}
        self.data_iterator = None
        self.current_market_states: Dict[str, Dict[str, Any]] = {}
        
        # Trading state
        self.positions: Dict[str, Dict[str, float]] = {}  # market -> {position_yes, position_no, unrealized_pnl}
        self.trade_history: List[Dict[str, Any]] = []
        self.cash_balance = self.episode_config['initial_cash']
        self.initial_portfolio_value = self.cash_balance
        
        # Episode tracking
        self.episode_count = 0
        self.episode_rewards: List[float] = []
        self.episode_start_time = None
        self.last_portfolio_value = self.cash_balance
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = self.cash_balance
        
        # Data loading (CRITICAL: All data must be preloaded here)
        self._preload_data()
        
        logger.info(f"Initialized KalshiTradingEnv: "
                   f"{len(self.market_tickers)} markets, "
                   f"obs_space={self.observation_space.shape}, "
                   f"action_space={self.action_space}")
    
    def _default_reward_config(self) -> Dict[str, Any]:
        """Default reward configuration."""
        return {
            'reward_type': 'pnl_based',  # 'pnl_based', 'sharpe_based', 'risk_adjusted'
            'pnl_scale': 0.01,  # Scale PnL to reasonable reward range
            'action_penalty': 0.001,  # Small penalty for taking actions (transaction costs)
            'position_penalty_scale': 0.0001,  # Penalty for large positions
            'drawdown_penalty': 0.01,  # Penalty for portfolio drawdown
            'diversification_bonus': 0.005,  # Bonus for diversified positions
            'win_rate_bonus_scale': 0.02,  # Bonus based on win rate
            'max_reward': 10.0,  # Cap rewards to prevent instability
            'min_reward': -10.0,  # Cap negative rewards
            'normalize_rewards': True  # Normalize rewards to [-1, 1] range
        }
    
    def _default_episode_config(self) -> Dict[str, Any]:
        """Default episode configuration."""
        return {
            'max_steps': getattr(config, 'MAX_EPISODE_STEPS', 1000),
            'initial_cash': 10000.0,  # Starting cash balance
            'min_episode_length': 100,  # Minimum episode steps
            'max_episode_length': 2000,  # Maximum episode steps
            'early_termination': True,  # Allow early termination
            'max_loss_threshold': 0.5,  # Terminate if portfolio loses 50%
            'max_position_threshold': 0.8  # Terminate if single position > 80% of portfolio
        }
    
    def _preload_data(self) -> None:
        """
        Preload all historical data for training.
        
        CRITICAL: This is called during __init__ and loads ALL data needed
        for training episodes. NO database queries allowed after this point.
        """
        logger.info("Preloading historical data for training...")
        
        # Configure data loading parameters
        self.data_config.market_tickers = self.market_tickers
        
        if not self.data_config.start_time or not self.data_config.end_time:
            # Default to recent data window
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=self.data_config.window_hours)
            self.data_config.start_time = start_time
            self.data_config.end_time = end_time
        
        # Use synchronous wrapper since Gymnasium environments are synchronous
        try:
            # Create event loop if none exists (for testing)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Load historical data
            loader = HistoricalDataLoader()
            self.historical_data = loop.run_until_complete(
                self._load_data_sync(loader)
            )
            
            if not self.historical_data:
                logger.warning("No historical data loaded - using dummy data for testing")
                self.historical_data = self._generate_dummy_data()
            
            # Validate loaded data
            total_points = sum(len(points) for points in self.historical_data.values())
            logger.info(f"Preloaded {total_points} data points for {len(self.historical_data)} markets")
            
            if total_points < self.episode_config['min_episode_length']:
                raise ValueError(f"Insufficient data: {total_points} < {self.episode_config['min_episode_length']}")
            
            # Set episode length based on available data
            self.episode_length = min(
                total_points // len(self.market_tickers) if self.market_tickers else total_points,
                self.episode_config['max_episode_length']
            )
            
            logger.info(f"Episode length set to {self.episode_length} steps")
            
        except Exception as e:
            logger.error(f"Failed to preload historical data: {e}")
            logger.warning("Using dummy data for testing")
            self.historical_data = self._generate_dummy_data()
            self.episode_length = self.episode_config['min_episode_length']
    
    async def _load_data_sync(self, loader: HistoricalDataLoader) -> Dict[str, List[HistoricalDataPoint]]:
        """Async wrapper for data loading."""
        try:
            await loader.connect()
            data = await loader.load_historical_data(self.data_config)
            await loader.disconnect()
            return data
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            await loader.disconnect()
            return {}
    
    def _generate_dummy_data(self) -> Dict[str, List[HistoricalDataPoint]]:
        """Generate dummy data for testing when real data is unavailable."""
        dummy_data = {}
        
        base_timestamp = int(time.time() * 1000) - 24 * 3600 * 1000  # 24 hours ago
        
        for market_ticker in self.market_tickers:
            data_points = []
            
            # Generate 500 dummy data points per market
            for i in range(500):
                timestamp = base_timestamp + i * 60000  # 1 minute intervals
                
                # Simulate realistic orderbook data
                yes_mid = 45 + 10 * np.sin(i * 0.1) + np.random.normal(0, 2)
                no_mid = 100 - yes_mid
                
                # Clamp to valid range
                yes_mid = max(10, min(90, yes_mid))
                no_mid = max(10, min(90, no_mid))
                
                spread = 1 + np.random.exponential(1)
                
                orderbook_state = {
                    'market_ticker': market_ticker,
                    'timestamp_ms': timestamp,
                    'sequence_number': i + 1,
                    'yes_bids': {int(yes_mid - spread/2): 100 + np.random.randint(0, 200)},
                    'yes_asks': {int(yes_mid + spread/2): 100 + np.random.randint(0, 200)},
                    'no_bids': {int(no_mid - spread/2): 100 + np.random.randint(0, 200)},
                    'no_asks': {int(no_mid + spread/2): 100 + np.random.randint(0, 200)},
                    'last_update_time': timestamp,
                    'last_sequence': i + 1,
                    'yes_spread': int(spread),
                    'no_spread': int(spread),
                    'yes_mid_price': yes_mid,
                    'no_mid_price': no_mid,
                    'total_volume': 400 + np.random.randint(0, 600)
                }
                
                data_point = HistoricalDataPoint(
                    timestamp_ms=timestamp,
                    market_ticker=market_ticker,
                    orderbook_state=orderbook_state,
                    sequence_number=i + 1,
                    is_snapshot=True
                )
                data_points.append(data_point)
            
            dummy_data[market_ticker] = data_points
        
        logger.info("Generated dummy data for testing")
        return dummy_data
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        CRITICAL: NO database queries allowed here - uses preloaded data only.
        
        Args:
            seed: Random seed
            options: Reset options
            
        Returns:
            (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset episode state
        self.current_step = 0
        self.episode_count += 1
        self.episode_start_time = time.time()
        
        # Reset trading state
        self.positions = {
            ticker: {'position_yes': 0.0, 'position_no': 0.0, 'unrealized_pnl': 0.0}
            for ticker in self.market_tickers
        }
        self.trade_history = []
        self.cash_balance = self.episode_config['initial_cash']
        self.initial_portfolio_value = self.cash_balance
        self.last_portfolio_value = self.cash_balance
        
        # Reset performance tracking for this episode
        episode_start_trades = self.total_trades
        episode_start_pnl = self.total_pnl
        
        # Create data iterator for this episode (deterministic by default for testing)
        if options and options.get('shuffle_data', False):
            # Only shuffle if explicitly requested 
            shuffled_data = {}
            for ticker, data_points in self.historical_data.items():
                shuffled_points = data_points.copy()
                random.shuffle(shuffled_points)
                shuffled_data[ticker] = shuffled_points[:self.episode_length]
            
            self.data_iterator = self._create_synchronized_iterator(shuffled_data)
        else:
            # Use sequential data (deterministic)
            self.data_iterator = self._create_synchronized_iterator(self.historical_data)
        
        # Get initial market states
        try:
            self.current_market_states = next(self.data_iterator)
        except StopIteration:
            logger.error("No data available for episode start")
            # Use last known states or generate minimal states
            self.current_market_states = self._get_minimal_market_states()
        
        # Build initial observation (CRITICAL: Uses same function as actor)
        initial_observation = build_observation_from_orderbook(
            orderbook_states=self._extract_orderbook_states(),
            position_states=self.positions,
            config=self.observation_config
        )
        
        # Create info dict
        info = {
            'episode': self.episode_count,
            'step': self.current_step,
            'cash_balance': self.cash_balance,
            'portfolio_value': self.cash_balance,
            'total_positions': sum(
                abs(pos['position_yes']) + abs(pos['position_no']) 
                for pos in self.positions.values()
            ),
            'markets': list(self.market_tickers),
            'market_states_available': len(self.current_market_states)
        }
        
        logger.debug(f"Reset episode {self.episode_count}: "
                    f"observation_shape={initial_observation.shape}, "
                    f"available_markets={len(self.current_market_states)}")
        
        return initial_observation, info
    
    def step(self, action: Union[np.ndarray, int]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        CRITICAL: NO database queries allowed here - uses preloaded data only.
        
        Args:
            action: Action to take (encoded according to action_space)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.current_step >= self.episode_length:
            logger.warning("Step called after episode termination")
            return self._get_final_state()
        
        self.current_step += 1
        
        # Decode action to market-specific actions
        try:
            market_actions = decode_action(
                action if isinstance(action, np.ndarray) else np.array([action]),
                self.market_tickers,
                self.action_config
            )
        except Exception as e:
            logger.error(f"Action decoding failed: {e}")
            market_actions = {}
        
        # Validate action constraints
        is_valid, violations = validate_action(
            action if isinstance(action, np.ndarray) else np.array([action]),
            self.market_tickers,
            self.positions,
            self._calculate_portfolio_value(),
            self.action_config
        )
        
        if not is_valid:
            logger.debug(f"Invalid action: {violations}")
            # Apply penalty for invalid action but continue episode
            reward = self.reward_config['min_reward'] * 0.1
        else:
            # Execute valid actions and calculate reward
            reward = self._execute_actions_and_calculate_reward(market_actions)
        
        # Advance to next market state
        try:
            self.current_market_states = next(self.data_iterator)
        except StopIteration:
            # End of data - terminate episode
            terminated = True
            truncated = False
        else:
            # Check termination conditions
            terminated, truncated = self._check_termination_conditions()
        
        # Update portfolio values and positions
        self._update_position_values()
        
        # Build next observation (CRITICAL: Same function as actor)
        observation = build_observation_from_orderbook(
            orderbook_states=self._extract_orderbook_states(),
            position_states=self.positions,
            config=self.observation_config
        )
        
        # Create info dict
        info = self._create_step_info(market_actions, is_valid, violations, reward)
        
        # Track episode rewards
        if terminated or truncated:
            self.episode_rewards.append(reward)
        
        return observation, reward, terminated, truncated, info
    
    def _execute_actions_and_calculate_reward(self, market_actions: Dict[str, Dict[str, Any]]) -> float:
        """Execute market actions and calculate step reward."""
        step_pnl = 0.0
        actions_taken = 0
        
        for market_ticker, action_info in market_actions.items():
            action_type = action_info['action_type']
            quantity = action_info['quantity']
            
            # Skip HOLD actions
            if action_type == ActionType.HOLD:
                continue
            
            # Get current market state for this ticker
            if market_ticker not in self.current_market_states:
                continue
                
            market_state = self.current_market_states[market_ticker].orderbook_state
            
            # Simulate trade execution
            executed_trade = self._simulate_trade_execution(
                market_ticker, action_type, quantity, market_state
            )
            
            if executed_trade:
                # Update positions
                self._update_position_from_trade(market_ticker, executed_trade)
                
                # Track trade
                self.trade_history.append(executed_trade)
                self.total_trades += 1
                actions_taken += 1
                
                # Add to step PnL
                step_pnl += executed_trade.get('immediate_pnl', 0.0)
        
        # Calculate total reward for this step
        reward = self._calculate_step_reward(step_pnl, actions_taken)
        
        return reward
    
    def _simulate_trade_execution(
        self,
        market_ticker: str,
        action_type: ActionType,
        quantity: int,
        market_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Simulate realistic trade execution based on orderbook."""
        
        if action_type == ActionType.HOLD:
            return None
        
        # Determine side and direction
        side = 'yes' if action_type in [ActionType.BUY_YES, ActionType.SELL_YES] else 'no'
        direction = 'buy' if action_type in [ActionType.BUY_YES, ActionType.BUY_NO] else 'sell'
        
        # Get relevant orderbook side
        if direction == 'buy':
            book_key = f'{side}_asks'  # Buy from asks
        else:
            book_key = f'{side}_bids'  # Sell to bids
        
        book = market_state.get(book_key, {})
        if not book:
            return None  # No liquidity available
        
        # Get best price
        if direction == 'buy':
            best_price = min(int(p) for p in book.keys())
        else:
            best_price = max(int(p) for p in book.keys())
        
        available_quantity = book.get(str(best_price), 0)
        
        # Execute partial fill if necessary
        executed_quantity = min(quantity, available_quantity)
        if executed_quantity <= 0:
            return None
        
        # Calculate trade value and fees
        trade_value = executed_quantity * best_price / 100.0  # Convert cents to dollars
        fee = trade_value * 0.01  # 1% fee (simplified)
        
        # Calculate immediate P&L impact
        current_position = self.positions.get(market_ticker, {})
        immediate_pnl = self._calculate_trade_pnl(
            current_position, side, direction, executed_quantity, best_price
        )
        
        return {
            'timestamp': market_state.get('timestamp_ms', int(time.time() * 1000)),
            'market_ticker': market_ticker,
            'side': side,
            'direction': direction,
            'quantity': executed_quantity,
            'price': best_price,
            'trade_value': trade_value,
            'fee': fee,
            'immediate_pnl': immediate_pnl
        }
    
    def _calculate_trade_pnl(
        self,
        current_position: Dict[str, float],
        side: str,
        direction: str,
        quantity: int,
        price: int
    ) -> float:
        """Calculate immediate P&L impact of a trade."""
        # Simplified P&L calculation
        # In reality, this would consider average cost basis, etc.
        
        current_qty = current_position.get(f'position_{side}', 0.0)
        trade_value = quantity * price / 100.0
        
        if direction == 'sell' and current_qty > 0:
            # Closing or reducing position - realize P&L
            avg_cost = 50.0  # Simplified: assume $0.50 average cost
            pnl = quantity * (price / 100.0 - avg_cost)
            return pnl
        else:
            # Opening or increasing position - no immediate P&L
            return 0.0
    
    def _update_position_from_trade(self, market_ticker: str, trade: Dict[str, Any]) -> None:
        """Update position tracking from executed trade."""
        if market_ticker not in self.positions:
            self.positions[market_ticker] = {'position_yes': 0.0, 'position_no': 0.0, 'unrealized_pnl': 0.0}
        
        side = trade['side']
        direction = trade['direction']
        quantity = trade['quantity']
        
        position_key = f'position_{side}'
        
        if direction == 'buy':
            self.positions[market_ticker][position_key] += quantity
        else:
            self.positions[market_ticker][position_key] -= quantity
        
        # Update cash balance
        trade_cost = trade['trade_value'] + trade['fee']
        if direction == 'buy':
            self.cash_balance -= trade_cost
        else:
            self.cash_balance += trade_cost - trade['fee']  # Subtract fee in both cases
    
    def _update_position_values(self) -> None:
        """Update unrealized P&L for all positions based on current market prices."""
        for market_ticker in self.positions:
            if market_ticker not in self.current_market_states:
                continue
            
            market_state = self.current_market_states[market_ticker].orderbook_state
            position = self.positions[market_ticker]
            
            # Calculate mark-to-market value
            yes_mid = market_state.get('yes_mid_price', 50.0)
            no_mid = market_state.get('no_mid_price', 50.0)
            
            yes_position = position['position_yes']
            no_position = position['position_no']
            
            # Calculate unrealized P&L (simplified)
            avg_cost = 50.0  # Simplified average cost
            yes_pnl = yes_position * (yes_mid / 100.0 - avg_cost)
            no_pnl = no_position * (no_mid / 100.0 - avg_cost)
            
            position['unrealized_pnl'] = yes_pnl + no_pnl
    
    def _calculate_step_reward(self, step_pnl: float, actions_taken: int) -> float:
        """Calculate reward for the current step."""
        reward = 0.0
        
        # Base reward from P&L
        reward += step_pnl * self.reward_config['pnl_scale']
        
        # Action penalty (transaction costs)
        reward -= actions_taken * self.reward_config['action_penalty']
        
        # Position penalty (risk management)
        total_position_value = sum(
            abs(pos['position_yes']) + abs(pos['position_no'])
            for pos in self.positions.values()
        )
        reward -= total_position_value * self.reward_config['position_penalty_scale']
        
        # Portfolio-level rewards
        current_portfolio_value = self._calculate_portfolio_value()
        portfolio_change = current_portfolio_value - self.last_portfolio_value
        
        # Drawdown penalty
        if portfolio_change < 0:
            reward -= abs(portfolio_change) * self.reward_config['drawdown_penalty']
        
        # Diversification bonus
        active_positions = sum(
            1 for pos in self.positions.values()
            if abs(pos['position_yes']) + abs(pos['position_no']) > 0.1
        )
        if active_positions > 1:
            reward += self.reward_config['diversification_bonus']
        
        # Update tracking
        self.last_portfolio_value = current_portfolio_value
        
        # Apply reward bounds
        reward = np.clip(
            reward,
            self.reward_config['min_reward'],
            self.reward_config['max_reward']
        )
        
        # Normalize if requested
        if self.reward_config['normalize_rewards']:
            reward = np.tanh(reward)
        
        return float(reward)
    
    def _calculate_portfolio_value(self) -> float:
        """Calculate current total portfolio value."""
        total_unrealized_pnl = sum(
            pos.get('unrealized_pnl', 0.0) for pos in self.positions.values()
        )
        return self.cash_balance + total_unrealized_pnl
    
    def _check_termination_conditions(self) -> Tuple[bool, bool]:
        """Check if episode should terminate."""
        terminated = False
        truncated = False
        
        # Check step limit
        if self.current_step >= self.episode_length:
            truncated = True
        
        # Check early termination conditions
        if self.episode_config['early_termination']:
            portfolio_value = self._calculate_portfolio_value()
            
            # Loss threshold
            loss_ratio = (self.initial_portfolio_value - portfolio_value) / self.initial_portfolio_value
            if loss_ratio > self.episode_config['max_loss_threshold']:
                terminated = True
                logger.info(f"Episode terminated due to loss threshold: {loss_ratio:.2%}")
            
            # Position concentration
            if self.positions:
                max_position_value = max(
                    abs(pos['position_yes']) + abs(pos['position_no'])
                    for pos in self.positions.values()
                )
                concentration_ratio = max_position_value * 50 / max(portfolio_value, 1.0)  # Assume $0.50 avg
                
                if concentration_ratio > self.episode_config['max_position_threshold']:
                    terminated = True
                    logger.info(f"Episode terminated due to position concentration: {concentration_ratio:.2%}")
        
        return terminated, truncated
    
    def _create_step_info(
        self,
        market_actions: Dict[str, Dict[str, Any]],
        is_valid: bool,
        violations: List[str],
        reward: float
    ) -> Dict[str, Any]:
        """Create info dictionary for step return."""
        portfolio_value = self._calculate_portfolio_value()
        
        # Create deterministic info dict (avoid non-deterministic elements)
        info = {
            'step': self.current_step,
            'episode': self.episode_count,
            'portfolio_value': float(portfolio_value),  # Ensure float type
            'cash_balance': float(self.cash_balance),
            'total_positions': float(sum(
                abs(pos['position_yes']) + abs(pos['position_no']) 
                for pos in self.positions.values()
            )),
            'total_unrealized_pnl': float(sum(
                pos.get('unrealized_pnl', 0.0) for pos in self.positions.values()
            )),
            'total_trades': len(self.trade_history),
            'actions_taken': len([
                action for action in market_actions.values()
                if action['action_type'] != ActionType.HOLD
            ]) if market_actions else 0,
            'action_valid': is_valid,
            'reward': float(reward),
            'markets_active': len(self.current_market_states),
            # Remove potentially non-deterministic fields for Gymnasium compatibility
            # 'action_violations': violations,  # This can vary in non-deterministic ways
            # 'action_description': get_action_description(...)  # Contains strings that may vary
        }
        
        return info
    
    def _get_final_state(self) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Get final state when episode is already terminated."""
        observation = build_observation_from_orderbook(
            orderbook_states=self._extract_orderbook_states(),
            position_states=self.positions,
            config=self.observation_config
        )
        
        info = {
            'episode_completed': True,
            'final_portfolio_value': self._calculate_portfolio_value(),
            'total_trades': len(self.trade_history),
            'episode_length': self.current_step
        }
        
        return observation, 0.0, True, False, info
    
    def _extract_orderbook_states(self) -> Dict[str, Dict[str, Any]]:
        """Extract orderbook states from current market states."""
        orderbook_states = {}
        for ticker, data_point in self.current_market_states.items():
            orderbook_states[ticker] = data_point.orderbook_state
        return orderbook_states
    
    def _get_minimal_market_states(self) -> Dict[str, HistoricalDataPoint]:
        """Generate minimal market states for fallback."""
        minimal_states = {}
        current_time = int(time.time() * 1000)
        
        for ticker in self.market_tickers:
            orderbook_state = {
                'market_ticker': ticker,
                'timestamp_ms': current_time,
                'sequence_number': 1,
                'yes_bids': {49: 100},
                'yes_asks': {51: 100},
                'no_bids': {49: 100},
                'no_asks': {51: 100},
                'last_update_time': current_time,
                'last_sequence': 1,
                'yes_spread': 2,
                'no_spread': 2,
                'yes_mid_price': 50.0,
                'no_mid_price': 50.0,
                'total_volume': 400
            }
            
            data_point = HistoricalDataPoint(
                timestamp_ms=current_time,
                market_ticker=ticker,
                orderbook_state=orderbook_state,
                sequence_number=1,
                is_snapshot=True
            )
            minimal_states[ticker] = data_point
        
        return minimal_states
    
    def _create_synchronized_iterator(
        self, 
        market_data: Dict[str, List[HistoricalDataPoint]]
    ) -> Iterator[Dict[str, HistoricalDataPoint]]:
        """Create synchronized iterator over market data."""
        # Simple implementation: round-robin through timestamps
        max_points = min(len(points) for points in market_data.values()) if market_data else 0
        max_points = min(max_points, self.episode_length)
        
        for i in range(max_points):
            step_data = {}
            for ticker, points in market_data.items():
                if i < len(points):
                    step_data[ticker] = points[i]
            
            if step_data:  # Only yield if we have data for at least one market
                yield step_data
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment (optional for RL training)."""
        if mode == 'human':
            portfolio_value = self._calculate_portfolio_value()
            print(f"\n=== Episode {self.episode_count}, Step {self.current_step} ===")
            print(f"Portfolio Value: ${portfolio_value:.2f}")
            print(f"Cash Balance: ${self.cash_balance:.2f}")
            print(f"Total Trades: {len(self.trade_history)}")
            
            for ticker, position in self.positions.items():
                if abs(position['position_yes']) + abs(position['position_no']) > 0.1:
                    print(f"{ticker}: YES={position['position_yes']:.1f}, "
                          f"NO={position['position_no']:.1f}, "
                          f"PnL=${position['unrealized_pnl']:.2f}")
            
            if self.current_market_states:
                print("\nCurrent Market States:")
                for ticker, data_point in self.current_market_states.items():
                    state = data_point.orderbook_state
                    print(f"  {ticker}: YES_MID=${state.get('yes_mid_price', 0):.2f}, "
                          f"NO_MID=${state.get('no_mid_price', 0):.2f}")
        
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        self.historical_data.clear()
        self.current_market_states.clear()
        self.data_iterator = None
        logger.info("Environment closed")
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the current episode."""
        portfolio_value = self._calculate_portfolio_value()
        total_return = (portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value
        
        winning_trades = sum(
            1 for trade in self.trade_history
            if trade.get('immediate_pnl', 0) > 0
        )
        
        win_rate = winning_trades / max(len(self.trade_history), 1)
        
        return {
            'episode': self.episode_count,
            'steps': self.current_step,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_trades': len(self.trade_history),
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'cash_balance': self.cash_balance,
            'active_positions': sum(
                1 for pos in self.positions.values()
                if abs(pos['position_yes']) + abs(pos['position_no']) > 0.1
            )
        }