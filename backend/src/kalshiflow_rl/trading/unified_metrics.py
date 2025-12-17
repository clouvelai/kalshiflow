"""
Unified position tracking and reward calculation for Kalshi RL environments.

This module provides a single position tracking system that works identically
for both training and inference, matching Kalshi API conventions exactly.
Implements simplified reward calculation based on portfolio value change only.

PRICE FORMAT CONVENTION:
- INPUT: Trade prices in integer cents (1-99) matching Kalshi API
- CALCULATIONS: All monetary values in integer cents throughout
- OUTPUT: Portfolio values and P&L in cents for exact Kalshi API compatibility
- POSITION TRACKING: Uses Kalshi convention (+YES contracts, -NO contracts)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np

logger = logging.getLogger(__name__)


# PositionInfo and UnifiedPositionTracker removed - now using OrderManager for position tracking


class UnifiedRewardCalculator:
    """
    Unified reward calculation for RL environments.
    
    Implements economically realistic reward calculation that includes:
    - Portfolio value change as base reward
    - Realistic transaction fees matching Kalshi's fee structure
    - Overtrading penalties to discourage excessive activity
    - Volatility penalties to prevent gambling behavior
    """
    
    def __init__(self, reward_scale: float = 0.0001, enable_penalties: bool = True):
        """
        Initialize reward calculator.
        
        Args:
            reward_scale: Scaling factor for rewards (default: 0.0001 for cents to reasonable scale)
            enable_penalties: Whether to enable overtrading and volatility penalties
        """
        self.reward_scale = reward_scale
        self.enable_penalties = enable_penalties
        self.previous_portfolio_value: Optional[int] = None
        self.episode_rewards: List[float] = []
        self.episode_portfolio_values: List[int] = []
        self.episode_start_value: Optional[int] = None
        
        # Track trading activity for overtrading penalty
        self.trades_this_episode: int = 0
        self.steps_this_episode: int = 0
        
        # Track portfolio values for volatility calculation
        self.recent_portfolio_values: List[int] = []  # Rolling window for volatility
        self.volatility_window_size: int = 20  # Calculate volatility over 20 steps
        
        logger.info(f"Initialized reward calculator with scale {reward_scale}, penalties={'enabled' if enable_penalties else 'disabled'}")
    
    def calculate_step_reward(
        self,
        current_portfolio_value: int,
        step_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate reward focused on profitable trading.
        
        Components:
        1. Base reward: Portfolio value change (includes realized P&L) - PRIMARY SIGNAL
        2. Transaction fees: 0.7% of trade value (realistic Kalshi taker fee)
        3. Market impact: 1% base cost for large orders relative to liquidity
        
        The model learns to trade profitably through natural profit/loss signals,
        not through artificial penalties that discourage trading.
        
        Args:
            current_portfolio_value: Current total portfolio value in cents
            step_info: Optional dict with:
                - 'action_taken': bool, whether a non-HOLD action was executed
                - 'spread_cents': int, the spread in cents for the trade
                - 'quantity': int, number of contracts traded
                - 'execution_price': int, actual execution price in cents
                - 'available_liquidity': int, available contracts at best price
            
        Returns:
            Reward value (can be positive or negative)
        """
        self.steps_this_episode += 1
        
        if self.previous_portfolio_value is None:
            # First step - record starting value, no reward
            self.previous_portfolio_value = current_portfolio_value
            self.episode_start_value = current_portfolio_value
            reward = 0.0
            value_change = 0
        else:
            # 1. BASE REWARD: Portfolio value change (includes realized P&L)
            value_change = current_portfolio_value - self.previous_portfolio_value
            base_reward = value_change * self.reward_scale
            reward = base_reward
            
            # Track portfolio values for volatility calculation
            self.recent_portfolio_values.append(current_portfolio_value)
            if len(self.recent_portfolio_values) > self.volatility_window_size:
                self.recent_portfolio_values.pop(0)
            
            # Apply penalties only if enabled
            if self.enable_penalties:
                
                # Components for detailed logging
                fee_penalty = 0.0
                impact_penalty = 0.0
                overtrading_penalty = 0.0
                volatility_penalty = 0.0
                hold_bonus = 0.0
                
                # Check if action was taken or HOLD
                action_taken = step_info and step_info.get('action_taken', False)
                
                if action_taken:
                    # TRADING ACTION - Apply aggressive penalties
                    self.trades_this_episode += 1
                    
                    quantity = step_info.get('quantity', 10)
                    execution_price = step_info.get('execution_price', 50)  # Default mid price
                    spread_cents = step_info.get('spread_cents', 2)
                    
                    # Calculate trade value
                    trade_value = quantity * execution_price  # In cents
                    
                    # 2. REALISTIC TRANSACTION FEES
                    # Kalshi's actual taker fee structure
                    taker_fee_rate = 0.007  # 0.7% - realistic Kalshi fee
                    
                    transaction_fee = trade_value * taker_fee_rate
                    fee_penalty = transaction_fee * self.reward_scale
                    reward -= fee_penalty
                    
                    # 3. MARKET IMPACT COST (for large orders relative to liquidity)
                    available_liquidity = step_info.get('available_liquidity', 1000)
                    if available_liquidity > 0:
                        liquidity_ratio = quantity / available_liquidity
                        if liquidity_ratio > 0.1:  # If taking >10% of available liquidity
                            # Realistic market impact
                            market_impact = (liquidity_ratio ** 2) * execution_price * 0.01  # 1% base impact
                            impact_penalty = market_impact * self.reward_scale
                            reward -= impact_penalty
                    
                # No HOLD bonus - let profit/loss teach when to hold
                
                # No overtrading penalty - let profit/loss teach optimal activity rate
                if self.steps_this_episode % 50 == 0:
                    activity_rate = self.trades_this_episode / self.steps_this_episode
                    logger.info(
                        f"Activity rate: {activity_rate:.1%} (no penalty applied)"
                    )
                
                # No volatility penalty - profitable trading often requires position taking
                if len(self.recent_portfolio_values) >= 5 and self.episode_start_value > 0:
                    portfolio_std = np.std(self.recent_portfolio_values)
                    volatility_pct = (portfolio_std / self.episode_start_value) * 100
                    
                    if self.steps_this_episode % 50 == 0:
                        logger.info(
                            f"Portfolio volatility: {volatility_pct:.1f}% (no penalty applied)"
                        )
                
                # Detailed logging every 20 steps for debugging
                if self.steps_this_episode % 20 == 0:
                    logger.info(
                        f"Step {self.steps_this_episode} Reward Breakdown: "
                        f"Base={base_reward:.6f}, "
                        f"Fees=-{fee_penalty:.6f}, "
                        f"Impact=-{impact_penalty:.6f}, "
                        f"Total={reward:.6f}"
                    )
            
            # Update for next step
            self.previous_portfolio_value = current_portfolio_value
        
        # Track episode data
        self.episode_rewards.append(reward)
        self.episode_portfolio_values.append(current_portfolio_value)
        
        return reward
    
    def calculate_reward(
        self,
        current_portfolio_value: int,
        step_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Alias for calculate_step_reward for backward compatibility.
        
        Args:
            current_portfolio_value: Current total portfolio value in cents
            step_info: Optional additional step information
            
        Returns:
            Reward value (can be positive or negative)
        """
        return self.calculate_step_reward(current_portfolio_value, step_info)
    
    def reset(self, initial_portfolio_value: Optional[int] = None) -> None:
        """
        Reset for new episode.
        
        Args:
            initial_portfolio_value: Starting portfolio value for episode in cents
        """
        self.previous_portfolio_value = initial_portfolio_value
        self.episode_start_value = initial_portfolio_value
        self.episode_rewards.clear()
        self.episode_portfolio_values.clear()
        
        # Reset tracking for penalties
        self.trades_this_episode = 0
        self.steps_this_episode = 0
        self.recent_portfolio_values.clear()
        
        logger.debug(f"Reset reward calculator (initial value: {initial_portfolio_value or 'TBD'}¢)")
    
    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get episode-level reward statistics.
        
        Returns:
            Episode reward statistics including returns, drawdown, Sharpe ratio, and activity metrics
        """
        if not self.episode_portfolio_values or self.episode_start_value is None:
            return {
                "total_return": 0.0,
                "total_reward": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "final_portfolio_value": self.previous_portfolio_value or 0,
                "episode_length": 0,
                "avg_reward_per_step": 0.0,
                "activity_rate": 0.0,
                "portfolio_volatility": 0.0,
                "trades_executed": 0
            }
        
        final_value = self.episode_portfolio_values[-1]
        
        # Total return calculation
        total_return = (final_value / self.episode_start_value - 1) * 100 if self.episode_start_value > 0 else 0.0
        total_reward = sum(self.episode_rewards)
        
        # Maximum drawdown calculation
        max_drawdown = 0.0
        peak_value = self.episode_start_value
        for value in self.episode_portfolio_values:
            if value > peak_value:
                peak_value = value
            drawdown = (peak_value - value) / peak_value * 100 if peak_value > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio calculation (using rewards as returns)
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        reward_std = np.std(self.episode_rewards) if len(self.episode_rewards) > 1 else 0.0
        sharpe_ratio = avg_reward / reward_std if reward_std > 0 else 0.0
        
        # Activity rate calculation
        activity_rate = (self.trades_this_episode / self.steps_this_episode * 100) if self.steps_this_episode > 0 else 0.0
        
        # Portfolio volatility as percentage of starting capital
        portfolio_volatility = 0.0
        if len(self.episode_portfolio_values) > 1 and self.episode_start_value > 0:
            portfolio_std = np.std(self.episode_portfolio_values)
            portfolio_volatility = (portfolio_std / self.episode_start_value) * 100
        
        return {
            "total_return": total_return,
            "total_reward": total_reward,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "final_portfolio_value": final_value,
            "episode_length": len(self.episode_rewards),
            "avg_reward_per_step": avg_reward,
            "reward_volatility": reward_std,
            "activity_rate": activity_rate,
            "portfolio_volatility": portfolio_volatility,
            "trades_executed": self.trades_this_episode
        }


# Position utility functions removed - now handled by OrderManager