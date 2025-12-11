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
    
    Implements simple reward = portfolio value change approach.
    This captures all important signals naturally without artificial complexity.
    """
    
    def __init__(self, reward_scale: float = 0.0001):
        """
        Initialize reward calculator.
        
        Args:
            reward_scale: Scaling factor for rewards (default: 0.0001 for cents to reasonable scale)
        """
        self.reward_scale = reward_scale
        self.previous_portfolio_value: Optional[int] = None
        self.episode_rewards: List[float] = []
        self.episode_portfolio_values: List[int] = []
        self.episode_start_value: Optional[int] = None
        
        logger.info(f"Initialized reward calculator with scale {reward_scale}")
    
    def calculate_step_reward(
        self,
        current_portfolio_value: int,
        step_info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Calculate reward based on portfolio value change.
        
        Simple approach: reward = (new_value - old_value) * scale
        This naturally captures all trading performance including:
        - Profit from correct predictions
        - Cost of holding positions
        - Opportunity cost of cash
        - Risk/return tradeoffs
        
        Args:
            current_portfolio_value: Current total portfolio value in cents
            step_info: Optional additional step information
            
        Returns:
            Reward value (can be positive or negative)
        """
        if self.previous_portfolio_value is None:
            # First step - record starting value, no reward
            self.previous_portfolio_value = current_portfolio_value
            self.episode_start_value = current_portfolio_value
            reward = 0.0
            value_change = 0
        else:
            # Calculate value change
            value_change = current_portfolio_value - self.previous_portfolio_value
            reward = value_change * self.reward_scale
            
            # Update for next step
            self.previous_portfolio_value = current_portfolio_value
        
        # Track episode data
        self.episode_rewards.append(reward)
        self.episode_portfolio_values.append(current_portfolio_value)
        
        logger.debug(f"Step reward: {reward:.6f} (portfolio: {current_portfolio_value}¢, change: {value_change}¢)")
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
        logger.debug(f"Reset reward calculator (initial value: {initial_portfolio_value or 'TBD'}¢)")
    
    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get episode-level reward statistics.
        
        Returns:
            Episode reward statistics including returns, drawdown, and Sharpe ratio
        """
        if not self.episode_portfolio_values or self.episode_start_value is None:
            return {
                "total_return": 0.0,
                "total_reward": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "final_portfolio_value": self.previous_portfolio_value or 0,
                "episode_length": 0,
                "avg_reward_per_step": 0.0
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
        
        return {
            "total_return": total_return,
            "total_reward": total_reward,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "final_portfolio_value": final_value,
            "episode_length": len(self.episode_rewards),
            "avg_reward_per_step": avg_reward,
            "reward_volatility": reward_std
        }


# Position utility functions removed - now handled by OrderManager