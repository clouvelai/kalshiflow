"""
Unified trading metrics calculator for consistent P&L and reward calculations.

This module provides a single source of truth for:
- Position tracking with cost basis
- P&L calculations (realized and unrealized)
- Reward calculations
- Trading fee calculations

Used by both training (kalshi_env.py) and inference (integration.py) to ensure
consistency between training and production environments.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class Position:
    """Represents a trading position with cost basis tracking."""
    
    def __init__(self, market_ticker: str):
        """Initialize an empty position."""
        self.market_ticker = market_ticker
        self.position_yes: float = 0.0
        self.position_no: float = 0.0
        self.avg_cost_yes: float = 0.0
        self.avg_cost_no: float = 0.0
        self.unrealized_pnl: float = 0.0
        self.realized_pnl: float = 0.0  # Track cumulative realized P&L
    
    def to_dict(self) -> Dict[str, float]:
        """Convert position to dictionary format."""
        return {
            'position_yes': self.position_yes,
            'position_no': self.position_no,
            'avg_cost_yes': self.avg_cost_yes,
            'avg_cost_no': self.avg_cost_no,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        }
    
    def update_from_trade(
        self, 
        side: str, 
        direction: str, 
        quantity: float, 
        price_cents: int,
        fee: float = 0.0
    ) -> float:
        """
        Update position from a trade execution.
        
        Args:
            side: 'yes' or 'no'
            direction: 'buy' or 'sell'
            quantity: Number of contracts
            price_cents: Price in cents
            fee: Trading fee in dollars
            
        Returns:
            Immediate realized P&L from this trade
        """
        price_dollars = price_cents / 100.0
        position_key = f'position_{side}'
        avg_cost_key = f'avg_cost_{side}'
        
        current_position = getattr(self, position_key)
        current_avg_cost = getattr(self, avg_cost_key)
        immediate_pnl = 0.0
        
        if direction == 'buy':
            # Update position quantity
            new_position = current_position + quantity
            
            # Update weighted average cost basis
            if current_position > 0:
                # Weighted average: (old_qty * old_avg + new_qty * new_price) / total_qty
                new_avg_cost = (current_position * current_avg_cost + quantity * price_dollars) / new_position
            else:
                # First purchase establishes cost basis
                new_avg_cost = price_dollars
            
            setattr(self, position_key, new_position)
            setattr(self, avg_cost_key, new_avg_cost)
            
        else:  # sell
            # Limit sell quantity to available position
            sell_quantity = min(quantity, current_position)
            
            if sell_quantity > 0 and current_avg_cost > 0:
                # Calculate realized P&L on sale
                immediate_pnl = sell_quantity * (price_dollars - current_avg_cost)
                self.realized_pnl += immediate_pnl
            
            # Reduce position by actual sold quantity
            new_position = max(0, current_position - sell_quantity)
            setattr(self, position_key, new_position)
            
            # If position goes to zero, reset cost basis
            if new_position <= 0:
                setattr(self, avg_cost_key, 0.0)
        
        return immediate_pnl
    
    def calculate_unrealized_pnl(self, yes_mid_cents: float, no_mid_cents: float) -> float:
        """
        Calculate unrealized P&L based on current mid prices.
        
        Args:
            yes_mid_cents: Current yes mid price in cents
            no_mid_cents: Current no mid price in cents
            
        Returns:
            Total unrealized P&L
        """
        yes_pnl = 0.0
        no_pnl = 0.0
        
        if self.position_yes > 0 and self.avg_cost_yes > 0:
            yes_pnl = self.position_yes * (yes_mid_cents / 100.0 - self.avg_cost_yes)
        
        if self.position_no > 0 and self.avg_cost_no > 0:
            no_pnl = self.position_no * (no_mid_cents / 100.0 - self.avg_cost_no)
        
        self.unrealized_pnl = yes_pnl + no_pnl
        return self.unrealized_pnl


class TradingMetricsCalculator:
    """
    Unified calculator for trading metrics, P&L, and rewards.
    
    Ensures consistency between training and inference environments.
    """
    
    def __init__(
        self,
        reward_config: Optional[Dict[str, Any]] = None,
        episode_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the metrics calculator.
        
        Args:
            reward_config: Reward calculation configuration
            episode_config: Episode/session configuration
        """
        self.reward_config = reward_config or self._default_reward_config()
        self.episode_config = episode_config or self._default_episode_config()
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        
        # Portfolio tracking
        self.cash_balance = self.episode_config.get('initial_cash', 10000.0)
        self.initial_cash = self.cash_balance
        self.last_portfolio_value = self.cash_balance
        
        # Performance tracking
        self.total_trades = 0
        self.total_fees_paid = 0.0
        
        logger.info(f"Initialized TradingMetricsCalculator with fee_rate={self.reward_config['trading_fee_rate']:.3f}")
    
    def _default_reward_config(self) -> Dict[str, Any]:
        """Default reward configuration matching kalshi_env.py."""
        return {
            'reward_type': 'pnl_based',
            'pnl_scale': 0.01,
            'action_penalty': 0.001,
            'position_penalty_scale': 0.0001,
            'drawdown_penalty': 0.01,
            'diversification_bonus': 0.005,
            'win_rate_bonus_scale': 0.02,
            'max_reward': 10.0,
            'min_reward': -10.0,
            'normalize_rewards': True,
            'trading_fee_rate': 0.01  # 1% default
        }
    
    def _default_episode_config(self) -> Dict[str, Any]:
        """Default episode configuration."""
        return {
            'initial_cash': 10000.0,
            'max_loss_threshold': 0.5,
            'max_position_threshold': 0.8,
            'max_position_size': 1000,  # Maximum contracts per position
            'max_position_value_ratio': 0.5  # Maximum 50% of portfolio in one position
        }
    
    def get_or_create_position(self, market_ticker: str) -> Position:
        """Get existing position or create new one."""
        if market_ticker not in self.positions:
            self.positions[market_ticker] = Position(market_ticker)
        return self.positions[market_ticker]
    
    def calculate_trade_fee(self, trade_value: float) -> float:
        """Calculate trading fee for a trade value."""
        return trade_value * self.reward_config['trading_fee_rate']
    
    def execute_trade(
        self,
        market_ticker: str,
        side: str,
        direction: str,
        quantity: float,
        price_cents: int
    ) -> Dict[str, Any]:
        """
        Execute a trade and update position tracking.
        
        Args:
            market_ticker: Market identifier
            side: 'yes' or 'no'
            direction: 'buy' or 'sell'
            quantity: Number of contracts
            price_cents: Execution price in cents
            
        Returns:
            Trade execution details including P&L and fees
        """
        position = self.get_or_create_position(market_ticker)
        
        # Calculate trade value and fees
        trade_value = quantity * price_cents / 100.0
        fee = self.calculate_trade_fee(trade_value)
        
        # Update position and get immediate P&L
        immediate_pnl = position.update_from_trade(
            side, direction, quantity, price_cents, fee
        )
        
        # Update cash balance
        if direction == 'buy':
            self.cash_balance -= (trade_value + fee)
        else:
            self.cash_balance += (trade_value - fee)
        
        # Track statistics
        self.total_trades += 1
        self.total_fees_paid += fee
        
        return {
            'market_ticker': market_ticker,
            'side': side,
            'direction': direction,
            'quantity': quantity,
            'price': price_cents,
            'trade_value': trade_value,
            'fee': fee,
            'immediate_pnl': immediate_pnl,
            'cash_balance_after': self.cash_balance
        }
    
    def update_market_prices(self, market_prices: Dict[str, Dict[str, float]]) -> None:
        """
        Update unrealized P&L for all positions based on current market prices.
        
        Args:
            market_prices: Dict of market_ticker -> {'yes_mid': cents, 'no_mid': cents}
        """
        for market_ticker, position in self.positions.items():
            if market_ticker in market_prices:
                prices = market_prices[market_ticker]
                position.calculate_unrealized_pnl(
                    prices.get('yes_mid', 50.0),
                    prices.get('no_mid', 50.0)
                )
    
    def calculate_portfolio_value(self) -> float:
        """Calculate current total portfolio value."""
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        return self.cash_balance + total_unrealized_pnl
    
    def calculate_step_reward(
        self,
        trades_executed: List[Dict[str, Any]],
        market_prices: Optional[Dict[str, Dict[str, float]]] = None
    ) -> float:
        """
        Calculate reward for a step based on trades and current state.
        
        Args:
            trades_executed: List of executed trades in this step
            market_prices: Current market mid prices for unrealized P&L
            
        Returns:
            Calculated reward value
        """
        # Update unrealized P&L if market prices provided
        if market_prices:
            self.update_market_prices(market_prices)
        
        # Calculate immediate P&L from trades
        step_pnl = sum(trade.get('immediate_pnl', 0.0) for trade in trades_executed)
        actions_taken = len(trades_executed)
        
        reward = 0.0
        
        # Base reward from P&L
        reward += step_pnl * self.reward_config['pnl_scale']
        
        # Action penalty (transaction costs beyond fees)
        reward -= actions_taken * self.reward_config.get('action_penalty', 0.001)
        
        # Position penalty (risk management)
        total_position_value = sum(
            abs(pos.position_yes) + abs(pos.position_no)
            for pos in self.positions.values()
        )
        reward -= total_position_value * self.reward_config.get('position_penalty_scale', 0.0001)
        
        # Portfolio-level rewards
        current_portfolio_value = self.calculate_portfolio_value()
        portfolio_change = current_portfolio_value - self.last_portfolio_value
        
        # Drawdown penalty
        if portfolio_change < 0:
            reward -= abs(portfolio_change) * self.reward_config.get('drawdown_penalty', 0.01)
        
        # Diversification bonus
        active_positions = sum(
            1 for pos in self.positions.values()
            if abs(pos.position_yes) + abs(pos.position_no) > 0.1
        )
        if active_positions > 1:
            reward += self.reward_config.get('diversification_bonus', 0.005)
        
        # Update tracking
        self.last_portfolio_value = current_portfolio_value
        
        # Apply reward bounds
        reward = np.clip(
            reward,
            self.reward_config.get('min_reward', -10.0),
            self.reward_config.get('max_reward', 10.0)
        )
        
        # Normalize if requested
        if self.reward_config.get('normalize_rewards', True):
            reward = np.tanh(reward)
        
        return float(reward)
    
    def get_positions_dict(self) -> Dict[str, Dict[str, float]]:
        """Get all positions as a dictionary (for compatibility)."""
        return {
            ticker: pos.to_dict() 
            for ticker, pos in self.positions.items()
        }
    
    def reset(self, initial_cash: Optional[float] = None) -> None:
        """
        Reset calculator state for new episode/session.
        
        Args:
            initial_cash: Starting cash balance (uses config default if not provided)
        """
        self.positions.clear()
        self.cash_balance = initial_cash or self.episode_config.get('initial_cash', 10000.0)
        self.initial_cash = self.cash_balance
        self.last_portfolio_value = self.cash_balance
        self.total_trades = 0
        self.total_fees_paid = 0.0
        
        logger.debug(f"Reset metrics calculator with initial_cash=${self.cash_balance:.2f}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        portfolio_value = self.calculate_portfolio_value()
        
        return {
            'portfolio_value': portfolio_value,
            'cash_balance': self.cash_balance,
            'total_realized_pnl': total_realized_pnl,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_pnl': total_realized_pnl + total_unrealized_pnl,
            'total_return_pct': (portfolio_value - self.initial_cash) / self.initial_cash * 100,
            'total_trades': self.total_trades,
            'total_fees_paid': self.total_fees_paid,
            'num_active_positions': sum(
                1 for pos in self.positions.values()
                if pos.position_yes > 0 or pos.position_no > 0
            ),
            'positions': self.get_positions_dict()
        }