"""
Unified position tracking and reward calculation for Kalshi RL environments.

This module provides a single position tracking system that works identically
for both training and inference, matching Kalshi API conventions exactly.
Implements simplified reward calculation based on portfolio value change only.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """
    Position information for a single market using Kalshi convention.
    
    Follows Kalshi API format:
    - position: +N for YES contracts, -N for NO contracts
    - cost_basis: Total dollars spent to acquire position
    - realized_pnl: Cumulative realized profit/loss from trades
    """
    position: int = 0           # +YES contracts / -NO contracts (Kalshi convention)
    cost_basis: float = 0.0     # Total cost in dollars to acquire position
    realized_pnl: float = 0.0   # Cumulative realized P&L from closed trades
    last_price: float = 50.0    # Last known market price for unrealized P&L
    

class UnifiedPositionTracker:
    """
    Unified position tracking using Kalshi API conventions.
    
    This tracker works identically for training and inference environments,
    ensuring consistent position management and P&L calculation across all
    use cases. Uses Kalshi's +YES/-NO convention for positions.
    """
    
    def __init__(self, initial_cash: float = 1000.0):
        """
        Initialize position tracker.
        
        Args:
            initial_cash: Starting cash balance
        """
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.positions: Dict[str, PositionInfo] = {}  # market_ticker -> PositionInfo
        self.trade_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized position tracker with ${initial_cash:,.2f} cash")
    
    def update_position(
        self,
        market_ticker: str,
        side: str,  # "YES" or "NO"
        quantity: int,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update position from a trade execution.
        
        Args:
            market_ticker: Market identifier
            side: "YES" or "NO" 
            quantity: Number of contracts (positive for buy, negative for sell)
            price: Trade price in cents
            timestamp: Trade timestamp
            
        Returns:
            Trade information dictionary
        """
        # Implementation placeholder - will be completed in M5
        if market_ticker not in self.positions:
            self.positions[market_ticker] = PositionInfo()
            
        position = self.positions[market_ticker]
        
        # Convert to Kalshi convention (+YES, -NO)
        if side == "YES":
            contracts_delta = quantity
        else:  # side == "NO"
            contracts_delta = -quantity
            
        # Calculate trade value
        trade_value = abs(quantity) * price / 100.0  # Convert cents to dollars
        
        # Update position (simplified - full implementation in M5)
        old_position = position.position
        position.position += contracts_delta
        position.last_price = price
        
        # Update cash (simplified)
        if quantity > 0:  # Buying
            self.cash_balance -= trade_value
            position.cost_basis += trade_value
        else:  # Selling
            self.cash_balance += trade_value
            # Calculate realized P&L (simplified)
            
        # Record trade
        trade_info = {
            "timestamp": timestamp or datetime.now(),
            "market_ticker": market_ticker,
            "side": side,
            "quantity": quantity,
            "price": price,
            "trade_value": trade_value,
            "position_before": old_position,
            "position_after": position.position,
            "cash_after": self.cash_balance
        }
        
        self.trade_history.append(trade_info)
        
        logger.debug(f"Updated position for {market_ticker}: {old_position} -> {position.position} contracts")
        return trade_info
    
    def calculate_unrealized_pnl(self, market_prices: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate unrealized P&L for all positions.
        
        Args:
            market_prices: Current market prices {market_ticker: price_cents}
            
        Returns:
            Unrealized P&L by market {market_ticker: pnl_dollars}
        """
        # Implementation placeholder - will be completed in M5
        unrealized_pnl = {}
        
        for market_ticker, position in self.positions.items():
            if position.position == 0:
                unrealized_pnl[market_ticker] = 0.0
                continue
                
            current_price = market_prices.get(market_ticker, position.last_price)
            
            # Simplified unrealized P&L calculation
            # Full implementation will handle YES/NO pricing correctly
            position_value = abs(position.position) * current_price / 100.0
            unrealized_pnl[market_ticker] = position_value - position.cost_basis
            
        return unrealized_pnl
    
    def get_total_portfolio_value(self, market_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and positions.
        
        Args:
            market_prices: Current market prices
            
        Returns:
            Total portfolio value in dollars
        """
        # Implementation placeholder - will be completed in M5
        unrealized_pnl = self.calculate_unrealized_pnl(market_prices)
        total_unrealized = sum(unrealized_pnl.values())
        
        total_portfolio_value = self.cash_balance + total_unrealized
        
        logger.debug(f"Portfolio value: ${total_portfolio_value:.2f} (cash: ${self.cash_balance:.2f}, unrealized: ${total_unrealized:.2f})")
        return total_portfolio_value
    
    def get_position_summary(self) -> Dict[str, Any]:
        """
        Get summary of all positions and portfolio metrics.
        
        Returns:
            Position summary dictionary
        """
        # Implementation placeholder - will be completed in M5
        active_positions = {k: v for k, v in self.positions.items() if v.position != 0}
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        
        return {
            "cash_balance": self.cash_balance,
            "active_positions": len(active_positions),
            "total_trades": len(self.trade_history),
            "total_realized_pnl": total_realized_pnl,
            "positions": {k: {"position": v.position, "cost_basis": v.cost_basis} 
                         for k, v in active_positions.items()}
        }
    
    def update_from_kalshi_api(self, api_positions: List[Dict[str, Any]]) -> None:
        """
        Sync positions from Kalshi API for inference pipeline.
        
        Args:
            api_positions: Position data from Kalshi API
        """
        # Implementation placeholder - will be completed in M11
        logger.info("Syncing positions from Kalshi API")
        pass
    
    def update_from_websocket(self, fill_message: Dict[str, Any]) -> None:
        """
        Update position from WebSocket fill message.
        
        Args:
            fill_message: Fill notification from Kalshi WebSocket
        """
        # Implementation placeholder - will be completed in M11
        logger.debug("Processing WebSocket fill message")
        pass
    
    def reset(self, initial_cash: Optional[float] = None) -> None:
        """
        Reset tracker for new episode.
        
        Args:
            initial_cash: Optional new starting cash amount
        """
        if initial_cash is not None:
            self.initial_cash = initial_cash
            
        self.cash_balance = self.initial_cash
        self.positions.clear()
        self.trade_history.clear()
        
        logger.info(f"Reset position tracker with ${self.initial_cash:,.2f} cash")


class UnifiedRewardCalculator:
    """
    Unified reward calculation for RL environments.
    
    Implements simple reward = portfolio value change approach.
    This captures all important signals naturally without artificial complexity.
    """
    
    def __init__(self, reward_scale: float = 0.01):
        """
        Initialize reward calculator.
        
        Args:
            reward_scale: Scaling factor for rewards (default: 1% = 0.01)
        """
        self.reward_scale = reward_scale
        self.previous_portfolio_value: Optional[float] = None
        
        logger.info(f"Initialized reward calculator with scale {reward_scale}")
    
    def calculate_reward(
        self,
        current_portfolio_value: float,
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
            current_portfolio_value: Current total portfolio value
            step_info: Optional additional step information
            
        Returns:
            Reward value (can be positive or negative)
        """
        if self.previous_portfolio_value is None:
            # First step - no reward
            self.previous_portfolio_value = current_portfolio_value
            reward = 0.0
        else:
            # Calculate value change
            value_change = current_portfolio_value - self.previous_portfolio_value
            reward = value_change * self.reward_scale
            
            # Update for next step
            self.previous_portfolio_value = current_portfolio_value
        
        logger.debug(f"Calculated reward: {reward:.6f} (portfolio value: ${current_portfolio_value:.2f})")
        return reward
    
    def reset(self, initial_portfolio_value: Optional[float] = None) -> None:
        """
        Reset for new episode.
        
        Args:
            initial_portfolio_value: Starting portfolio value for episode
        """
        self.previous_portfolio_value = initial_portfolio_value
        logger.debug(f"Reset reward calculator (initial value: ${initial_portfolio_value or 'TBD'})")
    
    def get_episode_stats(self) -> Dict[str, float]:
        """
        Get episode-level reward statistics.
        
        Returns:
            Episode reward statistics
        """
        # Implementation placeholder - will be completed in M5
        return {
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "final_portfolio_value": self.previous_portfolio_value or 0.0
        }


def validate_kalshi_position_format(position_data: Dict[str, Any]) -> bool:
    """
    Validate that position data matches Kalshi API format.
    
    Args:
        position_data: Position data to validate
        
    Returns:
        True if format is correct
    """
    # Implementation placeholder - will be completed in M5
    required_fields = ["position", "cost_basis", "realized_pnl"]
    
    for field in required_fields:
        if field not in position_data:
            logger.error(f"Missing required field: {field}")
            return False
            
    return True


def calculate_position_metrics(
    positions: Dict[str, PositionInfo],
    market_prices: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate portfolio-level position metrics.
    
    Args:
        positions: All current positions
        market_prices: Current market prices
        
    Returns:
        Portfolio metrics dictionary
    """
    # Implementation placeholder - will be completed in M5
    metrics = {
        "position_count": len([p for p in positions.values() if p.position != 0]),
        "total_exposure": 0.0,
        "long_exposure": 0.0,
        "short_exposure": 0.0,
        "concentration_risk": 0.0,
        "diversification_score": 1.0
    }
    
    return metrics