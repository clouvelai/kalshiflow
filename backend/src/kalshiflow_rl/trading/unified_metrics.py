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


@dataclass
class PositionInfo:
    """
    Position information for a single market using Kalshi convention.
    
    Follows Kalshi API format:
    - position: +N for YES contracts, -N for NO contracts
    - cost_basis: Total cents spent to acquire position
    - realized_pnl: Cumulative realized profit/loss from trades in cents
    """
    position: int = 0           # +YES contracts / -NO contracts (Kalshi convention)
    cost_basis: int = 0         # Total cost in cents to acquire position
    realized_pnl: int = 0       # Cumulative realized P&L from closed trades in cents
    last_price: float = 50.0    # Last known market price for unrealized P&L (still 0-99)
    

class UnifiedPositionTracker:
    """
    Unified position tracking using Kalshi API conventions.
    
    This tracker works identically for training and inference environments,
    ensuring consistent position management and P&L calculation across all
    use cases. Uses Kalshi's +YES/-NO convention for positions.
    """
    
    def __init__(self, initial_cash: int = 100000):
        """
        Initialize position tracker.
        
        Args:
            initial_cash: Starting cash balance in cents (default 100000 = $1000)
        """
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.positions: Dict[str, PositionInfo] = {}  # market_ticker -> PositionInfo
        self.trade_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized position tracker with ${initial_cash/100:,.2f} cash ({initial_cash:,} cents)")
    
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
        
        PRICE FORMAT: Input price in cents (1-99), all calculations in cents.
        Maintains exact Kalshi API compatibility throughout.
        
        Args:
            market_ticker: Market identifier
            side: "YES" or "NO" 
            quantity: Number of contracts (positive for buy, negative for sell)
            price: Trade price in integer cents (1-99)
            timestamp: Trade timestamp
            
        Returns:
            Trade information dictionary with:
            - price: Original price in cents
            - trade_value: Calculated value in cents (price * quantity)
            - position updates using Kalshi convention (+YES/-NO)
            - realized_pnl: Any realized P&L from the trade in cents
        """
        if market_ticker not in self.positions:
            self.positions[market_ticker] = PositionInfo()
            
        position = self.positions[market_ticker]
        
        # Convert to Kalshi convention (+YES, -NO)
        if side == "YES":
            contracts_delta = quantity
        else:  # side == "NO"
            contracts_delta = -quantity
            
        # Calculate trade value in cents
        trade_value = abs(quantity) * int(price)
        
        old_position = position.position
        realized_pnl = 0
        
        # Handle position updates with proper realized P&L calculation
        if quantity > 0:  # Buying contracts
            self.cash_balance -= trade_value
            
            # When buying, always add to cost basis (both YES and NO contracts cost their price)
            if position.position == 0 or (position.position > 0 and side == "YES") or (position.position < 0 and side == "NO"):
                # Opening new position or adding to existing position in same direction
                position.cost_basis += trade_value
            else:
                # Reducing existing position in opposite direction - realize P&L first
                contracts_closed = min(abs(quantity), abs(position.position))
                if contracts_closed > 0:
                    # Calculate P&L for closed contracts
                    avg_cost_per_contract = position.cost_basis // abs(position.position) if position.position != 0 else 0
                    cost_of_closed = contracts_closed * avg_cost_per_contract
                    
                    # Revenue calculation depends on what position we're closing
                    if position.position > 0:  # Closing YES position by buying NO
                        # YES position value = contracts * current_price
                        # But we're buying NO at current price, so realized value is like selling YES
                        realized_pnl = (contracts_closed * int(price)) - cost_of_closed
                    else:  # Closing NO position by buying YES
                        # NO position profits when price goes down from original purchase
                        realized_pnl = cost_of_closed - (contracts_closed * int(price))
                    
                    position.realized_pnl += realized_pnl
                    
                    # Update cost basis for remaining position
                    remaining_contracts = abs(position.position) - contracts_closed
                    if remaining_contracts > 0:
                        position.cost_basis = remaining_contracts * avg_cost_per_contract
                    else:
                        position.cost_basis = 0
                
                # Add cost basis for any new position in opposite direction
                new_contracts = abs(quantity) - contracts_closed
                if new_contracts > 0:
                    position.cost_basis += new_contracts * int(price)
                    
        else:  # Selling contracts (quantity < 0)
            self.cash_balance += trade_value
            
            # When selling, we're closing existing positions
            if position.position != 0:
                contracts_to_close = min(abs(quantity), abs(position.position))
                
                if (position.position > 0 and side == "YES") or (position.position < 0 and side == "NO"):
                    # Closing existing position - realize P&L
                    avg_cost_per_contract = position.cost_basis // abs(position.position)
                    cost_of_sold = contracts_to_close * avg_cost_per_contract
                    revenue_from_sale = contracts_to_close * int(price)
                    
                    if side == "YES":
                        realized_pnl = revenue_from_sale - cost_of_sold
                    else:  # side == "NO"
                        # For NO contracts, they're worth (100-price) when selling
                        revenue_from_sale = contracts_to_close * (100 - int(price))
                        realized_pnl = revenue_from_sale - cost_of_sold
                    
                    position.realized_pnl += realized_pnl
                    
                    # Update cost basis for remaining position
                    remaining_contracts = abs(position.position) - contracts_to_close
                    if remaining_contracts > 0:
                        position.cost_basis = remaining_contracts * avg_cost_per_contract
                    else:
                        position.cost_basis = 0
        
        # Update position count
        position.position += contracts_delta
        position.last_price = price
        
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
            "cash_after": self.cash_balance,
            "realized_pnl": realized_pnl
        }
        
        self.trade_history.append(trade_info)
        
        logger.debug(f"Trade executed: {market_ticker} {side} {quantity}@{price}¢ | Position: {old_position} -> {position.position} | Realized P&L: {realized_pnl}¢")
        return trade_info
    
    def calculate_unrealized_pnl(self, market_prices: Dict[str, float]) -> Dict[str, int]:
        """
        Calculate unrealized P&L for all positions.
        
        PRICE FORMAT: Input prices in cents, output P&L in cents.
        All calculations in cents for exact Kalshi API compatibility.
        
        Kalshi Position Convention:
        - Positive position = holding YES contracts
        - Negative position = holding NO contracts  
        - YES contracts profit when price increases
        - NO contracts profit when price decreases
        
        Args:
            market_prices: Current market prices {market_ticker: price_cents (1-99)}
            
        Returns:
            Unrealized P&L by market {market_ticker: pnl_cents}
            - Positive values = profit, negative values = loss
            - Values in cents for exact API compatibility
        """
        unrealized_pnl = {}
        
        for market_ticker, position in self.positions.items():
            if position.position == 0:
                unrealized_pnl[market_ticker] = 0
                continue
                
            current_price = market_prices.get(market_ticker, position.last_price)
            
            if position.position > 0:
                # Holding YES contracts - profit when price goes up
                current_value = position.position * int(current_price)
                unrealized_pnl[market_ticker] = current_value - position.cost_basis
            else:
                # Holding NO contracts - profit when price goes down
                # NO contract current value = position_count * (100 - current_price)
                current_value = abs(position.position) * (100 - int(current_price))
                unrealized_pnl[market_ticker] = current_value - position.cost_basis
                
        return unrealized_pnl
    
    def get_total_portfolio_value(self, market_prices: Dict[str, float]) -> int:
        """
        Calculate total portfolio value including cash and positions.
        
        PRICE FORMAT: Input prices in cents, output total value in cents.
        All calculations in cents for exact Kalshi API compatibility.
        
        Args:
            market_prices: Current market prices {market_ticker: price_cents (1-99)}
            
        Returns:
            Total portfolio value in cents:
            - cash_balance (cents) + position_values (cents)
            - Direct input to reward calculator
        """
        unrealized_pnl = self.calculate_unrealized_pnl(market_prices)
        total_unrealized = sum(unrealized_pnl.values())
        
        # Total portfolio = cash + cost_basis + unrealized_pnl
        total_cost_basis = sum(pos.cost_basis for pos in self.positions.values())
        total_portfolio_value = self.cash_balance + total_cost_basis + total_unrealized
        
        logger.debug(f"Portfolio value: {total_portfolio_value}¢ (${total_portfolio_value/100:.2f}) | cash: {self.cash_balance}¢, cost basis: {total_cost_basis}¢, unrealized: {total_unrealized}¢")
        return total_portfolio_value
    
    def get_position_summary(self, market_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Get summary of all positions and portfolio metrics.
        
        Args:
            market_prices: Optional current market prices for unrealized P&L
        
        Returns:
            Comprehensive position summary dictionary
        """
        active_positions = {k: v for k, v in self.positions.items() if v.position != 0}
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_cost_basis = sum(pos.cost_basis for pos in self.positions.values())
        
        summary = {
            "cash_balance": self.cash_balance,
            "active_positions": len(active_positions),
            "total_trades": len(self.trade_history),
            "total_realized_pnl": total_realized_pnl,
            "total_cost_basis": total_cost_basis,
            "positions": {
                k: {
                    "position": v.position,
                    "cost_basis": v.cost_basis,
                    "realized_pnl": v.realized_pnl,
                    "last_price": v.last_price
                } for k, v in active_positions.items()
            }
        }
        
        # Add unrealized P&L and total portfolio value if prices provided
        if market_prices:
            unrealized_pnl = self.calculate_unrealized_pnl(market_prices)
            total_unrealized = sum(unrealized_pnl.values())
            total_portfolio = self.get_total_portfolio_value(market_prices)
            
            summary.update({
                "total_unrealized_pnl": total_unrealized,
                "total_portfolio_value": total_portfolio,
                "total_pnl": total_realized_pnl + total_unrealized,
                "return_pct": ((total_portfolio / self.initial_cash) - 1) * 100 if self.initial_cash > 0 else 0.0,
                "unrealized_pnl_by_market": unrealized_pnl
            })
        
        return summary
    
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
    
    def reset(self, initial_cash: Optional[int] = None) -> None:
        """
        Reset tracker for new episode.
        
        Args:
            initial_cash: Optional new starting cash amount in cents
        """
        if initial_cash is not None:
            self.initial_cash = initial_cash
            
        self.cash_balance = self.initial_cash
        self.positions.clear()
        self.trade_history.clear()
        
        logger.info(f"Reset position tracker with ${self.initial_cash/100:,.2f} cash ({self.initial_cash:,} cents)")


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
        Portfolio metrics dictionary with exposure and risk metrics
    """
    active_positions = [p for p in positions.values() if p.position != 0]
    
    if not active_positions:
        return {
            "position_count": 0,
            "total_exposure": 0.0,
            "long_exposure": 0.0,
            "short_exposure": 0.0,
            "concentration_risk": 0.0,
            "diversification_score": 1.0,
            "avg_position_size": 0.0,
            "largest_position_pct": 0.0
        }
    
    total_exposure = 0.0
    long_exposure = 0.0
    short_exposure = 0.0
    position_values = []
    
    for ticker, position in positions.items():
        if position.position == 0:
            continue
            
        current_price = market_prices.get(ticker, position.last_price)
        
        if position.position > 0:
            # Long YES position
            exposure = position.position * int(current_price)
            long_exposure += exposure
        else:
            # Short NO position
            exposure = abs(position.position) * (100 - int(current_price))
            short_exposure += exposure
            
        position_values.append(exposure)
        total_exposure += exposure
    
    # Calculate concentration metrics
    largest_position = max(position_values) if position_values else 0.0
    largest_position_pct = (largest_position / total_exposure * 100) if total_exposure > 0 else 0.0
    
    # Simple diversification score (decreases with concentration)
    concentration_risk = largest_position_pct / 100.0
    diversification_score = 1.0 - concentration_risk
    
    avg_position_size = total_exposure / len(active_positions) if active_positions else 0.0
    
    return {
        "position_count": len(active_positions),
        "total_exposure": total_exposure,
        "long_exposure": long_exposure,
        "short_exposure": short_exposure,
        "concentration_risk": concentration_risk,
        "diversification_score": diversification_score,
        "avg_position_size": avg_position_size,
        "largest_position_pct": largest_position_pct
    }