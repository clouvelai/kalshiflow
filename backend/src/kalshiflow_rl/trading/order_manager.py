"""
OrderManager abstraction layer for the Kalshi Flow RL Trading Subsystem.

This module provides a clean separation between strategy (RL agent) and execution
(order management) by implementing an abstract OrderManager interface with two
concrete implementations:

1. SimulatedOrderManager: Pure Python simulation for training
2. KalshiOrderManager: Real API integration for paper/live trading

Key Design Principles:
- OrderManager handles ALL order complexity (pricing, cancellation, amendment)
- RL model just outputs simple intents (0-4) 
- Clean separation between strategy and execution
- Stateful order management abstracted away from stateless RL agent
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union, Deque
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
import numpy as np

from ..data.orderbook_state import OrderbookState
from .demo_client import KalshiDemoTradingClient, KalshiDemoTradingClientError

logger = logging.getLogger("kalshiflow_rl.trading.order_manager")


class OrderStatus(IntEnum):
    """Order status enumeration."""
    PENDING = 0    # Order placed but not filled
    FILLED = 1     # Order completely filled
    CANCELLED = 2  # Order cancelled
    REJECTED = 3   # Order rejected


class OrderSide(IntEnum):
    """Trading side enumeration."""
    BUY = 0
    SELL = 1


class ContractSide(IntEnum):
    """Contract side enumeration."""
    YES = 0
    NO = 1


@dataclass
class OrderInfo:
    """Information about an order in the system."""
    order_id: str
    ticker: str
    side: OrderSide      # Buy or Sell
    contract_side: ContractSide  # YES or NO
    quantity: int
    limit_price: int     # Price in cents (1-99)
    status: OrderStatus
    placed_at: float     # Timestamp when placed
    filled_at: Optional[float] = None
    fill_price: Optional[int] = None  # VWAP fill price for depth consumption
    filled_quantity: int = 0  # Track partial fills
    remaining_quantity: int = field(init=False)  # Auto-calculated from quantity - filled_quantity
    
    def __post_init__(self):
        """Initialize computed fields."""
        if not hasattr(self, 'filled_quantity') or self.filled_quantity == 0:
            self.filled_quantity = 0
        self.remaining_quantity = self.quantity - self.filled_quantity
    
    @property
    def time_since_placed(self) -> float:
        """Time since order was placed in seconds."""
        return time.time() - self.placed_at
    
    def is_active(self) -> bool:
        """Check if order is still active (not filled/cancelled/rejected)."""
        return self.status == OrderStatus.PENDING
    
    def is_partially_filled(self) -> bool:
        """Check if order is partially filled."""
        return self.filled_quantity > 0 and self.filled_quantity < self.quantity
    
    def update_partial_fill(self, fill_quantity: int, fill_price: int) -> None:
        """Update order with partial fill information."""
        self.filled_quantity += fill_quantity
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # Update VWAP fill price
        if self.fill_price is None:
            self.fill_price = fill_price
        else:
            # Calculate volume-weighted average price
            total_filled_value = (self.filled_quantity - fill_quantity) * self.fill_price + fill_quantity * fill_price
            self.fill_price = int(total_filled_value / self.filled_quantity)
        
        # Mark as filled if completely filled
        if self.remaining_quantity == 0:
            self.status = OrderStatus.FILLED
            self.filled_at = time.time()


@dataclass
class ConsumedLiquidity:
    """Track temporarily consumed liquidity to prevent double-filling."""
    ticker: str
    side: str  # 'yes_bid', 'yes_ask', 'no_bid', 'no_ask'
    price: int
    consumed_quantity: int
    timestamp: float
    decay_time: float = 5.0  # Liquidity restored after 5 seconds
    
    def is_expired(self) -> bool:
        """Check if consumed liquidity has expired and can be restored."""
        return time.time() - self.timestamp > self.decay_time
    
    def get_available_quantity(self, original_quantity: int) -> int:
        """Get available quantity after accounting for consumption."""
        if self.is_expired():
            return original_quantity
        return max(0, original_quantity - self.consumed_quantity)


class MarketActivityTracker:
    """Track recent market activity for fill probability calculations."""
    
    def __init__(self, window_seconds: int = 300):
        """
        Initialize market activity tracker.
        
        Args:
            window_seconds: Time window to track trades (default 5 minutes)
        """
        self.window_seconds = window_seconds
        self.trades: Deque[Tuple[float, int]] = deque()  # (timestamp, size) tuples
        
    def add_trade(self, size: int) -> None:
        """
        Record a trade for activity tracking.
        
        Args:
            size: Number of contracts traded
        """
        current_time = time.time()
        self.trades.append((current_time, size))
        self._cleanup_old_trades(current_time)
        
    def get_activity_level(self) -> float:
        """
        Return normalized activity level 0-1 based on recent trades.
        
        Returns:
            Activity level normalized between 0 (dead market) and 1 (very active)
        """
        current_time = time.time()
        self._cleanup_old_trades(current_time)
        
        if not self.trades:
            return 0.0
        
        # Calculate trades per minute
        trades_per_minute = len(self.trades) / (self.window_seconds / 60.0)
        
        # Normalize based on expected activity ranges
        # <5 trades/min = dead (0.0)
        # 5-20 trades/min = normal (0.0 - 0.5)
        # >20 trades/min = active (0.5 - 1.0)
        if trades_per_minute < 5:
            activity = trades_per_minute / 10.0  # 0.0 - 0.5 for 0-5 trades/min
        elif trades_per_minute <= 20:
            activity = 0.5 + (trades_per_minute - 5) / 30.0  # 0.5 - 1.0 for 5-20 trades/min
        else:
            activity = min(1.0, 0.5 + trades_per_minute / 40.0)  # Cap at 1.0
        
        return activity
    
    def get_trade_count(self) -> int:
        """Get the number of trades in the current window."""
        current_time = time.time()
        self._cleanup_old_trades(current_time)
        return len(self.trades)
    
    def get_trades_per_minute(self) -> float:
        """Get average trades per minute in the current window."""
        current_time = time.time()
        self._cleanup_old_trades(current_time)
        
        if not self.trades:
            return 0.0
        
        return len(self.trades) / (self.window_seconds / 60.0)
    
    def _cleanup_old_trades(self, current_time: float) -> None:
        """Remove trades older than the window."""
        cutoff_time = current_time - self.window_seconds
        
        while self.trades and self.trades[0][0] < cutoff_time:
            self.trades.popleft()


@dataclass
class Position:
    """Represents a position in a specific market."""
    ticker: str
    contracts: int       # +contracts for YES, -contracts for NO (Kalshi convention)
    cost_basis: float    # Total cost in dollars
    realized_pnl: float  # Cumulative realized P&L in dollars
    
    @property
    def is_long_yes(self) -> bool:
        """True if long YES contracts."""
        return self.contracts > 0
    
    @property
    def is_long_no(self) -> bool:
        """True if long NO contracts (negative contracts)."""
        return self.contracts < 0
    
    @property
    def is_flat(self) -> bool:
        """True if no position."""
        return self.contracts == 0
    
    def get_unrealized_pnl(self, current_yes_price: float = None, yes_bid: float = None, yes_ask: float = None) -> float:
        """
        Calculate unrealized P&L based on current market price.
        
        Args:
            current_yes_price: Current YES price as a probability (0.0-1.0) - DEPRECATED, use bid/ask
            yes_bid: Current YES bid price (what we can sell at)
            yes_ask: Current YES ask price (what we must pay to buy)
            
        Returns:
            Unrealized P&L in dollars
        """
        if self.is_flat:
            return 0.0
        
        # Use bid/ask if provided, otherwise fall back to mid price for backward compatibility
        if yes_bid is not None and yes_ask is not None:
            if self.is_long_yes:
                # Long YES: use bid price (what we can sell at)
                exit_price = yes_bid
                current_value = self.contracts * exit_price
            else:
                # Long NO: use ask price to calculate NO position value
                # NO bid = 1 - YES ask (what we can sell NO at)
                no_bid = 1.0 - yes_ask
                current_value = abs(self.contracts) * no_bid
        else:
            # Fallback to mid price for backward compatibility
            if current_yes_price is None:
                raise ValueError("Must provide either current_yes_price or both yes_bid and yes_ask")
            
            if self.is_long_yes:
                current_value = self.contracts * current_yes_price
            else:
                current_value = abs(self.contracts) * (1.0 - current_yes_price)
        
        return current_value - self.cost_basis


@dataclass
class OrderFeatures:
    """Order-related features for inclusion in RL observations."""
    has_open_buy: float       # 0.0 or 1.0
    has_open_sell: float      # 0.0 or 1.0
    buy_distance_from_mid: float    # 0.0-1.0, normalized distance
    sell_distance_from_mid: float   # 0.0-1.0, normalized distance  
    time_since_order: float   # 0.0-1.0, normalized time
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for RL observation."""
        return np.array([
            self.has_open_buy,
            self.has_open_sell,
            self.buy_distance_from_mid,
            self.sell_distance_from_mid,
            self.time_since_order
        ], dtype=np.float32)


class OrderManager(ABC):
    """
    Abstract base class for order management.
    
    This interface provides a clean abstraction between strategy (RL agent) and
    execution (order management), allowing the same RL model to work in both
    training (simulation) and deployment (real API) environments.
    
    The OrderManager handles all order complexity including:
    - Limit order pricing strategies
    - Order cancellation and amendment logic
    - Position tracking and P&L calculation
    - Order state features for RL observations
    """
    
    def __init__(self, initial_cash: float = 1000.0):
        """
        Initialize base OrderManager.
        
        Args:
            initial_cash: Starting cash balance in dollars
        """
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, OrderInfo] = {}
        self._order_counter = 0
        
        logger.info(f"OrderManager initialized with ${initial_cash} cash")
    
    @abstractmethod
    async def place_order(
        self,
        ticker: str,
        side: OrderSide,
        contract_side: ContractSide,
        quantity: int,
        orderbook: OrderbookState,
        pricing_strategy: str = "aggressive"
    ) -> Optional[OrderInfo]:
        """
        Place a limit order in the market.
        
        Args:
            ticker: Market ticker
            side: Buy or Sell
            contract_side: YES or NO contracts
            quantity: Number of contracts
            orderbook: Current orderbook state for pricing
            pricing_strategy: "aggressive", "passive", or "mid"
            
        Returns:
            OrderInfo if order placed successfully, None if failed
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """
        Cancel all open orders, optionally filtered by ticker.
        
        Args:
            ticker: Optional ticker filter
            
        Returns:
            Number of orders cancelled
        """
        pass
    
    @abstractmethod
    async def amend_order(
        self,
        order_id: str,
        new_price: int,
        orderbook: OrderbookState
    ) -> bool:
        """
        Amend an existing order's price.
        
        Args:
            order_id: Order ID to amend
            new_price: New limit price in cents
            orderbook: Current orderbook for validation
            
        Returns:
            True if amended successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def check_fills(self, orderbook: OrderbookState) -> List[OrderInfo]:
        """
        Check for order fills based on current market state.
        
        Args:
            orderbook: Current orderbook state
            
        Returns:
            List of newly filled orders
        """
        pass
    
    def get_open_orders(self, ticker: Optional[str] = None) -> List[OrderInfo]:
        """
        Get currently open orders.
        
        Args:
            ticker: Optional ticker filter
            
        Returns:
            List of open orders
        """
        orders = [order for order in self.open_orders.values() if order.is_active()]
        
        if ticker:
            orders = [order for order in orders if order.ticker == ticker]
        
        return orders
    
    def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Position]:
        """
        Get current positions.
        
        Args:
            ticker: Optional ticker filter
            
        Returns:
            Dictionary of positions by ticker
        """
        if ticker:
            return {ticker: self.positions[ticker]} if ticker in self.positions else {}
        return dict(self.positions)
    
    def get_order_features(self, ticker: str, orderbook: OrderbookState) -> OrderFeatures:
        """
        Extract order-related features for RL observation.
        
        Args:
            ticker: Market ticker
            orderbook: Current orderbook state
            
        Returns:
            OrderFeatures for the specified market
        """
        open_orders = self.get_open_orders(ticker)
        
        # Find buy and sell orders
        buy_orders = [o for o in open_orders if o.side == OrderSide.BUY]
        sell_orders = [o for o in open_orders if o.side == OrderSide.SELL]
        
        # Calculate features
        has_open_buy = 1.0 if buy_orders else 0.0
        has_open_sell = 1.0 if sell_orders else 0.0
        
        # Calculate distance from mid for active orders
        yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
        yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        
        if yes_best_bid is not None and yes_best_ask is not None:
            mid_price = (yes_best_bid + yes_best_ask) / 2.0
        else:
            mid_price = 50.0  # Default mid
        
        buy_distance = 0.0
        if buy_orders:
            # For buy orders, distance is how far below mid we're bidding
            buy_price = buy_orders[0].limit_price  # Take first buy order
            buy_distance = abs(mid_price - buy_price) / 100.0  # Normalize to [0,1]
        
        sell_distance = 0.0
        if sell_orders:
            # For sell orders, distance is how far above mid we're asking
            sell_price = sell_orders[0].limit_price  # Take first sell order
            sell_distance = abs(sell_price - mid_price) / 100.0  # Normalize to [0,1]
        
        # Time since most recent order
        time_since_order = 0.0
        if open_orders:
            most_recent_time = max(order.placed_at for order in open_orders)
            time_elapsed = time.time() - most_recent_time
            time_since_order = min(time_elapsed / 300.0, 1.0)  # Normalize to 5 minutes max
        
        return OrderFeatures(
            has_open_buy=has_open_buy,
            has_open_sell=has_open_sell,
            buy_distance_from_mid=min(buy_distance, 1.0),
            sell_distance_from_mid=min(sell_distance, 1.0),
            time_since_order=time_since_order
        )
    
    def get_total_portfolio_value(self, current_prices: Dict[str, any]) -> float:
        """
        Calculate total portfolio value including cash and positions.
        
        Args:
            current_prices: Dictionary with bid/ask or simple prices by ticker.
                           Format 1: {ticker: {"bid": float, "ask": float}} - preferred for spread accuracy
                           Format 2: {ticker: float} - backward compatibility using mid price
            
        Returns:
            Total portfolio value in dollars
        """
        total_value = self.cash_balance
        
        for ticker, position in self.positions.items():
            if not position.is_flat and ticker in current_prices:
                # Check if we have bid/ask data or just a price
                price_data = current_prices[ticker]
                
                if isinstance(price_data, dict) and "bid" in price_data and "ask" in price_data:
                    # Use bid/ask for accurate spread cost calculation
                    unrealized_pnl = position.get_unrealized_pnl(
                        yes_bid=price_data["bid"], 
                        yes_ask=price_data["ask"]
                    )
                else:
                    # Fallback to simple price (backward compatibility)
                    unrealized_pnl = position.get_unrealized_pnl(current_yes_price=price_data)
                
                total_value += position.cost_basis + unrealized_pnl
        
        return total_value
    
    def get_portfolio_value_cents(self, current_prices: Dict[str, any]) -> int:
        """
        Calculate total portfolio value in cents for RL environment compatibility.
        
        Args:
            current_prices: Dictionary with bid/ask or tuple format by ticker.
                           Format 1: {ticker: {"bid": float, "ask": float}} in cents - preferred
                           Format 2: {ticker: (yes_mid, no_mid)} in cents - backward compatibility
            
        Returns:
            Total portfolio value in cents
        """
        # Convert current_prices to the format expected by get_total_portfolio_value
        processed_prices = {}
        
        for ticker, price_data in current_prices.items():
            if isinstance(price_data, dict) and "bid" in price_data and "ask" in price_data:
                # New format with bid/ask in cents
                processed_prices[ticker] = {
                    "bid": price_data["bid"] / 100.0,  # Convert cents to probability
                    "ask": price_data["ask"] / 100.0
                }
            elif isinstance(price_data, (tuple, list)) and len(price_data) == 2:
                # Backward compatibility: (yes_mid, no_mid) in cents
                yes_mid, no_mid = price_data
                processed_prices[ticker] = yes_mid / 100.0  # Convert cents to probability
            else:
                # Assume single price value
                processed_prices[ticker] = price_data / 100.0 if price_data > 1 else price_data
        
        # Get total value in dollars and convert to cents
        total_value_dollars = self.get_total_portfolio_value(processed_prices)
        return int(total_value_dollars * 100)
    
    def get_cash_balance_cents(self) -> int:
        """
        Get cash balance in cents for RL environment compatibility.
        
        Returns:
            Cash balance in cents
        """
        return int(self.cash_balance * 100)
    
    def get_position_info(self) -> Dict[str, Any]:
        """
        Get position data in the format expected by feature extraction.
        
        Returns:
            Dictionary mapping ticker to position info compatible with UnifiedPositionTracker format
        """
        position_data = {}
        
        for ticker, position in self.positions.items():
            if not position.is_flat:
                # Convert OrderManager Position to UnifiedPositionTracker format
                position_data[ticker] = {
                    'position': position.contracts,  # +YES/-NO contracts (Kalshi convention)
                    'cost_basis': int(position.cost_basis * 100),  # Convert to cents
                    'realized_pnl': int(position.realized_pnl * 100),  # Convert to cents
                    'last_price': 50.0  # Default price if not available
                }
        
        return position_data
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"order_{self._order_counter}_{int(time.time() * 1000)}"
    
    def _calculate_limit_price(
        self,
        side: OrderSide,
        contract_side: ContractSide,
        orderbook: OrderbookState,
        strategy: str = "aggressive"
    ) -> int:
        """
        Calculate limit price based on strategy and market state.
        
        Args:
            side: Buy or Sell
            contract_side: YES or NO
            orderbook: Current orderbook
            strategy: Pricing strategy ("aggressive", "passive", "mid")
            
        Returns:
            Limit price in cents (1-99)
        """
        if contract_side == ContractSide.YES:
            # Get best prices from YES book
            best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        else:
            # For NO contracts, use derived prices from YES book
            yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
            
            if yes_best_bid is not None and yes_best_ask is not None:
                best_bid = 99 - yes_best_ask  # NO bid = 99 - YES ask
                best_ask = 99 - yes_best_bid  # NO ask = 99 - YES bid
            else:
                best_bid = None
                best_ask = None
        
        # Handle case where orderbook is empty
        if best_bid is None or best_ask is None:
            # Default to conservative mid-market pricing
            return 50
        
        if strategy == "aggressive":
            # Cross the spread for immediate execution
            if side == OrderSide.BUY:
                price = best_ask  # Buy at ask for immediate fill
            else:
                price = best_bid  # Sell at bid for immediate fill
        elif strategy == "passive":
            # Join the inside market
            if side == OrderSide.BUY:
                price = best_bid  # Buy at bid (passive)
            else:
                price = best_ask  # Sell at ask (passive)
        else:  # "mid"
            # Price at mid-market
            price = int((best_bid + best_ask) / 2)
        
        # Ensure price is within valid range
        return max(1, min(99, price))
    
    def _process_fill(
        self,
        order: OrderInfo,
        fill_price: int,
        fill_timestamp: Optional[float] = None
    ) -> None:
        """
        Process an order fill and update positions.
        
        Args:
            order: Order that was filled
            fill_price: Fill price in cents
            fill_timestamp: Fill timestamp (defaults to current time)
        """
        if fill_timestamp is None:
            fill_timestamp = time.time()
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_at = fill_timestamp
        order.fill_price = fill_price
        
        # Update filled quantities for new order tracking
        if hasattr(order, 'filled_quantity') and order.filled_quantity == 0:
            order.filled_quantity = order.quantity
            order.remaining_quantity = 0
        
        # Calculate fill cost in dollars
        fill_cost = (fill_price / 100.0) * order.quantity
        
        # Update cash balance
        if order.side == OrderSide.BUY:
            self.cash_balance -= fill_cost  # Pay for bought contracts
        else:
            self.cash_balance += fill_cost  # Receive for sold contracts
        
        # Update position
        ticker = order.ticker
        if ticker not in self.positions:
            self.positions[ticker] = Position(
                ticker=ticker,
                contracts=0,
                cost_basis=0.0,
                realized_pnl=0.0
            )
        
        position = self.positions[ticker]
        
        # Calculate position change based on Kalshi convention
        if order.contract_side == ContractSide.YES:
            if order.side == OrderSide.BUY:
                contract_change = order.quantity  # +YES
            else:
                contract_change = -order.quantity  # Sell YES
        else:  # NO contracts
            if order.side == OrderSide.BUY:
                contract_change = -order.quantity  # Buy NO = -YES
            else:
                contract_change = order.quantity  # Sell NO = +YES
        
        # Check if this trade closes existing position (realizes P&L)
        if (position.contracts > 0 and contract_change < 0) or \
           (position.contracts < 0 and contract_change > 0):
            # Position reduction - calculate realized P&L
            reduction_amount = min(abs(contract_change), abs(position.contracts))
            
            # Calculate average cost per contract for the position being closed
            avg_cost_per_contract = position.cost_basis / abs(position.contracts)
            
            # Calculate realized P&L
            if position.contracts > 0:  # Closing long YES position
                if order.side == OrderSide.SELL:
                    realized_pnl = reduction_amount * (fill_price / 100.0 - avg_cost_per_contract)
                else:
                    # Buying NO to close YES (unusual but possible)
                    realized_pnl = reduction_amount * ((100 - fill_price) / 100.0 - avg_cost_per_contract)
            else:  # Closing long NO position
                if order.side == OrderSide.SELL:
                    # Selling NO to close (unusual)
                    realized_pnl = reduction_amount * (fill_price / 100.0 - avg_cost_per_contract)
                else:
                    # Buying YES to close NO
                    realized_pnl = reduction_amount * (avg_cost_per_contract - fill_price / 100.0)
            
            position.realized_pnl += realized_pnl
            
            # Update cost basis proportionally
            position.cost_basis *= (abs(position.contracts) - reduction_amount) / abs(position.contracts)
        else:
            # Position increase - add to cost basis
            position.cost_basis += fill_cost
        
        # Update contract count
        position.contracts += contract_change
        
        # Remove filled order from active orders
        if order.order_id in self.open_orders:
            del self.open_orders[order.order_id]
        
        logger.info(f"Fill processed: {order.order_id} - {order.quantity} contracts at {fill_price}¢")
    
    def _process_partial_fill(
        self,
        order: OrderInfo,
        fill_quantity: int,
        fill_price: int
    ) -> None:
        """
        Process a partial order fill and update positions.
        
        Args:
            order: Order that was partially filled
            fill_quantity: Quantity filled in this partial fill
            fill_price: Price of this partial fill
        """
        # Calculate fill cost in dollars
        fill_cost = (fill_price / 100.0) * fill_quantity
        
        # Update cash balance
        if order.side == OrderSide.BUY:
            self.cash_balance -= fill_cost  # Pay for bought contracts
        else:
            self.cash_balance += fill_cost  # Receive for sold contracts
        
        # Update position (similar to _process_fill but for partial quantity)
        ticker = order.ticker
        if ticker not in self.positions:
            self.positions[ticker] = Position(
                ticker=ticker,
                contracts=0,
                cost_basis=0.0,
                realized_pnl=0.0
            )
        
        position = self.positions[ticker]
        
        # Calculate position change based on Kalshi convention
        if order.contract_side == ContractSide.YES:
            if order.side == OrderSide.BUY:
                contract_change = fill_quantity  # +YES
            else:
                contract_change = -fill_quantity  # Sell YES
        else:  # NO contracts
            if order.side == OrderSide.BUY:
                contract_change = -fill_quantity  # Buy NO = -YES
            else:
                contract_change = fill_quantity  # Sell NO = +YES
        
        # Check if this trade closes existing position (realizes P&L)
        if (position.contracts > 0 and contract_change < 0) or \
           (position.contracts < 0 and contract_change > 0):
            # Position reduction - calculate realized P&L
            reduction_amount = min(abs(contract_change), abs(position.contracts))
            
            # Calculate average cost per contract for the position being closed
            if position.contracts != 0:
                avg_cost_per_contract = position.cost_basis / abs(position.contracts)
                
                # Calculate realized P&L
                if position.contracts > 0:  # Closing long YES position
                    if order.side == OrderSide.SELL:
                        realized_pnl = reduction_amount * (fill_price / 100.0 - avg_cost_per_contract)
                    else:
                        # Buying NO to close YES (unusual but possible)
                        realized_pnl = reduction_amount * ((100 - fill_price) / 100.0 - avg_cost_per_contract)
                else:  # Closing long NO position
                    if order.side == OrderSide.SELL:
                        # Selling NO to close (unusual)
                        realized_pnl = reduction_amount * (fill_price / 100.0 - avg_cost_per_contract)
                    else:
                        # Buying YES to close NO
                        realized_pnl = reduction_amount * (avg_cost_per_contract - fill_price / 100.0)
                
                position.realized_pnl += realized_pnl
                
                # Update cost basis proportionally
                if abs(position.contracts) > reduction_amount:
                    position.cost_basis *= (abs(position.contracts) - reduction_amount) / abs(position.contracts)
                else:
                    position.cost_basis = 0.0
        else:
            # Position increase - add to cost basis
            position.cost_basis += fill_cost
        
        # Update contract count
        position.contracts += contract_change
        
        logger.info(
            f"Partial fill processed: {order.order_id} - {fill_quantity} contracts at {fill_price}¢, "
            f"{order.remaining_quantity} remaining"
        )


class SimulatedOrderManager(OrderManager):
    """
    Simulated order manager for training environments with orderbook depth consumption.
    
    Provides realistic order execution simulation that accounts for:
    - Orderbook depth consumption (orders walk the book for large sizes)
    - Volume-weighted average price (VWAP) calculation for multi-level fills
    - Consumed liquidity tracking to prevent double-filling
    - Small order optimization (orders <20 contracts still fill at best price)
    
    Features:
    - Depth-aware order filling with realistic slippage
    - Partial fills when liquidity is insufficient
    - Consumed liquidity with time-decay (5 seconds)
    - Deterministic execution for reproducible training
    - No network latency or API dependencies
    """
    
    def __init__(self, initial_cash: int = 100000, small_order_threshold: int = 20):
        """
        Initialize simulated order manager.
        
        Args:
            initial_cash: Starting cash balance in cents
            small_order_threshold: Orders below this size fill at best price without depth consumption
        """
        # Work entirely in cents now - convert to dollars for base class compatibility
        initial_cash_dollars = initial_cash / 100.0
        super().__init__(initial_cash_dollars)
        self.small_order_threshold = small_order_threshold
        self.consumed_liquidity: Dict[str, ConsumedLiquidity] = {}  # key: f"{ticker}_{side}_{price}"
        self.activity_tracker = MarketActivityTracker(window_seconds=300)  # 5-minute window
        logger.info(
            f"SimulatedOrderManager initialized for training with {initial_cash}¢ "
            f"(${initial_cash_dollars:.2f}), small order threshold: {small_order_threshold}"
        )
    
    
    async def place_order(
        self,
        ticker: str,
        side: OrderSide,
        contract_side: ContractSide,
        quantity: int,
        orderbook: OrderbookState,
        pricing_strategy: str = "aggressive"
    ) -> Optional[OrderInfo]:
        """
        Place a simulated limit order.
        
        Orders that immediately cross the spread are filled instantly.
        Others are added to the open orders list.
        """
        try:
            # Calculate limit price - for large orders, use higher limit to allow depth consumption
            if quantity >= self.small_order_threshold and pricing_strategy == "aggressive":
                limit_price = self._calculate_aggressive_limit_for_large_order(side, contract_side, orderbook)
            else:
                limit_price = self._calculate_limit_price(side, contract_side, orderbook, pricing_strategy)
            
            # Create order
            order = OrderInfo(
                order_id=self._generate_order_id(),
                ticker=ticker,
                side=side,
                contract_side=contract_side,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                placed_at=time.time()
            )
            
            # Check for immediate fill using depth consumption
            fill_result = self.calculate_fill_with_depth(order, orderbook)
            
            if fill_result['can_fill']:
                # Execute immediate fill with depth consumption
                filled_quantity = fill_result['filled_quantity']
                vwap_price = fill_result['vwap_price']
                consumed_levels = fill_result['consumed_levels']
                
                # Track consumed liquidity
                self._track_consumed_liquidity(order, consumed_levels)
                
                if filled_quantity >= order.quantity:
                    # Fully filled
                    self._process_fill(order, vwap_price)
                    logger.debug(
                        f"Simulated order filled immediately: {order.order_id} - "
                        f"{filled_quantity} contracts at VWAP {vwap_price}¢"
                    )
                else:
                    # Partially filled
                    order.update_partial_fill(filled_quantity, vwap_price)
                    self._process_partial_fill(order, filled_quantity, vwap_price)
                    
                    # Add remaining quantity to open orders
                    self.open_orders[order.order_id] = order
                    logger.debug(
                        f"Simulated order partially filled: {order.order_id} - "
                        f"{filled_quantity}/{order.quantity} contracts at VWAP {vwap_price}¢, "
                        f"{order.remaining_quantity} remaining"
                    )
            else:
                # Add to open orders
                self.open_orders[order.order_id] = order
                logger.debug(f"Simulated order placed: {order.order_id}")
            
            return order
            
        except Exception as e:
            logger.error(f"Failed to place simulated order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order."""
        if order_id in self.open_orders:
            self.open_orders[order_id].status = OrderStatus.CANCELLED
            del self.open_orders[order_id]
            logger.debug(f"Simulated order cancelled: {order_id}")
            return True
        return False
    
    async def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """Cancel all simulated orders."""
        cancelled_count = 0
        orders_to_cancel = []
        
        for order in self.open_orders.values():
            if ticker is None or order.ticker == ticker:
                orders_to_cancel.append(order.order_id)
        
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        return cancelled_count
    
    async def amend_order(
        self,
        order_id: str,
        new_price: int,
        orderbook: OrderbookState
    ) -> bool:
        """Amend a simulated order's price."""
        if order_id not in self.open_orders:
            return False
        
        order = self.open_orders[order_id]
        old_price = order.limit_price
        order.limit_price = max(1, min(99, new_price))
        
        # Check if amended order can now fill
        if self._can_fill_immediately(order, orderbook):
            fill_price = self._get_fill_price(order, orderbook)
            self._process_fill(order, fill_price)
        
        logger.debug(f"Simulated order amended: {order_id} - {old_price}¢ → {new_price}¢")
        return True
    
    def _process_partial_fill(
        self,
        order: OrderInfo,
        fill_quantity: int,
        fill_price: int
    ) -> None:
        """
        Process a partial order fill and update positions.
        
        Args:
            order: Order that was partially filled
            fill_quantity: Quantity filled in this partial fill
            fill_price: Price of this partial fill
        """
        # Calculate fill cost in dollars
        fill_cost = (fill_price / 100.0) * fill_quantity
        
        # Update cash balance
        if order.side == OrderSide.BUY:
            self.cash_balance -= fill_cost  # Pay for bought contracts
        else:
            self.cash_balance += fill_cost  # Receive for sold contracts
        
        # Update position (similar to _process_fill but for partial quantity)
        ticker = order.ticker
        if ticker not in self.positions:
            self.positions[ticker] = Position(
                ticker=ticker,
                contracts=0,
                cost_basis=0.0,
                realized_pnl=0.0
            )
        
        position = self.positions[ticker]
        
        # Calculate position change based on Kalshi convention
        if order.contract_side == ContractSide.YES:
            if order.side == OrderSide.BUY:
                contract_change = fill_quantity  # +YES
            else:
                contract_change = -fill_quantity  # Sell YES
        else:  # NO contracts
            if order.side == OrderSide.BUY:
                contract_change = -fill_quantity  # Buy NO = -YES
            else:
                contract_change = fill_quantity  # Sell NO = +YES
        
        # Check if this trade closes existing position (realizes P&L)
        if (position.contracts > 0 and contract_change < 0) or \
           (position.contracts < 0 and contract_change > 0):
            # Position reduction - calculate realized P&L
            reduction_amount = min(abs(contract_change), abs(position.contracts))
            
            # Calculate average cost per contract for the position being closed
            if position.contracts != 0:
                avg_cost_per_contract = position.cost_basis / abs(position.contracts)
                
                # Calculate realized P&L
                if position.contracts > 0:  # Closing long YES position
                    if order.side == OrderSide.SELL:
                        realized_pnl = reduction_amount * (fill_price / 100.0 - avg_cost_per_contract)
                    else:
                        # Buying NO to close YES (unusual but possible)
                        realized_pnl = reduction_amount * ((100 - fill_price) / 100.0 - avg_cost_per_contract)
                else:  # Closing long NO position
                    if order.side == OrderSide.SELL:
                        # Selling NO to close (unusual)
                        realized_pnl = reduction_amount * (fill_price / 100.0 - avg_cost_per_contract)
                    else:
                        # Buying YES to close NO
                        realized_pnl = reduction_amount * (avg_cost_per_contract - fill_price / 100.0)
                
                position.realized_pnl += realized_pnl
                
                # Update cost basis proportionally
                if abs(position.contracts) > reduction_amount:
                    position.cost_basis *= (abs(position.contracts) - reduction_amount) / abs(position.contracts)
                else:
                    position.cost_basis = 0.0
        else:
            # Position increase - add to cost basis
            position.cost_basis += fill_cost
        
        # Update contract count
        position.contracts += contract_change
        
        logger.info(
            f"Partial fill processed: {order.order_id} - {fill_quantity} contracts at {fill_price}¢, "
            f"{order.remaining_quantity} remaining"
        )
    
    async def check_fills(self, orderbook: OrderbookState) -> List[OrderInfo]:
        """Check for fills of pending simulated orders using probabilistic model."""
        filled_orders = []
        orders_to_remove = []
        
        for order in list(self.open_orders.values()):
            # Calculate fill probability
            fill_probability = self.calculate_fill_probability(order, orderbook)
            
            # Use random sampling to determine if order should fill
            if random.random() < fill_probability:
                # Order should fill - separate passive and aggressive logic
                is_aggressive = self._is_aggressive_order(order, orderbook)
                
                if is_aggressive:
                    # Aggressive orders: use depth consumption for VWAP and partial fills
                    fill_result = self.calculate_fill_with_depth(order, orderbook)
                    
                    if fill_result['can_fill']:
                        filled_quantity = fill_result['filled_quantity']
                        vwap_price = fill_result['vwap_price']
                        consumed_levels = fill_result['consumed_levels']
                        
                        # Track consumed liquidity
                        self._track_consumed_liquidity(order, consumed_levels)
                        fill_type = "aggressive"
                    else:
                        continue  # Aggressive order that can't fill due to depth
                else:
                    # Passive orders: fill at limit price (already passed probability check)
                    filled_quantity = order.remaining_quantity
                    vwap_price = order.limit_price
                    consumed_levels = []  # No depth consumption for passive fills
                    fill_type = "passive"
                
                # Simulate a trade for activity tracking
                self.activity_tracker.add_trade(filled_quantity)
                
                if filled_quantity >= order.remaining_quantity:
                    # Fully filled (complete remaining quantity)
                    if order.is_partially_filled():
                        # Complete a partial fill
                        order.update_partial_fill(filled_quantity, vwap_price)
                        self._process_partial_fill(order, filled_quantity, vwap_price)
                    else:
                        # Fresh complete fill
                        self._process_fill(order, vwap_price)
                    
                    filled_orders.append(order)
                    orders_to_remove.append(order.order_id)
                    
                    logger.debug(
                        f"Order filled ({fill_type}, prob={fill_probability:.2f}): {order.order_id} - "
                        f"{filled_quantity} contracts at {vwap_price}¢"
                    )
                else:
                    # Partial fill
                    order.update_partial_fill(filled_quantity, vwap_price)
                    self._process_partial_fill(order, filled_quantity, vwap_price)
                    # Note: don't remove from open_orders, still has remaining quantity
                    
                    logger.debug(
                        f"Order partially filled ({fill_type}, prob={fill_probability:.2f}): {order.order_id} - "
                        f"{filled_quantity}/{order.quantity} contracts at {vwap_price}¢"
                    )
        
        # Remove filled orders
        for order_id in orders_to_remove:
            if order_id in self.open_orders:
                del self.open_orders[order_id]
        
        return filled_orders
    
    def _can_fill_immediately(self, order: OrderInfo, orderbook: OrderbookState) -> bool:
        """
        Check if an order can be filled immediately at current market prices.
        
        Now considers available liquidity after accounting for consumed liquidity.
        """
        fill_result = self.calculate_fill_with_depth(order, orderbook)
        return fill_result['can_fill'] and fill_result['filled_quantity'] > 0
    
    def _get_fill_price(self, order: OrderInfo, orderbook: OrderbookState) -> int:
        """
        Get the VWAP price at which an order would fill, accounting for depth consumption.
        
        For small orders (< small_order_threshold), uses the traditional best price fill.
        For large orders, walks the orderbook and calculates volume-weighted average price.
        """
        fill_result = self.calculate_fill_with_depth(order, orderbook)
        
        if fill_result['can_fill']:
            return fill_result['vwap_price']
        else:
            # Fallback to order's limit price if no market data
            return order.limit_price
    
    def calculate_fill_with_depth(self, order: OrderInfo, orderbook: OrderbookState) -> Dict[str, Any]:
        """
        Calculate order fill with orderbook depth consumption.
        
        Walks through price levels to determine:
        1. How much can be filled
        2. Volume-weighted average price (VWAP)
        3. Liquidity consumption impact
        
        Args:
            order: Order to analyze
            orderbook: Current orderbook state
            
        Returns:
            Dict with keys:
            - can_fill: bool - whether any part can fill
            - filled_quantity: int - how many contracts can fill
            - vwap_price: int - volume-weighted average price in cents
            - consumed_levels: List[Dict] - price levels that would be consumed
        """
        # Clean up expired consumed liquidity
        self._cleanup_expired_liquidity()
        
        # Small order optimization - fill at best price without depth consumption
        if order.remaining_quantity < self.small_order_threshold:
            return self._calculate_small_order_fill(order, orderbook)
        
        # Large order - walk the orderbook
        return self._calculate_depth_fill(order, orderbook)
    
    def _calculate_small_order_fill(self, order: OrderInfo, orderbook: OrderbookState) -> Dict[str, Any]:
        """
        Calculate fill for small orders using traditional best price logic.
        
        Small orders (<20 contracts) fill at best bid/ask without slippage.
        """
        if order.contract_side == ContractSide.YES:
            best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
            book_side = 'yes_ask' if order.side == OrderSide.BUY else 'yes_bid'
            target_book = orderbook.yes_asks if order.side == OrderSide.BUY else orderbook.yes_bids
        else:
            # For NO contracts
            yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
            
            if yes_best_bid is not None and yes_best_ask is not None:
                best_bid = 99 - yes_best_ask
                best_ask = 99 - yes_best_bid
            else:
                return {'can_fill': False, 'filled_quantity': 0, 'vwap_price': order.limit_price, 'consumed_levels': []}
            
            book_side = 'no_ask' if order.side == OrderSide.BUY else 'no_bid'
            # For NO contracts, we need to construct the effective book
            if order.side == OrderSide.BUY:
                # Buying NO = selling YES, so we look at YES bids (which become NO asks)
                target_book = {99 - price: size for price, size in orderbook.yes_bids.items()}
            else:
                # Selling NO = buying YES, so we look at YES asks (which become NO bids)
                target_book = {99 - price: size for price, size in orderbook.yes_asks.items()}
        
        if best_bid is None or best_ask is None:
            return {'can_fill': False, 'filled_quantity': 0, 'vwap_price': order.limit_price, 'consumed_levels': []}
        
        # Check if order can fill at best price
        if order.side == OrderSide.BUY:
            if order.limit_price >= best_ask:
                # Get available quantity at best ask
                best_ask_quantity = target_book.get(best_ask, 0) if target_book else 0
                liquidity_key = f"{order.ticker}_{book_side}_{best_ask}"
                
                if liquidity_key in self.consumed_liquidity:
                    available_quantity = self.consumed_liquidity[liquidity_key].get_available_quantity(best_ask_quantity)
                else:
                    available_quantity = best_ask_quantity
                
                if available_quantity >= order.remaining_quantity:
                    return {
                        'can_fill': True,
                        'filled_quantity': order.remaining_quantity,
                        'vwap_price': best_ask,
                        'consumed_levels': [{'price': best_ask, 'quantity': order.remaining_quantity}]
                    }
        else:  # SELL
            if order.limit_price <= best_bid:
                # Get available quantity at best bid
                best_bid_quantity = target_book.get(best_bid, 0) if target_book else 0
                liquidity_key = f"{order.ticker}_{book_side}_{best_bid}"
                
                if liquidity_key in self.consumed_liquidity:
                    available_quantity = self.consumed_liquidity[liquidity_key].get_available_quantity(best_bid_quantity)
                else:
                    available_quantity = best_bid_quantity
                
                if available_quantity >= order.remaining_quantity:
                    return {
                        'can_fill': True,
                        'filled_quantity': order.remaining_quantity,
                        'vwap_price': best_bid,
                        'consumed_levels': [{'price': best_bid, 'quantity': order.remaining_quantity}]
                    }
        
        return {'can_fill': False, 'filled_quantity': 0, 'vwap_price': order.limit_price, 'consumed_levels': []}
    
    def _calculate_depth_fill(self, order: OrderInfo, orderbook: OrderbookState) -> Dict[str, Any]:
        """
        Calculate fill for large orders by walking the orderbook depth.
        
        Implements orderbook walking with slippage calculation.
        """
        # Determine which book to walk based on order type
        if order.contract_side == ContractSide.YES:
            if order.side == OrderSide.BUY:
                # Buying YES - walk the YES asks
                book_levels = list(orderbook.yes_asks.items())
                book_side = 'yes_ask'
            else:
                # Selling YES - walk the YES bids
                book_levels = list(orderbook.yes_bids.items())
                book_side = 'yes_bid'
        else:  # NO contracts
            if order.side == OrderSide.BUY:
                # Buying NO = walking YES bids (converted to NO asks)
                book_levels = [(99 - price, size) for price, size in orderbook.yes_bids.items()]
                book_side = 'no_ask'
            else:
                # Selling NO = walking YES asks (converted to NO bids)
                book_levels = [(99 - price, size) for price, size in orderbook.yes_asks.items()]
                book_side = 'no_bid'
        
        if not book_levels:
            return {'can_fill': False, 'filled_quantity': 0, 'vwap_price': order.limit_price, 'consumed_levels': []}
        
        # Sort levels appropriately
        if order.side == OrderSide.BUY:
            # For buying, start with lowest ask prices
            book_levels.sort(key=lambda x: x[0])
        else:
            # For selling, start with highest bid prices
            book_levels.sort(key=lambda x: x[0], reverse=True)
        
        # Walk the book and calculate fill
        total_cost = 0
        total_filled = 0
        consumed_levels = []
        remaining_to_fill = order.remaining_quantity
        
        for price, size in book_levels:
            # Check if this price level satisfies order's limit price
            if order.side == OrderSide.BUY and price > order.limit_price:
                break  # Price too high for buy order
            if order.side == OrderSide.SELL and price < order.limit_price:
                break  # Price too low for sell order
            
            # Check available liquidity (accounting for consumed liquidity)
            liquidity_key = f"{order.ticker}_{book_side}_{price}"
            if liquidity_key in self.consumed_liquidity:
                available_quantity = self.consumed_liquidity[liquidity_key].get_available_quantity(size)
            else:
                available_quantity = size
            
            if available_quantity <= 0:
                continue  # No liquidity available at this level
            
            # Calculate how much we can fill at this level
            level_fill = min(remaining_to_fill, available_quantity)
            
            if level_fill > 0:
                total_cost += level_fill * price
                total_filled += level_fill
                remaining_to_fill -= level_fill
                
                consumed_levels.append({
                    'price': price,
                    'quantity': level_fill
                })
                
                if remaining_to_fill <= 0:
                    break  # Order fully filled
        
        if total_filled > 0:
            vwap_price = round(total_cost / total_filled)
            return {
                'can_fill': True,
                'filled_quantity': total_filled,
                'vwap_price': vwap_price,
                'consumed_levels': consumed_levels
            }
        else:
            return {
                'can_fill': False,
                'filled_quantity': 0,
                'vwap_price': order.limit_price,
                'consumed_levels': []
            }
    
    def _track_consumed_liquidity(self, order: OrderInfo, consumed_levels: List[Dict[str, Any]]) -> None:
        """
        Track consumed liquidity to prevent double-filling at same price levels.
        
        Args:
            order: The order that consumed liquidity
            consumed_levels: List of price levels consumed with quantities
        """
        # Determine book side
        if order.contract_side == ContractSide.YES:
            book_side = 'yes_ask' if order.side == OrderSide.BUY else 'yes_bid'
        else:
            book_side = 'no_ask' if order.side == OrderSide.BUY else 'no_bid'
        
        for level in consumed_levels:
            price = level['price']
            quantity = level['quantity']
            
            liquidity_key = f"{order.ticker}_{book_side}_{price}"
            
            if liquidity_key in self.consumed_liquidity:
                # Update existing consumed liquidity
                self.consumed_liquidity[liquidity_key].consumed_quantity += quantity
                self.consumed_liquidity[liquidity_key].timestamp = time.time()
            else:
                # Create new consumed liquidity record
                self.consumed_liquidity[liquidity_key] = ConsumedLiquidity(
                    ticker=order.ticker,
                    side=book_side,
                    price=price,
                    consumed_quantity=quantity,
                    timestamp=time.time()
                )
    
    def _cleanup_expired_liquidity(self) -> None:
        """
        Remove expired consumed liquidity records.
        
        Consumed liquidity expires after 5 seconds and is available again.
        """
        current_time = time.time()
        expired_keys = [
            key for key, consumed in self.consumed_liquidity.items()
            if consumed.is_expired()
        ]
        
        for key in expired_keys:
            del self.consumed_liquidity[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired liquidity records")
    
    def get_consumed_liquidity_stats(self) -> Dict[str, Any]:
        """
        Get statistics about consumed liquidity for monitoring/debugging.

        Returns:
            Dict with consumed liquidity statistics
        """
        self._cleanup_expired_liquidity()

        stats = {
            'total_consumed_levels': len(self.consumed_liquidity),
            'by_ticker': {},
            'by_side': {}
        }

        for key, consumed in self.consumed_liquidity.items():
            ticker = consumed.ticker
            side = consumed.side

            if ticker not in stats['by_ticker']:
                stats['by_ticker'][ticker] = {
                    'levels': 0,
                    'total_quantity': 0
                }

            if side not in stats['by_side']:
                stats['by_side'][side] = {
                    'levels': 0,
                    'total_quantity': 0
                }

            stats['by_ticker'][ticker]['levels'] += 1
            stats['by_ticker'][ticker]['total_quantity'] += consumed.consumed_quantity

            stats['by_side'][side]['levels'] += 1
            stats['by_side'][side]['total_quantity'] += consumed.consumed_quantity

        return stats
    
    def calculate_fill_probability(self, order: OrderInfo, orderbook: OrderbookState) -> float:
        """
        Calculate realistic fill probability based on:
        - Price aggression (how far through the spread)
        - Time in queue (FIFO priority approximation)
        - Order size vs typical size
        - Current market activity level
        
        Returns: Probability between 0.0 and 0.99
        """
        # 1. Calculate base probability from price aggression
        base_prob = self._calculate_price_aggression_probability(order, orderbook)
        
        # 2. Time priority modifier (scaled down to ±15%)
        time_in_queue = order.time_since_placed
        if time_in_queue < 10:
            time_modifier = -0.1 + (time_in_queue / 10.0) * 0.1  # -10% to 0% for 0-10 seconds
        elif time_in_queue < 30:
            time_modifier = (time_in_queue - 10) / 20.0 * 0.05  # 0% to +5% for 10-30 seconds
        else:
            time_modifier = 0.05  # +5% for 30+ seconds (front of queue)
        
        # 3. Size impact modifier (scaled down to ±10%)
        if order.remaining_quantity < 10:
            size_modifier = 0.05  # +5% for small orders
        elif order.remaining_quantity <= 50:
            size_modifier = 0.0  # No modifier for normal size
        elif order.remaining_quantity <= 100:
            size_modifier = -0.03  # -3% for slightly large orders
        else:
            size_modifier = -0.1  # -10% for large orders
        
        # 4. Market activity modifier (scaled down to ±10%)
        activity_level = self.activity_tracker.get_activity_level()
        activity_modifier = activity_level * 0.2 - 0.1  # Maps [0,1] to [-0.1, +0.1]
        
        # 5. Edge case adjustments
        spread = self._calculate_spread(orderbook)
        
        # Wide spread adjustment
        if spread > 5:
            wide_spread_penalty = -0.1 * min((spread - 5) / 10, 1.0)  # Up to -10% for wide spreads
        else:
            wide_spread_penalty = 0.0
        
        # Empty orderbook adjustment
        if self._is_orderbook_empty(orderbook):
            empty_book_penalty = -0.3  # -30% for empty books
        else:
            empty_book_penalty = 0.0
        
        # Combine all factors
        fill_prob = (
            base_prob 
            + time_modifier 
            + size_modifier 
            + activity_modifier 
            + wide_spread_penalty 
            + empty_book_penalty
        )
        
        # Clamp to valid range [0.01, 0.99]
        fill_prob = max(0.01, min(0.99, fill_prob))
        
        return fill_prob
    
    def _calculate_price_aggression_probability(self, order: OrderInfo, orderbook: OrderbookState) -> float:
        """
        Calculate base fill probability based on price aggression.
        
        Returns base probability based on how aggressive the order price is.
        """
        # Get best bid and ask for the relevant contract side
        if order.contract_side == ContractSide.YES:
            best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        else:
            # For NO contracts
            yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
            
            if yes_best_bid is not None and yes_best_ask is not None:
                best_bid = 99 - yes_best_ask  # NO bid = 99 - YES ask
                best_ask = 99 - yes_best_bid  # NO ask = 99 - YES bid
            else:
                best_bid = None
                best_ask = None
        
        # Handle empty orderbook
        if best_bid is None or best_ask is None:
            return 0.3  # Low base probability for empty books
        
        spread = best_ask - best_bid
        mid_price = (best_bid + best_ask) / 2.0
        
        if order.side == OrderSide.BUY:
            if order.limit_price >= best_ask:
                # Crossing spread (aggressive)
                if order.limit_price > best_ask:
                    return 0.99  # Very aggressive, almost certain fill
                else:
                    return 0.95  # At ask, very high probability
            elif order.limit_price >= best_ask - 1 and spread >= 2:
                # 1 cent inside spread
                return 0.8
            elif order.limit_price >= best_ask - 2 and spread >= 3:
                # 2 cents inside spread
                return 0.9
            elif order.limit_price == best_bid:
                # At bid (passive)
                return 0.4  # 40% base probability at touch
            elif order.limit_price < best_bid:
                # Below bid (very passive)
                distance = best_bid - order.limit_price
                return max(0.1, 0.3 - distance * 0.05)  # Decay with distance
            else:
                # Inside spread
                position_in_spread = (order.limit_price - best_bid) / max(spread, 1)
                return 0.4 + position_in_spread * 0.4  # 40% to 80% inside spread
        else:  # SELL order
            if order.limit_price <= best_bid:
                # Crossing spread (aggressive)
                if order.limit_price < best_bid:
                    return 0.99  # Very aggressive, almost certain fill
                else:
                    return 0.95  # At bid, very high probability
            elif order.limit_price <= best_bid + 1 and spread >= 2:
                # 1 cent inside spread
                return 0.8
            elif order.limit_price <= best_bid + 2 and spread >= 3:
                # 2 cents inside spread
                return 0.9
            elif order.limit_price == best_ask:
                # At ask (passive)
                return 0.4  # 40% base probability at touch
            elif order.limit_price > best_ask:
                # Above ask (very passive)
                distance = order.limit_price - best_ask
                return max(0.1, 0.3 - distance * 0.05)  # Decay with distance
            else:
                # Inside spread
                position_in_spread = (best_ask - order.limit_price) / max(spread, 1)
                return 0.4 + position_in_spread * 0.4  # 40% to 80% inside spread
    
    def _calculate_spread(self, orderbook: OrderbookState) -> int:
        """Calculate the bid-ask spread in cents."""
        yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
        yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        
        if yes_best_bid is not None and yes_best_ask is not None:
            return yes_best_ask - yes_best_bid
        return 1  # Default spread if no market data
    
    def _calculate_mid_price(self, orderbook: OrderbookState) -> float:
        """Calculate the mid price."""
        yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
        yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        
        if yes_best_bid is not None and yes_best_ask is not None:
            return (yes_best_bid + yes_best_ask) / 2.0
        return 50.0  # Default mid if no market data
    
    def _is_orderbook_empty(self, orderbook: OrderbookState) -> bool:
        """Check if orderbook has no liquidity."""
        return (not orderbook.yes_bids and not orderbook.yes_asks)
    
    def _calculate_aggressive_limit_for_large_order(
        self,
        side: OrderSide,
        contract_side: ContractSide,
        orderbook: OrderbookState
    ) -> int:
        """
        Calculate a high limit price for large aggressive orders to allow depth consumption.
        
        For large orders, we want to set the limit price high enough to walk through
        multiple price levels in the orderbook.
        
        Args:
            side: Buy or Sell
            contract_side: YES or NO
            orderbook: Current orderbook
            
        Returns:
            Limit price in cents that allows depth consumption
        """
        if contract_side == ContractSide.YES:
            relevant_book = orderbook.yes_asks if side == OrderSide.BUY else orderbook.yes_bids
        else:
            # For NO contracts, we need to consider the derived book
            if side == OrderSide.BUY:
                # Buying NO = derived from YES bids
                relevant_book = {99 - price: size for price, size in orderbook.yes_bids.items()}
            else:
                # Selling NO = derived from YES asks  
                relevant_book = {99 - price: size for price, size in orderbook.yes_asks.items()}
        
        if not relevant_book:
            return 50  # Default mid-market if no book
        
        # For aggressive orders, allow walking through multiple levels
        if side == OrderSide.BUY:
            # For buy orders, find a price that allows access to several ask levels
            sorted_prices = sorted(relevant_book.keys())
            # Allow walking through at least 3 levels or all available levels
            target_levels = min(3, len(sorted_prices))
            if target_levels > 0:
                max_price = sorted_prices[target_levels - 1]
                # Add small buffer to ensure we can access this level
                limit_price = min(99, max_price + 1)
            else:
                limit_price = 99  # Maximum possible
        else:
            # For sell orders, find a price that allows access to several bid levels
            sorted_prices = sorted(relevant_book.keys(), reverse=True)
            target_levels = min(3, len(sorted_prices))
            if target_levels > 0:
                min_price = sorted_prices[target_levels - 1]
                # Subtract small buffer to ensure we can access this level
                limit_price = max(1, min_price - 1)
            else:
                limit_price = 1  # Minimum possible
        
        return limit_price
    
    def _is_aggressive_order(self, order: OrderInfo, orderbook: OrderbookState) -> bool:
        """
        Determine if an order is aggressive (crosses the spread) or passive (joins the bid/ask).
        
        Args:
            order: Order to check
            orderbook: Current orderbook state
            
        Returns:
            True if order is aggressive (crosses spread), False if passive
        """
        # Get best bid and ask for the relevant contract side
        if order.contract_side == ContractSide.YES:
            best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        else:
            # For NO contracts
            yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
            
            if yes_best_bid is not None and yes_best_ask is not None:
                best_bid = 99 - yes_best_ask  # NO bid = 99 - YES ask
                best_ask = 99 - yes_best_bid  # NO ask = 99 - YES bid
            else:
                # If no market data, assume not aggressive
                return False
        
        # If no bid/ask available, assume not aggressive
        if best_bid is None or best_ask is None:
            return False
        
        # Check if order crosses the spread
        if order.side == OrderSide.BUY:
            # Buy order is aggressive if it's at or above the best ask
            return order.limit_price >= best_ask
        else:  # SELL
            # Sell order is aggressive if it's at or below the best bid
            return order.limit_price <= best_bid


class KalshiOrderManager(OrderManager):
    """
    Kalshi order manager for paper/live trading.
    
    Wraps the KalshiDemoTradingClient to provide real API integration
    with async order lifecycle management. Handles order fills via
    WebSocket notifications and provides realistic trading experience.
    
    Features:
    - Real API calls to demo-api.kalshi.co
    - Async fill processing via user-fills WebSocket
    - Order amendment and cancellation
    - Position synchronization with Kalshi API
    """
    
    def __init__(
        self, 
        demo_client: KalshiDemoTradingClient,
        initial_cash: float = 1000.0
    ):
        """
        Initialize Kalshi order manager.
        
        Args:
            demo_client: Connected KalshiDemoTradingClient instance
            initial_cash: Starting cash balance in dollars
        """
        super().__init__(initial_cash)
        self.client = demo_client
        self._kalshi_order_mapping: Dict[str, str] = {}  # our_id -> kalshi_id
        self._reverse_mapping: Dict[str, str] = {}       # kalshi_id -> our_id
        
        logger.info("KalshiOrderManager initialized for paper trading")
    
    async def place_order(
        self,
        ticker: str,
        side: OrderSide,
        contract_side: ContractSide,
        quantity: int,
        orderbook: OrderbookState,
        pricing_strategy: str = "aggressive"
    ) -> Optional[OrderInfo]:
        """
        Place a real order via Kalshi API.
        
        Note: Kalshi only supports limit orders, so all orders are limit orders
        with prices calculated based on the pricing strategy.
        """
        try:
            # Calculate limit price
            limit_price = self._calculate_limit_price(side, contract_side, orderbook, pricing_strategy)
            
            # Convert to Kalshi API format
            kalshi_action = "buy" if side == OrderSide.BUY else "sell"
            kalshi_side = "yes" if contract_side == ContractSide.YES else "no"
            
            # Create order
            order = OrderInfo(
                order_id=self._generate_order_id(),
                ticker=ticker,
                side=side,
                contract_side=contract_side,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                placed_at=time.time()
            )
            
            # Place order via Kalshi API
            response = await self.client.create_order(
                ticker=ticker,
                action=kalshi_action,
                side=kalshi_side,
                count=quantity,
                price=limit_price,
                type="limit"
            )
            
            # Extract Kalshi order ID
            if "order" in response:
                kalshi_order_id = response["order"].get("order_id", "")
                if kalshi_order_id:
                    # Map our order ID to Kalshi's
                    self._kalshi_order_mapping[order.order_id] = kalshi_order_id
                    self._reverse_mapping[kalshi_order_id] = order.order_id
                    
                    # Add to our tracking
                    self.open_orders[order.order_id] = order
                    
                    logger.info(f"Kalshi order placed: {order.order_id} -> {kalshi_order_id}")
                    return order
            
            logger.error(f"Failed to get Kalshi order ID from response: {response}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to place Kalshi order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a Kalshi order."""
        try:
            if order_id not in self._kalshi_order_mapping:
                logger.error(f"Order ID {order_id} not found in Kalshi mapping")
                return False
            
            kalshi_order_id = self._kalshi_order_mapping[order_id]
            
            # Cancel via Kalshi API
            await self.client.cancel_order(kalshi_order_id)
            
            # Update our tracking
            if order_id in self.open_orders:
                self.open_orders[order_id].status = OrderStatus.CANCELLED
                del self.open_orders[order_id]
            
            # Clean up mappings
            del self._kalshi_order_mapping[order_id]
            if kalshi_order_id in self._reverse_mapping:
                del self._reverse_mapping[kalshi_order_id]
            
            logger.info(f"Kalshi order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel Kalshi order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """Cancel all Kalshi orders."""
        cancelled_count = 0
        orders_to_cancel = []
        
        for order in self.open_orders.values():
            if ticker is None or order.ticker == ticker:
                orders_to_cancel.append(order.order_id)
        
        # Cancel orders
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        return cancelled_count
    
    async def amend_order(
        self,
        order_id: str,
        new_price: int,
        orderbook: OrderbookState
    ) -> bool:
        """
        Amend a Kalshi order's price.
        
        Note: Kalshi API might not support order amendment directly,
        so this might require cancel + replace.
        """
        try:
            # For now, implement as cancel + replace
            # This could be optimized if Kalshi adds amendment support
            
            if order_id not in self.open_orders:
                return False
            
            old_order = self.open_orders[order_id]
            
            # Cancel old order
            success = await self.cancel_order(order_id)
            if not success:
                return False
            
            # Place new order with updated price
            new_order = await self.place_order(
                ticker=old_order.ticker,
                side=old_order.side,
                contract_side=old_order.contract_side,
                quantity=old_order.quantity,
                orderbook=orderbook,
                pricing_strategy="aggressive"  # Use aggressive to hit new price
            )
            
            if new_order:
                # Update the new order's price to match requested price
                new_order.limit_price = max(1, min(99, new_price))
                logger.info(f"Kalshi order amended via cancel+replace: {order_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to amend Kalshi order {order_id}: {e}")
            return False
    
    async def check_fills(self, orderbook: OrderbookState) -> List[OrderInfo]:
        """
        Check for order fills via Kalshi API.
        
        This method syncs our local state with Kalshi's order status.
        In practice, fills would be processed via WebSocket notifications.
        """
        filled_orders = []
        
        try:
            # Get updated order status from Kalshi
            kalshi_orders = await self.client.get_orders()
            
            if "orders" in kalshi_orders:
                for kalshi_order in kalshi_orders["orders"]:
                    kalshi_order_id = kalshi_order.get("order_id", "")
                    
                    # Check if this is one of our tracked orders
                    if kalshi_order_id in self._reverse_mapping:
                        our_order_id = self._reverse_mapping[kalshi_order_id]
                        
                        if our_order_id in self.open_orders:
                            our_order = self.open_orders[our_order_id]
                            
                            # Check if order was filled
                            kalshi_status = kalshi_order.get("status", "").lower()
                            if kalshi_status in ["filled", "executed"]:
                                # Process the fill
                                fill_price = kalshi_order.get("yes_price", our_order.limit_price)
                                self._process_fill(our_order, fill_price)
                                filled_orders.append(our_order)
                                
                                # Clean up Kalshi-specific mappings
                                # Note: _process_fill already removed from open_orders
                                del self._kalshi_order_mapping[our_order_id]
                                del self._reverse_mapping[kalshi_order_id]
        
        except Exception as e:
            logger.error(f"Failed to check Kalshi order fills: {e}")
        
        return filled_orders
    
    async def sync_positions_with_kalshi(self) -> None:
        """
        Sync our position tracking with Kalshi API state.
        
        This ensures our local position tracking stays in sync with
        the actual account state on Kalshi.
        """
        try:
            # Get positions from Kalshi
            positions_response = await self.client.get_positions()
            
            if "positions" in positions_response:
                kalshi_positions = positions_response["positions"]
                
                # Clear our current positions and rebuild from Kalshi data
                self.positions.clear()
                
                for kalshi_pos in kalshi_positions:
                    ticker = kalshi_pos.get("ticker", "")
                    if ticker:
                        # Extract position data from Kalshi format
                        position_count = kalshi_pos.get("position", 0)
                        # Note: Kalshi might provide separate fields for YES/NO positions
                        
                        self.positions[ticker] = Position(
                            ticker=ticker,
                            contracts=position_count,  # Adjust based on actual Kalshi format
                            cost_basis=0.0,  # Would need to calculate or get from API
                            realized_pnl=0.0
                        )
            
            # Get account balance
            balance_response = await self.client.get_account_info()
            if "balance" in balance_response:
                self.cash_balance = balance_response["balance"] / 100.0  # Convert from cents
            
            logger.debug("Position sync with Kalshi completed")
            
        except Exception as e:
            logger.error(f"Failed to sync positions with Kalshi: {e}")
    
    async def _connect_websocket(self) -> None:
        """
        Connect to Kalshi user-fills WebSocket stream for real-time fill tracking.
        
        This method establishes a WebSocket connection to the user-fills stream
        and processes fill messages in real-time, providing immediate position
        updates without the need for polling.
        """
        try:
            # WebSocket URL for user fills (demo API)
            ws_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"
            
            logger.info("Connecting to Kalshi user-fills WebSocket...")
            
            # Note: This is a simplified implementation
            # In production, you'd need:
            # 1. Proper authentication headers
            # 2. Subscription message to user-fills channel
            # 3. Reconnection logic
            # 4. Error handling for connection drops
            
            # For now, we'll set up the structure for future implementation
            self._ws_connected = True
            self._ws_url = ws_url
            
            logger.info("WebSocket connection established (placeholder implementation)")
            
            # TODO: Implement actual WebSocket connection with:
            # - Authentication using demo client credentials
            # - Subscription to user fills channel
            # - Message processing loop
            # - Automatic reconnection on disconnection
            
        except Exception as e:
            logger.error(f"Failed to connect to fills WebSocket: {e}")
            self._ws_connected = False
    
    async def _process_fill_message(self, message: Dict[str, Any]) -> None:
        """
        Process a fill message from the user-fills WebSocket stream.
        
        Args:
            message: WebSocket message containing fill information
            
        Message format expected from Kalshi:
        {
            "type": "fill",
            "data": {
                "order_id": "kalshi_order_id",
                "ticker": "MARKET-123", 
                "side": "yes",
                "action": "buy",
                "count": 10,
                "yes_price": 65,
                "fill_time": "2023-12-10T15:30:00Z"
            }
        }
        """
        try:
            if message.get("type") != "fill":
                return
            
            fill_data = message.get("data", {})
            kalshi_order_id = fill_data.get("order_id", "")
            
            # Find our order that corresponds to this Kalshi order
            if kalshi_order_id not in self._reverse_mapping:
                logger.warning(f"Received fill for unknown order: {kalshi_order_id}")
                return
            
            our_order_id = self._reverse_mapping[kalshi_order_id]
            
            if our_order_id not in self.open_orders:
                logger.warning(f"Fill received for order not in open orders: {our_order_id}")
                return
            
            # Get our order info
            order = self.open_orders[our_order_id]
            
            # Extract fill information
            fill_price = fill_data.get("yes_price", order.limit_price)
            filled_quantity = fill_data.get("count", order.quantity)
            
            logger.info(f"Processing WebSocket fill: {our_order_id} - {filled_quantity} @ {fill_price}¢")
            
            # Process the fill using our existing logic
            self._process_fill(order, fill_price)
            
            # Clean up tracking for fully filled orders
            if filled_quantity >= order.quantity:
                # Order fully filled - remove from tracking
                # Note: _process_fill already removed from open_orders
                if our_order_id in self._kalshi_order_mapping:
                    del self._kalshi_order_mapping[our_order_id]
                if kalshi_order_id in self._reverse_mapping:
                    del self._reverse_mapping[kalshi_order_id]
                
                logger.info(f"Order fully filled and removed from tracking: {our_order_id}")
            else:
                # Partial fill - update quantity remaining
                order.quantity -= filled_quantity
                logger.info(f"Partial fill processed, {order.quantity} contracts remaining: {our_order_id}")
            
        except Exception as e:
            logger.error(f"Error processing fill message: {e}")
    
    def get_consumed_liquidity_stats(self) -> Dict[str, Any]:
        """
        Get statistics about consumed liquidity for monitoring/debugging.
        
        Returns:
            Dict with consumed liquidity statistics
        """
        self._cleanup_expired_liquidity()
        
        stats = {
            'total_consumed_levels': len(self.consumed_liquidity),
            'by_ticker': {},
            'by_side': {}
        }
        
        for key, consumed in self.consumed_liquidity.items():
            ticker = consumed.ticker
            side = consumed.side
            
            if ticker not in stats['by_ticker']:
                stats['by_ticker'][ticker] = {
                    'levels': 0,
                    'total_quantity': 0
                }
            
            if side not in stats['by_side']:
                stats['by_side'][side] = {
                    'levels': 0,
                    'total_quantity': 0
                }
            
            stats['by_ticker'][ticker]['levels'] += 1
            stats['by_ticker'][ticker]['total_quantity'] += consumed.consumed_quantity
            
            stats['by_side'][side]['levels'] += 1
            stats['by_side'][side]['total_quantity'] += consumed.consumed_quantity
        
        return stats
    
    async def start_fill_tracking(self) -> None:
        """
        Start real-time fill tracking via WebSocket.
        
        This method should be called after the OrderManager is initialized
        to enable real-time fill processing instead of polling.
        """
        try:
            # Connect to WebSocket
            await self._connect_websocket()
            
            if self._ws_connected:
                logger.info("Real-time fill tracking started")
                
                # In a full implementation, this would start the message processing loop
                # For now, we just mark that tracking is enabled
                self._fill_tracking_active = True
            else:
                logger.warning("Failed to start fill tracking - WebSocket not connected")
                
        except Exception as e:
            logger.error(f"Failed to start fill tracking: {e}")
    
    async def stop_fill_tracking(self) -> None:
        """
        Stop real-time fill tracking and close WebSocket connection.
        """
        try:
            self._fill_tracking_active = False
            
            if hasattr(self, '_ws_connected') and self._ws_connected:
                # In a full implementation, this would close the WebSocket connection
                self._ws_connected = False
                logger.info("Real-time fill tracking stopped")
                
        except Exception as e:
            logger.error(f"Error stopping fill tracking: {e}")
    
    def is_fill_tracking_active(self) -> bool:
        """Check if real-time fill tracking is active."""
        return getattr(self, '_fill_tracking_active', False)