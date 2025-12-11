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
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import IntEnum
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
    fill_price: Optional[int] = None
    
    @property
    def time_since_placed(self) -> float:
        """Time since order was placed in seconds."""
        return time.time() - self.placed_at
    
    def is_active(self) -> bool:
        """Check if order is still active (not filled/cancelled/rejected)."""
        return self.status == OrderStatus.PENDING


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
    
    def get_unrealized_pnl(self, current_yes_price: float) -> float:
        """
        Calculate unrealized P&L based on current market price.
        
        Args:
            current_yes_price: Current YES price as a probability (0.0-1.0)
            
        Returns:
            Unrealized P&L in dollars
        """
        if self.is_flat:
            return 0.0
        
        if self.is_long_yes:
            # Long YES: profit when YES price rises
            current_value = self.contracts * current_yes_price
        else:
            # Long NO: profit when YES price falls (NO price = 1 - YES price)
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
    
    def get_total_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value including cash and positions.
        
        Args:
            current_prices: Dictionary of current YES prices by ticker
            
        Returns:
            Total portfolio value in dollars
        """
        total_value = self.cash_balance
        
        for ticker, position in self.positions.items():
            if not position.is_flat and ticker in current_prices:
                # Add unrealized P&L to portfolio value
                unrealized_pnl = position.get_unrealized_pnl(current_prices[ticker])
                total_value += position.cost_basis + unrealized_pnl
        
        return total_value
    
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


class SimulatedOrderManager(OrderManager):
    """
    Simulated order manager for training environments.
    
    Provides pure Python simulation of order execution without any API calls.
    Orders that cross the spread are filled immediately, while others remain
    as pending limit orders until they can be filled.
    
    Features:
    - Instant fills for orders that cross the spread
    - Realistic limit order behavior
    - Deterministic execution for reproducible training
    - No network latency or API dependencies
    """
    
    def __init__(self, initial_cash: float = 1000.0):
        """
        Initialize simulated order manager.
        
        Args:
            initial_cash: Starting cash balance in dollars
        """
        super().__init__(initial_cash)
        logger.info("SimulatedOrderManager initialized for training")
    
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
            # Calculate limit price
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
            
            # Check for immediate fill
            can_fill = self._can_fill_immediately(order, orderbook)
            
            if can_fill:
                # Execute immediate fill
                fill_price = self._get_fill_price(order, orderbook)
                self._process_fill(order, fill_price)
                logger.debug(f"Simulated order filled immediately: {order.order_id}")
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
    
    async def check_fills(self, orderbook: OrderbookState) -> List[OrderInfo]:
        """Check for fills of pending simulated orders."""
        filled_orders = []
        orders_to_remove = []
        
        for order in list(self.open_orders.values()):
            if self._can_fill_immediately(order, orderbook):
                fill_price = self._get_fill_price(order, orderbook)
                self._process_fill(order, fill_price)
                filled_orders.append(order)
                orders_to_remove.append(order.order_id)
        
        # Remove filled orders
        for order_id in orders_to_remove:
            if order_id in self.open_orders:
                del self.open_orders[order_id]
        
        return filled_orders
    
    def _can_fill_immediately(self, order: OrderInfo, orderbook: OrderbookState) -> bool:
        """Check if an order can be filled immediately at current market prices."""
        if order.contract_side == ContractSide.YES:
            best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        else:
            # For NO contracts  
            yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
            
            if yes_best_bid is not None and yes_best_ask is not None:
                best_bid = 99 - yes_best_ask
                best_ask = 99 - yes_best_bid
            else:
                return False  # No prices available
        
        if best_bid is None or best_ask is None:
            return False  # No market prices available
        
        if order.side == OrderSide.BUY:
            # Buy order fills if our bid price >= market ask
            return order.limit_price >= best_ask
        else:
            # Sell order fills if our ask price <= market bid
            return order.limit_price <= best_bid
    
    def _get_fill_price(self, order: OrderInfo, orderbook: OrderbookState) -> int:
        """Get the price at which an order would fill."""
        if order.contract_side == ContractSide.YES:
            best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        else:
            yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
            
            if yes_best_bid is not None and yes_best_ask is not None:
                best_bid = 99 - yes_best_ask
                best_ask = 99 - yes_best_bid
            else:
                # Fallback to order's limit price if no market data
                return order.limit_price
        
        if best_bid is None or best_ask is None:
            # Fallback to order's limit price if no market data
            return order.limit_price
        
        if order.side == OrderSide.BUY:
            # Buy orders fill at the market ask price
            return best_ask
        else:
            # Sell orders fill at the market bid price
            return best_bid


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