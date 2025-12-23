"""
OrderService - Extracted from OrderManager for TRADER 2.0

Handles order lifecycle management: placement, cancellation, and tracking.
Focused extraction of working order management functionality from the monolithic OrderManager.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from enum import IntEnum
from dataclasses import dataclass, field

from ..demo_client import KalshiDemoTradingClient, KalshiDemoTradingClientError
from ...data.orderbook_state import OrderbookState

logger = logging.getLogger("kalshiflow_rl.trading.services.order_service")


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


class OrderService:
    """
    Handles order lifecycle management.
    
    Extracted from OrderManager to focus solely on order placement, cancellation,
    and tracking. Integrates with Kalshi API for real order execution.
    """
    
    def __init__(
        self, 
        client: KalshiDemoTradingClient, 
        position_tracker: Optional['PositionTracker'] = None,
        status_logger: Optional['StatusLogger'] = None,
        websocket_manager=None
    ):
        """
        Initialize OrderService.
        
        Args:
            client: KalshiDemoTradingClient for API integration
            position_tracker: PositionTracker for cash balance checking
            status_logger: Optional StatusLogger for activity tracking
            websocket_manager: Global WebSocketManager for broadcasting (optional)
        """
        self.client = client
        self.position_tracker = position_tracker
        self.status_logger = status_logger
        self.websocket_manager = websocket_manager
        
        # Order tracking
        self.open_orders: Dict[str, OrderInfo] = {}
        self.order_history: List[OrderInfo] = []
        
        # Kalshi order ID mapping (our ID -> Kalshi ID)
        self._kalshi_order_mapping: Dict[str, str] = {}
        self._reverse_mapping: Dict[str, str] = {}  # Kalshi ID -> our ID
        
        # Order ID generation
        self._order_counter = 0
        
        # Event callbacks
        self.on_order_filled: Optional[Callable[[OrderInfo], None]] = None
        self.on_order_cancelled: Optional[Callable[[OrderInfo], None]] = None
        
        logger.info("OrderService initialized")
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"ORDER_{int(time.time())}_{self._order_counter:04d}"
    
    def _check_order_affordability(
        self,
        side: OrderSide,
        quantity: int,
        limit_price: int
    ) -> Dict[str, Any]:
        """
        Check if order is affordable using API-synced cash balance.
        
        Args:
            side: Buy or Sell
            quantity: Number of contracts
            limit_price: Limit price in cents
            
        Returns:
            Dict with affordability check results
        """
        if not self.position_tracker:
            # No position tracker - cannot check affordability
            return {
                "affordable": False,
                "error": "No position tracker available for affordability check",
                "required_cash": 0.0,
                "available_cash": 0.0
            }
        
        # Only BUY orders require cash
        if side == OrderSide.SELL:
            return {
                "affordable": True,
                "required_cash": 0.0,
                "available_cash": self.position_tracker.cash_balance,
                "message": "SELL orders do not require cash"
            }
        
        # Calculate required cash for BUY order
        required_cash = (limit_price / 100.0) * quantity
        available_cash = self.position_tracker.cash_balance
        
        # Check affordability with small buffer for precision
        affordable = available_cash >= (required_cash - 0.01)  # 1 cent tolerance
        
        return {
            "affordable": affordable,
            "required_cash": required_cash,
            "available_cash": available_cash,
            "error": None if affordable else f"Insufficient cash: need ${required_cash:.2f}, have ${available_cash:.2f}"
        }

    def _calculate_limit_price(
        self,
        side: OrderSide,
        contract_side: ContractSide,
        orderbook: OrderbookState,
        pricing_strategy: str = "aggressive"
    ) -> int:
        """
        Calculate limit price based on orderbook and pricing strategy.
        
        Extracted from OrderManager pricing logic.
        """
        try:
            # Get best bid/ask from orderbook
            best_bid = orderbook.best_bid_price() or 50
            best_ask = orderbook.best_ask_price() or 50
            
            if pricing_strategy == "aggressive":
                # Take liquidity - cross the spread
                if side == OrderSide.BUY:
                    if contract_side == ContractSide.YES:
                        # Buy YES at ask price (immediate execution)
                        return best_ask
                    else:
                        # Buy NO at (100 - bid) price
                        return 100 - best_bid
                else:  # SELL
                    if contract_side == ContractSide.YES:
                        # Sell YES at bid price
                        return best_bid
                    else:
                        # Sell NO at (100 - ask) price  
                        return 100 - best_ask
            
            elif pricing_strategy == "passive":
                # Make liquidity - join the inside
                if side == OrderSide.BUY:
                    if contract_side == ContractSide.YES:
                        return best_bid  # Join bid
                    else:
                        return 100 - best_ask  # Join bid for NO
                else:  # SELL
                    if contract_side == ContractSide.YES:
                        return best_ask  # Join ask
                    else:
                        return 100 - best_bid  # Join ask for NO
            
            else:
                # Default: mid-price
                mid_price = (best_bid + best_ask) // 2
                return mid_price
                
        except Exception as e:
            logger.warning(f"Error calculating limit price: {e}, using mid-price")
            return 50  # Safe fallback
    
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
        
        Args:
            ticker: Market ticker
            side: Buy or Sell
            contract_side: YES or NO
            quantity: Number of contracts
            orderbook: Current orderbook state for pricing
            pricing_strategy: "aggressive", "passive", or "mid"
            
        Returns:
            OrderInfo if successful, None if failed
        """
        start_time = time.time()
        
        try:
            # Calculate limit price
            limit_price = self._calculate_limit_price(side, contract_side, orderbook, pricing_strategy)
            
            # Check affordability before placing order
            affordability_check = self._check_order_affordability(side, quantity, limit_price)
            if not affordability_check["affordable"]:
                error_msg = affordability_check["error"]
                logger.warning(f"Order not affordable: {error_msg}")
                
                if self.status_logger:
                    await self.status_logger.log_action_result(
                        "order_rejected",
                        f"{ticker} - {error_msg}",
                        time.time() - start_time
                    )
                
                return None
            
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
            
            # Log activity
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "OrderService", "placing_order",
                    {"ticker": ticker, "side": kalshi_action, "quantity": quantity, "price": limit_price}
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
            
            duration = time.time() - start_time
            
            # Extract Kalshi order ID
            if "order" in response:
                kalshi_order_id = response["order"].get("order_id", "")
                if kalshi_order_id:
                    # Map our order ID to Kalshi's
                    self._kalshi_order_mapping[order.order_id] = kalshi_order_id
                    self._reverse_mapping[kalshi_order_id] = order.order_id
                    
                    # Add to our tracking
                    self.open_orders[order.order_id] = order
                    
                    # Broadcast order update
                    await self._broadcast_orders_update("place_order")
                    
                    # Log success
                    if self.status_logger:
                        await self.status_logger.log_action_result(
                            "order_placed",
                            f"{ticker} {kalshi_action.upper()} {quantity}@{limit_price}¢",
                            duration
                        )
                    
                    logger.info(f"Order placed: {order.order_id} -> {kalshi_order_id} ({duration:.2f}s)")
                    return order
            
            logger.error(f"Failed to get Kalshi order ID from response: {response}")
            
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "order_failed",
                    f"{ticker} {kalshi_action.upper()} - no order ID",
                    duration
                )
            
            return None
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Failed to place order: {error_msg}")
            
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "order_error",
                    f"{ticker} - {error_msg[:50]}",
                    duration
                )
            
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        start_time = time.time()
        
        try:
            # Check if order exists in our tracking
            if order_id not in self._kalshi_order_mapping:
                logger.error(f"Order ID {order_id} not found in Kalshi mapping")
                return False
            
            kalshi_order_id = self._kalshi_order_mapping[order_id]
            
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "OrderService", "cancelling_order",
                    {"order_id": order_id, "kalshi_order_id": kalshi_order_id}
                )
            
            # Cancel via Kalshi API
            await self.client.cancel_order(kalshi_order_id)
            
            duration = time.time() - start_time
            
            # Update our tracking
            if order_id in self.open_orders:
                order = self.open_orders[order_id]
                order.status = OrderStatus.CANCELLED
                
                # Move to history
                self.order_history.append(order)
                del self.open_orders[order_id]
                
                # Clean up mappings
                if order_id in self._kalshi_order_mapping:
                    del self._kalshi_order_mapping[order_id]
                if kalshi_order_id in self._reverse_mapping:
                    del self._reverse_mapping[kalshi_order_id]
                
                # Fire callback
                if self.on_order_cancelled:
                    self.on_order_cancelled(order)
                
                # Broadcast order update
                await self._broadcast_orders_update("cancel_order")
                
                # Log success
                if self.status_logger:
                    await self.status_logger.log_action_result(
                        "order_cancelled",
                        f"{order.ticker} {order.order_id}",
                        duration
                    )
                
                logger.info(f"Order cancelled: {order_id} ({duration:.2f}s)")
                return True
            
            logger.warning(f"Order {order_id} not found in open orders after cancellation")
            return True  # API call succeeded
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            logger.error(f"Failed to cancel order {order_id}: {error_msg}")
            
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "cancel_error",
                    f"{order_id} - {error_msg[:50]}",
                    duration
                )
            
            return False
    
    def get_open_orders(self, ticker: Optional[str] = None) -> Dict[str, OrderInfo]:
        """Get open orders, optionally filtered by ticker."""
        if ticker:
            return {oid: order for oid, order in self.open_orders.items() 
                   if order.ticker == ticker}
        return self.open_orders.copy()
    
    def get_order_by_id(self, order_id: str) -> Optional[OrderInfo]:
        """Get order by ID from open orders or history."""
        if order_id in self.open_orders:
            return self.open_orders[order_id]
        
        # Search history
        for order in self.order_history:
            if order.order_id == order_id:
                return order
        
        return None
    
    def get_order_by_kalshi_id(self, kalshi_order_id: str) -> Optional[OrderInfo]:
        """Get order by Kalshi order ID."""
        if kalshi_order_id in self._reverse_mapping:
            our_order_id = self._reverse_mapping[kalshi_order_id]
            return self.get_order_by_id(our_order_id)
        return None
    
    def handle_fill_event(self, kalshi_order_id: str, fill_data: Dict[str, Any]) -> Optional[OrderInfo]:
        """
        Handle fill event from WebSocket.
        
        Args:
            kalshi_order_id: Kalshi's order ID
            fill_data: Fill information from WebSocket
            
        Returns:
            OrderInfo if fill processed, None if order not found
        """
        try:
            # Find our order
            order = self.get_order_by_kalshi_id(kalshi_order_id)
            if not order:
                logger.warning(f"Received fill for unknown Kalshi order: {kalshi_order_id}")
                return None
            
            # Extract fill information
            fill_price = fill_data.get("yes_price", order.limit_price)
            filled_quantity = fill_data.get("count", order.quantity)
            
            logger.info(f"Processing fill event: {order.order_id} - {filled_quantity} @ {fill_price}¢")
            
            # Update order with fill info
            order.status = OrderStatus.FILLED
            order.filled_at = time.time()
            order.fill_price = fill_price
            order.filled_quantity = filled_quantity
            order.remaining_quantity = order.quantity - filled_quantity
            
            # Remove from open orders
            if order.order_id in self.open_orders:
                del self.open_orders[order.order_id]
            
            # Add to history
            self.order_history.append(order)
            
            # Clean up mappings
            if order.order_id in self._kalshi_order_mapping:
                del self._kalshi_order_mapping[order.order_id]
            if kalshi_order_id in self._reverse_mapping:
                del self._reverse_mapping[kalshi_order_id]
            
            # Fire callback for position tracking
            if self.on_order_filled:
                self.on_order_filled(order)
            
            # Broadcast order update
            asyncio.create_task(self._broadcast_orders_update("fill_event"))
            
            # Log activity
            if self.status_logger:
                asyncio.create_task(self.status_logger.log_action_result(
                    "fill_processed",
                    f"{order.ticker} {order.side.name} {filled_quantity}@{fill_price}¢",
                    0.0  # Fill processing duration
                ))
            
            logger.info(f"Fill processed successfully: {order.order_id}")
            return order
            
        except Exception as e:
            logger.error(f"Error processing fill event: {e}")
            return None
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics for monitoring."""
        total_orders = len(self.open_orders) + len(self.order_history)
        
        filled_orders = [order for order in self.order_history if order.status == OrderStatus.FILLED]
        cancelled_orders = [order for order in self.order_history if order.status == OrderStatus.CANCELLED]
        
        return {
            "orders_pending": len(self.open_orders),
            "orders_total": total_orders,
            "orders_filled": len(filled_orders),
            "orders_cancelled": len(cancelled_orders),
            "fill_rate": len(filled_orders) / max(1, total_orders),
            "avg_fill_time": sum([
                order.filled_at - order.placed_at for order in filled_orders 
                if order.filled_at
            ]) / max(1, len(filled_orders)),
            "last_activity": max([
                order.placed_at for order in list(self.open_orders.values()) + self.order_history
            ]) if total_orders > 0 else None
        }
    
    async def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders successfully cancelled
        """
        cancelled_count = 0
        open_order_ids = list(self.open_orders.keys())
        
        logger.info(f"Cancelling {len(open_order_ids)} open orders")
        
        for order_id in open_order_ids:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count}/{len(open_order_ids)} orders")
        return cancelled_count
    
    async def _broadcast_orders_update(self, source: str = "order_service") -> None:
        """
        Broadcast order updates via WebSocket.
        
        Args:
            source: Source of the update for tracking
        """
        if self.websocket_manager:
            try:
                # Prepare order data for broadcasting
                orders_data = {
                    "open_orders": [
                        {
                            "order_id": order.order_id,
                            "ticker": order.ticker,
                            "side": order.side.name,
                            "contract_side": order.contract_side.name,
                            "quantity": order.quantity,
                            "limit_price": order.limit_price,
                            "status": order.status.name,
                            "placed_at": order.placed_at,
                            "time_since_placed": order.time_since_placed
                        }
                        for order in self.open_orders.values()
                    ],
                    "total_open": len(self.open_orders),
                    "total_history": len(self.order_history),
                    "timestamp": time.time()
                }
                
                await self.websocket_manager.broadcast_orders_update(orders_data, source)
            except Exception as e:
                logger.warning(f"Failed to broadcast order update: {e}")