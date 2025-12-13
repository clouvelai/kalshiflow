"""
Consolidated Kalshi Multi-Market Order Manager for RL Trading.

This replaces both KalshiOrderManager and MultiMarketOrderManager with a single,
clean implementation that:
1. Handles multi-market trading via a single KalshiDemoTradingClient
2. Uses Option B cash tracking (deduct on place, restore on cancel)
3. Processes fills via a dedicated async queue for proper reconciliation
4. Maintains single cash pool across all markets

Architecture:
- ActorService owns action/event queue (processes actions from RL agent)
- KalshiMultiMarketOrderManager owns fills queue (processes fills from WebSocket)
- Single KalshiDemoTradingClient for all API calls
- Single cash pool with promised_cash tracking
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import IntEnum

from .demo_client import KalshiDemoTradingClient

logger = logging.getLogger("kalshiflow_rl.trading.kalshi_multi_market_order_manager")


class OrderStatus(IntEnum):
    """Order status enumeration."""
    PENDING = 0
    FILLED = 1
    CANCELLED = 2
    REJECTED = 3


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
    """Order tracking information."""
    order_id: str                    # Our internal order ID
    kalshi_order_id: str            # Kalshi's order ID
    ticker: str
    side: OrderSide
    contract_side: ContractSide
    quantity: int
    limit_price: int                # Price in cents (1-99)
    status: OrderStatus
    placed_at: float
    promised_cash: float            # Cash reserved for this order
    filled_at: Optional[float] = None
    fill_price: Optional[int] = None
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status == OrderStatus.PENDING


@dataclass
class Position:
    """Position in a specific market using Kalshi convention."""
    ticker: str
    contracts: int       # +contracts for YES, -contracts for NO
    cost_basis: float    # Total cost in dollars
    realized_pnl: float  # Cumulative realized P&L
    
    @property
    def is_flat(self) -> bool:
        """True if no position."""
        return self.contracts == 0
    
    def get_unrealized_pnl(self, current_yes_price: float) -> float:
        """Calculate unrealized P&L based on current YES price (0.0-1.0)."""
        if self.is_flat:
            return 0.0
        
        if self.contracts > 0:
            # Long YES: profit when YES price rises
            current_value = self.contracts * current_yes_price
        else:
            # Long NO: profit when YES price falls (NO price = 1 - YES price)
            current_value = abs(self.contracts) * (1.0 - current_yes_price)
        
        return current_value - self.cost_basis


@dataclass
class FillEvent:
    """Fill event for processing queue."""
    kalshi_order_id: str
    fill_price: int
    fill_quantity: int
    fill_timestamp: float
    
    @classmethod
    def from_kalshi_message(cls, message: Dict[str, Any]) -> Optional['FillEvent']:
        """Create fill event from Kalshi WebSocket message."""
        try:
            fill_data = message.get("data", {})
            return cls(
                kalshi_order_id=fill_data.get("order_id", ""),
                fill_price=fill_data.get("yes_price", 0),
                fill_quantity=fill_data.get("count", 0),
                fill_timestamp=time.time()  # Use current time for now
            )
        except Exception as e:
            logger.error(f"Error parsing fill message: {e}")
            return None


class KalshiMultiMarketOrderManager:
    """
    Consolidated multi-market order manager for RL trading.
    
    Features:
    - Single KalshiDemoTradingClient for all markets
    - Option B cash tracking (deduct on place, restore on cancel)
    - Dedicated fill queue for proper state reconciliation
    - Single cash pool across all markets
    - Clean integration with ActorService
    """
    
    def __init__(self, initial_cash: float = 10000.0):
        """
        Initialize the order manager.
        
        Args:
            initial_cash: Starting cash balance in dollars
        """
        # Cash management (Option B)
        self.cash_balance = initial_cash      # Available cash
        self.promised_cash = 0.0              # Cash reserved for open orders
        self.initial_cash = initial_cash
        
        # Order and position tracking
        self.open_orders: Dict[str, OrderInfo] = {}  # {our_order_id: OrderInfo}
        self.positions: Dict[str, Position] = {}     # {ticker: Position}
        
        # Order ID mapping
        self._kalshi_to_internal: Dict[str, str] = {}  # {kalshi_id: our_id}
        self._order_counter = 0
        
        # Fill processing queue
        self.fills_queue: asyncio.Queue[FillEvent] = asyncio.Queue()
        self._fill_processor_task: Optional[asyncio.Task] = None
        
        # Trading client
        self.trading_client: Optional[KalshiDemoTradingClient] = None
        
        # Monitoring
        self._orders_placed = 0
        self._orders_filled = 0
        self._orders_cancelled = 0
        self._total_volume_traded = 0.0
        
        logger.info(f"KalshiMultiMarketOrderManager initialized with ${initial_cash:.2f}")
    
    async def initialize(self) -> None:
        """Initialize the order manager and start fill processing."""
        logger.info("Initializing KalshiMultiMarketOrderManager...")
        
        # Initialize trading client
        self.trading_client = KalshiDemoTradingClient()
        await self.trading_client.connect()
        logger.info("✅ Demo trading client connected")
        
        # Start fill processor
        self._fill_processor_task = asyncio.create_task(self._process_fills())
        logger.info("✅ Fill processor started")
        
        logger.info("✅ KalshiMultiMarketOrderManager ready for trading")
    
    async def shutdown(self) -> None:
        """Shutdown the order manager."""
        logger.info("Shutting down KalshiMultiMarketOrderManager...")
        
        # Cancel all open orders (only if trading client is initialized)
        if self.trading_client is not None:
            await self.cancel_all_orders()
        
        # Stop fill processor
        if self._fill_processor_task:
            self._fill_processor_task.cancel()
            try:
                await self._fill_processor_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect trading client
        if self.trading_client:
            await self.trading_client.disconnect()
        
        logger.info("✅ KalshiMultiMarketOrderManager shutdown complete")
    
    async def execute_order(
        self, 
        market_ticker: str, 
        action: int,
        orderbook_snapshot: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute an order action for the specified market.
        
        Args:
            market_ticker: Market to trade
            action: Action ID (0=HOLD, 1=BUY_YES_LIMIT, 2=SELL_YES_LIMIT, 3=BUY_NO_LIMIT, 4=SELL_NO_LIMIT)
            orderbook_snapshot: Optional orderbook snapshot for price calculation
            
        Returns:
            Execution result dict or None if failed
        """
        if action == 0:  # HOLD
            return {"status": "hold", "action": action, "market": market_ticker}
        
        if not self.trading_client:
            logger.error("Trading client not initialized")
            return None
        
        try:
            # Determine order parameters (matches LimitOrderActions enum exactly)
            # Action space: 0=HOLD, 1=BUY_YES_LIMIT, 2=SELL_YES_LIMIT, 3=BUY_NO_LIMIT, 4=SELL_NO_LIMIT
            # This mapping MUST match backend/src/kalshiflow_rl/environments/limit_order_action_space.py
            if action == 1:  # BUY_YES_LIMIT
                side, contract_side = OrderSide.BUY, ContractSide.YES
            elif action == 2:  # SELL_YES_LIMIT
                side, contract_side = OrderSide.SELL, ContractSide.YES
            elif action == 3:  # BUY_NO_LIMIT
                side, contract_side = OrderSide.BUY, ContractSide.NO
            elif action == 4:  # SELL_NO_LIMIT
                side, contract_side = OrderSide.SELL, ContractSide.NO
            else:
                logger.error(f"Unknown action: {action} (valid range: 0-4)")
                return None
            
            # Fixed contract size matching training (10 contracts)
            quantity = 10
            
            # Calculate limit price from orderbook snapshot
            limit_price = self._calculate_limit_price_from_snapshot(
                orderbook_snapshot, side, contract_side
            )
            
            # Check cash for BUY orders (Option B)
            if side == OrderSide.BUY:
                order_cost = (limit_price / 100.0) * quantity
                if self.cash_balance < order_cost:
                    logger.warning(f"Insufficient cash for BUY order: ${self.cash_balance:.2f} < ${order_cost:.2f}")
                    return {"status": "insufficient_cash", "required": order_cost, "available": self.cash_balance}
            
            # Place order via Kalshi API
            result = await self._place_kalshi_order(
                ticker=market_ticker,
                side=side,
                contract_side=contract_side,
                quantity=quantity,
                limit_price=limit_price
            )
            
            if result:
                logger.info(f"Order placed: {action} for {market_ticker} -> {result['order_id']}")
                self._orders_placed += 1
                return {
                    "status": "placed",
                    "order_id": result["order_id"],
                    "action": action,
                    "market": market_ticker,
                    "side": side.name,
                    "contract_side": contract_side.name,
                    "quantity": quantity,
                    "limit_price": limit_price
                }
            else:
                return {"status": "failed", "reason": "kalshi_api_error"}
        
        except Exception as e:
            logger.error(f"Error executing order for {market_ticker}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def execute_limit_order_action(
        self,
        action: int,
        market_ticker: str,
        orderbook_snapshot: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute limit order action (wrapper for ActorService integration).
        
        This method provides the interface expected by ActorService while
        delegating to execute_order() for actual implementation.
        
        Args:
            action: Action ID (0=HOLD, 1=BUY_YES_LIMIT, 2=SELL_YES_LIMIT, 3=BUY_NO_LIMIT, 4=SELL_NO_LIMIT)
            market_ticker: Market to trade
            orderbook_snapshot: Orderbook snapshot for price calculation
            
        Returns:
            Execution result dict with "executed" key for ActorService compatibility
        """
        result = await self.execute_order(market_ticker, action, orderbook_snapshot)
        
        # Normalize result format for ActorService
        if result is None:
            return None
        
        # Convert status to "executed" boolean for ActorService
        if result.get("status") == "placed":
            result["executed"] = True
        else:
            result["executed"] = False
        
        return result
    
    async def _place_kalshi_order(
        self,
        ticker: str,
        side: OrderSide,
        contract_side: ContractSide,
        quantity: int,
        limit_price: int
    ) -> Optional[Dict[str, Any]]:
        """Place order via Kalshi API and track locally."""
        try:
            # Generate our internal order ID
            our_order_id = self._generate_order_id()
            
            # Convert to Kalshi API format
            kalshi_action = "buy" if side == OrderSide.BUY else "sell"
            kalshi_side = "yes" if contract_side == ContractSide.YES else "no"
            
            # Calculate promised cash (Option B)
            promised_cash = 0.0
            if side == OrderSide.BUY:
                promised_cash = (limit_price / 100.0) * quantity
                # Reserve cash immediately
                self.cash_balance -= promised_cash
                self.promised_cash += promised_cash
            
            # Place order via Kalshi
            response = await self.trading_client.create_order(
                ticker=ticker,
                action=kalshi_action,
                side=kalshi_side,
                count=quantity,
                price=limit_price,
                type="limit"
            )
            
            # Extract Kalshi order ID
            kalshi_order_id = response.get("order", {}).get("order_id", "")
            if not kalshi_order_id:
                # Restore cash on failure
                if side == OrderSide.BUY:
                    self.cash_balance += promised_cash
                    self.promised_cash -= promised_cash
                logger.error(f"No Kalshi order ID in response: {response}")
                return None
            
            # Create order tracking
            order_info = OrderInfo(
                order_id=our_order_id,
                kalshi_order_id=kalshi_order_id,
                ticker=ticker,
                side=side,
                contract_side=contract_side,
                quantity=quantity,
                limit_price=limit_price,
                status=OrderStatus.PENDING,
                placed_at=time.time(),
                promised_cash=promised_cash
            )
            
            # Track order
            self.open_orders[our_order_id] = order_info
            self._kalshi_to_internal[kalshi_order_id] = our_order_id
            
            logger.debug(f"Order tracked: {our_order_id} -> {kalshi_order_id}")
            
            return {
                "order_id": our_order_id,
                "kalshi_order_id": kalshi_order_id,
                "status": "placed"
            }
            
        except Exception as e:
            # Restore cash on error
            if side == OrderSide.BUY and 'promised_cash' in locals():
                self.cash_balance += promised_cash
                self.promised_cash -= promised_cash
            logger.error(f"Error placing Kalshi order: {e}")
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        if order_id not in self.open_orders:
            logger.warning(f"Order not found for cancellation: {order_id}")
            return False
        
        order = self.open_orders[order_id]
        
        try:
            # Cancel via Kalshi API
            await self.trading_client.cancel_order(order.kalshi_order_id)
            
            # Update local state (Option B cash management)
            if order.side == OrderSide.BUY:
                # Restore promised cash
                self.cash_balance += order.promised_cash
                self.promised_cash -= order.promised_cash
            
            # Remove from tracking
            order.status = OrderStatus.CANCELLED
            del self.open_orders[order_id]
            if order.kalshi_order_id in self._kalshi_to_internal:
                del self._kalshi_to_internal[order.kalshi_order_id]
            
            self._orders_cancelled += 1
            logger.info(f"Order cancelled: {order_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """Cancel all open orders, optionally filtered by ticker."""
        orders_to_cancel = []
        
        for order in self.open_orders.values():
            if ticker is None or order.ticker == ticker:
                orders_to_cancel.append(order.order_id)
        
        cancelled_count = 0
        for order_id in orders_to_cancel:
            if await self.cancel_order(order_id):
                cancelled_count += 1
        
        logger.info(f"Cancelled {cancelled_count} orders" + (f" for {ticker}" if ticker else ""))
        return cancelled_count
    
    async def queue_fill(self, kalshi_fill_message: Dict[str, Any]) -> None:
        """
        Queue a fill event for processing.
        
        This is called by external fill listeners (e.g., WebSocket handlers)
        to queue fills for sequential processing.
        """
        fill_event = FillEvent.from_kalshi_message(kalshi_fill_message)
        if fill_event:
            await self.fills_queue.put(fill_event)
            logger.debug(f"Fill queued: {fill_event.kalshi_order_id}")
    
    async def _process_fills(self) -> None:
        """Process fill events from the queue sequentially."""
        logger.info("Fill processor started")
        
        try:
            while True:
                # Wait for fill event
                fill_event = await self.fills_queue.get()
                
                try:
                    await self._process_single_fill(fill_event)
                except Exception as e:
                    logger.error(f"Error processing fill {fill_event.kalshi_order_id}: {e}")
                finally:
                    self.fills_queue.task_done()
                    
        except asyncio.CancelledError:
            logger.info("Fill processor cancelled")
            raise
        except Exception as e:
            logger.error(f"Fill processor error: {e}")
    
    async def _process_single_fill(self, fill_event: FillEvent) -> None:
        """Process a single fill event."""
        kalshi_order_id = fill_event.kalshi_order_id
        
        # Find our order
        if kalshi_order_id not in self._kalshi_to_internal:
            logger.warning(f"Fill for unknown order: {kalshi_order_id}")
            return
        
        our_order_id = self._kalshi_to_internal[kalshi_order_id]
        
        if our_order_id not in self.open_orders:
            logger.warning(f"Fill for non-open order: {our_order_id}")
            return
        
        order = self.open_orders[our_order_id]
        
        # Process the fill
        fill_cost = (fill_event.fill_price / 100.0) * fill_event.fill_quantity
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_at = fill_event.fill_timestamp
        order.fill_price = fill_event.fill_price
        
        # Option B cash management
        if order.side == OrderSide.BUY:
            # Cash was already deducted when order placed
            # Just reduce promised cash
            self.promised_cash -= order.promised_cash
        else:
            # SELL: add cash received
            self.cash_balance += fill_cost
        
        # Update position
        self._update_position(order, fill_event.fill_price, fill_event.fill_quantity)
        
        # Remove from tracking
        del self.open_orders[our_order_id]
        del self._kalshi_to_internal[kalshi_order_id]
        
        # Update metrics
        self._orders_filled += 1
        self._total_volume_traded += fill_cost
        
        logger.info(f"Fill processed: {our_order_id} - {fill_event.fill_quantity} @ {fill_event.fill_price}¢")
    
    def _update_position(self, order: OrderInfo, fill_price: int, fill_quantity: int) -> None:
        """Update position after a fill."""
        ticker = order.ticker
        fill_cost = (fill_price / 100.0) * fill_quantity
        
        # Ensure position exists
        if ticker not in self.positions:
            self.positions[ticker] = Position(
                ticker=ticker,
                contracts=0,
                cost_basis=0.0,
                realized_pnl=0.0
            )
        
        position = self.positions[ticker]
        
        # Calculate position change (Kalshi convention)
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
        
        # Check if reducing position (realize P&L)
        if (position.contracts > 0 and contract_change < 0) or \
           (position.contracts < 0 and contract_change > 0):
            # Position reduction
            reduction_amount = min(abs(contract_change), abs(position.contracts))
            
            if position.contracts != 0:
                avg_cost_per_contract = position.cost_basis / abs(position.contracts)
                
                if order.side == OrderSide.SELL:
                    realized_pnl = reduction_amount * (fill_price / 100.0 - avg_cost_per_contract)
                else:
                    # Buying opposite side to close
                    if position.contracts > 0:
                        # Closing YES with NO purchase
                        realized_pnl = reduction_amount * ((100 - fill_price) / 100.0 - avg_cost_per_contract)
                    else:
                        # Closing NO with YES purchase
                        realized_pnl = reduction_amount * (avg_cost_per_contract - fill_price / 100.0)
                
                position.realized_pnl += realized_pnl
                
                # Update cost basis proportionally
                remaining_contracts = abs(position.contracts) - reduction_amount
                if remaining_contracts > 0:
                    position.cost_basis *= remaining_contracts / abs(position.contracts)
                else:
                    position.cost_basis = 0.0
        else:
            # Position increase
            position.cost_basis += fill_cost
        
        # Update contract count
        position.contracts += contract_change
    
    def get_order_features(self, market_ticker: str) -> Dict[str, float]:
        """Get order-related features for RL observations."""
        open_orders = [o for o in self.open_orders.values() if o.ticker == market_ticker]
        
        has_open_buy = float(any(o.side == OrderSide.BUY for o in open_orders))
        has_open_sell = float(any(o.side == OrderSide.SELL for o in open_orders))
        
        # Time since most recent order (normalized to [0,1])
        time_since_order = 0.0
        if open_orders:
            most_recent_time = max(order.placed_at for order in open_orders)
            time_elapsed = time.time() - most_recent_time
            time_since_order = min(time_elapsed / 300.0, 1.0)  # 5 minutes max
        
        return {
            "has_open_buy": has_open_buy,
            "has_open_sell": has_open_sell,
            "time_since_order": time_since_order
        }
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        total_value = self.cash_balance
        
        # Add position values (would need current prices for mark-to-market)
        for position in self.positions.values():
            if not position.is_flat:
                # For now, use cost basis (would need market data for unrealized P&L)
                total_value += position.cost_basis
        
        return total_value
    
    def get_cash_balance(self) -> float:
        """Get available cash balance."""
        return self.cash_balance
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all positions in UnifiedPositionTracker format."""
        result = {}
        
        for ticker, position in self.positions.items():
            if not position.is_flat:
                result[ticker] = {
                    'position': position.contracts,
                    'cost_basis': int(position.cost_basis * 100),  # Convert to cents
                    'realized_pnl': int(position.realized_pnl * 100)  # Convert to cents
                }
        
        return result
    
    def get_open_orders(self, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders."""
        orders = []
        
        for order in self.open_orders.values():
            if ticker is None or order.ticker == ticker:
                orders.append({
                    'order_id': order.order_id,
                    'ticker': order.ticker,
                    'side': order.side.name,
                    'contract_side': order.contract_side.name,
                    'quantity': order.quantity,
                    'limit_price': order.limit_price,
                    'placed_at': order.placed_at
                })
        
        return orders
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return {
            "cash_balance": self.cash_balance,
            "promised_cash": self.promised_cash,
            "portfolio_value": self.get_portfolio_value(),
            "orders_placed": self._orders_placed,
            "orders_filled": self._orders_filled,
            "orders_cancelled": self._orders_cancelled,
            "fill_rate": self._orders_filled / max(self._orders_placed, 1),
            "open_orders_count": len(self.open_orders),
            "positions_count": len([p for p in self.positions.values() if not p.is_flat]),
            "total_volume_traded": self._total_volume_traded,
            "fill_queue_size": self.fills_queue.qsize()
        }
    
    def _calculate_limit_price_from_snapshot(
        self,
        orderbook_snapshot: Optional[Dict[str, Any]],
        side: OrderSide,
        contract_side: ContractSide
    ) -> int:
        """
        Calculate limit price from orderbook snapshot.
        
        Uses aggressive pricing strategy:
        - BUY orders: Use best ask (aggressive - take liquidity)
        - SELL orders: Use best bid (aggressive - take liquidity)
        
        Falls back to mid-market (50) if no orderbook data available.
        
        Args:
            orderbook_snapshot: Orderbook snapshot dict with yes_bids, yes_asks, etc.
            side: BUY or SELL
            contract_side: YES or NO
            
        Returns:
            Limit price in cents (1-99)
        """
        if not orderbook_snapshot:
            logger.warning("No orderbook snapshot provided, using default mid-market price (50)")
            return 50
        
        try:
            if contract_side == ContractSide.YES:
                # Get YES prices
                yes_bids = orderbook_snapshot.get('yes_bids', {})
                yes_asks = orderbook_snapshot.get('yes_asks', {})
                
                if side == OrderSide.BUY:
                    # BUY YES: use best ask (aggressive - take liquidity)
                    if yes_asks:
                        best_ask = min(map(int, yes_asks.keys()))
                        return best_ask
                else:
                    # SELL YES: use best bid (aggressive - take liquidity)
                    if yes_bids:
                        best_bid = max(map(int, yes_bids.keys()))
                        return best_bid
                
                # Fallback to mid-market if no prices available
                if yes_bids and yes_asks:
                    best_bid = max(map(int, yes_bids.keys()))
                    best_ask = min(map(int, yes_asks.keys()))
                    return int((best_bid + best_ask) / 2)
                    
            else:  # ContractSide.NO
                # For NO contracts, derive from YES prices
                yes_bids = orderbook_snapshot.get('yes_bids', {})
                yes_asks = orderbook_snapshot.get('yes_asks', {})
                
                if yes_bids and yes_asks:
                    yes_best_bid = max(map(int, yes_bids.keys()))
                    yes_best_ask = min(map(int, yes_asks.keys()))
                    
                    if side == OrderSide.BUY:
                        # BUY NO: NO ask = 99 - YES bid (aggressive - take liquidity)
                        return 99 - yes_best_bid
                    else:
                        # SELL NO: NO bid = 99 - YES ask (aggressive - take liquidity)
                        return 99 - yes_best_ask
                
                # Fallback to mid-market if no YES prices available
                if yes_bids and yes_asks:
                    yes_best_bid = max(map(int, yes_bids.keys()))
                    yes_best_ask = min(map(int, yes_asks.keys()))
                    yes_mid = (yes_best_bid + yes_best_ask) / 2
                    no_mid = 99 - yes_mid
                    return int(no_mid)
            
            # Default fallback
            logger.warning("Could not calculate limit price from orderbook, using default (50)")
            return 50
            
        except Exception as e:
            logger.error(f"Error calculating limit price: {e}, using default (50)")
            return 50
    
    def _generate_order_id(self) -> str:
        """Generate unique internal order ID."""
        self._order_counter += 1
        return f"order_{self._order_counter}_{int(time.time() * 1000)}"