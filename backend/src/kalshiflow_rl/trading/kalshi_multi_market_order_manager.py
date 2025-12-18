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
from typing import Dict, Any, Optional, List, Callable, Deque
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque

from .demo_client import KalshiDemoTradingClient
from ..config import config

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
    quantity: int                    # Remaining quantity (updated on partial fills)
    limit_price: int                # Price in cents (1-99)
    status: OrderStatus
    placed_at: float
    promised_cash: float            # Cash reserved for remaining order quantity
    original_quantity: Optional[int] = None  # Original order quantity (for partial fill tracking)
    filled_at: Optional[float] = None
    fill_price: Optional[int] = None
    model_decision: int = 0          # Original action decision from model (0-4)
    
    def __post_init__(self):
        """Set original_quantity if not provided."""
        if self.original_quantity is None:
            self.original_quantity = self.quantity
    
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status == OrderStatus.PENDING


@dataclass
class Position:
    """Position in a specific market using Kalshi convention."""
    ticker: str
    contracts: int       # +contracts for YES, -contracts for NO
    cost_basis: float    # Total cost in dollars (position_cost from Kalshi)
    realized_pnl: float  # Cumulative realized P&L
    kalshi_data: Dict[str, Any] = field(default_factory=dict)  # Raw Kalshi API response
    
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
class TradeDetail:
    """
    Individual trade detail for execution history.

    This is broadcast via WebSocket for frontend display.
    """
    trade_id: str               # Unique trade identifier
    timestamp: float            # Unix timestamp of execution
    ticker: str
    action: str                 # "BUY_YES", "SELL_YES", "BUY_NO", "SELL_NO" 
    quantity: int
    fill_price: int             # Fill price in cents
    order_id: str               # Our internal order ID
    model_decision: int         # Original action decision from model (0-4)


@dataclass
class ExecutionStats:
    """
    Aggregate execution statistics.
    """
    total_fills: int = 0
    maker_fills: int = 0        # Fills from passive orders
    taker_fills: int = 0        # Fills from aggressive orders
    avg_fill_time_ms: float = 0.0  # Average time from order to fill
    total_volume: float = 0.0   # Total volume traded in dollars


@dataclass
class FillEvent:
    """
    Fill event for processing queue.
    
    Based on Kalshi User Fills WebSocket message format:
    https://docs.kalshi.com/websockets/user-fills
    """
    kalshi_order_id: str
    fill_price: int              # YES price in cents (1-99)
    fill_quantity: int           # Number of contracts filled (count)
    fill_timestamp: float        # Unix timestamp
    market_ticker: str = ""      # Market ticker from fill message
    post_position: Optional[int] = None  # Position after fill (from Kalshi - authoritative!)
    action: str = ""             # "buy" or "sell"
    side: str = ""               # "yes" or "no"
    is_taker: bool = False       # Whether fill was aggressive (taker) or passive (maker)
    
    @classmethod
    def from_kalshi_message(cls, message: Dict[str, Any]) -> Optional['FillEvent']:
        """
        Create fill event from Kalshi WebSocket message.
        
        Note: Kalshi fill messages use 'msg' key, NOT 'data'!
        
        Example message:
        {
            "type": "fill",
            "sid": 13,
            "msg": {
                "trade_id": "...",
                "order_id": "...",
                "market_ticker": "HIGHNY-22DEC23-B53.5",
                "is_taker": true,
                "side": "yes",
                "yes_price": 75,
                "count": 278,
                "action": "buy",
                "ts": 1671899397,
                "post_position": 500
            }
        }
        """
        try:
            # IMPORTANT: Kalshi uses 'msg' not 'data'!
            fill_data = message.get("msg", {})
            
            if not fill_data:
                logger.warning(f"Empty fill data in message: {message}")
                return None
            
            # Parse timestamp - Kalshi uses 'ts' as Unix timestamp
            ts = fill_data.get("ts")
            fill_timestamp = float(ts) if ts else time.time()
            
            return cls(
                kalshi_order_id=fill_data.get("order_id", ""),
                fill_price=fill_data.get("yes_price", 0),
                fill_quantity=fill_data.get("count", 0),
                fill_timestamp=fill_timestamp,
                market_ticker=fill_data.get("market_ticker", ""),
                post_position=fill_data.get("post_position"),  # Can be None
                action=fill_data.get("action", ""),
                side=fill_data.get("side", ""),
                is_taker=fill_data.get("is_taker", False),
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
        
        # Fill listener (WebSocket connection for real-time fill notifications)
        self._fill_listener = None  # Type: FillListener (imported lazily)
        
        # Periodic sync task
        self._periodic_sync_task: Optional[asyncio.Task] = None
        
        # Trading client
        self.trading_client: Optional[KalshiDemoTradingClient] = None
        
        # Monitoring
        self._orders_placed = 0
        self._orders_filled = 0
        self._orders_cancelled = 0
        self._total_volume_traded = 0.0
        
        # State change callbacks (for UI updates)
        self._state_change_callbacks: List[Callable] = []
        
        # Execution history tracking (maxlen=100)
        self.execution_history: Deque[TradeDetail] = deque(maxlen=100)
        self.execution_stats = ExecutionStats()
        
        # Broadcast callbacks for WebSocket updates
        self._trade_broadcast_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # WebSocket manager for specific event broadcasts
        self._websocket_manager = None  # Will be set after initialization
        
        # Market configuration (from config)
        self._market_tickers = config.RL_MARKET_TICKERS
        self._market_count = len(self._market_tickers)
        
        logger.info(f"KalshiMultiMarketOrderManager initialized with ${initial_cash:.2f}")
    
    async def initialize(self) -> None:
        """
        Initialize the order manager and start fill processing.
        
        Requires valid Kalshi API credentials. Validates credentials before
        attempting connection and fails fast if missing.
        
        Raises:
            KalshiDemoAuthError: If credentials are missing or invalid
            Exception: If client connection fails
        """
        logger.info("Initializing KalshiMultiMarketOrderManager...")
        
        # Initialize action selector tracking
        self.action_selector = None
        
        # Validate credentials are available before attempting connection
        from ..config import config
        from .demo_client import KalshiDemoAuthError
        
        if not config.KALSHI_API_KEY_ID:
            raise KalshiDemoAuthError(
                "KALSHI_API_KEY_ID not configured. OrderManager requires valid credentials to function."
            )
        if not config.KALSHI_PRIVATE_KEY_CONTENT:
            raise KalshiDemoAuthError(
                "KALSHI_PRIVATE_KEY_CONTENT not configured. OrderManager requires valid credentials to function."
            )
        
        # Initialize trading client (will validate credentials again internally)
        self.trading_client = KalshiDemoTradingClient()
        await self.trading_client.connect()
        logger.info("✅ Demo trading client connected")
        
        # Start fill processor only after successful client connection
        self._fill_processor_task = asyncio.create_task(self._process_fills())
        logger.info("✅ Fill processor started")
        
        # Start fill listener (WebSocket for real-time fill notifications)
        try:
            from .fill_listener import FillListener
            self._fill_listener = FillListener(order_manager=self)
            await self._fill_listener.start()
            logger.info("✅ Fill listener started (WebSocket connected)")
        except Exception as e:
            # Fill listener is critical but we can fall back to periodic sync
            logger.warning(f"⚠️ Fill listener failed to start: {e}. Using periodic sync as fallback.")
            self._fill_listener = None
        
        # Check if cleanup is enabled BEFORE syncing (defaults to true)
        import os
        from ..config import config
        cleanup_enabled = os.getenv("RL_CLEANUP_ON_START", "true").lower() == "true"
        
        if cleanup_enabled:
            # Clean up any leftover orders/positions BEFORE syncing
            logger.info("Running startup cleanup to reset trader state...")
            cleanup_summary = await self.cleanup_orders_and_positions()
            
            if cleanup_summary.get("cleanup_success"):
                logger.info("✅ Startup cleanup completed successfully")
                # Broadcast cleanup summary to any connected WebSocket clients
                await self.broadcast_cleanup_summary(cleanup_summary)
            else:
                warnings = cleanup_summary.get("warnings", [])
                logger.warning(f"⚠️ Startup cleanup had issues: {'; '.join(warnings)}")
                # Still broadcast the summary so UI can show warnings
                await self.broadcast_cleanup_summary(cleanup_summary)
        else:
            logger.info("Startup cleanup disabled via RL_CLEANUP_ON_START=false")
        
        # Synchronize orders and positions with Kalshi on startup (if enabled)
        # This happens AFTER cleanup so we get fresh state
        if config.RL_ORDER_SYNC_ENABLED and config.RL_ORDER_SYNC_ON_STARTUP:
            logger.info("Starting order synchronization with Kalshi...")
            try:
                order_stats = await self.sync_orders_with_kalshi(is_startup=True)
                
                # Log summary (warnings for actual discrepancies are already logged by sync_orders_with_kalshi)
                # On startup, finding orders in Kalshi but not locally is expected, not a discrepancy
                actual_discrepancies = order_stats.get("discrepancies", 0) + order_stats.get("removed", 0)
                if actual_discrepancies == 0:
                    logger.info(
                        f"✅ Startup order sync complete: {order_stats['found_in_kalshi']} orders in Kalshi, "
                        f"{order_stats['found_in_memory']} orders in local memory - all in sync"
                    )
                # If there are actual discrepancies, they're already logged as warnings by sync_orders_with_kalshi
                
                # Sync positions
                await self._sync_positions_with_kalshi(is_startup=True)
                # Position sync messages are already logged by _sync_positions_with_kalshi
            except Exception as e:
                logger.error(f"Error during startup sync: {e}")
                logger.warning("Continuing without sync - local state may be out of sync")
        
        # Start periodic synchronization (if enabled)
        if config.RL_ORDER_SYNC_ENABLED:
            self._periodic_sync_task = asyncio.create_task(self._start_periodic_sync())
            logger.info(f"✅ Periodic sync started (interval: {config.RL_ORDER_SYNC_INTERVAL_SECONDS}s)")
        
        logger.info("✅ KalshiMultiMarketOrderManager ready for trading")
        
        # Notify about initial state
        await self._notify_state_change()
    
    async def cleanup_orders_and_positions(self) -> Dict[str, Any]:
        """
        Cancel all open orders on startup. Note: Positions cannot be cancelled, only closed via trades.
        
        This method is called during actor startup to clean up resting orders from previous sessions.
        It:
        1. Fetches all open orders from Kalshi API
        2. Batch cancels all resting orders to recover reserved cash
        3. Reports on existing positions (but cannot close them without cash)
        4. Returns summary for frontend display
        
        IMPORTANT: This does NOT close positions. Positions require opposite trades to close,
        which requires available cash. Positions will remain open until manually closed or
        markets settle.
        
        Returns:
            Dictionary with cleanup summary for UI display
        """
        if not self.trading_client:
            logger.error("Cannot cleanup: trading client not initialized")
            return {"error": "Trading client not initialized"}
        
        cleanup_summary = {
            "orders_before": 0,
            "orders_cancelled": 0,
            "orders_failed": 0,
            "cash_before": self.cash_balance,
            "cash_after": 0.0,
            "cash_recovered": 0.0,
            "positions_before": 0,
            "positions_after": 0,
            "cleanup_success": False,
            "warnings": []
        }
        
        try:
            logger.info("Starting cleanup of open orders (positions cannot be cancelled)...")
            
            # Step 1: Fetch all orders and positions for baseline
            orders_response = await self.trading_client.get_orders()
            initial_orders = orders_response.get("orders", [])
            cleanup_summary["orders_before"] = len(initial_orders)
            
            positions_response = await self.trading_client.get_positions()
            initial_positions = positions_response.get("positions", [])
            non_zero_positions = [p for p in initial_positions if p.get("position", 0) != 0]
            cleanup_summary["positions_before"] = len(non_zero_positions)
            
            logger.info(f"Found {len(initial_orders)} open orders and {len(non_zero_positions)} positions")
            
            # Early return if no orders to cancel
            if not initial_orders:
                logger.info(f"✅ No orders to cancel. {len(non_zero_positions)} positions remain open (require trades to close)")
                cleanup_summary["cleanup_success"] = True
                cleanup_summary["cash_after"] = self.cash_balance
                cleanup_summary["positions_after"] = len(non_zero_positions)
                if non_zero_positions:
                    cleanup_summary["warnings"].append(f"{len(non_zero_positions)} positions remain open (insufficient cash to close)")
                return cleanup_summary
            
            # Step 2: Batch cancel all orders (if any)
            cancelled_orders = []
            failed_cancellations = []
            
            if initial_orders:
                logger.info(f"Batch cancelling {len(initial_orders)} orders...")
                
                # Extract order IDs
                order_ids = [order.get("order_id", "") for order in initial_orders if order.get("order_id")]
                
                if order_ids:
                    try:
                        # Use batch cancel if available
                        if hasattr(self.trading_client, 'batch_cancel_orders'):
                            batch_result = await self.trading_client.batch_cancel_orders(order_ids)
                            cancelled_orders = batch_result.get("cancelled", [])
                            failed_cancellations = batch_result.get("errors", [])
                            
                            logger.info(f"Batch cancel complete: {len(cancelled_orders)} cancelled, {len(failed_cancellations)} failed")
                        else:
                            # Fallback to individual cancellations
                            logger.info("Batch cancel not available, using individual cancellations...")
                            for order_id in order_ids:
                                try:
                                    await self.trading_client.cancel_order(order_id)
                                    cancelled_orders.append(order_id)
                                except Exception as e:
                                    failed_cancellations.append({"order_id": order_id, "error": str(e)})
                            
                            logger.info(f"Individual cancellations complete: {len(cancelled_orders)} cancelled, {len(failed_cancellations)} failed")
                    
                    except Exception as e:
                        logger.error(f"Error during order cancellation: {e}")
                        cleanup_summary["warnings"].append(f"Order cancellation failed: {e}")
            
            cleanup_summary["orders_cancelled"] = len(cancelled_orders)
            cleanup_summary["orders_failed"] = len(failed_cancellations)
            
            # Step 3: Wait a moment for order cancellations to settle
            if cancelled_orders:
                logger.info("Waiting 2 seconds for order cancellations to settle...")
                await asyncio.sleep(2.0)
            
            # Step 4: Re-sync with Kalshi to verify cleanup and update cash balance
            logger.info("Re-syncing with Kalshi to verify cleanup...")
            
            # Sync positions first (affects cash calculation)
            await self._sync_positions_with_kalshi(is_startup=True)
            
            # Sync orders to ensure cleanup was successful
            sync_stats = await self.sync_orders_with_kalshi(is_startup=True)
            final_orders = sync_stats.get("found_in_kalshi", 0)
            
            # Update final metrics
            cleanup_summary["cash_after"] = self.cash_balance
            cleanup_summary["cash_recovered"] = cleanup_summary["cash_after"] - cleanup_summary["cash_before"]
            cleanup_summary["positions_after"] = len([p for p in self.positions.values() if not p.is_flat])
            
            # Check if cleanup was successful
            if final_orders == 0:
                cleanup_summary["cleanup_success"] = True
                logger.info(f"✅ Cleanup successful: cancelled {len(cancelled_orders)} orders, recovered ${cleanup_summary['cash_recovered']:.2f}")
            else:
                cleanup_summary["warnings"].append(f"Still have {final_orders} orders after cleanup")
                logger.warning(f"⚠️ Cleanup incomplete: {final_orders} orders remain after cancellation")
            
            # Warn if cash is low
            if cleanup_summary["cash_after"] < 500.0:
                warning_msg = f"Low cash balance after cleanup: ${cleanup_summary['cash_after']:.2f} (may need manual funding)"
                cleanup_summary["warnings"].append(warning_msg)
                logger.warning(f"⚠️ {warning_msg}")
            
            # Log summary
            logger.info(
                f"Cleanup summary: {cleanup_summary['orders_cancelled']} orders cancelled, "
                f"{cleanup_summary['positions_after']} positions remain open (cannot close without cash), "
                f"${cleanup_summary['cash_after']:.2f} cash balance"
            )
            
            # Add warning about open positions if any exist
            if cleanup_summary["positions_after"] > 0:
                cleanup_summary["warnings"].append(
                    f"{cleanup_summary['positions_after']} positions remain open. "
                    f"Closing requires opposite trades (need cash to place orders)"
                )
            
            return cleanup_summary
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            cleanup_summary["warnings"].append(f"Cleanup failed: {e}")
            cleanup_summary["cleanup_success"] = False
            cleanup_summary["cash_after"] = self.cash_balance
            return cleanup_summary
    
    async def broadcast_cleanup_summary(self, cleanup_summary: Dict[str, Any]) -> None:
        """
        Broadcast cleanup summary to frontend via WebSocket.
        
        Args:
            cleanup_summary: Dictionary from cleanup_orders_and_positions()
        """
        try:
            # Format message for UI display
            if cleanup_summary.get("cleanup_success"):
                if cleanup_summary["orders_before"] > 0:
                    message = (
                        f"✅ Order cleanup complete: Cancelled {cleanup_summary['orders_cancelled']} orders, "
                        f"recovered ${cleanup_summary['cash_recovered']:.2f} cash. "
                    )
                    if cleanup_summary.get("positions_after", 0) > 0:
                        message += f" Note: {cleanup_summary['positions_after']} positions remain open (need cash to close)."
                    message += f" Cash balance: ${cleanup_summary['cash_after']:.2f}"
                else:
                    if cleanup_summary.get("positions_after", 0) > 0:
                        message = (
                            f"✅ No orders to cancel. "
                            f"Note: {cleanup_summary['positions_after']} positions remain open (need cash to close). "
                            f"Cash balance: ${cleanup_summary['cash_after']:.2f}"
                        )
                    else:
                        message = f"✅ Clean state: No orders or positions. Cash balance: ${cleanup_summary['cash_after']:.2f}"
            else:
                message = f"⚠️ Cleanup issues: {'; '.join(cleanup_summary.get('warnings', ['Unknown error']))}"
            
            # Create WebSocket message
            cleanup_message = {
                "type": "cleanup_summary",
                "data": {
                    "timestamp": time.time(),
                    "summary": cleanup_summary,
                    "message": message,
                    "success": cleanup_summary.get("cleanup_success", False)
                }
            }
            
            # Broadcast via state change callbacks (includes WebSocket manager)
            for callback in self._state_change_callbacks:
                try:
                    await callback(cleanup_message)
                except Exception as e:
                    logger.error(f"Error broadcasting cleanup summary: {e}")
            
            logger.info(f"Cleanup summary broadcast: {message}")
            
        except Exception as e:
            logger.error(f"Error formatting cleanup summary for broadcast: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown the order manager."""
        logger.info("Shutting down KalshiMultiMarketOrderManager...")
        
        # Cancel all open orders (only if trading client is initialized)
        if self.trading_client is not None:
            await self.cancel_all_orders()
        
        # Stop fill listener first (stop receiving new fills)
        if self._fill_listener:
            try:
                await self._fill_listener.stop()
                logger.info("✅ Fill listener stopped")
            except Exception as e:
                logger.warning(f"Error stopping fill listener: {e}")
            self._fill_listener = None
        
        # Stop periodic sync task
        if self._periodic_sync_task:
            self._periodic_sync_task.cancel()
            try:
                await self._periodic_sync_task
            except asyncio.CancelledError:
                pass
        
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
            
            # Fixed contract size matching session 12 training (20 contracts)
            quantity = 20
            
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
                
                # Store model decision for trade tracking
                order_id = result["order_id"]
                if order_id in self.open_orders:
                    self.open_orders[order_id].model_decision = action
                
                await self._notify_state_change()  # Notify about order placement
                await self._broadcast_orders_update("order_placed")  # Specific orders update
                await self._broadcast_portfolio_update("order_placed")  # Portfolio update due to promised cash
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
            await self._notify_state_change()  # Notify about order cancellation
            await self._broadcast_orders_update("order_cancelled")  # Specific orders update
            await self._broadcast_portfolio_update("order_cancelled")  # Portfolio update due to released cash
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
        """
        Process a single fill event.
        
        Handles both partial and complete fills:
        - Partial fills: Update order quantity, release proportional cash, keep order open
        - Complete fills: Remove order from tracking
        
        Uses post_position from Kalshi fill message for accurate position tracking.
        """
        kalshi_order_id = fill_event.kalshi_order_id
        fill_quantity = fill_event.fill_quantity
        fill_price = fill_event.fill_price
        
        # Find our order
        if kalshi_order_id not in self._kalshi_to_internal:
            logger.warning(f"Fill for unknown order: {kalshi_order_id}")
            return
        
        our_order_id = self._kalshi_to_internal[kalshi_order_id]
        
        if our_order_id not in self.open_orders:
            logger.warning(f"Fill for non-open order: {our_order_id}")
            return
        
        order = self.open_orders[our_order_id]
        
        # Calculate fill cost
        fill_cost = (fill_price / 100.0) * fill_quantity
        
        # Calculate remaining quantity after this fill
        remaining_quantity = order.quantity - fill_quantity
        
        # Determine if this is a partial or complete fill
        is_partial_fill = remaining_quantity > 0
        
        # Option B cash management - PROPORTIONAL release for partial fills
        if order.side == OrderSide.BUY:
            # Calculate proportional cash to release based on fill ratio
            # Use order.quantity (remaining before this fill) for ratio calculation
            fill_ratio = fill_quantity / order.quantity if order.quantity > 0 else 1.0
            promised_cash_released = order.promised_cash * fill_ratio
            
            # Release proportional promised cash
            self.promised_cash -= promised_cash_released
            order.promised_cash -= promised_cash_released
            
            logger.debug(
                f"BUY fill cash: released ${promised_cash_released:.2f} of ${order.promised_cash + promised_cash_released:.2f} "
                f"({fill_quantity}/{order.quantity} contracts)"
            )
        else:
            # SELL: add cash received for the filled quantity
            self.cash_balance += fill_cost
            logger.debug(f"SELL fill cash: received ${fill_cost:.2f}")
        
        # Update position using traditional method first
        self._update_position(order, fill_price, fill_quantity)
        
        # If post_position is provided by Kalshi, use it as authoritative source
        # This eliminates race conditions and ensures position accuracy
        if fill_event.post_position is not None:
            ticker = fill_event.market_ticker or order.ticker
            if ticker in self.positions:
                old_contracts = self.positions[ticker].contracts
                self.positions[ticker].contracts = fill_event.post_position
                if old_contracts != fill_event.post_position:
                    logger.debug(
                        f"Position corrected via post_position: {ticker} "
                        f"{old_contracts} -> {fill_event.post_position}"
                    )
        
        # Update order tracking based on fill type
        if is_partial_fill:
            # Partial fill: Update order quantity, keep order open
            order.quantity = remaining_quantity
            order.fill_price = fill_price  # Track most recent fill price
            
            logger.info(
                f"Partial fill processed: {our_order_id} - {fill_quantity} @ {fill_price}¢ "
                f"(remaining: {remaining_quantity} contracts)"
            )
        else:
            # Complete fill: Update status and remove from tracking
            order.status = OrderStatus.FILLED
            order.filled_at = fill_event.fill_timestamp
            order.fill_price = fill_price
            order.quantity = 0  # Mark as fully filled
            
            # Remove from tracking
            del self.open_orders[our_order_id]
            del self._kalshi_to_internal[kalshi_order_id]
            
            logger.info(
                f"Fill complete: {our_order_id} - {fill_quantity} @ {fill_price}¢ "
                f"(order fully filled)"
            )
        
        # Track the fill in execution history
        await self._track_fill(
            order=order,
            fill_price=fill_price,
            fill_quantity=fill_quantity,
            fill_timestamp=fill_event.fill_timestamp,
            is_taker=fill_event.is_taker
        )
        
        # Update metrics
        self._orders_filled += 1
        self._total_volume_traded += fill_cost
        
        # Add debug logging
        logger.debug(f"Fill processed: {order.ticker} {order.side.name} {fill_quantity} @ {fill_price}¢")
        logger.debug(f"Portfolio after fill - Cash: ${self.cash_balance:.2f}, Positions: {len(self.positions)}")
        
        # Broadcast trades update
        await self._broadcast_trades()
        
        # Broadcast specific fill event and position updates
        fill_data = {
            "fill": {
                "kalshi_order_id": fill_event.kalshi_order_id,
                "fill_quantity": fill_event.fill_quantity,
                "fill_price": fill_event.fill_price,
                "fill_timestamp": fill_event.fill_timestamp,
                "is_taker": fill_event.is_taker,
                "market_ticker": fill_event.market_ticker
            },
            "updated_position": {
                "ticker": order.ticker,
                "contracts": self.positions.get(order.ticker).contracts if order.ticker in self.positions else 0,
                "average_cost_cents": self.positions.get(order.ticker).average_cost_cents if order.ticker in self.positions else 0
            }
        }
        await self._broadcast_fill_event(fill_data)
        await self._broadcast_positions_update("fill_processed")
        await self._broadcast_portfolio_update("fill_processed")
        
        # Notify about state change after fill
        await self._notify_state_change()
    
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
    
    async def _track_fill(
        self,
        order: OrderInfo,
        fill_price: int,
        fill_quantity: int,
        fill_timestamp: float,
        is_taker: bool = True
    ) -> None:
        """
        Track fill in execution history and update statistics.

        Args:
            order: Order that was filled
            fill_price: Fill price in cents
            fill_quantity: Number of contracts filled
            fill_timestamp: Timestamp of fill
            is_taker: Whether fill was aggressive (taker) or passive (maker)
        """
        # Generate unique trade ID
        trade_id = f"trade_{int(fill_timestamp * 1000)}_{order.order_id}"

        # Create action string for display
        if order.side == OrderSide.BUY:
            if order.contract_side == ContractSide.YES:
                action_str = "BUY_YES"
            else:
                action_str = "BUY_NO"
        else:
            if order.contract_side == ContractSide.YES:
                action_str = "SELL_YES"
            else:
                action_str = "SELL_NO"

        # Create trade detail
        trade_detail = TradeDetail(
            trade_id=trade_id,
            timestamp=fill_timestamp,
            ticker=order.ticker,
            action=action_str,
            quantity=fill_quantity,
            fill_price=fill_price,
            order_id=order.order_id,
            model_decision=getattr(order, 'model_decision', 0)
        )

        # Add to execution history (deque automatically limits to 100)
        self.execution_history.append(trade_detail)

        # Update execution statistics
        self.execution_stats.total_fills += 1
        if is_taker:
            self.execution_stats.taker_fills += 1
        else:
            self.execution_stats.maker_fills += 1

        # Update average fill time
        if hasattr(order, 'placed_at'):
            fill_time_ms = (fill_timestamp - order.placed_at) * 1000
            # Calculate rolling average
            total_fills = self.execution_stats.total_fills
            current_avg = self.execution_stats.avg_fill_time_ms
            self.execution_stats.avg_fill_time_ms = (
                (current_avg * (total_fills - 1) + fill_time_ms) / total_fills
            )

        # Update total volume
        fill_cost = (fill_price / 100.0) * fill_quantity
        self.execution_stats.total_volume += fill_cost

        logger.debug(f"Tracked fill: {trade_detail.action} {fill_quantity} @ {fill_price}¢ (taker: {is_taker})")
    
    def add_trade_broadcast_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback function to be called when trades are broadcast.

        Args:
            callback: Function that takes trade broadcast data as parameter
        """
        self._trade_broadcast_callbacks.append(callback)
    
    async def _broadcast_trades(self) -> None:
        """
        Broadcast recent trades and observation space data via WebSocket.

        This creates the message format specified in the Pow Wow Master Plan.
        """
        if not self._trade_broadcast_callbacks:
            return

        # Get recent fills (last 20 trades)
        recent_fills = list(self.execution_history)[-20:]
        recent_fills_data = []

        for trade in recent_fills:
            recent_fills_data.append({
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp,
                "ticker": trade.ticker,
                "action": trade.action,
                "quantity": trade.quantity,
                "fill_price": trade.fill_price,
                "order_id": trade.order_id,
                "model_decision": trade.model_decision
            })

        # Create execution stats
        execution_stats_data = {
            "total_fills": self.execution_stats.total_fills,
            "maker_fills": self.execution_stats.maker_fills,
            "taker_fills": self.execution_stats.taker_fills,
            "avg_fill_time_ms": round(self.execution_stats.avg_fill_time_ms, 2),
            "total_volume": round(self.execution_stats.total_volume, 2)
        }

        # Create observation space data (simplified for now - will be enhanced later)
        observation_space_data = {
            "orderbook_features": {
                "spread": {"value": 0.02, "intensity": "medium"},
                "bid_depth": {"value": 0.5, "intensity": "medium"},
                "ask_depth": {"value": 0.5, "intensity": "medium"}
            },
            "market_dynamics": {
                "momentum": {"value": 0.0, "intensity": "low"},
                "volatility": {"value": 0.1, "intensity": "low"},
                "activity": {"value": 0.3, "intensity": "medium"}
            },
            "portfolio_state": {
                "cash_ratio": {"value": self.cash_balance / (self.cash_balance + self.promised_cash + 1), "intensity": "high"},
                "exposure": {"value": len(self.positions) / 10.0, "intensity": "medium" if len(self.positions) < 5 else "high"},
                "risk_level": {"value": min(len(self.open_orders) / 10.0, 1.0), "intensity": "low" if len(self.open_orders) < 3 else "high"}
            }
        }

        # Create broadcast message
        broadcast_data = {
            "type": "trades",
            "data": {
                "recent_fills": recent_fills_data,
                "execution_stats": execution_stats_data,
                "observation_space": observation_space_data
            }
        }

        # Send to all registered callbacks
        for callback in self._trade_broadcast_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(broadcast_data)
                else:
                    callback(broadcast_data)
            except Exception as e:
                logger.error(f"Error in trade broadcast callback: {e}")
    
    async def _broadcast_orders_update(self, event_type: str = "general") -> None:
        """
        Broadcast orders update via WebSocket.
        
        Args:
            event_type: Type of event that triggered the update (e.g., "order_placed", "order_cancelled")
        """
        if not self._websocket_manager:
            return
            
        # Get current orders for broadcast
        orders_data = []
        for order_id, order in self.open_orders.items():
            orders_data.append({
                "order_id": order_id,
                "kalshi_order_id": order.kalshi_order_id,
                "ticker": order.ticker,
                "side": order.side.name,
                "contract_side": order.contract_side.name,
                "quantity": order.quantity,
                "limit_price": order.limit_price,
                "status": order.status.name,
                "placed_at": order.placed_at,
                "promised_cash": order.promised_cash
            })
        
        await self._websocket_manager.broadcast_orders_update(
            {"orders": orders_data},
            source=event_type
        )
        logger.debug(f"Broadcast orders update: {len(orders_data)} orders (event: {event_type})")
    
    async def _broadcast_positions_update(self, event_type: str = "general") -> None:
        """
        Broadcast positions update via WebSocket.
        
        Args:
            event_type: Type of event that triggered the update (e.g., "fill_processed", "position_sync")
        """
        if not self._websocket_manager:
            return
            
        # Calculate total portfolio value
        portfolio_value = self.cash_balance + self.promised_cash
        for position in self.positions.values():
            if hasattr(position, 'market_exposure_dollars') and position.market_exposure_dollars:
                portfolio_value += position.market_exposure_dollars
            else:
                # Fallback calculation if market_exposure_dollars not available
                portfolio_value += (position.average_cost_cents / 100.0) * position.contracts
        
        # Get positions data for broadcast
        positions_data = {}
        for ticker, position in self.positions.items():
            positions_data[ticker] = {
                "contracts": position.contracts,
                "average_cost_cents": position.average_cost_cents,
                "market_exposure_dollars": getattr(position, 'market_exposure_dollars', None),
                "realized_pnl": position.realized_pnl,
                "fees_paid": getattr(position, 'fees_paid', 0.0)
            }
        
        await self._websocket_manager.broadcast_positions_update(
            {
                "positions": positions_data,
                "total_value": portfolio_value
            },
            source=event_type
        )
        logger.debug(f"Broadcast positions update: {len(positions_data)} positions (event: {event_type})")
    
    async def _broadcast_portfolio_update(self, event_type: str = "general") -> None:
        """
        Broadcast portfolio/balance update via WebSocket.
        
        Args:
            event_type: Type of event that triggered the update (e.g., "balance_sync", "order_placed")
        """
        if not self._websocket_manager:
            return
            
        # Calculate portfolio value
        portfolio_value = self.cash_balance + self.promised_cash
        for position in self.positions.values():
            if hasattr(position, 'market_exposure_dollars') and position.market_exposure_dollars:
                portfolio_value += position.market_exposure_dollars
            else:
                portfolio_value += (position.average_cost_cents / 100.0) * position.contracts
        
        await self._websocket_manager.broadcast_portfolio_update({
            "cash_balance": self.cash_balance,
            "portfolio_value": portfolio_value
        })
        logger.debug(f"Broadcast portfolio update: cash={self.cash_balance:.2f}, portfolio={portfolio_value:.2f} (event: {event_type})")
    
    async def _broadcast_fill_event(self, fill_data: Dict[str, Any]) -> None:
        """
        Broadcast fill event notification via WebSocket.
        
        Args:
            fill_data: Fill event data containing fill details and updated position
        """
        if not self._websocket_manager:
            return
            
        await self._websocket_manager.broadcast_fill_event(fill_data)
        logger.debug(f"Broadcast fill event: {fill_data.get('fill', {}).get('kalshi_order_id', 'unknown')}")
    
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
    
    def get_portfolio_value(self, current_prices: Optional[Dict[str, Any]] = None) -> float:
        """Get portfolio value in dollars (cash + position values + unrealized P&L).
        
        Args:
            current_prices: Optional current market prices for unrealized P&L calculation.
                          Format: {ticker: {"bid": float, "ask": float}} in cents
                          If not provided, uses cost basis + realized P&L only
        """
        total = self.cash_balance
        
        # Add position values (cost basis + realized P&L + unrealized P&L)
        for position in self.positions.values():
            if not position.is_flat:
                total += position.cost_basis + position.realized_pnl
                
                # Add unrealized P&L if current prices are provided
                if current_prices and position.ticker in current_prices:
                    price_data = current_prices[position.ticker]
                    if isinstance(price_data, dict) and "bid" in price_data and "ask" in price_data:
                        # Convert cents to probability [0,1] for get_unrealized_pnl
                        yes_bid = price_data["bid"] / 100.0
                        yes_ask = price_data["ask"] / 100.0
                        yes_mid = (yes_bid + yes_ask) / 2.0
                        unrealized_pnl = position.get_unrealized_pnl(yes_mid)
                        total += unrealized_pnl
        
        return total

    def get_portfolio_value_cents(self, current_prices: Dict) -> int:
        """Get portfolio value in cents using bid/ask prices.
        
        Args:
            current_prices: Dictionary with market prices.
                          Format: {ticker: {"bid": float, "ask": float}} in cents
                          
        Returns:
            Total portfolio value in cents including unrealized P&L
        """
        # Use the updated get_portfolio_value method with current prices
        return int(self.get_portfolio_value(current_prices) * 100)

    def get_cash_balance_cents(self) -> int:
        """Get cash balance in cents."""
        return int(self.cash_balance * 100)

    def get_position_info(self) -> Dict[str, Any]:
        """Get position info for environment features."""
        return self.get_positions()

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        metrics = {
            "cash_balance": self.cash_balance,
            "promised_cash": self.promised_cash,
            "portfolio_value": self.get_portfolio_value(),  # Without current_prices for backward compatibility
            "orders_placed": self._orders_placed,
            "orders_filled": self._orders_filled,
            "orders_cancelled": self._orders_cancelled,
            "fill_rate": self._orders_filled / max(self._orders_placed, 1),
            "open_orders_count": len(self.open_orders),
            "positions_count": len([p for p in self.positions.values() if not p.is_flat]),
            "total_volume_traded": self._total_volume_traded,
            "fill_queue_size": self.fills_queue.qsize(),
        }
        
        # Include fill listener metrics if available
        if self._fill_listener:
            metrics["fill_listener"] = self._fill_listener.get_metrics()
        else:
            metrics["fill_listener"] = {"running": False, "connected": False}
        
        # Add execution history metrics
        metrics["execution_history"] = {
            "history_size": len(self.execution_history),
            "total_fills": self.execution_stats.total_fills,
            "maker_fills": self.execution_stats.maker_fills,
            "taker_fills": self.execution_stats.taker_fills,
            "avg_fill_time_ms": self.execution_stats.avg_fill_time_ms,
            "total_volume": self.execution_stats.total_volume
        }
        
        return metrics
    
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
    
    async def sync_orders_with_kalshi(self, is_startup: bool = False) -> Dict[str, Any]:
        """
        Sync local order state with Kalshi.
        
        Kalshi is the source of truth - local state is updated to match Kalshi.
        
        Args:
            is_startup: If True, orders found in Kalshi but not locally are expected
                       (process restart scenario) and won't trigger warnings.
        
        Returns:
            Dictionary with sync statistics
        """
        if not self.trading_client:
            logger.error("Cannot sync orders: trading client not initialized")
            return {"error": "trading_client_not_initialized"}
        
        try:
            # Fetch all orders from Kalshi
            kalshi_orders_response = await self.trading_client.get_orders()
            kalshi_orders = {
                order["order_id"]: order 
                for order in kalshi_orders_response.get("orders", [])
            }
            
            # Track reconciliation stats
            stats = {
                "found_in_kalshi": len(kalshi_orders),
                "found_in_memory": len(self.open_orders),
                "added": 0,
                "updated": 0,
                "removed": 0,
                "partial_fills": 0,
                "discrepancies": 0
            }
            
            # Reconcile each Kalshi order (Kalshi is authoritative)
            for kalshi_order_id, kalshi_order in kalshi_orders.items():
                await self._reconcile_order(kalshi_order_id, kalshi_order, stats, is_startup=is_startup)
            
            # Check for orders in memory but not in Kalshi
            # If Kalshi doesn't have it, it doesn't exist - remove from local state
            for our_order_id, order in list(self.open_orders.items()):
                if order.kalshi_order_id not in kalshi_orders:
                    # Order was cancelled externally or doesn't exist in Kalshi
                    # Trust Kalshi: remove from local tracking
                    await self._handle_external_cancellation(order, stats)
            
            # Log clear warning if discrepancies were found
            # On startup, finding orders in Kalshi but not locally is expected, not a discrepancy
            actual_discrepancies = stats["discrepancies"] + stats["removed"]
            if is_startup:
                # On startup, only warn about actual discrepancies (not orders added from Kalshi)
                if actual_discrepancies > 0:
                    logger.warning(
                        f"⚠️ STARTUP ORDER SYNC DISCREPANCIES DETECTED: "
                        f"{stats['discrepancies']} discrepancies (status/fill mismatches), "
                        f"{stats['removed']} orders removed (not in Kalshi), "
                        f"{stats['partial_fills']} partial fills processed. "
                        f"Local state was out of sync with Kalshi."
                    )
                elif stats["added"] > 0:
                    logger.info(
                        f"Startup sync: Found {stats['added']} order(s) in Kalshi from previous session "
                        f"(expected on restart). Restored to local tracking."
                    )
                elif stats["found_in_kalshi"] == 0:
                    logger.info("Startup sync: No orders found in Kalshi (clean state)")
            else:
                # During periodic sync, warn about any discrepancies including added orders
                if actual_discrepancies > 0 or stats["added"] > 0:
                    logger.warning(
                        f"⚠️ ORDER SYNC DISCREPANCIES DETECTED: "
                        f"{stats['discrepancies']} discrepancies, "
                        f"{stats['added']} orders added (not in local memory), "
                        f"{stats['updated']} orders updated, "
                        f"{stats['removed']} orders removed (not in Kalshi), "
                        f"{stats['partial_fills']} partial fills processed. "
                        f"Local state was out of sync with Kalshi."
                    )
                elif stats["found_in_kalshi"] > 0 or stats["found_in_memory"] > 0:
                    logger.debug(
                        f"Order sync complete: {stats['found_in_kalshi']} orders in Kalshi, "
                        f"{stats['found_in_memory']} orders in local memory - all in sync"
                    )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error syncing orders with Kalshi: {e}")
            return {"error": str(e)}
    
    async def _reconcile_order(
        self, 
        kalshi_order_id: str, 
        kalshi_order: Dict[str, Any], 
        stats: Dict[str, Any],
        is_startup: bool = False
    ) -> None:
        """
        Reconcile a single Kalshi order with local state.
        
        Kalshi state is authoritative - local state is updated to match.
        
        Args:
            kalshi_order_id: Kalshi order ID
            kalshi_order: Order data from Kalshi API
            stats: Statistics dictionary to update
        """
        # Map Kalshi status to internal status
        kalshi_status = kalshi_order.get("status", "").lower()
        if kalshi_status == "resting":
            kalshi_status_mapped = OrderStatus.PENDING
        elif kalshi_status == "executed":
            kalshi_status_mapped = OrderStatus.FILLED
        elif kalshi_status == "canceled":
            kalshi_status_mapped = OrderStatus.CANCELLED
        else:
            logger.warning(f"Unknown Kalshi order status: {kalshi_status}")
            kalshi_status_mapped = OrderStatus.PENDING
        
        # Check if order exists in local memory
        if kalshi_order_id in self._kalshi_to_internal:
            # Order exists locally - reconcile
            our_order_id = self._kalshi_to_internal[kalshi_order_id]
            if our_order_id not in self.open_orders:
                logger.warning(f"Order {kalshi_order_id} in mapping but not in open_orders")
                # Clean up mapping
                del self._kalshi_to_internal[kalshi_order_id]
                stats["discrepancies"] += 1
                return
            
            local_order = self.open_orders[our_order_id]
            
            # Check for discrepancies
            has_discrepancy = False
            
            # Status discrepancy
            if local_order.status != kalshi_status_mapped:
                logger.warning(
                    f"⚠️ ORDER STATUS MISMATCH: Order {kalshi_order_id} ({local_order.ticker}) - "
                    f"local={local_order.status.name}, Kalshi={kalshi_status}. "
                    f"Updating local state to match Kalshi."
                )
                has_discrepancy = True
                stats["discrepancies"] += 1
            
            # Fill count discrepancy
            fill_count = kalshi_order.get("fill_count", 0)
            remaining_count = kalshi_order.get("remaining_count", 0)
            initial_count = kalshi_order.get("initial_count", 0)
            
            # Check if order has fills we haven't processed
            # We compare local order quantity with remaining_count to detect fills
            expected_remaining = local_order.quantity
            if remaining_count < expected_remaining:
                # Order has been filled (partially or fully)
                filled_quantity = expected_remaining - remaining_count
                logger.warning(
                    f"⚠️ ORDER FILL DISCREPANCY: Order {kalshi_order_id} ({local_order.ticker}) - "
                    f"{filled_quantity} fills not processed locally "
                    f"(local remaining: {expected_remaining}, Kalshi remaining: {remaining_count}). "
                    f"Processing missed fill(s) from Kalshi."
                )
                await self._process_partial_fill_from_kalshi(
                    local_order, kalshi_order, filled_quantity, stats
                )
                has_discrepancy = True
                stats["discrepancies"] += 1
            
            # Update order status to match Kalshi
            if local_order.status != kalshi_status_mapped:
                old_status = local_order.status
                local_order.status = kalshi_status_mapped
                
                # If order is now filled or cancelled, handle accordingly
                if kalshi_status_mapped == OrderStatus.FILLED:
                    # Order is fully filled - process any remaining fills and remove from tracking
                    remaining_count = kalshi_order.get("remaining_count", 0)
                    if remaining_count < local_order.quantity:
                        # Process remaining fill
                        filled_quantity = local_order.quantity - remaining_count
                        # Get fill price - always use yes_price (NO price can be derived as 100 - yes_price) [FIXED: was 99-price]
                        # This matches the pattern used elsewhere in the codebase
                        yes_price = kalshi_order.get("yes_price", local_order.limit_price)
                        if local_order.contract_side == ContractSide.YES:
                            fill_price = yes_price
                        else:
                            # Derive NO price from YES price (consistent with limit price calculation)
                            fill_price = 100 - yes_price
                        
                        # Option B cash management
                        if local_order.side == OrderSide.BUY:
                            # Cash already deducted, just reduce promised cash
                            fill_ratio = filled_quantity / local_order.quantity
                            promised_cash_released = local_order.promised_cash * fill_ratio
                            self.promised_cash -= promised_cash_released
                        else:
                            # SELL: add cash received
                            fill_cost = (fill_price / 100.0) * filled_quantity
                            self.cash_balance += fill_cost
                        
                        # Update position
                        self._update_position(local_order, fill_price, filled_quantity)
                        self._orders_filled += 1
                        self._total_volume_traded += (fill_price / 100.0) * filled_quantity
                    
                    # Remove from tracking
                    if local_order.side == OrderSide.BUY:
                        # Release any remaining promised cash
                        self.cash_balance += local_order.promised_cash
                        self.promised_cash -= local_order.promised_cash
                    
                    del self.open_orders[our_order_id]
                    del self._kalshi_to_internal[kalshi_order_id]
                    stats["updated"] += 1
                    
                elif kalshi_status_mapped == OrderStatus.CANCELLED:
                    # Order was cancelled - restore cash if BUY
                    if local_order.side == OrderSide.BUY:
                        self.cash_balance += local_order.promised_cash
                        self.promised_cash -= local_order.promised_cash
                    
                    del self.open_orders[our_order_id]
                    del self._kalshi_to_internal[kalshi_order_id]
                    stats["updated"] += 1
                    logger.info(f"Order {kalshi_order_id} was cancelled externally")
            
            # Update quantity if remaining_count differs
            if remaining_count > 0 and local_order.quantity != remaining_count:
                logger.info(
                    f"Order {kalshi_order_id} quantity mismatch: "
                    f"local={local_order.quantity}, Kalshi remaining={remaining_count}"
                )
                local_order.quantity = remaining_count
                has_discrepancy = True
            
            if has_discrepancy:
                stats["updated"] += 1
        else:
            # Order not in local memory - add it (restart scenario or missed order)
            ticker = kalshi_order.get("ticker", "UNKNOWN")
            if is_startup:
                # On startup, this is expected behavior (process restart)
                logger.info(
                    f"Restoring order from Kalshi: {kalshi_order_id} ({ticker}) - "
                    f"found in Kalshi from previous session, adding to local tracking"
                )
            else:
                # During periodic sync, this indicates a missed order (unexpected)
                logger.warning(
                    f"⚠️ ORDER NOT IN LOCAL MEMORY: Order {kalshi_order_id} ({ticker}) "
                    f"found in Kalshi but not tracked locally. "
                    f"This indicates a missed order. Adding to local tracking."
                )
            await self._add_order_from_kalshi(kalshi_order_id, kalshi_order, stats)
            stats["added"] += 1
    
    async def _process_partial_fill_from_kalshi(
        self,
        local_order: OrderInfo,
        kalshi_order: Dict[str, Any],
        fill_count: int,
        stats: Dict[str, Any]
    ) -> None:
        """
        Process partial fill(s) from Kalshi order data.
        
        Args:
            local_order: Local order tracking
            kalshi_order: Kalshi order data
            fill_count: Number of contracts filled
            stats: Statistics dictionary
        """
        # Get fill price - always use yes_price (NO price can be derived as 100 - yes_price) [FIXED: was 99-price]
        # This matches the pattern used elsewhere in the codebase
        yes_price = kalshi_order.get("yes_price", local_order.limit_price)
        if local_order.contract_side == ContractSide.YES:
            fill_price = yes_price
        else:
            # Derive NO price from YES price (consistent with limit price calculation)
            fill_price = 100 - yes_price
        
        # Process fill through existing logic
        fill_cost = (fill_price / 100.0) * fill_count
        
        # Option B cash management
        if local_order.side == OrderSide.BUY:
            # Cash was already deducted when order placed
            # Reduce promised cash proportionally
            fill_ratio = fill_count / local_order.quantity
            promised_cash_released = local_order.promised_cash * fill_ratio
            self.promised_cash -= promised_cash_released
            local_order.promised_cash -= promised_cash_released
        else:
            # SELL: add cash received
            self.cash_balance += fill_cost
        
        # Update position
        self._update_position(local_order, fill_price, fill_count)
        
        # Update order quantity
        remaining_count = kalshi_order.get("remaining_count", 0)
        local_order.quantity = remaining_count
        
        # Update metrics
        self._orders_filled += 1
        self._total_volume_traded += fill_cost
        
        stats["partial_fills"] += 1
        logger.info(
            f"Processed partial fill: {local_order.order_id} - "
            f"{fill_count} contracts @ {fill_price}¢ (remaining: {remaining_count})"
        )
    
    async def _add_order_from_kalshi(
        self,
        kalshi_order_id: str,
        kalshi_order: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> None:
        """
        Add order from Kalshi to local tracking (restart scenario).
        
        Args:
            kalshi_order_id: Kalshi order ID
            kalshi_order: Order data from Kalshi
            stats: Statistics dictionary
        """
        # Parse Kalshi order data
        ticker = kalshi_order.get("ticker", "")
        kalshi_side = kalshi_order.get("side", "").lower()
        kalshi_action = kalshi_order.get("action", "").lower()
        
        # Map to internal enums
        if kalshi_side == "yes":
            contract_side = ContractSide.YES
        elif kalshi_side == "no":
            contract_side = ContractSide.NO
        else:
            logger.error(f"Unknown contract side: {kalshi_side}")
            return
        
        if kalshi_action == "buy":
            side = OrderSide.BUY
        elif kalshi_action == "sell":
            side = OrderSide.SELL
        else:
            logger.error(f"Unknown action: {kalshi_action}")
            return
        
        # Get order details - always use yes_price (NO price can be derived as 100 - yes_price) [FIXED: was 99-price]
        # This matches the pattern used elsewhere in the codebase
        yes_price = kalshi_order.get("yes_price", 50)
        if contract_side == ContractSide.YES:
            limit_price = yes_price
        else:
            # Derive NO price from YES price (consistent with limit price calculation)
            limit_price = 100 - yes_price
        
        remaining_count = kalshi_order.get("remaining_count", 0)
        initial_count = kalshi_order.get("initial_count", remaining_count)
        fill_count = kalshi_order.get("fill_count", 0)
        
        # Map status
        kalshi_status = kalshi_order.get("status", "").lower()
        if kalshi_status == "resting":
            status = OrderStatus.PENDING
        elif kalshi_status == "executed":
            status = OrderStatus.FILLED
        elif kalshi_status == "canceled":
            status = OrderStatus.CANCELLED
        else:
            status = OrderStatus.PENDING
        
        # Don't add filled or cancelled orders to open_orders
        # They're historical and don't need tracking
        if status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            logger.debug(
                f"Skipping {status.name} order from Kalshi: {kalshi_order_id} "
                f"({ticker}, {side.name} {contract_side.name})"
            )
            # Still process fills if order was filled to update positions
            if status == OrderStatus.FILLED and fill_count > 0:
                # Get fill price - always use yes_price (NO price can be derived as 100 - yes_price) [FIXED: was 99-price]
                # This matches the pattern used elsewhere in the codebase
                yes_price = kalshi_order.get("yes_price", limit_price)
                if contract_side == ContractSide.YES:
                    fill_price = yes_price
                else:
                    # Derive NO price from YES price (consistent with limit price calculation)
                    fill_price = 100 - yes_price
                # Create temporary order info for position update
                temp_order = OrderInfo(
                    order_id="temp",
                    kalshi_order_id=kalshi_order_id,
                    ticker=ticker,
                    side=side,
                    contract_side=contract_side,
                    quantity=fill_count,
                    limit_price=fill_price,
                    status=status,
                    placed_at=time.time(),
                    promised_cash=0.0
                )
                # Update position
                self._update_position(temp_order, fill_price, fill_count)
                # Update cash if SELL
                if side == OrderSide.SELL:
                    fill_cost = (fill_price / 100.0) * fill_count
                    self.cash_balance += fill_cost
                self._orders_filled += 1
                self._total_volume_traded += (fill_price / 100.0) * fill_count
            return
        
        # Generate internal order ID
        our_order_id = self._generate_order_id()
        
        # Calculate promised cash (if BUY and still pending)
        promised_cash = 0.0
        if side == OrderSide.BUY and status == OrderStatus.PENDING:
            promised_cash = (limit_price / 100.0) * remaining_count
            # Reserve cash
            self.cash_balance -= promised_cash
            self.promised_cash += promised_cash
        
        # Create order info
        order_info = OrderInfo(
            order_id=our_order_id,
            kalshi_order_id=kalshi_order_id,
            ticker=ticker,
            side=side,
            contract_side=contract_side,
            quantity=remaining_count,
            limit_price=limit_price,
            status=status,
            placed_at=time.time(),  # Use current time (we don't have original timestamp)
            promised_cash=promised_cash
        )
        
        # If order has partial fills, process them
        if fill_count > 0 and remaining_count > 0:
            fill_price = kalshi_order.get("yes_price", limit_price)
            await self._process_partial_fill_from_kalshi(
                order_info, kalshi_order, fill_count, stats
            )
        
        # Add to tracking
        self.open_orders[our_order_id] = order_info
        self._kalshi_to_internal[kalshi_order_id] = our_order_id
        
        logger.info(
            f"Added order from Kalshi: {our_order_id} -> {kalshi_order_id} "
            f"({ticker}, {side.name} {contract_side.name}, {remaining_count} remaining)"
        )
    
    async def _handle_external_cancellation(
        self,
        order: OrderInfo,
        stats: Dict[str, Any]
    ) -> None:
        """
        Handle order that exists locally but not in Kalshi.
        
        Kalshi is authoritative - if it doesn't exist in Kalshi, remove from local state.
        
        Args:
            order: Local order tracking
            stats: Statistics dictionary
        """
        logger.warning(
            f"⚠️ ORDER NOT IN KALSHI: Order {order.order_id} ({order.kalshi_order_id}, {order.ticker}) "
            f"exists locally but not in Kalshi. "
            f"This indicates external cancellation or state mismatch. Removing from local tracking."
        )
        
        # Restore promised cash if BUY order
        if order.side == OrderSide.BUY:
            self.cash_balance += order.promised_cash
            self.promised_cash -= order.promised_cash
        
        # Remove from tracking
        del self.open_orders[order.order_id]
        if order.kalshi_order_id in self._kalshi_to_internal:
            del self._kalshi_to_internal[order.kalshi_order_id]
        
        stats["removed"] += 1
        self._orders_cancelled += 1
    
    async def _sync_positions_with_kalshi(self, is_startup: bool = False) -> None:
        """
        Sync positions with Kalshi.
        
        Kalshi positions are authoritative - update local positions to match.
        
        Args:
            is_startup: If True, positions found in Kalshi but not locally are expected
                       (process restart scenario) and won't trigger warnings.
        """
        if not self.trading_client:
            logger.error("Cannot sync positions: trading client not initialized")
            return
        
        try:
            # Get positions from Kalshi
            kalshi_positions_response = await self.trading_client.get_positions()
            kalshi_positions_list = kalshi_positions_response.get("positions", kalshi_positions_response.get("market_positions", []))
            
            # Log the raw response structure for debugging
            if kalshi_positions_list:
                logger.debug(f"Kalshi positions response sample: {kalshi_positions_list[0] if kalshi_positions_list else 'empty'}")
            
            # Build dict of Kalshi positions by ticker
            kalshi_positions_dict = {}
            for kalshi_pos in kalshi_positions_list:
                ticker = kalshi_pos.get("ticker", "")
                if ticker:
                    kalshi_positions_dict[ticker] = kalshi_pos
            
            position_discrepancies = []
            
            # Update local positions to match Kalshi
            for ticker, kalshi_pos in kalshi_positions_dict.items():
                contracts = kalshi_pos.get("position", 0)
                
                if ticker not in self.positions:
                    # New position - store the entire Kalshi response
                    self.positions[ticker] = Position(
                        ticker=ticker,
                        contracts=contracts,
                        cost_basis=0.0,  # Will be calculated if needed
                        realized_pnl=0.0,  # Will be calculated if needed
                        kalshi_data=kalshi_pos  # Store complete Kalshi response
                    )
                    if is_startup:
                        logger.info(
                            f"Restoring position from Kalshi: {ticker} = {contracts} contracts "
                            f"(expected on restart)"
                        )
                    else:
                        logger.warning(
                            f"⚠️ POSITION NOT IN LOCAL MEMORY: {ticker} = {contracts} contracts "
                            f"found in Kalshi but not tracked locally. Adding to local tracking."
                        )
                        position_discrepancies.append(f"{ticker}: added ({contracts} contracts)")
                else:
                    # Update existing position with Kalshi data
                    local_pos = self.positions[ticker]
                    if local_pos.contracts != contracts:
                        logger.warning(
                            f"⚠️ POSITION MISMATCH: {ticker} - "
                            f"local={local_pos.contracts}, Kalshi={contracts}. "
                            f"Updating local position to match Kalshi."
                        )
                        position_discrepancies.append(
                            f"{ticker}: {local_pos.contracts} -> {contracts}"
                        )
                    
                    # Always update with complete Kalshi response (authoritative source)
                    local_pos.contracts = contracts
                    local_pos.kalshi_data = kalshi_pos
            
            # Remove positions that don't exist in Kalshi
            for ticker in list(self.positions.keys()):
                if ticker not in kalshi_positions_dict:
                    logger.warning(
                        f"⚠️ POSITION NOT IN KALSHI: {ticker} exists locally but not in Kalshi. "
                        f"Removing from local tracking."
                    )
                    del self.positions[ticker]
                    position_discrepancies.append(f"{ticker}: removed")
            
            if position_discrepancies:
                if is_startup:
                    logger.info(
                        f"Startup position sync: Restored {len(position_discrepancies)} position(s) "
                        f"from Kalshi - {', '.join(position_discrepancies)}"
                    )
                else:
                    logger.warning(
                        f"⚠️ POSITION SYNC DISCREPANCIES DETECTED: "
                        f"{len(position_discrepancies)} position(s) out of sync - {', '.join(position_discrepancies)}"
                    )
            else:
                if is_startup and len(self.positions) == 0:
                    logger.info("Startup position sync: No positions found in Kalshi (clean state)")
                else:
                    logger.debug(f"Position sync complete: {len(self.positions)} positions - all in sync")
            
            # Also sync cash balance from Kalshi
            try:
                account_info = await self.trading_client.get_account_info()
                if "balance" in account_info:
                    old_balance = self.cash_balance
                    self.cash_balance = account_info["balance"] / 100.0  # Convert cents to dollars
                    if abs(old_balance - self.cash_balance) > 0.01:
                        if is_startup:
                            logger.info(f"Cash balance synced: ${old_balance:.2f} → ${self.cash_balance:.2f} (from Kalshi)")
                        else:
                            logger.warning(f"Cash balance synced: ${old_balance:.2f} → ${self.cash_balance:.2f}")
            except Exception as e:
                logger.warning(f"Could not sync cash balance: {e}")
            
        except Exception as e:
            logger.error(f"Error syncing positions with Kalshi: {e}")
    
    async def _start_periodic_sync(self) -> None:
        """
        Background task for periodic order synchronization.
        
        Runs continuously, syncing orders with Kalshi at configured intervals.
        """
        from ..config import config
        
        logger.info("Periodic sync task started")
        
        try:
            while True:
                await asyncio.sleep(config.RL_ORDER_SYNC_INTERVAL_SECONDS)
                
                try:
                    logger.debug("Starting periodic order sync...")
                    stats = await self.sync_orders_with_kalshi()
                    
                    # Note: sync_orders_with_kalshi() already logs warnings for discrepancies
                    # Just log summary here if no discrepancies
                    if stats.get("discrepancies", 0) == 0 and stats.get("added", 0) == 0 and stats.get("removed", 0) == 0:
                        logger.debug(
                            f"Periodic sync complete: {stats['found_in_kalshi']} orders in sync"
                        )
                    
                    # Also sync positions periodically
                    await self._sync_positions_with_kalshi()
                    
                    # Broadcast specific updates after periodic sync
                    await self._broadcast_positions_update("periodic_sync")
                    await self._broadcast_portfolio_update("periodic_sync")
                    
                    # Notify about state changes after periodic sync
                    await self._notify_state_change()
                    
                except Exception as e:
                    logger.error(f"Error in periodic sync: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Periodic sync task cancelled")
            raise
        except Exception as e:
            logger.error(f"Periodic sync task error: {e}")
    
    def add_state_change_callback(self, callback) -> None:
        """
        Add a callback function to be called when trader state changes.
        
        Args:
            callback: Async function that takes current state as parameter
        """
        self._state_change_callbacks.append(callback)
        
    def set_websocket_manager(self, websocket_manager) -> None:
        """
        Set the WebSocket manager for specific event broadcasts.
        
        Args:
            websocket_manager: WebSocketManager instance for broadcasting specific events
        """
        self._websocket_manager = websocket_manager
        logger.info("WebSocket manager configured for specific event broadcasts")
        
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get current trading state for UI display.
        
        Returns:
            Dictionary containing portfolio state, positions, orders, and metrics
        """
        # Calculate portfolio value using actual market exposure from Kalshi
        portfolio_value = self.cash_balance + self.promised_cash
        for position in self.positions.values():
            # Use market_exposure_dollars if available, otherwise convert from cents
            kalshi_data = position.kalshi_data
            if "market_exposure_dollars" in kalshi_data:
                portfolio_value += float(kalshi_data["market_exposure_dollars"])
            elif "market_exposure" in kalshi_data:
                portfolio_value += float(kalshi_data["market_exposure"]) / 100.0
        
        # Calculate fill rate
        total_orders = self._orders_placed
        fill_rate = self._orders_filled / total_orders if total_orders > 0 else 0.0
        
        return {
            "portfolio_value": portfolio_value,
            "cash_balance": self.cash_balance,
            "promised_cash": self.promised_cash,
            "open_orders": [
                {
                    "order_id": order_info.order_id,
                    "ticker": order_info.ticker,
                    "side": "BUY" if order_info.side == OrderSide.BUY else "SELL",
                    "quantity": order_info.quantity,
                    "price": order_info.limit_price,
                    "status": order_info.status.name
                }
                for order_info in self.open_orders.values()
            ],
            "positions": {
                ticker: {
                    "ticker": ticker,
                    "side": "YES" if position.contracts > 0 else "NO",
                    "contracts": abs(position.contracts),
                    **position.kalshi_data  # Pass through complete Kalshi API response
                }
                for ticker, position in self.positions.items()
                if not position.is_flat
            },
            "metrics": {
                "orders_placed": self._orders_placed,
                "orders_filled": self._orders_filled,
                "orders_cancelled": self._orders_cancelled,
                "fill_rate": fill_rate,
                "volume_traded": self._total_volume_traded
            },
            "markets": {
                "tracked_tickers": self._market_tickers,
                "market_count": self._market_count,
                "per_market_activity": self._calculate_per_market_activity()
            },
            "actor": {
                "strategy": config.RL_ACTOR_STRATEGY,
                "enabled": config.RL_ACTOR_ENABLED,
                "throttle_ms": config.RL_ACTOR_THROTTLE_MS
            }
        }
        
    async def _notify_state_change(self) -> None:
        """Notify all registered callbacks of state changes."""
        if not self._state_change_callbacks:
            return
            
        current_state = await self.get_current_state()
        
        for callback in self._state_change_callbacks:
            try:
                await callback(current_state)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}")

    def _calculate_per_market_activity(self) -> Dict[str, Dict[str, Any]]:
        """Calculate per-market trading activity for state reporting."""
        activity = {}
        
        for ticker in self._market_tickers:
            # Count orders for this market
            orders_count = sum(1 for order in self.open_orders.values() 
                              if order.ticker == ticker)
            
            # Calculate volume traded for this market from execution history
            volume_traded = 0.0
            trades_count = 0
            
            for trade in self.execution_history:
                if trade.ticker == ticker:
                    volume_traded += (trade.fill_price / 100.0) * trade.quantity
                    trades_count += 1
            
            # Include open orders promised value
            for order in self.open_orders.values():
                if order.ticker == ticker:
                    volume_traded += (order.limit_price / 100.0) * order.quantity
            
            activity[ticker] = {
                "orders": orders_count,
                "volume": volume_traded,
                "trades": trades_count
            }
        
        return activity

    def register_action_selector(self, selector) -> None:
        """
        Register an action selector with mutual exclusivity protection.
        
        Only one selector can be registered at a time. If a selector is already
        registered, this will raise an error to prevent double-firing.
        
        Args:
            selector: Action selector to register (RLActionSelector or HardcodedSelector)
            
        Raises:
            RuntimeError: If a selector is already registered
        """
        if self.action_selector is not None:
            existing_selector_name = type(self.action_selector).__name__
            new_selector_name = type(selector).__name__
            logger.error(
                f"Attempted to register {new_selector_name} when {existing_selector_name} "
                f"is already registered. Only one action selector allowed per session."
            )
            raise RuntimeError(
                f"Action selector already registered: {existing_selector_name}. "
                f"Cannot register {new_selector_name}. Only one selector allowed per session."
            )
        
        self.action_selector = selector
        selector_name = type(selector).__name__
        logger.info(f"✅ Action selector registered: {selector_name}")
        logger.info(f"Strategy configuration confirmed: Only {selector_name} will fire")
    
    def _generate_order_id(self) -> str:
        """Generate unique internal order ID."""
        self._order_counter += 1
        return f"order_{self._order_counter}_{int(time.time() * 1000)}"