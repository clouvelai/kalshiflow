"""
TRADER V3 Trading Client Integration.

Integration layer between V3 and the Kalshi trading client (paper/production).
Provides clean abstraction for order management and position tracking.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from decimal import Decimal

from .demo_client import KalshiDemoTradingClient
from ..core.event_bus import EventBus, EventType
from ..sync.kalshi_data_sync import KalshiDataSync
from ..state.trader_state import TraderState, StateChange

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.trading_client_integration")


@dataclass
class TradingClientMetrics:
    """Metrics for trading client operations."""
    orders_placed: int = 0
    orders_cancelled: int = 0
    orders_filled: int = 0
    orders_rejected: int = 0
    positions_synced: int = 0
    api_calls: int = 0
    api_errors: int = 0
    last_order_time: Optional[float] = None
    last_fill_time: Optional[float] = None
    last_sync_time: Optional[float] = None
    connected_at: Optional[float] = None
    
    # Position tracking
    active_positions: Dict[str, Any] = field(default_factory=dict)
    open_orders: Dict[str, Any] = field(default_factory=dict)
    
    # Account state
    balance: Decimal = Decimal("0.00")
    portfolio_value: Decimal = Decimal("0.00")


# CalibrationData removed - replaced by TraderState and KalshiDataSync


class V3TradingClientIntegration:
    """
    Integration layer for trading client in TRADER V3.
    
    Features:
    - Wrapper around existing KalshiDemoTradingClient
    - Event bus integration for order/position updates
    - Calibration phase for position syncing
    - Metrics tracking for monitoring
    - Health checks and error recovery
    - Clean start/stop lifecycle
    """
    
    def __init__(
        self,
        trading_client: KalshiDemoTradingClient,
        event_bus: EventBus,
        max_orders: int = 10,
        max_position_size: int = 100
    ):
        """
        Initialize trading client integration.
        
        Args:
            trading_client: Existing trading client instance (paper/production)
            event_bus: Event bus for broadcasting updates
            max_orders: Maximum concurrent orders allowed
            max_position_size: Maximum position size per market
        """
        self._client = trading_client
        self._event_bus = event_bus
        self._max_orders = max_orders
        self._max_position_size = max_position_size
        
        # Initialize sync service
        self._kalshi_data_sync = KalshiDataSync(trading_client)
        self._trader_state: Optional[TraderState] = None
        
        self._metrics = TradingClientMetrics()
        self._running = False
        self._started_at: Optional[float] = None
        self._connected = False
        self._sync_complete = False  # Renamed from calibration_complete
        self._last_health_check: Optional[float] = None
        
        # Order group tracking
        self._order_group_id: Optional[str] = None
        self._order_groups_supported = False  # Will be set to True if order groups work
        
        # Health monitoring thresholds
        self._health_check_interval = 30.0  # Check health every 30 seconds
        self._max_api_errors = 5  # Maximum consecutive API errors before unhealthy
        self._consecutive_api_errors = 0
        
        logger.info(
            f"V3 Trading Client Integration initialized "
            f"(mode={trading_client.mode}, max_orders={max_orders}, max_position={max_position_size})"
        )
    
    @property
    def api_url(self) -> str:
        """Get the API URL being used by the trading client."""
        return self._client.rest_base_url
    
    async def start(self) -> None:
        """Start trading client integration."""
        if self._running:
            logger.warning("Trading client integration is already running")
            return
        
        logger.info(f"Starting V3 trading client integration (mode={self._client.mode})")
        self._running = True
        self._started_at = time.time()
        
        # Note: Connection will be established in wait_for_connection()
        # This follows the same pattern as OrderbookIntegration
        
        logger.info("✅ Trading client integration started (awaiting connection)")
    
    async def stop(self) -> None:
        """Stop trading client integration."""
        if not self._running:
            return
        
        logger.info("Stopping V3 trading client integration...")
        self._running = False
        
        # Reset order group if active  
        if self._order_group_id:
            try:
                await self.reset_order_group()
            except Exception as e:
                logger.warning(f"Could not reset order group on stop: {e}")
        
        # Disconnect from trading API
        if self._connected:
            try:
                await self._client.disconnect()
                self._connected = False
                logger.info("✅ Trading client disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting trading client: {e}")
        
        # Clear metrics
        self._metrics.active_positions.clear()
        self._metrics.open_orders.clear()
        
        logger.info(f"✅ V3 Trading Client Integration stopped - Final Metrics: {self.get_metrics()}")
    
    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for trading client to establish connection.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected, False if timeout
        """
        logger.info(f"Connecting to trading API (mode={self._client.mode}, timeout={timeout}s)...")
        
        try:
            # Connect to the trading API
            await self._client.connect()
            
            # Test connection with exchange_status (lightweight call)
            account_info = await self._client.get_account_info()
            
            self._connected = True
            self._metrics.connected_at = time.time()
            
            # Update balance from account info
            if "balance" in account_info:
                self._metrics.balance = Decimal(str(account_info["balance"])) / 100  # Convert cents to dollars
            if "portfolio_value" in account_info:
                self._metrics.portfolio_value = Decimal(str(account_info["portfolio_value"])) / 100
            
            logger.info(
                f"✅ Trading client connected (mode={self._client.mode}, "
                f"balance=${self._metrics.balance}, portfolio=${self._metrics.portfolio_value})"
            )
            
            # Create order group for portfolio limits
            try:
                order_group_id = await self.create_order_group(contracts_limit=10000)
                if order_group_id:
                    self._order_groups_supported = True
                    logger.info(f"✅ Order group ready: {order_group_id[:8]}...")
                else:
                    logger.warning("⚠️ Running without order group (no portfolio limits)")
            except Exception as e:
                logger.warning(f"⚠️ Could not create order group: {e}")
                # Continue without order group - it's not critical
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trading client connection failed: {e}")
            self._metrics.api_errors += 1
            return False
    
    async def sync_with_kalshi(self) -> Tuple[TraderState, Optional[StateChange]]:
        """
        Sync trader state with Kalshi.
        Replaces the old calibrate() method.
        
        Returns:
            Tuple of (new_state, changes_from_previous)
            changes will be None on first sync
        """
        if not self._connected:
            raise RuntimeError("Cannot sync - trading client not connected")
        
        # Delegate to sync service
        state, changes = await self._kalshi_data_sync.sync_with_kalshi()
        self._trader_state = state
        self._sync_complete = True
        
        # Update metrics for backward compatibility
        # Note: These are in CENTS now, not dollars
        self._metrics.balance = Decimal(state.balance) / 100  # Keep as dollars for old metrics
        self._metrics.portfolio_value = Decimal(state.portfolio_value) / 100
        self._metrics.active_positions = state.positions
        self._metrics.open_orders = state.orders
        self._metrics.positions_synced = state.position_count
        self._metrics.last_sync_time = state.sync_timestamp
        
        return state, changes
    
    @property
    def trader_state(self) -> Optional[TraderState]:
        """Get current trader state."""
        return self._trader_state
    
    async def place_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price: Optional[int] = None,
        order_type: str = "limit",
        order_group_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Place an order through the trading client.
        
        Args:
            ticker: Market ticker
            action: "buy" or "sell"
            side: "yes" or "no"
            count: Number of contracts
            price: Limit price in cents (1-99)
            order_type: "limit" or "market"
            order_group_id: Optional order group ID for portfolio limits
            
        Returns:
            Order response from API
        """
        if not self._connected:
            raise RuntimeError("Cannot place order - trading client not connected")
        
        # Validate against limits
        if len(self._metrics.open_orders) >= self._max_orders:
            raise ValueError(f"Maximum orders ({self._max_orders}) exceeded")
        
        # Check position size limits
        current_position = self._metrics.active_positions.get(ticker, {})
        current_size = current_position.get("position", 0)
        if abs(current_size + count) > self._max_position_size:
            raise ValueError(f"Position size would exceed limit ({self._max_position_size})")
        
        try:
            # Use order group if provided or default to instance group
            group_id = order_group_id or self._order_group_id

            # Place order through client
            response = await self._client.create_order(
                ticker=ticker,
                action=action,
                side=side,
                count=count,
                price=price,
                type=order_type,
                order_group_id=group_id
            )

            # Update metrics
            self._metrics.orders_placed += 1
            self._metrics.last_order_time = time.time()
            self._metrics.api_calls += 1

            # Update open orders tracking
            if "order" in response:
                order = response["order"]
                order_id = order.get("order_id")
                if order_id:
                    self._metrics.open_orders[order_id] = order

            # Note: Order events would require extending EventBus with a generic emit method
            # For now, we track via logging and metrics

            if group_id:
                logger.info(f"✅ Order placed: {action} {count} {side} {ticker} @ {price}¢ (group: {group_id[:8]}...)")
            else:
                logger.info(f"✅ Order placed: {action} {count} {side} {ticker} @ {price}¢ (no portfolio limits)")
            return response

        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            self._metrics.api_errors += 1
            self._metrics.orders_rejected += 1
            self._consecutive_api_errors += 1
            raise
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response
        """
        if not self._connected:
            raise RuntimeError("Cannot cancel order - trading client not connected")
        
        try:
            response = await self._client.cancel_order(order_id)
            
            # Update metrics
            self._metrics.orders_cancelled += 1
            self._metrics.api_calls += 1
            
            # Remove from open orders tracking
            if order_id in self._metrics.open_orders:
                del self._metrics.open_orders[order_id]
            
            # Note: Order events would require extending EventBus with a generic emit method
            
            logger.info(f"✅ Order cancelled: {order_id}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Order cancellation failed: {e}")
            self._metrics.api_errors += 1
            self._consecutive_api_errors += 1
            raise
    
    async def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancel all open orders.

        Returns:
            Dictionary with cancellation results
        """
        if not self._connected:
            raise RuntimeError("Cannot cancel orders - trading client not connected")

        if not self._metrics.open_orders:
            return {"cancelled": [], "errors": [], "total": 0}

        order_ids = list(self._metrics.open_orders.keys())
        logger.info(f"Cancelling {len(order_ids)} open orders...")

        try:
            response = await self._client.batch_cancel_orders(order_ids)

            # Update metrics
            cancelled_count = len(response.get("cancelled", []))
            self._metrics.orders_cancelled += cancelled_count
            self._metrics.api_calls += 1

            # Remove cancelled orders from tracking
            for order_id in response.get("cancelled", []):
                if order_id in self._metrics.open_orders:
                    del self._metrics.open_orders[order_id]

            # Note: Batch events would require extending EventBus with a generic emit method

            logger.info(f"✅ Batch cancel complete: {cancelled_count} cancelled")
            return response

        except Exception as e:
            logger.error(f"❌ Batch cancel failed: {e}")
            self._metrics.api_errors += 1
            self._consecutive_api_errors += 1
            raise

    async def cancel_orphaned_orders(self) -> Dict[str, Any]:
        """
        Cancel only orphaned orders (orders without an order_group_id).

        This method is designed for startup cleanup to remove legacy orders
        that were placed before order group management was implemented,
        while preserving orders that belong to the current session's order group.

        Returns:
            Dictionary with:
                - cancelled: List of cancelled order IDs
                - skipped: List of preserved order IDs (have order_group_id)
                - errors: List of errors during cancellation
                - total_orphaned: Count of orphaned orders found
                - total_preserved: Count of orders with order_group_id
        """
        if not self._connected:
            raise RuntimeError("Cannot cancel orders - trading client not connected")

        result = {
            "cancelled": [],
            "skipped": [],
            "errors": [],
            "total_orphaned": 0,
            "total_preserved": 0
        }

        if not self._metrics.open_orders:
            logger.info("No open orders to clean up")
            return result

        # Categorize orders by orphan status
        orphaned_order_ids = []
        preserved_order_ids = []

        for order_id, order in self._metrics.open_orders.items():
            # Check if order has an order_group_id
            order_group_id = order.get("order_group_id")

            if not order_group_id:
                # Orphaned order - no order_group_id
                orphaned_order_ids.append(order_id)
            else:
                # Order belongs to an order group - preserve it
                preserved_order_ids.append(order_id)

        result["total_orphaned"] = len(orphaned_order_ids)
        result["total_preserved"] = len(preserved_order_ids)

        if not orphaned_order_ids:
            logger.info(f"No orphaned orders to clean up ({len(preserved_order_ids)} orders preserved with order groups)")
            result["skipped"] = preserved_order_ids
            return result

        logger.info(
            f"Cleaning up {len(orphaned_order_ids)} orphaned orders "
            f"(preserving {len(preserved_order_ids)} with order groups)..."
        )

        # Cancel orphaned orders in batch
        try:
            response = await self._client.batch_cancel_orders(orphaned_order_ids)

            # Update metrics
            cancelled_count = len(response.get("cancelled", []))
            self._metrics.orders_cancelled += cancelled_count
            self._metrics.api_calls += 1

            # Remove cancelled orders from tracking
            for order_id in response.get("cancelled", []):
                if order_id in self._metrics.open_orders:
                    del self._metrics.open_orders[order_id]

            result["cancelled"] = response.get("cancelled", [])
            result["errors"] = response.get("errors", [])
            result["skipped"] = preserved_order_ids

            logger.info(
                f"✅ Orphaned order cleanup complete: "
                f"{len(result['cancelled'])} cancelled, "
                f"{len(result['skipped'])} preserved, "
                f"{len(result['errors'])} errors"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Orphaned order cleanup failed: {e}")
            self._metrics.api_errors += 1
            self._consecutive_api_errors += 1
            raise
    
    async def sync_positions(self) -> Dict[str, Any]:
        """
        Sync current positions with exchange.
        
        Returns:
            Current positions
        """
        if not self._connected:
            raise RuntimeError("Cannot sync positions - trading client not connected")
        
        try:
            response = await self._client.get_positions()
            
            # Update metrics
            self._metrics.active_positions = dict(self._client.positions)
            self._metrics.positions_synced = len(self._client.positions)
            self._metrics.last_sync_time = time.time()
            self._metrics.api_calls += 1
            self._consecutive_api_errors = 0  # Reset on success
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Position sync failed: {e}")
            self._metrics.api_errors += 1
            self._consecutive_api_errors += 1
            raise
    
    async def check_health(self) -> None:
        """Perform health check on trading client."""
        if not self._connected:
            return
        
        try:
            # Light-weight health check - just get account info
            account_info = await self._client.get_account_info()
            
            # Update balance
            if "balance" in account_info:
                self._metrics.balance = Decimal(str(account_info["balance"])) / 100
            if "portfolio_value" in account_info:
                self._metrics.portfolio_value = Decimal(str(account_info["portfolio_value"])) / 100
            
            self._last_health_check = time.time()
            self._consecutive_api_errors = 0  # Reset on successful health check
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            self._metrics.api_errors += 1
            self._consecutive_api_errors += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get trading client integration metrics."""
        uptime = time.time() - self._started_at if self._started_at else 0
        
        return {
            "running": self._running,
            "connected": self._connected,
            "mode": self._client.mode if self._client else "unknown",
            "calibrated": self._sync_complete,
            "orders_placed": self._metrics.orders_placed,
            "orders_cancelled": self._metrics.orders_cancelled,
            "orders_filled": self._metrics.orders_filled,
            "orders_rejected": self._metrics.orders_rejected,
            "positions_count": len(self._metrics.active_positions),
            "open_orders_count": len(self._metrics.open_orders),
            "balance": str(self._metrics.balance),
            "portfolio_value": str(self._metrics.portfolio_value),
            "api_calls": self._metrics.api_calls,
            "api_errors": self._metrics.api_errors,
            "last_order_time": self._metrics.last_order_time,
            "last_sync_time": self._metrics.last_sync_time,
            "uptime_seconds": uptime
        }
    
    def is_healthy(self) -> bool:
        """Check if trading client integration is healthy."""
        if not self._running:
            return False
        
        if not self._connected:
            # During initial startup, be lenient
            if self._started_at and (time.time() - self._started_at) < 30.0:
                return True  # Still starting up
            return False
        
        # Check for excessive API errors
        if self._consecutive_api_errors >= self._max_api_errors:
            logger.warning(f"Trading client unhealthy: {self._consecutive_api_errors} consecutive API errors")
            return False
        
        # Check if health check is stale
        if self._last_health_check:
            time_since_health_check = time.time() - self._last_health_check
            if time_since_health_check > self._health_check_interval * 3:  # 3x interval = unhealthy
                logger.warning(f"Trading client unhealthy: health check stale ({time_since_health_check:.1f}s)")
                return False
        
        return True
    
    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        metrics = self.get_metrics()
        now = time.time()
        
        time_since_health_check = None
        if self._last_health_check:
            time_since_health_check = now - self._last_health_check
        
        return {
            "healthy": self.is_healthy(),
            "running": self._running,
            "connected": self._connected,
            "calibrated": self._sync_complete,
            "mode": self._client.mode if self._client else "unknown",
            "positions_count": len(self._metrics.active_positions),
            "open_orders_count": len(self._metrics.open_orders),
            "balance": str(self._metrics.balance),
            "api_errors": self._metrics.api_errors,
            "consecutive_api_errors": self._consecutive_api_errors,
            "time_since_health_check": time_since_health_check,
            "uptime_seconds": metrics["uptime_seconds"],
            "connection_time": time.strftime("%H:%M:%S", time.localtime(self._metrics.connected_at)) if self._metrics.connected_at else None
        }
    
    def get_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific market.
        
        Args:
            ticker: Market ticker
            
        Returns:
            Position data or None if no position
        """
        return self._metrics.active_positions.get(ticker)
    
    def get_open_orders(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get open orders, optionally filtered by ticker.
        
        Args:
            ticker: Optional market ticker to filter
            
        Returns:
            Dictionary of open orders
        """
        if ticker is None:
            return dict(self._metrics.open_orders)
        
        # Filter by ticker
        filtered = {}
        for order_id, order in self._metrics.open_orders.items():
            if order.get("ticker") == ticker:
                filtered[order_id] = order
        
        return filtered
    
    # ========================
    # Order Group Management
    # ========================
    
    async def create_order_group(self, contracts_limit: int = 10000) -> str:
        """
        Create an order group session for portfolio limits.
        
        Args:
            contracts_limit: Maximum number of contracts (default 10000)
            
        Returns:
            Order group ID
            
        Raises:
            Exception if creation fails
        """
        if self._order_group_id:
            logger.warning(f"Order group already exists: {self._order_group_id[:8]}...")
            return self._order_group_id
        
        try:
            # Create order group via client
            response = await self._client.create_order_group(
                contracts_limit=contracts_limit
            )
            
            self._order_group_id = response["order_group_id"]
            
            # Pass to sync service
            self._kalshi_data_sync.set_order_group_id(self._order_group_id)
            
            logger.info(f"✅ Created order group: {self._order_group_id[:8]}... "
                       f"(contracts_limit: {contracts_limit})")
            
            # Broadcast system activity event
            if self._event_bus:
                await self._event_bus.emit_system_activity(
                    activity_type="operation",
                    message=f"Order group created: {self._order_group_id[:8]}... (limit: {contracts_limit} contracts)",
                    metadata={
                        "order_group_id": self._order_group_id,
                        "contracts_limit": contracts_limit,
                        "component": "TradingClient"
                    }
                )
            
            return self._order_group_id
            
        except Exception as e:
            logger.error(f"Failed to create order group: {e}")
            raise
    
    def set_order_group_id(self, order_group_id: str) -> None:
        """
        Set the order group ID for this integration.
        
        Args:
            order_group_id: The order group UUID
        """
        self._order_group_id = order_group_id
        self._kalshi_data_sync.set_order_group_id(order_group_id)
        logger.info(f"Order group ID set: {order_group_id[:8]}...")
    
    def get_order_group_id(self) -> Optional[str]:
        """Get current order group ID."""
        return self._order_group_id
    
    async def create_or_get_order_group(self, contracts_limit: int = 10000) -> Optional[str]:
        """
        Create a new order group or return existing one.
        
        Args:
            contracts_limit: Maximum number of contracts (default 10000)
            
        Returns:
            Order group ID or None if creation failed
        """
        if self._order_group_id:
            logger.debug(f"Using existing order group: {self._order_group_id[:8]}...")
            return self._order_group_id
        
        try:
            response = await self._client.create_order_group(contracts_limit)
            self.set_order_group_id(response["order_group_id"])
            self._order_groups_supported = True
            
            logger.info(f"✅ Created order group: {self._order_group_id[:8]}... "
                       f"(contracts_limit: {contracts_limit})")
            
            return self._order_group_id
            
        except Exception as e:
            logger.warning(f"Could not create order group: {e}")
            return None
    
    async def reset_order_group(self) -> bool:
        """
        Reset the current order group session.
        
        Returns:
            True if closed successfully, False otherwise
        """
        if not self._order_group_id:
            logger.debug("No order group to close")
            return True
        
        try:
            # Reset via client (using new API endpoint)
            await self._client.reset_order_group(self._order_group_id)
            
            logger.info(f"✅ Reset order group: {self._order_group_id[:8]}...")
            
            # Clear tracking
            self._order_group_id = None
            self._kalshi_data_sync.set_order_group_id(None)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset order group: {e}")
            return False
    
    # Backward compatibility alias
    async def close_order_group(self) -> bool:
        """Alias for reset_order_group for backward compatibility."""
        return await self.reset_order_group()
    
    async def list_order_groups(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all order groups for the account.

        Args:
            status: Optional filter by status (active, closed)

        Returns:
            List of order group dicts

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected:
            raise RuntimeError("Cannot list order groups - trading client not connected")

        try:
            response = await self._client.list_order_groups(status=status)
            return response.get("order_groups", [])
        except Exception as e:
            logger.error(f"Failed to list order groups: {e}")
            raise

    async def reset_order_group_by_id(self, order_group_id: str) -> bool:
        """
        Reset a specific order group by ID.

        Args:
            order_group_id: UUID of the order group to reset

        Returns:
            True if reset successfully, False otherwise
        """
        if not self._connected:
            raise RuntimeError("Cannot reset order group - trading client not connected")

        try:
            await self._client.reset_order_group(order_group_id)
            logger.info(f"Reset order group: {order_group_id[:8]}...")

            # If this was our current order group, clear the tracking
            if self._order_group_id == order_group_id:
                self._order_group_id = None
                self._kalshi_data_sync.set_order_group_id(None)

            return True

        except Exception as e:
            logger.error(f"Failed to reset order group {order_group_id[:8]}...: {e}")
            return False

    async def delete_order_group_by_id(self, order_group_id: str) -> bool:
        """Delete a specific order group by ID."""
        if not self._connected:
            raise RuntimeError("Cannot delete order group - trading client not connected")
        try:
            await self._client.delete_order_group(order_group_id)
            logger.info(f"Deleted order group: {order_group_id[:8]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to delete order group {order_group_id[:8]}...: {e}")
            return False

    @property
    def order_group_id(self) -> Optional[str]:
        """Get current order group ID."""
        return self._order_group_id

    @property
    def has_order_group(self) -> bool:
        """Check if order group is active."""
        return self._order_group_id is not None

    @property
    def order_groups_supported(self) -> bool:
        """Check if order groups are supported by API."""
        return self._order_groups_supported

    # ========================
    # Market Data Methods
    # ========================

    async def get_markets(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch market data for specific tickers.

        GET /trade-api/v2/markets?tickers=TICKER1,TICKER2,...

        Args:
            tickers: List of market tickers to fetch

        Returns:
            List of market data dicts with bid/ask prices, close_time, etc.

        Raises:
            RuntimeError: If not connected
        """
        if not self._connected:
            raise RuntimeError("Cannot get markets - trading client not connected")

        if not tickers:
            return []

        try:
            response = await self._client.get_markets(tickers=tickers)
            self._metrics.api_calls += 1
            self._consecutive_api_errors = 0  # Reset on success
            return response.get("markets", [])

        except Exception as e:
            logger.error(f"Failed to get markets: {e}")
            self._metrics.api_errors += 1
            self._consecutive_api_errors += 1
            raise