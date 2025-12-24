"""
TRADER V3 Trading Client Integration.

Integration layer between V3 and the Kalshi trading client (paper/production).
Provides clean abstraction for order management and position tracking.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from decimal import Decimal

from ...trading.demo_client import KalshiDemoTradingClient
from ..core.event_bus import EventBus, EventType

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


@dataclass
class CalibrationData:
    """Data collected during calibration phase."""
    positions: Dict[str, Any]
    orders: Dict[str, Any]
    balance: Decimal
    portfolio_value: Decimal
    settlements: List[Dict[str, Any]]
    timestamp: float
    duration_ms: float


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
        
        self._metrics = TradingClientMetrics()
        self._running = False
        self._started_at: Optional[float] = None
        self._connected = False
        self._calibration_complete = False
        self._last_health_check: Optional[float] = None
        
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
            
            # Note: Custom events would require extending EventBus with a generic emit method
            # For now, we'll rely on logging and metrics for tracking
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Trading client connection failed: {e}")
            self._metrics.api_errors += 1
            return False
    
    async def calibrate(self, fetch_settlements: bool = True) -> CalibrationData:
        """
        Calibrate by syncing current positions and orders.
        
        This is called during the CALIBRATING state to sync our internal
        state with the exchange's current state.
        
        Args:
            fetch_settlements: Whether to fetch settlement history
            
        Returns:
            CalibrationData with current positions, orders, and balance
        """
        if not self._connected:
            raise RuntimeError("Cannot calibrate - trading client not connected")
        
        logger.info("Starting calibration - syncing positions and orders...")
        start_time = time.time()
        
        try:
            # Fetch current positions
            positions_response = await self._client.get_positions()
            positions = self._client.positions  # Client updates internal state
            
            # Fetch open orders
            orders_response = await self._client.get_orders()
            orders = self._client.orders  # Client updates internal state
            
            # Fetch current balance
            account_info = await self._client.get_account_info()
            balance = self._client.balance
            portfolio_value = Decimal(str(account_info.get("portfolio_value", 0))) / 100
            
            # Optionally fetch settlements (for P&L tracking)
            settlements = []
            if fetch_settlements:
                try:
                    settlements_response = await self._client.get_settlements()
                    settlements = settlements_response.get("settlements", [])
                    logger.info(f"Fetched {len(settlements)} settlements")
                except Exception as e:
                    logger.warning(f"Could not fetch settlements: {e}")
            
            # Update metrics
            self._metrics.active_positions = dict(positions)
            self._metrics.open_orders = dict(orders)
            self._metrics.balance = balance
            self._metrics.portfolio_value = portfolio_value
            self._metrics.positions_synced = len(positions)
            self._metrics.last_sync_time = time.time()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Create calibration data
            calibration_data = CalibrationData(
                positions=dict(positions),
                orders=dict(orders),
                balance=balance,
                portfolio_value=portfolio_value,
                settlements=settlements,
                timestamp=start_time,
                duration_ms=duration_ms
            )
            
            self._calibration_complete = True
            
            logger.info(
                f"✅ Calibration complete in {duration_ms:.1f}ms - "
                f"Positions: {len(positions)}, Orders: {len(orders)}, "
                f"Balance: ${balance}, Portfolio: ${portfolio_value}"
            )
            
            # Note: Custom events would require extending EventBus with a generic emit method
            # Calibration complete is tracked via state transition to READY
            
            return calibration_data
            
        except Exception as e:
            logger.error(f"❌ Calibration failed: {e}")
            self._metrics.api_errors += 1
            raise
    
    async def place_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price: Optional[int] = None,
        order_type: str = "limit"
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
            # Place order through client
            response = await self._client.create_order(
                ticker=ticker,
                action=action,
                side=side,
                count=count,
                price=price,
                type=order_type
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
            
            logger.info(f"✅ Order placed: {action} {count} {side} {ticker} @ {price}¢")
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
            "calibrated": self._calibration_complete,
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
            "calibrated": self._calibration_complete,
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