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
    trade_sequence_id: Optional[str] = None  # Unique ID for tracking through lifecycle
    limit_price_dollars: Optional[str] = None  # Limit price in dollars from Kalshi API (if available)
    
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
    cost_basis: float    # Total cost in cents (position_cost from Kalshi, converted from centi-cents)
    realized_pnl: float  # Cumulative realized P&L in cents
    kalshi_data: Dict[str, Any] = field(default_factory=dict)  # Raw Kalshi API response
    last_updated_ts: Optional[str] = None  # ISO timestamp of last update from Kalshi
    opened_at: Optional[float] = None  # Unix timestamp when position was opened (for max hold time check)
    
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
    fill_price_dollars: Optional[str] = None  # YES price in dollars from Kalshi API (e.g., "0.750")
    
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
                fill_price_dollars=fill_data.get("yes_price_dollars"),  # Extract dollars field if available
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
        # cash_balance = synced from Kalshi (only updated on sync)
        # _calculated_cash_balance = real-time tracking (updated from fills/orders)
        self.cash_balance = initial_cash      # Synced from Kalshi (authoritative)
        self._calculated_cash_balance = initial_cash  # Real-time calculated (internal tracking)
        self.promised_cash = 0.0              # Cash reserved for open orders
        self.initial_cash = initial_cash
        
        # Cash reserve protection
        from ..config import config
        self.min_cash_reserve = config.RL_MIN_CASH_RESERVE
        
        # Session tracking (initialized after sync with Kalshi)
        self.session_start_cash: Optional[float] = None
        self.session_start_portfolio_value: Optional[float] = None
        
        # Calculated vs synced tracking (for drift monitoring)
        # Synced values come from Kalshi API (only updated on sync)
        # Calculated values are our internal real-time tracking
        self._last_sync_drift_cash: Optional[float] = None  # How far off calculated was from synced at last sync (None = not synced yet)
        self._last_sync_drift_portfolio: Optional[float] = None  # How far off calculated was from synced at last sync (None = not synced yet)
        
        # Cashflow tracking
        self.session_cash_invested: float = 0.0  # Total cash spent on BUY orders
        self.session_cash_recouped: float = 0.0   # Total cash received from SELL orders
        self.session_total_fees_paid: float = 0.0  # Total trading fees paid
        
        # Order and position tracking
        self.open_orders: Dict[str, OrderInfo] = {}  # {our_order_id: OrderInfo}
        self.positions: Dict[str, Position] = {}     # {ticker: Position}
        self.settled_positions: Dict[str, Dict[str, Any]] = {}  # {ticker: settlement_data} - API-synced settlements
        
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
        
        # Track previous position state for change detection
        self._previous_position_state: Dict[str, Dict[str, Any]] = {}
        
        # Track active closing reasons per ticker (cleared after position goes flat)
        self._active_closing_reasons: Dict[str, str] = {}  # {ticker: reason}
        
        # Trader status tracking
        self._trader_status: str = "trading"  # Current trader status
        self._trader_status_history: List[Dict[str, Any]] = []  # Status transition history (max 50 entries)
        self._state_entry_time: Optional[float] = None  # When current state started
        self._previous_state: Optional[str] = None  # Previous state before current
        self._previous_state_duration: Optional[float] = None  # How long we were in previous state
        self._trading_stats: Dict[str, int] = {"trades": 0, "no_ops": 0}  # Trading session stats
        self._trading_stats_start_time: Optional[float] = None  # When current trading session started
        
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
    
    async def initialize(self, initialization_tracker=None) -> None:
        """
        Initialize the order manager and start fill processing.
        
        Requires valid Kalshi API credentials. Validates credentials before
        attempting connection and fails fast if missing.
        
        Args:
            initialization_tracker: Optional InitializationTracker for reporting progress
        
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
        if initialization_tracker:
            await initialization_tracker.mark_step_in_progress("trader_client_health")
        
        self.trading_client = KalshiDemoTradingClient()
        await self.trading_client.connect()
        logger.info("✅ Demo trading client connected")
        
        # Report trader client health
        if initialization_tracker:
            from ..config import config
            await initialization_tracker.mark_step_complete("trader_client_health", {
                "api_url": config.KALSHI_API_URL,
                "connected": True,
            })
            await initialization_tracker.update_component_health("trader_client", "healthy", {
                "api_url": config.KALSHI_API_URL,
                "connected": True,
            })
        
        # Start fill processor only after successful client connection
        self._fill_processor_task = asyncio.create_task(self._process_fills())
        logger.info("✅ Fill processor started")
        
        # Start fill listener (WebSocket for real-time fill notifications)
        # No fallbacks - if fill listener fails, initialization fails
        if initialization_tracker:
            await initialization_tracker.mark_step_in_progress("fill_listener_health")
        
        from .fill_listener import FillListener
        self._fill_listener = FillListener(order_manager=self)
        await self._fill_listener.start()
        logger.info("✅ Fill listener started")
        
        # Wait for WebSocket connection to establish (connection happens asynchronously)
        # Similar to orderbook - give it a moment to connect before health check
        max_wait_time = 5.0  # Maximum wait time in seconds
        wait_interval = 0.2   # Check every 200ms
        elapsed = 0.0
        fill_listener_healthy = False
        
        while elapsed < max_wait_time:
            # Check if the listener task has failed/exited
            if self._fill_listener._listener_task and self._fill_listener._listener_task.done():
                try:
                    await self._fill_listener._listener_task  # This will raise the exception if task failed
                except Exception as e:
                    error_msg = f"FillListener connection failed: {e}"
                    logger.error(error_msg)
                    if initialization_tracker:
                        health_details = self._fill_listener.get_health_details()
                        await initialization_tracker.mark_step_failed("fill_listener_health", error_msg, {
                            "details": health_details
                        })
                    raise RuntimeError(error_msg)
            
            if self._fill_listener.is_healthy():
                fill_listener_healthy = True
                logger.info("✅ Fill listener WebSocket connected and healthy")
                break
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval
        
        if not fill_listener_healthy:
            # If not healthy and task is still running, connection is failing
            error_msg = f"FillListener not healthy after {max_wait_time}s wait - WebSocket connection failed"
            logger.error(error_msg)
            if initialization_tracker:
                health_details = self._fill_listener.get_health_details()
                await initialization_tracker.mark_step_failed("fill_listener_health", error_msg, {
                    "details": health_details
                })
            raise RuntimeError(error_msg)
        
        # Report fill listener health (wrap in try/except to handle WebSocket implementation differences)
        if initialization_tracker:
            try:
                health_details = self._fill_listener.get_health_details()
            except (AttributeError, TypeError) as e:
                logger.warning(f"Could not get detailed fill listener health (WebSocket attribute check issue): {e}")
                # Use basic health info - we know it's healthy since is_healthy() passed
                health_details = {
                    "running": self._fill_listener._running,
                    "connected": True,  # We know it's connected if is_healthy() passed
                    "ws_url": self._fill_listener.ws_url,
                    "fills_received": getattr(self._fill_listener, '_fills_received', 0),
                }
            
            await initialization_tracker.mark_step_complete("fill_listener_health", {
                "details": health_details
            })
            await initialization_tracker.update_component_health("fill_listener", "healthy", health_details)
        
        # Start position listener (WebSocket for real-time position updates)
        # No fallbacks - if position listener fails, initialization fails
        if initialization_tracker:
            await initialization_tracker.mark_step_in_progress("position_listener_health")
        
        from .position_listener import PositionListener
        self._position_listener = PositionListener(order_manager=self)
        await self._position_listener.start()
        logger.info("✅ Position listener started")
        
        # Wait for WebSocket connection to establish (connection happens asynchronously)
        # Similar to fill listener - give it a moment to connect before health check
        max_wait_time = 5.0  # Maximum wait time in seconds
        wait_interval = 0.2   # Check every 200ms
        elapsed = 0.0
        position_listener_healthy = False
        
        while elapsed < max_wait_time:
            if self._position_listener.is_healthy():
                position_listener_healthy = True
                logger.info("✅ Position listener WebSocket connected and healthy")
                break
            await asyncio.sleep(wait_interval)
            elapsed += wait_interval
        
        if not position_listener_healthy:
            error_msg = f"PositionListener not healthy after {max_wait_time}s wait - WebSocket connection failed"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Report position listener health (wrap in try/except to handle WebSocket implementation differences)
        if initialization_tracker:
            try:
                health_details = self._position_listener.get_health_details()
            except (AttributeError, TypeError) as e:
                logger.warning(f"Could not get detailed position listener health (WebSocket attribute check issue): {e}")
                # Use basic health info - we know it's healthy since is_healthy() passed
                health_details = {
                    "running": self._position_listener._running,
                    "connected": True,  # We know it's connected if is_healthy() passed
                    "ws_url": self._position_listener.ws_url,
                    "positions_received": getattr(self._position_listener, '_positions_received', 0),
                }
            
            await initialization_tracker.mark_step_complete("position_listener_health", {
                "details": health_details
            })
            await initialization_tracker.update_component_health("position_listener", "healthy", health_details)
        
        # Check if cleanup is enabled BEFORE syncing (defaults to false - user requested always disabled)
        import os
        from ..config import config
        cleanup_enabled = os.getenv("RL_CLEANUP_ON_START", "false").lower() == "true"
        
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
        # No try/except wrapper - let exceptions propagate to fail initialization
        if config.RL_ORDER_SYNC_ENABLED and config.RL_ORDER_SYNC_ON_STARTUP:
            logger.info("Starting state synchronization with Kalshi...")
            
            # Use shared sync method for all state synchronization
            if initialization_tracker:
                await initialization_tracker.mark_step_in_progress("sync_balance")
            
            sync_results = await self._sync_state_with_kalshi()
            
            # Report sync results to initialization tracker
            if initialization_tracker:
                await initialization_tracker.mark_step_complete("sync_balance", {
                    "balance": sync_results["cash_balance"],
                    "balance_before": sync_results["cash_before"],
                    "portfolio_value": sync_results["portfolio_value"],
                })
            
            if initialization_tracker:
                await initialization_tracker.mark_step_in_progress("sync_positions")
                await initialization_tracker.mark_step_complete("sync_positions", {
                    "positions_count": sync_results["active_positions"],
                    "positions": {ticker: {"contracts": pos.contracts, "cost_basis": pos.cost_basis} 
                                for ticker, pos in self.positions.items()},
                })
            
            if initialization_tracker:
                await initialization_tracker.mark_step_in_progress("sync_settlements")
                await initialization_tracker.mark_step_complete("sync_settlements", {
                    "total_fetched": sync_results["settlement_stats"].get("total_fetched", 0),
                    "new": sync_results["settlement_stats"].get("new", 0),
                    "updated": sync_results["settlement_stats"].get("updated", 0),
                    "unchanged": sync_results["settlement_stats"].get("unchanged", 0),
                })
            
            if initialization_tracker:
                await initialization_tracker.mark_step_in_progress("sync_orders")
                order_stats = sync_results.get("order_stats", {})
                await initialization_tracker.mark_step_complete("sync_orders", {
                    "found_in_kalshi": order_stats.get("found_in_kalshi", 0),
                    "found_in_memory": order_stats.get("found_in_memory", 0),
                    "orders_added": order_stats.get("added", 0),
                    "orders_removed": order_stats.get("removed", 0),
                    "discrepancies": order_stats.get("discrepancies", 0),
                })
            
            order_stats = sync_results.get("order_stats", {})
            logger.info(
                f"State sync complete: Cash=${sync_results['cash_before']:.2f} → ${sync_results['cash_balance']:.2f}, "
                f"Portfolio=${sync_results['portfolio_value']:.2f}, "
                f"Positions={sync_results['active_positions']}, "
                f"Orders={order_stats.get('found_in_kalshi', 0)}"
            )
        
        # Start periodic synchronization (if enabled)
        if config.RL_ORDER_SYNC_ENABLED:
            self._periodic_sync_task = asyncio.create_task(self._start_periodic_sync())
            logger.info(f"✅ Periodic sync and recalibration started (interval: {config.RL_RECALIBRATION_INTERVAL_SECONDS}s)")
        
        # Set initial trader status and initialize stats
        # Stats will be reset when transitioning to trading in future, but need to initialize here
        self._reset_trading_stats()
        self._state_entry_time = time.time()  # Initialize state entry time
        await self._update_trader_status("trading", "initialized and ready")
        
        # Capture session start values AFTER all sync steps complete
        # Both cash balance and portfolio_value MUST come directly from Kalshi API - NO calculations
        # If sync completed successfully, both values are guaranteed to be available
        if self.session_start_cash is None:
            # Both values must be available from sync - fail if not
            if not hasattr(self, '_portfolio_value_from_kalshi') or self._portfolio_value_from_kalshi is None:
                raise RuntimeError(
                    "Portfolio value not available from Kalshi sync. "
                    "Both balance and portfolio_value must be fetched from /portfolio/balance endpoint. "
                    "Cannot proceed with initialization."
                )
            
            self.session_start_cash = self.cash_balance  # From Kalshi balance endpoint
            self.session_start_portfolio_value = self._portfolio_value_from_kalshi  # From Kalshi balance endpoint
            
            logger.info(
                f"Session start values captured from Kalshi API: "
                f"Cash=${self.session_start_cash:.2f}, "
                f"Portfolio=${self.session_start_portfolio_value:.2f} "
                f"(both from /portfolio/balance endpoint - no calculations)"
            )
        
        # Update last sync time
        self._last_sync_time = time.time()
        
        logger.info("✅ KalshiMultiMarketOrderManager ready for trading")
        
        # Verify listeners are subscribed
        if initialization_tracker:
            await initialization_tracker.mark_step_in_progress("verify_fill_listener_subscription")
            if self._fill_listener and self._fill_listener.is_healthy():
                await initialization_tracker.mark_step_complete("verify_fill_listener_subscription", {
                    "fill_listener_active": True,
                })
            else:
                await initialization_tracker.mark_step_failed("verify_fill_listener_subscription", "Fill listener not active")
            
            await initialization_tracker.mark_step_in_progress("verify_position_listener_subscription")
            if self._position_listener and self._position_listener.is_healthy():
                await initialization_tracker.mark_step_complete("verify_position_listener_subscription", {
                    "position_listener_active": True,
                })
            else:
                await initialization_tracker.mark_step_failed("verify_position_listener_subscription", "Position listener not active")
            
            await initialization_tracker.mark_step_in_progress("verify_listeners")
            # Gather listener details for context
            listener_details = {
                "fill_listener": self._fill_listener is not None,
                "position_listener": self._position_listener is not None,
                "state_change_callbacks": True,  # Set later in app.py
            }
            
            # Add fill listener details if available
            if self._fill_listener:
                try:
                    fill_health = self._fill_listener.get_health_details() if hasattr(self._fill_listener, 'get_health_details') else {}
                    listener_details["fill_listener_ws_url"] = getattr(self._fill_listener, 'ws_url', 'N/A')
                    listener_details["fill_listener_connected"] = self._fill_listener.is_healthy() if hasattr(self._fill_listener, 'is_healthy') else True
                except Exception:
                    pass
            
            # Add position listener details if available
            if self._position_listener:
                try:
                    pos_health = self._position_listener.get_health_details() if hasattr(self._position_listener, 'get_health_details') else {}
                    listener_details["position_listener_ws_url"] = getattr(self._position_listener, 'ws_url', 'N/A')
                    listener_details["position_listener_connected"] = self._position_listener.is_healthy() if hasattr(self._position_listener, 'is_healthy') else True
                except Exception:
                    pass
            
            await initialization_tracker.mark_step_complete("verify_listeners", listener_details)
        
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
            
            # Sync state to verify cleanup was successful
            sync_results = await self._sync_state_with_kalshi()
            sync_stats = sync_results["order_stats"]
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
        
        # Stop position listener
        if self._position_listener:
            try:
                await self._position_listener.stop()
                logger.info("✅ Position listener stopped")
            except Exception as e:
                logger.warning(f"Error stopping position listener: {e}")
            self._position_listener = None
        
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
        orderbook_snapshot: Optional[Dict[str, Any]] = None,
        trade_sequence_id: Optional[str] = None,
        reason: Optional[str] = None
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
            return {"status": "hold", "action": action, "market": market_ticker, "trade_sequence_id": trade_sequence_id}
        
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
            
            # Fixed contract size (5 contracts)
            quantity = 5
            
            # Calculate limit price from orderbook snapshot
            limit_price = self._calculate_limit_price_from_snapshot(
                orderbook_snapshot, side, contract_side
            )
            
            # Check cash reserve threshold (applies to all trading actions)
            if self.cash_balance < self.min_cash_reserve:
                logger.warning(
                    f"Cash reserve threshold hit: ${self.cash_balance:.2f} < ${self.min_cash_reserve:.2f}. "
                    f"Trading stopped to maintain reserve."
                )
                return {
                    "status": "cash_reserve_hit",
                    "cash_balance": self.cash_balance,
                    "min_reserve": self.min_cash_reserve,
                    "message": f"Cash balance ${self.cash_balance:.2f} below reserve ${self.min_cash_reserve:.2f}"
                }
            
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
                limit_price=limit_price,
                trade_sequence_id=trade_sequence_id
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
                execution_result = {
                    "status": "placed",
                    "order_id": result["order_id"],
                    "action": action,
                    "market": market_ticker,
                    "side": side.name,
                    "contract_side": contract_side.name,
                    "quantity": quantity,
                    "limit_price": limit_price,
                    "trade_sequence_id": trade_sequence_id
                }
                if reason:
                    execution_result["reason"] = reason
                return execution_result
            else:
                return {"status": "failed", "reason": "kalshi_api_error"}
        
        except Exception as e:
            logger.error(f"Error executing order for {market_ticker}: {e}")
            return {"status": "error", "error": str(e)}
    
    async def execute_limit_order_action(
        self,
        action: int,
        market_ticker: str,
        orderbook_snapshot: Optional[Dict[str, Any]] = None,
        trade_sequence_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Execute limit order action (wrapper for ActorService integration).
        
        This method provides the interface expected by ActorService while
        delegating to execute_order() for actual implementation.
        
        Args:
            action: Action ID (0=HOLD, 1=BUY_YES_LIMIT, 2=SELL_YES_LIMIT, 3=BUY_NO_LIMIT, 4=SELL_NO_LIMIT)
            market_ticker: Market to trade
            orderbook_snapshot: Orderbook snapshot for price calculation
            reason: Optional reason for the action (e.g., "close_position:take_profit")
            
        Returns:
            Execution result dict with "executed" key for ActorService compatibility
        """
        result = await self.execute_order(market_ticker, action, orderbook_snapshot, trade_sequence_id, reason)
        
        # Normalize result format for ActorService
        if result is None:
            return None
        
        # Convert status to "executed" boolean for ActorService
        if result.get("status") == "placed":
            result["executed"] = True
        else:
            result["executed"] = False
        
        return result
    
    async def close_position(
        self,
        ticker: str,
        reason: str,
        orderbook_snapshot: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Close a position by placing an opposite order.
        
        Args:
            ticker: Market ticker
            reason: Reason for closing (e.g., "take_profit", "stop_loss", "cash_recovery", "market_closing")
            orderbook_snapshot: Optional orderbook snapshot for price calculation
            
        Returns:
            Execution result dict or None if failed
        """
        logger.info(f"close_position called for {ticker} with reason: {reason}")
        
        position = self.positions.get(ticker)
        if not position or position.is_flat:
            logger.warning(f"No position to close for {ticker} (position exists: {position is not None}, is_flat: {position.is_flat if position else 'N/A'})")
            return None
        
        logger.info(f"Closing position {ticker}: {position.contracts} contracts, cost_basis=${position.cost_basis/100:.2f}, reason={reason}")
        
        # Determine opposite action based on current position
        # Long YES (contracts > 0) -> SELL YES (action 2)
        # Long NO (contracts < 0) -> SELL NO (action 4)
        if position.contracts > 0:
            # Long YES -> SELL YES
            action = 2  # SELL_YES_LIMIT
            action_name = "SELL_YES_LIMIT"
        else:
            # Long NO -> SELL NO
            action = 4  # SELL_NO_LIMIT
            action_name = "SELL_NO_LIMIT"
        
        logger.debug(f"Position {ticker} is long {abs(position.contracts)} {'YES' if position.contracts > 0 else 'NO'} contracts, using action {action} ({action_name})")
        
        # Get orderbook snapshot if not provided
        if orderbook_snapshot is None:
            logger.debug(f"Fetching orderbook snapshot for {ticker}")
            orderbook_snapshot = await self._get_orderbook_snapshot(ticker)
            if orderbook_snapshot is None:
                logger.error(f"Could not get orderbook snapshot for {ticker}, cannot close position")
                return None
            logger.debug(f"Orderbook snapshot retrieved for {ticker}")
        else:
            logger.debug(f"Using provided orderbook snapshot for {ticker}")
        
        # Generate trade sequence ID for closing action
        trade_sequence_id = f"close_{ticker}_{int(time.time() * 1000)}"
        logger.debug(f"Generated trade_sequence_id: {trade_sequence_id}")
        
        # Execute closing order with reason
        closing_reason = f"close_position:{reason}"
        logger.info(f"Executing closing order for {ticker}: action={action}, reason={closing_reason}")
        
        result = await self.execute_limit_order_action(
            action=action,
            market_ticker=ticker,
            orderbook_snapshot=orderbook_snapshot,
            trade_sequence_id=trade_sequence_id,
            reason=closing_reason
        )
        
        if result and result.get("executed"):
            # Track closing reason for this ticker
            self._active_closing_reasons[ticker] = reason
            logger.info(f"Position closing initiated successfully: {ticker} ({reason}) -> order_id={result.get('order_id')}, status={result.get('status')}")
        else:
            error_msg = result.get("error") if result else "No result returned"
            status = result.get("status") if result else "unknown"
            logger.error(f"Failed to close position {ticker} ({reason}): status={status}, error={error_msg}, result={result}")
        
        return result
    
    async def _get_orderbook_snapshot(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get orderbook snapshot for a market ticker.
        
        Args:
            ticker: Market ticker
            
        Returns:
            Orderbook snapshot dict or None if unavailable
        """
        try:
            from ..data.orderbook_state import get_shared_orderbook_state
            shared_state = await get_shared_orderbook_state(ticker)
            if shared_state:
                return await shared_state.get_snapshot()
        except Exception as e:
            logger.debug(f"Could not get orderbook snapshot for {ticker}: {e}")
        return None
    
    async def _get_market_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get market info from Kalshi API (status, end time, etc.).
        
        Args:
            ticker: Market ticker
            
        Returns:
            Market info dict or None if unavailable
        """
        if not self.trading_client:
            return None
        
        try:
            # Use get_markets and filter by ticker
            # Note: Kalshi API doesn't have a direct get_market endpoint, so we use get_markets
            markets_response = await self.trading_client.get_markets(limit=100)
            markets = markets_response.get("markets", [])
            
            for market in markets:
                if market.get("ticker") == ticker:
                    return market
            
            logger.debug(f"Market {ticker} not found in markets list")
            return None
        except Exception as e:
            logger.debug(f"Could not get market info for {ticker}: {e}")
            return None
    
    async def _place_kalshi_order(
        self,
        ticker: str,
        side: OrderSide,
        contract_side: ContractSide,
        quantity: int,
        limit_price: int,
        trade_sequence_id: Optional[str] = None
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
                # Reserve cash immediately (update calculated, not synced)
                self._calculated_cash_balance -= promised_cash
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
                # Restore cash on failure (update calculated, not synced)
                if side == OrderSide.BUY:
                    self._calculated_cash_balance += promised_cash
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
                promised_cash=promised_cash,
                trade_sequence_id=trade_sequence_id
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
            # Restore cash on error (update calculated, not synced)
            if side == OrderSide.BUY and 'promised_cash' in locals():
                self._calculated_cash_balance += promised_cash
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
                # Restore promised cash (update calculated, not synced)
                self._calculated_cash_balance += order.promised_cash
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
            
            # BUY: deduct cash for the filled quantity (update calculated)
            self._calculated_cash_balance -= fill_cost
            
            logger.debug(
                f"BUY fill cash: released ${promised_cash_released:.2f} of ${order.promised_cash + promised_cash_released:.2f} "
                f"({fill_quantity}/{order.quantity} contracts), deducted ${fill_cost:.2f} from calculated cash"
            )
        else:
            # SELL: add cash received for the filled quantity (update calculated)
            self._calculated_cash_balance += fill_cost
            logger.debug(f"SELL fill cash: received ${fill_cost:.2f} (added to calculated cash)")
        
        # Track cashflow and fees for this fill
        fill_value = fill_cost  # fill_cost already calculated as (fill_price / 100.0) * fill_quantity
        
        # Calculate trading fee (Kalshi fee structure: 0.7% for taker orders)
        # Use is_taker from fill_event, default to True if not specified (conservative estimate)
        is_taker = fill_event.is_taker if hasattr(fill_event, 'is_taker') else True
        fee_rate = 0.007 if is_taker else 0.0  # 0.7% for taker, 0% for maker (maker rebate not tracked as negative fee)
        fill_fee = fill_value * fee_rate
        self.session_total_fees_paid += fill_fee
        
        # Track cashflow based on whether we're opening/adding or closing/reducing positions
        # Check position state BEFORE updating to determine the nature of the trade
        ticker = order.ticker
        position_before = self.positions.get(ticker)
        contracts_before = position_before.contracts if position_before else 0
        
        # Calculate contract change (same logic as _update_position)
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
        
        # Determine if opening/adding or closing/reducing
        is_opening_or_adding = (
            (contracts_before == 0) or  # Opening new position
            (contracts_before > 0 and contract_change > 0) or  # Adding to long
            (contracts_before < 0 and contract_change < 0)  # Adding to short
        )
        
        is_closing_or_reducing = (
            (contracts_before > 0 and contract_change < 0) or  # Reducing long
            (contracts_before < 0 and contract_change > 0)  # Reducing short
        )
        
        # Track cashflow correctly
        # Invested: Cash spent to open/add to positions (positive = spending, negative = receiving)
        # Recouped: Cash received from closing/reducing positions (positive = receiving, negative = spending)
        if is_opening_or_adding:
            if order.side == OrderSide.BUY:
                # BUY opening/adding: spending cash to acquire position
                self.session_cash_invested += fill_cost
            else:
                # SELL opening/adding short: receiving cash to open short position
                # This is negative investment (we're being paid to take a position)
                self.session_cash_invested -= fill_cost
        elif is_closing_or_reducing:
            if order.side == OrderSide.SELL:
                # SELL closing/reducing: receiving cash from closing position
                self.session_cash_recouped += fill_cost
            else:
                # BUY closing/reducing short: spending cash to close short position
                # This is negative recoup (we're paying to close)
                self.session_cash_recouped -= fill_cost
        
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
        fill_dict = {
            "kalshi_order_id": fill_event.kalshi_order_id,
            "fill_quantity": fill_event.fill_quantity,
            "fill_price": fill_event.fill_price,
            "fill_timestamp": fill_event.fill_timestamp,
            "is_taker": fill_event.is_taker,
            "market_ticker": fill_event.market_ticker,
            "trade_sequence_id": order.trade_sequence_id,
            "side": order.side.name,
            "contract_side": order.contract_side.name
        }
        # Include fill_price_dollars if available (from FillEvent)
        if fill_event.fill_price_dollars is not None:
            fill_dict["fill_price_dollars"] = fill_event.fill_price_dollars
            fill_dict["yes_price_dollars"] = fill_event.fill_price_dollars  # Also include as yes_price_dollars for compatibility
        
        fill_data = {
            "fill": fill_dict,
            "updated_position": {
                "ticker": order.ticker,
                "contracts": self.positions.get(order.ticker).contracts if order.ticker in self.positions else 0,
                "average_cost_cents": int(self.positions.get(order.ticker).cost_basis * 100) if order.ticker in self.positions else 0
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
            trade_dict = {
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp,
                "ticker": trade.ticker,
                "action": trade.action,
                "quantity": trade.quantity,
                "fill_price": trade.fill_price,
                "order_id": trade.order_id,
                "model_decision": trade.model_decision
            }
            # Include fill_price_dollars if available (TradeDetail doesn't have it yet, but prepare for future)
            # For now, we'll rely on FillEvent broadcasts which include it
            recent_fills_data.append(trade_dict)

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
            order_dict = {
                "order_id": order_id,
                "kalshi_order_id": order.kalshi_order_id,
                "ticker": order.ticker,
                "side": order.side.name,
                "contract_side": order.contract_side.name,
                "quantity": order.quantity,
                "limit_price": order.limit_price,
                "status": order.status.name,
                "placed_at": order.placed_at,
                "promised_cash": order.promised_cash,
                "trade_sequence_id": order.trade_sequence_id
            }
            # Include limit_price_dollars if available
            if order.limit_price_dollars is not None:
                order_dict["limit_price_dollars"] = order.limit_price_dollars
            orders_data.append(order_dict)
        
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
            
        # Portfolio value: Use Kalshi API value (synced), calculate internal value separately
        if hasattr(self, '_portfolio_value_from_kalshi') and self._portfolio_value_from_kalshi is not None:
            portfolio_value = self._portfolio_value_from_kalshi  # Synced from Kalshi
        else:
            portfolio_value = self._calculate_portfolio_value()  # Fallback to calculated
        
        # Get positions data for broadcast
        # All monetary values from Kalshi API are in cents - use directly, no conversion
        positions_data = {}
        for ticker, position in self.positions.items():
            # Extract market exposure from kalshi_data - API provides both cents and dollars
            kalshi_data = position.kalshi_data or {}
            market_exposure_cents = kalshi_data.get("market_exposure")  # Already in cents from API
            market_exposure_dollars = kalshi_data.get("market_exposure_dollars")  # String dollars for reference
            
            # Convert to float if needed
            if market_exposure_cents is not None:
                market_exposure_cents = float(market_exposure_cents) if isinstance(market_exposure_cents, str) else market_exposure_cents
            
            positions_data[ticker] = {
                "position": position.contracts,
                "contracts": position.contracts,
                "cost_basis": position.cost_basis,  # In cents
                "average_cost_cents": int(position.cost_basis),  # Already in cents
                "market_exposure_cents": market_exposure_cents,  # In cents (from API)
                "market_exposure_dollars": market_exposure_dollars,  # String dollars (from API, for reference)
                "realized_pnl": position.realized_pnl,  # In cents
                "fees_paid": getattr(position, 'fees_paid', 0.0),  # In cents
                "volume": getattr(position, 'volume', 0),
                "last_updated_ts": position.last_updated_ts
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
            
        # Portfolio value: Use Kalshi API value (synced), calculate internal value separately
        if hasattr(self, '_portfolio_value_from_kalshi') and self._portfolio_value_from_kalshi is not None:
            portfolio_value = self._portfolio_value_from_kalshi  # Synced from Kalshi
        else:
            portfolio_value = self._calculate_portfolio_value()  # Fallback to calculated
        
        await self._websocket_manager.broadcast_portfolio_update({
            "cash_balance": self.cash_balance,
            "portfolio_value": portfolio_value
        })
        logger.debug(f"Broadcast portfolio update: cash={self.cash_balance:.2f}, portfolio={portfolio_value:.2f} (event: {event_type})")
    
    def _extract_base_state(self, status: str) -> str:
        """
        Extract base state from status string.
        
        Base states: "initializing", "trading", "calibrating", "paused", "stopping", "low_cash"
        Sub-states like "calibrating -> syncing state" return "calibrating"
        """
        if " -> " in status:
            # Extract base state before arrow (e.g., "calibrating -> syncing state" -> "calibrating")
            return status.split(" -> ")[0]
        return status
    
    def get_current_status(self) -> str:
        """
        Get current trader status.
        
        Returns:
            Current trader status string
        """
        return self._trader_status
    
    async def _update_trader_status(
        self, 
        status: str, 
        result: Optional[str] = None,
        duration: Optional[float] = None
    ) -> None:
        """
        Update trader status and broadcast via WebSocket.
        
        Args:
            status: New trader status (e.g., "trading", "calibrating", "calibrating -> closing positions")
            result: Optional result message (e.g., "closed 2 positions", "no positions to close")
            duration: Optional duration in seconds for this status
        """
        timestamp = time.time()
        
        # Extract base states for comparison (before updating _trader_status)
        current_base_state = self._extract_base_state(self._trader_status)
        new_base_state = self._extract_base_state(status)
        
        # Calculate time in current state
        if self._state_entry_time is None:
            time_in_status = 0.0
        else:
            time_in_status = timestamp - self._state_entry_time
        
        # Check if base state changed
        state_changed = current_base_state != new_base_state
        
        # Handle state change: track previous state and reset timers/stats
        if state_changed:
            # Calculate and store previous state info for header display
            if self._state_entry_time is not None:
                self._previous_state_duration = timestamp - self._state_entry_time
            else:
                self._previous_state_duration = 0.0
            
            # Store previous state info for header display
            if current_base_state and current_base_state != "initializing":
                self._previous_state = current_base_state
            else:
                self._previous_state = None
                self._previous_state_duration = None
            
            # Reset state entry time for new state
            self._state_entry_time = timestamp
            
            # Reset trading stats if entering trading state
            if new_base_state == "trading":
                self._reset_trading_stats()
            
            # Reset time_in_status for new state entry
            time_in_status = 0.0
        
        # Build result with timing if provided
        full_result = result
        if duration is not None and result:
            full_result = f"{result} ({duration:.1f}s)"
        elif duration is not None:
            full_result = f"({duration:.1f}s)"
        
        # Always log status entry (single code path)
        status_entry = {
            "timestamp": timestamp,
            "status": status,
            "result": full_result,
            "duration": duration,
            "time_in_status": time_in_status
        }
        self._trader_status_history.append(status_entry)
        if len(self._trader_status_history) > 50:
            self._trader_status_history.pop(0)
        
        # Update current status
        self._trader_status = status
        
        # Broadcast status update via WebSocket
        if self._websocket_manager:
            status_data = {
                "current_status": status,
                "time_in_status": time_in_status,
                "previous_state": self._previous_state,
                "previous_state_duration": self._previous_state_duration,
                "status_history": self._trader_status_history[-20:]  # Send last 20 entries
            }
            await self._websocket_manager.broadcast_trader_status(status_data)
        
        logger.info(f"Trader status: {status}" + (f" - {full_result}" if full_result else ""))
    
    def _update_trading_stats(self, action: int, executed: bool) -> None:
        """
        Update trading session statistics.
        
        Args:
            action: Action taken (0=HOLD, 1=BUY_YES, 2=SELL_YES, 3=BUY_NO, 4=SELL_NO)
            executed: Whether the order was successfully executed (placed)
        """
        # Only track stats when in trading state
        if self._extract_base_state(self._trader_status) != "trading":
            return
        
        # Trades: executed orders (non-HOLD actions that were placed)
        if executed and action != 0:  # 0 is HOLD
            self._trading_stats["trades"] += 1
        
        # No-ops: HOLD actions, failed actions, or throttled actions
        if action == 0 or not executed:
            self._trading_stats["no_ops"] += 1
    
    def _reset_trading_stats(self) -> None:
        """Reset trading session statistics."""
        self._trading_stats = {"trades": 0, "no_ops": 0}
        self._trading_stats_start_time = time.time()
    
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
        """Get available cash balance (synced from Kalshi)."""
        return self.cash_balance
    
    def _calculate_cash_balance(self) -> float:
        """
        Get calculated cash balance from internal real-time tracking.
        
        This is our real-time calculated value based on:
        - Initial cash
        - Orders placed (deducted)
        - Orders cancelled (restored)
        - Fills processed
        """
        return self._calculated_cash_balance
    
    def _calculate_portfolio_value(self) -> float:
        """
        Calculate portfolio value from internal tracking (positions + cash).
        
        This is our real-time calculated value based on:
        - Cash balance (calculated)
        - Position values (cost basis + unrealized P&L)
        """
        total = self._calculate_cash_balance() + self.promised_cash
        
        # Add position values (cost basis + realized P&L + unrealized P&L if we have prices)
        # Note: position.cost_basis and position.realized_pnl are in cents, convert to dollars
        for position in self.positions.values():
            if not position.is_flat:
                # Convert cents to dollars
                total += (position.cost_basis + position.realized_pnl) / 100.0
                # Note: We don't add unrealized P&L here since we don't have current prices
                # This matches the calculation used in get_portfolio_value() without prices
        
        return total
    
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
    
    async def sync_orders_with_kalshi(self) -> Dict[str, Any]:
        """
        Sync local order state with Kalshi.
        
        Kalshi is the source of truth - local state is updated to match Kalshi.
        
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
                await self._reconcile_order(kalshi_order_id, kalshi_order, stats)
            
            # Check for orders in memory but not in Kalshi
            # If Kalshi doesn't have it, it doesn't exist - remove from local state
            for our_order_id, order in list(self.open_orders.items()):
                if order.kalshi_order_id not in kalshi_orders:
                    # Order was cancelled externally or doesn't exist in Kalshi
                    # Trust Kalshi: remove from local tracking
                    await self._handle_external_cancellation(order, stats)
            
            # Log warnings if discrepancies were found
            actual_discrepancies = stats["discrepancies"] + stats["removed"]
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
        stats: Dict[str, Any]
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
                            # SELL: add cash received (update calculated, not synced)
                            fill_cost = (fill_price / 100.0) * filled_quantity
                            self._calculated_cash_balance += fill_cost
                        
                        # Update position
                        self._update_position(local_order, fill_price, filled_quantity)
                        self._orders_filled += 1
                        self._total_volume_traded += (fill_price / 100.0) * filled_quantity
                    
                    # Remove from tracking
                    if local_order.side == OrderSide.BUY:
                        # Release any remaining promised cash (update calculated, not synced)
                        self._calculated_cash_balance += local_order.promised_cash
                        self.promised_cash -= local_order.promised_cash
                    
                    del self.open_orders[our_order_id]
                    del self._kalshi_to_internal[kalshi_order_id]
                    stats["updated"] += 1
                    
                elif kalshi_status_mapped == OrderStatus.CANCELLED:
                    # Order was cancelled - restore cash if BUY (update calculated, not synced)
                    if local_order.side == OrderSide.BUY:
                        self._calculated_cash_balance += local_order.promised_cash
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
            logger.warning(
                f"⚠️ ORDER NOT IN LOCAL MEMORY: Order {kalshi_order_id} ({ticker}) "
                f"found in Kalshi but not tracked locally. "
                f"Adding to local tracking."
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
            # SELL: add cash received (update calculated, not synced)
            self._calculated_cash_balance += fill_cost
        
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
                    self._calculated_cash_balance += fill_cost
                self._orders_filled += 1
                self._total_volume_traded += (fill_price / 100.0) * fill_count
            return
        
        # Generate internal order ID
        our_order_id = self._generate_order_id()
        
        # Calculate promised cash (if BUY and still pending)
        promised_cash = 0.0
        if side == OrderSide.BUY and status == OrderStatus.PENDING:
            promised_cash = (limit_price / 100.0) * remaining_count
            # Reserve cash (update calculated, not synced)
            self._calculated_cash_balance -= promised_cash
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
        
        # Restore promised cash if BUY order (update calculated, not synced)
        if order.side == OrderSide.BUY:
            self._calculated_cash_balance += order.promised_cash
            self.promised_cash -= order.promised_cash
        
        # Remove from tracking
        del self.open_orders[order.order_id]
        if order.kalshi_order_id in self._kalshi_to_internal:
            del self._kalshi_to_internal[order.kalshi_order_id]
        
        stats["removed"] += 1
        self._orders_cancelled += 1
    
    async def _sync_positions_with_kalshi(self) -> None:
        """
        Sync positions with Kalshi.
        
        Kalshi positions are authoritative - update local positions to match.
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
                
                # Extract cost basis from API data
                # Kalshi API provides values in cents (numeric fields) and dollars (string _dollars fields)
                # We use the cents values directly - no conversion needed
                # Cost basis = what we paid = market_exposure - realized_pnl (approximate)
                cost_basis = 0.0
                market_exposure_cents = kalshi_pos.get("market_exposure")  # Already in cents from API
                realized_pnl_cents = kalshi_pos.get("realized_pnl")  # Already in cents from API
                
                if market_exposure_cents is not None and realized_pnl_cents is not None:
                    # Both values are already in cents from API
                    exposure = float(market_exposure_cents) if isinstance(market_exposure_cents, str) else market_exposure_cents
                    pnl = float(realized_pnl_cents) if isinstance(realized_pnl_cents, str) else realized_pnl_cents
                    # Cost basis = what we paid = current value - profit (all in cents)
                    cost_basis = exposure - pnl
                elif "position_cost" in kalshi_pos:
                    # If position_cost is in the API response (shouldn't happen but handle it)
                    # Assume it's in centi-cents, convert to cents
                    cost_basis = float(kalshi_pos["position_cost"]) / 100.0 if isinstance(kalshi_pos["position_cost"], (int, float)) else 0.0
                
                # Extract realized P&L - use cents value directly from API
                realized_pnl = 0.0
                if realized_pnl_cents is not None:
                    realized_pnl = float(realized_pnl_cents) if isinstance(realized_pnl_cents, str) else realized_pnl_cents
                elif "realized_pnl" in kalshi_pos:
                    # Fallback: assume it's in centi-cents, convert to cents
                    realized_pnl = float(kalshi_pos["realized_pnl"]) / 100.0 if isinstance(kalshi_pos["realized_pnl"], (int, float)) else 0.0
                
                # Extract last_updated_ts
                last_updated_ts = kalshi_pos.get("last_updated_ts")
                
                # Extract fees_paid from API (already in cents)
                fees_paid_cents = kalshi_pos.get("fees_paid", 0)
                fees_paid_value = float(fees_paid_cents) if isinstance(fees_paid_cents, str) else fees_paid_cents
                
                if ticker not in self.positions:
                    # New position - store the entire Kalshi response
                    new_position = Position(
                        ticker=ticker,
                        contracts=contracts,
                        cost_basis=cost_basis,
                        realized_pnl=realized_pnl,
                        kalshi_data=kalshi_pos,  # Store complete Kalshi response
                        last_updated_ts=last_updated_ts
                    )
                    # Store fees_paid as attribute (in cents)
                    new_position.fees_paid = fees_paid_value
                    self.positions[ticker] = new_position
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
                    
                    # Preserve cost_basis from WebSocket if available (more accurate than API calculation)
                    # Only update if we don't have a cost_basis or if API provides better data
                    if cost_basis > 0.0 and (local_pos.cost_basis == 0.0 or local_pos.cost_basis is None):
                        local_pos.cost_basis = cost_basis
                    
                    # Update realized P&L from API (already in cents)
                    if realized_pnl != 0.0 or local_pos.realized_pnl == 0.0:
                        local_pos.realized_pnl = realized_pnl
                    
                    # Update fees_paid from API (already in cents)
                    fees_paid_cents = kalshi_pos.get("fees_paid")
                    if fees_paid_cents is not None:
                        fees_paid_value = float(fees_paid_cents) if isinstance(fees_paid_cents, str) else fees_paid_cents
                        local_pos.fees_paid = fees_paid_value
                    
                    # Always update with complete Kalshi response (authoritative source)
                    local_pos.contracts = contracts
                    local_pos.kalshi_data = kalshi_pos
                    
                    # Update last_updated_ts if provided and newer than current
                    if last_updated_ts:
                        from datetime import datetime
                        current_ts = local_pos.last_updated_ts
                        if current_ts:
                            # Parse both timestamps and compare
                            try:
                                current_dt = datetime.fromisoformat(current_ts.replace('Z', '+00:00'))
                                new_dt = datetime.fromisoformat(last_updated_ts.replace('Z', '+00:00'))
                                if new_dt > current_dt:
                                    local_pos.last_updated_ts = last_updated_ts
                            except (ValueError, AttributeError) as e:
                                # If parsing fails, log and use the new timestamp
                                logger.warning(f"Failed to parse timestamp for {ticker}: {e}, using new timestamp")
                                local_pos.last_updated_ts = last_updated_ts
                        else:
                            # No existing timestamp, use the new one
                            local_pos.last_updated_ts = last_updated_ts
            
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
                logger.warning(
                    f"⚠️ POSITION SYNC DISCREPANCIES DETECTED: "
                    f"{len(position_discrepancies)} position(s) out of sync - {', '.join(position_discrepancies)}"
                )
            else:
                logger.debug(f"Position sync complete: {len(self.positions)} positions - all in sync")
            
        except Exception as e:
            logger.error(f"Error syncing positions with Kalshi: {e}")
    
    async def sync_settlements_with_kalshi(self) -> Dict[str, Any]:
        """
        Sync settlements from Kalshi API.
        
        Fetches settlements from the past 24 hours and stores them with exact API structure.
        All fields except fee_cost are in cents. fee_cost is a string in dollars.
        
        Returns:
            Dictionary with sync statistics (count, new, updated, etc.)
        """
        if not self.trading_client:
            logger.error("Cannot sync settlements: trading client not initialized")
            return {"error": "Trading client not initialized"}
        
        try:
            # Fetch all settlements (no query parameters for simplicity)
            logger.info("Fetching all settlements from Kalshi...")
            try:
                settlements_response = await self.trading_client.get_settlements()
            except Exception as api_error:
                logger.error(f"Settlements API call failed: {api_error}")
                logger.error(f"Error type: {type(api_error).__name__}")
                raise
            
            settlements_list = settlements_response.get("settlements", [])
            logger.info(f"Retrieved {len(settlements_list)} settlements from Kalshi")
            
            # Sort settlements by settled_time (most recent first)
            # settled_time is ISO format string like "2023-11-07T05:31:56Z"
            def get_settled_timestamp(settlement):
                settled_time = settlement.get("settled_time", "")
                if settled_time:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(settled_time.replace('Z', '+00:00'))
                        return dt.timestamp()
                    except (ValueError, AttributeError):
                        return 0
                return 0
            
            settlements_list.sort(key=get_settled_timestamp, reverse=True)  # Most recent first
            logger.debug(f"Sorted {len(settlements_list)} settlements by settled_time (most recent first)")
            
            # Log actual field names from first settlement for debugging
            if settlements_list:
                first_settlement = settlements_list[0]
                logger.debug(f"Settlement API structure - available fields: {list(first_settlement.keys())}")
                # Specifically log revenue and value related fields
                revenue_fields = {k: v for k, v in first_settlement.items() if 'revenue' in k.lower()}
                value_fields = {k: v for k, v in first_settlement.items() if 'value' in k.lower()}
                if revenue_fields:
                    logger.debug(f"Revenue-related fields found: {revenue_fields}")
                if value_fields:
                    logger.debug(f"Value-related fields found: {value_fields}")
            
            stats = {
                "total_fetched": len(settlements_list),
                "new": 0,
                "updated": 0,
                "unchanged": 0
            }
            
            # Process each settlement
            for settlement in settlements_list:
                ticker = settlement.get("ticker", "")
                if not ticker:
                    logger.warning(f"Settlement missing ticker: {settlement}")
                    continue
                
                # Store exact API structure
                # This copies all fields including revenue and value (both integers in cents per Kalshi API)
                # According to Kalshi API docs: https://docs.kalshi.com/api-reference/portfolio/get-settlements
                # revenue: integer in cents
                # value: integer in cents
                # Always use API values, never calculate locally - settlements come only from API sync
                settlement_data = dict(settlement)  # Copy all fields as-is (revenue and value already included)
                
                # Convert fee_cost from string dollars to cents
                fee_cost_str = settlement.get("fee_cost", "0.0")
                try:
                    fee_cost_dollars = float(fee_cost_str)
                    fee_cost_cents = int(fee_cost_dollars * 100)
                    settlement_data["fee_cost_cents"] = fee_cost_cents
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse fee_cost '{fee_cost_str}' for {ticker}: {e}")
                    settlement_data["fee_cost_cents"] = 0
                
                # Extract revenue for P&L calculation (already in settlement_data via dict copy above)
                # revenue and value are preserved in settlement_data as-is (integers in cents from API)
                revenue = settlement.get("revenue")
                
                # Calculate final P&L: revenue - yes_total_cost - no_total_cost - fee_cost_cents
                # Use revenue from API (in cents), default to 0 if not present
                revenue_for_pnl = revenue if revenue is not None else 0
                yes_total_cost = settlement.get("yes_total_cost", 0)
                no_total_cost = settlement.get("no_total_cost", 0)
                final_pnl = revenue_for_pnl - yes_total_cost - no_total_cost - settlement_data["fee_cost_cents"]
                settlement_data["final_pnl"] = final_pnl
                
                # Check if settlement already exists
                existing_settlement = self.settled_positions.get(ticker)
                if existing_settlement:
                    # Compare settled_time to determine if this is newer
                    existing_time = existing_settlement.get("settled_time", "")
                    new_time = settlement_data.get("settled_time", "")
                    
                    if new_time > existing_time:
                        # Newer settlement - update
                        self.settled_positions[ticker] = settlement_data
                        stats["updated"] += 1
                        logger.debug(f"Updated settlement for {ticker} (newer settled_time)")
                    else:
                        # Older or same - keep existing
                        stats["unchanged"] += 1
                else:
                    # New settlement
                    self.settled_positions[ticker] = settlement_data
                    stats["new"] += 1
                    logger.debug(f"New settlement added: {ticker}")
                
                # Emit to event bus
                try:
                    from ..trading.event_bus import get_event_bus, EventType
                    event_bus = await get_event_bus()
                    
                    # Parse settled_time to get timestamp
                    settled_time_str = settlement_data.get("settled_time", "")
                    timestamp_ms = int(time.time() * 1000)  # Default to now if parsing fails
                    if settled_time_str:
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(settled_time_str.replace('Z', '+00:00'))
                            timestamp_ms = int(dt.timestamp() * 1000)
                        except (ValueError, AttributeError):
                            pass
                    
                    # Emit settlement event with full data in metadata
                    await event_bus.emit(
                        event_type=EventType.SETTLEMENT,
                        market_ticker=ticker,
                        sequence_number=0,
                        timestamp_ms=timestamp_ms,
                        metadata={
                            "settlement": settlement_data
                        }
                    )
                except Exception as e:
                    logger.debug(f"Event bus not available for settlement emission: {e}")
            
            logger.info(
                f"Settlement sync complete: {stats['new']} new, {stats['updated']} updated, "
                f"{stats['unchanged']} unchanged (total: {stats['total_fetched']})"
            )
            
            # Broadcast settlements update via WebSocket
            await self._broadcast_settlements_update()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error syncing settlements with Kalshi: {e}")
            return {"error": str(e)}
    
    async def _broadcast_settlements_update(self) -> None:
        """
        Broadcast settlements update via WebSocket.
        """
        if not self._websocket_manager:
            return
        
        await self._websocket_manager.broadcast_settlements_update({
            "settlements": self.settled_positions,
            "count": len(self.settled_positions),
            "timestamp": time.time()
        })
        logger.debug(f"Broadcast settlements update: {len(self.settled_positions)} settlements")
    
    def get_settled_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all settled positions.
        
        Returns:
            Dictionary of settlements keyed by ticker
        """
        return self.settled_positions.copy()
    
    async def update_position_from_websocket(self, position_data: Dict[str, Any]) -> None:
        """
        Update position from WebSocket position update message.
        
        This method is called by PositionListener when a real-time position update
        is received from Kalshi's market_positions WebSocket channel.
        
        Args:
            position_data: Position data from WebSocket, containing:
                - market_ticker: str
                - position: int (contracts)
                - position_cost: float (in cents, already converted from centi-cents)
                - realized_pnl: float (in cents, already converted from centi-cents)
                - fees_paid: float (in cents, already converted from centi-cents)
                - volume: int (trading volume)
                - raw_data: Dict (original Kalshi message)
        """
        market_ticker = position_data.get("market_ticker", "")
        if not market_ticker:
            logger.warning("Position update missing market_ticker")
            return
        
        # Get current position state for change detection
        current_position = self.positions.get(market_ticker)
        previous_state = self._previous_position_state.get(market_ticker, {})
        
        # Extract values from websocket data
        new_contracts = position_data.get("position", 0)
        new_position_cost = position_data.get("position_cost", 0.0)
        new_realized_pnl = position_data.get("realized_pnl", 0.0)
        new_fees_paid = position_data.get("fees_paid", 0.0)
        new_volume = position_data.get("volume", 0)
        
        # Extract last_updated_ts from raw_data if available, or use current time
        raw_data = position_data.get("raw_data", {})
        new_last_updated_ts = raw_data.get("last_updated_ts")
        if not new_last_updated_ts:
            # Use current timestamp as ISO string
            from datetime import datetime
            new_last_updated_ts = datetime.utcnow().isoformat() + "Z"
        
        # Track which fields changed
        changed_fields = []
        previous_values = {}
        
        # Check for changes
        if current_position:
            if current_position.contracts != new_contracts:
                changed_fields.append("position")
                previous_values["position"] = current_position.contracts
            
            if abs(current_position.cost_basis - new_position_cost) > 0.01:
                changed_fields.append("position_cost")
                previous_values["position_cost"] = current_position.cost_basis
            
            if abs(current_position.realized_pnl - new_realized_pnl) > 0.01:
                changed_fields.append("realized_pnl")
                previous_values["realized_pnl"] = current_position.realized_pnl
            
            # Check fees_paid if stored
            current_fees = getattr(current_position, 'fees_paid', 0.0)
            if abs(current_fees - new_fees_paid) > 0.01:
                changed_fields.append("fees_paid")
                previous_values["fees_paid"] = current_fees
            
            # Check volume if stored
            current_volume = getattr(current_position, 'volume', 0)
            if current_volume != new_volume:
                changed_fields.append("volume")
                previous_values["volume"] = current_volume
        else:
            # New position
            changed_fields.append("position")
            changed_fields.append("position_cost")
            changed_fields.append("realized_pnl")
            if new_fees_paid > 0:
                changed_fields.append("fees_paid")
            if new_volume > 0:
                changed_fields.append("volume")
        
        # Detect settlement: position goes to 0
        was_settled = False
        if current_position and current_position.contracts != 0 and new_contracts == 0:
            was_settled = True
            # Clear closing reason when position goes flat
            if market_ticker in self._active_closing_reasons:
                del self._active_closing_reasons[market_ticker]
            logger.info(
                f"🎯 Position settled: {market_ticker} - "
                f"Final P&L: ${new_realized_pnl:.2f}"
            )
        
        # Update or create position
        if current_position:
            # Update existing position
            prev_contracts = current_position.contracts
            current_position.contracts = new_contracts
            current_position.cost_basis = new_position_cost  # WebSocket provides accurate position_cost
            current_position.realized_pnl = new_realized_pnl
            current_position.kalshi_data = position_data.get("raw_data", {})
            current_position.last_updated_ts = new_last_updated_ts
            
            # Track opened_at: set when position transitions from flat to non-flat
            if prev_contracts == 0 and new_contracts != 0:
                current_position.opened_at = time.time()
            elif new_contracts == 0:
                # Reset when position goes flat
                current_position.opened_at = None
            
            # Store additional fields as attributes
            current_position.fees_paid = new_fees_paid
            current_position.volume = new_volume
        else:
            # Create new position
            new_position = Position(
                ticker=market_ticker,
                contracts=new_contracts,
                cost_basis=new_position_cost,  # WebSocket provides accurate position_cost
                realized_pnl=new_realized_pnl,
                kalshi_data=position_data.get("raw_data", {}),
                last_updated_ts=new_last_updated_ts
            )
            # Track opened_at: set when position is created with non-zero contracts
            if new_contracts != 0:
                new_position.opened_at = time.time()
            
            # Store additional fields as attributes
            new_position.fees_paid = new_fees_paid
            new_position.volume = new_volume
            self.positions[market_ticker] = new_position
            logger.info(f"New position created from WebSocket: {market_ticker} = {new_contracts} contracts")
        
        # Update previous state tracking
        self._previous_position_state[market_ticker] = {
            "position": new_contracts,
            "position_cost": new_position_cost,
            "realized_pnl": new_realized_pnl,
            "fees_paid": new_fees_paid,
            "volume": new_volume
        }
        
        # If position is now flat, we might want to keep it for a bit to show settlement
        # but we'll let the frontend handle the settled positions UI
        
        # Broadcast position update with change metadata
        if changed_fields:
            logger.debug(
                f"Position update: {market_ticker} - "
                f"Changed fields: {', '.join(changed_fields)}"
            )
            
            # Broadcast with change metadata for frontend animations
            await self._broadcast_position_update_with_changes(
                market_ticker,
                changed_fields,
                previous_values,
                was_settled
            )
        
        # Notify state change
        await self._notify_state_change()
    
    async def _broadcast_position_update_with_changes(
        self,
        market_ticker: str,
        changed_fields: List[str],
        previous_values: Dict[str, Any],
        was_settled: bool
    ) -> None:
        """
        Broadcast position update with change metadata for frontend animations.
        
        Args:
            market_ticker: Market ticker
            changed_fields: List of field names that changed
            previous_values: Previous values for changed fields
            was_settled: Whether this position was just settled
        """
        if not self._websocket_manager:
            return
        
        position = self.positions.get(market_ticker)
        if not position:
            return
        
        # Extract market exposure - API provides both cents and dollars
        # Use cents value directly from API - no conversion needed
        kalshi_data = position.kalshi_data or {}
        market_exposure_cents = kalshi_data.get("market_exposure")  # Already in cents from API
        market_exposure_dollars = kalshi_data.get("market_exposure_dollars")  # String dollars from API
        
        # Convert to float if needed
        if market_exposure_cents is not None:
            market_exposure_cents = float(market_exposure_cents) if isinstance(market_exposure_cents, str) else market_exposure_cents
        
        # Prepare position data with change metadata
        # All monetary values are in cents for consistency
        position_update_data = {
            "ticker": market_ticker,
            "position": position.contracts,
            "position_cost": position.cost_basis,  # In cents
            "cost_basis": position.cost_basis,  # In cents (alias)
            "realized_pnl": position.realized_pnl,  # In cents
            "fees_paid": getattr(position, 'fees_paid', 0.0),  # In cents
            "volume": getattr(position, 'volume', 0),
            "market_exposure_cents": market_exposure_cents,  # In cents (preferred)
            "market_exposure": market_exposure_dollars,  # In dollars (for backward compatibility)
            # Change metadata for animations
            "changed_fields": changed_fields,
            "previous_values": previous_values,
            "update_source": "websocket",
            "timestamp": time.time(),
            "was_settled": was_settled
        }
        
        # Include closing reason if position is being closed
        if market_ticker in self._active_closing_reasons:
            position_update_data["closing_reason"] = self._active_closing_reasons[market_ticker]
        
        # Broadcast via WebSocket manager
        await self._websocket_manager.broadcast_position_update(position_update_data)
    
    async def _monitor_position_health(self) -> List[tuple]:
        """
        Monitor position health and determine which positions should be closed.
        
        Returns:
            List of (ticker, reason) tuples for positions that should be closed
        """
        positions_to_close = []
        total_positions = len([p for p in self.positions.values() if not p.is_flat])
        
        logger.info(f"Monitoring position health for {total_positions} non-flat positions")
        
        for ticker, position in self.positions.items():
            if position.is_flat:
                logger.debug(f"Skipping flat position: {ticker}")
                continue
            
            logger.debug(f"Checking position health for {ticker}: {position.contracts} contracts, cost_basis=${position.cost_basis/100:.2f}")
            
            # Get current market price from orderbook
            orderbook_snapshot = await self._get_orderbook_snapshot(ticker)
            if not orderbook_snapshot:
                logger.warning(f"Could not get orderbook snapshot for {ticker}, skipping health check")
                continue
            
            # Calculate current YES price (mid price)
            yes_bid = orderbook_snapshot.get("yes_bid", 0)
            yes_ask = orderbook_snapshot.get("yes_ask", 0)
            if yes_bid > 0 and yes_ask > 0:
                current_yes_price = (yes_bid + yes_ask) / 2.0 / 100.0  # Convert cents to 0.0-1.0
                logger.debug(f"Orderbook snapshot for {ticker}: yes_bid={yes_bid}, yes_ask={yes_ask}, mid_price={current_yes_price:.4f}")
            else:
                logger.warning(f"Invalid orderbook data for {ticker} (yes_bid={yes_bid}, yes_ask={yes_ask}), skipping health check")
                continue
            
            # Calculate unrealized P&L
            unrealized_pnl = position.get_unrealized_pnl(current_yes_price)
            
            # Calculate P&L as percentage of cost basis
            if position.cost_basis > 0:
                pnl_percentage = unrealized_pnl / position.cost_basis
            else:
                pnl_percentage = 0.0
                logger.warning(f"Position {ticker} has zero cost basis, cannot calculate P&L percentage")
            
            logger.debug(
                f"Position {ticker} P&L: unrealized=${unrealized_pnl/100:.2f}, "
                f"percentage={pnl_percentage:.2%}, "
                f"cost_basis=${position.cost_basis/100:.2f}"
            )
            
            # Check take profit threshold
            if pnl_percentage >= config.RL_POSITION_TAKE_PROFIT_THRESHOLD:
                positions_to_close.append((ticker, "take_profit"))
                logger.info(
                    f"Position {ticker} hit take profit: {pnl_percentage:.2%} "
                    f"(threshold: {config.RL_POSITION_TAKE_PROFIT_THRESHOLD:.2%})"
                )
                continue
            
            # Check stop loss threshold
            if pnl_percentage <= config.RL_POSITION_STOP_LOSS_THRESHOLD:
                positions_to_close.append((ticker, "stop_loss"))
                logger.info(
                    f"Position {ticker} hit stop loss: {pnl_percentage:.2%} "
                    f"(threshold: {config.RL_POSITION_STOP_LOSS_THRESHOLD:.2%})"
                )
                continue
            
            # Check max hold time
            if position.opened_at:
                time_in_position = time.time() - position.opened_at
                logger.debug(f"Position {ticker} time in position: {time_in_position:.0f}s (threshold: {config.RL_POSITION_MAX_HOLD_TIME_SECONDS}s)")
                if time_in_position >= config.RL_POSITION_MAX_HOLD_TIME_SECONDS:
                    positions_to_close.append((ticker, "max_hold_time"))
                    logger.info(
                        f"Position {ticker} exceeded max hold time: {time_in_position:.0f}s "
                        f"(threshold: {config.RL_POSITION_MAX_HOLD_TIME_SECONDS}s)"
                    )
                    continue
            else:
                logger.debug(f"Position {ticker} has no opened_at timestamp, skipping max hold time check")
        
        logger.info(f"Position health check complete: {len(positions_to_close)} positions need closing out of {total_positions} checked")
        return positions_to_close
    
    async def _recover_cash_by_closing_positions(self) -> None:
        """
        Close worst-performing positions to recover cash when balance is low.
        """
        if self.cash_balance >= self.min_cash_reserve:
            return
        
        logger.warning(
            f"Cash balance ${self.cash_balance:.2f} below reserve ${self.min_cash_reserve:.2f}. "
            f"Closing positions to recover cash."
        )
        
        # Get all positions with their P&L
        position_pnl = []
        for ticker, position in self.positions.items():
            if position.is_flat:
                continue
            
            # Get current market price
            orderbook_snapshot = await self._get_orderbook_snapshot(ticker)
            if not orderbook_snapshot:
                continue
            
            yes_bid = orderbook_snapshot.get("yes_bid", 0)
            yes_ask = orderbook_snapshot.get("yes_ask", 0)
            if yes_bid > 0 and yes_ask > 0:
                current_yes_price = (yes_bid + yes_ask) / 2.0 / 100.0
                unrealized_pnl = position.get_unrealized_pnl(current_yes_price)
                position_pnl.append((ticker, position, unrealized_pnl))
        
        # Sort by worst P&L first (close losers first)
        position_pnl.sort(key=lambda x: x[2])  # Sort by P&L (ascending, worst first)
        
        # Close positions until we have enough cash
        target_cash = self.min_cash_reserve + 100.0  # Add buffer
        cash_recovered = 0.0
        
        for ticker, position, pnl in position_pnl:
            if self.cash_balance + cash_recovered >= target_cash:
                break
            
            # Estimate cash recovery from closing this position
            # Rough estimate: cost_basis (we'll recoup most of it)
            estimated_recovery = position.cost_basis / 100.0  # Convert cents to dollars
            
            # Close position
            result = await self.close_position(ticker, "cash_recovery")
            if result and result.get("executed"):
                cash_recovered += estimated_recovery
                logger.info(
                    f"Closed position {ticker} for cash recovery. "
                    f"Estimated recovery: ${estimated_recovery:.2f}"
                )
        
        if cash_recovered > 0:
            logger.info(f"Cash recovery complete. Estimated recovery: ${cash_recovered:.2f}")
        else:
            logger.warning("Could not recover sufficient cash by closing positions")
    
    async def _monitor_market_states(self) -> None:
        """
        Monitor market states and close positions in markets that are closing soon.
        """
        from datetime import datetime
        
        for ticker, position in self.positions.items():
            if position.is_flat:
                continue
            
            # Get market info
            market_info = await self._get_market_info(ticker)
            if not market_info:
                continue
            
            # Check market status
            market_status = market_info.get("status", "").lower()
            if market_status in ["closed", "ending"]:
                # Market is closing/closed, close position
                result = await self.close_position(ticker, "market_closing")
                if result and result.get("executed"):
                    logger.info(f"Closed position {ticker} due to market closing (status: {market_status})")
                continue
            
            # Check market end time if available
            end_time_str = market_info.get("end_time")
            if end_time_str:
                try:
                    # Parse ISO timestamp
                    end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
                    time_until_close = (end_time.timestamp() - time.time())
                    
                    if time_until_close <= config.RL_MARKET_CLOSING_BUFFER_SECONDS:
                        # Market closing soon, close position
                        result = await self.close_position(ticker, "market_closing")
                        if result and result.get("executed"):
                            logger.info(
                                f"Closed position {ticker} due to market closing soon "
                                f"({time_until_close:.0f}s until close)"
                            )
                except Exception as e:
                    logger.debug(f"Could not parse end_time for {ticker}: {e}")
    
    async def _monitor_and_close_positions(self) -> str:
        """
        Monitor position health and close positions that meet criteria.
        
        Returns:
            Result summary string (e.g., "closed 2 positions (take_profit: 1, stop_loss: 1)" or "no positions to close")
        """
        closing_result, _ = await self._monitor_and_close_positions_detailed()
        return closing_result
    
    async def _monitor_and_close_positions_detailed(self) -> tuple[str, List[str]]:
        """
        Monitor position health and close positions with detailed breakdown.
        
        Returns:
            Tuple of (result_summary, details_list)
        """
        logger.info("Starting position health monitoring...")
        
        total_positions = len([p for p in self.positions.values() if not p.is_flat])
        details = []
        
        if total_positions == 0:
            logger.info("No positions to close")
            return "no positions to close", []
        
        details.append(f"{total_positions} active positions")
        
        # Check position health
        positions_to_close = await self._monitor_position_health()
        
        logger.info(f"Position health check complete: {len(positions_to_close)} positions need closing")
        
        if not positions_to_close:
            logger.info("No positions need closing")
            return f"no positions to close ({total_positions} active)", details
        
        # Group positions by reason
        by_reason = {}
        for ticker, reason in positions_to_close:
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(ticker)
        
        # Track closing results
        closing_results = {}
        closed_count = 0
        failed_count = 0
        total_pnl = 0.0
        
        # Close positions grouped by reason
        for reason, tickers in by_reason.items():
            reason_count = len(tickers)
            reason_pnl = 0.0
            
            for ticker in tickers:
                position = self.positions.get(ticker)
                if not position:
                    continue
                
                # Calculate P&L before closing
                orderbook_snapshot = await self._get_orderbook_snapshot(ticker)
                if orderbook_snapshot:
                    yes_bid = orderbook_snapshot.get("yes_bid", 0)
                    yes_ask = orderbook_snapshot.get("yes_ask", 0)
                    if yes_bid > 0 and yes_ask > 0:
                        current_yes_price = (yes_bid + yes_ask) / 2.0 / 100.0
                        unrealized_pnl = position.get_unrealized_pnl(current_yes_price)
                        reason_pnl += unrealized_pnl
                
                logger.info(f"Attempting to close position {ticker} (reason: {reason})")
                result = await self.close_position(ticker, reason)
                if result and result.get("executed"):
                    closed_count += 1
                    if reason not in closing_results:
                        closing_results[reason] = 0
                    closing_results[reason] += 1
                    logger.info(f"Successfully closed position {ticker} ({reason})")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to close position {ticker} ({reason}): {result}")
            
            # Add reason-specific detail
            reason_label = {
                "take_profit": "above profit threshold",
                "stop_loss": "exceed loss threshold",
                "max_hold_time": "older than 24 hours",
                "cash_recovery": "for cash recovery",
                "market_closing": "in closing markets"
            }.get(reason, reason)
            
            pnl_sign = "+" if reason_pnl >= 0 else ""
            details.append(f"{reason_count} {reason_label}: closed {reason_count} -> {pnl_sign}${reason_pnl/100:.2f} P&L")
            total_pnl += reason_pnl
        
        # Build result summary
        positions_after = len([p for p in self.positions.values() if not p.is_flat])
        if closed_count > 0:
            pnl_sign = "+" if total_pnl >= 0 else ""
            result_summary = f"closed {closed_count} ({total_positions} -> {positions_after}) {pnl_sign}${total_pnl/100:.2f} P&L"
        else:
            result_summary = f"failed to close {failed_count} position{'s' if failed_count != 1 else ''}"
        
        logger.info(f"Position closing complete: {result_summary}")
        return result_summary, details
    
    async def _sync_state_with_kalshi(self) -> Dict[str, Any]:
        """
        Shared method to sync all trader state with Kalshi API.
        
        This is the single source of truth for syncing:
        - Balance and portfolio value
        - Positions
        - Settlements
        - Orders
        
        Used by both initialization and calibration.
        
        Returns:
            Dictionary with sync results including:
            - cash_balance: Cash balance from Kalshi (in dollars)
            - portfolio_value: Portfolio value from Kalshi (in dollars)
            - positions_value: Sum of (cost_basis + realized_pnl) for all positions (in dollars)
            - total_pnl: Sum of realized_pnl for all positions (in dollars)
            - active_positions: Count of non-flat positions
            - order_stats: Order sync statistics
            - settlement_stats: Settlement sync statistics
            - duration: Time taken for sync (in seconds)
        """
        sync_start = time.time()
        
        # Capture before state (using Kalshi values if available)
        cash_before = self.cash_balance
        portfolio_before = (
            self._portfolio_value_from_kalshi 
            if hasattr(self, '_portfolio_value_from_kalshi') and self._portfolio_value_from_kalshi is not None
            else None
        )
        positions_before = len([p for p in self.positions.values() if not p.is_flat])
        
        # Sync balance and portfolio value from Kalshi
        try:
            account_info = await self.trading_client.get_account_info()
            
            # Calculate drift before updating synced values
            calculated_cash = self._calculate_cash_balance()
            calculated_portfolio = self._calculate_portfolio_value()
            
            if "balance" in account_info:
                old_balance = self.cash_balance
                new_balance = account_info["balance"] / 100.0  # Convert cents to dollars
                
                # Calculate drift (how far off our calculation was)
                if self._last_sync_drift_cash is None:
                    # First sync: initialize calculated to match synced
                    self._calculated_cash_balance = new_balance
                    self._last_sync_drift_cash = 0.0
                else:
                    # Subsequent syncs: calculate drift
                    self._last_sync_drift_cash = calculated_cash - new_balance
                
                # Update synced value from Kalshi (authoritative)
                self.cash_balance = new_balance
                
                if abs(old_balance - self.cash_balance) > 0.01:
                    logger.debug(f"Cash balance synced: ${old_balance:.2f} → ${self.cash_balance:.2f} (calculated: ${calculated_cash:.2f}, drift: ${self._last_sync_drift_cash:.2f})")
            
            if "portfolio_value" in account_info:
                old_portfolio_value = getattr(self, '_portfolio_value_from_kalshi', None)
                new_portfolio_value = account_info["portfolio_value"] / 100.0  # Convert cents to dollars
                
                # Initialize drift tracking if first sync
                if self._last_sync_drift_portfolio is None:
                    self._last_sync_drift_portfolio = 0.0
                else:
                    # Subsequent syncs: calculate drift
                    self._last_sync_drift_portfolio = calculated_portfolio - new_portfolio_value
                
                # Update synced value from Kalshi
                self._portfolio_value_from_kalshi = new_portfolio_value
                
                if old_portfolio_value is None or abs(old_portfolio_value - self._portfolio_value_from_kalshi) > 0.01:
                    logger.debug(f"Portfolio value synced: ${old_portfolio_value or 0:.2f} → ${self._portfolio_value_from_kalshi:.2f} (calculated: ${calculated_portfolio:.2f}, drift: ${self._last_sync_drift_portfolio:.2f})")
        except Exception as e:
            logger.warning(f"Could not sync balance/portfolio: {e}")
        
        # Sync orders with Kalshi
        order_stats = await self.sync_orders_with_kalshi()
        
        # Sync positions with Kalshi
        await self._sync_positions_with_kalshi()
        
        # Sync settlements with Kalshi
        settlement_stats = await self.sync_settlements_with_kalshi()
        
        sync_duration = time.time() - sync_start
        
        # Capture after state (all from Kalshi)
        cash_after = self.cash_balance
        portfolio_after = (
            self._portfolio_value_from_kalshi 
            if hasattr(self, '_portfolio_value_from_kalshi') and self._portfolio_value_from_kalshi is not None
            else None
        )
        positions_after = len([p for p in self.positions.values() if not p.is_flat])
        
        # Calculate positions value and total P&L from position data (all in cents, convert to dollars)
        positions_value = sum(
            (p.cost_basis + p.realized_pnl) / 100.0 
            for p in self.positions.values() 
            if not p.is_flat
        )
        total_pnl = sum(
            p.realized_pnl / 100.0 
            for p in self.positions.values() 
            if not p.is_flat
        )
        
        return {
            "cash_balance": cash_after,
            "portfolio_value": portfolio_after,
            "positions_value": positions_value,
            "total_pnl": total_pnl,
            "active_positions": positions_after,
            "order_stats": order_stats,
            "settlement_stats": settlement_stats,
            "duration": sync_duration,
            "cash_before": cash_before,
            "portfolio_before": portfolio_before,
            "positions_before": positions_before,
        }
    
    async def _calibration_sync_state(self) -> Dict[str, Any]:
        """
        Stage 1 of calibration: Sync trader state with Kalshi API.
        
        Uses the shared _sync_state_with_kalshi() method and formats
        results for calibration status messages.
        
        Returns:
            Dictionary with sync results (same format as _sync_state_with_kalshi)
        """
        # Use shared sync method
        sync_results = await self._sync_state_with_kalshi()
        
        # Build detailed status message for calibration
        status_parts = []
        
        # Portfolio section
        portfolio_info = f"Portfolio: cash ${sync_results['cash_balance']:.2f}"
        if sync_results['portfolio_value'] is not None:
            portfolio_info += f" | positions ${sync_results['positions_value']:.2f} | total ${sync_results['portfolio_value']:.2f}"
        else:
            portfolio_info += f" | positions ${sync_results['positions_value']:.2f} | total N/A"
        if abs(sync_results['total_pnl']) > 0.01:  # Only show P&L if significant
            pnl_sign = "+" if sync_results['total_pnl'] >= 0 else ""
            portfolio_info += f" | P&L {pnl_sign}${sync_results['total_pnl']:.2f}"
        status_parts.append(portfolio_info)
        
        # Positions section
        status_parts.append(f"Positions: active: {sync_results['active_positions']}")
        
        # Settlements section
        if sync_results['settlement_stats'] and "total_fetched" in sync_results['settlement_stats']:
            status_parts.append(
                f"Settlements: fetched {sync_results['settlement_stats'].get('total_fetched', 0)}"
            )
        
        # Orders section
        if sync_results['order_stats'] and "found_in_kalshi" in sync_results['order_stats']:
            kalshi_orders = sync_results['order_stats'].get("found_in_kalshi", 0)
            local_orders = sync_results['order_stats'].get("found_in_memory", 0)
            status_parts.append(f"Orders: Kalshi {kalshi_orders} | Local {local_orders}")
        
        sync_result = " | ".join(status_parts)
        
        # Update trader status with detailed sync information
        await self._update_trader_status("calibrating -> syncing state", sync_result, duration=sync_results['duration'])
        
        return sync_results
    
    async def _calibration_navigate(self, sync_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stage 2 of calibration: Navigate - evaluate trader state and determine next action.
        
        Evaluates conditions based on synced state and determines what the trader
        should do next. Sets trader_state conditions (low_cash, rate_limit_exceeded, etc.)
        and returns the action to take.
        
        Args:
            sync_results: Results from _calibration_sync_state() containing synced state
            
        Returns:
            Dictionary with:
            - action: Next action to take ("recover_cash", "close_positions", "ready_to_trade")
            - conditions: Dict of trader_state conditions (e.g., {"low_cash": True})
            - reason: Human-readable reason for the action
        """
        conditions = {}
        actions = []
        reasons = []
        
        # Evaluate cash balance condition
        cash_balance = sync_results["cash_balance"]
        if cash_balance < self.min_cash_reserve:
            conditions["low_cash"] = True
            actions.append("recover_cash")
            reasons.append(f"cash ${cash_balance:.2f} < ${self.min_cash_reserve:.2f}")
        else:
            conditions["low_cash"] = False
            # Check if we were in low_cash and now recovered
            current_base_state = self._extract_base_state(self._trader_status)
            if current_base_state == "low_cash":
                reasons.append(f"cash restored: ${cash_balance:.2f} >= ${self.min_cash_reserve:.2f}")
        
        # Evaluate rate limit condition (future - placeholder)
        # TODO: Add rate limit tracking and evaluation
        conditions["rate_limit_exceeded"] = False
        
        # Evaluate position health condition
        active_positions = sync_results["active_positions"]
        if active_positions > 0:
            # Check if positions need closing (will be evaluated in action stage)
            # For now, just note that we have positions
            conditions["has_positions"] = True
        else:
            conditions["has_positions"] = False
        
        # Determine primary action based on conditions
        # Priority: recover_cash > close_positions > ready_to_trade
        if conditions.get("low_cash", False):
            primary_action = "recover_cash"
        elif conditions.get("has_positions", False):
            # Positions exist - check if they need closing (handled in action stage)
            primary_action = "ready_to_trade"
        else:
            primary_action = "ready_to_trade"
        
        # Build navigation result
        navigation_result = {
            "action": primary_action,
            "conditions": conditions,
            "reason": " | ".join(reasons) if reasons else f"State: {primary_action}",
            "cash_balance": cash_balance,
            "active_positions": active_positions,
        }
        
        # Update trader status with navigation result
        nav_reason = navigation_result["reason"]
        await self._update_trader_status(f"calibrating -> navigating", nav_reason)
        
        logger.debug(f"Navigation result: {navigation_result}")
        return navigation_result
    
    async def _start_periodic_sync(self) -> None:
        """
        Background task for periodic order synchronization and recalibration.
        
        Runs continuously, syncing orders with Kalshi and monitoring positions
        at configured intervals. This combines periodic sync with recalibration loop.
        """
        from ..config import config
        
        logger.info("Periodic sync and recalibration task started")
        
        try:
            while True:
                # Use recalibration interval if configured, otherwise use order sync interval
                interval = config.RL_RECALIBRATION_INTERVAL_SECONDS
                await asyncio.sleep(interval)
                
                try:
                    calibration_start = time.time()
                    logger.debug("Starting periodic sync and recalibration...")
                    
                    # Capture trading stats before transitioning away from trading
                    # (if we were in trading state)
                    if self._extract_base_state(self._trader_status) == "trading":
                        trades = self._trading_stats["trades"]
                        no_ops = self._trading_stats["no_ops"]
                        time_in_trading = time.time() - self._state_entry_time if self._state_entry_time else 0
                        trading_result = f"cash ${self.cash_balance:.2f} | {trades} trades | {no_ops} no-ops | {time_in_trading:.1f}s"
                        # Update status to calibrating (transition will be logged, stats reset when entering trading again)
                        await self._update_trader_status("calibrating", trading_result)
                    else:
                        # Already in calibrating or other state
                        await self._update_trader_status("calibrating", "periodic sync")
                    
                    # 1. Sync state with Kalshi (Stage 1: syncing_state)
                    sync_results = await self._calibration_sync_state()
                    
                    # 2. Navigate - evaluate conditions and determine next action (Stage 2: navigating)
                    navigation_result = await self._calibration_navigate(sync_results)
                    
                    # 3. Execute action based on navigation (Stage 3: action states)
                    action = navigation_result["action"]
                    conditions = navigation_result["conditions"]
                    
                    # Execute action based on navigation decision
                    if action == "recover_cash":
                        # Recover cash by closing worst-performing positions
                        recovery_start = time.time()
                        cash_before_recovery = self.cash_balance
                        await self._recover_cash_by_closing_positions()
                        recovery_duration = time.time() - recovery_start
                        cash_after_recovery = self.cash_balance
                        
                        recovery_result = f"cash ${cash_before_recovery:.2f} -> ${cash_after_recovery:.2f}"
                        if cash_after_recovery >= self.min_cash_reserve:
                            recovery_result += " (restored)"
                        await self._update_trader_status("calibrating -> recovering cash", recovery_result, duration=recovery_duration)
                    
                    # Always check position health and close positions that meet criteria
                    closing_start = time.time()
                    closing_result, closing_details = await self._monitor_and_close_positions_detailed()
                    closing_duration = time.time() - closing_start
                    
                    if closing_details:
                        full_closing_result = f"{closing_result} | " + " | ".join(closing_details)
                    else:
                        full_closing_result = closing_result
                    
                    await self._update_trader_status("calibrating -> closing positions", full_closing_result, duration=closing_duration)
                    
                    # Broadcast specific updates after periodic sync
                    await self._broadcast_positions_update("periodic_sync")
                    await self._broadcast_portfolio_update("periodic_sync")
                    
                    # Notify about state changes after periodic sync
                    await self._notify_state_change()
                    
                    # Final state transition based on conditions
                    # If cash is still low after all actions, transition to low_cash state
                    if self.cash_balance < self.min_cash_reserve:
                        low_cash_reason = f"cash ${self.cash_balance:.2f} < ${self.min_cash_reserve:.2f}"
                        await self._update_trader_status("low_cash", low_cash_reason)
                    else:
                        # Stats will be reset when entering trading state
                        await self._update_trader_status("trading", f"cash ${self.cash_balance:.2f}")
                    
                except Exception as e:
                    logger.error(f"Error in periodic sync and recalibration: {e}")
                    
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
        # Portfolio value: Use Kalshi API value (synced), calculate internal value separately
        if hasattr(self, '_portfolio_value_from_kalshi') and self._portfolio_value_from_kalshi is not None:
            portfolio_value = self._portfolio_value_from_kalshi  # Synced from Kalshi
        else:
            portfolio_value = self._calculate_portfolio_value()  # Fallback to calculated
        
        # Calculate internal portfolio value (real-time tracking)
        calculated_portfolio_value = self._calculate_portfolio_value()
        
        # Calculate internal cash balance (real-time tracking)
        calculated_cash_balance = self._calculate_cash_balance()
        
        # Calculate fill rate
        total_orders = self._orders_placed
        fill_rate = self._orders_filled / total_orders if total_orders > 0 else 0.0
        
        # Calculate session changes
        cash_balance_change = (
            self.cash_balance - self.session_start_cash 
            if self.session_start_cash is not None 
            else 0.0
        )
        portfolio_value_change = (
            portfolio_value - self.session_start_portfolio_value 
            if self.session_start_portfolio_value is not None 
            else 0.0
        )
        net_cashflow = self.session_cash_recouped - self.session_cash_invested
        
        return {
            # Synced values (from Kalshi API, only updated on sync)
            "portfolio_value": portfolio_value,
            "cash_balance": self.cash_balance,
            # Calculated values (internal real-time tracking)
            "calculated_portfolio_value": calculated_portfolio_value,
            "calculated_cash_balance": calculated_cash_balance,
            # Drift tracking (how far off calculated was from synced at last sync)
            "last_sync_drift_portfolio": self._last_sync_drift_portfolio,
            "last_sync_drift_cash": self._last_sync_drift_cash,
            # Other fields
            "promised_cash": self.promised_cash,
            # Session tracking
            "session_start_cash": self.session_start_cash if self.session_start_cash is not None else self.cash_balance,
            "session_start_portfolio_value": self.session_start_portfolio_value if self.session_start_portfolio_value is not None else portfolio_value,
            "cash_balance_change": cash_balance_change,
            "portfolio_value_change": portfolio_value_change,
            # Cashflow tracking
            "session_cash_invested": self.session_cash_invested,
            "session_cash_recouped": self.session_cash_recouped,
            "net_cashflow": net_cashflow,
            "session_total_fees_paid": self.session_total_fees_paid,
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
                    "position": position.contracts,
                    "side": "YES" if position.contracts > 0 else "NO",
                    "contracts": abs(position.contracts),
                    "cost_basis": position.cost_basis,  # How much we paid (in cents)
                    "realized_pnl": position.realized_pnl,  # Realized P&L (in cents)
                    "fees_paid": getattr(position, 'fees_paid', 0.0),  # Fees paid (in cents)
                    "volume": getattr(position, 'volume', 0),  # Trading volume
                    "last_updated_ts": position.last_updated_ts,  # ISO timestamp
                    **position.kalshi_data  # Pass through complete Kalshi API response (includes market_exposure_dollars, etc.)
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
    
    def is_healthy(self) -> bool:
        """
        Check if order manager is healthy.
        
        Returns:
            True if trading client is connected and fill processor is running
        """
        if not self.trading_client:
            return False
        
        # Check if trading client is connected
        # Assuming trading_client has a connected property or method
        if hasattr(self.trading_client, 'connected'):
            if not self.trading_client.connected:
                return False
        
        # Check if fill processor task is running
        if self._fill_processor_task and self._fill_processor_task.done():
            return False
        
        return True
    
    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information for initialization tracker.
        
        Returns:
            Dictionary with health status, connection info, and state details
        """
        from ..config import config
        return {
            "trading_client_initialized": self.trading_client is not None,
            "trading_client_connected": (
                self.trading_client.connected if self.trading_client and hasattr(self.trading_client, 'connected') else False
            ),
            "api_url": config.KALSHI_API_URL,
            "fill_listener_active": self._fill_listener is not None,
            "fill_processor_running": (
                self._fill_processor_task is not None and not self._fill_processor_task.done()
            ),
            "cash_balance": self.cash_balance,
            "open_orders_count": len(self.open_orders),
            "positions_count": len(self.positions),
            "last_sync_time": getattr(self, '_last_sync_time', None),
        }
    
    def get_last_sync_time(self) -> Optional[float]:
        """
        Get last sync time with Kalshi.
        
        Returns:
            Timestamp of last sync, or None if never synced
        """
        return getattr(self, '_last_sync_time', None)