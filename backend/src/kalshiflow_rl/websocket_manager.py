"""
WebSocket manager for broadcasting orderbook updates to frontend clients.

Manages WebSocket connections from frontend, broadcasts orderbook snapshots,
deltas, and statistics in real-time. Non-blocking from database operations.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

from starlette.websockets import WebSocket, WebSocketState
from websockets.exceptions import ConnectionClosed

from .data.orderbook_state import SharedOrderbookState, get_all_orderbook_states
from .config import config

logger = logging.getLogger("kalshiflow_rl.websocket_manager")


@dataclass
class ConnectionMessage:
    """Initial connection message sent to clients."""
    type: str = "connection"
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {
                "markets": [],
                "status": "connected",
                "version": "1.0.0"
            }


@dataclass
class OrderbookSnapshotMessage:
    """Orderbook snapshot message format."""
    type: str = "orderbook_snapshot"
    data: Dict[str, Any] = None


@dataclass
class OrderbookDeltaMessage:
    """Orderbook delta message format."""
    type: str = "orderbook_delta"
    data: Dict[str, Any] = None


@dataclass
class StatsMessage:
    """Statistics message format."""
    type: str = "stats"
    data: Dict[str, Any] = None


@dataclass
class TraderStateMessage:
    """Trader state update message."""
    type: str = "trader_state"
    data: Dict[str, Any] = None


@dataclass
class TraderActionMessage:
    """Trader action decision message."""
    type: str = "trader_action"
    data: Dict[str, Any] = None


@dataclass
class OrdersUpdateMessage:
    """Orders update message."""
    type: str = "orders_update"
    data: Dict[str, Any] = None


@dataclass
class PositionsUpdateMessage:
    """Positions update message."""
    type: str = "positions_update"
    data: Dict[str, Any] = None


@dataclass
class PositionUpdateMessage:
    """Individual position update message with change metadata."""
    type: str = "position_update"
    data: Dict[str, Any] = None


@dataclass
class SettlementsUpdateMessage:
    """Settlements update message."""
    type: str = "settlements_update"
    data: Dict[str, Any] = None


@dataclass
class PortfolioUpdateMessage:
    """Portfolio/Balance update message."""
    type: str = "portfolio_update"
    data: Dict[str, Any] = None


@dataclass
class FillEventMessage:
    """Fill event notification message."""
    type: str = "fill_event"
    data: Dict[str, Any] = None


@dataclass
class StatsUpdateMessage:
    """Stats update message (full stats, sent every 60s)."""
    type: str = "stats_update"
    data: Dict[str, Any] = None


@dataclass
class StatsSummaryMessage:
    """Lightweight stats summary message (sent every 1s)."""
    type: str = "stats_summary"
    data: Dict[str, Any] = None


@dataclass
class InitializationStartMessage:
    """Initialization sequence started message."""
    type: str = "initialization_start"
    data: Dict[str, Any] = None


@dataclass
class InitializationStepMessage:
    """Initialization step progress message."""
    type: str = "initialization_step"
    data: Dict[str, Any] = None


@dataclass
class InitializationCompleteMessage:
    """Initialization sequence completed message."""
    type: str = "initialization_complete"
    data: Dict[str, Any] = None


@dataclass
class ComponentHealthMessage:
    """Component health status update message."""
    type: str = "component_health"
    data: Dict[str, Any] = None


@dataclass
class TraderStatusMessage:
    """Trader status update message."""
    type: str = "trader_status"
    data: Dict[str, Any] = None


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts orderbook updates.
    
    Features:
    - Manages multiple concurrent WebSocket connections
    - Broadcasts orderbook snapshots and deltas to all clients
    - Sends statistics updates every second
    - Non-blocking broadcasts (doesn't wait for database operations)
    - Graceful handling of client disconnections
    """
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self._connections: Set[WebSocket] = set()
        self._running = False
        self._summary_stats_task: Optional[asyncio.Task] = None
        self._full_stats_task: Optional[asyncio.Task] = None
        self._orderbook_states: Dict[str, SharedOrderbookState] = {}
        self._market_tickers = config.RL_MARKET_TICKERS
        
        # Statistics tracking
        self._connection_count = 0
        self._messages_sent = 0
        self._snapshots_broadcast = 0
        self._deltas_broadcast = 0
        self._stats_broadcast = 0
        
        # For stats collector integration
        self.stats_collector = None
        
        # For trader state broadcasting
        self._order_manager = None
        self._actor_service = None
        
        # For initialization status (sent to new connections)
        self._initialization_status = None
        
        logger.info(f"WebSocketManager initialized for {len(self._market_tickers)} markets")
    
    def set_market_tickers(self, market_tickers: list):
        """
        Update the list of market tickers to monitor.
        
        Args:
            market_tickers: List of market tickers to monitor
        """
        self._market_tickers = market_tickers
        logger.info(f"Updated WebSocketManager to monitor {len(market_tickers)} markets")
    
    async def start_early(self):
        """
        Start WebSocket manager early for connection handling and initialization broadcasts.
        
        This allows the manager to accept connections and broadcast initialization
        updates before orderbook states are available. Call subscribe_to_orderbook_states()
        later to finish the setup once orderbook states are ready.
        """
        if self._running:
            logger.warning("WebSocketManager already running")
            return
        
        self._running = True
        logger.info("Starting WebSocketManager early (connection handling only)...")
        
        # Start statistics broadcast tasks (can run without orderbook states)
        # Summary stats every 1 second (lightweight)
        self._summary_stats_task = asyncio.create_task(self._summary_stats_loop())
        # Full stats every 60 seconds (includes large payloads)
        self._full_stats_task = asyncio.create_task(self._full_stats_loop())
        
        logger.info("WebSocketManager started early - ready for connections and initialization broadcasts")
    
    async def _subscribe_to_orderbook_states(self):
        """
        Internal method to subscribe to orderbook updates for all markets.
        
        Gets orderbook states and subscribes to updates. Can be called multiple times
        safely - will only subscribe to states that haven't been subscribed to yet.
        """
        # Get shared orderbook states for all markets
        self._orderbook_states = await get_all_orderbook_states()
        
        # Subscribe to orderbook updates for each market
        for market_ticker in self._market_tickers:
            if market_ticker in self._orderbook_states:
                state = self._orderbook_states[market_ticker]
                # Check if already subscribed to avoid duplicate subscriptions
                # We'll check by seeing if state has subscribers (simple check)
                # Note: add_subscriber is idempotent in practice, but we log anyway
                # Subscribe to snapshot and delta updates
                # Create a wrapper function that includes the market ticker
                def create_callback(ticker):
                    def callback(notification_data):
                        asyncio.create_task(
                            self._on_orderbook_notification(notification_data, ticker)
                        )
                    return callback
                
                state.add_subscriber(create_callback(market_ticker))
                logger.info(f"Subscribed to orderbook updates for: {market_ticker}")
    
    async def subscribe_to_orderbook_states(self):
        """
        Subscribe to orderbook updates for all markets.
        
        Call this after orderbook states have been created (e.g., after OrderbookClient starts).
        This method can be called multiple times safely.
        """
        if not self._running:
            logger.warning("WebSocketManager not running - call start_early() first")
            return
        
        logger.info("Subscribing to orderbook states...")
        await self._subscribe_to_orderbook_states()
        logger.info("Orderbook state subscriptions complete")
    
    async def start(self):
        """
        Start the WebSocket manager and subscribe to orderbook updates.
        
        This is a convenience method that calls start_early() then subscribes to orderbook states.
        Maintains backward compatibility with existing code.
        """
        if self._running:
            logger.warning("WebSocketManager already running")
            return
        
        # Start early (connection handling)
        await self.start_early()
        
        # Subscribe to orderbook states
        await self._subscribe_to_orderbook_states()
        
        logger.info("WebSocketManager started successfully")
    
    async def stop(self):
        """Stop the WebSocket manager and close all connections."""
        if not self._running:
            return
        
        logger.info("Stopping WebSocketManager...")
        self._running = False
        
        # Stop statistics tasks
        if self._summary_stats_task:
            self._summary_stats_task.cancel()
            try:
                await self._summary_stats_task
            except asyncio.CancelledError:
                pass
        
        if self._full_stats_task:
            self._full_stats_task.cancel()
            try:
                await self._full_stats_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for websocket in list(self._connections):
            try:
                await websocket.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        self._connections.clear()
        logger.info("WebSocketManager stopped")
    
    async def handle_connection(self, websocket: WebSocket):
        """
        Handle a new WebSocket connection from frontend.
        
        Args:
            websocket: Starlette WebSocket instance
        """
        await websocket.accept()
        self._connections.add(websocket)
        self._connection_count += 1
        
        logger.info(f"New WebSocket connection accepted. Total connections: {len(self._connections)}")
        
        try:
            # Send connection message with market list and API configuration
            connection_msg = ConnectionMessage(
                data={
                    "markets": self._market_tickers,
                    "status": "connected", 
                    "version": "1.0.0",
                    "kalshi_api_url": config.KALSHI_API_URL,
                    "kalshi_ws_url": config.KALSHI_WS_URL,
                    "environment": config.ENVIRONMENT
                }
            )
            await self._send_to_client(websocket, connection_msg)
            
            # Send initial snapshots for all markets (force send even if empty)
            snapshots_sent = 0
            for market_ticker in self._market_tickers:
                # Check connection state before each send
                if websocket.client_state != WebSocketState.CONNECTED:
                    logger.warning(f"WebSocket disconnected while sending initial snapshots (sent {snapshots_sent}/{len(self._market_tickers)})")
                    break
                    
                if market_ticker in self._orderbook_states:
                    state = self._orderbook_states[market_ticker]
                    snapshot = await state.get_snapshot()
                    
                    # Always send a snapshot message, even if snapshot is None or empty
                    snapshot_msg = OrderbookSnapshotMessage(
                        data={
                            "market_ticker": market_ticker,
                            "timestamp_ms": snapshot.get("timestamp_ms", 0) if snapshot else int(time.time() * 1000),
                            "sequence_number": snapshot.get("sequence_number", 0) if snapshot else 0,
                            "yes_bids": snapshot.get("yes_bids", {}) if snapshot else {},
                            "yes_asks": snapshot.get("yes_asks", {}) if snapshot else {},
                            "no_bids": snapshot.get("no_bids", {}) if snapshot else {},
                            "no_asks": snapshot.get("no_asks", {}) if snapshot else {},
                            "yes_mid_price": snapshot.get("yes_mid_price") if snapshot else None,
                            "no_mid_price": snapshot.get("no_mid_price") if snapshot else None,
                            "is_empty": snapshot is None or not any(snapshot.get(k) for k in ["yes_bids", "yes_asks", "no_bids", "no_asks"] if snapshot)
                        }
                    )
                    success = await self._send_to_client(websocket, snapshot_msg)
                    if success:
                        snapshots_sent += 1
                    else:
                        logger.warning(f"Failed to send snapshot for {market_ticker}, stopping initial snapshot send")
                        break
                else:
                    # Send empty snapshot for markets not yet initialized
                    snapshot_msg = OrderbookSnapshotMessage(
                        data={
                            "market_ticker": market_ticker,
                            "timestamp_ms": int(time.time() * 1000),
                            "sequence_number": 0,
                            "yes_bids": {},
                            "yes_asks": {},
                            "no_bids": {},
                            "no_asks": {},
                            "yes_mid_price": None,
                            "no_mid_price": None,
                            "is_empty": True,
                            "state_missing": True
                        }
                    )
                    success = await self._send_to_client(websocket, snapshot_msg)
                    if success:
                        snapshots_sent += 1
                    else:
                        logger.warning(f"Failed to send empty snapshot for {market_ticker}, stopping initial snapshot send")
                        break
            
            logger.info(f"Sent {snapshots_sent} initial snapshots to new WebSocket client")
            
            # Send initial trader state (always send, even if empty)
            if self._order_manager:
                try:
                    initial_state = await self._order_manager.get_current_state()
                    
                    # Include actor metrics if available
                    if self._actor_service:
                        try:
                            actor_metrics = self._actor_service.get_metrics()
                            initial_state["actor_metrics"] = actor_metrics
                        except Exception as e:
                            logger.warning(f"Could not get actor metrics for initial state: {e}")
                    
                    state_msg = TraderStateMessage(data=initial_state)
                    await self._send_to_client(websocket, state_msg)
                    logger.info("Sent initial trader state to new client")
                    
                    # Also send trader status separately for the status widget
                    if "trader_status" in initial_state:
                        status_data = initial_state["trader_status"]
                        status_msg = TraderStatusMessage(
                            data={
                                "current_status": status_data.get("current_status", "unknown"),
                                "status_history": status_data.get("status_history", []),
                                "timestamp": time.time()
                            }
                        )
                        await self._send_to_client(websocket, status_msg)
                        logger.info("Sent initial trader status to new client")
                except Exception as e:
                    logger.error(f"Failed to send initial trader state: {e}")
                    # Send empty trader state on error
                    empty_state = {
                        "positions": {},
                        "open_orders": {},
                        "cash_balance": 0.0,
                        "portfolio_value": 0.0,
                        "recent_actions": [],
                        "status": "error",
                        "error": str(e)
                    }
                    state_msg = TraderStateMessage(data=empty_state)
                    await self._send_to_client(websocket, state_msg)
                    logger.info("Sent error trader state to new client")
            else:
                # Always send initial trader state, even if OrderManager not available
                empty_state = {
                    "positions": {},
                    "open_orders": {},
                    "cash_balance": 0.0,
                    "portfolio_value": 0.0,
                    "recent_actions": [],
                    "status": "waiting_for_trader",
                    "message": "Trading functionality not enabled"
                }
                state_msg = TraderStateMessage(data=empty_state)
                await self._send_to_client(websocket, state_msg)
                logger.info("Sent empty trader state to new client (OrderManager not available)")
            
            # Send current initialization status if available (for clients connecting after initialization)
            if self._initialization_status:
                try:
                    if self._initialization_status.get("completed_at"):
                        # Send complete message
                        complete_msg = InitializationCompleteMessage(data=self._initialization_status)
                        await self._send_to_client(websocket, complete_msg)
                        logger.info("Sent initialization_complete status to new client")
                    else:
                        # Send start message if initialization is in progress
                        start_msg = InitializationStartMessage(data={
                            "started_at": self._initialization_status.get("started_at")
                        })
                        await self._send_to_client(websocket, start_msg)
                        # Send all current steps
                        steps = self._initialization_status.get("steps", {})
                        for step_id, step_data in steps.items():
                            step_msg = InitializationStepMessage(data=step_data)
                            await self._send_to_client(websocket, step_msg)
                        logger.info(f"Sent initialization status to new client ({len(steps)} steps)")
                except Exception as e:
                    logger.warning(f"Failed to send initialization status to new client: {e}")
            
            # Send full stats immediately on connection (client will receive summary updates every 1s after this)
            try:
                stats = await self._gather_stats()
                full_stats_msg = StatsUpdateMessage(
                    data={
                        "stats": stats,
                        "timestamp": time.time(),
                        "source": "stats_collector"
                    }
                )
                await self._send_to_client(websocket, full_stats_msg)
                logger.info("Sent initial full stats to new client")
            except Exception as e:
                logger.warning(f"Failed to send initial full stats to new client: {e}")
            
            # Keep connection alive and handle incoming messages
            while self._running and websocket.client_state == WebSocketState.CONNECTED:
                try:
                    # Wait for client messages (ping/pong handled by Starlette)
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=60.0  # 60-second timeout for client messages
                    )
                    
                    # Handle client messages if needed (currently just echo back)
                    if message:
                        logger.debug(f"Received client message: {message}")
                        
                except asyncio.TimeoutError:
                    # No message received, send ping to check connection
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break
                        
                except ConnectionClosed:
                    break
                    
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            
        finally:
            # Clean up connection
            self._connections.discard(websocket)
            logger.info(f"WebSocket connection closed. Total connections: {len(self._connections)}")
    
    async def broadcast_snapshot(self, market_ticker: str, snapshot: Dict[str, Any]):
        """
        Broadcast orderbook snapshot to all connected clients.
        
        Args:
            market_ticker: Market ticker symbol
            snapshot: Orderbook snapshot data
        """
        if not self._connections:
            return
        
        message = OrderbookSnapshotMessage(
            data={
                "market_ticker": market_ticker,
                "timestamp_ms": snapshot.get("timestamp_ms"),
                "sequence_number": snapshot.get("sequence_number"),
                "yes_bids": snapshot.get("yes_bids", {}),
                "yes_asks": snapshot.get("yes_asks", {}),
                "no_bids": snapshot.get("no_bids", {}),
                "no_asks": snapshot.get("no_asks", {}),
                "yes_mid_price": snapshot.get("yes_mid_price"),
                "no_mid_price": snapshot.get("no_mid_price")
            }
        )
        
        await self._broadcast_to_all(message)
        self._snapshots_broadcast += 1
    
    async def broadcast_delta(self, market_ticker: str, delta: Dict[str, Any]):
        """
        Broadcast orderbook delta to all connected clients.
        
        Args:
            market_ticker: Market ticker symbol
            delta: Orderbook delta data
        """
        if not self._connections:
            return
        
        # Throttling can be implemented here if needed
        message = OrderbookDeltaMessage(
            data={
                "market_ticker": market_ticker,
                "timestamp_ms": delta.get("timestamp_ms"),
                "sequence_number": delta.get("sequence_number"),
                "side": delta.get("side"),
                "action": delta.get("action"),
                "price": delta.get("price"),
                "old_size": delta.get("old_size"),
                "new_size": delta.get("new_size")
            }
        )
        
        await self._broadcast_to_all(message)
        self._deltas_broadcast += 1
    
    async def broadcast_stats(self, stats: Dict[str, Any]):
        """
        Broadcast full statistics to all connected clients.
        
        Args:
            stats: Full statistics data (includes per_market, orderbook_client, etc.)
        """
        if not self._connections:
            return
        
        # Use new StatsUpdateMessage format with timestamp and source
        message = StatsUpdateMessage(
            data={
                "stats": stats,
                "timestamp": time.time(),
                "source": "stats_collector"
            }
        )
        await self._broadcast_to_all(message)
        self._stats_broadcast += 1
    
    async def broadcast_summary_stats(self, summary_stats: Dict[str, Any]):
        """
        Broadcast lightweight summary statistics to all connected clients.
        
        This method sends only essential stats (uptime, snapshots, deltas, etc.)
        without the large payloads (per_market, orderbook_client details, etc.).
        Used for frequent updates (every 1 second).
        
        Args:
            summary_stats: Lightweight summary statistics data
        """
        if not self._connections:
            return
        
        # Use StatsSummaryMessage for lightweight updates
        message = StatsSummaryMessage(
            data={
                "stats": summary_stats,
                "timestamp": time.time(),
                "source": "stats_collector"
            }
        )
        await self._broadcast_to_all(message)
        # Note: We don't increment _stats_broadcast here to keep it separate from full stats
    
    async def broadcast_trader_state(self, state_data: Dict[str, Any]):
        """
        Broadcast trader state to all connected clients.
        
        Args:
            state_data: Trader state data including positions, orders, and metrics
        """
        if not self._connections:
            return
        
        message = TraderStateMessage(data=state_data)
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast trader state to {len(self._connections)} clients")
    
    async def broadcast_trader_action(self, action_data: Dict[str, Any]):
        """
        Broadcast trader action to all connected clients.
        
        Args:
            action_data: Trader action data including observation and decision
        """
        if not self._connections:
            return
        
        message = TraderActionMessage(data=action_data)
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast trader action to {len(self._connections)} clients")
    
    async def broadcast_orders_update(self, orders_data: Dict[str, Any], source: str = "websocket"):
        """
        Broadcast orders update to all connected clients.
        
        Args:
            orders_data: Orders data containing orders list
            source: Source of the update ("api_sync" or "websocket")
        """
        if not self._connections:
            return
        
        message = OrdersUpdateMessage(
            data={
                "orders": orders_data.get("orders", []),
                "timestamp": time.time(),
                "source": source
            }
        )
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast orders update to {len(self._connections)} clients (source: {source})")
    
    async def broadcast_positions_update(self, positions_data: Dict[str, Any], source: str = "websocket"):
        """
        Broadcast positions update to all connected clients.
        
        Args:
            positions_data: Positions data containing positions and total value
            source: Source of the update ("api_sync" or "websocket")
        """
        if not self._connections:
            return
        
        message = PositionsUpdateMessage(
            data={
                "positions": positions_data.get("positions", {}),
                "total_value": positions_data.get("total_value", 0.0),
                "timestamp": time.time(),
                "source": source
            }
        )
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast positions update to {len(self._connections)} clients (source: {source})")
    
    async def broadcast_position_update(self, position_data: Dict[str, Any]):
        """
        Broadcast individual position update with change metadata for animations.
        
        Args:
            position_data: Position data containing:
                - ticker: Market ticker
                - position: Current position (contracts)
                - position_cost: Position cost in dollars
                - realized_pnl: Realized P&L in dollars
                - fees_paid: Fees paid in dollars
                - volume: Trading volume
                - changed_fields: List of field names that changed
                - previous_values: Previous values for changed fields
                - update_source: "websocket" or "api_sync"
                - timestamp: Update timestamp
                - was_settled: Whether position was just settled
        """
        if not self._connections:
            return
        
        message = PositionUpdateMessage(data=position_data)
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast position update for {position_data.get('ticker', 'unknown')} to {len(self._connections)} clients")
    
    async def broadcast_settlements_update(self, settlements_data: Dict[str, Any]):
        """
        Broadcast settlements update to all connected clients.
        
        Args:
            settlements_data: Settlements data containing:
                - settlements: Dict of settlements keyed by ticker
                - count: Number of settlements
                - timestamp: Update timestamp
        """
        if not self._connections:
            return
        
        message = SettlementsUpdateMessage(data=settlements_data)
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast settlements update to {len(self._connections)} clients ({settlements_data.get('count', 0)} settlements)")
    
    async def broadcast_trader_status(self, status_data: Dict[str, Any]):
        """
        Broadcast trader status update to all connected clients.
        
        Args:
            status_data: Status data containing current_status and status_history
        """
        if not self._connections:
            return
        
        message = TraderStatusMessage(
            data={
                "current_status": status_data.get("current_status", "unknown"),
                "status_history": status_data.get("status_history", []),
                "timestamp": time.time()
            }
        )
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast trader status to {len(self._connections)} clients")
    
    async def broadcast_portfolio_update(self, portfolio_data: Dict[str, Any]):
        """
        Broadcast portfolio/balance update to all connected clients.
        
        Args:
            portfolio_data: Portfolio data containing cash balance and portfolio value
        """
        if not self._connections:
            return
        
        message = PortfolioUpdateMessage(
            data={
                "cash_balance": portfolio_data.get("cash_balance", 0.0),
                "portfolio_value": portfolio_data.get("portfolio_value", 0.0),
                "timestamp": time.time()
            }
        )
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast portfolio update to {len(self._connections)} clients")
    
    async def broadcast_fill_event(self, fill_data: Dict[str, Any]):
        """
        Broadcast fill event notification to all connected clients.
        
        Args:
            fill_data: Fill data containing fill details and updated position
        """
        if not self._connections:
            return
        
        message = FillEventMessage(
            data={
                "fill": fill_data.get("fill", {}),
                "updated_position": fill_data.get("updated_position", {}),
                "timestamp": time.time()
            }
        )
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast fill event to {len(self._connections)} clients")
    
    async def broadcast_initialization_start(self, start_data: Dict[str, Any]):
        """
        Broadcast initialization sequence start to all connected clients.
        
        Args:
            start_data: Initialization start data with started_at timestamp
        """
        # Store initial status
        if not self._initialization_status:
            self._initialization_status = {
                "started_at": start_data.get("started_at"),
                "steps": {},
                "completed_at": None
            }
        
        if not self._connections:
            return
        
        message = InitializationStartMessage(data=start_data)
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast initialization_start to {len(self._connections)} clients")
    
    async def broadcast_initialization_step(self, step_data: Dict[str, Any]):
        """
        Broadcast initialization step progress to all connected clients.
        
        Args:
            step_data: Step data with step_id, name, status, details
        """
        # Update stored status
        if not self._initialization_status:
            self._initialization_status = {"steps": {}}
        if "steps" not in self._initialization_status:
            self._initialization_status["steps"] = {}
        self._initialization_status["steps"][step_data.get("step_id")] = step_data
        
        if not self._connections:
            return
        
        message = InitializationStepMessage(data=step_data)
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast initialization_step: {step_data.get('step_id')} ({step_data.get('status')})")
    
    async def broadcast_initialization_complete(self, complete_data: Dict[str, Any]):
        """
        Broadcast initialization sequence completion to all connected clients.
        
        Args:
            complete_data: Completion data with steps summary, warnings, etc.
        """
        # Store the initialization status for new connections
        self._initialization_status = complete_data.copy()
        
        if not self._connections:
            return
        
        message = InitializationCompleteMessage(data=complete_data)
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast initialization_complete to {len(self._connections)} clients")
    
    async def broadcast_component_health(self, health_data: Dict[str, Any]):
        """
        Broadcast component health status update to all connected clients.
        
        Args:
            health_data: Health data with component, status, last_update, details
        """
        if not self._connections:
            return
        
        message = ComponentHealthMessage(data=health_data)
        await self._broadcast_to_all(message)
        logger.debug(f"Broadcast component_health: {health_data.get('component')} ({health_data.get('status')})")
    
    async def _broadcast_to_all(self, message: Any):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Message to broadcast (dataclass with asdict support)
        """
        if not self._connections:
            return
        
        # Convert message to dict
        msg_dict = asdict(message) if hasattr(message, '__dataclass_fields__') else message
        
        # Send to all connected clients
        disconnected = []
        for websocket in self._connections:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(msg_dict)
                    self._messages_sent += 1
                else:
                    disconnected.append(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            self._connections.discard(websocket)
    
    async def _send_to_client(self, websocket: WebSocket, message: Any):
        """
        Send message to specific client.
        
        Args:
            websocket: Target WebSocket connection
            message: Message to send (dataclass with asdict support)
        """
        try:
            # Check if WebSocket is still connected before sending
            if websocket.client_state != WebSocketState.CONNECTED:
                logger.debug("Skipping send - WebSocket not connected")
                self._connections.discard(websocket)
                return False
                
            msg_dict = asdict(message) if hasattr(message, '__dataclass_fields__') else message
            await websocket.send_json(msg_dict)
            self._messages_sent += 1
            return True
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            self._connections.discard(websocket)
            return False
    
    async def _on_orderbook_notification(self, notification_data: Dict[str, Any], market_ticker: str):
        """
        Handle orderbook notifications from SharedOrderbookState.
        
        Args:
            notification_data: Notification data from SharedOrderbookState
            market_ticker: Market ticker symbol
        """
        update_type = notification_data.get('update_type')
        
        # Get the full orderbook state for broadcasting
        if market_ticker in self._orderbook_states:
            state = self._orderbook_states[market_ticker]
            snapshot = await state.get_snapshot()
            
            if update_type == "snapshot":
                await self.broadcast_snapshot(market_ticker, snapshot)
            elif update_type == "delta":
                # For now, broadcast the full snapshot on delta updates too
                # Can be optimized later to send only the delta
                await self.broadcast_snapshot(market_ticker, snapshot)
    
    async def _summary_stats_loop(self):
        """Broadcast lightweight summary statistics every 1 second."""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                
                # Get lightweight summary stats
                if self.stats_collector:
                    summary_stats = self.stats_collector.get_summary_stats()
                    # Broadcast summary stats
                    await self.broadcast_summary_stats(summary_stats)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in summary stats broadcast loop: {e}")
    
    async def _full_stats_loop(self):
        """Broadcast full statistics every 60 seconds."""
        while self._running:
            try:
                await asyncio.sleep(60.0)
                
                # Gather full statistics
                stats = await self._gather_stats()
                
                # Broadcast full stats to all clients
                await self.broadcast_stats(stats)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in full stats broadcast loop: {e}")
    
    async def _gather_stats(self) -> Dict[str, Any]:
        """
        Gather current statistics for broadcasting.
        
        Returns:
            Statistics dictionary
        """
        # Get stats from stats collector if available
        if self.stats_collector:
            return self.stats_collector.get_stats()
        
        # Otherwise return basic stats
        return {
            "markets_active": len(self._market_tickers),
            "snapshots_processed": self._snapshots_broadcast,
            "deltas_processed": self._deltas_broadcast,
            "messages_per_second": 0,  # Will be calculated by stats collector
            "db_queue_size": 0,  # Will be provided by stats collector
            "uptime_seconds": 0,  # Will be calculated by stats collector
            "last_update_ms": int(time.time() * 1000)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current WebSocket manager statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "active_connections": len(self._connections),
            "total_connections": self._connection_count,
            "messages_sent": self._messages_sent,
            "snapshots_broadcast": self._snapshots_broadcast,
            "deltas_broadcast": self._deltas_broadcast,
            "stats_broadcast": self._stats_broadcast
        }
    
    def set_order_manager(self, order_manager):
        """
        Set reference to the OrderManager for trader state broadcasting.
        
        Args:
            order_manager: KalshiMultiMarketOrderManager instance
        """
        self._order_manager = order_manager
        logger.info("OrderManager reference configured for WebSocket broadcasting")
    
    def set_actor_service(self, actor_service):
        """
        Set reference to the ActorService for metrics broadcasting.
        
        Args:
            actor_service: ActorService instance
        """
        self._actor_service = actor_service
        logger.info("ActorService reference configured for WebSocket broadcasting")
    
    def is_healthy(self) -> bool:
        """
        Check if WebSocket manager is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return self._running


# Global WebSocket manager instance
websocket_manager = WebSocketManager()


# Export for use in app.py
__all__ = ["websocket_manager", "WebSocketManager"]