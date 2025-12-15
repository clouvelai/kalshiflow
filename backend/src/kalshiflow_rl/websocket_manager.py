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
        self._stats_task: Optional[asyncio.Task] = None
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
        
        logger.info(f"WebSocketManager initialized for {len(self._market_tickers)} markets")
    
    async def start(self):
        """Start the WebSocket manager and subscribe to orderbook updates."""
        if self._running:
            logger.warning("WebSocketManager already running")
            return
        
        self._running = True
        logger.info("Starting WebSocketManager...")
        
        # Get shared orderbook states for all markets
        self._orderbook_states = await get_all_orderbook_states()
        
        # Subscribe to orderbook updates for each market
        for market_ticker in self._market_tickers:
            if market_ticker in self._orderbook_states:
                state = self._orderbook_states[market_ticker]
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
        
        # Start statistics broadcast task (1-second interval)
        self._stats_task = asyncio.create_task(self._stats_broadcast_loop())
        
        logger.info("WebSocketManager started successfully")
    
    async def stop(self):
        """Stop the WebSocket manager and close all connections."""
        if not self._running:
            return
        
        logger.info("Stopping WebSocketManager...")
        self._running = False
        
        # Stop statistics task
        if self._stats_task:
            self._stats_task.cancel()
            try:
                await self._stats_task
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
            # Send connection message with market list
            connection_msg = ConnectionMessage(
                data={
                    "markets": self._market_tickers,
                    "status": "connected", 
                    "version": "1.0.0"
                }
            )
            await self._send_to_client(websocket, connection_msg)
            
            # Send initial snapshots for all markets
            for market_ticker in self._market_tickers:
                if market_ticker in self._orderbook_states:
                    state = self._orderbook_states[market_ticker]
                    snapshot = await state.get_snapshot()
                    if snapshot:
                        snapshot_msg = OrderbookSnapshotMessage(
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
                        await self._send_to_client(websocket, snapshot_msg)
            
            # Send initial trader state if OrderManager is available
            if self._order_manager:
                try:
                    initial_state = await self._order_manager.get_current_state()
                    state_msg = TraderStateMessage(data=initial_state)
                    await self._send_to_client(websocket, state_msg)
                    logger.info("Sent initial trader state to new client")
                except Exception as e:
                    logger.error(f"Failed to send initial trader state: {e}")
            else:
                logger.warning("OrderManager not available - skipping initial trader state broadcast")
            
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
        Broadcast statistics to all connected clients.
        
        Args:
            stats: Statistics data
        """
        if not self._connections:
            return
        
        message = StatsMessage(data=stats)
        await self._broadcast_to_all(message)
        self._stats_broadcast += 1
    
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
            msg_dict = asdict(message) if hasattr(message, '__dataclass_fields__') else message
            await websocket.send_json(msg_dict)
            self._messages_sent += 1
        except Exception as e:
            logger.error(f"Error sending to client: {e}")
            self._connections.discard(websocket)
    
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
    
    async def _stats_broadcast_loop(self):
        """Broadcast statistics every second."""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                
                # Gather statistics
                stats = await self._gather_stats()
                
                # Broadcast to all clients
                await self.broadcast_stats(stats)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in stats broadcast loop: {e}")
    
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