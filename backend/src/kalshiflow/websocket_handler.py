"""
WebSocket handler for real-time communication with frontend clients.

Provides the /ws/stream endpoint for frontend to receive trade updates
and broadcasts trade data to all connected clients.
"""

import json
import asyncio
import logging
from typing import Set, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.endpoints import WebSocketEndpoint

from .models import SnapshotMessage, RealtimeUpdateMessage, ChartDataMessage, AnalyticsUpdateMessage
from .trade_processor import get_trade_processor


logger = logging.getLogger(__name__)


class WebSocketBroadcaster:
    """Manages WebSocket connections and broadcasts messages to clients."""
    
    def __init__(self):
        """Initialize the broadcaster."""
        self.connections: Set[WebSocket] = set()
        self._running = False
        self._ping_interval = 30  # Send ping every 30 seconds
        self._ping_timeout = 10   # Wait 10 seconds for pong
        self._ping_tasks: Dict[WebSocket, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "active_connections": 0,
            "total_connections": 0,
            "messages_sent": 0,
            "broadcast_errors": 0,
            "pings_sent": 0,
            "pong_timeouts": 0
        }
    
    async def connect(self, websocket: WebSocket):
        """Add a new WebSocket connection."""
        await websocket.accept()
        self.connections.add(websocket)
        self.stats["active_connections"] = len(self.connections)
        self.stats["total_connections"] += 1
        
        logger.info(f"WebSocket connected. Active connections: {len(self.connections)}")
        
        # Start ping task for this connection
        ping_task = asyncio.create_task(self._ping_loop(websocket))
        self._ping_tasks[websocket] = ping_task
        
        # Send snapshot data to new client
        await self._send_snapshot(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.connections.discard(websocket)
        self.stats["active_connections"] = len(self.connections)
        
        # Cancel ping task for this connection
        if websocket in self._ping_tasks:
            ping_task = self._ping_tasks.pop(websocket)
            ping_task.cancel()
        
        logger.info(f"WebSocket disconnected. Active connections: {len(self.connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.connections:
            return
        
        # Custom encoder to handle datetime and Decimal objects
        def custom_encoder(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            raise TypeError(repr(obj) + " is not JSON serializable")
        
        message_json = json.dumps(message, default=custom_encoder)
        disconnected = set()
        
        # Send to all connected clients
        for connection in self.connections.copy():
            try:
                await connection.send_text(message_json)
                self.stats["messages_sent"] += 1
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")
                disconnected.add(connection)
                self.stats["broadcast_errors"] += 1
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)
    

    async def broadcast_chart_data(self, chart_data):
        """Broadcast historical chart data for chart rendering.
        
        This method sends complete historical time series data (excluding current period)
        for rendering chart bars/lines. Current period data comes from realtime_update.
        
        Args:
            chart_data: Dict containing historical time series for both hour/minute and day/hour modes
        """
        try:
            chart_message = ChartDataMessage(
                type="chart_data",
                data=chart_data
            )
            
            await self.broadcast(chart_message.model_dump())
            
            hour_bars = len(chart_data.get("hour_minute_mode", {}).get("time_series", []))
            day_bars = len(chart_data.get("day_hour_mode", {}).get("time_series", []))
            logger.debug(f"Broadcast chart data: {hour_bars} minute bars, {day_bars} hour bars (historical only)")
            
        except Exception as e:
            logger.error(f"Failed to broadcast chart data: {e}")

    async def broadcast_realtime_update(self, realtime_data):
        """Broadcast real-time updates for current period data everywhere it appears.
        
        This method provides instant updates for:
        - Current minute/hour stats boxes
        - Current period bar in chart (rightmost bar)
        - Total volume/trades (mode-specific: hour vs day)
        - Peak stats
        
        Args:
            realtime_data: Dict containing structured real-time update data
        """
        try:
            realtime_message = RealtimeUpdateMessage(
                type="realtime_update",
                data=realtime_data
            )
            
            await self.broadcast(realtime_message.model_dump())
            
            current_minute = realtime_data.get("current_minute", {})
            current_hour = realtime_data.get("current_hour", {})
            logger.debug(f"Broadcast realtime update: ${current_minute.get('volume_usd', 0):.2f} (minute), ${current_hour.get('volume_usd', 0):.2f} (hour)")
            
        except Exception as e:
            logger.error(f"Failed to broadcast realtime update: {e}")
    
    async def broadcast_analytics_update(self, analytics_data):
        """Broadcast unified analytics update for clean real-time updates.
        
        This method sends complete analytics data for a single mode, replacing
        the complex 3-message system (snapshot, realtime_update, chart_data).
        
        Provides:
        - Current period data (guaranteed timestamp consistency)
        - Summary statistics (total/peak for the mode window)
        - Time series data (limited recent history for chart rendering)
        
        Args:
            analytics_data: Dict containing unified analytics data for one mode
        """
        try:
            analytics_message = AnalyticsUpdateMessage(
                type="analytics_update",
                data=analytics_data
            )
            
            await self.broadcast(analytics_message.model_dump())
            
            mode = analytics_data.get("mode", "unknown")
            current_period = analytics_data.get("current_period", {})
            time_series_count = len(analytics_data.get("time_series", []))
            
            logger.debug(f"Broadcast analytics update ({mode} mode): "
                        f"${current_period.get('volume_usd', 0):.2f} current, "
                        f"{time_series_count} historical points")
            
        except Exception as e:
            logger.error(f"Failed to broadcast analytics update: {e}")
    
    async def _ping_loop(self, websocket: WebSocket):
        """Send periodic ping frames to keep connection alive."""
        try:
            while websocket in self.connections:
                await asyncio.sleep(self._ping_interval)
                
                if websocket not in self.connections:
                    break
                
                try:
                    # Send ping message
                    ping_message = {"type": "ping", "timestamp": asyncio.get_event_loop().time()}
                    await websocket.send_text(json.dumps(ping_message))
                    self.stats["pings_sent"] += 1
                    
                    logger.debug(f"Sent ping to WebSocket connection")
                    
                except Exception as e:
                    logger.warning(f"Failed to send ping: {e}")
                    # Connection likely closed, will be cleaned up by disconnect handler
                    break
                    
        except asyncio.CancelledError:
            logger.debug("Ping loop cancelled for WebSocket connection")
        except Exception as e:
            logger.error(f"Error in ping loop: {e}")
            
    async def _send_snapshot(self, websocket: WebSocket):
        """Send initial snapshot data to a newly connected client."""
        try:
            trade_processor = get_trade_processor()
            snapshot_data = await trade_processor.get_snapshot_data()
            
            snapshot_message = SnapshotMessage(
                type="snapshot",
                data=snapshot_data
            )
            
            # Custom encoder to handle datetime and Decimal objects
            def custom_encoder(obj):
                if isinstance(obj, Decimal):
                    return float(obj)
                elif isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                raise TypeError(repr(obj) + " is not JSON serializable")
            
            await websocket.send_text(json.dumps(snapshot_message.model_dump(), default=custom_encoder))
            
            logger.debug("Sent snapshot data to new client")
            
        except Exception as e:
            logger.error(f"Failed to send snapshot to client: {e}")
            # Don't disconnect the client - they might still receive live updates
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broadcaster statistics."""
        return self.stats.copy()


class TradeStreamEndpoint(WebSocketEndpoint):
    """WebSocket endpoint for streaming trade data to frontend."""
    
    encoding = "text"
    
    def __init__(self, scope, receive, send):
        super().__init__(scope, receive, send)
        self.broadcaster = get_websocket_broadcaster()
    
    async def on_connect(self, websocket: WebSocket):
        """Handle new WebSocket connection."""
        await self.broadcaster.connect(websocket)
    
    async def on_disconnect(self, websocket: WebSocket, close_code: int):
        """Handle WebSocket disconnection."""
        self.broadcaster.disconnect(websocket)
    
    async def on_receive(self, websocket: WebSocket, data: str):
        """Handle incoming messages from client."""
        try:
            message = json.loads(data)
            logger.debug(f"Received WebSocket message: {message}")
            
            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": asyncio.get_event_loop().time()}))
            elif message.get("type") == "pong":
                # Client responding to our ping - connection is alive
                logger.debug("Received pong from client")
            elif message.get("type") == "get_snapshot":
                await self.broadcaster._send_snapshot(websocket)
            else:
                logger.warning(f"Unknown message type: {message.get('type')}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {data}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")


class WebSocketManager:
    """High-level manager for WebSocket operations."""
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.broadcaster = WebSocketBroadcaster()
        self._initialized = False
    
    async def initialize(self):
        """Initialize the WebSocket manager."""
        if self._initialized:
            return
        
        # Set up trade processor integration
        trade_processor = get_trade_processor()
        trade_processor.set_websocket_broadcaster(self.broadcaster)
        
        self._initialized = True
        logger.info("WebSocket manager initialized")
    
    def get_broadcaster(self) -> WebSocketBroadcaster:
        """Get the broadcaster instance."""
        return self.broadcaster
    
    def get_endpoint_class(self):
        """Get the WebSocket endpoint class for routing."""
        return TradeStreamEndpoint
    
    async def broadcast_system_message(self, message: str, message_type: str = "system"):
        """Broadcast a system message to all connected clients."""
        system_message = {
            "type": message_type,
            "data": {
                "message": message,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        
        await self.broadcaster.broadcast(system_message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive WebSocket statistics."""
        return {
            "broadcaster": self.broadcaster.get_stats(),
            "initialized": self._initialized
        }


# Global instances
_websocket_manager = None
_websocket_broadcaster = None

def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    global _websocket_manager
    if _websocket_manager is None:
        _websocket_manager = WebSocketManager()
    return _websocket_manager

def get_websocket_broadcaster() -> WebSocketBroadcaster:
    """Get the global WebSocket broadcaster instance."""
    global _websocket_broadcaster
    if _websocket_broadcaster is None:
        manager = get_websocket_manager()
        _websocket_broadcaster = manager.get_broadcaster()
    return _websocket_broadcaster