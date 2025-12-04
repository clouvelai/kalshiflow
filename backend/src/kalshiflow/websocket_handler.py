"""
WebSocket handler for real-time communication with frontend clients.

Provides the /ws/stream endpoint for frontend to receive trade updates
and broadcasts trade data to all connected clients.
"""

import json
import asyncio
import logging
from typing import Set, Dict, Any
from starlette.websockets import WebSocket, WebSocketDisconnect
from starlette.endpoints import WebSocketEndpoint

from .models import SnapshotMessage
from .trade_processor import get_trade_processor


logger = logging.getLogger(__name__)


class WebSocketBroadcaster:
    """Manages WebSocket connections and broadcasts messages to clients."""
    
    def __init__(self):
        """Initialize the broadcaster."""
        self.connections: Set[WebSocket] = set()
        self._running = False
        
        # Statistics
        self.stats = {
            "active_connections": 0,
            "total_connections": 0,
            "messages_sent": 0,
            "broadcast_errors": 0
        }
    
    async def connect(self, websocket: WebSocket):
        """Add a new WebSocket connection."""
        await websocket.accept()
        self.connections.add(websocket)
        self.stats["active_connections"] = len(self.connections)
        self.stats["total_connections"] += 1
        
        logger.info(f"WebSocket connected. Active connections: {len(self.connections)}")
        
        # Send snapshot data to new client
        await self._send_snapshot(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.connections.discard(websocket)
        self.stats["active_connections"] = len(self.connections)
        
        logger.info(f"WebSocket disconnected. Active connections: {len(self.connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        if not self.connections:
            return
        
        message_json = json.dumps(message)
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
    
    async def _send_snapshot(self, websocket: WebSocket):
        """Send initial snapshot data to a newly connected client."""
        try:
            trade_processor = get_trade_processor()
            snapshot_data = await trade_processor.get_snapshot_data()
            
            snapshot_message = SnapshotMessage(
                type="snapshot",
                data=snapshot_data
            )
            
            await websocket.send_text(json.dumps(snapshot_message.dict()))
            
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
                await websocket.send_text(json.dumps({"type": "pong"}))
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