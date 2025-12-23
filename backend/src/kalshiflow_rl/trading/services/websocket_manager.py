"""
WebSocketManager - Created for TRADER 2.0

Handles WebSocket connections for real-time fill and position events.
Manages connections, subscriptions, and event routing to appropriate handlers.
"""

import asyncio
import json
import logging
import time
import websockets
from typing import Dict, Any, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.trading.services.websocket_manager")


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class WebSocketConfig:
    """WebSocket configuration."""
    url: str
    timeout: float = 10.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 5
    ping_interval: float = 30.0


class WebSocketManager:
    """
    Manages WebSocket connections for real-time trading events.
    
    Handles fills and position updates from Kalshi WebSocket streams
    with automatic reconnection and error recovery.
    """
    
    def __init__(
        self,
        client: 'KalshiDemoTradingClient',
        fill_processor: 'FillProcessor',
        status_logger: Optional['StatusLogger'] = None
    ):
        """
        Initialize WebSocketManager.
        
        Args:
            client: KalshiDemoTradingClient for authentication
            fill_processor: FillProcessor for handling fill events
            status_logger: Optional StatusLogger for activity tracking
        """
        self.client = client
        self.fill_processor = fill_processor
        self.status_logger = status_logger
        
        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connection_task: Optional[asyncio.Task] = None
        self.connected_at: Optional[float] = None
        
        # Configuration
        self.config = WebSocketConfig(
            url="wss://demo-api.kalshi.co/trade-api/ws/v2",  # Demo URL
            timeout=10.0,
            reconnect_delay=5.0,
            max_reconnect_attempts=5,
            ping_interval=30.0
        )
        
        # Reconnection tracking
        self.reconnect_attempts = 0
        self.last_error: Optional[str] = None
        self.stop_requested = False
        
        # Event handlers
        self.message_handlers: Dict[str, Callable] = {
            "fill": self._handle_fill_message,
            "position_update": self._handle_position_message,
            "error": self._handle_error_message
        }
        
        # Statistics
        self.stats = {
            "connection_attempts": 0,
            "successful_connections": 0,
            "messages_received": 0,
            "fills_processed": 0,
            "connection_errors": 0,
            "last_message_time": None,
            "uptime_total": 0.0
        }
        
        # Subscribed channels
        self.subscribed_channels: Set[str] = set()
        
        logger.info("WebSocketManager initialized")
    
    async def connect(self) -> bool:
        """
        Connect to WebSocket.
        
        Returns:
            True if connection succeeded
        """
        try:
            if self.state == ConnectionState.CONNECTED:
                logger.info("WebSocket already connected")
                return True
            
            logger.info(f"Connecting to WebSocket: {self.config.url}")
            
            self.state = ConnectionState.CONNECTING
            self.stats["connection_attempts"] += 1
            
            # Log connection attempt
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "WebSocketManager", "connecting",
                    {"url": self.config.url, "attempt": self.stats["connection_attempts"]}
                )
            
            # Establish connection
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.config.url,
                    ping_interval=self.config.ping_interval,
                    ping_timeout=self.config.timeout
                ),
                timeout=self.config.timeout
            )
            
            # Update state
            self.state = ConnectionState.CONNECTED
            self.connected_at = time.time()
            self.reconnect_attempts = 0
            self.stats["successful_connections"] += 1
            
            # Log success
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "WebSocketManager", "connected",
                    {"connected_at": self.connected_at}
                )
            
            logger.info("WebSocket connected successfully")
            
            # Start message handling
            self.connection_task = asyncio.create_task(self._message_loop())
            
            # Subscribe to channels
            await self._subscribe_to_channels()
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.last_error = error_msg
            self.state = ConnectionState.ERROR
            self.stats["connection_errors"] += 1
            
            logger.error(f"WebSocket connection failed: {error_msg}")
            
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "WebSocketManager", "connection_error",
                    {"error": error_msg}
                )
            
            return False
    
    async def disconnect(self) -> bool:
        """
        Disconnect from WebSocket.
        
        Returns:
            True if disconnected successfully
        """
        try:
            logger.info("Disconnecting WebSocket")
            
            # Signal stop
            self.stop_requested = True
            
            # Cancel connection task
            if self.connection_task and not self.connection_task.done():
                self.connection_task.cancel()
                try:
                    await self.connection_task
                except asyncio.CancelledError:
                    pass
            
            # Close websocket
            if self.websocket:
                await self.websocket.close()
                self.websocket = None
            
            # Update state
            self.state = ConnectionState.DISCONNECTED
            
            # Update uptime
            if self.connected_at:
                self.stats["uptime_total"] += time.time() - self.connected_at
                self.connected_at = None
            
            # Log disconnection
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "WebSocketManager", "disconnected",
                    {"total_uptime": self.stats["uptime_total"]}
                )
            
            logger.info("WebSocket disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
            return False
    
    async def _message_loop(self) -> None:
        """
        Main message handling loop.
        """
        logger.info("WebSocket message loop started")
        
        try:
            while not self.stop_requested and self.websocket:
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.config.timeout
                    )
                    
                    # Process message
                    await self._process_message(message)
                    
                except asyncio.TimeoutError:
                    logger.debug("WebSocket receive timeout")
                    continue
                    
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"WebSocket connection closed: {e}")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in message loop: {e}")
                    await asyncio.sleep(1.0)
        
        except Exception as e:
            logger.error(f"Fatal error in message loop: {e}")
        
        finally:
            # Handle disconnection
            if not self.stop_requested:
                logger.warning("WebSocket disconnected unexpectedly - attempting reconnect")
                await self._handle_reconnection()
            
            logger.info("WebSocket message loop ended")
    
    async def _process_message(self, message: str) -> None:
        """
        Process incoming WebSocket message.
        
        Args:
            message: Raw message string
        """
        try:
            # Parse JSON
            data = json.loads(message)
            
            # Update statistics
            self.stats["messages_received"] += 1
            self.stats["last_message_time"] = time.time()
            
            # Extract message type
            msg_type = data.get("type", "unknown")
            
            logger.debug(f"WebSocket message received: {msg_type}")
            
            # Route to appropriate handler
            if msg_type in self.message_handlers:
                await self.message_handlers[msg_type](data)
            else:
                logger.debug(f"Unhandled message type: {msg_type}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    async def _handle_fill_message(self, data: Dict[str, Any]) -> None:
        """
        Handle fill message.
        
        Args:
            data: Fill message data
        """
        try:
            # Extract fill information
            order_id = data.get("order_id")
            if not order_id:
                logger.warning("Fill message missing order_id")
                return
            
            # Extract fill details
            fill_data = {
                "yes_price": data.get("yes_price"),
                "no_price": data.get("no_price"),
                "count": data.get("count"),
                "timestamp": data.get("timestamp")
            }
            
            # Send to fill processor
            success = await self.fill_processor.add_fill_event(order_id, fill_data)
            
            if success:
                self.stats["fills_processed"] += 1
                logger.debug(f"Fill event queued: {order_id}")
            else:
                logger.warning(f"Failed to queue fill event: {order_id}")
            
            # Log activity
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "fill_received",
                    f"{order_id} - {fill_data.get('count', 'unknown')} contracts",
                    0.0
                )
            
        except Exception as e:
            logger.error(f"Error handling fill message: {e}")
    
    async def _handle_position_message(self, data: Dict[str, Any]) -> None:
        """
        Handle position update message.
        
        Args:
            data: Position message data
        """
        try:
            # Extract position information
            ticker = data.get("ticker")
            position = data.get("position")
            
            logger.debug(f"Position update received: {ticker} - {position}")
            
            # Note: Position updates could trigger state sync
            # For now, just log the event
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "position_update_received",
                    f"{ticker} - {position}",
                    0.0
                )
            
        except Exception as e:
            logger.error(f"Error handling position message: {e}")
    
    async def _handle_error_message(self, data: Dict[str, Any]) -> None:
        """
        Handle error message.
        
        Args:
            data: Error message data
        """
        try:
            error_code = data.get("code", "unknown")
            error_message = data.get("message", "Unknown error")
            
            logger.error(f"WebSocket error received: {error_code} - {error_message}")
            
            # Log error
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "websocket_error",
                    f"{error_code}: {error_message}",
                    0.0
                )
            
        except Exception as e:
            logger.error(f"Error handling error message: {e}")
    
    async def _subscribe_to_channels(self) -> bool:
        """
        Subscribe to required channels.
        
        Returns:
            True if subscriptions succeeded
        """
        try:
            if not self.websocket:
                return False
            
            # Subscribe to user fills
            fill_subscription = {
                "type": "subscribe",
                "channel": "fills",
                "params": {}
            }
            
            await self.websocket.send(json.dumps(fill_subscription))
            self.subscribed_channels.add("fills")
            
            # Subscribe to position updates
            position_subscription = {
                "type": "subscribe", 
                "channel": "positions",
                "params": {}
            }
            
            await self.websocket.send(json.dumps(position_subscription))
            self.subscribed_channels.add("positions")
            
            logger.info(f"Subscribed to channels: {list(self.subscribed_channels)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to channels: {e}")
            return False
    
    async def _handle_reconnection(self) -> None:
        """
        Handle reconnection logic.
        """
        if self.stop_requested:
            return
        
        self.state = ConnectionState.RECONNECTING
        
        while (self.reconnect_attempts < self.config.max_reconnect_attempts and 
               not self.stop_requested):
            
            self.reconnect_attempts += 1
            
            logger.info(f"Attempting WebSocket reconnection {self.reconnect_attempts}/{self.config.max_reconnect_attempts}")
            
            # Wait before reconnection
            await asyncio.sleep(self.config.reconnect_delay * self.reconnect_attempts)
            
            # Attempt reconnection
            if await self.connect():
                logger.info("WebSocket reconnection successful")
                return
            
            logger.warning(f"WebSocket reconnection attempt {self.reconnect_attempts} failed")
        
        # Max reconnection attempts reached
        logger.error(f"WebSocket reconnection failed after {self.config.max_reconnect_attempts} attempts")
        self.state = ConnectionState.ERROR
        
        if self.status_logger:
            await self.status_logger.log_service_status(
                "WebSocketManager", "reconnection_failed",
                {"attempts": self.reconnect_attempts}
            )
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self.state == ConnectionState.CONNECTED and self.websocket is not None
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information."""
        current_time = time.time()
        
        connection_duration = None
        if self.connected_at:
            connection_duration = current_time - self.connected_at
        
        return {
            "state": self.state.value,
            "connected": self.is_connected(),
            "connected_at": self.connected_at,
            "connection_duration": connection_duration,
            "reconnect_attempts": self.reconnect_attempts,
            "last_error": self.last_error,
            "subscribed_channels": list(self.subscribed_channels),
            "config": {
                "url": self.config.url,
                "timeout": self.config.timeout,
                "reconnect_delay": self.config.reconnect_delay,
                "max_reconnect_attempts": self.config.max_reconnect_attempts
            }
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get WebSocket statistics."""
        current_time = time.time()
        
        # Calculate uptime
        current_uptime = 0.0
        if self.connected_at:
            current_uptime = current_time - self.connected_at
        
        total_uptime = self.stats["uptime_total"] + current_uptime
        
        return {
            "connection_attempts": self.stats["connection_attempts"],
            "successful_connections": self.stats["successful_connections"],
            "connection_success_rate": self.stats["successful_connections"] / max(1, self.stats["connection_attempts"]),
            "messages_received": self.stats["messages_received"],
            "fills_processed": self.stats["fills_processed"],
            "connection_errors": self.stats["connection_errors"],
            "last_message_time": self.stats["last_message_time"],
            "time_since_last_message": current_time - self.stats["last_message_time"] if self.stats["last_message_time"] else None,
            "current_uptime": current_uptime,
            "total_uptime": total_uptime,
            "message_rate": self.stats["messages_received"] / max(1, total_uptime) if total_uptime > 0 else 0.0
        }
    
    async def start(self) -> bool:
        """
        Start the WebSocket manager.
        
        Returns:
            True if started successfully
        """
        try:
            logger.info("Starting WebSocket manager")
            
            self.stop_requested = False
            
            # Connect to WebSocket
            success = await self.connect()
            
            if success:
                logger.info("WebSocket manager started successfully")
            else:
                logger.error("Failed to start WebSocket manager")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting WebSocket manager: {e}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop the WebSocket manager.
        
        Returns:
            True if stopped successfully
        """
        try:
            logger.info("Stopping WebSocket manager")
            
            # Disconnect
            success = await self.disconnect()
            
            if success:
                logger.info("WebSocket manager stopped successfully")
            else:
                logger.error("Error stopping WebSocket manager")
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket manager: {e}")
            return False