"""
Kalshi WebSocket client with RSA authentication and reconnection logic.
"""

import asyncio
import json
import logging
import time
import random
from typing import Optional, Dict, Any, Callable, List
from urllib.parse import urlparse

import websockets
from websockets.exceptions import WebSocketException, ConnectionClosedError

from .auth import KalshiAuth, KalshiAuthError
from .models import TradeMessage, Trade, ConnectionStatus

logger = logging.getLogger(__name__)


class KalshiClientError(Exception):
    """Custom exception for Kalshi client errors."""
    pass


class KalshiWebSocketClient:
    """
    Kalshi WebSocket client for public trades stream.
    
    Features:
    - RSA authentication using private key file
    - Automatic reconnection with exponential backoff
    - Trade message subscription and parsing
    - Robust error handling and logging
    """
    
    def __init__(
        self,
        websocket_url: str,
        auth: KalshiAuth,
        on_trade_callback: Optional[Callable[[Trade], None]] = None,
        on_connection_change: Optional[Callable[[ConnectionStatus], None]] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 1.0,
        max_reconnect_delay: float = 60.0,
    ):
        """
        Initialize Kalshi WebSocket client.
        
        Args:
            websocket_url: WebSocket URL (e.g., wss://api.elections.kalshi.com/trade-api/ws/v2)
            auth: KalshiAuth instance for authentication
            on_trade_callback: Callback function for trade events
            on_connection_change: Callback function for connection status changes
            max_reconnect_attempts: Maximum number of reconnection attempts before giving up
            base_reconnect_delay: Base delay in seconds for reconnection (exponential backoff)
            max_reconnect_delay: Maximum delay in seconds between reconnection attempts
        """
        self.websocket_url = websocket_url
        self.auth = auth
        self.on_trade_callback = on_trade_callback
        self.on_connection_change = on_connection_change
        
        # Reconnection settings
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_reconnect_delay = base_reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        self.is_authenticated = False
        self.reconnect_attempts = 0
        self.should_reconnect = True
        
        # Tasks
        self.connection_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized Kalshi WebSocket client for {websocket_url}")
    
    async def connect(self) -> bool:
        """
        Connect to Kalshi WebSocket and authenticate.
        
        Returns:
            True if connection and authentication successful, False otherwise
        """
        try:
            logger.info(f"Connecting to Kalshi WebSocket: {self.websocket_url}")
            
            # Create authentication headers for WebSocket handshake
            auth_headers = self.auth.create_auth_headers("GET", "/trade-api/ws/v2")
            
            # Connect to WebSocket with authentication headers
            self.websocket = await websockets.connect(
                self.websocket_url,
                additional_headers=auth_headers,
                ping_interval=25,  # Send ping every 25 seconds (more aggressive for Railway)
                ping_timeout=15,   # Wait 15 seconds for pong
                close_timeout=10,  # Wait 10 seconds for close
                max_size=2**20,    # 1MB max message size
                compression=None,  # Disable compression for lower latency
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("WebSocket connection established")
            
            # Notify connection change
            await self._notify_connection_status(connected=True)
            
            # Authenticate
            if await self.authenticate():
                # Subscribe to trades
                if await self.subscribe_to_trades():
                    logger.info("Successfully connected and subscribed to Kalshi trades")
                    return True
                else:
                    logger.error("Failed to subscribe to trades")
                    await self.disconnect()
                    return False
            else:
                logger.error("Authentication failed")
                await self.disconnect()
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.is_connected = False
            await self._notify_connection_status(connected=False, error_message=str(e))
            return False
    
    async def authenticate(self) -> bool:
        """
        Authenticate with Kalshi WebSocket using RSA signature.
        
        Returns:
            True if authentication successful, False otherwise
        """
        # Authentication is now handled during WebSocket handshake via headers
        logger.info("WebSocket authentication completed via handshake headers")
        self.is_authenticated = True
        return True
    
    async def subscribe_to_trades(self) -> bool:
        """
        Subscribe to public trades channel.
        
        Returns:
            True if subscription successful, False otherwise
        """
        try:
            logger.info("Subscribing to public trades channel")
            
            # Subscribe to trade channel
            trades_message = {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["trade"]
                }
            }
            
            await self.websocket.send(json.dumps(trades_message))
            logger.debug("Sent trades subscription message")
            
            # Wait for subscription response
            response = await asyncio.wait_for(self.websocket.recv(), timeout=10.0)
            response_data = json.loads(response)
            
            logger.debug(f"Trades subscription response: {response_data}")
            
            # Check if subscription was successful
            if response_data.get("id") == 1 and response_data.get("type") == "subscribed":
                logger.info("Successfully subscribed to public trades")
                return True
            else:
                logger.error(f"Trades subscription failed: {response_data}")
                return False
                
        except asyncio.TimeoutError:
            logger.error("Trades subscription timeout")
            return False
        except Exception as e:
            logger.error(f"Trades subscription error: {e}")
            return False
    
    async def handle_message(self, message: str) -> None:
        """
        Handle incoming WebSocket message.
        
        Args:
            message: Raw JSON message string
        """
        try:
            data = json.loads(message)
            
            # Handle different message types
            message_type = data.get("type")
            
            if message_type == "trade":
                # Parse trade message
                try:
                    trade_msg = TradeMessage(msg=data.get("msg", {}))
                    trade = trade_msg.to_trade()
                    
                    # Call trade callback if provided
                    if self.on_trade_callback:
                        if asyncio.iscoroutinefunction(self.on_trade_callback):
                            await self.on_trade_callback(trade)
                        else:
                            self.on_trade_callback(trade)
                    
                    logger.debug(f"Processed trade: {trade.market_ticker} {trade.taker_side} {trade.price_display}")
                    
                except Exception as e:
                    logger.error(f"Failed to parse trade message: {e}")
                    logger.debug(f"Raw trade message: {data}")
            
            elif message_type == "heartbeat":
                logger.debug("Received heartbeat")
            
            elif message_type == "subscribed":
                logger.info(f"Subscription confirmed: {data.get('id')}")
            
            else:
                logger.debug(f"Received message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON message: {e}")
            logger.debug(f"Raw message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    async def listen(self) -> None:
        """
        Listen for incoming WebSocket messages.
        """
        try:
            logger.info("Starting message listener")
            
            async for message in self.websocket:
                await self.handle_message(message)
                
        except ConnectionClosedError:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
            self.is_authenticated = False
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
            self.is_authenticated = False
        except Exception as e:
            logger.error(f"Unexpected error in listener: {e}")
            self.is_connected = False
            self.is_authenticated = False
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket."""
        logger.info("Disconnecting from Kalshi WebSocket")
        
        self.should_reconnect = False
        self.is_connected = False
        self.is_authenticated = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        await self._notify_connection_status(connected=False)
        logger.info("Disconnected from Kalshi WebSocket")
    
    async def reconnect_with_backoff(self) -> None:
        """
        Attempt to reconnect with exponential backoff.
        """
        while self.should_reconnect and self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            
            # Calculate delay with exponential backoff and jitter
            delay = min(
                self.base_reconnect_delay * (2 ** (self.reconnect_attempts - 1)),
                self.max_reconnect_delay
            )
            
            # Add some jitter to avoid thundering herd
            jitter = random.uniform(0.1, 0.3) * delay
            total_delay = delay + jitter
            
            logger.info(
                f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts} "
                f"in {total_delay:.1f} seconds"
            )
            
            await asyncio.sleep(total_delay)
            
            if await self.connect():
                logger.info("Reconnection successful")
                return
            else:
                logger.warning(f"Reconnection attempt {self.reconnect_attempts} failed")
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Exceeded maximum reconnection attempts ({self.max_reconnect_attempts})")
            await self._notify_connection_status(
                connected=False, 
                error_message=f"Exceeded maximum reconnection attempts"
            )
    
    async def start(self) -> None:
        """
        Start the WebSocket client with automatic reconnection.
        """
        logger.info("Starting Kalshi WebSocket client")
        
        self.should_reconnect = True
        
        while self.should_reconnect:
            try:
                if await self.connect():
                    # Listen for messages
                    await self.listen()
                
                # If we get here, connection was lost
                if self.should_reconnect:
                    logger.info("Connection lost, attempting to reconnect...")
                    await self.reconnect_with_backoff()
                    
            except Exception as e:
                logger.error(f"Unexpected error in client: {e}")
                if self.should_reconnect:
                    await asyncio.sleep(5)  # Wait before retrying
    
    async def stop(self) -> None:
        """
        Stop the WebSocket client.
        """
        logger.info("Stopping Kalshi WebSocket client")
        await self.disconnect()
    
    async def _notify_connection_status(
        self, 
        connected: bool, 
        error_message: Optional[str] = None
    ) -> None:
        """
        Notify connection status change.
        
        Args:
            connected: Whether the connection is active
            error_message: Optional error message
        """
        if self.on_connection_change:
            status = ConnectionStatus(
                connected=connected,
                last_connected=None if not connected else None,
                reconnect_attempts=self.reconnect_attempts,
                error_message=error_message
            )
            self.on_connection_change(status)
    
    @classmethod
    def from_env(
        cls,
        on_trade_callback: Optional[Callable[[Trade], None]] = None,
        on_connection_change: Optional[Callable[[ConnectionStatus], None]] = None,
    ) -> 'KalshiWebSocketClient':
        """
        Create client instance from environment variables.
        
        Required environment variables:
        - KALSHI_API_KEY_ID: The API key ID
        - KALSHI_PRIVATE_KEY_PATH: Path to RSA private key file
        - KALSHI_WS_URL: WebSocket URL (default: wss://api.elections.kalshi.com/trade-api/ws/v2)
        
        Args:
            on_trade_callback: Callback function for trade events
            on_connection_change: Callback function for connection status changes
            
        Returns:
            KalshiWebSocketClient instance
        """
        import os
        
        websocket_url = os.getenv(
            "KALSHI_WS_URL", 
            "wss://api.elections.kalshi.com/trade-api/ws/v2"
        )
        
        auth = KalshiAuth.from_env()
        
        return cls(
            websocket_url=websocket_url,
            auth=auth,
            on_trade_callback=on_trade_callback,
            on_connection_change=on_connection_change,
        )