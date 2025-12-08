"""
WebSocket orderbook client for Kalshi RL Trading Subsystem.

Provides OrderbookClient that connects to Kalshi orderbook WebSocket,
processes snapshots and deltas, updates SharedOrderbookState non-blocking,
and queues messages for database persistence.
"""

import asyncio
import json
import logging
import time
import traceback
from typing import Dict, Any, Optional, Callable
import websockets
from websockets.exceptions import ConnectionClosed, InvalidMessage

from .auth import get_rl_auth
from .orderbook_state import get_shared_orderbook_state, SharedOrderbookState
from .write_queue import write_queue
from ..config import config

logger = logging.getLogger("kalshiflow_rl.orderbook_client")


class OrderbookClient:
    """
    WebSocket client for Kalshi orderbook data.
    
    Features:
    - Connects to Kalshi orderbook WebSocket with authentication
    - Subscribes to orderbook deltas for specified market
    - Processes snapshots and incremental updates
    - Updates in-memory SharedOrderbookState immediately (non-blocking)
    - Queues all messages for database persistence (non-blocking)
    - Automatic reconnection with exponential backoff
    - Sequence number tracking and validation
    """
    
    def __init__(self, market_ticker: str = None):
        """
        Initialize orderbook client.
        
        Args:
            market_ticker: Market ticker to subscribe to (defaults to config)
        """
        self.market_ticker = market_ticker or config.RL_MARKET_TICKER
        self.ws_url = config.KALSHI_WS_URL
        
        # WebSocket connection management
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._running = False
        self._reconnect_count = 0
        self._last_sequence = 0
        
        # Shared orderbook state
        self._orderbook_state: Optional[SharedOrderbookState] = None
        
        # Statistics and monitoring
        self._messages_received = 0
        self._snapshots_received = 0
        self._deltas_received = 0
        self._connection_start_time: Optional[float] = None
        self._last_message_time: Optional[float] = None
        
        # Event handlers
        self._on_connected: Optional[Callable] = None
        self._on_disconnected: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        
        logger.info(f"OrderbookClient initialized for market: {self.market_ticker}")
    
    async def start(self) -> None:
        """Start the orderbook client and begin connection."""
        if self._running:
            logger.warning("OrderbookClient is already running")
            return
        
        logger.info(f"Starting OrderbookClient for {self.market_ticker}")
        self._running = True
        self._reconnect_count = 0
        
        # Get shared orderbook state
        self._orderbook_state = await get_shared_orderbook_state(self.market_ticker)
        
        # Start connection loop
        await self._connection_loop()
    
    async def stop(self) -> None:
        """Stop the orderbook client."""
        logger.info("Stopping OrderbookClient...")
        self._running = False
        
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        logger.info(
            f"OrderbookClient stopped. Final stats: "
            f"messages={self._messages_received}, snapshots={self._snapshots_received}, "
            f"deltas={self._deltas_received}, reconnects={self._reconnect_count}"
        )
    
    async def _connection_loop(self) -> None:
        """Main connection loop with automatic reconnection."""
        while self._running:
            try:
                await self._connect_and_subscribe()
            except Exception as e:
                logger.error(f"Connection loop error: {e}")
                if self._on_error:
                    try:
                        await self._on_error(e)
                    except Exception:
                        pass
            
            if self._running:
                # Calculate reconnect delay with exponential backoff
                delay = min(config.WEBSOCKET_RECONNECT_DELAY * (2 ** self._reconnect_count), 60)
                logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count + 1})")
                await asyncio.sleep(delay)
                self._reconnect_count += 1
                
                if self._reconnect_count >= config.MAX_RECONNECT_ATTEMPTS:
                    logger.error(f"Max reconnect attempts ({config.MAX_RECONNECT_ATTEMPTS}) reached")
                    self._running = False
                    break
    
    async def _connect_and_subscribe(self) -> None:
        """Connect to WebSocket and subscribe to orderbook."""
        auth = get_rl_auth()
        headers = auth.create_websocket_headers()
        
        logger.info(f"Connecting to WebSocket: {self.ws_url}")
        
        async with websockets.connect(
            self.ws_url,
            extra_headers=headers,
            ping_interval=config.WEBSOCKET_PING_INTERVAL,
            ping_timeout=config.WEBSOCKET_TIMEOUT,
            max_size=1024*1024,  # 1MB max message size
            compression=None  # Disable compression for lower latency
        ) as websocket:
            
            self._websocket = websocket
            self._connection_start_time = time.time()
            self._reconnect_count = 0  # Reset on successful connection
            
            logger.info(f"WebSocket connected for {self.market_ticker}")
            
            if self._on_connected:
                try:
                    await self._on_connected()
                except Exception as e:
                    logger.error(f"Connection callback error: {e}")
            
            # Subscribe to orderbook
            await self._subscribe_to_orderbook()
            
            # Process messages
            await self._message_loop()
    
    async def _subscribe_to_orderbook(self) -> None:
        """Subscribe to orderbook channel for the market."""
        subscription_message = {
            "id": f"sub_{self.market_ticker}_{int(time.time())}",
            "cmd": "subscribe",
            "params": {
                "channels": [f"orderbook_delta.{self.market_ticker}"]
            }
        }
        
        await self._websocket.send(json.dumps(subscription_message))
        logger.info(f"Subscribed to orderbook for {self.market_ticker}")
    
    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        try:
            async for message in self._websocket:
                if not self._running:
                    break
                
                await self._process_message(message)
                
        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            if self._on_disconnected:
                try:
                    await self._on_disconnected()
                except Exception:
                    pass
        except InvalidMessage as e:
            logger.error(f"Invalid WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Message loop error: {e}\n{traceback.format_exc()}")
            raise
    
    async def _process_message(self, raw_message: str) -> None:
        """
        Process a single WebSocket message.
        
        Args:
            raw_message: Raw message string from WebSocket
        """
        try:
            # Parse message
            message = json.loads(raw_message)
            self._messages_received += 1
            self._last_message_time = time.time()
            
            # Log first few messages for debugging
            if self._messages_received <= 5:
                logger.debug(f"Received message {self._messages_received}: {message}")
            
            # Determine message type
            msg_type = self._get_message_type(message)
            
            if msg_type == "snapshot":
                await self._process_snapshot(message)
            elif msg_type == "delta":
                await self._process_delta(message)
            elif msg_type == "subscription_ack":
                logger.info(f"Subscription acknowledged for {self.market_ticker}")
            elif msg_type == "heartbeat":
                logger.debug("Received heartbeat")
            else:
                logger.debug(f"Unhandled message type: {msg_type}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message as JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}\n{traceback.format_exc()}")
    
    def _get_message_type(self, message: Dict[str, Any]) -> str:
        """Determine the type of WebSocket message."""
        if "channel" in message:
            channel = message["channel"]
            if "orderbook_delta" in channel:
                if message.get("type") == "snapshot":
                    return "snapshot"
                else:
                    return "delta"
        
        if message.get("msg") == "ack":
            return "subscription_ack"
        
        if "ping" in message or "pong" in message:
            return "heartbeat"
        
        return "unknown"
    
    async def _process_snapshot(self, message: Dict[str, Any]) -> None:
        """Process orderbook snapshot message."""
        try:
            # Extract snapshot data
            data = message.get("data", {})
            
            snapshot_data = {
                "market_ticker": self.market_ticker,
                "timestamp_ms": int(time.time() * 1000),
                "sequence_number": data.get("seq", 0),
                "yes_bids": data.get("yes", {}).get("b", {}),
                "yes_asks": data.get("yes", {}).get("a", {}),
                "no_bids": data.get("no", {}).get("b", {}),
                "no_asks": data.get("no", {}).get("a", {})
            }
            
            self._last_sequence = snapshot_data["sequence_number"]
            self._snapshots_received += 1
            
            # Update in-memory state immediately (non-blocking)
            if self._orderbook_state:
                await self._orderbook_state.apply_snapshot(snapshot_data)
            
            # Queue for database persistence (non-blocking)
            await write_queue.enqueue_snapshot(snapshot_data)
            
            total_levels = (len(snapshot_data['yes_bids']) + len(snapshot_data['yes_asks']) + 
                           len(snapshot_data['no_bids']) + len(snapshot_data['no_asks']))
            logger.info(
                f"Processed snapshot for {self.market_ticker}: seq={self._last_sequence}, "
                f"levels={total_levels}"
            )
            
        except Exception as e:
            logger.error(f"Error processing snapshot: {e}\n{traceback.format_exc()}")
    
    async def _process_delta(self, message: Dict[str, Any]) -> None:
        """Process orderbook delta message."""
        try:
            # Extract delta data
            data = message.get("data", {})
            
            delta_data = {
                "market_ticker": self.market_ticker,
                "timestamp_ms": int(time.time() * 1000),
                "sequence_number": data.get("seq", 0),
                "side": data.get("side"),  # "yes" or "no"
                "action": self._map_delta_action(data),
                "price": data.get("price"),
                "old_size": data.get("old_size"),
                "new_size": data.get("new_size", data.get("size"))
            }
            
            # Validate delta
            if not self._validate_delta(delta_data):
                return
            
            self._last_sequence = delta_data["sequence_number"]
            self._deltas_received += 1
            
            # Update in-memory state immediately (non-blocking)
            if self._orderbook_state:
                success = await self._orderbook_state.apply_delta(delta_data)
                if not success:
                    logger.warning(f"Failed to apply delta: seq={delta_data['sequence_number']}")
            
            # Queue for database persistence (non-blocking)
            await write_queue.enqueue_delta(delta_data)
            
            # Log periodically
            if self._deltas_received % 100 == 0:
                logger.debug(f"Processed {self._deltas_received} deltas for {self.market_ticker}")
            
        except Exception as e:
            logger.error(f"Error processing delta: {e}\n{traceback.format_exc()}")
    
    def _map_delta_action(self, data: Dict[str, Any]) -> str:
        """Map WebSocket delta data to standardized action."""
        new_size = data.get("new_size", data.get("size", 0))
        old_size = data.get("old_size", 0)
        
        if old_size == 0 and new_size > 0:
            return "add"
        elif old_size > 0 and new_size == 0:
            return "remove"
        elif old_size != new_size:
            return "update"
        else:
            return "update"  # Default to update
    
    def _validate_delta(self, delta_data: Dict[str, Any]) -> bool:
        """Validate delta data."""
        required_fields = ["side", "action", "price", "sequence_number"]
        
        for field in required_fields:
            if delta_data.get(field) is None:
                logger.warning(f"Invalid delta - missing {field}: {delta_data}")
                return False
        
        if delta_data["side"] not in ["yes", "no"]:
            logger.warning(f"Invalid side: {delta_data['side']}")
            return False
        
        if delta_data["action"] not in ["add", "remove", "update"]:
            logger.warning(f"Invalid action: {delta_data['action']}")
            return False
        
        return True
    
    # Event handlers
    
    def on_connected(self, callback: Callable) -> None:
        """Set callback for connection events."""
        self._on_connected = callback
    
    def on_disconnected(self, callback: Callable) -> None:
        """Set callback for disconnection events."""
        self._on_disconnected = callback
    
    def on_error(self, callback: Callable) -> None:
        """Set callback for error events."""
        self._on_error = callback
    
    # Statistics and monitoring
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        uptime = None
        if self._connection_start_time:
            uptime = time.time() - self._connection_start_time
        
        return {
            "market_ticker": self.market_ticker,
            "running": self._running,
            "connected": self._websocket is not None,
            "reconnect_count": self._reconnect_count,
            "last_sequence": self._last_sequence,
            "messages_received": self._messages_received,
            "snapshots_received": self._snapshots_received,
            "deltas_received": self._deltas_received,
            "uptime_seconds": uptime,
            "last_message_time": self._last_message_time
        }
    
    def is_healthy(self) -> bool:
        """Check if client is healthy and receiving data."""
        if not self._running or not self._websocket:
            return False
        
        # Check if we've received recent messages
        if self._last_message_time:
            time_since_message = time.time() - self._last_message_time
            if time_since_message > 60:  # No messages for 60 seconds
                return False
        
        return True


# Global orderbook client instance
orderbook_client = OrderbookClient()