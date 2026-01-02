"""
WebSocket client for Kalshi public trades stream.

Provides TradesClient that connects to Kalshi's trade WebSocket channel
for monitoring public trades across all markets. Used for whale detection
in the V3 trader.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, List
import websockets
from websockets.exceptions import ConnectionClosed, InvalidMessage

from kalshiflow.auth import KalshiAuth

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.trades_client")


class TradesClient:
    """
    WebSocket client for Kalshi public trades stream.

    Features:
    - Connects to Kalshi WebSocket with RSA authentication
    - Subscribes to "trade" channel for all public trades
    - Automatic reconnection with exponential backoff
    - Callback pattern for trade processing
    - Health monitoring for integration
    """

    def __init__(
        self,
        ws_url: str,
        auth: KalshiAuth,
        on_trade_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 1.0,
    ):
        """
        Initialize trades client.

        Args:
            ws_url: Kalshi WebSocket URL
            auth: KalshiAuth instance for authentication
            on_trade_callback: Async callback function(trade_data: dict) for trade events
            max_reconnect_attempts: Maximum reconnection attempts before giving up
            base_reconnect_delay: Base delay for exponential backoff
        """
        self.ws_url = ws_url
        self.auth = auth
        self.on_trade_callback = on_trade_callback
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_reconnect_delay = base_reconnect_delay

        # Connection state
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_count = 0

        # Statistics
        self._messages_received = 0
        self._trades_received = 0
        self._connection_start_time: Optional[float] = None
        self._last_message_time: Optional[float] = None
        self._client_start_time: Optional[float] = None

        # Connection tracking
        self._connection_established = asyncio.Event()

        # Client task for cleanup
        self._client_task: Optional[asyncio.Task] = None

        logger.info(f"TradesClient initialized for {ws_url}")

    async def start(self) -> None:
        """Start the trades client and begin connection."""
        if self._running:
            logger.warning("TradesClient is already running")
            return

        logger.info("Starting TradesClient")
        self._running = True
        self._reconnect_count = 0
        self._client_start_time = time.time()

        # Start connection loop
        await self._connection_loop()

    async def stop(self) -> None:
        """Stop the trades client."""
        logger.info("Stopping TradesClient...")
        self._running = False

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
            self._websocket = None

        logger.info(
            f"TradesClient stopped. Final stats: "
            f"messages={self._messages_received}, trades={self._trades_received}, "
            f"reconnects={self._reconnect_count}"
        )

    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """
        Wait for WebSocket connection to be established.

        Args:
            timeout: Maximum time to wait for connection (seconds)

        Returns:
            True if connection established, False if timeout
        """
        try:
            await asyncio.wait_for(self._connection_established.wait(), timeout=timeout)
            logger.info("TradesClient connection established successfully")
            return True
        except asyncio.TimeoutError:
            logger.error(f"TradesClient connection timeout after {timeout}s")
            return False

    async def _connection_loop(self) -> None:
        """Main connection loop with automatic reconnection."""
        while self._running:
            try:
                await self._connect_and_subscribe()
            except Exception as e:
                logger.error(f"Connection loop error: {e}", exc_info=True)

            if self._running:
                # Calculate reconnect delay with exponential backoff
                delay = min(self.base_reconnect_delay * (2 ** self._reconnect_count), 60)
                logger.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count + 1})")
                await asyncio.sleep(delay)
                self._reconnect_count += 1

                if self._reconnect_count >= self.max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts ({self.max_reconnect_attempts}) reached")
                    self._running = False
                    break

    async def _connect_and_subscribe(self) -> None:
        """Connect to WebSocket and subscribe to trades channel."""
        # Create authentication headers
        auth_headers = self.auth.create_auth_headers("GET", "/trade-api/ws/v2")
        logger.info(f"Connecting to trades WebSocket: {self.ws_url}")

        async with websockets.connect(
            self.ws_url,
            additional_headers=auth_headers,
            ping_interval=25,
            ping_timeout=15,
            close_timeout=10,
            max_size=2**20,
        ) as websocket:
            self._websocket = websocket
            self._connection_start_time = time.time()
            self._reconnect_count = 0  # Reset on successful connection

            logger.info("TradesClient WebSocket connected")

            # Subscribe to trades channel
            await self._subscribe_to_trades()

            # Process messages
            await self._message_loop()

        # Clear websocket when context exits
        self._websocket = None
        self._connection_start_time = None
        self._connection_established.clear()

    async def _subscribe_to_trades(self) -> None:
        """Subscribe to the public trades channel."""
        subscription_message = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["trade"]
            }
        }

        logger.info(f"Sending trades subscription: {json.dumps(subscription_message)}")
        await self._websocket.send(json.dumps(subscription_message))

        # Wait for subscription response
        try:
            response = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            response_data = json.loads(response)

            if response_data.get("id") == 1 and response_data.get("type") == "subscribed":
                logger.info("Successfully subscribed to public trades channel")
                self._connection_established.set()
            else:
                logger.warning(f"Unexpected subscription response: {response_data}")
                # Still mark as connected, subscription may have worked
                self._connection_established.set()
        except asyncio.TimeoutError:
            logger.warning("Trade subscription response timeout, assuming subscribed")
            self._connection_established.set()

    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        try:
            async for message in self._websocket:
                if not self._running:
                    break

                await self._process_message(message)

        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self._connection_established.clear()
        except InvalidMessage as e:
            logger.error(f"Invalid WebSocket message: {e}")
        except Exception as e:
            logger.error(f"Message loop error: {e}", exc_info=True)
            raise

    async def _process_message(self, raw_message: str) -> None:
        """
        Process a single WebSocket message.

        Args:
            raw_message: Raw message string from WebSocket
        """
        try:
            message = json.loads(raw_message)
            self._messages_received += 1
            self._last_message_time = time.time()

            msg_type = message.get("type")

            if msg_type == "trade":
                await self._process_trade(message)
            elif msg_type == "subscribed":
                logger.debug(f"Subscription confirmed: {message.get('id')}")
            elif msg_type == "heartbeat":
                pass  # Silently ignore heartbeats
            else:
                if self._messages_received % 100 == 0:
                    logger.debug(f"Unhandled message type: {msg_type}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message as JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

    async def _process_trade(self, message: Dict[str, Any]) -> None:
        """
        Process a trade message.

        Trade message format from Kalshi:
        {
            "type": "trade",
            "msg": {
                "market_ticker": "TICKER",
                "yes_price": 65,
                "no_price": 35,
                "count": 100,
                "taker_side": "yes" or "no",
                "ts": 1703700000 (seconds or milliseconds)
            }
        }

        Args:
            message: Raw trade message from WebSocket
        """
        try:
            msg_data = message.get("msg", {})

            # Extract and normalize trade data
            market_ticker = msg_data.get("market_ticker")
            if not market_ticker:
                logger.warning("Trade message missing market_ticker")
                return

            # Normalize timestamp to milliseconds
            raw_ts = msg_data.get("ts", 0)
            timestamp_ms = raw_ts * 1000 if raw_ts < 2000000000 else raw_ts

            # Build normalized trade data
            trade_data = {
                "market_ticker": market_ticker,
                "yes_price": msg_data.get("yes_price", 0),
                "no_price": msg_data.get("no_price", 0),
                "count": msg_data.get("count", 0),
                "taker_side": msg_data.get("taker_side", "unknown"),
                "timestamp_ms": timestamp_ms,
            }

            self._trades_received += 1

            # Call trade callback if provided
            if self.on_trade_callback:
                if asyncio.iscoroutinefunction(self.on_trade_callback):
                    await self.on_trade_callback(trade_data)
                else:
                    self.on_trade_callback(trade_data)

            # Log periodically
            if self._trades_received % 1000 == 0:
                logger.info(f"Trades checkpoint: {self._trades_received} trades processed")

        except Exception as e:
            logger.error(f"Error processing trade: {e}", exc_info=True)

    def is_healthy(self) -> bool:
        """
        Check if client is healthy based on connection and message activity.

        Returns:
            True if healthy, False otherwise
        """
        if not self._running:
            return False

        if not self._websocket:
            return False

        # Check if WebSocket is closed
        try:
            if hasattr(self._websocket, 'closed') and self._websocket.closed:
                return False
        except (AttributeError, TypeError):
            pass

        # Message-based health check
        current_time = time.time()

        if self._last_message_time:
            time_since_message = current_time - self._last_message_time

            if time_since_message > 300:  # No message for 5 minutes = unhealthy
                return False
            elif time_since_message > 60:  # Degraded if > 1 minute
                if not hasattr(self, '_last_degraded_warning') or (current_time - self._last_degraded_warning) > 60:
                    logger.warning(f"TradesClient degraded: {time_since_message:.1f}s since last message")
                    self._last_degraded_warning = current_time
        elif self._connection_start_time:
            # Grace period for first message
            time_since_connection = current_time - self._connection_start_time
            if time_since_connection > 30:  # 30 seconds grace period
                return False
        else:
            return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        uptime = None
        if self._connection_start_time:
            uptime = time.time() - self._connection_start_time

        message_age = None
        if self._last_message_time:
            message_age = time.time() - self._last_message_time

        return {
            "running": self._running,
            "connected": self._websocket is not None,
            "reconnect_count": self._reconnect_count,
            "messages_received": self._messages_received,
            "trades_received": self._trades_received,
            "uptime_seconds": uptime,
            "last_message_time": self._last_message_time,
            "last_message_age_seconds": message_age,
        }

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information.

        Returns:
            Dictionary with health status and operational details
        """
        stats = self.get_stats()
        return {
            "healthy": self.is_healthy(),
            "connected": stats.get("connected", False),
            "running": stats.get("running", False),
            "ws_url": self.ws_url,
            "messages_received": stats.get("messages_received", 0),
            "trades_received": stats.get("trades_received", 0),
            "last_message_time": stats.get("last_message_time"),
            "last_message_age_seconds": stats.get("last_message_age_seconds"),
            "uptime_seconds": stats.get("uptime_seconds"),
            "reconnect_count": stats.get("reconnect_count", 0),
        }

    @classmethod
    def from_env(
        cls,
        on_trade_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> 'TradesClient':
        """
        Create TradesClient instance from environment variables.

        Required environment variables:
        - KALSHI_API_KEY_ID: The API key ID
        - KALSHI_PRIVATE_KEY_CONTENT: RSA private key content
        - KALSHI_WS_URL: WebSocket URL

        Args:
            on_trade_callback: Callback function for trade events

        Returns:
            TradesClient instance
        """
        import os

        ws_url = os.getenv(
            "KALSHI_WS_URL",
            "wss://api.elections.kalshi.com/trade-api/ws/v2"
        )

        auth = KalshiAuth.from_env()

        return cls(
            ws_url=ws_url,
            auth=auth,
            on_trade_callback=on_trade_callback,
        )
