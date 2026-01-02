"""
WebSocket client for Kalshi market lifecycle events stream.

Provides LifecycleClient that connects to Kalshi's market_lifecycle_v2 WebSocket
channel for monitoring market creation, determination, and settlement events.
Used for Event Lifecycle Discovery mode in the V3 trader.

Purpose:
    LifecycleClient enables real-time discovery of new markets as they're created
    on Kalshi, supporting dynamic market tracking and orderbook subscription.

Key Responsibilities:
    1. **WebSocket Connection** - Connect to Kalshi with RSA authentication
    2. **Lifecycle Subscription** - Subscribe to market_lifecycle_v2 channel (all events)
    3. **Event Processing** - Parse and emit lifecycle events via callback
    4. **Reconnection** - Handle disconnections with exponential backoff
    5. **Health Monitoring** - Provide health status for V3 integration

Architecture Position:
    Used by:
    - V3LifecycleIntegration: Wraps this client for EventBus integration
    - V3Coordinator: Initializes and manages lifecycle in lifecycle discovery mode

Design Principles:
    - **Follows TradesClient pattern**: Same structure and interface
    - **Non-blocking**: Async callback pattern, no blocking operations
    - **Error Isolation**: Individual event errors don't break the stream
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Callable, Awaitable
import websockets
from websockets.exceptions import ConnectionClosed, InvalidMessage

from kalshiflow.auth import KalshiAuth

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.lifecycle_client")


class LifecycleClient:
    """
    WebSocket client for Kalshi market lifecycle events.

    Features:
    - Connects to Kalshi WebSocket with RSA authentication
    - Subscribes to "market_lifecycle_v2" channel (all events, no filter)
    - Automatic reconnection with exponential backoff
    - Callback pattern for lifecycle event processing
    - Health monitoring for V3 integration

    Lifecycle Event Types:
        - created: Market initialized (trigger for REST lookup)
        - activated: Market becomes tradeable
        - deactivated: Trading paused
        - close_date_updated: Settlement time modified
        - determined: Outcome resolved (trigger orderbook unsubscription)
        - settled: Positions liquidated (final state)
    """

    def __init__(
        self,
        ws_url: str,
        auth: KalshiAuth,
        on_lifecycle_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 1.0,
    ):
        """
        Initialize lifecycle client.

        Args:
            ws_url: Kalshi WebSocket URL
            auth: KalshiAuth instance for authentication
            on_lifecycle_callback: Async callback function(event_data: dict) for lifecycle events
            max_reconnect_attempts: Maximum reconnection attempts before giving up
            base_reconnect_delay: Base delay for exponential backoff
        """
        self.ws_url = ws_url
        self.auth = auth
        self.on_lifecycle_callback = on_lifecycle_callback
        self.max_reconnect_attempts = max_reconnect_attempts
        self.base_reconnect_delay = base_reconnect_delay

        # Connection state
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_count = 0

        # Statistics
        self._messages_received = 0
        self._events_received = 0
        self._events_by_type: Dict[str, int] = {}
        self._connection_start_time: Optional[float] = None
        self._last_message_time: Optional[float] = None
        self._client_start_time: Optional[float] = None

        # Connection tracking
        self._connection_established = asyncio.Event()

        # Client task for cleanup
        self._client_task: Optional[asyncio.Task] = None

        logger.info(f"LifecycleClient initialized for {ws_url}")

    async def start(self) -> None:
        """Start the lifecycle client and begin connection."""
        if self._running:
            logger.warning("LifecycleClient is already running")
            return

        logger.info("Starting LifecycleClient for market_lifecycle_v2 channel")
        self._running = True
        self._reconnect_count = 0
        self._client_start_time = time.time()

        # Start connection loop
        await self._connection_loop()

    async def stop(self) -> None:
        """Stop the lifecycle client."""
        logger.info("Stopping LifecycleClient...")
        self._running = False

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
            self._websocket = None

        logger.info(
            f"LifecycleClient stopped. Final stats: "
            f"messages={self._messages_received}, events={self._events_received}, "
            f"reconnects={self._reconnect_count}, by_type={self._events_by_type}"
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
            logger.info("LifecycleClient connection established successfully")
            return True
        except asyncio.TimeoutError:
            logger.error(f"LifecycleClient connection timeout after {timeout}s")
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
        """Connect to WebSocket and subscribe to lifecycle channel."""
        # Create authentication headers
        auth_headers = self.auth.create_auth_headers("GET", "/trade-api/ws/v2")
        logger.info(f"Connecting to lifecycle WebSocket: {self.ws_url}")

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

            logger.info("LifecycleClient WebSocket connected")

            # Subscribe to lifecycle channel
            await self._subscribe_to_lifecycle()

            # Process messages
            await self._message_loop()

        # Clear websocket when context exits
        self._websocket = None
        self._connection_start_time = None
        self._connection_established.clear()

    async def _subscribe_to_lifecycle(self) -> None:
        """
        Subscribe to the market_lifecycle_v2 channel.

        Subscribes without filters to receive ALL lifecycle events across all markets.
        This allows the EventLifecycleService to filter by category after REST lookup.
        """
        subscription_message = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ["market_lifecycle_v2"]
                # No filter = receive all lifecycle events
            }
        }

        logger.info(f"Sending lifecycle subscription: {json.dumps(subscription_message)}")
        await self._websocket.send(json.dumps(subscription_message))

        # Wait for subscription response
        try:
            response = await asyncio.wait_for(self._websocket.recv(), timeout=10.0)
            response_data = json.loads(response)

            if response_data.get("id") == 1 and response_data.get("type") == "subscribed":
                logger.info("Successfully subscribed to market_lifecycle_v2 channel")
                self._connection_established.set()
            else:
                logger.warning(f"Unexpected subscription response: {response_data}")
                # Still mark as connected, subscription may have worked
                self._connection_established.set()
        except asyncio.TimeoutError:
            logger.warning("Lifecycle subscription response timeout, assuming subscribed")
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

            if msg_type == "market_lifecycle_v2":
                await self._process_lifecycle_event(message)
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

    async def _process_lifecycle_event(self, message: Dict[str, Any]) -> None:
        """
        Process a lifecycle event message.

        Lifecycle message format from Kalshi:
        {
            "type": "market_lifecycle_v2",
            "sid": 12345,
            "msg": {
                "event_type": "created" | "activated" | "determined" | "settled" | etc.,
                "market_ticker": "KXNFL-25JAN05-DET",
                "open_ts": 1736100000,
                "close_ts": 1736200000,
                "additional_metadata": {
                    "title": "NFL: Detroit Lions vs ...",
                    "event_ticker": "KXNFL-25JAN05"
                    // Note: category is NOT included - must REST lookup
                }
            }
        }

        Args:
            message: Raw lifecycle message from WebSocket
        """
        try:
            msg_data = message.get("msg", {})

            # Extract core fields
            event_type = msg_data.get("event_type")
            market_ticker = msg_data.get("market_ticker")

            if not event_type or not market_ticker:
                logger.warning(f"Lifecycle event missing required fields: {msg_data}")
                return

            # Track event counts by type
            self._events_received += 1
            self._events_by_type[event_type] = self._events_by_type.get(event_type, 0) + 1

            # Extract timestamp - prefer from message, fallback to current time
            # Kalshi sends timestamps in seconds
            kalshi_ts = msg_data.get("open_ts") or msg_data.get("close_ts") or int(time.time())

            # Build normalized event data for callback
            event_data = {
                "event_type": event_type,
                "market_ticker": market_ticker,
                "open_ts": msg_data.get("open_ts"),
                "close_ts": msg_data.get("close_ts"),
                "kalshi_ts": kalshi_ts,
                "received_ts": time.time(),
                "sid": message.get("sid"),
                # Include additional metadata if present
                "additional_metadata": msg_data.get("additional_metadata", {}),
                # Include full msg for audit trail
                "raw_msg": msg_data,
            }

            # Log important events
            if event_type in ["created", "determined", "settled"]:
                logger.info(f"Lifecycle event: {event_type} for {market_ticker}")
            else:
                logger.debug(f"Lifecycle event: {event_type} for {market_ticker}")

            # Call lifecycle callback if provided
            if self.on_lifecycle_callback:
                try:
                    await self.on_lifecycle_callback(event_data)
                except Exception as e:
                    logger.error(f"Callback error for {event_type}/{market_ticker}: {e}")

            # Log checkpoint periodically
            if self._events_received % 100 == 0:
                logger.info(
                    f"Lifecycle checkpoint: {self._events_received} events processed, "
                    f"by_type={self._events_by_type}"
                )

        except Exception as e:
            logger.error(f"Error processing lifecycle event: {e}", exc_info=True)

    def is_connected(self) -> bool:
        """
        Check if client is currently connected.

        Returns:
            True if WebSocket is connected, False otherwise
        """
        if not self._websocket:
            return False

        try:
            if hasattr(self._websocket, 'closed') and self._websocket.closed:
                return False
        except (AttributeError, TypeError):
            pass

        return True

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
                    logger.warning(f"LifecycleClient degraded: {time_since_message:.1f}s since last message")
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
            "events_received": self._events_received,
            "events_by_type": dict(self._events_by_type),
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
            "events_received": stats.get("events_received", 0),
            "events_by_type": stats.get("events_by_type", {}),
            "last_message_time": stats.get("last_message_time"),
            "last_message_age_seconds": stats.get("last_message_age_seconds"),
            "uptime_seconds": stats.get("uptime_seconds"),
            "reconnect_count": stats.get("reconnect_count", 0),
        }

    def on_lifecycle_event(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Set the lifecycle event callback.

        Alternative method to setting callback in constructor.

        Args:
            callback: Async callback function(event_data: dict)
        """
        self.on_lifecycle_callback = callback
        logger.debug("Lifecycle event callback registered")

    @classmethod
    def from_env(
        cls,
        on_lifecycle_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> 'LifecycleClient':
        """
        Create LifecycleClient instance from environment variables.

        Required environment variables:
        - KALSHI_API_KEY_ID: The API key ID
        - KALSHI_PRIVATE_KEY_CONTENT: RSA private key content
        - KALSHI_WS_URL: WebSocket URL

        Args:
            on_lifecycle_callback: Callback function for lifecycle events

        Returns:
            LifecycleClient instance
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
            on_lifecycle_callback=on_lifecycle_callback,
        )
