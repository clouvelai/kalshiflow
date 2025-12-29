"""
FillListener - WebSocket listener for real-time order fill notifications.

This service connects to the Kalshi user WebSocket and subscribes to the
"fill" channel to receive real-time notifications when our orders are filled.
Updates are forwarded to the event bus for console UX and state updates.

Architecture:
    KalshiWS (fill channel) -> FillListener -> EventBus -> Console UX

Based on Kalshi Fill documentation:
https://docs.kalshi.com/websockets/user-fills

Fill Message Format (from Kalshi):
{
    "type": "fill",
    "sid": 13,
    "msg": {
        "trade_id": "d91bc706-ee49-470d-82d8-11418bda6fed",
        "order_id": "ee587a1c-8b87-4dcf-b721-9f6f790619fa",
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

Note: All price values from Kalshi fill WebSocket are in cents (1-99).
The fill channel requires authentication but ignores market_tickers parameter
(always sends all fills for your account).
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

# Import authentication from main kalshiflow package
from kalshiflow.auth import KalshiAuth

from ...config import config

if TYPE_CHECKING:
    from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.fill_listener")


class FillListenerError(Exception):
    """Base exception for FillListener errors."""
    pass


class FillListenerAuthError(FillListenerError):
    """Authentication error for fill listener."""
    pass


class FillListener:
    """
    WebSocket listener for Kalshi real-time order fill notifications.

    Connects to Kalshi WebSocket, subscribes to the "fill" channel,
    and emits fill events via the event bus for immediate console feedback.

    Features:
    - Automatic reconnection on disconnect
    - Authentication using KalshiAuth
    - Heartbeat monitoring
    - Clean shutdown handling
    - Metrics tracking (fills received, last fill time, etc.)
    """

    def __init__(
        self,
        event_bus: "EventBus",
        ws_url: Optional[str] = None,
        reconnect_delay_seconds: float = 5.0,
        heartbeat_timeout_seconds: float = 30.0,
    ):
        """
        Initialize the FillListener.

        Args:
            event_bus: EventBus instance for emitting fill events
            ws_url: WebSocket URL (defaults to config.KALSHI_WS_URL)
            reconnect_delay_seconds: Delay before reconnection attempts
            heartbeat_timeout_seconds: Timeout for heartbeat monitoring
        """
        self._event_bus = event_bus
        self.ws_url = ws_url or config.KALSHI_WS_URL
        self.reconnect_delay = reconnect_delay_seconds
        self.heartbeat_timeout = heartbeat_timeout_seconds

        # WebSocket state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._shutdown_requested = False
        self._listener_task: Optional[asyncio.Task] = None

        # Authentication - reuse KalshiAuth
        self._auth: Optional[KalshiAuth] = None
        self._temp_key_file: Optional[str] = None

        # Metrics
        self._fills_received = 0
        self._fills_processed = 0
        self._connection_count = 0
        self._last_fill_time: Optional[float] = None
        self._last_heartbeat_time: Optional[float] = None

        # Message counter for subscription IDs
        self._message_id = 0

        logger.info(f"FillListener initialized (ws_url={self.ws_url})")

    async def start(self) -> None:
        """
        Start the fill listener.

        Initializes authentication and starts the WebSocket listener task.
        """
        if self._running:
            logger.warning("FillListener already running")
            return

        logger.info("Starting FillListener...")

        # Initialize authentication
        await self._setup_auth()

        # Start listener task
        self._shutdown_requested = False
        self._running = True
        self._listener_task = asyncio.create_task(self._listener_loop())

        logger.info("FillListener started")

    async def stop(self) -> None:
        """
        Stop the fill listener gracefully.
        """
        if not self._running:
            return

        logger.info("Stopping FillListener...")

        self._shutdown_requested = True
        self._running = False

        # Close WebSocket connection
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self._ws = None

        # Cancel listener task
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        # Cleanup auth
        self._cleanup_auth()

        logger.info("FillListener stopped")

    async def _setup_auth(self) -> None:
        """
        Set up authentication for WebSocket connection.

        Creates KalshiAuth instance using configured credentials.
        """
        import tempfile
        import os

        if not config.KALSHI_API_KEY_ID:
            raise FillListenerAuthError("KALSHI_API_KEY_ID not configured")

        if not config.KALSHI_PRIVATE_KEY_CONTENT:
            raise FillListenerAuthError("KALSHI_PRIVATE_KEY_CONTENT not configured")

        try:
            # Create temporary file for private key
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pem', prefix='fill_listener_key_')
            self._temp_key_file = temp_path

            with os.fdopen(temp_fd, 'w') as temp_file:
                private_key_content = config.KALSHI_PRIVATE_KEY_CONTENT

                if not private_key_content.startswith('-----BEGIN'):
                    formatted_key = f"-----BEGIN PRIVATE KEY-----\n{private_key_content}\n-----END PRIVATE KEY-----"
                else:
                    formatted_key = private_key_content.replace('\\n', '\n')

                    # Handle case where newlines might be lost
                    if '\n' not in formatted_key and '-----BEGIN' in formatted_key:
                        begin_marker = '-----BEGIN'
                        end_marker = '-----END'
                        begin_idx = formatted_key.find(begin_marker)
                        end_idx = formatted_key.find(end_marker)

                        if begin_idx != -1 and end_idx != -1:
                            begin_end = formatted_key.find('-----', begin_idx + len(begin_marker))
                            if begin_end != -1:
                                begin_end += 5
                                content = formatted_key[begin_end:end_idx].strip()
                                formatted_key = (
                                    formatted_key[:begin_end] + '\n' +
                                    content + '\n' +
                                    formatted_key[end_idx:]
                                )

                temp_file.write(formatted_key)

            self._auth = KalshiAuth(
                api_key_id=config.KALSHI_API_KEY_ID,
                private_key_path=temp_path
            )

            logger.debug("FillListener authentication initialized")

        except Exception as e:
            self._cleanup_auth()
            raise FillListenerAuthError(f"Failed to initialize auth: {e}")

    def _cleanup_auth(self) -> None:
        """Clean up temporary authentication files."""
        import os

        if self._temp_key_file:
            try:
                os.unlink(self._temp_key_file)
                logger.debug("Cleaned up temporary key file")
            except Exception as e:
                logger.warning(f"Failed to clean up temp key file: {e}")
            self._temp_key_file = None

        self._auth = None

    def _get_next_message_id(self) -> int:
        """Get the next message ID for subscription commands."""
        self._message_id += 1
        return self._message_id

    async def _listener_loop(self) -> None:
        """
        Main listener loop with automatic reconnection.
        """
        while self._running and not self._shutdown_requested:
            try:
                await self._connect_and_listen()
            except asyncio.CancelledError:
                logger.info("Listener loop cancelled")
                break
            except Exception as e:
                if not self._shutdown_requested:
                    logger.error(f"WebSocket error: {e}", exc_info=True)
                    logger.info(f"Reconnecting in {self.reconnect_delay}s...")
                    await asyncio.sleep(self.reconnect_delay)

    async def _connect_and_listen(self) -> None:
        """
        Connect to WebSocket and listen for fill messages.
        """
        if not self._auth:
            raise FillListenerAuthError("Authentication not initialized")

        # Create auth headers for WebSocket
        ws_headers = self._auth.create_auth_headers("GET", "/trade-api/ws/v2")
        if 'Content-Type' in ws_headers:
            del ws_headers['Content-Type']

        logger.info(f"Connecting to WebSocket: {self.ws_url}")

        try:
            # Add timeout to connection attempt
            connection_manager = websockets.connect(
                self.ws_url,
                additional_headers=ws_headers,
                ping_interval=20,
                ping_timeout=10,
            )

            async def connect_with_timeout():
                return await connection_manager.__aenter__()

            ws = await asyncio.wait_for(connect_with_timeout(), timeout=10.0)

            try:
                self._ws = ws
                self._connection_count += 1
                self._last_heartbeat_time = time.time()

                logger.info(f"WebSocket connected (connection #{self._connection_count})")

                # Subscribe to fill channel
                await self._subscribe_to_fills()

                # Listen for messages
                async for message in ws:
                    if self._shutdown_requested:
                        break

                    self._last_heartbeat_time = time.time()
                    await self._handle_message(message)
            finally:
                await connection_manager.__aexit__(None, None, None)

        except asyncio.TimeoutError:
            logger.error(f"WebSocket connection timeout to {self.ws_url}")
            raise
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise

    async def _subscribe_to_fills(self) -> None:
        """
        Subscribe to the fill channel.

        Sends subscription command to Kalshi WebSocket.
        Note: The fill channel ignores market_tickers and sends all fills.
        """
        if not self._ws:
            return

        subscribe_msg = {
            "id": self._get_next_message_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["fill"]
            }
        }

        logger.info("Subscribing to 'fill' channel...")
        await self._ws.send(json.dumps(subscribe_msg))

    async def _handle_message(self, raw_message: str) -> None:
        """
        Handle incoming WebSocket message.

        Parses message and emits fill events via event bus.

        Args:
            raw_message: Raw JSON string from WebSocket
        """
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse WebSocket message: {e}")
            return

        msg_type = message.get("type")

        # Handle subscription acknowledgment
        if msg_type == "subscribed":
            channel = message.get("msg", {}).get("channel", "unknown")
            logger.info(f"Subscribed to '{channel}' channel")
            return

        # Handle heartbeat/ping
        if msg_type in ("heartbeat", "pong"):
            logger.debug("Heartbeat received")
            return

        # Handle fill message
        if msg_type == "fill":
            await self._handle_fill_message(message)
            return

        # Handle error
        if msg_type == "error":
            error_msg = message.get("msg", {})
            logger.error(f"WebSocket error: {error_msg}")
            return

        # Log unknown message types at debug level
        logger.debug(f"Unknown message type: {msg_type}")

    async def _handle_fill_message(self, message: Dict[str, Any]) -> None:
        """
        Handle a fill message from Kalshi.

        Emits fill event via event bus for console UX.

        Args:
            message: Parsed fill message from WebSocket
        """
        self._fills_received += 1
        self._last_fill_time = time.time()

        # Extract fill data from 'msg' key
        fill_data = message.get("msg", {})

        if not fill_data:
            logger.warning(f"Empty fill data in message: {message}")
            return

        # Extract all fill fields
        trade_id = fill_data.get("trade_id", "")
        order_id = fill_data.get("order_id", "")
        market_ticker = fill_data.get("market_ticker", "")
        is_taker = fill_data.get("is_taker", False)
        side = fill_data.get("side", "")
        action = fill_data.get("action", "")
        # Price is in cents (1-99) - no conversion needed
        yes_price = fill_data.get("yes_price", 0)
        count = fill_data.get("count", 0)
        post_position = fill_data.get("post_position", 0)
        fill_timestamp = fill_data.get("ts", 0)

        # Calculate total cost in cents
        total_cents = yes_price * count

        # Log the fill with nice formatting
        logger.info(
            f"ORDER FILLED: {action.upper()} {count} {side.upper()} "
            f"{market_ticker} @ {yes_price}c (${total_cents / 100:.2f})"
        )

        # Emit via event bus
        try:
            await self._event_bus.emit_order_fill(
                trade_id=trade_id,
                order_id=order_id,
                market_ticker=market_ticker,
                is_taker=is_taker,
                side=side,
                action=action,
                price_cents=yes_price,
                count=count,
                post_position=post_position,
                fill_timestamp=fill_timestamp,
            )
            self._fills_processed += 1
            logger.debug(f"Fill event emitted for {market_ticker}")

        except Exception as e:
            logger.error(f"Failed to emit fill event for {market_ticker}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get FillListener metrics.

        Returns:
            Dictionary with listener statistics
        """
        # Check connection status safely (different WS libraries have different attributes)
        connected = False
        if self._ws is not None:
            if hasattr(self._ws, 'closed'):
                connected = not self._ws.closed
            elif hasattr(self._ws, 'close_code'):
                connected = self._ws.close_code is None
            else:
                connected = True  # Assume connected if we can't check

        return {
            "running": self._running,
            "connected": connected,
            "fills_received": self._fills_received,
            "fills_processed": self._fills_processed,
            "connection_count": self._connection_count,
            "last_fill_time": self._last_fill_time,
            "last_heartbeat_time": self._last_heartbeat_time,
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get FillListener status for monitoring.

        Returns:
            Status dictionary
        """
        return {
            "service": "FillListener",
            "status": "running" if self._running else "stopped",
            "ws_url": self.ws_url,
            "metrics": self.get_metrics(),
        }

    def is_healthy(self) -> bool:
        """
        Check if fill listener is healthy.

        Returns:
            True if running and WebSocket is connected
        """
        if not self._running:
            return False

        if self._ws is None:
            return False

        try:
            if hasattr(self._ws, 'closed'):
                if self._ws.closed:
                    return False
            elif hasattr(self._ws, 'close_code'):
                if self._ws.close_code is not None:
                    return False
        except Exception as e:
            logger.debug(f"Could not check WebSocket connection state: {e}")
            pass

        return True

    def get_health_details(self) -> Dict[str, Any]:
        """
        Get detailed health information for initialization tracker.

        Returns:
            Dictionary with health status and connection details
        """
        metrics = self.get_metrics()

        connected = False
        if self._ws is not None:
            try:
                if hasattr(self._ws, 'closed'):
                    try:
                        connected = not self._ws.closed
                    except (AttributeError, TypeError):
                        connected = True
                elif hasattr(self._ws, 'close_code'):
                    try:
                        connected = self._ws.close_code is None
                    except (AttributeError, TypeError):
                        connected = True
                else:
                    connected = True
            except Exception as e:
                logger.debug(f"Could not determine WebSocket connection status: {e}")
                connected = True

        return {
            "running": self._running,
            "connected": connected,
            "ws_url": self.ws_url,
            "fills_received": metrics.get("fills_received", 0),
            "fills_processed": metrics.get("fills_processed", 0),
            "connection_count": metrics.get("connection_count", 0),
            "last_fill_time": metrics.get("last_fill_time"),
        }
