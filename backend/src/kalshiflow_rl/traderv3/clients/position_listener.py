"""
PositionListener - WebSocket listener for real-time position updates.

This service connects to the Kalshi user WebSocket and subscribes to the
"market_positions" channel to receive real-time position updates. Updates
are forwarded to the event bus for state container integration.

Architecture:
    KalshiWS (market_positions channel) -> PositionListener -> EventBus -> StateContainer

Based on Kalshi Market Positions documentation:
https://docs.kalshi.com/websockets/market-positions

Position Message Format (from Kalshi):
{
    "type": "market_position",
    "sid": 14,
    "msg": {
        "user_id": "user123",
        "market_ticker": "FED-23DEC-T3.00",
        "position": 100,           // contracts (+ YES, - NO)
        "position_cost": 500000,   // CURRENT VALUE in centi-cents (= REST's market_exposure)
        "realized_pnl": 100000,    // centi-cents
        "fees_paid": 10000,        // centi-cents
        "volume": 15
    }
}

IMPORTANT: position_cost is the CURRENT MARKET VALUE of the position (equivalent to
REST API's market_exposure), NOT the entry cost (total_traded). The total_traded field
(cost basis) only comes from REST API sync and is preserved via merge in state_container.

Note: All monetary values from Kalshi WebSocket are in centi-cents (1/10,000 of a dollar).
We convert to cents before emitting events.
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

logger = logging.getLogger("kalshiflow_rl.traderv3.position_listener")


class PositionListenerError(Exception):
    """Base exception for PositionListener errors."""
    pass


class PositionListenerAuthError(PositionListenerError):
    """Authentication error for position listener."""
    pass


class PositionListener:
    """
    WebSocket listener for Kalshi real-time position updates.

    Connects to Kalshi WebSocket, subscribes to the "market_positions" channel,
    and emits position update events via the event bus for immediate state updates.

    Features:
    - Automatic reconnection on disconnect
    - Authentication using KalshiAuth
    - Heartbeat monitoring
    - Clean shutdown handling
    - Centi-cents to cents conversion
    """

    def __init__(
        self,
        event_bus: "EventBus",
        ws_url: Optional[str] = None,
        reconnect_delay_seconds: float = 5.0,
        heartbeat_timeout_seconds: float = 30.0,
    ):
        """
        Initialize the PositionListener.

        Args:
            event_bus: EventBus instance for emitting position updates
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
        self._positions_received = 0
        self._positions_processed = 0
        self._connection_count = 0
        self._last_position_time: Optional[float] = None
        self._last_heartbeat_time: Optional[float] = None

        # Message counter for subscription IDs
        self._message_id = 0

        logger.info(f"PositionListener initialized (ws_url={self.ws_url})")

    async def start(self) -> None:
        """
        Start the position listener.

        Initializes authentication and starts the WebSocket listener task.
        """
        if self._running:
            logger.warning("PositionListener already running")
            return

        logger.info("Starting PositionListener...")

        # Initialize authentication
        await self._setup_auth()

        # Start listener task
        self._shutdown_requested = False
        self._running = True
        self._listener_task = asyncio.create_task(self._listener_loop())

        logger.info("✅ PositionListener started")

    async def stop(self) -> None:
        """
        Stop the position listener gracefully.
        """
        if not self._running:
            return

        logger.info("Stopping PositionListener...")

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

        logger.info("✅ PositionListener stopped")

    async def _setup_auth(self) -> None:
        """
        Set up authentication for WebSocket connection.

        Creates KalshiAuth instance using configured credentials.
        """
        import tempfile
        import os

        if not config.KALSHI_API_KEY_ID:
            raise PositionListenerAuthError("KALSHI_API_KEY_ID not configured")

        if not config.KALSHI_PRIVATE_KEY_CONTENT:
            raise PositionListenerAuthError("KALSHI_PRIVATE_KEY_CONTENT not configured")

        try:
            # Create temporary file for private key
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pem', prefix='position_listener_key_')
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

            logger.debug("PositionListener authentication initialized")

        except Exception as e:
            self._cleanup_auth()
            raise PositionListenerAuthError(f"Failed to initialize auth: {e}")

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
        Connect to WebSocket and listen for position messages.
        """
        if not self._auth:
            raise PositionListenerAuthError("Authentication not initialized")

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

                logger.info(f"✅ WebSocket connected (connection #{self._connection_count})")

                # Subscribe to market_positions channel
                await self._subscribe_to_positions()

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

    async def _subscribe_to_positions(self) -> None:
        """
        Subscribe to the market_positions channel.

        Sends subscription command to Kalshi WebSocket.
        """
        if not self._ws:
            return

        subscribe_msg = {
            "id": self._get_next_message_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["market_positions"]
            }
        }

        logger.info("Subscribing to 'market_positions' channel...")
        await self._ws.send(json.dumps(subscribe_msg))

    async def _handle_message(self, raw_message: str) -> None:
        """
        Handle incoming WebSocket message.

        Parses message and emits position events via event bus.

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
            logger.info(f"✅ Subscribed to '{channel}' channel")
            return

        # Handle heartbeat/ping
        if msg_type in ("heartbeat", "pong"):
            logger.debug("Heartbeat received")
            return

        # Handle position message
        if msg_type == "market_position":
            await self._handle_position_message(message)
            return

        # Handle error
        if msg_type == "error":
            error_msg = message.get("msg", {})
            logger.error(f"WebSocket error: {error_msg}")
            return

        # Log unknown message types at debug level
        logger.debug(f"Unknown message type: {msg_type}")

    async def _handle_position_message(self, message: Dict[str, Any]) -> None:
        """
        Handle a position message from Kalshi.

        Converts centi-cents to cents and emits via event bus.

        Args:
            message: Parsed position message from WebSocket
        """
        self._positions_received += 1
        self._last_position_time = time.time()

        # Extract position data from 'msg' key
        position_data = message.get("msg", {})

        if not position_data:
            logger.warning(f"Empty position data in message: {message}")
            return

        market_ticker = position_data.get("market_ticker", "")
        position = position_data.get("position", 0)

        # Convert centi-cents to cents (divide by 100)
        position_cost_centicents = position_data.get("position_cost", 0)
        realized_pnl_centicents = position_data.get("realized_pnl", 0)
        fees_paid_centicents = position_data.get("fees_paid", 0)

        # Debug: Log raw values BEFORE conversion for troubleshooting
        logger.debug(f"RAW from Kalshi: position_cost={position_cost_centicents}, ticker={market_ticker}")

        position_cost_cents = position_cost_centicents // 100
        realized_pnl_cents = realized_pnl_centicents // 100
        fees_paid_cents = fees_paid_centicents // 100

        # Sanity check: cost per contract should be $0.01-$1.00 (1-100 cents)
        if position != 0:
            cost_per_contract = position_cost_cents / abs(position)
            if cost_per_contract > 100:  # More than $1/contract is suspicious
                logger.warning(f"Suspicious cost/contract: {cost_per_contract}c for {market_ticker}")

        volume = position_data.get("volume", 0)

        logger.info(
            f"📊 Position update: market={market_ticker}, "
            f"position={position}, value={position_cost_cents}¢, "
            f"realized_pnl={realized_pnl_cents}¢"
        )

        # Emit via event bus
        try:
            # Create position data dict matching TraderState.positions format
            # NOTE: WebSocket position_cost represents CURRENT VALUE (same as REST's
            # market_exposure), NOT the entry cost (total_traded). We set market_exposure
            # here and let state_container merge preserve total_traded from REST sync.
            formatted_position = {
                "ticker": market_ticker,
                "position": position,
                "market_exposure": position_cost_cents,  # Current value from WebSocket
                # total_traded NOT set - preserved from REST sync via merge
                "realized_pnl": realized_pnl_cents,
                "fees_paid": fees_paid_cents,
                "volume": volume,
                "last_updated": time.time(),
            }

            await self._event_bus.emit_market_position_update(
                ticker=market_ticker,
                position_data=formatted_position
            )
            self._positions_processed += 1
            logger.debug(f"Position update emitted for {market_ticker}")

        except Exception as e:
            logger.error(f"Failed to emit position update for {market_ticker}: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get PositionListener metrics.

        Returns:
            Dictionary with listener statistics
        """
        return {
            "running": self._running,
            "connected": self._ws is not None and not self._ws.closed if self._ws else False,
            "positions_received": self._positions_received,
            "positions_processed": self._positions_processed,
            "connection_count": self._connection_count,
            "last_position_time": self._last_position_time,
            "last_heartbeat_time": self._last_heartbeat_time,
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get PositionListener status for monitoring.

        Returns:
            Status dictionary
        """
        return {
            "service": "PositionListener",
            "status": "running" if self._running else "stopped",
            "ws_url": self.ws_url,
            "metrics": self.get_metrics(),
        }

    def is_healthy(self) -> bool:
        """
        Check if position listener is healthy.

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
            "positions_received": metrics.get("positions_received", 0),
            "positions_processed": metrics.get("positions_processed", 0),
            "connection_count": metrics.get("connection_count", 0),
            "last_position_time": metrics.get("last_position_time"),
        }
