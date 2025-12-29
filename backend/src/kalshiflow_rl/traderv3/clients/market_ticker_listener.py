"""
MarketTickerListener - WebSocket listener for real-time market price updates.

This service connects to the Kalshi WebSocket and subscribes to the "ticker" channel
to receive real-time market price updates for position tickers. Updates are forwarded
to the event bus for state container integration.

Architecture:
    KalshiWS (ticker channel) -> MarketTickerListener -> EventBus -> StateContainer

Based on Kalshi Market Ticker documentation:
https://docs.kalshi.com/websockets/market-ticker

Ticker Message Format (from Kalshi):
{
    "type": "ticker",
    "sid": 1,
    "msg": {
        "market_ticker": "INXD-25JAN03",
        "price": 52,              // last traded price (cents, 1-99)
        "yes_bid": 50,            // best yes bid (cents)
        "yes_ask": 54,            // best yes ask (cents)
        "no_bid": 46,             // best no bid (cents)
        "no_ask": 50,             // best no ask (cents)
        "volume": 1500,           // total volume traded
        "open_interest": 12000,   // active contracts
        "ts": 1703808000          // unix timestamp (seconds)
    }
}

Key Features:
- Filtered subscription (only position tickers, not firehose)
- Dynamic subscription management (add/remove tickers as positions change)
- Throttled updates (configurable, default 500ms per ticker)
- Automatic reconnection on disconnect
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any, Optional, Set, List, TYPE_CHECKING

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from kalshiflow.auth import KalshiAuth
from ...config import config

if TYPE_CHECKING:
    from ..core.event_bus import EventBus

logger = logging.getLogger("kalshiflow_rl.traderv3.market_ticker_listener")


class MarketTickerListenerError(Exception):
    """Base exception for MarketTickerListener errors."""
    pass


class MarketTickerListenerAuthError(MarketTickerListenerError):
    """Authentication error for market ticker listener."""
    pass


class MarketTickerListener:
    """
    WebSocket listener for Kalshi real-time market ticker updates.

    Connects to Kalshi WebSocket, subscribes to the "ticker" channel for
    specific market tickers, and emits price update events via the event bus.

    Features:
    - Dynamic subscription management (add/remove tickers)
    - Throttled updates (configurable per-ticker)
    - Automatic reconnection on disconnect
    - Authentication using KalshiAuth
    - Clean shutdown handling
    """

    def __init__(
        self,
        event_bus: "EventBus",
        ws_url: Optional[str] = None,
        reconnect_delay_seconds: float = 5.0,
        throttle_ms: int = 500,
    ):
        """
        Initialize the MarketTickerListener.

        Args:
            event_bus: EventBus instance for emitting price updates
            ws_url: WebSocket URL (defaults to config.KALSHI_WS_URL)
            reconnect_delay_seconds: Delay before reconnection attempts
            throttle_ms: Minimum interval between updates per ticker (ms)
        """
        self._event_bus = event_bus
        self.ws_url = ws_url or config.KALSHI_WS_URL
        self.reconnect_delay = reconnect_delay_seconds
        self._throttle_seconds = throttle_ms / 1000

        # WebSocket state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._shutdown_requested = False
        self._listener_task: Optional[asyncio.Task] = None

        # Authentication
        self._auth: Optional[KalshiAuth] = None
        self._temp_key_file: Optional[str] = None

        # Subscription management
        self._subscribed_tickers: Set[str] = set()
        self._pending_subscriptions: Set[str] = set()
        self._pending_unsubscriptions: Set[str] = set()
        self._subscription_lock = asyncio.Lock()

        # Throttling - track last update time per ticker
        self._last_update_time: Dict[str, float] = {}

        # Metrics
        self._updates_received = 0
        self._updates_processed = 0
        self._updates_throttled = 0
        self._connection_count = 0
        self._last_update_time_global: Optional[float] = None
        self._per_ticker_stats: Dict[str, Dict[str, Any]] = {}

        # Message counter for subscription IDs
        self._message_id = 0

        logger.info(
            f"MarketTickerListener initialized (ws_url={self.ws_url}, "
            f"throttle_ms={throttle_ms})"
        )

    async def start(self) -> None:
        """
        Start the market ticker listener.

        Initializes authentication and starts the WebSocket listener task.
        """
        if self._running:
            logger.warning("MarketTickerListener already running")
            return

        logger.info("Starting MarketTickerListener...")

        # Initialize authentication
        await self._setup_auth()

        # Start listener task
        self._shutdown_requested = False
        self._running = True
        self._listener_task = asyncio.create_task(self._listener_loop())

        logger.info("MarketTickerListener started")

    async def stop(self) -> None:
        """
        Stop the market ticker listener gracefully.
        """
        if not self._running:
            return

        logger.info("Stopping MarketTickerListener...")

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

        logger.info("MarketTickerListener stopped")

    async def update_subscriptions(self, tickers: List[str]) -> None:
        """
        Update subscribed tickers.

        Computes diff and sends subscribe/unsubscribe commands as needed.
        Called by coordinator when positions change.

        Args:
            tickers: List of tickers to be subscribed (replaces current set)
        """
        async with self._subscription_lock:
            new_tickers = set(tickers)
            to_add = new_tickers - self._subscribed_tickers
            to_remove = self._subscribed_tickers - new_tickers

            if to_add:
                logger.info(f"Adding ticker subscriptions: {to_add}")
                self._pending_subscriptions.update(to_add)

            if to_remove:
                logger.info(f"Removing ticker subscriptions: {to_remove}")
                self._pending_unsubscriptions.update(to_remove)

            # If connected, process subscription changes immediately
            if self._ws:
                # Safe check for connection state (websockets library compatibility)
                ws_closed = False
                if hasattr(self._ws, 'closed'):
                    ws_closed = self._ws.closed
                elif hasattr(self._ws, 'close_code'):
                    ws_closed = self._ws.close_code is not None

                if not ws_closed:
                    await self._process_subscription_changes()

    async def _setup_auth(self) -> None:
        """
        Set up authentication for WebSocket connection.
        """
        import tempfile
        import os

        if not config.KALSHI_API_KEY_ID:
            raise MarketTickerListenerAuthError("KALSHI_API_KEY_ID not configured")

        if not config.KALSHI_PRIVATE_KEY_CONTENT:
            raise MarketTickerListenerAuthError("KALSHI_PRIVATE_KEY_CONTENT not configured")

        try:
            # Create temporary file for private key
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pem', prefix='market_ticker_key_')
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

            logger.debug("MarketTickerListener authentication initialized")

        except Exception as e:
            self._cleanup_auth()
            raise MarketTickerListenerAuthError(f"Failed to initialize auth: {e}")

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
        Connect to WebSocket and listen for ticker messages.
        """
        if not self._auth:
            raise MarketTickerListenerAuthError("Authentication not initialized")

        # Create auth headers for WebSocket
        ws_headers = self._auth.create_auth_headers("GET", "/trade-api/ws/v2")
        if 'Content-Type' in ws_headers:
            del ws_headers['Content-Type']

        logger.info(f"Connecting to WebSocket: {self.ws_url}")

        try:
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

                logger.info(f"WebSocket connected (connection #{self._connection_count})")

                # Subscribe to tickers if any pending
                await self._process_subscription_changes()

                # Also resubscribe to any previously subscribed tickers after reconnect
                if self._subscribed_tickers:
                    await self._subscribe_tickers(list(self._subscribed_tickers))

                # Listen for messages
                async for message in ws:
                    if self._shutdown_requested:
                        break

                    await self._handle_message(message)

            finally:
                await connection_manager.__aexit__(None, None, None)

        except asyncio.TimeoutError:
            logger.error(f"WebSocket connection timeout to {self.ws_url}")
            raise
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            raise

    async def _process_subscription_changes(self) -> None:
        """
        Process pending subscription additions and removals.
        """
        async with self._subscription_lock:
            # Process additions
            if self._pending_subscriptions:
                tickers_to_add = list(self._pending_subscriptions)
                self._pending_subscriptions.clear()
                await self._subscribe_tickers(tickers_to_add)
                self._subscribed_tickers.update(tickers_to_add)

            # Process removals
            if self._pending_unsubscriptions:
                tickers_to_remove = list(self._pending_unsubscriptions)
                self._pending_unsubscriptions.clear()
                await self._unsubscribe_tickers(tickers_to_remove)
                self._subscribed_tickers -= set(tickers_to_remove)
                # Clean up stats for removed tickers
                for ticker in tickers_to_remove:
                    self._last_update_time.pop(ticker, None)
                    self._per_ticker_stats.pop(ticker, None)

    async def _subscribe_tickers(self, tickers: List[str]) -> None:
        """
        Subscribe to ticker channel for specific tickers.

        Args:
            tickers: List of market tickers to subscribe to
        """
        if not self._ws or not tickers:
            return

        subscribe_msg = {
            "id": self._get_next_message_id(),
            "cmd": "subscribe",
            "params": {
                "channels": ["ticker"],
                "market_tickers": tickers
            }
        }

        logger.info(f"Subscribing to ticker channel for {len(tickers)} tickers: {tickers[:5]}...")
        await self._ws.send(json.dumps(subscribe_msg))

    async def _unsubscribe_tickers(self, tickers: List[str]) -> None:
        """
        Unsubscribe from ticker channel for specific tickers.

        Args:
            tickers: List of market tickers to unsubscribe from
        """
        if not self._ws or not tickers:
            return

        unsubscribe_msg = {
            "id": self._get_next_message_id(),
            "cmd": "unsubscribe",
            "params": {
                "channels": ["ticker"],
                "market_tickers": tickers
            }
        }

        logger.info(f"Unsubscribing from ticker channel for {len(tickers)} tickers")
        await self._ws.send(json.dumps(unsubscribe_msg))

    async def _handle_message(self, raw_message: str) -> None:
        """
        Handle incoming WebSocket message.

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

        # Handle unsubscription acknowledgment
        if msg_type == "unsubscribed":
            channel = message.get("msg", {}).get("channel", "unknown")
            logger.info(f"Unsubscribed from '{channel}' channel")
            return

        # Handle heartbeat/ping
        if msg_type in ("heartbeat", "pong"):
            logger.debug("Heartbeat received")
            return

        # Handle ticker message
        if msg_type == "ticker":
            await self._handle_ticker_message(message)
            return

        # Handle error
        if msg_type == "error":
            error_msg = message.get("msg", {})
            logger.error(f"WebSocket error: {error_msg}")
            return

        # Log unknown message types at debug level
        logger.debug(f"Unknown message type: {msg_type}")

    async def _handle_ticker_message(self, message: Dict[str, Any]) -> None:
        """
        Handle a ticker message from Kalshi.

        Applies throttling and emits via event bus if not throttled.

        Args:
            message: Parsed ticker message from WebSocket
        """
        self._updates_received += 1
        self._last_update_time_global = time.time()

        # Extract ticker data from 'msg' key
        ticker_data = message.get("msg", {})

        if not ticker_data:
            logger.warning(f"Empty ticker data in message: {message}")
            return

        market_ticker = ticker_data.get("market_ticker", "")

        if not market_ticker:
            logger.warning(f"No market_ticker in message: {message}")
            return

        # Check throttling
        now = time.time()
        last_update = self._last_update_time.get(market_ticker, 0)

        if now - last_update < self._throttle_seconds:
            self._updates_throttled += 1
            logger.debug(f"Throttled update for {market_ticker}")
            return

        # Update throttle tracking
        self._last_update_time[market_ticker] = now

        # Extract price data (all values are in cents, 1-99)
        last_price = ticker_data.get("price", 0)
        yes_bid = ticker_data.get("yes_bid", 0)
        yes_ask = ticker_data.get("yes_ask", 0)
        no_bid = ticker_data.get("no_bid", 0)
        no_ask = ticker_data.get("no_ask", 0)
        volume = ticker_data.get("volume", 0)
        open_interest = ticker_data.get("open_interest", 0)
        timestamp = ticker_data.get("ts", int(now))

        logger.debug(
            f"Ticker update: {market_ticker} "
            f"last={last_price}c bid/ask={yes_bid}/{yes_ask}c"
        )

        # Update per-ticker stats
        if market_ticker not in self._per_ticker_stats:
            self._per_ticker_stats[market_ticker] = {
                "updates": 0,
                "first_update": now,
            }
        self._per_ticker_stats[market_ticker]["updates"] += 1
        self._per_ticker_stats[market_ticker]["last_update"] = now
        self._per_ticker_stats[market_ticker]["last_price"] = last_price

        # Emit via event bus
        try:
            price_data = {
                "ticker": market_ticker,
                "last_price": last_price,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
                "volume": volume,
                "open_interest": open_interest,
                "timestamp": timestamp,
            }

            await self._event_bus.emit_market_ticker_update(
                ticker=market_ticker,
                price_data=price_data
            )
            self._updates_processed += 1

        except Exception as e:
            logger.error(f"Failed to emit ticker update for {market_ticker}: {e}")

    def get_subscribed_tickers(self) -> List[str]:
        """Get list of currently subscribed tickers."""
        return list(self._subscribed_tickers)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get MarketTickerListener metrics.

        Returns:
            Dictionary with listener statistics
        """
        connected = False
        if self._ws is not None:
            if hasattr(self._ws, 'closed'):
                connected = not self._ws.closed
            elif hasattr(self._ws, 'close_code'):
                connected = self._ws.close_code is None
            else:
                connected = True

        return {
            "running": self._running,
            "connected": connected,
            "subscribed_tickers": len(self._subscribed_tickers),
            "updates_received": self._updates_received,
            "updates_processed": self._updates_processed,
            "updates_throttled": self._updates_throttled,
            "connection_count": self._connection_count,
            "last_update_time": self._last_update_time_global,
            "throttle_ms": int(self._throttle_seconds * 1000),
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get MarketTickerListener status for monitoring.

        Returns:
            Status dictionary
        """
        return {
            "service": "MarketTickerListener",
            "status": "running" if self._running else "stopped",
            "ws_url": self.ws_url,
            "subscribed_tickers": list(self._subscribed_tickers),
            "metrics": self.get_metrics(),
            "per_ticker_stats": self._per_ticker_stats,
        }

    def is_healthy(self) -> bool:
        """
        Check if market ticker listener is healthy.

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
        Get detailed health information.

        Returns:
            Dictionary with health status and connection details
        """
        metrics = self.get_metrics()

        return {
            "running": self._running,
            "connected": metrics.get("connected", False),
            "ws_url": self.ws_url,
            "subscribed_tickers": metrics.get("subscribed_tickers", 0),
            "updates_received": metrics.get("updates_received", 0),
            "updates_processed": metrics.get("updates_processed", 0),
            "updates_throttled": metrics.get("updates_throttled", 0),
            "connection_count": metrics.get("connection_count", 0),
            "last_update_time": metrics.get("last_update_time"),
        }
