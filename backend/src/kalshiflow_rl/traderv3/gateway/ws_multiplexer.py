"""Single multiplexed WebSocket connection for KalshiGateway.

Manages one WebSocket connection with multiple channel subscriptions
(trade, ticker, fill, market_positions, orderbook_delta). Supports
dynamic ticker subscription management and auto-reconnect.

Kalshi WS protocol:
- Auth via HTTP headers on connect
- Subscribe: {"id": N, "cmd": "subscribe", "params": {"channels": [...], "market_tickers": [...]}}
- Unsubscribe: {"id": N, "cmd": "unsubscribe", "params": {"channels": [...], "market_tickers": [...]}}
- Messages: {"type": "<channel>", "sid": N, "msg": {...}}
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed

from .errors import KalshiConnectionError

logger = logging.getLogger("kalshiflow_rl.traderv3.gateway.ws_multiplexer")

# Channels that require market_tickers in subscription
TICKER_CHANNELS = {"orderbook_delta", "trade", "ticker"}

# Channels that receive all account events (ignore market_tickers)
ACCOUNT_CHANNELS = {"fill", "market_positions"}


class WSMultiplexer:
    """Single WebSocket connection multiplexing multiple Kalshi channels.

    Features:
    - Per-channel async callbacks
    - Dynamic ticker subscribe/unsubscribe
    - Exponential backoff reconnect (1s base, 60s max)
    - Health monitoring (last message time, reconnect count)
    - Thread-safe channel/ticker management via asyncio
    """

    def __init__(
        self,
        ws_url: str,
        auth_headers_fn: Callable[[], Dict[str, str]],
        ping_interval: float = 20.0,
        ping_timeout: float = 10.0,
    ):
        """
        Args:
            ws_url: Kalshi WebSocket URL.
            auth_headers_fn: Callable returning auth headers dict for WS connect.
            ping_interval: Seconds between ping frames.
            ping_timeout: Seconds to wait for pong.
        """
        self._ws_url = ws_url
        self._auth_headers_fn = auth_headers_fn
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout

        # Channel state
        self._callbacks: Dict[str, Callable] = {}  # channel -> async callback
        self._subscribed_tickers: Dict[str, Set[str]] = {}  # channel -> {tickers}

        # Connection state
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._message_id = 0

        # Health metrics
        self._last_message_time: Optional[float] = None
        self._reconnect_count = 0
        self._messages_received = 0
        self._connected_at: Optional[float] = None

    def register_callback(self, channel: str, callback: Callable) -> None:
        """Register an async callback for a channel.

        Args:
            channel: Kalshi WS channel name (e.g. "fill", "trade", "orderbook_delta")
            callback: async def callback(msg: dict) -> None
        """
        self._callbacks[channel] = callback
        if channel in TICKER_CHANNELS and channel not in self._subscribed_tickers:
            self._subscribed_tickers[channel] = set()

    async def start(self) -> None:
        """Start the WS connection loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._connection_loop())
        logger.info("WSMultiplexer started")

    async def stop(self) -> None:
        """Gracefully stop the connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("WSMultiplexer stopped")

    async def subscribe_tickers(self, channel: str, tickers: List[str]) -> None:
        """Subscribe to market tickers on a channel.

        If already connected, sends subscribe command immediately.
        Otherwise, tickers are stored and subscribed on next connect.
        """
        if channel not in TICKER_CHANNELS:
            logger.warning(f"Channel {channel} does not use market_tickers")
            return

        if channel not in self._subscribed_tickers:
            self._subscribed_tickers[channel] = set()

        new_tickers = [t for t in tickers if t not in self._subscribed_tickers[channel]]
        if not new_tickers:
            return

        self._subscribed_tickers[channel].update(new_tickers)

        if self._ws:
            await self._send_subscribe(channel, new_tickers)

    async def unsubscribe_tickers(self, channel: str, tickers: List[str]) -> None:
        """Unsubscribe from market tickers on a channel."""
        if channel not in self._subscribed_tickers:
            return

        existing = [t for t in tickers if t in self._subscribed_tickers[channel]]
        if not existing:
            return

        self._subscribed_tickers[channel] -= set(existing)

        if self._ws:
            await self._send_unsubscribe(channel, existing)

    # ------------------------------------------------------------------
    # Connection loop
    # ------------------------------------------------------------------

    async def _connection_loop(self) -> None:
        """Main loop: connect, subscribe, listen, reconnect on failure."""
        backoff = 1.0

        while self._running:
            try:
                await self._connect_and_listen()
                backoff = 1.0  # reset on clean disconnect
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._running:
                    break
                self._reconnect_count += 1
                logger.warning(
                    f"WS disconnected (reconnect #{self._reconnect_count}): {e}. "
                    f"Reconnecting in {backoff:.0f}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    async def _connect_and_listen(self) -> None:
        """Single connection lifecycle: connect, subscribe all, listen."""
        headers = self._auth_headers_fn()
        # Remove Content-Type - not valid for WS
        headers.pop("Content-Type", None)

        try:
            ws = await asyncio.wait_for(
                websockets.connect(
                    self._ws_url,
                    additional_headers=headers,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                ).__aenter__(),
                timeout=10.0,
            )
        except asyncio.TimeoutError:
            raise KalshiConnectionError("WS connection timeout")

        self._ws = ws
        self._connected_at = time.time()
        logger.info("WS connected")

        try:
            # Re-subscribe all channels and tickers
            await self._resubscribe_all()

            async for raw in ws:
                if not self._running:
                    break
                self._last_message_time = time.time()
                self._messages_received += 1
                await self._dispatch(raw)
        except ConnectionClosed as e:
            logger.info(f"WS closed: {e}")
        finally:
            self._ws = None

    async def _resubscribe_all(self) -> None:
        """Re-subscribe all channels and tickers after reconnect."""
        # Account channels (no tickers needed)
        account_channels = [ch for ch in self._callbacks if ch in ACCOUNT_CHANNELS]
        if account_channels:
            await self._send_subscribe_channels(account_channels)

        # Ticker channels (with their tickers)
        for channel, tickers in self._subscribed_tickers.items():
            if channel in self._callbacks and tickers:
                await self._send_subscribe(channel, list(tickers))

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, raw: str) -> None:
        """Parse and dispatch a WS message to the appropriate callback."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug(f"Non-JSON WS message: {raw[:100]}")
            return

        msg_type = data.get("type", "")
        msg = data.get("msg", {})

        if not msg_type or not msg:
            return

        callback = self._callbacks.get(msg_type)
        if callback:
            try:
                await callback(msg)
            except Exception as e:
                logger.error(f"Callback error for {msg_type}: {e}")

    # ------------------------------------------------------------------
    # Subscribe / unsubscribe commands
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        self._message_id += 1
        return self._message_id

    async def _send_subscribe_channels(self, channels: List[str]) -> None:
        """Subscribe to channels without tickers (account-level)."""
        if not self._ws:
            return
        cmd = {
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": channels},
        }
        await self._ws.send(json.dumps(cmd))
        logger.debug(f"Subscribed channels: {channels}")

    async def _send_subscribe(self, channel: str, tickers: List[str]) -> None:
        """Subscribe to a channel with specific tickers."""
        if not self._ws:
            return
        cmd = {
            "id": self._next_id(),
            "cmd": "subscribe",
            "params": {"channels": [channel], "market_tickers": tickers},
        }
        await self._ws.send(json.dumps(cmd))
        logger.debug(f"Subscribed {channel}: {len(tickers)} tickers")

    async def _send_unsubscribe(self, channel: str, tickers: List[str]) -> None:
        """Unsubscribe from a channel with specific tickers."""
        if not self._ws:
            return
        cmd = {
            "id": self._next_id(),
            "cmd": "unsubscribe",
            "params": {"channels": [channel], "market_tickers": tickers},
        }
        await self._ws.send(json.dumps(cmd))
        logger.debug(f"Unsubscribed {channel}: {len(tickers)} tickers")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._ws is not None

    def get_health(self) -> Dict[str, Any]:
        """Health metrics for monitoring."""
        return {
            "connected": self.is_connected,
            "last_message_time": self._last_message_time,
            "reconnect_count": self._reconnect_count,
            "messages_received": self._messages_received,
            "connected_at": self._connected_at,
            "channels": list(self._callbacks.keys()),
            "ticker_subscriptions": {
                ch: len(tickers) for ch, tickers in self._subscribed_tickers.items()
            },
        }
