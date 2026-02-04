"""
WebSocket client for Polymarket CLOB market channel.

Replaces REST polling with real-time streaming of price updates.
Connection: wss://ws-subscriptions-clob.polymarket.com/ws/market

Handles price_change, book, and best_bid_ask message types.
Supports dynamic subscription when new pairs are added.
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed, InvalidMessage

from ..core.event_bus import EventBus
from ..core.events.types import EventType
from ..core.events.arb_events import PolyPriceEvent
from ..services.pair_registry import PairRegistry

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.polymarket_ws_client")

POLYMARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolymarketWSClient:
    """
    WebSocket client for Polymarket CLOB market channel.

    Streams real-time price updates for paired tokens and emits
    POLY_PRICE_UPDATE events via the EventBus.
    """

    def __init__(
        self,
        pair_registry: PairRegistry,
        event_bus: EventBus,
        on_price_callback: Optional[Callable] = None,
        max_reconnect_attempts: int = 10,
        base_reconnect_delay: float = 1.0,
    ):
        self._pair_registry = pair_registry
        self._event_bus = event_bus
        self._on_price_callback = on_price_callback
        self._max_reconnect_attempts = max_reconnect_attempts
        self._base_reconnect_delay = base_reconnect_delay

        # Connection state
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_count = 0
        self._task: Optional[asyncio.Task] = None

        # Subscribed token IDs
        self._subscribed_tokens: set = set()

        # Statistics
        self._messages_received = 0
        self._price_updates = 0
        self._connection_start_time: Optional[float] = None
        self._last_message_time: Optional[float] = None
        self._connection_established = asyncio.Event()

        # Last known prices per token
        self._last_prices: Dict[str, int] = {}

        # Last known BBO per token (for blending partial updates)
        self._last_bid: Dict[str, float] = {}
        self._last_ask: Dict[str, float] = {}

    async def start(self) -> None:
        """Start the WebSocket client."""
        if self._running:
            logger.warning("PolymarketWSClient already running")
            return

        self._running = True
        self._reconnect_count = 0
        self._task = asyncio.create_task(self._connection_loop())
        logger.info("PolymarketWSClient started")

    async def stop(self) -> None:
        """Stop the WebSocket client."""
        logger.info("Stopping PolymarketWSClient...")
        self._running = False

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
            self._websocket = None

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"PolymarketWSClient stopped. Stats: "
            f"messages={self._messages_received}, price_updates={self._price_updates}, "
            f"reconnects={self._reconnect_count}"
        )

    async def wait_for_connection(self, timeout: float = 30.0) -> bool:
        """Wait for WebSocket connection to be established."""
        try:
            await asyncio.wait_for(self._connection_established.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.error(f"PolymarketWSClient connection timeout after {timeout}s")
            return False

    async def subscribe_tokens(self, token_ids: List[str]) -> None:
        """Subscribe to additional token IDs on the live connection."""
        new_ids = [t for t in token_ids if t not in self._subscribed_tokens]
        if not new_ids:
            return

        if self._websocket:
            msg = json.dumps({
                "assets_ids": new_ids,
                "type": "market",
            })
            try:
                await self._websocket.send(msg)
                self._subscribed_tokens.update(new_ids)
                logger.info(f"Subscribed to {len(new_ids)} additional Polymarket tokens")
            except Exception as e:
                logger.warning(f"Failed to subscribe new tokens: {e}")
        else:
            # Queue for subscription on next connect
            self._subscribed_tokens.update(new_ids)

    async def unsubscribe_tokens(self, token_ids: List[str]) -> None:
        """Unsubscribe from token IDs."""
        ids_to_remove = [t for t in token_ids if t in self._subscribed_tokens]
        if not ids_to_remove:
            return

        for tid in ids_to_remove:
            self._subscribed_tokens.discard(tid)
        logger.info(f"Unsubscribed from {len(ids_to_remove)} Polymarket tokens")

    async def _connection_loop(self) -> None:
        """Main connection loop with automatic reconnection."""
        while self._running:
            try:
                await self._connect_and_subscribe()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Polymarket WS connection error: {e}")

            if self._running:
                delay = min(self._base_reconnect_delay * (2 ** self._reconnect_count), 60)
                logger.info(f"Polymarket WS reconnecting in {delay:.1f}s (attempt {self._reconnect_count + 1})")
                await asyncio.sleep(delay)
                self._reconnect_count += 1

                if self._reconnect_count >= self._max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts ({self._max_reconnect_attempts}) reached")
                    self._running = False
                    break

    async def _connect_and_subscribe(self) -> None:
        """Connect to Polymarket WS and subscribe to market channel."""
        logger.info(f"Connecting to Polymarket WebSocket: {POLYMARKET_WS_URL}")

        async with websockets.connect(
            POLYMARKET_WS_URL,
            ping_interval=25,
            ping_timeout=15,
            close_timeout=10,
            max_size=2**20,
        ) as websocket:
            self._websocket = websocket
            self._connection_start_time = time.time()
            self._reconnect_count = 0

            logger.info("Polymarket WebSocket connected")

            # Merge registry tokens + dynamically-subscribed tokens from before disconnect
            token_ids = self._pair_registry.get_poly_token_ids()
            self._subscribed_tokens.update(token_ids)

            if self._subscribed_tokens:
                sub_msg = json.dumps({
                    "assets_ids": list(self._subscribed_tokens),
                    "type": "market",
                })
                await websocket.send(sub_msg)
                logger.info(f"Subscribed to {len(self._subscribed_tokens)} Polymarket tokens")

            self._connection_established.set()

            # Process messages
            await self._message_loop()

        self._websocket = None
        self._connection_start_time = None
        self._connection_established.clear()

    async def _message_loop(self) -> None:
        """Process incoming WebSocket messages."""
        try:
            async for message in self._websocket:
                if not self._running:
                    break
                await self._process_message(message)
        except ConnectionClosed as e:
            logger.warning(f"Polymarket WebSocket closed: {e}")
            self._connection_established.clear()
        except InvalidMessage as e:
            logger.error(f"Invalid Polymarket WS message: {e}")
        except Exception as e:
            logger.error(f"Polymarket message loop error: {e}", exc_info=True)
            raise

    async def _process_message(self, raw_message: str) -> None:
        """Process a single WebSocket message."""
        try:
            recv_time = time.time()
            messages = json.loads(raw_message)
            self._messages_received += 1
            self._last_message_time = recv_time

            # Polymarket sends arrays of events
            if isinstance(messages, list):
                for msg in messages:
                    await self._handle_event(msg, recv_time)
            elif isinstance(messages, dict):
                await self._handle_event(messages, recv_time)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Polymarket message: {e}")
        except Exception as e:
            logger.error(f"Error processing Polymarket message: {e}", exc_info=True)

    async def _handle_event(self, event: Dict[str, Any], recv_time: float = 0.0) -> None:
        """Handle a single market event from Polymarket."""
        event_type = event.get("event_type", "")

        if event_type == "price_change":
            await self._handle_price_change(event, recv_time)
        elif event_type == "book":
            await self._handle_book_snapshot(event, recv_time)
        elif event_type in ("tick_size_change", "last_trade_price"):
            pass  # Ignored
        elif event_type:
            if self._messages_received % 100 == 0:
                logger.debug(f"Unhandled Polymarket event type: {event_type}")

    async def _handle_price_change(self, event: Dict[str, Any], recv_time: float = 0.0) -> None:
        """Handle price_change event (post-Sep 2025 migration format).

        Each event contains a price_changes array with per-asset BBO.
        See: docs.polymarket.com/developers/CLOB/websocket/market-channel
        """
        price_changes = event.get("price_changes", [])
        if not price_changes:
            return

        for pc in price_changes:
            asset_id = pc.get("asset_id", "")
            if not asset_id:
                continue

            pair = self._pair_registry.get_by_poly(asset_id)
            if not pair:
                continue

            # Polymarket provides BBO directly per change
            best_bid_str = pc.get("best_bid")
            best_ask_str = pc.get("best_ask")

            best_bid = float(best_bid_str) if best_bid_str else None
            best_ask = float(best_ask_str) if best_ask_str else None

            # Update stored BBO
            if best_bid is not None:
                self._last_bid[asset_id] = best_bid
            if best_ask is not None:
                self._last_ask[asset_id] = best_ask

            # Compute mid from stored BBO
            bid = self._last_bid.get(asset_id)
            ask = self._last_ask.get(asset_id)

            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
            elif bid is not None:
                mid = bid
            elif ask is not None:
                mid = ask
            else:
                continue

            yes_cents = max(0, min(100, round(mid * 100)))
            await self._emit_price(pair, asset_id, yes_cents, recv_time)

    async def _handle_book_snapshot(self, event: Dict[str, Any], recv_time: float = 0.0) -> None:
        """Handle book snapshot event - extract BBO and compute mid.

        Also resets stored BBO so subsequent price_change events blend
        against the snapshot baseline.
        """
        asset_id = event.get("asset_id", "")
        if not asset_id:
            return

        pair = self._pair_registry.get_by_poly(asset_id)
        if not pair:
            return

        bids = event.get("bids", [])
        asks = event.get("asks", [])

        best_bid = None
        best_ask = None

        if bids:
            try:
                bid_prices = [float(b["price"] if isinstance(b, dict) else b[0]) for b in bids]
                best_bid = max(bid_prices)
            except (KeyError, IndexError, TypeError, ValueError):
                pass

        if asks:
            try:
                ask_prices = [float(a["price"] if isinstance(a, dict) else a[0]) for a in asks]
                best_ask = min(ask_prices)
            except (KeyError, IndexError, TypeError, ValueError):
                pass

        # Store snapshot BBO as baseline for subsequent price_change blending
        if best_bid is not None:
            self._last_bid[asset_id] = best_bid
        if best_ask is not None:
            self._last_ask[asset_id] = best_ask

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2.0
            yes_cents = max(0, min(100, round(mid * 100)))
        elif best_bid is not None:
            yes_cents = max(0, min(100, round(best_bid * 100)))
        elif best_ask is not None:
            yes_cents = max(0, min(100, round(best_ask * 100)))
        else:
            return

        await self._emit_price(pair, asset_id, yes_cents, recv_time)

    async def _emit_price(self, pair, token_id: str, yes_cents: int, recv_time: float = 0.0) -> None:
        """Emit a POLY_PRICE_UPDATE event and call optional callback."""
        no_cents = 100 - yes_cents
        self._price_updates += 1
        self._last_prices[token_id] = yes_cents

        latency_ms = (time.time() - recv_time) * 1000 if recv_time else None

        event = PolyPriceEvent(
            pair_id=pair.id,
            kalshi_ticker=pair.kalshi_ticker,
            poly_token_id=token_id,
            poly_yes_cents=yes_cents,
            poly_no_cents=no_cents,
            source="ws",
            latency_ms=latency_ms,
        )
        await self._event_bus.emit(EventType.POLY_PRICE_UPDATE, event)

        if self._on_price_callback:
            try:
                if asyncio.iscoroutinefunction(self._on_price_callback):
                    await self._on_price_callback(event)
                else:
                    self._on_price_callback(event)
            except Exception as e:
                logger.warning(f"Price callback error: {e}")

        if self._price_updates % 500 == 0:
            logger.info(f"Polymarket WS price updates: {self._price_updates}")

    def is_healthy(self) -> bool:
        """Check if client is healthy."""
        if not self._running:
            return False
        if not self._websocket:
            return False

        try:
            if hasattr(self._websocket, 'closed') and self._websocket.closed:
                return False
        except (AttributeError, TypeError):
            pass

        if self._last_message_time:
            if (time.time() - self._last_message_time) > 300:
                return False
        elif self._connection_start_time:
            if (time.time() - self._connection_start_time) > 60:
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
            "price_updates": self._price_updates,
            "tokens_subscribed": len(self._subscribed_tokens),
            "uptime_seconds": uptime,
            "last_message_age_seconds": message_age,
        }

    def get_status(self) -> Dict[str, Any]:
        """Get status for health/status endpoints (compatible with PolymarketPoller interface)."""
        return {
            "running": self._running,
            "connected": self._websocket is not None,
            "healthy": self.is_healthy(),
            "price_updates": self._price_updates,
            "tokens_tracked": len(self._subscribed_tokens),
            "messages_received": self._messages_received,
            "reconnect_count": self._reconnect_count,
            **({
                "last_message_age": round(time.time() - self._last_message_time, 1)
            } if self._last_message_time else {}),
        }
