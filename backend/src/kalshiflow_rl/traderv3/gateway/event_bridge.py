"""GatewayEventBridge - Translates gateway WS events into EventBus events.

Registers as a WS callback on the gateway's multiplexer, then publishes
translated events to the existing EventBus. This preserves the existing
pipeline powering microstructure signals, monitor, and index.

Channel -> EventType mapping:
    orderbook_delta (snapshot) -> ORDERBOOK_SNAPSHOT
    orderbook_delta (delta)   -> ORDERBOOK_DELTA
    trade                     -> PUBLIC_TRADE_RECEIVED
    ticker                    -> MARKET_TICKER_UPDATE
    fill                      -> ORDER_FILL
    market_positions          -> MARKET_POSITION_UPDATE
"""

import logging
import time
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from .ws_multiplexer import WSMultiplexer

logger = logging.getLogger("kalshiflow_rl.traderv3.gateway.event_bridge")


class GatewayEventBridge:
    """Bridges gateway WS messages to the existing EventBus.

    Registers callbacks on the WSMultiplexer for each channel and
    translates the messages into EventBus emit calls.
    """

    def __init__(self, event_bus: "EventBus", ws: "WSMultiplexer"):
        self._event_bus = event_bus
        self._ws = ws
        self._events_bridged = 0

    def wire(self) -> None:
        """Register all channel callbacks on the multiplexer."""
        self._ws.register_callback("orderbook_delta", self._on_orderbook)
        self._ws.register_callback("trade", self._on_trade)
        self._ws.register_callback("ticker", self._on_ticker)
        self._ws.register_callback("fill", self._on_fill)
        self._ws.register_callback("market_positions", self._on_position)
        logger.info("GatewayEventBridge wired to WSMultiplexer")

    # ------------------------------------------------------------------
    # Channel handlers
    # ------------------------------------------------------------------

    async def _on_orderbook(self, msg: Dict[str, Any]) -> None:
        """Handle orderbook_delta messages (both snapshots and deltas)."""
        market_ticker = msg.get("market_ticker", "")
        if not market_ticker:
            return

        msg_type = msg.get("type", "delta")
        seq = msg.get("seq", 0)
        ts = msg.get("ts", int(time.time() * 1000))

        metadata = {
            "sequence_number": seq,
            "timestamp_ms": ts,
            **msg,
        }

        if msg_type == "snapshot":
            await self._event_bus.emit_orderbook_snapshot(market_ticker, metadata)
        else:
            await self._event_bus.emit_orderbook_delta(market_ticker, metadata)

        self._events_bridged += 1

    async def _on_trade(self, msg: Dict[str, Any]) -> None:
        """Handle trade channel messages."""
        trade_data = {
            "market_ticker": msg.get("market_ticker", ""),
            "timestamp_ms": msg.get("ts", 0) * 1000 if msg.get("ts") else 0,
            "taker_side": msg.get("taker_side", ""),
            "yes_price": msg.get("yes_price", 0),
            "no_price": msg.get("no_price", 0),
            "count": msg.get("count", 0),
        }
        await self._event_bus.emit_public_trade(trade_data)
        self._events_bridged += 1

    async def _on_ticker(self, msg: Dict[str, Any]) -> None:
        """Handle ticker channel messages."""
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        price_data = {
            "price": msg.get("price", 0),
            "yes_bid": msg.get("yes_bid", 0),
            "yes_ask": msg.get("yes_ask", 0),
            "volume": msg.get("volume", 0),
            "open_interest": msg.get("open_interest", 0),
        }
        await self._event_bus.emit_market_ticker_update(ticker, price_data)
        self._events_bridged += 1

    async def _on_fill(self, msg: Dict[str, Any]) -> None:
        """Handle fill channel messages."""
        # Determine price in cents
        side = msg.get("side", "yes")
        if side == "yes":
            price_cents = msg.get("yes_price", 0)
        else:
            price_cents = msg.get("no_price", 0)

        await self._event_bus.emit_order_fill(
            trade_id=msg.get("trade_id", ""),
            order_id=msg.get("order_id", ""),
            market_ticker=msg.get("market_ticker", ""),
            is_taker=msg.get("is_taker", False),
            side=side,
            action=msg.get("action", ""),
            price_cents=price_cents,
            count=msg.get("count", 0),
            post_position=msg.get("post_position", 0),
            fill_timestamp=msg.get("ts", 0),
        )
        self._events_bridged += 1

    async def _on_position(self, msg: Dict[str, Any]) -> None:
        """Handle market_positions channel messages."""
        ticker = msg.get("market_ticker", "")
        if not ticker:
            return

        await self._event_bus.emit_market_position_update(ticker, msg)
        self._events_bridged += 1

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        return {"events_bridged": self._events_bridged}
