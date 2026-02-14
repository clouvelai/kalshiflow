"""MMMonitor - Bridges EventBus data into MMIndex for market making.

Subscribes to orderbook, ticker, and trade events. Updates MMIndex state.
Forwards signals to MMAttentionRouter for Captain reactive cycles.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional

from ..core.events.types import EventType
from ..core.events.market_events import OrderFillEvent

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.monitor")

STALE_THRESHOLD_SECONDS = 30.0


class MMMonitor:
    """Monitors market data for the market maker.

    Subscribes to EventBus, updates MMIndex, triggers attention signals.
    """

    def __init__(
        self,
        index,        # MMIndex
        event_bus,
        trading_client,
        config,
        attention_router=None,  # MMAttentionRouter
        broadcast_callback: Optional[Callable[..., Coroutine]] = None,
        quote_engine=None,  # QuoteEngine — for fill forwarding
    ):
        self._index = index
        self._event_bus = event_bus
        self._trading_client = trading_client
        self._config = config
        self._attention = attention_router
        self._broadcast = broadcast_callback
        self._quote_engine = quote_engine

        self._running = False
        self._poller_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None

        # Counters
        self._update_count = 0
        self._ticker_count = 0
        self._trade_count = 0
        self._poll_count = 0

        # Track previous spreads for change detection
        self._prev_spreads: Dict[str, int] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running = True

        self._event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)
        self._event_bus.subscribe(EventType.ORDERBOOK_DELTA, self._on_orderbook)
        self._event_bus.subscribe(EventType.TICKER_UPDATE, self._on_ticker)
        self._event_bus.subscribe(EventType.MARKET_TRADE, self._on_trade)

        # Subscribe to order fills for inventory + quote tracking
        if hasattr(self._event_bus, 'subscribe_to_order_fill'):
            await self._event_bus.subscribe_to_order_fill(self._on_order_fill)

        poll_interval = getattr(self._config, "mm_refresh_interval", 5.0)
        self._poller_task = asyncio.create_task(self._rest_poller_loop(poll_interval * 6))
        self._stats_task = asyncio.create_task(self._stats_broadcast_loop(5.0))

        logger.info(
            f"MMMonitor started: {len(self._index.market_tickers)} markets"
        )

    async def stop(self) -> None:
        self._running = False
        for task in [self._poller_task, self._stats_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        logger.info("MMMonitor stopped")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    async def _on_orderbook(self, market_ticker: str, metadata: Dict) -> None:
        """Handle orderbook snapshot/delta from EventBus.

        In hybrid mode (production REST + demo WS), the demo WS may send empty
        orderbook snapshots for markets that have real data on production.
        Skip empty updates to avoid overwriting production REST data.
        """
        if market_ticker not in self._index.market_tickers:
            return

        yes_levels = metadata.get("yes_levels", [])
        no_levels = metadata.get("no_levels", [])

        if yes_levels or no_levels:
            self._index.on_orderbook_update(market_ticker, yes_levels, no_levels, source="ws")
            self._update_count += 1
        else:
            # BBO-only update — only apply if we have actual data
            yes_bid = metadata.get("yes_bid")
            yes_ask = metadata.get("yes_ask")
            if yes_bid is not None or yes_ask is not None:
                bid_size = metadata.get("yes_bid_size", 0)
                ask_size = metadata.get("yes_ask_size", 0)
                self._index.on_bbo_update(market_ticker, yes_bid, yes_ask, bid_size, ask_size, source="ws")
                self._update_count += 1
            # else: empty WS snapshot — skip to preserve production REST data

        # Check for spread changes (signal to attention router)
        if self._attention:
            event_ticker = self._index.get_event_for_ticker(market_ticker)
            if event_ticker:
                event = self._index.events.get(event_ticker)
                market = event.markets.get(market_ticker) if event else None
                if market and market.spread is not None:
                    prev = self._prev_spreads.get(market_ticker)
                    if prev is not None and prev != market.spread:
                        self._attention.on_spread_change(
                            event_ticker, market_ticker, prev, market.spread
                        )
                    self._prev_spreads[market_ticker] = market.spread

                    # Check VPIN
                    if market.micro.vpin > 0:
                        threshold = getattr(self._config, "mm_refresh_interval", 0.95)
                        # Use QuoteConfig threshold if available
                        self._attention.on_vpin_spike(
                            event_ticker, market_ticker,
                            market.micro.vpin, 0.95,
                        )

        # Broadcast update
        if self._broadcast and self._update_count % 5 == 0:
            try:
                await self._broadcast("mm_market_update", {
                    "market_ticker": market_ticker,
                    "timestamp": time.time(),
                })
            except Exception:
                pass

    async def _on_ticker(self, market_ticker: str, metadata: Dict) -> None:
        if market_ticker not in self._index.market_tickers:
            return
        price = metadata.get("price")
        volume_delta = metadata.get("volume", 0)
        oi_delta = metadata.get("open_interest", 0)
        self._index.on_ticker_update(market_ticker, price, volume_delta, oi_delta)
        self._ticker_count += 1

    async def _on_trade(self, market_ticker: str, metadata: Dict) -> None:
        if market_ticker not in self._index.market_tickers:
            return
        self._index.on_trade(market_ticker, metadata)
        self._trade_count += 1

    async def _on_order_fill(self, event: OrderFillEvent) -> None:
        """Handle order fill — update inventory, quote engine, attention."""
        ticker = event.market_ticker
        if ticker not in self._index.market_tickers:
            return

        side = event.side.lower()    # "yes" or "no"
        action = event.action.lower()  # "buy" or "sell"
        price = event.price_cents
        count = event.count

        logger.info(
            f"[MM_FILL] {ticker}: {action} {count} {side} @ {price}c "
            f"(maker={not event.is_taker}, post_pos={event.post_position})"
        )

        # Forward to QuoteEngine (updates inventory, telemetry, clears filled quote)
        if self._quote_engine:
            self._quote_engine.on_fill(ticker, side, action, price, count)

        # Signal to attention router
        if self._attention:
            event_ticker = self._index.get_event_for_ticker(ticker) or ""
            self._attention.on_fill(event_ticker, ticker, side, action, price, count)

        # Broadcast fill event for trade log
        if self._broadcast:
            try:
                is_bid = (side == "yes" and action == "buy")
                # Convert ask fills (NO buys) to YES terms for frontend display
                display_price = price if is_bid else (100 - price)
                await self._broadcast("mm_quote_filled", {
                    "market_ticker": ticker,
                    "side": side,
                    "action": action,
                    "price_cents": display_price,
                    "count": count,
                    "is_taker": event.is_taker,
                    "order_id": event.order_id,
                    "quote_side": "bid" if is_bid else "ask",
                    "timestamp": event.timestamp,
                })
            except Exception:
                pass

    # ------------------------------------------------------------------
    # REST Fallback
    # ------------------------------------------------------------------

    async def _rest_poller_loop(self, interval: float) -> None:
        while self._running:
            try:
                await asyncio.sleep(interval)
                await self._poll_stale_markets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[MM_MONITOR] REST poller error: {e}")

    async def _poll_stale_markets(self) -> None:
        """Poll markets with stale WS data."""
        now = time.time()
        for et, event in self._index.events.items():
            for ticker, market in event.markets.items():
                if market.freshness_seconds > STALE_THRESHOLD_SECONDS:
                    try:
                        ob = await self._trading_client.get_orderbook(ticker)
                        if ob:
                            # ob.yes / ob.no are already List[List[int]] [[price, qty], ...]
                            self._index.on_orderbook_update(ticker, ob.yes or [], ob.no or [], source="api")
                            self._poll_count += 1
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def _stats_broadcast_loop(self, interval: float) -> None:
        while self._running:
            try:
                await asyncio.sleep(interval)
                if self._broadcast:
                    await self._broadcast("mm_stats", {
                        "orderbook_updates": self._update_count,
                        "ticker_updates": self._ticker_count,
                        "trades": self._trade_count,
                        "polls": self._poll_count,
                        "timestamp": time.time(),
                    })
            except asyncio.CancelledError:
                break
            except Exception:
                pass

    def get_stats(self) -> Dict:
        return {
            "orderbook_updates": self._update_count,
            "ticker_updates": self._ticker_count,
            "trades": self._trade_count,
            "polls": self._poll_count,
        }
