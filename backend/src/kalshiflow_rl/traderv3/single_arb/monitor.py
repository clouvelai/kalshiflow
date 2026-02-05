"""
EventArbMonitor - Bridges orderbook, ticker, and trade data into the EventArbIndex.

Four data paths:
1. WebSocket orderbook (primary): ORDERBOOK_SNAPSHOT/DELTA → full depth + BBO
2. WebSocket ticker_v2: TICKER_UPDATE → price, volume, OI deltas
3. WebSocket trade: MARKET_TRADE → public trade feed
4. REST Poller (fallback): Polls orderbooks via REST when WS data is stale
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

from ..core.events.types import EventType

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.monitor")

# Stale threshold: if WS data is older than this, trigger REST poll
STALE_THRESHOLD_SECONDS = 30.0


class EventArbMonitor:
    """
    Monitors orderbook, ticker, and trade events for all markets in the EventArbIndex.

    Subscribes to EventBus events, updates index with full orderbook depth,
    ticker price/volume data, and public trade feed.
    Runs a REST fallback poller for stale markets.
    Emits updates to frontend via WebSocket broadcast callback.
    """

    def __init__(
        self,
        index,  # EventArbIndex
        event_bus,
        trading_client,
        config,
        broadcast_callback: Optional[Callable[..., Coroutine]] = None,
        opportunity_callback: Optional[Callable[..., Coroutine]] = None,
    ):
        self._index = index
        self._event_bus = event_bus
        self._trading_client = trading_client
        self._config = config
        self._broadcast = broadcast_callback
        self._on_opportunity = opportunity_callback

        self._running = False
        self._poller_task: Optional[asyncio.Task] = None
        self._update_count = 0
        self._poll_count = 0
        self._ticker_count = 0
        self._trade_count = 0
        self._opportunities_detected = 0

    async def start(self) -> None:
        """Start monitoring: subscribe to events + start REST poller."""
        if self._running:
            return

        self._running = True

        # Subscribe to orderbook events
        self._event_bus.subscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)
        self._event_bus.subscribe(EventType.ORDERBOOK_DELTA, self._on_orderbook)

        # Subscribe to ticker_v2 events
        self._event_bus.subscribe(EventType.TICKER_UPDATE, self._on_ticker)

        # Subscribe to public trade events
        self._event_bus.subscribe(EventType.MARKET_TRADE, self._on_trade)

        # Start REST fallback poller
        poll_interval = getattr(self._config, "single_arb_poll_interval", 10.0)
        self._poller_task = asyncio.create_task(self._rest_poller_loop(poll_interval))

        logger.info(
            f"EventArbMonitor started: {len(self._index.market_tickers)} markets, "
            f"poll_interval={poll_interval}s (orderbook+ticker+trade channels)"
        )

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

        self._event_bus.unsubscribe(EventType.ORDERBOOK_SNAPSHOT, self._on_orderbook)
        self._event_bus.unsubscribe(EventType.ORDERBOOK_DELTA, self._on_orderbook)
        self._event_bus.unsubscribe(EventType.TICKER_UPDATE, self._on_ticker)
        self._event_bus.unsubscribe(EventType.MARKET_TRADE, self._on_trade)

        if self._poller_task:
            self._poller_task.cancel()
            try:
                await self._poller_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"EventArbMonitor stopped (updates={self._update_count}, "
            f"tickers={self._ticker_count}, trades={self._trade_count}, "
            f"polls={self._poll_count}, opportunities={self._opportunities_detected})"
        )

    async def _on_orderbook(self, market_ticker: str, metadata: Dict) -> None:
        """Handle orderbook snapshot/delta from EventBus - now with full depth."""
        if not self._running:
            return

        # Only process markets we're tracking
        if not self._index.get_event_for_ticker(market_ticker):
            return

        # Use full depth if available, otherwise fall back to BBO
        yes_levels = metadata.get("yes_levels")
        no_levels = metadata.get("no_levels")

        if yes_levels is not None and no_levels is not None:
            # Full depth path
            opportunity = self._index.on_orderbook_update(
                market_ticker=market_ticker,
                yes_levels=yes_levels,
                no_levels=no_levels,
                source="ws",
            )
        else:
            # BBO-only fallback
            yes_bid = metadata.get("yes_bid")
            yes_ask = metadata.get("yes_ask")
            bid_size = metadata.get("yes_bid_size", 0)
            ask_size = metadata.get("yes_ask_size", 0)
            opportunity = self._index.on_bbo_update(
                market_ticker=market_ticker,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                bid_size=bid_size,
                ask_size=ask_size,
                source="ws",
            )

        self._update_count += 1

        # Broadcast update to frontend
        event_ticker = self._index.get_event_for_ticker(market_ticker)
        if event_ticker:
            await self._broadcast_event_update(event_ticker)

        if opportunity:
            await self._handle_opportunity(opportunity)

    async def _on_ticker(self, market_ticker: str, metadata: Dict) -> None:
        """Handle ticker_v2 update from EventBus."""
        if not self._running:
            return

        if not self._index.get_event_for_ticker(market_ticker):
            return

        self._index.on_ticker_update(
            market_ticker=market_ticker,
            price=metadata.get("price"),
            volume_delta=metadata.get("volume_delta", 0),
            oi_delta=metadata.get("open_interest_delta", 0),
        )
        self._ticker_count += 1

        # Broadcast ticker update to frontend
        event_ticker = self._index.get_event_for_ticker(market_ticker)
        if event_ticker and self._broadcast:
            try:
                await self._broadcast({
                    "type": "event_arb_ticker",
                    "data": {
                        "event_ticker": event_ticker,
                        "market_ticker": market_ticker,
                        "price": metadata.get("price"),
                        "volume_delta": metadata.get("volume_delta", 0),
                        "open_interest_delta": metadata.get("open_interest_delta", 0),
                        "dollar_volume_delta": metadata.get("dollar_volume_delta", 0),
                        "ts": metadata.get("ts"),
                    },
                })
            except Exception as e:
                logger.debug(f"Ticker broadcast error: {e}")

    async def _on_trade(self, market_ticker: str, metadata: Dict) -> None:
        """Handle public trade from EventBus."""
        if not self._running:
            return

        if not self._index.get_event_for_ticker(market_ticker):
            return

        self._index.on_trade(market_ticker, metadata)
        self._trade_count += 1

        # Broadcast trade to frontend
        event_ticker = self._index.get_event_for_ticker(market_ticker)
        if event_ticker and self._broadcast:
            try:
                await self._broadcast({
                    "type": "event_arb_trade",
                    "data": {
                        "event_ticker": event_ticker,
                        "market_ticker": market_ticker,
                        "yes_price": metadata.get("yes_price"),
                        "no_price": metadata.get("no_price"),
                        "count": metadata.get("count", 0),
                        "taker_side": metadata.get("taker_side"),
                        "ts": metadata.get("ts"),
                    },
                })
            except Exception as e:
                logger.debug(f"Trade broadcast error: {e}")

    async def _rest_poller_loop(self, interval: float) -> None:
        """Periodically poll REST for stale markets."""
        while self._running:
            try:
                await asyncio.sleep(interval)
                if not self._running:
                    break
                await self._poll_stale_markets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"REST poller error: {e}")
                await asyncio.sleep(5.0)

    async def _poll_stale_markets(self) -> None:
        """Poll orderbooks for markets with stale WS data (depth=5)."""
        now = time.time()
        stale_tickers = []

        for event in self._index.events.values():
            for market in event.markets.values():
                if market.ws_updated_at == 0 or (now - market.ws_updated_at) > STALE_THRESHOLD_SECONDS:
                    stale_tickers.append(market.ticker)

        if not stale_tickers:
            return

        logger.debug(f"Polling {len(stale_tickers)} stale markets via REST (depth=5)")

        events_to_broadcast = set()

        for ticker in stale_tickers:
            try:
                resp = await self._trading_client.get_orderbook(ticker, depth=5)
                orderbook = resp.get("orderbook", resp)

                # Full depth levels from REST
                yes_levels = orderbook.get("yes", [])
                no_levels = orderbook.get("no", [])

                opportunity = self._index.on_orderbook_update(
                    market_ticker=ticker,
                    yes_levels=yes_levels,
                    no_levels=no_levels,
                    source="api",
                )

                self._poll_count += 1

                event_ticker = self._index.get_event_for_ticker(ticker)
                if event_ticker:
                    events_to_broadcast.add(event_ticker)

                if opportunity:
                    await self._handle_opportunity(opportunity)

            except Exception as e:
                logger.debug(f"REST poll failed for {ticker}: {e}")

        # Broadcast updates for all affected events
        for event_ticker in events_to_broadcast:
            await self._broadcast_event_update(event_ticker)

    async def _handle_opportunity(self, opportunity) -> None:
        """Handle detected arb opportunity."""
        self._opportunities_detected += 1
        logger.info(
            f"ARB OPPORTUNITY: {opportunity.event_ticker} "
            f"{opportunity.direction} edge={opportunity.edge_cents:.1f}c "
            f"(after fees: {opportunity.edge_after_fees:.1f}c, "
            f"{len(opportunity.legs)} legs)"
        )

        if self._on_opportunity:
            try:
                await self._on_opportunity(opportunity)
            except Exception as e:
                logger.error(f"Opportunity callback error: {e}")

    async def _broadcast_event_update(self, event_ticker: str) -> None:
        """Broadcast event arb state to frontend."""
        if not self._broadcast:
            return

        snapshot = self._index.get_event_snapshot(event_ticker)
        if snapshot:
            try:
                await self._broadcast({
                    "type": "event_arb_update",
                    "data": snapshot,
                })
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")

    def get_stats(self) -> Dict:
        """Get monitor stats."""
        return {
            "running": self._running,
            "update_count": self._update_count,
            "ticker_count": self._ticker_count,
            "trade_count": self._trade_count,
            "poll_count": self._poll_count,
            "opportunities_detected": self._opportunities_detected,
            "tracked_markets": len(self._index.market_tickers),
            "tracked_events": len(self._index.events),
        }
