"""
Trade Flow Service - Bridges public trades to UI events.

Subscribes to the global PUBLIC_TRADE_RECEIVED stream, filters to tracked
markets, accumulates per-market state, and emits TRADE_FLOW_* events that
the WebSocket manager already handles for frontend delivery.
"""

import logging
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..state.tracked_markets import TrackedMarketsState

logger = logging.getLogger("kalshiflow_rl.traderv3.services.trade_flow_service")


class TradeFlowService:
    """
    Bridges PUBLIC_TRADE_RECEIVED events to TRADE_FLOW_* events for the UI.

    Subscribes to the global public trades stream, filters to tracked markets,
    accumulates per-market state (YES/NO counts, price movement), and emits
    events that the WebSocket manager already handles.

    This provides trade flow state, decoupling the UI trade feed
    from the strategy pipeline.
    """

    def __init__(
        self,
        event_bus: 'EventBus',
        tracked_markets: 'TrackedMarketsState',
    ):
        self._event_bus = event_bus
        self._tracked_markets = tracked_markets
        self._market_states: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._trades_processed = 0
        self._trades_filtered = 0
        self._started_at: Optional[float] = None

    async def start(self) -> None:
        """Start the trade flow service by subscribing to public trades."""
        self._running = True
        self._started_at = time.time()
        await self._event_bus.subscribe_to_public_trade(self._handle_trade)
        logger.info("TradeFlowService started - listening for public trades")

    async def stop(self) -> None:
        """Stop the trade flow service."""
        self._running = False
        logger.info(
            f"TradeFlowService stopped - processed {self._trades_processed} trades, "
            f"filtered {self._trades_filtered}"
        )

    async def _handle_trade(self, event) -> None:
        """
        Process a public trade event.

        Filters to tracked markets, accumulates per-market state, and emits
        trade flow events for the WebSocket manager to broadcast.
        """
        if not self._running:
            return

        ticker = event.market_ticker

        # Filter: only tracked markets
        if not self._tracked_markets.is_tracked(ticker):
            self._trades_filtered += 1
            return

        self._trades_processed += 1

        # Get event_ticker from tracked markets
        market = self._tracked_markets.get_market(ticker)
        event_ticker = market.event_ticker if market else ""

        # Get or create per-market state
        state = self._market_states.get(ticker)
        if not state:
            state = {
                "yes_trades": 0,
                "no_trades": 0,
                "total_trades": 0,
                "yes_ratio": 0.0,
                "first_yes_price": None,
                "last_yes_price": None,
                "price_drop": 0,
                "last_trade_time": None,
            }
            self._market_states[ticker] = state

        # Accumulate trade data
        side = event.side
        if side == "yes":
            state["yes_trades"] += event.count
            if state["first_yes_price"] is None:
                state["first_yes_price"] = event.price_cents
            state["last_yes_price"] = event.price_cents
        else:
            state["no_trades"] += event.count

        state["total_trades"] = state["yes_trades"] + state["no_trades"]
        total = state["total_trades"]
        state["yes_ratio"] = state["yes_trades"] / total if total > 0 else 0.0

        if state["first_yes_price"] is not None and state["last_yes_price"] is not None:
            state["price_drop"] = state["first_yes_price"] - state["last_yes_price"]

        state["last_trade_time"] = time.time()

        # Emit trade arrived (triggers pulse animations on market cards)
        await self._event_bus.emit_trade_flow_trade_arrived(
            market_ticker=ticker,
            side=side,
            count=event.count,
            yes_price=event.price_cents,
            event_ticker=event_ticker,
        )

        # Emit market state update
        await self._event_bus.emit_trade_flow_market_update(
            market_ticker=ticker,
            state={**state, "market_ticker": ticker, "event_ticker": event_ticker},
        )

    def get_market_states(self, limit: int = 100) -> list:
        """Return all market states for snapshot delivery to new clients."""
        states = []
        for ticker, state in list(self._market_states.items())[:limit]:
            states.append({"market_ticker": ticker, **state})
        return states

    def get_trade_processing_stats(self) -> Dict[str, Any]:
        """Stats for health reporting."""
        return {
            "trades_processed": self._trades_processed,
            "trades_filtered": self._trades_filtered,
            "tracked_markets_with_trades": len(self._market_states),
        }
