"""MMIndex - Per-market state index for market making.

Wraps the existing EventArbIndex (reusing MarketMeta/EventMeta) and layers
on MM-specific state: active quotes, inventory, fair values, quote engine state.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ..single_arb.index import EventArbIndex, EventMeta, MarketMeta
from . import fee_calculator
from .models import (
    ActiveQuote,
    MarketInventory,
    MMEventSnapshot,
    MMMarketSnapshot,
    QuoteConfig,
    QuoteState,
)

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.index")


class MMIndex:
    """Market maker index layering MM state on top of EventArbIndex.

    Holds:
      - EventArbIndex for raw market data (orderbook, trades, microstructure)
      - Per-market active quotes (bid/ask order IDs, prices, sizes)
      - Per-market inventory (position, P&L)
      - Per-market fair values (computed by FairValueEstimator)
      - Aggregate quote state telemetry
    """

    def __init__(self, fee_per_contract_cents: int = 1, min_edge_cents: float = 0.5):
        self._arb_index = EventArbIndex(
            fee_per_contract_cents=fee_per_contract_cents,
            min_edge_cents=min_edge_cents,
        )
        # MM-specific state, keyed by market_ticker
        self._quotes: Dict[str, Dict[str, ActiveQuote]] = {}  # ticker -> {"bid": .., "ask": ..}
        self._inventory: Dict[str, MarketInventory] = {}       # ticker -> MarketInventory
        self._fair_values: Dict[str, float] = {}               # ticker -> fair value cents
        self._quote_state = QuoteState()

    @property
    def arb_index(self) -> EventArbIndex:
        return self._arb_index

    @property
    def events(self) -> Dict[str, EventMeta]:
        return self._arb_index.events

    @property
    def quote_state(self) -> QuoteState:
        return self._quote_state

    @property
    def market_tickers(self) -> List[str]:
        return self._arb_index.market_tickers

    @property
    def is_ready(self) -> bool:
        return self._arb_index.is_ready

    def get_event_for_ticker(self, market_ticker: str) -> Optional[str]:
        return self._arb_index.get_event_for_ticker(market_ticker)

    async def load_event(self, event_ticker: str, trading_client) -> Optional[EventMeta]:
        meta = await self._arb_index.load_event(event_ticker, trading_client)
        if meta:
            for ticker in meta.markets:
                if ticker not in self._inventory:
                    self._inventory[ticker] = MarketInventory()
                if ticker not in self._quotes:
                    self._quotes[ticker] = {}
        return meta

    # ------------------------------------------------------------------
    # Market Data Pass-Through
    # ------------------------------------------------------------------

    def on_orderbook_update(
        self,
        market_ticker: str,
        yes_levels: List[List[int]],
        no_levels: List[List[int]],
        source: str = "ws",
    ) -> None:
        """Update orderbook (no arb detection needed for MM)."""
        self._arb_index.on_orderbook_update(market_ticker, yes_levels, no_levels, source)

    def on_bbo_update(
        self,
        market_ticker: str,
        yes_bid: Optional[int],
        yes_ask: Optional[int],
        bid_size: int = 0,
        ask_size: int = 0,
        source: str = "ws",
    ) -> None:
        self._arb_index.on_bbo_update(market_ticker, yes_bid, yes_ask, bid_size, ask_size, source)

    def on_ticker_update(
        self,
        market_ticker: str,
        price: Optional[int] = None,
        volume_delta: int = 0,
        oi_delta: int = 0,
    ) -> None:
        self._arb_index.on_ticker_update(market_ticker, price, volume_delta, oi_delta)

    def on_trade(self, market_ticker: str, trade_data: Dict) -> None:
        self._arb_index.on_trade(market_ticker, trade_data)

    # ------------------------------------------------------------------
    # Quote Management
    # ------------------------------------------------------------------

    def set_quote(self, market_ticker: str, side: str, quote: ActiveQuote) -> None:
        """Register an active quote for a market."""
        if market_ticker not in self._quotes:
            self._quotes[market_ticker] = {}
        self._quotes[market_ticker][side] = quote

    def clear_quote(self, market_ticker: str, side: str) -> None:
        """Remove a quote (after cancel or fill)."""
        if market_ticker in self._quotes:
            self._quotes[market_ticker].pop(side, None)

    def clear_all_quotes(self, market_ticker: Optional[str] = None) -> None:
        """Clear all quotes for a market, or all markets if None."""
        if market_ticker:
            self._quotes[market_ticker] = {}
        else:
            for ticker in self._quotes:
                self._quotes[ticker] = {}

    def get_quotes(self, market_ticker: str) -> Dict[str, ActiveQuote]:
        """Get active quotes for a market."""
        return self._quotes.get(market_ticker, {})

    def get_all_active_order_ids(self) -> List[str]:
        """Get all active order IDs across all markets."""
        ids = []
        for quotes in self._quotes.values():
            for q in quotes.values():
                if q.order_id:
                    ids.append(q.order_id)
        return ids

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def get_inventory(self, market_ticker: str) -> MarketInventory:
        if market_ticker not in self._inventory:
            self._inventory[market_ticker] = MarketInventory()
        return self._inventory[market_ticker]

    def record_fill(
        self, market_ticker: str, side: str, action: str, price_cents: int, quantity: int
    ) -> None:
        """Record a fill and update inventory."""
        inv = self.get_inventory(market_ticker)
        inv.record_fill(side, action, price_cents, quantity)

    def total_event_exposure(self, event_ticker: str) -> int:
        """Total absolute position across all markets in an event."""
        event = self.events.get(event_ticker)
        if not event:
            return 0
        total = 0
        for ticker in event.markets:
            inv = self._inventory.get(ticker)
            if inv:
                total += abs(inv.position)
        return total

    def total_position_contracts(self) -> int:
        """Total absolute position across all markets."""
        return sum(abs(inv.position) for inv in self._inventory.values())

    def total_realized_pnl(self) -> float:
        return sum(inv.realized_pnl_cents for inv in self._inventory.values())

    def total_unrealized_pnl(self) -> float:
        return sum(inv.unrealized_pnl_cents for inv in self._inventory.values())

    # ------------------------------------------------------------------
    # Fair Value
    # ------------------------------------------------------------------

    def set_fair_value(self, market_ticker: str, fair_value: float) -> None:
        self._fair_values[market_ticker] = fair_value

    def get_fair_value(self, market_ticker: str) -> Optional[float]:
        return self._fair_values.get(market_ticker)

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def get_market_snapshot(self, market_ticker: str) -> Optional[MMMarketSnapshot]:
        """Build MM-enriched snapshot for a single market."""
        event_ticker = self.get_event_for_ticker(market_ticker)
        if not event_ticker:
            return None
        event = self.events.get(event_ticker)
        if not event:
            return None
        market = event.markets.get(market_ticker)
        if not market:
            return None

        inv = self.get_inventory(market_ticker)
        quotes = self.get_quotes(market_ticker)
        fv = self.get_fair_value(market_ticker)
        bid_q = quotes.get("bid")
        ask_q = quotes.get("ask")

        mid = market.yes_mid or 50.0
        mf = fee_calculator.maker_fee(int(round(mid)))

        return MMMarketSnapshot(
            ticker=market.ticker,
            title=market.title,
            yes_bid=market.yes_bid,
            yes_ask=market.yes_ask,
            yes_bid_size=market.yes_bid_size,
            yes_ask_size=market.yes_ask_size,
            spread=market.spread,
            microprice=market.microprice,
            fair_value=fv,
            vpin=market.micro.vpin,
            book_imbalance=market.micro.book_imbalance,
            volume_5m=market.micro.volume_5m,
            our_bid_price=bid_q.price_cents if bid_q else None,
            our_bid_size=bid_q.size if bid_q else 0,
            our_bid_queue=bid_q.queue_position if bid_q else None,
            our_ask_price=ask_q.price_cents if ask_q else None,
            our_ask_size=ask_q.size if ask_q else 0,
            our_ask_queue=ask_q.queue_position if ask_q else None,
            position=inv.position,
            avg_entry_cents=inv.avg_entry_cents,
            unrealized_pnl_cents=inv.unrealized_pnl_cents,
            yes_levels=market.yes_levels,
            no_levels=market.no_levels,
            maker_fee_cents=round(mf, 4),
        )

    def get_event_snapshot(self, event_ticker: str) -> Optional[MMEventSnapshot]:
        """Build MM-enriched snapshot for an event."""
        event = self.events.get(event_ticker)
        if not event:
            return None

        markets = {}
        total_pos = 0
        total_unrealized = 0.0
        total_realized = 0.0

        for ticker in event.markets:
            snap = self.get_market_snapshot(ticker)
            if snap:
                markets[ticker] = snap
                inv = self.get_inventory(ticker)
                total_pos += abs(inv.position)
                total_unrealized += inv.unrealized_pnl_cents
                total_realized += inv.realized_pnl_cents

        return MMEventSnapshot(
            event_ticker=event.event_ticker,
            title=event.title,
            category=event.category,
            mutually_exclusive=event.mutually_exclusive,
            market_count=len(event.markets),
            markets=markets,
            total_position_contracts=total_pos,
            total_unrealized_pnl_cents=total_unrealized,
            total_realized_pnl_cents=total_realized,
        )

    def get_full_snapshot(self) -> Dict[str, Any]:
        """Full state for WS broadcast."""
        events = {}
        for et in self.events:
            snap = self.get_event_snapshot(et)
            if snap:
                events[et] = snap.model_dump()

        return {
            "events": events,
            "quote_state": self._quote_state.to_dict(),
            "total_events": len(self.events),
            "total_markets": len(self.market_tickers),
            "timestamp": time.time(),
        }
