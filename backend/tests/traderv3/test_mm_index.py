"""Unit tests for MMIndex.

Tests quote management, inventory tracking, fair value storage,
and snapshot generation. Pure T1 tests with minimal setup.
"""

from tests.traderv3.conftest import make_event_meta, make_index, make_market_meta

from kalshiflow_rl.traderv3.market_maker.index import MMIndex
from kalshiflow_rl.traderv3.market_maker.models import ActiveQuote


def _make_mm_index_with_event(event_ticker="EVT-1", n_markets=3):
    """Create an MMIndex pre-loaded with one event."""
    idx = MMIndex()
    event = make_event_meta(event_ticker=event_ticker, n_markets=n_markets)
    # Wire into the underlying arb index
    idx._arb_index._events[event.event_ticker] = event
    for ticker in event.markets:
        idx._arb_index._ticker_to_event[ticker] = event.event_ticker
        # Also init MM state
        from kalshiflow_rl.traderv3.market_maker.models import MarketInventory
        idx._inventory[ticker] = MarketInventory()
        idx._quotes[ticker] = {}
    return idx, event


class TestMMIndexQuotes:
    """Tests for quote management."""

    def test_set_and_get_quote(self):
        idx, event = _make_mm_index_with_event()
        ticker = list(event.markets.keys())[0]
        q = ActiveQuote(order_id="ord-1", side="yes", action="buy", price_cents=40, size=10)
        idx.set_quote(ticker, "bid", q)
        quotes = idx.get_quotes(ticker)
        assert "bid" in quotes
        assert quotes["bid"].order_id == "ord-1"

    def test_clear_quote(self):
        idx, event = _make_mm_index_with_event()
        ticker = list(event.markets.keys())[0]
        q = ActiveQuote(order_id="ord-1", price_cents=40, size=10)
        idx.set_quote(ticker, "bid", q)
        idx.clear_quote(ticker, "bid")
        assert "bid" not in idx.get_quotes(ticker)

    def test_clear_all_quotes_single_market(self):
        idx, event = _make_mm_index_with_event()
        ticker = list(event.markets.keys())[0]
        idx.set_quote(ticker, "bid", ActiveQuote(order_id="b1", price_cents=40, size=10))
        idx.set_quote(ticker, "ask", ActiveQuote(order_id="a1", price_cents=45, size=10))
        idx.clear_all_quotes(ticker)
        assert idx.get_quotes(ticker) == {}

    def test_clear_all_quotes_all_markets(self):
        idx, event = _make_mm_index_with_event()
        for ticker in event.markets:
            idx.set_quote(ticker, "bid", ActiveQuote(order_id=f"b-{ticker}", price_cents=40, size=5))
        idx.clear_all_quotes()
        for ticker in event.markets:
            assert idx.get_quotes(ticker) == {}

    def test_get_all_active_order_ids(self):
        idx, event = _make_mm_index_with_event()
        tickers = list(event.markets.keys())
        idx.set_quote(tickers[0], "bid", ActiveQuote(order_id="b1", price_cents=40, size=10))
        idx.set_quote(tickers[0], "ask", ActiveQuote(order_id="a1", price_cents=45, size=10))
        idx.set_quote(tickers[1], "bid", ActiveQuote(order_id="b2", price_cents=30, size=5))
        ids = idx.get_all_active_order_ids()
        assert set(ids) == {"b1", "a1", "b2"}

    def test_empty_order_ids_skipped(self):
        idx, event = _make_mm_index_with_event()
        ticker = list(event.markets.keys())[0]
        idx.set_quote(ticker, "bid", ActiveQuote())  # Empty order_id
        assert idx.get_all_active_order_ids() == []


class TestMMIndexInventory:
    """Tests for inventory tracking."""

    def test_get_inventory_creates_default(self):
        idx, event = _make_mm_index_with_event()
        inv = idx.get_inventory("nonexistent-ticker")
        assert inv.position == 0

    def test_record_fill_updates_inventory(self):
        idx, event = _make_mm_index_with_event()
        ticker = list(event.markets.keys())[0]
        idx.record_fill(ticker, "yes", "buy", 40, 10)
        inv = idx.get_inventory(ticker)
        assert inv.position == 10

    def test_total_event_exposure(self):
        idx, event = _make_mm_index_with_event()
        tickers = list(event.markets.keys())
        idx.record_fill(tickers[0], "yes", "buy", 40, 10)
        idx.record_fill(tickers[1], "yes", "sell", 60, 5)
        exposure = idx.total_event_exposure("EVT-1")
        assert exposure == 15  # abs(10) + abs(-5)

    def test_total_event_exposure_unknown_event(self):
        idx, _ = _make_mm_index_with_event()
        assert idx.total_event_exposure("NONEXISTENT") == 0

    def test_total_position_contracts(self):
        idx, event = _make_mm_index_with_event()
        tickers = list(event.markets.keys())
        idx.record_fill(tickers[0], "yes", "buy", 40, 10)
        idx.record_fill(tickers[1], "yes", "sell", 60, 5)
        assert idx.total_position_contracts() == 15

    def test_total_realized_pnl(self):
        idx, event = _make_mm_index_with_event()
        ticker = list(event.markets.keys())[0]
        idx.record_fill(ticker, "yes", "buy", 40, 10)
        idx.record_fill(ticker, "yes", "sell", 50, 10)
        assert idx.total_realized_pnl() > 0


class TestMMIndexFairValue:
    """Tests for fair value storage."""

    def test_set_and_get_fair_value(self):
        idx, _ = _make_mm_index_with_event()
        idx.set_fair_value("MKT-A", 42.5)
        assert idx.get_fair_value("MKT-A") == 42.5

    def test_get_fair_value_missing(self):
        idx, _ = _make_mm_index_with_event()
        assert idx.get_fair_value("nonexistent") is None


class TestMMIndexSnapshots:
    """Tests for snapshot generation."""

    def test_get_market_snapshot(self):
        idx, event = _make_mm_index_with_event()
        ticker = list(event.markets.keys())[0]
        idx.set_fair_value(ticker, 33.0)
        idx.set_quote(ticker, "bid", ActiveQuote(order_id="b1", price_cents=31, size=10))
        snap = idx.get_market_snapshot(ticker)
        assert snap is not None
        assert snap.ticker == ticker
        assert snap.fair_value == 33.0
        assert snap.our_bid_price == 31
        assert snap.our_bid_size == 10

    def test_get_market_snapshot_unknown_ticker(self):
        idx, _ = _make_mm_index_with_event()
        assert idx.get_market_snapshot("NOPE") is None

    def test_get_event_snapshot(self):
        idx, event = _make_mm_index_with_event()
        snap = idx.get_event_snapshot("EVT-1")
        assert snap is not None
        assert snap.event_ticker == "EVT-1"
        assert snap.market_count == 3
        assert len(snap.markets) == 3

    def test_get_event_snapshot_unknown(self):
        idx, _ = _make_mm_index_with_event()
        assert idx.get_event_snapshot("NOPE") is None

    def test_get_full_snapshot(self):
        idx, event = _make_mm_index_with_event()
        snap = idx.get_full_snapshot()
        assert "events" in snap
        assert "quote_state" in snap
        assert "total_events" in snap
        assert snap["total_events"] == 1
        assert snap["total_markets"] == 3


class TestMMIndexPassthrough:
    """Tests for market data pass-through to EventArbIndex."""

    def test_on_bbo_update(self):
        idx, event = _make_mm_index_with_event()
        ticker = list(event.markets.keys())[0]
        idx.on_bbo_update(ticker, 42, 48, 20, 15, source="test")
        market = event.markets[ticker]
        assert market.yes_bid == 42
        assert market.yes_ask == 48

    def test_events_property(self):
        idx, _ = _make_mm_index_with_event()
        assert "EVT-1" in idx.events

    def test_market_tickers_property(self):
        idx, event = _make_mm_index_with_event()
        assert len(idx.market_tickers) == 3

    def test_quote_state_property(self):
        idx, _ = _make_mm_index_with_event()
        assert idx.quote_state.active_quotes == 0
