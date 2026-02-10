"""Unit tests for EventArbIndex, EventMeta, and MarketMeta.

Pure T1 tests — no async, no mocking, no network. Exercises BBO computation,
microstructure signals, arb detection, and snapshot generation.
"""

import time

from tests.traderv3.conftest import make_event_meta, make_index, make_market_meta


# ===========================================================================
# TestMarketMeta
# ===========================================================================


class TestMarketMeta:
    """Tests for MarketMeta BBO, microprice, OFI, orderbook, and micro signals."""

    # ---- BBO basics ----

    def test_update_bbo_computes_mid(self):
        m = make_market_meta(yes_bid=40, yes_ask=50)
        assert m.yes_mid == 45.0

    def test_update_bbo_computes_spread(self):
        m = make_market_meta(yes_bid=40, yes_ask=50)
        assert m.spread == 10

    def test_update_bbo_only_bid(self):
        m = make_market_meta(yes_bid=40, yes_ask=None)
        assert m.yes_mid == 40.0
        assert m.spread is None

    def test_update_bbo_only_ask(self):
        m = make_market_meta(yes_bid=None, yes_ask=60)
        assert m.yes_mid == 60.0
        assert m.spread is None

    def test_update_bbo_neither(self):
        m = make_market_meta(yes_bid=None, yes_ask=None)
        assert m.yes_mid is None
        assert m.spread is None

    # ---- Microprice ----

    def test_microprice_balanced(self):
        m = make_market_meta(yes_bid=40, yes_ask=50, bid_size=10, ask_size=10)
        assert m.microprice == 45.0

    def test_microprice_skewed(self):
        m = make_market_meta(yes_bid=40, yes_ask=50, bid_size=10, ask_size=90)
        # imbalance = 10/100 = 0.1
        # microprice = 50 * 0.1 + 40 * 0.9 = 5 + 36 = 41
        assert m.microprice == 41.0

    # ---- OFI ----

    def test_ofi_bid_increases(self):
        m = make_market_meta(yes_bid=40, yes_ask=50, bid_size=10, ask_size=10)
        # Second update: bid rises from 40 to 42
        m.update_bbo(42, 50, 10, 10, source="test")
        assert m.micro.ofi > 0

    # ---- Orderbook ----

    def test_orderbook_extracts_bbo(self):
        m = make_market_meta(yes_bid=None, yes_ask=None)
        m.update_orderbook(yes_levels=[[40, 10]], no_levels=[[55, 10]])
        assert m.yes_bid == 40
        # YES ask = 100 - NO bid price = 100 - 55 = 45
        assert m.yes_ask == 45

    def test_book_imbalance(self):
        m = make_market_meta(yes_bid=None, yes_ask=None)
        m.update_orderbook(
            yes_levels=[[40, 100], [39, 50]],
            no_levels=[[55, 10]],
        )
        # bid_depth = 150, ask_depth = 10 -> imbalance = (150 - 10) / 160 > 0
        assert m.micro.book_imbalance > 0
        assert m.micro.total_bid_depth == 150
        assert m.micro.total_ask_depth == 10

    # ---- Microstructure: trades ----

    def test_whale_detection(self):
        m = make_market_meta()
        m.add_trade({"count": 100, "ts": time.time(), "taker_side": "yes", "yes_price": 40})
        assert m.micro.whale_trade_count == 1

    def test_trade_cadence(self):
        m = make_market_meta()
        now = time.time()
        m.add_trade({"count": 5, "ts": now, "taker_side": "yes", "yes_price": 40})
        m.add_trade({"count": 5, "ts": now + 0.5, "taker_side": "yes", "yes_price": 41})
        assert m.micro.avg_inter_trade_ms > 0

    def test_rapid_sequence(self):
        m = make_market_meta()
        now = time.time()
        # Two trades less than 100ms apart
        m.add_trade({"count": 5, "ts": now, "taker_side": "yes", "yes_price": 40})
        m.add_trade({"count": 5, "ts": now + 0.05, "taker_side": "yes", "yes_price": 41})
        assert m.micro.rapid_sequence_count >= 1

    def test_size_consistency(self):
        m = make_market_meta()
        now = time.time()
        for i in range(20):
            m.add_trade({"count": 10, "ts": now + i * 0.2, "taker_side": "yes", "yes_price": 40})
        # All same size => high consistency
        assert m.micro.consistent_size_ratio > 0.9

    def test_vpin_bucketing(self):
        m = make_market_meta()
        now = time.time()
        # Push 50 contracts (VPIN_BUCKET_SIZE) all on buy side => max toxicity bucket
        for i in range(50):
            m.add_trade({"count": 1, "ts": now + i * 0.01, "taker_side": "yes", "yes_price": 40})
        assert m.micro.vpin > 0

    # ---- has_data / to_dict ----

    def test_has_data_true(self):
        m = make_market_meta(yes_bid=40, yes_ask=45)
        assert m.has_data is True

    def test_has_data_false(self):
        m = make_market_meta(yes_bid=None, yes_ask=None)
        assert m.has_data is False

    def test_to_dict_includes_micro(self):
        m = make_market_meta()
        d = m.to_dict()
        assert "micro" in d
        assert "whale_trade_count" in d["micro"]


# ===========================================================================
# TestEventMeta
# ===========================================================================


class TestEventMeta:
    """Tests for EventMeta signal operators."""

    def test_market_sum_mid(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 30, "yes_ask": 34},
                {"yes_bid": 30, "yes_ask": 34},
                {"yes_bid": 30, "yes_ask": 34},
            ],
        )
        # mids: 32, 32, 32 => sum = 96
        assert event.market_sum() == 96.0

    def test_market_sum_bid(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 30, "yes_ask": 34},
                {"yes_bid": 35, "yes_ask": 40},
                {"yes_bid": 40, "yes_ask": 45},
            ],
        )
        assert event.market_sum_bid() == 105

    def test_market_sum_ask(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 30, "yes_ask": 34},
                {"yes_bid": 35, "yes_ask": 40},
                {"yes_bid": 40, "yes_ask": 45},
            ],
        )
        assert event.market_sum_ask() == 119

    def test_market_sum_partial(self):
        """Partial data: sum methods return partial sums, edges return None."""
        event = make_event_meta(
            n_markets=2,
            market_prices=[
                {"yes_bid": 40, "yes_ask": 50},
                {"yes_bid": None, "yes_ask": None},
            ],
        )
        assert event.market_sum() == 45.0
        assert event.market_sum_bid() == 40
        assert event.market_sum_ask() == 50
        assert event.long_edge() is None
        assert event.short_edge() is None
        assert event.deviation() is None

    def test_long_edge(self):
        # sum_ask = 94, fee = 1*3 = 3, long_edge = 100 - 94 - 3 = 3
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 28, "yes_ask": 30},
                {"yes_bid": 30, "yes_ask": 32},
                {"yes_bid": 30, "yes_ask": 32},
            ],
        )
        assert event.long_edge(fee_per_contract=1) == 100 - 94 - 3

    def test_short_edge(self):
        # sum_bid = 106, fee = 1*3 = 3, short_edge = 106 - 100 - 3 = 3
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 35, "yes_ask": 40},
                {"yes_bid": 36, "yes_ask": 41},
                {"yes_bid": 35, "yes_ask": 40},
            ],
        )
        assert event.short_edge(fee_per_contract=1) == 106 - 100 - 3

    def test_deviation(self):
        event = make_event_meta(
            n_markets=2,
            market_prices=[
                {"yes_bid": 40, "yes_ask": 50},
                {"yes_bid": 40, "yes_ask": 50},
            ],
        )
        # mids: 45, 45 => sum = 90 => deviation = 10
        assert event.deviation() == 10.0

    def test_widest_spread(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 40, "yes_ask": 42},   # spread=2
                {"yes_bid": 30, "yes_ask": 40},   # spread=10
                {"yes_bid": 20, "yes_ask": 25},   # spread=5
            ],
        )
        widest = event.widest_spread_market()
        assert widest is not None
        assert widest.spread == 10

    def test_most_active(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 40, "yes_ask": 45},
                {"yes_bid": 30, "yes_ask": 35},
                {"yes_bid": 20, "yes_ask": 25},
            ],
        )
        # Add trades to the second market to make it most active
        tickers = list(event.markets.keys())
        now = time.time()
        for i in range(5):
            event.markets[tickers[1]].add_trade(
                {"count": 1, "ts": now + i * 0.1, "taker_side": "yes", "yes_price": 30}
            )
        most = event.most_active_market()
        assert most is not None
        assert most.ticker == tickers[1]

    def test_event_type_mutually_exclusive(self):
        event = make_event_meta(mutually_exclusive=True)
        assert event.event_type == "mutually_exclusive"

    def test_event_type_independent(self):
        event = make_event_meta(mutually_exclusive=False)
        assert event.event_type == "independent"


# ===========================================================================
# TestEventArbIndex
# ===========================================================================


class TestEventArbIndex:
    """Tests for arb detection and index pipeline."""

    def test_detect_long_arb(self):
        # sum_ask = 90 (3 * 30), fee = 1*3 = 3, long_edge = 100 - 90 - 3 = 7 > min_edge(3)
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 28, "yes_ask": 30},
                {"yes_bid": 28, "yes_ask": 30},
                {"yes_bid": 28, "yes_ask": 30},
            ],
        )
        index = make_index(events=[event], fee=1, min_edge=3.0)
        # Trigger detection via internal method
        opp = index._detect_arb(event.event_ticker)
        assert opp is not None
        assert opp.direction == "long"
        assert opp.edge_after_fees == 7.0

    def test_detect_short_arb(self):
        # sum_bid = 110 (3 * ~37), fee = 1*3 = 3, short_edge = 110 - 100 - 3 = 7
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 37, "yes_ask": 40},
                {"yes_bid": 37, "yes_ask": 40},
                {"yes_bid": 36, "yes_ask": 40},
            ],
        )
        index = make_index(events=[event], fee=1, min_edge=3.0)
        opp = index._detect_arb(event.event_ticker)
        assert opp is not None
        assert opp.direction == "short"

    def test_no_arb_below_threshold(self):
        # sum_ask = 99, long_edge = 100 - 99 - 3 = -2 (below threshold)
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 31, "yes_ask": 33},
                {"yes_bid": 31, "yes_ask": 33},
                {"yes_bid": 31, "yes_ask": 33},
            ],
        )
        index = make_index(events=[event], fee=1, min_edge=3.0)
        opp = index._detect_arb(event.event_ticker)
        assert opp is None

    def test_no_arb_missing_data(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 28, "yes_ask": 30},
                {"yes_bid": 28, "yes_ask": 30},
                {"yes_bid": None, "yes_ask": None},  # missing data
            ],
        )
        index = make_index(events=[event], fee=1, min_edge=3.0)
        opp = index._detect_arb(event.event_ticker)
        assert opp is None

    def test_long_legs_structure(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 28, "yes_ask": 30},
                {"yes_bid": 28, "yes_ask": 30},
                {"yes_bid": 28, "yes_ask": 30},
            ],
        )
        index = make_index(events=[event], fee=1, min_edge=3.0)
        opp = index._detect_arb(event.event_ticker)
        assert opp is not None
        for leg in opp.legs:
            assert leg.side == "yes"
            assert leg.action == "buy"
            assert leg.price_cents == 30

    def test_short_legs_structure(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 37, "yes_ask": 40},
                {"yes_bid": 37, "yes_ask": 40},
                {"yes_bid": 36, "yes_ask": 40},
            ],
        )
        index = make_index(events=[event], fee=1, min_edge=3.0)
        opp = index._detect_arb(event.event_ticker)
        assert opp is not None
        for leg in opp.legs:
            assert leg.side == "no"
            assert leg.action == "buy"

    def test_on_orderbook_update_pipeline(self):
        # Start with an event where markets have data but no arb
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 31, "yes_ask": 33},
                {"yes_bid": 31, "yes_ask": 33},
                {"yes_bid": 31, "yes_ask": 33},
            ],
        )
        index = make_index(events=[event], fee=1, min_edge=3.0)
        ticker_a = list(event.markets.keys())[0]

        # Push market A's ask down to create a long arb
        # New sum_ask = 25 + 33 + 33 = 91, long_edge = 100 - 91 - 3 = 6
        opp = index.on_orderbook_update(
            ticker_a,
            yes_levels=[[23, 10]],
            no_levels=[[75, 10]],   # yes_ask = 100 - 75 = 25
        )
        assert opp is not None
        assert opp.direction == "long"

    def test_on_bbo_update_pipeline(self):
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 31, "yes_ask": 33},
                {"yes_bid": 31, "yes_ask": 33},
                {"yes_bid": 31, "yes_ask": 33},
            ],
        )
        index = make_index(events=[event], fee=1, min_edge=3.0)
        ticker_a = list(event.markets.keys())[0]

        # Push market A's ask down to create long arb
        # sum_ask = 25 + 33 + 33 = 91, long_edge = 100 - 91 - 3 = 6
        opp = index.on_bbo_update(ticker_a, yes_bid=23, yes_ask=25, bid_size=10, ask_size=10)
        assert opp is not None
        assert opp.direction == "long"

    def test_unknown_market_ticker(self):
        index = make_index(events=[], fee=1, min_edge=3.0)
        opp = index.on_bbo_update("UNKNOWN-TICKER", yes_bid=40, yes_ask=50)
        assert opp is None

    def test_get_snapshot(self):
        event = make_event_meta(n_markets=3)
        index = make_index(events=[event], fee=1, min_edge=3.0)
        snap = index.get_snapshot()
        assert "events" in snap
        assert event.event_ticker in snap["events"]
        assert snap["total_events"] == 1
        assert snap["total_markets"] == 3
