"""Unit tests for fair_value.py.

Tests microprice, VWAP, complement constraint, and combined FV estimation.
"""

import time

import pytest

from tests.traderv3.conftest import make_event_meta, make_market_meta

from kalshiflow_rl.traderv3.market_maker.fair_value import (
    _compute_complement_fv,
    _compute_vwap,
    estimate_all_fair_values,
    estimate_fair_value,
)


class TestEstimateFairValue:
    """Tests for estimate_fair_value()."""

    def test_returns_mid_when_no_signals(self):
        market = make_market_meta(yes_bid=40, yes_ask=50, bid_size=0, ask_size=0)
        # Microprice requires sizes > 0, so it falls back to mid
        # Use non-ME event with single market to avoid complement signal
        event = make_event_meta(n_markets=1, mutually_exclusive=False)
        event.markets = {market.ticker: market}
        fv = estimate_fair_value(market, event)
        assert fv is not None
        assert fv == 45.0  # midpoint

    def test_uses_microprice(self):
        market = make_market_meta(yes_bid=40, yes_ask=50, bid_size=30, ask_size=10)
        event = make_event_meta(n_markets=1, mutually_exclusive=False)
        event.markets[market.ticker] = market
        fv = estimate_fair_value(market, event)
        assert fv is not None
        # With more bid depth, microprice should be closer to ask
        assert fv > 45  # Should be pulled toward ask side

    def test_returns_none_when_no_data(self):
        market = make_market_meta(yes_bid=None, yes_ask=None)
        event = make_event_meta(n_markets=1, mutually_exclusive=False)
        event.markets[market.ticker] = market
        fv = estimate_fair_value(market, event)
        assert fv is None


class TestComputeVwap:
    """Tests for _compute_vwap()."""

    def test_no_trades(self):
        market = make_market_meta()
        assert _compute_vwap(market) is None

    def test_with_recent_trades(self):
        market = make_market_meta()
        market.recent_trades = [
            {"ts": time.time(), "yes_price": 50, "count": 10},
            {"ts": time.time(), "yes_price": 48, "count": 5},
        ]
        vwap = _compute_vwap(market)
        assert vwap is not None
        # VWAP = (50*10 + 48*5) / 15 = 740/15 ≈ 49.33
        assert vwap == pytest.approx(49.333, abs=0.01)

    def test_old_trades_excluded(self):
        market = make_market_meta()
        old_ts = time.time() - 600  # 10 min ago (beyond 5min window)
        market.recent_trades = [
            {"ts": old_ts, "yes_price": 30, "count": 100},
        ]
        assert _compute_vwap(market) is None


class TestComputeComplementFv:
    """Tests for _compute_complement_fv()."""

    def test_me_event_complement(self):
        # 3-market ME event: if others = 30c + 35c, complement = 100 - 65 = 35
        event = make_event_meta(
            event_ticker="EVT-1",
            n_markets=3,
            mutually_exclusive=True,
            market_prices=[
                {"yes_bid": 28, "yes_ask": 32},  # Target market
                {"yes_bid": 28, "yes_ask": 32},
                {"yes_bid": 33, "yes_ask": 37},
            ],
        )
        tickers = list(event.markets.keys())
        target = event.markets[tickers[0]]
        comp = _compute_complement_fv(target, event)
        assert comp is not None
        # Other markets: mid ~30 and ~35
        # Complement = 100 - 30 - 35 = 35
        assert 30 <= comp <= 45

    def test_returns_none_if_missing_data(self):
        # Create event where one other market has no data at all
        event = make_event_meta(
            n_markets=3,
            mutually_exclusive=True,
            market_prices=[
                {"yes_bid": 28, "yes_ask": 32},
                {"yes_bid": None, "yes_ask": None},  # No data
                {"yes_bid": 33, "yes_ask": 37},
            ],
        )
        tickers = list(event.markets.keys())
        target = event.markets[tickers[0]]
        comp = _compute_complement_fv(target, event)
        assert comp is None

    def test_clamped_to_valid_range(self):
        # Other markets sum to more than 100 → complement < 1, clamped to 1
        event = make_event_meta(
            n_markets=3,
            market_prices=[
                {"yes_bid": 28, "yes_ask": 32},  # Target
                {"yes_bid": 48, "yes_ask": 52},  # ~50
                {"yes_bid": 53, "yes_ask": 57},  # ~55
            ],
        )
        tickers = list(event.markets.keys())
        target = event.markets[tickers[0]]
        comp = _compute_complement_fv(target, event)
        # Others ~50 + ~55 = ~105 → complement = -5 → clamped to 1
        assert comp is not None
        assert comp >= 1.0


class TestEstimateAllFairValues:
    """Tests for estimate_all_fair_values()."""

    def test_returns_fv_for_all_markets(self):
        event = make_event_meta(n_markets=3)
        fvs = estimate_all_fair_values(event)
        assert len(fvs) == 3
        for ticker, fv in fvs.items():
            assert isinstance(fv, float)
            assert 1.0 <= fv <= 99.0
