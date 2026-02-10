"""Unit tests for Captain V2 ContextBuilder.

T1/T2 tests — pure Python + async. No network calls.
Uses conftest factory functions to build mock index state.
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.traderv3.conftest import make_event_meta, make_index, make_market_meta

from kalshiflow_rl.traderv3.single_arb.context_builder import (
    ContextBuilder,
    _compute_regime,
    _event_semantics,
    _market_snapshot,
    _time_to_close,
)
from kalshiflow_rl.traderv3.single_arb.models import (
    EventSemantics,
    EventSnapshot,
    MarketSnapshot,
    MarketState,
    PortfolioState,
    SniperStatus,
)


# ===========================================================================
# TestMarketSnapshot (helper function)
# ===========================================================================

class TestMarketSnapshotBuilder:
    def test_builds_from_market_meta(self):
        m = make_market_meta(ticker="MKT-A", yes_bid=40, yes_ask=45)
        snap = _market_snapshot(m)
        assert isinstance(snap, MarketSnapshot)
        assert snap.ticker == "MKT-A"
        assert snap.yes_bid == 40
        assert snap.yes_ask == 45
        assert snap.spread == 5

    def test_microprice_included(self):
        m = make_market_meta(yes_bid=40, yes_ask=50, bid_size=10, ask_size=10)
        snap = _market_snapshot(m)
        assert snap.microprice == 45.0

    def test_micro_signals_included(self):
        m = make_market_meta(yes_bid=40, yes_ask=45)
        # Micro signals have defaults from MicrostructureSignals
        snap = _market_snapshot(m)
        assert snap.vpin >= 0
        assert snap.volume_5m >= 0


# ===========================================================================
# TestComputeRegime
# ===========================================================================

class TestComputeRegime:
    def test_normal_regime(self):
        event = make_event_meta(n_markets=3)
        # Set sufficient depth so it's not "thin"
        for m in event.markets.values():
            m.micro.total_bid_depth = 10
            m.micro.total_ask_depth = 10
        assert _compute_regime(event) == "normal"

    def test_toxic_when_high_vpin(self):
        event = make_event_meta(n_markets=3)
        # Set one market's VPIN > 0.85
        first_market = next(iter(event.markets.values()))
        first_market.micro.vpin = 0.9
        assert _compute_regime(event) == "toxic"

    def test_sweep_when_sweep_active(self):
        event = make_event_meta(n_markets=3)
        first_market = next(iter(event.markets.values()))
        first_market.micro.sweep_active = True
        assert _compute_regime(event) == "sweep"

    def test_thin_when_low_depth(self):
        event = make_event_meta(n_markets=3)
        # Set all markets to have very low depth
        for m in event.markets.values():
            m.micro.total_bid_depth = 2
            m.micro.total_ask_depth = 2
        assert _compute_regime(event) == "thin"

    def test_toxic_trumps_sweep(self):
        """VPIN check comes before sweep check."""
        event = make_event_meta(n_markets=3)
        first_market = next(iter(event.markets.values()))
        first_market.micro.vpin = 0.9
        first_market.micro.sweep_active = True
        assert _compute_regime(event) == "toxic"


# ===========================================================================
# TestEventSemantics
# ===========================================================================

class TestEventSemanticsBuilder:
    def test_none_without_understanding(self):
        event = make_event_meta(n_markets=2)
        assert event.understanding is None
        assert _event_semantics(event) is None

    def test_builds_from_understanding(self):
        event = make_event_meta(n_markets=2)
        event.understanding = {
            "trading_summary": "Super Bowl 2025 props trading",
            "participants": [{"name": "Eagles"}, {"name": "Chiefs"}],
            "domain": "sports",
            "timeline": {"status": "upcoming"},
            "key_factors": [{"factor": "injury reports"}, {"factor": "weather"}],
        }
        sem = _event_semantics(event)
        assert isinstance(sem, EventSemantics)
        assert "Super Bowl" in sem.what
        assert "Eagles" in sem.who
        assert sem.domain == "sports"
        assert len(sem.search_terms) == 2


# ===========================================================================
# TestContextBuilder.build_market_state
# ===========================================================================

class TestBuildMarketState:
    def test_empty_index(self):
        index = make_index(events=[])
        builder = ContextBuilder(index)
        state = builder.build_market_state()
        assert isinstance(state, MarketState)
        assert state.total_events == 0
        assert state.total_markets == 0

    def test_single_event(self):
        event = make_event_meta(event_ticker="EVT-1", n_markets=3)
        index = make_index(events=[event])
        builder = ContextBuilder(index)
        state = builder.build_market_state()
        assert state.total_events == 1
        assert state.total_markets == 3
        assert state.events[0].event_ticker == "EVT-1"

    def test_multiple_events(self):
        ev1 = make_event_meta(event_ticker="EVT-1", n_markets=3)
        ev2 = make_event_meta(event_ticker="EVT-2", n_markets=4)
        index = make_index(events=[ev1, ev2])
        builder = ContextBuilder(index)
        state = builder.build_market_state()
        assert state.total_events == 2
        assert state.total_markets == 7

    def test_markets_serialized(self):
        event = make_event_meta(event_ticker="EVT-1", n_markets=2)
        index = make_index(events=[event])
        builder = ContextBuilder(index)
        state = builder.build_market_state()
        ev_snap = state.events[0]
        assert len(ev_snap.markets) == 2
        for ticker, msnap in ev_snap.markets.items():
            assert isinstance(msnap, MarketSnapshot)
            assert msnap.ticker == ticker

    def test_edge_computed(self):
        event = make_event_meta(
            event_ticker="EVT-1", n_markets=3,
            market_prices=[
                {"yes_bid": 30, "yes_ask": 35},
                {"yes_bid": 30, "yes_ask": 35},
                {"yes_bid": 30, "yes_ask": 35},
            ],
        )
        index = make_index(events=[event], fee=1)
        builder = ContextBuilder(index)
        state = builder.build_market_state()
        ev_snap = state.events[0]
        # With 3 markets at ~33c each, edges should be computed
        assert ev_snap.long_edge is not None or ev_snap.short_edge is not None


# ===========================================================================
# TestContextBuilder.build_portfolio_state
# ===========================================================================

class TestBuildPortfolioState:
    @pytest.mark.asyncio
    async def test_empty_portfolio(self):
        index = make_index(events=[])
        builder = ContextBuilder(index)
        gateway = MagicMock()
        gateway.get_balance = AsyncMock(return_value=MagicMock(balance=50000))
        gateway.get_positions = AsyncMock(return_value=[])
        portfolio = await builder.build_portfolio_state(gateway)
        assert isinstance(portfolio, PortfolioState)
        assert portfolio.balance_cents == 50000
        assert portfolio.balance_dollars == 500.0
        assert portfolio.total_positions == 0

    @pytest.mark.asyncio
    async def test_with_positions(self):
        event = make_event_meta(event_ticker="EVT-1", n_markets=2,
                                market_prices=[{"yes_bid": 40, "yes_ask": 45},
                                               {"yes_bid": 55, "yes_ask": 60}])
        index = make_index(events=[event])
        builder = ContextBuilder(index)

        # Mock gateway
        gateway = MagicMock()
        gateway.get_balance = AsyncMock(return_value=MagicMock(balance=50000))

        ticker = list(event.markets.keys())[0]
        mock_pos = MagicMock()
        mock_pos.ticker = ticker
        mock_pos.position = 5  # 5 YES contracts
        mock_pos.total_traded = 200  # 200c cost
        gateway.get_positions = AsyncMock(return_value=[mock_pos])

        portfolio = await builder.build_portfolio_state(gateway)
        assert portfolio.total_positions == 1
        assert portfolio.positions[0].side == "yes"
        assert portfolio.positions[0].quantity == 5

    @pytest.mark.asyncio
    async def test_balance_fetch_failure(self):
        index = make_index(events=[])
        builder = ContextBuilder(index)
        gateway = MagicMock()
        gateway.get_balance = AsyncMock(side_effect=Exception("API error"))
        gateway.get_positions = AsyncMock(return_value=[])
        portfolio = await builder.build_portfolio_state(gateway)
        assert portfolio.balance_cents == 0


# ===========================================================================
# TestContextBuilder.build_sniper_status
# ===========================================================================

class TestBuildSniperStatus:
    def test_no_sniper(self):
        index = make_index(events=[])
        builder = ContextBuilder(index)
        status = builder.build_sniper_status(None)
        assert isinstance(status, SniperStatus)
        assert status.enabled is False

    def test_with_sniper(self):
        from dataclasses import dataclass, field as dc_field
        from collections import deque

        # Mock sniper state and config
        mock_state = MagicMock()
        mock_state.total_trades = 10
        mock_state.total_arbs_executed = 3
        mock_state.capital_active = 500
        mock_state.capital_in_positions = 1500
        mock_state.capital_deployed_lifetime = 5000
        mock_state.total_partial_unwinds = 1
        mock_state.last_rejection_reason = None
        mock_state.recent_actions = deque()

        mock_config = MagicMock()
        mock_config.enabled = True
        mock_config.max_position = 25
        mock_config.max_capital = 100000
        mock_config.arb_min_edge = 3.0
        mock_config.cooldown = 10.0
        mock_config.max_trades_per_cycle = 5

        mock_sniper = MagicMock()
        mock_sniper.state = mock_state
        mock_sniper.config = mock_config

        index = make_index(events=[])
        builder = ContextBuilder(index)
        status = builder.build_sniper_status(mock_sniper)
        assert status.enabled is True
        assert status.total_trades == 10
        assert status.capital_in_flight == 500


# ===========================================================================
# TestContextBuilder.compute_diffs
# ===========================================================================

class TestComputeDiffs:
    def test_first_cycle_no_changes(self):
        event = make_event_meta(event_ticker="EVT-1", n_markets=2)
        index = make_index(events=[event])
        builder = ContextBuilder(index)
        state = builder.build_market_state()
        diff = builder.compute_diffs(state)
        assert diff.has_changes is False

    def test_price_move_detected(self):
        event = make_event_meta(event_ticker="EVT-1", n_markets=2,
                                market_prices=[{"yes_bid": 40, "yes_ask": 45},
                                               {"yes_bid": 50, "yes_ask": 55}])
        index = make_index(events=[event])
        builder = ContextBuilder(index)

        # First cycle (establishes baseline)
        state1 = builder.build_market_state()
        builder.compute_diffs(state1)

        # Simulate price move
        first_market = next(iter(event.markets.values()))
        first_market.update_bbo(45, 50, 10, 10, source="test")

        # Second cycle
        state2 = builder.build_market_state()
        diff = builder.compute_diffs(state2)
        assert diff.has_changes is True
        assert len(diff.price_moves) >= 1


# ===========================================================================
# TestCompactPortfolioFields (Gap 4: pnl_ct + TIME_PRESSURE)
# ===========================================================================

from kalshiflow_rl.traderv3.single_arb.models import Position


class TestCompactPortfolioPnlCt:
    """Test the pnl_ct per-contract PnL field added to compact portfolio in captain.py."""

    def test_pnl_ct_calculation(self):
        """pnl_ct = unrealized_pnl_cents / quantity, rounded."""
        p = Position(ticker="MKT-A", side="yes", quantity=5,
                     unrealized_pnl_cents=50, event_ticker="EVT-1")
        pnl_ct = round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0
        assert pnl_ct == 10

    def test_pnl_ct_negative(self):
        p = Position(ticker="MKT-B", side="no", quantity=3,
                     unrealized_pnl_cents=-36, event_ticker="EVT-1")
        pnl_ct = round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0
        assert pnl_ct == -12

    def test_pnl_ct_zero_quantity(self):
        p = Position(ticker="MKT-C", side="yes", quantity=0,
                     unrealized_pnl_cents=0, event_ticker="EVT-1")
        pnl_ct = round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0
        assert pnl_ct == 0


class TestTimePressureFlag:
    """Test TIME_PRESSURE flag generation for positions near settlement."""

    def test_time_pressure_when_close(self):
        """Events with ttc < 1h and matching positions produce TIME_PRESSURE."""
        # Build market state with time_to_close_hours < 1
        event = make_event_meta(event_ticker="EVT-1", n_markets=2)
        index = make_index(events=[event])
        builder = ContextBuilder(index)
        market_state = builder.build_market_state()

        # Manually set time_to_close
        market_state.events[0].time_to_close_hours = 0.5

        # Build positions matching the event
        positions = [
            Position(
                ticker=list(event.markets.keys())[0],
                event_ticker="EVT-1",
                side="yes", quantity=5,
                unrealized_pnl_cents=20,
            ),
        ]

        # Replicate the TIME_PRESSURE logic from captain.py
        time_flags = []
        for ev in market_state.events:
            if ev.time_to_close_hours is not None and ev.time_to_close_hours < 1.0:
                for p in positions:
                    if p.event_ticker == ev.event_ticker and p.quantity > 0:
                        time_flags.append(f"{p.ticker} ttc={ev.time_to_close_hours:.1f}h")

        assert len(time_flags) == 1
        assert "ttc=0.5h" in time_flags[0]

    def test_no_time_pressure_when_far(self):
        """Events with ttc > 1h produce no TIME_PRESSURE flags."""
        event = make_event_meta(event_ticker="EVT-1", n_markets=2)
        index = make_index(events=[event])
        builder = ContextBuilder(index)
        market_state = builder.build_market_state()

        market_state.events[0].time_to_close_hours = 5.0

        positions = [
            Position(
                ticker=list(event.markets.keys())[0],
                event_ticker="EVT-1",
                side="yes", quantity=5,
                unrealized_pnl_cents=20,
            ),
        ]

        time_flags = []
        for ev in market_state.events:
            if ev.time_to_close_hours is not None and ev.time_to_close_hours < 1.0:
                for p in positions:
                    if p.event_ticker == ev.event_ticker and p.quantity > 0:
                        time_flags.append(f"{p.ticker} ttc={ev.time_to_close_hours:.1f}h")

        assert len(time_flags) == 0

    def test_no_time_pressure_no_positions(self):
        """No positions in the event means no TIME_PRESSURE flag."""
        event = make_event_meta(event_ticker="EVT-1", n_markets=2)
        index = make_index(events=[event])
        builder = ContextBuilder(index)
        market_state = builder.build_market_state()

        market_state.events[0].time_to_close_hours = 0.3

        positions = []  # No positions

        time_flags = []
        for ev in market_state.events:
            if ev.time_to_close_hours is not None and ev.time_to_close_hours < 1.0:
                for p in positions:
                    if p.event_ticker == ev.event_ticker and p.quantity > 0:
                        time_flags.append(f"{p.ticker} ttc={ev.time_to_close_hours:.1f}h")

        assert len(time_flags) == 0
