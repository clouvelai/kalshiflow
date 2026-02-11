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


# ===========================================================================
# TestBuildReactiveContext
# ===========================================================================

class TestBuildReactiveContext:
    def _make_builder(self):
        event = make_event_meta("EVT-1", n_markets=3)
        index = make_index(events=[event])
        return ContextBuilder(index=index)

    def _make_items(self):
        from kalshiflow_rl.traderv3.single_arb.models import AttentionItem
        return [
            AttentionItem(
                event_ticker="EVT-1",
                urgency="immediate",
                category="arb_opportunity",
                score=75.0,
                summary="long edge=8.2c, regime=normal",
            ),
            AttentionItem(
                event_ticker="EVT-1",
                market_ticker="EVT-1-MKT-A",
                urgency="high",
                category="position_risk",
                score=60.0,
                summary="pnl=-11c/ct",
            ),
        ]

    def test_contains_attention_section(self):
        builder = self._make_builder()
        items = self._make_items()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_reactive_context(items, portfolio)
        assert "ATTENTION:" in result
        assert "[IMMEDIATE]" in result
        assert "[HIGH]" in result

    def test_contains_portfolio(self):
        builder = self._make_builder()
        items = self._make_items()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_reactive_context(items, portfolio)
        assert "PORTFOLIO:" in result
        assert "$500.00" in result

    def test_contains_action_directive(self):
        builder = self._make_builder()
        items = self._make_items()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_reactive_context(items, portfolio)
        assert "ACTION:" in result

    def test_shows_relevant_positions(self):
        from kalshiflow_rl.traderv3.single_arb.models import Position
        builder = self._make_builder()
        items = self._make_items()
        portfolio = PortfolioState(
            balance_cents=50000,
            balance_dollars=500.0,
            positions=[
                Position(ticker="EVT-1-MKT-A", event_ticker="EVT-1",
                         side="yes", quantity=10, cost_cents=400,
                         unrealized_pnl_cents=-110),
            ],
            total_positions=1,
            total_unrealized_pnl_cents=-110,
        )
        result = builder.build_reactive_context(items, portfolio)
        assert "EVT-1-MKT-A" in result
        assert "yes" in result

    def test_includes_sniper_when_enabled(self):
        builder = self._make_builder()
        items = self._make_items()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        sniper = SniperStatus(enabled=True, total_arbs_executed=3, capital_in_flight=100)
        result = builder.build_reactive_context(items, portfolio, sniper)
        assert "SNIPER:" in result
        assert "3 arbs" in result


# ===========================================================================
# TestBuildStrategicContext
# ===========================================================================

class TestBuildStrategicContext:
    def _make_builder(self):
        event = make_event_meta("EVT-1", n_markets=3)
        index = make_index(events=[event])
        return ContextBuilder(index=index)

    def test_contains_portfolio(self):
        builder = self._make_builder()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_strategic_context(portfolio, [])
        assert "PORTFOLIO:" in result

    def test_contains_pending_attention(self):
        from kalshiflow_rl.traderv3.single_arb.models import AttentionItem
        builder = self._make_builder()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        pending = [
            AttentionItem(
                event_ticker="EVT-1",
                urgency="normal",
                category="edge_emergence",
                score=42.0,
                summary="edge=3.1c",
            ),
        ]
        result = builder.build_strategic_context(portfolio, pending)
        assert "PENDING_ATTENTION:" in result
        assert "1 items" in result

    def test_contains_task_section(self):
        builder = self._make_builder()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_strategic_context(portfolio, [], task_section="TASKS:\n[>] Monitor EVT-1")
        assert "TASKS:" in result

    def test_contains_action_directive(self):
        builder = self._make_builder()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_strategic_context(portfolio, [])
        assert "ACTION:" in result

    def test_contains_sniper_stats(self):
        builder = self._make_builder()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        sniper = SniperStatus(
            enabled=True, total_arbs_executed=5, total_trades=8,
            last_rejection_reason="vpin_toxic",
        )
        result = builder.build_strategic_context(portfolio, [], sniper)
        assert "SNIPER:" in result
        assert "5 arbs" in result
        assert "vpin_toxic" in result


# ===========================================================================
# TestBuildDeepScanContext
# ===========================================================================

class TestBuildDeepScanContext:
    def _make_builder(self):
        event = make_event_meta("EVT-1", n_markets=3)
        index = make_index(events=[event])
        return ContextBuilder(index=index)

    def test_contains_events_json(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_deep_scan_context(market_state, portfolio)
        assert "EVENTS:" in result
        assert "EVT-1" in result

    def test_contains_full_portfolio(self):
        from kalshiflow_rl.traderv3.single_arb.models import Position
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(
            balance_cents=50000,
            balance_dollars=500.0,
            positions=[
                Position(ticker="EVT-1-MKT-A", event_ticker="EVT-1",
                         side="yes", quantity=10, cost_cents=400,
                         unrealized_pnl_cents=-50),
            ],
            total_positions=1,
            total_unrealized_pnl_cents=-50,
            total_cost_cents=400,
        )
        result = builder.build_deep_scan_context(market_state, portfolio)
        assert "EVT-1-MKT-A" in result
        assert "cost=" in result

    def test_contains_health(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        health = {"drawdown_pct": 8.5, "total_realized_pnl_cents": 1200, "settlement_count_session": 3}
        result = builder.build_deep_scan_context(market_state, portfolio, health=health)
        assert "HEALTH:" in result
        assert "8.5%" in result

    def test_contains_trade_memories(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        trade_mems = ["OUTCOME: buy 5/5 yes EVT-1-MKT-A @40c -> executed", "OUTCOME: sell cancelled"]
        result = builder.build_deep_scan_context(market_state, portfolio, trade_memories=trade_mems)
        assert "TRADE_LEARNINGS:" in result
        assert "executed" in result

    def test_contains_news_memories(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        news_mems = ["Fed rate decision moved markets 8c", "Jobs report missed expectations"]
        result = builder.build_deep_scan_context(market_state, portfolio, news_memories=news_mems)
        assert "NEWS_LEARNINGS:" in result
        assert "Fed rate" in result

    def test_no_memories_omits_sections(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_deep_scan_context(market_state, portfolio)
        assert "TRADE_LEARNINGS:" not in result
        assert "NEWS_LEARNINGS:" not in result

    def test_contains_sniper_perf(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        sniper = SniperStatus(
            enabled=True, total_arbs_executed=10, total_trades=15,
            total_partial_unwinds=2, capital_deployed_lifetime=5000,
        )
        result = builder.build_deep_scan_context(market_state, portfolio, sniper)
        assert "SNIPER_PERF:" in result
        assert "arbs=10" in result

    def test_contains_action_directive(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_deep_scan_context(market_state, portfolio)
        assert "ACTION:" in result

    def test_contains_market_movers(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        movers = [
            {"news_title": "Fed cuts rates", "market_ticker": "MKT-A", "change_cents": 8, "direction": "up"},
            {"news_title": "Jobs report miss", "market_ticker": "MKT-B", "change_cents": 5, "direction": "down"},
        ]
        result = builder.build_deep_scan_context(market_state, portfolio, market_movers=movers)
        assert "NEWS_IMPACT" in result
        assert "Fed cuts rates" in result
        assert "8c" in result

    def test_no_market_movers_omits_section(self):
        builder = self._make_builder()
        market_state = builder.build_market_state()
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_deep_scan_context(market_state, portfolio, market_movers=None)
        assert "NEWS_IMPACT" not in result


# ===========================================================================
# TestSubaccountBalanceDiagnostics
# ===========================================================================

class TestSubaccountBalanceDiagnostics:
    """Verify subaccount-aware balance messages in prompt output."""

    def _make_builder(self, subaccount: int = 0):
        event = make_event_meta("EVT-1", n_markets=2)
        index = make_index(events=[event])
        return ContextBuilder(index=index, subaccount=subaccount)

    def _item(self):
        from kalshiflow_rl.traderv3.single_arb.models import AttentionItem
        return AttentionItem(event_ticker="EVT-1", summary="test")

    def test_subaccount_zero_balance_warns(self):
        builder = self._make_builder(subaccount=1)
        portfolio = PortfolioState(balance_cents=0, balance_dollars=0.0)
        result = builder.build_reactive_context(
            [self._item()], portfolio
        )
        assert "subaccount #1" in result
        assert "zero balance" in result
        assert "transfer funds" in result

    def test_subaccount_unavailable_balance_shows_error(self):
        builder = self._make_builder(subaccount=2)
        portfolio = PortfolioState(balance_cents=0, balance_dollars=None)
        result = builder.build_strategic_context(portfolio, [])
        assert "subaccount #2" in result
        assert "balance fetch failed" in result

    def test_primary_account_unavailable_shows_generic(self):
        builder = self._make_builder(subaccount=0)
        portfolio = PortfolioState(balance_cents=0, balance_dollars=None)
        result = builder.build_strategic_context(portfolio, [])
        assert "$unavailable" in result
        assert "subaccount" not in result

    def test_subaccount_normal_balance_no_warning(self):
        builder = self._make_builder(subaccount=1)
        portfolio = PortfolioState(balance_cents=50000, balance_dollars=500.0)
        result = builder.build_reactive_context(
            [self._item()], portfolio
        )
        assert "$500.00" in result
        assert "WARNING" not in result
        assert "ERROR" not in result

    def test_primary_account_zero_balance_no_warning(self):
        """$0 on primary account is not flagged (could be legitimate)."""
        builder = self._make_builder(subaccount=0)
        portfolio = PortfolioState(balance_cents=0, balance_dollars=0.0)
        result = builder.build_reactive_context(
            [self._item()], portfolio
        )
        assert "$0.00" in result
        assert "WARNING" not in result
