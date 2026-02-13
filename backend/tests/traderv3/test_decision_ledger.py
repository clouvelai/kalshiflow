"""Tests for DecisionLedger — order decision tracking + production price evaluation."""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from kalshiflow_rl.traderv3.single_arb.decision_ledger import DecisionLedger


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_index(markets: dict = None):
    """Create a mock EventArbIndex with given market data."""
    index = MagicMock()
    events = {}
    if markets:
        for mt, data in markets.items():
            event_ticker = data.get("event_ticker", "EVT-1")
            if event_ticker not in events:
                event = MagicMock()
                event.markets = {}
                events[event_ticker] = event
            market = MagicMock()
            market.yes_bid = data.get("yes_bid")
            market.yes_ask = data.get("yes_ask")
            market.yes_mid = data.get("yes_mid")
            market.spread = data.get("spread", 2)
            micro = MagicMock()
            micro.volume_5m = data.get("volume_5m", 100)
            micro.book_imbalance = data.get("book_imbalance", 0.5)
            market.micro = micro
            events[event_ticker].markets[mt] = market
    index.events = events
    return index


def _make_mock_pool(mock_conn):
    """Create a mock asyncpg pool with proper async context manager for acquire()."""
    mock_pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield mock_conn

    mock_pool.acquire = _acquire
    return mock_pool


def _make_row(overrides: dict = None):
    """Create a mock DB row for backfill testing."""
    base = {
        "id": "dec-1",
        "order_id": "ord-1",
        "market_ticker": "MKT-YES",
        "side": "yes",
        "action": "buy",
        "contracts": 10,
        "limit_price_cents": 52,
        "prod_yes_mid": 55.0,
        "prod_mid_1m": None,
        "prod_mid_5m": None,
        "prod_mid_15m": None,
        "prod_mid_1h": None,
        "demo_status": None,
        "demo_fill_count": 0,
        "direction_correct": None,
        "would_have_filled": None,
        "created_at": datetime.now(timezone.utc) - timedelta(minutes=10),
    }
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
#  Production snapshot tests
# ---------------------------------------------------------------------------

class TestRecordDecision:
    def test_captures_prod_snapshot(self):
        """record_decision should capture BBO snapshot from index."""
        index = _make_index({
            "MKT-YES": {"yes_bid": 50, "yes_ask": 54, "yes_mid": 52.0, "spread": 4,
                         "volume_5m": 200, "book_imbalance": 0.6},
        })
        ledger = DecisionLedger(index=index)
        snap = ledger._get_prod_snapshot("MKT-YES")

        assert snap["prod_yes_bid"] == 50
        assert snap["prod_yes_ask"] == 54
        assert snap["prod_yes_mid"] == 52.0
        assert snap["prod_spread"] == 4
        assert snap["prod_volume_5m"] == 200
        assert snap["prod_book_imbalance"] == 0.6

    def test_snapshot_missing_market(self):
        """Snapshot returns empty dict for unknown market."""
        index = _make_index({})
        ledger = DecisionLedger(index=index)
        snap = ledger._get_prod_snapshot("UNKNOWN-MKT")
        assert snap == {}

    @pytest.mark.asyncio
    async def test_graceful_without_db(self):
        """record_decision should not crash when DB pool is unavailable."""
        index = _make_index({"MKT-YES": {"yes_mid": 50.0}})
        ledger = DecisionLedger(index=index)
        # Force pool unavailable
        ledger._pool_checked = True
        ledger._pool = None

        # Should not raise
        await ledger.record_decision(
            order_id="ord-1", source="captain",
            event_ticker="EVT-1", market_ticker="MKT-YES",
            side="yes", action="buy", contracts=5,
            limit_price_cents=50,
        )

    @pytest.mark.asyncio
    async def test_record_calls_db_insert(self):
        """record_decision inserts row into captain_decisions."""
        index = _make_index({"MKT-YES": {"yes_bid": 48, "yes_ask": 52, "yes_mid": 50.0}})
        ledger = DecisionLedger(index=index)

        mock_conn = AsyncMock()
        ledger._pool = _make_mock_pool(mock_conn)
        ledger._pool_checked = True

        await ledger.record_decision(
            order_id="ord-1", source="captain",
            event_ticker="EVT-1", market_ticker="MKT-YES",
            side="yes", action="buy", contracts=5,
            limit_price_cents=50, reasoning="test",
            cycle_mode="strategic",
        )

        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        assert "INSERT INTO captain_decisions" in call_args[0][0]
        assert call_args[0][1] == "ord-1"  # order_id
        assert call_args[0][2] == "captain"  # source


# ---------------------------------------------------------------------------
#  Direction correctness tests
# ---------------------------------------------------------------------------

class TestDirectionCorrect:
    def test_buy_yes_up(self):
        """Buy YES, mid went up → direction correct."""
        row = _make_row({"side": "yes", "action": "buy", "prod_yes_mid": 50.0})
        updates = {"prod_mid_1h": 55.0}
        mids = DecisionLedger._collect_sampled_mids(row, updates)
        # Bullish (buy yes): correct if latest mid > entry mid
        latest = mids[-1]
        entry = row["prod_yes_mid"]
        assert latest > entry

    def test_buy_yes_down(self):
        """Buy YES, mid went down → direction wrong."""
        row = _make_row({"side": "yes", "action": "buy", "prod_yes_mid": 50.0})
        updates = {"prod_mid_1h": 45.0}
        mids = DecisionLedger._collect_sampled_mids(row, updates)
        latest = mids[-1]
        entry = row["prod_yes_mid"]
        assert latest < entry  # Wrong direction for bullish

    def test_buy_no_down(self):
        """Buy NO, YES mid went down → direction correct (bearish bet pays off)."""
        row = _make_row({"side": "no", "action": "buy", "prod_yes_mid": 50.0})
        updates = {"prod_mid_1h": 45.0}
        mids = DecisionLedger._collect_sampled_mids(row, updates)
        latest = mids[-1]
        entry = row["prod_yes_mid"]
        # Bearish (buy no): correct if latest < entry
        assert latest < entry

    def test_sell_yes_down(self):
        """Sell YES, mid went down → direction correct (bearish on yes)."""
        row = _make_row({"side": "yes", "action": "sell", "prod_yes_mid": 50.0})
        updates = {"prod_mid_1h": 45.0}
        mids = DecisionLedger._collect_sampled_mids(row, updates)
        latest = mids[-1]
        entry = row["prod_yes_mid"]
        # Bearish (sell yes): correct if latest < entry
        assert latest < entry


# ---------------------------------------------------------------------------
#  Would-have-filled tests
# ---------------------------------------------------------------------------

class TestWouldHaveFilled:
    def test_buy_yes_filled(self):
        """Buy YES at 52c, prod mid crossed below 52 → would have filled."""
        result = DecisionLedger._check_would_have_filled(
            side="yes", action="buy", limit_cents=52,
            sampled_mids=[55.0, 53.0, 51.0, 54.0],
        )
        assert result is True

    def test_buy_yes_not_filled(self):
        """Buy YES at 52c, mid stayed above → would NOT have filled."""
        result = DecisionLedger._check_would_have_filled(
            side="yes", action="buy", limit_cents=52,
            sampled_mids=[55.0, 56.0, 57.0, 58.0],
        )
        assert result is False

    def test_buy_no_filled(self):
        """Buy NO at 48c, YES mid crossed above 52 (NO ask ≤ 48) → filled."""
        # buy NO at 48c means we need YES mid >= 100-48 = 52
        result = DecisionLedger._check_would_have_filled(
            side="no", action="buy", limit_cents=48,
            sampled_mids=[50.0, 51.0, 53.0, 55.0],
        )
        assert result is True

    def test_buy_no_not_filled(self):
        """Buy NO at 48c, YES mid stayed below 52 → would NOT have filled."""
        result = DecisionLedger._check_would_have_filled(
            side="no", action="buy", limit_cents=48,
            sampled_mids=[45.0, 46.0, 47.0, 48.0],
        )
        assert result is False

    def test_sell_yes_filled(self):
        """Sell YES at 55c, mid crossed above 55 → filled."""
        result = DecisionLedger._check_would_have_filled(
            side="yes", action="sell", limit_cents=55,
            sampled_mids=[50.0, 53.0, 56.0, 54.0],
        )
        assert result is True

    def test_sell_no_filled(self):
        """Sell NO at 48c, YES mid crossed below 52 → NO value rose → filled."""
        # sell NO at 48c means buyer bids 48c for NO, needs YES mid <= 100-48 = 52
        result = DecisionLedger._check_would_have_filled(
            side="no", action="sell", limit_cents=48,
            sampled_mids=[55.0, 53.0, 51.0, 54.0],
        )
        assert result is True


# ---------------------------------------------------------------------------
#  Hypothetical P&L tests
# ---------------------------------------------------------------------------

class TestHypotheticalPnl:
    def test_buy_yes_profit(self):
        """Buy YES at 52c, marked at 58c → +6c * 10 contracts = +60c."""
        pnl = DecisionLedger._compute_pnl(
            side="yes", action="buy", limit_cents=52,
            mark_mid=58.0, contracts=10,
        )
        assert pnl == 60.0

    def test_buy_yes_loss(self):
        """Buy YES at 52c, marked at 48c → -4c * 10 = -40c."""
        pnl = DecisionLedger._compute_pnl(
            side="yes", action="buy", limit_cents=52,
            mark_mid=48.0, contracts=10,
        )
        assert pnl == -40.0

    def test_buy_no_profit(self):
        """Buy NO at 48c, YES mid now 45c → NO value = 55c → +7c * 10 = +70c."""
        pnl = DecisionLedger._compute_pnl(
            side="no", action="buy", limit_cents=48,
            mark_mid=45.0, contracts=10,
        )
        assert pnl == 70.0

    def test_sell_yes_profit(self):
        """Sell YES at 55c, marked at 50c → +5c * 10 = +50c."""
        pnl = DecisionLedger._compute_pnl(
            side="yes", action="sell", limit_cents=55,
            mark_mid=50.0, contracts=10,
        )
        assert pnl == 50.0

    def test_sell_no_profit(self):
        """Sell NO at 48c, YES mid now 55c → NO value = 45c → +3c * 10 = +30c."""
        pnl = DecisionLedger._compute_pnl(
            side="no", action="sell", limit_cents=48,
            mark_mid=55.0, contracts=10,
        )
        assert pnl == 30.0


# ---------------------------------------------------------------------------
#  Excursion tests
# ---------------------------------------------------------------------------

class TestExcursions:
    def test_max_favorable_adverse(self):
        """Correct max favorable and adverse from sampled prices."""
        # Buy YES at 50c, mids: [52, 48, 55, 51]
        # PnLs: [+2, -2, +5, +1]
        fav, adv = DecisionLedger._compute_excursions(
            side="yes", action="buy", limit_cents=50,
            sampled_mids=[52.0, 48.0, 55.0, 51.0],
        )
        assert fav == 5.0
        assert adv == -2.0

    def test_empty_mids(self):
        """No sampled mids → None, None."""
        fav, adv = DecisionLedger._compute_excursions(
            side="yes", action="buy", limit_cents=50,
            sampled_mids=[],
        )
        assert fav is None
        assert adv is None


# ---------------------------------------------------------------------------
#  Backfill window tests
# ---------------------------------------------------------------------------

class TestBackfillWindows:
    def test_collect_sampled_mids_from_updates(self):
        """Collect mids from both row (existing) and updates (new)."""
        row = _make_row({"prod_mid_1m": 51.0, "prod_mid_5m": None})
        updates = {"prod_mid_5m": 53.0}
        mids = DecisionLedger._collect_sampled_mids(row, updates)
        assert mids == [51.0, 53.0]

    def test_collect_sampled_mids_all_from_row(self):
        """All windows already filled in row."""
        row = _make_row({
            "prod_mid_1m": 51.0, "prod_mid_5m": 52.0,
            "prod_mid_15m": 53.0, "prod_mid_1h": 54.0,
        })
        mids = DecisionLedger._collect_sampled_mids(row, {})
        assert mids == [51.0, 52.0, 53.0, 54.0]

    def test_collect_sampled_mids_none(self):
        """No windows filled → empty list."""
        row = _make_row()
        mids = DecisionLedger._collect_sampled_mids(row, {})
        assert mids == []


# ---------------------------------------------------------------------------
#  Accuracy stats tests
# ---------------------------------------------------------------------------

class TestAccuracyStats:
    @pytest.mark.asyncio
    async def test_empty_stats_no_db(self):
        """No DB pool → return zeroed stats."""
        index = _make_index({})
        ledger = DecisionLedger(index=index)
        ledger._pool_checked = True
        ledger._pool = None

        stats = await ledger.get_accuracy_stats()
        assert stats["total_decisions"] == 0
        assert stats["direction_accuracy_pct"] == 0.0
        assert stats["by_source"] == {}

    @pytest.mark.asyncio
    async def test_stats_from_db(self):
        """Stats are fetched from SQL function and returned."""
        index = _make_index({})
        ledger = DecisionLedger(index=index)

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = {
            "total_decisions": 10,
            "decisions_with_outcomes": 8,
            "direction_correct_count": 6,
            "direction_accuracy_pct": 75.0,
            "avg_hypothetical_pnl": 5.5,
            "total_hypothetical_pnl": 44.0,
            "would_have_filled_count": 7,
            "would_have_filled_pct": 87.5,
            "by_source": {"captain": {"total": 8, "correct": 5}},
            "by_cycle_mode": {"strategic": {"total": 5, "correct": 4}},
        }
        ledger._pool = _make_mock_pool(mock_conn)
        ledger._pool_checked = True

        stats = await ledger.get_accuracy_stats(hours_back=24)
        assert stats["total_decisions"] == 10
        assert stats["direction_accuracy_pct"] == 75.0
        assert stats["by_source"]["captain"]["total"] == 8


# ---------------------------------------------------------------------------
#  Demo status sync test
# ---------------------------------------------------------------------------

class TestDemoStatusSync:
    @pytest.mark.asyncio
    async def test_demo_status_synced_from_tracked_orders(self):
        """backfill_outcomes syncs demo_status from tracked_orders dict."""
        index = _make_index({"MKT-YES": {"yes_mid": 55.0}})
        ledger = DecisionLedger(index=index)

        row = _make_row({
            "created_at": datetime.now(timezone.utc) - timedelta(minutes=2),
            "demo_status": None,
        })

        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [row]
        mock_conn.execute.return_value = None
        ledger._pool = _make_mock_pool(mock_conn)
        ledger._pool_checked = True

        tracked_orders = {
            "ord-1": {"status": "executed", "fill_count": 10},
        }

        await ledger.backfill_outcomes(tracked_orders)

        # Verify UPDATE was called
        assert mock_conn.execute.call_count >= 1
        update_call = mock_conn.execute.call_args
        sql = update_call[0][0]
        assert "UPDATE captain_decisions" in sql
