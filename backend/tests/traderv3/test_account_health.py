"""Unit tests for AccountHealthService.

Tests cover:
  1. HealthState basics (defaults, deque limits)
  2. Balance tracking (peak, trough, trend, low-balance alert)
  3. Settlement dedup + revenue accumulation
  4. Stale order detection + cancellation mocks
  5. Stale position detection from index
  6. Order group hygiene (skip current, reset+delete old)
  7. get_health_status() thresholds (healthy/warning/critical)
  8. Tick scheduling (modulo gating)
  9. Activity log and alert deque
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.single_arb.account_health import (
    AccountHealthService,
    HealthState,
    TICK_INTERVAL,
)
from kalshiflow_rl.traderv3.single_arb.models import (
    AccountHealthStatus,
    SettlementSummary,
    StalePosition,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@dataclass
class MockBalance:
    balance: int = 50000  # $500


@dataclass
class MockSettlement:
    settlement_id: str = "settle-1"
    result: str = "yes"
    revenue: int = 450
    market_ticker: str = "MKT-A"
    settled_time: str = "2026-01-01T00:00:00Z"


@dataclass
class MockPosition:
    ticker: str = "MKT-A"
    position: int = 5
    market_exposure: int = 200


@dataclass
class MockOrder:
    order_id: str = "ord-1"
    created_time: str = "2026-01-01T00:00:00Z"
    expiration_time: Optional[str] = None
    order_group_id: Optional[str] = None
    status: str = "resting"


@dataclass
class MockTradingSession:
    order_group_id: str = "grp-current"
    order_ttl: int = 60
    sniper_order_ids: Set[str] = field(default_factory=set)


def _make_gateway(
    balance: int = 50000,
    settlements: Optional[List] = None,
    positions: Optional[List] = None,
    orders: Optional[List] = None,
    order_groups: Optional[List] = None,
):
    gw = MagicMock()
    gw.get_balance = AsyncMock(return_value=MockBalance(balance=balance))
    gw.get_settlements = AsyncMock(return_value=settlements or [])
    gw.get_positions = AsyncMock(return_value=positions or [])
    gw.get_orders = AsyncMock(return_value=orders or [])
    gw.cancel_order = AsyncMock(return_value={})
    gw.list_order_groups = AsyncMock(return_value=order_groups or [])
    gw.reset_order_group = AsyncMock(return_value={})
    gw.delete_order_group = AsyncMock(return_value={})
    return gw


def _make_index(known_tickers=None):
    index = MagicMock()
    if known_tickers:
        index.get_event_for_ticker = MagicMock(
            side_effect=lambda t: "EVENT-1" if t in known_tickers else None
        )
    else:
        index.get_event_for_ticker = MagicMock(return_value=None)
    return index


def _make_service(
    gateway=None,
    index=None,
    order_group_id="grp-current",
    low_balance_threshold=500,
    balance=50000,
):
    gw = gateway or _make_gateway(balance=balance)
    idx = index or _make_index()
    session = MockTradingSession()
    return AccountHealthService(
        gateway=gw,
        index=idx,
        session=session,
        order_group_id=order_group_id,
        broadcast_callback=AsyncMock(),
        low_balance_threshold=low_balance_threshold,
    )


# ---------------------------------------------------------------------------
# HealthState basics
# ---------------------------------------------------------------------------


class TestHealthState:
    def test_defaults(self):
        s = HealthState()
        assert s.balance_cents == 0
        assert s.balance_peak_cents == 0
        assert len(s.balance_history) == 0
        assert s.total_settlement_count == 0
        assert s.stale_orders_cancelled == 0
        assert s.orphaned_groups_cleaned == 0

    def test_deque_maxlen(self):
        s = HealthState()
        assert s.balance_history.maxlen == 120
        assert s.recent_settlements.maxlen == 50
        assert s.alerts.maxlen == 100
        assert s.activity_log.maxlen == 100


# ---------------------------------------------------------------------------
# Balance tracking
# ---------------------------------------------------------------------------


class TestBalanceCheck:
    @pytest.mark.asyncio
    async def test_balance_tracking(self):
        svc = _make_service(balance=50000)
        changed = await svc._check_balance()
        assert changed is True
        assert svc.state.balance_cents == 50000
        assert svc.state.balance_peak_cents == 50000
        assert len(svc.state.balance_history) == 1

    @pytest.mark.asyncio
    async def test_peak_tracking(self):
        svc = _make_service(balance=50000)
        await svc._check_balance()
        assert svc.state.balance_peak_cents == 50000

        # Balance goes up
        svc._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=60000))
        await svc._check_balance()
        assert svc.state.balance_peak_cents == 60000

        # Balance goes down — peak stays
        svc._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=40000))
        await svc._check_balance()
        assert svc.state.balance_peak_cents == 60000

    @pytest.mark.asyncio
    async def test_trough_tracking(self):
        svc = _make_service(balance=50000)
        await svc._check_balance()
        assert svc.state.balance_trough_cents == 50000

        svc._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=30000))
        await svc._check_balance()
        assert svc.state.balance_trough_cents == 30000

    @pytest.mark.asyncio
    async def test_low_balance_alert(self):
        svc = _make_service(balance=400, low_balance_threshold=500)
        await svc._check_balance()
        assert len(svc.state.alerts) == 1
        assert svc.state.alerts[0]["type"] == "low_balance"
        assert svc.state.alerts[0]["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_no_alert_above_threshold(self):
        svc = _make_service(balance=1000, low_balance_threshold=500)
        await svc._check_balance()
        assert len(svc.state.alerts) == 0

    @pytest.mark.asyncio
    async def test_balance_api_failure(self):
        svc = _make_service()
        svc._gateway.get_balance = AsyncMock(side_effect=Exception("timeout"))
        changed = await svc._check_balance()
        assert changed is False


# ---------------------------------------------------------------------------
# Settlement tracking
# ---------------------------------------------------------------------------


class TestSettlementCheck:
    @pytest.mark.asyncio
    async def test_new_settlement_discovered(self):
        settlements = [MockSettlement(settlement_id="s1", revenue=450, market_ticker="MKT-A")]
        svc = _make_service(gateway=_make_gateway(settlements=settlements))
        changed = await svc._check_settlements()
        assert changed is True
        assert svc.state.total_settlement_count == 1
        assert svc.state.total_settlement_revenue == 450
        assert len(svc.state.recent_settlements) == 1

    @pytest.mark.asyncio
    async def test_settlement_dedup(self):
        settlements = [MockSettlement(settlement_id="s1")]
        svc = _make_service(gateway=_make_gateway(settlements=settlements))
        await svc._check_settlements()
        # Second call — same settlement, no new count
        changed = await svc._check_settlements()
        assert changed is False
        assert svc.state.total_settlement_count == 1

    @pytest.mark.asyncio
    async def test_multiple_settlements(self):
        settlements = [
            MockSettlement(settlement_id="s1", revenue=100),
            MockSettlement(settlement_id="s2", revenue=200),
        ]
        svc = _make_service(gateway=_make_gateway(settlements=settlements))
        await svc._check_settlements()
        assert svc.state.total_settlement_count == 2
        assert svc.state.total_settlement_revenue == 300

    @pytest.mark.asyncio
    async def test_settlement_alert_emitted(self):
        settlements = [MockSettlement(settlement_id="s1", revenue=450, market_ticker="MKT-A")]
        svc = _make_service(gateway=_make_gateway(settlements=settlements))
        await svc._check_settlements()
        assert len(svc.state.alerts) == 1
        assert svc.state.alerts[0]["type"] == "settlement_discovered"

    @pytest.mark.asyncio
    async def test_positions_cached(self):
        positions = [MockPosition(ticker="MKT-A", position=5)]
        svc = _make_service(gateway=_make_gateway(positions=positions))
        await svc._check_settlements()
        assert len(svc.state.cached_positions) == 1
        assert svc.state.cached_positions[0]["ticker"] == "MKT-A"


# ---------------------------------------------------------------------------
# Stale position detection
# ---------------------------------------------------------------------------


class TestStalePositions:
    @pytest.mark.asyncio
    async def test_position_not_in_index(self):
        svc = _make_service(index=_make_index(known_tickers=set()))
        svc.state.cached_positions = [{"ticker": "MKT-DEAD", "position": 5}]
        changed = await svc._check_stale_positions()
        assert changed is True
        assert len(svc.state.stale_positions) == 1
        assert svc.state.stale_positions[0].ticker == "MKT-DEAD"
        assert svc.state.stale_positions[0].reason == "not_in_index"

    @pytest.mark.asyncio
    async def test_position_in_index_not_stale(self):
        svc = _make_service(index=_make_index(known_tickers={"MKT-LIVE"}))
        svc.state.cached_positions = [{"ticker": "MKT-LIVE", "position": 5}]
        changed = await svc._check_stale_positions()
        assert changed is False
        assert len(svc.state.stale_positions) == 0

    @pytest.mark.asyncio
    async def test_zero_position_ignored(self):
        svc = _make_service(index=_make_index(known_tickers=set()))
        svc.state.cached_positions = [{"ticker": "MKT-DEAD", "position": 0}]
        await svc._check_stale_positions()
        assert len(svc.state.stale_positions) == 0


# ---------------------------------------------------------------------------
# Stale order cleanup
# ---------------------------------------------------------------------------


class TestStaleOrders:
    @pytest.mark.asyncio
    async def test_cancels_old_orphan_order(self):
        # Order created 2 hours ago, not in current group
        old_order = MockOrder(
            order_id="ord-old",
            created_time="2020-01-01T00:00:00Z",
            order_group_id="grp-other",
        )
        gw = _make_gateway(orders=[old_order])
        svc = _make_service(gateway=gw, order_group_id="grp-current")
        changed = await svc._check_stale_orders()
        assert changed is True
        assert svc.state.stale_orders_cancelled == 1
        gw.cancel_order.assert_called_once_with("ord-old")

    @pytest.mark.asyncio
    async def test_skips_current_group_orders(self):
        order = MockOrder(
            order_id="ord-mine",
            created_time="2020-01-01T00:00:00Z",
            order_group_id="grp-current",
        )
        gw = _make_gateway(orders=[order])
        svc = _make_service(gateway=gw, order_group_id="grp-current")
        changed = await svc._check_stale_orders()
        assert changed is False
        gw.cancel_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_order_alert(self):
        old_order = MockOrder(
            order_id="ord-stale",
            created_time="2020-01-01T00:00:00Z",
            order_group_id=None,
        )
        gw = _make_gateway(orders=[old_order])
        svc = _make_service(gateway=gw)
        await svc._check_stale_orders()
        assert any(a["type"] == "stale_order_cancelled" for a in svc.state.alerts)


# ---------------------------------------------------------------------------
# Order group hygiene
# ---------------------------------------------------------------------------


class TestOrderGroupHygiene:
    @pytest.mark.asyncio
    async def test_cleans_orphaned_group(self):
        groups = [
            {"order_group_id": "grp-current"},
            {"order_group_id": "grp-orphan"},
        ]
        gw = _make_gateway(order_groups=groups)
        svc = _make_service(gateway=gw, order_group_id="grp-current")
        changed = await svc._check_order_groups()
        assert changed is True
        assert svc.state.orphaned_groups_cleaned == 1
        gw.reset_order_group.assert_called_once_with("grp-orphan")
        gw.delete_order_group.assert_called_once_with("grp-orphan")

    @pytest.mark.asyncio
    async def test_skips_current_group(self):
        groups = [{"order_group_id": "grp-current"}]
        gw = _make_gateway(order_groups=groups)
        svc = _make_service(gateway=gw, order_group_id="grp-current")
        changed = await svc._check_order_groups()
        assert changed is False
        gw.reset_order_group.assert_not_called()


# ---------------------------------------------------------------------------
# get_health_status
# ---------------------------------------------------------------------------


class TestHealthStatus:
    def test_healthy_default(self):
        svc = _make_service()
        svc.state.balance_cents = 50000
        svc.state.balance_peak_cents = 50000
        status = svc.get_health_status()
        assert status.status == "healthy"
        assert status.drawdown_pct == 0.0

    def test_warning_on_drawdown(self):
        svc = _make_service()
        svc.state.balance_cents = 44000
        svc.state.balance_peak_cents = 50000  # 12% drawdown
        status = svc.get_health_status()
        assert status.status == "warning"
        assert status.drawdown_pct == 12.0

    def test_critical_on_high_drawdown(self):
        svc = _make_service()
        svc.state.balance_cents = 35000
        svc.state.balance_peak_cents = 50000  # 30% drawdown
        status = svc.get_health_status()
        assert status.status == "critical"

    def test_critical_on_low_balance(self):
        svc = _make_service(low_balance_threshold=500)
        svc.state.balance_cents = 400
        svc.state.balance_peak_cents = 400
        status = svc.get_health_status()
        assert status.status == "critical"

    def test_warning_on_stale_positions(self):
        svc = _make_service()
        svc.state.balance_cents = 50000
        svc.state.balance_peak_cents = 50000
        svc.state.stale_positions = [
            StalePosition(ticker="MKT-X", reason="not_in_index")
        ]
        status = svc.get_health_status()
        assert status.status == "warning"

    def test_balance_trend_rising(self):
        svc = _make_service()
        svc.state.balance_cents = 55000
        svc.state.balance_peak_cents = 55000
        # Fill history: first half low, second half high
        for _ in range(60):
            svc.state.balance_history.append(40000)
        for _ in range(60):
            svc.state.balance_history.append(55000)
        status = svc.get_health_status()
        assert status.balance_trend == "rising"

    def test_balance_trend_falling(self):
        svc = _make_service()
        svc.state.balance_cents = 40000
        svc.state.balance_peak_cents = 55000
        for _ in range(60):
            svc.state.balance_history.append(55000)
        for _ in range(60):
            svc.state.balance_history.append(40000)
        status = svc.get_health_status()
        assert status.balance_trend == "falling"

    def test_settlements_in_status(self):
        svc = _make_service()
        svc.state.balance_cents = 50000
        svc.state.balance_peak_cents = 50000
        svc.state.total_settlement_count = 3
        svc.state.total_settlement_revenue = 1200
        svc.state.recent_settlements.appendleft({
            "ticker": "MKT-A",
            "result": "yes",
            "revenue_cents": 450,
            "settled_at": None,
        })
        status = svc.get_health_status()
        assert status.settlement_count_session == 3
        assert status.total_realized_pnl_cents == 1200
        assert len(status.recent_settlements) == 1

    def test_model_serialization(self):
        svc = _make_service()
        svc.state.balance_cents = 50000
        svc.state.balance_peak_cents = 50000
        status = svc.get_health_status()
        d = status.model_dump()
        assert "status" in d
        assert "drawdown_pct" in d
        assert "activity_log" in d


# ---------------------------------------------------------------------------
# Tick scheduling
# ---------------------------------------------------------------------------


class TestTickScheduling:
    @pytest.mark.asyncio
    async def test_tick_calls_balance_every_time(self):
        svc = _make_service()
        await svc._tick()
        svc._gateway.get_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_settlements_on_tick_1(self):
        svc = _make_service()
        svc._tick_count = 1  # tick % 4 == 1
        await svc._tick()
        svc._gateway.get_settlements.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_settlements_not_on_tick_0(self):
        svc = _make_service()
        svc._tick_count = 0  # tick % 4 != 1
        await svc._tick()
        svc._gateway.get_settlements.assert_not_called()

    @pytest.mark.asyncio
    async def test_tick_stale_orders_on_tick_5(self):
        svc = _make_service()
        svc._tick_count = 5  # tick % 10 == 5
        await svc._tick()
        svc._gateway.get_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_tick_order_groups_on_tick_30(self):
        svc = _make_service()
        svc._tick_count = 30  # tick % 60 == 30
        await svc._tick()
        svc._gateway.list_order_groups.assert_called_once()


# ---------------------------------------------------------------------------
# Alert system
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_add_alert(self):
        svc = _make_service()
        svc._add_alert("test_type", "test message", "info")
        assert len(svc.state.alerts) == 1
        assert svc.state.alerts[0]["type"] == "test_type"
        assert svc.state.alerts[0]["severity"] == "info"

    def test_alert_in_activity_log(self):
        svc = _make_service()
        svc._add_alert("test_type", "test message", "info")
        assert len(svc.state.activity_log) == 1
        assert svc.state.activity_log[0]["type"] == "test_type"

    def test_alert_deque_limit(self):
        svc = _make_service()
        for i in range(150):
            svc._add_alert("flood", f"msg {i}", "info")
        assert len(svc.state.alerts) == 100  # maxlen


# ---------------------------------------------------------------------------
# Tool integration
# ---------------------------------------------------------------------------


class TestToolIntegration:
    @pytest.mark.asyncio
    async def test_get_account_health_tool(self):
        """Verify the tool returns AccountHealthStatus shape."""
        from kalshiflow_rl.traderv3.single_arb.tools import (
            get_account_health,
            ToolContext,
            set_context,
        )

        svc = _make_service()
        svc.state.balance_cents = 50000
        svc.state.balance_peak_cents = 50000

        ctx = MagicMock(spec=ToolContext)
        ctx.health_service = svc
        set_context(ctx)

        result = await get_account_health.ainvoke({})
        assert isinstance(result, dict)
        assert result["status"] == "healthy"
        assert "drawdown_pct" in result
        assert "activity_log" in result

    @pytest.mark.asyncio
    async def test_get_account_health_no_service(self):
        from kalshiflow_rl.traderv3.single_arb.tools import (
            get_account_health,
            ToolContext,
            set_context,
        )

        ctx = MagicMock(spec=ToolContext)
        ctx.health_service = None
        set_context(ctx)

        result = await get_account_health.ainvoke({})
        assert "error" in result


# ---------------------------------------------------------------------------
# Settlement-to-memory (Gap 3)
# ---------------------------------------------------------------------------


class TestSettlementMemoryFeedback:
    @pytest.mark.asyncio
    async def test_settlement_stores_to_memory(self):
        """New settlements should fire memory.store() with settlement_outcome type."""
        mock_memory = MagicMock()
        mock_memory.store = AsyncMock()

        gw = _make_gateway(
            settlements=[MockSettlement(settlement_id="s1", result="yes", revenue=450, market_ticker="MKT-A")],
        )
        svc = AccountHealthService(
            gateway=gw,
            index=_make_index(),
            session=MockTradingSession(),
            order_group_id="grp-current",
            broadcast_callback=AsyncMock(),
            memory=mock_memory,
        )

        changed = await svc._check_settlements()
        assert changed is True

        # Verify memory.store was called with settlement_outcome
        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args
        assert call_kwargs.kwargs["memory_type"] == "settlement_outcome"
        assert "SETTLEMENT" in call_kwargs.kwargs["content"]
        assert "MKT-A" in call_kwargs.kwargs["content"]
        assert call_kwargs.kwargs["metadata"]["result"] == "yes"
        assert call_kwargs.kwargs["metadata"]["revenue_cents"] == 450

    @pytest.mark.asyncio
    async def test_settlement_no_memory_graceful(self):
        """No error when memory=None and settlement discovered."""
        gw = _make_gateway(
            settlements=[MockSettlement(settlement_id="s2", result="no", revenue=-200, market_ticker="MKT-B")],
        )
        svc = AccountHealthService(
            gateway=gw,
            index=_make_index(),
            session=MockTradingSession(),
            order_group_id="grp-current",
            broadcast_callback=AsyncMock(),
            # memory not provided (defaults to None)
        )

        # Should not raise
        changed = await svc._check_settlements()
        assert changed is True
        assert svc.state.total_settlement_count == 1
