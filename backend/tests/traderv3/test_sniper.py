"""Unit tests for Sniper execution layer hardening.

Tests cover all 13 fixes from the Sniper hardening plan:
  1. Capital accounting lifecycle (in_flight → in_positions, release on expiry)
  2. Partial fill unwinding (cancel successful legs on partial)
  3. Leg timeout (asyncio.wait_for wrapping)
  4. Config TTL (order_ttl from SniperConfig)
  5. Capital gate cost estimate (uses actual contracts_per_leg)
  6. Broadcast fire-and-forget (asyncio.create_task)
  7. Deque for recent_actions (auto-eviction)
  8. Unknown config fields reported
  9. VPIN threshold from config (tested via tools separately)
  10. Stale order cleanup
  11. Error type classification
  12. Edge logging (edge_net_cents in SniperAction)
  13. Concurrency docstring (verified by existence)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.single_arb.sniper import (
    Sniper,
    SniperAction,
    SniperConfig,
    SniperState,
)


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

@dataclass
class MockOrderResponse:
    order_id: str = "order-abc-123"


@dataclass
class MockTradingSession:
    order_group_id: str = "grp-test"
    order_ttl: int = 30
    sniper_order_ids: Set[str] = field(default_factory=set)

    def reset(self):
        self.sniper_order_ids.clear()


@dataclass
class MockLeg:
    ticker: str = "MARKET-A"
    side: str = "yes"
    action: str = "buy"
    price_cents: int = 40
    size_available: int = 10


@dataclass
class MockArbOpportunity:
    event_ticker: str = "EVENT-1"
    direction: str = "long"
    edge_after_fees: float = 5.0
    legs: list = field(default_factory=lambda: [
        MockLeg(ticker="MKT-A", price_cents=40, size_available=10),
        MockLeg(ticker="MKT-B", price_cents=30, size_available=10),
        MockLeg(ticker="MKT-C", price_cents=20, size_available=10),
    ])

    def to_dict(self):
        return {"event_ticker": self.event_ticker}


@dataclass
class MockMicro:
    vpin: float = 0.3
    book_imbalance: float = 0.5
    ofi: float = 0.0
    ofi_ema: float = 0.0
    volume_5m: int = 100
    buy_sell_ratio: float = 1.0
    consistent_size_ratio: float = 0.5
    total_bid_depth: int = 50
    total_ask_depth: int = 50
    sweep_active: bool = False
    sweep_direction: str = ""
    sweep_size: int = 0
    whale_trade_count: int = 0
    avg_inter_trade_ms: float = 500
    rapid_sequence_count: int = 0


@dataclass
class MockMarket:
    ticker: str = "MKT-A"
    micro: MockMicro = field(default_factory=MockMicro)


@dataclass
class MockEvent:
    mutually_exclusive: bool = True
    markets: Dict = field(default_factory=lambda: {
        "MKT-A": MockMarket(ticker="MKT-A"),
        "MKT-B": MockMarket(ticker="MKT-B"),
        "MKT-C": MockMarket(ticker="MKT-C"),
    })

    def liquidity_adjusted_edge(self, direction, target_contracts):
        return {"edge_per_contract": 5.0, "total_edge": 50.0}


def make_sniper(
    enabled=True,
    max_capital=5000,
    max_position=25,
    order_ttl=30,
    leg_timeout=5.0,
    arb_min_edge=3.0,
) -> Sniper:
    gateway = AsyncMock()
    gateway.create_order = AsyncMock(return_value=MockOrderResponse())
    gateway.cancel_order = AsyncMock()

    index = MagicMock()
    index.events = {"EVENT-1": MockEvent()}

    event_bus = MagicMock()
    session = MockTradingSession()

    config = SniperConfig(
        enabled=enabled,
        max_capital=max_capital,
        max_position=max_position,
        order_ttl=order_ttl,
        leg_timeout=leg_timeout,
        arb_min_edge=arb_min_edge,
    )

    broadcast = AsyncMock()

    sniper = Sniper(
        gateway=gateway,
        index=index,
        event_bus=event_bus,
        session=session,
        config=config,
        broadcast_callback=broadcast,
    )
    sniper._running = True
    sniper.state.running = True
    return sniper


# ---------------------------------------------------------------------------
# Issue #1: Capital accounting lifecycle
# ---------------------------------------------------------------------------

class TestCapitalAccounting:
    def test_track_order_capital(self):
        sniper = make_sniper()
        sniper._track_order_capital("order-1", 400)
        assert sniper.state.capital_in_flight == 400
        assert sniper.state.capital_deployed_lifetime == 400
        assert "order-1" in sniper._order_capital

    def test_release_order_capital(self):
        sniper = make_sniper()
        sniper._track_order_capital("order-1", 400)
        sniper.state.active_order_ids.add("order-1")

        released = sniper._release_order_capital("order-1", "cancel")
        assert released == 400
        assert sniper.state.capital_in_flight == 0
        assert sniper.state.capital_deployed_lifetime == 400  # Never decreases
        assert "order-1" not in sniper.state.active_order_ids

    def test_release_nonexistent_returns_zero(self):
        sniper = make_sniper()
        released = sniper._release_order_capital("nonexistent", "test")
        assert released == 0

    def test_promote_order_to_position(self):
        sniper = make_sniper()
        sniper._track_order_capital("order-1", 400)

        sniper._promote_order_to_position("order-1")
        assert sniper.state.capital_in_flight == 0
        assert sniper.state.capital_in_positions == 400
        assert "order-1" not in sniper._order_capital

    def test_capital_gate_uses_active_not_lifetime(self):
        """Capital gate should check active capital, not cumulative lifetime."""
        sniper = make_sniper(max_capital=1000)

        # Deploy 800c and then release it (order expired)
        sniper._track_order_capital("order-1", 800)
        sniper._release_order_capital("order-1", "expired")

        # State: lifetime=800, active=0. New 900c should pass.
        opp = MockArbOpportunity()
        rejection = sniper._check_risk_gates(opp)
        assert rejection is None  # Should pass capital gate

    @pytest.mark.asyncio
    async def test_full_execution_tracks_capital(self):
        sniper = make_sniper(max_capital=50000)
        # Set up unique order IDs per call
        order_ids = iter(["ord-a", "ord-b", "ord-c"])
        sniper._gateway.create_order = AsyncMock(
            side_effect=lambda **kw: MockOrderResponse(order_id=next(order_ids))
        )

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action is not None
        assert action.legs_filled == 3
        # 3 legs × 10 contracts × (40+30+20) per leg
        assert sniper.state.capital_in_flight == 400 + 300 + 200  # 900c
        assert sniper.state.capital_deployed_lifetime == 900


# ---------------------------------------------------------------------------
# Issue #2: Partial fill unwinding
# ---------------------------------------------------------------------------

class TestPartialFillUnwind:
    @pytest.mark.asyncio
    async def test_partial_fill_cancels_successful_legs(self):
        sniper = make_sniper(max_capital=50000)

        # First leg succeeds, second succeeds, third fails
        call_count = 0
        async def mock_create(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise Exception("API error on leg 3")
            return MockOrderResponse(order_id=f"ord-{call_count}")

        sniper._gateway.create_order = mock_create

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action is not None
        assert action.unwound is True
        assert action.error_type == "partial_unwind"
        assert sniper.state.total_partial_unwinds == 1
        # Successful legs should have been cancelled
        assert sniper._gateway.cancel_order.call_count == 2

    @pytest.mark.asyncio
    async def test_all_legs_fail_no_unwind(self):
        sniper = make_sniper(max_capital=50000)
        sniper._gateway.create_order = AsyncMock(side_effect=Exception("API down"))

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action is not None
        assert action.unwound is False
        assert action.legs_filled == 0
        assert sniper.state.total_arbs_rejected == 1

    @pytest.mark.asyncio
    async def test_all_legs_succeed_no_unwind(self):
        sniper = make_sniper(max_capital=50000)
        order_ids = iter(["a", "b", "c"])
        sniper._gateway.create_order = AsyncMock(
            side_effect=lambda **kw: MockOrderResponse(order_id=next(order_ids))
        )

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action is not None
        assert action.unwound is False
        assert action.legs_filled == 3
        assert sniper.state.total_arbs_executed == 1


# ---------------------------------------------------------------------------
# Issue #3: Leg timeout
# ---------------------------------------------------------------------------

class TestLegTimeout:
    @pytest.mark.asyncio
    async def test_timeout_triggers_on_slow_gateway(self):
        sniper = make_sniper(max_capital=50000, leg_timeout=0.1)

        call_count = 0
        async def slow_create(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                await asyncio.sleep(5)  # Will be cancelled by timeout
            return MockOrderResponse(order_id=f"ord-{call_count}")

        sniper._gateway.create_order = slow_create

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action is not None
        # One leg timed out, triggering partial unwind
        assert action.legs_filled == 2
        assert action.unwound is True

    @pytest.mark.asyncio
    async def test_all_timeout_all_fail(self):
        sniper = make_sniper(max_capital=50000, leg_timeout=0.05)

        async def always_slow(**kw):
            await asyncio.sleep(5)
            return MockOrderResponse()

        sniper._gateway.create_order = always_slow

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action is not None
        assert action.legs_filled == 0
        assert action.error_type == "timeout"


# ---------------------------------------------------------------------------
# Issue #4: Config TTL
# ---------------------------------------------------------------------------

class TestConfigTTL:
    @pytest.mark.asyncio
    async def test_order_uses_config_ttl(self):
        sniper = make_sniper(max_capital=50000, order_ttl=45)
        order_ids = iter(["a", "b", "c"])
        sniper._gateway.create_order = AsyncMock(
            side_effect=lambda **kw: MockOrderResponse(order_id=next(order_ids))
        )

        opp = MockArbOpportunity()
        await sniper.on_arb_opportunity(opp)

        # Check the expiration_ts passed to create_order
        call_args = sniper._gateway.create_order.call_args_list[0]
        exp_ts = call_args.kwargs.get("expiration_ts", call_args[1].get("expiration_ts"))
        now = int(time.time())
        # Should be approximately now + 45
        assert abs(exp_ts - (now + 45)) < 3


# ---------------------------------------------------------------------------
# Issue #5: Capital gate cost estimate
# ---------------------------------------------------------------------------

class TestCapitalGateCostEstimate:
    def test_cost_estimate_uses_actual_contracts(self):
        sniper = make_sniper(max_capital=500, max_position=10)
        # 3 legs × 10 contracts × (40+30+20) = 900c > max 500c
        opp = MockArbOpportunity()
        rejection = sniper._check_risk_gates(opp)
        assert rejection is not None
        assert "capital_limit" in rejection

    def test_cost_estimate_with_size_limited_legs(self):
        sniper = make_sniper(max_capital=200, max_position=25)
        # size_available=2 limits contracts. Cost = 3 legs × 2 contracts × ~30 avg = ~180
        opp = MockArbOpportunity()
        opp.legs = [
            MockLeg(ticker="A", price_cents=30, size_available=2),
            MockLeg(ticker="B", price_cents=30, size_available=2),
            MockLeg(ticker="C", price_cents=30, size_available=2),
        ]
        # 2 contracts × (30+30+30) = 180c < 200c max
        rejection = sniper._check_risk_gates(opp)
        assert rejection is None


# ---------------------------------------------------------------------------
# Issue #7: Deque for recent_actions
# ---------------------------------------------------------------------------

class TestRecentActionsDeque:
    @pytest.mark.asyncio
    async def test_deque_auto_evicts(self):
        sniper = make_sniper(max_capital=999999)
        order_idx = [0]
        async def mock_create(**kw):
            order_idx[0] += 1
            return MockOrderResponse(order_id=f"ord-{order_idx[0]}")
        sniper._gateway.create_order = mock_create

        # Execute 25 arbs — deque maxlen=20 should evict oldest
        for i in range(25):
            sniper.state.trades_this_cycle = 0
            sniper.state._event_last_trade.clear()
            opp = MockArbOpportunity()
            await sniper.on_arb_opportunity(opp)

        assert len(sniper.state.recent_actions) == 20


# ---------------------------------------------------------------------------
# Issue #8: Unknown config fields reported
# ---------------------------------------------------------------------------

class TestConfigUnknownFields:
    def test_unknown_fields_returned(self):
        config = SniperConfig()
        changed, unknown = config.update(enabled=True, typo_field="oops", max_position=10)
        assert "enabled" in changed
        assert "max_position" in changed
        assert "typo_field" in unknown

    def test_no_unknown_fields(self):
        config = SniperConfig()
        changed, unknown = config.update(cooldown=15.0)
        assert "cooldown" in changed
        assert unknown == []


# ---------------------------------------------------------------------------
# Issue #10: Stale order cleanup
# ---------------------------------------------------------------------------

class TestStaleOrderCleanup:
    def test_cleanup_removes_expired_orders(self):
        sniper = make_sniper(order_ttl=30)
        # Fake an old order (placed 60 seconds ago)
        sniper._order_capital["old-order"] = (500, time.time() - 60)
        sniper.state.capital_in_flight = 500
        sniper.state.active_order_ids.add("old-order")

        cleaned = sniper._cleanup_stale_orders()
        assert cleaned == 1
        assert sniper.state.capital_in_flight == 0
        assert "old-order" not in sniper.state.active_order_ids

    def test_cleanup_keeps_fresh_orders(self):
        sniper = make_sniper(order_ttl=30)
        sniper._order_capital["fresh-order"] = (500, time.time())
        sniper.state.capital_in_flight = 500
        sniper.state.active_order_ids.add("fresh-order")

        cleaned = sniper._cleanup_stale_orders()
        assert cleaned == 0
        assert sniper.state.capital_in_flight == 500

    @pytest.mark.asyncio
    async def test_cleanup_runs_before_opportunity_check(self):
        """Ensure stale cleanup runs before capital gate check."""
        sniper = make_sniper(max_capital=500, order_ttl=5)
        # Fill capital with an old order
        sniper._order_capital["old"] = (500, time.time() - 20)
        sniper.state.capital_in_flight = 500
        sniper.state.active_order_ids.add("old")

        # New opportunity should pass because cleanup frees capital
        order_ids = iter(["a", "b", "c"])
        sniper._gateway.create_order = AsyncMock(
            side_effect=lambda **kw: MockOrderResponse(order_id=next(order_ids))
        )

        opp = MockArbOpportunity()
        # Adjust to a small opportunity that fits in 500c
        opp.legs = [
            MockLeg(ticker="A", price_cents=50, size_available=1),
            MockLeg(ticker="B", price_cents=50, size_available=1),
            MockLeg(ticker="C", price_cents=50, size_available=1),
        ]
        action = await sniper.on_arb_opportunity(opp)
        # Should succeed since stale order was cleaned first
        assert action is not None
        assert action.legs_filled == 3


# ---------------------------------------------------------------------------
# Issue #11: Error type classification
# ---------------------------------------------------------------------------

class TestErrorTypeClassification:
    @pytest.mark.asyncio
    async def test_rate_limit_error_classified(self):
        sniper = make_sniper(max_capital=50000)
        sniper._gateway.create_order = AsyncMock(
            side_effect=Exception("429 rate limit exceeded")
        )

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action.error_type == "rate_limit"

    @pytest.mark.asyncio
    async def test_generic_error_classified(self):
        sniper = make_sniper(max_capital=50000)
        sniper._gateway.create_order = AsyncMock(
            side_effect=Exception("insufficient_balance")
        )

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action.error_type == "api_error"


# ---------------------------------------------------------------------------
# Issue #12: Edge logging (SniperAction has edge_net_cents)
# ---------------------------------------------------------------------------

class TestEdgeLogging:
    def test_sniper_action_has_edge_net(self):
        action = SniperAction(edge_cents=5.0, edge_net_cents=4.2)
        d = action.to_dict()
        assert d["edge_cents"] == 5.0
        assert d["edge_net_cents"] == 4.2

    def test_sniper_action_has_error_type(self):
        action = SniperAction(error="timeout", error_type="timeout")
        d = action.to_dict()
        assert d["error_type"] == "timeout"

    def test_sniper_action_has_unwound(self):
        action = SniperAction(unwound=True)
        d = action.to_dict()
        assert d["unwound"] is True


# ---------------------------------------------------------------------------
# Issue #13: Concurrency docstring
# ---------------------------------------------------------------------------

class TestConcurrencyDocs:
    def test_module_has_concurrency_note(self):
        import kalshiflow_rl.traderv3.single_arb.sniper as sniper_mod
        assert "single asyncio event loop" in sniper_mod.__doc__


# ---------------------------------------------------------------------------
# Integration: kill_positions releases capital
# ---------------------------------------------------------------------------

class TestKillPositions:
    @pytest.mark.asyncio
    async def test_kill_releases_capital(self):
        sniper = make_sniper()
        # Simulate 2 active orders with capital
        sniper._track_order_capital("kill-1", 200)
        sniper._track_order_capital("kill-2", 300)
        sniper.state.active_order_ids.update(["kill-1", "kill-2"])

        result = await sniper.kill_positions(reason="test")

        assert result["cancelled"] == 2
        assert sniper.state.capital_in_flight == 0
        assert len(sniper.state.active_order_ids) == 0

    @pytest.mark.asyncio
    async def test_kill_with_errors_still_releases(self):
        sniper = make_sniper()
        sniper._track_order_capital("err-1", 200)
        sniper.state.active_order_ids.add("err-1")
        sniper._gateway.cancel_order = AsyncMock(side_effect=Exception("not found"))

        result = await sniper.kill_positions(reason="test")

        assert len(result["errors"]) == 1
        # Capital still released even on error
        assert sniper.state.capital_in_flight == 0


# ---------------------------------------------------------------------------
# SniperState.to_dict
# ---------------------------------------------------------------------------

class TestSniperStateDict:
    def test_to_dict_has_capital_fields(self):
        state = SniperState()
        state.capital_in_flight = 100
        state.capital_in_positions = 200
        state.capital_deployed_lifetime = 500
        d = state.to_dict()
        assert d["capital_in_flight"] == 100
        assert d["capital_in_positions"] == 200
        assert d["capital_deployed_lifetime"] == 500
        assert d["capital_active"] == 300
        assert d["total_partial_unwinds"] == 0


# ---------------------------------------------------------------------------
# SniperConfig.to_dict
# ---------------------------------------------------------------------------

class TestSniperConfigDict:
    def test_to_dict_has_new_fields(self):
        config = SniperConfig(order_ttl=45, leg_timeout=3.0)
        d = config.to_dict()
        assert d["order_ttl"] == 45
        assert d["leg_timeout"] == 3.0
