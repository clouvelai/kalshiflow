"""Unit tests for Sniper execution layer.

Tests cover:
  1. Balance-based capital gating (real API balance, portfolio limit, API failure)
  2. Partial fill unwinding (cancel successful legs on partial)
  3. Leg timeout (asyncio.wait_for wrapping)
  4. Config TTL (order_ttl from SniperConfig)
  5. Capital gate cost estimate (uses actual contracts_per_leg)
  6. Broadcast fire-and-forget (asyncio.create_task)
  7. Deque for recent_actions (auto-eviction)
  8. Unknown config fields reported
  9. VPIN threshold from config (tested via tools separately)
  10. Error type classification
  11. Edge logging (edge_net_cents in SniperAction)
  12. Concurrency docstring (verified by existence)
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
class MockBalance:
    balance: int = 100_000  # 100_000c = $1000
    portfolio_value: int = 0


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
    gateway.get_balance = AsyncMock(return_value=MockBalance())

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
# Balance-based capital gating
# ---------------------------------------------------------------------------

class TestBalanceGating:
    @pytest.mark.asyncio
    async def test_sufficient_balance_passes(self):
        """With enough balance, capital gate should pass."""
        sniper = make_sniper(max_capital=50000)
        sniper._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=100_000, portfolio_value=0))

        opp = MockArbOpportunity()
        rejection = await sniper._check_risk_gates(opp)
        assert rejection is None

    @pytest.mark.asyncio
    async def test_insufficient_balance_rejects(self):
        """If available balance < estimated cost, reject."""
        sniper = make_sniper(max_capital=50000)
        # Only 100c available, but est_cost = 3 legs × 10 contracts × (40+30+20) = 900c
        sniper._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=100, portfolio_value=0))

        opp = MockArbOpportunity()
        rejection = await sniper._check_risk_gates(opp)
        assert rejection is not None
        assert "insufficient_balance" in rejection

    @pytest.mark.asyncio
    async def test_capital_active_plus_est_exceeds_max_rejects(self):
        """If capital_active + est_cost > max_capital, reject."""
        sniper = make_sniper(max_capital=5000)
        sniper._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=100_000, portfolio_value=0))
        # Set capital_active to 4200c. Est cost = 3 legs x 10 x (40+30+20) = 900c.
        # 4200 + 900 = 5100 > 5000 → should reject
        sniper.state.capital_active = 4200

        opp = MockArbOpportunity()
        rejection = await sniper._check_risk_gates(opp)
        assert rejection is not None
        assert "sniper_capital_limit" in rejection

    @pytest.mark.asyncio
    async def test_capital_active_plus_est_within_max_passes(self):
        """If capital_active + est_cost <= max_capital, pass."""
        sniper = make_sniper(max_capital=5000)
        sniper._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=100_000, portfolio_value=0))
        # Set capital_active to 4000c. Est cost = 900c. 4000 + 900 = 4900 <= 5000 → should pass
        sniper.state.capital_active = 4000

        opp = MockArbOpportunity()
        rejection = await sniper._check_risk_gates(opp)
        assert rejection is None

    @pytest.mark.asyncio
    async def test_zero_capital_active_passes(self):
        """Fresh session with no capital deployed should pass."""
        sniper = make_sniper(max_capital=5000)
        sniper._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=100_000, portfolio_value=8000))
        # portfolio_value is high but irrelevant now; capital_active=0, est=900 <= 5000

        opp = MockArbOpportunity()
        rejection = await sniper._check_risk_gates(opp)
        assert rejection is None

    @pytest.mark.asyncio
    async def test_balance_api_failure_rejects_conservatively(self):
        """If balance API fails, reject conservatively."""
        sniper = make_sniper(max_capital=50000)
        sniper._gateway.get_balance = AsyncMock(side_effect=Exception("API down"))

        opp = MockArbOpportunity()
        rejection = await sniper._check_risk_gates(opp)
        assert rejection is not None
        assert "balance_api_error" in rejection

    @pytest.mark.asyncio
    async def test_balance_telemetry_populated(self):
        """Risk gate check should populate balance telemetry on state."""
        sniper = make_sniper(max_capital=50000)
        sniper._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=55_000, portfolio_value=1200))

        opp = MockArbOpportunity()
        await sniper._check_risk_gates(opp)

        assert sniper.state.last_balance_cents == 55_000
        assert sniper.state.last_portfolio_value_cents == 1200

    @pytest.mark.asyncio
    async def test_full_execution_tracks_capital(self):
        """Full execution should increment both capital_active and capital_deployed_lifetime."""
        sniper = make_sniper(max_capital=50000)
        order_ids = iter(["ord-a", "ord-b", "ord-c"])
        sniper._gateway.create_order = AsyncMock(
            side_effect=lambda **kw: MockOrderResponse(order_id=next(order_ids))
        )

        opp = MockArbOpportunity()
        action = await sniper.on_arb_opportunity(opp)

        assert action is not None
        assert action.legs_filled == 3
        # 3 legs × 10 contracts × (40+30+20) = 900c
        assert sniper.state.capital_active == 900
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
    @pytest.mark.asyncio
    async def test_insufficient_balance_for_estimated_cost(self):
        sniper = make_sniper(max_capital=50000, max_position=10)
        # 3 legs × 10 contracts × (40+30+20) = 900c. Balance only 500c.
        sniper._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=500, portfolio_value=0))
        opp = MockArbOpportunity()
        rejection = await sniper._check_risk_gates(opp)
        assert rejection is not None
        assert "insufficient_balance" in rejection

    @pytest.mark.asyncio
    async def test_cost_estimate_with_size_limited_legs(self):
        sniper = make_sniper(max_capital=50000, max_position=25)
        # size_available=2 limits contracts. Cost = 2 × (30+30+30) = 180c
        sniper._gateway.get_balance = AsyncMock(return_value=MockBalance(balance=200, portfolio_value=0))
        opp = MockArbOpportunity()
        opp.legs = [
            MockLeg(ticker="A", price_cents=30, size_available=2),
            MockLeg(ticker="B", price_cents=30, size_available=2),
            MockLeg(ticker="C", price_cents=30, size_available=2),
        ]
        # 180c < 200c balance, should pass
        rejection = await sniper._check_risk_gates(opp)
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
    async def test_kill_cancels_active_orders(self):
        sniper = make_sniper()
        sniper.state.active_order_ids.update(["kill-1", "kill-2"])

        result = await sniper.kill_positions(reason="test")

        assert result["cancelled"] == 2
        assert len(sniper.state.active_order_ids) == 0

    @pytest.mark.asyncio
    async def test_kill_with_errors_still_clears_order_ids(self):
        sniper = make_sniper()
        sniper.state.active_order_ids.add("err-1")
        sniper._gateway.cancel_order = AsyncMock(side_effect=Exception("not found"))

        result = await sniper.kill_positions(reason="test")

        assert len(result["errors"]) == 1
        assert len(sniper.state.active_order_ids) == 0


# ---------------------------------------------------------------------------
# SniperState.to_dict
# ---------------------------------------------------------------------------

class TestSniperStateDict:
    def test_to_dict_has_balance_fields(self):
        state = SniperState()
        state.capital_active = 300
        state.capital_deployed_lifetime = 500
        state.last_balance_cents = 90_000
        state.last_portfolio_value_cents = 1200
        d = state.to_dict()
        assert d["capital_active"] == 300
        assert d["capital_deployed_lifetime"] == 500
        assert d["last_balance_cents"] == 90_000
        assert d["last_portfolio_value_cents"] == 1200
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


# ---------------------------------------------------------------------------
# kill_positions capital reset (Gap 2)
# ---------------------------------------------------------------------------


class TestKillPositionsCapitalReset:
    @pytest.mark.asyncio
    async def test_kill_positions_resets_capital_active(self):
        """kill_positions() should reset capital_active to 0."""
        gateway = MagicMock()
        gateway.cancel_order = AsyncMock(return_value={})
        index = MagicMock()
        event_bus = MagicMock()
        session = MockTradingSession()
        config = SniperConfig(enabled=True)

        sniper = Sniper(gateway=gateway, index=index, event_bus=event_bus,
                        session=session, config=config)
        sniper.state.capital_active = 5000
        sniper.state.active_order_ids = {"ord-1", "ord-2"}

        result = await sniper.kill_positions(reason="test")
        assert sniper.state.capital_active == 0
        assert result["cancelled"] == 2

    @pytest.mark.asyncio
    async def test_kill_positions_resets_capital_even_on_errors(self):
        """capital_active resets to 0 even if cancels fail."""
        gateway = MagicMock()
        gateway.cancel_order = AsyncMock(side_effect=Exception("API error"))
        index = MagicMock()
        event_bus = MagicMock()
        session = MockTradingSession()
        config = SniperConfig(enabled=True)

        sniper = Sniper(gateway=gateway, index=index, event_bus=event_bus,
                        session=session, config=config)
        sniper.state.capital_active = 3000
        sniper.state.active_order_ids = {"ord-1"}

        result = await sniper.kill_positions(reason="test")
        assert sniper.state.capital_active == 0
        assert result["cancelled"] == 0
        assert len(result["errors"]) == 1
