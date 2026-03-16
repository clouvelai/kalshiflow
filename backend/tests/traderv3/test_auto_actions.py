"""Unit tests for AutoActionManager.

T1/T2 tests — pure Python + async. No network calls.
Tests trigger conditions, execution, and Captain override.
"""

import asyncio
import time

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.traderv3.conftest import make_event_meta, make_index, make_market_meta, make_config

from kalshiflow_rl.traderv3.single_arb.auto_actions import (
    AutoActionManager,
    AutoActionResult,
    AutoActionConfig,
)
from kalshiflow_rl.traderv3.single_arb.models import AttentionItem


# ===========================================================================
# Helpers
# ===========================================================================

def _mock_order_response(order_id="auto-order-001"):
    """Create a mock OrderResponse matching gateway.create_order() return type."""
    resp = MagicMock()
    resp.order.order_id = order_id
    resp.order.status = "resting"
    return resp


def _mock_position(ticker="EVT-1-MKT-A", position=10):
    """Create a mock Position matching gateway.get_positions() return type."""
    pos = MagicMock()
    pos.ticker = ticker
    pos.position = position
    return pos


def make_manager(events=None, sniper=None):
    """Create an AutoActionManager with mock gateway and pre-populated index."""
    if events is None:
        events = [make_event_meta("EVT-1", n_markets=3)]
    index = make_index(events=events)
    gateway = AsyncMock()
    gateway.create_order = AsyncMock(return_value=_mock_order_response())
    gateway.get_positions = AsyncMock(return_value=[_mock_position()])
    return AutoActionManager(
        gateway=gateway,
        index=index,
        sniper=sniper,
        config=make_config(),
    )


def make_position_risk_item(
    ticker="EVT-1-MKT-A",
    event_ticker="EVT-1",
    pnl_per_ct=-15,
    side="yes",
    quantity=10,
    **data_overrides,
):
    """Create an AttentionItem for position_risk category."""
    data = {
        "ticker": ticker,
        "side": side,
        "quantity": quantity,
        "pnl_per_contract": pnl_per_ct,
    }
    data.update(data_overrides)
    return AttentionItem(
        event_ticker=event_ticker,
        market_ticker=ticker,
        urgency="high",
        category="position_risk",
        score=60.0,
        summary=f"pnl={pnl_per_ct:+d}c/ct",
        data=data,
    )


def make_settlement_item(
    ticker="EVT-1-MKT-A",
    event_ticker="EVT-1",
    ttc_hours=0.3,
    side="yes",
    quantity=10,
):
    """Create an AttentionItem for settlement_approaching category."""
    return AttentionItem(
        event_ticker=event_ticker,
        market_ticker=ticker,
        urgency="high",
        category="settlement_approaching",
        score=55.0,
        summary=f"ttc={ttc_hours:.1f}h",
        data={
            "ticker": ticker,
            "side": side,
            "quantity": quantity,
            "ttc_hours": ttc_hours,
        },
    )


def make_regime_item(event_ticker="EVT-1", regime="toxic"):
    """Create an AttentionItem for regime_change category."""
    return AttentionItem(
        event_ticker=event_ticker,
        urgency="high",
        category="regime_change",
        score=50.0,
        summary=f"regime={regime}",
        data={"regime": regime},
    )


# ===========================================================================
# TestStopLoss
# ===========================================================================

class TestStopLoss:
    @pytest.mark.asyncio
    async def test_fires_when_pnl_below_threshold(self):
        """Stop loss should fire when P&L crosses threshold."""
        mgr = make_manager()
        item = make_position_risk_item(pnl_per_ct=-15)
        await mgr.on_attention_item(item)
        assert item.data.get("auto_handled")
        assert "stop_loss" in item.data["auto_handled"]

    @pytest.mark.asyncio
    async def test_does_not_fire_above_threshold(self):
        """Stop loss should not fire when P&L is above threshold."""
        mgr = make_manager()
        item = make_position_risk_item(pnl_per_ct=-5)
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_respects_per_ticker_threshold_override(self):
        """Per-ticker threshold override should be respected."""
        mgr = make_manager()
        # Set a tighter threshold for this ticker
        mgr.configure("stop_loss", {"ticker": "EVT-1-MKT-A", "threshold": -8})

        # -10 should fire with -8 override but not with default -12
        item = make_position_risk_item(pnl_per_ct=-10)
        await mgr.on_attention_item(item)
        assert item.data.get("auto_handled")

    @pytest.mark.asyncio
    async def test_disabled_does_not_fire(self):
        """Disabled stop loss should not fire."""
        mgr = make_manager()
        mgr.configure("stop_loss", {"enabled": False})
        item = make_position_risk_item(pnl_per_ct=-20)
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_places_sell_order(self):
        """Stop loss should place a sell order via gateway."""
        mgr = make_manager()
        item = make_position_risk_item(pnl_per_ct=-15)
        await mgr.on_attention_item(item)
        mgr._gateway.create_order.assert_called_once()
        call_kwargs = mgr._gateway.create_order.call_args[1]
        assert call_kwargs["ticker"] == "EVT-1-MKT-A"
        assert call_kwargs["action"] == "sell"

    @pytest.mark.asyncio
    async def test_missing_ticker_does_not_fire(self):
        """Item without ticker data should not fire."""
        mgr = make_manager()
        item = AttentionItem(
            event_ticker="EVT-1",
            urgency="high",
            category="position_risk",
            score=60.0,
            summary="test",
            data={"pnl_per_contract": -15},  # Missing ticker, side, quantity
        )
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data


# ===========================================================================
# TestTimeExit
# ===========================================================================

class TestTimeExit:
    @pytest.mark.asyncio
    async def test_fires_when_close_to_settlement(self):
        """Time exit should fire when event is close to settlement."""
        mgr = make_manager()
        item = make_settlement_item(ttc_hours=0.3)
        await mgr.on_attention_item(item)
        assert item.data.get("auto_handled")
        assert "time_exit" in item.data["auto_handled"]

    @pytest.mark.asyncio
    async def test_does_not_fire_far_from_settlement(self):
        """Time exit should not fire when event is far from settlement."""
        mgr = make_manager()
        item = make_settlement_item(ttc_hours=2.0)
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_hold_through_override(self):
        """Per-event hold_through override should prevent exit."""
        mgr = make_manager()
        mgr.configure("time_exit", {"event": "EVT-1", "hold_through": True})
        item = make_settlement_item(ttc_hours=0.1)
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_disabled_does_not_fire(self):
        """Disabled time exit should not fire."""
        mgr = make_manager()
        mgr.configure("time_exit", {"enabled": False})
        item = make_settlement_item(ttc_hours=0.1)
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data


# ===========================================================================
# TestRegimeGate
# ===========================================================================

class TestRegimeGate:
    @pytest.mark.asyncio
    async def test_fires_on_toxic_regime(self):
        """Regime gate should fire when regime becomes toxic."""
        sniper = MagicMock()
        mgr = make_manager(sniper=sniper)
        item = make_regime_item(regime="toxic")
        await mgr.on_attention_item(item)
        assert item.data.get("auto_handled")
        assert "regime_gate" in item.data["auto_handled"]

    @pytest.mark.asyncio
    async def test_does_not_fire_on_normal_regime(self):
        """Regime gate should not fire on non-toxic regime."""
        sniper = MagicMock()
        mgr = make_manager(sniper=sniper)
        item = make_regime_item(regime="normal")
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_cooldown_prevents_repeated_firing(self):
        """Regime gate should respect cooldown."""
        sniper = MagicMock()
        mgr = make_manager(sniper=sniper)
        mgr._regime_gate_cooldown = 60.0  # 60s cooldown

        item1 = make_regime_item(regime="toxic")
        await mgr.on_attention_item(item1)
        assert item1.data.get("auto_handled")

        # Second item should be within cooldown
        item2 = make_regime_item(regime="toxic")
        await mgr.on_attention_item(item2)
        assert "auto_handled" not in item2.data

    @pytest.mark.asyncio
    async def test_no_sniper_does_not_fire(self):
        """Regime gate without sniper should not fire."""
        mgr = make_manager(sniper=None)
        item = make_regime_item(regime="toxic")
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_ignore_regime_override(self):
        """Per-event ignore_regime override should prevent gating."""
        sniper = MagicMock()
        mgr = make_manager(sniper=sniper)
        mgr.configure("regime_gate", {"event": "EVT-1", "ignore_regime": True})
        item = make_regime_item(regime="toxic")
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data


# ===========================================================================
# TestAutoActionConfiguration
# ===========================================================================

class TestAutoActionConfiguration:
    def test_configure_stop_loss_threshold(self):
        mgr = make_manager()
        result = mgr.configure("stop_loss", {"threshold": -8})
        assert result["threshold"] == -8
        assert mgr._stop_loss_threshold == -8

    def test_configure_time_exit_threshold(self):
        mgr = make_manager()
        result = mgr.configure("time_exit", {"threshold_minutes": 15})
        assert result["threshold_minutes"] == 15

    def test_configure_regime_gate_cooldown(self):
        mgr = make_manager()
        result = mgr.configure("regime_gate", {"cooldown": 600.0})
        assert result["cooldown"] == 600.0

    def test_configure_enable_disable(self):
        mgr = make_manager()
        mgr.configure("stop_loss", {"enabled": False})
        assert not mgr._stop_loss.enabled
        mgr.configure("stop_loss", {"enabled": True})
        assert mgr._stop_loss.enabled

    def test_configure_per_ticker_override(self):
        mgr = make_manager()
        mgr.configure("stop_loss", {"ticker": "MKT-X", "threshold": -5})
        assert mgr._stop_loss.overrides["MKT-X"]["threshold"] == -5

    def test_configure_unknown_action(self):
        mgr = make_manager()
        result = mgr.configure("unknown_action", {})
        assert "error" in result

    def test_get_config_all(self):
        mgr = make_manager()
        config = mgr.get_config()
        assert "stop_loss" in config
        assert "time_exit" in config
        assert "regime_gate" in config

    def test_get_config_single(self):
        mgr = make_manager()
        config = mgr.get_config("stop_loss")
        assert "enabled" in config
        assert "threshold" in config

    def test_get_stats(self):
        mgr = make_manager()
        stats = mgr.get_stats()
        assert "config" in stats
        assert "recent_actions" in stats


# ===========================================================================
# TestAutoActionLogging
# ===========================================================================

class TestAutoActionLogging:
    @pytest.mark.asyncio
    async def test_actions_logged(self):
        """Fired actions should be logged."""
        mgr = make_manager()
        item = make_position_risk_item(pnl_per_ct=-15)
        await mgr.on_attention_item(item)
        assert len(mgr._action_log) == 1
        assert mgr._action_log[0]["action"] == "stop_loss"


# ===========================================================================
# TestPositionAlreadyClosed
# ===========================================================================

class TestPositionAlreadyClosed:
    @pytest.mark.asyncio
    async def test_position_already_closed_skips_exit(self):
        """Exit should not fire if the position no longer exists (race with Captain)."""
        mgr = make_manager()
        # Gateway returns no matching position (Captain already exited)
        mgr._gateway.get_positions.return_value = []

        item = make_position_risk_item(pnl_per_ct=-15)
        await mgr.on_attention_item(item)
        # Should not attempt to place an order
        mgr._gateway.create_order.assert_not_called()
        # auto_handled should NOT be set since no action was taken
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_position_zero_quantity_skips_exit(self):
        """Exit should not fire if position exists but has zero contracts."""
        mgr = make_manager()
        mgr._gateway.get_positions.return_value = [_mock_position(ticker="EVT-1-MKT-A", position=0)]

        item = make_position_risk_item(pnl_per_ct=-15)
        await mgr.on_attention_item(item)
        mgr._gateway.create_order.assert_not_called()
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_position_exists_proceeds_with_exit(self):
        """Exit should proceed normally when position still exists."""
        mgr = make_manager()
        # Ensure position exists
        mgr._gateway.get_positions.return_value = [_mock_position(ticker="EVT-1-MKT-A", position=10)]

        item = make_position_risk_item(pnl_per_ct=-15)
        await mgr.on_attention_item(item)
        mgr._gateway.create_order.assert_called_once()
        assert item.data.get("auto_handled")


# ===========================================================================
# TestOrderIdValidation
# ===========================================================================

class TestOrderIdValidation:
    @pytest.mark.asyncio
    async def test_empty_order_id_returns_error(self):
        """Exit should return error when create_order returns no order_id."""
        mgr = make_manager()
        # Override gateway to return empty order_id
        mgr._gateway.create_order.return_value = _mock_order_response(order_id="")

        item = make_position_risk_item(pnl_per_ct=-15)
        await mgr.on_attention_item(item)
        # auto_handled should NOT be set because the exit order failed
        assert "auto_handled" not in item.data

    @pytest.mark.asyncio
    async def test_missing_order_key_returns_error(self):
        """Exit should return error when create_order raises an exception."""
        mgr = make_manager()
        mgr._gateway.create_order.side_effect = Exception("API error")

        item = make_position_risk_item(pnl_per_ct=-15)
        await mgr.on_attention_item(item)
        assert "auto_handled" not in item.data


# ===========================================================================
# TestRegimeGatePausesSniper
# ===========================================================================

class TestRegimeGatePausesSniper:
    @pytest.mark.asyncio
    async def test_regime_gate_pauses_sniper(self):
        """Regime gate should call sniper.pause() on toxic regime."""
        sniper = MagicMock()
        sniper.pause = MagicMock()
        sniper.resume = MagicMock()
        mgr = make_manager(sniper=sniper)

        item = make_regime_item(regime="toxic")
        await mgr.on_attention_item(item)

        sniper.pause.assert_called_once()
        assert "toxic" in sniper.pause.call_args[0][0]

    @pytest.mark.asyncio
    async def test_regime_gate_resumes_sniper_after_cooldown(self):
        """Regime gate should resume sniper after cooldown expires."""
        sniper = MagicMock()
        sniper.pause = MagicMock()
        sniper.resume = MagicMock()
        mgr = make_manager(sniper=sniper)
        mgr._regime_gate_cooldown = 0.1  # Very short cooldown for test

        item = make_regime_item(regime="toxic")
        await mgr.on_attention_item(item)

        sniper.pause.assert_called_once()
        # Wait for the resume task to fire
        await asyncio.sleep(0.2)
        sniper.resume.assert_called_once()


    @pytest.mark.asyncio
    async def test_log_capped_at_50(self):
        """Action log should be capped at 50 entries."""
        mgr = make_manager()
        for i in range(55):
            item = make_position_risk_item(
                pnl_per_ct=-15,
                ticker=f"MKT-{i}",
            )
            # Reset the mock for each call and return matching position
            mgr._gateway.create_order.reset_mock()
            mgr._gateway.create_order.return_value = _mock_order_response(order_id=f"order-{i}")
            mgr._gateway.get_positions.return_value = [_mock_position(ticker=f"MKT-{i}")]
            await mgr.on_attention_item(item)
        assert len(mgr._action_log) <= 50
