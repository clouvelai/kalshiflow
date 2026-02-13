"""Unit tests for SingleArbCoordinator._register_sniper_order warning log.

Verifies that the method logs a warning when ToolContext is not yet wired,
instead of silently returning.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.single_arb.coordinator import SingleArbCoordinator


def make_coordinator():
    """Create a minimal coordinator for testing _register_sniper_order."""
    config = MagicMock()
    event_bus = MagicMock()
    ws_manager = MagicMock()
    ob_integration = MagicMock()

    coord = SingleArbCoordinator(
        config=config,
        event_bus=event_bus,
        websocket_manager=ws_manager,
        orderbook_integration=ob_integration,
    )
    # Wire a mock trading session (needed by _register_sniper_order)
    coord._trading_session = MagicMock()
    coord._trading_session.captain_order_ids = set()
    return coord


class TestRegisterSniperOrderWarning:
    def test_logs_warning_when_ctx_none(self, caplog):
        """_register_sniper_order should log a warning when get_context() returns None."""
        coord = make_coordinator()

        with patch(
            "kalshiflow_rl.traderv3.single_arb.tools.get_context",
            return_value=None,
        ), caplog.at_level(logging.WARNING, logger="kalshiflow_rl.traderv3.single_arb.coordinator"):
            coord._register_sniper_order(
                order_id="order-abc-12345678",
                ticker="MKT-TEST",
                side="yes",
                action="buy",
                contracts=5,
                price_cents=40,
                ttl_seconds=30,
            )

        # Verify warning was logged with relevant info
        assert any("ToolContext not ready" in record.message for record in caplog.records)
        assert any("order-ab" in record.message for record in caplog.records)
        assert any("MKT-TEST" in record.message for record in caplog.records)

    def test_no_warning_when_ctx_available(self, caplog):
        """_register_sniper_order should not warn when ToolContext is available."""
        coord = make_coordinator()

        mock_ctx = MagicMock()
        mock_ctx.captain_order_ids = set()
        mock_ctx.order_initial_states = {}

        with patch(
            "kalshiflow_rl.traderv3.single_arb.tools.get_context",
            return_value=mock_ctx,
        ), caplog.at_level(logging.WARNING, logger="kalshiflow_rl.traderv3.single_arb.coordinator"):
            coord._register_sniper_order(
                order_id="order-def-12345678",
                ticker="MKT-TEST",
                side="yes",
                action="buy",
                contracts=5,
                price_cents=40,
                ttl_seconds=30,
            )

        # No warning should be logged
        assert not any("ToolContext not ready" in record.message for record in caplog.records)
        # Order should be registered
        assert "order-def-12345678" in mock_ctx.captain_order_ids
