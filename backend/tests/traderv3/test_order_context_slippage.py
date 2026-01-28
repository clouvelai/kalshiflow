"""
Tests for slippage calculation in order context.

Slippage should be calculated as: fill_price - order_price
This measures execution quality (order-to-fill), not signal-to-fill.
"""

import pytest
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestSlippageCalculation:
    """Tests for slippage calculation in order context."""

    def test_slippage_positive_when_filled_above_order_price(self):
        """Positive slippage = paid more than order price (bad)."""
        from kalshiflow_rl.traderv3.state.order_context import (
            StagedOrderContext,
        )

        context = StagedOrderContext(
            order_id="test-001",
            market_ticker="TEST-TICKER",
            side="no",
            order_price_cents=35,  # We submitted order at 35c
            order_quantity=10,
        )

        # Fill at 37c (2c worse than order price)
        db_dict = context.to_db_dict(
            fill_count=10,
            fill_avg_price_cents=37,
            filled_at=time.time()
        )

        assert db_dict["slippage_cents"] == 2, "Should be +2c slippage (bad fill)"

    def test_slippage_negative_when_filled_below_order_price(self):
        """Negative slippage = paid less than order price (good, price improved)."""
        from kalshiflow_rl.traderv3.state.order_context import (
            StagedOrderContext,
        )

        context = StagedOrderContext(
            order_id="test-002",
            market_ticker="TEST-TICKER",
            side="no",
            order_price_cents=35,  # We submitted order at 35c
            order_quantity=10,
        )

        # Fill at 33c (2c better than order price - price improvement!)
        db_dict = context.to_db_dict(
            fill_count=10,
            fill_avg_price_cents=33,
            filled_at=time.time()
        )

        assert db_dict["slippage_cents"] == -2, "Should be -2c slippage (good fill, price improved)"

    def test_slippage_zero_when_filled_at_order_price(self):
        """Zero slippage = filled exactly at order price."""
        from kalshiflow_rl.traderv3.state.order_context import (
            StagedOrderContext,
        )

        context = StagedOrderContext(
            order_id="test-003",
            market_ticker="TEST-TICKER",
            side="no",
            order_price_cents=35,
            order_quantity=10,
        )

        # Fill at exactly 35c
        db_dict = context.to_db_dict(
            fill_count=10,
            fill_avg_price_cents=35,
            filled_at=time.time()
        )

        assert db_dict["slippage_cents"] == 0, "Should be 0c slippage (exact fill)"

    def test_slippage_none_when_order_price_missing(self):
        """Slippage should be None if order_price_cents is None."""
        from kalshiflow_rl.traderv3.state.order_context import (
            StagedOrderContext,
        )

        context = StagedOrderContext(
            order_id="test-004",
            market_ticker="TEST-TICKER",
            side="no",
            order_price_cents=None,  # No order price recorded
            order_quantity=10,
        )

        db_dict = context.to_db_dict(
            fill_count=10,
            fill_avg_price_cents=35,
            filled_at=time.time()
        )

        assert db_dict["slippage_cents"] is None, "Slippage should be None when order price unknown"

    def test_slippage_none_when_fill_price_missing(self):
        """Slippage should be None if fill_avg_price_cents is None."""
        from kalshiflow_rl.traderv3.state.order_context import (
            StagedOrderContext,
        )

        context = StagedOrderContext(
            order_id="test-005",
            market_ticker="TEST-TICKER",
            side="no",
            order_price_cents=35,
            order_quantity=10,
        )

        db_dict = context.to_db_dict(
            fill_count=10,
            fill_avg_price_cents=None,  # No fill price known
            filled_at=time.time()
        )

        assert db_dict["slippage_cents"] is None, "Slippage should be None when fill price unknown"

    def test_slippage_works_for_yes_side(self):
        """Slippage calculation should work the same for YES side orders."""
        from kalshiflow_rl.traderv3.state.order_context import (
            StagedOrderContext,
        )

        context = StagedOrderContext(
            order_id="test-006",
            market_ticker="TEST-TICKER",
            side="yes",  # YES side
            order_price_cents=65,
            order_quantity=10,
        )

        # Fill at 67c (2c worse)
        db_dict = context.to_db_dict(
            fill_count=10,
            fill_avg_price_cents=67,
            filled_at=time.time()
        )

        assert db_dict["slippage_cents"] == 2, "YES side should also have +2c slippage"
