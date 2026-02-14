"""Unit tests for inventory_manager.py.

Pure T1 tests — no async, no mocking, no network.
"""

import pytest

from kalshiflow_rl.traderv3.market_maker.inventory_manager import (
    check_event_exposure,
    check_position_limit,
    compute_skew,
    compute_unrealized_pnl,
    should_one_side_only,
)


class TestComputeSkew:
    """Tests for compute_skew()."""

    def test_flat_position_no_skew(self):
        assert compute_skew(0, 0.5, 5.0) == 0.0

    def test_long_position_negative_skew(self):
        # Long 10 → skew = -10 * 0.5 = -5.0
        skew = compute_skew(10, 0.5, 5.0)
        assert skew == -5.0

    def test_short_position_positive_skew(self):
        # Short 10 → skew = -(-10) * 0.5 = 5.0
        skew = compute_skew(-10, 0.5, 5.0)
        assert skew == 5.0

    def test_skew_capped_positive(self):
        # Very short position → skew capped at skew_cap
        skew = compute_skew(-100, 0.5, 5.0)
        assert skew == 5.0

    def test_skew_capped_negative(self):
        # Very long position → skew capped at -skew_cap
        skew = compute_skew(100, 0.5, 5.0)
        assert skew == -5.0

    def test_skew_proportional(self):
        s1 = compute_skew(5, 0.5, 10.0)
        s2 = compute_skew(10, 0.5, 10.0)
        assert abs(s2) == abs(s1) * 2

    def test_zero_skew_factor(self):
        assert compute_skew(50, 0.0, 5.0) == 0.0


class TestCheckPositionLimit:
    """Tests for check_position_limit()."""

    def test_bid_allowed_below_limit(self):
        assert check_position_limit(50, "bid", 100) is True

    def test_bid_blocked_at_limit(self):
        assert check_position_limit(100, "bid", 100) is False

    def test_ask_allowed_below_limit(self):
        assert check_position_limit(-50, "ask", 100) is True

    def test_ask_blocked_at_limit(self):
        assert check_position_limit(-100, "ask", 100) is False

    def test_bid_allowed_when_short(self):
        # Short position: bid is fine (reduces short)
        assert check_position_limit(-50, "bid", 100) is True

    def test_ask_allowed_when_long(self):
        # Long position: ask is fine (reduces long)
        assert check_position_limit(50, "ask", 100) is True


class TestCheckEventExposure:
    """Tests for check_event_exposure()."""

    def test_within_limits(self):
        assert check_event_exposure(200, 500) is True

    def test_at_limit(self):
        assert check_event_exposure(500, 500) is False

    def test_over_limit(self):
        assert check_event_exposure(600, 500) is False

    def test_zero_exposure(self):
        assert check_event_exposure(0, 500) is True


class TestShouldOneSideOnly:
    """Tests for should_one_side_only()."""

    def test_flat_both_sides(self):
        assert should_one_side_only(0, 100) is None

    def test_long_at_limit_ask_only(self):
        assert should_one_side_only(100, 100) == "ask"

    def test_short_at_limit_bid_only(self):
        assert should_one_side_only(-100, 100) == "bid"

    def test_within_limit_both_sides(self):
        assert should_one_side_only(50, 100) is None
        assert should_one_side_only(-50, 100) is None


class TestComputeUnrealizedPnl:
    """Tests for compute_unrealized_pnl()."""

    def test_long_profit(self):
        pnl = compute_unrealized_pnl(10, 40.0, 50.0)
        assert pnl == 100.0  # 10 * (50 - 40)

    def test_long_loss(self):
        pnl = compute_unrealized_pnl(10, 50.0, 40.0)
        assert pnl == -100.0  # 10 * (40 - 50)

    def test_short_profit(self):
        pnl = compute_unrealized_pnl(-10, 50.0, 40.0)
        assert pnl == 100.0  # 10 * (50 - 40)

    def test_short_loss(self):
        pnl = compute_unrealized_pnl(-10, 40.0, 50.0)
        assert pnl == -100.0  # 10 * (40 - 50)

    def test_flat_position(self):
        assert compute_unrealized_pnl(0, 40.0, 50.0) == 0.0

    def test_no_mid_price(self):
        assert compute_unrealized_pnl(10, 40.0, None) == 0.0
