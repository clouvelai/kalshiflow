"""Unit tests for fee_calculator.py.

Pure T1 tests — no async, no mocking, no network.
Exercises Kalshi maker/taker fee formulas and profitability math.
"""

import pytest

from kalshiflow_rl.traderv3.market_maker.fee_calculator import (
    break_even_spread,
    fee_schedule,
    maker_fee,
    spread_pnl_per_contract,
    taker_fee,
)


class TestMakerFee:
    """Tests for maker_fee()."""

    def test_at_50c_max(self):
        # maker_fee = 0.0175 * 0.5 * 0.5 * 100 = 0.4375
        assert maker_fee(50) == pytest.approx(0.4375)

    def test_at_10c(self):
        # 0.0175 * 0.1 * 0.9 * 100 = 0.1575
        assert maker_fee(10) == pytest.approx(0.1575)

    def test_at_90c(self):
        # 0.0175 * 0.9 * 0.1 * 100 = 0.1575
        assert maker_fee(90) == pytest.approx(0.1575)

    def test_symmetric(self):
        # Fee should be symmetric around 50
        assert maker_fee(30) == pytest.approx(maker_fee(70))
        assert maker_fee(20) == pytest.approx(maker_fee(80))

    def test_at_1c_minimal(self):
        # At extreme prices, fee should be very small
        fee = maker_fee(1)
        assert fee > 0
        assert fee < 0.05

    def test_at_99c_minimal(self):
        fee = maker_fee(99)
        assert fee > 0
        assert fee < 0.05


class TestTakerFee:
    """Tests for taker_fee()."""

    def test_at_50c_max(self):
        # taker_fee = 0.07 * 0.5 * 0.5 * 100 = 1.75
        assert taker_fee(50) == pytest.approx(1.75)

    def test_taker_more_than_maker(self):
        for p in [10, 30, 50, 70, 90]:
            assert taker_fee(p) > maker_fee(p)

    def test_maker_is_75pct_cheaper(self):
        # At any price, maker should be ~75% of taker
        for p in [10, 30, 50, 70, 90]:
            ratio = maker_fee(p) / taker_fee(p)
            assert ratio == pytest.approx(0.25, abs=0.01)


class TestBreakEvenSpread:
    """Tests for break_even_spread()."""

    def test_at_50c(self):
        # break_even = 2 * 0.4375 = 0.875
        assert break_even_spread(50.0) == pytest.approx(0.875)

    def test_wider_at_mid_range(self):
        # Break-even spread should be widest around 50c
        bes_50 = break_even_spread(50.0)
        bes_20 = break_even_spread(20.0)
        assert bes_50 > bes_20

    def test_at_extreme(self):
        bes = break_even_spread(5.0)
        assert bes < 0.2  # Very small break-even at extreme prices


class TestSpreadPnl:
    """Tests for spread_pnl_per_contract()."""

    def test_positive_pnl_wide_spread(self):
        # Bid 45, Ask 55 -> spread=10, fees at each side
        pnl = spread_pnl_per_contract(45, 55)
        assert pnl > 0

    def test_zero_spread_negative_pnl(self):
        # Bid=Ask -> fees make it negative
        pnl = spread_pnl_per_contract(50, 50)
        assert pnl < 0

    def test_narrow_spread_may_be_negative(self):
        # 1c spread at 50c: spread=1, fees=2*0.4375=0.875
        pnl = spread_pnl_per_contract(49, 50)
        assert pnl > 0  # 1 - ~0.87 > 0 (barely positive)

    def test_pnl_increases_with_spread(self):
        pnl_narrow = spread_pnl_per_contract(48, 52)
        pnl_wide = spread_pnl_per_contract(45, 55)
        assert pnl_wide > pnl_narrow


class TestFeeSchedule:
    """Tests for fee_schedule()."""

    def test_returns_all_fields(self):
        result = fee_schedule(50)
        assert "price_cents" in result
        assert "maker_fee" in result
        assert "taker_fee" in result
        assert "break_even_spread" in result
        assert "maker_savings_pct" in result

    def test_price_cents_passthrough(self):
        result = fee_schedule(42)
        assert result["price_cents"] == 42

    def test_maker_savings_around_75(self):
        result = fee_schedule(50)
        assert result["maker_savings_pct"] == pytest.approx(75.0, abs=0.1)
