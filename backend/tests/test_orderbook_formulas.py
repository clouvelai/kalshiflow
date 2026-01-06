"""
Unit tests for orderbook signal formulas.

Tests the mathematical correctness of:
- Spread volatility calculation
- Imbalance ratio calculation
- Depth-weighted mid-price calculation
- Queue position awareness logic
"""

import pytest
from datetime import datetime, timezone

from kalshiflow_rl.data.orderbook_signals import BucketState
from kalshiflow_rl.traderv3.state.order_context import (
    OrderbookContext,
    SpreadTier,
    BBODepthTier,
)


class TestSpreadVolatility:
    """Test spread volatility formula: (high - low) / avg_spread"""

    def test_same_instability_different_close(self):
        """
        Spread 2->10 should give same volatility regardless of where it closes.
        This was the bug we fixed - old formula used /close which gave different results.
        """
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))

        # Spread high=10, low=2
        # avg_spread = (10 + 2) / 2 = 6
        # volatility = (10 - 2) / 6 = 1.333...
        result = bucket._calc_spread_volatility(spread_high=10, spread_low=2)

        assert result == pytest.approx(1.3333, rel=0.01)

    def test_narrow_range(self):
        """Spread 3->5 should give lower volatility than 2->10"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))

        # avg_spread = 4, range = 2
        # volatility = 2 / 4 = 0.5
        result = bucket._calc_spread_volatility(spread_high=5, spread_low=3)

        assert result == pytest.approx(0.5, rel=0.01)

    def test_zero_range_same_high_low(self):
        """When high == low, volatility should be 0"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))

        # range = 0, avg = 4
        # volatility = 0 / 4 = 0
        result = bucket._calc_spread_volatility(spread_high=4, spread_low=4)

        assert result == 0.0

    def test_none_inputs(self):
        """Should return None when inputs are None"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))

        assert bucket._calc_spread_volatility(None, 5) is None
        assert bucket._calc_spread_volatility(5, None) is None
        assert bucket._calc_spread_volatility(None, None) is None

    def test_zero_avg_spread(self):
        """Edge case: both high and low are 0 should return None"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))

        result = bucket._calc_spread_volatility(spread_high=0, spread_low=0)

        assert result is None


class TestImbalanceRatio:
    """Test imbalance ratio formula: (bid - ask) / (bid + ask)"""

    def test_equal_volumes_is_zero(self):
        """Equal bid and ask volumes should give 0"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))
        bucket.no_bid_volumes = [100, 100, 100]
        bucket.no_ask_volumes = [100, 100, 100]

        signal_data = bucket.to_signal_data()

        assert signal_data["no_imbalance_ratio"] == pytest.approx(0.0, abs=0.01)

    def test_all_bid_pressure_is_positive_one(self):
        """All bid volume, no ask volume should give +1"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))
        bucket.no_bid_volumes = [100, 100, 100]
        bucket.no_ask_volumes = [0, 0, 0]

        signal_data = bucket.to_signal_data()

        assert signal_data["no_imbalance_ratio"] == pytest.approx(1.0, abs=0.01)

    def test_all_ask_pressure_is_negative_one(self):
        """All ask volume, no bid volume should give -1"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))
        bucket.no_bid_volumes = [0, 0, 0]
        bucket.no_ask_volumes = [100, 100, 100]

        signal_data = bucket.to_signal_data()

        assert signal_data["no_imbalance_ratio"] == pytest.approx(-1.0, abs=0.01)

    def test_bid_heavy_is_positive(self):
        """Bid-heavy volume should give positive ratio"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))
        bucket.no_bid_volumes = [300]  # 300 bid
        bucket.no_ask_volumes = [100]  # 100 ask
        # (300 - 100) / (300 + 100) = 200 / 400 = 0.5

        signal_data = bucket.to_signal_data()

        assert signal_data["no_imbalance_ratio"] == pytest.approx(0.5, abs=0.01)

    def test_ask_heavy_is_negative(self):
        """Ask-heavy volume should give negative ratio"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))
        bucket.no_bid_volumes = [100]  # 100 bid
        bucket.no_ask_volumes = [300]  # 300 ask
        # (100 - 300) / (100 + 300) = -200 / 400 = -0.5

        signal_data = bucket.to_signal_data()

        assert signal_data["no_imbalance_ratio"] == pytest.approx(-0.5, abs=0.01)

    def test_empty_volumes_is_none(self):
        """Empty volume lists should return None"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))
        bucket.no_bid_volumes = []
        bucket.no_ask_volumes = []

        signal_data = bucket.to_signal_data()

        assert signal_data["no_imbalance_ratio"] is None


class TestDepthWeightedMid:
    """Test depth-weighted mid-price calculation using BBO sizes"""

    def test_equal_bbo_gives_midpoint(self):
        """Equal BBO volumes should give simple midpoint"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=44,
            no_spread=4,
            spread_tier=SpreadTier.NORMAL,
            no_bid_size_at_bbo=100,
            no_ask_size_at_bbo=100,
            bbo_depth_tier=BBODepthTier.NORMAL,
        )

        # weight = 100 / 200 = 0.5
        # weighted_mid = 40 + 0.5 * 4 = 42
        price = ctx.get_recommended_entry_price(aggressive=False, max_spread=10)

        assert price == 42

    def test_ask_heavy_shifts_toward_bid(self):
        """Heavy ask volume should shift price toward bid (cheaper entry)"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=44,
            no_spread=4,
            spread_tier=SpreadTier.NORMAL,
            no_bid_size_at_bbo=50,
            no_ask_size_at_bbo=150,  # 3x more ask
            bbo_depth_tier=BBODepthTier.NORMAL,
        )

        # weight = 50 / 200 = 0.25
        # weighted_mid = 40 + 0.25 * 4 = 41
        price = ctx.get_recommended_entry_price(aggressive=False, max_spread=10)

        assert price == 41

    def test_bid_heavy_shifts_toward_ask(self):
        """Heavy bid volume should shift price toward ask"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=44,
            no_spread=4,
            spread_tier=SpreadTier.NORMAL,
            no_bid_size_at_bbo=150,  # 3x more bid
            no_ask_size_at_bbo=50,
            bbo_depth_tier=BBODepthTier.NORMAL,
        )

        # weight = 150 / 200 = 0.75
        # weighted_mid = 40 + 0.75 * 4 = 43
        price = ctx.get_recommended_entry_price(aggressive=False, max_spread=10)

        assert price == 43

    def test_zero_bbo_uses_simple_midpoint(self):
        """Zero BBO volumes should fall back to simple midpoint"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=44,
            no_spread=4,
            spread_tier=SpreadTier.NORMAL,
            no_bid_size_at_bbo=0,
            no_ask_size_at_bbo=0,
            bbo_depth_tier=BBODepthTier.THIN,
        )

        # Falls back to (40 + 44) // 2 = 42
        # But thin liquidity adds +1 -> 43
        price = ctx.get_recommended_entry_price(aggressive=False, max_spread=10)

        assert price == 43  # 42 + 1 for thin liquidity

    def test_tight_spread_uses_ask_minus_one(self):
        """Tight spread should price at ask - 1"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=42,  # 2c spread = TIGHT
            no_spread=2,
            spread_tier=SpreadTier.TIGHT,
            no_bid_size_at_bbo=100,
            no_ask_size_at_bbo=100,
            bbo_depth_tier=BBODepthTier.NORMAL,
        )

        price = ctx.get_recommended_entry_price(aggressive=False, max_spread=10)

        assert price == 41  # ask - 1

    def test_price_bounds_clamped(self):
        """Price should be clamped to [1, 99]"""
        # Test lower bound
        ctx_low = OrderbookContext(
            no_best_bid=0,
            no_best_ask=2,
            no_spread=2,
            spread_tier=SpreadTier.TIGHT,
            no_bid_size_at_bbo=100,
            no_ask_size_at_bbo=100,
            bbo_depth_tier=BBODepthTier.NORMAL,
        )
        price_low = ctx_low.get_recommended_entry_price(aggressive=False, max_spread=10)
        assert price_low >= 1

        # Test upper bound
        ctx_high = OrderbookContext(
            no_best_bid=98,
            no_best_ask=100,
            no_spread=2,
            spread_tier=SpreadTier.TIGHT,
            no_bid_size_at_bbo=100,
            no_ask_size_at_bbo=100,
            bbo_depth_tier=BBODepthTier.NORMAL,
        )
        price_high = ctx_high.get_recommended_entry_price(aggressive=False, max_spread=10)
        assert price_high <= 99


class TestQueuePositionAwareness:
    """Test queue position awareness logic"""

    def test_short_wait_no_adjustment(self):
        """Short queue wait (<30s) should not adjust price"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=44,
            no_spread=4,
            spread_tier=SpreadTier.NORMAL,
            no_bid_size_at_bbo=100,
            no_ask_size_at_bbo=30,  # 30 contracts at ask for queue calc
            bbo_depth_tier=BBODepthTier.NORMAL,
        )

        # trade_rate = 2 trades/sec
        # wait = 30 / 2 = 15 seconds (< 30s threshold)
        # weight = 100 / (100 + 30) = 0.77 -> mid = 40 + 0.77*4 = 43
        price = ctx.get_recommended_entry_price(
            aggressive=False, max_spread=10, trade_rate=2.0
        )

        # Wait time = 30/2 = 15s < 30s, no adjustment
        # Weighted mid = 40 + (100/130) * 4 = 40 + 3.08 = 43
        assert price == 43

    def test_long_wait_increases_aggression(self):
        """Long queue wait (>30s) should add +1c"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=44,
            no_spread=4,
            spread_tier=SpreadTier.NORMAL,
            no_bid_size_at_bbo=100,
            no_ask_size_at_bbo=100,  # equal = mid at 42
            bbo_depth_tier=BBODepthTier.NORMAL,
        )

        # trade_rate = 1 trade/sec
        # wait = 100 / 1 = 100 seconds (> 30s threshold)
        price = ctx.get_recommended_entry_price(
            aggressive=False, max_spread=10, trade_rate=1.0
        )

        # Should be weighted mid (42) + 1 for long queue = 43
        assert price == 43

    def test_no_trade_rate_no_adjustment(self):
        """No trade rate should skip queue logic"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=44,
            no_spread=4,
            spread_tier=SpreadTier.NORMAL,
            no_bid_size_at_bbo=100,
            no_ask_size_at_bbo=100,
            bbo_depth_tier=BBODepthTier.NORMAL,
        )

        price = ctx.get_recommended_entry_price(
            aggressive=False, max_spread=10, trade_rate=None
        )

        # Should be simple weighted mid = 42
        assert price == 42

    def test_zero_trade_rate_no_adjustment(self):
        """Zero trade rate should not cause division by zero"""
        ctx = OrderbookContext(
            no_best_bid=40,
            no_best_ask=44,
            no_spread=4,
            spread_tier=SpreadTier.NORMAL,
            no_bid_size_at_bbo=100,
            no_ask_size_at_bbo=100,
            bbo_depth_tier=BBODepthTier.NORMAL,
        )

        # trade_rate = 0 should be handled safely
        price = ctx.get_recommended_entry_price(
            aggressive=False, max_spread=10, trade_rate=0.0
        )

        # Should be simple weighted mid = 42 (no queue adjustment)
        assert price == 42


class TestLargeOrderAccumulation:
    """Test that large order count accumulates (max) instead of overwriting"""

    def test_large_order_count_uses_max(self):
        """Large order count should track max seen in bucket"""
        bucket = BucketState(bucket_start=datetime.now(timezone.utc))

        # First snapshot: 2 large orders
        bucket.update_from_snapshot({
            "no_bids": {50: 15000, 49: 12000},  # 2 large orders
            "no_asks": {52: 100},
            "yes_bids": {},
            "yes_asks": {},
        })
        assert bucket.large_order_count == 2

        # Second snapshot: 1 large order (whale left)
        bucket.update_from_snapshot({
            "no_bids": {50: 15000},  # Only 1 large order now
            "no_asks": {52: 100},
            "yes_bids": {},
            "yes_asks": {},
        })
        # Should still be 2 (max), not overwritten to 1
        assert bucket.large_order_count == 2

        # Third snapshot: 3 large orders (new whale arrived)
        bucket.update_from_snapshot({
            "no_bids": {50: 15000, 49: 12000, 48: 11000},  # 3 large orders
            "no_asks": {52: 100},
            "yes_bids": {},
            "yes_asks": {},
        })
        # Should update to 3 (new max)
        assert bucket.large_order_count == 3
