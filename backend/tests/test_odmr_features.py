"""
Unit tests for ODMR (Orderbook-Driven Mean Reversion) strategy features.

Tests the ODMR-specific filtering features:
- Tier 1: Whale YES filter
- Tier 2: Orderbook bid imbalance filter
- Backward compatibility with filters OFF
"""

import pytest
from collections import deque
from unittest.mock import Mock, patch

# Import the strategy components
from kalshiflow_rl.traderv3.strategies.plugins.odmr import (
    ODMRStrategy,
    DipMarketState,
    TradeRecord,
    TrackedTrade,
    DipSignal,
    RECENT_TRACKED_TRADES_SIZE,
)


class TestWhaleYesFilter:
    """Test Tier 1: Whale YES filter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ODMRStrategy()
        self.strategy._enable_whale_filter = True
        self.strategy._whale_multiplier = 2.0
        self.strategy._whale_lookback = 5
        self.strategy._min_history_for_whale = 10

    def _create_state_with_trades(
        self, trade_sizes: list, taker_sides: list = None
    ) -> DipMarketState:
        """Helper to create a DipMarketState with trade history."""
        state = DipMarketState(market_ticker="TEST-MARKET")

        if taker_sides is None:
            taker_sides = ["yes"] * len(trade_sizes)

        # Populate trade history
        for i, (size, side) in enumerate(zip(trade_sizes, taker_sides)):
            trade = TradeRecord(
                timestamp=1000.0 + i,
                taker_side=side,
                count=size,
                yes_price=50,
            )
            state.recent_trades.append(trade)
            state.trade_size_history.append(size)

        return state

    def test_whale_yes_trade_passes(self):
        """Whale YES trade (2x avg) should pass filter."""
        # Create history: 10 trades averaging 50 contracts
        # Then a whale YES trade of 100+ contracts (2x avg)
        base_sizes = [50] * 9
        whale_size = 150  # 3x avg, definitely a whale

        state = self._create_state_with_trades(
            trade_sizes=base_sizes + [whale_size],
            taker_sides=["no"] * 9 + ["yes"],  # Whale is on YES side
        )

        result = self.strategy._check_whale_yes_filter(state)
        assert result == "passed", "Whale YES trade should pass filter"

    def test_whale_no_trade_rejects(self):
        """Whale NO trade should NOT pass filter (direction matters)."""
        # Create history: 10 trades averaging 50 contracts
        # Then a whale NO trade (should NOT pass - wrong direction)
        base_sizes = [50] * 9
        whale_size = 150  # 3x avg, but on NO side

        state = self._create_state_with_trades(
            trade_sizes=base_sizes + [whale_size],
            taker_sides=["yes"] * 9 + ["no"],  # Whale is on NO side
        )

        result = self.strategy._check_whale_yes_filter(state)
        assert result == "no_whale", "Whale NO trade should not pass filter"

    def test_no_whale_rejects(self):
        """Normal-sized trades should not pass filter."""
        # All trades are average size - no whale
        state = self._create_state_with_trades(
            trade_sizes=[50] * 15,
            taker_sides=["yes"] * 15,
        )

        result = self.strategy._check_whale_yes_filter(state)
        assert result == "no_whale", "No whale trade should not pass filter"

    def test_insufficient_history_skips(self):
        """Should skip filter with insufficient trade history."""
        # Only 5 trades (less than min_history_for_whale=10)
        state = self._create_state_with_trades(
            trade_sizes=[50, 50, 50, 50, 150],  # 5 trades
            taker_sides=["yes"] * 5,
        )

        result = self.strategy._check_whale_yes_filter(state)
        assert result == "insufficient_history", "Should skip with insufficient history"

    def test_whale_outside_lookback_fails(self):
        """Whale trade outside lookback window should not pass."""
        # Whale trade was 10 trades ago (outside lookback=5)
        whale_size = 150
        base_sizes = [50] * 9

        state = self._create_state_with_trades(
            trade_sizes=[whale_size] + base_sizes,  # Whale first, then 9 normal
            taker_sides=["yes"] + ["yes"] * 9,
        )

        result = self.strategy._check_whale_yes_filter(state)
        assert result == "no_whale", "Whale outside lookback should not pass"


class TestBidImbalanceFilter:
    """Test Tier 2: Orderbook bid imbalance filter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ODMRStrategy()
        self.strategy._use_orderbook_filtering = True
        self.strategy._require_bid_imbalance = True

    def test_bullish_imbalance_passes(self):
        """Bullish orderbook (bid > ask) should pass."""
        snapshot = {
            "yes_bids": {50: 100, 49: 80, 48: 60},  # 240 total
            "yes_asks": {51: 50, 52: 40, 53: 30},   # 120 total
        }

        result = self.strategy._check_yes_bid_imbalance(snapshot)
        assert result == "passed", "Bullish imbalance should pass"

    def test_bearish_imbalance_rejects(self):
        """Bearish orderbook (ask >= bid) should reject."""
        snapshot = {
            "yes_bids": {50: 50, 49: 30, 48: 20},   # 100 total
            "yes_asks": {51: 100, 52: 80, 53: 60},  # 240 total
        }

        result = self.strategy._check_yes_bid_imbalance(snapshot)
        assert result == "rejected", "Bearish imbalance should reject"

    def test_equal_imbalance_rejects(self):
        """Equal bid/ask should reject (not bullish)."""
        snapshot = {
            "yes_bids": {50: 100},  # 100 total
            "yes_asks": {51: 100},  # 100 total
        }

        result = self.strategy._check_yes_bid_imbalance(snapshot)
        assert result == "rejected", "Equal imbalance should reject"

    def test_empty_snapshot_unavailable(self):
        """Empty snapshot should return unavailable."""
        result = self.strategy._check_yes_bid_imbalance({})
        assert result == "unavailable", "Empty snapshot should be unavailable"

    def test_none_snapshot_unavailable(self):
        """None snapshot should return unavailable."""
        result = self.strategy._check_yes_bid_imbalance(None)
        assert result == "unavailable", "None snapshot should be unavailable"

    def test_missing_bids_unavailable(self):
        """Missing bid data should return unavailable."""
        snapshot = {
            "yes_bids": {},
            "yes_asks": {51: 100},
        }

        result = self.strategy._check_yes_bid_imbalance(snapshot)
        assert result == "unavailable", "Missing bids should be unavailable"

    def test_missing_asks_unavailable(self):
        """Missing ask data should return unavailable."""
        snapshot = {
            "yes_bids": {50: 100},
            "yes_asks": {},
        }

        result = self.strategy._check_yes_bid_imbalance(snapshot)
        assert result == "unavailable", "Missing asks should be unavailable"


class TestBackwardCompatibility:
    """Test backward compatibility when ODMR filters are OFF."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ODMRStrategy()
        # Ensure filters are OFF (default)
        self.strategy._enable_whale_filter = False
        self.strategy._use_orderbook_filtering = False

    def test_filters_off_by_default(self):
        """Both ODMR filters should be OFF by default."""
        fresh_strategy = ODMRStrategy()
        assert fresh_strategy._enable_whale_filter is False
        assert fresh_strategy._use_orderbook_filtering is False

    def test_strategy_name_is_odmr(self):
        """Strategy should be registered as 'odmr'."""
        assert self.strategy.name == "odmr"
        assert "ODMR" in self.strategy.display_name

    def test_stats_include_odmr_section(self):
        """get_stats() should include ODMR section."""
        stats = self.strategy.get_stats()
        assert "odmr" in stats
        assert "enabled" in stats["odmr"]
        assert "whale_filter" in stats["odmr"]
        assert "orderbook_filter" in stats["odmr"]


class TestODMRConfiguration:
    """Test ODMR configuration loading."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ODMRStrategy()

    def test_default_whale_filter_params(self):
        """Default whale filter parameters should be set."""
        assert self.strategy._whale_multiplier == 2.0
        assert self.strategy._whale_lookback == 5
        assert self.strategy._min_history_for_whale == 10

    def test_default_orderbook_filter_params(self):
        """Default orderbook filter parameters should be set."""
        assert self.strategy._odmr_spread_max_cents == 2
        assert self.strategy._require_bid_imbalance is True


class TestTradeRecord:
    """Test TradeRecord dataclass."""

    def test_trade_record_creation(self):
        """TradeRecord should store all fields correctly."""
        record = TradeRecord(
            timestamp=1234567890.0,
            taker_side="yes",
            count=100,
            yes_price=55,
        )

        assert record.timestamp == 1234567890.0
        assert record.taker_side == "yes"
        assert record.count == 100
        assert record.yes_price == 55


class TestDipMarketStateTradeHistory:
    """Test DipMarketState trade history fields."""

    def test_recent_trades_deque_maxlen(self):
        """recent_trades should have maxlen=20."""
        state = DipMarketState(market_ticker="TEST")

        # Add 25 trades
        for i in range(25):
            state.recent_trades.append(
                TradeRecord(timestamp=float(i), taker_side="yes", count=10, yes_price=50)
            )

        # Should only keep last 20
        assert len(state.recent_trades) == 20
        assert state.recent_trades[0].timestamp == 5.0  # First 5 dropped

    def test_trade_size_history_deque_maxlen(self):
        """trade_size_history should have maxlen=50."""
        state = DipMarketState(market_ticker="TEST")

        # Add 60 sizes
        for i in range(60):
            state.trade_size_history.append(i)

        # Should only keep last 50
        assert len(state.trade_size_history) == 50
        assert state.trade_size_history[0] == 10  # First 10 dropped


class TestTrackedTrade:
    """Test TrackedTrade dataclass (Phase 8 - UI trade history)."""

    def test_tracked_trade_creation(self):
        """TrackedTrade should store all fields correctly."""
        trade = TrackedTrade(
            trade_id="TEST-MARKET:1234567890:1",
            market_ticker="TEST-MARKET",
            side="yes",
            price_cents=55,
            count=100,
            timestamp=1234567890.0,
        )

        assert trade.trade_id == "TEST-MARKET:1234567890:1"
        assert trade.market_ticker == "TEST-MARKET"
        assert trade.side == "yes"
        assert trade.price_cents == 55
        assert trade.count == 100
        assert trade.timestamp == 1234567890.0

    def test_tracked_trade_to_dict(self):
        """TrackedTrade.to_dict() should return proper dictionary."""
        trade = TrackedTrade(
            trade_id="TEST:123:1",
            market_ticker="TEST",
            side="no",
            price_cents=45,
            count=50,
            timestamp=1234567890.0,
        )

        result = trade.to_dict()

        assert result["trade_id"] == "TEST:123:1"
        assert result["market_ticker"] == "TEST"
        assert result["side"] == "no"
        assert result["price_cents"] == 45
        assert result["count"] == 50
        assert result["timestamp"] == 1234567890.0
        assert "age_seconds" in result


class TestTrackedTradesBuffer:
    """Test ODMR tracked trades buffer (Phase 8)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.strategy = ODMRStrategy()

    def test_recent_tracked_trades_size_constant(self):
        """RECENT_TRACKED_TRADES_SIZE should be 30."""
        assert RECENT_TRACKED_TRADES_SIZE == 30

    def test_recent_tracked_trades_initialized(self):
        """Strategy should have _recent_tracked_trades deque initialized."""
        assert hasattr(self.strategy, "_recent_tracked_trades")
        assert len(self.strategy._recent_tracked_trades) == 0

    def test_trade_counter_initialized(self):
        """Strategy should have _trade_counter initialized to 0."""
        assert hasattr(self.strategy, "_trade_counter")
        assert self.strategy._trade_counter == 0

    def test_get_recent_tracked_trades_empty(self):
        """get_recent_tracked_trades() should return empty list when no trades."""
        result = self.strategy.get_recent_tracked_trades()
        assert result == []

    def test_get_recent_tracked_trades_returns_trades(self):
        """get_recent_tracked_trades() should return tracked trades."""
        # Add some trades manually
        for i in range(5):
            trade = TrackedTrade(
                trade_id=f"TEST:{i}:{i}",
                market_ticker="TEST",
                side="yes",
                price_cents=50,
                count=10,
                timestamp=1000.0 + i,
            )
            self.strategy._recent_tracked_trades.append(trade)

        result = self.strategy.get_recent_tracked_trades()

        assert len(result) == 5
        # Most recent should be first (reversed order)
        assert result[0]["trade_id"] == "TEST:4:4"
        assert result[4]["trade_id"] == "TEST:0:0"

    def test_get_recent_tracked_trades_respects_limit(self):
        """get_recent_tracked_trades() should respect the limit parameter."""
        # Add 10 trades
        for i in range(10):
            trade = TrackedTrade(
                trade_id=f"TEST:{i}:{i}",
                market_ticker="TEST",
                side="yes",
                price_cents=50,
                count=10,
                timestamp=1000.0 + i,
            )
            self.strategy._recent_tracked_trades.append(trade)

        result = self.strategy.get_recent_tracked_trades(limit=3)

        assert len(result) == 3
        # Should return 3 most recent trades
        assert result[0]["trade_id"] == "TEST:9:9"

    def test_recent_tracked_trades_deque_maxlen(self):
        """_recent_tracked_trades should have maxlen=30."""
        # Add 35 trades to exceed maxlen
        for i in range(35):
            trade = TrackedTrade(
                trade_id=f"TEST:{i}:{i}",
                market_ticker="TEST",
                side="yes",
                price_cents=50,
                count=10,
                timestamp=1000.0 + i,
            )
            self.strategy._recent_tracked_trades.append(trade)

        # Should only keep last 30
        assert len(self.strategy._recent_tracked_trades) == 30
        # First 5 should be dropped
        assert self.strategy._recent_tracked_trades[0].trade_id == "TEST:5:5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
