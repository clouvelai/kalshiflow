"""
Unit tests for OrderbookSignalAggregator and BucketState.

Tests signal aggregation, OHLC spread tracking, volume imbalance calculation,
bucket rotation, and database flush behavior.
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from kalshiflow_rl.data.orderbook_signals import (
    BucketState,
    OrderbookSignalAggregator,
    LARGE_ORDER_THRESHOLD,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_orderbook_snapshot():
    """Realistic orderbook snapshot for signal testing."""
    return {
        "no_spread": 3,
        "yes_spread": 4,
        "no_bids": {"45": 500, "44": 300, "43": 200},
        "no_asks": {"48": 400, "49": 200},
        "yes_bids": {"52": 400, "51": 200},
        "yes_asks": {"55": 500, "56": 300},
    }


@pytest.fixture
def snapshot_with_large_order():
    """Snapshot containing a large order (>= 1000 contracts)."""
    return {
        "no_spread": 2,
        "yes_spread": 2,
        "no_bids": {"45": 1500, "44": 300},  # 1500 >= 1000 threshold
        "no_asks": {"47": 400},
        "yes_bids": {"53": 400},
        "yes_asks": {"55": 2000},  # 2000 >= 1000 threshold
    }


@pytest.fixture
def bucket_start():
    """Fixed bucket start time for deterministic tests."""
    return datetime(2025, 1, 4, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def bucket_state(bucket_start):
    """Create a BucketState with fixed start time."""
    return BucketState(bucket_start=bucket_start, bucket_seconds=10)


@pytest.fixture
def signal_aggregator():
    """Create aggregator with test session ID (no DB flush loop)."""
    return OrderbookSignalAggregator(
        session_id=999, bucket_seconds=10, flush_interval=60.0
    )


# =============================================================================
# BucketState Tests
# =============================================================================


class TestBucketStateInitialization:
    """Test BucketState initialization and defaults."""

    def test_default_values(self, bucket_start):
        """Test that BucketState initializes with correct defaults."""
        bucket = BucketState(bucket_start=bucket_start)

        assert bucket.bucket_start == bucket_start
        assert bucket.bucket_seconds == 10

        # Spread OHLC should be None
        assert bucket.no_spread_first is None
        assert bucket.no_spread_high is None
        assert bucket.no_spread_low is None
        assert bucket.no_spread_last is None
        assert bucket.yes_spread_first is None

        # Volume lists should be empty
        assert bucket.no_bid_volumes == []
        assert bucket.no_ask_volumes == []
        assert bucket.yes_bid_volumes == []
        assert bucket.yes_ask_volumes == []

        # BBO sizes should be empty
        assert bucket.no_bid_bbo_sizes == []
        assert bucket.no_ask_bbo_sizes == []

        # Counters should be zero
        assert bucket.snapshot_count == 0
        assert bucket.delta_count == 0
        assert bucket.large_order_count == 0

    def test_custom_bucket_seconds(self, bucket_start):
        """Test custom bucket size."""
        bucket = BucketState(bucket_start=bucket_start, bucket_seconds=30)
        assert bucket.bucket_seconds == 30


class TestBucketStateSpreadTracking:
    """Test OHLC spread tracking in BucketState."""

    def test_first_snapshot_sets_open(self, bucket_state, sample_orderbook_snapshot):
        """First snapshot sets spread open value."""
        bucket_state.update_from_snapshot(sample_orderbook_snapshot)

        assert bucket_state.no_spread_first == 3
        assert bucket_state.no_spread_last == 3
        assert bucket_state.yes_spread_first == 4
        assert bucket_state.yes_spread_last == 4

    def test_spread_ohlc_sequence(self, bucket_state):
        """Test that spread OHLC tracks correctly across snapshots."""
        # First snapshot: spread = 3
        bucket_state.update_from_snapshot({"no_spread": 3, "yes_spread": 3})

        # Second snapshot: spread = 5 (new high)
        bucket_state.update_from_snapshot({"no_spread": 5, "yes_spread": 5})

        # Third snapshot: spread = 1 (new low)
        bucket_state.update_from_snapshot({"no_spread": 1, "yes_spread": 1})

        # Fourth snapshot: spread = 2 (close)
        bucket_state.update_from_snapshot({"no_spread": 2, "yes_spread": 2})

        # Verify OHLC
        assert bucket_state.no_spread_first == 3  # Open
        assert bucket_state.no_spread_high == 5   # High
        assert bucket_state.no_spread_low == 1    # Low
        assert bucket_state.no_spread_last == 2   # Close

    def test_none_spread_ignored(self, bucket_state):
        """Snapshots with None spread don't update OHLC."""
        bucket_state.update_from_snapshot({"no_spread": 3, "yes_spread": 3})
        bucket_state.update_from_snapshot({"no_spread": None, "yes_spread": None})

        # Should still have original values
        assert bucket_state.no_spread_first == 3
        assert bucket_state.no_spread_last == 3
        assert bucket_state.snapshot_count == 2


class TestBucketStateVolumeTracking:
    """Test volume accumulation in BucketState."""

    def test_volume_accumulation(self, bucket_state, sample_orderbook_snapshot):
        """Test that volumes are accumulated per snapshot."""
        bucket_state.update_from_snapshot(sample_orderbook_snapshot)
        bucket_state.update_from_snapshot(sample_orderbook_snapshot)

        # Should have 2 entries for each volume list
        assert len(bucket_state.no_bid_volumes) == 2
        assert len(bucket_state.no_ask_volumes) == 2
        assert len(bucket_state.yes_bid_volumes) == 2
        assert len(bucket_state.yes_ask_volumes) == 2

        # Check actual values (sum of all level sizes)
        # no_bids: 500 + 300 + 200 = 1000
        assert bucket_state.no_bid_volumes[0] == 1000
        # no_asks: 400 + 200 = 600
        assert bucket_state.no_ask_volumes[0] == 600

    def test_empty_book_volumes(self, bucket_state):
        """Test handling of empty orderbooks."""
        bucket_state.update_from_snapshot({
            "no_spread": 2,
            "yes_spread": 2,
            "no_bids": {},
            "no_asks": {},
            "yes_bids": {},
            "yes_asks": {},
        })

        assert bucket_state.no_bid_volumes == [0]
        assert bucket_state.no_ask_volumes == [0]


class TestBucketStateBBODepth:
    """Test BBO (best bid/offer) depth tracking."""

    def test_bbo_sizes_tracked(self, bucket_state, sample_orderbook_snapshot):
        """Test that BBO sizes are correctly extracted."""
        bucket_state.update_from_snapshot(sample_orderbook_snapshot)

        # Best no_bid is at price 45 with size 500
        assert bucket_state.no_bid_bbo_sizes == [500]
        # Best no_ask is at price 48 with size 400
        assert bucket_state.no_ask_bbo_sizes == [400]
        # Best yes_bid is at price 52 with size 400
        assert bucket_state.yes_bid_bbo_sizes == [400]
        # Best yes_ask is at price 55 with size 500
        assert bucket_state.yes_ask_bbo_sizes == [500]

    def test_empty_book_no_bbo(self, bucket_state):
        """Empty books don't add BBO entries."""
        bucket_state.update_from_snapshot({
            "no_spread": None,
            "yes_spread": None,
            "no_bids": {},
            "no_asks": {},
            "yes_bids": {},
            "yes_asks": {},
        })

        assert bucket_state.no_bid_bbo_sizes == []
        assert bucket_state.no_ask_bbo_sizes == []


class TestBucketStateLargeOrders:
    """Test large order detection."""

    def test_large_order_detection(self, bucket_state, snapshot_with_large_order):
        """Large orders (>= 1000) are counted."""
        bucket_state.update_from_snapshot(snapshot_with_large_order)

        # 1500 in no_bids + 2000 in yes_asks = 2 large orders
        assert bucket_state.large_order_count == 2

    def test_no_large_orders(self, bucket_state, sample_orderbook_snapshot):
        """No large orders when all sizes < 1000."""
        bucket_state.update_from_snapshot(sample_orderbook_snapshot)
        assert bucket_state.large_order_count == 0

    def test_threshold_exact(self, bucket_state):
        """Exactly 1000 contracts triggers large order detection."""
        bucket_state.update_from_snapshot({
            "no_spread": 2,
            "yes_spread": 2,
            "no_bids": {"45": 1000},  # Exactly threshold
            "no_asks": {},
            "yes_bids": {},
            "yes_asks": {},
        })
        assert bucket_state.large_order_count == 1


class TestBucketStateDeltaCount:
    """Test delta counting."""

    def test_delta_increment(self, bucket_state):
        """Delta count increments correctly."""
        assert bucket_state.delta_count == 0

        bucket_state.increment_delta_count()
        assert bucket_state.delta_count == 1

        bucket_state.increment_delta_count()
        bucket_state.increment_delta_count()
        assert bucket_state.delta_count == 3


class TestBucketStateToSignalData:
    """Test conversion to signal data dict."""

    def test_to_signal_data_structure(self, bucket_state, sample_orderbook_snapshot):
        """Test that to_signal_data returns correct structure."""
        bucket_state.update_from_snapshot(sample_orderbook_snapshot)
        bucket_state.increment_delta_count()

        data = bucket_state.to_signal_data()

        # Check all expected keys exist
        expected_keys = {
            "bucket_seconds",
            "no_spread_open", "no_spread_high", "no_spread_low", "no_spread_close",
            "yes_spread_open", "yes_spread_high", "yes_spread_low", "yes_spread_close",
            "no_bid_volume_avg", "no_ask_volume_avg", "no_imbalance_ratio",
            "yes_bid_volume_avg", "yes_ask_volume_avg", "yes_imbalance_ratio",
            "no_bid_size_at_bbo_avg", "no_ask_size_at_bbo_avg",
            "yes_bid_size_at_bbo_avg", "yes_ask_size_at_bbo_avg",
            "snapshot_count", "delta_count", "large_order_count",
        }
        assert set(data.keys()) == expected_keys

    def test_to_signal_data_values(self, bucket_state, sample_orderbook_snapshot):
        """Test that to_signal_data returns correct values."""
        bucket_state.update_from_snapshot(sample_orderbook_snapshot)

        data = bucket_state.to_signal_data()

        assert data["bucket_seconds"] == 10
        assert data["no_spread_open"] == 3
        assert data["no_spread_close"] == 3
        assert data["snapshot_count"] == 1
        assert data["delta_count"] == 0

        # Volume averages (single snapshot, so avg = value)
        # no_bids: 500 + 300 + 200 = 1000
        assert data["no_bid_volume_avg"] == 1000
        # no_asks: 400 + 200 = 600
        assert data["no_ask_volume_avg"] == 600

    def test_imbalance_ratio_calculation(self, bucket_state):
        """Test imbalance ratio calculation."""
        # Create a snapshot with known volumes
        bucket_state.update_from_snapshot({
            "no_spread": 2,
            "yes_spread": 2,
            "no_bids": {"45": 800},  # bid volume = 800
            "no_asks": {"47": 200},  # ask volume = 200
            "yes_bids": {},
            "yes_asks": {},
        })

        data = bucket_state.to_signal_data()

        # no_imbalance = 800 / (800 + 200) = 0.8
        assert data["no_imbalance_ratio"] == 0.8

    def test_imbalance_zero_volume(self, bucket_state):
        """Zero volume returns 0.5 (neutral)."""
        bucket_state.update_from_snapshot({
            "no_spread": 2,
            "yes_spread": 2,
            "no_bids": {"45": 0},
            "no_asks": {"47": 0},
            "yes_bids": {},
            "yes_asks": {},
        })

        data = bucket_state.to_signal_data()
        assert data["no_imbalance_ratio"] == 0.5

    def test_imbalance_empty_lists_returns_none(self, bucket_state):
        """Empty volume lists return None for imbalance."""
        # No update, so lists are empty
        data = bucket_state.to_signal_data()
        assert data["no_imbalance_ratio"] is None


# =============================================================================
# OrderbookSignalAggregator Tests
# =============================================================================


class TestAggregatorInitialization:
    """Test OrderbookSignalAggregator initialization."""

    def test_initialization(self):
        """Test aggregator initializes with correct config."""
        agg = OrderbookSignalAggregator(
            session_id=123, bucket_seconds=15, flush_interval=30.0
        )

        assert agg.session_id == 123
        assert agg.bucket_seconds == 15
        assert agg.flush_interval == 30.0
        assert agg._running is False
        assert len(agg._buckets) == 0
        assert len(agg._pending_flush) == 0

    def test_get_stats_initial(self, signal_aggregator):
        """Test stats after initialization."""
        stats = signal_aggregator.get_stats()

        assert stats["running"] is False
        assert stats["session_id"] == 999
        assert stats["bucket_seconds"] == 10
        assert stats["active_buckets"] == 0
        assert stats["pending_flush"] == 0
        assert stats["signals_flushed"] == 0
        assert stats["snapshots_processed"] == 0


class TestAggregatorBucketManagement:
    """Test bucket creation and rotation."""

    def test_record_snapshot_creates_bucket(
        self, signal_aggregator, sample_orderbook_snapshot
    ):
        """Recording a snapshot creates a bucket for the market."""
        signal_aggregator.record_snapshot("TEST-MKT", sample_orderbook_snapshot)

        assert "TEST-MKT" in signal_aggregator._buckets
        assert signal_aggregator._buckets["TEST-MKT"].snapshot_count == 1
        assert signal_aggregator._snapshots_processed == 1

    def test_multiple_markets(self, signal_aggregator, sample_orderbook_snapshot):
        """Multiple markets get separate buckets."""
        signal_aggregator.record_snapshot("MKT-A", sample_orderbook_snapshot)
        signal_aggregator.record_snapshot("MKT-B", sample_orderbook_snapshot)
        signal_aggregator.record_snapshot("MKT-A", sample_orderbook_snapshot)

        assert len(signal_aggregator._buckets) == 2
        assert signal_aggregator._buckets["MKT-A"].snapshot_count == 2
        assert signal_aggregator._buckets["MKT-B"].snapshot_count == 1

    def test_record_delta(self, signal_aggregator, sample_orderbook_snapshot):
        """Recording delta increments counter."""
        signal_aggregator.record_snapshot("TEST-MKT", sample_orderbook_snapshot)
        signal_aggregator.record_delta("TEST-MKT")
        signal_aggregator.record_delta("TEST-MKT")

        assert signal_aggregator._buckets["TEST-MKT"].delta_count == 2

    def test_get_current_bucket_signals(
        self, signal_aggregator, sample_orderbook_snapshot
    ):
        """get_current_bucket_signals returns live data."""
        signal_aggregator.record_snapshot("TEST-MKT", sample_orderbook_snapshot)

        signals = signal_aggregator.get_current_bucket_signals("TEST-MKT")

        assert signals is not None
        assert signals["no_spread_close"] == 3
        assert signals["snapshot_count"] == 1

    def test_get_current_bucket_signals_unknown_market(self, signal_aggregator):
        """Unknown market returns None."""
        signals = signal_aggregator.get_current_bucket_signals("UNKNOWN")
        assert signals is None


class TestAggregatorLifecycle:
    """Test start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self, signal_aggregator):
        """Test start and stop lifecycle."""
        assert signal_aggregator._running is False

        await signal_aggregator.start()
        assert signal_aggregator._running is True
        assert signal_aggregator._flush_task is not None

        await signal_aggregator.stop()
        assert signal_aggregator._running is False

    @pytest.mark.asyncio
    async def test_double_start_warning(self, signal_aggregator):
        """Starting twice logs warning but doesn't crash."""
        await signal_aggregator.start()
        await signal_aggregator.start()  # Should log warning

        assert signal_aggregator._running is True

        await signal_aggregator.stop()


class TestAggregatorFlush:
    """Test database flush behavior."""

    @pytest.mark.asyncio
    async def test_flush_on_stop(self, signal_aggregator, sample_orderbook_snapshot):
        """Buckets are flushed when aggregator stops."""
        # Must start the aggregator first (stop() returns early if not running)
        await signal_aggregator.start()

        with patch.object(
            signal_aggregator, "_flush_pending", new_callable=AsyncMock
        ) as mock_flush:
            signal_aggregator.record_snapshot("TEST-MKT", sample_orderbook_snapshot)

            # Stop should trigger flush
            await signal_aggregator.stop()

            mock_flush.assert_called()

    @pytest.mark.asyncio
    async def test_bucket_completion_adds_to_pending(self, signal_aggregator):
        """Completed buckets are added to pending flush queue."""
        # Manually create and complete a bucket
        bucket_start = datetime(2025, 1, 4, 12, 0, 0, tzinfo=timezone.utc)
        bucket = BucketState(bucket_start=bucket_start)
        bucket.update_from_snapshot({"no_spread": 3, "yes_spread": 3})

        signal_aggregator._complete_bucket("TEST-MKT", bucket)

        assert len(signal_aggregator._pending_flush) == 1
        assert signal_aggregator._pending_flush[0]["market_ticker"] == "TEST-MKT"

    def test_empty_bucket_not_flushed(self, signal_aggregator, bucket_start):
        """Empty buckets (0 snapshots) are not added to flush queue."""
        bucket = BucketState(bucket_start=bucket_start)
        # Don't add any snapshots

        signal_aggregator._complete_bucket("TEST-MKT", bucket)

        assert len(signal_aggregator._pending_flush) == 0

    @pytest.mark.asyncio
    async def test_flush_calls_database(self, signal_aggregator):
        """Flush actually calls database batch insert."""
        # Add pending data
        signal_aggregator._pending_flush.append({
            "market_ticker": "TEST-MKT",
            "bucket_timestamp": datetime.now(timezone.utc),
            "no_spread_open": 3,
        })

        with patch(
            "kalshiflow_rl.data.orderbook_signals.rl_db.batch_insert_orderbook_signals",
            new_callable=AsyncMock,
            return_value=1,
        ) as mock_insert:
            await signal_aggregator._flush_pending()

            mock_insert.assert_called_once()
            assert signal_aggregator._signals_flushed == 1
            assert len(signal_aggregator._pending_flush) == 0

    @pytest.mark.asyncio
    async def test_flush_backlog_limiting(self, signal_aggregator):
        """Flush queue is limited to 1000 entries on failure."""
        # Add 1500 pending entries
        for i in range(1500):
            signal_aggregator._pending_flush.append({"id": i})

        # Mock DB failure
        with patch(
            "kalshiflow_rl.data.orderbook_signals.rl_db.batch_insert_orderbook_signals",
            new_callable=AsyncMock,
            side_effect=Exception("DB error"),
        ):
            await signal_aggregator._flush_pending()

            # Should keep only last 1000
            assert len(signal_aggregator._pending_flush) == 1000
            # Should keep most recent (ids 500-1499)
            assert signal_aggregator._pending_flush[0]["id"] == 500


class TestBucketStartCalculation:
    """Test bucket start time calculation."""

    def test_bucket_start_rounds_down(self, signal_aggregator):
        """Bucket start rounds down to nearest boundary."""
        # 12:00:23 should round to 12:00:20 (10-second buckets)
        test_time = datetime(2025, 1, 4, 12, 0, 23, tzinfo=timezone.utc)

        bucket_start = signal_aggregator._get_bucket_start(test_time)

        assert bucket_start.second == 20
        assert bucket_start.microsecond == 0

    def test_bucket_start_at_boundary(self, signal_aggregator):
        """Exact boundary stays at boundary."""
        test_time = datetime(2025, 1, 4, 12, 0, 30, tzinfo=timezone.utc)

        bucket_start = signal_aggregator._get_bucket_start(test_time)

        assert bucket_start.second == 30


# =============================================================================
# Integration Tests
# =============================================================================


class TestAggregatorIntegration:
    """Integration tests with realistic usage patterns."""

    def test_full_bucket_workflow(self, signal_aggregator, sample_orderbook_snapshot):
        """Test complete workflow: record -> complete -> signal data."""
        # Record multiple snapshots
        for _ in range(5):
            signal_aggregator.record_snapshot("TEST-MKT", sample_orderbook_snapshot)
            signal_aggregator.record_delta("TEST-MKT")

        # Get current signals
        signals = signal_aggregator.get_current_bucket_signals("TEST-MKT")

        assert signals["snapshot_count"] == 5
        assert signals["delta_count"] == 5
        assert signals["no_spread_open"] == 3
        assert signals["no_spread_close"] == 3

    def test_stats_update(self, signal_aggregator, sample_orderbook_snapshot):
        """Stats update correctly after operations."""
        signal_aggregator.record_snapshot("MKT-A", sample_orderbook_snapshot)
        signal_aggregator.record_snapshot("MKT-B", sample_orderbook_snapshot)

        stats = signal_aggregator.get_stats()

        assert stats["active_buckets"] == 2
        assert stats["snapshots_processed"] == 2
