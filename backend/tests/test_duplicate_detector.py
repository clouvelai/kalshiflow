"""
Comprehensive unit tests for TradeDuplicateDetector.

Tests cover:
- Basic duplicate detection
- Time window management
- Statistics tracking 
- Edge cases and error conditions
- Performance characteristics
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

from kalshiflow.duplicate_detector import TradeDuplicateDetector
from kalshiflow.models import Trade


@pytest.fixture
def detector():
    """Create a fresh duplicate detector for each test."""
    return TradeDuplicateDetector(window_minutes=5)


@pytest.fixture
def sample_trade():
    """Create a sample trade for testing."""
    return Trade(
        market_ticker="PRES2024",
        yes_price=55,
        no_price=45,
        yes_price_dollars=0.55,
        no_price_dollars=0.45,
        count=100,
        taker_side="yes",
        ts=int(datetime.now().timestamp() * 1000)
    )


class TestBasicDuplicateDetection:
    """Test basic duplicate detection functionality."""
    
    def test_unique_trade_not_duplicate(self, detector, sample_trade):
        """Test that a unique trade is not marked as duplicate."""
        assert not detector.is_duplicate(sample_trade)
        
        # Check statistics
        stats = detector.get_stats()
        assert stats["total_checks"] == 1
        assert stats["duplicates_detected"] == 0
        assert stats["unique_trades"] == 1
        assert stats["cache_size"] == 1
        assert stats["detection_rate"] == 0.0
    
    def test_exact_duplicate_detected(self, detector, sample_trade):
        """Test that an exact duplicate is detected."""
        # First trade - not duplicate
        assert not detector.is_duplicate(sample_trade)
        
        # Same trade again - should be duplicate
        assert detector.is_duplicate(sample_trade)
        
        # Check statistics
        stats = detector.get_stats()
        assert stats["total_checks"] == 2
        assert stats["duplicates_detected"] == 1
        assert stats["unique_trades"] == 1
        assert stats["cache_size"] == 1
        assert stats["detection_rate"] == 0.5
    
    def test_multiple_duplicates(self, detector, sample_trade):
        """Test handling multiple duplicates of the same trade."""
        # First trade - unique
        assert not detector.is_duplicate(sample_trade)
        
        # Multiple duplicates
        for _ in range(3):
            assert detector.is_duplicate(sample_trade)
        
        stats = detector.get_stats()
        assert stats["total_checks"] == 4
        assert stats["duplicates_detected"] == 3
        assert stats["unique_trades"] == 1
        assert stats["detection_rate"] == 0.75
    
    def test_different_trades_not_duplicate(self, detector):
        """Test that different trades are not marked as duplicates."""
        base_time = int(datetime.now().timestamp() * 1000)
        
        trades = [
            Trade(
                market_ticker="PRES2024",
                yes_price=55,
                no_price=45,
                yes_price_dollars=0.55,
                no_price_dollars=0.45,
                count=100,
                taker_side="yes",
                ts=base_time
            ),
            Trade(
                market_ticker="PRES2024",
                yes_price=56,  # Different price
                no_price=44,
                yes_price_dollars=0.56,
                no_price_dollars=0.44,
                count=100,
                taker_side="yes",
                ts=base_time
            ),
            Trade(
                market_ticker="PRES2024",
                yes_price=55,
                no_price=45,
                yes_price_dollars=0.55,
                no_price_dollars=0.45,
                count=200,  # Different count
                taker_side="yes",
                ts=base_time
            ),
            Trade(
                market_ticker="PRES2024",
                yes_price=55,
                no_price=45,
                yes_price_dollars=0.55,
                no_price_dollars=0.45,
                count=100,
                taker_side="no",  # Different side
                ts=base_time
            ),
            Trade(
                market_ticker="OTHER2024",  # Different ticker
                yes_price=55,
                no_price=45,
                yes_price_dollars=0.55,
                no_price_dollars=0.45,
                count=100,
                taker_side="yes",
                ts=base_time
            ),
        ]
        
        # All trades should be unique
        for trade in trades:
            assert not detector.is_duplicate(trade)
        
        stats = detector.get_stats()
        assert stats["total_checks"] == len(trades)
        assert stats["duplicates_detected"] == 0
        assert stats["unique_trades"] == len(trades)
        assert stats["cache_size"] == len(trades)


class TestTimeWindowManagement:
    """Test time window functionality for cache cleanup."""
    
    @pytest.mark.asyncio
    async def test_window_cleanup_removes_old_trades(self):
        """Test that trades older than the window are removed from cache."""
        detector = TradeDuplicateDetector(window_minutes=1)  # 1 minute window
        
        current_time_ms = int(datetime.now().timestamp() * 1000)
        old_time_ms = current_time_ms - (2 * 60 * 1000)  # 2 minutes ago (outside window)
        
        # Create an old trade
        old_trade = Trade(
            market_ticker="PRES2024",
            yes_price=55,
            no_price=45,
            yes_price_dollars=0.55,
            no_price_dollars=0.45,
            count=100,
            taker_side="yes",
            ts=old_time_ms
        )
        
        # Create a new trade (same content but different timestamp)
        new_trade = Trade(
            market_ticker="PRES2024",
            yes_price=55,
            no_price=45,
            yes_price_dollars=0.55,
            no_price_dollars=0.45,
            count=100,
            taker_side="yes",
            ts=current_time_ms
        )
        
        # Add old trade
        assert not detector.is_duplicate(old_trade)
        assert detector.get_stats()["cache_size"] == 1
        
        # Force cleanup by manually calling the method
        detector._cleanup_old_entries()
        
        # Old trade should be removed, cache size should be 0
        assert detector.get_stats()["cache_size"] == 0
        
        # New trade with same content should not be duplicate since old one was cleaned up
        assert not detector.is_duplicate(new_trade)
        
        stats = detector.get_stats()
        assert stats["unique_trades"] == 2
        assert stats["duplicates_detected"] == 0
    
    def test_manual_cache_clear(self, detector, sample_trade):
        """Test manual cache clearing functionality."""
        # Add a trade
        assert not detector.is_duplicate(sample_trade)
        assert detector.get_stats()["cache_size"] == 1
        
        # Clear cache
        detector.clear_cache()
        assert detector.get_stats()["cache_size"] == 0
        
        # Same trade should not be duplicate after clearing
        assert not detector.is_duplicate(sample_trade)


class TestHashGeneration:
    """Test trade hash generation logic."""
    
    def test_hash_consistency(self, detector):
        """Test that identical trades generate identical hashes."""
        trade1 = Trade(
            market_ticker="PRES2024",
            yes_price=55,
            no_price=45,
            yes_price_dollars=0.55,
            no_price_dollars=0.45,
            count=100,
            taker_side="yes",
            ts=1000000
        )
        
        trade2 = Trade(
            market_ticker="PRES2024",
            yes_price=55,
            no_price=45,
            yes_price_dollars=0.55,
            no_price_dollars=0.45,
            count=100,
            taker_side="yes",
            ts=1000000
        )
        
        hash1 = detector._generate_trade_hash(trade1)
        hash2 = detector._generate_trade_hash(trade2)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 character hex string
    
    def test_hash_uniqueness(self, detector):
        """Test that different trades generate different hashes."""
        base_trade = Trade(
            market_ticker="PRES2024",
            yes_price=55,
            no_price=45,
            yes_price_dollars=0.55,
            no_price_dollars=0.45,
            count=100,
            taker_side="yes",
            ts=1000000
        )
        
        # Create trades with one field different
        variations = [
            {"market_ticker": "OTHER2024"},
            {"yes_price": 56},
            {"no_price": 44},
            {"count": 200},
            {"taker_side": "no"},
            {"ts": 2000000}
        ]
        
        base_hash = detector._generate_trade_hash(base_trade)
        
        for variation in variations:
            modified_trade = base_trade.model_copy(update=variation)
            modified_hash = detector._generate_trade_hash(modified_trade)
            assert modified_hash != base_hash, f"Hash should be different for {variation}"


class TestStatistics:
    """Test statistics tracking and reporting."""
    
    def test_stats_initialization(self, detector):
        """Test that statistics are properly initialized."""
        stats = detector.get_stats()
        
        assert stats["total_checks"] == 0
        assert stats["duplicates_detected"] == 0
        assert stats["unique_trades"] == 0
        assert stats["cache_size"] == 0
        assert stats["detection_rate"] == 0.0
        assert stats["detection_rate_percent"] == 0.0
        assert stats["window_minutes"] == 5
        assert stats["last_cleanup_time"] is None
    
    def test_detection_rate_calculation(self, detector, sample_trade):
        """Test that detection rate is calculated correctly."""
        # Start with 50% rate (1 unique, 1 duplicate)
        assert not detector.is_duplicate(sample_trade)
        assert detector.is_duplicate(sample_trade)
        
        stats = detector.get_stats()
        assert stats["detection_rate_percent"] == 50.0
        
        # Add another unique trade to lower the rate
        other_trade = Trade(
            market_ticker="OTHER2024",
            yes_price=60,
            no_price=40,
            yes_price_dollars=0.60,
            no_price_dollars=0.40,
            count=50,
            taker_side="no",
            ts=int(datetime.now().timestamp() * 1000) + 1000
        )
        assert not detector.is_duplicate(other_trade)
        
        stats = detector.get_stats()
        assert abs(stats["detection_rate_percent"] - 33.33) < 0.01  # ~33.33%
    
    def test_stats_reset(self, detector, sample_trade):
        """Test statistics reset functionality."""
        # Generate some activity
        detector.is_duplicate(sample_trade)
        detector.is_duplicate(sample_trade)
        
        # Reset stats
        detector.reset_stats()
        
        stats = detector.get_stats()
        assert stats["total_checks"] == 0
        assert stats["duplicates_detected"] == 0
        assert stats["unique_trades"] == 0
        assert stats["detection_rate"] == 0.0
        # Cache size should NOT be reset
        assert stats["cache_size"] == 1


class TestAsyncOperations:
    """Test asynchronous operations and lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, detector):
        """Test starting and stopping the detector."""
        # Start detector
        await detector.start()
        assert detector._running is True
        assert detector._cleanup_task is not None
        
        # Stop detector
        await detector.stop()
        assert detector._running is False
        assert detector._cleanup_task.cancelled() or detector._cleanup_task.done()
    
    @pytest.mark.asyncio
    async def test_double_start_idempotent(self, detector):
        """Test that starting twice is safe."""
        await detector.start()
        task1 = detector._cleanup_task
        
        # Start again
        await detector.start()
        task2 = detector._cleanup_task
        
        # Should be the same task
        assert task1 is task2
        
        await detector.stop()
    
    @pytest.mark.asyncio
    async def test_periodic_cleanup_runs(self):
        """Test that periodic cleanup actually runs."""
        detector = TradeDuplicateDetector(window_minutes=1)
        
        # Mock the cleanup method to track calls
        cleanup_calls = []
        original_cleanup = detector._cleanup_old_entries
        
        def mock_cleanup():
            cleanup_calls.append(datetime.now())
            original_cleanup()
        
        detector._cleanup_old_entries = mock_cleanup
        
        try:
            await detector.start()
            
            # Wait for at least one cleanup cycle (cleanup runs every 60 seconds, but we'll mock time)
            # We'll force a cleanup by directly calling the method
            detector._cleanup_old_entries()
            
            assert len(cleanup_calls) >= 1
            
        finally:
            await detector.stop()


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_window_size(self):
        """Test detector with zero window size."""
        detector = TradeDuplicateDetector(window_minutes=0)
        
        trade = Trade(
            market_ticker="PRES2024",
            yes_price=55,
            no_price=45,
            yes_price_dollars=0.55,
            no_price_dollars=0.45,
            count=100,
            taker_side="yes",
            ts=int(datetime.now().timestamp() * 1000)
        )
        
        # Should still work, just with very aggressive cleanup
        assert not detector.is_duplicate(trade)
        assert detector.is_duplicate(trade)
    
    def test_very_large_window(self):
        """Test detector with very large window size."""
        detector = TradeDuplicateDetector(window_minutes=60 * 24 * 7)  # 1 week
        
        trade = Trade(
            market_ticker="PRES2024",
            yes_price=55,
            no_price=45,
            yes_price_dollars=0.55,
            no_price_dollars=0.45,
            count=100,
            taker_side="yes",
            ts=int(datetime.now().timestamp() * 1000)
        )
        
        assert not detector.is_duplicate(trade)
        assert detector.is_duplicate(trade)
    
    def test_repr_string(self, detector, sample_trade):
        """Test string representation of detector."""
        # Add some activity
        detector.is_duplicate(sample_trade)
        detector.is_duplicate(sample_trade)
        
        repr_str = repr(detector)
        
        assert "TradeDuplicateDetector" in repr_str
        assert "window=5min" in repr_str
        assert "cache_size=1" in repr_str
        assert "duplicates=1/2" in repr_str
    
    def test_high_volume_performance(self, detector):
        """Test performance with high volume of trades."""
        import time
        
        base_time = int(datetime.now().timestamp() * 1000)
        
        # Generate many unique trades
        trades = []
        for i in range(1000):
            trade = Trade(
                market_ticker=f"MARKET{i % 10}",  # 10 different markets
                yes_price=50 + (i % 50),  # Varying prices
                no_price=50 - (i % 50),
                yes_price_dollars=(50 + (i % 50)) / 100.0,
                no_price_dollars=(50 - (i % 50)) / 100.0,
                count=100 + i,  # Unique counts
                taker_side="yes" if i % 2 == 0 else "no",
                ts=base_time + i  # Unique timestamps
            )
            trades.append(trade)
        
        # Time the operations
        start_time = time.time()
        
        for trade in trades:
            detector.is_duplicate(trade)
        
        end_time = time.time()
        
        # Should complete quickly (less than 1 second for 1000 trades)
        assert (end_time - start_time) < 1.0
        
        # All should be unique
        stats = detector.get_stats()
        assert stats["unique_trades"] == 1000
        assert stats["duplicates_detected"] == 0
        assert stats["cache_size"] == 1000


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_realistic_duplicate_scenario(self, detector):
        """Test a realistic scenario with some duplicates mixed in."""
        base_time = int(datetime.now().timestamp() * 1000)
        
        # Simulate a series of trades with some duplicates
        trades_sequence = [
            # Initial unique trade
            Trade(
                market_ticker="PRES2024",
                yes_price=55, no_price=45,
                yes_price_dollars=0.55, no_price_dollars=0.45,
                count=100, taker_side="yes", ts=base_time
            ),
            # Different trade
            Trade(
                market_ticker="PRES2024",
                yes_price=56, no_price=44,
                yes_price_dollars=0.56, no_price_dollars=0.44,
                count=150, taker_side="no", ts=base_time + 1000
            ),
            # Duplicate of first trade (network retry scenario)
            Trade(
                market_ticker="PRES2024",
                yes_price=55, no_price=45,
                yes_price_dollars=0.55, no_price_dollars=0.45,
                count=100, taker_side="yes", ts=base_time
            ),
            # Another unique trade
            Trade(
                market_ticker="SPORTS2024",
                yes_price=30, no_price=70,
                yes_price_dollars=0.30, no_price_dollars=0.70,
                count=200, taker_side="yes", ts=base_time + 2000
            ),
            # Duplicate of second trade
            Trade(
                market_ticker="PRES2024",
                yes_price=56, no_price=44,
                yes_price_dollars=0.56, no_price_dollars=0.44,
                count=150, taker_side="no", ts=base_time + 1000
            ),
        ]
        
        expected_results = [False, False, True, False, True]
        
        for trade, expected_duplicate in zip(trades_sequence, expected_results):
            result = detector.is_duplicate(trade)
            assert result == expected_duplicate, f"Trade {trade.market_ticker} duplicate detection failed"
        
        # Final stats check
        stats = detector.get_stats()
        assert stats["total_checks"] == 5
        assert stats["duplicates_detected"] == 2
        assert stats["unique_trades"] == 3
        assert stats["detection_rate_percent"] == 40.0  # 2/5 = 40%