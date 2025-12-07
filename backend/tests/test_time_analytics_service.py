"""
Comprehensive unit tests for TimeAnalyticsService.

This test suite provides thorough validation of the TimeAnalyticsService implementation,
covering all core functionality, edge cases, and performance characteristics:

- Bucket creation and management
- Trade processing and volume calculations
- Mode data retrieval (hour/day modes)
- Caching system behavior
- Timestamp calculations and window management
- Error handling and service lifecycle
- Integration with Trade model and database recovery
- Performance and memory cleanup
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from kalshiflow.time_analytics_service import TimeAnalyticsService, Bucket, get_analytics_service
from kalshiflow.models import Trade


class TestBucket:
    """Test the Bucket dataclass functionality."""
    
    def test_bucket_initialization(self):
        """Test bucket creation with proper initialization."""
        timestamp = 1703894400000  # December 29, 2023 16:00:00 UTC
        bucket = Bucket(
            timestamp=timestamp,
            volume_usd=0.0,
            trade_count=0
        )
        
        assert bucket.timestamp == timestamp
        assert bucket.volume_usd == 0.0
        assert bucket.trade_count == 0
    
    def test_bucket_add_trade_yes_side(self):
        """Test adding a YES-side trade to bucket."""
        bucket = Bucket(timestamp=1703894400000, volume_usd=0.0, trade_count=0)
        trade = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1703894400000
        )
        
        bucket.add_trade(trade)
        
        # Volume = count × yes_price_dollars = 100 × 0.65 = $65.00
        assert bucket.volume_usd == 65.0
        assert bucket.trade_count == 1
    
    def test_bucket_add_trade_no_side(self):
        """Test adding a NO-side trade to bucket."""
        bucket = Bucket(timestamp=1703894400000, volume_usd=0.0, trade_count=0)
        trade = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=50,
            taker_side="no",
            ts=1703894400000
        )
        
        bucket.add_trade(trade)
        
        # Volume = count × no_price_dollars = 50 × 0.35 = $17.50
        assert bucket.volume_usd == 17.5
        assert bucket.trade_count == 1
    
    def test_bucket_add_multiple_trades(self):
        """Test adding multiple trades accumulates properly."""
        bucket = Bucket(timestamp=1703894400000, volume_usd=0.0, trade_count=0)
        
        # First trade: YES side, 100 shares @ $0.65 = $65.00
        trade1 = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1703894400000
        )
        
        # Second trade: NO side, 200 shares @ $0.30 = $60.00
        trade2 = Trade(
            market_ticker="PRESWIN25",
            yes_price=70,
            no_price=30,
            yes_price_dollars=0.70,
            no_price_dollars=0.30,
            count=200,
            taker_side="no",
            ts=1703894460000
        )
        
        bucket.add_trade(trade1)
        bucket.add_trade(trade2)
        
        # Total volume: $65.00 + $60.00 = $125.00
        assert bucket.volume_usd == 125.0
        assert bucket.trade_count == 2
    
    def test_bucket_to_dict(self):
        """Test bucket serialization to dictionary."""
        bucket = Bucket(timestamp=1703894400000, volume_usd=123.456789, trade_count=5)
        
        result = bucket.to_dict()
        
        expected = {
            "timestamp": 1703894400000,
            "volume_usd": 123.46,  # Rounded to 2 decimal places
            "trade_count": 5
        }
        assert result == expected


class TestTimeAnalyticsServiceInitialization:
    """Test service initialization and configuration."""
    
    def test_service_init_default_config(self):
        """Test service initialization with default configuration."""
        service = TimeAnalyticsService()
        
        assert service.window_minutes == 60
        assert service.window_hours == 24
        assert not service._running
        assert service._cleanup_task is None
        assert len(service.minute_buckets) == 0
        assert len(service.hour_buckets) == 0
        assert len(service._cache) == 0
        assert service._cache_ttl_ms == 5000
        
        # Check initial stats
        assert service.stats["total_trades_processed"] == 0
        assert service.stats["total_volume_usd"] == 0.0
        assert service.stats["buckets_created"] == 0
        assert service.stats["buckets_cleaned"] == 0
        assert service.stats["cache_hits"] == 0
        assert service.stats["cache_misses"] == 0
        assert service.stats["last_trade_time"] is None
        assert service.stats["started_at"] is None
    
    def test_service_init_custom_config(self):
        """Test service initialization with custom configuration."""
        service = TimeAnalyticsService(window_minutes=30, window_hours=12)
        
        assert service.window_minutes == 30
        assert service.window_hours == 12
        assert not service._running
    
    @pytest.mark.asyncio
    async def test_service_start_stop_lifecycle(self):
        """Test service start/stop lifecycle."""
        service = TimeAnalyticsService()
        
        # Initially not running
        assert not service._running
        assert service.stats["started_at"] is None
        
        # Start service
        await service.start()
        
        assert service._running
        assert service.stats["started_at"] is not None
        assert service._cleanup_task is not None
        
        # Stop service
        await service.stop()
        
        assert not service._running
    
    @pytest.mark.asyncio
    async def test_service_double_start_idempotent(self):
        """Test that starting service multiple times is safe."""
        service = TimeAnalyticsService()
        
        with patch('asyncio.create_task') as mock_create_task:
            # First start
            await service.start()
            assert mock_create_task.call_count == 1
            
            # Second start should be no-op
            await service.start()
            assert mock_create_task.call_count == 1  # Still 1
    
    @pytest.mark.asyncio
    async def test_service_double_stop_safe(self):
        """Test that stopping service multiple times is safe."""
        service = TimeAnalyticsService()
        
        # Stop without starting (should be safe)
        await service.stop()
        assert not service._running
        
        # Start then stop twice
        await service.start()
        await service.stop()
        await service.stop()  # Should be safe
        
        assert not service._running


class TestTimestampCalculations:
    """Test timestamp calculation methods."""
    
    def test_get_period_timestamp_minute_precision(self):
        """Test minute-precision timestamp calculation."""
        service = TimeAnalyticsService()
        
        # Test various timestamps within the same minute
        base_minute = 1703894400000  # December 29, 2023 16:00:00 UTC
        
        test_cases = [
            (base_minute, base_minute),  # Exact start of minute
            (base_minute + 15000, base_minute),  # 15 seconds later
            (base_minute + 30000, base_minute),  # 30 seconds later
            (base_minute + 59999, base_minute),  # 59.999 seconds later
            (base_minute + 60000, base_minute + 60000),  # Next minute
        ]
        
        for input_ts, expected_ts in test_cases:
            result = service._get_period_timestamp(input_ts, "minute")
            assert result == expected_ts
    
    def test_get_period_timestamp_hour_precision(self):
        """Test hour-precision timestamp calculation."""
        service = TimeAnalyticsService()
        
        # Test various timestamps within the same hour
        base_hour = 1703894400000  # December 29, 2023 16:00:00 UTC
        
        test_cases = [
            (base_hour, base_hour),  # Exact start of hour
            (base_hour + 900000, base_hour),  # 15 minutes later
            (base_hour + 1800000, base_hour),  # 30 minutes later
            (base_hour + 3599999, base_hour),  # 59:59.999 later
            (base_hour + 3600000, base_hour + 3600000),  # Next hour
        ]
        
        for input_ts, expected_ts in test_cases:
            result = service._get_period_timestamp(input_ts, "hour")
            assert result == expected_ts
    
    def test_get_period_timestamp_consistency(self):
        """Test that timestamp calculations are consistent across calls."""
        service = TimeAnalyticsService()
        timestamp = 1703894567890  # Some arbitrary timestamp
        
        # Multiple calls should return same result
        result1 = service._get_period_timestamp(timestamp, "minute")
        result2 = service._get_period_timestamp(timestamp, "minute")
        result3 = service._get_period_timestamp(timestamp, "minute")
        
        assert result1 == result2 == result3
        
        # Same for hours
        result1 = service._get_period_timestamp(timestamp, "hour")
        result2 = service._get_period_timestamp(timestamp, "hour")
        
        assert result1 == result2


class TestTradeProcessing:
    """Test trade processing functionality."""
    
    @pytest.mark.asyncio
    async def test_process_trade_basic(self):
        """Test basic trade processing."""
        service = TimeAnalyticsService()
        await service.start()
        
        trade = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1703894400000
        )
        
        service.process_trade(trade)
        
        # Check that buckets were created
        minute_timestamp = service._get_period_timestamp(trade.ts, "minute")
        hour_timestamp = service._get_period_timestamp(trade.ts, "hour")
        
        assert minute_timestamp in service.minute_buckets
        assert hour_timestamp in service.hour_buckets
        
        # Check bucket contents
        minute_bucket = service.minute_buckets[minute_timestamp]
        hour_bucket = service.hour_buckets[hour_timestamp]
        
        expected_volume = 100 * 0.65  # $65.00
        assert minute_bucket.volume_usd == expected_volume
        assert minute_bucket.trade_count == 1
        assert hour_bucket.volume_usd == expected_volume
        assert hour_bucket.trade_count == 1
        
        # Check service stats
        assert service.stats["total_trades_processed"] == 1
        assert service.stats["total_volume_usd"] == expected_volume
        assert service.stats["buckets_created"] == 2  # minute + hour
        assert service.stats["last_trade_time"] is not None
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_process_trade_same_buckets(self):
        """Test processing multiple trades in same time buckets."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Two trades in the same minute and hour
        trade1 = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1703894400000  # 16:00:00
        )
        
        trade2 = Trade(
            market_ticker="CONGRESS25",
            yes_price=45,
            no_price=55,
            yes_price_dollars=0.45,
            no_price_dollars=0.55,
            count=50,
            taker_side="no",
            ts=1703894430000  # 16:00:30 (same minute)
        )
        
        service.process_trade(trade1)
        service.process_trade(trade2)
        
        minute_timestamp = service._get_period_timestamp(trade1.ts, "minute")
        hour_timestamp = service._get_period_timestamp(trade1.ts, "hour")
        
        # Only one bucket per period should exist
        assert len(service.minute_buckets) == 1
        assert len(service.hour_buckets) == 1
        
        # Check accumulated values
        minute_bucket = service.minute_buckets[minute_timestamp]
        hour_bucket = service.hour_buckets[hour_timestamp]
        
        expected_volume = (100 * 0.65) + (50 * 0.55)  # $65.00 + $27.50 = $92.50
        assert minute_bucket.volume_usd == expected_volume
        assert minute_bucket.trade_count == 2
        assert hour_bucket.volume_usd == expected_volume
        assert hour_bucket.trade_count == 2
        
        # Check service stats
        assert service.stats["total_trades_processed"] == 2
        assert service.stats["total_volume_usd"] == expected_volume
        assert service.stats["buckets_created"] == 2  # Still only 2 buckets
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_process_trade_different_periods(self):
        """Test processing trades in different time periods."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Trade in minute 1
        trade1 = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1703894400000  # 16:00:00
        )
        
        # Trade in minute 2 (same hour)
        trade2 = Trade(
            market_ticker="CONGRESS25",
            yes_price=45,
            no_price=55,
            yes_price_dollars=0.45,
            no_price_dollars=0.55,
            count=50,
            taker_side="no",
            ts=1703894460000  # 16:01:00
        )
        
        # Trade in different hour
        trade3 = Trade(
            market_ticker="SENATE25",
            yes_price=80,
            no_price=20,
            yes_price_dollars=0.80,
            no_price_dollars=0.20,
            count=25,
            taker_side="yes",
            ts=1703898000000  # 17:00:00
        )
        
        service.process_trade(trade1)
        service.process_trade(trade2)
        service.process_trade(trade3)
        
        # Should have 3 minute buckets and 2 hour buckets
        assert len(service.minute_buckets) == 3
        assert len(service.hour_buckets) == 2
        
        # Check specific bucket contents
        minute1_ts = service._get_period_timestamp(trade1.ts, "minute")
        minute2_ts = service._get_period_timestamp(trade2.ts, "minute")
        minute3_ts = service._get_period_timestamp(trade3.ts, "minute")
        hour1_ts = service._get_period_timestamp(trade1.ts, "hour")
        hour2_ts = service._get_period_timestamp(trade3.ts, "hour")
        
        assert abs(service.minute_buckets[minute1_ts].volume_usd - 65.0) < 0.01
        assert abs(service.minute_buckets[minute2_ts].volume_usd - 27.5) < 0.01
        assert abs(service.minute_buckets[minute3_ts].volume_usd - 20.0) < 0.01
        
        # First hour should have trades 1 and 2
        assert abs(service.hour_buckets[hour1_ts].volume_usd - 92.5) < 0.01
        assert service.hour_buckets[hour1_ts].trade_count == 2
        
        # Second hour should have trade 3
        assert abs(service.hour_buckets[hour2_ts].volume_usd - 20.0) < 0.01
        assert service.hour_buckets[hour2_ts].trade_count == 1
        
        await service.stop()
    
    def test_process_trade_service_not_running(self):
        """Test that trades are ignored when service is not running."""
        service = TimeAnalyticsService()
        # Don't start service
        
        trade = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1703894400000
        )
        
        service.process_trade(trade)
        
        # No buckets should be created
        assert len(service.minute_buckets) == 0
        assert len(service.hour_buckets) == 0
        assert service.stats["total_trades_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_process_trade_invalidates_cache(self):
        """Test that processing trades invalidates cache."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Add some data to cache
        service._cache["test_key"] = {
            "data": {"test": "data"},
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        assert len(service._cache) == 1
        
        trade = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1703894400000
        )
        
        service.process_trade(trade)
        
        # Cache should be cleared
        assert len(service._cache) == 0
        
        await service.stop()


class TestModeDataRetrieval:
    """Test mode data retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_mode_data_hour_empty(self):
        """Test getting hour mode data with no trades."""
        service = TimeAnalyticsService()
        await service.start()
        
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2023, 12, 29, 16, 5, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result = service.get_mode_data("hour", limit=10)
        
        assert result["mode"] == "hour"
        assert result["current_period"]["volume_usd"] == 0.0
        assert result["current_period"]["trade_count"] == 0
        assert result["summary_stats"]["total_volume_usd"] == 0.0
        assert result["summary_stats"]["total_trades"] == 0
        assert result["summary_stats"]["peak_volume_usd"] == 0.0
        assert result["summary_stats"]["peak_trades"] == 0
        assert len(result["time_series"]) == 10  # Should create empty periods
        assert result["generated_at"] is not None
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_get_mode_data_hour_with_data(self):
        """Test getting hour mode data with actual trades."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Add some trades
        trades = [
            Trade(
                market_ticker="PRESWIN25",
                yes_price=65, no_price=35,
                yes_price_dollars=0.65, no_price_dollars=0.35,
                count=100, taker_side="yes",
                ts=1703894400000  # 16:00:00
            ),
            Trade(
                market_ticker="CONGRESS25",
                yes_price=45, no_price=55,
                yes_price_dollars=0.45, no_price_dollars=0.55,
                count=50, taker_side="no",
                ts=1703894460000  # 16:01:00
            ),
            Trade(
                market_ticker="SENATE25",
                yes_price=80, no_price=20,
                yes_price_dollars=0.80, no_price_dollars=0.20,
                count=25, taker_side="yes",
                ts=1703894520000  # 16:02:00
            )
        ]
        
        for trade in trades:
            service.process_trade(trade)
        
        # Mock current time to be in the same period as last trade
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(1703894550000 / 1000)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result = service.get_mode_data("hour", limit=5)
        
        assert result["mode"] == "hour"
        
        # Should have data in current period (16:02 minute)
        assert result["current_period"]["volume_usd"] == 20.0  # Last trade
        assert result["current_period"]["trade_count"] == 1
        
        # Summary stats should cover all trades
        total_volume = 65.0 + 27.5 + 20.0  # $112.50
        assert result["summary_stats"]["total_volume_usd"] == total_volume
        assert result["summary_stats"]["total_trades"] == 3
        assert result["summary_stats"]["peak_volume_usd"] == 65.0  # First trade was biggest
        assert result["summary_stats"]["peak_trades"] == 1  # Each minute had 1 trade
        
        # Time series should include recent periods
        assert len(result["time_series"]) == 5
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_get_mode_data_day_with_data(self):
        """Test getting day mode data with hour-level aggregation."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Add trades in different hours
        trades = [
            Trade(
                market_ticker="PRESWIN25",
                yes_price=65, no_price=35,
                yes_price_dollars=0.65, no_price_dollars=0.35,
                count=100, taker_side="yes",
                ts=1703894400000  # 16:00:00
            ),
            Trade(
                market_ticker="CONGRESS25",
                yes_price=45, no_price=55,
                yes_price_dollars=0.45, no_price_dollars=0.55,
                count=50, taker_side="no",
                ts=1703898000000  # 17:00:00
            )
        ]
        
        for trade in trades:
            service.process_trade(trade)
        
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime.fromtimestamp(1703898030000 / 1000)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result = service.get_mode_data("day", limit=3)
        
        assert result["mode"] == "day"
        
        # Current period should be 17:00 hour
        assert result["current_period"]["volume_usd"] == 27.5  # Second trade
        assert result["current_period"]["trade_count"] == 1
        
        # Summary should cover both hours
        total_volume = 65.0 + 27.5
        assert result["summary_stats"]["total_volume_usd"] == total_volume
        assert result["summary_stats"]["total_trades"] == 2
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_get_mode_data_caching(self):
        """Test that mode data is properly cached and served from cache."""
        service = TimeAnalyticsService()
        await service.start()
        
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            # Fixed time for consistent caching
            fixed_time = datetime(2023, 12, 29, 16, 5, 0)
            mock_datetime.now.return_value = fixed_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            # First call should be cache miss
            result1 = service.get_mode_data("hour", limit=10)
            assert service.stats["cache_misses"] == 1
            assert service.stats["cache_hits"] == 0
            
            # Second identical call should be cache hit
            result2 = service.get_mode_data("hour", limit=10)
            assert service.stats["cache_misses"] == 1
            assert service.stats["cache_hits"] == 1
            
            # Results should be identical
            assert result1 == result2
            
            # Different limit should be cache miss
            result3 = service.get_mode_data("hour", limit=5)
            assert service.stats["cache_misses"] == 2
            assert service.stats["cache_hits"] == 1
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_get_mode_data_cache_expiration(self):
        """Test that cached data expires after TTL."""
        service = TimeAnalyticsService()
        service._cache_ttl_ms = 1000  # 1 second TTL for testing
        await service.start()
        
        # First call
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            base_time = datetime(2023, 12, 29, 16, 5, 0)
            mock_datetime.now.return_value = base_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result1 = service.get_mode_data("hour", limit=10)
            assert service.stats["cache_misses"] == 1
        
        # Second call after TTL expiration
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            expired_time = base_time + timedelta(seconds=2)  # Past TTL
            mock_datetime.now.return_value = expired_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result2 = service.get_mode_data("hour", limit=10)
            assert service.stats["cache_misses"] == 2  # Should be miss due to expiration
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_get_mode_data_error_handling(self):
        """Test error handling in get_mode_data."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Mock an error during calculation by patching the first call that would succeed
        original_method = service._get_period_timestamp
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Test error")
            return original_method(*args, **kwargs)
        
        with patch.object(service, '_get_period_timestamp', side_effect=side_effect):
            result = service.get_mode_data("hour", limit=10)
        
        # Should return safe default data structure
        assert result["mode"] == "hour"
        assert result["current_period"]["volume_usd"] == 0.0
        assert result["current_period"]["trade_count"] == 0
        assert result["summary_stats"]["total_volume_usd"] == 0.0
        assert isinstance(result["time_series"], list)  # May be empty on error
        
        await service.stop()


class TestCachingSystem:
    """Test caching system functionality."""
    
    def test_cache_data_and_retrieval(self):
        """Test basic cache functionality."""
        service = TimeAnalyticsService()
        
        test_data = {"test": "value", "number": 123}
        cache_key = "test_key"
        
        # Cache data
        service._cache_data(cache_key, test_data)
        
        # Retrieve cached data
        cached = service._get_cached_data(cache_key)
        assert cached == test_data
        assert len(service._cache) == 1
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        service = TimeAnalyticsService()
        service._cache_ttl_ms = 1000  # 1 second
        
        test_data = {"test": "value"}
        cache_key = "test_key"
        
        # Mock current time for cache entry
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            base_time = datetime(2023, 12, 29, 16, 5, 0)
            mock_timestamp = int(base_time.timestamp() * 1000)
            mock_datetime.now.return_value = base_time
            
            service._cache_data(cache_key, test_data)
        
        # Mock time after expiration
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            expired_time = base_time + timedelta(seconds=2)
            expired_timestamp = int(expired_time.timestamp() * 1000)
            mock_datetime.now.return_value = expired_time
            
            cached = service._get_cached_data(cache_key)
            assert cached is None  # Should be expired
            assert len(service._cache) == 0  # Should be cleaned up
    
    def test_cache_size_management(self):
        """Test cache size limitation."""
        service = TimeAnalyticsService()
        
        # Add more than 10 entries to trigger cleanup
        for i in range(15):
            with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
                # Stagger timestamps so we can test oldest removal
                base_time = datetime(2023, 12, 29, 16, 5, i)
                mock_datetime.now.return_value = base_time
                mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
                
                service._cache_data(f"key_{i}", {"value": i})
        
        # Should only keep last 10 entries
        assert len(service._cache) == 10
        
        # Check that we have the most recent entries (key_5 through key_14)
        remaining_keys = set(service._cache.keys())
        expected_keys = {f"key_{i}" for i in range(5, 15)}
        assert remaining_keys == expected_keys
    
    def test_cache_invalidation(self):
        """Test cache invalidation."""
        service = TimeAnalyticsService()
        
        # Add some cache entries
        service._cache_data("key1", {"data": 1})
        service._cache_data("key2", {"data": 2})
        assert len(service._cache) == 2
        
        # Invalidate cache
        service._invalidate_cache()
        assert len(service._cache) == 0
    
    def test_cache_miss_returns_none(self):
        """Test cache miss behavior."""
        service = TimeAnalyticsService()
        
        # Request non-existent key
        cached = service._get_cached_data("nonexistent")
        assert cached is None


class TestStatistics:
    """Test service statistics functionality."""
    
    @pytest.mark.asyncio
    async def test_get_stats_initial_state(self):
        """Test get_stats with initial service state."""
        service = TimeAnalyticsService()
        
        stats = service.get_stats()
        
        assert stats["total_trades_processed"] == 0
        assert stats["total_volume_usd"] == 0.0
        assert stats["buckets_created"] == 0
        assert stats["buckets_cleaned"] == 0
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["last_trade_time"] is None
        assert stats["started_at"] is None
        assert stats["runtime_seconds"] is None
        assert stats["is_running"] is False
        assert stats["window_minutes"] == 60
        assert stats["window_hours"] == 24
        assert stats["active_minute_buckets"] == 0
        assert stats["active_hour_buckets"] == 0
        assert stats["cache_entries"] == 0
        assert stats["cache_hit_rate"] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_stats_with_runtime(self):
        """Test get_stats with running service."""
        service = TimeAnalyticsService()
        
        start_time = datetime(2023, 12, 29, 16, 0, 0)
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = start_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            await service.start()
        
        # Mock current time to calculate runtime
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            current_time = start_time + timedelta(seconds=30)
            mock_datetime.now.return_value = current_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            stats = service.get_stats()
        
        assert stats["is_running"] is True
        assert stats["runtime_seconds"] == 30.0
        assert stats["started_at"] == start_time
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_get_stats_with_data(self):
        """Test get_stats with actual trade data and cache activity."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Process some trades
        trades = [
            Trade(
                market_ticker="PRESWIN25",
                yes_price=65, no_price=35,
                yes_price_dollars=0.65, no_price_dollars=0.35,
                count=100, taker_side="yes",
                ts=1703894400000
            ),
            Trade(
                market_ticker="CONGRESS25",
                yes_price=45, no_price=55,
                yes_price_dollars=0.45, no_price_dollars=0.55,
                count=50, taker_side="no",
                ts=1703894460000
            )
        ]
        
        for trade in trades:
            service.process_trade(trade)
        
        # Trigger cache activity
        service.get_mode_data("hour", limit=10)  # Cache miss
        service.get_mode_data("hour", limit=10)  # Cache hit
        
        stats = service.get_stats()
        
        assert stats["total_trades_processed"] == 2
        assert stats["total_volume_usd"] == 92.5  # 65.0 + 27.5
        assert stats["buckets_created"] == 3  # 2 minute + 1 hour bucket (same hour)  
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
        assert stats["active_minute_buckets"] == 2
        assert stats["active_hour_buckets"] == 1  # Same hour
        assert stats["cache_entries"] == 1
        assert stats["cache_hit_rate"] == 0.5  # 1 hit / (1 hit + 1 miss)
        assert stats["last_trade_time"] is not None
        
        await service.stop()


class TestDatabaseRecovery:
    """Test database recovery functionality."""
    
    @pytest.mark.asyncio
    async def test_recover_disabled(self):
        """Test recovery with enable_recovery=False."""
        service = TimeAnalyticsService()
        
        result = await service.recover_from_database(enable_recovery=False)
        
        assert result["enabled"] is False
        assert result["success"] is True
        assert result["minute_trades_processed"] == 0
        assert result["hour_trades_processed"] == 0
        assert len(service.minute_buckets) == 0
        assert len(service.hour_buckets) == 0
    
    @pytest.mark.asyncio
    async def test_recover_no_trades(self):
        """Test recovery with no trades in database."""
        service = TimeAnalyticsService()
        
        with patch('kalshiflow.time_analytics_service.get_database') as mock_get_db:
            mock_db = AsyncMock()
            mock_db.get_recovery_trade_count.return_value = 0
            mock_get_db.return_value = mock_db
            
            result = await service.recover_from_database()
        
        assert result["enabled"] is True
        assert result["success"] is True
        assert result["minute_trades_processed"] == 0
        assert result["hour_trades_processed"] == 0
        assert mock_db.get_recovery_trade_count.called
        assert not mock_db.get_trades_for_minute_recovery.called
        assert not mock_db.get_trades_for_recovery.called
    
    @pytest.mark.asyncio
    async def test_recover_with_trades(self):
        """Test successful recovery with trade data."""
        service = TimeAnalyticsService()
        
        # Mock trade data from database
        minute_trades_data = [
            {
                "market_ticker": "PRESWIN25",
                "yes_price": 65,
                "no_price": 35,
                "yes_price_dollars": 0.65,
                "no_price_dollars": 0.35,
                "count": 100,
                "taker_side": "yes",
                "ts": 1703894400000
            }
        ]
        
        hour_trades_data = [
            {
                "market_ticker": "PRESWIN25",
                "yes_price": 65,
                "no_price": 35,
                "yes_price_dollars": 0.65,
                "no_price_dollars": 0.35,
                "count": 100,
                "taker_side": "yes",
                "ts": 1703894400000
            },
            {
                "market_ticker": "CONGRESS25",
                "yes_price": 45,
                "no_price": 55,
                "yes_price_dollars": 0.45,
                "no_price_dollars": 0.55,
                "count": 50,
                "taker_side": "no",
                "ts": 1703898000000
            }
        ]
        
        with patch('kalshiflow.time_analytics_service.get_database') as mock_get_db:
            mock_db = AsyncMock()
            mock_db.get_recovery_trade_count.return_value = 2
            mock_db.get_trades_for_minute_recovery.return_value = minute_trades_data
            mock_db.get_trades_for_recovery.return_value = hour_trades_data
            mock_get_db.return_value = mock_db
            
            result = await service.recover_from_database()
        
        assert result["enabled"] is True
        assert result["success"] is True
        assert result["minute_trades_processed"] == 1
        assert result["hour_trades_processed"] == 2
        assert result["minute_buckets_created"] == 1
        assert result["hour_buckets_created"] == 2  # Different hours
        assert result["duration_seconds"] > 0
        
        # Check that buckets were created correctly
        assert len(service.minute_buckets) == 1
        assert len(service.hour_buckets) == 2
        
        # Check service stats were updated
        assert service.stats["total_trades_processed"] == 1  # Based on minute buckets
        assert service.stats["total_volume_usd"] == 65.0
    
    @pytest.mark.asyncio
    async def test_recover_error_handling(self):
        """Test recovery error handling."""
        service = TimeAnalyticsService()
        
        with patch('kalshiflow.time_analytics_service.get_database') as mock_get_db:
            mock_db = AsyncMock()
            mock_db.get_recovery_trade_count.side_effect = Exception("Database error")
            mock_get_db.return_value = mock_db
            
            result = await service.recover_from_database()
        
        assert result["enabled"] is True
        assert result["success"] is False
        assert "error" in result
        assert result["error"] == "Database error"
        assert result["duration_seconds"] > 0
        
        # Buckets should be cleared on error
        assert len(service.minute_buckets) == 0
        assert len(service.hour_buckets) == 0


class TestBucketCleanup:
    """Test bucket cleanup functionality."""
    
    @pytest.mark.asyncio
    async def test_cleanup_old_buckets_minutes(self):
        """Test cleanup of old minute buckets."""
        service = TimeAnalyticsService(window_minutes=60)
        
        # Create buckets with different timestamps
        now = datetime(2023, 12, 29, 16, 30, 0)
        current_ts = int(now.timestamp() * 1000)
        
        # Current bucket (should remain)
        current_minute = service._get_period_timestamp(current_ts, "minute")
        service.minute_buckets[current_minute] = Bucket(current_minute, 100.0, 1)
        
        # Old bucket (should be removed - older than window + 5 minute buffer)
        old_time = now - timedelta(minutes=70)
        old_ts = int(old_time.timestamp() * 1000)
        old_minute = service._get_period_timestamp(old_ts, "minute")
        service.minute_buckets[old_minute] = Bucket(old_minute, 50.0, 1)
        
        # Recent bucket within window (should remain)
        recent_time = now - timedelta(minutes=30)
        recent_ts = int(recent_time.timestamp() * 1000)
        recent_minute = service._get_period_timestamp(recent_ts, "minute")
        service.minute_buckets[recent_minute] = Bucket(recent_minute, 75.0, 1)
        
        assert len(service.minute_buckets) == 3
        
        # Run cleanup with mocked current time
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            await service.cleanup_old_buckets()
        
        # Old bucket should be removed
        assert len(service.minute_buckets) == 2
        assert old_minute not in service.minute_buckets
        assert current_minute in service.minute_buckets
        assert recent_minute in service.minute_buckets
        assert service.stats["buckets_cleaned"] == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_old_buckets_hours(self):
        """Test cleanup of old hour buckets."""
        service = TimeAnalyticsService(window_hours=24)
        
        # Create buckets with different timestamps
        now = datetime(2023, 12, 29, 16, 0, 0)
        current_ts = int(now.timestamp() * 1000)
        
        # Current bucket (should remain)
        current_hour = service._get_period_timestamp(current_ts, "hour")
        service.hour_buckets[current_hour] = Bucket(current_hour, 100.0, 1)
        
        # Old bucket (should be removed - older than window + 1 hour buffer)
        old_time = now - timedelta(hours=26)
        old_ts = int(old_time.timestamp() * 1000)
        old_hour = service._get_period_timestamp(old_ts, "hour")
        service.hour_buckets[old_hour] = Bucket(old_hour, 50.0, 1)
        
        assert len(service.hour_buckets) == 2
        
        # Run cleanup with mocked current time
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            await service.cleanup_old_buckets()
        
        # Old bucket should be removed
        assert len(service.hour_buckets) == 1
        assert old_hour not in service.hour_buckets
        assert current_hour in service.hour_buckets
        assert service.stats["buckets_cleaned"] == 1
    
    @pytest.mark.asyncio
    async def test_cleanup_empty_buckets_safe(self):
        """Test that cleanup is safe with empty bucket collections."""
        service = TimeAnalyticsService()
        
        # No buckets exist
        assert len(service.minute_buckets) == 0
        assert len(service.hour_buckets) == 0
        
        # Cleanup should not error
        await service.cleanup_old_buckets()
        
        assert service.stats["buckets_cleaned"] == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_invalidates_cache(self):
        """Test that cleanup invalidates cache when buckets are removed."""
        service = TimeAnalyticsService()
        
        # Add cache entry
        service._cache["test"] = {"data": "test", "timestamp": int(datetime.now().timestamp() * 1000)}
        
        # Add old bucket that will be cleaned up
        old_time = datetime.now() - timedelta(hours=26)
        old_ts = int(old_time.timestamp() * 1000)
        old_hour = service._get_period_timestamp(old_ts, "hour")
        service.hour_buckets[old_hour] = Bucket(old_hour, 50.0, 1)
        
        assert len(service._cache) == 1
        
        await service.cleanup_old_buckets()
        
        # Cache should be invalidated if buckets were removed
        if service.stats["buckets_cleaned"] > 0:
            assert len(service._cache) == 0


class TestGlobalServiceInstance:
    """Test global service instance management."""
    
    def test_get_analytics_service_singleton(self):
        """Test that get_analytics_service returns singleton instance."""
        # Clear any existing instance
        import kalshiflow.time_analytics_service
        kalshiflow.time_analytics_service._analytics_service_instance = None
        
        # First call should create instance
        service1 = get_analytics_service()
        assert isinstance(service1, TimeAnalyticsService)
        
        # Second call should return same instance
        service2 = get_analytics_service()
        assert service1 is service2
    
    def test_get_analytics_service_default_config(self):
        """Test that global service uses default configuration."""
        # Clear any existing instance
        import kalshiflow.time_analytics_service
        kalshiflow.time_analytics_service._analytics_service_instance = None
        
        service = get_analytics_service()
        assert service.window_minutes == 60
        assert service.window_hours == 24


class TestErrorHandling:
    """Test error handling throughout the service."""
    
    @pytest.mark.asyncio
    async def test_process_trade_error_handling(self):
        """Test error handling during trade processing."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Mock an error in timestamp calculation
        with patch.object(service, '_get_period_timestamp', side_effect=Exception("Timestamp error")):
            trade = Trade(
                market_ticker="PRESWIN25",
                yes_price=65, no_price=35,
                yes_price_dollars=0.65, no_price_dollars=0.35,
                count=100, taker_side="yes",
                ts=1703894400000
            )
            
            # Should not raise exception, should handle gracefully
            service.process_trade(trade)
            
            # Stats should not be updated on error
            assert service.stats["total_trades_processed"] == 0
            assert len(service.minute_buckets) == 0
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self):
        """Test error handling during cleanup."""
        service = TimeAnalyticsService()
        
        # Mock an error in datetime.now
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            mock_datetime.now.side_effect = Exception("Time error")
            
            # Should not raise exception
            await service.cleanup_old_buckets()
            
            # Should not crash or affect stats
            assert service.stats["buckets_cleaned"] == 0


class TestPerformanceAndMemory:
    """Test performance characteristics and memory management."""
    
    @pytest.mark.asyncio
    async def test_large_number_of_buckets(self):
        """Test service with large number of buckets."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Create many buckets across different time periods
        base_time = 1703894400000  # December 29, 2023 16:00:00 UTC
        
        # Create 100 different minute buckets
        for i in range(100):
            trade_time = base_time + (i * 60000)  # Each trade 1 minute apart
            trade = Trade(
                market_ticker=f"MARKET{i % 10}",
                yes_price=50 + (i % 50),
                no_price=50 - (i % 50),
                yes_price_dollars=(50 + (i % 50)) / 100,
                no_price_dollars=(50 - (i % 50)) / 100,
                count=10 + (i % 20),
                taker_side="yes" if i % 2 == 0 else "no",
                ts=trade_time
            )
            service.process_trade(trade)
        
        # Should handle many buckets efficiently
        assert len(service.minute_buckets) == 100
        assert service.stats["total_trades_processed"] == 100
        # Each trade creates minute bucket + hour bucket, but hour buckets may be reused
        # 100 minutes = ~2 hours, so expect ~100 minute + ~2 hour buckets
        assert service.stats["total_trades_processed"] == 100
        assert len(service.minute_buckets) == 100
        # Hour buckets will be fewer due to multiple trades per hour
        assert len(service.hour_buckets) >= 1
        assert len(service.hour_buckets) <= 3  # Should be around 2 hours of data
        
        # Getting mode data should still work efficiently
        # Mock current time to be within the data range for proper window calculation
        with patch('kalshiflow.time_analytics_service.datetime') as mock_datetime:
            # Set current time to be at the end of our data range
            current_time = datetime.fromtimestamp((base_time + (99 * 60000)) / 1000)
            mock_datetime.now.return_value = current_time
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            result = service.get_mode_data("hour", limit=50)
        
        assert len(result["time_series"]) == 50
        assert result["summary_stats"]["total_trades"] > 0
        
        await service.stop()
    
    @pytest.mark.asyncio 
    async def test_rapid_trade_processing(self):
        """Test processing trades in rapid succession."""
        service = TimeAnalyticsService()
        await service.start()
        
        base_time = 1703894400000
        trades = []
        
        # Create 1000 trades in the same minute
        for i in range(1000):
            trades.append(Trade(
                market_ticker=f"RAPID{i % 5}",
                yes_price=50,
                no_price=50,
                yes_price_dollars=0.50,
                no_price_dollars=0.50,
                count=1,
                taker_side="yes",
                ts=base_time + (i * 10)  # 10ms apart, all in same minute
            ))
        
        # Process all trades rapidly
        for trade in trades:
            service.process_trade(trade)
        
        # Should handle rapid processing
        assert service.stats["total_trades_processed"] == 1000
        assert len(service.minute_buckets) == 1  # All in same minute
        
        # Volume should be accumulated correctly
        minute_bucket = list(service.minute_buckets.values())[0]
        assert minute_bucket.volume_usd == 500.0  # 1000 trades × $0.50
        assert minute_bucket.trade_count == 1000
        
        await service.stop()


class TestIntegrationWithTradeModel:
    """Test integration points with Trade model."""
    
    @pytest.mark.asyncio
    async def test_various_price_combinations(self):
        """Test processing trades with various price combinations."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Test edge cases for price calculations
        test_cases = [
            # (yes_price, no_price, count, taker_side, expected_volume)
            (1, 99, 100, "yes", 1.0),      # Very low YES price
            (99, 1, 100, "no", 1.0),       # Very low NO price
            (50, 50, 200, "yes", 100.0),   # Equal prices
            (100, 0, 50, "yes", 50.0),     # Maximum YES price
            (0, 100, 75, "no", 75.0),      # Maximum NO price
        ]
        
        total_volume = 0.0
        for i, (yes_price, no_price, count, taker_side, expected_volume) in enumerate(test_cases):
            trade = Trade(
                market_ticker=f"EDGE{i}",
                yes_price=yes_price,
                no_price=no_price,
                yes_price_dollars=yes_price / 100,
                no_price_dollars=no_price / 100,
                count=count,
                taker_side=taker_side,
                ts=1703894400000 + (i * 60000)  # Different minutes
            )
            
            service.process_trade(trade)
            total_volume += expected_volume
        
        # Verify total volume calculation
        assert service.stats["total_volume_usd"] == total_volume
        assert service.stats["total_trades_processed"] == len(test_cases)
        assert len(service.minute_buckets) == len(test_cases)
        
        await service.stop()
    
    @pytest.mark.asyncio
    async def test_high_volume_trades(self):
        """Test processing high-volume trades."""
        service = TimeAnalyticsService()
        await service.start()
        
        # Very high count trade
        high_volume_trade = Trade(
            market_ticker="HIGHVOLUME",
            yes_price=75,
            no_price=25,
            yes_price_dollars=0.75,
            no_price_dollars=0.25,
            count=1000000,  # 1 million shares
            taker_side="yes",
            ts=1703894400000
        )
        
        service.process_trade(high_volume_trade)
        
        # Volume = 1,000,000 × $0.75 = $750,000
        expected_volume = 750000.0
        assert service.stats["total_volume_usd"] == expected_volume
        
        minute_bucket = list(service.minute_buckets.values())[0]
        assert minute_bucket.volume_usd == expected_volume
        assert minute_bucket.trade_count == 1
        
        await service.stop()