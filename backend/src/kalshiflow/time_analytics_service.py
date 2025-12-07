"""
TimeAnalyticsService - Clean rewrite with simplified architecture.

Eliminates the 6 overlapping methods from the original service and provides:
- 3 focused methods: process_trade(), get_mode_data(), get_stats()  
- Single timestamp calculation method to prevent coordination issues
- Efficient caching to avoid redundant calculations
- Preserved database recovery and bucket management logic

Key improvements:
- Single responsibility principle for each method
- Consistent timestamp handling across all operations
- Cached calculations for peak/total statistics
- Clean data structures with minimal memory overhead
- Backward compatible with existing Trade model and database integration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Literal
from dataclasses import dataclass

from .models import Trade
from .database import get_database


logger = logging.getLogger(__name__)


@dataclass
class Bucket:
    """Unified bucket structure for both minute and hour aggregations."""
    timestamp: int  # Timestamp in milliseconds for the start of the period
    volume_usd: float  # Total USD volume for the period
    trade_count: int  # Total number of trades for the period
    
    def add_trade(self, trade: Trade):
        """Add a trade to this bucket with proper volume calculation.
        
        Volume Calculation:
        - USD Volume = trade.count × price_in_dollars
        - For YES trades: count × yes_price_dollars  
        - For NO trades: count × no_price_dollars
        - This represents the actual monetary value traded
        """
        # Calculate USD volume for this trade (count × price in dollars)
        if trade.taker_side == "yes":
            trade_volume_usd = trade.count * trade.yes_price_dollars
        else:
            trade_volume_usd = trade.count * trade.no_price_dollars
        
        self.volume_usd += trade_volume_usd
        self.trade_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "volume_usd": round(self.volume_usd, 2),
            "trade_count": self.trade_count
        }


class TimeAnalyticsService:
    """
    Clean analytics service with simplified 3-method architecture.
    
    Methods:
    1. process_trade(trade) - Update buckets only
    2. get_mode_data(mode, limit) - Complete data for one mode
    3. get_stats() - Service statistics only
    """
    
    def __init__(self, window_minutes: int = 60, window_hours: int = 24):
        """Initialize the analytics service.
        
        Args:
            window_minutes: Number of minutes to maintain in rolling window (default: 60)
            window_hours: Number of hours to maintain in rolling window (default: 24)
        """
        self.window_minutes = window_minutes
        self.window_hours = window_hours
        self.minute_buckets: Dict[int, Bucket] = {}
        self.hour_buckets: Dict[int, Bucket] = {}
        self._running = False
        self._cleanup_task = None
        
        # Simple cache for expensive calculations
        self._cache: Dict[str, Any] = {}
        self._cache_timestamp = 0
        self._cache_ttl_ms = 5000  # 5 second cache TTL
        
        # Service statistics
        self.stats = {
            "total_trades_processed": 0,
            "total_volume_usd": 0.0,
            "buckets_created": 0,
            "buckets_cleaned": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_trade_time": None,
            "started_at": None
        }
    
    async def start(self):
        """Start the analytics service."""
        if self._running:
            return
        
        logger.info("Starting TimeAnalyticsService...")
        
        self._running = True
        self.stats["started_at"] = datetime.now()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"TimeAnalyticsService started with {self.window_minutes}-minute and {self.window_hours}-hour windows")
    
    async def stop(self):
        """Stop the analytics service."""
        if not self._running:
            return
        
        logger.info("Stopping TimeAnalyticsService...")
        
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("TimeAnalyticsService stopped")
    
    async def recover_from_database(self, enable_recovery: bool = True) -> Dict[str, Any]:
        """Recover buckets from PostgreSQL database for warm restart.
        
        Preserves the database recovery logic from the original service.
        
        Args:
            enable_recovery: Whether to enable recovery (can be disabled for testing)
        
        Returns:
            Dictionary with recovery statistics
        """
        recovery_stats = {
            "enabled": enable_recovery,
            "minute_trades_processed": 0,
            "hour_trades_processed": 0,
            "minute_buckets_created": 0,
            "hour_buckets_created": 0,
            "duration_seconds": 0.0,
            "success": False
        }
        
        if not enable_recovery:
            logger.info("Recovery disabled - starting with empty buckets (cold start)")
            recovery_stats["success"] = True
            return recovery_stats
        
        start_time = datetime.now()
        logger.info("Starting TimeAnalyticsService recovery from database...")
        
        try:
            database = get_database()
            
            # Get trade count for progress estimation
            recovery_trade_count = await database.get_recovery_trade_count(hours=24)
            logger.info(f"Recovering from {recovery_trade_count} trades in last 24 hours")
            
            if recovery_trade_count == 0:
                logger.info("No trades found for recovery - starting with empty buckets")
                recovery_stats["success"] = True
                return recovery_stats
            
            # Recover minute-level data (last 60 minutes)
            minute_trades_data = await database.get_trades_for_minute_recovery(minutes=self.window_minutes)
            logger.info(f"Processing {len(minute_trades_data)} trades for minute-level recovery")
            
            # Recover hour-level data (last 24 hours)
            hour_trades_data = await database.get_trades_for_recovery(hours=self.window_hours)
            logger.info(f"Processing {len(hour_trades_data)} trades for hour-level recovery")
            
            # Process minute-level trades
            for trade_data in minute_trades_data:
                trade = Trade(
                    market_ticker=trade_data["market_ticker"],
                    yes_price=trade_data["yes_price"],
                    no_price=trade_data["no_price"],
                    yes_price_dollars=trade_data["yes_price_dollars"],
                    no_price_dollars=trade_data["no_price_dollars"],
                    count=trade_data["count"],
                    taker_side=trade_data["taker_side"],
                    ts=trade_data["ts"]
                )
                
                # Get minute bucket for this trade
                minute_timestamp = self._get_period_timestamp(trade.ts, "minute")
                
                if minute_timestamp not in self.minute_buckets:
                    self.minute_buckets[minute_timestamp] = Bucket(
                        timestamp=minute_timestamp,
                        volume_usd=0.0,
                        trade_count=0
                    )
                    recovery_stats["minute_buckets_created"] += 1
                
                self.minute_buckets[minute_timestamp].add_trade(trade)
                recovery_stats["minute_trades_processed"] += 1
            
            # Process hour-level trades
            for trade_data in hour_trades_data:
                trade = Trade(
                    market_ticker=trade_data["market_ticker"],
                    yes_price=trade_data["yes_price"],
                    no_price=trade_data["no_price"],
                    yes_price_dollars=trade_data["yes_price_dollars"],
                    no_price_dollars=trade_data["no_price_dollars"],
                    count=trade_data["count"],
                    taker_side=trade_data["taker_side"],
                    ts=trade_data["ts"]
                )
                
                # Get hour bucket for this trade
                hour_timestamp = self._get_period_timestamp(trade.ts, "hour")
                
                if hour_timestamp not in self.hour_buckets:
                    self.hour_buckets[hour_timestamp] = Bucket(
                        timestamp=hour_timestamp,
                        volume_usd=0.0,
                        trade_count=0
                    )
                    recovery_stats["hour_buckets_created"] += 1
                
                self.hour_buckets[hour_timestamp].add_trade(trade)
                recovery_stats["hour_trades_processed"] += 1
            
            # Update global stats
            total_volume_recovered = sum(bucket.volume_usd for bucket in self.minute_buckets.values())
            total_trades_recovered = sum(bucket.trade_count for bucket in self.minute_buckets.values())
            
            self.stats["total_trades_processed"] = total_trades_recovered
            self.stats["total_volume_usd"] = total_volume_recovered
            
            # Clear cache after recovery
            self._invalidate_cache()
            
            recovery_stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            recovery_stats["success"] = True
            
            logger.info(f"TimeAnalyticsService recovery completed in {recovery_stats['duration_seconds']:.2f}s")
            logger.info(f"Recovered {recovery_stats['minute_buckets_created']} minute buckets and {recovery_stats['hour_buckets_created']} hour buckets")
            logger.info(f"Total volume: ${total_volume_recovered:,.2f}, Total trades: {total_trades_recovered}")
            
        except Exception as e:
            recovery_stats["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            recovery_stats["error"] = str(e)
            logger.error(f"Error during TimeAnalyticsService recovery: {e}")
            logger.info("Continuing with empty buckets (cold start fallback)")
            # Clear any partially recovered data
            self.minute_buckets.clear()
            self.hour_buckets.clear()
            self._invalidate_cache()
        
        return recovery_stats
    
    def process_trade(self, trade: Trade):
        """
        Process a trade and update both minute and hour buckets.
        
        Single responsibility: Update buckets only.
        This is the ONLY method that modifies bucket data.
        """
        if not self._running:
            return
        
        try:
            # Single timestamp calculation method - eliminates coordination issues
            minute_timestamp = self._get_period_timestamp(trade.ts, "minute")
            hour_timestamp = self._get_period_timestamp(trade.ts, "hour")
            
            # Create or get minute bucket
            if minute_timestamp not in self.minute_buckets:
                self.minute_buckets[minute_timestamp] = Bucket(
                    timestamp=minute_timestamp,
                    volume_usd=0.0,
                    trade_count=0
                )
                self.stats["buckets_created"] += 1
            
            # Create or get hour bucket
            if hour_timestamp not in self.hour_buckets:
                self.hour_buckets[hour_timestamp] = Bucket(
                    timestamp=hour_timestamp,
                    volume_usd=0.0,
                    trade_count=0
                )
                self.stats["buckets_created"] += 1
            
            # Add trade to both buckets
            minute_bucket = self.minute_buckets[minute_timestamp]
            hour_bucket = self.hour_buckets[hour_timestamp]
            
            minute_volume_before = minute_bucket.volume_usd
            minute_bucket.add_trade(trade)
            hour_bucket.add_trade(trade)
            
            # Update global stats
            self.stats["total_trades_processed"] += 1
            self.stats["total_volume_usd"] += (minute_bucket.volume_usd - minute_volume_before)
            self.stats["last_trade_time"] = datetime.now()
            
            # Invalidate cache since data has changed
            self._invalidate_cache()
            
            logger.debug(f"Added trade to buckets {minute_timestamp} and {hour_timestamp}: volume=${minute_bucket.volume_usd:.2f}, count={minute_bucket.trade_count}")
            
        except Exception as e:
            logger.error(f"Error processing trade in TimeAnalyticsService: {e}")
    
    def get_mode_data(self, mode: Literal["hour", "day"], limit: int = 10) -> Dict[str, Any]:
        """
        Get complete data for one mode (hour or day).
        
        Returns current period + summary stats + limited time series.
        Uses efficient caching to avoid redundant calculations.
        
        Args:
            mode: "hour" for 60-minute window, "day" for 24-hour window
            limit: Number of recent time series points to include (default: 10)
        
        Returns:
            Dictionary with current_period, summary_stats, and time_series
        """
        try:
            # Check cache first
            cache_key = f"{mode}_{limit}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                self.stats["cache_hits"] += 1
                return cached_data
            
            self.stats["cache_misses"] += 1
            
            # Determine which buckets to use and current period
            now_ms = int(datetime.now().timestamp() * 1000)
            
            if mode == "hour":
                buckets = self.minute_buckets
                current_timestamp = self._get_period_timestamp(now_ms, "minute")
                window_size = self.window_minutes
                period_duration_ms = 60 * 1000  # 1 minute in ms
            else:  # mode == "day"
                buckets = self.hour_buckets
                current_timestamp = self._get_period_timestamp(now_ms, "hour")
                window_size = self.window_hours
                period_duration_ms = 3600 * 1000  # 1 hour in ms
            
            # Get current period data
            current_period = {
                "timestamp": current_timestamp,
                "volume_usd": 0.0,
                "trade_count": 0
            }
            
            if current_timestamp in buckets:
                bucket = buckets[current_timestamp]
                current_period["volume_usd"] = round(bucket.volume_usd, 2)
                current_period["trade_count"] = bucket.trade_count
            
            # Calculate window cutoff
            window_cutoff = current_timestamp - ((window_size - 1) * period_duration_ms)
            
            # Get all buckets within the window for summary stats
            window_buckets = [
                bucket for timestamp, bucket in buckets.items()
                if timestamp >= window_cutoff
            ]
            
            # Calculate summary statistics efficiently
            if window_buckets:
                volumes = [bucket.volume_usd for bucket in window_buckets]
                trade_counts = [bucket.trade_count for bucket in window_buckets]
                
                summary_stats = {
                    "total_volume_usd": round(sum(volumes), 2),
                    "total_trades": sum(trade_counts),
                    "peak_volume_usd": round(max(volumes), 2),
                    "peak_trades": max(trade_counts)
                }
            else:
                summary_stats = {
                    "total_volume_usd": 0.0,
                    "total_trades": 0,
                    "peak_volume_usd": 0.0,
                    "peak_trades": 0
                }
            
            # Generate limited time series (last 'limit' periods)
            time_series = []
            for i in range(min(limit, window_size)):
                period_offset = limit - 1 - i
                period_timestamp = current_timestamp - (period_offset * period_duration_ms)
                
                if period_timestamp in buckets:
                    time_series.append(buckets[period_timestamp].to_dict())
                else:
                    time_series.append({
                        "timestamp": period_timestamp,
                        "volume_usd": 0.0,
                        "trade_count": 0
                    })
            
            # Sort by timestamp to ensure proper order
            time_series.sort(key=lambda x: x["timestamp"])
            
            result = {
                "current_period": current_period,
                "summary_stats": summary_stats,
                "time_series": time_series,
                "mode": mode,
                "generated_at": now_ms
            }
            
            
            # Cache the result
            self._cache_data(cache_key, result)
            
            logger.debug(f"Generated {mode} mode data: {len(time_series)} time series points, volume=${summary_stats['total_volume_usd']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting {mode} mode data: {e}")
            # Return safe default data
            now_ms = int(datetime.now().timestamp() * 1000)
            current_timestamp = self._get_period_timestamp(now_ms, "minute" if mode == "hour" else "hour")
            
            return {
                "current_period": {
                    "timestamp": current_timestamp,
                    "volume_usd": 0.0,
                    "trade_count": 0
                },
                "summary_stats": {
                    "total_volume_usd": 0.0,
                    "total_trades": 0,
                    "peak_volume_usd": 0.0,
                    "peak_trades": 0
                },
                "time_series": [],
                "mode": mode,
                "generated_at": now_ms
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get service statistics only.
        
        Single responsibility: Return service metrics and status.
        """
        runtime_seconds = None
        if self.stats["started_at"]:
            runtime_seconds = (datetime.now() - self.stats["started_at"]).total_seconds()
        
        return {
            **self.stats,
            "runtime_seconds": runtime_seconds,
            "is_running": self._running,
            "window_minutes": self.window_minutes,
            "window_hours": self.window_hours,
            "active_minute_buckets": len(self.minute_buckets),
            "active_hour_buckets": len(self.hour_buckets),
            "cache_entries": len(self._cache),
            "cache_hit_rate": (
                self.stats["cache_hits"] / (self.stats["cache_hits"] + self.stats["cache_misses"])
                if (self.stats["cache_hits"] + self.stats["cache_misses"]) > 0 else 0.0
            )
        }
    
    # Private helper methods
    
    def _get_period_timestamp(self, timestamp_ms: int, period_type: Literal["minute", "hour"]) -> int:
        """
        Single timestamp calculation method - eliminates coordination issues.
        
        Convert a timestamp to the start of its period (minute or hour).
        This is the ONLY method used for timestamp calculations.
        
        Args:
            timestamp_ms: Input timestamp in milliseconds
            period_type: "minute" or "hour"
        
        Returns:
            Timestamp of period start in milliseconds
        """
        timestamp_seconds = timestamp_ms // 1000
        
        if period_type == "minute":
            period_start_seconds = (timestamp_seconds // 60) * 60
        else:  # period_type == "hour"
            period_start_seconds = (timestamp_seconds // 3600) * 3600
        
        return period_start_seconds * 1000
    
    def _get_cached_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if available and not expired."""
        if cache_key not in self._cache:
            return None
        
        cached_entry = self._cache[cache_key]
        current_time = int(datetime.now().timestamp() * 1000)
        
        if current_time - cached_entry["timestamp"] > self._cache_ttl_ms:
            # Cache expired
            del self._cache[cache_key]
            return None
        
        return cached_entry["data"]
    
    def _cache_data(self, cache_key: str, data: Dict[str, Any]):
        """Cache data with current timestamp."""
        self._cache[cache_key] = {
            "data": data,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        # Simple cache size management - keep only last 10 entries
        if len(self._cache) > 10:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
    
    def _invalidate_cache(self):
        """Clear all cached data when buckets are modified."""
        self._cache.clear()
    
    async def _cleanup_loop(self):
        """Background task to clean up old buckets."""
        try:
            while self._running:
                await self.cleanup_old_buckets()
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
        except asyncio.CancelledError:
            logger.debug("TimeAnalyticsService cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in TimeAnalyticsService cleanup loop: {e}")
    
    async def cleanup_old_buckets(self):
        """Remove buckets older than their respective windows."""
        try:
            # Cleanup minute buckets
            if self.minute_buckets:
                cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes + 5)
                cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
                cutoff_minute = self._get_period_timestamp(cutoff_timestamp, "minute")
                
                to_remove_minutes = [
                    timestamp for timestamp in self.minute_buckets.keys()
                    if timestamp < cutoff_minute
                ]
                
                for timestamp in to_remove_minutes:
                    del self.minute_buckets[timestamp]
                    self.stats["buckets_cleaned"] += 1
                
                if to_remove_minutes:
                    logger.debug(f"Cleaned up {len(to_remove_minutes)} old minute buckets")
                    self._invalidate_cache()  # Clear cache when buckets are removed
            
            # Cleanup hour buckets
            if self.hour_buckets:
                cutoff_time = datetime.now() - timedelta(hours=self.window_hours + 1)
                cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
                cutoff_hour = self._get_period_timestamp(cutoff_timestamp, "hour")
                
                to_remove_hours = [
                    timestamp for timestamp in self.hour_buckets.keys()
                    if timestamp < cutoff_hour
                ]
                
                for timestamp in to_remove_hours:
                    del self.hour_buckets[timestamp]
                    self.stats["buckets_cleaned"] += 1
                
                if to_remove_hours:
                    logger.debug(f"Cleaned up {len(to_remove_hours)} old hour buckets")
                    self._invalidate_cache()  # Clear cache when buckets are removed
            
        except Exception as e:
            logger.error(f"Error cleaning up old buckets in TimeAnalyticsService: {e}")


# Global service instance
_analytics_service_instance = None

def get_analytics_service() -> TimeAnalyticsService:
    """Get the global TimeAnalyticsService instance."""
    global _analytics_service_instance
    if _analytics_service_instance is None:
        _analytics_service_instance = TimeAnalyticsService()
    return _analytics_service_instance