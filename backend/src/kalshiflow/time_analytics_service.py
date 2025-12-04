"""
TimeAnalyticsService for minute-based aggregation over the last hour.

Provides time series data for volume (USD) and trade count aggregated by minute
for the last 60 minutes with real-time updates.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from .models import Trade


logger = logging.getLogger(__name__)


@dataclass
class MinuteBucket:
    """Data structure for a single minute's aggregated data."""
    timestamp: int  # Timestamp in milliseconds for the start of the minute
    volume_usd: float  # Total USD volume for the minute
    trade_count: int  # Total number of trades for the minute
    
    def add_trade(self, trade: Trade):
        """Add a trade to this minute bucket.
        
        Volume Calculation:
        - USD Volume = trade.count × price_in_dollars
        - For YES trades: count × yes_price_dollars  
        - For NO trades: count × no_price_dollars
        - This represents the actual monetary value traded (CORRECT approach)
        - Alternative approaches like counting contracts only are incorrect for USD volume
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


@dataclass  
class HourBucket:
    """Data structure for a single hour's aggregated data."""
    timestamp: int  # Timestamp in milliseconds for the start of the hour
    volume_usd: float  # Total USD volume for the hour
    trade_count: int  # Total number of trades for the hour
    
    def add_trade(self, trade: Trade):
        """Add a trade to this hour bucket.
        
        Volume Calculation:
        - USD Volume = trade.count × price_in_dollars
        - For YES trades: count × yes_price_dollars  
        - For NO trades: count × no_price_dollars
        - This represents the actual monetary value traded (CORRECT approach)
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
    """Service for maintaining dual time series analytics: minute-based (1 hour) and hour-based (1 day)."""
    
    def __init__(self, window_minutes: int = 60, window_hours: int = 24):
        """Initialize the analytics service.
        
        Args:
            window_minutes: Number of minutes to maintain in the rolling window (default: 60)
            window_hours: Number of hours to maintain in the rolling window (default: 24)
        """
        self.window_minutes = window_minutes
        self.window_hours = window_hours
        self.minute_buckets: Dict[int, MinuteBucket] = {}
        self.hour_buckets: Dict[int, HourBucket] = {}
        self._running = False
        self._cleanup_task = None
        
        # Statistics for monitoring
        self.stats = {
            "total_trades_processed": 0,
            "total_volume_usd": 0.0,
            "buckets_created": 0,
            "buckets_cleaned": 0,
            "last_trade_time": None,
            "started_at": None
        }
    
    async def start(self):
        """Start the analytics service."""
        if self._running:
            return
        
        logger.info("Starting time analytics service...")
        
        self._running = True
        self.stats["started_at"] = datetime.now()
        
        # Start cleanup task to remove old buckets
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Time analytics service started with {self.window_minutes}-minute window")
    
    async def stop(self):
        """Stop the analytics service."""
        if not self._running:
            return
        
        logger.info("Stopping time analytics service...")
        
        self._running = False
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Time analytics service stopped")
    
    def process_trade(self, trade: Trade):
        """Process a trade and update both minute and hour buckets."""
        if not self._running:
            return
        
        try:
            # Get the minute bucket for this trade's timestamp
            minute_timestamp = self._get_minute_timestamp(trade.ts)
            hour_timestamp = self._get_hour_timestamp(trade.ts)
            
            # Get or create the minute bucket
            if minute_timestamp not in self.minute_buckets:
                self.minute_buckets[minute_timestamp] = MinuteBucket(
                    timestamp=minute_timestamp,
                    volume_usd=0.0,
                    trade_count=0
                )
                self.stats["buckets_created"] += 1
            
            # Get or create the hour bucket
            if hour_timestamp not in self.hour_buckets:
                self.hour_buckets[hour_timestamp] = HourBucket(
                    timestamp=hour_timestamp,
                    volume_usd=0.0,
                    trade_count=0
                )
                self.stats["buckets_created"] += 1
            
            # Add the trade to both buckets
            minute_bucket = self.minute_buckets[minute_timestamp]
            hour_bucket = self.hour_buckets[hour_timestamp]
            
            minute_volume_before = minute_bucket.volume_usd
            minute_bucket.add_trade(trade)
            hour_bucket.add_trade(trade)
            
            # Update global stats (only count once for minute bucket)
            self.stats["total_trades_processed"] += 1
            self.stats["total_volume_usd"] += (minute_bucket.volume_usd - minute_volume_before)
            self.stats["last_trade_time"] = datetime.now()
            
            logger.debug(f"Added trade to minute bucket {minute_timestamp} and hour bucket {hour_timestamp}: volume=${minute_bucket.volume_usd:.2f}, count={minute_bucket.trade_count}")
            
        except Exception as e:
            logger.error(f"Error processing trade in analytics service: {e}")
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get dual time series data and summary stats for both hour/minute and day/hour modes."""
        try:
            now = datetime.now()
            now_timestamp_ms = int(now.timestamp() * 1000)
            current_minute_timestamp = self._get_minute_timestamp(now_timestamp_ms)
            current_hour_timestamp = self._get_hour_timestamp(now_timestamp_ms)
            
            # Generate hour/minute mode data (60 minutes × 1-minute buckets)
            hour_minute_series = []
            for i in range(self.window_minutes):
                minute_offset = (self.window_minutes - 1) - i
                minute_timestamp = current_minute_timestamp - (minute_offset * 60 * 1000)
                
                if minute_timestamp in self.minute_buckets:
                    bucket_data = self.minute_buckets[minute_timestamp].to_dict()
                else:
                    bucket_data = {
                        "timestamp": minute_timestamp,
                        "volume_usd": 0.0,
                        "trade_count": 0
                    }
                hour_minute_series.append(bucket_data)
            
            hour_minute_series.sort(key=lambda x: x["timestamp"])
            
            # Generate day/hour mode data (24 hours × 1-hour buckets)
            day_hour_series = []
            for i in range(self.window_hours):
                hour_offset = (self.window_hours - 1) - i
                hour_timestamp = current_hour_timestamp - (hour_offset * 3600 * 1000)
                
                if hour_timestamp in self.hour_buckets:
                    bucket_data = self.hour_buckets[hour_timestamp].to_dict()
                else:
                    bucket_data = {
                        "timestamp": hour_timestamp,
                        "volume_usd": 0.0,
                        "trade_count": 0
                    }
                day_hour_series.append(bucket_data)
            
            day_hour_series.sort(key=lambda x: x["timestamp"])
            
            # Calculate summary statistics for both modes
            hour_minute_summary = self._calculate_summary_stats(hour_minute_series, "minute", current_minute_timestamp)
            day_hour_summary = self._calculate_summary_stats(day_hour_series, "hour", current_hour_timestamp)
            
            logger.debug(f"Generated dual analytics: {len(hour_minute_series)} minute buckets, {len(day_hour_series)} hour buckets")
            
            return {
                "hour_minute_mode": {
                    "time_series": hour_minute_series,
                    "summary_stats": hour_minute_summary
                },
                "day_hour_mode": {
                    "time_series": day_hour_series,
                    "summary_stats": day_hour_summary
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {
                "hour_minute_mode": {
                    "time_series": [],
                    "summary_stats": self._get_empty_summary_stats()
                },
                "day_hour_mode": {
                    "time_series": [],
                    "summary_stats": self._get_empty_summary_stats()
                }
            }
    
    def _calculate_summary_stats(self, time_series_data: List[Dict[str, Any]], period_type: str, current_timestamp: int) -> Dict[str, Any]:
        """Calculate summary statistics from time series data."""
        if not time_series_data:
            return self._get_empty_summary_stats()
        
        # Calculate totals and peaks
        volumes = [data["volume_usd"] for data in time_series_data]
        trade_counts = [data["trade_count"] for data in time_series_data]
        
        peak_volume_usd = max(volumes) if volumes else 0.0
        total_volume_usd = sum(volumes) if volumes else 0.0
        peak_trades = max(trade_counts) if trade_counts else 0
        total_trades = sum(trade_counts) if trade_counts else 0
        
        # Get current period stats (last item in sorted time series)
        current_period_data = time_series_data[-1] if time_series_data else {}
        current_period_volume_usd = current_period_data.get("volume_usd", 0.0)
        current_period_trades = current_period_data.get("trade_count", 0)
        
        # Adjust field names based on period type
        current_volume_key = f"current_{period_type}_volume_usd"
        current_trades_key = f"current_{period_type}_trades"
        
        return {
            "peak_volume_usd": round(peak_volume_usd, 2),
            "total_volume_usd": round(total_volume_usd, 2),
            "peak_trades": peak_trades,
            "total_trades": total_trades,
            current_volume_key: round(current_period_volume_usd, 2),
            current_trades_key: current_period_trades
        }
    
    def _get_empty_summary_stats(self) -> Dict[str, Any]:
        """Get empty summary statistics structure."""
        return {
            "peak_volume_usd": 0.0,
            "total_volume_usd": 0.0,
            "peak_trades": 0,
            "total_trades": 0,
            "current_minute_volume_usd": 0.0,
            "current_minute_trades": 0,
            "current_hour_volume_usd": 0.0,
            "current_hour_trades": 0
        }
    
    def get_current_minute_stats(self) -> Dict[str, Any]:
        """Get statistics for the current minute bucket."""
        try:
            current_minute = self._get_minute_timestamp(int(datetime.now().timestamp() * 1000))
            
            if current_minute in self.minute_buckets:
                bucket = self.minute_buckets[current_minute]
                return {
                    "current_minute_volume_usd": round(bucket.volume_usd, 2),
                    "current_minute_trade_count": bucket.trade_count,
                    "current_minute_timestamp": current_minute
                }
            else:
                return {
                    "current_minute_volume_usd": 0.0,
                    "current_minute_trade_count": 0,
                    "current_minute_timestamp": current_minute
                }
        except Exception as e:
            logger.error(f"Error getting current minute stats: {e}")
            return {
                "current_minute_volume_usd": 0.0,
                "current_minute_trade_count": 0,
                "current_minute_timestamp": 0
            }
    
    def _get_minute_timestamp(self, timestamp_ms: int) -> int:
        """Convert a timestamp to the start of its minute (in milliseconds)."""
        # Convert to seconds, truncate to minute, then back to milliseconds
        timestamp_seconds = timestamp_ms // 1000
        minute_start_seconds = (timestamp_seconds // 60) * 60
        return minute_start_seconds * 1000
    
    def _get_hour_timestamp(self, timestamp_ms: int) -> int:
        """Convert a timestamp to the start of its hour (in milliseconds)."""
        # Convert to seconds, truncate to hour, then back to milliseconds
        timestamp_seconds = timestamp_ms // 1000
        hour_start_seconds = (timestamp_seconds // 3600) * 3600
        return hour_start_seconds * 1000
    
    async def _cleanup_loop(self):
        """Background task to clean up old minute buckets."""
        try:
            while self._running:
                await self.cleanup_old_buckets()
                # Run cleanup every 5 minutes
                await asyncio.sleep(300)
        except asyncio.CancelledError:
            logger.debug("Analytics cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in analytics cleanup loop: {e}")
    
    async def cleanup_old_buckets(self):
        """Remove minute and hour buckets older than their respective windows."""
        try:
            # Cleanup minute buckets
            if self.minute_buckets:
                cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes + 5)  # Add 5 minute buffer
                cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
                cutoff_minute = self._get_minute_timestamp(cutoff_timestamp)
                
                to_remove_minutes = [
                    timestamp for timestamp in self.minute_buckets.keys()
                    if timestamp < cutoff_minute
                ]
                
                for timestamp in to_remove_minutes:
                    del self.minute_buckets[timestamp]
                    self.stats["buckets_cleaned"] += 1
                
                if to_remove_minutes:
                    logger.debug(f"Cleaned up {len(to_remove_minutes)} old minute buckets")
            
            # Cleanup hour buckets
            if self.hour_buckets:
                cutoff_time = datetime.now() - timedelta(hours=self.window_hours + 1)  # Add 1 hour buffer
                cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
                cutoff_hour = self._get_hour_timestamp(cutoff_timestamp)
                
                to_remove_hours = [
                    timestamp for timestamp in self.hour_buckets.keys()
                    if timestamp < cutoff_hour
                ]
                
                for timestamp in to_remove_hours:
                    del self.hour_buckets[timestamp]
                    self.stats["buckets_cleaned"] += 1
                
                if to_remove_hours:
                    logger.debug(f"Cleaned up {len(to_remove_hours)} old hour buckets")
            
        except Exception as e:
            logger.error(f"Error cleaning up old buckets: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analytics service statistics."""
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
            "current_minute_stats": self.get_current_minute_stats()
        }


# Global analytics service instance
_analytics_service_instance = None

def get_analytics_service() -> TimeAnalyticsService:
    """Get the global analytics service instance."""
    global _analytics_service_instance
    if _analytics_service_instance is None:
        _analytics_service_instance = TimeAnalyticsService()
    return _analytics_service_instance