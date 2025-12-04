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


class TimeAnalyticsService:
    """Service for maintaining minute-based time series analytics."""
    
    def __init__(self, window_minutes: int = 60):
        """Initialize the analytics service.
        
        Args:
            window_minutes: Number of minutes to maintain in the rolling window
        """
        self.window_minutes = window_minutes
        self.minute_buckets: Dict[int, MinuteBucket] = {}
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
        """Process a trade and update the appropriate minute bucket."""
        if not self._running:
            return
        
        try:
            # Get the minute bucket for this trade's timestamp
            minute_timestamp = self._get_minute_timestamp(trade.ts)
            
            # Get or create the bucket
            if minute_timestamp not in self.minute_buckets:
                self.minute_buckets[minute_timestamp] = MinuteBucket(
                    timestamp=minute_timestamp,
                    volume_usd=0.0,
                    trade_count=0
                )
                self.stats["buckets_created"] += 1
            
            # Add the trade to the bucket
            bucket = self.minute_buckets[minute_timestamp]
            trade_volume_before = bucket.volume_usd
            bucket.add_trade(trade)
            
            # Update global stats
            self.stats["total_trades_processed"] += 1
            self.stats["total_volume_usd"] += (bucket.volume_usd - trade_volume_before)
            self.stats["last_trade_time"] = datetime.now()
            
            logger.debug(f"Added trade to minute bucket {minute_timestamp}: volume=${bucket.volume_usd:.2f}, count={bucket.trade_count}")
            
        except Exception as e:
            logger.error(f"Error processing trade in analytics service: {e}")
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get the time series data and summary stats for the last hour, including the current minute."""
        try:
            now = datetime.now()
            # Calculate the current minute timestamp
            current_minute_timestamp = self._get_minute_timestamp(int(now.timestamp() * 1000))
            
            # Create a complete list of minute buckets for the window
            time_series_data = []
            
            # Generate 60 minutes ending with the current minute
            for i in range(self.window_minutes):
                # Start from (window_minutes - 1) minutes ago and go to current minute
                minute_offset = (self.window_minutes - 1) - i
                minute_timestamp = current_minute_timestamp - (minute_offset * 60 * 1000)
                
                if minute_timestamp in self.minute_buckets:
                    bucket_data = self.minute_buckets[minute_timestamp].to_dict()
                else:
                    # Create empty bucket for minutes with no trades
                    bucket_data = {
                        "timestamp": minute_timestamp,
                        "volume_usd": 0.0,
                        "trade_count": 0
                    }
                
                time_series_data.append(bucket_data)
            
            # Sort by timestamp to ensure proper ordering
            time_series_data.sort(key=lambda x: x["timestamp"])
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(time_series_data)
            
            logger.debug(f"Generated analytics data with {len(time_series_data)} minute buckets, current minute: {current_minute_timestamp}")
            
            return {
                "time_series": time_series_data,
                "summary": summary_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics data: {e}")
            return {
                "time_series": [],
                "summary": {
                    "peak_volume_usd": 0.0,
                    "total_volume_usd": 0.0,
                    "peak_trades": 0,
                    "total_trades": 0,
                    "current_minute_volume_usd": 0.0,
                    "current_minute_trades": 0
                }
            }
    
    def _calculate_summary_stats(self, time_series_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from time series data."""
        if not time_series_data:
            return {
                "peak_volume_usd": 0.0,
                "total_volume_usd": 0.0,
                "peak_trades": 0,
                "total_trades": 0,
                "current_minute_volume_usd": 0.0,
                "current_minute_trades": 0
            }
        
        # Calculate totals and peaks
        volumes = [data["volume_usd"] for data in time_series_data]
        trade_counts = [data["trade_count"] for data in time_series_data]
        
        peak_volume_usd = max(volumes) if volumes else 0.0
        total_volume_usd = sum(volumes) if volumes else 0.0
        peak_trades = max(trade_counts) if trade_counts else 0
        total_trades = sum(trade_counts) if trade_counts else 0
        
        # Get current minute stats (last item in sorted time series)
        current_minute_data = time_series_data[-1] if time_series_data else {}
        current_minute_volume_usd = current_minute_data.get("volume_usd", 0.0)
        current_minute_trades = current_minute_data.get("trade_count", 0)
        
        return {
            "peak_volume_usd": round(peak_volume_usd, 2),
            "total_volume_usd": round(total_volume_usd, 2),
            "peak_trades": peak_trades,
            "total_trades": total_trades,
            "current_minute_volume_usd": round(current_minute_volume_usd, 2),
            "current_minute_trades": current_minute_trades
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
        """Remove minute buckets older than the window."""
        try:
            if not self.minute_buckets:
                return
            
            cutoff_time = datetime.now() - timedelta(minutes=self.window_minutes + 5)  # Add 5 minute buffer
            cutoff_timestamp = int(cutoff_time.timestamp() * 1000)
            cutoff_minute = self._get_minute_timestamp(cutoff_timestamp)
            
            # Find buckets to remove
            to_remove = [
                timestamp for timestamp in self.minute_buckets.keys()
                if timestamp < cutoff_minute
            ]
            
            # Remove old buckets
            for timestamp in to_remove:
                del self.minute_buckets[timestamp]
                self.stats["buckets_cleaned"] += 1
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} old minute buckets")
            
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
            "active_buckets": len(self.minute_buckets),
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