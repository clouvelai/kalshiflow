"""
In-memory trade duplicate detection with sliding window cache.

This module provides efficient duplicate detection for incoming trades using
a time-windowed cache of recently seen trade hashes. Duplicates are identified
by creating a unique hash from trade attributes and checking against the cache.
"""

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Set, Optional, Dict, Any, Tuple
import hashlib

from .models import Trade


class TradeDuplicateDetector:
    """Detects duplicate trades using a time-windowed cache of trade hashes.
    
    Maintains a sliding window of recently seen trades to detect duplicates
    that may occur due to network issues, retries, or feed problems. Uses
    efficient hashing and automatic cleanup to minimize memory usage.
    """
    
    def __init__(self, window_minutes: int = 5):
        """Initialize the duplicate detector with a time window.
        
        Args:
            window_minutes: How long to keep trade hashes in cache (default 5 minutes)
        """
        self.window_minutes = window_minutes
        self.window_ms = window_minutes * 60 * 1000  # Convert to milliseconds
        
        # Main cache: set of trade hashes for O(1) lookup
        self._trade_hashes: Set[str] = set()
        
        # Time-ordered queue for efficient cleanup: (timestamp_ms, hash)
        self._hash_queue: deque = deque()
        
        # Statistics tracking
        self._stats = {
            "total_checks": 0,
            "duplicates_detected": 0,
            "unique_trades": 0,
            "cache_size": 0,
            "last_cleanup_time": None,
            "detection_rate": 0.0
        }
        
        # Cleanup management
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_cleanup_ms = int(datetime.now().timestamp() * 1000)
        
        self.logger = logging.getLogger(__name__)
    
    def _generate_trade_hash(self, trade: Trade) -> str:
        """Generate a unique hash for a trade.
        
        Creates a hash from the trade's identifying attributes. This ensures
        that identical trades can be detected as duplicates while allowing
        different trades on the same market to pass through.
        
        Args:
            trade: The trade to hash
            
        Returns:
            A unique hash string for the trade
        """
        # Create a string representation of unique trade attributes
        # Include all fields that uniquely identify a trade
        trade_string = f"{trade.market_ticker}|{trade.ts}|{trade.yes_price}|{trade.no_price}|{trade.count}|{trade.taker_side}"
        
        # Use SHA-256 for consistent, fast hashing
        return hashlib.sha256(trade_string.encode()).hexdigest()
    
    def is_duplicate(self, trade: Trade) -> bool:
        """Check if a trade is a duplicate and update the cache.
        
        This method performs three operations atomically:
        1. Checks if the trade hash exists in the cache
        2. If not duplicate, adds it to the cache
        3. Updates statistics
        
        Args:
            trade: The trade to check
            
        Returns:
            True if the trade is a duplicate, False otherwise
        """
        # Update check counter
        self._stats["total_checks"] += 1
        
        # Generate hash for this trade
        trade_hash = self._generate_trade_hash(trade)
        
        # Check if we've seen this trade before
        if trade_hash in self._trade_hashes:
            # Duplicate detected
            self._stats["duplicates_detected"] += 1
            self._update_detection_rate()
            
            # Log duplicate detection (at debug level to avoid spam)
            self.logger.debug(
                f"Duplicate trade detected: {trade.market_ticker} @ "
                f"{trade.yes_price}¢ YES/{trade.no_price}¢ NO, "
                f"{trade.count} shares, ts={trade.ts}"
            )
            
            return True
        
        # Not a duplicate - add to cache
        self._trade_hashes.add(trade_hash)
        self._hash_queue.append((trade.ts, trade_hash))
        self._stats["unique_trades"] += 1
        self._stats["cache_size"] = len(self._trade_hashes)
        self._update_detection_rate()
        
        # Perform cleanup if needed (inline for simplicity)
        current_time_ms = int(datetime.now().timestamp() * 1000)
        if current_time_ms - self._last_cleanup_ms > 60000:  # Cleanup every minute
            self._cleanup_old_entries()
        
        return False
    
    def _cleanup_old_entries(self):
        """Remove trade hashes older than the window from the cache.
        
        This method is called periodically to prevent unbounded memory growth.
        It removes entries older than the configured window while maintaining
        the integrity of the cache.
        """
        current_time_ms = int(datetime.now().timestamp() * 1000)
        window_start_ms = current_time_ms - self.window_ms
        
        # Remove old entries from the front of the queue
        removed_count = 0
        while self._hash_queue and self._hash_queue[0][0] < window_start_ms:
            timestamp_ms, old_hash = self._hash_queue.popleft()
            self._trade_hashes.discard(old_hash)
            removed_count += 1
        
        # Update statistics
        self._stats["cache_size"] = len(self._trade_hashes)
        self._stats["last_cleanup_time"] = datetime.now().isoformat()
        self._last_cleanup_ms = current_time_ms
        
        if removed_count > 0:
            self.logger.debug(
                f"Cleaned up {removed_count} old trade hashes. "
                f"Cache size: {self._stats['cache_size']}"
            )
    
    def _update_detection_rate(self):
        """Update the duplicate detection rate statistic."""
        if self._stats["total_checks"] > 0:
            self._stats["detection_rate"] = (
                self._stats["duplicates_detected"] / self._stats["total_checks"]
            )
    
    async def start(self):
        """Start the duplicate detector and its cleanup task."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self.logger.info(
            f"Trade duplicate detector started with {self.window_minutes} minute window"
        )
    
    async def stop(self):
        """Stop the duplicate detector and cleanup task."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Trade duplicate detector stopped")
    
    async def _periodic_cleanup(self):
        """Periodically clean up old entries from the cache."""
        while self._running:
            try:
                # Wait for cleanup interval (every minute)
                await asyncio.sleep(60)
                
                # Perform cleanup
                self._cleanup_old_entries()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error during periodic cleanup: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics about duplicate detection.
        
        Returns:
            Dictionary containing detection statistics
        """
        return {
            **self._stats,
            "window_minutes": self.window_minutes,
            "detection_rate_percent": round(self._stats["detection_rate"] * 100, 2)
        }
    
    def reset_stats(self):
        """Reset statistics counters (useful for testing)."""
        self._stats["total_checks"] = 0
        self._stats["duplicates_detected"] = 0
        self._stats["unique_trades"] = 0
        self._stats["detection_rate"] = 0.0
        # Don't reset cache_size as it reflects current state
    
    def clear_cache(self):
        """Clear the entire duplicate cache (useful for testing)."""
        self._trade_hashes.clear()
        self._hash_queue.clear()
        self._stats["cache_size"] = 0
        self.logger.info("Trade duplicate cache cleared")
    
    def __repr__(self) -> str:
        """String representation of the detector state."""
        return (
            f"TradeDuplicateDetector("
            f"window={self.window_minutes}min, "
            f"cache_size={self._stats['cache_size']}, "
            f"duplicates={self._stats['duplicates_detected']}/{self._stats['total_checks']})"
        )