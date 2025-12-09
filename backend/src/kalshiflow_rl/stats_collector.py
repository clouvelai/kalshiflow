"""
Statistics collector for monitoring RL orderbook collector service health and performance.

Tracks snapshots, deltas, database queue health, message rates, and system uptime.
Thread-safe implementation for concurrent access from multiple components.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional
from threading import Lock
from collections import defaultdict, deque

from .data.write_queue import write_queue
from .config import config

logger = logging.getLogger("kalshiflow_rl.stats_collector")


class StatsCollector:
    """
    Lightweight statistics tracking for system monitoring.
    
    Features:
    - Per-market snapshot and delta counts
    - Database queue health monitoring
    - Messages per second calculation (rolling average)
    - WebSocket connection tracking
    - Service uptime counter
    - Thread-safe counters for concurrent access
    """
    
    def __init__(self):
        """Initialize statistics collector."""
        self._lock = Lock()
        self._start_time = time.time()
        
        # Per-market statistics
        self._market_snapshots = defaultdict(int)
        self._market_deltas = defaultdict(int)
        
        # Global counters
        self._total_snapshots = 0
        self._total_deltas = 0
        self._total_messages = 0
        
        # Rate calculation (rolling window)
        self._message_times = deque(maxlen=100)  # Last 100 messages for rate calculation
        self._last_rate_calculation = time.time()
        self._messages_per_second = 0.0
        
        # WebSocket statistics
        self._websocket_connections = 0
        
        # Component references
        self.orderbook_client = None
        self.websocket_manager = None
        
        logger.info("StatsCollector initialized")
    
    def track_snapshot(self, market_ticker: str):
        """
        Track a processed orderbook snapshot.
        
        Args:
            market_ticker: Market ticker symbol
        """
        with self._lock:
            self._market_snapshots[market_ticker] += 1
            self._total_snapshots += 1
            self._total_messages += 1
            self._message_times.append(time.time())
            self._update_message_rate()
    
    def track_delta(self, market_ticker: str):
        """
        Track a processed orderbook delta.
        
        Args:
            market_ticker: Market ticker symbol
        """
        with self._lock:
            self._market_deltas[market_ticker] += 1
            self._total_deltas += 1
            self._total_messages += 1
            self._message_times.append(time.time())
            self._update_message_rate()
    
    def track_websocket_connection(self, delta: int = 1):
        """
        Track WebSocket connection count change.
        
        Args:
            delta: Change in connection count (+1 for new, -1 for closed)
        """
        with self._lock:
            self._websocket_connections += delta
    
    def _update_message_rate(self):
        """Update messages per second calculation (internal, must hold lock)."""
        current_time = time.time()
        
        # Only recalculate every second to avoid excessive computation
        if current_time - self._last_rate_calculation < 1.0:
            return
        
        if len(self._message_times) >= 2:
            # Calculate rate from message timestamps
            time_span = self._message_times[-1] - self._message_times[0]
            if time_span > 0:
                self._messages_per_second = (len(self._message_times) - 1) / time_span
        else:
            self._messages_per_second = 0.0
        
        self._last_rate_calculation = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current statistics snapshot.
        
        Returns:
            Dictionary with all statistics
        """
        with self._lock:
            uptime = time.time() - self._start_time
            
            # Get database queue stats
            queue_stats = write_queue.get_stats() if write_queue else {}
            
            # Get component stats if available
            orderbook_stats = {}
            if self.orderbook_client:
                orderbook_stats = self.orderbook_client.get_stats()
            
            websocket_stats = {}
            if self.websocket_manager:
                websocket_stats = self.websocket_manager.get_stats()
            
            return {
                "markets_active": len(self._market_snapshots),
                "snapshots_processed": self._total_snapshots,
                "deltas_processed": self._total_deltas,
                "messages_per_second": round(self._messages_per_second, 2),
                "db_queue_size": queue_stats.get("queue_size", 0),
                "db_queue_healthy": queue_stats.get("is_healthy", True),
                "db_messages_written": queue_stats.get("messages_written", 0),
                "uptime_seconds": int(uptime),
                "last_update_ms": int(time.time() * 1000),
                "websocket_connections": websocket_stats.get("active_connections", 0),
                "per_market": {
                    ticker: {
                        "snapshots": self._market_snapshots[ticker],
                        "deltas": self._market_deltas[ticker]
                    }
                    for ticker in self._market_snapshots
                },
                "orderbook_client": orderbook_stats,
                "websocket_manager": websocket_stats
            }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get simplified statistics summary for health checks.
        
        Returns:
            Simplified statistics dictionary
        """
        stats = self.get_stats()
        return {
            "markets_active": stats["markets_active"],
            "total_messages": self._total_messages,
            "messages_per_second": stats["messages_per_second"],
            "uptime_seconds": stats["uptime_seconds"],
            "db_queue_healthy": stats["db_queue_healthy"],
            "websocket_connections": stats["websocket_connections"]
        }
    
    def is_healthy(self) -> bool:
        """
        Check if statistics indicate healthy operation.
        
        Returns:
            True if all systems appear healthy
        """
        stats = self.get_stats()
        
        # Check database queue health
        if not stats.get("db_queue_healthy", True):
            return False
        
        # Check if we've received any data (after initial startup period)
        uptime = stats.get("uptime_seconds", 0)
        if uptime > 30 and stats.get("snapshots_processed", 0) == 0:
            return False
        
        return True
    
    async def start(self):
        """Start the statistics collector (if any background tasks needed)."""
        logger.info("StatsCollector started")
        # Currently no background tasks, but method exists for future expansion
    
    async def stop(self):
        """Stop the statistics collector."""
        logger.info("StatsCollector stopped")
        # Clean shutdown if needed
    
    def reset(self):
        """Reset all statistics (useful for testing)."""
        with self._lock:
            self._market_snapshots.clear()
            self._market_deltas.clear()
            self._total_snapshots = 0
            self._total_deltas = 0
            self._total_messages = 0
            self._message_times.clear()
            self._messages_per_second = 0.0
            self._websocket_connections = 0
            self._start_time = time.time()
            logger.info("Statistics reset")


# Global statistics collector instance
stats_collector = StatsCollector()


# Export for use in app.py
__all__ = ["stats_collector", "StatsCollector"]