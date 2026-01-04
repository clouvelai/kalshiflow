"""
Orderbook signal aggregation for 10-second bucket metrics.

Aggregates real-time orderbook state into time-bucketed signals for:
- Spread OHLC tracking (both YES and NO sides)
- Volume imbalance calculation
- BBO depth averaging
- Activity metrics

This replaces raw delta storage with lightweight aggregated metrics,
reducing storage from ~1MB/market/day to ~1.7KB/market/day.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from .database import rl_db

logger = logging.getLogger("kalshiflow_rl.orderbook_signals")


# Threshold for "large order" detection
LARGE_ORDER_THRESHOLD = 1000


@dataclass
class BucketState:
    """State accumulated within a single 10-second bucket for one market."""

    # Bucket timing
    bucket_start: datetime
    bucket_seconds: int = 10

    # NO side spread tracking (OHLC)
    no_spread_first: Optional[int] = None  # Open
    no_spread_high: Optional[int] = None
    no_spread_low: Optional[int] = None
    no_spread_last: Optional[int] = None  # Close

    # YES side spread tracking (OHLC)
    yes_spread_first: Optional[int] = None
    yes_spread_high: Optional[int] = None
    yes_spread_low: Optional[int] = None
    yes_spread_last: Optional[int] = None

    # NO side volume accumulators (for averaging)
    no_bid_volumes: List[int] = field(default_factory=list)
    no_ask_volumes: List[int] = field(default_factory=list)

    # YES side volume accumulators
    yes_bid_volumes: List[int] = field(default_factory=list)
    yes_ask_volumes: List[int] = field(default_factory=list)

    # BBO depth accumulators
    no_bid_bbo_sizes: List[int] = field(default_factory=list)
    no_ask_bbo_sizes: List[int] = field(default_factory=list)
    yes_bid_bbo_sizes: List[int] = field(default_factory=list)
    yes_ask_bbo_sizes: List[int] = field(default_factory=list)

    # Activity counters
    snapshot_count: int = 0
    delta_count: int = 0
    large_order_count: int = 0

    def update_from_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Update bucket state from an orderbook snapshot.

        Args:
            snapshot: Dict from SharedOrderbookState.get_snapshot() with:
                - no_spread, yes_spread (int, cents)
                - no_bids, no_asks, yes_bids, yes_asks (Dict[str, int] price->size)
        """
        self.snapshot_count += 1

        # Extract spreads
        no_spread = snapshot.get("no_spread")
        yes_spread = snapshot.get("yes_spread")

        # Update NO spread OHLC
        if no_spread is not None:
            if self.no_spread_first is None:
                self.no_spread_first = no_spread
            if self.no_spread_high is None or no_spread > self.no_spread_high:
                self.no_spread_high = no_spread
            if self.no_spread_low is None or no_spread < self.no_spread_low:
                self.no_spread_low = no_spread
            self.no_spread_last = no_spread

        # Update YES spread OHLC
        if yes_spread is not None:
            if self.yes_spread_first is None:
                self.yes_spread_first = yes_spread
            if self.yes_spread_high is None or yes_spread > self.yes_spread_high:
                self.yes_spread_high = yes_spread
            if self.yes_spread_low is None or yes_spread < self.yes_spread_low:
                self.yes_spread_low = yes_spread
            self.yes_spread_last = yes_spread

        # Extract and sum volumes (levels 1-5)
        no_bids = snapshot.get("no_bids", {})
        no_asks = snapshot.get("no_asks", {})
        yes_bids = snapshot.get("yes_bids", {})
        yes_asks = snapshot.get("yes_asks", {})

        # Total volumes
        no_bid_vol = sum(no_bids.values()) if no_bids else 0
        no_ask_vol = sum(no_asks.values()) if no_asks else 0
        yes_bid_vol = sum(yes_bids.values()) if yes_bids else 0
        yes_ask_vol = sum(yes_asks.values()) if yes_asks else 0

        self.no_bid_volumes.append(no_bid_vol)
        self.no_ask_volumes.append(no_ask_vol)
        self.yes_bid_volumes.append(yes_bid_vol)
        self.yes_ask_volumes.append(yes_ask_vol)

        # BBO sizes (best bid/ask)
        if no_bids:
            best_no_bid_price = max(int(p) for p in no_bids.keys())
            self.no_bid_bbo_sizes.append(no_bids.get(str(best_no_bid_price), 0))
        if no_asks:
            best_no_ask_price = min(int(p) for p in no_asks.keys())
            self.no_ask_bbo_sizes.append(no_asks.get(str(best_no_ask_price), 0))
        if yes_bids:
            best_yes_bid_price = max(int(p) for p in yes_bids.keys())
            self.yes_bid_bbo_sizes.append(yes_bids.get(str(best_yes_bid_price), 0))
        if yes_asks:
            best_yes_ask_price = min(int(p) for p in yes_asks.keys())
            self.yes_ask_bbo_sizes.append(yes_asks.get(str(best_yes_ask_price), 0))

        # Detect large orders (any level > threshold)
        all_sizes = (
            list(no_bids.values())
            + list(no_asks.values())
            + list(yes_bids.values())
            + list(yes_asks.values())
        )
        large_orders = sum(1 for size in all_sizes if size >= LARGE_ORDER_THRESHOLD)
        if large_orders > 0:
            self.large_order_count += large_orders

    def increment_delta_count(self) -> None:
        """Increment delta count when a delta is processed."""
        self.delta_count += 1

    def to_signal_data(self) -> Dict[str, Any]:
        """
        Convert bucket state to signal data dict for database insertion.

        Returns:
            Dict matching orderbook_signals table schema
        """

        def safe_avg(values: List[int]) -> Optional[int]:
            return int(sum(values) / len(values)) if values else None

        def calc_imbalance(bid_vols: List[int], ask_vols: List[int]) -> Optional[float]:
            """Calculate imbalance ratio: bid_vol / (bid_vol + ask_vol)."""
            if not bid_vols or not ask_vols:
                return None
            total_bid = sum(bid_vols)
            total_ask = sum(ask_vols)
            total = total_bid + total_ask
            if total == 0:
                return 0.5  # Neutral when no volume
            return round(total_bid / total, 4)

        return {
            "bucket_seconds": self.bucket_seconds,
            # NO spread OHLC
            "no_spread_open": self.no_spread_first,
            "no_spread_high": self.no_spread_high,
            "no_spread_low": self.no_spread_low,
            "no_spread_close": self.no_spread_last,
            # YES spread OHLC
            "yes_spread_open": self.yes_spread_first,
            "yes_spread_high": self.yes_spread_high,
            "yes_spread_low": self.yes_spread_low,
            "yes_spread_close": self.yes_spread_last,
            # Volume averages
            "no_bid_volume_avg": safe_avg(self.no_bid_volumes),
            "no_ask_volume_avg": safe_avg(self.no_ask_volumes),
            "no_imbalance_ratio": calc_imbalance(self.no_bid_volumes, self.no_ask_volumes),
            "yes_bid_volume_avg": safe_avg(self.yes_bid_volumes),
            "yes_ask_volume_avg": safe_avg(self.yes_ask_volumes),
            "yes_imbalance_ratio": calc_imbalance(
                self.yes_bid_volumes, self.yes_ask_volumes
            ),
            # BBO depth averages
            "no_bid_size_at_bbo_avg": safe_avg(self.no_bid_bbo_sizes),
            "no_ask_size_at_bbo_avg": safe_avg(self.no_ask_bbo_sizes),
            "yes_bid_size_at_bbo_avg": safe_avg(self.yes_bid_bbo_sizes),
            "yes_ask_size_at_bbo_avg": safe_avg(self.yes_ask_bbo_sizes),
            # Activity
            "snapshot_count": self.snapshot_count,
            "delta_count": self.delta_count,
            "large_order_count": self.large_order_count,
        }


class OrderbookSignalAggregator:
    """
    Aggregates orderbook snapshots into 10-second signal buckets.

    Usage:
        aggregator = OrderbookSignalAggregator(session_id=123, bucket_seconds=10)
        await aggregator.start()

        # On each snapshot:
        aggregator.record_snapshot("TICKER-XXX", snapshot_data)

        # On each delta (optional):
        aggregator.record_delta("TICKER-XXX")

        # Shutdown:
        await aggregator.stop()
    """

    def __init__(
        self,
        session_id: int,
        bucket_seconds: int = 10,
        flush_interval: float = 10.0,
    ):
        """
        Initialize aggregator.

        Args:
            session_id: Database session ID for this collection run
            bucket_seconds: Size of each bucket in seconds (default 10)
            flush_interval: How often to flush completed buckets to DB
        """
        self.session_id = session_id
        self.bucket_seconds = bucket_seconds
        self.flush_interval = flush_interval

        # Market ticker -> current bucket state
        self._buckets: Dict[str, BucketState] = {}

        # Completed buckets waiting to be flushed
        self._pending_flush: List[Dict[str, Any]] = []

        # Control
        self._running = False
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Stats
        self._signals_flushed = 0
        self._snapshots_processed = 0

        logger.info(
            f"OrderbookSignalAggregator initialized: session={session_id}, "
            f"bucket_seconds={bucket_seconds}, flush_interval={flush_interval}"
        )

    def _get_bucket_start(self, timestamp: Optional[datetime] = None) -> datetime:
        """Get the bucket start time for a given timestamp."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # Round down to nearest bucket boundary
        seconds = timestamp.second
        bucket_start_second = (seconds // self.bucket_seconds) * self.bucket_seconds

        return timestamp.replace(
            second=bucket_start_second, microsecond=0
        )

    def _get_or_create_bucket(self, market_ticker: str) -> BucketState:
        """Get current bucket for market, creating new one if needed."""
        current_bucket_start = self._get_bucket_start()

        existing = self._buckets.get(market_ticker)

        if existing is None or existing.bucket_start != current_bucket_start:
            # Complete old bucket if it exists
            if existing is not None:
                self._complete_bucket(market_ticker, existing)

            # Create new bucket
            new_bucket = BucketState(
                bucket_start=current_bucket_start,
                bucket_seconds=self.bucket_seconds,
            )
            self._buckets[market_ticker] = new_bucket
            return new_bucket

        return existing

    def _complete_bucket(self, market_ticker: str, bucket: BucketState) -> None:
        """Mark a bucket as complete and add to pending flush queue."""
        if bucket.snapshot_count == 0:
            # Skip empty buckets
            return

        signal_data = bucket.to_signal_data()
        signal_data["market_ticker"] = market_ticker
        signal_data["bucket_timestamp"] = bucket.bucket_start

        self._pending_flush.append(signal_data)
        logger.debug(
            f"Bucket completed for {market_ticker}: "
            f"{bucket.snapshot_count} snapshots, spread OHLC {bucket.no_spread_first}->"
            f"{bucket.no_spread_last}"
        )

    def record_snapshot(
        self, market_ticker: str, snapshot: Dict[str, Any]
    ) -> None:
        """
        Record an orderbook snapshot for aggregation.

        Called by orderbook client on each snapshot update.

        Args:
            market_ticker: Market ticker
            snapshot: Snapshot dict from SharedOrderbookState.get_snapshot()
        """
        bucket = self._get_or_create_bucket(market_ticker)
        bucket.update_from_snapshot(snapshot)
        self._snapshots_processed += 1

    def record_delta(self, market_ticker: str) -> None:
        """
        Record a delta event (just increments counter).

        Called by orderbook client on each delta (if tracking).

        Args:
            market_ticker: Market ticker
        """
        bucket = self._get_or_create_bucket(market_ticker)
        bucket.increment_delta_count()

    async def start(self) -> None:
        """Start the background flush loop."""
        if self._running:
            logger.warning("OrderbookSignalAggregator already running")
            return

        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("OrderbookSignalAggregator started")

    async def stop(self) -> None:
        """Stop the flush loop and flush remaining buckets."""
        if not self._running:
            return

        logger.info("Stopping OrderbookSignalAggregator...")
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Complete and flush all remaining buckets
        async with self._lock:
            for market_ticker, bucket in list(self._buckets.items()):
                self._complete_bucket(market_ticker, bucket)
            self._buckets.clear()

            await self._flush_pending()

        logger.info(
            f"OrderbookSignalAggregator stopped. "
            f"Total signals flushed: {self._signals_flushed}, "
            f"Snapshots processed: {self._snapshots_processed}"
        )

    async def _flush_loop(self) -> None:
        """Background loop to periodically flush completed buckets."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)

                async with self._lock:
                    # Check for stale buckets (from previous intervals)
                    current_bucket_start = self._get_bucket_start()
                    for market_ticker, bucket in list(self._buckets.items()):
                        if bucket.bucket_start < current_bucket_start:
                            self._complete_bucket(market_ticker, bucket)
                            # Create fresh bucket
                            self._buckets[market_ticker] = BucketState(
                                bucket_start=current_bucket_start,
                                bucket_seconds=self.bucket_seconds,
                            )

                    await self._flush_pending()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in signal flush loop: {e}")
                await asyncio.sleep(1.0)

    async def _flush_pending(self) -> None:
        """Flush pending signal buckets to database."""
        if not self._pending_flush:
            return

        try:
            count = await rl_db.batch_insert_orderbook_signals(
                self._pending_flush, self.session_id
            )
            self._signals_flushed += count
            logger.debug(f"Flushed {count} signal buckets to database")
            self._pending_flush.clear()

        except Exception as e:
            logger.error(f"Failed to flush signal buckets: {e}")
            # Keep pending for retry (but limit size to prevent memory issues)
            if len(self._pending_flush) > 1000:
                dropped = len(self._pending_flush) - 1000
                self._pending_flush = self._pending_flush[-1000:]
                logger.warning(f"Dropped {dropped} old signal buckets due to flush backlog")

    def get_stats(self) -> Dict[str, Any]:
        """Get current aggregator statistics."""
        return {
            "running": self._running,
            "session_id": self.session_id,
            "bucket_seconds": self.bucket_seconds,
            "active_buckets": len(self._buckets),
            "pending_flush": len(self._pending_flush),
            "signals_flushed": self._signals_flushed,
            "snapshots_processed": self._snapshots_processed,
        }

    def get_current_bucket_signals(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get the current (in-progress) bucket's signal data for a market.

        Returns the live aggregated data within the current 10-second window.

        Args:
            market_ticker: Market ticker to get signals for

        Returns:
            Signal data dict or None if no bucket exists
        """
        bucket = self._buckets.get(market_ticker)
        if bucket is None:
            return None
        return bucket.to_signal_data()


# Global aggregator instance (lazy initialized)
_signal_aggregator: Optional[OrderbookSignalAggregator] = None


def get_signal_aggregator() -> Optional[OrderbookSignalAggregator]:
    """Get the global signal aggregator instance."""
    return _signal_aggregator


def create_signal_aggregator(
    session_id: int,
    bucket_seconds: int = 10,
    flush_interval: float = 10.0,
) -> OrderbookSignalAggregator:
    """
    Create and set the global signal aggregator instance.

    Args:
        session_id: Database session ID
        bucket_seconds: Size of each bucket
        flush_interval: How often to flush to DB

    Returns:
        The created aggregator
    """
    global _signal_aggregator
    _signal_aggregator = OrderbookSignalAggregator(
        session_id=session_id,
        bucket_seconds=bucket_seconds,
        flush_interval=flush_interval,
    )
    return _signal_aggregator


async def stop_signal_aggregator() -> None:
    """Stop and clear the global signal aggregator."""
    global _signal_aggregator
    if _signal_aggregator is not None:
        await _signal_aggregator.stop()
        _signal_aggregator = None
