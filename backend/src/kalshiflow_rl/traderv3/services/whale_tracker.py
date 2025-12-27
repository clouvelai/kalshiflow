"""
Whale Tracker Service for TRADER V3.

Maintains a priority queue of the biggest bets (whales) from public trades,
enabling "Follow the Whale" trading strategies.
"""

import asyncio
import heapq
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional

from ..core.event_bus import EventBus, EventType, PublicTradeEvent

logger = logging.getLogger("kalshiflow_rl.traderv3.services.whale_tracker")


# Configuration from environment
WHALE_QUEUE_SIZE = int(os.getenv("WHALE_QUEUE_SIZE", "10"))
WHALE_WINDOW_MINUTES = int(os.getenv("WHALE_WINDOW_MINUTES", "5"))
WHALE_MIN_SIZE_CENTS = int(os.getenv("WHALE_MIN_SIZE_CENTS", "10000"))  # $100 minimum


@dataclass(order=True)
class BigBet:
    """
    A significant trade (whale bet) from the public trades stream.

    Uses max(cost, payout) as the whale_size metric to capture both
    high-conviction bets (expensive) and high-leverage bets (cheap but big payout).

    Attributes:
        market_ticker: Kalshi market ticker
        timestamp_ms: Trade timestamp in milliseconds
        side: Taker side ("yes" or "no")
        price_cents: Trade price in cents (0-100)
        count: Number of contracts traded
    """
    # For heap ordering (negative for max-heap behavior)
    sort_key: int = field(repr=False, compare=True)

    # Actual data fields (not used for comparison)
    market_ticker: str = field(compare=False)
    timestamp_ms: int = field(compare=False)
    side: Literal["yes", "no"] = field(compare=False)
    price_cents: int = field(compare=False)
    count: int = field(compare=False)

    @property
    def whale_size(self) -> int:
        """
        Calculate whale size: max(cost, payout).

        - cost = count * price_cents (what they paid)
        - payout = count * 100 (what they get if right)

        This captures both:
        - High-conviction whales (buying at 80c, high cost)
        - High-leverage whales (buying at 5c, high potential payout)
        """
        cost = self.count * self.price_cents
        payout = self.count * 100
        return max(cost, payout)

    @property
    def cost_cents(self) -> int:
        """Cost in cents."""
        return self.count * self.price_cents

    @property
    def payout_cents(self) -> int:
        """Potential payout in cents."""
        return self.count * 100

    @property
    def cost_dollars(self) -> float:
        """Cost in dollars."""
        return self.cost_cents / 100.0

    @property
    def payout_dollars(self) -> float:
        """Potential payout in dollars."""
        return self.payout_cents / 100.0

    @property
    def whale_size_dollars(self) -> float:
        """Whale size in dollars."""
        return self.whale_size / 100.0

    def age_seconds(self, now_ms: Optional[int] = None) -> float:
        """Calculate age of the bet in seconds."""
        if now_ms is None:
            now_ms = int(time.time() * 1000)
        return (now_ms - self.timestamp_ms) / 1000.0

    def to_dict(self, now_ms: Optional[int] = None) -> Dict[str, Any]:
        """Convert to dictionary for WebSocket broadcast."""
        return {
            "market_ticker": self.market_ticker,
            "side": self.side,
            "price_cents": self.price_cents,
            "count": self.count,
            "cost_dollars": self.cost_dollars,
            "payout_dollars": self.payout_dollars,
            "whale_size_dollars": self.whale_size_dollars,
            "age_seconds": round(self.age_seconds(now_ms), 1),
            "timestamp_ms": self.timestamp_ms,
        }

    @classmethod
    def from_trade_event(cls, event: PublicTradeEvent) -> 'BigBet':
        """Create BigBet from PublicTradeEvent."""
        # Calculate whale size for sort key (negative for max-heap)
        cost = event.count * event.price_cents
        payout = event.count * 100
        whale_size = max(cost, payout)

        return cls(
            sort_key=-whale_size,  # Negative for max-heap behavior
            market_ticker=event.market_ticker,
            timestamp_ms=event.timestamp_ms,
            side=event.side,
            price_cents=event.price_cents,
            count=event.count,
        )


class WhaleTracker:
    """
    Service that tracks the biggest bets (whales) across all markets.

    Features:
    - Subscribes to PUBLIC_TRADE_RECEIVED events from EventBus
    - Maintains a priority queue (max-heap) of biggest bets
    - Sliding window pruning (removes bets older than window)
    - Configurable minimum threshold, queue size, and window
    - Emits WHALE_QUEUE_UPDATED when queue changes

    Configuration (environment variables):
    - WHALE_QUEUE_SIZE: Maximum bets in queue (default: 10)
    - WHALE_WINDOW_MINUTES: Sliding window duration (default: 5)
    - WHALE_MIN_SIZE_CENTS: Minimum whale_size to track (default: 10000 = $100)
    """

    def __init__(
        self,
        event_bus: EventBus,
        queue_size: int = WHALE_QUEUE_SIZE,
        window_minutes: int = WHALE_WINDOW_MINUTES,
        min_size_cents: int = WHALE_MIN_SIZE_CENTS,
    ):
        """
        Initialize whale tracker.

        Args:
            event_bus: V3 EventBus for event subscription/emission
            queue_size: Maximum number of whales to track
            window_minutes: Sliding window duration in minutes
            min_size_cents: Minimum whale_size in cents to track
        """
        self._event_bus = event_bus
        self._queue_size = queue_size
        self._window_minutes = window_minutes
        self._min_size_cents = min_size_cents

        # Priority queue (min-heap with negative values = max-heap)
        self._queue: List[BigBet] = []

        # Statistics
        self._trades_seen = 0
        self._trades_discarded = 0
        self._last_update_time: Optional[float] = None

        # State
        self._running = False
        self._started_at: Optional[float] = None
        self._prune_task: Optional[asyncio.Task] = None

        logger.info(
            f"WhaleTracker initialized: queue_size={queue_size}, "
            f"window={window_minutes}min, min_size=${min_size_cents/100:.2f}"
        )

    async def _handle_trade(self, event: PublicTradeEvent) -> None:
        """
        Handle incoming public trade event.

        Checks if trade qualifies as a whale, adds to queue if so,
        and emits WHALE_QUEUE_UPDATED event.

        Args:
            event: PublicTradeEvent from EventBus
        """
        if not self._running:
            return

        self._trades_seen += 1

        # Create BigBet to calculate whale_size
        bet = BigBet.from_trade_event(event)

        # Check if it meets minimum threshold
        if bet.whale_size < self._min_size_cents:
            self._trades_discarded += 1
            return

        # Add to queue
        heapq.heappush(self._queue, bet)

        # Trim queue to max size (keep only top N)
        while len(self._queue) > self._queue_size:
            heapq.heappop(self._queue)

        self._last_update_time = time.time()

        # Emit whale queue update
        await self._emit_queue_update()

        # Log significant whales
        if bet.whale_size >= 50000:  # $500+ whale
            logger.info(
                f"Whale detected: {bet.market_ticker} {bet.side.upper()} "
                f"${bet.whale_size_dollars:.2f} ({bet.count} @ {bet.price_cents}c)"
            )

    async def _emit_queue_update(self) -> None:
        """Emit WHALE_QUEUE_UPDATED event with current queue state."""
        now_ms = int(time.time() * 1000)

        # Convert queue to list sorted by whale_size (descending)
        sorted_queue = sorted(self._queue, key=lambda b: -b.whale_size)
        queue_data = [bet.to_dict(now_ms) for bet in sorted_queue]

        # Calculate stats
        discard_rate = 0.0
        if self._trades_seen > 0:
            discard_rate = (self._trades_discarded / self._trades_seen) * 100

        stats = {
            "trades_seen": self._trades_seen,
            "trades_discarded": self._trades_discarded,
            "discard_rate_percent": round(discard_rate, 1),
            "queue_size": len(self._queue),
            "window_minutes": self._window_minutes,
            "min_size_dollars": self._min_size_cents / 100.0,
        }

        await self._event_bus.emit_whale_queue(queue_data, stats)

    async def _prune_loop(self) -> None:
        """
        Background task to prune old bets from the queue.

        Runs every 30 seconds to remove bets older than the window.
        """
        while self._running:
            try:
                await asyncio.sleep(30)  # Prune every 30 seconds

                if not self._running:
                    break

                now_ms = int(time.time() * 1000)
                window_ms = self._window_minutes * 60 * 1000
                cutoff_ms = now_ms - window_ms

                # Remove old bets
                old_len = len(self._queue)
                self._queue = [
                    bet for bet in self._queue
                    if bet.timestamp_ms > cutoff_ms
                ]
                heapq.heapify(self._queue)

                removed = old_len - len(self._queue)
                if removed > 0:
                    logger.debug(f"Pruned {removed} old whale bets from queue")
                    await self._emit_queue_update()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in prune loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def start(self) -> None:
        """Start the whale tracker service."""
        if self._running:
            logger.warning("WhaleTracker is already running")
            return

        logger.info("Starting WhaleTracker service")
        self._running = True
        self._started_at = time.time()

        # Subscribe to public trade events
        await self._event_bus.subscribe_to_public_trade(self._handle_trade)
        logger.info("Subscribed to PUBLIC_TRADE_RECEIVED events")

        # Start prune loop
        self._prune_task = asyncio.create_task(self._prune_loop())

        logger.info("WhaleTracker service started")

    async def stop(self) -> None:
        """Stop the whale tracker service."""
        if not self._running:
            return

        logger.info("Stopping WhaleTracker service...")
        self._running = False

        # Cancel prune task
        if self._prune_task and not self._prune_task.done():
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass

        # Clear queue
        self._queue.clear()

        logger.info(
            f"WhaleTracker stopped. Final stats: "
            f"trades_seen={self._trades_seen}, trades_discarded={self._trades_discarded}"
        )

    def is_healthy(self) -> bool:
        """Check if whale tracker is healthy."""
        if not self._running:
            return False

        # Check if prune task is running
        if self._prune_task is None or self._prune_task.done():
            return False

        return True

    def get_queue_state(self) -> Dict[str, Any]:
        """
        Get current queue state for status endpoint.

        Returns:
            Dictionary with queue contents and stats
        """
        now_ms = int(time.time() * 1000)

        # Convert queue to sorted list
        sorted_queue = sorted(self._queue, key=lambda b: -b.whale_size)
        queue_data = [bet.to_dict(now_ms) for bet in sorted_queue]

        # Calculate stats
        discard_rate = 0.0
        if self._trades_seen > 0:
            discard_rate = (self._trades_discarded / self._trades_seen) * 100

        return {
            "queue": queue_data,
            "stats": {
                "trades_seen": self._trades_seen,
                "trades_discarded": self._trades_discarded,
                "discard_rate_percent": round(discard_rate, 1),
                "queue_size": len(self._queue),
                "max_queue_size": self._queue_size,
                "window_minutes": self._window_minutes,
                "min_size_dollars": self._min_size_cents / 100.0,
            },
            "healthy": self.is_healthy(),
            "running": self._running,
            "uptime_seconds": time.time() - self._started_at if self._started_at else 0,
        }

    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        state = self.get_queue_state()
        return {
            "healthy": state["healthy"],
            "running": state["running"],
            "queue_size": state["stats"]["queue_size"],
            "trades_seen": state["stats"]["trades_seen"],
            "trades_discarded": state["stats"]["trades_discarded"],
            "discard_rate_percent": state["stats"]["discard_rate_percent"],
            "uptime_seconds": state["uptime_seconds"],
            "prune_task_running": self._prune_task is not None and not self._prune_task.done(),
            "last_update_time": self._last_update_time,
        }
