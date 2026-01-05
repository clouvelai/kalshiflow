"""
Tracked Markets State for Event Lifecycle Discovery Mode.

This module provides centralized state management for markets tracked via
the lifecycle discovery system. It maintains in-memory state with DB
persistence support for restart recovery.

Purpose:
    TrackedMarketsState is the single source of truth for lifecycle-discovered
    markets, managing capacity limits, status tracking, and providing clean
    APIs for market management.

Key Responsibilities:
    1. **Market Tracking** - Store tracked markets with full metadata
    2. **Capacity Management** - Enforce hard limits on tracked markets
    3. **Status Tracking** - Track market lifecycle status (active/determined/settled)
    4. **DB Persistence** - Load from DB on startup, persist on changes
    5. **Statistics** - Provide stats for frontend display

Architecture Position:
    Used by:
    - EventLifecycleService: Adds/updates tracked markets
    - TrackedMarketsSyncer: Updates market info from REST
    - V3WebSocketManager: Broadcasts tracked markets state
    - Coordinator: Initializes on startup, loads from DB
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.traderv3.state.tracked_markets")


class MarketStatus(Enum):
    """Status of a tracked market in its lifecycle."""
    ACTIVE = "active"           # Trading enabled, subscribed to orderbook
    DETERMINED = "determined"   # Outcome resolved, unsubscribed from orderbook
    SETTLED = "settled"         # Positions liquidated, P&L finalized


@dataclass
class TrackedMarket:
    """
    A single tracked market from lifecycle discovery.

    Contains full market metadata from REST API lookup plus
    tracking state managed by the lifecycle system.

    Attributes:
        ticker: Market ticker (e.g., "KXNFL-25JAN05-DET")
        event_ticker: Parent event ticker
        title: Market title
        category: Market category (e.g., "Sports")
        status: Current lifecycle status
        created_ts: Kalshi creation timestamp (Unix seconds)
        open_ts: Market open timestamp (Unix seconds)
        close_ts: Market close timestamp (Unix seconds)
        determined_ts: When outcome was determined (optional)
        settled_ts: When settled (optional)
        tracked_at: When we started tracking (Unix timestamp)
        market_info: Full REST response cached for UI display
        last_sync: Timestamp of last REST sync

        # Real-time market data (updated by syncer)
        price: Current YES price in cents
        volume: Volume traded
        volume_24h: Volume traded in last 24 hours
        open_interest: Open interest
        yes_bid: Best YES bid
        yes_ask: Best YES ask
    """
    ticker: str
    event_ticker: str = ""
    title: str = ""
    category: str = ""
    status: MarketStatus = MarketStatus.ACTIVE
    created_ts: int = 0
    open_ts: int = 0
    close_ts: int = 0
    determined_ts: Optional[int] = None
    settled_ts: Optional[int] = None
    tracked_at: float = field(default_factory=time.time)
    market_info: Dict[str, Any] = field(default_factory=dict)
    last_sync: Optional[float] = None

    # Real-time market data
    price: int = 0
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    yes_bid: int = 0
    yes_ask: int = 0

    # Discovery source tracking
    discovery_source: str = "lifecycle_ws"  # "lifecycle_ws" | "api" | "db_recovery"

    # Market result (set when determined)
    result: Optional[str] = None  # Market result: "yes", "no", "void", or None if undetermined

    # True Market Open (TMO) data - fetched from candlestick API
    true_market_open: Optional[int] = None  # YES price at market open (cents) from candlestick API
    tmo_fetched_at: Optional[float] = None  # When TMO was fetched
    tmo_fetch_failed: bool = False           # Flag if TMO fetch failed (after retries)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ticker": self.ticker,
            "event_ticker": self.event_ticker,
            "title": self.title,
            "category": self.category,
            "status": self.status.value,
            "created_ts": self.created_ts,
            "open_ts": self.open_ts,
            "close_ts": self.close_ts,
            "determined_ts": self.determined_ts,
            "settled_ts": self.settled_ts,
            "tracked_at": self.tracked_at,
            "last_sync": self.last_sync,
            "price": self.price,
            "volume": self.volume,
            "volume_24h": self.volume_24h,
            "open_interest": self.open_interest,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "discovery_source": self.discovery_source,
            "result": self.result,
            # True Market Open (TMO) data
            "true_market_open": self.true_market_open,
            "tmo_fetched_at": self.tmo_fetched_at,
            "tmo_fetch_failed": self.tmo_fetch_failed,
            # Include time until close for UI
            "time_to_close_seconds": max(0, self.close_ts - int(time.time())) if self.close_ts else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedMarket":
        """Create from dictionary (for DB loading)."""
        status_str = data.get("status", "active")
        try:
            status = MarketStatus(status_str)
        except ValueError:
            status = MarketStatus.ACTIVE

        return cls(
            ticker=data.get("ticker", ""),
            event_ticker=data.get("event_ticker", ""),
            title=data.get("title", ""),
            category=data.get("category", ""),
            status=status,
            created_ts=data.get("created_ts", 0),
            open_ts=data.get("open_ts", 0),
            close_ts=data.get("close_ts", 0),
            determined_ts=data.get("determined_ts"),
            settled_ts=data.get("settled_ts"),
            tracked_at=data.get("tracked_at", time.time()),
            market_info=data.get("market_info", {}),
            last_sync=data.get("last_sync"),
            price=data.get("price", 0),
            volume=data.get("volume", 0),
            volume_24h=data.get("volume_24h", 0),
            open_interest=data.get("open_interest", 0),
            yes_bid=data.get("yes_bid", 0),
            yes_ask=data.get("yes_ask", 0),
            discovery_source=data.get("discovery_source", "lifecycle_ws"),
            result=data.get("result"),
            true_market_open=data.get("true_market_open"),
            tmo_fetched_at=data.get("tmo_fetched_at"),
            tmo_fetch_failed=data.get("tmo_fetch_failed", False),
        )


class TrackedMarketsState:
    """
    Central state container for tracked markets in lifecycle discovery mode.

    Features:
        - In-memory storage with DB persistence support
        - Capacity enforcement with configurable limits
        - Version tracking for change detection
        - Category-based statistics
        - Restart recovery via DB load

    Attributes:
        _markets: Dict mapping ticker -> TrackedMarket
        _max_markets: Hard capacity limit
        _version: Increments on any change
        _lock: Async lock for concurrent access
    """

    def __init__(self, max_markets: int = 50):
        """
        Initialize tracked markets state.

        Args:
            max_markets: Maximum number of markets to track (hard limit)
        """
        self._markets: Dict[str, TrackedMarket] = {}
        self._max_markets = max_markets
        self._version = 0
        self._lock = asyncio.Lock()

        # Statistics tracking
        self._markets_tracked_total = 0
        self._markets_rejected_capacity = 0
        self._markets_rejected_category = 0
        self._markets_determined_today = 0

        # Timestamp tracking
        self._created_at = time.time()
        self._last_update = time.time()

        # Subscription callbacks for orderbook sync
        # Called when markets are added/removed to keep subscriptions in sync
        self._on_market_added: Optional[Callable[[str], Awaitable[bool]]] = None
        self._on_market_removed: Optional[Callable[[str], Awaitable[bool]]] = None

        logger.info(f"TrackedMarketsState initialized (max_markets={max_markets})")

    def set_subscription_callbacks(
        self,
        on_added: Optional[Callable[[str], Awaitable[bool]]] = None,
        on_removed: Optional[Callable[[str], Awaitable[bool]]] = None,
    ) -> None:
        """
        Register callbacks for orderbook subscription management.

        These callbacks are invoked when markets are added/removed from tracking,
        allowing automatic synchronization of orderbook subscriptions.

        Args:
            on_added: Async callback(ticker) called when a market is added
            on_removed: Async callback(ticker) called when a market is removed
        """
        self._on_market_added = on_added
        self._on_market_removed = on_removed
        logger.info("Subscription callbacks registered on TrackedMarketsState")

    # ======== Market Management ========

    async def add_market(self, market: TrackedMarket) -> bool:
        """
        Add a market to tracking.

        Args:
            market: TrackedMarket to add

        Returns:
            True if added successfully, False if at capacity or already tracked
        """
        added = False
        ticker = market.ticker

        async with self._lock:
            # Check if already tracked
            if ticker in self._markets:
                logger.debug(f"Market {ticker} already tracked")
                return False

            # Check capacity
            active_count = sum(1 for m in self._markets.values() if m.status == MarketStatus.ACTIVE)
            if active_count >= self._max_markets:
                logger.warning(f"At capacity ({active_count}/{self._max_markets}), rejecting {ticker}")
                self._markets_rejected_capacity += 1
                return False

            # Add market
            self._markets[ticker] = market
            self._version += 1
            self._last_update = time.time()
            self._markets_tracked_total += 1
            added = True

            logger.info(
                f"Market tracked: {ticker} ({market.category}) - "
                f"{len(self._markets)}/{self._max_markets}"
            )

        # Trigger subscription callback OUTSIDE lock to avoid deadlock
        if added and self._on_market_added:
            try:
                await self._on_market_added(ticker)
            except Exception as e:
                logger.warning(f"Failed to subscribe to {ticker}: {e}")

        return added

    async def update_market(self, ticker: str, **kwargs) -> bool:
        """
        Update a tracked market's fields.

        Args:
            ticker: Market ticker to update
            **kwargs: Fields to update (e.g., price=50, volume=100)

        Returns:
            True if updated, False if market not found
        """
        async with self._lock:
            if ticker not in self._markets:
                return False

            market = self._markets[ticker]

            for key, value in kwargs.items():
                if hasattr(market, key):
                    setattr(market, key, value)

            market.last_sync = time.time()
            self._version += 1
            self._last_update = time.time()

            return True

    async def update_status(self, ticker: str, status: MarketStatus, **kwargs) -> bool:
        """
        Update market status (status transition).

        Args:
            ticker: Market ticker
            status: New status
            **kwargs: Additional fields (e.g., determined_ts=timestamp)

        Returns:
            True if updated, False if market not found
        """
        async with self._lock:
            if ticker not in self._markets:
                return False

            market = self._markets[ticker]
            old_status = market.status
            market.status = status

            # Set timestamps based on status
            if status == MarketStatus.DETERMINED and not market.determined_ts:
                market.determined_ts = int(time.time())
                self._markets_determined_today += 1
            elif status == MarketStatus.SETTLED and not market.settled_ts:
                market.settled_ts = int(time.time())

            # Apply additional updates
            for key, value in kwargs.items():
                if hasattr(market, key):
                    setattr(market, key, value)

            self._version += 1
            self._last_update = time.time()

            logger.info(f"Market {ticker} status: {old_status.value} -> {status.value}")

            return True

    def get_market(self, ticker: str) -> Optional[TrackedMarket]:
        """Get a specific tracked market (thread-safe)."""
        # Note: Using synchronous access since async lock would change API
        # Python GIL makes dict access safe, but we document thread-safety intent
        return self._markets.get(ticker)

    def is_tracked(self, ticker: str) -> bool:
        """Check if a market is tracked (thread-safe)."""
        return ticker in self._markets

    async def remove_market(self, ticker: str) -> bool:
        """
        Remove a market from tracking after settlement cleanup.

        Called when MARKET_DETERMINED event is processed and all cleanup
        is complete. This frees memory by removing the TrackedMarket object.

        Args:
            ticker: Market ticker to remove

        Returns:
            True if market was found and removed, False if not found

        Note:
            Only call this after settlement/determination is fully processed.
            The EventLifecycleService should update status to DETERMINED/SETTLED
            before calling this method.
        """
        removed = False
        status_value = None

        async with self._lock:
            if ticker not in self._markets:
                return False

            market = self._markets[ticker]
            status_value = market.status.value

            del self._markets[ticker]
            self._version += 1
            self._last_update = time.time()
            removed = True

            logger.info(f"Market removed from tracking: {ticker} (was {status_value})")

        # Trigger unsubscribe callback OUTSIDE lock to avoid deadlock
        if removed and self._on_market_removed:
            try:
                await self._on_market_removed(ticker)
            except Exception as e:
                logger.warning(f"Failed to unsubscribe from {ticker}: {e}")

        return removed

    def get_all(self) -> List[TrackedMarket]:
        """Get all tracked markets."""
        return list(self._markets.values())

    def get_active(self) -> List[TrackedMarket]:
        """Get all active (tradeable) markets."""
        return [m for m in self._markets.values() if m.status == MarketStatus.ACTIVE]

    def get_active_tickers(self) -> List[str]:
        """Get tickers of all active markets (for orderbook subscription)."""
        return [m.ticker for m in self._markets.values() if m.status == MarketStatus.ACTIVE]

    def get_by_status(self, status: MarketStatus) -> List[TrackedMarket]:
        """Get markets by status."""
        return [m for m in self._markets.values() if m.status == status]

    def get_by_category(self, category: str) -> List[TrackedMarket]:
        """Get markets by category."""
        return [m for m in self._markets.values() if m.category.lower() == category.lower()]

    # ======== Capacity Management ========

    @property
    def capacity(self) -> int:
        """Get maximum capacity."""
        return self._max_markets

    @property
    def active_count(self) -> int:
        """Get count of active markets."""
        return sum(1 for m in self._markets.values() if m.status == MarketStatus.ACTIVE)

    @property
    def total_count(self) -> int:
        """Get total tracked markets count."""
        return len(self._markets)

    def at_capacity(self) -> bool:
        """Check if at capacity for active markets."""
        return self.active_count >= self._max_markets

    def capacity_remaining(self) -> int:
        """Get remaining capacity for active markets."""
        return max(0, self._max_markets - self.active_count)

    # ======== Statistics ========

    def get_stats(self) -> Dict[str, Any]:
        """Get tracked markets statistics for frontend."""
        # Count by category
        by_category: Dict[str, int] = {}
        for market in self._markets.values():
            if market.status == MarketStatus.ACTIVE:
                cat = market.category or "unknown"
                by_category[cat] = by_category.get(cat, 0) + 1

        # Count by status
        by_status = {
            "active": sum(1 for m in self._markets.values() if m.status == MarketStatus.ACTIVE),
            "determined": sum(1 for m in self._markets.values() if m.status == MarketStatus.DETERMINED),
            "settled": sum(1 for m in self._markets.values() if m.status == MarketStatus.SETTLED),
        }

        return {
            "tracked": self.active_count,
            "capacity": self._max_markets,
            "total": len(self._markets),
            "by_category": by_category,
            "by_status": by_status,
            "determined_today": self._markets_determined_today,
            "tracked_total": self._markets_tracked_total,
            "rejected_capacity": self._markets_rejected_capacity,
            "rejected_category": self._markets_rejected_category,
            "version": self._version,
        }

    # ======== Versioning ========

    @property
    def version(self) -> int:
        """Get current version (increments on change)."""
        return self._version

    def has_changed_since(self, version: int) -> bool:
        """Check if state has changed since a given version."""
        return self._version > version

    # ======== DB Persistence ========

    async def load_from_db(self, db_markets: List[Dict[str, Any]]) -> int:
        """
        Load tracked markets from database on startup.

        Args:
            db_markets: List of market dicts from database

        Returns:
            Number of markets loaded
        """
        async with self._lock:
            loaded = 0
            for data in db_markets:
                try:
                    market = TrackedMarket.from_dict(data)
                    # Only load non-settled markets for active tracking
                    if market.status != MarketStatus.SETTLED:
                        self._markets[market.ticker] = market
                        loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load market from DB: {e}")

            self._version += 1
            self._last_update = time.time()

            logger.info(f"Loaded {loaded} tracked markets from database")
            return loaded

    def get_for_persistence(self) -> List[Dict[str, Any]]:
        """Get all markets in dict format for DB persistence."""
        return [m.to_dict() for m in self._markets.values()]

    # ======== Serialization ========

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get full state snapshot for WebSocket broadcast.

        Returns:
            Dict with markets list and stats suitable for frontend.
        """
        markets_data = [m.to_dict() for m in self._markets.values() if m.status == MarketStatus.ACTIVE]

        # Sort by time to close (soonest first)
        markets_data.sort(key=lambda m: m.get("time_to_close_seconds") or float('inf'))

        return {
            "markets": markets_data,
            "stats": self.get_stats(),
            "version": self._version,
            "timestamp": time.time(),
        }

    def get_market_info_update(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get market info update for single market broadcast.

        Args:
            ticker: Market ticker

        Returns:
            Dict with updated market data, or None if not found.
        """
        market = self._markets.get(ticker)
        if not market:
            return None

        return {
            "ticker": ticker,
            "price": market.price,
            "volume": market.volume,
            "open_interest": market.open_interest,
            "yes_bid": market.yes_bid,
            "yes_ask": market.yes_ask,
            "last_sync": market.last_sync,
        }

    # ======== Health ========

    def is_healthy(self) -> bool:
        """Check if state is healthy."""
        return True  # State container is always healthy if it exists

    def get_health_details(self) -> Dict[str, Any]:
        """Get health details for monitoring."""
        return {
            "healthy": True,
            "active_markets": self.active_count,
            "total_markets": len(self._markets),
            "capacity": self._max_markets,
            "at_capacity": self.at_capacity(),
            "version": self._version,
            "uptime_seconds": time.time() - self._created_at,
            "last_update": self._last_update,
        }

    # ======== Utility ========

    def record_category_rejection(self) -> None:
        """Record a category-based rejection for stats."""
        self._markets_rejected_category += 1

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at midnight)."""
        self._markets_determined_today = 0
        logger.info("Daily tracked markets stats reset")
