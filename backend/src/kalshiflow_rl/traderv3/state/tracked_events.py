"""
Tracked Events State for Event Lifecycle Discovery Mode.

This module provides event-level grouping on top of TrackedMarketsState.
A TrackedEvent groups 1-N markets under a parent event, enabling
event-level lifecycle tracking, timeline generation, and early bird detection.

Key Responsibilities:
    1. **Event Grouping** - Group markets under parent events
    2. **Lifecycle Tracking** - Derive event status from constituent markets
    3. **Timeline Generation** - Chronological milestones (opens, closes, determinations)
    4. **Serialization** - to_dict/from_dict for DB persistence + WebSocket broadcast
    5. **Version Tracking** - Change detection for efficient WS broadcasts

Architecture Position:
    Used by:
    - EventLifecycleService: Creates/updates events when markets are discovered
    - LifecyclePersistenceService: Persists events to DB
    - V3WebSocketManager: Broadcasts lifecycle timeline to frontend
    - EarlyBirdService: Queries recently activated events
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.traderv3.state.tracked_events")


class EventStatus(Enum):
    """Derived status of a tracked event from its constituent markets."""
    PENDING = "pending"           # All markets pending (not yet open)
    ACTIVE = "active"             # At least one market is active
    PARTIALLY_DETERMINED = "partially_determined"  # Some markets determined, some active
    DETERMINED = "determined"     # All markets determined
    SETTLED = "settled"           # All markets settled


@dataclass
class TrackedEvent:
    """
    A tracked event grouping 1-N markets.

    Attributes:
        event_ticker: Event ticker (e.g., "KXNFL-25JAN05")
        title: Event title
        category: Event category (e.g., "Sports")
        series_ticker: Series ticker for pattern recognition
        mutually_exclusive: Whether markets in this event are mutually exclusive
        status: Derived status from constituent markets
        earliest_open_ts: Earliest market open timestamp (Unix seconds)
        latest_close_ts: Latest market close timestamp (Unix seconds)
        first_seen_at: When we first discovered this event
        discovery_source: "lifecycle_ws" | "api" | "db_recovery"
        market_tickers: List of child market tickers
    """
    event_ticker: str
    title: str = ""
    category: str = ""
    series_ticker: str = ""
    mutually_exclusive: bool = True
    status: EventStatus = EventStatus.PENDING
    earliest_open_ts: int = 0
    latest_close_ts: int = 0
    first_seen_at: float = field(default_factory=time.time)
    discovery_source: str = "lifecycle_ws"
    market_tickers: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_ticker": self.event_ticker,
            "title": self.title,
            "category": self.category,
            "series_ticker": self.series_ticker,
            "mutually_exclusive": self.mutually_exclusive,
            "status": self.status.value,
            "earliest_open_ts": self.earliest_open_ts,
            "latest_close_ts": self.latest_close_ts,
            "first_seen_at": self.first_seen_at,
            "discovery_source": self.discovery_source,
            "market_tickers": list(self.market_tickers),
            "updated_at": self.updated_at,
            "market_count": len(self.market_tickers),
            "time_to_close_seconds": max(0, self.latest_close_ts - int(time.time())) if self.latest_close_ts else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackedEvent":
        """Create from dictionary (for DB loading)."""
        status_str = data.get("status", "pending")
        try:
            status = EventStatus(status_str)
        except ValueError:
            status = EventStatus.PENDING

        market_tickers = data.get("market_tickers", [])
        if isinstance(market_tickers, str):
            import json
            try:
                market_tickers = json.loads(market_tickers)
            except (json.JSONDecodeError, TypeError):
                market_tickers = []

        return cls(
            event_ticker=data.get("event_ticker", ""),
            title=data.get("title", ""),
            category=data.get("category", ""),
            series_ticker=data.get("series_ticker", ""),
            mutually_exclusive=data.get("mutually_exclusive", True),
            status=status,
            earliest_open_ts=data.get("earliest_open_ts", 0),
            latest_close_ts=data.get("latest_close_ts", 0),
            first_seen_at=data.get("first_seen_at", time.time()),
            discovery_source=data.get("discovery_source", "lifecycle_ws"),
            market_tickers=list(market_tickers),
            updated_at=data.get("updated_at", time.time()),
        )


class TrackedEventsState:
    """
    Central state container for tracked events in lifecycle discovery mode.

    Features:
        - In-memory storage with DB persistence support
        - Version tracking for change detection
        - Timeline generation for frontend display
        - Event status derivation from constituent markets
    """

    def __init__(self):
        self._events: Dict[str, TrackedEvent] = {}
        self._version = 0
        self._lock = asyncio.Lock()
        self._last_update = time.time()

        # Change callbacks for persistence
        self._on_event_changed: Optional[Any] = None

        logger.info("TrackedEventsState initialized")

    def set_change_callback(self, callback) -> None:
        """Register callback for persistence on change."""
        self._on_event_changed = callback

    async def upsert_event(self, event: TrackedEvent) -> bool:
        """
        Insert or update a tracked event.

        Returns:
            True if event was inserted (new), False if updated (existing)
        """
        is_new = False
        async with self._lock:
            existing = self._events.get(event.event_ticker)
            if existing:
                # Update fields that may have changed
                existing.title = event.title or existing.title
                existing.category = event.category or existing.category
                existing.series_ticker = event.series_ticker or existing.series_ticker
                existing.mutually_exclusive = event.mutually_exclusive
                if event.earliest_open_ts:
                    existing.earliest_open_ts = min(
                        existing.earliest_open_ts or event.earliest_open_ts,
                        event.earliest_open_ts
                    ) if existing.earliest_open_ts else event.earliest_open_ts
                if event.latest_close_ts:
                    existing.latest_close_ts = max(
                        existing.latest_close_ts or 0,
                        event.latest_close_ts
                    )
                existing.updated_at = time.time()
            else:
                self._events[event.event_ticker] = event
                is_new = True

            self._version += 1
            self._last_update = time.time()

        # Fire change callback outside lock
        if self._on_event_changed:
            evt = self._events.get(event.event_ticker)
            if evt:
                try:
                    result = self._on_event_changed(evt)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as e:
                    logger.warning(f"Event change callback failed: {e}")

        if is_new:
            logger.info(f"Event tracked: {event.event_ticker} ({event.category})")
        return is_new

    async def add_market_to_event(self, event_ticker: str, market_ticker: str) -> bool:
        """
        Add a market ticker to an event's market list.

        Returns:
            True if added, False if event not found or already present
        """
        async with self._lock:
            event = self._events.get(event_ticker)
            if not event:
                return False
            if market_ticker in event.market_tickers:
                return False
            event.market_tickers.append(market_ticker)
            event.updated_at = time.time()
            self._version += 1
            self._last_update = time.time()
            return True

    async def update_event_status(self, event_ticker: str, status: EventStatus) -> bool:
        """Update an event's derived status."""
        async with self._lock:
            event = self._events.get(event_ticker)
            if not event:
                return False
            if event.status == status:
                return False
            old_status = event.status
            event.status = status
            event.updated_at = time.time()
            self._version += 1
            self._last_update = time.time()
            logger.info(f"Event {event_ticker} status: {old_status.value} -> {status.value}")
            return True

    async def update_close_ts(self, event_ticker: str, close_ts: int) -> bool:
        """Update an event's latest_close_ts."""
        async with self._lock:
            event = self._events.get(event_ticker)
            if not event:
                return False
            event.latest_close_ts = max(event.latest_close_ts or 0, close_ts)
            event.updated_at = time.time()
            self._version += 1
            return True

    def get_event(self, event_ticker: str) -> Optional[TrackedEvent]:
        """Get a specific tracked event."""
        return self._events.get(event_ticker)

    def get_all(self) -> List[TrackedEvent]:
        """Get all tracked events."""
        return list(self._events.values())

    def get_active(self) -> List[TrackedEvent]:
        """Get all active events."""
        return [e for e in self._events.values() if e.status in (EventStatus.PENDING, EventStatus.ACTIVE)]

    def get_by_status(self, status: EventStatus) -> List[TrackedEvent]:
        """Get events by status."""
        return [e for e in self._events.values() if e.status == status]

    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        Get chronological timeline of milestones for frontend display.

        Returns list of timeline items sorted by timestamp, including:
        - Upcoming opens (events with future earliest_open_ts)
        - Active events (currently trading)
        - Closing soon (events closing within 1 hour)
        - Recently determined events
        """
        now = int(time.time())
        items = []

        for event in self._events.values():
            # Upcoming opens
            if event.status == EventStatus.PENDING and event.earliest_open_ts > now:
                items.append({
                    "type": "upcoming_open",
                    "event_ticker": event.event_ticker,
                    "title": event.title,
                    "category": event.category,
                    "timestamp": event.earliest_open_ts,
                    "countdown_seconds": event.earliest_open_ts - now,
                    "market_count": len(event.market_tickers),
                })

            # Active events
            elif event.status == EventStatus.ACTIVE:
                time_to_close = (event.latest_close_ts - now) if event.latest_close_ts else None

                item = {
                    "type": "active",
                    "event_ticker": event.event_ticker,
                    "title": event.title,
                    "category": event.category,
                    "timestamp": event.earliest_open_ts or event.first_seen_at,
                    "market_count": len(event.market_tickers),
                    "mutually_exclusive": event.mutually_exclusive,
                }

                # Flag closing soon (within 1 hour)
                if time_to_close is not None and 0 < time_to_close <= 3600:
                    item["type"] = "closing_soon"
                    item["countdown_seconds"] = time_to_close

                items.append(item)

            # Recently determined
            elif event.status in (EventStatus.DETERMINED, EventStatus.PARTIALLY_DETERMINED):
                items.append({
                    "type": "determined",
                    "event_ticker": event.event_ticker,
                    "title": event.title,
                    "category": event.category,
                    "timestamp": event.updated_at,
                    "market_count": len(event.market_tickers),
                })

        # Sort by timestamp (soonest first)
        items.sort(key=lambda x: x.get("timestamp", 0))
        return items

    def get_snapshot(self) -> Dict[str, Any]:
        """Get full state snapshot for WebSocket broadcast."""
        events_data = [e.to_dict() for e in self._events.values()]
        events_data.sort(key=lambda e: e.get("earliest_open_ts") or float('inf'))

        return {
            "events": events_data,
            "timeline": self.get_timeline(),
            "stats": self.get_stats(),
            "version": self._version,
            "timestamp": time.time(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get tracked events statistics."""
        by_status = {}
        for event in self._events.values():
            s = event.status.value
            by_status[s] = by_status.get(s, 0) + 1

        by_category = {}
        for event in self._events.values():
            if event.status in (EventStatus.PENDING, EventStatus.ACTIVE):
                cat = event.category or "unknown"
                by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total": len(self._events),
            "by_status": by_status,
            "by_category": by_category,
            "total_markets": sum(len(e.market_tickers) for e in self._events.values()),
            "version": self._version,
        }

    @property
    def version(self) -> int:
        return self._version

    def has_changed_since(self, version: int) -> bool:
        return self._version > version

    @property
    def total_count(self) -> int:
        return len(self._events)
