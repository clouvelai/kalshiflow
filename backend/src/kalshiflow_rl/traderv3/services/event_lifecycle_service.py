"""
Event Lifecycle Service for TRADER V3.

Processes lifecycle events from Kalshi to discover and track new markets
for the Event Lifecycle Discovery mode. Handles REST lookups, category
filtering, and coordinating orderbook subscriptions.

Purpose:
    EventLifecycleService is the central processing unit for lifecycle discovery.
    It receives raw lifecycle events, enriches them via REST API, filters by
    category, and manages the tracked markets state.

Key Responsibilities:
    1. **Event Processing** - Subscribe to MARKET_LIFECYCLE_EVENT from EventBus
    2. **REST Enrichment** - Lookup market info via trading client (category not in WS)
    3. **Category Filtering** - Accept/reject based on configured categories
    4. **State Management** - Add/update markets in TrackedMarketsState
    5. **Subscription Coordination** - Trigger orderbook subscribe/unsubscribe callbacks
    6. **Audit Trail** - Store ALL lifecycle events to database

Architecture Position:
    Used by:
    - V3Coordinator: Initializes and starts service in lifecycle mode
    - V3LifecycleIntegration: Emits MARKET_LIFECYCLE_EVENT to this service

Design Principles:
    - **REST lookup BEFORE filtering**: Lifecycle WS doesn't include category
    - **Capacity-first checks**: Fast fail before expensive REST lookup
    - **Non-blocking**: Async callback pattern for orderbook subscription
    - **Audit everything**: Store all events regardless of tracking decision
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable, TYPE_CHECKING

from ..core.event_bus import EventBus, EventType, MarketLifecycleEvent
from ..state.tracked_markets import TrackedMarketsState, TrackedMarket, MarketStatus
from ...data.database import RLDatabase

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration

logger = logging.getLogger("kalshiflow_rl.traderv3.services.event_lifecycle_service")


# Configuration from environment
DEFAULT_LIFECYCLE_CATEGORIES = ["sports", "media_mentions", "entertainment", "crypto"]
LIFECYCLE_CATEGORIES = os.getenv(
    "LIFECYCLE_CATEGORIES",
    ",".join(DEFAULT_LIFECYCLE_CATEGORIES)
).lower().split(",")


class EventLifecycleService:
    """
    Service that processes lifecycle events for market discovery.

    Features:
    - Subscribes to MARKET_LIFECYCLE_EVENT from EventBus
    - REST lookup via trading client for market info (includes category)
    - Category filtering with configurable allowed categories
    - Capacity management via TrackedMarketsState
    - Triggers orderbook subscription on tracking, unsubscription on determination
    - Stores all events to lifecycle_events table for audit

    Processing Flow (on 'created' event):
        1. Check capacity (fast fail)
        2. REST lookup GET /markets/{ticker}
        3. Category filter
        4. Persist to TrackedMarketsState and DB
        5. Emit MARKET_TRACKED event
        6. Call orderbook subscribe callback

    Processing Flow (on 'determined' event):
        1. Update TrackedMarketsState status
        2. Update DB status
        3. Emit MARKET_DETERMINED event
        4. Call orderbook unsubscribe callback
    """

    def __init__(
        self,
        event_bus: EventBus,
        tracked_markets: TrackedMarketsState,
        trading_client: "V3TradingClientIntegration",
        db: RLDatabase,
        categories: Optional[List[str]] = None,
    ):
        """
        Initialize event lifecycle service.

        Args:
            event_bus: V3 EventBus for event subscription/emission
            tracked_markets: TrackedMarketsState for market tracking
            trading_client: Trading client integration for REST API calls
            db: Database instance for audit trail persistence
            categories: List of allowed categories (default from env)
        """
        self._event_bus = event_bus
        self._tracked_markets = tracked_markets
        self._trading_client = trading_client
        self._db = db
        self._categories = [c.strip().lower() for c in (categories or LIFECYCLE_CATEGORIES)]

        # Callbacks for orderbook subscription management
        # Set by coordinator after initialization
        self._on_subscribe: Optional[Callable[[str], Awaitable[bool]]] = None
        self._on_unsubscribe: Optional[Callable[[str], Awaitable[bool]]] = None

        # Statistics
        self._events_received = 0
        self._events_by_type: Dict[str, int] = {}
        self._markets_tracked = 0
        self._markets_from_api = 0  # Markets tracked via API discovery
        self._markets_rejected_capacity = 0
        self._markets_rejected_category = 0
        self._rest_lookups = 0
        self._rest_errors = 0
        self._last_event_time: Optional[float] = None

        # State
        self._running = False
        self._started_at: Optional[float] = None

        logger.info(
            f"EventLifecycleService initialized with categories: {self._categories}"
        )

    def set_subscribe_callback(
        self,
        callback: Callable[[str], Awaitable[bool]]
    ) -> None:
        """
        Set callback for orderbook subscription.

        Called when a market is tracked and needs orderbook subscription.

        Args:
            callback: Async function(ticker: str) -> bool
        """
        self._on_subscribe = callback
        logger.debug("Subscribe callback registered")

    def set_unsubscribe_callback(
        self,
        callback: Callable[[str], Awaitable[bool]]
    ) -> None:
        """
        Set callback for orderbook unsubscription.

        Called when a market is determined and needs orderbook unsubscription.

        Args:
            callback: Async function(ticker: str) -> bool
        """
        self._on_unsubscribe = callback
        logger.debug("Unsubscribe callback registered")

    async def _handle_lifecycle_event(self, event: MarketLifecycleEvent) -> None:
        """
        Handle incoming lifecycle event from EventBus.

        Routes to appropriate handler based on event type and stores
        to audit trail.

        Args:
            event: MarketLifecycleEvent from EventBus
        """
        if not self._running:
            return

        self._events_received += 1
        self._last_event_time = time.time()

        lifecycle_type = event.lifecycle_event_type
        market_ticker = event.market_ticker
        payload = event.payload or {}

        # Track by type
        self._events_by_type[lifecycle_type] = self._events_by_type.get(lifecycle_type, 0) + 1

        # Store ALL events to audit trail (regardless of tracking decision)
        await self._store_lifecycle_event(market_ticker, lifecycle_type, payload)

        # Route to appropriate handler
        try:
            if lifecycle_type == "created":
                await self._handle_created(market_ticker, payload)
            elif lifecycle_type == "determined":
                await self._handle_determined(market_ticker, payload)
            elif lifecycle_type == "settled":
                await self._handle_settled(market_ticker, payload)
            else:
                # Store but no action for: activated, deactivated, close_date_updated
                logger.debug(f"Stored {lifecycle_type} event for {market_ticker} (no action)")

        except Exception as e:
            logger.error(f"Error handling {lifecycle_type} for {market_ticker}: {e}", exc_info=True)

    async def _store_lifecycle_event(
        self,
        market_ticker: str,
        event_type: str,
        payload: Dict[str, Any]
    ) -> None:
        """
        Store lifecycle event to audit trail database.

        All events are stored regardless of whether the market is tracked,
        providing a complete audit trail for debugging.

        Args:
            market_ticker: Market ticker from event
            event_type: Lifecycle event type
            payload: Full event payload
        """
        try:
            # Extract Kalshi timestamp from payload
            kalshi_ts = payload.get("open_ts") or payload.get("kalshi_ts") or int(time.time())

            await self._db.insert_lifecycle_event(
                market_ticker=market_ticker,
                event_type=event_type,
                payload=payload,
                kalshi_ts=kalshi_ts,
            )
        except Exception as e:
            logger.error(f"Failed to store lifecycle event: {e}")

    async def _handle_created(self, market_ticker: str, payload: Dict[str, Any]) -> None:
        """
        Handle 'created' lifecycle event.

        Processing flow:
        1. Check capacity (fast fail)
        2. REST lookup for market info (includes category)
        3. Category filter
        4. Persist to state and DB
        5. Emit MARKET_TRACKED event
        6. Request orderbook subscription

        Args:
            market_ticker: Market ticker that was created
            payload: Event payload with timestamps
        """
        # Step 1: Capacity check (fast fail)
        if self._tracked_markets.at_capacity():
            self._markets_rejected_capacity += 1
            logger.debug(f"Rejected {market_ticker}: at capacity")
            return

        # Check if already tracked
        if self._tracked_markets.is_tracked(market_ticker):
            logger.debug(f"Already tracking {market_ticker}")
            return

        # Step 2: REST lookup for market info
        market_info = await self._fetch_market_info(market_ticker)
        if not market_info:
            logger.warning(f"REST lookup failed for {market_ticker}, skipping")
            return

        # Step 3: Category filter
        category = market_info.get("category", "").lower()
        if not self._is_allowed_category(category):
            self._markets_rejected_category += 1
            self._tracked_markets.record_category_rejection()
            logger.debug(f"Rejected {market_ticker}: category '{category}' not allowed")
            return

        # Step 4: Create tracked market and persist
        tracked_market = TrackedMarket(
            ticker=market_ticker,
            event_ticker=market_info.get("event_ticker", ""),
            title=market_info.get("title", ""),
            category=market_info.get("category", ""),
            status=MarketStatus.ACTIVE,
            created_ts=payload.get("open_ts", 0),
            open_ts=payload.get("open_ts", 0),
            close_ts=payload.get("close_ts", 0),
            tracked_at=time.time(),
            market_info=market_info,
            discovery_source="lifecycle_ws",
        )

        # Add to state
        added = await self._tracked_markets.add_market(tracked_market)
        if not added:
            logger.warning(f"Failed to add {market_ticker} to state")
            return

        # Persist to DB
        await self._db.insert_tracked_market(
            market_ticker=market_ticker,
            event_ticker=tracked_market.event_ticker,
            title=tracked_market.title,
            category=tracked_market.category,
            created_ts=tracked_market.created_ts,
            open_ts=tracked_market.open_ts,
            close_ts=tracked_market.close_ts,
            market_info=market_info,
            discovery_source="lifecycle_ws",
        )

        self._markets_tracked += 1

        # Step 5: Emit MARKET_TRACKED event
        await self._event_bus.emit_market_tracked(
            market_ticker=market_ticker,
            category=tracked_market.category,
            market_info=market_info,
        )

        # Step 6: Request orderbook subscription
        if self._on_subscribe:
            try:
                success = await self._on_subscribe(market_ticker)
                if success:
                    logger.info(f"Tracked and subscribed to {market_ticker} ({category})")
                else:
                    logger.warning(f"Tracked {market_ticker} but orderbook subscription failed")
            except Exception as e:
                logger.error(f"Error subscribing to {market_ticker}: {e}")
        else:
            logger.info(f"Tracked {market_ticker} ({category}) - no subscribe callback")

    async def _handle_determined(self, market_ticker: str, payload: Dict[str, Any]) -> None:
        """
        Handle 'determined' lifecycle event.

        Processing flow:
        1. Update state to DETERMINED
        2. Update DB status
        3. Emit MARKET_DETERMINED event
        4. Request orderbook unsubscription

        Args:
            market_ticker: Market ticker that was determined
            payload: Event payload with result info
        """
        # Only process if we're tracking this market
        if not self._tracked_markets.is_tracked(market_ticker):
            logger.debug(f"Ignoring determined for untracked market: {market_ticker}")
            return

        market = self._tracked_markets.get_market(market_ticker)
        if not market or market.status != MarketStatus.ACTIVE:
            logger.debug(f"Market {market_ticker} not active, skipping determined")
            return

        # Extract determination timestamp
        determined_ts = payload.get("kalshi_ts") or int(time.time())
        result = payload.get("result", "")

        # Step 1: Update state
        await self._tracked_markets.update_status(
            market_ticker,
            MarketStatus.DETERMINED,
            determined_ts=determined_ts,
        )

        # Step 2: Update DB
        await self._db.update_tracked_market_status(
            market_ticker=market_ticker,
            status="determined",
            determined_ts=determined_ts,
        )

        # Step 3: Emit MARKET_DETERMINED event
        await self._event_bus.emit_market_determined(
            market_ticker=market_ticker,
            result=result,
            determined_ts=determined_ts,
        )

        # Step 4: Request orderbook unsubscription
        if self._on_unsubscribe:
            try:
                success = await self._on_unsubscribe(market_ticker)
                if success:
                    logger.info(f"Unsubscribed from determined market: {market_ticker}")
                else:
                    logger.warning(f"Failed to unsubscribe from {market_ticker}")
            except Exception as e:
                logger.error(f"Error unsubscribing from {market_ticker}: {e}")
        else:
            logger.info(f"Market determined: {market_ticker} - no unsubscribe callback")

    async def _handle_settled(self, market_ticker: str, payload: Dict[str, Any]) -> None:
        """
        Handle 'settled' lifecycle event.

        Only updates status - no orderbook action (already unsubscribed on determined).

        Args:
            market_ticker: Market ticker that was settled
            payload: Event payload
        """
        # Only process if we're tracking this market
        if not self._tracked_markets.is_tracked(market_ticker):
            logger.debug(f"Ignoring settled for untracked market: {market_ticker}")
            return

        market = self._tracked_markets.get_market(market_ticker)
        if not market:
            return

        # Extract settlement timestamp
        settled_ts = payload.get("kalshi_ts") or int(time.time())

        # Update state
        await self._tracked_markets.update_status(
            market_ticker,
            MarketStatus.SETTLED,
            settled_ts=settled_ts,
        )

        # Update DB
        await self._db.update_tracked_market_status(
            market_ticker=market_ticker,
            status="settled",
            settled_ts=settled_ts,
        )

        logger.info(f"Market settled: {market_ticker}")

    async def _fetch_market_info(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch market info via REST API.

        Args:
            market_ticker: Market ticker to look up

        Returns:
            Market info dict or None if lookup failed
        """
        self._rest_lookups += 1

        try:
            # Use trading client's get_market method
            market_info = await self._trading_client.get_market(market_ticker)

            if market_info:
                logger.debug(f"REST lookup success for {market_ticker}")
                return market_info
            else:
                self._rest_errors += 1
                logger.warning(f"REST lookup returned None for {market_ticker}")
                return None

        except Exception as e:
            self._rest_errors += 1
            logger.error(f"REST lookup error for {market_ticker}: {e}")
            return None

    def _is_allowed_category(self, category: str) -> bool:
        """
        Check if category is in allowed list.

        Uses case-insensitive substring matching to handle
        category variations (e.g., "Sports" vs "sports").

        Args:
            category: Category string from market info

        Returns:
            True if category is allowed, False otherwise
        """
        category_lower = category.lower()

        for allowed in self._categories:
            if allowed in category_lower or category_lower in allowed:
                return True

        return False

    async def track_market_from_api_data(
        self,
        market_info: Dict[str, Any],
    ) -> bool:
        """
        Track a market directly from API data (no REST lookup needed).

        Called by ApiDiscoverySyncer when discovering already-open markets
        via REST API. This bypasses the REST lookup step since the API
        discovery already has the full market data.

        Flow:
            1. Capacity check (fast fail)
            2. Duplicate check
            3. Category filter using _is_allowed_category()
            4. Create TrackedMarket with discovery_source="api"
            5. Persist to state and DB
            6. Emit MARKET_TRACKED event
            7. Call orderbook subscribe callback

        Args:
            market_info: Full market data dict from REST API

        Returns:
            True if tracked successfully, False if rejected
        """
        market_ticker = market_info.get("ticker", "")
        if not market_ticker:
            logger.warning("track_market_from_api_data: No ticker in market_info")
            return False

        # Step 1: Capacity check (fast fail)
        if self._tracked_markets.at_capacity():
            self._markets_rejected_capacity += 1
            logger.debug(f"Rejected {market_ticker}: at capacity")
            return False

        # Step 2: Duplicate check
        if self._tracked_markets.is_tracked(market_ticker):
            logger.debug(f"Already tracking {market_ticker}")
            return False

        # Step 3: Category filter
        category = market_info.get("category", "").lower()
        if not self._is_allowed_category(category):
            self._markets_rejected_category += 1
            self._tracked_markets.record_category_rejection()
            logger.debug(f"Rejected {market_ticker}: category '{category}' not allowed")
            return False

        # Step 4: Create tracked market with discovery_source="api"
        # Parse timestamps from API format
        open_time = market_info.get("open_time", "")
        close_time = market_info.get("close_time", "")

        # Convert ISO timestamps to Unix timestamps
        open_ts = 0
        close_ts = 0
        try:
            from datetime import datetime
            if open_time:
                if isinstance(open_time, str):
                    open_dt = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
                    open_ts = int(open_dt.timestamp())
                else:
                    open_ts = int(open_time)
            if close_time:
                if isinstance(close_time, str):
                    close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                    close_ts = int(close_dt.timestamp())
                else:
                    close_ts = int(close_time)
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse timestamps for {market_ticker}: {e}")

        tracked_market = TrackedMarket(
            ticker=market_ticker,
            event_ticker=market_info.get("event_ticker", ""),
            title=market_info.get("title", ""),
            category=market_info.get("category", ""),
            status=MarketStatus.ACTIVE,
            created_ts=open_ts,  # Use open_ts as created_ts for API-discovered markets
            open_ts=open_ts,
            close_ts=close_ts,
            tracked_at=time.time(),
            market_info=market_info,
            discovery_source="api",
            # Populate volume fields from API data
            volume=market_info.get("volume", 0),
            volume_24h=market_info.get("volume_24h", 0),
            open_interest=market_info.get("open_interest", 0),
        )

        # Step 5: Add to state
        added = await self._tracked_markets.add_market(tracked_market)
        if not added:
            logger.warning(f"Failed to add {market_ticker} to state")
            return False

        # Persist to DB
        await self._db.insert_tracked_market(
            market_ticker=market_ticker,
            event_ticker=tracked_market.event_ticker,
            title=tracked_market.title,
            category=tracked_market.category,
            created_ts=tracked_market.created_ts,
            open_ts=tracked_market.open_ts,
            close_ts=tracked_market.close_ts,
            market_info=market_info,
            discovery_source="api",
        )

        self._markets_tracked += 1
        self._markets_from_api += 1

        # Step 6: Emit MARKET_TRACKED event
        await self._event_bus.emit_market_tracked(
            market_ticker=market_ticker,
            category=tracked_market.category,
            market_info=market_info,
        )

        # Step 7: Request orderbook subscription
        if self._on_subscribe:
            try:
                success = await self._on_subscribe(market_ticker)
                if success:
                    logger.info(f"API discovered and subscribed to {market_ticker} ({category})")
                else:
                    logger.warning(f"API discovered {market_ticker} but orderbook subscription failed")
            except Exception as e:
                logger.error(f"Error subscribing to {market_ticker}: {e}")
        else:
            logger.info(f"API discovered {market_ticker} ({category}) - no subscribe callback")

        return True

    async def start(self) -> None:
        """Start the event lifecycle service."""
        if self._running:
            logger.warning("EventLifecycleService is already running")
            return

        logger.info("Starting EventLifecycleService")
        self._running = True
        self._started_at = time.time()

        # Subscribe to lifecycle events
        await self._event_bus.subscribe_to_market_lifecycle(self._handle_lifecycle_event)
        logger.info("Subscribed to MARKET_LIFECYCLE_EVENT")

        logger.info(
            f"EventLifecycleService started with {len(self._categories)} categories: "
            f"{', '.join(self._categories)}"
        )

    async def stop(self) -> None:
        """Stop the event lifecycle service."""
        if not self._running:
            return

        logger.info("Stopping EventLifecycleService...")
        self._running = False

        # Unsubscribe from EventBus to prevent stale handlers
        self._event_bus.unsubscribe(EventType.MARKET_LIFECYCLE_EVENT, self._handle_lifecycle_event)
        logger.debug("Unsubscribed from MARKET_LIFECYCLE_EVENT")

        logger.info(
            f"EventLifecycleService stopped. Final stats: "
            f"events={self._events_received}, tracked={self._markets_tracked}, "
            f"rejected_capacity={self._markets_rejected_capacity}, "
            f"rejected_category={self._markets_rejected_category}, "
            f"rest_lookups={self._rest_lookups}, rest_errors={self._rest_errors}"
        )

    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "events_received": self._events_received,
            "events_by_type": dict(self._events_by_type),
            "markets_tracked": self._markets_tracked,
            "markets_from_api": self._markets_from_api,
            "rejected_capacity": self._markets_rejected_capacity,
            "rejected_category": self._markets_rejected_category,
            "rest_lookups": self._rest_lookups,
            "rest_errors": self._rest_errors,
            "categories": self._categories,
            "uptime_seconds": uptime,
            "last_event_time": self._last_event_time,
        }

    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        stats = self.get_stats()

        time_since_event = None
        if self._last_event_time:
            time_since_event = time.time() - self._last_event_time

        return {
            "healthy": self.is_healthy(),
            "running": self._running,
            "events_received": stats["events_received"],
            "events_by_type": stats["events_by_type"],
            "markets_tracked": stats["markets_tracked"],
            "markets_from_api": stats["markets_from_api"],
            "rejected_capacity": stats["rejected_capacity"],
            "rejected_category": stats["rejected_category"],
            "rest_lookups": stats["rest_lookups"],
            "rest_errors": stats["rest_errors"],
            "rest_error_rate": (
                stats["rest_errors"] / max(stats["rest_lookups"], 1) * 100
            ),
            "categories": stats["categories"],
            "uptime_seconds": stats["uptime_seconds"],
            "time_since_event": time_since_event,
            "has_subscribe_callback": self._on_subscribe is not None,
            "has_unsubscribe_callback": self._on_unsubscribe is not None,
        }
