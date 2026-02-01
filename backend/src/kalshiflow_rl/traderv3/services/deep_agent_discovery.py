"""
Deep Agent Discovery Service - Lightweight REST API market discovery.

Replaces the full lifecycle WebSocket pipeline (LifecycleClient + EventLifecycleService
+ ApiDiscoverySyncer) with direct REST API calls to discover tradeable markets.

This is used in non-lifecycle mode (RL_MODE=discovery or config) to populate
TrackedMarketsState so the deep agent can find markets via extraction signals.

Discovery approach:
- Paginates GET /events?status=open&with_nested_markets=true
- Filters by exact match on event.category (uses structured API field, no substring matching)
- Time-filters using min_close_ts at the API level
- Periodic refresh every 5 minutes
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, TYPE_CHECKING

from ..state.tracked_markets import TrackedMarketsState, TrackedMarket, MarketStatus

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.event_bus import EventBus
    from ..core.websocket_manager import V3WebSocketManager

logger = logging.getLogger("kalshiflow_rl.traderv3.services.deep_agent_discovery")

# Discovery timing
DISCOVERY_MIN_HOURS = float(os.getenv("DEEP_AGENT_DISCOVERY_MIN_HOURS", "0.5"))
DISCOVERY_MAX_DAYS = int(os.getenv("DEEP_AGENT_DISCOVERY_MAX_DAYS", "30"))
DISCOVERY_REFRESH_SECONDS = int(os.getenv("DEEP_AGENT_DISCOVERY_REFRESH_SECONDS", "300"))
DISCOVERY_MAX_PAGES = int(os.getenv("DEEP_AGENT_DISCOVERY_MAX_PAGES", "25"))


class DeepAgentDiscoveryService:
    """
    Lightweight market discovery via REST API for the deep agent.

    On startup, fetches open events from the Kalshi API with nested markets
    and populates TrackedMarketsState. Uses the event.category field for
    exact-match filtering (no substring or prefix matching).
    """

    def __init__(
        self,
        tracked_markets: TrackedMarketsState,
        trading_client: 'V3TradingClientIntegration',
        event_bus: Optional['EventBus'] = None,
        categories: Optional[List[str]] = None,
        websocket_manager: Optional['V3WebSocketManager'] = None,
    ):
        self._tracked_markets = tracked_markets
        self._trading_client = trading_client
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager

        # Normalize categories to lowercase for case-insensitive exact matching
        # against the event.category field from the Kalshi API
        if categories:
            self._categories: Set[str] = {c.strip().lower() for c in categories}
        else:
            # Default categories matching Kalshi's API category values
            self._categories = {"politics", "economics"}

        self._refresh_task: Optional[asyncio.Task] = None
        self._running = False

        # Stats
        self._events_seen = 0
        self._events_matched = 0
        self._markets_discovered = 0
        self._markets_added = 0
        self._discovery_runs = 0
        self._last_discovery: Optional[float] = None
        self._errors = 0

        logger.info(
            f"DeepAgentDiscoveryService initialized "
            f"(categories={sorted(self._categories)}, refresh={DISCOVERY_REFRESH_SECONDS}s)"
        )

    async def start(self) -> None:
        """Start discovery: run initial fetch then launch periodic refresh."""
        self._running = True

        count = await self._discover_markets()
        logger.info(f"DeepAgentDiscoveryService: initial discovery added {count} markets")

        if self._event_bus:
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Deep Agent Discovery: found {count} markets across {self._events_matched} events",
                metadata={
                    "feature": "deep_agent_discovery",
                    "markets_added": count,
                    "events_matched": self._events_matched,
                    "categories": sorted(self._categories),
                    "severity": "info",
                }
            )

        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info("DeepAgentDiscoveryService started")

    async def stop(self) -> None:
        """Stop the discovery service."""
        self._running = False
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"DeepAgentDiscoveryService stopped "
            f"(runs={self._discovery_runs}, added={self._markets_added}, errors={self._errors})"
        )

    async def _refresh_loop(self) -> None:
        """Periodically re-discover markets."""
        while self._running:
            try:
                await asyncio.sleep(DISCOVERY_REFRESH_SECONDS)
                if not self._running:
                    break
                count = await self._discover_markets()
                if count > 0:
                    logger.info(f"DeepAgentDiscoveryService: refresh added {count} new markets")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._errors += 1
                logger.error(f"DeepAgentDiscoveryService refresh error: {e}")
                await asyncio.sleep(60)

    async def _discover_markets(self) -> int:
        """
        Fetch open events with nested markets, filter by category, add to tracked state.

        Uses GET /events?status=open&with_nested_markets=true&min_close_ts=X with
        cursor pagination. Filters events by exact match on event.category.
        """
        self._discovery_runs += 1
        self._last_discovery = time.time()
        added = 0

        try:
            now_ts = int(datetime.now(timezone.utc).timestamp())
            min_close_ts = now_ts + int(DISCOVERY_MIN_HOURS * 3600)
            max_close_ts = now_ts + (DISCOVERY_MAX_DAYS * 24 * 3600)

            cursor: Optional[str] = None
            pages_fetched = 0

            while pages_fetched < DISCOVERY_MAX_PAGES:
                # Use the low-level client to pass all params including min_close_ts
                response = await self._trading_client._client.get_events(
                    status="open",
                    with_nested_markets=True,
                    limit=200,
                    cursor=cursor,
                    min_close_ts=min_close_ts,
                )
                pages_fetched += 1

                events = response.get("events", [])
                cursor = response.get("cursor")

                for event in events:
                    self._events_seen += 1
                    event_category = (event.get("category") or "").lower()

                    # Exact category match against configured categories
                    if event_category not in self._categories:
                        continue

                    self._events_matched += 1
                    event_ticker = event.get("event_ticker", "")

                    for market in event.get("markets", []):
                        if market.get("status") not in ("open", "active"):
                            continue

                        # Skip zero-volume markets (no recent activity)
                        if market.get("volume_24h", 0) <= 0:
                            continue

                        ticker = market.get("ticker", "")
                        if not ticker:
                            continue

                        # Time filter: skip markets closing too far out
                        close_ts = self._parse_close_ts(market.get("close_time"))
                        if close_ts and close_ts > max_close_ts:
                            continue

                        self._markets_discovered += 1

                        if self._tracked_markets.is_tracked(ticker):
                            continue

                        # Enrich market with event-level fields
                        market["category"] = event.get("category", "")
                        market["event_ticker"] = event_ticker
                        market["event_title"] = event.get("title", "")

                        tracked = self._build_tracked_market(market)
                        if tracked:
                            success = await self._tracked_markets.add_market(tracked)
                            if success:
                                added += 1

                if not cursor:
                    break

            self._markets_added += added
            logger.info(
                f"DeepAgentDiscoveryService: scanned {self._events_seen} events "
                f"({self._events_matched} matched categories), "
                f"{self._markets_discovered} markets found, {added} newly added "
                f"(pages={pages_fetched})"
            )

            # Broadcast updated tracked markets to all connected clients
            if added > 0 and self._websocket_manager:
                try:
                    await self._websocket_manager.broadcast_tracked_markets()
                except Exception as e:
                    logger.warning(f"Failed to broadcast tracked markets update: {e}")

        except Exception as e:
            self._errors += 1
            logger.error(f"DeepAgentDiscoveryService: discovery failed: {e}")

        return added

    @staticmethod
    def _parse_close_ts(close_time) -> Optional[int]:
        """Parse close_time to Unix timestamp."""
        if not close_time:
            return None
        try:
            if isinstance(close_time, str):
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                return int(close_dt.timestamp())
            return int(close_time)
        except (ValueError, TypeError):
            return None

    def _build_tracked_market(self, market_info: Dict[str, Any]) -> Optional[TrackedMarket]:
        """Build a TrackedMarket from API market data."""
        ticker = market_info.get("ticker", "")
        if not ticker:
            return None

        open_ts = self._parse_close_ts(market_info.get("open_time")) or 0
        close_ts = self._parse_close_ts(market_info.get("close_time")) or 0

        return TrackedMarket(
            ticker=ticker,
            event_ticker=market_info.get("event_ticker", ""),
            title=market_info.get("title", ""),
            category=market_info.get("category", ""),
            event_title=market_info.get("event_title", ""),
            yes_sub_title=market_info.get("yes_sub_title", ""),
            no_sub_title=market_info.get("no_sub_title", ""),
            subtitle=market_info.get("subtitle", ""),
            rules_primary=market_info.get("rules_primary", ""),
            status=MarketStatus.ACTIVE,
            created_ts=open_ts,
            open_ts=open_ts,
            close_ts=close_ts,
            tracked_at=time.time(),
            market_info=market_info,
            discovery_source="deep_agent_discovery",
            volume=market_info.get("volume", 0),
            volume_24h=market_info.get("volume_24h", 0),
            open_interest=market_info.get("open_interest", 0),
            yes_bid=market_info.get("yes_bid", 0) or 0,
            yes_ask=market_info.get("yes_ask", 0) or 0,
            price=market_info.get("last_price", 0) or 0,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get discovery stats for monitoring."""
        return {
            "discovery_runs": self._discovery_runs,
            "events_seen": self._events_seen,
            "events_matched": self._events_matched,
            "markets_discovered": self._markets_discovered,
            "markets_added": self._markets_added,
            "errors": self._errors,
            "last_discovery": self._last_discovery,
            "categories": sorted(self._categories),
            "refresh_seconds": DISCOVERY_REFRESH_SECONDS,
        }
