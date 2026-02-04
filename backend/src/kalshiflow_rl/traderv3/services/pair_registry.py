"""
Pair registry for cross-venue market matching.

Maps Kalshi tickers to Polymarket token IDs for arbitrage spread monitoring.
Loads from Supabase paired_markets table on startup, supports runtime updates.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.services.pair_registry")


@dataclass
class MarketPair:
    """A matched pair of markets across Kalshi and Polymarket."""
    id: str  # UUID from Supabase
    kalshi_ticker: str
    kalshi_event_ticker: Optional[str] = None
    poly_condition_id: str = ""
    poly_token_id_yes: str = ""
    poly_token_id_no: Optional[str] = None
    question: str = ""
    match_method: str = "manual"
    match_confidence: float = 1.0
    threshold_override_cents: Optional[int] = None
    status: str = "active"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class EventGroup:
    """A group of paired markets belonging to the same Kalshi event."""
    kalshi_event_ticker: str
    title: str = ""
    category: str = ""
    volume_24h: int = 0  # Combined 24h volume (cents)
    market_count: int = 0
    pairs: List[MarketPair] = field(default_factory=list)
    is_tradeable: bool = True  # Trading whitelist flag


class PairRegistry:
    """
    In-memory registry of cross-venue market pairs.

    Provides fast lookups by Kalshi ticker or Polymarket token ID.
    Loaded from Supabase on startup; updated when agent discovers new pairs.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._pairs: Dict[str, MarketPair] = {}  # pair_id -> MarketPair
        self._by_kalshi: Dict[str, MarketPair] = {}  # kalshi_ticker -> MarketPair
        self._by_poly: Dict[str, MarketPair] = {}  # poly_token_id_yes -> MarketPair
        self._by_event: Dict[str, EventGroup] = {}  # kalshi_event_ticker -> EventGroup
        self._last_sync: Optional[float] = None

    async def add_pair(self, pair: MarketPair) -> None:
        """Add or update a pair in the registry (async-safe)."""
        async with self._lock:
            self._pairs[pair.id] = pair
            self._by_kalshi[pair.kalshi_ticker] = pair
            if pair.poly_token_id_yes:
                self._by_poly[pair.poly_token_id_yes] = pair
            if pair.poly_token_id_no:
                self._by_poly[pair.poly_token_id_no] = pair

            # Index into event groups
            if pair.kalshi_event_ticker:
                event_ticker = pair.kalshi_event_ticker
                if event_ticker not in self._by_event:
                    self._by_event[event_ticker] = EventGroup(
                        kalshi_event_ticker=event_ticker,
                    )
                group = self._by_event[event_ticker]
                # Avoid duplicates
                existing_ids = {p.id for p in group.pairs}
                if pair.id not in existing_ids:
                    group.pairs.append(pair)
                    group.market_count = len(group.pairs)

        token_preview = pair.poly_token_id_yes[:12] if pair.poly_token_id_yes else "none"
        logger.info(f"Registered pair: {pair.kalshi_ticker} <-> poly:{token_preview}... [{pair.match_method}]")

    def remove_pair(self, pair_id: str) -> None:
        """Remove a pair from the registry (thread-safe)."""
        pair = self._pairs.pop(pair_id, None)
        if pair:
            self._by_kalshi.pop(pair.kalshi_ticker, None)
            if pair.poly_token_id_yes:
                self._by_poly.pop(pair.poly_token_id_yes, None)
            if pair.poly_token_id_no:
                self._by_poly.pop(pair.poly_token_id_no, None)
            logger.info(f"Removed pair: {pair.kalshi_ticker}")

    def get_by_id(self, pair_id: str) -> Optional[MarketPair]:
        """Look up pair by UUID."""
        return self._pairs.get(pair_id)

    def get_by_kalshi(self, kalshi_ticker: str) -> Optional[MarketPair]:
        """Look up pair by Kalshi ticker."""
        return self._by_kalshi.get(kalshi_ticker)

    def get_by_poly(self, poly_token_id: str) -> Optional[MarketPair]:
        """Look up pair by Polymarket token ID."""
        return self._by_poly.get(poly_token_id)

    def get_all_active(self) -> List[MarketPair]:
        """Get all active pairs."""
        return [p for p in self._pairs.values() if p.status == "active"]

    def get_poly_token_ids(self) -> List[str]:
        """Get all Polymarket YES token IDs for active pairs."""
        return [p.poly_token_id_yes for p in self._pairs.values()
                if p.status == "active" and p.poly_token_id_yes]

    @property
    def count(self) -> int:
        """Number of registered pairs."""
        return len(self._pairs)

    def get_events_grouped(self) -> List[EventGroup]:
        """Get all event groups sorted by volume desc."""
        groups = list(self._by_event.values())
        groups.sort(key=lambda g: g.volume_24h, reverse=True)
        return groups

    def get_event(self, event_ticker: str) -> Optional[EventGroup]:
        """Look up a single event group."""
        return self._by_event.get(event_ticker)

    def update_event_metadata(
        self, event_ticker: str, title: str = "", category: str = "",
        volume_24h: int = 0, is_tradeable: bool = True,
    ) -> None:
        """Update metadata on an event group (creates if not exists)."""
        group = self._by_event.get(event_ticker)
        if not group:
            group = EventGroup(kalshi_event_ticker=event_ticker)
            self._by_event[event_ticker] = group
        if title:
            group.title = title
        if category:
            group.category = category
        group.volume_24h = volume_24h
        group.is_tradeable = is_tradeable

    async def load_from_supabase(self, supabase_client: Any) -> int:
        """
        Load pairs from Supabase paired_markets table.

        Args:
            supabase_client: Supabase client instance

        Returns:
            Number of pairs loaded
        """
        try:
            result = supabase_client.table("paired_markets").select("*").eq("status", "active").execute()
            rows = result.data or []

            for row in rows:
                pair = MarketPair(
                    id=row["id"],
                    kalshi_ticker=row["kalshi_ticker"],
                    kalshi_event_ticker=row.get("kalshi_event_ticker"),
                    poly_condition_id=row.get("poly_condition_id", ""),
                    poly_token_id_yes=row.get("poly_token_id_yes", ""),
                    poly_token_id_no=row.get("poly_token_id_no"),
                    question=row.get("question", ""),
                    match_method=row.get("match_method", "manual"),
                    match_confidence=row.get("match_confidence", 1.0),
                    threshold_override_cents=row.get("threshold_override_cents"),
                    status=row.get("status", "active"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at"),
                )
                await self.add_pair(pair)

            self._last_sync = time.time()
            logger.info(f"Loaded {len(rows)} active pairs from Supabase")
            return len(rows)

        except Exception as e:
            logger.error(f"Failed to load pairs from Supabase: {e}")
            return 0

    def get_status(self) -> Dict[str, Any]:
        """Get registry status for health/status endpoints."""
        return {
            "total_pairs": len(self._pairs),
            "active_pairs": len(self.get_all_active()),
            "poly_tokens_tracked": len(self.get_poly_token_ids()),
            "last_sync": self._last_sync,
        }
