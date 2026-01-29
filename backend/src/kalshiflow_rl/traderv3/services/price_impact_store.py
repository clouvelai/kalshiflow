"""
Price Impact Signal Store - In-memory cache with TTL.

Provides fast in-memory access to price impact signals from the entity pipeline.
Follows the global singleton pattern used by EntityMarketIndex and DistilledTruthSignalStore.

The store is populated by PriceImpactAgent when signals are created, enabling
the DeepAgent to query signals without hitting Supabase every cycle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.services.price_impact_store")


@dataclass(frozen=True)
class CachedPriceImpact:
    """Immutable price impact signal for caching."""

    signal_id: str
    market_ticker: str
    entity_id: str
    entity_name: str
    sentiment_score: int  # Mapped from 5-point scale: -75, -40, 0, 40, 75
    price_impact_score: int  # Transformed for market type: -75, -40, 0, 40, 75
    confidence: float  # From LLM: 0.5, 0.7, 0.9
    market_type: str  # OUT, WIN, CONFIRM, NOMINEE
    event_ticker: str
    transformation_logic: str  # Explains the sentiment→impact transformation
    source_subreddit: str
    created_at: float  # Unix timestamp
    source_title: str = ""  # Reddit post title for context
    context_snippet: str = ""  # Text around entity mention
    source_created_at: Optional[float] = None  # Original Reddit post creation time
    source_type: str = "reddit_text"  # Source type: reddit_text, video_transcript, article_extract
    content_type: str = "text"  # Content type: text, video, link, image, social
    source_domain: str = "reddit.com"  # Source domain: youtube.com, foxnews.com, reddit.com
    agent_status: str = "pending"  # Agent status: pending, viewed, traded, observed, rejected


class PriceImpactStore:
    """
    In-memory store for price impact signals with TTL-based expiration.

    Signals are ingested by PriceImpactAgent and queried by DeepAgentTools.
    Uses a simple dict with timestamp tracking for expiration.
    """

    def __init__(self, ttl_seconds: float = 7200.0):  # 2 hour TTL
        """
        Initialize the store.

        Args:
            ttl_seconds: Time-to-live for cached signals (default: 2 hours)
        """
        self._ttl_seconds = ttl_seconds
        # signal_id -> (CachedPriceImpact, ingested_at_timestamp)
        self._signals: Dict[str, Tuple[CachedPriceImpact, float]] = {}

        # Stats
        self._total_ingested = 0
        self._total_pruned = 0

    def ingest(self, signal_data: dict) -> None:
        """
        Add a signal to the cache with current timestamp.

        Args:
            signal_data: Dict with all CachedPriceImpact fields
        """
        try:
            # Deduplicate: Skip if signal already exists
            signal_id = signal_data.get("signal_id", "")
            if signal_id and signal_id in self._signals:
                logger.debug(f"[price_impact_store] Duplicate signal skipped: {signal_id}")
                return
            # Parse source_created_at if present
            source_created_at = signal_data.get("source_created_at")
            if source_created_at is not None:
                source_created_at = float(source_created_at)

            signal = CachedPriceImpact(
                signal_id=signal_data["signal_id"],
                market_ticker=signal_data["market_ticker"],
                entity_id=signal_data["entity_id"],
                entity_name=signal_data["entity_name"],
                sentiment_score=int(signal_data["sentiment_score"]),
                price_impact_score=int(signal_data["price_impact_score"]),
                confidence=float(signal_data["confidence"]),
                market_type=signal_data["market_type"],
                event_ticker=signal_data["event_ticker"],
                transformation_logic=signal_data["transformation_logic"],
                source_subreddit=signal_data["source_subreddit"],
                created_at=float(signal_data["created_at"]),
                source_title=signal_data.get("source_title", ""),
                context_snippet=signal_data.get("context_snippet", ""),
                source_created_at=source_created_at,
                source_type=signal_data.get("source_type", "reddit_text"),
                content_type=signal_data.get("content_type", "text"),
                source_domain=signal_data.get("source_domain", "reddit.com"),
                agent_status=signal_data.get("agent_status", "pending"),
            )

            self._signals[signal.signal_id] = (signal, time.time())
            self._total_ingested += 1

            # Prune expired signals periodically
            if self._total_ingested % 10 == 0:
                self._prune_expired()

            logger.debug(
                f"[price_impact_store] Ingested signal {signal.signal_id} "
                f"for {signal.market_ticker}"
            )

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"[price_impact_store] Failed to ingest signal: {e}")

    async def get_impacts_for_trading(
        self,
        min_confidence: float = 0.5,
        min_impact_magnitude: int = 30,
        limit: int = 20,
        max_age_hours: float = 2.0,
    ) -> List[CachedPriceImpact]:
        """
        Query signals matching trading criteria.

        Args:
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            min_impact_magnitude: Minimum |price_impact_score| to include
            limit: Maximum number of signals to return
            max_age_hours: Maximum signal age in hours

        Returns:
            List of CachedPriceImpact objects sorted by created_at DESC
        """
        cutoff = time.time() - (max_age_hours * 3600)
        results = []

        for signal, added_at in self._signals.values():
            # Check signal age (based on signal's created_at, not cache time)
            if signal.created_at < cutoff:
                continue

            # Check confidence
            if signal.confidence < min_confidence:
                continue

            # Check impact magnitude
            if abs(signal.price_impact_score) < min_impact_magnitude:
                continue

            results.append(signal)

        # Sort by recency (most recent first)
        results.sort(key=lambda s: s.created_at, reverse=True)

        logger.info(
            f"[price_impact_store] Query returned {len(results[:limit])} signals "
            f"(min_conf={min_confidence}, min_impact={min_impact_magnitude})"
        )

        return results[:limit]

    def get_by_market(self, market_ticker: str) -> List[CachedPriceImpact]:
        """
        Get all signals for a specific market.

        Args:
            market_ticker: The market ticker to filter by

        Returns:
            List of signals for the market, sorted by recency
        """
        results = [
            signal
            for signal, _ in self._signals.values()
            if signal.market_ticker == market_ticker
        ]
        results.sort(key=lambda s: s.created_at, reverse=True)
        return results

    def get_by_entity(self, entity_id: str) -> List[CachedPriceImpact]:
        """
        Get all signals for a specific entity.

        Args:
            entity_id: The entity ID to filter by

        Returns:
            List of signals for the entity, sorted by recency
        """
        results = [
            signal
            for signal, _ in self._signals.values()
            if signal.entity_id == entity_id
        ]
        results.sort(key=lambda s: s.created_at, reverse=True)
        return results

    def _prune_expired(self) -> None:
        """Remove signals that have exceeded their TTL."""
        cutoff = time.time() - self._ttl_seconds
        original_count = len(self._signals)

        self._signals = {
            k: v for k, v in self._signals.items() if v[1] > cutoff
        }

        pruned = original_count - len(self._signals)
        if pruned > 0:
            self._total_pruned += pruned
            logger.info(f"[price_impact_store] Pruned {pruned} expired signals")

    async def load_from_supabase(self, supabase_client, max_age_hours: float = 2.0) -> int:
        """
        Load recent signals from Supabase on startup.

        This method populates the in-memory store with recent signals from
        the market_price_impacts table, preventing the store from being
        empty after a restart.

        Args:
            supabase_client: Async Supabase client instance
            max_age_hours: Maximum age of signals to load (default: 2 hours)

        Returns:
            Number of signals loaded
        """
        if not supabase_client:
            logger.warning("[price_impact_store] No Supabase client for startup sync")
            return 0

        try:
            from datetime import datetime, timedelta, timezone

            # Calculate cutoff time
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            cutoff_iso = cutoff.isoformat()

            # Query recent signals from Supabase
            result = await supabase_client.table("market_price_impacts") \
                .select("*") \
                .gte("created_at", cutoff_iso) \
                .order("created_at", desc=True) \
                .limit(500) \
                .execute()

            if not result.data:
                logger.info("[price_impact_store] No recent signals in Supabase")
                return 0

            loaded = 0
            for row in result.data:
                try:
                    # Convert Supabase row to signal_data format
                    # Generate signal_id from source_post_id + entity_id + market_ticker
                    signal_id = f"{row.get('source_post_id', '')}_{row.get('entity_id', '')}_{row.get('market_ticker', '')}"

                    # Parse created_at timestamp
                    created_at_str = row.get("created_at", "")
                    if created_at_str:
                        # Parse ISO format timestamp
                        created_at_dt = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        created_at = created_at_dt.timestamp()
                    else:
                        created_at = time.time()

                    # Parse source_created_at if present
                    source_created_at = None
                    source_created_at_str = row.get("source_created_at", "")
                    if source_created_at_str:
                        try:
                            source_created_at_dt = datetime.fromisoformat(source_created_at_str.replace("Z", "+00:00"))
                            source_created_at = source_created_at_dt.timestamp()
                        except (ValueError, TypeError):
                            pass

                    signal_data = {
                        "signal_id": signal_id,
                        "market_ticker": row.get("market_ticker", ""),
                        "entity_id": row.get("entity_id", ""),
                        "entity_name": row.get("entity_name", ""),
                        "sentiment_score": int(row.get("sentiment_score", 0)),
                        "price_impact_score": int(row.get("price_impact_score", 0)),
                        "confidence": float(row.get("confidence", 0.5)),
                        "market_type": row.get("market_type", ""),
                        "event_ticker": row.get("event_ticker", ""),
                        "transformation_logic": row.get("transformation_logic", ""),
                        "source_subreddit": row.get("source_subreddit", ""),
                        "created_at": created_at,
                        "source_title": row.get("source_title", ""),
                        "context_snippet": row.get("context_snippet", ""),
                        "source_created_at": source_created_at,
                        "source_type": row.get("source_type", "reddit_text"),
                        "content_type": row.get("content_type", "text"),
                        "source_domain": row.get("source_domain", "reddit.com"),
                        "agent_status": row.get("agent_status", "pending"),
                    }

                    # Ingest into store (skip if already exists)
                    if signal_id not in self._signals:
                        self.ingest(signal_data)
                        loaded += 1

                except (KeyError, ValueError, TypeError) as e:
                    logger.debug(f"[price_impact_store] Skipping malformed row: {e}")
                    continue

            logger.info(f"[price_impact_store] Loaded {loaded} signals from Supabase")
            return loaded

        except Exception as e:
            logger.error(f"[price_impact_store] Error loading from Supabase: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get store statistics."""
        return {
            "signal_count": len(self._signals),
            "total_ingested": self._total_ingested,
            "total_pruned": self._total_pruned,
            "ttl_seconds": self._ttl_seconds,
        }

    def get_recent_for_snapshot(self, limit: int = 30) -> List[dict]:
        """
        Get recent signals for client snapshot.

        Returns formatted signal data suitable for WebSocket snapshot.

        Args:
            limit: Maximum signals to return

        Returns:
            List of signal dicts sorted by recency
        """
        # Get all signals, sorted by created_at descending
        signals = [signal for signal, _ in self._signals.values()]
        signals.sort(key=lambda s: s.created_at, reverse=True)

        # Convert to dicts for JSON serialization
        result = []
        for signal in signals[:limit]:
            result.append({
                "signal_id": signal.signal_id,
                "market_ticker": signal.market_ticker,
                "entity_id": signal.entity_id,
                "entity_name": signal.entity_name,
                "sentiment_score": signal.sentiment_score,
                "price_impact_score": signal.price_impact_score,
                "confidence": signal.confidence,
                "market_type": signal.market_type,
                "event_ticker": signal.event_ticker,
                "transformation_logic": signal.transformation_logic,
                "source_subreddit": signal.source_subreddit,
                "source_title": signal.source_title,
                "context_snippet": signal.context_snippet,
                "created_at": signal.created_at,
                "source_created_at": signal.source_created_at,
                "source_type": signal.source_type,
                "content_type": signal.content_type,
                "source_domain": signal.source_domain,
                "agent_status": signal.agent_status,
            })

        return result

    def update_agent_status(self, signal_id: str, status: str) -> bool:
        """
        Update the agent status for a signal.

        Since CachedPriceImpact is frozen, we create a new instance
        with the updated status.

        Args:
            signal_id: The signal ID to update
            status: New status (pending, viewed, traded, observed, rejected)

        Returns:
            True if updated, False if signal not found
        """
        if signal_id not in self._signals:
            return False

        old_signal, added_at = self._signals[signal_id]

        # Create new signal with updated status
        new_signal = CachedPriceImpact(
            signal_id=old_signal.signal_id,
            market_ticker=old_signal.market_ticker,
            entity_id=old_signal.entity_id,
            entity_name=old_signal.entity_name,
            sentiment_score=old_signal.sentiment_score,
            price_impact_score=old_signal.price_impact_score,
            confidence=old_signal.confidence,
            market_type=old_signal.market_type,
            event_ticker=old_signal.event_ticker,
            transformation_logic=old_signal.transformation_logic,
            source_subreddit=old_signal.source_subreddit,
            created_at=old_signal.created_at,
            source_title=old_signal.source_title,
            context_snippet=old_signal.context_snippet,
            source_created_at=old_signal.source_created_at,
            source_type=old_signal.source_type,
            content_type=old_signal.content_type,
            source_domain=old_signal.source_domain,
            agent_status=status,
        )

        self._signals[signal_id] = (new_signal, added_at)
        logger.debug(f"[price_impact_store] Updated {signal_id} status to {status}")
        return True

    def clear(self) -> None:
        """Clear all signals from the store."""
        self._signals.clear()
        logger.info("[price_impact_store] Cleared all signals")


# Global singleton instance
_store: Optional[PriceImpactStore] = None


def get_price_impact_store() -> PriceImpactStore:
    """
    Get the global PriceImpactStore singleton.

    The store is lazily initialized on first access.
    """
    global _store
    if _store is None:
        _store = PriceImpactStore()
        logger.info("[price_impact_store] Initialized global PriceImpactStore")
    return _store


def set_price_impact_store(store: PriceImpactStore) -> None:
    """
    Set a custom PriceImpactStore as the global singleton.

    Useful for testing or custom configurations.
    """
    global _store
    _store = store
    logger.info("[price_impact_store] Set custom PriceImpactStore")
