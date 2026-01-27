"""
Related Entity Store for Second-Hand Signal Analysis.

Stores non-market entities (PERSON, ORG, GPE, EVENT) discovered via
general NER for second-hand signal analysis by the DeepAgent.

These entities are not directly linked to Kalshi markets but may
provide contextual signals. For example:
- "Putin meets Xi" → geopolitical context for Taiwan markets
- "Fed Chair Powell speaks" → economic context for various markets

The store persists to Supabase and supports queries for:
- Entity type filtering (PERSON, ORG, GPE, EVENT)
- Co-occurrence with market entities
- Sentiment magnitude filtering
- Temporal queries
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from supabase import AsyncClient

logger = logging.getLogger("kalshiflow_rl.traderv3.nlp.entity_store")


@dataclass
class RelatedEntity:
    """
    A non-market entity discovered via general NER.

    These entities are not linked to Kalshi markets but provide
    contextual information for second-hand signal analysis.
    """

    # Entity identification
    entity_text: str  # Original text: "Putin"
    entity_type: str  # PERSON, ORG, GPE, EVENT
    normalized_id: str  # Normalized: "vladimir_putin"

    # Sentiment
    sentiment_score: int  # -100 to +100
    confidence: float = 1.0

    # Source context
    source_post_id: str = ""
    source_subreddit: str = ""
    context_snippet: str = ""

    # Co-occurrence with market entities
    co_occurring_market_entities: List[str] = field(default_factory=list)

    # Metadata
    created_at: float = field(default_factory=time.time)
    id: Optional[str] = None  # Database ID (set after insert)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database insertion."""
        return {
            "entity_text": self.entity_text,
            "entity_type": self.entity_type,
            "normalized_id": self.normalized_id,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "source_post_id": self.source_post_id,
            "source_subreddit": self.source_subreddit,
            "context_snippet": self.context_snippet,
            "co_occurring_market_entities": self.co_occurring_market_entities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelatedEntity":
        """Create from database row."""
        return cls(
            id=data.get("id"),
            entity_text=data.get("entity_text", ""),
            entity_type=data.get("entity_type", ""),
            normalized_id=data.get("normalized_id", ""),
            sentiment_score=data.get("sentiment_score", 0),
            confidence=data.get("confidence", 1.0),
            source_post_id=data.get("source_post_id", ""),
            source_subreddit=data.get("source_subreddit", ""),
            context_snippet=data.get("context_snippet", ""),
            co_occurring_market_entities=data.get("co_occurring_market_entities", []),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class RelatedEntityQuery:
    """Query parameters for related entity searches."""

    entity_type: Optional[str] = None  # Filter by type
    co_occurs_with: Optional[str] = None  # Must co-occur with this market entity
    min_sentiment_magnitude: int = 0  # Minimum |sentiment|
    subreddit: Optional[str] = None  # Filter by source subreddit
    limit: int = 20
    since_hours: Optional[int] = None  # Only entities from last N hours


class RelatedEntityStore:
    """
    Store for non-market entities discovered via NER.

    Provides:
    - Async insertion to Supabase related_entities table
    - Query support for DeepAgent context retrieval
    - In-memory cache for fast lookups
    """

    def __init__(
        self,
        supabase_client: Optional["AsyncClient"] = None,
        cache_size: int = 1000,
    ):
        """
        Initialize the store.

        Args:
            supabase_client: Async Supabase client for persistence
            cache_size: Maximum entities to cache in memory
        """
        self._supabase = supabase_client
        self._cache_size = cache_size

        # In-memory cache: normalized_id -> most recent RelatedEntity
        self._cache: Dict[str, RelatedEntity] = {}

        # Stats
        self._inserts = 0
        self._cache_hits = 0
        self._queries = 0

    async def insert(self, entity: RelatedEntity) -> Optional[str]:
        """
        Insert a related entity into the store.

        Args:
            entity: RelatedEntity to insert

        Returns:
            Database ID if successful, None otherwise
        """
        # Add to cache
        self._cache[entity.normalized_id] = entity
        self._prune_cache()

        # Insert to database
        if self._supabase:
            try:
                result = await self._supabase.table("related_entities").insert(
                    entity.to_dict()
                ).execute()

                if result.data:
                    self._inserts += 1
                    entity.id = result.data[0].get("id")
                    logger.debug(f"[entity_store] Inserted related entity: {entity.normalized_id}")
                    return entity.id

            except Exception as e:
                logger.warning(f"[entity_store] Insert failed: {e}")

        return None

    async def insert_batch(self, entities: List[RelatedEntity]) -> int:
        """
        Insert multiple entities in a batch.

        Args:
            entities: List of RelatedEntity objects

        Returns:
            Number of entities successfully inserted
        """
        if not entities:
            return 0

        # Add all to cache
        for entity in entities:
            self._cache[entity.normalized_id] = entity
        self._prune_cache()

        # Batch insert to database
        if self._supabase:
            try:
                data = [e.to_dict() for e in entities]
                result = await self._supabase.table("related_entities").insert(
                    data
                ).execute()

                if result.data:
                    count = len(result.data)
                    self._inserts += count
                    logger.info(f"[entity_store] Batch inserted {count} related entities")
                    return count

            except Exception as e:
                logger.warning(f"[entity_store] Batch insert failed: {e}")

        return 0

    async def query(self, params: RelatedEntityQuery) -> List[RelatedEntity]:
        """
        Query related entities with filters.

        Args:
            params: Query parameters

        Returns:
            List of matching RelatedEntity objects
        """
        self._queries += 1

        if not self._supabase:
            # Fallback to cache query
            return self._query_cache(params)

        try:
            query = self._supabase.table("related_entities").select("*")

            # Apply filters
            if params.entity_type:
                query = query.eq("entity_type", params.entity_type)

            if params.co_occurs_with:
                query = query.contains(
                    "co_occurring_market_entities",
                    [params.co_occurs_with]
                )

            if params.min_sentiment_magnitude > 0:
                # Filter by absolute sentiment value
                # Note: Supabase doesn't support abs() directly
                # We use OR: sentiment >= min OR sentiment <= -min
                query = query.or_(
                    f"sentiment_score.gte.{params.min_sentiment_magnitude},"
                    f"sentiment_score.lte.{-params.min_sentiment_magnitude}"
                )

            if params.subreddit:
                query = query.eq("source_subreddit", params.subreddit)

            if params.since_hours:
                cutoff = time.time() - (params.since_hours * 3600)
                query = query.gte("created_at", cutoff)

            query = query.order("created_at", desc=True).limit(params.limit)

            result = await query.execute()

            if result.data:
                return [RelatedEntity.from_dict(row) for row in result.data]

        except Exception as e:
            logger.warning(f"[entity_store] Query failed: {e}")

        return []

    def _query_cache(self, params: RelatedEntityQuery) -> List[RelatedEntity]:
        """Query in-memory cache (fallback when DB unavailable)."""
        results = []

        for entity in self._cache.values():
            # Apply filters
            if params.entity_type and entity.entity_type != params.entity_type:
                continue

            if params.co_occurs_with:
                if params.co_occurs_with not in entity.co_occurring_market_entities:
                    continue

            if abs(entity.sentiment_score) < params.min_sentiment_magnitude:
                continue

            if params.subreddit and entity.source_subreddit != params.subreddit:
                continue

            if params.since_hours:
                cutoff = time.time() - (params.since_hours * 3600)
                if entity.created_at < cutoff:
                    continue

            results.append(entity)
            self._cache_hits += 1

            if len(results) >= params.limit:
                break

        return results

    def _prune_cache(self) -> None:
        """Prune cache to max size by removing oldest entries."""
        if len(self._cache) <= self._cache_size:
            return

        # Sort by created_at and keep newest
        sorted_items = sorted(
            self._cache.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )

        self._cache = dict(sorted_items[:self._cache_size])

    def get_cached(self, normalized_id: str) -> Optional[RelatedEntity]:
        """
        Get entity from cache by normalized ID.

        Args:
            normalized_id: Normalized entity ID

        Returns:
            RelatedEntity if found, None otherwise
        """
        entity = self._cache.get(normalized_id)
        if entity:
            self._cache_hits += 1
        return entity

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return {
            "cached_entities": len(self._cache),
            "cache_size_limit": self._cache_size,
            "total_inserts": self._inserts,
            "total_queries": self._queries,
            "cache_hits": self._cache_hits,
            "has_db_connection": self._supabase is not None,
        }

    async def get_context_for_market(
        self,
        market_entity_id: str,
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[RelatedEntity]:
        """
        Get related entities that co-occur with a market entity.

        This is a convenience method for DeepAgent context retrieval.

        Args:
            market_entity_id: The market entity to find context for
            entity_types: Optional filter for entity types
            limit: Maximum results

        Returns:
            List of co-occurring related entities
        """
        results = []

        for entity_type in (entity_types or ["PERSON", "ORG", "GPE", "EVENT"]):
            query = RelatedEntityQuery(
                entity_type=entity_type,
                co_occurs_with=market_entity_id,
                limit=limit // len(entity_types) if entity_types else limit,
            )
            type_results = await self.query(query)
            results.extend(type_results)

        return results[:limit]


# =============================================================================
# Global Instance
# =============================================================================

_global_store: Optional[RelatedEntityStore] = None


def get_related_entity_store() -> Optional[RelatedEntityStore]:
    """Get the global RelatedEntityStore instance."""
    return _global_store


def set_related_entity_store(store: RelatedEntityStore) -> None:
    """Set the global RelatedEntityStore instance."""
    global _global_store
    _global_store = store
    logger.info("[entity_store] Set global RelatedEntityStore")


async def init_related_entity_store() -> RelatedEntityStore:
    """
    Initialize and return the global RelatedEntityStore.

    Creates a Supabase client and sets up the store.

    Returns:
        Initialized RelatedEntityStore
    """
    import os

    supabase_client = None

    try:
        from supabase import acreate_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

        if url and key:
            supabase_client = await acreate_client(url, key)
            logger.info("[entity_store] Connected to Supabase")

    except Exception as e:
        logger.warning(f"[entity_store] Supabase init failed: {e}")

    store = RelatedEntityStore(supabase_client=supabase_client)
    set_related_entity_store(store)
    return store
