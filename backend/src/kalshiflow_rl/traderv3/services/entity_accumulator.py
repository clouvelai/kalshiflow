"""
Entity Accumulator - Sliding-window entity mention accumulator.

Aggregates entity mentions over a configurable time window, computing
signal strength from mention frequency, sentiment, engagement metrics,
and source diversity. Syncs state to Supabase kb_entities table for
Realtime-powered frontend visualization.

Signal strength formula:
    signal_strength = (
        mention_factor * 0.35      # log2(mentions+1)/4, capped at 1.0
        + sentiment_factor * 0.30  # avg |sentiment| / 75
        + engagement_factor * 0.20 # log2(max_reddit_score+1) / 10
        + diversity_bonus * 0.15   # unique source types / 3
    )
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..schemas.kb_schemas import (
    EntityMention,
    EntityRelation,
    AccumulatedEntitySignal,
    EntitySignalSummary,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("kalshiflow_rl.traderv3.services.entity_accumulator")


@dataclass
class EntityAccumulatorConfig:
    """Configuration for the Entity Accumulator."""
    window_seconds: float = 7200.0  # 2-hour sliding window
    signal_threshold: float = 0.4  # Min strength to emit signal
    re_emit_threshold: float = 0.2  # Min strength increase to re-emit
    bypass_threshold_sentiment: int = 75  # Bypass accumulation for very strong signals
    bypass_threshold_confidence: float = 0.9


class EntityAccumulator:
    """Sliding-window accumulator for entity mentions.

    Aggregates mentions per entity, computes signal strength,
    and syncs state to Supabase kb_entities table.
    """

    def __init__(
        self,
        config: Optional[EntityAccumulatorConfig] = None,
        supabase_client=None,
    ):
        self._config = config or EntityAccumulatorConfig()
        self._supabase = supabase_client

        # entity_id -> list of EntityMention (within window)
        self._mentions: Dict[str, List[EntityMention]] = defaultdict(list)

        # entity_id -> last computed AccumulatedEntitySignal
        self._signals: Dict[str, AccumulatedEntitySignal] = {}

        # entity_id -> last emitted signal_strength (for re-emit gating)
        self._last_emitted_strength: Dict[str, float] = {}

        # entity_id -> canonical_name (persisted across mentions)
        self._entity_names: Dict[str, str] = {}

        # entity_id -> entity_category
        self._entity_categories: Dict[str, str] = {}

        # entity_id -> list of EntityRelation (within window, subject or object)
        self._relations: Dict[str, List[EntityRelation]] = defaultdict(list)

        # Stats
        self._total_mentions = 0
        self._total_relations = 0
        self._signals_emitted = 0
        self._bypasses = 0

    def set_supabase_client(self, client) -> None:
        """Set or update the Supabase client."""
        self._supabase = client

    async def add_mention(
        self,
        mention: EntityMention,
        canonical_name: str = "",
        entity_category: str = "person",
        linked_market_tickers: Optional[List[str]] = None,
    ) -> AccumulatedEntitySignal:
        """Add a mention, update accumulated state, sync to Supabase.

        Args:
            mention: The entity mention to add
            canonical_name: Display name for the entity
            entity_category: Category (person, organization, objective)
            linked_market_tickers: Market tickers linked to this entity

        Returns:
            Current accumulated signal for this entity
        """
        self._total_mentions += 1
        entity_id = mention.entity_id

        # Check for bypass: very strong signals skip normal accumulation
        bypassed = self.should_bypass(mention)
        if bypassed:
            self._bypasses += 1
            logger.info(
                f"[accumulator] Bypass triggered for {entity_id}: "
                f"sentiment={mention.sentiment_score}, confidence={mention.confidence}"
            )

        # Store name and category
        if canonical_name:
            self._entity_names[entity_id] = canonical_name
        if entity_category:
            self._entity_categories[entity_id] = entity_category

        # Prune expired mentions for this entity
        self._prune_window(entity_id)

        # Add new mention
        self._mentions[entity_id].append(mention)

        # Recompute accumulated signal
        signal = self._compute_signal(
            entity_id,
            linked_market_tickers=linked_market_tickers or [],
        )

        # If bypassed and signal_strength is below threshold, boost to threshold
        # so the signal is immediately emittable
        if bypassed and signal.signal_strength < self._config.signal_threshold:
            signal.signal_strength = self._config.signal_threshold

        self._signals[entity_id] = signal

        # Sync to Supabase (non-blocking, fire-and-forget on error)
        await self._sync_to_supabase(entity_id, signal)

        # Save mention to Supabase entity_mentions table
        await self._save_mention_to_supabase(mention)

        return signal

    def should_bypass(self, mention: EntityMention) -> bool:
        """Check if a mention should bypass accumulation (very strong signal).

        Returns True for mentions with |sentiment| >= bypass_threshold_sentiment
        AND confidence >= bypass_threshold_confidence.
        """
        return (
            abs(mention.sentiment_score) >= self._config.bypass_threshold_sentiment
            and mention.confidence >= self._config.bypass_threshold_confidence
        )

    def should_emit_signal(self, entity_id: str, signal: AccumulatedEntitySignal) -> bool:
        """Check if a signal should be emitted based on threshold and re-emit gate.

        Returns True if:
        - signal_strength >= signal_threshold AND
        - Either never emitted before OR strength increased by >= re_emit_threshold
        """
        if signal.signal_strength < self._config.signal_threshold:
            return False

        last_strength = self._last_emitted_strength.get(entity_id, 0.0)
        if last_strength == 0.0:
            return True  # First emission

        return (signal.signal_strength - last_strength) >= self._config.re_emit_threshold

    def record_emission(self, entity_id: str) -> None:
        """Record that a signal was emitted for re-emit gating."""
        signal = self._signals.get(entity_id)
        if signal:
            self._last_emitted_strength[entity_id] = signal.signal_strength
            self._signals_emitted += 1

    def get_signal(self, entity_id: str) -> Optional[AccumulatedEntitySignal]:
        """Current accumulated signal for entity."""
        self._prune_window(entity_id)
        return self._signals.get(entity_id)

    async def add_relation(self, relation: EntityRelation) -> None:
        """Add a relation and index it under both subject and object entity IDs.

        Also saves the relation to Supabase if available.
        """
        self._total_relations += 1
        self._relations[relation.subject_entity_id].append(relation)
        if relation.object_entity_id != relation.subject_entity_id:
            self._relations[relation.object_entity_id].append(relation)

        await self._save_relation_to_supabase(relation)

    def get_entity_relations(
        self,
        entity_id: str,
        limit: int = 10,
    ) -> List[EntityRelation]:
        """Get recent relations involving an entity (as subject or object).

        Prunes expired relations outside the sliding window.
        """
        cutoff = time.time() - self._config.window_seconds
        relations = self._relations.get(entity_id, [])
        # Prune expired
        active = [r for r in relations if r.created_at >= cutoff]
        self._relations[entity_id] = active
        # Sort by recency
        active.sort(key=lambda r: r.created_at, reverse=True)
        return active[:limit]

    async def _save_relation_to_supabase(self, relation: EntityRelation) -> None:
        """Save a relation to the entity_relations table."""
        if not self._supabase:
            return

        try:
            from datetime import datetime, timezone

            self._supabase.table("entity_relations").insert({
                "subject_entity_id": relation.subject_entity_id,
                "subject_name": relation.subject_name,
                "relation": relation.relation,
                "object_entity_id": relation.object_entity_id,
                "object_name": relation.object_name,
                "confidence": relation.confidence,
                "source_post_id": relation.source_post_id or None,
                "context_snippet": relation.context_snippet[:500] if relation.context_snippet else "",
                "created_at": datetime.fromtimestamp(
                    relation.created_at, tz=timezone.utc
                ).isoformat(),
            }).execute()

        except Exception as e:
            logger.debug(f"[accumulator] Relation save error: {e}")

    def get_all_signals(
        self,
        min_strength: float = 0.3,
        min_mentions: int = 2,
    ) -> List[AccumulatedEntitySignal]:
        """All entities with active accumulated signals above threshold."""
        # Prune all windows first
        for entity_id in list(self._mentions.keys()):
            self._prune_window(entity_id)

        results = []
        for entity_id, signal in self._signals.items():
            if (
                signal.signal_strength >= min_strength
                and signal.mention_count >= min_mentions
            ):
                results.append(signal)

        # Sort by signal_strength descending
        results.sort(key=lambda s: s.signal_strength, reverse=True)
        return results

    def get_entity_signal_summaries(
        self,
        min_strength: float = 0.3,
        min_mentions: int = 1,
        limit: int = 15,
    ) -> List[EntitySignalSummary]:
        """Get deep agent-facing summaries of accumulated signals."""
        signals = self.get_all_signals(min_strength=min_strength, min_mentions=min_mentions)

        summaries = []
        for signal in signals[:limit]:
            window_hours = self._config.window_seconds / 3600.0

            # Gather relations for this entity
            entity_relations = self.get_entity_relations(signal.entity_id, limit=5)
            relations_dicts = [
                {
                    "subject": r.subject_name,
                    "relation": r.relation,
                    "object": r.object_name,
                    "confidence": round(r.confidence, 2),
                }
                for r in entity_relations
            ]

            summaries.append(EntitySignalSummary(
                entity_id=signal.entity_id,
                canonical_name=signal.canonical_name,
                entity_category=signal.entity_category,
                mention_count=signal.mention_count,
                unique_sources=signal.unique_sources,
                weighted_sentiment=round(signal.weighted_sentiment, 1),
                signal_strength=round(signal.signal_strength, 3),
                max_reddit_score=signal.max_reddit_score,
                total_reddit_comments=signal.total_reddit_comments,
                linked_market_tickers=signal.linked_market_tickers,
                categories=signal.categories,
                latest_context=signal.latest_context,
                window_hours=window_hours,
                relations=relations_dicts,
            ))

        return summaries

    def _prune_window(self, entity_id: str) -> None:
        """Remove mentions outside the sliding window."""
        cutoff = time.time() - self._config.window_seconds
        mentions = self._mentions.get(entity_id, [])
        self._mentions[entity_id] = [
            m for m in mentions if m.created_at >= cutoff
        ]

        # If no mentions left, clean up the signal
        if not self._mentions[entity_id]:
            self._signals.pop(entity_id, None)
            self._mentions.pop(entity_id, None)

    def _compute_signal(
        self,
        entity_id: str,
        linked_market_tickers: Optional[List[str]] = None,
    ) -> AccumulatedEntitySignal:
        """Compute accumulated signal from current mentions."""
        mentions = self._mentions.get(entity_id, [])

        if not mentions:
            return AccumulatedEntitySignal(
                entity_id=entity_id,
                canonical_name=self._entity_names.get(entity_id, entity_id),
                entity_category=self._entity_categories.get(entity_id, "person"),
            )

        # Basic counts
        mention_count = len(mentions)
        source_ids = set()
        source_types = set()
        all_categories = set()
        max_reddit_score = 0
        total_reddit_comments = 0

        # Weighted sentiment (by Reddit score)
        total_weighted_sentiment = 0.0
        total_weight = 0.0

        for m in mentions:
            source_ids.add(m.source_post_id or str(m.created_at))
            source_types.add(m.source_type)
            all_categories.update(m.categories)

            if m.reddit_score > max_reddit_score:
                max_reddit_score = m.reddit_score
            total_reddit_comments += m.reddit_comments

            # Weight sentiment by Reddit engagement (min weight = 1)
            weight = max(1, m.reddit_score)
            total_weighted_sentiment += m.sentiment_score * weight
            total_weight += weight

        weighted_sentiment = total_weighted_sentiment / total_weight if total_weight > 0 else 0.0
        unique_sources = len(source_ids)

        # Signal strength formula
        mention_factor = min(1.0, math.log2(mention_count + 1) / 4.0)
        sentiment_factor = min(1.0, abs(weighted_sentiment) / 75.0)
        engagement_factor = min(1.0, math.log2(max_reddit_score + 1) / 10.0)
        diversity_bonus = min(1.0, len(source_types) / 3.0)

        signal_strength = (
            mention_factor * 0.35
            + sentiment_factor * 0.30
            + engagement_factor * 0.20
            + diversity_bonus * 0.15
        )

        # Get latest context
        latest_mention = max(mentions, key=lambda m: m.created_at)

        # Merge linked market tickers from all mentions + explicit param
        all_tickers = set(linked_market_tickers or [])

        return AccumulatedEntitySignal(
            entity_id=entity_id,
            canonical_name=self._entity_names.get(entity_id, entity_id),
            entity_category=self._entity_categories.get(entity_id, "person"),
            mention_count=mention_count,
            unique_sources=unique_sources,
            weighted_sentiment=round(weighted_sentiment, 1),
            signal_strength=round(signal_strength, 4),
            max_reddit_score=max_reddit_score,
            total_reddit_comments=total_reddit_comments,
            source_types=list(source_types),
            categories=list(all_categories),
            linked_market_tickers=list(all_tickers),
            latest_context=latest_mention.context_snippet,
            first_mention_at=min(m.created_at for m in mentions),
            last_mention_at=max(m.created_at for m in mentions),
        )

    async def _sync_to_supabase(
        self, entity_id: str, signal: AccumulatedEntitySignal
    ) -> None:
        """Upsert to kb_entities table (triggers Supabase Realtime)."""
        if not self._supabase:
            return

        try:
            from datetime import datetime, timezone

            def _ts_to_iso(ts: float) -> Optional[str]:
                if ts <= 0:
                    return None
                return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

            self._supabase.table("kb_entities").upsert({
                "entity_id": entity_id,
                "canonical_name": signal.canonical_name,
                "entity_category": signal.entity_category,
                "current_sentiment": signal.weighted_sentiment,
                "mention_count": signal.mention_count,
                "unique_sources": signal.unique_sources,
                "signal_strength": signal.signal_strength,
                "max_reddit_score": signal.max_reddit_score,
                "total_reddit_comments": signal.total_reddit_comments,
                "source_types": signal.source_types,
                "categories": signal.categories,
                "linked_market_tickers": signal.linked_market_tickers,
                "latest_context": signal.latest_context[:500],
                "first_mention_at": _ts_to_iso(signal.first_mention_at),
                "last_mention_at": _ts_to_iso(signal.last_mention_at),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, on_conflict="entity_id").execute()

        except Exception as e:
            logger.warning(f"[accumulator] KB sync error for {entity_id}: {e}")

    async def _save_mention_to_supabase(self, mention: EntityMention) -> None:
        """Save individual mention to entity_mentions table."""
        if not self._supabase:
            return

        try:
            from datetime import datetime, timezone

            self._supabase.table("entity_mentions").insert({
                "entity_id": mention.entity_id,
                "source_type": mention.source_type,
                "source_post_id": mention.source_post_id or None,
                "sentiment_score": mention.sentiment_score,
                "confidence": mention.confidence,
                "categories": mention.categories,
                "reddit_score": mention.reddit_score,
                "reddit_comments": mention.reddit_comments,
                "context_snippet": mention.context_snippet[:500] if mention.context_snippet else "",
                "created_at": datetime.fromtimestamp(
                    mention.created_at, tz=timezone.utc
                ).isoformat(),
            }).execute()

        except Exception as e:
            logger.debug(f"[accumulator] Mention save error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get accumulator statistics."""
        return {
            "total_mentions": self._total_mentions,
            "total_relations": self._total_relations,
            "active_entities": len(self._mentions),
            "active_relations": sum(len(v) for v in self._relations.values()),
            "signals_emitted": self._signals_emitted,
            "bypasses": self._bypasses,
            "window_seconds": self._config.window_seconds,
            "signal_threshold": self._config.signal_threshold,
        }


# Global singleton
_global_accumulator: Optional[EntityAccumulator] = None


def get_entity_accumulator() -> Optional[EntityAccumulator]:
    """Get the global EntityAccumulator instance."""
    return _global_accumulator


def set_entity_accumulator(accumulator: EntityAccumulator) -> None:
    """Set the global EntityAccumulator instance."""
    global _global_accumulator
    _global_accumulator = accumulator
    logger.info("[accumulator] Set global EntityAccumulator instance")
