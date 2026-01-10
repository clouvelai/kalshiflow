"""
Truth Social distilled signal store.

Stores:
- minimal post metadata (no raw content)
- distilled narrative signals derived from posts

This is intentionally in-memory with TTL, because Truth Social content can be sensitive
and we want to avoid retaining full post bodies in-process.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.services.truth_social_signal_store")


@dataclass(frozen=True)
class TruthPostMeta:
    """Minimal metadata for a Truth Social post (no raw content)."""

    post_id: str
    author_handle: str
    created_at: float  # unix timestamp
    source_url: str
    is_verified: bool
    engagement_score: float


@dataclass(frozen=True)
class DistilledTruthSignal:
    """
    A distilled "narrative / intent" signal extracted from a post.

    NOTE: This is *not* independently verified. It's a structured summary.
    """

    signal_id: str
    created_at: float
    author_handle: str
    is_verified: bool
    engagement_score: float
    claim: str
    claim_type: str  # intent/announcement/denial/rumor/quote
    entities: List[str] = field(default_factory=list)
    linked_roles: Optional[List[str]] = None
    confidence: float = 0.5
    reasoning_short: str = ""
    source_url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "created_at": self.created_at,
            "author_handle": self.author_handle,
            "is_verified": self.is_verified,
            "engagement_score": self.engagement_score,
            "claim": self.claim,
            "claim_type": self.claim_type,
            "entities": list(self.entities or []),
            "linked_roles": list(self.linked_roles or []) if self.linked_roles is not None else None,
            "confidence": float(self.confidence),
            "reasoning_short": self.reasoning_short,
            "source_url": self.source_url,
        }


class DistilledTruthSignalStore:
    """
    In-memory store with TTL.

    Designed for:
    - quick queries during event research
    - avoiding retention of raw post content
    """

    def __init__(self, *, ttl_seconds: Optional[float] = None):
        if ttl_seconds is None:
            ttl_seconds = float(os.getenv("TRUTHSOCIAL_SIGNAL_TTL_SECONDS", "86400"))  # 24h
        self._ttl_seconds = float(ttl_seconds)

        # post_id -> (meta, expires_at)
        self._posts: Dict[str, Tuple[TruthPostMeta, float]] = {}
        # signal_id -> (signal, expires_at)
        self._signals: Dict[str, Tuple[DistilledTruthSignal, float]] = {}

        # lightweight ingest stats
        self._last_ingest_at: Optional[float] = None
        self._last_ingest_posts_seen: int = 0
        self._last_ingest_signals_emitted: int = 0
        self._last_ingest_unique_authors: int = 0
        self._last_ingest_verified_count: int = 0

    def _purge_expired(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        if self._posts:
            expired_posts = [pid for pid, (_, exp) in self._posts.items() if exp <= now]
            for pid in expired_posts:
                self._posts.pop(pid, None)
        if self._signals:
            expired_signals = [sid for sid, (_, exp) in self._signals.items() if exp <= now]
            for sid in expired_signals:
                self._signals.pop(sid, None)

    def ingest(
        self,
        *,
        posts: Iterable[TruthPostMeta],
        signals: Iterable[DistilledTruthSignal],
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Ingest a batch of post metas and derived signals.

        Returns a small stats dict for observability.
        """
        now = now or time.time()
        self._purge_expired(now)
        expires_at = now + self._ttl_seconds

        posts_list = list(posts or [])
        signals_list = list(signals or [])

        for p in posts_list:
            if not p.post_id:
                continue
            self._posts[p.post_id] = (p, expires_at)

        for s in signals_list:
            if not s.signal_id:
                continue
            self._signals[s.signal_id] = (s, expires_at)

        authors: Set[str] = {p.author_handle for p in posts_list if p.author_handle}
        verified_count = sum(1 for p in posts_list if p.is_verified)

        self._last_ingest_at = now
        self._last_ingest_posts_seen = len(posts_list)
        self._last_ingest_signals_emitted = len(signals_list)
        self._last_ingest_unique_authors = len(authors)
        self._last_ingest_verified_count = int(verified_count)

        return {
            "posts_seen": len(posts_list),
            "signals_emitted": len(signals_list),
            "unique_authors": len(authors),
            "verified_count": int(verified_count),
            "ingested_at": now,
            "ttl_seconds": self._ttl_seconds,
        }

    def get_last_ingest_stats(self) -> Dict[str, Any]:
        return {
            "posts_seen": self._last_ingest_posts_seen,
            "signals_emitted": self._last_ingest_signals_emitted,
            "unique_authors": self._last_ingest_unique_authors,
            "verified_count": self._last_ingest_verified_count,
            "ingested_at": self._last_ingest_at,
        }

    def query_signals(
        self,
        *,
        keywords: List[str],
        window_hours: float,
        limit: int = 10,
    ) -> List[DistilledTruthSignal]:
        """
        Query signals by simple keyword match over (claim + entities).
        """
        now = time.time()
        self._purge_expired(now)
        if not keywords:
            return []
        kw = [k.strip().lower() for k in keywords if k and k.strip()]
        if not kw:
            return []

        cutoff = now - float(window_hours) * 3600.0

        candidates: List[DistilledTruthSignal] = []
        for signal, _exp in self._signals.values():
            if signal.created_at and signal.created_at < cutoff:
                continue
            hay = " ".join([signal.claim] + (signal.entities or [])).lower()
            if any(k in hay for k in kw):
                candidates.append(signal)

        # Rank: confidence-weighted engagement, then recency
        candidates.sort(
            key=lambda s: (
                float(s.confidence or 0.0) * float(s.engagement_score or 0.0),
                float(s.created_at or 0.0),
            ),
            reverse=True,
        )
        return candidates[: max(0, int(limit))]

    def get_window_post_stats(self, *, window_hours: float) -> Dict[str, Any]:
        """
        Aggregate post stats over the current in-memory window (TTL-limited).
        """
        now = time.time()
        self._purge_expired(now)
        cutoff = now - float(window_hours) * 3600.0

        posts: List[TruthPostMeta] = []
        for p, _exp in self._posts.values():
            if p.created_at and p.created_at >= cutoff:
                posts.append(p)

        authors: Set[str] = {p.author_handle for p in posts if p.author_handle}
        verified_count = sum(1 for p in posts if p.is_verified)

        return {
            "posts_seen": len(posts),
            "unique_authors": len(authors),
            "verified_count": int(verified_count),
            "window_hours": float(window_hours),
        }


# Global singleton instance (mirrors cache singleton pattern)
_global_signal_store: Optional[DistilledTruthSignalStore] = None


def get_truth_social_signal_store() -> DistilledTruthSignalStore:
    global _global_signal_store
    if _global_signal_store is None:
        _global_signal_store = DistilledTruthSignalStore()
        logger.info("Initialized global DistilledTruthSignalStore")
    return _global_signal_store

