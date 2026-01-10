"""
Truth Social Evidence Tool - Converts cached Truth Social posts into Evidence objects.

This module provides the adapter between TruthSocialCacheService and the event-first
research pipeline's Evidence format.
"""

import logging
import time
from typing import List, Optional, Dict, Any

from ..state.event_research_context import Evidence, EvidenceReliability
from .truth_social_cache import TruthSocialCacheService
from .truth_social_signal_router import TruthSocialSignalRouter, get_truth_social_signal_router
from .truth_social_signal_store import DistilledTruthSignal

logger = logging.getLogger("kalshiflow_rl.traderv3.services.truth_social_evidence")


class TruthSocialEvidenceTool:
    """
    Tool that queries TruthSocialCacheService and converts results to Evidence.

    This adapter maintains compatibility with the existing Evidence dataclass
    while enriching it with Truth Social-specific metadata (engagement metrics, etc.).
    """

    def __init__(
        self,
        cache_service: Optional[TruthSocialCacheService] = None,
        router: Optional[TruthSocialSignalRouter] = None,
    ):
        """
        Initialize Truth Social evidence tool.

        Args:
            cache_service: TruthSocialCacheService instance (if None, uses global singleton)
        """
        self._cache = cache_service
        self._router = router or get_truth_social_signal_router()

    def gather(
        self,
        *,
        event_title: str,
        primary_driver: str,
        queries: List[str],
        hours_back: Optional[float] = None,
        max_items: int = 25,
        context: Optional[Dict[str, Any]] = None,
    ) -> Evidence:
        """
        Gather Truth Social evidence matching the query criteria.

        Args:
            event_title: Event title for context
            primary_driver: Primary driver keyword
            queries: List of search queries/keywords
            hours_back: Optional time window (uses cache default if None)
            max_items: Maximum posts to return
            context: Optional additional context (unused for now)

        Returns:
            Evidence object with Truth Social posts converted to key_evidence and sources
        """
        if not self._cache or not self._cache.is_available():
            return Evidence(
                evidence_summary="Truth Social evidence unavailable (cache not running or following discovery failed)",
                reliability=EvidenceReliability.LOW,
                metadata={"truth_social": {"status": "unavailable"}},
            )

        # Combine queries with event context
        search_keywords = list(queries)
        if primary_driver:
            search_keywords.append(primary_driver)
        # Add event title words (split and filter short ones)
        if event_title:
            title_words = [w.strip() for w in event_title.split() if len(w.strip()) > 3]
            search_keywords.extend(title_words[:3])  # Top 3 words from title

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in search_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)

        window_hours = float(hours_back or self._cache.hours_back)
        routed = self._router.route(
            event_title=event_title,
            primary_driver=primary_driver or "",
            semantic_frame=(context or {}).get("semantic_frame"),
            window_hours=window_hours,
            limit=min(10, int(max_items)),
            extra_keywords=unique_keywords,
        )

        top_signals: List[DistilledTruthSignal] = routed.top_signals or []
        if not top_signals:
            router_stats = routed.stats or {}
            posts_seen = int(router_stats.get("posts_seen", 0))
            unique_authors = int(router_stats.get("unique_authors", 0))
            verified_count = int(router_stats.get("verified_count", 0))
            return Evidence(
                evidence_summary=f"No Truth Social narrative signals found (keywords: {', '.join(unique_keywords[:5])})",
                reliability=EvidenceReliability.LOW,
                metadata={
                    "truth_social": {
                        "status": "no_matches",
                        "posts_seen": posts_seen,
                        "signals_emitted": 0,
                        "unique_authors": unique_authors,
                        "verified_count": verified_count,
                        "window_hours": float(router_stats.get("window_hours", window_hours)),
                        "gathered_at": time.time(),
                        "top_signals": [],
                    }
                },
            )

        # Convert signals to evidence
        key_evidence: List[str] = []
        sources: List[str] = []

        # Conservative: treat Truth Social as narrative signal, not verified fact
        verified_count = sum(1 for s in top_signals if s.is_verified)
        avg_conf = sum(float(s.confidence or 0.0) for s in top_signals) / max(1, len(top_signals))

        for s in top_signals[: max_items]:
            claim = (s.claim or "").strip()
            if not claim:
                continue
            line = f"{claim} (@{s.author_handle}, {s.claim_type}, conf={int((s.confidence or 0.0)*100)}%)"
            key_evidence.append(line)
            if s.source_url:
                sources.append(s.source_url)

        # Reliability is capped at MEDIUM because these are narrative/intent signals.
        if verified_count >= 1 and len(top_signals) >= 2 and avg_conf >= 0.6:
            reliability = EvidenceReliability.MEDIUM
        else:
            reliability = EvidenceReliability.LOW

        reliability_reasoning = (
            f"Distilled {len(top_signals)} narrative signals "
            f"(verified_authors={verified_count}, avg_confidence={avg_conf:.2f}). "
            "Treat as intent/narrative, not independently verified fact."
        )

        # Build evidence summary
        evidence_summary = (
            f"Found {len(top_signals)} distilled Truth Social narrative signals "
            f"(keywords: {', '.join(unique_keywords[:5])})."
        )

        router_stats = routed.stats or {}
        window_stats = {
            "posts_seen": int(router_stats.get("posts_seen", 0)),
            "unique_authors": int(router_stats.get("unique_authors", 0)),
            "verified_count": int(router_stats.get("verified_count", verified_count)),
            "window_hours": float(router_stats.get("window_hours", window_hours)),
        }

        return Evidence(
            key_evidence=key_evidence,
            evidence_summary=evidence_summary,
            sources=sources,
            sources_checked=len(sources),
            reliability=reliability,
            reliability_reasoning=reliability_reasoning,
            metadata={
                "truth_social": {
                    "status": "success",
                    "posts_seen": window_stats["posts_seen"],
                    "signals_emitted": len(top_signals),
                    "unique_authors": window_stats["unique_authors"],
                    "verified_count": window_stats["verified_count"],
                    "window_hours": window_stats["window_hours"],
                    "top_signals": [s.to_dict() for s in top_signals[:10]],
                    "gathered_at": time.time(),
                }
            },
        )
