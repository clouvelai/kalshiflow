"""
Truth Social signal router.

Given an event (title + driver + semantic frame), it:
- builds a keyword frame for querying distilled signals
- queries the global signal store
- annotates signals with linked roles (when semantic frame is available)
- reranks by relevance + confidence/engagement
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..state.event_research_context import SemanticFrame, SemanticRole
from .truth_social_signal_store import DistilledTruthSignal, DistilledTruthSignalStore, get_truth_social_signal_store

logger = logging.getLogger("kalshiflow_rl.traderv3.services.truth_social_signal_router")


def _title_keywords(title: str, *, max_words: int = 5) -> List[str]:
    if not title:
        return []
    words = [w.strip() for w in re.split(r"\s+", title) if len(w.strip()) > 3]
    return words[:max_words]


def _dedupe_preserve(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        k = x.strip()
        if not k:
            continue
        kk = k.lower()
        if kk in seen:
            continue
        seen.add(kk)
        out.append(k)
    return out


def _role_aliases(role: SemanticRole) -> List[str]:
    aliases = []
    if role.canonical_name:
        aliases.append(role.canonical_name)
    for a in (role.aliases or []):
        if a:
            aliases.append(a)
    return _dedupe_preserve(aliases)[:6]


def _link_roles_for_signal(signal: DistilledTruthSignal, frame: SemanticFrame) -> List[str]:
    """
    Attempt to link a signal to semantic roles via entity/alias matching.
    Returns list of role entity_ids.
    """
    if not frame:
        return []
    signal_text = " ".join([signal.claim] + (signal.entities or [])).lower()
    linked: List[str] = []

    def _maybe_add(role: SemanticRole) -> None:
        if not role or not role.entity_id:
            return
        for alias in _role_aliases(role):
            if alias and alias.lower() in signal_text:
                linked.append(role.entity_id)
                return

    for r in (frame.actors or []):
        _maybe_add(r)
    for r in (frame.objects or []):
        _maybe_add(r)
    for r in (frame.candidates or []):
        _maybe_add(r)

    return _dedupe_preserve(linked)[:8]


def _relevance_score(signal: DistilledTruthSignal, keywords: List[str], linked_roles: List[str]) -> float:
    """
    Cheap scoring:
    - keyword hits boost
    - linked_roles boost
    - confidence/engagement base
    - mild recency boost
    """
    hay = " ".join([signal.claim] + (signal.entities or [])).lower()
    hits = sum(1 for k in keywords if k.lower() in hay)
    base = float(signal.confidence or 0.0) * (1.0 + float(signal.engagement_score or 0.0) / 250.0)
    role_boost = 0.35 * min(3, len(linked_roles))
    hit_boost = 0.15 * min(6, hits)
    recency_boost = min(0.25, max(0.0, (time.time() - float(signal.created_at or 0.0)) / -86400.0))  # ~0..0.25
    return base + role_boost + hit_boost + recency_boost


@dataclass(frozen=True)
class RoutedTruthSignals:
    top_signals: List[DistilledTruthSignal]
    stats: Dict[str, Any]


class TruthSocialSignalRouter:
    def __init__(self, store: Optional[DistilledTruthSignalStore] = None):
        self._store = store or get_truth_social_signal_store()

    def route(
        self,
        *,
        event_title: str,
        primary_driver: str,
        semantic_frame: Optional[SemanticFrame],
        window_hours: float = 24.0,
        limit: int = 10,
        extra_keywords: Optional[List[str]] = None,
    ) -> RoutedTruthSignals:
        keywords: List[str] = []
        if primary_driver:
            keywords.append(primary_driver)
        keywords.extend(_title_keywords(event_title))
        if semantic_frame and semantic_frame.signal_keywords:
            keywords.extend(list(semantic_frame.signal_keywords)[:8])

        # add semantic role names/aliases for better recall
        if semantic_frame:
            for role in (semantic_frame.actors or [])[:2]:
                keywords.extend(_role_aliases(role))
            for role in (semantic_frame.candidates or [])[:3]:
                keywords.extend(_role_aliases(role))

        if extra_keywords:
            keywords.extend(extra_keywords)

        keywords = _dedupe_preserve(keywords)[:20]

        signals = self._store.query_signals(keywords=keywords, window_hours=window_hours, limit=max(25, int(limit) * 3))

        annotated: List[Tuple[float, DistilledTruthSignal]] = []
        for s in signals:
            linked_roles = _link_roles_for_signal(s, semantic_frame) if semantic_frame else []
            if linked_roles:
                s = DistilledTruthSignal(
                    **{
                        **s.to_dict(),
                        "linked_roles": linked_roles,
                    }
                )
            score = _relevance_score(s, keywords, linked_roles)
            annotated.append((score, s))

        annotated.sort(key=lambda t: t[0], reverse=True)
        top = [s for _score, s in annotated[: max(0, int(limit))]]

        # Aggregate stats for metadata (posts_seen = window stats, not just matches)
        window_stats = self._store.get_window_post_stats(window_hours=window_hours)
        last_ingest = self._store.get_last_ingest_stats()

        stats = {
            **window_stats,
            "signals_emitted": len(top),
            "gathered_at": time.time(),
            "keywords_used": keywords[:10],
            "last_ingest": last_ingest,
            "window_hours": float(window_hours),
        }
        return RoutedTruthSignals(top_signals=top, stats=stats)


_global_router: Optional[TruthSocialSignalRouter] = None


def get_truth_social_signal_router() -> TruthSocialSignalRouter:
    global _global_router
    if _global_router is None:
        _global_router = TruthSocialSignalRouter()
        logger.info("Initialized global TruthSocialSignalRouter")
    return _global_router

