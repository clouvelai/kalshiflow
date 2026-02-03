"""
Pairing Service - Deterministic two-level matching engine.

Matches Kalshi events/markets to Polymarket events/markets using structured
API data. No LLM dependency - pure text similarity + optional embeddings.

Pipeline: Fetch -> Normalize -> Match Events -> Match Markets -> Score -> Return Candidates
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("kalshiflow_rl.traderv3.services.pairing_service")

# Category mapping: Kalshi category -> compatible Polymarket categories/tags
CATEGORY_MAP = {
    "politics": {"politics", "us-politics", "government", "elections", "political"},
    "economics": {"economics", "finance", "fed", "inflation", "economy"},
    "crypto": {"crypto", "cryptocurrency", "bitcoin", "ethereum", "defi"},
    "financials": {"finance", "markets", "stocks", "economics", "financial"},
    "sports": {"sports", "nfl", "nba", "mlb", "soccer", "football", "basketball"},
    "climate and weather": {"weather", "climate", "environment"},
    "science and technology": {"technology", "science", "ai", "tech"},
    "entertainment": {"entertainment", "culture", "media", "movies", "tv"},
}

# Prefixes to strip for better market-level matching
PREFIXES_TO_STRIP = [
    r"^will trump (?:next )?nominate\s+",
    r"^will the\s+",
    r"^will\s+",
    r"^what will\s+",
    r"^how many\s+",
    r"^who will\s+",
    r"^will there be\s+",
    r"^is\s+",
    r"^does\s+",
    r"^do\s+",
]

# Suffixes to strip
SUFFIXES_TO_STRIP = [
    r"\s*\?$",
    r"\s+as (?:the )?(?:next )?fed chair\s*\??$",
    r"\s+be the next\s+.*$",
    r"\s+win the\s+\w+\s+senate race\s*\??$",
    r"\s+win the\s+\w+\s+\w+\s+senate race\s*\??$",
]

# Pre-computed union of all known Polymarket categories (used for compatibility check)
_ALL_KNOWN_POLY_CATS = frozenset(
    cat for cats in CATEGORY_MAP.values() for cat in cats
)


# TTL for cached series lookup (seconds)
_SERIES_CACHE_TTL = 600.0


@dataclass
class NormalizedMarket:
    """A normalized market from either venue."""
    venue: str  # "kalshi" | "polymarket"
    event_id: str  # kalshi event_ticker or poly event id
    market_id: str  # kalshi ticker or poly condition_id
    question: str  # market title / question
    normalized_question: str  # stripped prefixes, lowered
    close_time: Optional[datetime] = None
    is_active: bool = True
    # Venue-specific IDs for registration
    kalshi_ticker: Optional[str] = None
    kalshi_event_ticker: Optional[str] = None
    poly_condition_id: Optional[str] = None
    poly_token_id_yes: Optional[str] = None
    poly_token_id_no: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedEvent:
    """A normalized event from either venue."""
    venue: str  # "kalshi" | "polymarket"
    event_id: str  # kalshi event_ticker or poly event slug/id
    title: str
    normalized_title: str
    category: str
    close_time: Optional[datetime] = None
    mutually_exclusive: bool = False
    market_count: int = 0
    markets: List[NormalizedMarket] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePair:
    """A candidate cross-venue market pair with scoring."""
    kalshi_event: NormalizedEvent
    poly_event: NormalizedEvent
    kalshi_market: NormalizedMarket
    poly_market: NormalizedMarket
    event_title_score: float
    market_title_score: float
    embedding_score: Optional[float] = None
    combined_score: float = 0.0
    match_signals: List[str] = field(default_factory=list)
    # Pre-computed for direct activation
    kalshi_ticker: str = ""
    kalshi_event_ticker: str = ""
    poly_condition_id: str = ""
    poly_token_id_yes: str = ""
    poly_token_id_no: Optional[str] = None
    question: str = ""


class PairingService:
    """
    Deterministic two-level matching engine for cross-venue market pairing.

    Level 1: Match events across Kalshi and Polymarket
    Level 2: Match individual markets within matched events
    """

    def __init__(
        self,
        poly_client: Any,
        trading_client: Any,
        pair_registry: Any,
        embedding_model: Any = None,
        supabase_client: Any = None,
        spread_monitor: Any = None,
        event_bus: Any = None,
        openai_client: Any = None,
    ):
        self._poly_client = poly_client
        self._trading_client = trading_client
        self._pair_registry = pair_registry
        self._embedding_model = embedding_model
        self._supabase = supabase_client
        self._spread_monitor = spread_monitor
        self._event_bus = event_bus
        self._openai_client = openai_client

        # Kalshi series_ticker -> category cache (built from API)
        self._series_category_cache: Dict[str, str] = {}
        self._series_cache_expires_at: float = 0.0

        # Stats
        self._last_scan_at: Optional[float] = None
        self._total_candidates_found = 0
        self._total_pairs_activated = 0
        self._scan_count = 0

    # ─── Fetch + Normalize ─────────────────────────────────────────────

    async def _get_series_category_map(self) -> Dict[str, str]:
        """Fetch series_ticker -> category mapping from the Kalshi API.

        Cached for _SERIES_CACHE_TTL seconds to avoid repeated API calls.
        """
        now = time.monotonic()
        if self._series_category_cache and now < self._series_cache_expires_at:
            return self._series_category_cache

        if not self._trading_client:
            return self._series_category_cache

        try:
            series_list = await self._trading_client.get_series()
            lookup: Dict[str, str] = {}
            for s in series_list:
                ticker = (s.get("series_ticker") or s.get("ticker") or "").strip()
                cat = (s.get("category", "") or "").lower().strip()
                if ticker and cat:
                    lookup[ticker] = cat
            self._series_category_cache = lookup
            self._series_cache_expires_at = now + _SERIES_CACHE_TTL
            logger.info(f"Cached {len(lookup)} Kalshi series->category mappings from API")
        except Exception as e:
            logger.warning(f"Failed to fetch Kalshi series for category lookup: {e}")

        return self._series_category_cache

    async def fetch_kalshi_events(self, limit: int = 200, max_pages: int = 5) -> List[NormalizedEvent]:
        """Fetch and normalize Kalshi events with nested markets.

        Paginates through cursor pages to get broader coverage (Kalshi often
        has 500+ open events but returns max 200 per page).
        """
        if not self._trading_client:
            logger.warning("No trading client available for Kalshi event fetch")
            return []

        raw_events = []
        cursor = None
        for page in range(max_pages):
            try:
                result = await self._trading_client.get_events(
                    limit=limit,
                    status="open",
                    with_nested_markets=True,
                    cursor=cursor,
                )
                page_events = result.get("events", []) if isinstance(result, dict) else (result or [])
                if not page_events:
                    break
                raw_events.extend(page_events)
                cursor = result.get("cursor") if isinstance(result, dict) else None
                if not cursor:
                    break
            except Exception as e:
                logger.error(f"Failed to fetch Kalshi events page {page}: {e}")
                break

        if not raw_events:
            return []

        # Build series_ticker -> category lookup from API (cached)
        series_category_map = await self._get_series_category_map()

        events = []
        for raw in raw_events:
            raw_markets = raw.get("markets", [])
            title = raw.get("title", "")
            event_ticker = raw.get("event_ticker", "")
            category = (raw.get("category", "") or "").lower().strip()

            # Fallback: look up category from series API when deprecated field is empty
            if not category:
                series_ticker = (raw.get("series_ticker", "") or "").strip()
                category = series_category_map.get(series_ticker, "")

            # Parse close time from earliest market close
            close_time = self._parse_kalshi_close_time(raw_markets)

            markets = []
            for m in raw_markets:
                m_status = (m.get("status", "") or "").lower()
                m_title = m.get("title", "") or m.get("subtitle", "") or ""
                ticker = m.get("ticker", "")

                markets.append(NormalizedMarket(
                    venue="kalshi",
                    event_id=event_ticker,
                    market_id=ticker,
                    question=m_title,
                    normalized_question=self.normalize_question(m_title),
                    close_time=self._parse_iso_datetime(m.get("close_time")),
                    is_active=m_status in ("open", "active", ""),
                    kalshi_ticker=ticker,
                    kalshi_event_ticker=event_ticker,
                    raw=m,
                ))

            events.append(NormalizedEvent(
                venue="kalshi",
                event_id=event_ticker,
                title=title,
                normalized_title=self._normalize_title(title),
                category=category,
                close_time=close_time,
                mutually_exclusive=raw.get("mutually_exclusive", False),
                market_count=len(markets),
                markets=markets,
                raw=raw,
            ))

        logger.info(f"Fetched {len(events)} Kalshi events with {sum(e.market_count for e in events)} markets")
        return events

    async def fetch_poly_events(self, limit: int = 100, pages: int = 5) -> List[NormalizedEvent]:
        """Fetch and normalize Polymarket events with nested markets.

        Paginates through multiple pages to get broader coverage.
        """
        if not self._poly_client:
            logger.warning("No Polymarket client available for event fetch")
            return []

        # Paginate to get more events
        raw_events = []
        for page in range(pages):
            try:
                page_events = await self._poly_client.get_events(
                    limit=limit, active=True, offset=page * limit
                )
                if not page_events:
                    break
                raw_events.extend(page_events)
            except Exception as e:
                logger.error(f"Failed to fetch Polymarket events page {page}: {e}")
                break

        if not raw_events:
            return []

        events = []
        for raw in raw_events:
            title = raw.get("title", "")
            slug = raw.get("slug", "")
            event_id = slug or raw.get("id", "")
            category = (raw.get("category", "") or "").lower().strip()
            raw_tags = raw.get("tags", []) or []
            tags = []
            for t in raw_tags:
                if isinstance(t, str):
                    tags.append(t.lower().strip())
                elif isinstance(t, dict):
                    # Polymarket tags can be {"label": "politics", "slug": "politics"}
                    tag_label = t.get("label", "") or t.get("slug", "") or t.get("name", "")
                    if tag_label:
                        tags.append(str(tag_label).lower().strip())

            # Polymarket events may have "enableNegRisk" indicating multi-outcome
            mutually_exclusive = bool(raw.get("enableNegRisk", False))

            end_date = self._parse_iso_datetime(raw.get("endDate"))

            raw_markets = raw.get("markets", [])
            markets = []
            for m in raw_markets:
                question = m.get("question", "") or m.get("groupItemTitle", "") or ""
                condition_id = m.get("conditionId", "") or m.get("condition_id", "")
                # clobTokenIds may be a JSON string or a native array
                raw_clob = m.get("clobTokenIds") or []
                if isinstance(raw_clob, str):
                    try:
                        raw_clob = json.loads(raw_clob)
                    except (json.JSONDecodeError, ValueError):
                        raw_clob = []
                clob_token_ids = raw_clob if isinstance(raw_clob, list) else []
                token_yes = clob_token_ids[0] if len(clob_token_ids) > 0 else ""
                token_no = clob_token_ids[1] if len(clob_token_ids) > 1 else None

                is_active = bool(m.get("active", True)) and not bool(m.get("closed", False))

                markets.append(NormalizedMarket(
                    venue="polymarket",
                    event_id=event_id,
                    market_id=condition_id,
                    question=question,
                    normalized_question=self.normalize_question(question),
                    close_time=self._parse_iso_datetime(m.get("endDate")) or end_date,
                    is_active=is_active,
                    poly_condition_id=condition_id,
                    poly_token_id_yes=token_yes,
                    poly_token_id_no=token_no,
                    raw=m,
                ))

            # Combine category + tags for matching
            combined_category = category or (tags[0] if tags else "")

            events.append(NormalizedEvent(
                venue="polymarket",
                event_id=event_id,
                title=title,
                normalized_title=self._normalize_title(title),
                category=combined_category,
                close_time=end_date,
                mutually_exclusive=mutually_exclusive,
                market_count=len(markets),
                markets=markets,
                raw={**raw, "_tags": tags},
            ))

        logger.info(f"Fetched {len(events)} Polymarket events with {sum(e.market_count for e in events)} markets")
        return events

    # ─── Two-Level Matching ────────────────────────────────────────────

    def match_events(
        self,
        kalshi_events: List[NormalizedEvent],
        poly_events: List[NormalizedEvent],
        min_event_score: float = 0.25,
    ) -> List[Tuple[NormalizedEvent, NormalizedEvent, float]]:
        """
        Match Kalshi events to Polymarket events by combining:
        1. SequenceMatcher on normalized titles
        2. Keyword/token overlap (Jaccard similarity)
        3. Market title cross-matching (do any markets overlap?)

        Returns list of (kalshi_event, poly_event, score) sorted by score desc.
        """
        candidates = []

        # Pre-compute keyword sets for all events
        k_keywords = {ke.event_id: self._extract_keywords(ke.normalized_title) for ke in kalshi_events}
        p_keywords = {pe.event_id: self._extract_keywords(pe.normalized_title) for pe in poly_events}

        # Pre-compute market keyword sets for cross-matching
        def _event_market_keywords(events):
            return {
                e.event_id: {
                    kw for m in e.markets
                    for kw in self._extract_keywords(m.normalized_question)
                }
                for e in events
            }

        k_market_kw = _event_market_keywords(kalshi_events)
        p_market_kw = _event_market_keywords(poly_events)

        for ke in kalshi_events:
            if ke.market_count == 0:
                continue

            for pe in poly_events:
                if pe.market_count == 0:
                    continue

                # Compute close-time gap once for both pre-filter and bonus
                if ke.close_time and pe.close_time:
                    gap_days = abs((ke.close_time - pe.close_time).total_seconds()) / 86400
                else:
                    gap_days = None

                # Skip if close times are wildly different AND title similarity is low
                if gap_days is not None and gap_days > 365:
                    pre_jaccard = self._jaccard(
                        k_keywords.get(ke.event_id, set()),
                        p_keywords.get(pe.event_id, set()),
                    )
                    if pre_jaccard < 0.5:
                        continue

                # Compute multiple similarity signals
                title_seq = SequenceMatcher(
                    None, ke.normalized_title, pe.normalized_title
                ).ratio()

                k_kw = k_keywords[ke.event_id]
                p_kw = p_keywords[pe.event_id]
                keyword_jaccard = self._jaccard(k_kw, p_kw)

                market_keyword_overlap = self._jaccard(
                    k_market_kw[ke.event_id], p_market_kw[pe.event_id]
                )

                # Combined event score: best of three signals
                score = max(
                    title_seq,
                    keyword_jaccard * 1.2,  # Boost keyword matching
                    market_keyword_overlap * 0.8,  # Market overlap is a weaker signal
                )

                if score < min_event_score:
                    continue

                # Bonus signals
                if ke.mutually_exclusive and pe.mutually_exclusive:
                    score += 0.05
                if ke.market_count > 1 and pe.market_count > 1:
                    count_ratio = min(ke.market_count, pe.market_count) / max(ke.market_count, pe.market_count)
                    if count_ratio > 0.5:
                        score += 0.05
                if self._categories_compatible(ke.category, pe.category, pe.raw.get("_tags", [])):
                    score += 0.03
                # Close time proximity bonus (reuse gap_days computed above)
                if gap_days is not None:
                    if gap_days < 7:
                        score += 0.05
                    elif gap_days < 30:
                        score += 0.02

                candidates.append((ke, pe, round(min(score, 1.0), 4)))

        candidates.sort(key=lambda x: x[2], reverse=True)
        logger.info(f"Event matching: {len(candidates)} candidate event pairs from {len(kalshi_events)}x{len(poly_events)}")
        return candidates

    def match_markets_within_events(
        self,
        kalshi_event: NormalizedEvent,
        poly_event: NormalizedEvent,
        min_market_score: float = 0.3,
    ) -> List[Tuple[NormalizedMarket, NormalizedMarket, float]]:
        """
        Match individual markets within a pair of matched events.

        Uses normalized question similarity + keyword overlap (greedy best-match).
        """
        k_markets = [m for m in kalshi_event.markets if m.is_active]
        p_markets = [m for m in poly_event.markets if m.is_active]

        if not k_markets or not p_markets:
            return []

        # Compute all pairwise similarities using both methods
        scores = []
        for ki, km in enumerate(k_markets):
            km_kw = self._extract_keywords(km.normalized_question)
            for pi, pm in enumerate(p_markets):
                pm_kw = self._extract_keywords(pm.normalized_question)

                seq_sim = SequenceMatcher(
                    None, km.normalized_question, pm.normalized_question
                ).ratio()
                kw_sim = self._jaccard(km_kw, pm_kw)

                # Use the better of the two signals
                sim = min(max(seq_sim, kw_sim * 1.1), 1.0)

                if sim >= min_market_score:
                    scores.append((ki, pi, sim, km, pm))

        # Greedy best-match assignment
        scores.sort(key=lambda x: x[2], reverse=True)
        used_k = set()
        used_p = set()
        pairs = []

        for ki, pi, sim, km, pm in scores:
            if ki in used_k or pi in used_p:
                continue
            used_k.add(ki)
            used_p.add(pi)
            pairs.append((km, pm, round(sim, 4)))

        return pairs

    # ─── Scoring ───────────────────────────────────────────────────────

    async def score_candidates(
        self,
        event_pairs: List[Tuple[NormalizedEvent, NormalizedEvent, float]],
        use_embeddings: bool = True,
        min_combined_score: float = 0.5,
    ) -> List[CandidatePair]:
        """
        Score all market pairs from matched events.

        Combines text similarity with optional embedding similarity.
        """
        # Collect all market pairs across event pairs
        raw_candidates: List[Tuple[NormalizedEvent, NormalizedEvent, float, NormalizedMarket, NormalizedMarket, float]] = []

        for ke, pe, event_score in event_pairs:
            market_pairs = self.match_markets_within_events(ke, pe)
            for km, pm, market_score in market_pairs:
                raw_candidates.append((ke, pe, event_score, km, pm, market_score))

        if not raw_candidates:
            return []

        # Batch embedding if available
        embedding_scores: Dict[int, float] = {}
        if use_embeddings and self._embedding_model:
            try:
                texts = []
                for _, _, _, km, pm, _ in raw_candidates:
                    texts.append(km.question)
                    texts.append(pm.question)

                embeddings = await self._embedding_model.aembed_documents(texts)

                import numpy as np
                for i in range(0, len(embeddings), 2):
                    idx = i // 2
                    a = np.array(embeddings[i])
                    b = np.array(embeddings[i + 1])
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)
                    if norm_a > 0 and norm_b > 0:
                        sim = float(np.dot(a, b) / (norm_a * norm_b))
                        embedding_scores[idx] = round(sim, 4)

                logger.info(f"Computed embeddings for {len(raw_candidates)} candidate pairs")
            except Exception as e:
                logger.warning(f"Embedding scoring failed (falling back to text only): {e}")

        # Build CandidatePair objects
        candidates = []
        for idx, (ke, pe, event_score, km, pm, market_score) in enumerate(raw_candidates):
            emb_score = embedding_scores.get(idx)

            # Combined score: weight embeddings higher when available
            if emb_score is not None:
                combined = 0.3 * market_score + 0.7 * emb_score
            else:
                combined = market_score

            if combined < min_combined_score:
                continue

            # Build match signals
            signals = []
            if self._categories_compatible(ke.category, pe.category, pe.raw.get("_tags", [])):
                signals.append("same_category")
            if ke.close_time and pe.close_time:
                gap_days = abs((ke.close_time - pe.close_time).total_seconds()) / 86400
                if gap_days < 7:
                    signals.append("close_time_match")
            if market_score > 0.8:
                signals.append("high_text_sim")
            if emb_score and emb_score > 0.9:
                signals.append("high_embedding_sim")
            if ke.mutually_exclusive and pe.mutually_exclusive:
                signals.append("both_multi_outcome")

            candidates.append(CandidatePair(
                kalshi_event=ke,
                poly_event=pe,
                kalshi_market=km,
                poly_market=pm,
                event_title_score=event_score,
                market_title_score=market_score,
                embedding_score=emb_score,
                combined_score=round(combined, 4),
                match_signals=signals,
                kalshi_ticker=km.kalshi_ticker or "",
                kalshi_event_ticker=km.kalshi_event_ticker or "",
                poly_condition_id=pm.poly_condition_id or "",
                poly_token_id_yes=pm.poly_token_id_yes or "",
                poly_token_id_no=pm.poly_token_id_no,
                question=km.question or pm.question,
            ))

        candidates.sort(key=lambda c: c.combined_score, reverse=True)
        return candidates

    # ─── Targeted Pipeline ──────────────────────────────────────────────

    async def match_for_events(
        self,
        kalshi_events: List[NormalizedEvent],
        min_score: float = 0.5,
        max_candidates: int = 50,
        use_llm: bool = False,
    ) -> List[CandidatePair]:
        """Match a specific set of Kalshi events against Polymarket.

        Unlike find_candidates() which fetches everything, this matches
        only the provided Kalshi events against the Poly catalog.

        When use_llm=True and an OpenAI client is available, uses GPT-4o-mini
        to find correct market-level pairings from matched events (much more
        accurate than text similarity alone for cross-venue phrasing).
        """
        poly_events = await self.fetch_poly_events()
        if not poly_events:
            return []

        event_pairs = self.match_events(kalshi_events, poly_events)
        if not event_pairs:
            return []

        # LLM path: let the model find market-level pairings from event pairs
        if use_llm and self._openai_client:
            logger.info(f"Pre-filter: {len(event_pairs)} event pairs -> sending to LLM for market matching")
            llm_candidates = await self.validate_pairs_llm(
                candidates=[],
                event_pairs=event_pairs,
            )
            if llm_candidates:
                return llm_candidates[:max_candidates]
            # Fall through to text-based matching if LLM returns nothing
            logger.info("LLM returned no pairs - falling back to text matching")

        candidates = await self.score_candidates(event_pairs, min_combined_score=min_score)
        candidates = [c for c in candidates if not self._already_paired(c.kalshi_ticker)]
        return candidates[:max_candidates]

    # ─── Full Pipeline ─────────────────────────────────────────────────

    async def find_candidates(
        self,
        min_score: float = 0.6,
        max_candidates: int = 30,
    ) -> List[CandidatePair]:
        """
        Run the full matching pipeline: fetch -> normalize -> match -> score.

        Returns pre-scored candidates for LLM validation.
        """
        self._scan_count += 1
        self._last_scan_at = time.time()

        # Fetch in parallel
        kalshi_events, poly_events = await asyncio.gather(
            self.fetch_kalshi_events(),
            self.fetch_poly_events(),
        )

        if not kalshi_events or not poly_events:
            logger.info("No events from one or both venues - nothing to match")
            return []

        # Match events
        event_pairs = self.match_events(kalshi_events, poly_events)
        if not event_pairs:
            logger.info("No event-level matches found")
            return []

        # Score all market pairs (includes embedding if available)
        candidates = await self.score_candidates(
            event_pairs, min_combined_score=min_score
        )

        # Filter already-paired markets
        candidates = [
            c for c in candidates
            if not self._already_paired(c.kalshi_ticker)
        ]

        # Limit results
        candidates = candidates[:max_candidates]

        self._total_candidates_found += len(candidates)
        logger.info(f"Pipeline complete: {len(candidates)} new candidates (min_score={min_score})")
        return candidates

    # ─── Activation ────────────────────────────────────────────────────

    async def activate_pair(
        self,
        candidate: CandidatePair,
        match_method: str = "agent",
        confidence: float = 0.9,
    ) -> Dict[str, Any]:
        """
        Activate a single validated pair: DB upsert + registry + spread monitor.
        """
        if not candidate.kalshi_ticker or not candidate.poly_condition_id:
            return {"error": "Missing required IDs", "kalshi_ticker": candidate.kalshi_ticker}

        if not candidate.poly_token_id_yes:
            return {"error": "Missing poly_token_id_yes", "kalshi_ticker": candidate.kalshi_ticker}

        # 1. Upsert to DB
        pair_id = None
        if self._supabase:
            base_row = {
                "kalshi_ticker": candidate.kalshi_ticker,
                "kalshi_event_ticker": candidate.kalshi_event_ticker,
                "poly_condition_id": candidate.poly_condition_id,
                "poly_token_id_yes": candidate.poly_token_id_yes,
                "question": candidate.question,
                "match_method": match_method,
                "match_confidence": confidence,
                "status": "active",
            }
            if candidate.poly_token_id_no:
                base_row["poly_token_id_no"] = candidate.poly_token_id_no

            metadata_fields = {
                "text_score": candidate.market_title_score,
                "embedding_score": candidate.embedding_score,
                "event_title_score": candidate.event_title_score,
                "match_signals": candidate.match_signals,
                "validated_by": match_method,
            }

            # Try with metadata columns first, fall back to base row if columns missing
            for row in [{**base_row, **metadata_fields}, base_row]:
                try:
                    result = self._supabase.table("paired_markets").upsert(
                        row, on_conflict="kalshi_ticker,poly_condition_id"
                    ).execute()
                    pair_data = result.data[0] if result.data else {}
                    pair_id = pair_data.get("id")
                    break
                except Exception as e:
                    if row is not base_row:
                        logger.debug(f"Upsert with metadata failed, retrying without: {e}")
                        continue
                    logger.error(f"Failed to upsert pair {candidate.kalshi_ticker}: {e}")
                    return {"error": str(e), "kalshi_ticker": candidate.kalshi_ticker}

        # 2. Register in PairRegistry
        if pair_id and self._pair_registry:
            from .pair_registry import MarketPair
            pair = MarketPair(
                id=pair_id,
                kalshi_ticker=candidate.kalshi_ticker,
                kalshi_event_ticker=candidate.kalshi_event_ticker,
                poly_condition_id=candidate.poly_condition_id,
                poly_token_id_yes=candidate.poly_token_id_yes,
                poly_token_id_no=candidate.poly_token_id_no,
                question=candidate.question,
                match_method=match_method,
                match_confidence=confidence,
            )
            await self._pair_registry.add_pair(pair)

            # 3. Register with SpreadMonitor
            if self._spread_monitor:
                self._spread_monitor.register_pair(pair)

            # 4. Emit PAIR_MATCHED event
            if self._event_bus:
                from ..core.events.types import EventType
                from ..core.events.arb_events import PairMatchedEvent
                try:
                    await self._event_bus.emit(
                        EventType.PAIR_MATCHED,
                        PairMatchedEvent(
                            pair_id=pair_id,
                            kalshi_ticker=candidate.kalshi_ticker,
                            poly_condition_id=candidate.poly_condition_id,
                            poly_token_id_yes=candidate.poly_token_id_yes,
                            question=candidate.question,
                            match_method=match_method,
                            match_confidence=confidence,
                        ),
                    )
                except Exception:
                    pass

        self._total_pairs_activated += 1
        logger.info(f"Activated pair: {candidate.kalshi_ticker} <-> poly:{candidate.poly_condition_id[:16]}... (score={candidate.combined_score})")

        return {
            "pair_id": pair_id,
            "kalshi_ticker": candidate.kalshi_ticker,
            "question": candidate.question,
            "combined_score": candidate.combined_score,
            "status": "activated",
        }

    async def activate_pairs_batch(
        self,
        candidates: List[CandidatePair],
        match_method: str = "agent",
        default_confidence: float = 0.9,
    ) -> Dict[str, Any]:
        """Activate multiple validated pairs."""
        results = []
        activated = 0
        failed = 0

        for c in candidates:
            result = await self.activate_pair(c, match_method=match_method, confidence=default_confidence)
            results.append(result)
            if result.get("status") == "activated":
                activated += 1
            else:
                failed += 1

        return {
            "activated": activated,
            "failed": failed,
            "results": results,
        }

    # ─── LLM Pairing ────────────────────────────────────────────────────

    async def validate_pairs_llm(
        self,
        candidates: List[CandidatePair],
        event_pairs: Optional[List[tuple]] = None,
    ) -> List[CandidatePair]:
        """
        Batch pair-match using GPT-4o-mini: validates existing candidates AND
        discovers correct pairings that text similarity missed.

        When event_pairs is provided, the LLM sees all markets from matched events
        and can find the right market-level pairings even when text scores are low.

        Falls back to returning candidates as-is if no OpenAI client is available.
        """
        if not self._openai_client:
            logger.debug("No OpenAI client - skipping LLM pairing")
            return candidates

        if not candidates and not event_pairs:
            return candidates

        # Build a rich context: for each matched event pair, list ALL markets
        # so the LLM can find correct pairings text similarity missed.
        event_pair_map: Dict[str, tuple] = {}  # key -> (kalshi_event, poly_event)
        if event_pairs:
            for ke, pe, _score in event_pairs:
                key = f"{ke.event_id}|{pe.event_id}"
                event_pair_map[key] = (ke, pe)

        # Also extract event pairs from candidates if event_pairs not provided
        if not event_pair_map and candidates:
            for c in candidates:
                key = f"{c.kalshi_event.event_id}|{c.poly_event.event_id}"
                if key not in event_pair_map:
                    event_pair_map[key] = (c.kalshi_event, c.poly_event)

        # Build prompt with full market listings per event pair
        sections = []
        # Track all markets for building CandidatePairs from LLM output
        kalshi_market_index: Dict[str, NormalizedMarket] = {}
        poly_market_index: Dict[str, NormalizedMarket] = {}
        event_index: Dict[str, NormalizedEvent] = {}

        for pair_idx, (key, (ke, pe)) in enumerate(event_pair_map.items()):
            event_index[ke.event_id] = ke
            event_index[pe.event_id] = pe

            k_markets = [m for m in ke.markets if m.is_active]
            p_markets = [m for m in pe.markets if m.is_active]

            k_lines = []
            for i, m in enumerate(k_markets):
                kalshi_market_index[m.market_id] = m
                k_lines.append(f"  K{i}: [{m.market_id}] \"{m.question}\"")

            p_lines = []
            for i, m in enumerate(p_markets):
                poly_market_index[m.market_id] = m
                p_lines.append(f"  P{i}: [{m.market_id}] \"{m.question}\"")

            sections.append(
                f"EVENT PAIR {pair_idx}:\n"
                f"  Kalshi event: \"{ke.title}\" ({ke.event_id})\n"
                f"{chr(10).join(k_lines)}\n"
                f"  Polymarket event: \"{pe.title}\" ({pe.event_id})\n"
                f"{chr(10).join(p_lines)}"
            )

        event_listing = "\n\n".join(sections)

        prompt = (
            "You are a cross-venue market matcher for prediction markets.\n"
            "Below are event pairs (Kalshi vs Polymarket) with their sub-markets.\n"
            "Find ALL market-level pairs that resolve on the SAME real-world outcome.\n"
            "Markets match when they ask the same question about the same outcome, "
            "even if worded differently.\n\n"
            f"{event_listing}\n\n"
            "Return JSON with ALL valid market pairs you can find:\n"
            "{\"pairs\": [\n"
            "  {\"kalshi_market_id\": \"...\", \"poly_market_id\": \"...\", "
            "\"confidence\": 0.95, \"reason\": \"both ask if X happens\"}\n"
            "]}\n"
            "Only include pairs with confidence >= 0.7. "
            "Use the market IDs from the brackets [...]."
        )

        try:
            response = await self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=2048,
            )

            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            llm_pairs = result.get("pairs", [])

            # Build CandidatePair objects from LLM output
            validated: List[CandidatePair] = []
            seen_keys: set = set()

            for lp in llm_pairs:
                k_id = lp.get("kalshi_market_id", "")
                p_id = lp.get("poly_market_id", "")
                confidence = lp.get("confidence", 0.0)

                if confidence < 0.7:
                    continue

                km = kalshi_market_index.get(k_id)
                pm = poly_market_index.get(p_id)
                if not km or not pm:
                    continue

                pair_key = f"{k_id}|{p_id}"
                if pair_key in seen_keys:
                    continue
                seen_keys.add(pair_key)

                # Skip already-paired markets
                if self._already_paired(km.kalshi_ticker or k_id):
                    continue

                ke = event_index.get(km.event_id)
                pe = event_index.get(pm.event_id)
                if not ke or not pe:
                    continue

                validated.append(CandidatePair(
                    kalshi_event=ke,
                    poly_event=pe,
                    kalshi_market=km,
                    poly_market=pm,
                    event_title_score=0.0,
                    market_title_score=confidence,
                    embedding_score=None,
                    combined_score=confidence,
                    match_signals=["llm_matched"],
                    kalshi_ticker=km.kalshi_ticker or "",
                    kalshi_event_ticker=km.kalshi_event_ticker or "",
                    poly_condition_id=pm.poly_condition_id or "",
                    poly_token_id_yes=pm.poly_token_id_yes or "",
                    poly_token_id_no=pm.poly_token_id_no,
                    question=km.question or pm.question,
                ))

            logger.info(
                f"LLM pairing: {len(validated)} pairs found from "
                f"{len(event_pair_map)} event pairs "
                f"(model=gpt-4o-mini)"
            )
            return validated

        except Exception as e:
            logger.warning(f"LLM pairing failed (returning pre-filter candidates): {e}")
            return candidates

    # ─── Helpers ───────────────────────────────────────────────────────

    # Stopwords to ignore in keyword extraction
    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "will", "would", "could", "should", "may", "might", "can",
        "do", "does", "did", "have", "has", "had", "having",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "and", "or", "but", "not", "no", "nor",
        "this", "that", "these", "those",
        "it", "its", "he", "she", "they", "them", "his", "her", "their",
        "what", "which", "who", "whom", "whose", "how", "when", "where", "why",
        "any", "all", "each", "every", "some", "more", "most", "other",
        "than", "then", "also", "just", "about", "before", "after",
        "between", "into", "through", "during", "above", "below",
        "up", "down", "out", "off", "over", "under",
        "next", "first", "last", "new", "many",
        "if", "as", "so", "yet", "both", "either", "neither",
    })

    @classmethod
    def _extract_keywords(cls, text: str) -> set:
        """Extract significant keywords from text, removing stopwords."""
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if w not in cls._STOPWORDS and len(w) > 1}

    @staticmethod
    def _jaccard(set_a: set, set_b: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    @staticmethod
    def normalize_question(text: str) -> str:
        """Normalize a market question for comparison."""
        text = text.lower().strip()

        for pattern in PREFIXES_TO_STRIP:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        for pattern in SUFFIXES_TO_STRIP:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _normalize_title(text: str) -> str:
        """Normalize an event title for comparison."""
        text = text.lower().strip()
        # Remove common filler words
        text = re.sub(r"\s+", " ", text)
        return text

    def _categories_compatible(self, kalshi_cat: str, poly_cat: str, poly_tags: Optional[List[str]] = None) -> bool:
        """Check if categories are compatible across venues.

        Lenient by design: only rejects when we're CERTAIN the categories are
        incompatible (e.g., "sports" vs "crypto"). Unknown categories pass through.
        """
        if not kalshi_cat or not poly_cat:
            return True

        kalshi_cat = kalshi_cat.lower().strip()
        poly_cat = poly_cat.lower().strip()
        poly_tags = [t.lower().strip() for t in (poly_tags or [])]

        if kalshi_cat == poly_cat:
            return True

        compatible_set = CATEGORY_MAP.get(kalshi_cat, set())
        if poly_cat in compatible_set:
            return True
        if any(tag in compatible_set for tag in poly_tags):
            return True

        # Unknown poly category or unknown kalshi category = allow match.
        # Many Poly categories are person names or niche tags that don't map.
        if poly_cat not in _ALL_KNOWN_POLY_CATS:
            return True
        if kalshi_cat not in CATEGORY_MAP:
            return True

        return False

    def _already_paired(self, kalshi_ticker: str) -> bool:
        """Check if a Kalshi ticker is already paired."""
        if not self._pair_registry:
            return False
        return self._pair_registry.get_by_kalshi(kalshi_ticker) is not None

    @staticmethod
    def _parse_iso_datetime(value) -> Optional[datetime]:
        """Parse ISO datetime string to datetime."""
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            # Handle various formats
            value = str(value).replace("Z", "+00:00")
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _parse_kalshi_close_time(cls, markets: List[Dict]) -> Optional[datetime]:
        """Get the earliest close time from Kalshi markets."""
        close_times = []
        for m in markets:
            ct = cls._parse_iso_datetime(
                m.get("close_time") or m.get("expected_expiration_time")
            )
            if ct:
                close_times.append(ct)
        return min(close_times) if close_times else None

    def get_status(self) -> Dict[str, Any]:
        """Get pairing service status."""
        return {
            "scan_count": self._scan_count,
            "last_scan_at": self._last_scan_at,
            "total_candidates_found": self._total_candidates_found,
            "total_pairs_activated": self._total_pairs_activated,
            "has_embedding_model": self._embedding_model is not None,
            "has_openai_client": self._openai_client is not None,
            "has_supabase": self._supabase is not None,
        }
