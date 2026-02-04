"""
PairIndexBuilder - Main orchestrator for the pair index pipeline.

Dual-mode:
  1. CLI one-shot: build() fetches all events, matches, persists
  2. Service refresh: refresh() incrementally updates the index

Pipeline: Fetch -> Extract Features -> Tier 1 Text Match -> Tier 2 LLM Match -> Persist
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .fetchers.kalshi import KalshiFetcher
from .fetchers.models import MatchCandidate, NormalizedEvent
from .fetchers.polymarket import PolymarketFetcher
from .matchers.llm_matcher import LLMMatcher
from .matchers.text_matcher import TextMatcher, populate_features
from .store import PairStore

logger = logging.getLogger("kalshiflow_rl.pair_index.builder")


@dataclass
class BuildResult:
    """Result of a build/refresh cycle."""
    kalshi_events: int = 0
    poly_events: int = 0
    tier1_event_matches: int = 0
    tier1_market_pairs: int = 0
    tier1_near_misses: int = 0
    tier2_event_matches: int = 0
    tier2_market_pairs: int = 0
    total_pairs: int = 0
    upserted: int = 0
    deactivated: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Kalshi events: {self.kalshi_events}",
            f"Poly events:   {self.poly_events}",
            f"Tier 1: {self.tier1_event_matches} event matches -> {self.tier1_market_pairs} market pairs ({self.tier1_near_misses} near misses)",
            f"Tier 2: {self.tier2_event_matches} event matches -> {self.tier2_market_pairs} market pairs",
            f"Total:  {self.total_pairs} pairs",
            f"DB:     {self.upserted} upserted, {self.deactivated} deactivated",
            f"Time:   {self.duration_seconds:.1f}s",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


class PairIndexBuilder:
    """
    Orchestrates the full pair index pipeline.

    Can be used as:
      - CLI tool (one-shot build)
      - Library (called from trader for background refresh)
    """

    def __init__(
        self,
        kalshi_fetcher: Optional[KalshiFetcher] = None,
        poly_fetcher: Optional[PolymarketFetcher] = None,
        text_matcher: Optional[TextMatcher] = None,
        llm_matcher: Optional[LLMMatcher] = None,
        store: Optional[PairStore] = None,
    ):
        self._kalshi = kalshi_fetcher or KalshiFetcher()
        self._poly = poly_fetcher or PolymarketFetcher()
        self._text_matcher = text_matcher or TextMatcher()
        self._llm_matcher = llm_matcher or LLMMatcher()
        self._store = store or PairStore()

        # Track known active tickers for incremental refresh
        self._known_kalshi_tickers: Set[str] = set()

    async def build(
        self,
        dry_run: bool = False,
        no_llm: bool = False,
        verbose: bool = False,
    ) -> BuildResult:
        """
        Full pipeline: fetch -> extract features -> match -> persist.

        Args:
            dry_run: Show matches without writing to DB
            no_llm: Skip Tier 2 LLM matching (deterministic only)
            verbose: Extra logging
        """
        result = BuildResult()
        start = time.monotonic()

        try:
            # 1. Fetch events from both venues
            kalshi_events, poly_events = await self._fetch_all(result, verbose)
            if not kalshi_events or not poly_events:
                result.duration_seconds = time.monotonic() - start
                return result

            # 2. Extract text features on all events/markets
            self._extract_features(kalshi_events, poly_events)

            # 3. Tier 1: Deterministic text matching
            all_pairs = self._run_tier1(kalshi_events, poly_events, result, verbose)

            # 4. Tier 2: LLM fallback (if enabled)
            if not no_llm:
                await self._run_tier2(kalshi_events, poly_events, result, all_pairs, verbose)

            result.total_pairs = len(all_pairs)

            # 5. Deduplicate
            all_pairs = self._deduplicate(all_pairs)

            if verbose:
                for p in all_pairs:
                    logger.info(
                        f"  PAIR: {p.kalshi_ticker} <-> {p.poly_condition_id[:16]}... "
                        f"[{p.match_method}/tier{p.match_tier}] score={p.combined_score:.2f}"
                    )

            # 6. Persist to DB
            if not dry_run and self._store.available:
                result.upserted = self._store.upsert_pairs(all_pairs)

                # Track active tickers
                active_tickers = {e.event_id for e in kalshi_events}
                # Flatten market tickers
                for e in kalshi_events:
                    for m in e.markets:
                        if m.kalshi_ticker:
                            active_tickers.add(m.kalshi_ticker)

                result.deactivated = self._store.deactivate_stale(active_tickers)
                self._known_kalshi_tickers = active_tickers
            elif dry_run:
                logger.info(f"DRY RUN: would upsert {len(all_pairs)} pairs")

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Build failed: {e}", exc_info=True)

        result.duration_seconds = time.monotonic() - start
        return result

    async def refresh(
        self,
        no_llm: bool = False,
    ) -> BuildResult:
        """
        Incremental refresh: re-fetch, match new events, deactivate closed.

        Same as build() but intended for background use inside the trader.
        Only processes events not already in _known_kalshi_tickers (if populated).
        """
        # For now, refresh is the same as build. Incremental optimization
        # can be added later (skip already-matched events).
        return await self.build(no_llm=no_llm)

    def report_bad_pair(self, kalshi_ticker: str, reason: str = "") -> bool:
        """
        Mark a pair as bad (called by orchestrator when mismatch detected).

        Deactivates in DB so it won't be loaded on next startup.
        """
        if not self._store.available:
            logger.warning(f"Cannot report bad pair {kalshi_ticker}: no DB connection")
            return False
        return self._store.deactivate_pair(kalshi_ticker, reason)

    async def status(self) -> Dict[str, Any]:
        """Get current index stats from DB."""
        return self._store.get_stats()

    # ── Internal pipeline steps ─────────────────────────────────────────

    async def _fetch_all(
        self,
        result: BuildResult,
        verbose: bool,
    ) -> tuple:
        """Fetch events from both venues."""
        kalshi_events = await self._kalshi.fetch_events()
        poly_events = await self._poly.fetch_events()

        result.kalshi_events = len(kalshi_events)
        result.poly_events = len(poly_events)

        if verbose:
            logger.info(f"Fetched {len(kalshi_events)} Kalshi, {len(poly_events)} Poly events")

        if not kalshi_events:
            result.errors.append("No Kalshi events fetched")
        if not poly_events:
            result.errors.append("No Polymarket events fetched")

        return kalshi_events, poly_events

    def _extract_features(
        self,
        kalshi_events: List[NormalizedEvent],
        poly_events: List[NormalizedEvent],
    ) -> None:
        """Populate text features on all events and markets."""
        for e in kalshi_events:
            populate_features(e)
        for e in poly_events:
            populate_features(e)

    def _run_tier1(
        self,
        kalshi_events: List[NormalizedEvent],
        poly_events: List[NormalizedEvent],
        result: BuildResult,
        verbose: bool,
    ) -> List[MatchCandidate]:
        """Run Tier 1 deterministic matching."""
        # Event matching
        matched_events, self._unmatched_k, self._unmatched_p = (
            self._text_matcher.match_events(kalshi_events, poly_events)
        )
        result.tier1_event_matches = len(matched_events)

        if verbose:
            for ke, pe, score in matched_events:
                logger.info(f"  EVENT: \"{ke.title}\" <-> \"{pe.title}\" (score={score:.2f})")

        # Market matching within event pairs
        confirmed, self._near_misses = self._text_matcher.match_markets(matched_events)
        result.tier1_market_pairs = len(confirmed)
        result.tier1_near_misses = len(self._near_misses)

        return confirmed

    async def _run_tier2(
        self,
        kalshi_events: List[NormalizedEvent],
        poly_events: List[NormalizedEvent],
        result: BuildResult,
        all_pairs: List[MatchCandidate],
        verbose: bool,
    ) -> None:
        """Run Tier 2 LLM matching on unmatched events + near misses."""
        if not self._llm_matcher.available:
            if verbose:
                logger.info("Tier 2: LLM not available, skipping")
            return

        unmatched_k = getattr(self, "_unmatched_k", [])
        unmatched_p = getattr(self, "_unmatched_p", [])
        near_misses = getattr(self, "_near_misses", [])

        if not unmatched_k and not near_misses:
            return

        # LLM event matching on unmatched events
        llm_event_matches = await self._llm_matcher.match_events(unmatched_k, unmatched_p)
        result.tier2_event_matches = len(llm_event_matches)

        # LLM market matching
        if llm_event_matches:
            llm_market_pairs = await self._llm_matcher.match_markets(
                llm_event_matches, near_misses
            )
            result.tier2_market_pairs = len(llm_market_pairs)
            all_pairs.extend(llm_market_pairs)

    def _deduplicate(self, pairs: List[MatchCandidate]) -> List[MatchCandidate]:
        """Remove duplicate pairs, keeping highest score."""
        seen: Dict[str, MatchCandidate] = {}
        for p in pairs:
            key = f"{p.kalshi_ticker}|{p.poly_condition_id}"
            if key not in seen or p.combined_score > seen[key].combined_score:
                seen[key] = p
        return list(seen.values())
