"""SwingNewsService - Causal news search for price swings + background loop.

Key Responsibilities:
    - Given a PriceSwing, search for causal news around the swing timestamp
    - Store (news_embedding, price_impact) pairs into memory + swing_news_associations
    - Run a background loop that processes unsearched swings from SwingDetector
    - Refresh candlestick data periodically to detect new historical swings

Architecture Position:
    Background service started by SingleArbCoordinator alongside Captain.
    SwingDetector detects swings → SwingNewsService finds causal news →
    stores to memory (type="swing_news") + swing_news_associations table →
    Captain's search_news recalls patterns for predictive trading.

Design Principles:
    - Budget-aware: respects TavilyBudgetManager limits
    - Non-blocking: fire-and-forget memory writes
    - Signal quality: higher impact swings get higher signal_quality scores
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .index import EventArbIndex
    from .memory.session_store import SessionMemoryStore
    from .swing_detector import PriceSwing, SwingDetector
    from .tavily_budget import TavilyBudgetManager
    from .tavily_service import TavilySearchService
    from .article_analyzer import ArticleAnalyzer

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.swing_news_service")


def _signal_quality(change_cents: float) -> float:
    """Map swing magnitude to signal quality score."""
    if change_cents >= 10.0:
        return 1.0
    if change_cents >= 5.0:
        return 0.85
    if change_cents >= 3.0:
        return 0.7
    return 0.5


def _causal_confidence(
    time_delta_hours: float,
    relevance_score: float,
) -> float:
    """Estimate causal confidence from time proximity and article relevance.

    Closer in time + higher relevance → higher confidence.
    """
    # Time proximity score: 1.0 within 1h, decays linearly to 0.3 at 12h+
    if time_delta_hours <= 1.0:
        time_score = 1.0
    elif time_delta_hours <= 6.0:
        time_score = 1.0 - 0.1 * time_delta_hours
    else:
        time_score = max(0.3, 1.0 - 0.05 * time_delta_hours)

    # Combine: weighted average (time=0.4, relevance=0.6)
    return round(min(1.0, time_score * 0.4 + relevance_score * 0.6), 3)


class SwingNewsService:
    """Given a PriceSwing, find causal news and store as swing_news memory."""

    def __init__(
        self,
        search_service: "TavilySearchService",
        memory_store: "SessionMemoryStore",
        index: "EventArbIndex",
        budget_manager: "TavilyBudgetManager",
        article_analyzer: Optional["ArticleAnalyzer"] = None,
    ):
        self._search = search_service
        self._memory = memory_store
        self._index = index
        self._budget = budget_manager
        self._analyzer = article_analyzer

        # Stats
        self._stats = {
            "swings_processed": 0,
            "articles_stored": 0,
            "associations_created": 0,
            "errors": 0,
        }

    async def process_swing(self, swing: "PriceSwing") -> int:
        """Search for causal news for a price swing and store results.

        Returns number of articles stored.
        """
        if self._budget.should_fallback():
            logger.debug("[SWING_NEWS] Budget exhausted, skipping swing search")
            return 0

        event = self._index.events.get(swing.event_ticker)
        if not event:
            return 0

        # Build query from event title + understanding search terms
        query = event.title
        if event.understanding and isinstance(event.understanding, dict):
            search_terms = event.understanding.get("search_terms", [])
            if search_terms:
                query += " " + " ".join(search_terms[:2])
        # Add swing direction hint
        query += f" {swing.direction} price move"

        # Determine time windows based on swing source
        if swing.source == "live":
            window_before = 6.0
            window_after = 1.0
        else:
            window_before = 12.0
            window_after = 2.0

        # Search using Tavily's date-range params
        try:
            results = await self._search.search_around_swing(
                query=query,
                swing_ts=swing.swing_end_ts,
                window_hours_before=window_before,
                window_hours_after=window_after,
                event_ticker=swing.event_ticker,
                search_depth="advanced",
                include_raw_content=True,
            )
        except Exception as e:
            logger.warning(f"[SWING_NEWS] Search failed for {swing.event_ticker}: {e}")
            self._stats["errors"] += 1
            return 0

        if not results:
            self._stats["swings_processed"] += 1
            return 0

        # Process and store each article
        articles_stored = 0
        for result in results[:3]:
            try:
                stored = await self._store_swing_article(swing, result, event)
                if stored:
                    articles_stored += 1
            except Exception as e:
                logger.debug(f"[SWING_NEWS] Article store error: {e}")

        self._stats["swings_processed"] += 1
        self._stats["articles_stored"] += articles_stored

        if articles_stored:
            logger.info(
                f"[SWING_NEWS] {swing.event_ticker}: {swing.direction} {swing.change_cents:.0f}c swing → "
                f"{articles_stored} articles stored"
            )

        return articles_stored

    async def _store_swing_article(
        self,
        swing: "PriceSwing",
        result: Dict[str, Any],
        event: Any,
    ) -> bool:
        """Store a single article as swing_news memory + DB association."""
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        raw_content = result.get("raw_content", "")
        published_date = result.get("published_date", "")
        score = result.get("score", 0.5)

        if not content and not raw_content:
            return False

        # Calculate time delta for causal confidence
        time_delta_hours = 6.0  # default
        if published_date:
            try:
                pub_dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                swing_dt = datetime.fromtimestamp(swing.swing_end_ts, tz=timezone.utc)
                time_delta_hours = abs((swing_dt - pub_dt).total_seconds()) / 3600.0
            except (ValueError, TypeError):
                pass

        confidence = _causal_confidence(time_delta_hours, score)
        sig_quality = _signal_quality(swing.change_cents)

        # Build memory content
        memory_content = (
            f"SWING_NEWS: {title}\n"
            f"Source: {result.get('source', 'tavily')} | {published_date}\n"
            f"URL: {url}\n"
            f"Impact: {swing.direction} {swing.change_cents:.0f}c on {swing.market_ticker}\n\n"
            f"{raw_content[:1500] if raw_content else content[:500]}"
        )

        # Build price snapshot from current index state
        price_snapshot = None
        market = event.markets.get(swing.market_ticker)
        if market and market.yes_mid is not None:
            price_snapshot = {
                swing.market_ticker: {
                    "yes_bid": market.yes_bid,
                    "yes_ask": market.yes_ask,
                    "yes_mid": market.yes_mid,
                    "spread": market.spread,
                },
                "_ts": time.time(),
            }

        metadata = {
            "event_ticker": swing.event_ticker,
            "news_url": url,
            "news_title": title,
            "news_published_at": published_date or None,
            "price_impact_cents": swing.change_cents,
            "price_impact_direction": swing.direction,
            "event_category": event.category,
            "causal_confidence": confidence,
            "price_snapshot": price_snapshot,
            "swing_source": swing.source,
            "signal_quality": sig_quality,
        }

        # Run article analysis if we have raw content and an analyzer
        analysis_dict = None
        if self._analyzer and raw_content:
            try:
                analysis = await self._analyzer.analyze(
                    article_content=raw_content,
                    article_title=title,
                    event_title=event.title,
                )
                if analysis:
                    import dataclasses
                    analysis_dict = dataclasses.asdict(analysis)
                    metadata["article_analysis"] = analysis_dict
            except Exception:
                pass

        # Store to session memory as type="swing_news"
        await self._memory.store(
            content=memory_content,
            memory_type="swing_news",
            metadata=metadata,
        )

        # Store to swing_news_associations table (fire-and-forget)
        asyncio.create_task(self._insert_association(swing, title, url, published_date, confidence, analysis_dict))

        return True

    async def _insert_association(
        self,
        swing: "PriceSwing",
        news_title: str,
        news_url: str,
        published_date: str,
        confidence: float,
        analysis: Optional[Dict],
    ) -> None:
        """Insert a swing_news_association row."""
        try:
            from kalshiflow_rl.data.database import rl_db
            pool = await rl_db.get_pool()
        except Exception as e:
            logger.debug(f"[SWING_NEWS] DB not available for association: {e}")
            return

        try:
            # Parse published_date to timestamptz
            pub_ts = None
            if published_date:
                try:
                    pub_ts = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            swing_start = datetime.fromtimestamp(swing.swing_start_ts, tz=timezone.utc)
            swing_end = datetime.fromtimestamp(swing.swing_end_ts, tz=timezone.utc)

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO swing_news_associations (
                        event_ticker, market_ticker, direction, change_cents,
                        price_before, price_after, swing_start_ts, swing_end_ts,
                        volume_during, source,
                        news_title, news_url, news_published_at,
                        causal_confidence, article_analysis
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15::jsonb)
                    """,
                    swing.event_ticker,
                    swing.market_ticker,
                    swing.direction,
                    swing.change_cents,
                    swing.price_before,
                    swing.price_after,
                    swing_start,
                    swing_end,
                    swing.volume_during,
                    swing.source,
                    news_title,
                    news_url,
                    pub_ts,
                    confidence,
                    json.dumps(analysis) if analysis else None,
                )
            self._stats["associations_created"] += 1
        except Exception as e:
            logger.debug(f"[SWING_NEWS] Association insert failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)


class SwingNewsLoop:
    """Background service that processes unsearched swings and refreshes candlesticks.

    Main loop:
        1. At startup: scan all historical candlesticks for swings
        2. Process unsearched swings (max N per iteration, budget-aware)
        3. Refresh candlesticks hourly (detects new swings from fresh candle data)
        4. Sleep: 10s if swings queued, 60s otherwise
    """

    def __init__(
        self,
        swing_detector: "SwingDetector",
        swing_news_service: SwingNewsService,
        index: "EventArbIndex",
        config: Any,
        candle_fetch_callback=None,
    ):
        self._detector = swing_detector
        self._service = swing_news_service
        self._index = index
        self._config = config
        self._candle_fetch_callback = candle_fetch_callback

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_candle_refresh: float = 0.0

        # Config
        self._max_searches = getattr(config, "swing_max_searches_per_loop", 3)
        self._candle_refresh_interval = getattr(config, "swing_candle_refresh_seconds", 3600.0)

    async def start(self) -> None:
        """Start the background loop."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[SWING_NEWS_LOOP] Started")

    async def stop(self) -> None:
        """Stop the background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[SWING_NEWS_LOOP] Stopped")

    async def _run_loop(self) -> None:
        """Main background loop."""
        # Initial candlestick scan
        try:
            swings = self._detector.scan_all_events(self._index.events)
            if swings:
                logger.info(f"[SWING_NEWS_LOOP] Initial scan found {len(swings)} historical swings")
            self._last_candle_refresh = time.time()
        except Exception as e:
            logger.warning(f"[SWING_NEWS_LOOP] Initial scan error: {e}")

        while self._running:
            try:
                # Process unsearched swings
                processed = await self._process_pending_swings()

                # Refresh candlesticks if interval elapsed
                if time.time() - self._last_candle_refresh >= self._candle_refresh_interval:
                    await self._refresh_candlesticks()
                    self._last_candle_refresh = time.time()

                # Adaptive sleep
                unsearched = self._detector.get_unsearched_swings(limit=1)
                sleep_time = 10.0 if unsearched else 60.0
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[SWING_NEWS_LOOP] Loop error: {e}")
                await asyncio.sleep(30.0)

    async def _process_pending_swings(self) -> int:
        """Process up to max_searches unsearched swings."""
        swings = self._detector.get_unsearched_swings(limit=self._max_searches)
        if not swings:
            return 0

        processed = 0
        for swing in swings:
            try:
                articles = await self._service.process_swing(swing)
                self._detector.mark_searched(swing, news_found=articles > 0)
                processed += 1
            except Exception as e:
                logger.debug(f"[SWING_NEWS_LOOP] Swing processing error: {e}")
                self._detector.mark_searched(swing, news_found=False)

        if processed:
            logger.info(f"[SWING_NEWS_LOOP] Processed {processed} swings")
        return processed

    async def _refresh_candlesticks(self) -> None:
        """Refresh candlestick data and scan for new swings."""
        if self._candle_fetch_callback:
            try:
                await self._candle_fetch_callback()
            except Exception as e:
                logger.warning(f"[SWING_NEWS_LOOP] Candlestick refresh failed: {e}")
                return

        # Scan for new swings from refreshed data
        try:
            new_swings = self._detector.scan_all_events(self._index.events)
            if new_swings:
                logger.info(f"[SWING_NEWS_LOOP] Candle refresh found {len(new_swings)} new swings")
        except Exception as e:
            logger.warning(f"[SWING_NEWS_LOOP] Candle scan error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "last_candle_refresh": self._last_candle_refresh,
            "detector_stats": self._detector.stats,
            "service_stats": self._service.get_stats(),
        }
