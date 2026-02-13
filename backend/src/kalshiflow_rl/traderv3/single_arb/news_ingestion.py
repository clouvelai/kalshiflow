"""NewsIngestionService - Background news polling for all active events.

Key Responsibilities:
    - Periodically fetch news articles for all monitored events in the EventArbIndex
    - Adaptive poll intervals based on time-to-close (more frequent near settlement)
    - Deduplicate articles by URL and headline similarity
    - Optionally extract full article content and run structured analysis
    - Store articles and chunks into SessionMemoryStore (FAISS + pgvector)
    - Respect Tavily credit budget via TavilyBudgetManager

Architecture Position:
    Background service started by the SingleArbCoordinator alongside Captain,
    AttentionRouter, and Sniper. Feeds the memory layer with news context that
    Captain can recall via recall_memory tool.

Design Principles:
    - Async-first: non-blocking loop with proper CancelledError handling
    - Budget-aware: stops early when Tavily credits exhausted
    - Adaptive scheduling: polls more frequently as events approach settlement
    - Deduplication: URL-based + headline similarity (SequenceMatcher >= 0.8)
    - Fire-and-forget storage: memory writes should not block the poll loop
"""

from __future__ import annotations

import asyncio
import logging
import time
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from .index import EventArbIndex
    from .memory.session_store import SessionMemoryStore
    from .tavily_service import TavilySearchService
    from .tavily_budget import TavilyBudgetManager
    from .article_analyzer import ArticleAnalyzer

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.news_ingestion")

# Default configuration
_DEFAULT_CONFIG = {
    "enabled": True,
    "max_credits_per_cycle": 20,
    "extract_top_n": 2,
}

# Minimum loop sleep interval (seconds)
_MIN_LOOP_INTERVAL = 3600  # 1 hour (reduced; swing detection handles urgent news)


class NewsIngestionService:
    """Background service that periodically fetches news for all monitored events.

    Polls events at adaptive intervals based on time-to-close, deduplicates
    articles, optionally extracts full content and runs analysis, then stores
    results into the SessionMemoryStore for Captain recall.
    """

    def __init__(
        self,
        search_service: "TavilySearchService",
        memory_store: "SessionMemoryStore",
        index: "EventArbIndex",
        budget_manager: "TavilyBudgetManager",
        config: Optional[Dict[str, Any]] = None,
        article_analyzer: Optional["ArticleAnalyzer"] = None,
    ):
        self._search_service = search_service
        self._memory = memory_store
        self._index = index
        self._budget = budget_manager
        self._article_analyzer = article_analyzer

        # Merge user config over defaults
        self._config = {**_DEFAULT_CONFIG}
        if config:
            self._config.update(config)

        # Runtime state
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None

        # Per-event last-polled timestamps
        self._event_last_polled: Dict[str, float] = {}

        # In-memory URL dedup set
        self._seen_urls: Set[str] = set()

        # Headline dedup cache: list of (event_ticker, headline) tuples
        self._seen_headlines: List[tuple] = []

        # Telemetry
        self._stats: Dict[str, Any] = {
            "cycles": 0,
            "articles_ingested": 0,
            "events_polled": 0,
            "last_cycle_ts": None,
            "errors": 0,
        }

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the background polling loop."""
        if not self._config.get("enabled", True):
            logger.info("[NEWS_INGESTION] Disabled by config, not starting")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[NEWS_INGESTION] Started background news polling")

    async def stop(self) -> None:
        """Stop the background polling loop gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[NEWS_INGESTION] Stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Return a copy of ingestion statistics."""
        return dict(self._stats)

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    async def _run_loop(self) -> None:
        """Main background loop: poll all events, sleep, repeat."""
        logger.info("[NEWS_INGESTION] Poll loop started")

        while self._running:
            try:
                await self._poll_all_events()
                self._stats["cycles"] += 1
                await asyncio.sleep(_MIN_LOOP_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[NEWS_INGESTION] Unexpected error in poll loop")
                self._stats["errors"] += 1
                # Back off on error before retrying
                await asyncio.sleep(60)

        logger.info("[NEWS_INGESTION] Poll loop stopped")

    # ------------------------------------------------------------------ #
    #  Poll scheduling
    # ------------------------------------------------------------------ #

    def _get_poll_interval_seconds(self, time_to_close_hours: Optional[float]) -> float:
        """Determine poll interval based on how close the event is to settlement.

        Args:
            time_to_close_hours: Hours until event closes. None if unknown.

        Returns:
            Poll interval in seconds.
        """
        if time_to_close_hours is None:
            return 3600.0  # 1 hour default

        if time_to_close_hours < 2.0:
            return 600.0    # 10 minutes
        elif time_to_close_hours < 24.0:
            return 1800.0   # 30 minutes
        elif time_to_close_hours < 168.0:  # 7 days
            return 7200.0   # 2 hours
        else:
            return 21600.0  # 6 hours

    # ------------------------------------------------------------------ #
    #  Event polling
    # ------------------------------------------------------------------ #

    async def _poll_all_events(self) -> None:
        """Iterate over all monitored events, poll those that are due."""
        events = self._index.events
        if not events:
            return

        now = time.time()
        credits_used_this_cycle = 0
        max_credits = self._config.get("max_credits_per_cycle", 20)

        for event_ticker, event in events.items():
            # Budget gate: stop early if we have used too many credits this cycle
            if credits_used_this_cycle >= max_credits:
                logger.debug(
                    "[NEWS_INGESTION] Credit limit reached for this cycle "
                    f"({credits_used_this_cycle}/{max_credits})"
                )
                break

            # Budget gate: stop if global budget is exhausted
            if self._budget.should_fallback():
                logger.debug("[NEWS_INGESTION] Global budget exhausted, stopping cycle")
                break

            # Check if enough time has passed since last poll for this event
            last_polled = self._event_last_polled.get(event_ticker, 0.0)

            # Determine time_to_close from the event's understanding or market data
            time_to_close_hours = self._get_event_time_to_close(event)
            poll_interval = self._get_poll_interval_seconds(time_to_close_hours)

            if (now - last_polled) < poll_interval:
                continue

            # Poll this event
            try:
                articles_found = await self._process_event(event_ticker, event)
                credits_used_this_cycle += 1  # basic search = 1 credit (+ extract if used)
                if articles_found > 0:
                    logger.info(
                        f"[NEWS_INGESTION] {event_ticker}: ingested {articles_found} articles"
                    )
            except Exception:
                logger.exception(
                    f"[NEWS_INGESTION] Error processing event {event_ticker}"
                )

        self._stats["last_cycle_ts"] = now

    def _get_event_time_to_close(self, event: Any) -> Optional[float]:
        """Extract time-to-close in hours from an EventMeta.

        Checks understanding first (pre-computed), then falls back to
        parsing close_time from the earliest market.
        """
        # Check understanding (if populated by UnderstandingBuilder)
        if event.understanding and isinstance(event.understanding, dict):
            ttc = event.understanding.get("time_to_close_hours")
            if ttc is not None:
                return float(ttc)

        # Fallback: find earliest close_time across markets
        earliest_close = None
        for market in event.markets.values():
            ct = market.close_time
            if ct:
                try:
                    from datetime import datetime, timezone
                    # Handle ISO 8601 format
                    if ct.endswith("Z"):
                        ct = ct[:-1] + "+00:00"
                    close_dt = datetime.fromisoformat(ct)
                    if close_dt.tzinfo is None:
                        close_dt = close_dt.replace(tzinfo=timezone.utc)
                    if earliest_close is None or close_dt < earliest_close:
                        earliest_close = close_dt
                except (ValueError, TypeError):
                    continue

        if earliest_close is not None:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            delta_hours = (earliest_close - now).total_seconds() / 3600.0
            return max(0.0, delta_hours)

        return None

    # ------------------------------------------------------------------ #
    #  Per-event processing
    # ------------------------------------------------------------------ #

    async def _process_event(self, event_ticker: str, event: Any) -> int:
        """Fetch news for a single event, deduplicate, store to memory.

        Args:
            event_ticker: The event ticker string.
            event: The EventMeta object.

        Returns:
            Number of new articles ingested.
        """
        # Build search query from event title + understanding search terms
        query = self._build_search_query(event)
        if not query:
            return 0

        # Determine days_back based on time_to_close
        time_to_close = self._get_event_time_to_close(event)
        if time_to_close is not None and time_to_close < 24.0:
            days_back = 1
        elif time_to_close is not None and time_to_close < 168.0:
            days_back = 7
        else:
            days_back = 7

        # Search for news
        results = await self._search_service.search_for_event(
            query=query,
            days_back=days_back,
            max_results=5,
            event_ticker=event_ticker,
        )

        if not results:
            self._event_last_polled[event_ticker] = time.time()
            self._stats["events_polled"] += 1
            return 0

        # Filter: deduplicate by URL and headline similarity
        new_results = []
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")

            # URL dedup
            if url and url in self._seen_urls:
                continue

            # Headline similarity dedup
            if title and self._is_duplicate_headline(event_ticker, title):
                continue

            new_results.append(result)

        if not new_results:
            self._event_last_polled[event_ticker] = time.time()
            self._stats["events_polled"] += 1
            return 0

        # Optionally extract full content for top N articles
        extract_top_n = self._config.get("extract_top_n", 2)
        urls_to_extract = [
            r["url"] for r in new_results[:extract_top_n]
            if r.get("url")
        ]

        extracted_content: Dict[str, str] = {}
        if urls_to_extract:
            try:
                extracted = await self._search_service.extract_articles(
                    urls=urls_to_extract,
                    event_ticker=event_ticker,
                )
                for item in extracted:
                    eurl = item.get("url", "")
                    econtent = item.get("raw_content", "")
                    if eurl and econtent:
                        extracted_content[eurl] = econtent
            except Exception:
                logger.debug(
                    f"[NEWS_INGESTION] Extract failed for {event_ticker}, "
                    "continuing with search snippets"
                )

        # Store each new article
        # Articles from search_for_event() have NLP summaries (content) but no raw_content.
        # extracted_content dict has full page text for URLs where extraction succeeded.
        articles_ingested = 0
        for result in new_results:
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("content", "")  # NLP summary from basic search
            raw_content = extracted_content.get(url, "")  # Full text if extracted

            # Use extracted content if available, otherwise NLP summary from search
            store_content = raw_content if raw_content else content
            if not store_content:
                continue

            metadata = {
                "event_ticker": event_ticker,
                "news_url": url,
                "news_title": title,
                "published_date": result.get("published_date", ""),
                "source": result.get("source", "tavily"),
                "score": result.get("score", 0.0),
                "has_extracted_content": bool(raw_content),
            }

            # Optionally run article analysis (only on extracted full content)
            if self._article_analyzer and raw_content:
                try:
                    analysis = await self._article_analyzer.analyze(
                        article_content=raw_content,
                        article_title=title,
                        event_title=getattr(event, "title", event_ticker),
                    )
                    if analysis:
                        import dataclasses
                        metadata["article_analysis"] = dataclasses.asdict(analysis)
                        metadata["signal_quality"] = {
                            "high": 1.0, "medium": 0.8, "low": 0.6,
                        }.get(analysis.confidence, 0.5)
                except Exception:
                    logger.debug(
                        f"[NEWS_INGESTION] Article analysis failed for {url}"
                    )

            # Store to memory (FAISS + pgvector)
            try:
                # Use store_chunked for articles with extracted content (longer)
                if raw_content and len(raw_content) > 2000:
                    from .chunking import ChunkingPipeline
                    chunks = ChunkingPipeline.chunk_article(raw_content, metadata)
                    await self._memory.store_chunked(
                        content=f"[{event_ticker}] {title}: {content[:500]}",
                        memory_type="news",
                        metadata=metadata,
                        chunks=chunks,
                    )
                else:
                    # No extraction or short content: store NLP summary directly
                    await self._memory.store(
                        content=f"[{event_ticker}] {title}: {store_content[:1000]}",
                        memory_type="news",
                        metadata=metadata,
                    )

                # Track as seen
                if url:
                    self._seen_urls.add(url)
                if title:
                    self._seen_headlines.append((event_ticker, title))

                articles_ingested += 1
            except Exception:
                logger.debug(
                    f"[NEWS_INGESTION] Failed to store article for {event_ticker}: {url}"
                )

        self._stats["articles_ingested"] += articles_ingested
        self._stats["events_polled"] += 1
        self._event_last_polled[event_ticker] = time.time()

        return articles_ingested

    # ------------------------------------------------------------------ #
    #  Query building
    # ------------------------------------------------------------------ #

    def _build_search_query(self, event: Any) -> str:
        """Build a search query from event title and understanding data.

        Uses the event title as the base query. If structured understanding
        is available with search_terms, appends the most relevant terms.
        """
        title = getattr(event, "title", "")
        if not title:
            return ""

        # Check understanding for additional search context
        if event.understanding and isinstance(event.understanding, dict):
            # Look for search_terms or key_factors
            search_terms = event.understanding.get("search_terms", [])
            if search_terms and isinstance(search_terms, list):
                # Append top 2 search terms to title
                extra = " ".join(search_terms[:2])
                return f"{title} {extra}"

            # Fallback: use trading_summary keywords
            trading_summary = event.understanding.get("trading_summary", "")
            if trading_summary:
                # Just use title + first few words of summary
                words = trading_summary.split()[:5]
                return f"{title} {' '.join(words)}"

        return title

    # ------------------------------------------------------------------ #
    #  Deduplication
    # ------------------------------------------------------------------ #

    def _is_similar_headline(self, h1: str, h2: str, threshold: float = 0.8) -> bool:
        """Check if two headlines are similar using SequenceMatcher.

        Args:
            h1: First headline.
            h2: Second headline.
            threshold: Similarity threshold (0.0 to 1.0).

        Returns:
            True if headlines are similar above threshold.
        """
        return SequenceMatcher(None, h1.lower(), h2.lower()).ratio() >= threshold

    def _is_duplicate_headline(self, event_ticker: str, headline: str) -> bool:
        """Check if a headline is a duplicate of any previously seen headline.

        Only compares within the same event to avoid false positives
        across unrelated events.
        """
        for seen_ticker, seen_headline in self._seen_headlines:
            if seen_ticker == event_ticker:
                if self._is_similar_headline(headline, seen_headline):
                    return True
        return False
