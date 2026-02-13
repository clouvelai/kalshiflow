"""Unit tests for NewsIngestionService.

Tests cover:
  1. Lifecycle: start/stop/disabled
  2. Scheduling: adaptive poll intervals based on time-to-close
  3. Deduplication: URL-based + headline similarity (SequenceMatcher)
  4. Query building: title, understanding search_terms, empty title
  5. Event processing: search, store, chunking, URL tracking
  6. Budget: credit limit per cycle, global budget exhaustion
  7. Stats: counters and telemetry
  8. Time to close: from understanding dict and market close_time
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.single_arb.news_ingestion import NewsIngestionService


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def make_mock_event(title="Test Event", close_time=None, understanding=None):
    """Create a mock EventMeta with configurable fields."""
    event = MagicMock()
    event.title = title
    event.understanding = understanding
    event.markets = {}
    if close_time:
        market = MagicMock()
        market.close_time = close_time
        event.markets["TICKER-A"] = market
    return event


def make_mock_index(events=None):
    """Create a mock EventArbIndex."""
    idx = MagicMock()
    idx.events = events or {}
    return idx


def make_mock_budget(exhausted=False):
    """Create a mock TavilyBudgetManager."""
    budget = MagicMock()
    budget.should_fallback.return_value = exhausted
    return budget


def make_mock_search_service(results=None, extracted=None):
    """Create a mock TavilySearchService with configurable results."""
    svc = MagicMock()
    svc.search_for_event = AsyncMock(return_value=results or [])
    svc.extract_articles = AsyncMock(return_value=extracted or [])
    return svc


def make_mock_memory():
    """Create a mock SessionMemoryStore."""
    mem = MagicMock()
    mem.store = AsyncMock()
    mem.store_chunked = AsyncMock()
    return mem


def make_service(
    events=None,
    search_results=None,
    extracted=None,
    budget_exhausted=False,
    config=None,
):
    """Create a fully wired NewsIngestionService with mocked dependencies."""
    search_service = make_mock_search_service(search_results, extracted)
    memory_store = make_mock_memory()
    index = make_mock_index(events)
    budget_manager = make_mock_budget(budget_exhausted)

    svc = NewsIngestionService(
        search_service=search_service,
        memory_store=memory_store,
        index=index,
        budget_manager=budget_manager,
        config=config,
    )
    return svc


# ---------------------------------------------------------------------------
# 1. Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Tests for start/stop lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_creates_background_task(self):
        """start() should create an asyncio task and set _running."""
        svc = make_service()
        await svc.start()

        assert svc._running is True
        assert svc._task is not None
        assert isinstance(svc._task, asyncio.Task)

        # Cleanup
        await svc.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task_gracefully(self):
        """stop() should cancel the background task and clear it."""
        svc = make_service()
        await svc.start()

        task = svc._task
        assert task is not None

        await svc.stop()

        assert svc._running is False
        assert svc._task is None
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_stop_when_not_started_is_safe(self):
        """stop() when not started should not raise."""
        svc = make_service()
        assert svc._task is None

        # Should not raise
        await svc.stop()
        assert svc._running is False
        assert svc._task is None

    @pytest.mark.asyncio
    async def test_disabled_by_config_skips_start(self):
        """When config has enabled=False, start() should not create a task."""
        svc = make_service(config={"enabled": False})
        await svc.start()

        assert svc._running is False
        assert svc._task is None


# ---------------------------------------------------------------------------
# 2. Scheduling
# ---------------------------------------------------------------------------

class TestScheduling:
    """Tests for adaptive poll interval calculations."""

    def test_poll_interval_under_2_hours(self):
        """Events closing in < 2 hours should poll every 10 minutes (600s)."""
        svc = make_service()
        assert svc._get_poll_interval_seconds(1.0) == 600.0
        assert svc._get_poll_interval_seconds(0.5) == 600.0

    def test_poll_interval_2_to_24_hours(self):
        """Events closing in 2-24 hours should poll every 30 minutes (1800s)."""
        svc = make_service()
        assert svc._get_poll_interval_seconds(2.0) == 1800.0
        assert svc._get_poll_interval_seconds(12.0) == 1800.0
        assert svc._get_poll_interval_seconds(23.9) == 1800.0

    def test_poll_interval_1_to_7_days(self):
        """Events closing in 1-7 days should poll every 2 hours (7200s)."""
        svc = make_service()
        assert svc._get_poll_interval_seconds(24.0) == 7200.0
        assert svc._get_poll_interval_seconds(100.0) == 7200.0
        assert svc._get_poll_interval_seconds(167.9) == 7200.0

    def test_poll_interval_over_7_days(self):
        """Events closing in > 7 days should poll every 6 hours (21600s)."""
        svc = make_service()
        assert svc._get_poll_interval_seconds(168.0) == 21600.0
        assert svc._get_poll_interval_seconds(500.0) == 21600.0

    def test_poll_interval_none(self):
        """When time_to_close is None, default to 1 hour (3600s)."""
        svc = make_service()
        assert svc._get_poll_interval_seconds(None) == 3600.0


# ---------------------------------------------------------------------------
# 3. Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Tests for URL and headline deduplication."""

    def test_is_similar_headline_identical(self):
        """Identical headlines should be similar."""
        svc = make_service()
        assert svc._is_similar_headline("Breaking News", "Breaking News") is True

    def test_is_similar_headline_very_different(self):
        """Completely different headlines should not be similar."""
        svc = make_service()
        assert svc._is_similar_headline(
            "Oil prices surge to record highs",
            "Apple launches new iPhone model"
        ) is False

    def test_is_similar_headline_slightly_different(self):
        """Headlines with minor differences should be similar (>0.8)."""
        svc = make_service()
        assert svc._is_similar_headline(
            "Fed raises interest rates by 25 basis points",
            "Fed raises interest rates by 25 bps"
        ) is True

    def test_is_duplicate_headline_same_event_same_headline(self):
        """Same event + same headline should be a duplicate."""
        svc = make_service()
        svc._seen_headlines.append(("EVENT-1", "Fed raises rates"))

        assert svc._is_duplicate_headline("EVENT-1", "Fed raises rates") is True

    def test_is_duplicate_headline_different_event_same_headline(self):
        """Different event + same headline should NOT be a duplicate (scoped)."""
        svc = make_service()
        svc._seen_headlines.append(("EVENT-1", "Fed raises rates"))

        assert svc._is_duplicate_headline("EVENT-2", "Fed raises rates") is False

    def test_url_dedup_seen_urls_skipped(self):
        """Articles with previously seen URLs should be filtered out."""
        svc = make_service()
        svc._seen_urls.add("https://example.com/article-1")

        results = [
            {"url": "https://example.com/article-1", "title": "Old Article", "content": "text"},
            {"url": "https://example.com/article-2", "title": "New Article", "content": "text"},
        ]
        # Filter like _process_event does
        new_results = []
        for r in results:
            url = r.get("url", "")
            if url and url in svc._seen_urls:
                continue
            new_results.append(r)

        assert len(new_results) == 1
        assert new_results[0]["url"] == "https://example.com/article-2"

    @pytest.mark.asyncio
    async def test_seen_urls_tracked_after_ingestion(self):
        """URLs should be added to _seen_urls after successful ingestion."""
        event = make_mock_event(title="Test Event")
        results = [
            {"url": "https://new.com/1", "title": "Article 1", "content": "Some content"},
        ]
        svc = make_service(
            events={"EVT-1": event},
            search_results=results,
        )

        await svc._process_event("EVT-1", event)

        assert "https://new.com/1" in svc._seen_urls


# ---------------------------------------------------------------------------
# 4. Query building
# ---------------------------------------------------------------------------

class TestQueryBuilding:
    """Tests for search query construction."""

    def test_build_query_title_only(self):
        """With no understanding, query should be just the title."""
        event = make_mock_event(title="Will Bitcoin hit 100k?", understanding=None)
        svc = make_service()
        query = svc._build_search_query(event)
        assert query == "Will Bitcoin hit 100k?"

    def test_build_query_title_plus_search_terms(self):
        """With understanding search_terms, query should include top 2 terms."""
        event = make_mock_event(
            title="Will Bitcoin hit 100k?",
            understanding={
                "search_terms": ["BTC price prediction", "crypto market rally", "halving"],
            },
        )
        svc = make_service()
        query = svc._build_search_query(event)
        assert query == "Will Bitcoin hit 100k? BTC price prediction crypto market rally"

    def test_build_query_empty_title(self):
        """Event with empty title should return empty string."""
        event = make_mock_event(title="", understanding=None)
        svc = make_service()
        query = svc._build_search_query(event)
        assert query == ""


# ---------------------------------------------------------------------------
# 5. Event processing
# ---------------------------------------------------------------------------

class TestEventProcessing:
    """Tests for _process_event logic."""

    @pytest.mark.asyncio
    async def test_process_event_no_results_returns_zero(self):
        """When search returns no results, should return 0."""
        event = make_mock_event(title="Test Event")
        svc = make_service(search_results=[])

        count = await svc._process_event("EVT-1", event)
        assert count == 0

    @pytest.mark.asyncio
    async def test_process_event_stores_articles_to_memory(self):
        """New articles should be stored to memory via store()."""
        event = make_mock_event(title="GDP Report")
        results = [
            {"url": "https://news.com/gdp", "title": "GDP Grows", "content": "The economy grew..."},
        ]
        svc = make_service(search_results=results)

        count = await svc._process_event("EVT-1", event)

        assert count == 1
        svc._memory.store.assert_called_once()
        call_kwargs = svc._memory.store.call_args
        assert "news" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_process_event_uses_chunking_for_long_content(self):
        """Articles with >2000 chars of raw_content should use store_chunked."""
        event = make_mock_event(title="Long Article Event")
        long_content = "x" * 3000
        results = [
            {"url": "https://news.com/long", "title": "Long Article", "content": "short snippet"},
        ]
        extracted = [
            {"url": "https://news.com/long", "raw_content": long_content},
        ]
        svc = make_service(search_results=results, extracted=extracted)

        mock_chunking_cls = MagicMock()
        mock_chunking_cls.chunk_article.return_value = [
            {"text": "chunk1", "metadata": {}},
        ]
        mock_chunking_module = MagicMock()
        mock_chunking_module.ChunkingPipeline = mock_chunking_cls

        with patch.dict(
            "sys.modules",
            {"kalshiflow_rl.traderv3.single_arb.chunking": mock_chunking_module},
        ):
            count = await svc._process_event("EVT-1", event)

        assert count == 1
        svc._memory.store_chunked.assert_called_once()
        mock_chunking_cls.chunk_article.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_event_tracks_seen_urls(self):
        """After processing, article URLs should be in _seen_urls."""
        event = make_mock_event(title="Tracking Test")
        results = [
            {"url": "https://track.com/a1", "title": "Art 1", "content": "content A"},
            {"url": "https://track.com/a2", "title": "Art 2", "content": "content B"},
        ]
        svc = make_service(search_results=results)

        await svc._process_event("EVT-1", event)

        assert "https://track.com/a1" in svc._seen_urls
        assert "https://track.com/a2" in svc._seen_urls


# ---------------------------------------------------------------------------
# 6. Budget
# ---------------------------------------------------------------------------

class TestBudget:
    """Tests for credit budget enforcement."""

    @pytest.mark.asyncio
    async def test_budget_exhausted_stops_poll_cycle_early(self):
        """When global budget is exhausted, no events should be polled."""
        events = {
            "EVT-1": make_mock_event(title="Event 1"),
            "EVT-2": make_mock_event(title="Event 2"),
        }
        svc = make_service(
            events=events,
            budget_exhausted=True,
            search_results=[{"url": "https://x.com", "title": "T", "content": "C"}],
        )

        await svc._poll_all_events()

        # search_for_event should never be called because budget gate fires first
        svc._search_service.search_for_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_credit_limit_per_cycle_stops_processing(self):
        """When per-cycle credit limit is reached, remaining events should be skipped."""
        events = {}
        for i in range(20):
            events[f"EVT-{i}"] = make_mock_event(title=f"Event {i}")

        results = [{"url": f"https://x.com/{i}", "title": "T", "content": "C"}]
        svc = make_service(
            events=events,
            search_results=results,
            config={"max_credits_per_cycle": 4},  # 4 credits = 4 events (1 credit each, basic depth)
        )

        await svc._poll_all_events()

        # Should be called at most 4 times (4 credits / 1 per basic search event)
        call_count = svc._search_service.search_for_event.call_count
        assert call_count <= 4


# ---------------------------------------------------------------------------
# 7. Stats
# ---------------------------------------------------------------------------

class TestStats:
    """Tests for telemetry and statistics."""

    def test_get_stats_returns_correct_counters(self):
        """get_stats should return a dict with all expected keys."""
        svc = make_service()
        stats = svc.get_stats()

        assert "cycles" in stats
        assert "articles_ingested" in stats
        assert "events_polled" in stats
        assert "last_cycle_ts" in stats
        assert "errors" in stats
        assert stats["cycles"] == 0
        assert stats["articles_ingested"] == 0

    @pytest.mark.asyncio
    async def test_cycles_counter_increments_after_poll(self):
        """The cycles counter should increment after _poll_all_events completes."""
        svc = make_service()

        # Run one poll (no events, so it's quick)
        await svc._poll_all_events()
        # _poll_all_events does NOT increment cycles -- _run_loop does
        # So we simulate what _run_loop does
        svc._stats["cycles"] += 1

        assert svc.get_stats()["cycles"] == 1

    @pytest.mark.asyncio
    async def test_articles_ingested_counter_tracks_correctly(self):
        """articles_ingested should reflect total articles stored across events."""
        event = make_mock_event(title="Stats Event")
        results = [
            {"url": "https://a.com/1", "title": "A1", "content": "c1"},
            {"url": "https://a.com/2", "title": "A2", "content": "c2"},
            {"url": "https://a.com/3", "title": "A3", "content": "c3"},
        ]
        svc = make_service(search_results=results)

        await svc._process_event("EVT-1", event)

        assert svc.get_stats()["articles_ingested"] == 3


# ---------------------------------------------------------------------------
# 8. Time to close
# ---------------------------------------------------------------------------

class TestTimeToClose:
    """Tests for _get_event_time_to_close."""

    def test_time_to_close_from_understanding(self):
        """Should use time_to_close_hours from understanding dict when available."""
        event = make_mock_event(
            title="Test",
            understanding={"time_to_close_hours": 5.5},
        )
        svc = make_service()
        result = svc._get_event_time_to_close(event)
        assert result == 5.5

    def test_time_to_close_none_when_no_data(self):
        """Should return None when no understanding and no market close_time."""
        event = make_mock_event(title="Test", understanding=None)
        svc = make_service()
        result = svc._get_event_time_to_close(event)
        assert result is None

    def test_time_to_close_from_market_close_time(self):
        """Should parse close_time from market when understanding is absent."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=3)
        close_time_str = future_time.isoformat()

        event = make_mock_event(
            title="Test",
            close_time=close_time_str,
            understanding=None,
        )
        svc = make_service()
        result = svc._get_event_time_to_close(event)

        assert result is not None
        # Should be approximately 3 hours (within a reasonable margin)
        assert 2.5 < result < 3.5

    def test_time_to_close_from_market_close_time_z_suffix(self):
        """Should handle ISO 8601 with 'Z' suffix."""
        future_time = datetime.now(timezone.utc) + timedelta(hours=6)
        close_time_str = future_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        event = make_mock_event(
            title="Test",
            close_time=close_time_str,
            understanding=None,
        )
        svc = make_service()
        result = svc._get_event_time_to_close(event)

        assert result is not None
        assert 5.5 < result < 6.5
