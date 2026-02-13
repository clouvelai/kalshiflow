"""Unit tests for TavilySearchService timeouts.

Verifies:
- Async client search times out after 15s and falls through to sync
- Sync client search times out after 20s
- extract_articles has timeout on both async and sync paths
- search_with_dates (swing search) has timeout
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kalshiflow_rl.traderv3.single_arb.tavily_service import TavilySearchService


def _make_service(
    async_client=None,
    sync_client=None,
    async_unavailable=False,
    sync_unavailable=False,
):
    """Create a TavilySearchService with injectable clients."""
    budget = MagicMock()
    budget.should_fallback.return_value = False
    budget.can_afford.return_value = True
    budget.record_search = MagicMock()
    budget.record_extract = MagicMock()
    budget.record_usage = MagicMock()
    budget.credits_remaining.return_value = 100

    service = TavilySearchService(
        api_key="test-key",
        budget_manager=budget,
    )

    # Inject clients directly (bypass lazy init)
    service._async_client = async_client
    service._sync_client = sync_client
    service._async_unavailable = async_unavailable
    service._sync_unavailable = sync_unavailable

    return service


def _make_search_response():
    """Standard Tavily search response."""
    return {
        "results": [
            {"title": "Test", "url": "https://example.com", "content": "body", "score": 0.9}
        ],
        "usage": {"tokens": 100},
    }


def _make_extract_response():
    """Standard Tavily extract response."""
    return {
        "results": [
            {"url": "https://example.com/article", "raw_content": "Full article text..."}
        ],
        "usage": {"tokens": 500},
    }


# ============================================================================
# _tavily_search timeouts
# ============================================================================


class TestTavilySearchTimeout:
    """Async and sync search paths should have explicit timeouts."""

    @pytest.mark.asyncio
    async def test_async_search_timeout_falls_through_to_sync(self):
        """When async client hangs, should timeout and fall through to sync."""
        async_client = MagicMock()

        async def hang_forever(**kwargs):
            await asyncio.sleep(999)

        async_client.search = hang_forever

        sync_client = MagicMock()
        sync_client.search = MagicMock(return_value=_make_search_response())

        service = _make_service(async_client=async_client, sync_client=sync_client)

        # Patch wait_for on the async path to simulate timeout
        original_wait_for = asyncio.wait_for

        call_count = 0

        async def patched_wait_for(coro, *, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and timeout == 15.0:
                # First call is the async search — simulate timeout
                # Must still consume the coroutine
                coro.close()
                raise asyncio.TimeoutError()
            return await original_wait_for(coro, timeout=timeout)

        with patch("kalshiflow_rl.traderv3.single_arb.tavily_service.asyncio.wait_for", patched_wait_for):
            results = await service._tavily_search(
                query="test query", event_ticker="EVT-1",
            )

        # Should have fallen through to sync client
        assert results is not None
        assert len(results) == 1
        assert results[0]["title"] == "Test"

    @pytest.mark.asyncio
    async def test_async_search_timeout_returns_none_if_no_sync(self):
        """When async client times out and no sync client, returns None."""
        async_client = MagicMock()

        async def hang_forever(**kwargs):
            await asyncio.sleep(999)

        async_client.search = hang_forever

        service = _make_service(async_client=async_client, sync_unavailable=True)

        original_wait_for = asyncio.wait_for

        async def patched_wait_for(coro, *, timeout):
            if timeout == 15.0:
                coro.close()
                raise asyncio.TimeoutError()
            return await original_wait_for(coro, timeout=timeout)

        with patch("kalshiflow_rl.traderv3.single_arb.tavily_service.asyncio.wait_for", patched_wait_for):
            results = await service._tavily_search(
                query="test query", event_ticker="EVT-1",
            )

        assert results is None

    @pytest.mark.asyncio
    async def test_sync_search_timeout_returns_none(self):
        """When sync client wrapped in to_thread times out, returns None."""
        sync_client = MagicMock()

        # Sync client will block in thread
        def block_forever(**kwargs):
            import time
            time.sleep(999)

        sync_client.search = block_forever

        service = _make_service(async_unavailable=True, sync_client=sync_client)

        original_wait_for = asyncio.wait_for

        async def patched_wait_for(coro, *, timeout):
            if timeout == 20.0:
                # Cancel the thread-wrapped coro
                coro.close()
                raise asyncio.TimeoutError()
            return await original_wait_for(coro, timeout=timeout)

        with patch("kalshiflow_rl.traderv3.single_arb.tavily_service.asyncio.wait_for", patched_wait_for):
            results = await service._tavily_search(
                query="test query", event_ticker="EVT-1",
            )

        assert results is None


# ============================================================================
# extract_articles timeouts
# ============================================================================


class TestExtractArticlesTimeout:
    """extract_articles should timeout on both async and sync paths."""

    @pytest.mark.asyncio
    async def test_async_extract_timeout_falls_through_to_sync(self):
        """When async extract hangs, should fall through to sync client."""
        async_client = MagicMock()

        async def hang_forever(**kwargs):
            await asyncio.sleep(999)

        async_client.extract = hang_forever

        sync_client = MagicMock()
        sync_client.extract = MagicMock(return_value=_make_extract_response())

        service = _make_service(async_client=async_client, sync_client=sync_client)

        original_wait_for = asyncio.wait_for
        call_count = 0

        async def patched_wait_for(coro, *, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and timeout == 15.0:
                coro.close()
                raise asyncio.TimeoutError()
            return await original_wait_for(coro, timeout=timeout)

        with patch("kalshiflow_rl.traderv3.single_arb.tavily_service.asyncio.wait_for", patched_wait_for):
            results = await service.extract_articles(
                urls=["https://example.com/article"],
                event_ticker="EVT-1",
            )

        assert len(results) == 1
        assert results[0]["url"] == "https://example.com/article"

    @pytest.mark.asyncio
    async def test_sync_extract_timeout_returns_empty(self):
        """When sync extract times out, returns empty list."""
        sync_client = MagicMock()

        def block_forever(**kwargs):
            import time
            time.sleep(999)

        sync_client.extract = block_forever

        service = _make_service(async_unavailable=True, sync_client=sync_client)

        original_wait_for = asyncio.wait_for

        async def patched_wait_for(coro, *, timeout):
            if timeout == 20.0:
                coro.close()
                raise asyncio.TimeoutError()
            return await original_wait_for(coro, timeout=timeout)

        with patch("kalshiflow_rl.traderv3.single_arb.tavily_service.asyncio.wait_for", patched_wait_for):
            results = await service.extract_articles(
                urls=["https://example.com/article"],
                event_ticker="EVT-1",
            )

        assert results == []


# ============================================================================
# _tavily_search_with_dates timeouts
# ============================================================================


class TestSwingSearchTimeout:
    """_tavily_search_with_dates should have same timeout behavior."""

    @pytest.mark.asyncio
    async def test_async_swing_search_timeout_falls_through(self):
        """Async swing search timeout falls through to sync client."""
        async_client = MagicMock()

        async def hang_forever(**kwargs):
            await asyncio.sleep(999)

        async_client.search = hang_forever

        sync_client = MagicMock()
        sync_client.search = MagicMock(return_value=_make_search_response())

        service = _make_service(async_client=async_client, sync_client=sync_client)

        original_wait_for = asyncio.wait_for
        call_count = 0

        async def patched_wait_for(coro, *, timeout):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and timeout == 15.0:
                coro.close()
                raise asyncio.TimeoutError()
            return await original_wait_for(coro, timeout=timeout)

        with patch("kalshiflow_rl.traderv3.single_arb.tavily_service.asyncio.wait_for", patched_wait_for):
            results = await service._tavily_search_with_dates(
                query="market swing news",
                event_ticker="EVT-1",
            )

        assert results is not None
        assert len(results) == 1
