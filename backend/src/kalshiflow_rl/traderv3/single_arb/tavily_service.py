"""
TavilySearchService - Unified web search with DDG fallback.

Primary search via Tavily Python SDK (AsyncTavilyClient for native async),
falls back to DuckDuckGo on:
- API errors / rate limits (429)
- Budget exhaustion (soft limit)
- Missing API key

Also provides Tavily Extract API for full article content retrieval.

Returns normalized results: [{title, url, content, raw_content, published_date, score, source}]
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .tavily_budget import TavilyBudgetManager

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.tavily_service")


class TavilySearchService:
    """Web search service with Tavily primary + DuckDuckGo fallback.

    Two-tier search strategy:
    - Captain's search_news: depth="fast", no raw_content (1 credit, ~1-2s)
    - Background ingestion: depth="basic", no raw_content + separate extract (2 credits, async)

    All Tavily calls include include_usage=True for token consumption tracking.

    Uses AsyncTavilyClient for native async (no to_thread wrapper needed).
    """

    def __init__(
        self,
        api_key: str,
        budget_manager: TavilyBudgetManager,
        search_depth: str = "basic",
        max_results: int = 10,
    ):
        self._api_key = api_key
        self._budget = budget_manager
        self._search_depth = search_depth
        self._max_results = max_results
        self._async_client = None  # Lazy init AsyncTavilyClient
        self._sync_client = None  # Fallback sync client
        self._async_unavailable = False  # Set True on ImportError (stop retrying)
        self._sync_unavailable = False   # Set True on ImportError (stop retrying)

    def _get_async_client(self):
        """Lazy-init async Tavily client."""
        if self._async_unavailable:
            return None
        if self._async_client is None:
            try:
                from tavily import AsyncTavilyClient
                self._async_client = AsyncTavilyClient(api_key=self._api_key)
            except ImportError:
                self._async_unavailable = True
                logger.debug("[TAVILY] AsyncTavilyClient not available, falling back to sync")
        return self._async_client

    def _get_sync_client(self):
        """Lazy-init sync Tavily client (fallback if async unavailable)."""
        if self._sync_unavailable:
            return None
        if self._sync_client is None:
            try:
                from tavily import TavilyClient
                self._sync_client = TavilyClient(api_key=self._api_key)
            except ImportError:
                self._sync_unavailable = True
                logger.warning("[TAVILY] tavily-python not installed, DDG-only mode")
        return self._sync_client

    async def search(
        self,
        query: str,
        topic: str = "general",
        max_results: Optional[int] = None,
        time_range: Optional[str] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        event_ticker: str = "",
    ) -> List[Dict[str, Any]]:
        """Primary search. Falls back to DDG on failure.

        Args:
            query: Search query
            topic: "general" or "news"
            max_results: Override default max_results (None = use service default)
            time_range: "day", "week", "month", or None for no filter
            include_answer: Include LLM-generated answer summary
            include_raw_content: Include full page content in results
            event_ticker: For budget tracking
        """
        effective_max = max_results or self._max_results

        if self._budget.should_fallback():
            logger.debug("[TAVILY] Budget exhausted, using DDG fallback")
            return await self._ddg_fallback(query, effective_max)

        results = await self._tavily_search(
            query=query,
            topic=topic,
            max_results=effective_max,
            time_range=time_range,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
            event_ticker=event_ticker,
            search_type="general",
        )
        if results is not None:
            return results

        return await self._ddg_fallback(query, effective_max)

    async def search_news(
        self,
        query: str,
        max_results: Optional[int] = None,
        time_range: str = "week",
        include_raw_content: bool = False,
        event_ticker: str = "",
        search_depth: str = "fast",
    ) -> List[Dict[str, Any]]:
        """News-focused search (topic='news', include_answer=True).

        Uses Tavily's news topic for recency-ranked results.
        Always includes answer summary for quick context.

        Args:
            search_depth: Override search depth. "fast" (1 credit, ~1-2s) for
                Captain tool calls, "basic" for background ingestion.
        """
        effective_max = max_results or self._max_results

        if self._budget.should_fallback():
            logger.debug("[TAVILY] Budget exhausted, using DDG fallback for news")
            return await self._ddg_fallback(query, effective_max)

        results = await self._tavily_search(
            query=query,
            topic="news",
            max_results=effective_max,
            time_range=time_range,
            include_answer=True,
            include_raw_content=include_raw_content,
            event_ticker=event_ticker,
            search_type="news",
            search_depth_override=search_depth,
        )
        if results is not None:
            return results

        return await self._ddg_fallback(query, effective_max)

    async def search_for_event(
        self,
        query: str,
        days_back: int = 7,
        max_results: int = 5,
        event_ticker: str = "",
    ) -> List[Dict[str, Any]]:
        """Event-focused news search with date range for background ingestion.

        Uses search_depth="basic" with include_raw_content=False (1 credit).
        Full content is obtained separately via extract_articles() for top URLs.

        Args:
            query: Search query (event-specific)
            days_back: How many days back to search
            max_results: Max results
            event_ticker: For budget tracking
        """
        if self._budget.should_fallback():
            return await self._ddg_fallback(query, max_results)

        # Map days_back to Tavily time_range
        if days_back <= 1:
            time_range = "day"
        elif days_back <= 7:
            time_range = "week"
        else:
            time_range = "month"

        results = await self._tavily_search(
            query=query,
            topic="news",
            max_results=max_results,
            time_range=time_range,
            include_answer=False,
            include_raw_content=False,
            event_ticker=event_ticker,
            search_type="event_news",
            search_depth_override="basic",
        )
        if results is not None:
            return results

        return await self._ddg_fallback(query, max_results)

    async def extract_articles(
        self,
        urls: List[str],
        event_ticker: str = "",
    ) -> List[Dict[str, Any]]:
        """Extract full article content using Tavily Extract API.

        1 credit per 5 URLs. Returns list of {url, raw_content} dicts.

        Args:
            urls: List of article URLs to extract
            event_ticker: For budget tracking
        """
        if not urls:
            return []

        # Pre-flight budget check (1 credit per 5 URLs, minimum 1)
        credits_needed = max(1, (len(urls) + 4) // 5)
        if not self._budget.can_afford(credits_needed):
            logger.debug("[TAVILY] Cannot afford extract, skipping")
            return []

        # Try async client first
        async_client = self._get_async_client()
        if async_client:
            try:
                response = await asyncio.wait_for(
                    async_client.extract(
                        urls=urls,
                        include_usage=True,
                    ),
                    timeout=15.0,
                )
                self._budget.record_extract(credits_needed, event_ticker)

                # Record token usage if returned
                usage = response.get("usage", {})
                if usage:
                    tokens = usage.get("tokens", 0)
                    if tokens:
                        self._budget.record_usage(tokens)

                results = []
                for r in response.get("results", []):
                    results.append({
                        "url": r.get("url", ""),
                        "raw_content": r.get("raw_content", ""),
                    })

                logger.info(
                    f"[TAVILY] Extract: {len(results)}/{len(urls)} URLs extracted "
                    f"(credits={credits_needed}, tokens={usage.get('tokens', 'n/a')})"
                )
                return results
            except asyncio.TimeoutError:
                logger.warning(f"[TAVILY] Async extract timeout (>15s) for {len(urls)} URLs")
                # Fall through to sync client
            except Exception as e:
                logger.warning(f"[TAVILY] Extract failed: {e}")
                return []

        # Fallback to sync client
        sync_client = self._get_sync_client()
        if sync_client:
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        sync_client.extract,
                        urls=urls,
                        include_usage=True,
                    ),
                    timeout=20.0,
                )
                self._budget.record_extract(credits_needed, event_ticker)

                # Record token usage if returned
                usage = response.get("usage", {})
                if usage:
                    tokens = usage.get("tokens", 0)
                    if tokens:
                        self._budget.record_usage(tokens)

                results = []
                for r in response.get("results", []):
                    results.append({
                        "url": r.get("url", ""),
                        "raw_content": r.get("raw_content", ""),
                    })
                return results
            except asyncio.TimeoutError:
                logger.warning(f"[TAVILY] Sync extract timeout (>20s) for {len(urls)} URLs")
                return []
            except Exception as e:
                logger.warning(f"[TAVILY] Sync extract failed: {e}")
                return []

        return []

    async def search_around_swing(
        self,
        query: str,
        swing_ts: float,
        window_hours_before: float = 6.0,
        window_hours_after: float = 1.0,
        event_ticker: str = "",
        search_depth: str = "advanced",
        include_raw_content: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search for news around a specific timestamp (for swing-news correlation).

        Uses Tavily's start_date/end_date for precise filtering and topic="finance"
        for market-relevant results.

        Args:
            query: Search query
            swing_ts: Unix timestamp of the price swing
            window_hours_before: Hours before swing to search
            window_hours_after: Hours after swing to search
            event_ticker: For budget tracking
            search_depth: "basic" (1 credit) or "advanced" (2 credits)
            include_raw_content: Get full article markdown (eliminates separate extract)
        """
        from datetime import datetime, timezone, timedelta

        swing_dt = datetime.fromtimestamp(swing_ts, tz=timezone.utc)
        start_dt = swing_dt - timedelta(hours=window_hours_before)
        end_dt = swing_dt + timedelta(hours=window_hours_after)

        start_date = start_dt.strftime("%Y-%m-%d")
        end_date = end_dt.strftime("%Y-%m-%d")

        # Ensure start_date is at least 1 day before end_date (Tavily errors on same-day range)
        if start_date >= end_date:
            from datetime import timedelta as td
            end_date = (start_dt + td(days=1)).strftime("%Y-%m-%d")

        if self._budget.should_fallback():
            return await self._ddg_fallback(query, 5)

        results = await self._tavily_search_with_dates(
            query=query,
            topic="finance",
            max_results=5,
            start_date=start_date,
            end_date=end_date,
            include_answer="advanced",
            include_raw_content=include_raw_content,
            event_ticker=event_ticker,
            search_type="swing_news",
            search_depth_override=search_depth,
        )
        if results is not None:
            return results

        return await self._ddg_fallback(query, 5)

    async def _tavily_search(
        self,
        query: str,
        topic: str = "general",
        max_results: int = 10,
        time_range: Optional[str] = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
        event_ticker: str = "",
        search_type: str = "general",
        search_depth_override: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Internal Tavily call. Returns None on failure (triggers fallback).

        Tries AsyncTavilyClient first, falls back to sync TavilyClient.
        All calls include include_usage=True for token consumption tracking.

        Args:
            search_depth_override: Override the instance-level search_depth.
                "fast" (1 credit, ~1-2s), "basic" (1 credit), "advanced" (2 credits).
        """
        effective_depth = search_depth_override or self._search_depth
        kwargs = {
            "query": query,
            "search_depth": effective_depth,
            "topic": topic,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_usage": True,
        }
        if time_range:
            kwargs["time_range"] = time_range

        # Try async client first
        async_client = self._get_async_client()
        if async_client:
            try:
                response = await asyncio.wait_for(
                    async_client.search(**kwargs),
                    timeout=15.0,
                )
                return self._process_search_response(
                    response, query, topic, event_ticker, search_type,
                    effective_depth,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[TAVILY] Async search timeout (>15s) for query: {query[:60]}")
                # Fall through to sync client
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    logger.warning(f"[TAVILY] Rate limited (async): {e}")
                else:
                    logger.warning(f"[TAVILY] Async search failed: {e}")
                # Fall through to sync client

        # Fallback to sync client wrapped in thread
        sync_client = self._get_sync_client()
        if not sync_client:
            return None

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(sync_client.search, **kwargs),
                timeout=20.0,
            )
            return self._process_search_response(
                response, query, topic, event_ticker, search_type,
                effective_depth,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[TAVILY] Sync search timeout (>20s) for query: {query[:60]}")
            return None
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                logger.warning(f"[TAVILY] Rate limited: {e}")
            else:
                logger.warning(f"[TAVILY] Search failed: {e}")
            return None

    async def _tavily_search_with_dates(
        self,
        query: str,
        topic: str = "finance",
        max_results: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_answer: Any = False,
        include_raw_content: bool = True,
        event_ticker: str = "",
        search_type: str = "swing_news",
        search_depth_override: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Internal Tavily call with start_date/end_date support.

        Uses AsyncTavilyClient first, falls back to sync. Supports Tavily's
        date range params (YYYY-MM-DD) and include_answer="advanced".
        """
        effective_depth = search_depth_override or self._search_depth
        kwargs = {
            "query": query,
            "search_depth": effective_depth,
            "topic": topic,
            "max_results": max_results,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
            "include_usage": True,
        }
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        async_client = self._get_async_client()
        if async_client:
            try:
                response = await asyncio.wait_for(
                    async_client.search(**kwargs),
                    timeout=15.0,
                )
                return self._process_search_response(
                    response, query, topic, event_ticker, search_type,
                    effective_depth,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[TAVILY] Async swing search timeout (>15s) for query: {query[:60]}")
                # Fall through to sync client
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    logger.warning(f"[TAVILY] Rate limited (swing search): {e}")
                else:
                    logger.warning(f"[TAVILY] Swing search failed: {e}")

        sync_client = self._get_sync_client()
        if not sync_client:
            return None

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(sync_client.search, **kwargs),
                timeout=20.0,
            )
            return self._process_search_response(
                response, query, topic, event_ticker, search_type,
                effective_depth,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[TAVILY] Sync swing search timeout (>20s) for query: {query[:60]}")
            return None
        except Exception as e:
            logger.warning(f"[TAVILY] Sync swing search failed: {e}")
            return None

    def _process_search_response(
        self,
        response: Dict,
        query: str,
        topic: str,
        event_ticker: str,
        search_type: str,
        effective_depth: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Normalize Tavily search response into standard result format."""
        depth = effective_depth or self._search_depth
        # Record credits (advanced=2, basic/fast=1)
        credits = 2 if depth == "advanced" else 1
        self._budget.record_search(credits, event_ticker, search_type)

        # Record token usage from include_usage response
        usage = response.get("usage", {})
        if usage:
            tokens = usage.get("tokens", 0)
            if tokens:
                self._budget.record_usage(tokens)

        # Extract answer if present
        answer = response.get("answer", "")

        # Normalize results
        raw_results = response.get("results", [])
        normalized = []
        for r in raw_results:
            normalized.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "raw_content": r.get("raw_content", ""),
                "score": r.get("score", 0.0),
                "published_date": r.get("published_date", ""),
                "source": "tavily",
            })

        # Attach answer as metadata on first result (if present)
        if answer and normalized:
            normalized[0]["answer_summary"] = answer

        logger.info(
            f"[TAVILY] Search '{query[:60]}': {len(normalized)} results "
            f"(depth={depth}, topic={topic}, credits={credits}, "
            f"tokens={usage.get('tokens', 'n/a')}, "
            f"remaining={self._budget.credits_remaining()})"
        )
        return normalized

    async def _ddg_fallback(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """DuckDuckGo fallback, reusing existing pattern from mentions_context.py."""
        try:
            from duckduckgo_search import DDGS

            def _search():
                results = []
                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=max_results):
                        results.append({
                            "title": r.get("title", ""),
                            "url": r.get("href", ""),
                            "content": r.get("body", ""),
                            "raw_content": "",
                            "score": 0.0,
                            "published_date": "",
                            "source": "duckduckgo",
                        })
                return results

            results = await asyncio.to_thread(_search)
            logger.debug(
                f"[TAVILY] DDG fallback '{query[:50]}': {len(results)} results"
            )
            return results

        except ImportError:
            logger.warning("[TAVILY] duckduckgo_search not installed")
            return []
        except Exception as e:
            logger.warning(f"[TAVILY] DDG fallback failed: {e}")
            return []
