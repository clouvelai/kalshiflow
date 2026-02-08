"""
TavilySearchService - Unified web search with DDG fallback.

Primary search via Tavily Python SDK (advanced depth by default),
falls back to DuckDuckGo on:
- API errors / rate limits (429)
- Budget exhaustion (soft limit)
- Missing API key

Returns normalized results: [{title, url, content, published_date, score, source}]
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .tavily_budget import TavilyBudgetManager

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.tavily_service")


class TavilySearchService:
    """Web search service with Tavily primary + DuckDuckGo fallback.

    Defaults to advanced depth for maximum result quality. Each advanced
    search costs 2 credits (vs 1 for basic). Advanced depth returns
    raw page content and relevance scores.
    """

    def __init__(
        self,
        api_key: str,
        budget_manager: TavilyBudgetManager,
        search_depth: str = "advanced",
        max_results: int = 10,
    ):
        self._api_key = api_key
        self._budget = budget_manager
        self._search_depth = search_depth
        self._max_results = max_results
        self._client = None  # Lazy init

    def _get_client(self):
        """Lazy-init Tavily client."""
        if self._client is None:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self._api_key)
            except ImportError:
                logger.warning("[TAVILY] tavily-python not installed, DDG-only mode")
        return self._client

    async def search(
        self,
        query: str,
        topic: str = "general",
        max_results: Optional[int] = None,
        time_range: Optional[str] = None,
        include_answer: bool = False,
        event_ticker: str = "",
    ) -> List[Dict[str, Any]]:
        """Primary search. Falls back to DDG on failure.

        Args:
            query: Search query
            topic: "general" or "news"
            max_results: Override default max_results (None = use service default)
            time_range: "day", "week", "month", or None for no filter
            include_answer: Include LLM-generated answer summary
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
        event_ticker: str = "",
    ) -> List[Dict[str, Any]]:
        """News-focused search (topic='news', include_answer=True).

        Uses Tavily's news topic for recency-ranked results.
        Always includes answer summary for quick context.
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
            event_ticker=event_ticker,
            search_type="news",
        )
        if results is not None:
            return results

        return await self._ddg_fallback(query, effective_max)

    async def _tavily_search(
        self,
        query: str,
        topic: str = "general",
        max_results: int = 10,
        time_range: Optional[str] = None,
        include_answer: bool = False,
        event_ticker: str = "",
        search_type: str = "general",
    ) -> Optional[List[Dict[str, Any]]]:
        """Internal Tavily call. Returns None on failure (triggers fallback).

        Advanced depth includes raw_content for deeper context extraction.
        """
        client = self._get_client()
        if not client:
            return None

        try:
            kwargs = {
                "query": query,
                "search_depth": self._search_depth,
                "topic": topic,
                "max_results": max_results,
                "include_answer": include_answer,
            }
            # time_range is a Tavily API param: "day", "week", "month", "year"
            if time_range:
                kwargs["time_range"] = time_range

            # Tavily SDK is sync, wrap in thread
            response = await asyncio.to_thread(client.search, **kwargs)

            # Record credits (basic=1, advanced=2)
            credits = 2 if self._search_depth == "advanced" else 1
            self._budget.record_search(credits, event_ticker, search_type)

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
                    "score": r.get("score", 0.0),
                    "published_date": r.get("published_date", ""),
                    "source": "tavily",
                })

            # Attach answer as metadata on first result (if present)
            if answer and normalized:
                normalized[0]["answer_summary"] = answer

            logger.info(
                f"[TAVILY] Search '{query[:60]}': {len(normalized)} results "
                f"(depth={self._search_depth}, topic={topic}, credits={credits}, "
                f"remaining={self._budget.credits_remaining()})"
            )
            return normalized

        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                logger.warning(f"[TAVILY] Rate limited: {e}")
            else:
                logger.warning(f"[TAVILY] Search failed: {e}")
            return None

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
