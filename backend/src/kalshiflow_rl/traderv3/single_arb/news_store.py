"""
NewsStore - Persists fetched news articles as embeddings with price snapshots.

Enables historical news search and news-price correlation by storing
every article fetched via Tavily into the DualMemoryStore with:
- Full article metadata (title, URL, source, published date)
- Price snapshot at time of storage (current BBO for all markets in event)
- Event ticker linkage for filtered retrieval

Deduplication via news_url unique index in Supabase.
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.news_store")


class NewsStore:
    """Persists news articles as embeddings with price context."""

    def __init__(self, memory_store, index, impact_tracker=None):
        """
        Args:
            memory_store: DualMemoryStore instance
            index: EventArbIndex for price snapshots
            impact_tracker: PriceImpactTracker for scheduling post-news snapshots
        """
        self._memory_store = memory_store
        self._index = index
        self._impact_tracker = impact_tracker

    def _build_price_snapshot(self, event_ticker: str) -> Dict[str, Any]:
        """Build price snapshot from current EventMeta market prices."""
        snapshot = {}
        if not self._index:
            return snapshot

        event = self._index.events.get(event_ticker)
        if not event:
            return snapshot

        snapshot["ts"] = time.time()
        snapshot["markets"] = {}
        for ticker, m in event.markets.items():
            if m.yes_bid is not None or m.yes_ask is not None:
                snapshot["markets"][ticker] = {
                    "yes_bid": m.yes_bid,
                    "yes_ask": m.yes_ask,
                    "yes_mid": m.yes_mid,
                    "spread": m.spread,
                }
        return snapshot

    async def persist_articles(
        self,
        articles: List[Dict],
        event_ticker: str,
    ) -> int:
        """Embed and store articles with price snapshots. Returns count stored.

        Skips duplicates (by URL). Each article is stored as a memory with
        type="news" and enriched with price_snapshot at time of ingestion.

        Args:
            articles: List of Tavily search result dicts
            event_ticker: The event these articles relate to

        Returns:
            Number of newly stored articles
        """
        if not articles or not self._memory_store:
            return 0

        price_snapshot = self._build_price_snapshot(event_ticker)
        stored = 0

        for article in articles:
            url = article.get("url", "")
            title = article.get("title", "")
            content = article.get("content", article.get("snippet", ""))
            source = article.get("source", "")
            published_date = article.get("published_date", article.get("publishedDate", ""))

            if not content or not url:
                continue

            # Build rich content for embedding (title + snippet for better semantic match)
            embed_content = f"{title}\n\n{content[:500]}" if title else content[:500]

            try:
                self._memory_store.append(
                    content=embed_content,
                    memory_type="news",
                    metadata={
                        "event_ticker": event_ticker,
                        "event_tickers": [event_ticker],
                        "news_url": url,
                        "news_title": title,
                        "news_source": source,
                        "news_published_at": published_date,
                        "price_snapshot": price_snapshot,
                    },
                )
                stored += 1

                # Schedule price impact snapshots (T+1h, T+4h, T+24h)
                if self._impact_tracker and url and price_snapshot:
                    try:
                        self._impact_tracker.schedule_snapshots(
                            news_url=url,
                            event_ticker=event_ticker,
                            price_at_news=price_snapshot,
                        )
                    except Exception as ie:
                        logger.debug(f"[NEWS_STORE] Impact scheduling failed: {ie}")
            except Exception as e:
                # Likely a duplicate URL (unique constraint) - that's fine
                if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                    continue
                logger.debug(f"[NEWS_STORE] Failed to store article: {e}")

        if stored > 0:
            logger.info(
                f"[NEWS_STORE] Stored {stored}/{len(articles)} articles for {event_ticker}"
            )
        return stored

    async def search_news(
        self,
        query: str,
        event_ticker: Optional[str] = None,
        hours: int = 168,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic search over persisted news articles.

        Args:
            query: Natural language search query
            event_ticker: Filter to specific event
            hours: How far back to search (default 168 = 7 days)
            limit: Max results

        Returns:
            List of dicts with content, title, url, source, price_snapshot, similarity
        """
        results = await self._memory_store.search(
            query=query,
            limit=limit,
            memory_types=["news"],
            event_ticker=event_ticker,
            min_recency_hours=float(hours),
        )

        # Enrich results with news-specific fields
        now = time.time()
        formatted = []
        for r in results:
            entry = {
                "content": r.get("content", ""),
                "title": r.get("news_title", ""),
                "url": r.get("news_url", ""),
                "source": r.get("news_source", ""),
                "similarity": round(r.get("similarity", 0.0), 3) if "similarity" in r else None,
                "price_snapshot": r.get("price_snapshot"),
            }
            # Compute age
            ts = r.get("created_at") or r.get("timestamp")
            if ts:
                if isinstance(ts, (int, float)):
                    entry["age_hours"] = round((now - ts) / 3600, 1)
                else:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                        entry["age_hours"] = round((now - dt.timestamp()) / 3600, 1)
                    except (ValueError, TypeError):
                        pass
            formatted.append(entry)

        return formatted
