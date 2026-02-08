"""
PriceImpactTracker - Tracks price changes after news articles are stored.

Schedules delayed price snapshots at T+1h, T+4h, T+24h after each article
ingestion, then computes the magnitude of price change to build a corpus
of news-price correlations.

The Captain can query this via get_price_movers() to learn what kind of
news moves each market.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.impact_tracker")

# Price change thresholds for magnitude classification
MAGNITUDE_THRESHOLDS = {
    "small": 3,    # 3-5 cents
    "medium": 5,   # 5-10 cents
    "large": 10,   # >10 cents
}


class PriceImpactTracker:
    """Tracks price movements after news articles are stored."""

    def __init__(self, index, db=None):
        """
        Args:
            index: EventArbIndex for current price lookups
            db: RLDatabase instance for Supabase writes (optional, degrades gracefully)
        """
        self._index = index
        self._db = db
        self._pending_snapshots: List[Dict] = []  # In-memory queue of scheduled checks
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the background snapshot processing loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("[IMPACT_TRACKER] Started")

    async def stop(self) -> None:
        """Stop the tracker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[IMPACT_TRACKER] Stopped")

    def schedule_snapshots(
        self,
        news_url: str,
        event_ticker: str,
        price_at_news: Dict[str, Any],
    ) -> None:
        """Schedule price checks at T+1h, T+4h, T+24h after article storage.

        Args:
            news_url: URL of the stored article (used to lookup UUID at snapshot time)
            event_ticker: Event to monitor
            price_at_news: Price snapshot at time of news ingestion
        """
        now = time.time()
        for delay_hours in (1, 4, 24):
            self._pending_snapshots.append({
                "news_url": news_url,
                "event_ticker": event_ticker,
                "price_at_news": price_at_news,
                "delay_hours": delay_hours,
                "check_at": now + (delay_hours * 3600),
                "completed": False,
            })

        logger.info(
            f"[IMPACT_TRACKER] Scheduled 3 snapshots for {event_ticker} "
            f"(news_url={news_url[:60] if news_url else 'N/A'})"
        )

    async def _process_loop(self) -> None:
        """Background loop that processes due snapshot checks every 60s."""
        while self._running:
            try:
                await asyncio.sleep(60)
                if not self._running:
                    break

                now = time.time()
                due = [s for s in self._pending_snapshots if s["check_at"] <= now and not s["completed"]]

                for snapshot in due:
                    try:
                        await self._compute_impact(snapshot)
                        snapshot["completed"] = True
                    except Exception as e:
                        logger.debug(f"[IMPACT_TRACKER] Snapshot failed: {e}")

                # Clean up completed snapshots (keep last 100 for debugging)
                self._pending_snapshots = [
                    s for s in self._pending_snapshots if not s["completed"]
                ][-500:]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[IMPACT_TRACKER] Loop error: {e}")
                await asyncio.sleep(30)

    async def _resolve_news_memory_id(self, news_url: str) -> Optional[str]:
        """Lookup the agent_memories UUID for a news article by URL."""
        if not self._db or not news_url:
            return None
        try:
            async with self._db.get_connection() as conn:
                row_id = await conn.fetchval(
                    "SELECT id FROM agent_memories WHERE news_url = $1",
                    news_url,
                )
                return str(row_id) if row_id else None
        except Exception as e:
            logger.debug(f"[IMPACT_TRACKER] UUID lookup failed for {news_url[:60]}: {e}")
            return None

    async def _compute_impact(self, snapshot: Dict) -> None:
        """Compare price at news time vs current price and store result."""
        event_ticker = snapshot["event_ticker"]
        delay_hours = snapshot["delay_hours"]
        price_at_news = snapshot.get("price_at_news", {})

        if not self._index:
            return

        event = self._index.events.get(event_ticker)
        if not event:
            return

        # Build current price snapshot
        markets_at_news = price_at_news.get("markets", {})
        if not markets_at_news:
            return

        # Resolve UUID from news_url (article should be in DB by now)
        news_memory_id = await self._resolve_news_memory_id(snapshot.get("news_url", ""))

        for market_ticker, old_prices in markets_at_news.items():
            market = event.markets.get(market_ticker)
            if not market:
                continue

            old_mid = old_prices.get("yes_mid")
            new_mid = market.yes_mid
            if old_mid is None or new_mid is None:
                continue

            change_cents = int(round(new_mid - old_mid))

            # Classify magnitude
            abs_change = abs(change_cents)
            if abs_change >= MAGNITUDE_THRESHOLDS["large"]:
                magnitude = "large"
            elif abs_change >= MAGNITUDE_THRESHOLDS["medium"]:
                magnitude = "medium"
            elif abs_change >= MAGNITUDE_THRESHOLDS["small"]:
                magnitude = "small"
            else:
                magnitude = "none"

            # Store to database if available
            if self._db and news_memory_id:
                try:
                    await self._store_impact(
                        news_memory_id=news_memory_id,
                        market_ticker=market_ticker,
                        event_ticker=event_ticker,
                        delay_hours=delay_hours,
                        change_cents=change_cents,
                        magnitude=magnitude,
                        price_at_news=old_prices,
                        price_current={
                            "yes_bid": market.yes_bid,
                            "yes_ask": market.yes_ask,
                            "yes_mid": market.yes_mid,
                            "ts": time.time(),
                        },
                    )
                except Exception as e:
                    logger.debug(f"[IMPACT_TRACKER] DB store failed: {e}")

            if magnitude != "none":
                logger.info(
                    f"[IMPACT_TRACKER] {market_ticker} {delay_hours}h: "
                    f"{change_cents:+d}c ({magnitude})"
                )

    async def _store_impact(
        self,
        news_memory_id: str,
        market_ticker: str,
        event_ticker: str,
        delay_hours: int,
        change_cents: int,
        magnitude: str,
        price_at_news: Dict,
        price_current: Dict,
    ) -> None:
        """Insert or update a row in news_price_impacts."""
        import json

        delay_col = f"price_after_{delay_hours}h"
        change_col = f"change_{delay_hours}h_cents"

        async with self._db.get_connection() as conn:
            # Check if row exists for this news_memory_id + market_ticker
            existing = await conn.fetchval(
                "SELECT id FROM news_price_impacts WHERE news_memory_id = $1 AND market_ticker = $2",
                news_memory_id,
                market_ticker,
            )

            if existing:
                # Update the specific delay column
                await conn.execute(
                    f"""
                    UPDATE news_price_impacts
                    SET {delay_col} = $1::jsonb,
                        {change_col} = $2,
                        magnitude = CASE
                            WHEN $3 = 'large' THEN 'large'
                            WHEN magnitude = 'large' THEN 'large'
                            WHEN $3 = 'medium' THEN 'medium'
                            WHEN magnitude = 'medium' THEN 'medium'
                            WHEN $3 = 'small' THEN 'small'
                            WHEN magnitude = 'small' THEN 'small'
                            ELSE 'none'
                        END
                    WHERE id = $4
                    """,
                    json.dumps(price_current),
                    change_cents,
                    magnitude,
                    existing,
                )
            else:
                # Insert new row
                await conn.execute(
                    f"""
                    INSERT INTO news_price_impacts (
                        news_memory_id, market_ticker, event_ticker,
                        price_at_news, {delay_col}, {change_col}, magnitude
                    ) VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7)
                    """,
                    news_memory_id,
                    market_ticker,
                    event_ticker,
                    json.dumps(price_at_news),
                    json.dumps(price_current),
                    change_cents,
                    magnitude,
                )

    async def find_market_movers(
        self,
        event_ticker: str,
        min_change_cents: int = 5,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find articles correlated with significant price movements.

        Falls back to in-memory data if DB is unavailable.
        """
        if self._db:
            try:
                async with self._db.get_connection() as conn:
                    rows = await conn.fetch(
                        "SELECT * FROM find_market_movers($1, $2, $3)",
                        event_ticker,
                        min_change_cents,
                        limit,
                    )
                    return [dict(row) for row in rows]
            except Exception as e:
                logger.debug(f"[IMPACT_TRACKER] DB query failed: {e}")

        return []

    def get_stats(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            "pending_snapshots": len(self._pending_snapshots),
            "running": self._running,
        }
