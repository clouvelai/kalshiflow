"""
EventCodex Service - Background data enrichment for all tracked events.

Syncs ALL event data from Kalshi and Polymarket APIs:
1. Kalshi event details + nested markets
2. Kalshi event candlesticks (1h, 1-min interval)
3. Polymarket event metadata (Gamma API)
4. Polymarket live volume (Data API)
5. Polymarket price history per paired market (CLOB API)

All in-memory. Full re-sync on restart. No DB.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kalshiflow_rl.traderv3.services.event_codex")


@dataclass
class CandlePoint:
    ts: int  # unix seconds (end of period)
    open: Optional[float] = None  # cents (Kalshi 0-99, Poly normalized 0-100)
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None


@dataclass
class MarketCandles:
    kalshi_ticker: str
    question: str
    pair_id: Optional[str] = None
    poly_token_id: Optional[str] = None
    kalshi: List[CandlePoint] = field(default_factory=list)
    poly: List[CandlePoint] = field(default_factory=list)
    fetched_at: float = 0.0


@dataclass
class CodexEntry:
    kalshi_event_ticker: str
    series_ticker: str
    title: str
    category: str

    # Kalshi event enrichment (GET /events/{ticker})
    kalshi_subtitle: Optional[str] = None
    kalshi_mutually_exclusive: bool = False
    kalshi_strike_date: Optional[str] = None
    kalshi_product_metadata: Optional[Dict] = None
    kalshi_markets: List[Dict] = field(default_factory=list)

    # Poly event enrichment (Gamma /events/{id})
    poly_event_id: Optional[str] = None
    poly_title: Optional[str] = None
    poly_description: Optional[str] = None
    poly_slug: Optional[str] = None
    poly_volume: Optional[float] = None
    poly_volume_24h: Optional[float] = None
    poly_liquidity: Optional[float] = None
    poly_live_volume: Optional[float] = None

    # Per-market candle data
    market_candles: List[MarketCandles] = field(default_factory=list)

    # Cache metadata
    metadata_fetched_at: float = 0.0
    candles_fetched_at: float = 0.0


def _codex_entry_to_dict(entry: CodexEntry) -> Dict[str, Any]:
    """Serialize CodexEntry to JSON-safe dict."""
    d = asdict(entry)
    return d


class EventCodexService:
    """
    Background data enrichment service. Syncs ALL event data from APIs.

    Single poll loop fetches for every event in PairRegistry:
    1. Kalshi event details + nested markets
    2. Kalshi event candlesticks (1h, 1-min interval)
    3. Polymarket event metadata (Gamma API)
    4. Polymarket live volume (Data API)
    5. Polymarket price history per paired market (CLOB API)

    All in-memory. Full re-sync on restart. No DB.
    """

    def __init__(
        self,
        pair_registry,
        trading_client,
        poly_client,
        event_bus,
        websocket_manager,
        config,
    ):
        self._pair_registry = pair_registry
        self._trading_client = trading_client
        self._poly_client = poly_client
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._config = config

        self._cache: Dict[str, CodexEntry] = {}
        self._poll_interval = config.event_codex_poll_interval
        self._candle_window_minutes = config.event_codex_candle_window

        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._sync_count = 0
        self._last_sync_at: Optional[float] = None
        self._last_sync_duration: float = 0.0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("EventCodexService started")

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        logger.info(f"EventCodexService stopped ({self._sync_count} syncs)")

    async def _poll_loop(self) -> None:
        # Initial delay to let pair index populate
        await asyncio.sleep(15.0)

        while self._running:
            try:
                await self._sync_all()
            except Exception as e:
                logger.error(f"EventCodex sync error: {e}")

            try:
                await asyncio.sleep(self._poll_interval)
            except asyncio.CancelledError:
                break

    async def _sync_all(self) -> None:
        """Iterate all events in registry, enrich each, broadcast snapshot."""
        start = time.time()
        events = self._pair_registry.get_events_grouped()
        if not events:
            return

        sem = asyncio.Semaphore(3)
        tasks = []

        for group in events:
            tasks.append(self._enrich_event_throttled(sem, group))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"EventCodex enrich error: {result}")
            elif result is not None:
                self._cache[result.kalshi_event_ticker] = result

        self._sync_count += 1
        self._last_sync_at = time.time()
        self._last_sync_duration = time.time() - start

        # Broadcast snapshot
        snapshot = self.get_full_snapshot()
        await self._websocket_manager.broadcast_message(
            "event_codex_snapshot", snapshot
        )

        logger.info(
            f"EventCodex sync #{self._sync_count}: "
            f"{len(self._cache)} entries, {self._last_sync_duration:.1f}s"
        )

    async def _enrich_event_throttled(self, sem, event_group) -> Optional[CodexEntry]:
        async with sem:
            return await self._enrich_event(event_group)

    async def _enrich_event(self, event_group) -> Optional[CodexEntry]:
        """Enrich a single event with all API data."""
        event_ticker = event_group.kalshi_event_ticker
        series_ticker = event_group.series_ticker
        poly_event_id = event_group.poly_event_id

        if not series_ticker:
            # Parse from event_ticker
            parts = event_ticker.rsplit("-", 1)
            if len(parts) == 2:
                series_ticker = parts[0]

        if not series_ticker:
            logger.warning(f"No series_ticker for {event_ticker}, skipping Kalshi candles")
            series_ticker = ""

        entry = CodexEntry(
            kalshi_event_ticker=event_ticker,
            series_ticker=series_ticker,
            title=event_group.title,
            category=event_group.category,
            poly_event_id=poly_event_id,
        )

        # Gather all API calls concurrently
        tasks = {
            "kalshi_event": self._fetch_kalshi_event(event_ticker),
            "kalshi_candles": self._fetch_kalshi_candles(series_ticker, event_ticker) if series_ticker else None,
            "poly_event": self._fetch_poly_event(poly_event_id) if poly_event_id else None,
            "poly_volume": self._fetch_poly_volume(poly_event_id) if poly_event_id else None,
        }

        # Only gather non-None tasks
        active_keys = [k for k, v in tasks.items() if v is not None]
        active_tasks = [tasks[k] for k in active_keys]

        results_list = await asyncio.gather(*active_tasks, return_exceptions=True)

        # Map results back to named dict
        raw_results = {}
        for key, result in zip(active_keys, results_list):
            if isinstance(result, Exception):
                logger.warning(f"EventCodex fetch '{key}' failed for {event_ticker}: {result}")
                raw_results[key] = {}
            else:
                raw_results[key] = result if result is not None else {}

        kalshi_event_data = raw_results.get("kalshi_event", {})
        kalshi_candles_data = raw_results.get("kalshi_candles", {})
        poly_event_data = raw_results.get("poly_event", {})
        poly_volume_data = raw_results.get("poly_volume", {})

        # Populate Kalshi event enrichment
        if isinstance(kalshi_event_data, dict) and kalshi_event_data:
            entry.kalshi_subtitle = kalshi_event_data.get("sub_title") or kalshi_event_data.get("subtitle")
            entry.kalshi_mutually_exclusive = kalshi_event_data.get("mutually_exclusive", False)
            entry.kalshi_strike_date = kalshi_event_data.get("strike_date")
            entry.kalshi_product_metadata = kalshi_event_data.get("product_metadata")
            markets = kalshi_event_data.get("markets", [])
            entry.kalshi_markets = [
                {
                    "ticker": m.get("ticker", ""),
                    "yes_sub_title": m.get("yes_sub_title") or m.get("title", ""),
                    "volume": m.get("volume", 0),
                    "volume_24h": m.get("volume_24h", 0),
                    "last_price": m.get("last_price"),
                    "yes_bid": m.get("yes_bid"),
                    "yes_ask": m.get("yes_ask"),
                }
                for m in markets
            ]
            entry.metadata_fetched_at = time.time()

        # Populate Poly event enrichment
        if isinstance(poly_event_data, dict) and poly_event_data:
            entry.poly_title = poly_event_data.get("title")
            entry.poly_description = poly_event_data.get("description")
            entry.poly_slug = poly_event_data.get("slug")
            entry.poly_volume = _safe_float(poly_event_data.get("volume"))
            entry.poly_volume_24h = _safe_float(poly_event_data.get("volume24hr"))
            entry.poly_liquidity = _safe_float(poly_event_data.get("liquidity"))

        # Populate Poly live volume
        if isinstance(poly_volume_data, (dict, list)):
            if isinstance(poly_volume_data, list) and poly_volume_data:
                vol_entry = poly_volume_data[0] if poly_volume_data else {}
            else:
                vol_entry = poly_volume_data
            if isinstance(vol_entry, dict):
                entry.poly_live_volume = _safe_float(vol_entry.get("volume"))

        # Build per-market candle data
        entry.market_candles = await self._build_market_candles(
            event_group, kalshi_candles_data, entry.kalshi_markets
        )
        entry.candles_fetched_at = time.time()

        return entry

    async def _build_market_candles(
        self, event_group, kalshi_candles_data, kalshi_markets
    ) -> List[MarketCandles]:
        """Build candle data per market from Kalshi event candles + Poly price history."""
        market_candles_list = []

        # Parse Kalshi event candles (one call gives all markets)
        kalshi_by_ticker: Dict[str, List[CandlePoint]] = {}
        if isinstance(kalshi_candles_data, dict):
            tickers = kalshi_candles_data.get("market_tickers", [])
            candle_arrays = kalshi_candles_data.get("market_candlesticks", [])
            for i, ticker in enumerate(tickers):
                if i < len(candle_arrays) and candle_arrays[i]:
                    kalshi_by_ticker[ticker] = [
                        CandlePoint(
                            ts=c.get("end_period_ts", 0),
                            open=_safe_float(c.get("price", {}).get("open")),
                            high=_safe_float(c.get("price", {}).get("high")),
                            low=_safe_float(c.get("price", {}).get("low")),
                            close=_safe_float(c.get("price", {}).get("close")),
                            volume=c.get("volume"),
                        )
                        for c in candle_arrays[i]
                    ]

        # Map kalshi markets to questions
        market_questions: Dict[str, str] = {}
        for m in kalshi_markets:
            market_questions[m["ticker"]] = m.get("yes_sub_title", "")

        # For each pair in the event group, fetch Poly price history
        pair_by_kalshi: Dict[str, Any] = {}
        for pair in event_group.pairs:
            pair_by_kalshi[pair.kalshi_ticker] = pair

        # Fetch Poly price history concurrently for all paired markets
        poly_tasks = {}
        for pair in event_group.pairs:
            if pair.poly_token_id_yes:
                poly_tasks[pair.kalshi_ticker] = self._fetch_poly_price_history(
                    pair.poly_token_id_yes
                )

        poly_results = {}
        if poly_tasks:
            keys = list(poly_tasks.keys())
            results = await asyncio.gather(*poly_tasks.values(), return_exceptions=True)
            for key, result in zip(keys, results):
                if isinstance(result, Exception):
                    logger.debug(f"Poly price history error for {key}: {result}")
                    poly_results[key] = []
                else:
                    poly_results[key] = result

        # Build MarketCandles for each Kalshi market (paired or not)
        all_tickers = set(kalshi_by_ticker.keys())
        for pair in event_group.pairs:
            all_tickers.add(pair.kalshi_ticker)

        if not all_tickers:
            logger.info(f"No market tickers to build candles for event {event_group.kalshi_event_ticker}")

        for ticker in sorted(all_tickers):
            pair = pair_by_kalshi.get(ticker)
            question = market_questions.get(ticker, "")
            if not question and pair:
                question = pair.question

            # Poly candles from price history
            poly_candles = []
            poly_history = poly_results.get(ticker, [])
            if isinstance(poly_history, list):
                for pt in poly_history:
                    if isinstance(pt, dict):
                        price_01 = _safe_float(pt.get("p"))
                        price_cents = price_01 * 100 if price_01 is not None else None
                        poly_candles.append(CandlePoint(
                            ts=int(pt.get("t", 0)),
                            close=price_cents,
                        ))

            mc = MarketCandles(
                kalshi_ticker=ticker,
                question=question,
                pair_id=pair.id if pair else None,
                poly_token_id=pair.poly_token_id_yes if pair else None,
                kalshi=kalshi_by_ticker.get(ticker, []),
                poly=poly_candles,
                fetched_at=time.time(),
            )
            market_candles_list.append(mc)

        return market_candles_list

    async def _fetch_kalshi_event(self, event_ticker: str) -> Dict:
        """GET /events/{ticker} via trading client."""
        if not self._trading_client:
            return {}
        try:
            return await self._trading_client.get_event(event_ticker)
        except Exception as e:
            logger.debug(f"Kalshi event fetch failed for {event_ticker}: {e}")
            return {}

    async def _fetch_kalshi_candles(self, series_ticker: str, event_ticker: str) -> Dict:
        """GET /series/{series}/events/{event}/candlesticks via trading client."""
        if not self._trading_client or not series_ticker:
            return {}
        try:
            now = int(time.time())
            start = now - (self._candle_window_minutes * 60)
            # Use hourly candles for windows >= 6h, 1-min for shorter
            period = 60 if self._candle_window_minutes >= 360 else 1
            result = await self._trading_client.get_event_candlesticks(
                series_ticker=series_ticker,
                event_ticker=event_ticker,
                start_ts=start,
                end_ts=now,
                period_interval=period,
            )
            tickers = result.get("market_tickers", []) if isinstance(result, dict) else []
            candles = result.get("market_candlesticks", []) if isinstance(result, dict) else []
            total = sum(len(c) for c in candles if c)
            if total == 0:
                logger.info(f"Kalshi candles empty for {event_ticker} (series={series_ticker}, window={self._candle_window_minutes}m, period={period}m)")
            else:
                logger.info(f"Kalshi candles: {total} points across {len(tickers)} markets for {event_ticker}")
            return result
        except Exception as e:
            logger.warning(
                f"Kalshi candles fetch failed for {event_ticker} "
                f"(series={series_ticker}): [{type(e).__name__}] {e}"
            )
            return {}

    async def _fetch_poly_event(self, poly_event_id: str) -> Dict:
        """GET Gamma /events/{id}."""
        if not self._poly_client or not poly_event_id:
            return {}
        try:
            return await self._poly_client.get_event_by_id(poly_event_id)
        except Exception as e:
            logger.debug(f"Poly event fetch failed for {poly_event_id}: {e}")
            return {}

    async def _fetch_poly_volume(self, poly_event_id: str) -> Any:
        """GET data-api /live-volume."""
        if not self._poly_client or not poly_event_id:
            return {}
        try:
            return await self._poly_client.get_live_volume(poly_event_id)
        except Exception as e:
            logger.debug(f"Poly volume fetch failed for {poly_event_id}: {e}")
            return {}

    def _poly_interval(self) -> str:
        """Map _candle_window_minutes to a valid Polymarket interval."""
        mins = self._candle_window_minutes
        if mins <= 60:
            return "1h"
        if mins <= 360:
            return "6h"
        if mins <= 1440:
            return "1d"
        if mins <= 10080:
            return "1w"
        return "max"

    async def _fetch_poly_price_history(self, token_id: str) -> List[Dict]:
        """GET CLOB /prices-history."""
        if not self._poly_client or not token_id:
            return []
        try:
            interval = self._poly_interval()
            result = await self._poly_client.get_price_history(
                token_id=token_id,
                interval=interval,
                fidelity=1,
            )
            if not result:
                logger.info(f"Poly price history empty for token {token_id[:12]}... (interval={interval})")
            else:
                logger.debug(f"Poly price history: {len(result)} points for {token_id[:12]}... (interval={interval})")
            return result
        except Exception as e:
            logger.warning(f"Poly price history failed for {token_id[:12]}...: {e}")
            return []

    def get_entry(self, event_ticker: str) -> Optional[CodexEntry]:
        return self._cache.get(event_ticker)

    def get_full_snapshot(self) -> Dict[str, Any]:
        """All entries serialized for WebSocket broadcast."""
        entries = {}
        for ticker, entry in self._cache.items():
            entries[ticker] = _codex_entry_to_dict(entry)
        return {
            "entries": entries,
            "total": len(entries),
            "last_sync_at": self._last_sync_at,
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "sync_count": self._sync_count,
            "entries": len(self._cache),
            "last_sync_at": self._last_sync_at,
            "last_sync_duration": self._last_sync_duration,
            "poll_interval": self._poll_interval,
        }

    def is_healthy(self) -> bool:
        return self._running


def _safe_float(val) -> Optional[float]:
    """Safely convert to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
