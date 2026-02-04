"""
Polymarket read-only client for cross-venue price data.

Provides async access to:
- Gamma API: Events and markets metadata
- CLOB API: Live prices, orderbooks, midpoints

No authentication required. All endpoints are public.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.polymarket_client")

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"

# Cache TTLs
PRICE_CACHE_TTL = 2.0  # seconds
METADATA_CACHE_TTL = 60.0  # seconds


class _CacheEntry:
    """Simple TTL cache entry."""
    __slots__ = ("data", "expires_at")

    def __init__(self, data: Any, ttl: float):
        self.data = data
        self.expires_at = time.monotonic() + ttl

    @property
    def valid(self) -> bool:
        return time.monotonic() < self.expires_at


class PolymarketClient:
    """
    Async read-only client for Polymarket public APIs.

    Usage:
        client = PolymarketClient()
        events = await client.get_events(limit=10)
        prices = await client.get_prices(token_ids=["0xabc...", "0xdef..."])
        await client.close()
    """

    def __init__(self, timeout: float = 10.0):
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
        self._cache: Dict[str, _CacheEntry] = {}
        self._request_count = 0
        self._error_count = 0
        self._last_request_time: Optional[float] = None

    async def get_events(
        self,
        limit: int = 50,
        active: bool = True,
        closed: bool = False,
        offset: int = 0,
        order: Optional[str] = None,
        ascending: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch events from Gamma API.

        Args:
            order: Sort field, e.g. "volume24hr", "startDate", "endDate".
            ascending: Sort direction. False = descending (highest first).
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "offset": offset,
        }
        if order is not None:
            params["order"] = order
        if ascending is not None:
            params["ascending"] = str(ascending).lower()

        cache_key = f"events:{limit}:{active}:{closed}:{offset}:{order}:{ascending}"
        return await self._get_cached(
            f"{GAMMA_API_BASE}/events",
            params=params,
            cache_key=cache_key,
            ttl=METADATA_CACHE_TTL,
        )

    async def get_markets(
        self,
        limit: int = 100,
        active: bool = True,
        closed: bool = False,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Fetch markets from Gamma API."""
        params: Dict[str, Any] = {
            "limit": limit,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "offset": offset,
        }
        return await self._get_cached(
            f"{GAMMA_API_BASE}/markets",
            params=params,
            cache_key=f"markets:{limit}:{active}:{closed}:{offset}",
            ttl=METADATA_CACHE_TTL,
        )

    async def search_events(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search Polymarket events by text query (not cached)."""
        return await self._get(f"{GAMMA_API_BASE}/events", params={"q": query, "limit": limit})

    async def get_prices(self, token_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Get current BUY/SELL prices for token IDs.

        Returns dict of token_id -> {"BUY": price, "SELL": price} (0.0 to 1.0).
        Uses POST /prices with JSON array of {token_id, side} objects.
        """
        if not token_ids:
            return {}

        cache_key = f"prices:{','.join(sorted(token_ids))}"
        entry = self._cache.get(cache_key)
        if entry and entry.valid:
            return entry.data

        # Build POST payload: request BUY and SELL for each token
        payload = []
        for tid in token_ids:
            payload.append({"token_id": tid, "side": "BUY"})
            payload.append({"token_id": tid, "side": "SELL"})

        result = await self._post(f"{CLOB_API_BASE}/prices", json_body=payload)
        if not isinstance(result, dict):
            return {}

        parsed: Dict[str, Dict[str, float]] = {}
        for token_id, sides in result.items():
            if isinstance(sides, dict):
                parsed[token_id] = {}
                for side, price in sides.items():
                    try:
                        parsed[token_id][side] = float(price)
                    except (ValueError, TypeError):
                        pass
            elif isinstance(sides, (int, float, str)):
                try:
                    parsed[token_id] = {"BUY": float(sides)}
                except (ValueError, TypeError):
                    pass

        self._cache[cache_key] = _CacheEntry(parsed, PRICE_CACHE_TTL)
        return parsed

    async def get_price(self, token_id: str, side: str = "BUY") -> Optional[float]:
        """Get market price for a single token and side (BUY or SELL).

        Uses GET /price?token_id=XXX&side=BUY per Polymarket docs.
        Returns None if no orderbook exists (404).
        """
        params = {"token_id": token_id, "side": side}
        result = await self._get_cached(
            f"{CLOB_API_BASE}/price",
            params=params,
            cache_key=f"price:{token_id}:{side}",
            ttl=PRICE_CACHE_TTL,
        )
        if isinstance(result, dict) and "price" in result:
            try:
                return float(result["price"])
            except (ValueError, TypeError):
                return None
        return None

    async def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get midpoint price for a single token via /midpoint endpoint.

        Note: May return 0.5 for thin/empty books. Prefer get_midpoints()
        which uses the batch /prices endpoint for more accurate pricing.
        """
        params = {"token_id": token_id}
        result = await self._get_cached(
            f"{CLOB_API_BASE}/midpoint",
            params=params,
            cache_key=f"midpoint:{token_id}",
            ttl=PRICE_CACHE_TTL,
        )
        if isinstance(result, dict) and "mid" in result:
            return float(result["mid"])
        return None

    async def get_book(self, token_id: str) -> Dict[str, Any]:
        """Get orderbook for a token."""
        params = {"token_id": token_id}
        return await self._get_cached(
            f"{CLOB_API_BASE}/book",
            params=params,
            cache_key=f"book:{token_id}",
            ttl=PRICE_CACHE_TTL,
        )

    async def get_midpoints(self, token_ids: List[str]) -> Dict[str, float]:
        """Get midpoints for multiple tokens via batch /prices endpoint.

        Computes midpoint as (BUY + SELL) / 2 from the /prices response.
        Single API call for all tokens. Falls back to BUY-only if SELL missing.
        """
        if not token_ids:
            return {}

        prices = await self.get_prices(token_ids)

        results: Dict[str, float] = {}
        for tid in token_ids:
            sides = prices.get(tid)
            if not sides:
                continue
            buy = sides.get("BUY")
            sell = sides.get("SELL")
            if buy is not None and sell is not None:
                results[tid] = (buy + sell) / 2.0
            elif buy is not None:
                results[tid] = buy
            elif sell is not None:
                results[tid] = sell

        if results:
            logger.debug(f"Batch prices: {len(results)}/{len(token_ids)} tokens priced")

        return results

    async def get_event_by_id(self, event_id: str) -> Dict[str, Any]:
        """Fetch a single event by ID from Gamma API.

        GET https://gamma-api.polymarket.com/events/{id}
        """
        return await self._get_cached(
            f"{GAMMA_API_BASE}/events/{event_id}",
            params={},
            cache_key=f"event:{event_id}",
            ttl=METADATA_CACHE_TTL,
        )

    async def get_live_volume(self, event_id: str) -> Any:
        """Fetch live volume data for an event.

        GET https://data-api.polymarket.com/live-volume?id={event_id}
        """
        return await self._get(
            "https://data-api.polymarket.com/live-volume",
            params={"id": event_id},
        )

    async def get_price_history(
        self, token_id: str, interval: str = "1h", fidelity: int = 1,
    ) -> List[Dict]:
        """Fetch price history for a token from CLOB API.

        GET https://clob.polymarket.com/prices-history
        Returns list of {t: unix_seconds, p: 0.0-1.0} points.

        The CLOB API returns {"history": [{t, p}, ...]}, so we extract
        the inner list before returning.
        """
        result = await self._get(
            f"{CLOB_API_BASE}/prices-history",
            params={"market": token_id, "interval": interval, "fidelity": fidelity},
        )
        # CLOB API wraps points in {"history": [...]}
        if isinstance(result, dict) and "history" in result:
            return result["history"]
        # Fallback: already a list (future-proof)
        if isinstance(result, list):
            return result
        return []

    async def _get_cached(
        self, url: str, params: Dict[str, Any], cache_key: str, ttl: float
    ) -> Any:
        """GET with TTL cache."""
        entry = self._cache.get(cache_key)
        if entry and entry.valid:
            return entry.data

        data = await self._get(url, params=params)
        self._cache[cache_key] = _CacheEntry(data, ttl)
        return data

    async def _get(self, url: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Raw GET request with error handling."""
        self._request_count += 1
        self._last_request_time = time.monotonic()

        try:
            resp = await self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            self._error_count += 1
            logger.warning(f"Polymarket API error {e.response.status_code}: {url}")
            return [] if "events" in url or "markets" in url else {}
        except httpx.RequestError as e:
            self._error_count += 1
            logger.warning(f"Polymarket request error: {e}")
            return [] if "events" in url or "markets" in url else {}

    async def _post(self, url: str, json_body: Any = None) -> Any:
        """Raw POST request with error handling."""
        self._request_count += 1
        self._last_request_time = time.monotonic()

        try:
            resp = await self._http.post(url, json=json_body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            self._error_count += 1
            logger.warning(f"Polymarket API error {e.response.status_code}: {url}")
            return {}
        except httpx.RequestError as e:
            self._error_count += 1
            logger.warning(f"Polymarket request error: {e}")
            return {}

    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics for health monitoring."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "cache_entries": len(self._cache),
            "last_request_time": self._last_request_time,
        }

    def is_healthy(self) -> bool:
        """Basic health check."""
        # Healthy if we've made at least one request without too many errors
        if self._request_count == 0:
            return True  # Haven't started yet
        error_rate = self._error_count / self._request_count
        return error_rate < 0.5

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()
