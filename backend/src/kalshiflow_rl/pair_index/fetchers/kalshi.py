"""
Standalone Kalshi event fetcher.

Uses httpx directly with RSA signature auth. Zero traderv3 imports.
"""

import base64
import json
import logging
import os
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from .models import NormalizedEvent, NormalizedMarket

logger = logging.getLogger("kalshiflow_rl.pair_index.fetchers.kalshi")


def _load_private_key(content: str) -> rsa.RSAPrivateKey:
    """Load RSA private key from string content."""
    if not content.startswith("-----BEGIN"):
        content = f"-----BEGIN PRIVATE KEY-----\n{content}\n-----END PRIVATE KEY-----"
    key = serialization.load_pem_private_key(content.encode(), password=None)
    if not isinstance(key, rsa.RSAPrivateKey):
        raise ValueError("Key must be RSA")
    return key


def _sign(private_key: rsa.RSAPrivateKey, message: str) -> str:
    """Sign message with RSA-PSS + SHA256, return base64."""
    sig = private_key.sign(
        message.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def _auth_headers(api_key_id: str, private_key: rsa.RSAPrivateKey, method: str, path: str) -> Dict[str, str]:
    """Create Kalshi auth headers."""
    ts = str(int(time.time() * 1000))
    sig = _sign(private_key, ts + method + path)
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse ISO datetime string."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class KalshiFetcher:
    """Fetch and normalize Kalshi events with nested markets."""

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_content: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self._api_key_id = api_key_id or os.environ.get("KALSHI_API_KEY_ID", "")
        key_content = private_key_content or os.environ.get("KALSHI_PRIVATE_KEY_CONTENT", "")
        self._api_url = (api_url or os.environ.get("KALSHI_API_URL", "https://api.elections.kalshi.com/trade-api/v2")).rstrip("/")

        self._private_key: Optional[rsa.RSAPrivateKey] = None
        if key_content:
            try:
                self._private_key = _load_private_key(key_content)
            except Exception as e:
                logger.warning(f"Failed to load Kalshi private key: {e}")

        # Series -> category cache
        self._series_category: Dict[str, str] = {}

    async def fetch_events(
        self,
        limit: int = 200,
        max_pages: int = 5,
        status: str = "open",
    ) -> List[NormalizedEvent]:
        """Fetch Kalshi events with nested markets, paginating through cursor pages."""
        if not self._api_key_id or not self._private_key:
            logger.error("Kalshi credentials not configured")
            return []

        raw_events: List[Dict] = []
        cursor: Optional[str] = None

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Fetch series for category mapping (cached)
            if not self._series_category:
                await self._fetch_series(client)

            for page in range(max_pages):
                path = f"/trade-api/v2/events?limit={limit}&status={status}&with_nested_markets=true"
                if cursor:
                    path += f"&cursor={cursor}"

                headers = _auth_headers(self._api_key_id, self._private_key, "GET", path)
                try:
                    resp = await client.get(f"{self._api_url}/events", params={
                        "limit": limit,
                        "status": status,
                        "with_nested_markets": "true",
                        **({"cursor": cursor} if cursor else {}),
                    }, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error(f"Kalshi events page {page} failed: {e}")
                    break

                page_events = data.get("events", [])
                if not page_events:
                    break
                raw_events.extend(page_events)
                cursor = data.get("cursor")
                if not cursor:
                    break

        if not raw_events:
            return []

        events = self._normalize_events(raw_events)
        logger.info(f"Fetched {len(events)} Kalshi events ({sum(e.market_count for e in events)} markets)")
        return events

    async def _fetch_series(self, client: httpx.AsyncClient) -> None:
        """Fetch series -> category mapping."""
        path = "/trade-api/v2/series"
        headers = _auth_headers(self._api_key_id, self._private_key, "GET", path)
        try:
            resp = await client.get(f"{self._api_url}/series", headers=headers)
            resp.raise_for_status()
            data = resp.json()
            for s in data.get("series", []):
                ticker = (s.get("series_ticker") or s.get("ticker") or "").strip()
                cat = (s.get("category", "") or "").lower().strip()
                if ticker and cat:
                    self._series_category[ticker] = cat
            logger.info(f"Cached {len(self._series_category)} Kalshi series->category mappings")
        except Exception as e:
            logger.warning(f"Failed to fetch Kalshi series: {e}")

    def _normalize_events(self, raw_events: List[Dict]) -> List[NormalizedEvent]:
        """Convert raw API events to NormalizedEvent objects."""
        events = []
        for raw in raw_events:
            title = raw.get("title", "")
            event_ticker = raw.get("event_ticker", "")
            category = (raw.get("category", "") or "").lower().strip()

            # Fallback to series API category
            if not category:
                series_ticker = (raw.get("series_ticker", "") or "").strip()
                category = self._series_category.get(series_ticker, "")

            raw_markets = raw.get("markets", [])

            # Compute volume
            volume = 0
            for m in raw_markets:
                volume += m.get("volume", 0) or 0
                volume += m.get("volume_24h", 0) or 0

            # Parse close time from earliest market
            close_times = []
            for m in raw_markets:
                ct = _parse_iso(m.get("close_time") or m.get("expected_expiration_time"))
                if ct:
                    close_times.append(ct)
            close_time = min(close_times) if close_times else None

            markets = []
            for m in raw_markets:
                m_status = (m.get("status", "") or "").lower()
                m_title = m.get("title", "") or m.get("subtitle", "") or ""
                ticker = m.get("ticker", "")

                markets.append(NormalizedMarket(
                    venue="kalshi",
                    event_id=event_ticker,
                    market_id=ticker,
                    question=m_title,
                    close_time=_parse_iso(m.get("close_time")),
                    is_active=m_status in ("open", "active", ""),
                    kalshi_ticker=ticker,
                    kalshi_event_ticker=event_ticker,
                ))

            events.append(NormalizedEvent(
                venue="kalshi",
                event_id=event_ticker,
                title=title,
                category=category,
                markets=markets,
                close_time=close_time,
                mutually_exclusive=raw.get("mutually_exclusive", False),
                market_count=len(markets),
                volume_24h=volume,
                raw=raw,
            ))

        return events
