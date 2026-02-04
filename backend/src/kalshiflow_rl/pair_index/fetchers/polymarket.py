"""
Standalone Polymarket event fetcher.

Uses httpx directly against the Gamma API. No auth needed.
Zero traderv3 imports.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from .models import NormalizedEvent, NormalizedMarket

logger = logging.getLogger("kalshiflow_rl.pair_index.fetchers.polymarket")

GAMMA_API = "https://gamma-api.polymarket.com"


def _parse_iso(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


class PolymarketFetcher:
    """Fetch and normalize Polymarket events with nested markets."""

    def __init__(self, api_url: Optional[str] = None):
        self._api_url = (api_url or GAMMA_API).rstrip("/")

    async def fetch_events(
        self,
        limit: int = 100,
        max_pages: int = 5,
    ) -> List[NormalizedEvent]:
        """Fetch Polymarket events sorted by 24h volume descending."""
        raw_events: List[Dict] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for page in range(max_pages):
                try:
                    resp = await client.get(f"{self._api_url}/events", params={
                        "active": "true",
                        "limit": limit,
                        "offset": page * limit,
                        "order": "volume24hr",
                        "ascending": "false",
                    })
                    resp.raise_for_status()
                    page_events = resp.json()
                except Exception as e:
                    logger.error(f"Polymarket events page {page} failed: {e}")
                    break

                if not page_events:
                    break
                raw_events.extend(page_events)

        if not raw_events:
            return []

        events = self._normalize_events(raw_events)
        logger.info(f"Fetched {len(events)} Polymarket events ({sum(e.market_count for e in events)} markets)")
        return events

    def _normalize_events(self, raw_events: List[Dict]) -> List[NormalizedEvent]:
        """Convert raw Gamma API events to NormalizedEvent objects."""
        events = []
        for raw in raw_events:
            title = raw.get("title", "")
            slug = raw.get("slug", "")
            event_id = slug or raw.get("id", "")
            category = (raw.get("category", "") or "").lower().strip()

            # Parse tags
            raw_tags = raw.get("tags", []) or []
            tags = []
            for t in raw_tags:
                if isinstance(t, str):
                    tags.append(t.lower().strip())
                elif isinstance(t, dict):
                    label = t.get("label", "") or t.get("slug", "") or t.get("name", "")
                    if label:
                        tags.append(str(label).lower().strip())

            combined_category = category or (tags[0] if tags else "")
            mutually_exclusive = bool(raw.get("enableNegRisk", False))
            end_date = _parse_iso(raw.get("endDate"))

            # Volume
            volume = 0
            try:
                volume = int(float(raw.get("volume24hr", 0) or 0))
            except (ValueError, TypeError):
                pass

            raw_markets = raw.get("markets", [])
            markets = []
            for m in raw_markets:
                question = m.get("question", "") or m.get("groupItemTitle", "") or ""
                condition_id = m.get("conditionId", "") or m.get("condition_id", "")

                # clobTokenIds may be JSON string or list
                raw_clob = m.get("clobTokenIds") or []
                if isinstance(raw_clob, str):
                    try:
                        raw_clob = json.loads(raw_clob)
                    except (json.JSONDecodeError, ValueError):
                        raw_clob = []
                clob_token_ids = raw_clob if isinstance(raw_clob, list) else []
                token_yes = clob_token_ids[0] if len(clob_token_ids) > 0 else ""
                token_no = clob_token_ids[1] if len(clob_token_ids) > 1 else None

                is_active = bool(m.get("active", True)) and not bool(m.get("closed", False))

                markets.append(NormalizedMarket(
                    venue="polymarket",
                    event_id=event_id,
                    market_id=condition_id,
                    question=question,
                    close_time=_parse_iso(m.get("endDate")) or end_date,
                    is_active=is_active,
                    poly_condition_id=condition_id,
                    poly_token_id_yes=token_yes,
                    poly_token_id_no=token_no,
                ))

            events.append(NormalizedEvent(
                venue="polymarket",
                event_id=event_id,
                title=title,
                category=combined_category,
                markets=markets,
                close_time=end_date,
                mutually_exclusive=mutually_exclusive,
                market_count=len(markets),
                volume_24h=volume,
                raw=raw,
            ))

        return events
