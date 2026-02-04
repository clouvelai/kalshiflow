"""
Persistence layer for the pair index.

Reads/writes to the existing `paired_markets` Supabase table.
No schema changes needed.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Set

from .fetchers.models import MatchCandidate

logger = logging.getLogger("kalshiflow_rl.pair_index.store")


class PairStore:
    """Supabase persistence for paired_markets table."""

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        supabase_client: Optional[Any] = None,
    ):
        self._client = supabase_client
        if not self._client:
            url = supabase_url or os.environ.get("SUPABASE_URL", "")
            key = supabase_key or os.environ.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_KEY", ""))
            if url and key:
                try:
                    from supabase import create_client
                    self._client = create_client(url, key)
                except Exception as e:
                    logger.error(f"Failed to create Supabase client: {e}")

    @property
    def available(self) -> bool:
        return self._client is not None

    def upsert_pairs(self, candidates: List[MatchCandidate]) -> int:
        """
        Upsert matched pairs to paired_markets table.

        Returns count of rows upserted.
        """
        if not self._client or not candidates:
            return 0

        upserted = 0
        for c in candidates:
            row = {
                "kalshi_ticker": c.kalshi_ticker,
                "kalshi_event_ticker": c.kalshi_event_ticker,
                "poly_condition_id": c.poly_condition_id,
                "poly_token_id_yes": c.poly_token_id_yes,
                "question": c.question,
                "match_method": f"tier{c.match_tier}_{c.match_method}",
                "match_confidence": c.combined_score,
                "status": "active",
            }
            if c.poly_token_id_no:
                row["poly_token_id_no"] = c.poly_token_id_no

            try:
                self._client.table("paired_markets").upsert(
                    row, on_conflict="kalshi_ticker,poly_condition_id"
                ).execute()
                upserted += 1
            except Exception as e:
                logger.warning(f"Failed to upsert {c.kalshi_ticker}: {e}")

        logger.info(f"Upserted {upserted}/{len(candidates)} pairs to DB")
        return upserted

    def load_active_pairs(self) -> List[Dict[str, Any]]:
        """Load all active pairs from DB."""
        if not self._client:
            return []
        try:
            result = self._client.table("paired_markets").select("*").eq("status", "active").execute()
            return result.data or []
        except Exception as e:
            logger.error(f"Failed to load active pairs: {e}")
            return []

    def deactivate_stale(self, active_kalshi_tickers: Set[str]) -> int:
        """Mark pairs inactive if their Kalshi ticker is no longer in the active set."""
        if not self._client or not active_kalshi_tickers:
            return 0

        try:
            # Load all active pairs
            result = self._client.table("paired_markets").select("id,kalshi_ticker").eq("status", "active").execute()
            rows = result.data or []

            stale_ids = [
                r["id"] for r in rows
                if r["kalshi_ticker"] not in active_kalshi_tickers
            ]

            if not stale_ids:
                return 0

            for pair_id in stale_ids:
                try:
                    self._client.table("paired_markets").update(
                        {"status": "inactive"}
                    ).eq("id", pair_id).execute()
                except Exception as e:
                    logger.warning(f"Failed to deactivate pair {pair_id}: {e}")

            logger.info(f"Deactivated {len(stale_ids)} stale pairs")
            return len(stale_ids)

        except Exception as e:
            logger.error(f"Failed to deactivate stale pairs: {e}")
            return 0

    def deactivate_pair(self, kalshi_ticker: str, reason: str = "") -> bool:
        """Deactivate a specific pair (e.g. reported as bad by orchestrator)."""
        if not self._client:
            return False
        try:
            self._client.table("paired_markets").update(
                {"status": "inactive"}
            ).eq("kalshi_ticker", kalshi_ticker).eq("status", "active").execute()
            logger.info(f"Deactivated pair {kalshi_ticker}: {reason}")
            return True
        except Exception as e:
            logger.warning(f"Failed to deactivate {kalshi_ticker}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return index statistics."""
        if not self._client:
            return {"available": False}

        try:
            active = self._client.table("paired_markets").select("id", count="exact").eq("status", "active").execute()
            inactive = self._client.table("paired_markets").select("id", count="exact").eq("status", "inactive").execute()

            # Method breakdown
            methods = self._client.table("paired_markets").select("match_method").eq("status", "active").execute()
            method_counts: Dict[str, int] = {}
            for r in (methods.data or []):
                m = r.get("match_method", "unknown")
                method_counts[m] = method_counts.get(m, 0) + 1

            return {
                "available": True,
                "active_pairs": active.count if active.count is not None else len(active.data or []),
                "inactive_pairs": inactive.count if inactive.count is not None else len(inactive.data or []),
                "methods": method_counts,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"available": True, "error": str(e)}
