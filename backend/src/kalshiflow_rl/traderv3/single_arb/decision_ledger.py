"""DecisionLedger - Structured order decision tracking with production price evolution.

Records every Captain/Sniper order with production market context at decision time,
then backfills production price evolution at 1m/5m/15m/1h to compute:
- direction_correct: Was the Captain right about price direction?
- would_have_filled: Would the limit order have filled on production?
- hypothetical_pnl_cents: What P&L would have resulted?

Key Design:
- Fire-and-forget recording (never blocks order execution)
- Sampled price evolution (1m/5m/15m/1h windows, not continuous)
- Graceful degradation when DB unavailable
- Follows existing _backfill_news_impacts() pattern
"""

import asyncio
import json
import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .index import EventArbIndex
    from .memory.session_store import SessionMemoryStore

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.decision_ledger")

# Backfill windows: (min_age_seconds, column_name)
BACKFILL_WINDOWS = [
    (60, "prod_mid_1m"),
    (300, "prod_mid_5m"),
    (900, "prod_mid_15m"),
    (3600, "prod_mid_1h"),
]


class DecisionLedger:
    """Records and evaluates Captain/Sniper order decisions against production prices."""

    def __init__(self, index: "EventArbIndex", memory_store: Optional["SessionMemoryStore"] = None):
        self._index = index
        self._memory = memory_store
        self._pool = None
        self._pool_checked = False

    async def _get_pool(self):
        """Lazy-acquire DB pool. Returns None if unavailable."""
        if self._pool is not None:
            return self._pool
        if self._pool_checked:
            return None
        self._pool_checked = True
        try:
            from kalshiflow_rl.data.database import rl_db
            self._pool = await asyncio.wait_for(rl_db.get_pool(), timeout=5.0)
            return self._pool
        except Exception as e:
            logger.debug(f"[DECISION_LEDGER] DB pool unavailable: {e}")
            return None

    def _get_prod_snapshot(self, market_ticker: str) -> Dict[str, Any]:
        """Capture production BBO snapshot from the live index."""
        snapshot = {}
        if not self._index:
            return snapshot
        for event in self._index.events.values():
            market = event.markets.get(market_ticker)
            if market and market.yes_mid is not None:
                snapshot = {
                    "prod_yes_bid": market.yes_bid,
                    "prod_yes_ask": market.yes_ask,
                    "prod_yes_mid": market.yes_mid,
                    "prod_spread": market.spread,
                    "prod_volume_5m": market.micro.volume_5m if market.micro else 0,
                    "prod_book_imbalance": round(market.micro.book_imbalance, 3) if market.micro else 0.0,
                }
                break
        return snapshot

    async def record_decision(
        self,
        order_id: Optional[str],
        source: str,
        event_ticker: Optional[str],
        market_ticker: str,
        side: str,
        action: str,
        contracts: int,
        limit_price_cents: int,
        reasoning: Optional[str] = None,
        cycle_mode: Optional[str] = None,
    ) -> None:
        """Record a trading decision with production market snapshot. Fire-and-forget."""
        pool = await self._get_pool()
        if not pool:
            return

        snap = self._get_prod_snapshot(market_ticker)

        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO captain_decisions (
                        order_id, source, event_ticker, market_ticker,
                        side, action, contracts, limit_price_cents,
                        reasoning, cycle_mode,
                        prod_yes_bid, prod_yes_ask, prod_yes_mid,
                        prod_spread, prod_volume_5m, prod_book_imbalance
                    ) VALUES (
                        $1, $2, $3, $4,
                        $5, $6, $7, $8,
                        $9, $10,
                        $11, $12, $13,
                        $14, $15, $16
                    )
                    """,
                    order_id, source, event_ticker, market_ticker,
                    side, action, contracts, limit_price_cents,
                    reasoning, cycle_mode,
                    snap.get("prod_yes_bid"),
                    snap.get("prod_yes_ask"),
                    snap.get("prod_yes_mid"),
                    snap.get("prod_spread"),
                    snap.get("prod_volume_5m"),
                    snap.get("prod_book_imbalance"),
                )
            logger.debug(
                f"[DECISION_LEDGER] Recorded {source} decision: "
                f"{action} {contracts} {side} {market_ticker} @{limit_price_cents}c"
            )
        except Exception as e:
            logger.warning(f"[DECISION_LEDGER] Record failed: {e}")

    async def backfill_outcomes(self, tracked_orders: Dict[str, dict]) -> None:
        """Backfill price evolution and compute metrics for recent decisions.

        Called periodically by coordinator. For each decision in last 2 hours
        where outcome columns are NULL, fills in sampled prices and computes
        direction_correct, would_have_filled, hypothetical_pnl_cents.
        """
        pool = await self._get_pool()
        if not pool:
            return

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, order_id, market_ticker, side, action,
                           contracts, limit_price_cents, prod_yes_mid,
                           prod_mid_1m, prod_mid_5m, prod_mid_15m, prod_mid_1h,
                           demo_status, demo_fill_count,
                           direction_correct, would_have_filled,
                           created_at
                    FROM captain_decisions
                    WHERE created_at >= now() - INTERVAL '2 hours'
                      AND (prod_mid_1h IS NULL OR direction_correct IS NULL)
                    ORDER BY created_at ASC
                    """,
                )

                if not rows:
                    return

                now = time.time()
                updated = 0

                for row in rows:
                    decision_id = row["id"]
                    market_ticker = row["market_ticker"]
                    created_ts = row["created_at"].timestamp()
                    age_seconds = now - created_ts
                    entry_mid = row["prod_yes_mid"]

                    updates = {}
                    params = []

                    # Get current mid for filling windows
                    current_mid = self._get_current_mid(market_ticker)

                    # Fill price windows based on age
                    for min_age, col in BACKFILL_WINDOWS:
                        if row[col] is None and age_seconds >= min_age and current_mid is not None:
                            updates[col] = current_mid

                    # Sync demo order status from tracked_orders
                    order_id = row["order_id"]
                    if order_id and order_id in tracked_orders:
                        tracked = tracked_orders[order_id]
                        tracked_status = tracked.get("status", "")
                        tracked_fills = tracked.get("fill_count", 0)
                        if tracked_status and row["demo_status"] != tracked_status:
                            updates["demo_status"] = tracked_status
                        if tracked_fills and (row["demo_fill_count"] or 0) != tracked_fills:
                            updates["demo_fill_count"] = tracked_fills

                    # Compute metrics once we have the 1h window or all available data
                    sampled_mids = self._collect_sampled_mids(row, updates)

                    if entry_mid is not None and sampled_mids:
                        # Direction correct
                        if row["direction_correct"] is None:
                            # Use the latest available window
                            latest_mid = sampled_mids[-1]
                            is_bullish = (row["action"] == "buy" and row["side"] == "yes") or \
                                         (row["action"] == "sell" and row["side"] == "no")
                            if is_bullish:
                                updates["direction_correct"] = latest_mid > entry_mid
                            else:
                                updates["direction_correct"] = latest_mid < entry_mid

                        # Would have filled
                        if row["would_have_filled"] is None:
                            limit = row["limit_price_cents"]
                            updates["would_have_filled"] = self._check_would_have_filled(
                                row["side"], row["action"], limit, sampled_mids
                            )

                        # Hypothetical P&L (only if would have filled)
                        filled = updates.get("would_have_filled", row["would_have_filled"])
                        if filled:
                            latest_mid = sampled_mids[-1]
                            updates["hypothetical_pnl_cents"] = self._compute_pnl(
                                row["side"], row["action"], row["limit_price_cents"],
                                latest_mid, row["contracts"]
                            )

                        # Max favorable / adverse excursion
                        fav, adv = self._compute_excursions(
                            row["side"], row["action"], row["limit_price_cents"],
                            sampled_mids
                        )
                        if fav is not None:
                            updates["max_favorable_cents"] = fav
                        if adv is not None:
                            updates["max_adverse_cents"] = adv

                    if not updates:
                        continue

                    updates["updated_at"] = "now()"

                    # Build UPDATE query
                    set_parts = []
                    param_values = []
                    idx = 1
                    for col, val in updates.items():
                        if col == "updated_at":
                            set_parts.append(f"{col} = now()")
                        else:
                            set_parts.append(f"{col} = ${idx}")
                            param_values.append(val)
                            idx += 1

                    param_values.append(decision_id)
                    await conn.execute(
                        f"UPDATE captain_decisions SET {', '.join(set_parts)} WHERE id = ${idx}",
                        *param_values,
                    )
                    updated += 1

                if updated:
                    logger.info(f"[DECISION_LEDGER] Backfilled {updated}/{len(rows)} decisions")

        except Exception as e:
            logger.warning(f"[DECISION_LEDGER] Backfill error: {e}")

    async def get_accuracy_stats(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get aggregated accuracy statistics via SQL function."""
        pool = await self._get_pool()
        if not pool:
            return self._empty_stats()

        try:
            async with pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT get_captain_accuracy_stats($1, $2)",
                    hours_back, None,
                )
                if result:
                    return json.loads(result) if isinstance(result, str) else result
        except Exception as e:
            logger.debug(f"[DECISION_LEDGER] Stats query failed: {e}")

        return self._empty_stats()

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _get_current_mid(self, market_ticker: str) -> Optional[float]:
        """Get current yes_mid from live index."""
        if not self._index:
            return None
        for event in self._index.events.values():
            market = event.markets.get(market_ticker)
            if market and market.yes_mid is not None:
                return market.yes_mid
        return None

    @staticmethod
    def _collect_sampled_mids(row: dict, updates: dict) -> List[float]:
        """Collect all available sampled mid prices (from row + updates)."""
        mids = []
        for _, col in BACKFILL_WINDOWS:
            val = updates.get(col) if col in updates else row.get(col)
            if val is not None:
                mids.append(float(val))
        return mids

    @staticmethod
    def _check_would_have_filled(
        side: str, action: str, limit_cents: int, sampled_mids: List[float]
    ) -> bool:
        """Check if limit order would have filled based on sampled production prices.

        For buy YES: filled if prod mid <= limit (seller willing at or below our bid)
        For buy NO: filled if prod YES mid >= (100 - limit) (NO ask <= our limit)
        For sell YES: filled if prod mid >= limit (buyer willing at or above our ask)
        For sell NO: filled if prod YES mid <= (100 - limit)
        """
        for mid in sampled_mids:
            if action == "buy":
                if side == "yes" and mid <= limit_cents:
                    return True
                if side == "no" and mid >= (100 - limit_cents):
                    return True
            else:  # sell
                if side == "yes" and mid >= limit_cents:
                    return True
                if side == "no" and mid <= (100 - limit_cents):
                    return True
        return False

    @staticmethod
    def _compute_pnl(
        side: str, action: str, limit_cents: int, mark_mid: float, contracts: int
    ) -> float:
        """Compute hypothetical P&L in cents if order would have filled.

        For buy YES: pnl = (mark_mid - limit) * contracts
        For buy NO:  pnl = ((100 - mark_mid) - limit) * contracts
        For sell YES: pnl = (limit - mark_mid) * contracts
        For sell NO:  pnl = (limit - (100 - mark_mid)) * contracts
        """
        if side == "yes":
            if action == "buy":
                return round((mark_mid - limit_cents) * contracts, 2)
            else:
                return round((limit_cents - mark_mid) * contracts, 2)
        else:  # no
            no_value = 100 - mark_mid
            if action == "buy":
                return round((no_value - limit_cents) * contracts, 2)
            else:
                return round((limit_cents - no_value) * contracts, 2)

    @staticmethod
    def _compute_excursions(
        side: str, action: str, limit_cents: int, sampled_mids: List[float]
    ) -> tuple:
        """Compute max favorable and max adverse excursion from sampled prices.

        Returns (max_favorable_cents, max_adverse_cents) or (None, None).
        """
        if not sampled_mids:
            return None, None

        pnls = []
        for mid in sampled_mids:
            if side == "yes":
                if action == "buy":
                    pnls.append(mid - limit_cents)
                else:
                    pnls.append(limit_cents - mid)
            else:
                no_value = 100 - mid
                if action == "buy":
                    pnls.append(no_value - limit_cents)
                else:
                    pnls.append(limit_cents - no_value)

        max_favorable = round(max(pnls), 2) if pnls else None
        max_adverse = round(min(pnls), 2) if pnls else None
        return max_favorable, max_adverse

    @staticmethod
    def _empty_stats() -> Dict[str, Any]:
        """Return zeroed-out stats dict."""
        return {
            "total_decisions": 0,
            "decisions_with_outcomes": 0,
            "direction_correct_count": 0,
            "direction_accuracy_pct": 0.0,
            "avg_hypothetical_pnl": 0.0,
            "total_hypothetical_pnl": 0.0,
            "would_have_filled_count": 0,
            "would_have_filled_pct": 0.0,
            "by_source": {},
            "by_cycle_mode": {},
        }
