"""
Order Context Service - Manages order context capture and persistence.

Handles the lifecycle of order contexts:
1. STAGE: Capture context in memory when order is placed
2. PERSIST: Write to DB when fill is confirmed
3. LINK: Update settlement data when market settles
4. EXPORT: Provide CSV export for quant analysis

Architecture:
    - Uses shared RLDatabase connection pool
    - Non-blocking async writes
    - In-memory staging for unfilled orders
    - Settlement linking via order_id
"""

import asyncio
import csv
import io
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

from ..state.order_context import StagedOrderContext

logger = logging.getLogger("kalshiflow_rl.traderv3.services.order_context_service")


class OrderContextService:
    """
    Service for capturing and persisting order context.

    Maintains staged contexts in memory until fill confirmation,
    then persists to PostgreSQL for quant analysis.
    """

    def __init__(self, db_pool=None):
        """
        Initialize the order context service.

        Args:
            db_pool: asyncpg connection pool (optional, for testing)
        """
        self._staged_contexts: Dict[str, StagedOrderContext] = {}
        self._db_pool = db_pool
        self._initialized = False
        self._init_lock = asyncio.Lock()

        # Metrics
        self._staged_count = 0
        self._persisted_count = 0
        self._settlement_linked_count = 0
        self._errors_count = 0

    async def initialize(self, db_pool=None) -> None:
        """Initialize with database connection pool."""
        async with self._init_lock:
            if self._initialized:
                return

            if db_pool:
                self._db_pool = db_pool

            self._initialized = True
            logger.info("OrderContextService initialized")

    @property
    def db_pool(self):
        """Public accessor for database connection pool."""
        return self._db_pool

    def stage_context(self, context: StagedOrderContext) -> None:
        """
        Stage order context in memory when order is placed.

        Context will be persisted to DB only when fill is confirmed.

        Args:
            context: StagedOrderContext with all order-time context
        """
        # Compute derived fields
        context.compute_derived_fields()

        self._staged_contexts[context.order_id] = context
        self._staged_count += 1

        logger.debug(
            f"Staged order context: order_id={context.order_id}, "
            f"ticker={context.market_ticker}, strategy={context.strategy}"
        )

    def has_staged_context(self, order_id: str) -> bool:
        """Check if a staged context exists for an order."""
        return order_id in self._staged_contexts

    def get_staged_context(self, order_id: str) -> Optional[StagedOrderContext]:
        """Get staged context for an order (if exists)."""
        return self._staged_contexts.get(order_id)

    async def persist_on_fill(
        self,
        order_id: str,
        fill_count: int,
        fill_avg_price_cents: Optional[int],
        filled_at: float,
    ) -> bool:
        """
        Persist staged context to database when fill is confirmed.

        Removes context from staging after successful persistence.

        Args:
            order_id: Kalshi order ID
            fill_count: Number of contracts filled
            fill_avg_price_cents: Average fill price in cents
            filled_at: Unix timestamp when filled

        Returns:
            True if successfully persisted, False otherwise
        """
        context = self._staged_contexts.get(order_id)
        if not context:
            logger.warning(f"No staged context found for order_id={order_id}")
            return False

        if not self._db_pool:
            logger.warning("No database pool configured, skipping persistence")
            # Remove from staging anyway to prevent memory leak
            del self._staged_contexts[order_id]
            return False

        try:
            db_dict = context.to_db_dict(
                fill_count=fill_count,
                fill_avg_price_cents=fill_avg_price_cents,
                filled_at=filled_at,
            )

            async with self._db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO order_contexts (
                        order_id, market_ticker, session_id,
                        strategy, signal_id, signal_detected_at, signal_params,
                        market_category, market_close_ts, hours_to_settlement, trades_in_market,
                        no_price_at_signal, bucket_5c,
                        best_bid_cents, best_ask_cents, bid_ask_spread_cents, spread_tier,
                        bid_size_contracts, ask_size_contracts,
                        existing_position_count, existing_position_side, is_reentry, entry_number,
                        balance_cents, open_position_count,
                        action, side, order_price_cents, order_quantity, order_type,
                        placed_at, hour_of_day_utc, day_of_week, calendar_week,
                        fill_count, fill_avg_price_cents, filled_at, time_to_fill_ms, slippage_cents,
                        strategy_version
                    ) VALUES (
                        $1, $2, $3,
                        $4, $5, $6, $7,
                        $8, $9, $10, $11,
                        $12, $13,
                        $14, $15, $16, $17,
                        $18, $19,
                        $20, $21, $22, $23,
                        $24, $25,
                        $26, $27, $28, $29, $30,
                        $31, $32, $33, $34,
                        $35, $36, $37, $38, $39,
                        $40
                    )
                    ON CONFLICT (order_id) DO NOTHING
                    """,
                    db_dict["order_id"],
                    db_dict["market_ticker"],
                    db_dict["session_id"],
                    db_dict["strategy"],
                    db_dict["signal_id"],
                    db_dict["signal_detected_at"],
                    json.dumps(db_dict["signal_params"]),
                    db_dict["market_category"],
                    db_dict["market_close_ts"],
                    db_dict["hours_to_settlement"],
                    db_dict["trades_in_market"],
                    db_dict["no_price_at_signal"],
                    db_dict["bucket_5c"],
                    db_dict["best_bid_cents"],
                    db_dict["best_ask_cents"],
                    db_dict["bid_ask_spread_cents"],
                    db_dict["spread_tier"],
                    db_dict["bid_size_contracts"],
                    db_dict["ask_size_contracts"],
                    db_dict["existing_position_count"],
                    db_dict["existing_position_side"],
                    db_dict["is_reentry"],
                    db_dict["entry_number"],
                    db_dict["balance_cents"],
                    db_dict["open_position_count"],
                    db_dict["action"],
                    db_dict["side"],
                    db_dict["order_price_cents"],
                    db_dict["order_quantity"],
                    db_dict["order_type"],
                    db_dict["placed_at"],
                    db_dict["hour_of_day_utc"],
                    db_dict["day_of_week"],
                    db_dict["calendar_week"],
                    db_dict["fill_count"],
                    db_dict["fill_avg_price_cents"],
                    db_dict["filled_at"],
                    db_dict["time_to_fill_ms"],
                    db_dict["slippage_cents"],
                    db_dict["strategy_version"],
                )

            # Remove from staging after successful persistence
            del self._staged_contexts[order_id]
            self._persisted_count += 1

            logger.info(
                f"Persisted order context: order_id={order_id}, "
                f"ticker={context.market_ticker}, fill_count={fill_count}"
            )
            return True

        except Exception as e:
            self._errors_count += 1
            logger.error(f"Failed to persist order context for {order_id}: {e}")
            # Delete staged context to prevent memory leak (data is lost but prevents unbounded growth)
            if order_id in self._staged_contexts:
                del self._staged_contexts[order_id]
                logger.warning(f"Discarded staged context for {order_id} after persist failure")
            return False

    async def link_settlement(
        self,
        order_id: str,
        market_result: str,
        realized_pnl_cents: int,
        settled_at: float,
    ) -> bool:
        """
        Link settlement data to persisted order context.

        Updates the existing DB record with settlement outcome.

        Args:
            order_id: Kalshi order ID
            market_result: Settlement result ('yes', 'no', 'void')
            realized_pnl_cents: Realized P&L in cents
            settled_at: Unix timestamp when settled

        Returns:
            True if successfully updated, False otherwise
        """
        if not self._db_pool:
            logger.warning("No database pool configured, skipping settlement link")
            return False

        try:
            settled_dt = datetime.fromtimestamp(settled_at, tz=timezone.utc)

            async with self._db_pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE order_contexts
                    SET market_result = $1,
                        realized_pnl_cents = $2,
                        settled_at = $3
                    WHERE order_id = $4
                    """,
                    market_result,
                    realized_pnl_cents,
                    settled_dt,
                    order_id,
                )

                # Check if any row was updated
                if result == "UPDATE 1":
                    self._settlement_linked_count += 1
                    logger.info(
                        f"Linked settlement: order_id={order_id}, "
                        f"result={market_result}, pnl={realized_pnl_cents}c"
                    )
                    return True
                else:
                    logger.debug(f"No order context found to link settlement: {order_id}")
                    return False

        except Exception as e:
            self._errors_count += 1
            logger.error(f"Failed to link settlement for {order_id}: {e}")
            return False

    async def get_contexts_for_export(
        self,
        strategy: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        settled_only: bool = True,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Query order contexts for export.

        Args:
            strategy: Filter by strategy name
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            settled_only: Only include settled orders
            limit: Maximum number of records

        Returns:
            List of order context dictionaries
        """
        if not self._db_pool:
            logger.warning("No database pool configured")
            return []

        try:
            # Build query with filters
            conditions = []
            params = []
            param_idx = 1

            if settled_only:
                conditions.append("settled_at IS NOT NULL")

            if strategy:
                conditions.append(f"strategy = ${param_idx}")
                params.append(strategy)
                param_idx += 1

            if from_date:
                conditions.append(f"filled_at >= ${param_idx}")
                params.append(datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc))
                param_idx += 1

            if to_date:
                conditions.append(f"filled_at <= ${param_idx}")
                params.append(datetime.strptime(to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc))
                param_idx += 1

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            query = f"""
                SELECT *
                FROM order_contexts
                WHERE {where_clause}
                ORDER BY filled_at DESC
                LIMIT {limit}
            """

            async with self._db_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                return [dict(row) for row in rows]

        except Exception as e:
            self._errors_count += 1
            logger.error(f"Failed to query order contexts: {e}")
            return []

    def generate_csv(self, contexts: List[Dict[str, Any]]) -> str:
        """
        Generate CSV string from order contexts.

        Args:
            contexts: List of order context dictionaries

        Returns:
            CSV string content
        """
        if not contexts:
            return ""

        output = io.StringIO()

        # Define column order for CSV
        columns = [
            "order_id", "market_ticker", "strategy", "signal_id",
            "market_category", "hours_to_settlement", "trades_in_market",
            "no_price_at_signal", "bucket_5c",
            "best_bid_cents", "best_ask_cents", "bid_ask_spread_cents", "spread_tier",
            "existing_position_count", "is_reentry", "balance_cents",
            "action", "side", "order_price_cents", "order_quantity",
            "fill_count", "fill_avg_price_cents", "filled_at",
            "slippage_cents", "time_to_fill_ms",
            "market_result", "settled_at", "realized_pnl_cents",
            "signal_params", "calendar_week", "session_id",
        ]

        writer = csv.DictWriter(output, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()

        for ctx in contexts:
            # Convert datetimes to ISO format
            row = dict(ctx)
            for key in ["filled_at", "settled_at", "signal_detected_at", "market_close_ts", "placed_at"]:
                if key in row and row[key]:
                    if isinstance(row[key], datetime):
                        row[key] = row[key].isoformat()

            # Convert signal_params to JSON string if needed
            if "signal_params" in row and row["signal_params"]:
                if not isinstance(row["signal_params"], str):
                    row["signal_params"] = json.dumps(row["signal_params"])

            writer.writerow(row)

        return output.getvalue()

    def discard_staged_context(self, order_id: str) -> bool:
        """
        Discard a staged context (e.g., for cancelled orders).

        Args:
            order_id: Order ID to discard

        Returns:
            True if context was found and discarded
        """
        if order_id in self._staged_contexts:
            del self._staged_contexts[order_id]
            logger.debug(f"Discarded staged context for order_id={order_id}")
            return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            "staged_count": len(self._staged_contexts),
            "total_staged": self._staged_count,
            "total_persisted": self._persisted_count,
            "total_settlement_linked": self._settlement_linked_count,
            "total_errors": self._errors_count,
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Clear staged contexts (they will be lost, but that's expected on shutdown)
        staged_count = len(self._staged_contexts)
        if staged_count > 0:
            logger.warning(f"Discarding {staged_count} staged contexts on cleanup")
        self._staged_contexts.clear()
        self._initialized = False


# Global service instance
_order_context_service: Optional[OrderContextService] = None


def get_order_context_service() -> OrderContextService:
    """Get the global OrderContextService instance."""
    global _order_context_service
    if _order_context_service is None:
        _order_context_service = OrderContextService()
    return _order_context_service
