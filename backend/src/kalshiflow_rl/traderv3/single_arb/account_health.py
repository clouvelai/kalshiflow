"""AccountHealthService - Non-LLM background loop for account hygiene.

Runs on 30-second ticks, separate from the Captain agent.
Performs balance tracking, settlement discovery, stale order cleanup,
stale position detection, and order group hygiene.

Auto-fixes: stale orders (cancels), orphaned order groups (deletes).
Report-only: drawdown, settlements, stale positions (for Captain).
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, TYPE_CHECKING

from .models import AccountHealthStatus, SettlementSummary, StalePosition

if TYPE_CHECKING:
    from .index import EventArbIndex
    from .memory.session_store import SessionMemoryStore
    from ..agent_tools.session import TradingSession
    from ..gateway.client import KalshiGateway

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.account_health")

TICK_INTERVAL = 30  # seconds


@dataclass
class HealthState:
    """Mutable internal state for AccountHealthService."""

    # Balance tracking
    balance_cents: int = 0
    balance_peak_cents: int = 0
    balance_trough_cents: int = 0
    balance_history: Deque[int] = field(default_factory=lambda: deque(maxlen=120))  # 1 hr at 30s

    # Settlements
    known_settlement_ids: Set[str] = field(default_factory=set)
    recent_settlements: Deque[Dict] = field(default_factory=lambda: deque(maxlen=50))
    total_settlement_revenue: int = 0
    total_settlement_count: int = 0

    # Hygiene counters
    stale_orders_cancelled: int = 0
    orphaned_groups_cleaned: int = 0

    # Position lifecycle
    stale_positions: List[StalePosition] = field(default_factory=list)
    cached_positions: List[Dict] = field(default_factory=list)

    # Alerts
    alerts: Deque[Dict] = field(default_factory=lambda: deque(maxlen=100))

    # Activity log (for frontend)
    activity_log: Deque[Dict] = field(default_factory=lambda: deque(maxlen=100))


class AccountHealthService:
    """Background service for account hygiene and health monitoring.

    Runs a 30-second tick loop performing staggered checks:
    - Every tick: balance check
    - Every 4 ticks (~2min): settlements + positions cache
    - Every 4 ticks (~2min): stale position detection
    - Every 10 ticks (~5min): stale order cleanup (auto-fix)
    - Every 60 ticks (~30min): order group hygiene (auto-fix)
    """

    def __init__(
        self,
        gateway: "KalshiGateway",
        index: "EventArbIndex",
        session: "TradingSession",
        order_group_id: Optional[str] = None,
        broadcast_callback: Optional[Callable[..., Coroutine]] = None,
        low_balance_threshold: int = 500,  # $5.00 in cents
        max_drawdown_pct: float = 25.0,  # Pause Captain when drawdown exceeds this
        pause_callback: Optional[Callable[[], None]] = None,
        resume_callback: Optional[Callable[[], None]] = None,
        memory: Optional["SessionMemoryStore"] = None,
    ):
        self._gateway = gateway
        self._index = index
        self._session = session
        self._order_group_id = order_group_id
        self._broadcast = broadcast_callback
        self._low_balance_threshold = low_balance_threshold
        self._max_drawdown_pct = max_drawdown_pct
        self._pause_callback = pause_callback
        self._resume_callback = resume_callback
        self._memory = memory

        self.state = HealthState()
        self._tick_count = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._paused_by_drawdown = False

    async def start(self) -> None:
        """Start the background health loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("[HEALTH] AccountHealthService started (30s ticks)")

    async def stop(self) -> None:
        """Stop the background health loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("[HEALTH] AccountHealthService stopped")

    async def _run_loop(self) -> None:
        """Main tick loop."""
        while self._running:
            try:
                await self._tick()
                self._tick_count += 1
                await asyncio.sleep(TICK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[HEALTH] Tick error: {e}")
                await asyncio.sleep(10)

    async def _tick(self) -> None:
        """Execute one health check tick with staggered checks."""
        tick = self._tick_count
        state_changed = False

        # Every tick: balance
        changed = await self._check_balance()
        state_changed = state_changed or changed

        # Every 4 ticks: settlements + positions
        if tick % 4 == 1:
            changed = await self._check_settlements()
            state_changed = state_changed or changed
            changed = await self._check_stale_positions()
            state_changed = state_changed or changed

        # Every 10 ticks: stale orders (auto-fix)
        if tick % 10 == 5:
            changed = await self._check_stale_orders()
            state_changed = state_changed or changed

        # Every 60 ticks: order group hygiene (auto-fix) + memory retention
        if tick % 60 == 30:
            changed = await self._check_order_groups()
            state_changed = state_changed or changed
            await self._enforce_memory_retention()

        # Broadcast if anything changed
        if state_changed and self._broadcast:
            try:
                status = self.get_health_status()
                await self._broadcast({
                    "type": "account_health_update",
                    "data": status.model_dump(),
                })
            except Exception as e:
                logger.debug(f"[HEALTH] Broadcast error: {e}")

    # ------------------------------------------------------------------ #
    #  Balance check (every tick)                                          #
    # ------------------------------------------------------------------ #

    async def _check_balance(self) -> bool:
        """Check balance, track peak/trough/trend, emit low-balance alert."""
        try:
            bal = await self._gateway.get_balance()
            new_balance = bal.balance
        except Exception as e:
            logger.debug(f"[HEALTH] Balance check failed: {e}")
            return False

        prev = self.state.balance_cents
        self.state.balance_cents = new_balance
        self.state.balance_history.append(new_balance)

        # Track peak/trough
        if new_balance > self.state.balance_peak_cents:
            self.state.balance_peak_cents = new_balance
        if self.state.balance_trough_cents == 0 or new_balance < self.state.balance_trough_cents:
            self.state.balance_trough_cents = new_balance

        # Low balance alert
        if new_balance < self._low_balance_threshold and (prev >= self._low_balance_threshold or prev == 0):
            self._add_alert(
                "low_balance",
                f"Balance ${new_balance / 100:.2f} below ${self._low_balance_threshold / 100:.2f} threshold",
                "warning",
            )
            return True

        # Drawdown circuit breaker
        if self.state.balance_peak_cents > 0:
            drawdown_pct = (self.state.balance_peak_cents - new_balance) / self.state.balance_peak_cents * 100
            if drawdown_pct >= self._max_drawdown_pct and not self._paused_by_drawdown:
                self._paused_by_drawdown = True
                self._add_alert(
                    "drawdown_circuit_breaker",
                    f"Drawdown {drawdown_pct:.1f}% >= {self._max_drawdown_pct}% threshold — PAUSING Captain",
                    "critical",
                )
                if self._pause_callback:
                    self._pause_callback()
                return True
            elif self._paused_by_drawdown and drawdown_pct < (self._max_drawdown_pct - 5.0):
                # Resume when drawdown recovers below threshold - 5%
                self._paused_by_drawdown = False
                self._add_alert(
                    "drawdown_recovered",
                    f"Drawdown recovered to {drawdown_pct:.1f}% — RESUMING Captain",
                    "info",
                )
                if self._resume_callback:
                    self._resume_callback()
                return True

        return new_balance != prev

    # ------------------------------------------------------------------ #
    #  Settlement check (every 4 ticks)                                    #
    # ------------------------------------------------------------------ #

    async def _check_settlements(self) -> bool:
        """Discover new settlements and track revenue."""
        try:
            settlements = await self._gateway.get_settlements(limit=50)
        except Exception as e:
            logger.debug(f"[HEALTH] Settlement check failed: {e}")
            return False

        new_count = 0
        for s in settlements:
            sid = s.settlement_id if hasattr(s, "settlement_id") else getattr(s, "id", str(id(s)))
            if sid in self.state.known_settlement_ids:
                continue

            self.state.known_settlement_ids.add(sid)
            result = s.result if hasattr(s, "result") else "unknown"
            revenue = s.revenue if hasattr(s, "revenue") else 0
            ticker = s.market_ticker if hasattr(s, "market_ticker") else ""
            settled_at = s.settled_time if hasattr(s, "settled_time") else None

            self.state.total_settlement_revenue += revenue
            self.state.total_settlement_count += 1
            new_count += 1

            summary = {
                "ticker": ticker,
                "result": result,
                "revenue_cents": revenue,
                "settled_at": str(settled_at) if settled_at else None,
            }
            self.state.recent_settlements.appendleft(summary)

            self._add_alert(
                "settlement_discovered",
                f"{ticker} settled {result.upper()}, revenue {'+' if revenue >= 0 else ''}{revenue / 100:.2f}",
                "info",
            )

            # Store settlement outcome to memory for Captain learning
            if self._memory:
                content = (
                    f"SETTLEMENT: {ticker} settled {result.upper()}, "
                    f"revenue {'+' if revenue >= 0 else ''}{revenue / 100:.2f}"
                )
                asyncio.create_task(self._memory.store(
                    content=content,
                    memory_type="settlement_outcome",
                    metadata={"ticker": ticker, "result": result, "revenue_cents": revenue},
                ))

        # Also cache positions while we're making API calls
        try:
            positions = await self._gateway.get_positions()
            self.state.cached_positions = [
                {
                    "ticker": p.ticker if hasattr(p, "ticker") else p.get("ticker", ""),
                    "position": p.position if hasattr(p, "position") else p.get("position", 0),
                    "market_exposure": p.market_exposure if hasattr(p, "market_exposure") else p.get("market_exposure", 0),
                }
                for p in positions
            ]
        except Exception:
            pass

        return new_count > 0

    # ------------------------------------------------------------------ #
    #  Stale position check (every 4 ticks)                                #
    # ------------------------------------------------------------------ #

    async def _check_stale_positions(self) -> bool:
        """Detect positions in settled/closed markets using the index."""
        if not self._index:
            return False

        stale = []
        for pos in self.state.cached_positions:
            ticker = pos.get("ticker", "")
            qty = pos.get("position", 0)
            if qty == 0:
                continue

            # Check if market exists in index
            event_ticker = self._index.get_event_for_ticker(ticker) if hasattr(self._index, "get_event_for_ticker") else None
            if not event_ticker:
                # Market not in index - might be stale
                stale.append(StalePosition(
                    ticker=ticker,
                    event_ticker="",
                    side="yes" if qty > 0 else "no",
                    quantity=abs(qty),
                    reason="not_in_index",
                ))

        prev_count = len(self.state.stale_positions)
        self.state.stale_positions = stale
        return len(stale) != prev_count

    # ------------------------------------------------------------------ #
    #  Stale order cleanup (every 10 ticks — auto-fix)                     #
    # ------------------------------------------------------------------ #

    async def _check_stale_orders(self) -> bool:
        """Find and cancel stale resting orders."""
        try:
            orders = await self._gateway.get_orders(status="resting")
        except Exception as e:
            logger.debug(f"[HEALTH] Order check failed: {e}")
            return False

        cancelled = 0
        now = time.time()

        for order in orders:
            order_id = order.order_id if hasattr(order, "order_id") else order.get("order_id", "")
            created_time = order.created_time if hasattr(order, "created_time") else order.get("created_time")
            expiration_time = order.expiration_time if hasattr(order, "expiration_time") else order.get("expiration_time")

            # Skip orders in the current session's order group (Captain manages those)
            order_group = order.order_group_id if hasattr(order, "order_group_id") else order.get("order_group_id")
            if order_group and order_group == self._order_group_id:
                continue

            # Check if order is stale (older than 30 minutes with no group)
            if created_time:
                try:
                    from datetime import datetime, timezone
                    if isinstance(created_time, str):
                        # Parse ISO format
                        ct = datetime.fromisoformat(created_time.replace("Z", "+00:00"))
                        age_seconds = now - ct.timestamp()
                    else:
                        age_seconds = now - float(created_time)

                    if age_seconds > 1800:  # 30 minutes
                        try:
                            await self._gateway.cancel_order(order_id)
                            cancelled += 1
                            self.state.stale_orders_cancelled += 1
                            self._add_alert(
                                "stale_order_cancelled",
                                f"Cancelled stale order {order_id[:8]}... (age={int(age_seconds)}s)",
                                "info",
                            )
                        except Exception as e:
                            logger.debug(f"[HEALTH] Cancel stale order {order_id[:8]} failed: {e}")
                except (ValueError, TypeError):
                    pass

        if cancelled:
            logger.info(f"[HEALTH] Cancelled {cancelled} stale orders")
        return cancelled > 0

    # ------------------------------------------------------------------ #
    #  Order group hygiene (every 60 ticks — auto-fix)                     #
    # ------------------------------------------------------------------ #

    async def _check_order_groups(self) -> bool:
        """Find and clean orphaned order groups (not current session)."""
        try:
            groups = await self._gateway.list_order_groups()
        except Exception as e:
            logger.debug(f"[HEALTH] Order group check failed: {e}")
            return False

        cleaned = 0
        for group in groups:
            gid = group.get("order_group_id", "")
            # Skip current session's group
            if gid == self._order_group_id:
                continue

            # Try to reset then delete orphaned groups
            try:
                await self._gateway.reset_order_group(gid)
                await self._gateway.delete_order_group(gid)
                cleaned += 1
                self.state.orphaned_groups_cleaned += 1
                self._add_alert(
                    "order_group_cleaned",
                    f"Cleaned orphaned order group {gid[:8]}...",
                    "info",
                )
            except Exception as e:
                logger.debug(f"[HEALTH] Clean order group {gid[:8]} failed: {e}")

        if cleaned:
            logger.info(f"[HEALTH] Cleaned {cleaned} orphaned order groups")
        return cleaned > 0

    # ------------------------------------------------------------------ #
    #  Status snapshot (zero I/O)                                          #
    # ------------------------------------------------------------------ #

    def get_health_status(self) -> AccountHealthStatus:
        """Build a zero-I/O snapshot of current health state."""
        s = self.state

        # Compute drawdown
        drawdown_pct = 0.0
        if s.balance_peak_cents > 0:
            drawdown_pct = round((s.balance_peak_cents - s.balance_cents) / s.balance_peak_cents * 100, 1)

        # Compute trend from history
        trend = "stable"
        if len(s.balance_history) >= 4:
            recent = list(s.balance_history)[-4:]
            avg_recent = sum(recent) / len(recent)
            older = list(s.balance_history)[:max(1, len(s.balance_history) // 2)]
            avg_older = sum(older) / len(older)
            if avg_recent > avg_older + 50:
                trend = "rising"
            elif avg_recent < avg_older - 50:
                trend = "falling"

        # Determine overall status
        status = "healthy"
        if drawdown_pct > 20 or s.balance_cents < self._low_balance_threshold:
            status = "critical"
        elif drawdown_pct > 10 or len(s.stale_positions) > 0:
            status = "warning"

        # Build settlement summaries
        settlements = [
            SettlementSummary(**ss)
            for ss in list(s.recent_settlements)[:5]
        ]

        return AccountHealthStatus(
            status=status,
            balance_cents=s.balance_cents,
            balance_dollars=round(s.balance_cents / 100, 2),
            balance_peak_cents=s.balance_peak_cents,
            drawdown_pct=drawdown_pct,
            balance_trend=trend,
            settlement_count_session=s.total_settlement_count,
            total_realized_pnl_cents=s.total_settlement_revenue,
            recent_settlements=settlements,
            stale_positions=list(s.stale_positions),
            stale_orders_cleaned=s.stale_orders_cancelled,
            orphaned_groups_cleaned=s.orphaned_groups_cleaned,
            alerts=list(s.alerts)[-10:],
            activity_log=list(s.activity_log)[-20:],
        )

    # ------------------------------------------------------------------ #
    #  Memory retention (every 60 ticks)                                   #
    # ------------------------------------------------------------------ #

    async def _enforce_memory_retention(self) -> None:
        """Call pgvector enforce_retention_policy() RPC to prevent unbounded growth."""
        try:
            from kalshiflow_rl.data.database import rl_db
            pool = await rl_db.get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchval("SELECT enforce_retention_policy()")
                if result and result > 0:
                    logger.info(f"[HEALTH] Memory retention: deactivated {result} old memories")
        except Exception as e:
            logger.debug(f"[HEALTH] Memory retention check failed (non-critical): {e}")

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _add_alert(self, alert_type: str, message: str, severity: str = "info") -> None:
        """Add an alert to both alerts deque and activity log."""
        entry = {
            "ts": time.time(),
            "type": alert_type,
            "message": message,
            "severity": severity,
        }
        self.state.alerts.appendleft(entry)
        self.state.activity_log.appendleft(entry)
        logger.info(f"[HEALTH:{severity.upper()}] {alert_type}: {message}")
