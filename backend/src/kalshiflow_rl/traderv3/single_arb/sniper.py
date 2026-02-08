"""Sniper - Sub-second execution layer for market rebalancing arbitrage.

Targets the $29M/year S1_ARB market rebalancing opportunity identified in
"Unravelling the Probabilistic Forest" (arXiv:2508.03474v1):
  Buy all YES < $1 (long arb) or buy all NO (short arb) on mutually exclusive events.

Architecture:
  - Receives ArbOpportunity from EventArbIndex via coordinator callback
  - Validates through risk gates (position, capital, cooldown, VPIN toxicity)
  - Executes parallel legs via KalshiGateway.create_order()
  - Tracks fills and P&L in SniperState telemetry
  - Captain configures/monitors via tools (configure_sniper, get_sniper_status)

Design principles:
  - Hot path is on_arb_opportunity() → _check_risk_gates() → parallel execute
  - No LLM calls on the hot path — pure Python + async I/O
  - Captain owns strategy; Sniper owns execution speed
  - All directional strategies (S4/S6/S7) disabled by default, Captain enables

Concurrency note:
  All methods run on a single asyncio event loop. No locking is needed because
  Python's GIL + cooperative async scheduling guarantee no concurrent mutation
  of Sniper state within a single coroutine step.
"""

import asyncio
import collections
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple

from ..core.events.types import EventType
from .index import ArbOpportunity, EventArbIndex

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.sniper")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SniperConfig:
    """Mutable config — Captain can tune these at runtime via configure_sniper()."""

    enabled: bool = False
    max_position: int = 25          # Max contracts per market
    max_capital: int = 5000         # Max capital at risk in cents ($50)
    cooldown: float = 10.0          # Seconds between trades on same event
    max_trades_per_cycle: int = 5   # Between Captain cycles
    arb_min_edge: float = 3.0       # Min edge cents for S1_ARB
    vpin_reject_threshold: float = 0.7  # VPIN > this → toxic flow, skip
    order_ttl: int = 30             # Order TTL in seconds
    leg_timeout: float = 5.0        # Timeout per leg placement in seconds

    # Directional strategies (disabled by default, Captain enables)
    s4_obi_enabled: bool = False    # OBI momentum
    s6_spread_enabled: bool = False # Spread capture
    s7_sweep_enabled: bool = False  # Sweep following

    def to_dict(self) -> Dict:
        return {
            "enabled": self.enabled,
            "max_position": self.max_position,
            "max_capital": self.max_capital,
            "cooldown": self.cooldown,
            "max_trades_per_cycle": self.max_trades_per_cycle,
            "arb_min_edge": self.arb_min_edge,
            "vpin_reject_threshold": self.vpin_reject_threshold,
            "order_ttl": self.order_ttl,
            "leg_timeout": self.leg_timeout,
            "s4_obi_enabled": self.s4_obi_enabled,
            "s6_spread_enabled": self.s6_spread_enabled,
            "s7_sweep_enabled": self.s7_sweep_enabled,
        }

    def update(self, **kwargs) -> Tuple[List[str], List[str]]:
        """Partial update. Returns (changed_fields, unknown_fields)."""
        changed = []
        unknown = []
        for k, v in kwargs.items():
            if hasattr(self, k):
                if getattr(self, k) != v:
                    setattr(self, k, v)
                    changed.append(k)
            else:
                unknown.append(k)
        if unknown:
            logger.warning(f"[SNIPER:CONFIG] Unknown fields ignored: {unknown}")
        return changed, unknown


@dataclass
class SniperAction:
    """Record of a single Sniper execution attempt."""

    timestamp: float = field(default_factory=time.time)
    event_ticker: str = ""
    direction: str = ""             # "long" or "short"
    strategy: str = "S1_ARB"       # Strategy tag
    edge_cents: float = 0.0
    edge_net_cents: float = 0.0     # Edge after estimated fees
    legs_attempted: int = 0
    legs_filled: int = 0
    order_ids: List[str] = field(default_factory=list)
    total_cost_cents: int = 0
    error: Optional[str] = None
    error_type: Optional[str] = None  # "timeout", "api_error", "rate_limit", "partial_unwind"
    latency_ms: float = 0.0
    unwound: bool = False           # True if partial fill was unwound

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "event_ticker": self.event_ticker,
            "direction": self.direction,
            "strategy": self.strategy,
            "edge_cents": round(self.edge_cents, 2),
            "edge_net_cents": round(self.edge_net_cents, 2),
            "legs_attempted": self.legs_attempted,
            "legs_filled": self.legs_filled,
            "order_ids": self.order_ids,
            "total_cost_cents": self.total_cost_cents,
            "error": self.error,
            "error_type": self.error_type,
            "latency_ms": round(self.latency_ms, 1),
            "unwound": self.unwound,
        }


@dataclass
class SniperState:
    """Read-only telemetry for Captain and health monitor."""

    running: bool = False
    trades_this_cycle: int = 0
    total_trades: int = 0
    total_arbs_detected: int = 0
    total_arbs_executed: int = 0
    total_arbs_rejected: int = 0
    total_partial_unwinds: int = 0
    # Capital lifecycle accounting
    capital_in_flight: int = 0          # Cents in resting orders (not yet filled/expired)
    capital_in_positions: int = 0       # Cents in filled positions
    capital_deployed_lifetime: int = 0  # Cumulative audit trail (never decreases)
    active_order_ids: Set[str] = field(default_factory=set)
    last_trade_at: Optional[float] = None
    last_rejection_reason: Optional[str] = None
    recent_actions: collections.deque = field(default_factory=lambda: collections.deque(maxlen=20))
    # Per-event cooldown tracking
    _event_last_trade: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "running": self.running,
            "trades_this_cycle": self.trades_this_cycle,
            "total_trades": self.total_trades,
            "total_arbs_detected": self.total_arbs_detected,
            "total_arbs_executed": self.total_arbs_executed,
            "total_arbs_rejected": self.total_arbs_rejected,
            "total_partial_unwinds": self.total_partial_unwinds,
            "capital_in_flight": self.capital_in_flight,
            "capital_in_positions": self.capital_in_positions,
            "capital_deployed_lifetime": self.capital_deployed_lifetime,
            "capital_active": self.capital_in_flight + self.capital_in_positions,
            "active_orders": len(self.active_order_ids),
            "last_trade_at": self.last_trade_at,
            "last_rejection_reason": self.last_rejection_reason,
            "recent_actions": [a.to_dict() for a in list(self.recent_actions)[-10:]],
        }


# ---------------------------------------------------------------------------
# Sniper class
# ---------------------------------------------------------------------------

class Sniper:
    """Sub-second execution layer for S1_ARB market rebalancing.

    Uses KalshiGateway directly for minimum-latency order placement.
    Shares the coordinator's TradingSession for order ID tracking.
    """

    def __init__(
        self,
        gateway,
        index: EventArbIndex,
        event_bus,
        session,
        config: SniperConfig,
        broadcast_callback: Optional[Callable[..., Coroutine]] = None,
    ):
        self._gateway = gateway
        self._index = index
        self._event_bus = event_bus
        self._session = session  # TradingSession (shared with Captain tools)
        self.config = config
        self._broadcast = broadcast_callback

        self.state = SniperState()
        self._running = False

        # Per-order capital tracking: order_id -> (cost_cents, placed_at_ts)
        self._order_capital: Dict[str, Tuple[int, float]] = {}

    async def start(self) -> None:
        """Subscribe to EventBus and mark as running."""
        if self._running:
            return
        self._running = True
        self.state.running = True
        logger.info(
            f"[SNIPER] Started (enabled={self.config.enabled}, "
            f"arb_edge>{self.config.arb_min_edge}c, "
            f"max_pos={self.config.max_position}, "
            f"ttl={self.config.order_ttl}s)"
        )

    async def stop(self) -> None:
        """Unsubscribe and stop."""
        self._running = False
        self.state.running = False
        logger.info(
            f"[SNIPER] Stopped (trades={self.state.total_trades}, "
            f"arbs={self.state.total_arbs_executed})"
        )

    def reset_cycle_counter(self) -> None:
        """Called at start of each Captain cycle to reset per-cycle limits."""
        self.state.trades_this_cycle = 0

    # ------------------------------------------------------------------
    # Capital lifecycle management
    # ------------------------------------------------------------------

    def _track_order_capital(self, order_id: str, cost_cents: int) -> None:
        """Track capital for a newly placed order (in-flight)."""
        self._order_capital[order_id] = (cost_cents, time.time())
        self.state.capital_in_flight += cost_cents
        self.state.capital_deployed_lifetime += cost_cents

    def _release_order_capital(self, order_id: str, reason: str) -> int:
        """Release capital for a cancelled/expired order. Returns cents released."""
        entry = self._order_capital.pop(order_id, None)
        if not entry:
            return 0
        cost_cents, _ = entry
        self.state.capital_in_flight = max(0, self.state.capital_in_flight - cost_cents)
        self.state.active_order_ids.discard(order_id)
        logger.debug(f"[SNIPER:CAPITAL] Released {cost_cents}c for {order_id[:8]}... ({reason})")
        return cost_cents

    def _promote_order_to_position(self, order_id: str) -> None:
        """Move capital from in-flight to in-positions (order filled)."""
        entry = self._order_capital.pop(order_id, None)
        if not entry:
            return
        cost_cents, _ = entry
        self.state.capital_in_flight = max(0, self.state.capital_in_flight - cost_cents)
        self.state.capital_in_positions += cost_cents

    def _cleanup_stale_orders(self) -> int:
        """Remove orders past TTL + 10s buffer. Returns count cleaned."""
        now = time.time()
        ttl_with_buffer = self.config.order_ttl + 10
        stale_ids = [
            oid for oid, (_, placed_at) in self._order_capital.items()
            if (now - placed_at) > ttl_with_buffer
        ]
        for oid in stale_ids:
            self._release_order_capital(oid, "ttl_expired")
        if stale_ids:
            logger.info(f"[SNIPER:CLEANUP] Released {len(stale_ids)} stale orders")
        return len(stale_ids)

    # ------------------------------------------------------------------
    # Hot path: arb opportunity handler
    # ------------------------------------------------------------------

    async def on_arb_opportunity(self, opportunity: ArbOpportunity) -> Optional[SniperAction]:
        """S1_ARB hot path. Called by coordinator when index detects arb.

        Returns SniperAction on execution attempt, None if rejected.
        """
        self.state.total_arbs_detected += 1

        if not self.config.enabled or not self._running:
            return None

        # Cleanup stale orders before checking capital gate
        self._cleanup_stale_orders()

        # Risk gates
        rejection = self._check_risk_gates(opportunity)
        if rejection:
            self.state.total_arbs_rejected += 1
            self.state.last_rejection_reason = rejection
            logger.debug(f"[SNIPER:S1_ARB] Rejected {opportunity.event_ticker}: {rejection}")
            return None

        # Check liquidity-adjusted edge (walk depth levels, not just BBO)
        event = self._index.events.get(opportunity.event_ticker)
        if event:
            la = event.liquidity_adjusted_edge(
                direction=opportunity.direction,
                target_contracts=min(self.config.max_position, 10),
            )
            if "error" in la:
                self.state.total_arbs_rejected += 1
                self.state.last_rejection_reason = f"liquidity_check_failed: {la['error']}"
                return None
            if la["edge_per_contract"] < self.config.arb_min_edge:
                self.state.total_arbs_rejected += 1
                self.state.last_rejection_reason = (
                    f"liq_adj_edge={la['edge_per_contract']:.1f}c < min={self.config.arb_min_edge}c"
                )
                logger.debug(
                    f"[SNIPER:S1_ARB] Liquidity-adjusted edge too low: "
                    f"{la['edge_per_contract']:.1f}c (BBO edge={opportunity.edge_after_fees:.1f}c)"
                )
                return None

        # Execute
        action = await self._execute_arb(opportunity)
        return action

    # ------------------------------------------------------------------
    # Risk gates
    # ------------------------------------------------------------------

    def _check_risk_gates(self, opportunity: ArbOpportunity) -> Optional[str]:
        """Check all risk gates. Returns rejection reason or None if clear."""

        # 1. Per-cycle trade limit
        if self.state.trades_this_cycle >= self.config.max_trades_per_cycle:
            return f"cycle_limit ({self.config.max_trades_per_cycle})"

        # 2. Edge threshold (BBO-level quick check before depth walk)
        if opportunity.edge_after_fees < self.config.arb_min_edge:
            return f"edge={opportunity.edge_after_fees:.1f}c < min={self.config.arb_min_edge}c"

        # 3. Per-event cooldown
        et = opportunity.event_ticker
        last_trade = self.state._event_last_trade.get(et, 0)
        elapsed = time.time() - last_trade
        if elapsed < self.config.cooldown:
            return f"cooldown ({elapsed:.0f}s < {self.config.cooldown}s)"

        # 4. Capital limit (lifecycle-aware: only count active capital)
        contracts_per_leg = self.config.max_position
        for leg in opportunity.legs:
            contracts_per_leg = min(contracts_per_leg, leg.size_available)
        contracts_per_leg = max(1, contracts_per_leg)
        est_cost = sum(leg.price_cents for leg in opportunity.legs) * contracts_per_leg
        active_capital = self.state.capital_in_flight + self.state.capital_in_positions
        if active_capital + est_cost > self.config.max_capital:
            return (
                f"capital_limit (active={active_capital}c "
                f"[flight={self.state.capital_in_flight}c + pos={self.state.capital_in_positions}c] "
                f"+ est={est_cost}c > max={self.config.max_capital}c)"
            )

        # 5. VPIN toxicity check (any market with VPIN > threshold)
        event = self._index.events.get(et)
        if event:
            for m in event.markets.values():
                if m.micro.vpin > self.config.vpin_reject_threshold:
                    return f"vpin_toxic ({m.ticker}: vpin={m.micro.vpin:.2f} > {self.config.vpin_reject_threshold})"

        # 6. Mutually exclusive check (S1_ARB only works on ME events)
        if event and not event.mutually_exclusive:
            return "not_mutually_exclusive"

        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute_arb(self, opportunity: ArbOpportunity) -> SniperAction:
        """Execute parallel-leg arb via gateway.

        Places all legs concurrently with asyncio.gather().
        Tracks order IDs in session for attribution.
        On partial fill (some legs succeed, some fail), unwinds successful legs
        by cancelling them to avoid unhedged directional exposure.
        """
        action = SniperAction(
            event_ticker=opportunity.event_ticker,
            direction=opportunity.direction,
            strategy="S1_ARB",
            edge_cents=opportunity.edge_after_fees,
            edge_net_cents=opportunity.edge_after_fees,  # Same for now; extend if fee model changes
            legs_attempted=len(opportunity.legs),
        )
        start = time.time()

        # Determine contract count per leg (min of: config max, available size)
        contracts_per_leg = self.config.max_position
        for leg in opportunity.legs:
            contracts_per_leg = min(contracts_per_leg, leg.size_available)
        contracts_per_leg = max(1, contracts_per_leg)

        # Compute expiration timestamp from config TTL
        expiration_ts = int(time.time()) + self.config.order_ttl

        # Build order coroutines with per-leg timeout
        order_coros = []
        for leg in opportunity.legs:
            coro = asyncio.wait_for(
                self._place_leg(
                    ticker=leg.ticker,
                    side=leg.side,
                    action=leg.action,
                    price_cents=leg.price_cents,
                    contracts=contracts_per_leg,
                    expiration_ts=expiration_ts,
                ),
                timeout=self.config.leg_timeout,
            )
            order_coros.append(coro)

        # Execute all legs in parallel
        results = await asyncio.gather(*order_coros, return_exceptions=True)

        # Process results
        filled = 0
        total_cost = 0
        successful_order_ids = []
        for i, result in enumerate(results):
            if isinstance(result, asyncio.TimeoutError):
                logger.warning(
                    f"[SNIPER:S1_ARB] Leg {i} TIMEOUT for {opportunity.event_ticker} "
                    f"(>{self.config.leg_timeout}s)"
                )
                if not action.error:
                    action.error = f"leg_{i}_timeout"
                    action.error_type = "timeout"
            elif isinstance(result, Exception):
                error_type = "api_error"
                error_str = str(result)
                # Distinguish known error types
                if "rate" in error_str.lower() or "429" in error_str:
                    error_type = "rate_limit"
                logger.warning(
                    f"[SNIPER:S1_ARB] Leg {i} failed ({error_type}) for "
                    f"{opportunity.event_ticker}: {result}"
                )
                if not action.error:
                    action.error = error_str
                    action.error_type = error_type
            elif result:
                order_id, cost = result
                action.order_ids.append(order_id)
                total_cost += cost
                filled += 1
                successful_order_ids.append(order_id)

                # Track in session and capital lifecycle
                self._session.sniper_order_ids.add(order_id)
                self.state.active_order_ids.add(order_id)
                self._track_order_capital(order_id, cost)

        action.legs_filled = filled
        action.total_cost_cents = total_cost
        action.latency_ms = (time.time() - start) * 1000

        # Partial fill unwinding: if some legs filled but not all, cancel successful legs
        # to avoid unhedged directional exposure
        if 0 < filled < len(opportunity.legs):
            logger.warning(
                f"[SNIPER:S1_ARB] PARTIAL {opportunity.event_ticker} {opportunity.direction} "
                f"legs={filled}/{len(opportunity.legs)} — UNWINDING successful legs"
            )
            cancelled = await self._unwind_legs(successful_order_ids)
            action.unwound = True
            action.error = f"partial_fill_unwound (cancelled={cancelled}/{filled})"
            action.error_type = "partial_unwind"
            self.state.total_partial_unwinds += 1
            # Don't count as executed
            self.state.total_arbs_rejected += 1
        elif filled == len(opportunity.legs):
            # Update state for full execution
            self.state.trades_this_cycle += 1
            self.state.total_trades += 1
            self.state.total_arbs_executed += 1
            self.state.last_trade_at = time.time()
            self.state._event_last_trade[opportunity.event_ticker] = time.time()
            logger.info(
                f"[SNIPER:S1_ARB] EXECUTED {opportunity.event_ticker} {opportunity.direction} "
                f"edge={opportunity.edge_after_fees:.1f}c legs={filled}/{len(opportunity.legs)} "
                f"cost={total_cost}c latency={action.latency_ms:.0f}ms"
            )
        else:
            self.state.total_arbs_rejected += 1
            logger.warning(
                f"[SNIPER:S1_ARB] FAILED {opportunity.event_ticker}: all legs failed"
            )

        # Record action (deque auto-evicts oldest)
        self.state.recent_actions.append(action)

        # Broadcast to frontend (fire-and-forget to avoid blocking hot path)
        if self._broadcast:
            asyncio.create_task(self._broadcast({
                "type": "sniper_execution",
                "data": action.to_dict(),
            }))

        return action

    async def _unwind_legs(self, order_ids: List[str]) -> int:
        """Cancel successfully placed legs after partial fill. Returns count cancelled."""
        cancel_coros = [
            self._cancel_leg_safe(oid) for oid in order_ids
        ]
        results = await asyncio.gather(*cancel_coros, return_exceptions=True)
        cancelled = 0
        for oid, result in zip(order_ids, results):
            if isinstance(result, Exception):
                logger.warning(f"[SNIPER:UNWIND] Failed to cancel {oid[:8]}...: {result}")
            elif result:
                cancelled += 1
                self._release_order_capital(oid, "partial_unwind")
        return cancelled

    async def _cancel_leg_safe(self, order_id: str) -> bool:
        """Cancel a single order with timeout. Returns True on success."""
        try:
            await asyncio.wait_for(
                self._gateway.cancel_order(order_id),
                timeout=5.0,
            )
            self.state.active_order_ids.discard(order_id)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"[SNIPER:UNWIND] Cancel timeout for {order_id[:8]}...")
            return False
        except Exception as e:
            # Order may have already filled or expired — not an error
            logger.debug(f"[SNIPER:UNWIND] Cancel failed for {order_id[:8]}...: {e}")
            self.state.active_order_ids.discard(order_id)
            return False

    async def _place_leg(
        self,
        ticker: str,
        side: str,
        action: str,
        price_cents: int,
        contracts: int,
        expiration_ts: int,
    ) -> Optional[tuple]:
        """Place a single order leg via gateway.

        Returns (order_id, cost_cents) on success, None on failure.
        Raises on gateway error (caught by asyncio.gather return_exceptions=True).
        """
        order_resp = await self._gateway.create_order(
            ticker=ticker,
            action=action,
            side=side,
            count=contracts,
            price=price_cents,
            type="limit",
            order_group_id=self._session.order_group_id or None,
            expiration_ts=expiration_ts,
        )
        order_id = order_resp.order_id if hasattr(order_resp, "order_id") else order_resp.order.order_id
        cost = price_cents * contracts
        logger.debug(
            f"[SNIPER:LEG] {ticker} {action} {side} {contracts}@{price_cents}c "
            f"→ {order_id[:8]}..."
        )
        return (order_id, cost)

    # ------------------------------------------------------------------
    # Emergency liquidation
    # ------------------------------------------------------------------

    async def kill_positions(self, reason: str = "captain_request") -> Dict:
        """Emergency: cancel all active sniper orders.

        Called by Captain via kill_sniper_positions() tool.
        Releases capital for all cancelled orders.
        """
        cancelled = 0
        errors = []
        for order_id in list(self.state.active_order_ids):
            try:
                await self._gateway.cancel_order(order_id)
                cancelled += 1
                self._release_order_capital(order_id, reason)
            except Exception as e:
                errors.append(f"{order_id[:8]}: {e}")
                # Still release capital — order likely already filled/expired
                self._release_order_capital(order_id, f"{reason}_error")

        logger.info(
            f"[SNIPER:KILL] Cancelled {cancelled} orders (reason={reason}, "
            f"errors={len(errors)})"
        )

        return {
            "cancelled": cancelled,
            "errors": errors,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_health_details(self) -> Dict:
        """Health details for the coordinator health endpoint."""
        return {
            "running": self._running,
            "enabled": self.config.enabled,
            "config": self.config.to_dict(),
            "state": self.state.to_dict(),
        }
