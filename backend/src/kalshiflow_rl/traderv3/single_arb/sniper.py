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
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Coroutine, Dict, List, Optional, Set, Tuple

from ..core.events.types import EventType
from .index import ArbOpportunity, EventArbIndex
from .models import AttentionItem

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.sniper")

# Minimum capital floor — prevents Captain from setting max_capital too low
SNIPER_MIN_CAPITAL = 5000  # $50

# Type alias for order registration callback
# Signature: (order_id, ticker, side, action, contracts, price_cents, ttl_seconds) -> None
OrderRegisterCallback = Callable[[str, str, str, str, int, int, int], None]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SniperConfig:
    """Mutable config — Captain can tune these at runtime via configure_sniper()."""

    enabled: bool = False
    max_position: int = 25          # Max contracts per market
    max_capital: int = 100000       # Max capital at risk in cents ($1000)
    cooldown: float = 10.0          # Seconds between trades on same event
    max_trades_per_cycle: int = 5   # Between Captain cycles
    arb_min_edge: float = 3.0       # Min edge cents for S1_ARB
    vpin_reject_threshold: float = 0.98  # VPIN > this → toxic flow, skip (high for Kalshi's thin markets)
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

    # Expected types for each field — used for LLM input coercion
    _FIELD_TYPES: ClassVar[Dict[str, type]] = {
        "enabled": bool, "max_position": int, "max_capital": int,
        "cooldown": float, "max_trades_per_cycle": int, "arb_min_edge": float,
        "vpin_reject_threshold": float, "order_ttl": int, "leg_timeout": float,
        "s4_obi_enabled": bool, "s6_spread_enabled": bool, "s7_sweep_enabled": bool,
    }

    def update(self, **kwargs) -> Tuple[List[str], List[str]]:
        """Partial update. Returns (changed_fields, unknown_fields)."""
        changed = []
        unknown = []
        for k, v in kwargs.items():
            if hasattr(self, k):
                # Coerce LLM-provided values to expected types
                expected = self._FIELD_TYPES.get(k)
                if expected is bool and isinstance(v, str):
                    v = v.lower() in ("true", "1", "yes", "on")
                elif expected and not isinstance(v, expected):
                    try:
                        v = expected(v)
                    except (ValueError, TypeError):
                        logger.warning(f"[SNIPER:CONFIG] Cannot coerce {k}={v!r} to {expected.__name__}, skipped")
                        unknown.append(k)
                        continue
                if getattr(self, k) != v:
                    setattr(self, k, v)
                    changed.append(k)
            else:
                unknown.append(k)
        if unknown:
            logger.warning(f"[SNIPER:CONFIG] Unknown fields ignored: {unknown}")
        # Validation: clamp max_capital to floor
        if self.max_capital < SNIPER_MIN_CAPITAL:
            logger.warning(
                f"[SNIPER:CONFIG] max_capital={self.max_capital}c below floor, "
                f"clamped to {SNIPER_MIN_CAPITAL}c"
            )
            self.max_capital = SNIPER_MIN_CAPITAL
            if "max_capital" not in changed:
                changed.append("max_capital")
        # Validation: clamp max_position to minimum 1
        if self.max_position < 1:
            logger.warning(
                f"[SNIPER:CONFIG] max_position={self.max_position} invalid, clamped to 1"
            )
            self.max_position = 1
            if "max_position" not in changed:
                changed.append("max_position")
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
    order_leg_map: Dict[str, Dict] = field(default_factory=dict)  # order_id -> leg info
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
    capital_active: int = 0  # Cost of sniper orders this session (not yet settled)
    capital_deployed_lifetime: int = 0  # Cumulative audit trail (never decreases)
    # Balance telemetry from API (populated on each risk gate check)
    last_balance_cents: Optional[int] = None
    last_portfolio_value_cents: Optional[int] = None
    # Balance cache (avoid API call on every arb opportunity)
    _cached_balance_cents: Optional[int] = None
    _balance_cached_at: float = 0.0
    active_order_ids: Set[str] = field(default_factory=set)
    order_costs: Dict[str, int] = field(default_factory=dict)  # order_id -> cost in cents
    unhedged_positions: List[Dict] = field(default_factory=list)  # Filled legs from partial arbs
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
            "capital_active": self.capital_active,
            "capital_deployed_lifetime": self.capital_deployed_lifetime,
            "last_balance_cents": self.last_balance_cents,
            "last_portfolio_value_cents": self.last_portfolio_value_cents,
            "active_orders": len(self.active_order_ids),
            "unhedged_positions": self.unhedged_positions,
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
        order_register_callback: Optional[OrderRegisterCallback] = None,
        attention_callback: Optional[Callable] = None,
    ):
        self._gateway = gateway
        self._index = index
        self._event_bus = event_bus
        self._session = session  # TradingSession (shared with Captain tools)
        self.config = config
        self._broadcast = broadcast_callback
        self._register_order = order_register_callback
        self._attention_callback = attention_callback

        self.state = SniperState()
        self._running = False
        self._paused = False

        # Throttle SNIPER:SCALE log spam: maps event_ticker -> (last_scaled_value, last_log_time)
        self._last_scale_log: Dict[str, tuple] = {}

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

    def pause(self, reason: str = "") -> None:
        """Pause sniper execution. Called by AutoActionManager on toxic regime."""
        self._paused = True
        logger.info(f"[SNIPER:PAUSE] {reason}")

    def resume(self) -> None:
        """Resume sniper execution after pause."""
        self._paused = False
        logger.info("[SNIPER:RESUME]")

    @property
    def is_paused(self) -> bool:
        return self._paused

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

        if self._paused:
            return None

        # Risk gates (async — queries real API balance, returns scaled contracts)
        rejection, approved_contracts = await self._check_risk_gates(opportunity)
        if rejection:
            self.state.total_arbs_rejected += 1
            self.state.last_rejection_reason = rejection
            logger.debug(f"[SNIPER:S1_ARB] Rejected {opportunity.event_ticker}: {rejection}")

            # Emit attention for notable rejections (capital limit, VPIN toxic)
            if self._attention_callback and (
                "sniper_capital_limit" in rejection or "vpin_toxic" in rejection
            ):
                self._attention_callback(AttentionItem(
                    event_ticker=opportunity.event_ticker,
                    category="sniper_rejection",
                    urgency="normal",
                    summary=f"sniper rejected: {rejection}",
                    score=40.0,
                    data={
                        "rejection_reason": rejection,
                        "edge_cents": opportunity.edge_after_fees,
                        "capital_active": self.state.capital_active,
                        "total_rejections": self.state.total_arbs_rejected,
                    },
                    ttl_seconds=60.0,
                ))

            return None

        # Check liquidity-adjusted edge (walk depth levels, not just BBO)
        event = self._index.events.get(opportunity.event_ticker)
        if event:
            la = event.liquidity_adjusted_edge(
                direction=opportunity.direction,
                target_contracts=approved_contracts,
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

        # Execute with pre-approved contract count
        action = await self._execute_arb(opportunity, approved_contracts)
        return action

    # ------------------------------------------------------------------
    # Risk gates
    # ------------------------------------------------------------------

    async def _check_risk_gates(self, opportunity: ArbOpportunity) -> Tuple[Optional[str], int]:
        """Check all risk gates with trade size scaling.

        Returns (rejection_reason, approved_contracts).
        If rejected, approved_contracts is 0.
        If approved, approved_contracts is the scaled contract count per leg.
        """

        # 1. Per-cycle trade limit
        if self.state.trades_this_cycle >= self.config.max_trades_per_cycle:
            return f"cycle_limit ({self.config.max_trades_per_cycle})", 0

        # 2. Edge threshold (BBO-level quick check before depth walk)
        if opportunity.edge_after_fees < self.config.arb_min_edge:
            return f"edge={opportunity.edge_after_fees:.1f}c < min={self.config.arb_min_edge}c", 0

        # 3. Per-event cooldown (skip check on first trade for this event)
        et = opportunity.event_ticker
        if et in self.state._event_last_trade:
            elapsed = time.time() - self.state._event_last_trade[et]
            if elapsed < self.config.cooldown:
                return f"cooldown ({elapsed:.0f}s < {self.config.cooldown}s)", 0

        # 4. Balance + capital scaling (scale down instead of all-or-nothing reject)
        liquidity_contracts = self.config.max_position
        for leg in opportunity.legs:
            liquidity_contracts = min(liquidity_contracts, leg.size_available)
        liquidity_contracts = max(1, liquidity_contracts)

        total_price_per_contract = sum(leg.price_cents for leg in opportunity.legs)
        if total_price_per_contract <= 0:
            return "invalid_price (total_price_per_contract=0)", 0

        try:
            now = time.time()
            if (self.state._cached_balance_cents is not None
                    and now - self.state._balance_cached_at < 2.0):
                balance_cents = self.state._cached_balance_cents
                portfolio_value = self.state.last_portfolio_value_cents or 0
            else:
                balance = await self._gateway.get_balance()
                balance_cents = balance.balance
                portfolio_value = balance.portfolio_value
                self.state._cached_balance_cents = balance_cents
                self.state._balance_cached_at = now
                self.state.last_portfolio_value_cents = portfolio_value

            self.state.last_balance_cents = balance_cents

            capital_headroom = self.config.max_capital - self.state.capital_active
            if capital_headroom <= 0:
                return (
                    f"sniper_capital_limit (active={self.state.capital_active}c "
                    f">= max={self.config.max_capital}c)"
                ), 0

            max_from_capital = capital_headroom // total_price_per_contract
            max_from_balance = balance_cents // total_price_per_contract

            contracts_per_leg = min(liquidity_contracts, max_from_capital, max_from_balance)

            if contracts_per_leg < 1:
                if max_from_balance < 1:
                    return (
                        f"insufficient_balance (available={balance_cents}c, "
                        f"min_cost={total_price_per_contract}c)"
                    ), 0
                return (
                    f"sniper_capital_limit (headroom={capital_headroom}c "
                    f"< min_cost={total_price_per_contract}c)"
                ), 0

            # Log when scaling occurs (throttled: only on value change or every 60s per event)
            if contracts_per_leg < liquidity_contracts:
                et_key = opportunity.event_ticker
                now_ts = time.time()
                prev = self._last_scale_log.get(et_key)
                should_log = (
                    prev is None
                    or prev[0] != contracts_per_leg
                    or (now_ts - prev[1]) >= 60.0
                )
                if should_log:
                    self._last_scale_log[et_key] = (contracts_per_leg, now_ts)
                    logger.info(
                        f"[SNIPER:SCALE] {opportunity.event_ticker} scaled "
                        f"{liquidity_contracts} -> {contracts_per_leg} contracts/leg "
                        f"(headroom={capital_headroom}c, balance={balance_cents}c)"
                    )
        except Exception as e:
            logger.warning(f"[SNIPER:GATE] Balance API failed, rejecting conservatively: {e}")
            return f"balance_api_error ({e})", 0

        # 5. VPIN toxicity check (any market with VPIN > threshold)
        event = self._index.events.get(et)
        if event:
            for m in event.markets.values():
                if m.micro.vpin > self.config.vpin_reject_threshold:
                    return f"vpin_toxic ({m.ticker}: vpin={m.micro.vpin:.2f} > {self.config.vpin_reject_threshold})", 0

        # 6. Mutually exclusive check (S1_ARB only works on ME events)
        if event and not event.mutually_exclusive:
            return "not_mutually_exclusive", 0

        return None, contracts_per_leg

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute_arb(
        self, opportunity: ArbOpportunity, contracts_per_leg: Optional[int] = None
    ) -> SniperAction:
        """Execute parallel-leg arb via gateway.

        Places all legs concurrently with asyncio.gather().
        Tracks order IDs in session for attribution.
        On partial fill (some legs succeed, some fail), unwinds successful legs
        by cancelling them to avoid unhedged directional exposure.

        Args:
            opportunity: The arb opportunity to execute.
            contracts_per_leg: Pre-approved contract count from risk gates.
                If None, falls back to computing from config and liquidity.
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

        # Use pre-approved count if provided, otherwise compute from scratch (fallback)
        if contracts_per_leg is None:
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
                leg = opportunity.legs[i]
                action.order_ids.append(order_id)
                action.order_leg_map[order_id] = {
                    "ticker": leg.ticker,
                    "side": leg.side,
                    "contracts": contracts_per_leg,
                    "price_cents": leg.price_cents,
                }
                total_cost += cost
                filled += 1
                successful_order_ids.append(order_id)

                # Track in session for attribution
                self._session.sniper_order_ids.add(order_id)
                self.state.active_order_ids.add(order_id)
                self.state.order_costs[order_id] = cost

                # Register with Captain's order tracking (enables polling, capital release, fills)
                if self._register_order:
                    self._register_order(
                        order_id, leg.ticker, leg.side, leg.action,
                        contracts_per_leg, leg.price_cents, self.config.order_ttl,
                    )

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
            cancelled = await self._unwind_legs(successful_order_ids, opportunity, action)
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
            self.state.capital_active += total_cost  # Session-scoped active capital
            self.state.capital_deployed_lifetime += total_cost  # Audit trail only
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

        # Emit attention item for Captain awareness (successful executions only)
        if self._attention_callback and filled == len(opportunity.legs):
            capital_pct = (self.state.capital_active / self.config.max_capital * 100) if self.config.max_capital > 0 else 0
            urgency = "high" if capital_pct > 70 else "normal"
            self._attention_callback(AttentionItem(
                event_ticker=opportunity.event_ticker,
                category="sniper_execution",
                urgency=urgency,
                summary=(
                    f"sniper {opportunity.direction} arb {filled}/{len(opportunity.legs)} legs "
                    f"edge={opportunity.edge_after_fees:.1f}c cost={total_cost}c "
                    f"capital={capital_pct:.0f}%"
                ),
                score=55.0 if urgency == "high" else 45.0,
                data={
                    "direction": opportunity.direction,
                    "edge_cents": opportunity.edge_after_fees,
                    "legs_filled": filled,
                    "total_cost": total_cost,
                    "capital_active": self.state.capital_active,
                    "capital_pct": round(capital_pct, 1),
                    "total_trades": self.state.total_trades,
                },
                ttl_seconds=90.0,
            ))

        return action

    async def _unwind_legs(self, order_ids: List[str], opportunity: "ArbOpportunity" = None, action: "SniperAction" = None) -> int:
        """Cancel or track successfully placed legs after partial fill.

        For each order: check if it has already filled. If resting, cancel it.
        If already filled, track as an unhedged position (Captain must exit manually).
        Returns count of orders successfully cancelled.
        """
        cancelled = 0
        for oid in order_ids:
            # Try to cancel first — if order already filled, cancel will fail
            success = await self._cancel_leg_safe(oid)
            if success:
                cancelled += 1
            else:
                # Order couldn't be cancelled — likely already filled
                # Track as unhedged position for Captain visibility
                leg_info = self._find_leg_for_order(oid, opportunity, action)
                self.state.unhedged_positions.append({
                    "order_id": oid,
                    "ticker": leg_info.get("ticker", ""),
                    "side": leg_info.get("side", ""),
                    "contracts": leg_info.get("contracts", 0),
                    "price_cents": leg_info.get("price_cents", 0),
                    "event_ticker": opportunity.event_ticker if opportunity else "",
                    "reason": "partial_fill_unwind_failed",
                    "ts": time.time(),
                })
                logger.warning(
                    f"[SNIPER:UNWIND] Order {oid[:8]}... already filled — "
                    f"tracked as unhedged position ({leg_info.get('ticker', '?')})"
                )
        return cancelled

    def _find_leg_for_order(self, order_id: str, opportunity: "ArbOpportunity" = None, action: "SniperAction" = None) -> Dict:
        """Find leg details for an order ID.

        Checks the action's order_leg_map first (populated during _execute_arb).
        Falls back to opportunity legs if map not available.
        """
        # Primary: lookup from action's order_leg_map
        if action and order_id in action.order_leg_map:
            return action.order_leg_map[order_id]

        # Check recent actions for the mapping
        for recent in reversed(self.state.recent_actions):
            if order_id in recent.order_leg_map:
                return recent.order_leg_map[order_id]

        # Fallback: return empty if no opportunity
        if not opportunity or not opportunity.legs:
            return {}

        # Last resort: return first leg info (legacy behavior)
        leg = opportunity.legs[0]
        return {"ticker": leg.ticker, "side": leg.side, "contracts": 0, "price_cents": leg.price_cents}

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
                self.state.active_order_ids.discard(order_id)
            except Exception as e:
                errors.append(f"{order_id[:8]}: {e}")
                self.state.active_order_ids.discard(order_id)

        # Reset active capital since all orders are cancelled/gone
        self.state.capital_active = 0

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
    # Stale event pruning
    # ------------------------------------------------------------------

    def prune_stale_events(self, active_event_tickers: Set[str]) -> int:
        """Remove entries from event-keyed dicts for events no longer active.

        Prevents unbounded growth of _event_last_trade and _last_scale_log
        over weeks of operation.

        Args:
            active_event_tickers: Set of event tickers currently in the index.

        Returns:
            Number of stale entries removed.
        """
        removed = 0

        stale_events = set(self.state._event_last_trade.keys()) - active_event_tickers
        for et in stale_events:
            del self.state._event_last_trade[et]
            removed += 1

        stale_scale = set(self._last_scale_log.keys()) - active_event_tickers
        for et in stale_scale:
            del self._last_scale_log[et]
            removed += 1

        if removed > 0:
            logger.info(f"[SNIPER:PRUNE] Removed {removed} stale event entries")

        return removed

    # ------------------------------------------------------------------
    # Capital reconciliation
    # ------------------------------------------------------------------

    def reconcile_capital(self, resting_order_ids: Set[str]) -> None:
        """Reconcile capital_active against actually resting orders.

        Orders that expired via TTL, settled, or were cancelled by exchange
        cause permanent upward drift in capital_active. This method corrects
        that by recalculating from only still-resting sniper orders.

        Args:
            resting_order_ids: Set of order IDs currently resting on the exchange.
        """
        # Find sniper orders that are no longer resting
        gone = self.state.active_order_ids - resting_order_ids
        if not gone:
            return

        old_capital = self.state.capital_active

        # Remove gone orders from tracking
        for oid in gone:
            self.state.active_order_ids.discard(oid)
            self.state.order_costs.pop(oid, None)

        # Recalculate capital_active from remaining orders
        self.state.capital_active = sum(
            self.state.order_costs.get(oid, 0)
            for oid in self.state.active_order_ids
        )

        new_capital = self.state.capital_active
        if old_capital != new_capital:
            logger.info(
                f"SNIPER reconcile_capital: was={old_capital}, now={new_capital}, "
                f"released={old_capital - new_capital}"
            )

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
