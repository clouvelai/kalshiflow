"""
Trade Executor - Pure-Python execution engine for trade intents.

Receives TradeIntent objects from the deep agent (strategist) and handles
all order mechanics: preflight checks, order placement, fill monitoring,
position tracking, and exit execution.

Key design:
- Zero LLM calls - all execution logic is pure Python
- 30-second loop for fast order lifecycle management
- Reports results back to the deep agent via WebSocket for reflection
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .tools import DeepAgentTools
    from .reflection import ReflectionEngine, PendingTrade
    from ..core.websocket_manager import V3WebSocketManager

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.trade_executor")


@dataclass
class TradeIntent:
    """A trade intent submitted by the deep agent strategist."""
    intent_id: str
    event_ticker: str
    market_ticker: str
    side: str                   # "yes" or "no"
    action: str                 # "buy" or "sell"
    contracts: int
    max_price_cents: int        # Maximum acceptable price
    thesis: str                 # Why this trade (stored for reflection)
    exit_criteria: str          # When to exit
    confidence: str             # low/medium/high
    execution_strategy: str     # aggressive/moderate/passive
    created_at: float
    status: str                 # pending | executing | filled | failed | expired
    max_hold_cycles: int = 20   # Auto-expire intent after N executor cycles
    cycles_alive: int = 0       # How many executor loops this intent has survived
    fill_result: Optional[Dict] = None
    failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TradeExecutor:
    """
    Pure-Python trade execution engine. No LLM calls.

    Runs a 30-second loop that:
    1. Drains the intent queue and attempts execution
    2. Monitors active positions for exit criteria
    3. Cleans up expired intents
    4. Reports all events to WebSocket for frontend visibility
    """

    LOOP_INTERVAL = 30  # seconds between executor ticks

    def __init__(
        self,
        tools: 'DeepAgentTools',
        ws_manager: Optional['V3WebSocketManager'] = None,
        reflection_engine: Optional['ReflectionEngine'] = None,
    ):
        self._tools = tools
        self._ws_manager = ws_manager
        self._reflection = reflection_engine
        self._intent_queue: asyncio.Queue[TradeIntent] = asyncio.Queue()
        self._active_intents: Dict[str, TradeIntent] = {}  # intent_id -> TradeIntent
        self._completed_intents: List[TradeIntent] = []     # Recent completed for reporting
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_count = 0
        self._total_intents_received = 0
        self._total_fills = 0
        self._total_failures = 0

    async def start(self) -> None:
        """Start the executor loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._executor_loop())
        logger.info("[trade_executor] Started executor loop (interval=%ds)", self.LOOP_INTERVAL)

    async def stop(self) -> None:
        """Stop the executor loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[trade_executor] Stopped")

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    async def submit_intent(self, intent: TradeIntent) -> None:
        """Called by deep agent's submit_trade_intent tool."""
        self._total_intents_received += 1
        await self._intent_queue.put(intent)
        logger.info(
            "[trade_executor] Intent submitted: %s %s %s x%d @ max %dc (%s)",
            intent.intent_id, intent.action, intent.market_ticker,
            intent.contracts, intent.max_price_cents, intent.confidence,
        )

        # Broadcast to frontend
        if self._ws_manager:
            await self._ws_manager.broadcast_message("trade_executor_intent", {
                "intent_id": intent.intent_id,
                "market_ticker": intent.market_ticker,
                "side": intent.side,
                "action": intent.action,
                "contracts": intent.contracts,
                "max_price_cents": intent.max_price_cents,
                "confidence": intent.confidence,
                "execution_strategy": intent.execution_strategy,
                "thesis": intent.thesis[:200],
                "status": "submitted",
                "timestamp": time.strftime("%H:%M:%S"),
            })

    def get_status_summary(self) -> Dict[str, Any]:
        """Summary for deep agent context injection."""
        pending = []
        try:
            # Peek at queue without consuming
            pending_count = self._intent_queue.qsize()
        except Exception:
            pending_count = 0

        active_summary = []
        for intent in self._active_intents.values():
            active_summary.append({
                "intent_id": intent.intent_id,
                "market_ticker": intent.market_ticker,
                "side": intent.side,
                "action": intent.action,
                "contracts": intent.contracts,
                "status": intent.status,
                "confidence": intent.confidence,
                "cycles_alive": intent.cycles_alive,
                "thesis": intent.thesis[:100],
            })

        recent_completed = []
        for intent in self._completed_intents[-5:]:
            recent_completed.append({
                "intent_id": intent.intent_id,
                "market_ticker": intent.market_ticker,
                "side": intent.side,
                "status": intent.status,
                "failure_reason": intent.failure_reason,
            })

        return {
            "running": self.is_running,
            "tick_count": self._tick_count,
            "pending_in_queue": pending_count,
            "active_intents": active_summary,
            "recent_completed": recent_completed,
            "totals": {
                "intents_received": self._total_intents_received,
                "fills": self._total_fills,
                "failures": self._total_failures,
            },
        }

    # ─── Main Loop ─────────────────────────────────

    async def _executor_loop(self) -> None:
        """Main loop - runs every LOOP_INTERVAL seconds."""
        while self._running:
            try:
                self._tick_count += 1

                # 1. Process new intents from queue
                drained = 0
                while not self._intent_queue.empty():
                    try:
                        intent = self._intent_queue.get_nowait()
                        await self._execute_intent(intent)
                        drained += 1
                    except asyncio.QueueEmpty:
                        break

                # 2. Monitor active positions (check exit criteria)
                await self._monitor_positions()

                # 3. Clean up expired intents
                self._cleanup_expired()

                if drained > 0 or self._active_intents:
                    logger.info(
                        "[trade_executor] Tick %d: drained=%d, active=%d, fills=%d",
                        self._tick_count, drained, len(self._active_intents),
                        self._total_fills,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[trade_executor] Error in executor loop: %s", e, exc_info=True)

            await asyncio.sleep(self.LOOP_INTERVAL)

    # ─── Intent Execution ─────────────────────────────────

    async def _execute_intent(self, intent: TradeIntent) -> None:
        """Preflight -> price check -> trade -> verify."""
        intent.status = "executing"

        try:
            # Run preflight check
            preflight = await self._tools.preflight_check(
                ticker=intent.market_ticker,
                side=intent.side,
                contracts=intent.contracts,
                execution_strategy=intent.execution_strategy,
            )

            if not preflight.get("tradeable"):
                blockers = preflight.get("blockers", [])
                intent.status = "failed"
                intent.failure_reason = f"Preflight blocked: {', '.join(blockers)}" if blockers else "Preflight: not tradeable"
                self._total_failures += 1
                self._completed_intents.append(intent)
                self._broadcast_intent_update(intent)
                logger.info(
                    "[trade_executor] Intent %s failed preflight: %s",
                    intent.intent_id, intent.failure_reason,
                )
                return

            # Check price is within max_price_cents
            estimated_price = preflight.get("estimated_limit_price", 0)
            if estimated_price > intent.max_price_cents:
                intent.status = "failed"
                intent.failure_reason = (
                    f"Price too high: {estimated_price}c > max {intent.max_price_cents}c"
                )
                self._total_failures += 1
                self._completed_intents.append(intent)
                self._broadcast_intent_update(intent)
                logger.info(
                    "[trade_executor] Intent %s price too high: %dc > %dc",
                    intent.intent_id, estimated_price, intent.max_price_cents,
                )
                return

            # Check spread
            spread = preflight.get("spread", 99)
            if spread > 12:
                # Don't fail immediately - retry next tick
                intent.status = "pending"
                intent.cycles_alive += 1
                self._active_intents[intent.intent_id] = intent
                logger.info(
                    "[trade_executor] Intent %s spread too wide (%dc), will retry",
                    intent.intent_id, spread,
                )
                return

            # Execute trade
            result = await self._tools.trade(
                ticker=intent.market_ticker,
                side=intent.side,
                contracts=intent.contracts,
                reasoning=intent.thesis,
                execution_strategy=intent.execution_strategy,
                action=intent.action,
            )

            intent.fill_result = result.to_dict()

            if result.success:
                intent.status = "filled"
                self._total_fills += 1
                self._active_intents[intent.intent_id] = intent
                logger.info(
                    "[trade_executor] Intent %s FILLED: %s %s x%d @ %dc",
                    intent.intent_id, intent.side, intent.market_ticker,
                    intent.contracts, result.price_cents or 0,
                )

                # Record for reflection (same as agent._execute_trade does)
                await self._record_pending_trade(intent, result)
            else:
                intent.status = "failed"
                intent.failure_reason = result.error or "Trade execution failed"
                self._total_failures += 1
                self._completed_intents.append(intent)
                logger.info(
                    "[trade_executor] Intent %s FAILED: %s",
                    intent.intent_id, intent.failure_reason,
                )

            self._broadcast_intent_update(intent)

        except Exception as e:
            intent.status = "failed"
            intent.failure_reason = f"Exception: {str(e)[:200]}"
            self._total_failures += 1
            self._completed_intents.append(intent)
            self._broadcast_intent_update(intent)
            logger.error(
                "[trade_executor] Intent %s exception: %s",
                intent.intent_id, e, exc_info=True,
            )

    async def _record_pending_trade(self, intent: TradeIntent, result) -> None:
        """Record a successful fill as a PendingTrade for reflection tracking."""
        if not self._reflection:
            return

        try:
            from .reflection import PendingTrade

            # Clamp fill price to binary range (1-99c); demo API can return >100c
            raw_price = result.price_cents or result.limit_price_cents or 0
            entry_price = raw_price if 1 <= raw_price <= 99 else (result.limit_price_cents or 50)

            pending = PendingTrade(
                trade_id=f"{intent.market_ticker}:{result.order_id or intent.intent_id}",
                ticker=intent.market_ticker,
                event_ticker=intent.event_ticker,
                side=intent.side,
                contracts=intent.contracts,
                entry_price_cents=entry_price,
                reasoning=intent.thesis,
                timestamp=time.time(),
                order_id=result.order_id or "",
                estimated_probability=None,
                what_could_go_wrong=intent.exit_criteria,
            )
            self._reflection.add_pending_trade(pending)
            logger.info(
                "[trade_executor] Recorded PendingTrade for %s (order_id=%s)",
                intent.market_ticker, result.order_id,
            )
        except Exception as e:
            logger.warning("[trade_executor] Failed to record PendingTrade: %s", e)

    # ─── Position Monitoring ─────────────────────────────────

    async def _monitor_positions(self) -> None:
        """Check active filled intents for exit conditions.

        Simple rules for now:
        - Auto-exit on max_hold_cycles expiry
        - More sophisticated exit logic can be added later
        """
        to_remove = []
        for intent_id, intent in self._active_intents.items():
            intent.cycles_alive += 1

            # Only monitor filled positions (pending intents just age out)
            if intent.status == "filled":
                # Check max hold time
                if intent.cycles_alive >= intent.max_hold_cycles:
                    logger.info(
                        "[trade_executor] Intent %s expired after %d cycles, auto-closing not implemented yet",
                        intent_id, intent.cycles_alive,
                    )
                    # TODO: Auto-exit via sell order when position monitoring is more mature
                    # For now, just mark as completed and let it ride to settlement
                    intent.status = "expired"
                    self._completed_intents.append(intent)
                    to_remove.append(intent_id)

        for intent_id in to_remove:
            del self._active_intents[intent_id]

    # ─── Cleanup ─────────────────────────────────

    def _cleanup_expired(self) -> None:
        """Clean up pending intents that exceeded max_hold_cycles without filling."""
        to_remove = []
        for intent_id, intent in self._active_intents.items():
            if intent.status == "pending" and intent.cycles_alive >= intent.max_hold_cycles:
                intent.status = "expired"
                intent.failure_reason = f"Expired after {intent.cycles_alive} cycles without fill"
                self._completed_intents.append(intent)
                to_remove.append(intent_id)
                logger.info(
                    "[trade_executor] Intent %s expired (pending, %d cycles)",
                    intent_id, intent.cycles_alive,
                )

        for intent_id in to_remove:
            del self._active_intents[intent_id]

        # Cap completed history
        if len(self._completed_intents) > 50:
            self._completed_intents = self._completed_intents[-50:]

    # ─── Broadcasting ─────────────────────────────────

    def _broadcast_intent_update(self, intent: TradeIntent) -> None:
        """Fire-and-forget broadcast of intent status update."""
        if not self._ws_manager:
            return
        asyncio.create_task(self._ws_manager.broadcast_message(
            "trade_executor_update",
            {
                "intent_id": intent.intent_id,
                "market_ticker": intent.market_ticker,
                "side": intent.side,
                "action": intent.action,
                "status": intent.status,
                "failure_reason": intent.failure_reason,
                "fill_result": intent.fill_result,
                "cycles_alive": intent.cycles_alive,
                "timestamp": time.strftime("%H:%M:%S"),
            },
        ))
