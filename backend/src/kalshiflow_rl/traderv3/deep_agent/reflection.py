"""
Deep Agent Reflection Engine - Learning from trade outcomes.

The reflection engine monitors trade settlements and triggers
the agent to reflect on outcomes, extract learnings, and update
memory files for continuous improvement.

Key responsibilities:
- Track pending trades awaiting settlement
- Detect when trades settle (win/lose)
- Generate reflection prompts with full context
- Parse and store extracted learnings
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.websocket_manager import V3WebSocketManager
    from ..core.state_container import V3StateContainer

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.reflection")


@dataclass
class PendingTrade:
    """A trade awaiting settlement for reflection."""
    trade_id: str
    ticker: str
    event_ticker: str
    side: str  # "yes" or "no"
    contracts: int
    entry_price_cents: int
    reasoning: str
    timestamp: float
    order_id: Optional[str] = None  # Kalshi order ID for precise matching
    # Filled after settlement
    settled: bool = False
    exit_price_cents: Optional[int] = None
    pnl_cents: Optional[int] = None
    result: Optional[str] = None  # "win", "loss", "break_even"
    settled_at: Optional[float] = None


@dataclass
class ReflectionResult:
    """Result of a reflection on a settled trade."""
    trade_id: str
    ticker: str
    result: str  # "win", "loss", "break_even"
    pnl_cents: int
    # Extracted learnings
    learning: str
    should_update_strategy: bool
    strategy_update: Optional[str] = None
    mistake_identified: Optional[str] = None
    pattern_identified: Optional[str] = None
    # Metadata
    reflection_timestamp: float = field(default_factory=time.time)


class ReflectionEngine:
    """
    Monitors trade outcomes and triggers reflection for learning.

    The engine maintains a list of pending trades and periodically
    checks for settlements. When a trade settles, it generates a
    reflection prompt and stores the extracted learnings.
    """

    def __init__(
        self,
        state_container: Optional['V3StateContainer'] = None,
        websocket_manager: Optional['V3WebSocketManager'] = None,
        memory_dir: Optional[Path] = None,
    ):
        """
        Initialize the reflection engine.

        Args:
            state_container: Container for trading state
            websocket_manager: WebSocket manager for streaming updates
            memory_dir: Directory for memory files
        """
        self._state_container = state_container
        self._ws_manager = websocket_manager
        self._memory_dir = memory_dir or Path(__file__).parent / "memory"

        # Pending trades awaiting settlement
        self._pending_trades: Dict[str, PendingTrade] = {}

        # Completed reflections
        self._reflections: List[ReflectionResult] = []

        # Callback for triggering agent reflection
        self._reflection_callback: Optional[Callable] = None

        # Background task for settlement monitoring
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # Settings
        self._check_interval_seconds = 30.0
        self._max_pending_trades = 100

        # File lock to prevent race conditions on memory file writes
        self._file_lock = asyncio.Lock()

        # Queue for background reflection processing (prevents blocking main loop)
        self._reflection_queue: asyncio.Queue = asyncio.Queue()
        self._reflection_worker_task: Optional[asyncio.Task] = None

    def set_reflection_callback(self, callback: Callable) -> None:
        """
        Set the callback to trigger agent reflection.

        The callback receives a PendingTrade that has settled.

        Args:
            callback: Async function(trade: PendingTrade) -> ReflectionResult
        """
        self._reflection_callback = callback

    def set_state_container(self, state_container: 'V3StateContainer') -> None:
        """Set the state container."""
        self._state_container = state_container

    async def start(self) -> None:
        """Start the reflection engine."""
        if self._running:
            logger.warning("[deep_agent.reflection] Already running")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._settlement_monitor_loop())
        self._reflection_worker_task = asyncio.create_task(self._reflection_worker_loop())
        logger.info("[deep_agent.reflection] Started reflection engine")

    async def stop(self) -> None:
        """Stop the reflection engine."""
        if not self._running:
            return

        self._running = False

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        if self._reflection_worker_task and not self._reflection_worker_task.done():
            self._reflection_worker_task.cancel()
            try:
                await self._reflection_worker_task
            except asyncio.CancelledError:
                pass

        logger.info("[deep_agent.reflection] Stopped reflection engine")

    def record_trade(
        self,
        trade_id: str,
        ticker: str,
        event_ticker: str,
        side: str,
        contracts: int,
        entry_price_cents: int,
        reasoning: str,
        order_id: Optional[str] = None,
    ) -> None:
        """
        Record a trade for later reflection.

        Args:
            trade_id: Unique trade identifier (generated UUID)
            ticker: Market ticker
            event_ticker: Event ticker
            side: "yes" or "no"
            contracts: Number of contracts
            entry_price_cents: Entry price in cents
            reasoning: Agent's reasoning for the trade
            order_id: Kalshi order ID for precise settlement matching
        """
        if len(self._pending_trades) >= self._max_pending_trades:
            # Remove oldest trade
            oldest_id = min(
                self._pending_trades.keys(),
                key=lambda k: self._pending_trades[k].timestamp
            )
            del self._pending_trades[oldest_id]

        trade = PendingTrade(
            trade_id=trade_id,
            ticker=ticker,
            event_ticker=event_ticker,
            side=side,
            contracts=contracts,
            entry_price_cents=entry_price_cents,
            reasoning=reasoning,
            timestamp=time.time(),
            order_id=order_id,
        )

        self._pending_trades[trade_id] = trade
        logger.info(f"[deep_agent.reflection] Recorded pending trade: {ticker} {side} @ {entry_price_cents}c (order_id={order_id})")

    async def _settlement_monitor_loop(self) -> None:
        """Background loop to check for settled trades."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval_seconds)
                await self._check_settlements()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[deep_agent.reflection] Error in monitor loop: {e}")

    async def _reflection_worker_loop(self) -> None:
        """Background worker that processes reflection queue without blocking main loop."""
        while self._running:
            try:
                # Wait for a trade to reflect on (with timeout to check _running)
                try:
                    trade = await asyncio.wait_for(
                        self._reflection_queue.get(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Process reflection
                # NOTE: Agent has full control over memory writes via write_memory() tool
                # We no longer auto-append to avoid duplicates - agent decides what to save
                if self._reflection_callback:
                    try:
                        reflection_result = await self._reflection_callback(trade)
                        if reflection_result:
                            self._reflections.append(reflection_result)
                            # Removed: await self._store_learning(reflection_result)
                            # Agent now handles all memory writes directly
                            logger.info(
                                f"[deep_agent.reflection] Completed reflection for "
                                f"{trade.ticker} ({trade.result})"
                            )
                    except Exception as e:
                        logger.error(f"[deep_agent.reflection] Error in reflection callback: {e}")

                self._reflection_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[deep_agent.reflection] Error in reflection worker: {e}")

    async def _check_settlements(self) -> None:
        """Check for settled trades and trigger reflection."""
        if not self._state_container:
            return

        if not self._pending_trades:
            return

        try:
            # Get settlements from state container
            summary = self._state_container.get_trading_summary()
            settlements = summary.get("settlements", [])

            # Build lookup structures for matching
            # Primary: order_id -> settlement
            settlements_by_order_id = {
                s.get("order_id"): s for s in settlements if s.get("order_id")
            }
            # Secondary: (ticker, side) -> list of settlements
            settlements_by_ticker_side: Dict[tuple, List] = {}
            for s in settlements:
                key = (s.get("ticker"), s.get("side", "").lower())
                if key not in settlements_by_ticker_side:
                    settlements_by_ticker_side[key] = []
                settlements_by_ticker_side[key].append(s)

            # Check each pending trade
            for trade_id, trade in list(self._pending_trades.items()):
                if trade.settled:
                    continue

                matched_settlement = None

                # Primary match: by order_id (most precise)
                if trade.order_id and trade.order_id in settlements_by_order_id:
                    matched_settlement = settlements_by_order_id[trade.order_id]
                    logger.debug(
                        f"[deep_agent.reflection] Matched by order_id: {trade.order_id}"
                    )

                # Secondary match: by ticker + side + contracts (fallback)
                if matched_settlement is None:
                    key = (trade.ticker, trade.side.lower())
                    candidates = settlements_by_ticker_side.get(key, [])
                    for settlement in candidates:
                        # Match by contract count for disambiguation
                        settlement_contracts = settlement.get("contracts", 0)
                        if settlement_contracts == trade.contracts:
                            matched_settlement = settlement
                            logger.debug(
                                f"[deep_agent.reflection] Matched by ticker+side+contracts: "
                                f"{trade.ticker} {trade.side} x{trade.contracts}"
                            )
                            break
                    # If no exact contract match, take first match (backward compat)
                    if matched_settlement is None and candidates:
                        matched_settlement = candidates[0]
                        logger.debug(
                            f"[deep_agent.reflection] Matched by ticker+side (fallback): "
                            f"{trade.ticker} {trade.side}"
                        )

                if matched_settlement:
                    await self._handle_settlement(trade, matched_settlement)

        except Exception as e:
            logger.error(f"[deep_agent.reflection] Error checking settlements: {e}")

    async def _handle_settlement(
        self,
        trade: PendingTrade,
        settlement: Dict[str, Any],
    ) -> None:
        """Handle a trade that has settled."""
        # Calculate result
        pnl_cents = int(settlement.get("pnl", 0) * 100)

        if pnl_cents > 0:
            result = "win"
        elif pnl_cents < 0:
            result = "loss"
        else:
            result = "break_even"

        # Update trade record
        trade.settled = True
        trade.exit_price_cents = settlement.get("price_cents")
        trade.pnl_cents = pnl_cents
        trade.result = result
        trade.settled_at = time.time()

        logger.info(
            f"[deep_agent.reflection] Trade settled: {trade.ticker} {result} "
            f"P&L: ${pnl_cents / 100:.2f}"
        )

        # Broadcast settlement to WebSocket
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_settlement", {
                "ticker": trade.ticker,
                "side": trade.side,
                "contracts": trade.contracts,
                "entry_price": trade.entry_price_cents,
                "exit_price": trade.exit_price_cents,
                "pnl_cents": pnl_cents,
                "result": result,
                "reasoning": trade.reasoning[:200],
                "timestamp": time.strftime("%H:%M:%S"),
            })

        # Queue reflection for background processing (non-blocking)
        if self._reflection_callback:
            await self._reflection_queue.put(trade)
            logger.info(f"[deep_agent.reflection] Queued reflection for {trade.ticker}")

        # Remove from pending
        del self._pending_trades[trade.trade_id]

    async def _store_learning(self, reflection: ReflectionResult) -> None:
        """Store a learning in the appropriate memory file."""
        try:
            # Format the learning entry
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime("%H:%M")

            entry = (
                f"\n## {date_str} {time_str} - {reflection.ticker} ({reflection.result})\n"
                f"- P&L: ${reflection.pnl_cents / 100:.2f}\n"
                f"- Learning: {reflection.learning}\n"
            )

            # Use lock to prevent race conditions on concurrent file writes
            async with self._file_lock:
                # Append to learnings.md
                learnings_path = self._memory_dir / "learnings.md"
                self._append_to_file(learnings_path, entry)

                # If mistake identified, add to mistakes.md
                if reflection.mistake_identified:
                    mistake_entry = f"\n- {reflection.mistake_identified} (from {reflection.ticker} {reflection.result})\n"
                    mistakes_path = self._memory_dir / "mistakes.md"
                    self._append_to_file(mistakes_path, mistake_entry)

                # If pattern identified, add to patterns.md
                if reflection.pattern_identified:
                    pattern_entry = f"\n- {reflection.pattern_identified}\n"
                    patterns_path = self._memory_dir / "patterns.md"
                    self._append_to_file(patterns_path, pattern_entry)

            logger.info(f"[deep_agent.reflection] Stored learning from {reflection.ticker}")

        except Exception as e:
            logger.error(f"[deep_agent.reflection] Error storing learning: {e}")

    def _append_to_file(self, path: Path, content: str) -> None:
        """Append content to a file, creating if needed. Must be called under _file_lock."""
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            path.write_text(existing + content, encoding="utf-8")
        else:
            path.write_text(content, encoding="utf-8")

    def _build_performance_scorecard(self) -> str:
        """
        Build a quantitative performance scorecard for injection into reflections.

        Returns a compact summary of all-time stats, recent trend (last 5 trades),
        and P&L trajectory to give the agent concrete feedback on improvement.
        """
        if not self._reflections:
            return ""

        total = len(self._reflections)
        wins = sum(1 for r in self._reflections if r.result == "win")
        losses = sum(1 for r in self._reflections if r.result == "loss")
        total_pnl = sum(r.pnl_cents for r in self._reflections)
        win_rate = wins / total if total > 0 else 0.0

        # Last 5 trades trend
        recent = self._reflections[-5:]
        recent_wins = sum(1 for r in recent if r.result == "win")
        recent_pnl = sum(r.pnl_cents for r in recent)
        recent_win_rate = recent_wins / len(recent) if recent else 0.0

        # Strategy update count
        strategy_updates = sum(1 for r in self._reflections if r.should_update_strategy)
        mistakes_found = sum(1 for r in self._reflections if r.mistake_identified)

        # Trend arrow
        if len(self._reflections) >= 5:
            first_half_pnl = sum(r.pnl_cents for r in self._reflections[:total // 2])
            second_half_pnl = sum(r.pnl_cents for r in self._reflections[total // 2:])
            trend = "IMPROVING" if second_half_pnl > first_half_pnl else "DECLINING" if second_half_pnl < first_half_pnl else "FLAT"
        else:
            trend = "TOO_EARLY"

        lines = [
            "### Performance Scorecard",
            f"- **All-Time**: {wins}W/{losses}L ({win_rate:.0%} win rate), P&L: ${total_pnl / 100:.2f}",
            f"- **Last 5 Trades**: {recent_wins}W/{len(recent) - recent_wins}L ({recent_win_rate:.0%}), P&L: ${recent_pnl / 100:.2f}",
            f"- **Trend**: {trend}",
            f"- **Strategy Updates**: {strategy_updates} | **Mistakes Found**: {mistakes_found}",
        ]

        return "\n".join(lines)

    def generate_reflection_prompt(self, trade: PendingTrade) -> str:
        """
        Generate a reflection prompt for a settled trade.

        This prompt is sent to the agent to extract learnings.

        Args:
            trade: The settled trade to reflect on

        Returns:
            Structured prompt for reflection
        """
        result_emoji = "✅" if trade.result == "win" else "❌" if trade.result == "loss" else "➖"

        # Build performance scorecard for quantitative context
        scorecard = self._build_performance_scorecard()
        scorecard_section = f"\n{scorecard}\n" if scorecard else ""

        return f"""
## Trade Settlement - Time to Reflect {result_emoji}

A trade you made has settled. Analyze the outcome and extract learnings.

### Trade Details
- **Market**: {trade.ticker}
- **Side**: {trade.side.upper()}
- **Contracts**: {trade.contracts}
- **Entry Price**: {trade.entry_price_cents}c
- **Exit Price**: {trade.exit_price_cents}c
- **P&L**: ${(trade.pnl_cents or 0) / 100:.2f}
- **Result**: {trade.result.upper()}

### Your Original Reasoning
{trade.reasoning}
{scorecard_section}
### Reflection Questions
1. **Why did this trade {trade.result}?** What was the key factor?
2. **Was your reasoning correct?** Did the market move as you expected?
3. **What can you learn?** What would you do differently next time?
4. **Should you update your strategy?** Is there a rule to add/modify?

### Instructions
Use the `reflect()` tool to record your structured analysis. It auto-appends to the right memory files.

Call `reflect(trade_ticker, outcome_analysis, reasoning_accuracy, key_learning, ...)` with:
- **trade_ticker**: "{trade.ticker}"
- **outcome_analysis**: Why did this trade {trade.result}?
- **reasoning_accuracy**: "correct", "partially_correct", or "wrong"
- **key_learning**: Specific, actionable insight
- **mistake**: (optional) Clear error to avoid
- **pattern**: (optional) Repeatable winning setup
- **strategy_update_needed**: true/false
- **confidence_in_learning**: "high", "medium", or "low"

If strategy_update_needed=true, also call `read_memory("strategy.md")` then `write_memory("strategy.md", ...)` with the updated version.

Be specific and actionable in your learnings.
"""

    def get_pending_trades(self) -> List[Dict]:
        """Get list of pending trades awaiting settlement."""
        return [asdict(t) for t in self._pending_trades.values()]

    def get_pending_trades_serializable(self) -> List[Dict]:
        """
        Get pending trades in serializable format for WebSocket snapshot.

        Returns a simplified version of pending trades suitable for
        frontend restoration after page refresh.

        Returns:
            List of dicts with trade details
        """
        return [
            {
                "trade_id": t.trade_id,
                "ticker": t.ticker,
                "side": t.side,
                "contracts": t.contracts,
                "entry_price_cents": t.entry_price_cents,
                "reasoning": t.reasoning[:200] if t.reasoning else "",
                "timestamp": t.timestamp,
            }
            for t in self._pending_trades.values()
        ]

    def get_recent_reflections(self, limit: int = 10) -> List[Dict]:
        """Get recent reflection results."""
        return [asdict(r) for r in self._reflections[-limit:]]

    def get_stats(self) -> Dict[str, Any]:
        """Get reflection engine statistics."""
        wins = sum(1 for r in self._reflections if r.result == "win")
        losses = sum(1 for r in self._reflections if r.result == "loss")
        total = len(self._reflections)

        return {
            "running": self._running,
            "pending_trades": len(self._pending_trades),
            "total_reflections": total,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total if total > 0 else 0.0,
            "strategy_updates": sum(1 for r in self._reflections if r.should_update_strategy),
            "mistakes_identified": sum(1 for r in self._reflections if r.mistake_identified),
            "patterns_identified": sum(1 for r in self._reflections if r.pattern_identified),
        }
