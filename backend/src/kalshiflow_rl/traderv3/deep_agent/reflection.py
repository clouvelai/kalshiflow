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
import json
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
    # Extraction snapshot at trade time (for learning loop feedback)
    extraction_ids: List[str] = field(default_factory=list)
    extraction_snapshot: List[Dict] = field(default_factory=list)
    # GDELT query snapshot at trade time (for reflection on news confirmation)
    gdelt_snapshot: List[Dict] = field(default_factory=list)
    # Microstructure snapshot at trade time (orderbook + trade flow)
    microstructure_snapshot: Optional[Dict[str, Any]] = None
    # Calibration fields (from think() tool)
    estimated_probability: Optional[int] = None
    what_could_go_wrong: Optional[str] = None
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

        # Persistent pending trades (survives restarts for reflection continuity)
        self._pending_trades_path = self._memory_dir / "pending_trades.json"

        # Persistent scorecard (survives restarts)
        self._scorecard_path = self._memory_dir / "scorecard.json"
        self._scorecard: List[Dict[str, Any]] = []
        self._scorecard_max_entries = 500

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

        # Load persistent state from previous sessions
        self._load_scorecard()
        self._load_pending_trades()

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

        # Persist pending trades so they survive restart
        self._save_pending_trades()

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
        extraction_ids: Optional[List[str]] = None,
        extraction_snapshot: Optional[List[Dict]] = None,
        gdelt_snapshot: Optional[List[Dict]] = None,
        estimated_probability: Optional[int] = None,
        what_could_go_wrong: Optional[str] = None,
        microstructure_snapshot: Optional[Dict[str, Any]] = None,
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
            extraction_ids: IDs of extractions that drove this trade
            extraction_snapshot: Full extraction data at trade time
            gdelt_snapshot: Recent GDELT queries at trade time (for reflection)
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
            extraction_ids=extraction_ids or [],
            extraction_snapshot=extraction_snapshot or [],
            gdelt_snapshot=gdelt_snapshot or [],
            microstructure_snapshot=microstructure_snapshot,
            estimated_probability=estimated_probability,
            what_could_go_wrong=what_could_go_wrong,
        )

        self._pending_trades[trade_id] = trade
        self._save_pending_trades()
        logger.info(
            f"[deep_agent.reflection] Recorded pending trade: {ticker} {side} @ {entry_price_cents}c "
            f"(order_id={order_id}, extractions={len(trade.extraction_ids)})"
        )

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

            # T4.4: Per-trade try/except so one bad trade doesn't abort all matching
            for trade_id, trade in list(self._pending_trades.items()):
                if trade.settled:
                    continue

                try:
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
                        # No fuzzy fallback - require order_id or ticker+side+contracts match
                        # to avoid mismatching settlements across different trades

                    if matched_settlement:
                        await self._handle_settlement(trade, matched_settlement)

                except Exception as e:
                    logger.error(
                        f"[deep_agent.reflection] Error matching trade {trade_id} "
                        f"({trade.ticker}): {e}"
                    )

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
        # T4.5: Derive exit_price if settlement doesn't provide it
        exit_price = settlement.get("price_cents")
        if exit_price is None and trade.contracts > 0:
            # Try to derive from P&L: exit = entry + (pnl / contracts)
            try:
                exit_price = trade.entry_price_cents + (pnl_cents // trade.contracts)
            except (ZeroDivisionError, TypeError):
                logger.warning(
                    f"[deep_agent.reflection] Could not derive exit_price for {trade.ticker}"
                )
        trade.exit_price_cents = exit_price
        trade.pnl_cents = pnl_cents
        trade.result = result
        trade.settled_at = time.time()

        logger.info(
            f"[deep_agent.reflection] Trade settled: {trade.ticker} {result} "
            f"P&L: ${pnl_cents / 100:.2f}"
        )

        # Record to persistent scorecard
        self._record_scorecard_entry(trade)

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

        # Remove from pending and persist
        del self._pending_trades[trade.trade_id]
        self._save_pending_trades()

    def _compute_reflection_aggregates(self) -> Dict[str, Any]:
        """Compute win/loss/total/pnl aggregates over settled reflections."""
        total = len(self._reflections)
        wins = sum(1 for r in self._reflections if r.result == "win")
        losses = sum(1 for r in self._reflections if r.result == "loss")
        total_pnl = sum(r.pnl_cents for r in self._reflections)
        strategy_updates = sum(1 for r in self._reflections if r.should_update_strategy)
        mistakes_found = sum(1 for r in self._reflections if r.mistake_identified)
        patterns_found = sum(1 for r in self._reflections if r.pattern_identified)
        return {
            "total": total,
            "wins": wins,
            "losses": losses,
            "total_pnl": total_pnl,
            "win_rate": wins / total if total > 0 else 0.0,
            "strategy_updates": strategy_updates,
            "mistakes_found": mistakes_found,
            "patterns_found": patterns_found,
        }

    def _build_performance_scorecard(self) -> str:
        """
        Build a quantitative performance scorecard for injection into reflections.

        Returns a compact summary of all-time stats (including rolled aggregates),
        recent trend, P&L trajectory, per-event breakdown, and open position summary.
        """
        if not self._reflections and not self._state_container:
            return ""

        lines = ["### Performance Scorecard"]

        # Settled trade stats (only if we have reflections)
        if self._reflections:
            agg = self._compute_reflection_aggregates()
            session_total = agg["total"]
            session_wins = agg["wins"]
            session_losses = agg["losses"]
            session_pnl = agg["total_pnl"]

            # Combined all-time stats (rolled aggregates + current window)
            all_time = getattr(self, '_all_time_aggregates', self._empty_all_time_aggregates())
            all_total = all_time.get("total_trades", 0) + len(self._scorecard)
            all_wins = all_time.get("wins", 0) + sum(1 for t in self._scorecard if t.get("result") == "win")
            all_losses = all_time.get("losses", 0) + sum(1 for t in self._scorecard if t.get("result") == "loss")
            all_pnl = all_time.get("total_pnl_cents", 0) + sum(t.get("pnl_cents", 0) for t in self._scorecard)
            all_win_rate = all_wins / all_total if all_total > 0 else 0.0

            # Last 5 trades trend
            recent = self._reflections[-5:]
            recent_wins = sum(1 for r in recent if r.result == "win")
            recent_pnl = sum(r.pnl_cents for r in recent)
            recent_win_rate = recent_wins / len(recent) if recent else 0.0

            strategy_updates = agg["strategy_updates"]
            mistakes_found = agg["mistakes_found"]

            # Trend arrow
            if len(self._reflections) >= 5:
                first_half_pnl = sum(r.pnl_cents for r in self._reflections[:session_total // 2])
                second_half_pnl = sum(r.pnl_cents for r in self._reflections[session_total // 2:])
                trend = "IMPROVING" if second_half_pnl > first_half_pnl else "DECLINING" if second_half_pnl < first_half_pnl else "FLAT"
            else:
                trend = "TOO_EARLY"

            lines.extend([
                f"- **All-Time**: {all_total} trades, {all_wins}W/{all_losses}L ({all_win_rate:.0%}), P&L: ${all_pnl / 100:.2f}",
                f"- **This Session**: {session_wins}W/{session_losses}L, P&L: ${session_pnl / 100:.2f}",
                f"- **Last 5 Trades**: {recent_wins}W/{len(recent) - recent_wins}L ({recent_win_rate:.0%}), P&L: ${recent_pnl / 100:.2f}",
                f"- **Trend**: {trend}",
                f"- **Strategy Updates**: {strategy_updates} | **Mistakes Found**: {mistakes_found}",
            ])

            # Per-event breakdown (top 5 by trade count)
            by_event = getattr(self, '_by_event', {})
            if by_event:
                sorted_events = sorted(by_event.items(), key=lambda x: x[1]["trades"], reverse=True)[:5]
                event_lines = []
                for et, ev in sorted_events:
                    ew = ev.get("wins", 0)
                    el = ev.get("losses", 0)
                    ep = ev.get("pnl_cents", 0)
                    event_lines.append(f"  {et}: {ev['trades']}T {ew}W/{el}L ${ep/100:.2f}")
                lines.append("- **By Event**: " + " | ".join(event_lines))

        # Open position summary (deep_agent positions only)
        # Use pending trades as source of truth for agent's positions
        agent_tickers = {trade.ticker for trade in self._pending_trades.values()}
        if agent_tickers and self._state_container:
            try:
                summary = self._state_container.get_trading_summary()
                all_positions = summary.get("positions_details", [])
                agent_positions = [
                    p for p in all_positions
                    if p.get("ticker") in agent_tickers
                ]
                if agent_positions:
                    open_count = len(agent_positions)
                    total_unrealized = sum(
                        p.get("unrealized_pnl", 0) for p in agent_positions
                    )
                    winning = sum(
                        1 for p in agent_positions if p.get("unrealized_pnl", 0) > 0
                    )
                    losing = sum(
                        1 for p in agent_positions if p.get("unrealized_pnl", 0) < 0
                    )
                    sign = "+" if total_unrealized >= 0 else ""
                    lines.append(
                        f"- **Open Positions**: {open_count} open, "
                        f"unrealized {sign}${total_unrealized / 100:.2f} "
                        f"({winning} winning, {losing} losing)"
                    )
            except Exception as e:
                logger.debug(f"[deep_agent.reflection] Could not get open positions for scorecard: {e}")

        # True state from Kalshi API (strategy-filtered)
        if self._state_container:
            try:
                summary = self._state_container.get_trading_summary()
                settlements = summary.get("settlements", [])
                agent_settlements = [s for s in settlements if s.get("strategy_id") == "deep_agent"]
                if agent_settlements:
                    true_wins = sum(1 for s in agent_settlements if s.get("net_pnl", 0) > 0)
                    true_losses = len(agent_settlements) - true_wins
                    true_realized = sum(s.get("net_pnl", 0) for s in agent_settlements)

                    # Also get unrealized from open positions
                    all_positions = summary.get("positions_details", [])
                    agent_tickers_from_settlements = {s.get("ticker") for s in agent_settlements}
                    agent_tickers_from_pending = {t.ticker for t in self._pending_trades.values()}
                    combined_tickers = agent_tickers_from_settlements | agent_tickers_from_pending
                    true_unrealized = sum(
                        p.get("unrealized_pnl", 0) for p in all_positions
                        if p.get("ticker") in combined_tickers
                    )
                    true_total = true_realized + true_unrealized

                    unrealized_str = ""
                    if true_unrealized != 0:
                        sign = "+" if true_unrealized >= 0 else ""
                        unrealized_str = f", unrealized: {sign}${true_unrealized/100:.2f}"

                    lines.append(
                        f"- **True State (Kalshi)**: {len(agent_settlements)} settled, "
                        f"{true_wins}W/{true_losses}L, "
                        f"realized: ${true_realized/100:.2f}{unrealized_str}, "
                        f"total: ${true_total/100:.2f}"
                    )
            except Exception as e:
                logger.debug(f"[deep_agent.reflection] Could not get true state for scorecard: {e}")

        # Only return content if we have more than just the header
        if len(lines) <= 1:
            return ""

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

        # Build extraction snapshot section if available
        extraction_section = ""
        if trade.extraction_snapshot:
            extraction_lines = ["### Extraction Signals That Drove This Trade"]
            for ext in trade.extraction_snapshot[:10]:
                ext_id = ext.get("id", "?")
                ext_class = ext.get("extraction_class", "?")
                ext_text = ext.get("extraction_text", "")[:150]
                attrs = ext.get("attributes", {})
                direction = attrs.get("direction", "?")
                magnitude = attrs.get("magnitude", "?")
                confidence = attrs.get("confidence", "?")
                extraction_lines.append(
                    f"- **[{ext_id}]** {ext_class}: \"{ext_text}\" "
                    f"(direction={direction}, magnitude={magnitude}, confidence={confidence})"
                )
            extraction_section = "\n".join(extraction_lines) + "\n"

        # Build GDELT snapshot section if available
        gdelt_section = ""
        if trade.gdelt_snapshot:
            gdelt_lines = ["### GDELT News Context at Trade Time"]
            for gq in trade.gdelt_snapshot[:5]:
                terms = ", ".join(gq.get("search_terms", []))
                articles = gq.get("article_count", 0)
                sources = gq.get("source_diversity", 0)
                tone = gq.get("avg_tone", 0)
                gdelt_lines.append(
                    f"- Query [{terms}]: {articles} articles, {sources} sources, tone={tone:.1f}"
                )
            gdelt_section = "\n".join(gdelt_lines) + "\n"

        # Build microstructure snapshot section if available
        micro_section = ""
        if trade.microstructure_snapshot:
            ms = trade.microstructure_snapshot
            micro_lines = ["### Microstructure at Trade Entry"]
            if ms.get("total_trades"):
                micro_lines.append(f"- Trade Flow: {ms.get('yes_ratio', 0):.0%} YES ({ms.get('total_trades', 0)} trades)")
            if ms.get("price_drop") and ms["price_drop"] != 0:
                micro_lines.append(f"- Price Move: {ms['price_drop']:+d}c from open")
            ob = ms.get("orderbook", {})
            if ob:
                if ob.get("spread_close") is not None:
                    micro_lines.append(f"- Spread: {ob['spread_close']}c")
                imb = ob.get("imbalance_ratio")
                if imb is not None and abs(imb) > 0.1:
                    micro_lines.append(f"- Volume Imbalance: {imb:+.2f} ({'buy' if imb > 0 else 'sell'} pressure)")
                if ob.get("large_order_count", 0) > 0:
                    micro_lines.append(f"- Large Orders: {ob['large_order_count']} (whale activity)")
            micro_section = "\n".join(micro_lines) + "\n"

        # Build evaluate instruction if we have extraction IDs
        evaluate_instruction = ""
        if trade.extraction_ids:
            evaluate_instruction = f"""
### Extraction Feedback (IMPORTANT)
Call `evaluate_extractions()` to score each extraction's accuracy now that you know the outcome:
- **trade_ticker**: "{trade.ticker}"
- **trade_outcome**: "{trade.result}"
- **evaluations**: Array of {{"extraction_id": "...", "accuracy": "accurate|partially_accurate|inaccurate|noise", "note": "..."}}

Accurate extractions from winning trades are automatically promoted as examples for future extraction calls.
"""

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
{extraction_section}{gdelt_section}{micro_section}
### Reflection Questions
1. **Why did this trade {trade.result}?** What was the key factor?
2. **Was your reasoning correct?** Did the market move as you expected?
3. **What can you learn?** What would you do differently next time?
4. **Should you update your strategy?** Is there a rule to add/modify?
5. **Were the extraction signals accurate?** Did they correctly predict the direction?
6. **Was GDELT confirmation useful?** Did mainstream news coverage align with the outcome?
7. **Did microstructure support the trade?** Was trade flow aligned? Was spread reasonable? Any whale contradiction?

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
{evaluate_instruction}
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
        agg = self._compute_reflection_aggregates()
        return {
            "running": self._running,
            "pending_trades": len(self._pending_trades),
            "total_reflections": agg["total"],
            "wins": agg["wins"],
            "losses": agg["losses"],
            "win_rate": agg["win_rate"],
            "session_pnl_cents": agg["total_pnl"],
            "strategy_updates": agg["strategy_updates"],
            "mistakes_identified": agg["mistakes_found"],
            "patterns_identified": agg["patterns_found"],
            "scorecard_entries": len(self._scorecard),
        }

    # === Persistent Pending Trades ===

    def _save_pending_trades(self) -> None:
        """Persist pending trades to disk so they survive restarts."""
        try:
            data = {
                trade_id: asdict(trade)
                for trade_id, trade in self._pending_trades.items()
                if not trade.settled
            }
            self._pending_trades_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
            logger.debug(
                f"[deep_agent.reflection] Persisted {len(data)} pending trades"
            )
        except Exception as e:
            logger.error(f"[deep_agent.reflection] Failed to persist pending trades: {e}")

    def _load_pending_trades(self) -> None:
        """Load pending trades from disk on startup."""
        if not self._pending_trades_path.exists():
            logger.info("[deep_agent.reflection] No pending_trades.json found — starting fresh")
            return

        try:
            data = json.loads(self._pending_trades_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("[deep_agent.reflection] Invalid pending_trades.json format — skipping")
                return

            loaded = 0
            for trade_id, trade_data in data.items():
                try:
                    trade = PendingTrade(
                        trade_id=trade_data["trade_id"],
                        ticker=trade_data["ticker"],
                        event_ticker=trade_data["event_ticker"],
                        side=trade_data["side"],
                        contracts=trade_data["contracts"],
                        entry_price_cents=trade_data["entry_price_cents"],
                        reasoning=trade_data["reasoning"],
                        timestamp=trade_data["timestamp"],
                        order_id=trade_data.get("order_id"),
                        extraction_ids=trade_data.get("extraction_ids", []),
                        extraction_snapshot=trade_data.get("extraction_snapshot", []),
                        gdelt_snapshot=trade_data.get("gdelt_snapshot", []),
                        microstructure_snapshot=trade_data.get("microstructure_snapshot"),
                        estimated_probability=trade_data.get("estimated_probability"),
                        what_could_go_wrong=trade_data.get("what_could_go_wrong"),
                    )
                    self._pending_trades[trade_id] = trade
                    loaded += 1
                except (KeyError, TypeError) as e:
                    logger.warning(
                        f"[deep_agent.reflection] Skipping malformed pending trade {trade_id}: {e}"
                    )

            logger.info(
                f"[deep_agent.reflection] Loaded {loaded} pending trades from previous session"
            )
        except Exception as e:
            logger.warning(f"[deep_agent.reflection] Failed to load pending trades: {e}")

    # === Persistent Scorecard ===

    def _load_scorecard(self) -> None:
        """Load scorecard from disk on startup."""
        if not self._scorecard_path.exists():
            logger.info("[deep_agent.reflection] No scorecard.json found — starting fresh")
            self._all_time_aggregates = self._empty_all_time_aggregates()
            self._by_event = {}
            self._weekly_pnl = []
            return

        try:
            data = json.loads(self._scorecard_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._scorecard = data.get("trades", [])
                self._all_time_aggregates = data.get(
                    "all_time_aggregates", self._empty_all_time_aggregates()
                )
                self._by_event = data.get("by_event", {})
                self._weekly_pnl = data.get("weekly_pnl", [])
            elif isinstance(data, list):
                self._scorecard = data
                self._all_time_aggregates = self._empty_all_time_aggregates()
                self._by_event = {}
                self._weekly_pnl = []
            else:
                self._scorecard = []
                self._all_time_aggregates = self._empty_all_time_aggregates()
                self._by_event = {}
                self._weekly_pnl = []

            # Enforce max entries — roll overflow into all_time_aggregates first
            if len(self._scorecard) > self._scorecard_max_entries:
                overflow = self._scorecard[:-self._scorecard_max_entries]
                self._roll_into_aggregates(overflow)
                self._scorecard = self._scorecard[-self._scorecard_max_entries:]

            logger.info(
                f"[deep_agent.reflection] Loaded scorecard: {len(self._scorecard)} trades, "
                f"all-time: {self._all_time_aggregates.get('total_trades', 0)} trades"
            )
        except Exception as e:
            logger.warning(f"[deep_agent.reflection] Failed to load scorecard: {e}")
            self._scorecard = []
            self._all_time_aggregates = self._empty_all_time_aggregates()
            self._by_event = {}
            self._weekly_pnl = []

    @staticmethod
    def _empty_all_time_aggregates() -> dict:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl_cents": 0,
            "first_trade_at": None,
            "last_trade_at": None,
        }

    def _roll_into_aggregates(self, trades: list) -> None:
        """Roll trade entries into all_time_aggregates before discarding them."""
        agg = self._all_time_aggregates
        for t in trades:
            agg["total_trades"] += 1
            if t.get("result") == "win":
                agg["wins"] += 1
            elif t.get("result") == "loss":
                agg["losses"] += 1
            agg["total_pnl_cents"] += t.get("pnl_cents", 0)

            ts = t.get("timestamp")
            if ts:
                if agg["first_trade_at"] is None or ts < agg["first_trade_at"]:
                    agg["first_trade_at"] = ts
                if agg["last_trade_at"] is None or ts > agg["last_trade_at"]:
                    agg["last_trade_at"] = ts

            # Update by_event
            event = t.get("event_ticker", "unknown")
            if event:
                if event not in self._by_event:
                    self._by_event[event] = {"trades": 0, "wins": 0, "losses": 0, "pnl_cents": 0}
                ev = self._by_event[event]
                ev["trades"] += 1
                if t.get("result") == "win":
                    ev["wins"] += 1
                elif t.get("result") == "loss":
                    ev["losses"] += 1
                ev["pnl_cents"] += t.get("pnl_cents", 0)

        # Prune by_event to top 20 by trade count
        if len(self._by_event) > 20:
            sorted_events = sorted(self._by_event.items(), key=lambda x: x[1]["trades"], reverse=True)
            self._by_event = dict(sorted_events[:20])

    def _persist_scorecard(self) -> None:
        """Write scorecard to disk after each settlement."""
        try:
            # Enforce cap — roll overflow into all_time_aggregates first
            if len(self._scorecard) > self._scorecard_max_entries:
                overflow = self._scorecard[:-self._scorecard_max_entries]
                self._roll_into_aggregates(overflow)
                self._scorecard = self._scorecard[-self._scorecard_max_entries:]

            # Compute current window aggregates
            total = len(self._scorecard)
            wins = sum(1 for t in self._scorecard if t.get("result") == "win")
            losses = sum(1 for t in self._scorecard if t.get("result") == "loss")
            total_pnl = sum(t.get("pnl_cents", 0) for t in self._scorecard)

            # Update weekly_pnl tracking
            self._update_weekly_pnl()

            data = {
                "trades": self._scorecard,
                "aggregates": {
                    "total_trades": total,
                    "wins": wins,
                    "losses": losses,
                    "total_pnl_cents": total_pnl,
                    "win_rate": wins / total if total > 0 else 0.0,
                    "updated_at": datetime.now().isoformat(),
                },
                "all_time_aggregates": self._all_time_aggregates,
                "by_event": self._by_event,
                "weekly_pnl": self._weekly_pnl,
            }

            self._scorecard_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            logger.info(
                f"[deep_agent.reflection] Scorecard persisted: "
                f"{total} trades, ${total_pnl/100:.2f} P&L"
            )
        except Exception as e:
            logger.error(f"[deep_agent.reflection] Failed to persist scorecard: {e}")

    def _update_weekly_pnl(self) -> None:
        """Update weekly_pnl list with current week's data."""
        now = datetime.now()
        week_key = now.strftime("%Y-W%W")

        # Sum PnL for current week from scorecard entries
        week_pnl = 0
        week_trades = 0
        for t in self._scorecard:
            ts = t.get("timestamp", "")
            if isinstance(ts, str) and ts[:4].isdigit():
                try:
                    trade_date = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if trade_date.strftime("%Y-W%W") == week_key:
                        week_pnl += t.get("pnl_cents", 0)
                        week_trades += 1
                except (ValueError, TypeError):
                    pass

        # Update or append current week entry
        updated = False
        for entry in self._weekly_pnl:
            if entry.get("week") == week_key:
                entry["pnl_cents"] = week_pnl
                entry["trades"] = week_trades
                updated = True
                break
        if not updated:
            self._weekly_pnl.append({
                "week": week_key,
                "pnl_cents": week_pnl,
                "trades": week_trades,
            })

        # Keep last 52 weeks
        if len(self._weekly_pnl) > 52:
            self._weekly_pnl = self._weekly_pnl[-52:]

    def _record_scorecard_entry(self, trade: PendingTrade) -> None:
        """Add a settled trade to the persistent scorecard."""
        entry = {
            "ticker": trade.ticker,
            "event_ticker": trade.event_ticker,
            "side": trade.side,
            "contracts": trade.contracts,
            "entry_cents": trade.entry_price_cents,
            "exit_cents": trade.exit_price_cents,
            "pnl_cents": trade.pnl_cents or 0,
            "result": trade.result or "unknown",
            "timestamp": datetime.now().isoformat(),
            "reasoning": (trade.reasoning[:200] if trade.reasoning else ""),
            "estimated_probability": trade.estimated_probability,
            "what_could_go_wrong": (trade.what_could_go_wrong[:200] if trade.what_could_go_wrong else None),
            "microstructure_snapshot": trade.microstructure_snapshot,
        }
        self._scorecard.append(entry)

        # Update by_event tracking for current trade
        event = trade.event_ticker or "unknown"
        if event not in self._by_event:
            self._by_event[event] = {"trades": 0, "wins": 0, "losses": 0, "pnl_cents": 0}
        ev = self._by_event[event]
        ev["trades"] += 1
        if trade.result == "win":
            ev["wins"] += 1
        elif trade.result == "loss":
            ev["losses"] += 1
        ev["pnl_cents"] += trade.pnl_cents or 0

        # Update all_time_aggregates last_trade_at
        self._all_time_aggregates["last_trade_at"] = entry["timestamp"]
        if self._all_time_aggregates["first_trade_at"] is None:
            self._all_time_aggregates["first_trade_at"] = entry["timestamp"]

        self._persist_scorecard()

    def get_scorecard_summary(self) -> str:
        """
        Build a scorecard summary combining all-time aggregates + current window.

        Returns empty string if no historical data.
        """
        # Current window stats
        window_total = len(self._scorecard)
        window_wins = sum(1 for t in self._scorecard if t.get("result") == "win")
        window_losses = sum(1 for t in self._scorecard if t.get("result") == "loss")
        window_pnl = sum(t.get("pnl_cents", 0) for t in self._scorecard)

        # Combined all-time + current window
        agg = getattr(self, '_all_time_aggregates', self._empty_all_time_aggregates())
        all_total = agg.get("total_trades", 0) + window_total
        all_wins = agg.get("wins", 0) + window_wins
        all_losses = agg.get("losses", 0) + window_losses
        all_pnl = agg.get("total_pnl_cents", 0) + window_pnl

        if all_total == 0:
            return ""

        win_rate = all_wins / all_total if all_total > 0 else 0.0

        summary = (
            f"All-Time: {all_total} trades, {all_wins}W/{all_losses}L "
            f"({win_rate:.0%}), P&L: ${all_pnl/100:.2f}"
        )

        # Add weekly trend if available
        weekly = getattr(self, '_weekly_pnl', [])
        if len(weekly) >= 2:
            last_week = weekly[-1].get("pnl_cents", 0)
            prev_week = weekly[-2].get("pnl_cents", 0)
            trend = "UP" if last_week > prev_week else "DOWN" if last_week < prev_week else "FLAT"
            summary += f" | Week trend: {trend}"

        return summary

    def get_by_event_stats(self) -> dict:
        """Get per-event statistics for distillation evidence."""
        return getattr(self, '_by_event', {})
