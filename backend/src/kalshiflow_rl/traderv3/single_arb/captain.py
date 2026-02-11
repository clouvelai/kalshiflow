"""Captain V2 - Attention-driven LLM Captain for single-event arbitrage.

Architecture: Infrastructure is the main loop, LLM is a judgment function called on-demand.

  Index → AttentionRouter → [only what matters, when it matters] → Captain LLM

Three invocation modes:
- REACTIVE: Fired by AttentionRouter when high-urgency signals emerge. Compact prompt
  with only attention items + relevant positions. (~200-400 tokens)
- STRATEGIC: Every 5 minutes. Portfolio review, sniper tuning, task planning. (~400-600 tokens)
- DEEP_SCAN: Every 30 minutes. All events, full positions, health, memories. (~800-1200 tokens)
"""

import asyncio
import logging
import os
import time
import traceback
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional, TYPE_CHECKING

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.messages import HumanMessage
from langgraph.cache.memory import InMemoryCache

from .context_builder import ContextBuilder
from .models import MarketState, PortfolioState, SniperStatus
from .task_ledger import TaskLedger
from .tools import ALL_TOOLS, TOOL_CATEGORIES

if TYPE_CHECKING:
    from .attention import AttentionRouter

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.captain")

CAPTAIN_PROMPT_TEMPLATE = """You are the Captain — an autonomous Kalshi prediction market trader.
Your job: extract profit from mispriced prediction markets using judgment the infrastructure can't provide.

PRICING: cents (0-100). YES@60c costs $0.60, pays $1.00 if YES wins. Tool inputs use cents. Balance in briefing is DOLLARS.

INVOCATION MODES:
You are invoked in one of three modes. Each cycle header tells you which mode you're in.

REACTIVE — You are called because something urgent happened. The ATTENTION section lists
signals that crossed scoring thresholds. Respond to them: trade, exit, or note why you're
passing. Be fast and decisive. Only the relevant context is provided.

STRATEGIC — You are called on a regular interval for planning. Review your portfolio,
tune sniper configuration, plan news research, manage positions. No urgency unless
PENDING_ATTENTION items are close to threshold.

DEEP_SCAN — You are called for comprehensive review. All events are summarized.
Full positions with P&L. Health metrics. Recent memories. Rebalance, store learnings,
adjust overall strategy.

EXITS: sell the side you hold. NEVER buy the opposite side to "hedge."
SNIPER: Auto-executes S1_ARB. You CONFIGURE it (edge threshold, capital, cooldown), it EXECUTES.
REGIME: Per-market vpin/sweep is informational. Only hard stops: negative edge, spread > 15c.

TASK PLANNING: write_todos in STRATEGIC and DEEP_SCAN cycles. Without it you lose context.
Track: positions to manage, edge opportunities, sniper config. Prefix [HIGH]/[MED]/[LOW].

{guidance_section}"""

# Path to the trading guidance file
_GUIDANCE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "skills", "trading-guidance", "GUIDANCE.md",
)

_guidance_cache: Optional[str] = None


def _load_guidance() -> str:
    """Load trading guidance from file (cached after first read)."""
    global _guidance_cache
    if _guidance_cache is not None:
        return _guidance_cache
    try:
        with open(_GUIDANCE_PATH, "r") as f:
            _guidance_cache = f.read()
        return _guidance_cache
    except FileNotFoundError:
        logger.warning(f"[CAPTAIN] Trading guidance not found: {_GUIDANCE_PATH}")
        return ""


class ArbCaptain:
    """Attention-driven LLM Captain for single-event arbitrage.

    Uses create_deep_agent with 12 tools (no subagents).
    Three invocation modes: reactive, strategic, deep_scan.
    AttentionRouter drives reactive cycles; timers drive strategic/deep_scan.
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
        attention_router: Optional["AttentionRouter"] = None,
        config=None,
        model_name: Optional[str] = None,
        cycle_interval: float = 60.0,
        event_callback: Optional[Callable[..., Coroutine]] = None,
        sniper_ref=None,
        system_ready: Optional[asyncio.Event] = None,
    ):
        from .mentions_models import get_captain_model
        model_name = model_name or get_captain_model()

        self._model_name = model_name
        self._cycle_interval = cycle_interval
        self._event_callback = event_callback
        self._ctx = context_builder
        self._sniper_ref = sniper_ref
        self._system_ready = system_ready
        self._attention_router = attention_router
        self._config = config

        # Intervals from config or defaults
        self._strategic_interval = getattr(config, "strategic_interval", 300.0) if config else 300.0
        self._deep_scan_interval = getattr(config, "deep_scan_interval", 1800.0) if config else 1800.0

        # Node-level cache
        self._cache = InMemoryCache()

        # StateBackend only (no FilesystemBackend - memory is in SessionMemoryStore)
        backend_factory = lambda rt: StateBackend(rt)

        # Load guidance once at agent creation (cached in system prompt for Anthropic prompt caching)
        guidance_text = _load_guidance()
        guidance_section = f"TRADING GUIDANCE:\n{guidance_text}" if guidance_text else ""
        system_prompt = CAPTAIN_PROMPT_TEMPLATE.format(guidance_section=guidance_section)

        self._agent = create_deep_agent(
            model=model_name,
            tools=ALL_TOOLS,
            system_prompt=system_prompt,
            backend=backend_factory,
            cache=self._cache,
        )

        self._running = False
        self._paused = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_cycle_at: Optional[float] = None
        self._errors: List[str] = []  # Capped at 50 entries
        self._task_ledger = TaskLedger(session_id=str(uuid.uuid4()))

        # Mode counters for stats
        self._reactive_count = 0
        self._strategic_count = 0
        self._deep_scan_count = 0

        # Per-cycle tool call tracking (reset in _cycle_preamble)
        self._current_cycle_tools: Dict[str, int] = {}

    async def start(self) -> None:
        """Start the Captain cycle loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[CAPTAIN:START] model={self._model_name} interval={self._cycle_interval}s")

    async def stop(self) -> None:
        """Stop the Captain."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._task_ledger.flush(timeout=3.0)
        logger.info(f"[CAPTAIN:STOP] cycles={self._cycle_count}")

    def pause(self) -> None:
        """Pause Captain after current cycle completes."""
        self._paused = True
        logger.info("[CAPTAIN:PAUSE]")

    def resume(self) -> None:
        """Resume Captain cycles."""
        self._paused = False
        logger.info("[CAPTAIN:RESUME]")

    @property
    def is_paused(self) -> bool:
        return self._paused

    def _record_error(self, error: str) -> None:
        """Append an error and cap the list at 50 entries."""
        self._errors.append(error)
        if len(self._errors) > 50:
            self._errors = self._errors[-50:]

    async def _run_loop(self) -> None:
        """Event-driven main loop with three invocation modes.

        Reactive: fires when AttentionRouter has high-urgency items.
        Strategic: fires every strategic_interval (default 5 min).
        Deep scan: fires every deep_scan_interval (default 30 min).

        Falls back to fixed-interval polling if no AttentionRouter is available.
        """
        index = self._ctx._index

        # Wait for index readiness (short timeout - REST prefetch should have populated data)
        max_wait = 10
        waited = 0
        while waited < max_wait:
            if index and index.is_ready:
                logger.info(f"[CAPTAIN] Index ready after {waited}s: {index.readiness_summary}")
                break
            await asyncio.sleep(1.0)
            waited += 1
            if waited % 5 == 0:
                summary = index.readiness_summary if index else "No index"
                logger.info(f"[CAPTAIN] Waiting for index ({waited}s): {summary}")
        else:
            summary = index.readiness_summary if index else "No index"
            logger.warning(f"[CAPTAIN] Starting despite incomplete index: {summary}")

        # Wait for system initialization (understanding builds, etc.) with timeout
        if self._system_ready:
            logger.info("[CAPTAIN] Waiting for system initialization...")
            try:
                await asyncio.wait_for(self._system_ready.wait(), timeout=300.0)
                logger.info("[CAPTAIN] System ready, starting cycles")
            except asyncio.TimeoutError:
                logger.warning("[CAPTAIN] System init timeout after 5min, starting anyway")

        last_strategic = time.time()
        last_deep_scan = time.time()

        # If no attention router, fall back to fixed-interval cycling
        if not self._attention_router:
            logger.info("[CAPTAIN] No AttentionRouter — falling back to fixed-interval mode")
            await self._run_loop_legacy()
            return

        logger.info(
            f"[CAPTAIN] Attention-driven mode: strategic={self._strategic_interval}s "
            f"deep_scan={self._deep_scan_interval}s"
        )

        # Emit config so frontend can show countdown timers
        await self._emit_event({
            "type": "captain_config",
            "data": {
                "strategic_interval": self._strategic_interval,
                "deep_scan_interval": self._deep_scan_interval,
            },
        })

        # Run initial deep_scan so there's always activity on startup
        try:
            logger.info("[CAPTAIN] Running initial deep_scan on startup")
            await self._run_with_timeout(self._run_deep_scan(), "deep_scan")
            self._cycle_count += 1
            self._last_cycle_at = time.time()
            last_deep_scan = time.time()
            last_strategic = time.time()
        except Exception as e:
            logger.warning(f"[CAPTAIN] Initial deep_scan failed: {e}")

        while self._running:
            if self._paused:
                await asyncio.sleep(5.0)
                continue

            try:
                trigger = await self._wait_for_trigger(
                    last_strategic, last_deep_scan,
                )

                if trigger == "reactive":
                    items = self._attention_router.drain_items(min_urgency="high")
                    if items:
                        await self._run_with_timeout(
                            self._run_reactive(items), "reactive"
                        )
                elif trigger == "strategic":
                    await self._run_with_timeout(
                        self._run_strategic(), "strategic"
                    )
                    last_strategic = time.time()
                elif trigger == "deep_scan":
                    await self._run_with_timeout(
                        self._run_deep_scan(), "deep_scan"
                    )
                    last_deep_scan = time.time()

                self._cycle_count += 1
                self._last_cycle_at = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CAPTAIN:ERROR] cycle={self._cycle_count + 1} error={e}\n{traceback.format_exc()}")
                self._record_error(str(e))
                await asyncio.sleep(30.0)

    async def _run_loop_legacy(self) -> None:
        """Fixed-interval fallback loop (no AttentionRouter)."""
        while self._running:
            if self._paused:
                await asyncio.sleep(self._cycle_interval)
                continue
            try:
                # Run a deep scan every cycle in legacy mode (same as old _run_cycle)
                await self._run_with_timeout(
                    self._run_deep_scan(), "deep_scan"
                )
                self._cycle_count += 1
                self._last_cycle_at = time.time()
                await asyncio.sleep(self._cycle_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CAPTAIN:ERROR] cycle={self._cycle_count + 1} error={e}\n{traceback.format_exc()}")
                self._record_error(str(e))
                await asyncio.sleep(30.0)

    async def _run_with_timeout(self, coro, mode: str) -> None:
        """Run a cycle coroutine with timeout and error tracking."""
        cycle_timeout = max(self._cycle_interval * 2, 120.0)
        try:
            await asyncio.wait_for(coro, timeout=cycle_timeout)
        except asyncio.TimeoutError:
            cycle_num = self._cycle_count + 1
            logger.warning(f"[CAPTAIN:TIMEOUT] cycle={cycle_num} mode={mode} exceeded {cycle_timeout:.0f}s")
            logger.info(f"[CAPTAIN:CYCLE_END] cycle={cycle_num} mode={mode} status=timeout duration={cycle_timeout:.0f}s")
            self._record_error(f"timeout_{mode}_{cycle_num}")

    async def _wait_for_trigger(
        self, last_strategic: float, last_deep_scan: float,
    ) -> str:
        """Wait for the next trigger: reactive event, strategic timer, or deep scan timer.

        Returns "reactive", "strategic", or "deep_scan".
        """
        router = self._attention_router
        now = time.time()

        # Check if deep scan or strategic are already due
        if now - last_deep_scan >= self._deep_scan_interval:
            return "deep_scan"
        if now - last_strategic >= self._strategic_interval:
            return "strategic"

        # Update positions in router for P&L scoring
        from .tools import _ctx as tool_ctx
        gateway = tool_ctx.gateway if tool_ctx else None
        if gateway and router:
            try:
                portfolio = await self._ctx.build_portfolio_state(gateway)
                positions_for_router = [
                    {
                        "ticker": p.ticker,
                        "event_ticker": p.event_ticker,
                        "side": p.side,
                        "quantity": p.quantity,
                        "pnl_per_ct": round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0,
                    }
                    for p in portfolio.positions
                ]
                router.update_positions(positions_for_router)
            except Exception:
                pass

        # Wait for attention router notification OR next timer, whichever comes first
        time_to_strategic = max(0.1, self._strategic_interval - (now - last_strategic))
        time_to_deep_scan = max(0.1, self._deep_scan_interval - (now - last_deep_scan))
        next_timer = min(time_to_strategic, time_to_deep_scan)

        try:
            await asyncio.wait_for(router.notify_event.wait(), timeout=next_timer)
            # Attention router fired — peek (not drain) to check if items qualify
            if router.has_items:
                return "reactive"
        except asyncio.TimeoutError:
            pass

        # Timer expired — figure out which one
        now = time.time()
        if now - last_deep_scan >= self._deep_scan_interval:
            return "deep_scan"
        return "strategic"

    # ------------------------------------------------------------------
    # Cycle pre-work (shared across all modes)
    # ------------------------------------------------------------------

    def _cycle_preamble(self) -> int:
        """Shared pre-cycle work. Returns cycle number."""
        cycle_num = self._cycle_count + 1

        # Reset sniper cycle counter
        if self._sniper_ref:
            self._sniper_ref.reset_cycle_counter()

        # Reset per-cycle capital budget
        from .tools import _ctx as tool_ctx_local
        if tool_ctx_local:
            tool_ctx_local.cycle_capital_spent_cents = 0

        # Reset per-cycle tool call tracking
        self._current_cycle_tools = {}

        # Tick task ledger (stale detection, pruning)
        self._task_ledger.tick(cycle_num)

        return cycle_num

    def _build_task_section(self) -> str:
        """Build task ledger section with replan hint if needed."""
        section = self._task_ledger.to_prompt_section() or ""
        if self._task_ledger.needs_replan():
            section += "\nREPLAN: Multiple stale tasks. Re-evaluate your plan."
        return section

    async def _cycle_postamble(self, mode: str, cycle_num: int, cycle_start: float,
                                portfolio: PortfolioState) -> None:
        """Shared post-cycle logging."""
        cycle_duration = time.time() - cycle_start
        balance_str = f"${portfolio.balance_dollars}" if portfolio.balance_cents else "?"
        tools_str = ",".join(f"{k}:{v}" for k, v in sorted(self._current_cycle_tools.items())) or "none"
        logger.info(
            f"[CAPTAIN:SUMMARY] cycle={cycle_num} mode={mode} duration={cycle_duration:.1f}s "
            f"positions={portfolio.total_positions} errors={len(self._errors)} balance={balance_str} "
            f"tools={tools_str}"
        )
        logger.info(f"[CAPTAIN:CYCLE_END] cycle={cycle_num} mode={mode} status=ok duration={cycle_duration:.1f}s")

        await self._emit_event({
            "type": "captain_cycle_complete",
            "data": {
                "mode": mode,
                "cycle_num": cycle_num,
                "duration_s": round(cycle_duration, 1),
            },
        })

    async def _get_portfolio(self) -> PortfolioState:
        """Build portfolio state (1-2 API calls)."""
        from .tools import _ctx as tool_ctx
        gateway = tool_ctx.gateway if tool_ctx else None
        if gateway:
            return await self._ctx.build_portfolio_state(gateway)
        return PortfolioState()

    async def _invoke_agent(self, mode: str, cycle_num: int, prompt: str, trigger_items: int = 0) -> None:
        """Shared pattern: emit cycle_start, run agent, emit completion events."""
        await self._emit_event({
            "type": "captain_cycle_start",
            "data": {"mode": mode, "cycle_num": cycle_num, "trigger_items": trigger_items},
        })
        await self._emit_event({
            "type": "agent_message",
            "subtype": "subagent_start",
            "agent": "single_arb_captain",
            "mode": mode,
            "prompt": prompt[:200],
        })

        result = await self._run_agent(prompt)

        await self._emit_event({
            "type": "agent_message",
            "subtype": "subagent_complete",
            "agent": "single_arb_captain",
            "mode": mode,
            "response_preview": str(result)[:500] if result else "",
        })

    # ------------------------------------------------------------------
    # Mode 1: REACTIVE — attention-driven, compact prompt
    # ------------------------------------------------------------------

    async def _run_reactive(self, items) -> None:
        """Reactive cycle: respond to high-urgency attention items."""
        from .models import AttentionItem
        cycle_num = self._cycle_preamble()
        cycle_start = time.time()
        self._reactive_count += 1

        portfolio = await self._get_portfolio()
        sniper_status = self._ctx.build_sniper_status(self._sniper_ref)

        # Build compact reactive context via context builder
        body = self._ctx.build_reactive_context(items, portfolio, sniper_status)
        prompt = f"Cycle #{cycle_num} REACTIVE.\n{body}"

        logger.info(
            f"[CAPTAIN:CYCLE_START] cycle={cycle_num} mode=reactive "
            f"items={len(items)} urgencies={[i.urgency for i in items]}"
        )

        await self._invoke_agent("reactive", cycle_num, prompt, trigger_items=len(items))
        await self._cycle_postamble("reactive", cycle_num, cycle_start, portfolio)

    # ------------------------------------------------------------------
    # Mode 2: STRATEGIC — every 5 min, planning and tuning
    # ------------------------------------------------------------------

    async def _run_strategic(self) -> None:
        """Strategic cycle: portfolio review, sniper tuning, task planning."""
        cycle_num = self._cycle_preamble()
        cycle_start = time.time()
        self._strategic_count += 1

        portfolio = await self._get_portfolio()
        sniper_status = self._ctx.build_sniper_status(self._sniper_ref)

        # Pending attention items (sub-threshold, for awareness)
        pending = []
        if self._attention_router:
            pending = self._attention_router.pending_items()

        body = self._ctx.build_strategic_context(portfolio, pending, sniper_status, self._build_task_section())
        prompt = f"Cycle #{cycle_num} STRATEGIC.\n{body}"

        logger.info(
            f"[CAPTAIN:CYCLE_START] cycle={cycle_num} mode=strategic "
            f"positions={portfolio.total_positions} pending_attention={len(pending)}"
        )

        await self._invoke_agent("strategic", cycle_num, prompt, trigger_items=len(pending))
        await self._cycle_postamble("strategic", cycle_num, cycle_start, portfolio)

    # ------------------------------------------------------------------
    # Mode 3: DEEP SCAN — every 30 min, comprehensive review
    # ------------------------------------------------------------------

    async def _run_deep_scan(self) -> None:
        """Deep scan cycle: all events, full positions, health, memories."""
        cycle_num = self._cycle_preamble()
        cycle_start = time.time()
        self._deep_scan_count += 1

        market_state = self._ctx.build_market_state()
        portfolio = await self._get_portfolio()
        sniper_status = self._ctx.build_sniper_status(self._sniper_ref)

        # Health snapshot
        health = None
        from .tools import _ctx as tool_ctx
        if tool_ctx and tool_ctx.health_service:
            try:
                hs = tool_ctx.health_service.get_health_status()
                health = {
                    "drawdown_pct": hs.drawdown_pct,
                    "total_realized_pnl_cents": hs.total_realized_pnl_cents,
                    "settlement_count_session": hs.settlement_count_session,
                }
            except Exception:
                pass

        # Trade outcome memories — what worked, what didn't
        trade_memories = None
        if tool_ctx and tool_ctx.memory:
            try:
                recalled = await tool_ctx.memory.recall(
                    query="trade outcomes profit loss executed cancelled",
                    limit=3, memory_types=["trade_outcome"],
                )
                if recalled and recalled.results:
                    trade_memories = [r.content[:120] for r in recalled.results]
            except Exception:
                pass

        # High-signal news memories — event-aware query using top tracked events
        news_memories = None
        if tool_ctx and tool_ctx.memory:
            try:
                top_tickers = [ev.event_ticker for ev in market_state.events[:3]]
                news_query = f"news price impact {' '.join(top_tickers)}"
                recalled = await tool_ctx.memory.recall(
                    query=news_query, limit=3, memory_types=["news"],
                )
                if recalled and recalled.results:
                    news_memories = [r.content[:120] for r in recalled.results]
            except Exception:
                pass

        # News-price impact learnings (market movers from last 48h)
        market_movers = None
        try:
            from kalshiflow_rl.data.database import rl_db
            pool = await rl_db.get_pool()
            async with pool.acquire() as conn:
                # Query across all monitored events, top 5 movers
                rows = await conn.fetch(
                    """SELECT am.news_title, npi.market_ticker, npi.event_ticker,
                              GREATEST(ABS(COALESCE(npi.change_1h_cents,0)),
                                       ABS(COALESCE(npi.change_4h_cents,0)),
                                       ABS(COALESCE(npi.change_24h_cents,0))) AS change_cents,
                              CASE WHEN COALESCE(npi.change_1h_cents,0) > 0 THEN 'up' ELSE 'down' END AS direction
                       FROM news_price_impacts npi
                       JOIN agent_memories am ON am.id = npi.news_memory_id
                       WHERE npi.created_at >= NOW() - INTERVAL '48 hours'
                         AND GREATEST(ABS(COALESCE(npi.change_1h_cents,0)),
                                      ABS(COALESCE(npi.change_4h_cents,0)),
                                      ABS(COALESCE(npi.change_24h_cents,0))) >= 3
                       ORDER BY change_cents DESC LIMIT 5"""
                )
                if rows:
                    market_movers = [dict(r) for r in rows]
        except Exception:
            pass  # DB not available or table empty — non-critical

        body = self._ctx.build_deep_scan_context(
            market_state, portfolio, sniper_status, health,
            trade_memories=trade_memories, news_memories=news_memories,
            task_section=self._build_task_section(), market_movers=market_movers,
        )
        prompt = f"Cycle #{cycle_num} DEEP_SCAN.\n{body}"

        logger.info(
            f"[CAPTAIN:CYCLE_START] cycle={cycle_num} mode=deep_scan "
            f"events={len(market_state.events)} positions={portfolio.total_positions}"
        )

        await self._invoke_agent("deep_scan", cycle_num, prompt)
        await self._cycle_postamble("deep_scan", cycle_num, cycle_start, portfolio)

    async def _run_agent(self, prompt: str) -> Optional[str]:
        """Run the agent, streaming categorized tool call events."""
        try:
            result_text = None
            token_buffer = []
            token_count = 0

            config = {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": 100,
            }

            async for event in self._agent.astream_events(
                {"messages": [HumanMessage(content=prompt)]},
                version="v2",
                config=config,
            ):
                kind = event.get("event")

                # Stream thinking tokens
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        text = self._extract_text(chunk.content)
                        if text:
                            token_buffer.append(text)
                        token_count += 1
                        if token_count % 10 == 0:
                            await self._emit_event({
                                "type": "agent_message",
                                "subtype": "thinking_delta",
                                "agent": "single_arb_captain",
                                "text": "".join(token_buffer),
                            })

                # Tool call start
                elif kind == "on_tool_start":
                    if token_buffer:
                        await self._emit_event({
                            "type": "agent_message",
                            "subtype": "thinking_complete",
                            "agent": "single_arb_captain",
                            "text": "".join(token_buffer),
                        })
                        token_buffer.clear()
                        token_count = 0

                    tool_name = event.get("name", "unknown")
                    tool_input = event.get("data", {}).get("input", "")
                    category = TOOL_CATEGORIES.get(tool_name, "system")

                    # Track tool calls per cycle
                    self._current_cycle_tools[tool_name] = self._current_cycle_tools.get(tool_name, 0) + 1

                    # Intercept write_todos: enrich via TaskLedger and broadcast
                    if tool_name == "write_todos":
                        raw_todos = tool_input.get("todos", []) if isinstance(tool_input, dict) else []
                        self._task_ledger.reconcile(raw_todos, self._cycle_count + 1)
                        await self._persist_tasks()
                        await self._emit_event({
                            "type": "agent_message",
                            "subtype": "todo_update",
                            "agent": "single_arb_captain",
                            "todos": self._task_ledger.to_broadcast(),
                        })
                        continue

                    await self._emit_event({
                        "type": "agent_message",
                        "subtype": "tool_call",
                        "category": category,
                        "agent": "single_arb_captain",
                        "tool_name": tool_name,
                        "tool_input": str(tool_input)[:200],
                    })

                # Tool call end
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    tool_output = event.get("data", {}).get("output", "")
                    category = TOOL_CATEGORIES.get(tool_name, "system")

                    if tool_name == "write_todos":
                        continue

                    await self._emit_event({
                        "type": "agent_message",
                        "subtype": "tool_result",
                        "category": category,
                        "agent": "single_arb_captain",
                        "tool_name": tool_name,
                        "tool_output": str(tool_output)[:300],
                    })

                # Extract final response
                elif kind == "on_chain_end":
                    if event.get("name") == "LangGraph":
                        output = event.get("data", {}).get("output", {})
                        messages = output.get("messages", [])
                        if messages:
                            result_text = self._extract_text(messages[-1].content)
                        # Fallback: extract todos from chain output
                        todos = output.get("todos")
                        if todos is not None:
                            self._task_ledger.reconcile(
                                todos if isinstance(todos, list) else [],
                                self._cycle_count + 1,
                            )
                            await self._persist_tasks()
                            await self._emit_event({
                                "type": "agent_message",
                                "subtype": "todo_update",
                                "agent": "single_arb_captain",
                                "todos": self._task_ledger.to_broadcast(),
                            })

            if token_buffer:
                await self._emit_event({
                    "type": "agent_message",
                    "subtype": "thinking_complete",
                    "agent": "single_arb_captain",
                    "text": "".join(token_buffer),
                })

            return result_text

        except Exception as e:
            logger.error(f"[CAPTAIN:ERROR] agent error: {e}")
            await self._emit_event({
                "type": "agent_message",
                "subtype": "subagent_error",
                "agent": "single_arb_captain",
                "error": str(e),
            })
            return None

    async def _persist_tasks(self):
        """Fire-and-forget task ledger persistence."""
        try:
            from kalshiflow_rl.data.database import rl_db
            pool = await rl_db.get_pool()
            self._task_ledger.persist(pool)
        except Exception:
            pass  # Non-critical audit trail

    @staticmethod
    def _extract_text(content) -> str:
        """Extract text from LLM content (string or list of blocks)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts)
        return ""

    async def _emit_event(self, event_data: Dict) -> None:
        """Emit event to frontend via callback."""
        if not self._event_callback:
            return
        try:
            event_data.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            await self._event_callback(event_data)
        except Exception as e:
            logger.warning(f"[CAPTAIN:BROADCAST_ERROR] {e}")

    def get_stats(self) -> Dict:
        """Get Captain stats."""
        has_router = self._attention_router is not None
        return {
            "running": self._running,
            "paused": self._paused,
            "cycle_count": self._cycle_count,
            "last_cycle_at": self._last_cycle_at,
            "errors": self._errors[-5:],
            "model": self._model_name,
            "cycle_interval": self._cycle_interval,
            "version": "v2",
            "attention_driven": has_router,
            "mode_counts": {
                "reactive": self._reactive_count,
                "strategic": self._strategic_count,
                "deep_scan": self._deep_scan_count,
            },
            "intervals": {
                "strategic": self._strategic_interval,
                "deep_scan": self._deep_scan_interval,
            },
        }
