"""Captain V2 - Single-agent LLM Captain for single-event arbitrage.

Key simplifications over V1:
- Single agent with 10 tools (no subagents, no tool duplication)
- Structured JSON context (MarketState + PortfolioState + SniperStatus)
- Session memory via FAISS (no markdown files)
- ~400 tokens system prompt (down from ~800 + 500-2000 memory files)
- 60-75% fewer input tokens per cycle
"""

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.messages import HumanMessage
from langgraph.cache.memory import InMemoryCache

from .context_builder import ContextBuilder
from .models import MarketState, PortfolioState, SniperStatus, CycleDiff
from .task_ledger import TaskLedger
from .tools import ALL_TOOLS, TOOL_CATEGORIES

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.captain")

CAPTAIN_PROMPT_TEMPLATE = """You are the Captain — an autonomous Kalshi prediction market trader.
Your job: extract profit from mispriced prediction markets. Act on edge. Manage positions. Build memory.

PRICING: cents (0-100). YES@60c costs $0.60, pays $1.00 if YES wins. Tool inputs use cents. Balance in briefing is DOLLARS.

EACH CYCLE you receive live JSON: MARKET_STATE, PORTFOLIO, SNIPER, CHANGES.

DECISION RULES:
- Edge >= 5c AND spread < 10c → execute_arb or place_order (5-25 contracts based on liquidity)
- Edge 2-5c AND spread < 8c → place_order (1-5 contracts)
- Position PnL > +10c/contract → take profit (sell)
- Position PnL < -12c/contract → cut loss (sell)
- Position with < 1h to close → exit unless high conviction
- No edge above 2c → configure sniper, update todos, move on

FLOW: write_todos → check exits → scan edge → research if needed → trade → store_insight

TASK PLANNING: write_todos EVERY cycle. Without it you lose all context between cycles.
Track: positions to manage, edge opportunities, sniper config. Prefix [HIGH]/[MED]/[LOW].

EXITS: sell the side you hold. NEVER buy the opposite side to "hedge."
SNIPER: Auto-executes S1_ARB. You CONFIGURE it (edge threshold, capital, cooldown), it EXECUTES.
REGIME: Per-market vpin/sweep is informational. Only hard stops: negative edge, spread > 15c.

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
    """Single-agent LLM Captain for single-event arbitrage.

    Uses create_deep_agent with 10 tools (no subagents).
    Structured JSON context injected each cycle.
    Session memory via FAISS + pgvector.
    """

    def __init__(
        self,
        context_builder: ContextBuilder,
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
        self._errors: List[str] = []
        self._prev_market_state: Optional[MarketState] = None
        self._full_scan_ts: Dict[str, float] = {}  # event_ticker -> last full scan timestamp
        self._task_ledger = TaskLedger(session_id=str(uuid.uuid4()))

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

    async def _run_loop(self) -> None:
        """Main cycle loop."""
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

        while self._running:
            if self._paused:
                await asyncio.sleep(self._cycle_interval)
                continue
            try:
                cycle_timeout = max(self._cycle_interval * 2, 120.0)
                await asyncio.wait_for(self._run_cycle(), timeout=cycle_timeout)
                self._cycle_count += 1
                self._last_cycle_at = time.time()
                await asyncio.sleep(self._cycle_interval)
            except asyncio.TimeoutError:
                logger.warning(f"[CAPTAIN:TIMEOUT] cycle={self._cycle_count + 1} exceeded {cycle_timeout:.0f}s")
                logger.info(f"[CAPTAIN:CYCLE_END] cycle={self._cycle_count + 1} status=timeout")
                self._errors.append(f"cycle_timeout_{self._cycle_count + 1}")
                if len(self._errors) > 50:
                    self._errors = self._errors[-50:]
                self._cycle_count += 1
                self._last_cycle_at = time.time()
                await asyncio.sleep(self._cycle_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[CAPTAIN:ERROR] cycle={self._cycle_count + 1} error={e}\n{traceback.format_exc()}")
                self._errors.append(str(e))
                if len(self._errors) > 50:
                    self._errors = self._errors[-50:]
                await asyncio.sleep(30.0)

    async def _run_cycle(self) -> None:
        """Run one Captain cycle with structured JSON context."""
        cycle_num = self._cycle_count + 1
        cycle_start = time.time()

        # Reset sniper cycle counter
        if self._sniper_ref:
            self._sniper_ref.reset_cycle_counter()

        # Reset per-cycle capital budget
        from .tools import _ctx as tool_ctx_local
        if tool_ctx_local:
            tool_ctx_local.cycle_capital_spent_cents = 0

        # Tick task ledger (stale detection, pruning)
        self._task_ledger.tick(cycle_num)

        # Build structured context (0 API calls for market data)
        market_state = self._ctx.build_market_state()

        # Build portfolio (1-2 API calls)
        from .tools import _ctx as tool_ctx
        gateway = tool_ctx.gateway if tool_ctx else None
        portfolio = await self._ctx.build_portfolio_state(gateway) if gateway else PortfolioState()

        # Build sniper status (0 API calls)
        sniper_status = self._ctx.build_sniper_status(self._sniper_ref)

        # Compute diffs since last cycle
        diffs = self._ctx.compute_diffs(market_state)

        # Build the prompt with structured JSON context
        FULL_SCAN_TTL = 3600  # 1 hour

        # Find best opportunity across all events for headline
        best_edge = None
        best_event = None
        for ev in market_state.events:
            edge = max(ev.long_edge or 0, ev.short_edge or 0)
            if best_edge is None or edge > best_edge:
                best_edge = edge
                best_event = ev

        context_parts = [f"Cycle #{cycle_num}."]
        if best_event and best_edge and best_edge > 0:
            direction = "long" if (best_event.long_edge or 0) >= (best_event.short_edge or 0) else "short"
            context_parts.append(
                f"BEST_EDGE: {best_event.event_ticker} {direction} edge={best_edge:.1f}c "
                f"regime={best_event.regime}"
            )

        # Compact market state with TTL-based full scan
        now = time.time()
        compact_events = []
        for ev in market_state.events:
            compact_ev = {
                "event_ticker": ev.event_ticker,
                "title": ev.title,
                "me": ev.mutually_exclusive,
                "markets": ev.market_count,
                "long_edge": ev.long_edge,
                "short_edge": ev.short_edge,
                "vol_5m": ev.total_volume_5m,
                "regime": ev.regime,
                "ttc_hours": ev.time_to_close_hours,
            }
            if ev.semantics and ev.semantics.what:
                compact_ev["context"] = ev.semantics.what[:100]

            # TTL-based market selection: full scan on first see or TTL expiry, top 10 otherwise
            all_mkts = sorted(ev.markets.values(), key=lambda m: m.volume_5m, reverse=True)
            last_scan = self._full_scan_ts.get(ev.event_ticker, 0)
            if now - last_scan > FULL_SCAN_TTL:
                show_mkts = all_mkts
                self._full_scan_ts[ev.event_ticker] = now
            else:
                show_mkts = all_mkts[:10]

            compact_ev["top_markets"] = [
                {"t": m.ticker, "n": m.title[:40] if m.title else None,
                 "bid": m.yes_bid, "ask": m.yes_ask, "sp": m.spread,
                 "mp": round(m.microprice, 1) if m.microprice else None,
                 "vol": m.volume_5m, "vpin": m.vpin, "regime": m.regime}
                for m in show_mkts
            ]
            compact_events.append(compact_ev)

        context_parts.append(f"MARKET_STATE: {json.dumps(compact_events, separators=(',', ':'))}")

        # Portfolio
        if portfolio.positions:
            compact_portfolio = {
                "balance_dollars": f"${portfolio.balance_dollars:,.2f}",
                "positions": portfolio.total_positions,
                "pnl": f"${portfolio.total_unrealized_pnl_cents / 100:+.2f}",
                "top": [
                    {"t": p.ticker, "side": p.side, "qty": p.quantity,
                     "exit": p.exit_price, "pnl": p.unrealized_pnl_cents,
                     "pnl_ct": round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0}
                    for p in portfolio.positions[:5]
                ],
            }
            context_parts.append(f"PORTFOLIO: {json.dumps(compact_portfolio, separators=(',', ':'))}")

            # Time pressure flag for positions nearing settlement
            time_flags = []
            for ev in market_state.events:
                if ev.time_to_close_hours is not None and ev.time_to_close_hours < 1.0:
                    for p in portfolio.positions:
                        if p.event_ticker == ev.event_ticker and p.quantity > 0:
                            time_flags.append(f"{p.ticker} ttc={ev.time_to_close_hours:.1f}h")
            if time_flags:
                context_parts.append(f"TIME_PRESSURE: {', '.join(time_flags)}")
        else:
            context_parts.append(f"PORTFOLIO: balance=${portfolio.balance_dollars:,.2f} ({portfolio.balance_cents} cents), no positions.")

        # Sniper: only inject detail if there was activity, otherwise one word
        if sniper_status.enabled:
            has_sniper_activity = (
                sniper_status.total_trades > 0
                or sniper_status.last_action_summary
                or sniper_status.last_rejection_reason
            )
            if has_sniper_activity:
                compact_sniper = {
                    "trades": sniper_status.total_trades,
                    "arbs": sniper_status.total_arbs_executed,
                    "capital_active": sniper_status.capital_in_flight + sniper_status.capital_in_positions,
                    "unwinds": sniper_status.total_partial_unwinds,
                }
                if sniper_status.last_rejection_reason:
                    compact_sniper["last_reject"] = sniper_status.last_rejection_reason
                if sniper_status.last_action_summary:
                    compact_sniper["last_exec"] = sniper_status.last_action_summary
                context_parts.append(f"SNIPER: {json.dumps(compact_sniper, separators=(',', ':'))}")
            else:
                context_parts.append("SNIPER: enabled")
        else:
            context_parts.append("SNIPER: OFF")

        # Health: only inject if drawdown > 5% or alerts exist
        if tool_ctx and tool_ctx.health_service:
            try:
                hs = tool_ctx.health_service.get_health_status()
                if hs.drawdown_pct > 5.0 or hs.alerts:
                    context_parts.append(
                        f"HEALTH: {{\"drawdown\":\"{hs.drawdown_pct}%\","
                        f"\"realized_pnl\":{hs.total_realized_pnl_cents},"
                        f"\"settlements\":{hs.settlement_count_session}}}"
                    )
            except Exception:
                pass

        # Diffs
        if diffs.has_changes:
            diff_parts = []
            if diffs.price_moves:
                diff_parts.append(f"price_moves: {', '.join(diffs.price_moves)}")
            if diffs.volume_spikes:
                diff_parts.append(f"volume: {', '.join(diffs.volume_spikes)}")
            context_parts.append(f"CHANGES ({int(diffs.elapsed_seconds)}s): {'; '.join(diff_parts)}")

        # Inject relevant past memories for events with edge
        if tool_ctx and tool_ctx.memory:
            memory_hints = []
            for ev in market_state.events:
                has_edge = (ev.long_edge and ev.long_edge > 2.0) or (ev.short_edge and ev.short_edge > 2.0)
                if has_edge:
                    try:
                        recalled = await tool_ctx.memory.recall(
                            query=f"trades outcomes {ev.event_ticker}", limit=2,
                        )
                        if recalled and recalled.results:
                            for r in recalled.results:
                                memory_hints.append(f"[{ev.event_ticker}] {r.content[:120]}")
                    except Exception:
                        pass
            if memory_hints:
                context_parts.append("MEMORY:\n" + "\n".join(memory_hints[:5]))

        # Session memory: inject journal of what we've done/learned this session
        if tool_ctx and hasattr(tool_ctx.memory, "journal_summary"):
            journal = tool_ctx.memory.journal_summary(max_entries=5)
            if journal:
                context_parts.append(journal)

        # Task ledger: inject active tasks from previous cycles
        task_section = self._task_ledger.to_prompt_section()
        if task_section:
            context_parts.append(task_section)
        if self._task_ledger.needs_replan():
            context_parts.append("REPLAN: Multiple stale tasks. Re-evaluate your plan.")

        logger.info(f"[CAPTAIN:CYCLE_START] cycle={cycle_num}")
        prompt = "\n".join(context_parts)

        await self._emit_event({
            "type": "agent_message",
            "subtype": "subagent_start",
            "agent": "single_arb_captain",
            "prompt": prompt[:200],
        })

        result = await self._run_agent(prompt)

        await self._emit_event({
            "type": "agent_message",
            "subtype": "subagent_complete",
            "agent": "single_arb_captain",
            "response_preview": str(result)[:500] if result else "",
        })

        cycle_duration = time.time() - cycle_start
        balance_str = f"${portfolio.balance_dollars}" if portfolio.balance_cents else "?"
        logger.info(
            f"[CAPTAIN:SUMMARY] cycle={cycle_num} duration={cycle_duration:.1f}s "
            f"events={len(market_state.events)} positions={portfolio.total_positions} "
            f"errors={len(self._errors)} balance={balance_str}"
        )
        logger.info(f"[CAPTAIN:CYCLE_END] cycle={cycle_num} status=ok duration={cycle_duration:.1f}s")

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

                    if tool_name == "execute":
                        continue

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

                    if tool_name in ("execute", "write_todos"):
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
        return {
            "running": self._running,
            "paused": self._paused,
            "cycle_count": self._cycle_count,
            "last_cycle_at": self._last_cycle_at,
            "errors": self._errors[-5:],
            "model": self._model_name,
            "cycle_interval": self._cycle_interval,
            "version": "v2",
        }
