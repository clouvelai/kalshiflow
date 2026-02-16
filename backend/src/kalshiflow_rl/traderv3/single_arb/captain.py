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

from deepagents.backends import StateBackend
from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware, _compute_summarization_defaults
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import init_chat_model
from langchain_anthropic.middleware import AnthropicPromptCachingMiddleware
from langchain_core.messages import HumanMessage
from langgraph.cache.memory import InMemoryCache

from .context_builder import ContextBuilder
from .models import MarketState, PortfolioState, SniperStatus
from .task_ledger import TaskLedger
from .tools import ALL_TOOLS, TOOL_CATEGORIES

if TYPE_CHECKING:
    from .attention import AttentionRouter

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.captain")


# LLM cost rates per 1M tokens (approximate, for visibility logging)
_MODEL_COST_RATES = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.8, "output": 4.0},
    "gemini-2.0-flash": {"input": 0.1, "output": 0.4},
}


CAPTAIN_PROMPT_TEMPLATE = """You are the Captain — an autonomous Kalshi prediction market trader.
Your edge: fast news reaction, early bird positioning on new markets, event expertise. Not speed.

UNITS: Cents (0-100). YES@60c = $0.60 cost, $1.00 payout. Briefing balance is DOLLARS. Tool prices are CENTS.

STRATEGY PATHS (choose per opportunity):

EARLY_BIRD (highest priority, time-critical):
  complement strategy (fair_value provided):
    1. place_order: YES limit at fair_value-2c
    2. place_order: NO limit at (100-fair_value)-2c
    3. Size {eb_complement_size}ct. Both orders are maker = 0% fees. The 4c spread IS profit.
    4. If one fills, cancel the other. If both fill, you captured the spread.
    DO NOT search_news first. Trust the deterministic fair_value. Act in seconds.
  captain_decide strategy (no fair_value):
    1. search_news on event title
    2. get_market_state to see orderbook
    3. Estimate fair price, place resting limit at your estimate
    4. Size {eb_decide_size}ct. store_insight with your reasoning.

NEWS-DRIVEN (your second-best edge):
  1. search_news with specific query (entity + event + timeframe)
  2. Assess: does this news shift probability vs. current price?
  3. If shift > 5c: place_order at your estimate. Size per edge (see SIZING in guidance).
  4. store_insight: "EVENT:[ticker] | news=[headline] | shift=[+/-Xc] | confidence=[H/M/L]"

POSITION MANAGEMENT (auto-actions handle defaults, you handle overrides):
  Auto-actions are ACTIVE and fire autonomously:
  - stop_loss: Triggers when any position's per-contract loss exceeds the threshold (configurable via configure_automation)
  - time_exit: Triggers when a position's time-to-close falls below the threshold (configurable via configure_automation)
  - regime_gate: Blocks new entries when market regime shifts are detected
  You can tune these with the configure_automation tool. They run on every attention cycle.
  Your job: override when you have BETTER INFORMATION (recent news, event knowledge).
  To exit: place_order(side=[side you hold], action="sell"). NEVER buy opposite side.

ME ARB (Sniper handles this autonomously):
  Sniper auto-executes ME arb when edge detected. You monitor via sniper stats in briefing.
  Captain does NOT call execute_arb for routine ME arb — Sniper is faster.
  Use execute_arb ONLY for manual arb when you spot an edge Sniper missed.

MARKET MAKING (QuoteEngine runs autonomously — you add the information edge):
  QuoteEngine posts 2-sided quotes using microstructure (no news). YOUR job: supply direction.
  Tools: configure_quotes, pull_quotes, resume_quotes, get_quote_performance, search_news, recall_memory

  REACTIVE (mm_* attention signals):
    mm_fill: get_quote_performance. If fills are one-sided (bid >> ask or vice versa), the market
      is moving directionally. search_news on the MM event to find why. Adjust skew via
      configure_quotes(skew_factor=...) to lean INTO the informed flow, or pull_quotes if news is severe.
    mm_vpin_spike: Do NOT auto-pull. Kalshi VPIN is structurally high. Instead:
      1. search_news on the MM event. If adverse news found → pull_quotes.
      2. If no news → configure_quotes(base_spread_cents=+2) to widen spread. Resume normal next cycle.
    mm_inventory_warning: get_quote_performance to see which side is accumulating.
      configure_quotes(skew_factor=...) to push quotes toward reducing inventory.
    mm_fill_storm: pull_quotes immediately. search_news for catalyst. Resume only after reviewing.
    mm_spread_change: Informational. Note in store_insight if spread regime changed.

  STRATEGIC (every 5 min when MM active):
    1. Always get_quote_performance. Check: adverse_selection > 60% of spread_captured? → problem.
    2. If adverse is high: search_news on MM event tickers (listed in briefing) for directional risk.
    3. If news found: configure_quotes(skew_factor=...) per this logic:
       - Bullish news + long inventory → reduce skew (let it ride)
       - Bullish news + short inventory → increase skew toward buying YES
       - Bearish news → mirror the above
       - No news + one-sided fills → widen spread by 1-2c
    4. If quotes are PULLED: evaluate whether to resume_quotes based on news + time since pull.
    5. store_insight: "MM:[event] | adverse_ratio=X | spread=Yc | action=[what you changed]"

  DEEP_SCAN (every 30 min — full MM review):
    1. get_quote_performance for session totals.
    2. recall_memory("MM performance") for historical patterns on these events.
    3. search_news on EACH MM event ticker for directional positioning.
    4. Review per-market inventory in briefing. For each non-zero position:
       - Is the position intentional (from skew) or accidental (from adverse fills)?
       - Should you adjust max_position or skew_cap_cents?
    5. Compare current spread to fee breakeven (~1.4c for 50c fair value). If spread < breakeven+1c, widen.
    6. store_insight: "MM_REVIEW:[event] | net_pnl=Xc | adverse_ratio=Y | action=[changes]"
    7. Task ledger: "[MED] Monitor [event] MM — last news was [headline]" for multi-cycle tracking.

MODE BEHAVIOR:
  REACTIVE (1-3 tool calls, <45s): Respond to ATTENTION items. Trade, exit, or note why you pass.
    For mm_* signals: follow MM reactive playbook above — never blindly pull on VPIN.
    Do NOT research unrelated events, write tasks, or call get_portfolio.
  STRATEGIC (3-8 tool calls, <2min): search_news on 1-2 active events. Check early_bird.
    When MM active: ALWAYS get_quote_performance and tune if needed (see MM STRATEGIC above).
    Manage positions. Update task ledger with concrete next actions.
  DEEP_SCAN (5-15 tool calls, <3min): Full review. search_news on top events.
    Full MM review with recall_memory + per-event news search (see MM DEEP_SCAN above).
    Review all positions. Check early_bird. Recall memories for patterns. Update tasks.

BRIEFING DATA (already in context — do NOT re-fetch unless data changed mid-cycle):
  REACTIVE: attention items, relevant positions, sniper status, early_bird detail
  STRATEGIC: portfolio (5 pos), health, early_bird, pending attention, sniper, MM summary + event tickers, tasks
  DEEP_SCAN: all events (compact), all positions, sniper perf, health, early_bird,
    trade memories, news memories, news impact, news patterns, MM detail + per-market inventory, tasks

MEMORY STRATEGY (build a knowledge graph over time):
  STORE (via store_insight, always include event_ticker):
    - Price-news correlations: "NEWS: [headline] moved [ticker] [+/-]Xc in Yh"
    - Event observations: "[event] complement pricing held/diverged by Xc"
    - Timing patterns: "[category] markets typically settle [pattern]"
    - MM telemetry: "MM:[event] | adverse_ratio=X | spread=Yc | skew=Z | net_pnl=Wc"
    - MM directional learnings: "MM:[event] | [headline] caused [bid/ask]-heavy fills for Xmin"
    DO NOT store: behavioral rules, self-restrictions, recovery plans, emotions.
  RECALL (via recall_memory):
    - Before trading an event: recall past observations on that event_ticker
    - Before searching news: recall what you already know (avoid redundant searches)
    - Before adjusting MM params: recall "MM [event_ticker]" for past tuning outcomes
    - In DEEP_SCAN: recall "price patterns [category]" for strategic insights

TASK LEDGER (externalized working memory — use write_todos):
  Short-term tasks (this session): [HIGH] things to do THIS cycle or next
  Long-term research (multi-cycle): [MED] events to monitor, patterns to track
  Stale tasks get marked [STALE] automatically after 10 unchanged cycles.
  Good tasks: "[HIGH] Search news on KXEVENT-ABC (earnings tomorrow)"
  Bad tasks: "[HIGH] Improve trading accuracy" (unmeasurable, leads to toxic loops)

VPIN: Structurally 0.8-1.0 on Kalshi due to thin books. NORMAL. Not a reason to stop trading.

RULES:
  - BEFORE any place_order or execute_arb: state (1) bull case, (2) bear case, (3) why bull wins.
    If you cannot rebut the bear case, DO NOT TRADE. This is mandatory for every trade.
  - If losing, find the next signal. Losses are data, not crises.
  - Every cycle MUST include at least one tool call.
  - When in doubt between more analysis and placing a small order, place the order.
  - store_insight: factual observations ONLY (prices, news, correlations, timestamps).

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


def _build_sizing_table(config) -> str:
    """Generate sizing table from config for prompt injection."""
    s = {
        "eb_complement_size": getattr(config, "captain_eb_complement_size", "100-250"),
        "eb_decide_size": getattr(config, "captain_eb_decide_size", "50-150"),
        "news_size_small": getattr(config, "captain_news_size_small", "10-25"),
        "news_size_medium": getattr(config, "captain_news_size_medium", "25-50"),
        "news_size_large": getattr(config, "captain_news_size_large", "50-100"),
        "max_contracts": getattr(config, "captain_max_contracts_per_market", 200),
        "max_capital_pct": getattr(config, "captain_max_capital_pct_per_event", 20),
    }
    return (
        "## Sizing\n"
        "| Edge     | Contracts | Condition |\n"
        "|----------|-----------|-------------------------------|\n"
        f"| >= 10c   | {s['news_size_large']} | Verify depth > 20 at level |\n"
        f"| 5-10c    | {s['news_size_medium']} |  |\n"
        f"| 2-5c     | {s['news_size_small']} |  |\n"
        f"| Early bird complement | {s['eb_complement_size']} | Fair value provided |\n"
        f"| Early bird captain_decide | {s['eb_decide_size']} | News-based estimate |\n"
        f"\nLimits: Max {s['max_capital_pct']}% capital per event. "
        f"Max {s['max_contracts']} contracts per market."
    )


class ArbCaptain:
    """Attention-driven LLM Captain for single-event arbitrage.

    Uses create_agent with 13 tools + hand-picked middleware (no filesystem, no subagents).
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

        # Load guidance once at agent creation (cached in system prompt for Anthropic prompt caching)
        guidance_text = _load_guidance()
        sizing_table = _build_sizing_table(config) if config else _build_sizing_table(None)
        guidance_section = f"TRADING GUIDANCE:\n{sizing_table}\n\n{guidance_text}" if guidance_text else ""

        # Resolve sizing values for prompt template placeholders
        sizing = {
            "eb_complement_size": getattr(config, "captain_eb_complement_size", "100-250") if config else "100-250",
            "eb_decide_size": getattr(config, "captain_eb_decide_size", "50-150") if config else "50-150",
        }
        system_prompt = CAPTAIN_PROMPT_TEMPLATE.format(guidance_section=guidance_section, **sizing)

        # Resolve model for middleware that need it
        resolved_model = init_chat_model(model_name)
        summarization_defaults = _compute_summarization_defaults(resolved_model)
        backend_factory = lambda rt: StateBackend(rt)

        # Middleware stack: only what Captain needs (no filesystem, no subagents)
        captain_middleware = [
            TodoListMiddleware(),
            SummarizationMiddleware(
                model=resolved_model,
                backend=backend_factory,
                trigger=summarization_defaults["trigger"],
                keep=summarization_defaults["keep"],
                trim_tokens_to_summarize=None,
                truncate_args_settings=summarization_defaults["truncate_args_settings"],
            ),
            AnthropicPromptCachingMiddleware(unsupported_model_behavior="ignore"),
            PatchToolCallsMiddleware(),
        ]

        self._captain_middleware = captain_middleware
        self._agent = create_agent(
            resolved_model,
            tools=ALL_TOOLS,
            system_prompt=system_prompt,
            middleware=captain_middleware,
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

        # Backoff: consecutive LLM failures
        self._consecutive_failures = 0

        # Consecutive timeout tracking for escalation
        self._consecutive_timeouts: int = 0

        # Per-cycle token tracking (reset in _cycle_preamble)
        self._cycle_input_tokens = 0
        self._cycle_output_tokens = 0

        # Portfolio cache to avoid redundant API calls in _wait_for_trigger
        self._cached_portfolio: Optional[PortfolioState] = None
        self._portfolio_cached_at: float = 0

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

    def _failure_backoff(self) -> float:
        """Compute sleep duration based on consecutive LLM failures.

        Returns seconds to sleep:
          0-2 failures: 30s (normal)
          3-5 failures: 300s (5 min)
          6+  failures: 900s (15 min)
        """
        count = self._consecutive_failures
        if count <= 2:
            return 30.0
        if count <= 5:
            return 300.0
        return 900.0

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
                sleep_s = self._failure_backoff()
                await asyncio.sleep(sleep_s)

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
                sleep_s = self._failure_backoff()
                await asyncio.sleep(sleep_s)

    async def _run_with_timeout(self, coro, mode: str) -> None:
        """Run a cycle coroutine with mode-specific timeout."""
        mode_timeouts = {"reactive": 45.0, "strategic": 120.0, "deep_scan": 180.0}
        cycle_timeout = mode_timeouts.get(mode, 120.0)
        try:
            await asyncio.wait_for(coro, timeout=cycle_timeout)
        except asyncio.CancelledError:
            # External cancellation (from stop()) — propagate, don't treat as timeout
            raise
        except asyncio.TimeoutError:
            cycle_num = self._cycle_count + 1
            self._consecutive_timeouts += 1
            logger.warning(f"[CAPTAIN:TIMEOUT] cycle={cycle_num} mode={mode} exceeded {cycle_timeout:.0f}s (consecutive={self._consecutive_timeouts})")
            logger.info(f"[CAPTAIN:CYCLE_END] cycle={cycle_num} mode={mode} status=timeout duration={cycle_timeout:.0f}s")
            self._record_error(f"timeout_{mode}_{cycle_num}")
            # Emit timeout event so frontend knows the cycle ended
            await self._emit_event({
                "type": "captain_cycle_complete",
                "data": {
                    "mode": mode,
                    "cycle_num": cycle_num,
                    "duration_s": cycle_timeout,
                    "status": "timeout",
                },
            })
            # Escalate if 3+ consecutive timeouts — pause and auto-resume after 15 min
            if self._consecutive_timeouts >= 3:
                logger.error(f"[CAPTAIN] {self._consecutive_timeouts} consecutive timeouts - pausing for recovery")
                await self._emit_event({
                    "type": "captain_timeout_escalation",
                    "data": {"consecutive": self._consecutive_timeouts},
                })
                self._paused = True
                asyncio.get_event_loop().call_later(900, lambda: setattr(self, '_paused', False))

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

        # Update positions in router for P&L scoring (cached to avoid redundant API calls)
        from .tools import _ctx as tool_ctx
        gateway = tool_ctx.gateway if tool_ctx else None
        portfolio_ttl = getattr(self._config, 'portfolio_cache_ttl', 15.0) if self._config else 15.0
        if gateway and router and time.time() - self._portfolio_cached_at > portfolio_ttl:
            try:
                portfolio = await self._ctx.build_portfolio_state(gateway)
                self._cached_portfolio = portfolio
                self._portfolio_cached_at = time.time()
                router.update_positions(self._portfolio_to_router_positions(portfolio))
            except Exception:
                pass
        elif self._cached_portfolio and router:
            router.update_positions(self._portfolio_to_router_positions(self._cached_portfolio))

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

    @staticmethod
    def _portfolio_to_router_positions(portfolio: PortfolioState) -> List[Dict]:
        """Convert PortfolioState positions to the format AttentionRouter expects."""
        return [
            {
                "ticker": p.ticker,
                "event_ticker": p.event_ticker,
                "side": p.side,
                "quantity": p.quantity,
                "pnl_per_ct": round(p.unrealized_pnl_cents / p.quantity) if p.quantity else 0,
            }
            for p in portfolio.positions
        ]

    # ------------------------------------------------------------------
    # Cycle pre-work (shared across all modes)
    # ------------------------------------------------------------------

    def _cycle_preamble(self, mode: str = "deep_scan") -> int:
        """Shared pre-cycle work. Returns cycle number."""
        cycle_num = self._cycle_count + 1

        # Reset sniper cycle counter
        if self._sniper_ref:
            self._sniper_ref.reset_cycle_counter()

        # Reset per-cycle capital budget and set cycle mode
        from .tools import _ctx as tool_ctx_local
        if tool_ctx_local:
            tool_ctx_local.cycle_capital_spent_cents = 0
            tool_ctx_local.cycle_mode = mode

        # Reset per-cycle tool call tracking
        self._current_cycle_tools = {}

        # Reset per-cycle token tracking
        self._cycle_input_tokens = 0
        self._cycle_output_tokens = 0

        # Clear node-level cache between cycles
        self._cache.clear()

        # Tick task ledger (stale detection, pruning)
        self._task_ledger.tick(cycle_num)

        return cycle_num

    def _build_task_section(self) -> str:
        """Build task ledger section for prompt injection."""
        return self._task_ledger.to_prompt_section() or ""

    async def _cycle_postamble(self, mode: str, cycle_num: int, cycle_start: float,
                                portfolio: PortfolioState) -> None:
        """Shared post-cycle logging."""
        # Successful cycle completion — reset timeout escalation
        self._consecutive_timeouts = 0
        cycle_duration = time.time() - cycle_start
        balance_str = f"${portfolio.balance_dollars}" if portfolio.balance_cents else "?"
        tools_str = ",".join(f"{k}:{v}" for k, v in sorted(self._current_cycle_tools.items())) or "none"

        # Estimate LLM cost for this cycle
        active_model = self._model_name
        rates = _MODEL_COST_RATES.get(active_model, {"input": 3.0, "output": 15.0})
        est_cost = (
            self._cycle_input_tokens * rates["input"] / 1_000_000
            + self._cycle_output_tokens * rates["output"] / 1_000_000
        )

        logger.info(
            f"[CAPTAIN:SUMMARY] cycle={cycle_num} mode={mode} duration={cycle_duration:.1f}s "
            f"positions={portfolio.total_positions} errors={len(self._errors)} balance={balance_str} "
            f"tools={tools_str}"
        )
        logger.info(
            f"[CAPTAIN:LLM_COST] cycle={cycle_num} model={active_model} "
            f"tokens_in={self._cycle_input_tokens} tokens_out={self._cycle_output_tokens} "
            f"est_cost=${est_cost:.4f}"
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
        """Build portfolio state, reusing cache if fresh."""
        portfolio_ttl = getattr(self._config, 'portfolio_cache_ttl', 15.0) if self._config else 15.0
        if self._cached_portfolio and time.time() - self._portfolio_cached_at < portfolio_ttl:
            return self._cached_portfolio
        from .tools import _ctx as tool_ctx
        gateway = tool_ctx.gateway if tool_ctx else None
        if gateway:
            portfolio = await self._ctx.build_portfolio_state(gateway)
            self._cached_portfolio = portfolio
            self._portfolio_cached_at = time.time()
            return portfolio
        return PortfolioState()

    def _fetch_health_snapshot(self) -> Optional[Dict]:
        """Fetch health status from AccountHealthService. Returns None on failure."""
        from .tools import _ctx as tool_ctx
        if not (tool_ctx and tool_ctx.health_service):
            return None
        try:
            hs = tool_ctx.health_service.get_health_status()
            return {
                "drawdown_pct": hs.drawdown_pct,
                "total_realized_pnl_cents": hs.total_realized_pnl_cents,
                "settlement_count_session": hs.settlement_count_session,
            }
        except Exception:
            return None

    def _fetch_early_bird(self, mode: str) -> Optional[list]:
        """Fetch early bird opportunities. Returns None on failure."""
        from .tools import _ctx as tool_ctx
        if not (tool_ctx and tool_ctx.early_bird_service):
            return None
        try:
            return tool_ctx.early_bird_service.get_recent_opportunities()
        except Exception as e:
            logger.debug(f"[CAPTAIN:{mode.upper()}] Early bird fetch failed: {e}")
            return None

    def _fetch_decision_accuracy(self) -> Optional[Dict]:
        """Fetch cached decision accuracy stats. Returns None on failure."""
        from .tools import _ctx as tool_ctx
        if not (tool_ctx and tool_ctx.health_service):
            return None
        try:
            cached = tool_ctx.health_service._cached_accuracy
            if cached:
                return cached.model_dump() if hasattr(cached, "model_dump") else cached
        except Exception:
            pass
        return None

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

        result = await self._run_agent(prompt, mode=mode)

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
        cycle_num = self._cycle_preamble(mode="reactive")
        cycle_start = time.time()
        self._reactive_count += 1

        # Bust portfolio cache for fresh data in reactive mode
        self._portfolio_cached_at = 0
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

    def _get_mm_state(self) -> Optional[dict]:
        """Get QuoteEngine state for context injection. Returns None if MM not active."""
        from .tools import _ctx as tool_ctx
        if not tool_ctx or not tool_ctx.quote_engine:
            return None
        try:
            state = tool_ctx.quote_engine.state
            mm_index = tool_ctx.mm_index
            realized = mm_index.total_realized_pnl() if mm_index else 0.0
            result = {
                "enabled": tool_ctx.quote_config.enabled if tool_ctx.quote_config else False,
                "mm_event_tickers": list(mm_index.events.keys()) if mm_index else [],
                "total_fills": state.total_fills_bid + state.total_fills_ask,
                "total_fills_bid": state.total_fills_bid,
                "total_fills_ask": state.total_fills_ask,
                "spread_captured_cents": state.spread_captured_cents,
                "adverse_selection_cents": state.adverse_selection_cents,
                "net_pnl_cents": realized,
                "quotes_pulled": state.quotes_pulled,
                "spread_multiplier": state.spread_multiplier,
            }
            # Per-market positions for deep scan
            if mm_index:
                positions = []
                for ticker in mm_index.market_tickers:
                    inv = mm_index.get_inventory(ticker)
                    if inv.position != 0 or inv.realized_pnl_cents != 0:
                        positions.append({
                            "ticker": ticker,
                            "position": inv.position,
                            "realized_pnl_cents": inv.realized_pnl_cents,
                        })
                result["positions"] = positions
            return result
        except Exception as e:
            logger.debug(f"[CAPTAIN] MM state fetch failed: {e}")
            return None

    def _has_actionable_state(self, portfolio: PortfolioState) -> bool:
        """Check if there's anything worth invoking the LLM for.

        Returns False when the index is empty AND portfolio has no positions,
        meaning the Captain would just analyze nothing and burn tokens.
        """
        index = self._ctx._index
        has_events = bool(index and index.events)
        has_positions = portfolio.total_positions > 0
        return has_events or has_positions

    async def _run_strategic(self) -> None:
        """Strategic cycle: portfolio review, sniper tuning, task planning."""
        cycle_num = self._cycle_preamble(mode="strategic")
        cycle_start = time.time()
        self._strategic_count += 1

        portfolio = await self._get_portfolio()
        sniper_status = self._ctx.build_sniper_status(self._sniper_ref)

        # Early exit: nothing to analyze — skip LLM invocation
        pending = []
        if self._attention_router:
            pending = self._attention_router.pending_items()
        if not self._has_actionable_state(portfolio) and not pending:
            logger.info(
                f"[CAPTAIN:SKIP] cycle={cycle_num} mode=strategic "
                f"reason=no_events_no_positions (waiting for lifecycle discovery)"
            )
            await self._emit_event({
                "type": "captain_cycle_complete",
                "data": {"mode": "strategic", "cycle_num": cycle_num, "status": "skipped",
                         "reason": "no_events_no_positions"},
            })
            return

        health = self._fetch_health_snapshot()
        early_bird_opportunities = self._fetch_early_bird("strategic")
        decision_accuracy = self._fetch_decision_accuracy()
        mm_state = self._get_mm_state()

        body = self._ctx.build_strategic_context(
            portfolio, pending, sniper_status, self._build_task_section(),
            health=health, early_bird_opportunities=early_bird_opportunities,
            mm_state=mm_state, decision_accuracy=decision_accuracy,
        )
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
        cycle_num = self._cycle_preamble(mode="deep_scan")
        cycle_start = time.time()
        self._deep_scan_count += 1

        market_state = self._ctx.build_market_state()
        portfolio = await self._get_portfolio()
        sniper_status = self._ctx.build_sniper_status(self._sniper_ref)

        # Early exit: nothing to analyze — skip LLM invocation
        if not self._has_actionable_state(portfolio):
            logger.info(
                f"[CAPTAIN:SKIP] cycle={cycle_num} mode=deep_scan "
                f"reason=no_events_no_positions (waiting for lifecycle discovery)"
            )
            await self._emit_event({
                "type": "captain_cycle_complete",
                "data": {"mode": "deep_scan", "cycle_num": cycle_num, "status": "skipped",
                         "reason": "no_events_no_positions"},
            })
            return

        health = self._fetch_health_snapshot()
        early_bird_opportunities = self._fetch_early_bird("deep_scan")

        # Parallelize independent memory/DB fetches (each with per-op timeout)
        from .tools import _ctx as tool_ctx
        async def _fetch_trade_memories():
            if not (tool_ctx and tool_ctx.memory):
                return None
            recalled = await asyncio.wait_for(
                tool_ctx.memory.recall(
                    query="trade outcomes profit loss executed cancelled",
                    limit=3, memory_types=["trade_outcome"],
                ),
                timeout=10.0,
            )
            if recalled and recalled.results:
                return [r.content[:120] for r in recalled.results]
            return None

        async def _fetch_news_memories():
            if not (tool_ctx and tool_ctx.memory):
                return None
            top_tickers = [ev.event_ticker for ev in market_state.events[:3]]
            news_query = f"news price impact {' '.join(top_tickers)}"
            recalled = await asyncio.wait_for(
                tool_ctx.memory.recall(
                    query=news_query, limit=3, memory_types=["news"],
                ),
                timeout=10.0,
            )
            if recalled and recalled.results:
                return [r.content[:120] for r in recalled.results]
            return None

        async def _fetch_market_movers():
            from kalshiflow_rl.data.database import rl_db
            pool = await asyncio.wait_for(rl_db.get_pool(), timeout=3.0)
            async with asyncio.timeout(5.0):
                async with pool.acquire() as conn:
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
                        return [dict(r) for r in rows]
            return None

        async def _fetch_swing_patterns():
            """Fetch top swing-news patterns from the index."""
            from kalshiflow_rl.data.database import rl_db
            pool = await asyncio.wait_for(rl_db.get_pool(), timeout=3.0)
            async with asyncio.timeout(5.0):
                async with pool.acquire() as conn:
                    rows = await conn.fetch(
                        """SELECT news_title, news_url, direction, change_cents,
                                  causal_confidence, event_ticker, market_ticker
                           FROM swing_news_associations
                           WHERE causal_confidence >= 0.5
                             AND news_memory_id IS NOT NULL
                           ORDER BY change_cents DESC, causal_confidence DESC
                           LIMIT 5"""
                    )
                    if rows:
                        return [dict(r) for r in rows]
            return None

        # Emit phase event so frontend shows "Loading data..."
        await self._emit_event({
            "type": "agent_message",
            "subtype": "captain_phase",
            "phase": "data_fetch",
            "detail": "Loading memories and market data...",
        })

        # Outer safety net: 15s for all fetches combined
        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    _fetch_trade_memories(),
                    _fetch_news_memories(),
                    _fetch_market_movers(),
                    _fetch_swing_patterns(),
                    return_exceptions=True,
                ),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[CAPTAIN:DEEP_SCAN] Data fetch gather exceeded 15s, continuing with no enrichment")
            results = [None, None, None, None]

        fetch_names = ("trade_memories", "news_memories", "market_movers", "swing_patterns")
        resolved = []
        for name, result in zip(fetch_names, results):
            if isinstance(result, Exception):
                logger.warning(f"[CAPTAIN:DEEP_SCAN] {name} fetch failed: {result}")
                resolved.append(None)
            else:
                resolved.append(result)
        trade_memories, news_memories, market_movers, swing_patterns = resolved

        await self._emit_event({
            "type": "agent_message",
            "subtype": "captain_phase",
            "phase": "data_ready",
            "detail": "Data loaded, invoking agent...",
        })

        decision_accuracy = self._fetch_decision_accuracy()
        mm_state = self._get_mm_state()

        body = self._ctx.build_deep_scan_context(
            market_state, portfolio, sniper_status, health,
            trade_memories=trade_memories, news_memories=news_memories,
            task_section=self._build_task_section(), market_movers=market_movers,
            swing_patterns=swing_patterns,
            early_bird_opportunities=early_bird_opportunities,
            mm_state=mm_state, decision_accuracy=decision_accuracy,
        )
        prompt = f"Cycle #{cycle_num} DEEP_SCAN.\n{body}"

        logger.info(
            f"[CAPTAIN:CYCLE_START] cycle={cycle_num} mode=deep_scan "
            f"events={len(market_state.events)} positions={portfolio.total_positions}"
        )

        await self._invoke_agent("deep_scan", cycle_num, prompt)
        await self._cycle_postamble("deep_scan", cycle_num, cycle_start, portfolio)

    async def _run_agent(self, prompt: str, mode: str = "deep_scan") -> Optional[str]:
        """Run the agent, streaming categorized tool call events."""
        try:
            result_text = None
            token_buffer = []
            token_count = 0
            last_frontend_emit = time.time()

            limits = {"reactive": 25, "strategic": 75, "deep_scan": 100}
            config = {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": limits.get(mode, 50),
            }

            async for event in self._agent.astream_events(
                {"messages": [HumanMessage(content=prompt)]},
                version="v2",
                config=config,
            ):
                kind = event.get("event")

                # Heartbeat: if >5s since last frontend emit, send a pulse
                now = time.time()
                if now - last_frontend_emit > 5.0:
                    await self._emit_event({
                        "type": "agent_message",
                        "subtype": "captain_heartbeat",
                        "agent": "single_arb_captain",
                    })
                    last_frontend_emit = now

                # Stream thinking tokens
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        # Track token usage from chunk metadata
                        usage = getattr(chunk, "usage_metadata", None)
                        if usage:
                            self._cycle_input_tokens += getattr(usage, "input_tokens", 0) or 0
                            self._cycle_output_tokens += getattr(usage, "output_tokens", 0) or 0
                        if hasattr(chunk, "content") and chunk.content:
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
                            last_frontend_emit = time.time()

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
                        last_frontend_emit = time.time()

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
                    last_frontend_emit = time.time()

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
                    last_frontend_emit = time.time()

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

            self._consecutive_failures = 0
            return result_text

        except Exception as e:
            self._consecutive_failures += 1
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
            if pool is None:
                logger.warning("[CAPTAIN:TASKS] DB pool returned None — task ledger persistence skipped")
                return
            self._task_ledger.persist(pool)
        except Exception as e:
            logger.warning(f"[CAPTAIN:TASKS] Task ledger persistence failed: {e}")

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
            "consecutive_timeouts": self._consecutive_timeouts,
        }
