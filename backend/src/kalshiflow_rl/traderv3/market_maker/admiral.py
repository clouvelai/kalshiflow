"""Admiral - Attention-driven LLM agent for market making.

Mirrors Captain architecture but targets market making. Three invocation modes:
  REACTIVE: Fill notifications, inventory warnings, VPIN spikes → adjust spread/size
  STRATEGIC (5min): P&L review, queue positions, spread tuning
  DEEP_SCAN (30min): Full analysis, news search, market selection evaluation

Uses deepagents framework with Claude Sonnet. Admiral configures the QuoteEngine;
QuoteEngine handles deterministic execution.
"""

import asyncio
import logging
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

from .context_builder import MMContextBuilder
from .models import MMAttentionItem
from .tools import get_mm_tools

if TYPE_CHECKING:
    from .attention import MMAttentionRouter

logger = logging.getLogger("kalshiflow_rl.traderv3.market_maker.admiral")

ADMIRAL_PROMPT = """You are the Admiral — an autonomous Kalshi prediction market maker.

Your job: maintain continuous two-sided quotes that capture spread while managing inventory risk.
You configure a deterministic QuoteEngine. It places and manages quotes. You optimize parameters.

UNITS: Cents (0-100). YES@60c = 60c cost, 100c payout. Balance in DOLLARS. Tool prices in CENTS.

MARKET MAKING BASICS:
  - You earn the spread: buy at bid, sell at ask. Net = spread - maker_fees - adverse_selection.
  - Maker fee: 0.0175 * P * (1-P) per contract. Max 0.4375c at 50c. ~75% cheaper than taker.
  - Break-even spread ≈ 2 * maker_fee + adverse_selection. Typically 1-3c at mid-range prices.
  - Inventory risk: accumulating one-sided position loses money if price moves against you.
  - QuoteEngine automatically skews quotes to reduce inventory (skew_factor parameter).

YOUR TOOLS:
  Observation: get_mm_state, get_inventory, get_quote_performance, get_resting_orders
  Configuration: configure_quotes (spread, size, skew), set_market_override, pull_quotes, resume_quotes
  Intelligence: search_news (3 depth tiers, auto by cycle mode), get_market_movers, recall_memory, store_insight

SPREAD OPTIMIZATION:
  - Wider spreads = less fills but more profit per fill. Narrower = more fills but tighter margins.
  - Start conservative (4c spread), tighten as you learn the market.
  - Widen spread when: high VPIN (toxic flow), low liquidity, approaching position limits.
  - Tighten spread when: stable market, high volume, close to flat inventory.

INVENTORY MANAGEMENT:
  - Position skew is automatic (skew_factor). You tune the multiplier.
  - If position hits max_position limit, engine quotes one side only.
  - pull_quotes is your emergency brake - use for unexpected news or toxic flow.

MODE BEHAVIOR:
  REACTIVE (1-3 tool calls, <30s): Respond to fills and inventory warnings.
    - After fill: check if spread needs adjustment
    - Inventory warning: consider widening spread or pull_quotes
    - VPIN spike: consider pull_quotes or widen
  STRATEGIC (3-6 tool calls, <2min): Review P&L and optimize.
    - Check get_quote_performance for fill rates and adverse selection
    - Adjust spread/size based on performance
    - Consider market overrides for specific markets
  DEEP_SCAN (5-10 tool calls, <3min): Full review.
    - search_news for event context (auto-selects advanced depth with pattern enrichment)
    - get_market_movers to find which news actually moved prices historically
    - Review all market microstructure
    - Adjust strategy based on market conditions and news intelligence
    - recall_memory for past learnings on this event

NEWS INTELLIGENCE:
  - search_news has 3 depth tiers (auto-selected by cycle mode): ultra_fast (memory), fast (+Tavily), advanced (+full content).
  - Articles are enriched with similar_patterns: historical news that looked similar AND moved prices.
  - Use pattern predictions to adjust fair value estimates and spread decisions.
  - get_market_movers shows which past news actually moved prices on this event.
  - In DEEP_SCAN: search news for each event, check market_movers, update your fair value thesis.
  - Event context (what/who/when/domain) is injected into your briefing when available.

RULES:
  - Every cycle MUST include at least one tool call.
  - Prefer small parameter adjustments over large ones.
  - When uncertain, widen spreads rather than tighten.
  - store_insight: factual observations only (spread performance, fill patterns, market behavior).
  - NEVER store behavioral rules or self-restrictions.
"""


class Admiral:
    """LLM market making agent.

    Configures QuoteEngine parameters based on market conditions.
    Three invocation modes driven by MMAttentionRouter and timers.
    """

    def __init__(
        self,
        context_builder: MMContextBuilder,
        attention_router: Optional["MMAttentionRouter"] = None,
        config=None,
        model_name: Optional[str] = None,
        event_callback: Optional[Callable[..., Coroutine]] = None,
        system_ready: Optional[asyncio.Event] = None,
    ):
        from ..single_arb.mentions_models import get_captain_model
        model_name = model_name or get_captain_model()

        self._model_name = model_name
        self._event_callback = event_callback
        self._ctx = context_builder
        self._attention_router = attention_router
        self._config = config
        self._system_ready = system_ready

        # Intervals from config
        self._strategic_interval = getattr(config, "mm_strategic_interval", 300.0) if config else 300.0
        self._deep_scan_interval = getattr(config, "mm_deep_scan_interval", 1800.0) if config else 1800.0

        self._cache = InMemoryCache()

        # Build agent
        resolved_model = init_chat_model(model_name)
        summarization_defaults = _compute_summarization_defaults(resolved_model)
        backend_factory = lambda rt: StateBackend(rt)

        middleware = [
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

        self._agent = create_agent(
            resolved_model,
            tools=get_mm_tools(),
            system_prompt=ADMIRAL_PROMPT,
            middleware=middleware,
            cache=self._cache,
        )

        self._running = False
        self._paused = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_cycle_at: Optional[float] = None
        self._errors: List[str] = []

        self._reactive_count = 0
        self._strategic_count = 0
        self._deep_scan_count = 0

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[ADMIRAL:START] model={self._model_name}")

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[ADMIRAL:STOP] cycles={self._cycle_count}")

    def pause(self) -> None:
        self._paused = True
        logger.info("[ADMIRAL:PAUSE]")

    def resume(self) -> None:
        self._paused = False
        logger.info("[ADMIRAL:RESUME]")

    @property
    def is_paused(self) -> bool:
        return self._paused

    def _record_error(self, error: str) -> None:
        self._errors.append(error)
        if len(self._errors) > 50:
            self._errors = self._errors[-50:]

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "paused": self._paused,
            "model": self._model_name,
            "cycle_count": self._cycle_count,
            "reactive_count": self._reactive_count,
            "strategic_count": self._strategic_count,
            "deep_scan_count": self._deep_scan_count,
            "last_cycle_at": self._last_cycle_at,
            "recent_errors": self._errors[-5:],
        }

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    async def _run_loop(self) -> None:
        """Event-driven main loop with three invocation modes."""
        # Wait for system ready
        if self._system_ready:
            logger.info("[ADMIRAL] Waiting for system initialization...")
            try:
                await asyncio.wait_for(self._system_ready.wait(), timeout=120.0)
                logger.info("[ADMIRAL] System ready, starting cycles")
            except asyncio.TimeoutError:
                logger.warning("[ADMIRAL] System init timeout, starting anyway")

        last_strategic = time.time()
        last_deep_scan = time.time()

        if not self._attention_router:
            logger.info("[ADMIRAL] No AttentionRouter — fixed-interval mode")
            await self._run_fixed_interval()
            return

        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(5.0)
                    continue

                now = time.time()

                # Deep scan check
                if now - last_deep_scan >= self._deep_scan_interval:
                    await self._run_cycle("deep_scan")
                    last_deep_scan = now
                    last_strategic = now  # Reset strategic timer too
                    continue

                # Strategic check
                if now - last_strategic >= self._strategic_interval:
                    await self._run_cycle("strategic")
                    last_strategic = now
                    continue

                # Reactive: wait for attention signal or timeout
                try:
                    await asyncio.wait_for(
                        self._attention_router.notify_event.wait(),
                        timeout=min(
                            self._strategic_interval - (now - last_strategic),
                            self._deep_scan_interval - (now - last_deep_scan),
                            30.0,
                        ),
                    )
                    # Attention signal received
                    items = self._attention_router.drain()
                    if items:
                        await self._run_cycle("reactive", items=items)
                except asyncio.TimeoutError:
                    pass  # Timer will handle strategic/deep_scan

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._record_error(str(e))
                logger.error(f"[ADMIRAL] Loop error: {e}", exc_info=True)
                await asyncio.sleep(10.0)

    async def _run_fixed_interval(self) -> None:
        """Fallback fixed-interval loop (no attention router)."""
        interval = 60.0
        cycle = 0
        while self._running:
            try:
                if self._paused:
                    await asyncio.sleep(5.0)
                    continue

                cycle += 1
                mode = "deep_scan" if cycle % 30 == 0 else ("strategic" if cycle % 5 == 0 else "reactive")
                await self._run_cycle(mode)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._record_error(str(e))
                logger.error(f"[ADMIRAL] Fixed loop error: {e}", exc_info=True)
                await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Cycle Execution
    # ------------------------------------------------------------------

    async def _run_cycle(
        self,
        mode: str,
        items: Optional[List[MMAttentionItem]] = None,
    ) -> None:
        """Execute a single Admiral cycle."""
        self._cycle_count += 1
        self._last_cycle_at = time.time()
        cycle_start = time.time()

        # Build context
        balance = 0  # TODO: get from gateway on strategic/deep_scan
        if mode == "reactive":
            self._reactive_count += 1
            context = self._ctx.build_reactive(self._cycle_count, items or [], balance)
        elif mode == "strategic":
            self._strategic_count += 1
            context = self._ctx.build_strategic(self._cycle_count, balance)
        else:  # deep_scan
            self._deep_scan_count += 1
            context = self._ctx.build_deep_scan(self._cycle_count, balance)

        logger.info(f"[ADMIRAL:CYCLE] mode={mode} #{self._cycle_count} ({len(context)} chars)")

        # Run agent with fresh thread_id
        thread_id = str(uuid.uuid4())
        try:
            result = await self._agent.ainvoke(
                {"messages": [HumanMessage(content=context)]},
                config={"configurable": {"thread_id": thread_id}},
            )

            elapsed = time.time() - cycle_start
            messages = result.get("messages", [])
            tool_calls = sum(
                1 for m in messages
                if hasattr(m, "tool_calls") and m.tool_calls
            )
            logger.info(
                f"[ADMIRAL:CYCLE] mode={mode} #{self._cycle_count} "
                f"completed in {elapsed:.1f}s ({tool_calls} tool calls, "
                f"{len(messages)} messages)"
            )

            # Broadcast cycle result
            if self._event_callback:
                try:
                    await self._event_callback("agent_message", {
                        "cycle": self._cycle_count,
                        "mode": mode,
                        "tool_calls": tool_calls,
                        "elapsed": round(elapsed, 1),
                        "agent": "admiral",
                    })
                except Exception:
                    pass

        except Exception as e:
            elapsed = time.time() - cycle_start
            error_msg = f"{mode} cycle #{self._cycle_count} failed after {elapsed:.1f}s: {e}"
            self._record_error(error_msg)
            logger.error(f"[ADMIRAL:CYCLE] {error_msg}")
            logger.debug(traceback.format_exc())
