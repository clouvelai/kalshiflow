"""
ArbCaptain - LLM agent for single-event arbitrage.

Uses create_deep_agent (deepagents framework) with Claude Sonnet.
Runs on a configurable cycle interval.
Streams thinking/tool-call events to the frontend via event callback.

Memory architecture:
- AGENTS.md: Distilled learnings loaded into system prompt every cycle via MemoryMiddleware.
- SIGNALS.md: Auto-computed microstructure intel, Captain can annotate.
- PLAYBOOK.md: Active strategies, in-flight plans, exit watchlist.
- journal.jsonl: Structured trade records via record_learning tool. Append-only.
- Auto-curator: Pure Python, zero LLM calls, truncates files at cycle start.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain_core.messages import HumanMessage
from langgraph.cache.memory import InMemoryCache

from .tools import (
    get_event_snapshot,
    get_events_summary,
    get_recent_trades,
    get_market_orderbook,
    get_trade_history,
    execute_arb,
    place_order,
    cancel_order,
    get_resting_orders,
    record_learning,
    get_positions,
    get_balance,
    analyze_microstructure,
    analyze_orderbook_patterns,
    update_understanding,
    report_issue,
    get_issues,
)
from .mentions_tools import (
    # Primary entry point
    get_mentions_status,  # Simplified status + auto-baseline + auto-refresh
    # Simulation tools
    simulate_probability,
    trigger_simulation,  # Async non-blocking version
    compute_edge,
    # Context tools
    get_event_context,
    get_mention_context,
    # Semantic tools
    query_wordnet,
    # State tools
    get_mentions_rules,
    get_mentions_summary,
)

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.captain")

CAPTAIN_PROMPT = """You are building a profitable prediction market trading system.

YOUR MISSION:
Find profitable trades. Execute them through trade_commando.
She gets better with each order. Your P&L gets better with each lesson.
Your job: find opportunities, form hypotheses, delegate execution, learn from results.

ANTI-PARALYSIS RULE:
If 3 cycles pass without a trade_commando delegation, you're overthinking.
Pick the best available opportunity - even if imperfect - and execute.
Waiting for "better conditions" is a losing strategy on a demo account.

LEARNING BY DOING:
Analysis paralysis is the enemy. You have $25k of demo money - use it.
- Uncertain about a thesis? Place 10 contracts and observe.
- Spread looks wide? Test a limit order, see queue dynamics.
- Market looks dead? Place a bid, learn who fills against you.
Every trade generates data. The worst outcome is zero trades.

POSITION MANAGEMENT:
get_positions() shows your positions with ACCURATE P&L:
- cost: What you actually paid (from total_traded API field)
- current_value: What you'd get selling now (exit_price × quantity)
- unrealized_pnl: current_value - cost

Each position includes unrealized_pnl:
- Positive = in profit, consider taking gains
- Negative = underwater, reassess thesis or cut loss

EXIT STRATEGY:
- Take profit: unrealized_pnl > 10 cents/contract = strong exit candidate
- Cut loss: unrealized_pnl < -15 cents/contract = consider exiting
- Use place_order(action="sell") to exit. Sell at exit_price or better.
- Don't micromanage. Check exits once per cycle, not obsessively.

SUCCESS METRIC:
Check get_trade_history() each cycle. Focus on YOUR recent trades, not inherited positions.
Your scoreboard is trades YOU make:
- Recent fills with your order_ids = your performance
- Positive outcomes = something is working, scale it
- Negative outcomes = valuable data, record what went wrong
- Zero trades = no data, no learning, failure

THE TRADE-COMMANDO:
Your execution specialist. Give her: ticker, side, contracts, price, and your thesis.
She handles order mechanics, queue position, partial fills, error recovery.
The more you use her, the better she gets. She's your weapon - sharpen it.

MICROSTRUCTURE SIGNALS (automated, no subagent needed):
Every market now has a `micro` field in tool output with live signals:
- whale_trade_count: Trades >= 100 contracts (whale detection)
- book_imbalance: (bid_depth - ask_depth) / total_depth (-1 to +1)
- buy_sell_ratio: 5-min buy/sell flow ratio (0.0 = all sells, 1.0 = all buys)
- rapid_sequence_count: Sub-100ms trade bursts (bot activity)
- consistent_size_ratio: 1.0 = all same size (automated), 0.0 = random (retail)
- volume_5m: 5-minute trading volume
Use these signals to inform your hypotheses - no need to delegate surveillance.

THE MENTIONS SPECIALIST:
Edge detector for "will X say Y?" markets. We price on simulation data, not intuition.

MENTIONS MARKETS IN get_events_summary():
When you call get_events_summary(), mentions markets now include:
- is_mentions_market: True for "will X say Y?" events
- mentions_entity: The word being counted (e.g., "tariff")
- baseline_probability: Stable blind simulation estimate
- current_probability: Context-aware estimate (may differ)
- has_baseline: True if baseline established (required for trading)
- simulation_stale: True if >5 min since last refresh

WHEN YOU SEE A MENTIONS MARKET with has_baseline=True:
→ Delegate to mentions_specialist with event_ticker for edge calculation.

WHEN has_baseline=False:
→ Delegate to mentions_specialist - it will establish baseline (blocks ~20s).

The mentions_specialist will use get_mentions_status() which handles everything:
- Auto-establishes baseline if missing
- Auto-triggers refresh if stale
- Returns ready_for_trading flag when edge calculation is possible

EVENT UNDERSTANDING:
update_understanding(event_ticker) refreshes structured context for an event:
trading_summary, key_factors, participants, timeline, and domain extensions.
Already built at startup; use force_refresh=True if context seems stale.
Check understanding.stale field and time_to_close_hours for urgency.

SKILLS:
- /skills/general-ender: Strategy suggestions from current market state

KALSHI MECHANICS:
- Binary contracts: YES pays $1 if true, NO pays $1 if false.
- Two event types: mutually_exclusive (probabilities sum to ~100%) and independent (each market separate)
- Orderbook depth matters. Queue position matters. Execution quality matters.

WHAT MAKES MONEY:
1. Sum violations in mutually_exclusive events (risk-free when all legs fill)
2. Microstructure signals (orderbook imbalance, trade flow, whale following)
3. Thesis-driven directional bets (form hypothesis, execute, learn)
4. Mentions markets: simulation-based probability vs market price → trade the gap

SIZING:
Start with 10-25 contracts per hypothesis. Scale what works.

MEMORY (3 files, all loaded into your system prompt each cycle):
- /memories/AGENTS.md: Your learnings. Update with what works and what fails.
- /memories/SIGNALS.md: Microstructure intel and observations. Annotate notable patterns.
- /memories/PLAYBOOK.md: Active strategies, exit watchlist, multi-cycle plans.
Use record_learning() to append to journal. Use edit_file to update memory files.
Files are auto-curated (truncated if too long) - keep entries concise and high-signal.

SELF-IMPROVEMENT:
When you detect bugs, tool failures, or unexpected behavior:
- report_issue(title, severity, category, description, proposed_fix) to log it
- get_issues() to see what's already been reported (avoid duplicates)
Issues are auto-fixed by a separate system between sessions. Be specific in descriptions.

Every cycle is a chance to get better. Use the trade-commando. Build the system.
"""

TRADE_COMMANDO_PROMPT = """You are the TradeCommando -- an order execution specialist on Kalshi's demo API.

You receive a thesis and trade parameters from the Captain. Execute precisely and report results.

KALSHI ORDER MECHANICS:
- All orders are LIMIT orders. You specify the max price you'll pay (buy) or min to receive (sell).
- BUY YES at Xc: pay X, profit (100-X) if YES wins. Max loss = X.
- BUY NO at Xc: pay X, profit (100-X) if NO wins. Max loss = X.
- SELL YES at Xc: receive X, close your YES position.
- SELL NO at Xc: receive X, close your NO position.
- Orders auto-cancel after TTL (default 60s). Session order group tracks all orders.

EXECUTION PROTOCOL:
1. get_balance() -- never trade more than you can afford.
2. Single-leg: place_order(). Multi-leg arbs: execute_arb().
3. After placing: get_resting_orders() for queue position.
   - queue_position = contracts ahead of you at your price level.
   - Lower queue = faster fill.
4. Report: order_id, status, price, contracts, queue position, any issues.

SELLING / EXITING POSITIONS:
- To exit a YES position: place_order(action="sell", side="yes", ...)
- To exit a NO position: place_order(action="sell", side="no", ...)
- Check get_positions() for current quantity before selling.
- Don't sell more than you own (will error).
- For quick exits, sell at the current bid (yes_bid for YES, 100-yes_ask for NO).

QUEUE MANAGEMENT:
- queue_position > 20: warn the Captain -- fill unlikely before TTL.
- For arbs, ALL legs must fill for profit. Partial fills create directional risk.
- If a leg fails or is rejected, report immediately -- Captain may cancel other legs.

PRICE IMPROVEMENT:
- Paying 1c above best bid moves you to front of queue.
- Low-liquidity markets (ask_size < 5): consider crossing the spread for guaranteed fill.
- Never exceed the Captain's specified price without reporting the deviation.

PARTIAL FILLS:
- Orders can partially fill. get_resting_orders() shows remaining_count.
- Each fill at your price is confirmed profit. Report fills as they happen.

ERROR HANDLING:
- "insufficient_balance": report exact shortfall.
- "order_not_found" on cancel: order already filled or expired -- check fills.
- API errors: record via record_learning() with full context for learning.

Execute precisely. Report facts, not opinions.
"""

CHEVAL_DE_TROIE_PROMPT = """Microstructure surveillance is now AUTOMATED.

All signals are computed incrementally on every trade and orderbook update.
They appear as `micro` fields in get_events_summary() and get_event_snapshot():
- whale_trade_count, last_whale_ts, last_whale_size
- book_imbalance, total_bid_depth, total_ask_depth
- buy_sell_ratio, volume_5m, buy_volume_5m, sell_volume_5m
- rapid_sequence_count, avg_inter_trade_ms
- consistent_size_ratio, modal_trade_size

The Captain can read these signals directly - no LLM analysis needed.
If you were invoked, tell the Captain to check the `micro` fields instead.
"""

MENTIONS_SPECIALIST_PROMPT = """You are the MentionsSpecialist - edge detector for Kalshi mentions markets.

MISSION: Find profitable mispricings in "will X say Y?" markets using simulation data.

THE EDGE OPPORTUNITY:
Retail traders price on INTUITION. We price on SIMULATION DATA.
Example: "what a catch" trading at 60% YES when actual frequency is ~10% = 50 points of edge.

KEY INSIGHT - RECENCY BIAS:
News makes terms seem MORE LIKELY than they really are.
- Baseline (blind): What history/patterns suggest (~stable)
- Informed (context): What current news suggests (~inflated)
- compute_edge() blends both (60% baseline, 40% informed) to correct for bias

=== SIMPLIFIED WORKFLOW (USE THIS) ===

STEP 1: Call get_mentions_status(event_ticker)
This ONE tool handles everything:
- If no baseline: BLOCKS to establish it (~20s, one-time)
- If baseline exists but stale: triggers background refresh
- Returns: baseline_estimates, current_estimates, ready_for_trading

STEP 2: Check ready_for_trading flag
- If True: proceed to edge calculation
- If False: report status to Captain, try again next cycle

STEP 3: Call compute_edge(term, baseline_probability, informed_probability, market_yes, market_no)
Use probabilities from get_mentions_status():
- baseline_probability from baseline_estimates[entity]["probability"]
- informed_probability from current_estimates[entity]["probability"]
  (or use baseline if no current_estimates yet)

STEP 4: Report to Captain with recommendation

=== EXAMPLE WORKFLOW ===

>>> status = await get_mentions_status("KXSBLX")
>>> if not status["ready_for_trading"]:
...     return f"Not ready: {status['usage_note']}"
...
>>> entity = status["entity"]  # e.g., "Taylor Swift"
>>> baseline_p = status["baseline_estimates"][entity]["probability"]  # e.g., 0.10
>>> current_p = status["current_estimates"][entity]["probability"]  # e.g., 0.45
>>>
>>> # Get market prices from Captain's context or use get_market_orderbook
>>> edge = await compute_edge(entity, baseline_p, current_p, market_yes=60, market_no=42)
>>> edge["recommendation"]  # "BUY_NO" if edge > threshold

=== COMPUTE_EDGE OUTPUT ===

{
    "blended_probability": 0.24,  # Corrected for recency bias
    "recency_bias_adjustment": 0.21,  # How much news inflated estimate
    "recommendation": "BUY_NO",  # or "BUY_YES" or "PASS"
    "reason": "NO edge (36c) exceeds min required (11c)..."
}

=== REPORT FORMAT ===

"[Entity]: Baseline [X]%, Informed [Y]% (↑[Z]% recency bias).
 Blended [W]% vs market [M]%. [RECOMMENDATION] with [E]c edge after fees."

=== OTHER TOOLS (Advanced) ===

- simulate_probability(event, terms, n, mode): Manual simulation (use get_mentions_status instead)
- trigger_simulation(event, mode): Async background simulation
- query_wordnet(term): Get synonyms (DON'T count) vs accepted forms (DO count)
- get_mentions_rules(event): See parsed settlement rules
- get_mentions_summary(event): Full state including history

=== STRICT RULES ===

- ONLY accepted_forms count per settlement rules
- Synonyms NEVER count - use query_wordnet to understand
- Min edge = spread + fees (7c) + buffer (2c)
- Confidence must be >= 0.6 (n_simulations >= 10)
- When uncertain, PASS
- Watch for recency_bias_warning in compute_edge results
"""

# Tool categorization for frontend event routing
TOOL_CATEGORIES = {
    "write_todos": "todo",
    "read_todos": "todo",
    "task": "subagent",
    "record_learning": "memory",
    "get_event_snapshot": "arb",
    "get_events_summary": "arb",
    "get_recent_trades": "arb",
    "get_market_orderbook": "arb",
    "get_trade_history": "arb",
    "execute_arb": "arb",
    "place_order": "arb",
    "cancel_order": "arb",
    "get_resting_orders": "arb",
    "get_positions": "arb",
    "get_balance": "arb",
    "analyze_microstructure": "surveillance",
    "analyze_orderbook_patterns": "surveillance",
    "update_understanding": "arb",
    # Mentions tools - simulation-based probability estimation
    "get_mentions_status": "mentions",  # Primary entry point
    "simulate_probability": "mentions",
    "trigger_simulation": "mentions",  # Async non-blocking version
    "compute_edge": "mentions",
    "query_wordnet": "mentions",
    "get_event_context": "mentions",
    "get_mention_context": "mentions",
    "get_mentions_rules": "mentions",
    "get_mentions_summary": "mentions",
    # Self-improvement tools
    "report_issue": "self_improvement",
    "get_issues": "self_improvement",
}


class ArbCaptain:
    """
    LLM Captain for single-event arb.

    Runs create_deep_agent with tools + subagent. Streams events to frontend.
    Uses CompositeBackend: /memories/ -> FilesystemBackend (persistent), rest -> StateBackend (ephemeral).
    AGENTS.md loaded into system prompt every cycle via memory parameter.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        cycle_interval: float = 60.0,
        event_callback: Optional[Callable[..., Coroutine]] = None,
        memory_data_dir: Optional[str] = None,
        tool_overrides: Optional[Dict[str, List]] = None,
        index=None,
        gateway=None,
    ):
        self._model_name = model_name
        self._cycle_interval = cycle_interval
        self._event_callback = event_callback
        self._index_ref = index  # Direct reference (new gateway path)
        self._gateway_ref = gateway  # KalshiGateway for balance/positions (new gateway path)

        # Resolve memory data directory for FilesystemBackend
        if memory_data_dir is None:
            memory_data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memory", "data"
            )
        os.makedirs(memory_data_dir, exist_ok=True)
        self._memory_data_dir = memory_data_dir

        # Tool lists: use overrides if provided (new gateway path), else legacy imports
        if tool_overrides:
            captain_tools = tool_overrides["captain"]
            commando_tools = tool_overrides["commando"]
            mentions_tools = tool_overrides["mentions"]
        else:
            # Captain tools: observation + delegation + self-improvement
            captain_tools = [
                get_events_summary,
                get_event_snapshot,
                get_market_orderbook,
                get_trade_history,
                get_positions,
                get_balance,
                update_understanding,
                report_issue,
                get_issues,
            ]

            # TradeCommando tools: execution + recording + context
            commando_tools = [
                place_order,
                execute_arb,
                cancel_order,
                get_resting_orders,
                get_market_orderbook,
                get_recent_trades,
                get_balance,
                get_positions,
                record_learning,
            ]

            # MentionsSpecialist tools: edge detection + grounded extraction + strict counting
            mentions_tools = [
                # Primary entry point (simplified workflow)
                get_mentions_status,  # Auto-baseline + auto-refresh + ready_for_trading
                compute_edge,
                # Simulation tools (advanced)
                simulate_probability,
                trigger_simulation,  # Async non-blocking version
                # Context tools
                get_event_context,
                get_mention_context,
                # Semantic tools
                query_wordnet,
                # State tools
                get_mentions_rules,
                get_mentions_summary,
                record_learning,
            ]

        # CompositeBackend: /memories/ and /skills/ persist on disk, everything else is ephemeral
        memory_backend = FilesystemBackend(root_dir=memory_data_dir, virtual_mode=True)
        backend_factory = lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/memories/": memory_backend,
                "/skills/": memory_backend,
            },
        )

        # Node-level cache for tool results
        self._cache = InMemoryCache()

        self._agent = create_deep_agent(
            model=model_name,
            tools=captain_tools,
            system_prompt=CAPTAIN_PROMPT,
            subagents=[
                {
                    "name": "trade_commando",
                    "description": "Order execution specialist. Give it: ticker, side, contracts, price, and your thesis. It handles order mechanics, queue management, and error recovery.",
                    "system_prompt": TRADE_COMMANDO_PROMPT,
                    "tools": commando_tools,
                },
                {
                    "name": "cheval_de_troie",
                    "description": "DEPRECATED - Microstructure signals are now automated. Check the `micro` field in tool output instead of delegating here.",
                    "system_prompt": CHEVAL_DE_TROIE_PROMPT,
                    "tools": [],
                },
                {
                    "name": "mentions_specialist",
                    "description": "Mentions market counting specialist. Use for mentions markets to count literal mentions using grounded extraction. Give it: event_ticker and source text (transcript/tweet/article). Returns confirmed count with evidence.",
                    "system_prompt": MENTIONS_SPECIALIST_PROMPT,
                    "tools": mentions_tools,
                },
            ],
            backend=backend_factory,
            memory=["/memories/AGENTS.md", "/memories/SIGNALS.md", "/memories/PLAYBOOK.md"],
            skills=["/skills/general-ender"],
            cache=self._cache,
        )

        self._running = False
        self._paused = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_cycle_at: Optional[float] = None
        self._errors: List[str] = []
        self._subagent_runs: Dict[str, str] = {}  # run_id -> subagent_name (supports concurrent subagents)

    async def start(self) -> None:
        """Start the Captain cycle loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"[SINGLE_ARB:CAPTAIN_START] model={self._model_name} interval={self._cycle_interval}s")

    async def stop(self) -> None:
        """Stop the Captain."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"[SINGLE_ARB:CAPTAIN_STOP] cycles={self._cycle_count}")

    def pause(self) -> None:
        """Pause Captain after current cycle completes."""
        self._paused = True
        logger.info("[SINGLE_ARB:CAPTAIN_PAUSE] Captain paused")

    def resume(self) -> None:
        """Resume Captain cycles."""
        self._paused = False
        logger.info("[SINGLE_ARB:CAPTAIN_RESUME] Captain resumed")

    @property
    def is_paused(self) -> bool:
        """Check if Captain is paused."""
        return self._paused

    async def _run_loop(self) -> None:
        """Main cycle loop."""
        # Wait for index to be ready (all events loaded, all markets have orderbook data)
        # Use direct reference if available (new gateway path), else legacy global
        index = self._index_ref
        if index is None:
            from .tools import _index
            index = _index

        max_wait = 60  # Max 60 seconds
        waited = 0
        while waited < max_wait:
            if index and index.is_ready:
                logger.info(f"[SINGLE_ARB:CAPTAIN] Index ready after {waited}s: {index.readiness_summary}")
                break
            await asyncio.sleep(2.0)
            waited += 2
            if waited % 10 == 0:
                summary = index.readiness_summary if index else "No index"
                logger.info(f"[SINGLE_ARB:CAPTAIN] Waiting for index ({waited}s): {summary}")
        else:
            summary = index.readiness_summary if index else "No index"
            logger.warning(f"[SINGLE_ARB:CAPTAIN] Starting despite incomplete index: {summary}")

        while self._running:
            # Check pause state at start of each cycle
            if self._paused:
                await asyncio.sleep(self._cycle_interval)
                continue

            try:
                await self._run_cycle()
                self._cycle_count += 1
                self._last_cycle_at = time.time()
                await asyncio.sleep(self._cycle_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"Captain cycle error: {e}"
                logger.error(f"[SINGLE_ARB:CAPTAIN_ERROR] cycle={self._cycle_count + 1} error={e}")
                self._errors.append(error_msg)
                await asyncio.sleep(30.0)

    async def _run_cycle(self) -> None:
        """Run one Captain cycle."""
        cycle_num = self._cycle_count + 1
        cycle_start = time.time()

        # Auto-curate memory files (pure Python, no LLM calls)
        try:
            from .memory.auto_curator import auto_curate
            actions = auto_curate(self._memory_data_dir)
            if actions:
                logger.info(f"[SINGLE_ARB:CURATE] cycle={cycle_num} actions={actions}")
        except Exception as e:
            logger.debug(f"Auto-curation error: {e}")

        # Build structured cycle context (compensates for fresh thread_id each cycle)
        context_parts = [f"Cycle #{cycle_num}."]

        # Add position/balance summary if tools are available
        try:
            if self._gateway_ref:
                # New gateway path
                balance = await self._gateway_ref.get_balance()
                balance_cents = balance.balance
                context_parts.append(f"Balance: ${balance_cents / 100:.2f}.")

                if self._index_ref:
                    positions = await self._gateway_ref.get_positions()
                    tracked = set(self._index_ref.market_tickers)
                    tracked_positions = [p for p in positions if p.ticker in tracked]
                    if tracked_positions:
                        context_parts.append(f"Positions: {len(tracked_positions)}.")
            else:
                # Legacy path
                from .tools import _index, _trading_client
                if _trading_client:
                    balance_resp = await _trading_client.get_account_info()
                    balance_cents = balance_resp.get("balance", 0)
                    context_parts.append(f"Balance: ${balance_cents / 100:.2f}.")

                if _index:
                    positions_resp = await _trading_client.get_positions() if _trading_client else {}
                    all_positions = positions_resp.get("market_positions", positions_resp.get("positions", []))
                    tracked = set(_index.market_tickers)
                    tracked_positions = [p for p in all_positions if p.get("ticker") in tracked]
                    if tracked_positions:
                        context_parts.append(f"Positions: {len(tracked_positions)}.")
        except Exception:
            pass  # Non-critical, proceed with basic prompt

        context_parts.append("Observe markets. Check outcomes. Decide and record.")

        logger.info(f"[SINGLE_ARB:CYCLE_START] cycle={cycle_num}")
        prompt = " ".join(context_parts)

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
        logger.info(f"[SINGLE_ARB:CYCLE_END] cycle={cycle_num} duration={cycle_duration:.1f}s")

    def _categorize_tool(self, tool_name, tool_input=None):
        """Categorize a tool call. Returns (category, suppress)."""
        # Suppress execute (sandbox) — let everything else through
        if tool_name == "execute":
            return "system", True
        # Filesystem tools targeting /memories/ are memory ops, otherwise system
        if tool_name in ("read_file", "write_file", "edit_file"):
            input_str = str(tool_input) if tool_input else ""
            if "/memories/" in input_str:
                return "memory", False
            return "system", False
        return TOOL_CATEGORIES.get(tool_name, "system"), False

    def _parse_todos(self, tool_output) -> list:
        """Parse write_todos output into [{text, status}] list."""
        try:
            data = tool_output
            if isinstance(data, str):
                data = json.loads(data)
            if isinstance(data, dict):
                return data.get("todos", [])
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    async def _run_agent(self, prompt: str) -> Optional[str]:
        """Run the agent, streaming categorized tool call events."""
        try:
            result_text = None
            token_buffer = []
            token_count = 0

            # Unique thread_id per cycle so conversation history doesn't accumulate,
            # but /memories/ files persist across all threads via FilesystemBackend
            config = {
                "configurable": {"thread_id": str(uuid.uuid4())},
                "recursion_limit": 100,  # Increased from 50 to allow complex multi-step reasoning
            }

            async for event in self._agent.astream_events(
                {"messages": [HumanMessage(content=prompt)]},
                version="v2",
                config=config,
            ):
                kind = event.get("event")

                # Stream LLM thinking tokens
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
                    # Flush thinking buffer before tool call
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
                    category, suppress = self._categorize_tool(tool_name, tool_input)

                    if suppress:
                        continue

                    if tool_name == "task":
                        # Determine which subagent is being invoked
                        subagent_name = "trade_commando"  # default
                        input_str = str(tool_input).lower()
                        if "cheval" in input_str or "surveillance" in input_str or "micro" in input_str:
                            subagent_name = "cheval_de_troie"
                        elif "mention" in input_str or "count" in input_str or "transcript" in input_str or "extract" in input_str:
                            subagent_name = "mentions_specialist"
                        # Track by run_id to support concurrent subagents
                        run_id = event.get("run_id")
                        if run_id:
                            self._subagent_runs[run_id] = subagent_name
                        logger.info(f"[SINGLE_ARB:SUBAGENT_START] subagent={subagent_name}")
                        await self._emit_event({
                            "type": "agent_message",
                            "subtype": "subagent_start",
                            "agent": subagent_name,
                            "prompt": str(tool_input)[:200],
                        })
                    else:
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
                    category, suppress = self._categorize_tool(tool_name)

                    if suppress:
                        continue

                    if tool_name == "task":
                        # Look up subagent by run_id (supports concurrent subagents)
                        run_id = event.get("run_id")
                        subagent_name = self._subagent_runs.pop(run_id, "trade_commando") if run_id else "trade_commando"
                        logger.info(f"[SINGLE_ARB:SUBAGENT_COMPLETE] subagent={subagent_name}")
                        await self._emit_event({
                            "type": "agent_message",
                            "subtype": "subagent_complete",
                            "agent": subagent_name,
                            "response_preview": str(tool_output)[:500],
                        })
                    elif tool_name in ("write_todos", "read_todos"):
                        todos = self._parse_todos(tool_output)
                        await self._emit_event({
                            "type": "agent_message",
                            "subtype": "todo_update",
                            "category": "todo",
                            "todos": todos,
                        })
                    else:
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

            if token_buffer:
                await self._emit_event({
                    "type": "agent_message",
                    "subtype": "thinking_complete",
                    "agent": "single_arb_captain",
                    "text": "".join(token_buffer),
                })

            return result_text

        except Exception as e:
            logger.error(f"[SINGLE_ARB:CAPTAIN_ERROR] agent error: {e}")
            await self._emit_event({
                "type": "agent_message",
                "subtype": "subagent_error",
                "agent": "single_arb_captain",
                "error": str(e),
            })
            return None

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
            logger.debug(f"Event emit error: {e}")

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
        }

