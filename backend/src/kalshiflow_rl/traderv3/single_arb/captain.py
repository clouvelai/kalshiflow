"""
ArbCaptain - LLM agent for single-event arbitrage.

Uses create_deep_agent (deepagents framework) with Claude Sonnet.
Runs on a configurable cycle interval.
Streams thinking/tool-call events to the frontend via event callback.

Memory architecture:
- AGENTS.md: Distilled learnings loaded into system prompt every cycle via MemoryMiddleware.
  Agent updates via edit_file. Persists on disk via FilesystemBackend.
- journal.jsonl: Structured trade records via memory_store tool. Append-only.
- pgvector: Semantic search over journal entries (fire-and-forget async).
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
    memory_store,
    get_positions,
    get_balance,
    analyze_microstructure,
    analyze_orderbook_patterns,
)
from .mentions_tools import (
    # Primary simulation tools
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
    record_evidence,  # Deprecated but kept for compatibility
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
get_positions() shows YOUR positions vs LEGACY positions separately:
- captain_positions: Trades YOU made this session (your responsibility)
- legacy_positions: Inherited positions (ignore unless relevant to your thesis)

Each position includes unrealized_pnl (current_value - total_cost):
- Positive = in profit, consider taking gains
- Negative = underwater, reassess thesis or cut loss

EXIT STRATEGY (for captain_positions only):
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

THE CHEVAL DE TROIE:
Your surveillance specialist. Analyzes microstructure to identify bots, whales, and anomalies.
Her intelligence informs your hypotheses.

THE MENTIONS SPECIALIST:
Edge detector for "will X say Y?" markets. We price on simulation data, not intuition.

HOW IT WORKS:
- Generates blind transcripts (LLM roleplay WITHOUT knowing the terms)
- Counts term appearances POST-HOC across 10+ simulations
- P(mention) = appearances / simulations

TWO MODES:
- Baseline (blind): Historical patterns only - stable reference point
- Informed (context): Includes current news/context - may differ from baseline
- compute_edge() blends both (configurable weighting) and compares to market

WHEN TO USE:
- Markets with "mention" or "say" in settlement rules
- Independent events only (each market settles separately)

ASYNC WORKFLOW (don't block yourself):
1. trigger_simulation(event, mode="blind") → returns immediately
2. Next cycle: check get_mentions_summary() for results
3. trigger_simulation(event, mode="informed") → context-aware refresh
4. compute_edge(term, baseline, informed, market_yes, market_no) → trade signal

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

MEMORY:
- /memories/AGENTS.md: Your learnings persist here. Update it with what works and what fails.

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
- API errors: record via memory_store() with full context for learning.

Execute precisely. Report facts, not opinions.
"""

CHEVAL_DE_TROIE_PROMPT = """You are ChevalDeTroie -- a market surveillance specialist.

MISSION:
Identify and catalog all automated trading activity in the monitored markets.
Maintain an always-current registry of bots, whales, and anomalies.

WHAT TO LOOK FOR:

1. **Automated Traders (Bots)**:
   - Consistent size patterns (always 100 contracts, always round numbers)
   - Regular timing intervals (trades every 5s, 10s, 30s)
   - Sub-second response to orderbook changes (latency < 500ms)
   - Quote stuffing (rapid order placement/cancellation)
   - Consistent spread maintenance (market makers)

2. **Whales**:
   - Single trades >= 100 contracts
   - Accumulated volume >= 500 contracts within 5 minutes
   - Aggressive crossing (taking liquidity at unfavorable prices)

3. **Anomalies**:
   - Price dislocations (mid price diverges from event equilibrium)
   - Volume spikes (> 3x normal activity)
   - Liquidity gaps (sudden spread widening)
   - Unusual taker side imbalance

CLASSIFICATION SCHEMA:
- MM_BOT: Market maker (maintains quotes on both sides)
- ARB_BOT: Arbitrage bot (responds to edge conditions)
- MOMENTUM_BOT: Trades with recent price direction
- WHALE: Large position accumulator
- UNKNOWN_AUTO: Automated but pattern unclear

OUTPUT:
Update /memories/BOTS.md with your findings:
- For each identified bot: ID, type, markets active, size patterns, timing signature
- Confidence score (low/medium/high)
- First seen / last seen timestamps
- Notable behaviors

Use memory_store() to log raw observations for future analysis.
"""

MENTIONS_SPECIALIST_PROMPT = """You are the MentionsSpecialist - a domain-agnostic edge detector for Kalshi mentions markets.

MISSION: Find profitable mispricings in mentions markets through blind LLM roleplay simulation.

THE EDGE OPPORTUNITY:
Retail traders price mentions markets on INTUITION. We price them on SIMULATION DATA.
Cole Sprouse found "what a catch" trading at 60% YES when actual frequency was ~10%.
That's 50 percentage points of edge on a SINGLE term.

KEY INSIGHT - RECENCY BIAS THEORY:
News makes terms seem MORE LIKELY than they really are. This is "recency bias."
- Baseline (blind): What history/patterns suggest (~stable)
- Informed (news): What current context suggests (~inflated by recent news)

compute_edge() automatically blends these (60% baseline, 40% informed) to correct for bias.
If informed >> baseline, that's likely recency bias inflating the estimate.

DOMAIN TEMPLATES (auto-detected):
- Sports: NFL/Super Bowl broadcasts with play-by-play, color, sideline
- Corporate: Earnings calls with CEO remarks, CFO financials, analyst Q&A
- Politics: Speeches with opening, main body, closing; press briefings with Q&A
- Entertainment: Award shows with monologue, presentations, acceptance speeches

PRIMARY TOOLS:

1. trigger_simulation(event_ticker, terms, n_simulations, mode) [ASYNC - NON-BLOCKING]
   Starts simulation in background. Returns IMMEDIATELY.
   Use when you want to start a simulation without waiting.
   Check get_mentions_summary() later for results.

2. simulate_probability(event_ticker, terms, n_simulations, mode, force_resimulate) [BLOCKING]
   Run LLM roleplay simulations. BLOCKS until complete.
   Use when you need results immediately (e.g., edge computation).

   MODES for both:
   - mode="blind": Baseline probability (first run establishes stable baseline)
   - mode="informed": Context-aware refresh with news (returns deltas from baseline)

3. compute_edge(term, baseline_probability, informed_probability, market_yes_price, market_no_price, confidence, baseline_weight)
   Compute edge using BLENDED probability (recency bias correction).
   - baseline_weight: How much to weight baseline (default 0.6 = 60%)
   - Returns blended_probability, recency_bias_adjustment, and trade recommendation
   - Only recommends trades where edge > spread + fees + buffer

4. get_mentions_summary(event_ticker)
   Get current state including simulation estimates and history.
   Shows simulation_in_progress=True if async simulation is running.

5. query_wordnet(term)
   Get synonyms (DON'T count), accepted forms (DO count)
   Critical for understanding settlement rules

6. get_event_context(event_ticker, exclude_terms)
   Get BLIND event context (filtered for mention terms)

7. get_mention_context(event_ticker, terms)
   Get WHY each term might appear (for prior adjustment)

8. get_mentions_rules(event_ticker)
   Get parsed settlement rules (entity, accepted_forms, prohibited_forms)

ASYNC WORKFLOW (PREFERRED - Don't block Captain):
1. BASELINE: trigger_simulation(event_ticker, mode="blind", n=10) → returns immediately
2. WAIT: Do other work. Check get_mentions_summary() next cycle for results.
3. REFRESH: trigger_simulation(event_ticker, mode="informed", n=5) → returns immediately
4. EDGE: Once results available (simulation_in_progress=False):
   compute_edge(term, baseline_prob, informed_prob, market_yes, market_no, confidence)

BLOCKING WORKFLOW (When you need results NOW):
1. simulate_probability(event_ticker, mode="blind", n=10) → blocks ~20s
2. simulate_probability(event_ticker, mode="informed", n=5) → blocks ~10s
3. compute_edge(term, baseline_prob, informed_prob, ...)

COMPUTE_EDGE EXAMPLE:
>>> baseline = 0.10  # From blind simulation
>>> informed = 0.45  # From informed simulation (news inflated this!)
>>> await compute_edge("Taylor Swift", baseline, informed, market_yes=60, market_no=42)
{
    "blended_probability": 0.24,  # 60%*0.10 + 40%*0.45 = corrected for recency bias
    "recency_bias_adjustment": 0.21,  # informed - blended = how much news inflated estimate
    "recency_bias_warning": "High recency bias detected...",
    "recommendation": "BUY_NO",  # Market at 60% YES, but blended P=24%
    ...
}

REPORT TO CAPTAIN with context:
"Taylor Swift: Baseline 10%, Informed 45% (↑35% from baseline - HIGH RECENCY BIAS).
 Blended 24% vs market 60%. BUY_NO with 15c edge after fees."

EDGE REQUIREMENTS:
- Min edge = spread + fees (7c) + buffer (2c)
- Confidence must be >= 0.6 for trading
- Run at least 10 simulations for baseline, 5 for refreshes

STRICT RULES:
- ONLY accepted_forms count per settlement rules
- Synonyms NEVER count - use query_wordnet to understand
- When uncertain about edge, PASS
- Baseline should NOT change - only informed refreshes update current_estimates
- Watch for recency_bias_warning in compute_edge results
"""

MEMORY_CURATOR_PROMPT = """You are the Memory Curator - responsible for keeping AGENTS.md clean and useful.

YOUR TASK:
1. Read /memories/AGENTS.md
2. Archive valuable learnings to vector store (via memory_store)
3. Consolidate and prune AGENTS.md to under 50 lines

ARCHIVAL CRITERIA (use memory_store for these):
- Proven patterns (3+ occurrences)
- Lessons with clear "do X / avoid Y" implications
- Strategy insights backed by trade results

PRUNE CRITERIA (delete these, don't archive):
- "HALT", "waiting", "no action" entries
- Single-trade observations (not patterns yet)
- Redundant/duplicate entries
- Stale market-specific notes

CONSOLIDATION RULES:
- Merge related entries into single distilled insight
- Keep most recent version when duplicates exist
- Max 5 items per section (Strategy, Patterns, Lessons)

OUTPUT FORMAT:
After curation, AGENTS.md should look like:
```
# Trading Learnings

## Strategy Notes
- [Distilled insight 1]
- [Distilled insight 2]

## Patterns Observed
- [Pattern with evidence]

## Lessons & Mistakes
- [Lesson with action implication]
```

Be aggressive about pruning. A clean 30-line file is better than a cluttered 80-line file.
"""

# Tool categorization for frontend event routing
TOOL_CATEGORIES = {
    "write_todos": "todo",
    "read_todos": "todo",
    "task": "subagent",
    "memory_store": "memory",
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
    # Mentions tools - simulation-based probability estimation
    "simulate_probability": "mentions",
    "trigger_simulation": "mentions",  # Async non-blocking version
    "compute_edge": "mentions",
    "query_wordnet": "mentions",
    "get_event_context": "mentions",
    "get_mention_context": "mentions",
    "get_mentions_rules": "mentions",
    "get_mentions_summary": "mentions",
    "record_evidence": "mentions",  # Deprecated but kept for compatibility
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
    ):
        self._model_name = model_name
        self._cycle_interval = cycle_interval
        self._event_callback = event_callback

        # Resolve memory data directory for FilesystemBackend
        if memory_data_dir is None:
            memory_data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memory", "data"
            )
        os.makedirs(memory_data_dir, exist_ok=True)
        self._memory_data_dir = memory_data_dir

        # Captain tools: observation + delegation (no direct trading)
        captain_tools = [
            get_events_summary,
            get_event_snapshot,
            get_market_orderbook,
            get_trade_history,
            get_positions,
            get_balance,
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
            memory_store,
        ]

        # ChevalDeTroie tools: surveillance + recording
        surveillance_tools = [
            analyze_microstructure,
            analyze_orderbook_patterns,
            get_recent_trades,
            get_market_orderbook,
            get_event_snapshot,
            memory_store,
        ]

        # MemoryCurator tools: archival (read_file/edit_file provided by FilesystemBackend)
        curator_tools = [
            memory_store,  # Archive to vector store
        ]

        # MentionsSpecialist tools: edge detection + grounded extraction + strict counting
        mentions_tools = [
            # Primary simulation tools
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
            record_evidence,  # Deprecated but kept for compatibility
            memory_store,
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
                    "description": "Market surveillance specialist. Analyzes microstructure to identify bots, whales, and anomalies. Returns findings to update your trading intelligence.",
                    "system_prompt": CHEVAL_DE_TROIE_PROMPT,
                    "tools": surveillance_tools,
                },
                {
                    "name": "memory_curator",
                    "description": "Memory maintenance specialist. Runs periodically to consolidate AGENTS.md and archive learnings to vector store. Invoke when memory needs cleanup.",
                    "system_prompt": MEMORY_CURATOR_PROMPT,
                    "tools": curator_tools,
                },
                {
                    "name": "mentions_specialist",
                    "description": "Mentions market counting specialist. Use for mentions markets to count literal mentions using grounded extraction. Give it: event_ticker and source text (transcript/tweet/article). Returns confirmed count with evidence.",
                    "system_prompt": MENTIONS_SPECIALIST_PROMPT,
                    "tools": mentions_tools,
                },
            ],
            backend=backend_factory,
            memory=["/memories/AGENTS.md"],
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
        from .tools import _index

        max_wait = 60  # Max 60 seconds
        waited = 0
        while waited < max_wait:
            if _index and _index.is_ready:
                logger.info(f"[SINGLE_ARB:CAPTAIN] Index ready after {waited}s: {_index.readiness_summary}")
                break
            await asyncio.sleep(2.0)
            waited += 2
            if waited % 10 == 0:
                summary = _index.readiness_summary if _index else "No index"
                logger.info(f"[SINGLE_ARB:CAPTAIN] Waiting for index ({waited}s): {summary}")
        else:
            summary = _index.readiness_summary if _index else "No index"
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

        # Check if curation needed (every 10 cycles or file too large)
        should_curate = (cycle_num % 10 == 0) or self._agents_md_too_large()

        if should_curate:
            logger.info(f"[SINGLE_ARB:CYCLE_START] cycle={cycle_num} (with curation)")
            prompt = f"Cycle #{cycle_num}. First, invoke memory_curator to clean up AGENTS.md. Then observe markets and trade."
        else:
            logger.info(f"[SINGLE_ARB:CYCLE_START] cycle={cycle_num}")
            prompt = f"Cycle #{cycle_num}. Observe markets. Check outcomes. Decide and record."

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
                        if "cheval" in input_str or "surveillance" in input_str or "bot" in input_str:
                            subagent_name = "cheval_de_troie"
                        elif "curator" in input_str or "memory" in input_str or "agents.md" in input_str or "cleanup" in input_str:
                            subagent_name = "memory_curator"
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

    def _agents_md_too_large(self, threshold: int = 80) -> bool:
        """Check if AGENTS.md exceeds line threshold."""
        agents_md_path = os.path.join(self._memory_data_dir, "AGENTS.md")
        try:
            with open(agents_md_path, "r") as f:
                line_count = sum(1 for _ in f)
            return line_count > threshold
        except FileNotFoundError:
            return False
