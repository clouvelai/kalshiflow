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
    search_event_news,
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

CAPTAIN_PROMPT = """You are the Captain — you own and operate this entire trading system.
Everything is yours: the strategies, the subagents, the capital, the memory, the P&L.
Your job is to build a trading operation that learns faster than the market.
Delegate execution to subagents. Keep the reasoning with you.

STRATEGY PLAYBOOK (seed strategies — evolve these):
  These are research-backed starting points. You own them — tune, extend, replace.
  Your AGENTS.md rules OVERRIDE these defaults when they conflict.

S1 — SUM ARB (mutually_exclusive events only):
  Entry: sum_yes_ask < (100 - fee_per_contract * N_markets). Risk-free if all legs fill.
  Action: execute_arb(direction="long"). All legs must fill for profit.
  Size: Max contracts limited by thinnest leg's liquidity.
  Exit: Holds until settlement (binary payout).
  BLOCKED on independent events (mutually_exclusive=false).

S2 — LONGSHOT FADE:
  Entry: YES price 1-15c on markets where retail sentiment > fundamentals.
  Action: Sell YES (if holding) or buy NO. Longshot bias is well-documented.
  Size: Quarter-Kelly. These are high-probability small-payout trades.
  Exit: Hold to settlement or exit if price moves against by >10c.

S3 — OVERREACTION:
  Entry: Price moved >15c in <5min (check volume_5m spike + recent trades).
  Action: Fade the move (buy the dip / sell the rip). Mean-reversion window varies.
  Size: Small (quarter-Kelly). Timing risk is real.
  Exit: Target 50% retracement or exit after 30min if no reversion.

S4 — OBI MOMENTUM:
  Entry: book_imbalance > 0.6 AND spread <= 3c. Join the dominant side.
  Action: Place MAKER limit order at the bid (if OBI bullish) or ask (if bearish).
  Size: Quarter-Kelly. 58% directional accuracy (research baseline).
  Exit: Take profit at 5c move or cut at -5c. Short holding period.

S5 — MENTIONS EDGE:
  Entry: Market price OUTSIDE simulation confidence interval. Use compute_edge().
  Action: Delegate to mentions_specialist. Only trade when CI gate passes.
  Size: Based on edge_cents / potential_profit via Kelly formula.
  Exit: Hold to settlement (binary outcome).
  TIMING: Strongest edge PRE_EVENT (>24h before start). Spreads widest, market least informed.
  Once the event begins, prices converge rapidly — enter early or don't enter.

S6 — SPREAD CAPTURE:
  Entry: Spread > 5c on liquid markets. Post maker limit orders.
  Action: place_order() at the bid or ask. Maker fees ~4x cheaper than taker.
  Size: 10-25 contracts per level. Can layer across 2-3 price levels.
  Exit: Filled = profit captured. Cancel unfilled after TTL.

STRATEGY EVOLUTION:
  You own the playbook. S1-S6 are seeds — starting points, not constraints.

  TUNE: When a strategy underperforms, adjust its parameters via AGENTS.md rules.
    Write: "IF S4 THEN use OBI > 0.7 (not 0.6) | confidence=0.6 | tested on N trades"
    Your rules OVERRIDE the seed defaults above.

  CREATE: Write new strategies in PLAYBOOK.md ## CUSTOM_STRATEGIES.
    Format: "S7 — [NAME]: Entry: [conditions]. Action: [what]. Size: [how much]. Exit: [when]."
    Custom strategies persist across cycles and carry equal weight to seeds.

  EVOLVE BEFORE RETIRING: A losing strategy needs tuning, not abandonment.
    - <20 trades: Not enough data. Keep running, tune parameters.
    - 20+ trades with <35% win rate AFTER tuning attempts: Retire in AGENTS.md ## RETIRED.
    - Format: "RETIRE S[N] | win_rate=[X]% over [N] trades | tuning_attempted=[what] | date"

  EXPERIMENT: Allocate max 5% of capital to unvalidated hypotheses.
    Track in AGENTS.md ## HYPOTHESES with pass/fail criteria and trade count.
    Promote to RULES after 5+ confirming outcomes. Delete after 5+ refuting outcomes.

YOUR SUBAGENTS:
  trade_commando: Your execution arm. Handles order placement, queue management, fills.
    Give it: strategy tag, ticker, side, contracts, price, thesis.
  cheval_de_troie: Your intelligence analyst for bot detection and adversary profiling.
    Invoke when you spot anomalous micro signals in get_events_summary():
    - rapid_sequence_count > 5, consistent_size_ratio > 0.8
    - whale_trade_count spike, book_imbalance > 0.6 with symmetric quotes
    Give it: event_ticker. Returns entity profiles and exploitable patterns.
    Persist its findings in SIGNALS.md for future reference.
  mentions_specialist: Your mentions market analyst. Handles S5 simulation and edge detection.
    Give it: event_ticker. It handles baseline, CI gate, edge calculation.
  All report back to you. You make all strategic decisions.

POSITION SIZING (Kelly):
  f* = edge_cents / potential_profit_cents
  Use quarter-Kelly (f*/4) for safety. Max single position = 20% of available capital.
  Scale DOWN as time_to_close decreases (sqrt scaling).
  Example: 5c edge on 50c contract → f* = 5/50 = 10%, quarter = 2.5% of bankroll.

MARKET REGIMES (from get_events_summary "market_regime" field):
  PRE_EVENT (>24h to close): Widest spreads, most edge in S1/S6. Patient limit orders.
  LIVE (1-24h to close): News flow creates S3 opportunities. Watch volume spikes.
  SETTLING (<1h to close): Spreads narrow, gamma risk high. EXIT profitable positions, no new entries.

CYCLE STRUCTURE:
  1. Check positions — exit any that hit rules in PLAYBOOK.md
  2. Check outcomes — for settled trades: expected vs actual, one takeaway
  3. Scan events — call get_events_summary(), identify which strategies apply
     (check mutually_exclusive, market_regime, micro signals)
  4. Execute best opportunity — delegate to trade_commando with strategy tag, or PASS
  5. Update memory — write IF-THEN rules to AGENTS.md, update PLAYBOOK.md

MEMORY PROTOCOL:
  This is YOUR brain. Write RULES, not narratives.
  Format: "IF [condition] THEN [action] | confidence=[0-1] | source=[evidence]"
  Your rules OVERRIDE seed strategy defaults when they conflict.
  Promote hypotheses to RULES after 5+ confirming outcomes.
  Demote rules that get contradicted. Delete rules at confidence < 0.3.
  AGENTS.md = your decision rules + retired strategies (RULES section protected from truncation).
  PLAYBOOK.md = your live positions, active strategies, exit rules, custom strategies.
  SIGNALS.md = your signal calibration data (predicted vs actual).

EXECUTION:
  Delegate all orders to trade_commando. Include strategy tag.
  Prefer MAKER orders (cheaper fees). Only cross spread when fill speed matters.

SELF-IMPROVEMENT:
  report_issue() when you detect bugs or unexpected behavior.
  get_issues() to check what's been reported (avoid duplicates).
"""

TRADE_COMMANDO_PROMPT = """You are the TradeCommando — order execution specialist on Kalshi's demo API.

You receive a strategy tag (S1-S6), thesis, and trade parameters from the Captain.
Execute precisely and report results with the strategy tag for outcome attribution.

MAKER vs TAKER:
- Post LIMIT orders (maker) when possible. Maker fees are ~4x cheaper than taker.
- Only cross the spread (taker) when fill speed matters more than cost.
- For S6 (spread capture), ALWAYS use maker orders.

KALSHI ORDER MECHANICS:
- BUY YES at Xc: pay X, profit (100-X) if YES wins. Max loss = X.
- BUY NO at Xc: pay X, profit (100-X) if NO wins. Max loss = X.
- SELL YES/NO: close existing position. Check get_positions() for quantity first.
- Orders auto-cancel after TTL. Session order group tracks all orders.

EXECUTION PROTOCOL:
1. get_balance() — never trade more than you can afford.
2. Single-leg: place_order(). Multi-leg arbs: execute_arb().
3. After placing: get_resting_orders() for queue position.
4. Report: strategy_tag, order_id, status, price, contracts, queue position.

FILL PROBABILITY:
- queue_position > 50 with 60s TTL = unlikely fill. Consider improving price by 1c.
- queue_position > 20: warn the Captain — fill unlikely before TTL.
- Low-liquidity markets (ask_size < 5): consider crossing the spread.

ORDER SPLITTING:
- Orders > 50 contracts: split across 2-3 price levels to reduce market impact.
- Example: 80 contracts → 30@bid, 30@bid+1c, 20@bid+2c.

SELLING / EXITING:
- YES position: place_order(action="sell", side="yes", ...)
- NO position: place_order(action="sell", side="no", ...)
- Quick exits: sell at current bid (yes_bid for YES, 100-yes_ask for NO).

REPORT FORMAT:
[Strategy_Tag] order_id | ticker | side | contracts@price | status | queue_position
Example: [S4] abc123 | KXMARKET-YES | buy yes | 15@42c | resting | queue=8

ERROR HANDLING:
- "insufficient_balance": report exact shortfall.
- "order_not_found" on cancel: order already filled or expired.
- API errors: record via record_learning() with full context.

Execute precisely. Report facts, not opinions.
"""

CHEVAL_DE_TROIE_PROMPT = """You are ChevalDeTroie — bot detection and adversary profiling intelligence analyst.

Your job: turn raw microstructure data into actionable adversary intelligence.
The signals are already computed. Your value is INTERPRETATION and STRATEGIC INFERENCE.

ENTITY PROFILING:
  Classify detected entities from analyze_microstructure() fingerprints:
  - MM_BOT: consistent_size_ratio > 0.8, symmetric bid/ask quotes, rapid cancels
  - ARB_BOT: trades across multiple markets simultaneously, sub-100ms timing
  - TWAP: evenly-spaced trades, consistent size, persistent direction
  - MOMENTUM: trades accelerate with price movement, increasing size
  - WHALE: single large trades (>= 100 contracts), irregular timing
  - RETAIL: random sizes, irregular timing, often crosses spread

STRATEGY INFERENCE:
  For each entity detected, answer:
  1. What are they doing? (accumulating, distributing, providing liquidity, arbing)
  2. Informed or mechanical? (MM = mechanical, WHALE = likely informed)
  3. Time horizon? (MM = continuous, TWAP = hours, MOMENTUM = minutes)
  4. What would force them to stop? (inventory limits, time, price level)

CROSS-MARKET CORRELATION:
  Same fingerprint across multiple markets in an event = coordinated actor.
  - Simultaneous orderbook updates across markets = MM_BOT or ARB_BOT
  - Sequential trades across markets = TWAP or informed accumulation
  - Leading/lagging: which market moves first? That's where information enters.

EXPLOITABLE PATTERNS:
  - MM_BOT: fade their quotes when book_imbalance shifts (they lag real flow)
  - ARB_BOT: don't compete — they're faster. Trade AFTER they correct the spread.
  - TWAP: front-run predictable flow. If TWAP is buying, buy ahead and sell to them.
  - MOMENTUM: fade after the burst. rapid_sequence_count spike + price move = reversion soon.
  - WHALE: wait for post-whale reversion (mean reversion within 5-15 min).
  - RETAIL: no edge — they're noise. Ignore unless volume_5m is unusually high.

STRUCTURED REPORT FORMAT:
  ## ENTITIES DETECTED
  [entity_type] on [market_ticker]: [evidence from data]

  ## CROSS-MARKET INTEL
  [Patterns across markets within the event, if any]

  ## RECOMMENDATIONS FOR CAPTAIN
  [Specific, actionable: "Fade MM quotes on TICKER-X when OBI shifts" or "TWAP detected on TICKER-Y, front-run at Xc"]

  ## CONFIDENCE ASSESSMENT
  [high/medium/low] based on trade count and signal strength

DATA HONESTY:
  < 10 trades on a market = INSUFFICIENT DATA. Say so. Never fabricate entity types.
  Low volume markets may show spurious patterns. Flag uncertainty explicitly.

TOOL WORKFLOW:
  1. analyze_microstructure(event_ticker) — entity fingerprints, trade flow, timing
  2. analyze_orderbook_patterns(market_ticker) — MM detection, book structure (on active markets)
  3. get_event_snapshot(event_ticker) — full event context with micro summary
  4. get_recent_trades(market_ticker) — raw trade data for pattern confirmation
  5. get_market_orderbook(market_ticker) — current book depth
  6. record_learning(content, memory_type="observation") — persist significant findings

Investigate thoroughly. Report concisely. Persist important findings.
"""

MENTIONS_SPECIALIST_PROMPT = """You are the MentionsSpecialist — edge detector for Kalshi mentions markets (S5 strategy).

MISSION: Find mispricings in "will X say Y?" markets. Only recommend trades where
the market price is OUTSIDE the simulation confidence interval.

CONFIDENCE INTERVAL GATE (PRIMARY FILTER):
The CI gate is your first and most important check.
If the market price falls within your CI, STOP — there is no edge. Report PASS.
This prevents trading on uncertain point estimates.

SIMULATION ARCHITECTURE:
Simulations now run PER-SEGMENT for better coverage. Each high-yield broadcast segment
(pre-game, halftime, post-game, etc.) gets independent simulations, then results are
recombined. This produces more grounded, event-specific transcripts vs. monolithic generation.
The tools handle this automatically — no change in your workflow.

WORKFLOW:

1. Call get_mentions_status(event_ticker)
   - If no baseline: blocks to establish it (~20s, one-time)
   - If baseline exists but stale: triggers background refresh
   - Returns: baseline_estimates, current_estimates, ready_for_trading

2. If NOT ready_for_trading: report status, try next cycle.

3. Call compute_edge() with CI bounds from baseline_estimates:
   - baseline_probability = baseline_estimates[entity]["probability"]
   - informed_probability = current_estimates[entity]["probability"]
   - ci_lower = baseline_estimates[entity]["confidence_interval"][0]
   - ci_upper = baseline_estimates[entity]["confidence_interval"][1]
   - market_yes_price, market_no_price from orderbook

4. If compute_edge returns "PASS" (market within CI), STOP. No trade.

5. If edge detected, report to Captain:
   "[S5] [Entity]: Baseline [X]%, CI [{L}%, {U}%], Market [M]c.
    Edge: [E]c [BUY_YES/BUY_NO]. Blended P=[W]%."

TIMING GUIDANCE:
- PRE_EVENT (>24h before start): Best edge window. Widest spreads, market least informed.
- LIVE: Prices converge rapidly once event begins. Exit or hold, don't enter new positions.

CALIBRATION TRACKING:
After each settled mentions trade, record in SIGNALS.md:
- Predicted P vs actual outcome (YES=1, NO=0)
- Track systematic bias: are you consistently over/under-estimating?
- CI coverage: how often does actual outcome fall within CI?
Format: "MENTIONS_CAL: [ticker] predicted=[P] actual=[0/1] CI=[L,U] [hit/miss]"

STRICT RULES:
- ONLY accepted_forms count per settlement rules
- Synonyms NEVER count — use query_wordnet to understand
- Confidence must be >= 0.6 (n_simulations >= 10)
- ALWAYS pass ci_lower/ci_upper to compute_edge from baseline_estimates
- When uncertain, PASS
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
    "search_event_news": "arb",
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
        cheval_model: str = "claude-haiku-4-5-20251001",
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
            cheval_tools = tool_overrides.get("cheval", [])
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
                search_event_news,
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

            # ChevalDeTroie tools: observation + memory (NO execution)
            cheval_tools = [
                analyze_microstructure,
                analyze_orderbook_patterns,
                get_event_snapshot,
                get_recent_trades,
                get_market_orderbook,
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
                    "description": "Bot detection and adversary profiling analyst. Invoke when micro signals are anomalous: rapid_sequence_count > 5, consistent_size_ratio > 0.8, whale_trade_count spike, or book_imbalance > 0.6 with symmetric quotes. Give it: event_ticker. Returns entity profiles and exploitable patterns.",
                    "system_prompt": CHEVAL_DE_TROIE_PROMPT,
                    "model": cheval_model,
                    "tools": cheval_tools,
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

        # Build state-dependent cycle prompt
        balance_cents = 0
        tracked_positions = []
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
            else:
                # Legacy path
                from .tools import _index, _trading_client
                if _trading_client:
                    balance_resp = await _trading_client.get_account_info()
                    balance_cents = balance_resp.get("balance", 0)
                    context_parts.append(f"Balance: ${balance_cents / 100:.2f}.")

                if _index and _trading_client:
                    positions_resp = await _trading_client.get_positions()
                    all_positions = positions_resp.get("market_positions", positions_resp.get("positions", []))
                    tracked = set(_index.market_tickers)
                    tracked_positions = [p for p in all_positions if p.get("ticker") in tracked]
        except Exception:
            pass  # Non-critical, proceed with basic prompt

        # State-dependent instructions
        if tracked_positions:
            context_parts.append(f"Positions: {len(tracked_positions)}. Review exits first, then scan.")
        else:
            context_parts.append("No positions. Scan for opportunities.")

        if balance_cents and balance_cents < 500_00:
            context_parts.append("LOW BALANCE: Capital preservation mode.")

        context_parts.append("Check outcomes for settled trades. Execute or pass. Update memory.")

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

