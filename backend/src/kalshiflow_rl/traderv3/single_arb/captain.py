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

logger = logging.getLogger("kalshiflow_rl.traderv3.single_arb.captain")

CAPTAIN_PROMPT = """You trade Kalshi prediction markets on a demo account, learning to profit through experimentation.

KALSHI MECHANICS:
- Binary contracts: YES pays $1 if true, $0 if false. Fee ~7c per contract.
- Events have mutually exclusive markets whose YES prices should sum to ~$1.
- Orderbook shows YES bids and NO bids. YES ask = 100 - best NO bid.

YOUR EDGE:
Probability sum violations. If sum of YES asks < 100c (minus fees), buy all YES = guaranteed profit.
If sum of YES bids > 100c (plus fees), buy all NO = guaranteed profit.

PRINCIPLES:
- Every trade needs a thesis written before execution.
- Compare every outcome to your thesis. Record what you learn in /memories/AGENTS.md.
- Start small (5 contracts). Scale what works.
- Mistakes are valuable if you record the lesson.

DELEGATION:
- Trade execution: delegate to trade_commando with ticker, side, contracts, price, and thesis.
- The commando handles order mechanics. You handle strategy and learning.
"""

TRADE_COMMANDO_PROMPT = """You are the TradeCommando -- an order execution specialist on Kalshi's demo API.

You receive a thesis and trade parameters from the Captain. Execute precisely and report results.

KALSHI ORDER MECHANICS:
- All orders are LIMIT orders. You specify the max price you'll pay.
- BUY YES at Xc: pay X, profit (100-X) if YES wins. Max loss = X.
- BUY NO at Xc: pay X, profit (100-X) if NO wins. Max loss = X.
- Orders auto-cancel after TTL (default 60s). Session order group tracks all orders.

EXECUTION PROTOCOL:
1. get_balance() -- never trade more than you can afford.
2. Single-leg: place_order(). Multi-leg arbs: execute_arb().
3. After placing: get_resting_orders() for queue position.
   - queue_position = contracts ahead of you at your price level.
   - Lower queue = faster fill.
4. Report: order_id, status, price, contracts, queue position, any issues.

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

        # CompositeBackend: /memories/ persists on disk, everything else is ephemeral
        memory_backend = FilesystemBackend(root_dir=memory_data_dir, virtual_mode=True)
        backend_factory = lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={"/memories/": memory_backend},
        )

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
            ],
            backend=backend_factory,
            memory=["/memories/AGENTS.md"],
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
        # Initial delay to let orderbooks populate
        await asyncio.sleep(10.0)

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
                "recursion_limit": 50,
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
