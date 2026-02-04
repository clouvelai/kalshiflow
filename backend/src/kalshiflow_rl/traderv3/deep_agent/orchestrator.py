"""
Arb Orchestrator v2 - Captain agent with EventAnalyst and MemoryCurator subagents.

Captain (create_react_agent) delegates to subagents via tool calls.
Trade execution via self-contained buy_arb_position / sell_arb_position tools.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from .prompts import CAPTAIN_PROMPT, EVENT_ANALYST_PROMPT, MEMORY_CURATOR_PROMPT

from .tools.kalshi_tools import (
    kalshi_get_events,
    kalshi_get_markets,
    kalshi_get_orderbook,
)
from .tools.poly_tools import (
    poly_get_events,
    poly_get_markets,
    poly_search_events,
)
from .tools.memory_tools import (
    memory_store,
    memory_search,
    save_validation,
    get_memory_stats,
    dedup_memories,
    consolidate_memories,
    prune_stale_memories,
)
from .tools.data_tools import (
    get_pair_snapshot,
    get_spread_snapshot,
    get_event_codex,
    get_validation_status,
)
from .tools.trade_tools import buy_arb_position, sell_arb_position
from .tools.db_tools import get_system_state, get_pair_history

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.orchestrator")


@dataclass
class ArbOrchestratorConfig:
    """Configuration for the arb orchestrator."""
    model_name: str = "claude-sonnet-4-20250514"
    scan_interval_seconds: float = 300.0  # 5 minutes between scan cycles
    max_cycle_duration_seconds: float = 120.0


class ArbOrchestrator:
    """
    Captain agent orchestrator with EventAnalyst and MemoryCurator subagents.

    Captain runs one create_react_agent invocation per cycle. Subagents are
    invoked via delegation tools that spin up temporary agents.
    """

    def __init__(self, config: Optional[ArbOrchestratorConfig] = None):
        self._config = config or ArbOrchestratorConfig()
        self._llm = ChatAnthropic(model=self._config.model_name)
        self._event_callback: Optional[Callable[..., Coroutine]] = None

        # Build subagent delegation tools (closures over self._llm and self._emit_event)
        delegate_analyst = self._make_delegate_event_analyst()
        delegate_curator = self._make_delegate_memory_curator()

        # Captain's tool set
        captain_tools = [
            delegate_analyst,
            delegate_curator,
            buy_arb_position,
            sell_arb_position,
            get_pair_snapshot,
            get_spread_snapshot,
            get_event_codex,
            get_validation_status,
            get_system_state,
            memory_store,
            memory_search,
        ]

        self._captain = create_react_agent(
            self._llm,
            captain_tools,
            prompt=SystemMessage(content=CAPTAIN_PROMPT),
        )

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._cycle_count = 0
        self._last_cycle_at: Optional[float] = None
        self._errors: List[str] = []

    def _make_delegate_event_analyst(self):
        """Create the delegate_event_analyst tool (closure over llm + emit)."""
        llm = self._llm
        orchestrator = self

        # EventAnalyst tool set — focused on validating pairings, not trading or portfolio
        analyst_tools = [
            get_event_codex,
            get_pair_snapshot,
            get_pair_history,
            kalshi_get_orderbook,
            memory_store,
            save_validation,
        ]

        @tool
        async def delegate_event_analyst(
            task: str,
            pair_id: Optional[str] = None,
            event_ticker: Optional[str] = None,
            question: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Delegate analysis to the EventAnalyst subagent.

            The EventAnalyst validates pair quality and data health.
            Use for: validating pairs, investigating events.

            Args:
                task: What to do - "validate" (validate pair for trading),
                      "analyze" (free-form analysis question)
                pair_id: UUID of the pair to analyze (required for validate)
                event_ticker: Kalshi event ticker for context
                question: Free-form analysis question (for task="analyze")

            Returns:
                Dict with the analyst's findings and recommendations
            """
            # Build prompt for analyst
            parts = [f"Task: {task}"]
            if pair_id:
                parts.append(f"Pair ID: {pair_id}")
            if event_ticker:
                parts.append(f"Event Ticker: {event_ticker}")
            if question:
                parts.append(f"Question: {question}")
            prompt = "\n".join(parts)

            await orchestrator._emit_event({
                "type": "agent_message",
                "subtype": "subagent_start",
                "agent": "event_analyst",
                "prompt": prompt[:200],
            })

            try:
                analyst_graph = create_react_agent(
                    llm, analyst_tools,
                    prompt=SystemMessage(content=EVENT_ANALYST_PROMPT),
                )
                result = await orchestrator._run_subagent_graph(
                    "event_analyst", analyst_graph, prompt
                )
                return result
            except Exception as e:
                logger.error(f"[V3:AGENT_ERROR] agent=event_analyst task={task} error={e}")
                logger.error(f"EventAnalyst failed: {e}")
                return {"error": str(e)}

        return delegate_event_analyst

    def _make_delegate_memory_curator(self):
        """Create the delegate_memory_curator tool (closure over llm + emit)."""
        llm = self._llm
        orchestrator = self

        curator_tools = [
            memory_search,
            memory_store,
            get_memory_stats,
            dedup_memories,
            consolidate_memories,
            prune_stale_memories,
        ]

        @tool
        async def delegate_memory_curator(
            task: str = "maintenance",
        ) -> Dict[str, Any]:
            """Delegate memory maintenance to the MemoryCurator subagent.

            The MemoryCurator cleans up duplicate, stale, and fragmented memories.
            Run periodically (every ~10 cycles) to keep memory efficient.

            Args:
                task: What to do - "maintenance" (full cleanup run)

            Returns:
                Dict with the curator's maintenance report
            """
            prompt = f"Run memory maintenance: {task}"

            await orchestrator._emit_event({
                "type": "agent_message",
                "subtype": "subagent_start",
                "agent": "memory_curator",
                "prompt": prompt,
            })

            try:
                curator_graph = create_react_agent(
                    llm, curator_tools,
                    prompt=SystemMessage(content=MEMORY_CURATOR_PROMPT),
                )
                result = await orchestrator._run_subagent_graph(
                    "memory_curator", curator_graph, prompt
                )
                return result
            except Exception as e:
                logger.error(f"[V3:AGENT_ERROR] agent=memory_curator error={e}")
                logger.error(f"MemoryCurator failed: {e}")
                return {"error": str(e)}

        return delegate_memory_curator

    def set_event_callback(self, callback) -> None:
        """Set callback for streaming agent events to frontend."""
        self._event_callback = callback

    async def start(self) -> None:
        """Start the orchestrator loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Arb orchestrator v2 started (model={self._config.model_name}, "
            f"interval={self._config.scan_interval_seconds}s)"
        )

    async def stop(self) -> None:
        """Stop the orchestrator loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Arb orchestrator stopped (cycles={self._cycle_count})")

    async def _run_loop(self) -> None:
        """Main orchestrator loop."""

        while self._running:
            try:
                await self._run_cycle()
                self._cycle_count += 1
                self._last_cycle_at = time.time()
                await asyncio.sleep(self._config.scan_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_msg = f"Orchestrator cycle error: {e}"
                logger.error(f"[V3:AGENT_ERROR] agent=captain cycle={self._cycle_count + 1} error={e}")
                logger.error(error_msg)
                self._errors.append(error_msg)
                await asyncio.sleep(30.0)

    async def _run_cycle(self) -> None:
        """Run one Captain cycle."""
        cycle_num = self._cycle_count + 1
        cycle_start = time.time()
        logger.info(f"[V3:CYCLE_START] cycle={cycle_num}")
        logger.info(f"Starting Captain cycle #{cycle_num}")

        # Build cycle-specific prompt
        prompt_parts = [
            f"Cycle #{cycle_num}.",
            "Scan spreads, check validations, and execute if opportunities exist.",
        ]
        if cycle_num % 10 == 0:
            prompt_parts.append(
                "This is a maintenance cycle. After trading checks, delegate to memory_curator for cleanup."
            )

        prompt = " ".join(prompt_parts)

        await self._emit_event({
            "type": "agent_message",
            "subtype": "subagent_start",
            "agent": "captain",
            "prompt": prompt[:200],
        })

        result = await self._run_captain(prompt)

        await self._emit_event({
            "type": "agent_message",
            "subtype": "subagent_complete",
            "agent": "captain",
            "response_preview": str(result)[:500] if result else "",
        })

        cycle_duration = time.time() - cycle_start
        logger.info(f"[V3:CYCLE_END] cycle={cycle_num} duration={cycle_duration:.1f}s")
        logger.info(f"Captain cycle #{cycle_num} complete")

    @staticmethod
    def _extract_text(content) -> str:
        """Extract text from LLM content (may be str or list of content blocks)."""
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
        return str(content)

    async def _run_captain(self, prompt: str) -> Optional[str]:
        """Run the Captain agent, streaming tool call and LLM token events."""
        try:
            result_text = None
            token_buffer = []
            token_count = 0
            async for event in self._captain.astream_events(
                {"messages": [HumanMessage(content=prompt)]},
                version="v2",
                config={"recursion_limit": 50},
            ):
                kind = event.get("event")

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
                                "agent": "captain",
                                "text": "".join(token_buffer),
                            })
                elif kind == "on_tool_start":
                    # Flush thinking buffer before tool call
                    if token_buffer:
                        await self._emit_event({
                            "type": "agent_message",
                            "subtype": "thinking_complete",
                            "agent": "captain",
                            "text": "".join(token_buffer),
                        })
                        token_buffer.clear()
                        token_count = 0
                    tool_name = event.get("name", "unknown")
                    await self._emit_event({
                        "type": "agent_message",
                        "subtype": "tool_call",
                        "agent": "captain",
                        "tool_name": tool_name,
                        "tool_input": str(event.get("data", {}).get("input", ""))[:200],
                    })
                elif kind == "on_tool_end":
                    await self._emit_event({
                        "type": "agent_message",
                        "subtype": "tool_result",
                        "agent": "captain",
                        "tool_name": event.get("name", "unknown"),
                        "tool_output": str(event.get("data", {}).get("output", ""))[:300],
                    })
                elif kind == "on_chain_end":
                    if event.get("name") == "LangGraph":
                        output = event.get("data", {}).get("output", {})
                        messages = output.get("messages", [])
                        if messages:
                            result_text = self._extract_text(messages[-1].content)

            # Flush remaining tokens
            if token_buffer:
                await self._emit_event({
                    "type": "agent_message",
                    "subtype": "thinking_complete",
                    "agent": "captain",
                    "text": "".join(token_buffer),
                })

            return result_text

        except Exception as e:
            logger.error(f"[V3:AGENT_ERROR] agent=captain error={e}")
            logger.error(f"Captain run failed: {e}")
            await self._emit_event({
                "type": "agent_message",
                "subtype": "subagent_error",
                "agent": "captain",
                "error": str(e),
            })
            return None

    async def _run_subagent_graph(
        self, name: str, graph, prompt: str
    ) -> Dict[str, Any]:
        """Run a subagent graph with streaming, return last message content as dict."""
        start = time.time()
        result_text = ""
        token_buffer = []
        token_count = 0

        async for event in graph.astream_events(
            {"messages": [HumanMessage(content=prompt)]},
            version="v2",
            config={"recursion_limit": 40},
        ):
            kind = event.get("event")

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
                            "agent": name,
                            "text": "".join(token_buffer),
                        })
            elif kind == "on_tool_start":
                # Flush thinking buffer before tool call
                if token_buffer:
                    await self._emit_event({
                        "type": "agent_message",
                        "subtype": "thinking_complete",
                        "agent": name,
                        "text": "".join(token_buffer),
                    })
                    token_buffer.clear()
                    token_count = 0
                await self._emit_event({
                    "type": "agent_message",
                    "subtype": "tool_call",
                    "agent": name,
                    "tool_name": event.get("name", "unknown"),
                    "tool_input": str(event.get("data", {}).get("input", ""))[:200],
                })
            elif kind == "on_tool_end":
                await self._emit_event({
                    "type": "agent_message",
                    "subtype": "tool_result",
                    "agent": name,
                    "tool_name": event.get("name", "unknown"),
                    "tool_output": str(event.get("data", {}).get("output", ""))[:300],
                })
            elif kind == "on_chain_end":
                if event.get("name") == "LangGraph":
                    output = event.get("data", {}).get("output", {})
                    messages = output.get("messages", [])
                    if messages:
                        result_text = self._extract_text(messages[-1].content)

        # Flush remaining tokens
        if token_buffer:
            await self._emit_event({
                "type": "agent_message",
                "subtype": "thinking_complete",
                "agent": name,
                "text": "".join(token_buffer),
            })

        duration = time.time() - start
        await self._emit_event({
            "type": "agent_message",
            "subtype": "subagent_complete",
            "agent": name,
            "duration": round(duration, 1),
            "response_preview": str(result_text)[:500],
        })

        logger.info(f"Subagent [{name}] completed in {duration:.1f}s")
        return {"response": result_text, "duration": round(duration, 1)}

    async def _emit_event(self, event_data: Dict[str, Any]) -> None:
        """Emit event via callback, never raising."""
        if not self._event_callback:
            return
        try:
            await self._event_callback(event_data)
        except Exception as e:
            logger.warning(f"Event callback failed: {e}")

    async def handle_user_message(self, message: str) -> str:
        """Handle a user message via the Captain agent."""
        try:
            result = await self._captain.ainvoke(
                {"messages": [HumanMessage(content=message)]},
                config={"recursion_limit": 50},
            )
            messages = result.get("messages", [])
            return messages[-1].content if messages else "No response"
        except Exception as e:
            return f"Error: {e}"

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "version": "v2",
            "cycle_count": self._cycle_count,
            "last_cycle_at": self._last_cycle_at,
            "model": self._config.model_name,
            "scan_interval": self._config.scan_interval_seconds,
            "recent_errors": self._errors[-5:] if self._errors else [],
        }

    def is_healthy(self) -> bool:
        """Health check."""
        return self._running and self._task is not None and not self._task.done()
