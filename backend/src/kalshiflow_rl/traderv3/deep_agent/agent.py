"""
Self-Improving Deep Agent - Main agent class for autonomous trading.

The SelfImprovingAgent implements the observe-act-reflect loop:
1. OBSERVE: Load memory, check markets, search news
2. ACT: Make trading decisions based on analysis
3. REFLECT: Learn from outcomes and update memory

The agent uses Claude as the LLM backbone with tool calling for
executing trades, searching news, and managing memory files.
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Literal, TYPE_CHECKING

from anthropic import AsyncAnthropic

from .tools import DeepAgentTools, set_global_tools
from .reflection import ReflectionEngine, PendingTrade, ReflectionResult

if TYPE_CHECKING:
    from ..core.websocket_manager import V3WebSocketManager
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.state_container import V3StateContainer
    from ..state.tracked_markets import TrackedMarketsState
    from ..services.event_position_tracker import EventPositionTracker

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.agent")


@dataclass
class DeepAgentConfig:
    """Configuration for the deep agent."""
    # Model settings
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Loop settings
    cycle_interval_seconds: float = 60.0  # How often to run observe-act cycle
    max_trades_per_cycle: int = 3
    max_positions: int = 5

    # Memory settings
    memory_dir: Optional[str] = None  # Defaults to ./memory
    memory_files: List[str] = field(default_factory=lambda: [
        "learnings.md",
        "strategy.md",
        "mistakes.md",
        "patterns.md",
    ])

    # Target events
    target_events: List[str] = field(default_factory=list)

    # Safety settings
    max_contracts_per_trade: int = 25
    min_spread_cents: int = 3  # Only trade if spread > this (inefficiency)
    require_fresh_news: bool = True
    max_news_age_hours: float = 2.0


# System prompt for the self-improving agent (ENTITY-BASED TRADING)
SYSTEM_PROMPT = """You are a self-improving prediction market trader on Kalshi.

## Your Data Source
You trade EXCLUSIVELY on Reddit entity signals. Each signal has been processed through:
1. Reddit post extraction (r/politics, r/news)
2. GLiNER entity recognition (people, organizations)
3. LLM sentiment scoring (-100 to +100)
4. Market-specific price impact transformation

## Key Insight: Sentiment ≠ Price Impact
The transformation depends on market type:
- **OUT markets**: INVERT sentiment (scandal = more likely OUT, so +impact)
- **WIN/CONFIRM/NOMINEE**: PRESERVE sentiment direction

Example:
- "Pam Bondi scandal" → sentiment: -98 (bad for her)
- "BONDI-OUT" market → price_impact: +98 (more likely she's OUT)

## Your Tools
- **get_price_impacts**: Query Reddit entity signals (PRIMARY DATA SOURCE)
- **get_markets**: Check current market prices and spreads
- **get_event_context**: Understand event relationships (CRITICAL for risk)
- **trade**: Execute YES/NO trades
- **get_session_state**: Check your positions and P&L
- **read_memory / write_memory**: Access your learnings

## CRITICAL: Event Awareness & Mutual Exclusivity

Markets within the same event are MUTUALLY EXCLUSIVE - only ONE can resolve YES.

### How Events Work
- Event "KXPRESNOMD" = Democratic Presidential Primary 2028
- Contains 20+ markets: Harris, Biden, Newsom, Shapiro, etc.
- ONLY ONE candidate can win → ONLY ONE market resolves YES
- All other markets resolve NO

### Risk Implications
When you buy NO on a candidate:
- You're betting they WON'T win
- This is CORRELATED with NO bets on other candidates
- If you hold multiple NO positions → CORRELATED RISK

### Risk Levels (for multiple NO positions in same event)
- **ARBITRAGE**: YES prices sum > 105c → NO is cheap, safe to hold all
- **NORMAL**: YES prices sum 100-105c → Fair pricing
- **HIGH_RISK**: YES prices sum 95-99c → NO is getting expensive
- **GUARANTEED_LOSS**: YES prices sum < 95c → You WILL lose money!

### When to Use get_event_context()
Call this tool when:
1. Before trading a market to check event-level exposure
2. When you have multiple positions in related markets
3. When a market's event_ticker looks like a multi-candidate event (e.g., KXPRES*)
4. To understand why the system might block a trade

Example: Before buying NO on KXPRESNOMD-HARRIS, call get_event_context("KXPRESNOMD")
to see all related markets and your current exposure.

## Trading Logic
1. Call get_price_impacts() to see recent signals
2. For high-confidence signals (>0.7), check if price_impact_score suggests edge
3. **IMPORTANT**: Check get_event_context() for the event to assess risk
4. Compare signal direction with current market price
5. If edge exists and within strategy.md limits, trade
6. Record your reasoning

## Signal Interpretation
Each price impact signal tells you:
- **entity**: Who the news is about (e.g., "Pam Bondi")
- **sentiment_score**: How news affects the entity (-100 to +100)
- **price_impact_score**: Expected price movement for THIS market
- **market_type**: OUT, WIN, CONFIRM, NOMINEE
- **confidence**: Signal reliability (0.0 to 1.0)
- **suggested_side**: "yes" or "no" based on impact direction

## The Self-Improvement Loop
You follow the OBSERVE → ACT → REFLECT cycle:

### 1. OBSERVE
- Read your memory files (learnings.md, strategy.md)
- Call get_price_impacts() to see recent signals
- Check current market prices with get_markets()
- **Check event context for markets you're considering**
- Review session state (positions, P&L)

### 2. ACT
Based on signals and strategy:
- **Trade**: If price_impact > 50 with confidence > 0.7
- **Hold**: If no strong signals or markets efficiently priced
- **Exit**: If contradicting signal appears
- **AVOID**: If event exposure is HIGH_RISK or GUARANTEED_LOSS

### 3. REFLECT (after trades settle)
- Did the price_impact_score predict correctly?
- Update learnings.md with the insight
- Adjust strategy.md if patterns emerge

## Memory Files
- **learnings.md**: What you've learned about Reddit signals
- **strategy.md**: Your current trading rules
- **mistakes.md**: Errors to avoid

## Trading Rules
- Only trade signals with confidence > 0.7
- Only trade when |price_impact_score| > 30
- Require spread < 5c for entry
- Max {max_contracts} contracts per trade
- Max {max_positions} positions at once
- **CHECK EVENT CONTEXT before adding NO positions in multi-candidate events**
- Always explain your reasoning before trading

## Important
- TRUST the price_impact_score direction
- Don't second-guess the sentiment→impact transformation
- Focus on signal strength and market conditions
- **Be aware of correlated risk in mutually exclusive markets**
- Learn from every outcome
"""


class SelfImprovingAgent:
    """
    Self-improving trading agent with observe-act-reflect loop.

    The agent runs autonomously, making trading decisions based on
    market data, news, and accumulated learnings. It continuously
    improves by reflecting on trade outcomes.
    """

    def __init__(
        self,
        trading_client: Optional['V3TradingClientIntegration'] = None,
        state_container: Optional['V3StateContainer'] = None,
        websocket_manager: Optional['V3WebSocketManager'] = None,
        config: Optional[DeepAgentConfig] = None,
        tracked_markets: Optional['TrackedMarketsState'] = None,
        event_position_tracker: Optional['EventPositionTracker'] = None,
    ):
        """
        Initialize the self-improving agent.

        Args:
            trading_client: Client for trade execution and market data
            state_container: Container for trading state
            websocket_manager: WebSocket manager for streaming updates
            config: Agent configuration
            tracked_markets: State container for tracked markets (for event context)
            event_position_tracker: Tracker for event-level position risk
        """
        self._config = config or DeepAgentConfig()
        self._trading_client = trading_client
        self._state_container = state_container
        self._ws_manager = websocket_manager
        self._tracked_markets = tracked_markets
        self._event_position_tracker = event_position_tracker

        # Memory directory
        if self._config.memory_dir:
            self._memory_dir = Path(self._config.memory_dir)
        else:
            self._memory_dir = Path(__file__).parent / "memory"

        # Initialize tools
        self._tools = DeepAgentTools(
            trading_client=trading_client,
            state_container=state_container,
            websocket_manager=websocket_manager,
            memory_dir=self._memory_dir,
            tracked_markets=tracked_markets,
            event_position_tracker=event_position_tracker,
        )
        set_global_tools(self._tools)

        # Initialize reflection engine
        self._reflection = ReflectionEngine(
            state_container=state_container,
            websocket_manager=websocket_manager,
            memory_dir=self._memory_dir,
        )
        self._reflection.set_reflection_callback(self._handle_reflection)

        # Initialize Anthropic client
        self._client = AsyncAnthropic()

        # Agent state
        self._running = False
        self._started_at: Optional[float] = None
        self._cycle_count = 0
        self._trades_executed = 0
        self._last_cycle_at: Optional[float] = None

        # Conversation history for context
        self._conversation_history: List[Dict[str, Any]] = []
        self._max_history_length = 20

        # Background task
        self._cycle_task: Optional[asyncio.Task] = None

        # Tool definitions for Claude
        self._tool_definitions = self._build_tool_definitions()

    def _build_tool_definitions(self) -> List[Dict]:
        """Build Claude tool definitions."""
        return [
            {
                "name": "get_price_impacts",
                "description": "Query Reddit entity signals (PRIMARY DATA SOURCE). Returns price impact signals derived from Reddit posts with sentiment transformed for specific markets.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "market_ticker": {
                            "type": "string",
                            "description": "Optional filter by specific market ticker"
                        },
                        "entity_id": {
                            "type": "string",
                            "description": "Optional filter by entity ID (e.g., 'pam_bondi')"
                        },
                        "min_confidence": {
                            "type": "number",
                            "description": "Minimum confidence threshold (default: 0.5)",
                            "default": 0.5
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum signals to return (default: 20)",
                            "default": 20
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_markets",
                "description": "Get current market prices and spreads. Use to check prices for signals from get_price_impacts.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_ticker": {
                            "type": "string",
                            "description": "Optional event ticker to filter markets (e.g., 'KXBONDIOUT')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum markets to return (default: 20)",
                            "default": 20
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_event_context",
                "description": "Get context about an event's markets and mutual exclusivity relationships. CRITICAL for understanding correlated risk when trading multiple markets in the same event (e.g., presidential primary candidates). Returns all markets in the event, YES price sum, risk level, and your current positions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_ticker": {
                            "type": "string",
                            "description": "The event ticker to get context for (e.g., 'KXPRESNOMD' for Democratic Primary)"
                        }
                    },
                    "required": ["event_ticker"]
                }
            },
            {
                "name": "trade",
                "description": "Execute a trade. Only use when you have identified edge.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Market ticker to trade"
                        },
                        "side": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "Buy YES or NO"
                        },
                        "contracts": {
                            "type": "integer",
                            "description": f"Number of contracts (1-{self._config.max_contracts_per_trade})"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Your reasoning for this trade (stored for reflection)"
                        }
                    },
                    "required": ["ticker", "side", "contracts", "reasoning"]
                }
            },
            {
                "name": "get_session_state",
                "description": "Get current trading session state including positions, P&L, and balance.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "read_memory",
                "description": "Read a memory file to access your learnings, strategy, or mistakes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Memory file to read (e.g., 'learnings.md', 'strategy.md', 'mistakes.md')"
                        }
                    },
                    "required": ["filename"]
                }
            },
            {
                "name": "write_memory",
                "description": "Write to a memory file to save learnings, update strategy, or record mistakes.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Memory file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write"
                        }
                    },
                    "required": ["filename", "content"]
                }
            }
        ]

    async def start(self) -> None:
        """Start the agent."""
        if self._running:
            logger.warning("[agent] Already running")
            return

        logger.info("[agent] Starting self-improving agent")
        self._running = True
        self._started_at = time.time()

        # Initialize memory files
        await self._init_memory_files()

        # Start reflection engine
        await self._reflection.start()

        # Start main cycle loop
        self._cycle_task = asyncio.create_task(self._main_loop())

        logger.info("[agent] Agent started successfully")

        # Broadcast status
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_status", {
                "status": "started",
                "config": asdict(self._config),
                "timestamp": time.strftime("%H:%M:%S"),
            })

    async def stop(self) -> None:
        """Stop the agent."""
        if not self._running:
            return

        logger.info("[agent] Stopping self-improving agent")
        self._running = False

        # Cancel cycle task
        if self._cycle_task and not self._cycle_task.done():
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass

        # Stop reflection engine
        await self._reflection.stop()

        logger.info("[agent] Agent stopped")

        # Broadcast status
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_status", {
                "status": "stopped",
                "cycle_count": self._cycle_count,
                "trades_executed": self._trades_executed,
                "timestamp": time.strftime("%H:%M:%S"),
            })

    async def _init_memory_files(self) -> None:
        """Initialize memory files with starter content if they don't exist."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # Initial content for each file
        initial_content = {
            "learnings.md": """# Trading Learnings

This file contains accumulated wisdom from trading experiences.
The agent updates this file after each trade settlement.

## Getting Started
- Add learnings as you trade
- Be specific and actionable
- Reference specific trades and outcomes
""",
            "strategy.md": """# Current Trading Strategy

## Entry Criteria
- Only trade when spread > 3c (indicates inefficiency)
- Require fresh news (<2 hours old) for directional trades
- Whale activity is confirmatory, not primary signal

## Position Sizing
- Max 25 contracts per trade
- Max 5 positions at once
- Never exceed 10% of bankroll per trade

## Exit Rules
- Take profit at 20% gain
- Cut losses at 15% decline
- Exit before settlement if uncertain

## Market Selection
- Focus on events with clear resolution criteria
- Prefer markets with high volume (more liquidity)
- Avoid illiquid markets (spread > 10c)
""",
            "mistakes.md": """# Mistakes to Avoid

This file tracks errors to prevent repeating them.

## Common Mistakes
1. **CHASING**: Don't buy after price moved 10%+ on news
2. **OVERTRADING**: More than 5 trades/hour = too many
3. **STALE NEWS**: News >2 hours old is already priced in
4. **WHALE FOMO**: Whale activity alone is not sufficient edge
5. **NO EXIT PLAN**: Always have exit criteria before entering
""",
            "patterns.md": """# Successful Patterns

This file records patterns that have led to profitable trades.

## Pattern Template
- **Setup**: What conditions were present
- **Trigger**: What prompted entry
- **Outcome**: How the trade performed
- **Why it worked**: The edge explanation
"""
        }

        for filename, content in initial_content.items():
            filepath = self._memory_dir / filename
            if not filepath.exists():
                filepath.write_text(content, encoding="utf-8")
                logger.info(f"[agent] Created initial memory file: {filename}")

    async def _main_loop(self) -> None:
        """Main observe-act-reflect loop."""
        while self._running:
            try:
                # Run one cycle
                await self._run_cycle()

                # Wait for next cycle
                await asyncio.sleep(self._config.cycle_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[agent] Error in main loop: {e}")
                await asyncio.sleep(5.0)  # Brief pause on error

    async def _run_cycle(self) -> None:
        """Run one observe-act cycle."""
        self._cycle_count += 1
        self._last_cycle_at = time.time()

        logger.info(f"[agent] === Starting cycle {self._cycle_count} ===")

        # Broadcast cycle start
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_cycle", {
                "cycle": self._cycle_count,
                "phase": "starting",
                "timestamp": time.strftime("%H:%M:%S"),
            })

        # Build context message for this cycle
        context = await self._build_cycle_context()

        try:
            # Run the agent with streaming
            await self._run_agent_cycle(context)

        except Exception as e:
            logger.error(f"[agent] Error in cycle: {e}")

            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_error", {
                    "error": str(e)[:200],
                    "cycle": self._cycle_count,
                    "timestamp": time.strftime("%H:%M:%S"),
                })

    async def _build_cycle_context(self) -> str:
        """Build the context message for a cycle."""
        # Get current session state
        session_state = await self._tools.get_session_state()

        # Get target events info
        target_events_str = (
            ", ".join(self._config.target_events)
            if self._config.target_events
            else "all available markets"
        )

        # Build event exposure section if we have positions and event tracker
        event_exposure_str = ""
        if session_state.position_count > 0 and self._event_position_tracker:
            event_groups = self._event_position_tracker.get_event_groups()
            # Find groups with positions
            groups_with_positions = [
                g for g in event_groups.values()
                if g.markets_with_positions > 0
            ]

            if groups_with_positions:
                event_lines = ["### Event Exposure (⚠️ Correlated Risk)"]
                for group in groups_with_positions:
                    risk_emoji = {
                        "ARBITRAGE": "✅",
                        "NORMAL": "🟢",
                        "HIGH_RISK": "🟡",
                        "GUARANTEED_LOSS": "🔴",
                    }.get(group.risk_level, "⚪")

                    event_lines.append(
                        f"- **{group.event_ticker}**: {group.markets_with_positions}/{group.market_count} markets, "
                        f"YES_sum={group.yes_sum}c, {risk_emoji} {group.risk_level}"
                    )

                event_exposure_str = "\n" + "\n".join(event_lines) + "\n"

        context = f"""
## New Trading Cycle - {datetime.now().strftime("%Y-%m-%d %H:%M")}

### Current Session State
- Balance: ${session_state.balance_cents / 100:.2f}
- Portfolio Value: ${session_state.portfolio_value_cents / 100:.2f}
- Total P&L: ${session_state.total_pnl_cents / 100:.2f}
- Positions: {session_state.position_count}
- Trade Count: {session_state.trade_count}
- Win Rate: {session_state.win_rate:.1%}
{event_exposure_str}
### Target Events
{target_events_str}

### Instructions
1. First, read your memory files to refresh your learnings and strategy
2. Check current market prices for opportunities
3. **If you have event exposure above, use get_event_context() to understand risk**
4. Search for fresh news if relevant to your targets
5. Decide whether to trade, hold, or exit positions
6. If you trade, provide clear reasoning

Remember: Only trade when you have identified edge. It's okay to hold.
"""
        return context

    async def _run_agent_cycle(self, context: str) -> None:
        """Run the agent with tool calling."""
        # Add context to conversation
        self._conversation_history.append({
            "role": "user",
            "content": context
        })

        # Trim history if too long
        if len(self._conversation_history) > self._max_history_length:
            self._conversation_history = self._conversation_history[-self._max_history_length:]

        # Build system prompt
        system = SYSTEM_PROMPT.format(
            max_contracts=self._config.max_contracts_per_trade,
            max_positions=self._config.max_positions,
        )

        # Track tool calls for this cycle
        tool_calls_this_cycle = 0
        max_tool_calls = 20  # Safety limit

        while tool_calls_this_cycle < max_tool_calls:
            # Call Claude with tools
            response = await self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                system=system,
                messages=self._conversation_history,
                tools=self._tool_definitions,
            )

            # Process response
            assistant_content = []
            tool_use_blocks = []

            for block in response.content:
                if block.type == "text":
                    # Stream thinking to WebSocket
                    if self._ws_manager and block.text.strip():
                        await self._ws_manager.broadcast_message("deep_agent_thinking", {
                            "text": block.text,
                            "cycle": self._cycle_count,
                            "timestamp": time.strftime("%H:%M:%S"),
                        })
                    assistant_content.append({"type": "text", "text": block.text})

                elif block.type == "tool_use":
                    tool_use_blocks.append(block)
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # Add assistant message to history
            self._conversation_history.append({
                "role": "assistant",
                "content": assistant_content
            })

            # If no tool calls, we're done
            if not tool_use_blocks:
                break

            # Execute tool calls
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_calls_this_cycle += 1
                result = await self._execute_tool(tool_block.name, tool_block.input)

                # Stream tool call to WebSocket
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_tool_call", {
                        "tool": tool_block.name,
                        "input": tool_block.input,
                        "output_preview": str(result)[:200],
                        "cycle": self._cycle_count,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                })

            # Add tool results to conversation
            self._conversation_history.append({
                "role": "user",
                "content": tool_results
            })

            # Check if we should stop (response indicates completion)
            if response.stop_reason == "end_turn":
                break

        logger.info(f"[agent] Cycle {self._cycle_count} completed with {tool_calls_this_cycle} tool calls")

    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """Execute a tool and return the result."""
        logger.info(f"[agent] Executing tool: {tool_name}")

        try:
            if tool_name == "get_price_impacts":
                impacts = await self._tools.get_price_impacts(
                    market_ticker=tool_input.get("market_ticker"),
                    entity_id=tool_input.get("entity_id"),
                    min_confidence=tool_input.get("min_confidence", 0.5),
                    limit=tool_input.get("limit", 20),
                )
                return [i.to_dict() for i in impacts]

            elif tool_name == "get_markets":
                markets = await self._tools.get_markets(
                    event_ticker=tool_input.get("event_ticker"),
                    limit=tool_input.get("limit", 20),
                )
                return [m.to_dict() for m in markets]

            elif tool_name == "trade":
                # Validate contracts
                contracts = min(tool_input["contracts"], self._config.max_contracts_per_trade)

                result = await self._tools.trade(
                    ticker=tool_input["ticker"],
                    side=tool_input["side"],
                    contracts=contracts,
                    reasoning=tool_input["reasoning"],
                )

                # Record for reflection if successful
                if result.success:
                    self._trades_executed += 1

                    # Get event_ticker from trading client
                    event_ticker = ""
                    try:
                        if self._trading_client:
                            market = await self._trading_client.get_market(tool_input["ticker"])
                            event_ticker = market.get("event_ticker", "") if market else ""
                    except Exception as e:
                        logger.warning(f"[agent] Could not fetch event_ticker: {e}")

                    self._reflection.record_trade(
                        trade_id=result.order_id or str(uuid.uuid4()),
                        ticker=tool_input["ticker"],
                        event_ticker=event_ticker,
                        side=tool_input["side"],
                        contracts=contracts,
                        entry_price_cents=result.price_cents or 50,
                        reasoning=tool_input["reasoning"],
                    )

                return result.to_dict()

            elif tool_name == "get_session_state":
                state = await self._tools.get_session_state()
                return state.to_dict()

            elif tool_name == "read_memory":
                content = await self._tools.read_memory(tool_input["filename"])
                return content

            elif tool_name == "write_memory":
                success = await self._tools.write_memory(
                    tool_input["filename"],
                    tool_input["content"],
                )
                return {"success": success}

            elif tool_name == "get_event_context":
                context = await self._tools.get_event_context(
                    event_ticker=tool_input["event_ticker"],
                )
                if context:
                    return context.to_dict()
                else:
                    return {"error": f"Event {tool_input['event_ticker']} not found or no tracked markets"}

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"[agent] Error executing {tool_name}: {e}")
            return {"error": str(e)}

    async def _handle_reflection(self, trade: PendingTrade) -> Optional[ReflectionResult]:
        """Handle reflection callback from the reflection engine."""
        logger.info(f"[agent] Reflecting on trade: {trade.ticker} ({trade.result})")

        # Generate reflection prompt
        prompt = self._reflection.generate_reflection_prompt(trade)

        # Add to conversation and run
        self._conversation_history.append({
            "role": "user",
            "content": prompt
        })

        try:
            # Run reflection cycle (similar to main cycle)
            await self._run_agent_cycle("")

            # Extract learning from the agent's response
            learning = self._extract_learning_from_response()
            mistake = self._extract_mistake_from_response()
            pattern = self._extract_pattern_from_response()

            return ReflectionResult(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                result=trade.result or "unknown",
                pnl_cents=trade.pnl_cents or 0,
                learning=learning,
                should_update_strategy=bool(pattern),
                mistake_identified=mistake,
                pattern_identified=pattern,
            )

        except Exception as e:
            logger.error(f"[agent] Error in reflection: {e}")
            return None

    def _extract_learning_from_response(self) -> str:
        """Extract learning insight from the last assistant message."""
        for msg in reversed(self._conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Handle list of content blocks (from tool calling)
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            return self._extract_insight_from_text(text, "learn")
                elif isinstance(content, str):
                    return self._extract_insight_from_text(content, "learn")
        return "Reflected on trade outcome"

    def _extract_mistake_from_response(self) -> Optional[str]:
        """Extract mistake identification from the last assistant message."""
        for msg in reversed(self._conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            insight = self._extract_insight_from_text(text, "mistake")
                            if insight != "Reflected on trade outcome":
                                return insight
                elif isinstance(content, str):
                    insight = self._extract_insight_from_text(content, "mistake")
                    if insight != "Reflected on trade outcome":
                        return insight
        return None

    def _extract_pattern_from_response(self) -> Optional[str]:
        """Extract pattern identification from the last assistant message."""
        for msg in reversed(self._conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            insight = self._extract_insight_from_text(text, "pattern")
                            if insight != "Reflected on trade outcome":
                                return insight
                elif isinstance(content, str):
                    insight = self._extract_insight_from_text(content, "pattern")
                    if insight != "Reflected on trade outcome":
                        return insight
        return None

    def _extract_insight_from_text(self, text: str, insight_type: str) -> str:
        """
        Extract a specific type of insight from text.

        Args:
            text: The text to search
            insight_type: Type of insight ("learn", "mistake", "pattern")

        Returns:
            Extracted insight or default message
        """
        if not text or not text.strip():
            return "Reflected on trade outcome"

        text_lower = text.lower()

        # Keywords by insight type
        keywords = {
            "learn": ["learned", "learning", "insight", "takeaway", "realized", "understand now"],
            "mistake": ["mistake", "error", "wrong", "shouldn't have", "should not have", "failed to"],
            "pattern": ["pattern", "when", "if", "tends to", "usually", "consistently", "signal"],
        }

        # Look for sentences containing keywords
        sentences = text.replace("\n", " ").split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if any(kw in sentence_lower for kw in keywords.get(insight_type, [])):
                # Clean and truncate the sentence
                clean = sentence.strip()
                if len(clean) > 200:
                    clean = clean[:197] + "..."
                if clean:
                    return clean

        # Fallback: return first meaningful sentence
        for sentence in sentences:
            clean = sentence.strip()
            if len(clean) > 20:  # Ignore very short fragments
                if len(clean) > 200:
                    clean = clean[:197] + "..."
                return clean

        return "Reflected on trade outcome"

    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running

    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        if not self._running:
            return False

        # Check if cycle is happening
        if self._last_cycle_at:
            time_since_cycle = time.time() - self._last_cycle_at
            if time_since_cycle > self._config.cycle_interval_seconds * 3:
                return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "running": self._running,
            "healthy": self.is_healthy(),
            "uptime_seconds": uptime,
            "cycle_count": self._cycle_count,
            "trades_executed": self._trades_executed,
            "last_cycle_at": self._last_cycle_at,
            "config": asdict(self._config),
            "tool_stats": self._tools.get_tool_stats(),
            "reflection_stats": self._reflection.get_stats(),
            "pending_trades": len(self._reflection.get_pending_trades()),
        }

    def get_consolidated_view(self) -> Dict[str, Any]:
        """Get consolidated view for frontend display."""
        stats = self.get_stats()

        return {
            "status": "active" if self._running else "stopped",
            "cycle_count": self._cycle_count,
            "trades_executed": self._trades_executed,
            "win_rate": stats["reflection_stats"]["win_rate"],
            "pending_trades": self._reflection.get_pending_trades(),
            "recent_reflections": self._reflection.get_recent_reflections(5),
            "tool_stats": stats["tool_stats"],
            "memory_dir": str(self._memory_dir),
            "target_events": self._config.target_events,
        }
