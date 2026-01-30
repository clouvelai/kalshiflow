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
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from anthropic import AsyncAnthropic

from .tools import DeepAgentTools, set_global_tools
from .reflection import ReflectionEngine, PendingTrade, ReflectionResult
from .prompts import build_system_prompt
from .signal_tracker import SignalEvaluationTracker

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
    max_tokens: int = 2048

    # Loop settings
    cycle_interval_seconds: float = 180.0  # How often to run observe-act cycle (3 minutes)
    max_trades_per_cycle: int = 3

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
    max_contracts_per_trade: int = 100  # Hard cap per single trade
    max_event_exposure_cents: int = 10000  # $100 per-event dollar exposure cap
    min_spread_cents: int = 3  # Only trade if spread > this (inefficiency)
    require_fresh_news: bool = True
    max_news_age_hours: float = 2.0

    # Circuit breaker settings (prevents repeated failures on same market)
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 3  # Failures before blacklisting
    circuit_breaker_window_seconds: float = 3600.0  # 1 hour window for failures
    circuit_breaker_cooldown_seconds: float = 1800.0  # 30 min blacklist duration

    # Signal lifecycle settings
    max_eval_cycles_per_signal: int = 3  # Max evaluations before auto-expire

    # Prompt settings (experimental)
    include_enders_game: bool = True  # Toggle Ender's Game framing for A/B testing


# System prompt is now built dynamically via build_system_prompt() from prompts.py
# This allows modular editing and A/B testing of prompt sections


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

        # Initialize signal lifecycle tracker
        self._signal_tracker = SignalEvaluationTracker(
            max_eval_cycles=self._config.max_eval_cycles_per_signal,
        )

        # Initialize tools
        self._tools = DeepAgentTools(
            trading_client=trading_client,
            state_container=state_container,
            websocket_manager=websocket_manager,
            memory_dir=self._memory_dir,
            tracked_markets=tracked_markets,
            event_position_tracker=event_position_tracker,
            signal_tracker=self._signal_tracker,
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
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. The deep agent requires Claude API access. "
                "Set ANTHROPIC_API_KEY in your .env.paper file."
            )
        logger.info("[deep_agent.agent] ANTHROPIC_API_KEY found, Claude API ready")
        self._client = AsyncAnthropic()

        # Agent state
        self._running = False
        self._started_at: Optional[float] = None
        self._cycle_count = 0
        self._trades_executed = 0
        self._last_cycle_at: Optional[float] = None

        # Conversation history for context
        self._conversation_history: List[Dict[str, Any]] = []
        self._max_history_length = 12  # ~2 cycles of history (saves tokens)

        # Background task
        self._cycle_task: Optional[asyncio.Task] = None

        # History buffers for session persistence (replay to new WebSocket clients)
        self._thinking_history: deque = deque(maxlen=10)
        self._tool_call_history: deque = deque(maxlen=50)
        self._trade_history: deque = deque(maxlen=50)
        self._learnings_history: deque = deque(maxlen=50)

        # Circuit breaker state (prevents repeated failures on same market)
        # Maps ticker -> list of failure timestamps
        self._failed_attempts: Dict[str, List[float]] = {}
        # Maps ticker -> blacklist expiry timestamp
        self._blacklisted_tickers: Dict[str, float] = {}

        # Think enforcement tracking (soft enforcement - warn if trade without think)
        self._last_think_decision: Optional[str] = None
        self._last_think_timestamp: Optional[float] = None

        # Token usage tracking for prompt caching metrics
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_created_tokens = 0

        # Reflection count for consolidation trigger
        self._reflection_count = 0
        self._consecutive_losses = 0  # Track loss streaks for urgent consolidation

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
                "description": "Execute a trade. Choose execution_strategy based on signal quality: aggressive for STRONG signals (crosses spread, fastest fill), moderate for MODERATE signals (midpoint pricing, saves spread), passive for speculative entries (near bid, cheapest but may not fill).",
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
                        },
                        "execution_strategy": {
                            "type": "string",
                            "enum": ["aggressive", "moderate", "passive"],
                            "description": "Order pricing strategy. aggressive=cross spread (immediate fill, highest cost). moderate=midpoint+1c (saves ~half the spread, good fill probability). passive=near bid+1c (cheapest entry, may not fill). Default: aggressive."
                        },
                        "signal_id": {
                            "type": "string",
                            "description": "The signal_id of the price impact being traded on (from get_price_impacts). Always include this."
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
                "description": "FULL REPLACE a memory file. Use ONLY for strategy.md rewrites where you need to restructure the entire file. For adding learnings/mistakes/patterns, use append_memory instead.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Memory file to write"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write (REPLACES entire file)"
                        }
                    },
                    "required": ["filename", "content"]
                }
            },
            {
                "name": "append_memory",
                "description": "APPEND to a memory file (safe, no data loss). Use for learnings.md, mistakes.md, and patterns.md. Automatically archives old content if file exceeds 10,000 chars.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Memory file to append to (e.g., 'learnings.md', 'mistakes.md', 'patterns.md')"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to append (added to end of file)"
                        }
                    },
                    "required": ["filename", "content"]
                }
            },
            {
                "name": "think",
                "description": "REQUIRED before every trade. Structured pre-trade analysis that forces consideration of key factors. Must be called before trade() - trades without prior think() will be flagged.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "signal_analysis": {
                            "type": "string",
                            "description": "What signal are you acting on? (entity, impact score, confidence, source)"
                        },
                        "strategy_check": {
                            "type": "string",
                            "description": "Does this trade align with your strategy.md rules? Which rules apply?"
                        },
                        "risk_assessment": {
                            "type": "string",
                            "description": "What are the risks? (spread, position limits, event exposure, market liquidity)"
                        },
                        "decision": {
                            "type": "string",
                            "enum": ["TRADE", "WAIT", "PASS"],
                            "description": "Your decision: TRADE (execute now), WAIT (wait for corroborating signal/trending validation), PASS (no edge)"
                        },
                        "reflection": {
                            "type": "string",
                            "description": "Optional: Any additional reasoning or notes for future learning"
                        },
                        "signal_id": {
                            "type": "string",
                            "description": "The signal_id of the price impact being evaluated (from get_price_impacts). Always include this."
                        }
                    },
                    "required": ["signal_analysis", "strategy_check", "risk_assessment", "decision"]
                }
            },
            {
                "name": "assess_trade_opportunity",
                "description": "Quantitative edge and risk/reward assessment. Call AFTER get_price_impacts() with signal data. Returns: expected edge (cents), signal-implied fair price, priced-in verdict (with 24h candlestick history when available), risk/reward ratio, and overall trade quality (STRONG/MODERATE/WEAK/AVOID). Includes current bid/ask so you can skip a separate get_markets() call.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "market_ticker": {
                            "type": "string",
                            "description": "Market ticker to assess (e.g., 'KXGOVSHUT-26JAN31')"
                        },
                        "signal_impact_score": {
                            "type": "integer",
                            "description": "Price impact score from get_price_impacts() (-75 to +75)"
                        },
                        "signal_confidence": {
                            "type": "number",
                            "description": "Signal confidence from get_price_impacts() (0.5, 0.7, or 0.9)"
                        },
                        "suggested_side": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "Suggested side from get_price_impacts()"
                        }
                    },
                    "required": ["market_ticker", "signal_impact_score", "signal_confidence", "suggested_side"]
                }
            },
            {
                "name": "reflect",
                "description": "Structured post-trade reflection. Use after a trade settles to record learnings. Auto-appends to appropriate memory files (no data loss). Preferred over manual write_memory calls for reflections.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "trade_ticker": {
                            "type": "string",
                            "description": "Market ticker of the settled trade"
                        },
                        "outcome_analysis": {
                            "type": "string",
                            "description": "Why did this trade win/lose? What was the key factor?"
                        },
                        "reasoning_accuracy": {
                            "type": "string",
                            "enum": ["correct", "partially_correct", "wrong"],
                            "description": "Was your original reasoning accurate?"
                        },
                        "key_learning": {
                            "type": "string",
                            "description": "The main insight from this trade. Be specific and actionable."
                        },
                        "mistake": {
                            "type": "string",
                            "description": "Optional: A clear error or anti-pattern to avoid next time"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Optional: A repeatable winning setup discovered"
                        },
                        "strategy_update_needed": {
                            "type": "boolean",
                            "description": "Should strategy.md be updated based on this learning?"
                        },
                        "confidence_in_learning": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "How confident are you in this learning? High = clear evidence, Low = speculative"
                        }
                    },
                    "required": ["trade_ticker", "outcome_analysis", "reasoning_accuracy", "key_learning", "strategy_update_needed", "confidence_in_learning"]
                }
            }
        ]

    # === Circuit Breaker Methods ===

    def _clean_expired_failures(self) -> None:
        """Remove expired failure records and blacklist entries."""
        now = time.time()

        # Clean expired blacklist entries
        expired_blacklist = [
            ticker for ticker, expiry in self._blacklisted_tickers.items()
            if now >= expiry
        ]
        for ticker in expired_blacklist:
            del self._blacklisted_tickers[ticker]
            logger.info(f"[deep_agent.circuit_breaker] {ticker} removed from blacklist (cooldown expired)")

        # Clean old failure records
        window = self._config.circuit_breaker_window_seconds
        for ticker in list(self._failed_attempts.keys()):
            self._failed_attempts[ticker] = [
                ts for ts in self._failed_attempts[ticker]
                if now - ts < window
            ]
            if not self._failed_attempts[ticker]:
                del self._failed_attempts[ticker]

    def _is_ticker_blacklisted(self, ticker: str) -> tuple[bool, Optional[str]]:
        """
        Check if a ticker is currently blacklisted.

        Returns:
            Tuple of (is_blacklisted, reason_message)
        """
        if not self._config.circuit_breaker_enabled:
            return False, None

        self._clean_expired_failures()

        if ticker in self._blacklisted_tickers:
            expiry = self._blacklisted_tickers[ticker]
            remaining = int(expiry - time.time())
            reason = (
                f"Market {ticker} is temporarily blacklisted after "
                f"{self._config.circuit_breaker_threshold} consecutive failures. "
                f"Cooldown: {remaining}s remaining. "
                f"This prevents wasting cycles on illiquid/impossible trades."
            )
            return True, reason

        return False, None

    def _record_trade_failure(self, ticker: str, error: str) -> bool:
        """
        Record a trade failure for circuit breaker tracking.

        Args:
            ticker: The market ticker that failed
            error: The error message

        Returns:
            True if ticker was blacklisted as a result
        """
        if not self._config.circuit_breaker_enabled:
            return False

        now = time.time()

        # Record the failure
        if ticker not in self._failed_attempts:
            self._failed_attempts[ticker] = []
        self._failed_attempts[ticker].append(now)

        # Clean old failures
        self._clean_expired_failures()

        # Check if threshold reached
        failure_count = len(self._failed_attempts.get(ticker, []))
        threshold = self._config.circuit_breaker_threshold

        if failure_count >= threshold:
            # Blacklist the ticker
            cooldown = self._config.circuit_breaker_cooldown_seconds
            self._blacklisted_tickers[ticker] = now + cooldown

            logger.warning(
                f"[deep_agent.circuit_breaker] {ticker} BLACKLISTED after {failure_count} failures. "
                f"Cooldown: {cooldown}s. Error: {error[:100]}"
            )

            # Clear failure records for this ticker (we've already blacklisted)
            self._failed_attempts.pop(ticker, None)

            return True

        logger.info(
            f"[deep_agent.circuit_breaker] {ticker} failure {failure_count}/{threshold}: {error[:100]}"
        )
        return False

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status for debugging/monitoring."""
        self._clean_expired_failures()
        now = time.time()

        return {
            "enabled": self._config.circuit_breaker_enabled,
            "threshold": self._config.circuit_breaker_threshold,
            "window_seconds": self._config.circuit_breaker_window_seconds,
            "cooldown_seconds": self._config.circuit_breaker_cooldown_seconds,
            "blacklisted_tickers": {
                ticker: int(expiry - now)
                for ticker, expiry in self._blacklisted_tickers.items()
            },
            "pending_failures": {
                ticker: len(failures)
                for ticker, failures in self._failed_attempts.items()
            },
        }

    async def start(self) -> None:
        """Start the agent."""
        if self._running:
            logger.warning("[deep_agent.agent] Already running")
            return

        logger.info("[deep_agent.agent] start() called, ws_manager present: %s", self._ws_manager is not None)
        logger.info("[deep_agent.agent] Starting self-improving agent")
        self._running = True
        self._started_at = time.time()

        # Clear history buffers for fresh state on restart
        self._thinking_history.clear()
        self._tool_call_history.clear()
        self._trade_history.clear()
        self._learnings_history.clear()
        self._conversation_history.clear()
        self._cycle_count = 0
        self._trades_executed = 0

        # Reset token usage counters
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_created_tokens = 0

        logger.info("[deep_agent.agent] Cleared history buffers for fresh session")

        # Validate Supabase connectivity (warn only — not fatal)
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_ANON_KEY")
        if not supabase_url or not supabase_key:
            logger.warning(
                "[deep_agent.agent] SUPABASE_URL or SUPABASE_KEY not set. "
                "Price impact signals will be unavailable (agent will see no trading signals)."
            )

        # Initialize memory files
        await self._init_memory_files()
        logger.info("[deep_agent.agent] Memory files initialized")

        # Reset reflection count and loss streak for consolidation trigger
        self._reflection_count = 0
        self._consecutive_losses = 0

        # Start reflection engine
        await self._reflection.start()
        logger.info("[deep_agent.agent] Reflection engine started")

        # Start main cycle loop
        self._cycle_task = asyncio.create_task(self._main_loop())
        logger.info("[deep_agent.agent] Main loop task created: %s", self._cycle_task)

        logger.info("[deep_agent.agent] Agent started successfully")

        # Broadcast status
        logger.info("[deep_agent.agent] About to broadcast status, ws_manager: %s", self._ws_manager)
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_status", {
                "status": "started",
                "config": asdict(self._config),
                "timestamp": time.strftime("%H:%M:%S"),
            })
            logger.info("[deep_agent.agent] Broadcast deep_agent_status with status='started'")
        else:
            logger.warning("[deep_agent.agent] No ws_manager - cannot broadcast status to frontend!")

    async def stop(self) -> None:
        """Stop the agent with end-of-session summary and memory archival."""
        if not self._running:
            return

        logger.info("[deep_agent.agent] Stopping self-improving agent")
        self._running = False

        # Cancel cycle task
        if self._cycle_task and not self._cycle_task.done():
            self._cycle_task.cancel()
            try:
                await self._cycle_task
            except asyncio.CancelledError:
                pass

        # Generate end-of-session summary (with timeout to prevent blocking shutdown)
        try:
            await asyncio.wait_for(self._generate_session_summary(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("[deep_agent.agent] Session summary timed out after 60s, skipping")
        except Exception as e:
            logger.error(f"[deep_agent.agent] Error generating session summary: {e}")

        # Archive memory files for this session
        try:
            self._archive_session_memory()
        except Exception as e:
            logger.error(f"[deep_agent.agent] Error archiving memory: {e}")

        # Stop reflection engine
        await self._reflection.stop()

        logger.info("[deep_agent.agent] Agent stopped")

        # Broadcast status
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_status", {
                "status": "stopped",
                "cycle_count": self._cycle_count,
                "trades_executed": self._trades_executed,
                "timestamp": time.strftime("%H:%M:%S"),
            })

    async def _generate_session_summary(self) -> None:
        """
        Generate an end-of-session summary before shutdown.

        Runs a final reflection cycle that reviews the full session:
        - Overall performance stats
        - Strategy effectiveness
        - Session-level patterns
        - Recommendations for next session
        """
        if self._cycle_count == 0 and self._trades_executed == 0:
            logger.info("[deep_agent.agent] No cycles/trades in session, skipping summary")
            return

        logger.info("[deep_agent.agent] Generating end-of-session summary...")

        # Build session stats
        stats = self._reflection.get_stats()
        uptime_seconds = time.time() - self._started_at if self._started_at else 0
        uptime_minutes = uptime_seconds / 60

        session_state = await self._tools.get_session_state()

        summary_prompt = f"""## End-of-Session Summary

Your trading session is ending. Review your performance and capture session-level insights.

### Session Stats
- **Duration**: {uptime_minutes:.0f} minutes
- **Cycles Run**: {self._cycle_count}
- **Trades Executed**: {self._trades_executed}
- **Win Rate**: {stats.get('win_rate', 0):.0%} ({stats.get('wins', 0)}W/{stats.get('losses', 0)}L)
- **Total P&L**: ${session_state.total_pnl_cents / 100:.2f}
- **Reflections**: {stats.get('total_reflections', 0)}
- **Strategy Updates**: {stats.get('strategy_updates', 0)}
- **Mistakes Found**: {stats.get('mistakes_identified', 0)}

### Instructions
1. Call `read_memory("strategy.md")` to review your current strategy
2. Consider: What worked this session? What didn't?
3. Are there session-level patterns not captured in individual trade reflections?
4. Call `append_memory("learnings.md", ...)` with session-level insights
5. If strategy needs updating, call `write_memory("strategy.md", ...)` with improvements
6. Keep it concise - focus on the most impactful learnings

This is your last chance to capture insights before the session ends.
"""

        self._conversation_history.append({
            "role": "user",
            "content": summary_prompt
        })

        try:
            # Temporarily re-enable running for this final cycle
            self._running = True
            await self._run_agent_cycle("")
            self._running = False
            logger.info("[deep_agent.agent] Session summary completed")
        except Exception as e:
            self._running = False
            logger.error(f"[deep_agent.agent] Error in session summary: {e}")

    def _archive_session_memory(self) -> None:
        """
        Archive all memory files for this session.

        Creates a timestamped copy of all memory files in memory_archive/
        to preserve the state at session end. This provides a historical
        record of how memory evolved across sessions.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        archive_dir = self._memory_dir / "memory_archive" / timestamp
        archive_dir.mkdir(parents=True, exist_ok=True)

        archived_count = 0
        for filename in self._config.memory_files:
            source = self._memory_dir / filename
            if source.exists():
                dest = archive_dir / filename
                dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                archived_count += 1

        logger.info(
            f"[deep_agent.agent] Archived {archived_count} memory files to {archive_dir}"
        )

    async def _init_memory_files(self) -> None:
        """Initialize memory files with minimal starter content if they don't exist."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)

        # Minimal templates - agent builds its own learnings from scratch
        # The system prompt already explains how to use each file
        initial_content = {
            "learnings.md": """# Trading Learnings

Trade-by-trade insights are recorded here.
""",
            "strategy.md": """# Trading Strategy

## Entry Rules
- Use assess_trade_opportunity() as your quantitative backbone — it returns STRONG/MODERATE/WEAK/AVOID
- Develop specific entry criteria through experience and record them here
- Every trade is data: track what works, discard what doesn't

## Position Sizing
- **$100 max exposure per event** (cost = contracts x price_in_cents / 100)
- Scale contracts based on conviction: stronger signal + higher confidence = larger position
- Always calculate dollar cost before placing a trade

## Risk Rules
- Check event context before correlated trades (get_event_context)
- Never exceed $100 total cost basis on any single event
- Watch for mutual exclusivity risk in multi-candidate events

## Exit Rules
- (Develop through experience)
""",
            "mistakes.md": """# Mistakes to Avoid

Errors and anti-patterns go here.
""",
            "patterns.md": """# Winning Patterns

Successful patterns go here.
"""
        }

        for filename, content in initial_content.items():
            filepath = self._memory_dir / filename
            if not filepath.exists():
                filepath.write_text(content, encoding="utf-8")
                logger.info(f"[deep_agent.agent] Created initial memory file: {filename}")

    async def _main_loop(self) -> None:
        """Main observe-act-reflect loop."""
        logger.info("[deep_agent.agent] _main_loop() STARTED - will run first cycle now")

        # Broadcast "active" status when main loop actually starts executing
        # This ensures clients connecting after startup see the correct status
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_status", {
                "status": "active",
                "cycle_count": self._cycle_count,
                "config": asdict(self._config),
                "timestamp": time.strftime("%H:%M:%S"),
            })
            logger.info("[deep_agent.agent] Broadcast deep_agent_status with status='active' (main loop running)")

        while self._running:
            try:
                logger.info("[deep_agent.agent] About to run cycle %d", self._cycle_count + 1)
                # Run one cycle
                await self._run_cycle()
                logger.info(
                    "[deep_agent.agent] Cycle %d completed, sleeping %ss",
                    self._cycle_count,
                    self._config.cycle_interval_seconds
                )

                # Wait for next cycle
                await asyncio.sleep(self._config.cycle_interval_seconds)

            except asyncio.CancelledError:
                logger.info("[deep_agent.agent] _main_loop() cancelled")
                break
            except Exception as e:
                logger.error("[deep_agent.agent] Error in main loop: %s", e, exc_info=True)
                await asyncio.sleep(5.0)  # Brief pause on error
        logger.info("[deep_agent.agent] _main_loop() STOPPED")

    async def _run_cycle(self) -> None:
        """Run one observe-act cycle."""
        self._cycle_count += 1
        self._last_cycle_at = time.time()

        logger.info(f"[deep_agent.agent] === Starting cycle {self._cycle_count} ===")

        # Broadcast cycle start
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_cycle", {
                "cycle": self._cycle_count,
                "phase": "starting",
                "timestamp": time.strftime("%H:%M:%S"),
            })

        # PRE-CHECK: Are there actionable signals? (cheap Supabase query, ~50ms, zero API tokens)
        pre_fetched_signals = await self._tools.get_price_impacts(
            min_confidence=0.5, min_impact_magnitude=30, limit=20
        )

        # Also check if we have open positions to monitor (always run cycle if positions exist)
        has_positions = (
            self._state_container
            and self._state_container.get_trading_summary().get("position_count", 0) > 0
        )

        if not pre_fetched_signals and not has_positions:
            logger.info(f"[deep_agent.agent] Cycle {self._cycle_count}: No signals, no positions - skipping Claude call")
            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_cycle", {
                    "cycle": self._cycle_count,
                    "phase": "skipped_no_signals",
                    "timestamp": time.strftime("%H:%M:%S"),
                })
            return

        # Signals exist OR we have positions to manage — run full cycle
        context = await self._build_cycle_context(pre_fetched_signals=pre_fetched_signals)

        try:
            # Run the agent with streaming
            await self._run_agent_cycle(context)

        except Exception as e:
            logger.error(f"[deep_agent.agent] Error in cycle: {e}")

            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_error", {
                    "error": str(e)[:200],
                    "cycle": self._cycle_count,
                    "timestamp": time.strftime("%H:%M:%S"),
                })

        # Compact history between cycles to save tokens on next call
        self._compact_history_between_cycles()

    def _compact_history_between_cycles(self):
        """Compact tool results between cycles to save tokens.

        Replaces large JSON tool results with truncated summaries.
        Preserves assistant reasoning text and tool call names/inputs.
        """
        for msg in self._conversation_history:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    result_text = block.get("content", "")
                    if isinstance(result_text, str) and len(result_text) > 500:
                        block["content"] = result_text[:200] + "... [compacted for efficiency]"

    async def _build_cycle_context(self, pre_fetched_signals=None) -> str:
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
                event_lines = ["### Event Exposure (Correlated Risk)"]
                for group in groups_with_positions:
                    risk_emoji = {
                        "ARBITRAGE": "+",
                        "NORMAL": "o",
                        "HIGH_RISK": "!",
                        "GUARANTEED_LOSS": "X",
                    }.get(group.risk_level, "?")

                    event_lines.append(
                        f"- **{group.event_ticker}**: {group.markets_with_positions}/{group.market_count} markets, "
                        f"YES_sum={group.yes_sum}c, [{risk_emoji}] {group.risk_level}"
                    )

                event_exposure_str = "\n" + "\n".join(event_lines) + "\n"

        # Load strategy (full content up to 2000 chars for complete visibility)
        strategy_excerpt = ""
        try:
            strategy = await self._tools.read_memory("strategy.md")
            if strategy and len(strategy) > 50:
                strategy_excerpt = f"\n### Your Current Strategy Rules\n{strategy[:2000]}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load strategy: {e}")

        # Load mistakes and patterns EVERY cycle (not just first 3)
        # Char limits scale down after cycle 10 to save tokens on mature sessions
        memory_context_str = ""
        try:
            mistakes = await self._tools.read_memory("mistakes.md")
            if mistakes and len(mistakes) > 50:
                mistakes_limit = 800 if self._cycle_count <= 10 else 400
                mistakes_preview = mistakes[:mistakes_limit]
                memory_context_str += f"\n### Mistakes to Avoid (from memory)\n{mistakes_preview}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load mistakes: {e}")

        try:
            patterns = await self._tools.read_memory("patterns.md")
            if patterns and len(patterns) > 50:
                patterns_limit = 800 if self._cycle_count <= 10 else 400
                patterns_preview = patterns[:patterns_limit]
                memory_context_str += f"\n### Winning Patterns (from memory)\n{patterns_preview}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load patterns: {e}")

        # Build open positions mark-to-market section (deep_agent positions only)
        # Use pending trades as source of truth for which positions belong to this agent,
        # then pull position-level P&L from state_container's positions_details.
        open_positions_str = ""
        agent_tickers = {
            trade.ticker for trade in self._reflection._pending_trades.values()
        }
        if agent_tickers and self._state_container:
            try:
                summary = self._state_container.get_trading_summary()
                all_positions = summary.get("positions_details", [])
                agent_positions = [
                    p for p in all_positions
                    if p.get("ticker") in agent_tickers
                ]

                if agent_positions:
                    # Sort by absolute unrealized P&L (largest impact first)
                    agent_positions.sort(
                        key=lambda p: abs(p.get("unrealized_pnl", 0)),
                        reverse=True,
                    )

                    pos_lines = ["### Open Positions (Mark-to-Market)"]
                    winning = 0
                    losing = 0
                    total_unrealized = 0

                    for pos in agent_positions:
                        ticker = pos.get("ticker", "???")
                        side = pos.get("side", "?").upper()
                        qty = abs(pos.get("position", 0))
                        total_cost = pos.get("total_cost", 0)
                        cost_per = total_cost // qty if qty > 0 else 0
                        current_value = pos.get("current_value", 0)
                        current_per = current_value // qty if qty > 0 else 0
                        unrealized = pos.get("unrealized_pnl", 0)
                        total_unrealized += unrealized
                        pnl_dollars = unrealized / 100
                        pct_change = (
                            (unrealized / total_cost) * 100
                            if total_cost > 0 else 0.0
                        )
                        sign = "+" if pnl_dollars >= 0 else ""
                        pos_lines.append(
                            f"- {ticker} {side} x{qty} @ {cost_per}c -> now {current_per}c "
                            f"({sign}${pnl_dollars:.2f}, {sign}{pct_change:.1f}%)"
                        )
                        if unrealized >= 0:
                            winning += 1
                        else:
                            losing += 1

                    sign = "+" if total_unrealized >= 0 else ""
                    pos_lines.append(
                        f"Unrealized P&L: {sign}${total_unrealized / 100:.2f} across "
                        f"{len(agent_positions)} position(s) "
                        f"({winning} winning, {losing} losing)"
                    )
                    open_positions_str = "\n" + "\n".join(pos_lines) + "\n"
            except Exception as e:
                logger.debug(f"[deep_agent.agent] Could not build open positions section: {e}")

        # Build performance scorecard every 10 cycles (~30 min at 3-min interval)
        scorecard_str = ""
        if self._cycle_count % 10 == 0:
            try:
                scorecard = self._reflection._build_performance_scorecard()
                if scorecard:
                    scorecard_str = f"\n{scorecard}\n"
            except Exception as e:
                logger.debug(f"[deep_agent.agent] Could not build scorecard: {e}")

        # Build pre-loaded signals section
        signals_section = ""
        if pre_fetched_signals:
            signals_lines = ["\n### Active Price Impact Signals (pre-loaded)"]
            for s in pre_fetched_signals:
                signals_lines.append(
                    f"- **{s.market_ticker}** | {s.entity_name} | "
                    f"impact={s.price_impact_score} conf={s.confidence} "
                    f"side={s.suggested_side} | spread={s.market_spread}c "
                    f"| signal_id={s.signal_id} | source={s.source_domain or 'reddit'}"
                )
            signals_lines.append(
                "\nSignals above are pre-loaded. Proceed directly to assess_trade_opportunity() for promising ones."
            )
            signals_section = "\n".join(signals_lines) + "\n"

            instructions = """### Instructions
1. **Signals are provided above** — review them for tradeable opportunities
2. For promising signals, call **assess_trade_opportunity()** to get a calibrated quality rating (STRONG/MODERATE/WEAK/AVOID)
3. Use the quality rating + your strategy.md criteria to decide: TRADE, WAIT, or PASS
4. **If you have event exposure above, use get_event_context() to understand risk**
5. If your criteria are met: think() → trade() — execute decisively
6. If no actionable signals, PASS and wait for next cycle
7. You can still call get_price_impacts() to re-query with different filters if needed

**IMPORTANT**: Use assess_trade_opportunity() to evaluate signals quantitatively. Apply your learned criteria from strategy.md — refine them after every trade."""
        else:
            signals_section = "\n### No new signals this cycle. Focus on managing existing positions.\n"
            instructions = """### Instructions
1. No new signals available — focus on managing existing positions
2. Check open positions for exit opportunities or risk management
3. You can call get_price_impacts() to re-query with different filters if needed
4. If nothing actionable, PASS and wait for next cycle"""

        context = f"""
## New Trading Cycle - {datetime.now().strftime("%Y-%m-%d %H:%M")}

### Current Session State
- Balance: ${session_state.balance_cents / 100:.2f}
- Portfolio Value: ${session_state.portfolio_value_cents / 100:.2f}
- Total P&L: ${session_state.total_pnl_cents / 100:.2f}
- Positions: {session_state.position_count}
- Trade Count: {session_state.trade_count}
- Win Rate: {session_state.win_rate:.1%}
{event_exposure_str}{open_positions_str}{strategy_excerpt}{memory_context_str}{scorecard_str}{signals_section}
### Target Events
{target_events_str}

{instructions}
"""
        return context

    async def _run_agent_cycle(self, context: str) -> None:
        """Run the agent with tool calling."""
        # Add context to conversation (only if we have content to add)
        if context:
            self._conversation_history.append({
                "role": "user",
                "content": context
            })

        # Trim history if too long - keep first 2 messages (session init) + last N
        if len(self._conversation_history) > self._max_history_length:
            keep_prefix = 2  # First 2 messages contain session setup context
            keep_suffix = self._max_history_length - keep_prefix
            dropped_count = len(self._conversation_history) - self._max_history_length
            logger.warning(
                f"[deep_agent.agent] Truncating conversation history: "
                f"dropping {dropped_count} messages (keeping first {keep_prefix} + last {keep_suffix})"
            )
            prefix = self._conversation_history[:keep_prefix]
            suffix = self._conversation_history[-keep_suffix:]
            self._conversation_history = prefix + suffix

        # Build system prompt using modular architecture
        system = build_system_prompt(
            include_enders_game=self._config.include_enders_game,
        )

        # Track tool calls for this cycle
        tool_calls_this_cycle = 0
        max_tool_calls = 20  # Safety limit

        # RF-2: Track conversation length at cycle start for mid-cycle trimming
        cycle_start_msg_idx = len(self._conversation_history)

        # === PROMPT CACHING SETUP ===
        # Use Claude's prompt caching to reduce token usage by ~50-60%
        # System prompt and tool definitions are cached with 5-min ephemeral TTL
        system_blocks = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"}  # 5-min TTL
            }
        ]

        # Add cache_control to the last tool definition for optimal caching
        # (Cache breakpoints should be at the end of large static content)
        cached_tools = [tool.copy() for tool in self._tool_definitions]
        if cached_tools:
            cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

        while tool_calls_this_cycle < max_tool_calls:
            # Call Claude with tools (with timeout to prevent hung cycles)
            try:
                response = await asyncio.wait_for(
                    self._client.messages.create(
                        model=self._config.model,
                        max_tokens=self._config.max_tokens,
                        temperature=self._config.temperature,
                        system=system_blocks,
                        messages=self._conversation_history,
                        tools=cached_tools,
                        extra_headers={
                            "anthropic-beta": "token-efficient-tools-2025-02-19"
                        },
                    ),
                    timeout=120.0,  # 2 minute timeout
                )

                # === CACHE METRICS LOGGING ===
                # Track prompt caching effectiveness for cost optimization
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    cache_created = getattr(usage, 'cache_creation_input_tokens', 0) or 0
                    cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
                    input_tokens = getattr(usage, 'input_tokens', 0) or 0
                    output_tokens = getattr(usage, 'output_tokens', 0) or 0

                    # Update cumulative metrics
                    self._total_input_tokens += input_tokens
                    self._total_output_tokens += output_tokens
                    self._total_cache_read_tokens += cache_read
                    self._total_cache_created_tokens += cache_created

                    # Log cache status
                    if cache_read > 0:
                        cache_pct = (cache_read / (input_tokens + cache_read)) * 100 if (input_tokens + cache_read) > 0 else 0
                        logger.info(
                            f"[deep_agent.agent] Cache HIT: {cache_read} tokens cached ({cache_pct:.1f}%), "
                            f"input={input_tokens}, output={output_tokens}"
                        )
                    elif cache_created > 0:
                        logger.info(
                            f"[deep_agent.agent] Cache CREATED: {cache_created} tokens cached for next call, "
                            f"input={input_tokens}, output={output_tokens}"
                        )
                    else:
                        logger.debug(
                            f"[deep_agent.agent] No cache: input={input_tokens}, output={output_tokens}"
                        )

            except asyncio.TimeoutError:
                logger.error("[deep_agent.agent] Claude API call timed out after 120s")
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_error", {
                        "error": "Claude API timeout (120s) - cycle terminated",
                        "cycle": self._cycle_count,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                return

            # Process response
            assistant_content = []
            tool_use_blocks = []

            for block in response.content:
                if block.type == "text":
                    # Stream thinking to WebSocket
                    if self._ws_manager and block.text.strip():
                        thinking_msg = {
                            "text": block.text,
                            "cycle": self._cycle_count,
                            "timestamp": time.strftime("%H:%M:%S"),
                        }
                        await self._ws_manager.broadcast_message("deep_agent_thinking", thinking_msg)
                        # Store in history for session persistence
                        self._thinking_history.append(thinking_msg)
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
            for i, tool_block in enumerate(tool_use_blocks):
                tool_calls_this_cycle += 1
                result = await self._execute_tool(tool_block.name, tool_block.input)

                # Stream tool call to WebSocket
                if self._ws_manager:
                    tool_call_msg = {
                        "tool": tool_block.name,
                        "input": tool_block.input,
                        "output_preview": str(result)[:200],
                        "cycle": self._cycle_count,
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    await self._ws_manager.broadcast_message("deep_agent_tool_call", tool_call_msg)
                    # Store in history for session persistence
                    self._tool_call_history.append(tool_call_msg)

                result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
                }
                # Cache breakpoint on last tool result — enables prefix caching for next API call
                if i == len(tool_use_blocks) - 1:
                    result_block["cache_control"] = {"type": "ephemeral"}
                tool_results.append(result_block)

            # Add tool results to conversation
            self._conversation_history.append({
                "role": "user",
                "content": tool_results
            })

            # RF-2: Mid-cycle history safeguard — prevent unbounded token growth.
            # Every 5 tool calls, compact old tool results and hard-trim if needed.
            if tool_calls_this_cycle > 0 and tool_calls_this_cycle % 5 == 0:
                # First: shrink large tool result payloads from before this cycle
                self._compact_history_between_cycles()
                # Second: hard-trim if message count still exceeds safe limit
                hard_limit = self._max_history_length + 30  # ~42 messages max
                if len(self._conversation_history) > hard_limit:
                    keep_prefix = 2  # Session init messages
                    # Keep all current-cycle messages + a few pre-cycle messages for context
                    current_cycle_msgs = self._conversation_history[cycle_start_msg_idx:]
                    pre_cycle = self._conversation_history[keep_prefix:cycle_start_msg_idx]
                    keep_pre = pre_cycle[-4:] if len(pre_cycle) > 4 else pre_cycle
                    dropped = len(pre_cycle) - len(keep_pre)
                    if dropped > 0:
                        logger.warning(
                            f"[deep_agent.agent] Mid-cycle trim: dropped {dropped} "
                            f"pre-cycle messages (history was {len(self._conversation_history)})"
                        )
                    self._conversation_history = (
                        self._conversation_history[:keep_prefix]
                        + keep_pre
                        + current_cycle_msgs
                    )
                    # Update cycle start index after trim
                    cycle_start_msg_idx = keep_prefix + len(keep_pre)

            # Check if we should stop (response indicates completion)
            if response.stop_reason == "end_turn":
                break

        logger.info(f"[deep_agent.agent] Cycle {self._cycle_count} completed with {tool_calls_this_cycle} tool calls")

    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """Execute a tool and return the result."""
        logger.info(f"[deep_agent.agent] Executing tool: {tool_name}")

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
                ticker = tool_input["ticker"]

                # === THINK ENFORCEMENT (SOFT - WARN ONLY) ===
                think_warning = None
                if self._last_think_decision is None:
                    think_warning = "No think() call before trade - you should always call think() first!"
                    logger.warning(f"[deep_agent.agent] {think_warning}")
                elif self._last_think_decision != "TRADE":
                    think_warning = f"Last think() decision was {self._last_think_decision}, not TRADE"
                    logger.warning(f"[deep_agent.agent] {think_warning}")
                elif self._last_think_timestamp and time.time() - self._last_think_timestamp > 120:
                    think_warning = "think() decision is stale (>2min) - consider calling think() again"
                    logger.warning(f"[deep_agent.agent] {think_warning}")

                # NOTE: Think state is cleared ONLY after successful trade execution
                # (moved to the success block below to prevent clearing on blocked trades)

                # === CIRCUIT BREAKER CHECK ===
                # Prevent repeated failures on the same market
                is_blacklisted, blacklist_reason = self._is_ticker_blacklisted(ticker)
                if is_blacklisted:
                    logger.warning(f"[deep_agent.agent] Trade blocked by circuit breaker: {ticker}")
                    return {
                        "success": False,
                        "ticker": ticker,
                        "side": tool_input["side"],
                        "contracts": tool_input["contracts"],
                        "error": blacklist_reason,
                        "circuit_breaker_blocked": True,
                    }

                # Validate contracts
                contracts = min(tool_input["contracts"], self._config.max_contracts_per_trade)

                result = await self._tools.trade(
                    ticker=ticker,
                    side=tool_input["side"],
                    contracts=contracts,
                    reasoning=tool_input["reasoning"],
                    execution_strategy=tool_input.get("execution_strategy", "aggressive"),
                )

                # === CIRCUIT BREAKER: Record failure if trade failed ===
                if not result.success and result.error:
                    was_blacklisted = self._record_trade_failure(ticker, result.error)
                    if was_blacklisted:
                        # Add note to result so agent knows about the blacklist
                        result.error = (
                            f"{result.error} | CIRCUIT BREAKER: Market now blacklisted "
                            f"after {self._config.circuit_breaker_threshold} failures. "
                            f"Try other markets instead."
                        )

                # Record for reflection if successful
                if result.success:
                    self._trades_executed += 1

                    # Clear think state only after successful execution
                    # (require fresh think for next trade)
                    self._last_think_decision = None
                    self._last_think_timestamp = None

                    # Mark signal as traded in lifecycle tracker
                    trade_signal_id = tool_input.get("signal_id", "")
                    if trade_signal_id and self._signal_tracker:
                        self._signal_tracker.mark_traded(trade_signal_id)
                        tracked = self._signal_tracker.get_signal_state(trade_signal_id)
                        if tracked and self._ws_manager:
                            await self._ws_manager.broadcast_message(
                                "signal_lifecycle_update",
                                tracked.to_dict(),
                            )

                    # Get event_ticker from trading client
                    event_ticker = ""
                    try:
                        if self._trading_client:
                            market = await self._trading_client.get_market(ticker)
                            event_ticker = market.get("event_ticker", "") if market else ""
                    except Exception as e:
                        logger.warning(f"[deep_agent.agent] Could not fetch event_ticker: {e}")

                    self._reflection.record_trade(
                        trade_id=result.order_id or str(uuid.uuid4()),
                        ticker=ticker,
                        event_ticker=event_ticker,
                        side=tool_input["side"],
                        contracts=contracts,
                        entry_price_cents=result.price_cents or 50,
                        reasoning=tool_input["reasoning"],
                    )

                    # Store in trade history for session persistence
                    trade_msg = {
                        "ticker": ticker,
                        "side": tool_input["side"],
                        "contracts": contracts,
                        "price_cents": result.price_cents or 50,
                        "reasoning": tool_input["reasoning"][:200],
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    self._trade_history.append(trade_msg)

                # Include think warning in result if present
                trade_result = result.to_dict()
                if think_warning:
                    trade_result["think_warning"] = think_warning
                return trade_result

            elif tool_name == "get_session_state":
                state = await self._tools.get_session_state()
                return state.to_dict()

            elif tool_name == "read_memory":
                content = await self._tools.read_memory(tool_input["filename"])
                return content

            elif tool_name == "write_memory":
                filename = tool_input["filename"]
                content = tool_input["content"]
                success = await self._tools.write_memory(filename, content)

                # Track learnings for session persistence
                if success and filename == "learnings.md":
                    learning_entry = {
                        "id": f"learning-{time.strftime('%H:%M:%S')}",
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    self._learnings_history.append(learning_entry)

                return {"success": success}

            elif tool_name == "append_memory":
                filename = tool_input["filename"]
                content = tool_input["content"]
                result = await self._tools.append_memory(filename, content)

                # Track learnings for session persistence
                if result.get("success") and filename == "learnings.md":
                    learning_entry = {
                        "id": f"learning-{time.strftime('%H:%M:%S')}",
                        "content": content[:200] + "..." if len(content) > 200 else content,
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    self._learnings_history.append(learning_entry)

                return result

            elif tool_name == "get_event_context":
                context = await self._tools.get_event_context(
                    event_ticker=tool_input["event_ticker"],
                )
                if context:
                    return context.to_dict()
                else:
                    return {"error": f"Event {tool_input['event_ticker']} not found or no tracked markets"}

            elif tool_name == "assess_trade_opportunity":
                assessment = await self._tools.assess_trade_opportunity(
                    market_ticker=tool_input["market_ticker"],
                    signal_impact_score=tool_input["signal_impact_score"],
                    signal_confidence=tool_input["signal_confidence"],
                    suggested_side=tool_input["suggested_side"],
                )
                return assessment.to_dict()

            elif tool_name == "think":
                # REQUIRED pre-trade analysis tool - forces structured reasoning
                # All 4 fields are required; reflection is optional for additional notes

                # Extract structured fields
                signal_analysis = tool_input.get("signal_analysis", "")
                strategy_check = tool_input.get("strategy_check", "")
                risk_assessment = tool_input.get("risk_assessment", "")
                decision = tool_input.get("decision", "")
                reflection = tool_input.get("reflection", "")  # Optional
                signal_id = tool_input.get("signal_id", "")  # Optional but encouraged

                # Validate required fields
                missing = []
                if not signal_analysis.strip():
                    missing.append("signal_analysis")
                if not strategy_check.strip():
                    missing.append("strategy_check")
                if not risk_assessment.strip():
                    missing.append("risk_assessment")
                if decision not in ("TRADE", "WAIT", "PASS"):
                    missing.append("decision (must be TRADE, WAIT, or PASS)")

                if missing:
                    return {"recorded": False, "error": f"Missing required fields: {', '.join(missing)}"}

                # Format structured analysis for display
                formatted_analysis = (
                    f"**Signal**: {signal_analysis}\n"
                    f"**Strategy**: {strategy_check}\n"
                    f"**Risks**: {risk_assessment}\n"
                    f"**Decision**: {decision}"
                )
                if reflection:
                    formatted_analysis += f"\n**Notes**: {reflection}"

                logger.info(f"[deep_agent.agent] Think: decision={decision}, signal={signal_analysis[:100]}...")

                # Build thinking message with structured data
                thinking_msg = {
                    "text": f"[Pre-Trade Analysis]\n{formatted_analysis}",
                    "cycle": self._cycle_count,
                    "timestamp": time.strftime("%H:%M:%S"),
                    "is_structured_reflection": True,
                    "decision": decision,
                    "structured_data": {
                        "signal_analysis": signal_analysis,
                        "strategy_check": strategy_check,
                        "risk_assessment": risk_assessment,
                        "decision": decision,
                        "reflection": reflection,
                    }
                }

                # Store in history for session persistence
                self._thinking_history.append(thinking_msg)

                # Update think enforcement tracking
                self._last_think_decision = decision
                self._last_think_timestamp = time.time()

                # Broadcast to connected WebSocket clients
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_thinking", thinking_msg)

                # Record evaluation in signal lifecycle tracker
                if signal_id and self._signal_tracker:
                    reason_summary = signal_analysis[:100] if signal_analysis else decision
                    self._signal_tracker.record_evaluation(
                        signal_id=signal_id,
                        cycle=self._cycle_count,
                        decision=decision,
                        reason=reason_summary,
                    )
                    # Broadcast lifecycle update
                    tracked = self._signal_tracker.get_signal_state(signal_id)
                    if tracked and self._ws_manager:
                        await self._ws_manager.broadcast_message(
                            "signal_lifecycle_update",
                            tracked.to_dict(),
                        )

                return {
                    "recorded": True,
                    "decision": decision,
                    "proceed_to_trade": decision == "TRADE",
                }

            elif tool_name == "reflect":
                # Structured post-trade reflection tool
                trade_ticker = tool_input["trade_ticker"]
                outcome_analysis = tool_input["outcome_analysis"]
                reasoning_accuracy = tool_input["reasoning_accuracy"]
                key_learning = tool_input["key_learning"]
                mistake = tool_input.get("mistake", "")
                pattern = tool_input.get("pattern", "")
                strategy_update_needed = tool_input.get("strategy_update_needed", False)
                confidence_in_learning = tool_input.get("confidence_in_learning", "medium")

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
                results = {"trade_ticker": trade_ticker, "appended_to": []}

                # Always append key learning to learnings.md
                learning_entry = (
                    f"\n## {timestamp} - {trade_ticker}\n"
                    f"- **Outcome**: {outcome_analysis}\n"
                    f"- **Reasoning Accuracy**: {reasoning_accuracy}\n"
                    f"- **Learning** [{confidence_in_learning}]: {key_learning}\n"
                )
                await self._tools.append_memory("learnings.md", learning_entry)
                results["appended_to"].append("learnings.md")

                # Append mistake if provided
                if mistake and mistake.strip():
                    mistake_entry = (
                        f"\n## {timestamp} - {trade_ticker}\n"
                        f"- {mistake}\n"
                    )
                    await self._tools.append_memory("mistakes.md", mistake_entry)
                    results["appended_to"].append("mistakes.md")

                # Append pattern if provided
                if pattern and pattern.strip():
                    pattern_entry = (
                        f"\n## {timestamp} - {trade_ticker}\n"
                        f"- {pattern}\n"
                    )
                    await self._tools.append_memory("patterns.md", pattern_entry)
                    results["appended_to"].append("patterns.md")

                results["strategy_update_needed"] = strategy_update_needed

                # Track learnings for session persistence
                learning_msg = {
                    "id": f"reflect-{time.strftime('%H:%M:%S')}",
                    "content": key_learning[:200],
                    "timestamp": time.strftime("%H:%M:%S"),
                }
                self._learnings_history.append(learning_msg)

                logger.info(
                    f"[deep_agent.agent] Reflect: {trade_ticker} "
                    f"accuracy={reasoning_accuracy} confidence={confidence_in_learning} "
                    f"files={results['appended_to']}"
                )

                return results

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"[deep_agent.agent] Error executing {tool_name}: {e}")
            return {"error": str(e)}

    async def _handle_reflection(self, trade: PendingTrade) -> Optional[ReflectionResult]:
        """Handle reflection callback from the reflection engine."""
        logger.info(f"[deep_agent.agent] Reflecting on trade: {trade.ticker} ({trade.result})")

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

            # Track consecutive losses for urgent consolidation
            trade_result = trade.result or "unknown"
            if trade_result == "loss":
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            # Increment reflection count and check for consolidation trigger
            self._reflection_count += 1
            should_consolidate, trigger_reason = self._should_consolidate(trade)
            if should_consolidate:
                await self._trigger_consolidation(trigger_reason)

            return ReflectionResult(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                result=trade_result,
                pnl_cents=trade.pnl_cents or 0,
                learning=learning,
                should_update_strategy=bool(pattern),
                mistake_identified=mistake,
                pattern_identified=pattern,
            )

        except Exception as e:
            logger.error(f"[deep_agent.agent] Error in reflection: {e}")
            return None

    def _should_consolidate(self, trade: PendingTrade) -> tuple:
        """
        Determine if memory consolidation should be triggered.

        Returns (should_consolidate, trigger_reason) tuple.

        Triggers:
        - Every 10 reflections (baseline)
        - Significant loss > $5
        - 3 consecutive losses (loss streak)
        - New mistake identified on a loss
        """
        # Baseline: every 10 reflections
        if self._reflection_count % 10 == 0:
            return True, "routine"

        # Urgent: significant loss (> $5 = 500 cents)
        if trade.pnl_cents is not None and trade.pnl_cents < -500:
            return True, "significant_loss"

        # Urgent: 3 consecutive losses
        if self._consecutive_losses >= 3:
            return True, "loss_streak"

        return False, ""

    async def _trigger_consolidation(self, trigger_reason: str = "routine") -> None:
        """Trigger memory consolidation after multiple reflections.

        Args:
            trigger_reason: Why consolidation was triggered (routine, significant_loss, loss_streak)
        """
        logger.info(
            f"[deep_agent.agent] Triggering consolidation after {self._reflection_count} reflections "
            f"(trigger: {trigger_reason})"
        )

        # Build urgency context based on trigger
        urgency_context = ""
        if trigger_reason == "significant_loss":
            urgency_context = (
                "\n**URGENT**: This consolidation was triggered by a significant loss (>$5). "
                "Focus on what went wrong and update risk rules immediately.\n"
            )
        elif trigger_reason == "loss_streak":
            urgency_context = (
                f"\n**URGENT**: This consolidation was triggered by {self._consecutive_losses} consecutive losses. "
                "Something systematic may be wrong. Review your recent trades for common patterns "
                "and consider tightening entry criteria.\n"
            )

        consolidation_prompt = f"""## Memory Consolidation Time
{urgency_context}
You have completed {self._reflection_count} reflections since session start.
Time to distill your learnings into actionable strategy updates.

### Instructions
1. Call read_memory("learnings.md") to review your recent insights
2. Identify patterns or rules that should be added to strategy.md
3. Call read_memory("strategy.md") to see your current rules
4. Call write_memory("strategy.md", ...) with updated rules if needed
5. Keep strategy.md concise and actionable - it's loaded every cycle

Focus on:
- New entry/exit rules discovered
- Position sizing adjustments
- Risk rules that worked or failed
- Patterns worth codifying

Be selective: only add rules with clear evidence from multiple trades.
"""
        self._conversation_history.append({
            "role": "user",
            "content": consolidation_prompt
        })

        try:
            await self._run_agent_cycle("")
            logger.info("[deep_agent.agent] Consolidation cycle completed")
        except Exception as e:
            logger.error(f"[deep_agent.agent] Error in consolidation: {e}")

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

        # Calculate cache efficiency
        total_potential_input = self._total_input_tokens + self._total_cache_read_tokens
        cache_hit_rate = (
            self._total_cache_read_tokens / total_potential_input * 100
            if total_potential_input > 0 else 0.0
        )

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
            "circuit_breaker": self.get_circuit_breaker_status(),
            "token_usage": {
                "total_input_tokens": self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "total_cache_read_tokens": self._total_cache_read_tokens,
                "total_cache_created_tokens": self._total_cache_created_tokens,
                "cache_hit_rate_pct": round(cache_hit_rate, 1),
                "tokens_saved_by_cache": self._total_cache_read_tokens,
            },
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

    def get_snapshot(self) -> Dict[str, Any]:
        """
        Get full agent state for new client initialization.

        Returns a snapshot containing all cumulative state needed to restore
        the frontend after a page refresh. This includes:
        - Agent status and configuration
        - Cycle count and trade statistics
        - Recent thinking, tool calls, and trades (from history buffers)
        - Pending trades awaiting settlement
        - Recent settlements/reflections

        Returns:
            Dict containing full agent state for WebSocket snapshot
        """
        stats = self.get_stats()
        return {
            "status": "active" if self._running else "stopped",
            "started_at": self._started_at,
            "cycle_count": self._cycle_count,
            "trades_executed": self._trades_executed,
            "win_rate": stats["reflection_stats"]["win_rate"],
            "config": asdict(self._config),
            "recent_thinking": list(self._thinking_history),
            "recent_tool_calls": list(self._tool_call_history),
            "recent_trades": list(self._trade_history),
            "recent_learnings": list(self._learnings_history),
            "pending_trades": self._reflection.get_pending_trades_serializable(),
            "settlements": self._reflection.get_recent_reflections(20),
            "tool_stats": self._tools.get_tool_stats(),
            "token_usage": stats["token_usage"],
            "signal_lifecycle": self._signal_tracker.get_all_states(),
        }
