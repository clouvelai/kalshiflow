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
import re
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from anthropic import AsyncAnthropic, BadRequestError

from .tools import DeepAgentTools
from .trade_executor import TradeExecutor, TradeIntent
from .reflection import ReflectionEngine, PendingTrade, ReflectionResult
from .prompts import build_system_prompt

if TYPE_CHECKING:
    from ..core.websocket_manager import V3WebSocketManager
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..core.state_container import V3StateContainer
    from ..state.tracked_markets import TrackedMarketsState
    from ..services.event_position_tracker import EventPositionTracker

logger = logging.getLogger("kalshiflow_rl.traderv3.deep_agent.agent")

# Per-token USD costs by model prefix (per million tokens)
MODEL_PRICING = {
    "claude-sonnet-4": {
        "input": 3.0 / 1_000_000,
        "output": 15.0 / 1_000_000,
        "cache_write": 3.75 / 1_000_000,
        "cache_read": 0.30 / 1_000_000,
    },
    "claude-haiku-4-5": {
        "input": 1.0 / 1_000_000,
        "output": 5.0 / 1_000_000,
        "cache_write": 1.25 / 1_000_000,
        "cache_read": 0.10 / 1_000_000,
    },
    "claude-3-5-haiku": {
        "input": 0.80 / 1_000_000,
        "output": 4.0 / 1_000_000,
        "cache_write": 1.0 / 1_000_000,
        "cache_read": 0.08 / 1_000_000,
    },
}


def _get_model_pricing(model: str) -> Dict[str, float]:
    """Resolve pricing by model prefix, falling back to Sonnet 4."""
    for prefix, pricing in MODEL_PRICING.items():
        if model.startswith(prefix):
            return pricing
    return MODEL_PRICING["claude-sonnet-4"]


@dataclass
class EventRanking:
    """A ranked event for the deep agent to focus on."""
    event_ticker: str
    score: float
    market_count: int
    signal_count: int
    max_consensus_strength: float
    has_position: bool
    has_thesis: bool
    top_market_ticker: str = ""
    top_consensus: str = ""


@dataclass
class EventFocus:
    """Tracks an ongoing thesis the agent has about an event."""
    event_ticker: str
    thesis: str
    confidence: str      # low/medium/high
    researched_at: float
    last_evaluated: float
    cycles_watched: int
    intent_submitted: bool


@dataclass
class DeepAgentConfig:
    """Configuration for the deep agent."""
    # Model settings
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096

    # Loop settings
    cycle_interval_seconds: float = 120.0  # How often to run observe-act cycle (2 minutes)
    max_trades_per_cycle: int = 3

    # Memory settings
    memory_dir: Optional[str] = None  # Defaults to ./memory
    memory_files: List[str] = field(default_factory=lambda: [
        "learnings.md",
        "strategy.md",
        "mistakes.md",
        "patterns.md",
        "golden_rules.md",
        "cycle_journal.md",
        "market_knowledge.md",
        "gdelt_reference.md",
    ])

    # Target events
    target_events: List[str] = field(default_factory=list)

    # Safety settings
    max_contracts_per_trade: int = 500  # Hard cap per single trade
    max_event_exposure_cents: int = 100_000  # $1,000 per-event dollar exposure cap
    distillation_model: str = ""  # Empty = use main model. Override with e.g. "claude-haiku-4-5-20250514"
    require_fresh_news: bool = True
    max_news_age_hours: float = 4.0

    # Circuit breaker settings (prevents repeated failures on same market)
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 3  # Failures before blacklisting
    circuit_breaker_window_seconds: float = 3600.0  # 1 hour window for failures
    circuit_breaker_cooldown_seconds: float = 1800.0  # 30 min blacklist duration

    # Signal lifecycle settings
    max_eval_cycles_per_signal: int = 3  # Max evaluations before auto-expire



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

        # Vector memory (initialized properly in start(), but set here for safety)
        self._vector_memory = None
        self._cycles_since_consolidation = 0

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

        # Wire circuit breaker callback and config into tools for preflight_check
        # (the callback is set after circuit breaker state is initialized below)
        self._tools._max_event_exposure_cents = self._config.max_event_exposure_cents
        self._tools._require_fresh_news = self._config.require_fresh_news
        self._tools._max_news_age_hours = self._config.max_news_age_hours

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
        self._cycles_since_last_trade = 0  # Anti-stagnation: tracks consecutive tradeless cycles
        self._last_cycle_at: Optional[float] = None

        # Conversation history for context
        self._conversation_history: List[Dict[str, Any]] = []
        self._max_history_length = 40  # ~8-10 cycles of history (expanded for reasoning continuity)

        # Trade executor (set by plugin after start)
        self._trade_executor: Optional[TradeExecutor] = None

        # Event focus tracker (theses across cycles)
        self._event_focus: Dict[str, EventFocus] = {}

        # Background task
        self._cycle_task: Optional[asyncio.Task] = None

        # History buffers for session persistence (replay to new WebSocket clients)
        self._thinking_history: deque = deque(maxlen=10)
        self._tool_call_history: deque = deque(maxlen=50)
        self._trade_history: deque = deque(maxlen=50)
        self._learnings_history: deque = deque(maxlen=50)

        # Cached extraction signals for snapshot persistence
        # Updated every cycle from get_extraction_signals() pre-fetch
        self._cached_extraction_signals: List[Dict] = []

        # Circuit breaker state (prevents repeated failures on same market)
        # Maps ticker -> list of failure timestamps
        self._failed_attempts: Dict[str, List[float]] = {}
        # Maps ticker -> blacklist expiry timestamp
        self._blacklisted_tickers: Dict[str, float] = {}

        # Wire circuit breaker checker callback into tools for preflight_check
        self._tools._circuit_breaker_checker = self._is_ticker_blacklisted

        # Wire token usage callback so tools can report their own API calls
        self._tools._token_usage_callback = self._accumulate_external_tokens

        # Think enforcement tracking (soft enforcement - warn if trade without think)
        self._last_think_decision: Optional[str] = None
        self._last_think_timestamp: Optional[float] = None
        self._last_think_estimated_probability: Optional[int] = None
        self._last_think_what_could_go_wrong: Optional[str] = None

        # Token usage tracking for prompt caching metrics
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_created_tokens = 0

        # Per-cycle token counters (reset each cycle)
        self._cycle_input_tokens = 0
        self._cycle_output_tokens = 0
        self._cycle_cache_read_tokens = 0
        self._cycle_cache_created_tokens = 0
        self._cycle_api_calls = 0
        self._model_pricing = _get_model_pricing(self._config.model)

        # Per-cycle memory-write tracking (reset each cycle, checked by _auto_memory_fallback)
        self._cycle_journal_written = False
        self._cycle_learnings_written = False

        # Reflection count for consolidation trigger
        self._reflection_count = 0
        self._consecutive_losses = 0  # Track loss streaks for urgent consolidation

        # Subagent registry (task() delegation targets)
        from .subagents import SUBAGENT_REGISTRY
        self._subagent_registry = SUBAGENT_REGISTRY
        self._subagent_instances: Dict[str, Any] = {}  # Lazy-initialized

        # Tool definitions for Claude
        self._tool_definitions = self._build_tool_definitions()

    def _build_task_tool_description(self) -> str:
        """Build the task() tool description from registered subagents."""
        parts = ["Delegate a task to a specialized subagent. The subagent runs its own LLM conversation and returns a concise summary. Available agents:"]
        for name, cls in self._subagent_registry.items():
            parts.append(f"- **{name}**: {cls.description}")
        return "\n".join(parts)

    def _get_subagent(self, name: str):
        """Lazy-initialize and return a subagent instance."""
        if name not in self._subagent_instances:
            cls = self._subagent_registry.get(name)
            if not cls:
                return None
            self._subagent_instances[name] = cls(
                client=self._client,
                memory_dir=self._tools._memory_dir,
                ws_manager=self._ws_manager,
            )
        return self._subagent_instances[name]

    def rebuild_tool_definitions(self) -> None:
        """Rebuild tool definitions after external clients are wired to tools.

        Called by coordinator after GDELT/microstructure clients are attached,
        so that tool availability reflects the actual wired state.
        """
        old_count = len(self._tool_definitions)
        self._tool_definitions = self._build_tool_definitions()
        new_count = len(self._tool_definitions)
        if new_count != old_count:
            logger.info(f"Tool definitions rebuilt: {old_count} → {new_count} tools")

    def _build_tool_definitions(self) -> List[Dict]:
        """Build Claude tool definitions."""
        tools = [
            {
                "name": "get_extraction_signals",
                "description": "PRIMARY DATA SOURCE: Query aggregated extraction signals from the extraction pipeline. Returns signals grouped by market with occurrence count, unique sources, engagement metrics, directional consensus, and recent extraction snippets. Higher unique_sources and engagement = stronger signal. Use this as your first tool every cycle.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "market_ticker": {
                            "type": "string",
                            "description": "Optional: filter by specific market ticker"
                        },
                        "event_ticker": {
                            "type": "string",
                            "description": "Optional: filter by event ticker"
                        },
                        "hours": {
                            "type": "number",
                            "description": "Time window in hours (default: 4)",
                            "default": 4
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of signals to return (default: 20)",
                            "default": 20
                        }
                    },
                    "required": []
                }
            },
            {
                "name": "get_markets",
                "description": "Get current market prices and spreads. Use to check prices before trading.",
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
                "name": "submit_trade_intent",
                "description": "Submit a trade intent to the executor. The executor handles preflight checks, pricing, and fill verification. You focus on WHAT to trade and WHY. Include exit criteria so the executor knows when to close. Use action='sell' to exit existing positions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "market_ticker": {
                            "type": "string",
                            "description": "Market ticker to trade (e.g., 'KXBONDIOUT-25FEB07-T42.5')"
                        },
                        "side": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "YES or NO contracts"
                        },
                        "contracts": {
                            "type": "integer",
                            "description": f"Number of contracts (1-{self._config.max_contracts_per_trade})"
                        },
                        "thesis": {
                            "type": "string",
                            "description": "Your thesis for this trade — why you believe the market is mispriced (stored for reflection)"
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Confidence in your thesis (default: medium)"
                        },
                        "exit_criteria": {
                            "type": "string",
                            "description": "When should the executor close this position? (e.g., 'price reaches 80c', 'thesis invalidated if funding bill passes')"
                        },
                        "max_price_cents": {
                            "type": "integer",
                            "description": "Maximum acceptable price in cents (default: 99). Executor won't pay more than this."
                        },
                        "execution_strategy": {
                            "type": "string",
                            "enum": ["aggressive", "moderate", "passive"],
                            "description": "aggressive=cross spread (immediate fill), moderate=midpoint, passive=near bid. Default: aggressive."
                        },
                        "action": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "buy=open new position (default), sell=close existing position."
                        }
                    },
                    "required": ["market_ticker", "side", "contracts", "thesis"]
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
                            "description": "What extraction signals are you acting on? (source count, engagement, direction, consensus)"
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
                            "description": "Your decision: TRADE (execute now), WAIT (wait for more sources), PASS (no edge)"
                        },
                        "reflection": {
                            "type": "string",
                            "description": "Optional: Any additional reasoning or notes for future learning"
                        },
                        "estimated_probability": {
                            "type": "number",
                            "description": "Your estimated probability the contract settles YES (0-100). Compare to market price to quantify your edge."
                        },
                        "what_could_go_wrong": {
                            "type": "string",
                            "description": "The specific scenario where this trade loses. REQUIRED for TRADE decisions."
                        },
                        "signal_id": {
                            "type": "string",
                            "description": "Optional: extraction ID or market ticker for tracking"
                        }
                    },
                    "required": ["signal_analysis", "strategy_check", "risk_assessment", "decision"]
                }
            },
            {
                "name": "understand_event",
                "description": "Build or refresh understanding of a Kalshi event. Produces the langextract specification (prompt, examples, extraction classes) for this event. Idempotent — call anytime to update. Results stored in event_configs table and automatically used by the extraction pipeline. Call this when you encounter a new event or want to improve extraction quality for an existing one.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_ticker": {
                            "type": "string",
                            "description": "The event ticker to research (e.g., 'KXBONDIOUT')"
                        },
                        "force_refresh": {
                            "type": "boolean",
                            "description": "Force re-research even if recently researched (default: false)",
                            "default": False
                        }
                    },
                    "required": ["event_ticker"]
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
            },
            {
                "name": "write_cycle_summary",
                "description": "Record your end-of-cycle reasoning trail. Call this at the end of EVERY cycle. Future-you reads this journal to remember what you were thinking, what you evaluated, and what you're watching. Auto-archives at 10KB like other append-only files.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "signals_observed": {
                            "type": "string",
                            "description": "What extraction signals did you evaluate this cycle? Which markets had activity?"
                        },
                        "decisions_made": {
                            "type": "string",
                            "description": "What decisions did you make (TRADE/WAIT/PASS) and for which markets?"
                        },
                        "reasoning_notes": {
                            "type": "string",
                            "description": "Key reasoning: why you acted or didn't act. What stood out?"
                        },
                        "markets_of_interest": {
                            "type": "string",
                            "description": "Markets you're watching for next cycle. Hypotheses to test."
                        }
                    },
                    "required": ["signals_observed", "decisions_made", "reasoning_notes"]
                }
            },
            {
                "name": "read_todos",
                "description": "Read your current TODO task list. Returns pending and completed tasks with priorities. Use at cycle start to check your plan, and after completing tasks to update the list.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "write_todos",
                "description": "Write your TODO task list (full replace). Use to plan multi-step research, track investigation goals, and prioritize across markets. Completed items auto-expire after 10 cycles. Call with all current items (pending + done) — this replaces the entire list.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "task": {
                                        "type": "string",
                                        "description": "What needs to be done"
                                    },
                                    "priority": {
                                        "type": "string",
                                        "enum": ["high", "medium", "low"],
                                        "description": "Task priority"
                                    },
                                    "status": {
                                        "type": "string",
                                        "enum": ["pending", "done"],
                                        "description": "Task status"
                                    }
                                },
                                "required": ["task", "priority", "status"]
                            },
                            "description": "Complete task list (replaces existing)"
                        }
                    },
                    "required": ["items"]
                }
            },
            {
                "name": "task",
                "description": self._build_task_tool_description(),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "agent": {
                            "type": "string",
                            "enum": list(self._subagent_registry.keys()),
                            "description": "Which subagent to delegate to"
                        },
                        "input": {
                            "type": "string",
                            "description": "Natural language task description with enough context for the subagent to act"
                        }
                    },
                    "required": ["agent", "input"]
                }
            },
            {
                "name": "get_reddit_daily_digest",
                "description": "Get the Reddit daily digest - top 25 posts from past 24h across tracked subreddits with comment analysis and extracted signals. Provides a high-level framework of what Reddit discussed today, which markets have directional consensus, and top posts driving narratives. Use at session start to ground your analysis, and periodically to refresh your understanding. Different from get_extraction_signals() which shows real-time signal aggregates.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "force_refresh": {
                            "type": "boolean",
                            "description": "Force a new digest run (default: false). Respects 6h cooldown.",
                            "default": False,
                        }
                    },
                    "required": []
                }
            }
        ]

        # Conditionally include GDELT tools only when clients are configured
        has_gdelt = (
            self._tools._gdelt_client is not None
            or self._tools._gdelt_doc_client is not None
        )
        if has_gdelt:
            # BigQuery tools (require _gdelt_client)
            if self._tools._gdelt_client is not None:
                tools.extend([
                    {
                        "name": "query_gdelt_news",
                        "description": "Query GDELT Global Knowledge Graph for mainstream news coverage. Searches thousands of news sources worldwide (updated every 15 min). Use to CONFIRM Reddit signals with authoritative sources, or DISCOVER news Reddit hasn't caught yet. Returns article count, source diversity, tone analysis, key persons/orgs/themes, top articles with URLs and quotations, and a timeline. Complements get_extraction_signals() (Reddit) with mainstream media coverage.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "search_terms": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Terms to search for across persons, organizations, names, and themes. Use entity names from extraction signals or watchlists."
                                },
                                "window_hours": {
                                    "type": "number",
                                    "description": "How far back to search (default: 4 hours). Wider windows cost more BigQuery bytes.",
                                    "default": 4.0
                                },
                                "tone_filter": {
                                    "type": "string",
                                    "enum": ["positive", "negative"],
                                    "description": "Optional: filter articles by tone direction"
                                },
                                "source_filter": {
                                    "type": "string",
                                    "description": "Optional: filter by source name (partial match, e.g. 'reuters', 'bbc')"
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Max articles to return (default: 100)",
                                    "default": 100
                                }
                            },
                            "required": ["search_terms"]
                        }
                    },
                    {
                        "name": "query_gdelt_events",
                        "description": "Query GDELT Events Database for structured Actor-Event-Actor triples with CAMEO coding and GoldsteinScale conflict/cooperation scoring. Searches thousands of sources for HOW actors interact (investigate, threaten, cooperate, protest). Use to understand event dynamics beyond news coverage. Search by entity names — results include human-readable CAMEO labels, QuadClass conflict/cooperation categories, and Goldstein scores (-10 hostile to +10 cooperative).",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "actor_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Entity names to search in both Actor1 and Actor2 positions (e.g., ['Pam Bondi', 'DOJ'])"
                                },
                                "country_filter": {
                                    "type": "string",
                                    "description": "Optional: 3-letter ISO country code filter (e.g., 'USA')"
                                },
                                "window_hours": {
                                    "type": "number",
                                    "description": "How far back to search (default: 4 hours)",
                                    "default": 4.0
                                },
                                "limit": {
                                    "type": "integer",
                                    "description": "Max events to return (default: 50)",
                                    "default": 50
                                }
                            },
                            "required": ["actor_names"]
                        }
                    },
                ])
            # Free DOC API tools (require _gdelt_doc_client)
            if self._tools._gdelt_doc_client is not None:
                tools.extend([
                    {
                        "name": "search_gdelt_articles",
                        "description": "Search GDELT for recent news articles (FREE, no BigQuery). Returns article titles, URLs, tone scores, and source diversity. Faster than query_gdelt_news but less structured (no entity/theme disaggregation). Best for quick article lookup, recent coverage checks, or when BigQuery is unavailable.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "search_terms": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Terms to search for in article text"
                                },
                                "timespan": {
                                    "type": "string",
                                    "description": "Time window: '4h', '1d', '2w', etc. Default: 4h",
                                    "default": "4h"
                                },
                                "tone_filter": {
                                    "type": "string",
                                    "enum": ["positive", "negative"],
                                    "description": "Optional: filter by sentiment direction"
                                },
                                "max_records": {
                                    "type": "integer",
                                    "description": "Max articles (default: 75, max: 250)",
                                    "default": 75
                                },
                                "sort": {
                                    "type": "string",
                                    "enum": ["datedesc", "dateasc", "tonedesc", "toneasc"],
                                    "description": "Sort order (default: datedesc)",
                                    "default": "datedesc"
                                }
                            },
                            "required": ["search_terms"]
                        }
                    },
                    {
                        "name": "get_gdelt_volume_timeline",
                        "description": "Get media coverage volume timeline from GDELT (FREE). Shows how coverage of a topic changes over time. Use to detect breaking news surges, identify when stories peaked, or compare coverage trends. Returns array of {date, value} data points.",
                        "input_schema": {
                            "type": "object",
                            "properties": {
                                "search_terms": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Terms to search for"
                                },
                                "timespan": {
                                    "type": "string",
                                    "description": "Time window: '4h', '1d', '2w', etc. Default: 4h",
                                    "default": "4h"
                                },
                                "tone_filter": {
                                    "type": "string",
                                    "enum": ["positive", "negative"],
                                    "description": "Optional: filter by sentiment direction"
                                }
                            },
                            "required": ["search_terms"]
                        }
                    },
                ])

        # News Intelligence sub-agent tool (requires news_analyzer to be wired)
        if self._tools._news_analyzer is not None:
            tools.append({
                "name": "get_news_intelligence",
                "description": "PREFERRED for news analysis. Analyzes FULL GDELT data via a fast sub-agent and returns structured trading intelligence: narrative summary, sentiment with trend, market signals with evidence, source diversity, freshness assessment, and a trading recommendation (act_now/monitor/wait/no_signal). Cached for 15 min. Use this INSTEAD of search_gdelt_articles() for better, cheaper results. Raw GDELT tools remain available as fallback.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Terms to search for in news articles (e.g., ['government shutdown', 'funding bill'])"
                        },
                        "context_hint": {
                            "type": "string",
                            "description": "Optional context about what you're investigating (e.g., 'Evaluating KXGOVTFUND-25FEB14 NO position'). Helps the analyzer focus on trading-relevant aspects."
                        }
                    },
                    "required": ["search_terms"]
                }
            })


        # search_memory: semantic search across all historical vector memories
        tools.append({
            "name": "search_memory",
            "description": (
                "Semantic search across ALL historical memories (weeks/months). "
                "Returns the most relevant memories by meaning, not just recency. "
                "Use when you need past experience with a specific market, event, or situation."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (natural language)"
                    },
                    "types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["learning", "mistake", "pattern", "journal", "signal", "research", "thesis"]
                        },
                        "description": "Filter by memory type(s). Omit to search all."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default 8)",
                        "default": 8
                    }
                },
                "required": ["query"]
            }
        })

        return tools

    # === Extraction Snapshot Helper ===

    async def _snapshot_trade_extractions(self, ticker: str) -> tuple:
        """
        Snapshot recent extraction signals and GDELT queries for a traded ticker.

        Called at trade time to capture the extractions and news context that
        drove the trade decision. This snapshot is stored with the PendingTrade
        and later used during reflection to evaluate extraction/GDELT accuracy.

        Args:
            ticker: Market ticker to snapshot extractions for

        Returns:
            Tuple of (extraction_ids, extraction_snapshot, gdelt_snapshot)
        """
        extraction_ids = []
        extraction_snapshot = []

        try:
            supabase = self._tools._get_supabase()
            if not supabase:
                return extraction_ids, extraction_snapshot, []

            from datetime import timezone, timedelta
            cutoff = datetime.now(timezone.utc) - timedelta(hours=4)

            result = supabase.table("extractions") \
                .select("id, extraction_class, extraction_text, attributes, source_id, engagement_score, created_at") \
                .eq("extraction_class", "market_signal") \
                .contains("market_tickers", [ticker]) \
                .gte("created_at", cutoff.isoformat()) \
                .order("created_at", desc=True) \
                .limit(10) \
                .execute()

            if result.data:
                for row in result.data:
                    ext_id = str(row.get("id", ""))
                    extraction_ids.append(ext_id)
                    extraction_snapshot.append({
                        "id": ext_id,
                        "extraction_class": row.get("extraction_class", ""),
                        "extraction_text": row.get("extraction_text", ""),
                        "attributes": row.get("attributes", {}),
                        "source_id": row.get("source_id", ""),
                        "engagement_score": row.get("engagement_score", 0),
                        "created_at": row.get("created_at", ""),
                    })

                logger.info(
                    f"[deep_agent.agent] Snapshotted {len(extraction_ids)} extractions for {ticker}"
                )

        except Exception as e:
            logger.warning(f"[deep_agent.agent] Failed to snapshot extractions for {ticker}: {e}")

        # Also snapshot recent GDELT queries for reflection
        gdelt_snapshot = list(self._tools._recent_gdelt_queries)[:5]

        return extraction_ids, extraction_snapshot, gdelt_snapshot

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

    @staticmethod
    def _parse_probability(value) -> Optional[int]:
        """
        Robustly parse an estimated probability from agent output.

        The agent sometimes returns reasoning text instead of a bare number
        (e.g., "Based on the evidence, I estimate around 75%").  This method
        extracts the first numeric value from such strings and clamps it to 0-100.

        Args:
            value: Raw probability value -- int, float, str, or None.

        Returns:
            Integer in [0, 100], or None if parsing fails.
        """
        if value is None:
            return None
        # Fast path: already numeric
        if isinstance(value, (int, float)):
            try:
                return max(0, min(100, int(value)))
            except (ValueError, OverflowError):
                return None
        # String path: try direct conversion first
        s = str(value).strip()
        if not s:
            return None
        # Handle "N/A", "n/a", "NA", "none" -- valid for PASS decisions
        if s.lower() in ("n/a", "na", "none", "null", "n.a.", "-"):
            return None
        try:
            return max(0, min(100, int(float(s))))
        except (ValueError, TypeError):
            pass
        # Extract first number from text (handles "about 75%", "I estimate 80", etc.)
        match = re.search(r"(\d+(?:\.\d+)?)\s*%?", s)
        if match:
            try:
                parsed = int(float(match.group(1)))
                return max(0, min(100, parsed))
            except (ValueError, OverflowError):
                pass
        logger.warning(f"[deep_agent.agent] Could not parse estimated_probability: {s[:100]}")
        return None

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

    # =========================================================================
    # Lifecycle (start / stop / main loop)
    # =========================================================================

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
        self._cycles_since_last_trade = 0

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

        # Prune old archive directories (keep max 10)
        self._prune_archives()

        # Initialize vector memory BEFORE distillation so _distill_archived_learnings()
        # can use it for vector-enhanced distillation (fixes AttributeError on startup)
        self._vector_memory = None
        self._cycles_since_consolidation = 0
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key and self._tools._get_supabase():
            try:
                from .vector_memory import VectorMemoryService
                self._vector_memory = VectorMemoryService(
                    self._tools._get_supabase, openai_key
                )
                self._tools._vector_memory = self._vector_memory
                logger.info("[deep_agent.agent] Vector memory service initialized")
                # Background backfill from flat files (non-blocking)
                asyncio.create_task(self._safe_vector_backfill())
            except Exception as e:
                logger.warning(f"[deep_agent.agent] Vector memory init failed (non-fatal): {e}")
        else:
            if not openai_key:
                logger.info("[deep_agent.agent] OPENAI_API_KEY not set - vector memory disabled")
            else:
                logger.info("[deep_agent.agent] Supabase not available - vector memory disabled")

        # Distill learnings from previous sessions into strategy
        # (must be after vector memory init so it can use vector-enhanced distillation)
        try:
            await self._distill_archived_learnings()
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Memory distillation failed (non-fatal): {e}")

        # Try to restore session state from crash-resilient file (< 1 hour old)
        restored = self._load_session_state()
        if not restored:
            # Fresh session - reset counters
            self._reflection_count = 0
            self._consecutive_losses = 0

        # Start reflection engine
        await self._reflection.start()
        logger.info("[deep_agent.agent] Reflection engine started")

        # Reconstruct event exposure from existing positions (prevents double-exposure after restart)
        try:
            reconstructed = self._tools.reconstruct_event_exposure()
            if reconstructed:
                logger.info(f"[deep_agent.agent] Reconstructed exposure for {len(reconstructed)} events")
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Exposure reconstruction failed (non-fatal): {e}")

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
        if self._trades_executed == 0:
            logger.info("[deep_agent.agent] No trades in session, skipping summary")
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
            logger.info("[deep_agent.agent] Session summary completed")
        except Exception as e:
            logger.error(f"[deep_agent.agent] Error in session summary: {e}")
        finally:
            self._running = False

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

## Core Edge
You see Reddit/news signals before the market absorbs them. Your advantage is
temporal — it decays as information spreads. Act on genuine new information;
skip noise and already-priced stories.

## Decision Framework

### TRADE — Signal represents new information the market hasn't priced
- The extraction signal describes something that JUST happened or was JUST reported
- The current market price doesn't reflect this information (check via preflight)
- You can articulate WHY this changes the probability of the outcome

### PASS — No clear information edge
- The signal is about old news (first_seen > 2h ago and market has moved)
- You can't explain how this changes the probability
- The market price already reflects the signal direction

### Position Sizing
- Default: $25-50 per trade while learning what works
- Scale up ONLY after you have evidence from settled trades that a pattern works
- $1,000 max per event (hard cap, non-negotiable)

### Execution
- Use moderate execution by default (saves spread vs. aggressive)
- Use aggressive only when the signal is breaking and time-sensitive

## What I Don't Know Yet
- Which signal patterns (source count, engagement level, topic) predict profitable trades
- How fast the market absorbs Reddit signals (minutes? hours?)
- Whether entity sentiment correlates with market outcomes
- Which event types are most predictable

Track these unknowns. Every settled trade is data. Update this file when you
learn something concrete.
""",
            "mistakes.md": """# Mistakes to Avoid

Errors and anti-patterns go here.
""",
            "patterns.md": """# Winning Patterns

Successful patterns go here.
""",
            "golden_rules.md": """# Golden Rules

High-confidence rules backed by 5+ profitable trades. These are permanently preserved.
""",
            "cycle_journal.md": """# Cycle Journal

Agent reasoning trail - each cycle's observations, decisions, and hypotheses.
""",
            "market_knowledge.md": """# Kalshi Market Knowledge (Empirical)

## Microstructure (Becker 2025, 72.1M trades)
- Makers +1.12% excess return, takers -1.12%
- NO outperforms YES at 69/99 price levels
- Fee: 0.07 x C x P x (1-P), peaks at 50c (~1.75c/contract)
- Category efficiency: Finance (0.17pp) > Sports (2.23pp) > Entertainment (4.79pp) > World Events (7.32pp)

## Behavioral Biases
- Longshot bias: extreme YES prices (<15c) systematically overpriced
- YES-side bias: bettors prefer buying YES (creates structural NO value)
- Recency bias: markets overreact to recent events, then revert

## Fee by Price Point
- 10c/90c: ~0.63c | 25c/75c: ~1.31c | 50c: ~1.75c (max)

## What Works
- Statistical models on modelable variables
- Information speed (react to news before market absorbs)
- Longshot bias exploitation (systematic NO at extreme YES)
- What does NOT work: pure forecasting, gut feel, high-volume low-conviction
""",
            "gdelt_reference.md": """# GDELT CAMEO Taxonomy Reference

## CAMEO Event Codes
- 01-04: Cooperative verbal (statements, appeals, consultations, intent)
- 05-08: Cooperative material (diplomacy, aid, yielding)
- 09: Investigate (inquiries, probes, audits)
- 10-12: Conflictual verbal (demands, disapproval, rejection)
- 13-14: Conflictual actions (threats, protests)
- 15-20: Conflictual material (force, reduced relations, coercion, assault, fighting)

## GoldsteinScale (-10 to +10)
- +7 to +10: Major cooperation | +1 to +6: Mild cooperation
- -1 to -6: Mild conflict | -7 to -10: Major conflict

## QuadClass
1=Verbal Cooperation, 2=Material Cooperation, 3=Verbal Conflict, 4=Material Conflict
""",
        }

        for filename, content in initial_content.items():
            filepath = self._memory_dir / filename
            if not filepath.exists():
                filepath.write_text(content, encoding="utf-8")
                logger.info(f"[deep_agent.agent] Created initial memory file: {filename}")

    async def _load_memory_section(
        self,
        filename: str,
        recall_types: List[str],
        header: str,
        char_limit: int,
        recall_query: Optional[str],
    ) -> str:
        """Load a memory file with vector recall + tail fallback, returning a formatted section.

        Returns empty string if the file is empty or unreadable.
        """
        try:
            content = await self._tools.read_memory(filename)
            if not content or len(content) <= 50:
                return ""

            if recall_query and self._vector_memory:
                try:
                    vec_limit = char_limit - 200
                    vector_text = await self._vector_memory.recall_for_context(
                        query=recall_query, types=recall_types, char_limit=vec_limit
                    )
                    tail_text = self._tail_entries(content, n=3, max_chars=200)
                    preview = vector_text + ("\n" + tail_text if tail_text else "")
                except Exception:
                    preview = self._priority_load_memory(content, char_limit)
            else:
                preview = self._priority_load_memory(content, char_limit)

            return f"\n### {header}\n{preview}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load {filename}: {e}")
            return ""

    def _priority_load_memory(self, content: str, char_limit: int) -> str:
        """Load memory content prioritized by confidence tags.

        Splits content into entries by ## YYYY-MM-DD headers, then loads in
        priority order: [high] first, then [medium], then recent untagged,
        then [low]. Falls back to tail of file if no tags found.

        Args:
            content: Full memory file content
            char_limit: Maximum characters to return

        Returns:
            Priority-ordered content within char_limit
        """
        import re

        # Split into entries by ## headers
        entries = re.split(r'\n(?=## )', content)
        if not entries:
            return content[:char_limit]

        # Categorize entries by priority tag
        high = []
        medium = []
        low = []
        untagged = []

        for entry in entries:
            entry_stripped = entry.strip()
            if not entry_stripped or len(entry_stripped) < 10:
                continue
            entry_lower = entry_stripped.lower()
            if "[high]" in entry_lower:
                high.append(entry_stripped)
            elif "[medium]" in entry_lower:
                medium.append(entry_stripped)
            elif "[low]" in entry_lower:
                low.append(entry_stripped)
            else:
                untagged.append(entry_stripped)

        # If no tags found at all, fall back to chronological tail
        if not high and not medium and not low:
            return content[-char_limit:] if len(content) > char_limit else content

        # Build output in priority order
        result_parts = []
        remaining = char_limit

        # High priority: all entries
        for entry in high:
            if remaining <= 0:
                break
            result_parts.append(entry)
            remaining -= len(entry) + 1

        # Medium priority: all entries
        for entry in medium:
            if remaining <= 0:
                break
            result_parts.append(entry)
            remaining -= len(entry) + 1

        # Untagged: most recent first (last entries in file = most recent)
        for entry in reversed(untagged):
            if remaining <= 0:
                break
            result_parts.append(entry)
            remaining -= len(entry) + 1

        # Low priority: limited selection (most recent only)
        for entry in reversed(low[-3:]):
            if remaining <= 0:
                break
            result_parts.append(entry)
            remaining -= len(entry) + 1

        return "\n\n".join(result_parts)

    def _prune_archives(self, max_archives: int = 10) -> None:
        """Prune old archive directories, keeping only the most recent ones.

        Args:
            max_archives: Maximum number of archive directories to keep
        """
        archive_dir = self._memory_dir / "memory_archive"
        if not archive_dir.exists():
            return

        archive_dirs = sorted(
            [d for d in archive_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )

        if len(archive_dirs) <= max_archives:
            return

        to_delete = archive_dirs[max_archives:]
        for old_dir in to_delete:
            try:
                shutil.rmtree(old_dir)
                logger.info(f"[deep_agent.agent] Pruned old archive: {old_dir.name}")
            except Exception as e:
                logger.warning(f"[deep_agent.agent] Failed to prune archive {old_dir.name}: {e}")

        logger.info(
            f"[deep_agent.agent] Archive pruning: kept {max_archives}, "
            f"deleted {len(to_delete)} old archives"
        )

    def _save_session_state(self) -> None:
        """Persist volatile counters to session_state.json for crash recovery."""
        state_path = self._memory_dir / "session_state.json"
        now = time.time()
        # Only persist unexpired blacklist entries
        active_blacklist = {
            ticker: expiry
            for ticker, expiry in self._blacklisted_tickers.items()
            if expiry > now
        }
        state = {
            "reflection_count": self._reflection_count,
            "consecutive_losses": self._consecutive_losses,
            "cycle_count": self._cycle_count,
            "trades_executed": self._trades_executed,
            "blacklisted_tickers": active_blacklist,
            "saved_at": now,
        }
        try:
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Failed to save session state: {e}")

    def _load_session_state(self) -> bool:
        """Load volatile counters from session_state.json if recent (< 1 hour).

        Returns True if state was restored, False otherwise.
        """
        state_path = self._memory_dir / "session_state.json"
        if not state_path.exists():
            return False
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            saved_at = data.get("saved_at", 0)
            age_seconds = time.time() - saved_at
            if age_seconds > 3600:
                logger.info(
                    f"[deep_agent.agent] Session state is {age_seconds/60:.0f}m old (>1h), starting fresh"
                )
                return False

            self._reflection_count = data.get("reflection_count", 0)
            self._consecutive_losses = data.get("consecutive_losses", 0)
            self._cycle_count = data.get("cycle_count", 0)
            self._trades_executed = data.get("trades_executed", 0)

            # Restore unexpired blacklist entries
            now = time.time()
            for ticker, expiry in data.get("blacklisted_tickers", {}).items():
                if expiry > now:
                    self._blacklisted_tickers[ticker] = expiry

            logger.info(
                f"[deep_agent.agent] Restored session state: cycle={self._cycle_count}, "
                f"trades={self._trades_executed}, reflections={self._reflection_count}, "
                f"consecutive_losses={self._consecutive_losses}, "
                f"blacklisted={len(self._blacklisted_tickers)}"
            )
            return True
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Failed to load session state: {e}")
            return False

    async def _distill_archived_learnings(self) -> None:
        """
        Distill learnings from previous sessions into the current strategy.

        When vector memory is available, supplements archive distillation with
        high-value memories (access_count >= 3) for evidence-backed strategy updates.
        Falls back to flat-file archives when vector memory is unavailable.
        """
        # Try vector-enhanced distillation source (high-value memories from full history)
        vector_distill_source = ""
        if self._vector_memory:
            try:
                supabase = self._vector_memory._get_supabase()
                if supabase:
                    high_value = (
                        supabase.table("agent_memories")
                        .select(
                            "content, memory_type, access_count, trade_result, pnl_cents"
                        )
                        .eq("is_active", True)
                        .gte("access_count", 3)
                        .order("access_count", desc=True)
                        .limit(30)
                        .execute()
                    )
                    if high_value.data and len(high_value.data) >= 5:
                        vector_distill_source = "\n---\n".join(
                            f"[{m['memory_type']}, accessed {m['access_count']}x"
                            f"{', ' + m['trade_result'] if m.get('trade_result') else ''}]"
                            f"\n{m['content']}"
                            for m in high_value.data
                        )
                        logger.info(
                            f"[deep_agent.agent] Vector distillation: {len(high_value.data)} "
                            f"high-value memories available"
                        )
            except Exception as e:
                logger.debug(f"[deep_agent.agent] Vector distillation source failed: {e}")

        archive_dir = self._memory_dir / "memory_archive"

        # Collect archived content from flat files
        archived_content = []
        archive_dirs = []
        if archive_dir.exists():
            # Find up to 10 most recent archive directories (bounded by pruning)
            archive_dirs = sorted(
                [d for d in archive_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
                reverse=True,
            )[:10]

            for ad in archive_dirs:
                session_parts = []
                for filename in ["learnings.md", "patterns.md", "mistakes.md"]:
                    filepath = ad / filename
                    if filepath.exists():
                        content = filepath.read_text(encoding="utf-8").strip()
                        if content and len(content) > 50:
                            # Limit per file — more archives but less per archive
                            char_limit = 1200 if len(archive_dirs) > 5 else 2000
                            session_parts.append(f"### {filename}\n{content[:char_limit]}")
                if session_parts:
                    archived_content.append(
                        f"## Session: {ad.name}\n" + "\n\n".join(session_parts)
                    )

        # Need at least archives or vector source to proceed
        total_archive_chars = sum(len(c) for c in archived_content)
        if not archived_content and not vector_distill_source:
            logger.info("[deep_agent.agent] No archives and no vector memories — skipping distillation")
            return
        if total_archive_chars < 500 and not vector_distill_source:
            logger.info(
                f"[deep_agent.agent] Archived content too thin ({total_archive_chars} chars) "
                f"and no vector memories — skipping distillation"
            )
            return

        # Read current strategy
        strategy_path = self._memory_dir / "strategy.md"
        current_strategy = strategy_path.read_text(encoding="utf-8")

        # Read golden rules (must be preserved)
        golden_rules = ""
        golden_path = self._memory_dir / "golden_rules.md"
        if golden_path.exists():
            golden_rules = golden_path.read_text(encoding="utf-8").strip()

        golden_section = ""
        if golden_rules and len(golden_rules) > 80:
            golden_section = f"""
## Golden Rules (MUST preserve in strategy)
{golden_rules[:1500]}
"""

        # Include scorecard by_event stats as evidence
        by_event_section = ""
        try:
            by_event = self._reflection.get_by_event_stats()
            if by_event:
                event_lines = []
                sorted_events = sorted(by_event.items(), key=lambda x: x[1]["trades"], reverse=True)[:10]
                for et, ev in sorted_events:
                    ew = ev.get("wins", 0)
                    el = ev.get("losses", 0)
                    ep = ev.get("pnl_cents", 0)
                    event_lines.append(f"- {et}: {ev['trades']}T {ew}W/{el}L ${ep/100:.2f}")
                by_event_section = "\n## Performance by Event (Evidence)\n" + "\n".join(event_lines) + "\n"
        except Exception:
            pass

        # Include high-value vector memories if available
        vector_section = ""
        if vector_distill_source:
            vector_section = f"""
## High-Value Memories (proven by repeated recall)
{vector_distill_source[:3000]}
"""

        distill_prompt = f"""You are updating a trading strategy based on learnings from previous sessions.

## Current Strategy
{current_strategy}
{golden_section}{by_event_section}{vector_section}
## Archived Learnings ({len(archive_dirs)} sessions)
{"---".join(archived_content)}

## Instructions
Update the strategy with concrete rules supported by evidence from the learnings and performance data.
- Add specific entry/exit rules that the data supports
- Remove or modify rules contradicted by evidence
- ALL golden rules MUST be preserved in the output (they are permanent)
- Keep the core structure
- Don't add rules you can't support with evidence
- Strategy MUST be under 3000 characters
- Be concise — focus on the highest-value rules

Return ONLY the updated strategy.md content, nothing else."""

        try:
            response = await asyncio.wait_for(
                self._client.messages.create(
                    model=self._config.distillation_model or self._config.model,
                    max_tokens=2048,
                    temperature=0.3,
                    messages=[{"role": "user", "content": distill_prompt}],
                ),
                timeout=45.0,
            )

            # Track distillation token usage
            if hasattr(response, 'usage') and response.usage:
                u = response.usage
                self._total_input_tokens += getattr(u, 'input_tokens', 0) or 0
                self._total_output_tokens += getattr(u, 'output_tokens', 0) or 0
                self._total_cache_read_tokens += getattr(u, 'cache_read_input_tokens', 0) or 0
                self._total_cache_created_tokens += getattr(u, 'cache_creation_input_tokens', 0) or 0
                logger.info(
                    f"[deep_agent.agent] Distillation tokens: "
                    f"input={getattr(u, 'input_tokens', 0)}, output={getattr(u, 'output_tokens', 0)}"
                )

            updated_strategy = response.content[0].text.strip()

            # Sanity check: must be non-empty, contain key markers, and not regress significantly
            is_valid = (
                len(updated_strategy) > 100
                and "strategy" in updated_strategy.lower()
                and len(updated_strategy) >= len(current_strategy) * 0.5
            )
            if is_valid:
                # Apply version counter and cap via tools helper
                updated_strategy = self._tools._apply_strategy_cap(strategy_path, updated_strategy)
                strategy_path.write_text(updated_strategy, encoding="utf-8")
                logger.info(
                    f"[deep_agent.agent] Strategy distilled from {len(archive_dirs)} archived sessions "
                    f"({len(updated_strategy)} chars)"
                )
            else:
                logger.warning(
                    f"[deep_agent.agent] Distillation returned suspicious output "
                    f"({len(updated_strategy)} chars vs {len(current_strategy)} current), "
                    f"keeping current strategy"
                )

        except asyncio.TimeoutError:
            logger.warning("[deep_agent.agent] Distillation timed out after 45s, keeping current strategy")
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Distillation failed (non-fatal): {e}")

    # =========================================================================
    # Vector Memory Helpers
    # =========================================================================

    async def _safe_vector_backfill(self) -> None:
        """Background backfill of flat-file memories into vector store."""
        try:
            if self._vector_memory:
                await self._vector_memory.backfill_from_files(
                    self._memory_dir,
                    str(self._started_at),
                    llm_callable=self._consolidation_llm_call,
                )
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Vector memory backfill failed (non-fatal): {e}")

    async def _consolidation_llm_call(self, prompt: str) -> str:
        """Lightweight LLM call for memory consolidation and extraction.
        Uses a small/cheap model to keep costs minimal."""
        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-20250414",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Consolidation LLM call failed: {e}")
            return ""

    def _build_recall_query(self, signals) -> str:
        """Build semantic probe from current cycle context for vector recall."""
        parts = []
        for s in (signals or [])[:5]:
            ticker = s.get("market_ticker", "")
            consensus = s.get("consensus", "")
            if ticker:
                parts.append(f"{ticker} {consensus}".strip())
        for ticker in list(self._tools._traded_tickers)[:5]:
            parts.append(ticker)
        if self._config.target_events:
            parts.extend(self._config.target_events[:3])
        return " ".join(filter(None, parts))

    def _tail_entries(self, content: str, n: int = 3, max_chars: int = 200) -> str:
        """Return last N entries from a markdown file within max_chars."""
        if not content:
            return ""
        entries = re.split(r"\n(?=## )", content)
        tail = entries[-n:] if len(entries) > n else entries
        result = "\n".join(e.strip() for e in tail)
        return result[-max_chars:] if len(result) > max_chars else result

    async def _run_consolidation(self) -> None:
        """Periodic background task: consolidate similar memories + enforce retention."""
        try:
            for mem_type in ["learning", "mistake", "pattern"]:
                consolidated = await self._vector_memory.consolidate(
                    mem_type, self._consolidation_llm_call
                )
                if consolidated > 0:
                    logger.info(
                        f"[deep_agent.agent] Consolidated {consolidated} {mem_type} memories"
                    )

            # Enforce retention policy
            retention_result = await self._vector_memory.enforce_retention()
            if retention_result:
                logger.info(f"[deep_agent.agent] Retention cleanup: {retention_result}")
        except Exception as e:
            logger.warning(
                f"[deep_agent.agent] Consolidation/retention failed (non-fatal): {e}"
            )

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

    async def bootstrap_events(self) -> Dict[str, int]:
        """Bootstrap event configs for all tracked events.

        Calls understand_event() for each tracked event. Results are cached
        in the event_configs table for 24h, so restarts are cheap.

        Returns:
            Dict with counts: {"total": N, "cached": M, "new": K, "errors": E}
        """
        if not self._tracked_markets:
            logger.warning("[deep_agent.agent] No tracked_markets — skipping bootstrap")
            return {"total": 0, "cached": 0, "new": 0, "errors": 0}

        markets_by_event = self._tracked_markets.get_markets_by_event()
        if not markets_by_event:
            logger.warning("[deep_agent.agent] No events in tracked_markets — skipping bootstrap")
            return {"total": 0, "cached": 0, "new": 0, "errors": 0}

        total = len(markets_by_event)
        cached = 0
        new = 0
        errors = 0

        logger.info(f"[deep_agent.agent] Bootstrapping event configs for {total} events")

        for event_ticker in markets_by_event:
            try:
                result = await self._tools.understand_event(event_ticker)
                if result.get("status") == "cached":
                    cached += 1
                elif result.get("error"):
                    errors += 1
                    logger.warning(f"[deep_agent.agent] Bootstrap failed for {event_ticker}: {result['error']}")
                else:
                    new += 1
                    logger.info(f"[deep_agent.agent] Bootstrapped {event_ticker}: {result.get('event_title', '?')}")
            except Exception as e:
                errors += 1
                logger.error(f"[deep_agent.agent] Bootstrap error for {event_ticker}: {e}")

        logger.info(
            f"[deep_agent.agent] Bootstrap complete: {total} events, "
            f"{cached} cached, {new} new, {errors} errors"
        )
        return {"total": total, "cached": cached, "new": new, "errors": errors}

    async def _bootstrap_new_events(self) -> None:
        """Check for newly discovered events and bootstrap their configs.

        Runs at the start of each cycle. Compares tracked events against
        existing event_configs in Supabase. Any events without configs get
        understand_event() called automatically.
        """
        if not self._tracked_markets:
            return

        try:
            supabase = self._tools._get_supabase()
            if not supabase:
                return

            # Get tracked event tickers
            markets_by_event = self._tracked_markets.get_markets_by_event()
            if not markets_by_event:
                return

            tracked_events = set(markets_by_event.keys())

            # Get existing event configs
            result = supabase.table("event_configs") \
                .select("event_ticker") \
                .eq("is_active", True) \
                .execute()
            existing_events = {row["event_ticker"] for row in (result.data or [])}

            # Find events missing configs
            missing = tracked_events - existing_events
            if not missing:
                return

            logger.info(f"[deep_agent.agent] Found {len(missing)} new events without configs, bootstrapping...")
            for event_ticker in missing:
                try:
                    result = await self._tools.understand_event(event_ticker)
                    status = result.get("status", result.get("error", "unknown"))
                    logger.info(f"[deep_agent.agent] Bootstrapped new event {event_ticker}: {status}")
                except Exception as e:
                    logger.warning(f"[deep_agent.agent] Failed to bootstrap {event_ticker}: {e}")

        except Exception as e:
            logger.debug(f"[deep_agent.agent] New event check failed: {e}")

    async def _run_cycle(self) -> None:
        """Run one observe-act cycle."""
        self._cycle_count += 1
        self._last_cycle_at = time.time()

        # Reset per-cycle token counters
        self._cycle_input_tokens = 0
        self._cycle_output_tokens = 0
        self._cycle_cache_read_tokens = 0
        self._cycle_cache_created_tokens = 0
        self._cycle_api_calls = 0

        # Reset per-cycle memory-write tracking
        self._cycle_journal_written = False
        self._cycle_learnings_written = False

        logger.info(f"[deep_agent.agent] === Starting cycle {self._cycle_count} ===")

        # Broadcast cycle start
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_cycle", {
                "cycle": self._cycle_count,
                "phase": "starting",
                "timestamp": time.strftime("%H:%M:%S"),
                "cycle_interval": self._config.cycle_interval_seconds,
            })

        # PRE-CHECK: Are there actionable signals? (cheap Supabase query, ~50ms, zero API tokens)
        try:
            pre_check_result = await self._tools.get_extraction_signals(window_hours=4.0, limit=20)
            pre_fetched_signals = pre_check_result.get("signals", [])
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Pre-check signal fetch failed (non-fatal): {e}")
            pre_fetched_signals = []

        # Cache signals for snapshot persistence (survives page refresh)
        if pre_fetched_signals:
            self._cached_extraction_signals = pre_fetched_signals

        # Also check if we have open positions to monitor (always run cycle if positions exist)
        has_positions = (
            self._state_container
            and self._state_container.get_trading_summary().get("position_count", 0) > 0
        )
        # Fallback: also check pending trades (covers cases where order_id is null
        # and the state_container position sync hasn't picked up the trade yet)
        if not has_positions and self._reflection:
            unsettled = [
                t for t in self._reflection._pending_trades.values()
                if not t.settled
            ]
            if unsettled:
                has_positions = True

        # Bootstrap any newly discovered events (cheap DB check, creates event_configs for new events)
        try:
            await self._bootstrap_new_events()
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Event bootstrap failed (non-fatal): {e}")

        # Periodic scan: run Claude every 5th cycle even without extraction signals
        # so microstructure and GDELT can still be evaluated
        is_periodic_scan = (self._cycle_count % 5 == 0)

        if not pre_fetched_signals and not has_positions and not is_periodic_scan:
            logger.info(f"[deep_agent.agent] Cycle {self._cycle_count}: No signals, no positions - skipping Claude call")
            self._cycles_since_last_trade += 1
            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_cycle", {
                    "cycle": self._cycle_count,
                    "phase": "skipped_no_signals",
                    "timestamp": time.strftime("%H:%M:%S"),
                })
            return

        if is_periodic_scan and not pre_fetched_signals and not has_positions:
            logger.info(f"[deep_agent.agent] Cycle {self._cycle_count}: Periodic scan (no signals) — running Claude for microstructure/GDELT evaluation")

        # Track tradeless cycles (incremented here, reset to 0 in trade execution)
        self._cycles_since_last_trade += 1

        # Signals exist OR positions to manage OR periodic scan — run full cycle
        context = await self._build_cycle_context(
            pre_fetched_signals=pre_fetched_signals,
        )

        try:
            # Run the agent with streaming
            await self._run_agent_cycle(context)

            # Broadcast cost data after successful cycle
            if self._ws_manager and self._cycle_api_calls > 0:
                await self._ws_manager.broadcast_message("deep_agent_cost", {
                    "cycle": self._cycle_count,
                    "model": self._config.model,
                    "cycle_cost": self._calculate_cost(
                        self._cycle_input_tokens, self._cycle_output_tokens,
                        self._cycle_cache_read_tokens, self._cycle_cache_created_tokens,
                    ),
                    "cycle_tokens": {
                        "input": self._cycle_input_tokens,
                        "output": self._cycle_output_tokens,
                        "cache_read": self._cycle_cache_read_tokens,
                        "cache_created": self._cycle_cache_created_tokens,
                        "api_calls": self._cycle_api_calls,
                    },
                    "session_cost": self._calculate_cost(
                        self._total_input_tokens, self._total_output_tokens,
                        self._total_cache_read_tokens, self._total_cache_created_tokens,
                    ),
                    "session_tokens": {
                        "input": self._total_input_tokens,
                        "output": self._total_output_tokens,
                        "cache_read": self._total_cache_read_tokens,
                        "cache_created": self._total_cache_created_tokens,
                    },
                    "timestamp": time.strftime("%H:%M:%S"),
                })

        except Exception as e:
            logger.error(f"[deep_agent.agent] Error in cycle: {e}")

            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_error", {
                    "error": str(e)[:200],
                    "cycle": self._cycle_count,
                    "timestamp": time.strftime("%H:%M:%S"),
                })

        # Post-cycle housekeeping (all non-fatal — errors here must not kill the loop)
        try:
            await self._auto_memory_fallback()
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Auto-memory fallback failed: {e}")

        try:
            self._compact_history_between_cycles()
        except Exception as e:
            logger.warning(f"[deep_agent.agent] History compaction failed (non-fatal): {e}")

        # Periodic full history reset every 5 cycles to prevent context pollution.
        # Memory files (strategy.md, cycle_journal.md, todos.json, etc.) provide
        # cross-cycle continuity, so conversation history is largely redundant.
        if self._cycle_count > 0 and self._cycle_count % 5 == 0:
            old_len = len(self._conversation_history)
            self._conversation_history.clear()
            logger.info(
                f"[deep_agent.agent] Periodic history reset at cycle {self._cycle_count}: "
                f"cleared {old_len} messages (memory files provide continuity)"
            )

        try:
            self._save_session_state()
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Session state save failed (non-fatal): {e}")

        # Periodic consolidation + retention (every 50 cycles)
        if self._vector_memory:
            self._cycles_since_consolidation += 1
            if self._cycles_since_consolidation >= 50:
                self._cycles_since_consolidation = 0
                asyncio.create_task(self._run_consolidation())

    async def _auto_memory_fallback(self) -> None:
        """Auto-generate journal + learnings entries if agent didn't write them.

        Uses per-cycle tracking flags instead of fragile history scanning.
        This guarantees memory accumulation even when Claude's output is truncated.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        decision = self._last_think_decision or "no_think"

        # --- Journal fallback ---
        if not self._cycle_journal_written:
            # Build a richer auto-entry using cached cycle state
            signal_count = len(self._cached_extraction_signals)
            signal_tickers = [s.get("market_ticker", "?") for s in self._cached_extraction_signals[:5]]
            tickers_str = ", ".join(signal_tickers) if signal_tickers else "none"

            entry = (
                f"\n## Cycle {self._cycle_count} - {timestamp} [auto]\n"
                f"- **Decision**: {decision}\n"
                f"- **Signals**: {signal_count} ({tickers_str})\n"
                f"- **Trades this session**: {self._trades_executed}\n"
            )
            await self._tools.append_memory("cycle_journal.md", entry)
            logger.info(f"[deep_agent.agent] Auto-journal fallback wrote cycle {self._cycle_count} entry")

        # --- Learnings fallback ---
        if not self._cycle_learnings_written:
            # Auto-generate a minimal observation from cycle state
            signal_count = len(self._cached_extraction_signals)
            parts = [f"Cycle {self._cycle_count} observation:"]

            if signal_count > 0:
                tickers = [s.get("market_ticker", "?") for s in self._cached_extraction_signals[:3]]
                consensuses = [
                    f"{s.get('market_ticker', '?')}={s.get('consensus', '?')}"
                    for s in self._cached_extraction_signals[:3]
                ]
                parts.append(f"{signal_count} signals active ({', '.join(tickers)})")
                parts.append(f"Consensus: {', '.join(consensuses)}")
            else:
                parts.append("No signals this cycle")

            parts.append(f"Decision: {decision}")
            if self._trades_executed > 0:
                parts.append(f"Session trades so far: {self._trades_executed}")

            learning_entry = f"\n### {timestamp} [auto]\n- " + "\n- ".join(parts) + "\n"
            await self._tools.append_memory("learnings.md", learning_entry)
            logger.info(f"[deep_agent.agent] Auto-learnings fallback wrote cycle {self._cycle_count} entry")

    def _fix_orphaned_tool_pairs(self):
        """Remove orphaned tool_use/tool_result messages after history trimming.

        The Anthropic API requires that every assistant message with tool_use
        blocks is immediately followed by a user message containing the matching
        tool_result blocks. History trimming can split these pairs, causing
        400 errors ('tool_use ids found without tool_result blocks').

        This method scans the history and removes any messages that would
        violate the pairing constraint.
        """
        if len(self._conversation_history) < 2:
            return

        to_remove = set()
        for i, msg in enumerate(self._conversation_history):
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue

            if msg.get("role") == "assistant":
                # Check if this assistant message has tool_use blocks
                tool_use_ids = {
                    b["id"] for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b
                }
                if not tool_use_ids:
                    continue

                # The next message must be a user message with matching tool_result blocks
                if i + 1 < len(self._conversation_history):
                    next_msg = self._conversation_history[i + 1]
                    next_content = next_msg.get("content", [])
                    if next_msg.get("role") == "user" and isinstance(next_content, list):
                        result_ids = {
                            b.get("tool_use_id") for b in next_content
                            if isinstance(b, dict) and b.get("type") == "tool_result"
                        }
                        if tool_use_ids.issubset(result_ids):
                            continue  # Valid pair, skip

                # Orphaned tool_use — mark for removal (and its result if present)
                to_remove.add(i)

            elif msg.get("role") == "user":
                # Check if this user message has tool_result blocks
                result_ids = {
                    b.get("tool_use_id") for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                }
                if not result_ids:
                    continue

                # The previous message must be an assistant message whose
                # tool_use ids are a superset of this message's tool_result ids.
                if i > 0:
                    prev_msg = self._conversation_history[i - 1]
                    prev_content = prev_msg.get("content", [])
                    if prev_msg.get("role") == "assistant" and isinstance(prev_content, list):
                        prev_tool_use_ids = {
                            b["id"] for b in prev_content
                            if isinstance(b, dict) and b.get("type") == "tool_use" and "id" in b
                        }
                        if result_ids.issubset(prev_tool_use_ids) and (i - 1) not in to_remove:
                            continue  # Valid pair — all result ids match a tool_use

                # Orphaned tool_result — mark for removal
                to_remove.add(i)

        if to_remove:
            logger.warning(
                f"[deep_agent.agent] Removing {len(to_remove)} orphaned tool messages "
                f"from history (indices: {sorted(to_remove)})"
            )
            self._conversation_history = [
                msg for i, msg in enumerate(self._conversation_history)
                if i not in to_remove
            ]

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

    def _rank_events(self, pre_fetched_signals: List[Dict]) -> List[EventRanking]:
        """Rank events by signal strength for focused deep agent analysis.

        Pure Python, zero LLM cost. Groups signals by event, scores them,
        and returns the top 5 for the deep agent to focus on.
        """
        if not self._tracked_markets:
            return []

        markets_by_event = self._tracked_markets.get_markets_by_event()
        if not markets_by_event:
            return []

        # Build signal lookup: market_ticker -> signal dict
        signal_by_ticker = {}
        for s in pre_fetched_signals:
            ticker = s.get("market_ticker", "")
            if ticker:
                signal_by_ticker[ticker] = s

        # Check which events have open positions
        agent_tickers = set()
        if self._reflection:
            agent_tickers = {
                t.ticker for t in self._reflection._pending_trades.values()
            }

        # Score each event
        event_rankings = []
        for event_ticker, tracked_markets in markets_by_event.items():
            # Extract ticker strings from TrackedMarket objects
            market_ticker_strs = {m.ticker for m in tracked_markets}

            event_signals = [
                signal_by_ticker[t] for t in market_ticker_strs
                if t in signal_by_ticker
            ]
            if not event_signals and event_ticker not in {
                f.event_ticker for f in self._event_focus.values()
            }:
                # No signals and no existing thesis - skip
                has_pos = bool(agent_tickers & market_ticker_strs)
                if not has_pos:
                    continue

            signal_count = len(event_signals)
            max_strength = max(
                (s.get("consensus_strength", 0) for s in event_signals),
                default=0,
            )
            total_sources = sum(
                s.get("unique_source_count", 0) for s in event_signals
            )
            has_position = bool(agent_tickers & market_ticker_strs)
            has_thesis = event_ticker in self._event_focus

            # Score: signal_strength * 0.4 + freshness * 0.2 + has_position * 0.25 + has_thesis * 0.15
            signal_score = min(total_sources / 10.0, 1.0) * 0.4
            freshness_score = min(signal_count / 5.0, 1.0) * 0.2
            position_score = 0.25 if has_position else 0.0
            thesis_score = 0.15 if has_thesis else 0.0
            score = signal_score + freshness_score + position_score + thesis_score

            # Find top signal market for display
            top_signal = max(event_signals, key=lambda s: s.get("consensus_strength", 0)) if event_signals else {}

            event_rankings.append(EventRanking(
                event_ticker=event_ticker,
                score=score,
                market_count=len(market_ticker_strs),
                signal_count=signal_count,
                max_consensus_strength=max_strength,
                has_position=has_position,
                has_thesis=has_thesis,
                top_market_ticker=top_signal.get("market_ticker", ""),
                top_consensus=top_signal.get("consensus", ""),
            ))

        # Sort by score descending, return top 5
        event_rankings.sort(key=lambda r: r.score, reverse=True)
        return event_rankings[:5]

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

        # Load golden rules (permanently preserved high-confidence rules)
        golden_rules_str = ""
        try:
            golden_rules = await self._tools.read_memory("golden_rules.md")
            if golden_rules and len(golden_rules) > 80:
                golden_rules_str = f"\n### Golden Rules (High-Confidence, Permanent)\n{golden_rules[:1500]}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load golden rules: {e}")

        # Load market knowledge on first cycle only (session grounding, not needed every cycle)
        market_knowledge_str = ""
        if self._cycle_count <= 1:
            try:
                market_knowledge = await self._tools.read_memory("market_knowledge.md")
                if market_knowledge and len(market_knowledge) > 80:
                    market_knowledge_str = f"\n### Market Knowledge (Empirical Reference)\n{market_knowledge[:1500]}\n"
            except Exception as e:
                logger.debug(f"[deep_agent.agent] Could not load market knowledge: {e}")

        # Load cycle journal (agent's reasoning trail from recent cycles)
        journal_str = ""
        try:
            journal = await self._tools.read_memory("cycle_journal.md")
            if journal and len(journal) > 80:
                journal_preview = self._priority_load_memory(journal, 2000)
                journal_str = f"\n### Recent Cycle Journal (Your Reasoning Trail)\n{journal_preview}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load cycle journal: {e}")

        # Load TODO task list (agent's forward-looking plan)
        todos_str = ""
        try:
            todos_data = await self._tools.read_todos()
            pending_todos = [
                i for i in todos_data.get("items", [])
                if i.get("status") != "done"
            ]
            if pending_todos:
                todo_lines = ["### Your TODO List (Pending Tasks)"]
                for item in pending_todos:
                    priority_tag = f"[{item.get('priority', 'medium')}]"
                    todo_lines.append(f"- {priority_tag} {item.get('task', '')}")
                todos_str = "\n" + "\n".join(todo_lines) + "\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load todos: {e}")

        # Check for RALPH patches (fixes applied by the coding agent)
        ralph_patches_str = ""
        try:
            issue_reporter = self._get_subagent("issue_reporter")
            if issue_reporter:
                patches_result = await issue_reporter._check_patches()
                patches = patches_result.get("patches", [])
                if patches:
                    patch_lines = ["### RALPH Patches (Code Fixes Applied)"]
                    patch_lines.append("The RALPH coding agent has applied the following fixes. **Validate these are working:**")
                    for p in patches[-5:]:  # Show last 5
                        status = p.get("fix_status", "applied")
                        desc = p.get("description", "")[:150]
                        issue_id = p.get("issue_id", "?")
                        patch_lines.append(f"- [{status}] {issue_id}: {desc}")
                    patch_lines.append("If the issue persists, use task(agent='issue_reporter') to re-report it.")
                    ralph_patches_str = "\n" + "\n".join(patch_lines) + "\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load RALPH patches: {e}")

        # Load mistakes, patterns, and learnings EVERY cycle
        # Uses vector recall (semantic similarity + access boost) when available,
        # with flat-file tail for freshness, falling back to priority_load_memory.
        memory_context_str = ""

        # Build recall query from current signals for vector search
        recall_query = None
        if self._vector_memory and pre_fetched_signals:
            recall_query = self._build_recall_query(pre_fetched_signals)

        memory_sections = [
            ("mistakes.md", ["mistake"], "Mistakes to Avoid (from memory)",
             800 if self._cycle_count <= 10 else 400),
            ("patterns.md", ["pattern"], "Winning Patterns (from memory)",
             800 if self._cycle_count <= 10 else 400),
            ("learnings.md", ["learning"], "Learnings (from memory)",
             1000 if self._cycle_count <= 10 else 500),
        ]
        for filename, recall_types, header, char_limit in memory_sections:
            section = await self._load_memory_section(
                filename, recall_types, header, char_limit, recall_query
            )
            if section:
                memory_context_str += section

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

        # Build performance scorecard every 3 cycles (~9 min at 3-min interval)
        scorecard_str = ""
        # Always include persistent cross-session summary (one line, cheap)
        try:
            historical_summary = self._reflection.get_scorecard_summary()
            if historical_summary:
                scorecard_str = f"\n### Cross-Session Performance\n{historical_summary}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not build historical scorecard: {e}")
        # Add detailed session scorecard every 3 cycles
        if self._cycle_count % 3 == 0:
            try:
                scorecard = self._reflection._build_performance_scorecard()
                if scorecard:
                    scorecard_str += f"\n{scorecard}\n"
            except Exception as e:
                logger.debug(f"[deep_agent.agent] Could not build scorecard: {e}")

        # Inject recent trade outcomes every cycle (zero cost when scorecard is empty)
        try:
            recent_outcomes = self._reflection.get_recent_outcomes_summary(limit=3)
            if recent_outcomes:
                scorecard_str += f"\n{recent_outcomes}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not build recent outcomes: {e}")

        # Build ranked events section (replaces showing all 81 markets)
        focus_events_str = ""
        focus_events = self._rank_events(pre_fetched_signals or [])
        if focus_events:
            focus_lines = ["### Top Events (Ranked by Signal Strength)"]
            for i, evt in enumerate(focus_events, 1):
                pos_tag = " [POSITION]" if evt.has_position else ""
                thesis_tag = ""
                if evt.has_thesis and evt.event_ticker in self._event_focus:
                    thesis = self._event_focus[evt.event_ticker].thesis
                    thesis_tag = f" | Thesis: {thesis[:60]}"
                focus_lines.append(
                    f"{i}. **{evt.event_ticker}** (score={evt.score:.2f}) | "
                    f"{evt.signal_count} signals, {evt.market_count} markets | "
                    f"top={evt.top_market_ticker} consensus={evt.top_consensus} "
                    f"strength={evt.max_consensus_strength:.0%}{pos_tag}{thesis_tag}"
                )

            # Add vector recall context per focused event
            for evt in focus_events[:3]:
                if self._vector_memory:
                    try:
                        thesis_summary = ""
                        if evt.event_ticker in self._event_focus:
                            thesis_summary = self._event_focus[evt.event_ticker].thesis
                        event_context = await self._vector_memory.recall_for_context(
                            query=f"{evt.event_ticker} {thesis_summary}",
                            types=["signal", "research", "thesis", "learning", "pattern"],
                            char_limit=400,
                        )
                        if event_context:
                            focus_lines.append(f"\n**{evt.event_ticker} memories:**\n{event_context}")
                    except Exception as e:
                        logger.debug(f"[deep_agent.agent] Vector recall for {evt.event_ticker} failed: {e}")

            focus_events_str = "\n" + "\n".join(focus_lines) + "\n"

            # Broadcast focus events
            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_cycle", {
                    "cycle": self._cycle_count,
                    "phase": "focus_events",
                    "focus_events": [asdict(e) for e in focus_events],
                    "timestamp": time.strftime("%H:%M:%S"),
                })

        # Build executor status section
        executor_status_str = ""
        if self._trade_executor:
            try:
                exec_status = self._trade_executor.get_status_summary()
                active = exec_status.get("active_intents", [])
                recent = exec_status.get("recent_completed", [])
                totals = exec_status.get("totals", {})

                if active or recent or totals.get("intents_received", 0) > 0:
                    exec_lines = ["### Trade Executor Status"]
                    exec_lines.append(
                        f"Fills: {totals.get('fills', 0)} | "
                        f"Failures: {totals.get('failures', 0)} | "
                        f"Pending: {exec_status.get('pending_in_queue', 0)}"
                    )
                    for a in active:
                        exec_lines.append(
                            f"- ACTIVE: {a['market_ticker']} {a['side']} x{a['contracts']} "
                            f"({a['status']}, cycle {a['cycles_alive']})"
                        )
                    for r in recent[-3:]:
                        exec_lines.append(
                            f"- COMPLETED: {r['market_ticker']} {r['side']} -> {r['status']}"
                            + (f" ({r['failure_reason'][:60]})" if r.get('failure_reason') else "")
                        )
                    executor_status_str = "\n" + "\n".join(exec_lines) + "\n"
            except Exception as e:
                logger.debug(f"[deep_agent.agent] Executor status failed: {e}")

        # Build pre-loaded signals section (enriched with entity + context data)
        signals_section = ""
        if pre_fetched_signals:
            signals_lines = ["\n### Active Extraction Signals (pre-loaded)"]
            for s in pre_fetched_signals:
                ticker = s.get("market_ticker", "?")
                count = s.get("occurrence_count", 0)
                sources = s.get("unique_source_count", 0)
                consensus = s.get("consensus", "?")
                strength = s.get("consensus_strength", 0)
                magnitude = s.get("avg_magnitude", 0)
                engagement = s.get("max_engagement", 0)
                total_comments = s.get("total_comments", 0)
                signals_lines.append(
                    f"- **{ticker}** | occurrences={count} sources={sources} | "
                    f"consensus={consensus} ({strength:.0%}) magnitude={magnitude} | "
                    f"engagement={engagement} comments={total_comments}"
                )

                # Show entity mentions if present (top 3)
                entity_mentions = s.get("entity_mentions", [])
                if entity_mentions:
                    entity_strs = []
                    for em in entity_mentions[:3]:
                        name = em.get("entity_name", "?")
                        sent = em.get("avg_sentiment", 0)
                        mc = em.get("mention_count", 0)
                        sent_label = "+" if sent > 0 else "" if sent == 0 else ""
                        entity_strs.append(f"{name}({sent_label}{sent:.0f}, {mc}x)")
                    signals_lines.append(f"  Entities: {', '.join(entity_strs)}")

                # Show context factors if present
                context_factors = s.get("context_factors", [])
                if context_factors:
                    ctx_strs = [
                        f"{cf.get('category', '?')}/{cf.get('direction', '?')}({cf.get('mention_count', 0)}x)"
                        for cf in context_factors[:3]
                    ]
                    signals_lines.append(f"  Context: {', '.join(ctx_strs)}")

            signals_lines.append(
                "\nSignals above are pre-loaded. Use think() to analyze, then submit_trade_intent() for actionable theses."
            )
            signals_section = "\n".join(signals_lines) + "\n"

            # Determine GDELT availability for instruction tailoring
            has_gdelt_tools = (
                self._tools._gdelt_client is not None
                or self._tools._gdelt_doc_client is not None
                or self._tools._news_analyzer is not None
            )

            if has_gdelt_tools:
                instructions = """### Instructions
1. **FIRST** — call `append_memory("learnings.md", ...)` with at least ONE observation from the signals/context above
2. Review **Top Events** above — focus on the top 3 ranked events
3. For events with strong signals: call `get_news_intelligence(search_terms, context_hint)` to cross-reference
4. For unfamiliar events: call `understand_event(event_ticker)` to build deep understanding
5. Call **think()** to structure your analysis before any trade intent
6. If thesis is actionable: **submit_trade_intent()** with exit criteria — the executor handles execution
7. If existing thesis invalidated: submit_trade_intent(action="sell") to exit
8. **END** — call `write_cycle_summary()` with your reasoning

**The executor handles order mechanics.** Focus on WHAT to trade and WHY."""
            else:
                instructions = """### Instructions
1. **FIRST** — call `append_memory("learnings.md", ...)` with at least ONE observation
2. Review **Top Events** above — focus on the top 3 ranked events
3. For unfamiliar events: call `understand_event(event_ticker)` to build understanding
4. Call **think()** to structure your analysis before any trade intent
5. If thesis is actionable: **submit_trade_intent()** with exit criteria
6. If no actionable signals, PASS and wait for next cycle
7. **END** — call `write_cycle_summary()` with your reasoning

**The executor handles order mechanics.** Focus on WHAT to trade and WHY."""
        else:
            signals_section = "\n### No new signals this cycle. Focus on managing existing positions — check if any should be exited (sell) or held.\n"

            has_gdelt_tools = (
                self._tools._gdelt_client is not None
                or self._tools._gdelt_doc_client is not None
                or self._tools._news_analyzer is not None
            )

            if has_gdelt_tools:
                instructions = """### Instructions
1. **FIRST** — call `append_memory("learnings.md", ...)` with at least ONE observation (e.g. no new signals, position status, market conditions)
2. **GDELT news scan** — call `get_news_intelligence(search_terms)` with topics relevant to your tracked markets to discover breaking news or emerging signals that Reddit hasn't captured yet.
3. Check open positions: should any be exited via `submit_trade_intent(action="sell")`?
4. You can call get_extraction_signals() to re-query with different filters if needed
5. If nothing actionable, PASS and wait for next cycle
6. **END** — call `write_cycle_summary()` with your reasoning

**The executor handles order mechanics.** Focus on WHAT to trade and WHY."""
            else:
                instructions = """### Instructions
1. **FIRST** — call `append_memory("learnings.md", ...)` with at least ONE observation (e.g. no new signals, position status, market conditions)
2. No new signals available — focus on managing existing positions
3. Check open positions: should any be exited via `submit_trade_intent(action="sell")`?
4. You can call get_extraction_signals() to re-query with different filters if needed
5. If nothing actionable, PASS and wait for next cycle
6. **END** — call `write_cycle_summary()` with your reasoning

**The executor handles order mechanics.** Focus on WHAT to trade and WHY."""

        # Anti-stagnation nudge: after 3+ cycles with no trade, push the agent to act
        anti_stagnation_str = ""
        if self._cycles_since_last_trade >= 3:
            anti_stagnation_str = (
                "\n### ANTI-STAGNATION ALERT\n"
                f"**{self._cycles_since_last_trade} consecutive cycles without a trade.** "
                "You MUST submit at least one trade intent this cycle. Paper trading = free education. "
                "Submit a speculative $25-50 intent on your best signal. "
                "Inaction produces zero learning.\n"
            )

        context = f"""
## New Trading Cycle - {datetime.now().strftime("%Y-%m-%d %H:%M")}

### Current Session State
- Balance: ${session_state.balance_cents / 100:.2f}
- Portfolio Value: ${session_state.portfolio_value_cents / 100:.2f}
- Total P&L: ${session_state.total_pnl_cents / 100:.2f}
- Positions: {session_state.position_count}
- Trade Count: {session_state.trade_count}
- Win Rate: {session_state.win_rate:.1%}
{event_exposure_str}{open_positions_str}{executor_status_str}{strategy_excerpt}{golden_rules_str}{market_knowledge_str}{journal_str}{todos_str}{ralph_patches_str}{memory_context_str}{scorecard_str}{focus_events_str}{signals_section}{anti_stagnation_str}
{instructions}
"""
        return context

    # =========================================================================
    # Agent Cycle (LLM interaction, tool dispatch)
    # =========================================================================

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
            # Fix any tool_use/tool_result pairs broken by the trim
            self._fix_orphaned_tool_pairs()

        # Build system prompt using modular architecture
        system = build_system_prompt(
            include_gdelt=self._tools._gdelt_client is not None or self._tools._gdelt_doc_client is not None,
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

        # Filter out GDELT tools based on which clients are available
        has_gkg = self._tools._gdelt_client is not None
        has_doc = self._tools._gdelt_doc_client is not None
        _gkg_tools = {"query_gdelt_news", "query_gdelt_events"}
        _doc_tools = {"search_gdelt_articles", "get_gdelt_volume_timeline"}
        active_tools = [
            t for t in self._tool_definitions
            if t["name"] not in _gkg_tools | _doc_tools  # non-GDELT: always include
            or (t["name"] in _gkg_tools and has_gkg)
            or (t["name"] in _doc_tools and has_doc)
        ]

        # Add cache_control to the last tool definition for optimal caching
        # (Cache breakpoints should be at the end of large static content)
        cached_tools = [tool.copy() for tool in active_tools]
        if cached_tools:
            cached_tools[-1] = {**cached_tools[-1], "cache_control": {"type": "ephemeral"}}

        while tool_calls_this_cycle < max_tool_calls:
            # Call Claude with tools using streaming for real-time frontend updates
            try:
                # --- Streaming state ---
                _text_accum = ""          # Accumulated text for current text block
                _last_emit_ts = 0.0       # Timestamp of last thinking emit
                _DEBOUNCE_S = 0.3         # Emit thinking every 300ms

                async with asyncio.timeout(120.0):
                    async with self._client.messages.stream(
                        model=self._config.model,
                        max_tokens=self._config.max_tokens,
                        temperature=self._config.temperature,
                        system=system_blocks,
                        messages=self._conversation_history,
                        tools=cached_tools,
                        extra_headers={
                            "anthropic-beta": "token-efficient-tools-2025-02-19"
                        },
                    ) as stream:
                        async for event in stream:
                            # --- Text deltas: accumulate and debounce emit ---
                            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                                _text_accum += event.delta.text
                                now = time.monotonic()
                                if now - _last_emit_ts >= _DEBOUNCE_S and self._ws_manager:
                                    thinking_msg = {
                                        "text": _text_accum,
                                        "cycle": self._cycle_count,
                                        "timestamp": time.strftime("%H:%M:%S"),
                                        "streaming": True,
                                    }
                                    await self._ws_manager.broadcast_message("deep_agent_thinking", thinking_msg)
                                    _last_emit_ts = now

                            # --- Tool use start: emit tool_start event ---
                            elif event.type == "content_block_start" and hasattr(event.content_block, "type") and event.content_block.type == "tool_use":
                                if self._ws_manager:
                                    await self._ws_manager.broadcast_message("deep_agent_tool_start", {
                                        "tool": event.content_block.name,
                                        "id": event.content_block.id,
                                        "cycle": self._cycle_count,
                                        "timestamp": time.strftime("%H:%M:%S"),
                                    })

                            # --- Text block finished: flush remaining text ---
                            elif event.type == "content_block_stop":
                                if _text_accum and self._ws_manager:
                                    thinking_msg = {
                                        "text": _text_accum,
                                        "cycle": self._cycle_count,
                                        "timestamp": time.strftime("%H:%M:%S"),
                                        "streaming": False,
                                    }
                                    await self._ws_manager.broadcast_message("deep_agent_thinking", thinking_msg)
                                    self._thinking_history.append(thinking_msg)
                                    _text_accum = ""
                                    _last_emit_ts = 0.0

                    # Get the complete response (provides usage stats for cache metrics)
                    response = await stream.get_final_message()

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

                    # Update per-cycle metrics
                    self._cycle_input_tokens += input_tokens
                    self._cycle_output_tokens += output_tokens
                    self._cycle_cache_read_tokens += cache_read
                    self._cycle_cache_created_tokens += cache_created
                    self._cycle_api_calls += 1

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

            except TimeoutError:
                logger.error("[deep_agent.agent] Claude API call timed out after 120s")
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_error", {
                        "error": "Claude API timeout (120s) - cycle terminated",
                        "cycle": self._cycle_count,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                return

            except BadRequestError as e:
                # History is corrupted (e.g., orphaned tool_use/tool_result pairs).
                # Clear it — memory files provide cross-cycle continuity.
                old_len = len(self._conversation_history)
                self._conversation_history.clear()
                logger.error(
                    f"[deep_agent.agent] Claude API 400 error — cleared {old_len} "
                    f"history messages to recover: {e}"
                )
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_error", {
                        "error": f"Claude API error (history reset to recover): {str(e)[:150]}",
                        "cycle": self._cycle_count,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                return

            except Exception as e:
                logger.error(f"[deep_agent.agent] Stream error: {e}")
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_error", {
                        "error": f"Stream error: {str(e)[:150]}",
                        "cycle": self._cycle_count,
                        "timestamp": time.strftime("%H:%M:%S"),
                    })
                return

            # Build assistant content and tool blocks from final response
            assistant_content = []
            tool_use_blocks = []

            for block in response.content:
                if block.type == "text":
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
                tool_start = time.time()
                try:
                    result = await asyncio.wait_for(
                        self._execute_tool(tool_block.name, tool_block.input),
                        timeout=60.0,
                    )
                except asyncio.TimeoutError:
                    logger.error(f"[deep_agent.agent] Tool '{tool_block.name}' timed out after 60s")
                    result = {"error": f"Tool '{tool_block.name}' timed out after 60s", "timed_out": True}
                tool_duration_ms = int((time.time() - tool_start) * 1000)

                # Stream tool call to WebSocket
                if self._ws_manager:
                    tool_call_msg = {
                        "tool": tool_block.name,
                        "input": tool_block.input,
                        "output_preview": str(result)[:200],
                        "cycle": self._cycle_count,
                        "timestamp": time.strftime("%H:%M:%S"),
                        "started_at": tool_start,
                        "duration_ms": tool_duration_ms,
                    }
                    await self._ws_manager.broadcast_message("deep_agent_tool_call", tool_call_msg)
                    # Store in history for session persistence
                    self._tool_call_history.append(tool_call_msg)

                    # Emit dedicated GDELT result for the News Intelligence panel
                    _gdelt_article_tools = ("query_gdelt_news", "search_gdelt_articles")
                    if tool_block.name in _gdelt_article_tools and isinstance(result, dict) and result.get("article_count", 0) >= 0 and "error" not in result:
                        # Build articles list (GKG uses "top_articles", DOC uses "articles")
                        articles = result.get("top_articles", result.get("articles", []))[:5]
                        gdelt_msg = {
                            "search_terms": tool_block.input.get("search_terms", []),
                            "window_hours": tool_block.input.get("window_hours"),
                            "timespan": tool_block.input.get("timespan"),
                            "tone_filter": tool_block.input.get("tone_filter"),
                            "article_count": result.get("article_count", 0),
                            "source_diversity": result.get("source_diversity", 0),
                            "tone_summary": result.get("tone_summary", {}),
                            "key_themes": result.get("key_themes", [])[:10],
                            "key_persons": result.get("key_persons", [])[:8],
                            "key_organizations": result.get("key_organizations", [])[:8],
                            "top_articles": articles,
                            "timeline": result.get("timeline", []),
                            "cached": result.get("_cached", False),
                            "source": "gkg" if tool_block.name == "query_gdelt_news" else "doc_api",
                            "cycle": self._cycle_count,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "duration_ms": tool_duration_ms,
                        }
                        await self._ws_manager.broadcast_message("deep_agent_gdelt_result", gdelt_msg)

                    # Emit GDELT volume timeline as a separate result type
                    elif tool_block.name == "get_gdelt_volume_timeline" and isinstance(result, dict) and "error" not in result:
                        gdelt_msg = {
                            "search_terms": tool_block.input.get("search_terms", []),
                            "timespan": tool_block.input.get("timespan"),
                            "tone_filter": tool_block.input.get("tone_filter"),
                            "timeline": result.get("timeline", []),
                            "data_points": len(result.get("timeline", [])),
                            "cached": result.get("_cached", False),
                            "source": "volume_timeline",
                            "cycle": self._cycle_count,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "duration_ms": tool_duration_ms,
                        }
                        await self._ws_manager.broadcast_message("deep_agent_gdelt_result", gdelt_msg)

                    # Emit GDELT events (Actor-Event-Actor triples) result
                    elif tool_block.name == "query_gdelt_events" and isinstance(result, dict) and "error" not in result:
                        gdelt_msg = {
                            "actor_names": tool_block.input.get("actor_names", []),
                            "window_hours": tool_block.input.get("window_hours"),
                            "event_count": result.get("event_count", 0),
                            "quad_class_summary": result.get("quad_class_summary", {}),
                            "goldstein_summary": result.get("goldstein_summary", {}),
                            "top_event_triples": result.get("top_event_triples", [])[:10],
                            "top_actors": result.get("top_actors", [])[:10],
                            "event_code_distribution": result.get("event_code_distribution", [])[:10],
                            "geo_hotspots": result.get("geo_hotspots", [])[:5],
                            "cached": result.get("_cached", False),
                            "source": "events",
                            "cycle": self._cycle_count,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "duration_ms": tool_duration_ms,
                        }
                        await self._ws_manager.broadcast_message("deep_agent_gdelt_result", gdelt_msg)

                    # Emit news intelligence sub-agent result
                    elif tool_block.name == "get_news_intelligence" and isinstance(result, dict) and result.get("status") not in ("error", "unavailable"):
                        news_msg = {
                            "search_terms": tool_block.input.get("search_terms", []),
                            "context_hint": tool_block.input.get("context_hint", ""),
                            "status": result.get("status", "unknown"),
                            "intelligence": result.get("intelligence", {}),
                            "metadata": result.get("metadata", {}),
                            "cycle": self._cycle_count,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "duration_ms": tool_duration_ms,
                        }
                        await self._ws_manager.broadcast_message("deep_agent_news_intelligence", news_msg)

                # Cap tool result size to prevent any single result from
                # dominating the context window. 4K chars is enough for the
                # model to extract key information; raw JSON from tools like
                # get_extraction_signals can be 5-15K chars.
                _MAX_TOOL_RESULT_CHARS = 4000
                raw_content = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                if len(raw_content) > _MAX_TOOL_RESULT_CHARS:
                    raw_content = raw_content[:_MAX_TOOL_RESULT_CHARS] + "... [truncated]"
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": raw_content,
                }
                # NOTE: Do NOT add cache_control to tool results. The API allows
                # a maximum of 4 blocks with cache_control. We already use 2
                # (system prompt + last tool definition). Adding cache_control to
                # tool results in conversation history would exceed the limit on
                # subsequent API calls within the same cycle.
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
                    # Fix any tool_use/tool_result pairs broken by the trim
                    self._fix_orphaned_tool_pairs()
                    # Update cycle start index after trim
                    cycle_start_msg_idx = keep_prefix + len(keep_pre)

            # Check if we should stop (response indicates completion)
            if response.stop_reason == "end_turn":
                break

        logger.info(f"[deep_agent.agent] Cycle {self._cycle_count} completed with {tool_calls_this_cycle} tool calls")

    async def _execute_tool(self, tool_name: str, tool_input: Dict) -> Any:
        """Execute a tool and return the result.

        Dispatches to:
        1. Simple pass-through tools (direct delegation to self._tools)
        2. Medium-complexity tools with side effects (write_memory, append_memory, etc.)
        3. Complex tool handlers (_execute_trade, _execute_think, etc.)
        """
        logger.info(f"[deep_agent.agent] Executing tool: {tool_name}")

        try:
            # --- Simple pass-through tools ---
            handler = self._SIMPLE_TOOL_DISPATCH.get(tool_name)
            if handler:
                return await handler(self, tool_input)

            # --- Medium-complexity tools with side effects ---
            if tool_name == "write_memory":
                return await self._execute_write_memory(tool_input)
            elif tool_name == "append_memory":
                return await self._execute_append_memory(tool_input)
            elif tool_name == "write_cycle_summary":
                return await self._execute_write_cycle_summary(tool_input)
            elif tool_name == "write_todos":
                return await self._execute_write_todos(tool_input)

            # --- Subagent delegation ---
            elif tool_name == "task":
                return await self._execute_task(tool_input)

            # --- Complex tool handlers ---
            elif tool_name == "think":
                return await self._execute_think(tool_input)
            elif tool_name == "reflect":
                return await self._execute_reflect(tool_input)

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"[deep_agent.agent] Error executing {tool_name}: {e}")
            return {"error": str(e)}

    # ---- Simple tool dispatch table ----
    # Maps tool name -> async handler(self, tool_input) -> result.
    # Each handler is a thin lambda/coroutine that delegates to self._tools.

    @staticmethod
    async def _dispatch_get_extraction_signals(self, inp: Dict) -> Any:
        return await self._tools.get_extraction_signals(
            market_ticker=inp.get("market_ticker"),
            event_ticker=inp.get("event_ticker"),
            window_hours=inp.get("hours", 4),
            limit=inp.get("limit", 20),
        )

    @staticmethod
    async def _dispatch_get_markets(self, inp: Dict) -> Any:
        markets = await self._tools.get_markets(
            event_ticker=inp.get("event_ticker"),
            limit=inp.get("limit", 20),
        )
        return [m.to_dict() for m in markets]

    @staticmethod
    async def _dispatch_get_session_state(self, inp: Dict) -> Any:
        state = await self._tools.get_session_state()
        return state.to_dict()

    @staticmethod
    async def _dispatch_get_true_performance(self, inp: Dict) -> Any:
        perf = await self._tools.get_true_performance()
        return perf.to_dict()

    @staticmethod
    async def _dispatch_get_trade_log(self, inp: Dict) -> Any:
        return self._reflection.get_trade_log(
            limit=inp.get("limit", 10),
            event_ticker=inp.get("event_ticker"),
        )

    @staticmethod
    async def _dispatch_read_memory(self, inp: Dict) -> Any:
        return await self._tools.read_memory(inp["filename"])

    @staticmethod
    async def _dispatch_get_event_context(self, inp: Dict) -> Any:
        context = await self._tools.get_event_context(
            event_ticker=inp["event_ticker"],
        )
        if context:
            return context.to_dict()
        return {"error": f"Event {inp['event_ticker']} not found or no tracked markets"}

    @staticmethod
    async def _dispatch_preflight_check(self, inp: Dict) -> Any:
        return await self._tools.preflight_check(
            ticker=inp["ticker"],
            side=inp["side"],
            contracts=min(inp["contracts"], self._config.max_contracts_per_trade),
            execution_strategy=inp.get("execution_strategy", "aggressive"),
        )

    @staticmethod
    async def _dispatch_understand_event(self, inp: Dict) -> Any:
        return await self._tools.understand_event(
            event_ticker=inp["event_ticker"],
            force_refresh=inp.get("force_refresh", False),
        )

    @staticmethod
    async def _dispatch_read_todos(self, inp: Dict) -> Any:
        return await self._tools.read_todos()

    @staticmethod
    async def _dispatch_get_reddit_daily_digest(self, inp: Dict) -> Any:
        return await self._tools.get_reddit_daily_digest(
            force_refresh=inp.get("force_refresh", False),
        )

    @staticmethod
    async def _dispatch_query_gdelt_news(self, inp: Dict) -> Any:
        return await self._tools.query_gdelt_news(
            search_terms=inp.get("search_terms", []),
            window_hours=inp.get("window_hours"),
            tone_filter=inp.get("tone_filter"),
            source_filter=inp.get("source_filter"),
            limit=inp.get("limit"),
        )

    @staticmethod
    async def _dispatch_query_gdelt_events(self, inp: Dict) -> Any:
        return await self._tools.query_gdelt_events(
            actor_names=inp.get("actor_names", []),
            country_filter=inp.get("country_filter"),
            window_hours=inp.get("window_hours"),
            limit=inp.get("limit"),
        )

    @staticmethod
    async def _dispatch_search_gdelt_articles(self, inp: Dict) -> Any:
        return await self._tools.search_gdelt_articles(
            search_terms=inp.get("search_terms", []),
            timespan=inp.get("timespan"),
            tone_filter=inp.get("tone_filter"),
            max_records=inp.get("max_records"),
            sort=inp.get("sort", "datedesc"),
        )

    @staticmethod
    async def _dispatch_get_gdelt_volume_timeline(self, inp: Dict) -> Any:
        return await self._tools.get_gdelt_volume_timeline(
            search_terms=inp.get("search_terms", []),
            timespan=inp.get("timespan"),
            tone_filter=inp.get("tone_filter"),
        )

    @staticmethod
    async def _dispatch_get_news_intelligence(self, inp: Dict) -> Any:
        return await self._tools.get_news_intelligence(
            search_terms=inp.get("search_terms", []),
            context_hint=inp.get("context_hint", ""),
        )

    @staticmethod
    async def _dispatch_get_microstructure(self, inp: Dict) -> Any:
        return await self._tools.get_microstructure(
            market_ticker=inp.get("market_ticker"),
        )

    @staticmethod
    async def _dispatch_get_candlesticks(self, inp: Dict) -> Any:
        return await self._tools.get_candlesticks(
            event_ticker=inp.get("event_ticker", ""),
            period=inp.get("period", "hourly"),
            hours_back=inp.get("hours_back", 24),
        )

    @staticmethod
    async def _dispatch_submit_trade_intent(self, inp: Dict) -> Any:
        result = await self._tools.submit_trade_intent(
            market_ticker=inp["market_ticker"],
            side=inp["side"],
            contracts=min(inp["contracts"], self._config.max_contracts_per_trade),
            thesis=inp["thesis"],
            confidence=inp.get("confidence", "medium"),
            exit_criteria=inp.get("exit_criteria", ""),
            max_price_cents=inp.get("max_price_cents", 99),
            execution_strategy=inp.get("execution_strategy", "aggressive"),
            action=inp.get("action", "buy"),
        )
        # Track trade intent for anti-stagnation
        if result.get("status") == "submitted":
            self._cycles_since_last_trade = 0
            # Update event focus
            evt = result.get("event_ticker", "")
            if evt:
                self._event_focus[evt] = EventFocus(
                    event_ticker=evt,
                    thesis=inp["thesis"][:200],
                    confidence=inp.get("confidence", "medium"),
                    researched_at=time.time(),
                    last_evaluated=time.time(),
                    cycles_watched=self._event_focus.get(evt, EventFocus(evt, "", "", 0, 0, 0, False)).cycles_watched,
                    intent_submitted=True,
                )
        return result

    @staticmethod
    async def _dispatch_search_memory(self, inp: Dict) -> Any:
        return await self._tools.search_memory(
            query=inp["query"],
            types=inp.get("types"),
            limit=inp.get("limit", 8),
        )

    _SIMPLE_TOOL_DISPATCH = {
        "get_extraction_signals": _dispatch_get_extraction_signals,
        "get_markets": _dispatch_get_markets,
        "read_memory": _dispatch_read_memory,
        "get_event_context": _dispatch_get_event_context,
        "understand_event": _dispatch_understand_event,
        "read_todos": _dispatch_read_todos,
        "get_reddit_daily_digest": _dispatch_get_reddit_daily_digest,
        "query_gdelt_news": _dispatch_query_gdelt_news,
        "query_gdelt_events": _dispatch_query_gdelt_events,
        "search_gdelt_articles": _dispatch_search_gdelt_articles,
        "get_gdelt_volume_timeline": _dispatch_get_gdelt_volume_timeline,
        "get_news_intelligence": _dispatch_get_news_intelligence,
        "submit_trade_intent": _dispatch_submit_trade_intent,
        "search_memory": _dispatch_search_memory,
    }

    # ---- Medium-complexity tool handlers (side effects) ----

    async def _execute_write_memory(self, tool_input: Dict) -> Any:
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

    async def _execute_append_memory(self, tool_input: Dict) -> Any:
        filename = tool_input["filename"]
        content = tool_input["content"]
        result = await self._tools.append_memory(filename, content)
        if filename == "learnings.md":
            self._cycle_learnings_written = True

        # Track learnings for session persistence
        if result.get("success") and filename == "learnings.md":
            learning_entry = {
                "id": f"learning-{time.strftime('%H:%M:%S')}",
                "content": content[:200] + "..." if len(content) > 200 else content,
                "timestamp": time.strftime("%H:%M:%S"),
            }
            self._learnings_history.append(learning_entry)

        return result

    async def _execute_write_cycle_summary(self, tool_input: Dict) -> Any:
        self._cycle_journal_written = True
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        signals = tool_input.get("signals_observed", "none")
        decisions = tool_input.get("decisions_made", "none")
        reasoning = tool_input.get("reasoning_notes", "")
        markets = tool_input.get("markets_of_interest", "")

        entry = (
            f"\n## Cycle {self._cycle_count} - {timestamp}\n"
            f"- **Signals**: {signals}\n"
            f"- **Decisions**: {decisions}\n"
            f"- **Reasoning**: {reasoning}\n"
        )
        if markets:
            entry += f"- **Watching**: {markets}\n"

        result = await self._tools.append_memory("cycle_journal.md", entry)
        return {
            "recorded": result.get("success", False),
            "cycle": self._cycle_count,
        }

    async def _execute_write_todos(self, tool_input: Dict) -> Any:
        result = await self._tools.write_todos(
            items=tool_input.get("items", []),
            current_cycle=self._cycle_count,
        )
        # Broadcast updated todos to frontend
        if result.get("success") and self._ws_manager:
            todos_data = await self._tools.read_todos()
            await self._ws_manager.broadcast_message("deep_agent_todos", {
                "items": todos_data.get("items", []),
                "cycle": self._cycle_count,
            })
        return result

    # ---- Subagent delegation ----

    async def _execute_task(self, tool_input: Dict) -> Any:
        """Delegate a task to a named subagent.

        The subagent runs its own isolated LLM conversation with its own
        tools. The parent agent only receives the final summary, not the
        intermediate tool calls -- solving context bloat.
        """
        agent_name = tool_input.get("agent", "")
        task_input = tool_input.get("input", "")

        subagent = self._get_subagent(agent_name)
        if not subagent:
            available = ", ".join(self._subagent_registry.keys())
            return {"error": f"Unknown subagent '{agent_name}'. Available: {available}"}

        logger.info(
            "[deep_agent.agent] Delegating to subagent '%s': %s",
            agent_name, task_input[:100],
        )

        # Broadcast delegation to frontend
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_subagent", {
                "agent": agent_name,
                "action": "started",
                "input_preview": task_input[:200],
                "timestamp": time.strftime("%H:%M:%S"),
            })

        result = await subagent.run(task_input)

        # Broadcast completion
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_subagent", {
                "agent": agent_name,
                "action": "completed",
                "success": result.success,
                "summary_preview": result.summary[:200],
                "timestamp": time.strftime("%H:%M:%S"),
            })

        if result.success:
            return {"summary": result.summary, "data": result.data}
        else:
            return {"error": result.error or "Subagent failed", "summary": result.summary}

    # ---- Complex tool handlers ----

    async def _execute_trade(self, tool_input: Dict) -> Any:
        """Execute a trade with think enforcement, circuit breaker, and reflection recording."""
        ticker = tool_input["ticker"]

        # === THINK ENFORCEMENT (HARD BLOCK) ===
        think_warning = None
        if self._last_think_decision is None:
            error_msg = (
                "BLOCKED: You must call think() before trade(). "
                "Structured pre-trade analysis is required."
            )
            logger.warning(f"[deep_agent.agent] {error_msg}")
            return {
                "success": False,
                "ticker": ticker,
                "side": tool_input.get("side", ""),
                "contracts": tool_input.get("contracts", 0),
                "error": error_msg,
                "think_required": True,
            }
        elif self._last_think_decision != "TRADE":
            error_msg = (
                f"BLOCKED: Last think() decision was '{self._last_think_decision}', not 'TRADE'. "
                f"Call think() again with decision='TRADE' if you want to proceed."
            )
            logger.warning(f"[deep_agent.agent] {error_msg}")
            return {
                "success": False,
                "ticker": ticker,
                "side": tool_input.get("side", ""),
                "contracts": tool_input.get("contracts", 0),
                "error": error_msg,
                "think_required": True,
            }
        elif self._last_think_timestamp and time.time() - self._last_think_timestamp > 60:
            think_warning = (
                "WARNING: think() decision is stale (>60s). "
                "Trade proceeding but consider calling think() again for fresh analysis."
            )
            logger.warning(f"[deep_agent.agent] {think_warning}")

        # NOTE: Think state is cleared ONLY after successful trade execution
        # (moved to the success block below to prevent clearing on blocked trades)

        # === CIRCUIT BREAKER CHECK ===
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
            action=tool_input.get("action", "buy"),
        )

        # === CIRCUIT BREAKER: Record failure if trade failed ===
        if not result.success and result.error:
            was_blacklisted = self._record_trade_failure(ticker, result.error)
            if was_blacklisted:
                result.error = (
                    f"{result.error} | CIRCUIT BREAKER: Market now blacklisted "
                    f"after {self._config.circuit_breaker_threshold} failures. "
                    f"Try other markets instead."
                )

        # Record for reflection if successful
        if result.success:
            self._trades_executed += 1
            self._cycles_since_last_trade = 0  # Reset anti-stagnation counter

            # Capture calibration data before clearing think state
            trade_estimated_probability = self._last_think_estimated_probability
            trade_what_could_go_wrong = self._last_think_what_could_go_wrong

            # Clear think state (require fresh think for next trade)
            self._last_think_decision = None
            self._last_think_timestamp = None
            self._last_think_estimated_probability = None
            self._last_think_what_could_go_wrong = None

            # Get event_ticker from trading client
            event_ticker = ""
            try:
                if self._trading_client:
                    market = await self._trading_client.get_market(ticker)
                    event_ticker = market.get("event_ticker", "") if market else ""
            except Exception as e:
                logger.warning(f"[deep_agent.agent] Could not fetch event_ticker: {e}")

            # T1.2: Use limit_price_cents as fallback — never fabricate a price
            entry_price = result.price_cents or result.limit_price_cents or 1

            # Snapshot extractions + GDELT queries that drove this trade (for learning loop)
            extraction_ids, extraction_snapshot, gdelt_snapshot = await self._snapshot_trade_extractions(ticker)

            # Snapshot microstructure at trade time for reflection
            microstructure_snapshot = None
            if self._tools._trade_flow_service or self._tools._orderbook_integration:
                try:
                    micro = await self._tools.get_microstructure(market_ticker=ticker)
                    if micro.get("has_data"):
                        microstructure_snapshot = micro.get("data")
                except Exception:
                    pass  # Non-fatal

            action = tool_input.get("action", "buy")
            if action == "sell":
                await self._handle_sell_close(
                    ticker=ticker,
                    side=tool_input["side"],
                    contracts=contracts,
                    sell_price_cents=entry_price,
                    reasoning=tool_input["reasoning"],
                )
            else:
                self._reflection.record_trade(
                    trade_id=result.order_id or str(uuid.uuid4()),
                    ticker=ticker,
                    event_ticker=event_ticker,
                    side=tool_input["side"],
                    contracts=contracts,
                    entry_price_cents=entry_price,
                    reasoning=tool_input["reasoning"],
                    order_id=result.order_id,
                    extraction_ids=extraction_ids,
                    extraction_snapshot=extraction_snapshot,
                    gdelt_snapshot=gdelt_snapshot,
                    estimated_probability=trade_estimated_probability,
                    what_could_go_wrong=trade_what_could_go_wrong,
                    microstructure_snapshot=microstructure_snapshot,
                )

            # Store in trade history for session persistence
            trade_msg = {
                "ticker": ticker,
                "side": tool_input["side"],
                "action": tool_input.get("action", "buy"),
                "contracts": contracts,
                "price_cents": entry_price,
                "limit_price_cents": result.limit_price_cents,
                "order_id": result.order_id,
                "order_status": result.order_status or "unknown",
                "reasoning": tool_input["reasoning"][:500],
                "timestamp": time.strftime("%H:%M:%S"),
            }
            self._trade_history.append(trade_msg)

        # Include think warning in result if present
        trade_result = result.to_dict()
        if think_warning:
            trade_result["think_warning"] = think_warning
        return trade_result

    async def _execute_think(self, tool_input: Dict) -> Any:
        """Pre-trade structured analysis tool - forces structured reasoning."""
        signal_analysis = tool_input.get("signal_analysis", "")
        strategy_check = tool_input.get("strategy_check", "")
        risk_assessment = tool_input.get("risk_assessment", "")
        decision = tool_input.get("decision", "")
        reflection = tool_input.get("reflection", "")
        estimated_probability = tool_input.get("estimated_probability")
        what_could_go_wrong = tool_input.get("what_could_go_wrong", "")

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
        if estimated_probability is not None:
            formatted_analysis += f"\n**Est. Probability (YES)**: {estimated_probability}%"
        if what_could_go_wrong:
            formatted_analysis += f"\n**What Could Go Wrong**: {what_could_go_wrong}"
        if reflection:
            formatted_analysis += f"\n**Notes**: {reflection}"

        logger.info(f"[deep_agent.agent] Think: decision={decision}, signal={signal_analysis[:100]}...")

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
                "estimated_probability": estimated_probability,
                "what_could_go_wrong": what_could_go_wrong,
            }
        }

        self._thinking_history.append(thinking_msg)

        # Update think enforcement tracking
        self._last_think_decision = decision
        self._last_think_timestamp = time.time()
        self._last_think_estimated_probability = self._parse_probability(estimated_probability)
        self._last_think_what_could_go_wrong = what_could_go_wrong or None

        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_thinking", thinking_msg)

        return {
            "recorded": True,
            "decision": decision,
            "proceed_to_trade": decision == "TRADE",
        }

    async def _execute_reflect(self, tool_input: Dict) -> Any:
        """Structured post-trade reflection tool."""
        self._cycle_learnings_written = True
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

        if mistake and mistake.strip():
            mistake_entry = (
                f"\n## {timestamp} - {trade_ticker}\n"
                f"- {mistake}\n"
            )
            await self._tools.append_memory("mistakes.md", mistake_entry)
            results["appended_to"].append("mistakes.md")

        if pattern and pattern.strip():
            pattern_entry = (
                f"\n## {timestamp} - {trade_ticker}\n"
                f"- {pattern}\n"
            )
            await self._tools.append_memory("patterns.md", pattern_entry)
            results["appended_to"].append("patterns.md")

        results["strategy_update_needed"] = strategy_update_needed

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

    async def _execute_evaluate_extractions(self, tool_input: Dict) -> Any:
        """Score extraction accuracy after trade settlement."""
        self._tools._tool_calls["evaluate_extractions"] = self._tools._tool_calls.get("evaluate_extractions", 0) + 1
        trade_ticker = tool_input["trade_ticker"]
        trade_outcome = tool_input["trade_outcome"]
        evaluations = tool_input.get("evaluations", [])

        accuracy_map = {
            "accurate": 1.0,
            "partially_accurate": 0.7,
            "inaccurate": 0.2,
            "noise": 0.0,
        }

        supabase = self._tools._get_supabase()
        if not supabase:
            return {"error": "Supabase not available"}

        results_summary = {
            "trade_ticker": trade_ticker,
            "trade_outcome": trade_outcome,
            "evaluated": 0,
            "promoted_examples": 0,
            "evaluations": [],
        }

        for eval_item in evaluations:
            ext_id = eval_item.get("extraction_id", "")
            accuracy = eval_item.get("accuracy", "noise")
            quality_score = accuracy_map.get(accuracy, 0.0)

            try:
                supabase.table("extractions") \
                    .update({"quality_score": quality_score}) \
                    .eq("id", ext_id) \
                    .execute()

                results_summary["evaluated"] += 1
                results_summary["evaluations"].append({
                    "extraction_id": ext_id,
                    "accuracy": accuracy,
                    "quality_score": quality_score,
                })

                # Auto-promote: accurate extraction + winning trade -> example
                if accuracy == "accurate" and trade_outcome == "win":
                    ext_result = supabase.table("extractions") \
                        .select("*") \
                        .eq("id", ext_id) \
                        .execute()

                    if ext_result.data:
                        ext_row = ext_result.data[0]
                        event_tickers = ext_row.get("event_tickers", [])
                        event_ticker = event_tickers[0] if event_tickers else ""

                        example_data = {
                            "event_ticker": event_ticker,
                            "extraction_class": ext_row.get("extraction_class", "market_signal"),
                            "source_text": ext_row.get("extraction_text", ""),
                            "expected_output": {
                                "extraction_class": ext_row.get("extraction_class", ""),
                                "extraction_text": ext_row.get("extraction_text", ""),
                                "attributes": ext_row.get("attributes", {}),
                            },
                            "quality_score": 1.0,
                            "source": "real",
                            "created_at": datetime.now().isoformat(),
                        }

                        supabase.table("extraction_examples") \
                            .insert(example_data) \
                            .execute()

                        results_summary["promoted_examples"] += 1
                        logger.info(
                            f"[deep_agent.agent] Promoted extraction {ext_id} as example "
                            f"for {event_ticker}"
                        )

            except Exception as e:
                logger.warning(
                    f"[deep_agent.agent] Failed to evaluate extraction {ext_id}: {e}"
                )
                results_summary["evaluations"].append({
                    "extraction_id": ext_id,
                    "error": str(e),
                })

        logger.info(
            f"[deep_agent.agent] evaluate_extractions: {results_summary['evaluated']} evaluated, "
            f"{results_summary['promoted_examples']} promoted for {trade_ticker}"
        )
        return results_summary

    async def _execute_refine_event(self, tool_input: Dict) -> Any:
        """Push learnings back to extraction pipeline via event_configs."""
        self._tools._tool_calls["refine_event"] = self._tools._tool_calls.get("refine_event", 0) + 1
        event_ticker = tool_input["event_ticker"]
        what_works = tool_input["what_works"]
        what_fails = tool_input["what_fails"]
        watchlist_additions = tool_input.get("suggested_watchlist_additions", [])
        prompt_refinement = tool_input.get("suggested_prompt_refinement", "")

        supabase = self._tools._get_supabase()
        if not supabase:
            return {"error": "Supabase not available"}

        try:
            result = supabase.table("event_configs") \
                .select("*") \
                .eq("event_ticker", event_ticker) \
                .execute()

            if not result.data:
                return {
                    "error": f"No event_configs found for {event_ticker}. "
                    f"Call understand_event() first.",
                    "event_ticker": event_ticker,
                }

            config = result.data[0]
            current_version = config.get("research_version", 1)
            updates = {}

            if prompt_refinement:
                current_prompt = config.get("prompt_description", "")
                version_tag = f"\n\n[v{current_version + 1} refinement]: "
                updates["prompt_description"] = current_prompt + version_tag + prompt_refinement

            if watchlist_additions:
                watchlist = config.get("watchlist", {})
                entities = watchlist.get("entities", [])
                for entity in watchlist_additions:
                    if entity not in entities:
                        entities.append(entity)
                watchlist["entities"] = entities
                updates["watchlist"] = watchlist

            updates["research_version"] = current_version + 1
            updates["last_researched_at"] = datetime.now().isoformat()
            updates["updated_at"] = datetime.now().isoformat()

            supabase.table("event_configs") \
                .update(updates) \
                .eq("event_ticker", event_ticker) \
                .execute()

            logger.info(
                f"[deep_agent.agent] refine_event: updated {event_ticker} "
                f"to v{updates['research_version']} "
                f"(prompt_refined={bool(prompt_refinement)}, "
                f"watchlist_added={len(watchlist_additions)})"
            )

            return {
                "status": "refined",
                "event_ticker": event_ticker,
                "new_version": updates["research_version"],
                "prompt_refined": bool(prompt_refinement),
                "watchlist_entities_added": len(watchlist_additions),
                "what_works": what_works[:200],
                "what_fails": what_fails[:200],
            }

        except Exception as e:
            logger.error(f"[deep_agent.agent] refine_event error: {e}")
            return {"error": str(e), "event_ticker": event_ticker}

    async def _execute_get_extraction_quality(self, tool_input: Dict) -> Any:
        """Query extraction quality metrics per event."""
        self._tools._tool_calls["get_extraction_quality"] = self._tools._tool_calls.get("get_extraction_quality", 0) + 1
        event_ticker = tool_input.get("event_ticker")

        supabase = self._tools._get_supabase()
        if not supabase:
            return {"error": "Supabase not available"}

        try:
            query = supabase.table("extractions") \
                .select("event_tickers, quality_score") \
                .not_.is_("quality_score", "null")

            if event_ticker:
                query = query.contains("event_tickers", [event_ticker])

            result = query.execute()

            if not result.data:
                return {
                    "events": [],
                    "message": "No evaluated extractions found" + (
                        f" for {event_ticker}" if event_ticker else ""
                    ),
                }

            # Aggregate by event ticker
            event_metrics: Dict[str, Dict] = {}
            for row in result.data:
                score = row.get("quality_score", 0)
                event_tickers_list = row.get("event_tickers", [])
                for et in event_tickers_list:
                    if et not in event_metrics:
                        event_metrics[et] = {
                            "event_ticker": et,
                            "total_evaluated": 0,
                            "scores": [],
                            "accurate_count": 0,
                            "partially_accurate_count": 0,
                            "inaccurate_count": 0,
                            "noise_count": 0,
                        }
                    m = event_metrics[et]
                    m["total_evaluated"] += 1
                    m["scores"].append(score)

                    if score >= 0.9:
                        m["accurate_count"] += 1
                    elif score >= 0.5:
                        m["partially_accurate_count"] += 1
                    elif score >= 0.1:
                        m["inaccurate_count"] += 1
                    else:
                        m["noise_count"] += 1

            events = []
            for et, m in event_metrics.items():
                scores = m.pop("scores")
                m["avg_quality"] = round(sum(scores) / len(scores), 2) if scores else 0.0
                m["needs_refinement"] = m["avg_quality"] < 0.6
                events.append(m)

            events.sort(key=lambda e: e["avg_quality"])

            return {
                "events": events,
                "total_events": len(events),
                "message": f"{len(events)} events with extraction quality data",
            }

        except Exception as e:
            logger.error(f"[deep_agent.agent] get_extraction_quality error: {e}")
            return {"error": str(e)}

    # =========================================================================
    # Reflection & Memory Consolidation
    # =========================================================================

    async def _handle_sell_close(
        self,
        ticker: str,
        side: str,
        contracts: int,
        sell_price_cents: int,
        reasoning: str,
    ) -> None:
        """
        Handle a sell-to-close by settling the original pending buy trade immediately.

        When the agent sells contracts it owns, this finds the matching pending trade
        and settles it with realized P&L computed from (sell_price - entry_price) * contracts.
        This gives instant feedback without waiting for event resolution.

        Args:
            ticker: Market ticker that was sold
            side: Side of the contracts sold ("yes" or "no")
            contracts: Number of contracts sold
            sell_price_cents: Execution price of the sell order (cents)
            reasoning: Agent's reasoning for the exit
        """
        # Find matching pending trade by ticker + side
        matched_trade = None
        for trade_id, pending in self._reflection._pending_trades.items():
            if pending.ticker == ticker and pending.side.lower() == side.lower() and not pending.settled:
                matched_trade = pending
                break

        if not matched_trade:
            logger.warning(
                f"[deep_agent.agent] Sell close: no matching pending trade for "
                f"{ticker} {side} x{contracts} — skipping settlement"
            )
            return

        # Compute realized P&L: for YES contracts, profit = (sell - entry) * contracts
        # For NO contracts, same formula applies since both entry and sell are in the same basis
        pnl_cents = (sell_price_cents - matched_trade.entry_price_cents) * min(contracts, matched_trade.contracts)

        logger.info(
            f"[deep_agent.agent] Sell close: {ticker} {side} x{contracts} "
            f"entry={matched_trade.entry_price_cents}c sell={sell_price_cents}c "
            f"P&L=${pnl_cents / 100:.2f}"
        )

        # Build a synthetic settlement dict matching state_container format
        synthetic_settlement = {
            "ticker": ticker,
            "position": contracts,
            "side": side,
            "net_pnl": pnl_cents,
            "price_cents": sell_price_cents,
            "strategy_id": "deep_agent",
            "market_result": "sold",  # Not an event settlement — agent exited early
        }

        # Settle the trade through the normal reflection pipeline
        await self._reflection._handle_settlement(matched_trade, synthetic_settlement)

    async def _handle_reflection(self, trade: PendingTrade) -> Optional[ReflectionResult]:
        """Handle reflection callback from the reflection engine."""
        logger.info(f"[deep_agent.agent] Reflecting on trade: {trade.ticker} ({trade.result})")

        # Release event exposure now that the position has settled
        self._tools.release_event_exposure(
            trade.event_ticker, trade.contracts, trade.entry_price_cents
        )

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
            learning = self._extract_insight_from_response("learn")
            mistake = self._extract_insight_from_response("mistake")
            pattern = self._extract_insight_from_response("pattern")

            # Track consecutive losses for urgent consolidation
            # T4.1: Don't reset counter on unknown results — only reset on wins/break_even
            trade_result = trade.result or "unknown"
            if trade_result == "loss":
                self._consecutive_losses += 1
            elif trade_result in ("win", "break_even"):
                self._consecutive_losses = 0
            # "unknown" intentionally skips — preserves current loss streak count

            # Increment reflection count and check for consolidation trigger
            self._reflection_count += 1
            should_consolidate, trigger_reason = self._should_consolidate(trade)
            if should_consolidate:
                await self._trigger_consolidation(trigger_reason)

            # Persist volatile state for crash recovery
            self._save_session_state()

            # T4.2: Explicit None check — don't silently convert None to 0 (looks like break-even)
            pnl_cents = trade.pnl_cents if trade.pnl_cents is not None else 0
            if trade.pnl_cents is None:
                logger.warning(
                    f"[deep_agent.agent] Trade {trade.ticker} has None pnl_cents, using 0"
                )

            return ReflectionResult(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                result=trade_result,
                pnl_cents=pnl_cents,
                learning=learning,
                should_update_strategy=bool(pattern),
                mistake_identified=mistake,
                pattern_identified=pattern,
            )

        except Exception as e:
            logger.error(f"[deep_agent.agent] Error in reflection: {e}")
            # T4.3: Return minimal ReflectionResult with known trade data
            # instead of None (which loses all outcome data)
            return ReflectionResult(
                trade_id=trade.trade_id,
                ticker=trade.ticker,
                result=trade.result or "unknown",
                pnl_cents=trade.pnl_cents if trade.pnl_cents is not None else 0,
                learning=f"Reflection failed: {e}",
                should_update_strategy=False,
            )

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

### Consolidation Checklist (follow in order)
1. Call `read_memory("learnings.md")` to review recent insights
2. Call `read_memory("strategy.md")` to see current rules
3. **Contradiction check**: verify each existing rule still has evidence support — remove rules contradicted by recent trades
4. **Redundancy check**: merge duplicate or overlapping rules
5. Call `write_memory("strategy.md", ...)` with updated rules (MUST be under 3000 chars)
   - Net rule count should stay stable or DECREASE — distill, don't accumulate
6. **Promote strong rules**: If any rule is backed by 5+ profitable trades, promote it to `golden_rules.md` via `append_memory("golden_rules.md", ...)`
7. **Abstraction check**: Can any specific trade learnings be generalized into broader principles?
   Example: "KXBONDIOUT signals at 5+ sources profitable" → "Political personnel markets respond well to 5+ source signals"
   Record abstractions in `patterns.md` via `append_memory("patterns.md", ...)`
8. **Microstructure pattern review**: Call `get_microstructure()` to see current market state.
   Review your recent trades: did microstructure conditions (trade flow ratio, spread, whale orders) correlate with outcomes?
   Example patterns to look for:
   - "Trades where whale orders contradicted my signal → always lost"
   - "Wide spreads (>6c) at entry → worse outcomes even when direction correct"
   Record any discovered patterns in patterns.md.
9. Check extraction quality: call `get_extraction_quality()` for events with poor accuracy
10. For events with avg_quality < 0.6, call `refine_event()` with learnings

### Focus Areas
- New entry/exit rules discovered (with trade evidence)
- Position sizing adjustments based on outcomes
- Risk rules that worked or failed
- Remove stale rules that no longer apply
- Extraction accuracy feedback

Be selective: only add rules with clear evidence. Remove weak rules as much as adding new ones.
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

    # =========================================================================
    # Response Parsing Helpers
    # =========================================================================

    def _extract_insight_from_response(self, insight_type: str) -> Optional[str]:
        """Extract an insight of the given type from the last assistant message.

        Args:
            insight_type: "learn", "mistake", or "pattern"

        Returns:
            Extracted insight string. For "learn", returns a default if nothing found.
            For "mistake"/"pattern", returns None if nothing found.
        """
        default = "Reflected on trade outcome" if insight_type == "learn" else None

        for msg in reversed(self._conversation_history):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            insight = self._extract_insight_from_text(text, insight_type)
                            if insight != "Reflected on trade outcome" or insight_type == "learn":
                                return insight
                elif isinstance(content, str):
                    insight = self._extract_insight_from_text(content, insight_type)
                    if insight != "Reflected on trade outcome" or insight_type == "learn":
                        return insight
        return default

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

    @property
    def is_cycle_running(self) -> bool:
        """Check if the main cycle task is alive."""
        return (
            self._cycle_task is not None
            and not self._cycle_task.done()
        )

    def is_healthy(self) -> bool:
        """Check if agent is healthy."""
        if not self._running:
            return False

        # Check if cycle task is alive
        if not self.is_cycle_running:
            return False

        # Check if cycle is happening
        if self._last_cycle_at:
            time_since_cycle = time.time() - self._last_cycle_at
            if time_since_cycle > self._config.cycle_interval_seconds * 3:
                return False

        return True

    def _accumulate_external_tokens(
        self, input_tokens: int, output_tokens: int,
        cache_read: int = 0, cache_created: int = 0,
    ) -> None:
        """Accumulate token usage from external API calls (tools, distillation)."""
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cache_read_tokens += cache_read
        self._total_cache_created_tokens += cache_created
        # Also accumulate into cycle counters if we're mid-cycle
        self._cycle_input_tokens += input_tokens
        self._cycle_output_tokens += output_tokens
        self._cycle_cache_read_tokens += cache_read
        self._cycle_cache_created_tokens += cache_created
        self._cycle_api_calls += 1

    def _calculate_cost(self, input_tokens: int, output_tokens: int,
                        cache_read_tokens: int, cache_created_tokens: int) -> Dict[str, float]:
        """Calculate USD cost from token counts using model pricing."""
        p = self._model_pricing
        input_cost = input_tokens * p["input"]
        output_cost = output_tokens * p["output"]
        cache_read_cost = cache_read_tokens * p["cache_read"]
        cache_write_cost = cache_created_tokens * p["cache_write"]
        total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost
        # What cache reads would have cost at full input price
        cache_savings = cache_read_tokens * (p["input"] - p["cache_read"])
        return {
            "total_cost_usd": round(total_cost, 6),
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "cache_read_cost_usd": round(cache_read_cost, 6),
            "cache_write_cost_usd": round(cache_write_cost, 6),
            "cache_savings_usd": round(cache_savings, 6),
        }

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
            "cost": self._calculate_cost(
                self._total_input_tokens, self._total_output_tokens,
                self._total_cache_read_tokens, self._total_cache_created_tokens,
            ),
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

    def _get_todos_for_snapshot(self) -> List[Dict]:
        """Read todos.json synchronously for snapshot inclusion."""
        try:
            todos_path = self._tools._memory_dir / "todos.json"
            if todos_path.exists():
                data = json.loads(todos_path.read_text(encoding="utf-8"))
                return data.get("items", [])
        except Exception:
            pass
        return []

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
            "cost_data": {
                "model": self._config.model,
                "session_cost": self._calculate_cost(
                    self._total_input_tokens, self._total_output_tokens,
                    self._total_cache_read_tokens, self._total_cache_created_tokens,
                ),
                "session_tokens": {
                    "input": self._total_input_tokens,
                    "output": self._total_output_tokens,
                    "cache_read": self._total_cache_read_tokens,
                    "cache_created": self._total_cache_created_tokens,
                },
                "cycle_count": self._cycle_count,
            },
            # Cached extraction signals from last cycle pre-fetch (for snapshot persistence)
            "extraction_signals": self._cached_extraction_signals[:20],
            # GDELT query history for News Intelligence panel persistence
            "gdelt_queries": [
                {
                    "search_terms": q.get("search_terms", []),
                    "article_count": q.get("article_count", 0),
                    "source_diversity": q.get("source_diversity", 0),
                    "avg_tone": q.get("avg_tone", 0),
                    "timestamp": q.get("timestamp", 0),
                }
                for q in list(self._tools._recent_gdelt_queries)[:10]
            ] if hasattr(self._tools, '_recent_gdelt_queries') else [],
            # Full GDELT results for News Intelligence panel (new client restore)
            "gdelt_results": list(self._tools._recent_gdelt_results)[:5]
                if hasattr(self._tools, '_recent_gdelt_results') else [],
            # Agent TODO task list for frontend display
            "todos": self._get_todos_for_snapshot(),
        }
