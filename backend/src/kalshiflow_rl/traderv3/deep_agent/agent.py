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
import shutil
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from anthropic import AsyncAnthropic

from .tools import DeepAgentTools
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
}


def _get_model_pricing(model: str) -> Dict[str, float]:
    """Resolve pricing by model prefix, falling back to Sonnet 4."""
    for prefix, pricing in MODEL_PRICING.items():
        if model.startswith(prefix):
            return pricing
    return MODEL_PRICING["claude-sonnet-4"]


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



# System prompt is now built dynamically via build_system_prompt() from prompts.py
# This allows modular editing of prompt sections


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

        # Wire circuit breaker callback and config into tools for preflight_check
        # (the callback is set after circuit breaker state is initialized below)
        self._tools._min_spread_cents = self._config.min_spread_cents
        self._tools._max_event_exposure_cents = self._config.max_event_exposure_cents

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
        self._max_history_length = 40  # ~8-10 cycles of history (expanded for reasoning continuity)

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

        # Wire circuit breaker checker callback into tools for preflight_check
        self._tools._circuit_breaker_checker = self._is_ticker_blacklisted

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

        # Reflection count for consolidation trigger
        self._reflection_count = 0
        self._consecutive_losses = 0  # Track loss streaks for urgent consolidation

        # Tool definitions for Claude
        self._tool_definitions = self._build_tool_definitions()

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
                "name": "preflight_check",
                "description": "RECOMMENDED before trade(). Bundled pre-trade safety check that combines market data + event context + all safety validations in ONE call. Returns prices, spread, estimated limit price, event risk, exposure cap status, circuit breaker status, and a tradeable go/no-go flag. Saves 2 API round-trips vs calling get_markets() + get_event_context() separately.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ticker": {
                            "type": "string",
                            "description": "Market ticker to check (e.g., 'KXBONDIOUT-25FEB07-T42.5')"
                        },
                        "side": {
                            "type": "string",
                            "enum": ["yes", "no"],
                            "description": "Intended trade side"
                        },
                        "contracts": {
                            "type": "integer",
                            "description": "Intended number of contracts"
                        },
                        "execution_strategy": {
                            "type": "string",
                            "enum": ["aggressive", "moderate", "passive"],
                            "description": "Pricing strategy for limit price computation (default: aggressive)"
                        }
                    },
                    "required": ["ticker", "side", "contracts"]
                }
            },
            {
                "name": "trade",
                "description": "Execute a trade (buy to open, sell to close). Choose execution_strategy based on signal quality: aggressive for STRONG signals (crosses spread, fastest fill), moderate for MODERATE signals (midpoint pricing, saves spread), passive for speculative entries (near bid, cheapest but may not fill). Use action='sell' to exit positions you own.",
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
                            "description": "YES or NO contracts"
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
                        "action": {
                            "type": "string",
                            "enum": ["buy", "sell"],
                            "description": "buy=open new position (default), sell=close existing position by selling contracts you own."
                        },
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
                "name": "get_true_performance",
                "description": "Get TRUE trading performance from Kalshi API. Returns strategy-filtered P&L (realized + unrealized), positions with live unrealized P&L, settlements, win rate, and per-event breakdown. Use this instead of scorecard for performance assessment — it's the ground truth. Includes unrealized P&L from open positions for immediate feedback before settlement. Idempotent — safe to call anytime.",
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
                "name": "evaluate_extractions",
                "description": "Score extraction accuracy after trade settlement. Updates quality_score on extractions and auto-promotes accurate extractions from winning trades as examples for future extraction calls. Call this during reflection after a trade settles.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "trade_ticker": {
                            "type": "string",
                            "description": "Market ticker of the settled trade"
                        },
                        "trade_outcome": {
                            "type": "string",
                            "enum": ["win", "loss", "break_even"],
                            "description": "Outcome of the trade"
                        },
                        "evaluations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "extraction_id": {
                                        "type": "string",
                                        "description": "ID of the extraction to evaluate"
                                    },
                                    "accuracy": {
                                        "type": "string",
                                        "enum": ["accurate", "partially_accurate", "inaccurate", "noise"],
                                        "description": "How accurate was this extraction signal?"
                                    },
                                    "note": {
                                        "type": "string",
                                        "description": "Optional note about why this accuracy rating"
                                    }
                                },
                                "required": ["extraction_id", "accuracy"]
                            },
                            "description": "Array of extraction evaluations"
                        }
                    },
                    "required": ["trade_ticker", "trade_outcome", "evaluations"]
                }
            },
            {
                "name": "refine_event",
                "description": "Push learnings back to the extraction pipeline for a specific event. Updates event_configs with prompt refinements and watchlist additions. Call during consolidation when you've identified what works/fails for an event's extractions.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_ticker": {
                            "type": "string",
                            "description": "Event ticker to refine extraction config for"
                        },
                        "what_works": {
                            "type": "string",
                            "description": "What extraction patterns have been accurate for this event"
                        },
                        "what_fails": {
                            "type": "string",
                            "description": "What extraction patterns have been inaccurate or noisy"
                        },
                        "suggested_watchlist_additions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional: new entities/keywords to add to the watchlist"
                        },
                        "suggested_prompt_refinement": {
                            "type": "string",
                            "description": "Optional: additional extraction instructions to append to the event's prompt"
                        }
                    },
                    "required": ["event_ticker", "what_works", "what_fails"]
                }
            },
            {
                "name": "get_extraction_quality",
                "description": "Get extraction quality metrics per event. Shows total evaluated, average quality, accurate/inaccurate counts. Use during consolidation to identify events with poor extraction accuracy that need refine_event().",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_ticker": {
                            "type": "string",
                            "description": "Optional: filter by specific event ticker"
                        }
                    },
                    "required": []
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

        # Distill learnings from previous sessions into strategy
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

        Reads up to 10 most recent archived sessions (bounded by _prune_archives),
        includes scorecard by_event stats as evidence, and preserves golden rules.
        Uses Claude to update strategy.md with evidence-backed rules.
        """
        archive_dir = self._memory_dir / "memory_archive"
        if not archive_dir.exists():
            logger.info("[deep_agent.agent] No memory_archive/ — skipping distillation")
            return

        # Find up to 10 most recent archive directories (bounded by pruning)
        archive_dirs = sorted(
            [d for d in archive_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )[:10]

        if not archive_dirs:
            logger.info("[deep_agent.agent] No archived sessions — skipping distillation")
            return

        # Collect archived content
        archived_content = []
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

        if not archived_content:
            logger.info("[deep_agent.agent] Archived files are empty — skipping distillation")
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

        distill_prompt = f"""You are updating a trading strategy based on learnings from previous sessions.

## Current Strategy
{current_strategy}
{golden_section}{by_event_section}
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

            updated_strategy = response.content[0].text.strip()

            # Sanity check: must be non-empty and contain key markers
            if len(updated_strategy) > 100 and "strategy" in updated_strategy.lower():
                # Apply version counter and cap via tools helper
                updated_strategy = self._tools._apply_strategy_cap(strategy_path, updated_strategy)
                strategy_path.write_text(updated_strategy, encoding="utf-8")
                logger.info(
                    f"[deep_agent.agent] Strategy distilled from {len(archive_dirs)} archived sessions "
                    f"({len(updated_strategy)} chars)"
                )
            else:
                logger.warning("[deep_agent.agent] Distillation returned suspicious output, keeping current strategy")

        except asyncio.TimeoutError:
            logger.warning("[deep_agent.agent] Distillation timed out after 45s, keeping current strategy")
        except Exception as e:
            logger.warning(f"[deep_agent.agent] Distillation failed (non-fatal): {e}")

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

        logger.info(f"[deep_agent.agent] === Starting cycle {self._cycle_count} ===")

        # Broadcast cycle start
        if self._ws_manager:
            await self._ws_manager.broadcast_message("deep_agent_cycle", {
                "cycle": self._cycle_count,
                "phase": "starting",
                "timestamp": time.strftime("%H:%M:%S"),
            })

        # PRE-CHECK: Are there actionable signals? (cheap Supabase query, ~50ms, zero API tokens)
        pre_check_result = await self._tools.get_extraction_signals(window_hours=4.0, limit=20)
        pre_fetched_signals = pre_check_result.get("signals", [])

        # Also check if we have open positions to monitor (always run cycle if positions exist)
        has_positions = (
            self._state_container
            and self._state_container.get_trading_summary().get("position_count", 0) > 0
        )

        # Bootstrap any newly discovered events (cheap DB check, creates event_configs for new events)
        await self._bootstrap_new_events()

        if not pre_fetched_signals and not has_positions:
            logger.info(f"[deep_agent.agent] Cycle {self._cycle_count}: No signals, no positions - skipping Claude call")
            if self._ws_manager:
                await self._ws_manager.broadcast_message("deep_agent_cycle", {
                    "cycle": self._cycle_count,
                    "phase": "skipped_no_signals",
                    "timestamp": time.strftime("%H:%M:%S"),
                })
            return

        # Signals exist OR positions to manage — run full cycle
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

        # Auto-journal fallback: if agent didn't call write_cycle_summary, generate minimal entry
        try:
            await self._auto_journal_fallback()
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Auto-journal fallback failed: {e}")

        # Compact history between cycles to save tokens on next call
        self._compact_history_between_cycles()

        # Persist session state for crash recovery
        self._save_session_state()

    async def _auto_journal_fallback(self) -> None:
        """Auto-generate a minimal journal entry if agent didn't call write_cycle_summary."""
        # Check if write_cycle_summary was called this cycle by scanning recent history
        for msg in reversed(self._conversation_history[-20:]):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if (isinstance(block, dict) and block.get("type") == "tool_use"
                                and block.get("name") == "write_cycle_summary"):
                            return  # Agent already wrote journal

        # Generate minimal entry from available state
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        decision = self._last_think_decision or "no_think"
        entry = (
            f"\n## Cycle {self._cycle_count} - {timestamp} [auto]\n"
            f"- **Decision**: {decision}\n"
            f"- **Trades this session**: {self._trades_executed}\n"
        )
        await self._tools.append_memory("cycle_journal.md", entry)

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

        # Load mistakes and patterns EVERY cycle, priority-ordered by confidence tags
        # Char limits scale down after cycle 10 to save tokens on mature sessions
        memory_context_str = ""
        try:
            mistakes = await self._tools.read_memory("mistakes.md")
            if mistakes and len(mistakes) > 50:
                mistakes_limit = 800 if self._cycle_count <= 10 else 400
                mistakes_preview = self._priority_load_memory(mistakes, mistakes_limit)
                memory_context_str += f"\n### Mistakes to Avoid (from memory)\n{mistakes_preview}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not load mistakes: {e}")

        try:
            patterns = await self._tools.read_memory("patterns.md")
            if patterns and len(patterns) > 50:
                patterns_limit = 800 if self._cycle_count <= 10 else 400
                patterns_preview = self._priority_load_memory(patterns, patterns_limit)
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
        # Always include persistent cross-session summary (one line, cheap)
        try:
            historical_summary = self._reflection.get_scorecard_summary()
            if historical_summary:
                scorecard_str = f"\n### Cross-Session Performance\n{historical_summary}\n"
        except Exception as e:
            logger.debug(f"[deep_agent.agent] Could not build historical scorecard: {e}")
        # Add detailed session scorecard every 10 cycles
        if self._cycle_count % 10 == 0:
            try:
                scorecard = self._reflection._build_performance_scorecard()
                if scorecard:
                    scorecard_str += f"\n{scorecard}\n"
            except Exception as e:
                logger.debug(f"[deep_agent.agent] Could not build scorecard: {e}")

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
                "\nSignals above are pre-loaded. Use preflight_check() + think() before trading."
            )
            signals_section = "\n".join(signals_lines) + "\n"

            instructions = """### Instructions
1. **Signals are provided above** — review them for tradeable opportunities
2. For promising signals, call **preflight_check(ticker, side, contracts)** for bundled safety check
3. If preflight returns `tradeable: true`, call **think()** for structured pre-trade analysis
4. Use your strategy.md criteria to decide: TRADE, WAIT, or PASS
5. If your criteria are met: trade() — execute decisively
6. If no actionable signals, PASS and wait for next cycle
7. You can call get_extraction_signals() to re-query with different filters if needed

**IMPORTANT**: Apply your learned criteria from strategy.md — refine them after every trade."""
        else:
            signals_section = "\n### No new signals this cycle. Focus on managing existing positions — check if any should be exited (sell) or held.\n"
            instructions = """### Instructions
1. No new signals available — focus on managing existing positions
2. Check open positions: should any be exited via trade(action="sell")? Look for contradicting signals or profit-taking opportunities.
3. You can call get_extraction_signals() to re-query with different filters if needed
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
{event_exposure_str}{open_positions_str}{strategy_excerpt}{golden_rules_str}{market_knowledge_str}{journal_str}{memory_context_str}{scorecard_str}{signals_section}
### Target Events
{target_events_str}

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

                result_block = {
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(result) if isinstance(result, (dict, list)) else str(result),
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
            if tool_name == "get_extraction_signals":
                result = await self._tools.get_extraction_signals(
                    market_ticker=tool_input.get("market_ticker"),
                    event_ticker=tool_input.get("event_ticker"),
                    window_hours=tool_input.get("hours", 4),
                    limit=tool_input.get("limit", 20),
                )
                return result

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
                    action=tool_input.get("action", "buy"),
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

                    # T1.2: Use limit_price_cents as fallback instead of fabricated 50c
                    entry_price = result.price_cents or result.limit_price_cents or 50

                    # Snapshot extractions + GDELT queries that drove this trade (for learning loop)
                    extraction_ids, extraction_snapshot, gdelt_snapshot = await self._snapshot_trade_extractions(ticker)

                    self._reflection.record_trade(
                        trade_id=result.order_id or str(uuid.uuid4()),
                        ticker=ticker,
                        event_ticker=event_ticker,
                        side=tool_input["side"],
                        contracts=contracts,
                        entry_price_cents=entry_price,
                        reasoning=tool_input["reasoning"],
                        extraction_ids=extraction_ids,
                        extraction_snapshot=extraction_snapshot,
                        gdelt_snapshot=gdelt_snapshot,
                        estimated_probability=self._last_think_estimated_probability,
                        what_could_go_wrong=self._last_think_what_could_go_wrong,
                    )

                    # Store in trade history for session persistence
                    trade_msg = {
                        "ticker": ticker,
                        "side": tool_input["side"],
                        "action": tool_input.get("action", "buy"),
                        "contracts": contracts,
                        "price_cents": entry_price,
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

            elif tool_name == "get_true_performance":
                perf = await self._tools.get_true_performance()
                return perf.to_dict()

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

            elif tool_name == "preflight_check":
                result = await self._tools.preflight_check(
                    ticker=tool_input["ticker"],
                    side=tool_input["side"],
                    contracts=min(tool_input["contracts"], self._config.max_contracts_per_trade),
                    execution_strategy=tool_input.get("execution_strategy", "aggressive"),
                )
                return result

            elif tool_name == "understand_event":
                result = await self._tools.understand_event(
                    event_ticker=tool_input["event_ticker"],
                    force_refresh=tool_input.get("force_refresh", False),
                )
                return result

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
                estimated_probability = tool_input.get("estimated_probability")  # 0-100 or None
                what_could_go_wrong = tool_input.get("what_could_go_wrong", "")  # Required for TRADE

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
                        "estimated_probability": estimated_probability,
                        "what_could_go_wrong": what_could_go_wrong,
                    }
                }

                # Store in history for session persistence
                self._thinking_history.append(thinking_msg)

                # Update think enforcement tracking
                self._last_think_decision = decision
                self._last_think_timestamp = time.time()
                self._last_think_estimated_probability = int(estimated_probability) if estimated_probability is not None else None
                self._last_think_what_could_go_wrong = what_could_go_wrong or None

                # Broadcast to connected WebSocket clients
                if self._ws_manager:
                    await self._ws_manager.broadcast_message("deep_agent_thinking", thinking_msg)

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

            elif tool_name == "write_cycle_summary":
                # Agent-written cycle journal entry
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

            elif tool_name == "evaluate_extractions":
                # Score extraction accuracy after trade settlement
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
                    note = eval_item.get("note", "")
                    quality_score = accuracy_map.get(accuracy, 0.0)

                    try:
                        # Update quality_score on the extraction
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
                            # Fetch the full extraction to build the example
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

            elif tool_name == "refine_event":
                # Push learnings back to extraction pipeline via event_configs
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
                    # Fetch current event_configs
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

                    # Append prompt refinement if provided
                    if prompt_refinement:
                        current_prompt = config.get("prompt_description", "")
                        version_tag = f"\n\n[v{current_version + 1} refinement]: "
                        updates["prompt_description"] = current_prompt + version_tag + prompt_refinement

                    # Merge watchlist additions
                    if watchlist_additions:
                        watchlist = config.get("watchlist", {})
                        entities = watchlist.get("entities", [])
                        for entity in watchlist_additions:
                            if entity not in entities:
                                entities.append(entity)
                        watchlist["entities"] = entities
                        updates["watchlist"] = watchlist

                    # Always bump version and timestamp
                    updates["research_version"] = current_version + 1
                    updates["last_researched_at"] = datetime.now().isoformat()
                    updates["updated_at"] = datetime.now().isoformat()

                    # UPSERT
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

            elif tool_name == "get_extraction_quality":
                # Query extraction quality metrics per event
                self._tools._tool_calls["get_extraction_quality"] = self._tools._tool_calls.get("get_extraction_quality", 0) + 1
                event_ticker = tool_input.get("event_ticker")

                supabase = self._tools._get_supabase()
                if not supabase:
                    return {"error": "Supabase not available"}

                try:
                    # Query evaluated extractions (those with quality_score set)
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

                    # Finalize: compute averages, remove raw scores
                    events = []
                    for et, m in event_metrics.items():
                        scores = m.pop("scores")
                        m["avg_quality"] = round(sum(scores) / len(scores), 2) if scores else 0.0
                        m["needs_refinement"] = m["avg_quality"] < 0.6
                        events.append(m)

                    # Sort by avg_quality ascending (worst first)
                    events.sort(key=lambda e: e["avg_quality"])

                    return {
                        "events": events,
                        "total_events": len(events),
                        "message": f"{len(events)} events with extraction quality data",
                    }

                except Exception as e:
                    logger.error(f"[deep_agent.agent] get_extraction_quality error: {e}")
                    return {"error": str(e)}

            elif tool_name == "get_reddit_daily_digest":
                result = await self._tools.get_reddit_daily_digest(
                    force_refresh=tool_input.get("force_refresh", False),
                )
                return result

            elif tool_name == "query_gdelt_news":
                result = await self._tools.query_gdelt_news(
                    search_terms=tool_input.get("search_terms", []),
                    window_hours=tool_input.get("window_hours"),
                    tone_filter=tool_input.get("tone_filter"),
                    source_filter=tool_input.get("source_filter"),
                    limit=tool_input.get("limit"),
                )
                return result

            elif tool_name == "query_gdelt_events":
                result = await self._tools.query_gdelt_events(
                    actor_names=tool_input.get("actor_names", []),
                    country_filter=tool_input.get("country_filter"),
                    window_hours=tool_input.get("window_hours"),
                    limit=tool_input.get("limit"),
                )
                return result

            elif tool_name == "search_gdelt_articles":
                result = await self._tools.search_gdelt_articles(
                    search_terms=tool_input.get("search_terms", []),
                    timespan=tool_input.get("timespan"),
                    tone_filter=tool_input.get("tone_filter"),
                    max_records=tool_input.get("max_records"),
                    sort=tool_input.get("sort", "datedesc"),
                )
                return result

            elif tool_name == "get_gdelt_volume_timeline":
                result = await self._tools.get_gdelt_volume_timeline(
                    search_terms=tool_input.get("search_terms", []),
                    timespan=tool_input.get("timespan"),
                    tone_filter=tool_input.get("tone_filter"),
                )
                return result

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"[deep_agent.agent] Error executing {tool_name}: {e}")
            return {"error": str(e)}

    # =========================================================================
    # Reflection & Memory Consolidation
    # =========================================================================

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
8. Check extraction quality: call `get_extraction_quality()` for events with poor accuracy
9. For events with avg_quality < 0.6, call `refine_event()` with learnings

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
            "signal_lifecycle": {},  # Removed: extraction signals tracked via extractions table
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
        }
