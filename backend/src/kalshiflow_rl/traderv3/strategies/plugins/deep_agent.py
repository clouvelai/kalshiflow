"""
Deep Agent Strategy Plugin - Self-improving trading agent integration.

This strategy integrates the SelfImprovingAgent into the V3 Trader system,
providing the observe-act-reflect loop for autonomous trading.

Key Features:
- Autonomous trading with memory and learning
- Reddit signal monitoring (optional)
- Real-time streaming to frontend
- Reflection on trade outcomes
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

from ..protocol import Strategy, StrategyContext
from ..registry import StrategyRegistry
from ...deep_agent import SelfImprovingAgent, DeepAgentConfig
from ...core.events import EventType
from ...services.event_position_tracker import EventPositionTracker

if TYPE_CHECKING:
    from ...core.event_bus import EventBus
    from ...core.websocket_manager import V3WebSocketManager
    from ...services.trading_decision_service import TradingDecisionService
    from ...core.state_container import V3StateContainer
    from ...state.tracked_markets import TrackedMarketsState

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.deep_agent")


@StrategyRegistry.register("deep_agent")
class DeepAgentStrategy:
    """
    Strategy that runs the self-improving deep agent.

    The deep agent operates autonomously with an observe-act-reflect loop,
    learning from trade outcomes and updating its strategy over time.

    This strategy plugin:
    1. Initializes the SelfImprovingAgent with context
    2. Optionally starts Reddit signal monitoring
    3. Manages agent lifecycle and health
    4. Provides consolidated view for frontend
    """

    name = "deep_agent"
    display_name = "Deep Agent (Self-Improving)"
    subscribed_events: Set[EventType] = set()  # Agent manages its own event handling

    def __init__(self):
        """Initialize the strategy."""
        self._context: Optional[StrategyContext] = None
        self._agent: Optional[SelfImprovingAgent] = None
        self._event_position_tracker: Optional[EventPositionTracker] = None
        self._running = False
        self._started_at: Optional[float] = None

        # Stats
        self._cycles_run = 0
        self._trades_executed = 0
        self._reflections_completed = 0

    async def start(self, context: StrategyContext) -> None:
        """
        Start the deep agent strategy.

        Args:
            context: Strategy context with all services
        """
        if self._running:
            logger.warning("[deep_agent] Already running")
            return

        logger.info("[deep_agent] start() called with context:")
        logger.info("[deep_agent]   - websocket_manager: %s", context.websocket_manager is not None)
        logger.info("[deep_agent]   - trading_client_integration: %s", context.trading_client_integration is not None)
        logger.info("[deep_agent]   - state_container: %s", context.state_container is not None)
        logger.info("[deep_agent]   - tracked_markets: %s", context.tracked_markets is not None)
        logger.info("[deep_agent]   - config: %s", context.config)

        logger.info("[deep_agent] Starting Deep Agent Strategy")
        self._context = context
        self._running = True
        self._started_at = time.time()

        # Load configuration from YAML or use defaults
        config = self._load_config(context.config)

        # Create EventPositionTracker for event-level risk awareness
        # This helps the agent understand mutual exclusivity and correlated positions
        if context.tracked_markets and context.state_container:
            try:
                # Get V3Config from load_config() which reads from environment
                from ...config.environment import load_config
                v3_config = load_config()
                self._event_position_tracker = EventPositionTracker(
                    tracked_markets=context.tracked_markets,
                    state_container=context.state_container,
                    config=v3_config,
                )
                logger.info("[deep_agent] EventPositionTracker created for event awareness")
            except Exception as e:
                logger.warning(f"[deep_agent] Could not create EventPositionTracker: {e}")
                self._event_position_tracker = None
        else:
            logger.warning("[deep_agent] Missing dependencies for EventPositionTracker")

        # Initialize the self-improving agent with event awareness
        logger.info("[deep_agent] Creating SelfImprovingAgent with:")
        logger.info("[deep_agent]   - trading_client: %s", context.trading_client_integration is not None)
        logger.info("[deep_agent]   - state_container: %s", context.state_container is not None)
        logger.info("[deep_agent]   - websocket_manager: %s", context.websocket_manager is not None)
        logger.info("[deep_agent]   - tracked_markets: %s", context.tracked_markets is not None)
        logger.info("[deep_agent]   - event_position_tracker: %s", self._event_position_tracker is not None)

        self._agent = SelfImprovingAgent(
            trading_client=context.trading_client_integration,
            state_container=context.state_container,
            websocket_manager=context.websocket_manager,
            config=config,
            tracked_markets=context.tracked_markets,
            event_position_tracker=self._event_position_tracker,
        )
        logger.info("[deep_agent] SelfImprovingAgent created")

        # Start the agent
        logger.info("[deep_agent] About to call agent.start()")
        await self._agent.start()
        logger.info("[deep_agent] agent.start() completed")

        # Wire agent to websocket manager for session persistence (snapshot on connect)
        if context.websocket_manager:
            context.websocket_manager.set_deep_agent(self._agent)
            logger.info("[deep_agent] Agent wired to websocket manager for session persistence")

        # Note: Reddit monitoring is now handled by the entity pipeline:
        # - RedditEntityAgent: Extracts entities from Reddit with market-led normalization
        # - PriceImpactAgent: Transforms sentiment to price impact signals
        # - PriceImpactStore: In-memory cache queried by DeepAgent tools
        # These agents are started by V3Coordinator when entity_system_enabled=True

        logger.info("[deep_agent] Strategy started successfully")

    async def stop(self) -> None:
        """Stop the deep agent strategy."""
        if not self._running:
            return

        logger.info("[deep_agent] Stopping Deep Agent Strategy")
        self._running = False

        # Stop the agent
        if self._agent:
            await self._agent.stop()
            self._agent = None

        # Clean up event position tracker
        self._event_position_tracker = None

        logger.info("[deep_agent] Strategy stopped")

    def is_healthy(self) -> bool:
        """Check if strategy is healthy."""
        if not self._running or not self._agent:
            return False
        return self._agent.is_healthy()

    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics for monitoring."""
        uptime = time.time() - self._started_at if self._started_at else 0

        agent_stats = {}
        if self._agent:
            agent_stats = self._agent.get_stats()

        event_tracker_stats = {}
        if self._event_position_tracker:
            event_tracker_stats = self._event_position_tracker.get_stats()

        return {
            "name": self.name,
            "display_name": self.display_name,
            "running": self._running,
            "healthy": self.is_healthy(),
            "uptime_seconds": uptime,
            "cycles_run": agent_stats.get("cycle_count", 0),
            "trades_executed": agent_stats.get("trades_executed", 0),
            "reflections_completed": agent_stats.get("reflection_stats", {}).get("total_reflections", 0),
            "win_rate": agent_stats.get("reflection_stats", {}).get("win_rate", 0.0),
            "tool_stats": agent_stats.get("tool_stats", {}),
            # Event awareness
            "event_tracker_enabled": self._event_position_tracker is not None,
            "event_tracker_stats": event_tracker_stats,
            # Config info
            "target_events": agent_stats.get("config", {}).get("target_events", []),
            "cycle_interval": agent_stats.get("config", {}).get("cycle_interval_seconds", 60),
        }

    def get_consolidated_view(self) -> Dict[str, Any]:
        """Get consolidated view for frontend display."""
        stats = self.get_stats()

        agent_view = {}
        if self._agent:
            agent_view = self._agent.get_consolidated_view()

        # Get event exposure info for display
        event_exposure = {}
        if self._event_position_tracker:
            event_exposure = self._event_position_tracker.get_event_groups_for_broadcast()

        return {
            "status": "active" if self._running else "stopped",
            "strategy_name": self.display_name,
            # Session metrics
            "session_pnl_cents": 0,  # TODO: Get from agent
            "trades_count": stats["trades_executed"],
            "win_rate": stats["win_rate"],
            # Cycle info
            "cycle_count": stats["cycles_run"],
            "last_cycle_at": agent_view.get("last_cycle_at"),
            # Pending trades
            "pending_trades": agent_view.get("pending_trades", []),
            # Recent reflections/learnings
            "recent_reflections": agent_view.get("recent_reflections", []),
            # Tool usage
            "tool_stats": stats["tool_stats"],
            # Memory
            "memory_dir": agent_view.get("memory_dir"),
            "target_events": agent_view.get("target_events", []),
            # Event awareness
            "event_tracker_enabled": stats["event_tracker_enabled"],
            "event_exposure": event_exposure,
        }

    def _load_config(self, strategy_config: Optional[Any]) -> DeepAgentConfig:
        """Load agent configuration from YAML config."""
        config = DeepAgentConfig()

        if strategy_config and hasattr(strategy_config, 'params'):
            params = strategy_config.params or {}
            _CONFIG_FIELDS = [
                "model", "temperature", "max_tokens",
                "cycle_interval_seconds", "max_trades_per_cycle",
                "target_events",
                "max_contracts_per_trade", "min_spread_cents",
                "require_fresh_news", "max_news_age_hours",
            ]
            for field_name in _CONFIG_FIELDS:
                if field_name in params:
                    setattr(config, field_name, params[field_name])

        return config

    # === StrategyCoordinator Integration Methods ===

    def get_market_states(self, limit: int = 100) -> list:
        """Return empty list - deep agent doesn't track market states like RLM."""
        return []

    def get_trade_processing_stats(self) -> Dict[str, Any]:
        """Get trade processing stats for heartbeat."""
        stats = self.get_stats()
        return {
            "trades_processed": stats["trades_executed"],
            "trades_filtered": 0,
            "signals_detected": stats["cycles_run"],
            "signals_executed": stats["trades_executed"],
            "signals_skipped": 0,
            "rate_limited_count": 0,
            "reentries": 0,
        }

    def get_recent_tracked_trades(self, limit: int = 20) -> list:
        """Return empty list - use get_consolidated_view for pending trades."""
        return []

    def get_decision_history(self, limit: int = 20) -> list:
        """Get recent decisions from the agent."""
        if self._agent:
            reflections = self._agent.get_consolidated_view().get("recent_reflections", [])
            return reflections[:limit]
        return []
