"""
Strategy Protocol and Context for TRADER V3 Plugin System.

This module defines the core interfaces for the plugin-based strategy system:
- StrategyContext: Shared resources passed to all strategies
- Strategy: Protocol that all trading strategies must implement

Design Principles:
    - **Protocol-based**: Uses typing.Protocol for duck typing
    - **Runtime checkable**: Can verify strategy implementations
    - **Dependency injection**: Context provides all needed services
    - **Async-first**: All strategy methods are async

Architecture Position:
    - Used by StrategyRegistry for type checking
    - Used by StrategyCoordinator to manage strategy lifecycle
    - Implemented by all strategy plugins in strategies/plugins/
"""

from dataclasses import dataclass
from typing import Dict, Any, Set, Optional, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.events import EventType
    from ..services.trading_decision_service import TradingDecisionService
    from ..core.state_container import V3StateContainer
    from ..clients.orderbook_integration import V3OrderbookIntegration
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..state.tracked_markets import TrackedMarketsState
    from .coordinator import StrategyConfig


@dataclass
class StrategyContext:
    """
    Shared context provided to all strategies.

    Contains all services and state containers that strategies need
    to operate. Passed to Strategy.start() for dependency injection.

    Attributes:
        event_bus: Central event bus for subscribing to events
        trading_service: Service for executing trading decisions
        state_container: Container for trading state (positions, orders, etc.)
        orderbook_integration: Integration for orderbook data access
        tracked_markets: State container for lifecycle-discovered markets
        trading_client_integration: Direct access to trading client for order management
        config: Strategy configuration loaded from YAML (optional)
    """
    event_bus: 'EventBus'
    trading_service: Optional['TradingDecisionService']
    state_container: 'V3StateContainer'
    orderbook_integration: 'V3OrderbookIntegration'
    tracked_markets: Optional['TrackedMarketsState']
    trading_client_integration: Optional['V3TradingClientIntegration'] = None
    config: Optional['StrategyConfig'] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/debugging."""
        return {
            "event_bus": "connected" if self.event_bus else "not available",
            "trading_service": "connected" if self.trading_service else "not available",
            "state_container": "connected" if self.state_container else "not available",
            "orderbook_integration": "connected" if self.orderbook_integration else "not available",
            "tracked_markets": "connected" if self.tracked_markets else "not available",
            "trading_client_integration": "connected" if self.trading_client_integration else "not available",
            "config": self.config.name if self.config else "not loaded",
        }


@runtime_checkable
class Strategy(Protocol):
    """
    Protocol defining the interface for all trading strategies.

    All strategy plugins must implement this interface. The protocol
    is runtime_checkable, allowing isinstance() checks for validation.

    Required Attributes:
        name: Unique identifier (e.g., "rlm_no", "s013")
        display_name: Human-readable name for UI
        subscribed_events: Set of EventType values this strategy handles

    Required Methods:
        start(): Initialize and begin processing
        stop(): Clean shutdown
        is_healthy(): Health check
        get_stats(): Return strategy statistics

    Usage:
        @StrategyRegistry.register("my_strategy")
        class MyStrategy:
            name = "my_strategy"
            display_name = "My Trading Strategy"
            subscribed_events = {EventType.PUBLIC_TRADE_RECEIVED}

            async def start(self, context: StrategyContext) -> None:
                # Subscribe to events, initialize state
                ...
    """

    # Required class attributes
    name: str
    display_name: str
    subscribed_events: Set['EventType']

    async def start(self, context: 'StrategyContext') -> None:
        """
        Start the strategy.

        Called by StrategyCoordinator after loading. Should:
        - Store reference to context
        - Subscribe to events via context.event_bus
        - Initialize any internal state

        Args:
            context: Shared strategy context with all services
        """
        ...

    async def stop(self) -> None:
        """
        Stop the strategy.

        Called during shutdown. Should:
        - Unsubscribe from all events
        - Cancel any pending operations
        - Clean up resources
        """
        ...

    def is_healthy(self) -> bool:
        """
        Check if strategy is healthy and operational.

        Returns:
            True if the strategy is functioning normally
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """
        Get strategy statistics for monitoring and UI.

        Returns:
            Dictionary containing:
            - name: Strategy name
            - display_name: Human-readable name
            - running: Whether strategy is active
            - signals_detected: Count of signals detected
            - trades_executed: Count of trades executed
            - last_signal_at: Timestamp of last signal
            - config: Strategy configuration parameters
            - Any strategy-specific metrics
        """
        ...
