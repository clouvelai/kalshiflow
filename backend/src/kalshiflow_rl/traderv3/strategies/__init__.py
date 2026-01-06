"""
Strategy Plugin System for TRADER V3.

This package provides the infrastructure for running multiple concurrent
trading strategies using a plugin-based architecture.

Key Components:
    - Strategy: Protocol that all strategies must implement
    - StrategyContext: Shared resources passed to strategies
    - StrategyRegistry: Central registry for strategy plugins
    - StrategyCoordinator: Manages strategy lifecycle and rate limiting

Usage:
    # 1. Register a strategy plugin
    from kalshiflow_rl.traderv3.strategies import StrategyRegistry

    @StrategyRegistry.register("my_strategy")
    class MyStrategy:
        name = "my_strategy"
        display_name = "My Trading Strategy"
        subscribed_events = {EventType.PUBLIC_TRADE_RECEIVED}

        async def start(self, context): ...
        async def stop(self): ...
        def is_healthy(self) -> bool: ...
        def get_stats(self) -> Dict[str, Any]: ...

    # 2. Create coordinator and start strategies
    from kalshiflow_rl.traderv3.strategies import (
        StrategyCoordinator,
        StrategyContext
    )

    context = StrategyContext(
        event_bus=event_bus,
        trading_service=trading_service,
        state_container=state_container,
        orderbook_integration=orderbook_integration,
        tracked_markets=tracked_markets,
    )

    coordinator = StrategyCoordinator(context)
    await coordinator.load_configs()
    await coordinator.start_all()

Configuration:
    Strategy configurations are stored as YAML files in strategies/config/.
    Each file defines one strategy with its parameters.

    Example (rlm_no.yaml):
        name: rlm_no
        enabled: true
        display_name: "Reverse Line Movement NO"
        max_positions: 60
        params:
          yes_threshold: 0.70
          min_trades: 25
"""

from .protocol import Strategy, StrategyContext
from .registry import StrategyRegistry
from .coordinator import StrategyCoordinator, StrategyConfig

# Import plugins to trigger registration
from . import plugins

__all__ = [
    # Protocol and context
    'Strategy',
    'StrategyContext',
    # Registry
    'StrategyRegistry',
    # Coordinator
    'StrategyCoordinator',
    'StrategyConfig',
]
