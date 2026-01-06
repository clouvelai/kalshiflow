"""
Hold Strategy Plugin - No-op Baseline for TRADER V3.

This is the simplest possible strategy - it does nothing.
Used for:
- Testing the plugin system infrastructure
- Running the trader without any active strategy
- Baseline comparison in A/B tests

The Hold strategy:
- Subscribes to no events
- Places no trades
- Always reports healthy
- Provides minimal stats

Usage:
    # Enabled by default when no other strategy is configured
    # Or explicitly via YAML:
    # name: hold
    # enabled: true
"""

import logging
import time
from typing import Dict, Any, Set, Optional

from ..protocol import Strategy, StrategyContext
from ..registry import StrategyRegistry
from ...core.events import EventType

logger = logging.getLogger("kalshiflow_rl.traderv3.strategies.plugins.hold")


@StrategyRegistry.register("hold")
class HoldStrategy:
    """
    No-op baseline strategy that does nothing.

    This is the default strategy when no active trading is desired.
    It's useful for:
    - Testing the plugin infrastructure
    - Monitoring-only mode
    - Baseline comparison for strategy performance

    Attributes:
        name: "hold"
        display_name: "Hold (No Trading)"
        subscribed_events: Empty set (subscribes to nothing)
    """

    name: str = "hold"
    display_name: str = "Hold (No Trading)"
    subscribed_events: Set[EventType] = set()  # Subscribe to nothing

    def __init__(self):
        """Initialize the Hold strategy."""
        self._context: Optional[StrategyContext] = None
        self._running: bool = False
        self._started_at: Optional[float] = None
        self._events_received: int = 0

        logger.debug("HoldStrategy initialized")

    async def start(self, context: StrategyContext) -> None:
        """
        Start the Hold strategy.

        Since this is a no-op strategy, start() just stores the context
        and marks the strategy as running. No event subscriptions are made.

        Args:
            context: Shared strategy context
        """
        self._context = context
        self._running = True
        self._started_at = time.time()

        logger.info(
            f"HoldStrategy started - No trading will occur. "
            f"Context: {context.to_dict()}"
        )

    async def stop(self) -> None:
        """
        Stop the Hold strategy.

        Since no events are subscribed, this just marks the strategy
        as stopped and cleans up references.
        """
        self._running = False
        self._context = None

        uptime = time.time() - self._started_at if self._started_at else 0
        logger.info(f"HoldStrategy stopped (uptime: {uptime:.1f}s)")

    def is_healthy(self) -> bool:
        """
        Check if the Hold strategy is healthy.

        The Hold strategy is always healthy when running.

        Returns:
            True if the strategy is running
        """
        return self._running

    def get_stats(self) -> Dict[str, Any]:
        """
        Get Hold strategy statistics.

        Returns minimal stats since this strategy does nothing.

        Returns:
            Dictionary with strategy statistics
        """
        uptime = time.time() - self._started_at if self._started_at else 0

        return {
            "name": self.name,
            "display_name": self.display_name,
            "running": self._running,
            "uptime_seconds": uptime,
            "signals_detected": 0,
            "trades_executed": 0,
            "last_signal_at": None,
            "config": {
                "description": "No-op baseline strategy - does not trade",
                "subscribed_events": [],
            },
        }
