"""
Listener Bootstrap Service for TRADER V3.

Centralizes initialization of real-time WebSocket listeners:
- PositionListener: Real-time position updates
- MarketTickerListener: Real-time market price updates
- FillListener: Real-time order fill notifications

This service extracts listener setup logic from coordinator.py to:
1. Reduce coordinator complexity (~130 lines)
2. Centralize listener configuration
3. Maintain identical behavior and health monitoring
"""

import logging
from typing import Optional, Callable, Awaitable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.position_listener import PositionListener
    from ..clients.market_ticker_listener import MarketTickerListener
    from ..clients.fill_listener import FillListener
    from ..core.event_bus import EventBus
    from ..core.health_monitor import V3HealthMonitor
    from ..core.status_reporter import V3StatusReporter
    from ..core.state_container import V3StateContainer
    from ..config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.services.listener_bootstrap_service")


class ListenerBootstrapService:
    """
    Bootstraps real-time WebSocket listeners for V3 trader.

    Manages initialization and event bus wiring for:
    - PositionListener: Instant position updates from Kalshi
    - MarketTickerListener: Real-time market prices for positions
    - FillListener: Order fill notifications for console UX

    Each listener is non-critical - failures fall back gracefully:
    - Position listener: Falls back to REST API polling
    - Market ticker listener: Prices unavailable in real-time
    - Fill listener: Falls back to REST API sync for order state
    """

    def __init__(
        self,
        config: 'V3Config',
        event_bus: 'EventBus',
        state_container: 'V3StateContainer',
        health_monitor: 'V3HealthMonitor',
        status_reporter: 'V3StatusReporter',
    ):
        """
        Initialize listener bootstrap service.

        Args:
            config: V3 configuration (provides ws_url)
            event_bus: Event bus for listener subscriptions and system activity
            state_container: State container for position/price updates
            health_monitor: Health monitor for listener registration
            status_reporter: Status reporter for listener registration
        """
        self._config = config
        self._event_bus = event_bus
        self._state_container = state_container
        self._health_monitor = health_monitor
        self._status_reporter = status_reporter

        logger.debug("ListenerBootstrapService initialized")

    async def connect_position_listener(
        self,
        on_position_update: Callable[[Any], Awaitable[None]],
    ) -> Optional['PositionListener']:
        """
        Connect to real-time position updates via WebSocket.

        Initializes the PositionListener which subscribes to the
        market_positions WebSocket channel for instant position updates.

        Args:
            on_position_update: Async callback for position update events

        Returns:
            PositionListener instance if successful, None on failure
        """
        try:
            # Import here to avoid circular imports
            from ..clients.position_listener import PositionListener

            logger.info("Starting real-time position listener...")

            # Create position listener
            position_listener = PositionListener(
                event_bus=self._event_bus,
                ws_url=self._config.ws_url,
                reconnect_delay_seconds=5.0,
            )

            # Subscribe to position update events
            await self._event_bus.subscribe_to_market_position(on_position_update)

            # Start the listener
            await position_listener.start()

            # Set on status reporter for health broadcasting
            self._status_reporter.set_position_listener(position_listener)

            # Register with health monitor for health tracking
            self._health_monitor.set_position_listener(position_listener)

            logger.info("Real-time position listener active")

            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Real-time position updates enabled",
                metadata={"channel": "market_positions", "severity": "info"}
            )

            return position_listener

        except Exception as e:
            # Don't fail startup if position listener fails - fall back to polling
            logger.warning(f"Position listener failed (falling back to polling): {e}")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Real-time positions unavailable: {e}",
                metadata={"fallback": "polling", "severity": "warning"}
            )
            return None

    async def connect_market_ticker_listener(
        self,
        on_ticker_update: Callable[[Any], Awaitable[None]],
    ) -> Optional['MarketTickerListener']:
        """
        Connect market ticker listener for real-time price updates.

        This is a non-critical component - failure falls back to no real-time prices.
        Subscribes to position tickers after starting if positions exist.

        Args:
            on_ticker_update: Async callback for ticker update events

        Returns:
            MarketTickerListener instance if successful, None on failure
        """
        try:
            # Import here to avoid circular imports
            from ..clients.market_ticker_listener import MarketTickerListener

            logger.info("Starting market ticker listener...")

            # Create market ticker listener with 500ms throttle
            market_ticker_listener = MarketTickerListener(
                event_bus=self._event_bus,
                ws_url=self._config.ws_url,
                throttle_ms=500,
            )

            # Subscribe to ticker update events
            await self._event_bus.subscribe_to_market_ticker(on_ticker_update)

            # Start the listener
            await market_ticker_listener.start()

            # Get current position tickers and subscribe
            if self._state_container.trading_state:
                tickers = list(self._state_container.trading_state.positions.keys())
                if tickers:
                    await market_ticker_listener.update_subscriptions(tickers)
                    logger.info(f"Subscribed to {len(tickers)} position tickers for price updates")

            # Set on status reporter for health broadcasting
            self._status_reporter.set_market_ticker_listener(market_ticker_listener)

            # Register with health monitor for health tracking
            self._health_monitor.set_market_ticker_listener(market_ticker_listener)

            logger.info("Market ticker listener active")

            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Real-time market prices enabled",
                metadata={"channel": "ticker", "severity": "info"}
            )

            return market_ticker_listener

        except Exception as e:
            # Don't fail startup if market ticker listener fails - prices are optional
            logger.warning(f"Market ticker listener failed (prices unavailable): {e}")
            self._state_container.set_component_degraded("market_ticker", True, str(e))
            return None

    async def connect_fill_listener(
        self,
        on_order_fill: Callable[[Any], Awaitable[None]],
    ) -> Optional['FillListener']:
        """
        Connect fill listener for real-time order fill notifications.

        This provides instant feedback when our orders get filled,
        rather than waiting for the next REST API sync.

        Args:
            on_order_fill: Async callback for order fill events

        Returns:
            FillListener instance if successful, None on failure
        """
        try:
            # Import here to avoid circular imports
            from ..clients.fill_listener import FillListener

            logger.info("Starting fill listener for order notifications...")

            # Create fill listener
            fill_listener = FillListener(
                event_bus=self._event_bus,
                ws_url=self._config.ws_url,
                reconnect_delay_seconds=5.0,
            )

            # Subscribe to fill events for console UX
            await self._event_bus.subscribe_to_order_fill(on_order_fill)

            # Start the listener
            await fill_listener.start()

            # Register with health monitor for health tracking
            self._health_monitor.set_fill_listener(fill_listener)

            logger.info("Fill listener active - instant order fill notifications enabled")

            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Order fill notifications enabled",
                metadata={"channel": "fill", "severity": "info"}
            )

            return fill_listener

        except Exception as e:
            # Don't fail startup if fill listener fails - REST API sync still works
            logger.warning(f"Fill listener failed (falling back to REST sync): {e}")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Real-time fill notifications unavailable: {e}",
                metadata={"fallback": "rest_sync", "severity": "warning"}
            )
            return None
