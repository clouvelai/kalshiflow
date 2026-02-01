"""
Health monitoring service for V3 trader.

Extracted from coordinator to reduce complexity and improve separation of concerns.
Monitors component health and triggers recovery when needed.

Component Criticality Classification:
    - CRITICAL_COMPONENTS: Must be healthy for READY state - failures trigger ERROR
    - NON_CRITICAL_COMPONENTS: Can be unhealthy - system continues in degraded mode
"""

import asyncio
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING, Set

if TYPE_CHECKING:
    from .state_machine import TraderStateMachine as V3StateMachine, TraderState as V3State
    from .event_bus import EventBus
    from .websocket_manager import V3WebSocketManager
    from .state_container import V3StateContainer
    from ..clients.orderbook_integration import V3OrderbookIntegration
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..clients.trades_integration import V3TradesIntegration
    from ..clients.market_ticker_listener import MarketTickerListener
    from ..clients.position_listener import PositionListener
    from ..clients.fill_listener import FillListener
    from ..services.market_price_syncer import MarketPriceSyncer
    from ..services.trading_state_syncer import TradingStateSyncer
    from ..strategies import DeepAgentStrategy
    from ..config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.core.health_monitor")

# Components that MUST be healthy for READY state
# Failures in these components trigger ERROR state transition
CRITICAL_COMPONENTS: Set[str] = {"state_machine", "event_bus", "websocket_manager"}

# Components that can be unhealthy (system continues with degraded component)
# Failures in these components are logged but don't trigger ERROR state
NON_CRITICAL_COMPONENTS: Set[str] = {
    "orderbook_integration",
    "trades_integration",
    "health_monitor",
    "trading_client",  # Trading can fail without crashing - degraded mode visibility
    # WebSocket listeners for real-time data
    "market_ticker_listener",
    "position_listener",
    "fill_listener",
    # Services
    "market_price_syncer",
    "trading_state_syncer",
    "deep_agent_strategy",  # Deep agent trading strategy
}


class V3HealthMonitor:
    """
    Health monitoring service for V3 trader components.
    
    Responsibilities:
    - Monitor component health periodically
    - Detect unhealthy states and trigger transitions
    - Support recovery from ERROR states
    - Track health check statistics
    """
    
    def __init__(
        self,
        config: 'V3Config',
        state_machine: 'V3StateMachine',
        event_bus: 'EventBus',
        websocket_manager: 'V3WebSocketManager',
        state_container: 'V3StateContainer',
        orderbook_integration: 'V3OrderbookIntegration',
        trading_client_integration: Optional['V3TradingClientIntegration'] = None,
        trades_integration: Optional['V3TradesIntegration'] = None,
        market_ticker_listener: Optional['MarketTickerListener'] = None,
        position_listener: Optional['PositionListener'] = None,
        market_price_syncer: Optional['MarketPriceSyncer'] = None
    ):
        """
        Initialize health monitor.

        Args:
            config: V3 configuration
            state_machine: State machine instance
            event_bus: Event bus instance
            websocket_manager: WebSocket manager instance
            state_container: State container instance
            orderbook_integration: Orderbook integration instance
            trading_client_integration: Optional trading client integration
            trades_integration: Optional trades integration for public trades stream
            market_ticker_listener: Optional real-time market price listener
            position_listener: Optional real-time position listener
            market_price_syncer: Optional REST market price syncer
        """
        self._config = config
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._state_container = state_container
        self._orderbook_integration = orderbook_integration
        self._trading_client_integration = trading_client_integration
        self._trades_integration = trades_integration
        # Optional components
        self._market_ticker_listener = market_ticker_listener
        self._position_listener = position_listener
        self._fill_listener: Optional['FillListener'] = None  # Set via setter during startup
        self._market_price_syncer = market_price_syncer
        self._trading_state_syncer = None  # Set via setter during startup
        self._deep_agent_strategy: Optional['DeepAgentStrategy'] = None  # Set via setter during lifecycle startup

        # Component registration for dynamic health monitoring
        # Allows new components to be registered without modifying _check_components_health()
        self._registered_components: Dict[str, Dict[str, Any]] = {}

        # Health monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self._health_check_count = 0
        self._was_all_healthy = True  # Track health state for change detection

        logger.info("Health monitor initialized")

    # Component registration for dynamic health monitoring
    def register_component(self, name: str, component: Any, critical: bool = False) -> None:
        """
        Register a component for health monitoring.

        This method allows new components to be added for health monitoring
        without modifying _check_components_health(). Components with an
        is_healthy() method will have it called; others are assumed healthy
        if they exist.

        Args:
            name: Component identifier for health reports (e.g., "market_price_syncer")
            component: The component instance (should have is_healthy() method)
            critical: If True, component failure marks overall health as unhealthy
                      and may trigger ERROR state transition
        """
        if component is None:
            # Unregister if component is None
            self._registered_components.pop(name, None)
            logger.debug(f"Unregistered component from health monitoring: {name}")
            return

        self._registered_components[name] = {
            "component": component,
            "critical": critical
        }
        logger.debug(f"Registered component for health monitoring: {name} (critical={critical})")

    # Setter methods for components created after health monitor initialization
    # These use register_component() internally for health monitoring
    def set_market_ticker_listener(self, listener: Optional['MarketTickerListener']) -> None:
        """Set market ticker listener reference (created during startup)."""
        self._market_ticker_listener = listener
        self.register_component("market_ticker_listener", listener, critical=False)

    def set_position_listener(self, listener: Optional['PositionListener']) -> None:
        """Set position listener reference (created during startup)."""
        self._position_listener = listener
        self.register_component("position_listener", listener, critical=False)

    def set_fill_listener(self, listener: Optional['FillListener']) -> None:
        """Set fill listener reference (created during startup)."""
        self._fill_listener = listener
        self.register_component("fill_listener", listener, critical=False)

    def set_market_price_syncer(self, syncer: Optional['MarketPriceSyncer']) -> None:
        """Set market price syncer reference (created during startup)."""
        self._market_price_syncer = syncer
        self.register_component("market_price_syncer", syncer, critical=False)

    def set_trading_state_syncer(self, syncer: Optional['TradingStateSyncer']) -> None:
        """Set trading state syncer reference (created during startup)."""
        self._trading_state_syncer = syncer
        self.register_component("trading_state_syncer", syncer, critical=False)

    def set_deep_agent_strategy(self, strategy: Optional['DeepAgentStrategy']) -> None:
        """Set deep agent strategy reference (created during lifecycle startup)."""
        self._deep_agent_strategy = strategy
        self.register_component("deep_agent_strategy", strategy, critical=False)

    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            logger.warning("Health monitor already running")
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self) -> None:
        """Main health monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                
                # Check component health
                components_health = await self._check_components_health()
                
                # Update state container with health status
                for name, healthy in components_health.items():
                    self._state_container.update_component_health(name, healthy)
                
                all_healthy = all(components_health.values())
                
                # Handle health state transitions
                await self._handle_health_state(components_health, all_healthy)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    async def _check_components_health(self) -> Dict[str, bool]:
        """
        Check health of all components.

        Core components (state_machine, event_bus, websocket_manager) are always
        checked directly. Orderbook has special staleness detection logic.
        Optional components use the registration pattern via _registered_components.
        """
        # Core components (always present, checked directly)
        components_health = {
            "state_machine": self._state_machine.is_healthy() if self._state_machine else False,
            "event_bus": self._event_bus.is_healthy() if self._event_bus else False,
            "websocket_manager": self._websocket_manager.is_healthy() if self._websocket_manager else False
        }

        # Orderbook has special staleness detection logic - keep it separate
        components_health["orderbook_integration"] = self._check_orderbook_health()

        # Add trading client health if configured (special case - not via setter)
        if self._trading_client_integration:
            components_health["trading_client"] = self._trading_client_integration.is_healthy()

        # Add trades integration health if configured (special case - not via setter)
        if self._trades_integration:
            components_health["trades_integration"] = self._trades_integration.is_healthy()

        # Registered optional components (added via setter methods or register_component())
        for name, info in self._registered_components.items():
            component = info["component"]
            if component and hasattr(component, "is_healthy"):
                try:
                    components_health[name] = component.is_healthy()
                except Exception as e:
                    logger.warning(f"Health check failed for {name}: {e}")
                    components_health[name] = False
            elif component:
                # Component exists but no is_healthy method - assume healthy
                components_health[name] = True

        return components_health

    def _check_orderbook_health(self) -> bool:
        """
        Check orderbook integration health with staleness detection.

        Orderbook has special health check logic that considers:
        - Time since last snapshot (stale if > 90 seconds)
        - Ping/pong health status
        - Fallback to basic is_healthy() check

        Returns:
            bool: True if orderbook is healthy, False otherwise
        """
        if not self._orderbook_integration:
            return False

        # Get detailed health info to check connection state
        health_details = self._orderbook_integration.get_health_details()

        # Check if we have recent data (within last 90 seconds)
        time_since_snapshot = health_details.get("time_since_snapshot")
        ping_health = health_details.get("ping_health", "unknown")

        if time_since_snapshot is not None:
            # If we haven't received data in 90s, mark as unhealthy
            if time_since_snapshot > 90:
                logger.debug(f"Orderbook unhealthy: No snapshot in {time_since_snapshot:.1f}s")
                return False
            return True
        elif ping_health == "unhealthy":
            # No recent ping/pong messages
            logger.debug(f"Orderbook unhealthy: Ping health is {ping_health}")
            return False

        # Fallback to basic health check
        return self._orderbook_integration.is_healthy()

    async def _handle_health_state(self, components_health: Dict[str, bool], all_healthy: bool) -> None:
        """
        Handle health state transitions based on component health.

        Uses component criticality classification:
        - CRITICAL components: Failures trigger ERROR state
        - NON_CRITICAL components: Failures set degraded mode, stay in READY
        """
        from .state_machine import TraderState as V3State

        current_state = self._state_machine.current_state

        # Split health checks by criticality
        critical_health = {k: v for k, v in components_health.items() if k in CRITICAL_COMPONENTS}
        non_critical_health = {k: v for k, v in components_health.items() if k in NON_CRITICAL_COMPONENTS}

        all_critical_healthy = all(critical_health.values()) if critical_health else True
        all_non_critical_healthy = all(non_critical_health.values()) if non_critical_health else True

        # Emit health check activity only when state changes (to reduce spam)
        if current_state == V3State.READY:
            self._health_check_count += 1

            # Only emit when health state changes (healthy -> degraded or vice versa)
            if all_healthy != self._was_all_healthy:
                await self._event_bus.emit_system_activity(
                    activity_type="health_check",
                    message=f"Health: {'Recovered - all components healthy' if all_healthy else 'Degraded - some components unhealthy'}",
                    metadata={
                        "components": components_health,
                        "all_healthy": all_healthy,
                        "critical_healthy": all_critical_healthy,
                        "non_critical_healthy": all_non_critical_healthy
                    }
                )
                self._was_all_healthy = all_healthy

        # Handle READY state
        if current_state == V3State.READY:
            # Check for CRITICAL component failures - these trigger ERROR state
            if not all_critical_healthy:
                unhealthy_critical = [k for k, v in critical_health.items() if not v]
                logger.error(f"CRITICAL components unhealthy: {unhealthy_critical}")
                await self._state_machine.transition_to(
                    V3State.ERROR,
                    context="Critical component health check failed",
                    metadata={
                        "reason": "Critical component health check failed",
                        "unhealthy_components": unhealthy_critical,
                        "session_cleanup_triggered": True
                    }
                )
                return

            # Handle NON-CRITICAL component failures - set degraded mode, stay in READY
            if not all_non_critical_healthy:
                unhealthy_non_critical = [k for k, v in non_critical_health.items() if not v]

                # Track which components are degraded
                for component in unhealthy_non_critical:
                    # Check if this is a new degradation
                    current_degraded = self._state_container.get_degraded_components()
                    if component not in current_degraded:
                        # Newly degraded component
                        reason = self._get_degradation_reason(component)
                        self._state_container.set_component_degraded(component, True, reason)

                        # Emit console message for degradation
                        await self._event_bus.emit_system_activity(
                            activity_type="degraded",
                            message=f"{component.replace('_', ' ').title()} degraded - {reason}",
                            metadata={
                                "component": component,
                                "reason": reason,
                                "severity": "warning"
                            }
                        )
                        logger.warning(f"Component degraded: {component} - {reason}")

            # Check for component recovery (non-critical components becoming healthy again)
            current_degraded = self._state_container.get_degraded_components()
            for component in list(current_degraded.keys()):
                if components_health.get(component, False):
                    # Component has recovered
                    self._state_container.set_component_degraded(component, False)

                    # Emit console message for recovery
                    await self._event_bus.emit_system_activity(
                        activity_type="recovered",
                        message=f"{component.replace('_', ' ').title()} recovered",
                        metadata={
                            "component": component,
                            "severity": "info"
                        }
                    )
                    logger.info(f"Component recovered: {component}")

        # Handle ERROR state - check for recovery possibilities
        elif current_state == V3State.ERROR:
            if all_critical_healthy:
                # All critical components healthy - can recover
                await self._attempt_recovery()
            else:
                # Still have critical failures
                self._error_state_count = getattr(self, '_error_state_count', 0) + 1
                if self._error_state_count % 12 == 1:
                    unhealthy_critical = [k for k, v in critical_health.items() if not v]
                    logger.info(f"Waiting for critical components to recover: {unhealthy_critical}")

    def _get_degradation_reason(self, component: str) -> str:
        """
        Get a human-readable reason for component degradation.

        Args:
            component: Name of the degraded component

        Returns:
            Human-readable reason string
        """
        if component == "orderbook_integration":
            # Try to get more specific reason from orderbook health details
            if self._orderbook_integration:
                health_details = self._orderbook_integration.get_health_details()
                time_since_snapshot = health_details.get("time_since_snapshot")
                if time_since_snapshot is not None and time_since_snapshot > 90:
                    return f"no data for {time_since_snapshot:.0f}s"
                ping_health = health_details.get("ping_health")
                if ping_health == "unhealthy":
                    return "connection lost"
            return "connection lost"
        elif component == "trades_integration":
            return "trades stream unavailable"
        elif component == "market_ticker_listener":
            return "real-time prices unavailable"
        elif component == "position_listener":
            return "real-time positions unavailable"
        elif component == "fill_listener":
            return "real-time fill notifications unavailable"
        elif component == "market_price_syncer":
            return "price sync unavailable"
        else:
            return "unavailable"

    async def _attempt_recovery(self) -> None:
        """Attempt recovery from ERROR state when all components are healthy."""
        from .state_machine import TraderState as V3State
        
        logger.info("All components healthy, attempting recovery from ERROR state...")
        
        # Ensure session is ready for recovery
        session_ready = await self._orderbook_integration.ensure_session_for_recovery()
        if not session_ready:
            logger.warning("Session not ready for recovery, will retry next health check")
            return
        
        # Check if orderbook integration has received snapshots
        orderbook_metrics = self._orderbook_integration.get_metrics()
        has_data = orderbook_metrics["snapshots_received"] > 0
        
        if has_data:
            # Direct to READY if we already have data flowing
            logger.info("Data flow confirmed, recovering to READY state")
            await self._state_machine.transition_to(
                V3State.READY,
                context=f"Recovered from error - {orderbook_metrics['markets_connected']} markets operational",
                metadata={
                    "recovery": True,
                    "session_recovery": session_ready,
                    "markets_connected": orderbook_metrics["markets_connected"],
                    "snapshots_received": orderbook_metrics["snapshots_received"]
                }
            )
        else:
            # Go through connection process if no data yet
            logger.info("No data flow yet, recovering via ORDERBOOK_CONNECT")
            await self._state_machine.transition_to(
                V3State.ORDERBOOK_CONNECT,
                context="Recovering connection after components healthy",
                metadata={
                    "recovery": True,
                    "session_recovery": session_ready
                }
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get health monitor status."""
        return {
            "running": self._running,
            "health_check_count": self._health_check_count,
            "check_interval": self._config.health_check_interval
        }
    
    def is_healthy(self) -> bool:
        """Check if health monitor is healthy."""
        return self._running and self._monitoring_task is not None and not self._monitoring_task.done()