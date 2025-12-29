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
    from ..services.whale_tracker import WhaleTracker
    from ..services.whale_execution_service import WhaleExecutionService
    from ..services.market_price_syncer import MarketPriceSyncer
    from ..services.trading_state_syncer import TradingStateSyncer
    from ..services.yes_80_90_service import Yes8090Service
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
    "whale_tracker",
    "health_monitor",
    # WebSocket listeners for real-time data
    "market_ticker_listener",
    "position_listener",
    "fill_listener",
    # Services
    "market_price_syncer",
    "trading_state_syncer",
    "whale_execution_service",
    "yes_80_90_service",
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
        whale_tracker: Optional['WhaleTracker'] = None,
        # New components for health monitoring
        market_ticker_listener: Optional['MarketTickerListener'] = None,
        position_listener: Optional['PositionListener'] = None,
        market_price_syncer: Optional['MarketPriceSyncer'] = None,
        whale_execution_service: Optional['WhaleExecutionService'] = None,
        yes_80_90_service: Optional['Yes8090Service'] = None
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
            whale_tracker: Optional whale tracker for big bet detection
            market_ticker_listener: Optional real-time market price listener
            position_listener: Optional real-time position listener
            market_price_syncer: Optional REST market price syncer
            whale_execution_service: Optional whale execution service
            yes_80_90_service: Optional YES 80-90c strategy service
        """
        self._config = config
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._state_container = state_container
        self._orderbook_integration = orderbook_integration
        self._trading_client_integration = trading_client_integration
        self._trades_integration = trades_integration
        self._whale_tracker = whale_tracker
        # New components
        self._market_ticker_listener = market_ticker_listener
        self._position_listener = position_listener
        self._fill_listener: Optional['FillListener'] = None  # Set via setter during startup
        self._market_price_syncer = market_price_syncer
        self._trading_state_syncer = None  # Set via setter during startup
        self._whale_execution_service = whale_execution_service
        self._yes_80_90_service = yes_80_90_service

        # Health monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self._health_check_count = 0

        logger.info("Health monitor initialized")

    # Setter methods for components created after health monitor initialization
    def set_market_ticker_listener(self, listener: Optional['MarketTickerListener']) -> None:
        """Set market ticker listener reference (created during startup)."""
        self._market_ticker_listener = listener

    def set_position_listener(self, listener: Optional['PositionListener']) -> None:
        """Set position listener reference (created during startup)."""
        self._position_listener = listener

    def set_fill_listener(self, listener: Optional['FillListener']) -> None:
        """Set fill listener reference (created during startup)."""
        self._fill_listener = listener

    def set_market_price_syncer(self, syncer: Optional['MarketPriceSyncer']) -> None:
        """Set market price syncer reference (created during startup)."""
        self._market_price_syncer = syncer

    def set_trading_state_syncer(self, syncer: Optional['TradingStateSyncer']) -> None:
        """Set trading state syncer reference (created during startup)."""
        self._trading_state_syncer = syncer

    def set_whale_execution_service(self, service: Optional['WhaleExecutionService']) -> None:
        """Set whale execution service reference."""
        self._whale_execution_service = service

    def set_yes_80_90_service(self, service: Optional['Yes8090Service']) -> None:
        """Set YES 80-90c service reference."""
        self._yes_80_90_service = service

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
        """Check health of all components."""
        components_health = {
            "state_machine": self._state_machine.is_healthy(),
            "event_bus": self._event_bus.is_healthy(),
            "websocket_manager": self._websocket_manager.is_healthy()
        }
        
        # Enhanced orderbook health check - check actual connection state
        orderbook_healthy = False
        if self._orderbook_integration:
            # Get detailed health info to check connection state
            health_details = self._orderbook_integration.get_health_details()
            
            # Check if we have recent data (within last 90 seconds)
            time_since_snapshot = health_details.get("time_since_snapshot")
            ping_health = health_details.get("ping_health", "unknown")
            
            if time_since_snapshot is not None:
                # If we haven't received data in 90s, mark as unhealthy
                if time_since_snapshot > 90:
                    logger.debug(f"Orderbook unhealthy: No snapshot in {time_since_snapshot:.1f}s")
                    orderbook_healthy = False
                else:
                    orderbook_healthy = True
            elif ping_health == "unhealthy":
                # No recent ping/pong messages
                logger.debug(f"Orderbook unhealthy: Ping health is {ping_health}")
                orderbook_healthy = False
            else:
                # Fallback to basic health check
                orderbook_healthy = self._orderbook_integration.is_healthy()
        
        components_health["orderbook_integration"] = orderbook_healthy
        
        # Add trading client health if configured
        if self._trading_client_integration:
            components_health["trading_client"] = self._trading_client_integration.is_healthy()

        # Add trades integration health if configured
        if self._trades_integration:
            components_health["trades_integration"] = self._trades_integration.is_healthy()

        # Add whale tracker health if configured
        if self._whale_tracker:
            components_health["whale_tracker"] = self._whale_tracker.is_healthy()

        # Add market ticker listener health if configured
        if self._market_ticker_listener:
            components_health["market_ticker_listener"] = self._market_ticker_listener.is_healthy()

        # Add position listener health if configured
        if self._position_listener:
            components_health["position_listener"] = self._position_listener.is_healthy()

        # Add fill listener health if configured
        if self._fill_listener:
            components_health["fill_listener"] = self._fill_listener.is_healthy()

        # Add market price syncer health if configured
        if self._market_price_syncer:
            components_health["market_price_syncer"] = self._market_price_syncer.is_healthy()

        # Add trading state syncer health if configured
        if self._trading_state_syncer:
            components_health["trading_state_syncer"] = self._trading_state_syncer.is_healthy()

        # Add whale execution service health if configured
        if self._whale_execution_service:
            components_health["whale_execution_service"] = self._whale_execution_service.is_healthy()

        # Add YES 80-90c service health if configured
        if self._yes_80_90_service:
            components_health["yes_80_90_service"] = self._yes_80_90_service.is_healthy()

        return components_health
    
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

        # Emit health check activity (only occasionally to avoid spam)
        if current_state == V3State.READY:
            self._health_check_count += 1

            # Emit every 5th check or if any component is unhealthy
            if not all_healthy or self._health_check_count % 5 == 0:
                await self._event_bus.emit_system_activity(
                    activity_type="health_check",
                    message=f"Health check: {'All components healthy' if all_healthy else 'Some components degraded'}",
                    metadata={
                        "components": components_health,
                        "all_healthy": all_healthy,
                        "critical_healthy": all_critical_healthy,
                        "non_critical_healthy": all_non_critical_healthy
                    }
                )

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
        elif component == "whale_tracker":
            return "whale detection unavailable"
        elif component == "market_ticker_listener":
            return "real-time prices unavailable"
        elif component == "position_listener":
            return "real-time positions unavailable"
        elif component == "fill_listener":
            return "real-time fill notifications unavailable"
        elif component == "market_price_syncer":
            return "price sync unavailable"
        elif component == "whale_execution_service":
            return "whale execution unavailable"
        elif component == "yes_80_90_service":
            return "YES 80-90c strategy unavailable"
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