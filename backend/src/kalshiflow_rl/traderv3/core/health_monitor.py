"""
Health monitoring service for V3 trader.

Extracted from coordinator to reduce complexity and improve separation of concerns.
Monitors component health and triggers recovery when needed.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state_machine import TraderStateMachine as V3StateMachine, TraderState as V3State
    from .event_bus import EventBus
    from .websocket_manager import V3WebSocketManager
    from .state_container import V3StateContainer
    from ..clients.orderbook_integration import V3OrderbookIntegration
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.core.health_monitor")


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
        trading_client_integration: Optional['V3TradingClientIntegration'] = None
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
        """
        self._config = config
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._state_container = state_container
        self._orderbook_integration = orderbook_integration
        self._trading_client_integration = trading_client_integration
        
        # Health monitoring state
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self._health_check_count = 0
        
        logger.info("Health monitor initialized")
    
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
        
        return components_health
    
    async def _handle_health_state(self, components_health: Dict[str, bool], all_healthy: bool) -> None:
        """Handle health state transitions based on component health."""
        from .state_machine import TraderState as V3State
        
        current_state = self._state_machine.current_state
        
        # Emit health check activity (only occasionally to avoid spam)
        if current_state == V3State.READY:
            self._health_check_count += 1
            
            # Emit every 5th check or if unhealthy
            if not all_healthy or self._health_check_count % 5 == 0:
                await self._event_bus.emit_system_activity(
                    activity_type="health_check",
                    message=f"Health check: {'All components healthy' if all_healthy else 'Some components unhealthy'}",
                    metadata={
                        "components": components_health,
                        "all_healthy": all_healthy
                    }
                )
        
        # If in READY state and something is unhealthy, transition to ERROR
        # BUT: Check if we're in degraded mode (orderbook down but trading still works)
        if current_state == V3State.READY and not all_healthy:
            unhealthy = [k for k, v in components_health.items() if not v]
            
            # Check if we're in degraded mode (trading without orderbook)
            is_degraded = self._state_container.machine_state_metadata.get("degraded", False)
            
            # In degraded mode, only orderbook_integration being unhealthy is acceptable
            if is_degraded and unhealthy == ["orderbook_integration"]:
                logger.debug("In degraded mode - orderbook unhealthy is expected")
                return  # Don't transition to ERROR
            
            logger.error(f"Components unhealthy: {unhealthy}")
            await self._state_machine.transition_to(
                V3State.ERROR,
                context="Component health check failed",
                metadata={
                    "reason": "Component health check failed",
                    "unhealthy_components": unhealthy,
                    "session_cleanup_triggered": True
                }
            )
        
        # If in ERROR state and everything is healthy, attempt recovery
        elif current_state == V3State.ERROR and all_healthy:
            await self._attempt_recovery()
    
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