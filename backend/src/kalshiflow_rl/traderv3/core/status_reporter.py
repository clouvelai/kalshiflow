"""
Status reporting service for V3 trader.

Extracted from coordinator to reduce complexity and improve separation of concerns.
Handles periodic status broadcasts and trading state updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .state_machine import TraderStateMachine as V3StateMachine
    from .event_bus import EventBus, EventType, TraderStatusEvent
    from .websocket_manager import V3WebSocketManager
    from .state_container import V3StateContainer
    from ..clients.orderbook_integration import V3OrderbookIntegration
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.core.status_reporter")


class V3StatusReporter:
    """
    Status reporting service for V3 trader.
    
    Responsibilities:
    - Periodic status updates to WebSocket clients
    - Trading state change broadcasts
    - Metrics aggregation and reporting
    - Status event formatting
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
        started_at: Optional[float] = None
    ):
        """
        Initialize status reporter.
        
        Args:
            config: V3 configuration
            state_machine: State machine instance
            event_bus: Event bus instance
            websocket_manager: WebSocket manager instance
            state_container: State container instance
            orderbook_integration: Orderbook integration instance
            trading_client_integration: Optional trading client integration
            started_at: System start timestamp
        """
        self._config = config
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._state_container = state_container
        self._orderbook_integration = orderbook_integration
        self._trading_client_integration = trading_client_integration
        self._started_at = started_at
        
        # Reporting state
        self._status_task: Optional[asyncio.Task] = None
        self._trading_state_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Status reporter initialized")
    
    async def start(self) -> None:
        """Start status reporting."""
        if self._running:
            logger.warning("Status reporter already running")
            return
        
        self._running = True
        
        # Start reporting tasks
        self._status_task = asyncio.create_task(self._report_status_loop())
        self._trading_state_task = asyncio.create_task(self._monitor_trading_state())
        
        logger.info("Status reporter started")
    
    async def stop(self) -> None:
        """Stop status reporting."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel tasks
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
        
        if self._trading_state_task:
            self._trading_state_task.cancel()
            try:
                await self._trading_state_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Status reporter stopped")
    
    def set_started_at(self, started_at: float) -> None:
        """Update the system start timestamp."""
        self._started_at = started_at
    
    async def emit_status_update(self, context: str = "") -> None:
        """Emit status update event immediately."""
        try:
            from .event_bus import EventType, TraderStatusEvent
            
            # Gather metrics from orderbook integration
            orderbook_metrics = self._orderbook_integration.get_metrics()
            ws_stats = self._websocket_manager.get_stats()
            
            # Get detailed health information
            health_details = self._orderbook_integration.get_health_details()
            
            # Get session information if available
            session_info = {}
            if hasattr(self._orderbook_integration, '_client') and self._orderbook_integration._client:
                client_stats = self._orderbook_integration._client.get_stats()
                session_info = {
                    "session_id": client_stats.get("session_id"),
                    "session_state": client_stats.get("session_state", "unknown"),
                    "health_state": client_stats.get("health_state", "unknown")
                }
            
            # Calculate uptime
            uptime = time.time() - self._started_at if self._started_at else 0.0
            
            # Get Kalshi API ping health
            ping_health = health_details.get("ping_health", "unknown")
            last_ping_age = health_details.get("last_ping_age_seconds")
            
            # Determine API connection status
            api_connected = False
            if self._trading_client_integration:
                trading_metrics = self._trading_client_integration.get_metrics()
                if trading_metrics.get("connected"):
                    api_connected = True
            elif orderbook_metrics["markets_connected"] > 0:
                api_connected = True
            
            # Check overall health
            is_healthy = all([
                self._state_machine.is_healthy(),
                self._event_bus.is_healthy(),
                self._websocket_manager.is_healthy(),
                self._orderbook_integration.is_healthy()
            ])
            
            if self._trading_client_integration:
                is_healthy = is_healthy and self._trading_client_integration.is_healthy()
            
            # Publish status event
            event = TraderStatusEvent(
                event_type=EventType.TRADER_STATUS,
                state=self._state_machine.current_state.value,
                health="",  # Keep for backwards compatibility
                metrics={
                    "uptime": uptime,
                    "state": self._state_machine.current_state.value,
                    "markets_connected": orderbook_metrics["markets_connected"],
                    "snapshots_received": orderbook_metrics["snapshots_received"],
                    "deltas_received": orderbook_metrics["deltas_received"],
                    "ws_clients": ws_stats["active_connections"],
                    "ws_messages_sent": ws_stats.get("total_messages_sent", 0),
                    "context": context,
                    "health": "healthy" if is_healthy else "unhealthy",
                    "ping_health": ping_health,
                    "last_ping_age": last_ping_age,
                    "api_connected": api_connected,
                    "api_url": self._config.api_url,
                    "ws_url": self._config.ws_url,
                    "connection_established": health_details.get("connection_established"),
                    "first_snapshot_received": health_details.get("first_snapshot_received"),
                    "session_id": session_info.get("session_id"),
                    "session_state": session_info.get("session_state"),
                    "health_state": session_info.get("health_state")
                },
                timestamp=time.time()
            )
            
            await self._event_bus.emit_trader_status(
                state=event.state,
                metrics=event.metrics,
                health=event.health
            )
            
            # Log summary
            session_display = f"Session: {session_info.get('session_id', 'N/A')} ({session_info.get('session_state', 'unknown')})" if session_info.get('session_id') else "Session: None"
            logger.info(
                f"STATUS: {self._state_machine.current_state.value} | "
                f"Markets: {orderbook_metrics['markets_connected']} | "
                f"Snapshots: {orderbook_metrics['snapshots_received']} | "
                f"Deltas: {orderbook_metrics['deltas_received']} | "
                f"WS Clients: {ws_stats['active_connections']} | "
                f"{session_display}"
                f"{' | ' + context if context else ''}"
            )
            
            logger.debug(
                f"Broadcasting status with connection fields: "
                f"connection_established={event.metrics.get('connection_established')}, "
                f"first_snapshot_received={event.metrics.get('first_snapshot_received')}, "
                f"ping_health={event.metrics.get('ping_health')}, "
                f"last_ping_age={event.metrics.get('last_ping_age')}"
            )
            
        except Exception as e:
            logger.error(f"Error emitting status update: {e}")
    
    async def emit_trading_state(self) -> None:
        """Emit trading state update event."""
        try:
            # Get order_group_id for filtering orders
            order_group_id = None
            if self._trading_client_integration:
                order_group_id = self._trading_client_integration.get_order_group_id()

            trading_summary = self._state_container.get_trading_summary(order_group_id)

            if not trading_summary.get("has_state"):
                return  # No trading state to broadcast

            # Broadcast trading state via websocket
            await self._websocket_manager.broadcast_message("trading_state", {
                "timestamp": time.time(),
                "version": trading_summary["version"],
                "balance": trading_summary["balance"],
                "portfolio_value": trading_summary["portfolio_value"],
                "position_count": trading_summary["position_count"],
                "order_count": trading_summary["order_count"],
                "positions": trading_summary["positions"],
                "open_orders": trading_summary["open_orders"],
                "order_list": trading_summary.get("order_list", []),  # Formatted order list
                "sync_timestamp": trading_summary["sync_timestamp"],
                "changes": trading_summary.get("changes"),
                "order_group": trading_summary.get("order_group"),
                # P&L and position details for frontend display
                "pnl": trading_summary.get("pnl"),
                "positions_details": trading_summary.get("positions_details", [])
            })

            logger.debug(f"Broadcast trading state v{trading_summary['version']}")

        except Exception as e:
            logger.error(f"Error emitting trading state: {e}")
    
    async def _report_status_loop(self) -> None:
        """Report system status periodically."""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Report every 10 seconds
                await self.emit_status_update("Periodic status update")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status reporting: {e}")
    
    async def _monitor_trading_state(self) -> None:
        """Monitor trading state version for broadcasts."""
        last_version = -1  # Start at -1 to ensure first check always broadcasts
        
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                # Check for version changes
                current_version = self._state_container.trading_state_version
                
                # Only broadcast if state changed
                if current_version > last_version:
                    await self.emit_trading_state()
                    last_version = current_version
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading state monitoring: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status reporter status."""
        return {
            "running": self._running,
            "status_interval": 10.0,
            "trading_state_interval": 1.0
        }
    
    def is_healthy(self) -> bool:
        """Check if status reporter is healthy."""
        return self._running and \
               (self._status_task is None or not self._status_task.done()) and \
               (self._trading_state_task is None or not self._trading_state_task.done())