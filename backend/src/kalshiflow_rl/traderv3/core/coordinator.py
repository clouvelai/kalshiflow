"""
TRADER V3 Coordinator - Orchestration Layer.

Lightweight coordinator that wires together all V3 components.
Simple, clean orchestration without business logic.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

from .state_machine import TraderStateMachine as V3StateMachine, TraderState as V3State
from .event_bus import EventBus, EventType, StateTransitionEvent, TraderStatusEvent
from .websocket_manager import V3WebSocketManager
from ..clients.orderbook_integration import V3OrderbookIntegration
from ..config.environment import V3Config

logger = logging.getLogger("kalshiflow_rl.traderv3.core.coordinator")


class V3Coordinator:
    """
    Central coordinator for TRADER V3.
    
    Responsibilities:
    - Component lifecycle management
    - Health monitoring
    - Status reporting
    - Graceful shutdown
    
    NO business logic - just orchestration.
    """
    
    def __init__(
        self,
        config: V3Config,
        state_machine: V3StateMachine,
        event_bus: EventBus,
        websocket_manager: V3WebSocketManager,
        orderbook_integration: V3OrderbookIntegration
    ):
        """
        Initialize coordinator.
        
        Args:
            config: V3 configuration
            state_machine: State machine instance
            event_bus: Event bus instance
            websocket_manager: WebSocket manager instance
            orderbook_integration: Orderbook integration instance
        """
        self._config = config
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._orderbook_integration = orderbook_integration
        
        self._started_at: Optional[float] = None
        self._running = False
        
        # Health monitoring task
        self._health_task: Optional[asyncio.Task] = None
        
        # Status reporting task
        self._status_task: Optional[asyncio.Task] = None
        
        logger.info("V3 Coordinator initialized")
    
    async def start(self) -> None:
        """Start the V3 trader system."""
        if self._running:
            logger.warning("V3 Coordinator is already running")
            return
        
        logger.info("=" * 60)
        logger.info("STARTING TRADER V3")
        logger.info(f"Environment: {self._config.get_environment_name()}")
        logger.info(f"Markets: {', '.join(self._config.market_tickers[:3])}{'...' if len(self._config.market_tickers) > 3 else ''}")
        logger.info("=" * 60)
        
        self._running = True
        self._started_at = time.time()
        
        try:
            # Start components in order
            logger.info("1/5 Starting Event Bus...")
            await self._event_bus.start()
            
            logger.info("2/5 Starting WebSocket Manager...")
            await self._websocket_manager.start()
            
            logger.info("3/5 Starting State Machine...")
            await self._state_machine.start()
            
            # Add metadata to STARTUP → INITIALIZING transition (retroactively)
            # The state machine already did this transition, but we can emit additional info
            startup_metadata = {
                "environment": self._config.get_environment_name(),
                "host": f"{self._config.host}:{self._config.port}",
                "log_level": self._config.log_level,
                "mode": "discovery" if "DISCOVERY" in str(self._config.market_tickers) else "config"
            }
            
            # Emit initial status
            await self._emit_status_update("System initializing")
            
            logger.info("4/5 Starting Orderbook Integration...")
            await self._orderbook_integration.start()
            
            logger.info("5/5 Starting monitoring tasks...")
            self._health_task = asyncio.create_task(self._monitor_health())
            self._status_task = asyncio.create_task(self._report_status())
            
            # The state machine already transitioned to INITIALIZING in start()
            # Now transition to orderbook connectivity
            logger.info("Transitioning to orderbook connectivity...")
            
            # Prepare metadata for the ORDERBOOK_CONNECT state
            metadata = {
                "ws_url": self._config.ws_url,
                "markets": self._config.market_tickers[:2] + (["...and {} more".format(len(self._config.market_tickers) - 2)] if len(self._config.market_tickers) > 2 else []),
                "market_count": len(self._config.market_tickers),
                "environment": self._config.get_environment_name()
            }
            
            await self._state_machine.transition_to(
                V3State.ORDERBOOK_CONNECT,
                context=f"Connecting to {len(self._config.market_tickers)} markets",
                metadata=metadata
            )
            
            # Emit connecting status
            await self._emit_status_update(f"Connecting to {len(self._config.market_tickers)} markets")
            
            # ACTUALLY WAIT FOR CONNECTION - No smoke and mirrors!
            connection_success = False
            data_flowing = False
            
            # Step 1: Wait for WebSocket connection
            logger.info("Waiting for orderbook WebSocket connection...")
            connection_success = await self._orderbook_integration.wait_for_connection(timeout=30.0)
            
            if not connection_success:
                logger.error("Failed to connect to orderbook WebSocket!")
                await self._state_machine.enter_error_state(
                    "Orderbook connection failed",
                    Exception("WebSocket connection timeout")
                )
                return
            
            # Step 2: Wait for first snapshot to confirm data flow
            logger.info("Waiting for initial orderbook snapshot...")
            data_flowing = await self._orderbook_integration.wait_for_first_snapshot(timeout=10.0)
            
            if not data_flowing:
                logger.warning("No orderbook data received - continuing anyway but system may not be fully operational")
            
            # Get ACTUAL metrics after connection
            orderbook_metrics = self._orderbook_integration.get_metrics()
            health_details = self._orderbook_integration.get_health_details()
            
            # Transition to ready state with REAL connection status
            logger.info("Transitioning to ready state...")
            
            metadata = {
                "markets_connected": orderbook_metrics["markets_connected"],
                "snapshots_received": orderbook_metrics["snapshots_received"],
                "deltas_received": orderbook_metrics["deltas_received"],
                "connection_established": health_details["connection_established"],
                "first_snapshot_received": health_details["first_snapshot_received"],
                "environment": self._config.get_environment_name()
            }
            
            # Determine context based on actual connection status
            if connection_success and data_flowing:
                context = f"System fully operational with {orderbook_metrics['markets_connected']} markets"
            elif connection_success:
                context = f"System connected (waiting for data) - {orderbook_metrics['markets_connected']} markets"
            else:
                context = f"System started (connection pending) - {len(self._config.market_tickers)} markets configured"
            
            await self._state_machine.transition_to(
                V3State.READY,
                context=context,
                metadata=metadata
            )
            
            # Emit ready status
            await self._emit_status_update(f"System ready with {len(self._config.market_tickers)} markets")
            
            logger.info("=" * 60)
            logger.info("✅ TRADER V3 STARTED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to start V3 Coordinator: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the V3 trader system."""
        if not self._running:
            return
        
        logger.info("=" * 60)
        logger.info("STOPPING TRADER V3")
        logger.info("=" * 60)
        
        self._running = False
        
        # Cancel monitoring tasks
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
        
        # Stop components in reverse order
        try:
            logger.info("1/5 Stopping Orderbook Integration...")
            await self._orderbook_integration.stop()
            
            logger.info("2/5 Transitioning to SHUTDOWN state...")
            await self._state_machine.transition_to(
                V3State.SHUTDOWN,
                context="Graceful shutdown initiated"
            )
            
            logger.info("3/5 Stopping State Machine...")
            await self._state_machine.stop()
            
            logger.info("4/5 Stopping WebSocket Manager...")
            await self._websocket_manager.stop()
            
            logger.info("5/5 Stopping Event Bus...")
            await self._event_bus.stop()
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        uptime = time.time() - self._started_at if self._started_at else 0
        logger.info("=" * 60)
        logger.info(f"✅ TRADER V3 STOPPED (uptime: {uptime:.1f}s)")
        logger.info("=" * 60)
    
    async def _monitor_health(self) -> None:
        """Monitor component health."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                
                # Check component health
                components_health = {
                    "state_machine": self._state_machine.is_healthy(),
                    "event_bus": self._event_bus.is_healthy(),
                    "websocket_manager": self._websocket_manager.is_healthy(),
                    "orderbook_integration": self._orderbook_integration.is_healthy()
                }
                
                all_healthy = all(components_health.values())
                
                # If in READY state and something is unhealthy, transition to ERROR
                if self._state_machine.current_state == V3State.READY and not all_healthy:
                    unhealthy = [k for k, v in components_health.items() if not v]
                    logger.error(f"Components unhealthy: {unhealthy}")
                    await self._state_machine.transition_to(
                        V3State.ERROR,
                        context="Component health check failed",
                        metadata={
                            "reason": "Component health check failed",
                            "unhealthy_components": unhealthy
                        }
                    )
                
                # If in ERROR state and everything is healthy, attempt recovery
                elif self._state_machine.current_state == V3State.ERROR and all_healthy:
                    logger.info("All components healthy, attempting recovery...")
                    await self._state_machine.transition_to(
                        V3State.ORDERBOOK_CONNECT,
                        context="Attempting recovery after all components healthy"
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
    
    async def _emit_status_update(self, context: str = "") -> None:
        """Emit status update event immediately."""
        try:
            # Gather metrics
            orderbook_metrics = self._orderbook_integration.get_metrics()
            ws_stats = self._websocket_manager.get_stats()
            
            # Get detailed health information including connection status fields
            health_details = self._orderbook_integration.get_health_details()
            
            uptime = time.time() - self._started_at if self._started_at else 0
            
            # Publish status event with proper event_type
            # Note: health field is now removed from top level (will be in metrics)
            event = TraderStatusEvent(
                event_type=EventType.TRADER_STATUS,
                state=self._state_machine.current_state.value,
                health="",  # Keep for backwards compatibility but empty
                metrics={
                    "uptime": uptime,
                    "state": self._state_machine.current_state.value,
                    "markets_connected": orderbook_metrics["markets_connected"],
                    "snapshots_received": orderbook_metrics["snapshots_received"],
                    "deltas_received": orderbook_metrics["deltas_received"],
                    "ws_clients": ws_stats["active_connections"],
                    "ws_messages_sent": ws_stats["messages_sent"],
                    "context": context,
                    # Add health status INSIDE metrics (fix for Issue 1)
                    "health": "healthy" if self.is_healthy() else "unhealthy",
                    # Add connection status fields (fix for Issue 2)
                    "connection_established": health_details.get("connection_established"),
                    "first_snapshot_received": health_details.get("first_snapshot_received")
                },
                timestamp=time.time()
            )
            
            await self._event_bus.publish(event)
            
            # Debug log to verify fields are present
            logger.debug(
                f"Broadcasting status with connection fields: "
                f"connection_established={event.metrics.get('connection_established')}, "
                f"first_snapshot_received={event.metrics.get('first_snapshot_received')}"
            )
            
            # Log summary
            logger.info(
                f"STATUS: {self._state_machine.current_state.value} | "
                f"Markets: {orderbook_metrics['markets_connected']} | "
                f"Snapshots: {orderbook_metrics['snapshots_received']} | "
                f"Deltas: {orderbook_metrics['deltas_received']} | "
                f"WS Clients: {ws_stats['active_connections']}"
                f"{' | ' + context if context else ''}"
            )
            
        except Exception as e:
            logger.error(f"Error emitting status update: {e}")
    
    async def _report_status(self) -> None:
        """Report system status periodically."""
        while self._running:
            try:
                await asyncio.sleep(10.0)  # Report every 10 seconds
                await self._emit_status_update("Periodic status update")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in status reporting: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = time.time() - self._started_at if self._started_at else 0
        
        return {
            "running": self._running,
            "uptime": uptime,
            "state": self._state_machine.current_state.value,
            "environment": self._config.get_environment_name(),
            "markets": self._config.market_tickers,
            "components": {
                "state_machine": self._state_machine.get_health_details(),
                "event_bus": self._event_bus.get_health_details(),
                "websocket_manager": self._websocket_manager.get_health_details(),
                "orderbook_integration": self._orderbook_integration.get_health_details()
            }
        }
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        if not self._running:
            return False
        
        return all([
            self._state_machine.is_healthy(),
            self._event_bus.is_healthy(),
            self._websocket_manager.is_healthy(),
            self._orderbook_integration.is_healthy()
        ])
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "healthy": self.is_healthy(),
            "status": "running" if self._running else "stopped",
            "state": self._state_machine.current_state.value,
            "uptime": time.time() - self._started_at if self._started_at else 0
        }