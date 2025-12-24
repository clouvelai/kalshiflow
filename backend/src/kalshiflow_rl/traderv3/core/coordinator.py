"""
TRADER V3 Coordinator - Orchestration Layer.

Lightweight coordinator that wires together all V3 components.
Simple, clean orchestration without business logic.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration

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
        orderbook_integration: V3OrderbookIntegration,
        trading_client_integration: Optional['V3TradingClientIntegration'] = None
    ):
        """
        Initialize coordinator.
        
        Args:
            config: V3 configuration
            state_machine: State machine instance
            event_bus: Event bus instance
            websocket_manager: WebSocket manager instance
            orderbook_integration: Orderbook integration instance
            trading_client_integration: Optional trading client integration
        """
        self._config = config
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._orderbook_integration = orderbook_integration
        self._trading_client_integration = trading_client_integration
        
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
            self._websocket_manager.set_coordinator(self)  # Set reference for metrics broadcasting
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
            
            # Check if trading client is configured
            if self._trading_client_integration:
                logger.info("Trading client is configured - connecting to trading API...")
                
                # Transition to TRADING_CLIENT_CONNECT state
                await self._state_machine.transition_to(
                    V3State.TRADING_CLIENT_CONNECT,
                    context="Connecting to trading API",
                    metadata={
                        "mode": self._trading_client_integration._client.mode,
                        "environment": self._config.get_environment_name(),
                        "api_url": self._trading_client_integration.api_url
                    }
                )
                
                # Wait for trading client connection
                trading_connected = await self._trading_client_integration.wait_for_connection(timeout=30.0)
                
                if not trading_connected:
                    logger.error("Failed to connect to trading API!")
                    await self._state_machine.enter_error_state(
                        "Trading client connection failed",
                        Exception("Trading API connection timeout")
                    )
                    return
                
                # Transition to CALIBRATING state
                logger.info("Trading client connected - calibrating positions...")
                await self._state_machine.transition_to(
                    V3State.CALIBRATING,
                    context="Syncing positions and orders",
                    metadata={"mode": self._trading_client_integration._client.mode}
                )
                
                # Perform calibration
                try:
                    calibration_data = await self._trading_client_integration.calibrate()
                    logger.info(
                        f"Calibration complete - Positions: {len(calibration_data.positions)}, "
                        f"Orders: {len(calibration_data.orders)}, Balance: ${calibration_data.balance}"
                    )
                except Exception as e:
                    logger.error(f"Calibration failed: {e}")
                    await self._state_machine.enter_error_state(
                        "Calibration failed",
                        e
                    )
                    return
            
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
            
            # Add trading client info if available
            if self._trading_client_integration:
                trading_metrics = self._trading_client_integration.get_metrics()
                metadata["trading_client"] = {
                    "connected": trading_metrics["connected"],
                    "mode": trading_metrics["mode"],
                    "positions_count": trading_metrics["positions_count"],
                    "balance": trading_metrics["balance"]
                }
            
            # Determine context based on actual connection status
            if connection_success and data_flowing:
                if self._trading_client_integration:
                    context = f"System fully operational with {orderbook_metrics['markets_connected']} markets and trading enabled"
                else:
                    context = f"System fully operational with {orderbook_metrics['markets_connected']} markets (orderbook only)"
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
            status_msg = f"System ready with {len(self._config.market_tickers)} markets"
            if self._trading_client_integration:
                status_msg += f" (trading enabled in {self._trading_client_integration._client.mode} mode)"
            await self._emit_status_update(status_msg)
            
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
            step = 1
            total_steps = 6 if self._trading_client_integration else 5
            
            if self._trading_client_integration:
                logger.info(f"{step}/{total_steps} Stopping Trading Client Integration...")
                await self._trading_client_integration.stop()
                step += 1
            
            logger.info(f"{step}/{total_steps} Stopping Orderbook Integration...")
            await self._orderbook_integration.stop()
            step += 1
            
            logger.info(f"{step}/{total_steps} Transitioning to SHUTDOWN state...")
            await self._state_machine.transition_to(
                V3State.SHUTDOWN,
                context="Graceful shutdown initiated"
            )
            step += 1
            
            logger.info(f"{step}/{total_steps} Stopping State Machine...")
            await self._state_machine.stop()
            step += 1
            
            logger.info(f"{step}/{total_steps} Stopping WebSocket Manager...")
            await self._websocket_manager.stop()
            step += 1
            
            logger.info(f"{step}/{total_steps} Stopping Event Bus...")
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
                
                # Add trading client health if configured
                if self._trading_client_integration:
                    components_health["trading_client"] = self._trading_client_integration.is_healthy()
                
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
                            "unhealthy_components": unhealthy,
                            "session_cleanup_triggered": True  # Health failure will trigger session cleanup
                        }
                    )
                
                # If in ERROR state and everything is healthy, attempt recovery
                elif self._state_machine.current_state == V3State.ERROR and all_healthy:
                    logger.info("All components healthy, attempting recovery from ERROR state...")
                    
                    # Ensure session is ready for recovery
                    session_ready = await self._orderbook_integration.ensure_session_for_recovery()
                    if not session_ready:
                        logger.warning("Session not ready for recovery, will retry next health check")
                        continue
                    
                    # Check if orderbook integration has received snapshots to determine recovery state
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
            
            # Get session information if available
            session_info = {}
            if hasattr(self._orderbook_integration, '_client') and self._orderbook_integration._client:
                client_stats = self._orderbook_integration._client.get_stats()
                session_info = {
                    "session_id": client_stats.get("session_id"),
                    "session_state": client_stats.get("session_state", "unknown"),
                    "health_state": client_stats.get("health_state", "unknown")
                }
            
            uptime = time.time() - self._started_at if self._started_at else 0
            
            # Get Kalshi API ping health from orderbook integration
            ping_health = health_details.get("ping_health", "unknown")
            last_ping_age = health_details.get("last_ping_age_seconds")
            
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
                    # Add Kalshi API ping health
                    "ping_health": ping_health,
                    "last_ping_age": last_ping_age,
                    # Add connection status fields (fix for Issue 2)
                    "connection_established": health_details.get("connection_established"),
                    "first_snapshot_received": health_details.get("first_snapshot_received"),
                    # Add session information for debugging
                    "session_id": session_info.get("session_id"),
                    "session_state": session_info.get("session_state"),
                    "health_state": session_info.get("health_state")
                },
                timestamp=time.time()
            )
            
            await self._event_bus.publish(event)
            
            # Debug log to verify fields are present
            logger.debug(
                f"Broadcasting status with connection fields: "
                f"connection_established={event.metrics.get('connection_established')}, "
                f"first_snapshot_received={event.metrics.get('first_snapshot_received')}, "
                f"ping_health={event.metrics.get('ping_health')}, "
                f"last_ping_age={event.metrics.get('last_ping_age')}"
            )
            
            # Log summary with session information
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
        
        components = {
            "state_machine": self._state_machine.get_health_details(),
            "event_bus": self._event_bus.get_health_details(),
            "websocket_manager": self._websocket_manager.get_health_details(),
            "orderbook_integration": self._orderbook_integration.get_health_details()
        }
        
        # Add trading client if configured
        if self._trading_client_integration:
            components["trading_client"] = self._trading_client_integration.get_health_details()
        
        status = {
            "running": self._running,
            "uptime": uptime,
            "state": self._state_machine.current_state.value,
            "environment": self._config.get_environment_name(),
            "markets": self._config.market_tickers,
            "components": components
        }
        
        # Add trading mode if trading client is configured
        if self._trading_client_integration:
            status["trading_mode"] = self._trading_client_integration._client.mode
        
        return status
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        if not self._running:
            return False
        
        health_checks = [
            self._state_machine.is_healthy(),
            self._event_bus.is_healthy(),
            self._websocket_manager.is_healthy(),
            self._orderbook_integration.is_healthy()
        ]
        
        # Add trading client health if configured
        if self._trading_client_integration:
            health_checks.append(self._trading_client_integration.is_healthy())
        
        return all(health_checks)
    
    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "healthy": self.is_healthy(),
            "status": "running" if self._running else "stopped",
            "state": self._state_machine.current_state.value,
            "uptime": time.time() - self._started_at if self._started_at else 0
        }