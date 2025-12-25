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
from .event_bus import EventBus, EventType, TraderStatusEvent
from .websocket_manager import V3WebSocketManager
from .state_container import V3StateContainer
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
        
        # Initialize state container
        self._state_container = V3StateContainer()
        
        self._started_at: Optional[float] = None
        self._running = False
        
        # Health monitoring task
        self._health_task: Optional[asyncio.Task] = None
        
        # Status reporting task
        self._status_task: Optional[asyncio.Task] = None
        
        # Trading state broadcast task
        self._trading_state_task: Optional[asyncio.Task] = None
        
        # Main event loop task
        self._event_loop_task: Optional[asyncio.Task] = None
        
        logger.info("V3 Coordinator initialized")
    
    async def start(self) -> None:
        """Start the V3 trader system - just initialization."""
        if self._running:
            logger.warning("V3 Coordinator is already running")
            return
        
        try:
            # Phase 1: Initialize components
            await self._initialize_components()
            
            # Phase 2: Establish connections
            await self._establish_connections()
            
            # Phase 3: Start event loop
            self._running = True
            self._event_loop_task = asyncio.create_task(self._run_event_loop())
            
            logger.info("=" * 60)
            logger.info("✅ TRADER V3 STARTED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Failed to start V3 Coordinator: {e}")
            await self.stop()
            raise
    
    async def _initialize_components(self) -> None:
        """Initialize all core components in order."""
        logger.info("=" * 60)
        logger.info("STARTING TRADER V3")
        logger.info(f"Environment: {self._config.get_environment_name()}")
        logger.info(f"Markets: {', '.join(self._config.market_tickers[:3])}{'...' if len(self._config.market_tickers) > 3 else ''}")
        logger.info("=" * 60)
        
        self._started_at = time.time()
        
        # Start components (no connections yet)
        logger.info("1/3 Starting Event Bus...")
        await self._event_bus.start()
        
        logger.info("2/3 Starting WebSocket Manager...")
        self._websocket_manager.set_coordinator(self)
        await self._websocket_manager.start()
        
        logger.info("3/3 Starting State Machine...")
        await self._state_machine.start()
        
        await self._emit_status_update("System initializing")
    
    async def _establish_connections(self) -> None:
        """Establish all external connections."""
        # Orderbook connection
        await self._connect_orderbook()
        
        # Trading client connection (if configured)
        if self._trading_client_integration:
            await self._connect_trading_client()
            await self._sync_trading_state()
        
        # Transition to READY with actual metrics
        await self._transition_to_ready()
    
    async def _connect_orderbook(self) -> None:
        """Connect to orderbook WebSocket and wait for data."""
        logger.info("Connecting to orderbook...")
        
        # Start integration
        await self._orderbook_integration.start()
        
        # Wait for connection
        logger.info("🔄 Waiting for orderbook connection...")
        connection_success = await self._orderbook_integration.wait_for_connection(timeout=30.0)
        
        if not connection_success:
            raise RuntimeError("Failed to connect to orderbook WebSocket")
        
        # Wait for first snapshot
        logger.info("Waiting for initial orderbook snapshot...")
        data_flowing = await self._orderbook_integration.wait_for_first_snapshot(timeout=10.0)
        
        if not data_flowing:
            logger.warning("No orderbook data received - continuing anyway")
        
        # Collect metrics and transition
        metrics = self._orderbook_integration.get_metrics()
        health_details = self._orderbook_integration.get_health_details()
        
        await self._state_machine.transition_to(
            V3State.ORDERBOOK_CONNECT,
            context=f"Connected to {metrics['markets_connected']} markets",
            metadata={
                "ws_url": self._config.ws_url,
                "markets": self._config.market_tickers[:2] + (["...and {} more".format(len(self._config.market_tickers) - 2)] if len(self._config.market_tickers) > 2 else []),
                "market_count": len(self._config.market_tickers),
                "environment": self._config.get_environment_name(),
                "markets_connected": metrics["markets_connected"],
                "snapshots_received": metrics["snapshots_received"],
                "deltas_received": metrics["deltas_received"],
                "connection_established": health_details.get("connection_established"),
                "first_snapshot_received": health_details.get("first_snapshot_received")
            }
        )
        
        # Update state container
        self._state_container.update_machine_state(
            V3State.ORDERBOOK_CONNECT,
            f"Connected to {metrics['markets_connected']} markets",
            {
                "markets_connected": metrics["markets_connected"],
                "snapshots_received": metrics["snapshots_received"],
                "ws_url": self._config.ws_url
            }
        )
        
        await self._emit_status_update(f"Connected to {metrics['markets_connected']} markets")
    
    async def _connect_trading_client(self) -> None:
        """Connect to trading API."""
        logger.info("Connecting to trading API...")
        
        await self._state_machine.transition_to(
            V3State.TRADING_CLIENT_CONNECT,
            context="Connecting to trading API",
            metadata={
                "mode": self._trading_client_integration._client.mode,
                "environment": self._config.get_environment_name(),
                "api_url": self._trading_client_integration.api_url
            }
        )
        
        logger.info("🔌 TRADING_CLIENT_CONNECT: Actually connecting to Kalshi Trading API...")
        connected = await self._trading_client_integration.wait_for_connection(timeout=30.0)
        
        if not connected:
            raise RuntimeError("Failed to connect to trading API")
    
    async def _sync_trading_state(self) -> None:
        """Perform initial trading state sync."""
        logger.info("🔄 Syncing with Kalshi...")
        
        state, changes = await self._trading_client_integration.sync_with_kalshi()
        
        # Store in container
        state_changed = self._state_container.update_trading_state(state, changes)
        
        # Emit trading state if changed
        if state_changed:
            await self._emit_trading_state()
        
        # Log the sync results
        if changes and (abs(changes.balance_change) > 0 or 
                      changes.position_count_change != 0 or 
                      changes.order_count_change != 0):
            logger.info(
                f"Kalshi sync complete - Balance: {state.balance} cents ({changes.balance_change:+d}), "
                f"Positions: {state.position_count} ({changes.position_count_change:+d}), "
                f"Orders: {state.order_count} ({changes.order_count_change:+d})"
            )
        else:
            logger.info(
                f"Initial Kalshi sync - Balance: {state.balance} cents, "
                f"Positions: {state.position_count}, Orders: {state.order_count}"
            )
        
        await self._state_machine.transition_to(
            V3State.KALSHI_DATA_SYNC,
            context=f"Synced: {state.position_count} positions, {state.order_count} orders",
            metadata={
                "mode": self._trading_client_integration._client.mode,
                "sync_type": "initial",
                "balance": state.balance,
                "portfolio_value": state.portfolio_value,
                "positions": state.position_count,
                "orders": state.order_count
            }
        )
    
    async def _transition_to_ready(self) -> None:
        """Transition to READY state with collected metrics."""
        # Gather metrics
        orderbook_metrics = self._orderbook_integration.get_metrics()
        health_details = self._orderbook_integration.get_health_details()
        
        # Build READY state metadata
        ready_metadata = {
            "markets_connected": orderbook_metrics["markets_connected"],
            "snapshots_received": orderbook_metrics["snapshots_received"],
            "deltas_received": orderbook_metrics["deltas_received"],
            "connection_established": health_details["connection_established"],
            "first_snapshot_received": health_details["first_snapshot_received"],
            "environment": self._config.get_environment_name()
        }
        
        # Add trading client info if available
        if self._trading_client_integration and self._state_container.trading_state:
            trading_state = self._state_container.trading_state
            ready_metadata["trading_client"] = {
                "connected": True,
                "mode": self._trading_client_integration._client.mode,
                "balance": trading_state.balance,
                "portfolio_value": trading_state.portfolio_value,
                "positions": trading_state.position_count,
                "orders": trading_state.order_count
            }
        
        # Determine context
        if orderbook_metrics["snapshots_received"] > 0:
            if self._trading_client_integration:
                context = f"System fully operational with {orderbook_metrics['markets_connected']} markets and trading enabled"
            else:
                context = f"System fully operational with {orderbook_metrics['markets_connected']} markets (orderbook only)"
        else:
            context = f"System connected (waiting for data) - {orderbook_metrics['markets_connected']} markets"
        
        await self._state_machine.transition_to(
            V3State.READY,
            context=context,
            metadata=ready_metadata
        )
        
        # Emit trading state immediately when READY
        if self._trading_client_integration and self._state_container.trading_state:
            await self._emit_trading_state()
        
        # Emit ready status
        status_msg = f"System ready with {len(self._config.market_tickers)} markets"
        if self._trading_client_integration:
            status_msg += f" (trading enabled in {self._trading_client_integration._client.mode} mode)"
        await self._emit_status_update(status_msg)
    
    async def _run_event_loop(self) -> None:
        """
        Main event loop - handles all periodic operations.
        This is the heart of the V3 trader after startup.
        """
        # Start monitoring tasks
        self._start_monitoring_tasks()
        
        last_sync_time = time.time()
        sync_interval = 30.0  # Sync every 30 seconds
        
        logger.info("Event loop started")
        
        while self._running:
            try:
                current_state = self._state_machine.current_state
                
                # State-specific handlers
                if current_state == V3State.READY:
                    # Check if sync needed
                    if self._trading_client_integration and \
                       time.time() - last_sync_time > sync_interval:
                        # Emit system activity for sync start
                        await self._event_bus.emit_system_activity(
                            activity_type="sync",
                            message="Starting periodic sync with Kalshi",
                            metadata={"sync_interval": sync_interval}
                        )
                        await self._handle_trading_sync()
                        last_sync_time = time.time()
                    
                    # Future: This is where ACTING state logic would go
                    # if self._has_pending_actions():
                    #     await self._handle_acting_state()
                    
                elif current_state == V3State.ERROR:
                    # Error recovery is handled by _monitor_health()
                    # Just sleep longer in error state
                    await asyncio.sleep(1.0)
                    continue
                
                elif current_state == V3State.SHUTDOWN:
                    # Exit loop on shutdown
                    break
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info("Event loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event loop: {e}")
                await self._state_machine.enter_error_state(
                    "Event loop error", e
                )
        
        logger.info("Event loop stopped")
    
    def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        # Health monitoring - critical for error recovery
        self._health_task = asyncio.create_task(self._monitor_health())
        
        # Status reporting - for UI updates
        self._status_task = asyncio.create_task(self._report_status())
        
        # Trading state monitoring - now simplified to just track version changes
        self._trading_state_task = asyncio.create_task(self._monitor_trading_state())
    
    async def _handle_trading_sync(self) -> None:
        """
        Handle periodic trading state synchronization.
        Extracted from _monitor_trading_state() lines 671-702.
        """
        if not self._trading_client_integration:
            return
        
        if self._state_machine.current_state != V3State.READY:
            return
        
        logger.debug("Performing periodic trading state sync...")
        try:
            # Use the sync service to get fresh data
            state, changes = await self._trading_client_integration.sync_with_kalshi()
            
            # Always update sync timestamp
            state.sync_timestamp = time.time()
            
            # Update state container
            state_changed = self._state_container.update_trading_state(state, changes)
            
            # Log and emit system activity for sync results
            if changes and (abs(changes.balance_change) > 0 or 
                          changes.position_count_change != 0 or 
                          changes.order_count_change != 0):
                logger.info(
                    f"Trading state updated - "
                    f"Balance: ${state.balance/100:.2f} ({changes.balance_change:+d} cents), "
                    f"Positions: {state.position_count} ({changes.position_count_change:+d}), "
                    f"Orders: {state.order_count} ({changes.order_count_change:+d})"
                )
                # Emit activity with changes
                await self._event_bus.emit_system_activity(
                    activity_type="sync",
                    message=f"Sync complete: Balance {changes.balance_change:+d} cents, Positions {changes.position_count_change:+d}, Orders {changes.order_count_change:+d}",
                    metadata={
                        "sync_type": "periodic",
                        "balance_change": changes.balance_change,
                        "position_count_change": changes.position_count_change,
                        "order_count_change": changes.order_count_change,
                        "balance": state.balance,
                        "position_count": state.position_count,
                        "order_count": state.order_count
                    }
                )
            else:
                # Emit activity for no-change sync
                await self._event_bus.emit_system_activity(
                    activity_type="sync",
                    message="Sync complete: No changes",
                    metadata={
                        "sync_type": "periodic",
                        "no_changes": True,
                        "balance": state.balance,
                        "position_count": state.position_count,
                        "order_count": state.order_count
                    }
                )
            
            # Broadcast state (even if unchanged, to update sync timestamp)
            if state_changed or True:  # Always broadcast on sync
                await self._emit_trading_state()
                
        except Exception as e:
            logger.error(f"Periodic trading sync failed: {e}")
            # Emit error activity
            await self._event_bus.emit_system_activity(
                activity_type="sync",
                message=f"Sync failed: {str(e)}",
                metadata={"error": str(e), "sync_type": "periodic"}
            )
            # Don't transition to ERROR for sync failures - just log and continue
    
    async def stop(self) -> None:
        """Stop the V3 trader system."""
        if not self._running:
            return
        
        logger.info("=" * 60)
        logger.info("STOPPING TRADER V3")
        logger.info("=" * 60)
        
        self._running = False
        
        # Cancel event loop task
        if self._event_loop_task:
            self._event_loop_task.cancel()
            try:
                await self._event_loop_task
            except asyncio.CancelledError:
                pass
        
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
        
        if self._trading_state_task:
            self._trading_state_task.cancel()
            try:
                await self._trading_state_task
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
                
                # Update state container with health status
                for name, healthy in components_health.items():
                    self._state_container.update_component_health(name, healthy)
                
                all_healthy = all(components_health.values())
                
                # Emit health check activity (only occasionally to avoid spam)
                if self._state_machine.current_state == V3State.READY:
                    # Emit health activity every 5th check (150 seconds) or if unhealthy
                    if not hasattr(self, '_health_check_count'):
                        self._health_check_count = 0
                    self._health_check_count += 1
                    
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
    
    async def _emit_trading_state(self) -> None:
        """Emit trading state update event."""
        try:
            trading_summary = self._state_container.get_trading_summary()
            
            if not trading_summary.get("has_state"):
                return  # No trading state to broadcast
            
            # Create a custom event for trading state
            # Since EventBus doesn't have a TradingStateEvent yet, we'll use the websocket directly
            await self._websocket_manager.broadcast_message("trading_state", {
                "timestamp": time.time(),
                "version": trading_summary["version"],
                "balance": trading_summary["balance"],
                "portfolio_value": trading_summary["portfolio_value"],
                "position_count": trading_summary["position_count"],
                "order_count": trading_summary["order_count"],
                "positions": trading_summary["positions"],
                "open_orders": trading_summary["open_orders"],
                "sync_timestamp": trading_summary["sync_timestamp"],
                "changes": trading_summary.get("changes")
            })
            
            logger.debug(f"Broadcast trading state v{trading_summary['version']}")
            
        except Exception as e:
            logger.error(f"Error emitting trading state: {e}")
    
    async def _emit_status_update(self, context: str = "") -> None:
        """Emit status update event immediately."""
        try:
            # Gather metrics from orderbook integration (includes persistent metrics)
            orderbook_metrics = self._orderbook_integration.get_metrics()
            ws_stats = self._websocket_manager.get_stats()
            
            # Metrics removed - no persistence needed
            
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
            
            # Calculate uptime
            uptime = time.time() - self._started_at if self._started_at else 0.0
            
            # Get Kalshi API ping health from orderbook integration
            ping_health = health_details.get("ping_health", "unknown")
            # Get last message age from orderbook integration
            last_ping_age = health_details.get("last_ping_age_seconds")
            
            # Determine API connection status for the UI
            api_connected = False
            if self._trading_client_integration:
                trading_metrics = self._trading_client_integration.get_metrics()
                if trading_metrics.get("connected"):
                    api_connected = True
            elif orderbook_metrics["markets_connected"] > 0:
                # If we have orderbook data but no trading client, we're still connected to Kalshi
                api_connected = True
            
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
                    # Use local session metrics
                    "snapshots_received": orderbook_metrics["snapshots_received"],
                    "deltas_received": orderbook_metrics["deltas_received"],
                    "ws_clients": ws_stats["active_connections"],
                    "ws_messages_sent": ws_stats.get("total_messages_sent", 0),
                    "context": context,
                    # Add health status INSIDE metrics
                    "health": "healthy" if self.is_healthy() else "unhealthy",
                    # Add Kalshi API ping health
                    "ping_health": ping_health,
                    "last_ping_age": last_ping_age,
                    # Add API connection status as boolean
                    "api_connected": api_connected,
                    # Add connection status fields
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
    
    async def _monitor_trading_state(self) -> None:
        """Monitor trading state version for broadcasts."""
        last_version = -1  # Start at -1 to ensure first check always broadcasts
        
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                
                # Just check for version changes
                # Syncing now happens in main event loop via _handle_trading_sync
                current_version = self._state_container.trading_state_version
                
                # Only broadcast if state changed
                if current_version > last_version:
                    await self._emit_trading_state()
                    last_version = current_version
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trading state monitoring: {e}")
    
    async def _update_metrics_regularly(self) -> None:
        """Update metrics and save periodically."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Update every second for real-time last_ping_age
                
                # Emit status with updated last_ping_age
                await self._emit_status_update()
                
                # Metrics persistence removed
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics update: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # Calculate uptime directly
        uptime = time.time() - self._started_at if self._started_at else 0.0
        
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
    
    @property
    def state_container(self) -> V3StateContainer:
        """Get state container for external access."""
        return self._state_container