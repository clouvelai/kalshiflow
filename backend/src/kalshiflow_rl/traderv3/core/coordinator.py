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
from .event_bus import EventBus
from .websocket_manager import V3WebSocketManager
from .state_container import V3StateContainer
from .health_monitor import V3HealthMonitor
from .status_reporter import V3StatusReporter
from .trading_flow_orchestrator import TradingFlowOrchestrator
from ..clients.orderbook_integration import V3OrderbookIntegration
from ..config.environment import V3Config
from ..services.trading_decision_service import TradingDecisionService, TradingStrategy

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
        
        # Initialize health monitor
        self._health_monitor = V3HealthMonitor(
            config=config,
            state_machine=state_machine,
            event_bus=event_bus,
            websocket_manager=websocket_manager,
            state_container=self._state_container,
            orderbook_integration=orderbook_integration,
            trading_client_integration=trading_client_integration
        )
        
        # Initialize status reporter
        self._status_reporter = V3StatusReporter(
            config=config,
            state_machine=state_machine,
            event_bus=event_bus,
            websocket_manager=websocket_manager,
            state_container=self._state_container,
            orderbook_integration=orderbook_integration,
            trading_client_integration=trading_client_integration
        )
        
        # Initialize trading decision service (if trading client available)
        self._trading_service = None
        self._trading_orchestrator = None
        if trading_client_integration:
            # Default to HOLD strategy for safety
            self._trading_service = TradingDecisionService(
                trading_client=trading_client_integration,
                state_container=self._state_container,
                event_bus=event_bus,
                strategy=TradingStrategy.HOLD
            )
            
            # Initialize trading flow orchestrator
            self._trading_orchestrator = TradingFlowOrchestrator(
                config=config,
                trading_client=trading_client_integration,
                orderbook_integration=orderbook_integration,
                trading_service=self._trading_service,
                state_container=self._state_container,
                event_bus=event_bus,
                state_machine=state_machine
            )
        
        self._started_at: Optional[float] = None
        self._running = False
        
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
        self._status_reporter.set_started_at(self._started_at)
        
        # Start components (no connections yet)
        logger.info("1/3 Starting Event Bus...")
        await self._event_bus.start()
        
        logger.info("2/3 Starting WebSocket Manager...")
        self._websocket_manager.set_coordinator(self)
        await self._websocket_manager.start()
        
        logger.info("3/3 Starting State Machine...")
        await self._state_machine.start()
        
        await self._status_reporter.emit_status_update("System initializing")
    
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
            logger.warning("⚠️ DEGRADED MODE: Orderbook WebSocket unavailable - continuing without live market data")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="⚠️ DEGRADED MODE: Running without orderbook connection",
                metadata={"degraded": True, "reason": "WebSocket unavailable", "severity": "warning"}
            )
            
            # Still transition to ORDERBOOK_CONNECT state but with degraded status
            await self._state_machine.transition_to(
                V3State.ORDERBOOK_CONNECT,
                context="Degraded mode - no orderbook connection",
                metadata={
                    "ws_url": self._config.ws_url,
                    "markets": self._config.market_tickers[:2] + (["...and {} more".format(len(self._config.market_tickers) - 2)] if len(self._config.market_tickers) > 2 else []),
                    "market_count": len(self._config.market_tickers),
                    "environment": self._config.get_environment_name(),
                    "markets_connected": 0,
                    "snapshots_received": 0,
                    "deltas_received": 0,
                    "connection_established": False,
                    "first_snapshot_received": False,
                    "degraded": True
                }
            )
            
            # Update state container with degraded status
            self._state_container.update_machine_state(
                V3State.ORDERBOOK_CONNECT,
                "Degraded mode - no orderbook connection",
                {
                    "markets_connected": 0,
                    "snapshots_received": 0,
                    "ws_url": self._config.ws_url,
                    "degraded": True
                }
            )
            
            await self._status_reporter.emit_status_update("Running in degraded mode without orderbook")
            # Continue to ready state in degraded mode - don't return here
        else:
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
        
        await self._status_reporter.emit_status_update(f"Connected to {metrics['markets_connected']} markets")
    
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
        
        # Setup order group for portfolio limits (simplified)
        logger.info("Setting up order group for portfolio limits...")
        order_group_id = await self._trading_client_integration.create_or_get_order_group(
            contracts_limit=10000  # 10K contracts limit
        )
        
        if order_group_id:
            logger.info(f"✅ Order group ready: {order_group_id[:8]}...")
            await self._event_bus.emit_system_activity(
                activity_type="order_group",
                message=f"Order group ready: {order_group_id[:8]}... (10K contract limit)",
                metadata={"order_group_id": order_group_id, "contracts_limit": 10000}
            )
        else:
            logger.warning("⚠️ Order groups not available - continuing without portfolio limits")
            await self._event_bus.emit_system_activity(
                activity_type="order_group",
                message="⚠️ DEGRADED MODE: Order groups unavailable",
                metadata={"degraded": True, "severity": "warning"}
            )
    
    async def _sync_trading_state(self) -> None:
        """Perform initial trading state sync."""
        logger.info("🔄 Syncing with Kalshi...")
        
        state, changes = await self._trading_client_integration.sync_with_kalshi()
        
        # Store in container
        state_changed = self._state_container.update_trading_state(state, changes)
        
        # Emit trading state if changed
        if state_changed:
            await self._status_reporter.emit_trading_state()
        
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
        
        # Check for degraded mode
        degraded = not health_details["connection_established"] and orderbook_metrics["markets_connected"] == 0
        
        # Determine context based on connection status
        if degraded:
            if self._trading_client_integration:
                context = f"DEGRADED MODE: Trading enabled without orderbook (paper mode)"
            else:
                context = f"DEGRADED MODE: No orderbook connection available"
            ready_metadata["degraded"] = True
        elif orderbook_metrics["snapshots_received"] > 0:
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
        
        # Update state container with degraded flag so health monitor can see it
        self._state_container.update_machine_state(
            V3State.READY,
            context,
            ready_metadata  # This includes degraded=True when appropriate
        )
        
        # Emit trading state immediately when READY
        if self._trading_client_integration and self._state_container.trading_state:
            await self._status_reporter.emit_trading_state()
        
        # Emit ready status
        status_msg = f"System ready with {len(self._config.market_tickers)} markets"
        if self._trading_client_integration:
            status_msg += f" (trading enabled in {self._trading_client_integration._client.mode} mode)"
        await self._status_reporter.emit_status_update(status_msg)
    
    async def _run_event_loop(self) -> None:
        """
        Main event loop - handles all periodic operations.
        This is the heart of the V3 trader after startup.
        """
        # Start monitoring tasks
        self._start_monitoring_tasks()
        
        last_sync_time = time.time()
        sync_interval = 30.0  # Sync every 30 seconds (for non-trading syncs)
        
        logger.info("Event loop started")
        
        while self._running:
            try:
                current_state = self._state_machine.current_state
                
                # State-specific handlers
                if current_state == V3State.READY:
                    # Use orchestrator for trading flow if available
                    if self._trading_orchestrator:
                        # Orchestrator handles its own sync and trading cycles
                        cycle_run = await self._trading_orchestrator.check_and_run_cycle()
                        
                        # If no cycle was run, check for periodic sync
                        if not cycle_run and self._trading_client_integration and \
                           time.time() - last_sync_time > sync_interval:
                            # Emit system activity for sync start
                            await self._event_bus.emit_system_activity(
                                activity_type="sync",
                                message="Starting periodic sync with Kalshi",
                                metadata={"sync_interval": sync_interval, "severity": "info"}
                            )
                            await self._handle_trading_sync()
                            last_sync_time = time.time()
                    else:
                        # No trading orchestrator - just do periodic syncs
                        if self._trading_client_integration and \
                           time.time() - last_sync_time > sync_interval:
                            await self._event_bus.emit_system_activity(
                                activity_type="sync",
                                message="Starting periodic sync with Kalshi",
                                metadata={"sync_interval": sync_interval, "severity": "info"}
                            )
                            await self._handle_trading_sync()
                            last_sync_time = time.time()
                    
                elif current_state == V3State.ERROR:
                    # In ERROR state, still allow trading syncs if trading client is available
                    # This supports degraded mode where orderbook is down but trading still works
                    if self._trading_client_integration and \
                       time.time() - last_sync_time > sync_interval:
                        logger.info("Performing trading sync in ERROR state (degraded mode)")
                        await self._event_bus.emit_system_activity(
                            activity_type="sync",
                            message="Trading sync in degraded mode",
                            metadata={"state": "error", "sync_interval": sync_interval, "severity": "info"}
                        )
                        try:
                            await self._handle_trading_sync()
                            last_sync_time = time.time()
                        except Exception as e:
                            logger.error(f"Trading sync failed in ERROR state: {e}")
                    
                    # Sleep to prevent CPU spinning
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
        # Start health monitor service
        asyncio.create_task(self._health_monitor.start())
        
        # Start status reporter service
        asyncio.create_task(self._status_reporter.start())
    
    async def _handle_trading_sync(self) -> None:
        """
        Handle periodic trading state synchronization.
        Extracted from _monitor_trading_state() lines 671-702.
        """
        if not self._trading_client_integration:
            return
        
        # Allow trading sync in READY or ERROR state to support degraded mode
        # This enables trading data display even when orderbook is down
        if self._state_machine.current_state not in [V3State.READY, V3State.ERROR]:
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
                        "order_count": state.order_count,
                        "severity": "info"
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
                        "order_count": state.order_count,
                        "severity": "info"
                    }
                )
            
            # Broadcast state (even if unchanged, to update sync timestamp)
            # Include order group status in metadata if available
            sync_metadata = {
                "sync_type": "periodic",
                "balance": state.balance,
                "position_count": state.position_count,
                "order_count": state.order_count
            }
            
            # Add order group info if available
            if self._trading_client_integration and self._trading_client_integration.has_order_group:
                sync_metadata["order_group"] = {
                    "id": self._trading_client_integration.order_group_id[:8] if self._trading_client_integration.order_group_id else "none",
                    "supported": self._trading_client_integration.order_groups_supported
                }
            elif self._trading_client_integration and not self._trading_client_integration.order_groups_supported:
                sync_metadata["order_group"] = {
                    "supported": False,
                    "message": "Order groups unavailable (Demo API limitation)"
                }
            
            if state_changed or True:  # Always broadcast on sync
                await self._status_reporter.emit_trading_state()
                
        except Exception as e:
            logger.error(f"Periodic trading sync failed: {e}")
            # Emit error activity
            await self._event_bus.emit_system_activity(
                activity_type="sync",
                message=f"Sync failed: {str(e)}",
                metadata={"error": str(e), "sync_type": "periodic", "severity": "error"}
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
        
        # Stop health monitor
        await self._health_monitor.stop()
        
        # Stop status reporter
        await self._status_reporter.stop()
        
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
    
    
    async def _update_metrics_regularly(self) -> None:
        """Update metrics and save periodically."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Update every second for real-time last_ping_age
                
                # Emit status with updated last_ping_age
                await self._status_reporter.emit_status_update("Metrics updated")
                
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
            "orderbook_integration": self._orderbook_integration.get_health_details(),
            "health_monitor": self._health_monitor.get_status(),
            "status_reporter": self._status_reporter.get_status()
        }
        
        # Add trading service if configured
        if self._trading_service:
            components["trading_service"] = self._trading_service.get_stats()
        
        # Add trading orchestrator if configured
        if self._trading_orchestrator:
            components["trading_orchestrator"] = self._trading_orchestrator.get_stats()
        
        # Add trading client if configured
        if self._trading_client_integration:
            components["trading_client"] = self._trading_client_integration.get_health_details()
        
        # Get metrics from various sources
        orderbook_metrics = self._orderbook_integration.get_metrics()
        ws_stats = self._websocket_manager.get_stats()
        
        # Build metrics similar to status_reporter
        metrics = {
            "uptime": uptime,
            "state": self._state_machine.current_state.value,
            "markets_connected": orderbook_metrics["markets_connected"],
            "snapshots_received": orderbook_metrics["snapshots_received"],
            "deltas_received": orderbook_metrics["deltas_received"],
            "ws_clients": ws_stats["active_connections"],
            "ws_messages_sent": ws_stats.get("total_messages_sent", 0),
            "api_url": self._config.api_url,
            "ws_url": self._config.ws_url
        }
        
        status = {
            "running": self._running,
            "uptime": uptime,
            "state": self._state_machine.current_state.value,
            "environment": self._config.get_environment_name(),
            "markets": self._config.market_tickers,
            "components": components,
            "metrics": metrics
        }
        
        # Add trading mode and strategy if configured
        if self._trading_client_integration:
            status["trading_mode"] = self._trading_client_integration._client.mode
            if self._trading_service:
                status["trading_strategy"] = self._trading_service.get_stats()["strategy"]
        
        return status
    
    def is_healthy(self) -> bool:
        """Check if system is healthy."""
        if not self._running:
            return False
        
        health_checks = [
            self._state_machine.is_healthy(),
            self._event_bus.is_healthy(),
            self._websocket_manager.is_healthy(),
            self._orderbook_integration.is_healthy(),
            self._health_monitor.is_healthy()
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
    
    @property
    def trading_service(self) -> Optional[TradingDecisionService]:
        """Get trading service for external access."""
        return self._trading_service
    
    def set_trading_strategy(self, strategy: TradingStrategy) -> None:
        """Set the trading strategy."""
        if not self._trading_service:
            logger.warning("No trading service configured")
            return
        
        self._trading_service.set_strategy(strategy)
        logger.info(f"Trading strategy set to: {strategy.value}")