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
    from ..clients.trades_integration import V3TradesIntegration
    from ..clients.position_listener import PositionListener
    from ..services.whale_tracker import WhaleTracker

from .state_machine import TraderStateMachine as V3StateMachine, TraderState as V3State
from .event_bus import EventBus
from .websocket_manager import V3WebSocketManager
from .state_container import V3StateContainer
from .health_monitor import V3HealthMonitor, CRITICAL_COMPONENTS
from .status_reporter import V3StatusReporter
from .trading_flow_orchestrator import TradingFlowOrchestrator
from ..clients.orderbook_integration import V3OrderbookIntegration
from ..config.environment import V3Config
from ..services.trading_decision_service import TradingDecisionService, TradingStrategy
from ..services.whale_execution_service import WhaleExecutionService
from ..services.yes_80_90_service import Yes8090Service

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
        trading_client_integration: Optional['V3TradingClientIntegration'] = None,
        trades_integration: Optional['V3TradesIntegration'] = None,
        whale_tracker: Optional['WhaleTracker'] = None
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
            trades_integration: Optional trades integration for public trades stream
            whale_tracker: Optional whale tracker for big bet detection
        """
        self._config = config
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._orderbook_integration = orderbook_integration
        self._trading_client_integration = trading_client_integration
        self._trades_integration = trades_integration
        self._whale_tracker = whale_tracker
        
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
            trading_client_integration=trading_client_integration,
            trades_integration=trades_integration,
            whale_tracker=whale_tracker
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
            # Get trading strategy from config (default: HOLD for safety)
            strategy = config.trading_strategy if hasattr(config, 'trading_strategy') else TradingStrategy.HOLD

            self._trading_service = TradingDecisionService(
                trading_client=trading_client_integration,
                state_container=self._state_container,
                event_bus=event_bus,
                strategy=strategy,
                whale_tracker=whale_tracker
            )

            # Set trading service on websocket manager for followed whale IDs
            self._websocket_manager.set_trading_service(self._trading_service)

            # Set state container for immediate trading state on client connect
            self._websocket_manager.set_state_container(self._state_container)

            # Initialize trading flow orchestrator
            self._trading_orchestrator = TradingFlowOrchestrator(
                config=config,
                trading_client=trading_client_integration,
                orderbook_integration=orderbook_integration,
                trading_service=self._trading_service,
                state_container=self._state_container,
                event_bus=event_bus,
                state_machine=state_machine,
                whale_tracker=whale_tracker
            )

            # Log strategy configuration
            if strategy == TradingStrategy.WHALE_FOLLOWER:
                if whale_tracker:
                    logger.info("WHALE_FOLLOWER strategy enabled with whale tracker")
                else:
                    logger.warning("WHALE_FOLLOWER strategy enabled but whale tracker not available")

        # Initialize whale execution service (if trading client and whale tracker available)
        self._whale_execution_service: Optional[WhaleExecutionService] = None
        if trading_client_integration and whale_tracker and self._trading_service:
            self._whale_execution_service = WhaleExecutionService(
                event_bus=event_bus,
                trading_service=self._trading_service,
                state_container=self._state_container,
                whale_tracker=whale_tracker,
            )
            # Connect whale execution service to WebSocket manager for decision history
            self._websocket_manager.set_whale_execution_service(self._whale_execution_service)
            logger.info("WhaleExecutionService initialized for event-driven whale following")

        # Initialize YES 80-90c service (if trading client available and strategy is YES_80_90)
        self._yes_80_90_service: Optional[Yes8090Service] = None
        if trading_client_integration and self._trading_service:
            if strategy == TradingStrategy.YES_80_90:
                self._yes_80_90_service = Yes8090Service(
                    event_bus=event_bus,
                    trading_service=self._trading_service,
                    state_container=self._state_container,
                    min_price_cents=config.yes8090_min_price,
                    max_price_cents=config.yes8090_max_price,
                    min_liquidity=config.yes8090_min_liquidity,
                    max_spread_cents=config.yes8090_max_spread,
                    contracts_per_trade=config.yes8090_contracts,
                    tier_a_contracts=config.yes8090_tier_a_contracts,
                    max_concurrent=config.yes8090_max_concurrent,
                )
                logger.info("Yes8090Service initialized for YES at 80-90c trading strategy")

        # Position listener for real-time position updates (initialized later if trading client available)
        self._position_listener: Optional['PositionListener'] = None

        self._started_at: Optional[float] = None
        self._running = False

        # Main event loop task
        self._event_loop_task: Optional[asyncio.Task] = None

        # Monitoring tasks (tracked for proper cleanup)
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._status_reporter_task: Optional[asyncio.Task] = None

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
        await self._websocket_manager.start()
        
        logger.info("3/3 Starting State Machine...")
        await self._state_machine.start()
        
        await self._status_reporter.emit_status_update("System initializing")
    
    async def _establish_connections(self) -> None:
        """Establish all external connections."""
        # Orderbook connection
        await self._connect_orderbook()

        # Trades connection (if configured - optional, for whale detection)
        if self._trades_integration:
            await self._connect_trades()

        # Trading client connection (if configured)
        if self._trading_client_integration:
            await self._connect_trading_client()
            await self._sync_trading_state()

            # Cleanup orphaned orders on startup if configured
            if self._config.cleanup_on_startup:
                await self._cleanup_orphaned_orders()

            # Connect real-time position listener (non-blocking, falls back to polling on failure)
            await self._connect_position_listener()

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

    async def _connect_trades(self) -> None:
        """
        Connect to trades WebSocket for whale detection.

        This connection is OPTIONAL - the system continues without it
        if connection fails. Whale detection is a non-critical feature.
        """
        if not self._trades_integration:
            return

        logger.info("Connecting to trades WebSocket for whale detection...")

        # Start the trades integration (which starts the trades client)
        await self._trades_integration.start()

        # Wait for connection with timeout
        logger.info("Waiting for trades WebSocket connection...")
        connection_success = await self._trades_integration.wait_for_connection(timeout=30.0)

        if not connection_success:
            logger.warning(
                "Trades WebSocket connection failed - continuing without whale detection. "
                "This is non-critical, orderbook and trading features remain functional."
            )
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Whale detection unavailable (trades WS connection failed)",
                metadata={"degraded": True, "feature": "whale_detection", "severity": "warning"}
            )
            return

        # Start whale tracker after trades connection
        if self._whale_tracker:
            await self._whale_tracker.start()
            logger.info("Whale tracker started - monitoring for big bets")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Whale detection enabled - monitoring public trades",
                metadata={"feature": "whale_detection", "severity": "info"}
            )

            # Start whale execution service for event-driven following
            if self._whale_execution_service:
                await self._whale_execution_service.start()
                logger.info("Whale execution service started - event-driven following enabled")
                await self._event_bus.emit_system_activity(
                    activity_type="connection",
                    message="Whale execution service started - following whales immediately",
                    metadata={"feature": "whale_execution", "severity": "info"}
                )
        else:
            logger.warning("Trades connected but no whale tracker configured")

        # Wait briefly for first trade to confirm data flow
        trade_flowing = await self._trades_integration.wait_for_first_trade(timeout=10.0)
        if trade_flowing:
            metrics = self._trades_integration.get_metrics()
            logger.info(f"Trades data flowing: {metrics['trades_received']} trades received")
        else:
            logger.info("No trades received yet - this is normal during quiet market periods")

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
    
    async def _cleanup_orphaned_orders(self) -> None:
        """
        Cleanup previous order groups on startup.

        This is called during startup to reset any existing order groups
        from previous sessions, ensuring a clean slate for the new session.
        The current session's order group (created in _connect_trading_client)
        is preserved.
        """
        if not self._trading_client_integration:
            return

        logger.info("Cleaning up previous order groups...")

        try:
            # Get the current order group ID (created earlier in _connect_trading_client)
            current_order_group_id = self._trading_client_integration.get_order_group_id()

            # List all order groups (API doesn't support status filter)
            order_groups = await self._trading_client_integration.list_order_groups()

            if not order_groups:
                logger.info("No previous order groups to clean up")
                return

            deleted_count = 0
            skip_count = 0
            error_count = 0

            for group in order_groups:
                # API returns "id" field, not "order_group_id"
                group_id = group.get("id", "")
                if not group_id:
                    continue

                # Skip the current session's order group
                if group_id == current_order_group_id:
                    logger.debug(f"Skipping current order group: {group_id[:8]}...")
                    skip_count += 1
                    continue

                # Delete old order groups
                try:
                    success = await self._trading_client_integration.delete_order_group_by_id(group_id)
                    if success:
                        deleted_count += 1
                        logger.info(f"Deleted previous order group: {group_id[:8]}...")
                    else:
                        error_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete order group {group_id[:8]}...: {e}")
                    error_count += 1

            if deleted_count > 0 or error_count > 0:
                # Emit system activity for cleanup results
                await self._event_bus.emit_system_activity(
                    activity_type="cleanup",
                    message=f"Startup cleanup: {deleted_count} order groups deleted, {skip_count} preserved",
                    metadata={
                        "deleted_count": deleted_count,
                        "skip_count": skip_count,
                        "error_count": error_count,
                        "total_groups": len(order_groups),
                        "severity": "info" if error_count == 0 else "warning"
                    }
                )
                logger.info(
                    f"Startup cleanup complete: {deleted_count} order groups deleted, "
                    f"{skip_count} preserved, {error_count} errors"
                )
            else:
                logger.info("No previous order groups needed cleanup")

        except Exception as e:
            # Don't fail startup on cleanup errors - just log and continue
            logger.warning(f"Order group cleanup failed (non-critical): {e}")
            await self._event_bus.emit_system_activity(
                activity_type="cleanup",
                message=f"Startup cleanup failed: {str(e)}",
                metadata={"error": str(e), "severity": "warning"}
            )

    async def _connect_position_listener(self) -> None:
        """
        Connect to real-time position updates via WebSocket.

        Initializes the PositionListener which subscribes to the
        market_positions WebSocket channel for instant position updates.
        """
        if not self._trading_client_integration:
            return

        try:
            # Import here to avoid circular imports
            from ..clients.position_listener import PositionListener

            logger.info("Starting real-time position listener...")

            # Create position listener
            self._position_listener = PositionListener(
                event_bus=self._event_bus,
                ws_url=self._config.ws_url,
                reconnect_delay_seconds=5.0,
            )

            # Subscribe to position update events
            await self._event_bus.subscribe_to_market_position(self._handle_position_update)

            # Start the listener
            await self._position_listener.start()

            # Set on status reporter for health broadcasting
            self._status_reporter.set_position_listener(self._position_listener)

            logger.info("✅ Real-time position listener active")

            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Real-time position updates enabled",
                metadata={"channel": "market_positions", "severity": "info"}
            )

        except Exception as e:
            # Don't fail startup if position listener fails - fall back to polling
            logger.warning(f"Position listener failed (falling back to polling): {e}")
            self._position_listener = None
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Real-time positions unavailable: {e}",
                metadata={"fallback": "polling", "severity": "warning"}
            )

    async def _handle_position_update(self, event) -> None:
        """
        Handle real-time position update from WebSocket.

        Updates the state container and broadcasts to frontend.

        Args:
            event: MarketPositionEvent from event bus
        """
        try:
            ticker = event.market_ticker
            position_data = event.position_data

            # Update state container
            if self._state_container.update_single_position(ticker, position_data):
                # Broadcast updated state to frontend
                await self._status_reporter.emit_trading_state()

                logger.debug(f"Position update broadcast: {ticker}")

        except Exception as e:
            logger.error(f"Error handling position update: {e}")

    async def _sync_trading_state(self) -> None:
        """Perform initial trading state sync."""
        logger.info("🔄 Syncing with Kalshi...")

        state, changes = await self._trading_client_integration.sync_with_kalshi()

        # Store in container
        state_changed = self._state_container.update_trading_state(state, changes)

        # Initialize session P&L tracking on first sync
        # This captures starting balance/portfolio for session P&L calculation
        self._state_container.initialize_session_pnl(state.balance, state.portfolio_value)

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
        
        # Start YES 80-90c service if configured
        if self._yes_80_90_service:
            await self._yes_80_90_service.start()
            logger.info("Yes8090Service started - monitoring for YES at 80-90c signals")
            await self._event_bus.emit_system_activity(
                activity_type="strategy_active",
                message="YES 80-90c strategy active - scanning orderbooks",
                metadata={
                    "strategy": "YES_80_90",
                    "config": self._yes_80_90_service.get_stats().get("config", {}),
                    "severity": "info"
                }
            )

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
        # Start health monitor service (track for cleanup)
        self._health_monitor_task = asyncio.create_task(self._health_monitor.start())

        # Start status reporter service (track for cleanup)
        self._status_reporter_task = asyncio.create_task(self._status_reporter.start())
    
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

        # Cancel monitoring tasks before stopping components
        for task, _name in [
            (self._health_monitor_task, "health_monitor"),
            (self._status_reporter_task, "status_reporter"),
        ]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop health monitor
        await self._health_monitor.stop()

        # Stop status reporter
        await self._status_reporter.stop()
        
        # Stop components in reverse order
        try:
            # Calculate total steps dynamically based on configured components
            total_steps = 5  # Base steps: orderbook, state transition, state machine, websocket, event bus
            if self._trading_client_integration:
                total_steps += 1
            if self._position_listener:
                total_steps += 1
            if self._whale_execution_service:
                total_steps += 1
            if self._yes_80_90_service:
                total_steps += 1
            if self._whale_tracker:
                total_steps += 1
            if self._trades_integration:
                total_steps += 1

            step = 1

            # Stop YES 80-90c service first (trading strategy)
            if self._yes_80_90_service:
                logger.info(f"{step}/{total_steps} Stopping YES 80-90c Service...")
                await self._yes_80_90_service.stop()
                step += 1

            # Stop whale execution service (depends on whale tracker)
            if self._whale_execution_service:
                logger.info(f"{step}/{total_steps} Stopping Whale Execution Service...")
                await self._whale_execution_service.stop()
                step += 1

            # Stop whale tracker (depends on trades integration)
            if self._whale_tracker:
                logger.info(f"{step}/{total_steps} Stopping Whale Tracker...")
                await self._whale_tracker.stop()
                step += 1

            # Stop trades integration
            if self._trades_integration:
                logger.info(f"{step}/{total_steps} Stopping Trades Integration...")
                await self._trades_integration.stop()
                step += 1

            # Stop position listener (before trading client)
            if self._position_listener:
                logger.info(f"{step}/{total_steps} Stopping Position Listener...")
                await self._position_listener.stop()
                step += 1

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

        # Add trades integration if configured
        if self._trades_integration:
            components["trades_integration"] = self._trades_integration.get_health_details()

        # Add whale tracker if configured
        if self._whale_tracker:
            components["whale_tracker"] = self._whale_tracker.get_health_details()

        # Add whale execution service if configured
        if self._whale_execution_service:
            components["whale_execution_service"] = self._whale_execution_service.get_stats()

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
        """
        Check if system is healthy.

        Only checks CRITICAL components for overall health status.
        Non-critical components (orderbook, trades, whale tracker) can be
        degraded without affecting overall system health.

        Returns:
            True if running and all critical components are healthy.
        """
        if not self._running:
            return False

        # Map component names to their health check methods
        component_health_map = {
            "state_machine": self._state_machine.is_healthy,
            "event_bus": self._event_bus.is_healthy,
            "websocket_manager": self._websocket_manager.is_healthy,
        }

        # Only check CRITICAL components for overall health
        for component_name in CRITICAL_COMPONENTS:
            if component_name in component_health_map:
                if not component_health_map[component_name]():
                    return False

        # Health monitor must also be healthy (it monitors the critical components)
        if not self._health_monitor.is_healthy():
            return False

        return True
    
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