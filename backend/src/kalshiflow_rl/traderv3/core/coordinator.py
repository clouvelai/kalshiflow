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
    from ..clients.market_ticker_listener import MarketTickerListener
    from ..clients.fill_listener import FillListener
    from ..clients.lifecycle_client import LifecycleClient
    from ..clients.lifecycle_integration import V3LifecycleIntegration
    from ..services.whale_tracker import WhaleTracker
    from ..services.market_price_syncer import MarketPriceSyncer
    from ..services.trading_state_syncer import TradingStateSyncer
    from ..services.event_lifecycle_service import EventLifecycleService
    from ..services.tracked_markets_syncer import TrackedMarketsSyncer
    from ..services.upcoming_markets_syncer import UpcomingMarketsSyncer
    from ..services.api_discovery_syncer import ApiDiscoverySyncer
    from ..state.tracked_markets import TrackedMarketsState

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
from ..services.rlm_service import RLMService

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
                whale_tracker=whale_tracker,
                config=config
            )

            # Set trading service on websocket manager for followed whale IDs
            self._websocket_manager.set_trading_service(self._trading_service)

            # Set state container for immediate trading state on client connect
            self._websocket_manager.set_state_container(self._state_container)

            # Initialize trading flow orchestrator only for cycle-based strategies
            # Event-driven strategies (WHALE_FOLLOWER, YES_80_90, RLM_NO) use dedicated services instead
            event_driven_strategies = {TradingStrategy.WHALE_FOLLOWER, TradingStrategy.YES_80_90, TradingStrategy.RLM_NO, TradingStrategy.HOLD}
            if strategy not in event_driven_strategies:
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
                logger.info(f"Trading flow orchestrator enabled for {strategy.value} strategy")
            else:
                logger.info(f"Skipping orchestrator for event-driven {strategy.value} strategy")

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
                config=self._config,
                whale_tracker=whale_tracker,
            )
            # Connect whale execution service to WebSocket manager for decision history
            self._websocket_manager.set_whale_execution_service(self._whale_execution_service)
            # Register with health monitor for health tracking
            self._health_monitor.set_whale_execution_service(self._whale_execution_service)
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
                # Register with health monitor for health tracking
                self._health_monitor.set_yes_80_90_service(self._yes_80_90_service)
                logger.info("Yes8090Service initialized for YES at 80-90c trading strategy")

        # RLM service (initialized in _connect_lifecycle when TrackedMarketsState is available)
        self._rlm_service: Optional[RLMService] = None

        # Position listener for real-time position updates (initialized later if trading client available)
        self._position_listener: Optional['PositionListener'] = None

        # Market ticker listener for real-time price updates (initialized later if trading client available)
        self._market_ticker_listener: Optional['MarketTickerListener'] = None

        # Market price syncer for REST API price fetching (initialized later if trading client available)
        self._market_price_syncer: Optional['MarketPriceSyncer'] = None

        # Trading state syncer for periodic balance/positions/orders/settlements sync
        self._trading_state_syncer: Optional['TradingStateSyncer'] = None

        # Fill listener for real-time order fill notifications
        self._fill_listener: Optional['FillListener'] = None

        # Lifecycle mode components (initialized when market_mode == "lifecycle")
        self._lifecycle_client: Optional['LifecycleClient'] = None
        self._lifecycle_integration: Optional['V3LifecycleIntegration'] = None
        self._tracked_markets_state: Optional['TrackedMarketsState'] = None
        self._event_lifecycle_service: Optional['EventLifecycleService'] = None
        self._lifecycle_syncer: Optional['TrackedMarketsSyncer'] = None
        self._upcoming_markets_syncer: Optional['UpcomingMarketsSyncer'] = None
        self._api_discovery_syncer: Optional['ApiDiscoverySyncer'] = None

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

        # Lifecycle connection (for lifecycle mode market discovery)
        await self._connect_lifecycle()

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

            # Connect market ticker listener for real-time prices (non-blocking, optional)
            await self._connect_market_ticker_listener()

            # Start market price syncer for REST API price fetching
            await self._start_market_price_syncer()

            # Start trading state syncer for periodic Kalshi sync
            await self._start_trading_state_syncer()

            # Connect fill listener for real-time order fill notifications
            await self._connect_fill_listener()

            # Start API discovery syncer AFTER trading client is connected
            # This must come after trading client because it uses REST API calls
            await self._start_api_discovery()

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

    async def _connect_lifecycle(self) -> None:
        """
        Connect to lifecycle WebSocket for market discovery.

        Only active when market_mode == "lifecycle". Creates and wires:
        - TrackedMarketsState for market state
        - LifecycleClient for lifecycle WebSocket
        - V3LifecycleIntegration for EventBus integration
        - EventLifecycleService for event processing
        - TrackedMarketsSyncer for REST price updates
        """
        if self._config.market_mode != "lifecycle":
            logger.debug(f"Skipping lifecycle connection ({self._config.market_mode} mode)")
            return

        logger.info("Starting lifecycle mode...")

        try:
            # Import here to avoid circular imports
            from ..clients.lifecycle_client import LifecycleClient
            from ..clients.lifecycle_integration import V3LifecycleIntegration
            from ..services.event_lifecycle_service import EventLifecycleService
            from ..services.tracked_markets_syncer import TrackedMarketsSyncer
            from ..services.upcoming_markets_syncer import UpcomingMarketsSyncer
            from ..state.tracked_markets import TrackedMarketsState
            from kalshiflow.auth import KalshiAuth
            from ...data.database import rl_db

            # 1. Create TrackedMarketsState
            self._tracked_markets_state = TrackedMarketsState(
                max_markets=self._config.lifecycle_max_markets
            )

            # 1a. Load tracked markets from database (startup recovery)
            db_markets = await rl_db.get_tracked_markets(include_settled=False)
            recovered_tickers = []
            if db_markets:
                loaded = await self._tracked_markets_state.load_from_db(db_markets)
                # Track which markets were recovered for orderbook subscription
                recovered_tickers = [m['market_ticker'] for m in db_markets if m.get('status') != 'settled']
                logger.info(f"Recovered {loaded} tracked markets from database")

            # 2. Create KalshiAuth for lifecycle WebSocket
            auth = KalshiAuth.from_env()

            # 3. Create LifecycleClient
            self._lifecycle_client = LifecycleClient(
                ws_url=self._config.ws_url,
                auth=auth,
                base_reconnect_delay=5.0,
            )

            # 4. Create V3LifecycleIntegration
            self._lifecycle_integration = V3LifecycleIntegration(
                lifecycle_client=self._lifecycle_client,
                event_bus=self._event_bus,
            )

            # 5. Create EventLifecycleService
            self._event_lifecycle_service = EventLifecycleService(
                event_bus=self._event_bus,
                tracked_markets=self._tracked_markets_state,
                trading_client=self._trading_client_integration,
                db=rl_db,
                categories=self._config.lifecycle_categories,
            )

            # 6. Wire orderbook callbacks (for dynamic subscription)
            self._event_lifecycle_service.set_subscribe_callback(
                self._orderbook_integration.subscribe_market
            )
            self._event_lifecycle_service.set_unsubscribe_callback(
                self._orderbook_integration.unsubscribe_market
            )

            # 7. Set TrackedMarketsState on WebSocketManager and StateContainer
            self._websocket_manager.set_tracked_markets_state(self._tracked_markets_state)
            self._state_container.set_tracked_markets(self._tracked_markets_state)

            # 8. Start lifecycle integration
            await self._lifecycle_integration.start()

            # 9. Wait for connection
            connected = await self._lifecycle_integration.wait_for_connection(timeout=30.0)
            if not connected:
                logger.warning("Lifecycle connection failed - lifecycle mode degraded")
                await self._event_bus.emit_system_activity(
                    activity_type="connection",
                    message="Lifecycle WebSocket connection failed - running in degraded mode",
                    metadata={"degraded": True, "feature": "lifecycle", "severity": "warning"}
                )
                return

            # 10. Start EventLifecycleService
            await self._event_lifecycle_service.start()

            # NOTE: ApiDiscoverySyncer is started AFTER trading client connects
            # in _start_api_discovery() to ensure the client is ready for REST calls

            # 10.1 Subscribe to MARKET_DETERMINED for state cleanup
            await self._event_bus.subscribe_to_market_determined(self._handle_market_determined_cleanup)
            logger.info("Subscribed to MARKET_DETERMINED for state cleanup")

            # 10a. Subscribe to orderbooks for recovered markets
            if recovered_tickers:
                subscribed = 0
                for ticker in recovered_tickers:
                    try:
                        await self._orderbook_integration.subscribe_market(ticker)
                        subscribed += 1
                    except Exception as e:
                        logger.warning(f"Failed to subscribe recovered market {ticker}: {e}")
                logger.info(f"Subscribed to {subscribed}/{len(recovered_tickers)} recovered market orderbooks")

            # 10b. Initialize RLMService (if strategy is RLM_NO and trading available)
            if self._trading_service and self._config.trading_strategy == TradingStrategy.RLM_NO:
                self._rlm_service = RLMService(
                    event_bus=self._event_bus,
                    trading_service=self._trading_service,
                    state_container=self._state_container,
                    tracked_markets_state=self._tracked_markets_state,
                    yes_threshold=self._config.rlm_yes_threshold,
                    min_trades=self._config.rlm_min_trades,
                    min_price_drop=self._config.rlm_min_price_drop,
                    contracts_per_trade=self._config.rlm_contracts,
                    max_concurrent=self._config.rlm_max_concurrent,
                    allow_reentry=self._config.rlm_allow_reentry,
                    orderbook_timeout=self._config.rlm_orderbook_timeout,
                    tight_spread=self._config.rlm_tight_spread,
                    normal_spread=self._config.rlm_normal_spread,
                    max_spread=self._config.rlm_max_spread,
                )
                # Register with health monitor for health tracking
                self._health_monitor.set_rlm_service(self._rlm_service)
                # Register with websocket manager for real-time RLM state broadcasting
                self._websocket_manager.set_rlm_service(self._rlm_service)
                logger.info("RLMService initialized for RLM_NO strategy in lifecycle mode")

            # 11. Start TrackedMarketsSyncer (for REST price/volume updates)
            self._lifecycle_syncer = TrackedMarketsSyncer(
                trading_client=self._trading_client_integration,
                tracked_markets_state=self._tracked_markets_state,
                event_bus=self._event_bus,
                sync_interval=self._config.lifecycle_sync_interval,
                on_market_closed=self._orderbook_integration.unsubscribe_market,
            )
            await self._lifecycle_syncer.start()

            # 12. Start UpcomingMarketsSyncer (for upcoming markets schedule)
            # NOTE: Paper/demo env has no realistic upcoming markets (test data opens in 2030).
            # This feature is most useful with production API where real markets are scheduled.
            self._upcoming_markets_syncer = UpcomingMarketsSyncer(
                trading_client=self._trading_client_integration,
                websocket_manager=self._websocket_manager,
                event_bus=self._event_bus,
                sync_interval=60.0,  # Refresh every 60s
                hours_ahead=24.0,    # 24-hour lookahead window
            )
            await self._upcoming_markets_syncer.start()
            self._websocket_manager.set_upcoming_markets_syncer(self._upcoming_markets_syncer)
            logger.info("UpcomingMarketsSyncer started - tracking markets opening within 4h")

            logger.info(
                f"Lifecycle mode active - tracking categories: {', '.join(self._config.lifecycle_categories)}"
            )
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Lifecycle mode active - tracking {', '.join(self._config.lifecycle_categories)}",
                metadata={
                    "feature": "lifecycle",
                    "categories": self._config.lifecycle_categories,
                    "max_markets": self._config.lifecycle_max_markets,
                    "severity": "info"
                }
            )

        except Exception as e:
            logger.error(f"Failed to start lifecycle mode: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Lifecycle mode failed: {str(e)}",
                metadata={"error": str(e), "severity": "error"}
            )

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

            # Register with health monitor for health tracking
            self._health_monitor.set_position_listener(self._position_listener)

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
            if await self._state_container.update_single_position(ticker, position_data):
                # Broadcast updated state to frontend
                await self._status_reporter.emit_trading_state()

                # Update market ticker subscriptions if positions changed
                await self._update_market_ticker_subscriptions()

                logger.debug(f"Position update broadcast: {ticker}")

        except Exception as e:
            logger.error(f"Error handling position update: {e}")

    async def _connect_market_ticker_listener(self) -> None:
        """
        Connect market ticker listener for real-time price updates.

        This is a non-critical component - failure falls back to no real-time prices.
        Only starts if trading client is available and has positions.
        """
        if not self._trading_client_integration:
            logger.debug("Skipping market ticker listener (no trading client)")
            return

        try:
            # Import here to avoid circular imports
            from ..clients.market_ticker_listener import MarketTickerListener

            logger.info("Starting market ticker listener...")

            # Create market ticker listener with 500ms throttle
            self._market_ticker_listener = MarketTickerListener(
                event_bus=self._event_bus,
                ws_url=self._config.ws_url,
                throttle_ms=500,
            )

            # Subscribe to ticker update events
            await self._event_bus.subscribe_to_market_ticker(self._handle_market_ticker_update)

            # Start the listener
            await self._market_ticker_listener.start()

            # Get current position tickers and subscribe
            if self._state_container.trading_state:
                tickers = list(self._state_container.trading_state.positions.keys())
                if tickers:
                    await self._market_ticker_listener.update_subscriptions(tickers)
                    logger.info(f"Subscribed to {len(tickers)} position tickers for price updates")

            # Set on status reporter for health broadcasting
            self._status_reporter.set_market_ticker_listener(self._market_ticker_listener)

            # Register with health monitor for health tracking
            self._health_monitor.set_market_ticker_listener(self._market_ticker_listener)

            logger.info("Market ticker listener active")

            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Real-time market prices enabled",
                metadata={"channel": "ticker", "severity": "info"}
            )

        except Exception as e:
            # Don't fail startup if market ticker listener fails - prices are optional
            logger.warning(f"Market ticker listener failed (prices unavailable): {e}")
            self._market_ticker_listener = None
            self._state_container.set_component_degraded("market_ticker", True, str(e))

    async def _handle_market_ticker_update(self, event) -> None:
        """
        Handle real-time market ticker update from WebSocket.

        Updates the state container with new market prices.

        Args:
            event: MarketTickerEvent from event bus
        """
        try:
            ticker = event.market_ticker
            price_data = event.price_data

            # Update state container (separate from position data)
            if self._state_container.update_market_price(ticker, price_data):
                # Broadcast updated state to frontend
                # Note: We piggyback on trading state broadcast since market prices
                # are included in get_trading_summary()
                await self._status_reporter.emit_trading_state()

        except Exception as e:
            logger.error(f"Error handling market ticker update: {e}")

    async def _handle_market_determined_cleanup(self, event) -> None:
        """
        Handle market determined events for state cleanup.

        Cleans up state container and optionally removes from tracked markets
        to prevent unbounded memory growth.

        Args:
            event: MarketDeterminedEvent from event bus
        """
        try:
            ticker = event.market_ticker

            # Clean up state container (market prices, session tracking)
            cleaned = self._state_container.cleanup_market(ticker)

            # Remove from tracked markets state to prevent unbounded memory growth
            if self._tracked_markets_state:
                removed = await self._tracked_markets_state.remove_market(ticker)
                if removed:
                    logger.info(f"Removed determined market from tracking: {ticker}")

            if cleaned:
                logger.info(f"Coordinator cleaned up state for determined market: {ticker}")

        except Exception as e:
            logger.error(f"Error handling market determined cleanup: {e}")

    async def _update_market_ticker_subscriptions(self) -> None:
        """
        Update market ticker subscriptions based on current positions.

        Called when positions change to add/remove ticker subscriptions.
        """
        if not self._market_ticker_listener:
            return

        if not self._state_container.trading_state:
            return

        tickers = list(self._state_container.trading_state.positions.keys())
        await self._market_ticker_listener.update_subscriptions(tickers)

    async def _start_market_price_syncer(self) -> None:
        """
        Start the market price syncer for REST API price fetching.

        This provides immediate market prices on startup and periodic refresh.
        Works alongside WebSocket ticker updates.
        """
        if not self._trading_client_integration:
            logger.debug("Skipping market price syncer (no trading client)")
            return

        try:
            # Import here to avoid circular imports
            from ..services.market_price_syncer import MarketPriceSyncer

            logger.info("Starting market price syncer...")

            # Create market price syncer with 30s refresh interval
            self._market_price_syncer = MarketPriceSyncer(
                trading_client=self._trading_client_integration,
                state_container=self._state_container,
                event_bus=self._event_bus,
                sync_interval=30.0,
            )

            # Start the syncer (performs initial sync immediately)
            await self._market_price_syncer.start()

            # Set on status reporter for health broadcasting
            self._status_reporter.set_market_price_syncer(self._market_price_syncer)

            # Register with health monitor for health tracking
            self._health_monitor.set_market_price_syncer(self._market_price_syncer)

            # Set on websocket manager for initial state sends to new clients
            self._websocket_manager.set_market_price_syncer(self._market_price_syncer)

            # Bump trading state version and emit immediately
            # This ensures any connected clients get the syncer health info
            self._state_container._trading_state_version += 1
            await self._status_reporter.emit_trading_state()

            logger.info("Market price syncer active")

            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Market price syncer enabled - REST API refresh every 30s",
                metadata={"feature": "market_price_syncer", "severity": "info"}
            )

        except Exception as e:
            # Don't fail startup if market price syncer fails - it's non-critical
            logger.warning(f"Market price syncer failed: {e}")
            self._market_price_syncer = None

    async def _start_trading_state_syncer(self) -> None:
        """Start trading state syncer for periodic balance/positions/orders/settlements sync."""
        if not self._trading_client_integration:
            logger.debug("Skipping trading state syncer (no trading client)")
            return

        try:
            # Import here to avoid circular imports
            from ..services.trading_state_syncer import TradingStateSyncer

            logger.info("Starting trading state syncer...")

            # Create trading state syncer with 20s refresh interval
            self._trading_state_syncer = TradingStateSyncer(
                trading_client=self._trading_client_integration,
                state_container=self._state_container,
                event_bus=self._event_bus,
                status_reporter=self._status_reporter,
                sync_interval=20.0,
            )

            # Start the syncer (performs initial sync immediately)
            await self._trading_state_syncer.start()

            # Register with health monitor for health tracking
            self._health_monitor.set_trading_state_syncer(self._trading_state_syncer)

            logger.info("Trading state syncer active")

            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Trading state syncer enabled - Kalshi sync every 20s",
                metadata={"feature": "trading_state_syncer", "severity": "info"}
            )

        except Exception as e:
            # Don't fail startup if trading state syncer fails - it's non-critical
            logger.warning(f"Trading state syncer failed: {e}")
            self._trading_state_syncer = None

    async def _start_api_discovery(self) -> None:
        """
        Start API discovery syncer for already-open markets.

        This MUST be called AFTER trading client connects since it uses
        REST API calls to fetch open markets. Called from _establish_connections()
        after _connect_trading_client().
        """
        # Only start if lifecycle mode is active and config enables it
        if self._config.market_mode != "lifecycle":
            logger.debug("Skipping API discovery (not in lifecycle mode)")
            return

        if not self._config.api_discovery_enabled:
            logger.debug("Skipping API discovery (disabled in config)")
            return

        if not self._trading_client_integration:
            logger.debug("Skipping API discovery (no trading client)")
            return

        if not self._event_lifecycle_service:
            logger.warning("Skipping API discovery (no event lifecycle service)")
            return

        try:
            from ..services.api_discovery_syncer import ApiDiscoverySyncer

            logger.info("Starting API discovery syncer for already-open markets...")

            self._api_discovery_syncer = ApiDiscoverySyncer(
                trading_client=self._trading_client_integration,
                event_lifecycle_service=self._event_lifecycle_service,
                tracked_markets_state=self._tracked_markets_state,
                event_bus=self._event_bus,
                categories=self._config.lifecycle_categories,
                sync_interval=float(self._config.api_discovery_interval),
                batch_size=self._config.api_discovery_batch_size,
                close_min_minutes=self._config.discovery_close_min_minutes,
            )
            await self._api_discovery_syncer.start()

            logger.info(
                f"API discovery syncer active - "
                f"categories: {', '.join(self._config.lifecycle_categories)}, "
                f"interval: {self._config.api_discovery_interval}s"
            )

        except Exception as e:
            logger.error(f"Failed to start API discovery syncer: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"API discovery syncer failed: {str(e)}",
                metadata={"error": str(e), "severity": "warning"}
            )

    async def _connect_fill_listener(self) -> None:
        """
        Connect fill listener for real-time order fill notifications.

        This provides instant feedback when our orders get filled,
        rather than waiting for the next REST API sync.
        """
        if not self._trading_client_integration:
            logger.debug("Skipping fill listener (no trading client)")
            return

        try:
            # Import here to avoid circular imports
            from ..clients.fill_listener import FillListener

            logger.info("Starting fill listener for order notifications...")

            # Create fill listener
            self._fill_listener = FillListener(
                event_bus=self._event_bus,
                ws_url=self._config.ws_url,
                reconnect_delay_seconds=5.0,
            )

            # Subscribe to fill events for console UX
            await self._event_bus.subscribe_to_order_fill(self._handle_order_fill)

            # Start the listener
            await self._fill_listener.start()

            # Register with health monitor for health tracking
            self._health_monitor.set_fill_listener(self._fill_listener)

            logger.info("Fill listener active - instant order fill notifications enabled")

            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Order fill notifications enabled",
                metadata={"channel": "fill", "severity": "info"}
            )

        except Exception as e:
            # Don't fail startup if fill listener fails - REST API sync still works
            logger.warning(f"Fill listener failed (falling back to REST sync): {e}")
            self._fill_listener = None
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Real-time fill notifications unavailable: {e}",
                metadata={"fallback": "rest_sync", "severity": "warning"}
            )

    async def _handle_order_fill(self, event) -> None:
        """
        Handle real-time order fill notification.

        Emits a satisfying console message when an order fills.
        This is the key UX moment for the fill listener.

        Args:
            event: OrderFillEvent from event bus
        """
        try:
            # Extract fill details
            ticker = event.market_ticker
            action = event.action.upper()  # BUY or SELL
            side = event.side.upper()  # YES or NO
            count = event.count
            price_cents = event.price_cents
            total_cents = price_cents * count

            # Format dollar amount nicely
            total_dollars = total_cents / 100

            # Build satisfying console message
            message = f"Order filled: {action} {count} {side} {ticker} @ {price_cents}c (${total_dollars:.2f})"

            # Emit as system activity with success severity for nice UX
            await self._event_bus.emit_system_activity(
                activity_type="order_fill",
                message=message,
                metadata={
                    "trade_id": event.trade_id,
                    "order_id": event.order_id,
                    "ticker": ticker,
                    "action": event.action,
                    "side": event.side,
                    "price_cents": price_cents,
                    "count": count,
                    "total_cents": total_cents,
                    "is_taker": event.is_taker,
                    "post_position": event.post_position,
                    "severity": "success"
                }
            )

            # Log at info level for visibility
            logger.info(f"ORDER FILL: {message}")

            # Remove filled order from state and broadcast update immediately
            # This provides instant UI feedback without waiting for REST sync
            if event.order_id:
                removed = self._state_container.remove_order(event.order_id)

                # Update trading attachment for tracked markets (real-time fill update)
                await self._state_container.mark_order_filled_in_attachment(
                    ticker=ticker,
                    order_id=event.order_id,
                    fill_count=count,
                    fill_price=price_cents
                )

                if removed:
                    await self._broadcast_trading_state()

        except Exception as e:
            logger.error(f"Error handling order fill event: {e}")

    async def _sync_trading_state(self) -> None:
        """Perform initial trading state sync."""
        logger.info("🔄 Syncing with Kalshi...")

        state, changes = await self._trading_client_integration.sync_with_kalshi()

        # Store in container
        state_changed = await self._state_container.update_trading_state(state, changes)

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

        # Start RLM service if configured (requires lifecycle mode)
        if self._rlm_service:
            await self._rlm_service.start()
            logger.info("RLMService started - monitoring public trades for RLM signals")
            await self._event_bus.emit_system_activity(
                activity_type="strategy_active",
                message="RLM NO strategy active - monitoring for reverse line movement",
                metadata={
                    "strategy": "RLM_NO",
                    "config": self._rlm_service.get_stats().get("config", {}),
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

        Note: Periodic trading state sync is now handled by TradingStateSyncer
        which runs in its own asyncio task for reliability.
        """
        # Start monitoring tasks
        self._start_monitoring_tasks()

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

                        # Broadcast state after orchestrator cycle (it syncs internally)
                        if cycle_run:
                            await self._status_reporter.emit_trading_state()
                    # Note: No else branch needed - TradingStateSyncer handles periodic sync

                elif current_state == V3State.ERROR:
                    # In ERROR state, just sleep to prevent CPU spinning
                    # TradingStateSyncer continues running in its own task
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
            if self._market_ticker_listener:
                total_steps += 1
            if self._market_price_syncer:
                total_steps += 1
            if self._fill_listener:
                total_steps += 1
            if self._whale_execution_service:
                total_steps += 1
            if self._yes_80_90_service:
                total_steps += 1
            if self._rlm_service:
                total_steps += 1
            if self._whale_tracker:
                total_steps += 1
            if self._trades_integration:
                total_steps += 1
            if self._lifecycle_syncer:
                total_steps += 1
            if self._upcoming_markets_syncer:
                total_steps += 1
            if self._api_discovery_syncer:
                total_steps += 1
            if self._event_lifecycle_service:
                total_steps += 1
            if self._lifecycle_integration:
                total_steps += 1
            if self._trading_state_syncer:
                total_steps += 1

            step = 1

            # Stop YES 80-90c service first (trading strategy)
            if self._yes_80_90_service:
                logger.info(f"{step}/{total_steps} Stopping YES 80-90c Service...")
                await self._yes_80_90_service.stop()
                step += 1

            # Stop RLM service (trading strategy)
            if self._rlm_service:
                logger.info(f"{step}/{total_steps} Stopping RLM Service...")
                await self._rlm_service.stop()
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

            # Stop lifecycle components (in reverse order of startup)
            if self._upcoming_markets_syncer:
                logger.info(f"{step}/{total_steps} Stopping Upcoming Markets Syncer...")
                await self._upcoming_markets_syncer.stop()
                step += 1

            if self._lifecycle_syncer:
                logger.info(f"{step}/{total_steps} Stopping Lifecycle Syncer...")
                await self._lifecycle_syncer.stop()
                step += 1

            if self._api_discovery_syncer:
                logger.info(f"{step}/{total_steps} Stopping API Discovery Syncer...")
                await self._api_discovery_syncer.stop()
                step += 1

            if self._event_lifecycle_service:
                logger.info(f"{step}/{total_steps} Stopping Event Lifecycle Service...")
                await self._event_lifecycle_service.stop()
                step += 1

            if self._lifecycle_integration:
                logger.info(f"{step}/{total_steps} Stopping Lifecycle Integration...")
                await self._lifecycle_integration.stop()
                step += 1

            # Stop position listener (before trading client)
            if self._position_listener:
                logger.info(f"{step}/{total_steps} Stopping Position Listener...")
                await self._position_listener.stop()
                step += 1

            # Stop market ticker listener
            if self._market_ticker_listener:
                logger.info(f"{step}/{total_steps} Stopping Market Ticker Listener...")
                await self._market_ticker_listener.stop()
                step += 1

            # Stop market price syncer
            if self._market_price_syncer:
                logger.info(f"{step}/{total_steps} Stopping Market Price Syncer...")
                await self._market_price_syncer.stop()
                step += 1

            # Stop fill listener
            if self._fill_listener:
                logger.info(f"{step}/{total_steps} Stopping Fill Listener...")
                await self._fill_listener.stop()
                step += 1

            # Stop trading state syncer
            if self._trading_state_syncer:
                logger.info(f"{step}/{total_steps} Stopping Trading State Syncer...")
                await self._trading_state_syncer.stop()
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

        # Add RLM service if configured
        if self._rlm_service:
            components["rlm_service"] = self._rlm_service.get_stats()

        # Add API discovery syncer if configured
        if self._api_discovery_syncer:
            components["api_discovery_syncer"] = self._api_discovery_syncer.get_health_details()

        # Add position listener if configured
        if self._position_listener:
            components["position_listener"] = self._position_listener.get_metrics()

        # Add market ticker listener if configured
        if self._market_ticker_listener:
            components["market_ticker_listener"] = self._market_ticker_listener.get_metrics()

        # Add market price syncer if configured
        if self._market_price_syncer:
            components["market_price_syncer"] = self._market_price_syncer.get_health_details()

        # Add fill listener if configured
        if self._fill_listener:
            components["fill_listener"] = self._fill_listener.get_health_details()

        # Add trading state syncer if configured
        if self._trading_state_syncer:
            components["trading_state_syncer"] = self._trading_state_syncer.get_health_details()

        # Add tracked markets state if configured (lifecycle mode)
        if self._tracked_markets_state:
            components["tracked_markets_state"] = self._tracked_markets_state.get_stats()

        # Add event lifecycle service if configured
        if self._event_lifecycle_service:
            components["event_lifecycle_service"] = self._event_lifecycle_service.get_stats()

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