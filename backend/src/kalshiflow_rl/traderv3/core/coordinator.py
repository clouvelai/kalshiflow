"""
TRADER V3 Coordinator - Orchestration Layer.

Lightweight coordinator that wires together all V3 components.
Simple, clean orchestration without business logic.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..clients.trading_client_integration import V3TradingClientIntegration
    from ..clients.trades_integration import V3TradesIntegration
    from ..clients.position_listener import PositionListener
    from ..clients.market_ticker_listener import MarketTickerListener
    from ..clients.fill_listener import FillListener
    from ..clients.lifecycle_client import LifecycleClient
    from ..clients.lifecycle_integration import V3LifecycleIntegration
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
from ..clients.orderbook_integration import V3OrderbookIntegration
from ..config.environment import V3Config
from ..services.trading_decision_service import TradingDecisionService
from ..services.listener_bootstrap_service import ListenerBootstrapService
from ..services.order_cleanup_service import OrderCleanupService
from ..services.event_position_tracker import EventPositionTracker

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
        trades_integration: Optional['V3TradesIntegration'] = None
    ):
        self._config = config
        self._state_machine = state_machine
        self._event_bus = event_bus
        self._websocket_manager = websocket_manager
        self._orderbook_integration = orderbook_integration
        self._trading_client_integration = trading_client_integration
        self._trades_integration = trades_integration

        # T5.1: Track degraded subsystems for health visibility
        self._degraded_subsystems: Set[str] = set()

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
            trades_integration=trades_integration
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

        # Initialize listener bootstrap service for real-time WebSocket listeners
        self._listener_bootstrap = ListenerBootstrapService(
            config=config,
            event_bus=event_bus,
            state_container=self._state_container,
            health_monitor=self._health_monitor,
            status_reporter=self._status_reporter,
        )

        # Initialize trading decision service (if trading client available)
        self._trading_service = None
        if trading_client_integration:
            self._trading_service = TradingDecisionService(
                trading_client=trading_client_integration,
                state_container=self._state_container,
                event_bus=event_bus,
                config=config,
                orderbook_integration=orderbook_integration
            )

            # Set trading service on websocket manager
            self._websocket_manager.set_trading_service(self._trading_service)

            # Set state container for immediate trading state on client connect
            self._websocket_manager.set_state_container(self._state_container)

        # Event position tracker (initialized in _connect_lifecycle when TrackedMarketsState is available)
        self._event_position_tracker: Optional[EventPositionTracker] = None

        # Position listener for real-time position updates
        self._position_listener: Optional['PositionListener'] = None

        # Market ticker listener for real-time price updates
        self._market_ticker_listener: Optional['MarketTickerListener'] = None

        # Market price syncer for REST API price fetching
        self._market_price_syncer: Optional['MarketPriceSyncer'] = None

        # Trading state syncer for periodic balance/positions/orders/settlements sync
        self._trading_state_syncer: Optional['TradingStateSyncer'] = None

        # Fill listener for real-time order fill notifications
        self._fill_listener: Optional['FillListener'] = None

        # Lifecycle mode components
        self._lifecycle_client: Optional['LifecycleClient'] = None
        self._lifecycle_integration: Optional['V3LifecycleIntegration'] = None
        self._tracked_markets_state: Optional['TrackedMarketsState'] = None
        self._event_lifecycle_service: Optional['EventLifecycleService'] = None
        self._lifecycle_syncer: Optional['TrackedMarketsSyncer'] = None
        self._upcoming_markets_syncer: Optional['UpcomingMarketsSyncer'] = None
        self._api_discovery_syncer: Optional['ApiDiscoverySyncer'] = None
        self._trade_flow_service = None

        # Arbitrage components (initialized in _transition_to_ready when arb_enabled)
        self._pair_registry = None
        self._poly_client = None
        self._poly_poller = None
        self._poly_ws_client = None
        self._arb_strategy = None
        self._pair_index_service = None
        self._arb_trades_client = None

        self._started_at: Optional[float] = None
        self._running = False

        # Main event loop task
        self._event_loop_task: Optional[asyncio.Task] = None

        # Monitoring tasks
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._status_reporter_task: Optional[asyncio.Task] = None

        logger.info("V3 Coordinator initialized")

    async def start(self) -> None:
        """Start the V3 trader system."""
        if self._running:
            logger.warning("V3 Coordinator is already running")
            return

        try:
            await self._initialize_components()
            self._running = True
            await self._establish_connections()
            self._event_loop_task = asyncio.create_task(self._run_event_loop())

            logger.info("=" * 60)
            logger.info("TRADER V3 STARTED SUCCESSFULLY")
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

        logger.info("1/3 Starting Event Bus...")
        await self._event_bus.start()

        logger.info("2/3 Starting WebSocket Manager...")
        await self._websocket_manager.start()

        logger.info("3/3 Starting State Machine...")
        await self._state_machine.start()

        await self._status_reporter.emit_status_update("System initializing")

    async def _establish_connections(self) -> None:
        """Establish all external connections."""
        await self._connect_orderbook()
        await self._connect_lifecycle()

        if self._trades_integration:
            await self._connect_trades()

        if self._trading_client_integration:
            await self._connect_trading_client()
            await self._sync_trading_state()

            if self._config.cleanup_on_startup:
                order_cleanup = OrderCleanupService(
                    trading_client=self._trading_client_integration,
                    event_bus=self._event_bus,
                )
                await order_cleanup.cleanup_orphaned_orders()

            await self._connect_position_listener()
            await self._connect_market_ticker_listener()
            await self._start_market_price_syncer()
            await self._start_trading_state_syncer()
            await self._connect_fill_listener()
            await self._start_api_discovery()

        await self._transition_to_ready()

    def _market_summary_list(self) -> list:
        """Return a short summary list of market tickers for state metadata."""
        tickers = self._config.market_tickers
        if len(tickers) > 2:
            return tickers[:2] + [f"...and {len(tickers) - 2} more"]
        return list(tickers)

    async def _emit_orderbook_state(
        self, context_msg: str, markets_connected: int, snapshots_received: int,
        deltas_received: int = 0, connection_established: Optional[bool] = None,
        first_snapshot_received: Optional[bool] = None, degraded: bool = False,
    ) -> None:
        """Emit orderbook state transition and update state container."""
        metadata = {
            "ws_url": self._config.ws_url,
            "markets": self._market_summary_list(),
            "market_count": len(self._config.market_tickers),
            "environment": self._config.get_environment_name(),
            "markets_connected": markets_connected,
            "snapshots_received": snapshots_received,
            "deltas_received": deltas_received,
            "connection_established": connection_established,
            "first_snapshot_received": first_snapshot_received,
        }
        if degraded:
            metadata["degraded"] = True

        await self._state_machine.transition_to(
            V3State.ORDERBOOK_CONNECT, context=context_msg, metadata=metadata,
        )
        self._state_container.update_machine_state(
            V3State.ORDERBOOK_CONNECT, context_msg,
            {"markets_connected": markets_connected, "snapshots_received": snapshots_received,
             "ws_url": self._config.ws_url, **({"degraded": True} if degraded else {})},
        )
        await self._status_reporter.emit_status_update(context_msg)

    async def _connect_orderbook(self) -> None:
        """Connect to orderbook WebSocket and wait for data."""
        logger.info("Connecting to orderbook...")
        await self._orderbook_integration.start()

        logger.info("Waiting for orderbook connection...")
        connection_success = await self._orderbook_integration.wait_for_connection(timeout=30.0)

        if not connection_success:
            logger.warning("DEGRADED MODE: Orderbook WebSocket unavailable")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="DEGRADED MODE: Running without orderbook connection",
                metadata={"degraded": True, "reason": "WebSocket unavailable", "severity": "warning"}
            )
            await self._emit_orderbook_state(
                "Running in degraded mode without orderbook",
                markets_connected=0, snapshots_received=0, deltas_received=0,
                connection_established=False, first_snapshot_received=False, degraded=True,
            )
        else:
            logger.info("Waiting for initial orderbook snapshot...")
            data_flowing = await self._orderbook_integration.wait_for_first_snapshot(timeout=10.0)
            if not data_flowing:
                logger.warning("No orderbook data received - continuing anyway")

        metrics = self._orderbook_integration.get_metrics()
        health_details = self._orderbook_integration.get_health_details()

        await self._emit_orderbook_state(
            f"Connected to {metrics['markets_connected']} markets",
            markets_connected=metrics["markets_connected"],
            snapshots_received=metrics["snapshots_received"],
            deltas_received=metrics["deltas_received"],
            connection_established=health_details.get("connection_established"),
            first_snapshot_received=health_details.get("first_snapshot_received"),
        )

    async def _connect_trades(self) -> None:
        """Connect to trades WebSocket for public trades stream (optional)."""
        if not self._trades_integration:
            return

        logger.info("Connecting to trades WebSocket...")

        if self._tracked_markets_state:
            self._trades_integration.set_tracked_markets(self._tracked_markets_state)
        else:
            logger.warning("TrackedMarketsState not available - trades will NOT be pre-filtered.")

        await self._trades_integration.start()

        logger.info("Waiting for trades WebSocket connection...")
        connection_success = await self._trades_integration.wait_for_connection(timeout=30.0)

        if not connection_success:
            logger.warning("Trades WebSocket connection failed - continuing without trades stream.")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Trades stream unavailable (WS connection failed)",
                metadata={"degraded": True, "feature": "trades_stream", "severity": "warning"}
            )
            return

        trade_flowing = await self._trades_integration.wait_for_first_trade(timeout=10.0)
        if trade_flowing:
            metrics = self._trades_integration.get_metrics()
            logger.info(f"Trades data flowing: {metrics['trades_received']} trades received")
        else:
            logger.info("No trades received yet - normal during quiet market periods")

    async def _connect_lifecycle(self) -> None:
        """
        Initialize TrackedMarketsState and lifecycle services.

        Always creates TrackedMarketsState for market state tracking.
        When no target tickers are specified, also starts lifecycle discovery services.
        """
        has_target_tickers = bool(self._config.market_tickers)
        mode_description = "Target tickers mode" if has_target_tickers else "Lifecycle discovery mode"
        logger.info(f"Starting {mode_description}...")

        from ..state.tracked_markets import TrackedMarketsState

        # Block 1: TrackedMarketsState (FATAL if fails)
        try:
            self._tracked_markets_state = TrackedMarketsState(
                max_markets=self._config.lifecycle_max_markets
            )
            self._tracked_markets_state.set_subscription_callbacks(
                on_added=self._orderbook_integration.subscribe_market,
                on_removed=self._orderbook_integration.unsubscribe_market,
            )
            logger.info("TrackedMarketsState initialized")

            self._websocket_manager.set_tracked_markets_state(self._tracked_markets_state)
            self._state_container.set_tracked_markets(self._tracked_markets_state)
            self._status_reporter.set_tracked_markets_state(self._tracked_markets_state)

            if self._trading_client_integration:
                self._state_container.set_trading_client(self._trading_client_integration)

            await self._event_bus.subscribe_to_market_determined(self._handle_market_determined_cleanup)
            logger.info("Subscribed to MARKET_DETERMINED for state cleanup")

        except Exception as e:
            logger.error(f"FATAL: TrackedMarketsState initialization failed: {e}")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"FATAL: TrackedMarketsState init failed: {str(e)}",
                metadata={"error": str(e), "severity": "error"}
            )
            raise

        # Block 2: EventPositionTracker (degraded if fails)
        try:
            if self._config.event_tracking_enabled and self._trading_service:
                self._event_position_tracker = EventPositionTracker(
                    tracked_markets=self._tracked_markets_state,
                    state_container=self._state_container,
                    config=self._config,
                )
                self._trading_service.set_event_tracker(self._event_position_tracker)
                self._status_reporter.set_event_position_tracker(self._event_position_tracker)
                logger.info("EventPositionTracker initialized")
        except Exception as e:
            logger.error(f"EventPositionTracker init failed (degraded): {e}")
            self._degraded_subsystems.add("event_position_tracker")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Event tracking subsystem degraded: {str(e)}",
                metadata={"error": str(e), "severity": "error", "degraded": True}
            )

        # Block 3: Lifecycle services (when no target tickers)
        if not has_target_tickers:
            await self._start_lifecycle_websocket()
        else:
            logger.info("Target tickers specified - skipping lifecycle discovery services")

    async def _start_lifecycle_websocket(self) -> None:
        """Start lifecycle WebSocket and discovery services."""
        try:
            from ..clients.lifecycle_client import LifecycleClient
            from ..clients.lifecycle_integration import V3LifecycleIntegration
            from ..services.event_lifecycle_service import EventLifecycleService
            from ..services.tracked_markets_syncer import TrackedMarketsSyncer
            from ..services.upcoming_markets_syncer import UpcomingMarketsSyncer
            from kalshiflow.auth import KalshiAuth
            from ...data.database import rl_db

            auth = KalshiAuth.from_env()

            self._lifecycle_client = LifecycleClient(
                ws_url=self._config.ws_url,
                auth=auth,
                base_reconnect_delay=5.0,
            )

            self._lifecycle_integration = V3LifecycleIntegration(
                lifecycle_client=self._lifecycle_client,
                event_bus=self._event_bus,
            )

            self._event_lifecycle_service = EventLifecycleService(
                event_bus=self._event_bus,
                tracked_markets=self._tracked_markets_state,
                trading_client=self._trading_client_integration,
                db=rl_db,
                categories=self._config.lifecycle_categories,
                sports_prefixes=self._config.sports_allowed_prefixes,
            )

            await self._lifecycle_integration.start()

            connected = await self._lifecycle_integration.wait_for_connection(timeout=30.0)
            if not connected:
                logger.warning("Lifecycle connection failed - lifecycle mode degraded")
                await self._event_bus.emit_system_activity(
                    activity_type="connection",
                    message="Lifecycle WebSocket connection failed - running in degraded mode",
                    metadata={"degraded": True, "feature": "lifecycle", "severity": "warning"}
                )
                return

            await self._event_lifecycle_service.start()

            self._lifecycle_syncer = TrackedMarketsSyncer(
                trading_client=self._trading_client_integration,
                tracked_markets_state=self._tracked_markets_state,
                event_bus=self._event_bus,
                sync_interval=self._config.lifecycle_sync_interval,
                on_market_closed=self._orderbook_integration.unsubscribe_market,
                config=self._config,
                state_container=self._state_container,
            )
            await self._lifecycle_syncer.start()

            self._upcoming_markets_syncer = UpcomingMarketsSyncer(
                trading_client=self._trading_client_integration,
                websocket_manager=self._websocket_manager,
                event_bus=self._event_bus,
                sync_interval=60.0,
                hours_ahead=24.0,
            )
            await self._upcoming_markets_syncer.start()
            self._websocket_manager.set_upcoming_markets_syncer(self._upcoming_markets_syncer)

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
            logger.error(f"Failed to start lifecycle WebSocket: {e}")
            self._degraded_subsystems.add("lifecycle_websocket")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message=f"Lifecycle WebSocket failed: {str(e)}",
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

        logger.info("TRADING_CLIENT_CONNECT: Connecting to Kalshi Trading API...")
        connected = await self._trading_client_integration.wait_for_connection(timeout=30.0)

        if not connected:
            raise RuntimeError("Failed to connect to trading API")

        logger.info("Setting up order group for portfolio limits...")
        order_group_id = await self._trading_client_integration.create_or_get_order_group(
            contracts_limit=10000
        )

        if order_group_id:
            logger.info(f"Order group ready: {order_group_id[:8]}...")
            await self._event_bus.emit_system_activity(
                activity_type="order_group",
                message=f"Order group ready: {order_group_id[:8]}... (10K contract limit)",
                metadata={"order_group_id": order_group_id, "contracts_limit": 10000}
            )
        else:
            logger.warning("Order groups not available - continuing without portfolio limits")
            await self._event_bus.emit_system_activity(
                activity_type="order_group",
                message="DEGRADED MODE: Order groups unavailable",
                metadata={"degraded": True, "severity": "warning"}
            )

    async def _connect_position_listener(self) -> None:
        """Connect to real-time position updates via WebSocket."""
        if not self._trading_client_integration:
            return
        self._position_listener = await self._listener_bootstrap.connect_position_listener(
            on_position_update=self._handle_position_update
        )

    async def _handle_position_update(self, event) -> None:
        """Handle real-time position update from WebSocket."""
        try:
            ticker = event.market_ticker
            position_data = event.position_data
            if await self._state_container.update_single_position(ticker, position_data):
                await self._status_reporter.emit_trading_state()
                await self._update_market_ticker_subscriptions()
                logger.debug(f"Position update broadcast: {ticker}")
        except Exception as e:
            logger.error(f"Error handling position update: {e}")

    async def _connect_market_ticker_listener(self) -> None:
        """Connect market ticker listener for real-time price updates."""
        if not self._trading_client_integration:
            return
        self._market_ticker_listener = await self._listener_bootstrap.connect_market_ticker_listener(
            on_ticker_update=self._handle_market_ticker_update
        )

    async def _handle_market_ticker_update(self, event) -> None:
        """Handle real-time market ticker update from WebSocket."""
        try:
            ticker = event.market_ticker
            price_data = event.price_data
            if self._state_container.update_market_price(ticker, price_data):
                await self._status_reporter.emit_trading_state()
        except Exception as e:
            logger.error(f"Error handling market ticker update: {e}")

    async def _handle_market_determined_cleanup(self, event) -> None:
        """Handle market determined events for state cleanup."""
        try:
            ticker = event.market_ticker
            cleaned = self._state_container.cleanup_market(ticker)
            if self._tracked_markets_state:
                removed = await self._tracked_markets_state.remove_market(ticker)
                if removed:
                    logger.info(f"Removed determined market from tracking: {ticker}")
            if cleaned:
                logger.info(f"Coordinator cleaned up state for determined market: {ticker}")
        except Exception as e:
            logger.error(f"Error handling market determined cleanup: {e}")

    async def _update_market_ticker_subscriptions(self) -> None:
        """Update market ticker subscriptions based on current positions."""
        if not self._market_ticker_listener:
            return
        if not self._state_container.trading_state:
            return
        tickers = list(self._state_container.trading_state.positions.keys())
        await self._market_ticker_listener.update_subscriptions(tickers)

    async def _start_market_price_syncer(self) -> None:
        """Start the market price syncer for REST API price fetching."""
        if not self._trading_client_integration:
            return

        try:
            from ..services.market_price_syncer import MarketPriceSyncer

            logger.info("Starting market price syncer...")
            self._market_price_syncer = MarketPriceSyncer(
                trading_client=self._trading_client_integration,
                state_container=self._state_container,
                event_bus=self._event_bus,
                sync_interval=30.0,
            )
            await self._market_price_syncer.start()

            self._status_reporter.set_market_price_syncer(self._market_price_syncer)
            self._health_monitor.set_market_price_syncer(self._market_price_syncer)
            self._websocket_manager.set_market_price_syncer(self._market_price_syncer)

            self._state_container._trading_state_version += 1
            await self._status_reporter.emit_trading_state()

            logger.info("Market price syncer active")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Market price syncer enabled - REST API refresh every 30s",
                metadata={"feature": "market_price_syncer", "severity": "info"}
            )

        except Exception as e:
            logger.error(f"Market price syncer failed: {e}")
            self._market_price_syncer = None
            self._degraded_subsystems.add("market_price_syncer")

    async def _start_trading_state_syncer(self) -> None:
        """Start trading state syncer for periodic sync."""
        if not self._trading_client_integration:
            return

        try:
            from ..services.trading_state_syncer import TradingStateSyncer

            logger.info("Starting trading state syncer...")
            self._trading_state_syncer = TradingStateSyncer(
                trading_client=self._trading_client_integration,
                state_container=self._state_container,
                event_bus=self._event_bus,
                status_reporter=self._status_reporter,
                sync_interval=20.0,
            )
            self._trading_state_syncer.set_config(self._config)
            await self._trading_state_syncer.start()
            self._health_monitor.set_trading_state_syncer(self._trading_state_syncer)

            logger.info("Trading state syncer active")
            await self._event_bus.emit_system_activity(
                activity_type="connection",
                message="Trading state syncer enabled - Kalshi sync every 20s",
                metadata={"feature": "trading_state_syncer", "severity": "info"}
            )

        except Exception as e:
            logger.error(f"Trading state syncer failed: {e}")
            self._trading_state_syncer = None
            self._degraded_subsystems.add("trading_state_syncer")

    async def _start_api_discovery(self) -> None:
        """Start API discovery syncer for already-open markets."""
        if self._config.market_tickers:
            return
        if not self._config.api_discovery_enabled:
            return
        if not self._trading_client_integration:
            return
        if not self._event_lifecycle_service:
            logger.warning("Skipping API discovery (no event lifecycle service)")
            return

        try:
            from ..services.api_discovery_syncer import ApiDiscoverySyncer

            logger.info("Starting API discovery syncer...")
            self._api_discovery_syncer = ApiDiscoverySyncer(
                trading_client=self._trading_client_integration,
                event_lifecycle_service=self._event_lifecycle_service,
                tracked_markets_state=self._tracked_markets_state,
                event_bus=self._event_bus,
                categories=self._config.lifecycle_categories,
                sports_prefixes=self._config.sports_allowed_prefixes,
                sync_interval=float(self._config.api_discovery_interval),
                batch_size=self._config.api_discovery_batch_size,
                min_hours_to_settlement=self._config.discovery_min_hours_to_settlement,
                max_days_to_settlement=self._config.discovery_max_days_to_settlement,
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
        """Connect fill listener for real-time order fill notifications."""
        if not self._trading_client_integration:
            return
        self._fill_listener = await self._listener_bootstrap.connect_fill_listener(
            on_order_fill=self._handle_order_fill
        )

    async def _handle_order_fill(self, event) -> None:
        """Handle real-time order fill notification."""
        try:
            ticker = event.market_ticker
            action = event.action.upper()
            side = event.side.upper()
            count = event.count
            price_cents = event.price_cents
            total_cents = price_cents * count
            total_dollars = total_cents / 100

            message = f"Order filled: {action} {count} {side} {ticker} @ {price_cents}c (${total_dollars:.2f})"

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

            logger.info(f"ORDER FILL: {message}")

            if event.order_id:
                removed = self._state_container.remove_order(event.order_id)
                await self._state_container.mark_order_filled_in_attachment(
                    ticker=ticker,
                    order_id=event.order_id,
                    fill_count=count,
                    fill_price=price_cents
                )
                if removed:
                    await self._status_reporter.emit_trading_state()

        except Exception as e:
            logger.error(f"Error handling order fill event: {e}")

    async def _sync_trading_state(self) -> None:
        """Perform initial trading state sync."""
        logger.info("Syncing with Kalshi...")

        state, changes = await self._trading_client_integration.sync_with_kalshi()
        state_changed = await self._state_container.update_trading_state(state, changes)
        self._state_container.initialize_session_pnl(state.balance, state.portfolio_value)

        if state_changed:
            await self._status_reporter.emit_trading_state()

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
        orderbook_metrics = self._orderbook_integration.get_metrics()
        health_details = self._orderbook_integration.get_health_details()

        ready_metadata = {
            "markets_connected": orderbook_metrics["markets_connected"],
            "snapshots_received": orderbook_metrics["snapshots_received"],
            "deltas_received": orderbook_metrics["deltas_received"],
            "connection_established": health_details["connection_established"],
            "first_snapshot_received": health_details["first_snapshot_received"],
            "environment": self._config.get_environment_name()
        }

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

        degraded = not health_details["connection_established"] and orderbook_metrics["markets_connected"] == 0

        if degraded:
            if self._trading_client_integration:
                context = "DEGRADED MODE: Trading enabled without orderbook (paper mode)"
            else:
                context = "DEGRADED MODE: No orderbook connection available"
            ready_metadata["degraded"] = True
        elif orderbook_metrics["snapshots_received"] > 0:
            if self._trading_client_integration:
                context = f"System fully operational with {orderbook_metrics['markets_connected']} markets and trading enabled"
            else:
                context = f"System fully operational with {orderbook_metrics['markets_connected']} markets (orderbook only)"
        else:
            context = f"System connected (waiting for data) - {orderbook_metrics['markets_connected']} markets"

        await self._state_machine.transition_to(
            V3State.READY, context=context, metadata=ready_metadata
        )

        self._state_container.update_machine_state(V3State.READY, context, ready_metadata)

        if self._trading_client_integration and self._state_container.trading_state:
            await self._status_reporter.emit_trading_state()

        # Start TradeFlowService for live trade feed to UX
        if self._trades_integration and self._tracked_markets_state:
            try:
                from ..services.trade_flow_service import TradeFlowService
                self._trade_flow_service = TradeFlowService(
                    event_bus=self._event_bus,
                    tracked_markets=self._tracked_markets_state,
                )
                await self._trade_flow_service.start()
                self._websocket_manager.set_trade_flow_service(self._trade_flow_service)
                logger.info("TradeFlowService started - live trades will stream to UX")
            except Exception as e:
                logger.error(f"TradeFlowService failed to start: {e}")
                self._degraded_subsystems.add("trade_flow_service")

        if self._config.arb_enabled and self._config.polymarket_enabled:
            await self._start_arb_system()

        status_msg = f"System ready with {len(self._config.market_tickers)} markets"
        if self._trading_client_integration:
            status_msg += f" (trading enabled in {self._trading_client_integration._client.mode} mode)"
        if self._pair_registry:
            status_msg += f" ({self._pair_registry.count} arb pairs)"
        await self._status_reporter.emit_status_update(status_msg)

    async def _start_arb_system(self) -> None:
        """Initialize and start the arbitrage subsystem.

        Two-layer architecture:
        1. DATA LAYER: PairIndexService owns event discovery, pairing, price feeds
        2. TRADING LAYER: ArbStrategy owns SpreadMonitor for trade execution
        """
        try:
            from ..clients.polymarket_client import PolymarketClient
            from ..clients.polymarket_ws_client import PolymarketWSClient
            from ..services.pair_registry import PairRegistry
            from ..services.pairing_service import PairingService
            from ..services.pair_index_service import PairIndexService
            from ..strategies.plugins.arb_strategy import ArbStrategy

            # 1. Create PairRegistry (empty — clean slate every startup)
            self._pair_registry = PairRegistry()

            # 2. Supabase client (for pair persistence)
            supabase = None
            try:
                import os
                from supabase import create_client
                supabase_url = os.environ.get("SUPABASE_URL", "")
                supabase_key = os.environ.get("SUPABASE_ANON_KEY", os.environ.get("SUPABASE_KEY", ""))
                if supabase_url and supabase_key:
                    supabase = create_client(supabase_url, supabase_key)
                else:
                    logger.warning("Supabase credentials not set - pairs will be in-memory only")
            except Exception as e:
                logger.warning(f"Failed to create Supabase client: {e}")

            # 3. Create Polymarket REST client
            self._poly_client = PolymarketClient()

            # 4. Create PolymarketWSClient -> start (connects, no subscriptions yet)
            self._poly_ws_client = PolymarketWSClient(
                pair_registry=self._pair_registry,
                event_bus=self._event_bus,
            )
            await self._poly_ws_client.start()
            self._health_monitor.register_component("poly_ws", self._poly_ws_client, critical=False)

            poly_connected = await self._poly_ws_client.wait_for_connection(timeout=15.0)
            if poly_connected:
                logger.info("Polymarket WebSocket connected")
            else:
                logger.warning("Polymarket WebSocket connection timeout - prices may be delayed")

            # 5. Create PairingService (deterministic matcher)
            trading_client = self._trading_client_integration._client if self._trading_client_integration else None

            # Initialize embedding model for better matching
            embedding_model = None
            try:
                from langchain_openai import OpenAIEmbeddings
                embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
                logger.info("Embedding model initialized for pairing")
            except Exception as e:
                logger.warning(f"Embedding model not available (text-only matching): {e}")

            # Initialize OpenAI client for LLM pair validation
            openai_client = None
            try:
                from openai import AsyncOpenAI
                openai_client = AsyncOpenAI()
                logger.info("OpenAI client initialized for LLM pair matching")
            except Exception as e:
                logger.warning(f"OpenAI client not available (text-only pairing): {e}")

            pairing_service = PairingService(
                poly_client=self._poly_client,
                trading_client=trading_client,
                pair_registry=self._pair_registry,
                embedding_model=embedding_model,
                supabase_client=supabase,
                spread_monitor=None,  # Will be set after ArbStrategy creates it
                event_bus=self._event_bus,
                openai_client=openai_client,
            )

            # 6. Create ArbStrategy (SpreadMonitor only) -> start
            self._arb_strategy = ArbStrategy(
                event_bus=self._event_bus,
                pair_registry=self._pair_registry,
                config=self._config,
                trading_client=trading_client,
                websocket_manager=self._websocket_manager,
                state_container=self._state_container,
                supabase_client=supabase,
                orderbook_integration=self._orderbook_integration,
                poly_client=self._poly_client,
            )
            await self._arb_strategy.start()
            self._health_monitor.register_component("arb_strategy", self._arb_strategy, critical=False)

            # Wire SpreadMonitor into PairingService for pair registration
            if self._arb_strategy._spread_monitor:
                pairing_service._spread_monitor = self._arb_strategy._spread_monitor

            # 7. Create PairIndexService -> start (drives discovery + subscriptions)
            self._pair_index_service = PairIndexService(
                pairing_service=pairing_service,
                pair_registry=self._pair_registry,
                event_bus=self._event_bus,
                websocket_manager=self._websocket_manager,
                config=self._config,
                supabase_client=supabase,
                orderbook_integration=self._orderbook_integration,
                poly_ws_client=self._poly_ws_client,
                spread_monitor=self._arb_strategy._spread_monitor,
            )
            await self._pair_index_service.start()
            self._websocket_manager.set_pair_index_service(self._pair_index_service)
            self._health_monitor.register_component("pair_index_service", self._pair_index_service, critical=False)

            # 8. Start PolymarketPoller as API price fallback
            #    WS is primary, but API poll fills gaps when WS prices are missing/stale
            try:
                from ..services.polymarket_poller import PolymarketPoller

                self._poly_poller = PolymarketPoller(
                    poly_client=self._poly_client,
                    pair_registry=self._pair_registry,
                    event_bus=self._event_bus,
                    poll_interval=self._config.arb_poll_interval_seconds,
                    supabase_client=supabase,
                )
                await self._poly_poller.start()
                self._health_monitor.register_component("poly_poller", self._poly_poller, critical=False)
                logger.info(f"Polymarket API poller started (interval={self._config.arb_poll_interval_seconds}s)")
            except Exception as e:
                logger.warning(f"Polymarket poller failed to start (WS-only mode): {e}")

            logger.info(
                f"Arbitrage system started: "
                f"{self._pair_registry.count} pairs, "
                f"{len(self._pair_registry.get_events_grouped())} events"
            )

        except Exception as e:
            logger.error(f"Failed to start arb system: {e}")
            self._degraded_subsystems.add("arb_system")

    async def _run_event_loop(self) -> None:
        """Main event loop."""
        self._start_monitoring_tasks()
        logger.info("Event loop started")

        while self._running:
            try:
                current_state = self._state_machine.current_state

                if current_state == V3State.ERROR:
                    await asyncio.sleep(1.0)
                    continue
                elif current_state == V3State.SHUTDOWN:
                    break

                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("Event loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in event loop: {e}")
                await self._state_machine.enter_error_state("Event loop error", e)

        logger.info("Event loop stopped")

    def _start_monitoring_tasks(self) -> None:
        """Start background monitoring tasks."""
        self._health_monitor_task = asyncio.create_task(self._health_monitor.start())
        self._status_reporter_task = asyncio.create_task(self._status_reporter.start())

    async def stop(self) -> None:
        """Stop the V3 trader system."""
        if not self._running:
            return

        logger.info("=" * 60)
        logger.info("STOPPING TRADER V3")
        logger.info("=" * 60)

        self._running = False

        if self._event_loop_task:
            self._event_loop_task.cancel()
            try:
                await self._event_loop_task
            except asyncio.CancelledError:
                pass

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

        await self._health_monitor.stop()
        await self._status_reporter.stop()

        shutdown_sequence = [
            (self._pair_index_service, "Pair Index Service", lambda c: c.stop()),
            (self._arb_strategy, "Arb Strategy", lambda c: c.stop()),
            (self._poly_ws_client, "Polymarket WS Client", lambda c: c.stop()),
            (self._poly_poller, "Polymarket Poller", lambda c: c.stop()),
            (self._poly_client, "Polymarket Client", lambda c: c.close()),
            (self._trade_flow_service, "Trade Flow Service", lambda c: c.stop()),
            (self._trades_integration, "Trades Integration", lambda c: c.stop()),
            (self._upcoming_markets_syncer, "Upcoming Markets Syncer", lambda c: c.stop()),
            (self._lifecycle_syncer, "Lifecycle Syncer", lambda c: c.stop()),
            (self._api_discovery_syncer, "API Discovery Syncer", lambda c: c.stop()),
            (self._event_lifecycle_service, "Event Lifecycle Service", lambda c: c.stop()),
            (self._lifecycle_integration, "Lifecycle Integration", lambda c: c.stop()),
            (self._position_listener, "Position Listener", lambda c: c.stop()),
            (self._market_ticker_listener, "Market Ticker Listener", lambda c: c.stop()),
            (self._market_price_syncer, "Market Price Syncer", lambda c: c.stop()),
            (self._fill_listener, "Fill Listener", lambda c: c.stop()),
            (self._trading_state_syncer, "Trading State Syncer", lambda c: c.stop()),
            (self._trading_client_integration, "Trading Client Integration", lambda c: c.stop()),
            (self._orderbook_integration, "Orderbook Integration", lambda c: c.stop()),
        ]

        active_components = [(c, n, s) for c, n, s in shutdown_sequence if c is not None]
        core_steps = 3
        total_steps = len(active_components) + core_steps

        for step, (component, name, stop_func) in enumerate(active_components, 1):
            try:
                logger.info(f"[{step}/{total_steps}] Stopping {name}...")
                result = stop_func(component)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"[{step}/{total_steps}] Error stopping {name}: {e}")

        step = len(active_components) + 1

        try:
            logger.info(f"[{step}/{total_steps}] Transitioning to SHUTDOWN state...")
            await self._state_machine.transition_to(
                V3State.SHUTDOWN, context="Graceful shutdown initiated"
            )
        except Exception as e:
            logger.error(f"[{step}/{total_steps}] Error transitioning to SHUTDOWN: {e}")

        step += 1
        try:
            logger.info(f"[{step}/{total_steps}] Stopping core components...")
            await self._state_machine.stop()
            await self._websocket_manager.stop()
            await self._event_bus.stop()
        except Exception as e:
            logger.error(f"[{step}/{total_steps}] Error stopping core components: {e}")

        uptime = time.time() - self._started_at if self._started_at else 0
        logger.info("=" * 60)
        logger.info(f"TRADER V3 STOPPED (uptime: {uptime:.1f}s)")
        logger.info("=" * 60)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        uptime = time.time() - self._started_at if self._started_at else 0.0

        components = {
            "state_machine": self._state_machine.get_health_details(),
            "event_bus": self._event_bus.get_health_details(),
            "websocket_manager": self._websocket_manager.get_health_details(),
            "orderbook_integration": self._orderbook_integration.get_health_details(),
            "health_monitor": self._health_monitor.get_status(),
            "status_reporter": self._status_reporter.get_status()
        }

        _OPTIONAL_STATUS_COMPONENTS = [
            ("_trading_service", "trading_service", "get_stats"),
            ("_trading_client_integration", "trading_client", "get_health_details"),
            ("_trades_integration", "trades_integration", "get_health_details"),
            ("_api_discovery_syncer", "api_discovery_syncer", "get_health_details"),
            ("_position_listener", "position_listener", "get_metrics"),
            ("_market_ticker_listener", "market_ticker_listener", "get_metrics"),
            ("_market_price_syncer", "market_price_syncer", "get_health_details"),
            ("_fill_listener", "fill_listener", "get_health_details"),
            ("_trading_state_syncer", "trading_state_syncer", "get_health_details"),
            ("_tracked_markets_state", "tracked_markets_state", "get_stats"),
            ("_event_lifecycle_service", "event_lifecycle_service", "get_stats"),
            ("_trade_flow_service", "trade_flow_service", "get_trade_processing_stats"),
            ("_pair_registry", "pair_registry", "get_status"),
            ("_poly_ws_client", "poly_ws", "get_status"),
            ("_poly_poller", "poly_poller", "get_status"),
            ("_arb_strategy", "arb_strategy", "get_status"),
            ("_pair_index_service", "pair_index_service", "get_status"),
        ]
        for attr, key, method in _OPTIONAL_STATUS_COMPONENTS:
            comp = getattr(self, attr, None)
            if comp is not None:
                components[key] = getattr(comp, method)()

        orderbook_metrics = self._orderbook_integration.get_metrics()
        ws_stats = self._websocket_manager.get_stats()
        health_details = self._orderbook_integration.get_health_details()

        tracked_markets_count = 0
        if self._tracked_markets_state:
            tracked_markets_count = self._tracked_markets_state.active_count

        subscribed_markets = self._orderbook_integration.get_subscribed_market_count()
        signal_aggregator_stats = orderbook_metrics.get("signal_aggregator")

        metrics = {
            "uptime": uptime,
            "state": self._state_machine.current_state.value,
            "tracked_markets": tracked_markets_count,
            "subscribed_markets": subscribed_markets,
            "markets_connected": orderbook_metrics["markets_connected"],
            "snapshots_received": orderbook_metrics["snapshots_received"],
            "deltas_received": orderbook_metrics["deltas_received"],
            "ws_clients": ws_stats["active_connections"],
            "ws_messages_sent": ws_stats.get("total_messages_sent", 0),
            "api_url": self._config.api_url,
            "ws_url": self._config.ws_url,
            "ping_health": health_details.get("ping_health"),
            "last_ping_age": health_details.get("last_ping_age_seconds"),
            "health": "healthy" if self.is_healthy() else "unhealthy",
            "api_connected": orderbook_metrics["markets_connected"] > 0,
            "signal_aggregator": signal_aggregator_stats,
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

        if self._trading_client_integration:
            status["trading_mode"] = self._trading_client_integration._client.mode

        return status

    def is_healthy(self) -> bool:
        """Check if system is healthy (only critical components)."""
        if not self._running:
            return False

        component_health_map = {
            "state_machine": self._state_machine.is_healthy,
            "event_bus": self._event_bus.is_healthy,
            "websocket_manager": self._websocket_manager.is_healthy,
        }

        for component_name in CRITICAL_COMPONENTS:
            if component_name in component_health_map:
                if not component_health_map[component_name]():
                    return False

        if not self._health_monitor.is_healthy():
            return False

        return True

    def get_health(self) -> Dict[str, Any]:
        """Get health status."""
        return {
            "healthy": self.is_healthy(),
            "status": "running" if self._running else "stopped",
            "state": self._state_machine.current_state.value,
            "uptime": time.time() - self._started_at if self._started_at else 0,
            "degraded_subsystems": list(self._degraded_subsystems),
        }

    def get_degraded_subsystems(self) -> Set[str]:
        """Get set of subsystems that failed to start or are in degraded state."""
        return self._degraded_subsystems.copy()

    @property
    def state_container(self) -> V3StateContainer:
        """Get state container for external access."""
        return self._state_container

    @property
    def trading_service(self) -> Optional[TradingDecisionService]:
        """Get trading service for external access."""
        return self._trading_service
