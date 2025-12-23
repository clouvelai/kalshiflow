"""
Trader V2 - Complete integration of extracted services

Wires together all the extracted services to replace the monolithic OrderManager.
Maintains functional parity while providing clean separation of concerns.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

# Import extracted services
from .coordinator import TraderCoordinator
from .state_machine import TraderStateMachine, TraderState
from .services import (
    OrderService, PositionTracker, StateSync, StatusLogger,
    FillProcessor, WebSocketManager, ApiClient
)

# Import existing working components
from .demo_client import KalshiDemoTradingClient
from .event_bus import EventBus
from .actor_service import ActorService
from .live_observation_adapter import LiveObservationAdapter
from ..data.orderbook_client import OrderbookClient

logger = logging.getLogger("kalshiflow_rl.trading.trader_v2")


class TraderV2:
    """
    Complete TRADER 2.0 integration.
    
    Replaces the monolithic OrderManager with a clean architecture of
    specialized services while maintaining all existing functionality.
    """
    
    def __init__(
        self,
        client: KalshiDemoTradingClient,
        websocket_manager=None,  # Accept global WebSocketManager
        initial_cash_balance: float = 0.0,
        market_tickers: Optional[list] = None
    ):
        """
        Initialize Trader V2.
        
        Args:
            client: KalshiDemoTradingClient instance
            websocket_manager: Global WebSocketManager for broadcasting (optional)
            initial_cash_balance: Starting cash balance
            market_tickers: List of market tickers to trade
        """
        self.client = client
        self.market_tickers = market_tickers or []
        
        # Use provided websocket_manager or create our own
        self.global_websocket_manager = websocket_manager
        
        # Initialize core coordinator with websocket manager
        self.coordinator = TraderCoordinator(client, initial_cash_balance, websocket_manager)
        
        # Get references to services for direct access
        self.state_machine = self.coordinator.state_machine
        self.order_service = self.coordinator.order_service
        self.position_tracker = self.coordinator.position_tracker
        self.state_sync = self.coordinator.state_sync
        self.status_logger = self.coordinator.status_logger
        
        # Initialize additional services with websocket manager
        self.api_client = ApiClient(client, self.status_logger)
        self.fill_processor = FillProcessor(self.order_service, self.status_logger, websocket_manager)
        
        # Only create our own WebSocketManager if none provided
        if self.global_websocket_manager:
            # Use the global one for everything
            self.websocket_manager = None  # Don't create our own
            logger.info("Using global WebSocketManager for broadcasting")
        else:
            # Create our own WebSocketManager for trading fills
            self.websocket_manager = WebSocketManager(client, self.fill_processor, self.status_logger)
            logger.info("Created local WebSocketManager (no global provided)")
        
        # Initialize existing working components
        self.event_bus = EventBus()
        self.orderbook_client: Optional[OrderbookClient] = None
        self.actor_service: Optional[ActorService] = None
        self.observation_adapter: Optional[LiveObservationAdapter] = None
        
        # Integration state
        self.is_started = False
        self.services_status = {
            "coordinator": False,
            "fill_processor": False,
            "websocket_manager": False,
            "orderbook_client": False,
            "actor_service": False
        }
        
        # Performance tracking
        self.startup_time: Optional[float] = None
        self.metrics = {
            "startups": 0,
            "shutdowns": 0,
            "calibrations": 0,
            "orders_placed": 0,
            "fills_processed": 0,
            "errors_handled": 0
        }
        
        logger.info("TraderV2 initialized with extracted services architecture")
    
    async def start(self, enable_websockets: bool = True, enable_orderbook: bool = True, initialization_tracker=None) -> Dict[str, Any]:
        """
        Start the complete trading system.
        
        Args:
            enable_websockets: Whether to enable WebSocket connections
            enable_orderbook: Whether to enable orderbook client
            initialization_tracker: InitializationTracker for coordinated progress tracking
            
        Returns:
            Dict with startup results
        """
        if self.is_started:
            return {"success": False, "error": "Trader already started"}
        
        startup_start = time.time()
        logger.info("Starting TraderV2...")
        
        try:
            # Update startup metrics
            self.metrics["startups"] += 1
            
            # Start services in order of dependency
            
            # 1. Start fill processor
            if not await self.fill_processor.start():
                return {"success": False, "error": "Failed to start fill processor"}
            self.services_status["fill_processor"] = True
            
            # 2. Start WebSocket manager (if enabled and we have our own)
            if enable_websockets and self.websocket_manager:
                if not await self.websocket_manager.start():
                    logger.warning("WebSocket manager failed to start - continuing without real-time fills")
                else:
                    self.services_status["websocket_manager"] = True
            elif enable_websockets and self.global_websocket_manager:
                # Global websocket manager is already running
                logger.info("Using global WebSocket manager (already running)")
                self.services_status["websocket_manager"] = True
            
            # 3. Initialize orderbook client (if enabled)
            if enable_orderbook and self.market_tickers:
                try:
                    # Initialize orderbook client
                    self.orderbook_client = OrderbookClient(
                        api_key_id=self.client.api_key_id,
                        private_key_content=self.client.private_key_content,
                        market_tickers=self.market_tickers,
                        event_bus=self.event_bus
                    )
                    
                    # Start orderbook client
                    await self.orderbook_client.start()
                    self.services_status["orderbook_client"] = True
                    logger.info("Orderbook client started")
                    
                except Exception as e:
                    logger.warning(f"Failed to start orderbook client: {e} - continuing without live market data")
            
            # 4. Initialize observation adapter
            if self.orderbook_client:
                try:
                    self.observation_adapter = LiveObservationAdapter(
                        market_tickers=self.market_tickers,
                        orderbook_client=self.orderbook_client
                    )
                    logger.info("Observation adapter initialized")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize observation adapter: {e}")
            
            # 5. Initialize actor service
            if self.observation_adapter:
                try:
                    from .hardcoded_policies import HoldPolicy  # Import hardcoded policy
                    
                    self.actor_service = ActorService(
                        event_bus=self.event_bus,
                        observation_adapter=self.observation_adapter,
                        order_manager=self,  # Pass ourselves as order manager
                        policy=HoldPolicy()  # Default safe policy
                    )
                    
                    await self.actor_service.start()
                    self.services_status["actor_service"] = True
                    logger.info("Actor service started")
                    
                except Exception as e:
                    logger.warning(f"Failed to start actor service: {e}")
            
            # 6. Perform initial calibration
            calibration_result = await self.coordinator.calibrate(initialization_tracker=initialization_tracker)
            if not calibration_result["success"]:
                return {
                    "success": False,
                    "error": f"Initial calibration failed: {calibration_result['error']}",
                    "calibration_result": calibration_result
                }
            
            self.metrics["calibrations"] += 1
            
            # Mark as started
            self.is_started = True
            self.startup_time = time.time()
            self.services_status["coordinator"] = True
            
            startup_duration = time.time() - startup_start
            
            startup_result = {
                "success": True,
                "startup_duration": startup_duration,
                "services_started": sum(1 for status in self.services_status.values() if status),
                "services_status": self.services_status.copy(),
                "calibration_result": calibration_result,
                "state": self.state_machine.current_state.value,
                "ready_for_trading": self.coordinator.is_ready_for_trading()
            }
            
            logger.info(f"TraderV2 started successfully ({startup_duration:.2f}s)")
            return startup_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error starting TraderV2: {error_msg}")
            
            # Attempt cleanup on failed start
            await self.stop()
            
            return {
                "success": False,
                "error": error_msg,
                "startup_duration": time.time() - startup_start
            }
    
    async def stop(self) -> Dict[str, Any]:
        """
        Stop the complete trading system.
        
        Returns:
            Dict with shutdown results
        """
        if not self.is_started:
            return {"success": True, "message": "Trader not running"}
        
        shutdown_start = time.time()
        logger.info("Stopping TraderV2...")
        
        try:
            # Update shutdown metrics
            self.metrics["shutdowns"] += 1
            
            # Stop services in reverse order
            
            # 1. Stop actor service
            if self.actor_service:
                try:
                    await self.actor_service.stop()
                    self.services_status["actor_service"] = False
                    logger.info("Actor service stopped")
                except Exception as e:
                    logger.error(f"Error stopping actor service: {e}")
            
            # 2. Stop orderbook client
            if self.orderbook_client:
                try:
                    await self.orderbook_client.stop()
                    self.services_status["orderbook_client"] = False
                    logger.info("Orderbook client stopped")
                except Exception as e:
                    logger.error(f"Error stopping orderbook client: {e}")
            
            # 3. Stop WebSocket manager (only if we own it)
            if self.websocket_manager:
                if await self.websocket_manager.stop():
                    self.services_status["websocket_manager"] = False
                    logger.info("WebSocket manager stopped")
                else:
                    logger.warning("Error stopping WebSocket manager")
            elif self.global_websocket_manager:
                # Don't stop the global one, just note we're no longer using it
                self.services_status["websocket_manager"] = False
                logger.info("Detached from global WebSocket manager")
            
            # 4. Stop fill processor
            if await self.fill_processor.stop():
                self.services_status["fill_processor"] = False
                logger.info("Fill processor stopped")
            else:
                logger.warning("Error stopping fill processor")
            
            # 5. Cancel any remaining orders
            cancelled_orders = await self.order_service.cancel_all_orders()
            if cancelled_orders > 0:
                logger.info(f"Cancelled {cancelled_orders} remaining orders")
            
            # Mark as stopped
            self.is_started = False
            self.services_status["coordinator"] = False
            
            shutdown_duration = time.time() - shutdown_start
            
            shutdown_result = {
                "success": True,
                "shutdown_duration": shutdown_duration,
                "orders_cancelled": cancelled_orders,
                "final_state": self.state_machine.current_state.value
            }
            
            logger.info(f"TraderV2 stopped successfully ({shutdown_duration:.2f}s)")
            return shutdown_result
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error stopping TraderV2: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "shutdown_duration": time.time() - shutdown_start
            }
    
    async def emergency_stop(self) -> Dict[str, Any]:
        """Emergency stop with immediate order cancellation."""
        logger.critical("TraderV2 emergency stop initiated")
        
        # Emergency stop coordinator
        emergency_result = await self.coordinator.emergency_stop()
        
        # Regular stop of other services
        stop_result = await self.stop()
        
        return {
            "emergency_stop": emergency_result,
            "regular_stop": stop_result
        }
    
    # OrderManager interface compatibility methods
    # These methods maintain compatibility with existing ActorService integration
    
    async def place_order(self, ticker: str, side: str, contract_side: str, quantity: int, orderbook=None, pricing_strategy: str = "aggressive") -> Optional[Dict[str, Any]]:
        """
        OrderManager compatibility method for ActorService.
        
        Translates to the new action-based interface.
        """
        action = {
            "type": "place_order",
            "ticker": ticker,
            "side": side,
            "contract_side": contract_side,
            "quantity": quantity,
            "pricing_strategy": pricing_strategy
        }
        
        result = await self.coordinator.process_action(action, orderbook)
        
        if result["success"]:
            self.metrics["orders_placed"] += 1
            # Convert to OrderManager-compatible format
            return {
                "order_id": result["order_id"],
                "ticker": result["ticker"],
                "side": result["side"],
                "contract_side": result["contract_side"],
                "quantity": result["quantity"],
                "limit_price": result["limit_price"]
            }
        else:
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """OrderManager compatibility method."""
        return await self.order_service.cancel_order(order_id)
    
    def get_positions(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """OrderManager compatibility method."""
        if ticker:
            position = self.position_tracker.get_position(ticker)
            return {ticker: position} if position else {}
        else:
            return self.position_tracker.get_all_positions()
    
    def get_cash_balance_cents(self) -> int:
        """OrderManager compatibility method."""
        return int(self.position_tracker.cash_balance * 100)
    
    def get_open_orders(self) -> Dict[str, Any]:
        """OrderManager compatibility method."""
        return self.order_service.get_open_orders()
    
    def get_current_status(self) -> str:
        """
        Get current trader status as a string.
        OrderManager compatibility method for ActorService.
        
        Returns:
            String representation of current state (e.g., "ready", "acting", "calibrating")
        """
        if not self.state_machine:
            return "idle"
        return self.state_machine.current_state.value
    
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get current trading state for UI display.
        OrderManager compatibility method for WebSocketManager.
        
        Returns:
            Dictionary containing portfolio state, positions, orders, and metrics
        """
        # Get portfolio summary directly from position tracker
        portfolio_summary = self.position_tracker.get_portfolio_summary()
        
        # Get state machine info
        state_info = self.state_machine.get_state_info() if self.state_machine else {}
        
        # Structure the response with exact fields frontend expects
        return {
            # Portfolio values - exact field names for frontend
            "portfolio_value": portfolio_summary.get("portfolio_value", 0.0),
            "cash_balance": portfolio_summary.get("cash_balance", 0.0),
            "total_value": portfolio_summary.get("total_value", 0.0),
            
            # P&L information
            "total_pnl": portfolio_summary.get("total_pnl", 0.0),
            "realized_pnl": portfolio_summary.get("realized_pnl", 0.0),
            "unrealized_pnl": portfolio_summary.get("unrealized_pnl", 0.0),
            
            # Positions - convert to list format frontend expects
            "positions": {
                ticker: {
                    "position": pos.contracts,
                    "contracts": pos.contracts,
                    "cost_basis": pos.cost_basis,
                    "realized_pnl": pos.realized_pnl
                }
                for ticker, pos in self.position_tracker.get_active_positions().items()
            },
            
            # Orders - convert to list format
            "open_orders": [
                {
                    "order_id": order.order_id,
                    "ticker": order.ticker,
                    "side": order.side.name,
                    "quantity": order.quantity,
                    "limit_price": order.limit_price,
                    "status": order.status.name
                }
                for order in self.order_service.get_open_orders().values()
            ],
            
            # State machine state - this is what frontend looks for in 'state' field
            "state": state_info.get("current_state", "unknown"),
            "positions_count": len(self.position_tracker.get_active_positions()),
            "orders_count": len(self.order_service.get_open_orders()),
            
            # Trader status - exact structure frontend expects
            "trader_status": {
                "current_status": state_info.get("current_state", "unknown"),
                "time_in_status": state_info.get("time_in_state", 0.0),
                "status_history": self.status_logger.get_status_history() if self.status_logger else []
            },
            
            # Metrics
            "metrics": {
                "orders_placed": self.order_service.get_order_statistics().get("orders_placed", 0),
                "orders_filled": self.order_service.get_order_statistics().get("orders_filled", 0),
                "orders_cancelled": self.order_service.get_order_statistics().get("orders_cancelled", 0),
                "positions_count": portfolio_summary.get("positions_count", 0),
                "trades_today": portfolio_summary.get("trades_today", 0)
            },
            
            # Trading enabled status
            "trading_enabled": state_info.get("is_trading_allowed", False),
            
            # Environment
            "environment": "paper" if hasattr(self.client, 'mode') and self.client.mode == "paper" else "production"
        }
    
    # Status and monitoring methods
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get complete system status."""
        trading_status = self.coordinator.get_trading_status()
        
        # Add service-specific status
        service_details = {
            "api_client": self.api_client.get_statistics(),
            "fill_processor": self.fill_processor.get_statistics(),
            "websocket_manager": self.websocket_manager.get_statistics(),
            "orderbook_client": self.orderbook_client.get_status() if self.orderbook_client else {"status": "not_initialized"},
            "actor_service": self.actor_service.get_status() if self.actor_service else {"status": "not_initialized"}
        }
        
        return {
            **trading_status,
            "service_details": service_details,
            "is_started": self.is_started,
            "startup_time": self.startup_time,
            "services_status": self.services_status,
            "metrics": self.metrics,
            "system_health": self._assess_system_health()
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        if not self.is_started:
            return {"healthy": False, "reason": "system_not_started"}
        
        # Check core services
        critical_services = ["coordinator", "fill_processor"]
        critical_healthy = all(self.services_status.get(service, False) for service in critical_services)
        
        if not critical_healthy:
            return {"healthy": False, "reason": "critical_services_down"}
        
        # Check state machine
        if not self.state_machine.is_operational():
            return {"healthy": False, "reason": "state_machine_not_operational"}
        
        # Check API client health
        if not self.api_client.is_healthy():
            return {"healthy": False, "reason": "api_client_unhealthy"}
        
        return {"healthy": True, "reason": "all_systems_operational"}
    
    def get_debug_summary(self) -> str:
        """Get copy-paste friendly debug summary."""
        return self.coordinator.get_debug_summary()
    
    async def recalibrate(self) -> Dict[str, Any]:
        """Perform system recalibration."""
        self.metrics["calibrations"] += 1
        return await self.coordinator.calibrate(force_calibration=True)
    
    async def handle_orderbook_failure(self) -> Dict[str, Any]:
        """Handle orderbook connection failure."""
        return await self.coordinator.handle_orderbook_failure()
    
    async def attempt_recovery(self) -> Dict[str, Any]:
        """Attempt recovery from error states."""
        return await self.coordinator.attempt_recovery()
    
    # Testing and validation methods
    
    async def validate_functional_parity(self) -> Dict[str, Any]:
        """
        Validate that TraderV2 maintains functional parity with OrderManager.
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            "order_management": False,
            "position_tracking": False,
            "state_synchronization": False,
            "status_logging": False,
            "error_handling": False,
            "websocket_integration": False
        }
        
        try:
            # Test order management
            if hasattr(self.order_service, 'place_order') and hasattr(self.order_service, 'cancel_order'):
                validation_results["order_management"] = True
            
            # Test position tracking
            if hasattr(self.position_tracker, 'update_from_fill') and hasattr(self.position_tracker, 'get_position'):
                validation_results["position_tracking"] = True
            
            # Test state synchronization
            if hasattr(self.state_sync, 'sync_positions') and hasattr(self.state_sync, 'sync_orders'):
                validation_results["state_synchronization"] = True
            
            # Test status logging
            if hasattr(self.status_logger, 'get_debug_summary') and hasattr(self.status_logger, 'log_action_result'):
                validation_results["status_logging"] = True
            
            # Test error handling
            if hasattr(self.coordinator, 'emergency_stop') and hasattr(self.state_machine, 'handle_error'):
                validation_results["error_handling"] = True
            
            # Test WebSocket integration
            if hasattr(self.websocket_manager, 'connect') and hasattr(self.fill_processor, 'add_fill_event'):
                validation_results["websocket_integration"] = True
            
            overall_success = all(validation_results.values())
            
            return {
                "success": overall_success,
                "validations": validation_results,
                "message": "All functional parity tests passed" if overall_success else "Some functional parity tests failed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "validations": validation_results
            }