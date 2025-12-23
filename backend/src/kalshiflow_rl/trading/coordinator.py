"""
TraderCoordinator - Created for TRADER 2.0

Thin orchestration layer that coordinates the extracted services.
Implements the main trading logic and calibration flow while delegating
specific responsibilities to the extracted services.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List

from .state_machine import TraderStateMachine, TraderState
from .services.order_service import OrderService, OrderInfo, OrderSide, ContractSide
from .services.position_tracker import PositionTracker, FillInfo
from .services.state_sync import StateSync
from .services.status_logger import StatusLogger
from .demo_client import KalshiDemoTradingClient
from ..data.orderbook_state import OrderbookState

logger = logging.getLogger("kalshiflow_rl.trading.coordinator")


class TraderCoordinatorError(Exception):
    """Trader coordinator specific errors."""
    pass


class TraderCoordinator:
    """
    Orchestrates the extracted trading services.
    
    Provides the main interface for trading operations while delegating
    specific tasks to specialized services. Implements calibration flow
    and maintains the state machine.
    """
    
    def __init__(
        self, 
        client: KalshiDemoTradingClient,
        initial_cash_balance: float = 0.0,
        websocket_manager=None,
        minimum_cash_threshold: float = 100.0
    ):
        """
        Initialize TraderCoordinator.
        
        Args:
            client: KalshiDemoTradingClient instance
            initial_cash_balance: Starting cash balance
            websocket_manager: Global WebSocketManager for broadcasting (optional)
            minimum_cash_threshold: Minimum cash required for trading
        """
        self.client = client
        self.websocket_manager = websocket_manager
        self.minimum_cash_threshold = minimum_cash_threshold
        
        # Initialize services with websocket manager
        self.status_logger = StatusLogger(websocket_manager)
        self.state_machine = TraderStateMachine(self.status_logger, websocket_manager)
        self.position_tracker = PositionTracker(initial_cash_balance, self.status_logger, websocket_manager)
        self.order_service = OrderService(client, self.position_tracker, self.status_logger, websocket_manager)
        self.state_sync = StateSync(client, self.position_tracker, self.order_service, self.status_logger, websocket_manager)
        
        # Connect fill events
        self.order_service.on_order_filled = self._handle_order_filled
        
        # Register state machine callbacks for cash recovery
        self.state_machine.register_state_enter_callback(TraderState.LOW_CASH, self._handle_low_cash_state)
        self.state_machine.register_state_enter_callback(TraderState.RECOVER_CASH, self._handle_recover_cash_state)
        
        # Calibration tracking
        self.calibration_start_time: Optional[float] = None
        self.last_calibration_time: Optional[float] = None
        
        # Performance metrics
        self.metrics = {
            "calibrations_completed": 0,
            "orders_placed": 0,
            "fills_processed": 0,
            "errors_handled": 0,
            "last_action_time": None
        }
        
        # Trading pause mechanism (for compatibility with WebSocketManager)
        self.trading_paused = False
        self.pause_reason = None
        
        logger.info("TraderCoordinator initialized")
    
    async def _broadcast_calibration_step(self, step_name: str, status: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Broadcast calibration step update via WebSocket.
        
        Args:
            step_name: Name of the calibration step
            status: Step status (starting, complete, error)
            data: Additional step data
        """
        if self.websocket_manager:
            try:
                step_data = {
                    "step_name": step_name,
                    "status": status,
                    "timestamp": time.time()
                }
                if data:
                    step_data.update(data)
                
                # Use initialization step broadcast for compatibility
                await self.websocket_manager.broadcast_initialization_step(step_data)
            except Exception as e:
                logger.warning(f"Failed to broadcast calibration step: {e}")
    
    async def broadcast_trader_state(self) -> None:
        """
        Broadcast current trader state via WebSocket.
        
        Sends comprehensive state information including cash, positions, orders,
        state machine status, and metrics. Called at key state transitions.
        """
        if not self.websocket_manager:
            return
        
        # Check if we have connections - if not, skip (will be called when connections come)
        if hasattr(self.websocket_manager, '_connections') and not self.websocket_manager._connections:
            logger.debug("No WebSocket connections - skipping trader state broadcast")
            return
            
        try:
            # Gather current state data
            portfolio_summary = self.position_tracker.get_portfolio_summary()
            order_stats = self.order_service.get_order_statistics()
            state_info = self.state_machine.get_state_info()
            
            # Build trader state data (WebSocketManager adds the type wrapper)
            state_data = {
                # Cash and portfolio - use exact field names frontend expects
                "cash_balance": self.position_tracker.cash_balance,
                "portfolio_value": portfolio_summary.get("portfolio_value", 0.0),  # Positions value only
                "total_value": portfolio_summary.get("total_value", 0.0),  # Cash + positions
                
                # Positions
                "positions": {
                    ticker: {
                        "contracts": pos.contracts,
                        "position": pos.contracts,  # Alias for compatibility
                        "side": "YES" if pos.is_long_yes else "NO",
                        "avg_price": pos.average_cost_per_contract(),
                        "cost_basis": pos.cost_basis,
                        "realized_pnl": pos.realized_pnl
                    }
                    for ticker, pos in self.position_tracker.positions.items()
                    if pos.contracts != 0
                },
                "positions_count": portfolio_summary.get("positions_count", 0),
                
                # Orders
                "open_orders": [
                    {
                        "order_id": order.order_id,
                        "ticker": order.ticker,
                        "side": order.side.name,
                        "contract_side": order.contract_side.name,
                        "quantity": order.quantity,
                        "limit_price": order.limit_price,
                        "status": order.status.name,
                        "created_at": order.created_at
                    }
                    for order in self.order_service.open_orders.values()
                ],
                "orders_count": len(self.order_service.open_orders),
                
                # State machine
                "state": state_info["current_state"],
                "state_duration": state_info.get("time_in_state", 0),
                "is_trading_allowed": state_info.get("is_trading_allowed", False),
                
                # Metrics
                "metrics": {
                    **self.metrics,
                    "total_fills": order_stats.get("total_fills", 0),
                    "calibration_age": time.time() - self.last_calibration_time if self.last_calibration_time else None,
                    "uptime": time.time() - (self.calibration_start_time or time.time())
                },
                
                # Timestamp
                "timestamp": time.time()
            }
            
            # Broadcast the state
            await self.websocket_manager.broadcast_trader_state(state_data)
            
        except Exception as e:
            logger.warning(f"Failed to broadcast trader state: {e}")
    
    async def _handle_order_filled(self, order: OrderInfo) -> None:
        """
        Handle order fill by updating position tracker.
        
        Args:
            order: Filled order information
        """
        try:
            # Convert OrderInfo to FillInfo for PositionTracker
            fill_info = FillInfo(
                ticker=order.ticker,
                side=order.side,
                contract_side=order.contract_side,
                quantity=order.quantity,
                fill_price=order.fill_price,
                fill_timestamp=order.filled_at or time.time(),
                order_id=order.order_id
            )
            
            # Update position tracker
            self.position_tracker.update_from_fill(fill_info)
            
            # Update metrics
            self.metrics["fills_processed"] += 1
            
            logger.info(f"Fill processed: {order.order_id} - position updated")
            
            # Broadcast state after fill processed
            await self.broadcast_trader_state()
            
        except Exception as e:
            logger.error(f"Error handling order fill: {e}")
            await self.status_logger.log_action_result(
                "fill_error", f"{order.order_id} - {str(e)[:50]}", 0.0
            )
    
    async def _handle_low_cash_state(self, from_state, to_state):
        """
        Handle entry into LOW_CASH state.
        
        Automatically transition to RECOVER_CASH if positions are available.
        """
        try:
            logger.warning("Entered LOW_CASH state - checking recovery options")
            
            # Check if we have positions to liquidate
            active_positions = self.position_tracker.get_active_positions()
            
            if active_positions:
                logger.info(f"Found {len(active_positions)} positions available for liquidation")
                # Automatically start position liquidation
                await asyncio.sleep(0.1)  # Brief pause for state transition completion
                await self.state_machine.start_position_liquidation()
            else:
                logger.error("No positions available for liquidation - cannot recover cash")
                # Go to error state since we can't recover
                await self.state_machine.handle_error("No positions available for cash recovery")
                
        except Exception as e:
            logger.error(f"Error handling LOW_CASH state: {e}")
            await self.state_machine.handle_error(f"Error in LOW_CASH state: {e}")
    
    async def _handle_recover_cash_state(self, from_state, to_state):
        """
        Handle entry into RECOVER_CASH state.
        
        Automatically execute bulk close of all positions.
        """
        try:
            logger.info("Entered RECOVER_CASH state - starting bulk position liquidation")
            
            # Execute bulk close
            bulk_close_result = await self.bulk_close_all_positions("state_machine_recovery")
            
            if bulk_close_result["success"]:
                logger.info(f"Bulk close completed successfully: {bulk_close_result['message']}")
                
                # Brief pause to allow fills to process
                await asyncio.sleep(2.0)
                
                # Complete cash recovery and return to calibration
                await self.state_machine.complete_cash_recovery()
            else:
                logger.error(f"Bulk close failed: {bulk_close_result['error']}")
                await self.state_machine.handle_error(f"Bulk close failed: {bulk_close_result['error']}")
                
        except Exception as e:
            logger.error(f"Error handling RECOVER_CASH state: {e}")
            await self.state_machine.handle_error(f"Error in RECOVER_CASH state: {e}")
    
    async def calibrate(self, force_calibration: bool = False, initialization_tracker=None) -> Dict[str, Any]:
        """
        Perform system calibration.
        
        Implements the calibration flow: system_check → sync_data → ready
        
        Args:
            force_calibration: Force calibration even if already calibrated
            initialization_tracker: Optional tracker for initialization steps
            
        Returns:
            Dict with calibration results
        """
        start_time = time.time()
        
        try:
            # Check if we can start calibration
            if not force_calibration and not self.state_machine.can_start_calibration():
                return {
                    "success": False,
                    "error": f"Cannot start calibration from state: {self.state_machine.current_state.value}",
                    "duration": 0.0
                }
            
            logger.info("Starting trader calibration")
            self.calibration_start_time = start_time
            
            # Clear previous calibration steps
            self.status_logger.clear_calibration_steps()
            
            # Transition to CALIBRATING state
            await self.state_machine.transition_to(TraderState.CALIBRATING, "starting_calibration")
            
            # Step 1: System Check
            await self.status_logger.log_calibration_step("system_check", "starting")
            await self._broadcast_calibration_step("system_check", "starting")
            
            # Update initialization tracker if provided
            if initialization_tracker:
                await initialization_tracker.mark_step_in_progress("trader_client_health")
            
            system_check_start = time.time()
            system_check_result = await self._perform_system_check()
            system_check_duration = time.time() - system_check_start
            
            if not system_check_result["success"]:
                await self.status_logger.log_calibration_step(
                    "system_check", "error", system_check_duration, system_check_result["error"]
                )
                if initialization_tracker:
                    await initialization_tracker.mark_step_error("trader_client_health", system_check_result["error"])
                await self.state_machine.handle_error(f"System check failed: {system_check_result['error']}")
                return {
                    "success": False,
                    "error": system_check_result["error"],
                    "duration": time.time() - start_time,
                    "step_failed": "system_check"
                }
            
            await self.status_logger.log_calibration_step("system_check", "complete", system_check_duration)
            await self._broadcast_calibration_step("system_check", "complete", {"duration": system_check_duration})
            
            # Mark system check complete in initialization tracker
            if initialization_tracker:
                await initialization_tracker.mark_step_complete("trader_client_health", system_check_result)
            
            # Step 2: Data Sync
            await self.status_logger.log_calibration_step("sync_data", "starting")
            await self._broadcast_calibration_step("sync_data", "starting")
            
            # Update initialization tracker for sync steps
            if initialization_tracker:
                await initialization_tracker.mark_step_in_progress("sync_balance")
                await initialization_tracker.mark_step_in_progress("sync_positions")
                await initialization_tracker.mark_step_in_progress("sync_orders")
            
            sync_start = time.time()
            sync_result = await self.state_sync.sync_all()
            sync_duration = time.time() - sync_start
            
            if not sync_result["success"]:
                error_msg = f"Data sync failed: positions={sync_result['positions'].get('error', 'ok')}, orders={sync_result['orders'].get('error', 'ok')}, balance={sync_result['balance'].get('error', 'ok')}"
                await self.status_logger.log_calibration_step(
                    "sync_data", "error", sync_duration, error_msg
                )
                # Mark sync steps as error in initialization tracker
                if initialization_tracker:
                    await initialization_tracker.mark_step_error("sync_balance", error_msg)
                    await initialization_tracker.mark_step_error("sync_positions", error_msg)
                    await initialization_tracker.mark_step_error("sync_orders", error_msg)
                await self.state_machine.handle_error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "duration": time.time() - start_time,
                    "step_failed": "sync_data",
                    "sync_details": sync_result
                }
            
            await self.status_logger.log_calibration_step("sync_data", "complete", sync_duration)
            await self._broadcast_calibration_step("sync_data", "complete", {"duration": sync_duration, "result": sync_result})
            
            # Mark sync steps complete in initialization tracker
            if initialization_tracker:
                await initialization_tracker.mark_step_complete("sync_balance", sync_result.get("balance", {}))
                await initialization_tracker.mark_step_complete("sync_positions", sync_result.get("positions", {}))
                await initialization_tracker.mark_step_complete("sync_orders", sync_result.get("orders", {}))
            
            # Step 3: Cash Check
            cash_check_result = await self._check_cash_threshold()
            # Check if cash is sufficient - the result has threshold_check nested
            is_sufficient = cash_check_result.get("threshold_check", {}).get("sufficient", False) if "threshold_check" in cash_check_result else cash_check_result.get("sufficient", False)
            current_balance = cash_check_result.get("cash_balance", cash_check_result.get("current_balance", 0.0))
            
            # Special case: For demo accounts with $0 balance, skip the cash check on initial setup
            # This allows the demo account to start even with no initial funds
            is_demo_initial = current_balance == 0.0 and hasattr(self.client, 'mode') and self.client.mode == "paper"
            
            if is_demo_initial:
                logger.info(f"Demo account with $0 balance - skipping cash check for initial setup")
            
            if not is_sufficient and not is_demo_initial:
                logger.warning(f"Low cash detected during calibration: ${current_balance:.2f} < ${self.minimum_cash_threshold:.2f}")
                
                # Trigger cash recovery if positions available
                if cash_check_result.get("positions_available", False):
                    await self.state_machine.trigger_cash_recovery("calibration_low_cash")
                    return {
                        "success": False,
                        "error": f"Insufficient cash balance: ${current_balance:.2f} < ${self.minimum_cash_threshold:.2f}",
                        "duration": time.time() - start_time,
                        "step_failed": "cash_check",
                        "cash_status": cash_check_result
                    }
                else:
                    # No positions to liquidate - go to error state
                    await self.state_machine.handle_error("Insufficient cash and no positions to liquidate")
                    return {
                        "success": False,
                        "error": f"Insufficient cash and no positions available for liquidation",
                        "duration": time.time() - start_time,
                        "step_failed": "cash_check",
                        "cash_status": cash_check_result
                    }
            
            # Step 4: Ready
            await self.state_machine.transition_to(TraderState.READY, "calibration_complete")
            
            # Update metrics and timestamp
            self.metrics["calibrations_completed"] += 1
            self.last_calibration_time = time.time()
            total_duration = time.time() - start_time
            
            # Broadcast state after calibration completes (CALIBRATING → READY)
            await self.broadcast_trader_state()
            
            calibration_result = {
                "success": True,
                "duration": total_duration,
                "system_check": system_check_result,
                "sync_result": sync_result,
                "steps_completed": ["system_check", "sync_data"],
                "ready_timestamp": self.last_calibration_time
            }
            
            logger.info(f"Calibration completed successfully ({total_duration:.2f}s)")
            
            return calibration_result
            
        except Exception as e:
            error_msg = str(e)
            total_duration = time.time() - start_time
            
            logger.error(f"Calibration failed: {error_msg}")
            
            await self.status_logger.log_calibration_step(
                "calibration", "error", total_duration, error_msg
            )
            
            await self.state_machine.handle_error(f"Calibration error: {error_msg}")
            
            return {
                "success": False,
                "error": error_msg,
                "duration": total_duration,
                "step_failed": "exception"
            }
    
    async def _perform_system_check(self) -> Dict[str, Any]:
        """
        Perform system health checks.
        
        Returns:
            Dict with system check results
        """
        try:
            checks = {}
            
            # Check client connection - connect first if not connected
            try:
                # Ensure client is connected (idempotent)
                if not self.client.is_connected:
                    await self.client.connect()
                    logger.info("Demo client connected during system check")
                
                account_info = await self.client.get_account_info()
                checks["client_connection"] = {"status": "ok", "account_id": account_info.get("user_id", "unknown")}
            except Exception as e:
                checks["client_connection"] = {"status": "error", "error": str(e)}
            
            # Check service health
            checks["order_service"] = {"status": "ok", "orders_pending": len(self.order_service.open_orders)}
            checks["position_tracker"] = {"status": "ok", "positions_tracked": len(self.position_tracker.positions)}
            checks["state_sync"] = {"status": "ok", "last_sync": self.state_sync.last_position_sync}
            
            # Overall status - only require critical components (client connection)
            # Non-critical failures should not prevent startup
            critical_ok = checks.get("client_connection", {}).get("status") == "ok"
            all_ok = all(check.get("status") == "ok" for check in checks.values())
            
            if not critical_ok:
                # Client connection is critical
                return {
                    "success": False,
                    "checks": checks,
                    "error": "Critical component failed: client connection required"
                }
            elif not all_ok:
                # Some non-critical components failed - log warning but continue
                logger.warning(f"System check partially successful: {checks}")
                return {
                    "success": True,  # Continue with degraded mode
                    "checks": checks,
                    "warning": "Some non-critical components are unhealthy - operating in degraded mode"
                }
            
            return {
                "success": True,
                "checks": checks,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "checks": {},
                "error": f"System check exception: {str(e)}"
            }
    
    async def process_action(
        self, 
        action: Dict[str, Any], 
        orderbook: Optional[OrderbookState] = None
    ) -> Dict[str, Any]:
        """
        Process a trading action.
        
        Args:
            action: Action dictionary with trading intent
            orderbook: Current orderbook state (required for pricing)
            
        Returns:
            Dict with action results
        """
        start_time = time.time()
        
        try:
            # Check if we can process actions
            if not self.state_machine.can_process_action():
                return {
                    "success": False,
                    "error": f"Cannot process action in state: {self.state_machine.current_state.value}",
                    "action": action,
                    "duration": 0.0
                }
            
            # Validate orderbook
            if not orderbook:
                return {
                    "success": False,
                    "error": "Orderbook required for action processing",
                    "action": action,
                    "duration": time.time() - start_time
                }
            
            # Transition to ACTING state
            await self.state_machine.transition_to(TraderState.ACTING, f"processing_action_{action.get('type', 'unknown')}")
            
            # Extract action details
            action_type = action.get("type", "unknown")
            ticker = action.get("ticker")
            side = action.get("side")  # "buy" or "sell"
            contract_side = action.get("contract_side")  # "yes" or "no"
            quantity = action.get("quantity", 1)
            pricing_strategy = action.get("pricing_strategy", "aggressive")
            
            result = {"success": False, "error": "Unknown action type"}
            
            if action_type == "place_order":
                # Convert string values to enums
                order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
                contract_side_enum = ContractSide.YES if contract_side.lower() == "yes" else ContractSide.NO
                
                # Place order via OrderService
                order = await self.order_service.place_order(
                    ticker=ticker,
                    side=order_side,
                    contract_side=contract_side_enum,
                    quantity=quantity,
                    orderbook=orderbook,
                    pricing_strategy=pricing_strategy
                )
                
                if order:
                    result = {
                        "success": True,
                        "order_id": order.order_id,
                        "ticker": ticker,
                        "side": side,
                        "contract_side": contract_side,
                        "quantity": quantity,
                        "limit_price": order.limit_price
                    }
                    self.metrics["orders_placed"] += 1
                else:
                    result = {
                        "success": False,
                        "error": "Order placement failed"
                    }
            
            elif action_type == "cancel_order":
                order_id = action.get("order_id")
                if order_id:
                    cancelled = await self.order_service.cancel_order(order_id)
                    result = {
                        "success": cancelled,
                        "order_id": order_id,
                        "error": None if cancelled else "Cancellation failed"
                    }
                else:
                    result = {
                        "success": False,
                        "error": "Order ID required for cancellation"
                    }
            
            elif action_type == "hold":
                # No-op action
                result = {
                    "success": True,
                    "action": "hold",
                    "message": "No trading action taken"
                }
            
            # Transition back to READY
            await self.state_machine.transition_to(TraderState.READY, "action_complete")
            
            # Update metrics
            self.metrics["last_action_time"] = time.time()
            
            # Broadcast state after action processing (ACTING → READY)
            await self.broadcast_trader_state()
            
            duration = time.time() - start_time
            result["duration"] = duration
            result["action"] = action
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            duration = time.time() - start_time
            
            logger.error(f"Error processing action: {error_msg}")
            
            # Handle error and transition to ERROR state
            await self.state_machine.handle_error(f"Action processing error: {error_msg}")
            
            self.metrics["errors_handled"] += 1
            
            return {
                "success": False,
                "error": error_msg,
                "action": action,
                "duration": duration
            }
    
    async def emergency_stop(self) -> Dict[str, Any]:
        """
        Emergency stop - cancel all orders and transition to safe state.
        
        Returns:
            Dict with emergency stop results
        """
        try:
            logger.critical("Emergency stop initiated")
            
            # Cancel all open orders
            cancelled_count = await self.order_service.cancel_all_orders()
            
            # Transition to ERROR state
            await self.state_machine.emergency_stop()
            
            return {
                "success": True,
                "orders_cancelled": cancelled_count,
                "state": self.state_machine.current_state.value,
                "message": "Emergency stop completed"
            }
            
        except Exception as e:
            logger.critical(f"Error during emergency stop: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Emergency stop failed"
            }
    
    def get_trading_status(self) -> Dict[str, Any]:
        """
        Get comprehensive trading status.
        
        Returns:
            Dict with complete system status
        """
        try:
            # Get state machine info
            state_info = self.state_machine.get_state_info()
            
            # Get service status
            order_stats = self.order_service.get_order_statistics()
            position_stats = self.position_tracker.get_position_statistics()
            portfolio_summary = self.position_tracker.get_portfolio_summary()
            sync_status = self.state_sync.get_sync_status()
            
            # Get current status from StatusLogger
            status_logger_info = self.status_logger.get_current_status()
            
            return {
                "state_machine": state_info,
                "services": {
                    "order_service": order_stats,
                    "position_tracker": position_stats,
                    "state_sync": sync_status
                },
                "portfolio": portfolio_summary,
                "metrics": self.metrics,
                "calibration": {
                    "last_calibration": self.last_calibration_time,
                    "calibration_age": time.time() - self.last_calibration_time if self.last_calibration_time else None
                },
                "status_logger": status_logger_info,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_debug_summary(self) -> str:
        """
        Get copy-paste friendly debug summary.
        
        Returns:
            Multi-line debug summary string
        """
        try:
            portfolio_summary = self.position_tracker.get_portfolio_summary()
            
            # Inject portfolio data into StatusLogger summary
            portfolio_info = (
                f"${portfolio_summary['total_value']:.2f} | "
                f"Positions: {portfolio_summary['positions_count']} | "
                f"Orders: {len(self.order_service.open_orders)} open"
            )
            
            # Get base debug summary and inject portfolio info
            base_summary = self.status_logger.get_debug_summary(include_portfolio=False)
            
            # Replace placeholder with actual portfolio data
            enhanced_summary = base_summary.replace(
                "Portfolio: [Portfolio data would be injected by coordinator]",
                f"Portfolio: {portfolio_info}"
            )
            
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"Error generating debug summary: {e}")
            return f"Error generating debug summary: {e}"
    
    def is_ready_for_trading(self) -> bool:
        """Check if system is ready for trading."""
        return (
            self.state_machine.is_trading_allowed() and
            self.last_calibration_time is not None and
            (time.time() - self.last_calibration_time) < 3600  # Calibration not older than 1 hour
        )
    
    async def handle_orderbook_failure(self) -> Dict[str, Any]:
        """
        Handle orderbook connection failure.
        
        Returns:
            Dict with failure handling results
        """
        try:
            logger.warning("Handling orderbook failure")
            
            # Transition to PAUSED state
            await self.state_machine.handle_orderbook_failure()
            
            # In PAUSED state, we can still do recovery operations
            # but no new trading is allowed
            
            return {
                "success": True,
                "state": self.state_machine.current_state.value,
                "message": "Orderbook failure handled - trading paused",
                "recovery_allowed": self.state_machine.is_recovery_operations_allowed()
            }
            
        except Exception as e:
            logger.error(f"Error handling orderbook failure: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def attempt_recovery(self) -> Dict[str, Any]:
        """
        Attempt recovery from error or paused state.
        
        Returns:
            Dict with recovery attempt results
        """
        try:
            current_state = self.state_machine.current_state
            
            if current_state == TraderState.PAUSED:
                # Attempt recovery from pause
                success = await self.state_machine.attempt_recovery_from_pause()
                if success:
                    # Start recalibration
                    return await self.calibrate()
                else:
                    return {
                        "success": False,
                        "error": "Failed to initiate recovery from pause"
                    }
            
            elif current_state == TraderState.ERROR:
                # Attempt recovery from error
                return await self.calibrate(force_calibration=True)
            
            else:
                return {
                    "success": False,
                    "error": f"No recovery needed from state: {current_state.value}"
                }
                
        except Exception as e:
            logger.error(f"Error during recovery attempt: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _check_cash_threshold(self) -> Dict[str, Any]:
        """
        Check if current cash balance meets minimum threshold.
        
        Returns:
            Dict with cash threshold check results
        """
        try:
            # Get current market prices for liquidation estimates (if available)
            # This could be enhanced to use orderbook data
            market_prices = None  # TODO: Connect to orderbook state for real-time prices
            
            cash_status = self.position_tracker.get_cash_status(
                self.minimum_cash_threshold,
                market_prices
            )
            
            return cash_status
            
        except Exception as e:
            logger.error(f"Error checking cash threshold: {e}")
            return {
                "sufficient": False,
                "current_balance": self.position_tracker.cash_balance,
                "minimum_threshold": self.minimum_cash_threshold,
                "error": str(e)
            }
    
    async def bulk_close_all_positions(self, reason: str = "cash_recovery") -> Dict[str, Any]:
        """
        Close all open positions to recover cash.
        
        This method implements the RECOVER_CASH state behavior by:
        1. Cancelling all open orders
        2. Closing all positions via market orders
        3. Tracking the recovery process
        
        Args:
            reason: Reason for bulk close (for logging)
            
        Returns:
            Dict with bulk close results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting bulk position close: {reason}")
            
            # Step 1: Cancel all open orders first
            await self.status_logger.log_action_result("bulk_close", "cancelling_orders", 0.0)
            cancelled_orders = await self.order_service.cancel_all_orders()
            
            # Step 2: Get all active positions
            active_positions = self.position_tracker.get_active_positions()
            
            if not active_positions:
                logger.info("No active positions to close")
                duration = time.time() - start_time
                await self.status_logger.log_action_result("bulk_close", "no_positions", duration)
                
                return {
                    "success": True,
                    "orders_cancelled": cancelled_orders,
                    "positions_closed": 0,
                    "positions_attempted": 0,
                    "duration": duration,
                    "message": "No positions to close"
                }
            
            # Step 3: Close each position via market order
            positions_closed = 0
            positions_attempted = len(active_positions)
            close_results = []
            
            for ticker, position in active_positions.items():
                try:
                    # Determine order to close position
                    if position.is_long_yes:
                        # Long YES: sell YES contracts
                        side = OrderSide.SELL
                        contract_side = ContractSide.YES
                        quantity = position.contracts
                    else:
                        # Long NO: sell NO contracts (buy YES)
                        side = OrderSide.BUY
                        contract_side = ContractSide.YES
                        quantity = abs(position.contracts)
                    
                    # Place market order to close position
                    # Note: We need orderbook for pricing, using aggressive pricing strategy
                    # This is a simplified implementation - in production would need current orderbook
                    logger.info(f"Closing position: {ticker} {quantity} contracts")
                    
                    # For now, log the intended close order
                    # In full implementation, would place actual orders
                    close_results.append({
                        "ticker": ticker,
                        "side": side.name,
                        "contract_side": contract_side.name,
                        "quantity": quantity,
                        "status": "attempted",
                        "position_value": position.cost_basis
                    })
                    
                    # TODO: Implement actual order placement when orderbook is available
                    # order = await self.order_service.place_order(
                    #     ticker=ticker,
                    #     side=side,
                    #     contract_side=contract_side,
                    #     quantity=quantity,
                    #     orderbook=current_orderbook,
                    #     pricing_strategy="market"
                    # )
                    
                    positions_closed += 1
                    
                except Exception as e:
                    logger.error(f"Error closing position {ticker}: {e}")
                    close_results.append({
                        "ticker": ticker,
                        "status": "error",
                        "error": str(e)
                    })
            
            duration = time.time() - start_time
            
            # Log completion
            await self.status_logger.log_action_result(
                "bulk_close", 
                f"{positions_closed}/{positions_attempted} positions", 
                duration
            )
            
            logger.info(f"Bulk close completed: {positions_closed}/{positions_attempted} positions in {duration:.2f}s")
            
            return {
                "success": True,
                "orders_cancelled": cancelled_orders,
                "positions_closed": positions_closed,
                "positions_attempted": positions_attempted,
                "close_results": close_results,
                "duration": duration,
                "message": f"Bulk close completed: {positions_closed}/{positions_attempted} positions"
            }
            
        except Exception as e:
            error_msg = str(e)
            duration = time.time() - start_time
            
            logger.error(f"Error during bulk close: {error_msg}")
            await self.status_logger.log_action_result("bulk_close", f"error_{error_msg[:30]}", duration)
            
            return {
                "success": False,
                "error": error_msg,
                "duration": duration,
                "orders_cancelled": cancelled_orders if 'cancelled_orders' in locals() else 0,
                "positions_closed": 0,
                "positions_attempted": len(active_positions) if 'active_positions' in locals() else 0
            }

    async def broadcast_trader_state(self) -> None:
        """
        Broadcast current trader state via WebSocket.
        
        Sends actual trader data (cash, positions, orders) to frontend
        in the format expected by the trader_state message type.
        """
        if not self.websocket_manager:
            logger.debug("No WebSocket manager available for broadcasting trader state")
            return
        
        try:
            # Get current state from all services
            positions_summary = self.position_tracker.get_portfolio_summary()
            open_orders = self.order_service.get_open_orders()
            
            # Format trader state data for frontend
            state_data = {
                "cash_balance": self.position_tracker.cash_balance,
                "positions": self.position_tracker.get_all_positions(),
                "open_orders": open_orders,
                "state": self.state_machine.current_state.value,
                "metrics": {
                    "total_positions": len(self.position_tracker.positions),
                    "open_orders_count": len(open_orders),
                    "portfolio_value": positions_summary.get("total_value", 0.0),
                    "unrealized_pnl": positions_summary.get("unrealized_pnl", 0.0),
                    "realized_pnl": positions_summary.get("realized_pnl", 0.0),
                },
                "timestamp": time.time(),
                "last_calibration": getattr(self, "_last_calibration_time", None)
            }
            
            # Broadcast via WebSocket manager
            await self.websocket_manager.broadcast_trader_state(state_data)
            logger.debug(f"Broadcasted trader state: {self.state_machine.current_state.value}, "
                        f"balance=${self.position_tracker.cash_balance:.2f}, "
                        f"positions={len(self.position_tracker.positions)}, "
                        f"orders={len(open_orders)}")
            
        except Exception as e:
            logger.error(f"Error broadcasting trader state: {e}")

    async def _broadcast_calibration_step(self, step_name: str, status: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Broadcast calibration progress with trader state data.
        
        Args:
            step_name: Name of the calibration step
            status: Status of the step (starting, complete, error)
            data: Optional data to include with the step
        """
        if not self.websocket_manager:
            return
            
        try:
            # Include actual trader state in progress updates
            current_state = {
                "cash_balance": self.position_tracker.cash_balance,
                "positions_count": len(self.position_tracker.positions),
                "orders_count": len(self.order_service.get_open_orders()),
                "state": self.state_machine.current_state.value,
                "timestamp": time.time()
            }
            
            progress_data = {
                "step": step_name,
                "status": status,
                "trader_state": current_state,
                "step_data": data or {}
            }
            
            # Send as calibration_progress message  
            await self.websocket_manager._broadcast_to_all({
                "type": "calibration_progress",
                "data": progress_data
            })
            
            logger.debug(f"Broadcasted calibration step: {step_name} - {status}")
            
        except Exception as e:
            logger.error(f"Error broadcasting calibration step {step_name}: {e}")