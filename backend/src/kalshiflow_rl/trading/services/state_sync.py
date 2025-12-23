"""
StateSync - Extracted from OrderManager for TRADER 2.0

Handles synchronization with Kalshi API to reconcile positions, orders, and balance.
Focused extraction of working state reconciliation functionality from the monolithic OrderManager.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..demo_client import KalshiDemoTradingClient, KalshiDemoTradingClientError

logger = logging.getLogger("kalshiflow_rl.trading.services.state_sync")


class StateSync:
    """
    Handles state synchronization with Kalshi API.
    
    Extracted from OrderManager to focus solely on keeping local state
    in sync with the authoritative state on Kalshi servers.
    """
    
    def __init__(
        self,
        client: KalshiDemoTradingClient,
        position_tracker: 'PositionTracker',
        order_service: 'OrderService',
        status_logger: Optional['StatusLogger'] = None,
        websocket_manager=None
    ):
        """
        Initialize StateSync.
        
        Args:
            client: KalshiDemoTradingClient for API integration
            position_tracker: PositionTracker instance to sync
            order_service: OrderService instance to sync
            status_logger: Optional StatusLogger for activity tracking
            websocket_manager: Global WebSocketManager for broadcasting (optional)
        """
        self.client = client
        self.position_tracker = position_tracker
        self.order_service = order_service
        self.status_logger = status_logger
        self.websocket_manager = websocket_manager
        
        # Sync tracking
        self.last_position_sync: Optional[float] = None
        self.last_order_sync: Optional[float] = None
        self.last_balance_sync: Optional[float] = None
        
        # Sync intervals (seconds)
        self.position_sync_interval = 300  # 5 minutes
        self.order_sync_interval = 60     # 1 minute
        self.balance_sync_interval = 300  # 5 minutes
        
        # Sync statistics
        self.sync_stats = {
            "positions_synced": 0,
            "orders_synced": 0,
            "balance_synced": 0,
            "sync_errors": 0,
            "last_error": None
        }
        
        logger.info("StateSync initialized")
    
    async def sync_positions(self) -> Dict[str, Any]:
        """
        Sync positions with Kalshi API.
        
        Returns:
            Dict with sync results and statistics
        """
        start_time = time.time()
        sync_result = {
            "success": False,
            "positions_updated": 0,
            "drift_detected": {},
            "error": None,
            "duration": 0.0
        }
        error_msg = None  # Initialize error_msg
        
        try:
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "StateSync", "syncing_positions",
                    {"operation": "get_positions"}
                )
            
            # Get positions from Kalshi
            positions_response = await self.client.get_positions()
            
            if "positions" in positions_response:
                kalshi_positions = positions_response["positions"]
                
                # Convert to format expected by PositionTracker
                api_positions = {}
                for kalshi_pos in kalshi_positions:
                    ticker = kalshi_pos.get("ticker", "")
                    if ticker:
                        # Convert Kalshi position format to our internal format
                        position_count = kalshi_pos.get("position", 0)
                        total_cost = kalshi_pos.get("total_cost", 0.0)
                        
                        # Handle Kalshi position format quirks
                        if isinstance(total_cost, int):
                            total_cost = total_cost / 100.0  # Convert cents to dollars
                        
                        api_positions[ticker] = {
                            "ticker": ticker,
                            "position": position_count,
                            "total_cost": total_cost
                        }
                
                # Sync positions using PositionTracker
                position_sync_results = self.position_tracker.sync_positions_from_api(api_positions)
                
                sync_result["success"] = True
                sync_result["positions_updated"] = len(position_sync_results)
                
                # Track drift
                for ticker, status in position_sync_results.items():
                    if "drift" in status:
                        sync_result["drift_detected"][ticker] = status
                
                # Update sync timestamp
                self.last_position_sync = time.time()
                self.sync_stats["positions_synced"] += 1
                
                logger.info(f"Position sync completed: {len(api_positions)} positions processed")
                
            else:
                # No positions field means no positions - this is actually success with 0 positions
                sync_result["success"] = True
                sync_result["positions_updated"] = 0
                self.last_position_sync = time.time()
                self.sync_stats["positions_synced"] += 1
                logger.info("No positions field in API response - treating as 0 positions")
        
        except Exception as e:
            error_msg = str(e)
            sync_result["error"] = error_msg
            self.sync_stats["sync_errors"] += 1
            self.sync_stats["last_error"] = error_msg
            logger.error(f"Failed to sync positions: {error_msg}")
        
        finally:
            sync_result["duration"] = time.time() - start_time
            
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "positions_synced" if sync_result["success"] else "positions_sync_error",
                    f"{sync_result['positions_updated']} positions" if sync_result["success"] else (error_msg[:50] if error_msg else "unknown error"),
                    sync_result["duration"]
                )
        
        return sync_result
    
    async def sync_orders(self) -> Dict[str, Any]:
        """
        Sync orders with Kalshi API.
        
        Returns:
            Dict with sync results and statistics
        """
        start_time = time.time()
        sync_result = {
            "success": False,
            "orders_updated": 0,
            "orders_cancelled": 0,
            "orders_filled": 0,
            "error": None,
            "duration": 0.0
        }
        error_msg = None  # Initialize error_msg
        
        try:
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "StateSync", "syncing_orders",
                    {"operation": "get_orders"}
                )
            
            # Get orders from Kalshi
            orders_response = await self.client.get_orders()
            
            if "orders" in orders_response:
                kalshi_orders = orders_response["orders"]
                
                # Build set of active Kalshi order IDs
                active_kalshi_orders = set()
                
                for kalshi_order in kalshi_orders:
                    kalshi_order_id = kalshi_order.get("order_id", "")
                    order_status = kalshi_order.get("status", "").lower()
                    
                    if kalshi_order_id:
                        active_kalshi_orders.add(kalshi_order_id)
                        
                        # Find corresponding local order
                        local_order = self.order_service.get_order_by_kalshi_id(kalshi_order_id)
                        if local_order:
                            # Check for status changes
                            if order_status == "filled" and local_order.status != 1:  # OrderStatus.FILLED
                                # Order was filled on Kalshi side
                                fill_data = {
                                    "yes_price": kalshi_order.get("yes_price", local_order.limit_price),
                                    "count": kalshi_order.get("remaining_count", local_order.quantity)
                                }
                                
                                # Process fill through OrderService
                                filled_order = self.order_service.handle_fill_event(kalshi_order_id, fill_data)
                                if filled_order:
                                    sync_result["orders_filled"] += 1
                                    logger.info(f"Synced order fill: {filled_order.order_id}")
                            
                            elif order_status == "canceled" and local_order.status == 0:  # OrderStatus.PENDING
                                # Order was cancelled on Kalshi side
                                # Update local order status
                                local_order.status = 2  # OrderStatus.CANCELLED
                                
                                # Remove from open orders
                                if local_order.order_id in self.order_service.open_orders:
                                    del self.order_service.open_orders[local_order.order_id]
                                    self.order_service.order_history.append(local_order)
                                
                                sync_result["orders_cancelled"] += 1
                                logger.info(f"Synced order cancellation: {local_order.order_id}")
                
                # Check for orders that exist locally but not on Kalshi (likely filled/cancelled)
                local_kalshi_mapping = self.order_service._kalshi_order_mapping
                for our_order_id, kalshi_order_id in list(local_kalshi_mapping.items()):
                    if kalshi_order_id not in active_kalshi_orders:
                        # Order no longer active on Kalshi - assume filled or cancelled
                        local_order = self.order_service.get_order_by_id(our_order_id)
                        if local_order and local_order.status == 0:  # Still pending locally
                            # Remove from tracking (status unknown, but no longer active)
                            if our_order_id in self.order_service.open_orders:
                                del self.order_service.open_orders[our_order_id]
                                local_order.status = 2  # Mark as cancelled
                                self.order_service.order_history.append(local_order)
                                
                                logger.warning(f"Order {our_order_id} no longer active on Kalshi - marked as cancelled")
                
                sync_result["success"] = True
                sync_result["orders_updated"] = len(kalshi_orders)
                
                # Update sync timestamp
                self.last_order_sync = time.time()
                self.sync_stats["orders_synced"] += 1
                
                logger.info(f"Order sync completed: {len(kalshi_orders)} orders processed")
                
            else:
                # No orders field means no orders - this is actually success with 0 orders
                sync_result["success"] = True
                sync_result["orders_updated"] = 0
                self.last_order_sync = time.time()
                self.sync_stats["orders_synced"] += 1
                logger.info("No orders field in API response - treating as 0 orders")
        
        except Exception as e:
            error_msg = str(e)
            sync_result["error"] = error_msg
            self.sync_stats["sync_errors"] += 1
            self.sync_stats["last_error"] = error_msg
            logger.error(f"Failed to sync orders: {error_msg}")
        
        finally:
            sync_result["duration"] = time.time() - start_time
            
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "orders_synced" if sync_result["success"] else "orders_sync_error",
                    f"{sync_result['orders_updated']} orders" if sync_result["success"] else (error_msg[:50] if error_msg else "unknown error"),
                    sync_result["duration"]
                )
        
        return sync_result
    
    async def sync_balance(self) -> Dict[str, Any]:
        """
        Sync account balance with Kalshi API.
        
        Returns:
            Dict with sync results and statistics
        """
        start_time = time.time()
        sync_result = {
            "success": False,
            "old_balance": self.position_tracker.cash_balance,
            "new_balance": 0.0,
            "balance_drift": 0.0,
            "error": None,
            "duration": 0.0
        }
        error_msg = None  # Initialize error_msg
        
        try:
            if self.status_logger:
                await self.status_logger.log_service_status(
                    "StateSync", "syncing_balance",
                    {"operation": "get_account_info"}
                )
            
            # Get account info from Kalshi
            account_response = await self.client.get_account_info()
            
            if "balance" in account_response:
                # Extract balance (might be in cents)
                api_balance = account_response["balance"]
                
                # Convert to dollars if needed
                if isinstance(api_balance, int) and api_balance > 1000:
                    # Likely in cents
                    api_balance_dollars = api_balance / 100.0
                else:
                    # Already in dollars
                    api_balance_dollars = float(api_balance)
                
                # Calculate drift
                old_balance = self.position_tracker.cash_balance
                balance_drift = abs(api_balance_dollars - old_balance)
                
                # Update balance in PositionTracker
                self.position_tracker.cash_balance = api_balance_dollars
                
                # Also store API portfolio_value if available (existing positions value)
                if "portfolio_value" in account_response:
                    api_portfolio_value = account_response["portfolio_value"]
                    # Convert cents to dollars
                    if isinstance(api_portfolio_value, int):
                        self.position_tracker.api_portfolio_value = api_portfolio_value / 100.0
                    else:
                        self.position_tracker.api_portfolio_value = float(api_portfolio_value)
                
                sync_result["success"] = True
                sync_result["new_balance"] = api_balance_dollars
                sync_result["balance_drift"] = balance_drift
                
                # Update sync timestamp
                self.last_balance_sync = time.time()
                self.sync_stats["balance_synced"] += 1
                
                if balance_drift > 0.01:  # More than 1 cent drift
                    logger.warning(f"Balance drift detected: ${old_balance:.2f} -> ${api_balance_dollars:.2f} (drift: ${balance_drift:.2f})")
                else:
                    logger.debug(f"Balance synced: ${api_balance_dollars:.2f}")
                
            else:
                # No balance field - this shouldn't happen but treat as 0 balance
                sync_result["success"] = True
                sync_result["new_balance"] = 0.0
                sync_result["balance_drift"] = self.position_tracker.cash_balance
                self.position_tracker.update_cash_balance_from_api(0.0)
                self.last_balance_sync = time.time()
                self.sync_stats["balance_synced"] += 1
                logger.warning("No balance field in API response - treating as $0 balance")
        
        except Exception as e:
            error_msg = str(e)
            sync_result["error"] = error_msg
            self.sync_stats["sync_errors"] += 1
            self.sync_stats["last_error"] = error_msg
            logger.error(f"Failed to sync balance: {error_msg}")
        
        finally:
            sync_result["duration"] = time.time() - start_time
            
            if self.status_logger:
                await self.status_logger.log_action_result(
                    "balance_synced" if sync_result["success"] else "balance_sync_error",
                    f"${sync_result['new_balance']:.2f}" if sync_result["success"] else (error_msg[:50] if error_msg else "unknown error"),
                    sync_result["duration"]
                )
        
        return sync_result
    
    async def sync_all(self) -> Dict[str, Any]:
        """
        Sync all state components (positions, orders, balance).
        
        Returns:
            Dict with combined sync results
        """
        start_time = time.time()
        
        logger.info("Starting full state sync")
        
        if self.status_logger:
            await self.status_logger.log_service_status(
                "StateSync", "syncing_all",
                {"operation": "full_sync"}
            )
        
        # Run syncs in parallel for efficiency
        results = await asyncio.gather(
            self.sync_positions(),
            self.sync_orders(),
            self.sync_balance(),
            return_exceptions=True
        )
        
        position_result, order_result, balance_result = results
        
        # Handle exceptions
        if isinstance(position_result, Exception):
            position_result = {"success": False, "error": str(position_result)}
        if isinstance(order_result, Exception):
            order_result = {"success": False, "error": str(order_result)}
        if isinstance(balance_result, Exception):
            balance_result = {"success": False, "error": str(balance_result)}
        
        # Combine results
        combined_result = {
            "success": all([
                position_result.get("success", False),
                order_result.get("success", False),
                balance_result.get("success", False)
            ]),
            "positions": position_result,
            "orders": order_result,
            "balance": balance_result,
            "total_duration": time.time() - start_time
        }
        
        # Log summary
        if combined_result["success"]:
            logger.info(f"Full state sync completed successfully ({combined_result['total_duration']:.2f}s)")
        else:
            logger.error(f"Full state sync had errors ({combined_result['total_duration']:.2f}s)")
        
        if self.status_logger:
            await self.status_logger.log_action_result(
                "full_sync_completed" if combined_result["success"] else "full_sync_error",
                f"pos:{position_result.get('positions_updated', 0)} ord:{order_result.get('orders_updated', 0)} bal:${balance_result.get('new_balance', 0):.0f}",
                combined_result['total_duration']
            )
        
        # Broadcast sync complete
        if self.websocket_manager:
            await self._broadcast_sync_complete(combined_result)
        
        return combined_result
    
    def should_sync_positions(self) -> bool:
        """Check if positions need syncing based on interval."""
        if self.last_position_sync is None:
            return True
        return time.time() - self.last_position_sync > self.position_sync_interval
    
    def should_sync_orders(self) -> bool:
        """Check if orders need syncing based on interval."""
        if self.last_order_sync is None:
            return True
        return time.time() - self.last_order_sync > self.order_sync_interval
    
    def should_sync_balance(self) -> bool:
        """Check if balance needs syncing based on interval."""
        if self.last_balance_sync is None:
            return True
        return time.time() - self.last_balance_sync > self.balance_sync_interval
    
    async def auto_sync(self) -> Dict[str, Any]:
        """
        Perform automatic sync based on intervals.
        
        Returns:
            Dict with sync operations performed
        """
        operations = []
        
        if self.should_sync_positions():
            operations.append("positions")
        if self.should_sync_orders():
            operations.append("orders")
        if self.should_sync_balance():
            operations.append("balance")
        
        if not operations:
            return {"operations": [], "message": "No sync needed"}
        
        logger.debug(f"Auto-sync performing: {operations}")
        
        # Perform only needed syncs
        tasks = []
        if "positions" in operations:
            tasks.append(self.sync_positions())
        if "orders" in operations:
            tasks.append(self.sync_orders())
        if "balance" in operations:
            tasks.append(self.sync_balance())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            "operations": operations,
            "results": results,
            "success": all(r.get("success", False) if isinstance(r, dict) else False for r in results)
        }
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get sync status and statistics."""
        current_time = time.time()
        
        return {
            "last_syncs": {
                "positions": {
                    "timestamp": self.last_position_sync,
                    "age_seconds": current_time - self.last_position_sync if self.last_position_sync else None,
                    "needs_sync": self.should_sync_positions()
                },
                "orders": {
                    "timestamp": self.last_order_sync,
                    "age_seconds": current_time - self.last_order_sync if self.last_order_sync else None,
                    "needs_sync": self.should_sync_orders()
                },
                "balance": {
                    "timestamp": self.last_balance_sync,
                    "age_seconds": current_time - self.last_balance_sync if self.last_balance_sync else None,
                    "needs_sync": self.should_sync_balance()
                }
            },
            "statistics": self.sync_stats.copy(),
            "intervals": {
                "position_sync_interval": self.position_sync_interval,
                "order_sync_interval": self.order_sync_interval,
                "balance_sync_interval": self.balance_sync_interval
            }
        }
    
    async def _broadcast_sync_complete(self, result: Dict[str, Any]) -> None:
        """
        Broadcast sync complete via WebSocket.
        
        Args:
            result: Sync result data
        """
        if self.websocket_manager:
            try:
                # Use initialization complete message for compatibility
                sync_data = {
                    "sync_successful": result["success"],
                    "positions_synced": result["positions"].get("positions_updated", 0),
                    "orders_synced": result["orders"].get("orders_updated", 0),
                    "balance": result["balance"].get("new_balance", 0),
                    "duration": result["total_duration"],
                    "timestamp": time.time()
                }
                
                # Could use a specific sync message or reuse initialization complete
                await self.websocket_manager.broadcast_initialization_complete(sync_data)
            except Exception as e:
                logger.warning(f"Failed to broadcast sync complete: {e}")