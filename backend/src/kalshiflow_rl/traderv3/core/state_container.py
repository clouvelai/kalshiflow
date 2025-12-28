"""
V3 State Container - Central State Management for TRADER V3.

This module provides a centralized container for all state in the V3 system,
serving as the single source of truth for trading data, health metrics, and
system status. It's NOT a state machine - just organized, versioned storage.

Purpose:
    The V3StateContainer consolidates all system state into one place, making
    it easy to access current data, track changes, and broadcast updates. It
    provides versioning to detect changes and clean APIs for state access.

Key Responsibilities:
    1. **Trading State Storage** - Positions, orders, balance from Kalshi
    2. **Health Tracking** - Component health metrics and error counts
    3. **State Machine Reference** - Current operational state
    4. **Change Detection** - Version tracking for efficient updates
    5. **State Aggregation** - Combines all state for broadcasting

State Types Managed:
    - Trading State: Balance, positions, orders from Kalshi API
    - Component Health: Health status of each V3 component
    - Machine State: Current state from the V3StateMachine
    - Container Metadata: Timestamps, versions, uptime

Architecture Position:
    The StateContainer is used by:
    - V3Coordinator: Updates all state types and reads for broadcasting
    - V3TradingClientIntegration: Provides trading state updates
    - V3WebSocketManager: Reads state for client broadcasts
    - HTTP endpoints: Query container for /status and /health

Design Principles:
    - **Single Source of Truth**: All state in one place
    - **Immutable Updates**: State is replaced, not mutated
    - **Version Tracking**: Detect changes efficiently
    - **Clean APIs**: Simple methods for state access
    - **No Business Logic**: Just storage and retrieval

Thread Safety:
    Designed for single-threaded async operation. All access
    should occur from the same asyncio event loop.
"""

import time
import logging
import asyncio
from collections import deque
from typing import Dict, Any, Optional, Callable, Awaitable, TypeVar, List
from dataclasses import dataclass, field
from enum import Enum

from ..state.trader_state import TraderState, StateChange, SessionPnLState
from .state_machine import TraderState as V3State  # State machine states
from ..services.whale_tracker import BigBet

logger = logging.getLogger("kalshiflow_rl.traderv3.core.state_container")

T = TypeVar('T')


@dataclass
class ComponentHealth:
    """
    Health status for a single V3 component.
    
    Tracks the health of individual components with error counting
    and staleness detection. Used to monitor system health and
    trigger recovery when components fail.
    
    Attributes:
        name: Component identifier (e.g., "orderbook_integration")
        healthy: Current health status
        last_check: Timestamp of last health update
        details: Component-specific health details
        error_count: Number of errors since last healthy state
        last_error: Most recent error message
    """
    name: str
    healthy: bool
    last_check: float
    details: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None
    
    def update(self, healthy: bool, details: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        """Update health status."""
        self.healthy = healthy
        self.last_check = time.time()
        if details:
            self.details = details
        if error:
            self.last_error = error
            self.error_count += 1
        elif healthy:
            self.error_count = 0  # Reset on healthy status


@dataclass
class WhaleQueueState:
    """
    State for the whale tracker queue.

    Tracks the current state of whale detection, including the queue
    of big bets and statistics about trade filtering.

    Attributes:
        queue: List of BigBet objects in the whale queue
        trades_seen: Total number of public trades processed
        trades_discarded: Number of trades that didn't meet threshold
        window_minutes: Sliding window duration
        last_update: Timestamp of last queue update
    """
    queue: List[BigBet] = field(default_factory=list)
    trades_seen: int = 0
    trades_discarded: int = 0
    window_minutes: int = 5
    last_update: float = 0.0


class V3StateContainer:
    """
    Central container for all V3 state management.
    
    This is the single source of truth for all system state in TRADER V3.
    It provides organized storage with versioning, change detection, and
    clean APIs for state access. It does NOT implement any business logic -
    just storage, retrieval, and aggregation.
    
    Key Features:
        - **Versioned Trading State**: Tracks changes with version numbers
        - **Health Monitoring**: Component health with staleness detection
        - **State Machine Reference**: Current operational state
        - **Change Detection**: Only broadcasts when state changes
        - **State Aggregation**: Combines all state for easy access
    
    Attributes:
        _trading_state: Current positions, orders, balance from Kalshi
        _last_state_change: Delta from previous trading state
        _trading_state_version: Increments on each state change
        _component_health: Health status for each component
        _machine_state: Current state from V3StateMachine
        _machine_state_context: Human-readable state description
        _machine_state_metadata: Additional state metadata
    
    Usage Pattern:
        ```python
        container = V3StateContainer()
        
        # Update trading state
        state_changed = container.update_trading_state(new_state, changes)
        
        # Check component health
        container.update_component_health("orderbook", True)
        is_healthy = container.is_system_healthy()
        
        # Get state for broadcasting
        full_state = container.get_full_state()
        ```
    """
    
    def __init__(self):
        """Initialize state container."""
        # Trading state - data from Kalshi
        self._trading_state: Optional[TraderState] = None
        self._last_state_change: Optional[StateChange] = None
        self._trading_state_version = 0  # Increment on each update for change detection

        # Session P&L state - tracks P&L from session start
        self._session_pnl_state: Optional[SessionPnLState] = None

        # Session-updated tickers - tracks which positions were updated this session
        # via real-time WebSocket (not from initial sync)
        self._session_updated_tickers: set = set()

        # Settled positions history - stores last 50 closed positions for UI display
        # Positions are captured here before being removed from active positions
        self._settled_positions: deque = deque(maxlen=50)
        self._total_settlements_count = 0

        # Whale queue state - data from whale tracker
        self._whale_state: Optional[WhaleQueueState] = None
        self._whale_state_version = 0  # Increment on each whale queue update

        # Component health tracking
        self._component_health: Dict[str, ComponentHealth] = {}
        self._health_check_interval = 30.0  # Expected interval between health checks

        # State machine reference (set by coordinator)
        self._machine_state: Optional[V3State] = None
        self._machine_state_context: str = ""
        self._machine_state_metadata: Dict[str, Any] = {}

        # Container metadata
        self._created_at = time.time()
        self._last_update = time.time()

        # Versioning and locking for atomic updates
        self._global_version = 0  # Increments on ANY state change
        self._lock = asyncio.Lock()  # For atomic operations

        logger.info("V3StateContainer initialized with versioning protocol")
    
    # ======== Trading State Management ========
    
    def update_trading_state(self, state: TraderState, changes: Optional[StateChange] = None) -> bool:
        """
        Update trading state from Kalshi sync.
        
        This is called after each Kalshi API sync to update the trading
        state. It performs change detection to avoid unnecessary updates
        and increments the version number when changes occur.
        
        Args:
            state: New trader state from Kalshi with positions, orders, balance
            changes: Optional delta object showing what changed
            
        Returns:
            True if state actually changed (version incremented),
            False if state is identical to current (no version change)
        
        Side Effects:
            - Updates _trading_state if changed
            - Increments _trading_state_version if changed
            - Logs significant changes (balance, positions, orders)
        
        Note:
            Ignores sync_timestamp changes as those occur on every sync
            even when no actual data changes.
        """
        # Check if state actually changed
        if self._trading_state and self._states_are_equal(self._trading_state, state):
            logger.debug("Trading state unchanged, skipping update")
            return False
        
        self._trading_state = state
        self._last_state_change = changes
        self._trading_state_version += 1
        self._last_update = time.time()
        
        # Log significant changes
        if changes:
            if changes.balance_change != 0:
                logger.info(f"Balance changed: {changes.balance_change:+d} cents")
            if changes.position_count_change != 0:
                logger.info(f"Positions changed: {changes.position_count_change:+d}")
            if changes.order_count_change != 0:
                logger.info(f"Orders changed: {changes.order_count_change:+d}")
        
        return True
    
    def _states_are_equal(self, state1: TraderState, state2: TraderState) -> bool:
        """
        Compare two trading states for equality.

        Only compares the important fields, not timestamps.
        Note: Intentionally ignores sync_timestamp as that updates every sync.
        """
        return (
            state1.balance == state2.balance and
            state1.portfolio_value == state2.portfolio_value and
            state1.position_count == state2.position_count and
            state1.order_count == state2.order_count and
            state1.positions == state2.positions and
            state1.orders == state2.orders
        )

    def update_single_position(self, ticker: str, position_data: Dict[str, Any]) -> bool:
        """
        Update a single position from real-time WebSocket push.

        This is called when receiving market_position updates from the
        PositionListener WebSocket subscription. It updates the position
        in-place without requiring a full API sync.

        Args:
            ticker: Market ticker to update
            position_data: Position data dict from WebSocket with keys:
                - position: Contract count (+ long, - short)
                - market_exposure: Current market value in cents
                - realized_pnl: Realized P&L in cents
                - fees_paid: Fees in cents
                - volume: Total volume traded
                - last_updated: Timestamp of WebSocket update
            Note: total_traded (cost basis) is NOT in WebSocket updates -
            it's preserved from REST sync via merge.

        Returns:
            True if state was updated, False if no trading state exists

        Side Effects:
            - Updates the specific position in _trading_state.positions
            - Recalculates position_count
            - Recalculates portfolio_value from position costs
            - Increments _trading_state_version
        """
        if not self._trading_state:
            logger.warning(f"Cannot update position {ticker}: no trading state")
            return False

        position_count = position_data.get("position", 0)

        # If position is 0, capture settlement data then remove from positions dict
        if position_count == 0:
            if ticker in self._trading_state.positions:
                # Capture position data before deletion for settlement history
                closing_position = self._trading_state.positions[ticker]
                settlement = {
                    "ticker": ticker,
                    "position": closing_position.get("position", 0),
                    "side": "yes" if closing_position.get("position", 0) > 0 else "no",
                    "total_traded": closing_position.get("total_traded", 0),
                    "market_exposure": closing_position.get("market_exposure", 0),
                    "realized_pnl": position_data.get("realized_pnl", 0),
                    "fees_paid": position_data.get("fees_paid", 0),
                    "closed_at": time.time(),
                }
                self._settled_positions.appendleft(settlement)
                self._total_settlements_count += 1

                del self._trading_state.positions[ticker]
                logger.info(
                    f"Position closed: {ticker}, "
                    f"realized_pnl={settlement['realized_pnl']}¢"
                )
        else:
            # Update or add position - MERGE to preserve fields not in WebSocket update
            # This is critical: WebSocket updates don't include total_traded (cost basis),
            # which comes from REST sync. Merging ensures we preserve total_traded.
            existing = self._trading_state.positions.get(ticker, {})
            merged = {**existing, **position_data}
            self._trading_state.positions[ticker] = merged
            logger.info(
                f"Position updated: {ticker} = {position_count} contracts, "
                f"value={position_data.get('market_exposure', 0)}¢"
            )

        # Track that this ticker was updated this session via WebSocket
        self._session_updated_tickers.add(ticker)

        # Recalculate aggregates
        self._trading_state.position_count = len(self._trading_state.positions)

        # Recalculate portfolio_value as sum of current market values
        # Note: REST sync from Kalshi provides authoritative portfolio_value;
        # this is a local estimate between syncs based on WebSocket updates.
        new_portfolio_value = sum(
            pos.get("market_exposure", 0)
            for pos in self._trading_state.positions.values()
        )
        self._trading_state.portfolio_value = new_portfolio_value

        # Update version and timestamp
        self._trading_state_version += 1
        self._last_update = time.time()

        logger.debug(
            f"Position update complete: {len(self._trading_state.positions)} positions, "
            f"portfolio_value={new_portfolio_value}¢, version={self._trading_state_version}"
        )

        return True

    @property
    def trading_state(self) -> Optional[TraderState]:
        """Get current trading state."""
        return self._trading_state
    
    @property
    def last_state_change(self) -> Optional[StateChange]:
        """Get last state change."""
        return self._last_state_change
    
    @property
    def trading_state_version(self) -> int:
        """Get trading state version (increments on change)."""
        return self._trading_state_version
    
    def _format_order_list(self, order_group_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Format orders for frontend display.

        Filters orders by order_group_id if provided, and returns a
        simplified list suitable for frontend rendering.

        Args:
            order_group_id: Optional order group ID to filter by.
                           If None, returns all orders.

        Returns:
            List of formatted order dicts with keys:
                - order_id: First 8 chars of order ID
                - ticker: Market ticker
                - side: "yes" or "no"
                - action: "buy" or "sell"
                - price: Price in cents
                - count: Number of contracts
                - status: Order status
                - created_time: ISO timestamp or raw value
        """
        if not self._trading_state or not self._trading_state.orders:
            return []

        formatted_orders = []
        for order_id, order in self._trading_state.orders.items():
            # Filter by order_group_id if specified
            if order_group_id:
                if order.get("order_group_id") != order_group_id:
                    continue

            # Determine the price - Kalshi uses yes_price or no_price
            price = order.get("yes_price") or order.get("no_price") or 0

            formatted_orders.append({
                "order_id": order_id[:8] if order_id else "unknown",
                "ticker": order.get("ticker", ""),
                "side": order.get("side", ""),
                "action": order.get("action", ""),
                "price": price,
                "count": order.get("remaining_count", order.get("count", 0)),
                "status": order.get("status", ""),
                "created_time": order.get("created_time", "")
            })

        return formatted_orders

    def get_trading_summary(self, order_group_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trading state summary for broadcasting.

        Provides a clean summary of trading state suitable for
        WebSocket broadcast to frontend clients. Includes version
        number for change detection on the client side.

        Args:
            order_group_id: Optional order group ID to filter orders by.

        Returns:
            Dict containing:
                - has_state: Whether trading state exists
                - version: State version number (for change detection)
                - balance: Account balance in cents
                - portfolio_value: Total portfolio value in cents
                - position_count: Number of open positions
                - order_count: Number of open orders
                - positions: List of market tickers with positions
                - open_orders: Count of open orders
                - order_list: Formatted list of orders for display
                - changes: Delta from last state (if available)

        Usage:
            Called by V3Coordinator to prepare trading state
            for WebSocket broadcast to frontend clients.
        """
        if not self._trading_state:
            return {
                "has_state": False,
                "version": 0
            }

        state = self._trading_state

        summary = {
            "has_state": True,
            "version": self._trading_state_version,
            "balance": state.balance,  # In cents
            "portfolio_value": state.portfolio_value,  # In cents
            "position_count": state.position_count,
            "order_count": state.order_count,
            "sync_timestamp": state.sync_timestamp,
            "positions": list(state.positions.keys()),  # Just tickers
            "open_orders": len(state.orders),  # Count of orders
            "order_list": self._format_order_list(order_group_id)  # Formatted order list
        }

        # Add order group info if available
        if state.order_group:
            summary["order_group"] = {
                "id": state.order_group.order_group_id[:8] if state.order_group.order_group_id else "none",
                "status": "active",
                "order_count": len(state.order_group.order_ids),
                "order_ids": [oid[:8] for oid in state.order_group.order_ids[:5]],
            }
        else:
            summary["order_group"] = None

        # Add changes if available
        if self._last_state_change:
            summary["changes"] = {
                "balance": self._last_state_change.balance_change,
                "portfolio_value": self._last_state_change.portfolio_value_change,
                "positions": self._last_state_change.position_count_change,
                "orders": self._last_state_change.order_count_change
            }

        # Add detailed position data with per-position P&L
        # (computed first so we can pass to compute_pnl for aggregation)
        positions_details = self._format_position_details()
        summary["positions_details"] = positions_details

        # Add session P&L if initialized (includes realized/unrealized breakdown)
        if self._session_pnl_state and self._trading_state:
            summary["pnl"] = self._session_pnl_state.compute_pnl(
                self._trading_state.balance,
                self._trading_state.portfolio_value,
                positions_details
            )

        # Add session-updated positions info
        summary["session_updates"] = {
            "updated_tickers": list(self._session_updated_tickers),
            "count": len(self._session_updated_tickers),
        }

        # Add settlements history for UI display
        summary["settlements"] = list(self._settled_positions)
        summary["settlements_count"] = self._total_settlements_count

        return summary

    # ======== Session P&L Management ========

    def initialize_session_pnl(self, balance: int, portfolio_value: int) -> None:
        """
        Initialize session P&L tracking on first sync.

        Called by the coordinator after the first successful Kalshi sync
        to capture the starting state for session P&L calculation.

        Args:
            balance: Starting balance in cents
            portfolio_value: Starting portfolio value in cents

        Note:
            This is a one-time initialization per session. Subsequent calls
            are ignored to preserve the original session start state.
        """
        if self._session_pnl_state is None:
            self._session_pnl_state = SessionPnLState(
                session_start_time=time.time(),
                starting_balance=balance,
                starting_portfolio_value=portfolio_value
            )
            logger.info(
                f"Session P&L initialized: starting equity "
                f"{balance + portfolio_value} cents "
                f"(balance={balance}, portfolio={portfolio_value})"
            )

    def _format_position_details(self) -> List[Dict[str, Any]]:
        """
        Format positions with P&L for frontend display.

        Extracts full position data from TraderState and calculates
        unrealized P&L per position for the frontend to display.

        Returns:
            List of position dicts with P&L metrics. Each position contains:
                - ticker: Market ticker
                - position: Contract count (positive=YES, negative=NO)
                - side: "yes" or "no"
                - total_traded: Entry cost in cents
                - market_exposure: Current value in cents
                - realized_pnl: Realized P&L in cents
                - unrealized_pnl: Unrealized P&L in cents
                - fees_paid: Fees paid in cents
        """
        if not self._trading_state:
            return []

        details = []
        for ticker, pos in self._trading_state.positions.items():
            position_count = pos.get("position", 0)
            total_traded = pos.get("total_traded", 0)
            market_exposure = pos.get("market_exposure", 0)
            # Unrealized P&L = current value - cost basis
            unrealized_pnl = market_exposure - total_traded

            details.append({
                "ticker": ticker,
                "position": position_count,
                "side": "yes" if position_count > 0 else "no",
                "total_traded": total_traded,        # Entry cost (cents)
                "market_exposure": market_exposure,  # Current value (cents)
                "realized_pnl": pos.get("realized_pnl", 0),
                "unrealized_pnl": unrealized_pnl,
                "fees_paid": pos.get("fees_paid", 0),
                "session_updated": ticker in self._session_updated_tickers,  # Was this updated this session?
                "last_updated": pos.get("last_updated"),  # WebSocket update timestamp
            })

        return details

    def _compute_invested_amount(self) -> int:
        """
        Sum of all position market exposures (total invested in positions).

        Returns:
            Total invested amount in cents
        """
        if not self._trading_state:
            return 0
        return sum(
            pos.get("market_exposure", 0)
            for pos in self._trading_state.positions.values()
        )

    # ======== Whale Queue State Management ========

    def update_whale_queue(self, state: WhaleQueueState) -> bool:
        """
        Update whale queue state from whale tracker.

        This is called by the whale tracker when the queue changes,
        either due to new whales being detected or old ones being pruned.

        Args:
            state: New whale queue state with queue contents and stats

        Returns:
            True if state was updated (version incremented),
            False if state is identical to current

        Side Effects:
            - Updates _whale_state
            - Increments _whale_state_version
            - Updates _last_update timestamp
        """
        # Always update whale state (queue changes are meaningful)
        self._whale_state = state
        self._whale_state_version += 1
        self._last_update = time.time()

        logger.debug(
            f"Whale queue updated: {len(state.queue)} whales, "
            f"trades_seen={state.trades_seen}, discarded={state.trades_discarded}"
        )

        return True

    @property
    def whale_state(self) -> Optional[WhaleQueueState]:
        """Get current whale queue state."""
        return self._whale_state

    @property
    def whale_state_version(self) -> int:
        """Get whale state version (increments on change)."""
        return self._whale_state_version

    def get_whale_summary(self) -> Dict[str, Any]:
        """
        Get whale queue summary for broadcasting.

        Provides a clean summary of whale queue state suitable for
        WebSocket broadcast to frontend clients.

        Returns:
            Dict containing:
                - has_state: Whether whale state exists
                - version: State version number
                - queue: List of whale bets (serialized)
                - stats: Queue statistics
        """
        if not self._whale_state:
            return {
                "has_state": False,
                "version": 0
            }

        state = self._whale_state
        now_ms = int(time.time() * 1000)

        # Serialize queue for transport
        queue_data = [bet.to_dict(now_ms) for bet in state.queue]

        # Calculate discard rate
        discard_rate = 0.0
        if state.trades_seen > 0:
            discard_rate = (state.trades_discarded / state.trades_seen) * 100

        return {
            "has_state": True,
            "version": self._whale_state_version,
            "queue": queue_data,
            "stats": {
                "trades_seen": state.trades_seen,
                "trades_discarded": state.trades_discarded,
                "discard_rate_percent": round(discard_rate, 1),
                "queue_size": len(state.queue),
                "window_minutes": state.window_minutes,
                "last_update": state.last_update,
            }
        }

    # ======== Component Health Management ========

    def update_component_health(
        self, 
        name: str, 
        healthy: bool, 
        details: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update health status for a component.
        
        Called by the V3Coordinator's health monitoring task to
        record the health status of each component. Tracks error
        counts and timestamps for staleness detection.
        
        Args:
            name: Component identifier (e.g., "event_bus", "orderbook_integration")
            healthy: Current health status of the component
            details: Optional component-specific health details
            error: Optional error message if unhealthy
        
        Side Effects:
            - Creates ComponentHealth if first time seeing component
            - Updates existing ComponentHealth if already tracked
            - Increments error count if error provided
            - Resets error count when component becomes healthy
        
        Note:
            Components are considered stale if not updated within
            2x the health check interval (60 seconds by default).
        """
        if name not in self._component_health:
            self._component_health[name] = ComponentHealth(
                name=name,
                healthy=healthy,
                last_check=time.time(),
                details=details or {}
            )
        else:
            self._component_health[name].update(healthy, details, error)
        
        self._last_update = time.time()
    
    def get_component_health(self, name: str) -> Optional[ComponentHealth]:
        """Get health status for specific component."""
        return self._component_health.get(name)
    
    def get_all_component_health(self) -> Dict[str, ComponentHealth]:
        """Get health status for all components."""
        return self._component_health.copy()
    
    def is_system_healthy(self) -> bool:
        """Check if all components are healthy."""
        if not self._component_health:
            return True  # No components registered yet

        return all(c.healthy for c in self._component_health.values())

    # ======== Component Degradation Tracking ========

    def set_component_degraded(self, component: str, is_degraded: bool, reason: str = None) -> None:
        """
        Track degraded status per component.

        Non-critical components can be marked as degraded while the system
        remains in READY state. This allows the system to continue operating
        with reduced functionality.

        Args:
            component: Name of the component (e.g., "orderbook_integration")
            is_degraded: True if component is degraded, False if recovered
            reason: Human-readable reason for degradation (optional)

        Side Effects:
            - Updates _machine_state_metadata["degraded_components"]
            - Increments _global_version for change detection
        """
        if "degraded_components" not in self._machine_state_metadata:
            self._machine_state_metadata["degraded_components"] = {}

        if is_degraded:
            self._machine_state_metadata["degraded_components"][component] = reason or "unavailable"
            logger.debug(f"Component marked degraded: {component} - {reason}")
        else:
            self._machine_state_metadata["degraded_components"].pop(component, None)
            logger.debug(f"Component recovered: {component}")

        self._global_version += 1
        self._last_update = time.time()

    def get_degraded_components(self) -> Dict[str, str]:
        """
        Get dict of component -> reason for all degraded components.

        Returns:
            Dictionary mapping component names to degradation reasons.
            Empty dict if no components are degraded.

        Example:
            {"orderbook_integration": "connection lost", "whale_tracker": "unavailable"}
        """
        return self._machine_state_metadata.get("degraded_components", {}).copy()

    def is_fully_operational(self) -> bool:
        """
        Check if no components are degraded.

        Returns:
            True if no components are degraded, False otherwise.

        Note:
            This is different from is_system_healthy() which checks component
            health status. A system can be "healthy" (READY state) but not
            "fully operational" (some non-critical components degraded).
        """
        return len(self.get_degraded_components()) == 0
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for broadcasting."""
        now = time.time()
        
        components = {}
        for name, health in self._component_health.items():
            age = now - health.last_check
            components[name] = {
                "healthy": health.healthy,
                "last_check_age": age,
                "stale": age > self._health_check_interval * 2,  # Consider stale after 2x interval
                "error_count": health.error_count,
                "last_error": health.last_error
            }
        
        degraded_components = self.get_degraded_components()

        return {
            "system_healthy": self.is_system_healthy(),
            "fully_operational": self.is_fully_operational(),
            "components": components,
            "component_count": len(self._component_health),
            "unhealthy_count": sum(1 for c in self._component_health.values() if not c.healthy),
            "degraded_components": degraded_components,
            "degraded_count": len(degraded_components)
        }
    
    # ======== State Machine Reference ========
    
    def update_machine_state(
        self, 
        state: V3State, 
        context: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update state machine reference.
        
        Args:
            state: Current state machine state
            context: State context/description
            metadata: State metadata
        """
        self._machine_state = state
        self._machine_state_context = context
        self._machine_state_metadata = metadata or {}
        self._last_update = time.time()
    
    @property
    def machine_state(self) -> Optional[V3State]:
        """Get current state machine state."""
        return self._machine_state
    
    @property
    def machine_state_context(self) -> str:
        """Get state machine context."""
        return self._machine_state_context
    
    @property
    def machine_state_metadata(self) -> Dict[str, Any]:
        """Get state machine metadata."""
        return self._machine_state_metadata.copy()
    
    # ======== Container Management ========
    
    def get_full_state(self) -> Dict[str, Any]:
        """
        Get complete state snapshot for debugging/inspection.
        
        Provides a comprehensive view of all state in the container,
        including trading data, health metrics, machine state, and
        container metadata. Used by status endpoints and debugging.
        
        Returns:
            Dict containing:
                - trading: Complete trading state summary
                - health: System and component health status
                - machine: Current state machine information
                - container: Metadata about the container itself
        
        Usage:
            Called by HTTP /status endpoint to provide complete
            system visibility for operators and monitoring tools.
        """
        return {
            "trading": self.get_trading_summary(),
            "whale": self.get_whale_summary(),
            "health": self.get_health_summary(),
            "machine": {
                "state": self._machine_state.value if self._machine_state else None,
                "context": self._machine_state_context,
                "metadata": self._machine_state_metadata
            },
            "container": {
                "created_at": self._created_at,
                "last_update": self._last_update,
                "uptime": time.time() - self._created_at,
                "trading_version": self._trading_state_version,
                "whale_version": self._whale_state_version
            }
        }
    
    # ======== Atomic Update Protocol (Step 0: Safety First) ========
    
    async def atomic_update(
        self,
        update_func: Callable[['V3StateContainer'], Awaitable[T]]
    ) -> tuple[T, int]:
        """
        Perform atomic state update with version increment.
        
        This is the primary mechanism for preventing race conditions
        during the refactoring. All state modifications should go
        through this method to ensure atomicity and proper versioning.
        
        The update function receives the container (self) and can
        modify any state. The global version is automatically
        incremented after a successful update.
        
        Args:
            update_func: Async function that performs the state update.
                        Receives the container as argument.
        
        Returns:
            Tuple of (result from update_func, new global version)
        
        Usage:
            ```python
            async def update_trading(container):
                container._trading_state = new_state
                container._trading_state_version += 1
                return True
            
            changed, version = await container.atomic_update(update_trading)
            ```
        
        Thread Safety:
            This method uses an async lock to ensure only one update
            happens at a time, preventing race conditions.
        """
        async with self._lock:
            try:
                # Execute the update function
                result = await update_func(self)
                
                # Increment global version on successful update
                self._global_version += 1
                self._last_update = time.time()
                
                logger.debug(f"Atomic update completed, global version: {self._global_version}")
                return result, self._global_version
                
            except Exception as e:
                logger.error(f"Atomic update failed: {e}")
                raise
    
    def atomic_update_sync(
        self,
        update_func: Callable[['V3StateContainer'], T]
    ) -> tuple[T, int]:
        """
        Perform atomic state update (synchronous version).
        
        This is a synchronous wrapper for atomic updates that don't
        require async operations. It creates a temporary async wrapper
        around the sync function.
        
        Args:
            update_func: Sync function that performs the state update
        
        Returns:
            Tuple of (result from update_func, new global version)
        
        Usage:
            ```python
            def update_health(container):
                container._component_health[name].healthy = True
                return True
            
            # This won't work in async context - use atomic_update instead
            changed, version = container.atomic_update_sync(update_health)
            ```
        
        Note:
            This method is provided for backward compatibility but
            atomic_update (async) is preferred.
        """
        async def async_wrapper(container):
            return update_func(container)
        
        # Create event loop if needed (for sync contexts)
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, can't use sync version
            raise RuntimeError(
                "Cannot use atomic_update_sync from async context. "
                "Use atomic_update instead."
            )
        except RuntimeError:
            # No running loop, we can create one
            return asyncio.run(self.atomic_update(async_wrapper))
    
    @property
    def global_version(self) -> int:
        """
        Get current global version number.
        
        This version increments on ANY state change in the container,
        providing a simple way to detect if anything has changed.
        
        Returns:
            Current global version number
        
        Usage:
            Used by coordinator to detect if state has changed
            and needs to be broadcast to clients.
        """
        return self._global_version
    
    async def compare_and_swap(
        self,
        expected_version: int,
        update_func: Callable[['V3StateContainer'], Awaitable[T]]
    ) -> tuple[bool, T, int]:
        """
        Perform atomic compare-and-swap update.
        
        Only executes the update if the current version matches
        the expected version, preventing lost updates in concurrent
        scenarios.
        
        Args:
            expected_version: Expected current version
            update_func: Update function to execute if version matches
        
        Returns:
            Tuple of (success, result, new_version)
            - success: True if update executed, False if version mismatch
            - result: Result from update_func (None if not executed)
            - new_version: Current version after operation
        
        Usage:
            ```python
            current_version = container.global_version
            # ... prepare update ...
            success, result, new_version = await container.compare_and_swap(
                current_version, update_func
            )
            if not success:
                # Version changed, retry with new state
                pass
            ```
        """
        async with self._lock:
            if self._global_version != expected_version:
                logger.debug(
                    f"Compare-and-swap version mismatch: "
                    f"expected {expected_version}, got {self._global_version}"
                )
                return False, None, self._global_version
            
            try:
                result = await update_func(self)
                self._global_version += 1
                self._last_update = time.time()
                
                logger.debug(f"Compare-and-swap succeeded, new version: {self._global_version}")
                return True, result, self._global_version
                
            except Exception as e:
                logger.error(f"Compare-and-swap update failed: {e}")
                raise
    
    def has_changed_since(self, version: int) -> bool:
        """
        Check if state has changed since a given version.
        
        Args:
            version: Version to compare against
        
        Returns:
            True if current version is newer than given version
        
        Usage:
            Used by WebSocket manager to avoid broadcasting
            unchanged state to clients.
        """
        return self._global_version > version
    
    def reset(self) -> None:
        """Reset all state (for testing or recovery)."""
        logger.warning("Resetting V3StateContainer - all state will be cleared")

        self._trading_state = None
        self._last_state_change = None
        self._trading_state_version = 0

        self._session_pnl_state = None
        self._session_updated_tickers = set()

        self._whale_state = None
        self._whale_state_version = 0

        self._component_health.clear()

        self._machine_state = None
        self._machine_state_context = ""
        self._machine_state_metadata = {}

        self._last_update = time.time()
        self._global_version = 0  # Reset global version

        logger.info("V3StateContainer reset complete")