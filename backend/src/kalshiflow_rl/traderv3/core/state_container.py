"""
V3 State Container - Central state management for TRADER V3.

Organizes and provides clean access to all state types:
- Trading state (positions, orders, balance)
- Component health metrics
- State machine reference
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..state.trader_state import TraderState, StateChange
from .state_machine import TraderState as V3State  # State machine states

logger = logging.getLogger("kalshiflow_rl.traderv3.core.state_container")


@dataclass
class ComponentHealth:
    """Health status for a single component."""
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


class V3StateContainer:
    """
    Central container for all V3 state management.
    
    This container:
    - Stores trading state from Kalshi syncs
    - Tracks component health metrics
    - Provides state machine reference
    - Manages state versioning for change detection
    
    NOT a state machine itself - just organized storage.
    """
    
    def __init__(self):
        """Initialize state container."""
        # Trading state - data from Kalshi
        self._trading_state: Optional[TraderState] = None
        self._last_state_change: Optional[StateChange] = None
        self._trading_state_version = 0  # Increment on each update for change detection
        
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
        
        logger.info("V3StateContainer initialized")
    
    # ======== Trading State Management ========
    
    def update_trading_state(self, state: TraderState, changes: Optional[StateChange] = None) -> bool:
        """
        Update trading state from Kalshi sync.
        
        Args:
            state: New trader state from Kalshi
            changes: Optional changes from previous state
            
        Returns:
            True if state changed, False if identical
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
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """
        Get trading state summary for broadcasting.
        
        Returns clean data suitable for WebSocket messages.
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
            "open_orders": len(state.orders)  # Count of orders
        }
        
        # Add changes if available
        if self._last_state_change:
            summary["changes"] = {
                "balance": self._last_state_change.balance_change,
                "portfolio_value": self._last_state_change.portfolio_value_change,
                "positions": self._last_state_change.position_count_change,
                "orders": self._last_state_change.order_count_change
            }
        
        return summary
    
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
        
        Args:
            name: Component name
            healthy: Whether component is healthy
            details: Optional health details
            error: Optional error message
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
        
        return {
            "system_healthy": self.is_system_healthy(),
            "components": components,
            "component_count": len(self._component_health),
            "unhealthy_count": sum(1 for c in self._component_health.values() if not c.healthy)
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
        Get complete state snapshot.
        
        Returns everything - for debugging/inspection.
        """
        return {
            "trading": self.get_trading_summary(),
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
                "trading_version": self._trading_state_version
            }
        }
    
    def reset(self) -> None:
        """Reset all state (for testing or recovery)."""
        logger.warning("Resetting V3StateContainer - all state will be cleared")
        
        self._trading_state = None
        self._last_state_change = None
        self._trading_state_version = 0
        
        self._component_health.clear()
        
        self._machine_state = None
        self._machine_state_context = ""
        self._machine_state_metadata = {}
        
        self._last_update = time.time()
        
        logger.info("V3StateContainer reset complete")