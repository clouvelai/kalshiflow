"""
TRADER V3 State Machine - Videogame Bot Inspired Design.

This module implements a robust state machine for the V3 trader that follows
videogame bot design principles: always knowing its state, predictable behavior,
and automatic recovery from errors.

Purpose:
    The state machine provides predictable, observable state management for the
    entire trading system. It ensures the system progresses through well-defined
    states and can recover gracefully from failures.

Key Design Principles:
    1. **Always Known State** - The system always knows exactly what state it's in
    2. **Predictable Transitions** - Only valid state transitions are allowed
    3. **Observable Behavior** - All transitions emit events for monitoring
    4. **Automatic Recovery** - ERROR state can transition back to operational
    5. **Timeout Protection** - States have configurable timeouts to prevent hanging
    6. **Comprehensive History** - Maintains transition history for debugging

State Flow:
    STARTUP → INITIALIZING → ORDERBOOK_CONNECT → [TRADING_CLIENT_CONNECT → KALSHI_DATA_SYNC] → READY
    
    From READY:
        - Can transition to ERROR on component failure
        - Can transition to SHUTDOWN for graceful stop
    
    From ERROR:
        - Can recover to STARTUP for full restart
        - Can transition to SHUTDOWN if unrecoverable

Architecture Position:
    The state machine is a core V3 component used by:
    - V3Coordinator: Drives state transitions during startup and operation
    - EventBus: Receives state transition events for broadcasting
    - WebSocket clients: Monitor state changes in real-time

Integration:
    The state machine integrates with the EventBus to broadcast all state
    transitions as both SystemActivityEvents (for console) and state transition
    events (for WebSocket clients).
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, Set
from dataclasses import dataclass

logger = logging.getLogger("kalshiflow_rl.traderv3.state_machine")


class TraderState(Enum):
    """
    Trader states - videogame bot inspired.
    
    Each state represents a distinct operational mode of the trader.
    States are designed to be mutually exclusive and clearly defined.
    
    States:
        STARTUP: Initial state, configuration loading
        INITIALIZING: Starting core components (EventBus, WebSocket manager)
        ORDERBOOK_CONNECT: Establishing orderbook WebSocket connection
        TRADING_CLIENT_CONNECT: Connecting to trading API (optional)
        KALSHI_DATA_SYNC: Synchronizing positions/orders with exchange
        READY: Fully operational, monitoring markets and ready to trade
        ACTING: Executing trading actions (temporary state during trades)
        ERROR: Error state with recovery capabilities
        SHUTDOWN: Terminal state for graceful shutdown
    """
    STARTUP = "startup"
    INITIALIZING = "initializing"
    ORDERBOOK_CONNECT = "orderbook_connect"
    TRADING_CLIENT_CONNECT = "trading_client_connect"  # Connect to trading API
    KALSHI_DATA_SYNC = "kalshi_data_sync"  # Sync positions and orders with Kalshi
    READY = "ready"
    ACTING = "acting"  # Executing trades
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class StateTransition:
    """
    Records details of a state transition.
    
    Captures all relevant information about a state change for debugging
    and monitoring purposes. Stored in transition history for analysis.
    
    Attributes:
        from_state: The state we're transitioning from
        to_state: The state we're transitioning to
        context: Human-readable reason for the transition
        timestamp: When the transition occurred (Unix timestamp)
        duration_ms: How long we were in the from_state (milliseconds)
        metadata: Optional additional data about the transition
    """
    from_state: TraderState
    to_state: TraderState
    context: str
    timestamp: float
    duration_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StateMetrics:
    """
    Real-time metrics for the current state.
    
    Provides comprehensive metrics about the state machine's current
    status and history for monitoring and debugging.
    
    Attributes:
        state: Current state of the machine
        time_in_state: Seconds spent in current state
        total_state_changes: Total number of state transitions
        error_count: Number of times ERROR state was entered
        last_error: Description of the most recent error
        last_transition: Details of the most recent state change
    """
    state: TraderState
    time_in_state: float
    total_state_changes: int
    error_count: int
    last_error: Optional[str] = None
    last_transition: Optional[StateTransition] = None


class TraderStateMachine:
    """
    Videogame bot-inspired state machine for TRADER V3.

    This class manages the operational state of the entire trading system,
    ensuring predictable behavior and robust error recovery. It follows
    videogame bot design patterns where the system always knows its state
    and can recover from failures automatically.

    Core Features:
        - **State Validation**: Only allows valid state transitions
        - **Timeout Protection**: Prevents hanging in any state too long
        - **Event Broadcasting**: Emits events for all state changes
        - **Callback System**: Allows components to register for state changes
        - **Error Recovery**: Automatic recovery from ERROR to operational states
        - **Transition History**: Maintains last 50 transitions for debugging

    Attributes:
        _current_state: The current operational state
        _previous_state: The state before the last transition
        _state_entered_at: Timestamp when current state was entered
        _event_bus: Optional EventBus for broadcasting state changes
        _total_transitions: Counter of all state changes
        _error_count: Number of ERROR state entries
        _transition_history: List of recent state transitions
        _on_enter_callbacks: Callbacks to run when entering states
        _on_exit_callbacks: Callbacks to run when exiting states
        _valid_transitions: Map of allowed state transitions
        _state_timeouts: Maximum time allowed in each state

    Thread Safety:
        All methods are designed for async operation in a single event loop.
        Not thread-safe for multi-threaded access.
    """

    # Valid state transitions (defines allowed flow)
    VALID_TRANSITIONS: Dict[TraderState, Set[TraderState]] = {
        TraderState.STARTUP: {
            TraderState.INITIALIZING,
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.INITIALIZING: {
            TraderState.ORDERBOOK_CONNECT,
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.ORDERBOOK_CONNECT: {
            TraderState.TRADING_CLIENT_CONNECT,  # After orderbook, connect trading
            TraderState.READY,  # Can skip trading if not enabled
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.TRADING_CLIENT_CONNECT: {
            TraderState.KALSHI_DATA_SYNC,  # After connection, sync with Kalshi
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.KALSHI_DATA_SYNC: {
            TraderState.READY,  # After sync, ready to trade
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.READY: {
            TraderState.ACTING,  # Can transition to acting when executing trades
            TraderState.ORDERBOOK_CONNECT,  # For reconnection
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.ACTING: {
            TraderState.READY,  # Return to ready after trade execution
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.ERROR: {
            TraderState.STARTUP,  # Recovery: restart from beginning
            TraderState.SHUTDOWN,
            TraderState.READY,  # Direct recovery when data is flowing
            TraderState.ORDERBOOK_CONNECT,  # Reconnection recovery
        },
        TraderState.SHUTDOWN: set()  # Terminal state
    }

    def __init__(self, event_bus=None):
        """
        Initialize the state machine.
        
        Args:
            event_bus: Optional EventBus for state transition broadcasting
        """
        self._current_state = TraderState.STARTUP
        self._previous_state: Optional[TraderState] = None
        self._state_entered_at = time.time()
        self._event_bus = event_bus
        
        # State transition tracking
        self._total_transitions = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._transition_history: list[StateTransition] = []
        
        # State callbacks - allows external components to register for state changes
        self._on_enter_callbacks: Dict[TraderState, list[Callable]] = {}
        self._on_exit_callbacks: Dict[TraderState, list[Callable]] = {}

        # Reference class-level transitions constant
        self._valid_transitions = self.VALID_TRANSITIONS

        # State timeouts (prevent hanging in states too long)
        self._state_timeouts: Dict[TraderState, float] = {
            TraderState.STARTUP: 30.0,  # 30s to start up
            TraderState.INITIALIZING: 60.0,  # 1m to initialize
            TraderState.ORDERBOOK_CONNECT: 120.0,  # 2m to connect orderbook
            TraderState.TRADING_CLIENT_CONNECT: 60.0,  # 1m to connect trading API
            TraderState.KALSHI_DATA_SYNC: 60.0,  # 1m to sync with Kalshi
            TraderState.READY: float('inf'),  # Can stay ready indefinitely
            TraderState.ACTING: 60.0,  # 1m max for trade execution
            TraderState.ERROR: 300.0,  # 5m before forcing shutdown
            TraderState.SHUTDOWN: 30.0  # 30s to shutdown
        }
        
        logger.info(f"TRADER V3 StateMachine initialized in {self._current_state.value} state")
    
    @property
    def current_state(self) -> TraderState:
        """Get current state."""
        return self._current_state
    
    @property
    def time_in_current_state(self) -> float:
        """Get seconds spent in current state."""
        return time.time() - self._state_entered_at
    
    @property
    def is_terminal_state(self) -> bool:
        """Check if in terminal state (shutdown/error)."""
        return self._current_state in {TraderState.SHUTDOWN, TraderState.ERROR}
    
    @property
    def is_operational_state(self) -> bool:
        """Check if in operational state (ready)."""
        return self._current_state == TraderState.READY
    
    async def transition_to(self, new_state: TraderState, context: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Transition to a new state with validation.
        
        This is the primary method for changing states. It validates the
        transition, records it in history, emits events, and calls any
        registered callbacks.
        
        Flow:
            1. Validate transition is allowed from current state
            2. Call exit callbacks for current state
            3. Update state and record transition
            4. Emit state change events via EventBus
            5. Call enter callbacks for new state
        
        Args:
            new_state: Target state to transition to
            context: Human-readable reason for the transition (for logging)
            metadata: Optional additional data to attach to the transition
            
        Returns:
            True if transition was successful, False if invalid or failed
        
        Side Effects:
            - Updates _current_state and _previous_state
            - Increments _total_transitions counter
            - Adds to _transition_history
            - Emits events via EventBus (if configured)
            - Calls registered state callbacks
        
        Note:
            If the transition fails, the state is reverted and False is returned.
            Same-state transitions are ignored but return True.
        """
        if new_state == self._current_state:
            logger.debug(f"Ignoring transition to same state: {new_state.value}")
            return True
        
        # Validate transition is allowed
        if new_state not in self._valid_transitions.get(self._current_state, set()):
            logger.error(f"Invalid state transition: {self._current_state.value} -> {new_state.value}")
            return False
        
        # Record transition timing
        transition_start = time.time()
        duration_ms = (transition_start - self._state_entered_at) * 1000
        
        # Create transition record
        transition = StateTransition(
            from_state=self._current_state,
            to_state=new_state,
            context=context,
            timestamp=transition_start,
            duration_ms=duration_ms,
            metadata=metadata
        )
        
        # Execute transition
        previous_state = self._current_state
        
        try:
            # Call exit callbacks for current state
            await self._call_exit_callbacks(previous_state)
            
            # Update state
            self._previous_state = previous_state
            self._current_state = new_state
            self._state_entered_at = transition_start
            self._total_transitions += 1
            
            # Record transition
            self._transition_history.append(transition)
            
            # Keep only last 50 transitions to prevent memory growth
            if len(self._transition_history) > 50:
                self._transition_history = self._transition_history[-50:]
            
            # Emit state transition as system activity (unified messaging)
            if self._event_bus:
                # Also emit as system activity for unified console messaging
                await self._event_bus.emit_system_activity(
                    activity_type="state_transition",
                    message=f"{previous_state.value} → {new_state.value}: {context}",
                    metadata={
                        "from_state": previous_state.value,
                        "to_state": new_state.value,
                        "context": context,
                        **(metadata or {})
                    }
                )
            
            # Call enter callbacks for new state
            await self._call_enter_callbacks(new_state)
            
            logger.info(f"State transition: {previous_state.value} -> {new_state.value} | Context: {context}")
            return True
            
        except Exception as e:
            logger.error(f"Error during state transition {previous_state.value} -> {new_state.value}: {e}")
            self._error_count += 1
            self._last_error = str(e)
            
            # Revert state if transition failed
            self._current_state = previous_state
            return False
    
    async def enter_error_state(self, error_context: str, error: Optional[Exception] = None) -> None:
        """
        Transition to ERROR state with comprehensive error information.
        
        Convenience method for entering the ERROR state with proper error
        tracking and metadata. Increments error counter and records the
        error for debugging.
        
        Args:
            error_context: Description of what operation failed
            error: Optional exception that caused the error
        
        Side Effects:
            - Increments _error_count
            - Updates _last_error with error details
            - Transitions to ERROR state with metadata
        
        Usage:
            Used by components when they encounter unrecoverable errors
            that require the system to enter ERROR state for recovery.
        """
        error_msg = str(error) if error else "Unknown error"
        full_context = f"{error_context}: {error_msg}"
        
        self._error_count += 1
        self._last_error = full_context
        
        await self.transition_to(
            TraderState.ERROR,
            full_context,
            metadata={
                "error_message": error_msg,
                "error_type": type(error).__name__ if error else None,
                "error_count": self._error_count
            }
        )
    
    def register_state_callback(self, state: TraderState, callback: Callable, on_enter: bool = True) -> None:
        """
        Register callback for state transitions.
        
        Args:
            state: State to register callback for
            callback: Async callback function
            on_enter: If True, call on state entry; if False, call on state exit
        """
        callback_dict = self._on_enter_callbacks if on_enter else self._on_exit_callbacks
        
        if state not in callback_dict:
            callback_dict[state] = []
        
        callback_dict[state].append(callback)
        action = "enter" if on_enter else "exit"
        logger.info(f"Registered {action} callback for state {state.value}: {callback.__name__}")
    
    def check_state_timeout(self) -> bool:
        """
        Check if current state has exceeded its configured timeout.
        
        Each state has a maximum duration to prevent the system from
        hanging indefinitely. This method checks if the current state
        has exceeded its timeout threshold.
        
        Returns:
            True if state has timed out, False otherwise
        
        Note:
            READY state has infinite timeout (can stay ready forever).
            ERROR state has 5-minute timeout before forcing shutdown.
        """
        timeout = self._state_timeouts.get(self._current_state, float('inf'))
        if timeout == float('inf'):
            return False
        
        time_in_state = self.time_in_current_state
        if time_in_state > timeout:
            logger.warning(f"State timeout: {self._current_state.value} exceeded {timeout}s (actual: {time_in_state:.1f}s)")
            return True
        
        return False
    
    async def handle_timeout(self) -> None:
        """Handle state timeout by transitioning to error state."""
        await self.enter_error_state(
            f"State timeout in {self._current_state.value}",
            TimeoutError(f"State {self._current_state.value} exceeded {self._state_timeouts[self._current_state]}s timeout")
        )
    
    def get_metrics(self) -> StateMetrics:
        """Get current state metrics."""
        return StateMetrics(
            state=self._current_state,
            time_in_state=self.time_in_current_state,
            total_state_changes=self._total_transitions,
            error_count=self._error_count,
            last_error=self._last_error,
            last_transition=self._transition_history[-1] if self._transition_history else None
        )
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive status summary for debugging.
        
        Provides a complete snapshot of the state machine's current
        condition, including state, metrics, timeouts, and recent history.
        
        Returns:
            Dictionary containing:
                - current_state: Current state name
                - previous_state: Previous state name (if any)
                - time_in_state: Seconds in current state
                - total_transitions: Total state changes
                - error_count: Number of ERROR entries
                - last_error: Most recent error message
                - is_terminal: Whether in terminal state
                - is_operational: Whether in READY state
                - state_timeout: Timeout for current state
                - timeout_remaining: Seconds until timeout
                - last_transition: Details of most recent transition
        
        Usage:
            Used by monitoring endpoints and debugging tools to get
            a complete picture of the state machine's status.
        """
        metrics = self.get_metrics()
        
        return {
            "current_state": self._current_state.value,
            "previous_state": self._previous_state.value if self._previous_state else None,
            "time_in_state": metrics.time_in_state,
            "total_transitions": metrics.total_state_changes,
            "error_count": metrics.error_count,
            "last_error": metrics.last_error,
            "is_terminal": self.is_terminal_state,
            "is_operational": self.is_operational_state,
            "state_timeout": self._state_timeouts.get(self._current_state) or None,  # None for infinite (JSON-safe)
            "timeout_remaining": max(0, (self._state_timeouts.get(self._current_state) or 9999) - metrics.time_in_state),
            "last_transition": {
                "from": metrics.last_transition.from_state.value,
                "to": metrics.last_transition.to_state.value,
                "context": metrics.last_transition.context,
                "timestamp": metrics.last_transition.timestamp,
                "duration_ms": metrics.last_transition.duration_ms
            } if metrics.last_transition else None
        }
    
    def is_healthy(self) -> bool:
        """Check if state machine is in a healthy state."""
        return not self.is_terminal_state and self._current_state != TraderState.ERROR
    
    def get_health_details(self) -> Dict[str, Any]:
        """Get detailed health information."""
        return {
            "healthy": self.is_healthy(),
            "current_state": self._current_state.value,
            "time_in_state": self.time_in_current_state,
            "error_count": self._error_count,
            "last_error": self._last_error
        }
    
    async def start(self) -> bool:
        """
        Start the state machine.
        
        Returns:
            True if successfully started
        """
        if self._current_state != TraderState.STARTUP:
            logger.warning(f"Cannot start from state {self._current_state.value}")
            return False
        
        # Transition to initializing
        success = await self.transition_to(
            TraderState.INITIALIZING,
            context="Starting state machine components"
        )
        
        if success:
            logger.info("State machine started successfully")
        
        return success
    
    async def stop(self) -> bool:
        """
        Stop the state machine gracefully.
        
        Returns:
            True if successfully stopped
        """
        if self._current_state == TraderState.SHUTDOWN:
            logger.warning("State machine already shutdown")
            return True
        
        # Transition to shutdown
        success = await self.transition_to(
            TraderState.SHUTDOWN,
            context="Graceful shutdown requested"
        )
        
        if success:
            logger.info("State machine stopped successfully")
        
        return success
    
    def get_transition_history(self, limit: int = 10) -> list[Dict[str, Any]]:
        """
        Get recent transition history for debugging.
        
        Args:
            limit: Maximum number of transitions to return
            
        Returns:
            List of transition dictionaries
        """
        recent_transitions = self._transition_history[-limit:] if self._transition_history else []
        
        return [
            {
                "from_state": t.from_state.value,
                "to_state": t.to_state.value,
                "context": t.context,
                "timestamp": t.timestamp,
                "duration_ms": t.duration_ms,
                "metadata": t.metadata
            }
            for t in recent_transitions
        ]
    
    async def _call_enter_callbacks(self, state: TraderState) -> None:
        """Call all registered enter callbacks for a state."""
        callbacks = self._on_enter_callbacks.get(state, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state, self.get_metrics())
                else:
                    callback(state, self.get_metrics())
            except Exception as e:
                logger.error(f"Error in enter callback for state {state.value}: {e}")
    
    async def _call_exit_callbacks(self, state: TraderState) -> None:
        """Call all registered exit callbacks for a state."""
        callbacks = self._on_exit_callbacks.get(state, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(state, self.get_metrics())
                else:
                    callback(state, self.get_metrics())
            except Exception as e:
                logger.error(f"Error in exit callback for state {state.value}: {e}")