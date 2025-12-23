"""
TRADER V3 State Machine - Videogame Bot Inspired Design.

A clean, predictable state machine that always knows what it's doing.
Designed like a videogame bot: clear states, predictable transitions, 
comprehensive status reporting, and bulletproof error recovery.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, Callable, Set
from dataclasses import dataclass

logger = logging.getLogger("kalshiflow_rl.traderv3.state_machine")


class TraderState(Enum):
    """Trader states - videogame bot inspired."""
    STARTUP = "startup"
    INITIALIZING = "initializing"
    ORDERBOOK_CONNECT = "orderbook_connect"
    READY = "ready"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class StateTransition:
    """State transition data."""
    from_state: TraderState
    to_state: TraderState
    context: str
    timestamp: float
    duration_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StateMetrics:
    """Metrics for current state."""
    state: TraderState
    time_in_state: float
    total_state_changes: int
    error_count: int
    last_error: Optional[str] = None
    last_transition: Optional[StateTransition] = None


class TraderStateMachine:
    """
    Videogame bot-inspired state machine for TRADER V3.
    
    Features:
    - Always knows what state it's in
    - Clear, predictable state transitions
    - Comprehensive status reporting
    - Error recovery with context
    - State timeouts to prevent hanging
    - Event emission for real-time monitoring
    """
    
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
        
        # Valid state transitions (defines allowed flow)
        self._valid_transitions: Dict[TraderState, Set[TraderState]] = {
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
                TraderState.READY,
                TraderState.ERROR,
                TraderState.SHUTDOWN
            },
            TraderState.READY: {
                TraderState.ORDERBOOK_CONNECT,  # For reconnection
                TraderState.ERROR,
                TraderState.SHUTDOWN
            },
            TraderState.ERROR: {
                TraderState.STARTUP,  # Recovery: restart from beginning
                TraderState.SHUTDOWN
            },
            TraderState.SHUTDOWN: set()  # Terminal state
        }
        
        # State timeouts (prevent hanging in states too long)
        self._state_timeouts: Dict[TraderState, float] = {
            TraderState.STARTUP: 30.0,  # 30s to start up
            TraderState.INITIALIZING: 60.0,  # 1m to initialize
            TraderState.ORDERBOOK_CONNECT: 120.0,  # 2m to connect orderbook
            TraderState.READY: float('inf'),  # Can stay ready indefinitely
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
        
        Args:
            new_state: Target state
            context: Human-readable reason for transition
            metadata: Optional additional data
            
        Returns:
            True if transition was successful, False otherwise
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
            
            # Emit state transition event
            if self._event_bus:
                await self._event_bus.emit_state_transition(
                    from_state=previous_state.value,
                    to_state=new_state.value,
                    context=context,
                    metadata=metadata
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
        Transition to error state with error context.
        
        Args:
            error_context: Description of what went wrong
            error: Optional exception that caused the error
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
        Check if current state has exceeded timeout.
        
        Returns:
            True if state has timed out
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
        """Get comprehensive status summary for debugging."""
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
            "state_timeout": self._state_timeouts.get(self._current_state, float('inf')),
            "timeout_remaining": max(0, self._state_timeouts.get(self._current_state, float('inf')) - metrics.time_in_state),
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


# State machine convenience functions

async def create_state_machine(event_bus=None) -> TraderStateMachine:
    """
    Create and initialize a state machine.
    
    Args:
        event_bus: Optional EventBus for state broadcasting
        
    Returns:
        Initialized TraderStateMachine
    """
    state_machine = TraderStateMachine(event_bus=event_bus)
    logger.info("TRADER V3 state machine created and ready")
    return state_machine


def get_state_description(state: TraderState) -> str:
    """
    Get human-readable description of a state.
    
    Args:
        state: State to describe
        
    Returns:
        Human-readable state description
    """
    descriptions = {
        TraderState.STARTUP: "System starting up, loading configuration",
        TraderState.INITIALIZING: "Initializing core components (event bus, WebSocket manager)",
        TraderState.ORDERBOOK_CONNECT: "Connecting to Kalshi, starting orderbook client",
        TraderState.READY: "System ready, orderbook connected, monitoring markets",
        TraderState.ERROR: "System error, attempting recovery or requiring intervention",
        TraderState.SHUTDOWN: "System shutting down gracefully"
    }
    
    return descriptions.get(state, f"Unknown state: {state.value}")


def get_next_states(current_state: TraderState) -> Set[TraderState]:
    """
    Get valid next states from current state.
    
    Args:
        current_state: Current state
        
    Returns:
        Set of valid next states
    """
    valid_transitions = {
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
            TraderState.READY,
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.READY: {
            TraderState.ORDERBOOK_CONNECT,
            TraderState.ERROR,
            TraderState.SHUTDOWN
        },
        TraderState.ERROR: {
            TraderState.STARTUP,
            TraderState.SHUTDOWN
        },
        TraderState.SHUTDOWN: set()
    }
    
    return valid_transitions.get(current_state, set())