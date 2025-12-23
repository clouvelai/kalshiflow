"""
State Machine - Created for TRADER 2.0

Simple videogame bot style state machine: IDLE → CALIBRATING → READY ↔ ACTING → ERROR
Focused on clear state awareness and self-recovery patterns.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, Set, List
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger("kalshiflow_rl.trading.state_machine")


class TraderState(Enum):
    """Trader state machine states."""
    IDLE = "idle"           # Not started
    CALIBRATING = "calibrating"  # Running system_check → sync_data
    READY = "ready"         # Calibrated and waiting for actions
    ACTING = "acting"       # Processing an action
    ERROR = "error"         # Self-recovery mode
    PAUSED = "paused"       # Paused (orderbook failure) but can do recovery operations
    LOW_CASH = "low_cash"   # Insufficient cash balance for trading
    RECOVER_CASH = "recover_cash"  # Auto-closing positions to recover cash


@dataclass
class StateTransition:
    """Represents a state transition."""
    from_state: TraderState
    to_state: TraderState
    condition: str
    timestamp: float
    reason: str = ""


class StateMachineError(Exception):
    """State machine specific errors."""
    pass


class TraderStateMachine:
    """
    Simple videogame bot style state machine.
    
    States and transitions:
    - IDLE → CALIBRATING (on start)
    - CALIBRATING → READY (on success) or ERROR (on fail)
    - READY ↔ ACTING (on actions)
    - ANY → ERROR → CALIBRATING (self-recovery)
    - READY/ACTING → PAUSED (on orderbook failure)
    - PAUSED → CALIBRATING (on recovery attempt)
    """
    
    def __init__(self, status_logger: Optional['StatusLogger'] = None, websocket_manager=None):
        """
        Initialize state machine.
        
        Args:
            status_logger: Optional StatusLogger for transition tracking
            websocket_manager: Global WebSocketManager for broadcasting (optional)
        """
        self.status_logger = status_logger
        self.websocket_manager = websocket_manager
        
        # Current state
        self.current_state = TraderState.IDLE
        self.state_entered_at = time.time()
        
        # State transition history
        self.transition_history: List[StateTransition] = []
        
        # Valid state transitions
        self.valid_transitions: Dict[TraderState, Set[TraderState]] = {
            TraderState.IDLE: {TraderState.CALIBRATING, TraderState.ERROR},
            TraderState.CALIBRATING: {TraderState.READY, TraderState.LOW_CASH, TraderState.ERROR, TraderState.PAUSED},
            TraderState.READY: {TraderState.ACTING, TraderState.LOW_CASH, TraderState.ERROR, TraderState.PAUSED, TraderState.CALIBRATING},
            TraderState.ACTING: {TraderState.READY, TraderState.LOW_CASH, TraderState.ERROR, TraderState.PAUSED},
            TraderState.ERROR: {TraderState.CALIBRATING, TraderState.IDLE},
            TraderState.PAUSED: {TraderState.CALIBRATING, TraderState.ERROR, TraderState.IDLE},
            TraderState.LOW_CASH: {TraderState.RECOVER_CASH, TraderState.ERROR, TraderState.CALIBRATING},
            TraderState.RECOVER_CASH: {TraderState.CALIBRATING, TraderState.ERROR, TraderState.LOW_CASH}
        }
        
        # State callbacks
        self.state_enter_callbacks: Dict[TraderState, List[Callable]] = {
            state: [] for state in TraderState
        }
        self.state_exit_callbacks: Dict[TraderState, List[Callable]] = {
            state: [] for state in TraderState
        }
        
        # Error recovery tracking
        self.error_count = 0
        self.last_error: Optional[str] = None
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        logger.info(f"TraderStateMachine initialized in state: {self.current_state.value}")
    
    def can_transition_to(self, target_state: TraderState) -> bool:
        """
        Check if transition to target state is valid.
        
        Args:
            target_state: State to transition to
            
        Returns:
            True if transition is allowed
        """
        return target_state in self.valid_transitions.get(self.current_state, set())
    
    async def transition_to(self, target_state: TraderState, reason: str = "") -> bool:
        """
        Transition to target state.
        
        Args:
            target_state: State to transition to
            reason: Reason for transition
            
        Returns:
            True if transition succeeded
            
        Raises:
            StateMachineError: If transition is invalid
        """
        try:
            # Check if transition is valid
            if not self.can_transition_to(target_state):
                valid_states = list(self.valid_transitions.get(self.current_state, set()))
                error_msg = f"Invalid transition from {self.current_state.value} to {target_state.value}. Valid transitions: {[s.value for s in valid_states]}"
                logger.error(error_msg)
                raise StateMachineError(error_msg)
            
            # Record transition
            old_state = self.current_state
            transition_time = time.time()
            time_in_state = transition_time - self.state_entered_at
            
            transition = StateTransition(
                from_state=old_state,
                to_state=target_state,
                condition="manual",
                timestamp=transition_time,
                reason=reason
            )
            
            # Call exit callbacks for current state
            for callback in self.state_exit_callbacks[old_state]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(old_state, target_state)
                    else:
                        callback(old_state, target_state)
                except Exception as e:
                    logger.error(f"Error in state exit callback: {e}")
            
            # Update state
            self.current_state = target_state
            self.state_entered_at = transition_time
            self.transition_history.append(transition)
            
            # Reset recovery tracking on successful non-error transitions
            if target_state != TraderState.ERROR:
                self.recovery_attempts = 0
            
            # Log transition
            if self.status_logger:
                await self.status_logger.log_state_transition(
                    old_state.value, 
                    target_state.value, 
                    reason
                )
            
            logger.info(f"State transition: {old_state.value} → {target_state.value} "
                       f"(reason: {reason}, time in {old_state.value}: {time_in_state:.2f}s)")
            
            # Call enter callbacks for new state
            for callback in self.state_enter_callbacks[target_state]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(old_state, target_state)
                    else:
                        callback(old_state, target_state)
                except Exception as e:
                    logger.error(f"Error in state enter callback: {e}")
            
            return True
            
        except StateMachineError:
            raise
        except Exception as e:
            logger.error(f"Error during state transition: {e}")
            return False
    
    async def handle_error(self, error: str, auto_recovery: bool = True) -> bool:
        """
        Handle error and optionally attempt recovery.
        
        Args:
            error: Error description
            auto_recovery: Whether to automatically attempt recovery
            
        Returns:
            True if error handled successfully
        """
        try:
            self.error_count += 1
            self.last_error = error
            
            logger.error(f"Handling error ({self.error_count}): {error}")
            
            # Transition to ERROR state
            await self.transition_to(TraderState.ERROR, f"error: {error}")
            
            if auto_recovery and self.recovery_attempts < self.max_recovery_attempts:
                self.recovery_attempts += 1
                
                # Wait a bit before recovery attempt
                await asyncio.sleep(1.0 * self.recovery_attempts)  # Exponential backoff
                
                # Attempt recovery by going back to calibration
                logger.info(f"Attempting recovery {self.recovery_attempts}/{self.max_recovery_attempts}")
                return await self.transition_to(TraderState.CALIBRATING, f"recovery_attempt_{self.recovery_attempts}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling error (!): {e}")
            return False
    
    async def handle_orderbook_failure(self) -> bool:
        """
        Handle orderbook failure by transitioning to PAUSED state.
        
        In PAUSED state, trading is blocked but recovery operations 
        (cancelling orders, closing positions) are still allowed.
        
        Returns:
            True if transition succeeded
        """
        try:
            if self.current_state in [TraderState.READY, TraderState.ACTING]:
                logger.warning("Orderbook failure detected - pausing trading")
                return await self.transition_to(TraderState.PAUSED, "orderbook_failure")
            else:
                logger.warning(f"Orderbook failure in state {self.current_state.value} - no action needed")
                return True
                
        except Exception as e:
            logger.error(f"Error handling orderbook failure: {e}")
            return False
    
    async def attempt_recovery_from_pause(self) -> bool:
        """
        Attempt recovery from PAUSED state by recalibrating.
        
        Returns:
            True if recovery initiated
        """
        try:
            if self.current_state == TraderState.PAUSED:
                logger.info("Attempting recovery from PAUSED state")
                return await self.transition_to(TraderState.CALIBRATING, "recovery_from_pause")
            else:
                logger.warning(f"Cannot recover from pause - not in PAUSED state (current: {self.current_state.value})")
                return False
                
        except Exception as e:
            logger.error(f"Error during pause recovery: {e}")
            return False
    
    def is_operational(self) -> bool:
        """Check if system is in an operational state (READY or ACTING)."""
        return self.current_state in [TraderState.READY, TraderState.ACTING]
    
    def is_trading_allowed(self) -> bool:
        """Check if trading actions are allowed."""
        return self.current_state in [TraderState.READY, TraderState.ACTING]
    
    def is_recovery_operations_allowed(self) -> bool:
        """Check if recovery operations (cancel orders, close positions) are allowed."""
        # Recovery operations allowed in most states except IDLE
        return self.current_state not in [TraderState.IDLE]
    
    def is_cash_recovery_required(self) -> bool:
        """Check if system is in cash recovery mode."""
        return self.current_state in [TraderState.LOW_CASH, TraderState.RECOVER_CASH]
    
    def can_start_calibration(self) -> bool:
        """Check if calibration can be started."""
        return self.current_state in [TraderState.IDLE, TraderState.ERROR, TraderState.PAUSED]
    
    def can_process_action(self) -> bool:
        """Check if actions can be processed."""
        return self.current_state == TraderState.READY
    
    def register_state_enter_callback(self, state: TraderState, callback: Callable) -> None:
        """
        Register callback for when entering a state.
        
        Args:
            state: State to register callback for
            callback: Callback function (can be async)
        """
        self.state_enter_callbacks[state].append(callback)
    
    def register_state_exit_callback(self, state: TraderState, callback: Callable) -> None:
        """
        Register callback for when exiting a state.
        
        Args:
            state: State to register callback for
            callback: Callback function (can be async)
        """
        self.state_exit_callbacks[state].append(callback)
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get current state information.
        
        Returns:
            Dict with state information
        """
        current_time = time.time()
        time_in_state = current_time - self.state_entered_at
        
        # Get last transition
        last_transition = None
        if self.transition_history:
            last_trans = self.transition_history[-1]
            last_transition = {
                "from_state": last_trans.from_state.value,
                "to_state": last_trans.to_state.value,
                "reason": last_trans.reason,
                "timestamp": last_trans.timestamp
            }
        
        return {
            "current_state": self.current_state.value,
            "time_in_state": time_in_state,
            "state_entered_at": self.state_entered_at,
            "last_transition": last_transition,
            "error_count": self.error_count,
            "last_error": self.last_error,
            "recovery_attempts": self.recovery_attempts,
            "max_recovery_attempts": self.max_recovery_attempts,
            "is_operational": self.is_operational(),
            "is_trading_allowed": self.is_trading_allowed(),
            "can_start_calibration": self.can_start_calibration(),
            "can_process_action": self.can_process_action()
        }
    
    def get_transition_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent transition history.
        
        Args:
            limit: Maximum number of transitions to return
            
        Returns:
            List of transition dictionaries
        """
        recent_transitions = self.transition_history[-limit:] if limit else self.transition_history
        
        return [
            {
                "from_state": trans.from_state.value,
                "to_state": trans.to_state.value,
                "condition": trans.condition,
                "reason": trans.reason,
                "timestamp": trans.timestamp
            }
            for trans in recent_transitions
        ]
    
    def reset_to_idle(self) -> None:
        """Reset state machine to IDLE (for testing or emergency stop)."""
        old_state = self.current_state
        self.current_state = TraderState.IDLE
        self.state_entered_at = time.time()
        self.recovery_attempts = 0
        
        logger.warning(f"State machine reset: {old_state.value} → {TraderState.IDLE.value}")
    
    async def trigger_cash_recovery(self, reason: str = "insufficient_cash") -> bool:
        """
        Trigger cash recovery by transitioning to LOW_CASH state.
        
        Args:
            reason: Reason for cash recovery (for logging)
            
        Returns:
            True if transition succeeded
        """
        try:
            # Can transition to LOW_CASH from CALIBRATING, READY, ACTING
            if self.current_state in [TraderState.CALIBRATING, TraderState.READY, TraderState.ACTING]:
                logger.warning(f"Cash recovery triggered: {reason}")
                return await self.transition_to(TraderState.LOW_CASH, f"cash_recovery_{reason}")
            else:
                logger.warning(f"Cannot trigger cash recovery from state {self.current_state.value}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering cash recovery: {e}")
            return False
    
    async def start_position_liquidation(self) -> bool:
        """
        Start position liquidation by transitioning to RECOVER_CASH state.
        
        Returns:
            True if transition succeeded
        """
        try:
            if self.current_state == TraderState.LOW_CASH:
                logger.info("Starting position liquidation to recover cash")
                return await self.transition_to(TraderState.RECOVER_CASH, "liquidate_positions")
            else:
                logger.warning(f"Cannot start liquidation from state {self.current_state.value}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting liquidation: {e}")
            return False
    
    async def complete_cash_recovery(self) -> bool:
        """
        Complete cash recovery and return to calibration.
        
        Returns:
            True if transition succeeded
        """
        try:
            if self.current_state == TraderState.RECOVER_CASH:
                logger.info("Cash recovery completed - returning to calibration")
                return await self.transition_to(TraderState.CALIBRATING, "cash_recovery_complete")
            else:
                logger.warning(f"Cannot complete cash recovery from state {self.current_state.value}")
                return False
                
        except Exception as e:
            logger.error(f"Error completing cash recovery: {e}")
            return False

    async def emergency_stop(self) -> bool:
        """
        Emergency stop - transition to ERROR state.
        
        Returns:
            True if emergency stop succeeded
        """
        try:
            logger.critical("Emergency stop initiated")
            return await self.transition_to(TraderState.ERROR, "emergency_stop")
        except Exception as e:
            logger.critical(f"Error during emergency stop: {e}")
            # Force reset if transition fails
            self.reset_to_idle()
            return False