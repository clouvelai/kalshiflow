"""
StatusLogger - Created for TRADER 2.0

Clean status tracking for state machine and service health with debugging value.
Designed to replace and improve upon status tracking scattered throughout OrderManager.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Deque
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.trading.services.status_logger")


class TraderState(Enum):
    """Trader state machine states."""
    IDLE = "idle"
    CALIBRATING = "calibrating"
    READY = "ready"
    ACTING = "acting"
    ERROR = "error"
    PAUSED = "paused"  # For orderbook failures while allowing recovery operations


@dataclass
class StatusEntry:
    """Individual status log entry."""
    timestamp: float
    entry_type: str  # "state_transition", "service_status", "action_result", "calibration_step"
    details: Dict[str, Any]
    level: str = "info"  # "info", "warning", "error"
    
    @property
    def age_seconds(self) -> float:
        """Time since this entry in seconds."""
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "entry_type": self.entry_type,
            "details": self.details,
            "level": self.level,
            "age_seconds": self.age_seconds
        }


@dataclass
class ServiceStatus:
    """Track individual service health."""
    service_name: str
    status: str  # "healthy", "degraded", "error", "inactive"
    last_activity: float
    activity_details: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None
    
    def update_activity(self, status: str, details: Dict[str, Any] = None) -> None:
        """Update service activity."""
        self.status = status
        self.last_activity = time.time()
        if details:
            self.activity_details.update(details)
    
    def record_error(self, error: str) -> None:
        """Record service error."""
        self.status = "error"
        self.error_count += 1
        self.last_error = error
        self.last_activity = time.time()
    
    @property
    def time_since_activity(self) -> float:
        """Time since last activity in seconds."""
        return time.time() - self.last_activity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for status reporting."""
        return {
            "service_name": self.service_name,
            "status": self.status,
            "last_activity": self.last_activity,
            "time_since_activity": self.time_since_activity,
            "activity_details": self.activity_details,
            "error_count": self.error_count,
            "last_error": self.last_error
        }


class StatusLogger:
    """
    Clean status tracking for state machine and service health.
    
    Provides structured logging with copy-paste debugging value while maintaining
    clean separation between state machine transitions, service health, and activities.
    """
    
    def __init__(self, websocket_manager=None, max_history_size: int = 100):
        """
        Initialize StatusLogger.
        
        Args:
            websocket_manager: Global WebSocketManager for broadcasting (optional)
            max_history_size: Maximum number of status entries to keep in history
        """
        self.websocket_manager = websocket_manager
        self.max_history_size = max_history_size
        
        # State machine tracking
        self.current_state = TraderState.IDLE
        self.state_entered_at = time.time()
        self.last_state_transition: Optional[Dict[str, Any]] = None
        
        # Service status tracking
        self.services: Dict[str, ServiceStatus] = {}
        
        # Status entry history
        self.status_history: Deque[StatusEntry] = deque(maxlen=max_history_size)
        
        # Calibration step tracking
        self.calibration_steps: List[Dict[str, Any]] = []
        
        # Error tracking
        self.recent_errors: Deque[Dict[str, Any]] = deque(maxlen=20)
        
        logger.info("StatusLogger initialized")
    
    async def log_state_transition(self, from_state: str, to_state: str, reason: str = None) -> None:
        """
        Track state machine transitions.
        
        Args:
            from_state: Previous state
            to_state: New state
            reason: Optional reason for transition
        """
        try:
            transition_time = time.time()
            time_in_previous_state = transition_time - self.state_entered_at
            
            # Update current state
            old_state = self.current_state
            self.current_state = TraderState(to_state.lower())
            self.state_entered_at = transition_time
            
            # Record transition
            self.last_state_transition = {
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
                "timestamp": transition_time,
                "time_in_previous_state": time_in_previous_state
            }
            
            # Add to history
            entry = StatusEntry(
                timestamp=transition_time,
                entry_type="state_transition",
                details={
                    "from_state": from_state,
                    "to_state": to_state,
                    "reason": reason,
                    "time_in_previous_state": time_in_previous_state
                },
                level="info"
            )
            self.status_history.append(entry)
            
            logger.info(f"State transition: {from_state} → {to_state} (reason: {reason}, "
                       f"time in {from_state}: {time_in_previous_state:.2f}s)")
            
            # Broadcast dedicated state transition to WebSocket clients
            await self._broadcast_state_transition(from_state, to_state, reason, time_in_previous_state)
            
        except Exception as e:
            logger.error(f"Error logging state transition: {e}")
    
    async def log_calibration_step(self, step: str, status: str, duration: float = None, error: str = None) -> None:
        """
        Track calibration step progress.
        
        Args:
            step: Calibration step name (e.g., "system_check", "sync_positions")
            status: Step status ("starting", "complete", "error")
            duration: Optional step duration in seconds
            error: Optional error message for failed steps
        """
        try:
            step_entry = {
                "step": step,
                "status": status,
                "timestamp": time.time(),
                "duration": duration,
                "error": error
            }
            
            # Update or add calibration step
            existing_step = None
            for i, cal_step in enumerate(self.calibration_steps):
                if cal_step["step"] == step:
                    existing_step = i
                    break
            
            if existing_step is not None:
                self.calibration_steps[existing_step].update(step_entry)
            else:
                self.calibration_steps.append(step_entry)
            
            # Add to history
            entry = StatusEntry(
                timestamp=time.time(),
                entry_type="calibration_step",
                details=step_entry,
                level="error" if status == "error" else "info"
            )
            self.status_history.append(entry)
            
            if status == "error":
                self.recent_errors.append({
                    "timestamp": time.time(),
                    "error": f"Calibration step {step} failed: {error}",
                    "context": "calibration",
                    "step": step
                })
            
            logger.info(f"Calibration step {step}: {status}" + 
                       (f" ({duration:.2f}s)" if duration else "") +
                       (f" - {error}" if error else ""))
            
            # Broadcast status update to WebSocket clients
            await self._broadcast_status_update()
            
        except Exception as e:
            logger.error(f"Error logging calibration step: {e}")
    
    async def log_service_status(self, service: str, status: str, details: Dict[str, Any] = None) -> None:
        """
        Track service health and activity.
        
        Args:
            service: Service name (e.g., "OrderService", "PositionTracker")
            status: Service status ("healthy", "degraded", "error", "inactive", custom)
            details: Optional additional details
        """
        try:
            if service not in self.services:
                self.services[service] = ServiceStatus(
                    service_name=service,
                    status=status,
                    last_activity=time.time()
                )
            
            service_status = self.services[service]
            
            if status == "error":
                error_msg = details.get("error", "Unknown error") if details else "Unknown error"
                service_status.record_error(error_msg)
            else:
                service_status.update_activity(status, details or {})
            
            # Add to history
            entry = StatusEntry(
                timestamp=time.time(),
                entry_type="service_status",
                details={
                    "service": service,
                    "status": status,
                    "details": details or {}
                },
                level="error" if status == "error" else "info"
            )
            self.status_history.append(entry)
            
            if status == "error":
                error_msg = details.get("error", "Unknown error") if details else "Unknown error"
                self.recent_errors.append({
                    "timestamp": time.time(),
                    "error": f"Service {service}: {error_msg}",
                    "context": "service",
                    "service": service
                })
            
            logger.debug(f"Service {service}: {status}" + (f" - {details}" if details else ""))
            
        except Exception as e:
            logger.error(f"Error logging service status: {e}")
    
    async def log_action_result(self, action: str, result: str, duration: float = None, error: str = None) -> None:
        """
        Track action results (orders, syncs, etc.).
        
        Args:
            action: Action name (e.g., "order_placed", "fill_processed", "positions_synced")
            result: Action result description
            duration: Optional action duration in seconds
            error: Optional error message
        """
        try:
            entry_details = {
                "action": action,
                "result": result,
                "duration": duration
            }
            
            level = "error" if error else "info"
            if error:
                entry_details["error"] = error
            
            # Add to history
            entry = StatusEntry(
                timestamp=time.time(),
                entry_type="action_result",
                details=entry_details,
                level=level
            )
            self.status_history.append(entry)
            
            if error:
                self.recent_errors.append({
                    "timestamp": time.time(),
                    "error": f"Action {action} failed: {error}",
                    "context": "action",
                    "action": action
                })
            
            log_msg = f"Action {action}: {result}"
            if duration:
                log_msg += f" ({duration:.2f}s)"
            if error:
                log_msg += f" - ERROR: {error}"
            
            if error:
                logger.error(log_msg)
            else:
                logger.info(log_msg)
                
            # Broadcast status update to WebSocket clients
            await self._broadcast_status_update()
            
        except Exception as e:
            logger.error(f"Error logging action result: {e}")
    
    def get_debug_summary(self, include_portfolio: bool = True) -> str:
        """
        Get copy-paste friendly debug summary.
        
        Args:
            include_portfolio: Include portfolio information if available
            
        Returns:
            Multi-line debug summary string
        """
        try:
            current_time = time.time()
            time_in_state = current_time - self.state_entered_at
            
            lines = [
                "=== TRADER STATUS ===",
                f"State: {self.current_state.value.upper()} ({time_in_state:.1f}s)"
            ]
            
            if self.last_state_transition:
                transition = self.last_state_transition
                lines.append(f"Last: {transition['from_state']} → {transition['to_state']} ({transition.get('reason', 'no reason')})")
            
            lines.append("")
            lines.append("Services:")
            
            for service_name, service in self.services.items():
                status_icon = "✅" if service.status == "healthy" else "❌" if service.status == "error" else "⚠️"
                age = service.time_since_activity
                activity_desc = ""
                
                # Add service-specific details
                if "operation" in service.activity_details:
                    activity_desc = f" - {service.activity_details['operation']}"
                elif "orders_pending" in service.activity_details:
                    activity_desc = f" - {service.activity_details['orders_pending']} pending"
                elif "positions_tracked" in service.activity_details:
                    activity_desc = f" - {service.activity_details['positions_tracked']} positions"
                
                lines.append(f"{status_icon} {service_name}: {service.status} (last: {age:.0f}s ago){activity_desc}")
            
            # Portfolio summary if requested and available
            if include_portfolio:
                # This would be filled in by the coordinator with actual portfolio data
                lines.append("")
                lines.append("Portfolio: [Portfolio data would be injected by coordinator]")
            
            # Recent activity
            lines.append("")
            lines.append("Recent Activity:")
            
            recent_entries = list(self.status_history)[-5:] if self.status_history else []
            for entry in recent_entries:
                timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
                
                if entry.entry_type == "action_result":
                    action = entry.details.get("action", "unknown")
                    result = entry.details.get("result", "unknown")
                    duration = entry.details.get("duration")
                    duration_str = f" ({duration:.2f}s)" if duration else ""
                    lines.append(f"{timestamp_str} - {action}: {result}{duration_str}")
                
                elif entry.entry_type == "state_transition":
                    from_state = entry.details.get("from_state", "unknown")
                    to_state = entry.details.get("to_state", "unknown")
                    reason = entry.details.get("reason", "no reason")
                    lines.append(f"{timestamp_str} - state: {from_state} → {to_state} ({reason})")
                
                elif entry.entry_type == "calibration_step":
                    step = entry.details.get("step", "unknown")
                    status = entry.details.get("status", "unknown")
                    lines.append(f"{timestamp_str} - calibration: {step} {status}")
            
            # Recent errors
            if self.recent_errors:
                lines.append("")
                lines.append("Recent Errors:")
                recent_errors = list(self.recent_errors)[-3:] if len(self.recent_errors) > 3 else list(self.recent_errors)
                for error_entry in recent_errors:
                    timestamp_str = datetime.fromtimestamp(error_entry["timestamp"]).strftime("%H:%M:%S")
                    error_msg = error_entry.get("error", "Unknown error")
                    lines.append(f"{timestamp_str} - ERROR: {error_msg}")
            else:
                lines.append("")
                lines.append("Last Error: None")
            
            lines.append("================")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error generating debug summary: {e}")
            return f"Error generating debug summary: {e}"
    
    def get_status_history(self, limit: int = 20, entry_type: str = None) -> List[Dict[str, Any]]:
        """
        Get recent status entries for WebSocket broadcasting.
        
        Args:
            limit: Maximum number of entries to return
            entry_type: Optional filter by entry type
            
        Returns:
            List of status entry dictionaries
        """
        try:
            entries = list(self.status_history)
            
            if entry_type:
                entries = [e for e in entries if e.entry_type == entry_type]
            
            # Return most recent entries up to limit
            entries = entries[-limit:] if len(entries) > limit else entries
            
            return [entry.to_dict() for entry in entries]
            
        except Exception as e:
            logger.error(f"Error getting status history: {e}")
            return []
    
    def get_current_status(self) -> Dict[str, Any]:
        """
        Get current comprehensive status for APIs.
        
        Returns:
            Dict with current state, service status, and recent activity
        """
        try:
            current_time = time.time()
            
            return {
                "state_machine": {
                    "current_state": self.current_state.value,
                    "time_in_state": current_time - self.state_entered_at,
                    "last_transition": self.last_state_transition,
                    "calibration_steps": self.calibration_steps
                },
                "services": {name: service.to_dict() for name, service in self.services.items()},
                "recent_activities": self.get_status_history(10),
                "recent_errors": list(self.recent_errors)[-5:] if self.recent_errors else [],
                "status_timestamp": current_time
            }
            
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return {"error": str(e)}
    
    def clear_calibration_steps(self) -> None:
        """Clear calibration steps (called when starting new calibration)."""
        self.calibration_steps.clear()
    
    def get_service_status(self, service_name: str) -> Optional[ServiceStatus]:
        """Get status for a specific service."""
        return self.services.get(service_name)
    
    def is_service_healthy(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        service = self.services.get(service_name)
        return service is not None and service.status == "healthy"
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring."""
        total_errors = sum(service.error_count for service in self.services.values())
        recent_error_count = len(self.recent_errors)
        
        return {
            "total_service_errors": total_errors,
            "recent_errors": recent_error_count,
            "services_with_errors": [
                name for name, service in self.services.items() 
                if service.error_count > 0
            ],
            "last_error": list(self.recent_errors)[-1] if self.recent_errors else None
        }
    
    async def _broadcast_state_transition(
        self, 
        from_state: str, 
        to_state: str, 
        reason: str = None, 
        time_in_previous_state: float = None
    ) -> None:
        """
        Broadcast state machine transition via WebSocket.
        
        Args:
            from_state: Previous state
            to_state: New state
            reason: Optional reason for transition
            time_in_previous_state: Time spent in previous state (seconds)
        """
        try:
            if not self.websocket_manager:
                logger.debug("No websocket manager available for state transition broadcast")
                return
            
            # Prepare state transition data
            transition_data = {
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
                "time_in_previous_state": time_in_previous_state,
                "timestamp": time.time()
            }
            
            # Broadcast via WebSocketManager
            await self.websocket_manager.broadcast_state_transition(transition_data)
            logger.debug(f"Broadcasted state transition: {from_state} → {to_state}")
            
        except Exception as e:
            logger.error(f"Error broadcasting state transition: {e}")

    async def _broadcast_status_update(self) -> None:
        """Broadcast current trader status via WebSocket."""
        try:
            if not self.websocket_manager:
                logger.debug("No websocket manager available for status broadcast")
                return
            
            # Get current status and history
            current_status = self.get_current_status()
            status_history = self.get_status_history(limit=20)
            
            # Prepare data for WebSocket broadcast
            status_data = {
                "current_status": self.current_state.value,
                "status_history": status_history,
                "timestamp": time.time()
            }
            
            # Broadcast via WebSocketManager
            await self.websocket_manager.broadcast_trader_status(status_data)
            logger.debug(f"Broadcasted trader status: {self.current_state.value}")
            
        except Exception as e:
            logger.error(f"Error broadcasting status update: {e}")