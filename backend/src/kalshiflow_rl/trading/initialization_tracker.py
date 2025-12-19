"""
Initialization Tracker for Trader/Actor startup sequence.

Manages a formalized, step-by-step initialization checklist that verifies
the trader properly syncs with Kalshi and is ready to resume trading.

Core responsibility: Track initialization progress and broadcast updates
via WebSocket so the frontend can display a TurboTax-style checklist.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("kalshiflow_rl.trading.initialization_tracker")


class StepStatus(Enum):
    """Status of an initialization step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class InitializationStep:
    """Represents a single initialization step."""
    step_id: str
    name: str
    status: StepStatus = StepStatus.PENDING
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def mark_in_progress(self) -> None:
        """Mark step as in progress."""
        self.status = StepStatus.IN_PROGRESS
        self.started_at = time.time()
    
    def mark_complete(self, details: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as complete with optional details."""
        self.status = StepStatus.COMPLETE
        self.completed_at = time.time()
        if details:
            self.details.update(details)
        self.error = None
    
    def mark_failed(self, error: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Mark step as failed with error message."""
        self.status = StepStatus.FAILED
        self.completed_at = time.time()
        self.error = error
        if details:
            self.details.update(details)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for WebSocket broadcast."""
        return {
            "step_id": self.step_id,
            "name": self.name,
            "status": self.status.value,
            "details": self.details,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class InitializationTracker:
    """
    Tracks initialization progress and broadcasts updates via WebSocket.
    
    Manages a checklist of initialization steps that verify:
    1. Client health checks (orderbook, trader client, fill listener, event bus)
    2. Orderbook sync verification (receiving data, markets subscribed)
    3. Trader state sync with Kalshi (balance, positions, orders)
    4. Listener subscription verification
    
    After all steps complete, the trader is ready to resume trading.
    """
    
    # Define all initialization steps in order
    STEP_IDS = {
        # Phase 1: Connection & Health Checks
        "orderbook_health": "Orderbook Client Health Check",
        "trader_client_health": "Trader Client Health Check",
        "fill_listener_health": "Fill Listener Health Check",
        "event_bus_health": "Event Bus Health Check",
        
        # Phase 2: State Discovery & Sync
        "sync_balance": "Sync Portfolio Balance/Cash",
        "sync_positions": "Sync Positions",
        "sync_orders": "Sync Orders",
        
        # Phase 3: Listener Subscription Verification
        "verify_orderbook_subscriptions": "Verify Orderbook Subscriptions",
        "verify_fill_listener_subscription": "Verify Fill Listener Subscription",
        "verify_listeners": "Verify Order/Position/Fill Listeners",
        
        # Phase 4: Ready to Resume
        "initialization_complete": "Initialization Complete",
    }
    
    def __init__(self, websocket_manager=None):
        """
        Initialize the tracker.
        
        Args:
            websocket_manager: WebSocketManager instance for broadcasting updates
        """
        self.websocket_manager = websocket_manager
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.warnings: List[str] = []
        
        # Create step tracking
        self.steps: Dict[str, InitializationStep] = {}
        for step_id, name in self.STEP_IDS.items():
            self.steps[step_id] = InitializationStep(step_id=step_id, name=name)
        
        # Component health tracking
        self.component_health: Dict[str, Dict[str, Any]] = {}
        
        logger.info("InitializationTracker created")
    
    async def start(self) -> None:
        """Start initialization sequence."""
        self.started_at = time.time()
        logger.info("Starting initialization sequence...")
        
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast_initialization_start({
                    "started_at": self.started_at
                })
            except Exception as e:
                logger.error(f"Failed to broadcast initialization_start: {e}")
    
    async def mark_step_in_progress(self, step_id: str) -> None:
        """Mark a step as in progress."""
        if step_id not in self.steps:
            logger.warning(f"Unknown step_id: {step_id}")
            return
        
        step = self.steps[step_id]
        step.mark_in_progress()
        
        logger.info(f"Initialization step in progress: {step.name}")
        await self._broadcast_step_update(step)
    
    async def mark_step_complete(
        self,
        step_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a step as complete."""
        if step_id not in self.steps:
            logger.warning(f"Unknown step_id: {step_id}")
            return
        
        step = self.steps[step_id]
        step.mark_complete(details)
        
        logger.info(f"Initialization step complete: {step.name}")
        await self._broadcast_step_update(step)
    
    async def mark_step_failed(
        self,
        step_id: str,
        error: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Mark a step as failed."""
        if step_id not in self.steps:
            logger.warning(f"Unknown step_id: {step_id}")
            return
        
        step = self.steps[step_id]
        step.mark_failed(error, details)
        
        logger.error(f"Initialization step failed: {step.name} - {error}")
        self.warnings.append(f"{step.name}: {error}")
        await self._broadcast_step_update(step)
    
    async def complete_initialization(self, summary: Optional[Dict[str, Any]] = None) -> None:
        """Mark initialization as complete."""
        self.completed_at = time.time()
        
        # Ensure final step is marked complete
        if "initialization_complete" in self.steps:
            final_step = self.steps["initialization_complete"]
            final_step.mark_complete()
        
        summary_data = {
            "completed_at": self.completed_at,
            "duration_seconds": self.completed_at - self.started_at if self.started_at else 0.0,
            "steps": {step_id: step.to_dict() for step_id, step in self.steps.items()},
            "warnings": self.warnings.copy(),
        }
        
        if summary:
            summary_data.update(summary)
        
        logger.info(
            f"Initialization complete in {summary_data['duration_seconds']:.2f}s. "
            f"{len([s for s in self.steps.values() if s.status == StepStatus.COMPLETE])}/{len(self.steps)} steps completed"
        )
        
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast_initialization_complete(summary_data)
            except Exception as e:
                logger.error(f"Failed to broadcast initialization_complete: {e}")
    
    async def update_component_health(
        self,
        component: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update health status for a component.
        
        Args:
            component: Component name (e.g., "orderbook_client")
            status: Health status ("healthy", "unhealthy", "degraded")
            details: Additional health details
        """
        self.component_health[component] = {
            "status": status,
            "last_update": time.time(),
            "details": details or {}
        }
        
        if self.websocket_manager:
            try:
                await self.websocket_manager.broadcast_component_health({
                    "component": component,
                    "status": status,
                    "last_update": self.component_health[component]["last_update"],
                    "details": details or {}
                })
            except Exception as e:
                logger.error(f"Failed to broadcast component_health: {e}")
    
    async def _broadcast_step_update(self, step: InitializationStep) -> None:
        """Broadcast step update via WebSocket."""
        if not self.websocket_manager:
            return
        
        try:
            await self.websocket_manager.broadcast_initialization_step(step.to_dict())
        except Exception as e:
            logger.error(f"Failed to broadcast step update: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current initialization status."""
        completed = len([s for s in self.steps.values() if s.status == StepStatus.COMPLETE])
        failed = len([s for s in self.steps.values() if s.status == StepStatus.FAILED])
        in_progress = len([s for s in self.steps.values() if s.status == StepStatus.IN_PROGRESS])
        
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_steps": len(self.steps),
            "completed_steps": completed,
            "failed_steps": failed,
            "in_progress_steps": in_progress,
            "steps": {step_id: step.to_dict() for step_id, step in self.steps.items()},
            "warnings": self.warnings.copy(),
            "component_health": self.component_health.copy(),
        }
    
    def is_complete(self) -> bool:
        """Check if initialization is complete."""
        return self.completed_at is not None
    
    def has_critical_failures(self) -> bool:
        """Check if any critical steps failed."""
        # Critical steps that must succeed
        critical_steps = [
            "orderbook_health",
            "trader_client_health",
            "sync_balance",
            "sync_positions",
            "sync_orders",
        ]
        
        for step_id in critical_steps:
            if step_id in self.steps:
                if self.steps[step_id].status == StepStatus.FAILED:
                    return True
        
        return False

