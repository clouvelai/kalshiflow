"""
Base Agent class for the multi-agent research pipeline.

Provides common functionality for all agents:
- Lifecycle management (start/stop)
- Health monitoring
- Event emission
- Error handling
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.event_bus import EventBus
    from ..core.websocket_manager import V3WebSocketManager

logger = logging.getLogger("kalshiflow_rl.traderv3.agents.base_agent")


class AgentStatus(Enum):
    """Agent lifecycle status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AgentStats:
    """Statistics tracked by all agents."""
    started_at: Optional[float] = None
    stopped_at: Optional[float] = None
    events_processed: int = 0
    errors_count: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[float] = None
    custom_stats: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the pipeline.

    Provides:
    - Async lifecycle (start/stop)
    - Health status tracking
    - Event bus integration
    - WebSocket broadcasting
    - Error handling with recovery
    """

    def __init__(
        self,
        name: str,
        display_name: Optional[str] = None,
        event_bus: Optional["EventBus"] = None,
        websocket_manager: Optional["V3WebSocketManager"] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: Internal agent name (e.g., "reddit_entity")
            display_name: Human-readable name (e.g., "Reddit Entity Agent")
            event_bus: Event bus for inter-agent communication
            websocket_manager: WebSocket manager for frontend broadcasting
        """
        self._name = name
        self._display_name = display_name or name.replace("_", " ").title()
        self._event_bus = event_bus
        self._ws_manager = websocket_manager

        self._status = AgentStatus.STOPPED
        self._stats = AgentStats()
        self._running = False
        self._main_task: Optional[asyncio.Task] = None

    @property
    def name(self) -> str:
        """Get the agent's internal name."""
        return self._name

    @property
    def display_name(self) -> str:
        """Get the agent's display name."""
        return self._display_name

    @property
    def status(self) -> AgentStatus:
        """Get the agent's current status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if the agent is running."""
        return self._running and self._status == AgentStatus.RUNNING

    async def start(self) -> None:
        """Start the agent."""
        if self._running:
            logger.warning(f"[{self._name}] Agent already running")
            return

        logger.info(f"[{self._name}] Starting {self._display_name}...")
        self._status = AgentStatus.STARTING
        self._running = True
        self._stats.started_at = time.time()

        try:
            # Call subclass initialization
            await self._on_start()
            self._status = AgentStatus.RUNNING
            logger.info(f"[{self._name}] {self._display_name} started successfully")

        except Exception as e:
            logger.error(f"[{self._name}] Failed to start: {e}")
            self._status = AgentStatus.ERROR
            self._record_error(str(e))
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop the agent."""
        if not self._running:
            return

        logger.info(f"[{self._name}] Stopping {self._display_name}...")
        self._status = AgentStatus.STOPPING
        self._running = False

        try:
            # Cancel main task if running
            if self._main_task and not self._main_task.done():
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass

            # Call subclass cleanup
            await self._on_stop()

            self._status = AgentStatus.STOPPED
            self._stats.stopped_at = time.time()
            logger.info(f"[{self._name}] {self._display_name} stopped")

        except Exception as e:
            logger.error(f"[{self._name}] Error during stop: {e}")
            self._status = AgentStatus.ERROR
            self._record_error(str(e))

    @abstractmethod
    async def _on_start(self) -> None:
        """
        Called when the agent starts.
        Subclasses should initialize resources here.
        """
        pass

    @abstractmethod
    async def _on_stop(self) -> None:
        """
        Called when the agent stops.
        Subclasses should cleanup resources here.
        """
        pass

    def _record_error(self, error: str) -> None:
        """Record an error in stats."""
        self._stats.errors_count += 1
        self._stats.last_error = error
        self._stats.last_error_at = time.time()

    def _record_event_processed(self) -> None:
        """Record that an event was processed."""
        self._stats.events_processed += 1

    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to the event bus."""
        if self._event_bus:
            await self._event_bus.emit(event_type, data)

    async def _broadcast_to_frontend(self, message_type: str, data: Dict[str, Any]) -> None:
        """Broadcast a message to frontend clients."""
        if self._ws_manager:
            await self._ws_manager.broadcast_message(message_type, data)

    def get_health_details(self) -> Dict[str, Any]:
        """Get agent health details."""
        return {
            "name": self._name,
            "display_name": self._display_name,
            "status": self._status.value,
            "is_running": self._running,
            "stats": {
                "started_at": self._stats.started_at,
                "stopped_at": self._stats.stopped_at,
                "events_processed": self._stats.events_processed,
                "errors_count": self._stats.errors_count,
                "last_error": self._stats.last_error,
                "last_error_at": self._stats.last_error_at,
                **self._stats.custom_stats,
            },
            **self._get_agent_stats(),
        }

    def _get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent-specific stats.
        Subclasses can override to add custom stats.
        """
        return {}
