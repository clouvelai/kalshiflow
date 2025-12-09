"""
Training session lifecycle management for Kalshi Flow RL Trading Subsystem.

Provides complete training session management including session creation,
execution coordination, cleanup, and persistent session state tracking.
Integrates with model registry, training monitoring, and database persistence.
"""

import asyncio
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

from ..config import config
from ..data.database import rl_db
from .model_registry import model_registry
from .training_config import TrainingConfig
from .training_monitor import TrainingMonitor, TrainingProgressCallback, create_training_monitor

logger = logging.getLogger("kalshiflow_rl.session_manager")


class SessionStatus(Enum):
    """Training session status."""
    PENDING = "pending"
    INITIALIZING = "initializing"  
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class SessionState:
    """Persistent session state."""
    session_id: str
    model_id: int
    config: TrainingConfig
    status: SessionStatus
    
    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress
    current_step: int = 0
    total_steps: int = 0
    current_episode: int = 0
    
    # Performance
    best_reward: float = float('-inf')
    best_return: float = float('-inf')
    current_metrics: Optional[Dict[str, Any]] = None
    
    # Resource tracking
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Error tracking
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        # Convert enums and datetime objects
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        if self.started_at:
            result['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            result['completed_at'] = self.completed_at.isoformat()
        # Convert config to dict
        result['config'] = self.config.to_dict()
        return result


class SessionManager:
    """
    Manages training session lifecycle and persistence.
    
    Provides:
    - Session creation and initialization
    - Progress tracking and state persistence
    - Resource monitoring and cleanup
    - Session recovery and resumption
    - Multi-session coordination
    """
    
    def __init__(self, max_concurrent_sessions: int = 1, max_memory_per_session_mb: int = 4000):
        """Initialize session manager.
        
        Args:
            max_concurrent_sessions: Maximum number of concurrent training sessions
            max_memory_per_session_mb: Maximum memory per session in MB
        """
        self.active_sessions: Dict[str, SessionState] = {}
        self.session_monitors: Dict[str, TrainingMonitor] = {}
        self.session_locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Resource limits
        self.max_concurrent_sessions = max_concurrent_sessions
        self.max_memory_per_session_mb = max_memory_per_session_mb
        
        # Cleanup settings
        self.session_timeout_hours = 48  # Max session duration
        self.cleanup_frequency_minutes = 30  # How often to run cleanup
        
        logger.info(f"Session manager initialized with max_concurrent_sessions={max_concurrent_sessions}")
    
    async def initialize(self) -> None:
        """Initialize async components of session manager."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())
            logger.info("Started background cleanup task")
    
    async def create_session(
        self,
        config: TrainingConfig,
        session_id: Optional[str] = None
    ) -> str:
        """
        Create new training session.
        
        Args:
            config: Training configuration
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        # Generate session ID if not provided
        if not session_id:
            timestamp = int(time.time())
            session_id = f"{config.model_name}_{config.version}_{timestamp}"
        
        # Check concurrent session limit
        with self._global_lock:
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                raise RuntimeError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
        
        # Create session state
        session_state = SessionState(
            session_id=session_id,
            model_id=0,  # Will be set during initialization
            config=config,
            status=SessionStatus.PENDING,
            created_at=datetime.utcnow(),
            total_steps=config.total_timesteps
        )
        
        # Add to active sessions
        with self._global_lock:
            self.active_sessions[session_id] = session_state
            self.session_locks[session_id] = threading.Lock()
        
        logger.info(f"Created training session {session_id}")
        return session_id
    
    async def initialize_session(self, session_id: str) -> bool:
        """
        Initialize training session.
        
        Args:
            session_id: Session to initialize
            
        Returns:
            True if initialization successful
        """
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session_state = self.active_sessions[session_id]
        
        try:
            # Update status
            await self._update_session_status(session_id, SessionStatus.INITIALIZING)
            
            # Register model in database
            from .model_registry import ModelConfig
            model_config = ModelConfig(
                model_name=session_state.config.model_name,
                version=session_state.config.version,
                algorithm=session_state.config.algorithm.value,
                market_ticker=session_state.config.market_tickers[0],  # Primary market
                hyperparameters=session_state.config.to_dict()
            )
            
            model_id = await model_registry.register_model(
                model_config=model_config,
                initial_metrics={'session_id': session_id}
            )
            
            # Update session with model ID
            with self.session_locks[session_id]:
                session_state.model_id = model_id
            
            # Create training monitor
            monitor, callback = create_training_monitor(
                model_id=model_id,
                performance_targets=session_state.config.env_config.get('performance_targets')
            )
            
            self.session_monitors[session_id] = monitor
            
            # Set session as ready for training
            await self._update_session_status(session_id, SessionStatus.PENDING)
            
            logger.info(f"Session {session_id} initialized with model ID {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize session {session_id}: {e}")
            await self._update_session_status(session_id, SessionStatus.FAILED, str(e))
            return False
    
    async def start_training(self, session_id: str) -> bool:
        """
        Start training for session.
        
        Args:
            session_id: Session to start training
            
        Returns:
            True if training started successfully
        """
        if session_id not in self.active_sessions:
            logger.error(f"Session {session_id} not found")
            return False
        
        session_state = self.active_sessions[session_id]
        
        if session_state.status != SessionStatus.PENDING:
            logger.error(f"Session {session_id} not ready for training (status: {session_state.status})")
            return False
        
        try:
            # Update status and timing
            await self._update_session_status(session_id, SessionStatus.TRAINING)
            
            with self.session_locks[session_id]:
                session_state.started_at = datetime.utcnow()
            
            # Training will be handled by TrainingManager
            logger.info(f"Training started for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start training for session {session_id}: {e}")
            await self._update_session_status(session_id, SessionStatus.FAILED, str(e))
            return False
    
    async def update_session_progress(
        self,
        session_id: str,
        current_step: int,
        current_episode: int,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update session progress.
        
        Args:
            session_id: Session to update
            current_step: Current training step
            current_episode: Current episode number
            metrics: Optional performance metrics
        """
        if session_id not in self.active_sessions:
            return
        
        with self.session_locks[session_id]:
            session_state = self.active_sessions[session_id]
            session_state.current_step = current_step
            session_state.current_episode = current_episode
            
            if metrics:
                session_state.current_metrics = metrics
                
                # Update best performance
                if 'avg_episode_reward' in metrics:
                    session_state.best_reward = max(
                        session_state.best_reward,
                        metrics['avg_episode_reward']
                    )
                
                if 'avg_portfolio_return' in metrics:
                    session_state.best_return = max(
                        session_state.best_return,
                        metrics['avg_portfolio_return']
                    )
    
    async def complete_session(
        self,
        session_id: str,
        success: bool,
        final_metrics: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Mark session as completed.
        
        Args:
            session_id: Session to complete
            success: Whether training was successful
            final_metrics: Final performance metrics
            error_message: Error message if failed
        """
        if session_id not in self.active_sessions:
            return
        
        # Determine final status
        if success:
            final_status = SessionStatus.COMPLETED
        else:
            final_status = SessionStatus.FAILED
        
        # Update session state
        with self.session_locks[session_id]:
            session_state = self.active_sessions[session_id]
            session_state.completed_at = datetime.utcnow()
            session_state.error_message = error_message
            if final_metrics:
                session_state.current_metrics = final_metrics
        
        # Update status
        await self._update_session_status(session_id, final_status, error_message)
        
        # Update model status in registry
        if session_state.model_id and success:
            await model_registry.set_model_status(session_state.model_id, 'active')
        elif session_state.model_id:
            await model_registry.set_model_status(session_state.model_id, 'failed')
        
        logger.info(f"Session {session_id} completed with status {final_status.value}")
    
    async def cancel_session(self, session_id: str) -> bool:
        """
        Cancel active session.
        
        Args:
            session_id: Session to cancel
            
        Returns:
            True if cancelled successfully
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found for cancellation")
            return False
        
        session_state = self.active_sessions[session_id]
        
        if session_state.status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED]:
            logger.warning(f"Session {session_id} already finished, cannot cancel")
            return False
        
        # Update status
        await self._update_session_status(session_id, SessionStatus.CANCELLED)
        
        # Clean up resources
        await self._cleanup_session_resources(session_id)
        
        logger.info(f"Session {session_id} cancelled")
        return True
    
    async def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session status and metrics."""
        if session_id not in self.active_sessions:
            return None
        
        session_state = self.active_sessions[session_id]
        
        # Get latest metrics from monitor if available
        latest_metrics = None
        if session_id in self.session_monitors:
            monitor = self.session_monitors[session_id]
            latest_metrics = monitor.get_current_metrics().to_dict()
        
        status_dict = session_state.to_dict()
        if latest_metrics:
            status_dict['latest_metrics'] = latest_metrics
            status_dict['training_summary'] = self.session_monitors[session_id].get_training_summary()
        
        return status_dict
    
    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        sessions = []
        for session_id, session_state in self.active_sessions.items():
            status = await self.get_session_status(session_id)
            if status:
                sessions.append(status)
        return sessions
    
    async def cleanup_session(self, session_id: str) -> None:
        """Clean up session resources and remove from active sessions."""
        if session_id not in self.active_sessions:
            return
        
        # Clean up resources
        await self._cleanup_session_resources(session_id)
        
        # Remove from active sessions
        with self._global_lock:
            self.active_sessions.pop(session_id, None)
            self.session_locks.pop(session_id, None)
            self.session_monitors.pop(session_id, None)
        
        logger.info(f"Session {session_id} cleaned up")
    
    async def _update_session_status(
        self,
        session_id: str,
        status: SessionStatus,
        error_message: Optional[str] = None
    ) -> None:
        """Update session status."""
        if session_id not in self.active_sessions:
            return
        
        with self.session_locks[session_id]:
            session_state = self.active_sessions[session_id]
            session_state.status = status
            if error_message:
                session_state.error_message = error_message
        
        logger.debug(f"Session {session_id} status updated to {status.value}")
    
    async def _cleanup_session_resources(self, session_id: str) -> None:
        """Clean up resources for a session."""
        try:
            # Clean up monitor
            if session_id in self.session_monitors:
                # Monitor cleanup is automatic via garbage collection
                pass
            
            # Model registry cleanup is handled by model lifecycle
            
            logger.debug(f"Resources cleaned up for session {session_id}")
            
        except Exception as e:
            logger.warning(f"Error during session cleanup {session_id}: {e}")
    
    async def _background_cleanup_loop(self) -> None:
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_frequency_minutes * 60)
                await self._cleanup_expired_sessions()
            except Exception as e:
                logger.warning(f"Error in background cleanup: {e}")
    
    async def _cleanup_expired_sessions(self) -> None:
        """Clean up expired and completed sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_state in self.active_sessions.items():
            # Check for timeout
            if session_state.created_at < current_time - timedelta(hours=self.session_timeout_hours):
                expired_sessions.append(session_id)
                continue
            
            # Check for completed sessions older than 1 hour
            if (session_state.status in [SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED] and
                session_state.completed_at and 
                session_state.completed_at < current_time - timedelta(hours=1)):
                expired_sessions.append(session_id)
        
        # Clean up expired sessions
        for session_id in expired_sessions:
            await self.cleanup_session(session_id)
            
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown session manager."""
        logger.info(f"Shutting down session manager with {len(self.active_sessions)} active sessions")
        
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Clean up all active sessions
        cleanup_tasks = []
        for session_id in list(self.active_sessions.keys()):
            cleanup_tasks.append(asyncio.create_task(self.cleanup_session(session_id)))
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        logger.info("Session manager shutdown complete")


# Global session manager instance
session_manager = SessionManager()