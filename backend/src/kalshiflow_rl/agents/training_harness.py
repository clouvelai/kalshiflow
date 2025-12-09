"""
Training harness for Kalshi Flow RL Trading Subsystem.

Integrates Stable Baselines3 (SB3) with KalshiTradingEnv for model training.
Provides training session management, multi-market support, hyperparameter
optimization, and comprehensive model lifecycle management.

CRITICAL ARCHITECTURAL REQUIREMENTS:
1. Training uses ONLY historical data (no live WebSocket connections)
2. All database writes are non-blocking via async queues  
3. Training mode is enforced (no 'live' mode allowed)
4. Multi-market support with proper scaling
5. Model checkpointing and hot-reload capabilities
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import threading
import signal
import os

import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm  
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..environments.kalshi_env import KalshiTradingEnv
from ..environments.observation_space import ObservationConfig
from ..environments.action_space import ActionConfig
from ..environments.historical_data_loader import DataLoadConfig
from .model_registry import model_registry, ModelConfig
from .training_config import TrainingConfig, AlgorithmType, TrainingMode
from ..config import config
from ..data.database import rl_db

logger = logging.getLogger("kalshiflow_rl.training_harness")


class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring and episode logging."""
    
    def __init__(
        self,
        training_session: 'TrainingSession',
        model_id: int,
        log_frequency: int = 100,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.training_session = training_session
        self.model_id = model_id
        self.log_frequency = log_frequency
        self.episode_count = 0
        self.last_log_time = time.time()
        
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check for early stopping
        if self.training_session._should_stop:
            logger.info("Training stopped by external signal")
            return False
        
        # Log progress periodically
        if self.num_timesteps % self.log_frequency == 0:
            current_time = time.time()
            time_elapsed = current_time - self.last_log_time
            steps_per_second = self.log_frequency / time_elapsed if time_elapsed > 0 else 0
            
            logger.info(f"Training step {self.num_timesteps}: {steps_per_second:.1f} steps/sec")
            self.last_log_time = current_time
        
        return True
    
    def _on_episode_end(self) -> None:
        """Called at the end of each episode."""
        self.episode_count += 1
        
        # Get episode statistics from environment
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]  # First environment in vector
        else:
            env = self.training_env
        
        if hasattr(env, 'get_episode_stats'):
            episode_stats = env.get_episode_stats()
            
            # Schedule async database write (non-blocking)
            # Use asyncio.ensure_future for compatibility with different contexts
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._log_episode_async(episode_stats))
                else:
                    # If no running loop, log synchronously
                    logger.warning("No running event loop, skipping episode logging")
            except RuntimeError:
                logger.warning("Cannot access event loop, skipping episode logging")
    
    async def _log_episode_async(self, episode_stats: Dict[str, Any]) -> None:
        """Asynchronously log episode to database."""
        try:
            # Convert episode stats to database format
            episode_data = {
                'model_id': self.model_id,
                'episode_number': episode_stats.get('episode', 0),
                'market_ticker': self.training_session.config.market_tickers[0],  # Primary market
                'start_timestamp_ms': int((time.time() - 3600) * 1000),  # Approximate start
                'end_timestamp_ms': int(time.time() * 1000),
                'start_balance': 10000.0,  # Default start balance
                'end_balance': episode_stats.get('portfolio_value', 10000.0),
                'total_return': episode_stats.get('total_return', 0.0),
                'max_drawdown': None,  # TODO: Calculate from episode
                'sharpe_ratio': None,  # TODO: Calculate from episode
                'num_actions': 0,  # TODO: Track during episode
                'num_trades': episode_stats.get('total_trades', 0),
                'episode_reward': None,  # TODO: Track cumulative reward
                'episode_length': episode_stats.get('steps', 0)
            }
            
            await rl_db.create_training_episode(episode_data)
            
        except Exception as e:
            logger.warning(f"Failed to log episode to database: {e}")


class TrainingSession:
    """
    Manages a complete training session for an RL model.
    
    Handles environment setup, model initialization, training loop,
    checkpointing, evaluation, and performance tracking.
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        env_config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[BaseCallback]] = None
    ):
        """
        Initialize training session.
        
        Args:
            config: Training configuration
            env_config: Environment configuration overrides
            callbacks: Additional SB3 callbacks
        """
        self.config = config
        self.env_config = env_config or {}
        self.additional_callbacks = callbacks or []
        
        # Training state
        self.model: Optional[BaseAlgorithm] = None
        self.env: Optional[Union[KalshiTradingEnv, DummyVecEnv]] = None
        self.eval_env: Optional[Union[KalshiTradingEnv, DummyVecEnv]] = None
        self.model_id: Optional[int] = None
        
        # Training control
        self._should_stop = False
        self._training_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        
        # Metrics tracking
        self.training_metrics: Dict[str, Any] = {}
        self.evaluation_results: List[Dict[str, Any]] = []
        
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            raise ValueError(f"Invalid training configuration: {', '.join(config_errors)}")
        
        logger.info(f"Initialized training session: {config.model_name}:{config.version}")
    
    async def setup(self) -> None:
        """Set up training session components."""
        try:
            # Create environments
            await self._create_environments()
            
            # Register model in database
            await self._register_model()
            
            # Create SB3 model
            self._create_sb3_model()
            
            # Set up callbacks
            self._setup_callbacks()
            
            logger.info(f"Training session setup complete for model {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to setup training session: {e}")
            raise
    
    async def train(self) -> Dict[str, Any]:
        """
        Execute training loop.
        
        Returns:
            Training results and metrics
        """
        if not self.model:
            raise RuntimeError("Training session not set up. Call setup() first.")
        
        self._start_time = time.time()
        training_results = {}
        
        try:
            logger.info(f"Starting training: {self.config.total_timesteps} timesteps")
            
            # Update model status to training
            await rl_db.update_model_status(self.model_id, 'training')
            
            # Execute training loop in thread to avoid blocking async context
            training_future = asyncio.get_event_loop().run_in_executor(
                None, self._execute_training_loop
            )
            
            # Wait for training completion with periodic checks
            while not training_future.done():
                await asyncio.sleep(1.0)
                
                # Check training time limit
                if self._start_time and time.time() - self._start_time > self.config.max_training_hours * 3600:
                    logger.warning("Training time limit exceeded")
                    self._should_stop = True
                    break
            
            # Get training results
            if training_future.done():
                training_results = training_future.result()
            
            # Final model checkpoint
            await self._save_final_checkpoint()
            
            # Update model status based on training success
            final_status = 'ready' if training_results.get('success', False) else 'failed'
            await rl_db.update_model_status(self.model_id, final_status)
            
            # Calculate final metrics
            training_duration = time.time() - self._start_time
            training_results.update({
                'training_duration_seconds': training_duration,
                'model_id': self.model_id,
                'final_status': final_status
            })
            
            logger.info(f"Training completed in {training_duration:.1f}s with status {final_status}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            traceback.print_exc()
            
            # Update model status to failed
            if self.model_id:
                await rl_db.update_model_status(self.model_id, 'failed')
            
            training_results = {
                'success': False,
                'error': str(e),
                'model_id': self.model_id
            }
        
        return training_results
    
    def stop_training(self) -> None:
        """Stop training gracefully."""
        logger.info("Training stop requested")
        self._should_stop = True
    
    async def cleanup(self) -> None:
        """Clean up training session resources."""
        try:
            # Close environments
            if self.env:
                self.env.close()
            if self.eval_env:
                self.eval_env.close()
            
            # Clear model from memory
            self.model = None
            
            logger.info("Training session cleanup complete")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    async def _create_environments(self) -> None:
        """Create training and evaluation environments."""
        # Merge environment configuration
        env_config = {
            'market_tickers': self.config.market_tickers,
            'observation_config': ObservationConfig(),
            'action_config': ActionConfig(),
            'data_config': DataLoadConfig(
                window_hours=config.TRAINING_DATA_WINDOW_HOURS
            ),
            **self.env_config
        }
        
        # Create training environment
        self.env = KalshiTradingEnv(**env_config)
        
        # Validate environment
        try:
            check_env(self.env)
            logger.info("Environment validation passed")
        except Exception as e:
            logger.warning(f"Environment validation warning: {e}")
        
        # Wrap in vectorized environment for SB3
        self.env = DummyVecEnv([lambda: Monitor(self.env)])
        
        # Create evaluation environment (separate instance)
        eval_env = KalshiTradingEnv(**env_config)
        self.eval_env = DummyVecEnv([lambda: Monitor(eval_env)])
        
        logger.info(f"Created environments for {len(self.config.market_tickers)} markets")
    
    async def _register_model(self) -> None:
        """Register model in database and model registry."""
        # Create model configuration
        model_config = ModelConfig(
            model_name=self.config.model_name,
            version=self.config.version,
            algorithm=self.config.algorithm.value,
            market_ticker=self.config.market_tickers[0],  # Primary market
            hyperparameters=self.config.to_dict()
        )
        
        # Register with model registry
        self.model_id = await model_registry.register_model(
            model_config=model_config,
            initial_metrics={'training_started_at': datetime.utcnow().isoformat()}
        )
        
        logger.info(f"Registered model with ID {self.model_id}")
    
    def _create_sb3_model(self) -> None:
        """Create Stable Baselines3 model."""
        # Get hyperparameters for SB3
        sb3_params = self.config.get_sb3_hyperparameters()
        
        # Create model based on algorithm
        if self.config.algorithm == AlgorithmType.PPO:
            self.model = PPO(
                policy="MlpPolicy",
                env=self.env,
                **sb3_params
            )
        elif self.config.algorithm == AlgorithmType.A2C:
            self.model = A2C(
                policy="MlpPolicy", 
                env=self.env,
                **sb3_params
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.config.algorithm}")
        
        logger.info(f"Created {self.config.algorithm.value} model with policy MlpPolicy")
    
    def _setup_callbacks(self) -> None:
        """Set up training callbacks."""
        callbacks = []
        
        # Training monitoring callback
        training_callback = TrainingCallback(
            training_session=self,
            model_id=self.model_id,
            log_frequency=self.config.eval_freq // 10
        )
        callbacks.append(training_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(Path(config.MODEL_STORAGE_PATH) / f"model_{self.model_id}_checkpoints"),
            name_prefix=f"checkpoint_{self.model_id}"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env=self.eval_env,
            n_eval_episodes=self.config.eval_episodes,
            eval_freq=self.config.eval_freq,
            log_path=str(Path(config.MODEL_STORAGE_PATH) / f"model_{self.model_id}_eval"),
            best_model_save_path=str(Path(config.MODEL_STORAGE_PATH) / f"model_{self.model_id}_best"),
            deterministic=True,
            verbose=1
        )
        callbacks.append(eval_callback)
        
        # Add custom callbacks
        callbacks.extend(self.additional_callbacks)
        
        # Create callback list
        self.callback_list = CallbackList(callbacks)
    
    def _execute_training_loop(self) -> Dict[str, Any]:
        """Execute the actual training loop (runs in thread)."""
        try:
            # Learn with callbacks
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=self.callback_list,
                log_interval=10,
                reset_num_timesteps=True
            )
            
            return {
                'success': True,
                'total_timesteps': self.config.total_timesteps
            }
            
        except Exception as e:
            logger.error(f"Training loop failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _save_final_checkpoint(self) -> None:
        """Save final model checkpoint."""
        if not self.model:
            return
        
        try:
            # Generate final checkpoint path
            final_path = Path(config.MODEL_STORAGE_PATH) / f"model_{self.model_id}_final"
            final_path.mkdir(parents=True, exist_ok=True)
            
            # Save model
            checkpoint_path = str(final_path / "final_model")
            await model_registry.save_model_checkpoint(
                model=self.model,
                file_path=checkpoint_path,
                metadata={
                    'training_completed_at': datetime.utcnow().isoformat(),
                    'total_timesteps': self.config.total_timesteps,
                    'model_id': self.model_id
                }
            )
            
            # Update model file path in database
            await rl_db.update_model_status(self.model_id, 'ready')
            
            logger.info(f"Saved final checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save final checkpoint: {e}")


class TrainingManager:
    """
    High-level manager for coordinating multiple training sessions.
    
    Provides:
    - Multi-market training orchestration
    - Training queue management
    - Resource allocation and monitoring
    - Training session lifecycle management
    """
    
    def __init__(self):
        """Initialize training manager."""
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.training_queue: List[TrainingConfig] = []
        self.max_concurrent_sessions = 1  # Limit concurrent training
        self._shutdown_requested = False
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        
        logger.info("Training manager initialized")
    
    async def start_training(
        self,
        training_config: TrainingConfig,
        env_config: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[BaseCallback]] = None
    ) -> str:
        """
        Start new training session.
        
        Args:
            training_config: Training configuration
            env_config: Environment configuration overrides
            callbacks: Additional SB3 callbacks
            
        Returns:
            Session ID for tracking
        """
        # Generate session ID
        session_id = f"{training_config.model_name}_{training_config.version}_{int(time.time())}"
        
        # Check concurrent session limit
        if len(self.active_sessions) >= self.max_concurrent_sessions:
            # Add to queue
            self.training_queue.append(training_config)
            logger.info(f"Training session {session_id} added to queue")
            return session_id
        
        # Create and start session
        session = TrainingSession(
            config=training_config,
            env_config=env_config,
            callbacks=callbacks
        )
        
        try:
            # Setup session
            await session.setup()
            
            # Add to active sessions
            self.active_sessions[session_id] = session
            
            # Start training in background
            asyncio.create_task(self._run_training_session(session_id, session))
            
            logger.info(f"Started training session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to start training session {session_id}: {e}")
            raise
        
        return session_id
    
    async def stop_training(self, session_id: str) -> bool:
        """
        Stop training session.
        
        Args:
            session_id: Session to stop
            
        Returns:
            True if stopped successfully
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Training session {session_id} not found")
            return False
        
        session = self.active_sessions[session_id]
        session.stop_training()
        
        logger.info(f"Requested stop for training session {session_id}")
        return True
    
    async def get_training_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of training session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        
        status = {
            'session_id': session_id,
            'model_name': session.config.model_name,
            'version': session.config.version,
            'algorithm': session.config.algorithm.value,
            'markets': session.config.market_tickers,
            'model_id': session.model_id,
            'is_training': not session._should_stop,
            'start_time': session._start_time
        }
        
        if session._start_time:
            status['elapsed_seconds'] = time.time() - session._start_time
        
        return status
    
    async def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active training sessions."""
        sessions = []
        for session_id, session in self.active_sessions.items():
            status = await self.get_training_status(session_id)
            if status:
                sessions.append(status)
        return sessions
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all training sessions."""
        self._shutdown_requested = True
        
        logger.info(f"Shutting down {len(self.active_sessions)} active training sessions")
        
        # Stop all active sessions
        stop_tasks = []
        for session_id, session in self.active_sessions.items():
            session.stop_training()
            stop_tasks.append(asyncio.create_task(session.cleanup()))
        
        # Wait for all sessions to stop
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # Clear active sessions
        self.active_sessions.clear()
        self.training_queue.clear()
        
        logger.info("Training manager shutdown complete")
    
    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(self.shutdown())
    
    async def _run_training_session(self, session_id: str, session: TrainingSession) -> None:
        """Run training session to completion."""
        try:
            # Execute training
            results = await session.train()
            
            # Log results
            success = results.get('success', False)
            if success:
                logger.info(f"Training session {session_id} completed successfully")
            else:
                logger.error(f"Training session {session_id} failed: {results.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Training session {session_id} crashed: {e}")
            traceback.print_exc()
            
        finally:
            # Cleanup session
            try:
                await session.cleanup()
            except Exception as e:
                logger.warning(f"Error during session cleanup {session_id}: {e}")
            
            # Remove from active sessions
            self.active_sessions.pop(session_id, None)
            
            # Start next queued training if available
            if not self._shutdown_requested and self.training_queue:
                next_config = self.training_queue.pop(0)
                await self.start_training(next_config)


# Global training manager instance
training_manager = TrainingManager()