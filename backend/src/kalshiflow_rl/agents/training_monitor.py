"""
Training monitoring and metrics tracking for Kalshi Flow RL Trading Subsystem.

Provides comprehensive training progress tracking, performance metrics calculation,
model checkpoint versioning, and training session lifecycle management.
Integrates with database for persistent monitoring and model registry for
automated model deployment based on performance thresholds.
"""

import asyncio
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
import numpy as np
from collections import deque, defaultdict

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

from ..config import config
from ..data.database import rl_db
from .model_registry import model_registry

logger = logging.getLogger("kalshiflow_rl.training_monitor")


@dataclass 
class PerformanceMetrics:
    """Structured performance metrics for training monitoring."""
    
    # Episode-level metrics
    total_episodes: int = 0
    avg_episode_reward: float = 0.0
    avg_episode_length: float = 0.0
    avg_portfolio_return: float = 0.0
    avg_sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    
    # Training efficiency metrics
    steps_per_second: float = 0.0
    training_duration: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Model performance metrics
    avg_policy_loss: float = 0.0
    avg_value_loss: float = 0.0
    avg_entropy: float = 0.0
    explained_variance: float = 0.0
    
    # Trading-specific metrics
    total_trades: int = 0
    avg_trade_return: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)
    training_started: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


class TrainingMonitor:
    """
    Comprehensive training monitoring system.
    
    Tracks training progress, calculates performance metrics,
    manages model checkpoints, and triggers automated actions
    based on performance thresholds.
    """
    
    def __init__(
        self,
        model_id: int,
        update_frequency: int = 100,
        checkpoint_frequency: int = 1000,
        max_metrics_history: int = 1000
    ):
        """
        Initialize training monitor.
        
        Args:
            model_id: Database model ID
            update_frequency: How often to update metrics (steps)
            checkpoint_frequency: How often to save checkpoints (steps)
            max_metrics_history: Maximum metrics history to keep in memory
        """
        self.model_id = model_id
        self.update_frequency = update_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.max_metrics_history = max_metrics_history
        
        # Current metrics
        self.current_metrics = PerformanceMetrics()
        self.current_metrics.training_started = datetime.utcnow()
        
        # Metrics history for trend analysis
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        
        # Episode tracking
        self.episode_rewards: deque = deque(maxlen=1000)
        self.episode_lengths: deque = deque(maxlen=1000)
        self.episode_returns: deque = deque(maxlen=1000)
        self.recent_losses: Dict[str, deque] = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100)
        }
        
        # Trading metrics
        self.trade_returns: deque = deque(maxlen=1000)
        self.portfolio_values: deque = deque(maxlen=1000)
        self.drawdown_history: deque = deque(maxlen=1000)
        
        # Internal state
        self._lock = threading.Lock()
        self._last_checkpoint_step = 0
        self._last_update_time = time.time()
        self._step_count = 0
        
        # Callbacks for performance thresholds
        self._threshold_callbacks: List[Tuple[str, float, Callable]] = []
        
        logger.info(f"Training monitor initialized for model {model_id}")
    
    def add_episode_data(
        self,
        episode_reward: float,
        episode_length: int,
        portfolio_return: float,
        total_trades: int = 0,
        episode_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add data from completed episode."""
        with self._lock:
            # Update episode tracking
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.episode_returns.append(portfolio_return)
            
            # Update trade tracking if available
            if episode_stats:
                trades_this_episode = episode_stats.get('total_trades', 0)
                self.current_metrics.total_trades += trades_this_episode
                
                # Track individual trade returns
                if 'trade_returns' in episode_stats:
                    self.trade_returns.extend(episode_stats['trade_returns'])
                
                # Track portfolio value
                portfolio_value = episode_stats.get('portfolio_value', 10000.0)
                self.portfolio_values.append(portfolio_value)
                
                # Calculate drawdown
                if len(self.portfolio_values) > 1:
                    peak = max(self.portfolio_values)
                    current_drawdown = (peak - portfolio_value) / peak
                    self.drawdown_history.append(current_drawdown)
            
            # Update metrics
            self._update_performance_metrics()
    
    def add_training_step_data(
        self,
        step: int,
        policy_loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        explained_variance: Optional[float] = None
    ) -> None:
        """Add data from training step."""
        with self._lock:
            self._step_count = step
            
            # Track losses
            if policy_loss is not None:
                self.recent_losses['policy_loss'].append(policy_loss)
            if value_loss is not None:
                self.recent_losses['value_loss'].append(value_loss)
            if entropy is not None:
                self.recent_losses['entropy'].append(entropy)
            
            # Update explained variance
            if explained_variance is not None:
                self.current_metrics.explained_variance = explained_variance
            
            # Check for metric updates
            if step % self.update_frequency == 0:
                self._update_performance_metrics()
                # Only create async tasks if we're in an async context
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(self._save_metrics_async())
                except RuntimeError:
                    # No event loop running, schedule for later or run synchronously
                    logger.debug("No event loop running, skipping async metrics save")
            
            # Check for checkpoints
            if step % self.checkpoint_frequency == 0:
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(self._trigger_checkpoint_async(step))
                except RuntimeError:
                    logger.debug(f"No event loop running, skipping async checkpoint for step {step}")
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            return self.current_metrics
    
    def get_metrics_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics history."""
        with self._lock:
            history = list(self.metrics_history)
            if last_n:
                history = history[-last_n:]
            return [m.to_dict() for m in history]
    
    def add_threshold_callback(
        self,
        metric_name: str,
        threshold_value: float,
        callback: Callable[[PerformanceMetrics], None]
    ) -> None:
        """
        Add callback to trigger when metric exceeds threshold.
        
        Args:
            metric_name: Name of metric to monitor
            threshold_value: Threshold value to trigger callback
            callback: Function to call when threshold is exceeded
        """
        self._threshold_callbacks.append((metric_name, threshold_value, callback))
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        with self._lock:
            metrics = self.current_metrics
            
            # Calculate training efficiency
            if metrics.training_started:
                duration = time.time() - metrics.training_started.timestamp()
                # Only use positive duration (avoid test timing issues)
                duration = max(0, duration)
            else:
                duration = 0
            steps_per_hour = (self._step_count / duration * 3600) if duration > 0 else 0
            
            # Calculate trend indicators
            recent_rewards = list(self.episode_rewards)[-100:] if len(self.episode_rewards) >= 100 else list(self.episode_rewards)
            reward_trend = np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0] if len(recent_rewards) > 1 else 0.0
            
            summary = {
                'model_id': self.model_id,
                'current_step': self._step_count,
                'training_duration_hours': duration / 3600,
                'steps_per_hour': steps_per_hour,
                
                # Performance
                'current_metrics': metrics.to_dict(),
                'reward_trend': float(reward_trend),
                
                # Progress indicators
                'total_episodes': len(self.episode_rewards),
                'recent_avg_reward': float(np.mean(recent_rewards)) if recent_rewards else 0.0,
                'best_episode_reward': float(max(self.episode_rewards)) if self.episode_rewards else 0.0,
                'worst_episode_reward': float(min(self.episode_rewards)) if self.episode_rewards else 0.0,
                
                # Trading performance
                'total_trades': metrics.total_trades,
                'best_portfolio_return': float(max(self.episode_returns)) if self.episode_returns else 0.0,
                'current_max_drawdown': metrics.max_drawdown,
                
                # Training stability
                'avg_policy_loss': metrics.avg_policy_loss,
                'avg_value_loss': metrics.avg_value_loss,
                'explained_variance': metrics.explained_variance,
                
                'last_updated': metrics.last_updated.isoformat()
            }
            
            return summary
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics from collected data."""
        current_time = time.time()
        
        # Episode-level metrics
        if self.episode_rewards:
            self.current_metrics.total_episodes = len(self.episode_rewards)
            self.current_metrics.avg_episode_reward = float(np.mean(self.episode_rewards))
            
        if self.episode_lengths:
            self.current_metrics.avg_episode_length = float(np.mean(self.episode_lengths))
            
        if self.episode_returns:
            self.current_metrics.avg_portfolio_return = float(np.mean(self.episode_returns))
            # Win rate: percentage of positive returns
            wins = sum(1 for r in self.episode_returns if r > 0)
            self.current_metrics.win_rate = wins / len(self.episode_returns)
        
        # Calculate Sharpe ratio approximation
        if len(self.episode_returns) > 1:
            returns_array = np.array(self.episode_returns)
            if np.std(returns_array) > 0:
                self.current_metrics.avg_sharpe_ratio = float(np.mean(returns_array) / np.std(returns_array))
        
        # Training efficiency
        time_elapsed = current_time - self._last_update_time
        if time_elapsed > 0:
            self.current_metrics.steps_per_second = self.update_frequency / time_elapsed
        
        # Only update training duration if it would be positive (training has started)
        if self.current_metrics.training_started:
            duration = current_time - self.current_metrics.training_started.timestamp()
            # Only update if positive (avoid test timing issues)
            if duration >= 0:
                self.current_metrics.training_duration = duration
        else:
            self.current_metrics.training_duration = 0
        
        # Loss metrics
        if self.recent_losses['policy_loss']:
            self.current_metrics.avg_policy_loss = float(np.mean(self.recent_losses['policy_loss']))
        if self.recent_losses['value_loss']:
            self.current_metrics.avg_value_loss = float(np.mean(self.recent_losses['value_loss']))
        if self.recent_losses['entropy']:
            self.current_metrics.avg_entropy = float(np.mean(self.recent_losses['entropy']))
        
        # Trading metrics
        if self.trade_returns:
            self.current_metrics.avg_trade_return = float(np.mean(self.trade_returns))
            
            # Profit factor: gross profit / gross loss
            profits = [r for r in self.trade_returns if r > 0]
            losses = [abs(r) for r in self.trade_returns if r < 0]
            if profits and losses:
                gross_profit = sum(profits)
                gross_loss = sum(losses)
                self.current_metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        if self.drawdown_history:
            self.current_metrics.max_drawdown = float(max(self.drawdown_history))
        
        # Update timestamp
        self.current_metrics.last_updated = datetime.utcnow()
        
        # Add to history
        self.metrics_history.append(self.current_metrics)
        
        # Check threshold callbacks
        self._check_threshold_callbacks()
        
        # Update timing
        self._last_update_time = current_time
    
    def _check_threshold_callbacks(self) -> None:
        """Check if any performance thresholds have been exceeded."""
        metrics_dict = self.current_metrics.to_dict()
        
        for metric_name, threshold, callback in self._threshold_callbacks:
            if metric_name in metrics_dict:
                current_value = metrics_dict[metric_name]
                if isinstance(current_value, (int, float)) and current_value >= threshold:
                    try:
                        callback(self.current_metrics)
                    except Exception as e:
                        logger.warning(f"Threshold callback failed for {metric_name}: {e}")
    
    async def _save_metrics_async(self) -> None:
        """Save current metrics to database asynchronously."""
        try:
            # Update model metrics in database
            training_metrics = self.current_metrics.to_dict()
            
            # Calculate validation metrics if we have enough data
            validation_metrics = {}
            if len(self.episode_returns) >= 10:
                recent_returns = list(self.episode_returns)[-10:]
                validation_metrics = {
                    'recent_avg_return': float(np.mean(recent_returns)),
                    'recent_std_return': float(np.std(recent_returns)),
                    'recent_win_rate': sum(1 for r in recent_returns if r > 0) / len(recent_returns),
                    'consistency_score': 1.0 - (np.std(recent_returns) / max(abs(np.mean(recent_returns)), 0.001))
                }
            
            await model_registry.update_model_metrics(
                self.model_id,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                auto_deploy=True  # Allow auto-deployment based on performance
            )
            
        except Exception as e:
            logger.warning(f"Failed to save metrics for model {self.model_id}: {e}")
    
    async def _trigger_checkpoint_async(self, step: int) -> None:
        """Trigger model checkpoint asynchronously."""
        try:
            # This would typically be handled by the training harness
            # but we log the checkpoint trigger here
            logger.info(f"Checkpoint triggered for model {self.model_id} at step {step}")
            
            # Record checkpoint in metrics
            self._last_checkpoint_step = step
            
        except Exception as e:
            logger.warning(f"Failed to trigger checkpoint for model {self.model_id}: {e}")


class TrainingProgressCallback(BaseCallback):
    """
    Stable Baselines3 callback that integrates with TrainingMonitor.
    
    Automatically feeds training data to the monitoring system.
    """
    
    def __init__(
        self,
        training_monitor: TrainingMonitor,
        verbose: int = 1
    ):
        """
        Initialize callback.
        
        Args:
            training_monitor: TrainingMonitor instance to feed data to
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.training_monitor = training_monitor
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        
    def _on_training_start(self) -> None:
        """Called when training starts."""
        logger.info("Training progress monitoring started")
    
    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Track episode progress
        self.current_episode_length += 1
        
        # Get current reward
        if len(self.locals.get('rewards', [])) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
        
        # Check if episode ended
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            self._on_episode_end()
        
        # Extract training losses if available
        policy_loss = None
        value_loss = None
        entropy = None
        explained_variance = None
        
        # Get losses from model logs (algorithm-specific)
        if hasattr(self.model, 'logger') and self.model.logger:
            try:
                # Try to get recent log entries
                if hasattr(self.model.logger, 'name_to_value'):
                    logs = self.model.logger.name_to_value
                    policy_loss = logs.get('train/policy_loss')
                    value_loss = logs.get('train/value_loss')
                    entropy = logs.get('train/entropy_loss')
                    explained_variance = logs.get('train/explained_variance')
            except:
                pass  # Logs might not be available yet
        
        # Feed data to monitor
        self.training_monitor.add_training_step_data(
            step=self.num_timesteps,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            explained_variance=explained_variance
        )
        
        return True
    
    def _on_episode_end(self) -> None:
        """Called when episode ends."""
        # Get episode statistics from environment if available
        episode_stats = {}
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]  # First environment in vector
        else:
            env = self.training_env
        
        if hasattr(env, 'get_episode_stats'):
            episode_stats = env.get_episode_stats()
        
        # Calculate portfolio return
        portfolio_return = 0.0
        if episode_stats:
            initial_value = 10000.0  # Default starting value
            final_value = episode_stats.get('portfolio_value', initial_value)
            portfolio_return = (final_value - initial_value) / initial_value
        
        # Feed episode data to monitor
        self.training_monitor.add_episode_data(
            episode_reward=self.current_episode_reward,
            episode_length=self.current_episode_length,
            portfolio_return=portfolio_return,
            episode_stats=episode_stats
        )
        
        # Reset episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_length = 0


def create_training_monitor(
    model_id: int,
    performance_targets: Optional[Dict[str, float]] = None
) -> Tuple[TrainingMonitor, TrainingProgressCallback]:
    """
    Create training monitor and callback for a training session.
    
    Args:
        model_id: Database model ID
        performance_targets: Optional performance targets for auto-deployment
        
    Returns:
        (TrainingMonitor, TrainingProgressCallback) tuple
    """
    # Create monitor
    monitor = TrainingMonitor(model_id)
    
    # Set up performance threshold callbacks
    if performance_targets:
        for metric_name, target_value in performance_targets.items():
            def create_threshold_callback(metric, target):
                async def callback(metrics: PerformanceMetrics):
                    logger.info(f"Performance target reached: {metric} = {getattr(metrics, metric, 0)} >= {target}")
                    # Trigger model activation
                    await model_registry.set_model_status(model_id, 'ready')
                return callback
            
            if hasattr(PerformanceMetrics(), metric_name):
                monitor.add_threshold_callback(
                    metric_name, 
                    target_value, 
                    create_threshold_callback(metric_name, target_value)
                )
    
    # Create callback
    callback = TrainingProgressCallback(monitor)
    
    return monitor, callback