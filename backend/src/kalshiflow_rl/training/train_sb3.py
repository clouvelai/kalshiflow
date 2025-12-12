#!/usr/bin/env python3
"""
SB3 Training Module for MarketAgnosticKalshiEnv.

This module provides a comprehensive training pipeline using Stable Baselines3
with enhanced portfolio metrics tracking and curriculum learning capabilities.

Usage:
    # Regular training on single session
    python train_sb3.py --session 9 --algorithm ppo --total-timesteps 100000
    
    # Curriculum training - each market gets full episode
    python train_sb3.py --session 9 --curriculum --algorithm ppo

Examples:
    $ python train_sb3.py --session 9 --algorithm ppo --total-timesteps 100000
    Trains PPO agent on session 9 data
    
    $ python train_sb3.py --session 9 --curriculum --algorithm ppo
    Trains PPO using curriculum learning on all viable markets in session 9
"""

import asyncio
import argparse
import os
import sys
import logging
import json
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# SB3 imports
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback, 
    StopTrainingOnRewardThreshold, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Our imports
from kalshiflow_rl.training.sb3_wrapper import (
    SessionBasedEnvironment, SB3TrainingConfig, CurriculumEnvironmentFactory,
    create_sb3_env, create_env_config, create_training_config
)
from kalshiflow_rl.training.simple_curriculum import SimpleMarketCurriculum
from kalshiflow_rl.environments.market_agnostic_env import EnvConfig
from kalshiflow_rl.diagnostics import M10DiagnosticsCallback


def generate_safe_model_path(session_ids: List[int], algorithm: str, is_curriculum: bool = False, 
                           base_dir: str = "trained_models") -> str:
    """
    Generate a safe, timestamped model save path to prevent overwriting checkpoints.
    
    Args:
        session_ids: List of session IDs being trained on
        algorithm: RL algorithm name (ppo, a2c)
        is_curriculum: Whether this is curriculum training
        base_dir: Base directory for saving models
        
    Returns:
        Safe path like: trained_models/session10_ppo_20251212_031500/model.zip
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_curriculum:
        if len(session_ids) == 1:
            dir_name = f"curriculum_session{session_ids[0]}_{algorithm}_{timestamp}"
        else:
            session_str = "_".join(map(str, session_ids))
            dir_name = f"curriculum_sessions{session_str}_{algorithm}_{timestamp}"
    else:
        if len(session_ids) == 1:
            dir_name = f"session{session_ids[0]}_{algorithm}_{timestamp}"
        else:
            session_str = "_".join(map(str, session_ids))
            dir_name = f"sessions{session_str}_{algorithm}_{timestamp}"
    
    safe_dir = Path(base_dir) / dir_name
    safe_dir.mkdir(parents=True, exist_ok=True)
    
    return str(safe_dir / "model.zip")


class PortfolioMetricsCallback(BaseCallback):
    """
    Enhanced callback to track portfolio metrics during training episodes.
    
    This callback captures portfolio dynamics DURING episodes (not just at end)
    by sampling portfolio state every N steps, providing better insight into
    trading behavior and portfolio fluctuations throughout episodes.
    """
    
    def __init__(self, 
                 log_freq: int = 1000, 
                 sample_freq: int = 100,
                 verbose: int = 0):
        """
        Initialize portfolio metrics callback.
        
        Args:
            log_freq: Frequency of episode summary logging (episodes)
            sample_freq: Frequency of portfolio sampling during episodes (steps)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        
        # Episode-level tracking
        self.episode_portfolios = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
        # Intra-episode tracking
        self.current_episode_samples = []
        self.current_episode_start_value = None
        self.step_count = 0
        
    def _on_step(self) -> bool:
        """Called on each environment step."""
        self.step_count += 1
        
        # Sample portfolio state periodically during episode
        if self.step_count % self.sample_freq == 0:
            self._sample_portfolio_state()
        
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            self._on_episode_end()
        
        return True
    
    def _sample_portfolio_state(self):
        """Sample current portfolio state during episode."""
        try:
            env = self.training_env.envs[0]
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'order_manager'):
                order_manager = env.unwrapped.order_manager
                
                # Get current portfolio metrics
                current_prices = env.unwrapped._get_current_market_prices()
                portfolio_value = order_manager.get_portfolio_value_cents(current_prices)
                cash_balance = order_manager.get_cash_balance_cents()
                position_info = order_manager.get_position_info()
                
                # Store sample point
                sample = {
                    'step': env.unwrapped.current_step,
                    'global_step': self.step_count,
                    'portfolio_value_cents': portfolio_value,
                    'cash_balance_cents': cash_balance,
                    'position_count': len(position_info),
                    'total_position_value_cents': sum(
                        pos.get('current_value_cents', 0) for pos in position_info.values()
                    ),
                    'timestamp': time.time()
                }
                
                self.current_episode_samples.append(sample)
                
                # Record episode start value for reference
                if self.current_episode_start_value is None:
                    self.current_episode_start_value = portfolio_value
                    
        except Exception as e:
            # Don't fail training due to metrics collection issues
            if self.verbose > 0:
                print(f"Warning: Failed to sample portfolio state: {e}")
    
    def _on_episode_end(self):
        """Process episode completion and extract portfolio dynamics."""
        self.episode_count += 1
        
        try:
            env = self.training_env.envs[0]
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'order_manager'):
                order_manager = env.unwrapped.order_manager
                
                # Get final portfolio metrics
                current_prices = env.unwrapped._get_current_market_prices()
                final_portfolio_value = order_manager.get_portfolio_value_cents(current_prices)
                cash_balance = order_manager.get_cash_balance_cents()
                position_info = order_manager.get_position_info()
                
                # Analyze portfolio dynamics during episode
                portfolio_dynamics = self._analyze_episode_dynamics()
                
                # Complete episode metrics
                episode_metrics = {
                    'episode': self.episode_count,
                    'episode_length': env.unwrapped.current_step,
                    'market_ticker': getattr(env.unwrapped, 'current_market', 'unknown'),
                    
                    # Final state
                    'final_portfolio_value_cents': final_portfolio_value,
                    'final_cash_balance_cents': cash_balance,
                    'position_count': len(position_info),
                    'total_position_value_cents': sum(
                        pos.get('current_value_cents', 0) for pos in position_info.values()
                    ),
                    
                    # Episode dynamics
                    'portfolio_dynamics': portfolio_dynamics,
                    
                    # Sample metadata
                    'samples_collected': len(self.current_episode_samples),
                    'sample_frequency': self.sample_freq
                }
                
                self.episode_portfolios.append(episode_metrics)
                
                # Log summary statistics periodically
                if self.episode_count % self.log_freq == 0:
                    self._log_portfolio_summary()
            
        except Exception as e:
            # Don't fail training due to metrics collection issues
            if self.verbose > 0:
                print(f"Warning: Failed to process episode end: {e}")
        
        finally:
            # Reset episode tracking
            self._reset_episode_tracking()
    
    def _analyze_episode_dynamics(self) -> Dict[str, Any]:
        """Analyze portfolio dynamics during the completed episode."""
        if not self.current_episode_samples:
            return {}
        
        # Extract portfolio values from samples
        portfolio_values = [s['portfolio_value_cents'] for s in self.current_episode_samples]
        
        if not portfolio_values:
            return {}
        
        # Basic statistics
        dynamics = {
            'min_value_cents': float(np.min(portfolio_values)),
            'max_value_cents': float(np.max(portfolio_values)),
            'start_value_cents': portfolio_values[0],
            'end_value_cents': portfolio_values[-1],
            'value_range_cents': float(np.max(portfolio_values) - np.min(portfolio_values)),
            'volatility': float(np.std(portfolio_values)) if len(portfolio_values) > 1 else 0.0,
            'mean_value_cents': float(np.mean(portfolio_values)),
        }
        
        # Calculate changes
        if len(portfolio_values) > 1:
            changes = np.diff(portfolio_values)
            dynamics.update({
                'total_change_cents': float(portfolio_values[-1] - portfolio_values[0]),
                'max_gain_cents': float(np.max(changes)) if len(changes) > 0 else 0.0,
                'max_loss_cents': float(np.min(changes)) if len(changes) > 0 else 0.0,
                'positive_changes': int(np.sum(changes > 0)),
                'negative_changes': int(np.sum(changes < 0)),
                'zero_changes': int(np.sum(changes == 0))
            })
        else:
            dynamics.update({
                'total_change_cents': 0.0,
                'max_gain_cents': 0.0,
                'max_loss_cents': 0.0,
                'positive_changes': 0,
                'negative_changes': 0,
                'zero_changes': 0
            })
        
        return dynamics
    
    def _reset_episode_tracking(self):
        """Reset tracking for next episode."""
        self.current_episode_samples = []
        self.current_episode_start_value = None
    
    def _log_portfolio_summary(self):
        """Log portfolio performance summary."""
        if not self.episode_portfolios:
            return
        
        recent_episodes = self.episode_portfolios[-self.log_freq:]
        
        # Extract values for analysis
        final_values = [ep['final_portfolio_value_cents'] for ep in recent_episodes]
        episode_lengths = [ep['episode_length'] for ep in recent_episodes]
        
        # Analyze dynamics across episodes
        dynamics_data = [ep['portfolio_dynamics'] for ep in recent_episodes if ep['portfolio_dynamics']]
        
        if dynamics_data:
            total_changes = [d.get('total_change_cents', 0) for d in dynamics_data]
            volatilities = [d.get('volatility', 0) for d in dynamics_data]
            value_ranges = [d.get('value_range_cents', 0) for d in dynamics_data]
            
            dynamics_summary = {
                'avg_episode_change_cents': np.mean(total_changes),
                'avg_volatility': np.mean(volatilities),
                'avg_value_range_cents': np.mean(value_ranges),
                'episodes_with_positive_change': sum(1 for c in total_changes if c > 0),
                'episodes_with_negative_change': sum(1 for c in total_changes if c < 0),
                'episodes_with_no_change': sum(1 for c in total_changes if c == 0)
            }
        else:
            dynamics_summary = {}
        
        # Overall summary
        summary = {
            'episodes_completed': len(recent_episodes),
            'avg_final_portfolio_value_cents': np.mean(final_values),
            'std_final_portfolio_value_cents': np.std(final_values),
            'min_final_portfolio_value_cents': np.min(final_values),
            'max_final_portfolio_value_cents': np.max(final_values),
            'avg_episode_length': np.mean(episode_lengths),
            'unique_markets_trained': len(set(ep['market_ticker'] for ep in recent_episodes)),
            **dynamics_summary
        }
        
        # Log summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Portfolio Summary (last {len(recent_episodes)} episodes):")
        self.logger.info(f"  Final portfolio values:")
        self.logger.info(f"    Average: {summary['avg_final_portfolio_value_cents']:.2f} cents")
        self.logger.info(f"    Range: {summary['min_final_portfolio_value_cents']:.2f} - {summary['max_final_portfolio_value_cents']:.2f}")
        self.logger.info(f"    Std dev: {summary['std_final_portfolio_value_cents']:.2f}")
        
        if dynamics_summary:
            self.logger.info(f"  Episode dynamics:")
            self.logger.info(f"    Avg change per episode: {dynamics_summary['avg_episode_change_cents']:.2f} cents")
            self.logger.info(f"    Avg volatility: {dynamics_summary['avg_volatility']:.2f}")
            self.logger.info(f"    Avg intra-episode range: {dynamics_summary['avg_value_range_cents']:.2f} cents")
            self.logger.info(f"    Episodes with gains: {dynamics_summary['episodes_with_positive_change']}")
            self.logger.info(f"    Episodes with losses: {dynamics_summary['episodes_with_negative_change']}")
        
        self.logger.info(f"  Training efficiency:")
        self.logger.info(f"    Avg episode length: {summary['avg_episode_length']:.1f} steps")
        self.logger.info(f"    Markets trained on: {summary['unique_markets_trained']}")
        self.logger.info(f"{'='*60}")
        
        # Log to tensorboard if available
        if hasattr(self, 'logger') and hasattr(self.logger, 'record'):
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    self.logger.record(f"portfolio/{key}", value)
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get complete episode statistics including dynamics analysis."""
        if not self.episode_portfolios:
            return {}
        
        final_values = [ep['final_portfolio_value_cents'] for ep in self.episode_portfolios]
        episode_lengths = [ep['episode_length'] for ep in self.episode_portfolios]
        
        # Analyze dynamics across all episodes
        dynamics_data = [ep['portfolio_dynamics'] for ep in self.episode_portfolios if ep['portfolio_dynamics']]
        
        dynamics_stats = {}
        if dynamics_data:
            total_changes = [d.get('total_change_cents', 0) for d in dynamics_data]
            volatilities = [d.get('volatility', 0) for d in dynamics_data]
            value_ranges = [d.get('value_range_cents', 0) for d in dynamics_data]
            
            dynamics_stats = {
                'total_change_stats': {
                    'mean': np.mean(total_changes),
                    'std': np.std(total_changes),
                    'min': np.min(total_changes),
                    'max': np.max(total_changes)
                },
                'volatility_stats': {
                    'mean': np.mean(volatilities),
                    'std': np.std(volatilities),
                    'min': np.min(volatilities),
                    'max': np.max(volatilities)
                },
                'value_range_stats': {
                    'mean': np.mean(value_ranges),
                    'std': np.std(value_ranges),
                    'min': np.min(value_ranges),
                    'max': np.max(value_ranges)
                },
                'trading_outcomes': {
                    'positive_episodes': sum(1 for c in total_changes if c > 0),
                    'negative_episodes': sum(1 for c in total_changes if c < 0),
                    'neutral_episodes': sum(1 for c in total_changes if c == 0),
                    'win_rate': sum(1 for c in total_changes if c > 0) / len(total_changes) if total_changes else 0.0
                }
            }
        
        return {
            'total_episodes': len(self.episode_portfolios),
            'portfolio_stats': {
                'mean': np.mean(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values),
                'median': np.median(final_values)
            },
            'episode_length_stats': {
                'mean': np.mean(episode_lengths),
                'std': np.std(episode_lengths),
                'min': np.min(episode_lengths),
                'max': np.max(episode_lengths)
            },
            'unique_markets': len(set(ep['market_ticker'] for ep in self.episode_portfolios)),
            'episodes_by_market': {
                market: sum(1 for ep in self.episode_portfolios if ep['market_ticker'] == market)
                for market in set(ep['market_ticker'] for ep in self.episode_portfolios)
            },
            'dynamics_statistics': dynamics_stats,
            'sample_metadata': {
                'sample_frequency': self.sample_freq,
                'total_samples_collected': sum(ep['samples_collected'] for ep in self.episode_portfolios),
                'avg_samples_per_episode': np.mean([ep['samples_collected'] for ep in self.episode_portfolios])
            }
        }


class TrainingProgressMonitor:
    """
    Monitor training progress with comprehensive metrics tracking.
    
    Tracks episode rewards, portfolio performance, session completion statistics,
    and training efficiency metrics.
    """
    
    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.training_start = datetime.now()
        self.session_stats = {}
        self.training_metrics = {
            'timesteps_trained': 0,
            'episodes_completed': 0,
            'total_reward': 0.0,
            'best_episode_reward': -float('inf'),
            'worst_episode_reward': float('inf'),
            'sessions_trained': set(),
            'markets_encountered': set()
        }
        
    def update_metrics(self, timestep: int, episode_reward: float, 
                      session_id: int, market_ticker: str):
        """Update training metrics with latest episode data."""
        self.training_metrics['timesteps_trained'] = timestep
        self.training_metrics['episodes_completed'] += 1
        self.training_metrics['total_reward'] += episode_reward
        self.training_metrics['best_episode_reward'] = max(
            self.training_metrics['best_episode_reward'], episode_reward
        )
        self.training_metrics['worst_episode_reward'] = min(
            self.training_metrics['worst_episode_reward'], episode_reward
        )
        self.training_metrics['sessions_trained'].add(session_id)
        self.training_metrics['markets_encountered'].add(market_ticker)
        
        # Update session-specific stats
        if session_id not in self.session_stats:
            self.session_stats[session_id] = {
                'episodes': 0,
                'total_reward': 0.0,
                'markets': set()
            }
        
        self.session_stats[session_id]['episodes'] += 1
        self.session_stats[session_id]['total_reward'] += episode_reward
        self.session_stats[session_id]['markets'].add(market_ticker)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive training progress summary."""
        training_duration = datetime.now() - self.training_start
        
        # Convert sets to counts for JSON serialization
        metrics = self.training_metrics.copy()
        metrics['sessions_trained'] = len(metrics['sessions_trained'])
        metrics['markets_encountered'] = len(metrics['markets_encountered'])
        
        # Calculate averages
        if metrics['episodes_completed'] > 0:
            avg_reward = metrics['total_reward'] / metrics['episodes_completed']
        else:
            avg_reward = 0.0
        
        # Session stats for JSON
        session_stats_serializable = {}
        for session_id, stats in self.session_stats.items():
            session_stats_serializable[str(session_id)] = {
                'episodes': stats['episodes'],
                'total_reward': stats['total_reward'],
                'avg_reward': stats['total_reward'] / stats['episodes'] if stats['episodes'] > 0 else 0.0,
                'markets_count': len(stats['markets'])
            }
        
        return {
            'training_duration_seconds': training_duration.total_seconds(),
            'training_duration_str': str(training_duration),
            'timesteps_per_second': (
                metrics['timesteps_trained'] / training_duration.total_seconds()
                if training_duration.total_seconds() > 0 else 0
            ),
            'episodes_per_hour': (
                metrics['episodes_completed'] / (training_duration.total_seconds() / 3600)
                if training_duration.total_seconds() > 0 else 0
            ),
            'avg_episode_reward': avg_reward,
            'training_metrics': metrics,
            'session_stats': session_stats_serializable
        }
    
    def save_progress(self):
        """Save progress summary to file."""
        summary = self.get_progress_summary()
        
        with open(self.save_path, 'w') as f:
            json.dump(summary, f, indent=2)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )


def create_model(algorithm: str, env, model_params: Dict[str, Any], device: str = "auto"):
    """
    Create SB3 model with specified algorithm and parameters.
    
    Args:
        algorithm: 'ppo' or 'a2c'
        env: Training environment
        model_params: Algorithm-specific parameters
        device: Device to use ('auto', 'cpu', 'cuda')
        
    Returns:
        Initialized SB3 model
    """
    algorithm = algorithm.lower()
    
    if algorithm == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=env,
            device=device,
            **model_params
        )
    elif algorithm == "a2c":
        return A2C(
            policy="MlpPolicy", 
            env=env,
            device=device,
            **model_params
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'ppo' or 'a2c'")


def get_default_model_params(algorithm: str) -> Dict[str, Any]:
    """Get default parameters for SB3 algorithms - tuned for trading."""
    if algorithm.lower() == "ppo":
        return {
            "learning_rate": 1e-4,  # Reduced from 3e-4 for more stable updates
            "n_steps": 4096,  # Increased from 2048 for better advantage estimation
            "batch_size": 256,  # Increased from 64 for more stable gradients
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.05,  # Increased from 0.01 to encourage exploration
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1
        }
    elif algorithm.lower() == "a2c":
        return {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.25,
            "max_grad_norm": 0.5,
            "verbose": 1
        }
    else:
        return {}


def create_callbacks(model_save_path: str, 
                    save_freq: int = 10000,
                    eval_freq: int = 5000,
                    eval_env = None,
                    portfolio_log_freq: int = 1000,
                    portfolio_sample_freq: int = 100,
                    # M10 Diagnostics parameters
                    enable_m10_diagnostics: bool = True,
                    session_id: Optional[int] = None,
                    algorithm: str = "unknown",
                    m10_console_freq: int = 500) -> CallbackList:
    """
    Create training callbacks for monitoring and checkpointing.
    
    Args:
        model_save_path: Base path for saving models
        save_freq: Frequency of model checkpointing (timesteps)
        eval_freq: Frequency of model evaluation (timesteps)
        eval_env: Environment for evaluation (optional)
        portfolio_log_freq: Frequency of portfolio metrics logging (episodes)
        portfolio_sample_freq: Frequency of portfolio sampling during episodes (steps)
        enable_m10_diagnostics: Enable M10 comprehensive diagnostics
        session_id: Session ID for organized diagnostics output
        algorithm: Algorithm name for diagnostics organization
        
    Returns:
        CallbackList with configured callbacks
    """
    callbacks = []
    
    # M10 Diagnostics callback (highest priority for HOLD behavior analysis)
    if enable_m10_diagnostics:
        models_dir = Path(model_save_path).parent
        m10_callback = M10DiagnosticsCallback(
            output_dir=str(models_dir),
            session_id=session_id,
            algorithm=algorithm,
            action_tracking=True,
            reward_analysis=True,
            observation_validation=True,
            console_output=True,
            detailed_logging=True,
            step_log_freq=1000,
            episode_log_freq=100,
            validation_freq=100,
            console_summary_freq=m10_console_freq,
            verbose=1
        )
        callbacks.append(m10_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(Path(model_save_path).parent / "checkpoints"),
        name_prefix=Path(model_save_path).stem
    )
    callbacks.append(checkpoint_callback)
    
    # Enhanced portfolio metrics callback (for backward compatibility)
    portfolio_callback = PortfolioMetricsCallback(
        log_freq=portfolio_log_freq,
        sample_freq=portfolio_sample_freq
    )
    callbacks.append(portfolio_callback)
    
    # Evaluation callback (if eval environment provided)
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            best_model_save_path=str(Path(model_save_path).parent / "best_model"),
            log_path=str(Path(model_save_path).parent / "eval_logs"),
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
    
    return CallbackList(callbacks)


async def train_with_curriculum(args) -> Dict[str, Any]:
    """
    Train model using SimpleMarketCurriculum - trains on each viable market once.
    
    Args:
        args: Parsed command-line arguments with curriculum enabled
        
    Returns:
        Dictionary with curriculum training results
    """
    # Setup logging
    log_file = f"curriculum_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log" if args.log_file else None
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SB3 curriculum training pipeline...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set in environment")
    
    # Initialize curriculum
    curriculum = SimpleMarketCurriculum(
        database_url=database_url,
        session_id=args.session,
        min_timesteps=args.min_episode_length
    )
    await curriculum.initialize()
    
    logger.info(f"Curriculum initialized: {len(curriculum.viable_markets)} viable markets")
    
    # Get model parameters
    model_params = get_default_model_params(args.algorithm)
    if args.learning_rate:
        model_params['learning_rate'] = args.learning_rate
    
    # Initialize model - either load from checkpoint or create later
    model = None
    
    # Load pre-trained model if specified
    if args.from_model_checkpoint and Path(args.from_model_checkpoint).exists():
        logger.info(f"Loading pre-trained model from: {args.from_model_checkpoint}")
        logger.info("Note: Using pre-trained policy for curriculum learning on new session")
        # Create dummy env to load model (will be replaced with actual market env)
        from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv
        # Get first viable market to create dummy env
        first_market_ticker = curriculum.viable_markets[0][0] if curriculum.viable_markets else None
        if first_market_ticker:
            first_market_view = curriculum.create_market_view(first_market_ticker)
            dummy_env = DummyVecEnv([lambda: Monitor(MarketAgnosticKalshiEnv(
                market_view=first_market_view,
                config=create_env_config(cash_start=args.cash_start, max_markets=1, temporal_features=True)
            ), filename=None)])
            
            if args.algorithm.lower() == "ppo":
                model = PPO.load(args.from_model_checkpoint, env=dummy_env, device=args.device,
                               custom_objects={'learning_rate': model_params.get('learning_rate', 3e-4)})
            else:  # a2c
                model = A2C.load(args.from_model_checkpoint, env=dummy_env, device=args.device,
                               custom_objects={'learning_rate': model_params.get('learning_rate', 7e-4)})
            model.num_timesteps = 0  # Reset timestep counter for fresh curriculum
            logger.info(f"Pre-trained model loaded, starting curriculum on session {args.session}")
        else:
            logger.warning("No viable markets found, cannot load pre-trained model")
    
    # Setup model save paths
    models_dir = Path(args.model_save_path).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Track curriculum results
    curriculum_results = {
        'session_id': args.session,
        'total_markets': len(curriculum.viable_markets),
        'markets_trained': [],
        'market_results': {},
        'training_start': datetime.now(),
        'algorithm': args.algorithm,
        'total_timesteps_per_market': args.total_timesteps or 'full_episode'
    }
    
    # Train on each market exactly once
    market_ticker = curriculum.get_next_market()
    market_count = 0
    
    while market_ticker is not None:
        market_count += 1
        market_timesteps = curriculum.viable_markets[curriculum.current_market_index][1]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING MARKET {market_count}/{len(curriculum.viable_markets)}: {market_ticker}")
        logger.info(f"Market has {market_timesteps} timesteps")
        logger.info(f"{'='*80}")
        
        try:
            # Create market view
            market_view = curriculum.create_market_view(market_ticker)
            if market_view is None:
                logger.error(f"Failed to create market view for {market_ticker}")
                curriculum.advance_to_next_market()
                market_ticker = curriculum.get_next_market()
                continue
            
            # Create environment for this market
            env_config = create_env_config(
                cash_start=args.cash_start,
                max_markets=1,
                temporal_features=True
            )
            
            # Create MarketAgnosticKalshiEnv with this market view
            from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv
            
            market_env = MarketAgnosticKalshiEnv(
                market_view=market_view,
                config=env_config
            )
            
            # Wrap and vectorize
            market_env = Monitor(market_env, filename=None)
            vec_env = DummyVecEnv([lambda: market_env])
            
            # Create or reuse model
            if model is None:
                logger.info(f"Creating initial {args.algorithm.upper()} model...")
                model = create_model(args.algorithm, vec_env, model_params, args.device)
            else:
                logger.info(f"Reusing existing {args.algorithm.upper()} model...")
                # Update environment for existing model
                model.set_env(vec_env)
            
            # Create callbacks for this market
            market_callbacks = create_callbacks(
                model_save_path=str(models_dir / f"market_{market_count}_{market_ticker}_model.zip"),
                save_freq=args.save_freq,
                eval_freq=args.eval_freq,
                portfolio_log_freq=args.portfolio_log_freq,
                portfolio_sample_freq=args.portfolio_sample_freq,
                enable_m10_diagnostics=not args.disable_m10_diagnostics,
                session_id=args.session,
                algorithm=args.algorithm,
                m10_console_freq=args.m10_console_freq
            )
            
            # Determine timesteps for this market (full episode or user override)
            if args.total_timesteps and args.total_timesteps > 0:
                # User specified limit per market
                timesteps_to_train = args.total_timesteps
                logger.info(f"Training on {market_ticker} for {timesteps_to_train} timesteps (user override)...")
            else:
                # Use full episode length
                timesteps_to_train = market_timesteps
                logger.info(f"Training on {market_ticker} for {timesteps_to_train} timesteps (full episode)...")
            
            training_start = time.time()
            
            model.learn(
                total_timesteps=timesteps_to_train,
                callback=market_callbacks,
                log_interval=args.log_interval,
                reset_num_timesteps=False  # Keep accumulating timesteps across markets
            )
            
            training_duration = time.time() - training_start
            
            # Get portfolio callback for statistics
            portfolio_callback = next(
                (cb for cb in market_callbacks.callbacks if isinstance(cb, PortfolioMetricsCallback)),
                None
            )
            
            # Store market results
            market_result = {
                'market_ticker': market_ticker,
                'market_timesteps': market_timesteps,
                'training_duration_seconds': training_duration,
                'timesteps_trained': timesteps_to_train,
                'timesteps_per_second': timesteps_to_train / training_duration,
                'full_episode': timesteps_to_train == market_timesteps,
                'portfolio_statistics': portfolio_callback.get_episode_statistics() if portfolio_callback else {}
            }
            
            curriculum_results['markets_trained'].append(market_ticker)
            curriculum_results['market_results'][market_ticker] = market_result
            
            logger.info(f"✅ Training completed for {market_ticker} in {training_duration:.2f}s")
            
            # Close environment
            vec_env.close()
            
        except Exception as e:
            logger.error(f"❌ Training failed for {market_ticker}: {e}")
            curriculum_results['market_results'][market_ticker] = {'error': str(e)}
            import traceback
            traceback.print_exc()
        
        # Move to next market
        curriculum.advance_to_next_market()
        market_ticker = curriculum.get_next_market()
    
    # Save final model after all markets
    if model is not None:
        # Auto-generate safe save path if loading from checkpoint
        final_model_save_path = args.model_save_path
        if args.from_model_checkpoint:
            # Check if user explicitly provided a different save path
            default_save_path = 'trained_models/trained_model.zip'
            
            if args.model_save_path == default_save_path:
                # User didn't override save path, so generate a safe one for curriculum
                final_model_save_path = generate_safe_model_path(
                    session_ids=[args.session], 
                    algorithm=args.algorithm, 
                    is_curriculum=True
                )
                logger.info(f"Auto-generated safe save path for curriculum training:")
                logger.info(f"  Source checkpoint: {args.from_model_checkpoint}")
                logger.info(f"  New save location: {final_model_save_path}")
            else:
                logger.info(f"Using user-specified save path: {final_model_save_path}")
        
        logger.info(f"Saving final curriculum-trained model to {final_model_save_path}")
        model.save(final_model_save_path)
        curriculum_results['final_model_path'] = final_model_save_path
    
    # Finalize results
    curriculum_results['training_end'] = datetime.now()
    curriculum_results['total_duration'] = (
        curriculum_results['training_end'] - curriculum_results['training_start']
    ).total_seconds()
    
    # Save curriculum results
    results_path = models_dir / "curriculum_training_results.json"
    with open(results_path, 'w') as f:
        json.dump(curriculum_results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"CURRICULUM TRAINING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Session {args.session}: Trained on {len(curriculum_results['markets_trained'])} markets")
    logger.info(f"Total duration: {curriculum_results['total_duration']:.2f} seconds")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Final model saved to: {curriculum_results['final_model_path']}")
    
    return curriculum_results


async def train_model(args) -> Dict[str, Any]:
    """
    Main training function that coordinates the entire training pipeline.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary with training results and statistics
    """
    # Setup logging
    log_file = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log" if args.log_file else None
    setup_logging(args.log_level, log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting SB3 training pipeline...")
    logger.info(f"Arguments: {vars(args)}")
    
    # Get database URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not set in environment")
    
    # Parse session IDs
    if args.sessions:
        session_ids = [int(x.strip()) for x in args.sessions.split(',')]
    else:
        session_ids = [args.session]  # Make it a list for consistency
    
    logger.info(f"Training on sessions: {session_ids}")
    
    # Create training configuration
    env_config = create_env_config(
        cash_start=args.cash_start,
        max_markets=1,
        temporal_features=True
    )
    
    training_config = create_training_config(
        min_episode_length=args.min_episode_length,
        max_episode_steps=None,  # Always None - episodes run to completion
        skip_failed_markets=True
    )
    
    # Create environment
    logger.info("Creating training environment...")
    
    # Combine configs properly
    full_config = SB3TrainingConfig(
        env_config=env_config,
        min_episode_length=training_config.min_episode_length,
        max_episode_steps=None,  # Force no artificial limits - episodes run to completion
        skip_failed_markets=training_config.skip_failed_markets
    )
    
    env = await create_sb3_env(
        database_url=database_url,
        session_ids=session_ids,
        config=full_config
    )
    
    # Wrap environment for monitoring
    env = Monitor(env, filename=None)  # Monitor without file logging
    vec_env = DummyVecEnv([lambda: env])
    
    # Get model parameters
    model_params = get_default_model_params(args.algorithm)
    if args.learning_rate:
        model_params['learning_rate'] = args.learning_rate
    
    # Create or load model
    logger.info(f"Creating {args.algorithm.upper()} model...")
    
    # Handle model loading with two different modes
    if args.resume_from and Path(args.resume_from).exists():
        # Resume training (same session/environment) - preserves optimizer state
        logger.info(f"Resuming training from checkpoint: {args.resume_from}")
        if args.algorithm.lower() == "ppo":
            model = PPO.load(args.resume_from, env=vec_env, device=args.device)
        else:  # a2c
            model = A2C.load(args.resume_from, env=vec_env, device=args.device)
    elif args.from_model_checkpoint and Path(args.from_model_checkpoint).exists():
        # Transfer learning (different session/environment) - loads policy but resets optimizer
        logger.info(f"Loading pre-trained model from: {args.from_model_checkpoint}")
        logger.info("Note: Transferring learned policy to new environment, optimizer state reset")
        if args.algorithm.lower() == "ppo":
            model = PPO.load(args.from_model_checkpoint, env=vec_env, device=args.device, 
                           custom_objects={'learning_rate': model_params.get('learning_rate', 3e-4)})
        else:  # a2c
            model = A2C.load(args.from_model_checkpoint, env=vec_env, device=args.device,
                           custom_objects={'learning_rate': model_params.get('learning_rate', 7e-4)})
        # Reset num timesteps for fresh training on new session
        model.num_timesteps = 0
        logger.info(f"Model loaded successfully, starting fresh training on session(s): {session_ids}")
    else:
        model = create_model(args.algorithm, vec_env, model_params, args.device)
    
    # Auto-generate safe save path if loading from checkpoint
    final_model_save_path = args.model_save_path
    if args.resume_from or args.from_model_checkpoint:
        # Check if user explicitly provided a different save path
        parser_defaults = argparse.ArgumentParser().parse_args([])
        default_save_path = 'trained_models/trained_model.zip'
        
        if args.model_save_path == default_save_path:
            # User didn't override save path, so generate a safe one
            checkpoint_source = args.resume_from if args.resume_from else args.from_model_checkpoint
            final_model_save_path = generate_safe_model_path(
                session_ids=session_ids, 
                algorithm=args.algorithm, 
                is_curriculum=False
            )
            logger.info(f"Auto-generated safe save path to prevent overwriting checkpoint:")
            logger.info(f"  Source checkpoint: {checkpoint_source}")
            logger.info(f"  New save location: {final_model_save_path}")
        else:
            logger.info(f"Using user-specified save path: {final_model_save_path}")
    
    # Setup model save paths
    models_dir = Path(final_model_save_path).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create progress monitor
    progress_monitor = TrainingProgressMonitor(
        str(models_dir / "training_progress.json")
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        model_save_path=final_model_save_path,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        portfolio_log_freq=args.portfolio_log_freq,
        portfolio_sample_freq=args.portfolio_sample_freq,
        enable_m10_diagnostics=not args.disable_m10_diagnostics,
        session_id=session_ids[0] if len(session_ids) == 1 else None,  # Single session for diagnostics
        algorithm=args.algorithm,
        m10_console_freq=args.m10_console_freq
    )
    
    # Training loop
    logger.info(f"Starting training for {args.total_timesteps} timesteps...")
    training_start = time.time()
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
            reset_num_timesteps=not bool(args.resume_from)
        )
        
        training_duration = time.time() - training_start
        logger.info(f"Training completed in {training_duration:.2f} seconds")
        
        # Save final model
        logger.info(f"Saving final model to {final_model_save_path}")
        model.save(final_model_save_path)
        
        # Get portfolio callback for statistics
        portfolio_callback = next(
            (cb for cb in callbacks.callbacks if isinstance(cb, PortfolioMetricsCallback)),
            None
        )
        
        # Compile training results
        training_results = {
            'training_completed': True,
            'total_timesteps': args.total_timesteps,
            'training_duration_seconds': training_duration,
            'timesteps_per_second': args.total_timesteps / training_duration,
            'algorithm': args.algorithm,
            'session_ids': session_ids,
            'final_model_path': final_model_save_path,
            'environment_info': env.unwrapped.get_market_rotation_info() if hasattr(env.unwrapped, 'get_market_rotation_info') else {},
            'portfolio_statistics': portfolio_callback.get_episode_statistics() if portfolio_callback else {},
            'progress_summary': progress_monitor.get_progress_summary()
        }
        
        # Save training results
        results_path = models_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_results, f, indent=2, default=str)
        
        logger.info(f"Training results saved to {results_path}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        env.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Train MarketAgnosticKalshiEnv with Stable Baselines3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regular training with M10 diagnostics (default)
  python train_sb3.py --session 9 --algorithm ppo --total-timesteps 10000
  
  # Curriculum training - each market gets its FULL episode length
  python train_sb3.py --session 9 --curriculum --algorithm ppo
  
  # Training with M10 diagnostics disabled for speed
  python train_sb3.py --session 9 --algorithm ppo --disable-m10-diagnostics
  
  # Training with custom M10 console frequency
  python train_sb3.py --session 9 --algorithm ppo --m10-console-freq 1000
  
  # Multiple sessions for regular training
  python train_sb3.py --sessions 6,7,8,9 --algorithm a2c --total-timesteps 50000
        """
    )
    
    # Session configuration
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument('--session', type=int,
                              help='Single session ID to train on')
    session_group.add_argument('--sessions', type=str,
                              help='Comma-separated list of session IDs for curriculum learning')
    
    # Algorithm configuration
    parser.add_argument('--algorithm', choices=['ppo', 'a2c'], default='ppo',
                       help='RL algorithm to use (default: ppo)')
    parser.add_argument('--total-timesteps', type=int, default=None,
                       help='Training timesteps per market (default: None = full episode length). Use for testing/debugging only.')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate (uses algorithm default if not specified)')
    
    # Curriculum learning
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning: train on each viable market in session exactly once')
    
    # Environment configuration
    parser.add_argument('--cash-start', type=int, default=10000,
                       help='Starting cash in cents (default: 10000 = $100)')
    parser.add_argument('--min-episode-length', type=int, default=10,
                       help='Minimum episode length for valid markets (default: 10)')
    
    # Model persistence
    parser.add_argument('--model-save-path', default='trained_models/trained_model.zip',
                       help='Path to save final trained model (default: trained_models/trained_model.zip)')
    parser.add_argument('--resume-from', type=str,
                       help='Path to model checkpoint to resume training from (same session/environment)')
    parser.add_argument('--from-model-checkpoint', type=str,
                       help='Path to model checkpoint to continue training from (different session/environment). ' + 
                            'Use this to transfer learned knowledge from one session to another.')
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='Frequency of model checkpointing in timesteps (default: 10000)')
    
    # Evaluation and monitoring
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Frequency of model evaluation in timesteps (default: 5000)')
    parser.add_argument('--portfolio-log-freq', type=int, default=1000,
                       help='Frequency of portfolio metrics logging in episodes (default: 1000)')
    parser.add_argument('--portfolio-sample-freq', type=int, default=1,
                       help='Frequency of portfolio sampling during episodes in steps (default: 100)')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Frequency of training log output in timesteps (default: 100)')
    
    # M10 Diagnostics
    parser.add_argument('--disable-m10-diagnostics', action='store_true',
                       help='Disable M10 comprehensive diagnostics (action/reward/observation tracking)')
    parser.add_argument('--m10-console-freq', type=int, default=500,
                       help='Frequency of M10 console diagnostic summaries in steps (default: 500)')
    
    # Technical configuration  
    parser.add_argument('--device', default='auto',
                       help='Device for training: auto, cpu, or cuda (default: auto)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str,
                       help='Optional log file path (logs to console if not specified)')
    
    args = parser.parse_args()
    
    # Validate curriculum arguments
    if args.curriculum:
        if args.sessions:
            print("❌ Error: --curriculum mode only supports single session (--session), not multiple sessions (--sessions)")
            sys.exit(1)
        if not args.session:
            print("❌ Error: --curriculum mode requires --session to be specified")
            sys.exit(1)
    
    # Validate model loading arguments
    if args.resume_from and args.from_model_checkpoint:
        print("❌ Error: Cannot use both --resume-from and --from-model-checkpoint")
        print("  --resume-from: Continue training on same session (preserves optimizer state)")
        print("  --from-model-checkpoint: Transfer learning to new session (resets optimizer)")
        sys.exit(1)
    
    # Run training
    try:
        if args.curriculum:
            print(f"🎓 Starting curriculum training on session {args.session}")
            results = asyncio.run(train_with_curriculum(args))
        else:
            results = asyncio.run(train_model(args))
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
        
        if args.curriculum:
            # Curriculum training results
            print(f"🎓 CURRICULUM TRAINING RESULTS")
            print(f"Algorithm: {results['algorithm'].upper()}")
            print(f"Session: {results['session_id']}")
            print(f"Markets trained: {len(results['markets_trained'])}/{results['total_markets']}")
            if isinstance(results['total_timesteps_per_market'], int):
                print(f"Timesteps per market: {results['total_timesteps_per_market']:,} (user override)")
            else:
                print(f"Timesteps per market: {results['total_timesteps_per_market']} (varies by market)")
            print(f"Total training duration: {results['total_duration']:.2f} seconds")
            print(f"Model saved to: {results['final_model_path']}")
            
            print(f"\n📊 MARKET BREAKDOWN:")
            for market in results['markets_trained'][:5]:  # Show first 5
                market_result = results['market_results'][market]
                if 'error' not in market_result:
                    episode_type = "FULL" if market_result.get('full_episode', False) else "LIMITED"
                    print(f"  ✅ {market}: {market_result['timesteps_trained']:,} steps ({episode_type}), "
                          f"{market_result['training_duration_seconds']:.1f}s, {market_result['timesteps_per_second']:.1f} ts/s")
                else:
                    print(f"  ❌ {market}: {market_result['error']}")
            
            if len(results['markets_trained']) > 5:
                print(f"  ... and {len(results['markets_trained']) - 5} more markets")
        
        else:
            # Regular training results
            print(f"Algorithm: {results['algorithm'].upper()}")
            print(f"Total timesteps: {results['total_timesteps']:,}")
            print(f"Training duration: {results['training_duration_seconds']:.2f} seconds")
            print(f"Timesteps per second: {results['timesteps_per_second']:.2f}")
            print(f"Sessions trained: {len(results['session_ids'])}")
            print(f"Model saved to: {results['final_model_path']}")
            
            if results.get('portfolio_statistics'):
                stats = results['portfolio_statistics']
                print(f"\nPortfolio Statistics:")
                print(f"  Episodes completed: {stats['total_episodes']}")
                print(f"  Unique markets: {stats['unique_markets']}")
                if stats.get('portfolio_stats'):
                    port_stats = stats['portfolio_stats']
                    print(f"  Avg portfolio value: {port_stats['mean']:.2f} cents")
                    print(f"  Portfolio range: {port_stats['min']:.2f} - {port_stats['max']:.2f}")
                
                # Show dynamics statistics if available
                if stats.get('dynamics_statistics'):
                    dyn_stats = stats['dynamics_statistics']
                    if dyn_stats.get('trading_outcomes'):
                        outcomes = dyn_stats['trading_outcomes']
                        print(f"  Trading win rate: {outcomes['win_rate']:.2%}")
                        print(f"  Episodes with gains: {outcomes['positive_episodes']}")
        
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()