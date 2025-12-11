#!/usr/bin/env python3
"""
Comprehensive SB3 training script for MarketAgnosticKalshiEnv.

This script provides a complete training pipeline integrating MarketAgnosticKalshiEnv
with Stable Baselines3 using SimpleSessionCurriculum for curriculum learning.

Features:
- PPO and A2C training support
- Session-based curriculum learning using train_single_session() and train_multiple_sessions()
- Model persistence with checkpointing and resumption
- Training metrics extraction using OrderManager API
- Progress monitoring with episode rewards and portfolio performance
- Comprehensive error handling and logging

Usage:
    # Train on single session
    python train_with_sb3.py --session 9 --algorithm ppo --total-timesteps 10000
    
    # Train on multiple sessions with curriculum learning  
    python train_with_sb3.py --sessions 6,7,8,9 --algorithm a2c --total-timesteps 50000
    
    # Resume from checkpoint
    python train_with_sb3.py --sessions 6,7,8,9 --algorithm ppo --resume-from models/ppo_checkpoint.zip
    
    # Train with custom configuration
    python train_with_sb3.py --session 9 --algorithm ppo --total-timesteps 20000 --save-freq 5000 --eval-freq 2500

Examples:
    $ python train_with_sb3.py --session 9 --algorithm ppo --total-timesteps 10000
    Trains PPO agent on session 9 for 10k timesteps
    
    $ python train_with_sb3.py --sessions 6,7,8,9 --algorithm a2c --total-timesteps 50000 --curriculum
    Trains A2C with curriculum learning across multiple sessions
"""

import asyncio
import argparse
import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
import json
import time
import numpy as np

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
from kalshiflow_rl.environments.market_agnostic_env import EnvConfig
from kalshiflow_rl.training.curriculum import train_single_session, train_multiple_sessions


class PortfolioMetricsCallback(BaseCallback):
    """
    Custom callback to track portfolio metrics during training.
    
    This callback extracts training metrics using OrderManager API methods
    and logs portfolio performance statistics.
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_portfolios = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            self.episode_count += 1
            
            # Extract portfolio metrics from environment
            env = self.training_env.envs[0]
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'order_manager'):
                order_manager = env.unwrapped.order_manager
                
                # Get portfolio metrics using OrderManager API
                portfolio_value = order_manager.get_portfolio_value_cents(env.unwrapped._get_current_market_prices())
                cash_balance = order_manager.get_cash_balance_cents()
                position_info = order_manager.get_position_info()
                
                portfolio_metrics = {
                    'episode': self.episode_count,
                    'portfolio_value_cents': portfolio_value,
                    'cash_balance_cents': cash_balance,
                    'position_count': len(position_info),
                    'total_position_value_cents': sum(
                        pos.get('current_value_cents', 0) for pos in position_info.values()
                    ),
                    'episode_length': env.unwrapped.current_step,
                    'market_ticker': getattr(env.unwrapped, 'current_market', 'unknown')
                }
                
                self.episode_portfolios.append(portfolio_metrics)
                
                # Log summary statistics periodically
                if self.episode_count % self.log_freq == 0:
                    self._log_portfolio_summary()
        
        return True
    
    def _log_portfolio_summary(self):
        """Log portfolio performance summary."""
        if not self.episode_portfolios:
            return
        
        recent_episodes = self.episode_portfolios[-self.log_freq:]
        
        portfolio_values = [ep['portfolio_value_cents'] for ep in recent_episodes]
        episode_lengths = [ep['episode_length'] for ep in recent_episodes]
        
        summary = {
            'episodes_completed': len(recent_episodes),
            'avg_portfolio_value_cents': np.mean(portfolio_values),
            'std_portfolio_value_cents': np.std(portfolio_values),
            'min_portfolio_value_cents': np.min(portfolio_values),
            'max_portfolio_value_cents': np.max(portfolio_values),
            'avg_episode_length': np.mean(episode_lengths),
            'unique_markets_trained': len(set(ep['market_ticker'] for ep in recent_episodes))
        }
        
        self.logger.info(f"Portfolio Summary (last {len(recent_episodes)} episodes):")
        self.logger.info(f"  Avg portfolio value: {summary['avg_portfolio_value_cents']:.2f} cents")
        self.logger.info(f"  Portfolio range: {summary['min_portfolio_value_cents']:.2f} - {summary['max_portfolio_value_cents']:.2f}")
        self.logger.info(f"  Avg episode length: {summary['avg_episode_length']:.1f} steps")
        self.logger.info(f"  Markets trained on: {summary['unique_markets_trained']}")
        
        # Log to tensorboard if available
        if hasattr(self, 'logger') and hasattr(self.logger, 'record'):
            for key, value in summary.items():
                self.logger.record(f"portfolio/{key}", value)
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Get complete episode statistics."""
        if not self.episode_portfolios:
            return {}
        
        portfolio_values = [ep['portfolio_value_cents'] for ep in self.episode_portfolios]
        episode_lengths = [ep['episode_length'] for ep in self.episode_portfolios]
        
        return {
            'total_episodes': len(self.episode_portfolios),
            'portfolio_stats': {
                'mean': np.mean(portfolio_values),
                'std': np.std(portfolio_values),
                'min': np.min(portfolio_values),
                'max': np.max(portfolio_values),
                'median': np.median(portfolio_values)
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
    """Get default parameters for SB3 algorithms."""
    if algorithm.lower() == "ppo":
        return {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
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
                    portfolio_log_freq: int = 1000) -> CallbackList:
    """
    Create training callbacks for monitoring and checkpointing.
    
    Args:
        model_save_path: Base path for saving models
        save_freq: Frequency of model checkpointing (timesteps)
        eval_freq: Frequency of model evaluation (timesteps)
        eval_env: Environment for evaluation (optional)
        portfolio_log_freq: Frequency of portfolio metrics logging (episodes)
        
    Returns:
        CallbackList with configured callbacks
    """
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(Path(model_save_path).parent / "checkpoints"),
        name_prefix=Path(model_save_path).stem
    )
    callbacks.append(checkpoint_callback)
    
    # Portfolio metrics callback
    portfolio_callback = PortfolioMetricsCallback(log_freq=portfolio_log_freq)
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
        max_episode_steps=args.max_episode_steps,
        skip_failed_markets=True
    )
    
    # Create environment
    logger.info("Creating training environment...")
    
    # Combine configs properly
    full_config = SB3TrainingConfig(
        env_config=env_config,
        min_episode_length=training_config.min_episode_length,
        max_episode_steps=training_config.max_episode_steps,
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
    
    if args.resume_from and Path(args.resume_from).exists():
        logger.info(f"Loading model from {args.resume_from}")
        if args.algorithm.lower() == "ppo":
            model = PPO.load(args.resume_from, env=vec_env, device=args.device)
        else:  # a2c
            model = A2C.load(args.resume_from, env=vec_env, device=args.device)
    else:
        model = create_model(args.algorithm, vec_env, model_params, args.device)
    
    # Setup model save paths
    models_dir = Path(args.model_save_path).parent
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create progress monitor
    progress_monitor = TrainingProgressMonitor(
        str(models_dir / "training_progress.json")
    )
    
    # Create callbacks
    callbacks = create_callbacks(
        model_save_path=args.model_save_path,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        portfolio_log_freq=args.portfolio_log_freq
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
        logger.info(f"Saving final model to {args.model_save_path}")
        model.save(args.model_save_path)
        
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
            'final_model_path': args.model_save_path,
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
  python train_with_sb3.py --session 9 --algorithm ppo --total-timesteps 10000
  python train_with_sb3.py --sessions 6,7,8,9 --algorithm a2c --total-timesteps 50000
  python train_with_sb3.py --session 9 --algorithm ppo --resume-from models/ppo_checkpoint.zip
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
    parser.add_argument('--total-timesteps', type=int, default=10000,
                       help='Total training timesteps (default: 10000)')
    parser.add_argument('--learning-rate', type=float,
                       help='Learning rate (uses algorithm default if not specified)')
    
    # Environment configuration
    parser.add_argument('--cash-start', type=int, default=10000,
                       help='Starting cash in cents (default: 10000 = $100)')
    parser.add_argument('--min-episode-length', type=int, default=10,
                       help='Minimum episode length for valid markets (default: 10)')
    parser.add_argument('--max-episode-steps', type=int,
                       help='Maximum steps per episode (no limit if not specified)')
    
    # Model persistence
    parser.add_argument('--model-save-path', default='backend/src/kalshiflow_rl/trained_models/trained_model.zip',
                       help='Path to save final trained model (default: backend/src/kalshiflow_rl/trained_models/trained_model.zip)')
    parser.add_argument('--resume-from', type=str,
                       help='Path to model checkpoint to resume training from')
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='Frequency of model checkpointing in timesteps (default: 10000)')
    
    # Evaluation and monitoring
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='Frequency of model evaluation in timesteps (default: 5000)')
    parser.add_argument('--portfolio-log-freq', type=int, default=1000,
                       help='Frequency of portfolio metrics logging in episodes (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=100,
                       help='Frequency of training log output in timesteps (default: 100)')
    
    # Technical configuration  
    parser.add_argument('--device', default='auto',
                       help='Device for training: auto, cpu, or cuda (default: auto)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str,
                       help='Optional log file path (logs to console if not specified)')
    
    args = parser.parse_args()
    
    # Run training
    try:
        results = asyncio.run(train_model(args))
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*80)
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
        
        print("="*80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()