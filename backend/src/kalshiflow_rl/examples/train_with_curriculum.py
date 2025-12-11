#!/usr/bin/env python3
"""
Training example demonstrating curriculum-based single-market training.

This example shows how to use the clean single-market training architecture:
1. Load session data with multiple markets
2. Use CurriculumService to select markets for training
3. Create MarketSessionViews for efficient single-market episodes
4. Train model across multiple markets to learn universal patterns

Usage:
    python train_with_curriculum.py [--session-id SESSION_ID] [--episodes 1000] [--strategy diverse_difficulty]
"""

import asyncio
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig
from kalshiflow_rl.environments.curriculum_service import CurriculumService, CurriculumConfig, CurriculumStrategy

logger = logging.getLogger(__name__)


class CurriculumTrainingDemo:
    """
    Demonstration of curriculum-based single-market training.
    
    This class orchestrates the training process using:
    - SessionDataLoader to load multi-market session data
    - CurriculumService to manage market selection and progression
    - MarketSessionView for efficient single-market training episodes
    - MarketAgnosticKalshiEnv for universal pattern learning
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        curriculum_strategy: CurriculumStrategy = CurriculumStrategy.DIVERSE_DIFFICULTY
    ):
        """
        Initialize training demo.
        
        Args:
            database_url: Database connection URL (defaults to environment variable)
            curriculum_strategy: Strategy for market selection
        """
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL must be set or provided as parameter")
        
        # Initialize components
        self.session_loader = SessionDataLoader(database_url=self.database_url)
        self.curriculum_config = CurriculumConfig(
            strategy=curriculum_strategy,
            episodes_per_market=50,  # Train 50 episodes per market before considering switch
            min_timesteps_per_market=500,  # Minimum 500 timesteps per market
            difficulty_progression=True,  # Start easy, progress to hard
            randomize_order=True,  # Add randomization within difficulty tiers
            patience=20  # Switch after 20 episodes without improvement
        )
        self.curriculum_service = CurriculumService(self.curriculum_config)
        self.env = None
        self.current_session = None
        
        logger.info(f"CurriculumTrainingDemo initialized with strategy: {curriculum_strategy.value}")
    
    async def load_session_data(self, session_id: Optional[int] = None) -> bool:
        """
        Load session data for training.
        
        Args:
            session_id: Specific session to load, or None to use latest
            
        Returns:
            True if session loaded successfully
        """
        try:
            # Find available sessions
            available_sessions = await self.session_loader.get_available_sessions()
            if not available_sessions:
                logger.error("No sessions available in database")
                return False
            
            # Select session
            if session_id is None:
                # Use most recent session with good quality
                for session in available_sessions:
                    quality = await self.session_loader.validate_session_quality(session['session_id'])
                    if quality >= 0.7:  # Good quality threshold
                        session_id = session['session_id']
                        break
                
                if session_id is None:
                    # Fallback to most recent session
                    session_id = available_sessions[0]['session_id']
                    logger.warning(f"Using session {session_id} with potentially lower quality")
            
            # Load session data
            logger.info(f"Loading session {session_id}...")
            self.current_session = await self.session_loader.load_session(session_id)
            
            if not self.current_session:
                logger.error(f"Failed to load session {session_id}")
                return False
            
            logger.info(f"Session loaded: {len(self.current_session.markets_involved)} markets, "
                       f"{self.current_session.get_episode_length()} timesteps, "
                       f"quality: {self.current_session.data_quality_score:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return False
    
    async def demonstrate_curriculum_training(self, max_episodes: int = 200) -> None:
        """
        Demonstrate curriculum-based training across multiple markets.
        
        Args:
            max_episodes: Maximum number of training episodes
        """
        if not self.current_session:
            logger.error("No session data loaded")
            return
        
        logger.info(f"Starting curriculum training demonstration with {max_episodes} episodes")
        logger.info(f"Session: {self.current_session.session_id}, "
                   f"Markets: {len(self.current_session.markets_involved)}")
        
        # Analyze available markets
        market_stats = self.curriculum_service.analyze_session_markets(self.current_session)
        logger.info(f"Analyzed {len(market_stats)} viable markets:")
        
        for ticker, stats in sorted(market_stats.items(), key=lambda x: x[1]['difficulty']):
            logger.info(f"  {ticker}: difficulty={stats['difficulty']:.2f}, "
                       f"volume={stats['total_volume']:.0f}, "
                       f"episodes={stats['episode_length']}")
        
        # Training loop
        episode_count = 0
        training_stats = {
            'episodes_per_market': {},
            'rewards_per_market': {},
            'best_rewards': {},
            'market_switches': 0
        }
        
        while episode_count < max_episodes:
            try:
                # Select market using curriculum strategy
                current_market = self.curriculum_service.select_market(self.current_session)
                if not current_market:
                    logger.error("No suitable market selected by curriculum service")
                    break
                
                # Check if we should switch markets
                if self.curriculum_service.should_switch_market():
                    # Force market selection (curriculum service will handle switching)
                    new_market = self.curriculum_service.select_market(self.current_session)
                    if new_market != current_market:
                        training_stats['market_switches'] += 1
                        logger.info(f"Market switch #{training_stats['market_switches']}: {current_market} → {new_market}")
                    current_market = new_market
                
                # Create market view for efficient single-market training
                market_view = self.current_session.create_market_view(current_market)
                if not market_view:
                    logger.warning(f"Failed to create view for market {current_market}")
                    continue
                
                # Initialize environment with market view (if needed)
                if self.env is None or (hasattr(self.env, 'session_data') and 
                                       not isinstance(self.env.session_data, type(market_view))):
                    env_config = EnvConfig(
                        max_markets=1,  # Single market training
                        temporal_features=True,
                        cash_start=10000  # $100 starting cash
                    )
                    self.env = MarketAgnosticKalshiEnv(market_view, env_config)
                    logger.info(f"Environment initialized for single-market training")
                else:
                    # Update session data for curriculum learning
                    self.env.set_session_data(market_view)
                
                # Run episode
                episode_reward = await self._run_episode(episode_count, current_market)
                
                # Update curriculum service with results
                self.curriculum_service.update_progress(
                    current_market, 
                    episode_reward, 
                    market_view.get_episode_length()
                )
                
                # Track statistics
                if current_market not in training_stats['episodes_per_market']:
                    training_stats['episodes_per_market'][current_market] = 0
                    training_stats['rewards_per_market'][current_market] = []
                    training_stats['best_rewards'][current_market] = -float('inf')
                
                training_stats['episodes_per_market'][current_market] += 1
                training_stats['rewards_per_market'][current_market].append(episode_reward)
                training_stats['best_rewards'][current_market] = max(
                    training_stats['best_rewards'][current_market], episode_reward
                )
                
                episode_count += 1
                
                # Periodic progress report
                if episode_count % 25 == 0:
                    await self._print_training_progress(episode_count, training_stats)
                
            except Exception as e:
                logger.error(f"Error in episode {episode_count}: {e}")
                continue
        
        # Final training summary
        logger.info(f"\n{'='*60}")
        logger.info("CURRICULUM TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        await self._print_final_summary(training_stats)
    
    async def _run_episode(self, episode_num: int, market_ticker: str) -> float:
        """
        Run a single training episode.
        
        Args:
            episode_num: Episode number
            market_ticker: Market being trained on
            
        Returns:
            Total episode reward
        """
        # Reset environment
        obs, info = self.env.reset()
        
        total_reward = 0.0
        step_count = 0
        done = False
        
        logger.debug(f"Episode {episode_num}: Starting training on {market_ticker} "
                    f"(length: {info['episode_length']} steps)")
        
        # Simple random policy for demonstration (replace with actual RL algorithm)
        while not done and step_count < info['episode_length']:
            # Random action selection (0-4: HOLD, BUY_YES, SELL_YES, BUY_NO, SELL_NO)
            action = self.env.action_space.sample()
            
            # Execute step
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            
            total_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Optional: Add actual RL training logic here
            # model.update(obs, action, reward, next_obs, done)
        
        logger.debug(f"Episode {episode_num}: {market_ticker} completed "
                    f"({step_count} steps, reward: {total_reward:.2f})")
        
        return total_reward
    
    async def _print_training_progress(self, episode_count: int, training_stats: dict) -> None:
        """Print periodic training progress report."""
        status = self.curriculum_service.get_curriculum_status()
        
        logger.info(f"\n--- Episode {episode_count} Progress Report ---")
        logger.info(f"Current market: {status['current_market']}")
        logger.info(f"Episodes since switch: {status['episodes_since_switch']}")
        logger.info(f"Markets trained: {status['markets_trained']}")
        logger.info(f"Market switches: {training_stats['market_switches']}")
        
        if 'current_market_progress' in status:
            progress = status['current_market_progress']
            logger.info(f"Current market stats: episodes={progress['episodes_trained']}, "
                       f"avg_reward={progress['avg_reward']:.3f}, "
                       f"best_reward={progress['best_reward']:.3f}")
    
    async def _print_final_summary(self, training_stats: dict) -> None:
        """Print final training summary."""
        import numpy as np
        
        logger.info(f"Total episodes: {sum(training_stats['episodes_per_market'].values())}")
        logger.info(f"Markets trained: {len(training_stats['episodes_per_market'])}")
        logger.info(f"Market switches: {training_stats['market_switches']}")
        
        logger.info(f"\nPer-market performance:")
        for market in sorted(training_stats['episodes_per_market'].keys()):
            episodes = training_stats['episodes_per_market'][market]
            rewards = training_stats['rewards_per_market'][market]
            avg_reward = np.mean(rewards) if rewards else 0.0
            best_reward = training_stats['best_rewards'][market]
            
            logger.info(f"  {market}: {episodes:3d} episodes, "
                       f"avg_reward={avg_reward:6.3f}, "
                       f"best_reward={best_reward:6.3f}")
        
        # Overall statistics
        all_rewards = []
        for rewards_list in training_stats['rewards_per_market'].values():
            all_rewards.extend(rewards_list)
        
        if all_rewards:
            logger.info(f"\nOverall performance:")
            logger.info(f"  Mean reward: {np.mean(all_rewards):.3f}")
            logger.info(f"  Std reward:  {np.std(all_rewards):.3f}")
            logger.info(f"  Min reward:  {np.min(all_rewards):.3f}")
            logger.info(f"  Max reward:  {np.max(all_rewards):.3f}")
    
    async def close(self) -> None:
        """Clean up resources."""
        if self.env:
            self.env.close()
        await self.session_loader.close()


async def main():
    """Main entry point for curriculum training demonstration."""
    parser = argparse.ArgumentParser(
        description='Demonstrate curriculum-based single-market RL training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (diverse difficulty strategy)
  python train_with_curriculum.py
  
  # Train on specific session with highest volume strategy
  python train_with_curriculum.py --session-id 5 --strategy highest_volume
  
  # Extended training with random market selection
  python train_with_curriculum.py --episodes 500 --strategy random
        """
    )
    
    parser.add_argument('--session-id', type=int,
                       help='Session ID to train on (uses latest if not provided)')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes (default: 200)')
    parser.add_argument('--strategy', type=str, default='diverse_difficulty',
                       choices=['highest_volume', 'most_active', 'diverse_difficulty', 'random'],
                       help='Curriculum strategy (default: diverse_difficulty)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Convert strategy string to enum
    strategy_map = {
        'highest_volume': CurriculumStrategy.HIGHEST_VOLUME,
        'most_active': CurriculumStrategy.MOST_ACTIVE,
        'diverse_difficulty': CurriculumStrategy.DIVERSE_DIFFICULTY,
        'random': CurriculumStrategy.RANDOM
    }
    curriculum_strategy = strategy_map[args.strategy]
    
    logger.info(f"Starting curriculum training demonstration")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Episodes: {args.episodes}")
    if args.session_id:
        logger.info(f"Session ID: {args.session_id}")
    
    # Initialize and run training
    demo = CurriculumTrainingDemo(curriculum_strategy=curriculum_strategy)
    
    try:
        # Load session data
        success = await demo.load_session_data(args.session_id)
        if not success:
            logger.error("Failed to load session data")
            sys.exit(1)
        
        # Run curriculum training
        await demo.demonstrate_curriculum_training(args.episodes)
        
        logger.info("Training demonstration completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await demo.close()


if __name__ == "__main__":
    asyncio.run(main())