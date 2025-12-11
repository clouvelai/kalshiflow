"""
Simple curriculum learning system for session-based market training.

This module provides a straightforward "train on all valid markets" curriculum
that loads session data, creates market views for each market with sufficient data,
and tracks training performance across all markets in a session.

Key Features:
- Loads session data using SessionDataLoader
- Creates MarketSessionView for every valid market (≥1 snapshot, ≥1 delta)
- Trains MarketAgnosticKalshiEnv on each market view
- Tracks metadata (success/fail, rewards, episode lengths) per market
- Aggregates performance statistics per session

This implements the M8_CURRICULUM_LEARNING milestone with a simple approach
that can be extended with more sophisticated selection strategies later.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
import numpy as np
from pathlib import Path

from ..environments.session_data_loader import SessionDataLoader, SessionData, MarketSessionView
from ..environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig

logger = logging.getLogger(__name__)


@dataclass
class MarketTrainingResult:
    """Training result for a single market within a session."""
    market_ticker: str
    session_id: int
    
    # Episode execution metadata
    success: bool = False  # Did episode complete without errors?
    error_message: Optional[str] = None
    
    # Performance metrics
    total_reward: float = 0.0
    episode_length: int = 0
    final_cash: float = 0.0
    final_position_value: float = 0.0
    
    # Market characteristics
    market_coverage: float = 0.0  # Fraction of session where market was active
    avg_spread: float = 0.0
    volatility_score: float = 0.0
    
    # Execution timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[timedelta] = None
    
    def __post_init__(self):
        """Calculate derived fields."""
        if self.start_time and self.end_time:
            self.duration = self.end_time - self.start_time


@dataclass 
class SessionTrainingResults:
    """Aggregated training results for an entire session."""
    session_id: int
    total_markets: int = 0
    successful_markets: int = 0
    failed_markets: int = 0
    
    # Aggregate performance
    total_episodes: int = 0
    total_timesteps: int = 0
    avg_reward: float = 0.0
    best_reward: float = -float('inf')
    worst_reward: float = float('inf')
    
    # Market results
    market_results: List[MarketTrainingResult] = field(default_factory=list)
    
    # Session timing
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: Optional[timedelta] = None
    
    def add_result(self, result: MarketTrainingResult):
        """Add a market training result and update aggregates."""
        self.market_results.append(result)
        self.total_markets += 1
        
        if result.success:
            self.successful_markets += 1
            self.total_episodes += 1
            self.total_timesteps += result.episode_length
            
            # Update reward statistics
            self.best_reward = max(self.best_reward, result.total_reward)
            self.worst_reward = min(self.worst_reward, result.total_reward)
            
            # Update rolling average reward
            if self.total_episodes == 1:
                self.avg_reward = result.total_reward
            else:
                # Incremental average update
                self.avg_reward = ((self.avg_reward * (self.total_episodes - 1)) + result.total_reward) / self.total_episodes
        else:
            self.failed_markets += 1
    
    def get_success_rate(self) -> float:
        """Calculate success rate of market training."""
        if self.total_markets == 0:
            return 0.0
        return self.successful_markets / self.total_markets
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'session_id': self.session_id,
            'total_markets': self.total_markets,
            'success_rate': self.get_success_rate(),
            'successful_markets': self.successful_markets,
            'failed_markets': self.failed_markets,
            'total_episodes': self.total_episodes,
            'total_timesteps': self.total_timesteps,
            'avg_reward': self.avg_reward,
            'best_reward': self.best_reward if self.best_reward != -float('inf') else 0.0,
            'worst_reward': self.worst_reward if self.worst_reward != float('inf') else 0.0,
            'total_duration': str(self.total_duration) if self.total_duration else None
        }


class SimpleSessionCurriculum:
    """
    Simple curriculum learning system that trains on all valid markets in a session.
    
    This implements a straightforward "train on all valid markets" approach:
    1. Load session data using SessionDataLoader
    2. Create MarketSessionView for each market with sufficient data
    3. Train MarketAgnosticKalshiEnv on each market view
    4. Track performance metadata per market and aggregate per session
    
    No complex selection heuristics - just iterate through all available markets
    and track results for understanding performance patterns.
    """
    
    def __init__(self, 
                 database_url: str,
                 env_config: Optional[EnvConfig] = None):
        """
        Initialize simple session curriculum.
        
        Args:
            database_url: Database connection string for SessionDataLoader
            env_config: Configuration for MarketAgnosticKalshiEnv instances
        """
        self.database_url = database_url
        self.env_config = env_config or EnvConfig()
        
        # Initialize data loader
        self.data_loader = SessionDataLoader(database_url=database_url)
        
        # Training state
        self.current_session_results: Optional[SessionTrainingResults] = None
        self.session_history: List[SessionTrainingResults] = []
        
        logger.info(f"SimpleSessionCurriculum initialized with env_config: {self.env_config}")
    
    async def train_session(self, session_id: int, 
                          min_snapshots: int = 1,
                          min_deltas: int = 1) -> SessionTrainingResults:
        """
        Train on all valid markets in a session.
        
        Args:
            session_id: Session to load and train on
            min_snapshots: Minimum snapshots required for a market to be valid
            min_deltas: Minimum deltas required for a market to be valid
            
        Returns:
            SessionTrainingResults with per-market results and aggregated metrics
        """
        logger.info(f"Starting session training for session_id={session_id}")
        
        # Initialize session results
        session_results = SessionTrainingResults(
            session_id=session_id,
            start_time=datetime.now()
        )
        self.current_session_results = session_results
        
        try:
            # 1. Load session data
            logger.info(f"Loading session data for session_id={session_id}")
            session_data = await self.data_loader.load_session(session_id)
            
            if not session_data:
                logger.error(f"Failed to load session data for session_id={session_id}")
                session_results.end_time = datetime.now()
                session_results.total_duration = session_results.end_time - session_results.start_time
                return session_results
            
            logger.info(f"Loaded session {session_id}: {len(session_data.markets_involved)} markets, "
                       f"{session_data.get_episode_length()} timesteps")
            
            # 2. Identify valid markets and create views
            valid_markets = []
            
            for market_ticker in session_data.markets_involved:
                # Check market data sufficiency
                if await self._is_market_valid(session_data, market_ticker, min_snapshots, min_deltas):
                    valid_markets.append(market_ticker)
                else:
                    logger.debug(f"Skipping market {market_ticker}: insufficient data")
            
            logger.info(f"Found {len(valid_markets)} valid markets out of {len(session_data.markets_involved)} total")
            
            # 3. Train on each valid market
            for market_ticker in valid_markets:
                result = await self._train_market(session_data, market_ticker)
                session_results.add_result(result)
                
                logger.info(f"Completed {market_ticker}: "
                           f"success={result.success}, "
                           f"reward={result.total_reward:.2f}, "
                           f"length={result.episode_length}")
            
            # 4. Finalize session results
            session_results.end_time = datetime.now()
            session_results.total_duration = session_results.end_time - session_results.start_time
            
            # Add to history
            self.session_history.append(session_results)
            
            logger.info(f"Session {session_id} training complete: "
                       f"success_rate={session_results.get_success_rate():.1%}, "
                       f"avg_reward={session_results.avg_reward:.2f}, "
                       f"duration={session_results.total_duration}")
            
            return session_results
            
        except Exception as e:
            logger.error(f"Error during session {session_id} training: {e}")
            session_results.end_time = datetime.now()
            session_results.total_duration = session_results.end_time - session_results.start_time
            return session_results
    
    async def _is_market_valid(self, session_data: SessionData, market_ticker: str,
                             min_snapshots: int, min_deltas: int) -> bool:
        """
        Check if a market has sufficient data for training.
        
        Args:
            session_data: Loaded session data
            market_ticker: Market to check
            min_snapshots: Minimum snapshots required
            min_deltas: Minimum deltas required
            
        Returns:
            True if market meets minimum data requirements
        """
        try:
            # Create market view to analyze data
            market_view = session_data.create_market_view(market_ticker)
            
            if not market_view:
                return False
            
            # Check minimum episode length (proxy for data sufficiency)
            if market_view.get_episode_length() < max(min_snapshots, min_deltas):
                return False
            
            # Additional checks could be added here:
            # - Check for minimum spread data
            # - Verify orderbook reconstruction quality
            # - Check temporal coverage
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating market {market_ticker}: {e}")
            return False
    
    async def _train_market(self, session_data: SessionData, market_ticker: str) -> MarketTrainingResult:
        """
        Train MarketAgnosticKalshiEnv on a single market view.
        
        Args:
            session_data: Complete session data
            market_ticker: Market ticker to train on
            
        Returns:
            MarketTrainingResult with training outcome and metrics
        """
        result = MarketTrainingResult(
            market_ticker=market_ticker,
            session_id=session_data.session_id,
            start_time=datetime.now()
        )
        
        try:
            # 1. Create market view
            market_view = session_data.create_market_view(market_ticker)
            
            if not market_view:
                result.success = False
                result.error_message = "Failed to create market view"
                result.end_time = datetime.now()
                return result
            
            # 2. Extract market characteristics
            result.market_coverage = market_view.market_coverage
            result.avg_spread = market_view.avg_spread
            result.volatility_score = market_view.volatility_score
            
            # 3. Initialize environment with market view
            env = MarketAgnosticKalshiEnv(
                market_view=market_view,
                config=self.env_config
            )
            
            # 4. Run single episode through the full market view
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            
            while not done:
                # Simple random action for now (curriculum focuses on data loading/tracking)
                action = env.action_space.sample()
                
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                step_count += 1
                
                done = terminated or truncated
                
                # Safety check for infinite loops
                if step_count > market_view.get_episode_length() + 10:
                    logger.warning(f"Episode {market_ticker} exceeded expected length, terminating")
                    break
            
            # 5. Extract final metrics
            result.total_reward = total_reward
            result.episode_length = step_count
            result.final_cash = env.order_manager.cash
            result.final_position_value = env.order_manager.get_total_position_value()
            result.success = True
            
            logger.debug(f"Market {market_ticker} training completed: "
                        f"reward={total_reward:.2f}, steps={step_count}")
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            logger.error(f"Error training market {market_ticker}: {e}")
            
        finally:
            result.end_time = datetime.now()
            
        return result
    
    def get_session_summary(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get summary for a specific session."""
        for session_results in self.session_history:
            if session_results.session_id == session_id:
                return session_results.get_summary()
        return None
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get summary across all trained sessions."""
        if not self.session_history:
            return {
                'total_sessions': 0,
                'total_markets': 0,
                'overall_success_rate': 0.0,
                'avg_reward_across_sessions': 0.0
            }
        
        total_sessions = len(self.session_history)
        total_markets = sum(s.total_markets for s in self.session_history)
        successful_markets = sum(s.successful_markets for s in self.session_history)
        
        # Calculate overall success rate
        overall_success_rate = successful_markets / total_markets if total_markets > 0 else 0.0
        
        # Calculate average reward across sessions
        session_rewards = [s.avg_reward for s in self.session_history if s.total_episodes > 0]
        avg_reward_across_sessions = np.mean(session_rewards) if session_rewards else 0.0
        
        return {
            'total_sessions': total_sessions,
            'total_markets': total_markets,
            'successful_markets': successful_markets,
            'failed_markets': total_markets - successful_markets,
            'overall_success_rate': overall_success_rate,
            'avg_reward_across_sessions': avg_reward_across_sessions,
            'session_summaries': [s.get_summary() for s in self.session_history]
        }
    
    def reset(self):
        """Reset curriculum state (clear history)."""
        self.session_history.clear()
        self.current_session_results = None
        logger.info("SimpleSessionCurriculum state reset")


# Utility functions for easy integration

async def train_single_session(session_id: int, 
                             database_url: str,
                             env_config: Optional[EnvConfig] = None,
                             min_snapshots: int = 1,
                             min_deltas: int = 1) -> SessionTrainingResults:
    """
    Convenience function to train on a single session.
    
    Args:
        session_id: Session to train on
        database_url: Database connection string
        env_config: Environment configuration
        min_snapshots: Minimum snapshots required per market
        min_deltas: Minimum deltas required per market
        
    Returns:
        SessionTrainingResults with complete training results
    """
    curriculum = SimpleSessionCurriculum(database_url, env_config)
    return await curriculum.train_session(session_id, min_snapshots, min_deltas)


async def train_multiple_sessions(session_ids: List[int],
                                database_url: str,
                                env_config: Optional[EnvConfig] = None) -> List[SessionTrainingResults]:
    """
    Convenience function to train on multiple sessions sequentially.
    
    Args:
        session_ids: List of session IDs to train on
        database_url: Database connection string  
        env_config: Environment configuration
        
    Returns:
        List of SessionTrainingResults for each session
    """
    curriculum = SimpleSessionCurriculum(database_url, env_config)
    results = []
    
    for session_id in session_ids:
        result = await curriculum.train_session(session_id)
        results.append(result)
        
        logger.info(f"Session {session_id} complete: "
                   f"success_rate={result.get_success_rate():.1%}, "
                   f"avg_reward={result.avg_reward:.2f}")
    
    return results