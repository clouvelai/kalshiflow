"""
Curriculum service for managing market selection and training progression.

This module provides the CurriculumService class that manages market selection
strategies for single-market training. The service tracks training progress
per market and decides when to switch markets based on configurable strategies.

Key strategies:
- highest_volume: Select markets with highest trading activity
- most_active: Select markets with most orderbook updates
- diverse_difficulty: Balance across different market types and difficulties
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import random
import numpy as np
from datetime import datetime

from .session_data_loader import SessionData, MarketSessionView

logger = logging.getLogger(__name__)


class CurriculumStrategy(Enum):
    """Market selection strategies for curriculum learning."""
    HIGHEST_VOLUME = "highest_volume"
    MOST_ACTIVE = "most_active" 
    DIVERSE_DIFFICULTY = "diverse_difficulty"
    RANDOM = "random"


@dataclass
class MarketProgress:
    """Tracks training progress for a specific market."""
    market_ticker: str
    episodes_trained: int = 0
    total_timesteps: int = 0
    best_reward: float = -float('inf')
    avg_reward: float = 0.0
    last_trained: Optional[datetime] = None
    
    # Market characteristics
    avg_spread: float = 0.0
    volatility: float = 0.0
    coverage: float = 0.0  # Fraction of session where market was active
    difficulty_score: float = 0.5  # [0,1] where 1 = most difficult


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    strategy: CurriculumStrategy = CurriculumStrategy.DIVERSE_DIFFICULTY
    episodes_per_market: int = 100  # Episodes before considering market switch
    min_timesteps_per_market: int = 1000  # Minimum timesteps before switching
    difficulty_progression: bool = True  # Start easy, progress to hard
    randomize_order: bool = True  # Randomize within difficulty groups
    patience: int = 50  # Episodes without improvement before switching


class CurriculumService:
    """
    Manages market selection and training progression for curriculum learning.
    
    This service coordinates single-market training by:
    1. Analyzing available markets in session data
    2. Selecting appropriate markets based on strategy
    3. Tracking training progress per market
    4. Deciding when to switch to new markets
    5. Providing curriculum progression insights
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        """
        Initialize curriculum service.
        
        Args:
            config: Curriculum configuration
        """
        self.config = config or CurriculumConfig()
        self.market_progress: Dict[str, MarketProgress] = {}
        self.available_markets: List[str] = []
        self.current_market: Optional[str] = None
        self.episodes_since_switch: int = 0
        self.total_episodes: int = 0
        
        logger.info(f"CurriculumService initialized with strategy: {self.config.strategy.value}")
    
    def analyze_session_markets(self, session_data: SessionData) -> Dict[str, Dict[str, float]]:
        """
        Analyze markets available in session data for curriculum planning.
        
        Computes market characteristics like volume, activity, spread, volatility
        that are used for market selection and difficulty assessment.
        
        Args:
            session_data: Complete session data to analyze
            
        Returns:
            Dictionary mapping market ticker to characteristics:
            {
                "ticker": {
                    "total_volume": float,
                    "activity_score": float,
                    "avg_spread": float,
                    "volatility": float,
                    "coverage": float,
                    "difficulty": float
                }
            }
        """
        market_stats = {}
        
        # Analyze each market in the session
        for market_ticker in session_data.markets_involved:
            # Create temporary market view to get stats
            market_view = session_data.create_market_view(market_ticker)
            if not market_view or market_view.get_episode_length() < 10:
                continue  # Skip markets with insufficient data
            
            # Calculate market characteristics
            total_volume = 0
            activity_scores = []
            spreads = []
            
            for point in market_view.data_points:
                if market_ticker in point.markets_data:
                    market_data = point.markets_data[market_ticker]
                    
                    # Total volume calculation
                    volume = 0
                    for side in ['yes_bids', 'yes_asks', 'no_bids', 'no_asks']:
                        if side in market_data:
                            volume += sum(market_data[side].values())
                    total_volume += volume
                    
                    # Activity score
                    activity_scores.append(point.activity_score)
                    
                    # Spreads
                    if market_ticker in point.spreads:
                        yes_spread, no_spread = point.spreads[market_ticker]
                        if yes_spread is not None:
                            spreads.append(yes_spread)
                        if no_spread is not None:
                            spreads.append(no_spread)
            
            # Calculate summary statistics
            avg_activity = np.mean(activity_scores) if activity_scores else 0.0
            avg_spread = np.mean(spreads) if spreads else 0.0
            volatility = market_view.volatility_score
            coverage = market_view.market_coverage
            
            # Calculate difficulty score [0,1]
            # Higher spread = easier (more profit opportunity)
            # Higher volatility = harder (more unpredictable)
            # Lower coverage = harder (less training data)
            spread_factor = max(0, 1.0 - (avg_spread / 10.0))  # 10 cents = max expected spread
            volatility_factor = min(1.0, volatility / 0.5)  # 0.5 = high volatility threshold
            coverage_factor = max(0, 1.0 - coverage)  # Low coverage = harder
            
            difficulty = (spread_factor * 0.3 + volatility_factor * 0.5 + coverage_factor * 0.2)
            difficulty = np.clip(difficulty, 0.0, 1.0)
            
            market_stats[market_ticker] = {
                "total_volume": total_volume,
                "activity_score": avg_activity,
                "avg_spread": avg_spread,
                "volatility": volatility,
                "coverage": coverage,
                "difficulty": difficulty,
                "episode_length": market_view.get_episode_length()
            }
        
        logger.info(f"Analyzed {len(market_stats)} markets in session {session_data.session_id}")
        return market_stats
    
    def select_market(self, session_data: SessionData) -> Optional[str]:
        """
        Select next market for training based on curriculum strategy.
        
        Args:
            session_data: Session data containing available markets
            
        Returns:
            Market ticker to train on next, or None if no suitable markets
        """
        # Analyze available markets if not done yet
        market_stats = self.analyze_session_markets(session_data)
        if not market_stats:
            logger.warning("No suitable markets found in session")
            return None
        
        # Update available markets
        self.available_markets = list(market_stats.keys())
        
        # Initialize progress tracking for new markets
        for market_ticker in self.available_markets:
            if market_ticker not in self.market_progress:
                stats = market_stats[market_ticker]
                self.market_progress[market_ticker] = MarketProgress(
                    market_ticker=market_ticker,
                    avg_spread=stats["avg_spread"],
                    volatility=stats["volatility"],
                    coverage=stats["coverage"],
                    difficulty_score=stats["difficulty"]
                )
        
        # Select market based on strategy
        if self.config.strategy == CurriculumStrategy.HIGHEST_VOLUME:
            selected = self._select_by_volume(market_stats)
        elif self.config.strategy == CurriculumStrategy.MOST_ACTIVE:
            selected = self._select_by_activity(market_stats)
        elif self.config.strategy == CurriculumStrategy.DIVERSE_DIFFICULTY:
            selected = self._select_by_difficulty(market_stats)
        elif self.config.strategy == CurriculumStrategy.RANDOM:
            selected = random.choice(self.available_markets)
        else:
            # Fallback to random
            selected = random.choice(self.available_markets)
        
        # Update current market and reset episode counter
        if selected != self.current_market:
            logger.info(f"Switching from {self.current_market} to {selected} "
                       f"(strategy: {self.config.strategy.value})")
            self.current_market = selected
            self.episodes_since_switch = 0
        
        return selected
    
    def update_progress(self, market_ticker: str, episode_reward: float, timesteps: int) -> None:
        """
        Update training progress for a market after episode completion.
        
        Args:
            market_ticker: Market that was trained on
            episode_reward: Total episode reward
            timesteps: Number of timesteps in episode
        """
        if market_ticker not in self.market_progress:
            logger.warning(f"Unknown market {market_ticker} in progress update")
            return
        
        progress = self.market_progress[market_ticker]
        
        # Update statistics
        progress.episodes_trained += 1
        progress.total_timesteps += timesteps
        progress.best_reward = max(progress.best_reward, episode_reward)
        
        # Update rolling average reward
        alpha = 0.1  # Learning rate for exponential moving average
        if progress.episodes_trained == 1:
            progress.avg_reward = episode_reward
        else:
            progress.avg_reward = (1 - alpha) * progress.avg_reward + alpha * episode_reward
        
        progress.last_trained = datetime.now()
        
        # Update global counters
        self.episodes_since_switch += 1
        self.total_episodes += 1
        
        logger.debug(f"Updated progress for {market_ticker}: "
                    f"episodes={progress.episodes_trained}, "
                    f"avg_reward={progress.avg_reward:.3f}, "
                    f"best_reward={progress.best_reward:.3f}")
    
    def should_switch_market(self) -> bool:
        """
        Determine if it's time to switch to a different market.
        
        Returns:
            True if curriculum suggests switching markets
        """
        if not self.current_market:
            return True  # No current market, should select one
        
        progress = self.market_progress.get(self.current_market)
        if not progress:
            return True  # No progress data, should switch
        
        # Check minimum episodes threshold
        if self.episodes_since_switch < self.config.episodes_per_market:
            return False
        
        # Check minimum timesteps threshold
        if progress.total_timesteps < self.config.min_timesteps_per_market:
            return False
        
        # Check improvement patience
        if self.config.difficulty_progression and self.episodes_since_switch >= self.config.patience:
            # Look for improvement in recent episodes
            # For now, switch after patience episodes regardless
            logger.info(f"Switching market after {self.episodes_since_switch} episodes (patience reached)")
            return True
        
        return False
    
    def get_curriculum_status(self) -> Dict[str, Any]:
        """
        Get current curriculum learning status and statistics.
        
        Returns:
            Dictionary containing curriculum status information
        """
        status = {
            "current_market": self.current_market,
            "episodes_since_switch": self.episodes_since_switch,
            "total_episodes": self.total_episodes,
            "available_markets": len(self.available_markets),
            "strategy": self.config.strategy.value,
            "markets_trained": len([p for p in self.market_progress.values() if p.episodes_trained > 0]),
            "total_timesteps": sum(p.total_timesteps for p in self.market_progress.values())
        }
        
        if self.current_market and self.current_market in self.market_progress:
            current_progress = self.market_progress[self.current_market]
            status["current_market_progress"] = {
                "episodes_trained": current_progress.episodes_trained,
                "avg_reward": current_progress.avg_reward,
                "best_reward": current_progress.best_reward,
                "difficulty": current_progress.difficulty_score
            }
        
        return status
    
    def _select_by_volume(self, market_stats: Dict[str, Dict[str, float]]) -> str:
        """Select market with highest total volume."""
        return max(market_stats.items(), key=lambda x: x[1]["total_volume"])[0]
    
    def _select_by_activity(self, market_stats: Dict[str, Dict[str, float]]) -> str:
        """Select market with highest activity score."""
        return max(market_stats.items(), key=lambda x: x[1]["activity_score"])[0]
    
    def _select_by_difficulty(self, market_stats: Dict[str, Dict[str, float]]) -> str:
        """
        Select market based on curriculum difficulty progression.
        
        Starts with easier markets (low difficulty score) and progresses
        to harder markets as training advances.
        """
        # Sort markets by difficulty
        sorted_markets = sorted(
            market_stats.items(),
            key=lambda x: x[1]["difficulty"]
        )
        
        if self.config.difficulty_progression:
            # Progressive difficulty based on total training progress
            progress_ratio = min(1.0, self.total_episodes / (len(self.available_markets) * self.config.episodes_per_market))
            
            # Select from easier markets early, harder markets later
            if progress_ratio < 0.3:
                # Easy markets (bottom 30% difficulty)
                candidates = sorted_markets[:max(1, len(sorted_markets) // 3)]
            elif progress_ratio < 0.7:
                # Medium markets (middle 40% difficulty)
                start_idx = len(sorted_markets) // 3
                end_idx = start_idx + max(1, len(sorted_markets) // 2)
                candidates = sorted_markets[start_idx:end_idx]
            else:
                # Hard markets (top 30% difficulty)
                candidates = sorted_markets[max(0, len(sorted_markets) - len(sorted_markets) // 3):]
        else:
            # No progression, select from all available
            candidates = sorted_markets
        
        # Avoid recent markets to encourage diversity
        if len(candidates) > 1 and self.current_market:
            candidates = [c for c in candidates if c[0] != self.current_market]
        
        # Randomize within difficulty tier if configured
        if self.config.randomize_order and candidates:
            return random.choice(candidates)[0]
        else:
            return candidates[0][0] if candidates else sorted_markets[0][0]