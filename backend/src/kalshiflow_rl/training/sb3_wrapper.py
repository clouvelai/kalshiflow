"""
Stable Baselines3 wrapper for session-based episode training.

This module provides a wrapper that converts MarketSessionView data into 
SB3-compatible environment instances, enabling seamless integration with
SimpleSessionCurriculum for curriculum learning.

Key features:
- Session-based episode generation from MarketSessionView data
- Automatic environment initialization with pre-loaded session data
- Integration with train_single_session() and train_multiple_sessions()
- Proper error handling for insufficient data and failed market selection
- Curriculum learning support across multiple sessions
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Iterator
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig
from ..environments.session_data_loader import SessionDataLoader, MarketSessionView
# Note: Simple curriculum functionality is handled directly in train_with_sb3.py
# The complex curriculum system has been replaced with SimpleMarketCurriculum

logger = logging.getLogger("kalshiflow_rl.training.sb3_wrapper")


@dataclass
class SB3TrainingConfig:
    """Configuration for SB3 training wrapper."""
    # Environment configuration
    env_config: EnvConfig = None  # Will be set to EnvConfig() in __post_init__
    
    # Session data requirements
    min_snapshots: int = 1
    min_deltas: int = 1
    min_episode_length: int = 10
    
    # Training parameters
    max_episode_steps: Optional[int] = None  # DEPRECATED: Episodes run to natural completion
    
    # Error handling
    skip_failed_markets: bool = True  # Skip markets that fail to initialize
    max_retries: int = 3  # Max retries for failed operations
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.env_config is None:
            self.env_config = EnvConfig()


class SessionBasedEnvironment(gym.Env):
    """
    Gym environment that provides session-based episodes from MarketSessionView data.
    
    This environment enables using MarketAgnosticKalshiEnv with SB3 algorithms by
    automatically managing session data loading and market view creation.
    
    The environment operates in two modes:
    1. Single session mode: Cycles through markets in one session
    2. Multi-session mode: Cycles through markets across multiple sessions
    
    Each reset() returns a new episode from the next available market view,
    enabling curriculum learning across diverse market conditions.
    """
    
    def __init__(self,
                 database_url: str,
                 session_ids: Union[int, List[int]],
                 config: Optional[SB3TrainingConfig] = None):
        """
        Initialize session-based environment.
        
        Args:
            database_url: Database connection string for session loading
            session_ids: Single session ID or list of session IDs to cycle through
            config: Training configuration
        """
        super().__init__()
        
        self.database_url = database_url
        self.session_ids = [session_ids] if isinstance(session_ids, int) else session_ids
        self.config = config or SB3TrainingConfig()
        
        # Note: Curriculum system disabled - using SimpleMarketCurriculum in train_with_sb3.py
        # self.curriculum = SimpleSessionCurriculum(
        #     database_url=database_url,
        #     env_config=self.config.env_config
        # )
        
        # Market view management
        self.market_views: List[MarketSessionView] = []
        self.current_market_index = 0
        self.current_env: Optional[MarketAgnosticKalshiEnv] = None
        
        # Load all market views from sessions
        logger.info(f"Initializing SessionBasedEnvironment with sessions: {self.session_ids}")
        
        # Since we need async loading, we'll defer the actual loading to reset()
        # But we need to set spaces immediately for SB3 compatibility
        self._spaces_initialized = False
        self._dummy_env: Optional[MarketAgnosticKalshiEnv] = None
        
        # Initialize observation and action spaces with default values
        # These will be properly validated in _ensure_spaces_initialized()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        
        logger.info(f"SessionBasedEnvironment initialized with default spaces: obs={self.observation_space.shape}, act={self.action_space.n}")
    
    async def _load_market_views(self) -> None:
        """Load all market views from configured sessions."""
        logger.info("Loading market views from sessions...")
        
        data_loader = SessionDataLoader(database_url=self.database_url)
        
        for session_id in self.session_ids:
            try:
                session_data = await data_loader.load_session(session_id)
                if not session_data:
                    logger.warning(f"Failed to load session {session_id}")
                    continue
                
                logger.info(f"Processing session {session_id}: {len(session_data.markets_involved)} markets")
                
                # Create market views for all valid markets
                for market_ticker in session_data.markets_involved:
                    try:
                        market_view = session_data.create_market_view(market_ticker)
                        if (market_view and 
                            market_view.get_episode_length() >= self.config.min_episode_length):
                            self.market_views.append(market_view)
                            logger.debug(f"Added market view: {market_ticker} ({market_view.get_episode_length()} steps)")
                    except Exception as e:
                        if not self.config.skip_failed_markets:
                            raise
                        logger.warning(f"Failed to create market view for {market_ticker}: {e}")
                
            except Exception as e:
                if not self.config.skip_failed_markets:
                    raise
                logger.warning(f"Failed to process session {session_id}: {e}")
        
        logger.info(f"Loaded {len(self.market_views)} market views total")
        
        if not self.market_views:
            raise ValueError("No valid market views found in provided sessions")
    
    def _ensure_spaces_initialized(self):
        """Ensure observation and action spaces are initialized."""
        if self._spaces_initialized:
            return
        
        # Market views should already be loaded by now
        if not self.market_views:
            logger.warning("No market views available for space initialization")
            # Use default spaces as fallback
            self._spaces_initialized = True
            return
        
        try:
            # Create dummy environment from first market view
            self._dummy_env = MarketAgnosticKalshiEnv(
                market_view=self.market_views[0],
                config=self.config.env_config
            )
            
            # Update spaces to match actual environment
            self.observation_space = self._dummy_env.observation_space
            self.action_space = self._dummy_env.action_space
            
            logger.info(f"Validated spaces: obs={self.observation_space.shape}, act={self.action_space.n}")
            self._spaces_initialized = True
        
        except Exception as e:
            logger.error(f"Failed to initialize spaces: {e}")
            # Keep default spaces as fallback
            self._spaces_initialized = True
    
    @property
    def unwrapped(self):
        """Return the base environment."""
        # For compatibility with gym.Wrapper interface
        return self.current_env.unwrapped if self.current_env else self
    
    def reset(self, **kwargs):
        """
        Reset environment with next market view in rotation.
        
        Returns:
            Tuple of (observation, info) from the new episode
        """
        # Ensure market views are loaded and spaces are initialized
        if not self.market_views:
            raise RuntimeError(
                "No market views available for reset. "
                "Market views should be loaded during environment creation."
            )
        
        self._ensure_spaces_initialized()
        
        # Get next market view
        market_view = self.market_views[self.current_market_index]
        self.current_market_index = (self.current_market_index + 1) % len(self.market_views)
        
        # Create new environment instance for this episode
        self.current_env = MarketAgnosticKalshiEnv(
            market_view=market_view,
            config=self.config.env_config
        )
        
        # Reset the environment
        obs, info = self.current_env.reset(**kwargs)
        
        # Add session metadata to info
        info.update({
            'session_id': market_view.session_id,
            'market_ticker': market_view.target_market,
            'episode_length': market_view.get_episode_length(),
            'market_coverage': getattr(market_view, 'market_coverage', 0.0),
            'avg_spread': market_view.avg_spread
        })
        
        logger.debug(f"Reset to market {market_view.target_market} "
                    f"(session {market_view.session_id}, {market_view.get_episode_length()} steps)")
        
        return obs, info
    
    def step(self, action):
        """
        Execute action in current environment.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self.current_env:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        
        # REMOVED: Artificial episode length truncation 
        # Episodes now run to natural completion (end of session data or bankruptcy)
        # This ensures full market data utilization and natural strategy learning
        
        return obs, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        """Render current environment."""
        if self.current_env:
            return self.current_env.render(*args, **kwargs)
    
    def close(self):
        """Close environment."""
        if self.current_env:
            self.current_env.close()
    
    def get_market_rotation_info(self) -> Dict[str, Any]:
        """
        Get information about market rotation and session coverage.
        
        Returns:
            Dict with rotation statistics
        """
        if not self.market_views:
            return {'total_markets': 0, 'sessions_covered': []}
        
        sessions_covered = list(set(mv.session_id for mv in self.market_views))
        
        return {
            'total_markets': len(self.market_views),
            'sessions_covered': sessions_covered,
            'current_market_index': self.current_market_index,
            'current_market': (
                self.market_views[self.current_market_index].target_market 
                if self.market_views else None
            )
        }


class CurriculumEnvironmentFactory:
    """
    Factory for creating session-based environments with curriculum learning.
    
    This factory provides convenient methods for creating SB3-compatible
    environments from session data, with support for both single-session
    and multi-session training.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize environment factory.
        
        Args:
            database_url: Database connection string
        """
        self.database_url = database_url
        self.data_loader = SessionDataLoader(database_url=database_url)
    
    async def create_single_session_env(self, 
                                      session_id: int,
                                      config: Optional[SB3TrainingConfig] = None) -> SessionBasedEnvironment:
        """
        Create environment for training on a single session.
        
        Args:
            session_id: Session to train on
            config: Training configuration
            
        Returns:
            SessionBasedEnvironment instance
        """
        config = config or SB3TrainingConfig()
        
        # Validate session exists
        session_data = await self.data_loader.load_session(session_id)
        if not session_data:
            raise ValueError(f"Session {session_id} not found or could not be loaded")
        
        logger.info(f"Creating single-session environment for session {session_id}")
        
        return SessionBasedEnvironment(
            database_url=self.database_url,
            session_ids=session_id,
            config=config
        )
    
    async def create_multi_session_env(self,
                                     session_ids: List[int],
                                     config: Optional[SB3TrainingConfig] = None) -> SessionBasedEnvironment:
        """
        Create environment for curriculum learning across multiple sessions.
        
        Args:
            session_ids: List of session IDs to cycle through
            config: Training configuration
            
        Returns:
            SessionBasedEnvironment instance
        """
        config = config or SB3TrainingConfig()
        
        # Validate sessions exist
        valid_sessions = []
        for session_id in session_ids:
            session_data = await self.data_loader.load_session(session_id)
            if session_data:
                valid_sessions.append(session_id)
            elif not config.skip_failed_markets:
                raise ValueError(f"Session {session_id} not found or could not be loaded")
        
        if not valid_sessions:
            raise ValueError("No valid sessions found")
        
        logger.info(f"Creating multi-session environment for {len(valid_sessions)} sessions")
        
        return SessionBasedEnvironment(
            database_url=self.database_url,
            session_ids=valid_sessions,
            config=config
        )
    
    async def create_env_from_curriculum(self,
                                       session_ids: Union[int, List[int]],
                                       config: Optional[SB3TrainingConfig] = None) -> SessionBasedEnvironment:
        """
        Create environment using curriculum learning approach.
        
        This method automatically determines whether to use single-session
        or multi-session training based on the input.
        
        Args:
            session_ids: Single session ID or list of session IDs
            config: Training configuration
            
        Returns:
            SessionBasedEnvironment instance
        """
        if isinstance(session_ids, int):
            return await self.create_single_session_env(session_ids, config)
        else:
            return await self.create_multi_session_env(session_ids, config)
    
    async def get_available_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of available sessions for training.
        
        Returns:
            List of session metadata dictionaries
        """
        return await self.data_loader.get_available_sessions()


# Convenience functions for integration

async def create_sb3_env(database_url: str,
                        session_ids: Union[int, List[int]],
                        config: Optional[SB3TrainingConfig] = None) -> SessionBasedEnvironment:
    """
    Convenience function to create SB3-compatible environment.
    
    Args:
        database_url: Database connection string
        session_ids: Single session ID or list of session IDs
        config: Training configuration
        
    Returns:
        SessionBasedEnvironment ready for SB3 training
    """
    factory = CurriculumEnvironmentFactory(database_url)
    env = await factory.create_env_from_curriculum(session_ids, config)
    
    # Load market views immediately to avoid async issues during training
    await env._load_market_views()
    
    # Validate spaces now that we have market views
    env._ensure_spaces_initialized()
    
    return env


def create_env_config(cash_start: int = 10000,
                     max_markets: int = 1,
                     temporal_features: bool = True) -> EnvConfig:
    """
    Create environment configuration with common settings.
    
    Args:
        cash_start: Starting cash in cents
        max_markets: Maximum markets (should be 1 for market-agnostic training)
        temporal_features: Enable temporal feature extraction
        
    Returns:
        EnvConfig instance
    """
    return EnvConfig(
        max_markets=max_markets,
        temporal_features=temporal_features,
        cash_start=cash_start
    )


def create_training_config(min_episode_length: int = 10,
                          max_episode_steps: Optional[int] = None,
                          skip_failed_markets: bool = True) -> SB3TrainingConfig:
    """
    Create training configuration with common settings.
    
    Args:
        min_episode_length: Minimum episode length for valid markets
        max_episode_steps: Maximum steps per episode (None for no limit)
        skip_failed_markets: Skip markets that fail to initialize
        
    Returns:
        SB3TrainingConfig instance
    """
    return SB3TrainingConfig(
        min_episode_length=min_episode_length,
        max_episode_steps=max_episode_steps,
        skip_failed_markets=skip_failed_markets
    )