"""
ActionSelector for Kalshi Trading Actor - M3 Modular Implementation.

Provides modular action selection with support for:
- RL model-based selection (cached PPO models)
- Hardcoded strategies (always hold, etc.)
- Factory function for strategy selection based on config

Design:
- Abstract ActionSelector interface for all selectors
- Model loading happens ONCE at initialization (never in hot path)
- Callable interface for ActorService integration
- Backward compatibility with M1-M2 stub (deprecated)
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path
import numpy as np

from ..environments.limit_order_action_space import LimitOrderActions, ActionType
from ..config import config

# Import Stable Baselines3 for model loading
try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    PPO = None

logger = logging.getLogger("kalshiflow_rl.trading.action_selector")


class ActionSelector(ABC):
    """
    Abstract base class for action selection strategies.
    
    All action selectors must implement:
    - select_action(): Choose action (0-20) based on observation
    - get_strategy_name(): Return descriptive strategy name
    
    Action space (21 actions total):
    - 0: HOLD
    - 1-5: BUY_YES (5, 10, 20, 50, 100 contracts)
    - 6-10: SELL_YES (5, 10, 20, 50, 100 contracts)  
    - 11-15: BUY_NO (5, 10, 20, 50, 100 contracts)
    - 16-20: SELL_NO (5, 10, 20, 50, 100 contracts)
    """
    
    @abstractmethod
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        """
        Select action based on observation.
        
        Args:
            observation: 52-feature observation vector from LiveObservationAdapter
            market_ticker: Market context (used for logging/debugging, not model input)
                          since model is market-agnostic
        
        Returns:
            int: Action ID (valid range depends on model's action space)
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Return strategy name for logging and identification.
        
        Returns:
            str: Descriptive strategy name
        """
        pass
    
    async def __call__(self, observation: np.ndarray, market_ticker: str) -> int:
        """
        Make ActionSelector callable for ActorService integration.
        
        Delegates to select_action() method.
        """
        return await self.select_action(observation, market_ticker)


class RLModelSelector(ActionSelector):
    """
    RL Model-based action selector using cached PPO model.
    
    Loads model ONCE at initialization and caches for all predictions.
    This ensures <1ms prediction times (model loading would be >100ms).
    """
    
    def __init__(self, model_path: str):
        """
        Initialize RL model selector.
        
        Args:
            model_path: Path to trained PPO model file (.zip)
        
        Raises:
            ValueError: If model file doesn't exist or SB3 not available
            RuntimeError: If model loading fails
        """
        if not SB3_AVAILABLE:
            raise RuntimeError(
                "Stable Baselines3 not available. Install with: pip install stable-baselines3"
            )
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise ValueError(f"Model file not found: {model_path}")
        
        logger.info(f"Loading RL model from: {model_path}")
        start_time = time.time()
        
        try:
            # Load model ONCE at initialization (never in hot path)
            self.model = PPO.load(str(model_file))
            self.model_path = str(model_file)
            self._model_loaded = True
            self._load_error = None
            
            # Derive action space bounds from the model itself
            self.action_space_n = self.model.action_space.n
            self.min_action = 0
            self.max_action = self.action_space_n - 1
            
            load_time = time.time() - start_time
            logger.info(
                f"✅ RL model loaded and cached successfully in {load_time:.3f}s. "
                f"Action space: {self.action_space_n} actions (0-{self.max_action})"
            )
            
        except Exception as e:
            self._model_loaded = False
            self._load_error = str(e)
            self.model = None
            self.action_space_n = None
            self.min_action = None
            self.max_action = None
            logger.error(f"❌ Failed to load RL model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        """
        Select action using cached RL model.
        
        Uses cached model for fast prediction (<1ms target).
        
        Args:
            observation: 52-feature observation array
            market_ticker: Market ticker (for logging only)
        
        Returns:
            int: Action ID (valid range derived from model), falls back to HOLD on error
        """
        if not self._model_loaded or self.model is None:
            logger.error(f"Model not loaded for {market_ticker}, returning HOLD")
            return LimitOrderActions.HOLD.value
        
        try:
            # Use cached model - no loading here (performance critical)
            action, _states = self.model.predict(observation, deterministic=True)
            action_int = int(action)
            
            # Validate action is in valid range (dynamically derived from model)
            if not (self.min_action <= action_int <= self.max_action):
                logger.warning(
                    f"Model returned invalid action {action_int} for {market_ticker} "
                    f"(valid range: {self.min_action}-{self.max_action}), "
                    f"returning HOLD"
                )
                return LimitOrderActions.HOLD.value
            
            return action_int
            
        except Exception as e:
            logger.error(f"Prediction error for {market_ticker}: {e}")
            # Fall back to HOLD on any prediction error
            return LimitOrderActions.HOLD.value
    
    def get_strategy_name(self) -> str:
        """Return strategy name with model path."""
        if self._model_loaded:
            return f"RL_Model({Path(self.model_path).name})"
        else:
            return f"RL_Model(FAILED: {self._load_error})"
    
    def is_model_loaded(self) -> bool:
        """Check if model is successfully loaded."""
        return self._model_loaded and self.model is not None


class HardcodedSelector(ActionSelector):
    """
    Enhanced hardcoded action selector for comprehensive trading mechanics testing.
    
    Strategy Distribution:
    - 75% HOLD (action 0) - Conservative baseline
    - 6.25% BUY_YES (action 1) - Test YES position building
    - 6.25% SELL_YES (action 2) - Test YES position closing 
    - 6.25% BUY_NO (action 3) - Test NO position building
    - 6.25% SELL_NO (action 4) - Test NO position closing
    
    This strategy exercises all trading mechanics with realistic position sizing
    while remaining conservative enough for safe testing on paper accounts.
    """
    
    def __init__(self):
        """Initialize hardcoded selector with action distribution."""
        self.action_count = 0
        self.action_history = []
        logger.info("Enhanced HardcodedSelector initialized with diversified action distribution")
        logger.info("Strategy: 75% HOLD, 25% trading across all 4 trade directions")
    
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        """
        Select action using weighted random distribution to exercise all mechanics.
        
        Uses deterministic cycling with randomness to ensure good coverage
        while being reproducible for testing purposes.
        
        Args:
            observation: Market observation (used for basic validity checks)
            market_ticker: Market ticker (used for logging)
        
        Returns:
            int: Action ID (0-4) selected according to strategy distribution
        """
        self.action_count += 1
        
        # Use a mix of deterministic cycling and basic randomness
        # This ensures all actions get tested while being somewhat predictable
        cycle_position = self.action_count % 16
        
        # Enhanced distribution pattern over 16 cycles:
        # 12 HOLD, 1 BUY_YES, 1 SELL_YES, 1 BUY_NO, 1 SELL_NO
        if cycle_position in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            action = LimitOrderActions.HOLD.value
        elif cycle_position == 13:
            action = LimitOrderActions.BUY_YES_LIMIT.value
        elif cycle_position == 14:
            action = LimitOrderActions.SELL_YES_LIMIT.value
        elif cycle_position == 15:
            action = LimitOrderActions.BUY_NO_LIMIT.value
        else:  # cycle_position == 0
            action = LimitOrderActions.SELL_NO_LIMIT.value
        
        # Track action history for assessment
        self.action_history.append({
            'action': action,
            'cycle_position': cycle_position,
            'market_ticker': market_ticker,
            'timestamp': self.action_count
        })
        
        # Log trading actions for monitoring
        if action != LimitOrderActions.HOLD.value:
            action_names = {
                1: "BUY_YES",
                2: "SELL_YES", 
                3: "BUY_NO",
                4: "SELL_NO"
            }
            logger.info(f"Trading action selected: {action_names[action]} for {market_ticker} "
                       f"(cycle {cycle_position}/16, total actions: {self.action_count})")
        
        return action
    
    def get_strategy_name(self) -> str:
        """Return strategy name with distribution info."""
        return "Enhanced_Hardcoded(75%_HOLD_25%_Trading)"
    
    def get_action_statistics(self) -> dict:
        """Get statistics about action distribution for assessment."""
        if not self.action_history:
            return {"total_actions": 0, "distribution": {}}
        
        total = len(self.action_history)
        action_counts = {}
        
        for entry in self.action_history:
            action = entry['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        distribution = {
            action: (count / total * 100) for action, count in action_counts.items()
        }
        
        action_names = {
            0: "HOLD",
            1: "BUY_YES", 
            2: "SELL_YES",
            3: "BUY_NO",
            4: "SELL_NO"
        }
        
        named_distribution = {
            action_names.get(action, f"ACTION_{action}"): f"{percentage:.1f}%"
            for action, percentage in distribution.items()
        }
        
        return {
            "total_actions": total,
            "distribution": named_distribution,
            "raw_counts": action_counts,
            "recent_actions": self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history
        }


# ===============================================================================
# Factory Function
# ===============================================================================

def create_action_selector(
    strategy: Optional[str] = None,
    model_path: Optional[str] = None
) -> ActionSelector:
    """
    Create action selector based on strategy configuration.
    
    Args:
        strategy: Strategy name ("rl_model", "hardcoded", "disabled")
                 If None, reads from config.RL_ACTOR_STRATEGY
        model_path: Path to model file (required for "rl_model" strategy)
                   If None, reads from config.RL_ACTOR_MODEL_PATH
    
    Returns:
        ActionSelector instance
    
    Raises:
        ValueError: If strategy is "rl_model" but model_path not provided
    """
    # Use config values if not provided
    if strategy is None:
        strategy = config.RL_ACTOR_STRATEGY
    
    if model_path is None:
        model_path = config.RL_ACTOR_MODEL_PATH
    
    # Normalize strategy name
    strategy = strategy.lower().strip()
    
    # Create appropriate selector
    if strategy == "rl_model":
        if not model_path or not str(model_path).strip():
            raise ValueError(
                "RL_ACTOR_MODEL_PATH must be provided for 'rl_model' strategy. "
                f"Current value: {model_path}"
            )
        logger.info(f"Creating RLModelSelector with model: {model_path}")
        return RLModelSelector(model_path)
    
    elif strategy in ("hardcoded", "disabled"):
        logger.info(f"Creating HardcodedSelector (strategy: {strategy})")
        return HardcodedSelector()
    
    else:
        logger.warning(
            f"Unknown strategy '{strategy}', defaulting to HardcodedSelector"
        )
        return HardcodedSelector()


