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


class QuantHardCodedAction(ActionSelector):
    """
    Aggressive quantitative hardcoded action selector for low-volume environments.
    
    Strategy Design (AGGRESSIVE TRADING - 70-80% trade rate):
    - Accepts wider spreads (up to 10 cents) to ensure trading happens
    - Uses microprice divergence as primary signal (strongest predictor)
    - Aggressive mean reversion: Buy < 40¢, Sell > 60¢
    - Momentum following during activity bursts
    - Liquidity imbalance exploitation
    - Only holds when spreads are extreme (>10 cents) or no liquidity
    
    This strategy is designed for LOW-VOLUME ENVIRONMENTS where opportunities
    are scarce and we need to be aggressive to capture them.
    """
    
    def __init__(self):
        """Initialize aggressive quant selector with trading statistics tracking."""
        self.action_count = 0
        self.hold_count = 0
        self.action_history = []
        
        # Aggressive thresholds for low-volume environments
        self.max_acceptable_spread = 0.10  # 10 cents (very wide)
        self.mean_reversion_buy_threshold = 0.40  # Buy below 40 cents
        self.mean_reversion_sell_threshold = 0.60  # Sell above 60 cents
        self.microprice_divergence_threshold = 0.02  # 2 cent divergence triggers signal
        self.momentum_threshold = 0.3  # Moderate momentum threshold
        
        logger.info("Aggressive QuantHardCodedAction initialized")
        logger.info(f"Max spread: {self.max_acceptable_spread*100:.0f}¢, "
                   f"Mean reversion: Buy<{self.mean_reversion_buy_threshold*100:.0f}¢, "
                   f"Sell>{self.mean_reversion_sell_threshold*100:.0f}¢")
    
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        """
        Select action using aggressive quantitative strategy.
        
        Analyzes multiple market signals to make aggressive trading decisions
        in low-volume environments. Prioritizes trading over holding.
        
        Args:
            observation: 52-feature market observation
            market_ticker: Market ticker (for logging)
        
        Returns:
            int: Action ID (0=HOLD, 1=BUY_YES, 2=SELL_YES, 3=BUY_NO, 4=SELL_NO)
        """
        self.action_count += 1
        
        # Extract key features from observation
        # Market features (indices 0-27)
        best_yes_bid = observation[0] if len(observation) > 0 else 0.5
        best_yes_ask = observation[1] if len(observation) > 1 else 0.5
        best_no_bid = observation[2] if len(observation) > 2 else 0.5
        best_no_ask = observation[3] if len(observation) > 3 else 0.5
        yes_mid = observation[4] if len(observation) > 4 else 0.5
        no_mid = observation[5] if len(observation) > 5 else 0.5
        
        # Spread features
        yes_spread = observation[6] if len(observation) > 6 else 0.02
        no_spread = observation[7] if len(observation) > 7 else 0.02
        
        # Advanced microstructure features
        yes_microprice = observation[14] if len(observation) > 14 else yes_mid
        no_microprice = observation[15] if len(observation) > 15 else no_mid
        volume_imbalance = observation[17] if len(observation) > 17 else 0.0
        bid_ask_depth_ratio = observation[19] if len(observation) > 19 else 0.0
        
        # Temporal features (indices 28-37)
        activity_burst = observation[32] if len(observation) > 32 else 0.0
        price_momentum = observation[34] if len(observation) > 34 else 0.0
        
        # === AGGRESSIVE TRADING LOGIC ===
        
        # 1. CHECK SPREAD CONDITIONS (very lenient)
        min_spread = min(yes_spread, no_spread)
        
        # Only reject if spreads are truly extreme or no liquidity
        if min_spread > self.max_acceptable_spread:
            # Even with wide spreads, only hold 50% of the time (stay aggressive)
            if np.random.random() < 0.5:
                logger.info(f"Wide spread {min_spread*100:.1f}¢ but trading anyway for {market_ticker}")
                # Trade on the side with better spread
                if yes_spread < no_spread:
                    action = LimitOrderActions.BUY_YES_LIMIT.value if yes_mid < 0.5 else LimitOrderActions.SELL_YES_LIMIT.value
                else:
                    action = LimitOrderActions.BUY_NO_LIMIT.value if no_mid < 0.5 else LimitOrderActions.SELL_NO_LIMIT.value
            else:
                self.hold_count += 1
                logger.info(f"HOLD due to extreme spread {min_spread*100:.1f}¢ for {market_ticker}")
                action = LimitOrderActions.HOLD.value
        
        # 2. CALCULATE TRADING SIGNALS
        else:
            # A. Microprice Divergence Signal (strongest predictor)
            yes_micro_divergence = yes_microprice - yes_mid
            no_micro_divergence = no_microprice - no_mid
            
            # B. Mean Reversion Signal (aggressive thresholds)
            yes_mean_reversion_buy = yes_mid < self.mean_reversion_buy_threshold
            yes_mean_reversion_sell = yes_mid > self.mean_reversion_sell_threshold
            no_mean_reversion_buy = no_mid < self.mean_reversion_buy_threshold
            no_mean_reversion_sell = no_mid > self.mean_reversion_sell_threshold
            
            # C. Momentum Signal (follow trends during high activity)
            momentum_signal = price_momentum if activity_burst > 0.5 else 0
            
            # D. Liquidity Imbalance Signal (trade against imbalance)
            liquidity_signal = -volume_imbalance  # Trade against the imbalance
            
            # === COMPOSITE SCORING SYSTEM ===
            buy_yes_score = 0.0
            sell_yes_score = 0.0
            buy_no_score = 0.0
            sell_no_score = 0.0
            
            # Microprice signals (strongest weight)
            if yes_micro_divergence > self.microprice_divergence_threshold:
                buy_yes_score += 3.0  # Microprice above mid = bullish
            elif yes_micro_divergence < -self.microprice_divergence_threshold:
                sell_yes_score += 3.0  # Microprice below mid = bearish
                
            if no_micro_divergence > self.microprice_divergence_threshold:
                buy_no_score += 3.0
            elif no_micro_divergence < -self.microprice_divergence_threshold:
                sell_no_score += 3.0
            
            # Mean reversion signals (strong weight)
            if yes_mean_reversion_buy:
                buy_yes_score += 2.5
            if yes_mean_reversion_sell:
                sell_yes_score += 2.5
            if no_mean_reversion_buy:
                buy_no_score += 2.5
            if no_mean_reversion_sell:
                sell_no_score += 2.5
            
            # Momentum signals (moderate weight during high activity)
            if activity_burst > 0.5:
                if momentum_signal > self.momentum_threshold:
                    buy_yes_score += 1.5
                    sell_no_score += 1.5
                elif momentum_signal < -self.momentum_threshold:
                    sell_yes_score += 1.5
                    buy_no_score += 1.5
            
            # Liquidity imbalance signals
            if liquidity_signal > 0.2:
                buy_yes_score += 1.0
                buy_no_score += 0.5
            elif liquidity_signal < -0.2:
                sell_yes_score += 1.0
                sell_no_score += 0.5
            
            # Spread-adjusted scoring (prefer tighter spreads)
            if yes_spread < no_spread:
                buy_yes_score *= 1.2
                sell_yes_score *= 1.2
            else:
                buy_no_score *= 1.2
                sell_no_score *= 1.2
            
            # === SELECT ACTION BASED ON SCORES ===
            scores = {
                'buy_yes': buy_yes_score,
                'sell_yes': sell_yes_score,
                'buy_no': buy_no_score,
                'sell_no': sell_no_score
            }
            
            max_score = max(scores.values())
            
            # Aggressive: Only hold if no signal at all (very rare)
            if max_score < 0.5:  # Very low threshold
                # Even with weak signals, trade 70% of the time
                if np.random.random() < 0.7:
                    # Random aggressive trade based on slight preferences
                    if yes_mid < 0.5:
                        action = LimitOrderActions.BUY_YES_LIMIT.value
                    elif yes_mid > 0.5:
                        action = LimitOrderActions.SELL_YES_LIMIT.value
                    else:
                        # Exactly at 0.5, trade based on spread
                        action = LimitOrderActions.BUY_YES_LIMIT.value if yes_spread < no_spread else LimitOrderActions.BUY_NO_LIMIT.value
                    logger.info(f"Weak signals but trading aggressively: action {action} for {market_ticker}")
                else:
                    self.hold_count += 1
                    action = LimitOrderActions.HOLD.value
            else:
                # Execute highest scoring action
                best_action = max(scores, key=scores.get)
                action_map = {
                    'buy_yes': LimitOrderActions.BUY_YES_LIMIT.value,
                    'sell_yes': LimitOrderActions.SELL_YES_LIMIT.value,
                    'buy_no': LimitOrderActions.BUY_NO_LIMIT.value,
                    'sell_no': LimitOrderActions.SELL_NO_LIMIT.value
                }
                action = action_map[best_action]
                
                logger.info(f"Quant signal: {best_action} (score={max_score:.2f}) for {market_ticker} "
                          f"[mid={yes_mid:.2f}, micro={yes_microprice:.2f}, momentum={price_momentum:.2f}]")
        
        # Track action history
        self.action_history.append({
            'action': action,
            'market_ticker': market_ticker,
            'timestamp': self.action_count,
            'yes_mid': yes_mid,
            'yes_spread': yes_spread,
            'signals': {
                'microprice_div': yes_microprice - yes_mid if 'yes_microprice' in locals() else 0,
                'momentum': price_momentum if 'price_momentum' in locals() else 0,
                'volume_imb': volume_imbalance if 'volume_imbalance' in locals() else 0
            }
        })
        
        return action
    
    def get_strategy_name(self) -> str:
        """Return strategy name with statistics."""
        hold_rate = (self.hold_count / max(self.action_count, 1)) * 100
        return f"Aggressive_Quant(HOLD={hold_rate:.1f}%_Target<30%)"
    
    def get_action_statistics(self) -> dict:
        """Get detailed statistics about action distribution and signals."""
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
        
        # Calculate average signals
        recent_history = self.action_history[-100:] if len(self.action_history) > 100 else self.action_history
        avg_signals = {}
        if recent_history:
            avg_signals = {
                'avg_microprice_divergence': np.mean([e['signals']['microprice_div'] for e in recent_history]),
                'avg_momentum': np.mean([e['signals']['momentum'] for e in recent_history]),
                'avg_volume_imbalance': np.mean([e['signals']['volume_imb'] for e in recent_history])
            }
        
        return {
            "total_actions": total,
            "hold_rate": f"{(self.hold_count / total * 100):.1f}%",
            "trade_rate": f"{((total - self.hold_count) / total * 100):.1f}%",
            "distribution": named_distribution,
            "raw_counts": action_counts,
            "recent_actions": self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history,
            "avg_signals": avg_signals,
            "strategy_type": "aggressive_quantitative"
        }


class HardcodedSelector(ActionSelector):
    """
    Active trading hardcoded action selector for comprehensive trading mechanics testing.
    
    Strategy Distribution (NO HOLD ACTIONS):
    - 25% BUY_YES (action 1) - Test YES position building
    - 25% SELL_YES (action 2) - Test YES position closing 
    - 25% BUY_NO (action 3) - Test NO position building
    - 25% SELL_NO (action 4) - Test NO position closing
    
    This strategy forces active trading on every decision to test order submission,
    position synchronization, and fill processing across all action types without
    relying on HOLD actions. Designed for testing trading pipeline mechanics.
    """
    
    def __init__(self):
        """Initialize hardcoded selector with active trading distribution."""
        self.action_count = 0
        self.action_history = []
        logger.info("Active HardcodedSelector initialized - NO HOLD ACTIONS")
        logger.info("Strategy: 25% BUY_YES, 25% SELL_YES, 25% BUY_NO, 25% SELL_NO")
    
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        """
        Select action using equal distribution across trading actions only.
        
        Uses simple cycling pattern to ensure equal distribution and predictable
        testing of all trading mechanics without any HOLD actions.
        
        Args:
            observation: Market observation (used for basic validity checks)
            market_ticker: Market ticker (used for logging)
        
        Returns:
            int: Action ID (1-4) selected according to equal trading distribution
        """
        self.action_count += 1
        
        # Simple 4-cycle pattern for equal distribution
        # No HOLD actions - always force trading decisions
        cycle_position = self.action_count % 4
        
        # Equal distribution pattern over 4 cycles:
        # BUY_YES, SELL_YES, BUY_NO, SELL_NO (rotating)
        if cycle_position == 1:
            action = LimitOrderActions.BUY_YES_LIMIT.value
        elif cycle_position == 2:
            action = LimitOrderActions.SELL_YES_LIMIT.value
        elif cycle_position == 3:
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
        
        # Log all actions since they're all trading actions now
        action_names = {
            1: "BUY_YES",
            2: "SELL_YES", 
            3: "BUY_NO",
            4: "SELL_NO"
        }
        logger.info(f"Active trading action: {action_names[action]} for {market_ticker} "
                   f"(cycle {cycle_position}/4, total actions: {self.action_count})")
        
        return action
    
    def get_strategy_name(self) -> str:
        """Return strategy name with distribution info."""
        return "Active_Hardcoded(NO_HOLD_25%_Each_Trading)"
    
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
        
        # Updated action names without HOLD (only trading actions)
        action_names = {
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
            "recent_actions": self.action_history[-10:] if len(self.action_history) >= 10 else self.action_history,
            "strategy_type": "active_trading_no_hold"
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
    
    elif strategy == "quant_hardcoded":
        logger.info("Creating aggressive QuantHardCodedAction selector")
        return QuantHardCodedAction()
    
    elif strategy in ("hardcoded", "disabled"):
        logger.info(f"Creating HardcodedSelector (strategy: {strategy})")
        return HardcodedSelector()
    
    else:
        logger.warning(
            f"Unknown strategy '{strategy}', defaulting to HardcodedSelector"
        )
        return HardcodedSelector()


