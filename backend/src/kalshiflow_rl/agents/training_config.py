"""
Training configuration validation and management for Kalshi Flow RL Trading Subsystem.

Provides standardized hyperparameter configurations for different RL algorithms,
validation of training setups, and templates for multi-market training scenarios.
Enforces architectural constraints and ensures consistent training environments.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..config import config

logger = logging.getLogger("kalshiflow_rl.training_config")


class AlgorithmType(Enum):
    """Supported RL algorithms."""
    PPO = "PPO"
    A2C = "A2C"


class TrainingMode(Enum):
    """Training modes."""
    TRAINING = "training"  # Training mode using historical data
    PAPER = "paper"       # Paper trading mode


@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None


@dataclass
class A2CConfig:
    """A2C-specific hyperparameters."""
    learning_rate: float = 7e-4
    n_steps: int = 5
    gamma: float = 0.99
    gae_lambda: float = 1.0
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rms_prop_eps: float = 1e-5
    use_rms_prop: bool = True
    use_sde: bool = False
    normalize_advantage: bool = False


@dataclass
class TrainingConfig:
    """Complete training configuration for RL models."""
    
    # Model identification
    model_name: str
    version: str
    algorithm: AlgorithmType
    market_tickers: List[str]
    
    # Training parameters
    total_timesteps: int = 100000
    eval_freq: int = 5000
    eval_episodes: int = 10
    save_freq: int = 10000
    
    # Algorithm-specific hyperparameters
    algorithm_config: Union[PPOConfig, A2CConfig] = field(default_factory=PPOConfig)
    
    # Environment configuration
    env_config: Dict[str, Any] = field(default_factory=dict)
    
    # Training constraints and safety
    training_mode: TrainingMode = TrainingMode.TRAINING
    max_training_hours: float = 24.0
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.001
    
    # Multi-market specific
    market_rotation: bool = False  # Whether to rotate markets during training
    market_weights: Optional[Dict[str, float]] = None  # Market sampling weights
    
    # Performance targets
    target_return: Optional[float] = None
    target_sharpe: Optional[float] = None
    target_win_rate: Optional[float] = None
    
    # Model management
    checkpoint_frequency: int = 1000
    max_checkpoints: int = 5
    auto_deploy_threshold: Optional[float] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid training configuration: {', '.join(errors)}")
    
    def validate(self) -> List[str]:
        """Validate training configuration."""
        errors = []
        
        # Basic validation
        if not self.model_name:
            errors.append("model_name is required")
        
        if not self.version:
            errors.append("version is required")
        
        if not isinstance(self.algorithm, AlgorithmType):
            errors.append("algorithm must be an AlgorithmType")
        
        if not self.market_tickers:
            errors.append("At least one market_ticker is required")
        
        # Validate market tickers
        for ticker in self.market_tickers:
            if not isinstance(ticker, str) or not ticker.strip():
                errors.append(f"Invalid market ticker: {ticker}")
        
        # Check multi-market limits
        if len(self.market_tickers) > 10:
            errors.append("Maximum 10 markets supported per training session")
        
        # Training parameter validation
        if self.total_timesteps <= 0:
            errors.append("total_timesteps must be positive")
        
        if self.eval_freq <= 0:
            errors.append("eval_freq must be positive")
        
        if self.save_freq <= 0:
            errors.append("save_freq must be positive")
        
        if self.max_training_hours <= 0:
            errors.append("max_training_hours must be positive")
        
        # Algorithm-specific validation
        if self.algorithm == AlgorithmType.PPO and not isinstance(self.algorithm_config, PPOConfig):
            errors.append("PPO algorithm requires PPOConfig")
        
        if self.algorithm == AlgorithmType.A2C and not isinstance(self.algorithm_config, A2CConfig):
            errors.append("A2C algorithm requires A2CConfig")
        
        # Training mode validation  
        if self.training_mode not in [TrainingMode.TRAINING, TrainingMode.PAPER]:
            errors.append(f"Invalid training mode: {self.training_mode}")
        
        # CRITICAL: Enforce architectural constraint - only training and paper modes allowed
        if self.training_mode.value not in config.ALLOWED_TRADING_MODES:
            errors.append(f"Training mode {self.training_mode.value} not allowed. Only {config.ALLOWED_TRADING_MODES} are permitted.")
        
        # Multi-market validation
        if self.market_weights:
            if set(self.market_weights.keys()) != set(self.market_tickers):
                errors.append("market_weights keys must match market_tickers")
            
            if not all(w > 0 for w in self.market_weights.values()):
                errors.append("All market weights must be positive")
        
        # Performance targets validation
        if self.target_return is not None and self.target_return < -1.0:
            errors.append("target_return cannot be less than -100%")
        
        if self.target_sharpe is not None and self.target_sharpe < 0.0:
            errors.append("target_sharpe must be non-negative")
        
        if self.target_win_rate is not None and not (0.0 <= self.target_win_rate <= 1.0):
            errors.append("target_win_rate must be between 0.0 and 1.0")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        config_dict = asdict(self)
        
        # Convert enums to strings
        config_dict['algorithm'] = self.algorithm.value
        config_dict['training_mode'] = self.training_mode.value
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create TrainingConfig from dictionary."""
        # Handle enum conversions
        if isinstance(config_dict.get('algorithm'), str):
            config_dict['algorithm'] = AlgorithmType(config_dict['algorithm'])
        
        if isinstance(config_dict.get('training_mode'), str):
            config_dict['training_mode'] = TrainingMode(config_dict['training_mode'])
        
        # Handle algorithm config
        algorithm = config_dict.get('algorithm', AlgorithmType.PPO)
        algorithm_config_dict = config_dict.get('algorithm_config', {})
        
        if algorithm == AlgorithmType.PPO:
            config_dict['algorithm_config'] = PPOConfig(**algorithm_config_dict)
        elif algorithm == AlgorithmType.A2C:
            config_dict['algorithm_config'] = A2CConfig(**algorithm_config_dict)
        
        return cls(**config_dict)
    
    def get_sb3_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters in format expected by Stable Baselines3."""
        params = asdict(self.algorithm_config)
        
        # Add common parameters
        params.update({
            'gamma': params.get('gamma', 0.99),
            'verbose': 1,
            'seed': None,  # Will be set during training
        })
        
        return params


class TrainingConfigBuilder:
    """Builder class for creating training configurations."""
    
    @staticmethod
    def create_default_ppo_config(
        model_name: str,
        version: str,
        market_tickers: Union[str, List[str]],
        **kwargs
    ) -> TrainingConfig:
        """Create default PPO configuration."""
        if isinstance(market_tickers, str):
            market_tickers = [market_tickers]
        
        # Default PPO configuration optimized for trading
        ppo_config = PPOConfig(
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Small entropy for exploration
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        
        # Extract total_timesteps and eval_freq from kwargs if present to avoid duplicates
        total_timesteps = kwargs.pop('total_timesteps', 100000)
        eval_freq = kwargs.pop('eval_freq', 5000)
        
        return TrainingConfig(
            model_name=model_name,
            version=version,
            algorithm=AlgorithmType.PPO,
            market_tickers=market_tickers,
            algorithm_config=ppo_config,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            **kwargs
        )
    
    @staticmethod
    def create_default_a2c_config(
        model_name: str,
        version: str,
        market_tickers: Union[str, List[str]],
        **kwargs
    ) -> TrainingConfig:
        """Create default A2C configuration."""
        if isinstance(market_tickers, str):
            market_tickers = [market_tickers]
        
        # Default A2C configuration optimized for trading
        a2c_config = A2CConfig(
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=1.0,
            ent_coef=0.01,  # Small entropy for exploration
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        
        # Extract total_timesteps and eval_freq from kwargs if present to avoid duplicates
        total_timesteps = kwargs.pop('total_timesteps', 50000)  # A2C typically needs fewer steps
        eval_freq = kwargs.pop('eval_freq', 2500)
        
        return TrainingConfig(
            model_name=model_name,
            version=version,
            algorithm=AlgorithmType.A2C,
            market_tickers=market_tickers,
            algorithm_config=a2c_config,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            **kwargs
        )
    
    @staticmethod
    def create_multi_market_config(
        model_name: str,
        version: str,
        market_tickers: List[str],
        algorithm: AlgorithmType = AlgorithmType.PPO,
        market_rotation: bool = True,
        **kwargs
    ) -> TrainingConfig:
        """Create configuration optimized for multi-market training."""
        
        # Use appropriate default algorithm config
        if algorithm == AlgorithmType.PPO:
            base_config = TrainingConfigBuilder.create_default_ppo_config(
                model_name, version, market_tickers, **kwargs
            )
        else:
            base_config = TrainingConfigBuilder.create_default_a2c_config(
                model_name, version, market_tickers, **kwargs
            )
        
        # Multi-market specific adjustments
        base_config.market_rotation = market_rotation
        base_config.total_timesteps *= len(market_tickers)  # Scale with market count
        base_config.eval_freq *= len(market_tickers)
        
        # Equal weights for all markets by default
        base_config.market_weights = {ticker: 1.0 for ticker in market_tickers}
        
        return base_config
    
    @staticmethod
    def create_production_config(
        model_name: str,
        version: str,
        market_ticker: str,
        algorithm: AlgorithmType = AlgorithmType.PPO,
        **kwargs
    ) -> TrainingConfig:
        """Create production-ready configuration with conservative settings."""
        
        if algorithm == AlgorithmType.PPO:
            # Conservative PPO settings for production
            ppo_config = PPOConfig(
                learning_rate=1e-4,  # Lower learning rate
                n_steps=4096,        # Larger batch for stability
                batch_size=128,      # Larger batch size
                n_epochs=5,          # Fewer epochs to prevent overfitting
                gamma=0.995,         # Higher discount factor
                gae_lambda=0.95,
                clip_range=0.1,      # More conservative clipping
                ent_coef=0.005,      # Lower entropy
                vf_coef=0.5,
                max_grad_norm=0.25   # Lower gradient norm
            )
            
            return TrainingConfig(
                model_name=model_name,
                version=version,
                algorithm=AlgorithmType.PPO,
                market_tickers=[market_ticker],
                algorithm_config=ppo_config,
                total_timesteps=200000,  # More training steps
                eval_freq=10000,         # Less frequent evaluation
                early_stopping_patience=20,  # More patience
                **kwargs
            )
        
        else:  # A2C
            a2c_config = A2CConfig(
                learning_rate=3e-4,  # Lower learning rate
                n_steps=8,           # Slightly larger steps
                gamma=0.995,         # Higher discount factor
                ent_coef=0.005,      # Lower entropy
                vf_coef=0.5,
                max_grad_norm=0.25   # Lower gradient norm
            )
            
            return TrainingConfig(
                model_name=model_name,
                version=version,
                algorithm=AlgorithmType.A2C,
                market_tickers=[market_ticker],
                algorithm_config=a2c_config,
                total_timesteps=100000,  # More training steps
                eval_freq=5000,          # Less frequent evaluation
                early_stopping_patience=20,  # More patience
                **kwargs
            )


def validate_training_setup(
    training_config: TrainingConfig,
    env_config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate complete training setup including environment compatibility.
    
    Args:
        training_config: Training configuration to validate
        env_config: Environment configuration (optional)
        
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    # Validate training config
    config_errors = training_config.validate()
    errors.extend(config_errors)
    
    # Validate environment compatibility
    if env_config:
        # Check if market tickers match
        env_markets = env_config.get('market_tickers', [])
        if set(training_config.market_tickers) != set(env_markets):
            errors.append("Training config market_tickers must match environment market_tickers")
        
        # Check observation space compatibility with algorithm
        obs_config = env_config.get('observation_config', {})
        max_markets = obs_config.get('max_markets', 10)
        if len(training_config.market_tickers) > max_markets:
            errors.append(f"Training uses {len(training_config.market_tickers)} markets but environment supports max {max_markets}")
    
    # Architecture constraint validation
    if training_config.training_mode.value not in config.ALLOWED_TRADING_MODES:
        errors.append(f"Training mode {training_config.training_mode.value} violates architectural constraints")
    
    # Resource validation
    if training_config.total_timesteps > 1000000:
        errors.append("total_timesteps > 1M may cause resource issues")
    
    if len(training_config.market_tickers) > 5:
        errors.append("Training with >5 markets simultaneously may cause performance issues")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_recommended_config(
    market_ticker: str,
    model_complexity: str = "medium"
) -> TrainingConfig:
    """
    Get recommended training configuration based on market and complexity.
    
    Args:
        market_ticker: Market to train on
        model_complexity: 'simple', 'medium', 'complex'
        
    Returns:
        Recommended TrainingConfig
    """
    # Generate model name based on market and complexity
    model_name = f"kalshi_{market_ticker.replace('-', '_').lower()}_{model_complexity}"
    version = "v1.0"
    
    if model_complexity == "simple":
        # Fast training with A2C
        return TrainingConfigBuilder.create_default_a2c_config(
            model_name=model_name,
            version=version,
            market_tickers=[market_ticker],
            total_timesteps=25000,
            eval_freq=2500
        )
    
    elif model_complexity == "medium":
        # Balanced PPO configuration
        return TrainingConfigBuilder.create_default_ppo_config(
            model_name=model_name,
            version=version,
            market_tickers=[market_ticker],
            total_timesteps=100000,
            eval_freq=5000
        )
    
    else:  # complex
        # Production-ready configuration
        return TrainingConfigBuilder.create_production_config(
            model_name=model_name,
            version=version,
            market_ticker=market_ticker,
            total_timesteps=200000,
            eval_freq=10000
        )


# Pre-defined configurations for common scenarios
QUICK_TEST_CONFIG = TrainingConfig(
    model_name="quick_test",
    version="v1.0",
    algorithm=AlgorithmType.A2C,
    market_tickers=["INXD-25JAN03"],
    algorithm_config=A2CConfig(learning_rate=1e-3, n_steps=5),
    total_timesteps=1000,
    eval_freq=500,
    save_freq=500
)

MULTI_MARKET_CONFIG = TrainingConfigBuilder.create_multi_market_config(
    model_name="multi_market",
    version="v1.0",
    market_tickers=["INXD-25JAN03", "KXCABOUT-29"],
    algorithm=AlgorithmType.PPO,
    market_rotation=True
)