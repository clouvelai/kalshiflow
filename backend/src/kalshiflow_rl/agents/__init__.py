"""
RL Agents module for Kalshi Flow RL Trading Subsystem.

Provides high-level interfaces for:
- Model registry and lifecycle management
- Training configuration and validation 
- SB3 training harness integration
- Multi-market model training
- Hot-reload and deployment management

Key architectural principles enforced:
- Training uses only historical data
- No live mode allowed (training/paper only)
- Non-blocking database operations
- Multi-market support with proper scaling
"""

from .model_registry import (
    ModelRegistry,
    ModelConfig,
    model_registry
)

from .training_config import (
    TrainingConfig,
    PPOConfig,
    A2CConfig,
    AlgorithmType,
    TrainingMode,
    TrainingConfigBuilder,
    validate_training_setup,
    get_recommended_config,
    QUICK_TEST_CONFIG,
    MULTI_MARKET_CONFIG
)

from .training_harness import (
    TrainingSession,
    TrainingManager,
    TrainingCallback,
    training_manager
)

from .training_monitor import (
    TrainingMonitor,
    TrainingProgressCallback,
    PerformanceMetrics,
    create_training_monitor
)

from .session_manager import (
    SessionManager,
    SessionState,
    SessionStatus,
    session_manager
)

__all__ = [
    # Model Registry
    'ModelRegistry',
    'ModelConfig', 
    'model_registry',
    
    # Training Configuration
    'TrainingConfig',
    'PPOConfig',
    'A2CConfig',
    'AlgorithmType',
    'TrainingMode',
    'TrainingConfigBuilder',
    'validate_training_setup',
    'get_recommended_config',
    'QUICK_TEST_CONFIG',
    'MULTI_MARKET_CONFIG',
    
    # Training Harness
    'TrainingSession',
    'TrainingManager',
    'TrainingCallback',
    'training_manager',
    
    # Training Monitor
    'TrainingMonitor',
    'TrainingProgressCallback',
    'PerformanceMetrics',
    'create_training_monitor',
    
    # Session Manager
    'SessionManager',
    'SessionState',
    'SessionStatus',
    'session_manager'
]