"""
RL Agents module for Kalshi Flow RL Trading Subsystem.

Provides high-level interfaces for:
- Model registry and lifecycle management

Key architectural principles enforced:
- Training uses only historical data
- No live mode allowed (training/paper only)
- Non-blocking database operations
- Multi-market support with proper scaling

Note: Training configuration, harness, monitor, and session management
modules are planned but not yet implemented. The following are available:
- model_registry: Model versioning and checkpoint management
"""

from .model_registry import (
    ModelRegistry,
    ModelConfig,
    model_registry
)

# Note: The following modules are not yet implemented:
# - training_config (TrainingConfig, AlgorithmType, etc.)
# - training_harness (TrainingSession, TrainingManager, etc.)
# - training_monitor (TrainingMonitor, etc.)
# - session_manager (SessionManager, SessionState, etc.)
# The session_manager.py file exists but has dependencies on unimplemented modules.

__all__ = [
    # Model Registry
    'ModelRegistry',
    'ModelConfig', 
    'model_registry',
]