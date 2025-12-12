"""
M10 RL System Diagnostics Module.

This module provides comprehensive instrumentation and diagnostics for the RL system
to understand why agents exhibit HOLD-only behavior and track training progress.

Key components:
- ActionTracker: Tracks action distribution and exploration patterns
- RewardAnalyzer: Analyzes reward signal quality and sparsity  
- ObservationValidator: Validates observation space quality
- DiagnosticsLogger: Consolidated logging to file and console
- M10Callback: SB3 callback for training-time diagnostics
"""

from .action_tracker import ActionTracker
from .reward_analyzer import RewardAnalyzer
from .observation_validator import ObservationValidator
from .diagnostics_logger import DiagnosticsLogger

__all__ = [
    'ActionTracker',
    'RewardAnalyzer', 
    'ObservationValidator',
    'DiagnosticsLogger'
]

# Optional SB3 callback import (requires stable_baselines3)
try:
    from .m10_callback import M10DiagnosticsCallback
    __all__.append('M10DiagnosticsCallback')
except ImportError:
    # stable_baselines3 not available - SB3 callback not available
    pass