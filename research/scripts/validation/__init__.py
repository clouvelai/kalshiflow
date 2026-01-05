"""
Strategy Validation Framework for Kalshi Flow Research.

This package provides a standardized, automated validation pipeline for
trading strategy hypotheses. It replaces the manual ~400-line validation
scripts with a configuration-driven approach.

Core Components:
    - config: Strategy configuration schema and YAML parser
    - cache: Precomputed data layer for fast validation (10-20x speedup)
    - validator: LSD mode (quick) and Full mode (rigorous) validation
    - cli: Command-line interface for running validations

Usage:
    # Quick LSD screening
    python -m research.scripts.validate_strategy --config strategies/h123_rlm.yaml --mode lsd

    # Full validation
    python -m research.scripts.validate_strategy --config strategies/h123_rlm.yaml --mode full

Architecture:
    +-------------------+     +-------------------+     +-------------------+
    |  Strategy Config  | --> |  Validation Core  | --> |  Report Generator |
    |     (YAML)        |     |     (Python)      |     |   (JSON/Console)  |
    +-------------------+     +-------------------+     +-------------------+
                                      |
                                      v
                              +-------------------+
                              |  Cached Data Layer|
                              |    (Parquet)      |
                              +-------------------+
"""

from .config import StrategyConfig, SignalCondition, load_config
from .cache import CachedDataLayer
from .validator import StrategyValidator, ValidationResult

__all__ = [
    "StrategyConfig",
    "SignalCondition",
    "load_config",
    "CachedDataLayer",
    "StrategyValidator",
    "ValidationResult",
]
