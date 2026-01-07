"""
Strategy implementations for point-in-time backtesting.

All strategies must implement the Strategy protocol defined in protocol.py.

Available Strategies:
    - RLMNoStrategy: Reverse Line Movement - bet NO when YES dominates but price drops
    - SLateTimingStrategy: Late-Arriving Large Money - bet NO when large trades arrive late
    - create_rlm_strategy: Factory function for custom RLM parameters
    - create_slate_strategy: Factory function for custom S-LATE parameters

Usage:
    from research.backtest.strategies import RLMNoStrategy, SLateTimingStrategy

    # Default parameters (matches production)
    rlm = RLMNoStrategy()
    slate = SLateTimingStrategy()

    # Custom parameters for sensitivity analysis
    rlm = create_rlm_strategy(yes_threshold=0.80, min_price_drop=10)
    slate = create_slate_strategy(min_trades=100, large_threshold_dollars=75)
"""

from .protocol import Strategy
from .rlm_no import RLMNoStrategy, create_rlm_strategy
from .s_late_timing import SLateTimingStrategy, create_slate_strategy

__all__ = [
    'Strategy',
    'RLMNoStrategy',
    'create_rlm_strategy',
    'SLateTimingStrategy',
    'create_slate_strategy',
]
