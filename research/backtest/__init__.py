"""
Point-in-Time Backtesting Engine

This module provides a backtesting framework that eliminates look-ahead bias
by processing trades chronologically and only using information available
at the moment a signal fires.

Key Principle:
    Traditional backtests often use FINAL market statistics (total trades,
    final prices, settlement results) when evaluating entry decisions.
    This creates look-ahead bias because that information wasn't available
    when the trading decision would have been made in real-time.

    This engine solves that by:
    1. Processing trades in strict chronological order
    2. Maintaining point-in-time state that only reflects past trades
    3. Firing signals at the FIRST moment conditions are met
    4. Recording entry prices at signal time, not final prices

Usage:
    from research.backtest import PointInTimeBacktester, Trade, MarketState
    from research.backtest.strategies import YourStrategy

    strategy = YourStrategy(param1=value1)
    backtester = PointInTimeBacktester(strategy)

    results = backtester.run(trades, settlements)
    print(f"Win rate: {results.win_rate:.1%}")
    print(f"Edge: {results.raw_edge:.1%}")

Components:
    - state.py: Core data structures (Trade, MarketState, SignalEntry)
    - engine.py: PointInTimeBacktester that processes trades chronologically
    - validation.py: Statistical validation of backtest results
    - signals.py: Reusable signal building blocks for strategies
    - strategies/: Strategy implementations following the Protocol interface
"""

from .state import Trade, MarketState, SignalEntry
from .engine import PointInTimeBacktester, BacktestResults
from .validation import (
    validate_results,
    ValidationReport,
    BucketStats,
    bucket_matched_validation,
    binomial_test,
    quick_validate,
)

__all__ = [
    # Core data structures
    'Trade',
    'MarketState',
    'SignalEntry',
    # Backtester
    'PointInTimeBacktester',
    'BacktestResults',
    # Validation
    'validate_results',
    'ValidationReport',
    'BucketStats',
    'bucket_matched_validation',
    'binomial_test',
    'quick_validate',
]
