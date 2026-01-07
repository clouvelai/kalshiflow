"""
Strategy protocol (interface) for point-in-time backtesting.

All trading strategies must implement this protocol to be used with
the PointInTimeBacktester.

Design Philosophy:
    The protocol is intentionally minimal to allow maximum flexibility
    in strategy implementation. Strategies can be as simple as a price
    threshold check or as complex as a multi-factor model.

    The key constraint is that on_trade() must ONLY use information
    available at the time of the trade - no looking at future trades
    or final settlement results.

Example Implementation:
    class SimpleYesBuyStrategy:
        name = "simple_yes_buy"

        def __init__(self, price_threshold: int = 85):
            self.price_threshold = price_threshold

        def on_trade(self, trade, state):
            # Buy YES when price >= threshold
            if trade.yes_price >= self.price_threshold:
                return SignalEntry(
                    market_ticker=trade.market_ticker,
                    signal_time=trade.timestamp,
                    entry_price_cents=trade.yes_price,
                    side='yes',
                    signal_strength=1.0,
                    metadata={'trigger': 'price_threshold'}
                )
            return None

        def get_parameters(self):
            return {'price_threshold': self.price_threshold}

        def reset(self):
            pass  # No internal state to reset
"""

from typing import Protocol, Optional, Dict, Any, runtime_checkable
import sys
import os

# Handle both package and direct imports
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

try:
    from ..state import Trade, MarketState, SignalEntry
except ImportError:
    from state import Trade, MarketState, SignalEntry


@runtime_checkable
class Strategy(Protocol):
    """
    Interface for any trading strategy.

    This protocol defines the contract that all strategies must fulfill
    to work with the PointInTimeBacktester.

    Attributes:
        name: Unique identifier for the strategy (used in logging/results)

    Methods:
        on_trade: Process a trade, optionally return a signal
        get_parameters: Return parameter dict for reproducibility
        reset: Clear internal state between backtests

    Implementation Notes:
        1. on_trade() is called AFTER state.update(trade), so state
           reflects the market AFTER the triggering trade occurred.

        2. Return None from on_trade() if no signal fires. Return a
           SignalEntry if the strategy wants to enter a position.

        3. For strategies that track internal state (e.g., moving averages),
           implement reset() to clear that state between backtests.

        4. get_parameters() should return ALL parameters that affect
           strategy behavior, enabling reproducible backtests.
    """

    name: str

    def on_trade(self, trade: Trade, state: MarketState) -> Optional[SignalEntry]:
        """
        Process a trade, return SignalEntry if signal fires.

        This is the core strategy logic. It's called for every trade
        in chronological order.

        IMPORTANT: This is called AFTER state.update(trade), so state
        reflects the market AFTER this trade occurred. This is the
        correct behavior because:
        - The trade has happened (you can't un-see it)
        - You're deciding whether to enter AFTER this trade
        - Your entry would be at the next available price (approximated
          by current trade price in backtest)

        Args:
            trade: The trade that just occurred
            state: Current market state (AFTER this trade)

        Returns:
            SignalEntry if the strategy wants to enter a position
            None if no signal fires

        Example:
            def on_trade(self, trade, state):
                # Only consider markets with 5+ trades
                if state.total_trades < 5:
                    return None

                # Buy YES if price >= 85 and strong YES flow
                if trade.yes_price >= 85 and state.yes_ratio >= 0.7:
                    return SignalEntry(
                        market_ticker=trade.market_ticker,
                        signal_time=trade.timestamp,
                        entry_price_cents=trade.yes_price,
                        side='yes',
                        signal_strength=state.yes_ratio,
                        metadata={
                            'trigger': 'high_price_strong_flow',
                            'yes_ratio': state.yes_ratio,
                            'total_trades': state.total_trades
                        }
                    )
                return None
        """
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current parameter values for logging/reproducibility.

        This should return a dictionary containing ALL parameters that
        affect strategy behavior. This enables:
        - Reproducing backtest results
        - Comparing different parameter configurations
        - Logging strategy configuration in results

        Returns:
            Dictionary mapping parameter names to values

        Example:
            def get_parameters(self):
                return {
                    'price_min': self.price_min,
                    'price_max': self.price_max,
                    'min_trades': self.min_trades,
                    'yes_ratio_threshold': self.yes_ratio_threshold
                }
        """
        ...

    def reset(self) -> None:
        """
        Reset any internal state (called between backtests).

        This is called by the backtester before starting a new run.
        Strategies that maintain internal state (e.g., running calculations,
        learned parameters, cached data) should reset that state here.

        For stateless strategies that only look at current trade/state,
        this can be a no-op.

        Example (stateless strategy):
            def reset(self):
                pass  # No internal state

        Example (stateful strategy):
            def reset(self):
                self.price_history.clear()
                self.running_avg = None
                self.signals_fired = 0
        """
        ...


class BaseStrategy:
    """
    Optional base class providing common functionality for strategies.

    Strategies don't have to inherit from this - they just need to
    implement the Strategy protocol. But this base class provides
    sensible defaults and helper methods.

    Usage:
        class MyStrategy(BaseStrategy):
            name = "my_strategy"

            def __init__(self, threshold: int = 50):
                super().__init__()
                self.threshold = threshold

            def on_trade(self, trade, state):
                # Strategy logic here
                ...

            def get_parameters(self):
                return {'threshold': self.threshold}
    """

    name: str = "base_strategy"

    def __init__(self):
        """Initialize base strategy."""
        self._signals_fired: int = 0

    def on_trade(self, trade: Trade, state: MarketState) -> Optional[SignalEntry]:
        """Default implementation returns None (no signal)."""
        return None

    def get_parameters(self) -> Dict[str, Any]:
        """Default implementation returns empty dict."""
        return {}

    def reset(self) -> None:
        """Reset internal state."""
        self._signals_fired = 0

    def _create_signal(
        self,
        trade: Trade,
        side: str,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SignalEntry:
        """
        Helper to create a SignalEntry with consistent formatting.

        Args:
            trade: The triggering trade
            side: 'yes' or 'no'
            strength: Signal confidence (0.0 to 1.0)
            metadata: Optional strategy-specific data

        Returns:
            SignalEntry ready to return from on_trade()
        """
        self._signals_fired += 1
        return SignalEntry(
            market_ticker=trade.market_ticker,
            signal_time=trade.timestamp,
            entry_price_cents=trade.yes_price,
            side=side,
            signal_strength=strength,
            metadata=metadata or {}
        )
