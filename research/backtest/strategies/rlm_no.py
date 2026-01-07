"""
RLM NO Strategy - Research Version for Point-in-Time Backtesting.

Purpose:
    Implements the Reverse Line Movement (RLM) strategy for backtesting.
    This strategy bets NO when YES trades dominate but the price drops.

Key Mechanics:
    - Signal: YES ratio >= threshold + price dropped from open + min trades
    - Entry: Current NO price at signal time
    - Exit: Hold to settlement (no early exits)

Production Parity:
    The research parameters MUST match the production configuration at:
    backend/src/kalshiflow_rl/traderv3/strategies/config/rlm_no.yaml

    Current production parameters:
    - yes_threshold: 0.70 (70% YES trades)
    - min_trades: 25 (minimum trades before signal)
    - min_price_drop: 5 (cents drop from open)
    - min_no_price: 35 (min entry price to avoid cheap contracts)

Validated Edge:
    - Overall: +17.38% on 1.7M+ trades across 72k+ unique markets
    - Bucket-matched edge must be positive across price tiers
    - Reference: research/strategies/h014_rlm_validation.md

Architecture Position:
    Used by PointInTimeBacktester to evaluate the RLM strategy:

        from research.backtest.strategies.rlm_no import RLMNoStrategy

        strategy = RLMNoStrategy()
        backtester = PointInTimeBacktester(strategy)
        results = backtester.run(trades, settlements)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
import sys
import os

# Add parent directory to path for imports
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Runtime imports for creating objects
try:
    from ..state import Trade, MarketState, SignalEntry
except ImportError:
    from state import Trade, MarketState, SignalEntry


@dataclass
class RLMNoStrategy:
    """
    RLM NO Strategy for research backtesting.

    Signal fires when:
    1. YES trade ratio >= yes_threshold (default: 70%)
    2. Total trades >= min_trades (default: 25)
    3. Price drop >= min_price_drop (default: 5 cents)
    4. NO price >= min_no_price (default: 35 cents)

    Parameters match production config at:
    backend/src/kalshiflow_rl/traderv3/strategies/config/rlm_no.yaml

    Usage:
        strategy = RLMNoStrategy()
        signal = strategy.on_trade(trade, state)
        if signal:
            print(f"Signal fired: {signal.market_ticker} @ {signal.entry_price_cents}c")
    """

    name: str = "rlm_no"

    # Signal thresholds - MUST MATCH PRODUCTION CONFIG
    yes_threshold: float = 0.70
    min_trades: int = 25
    min_price_drop: int = 5
    min_no_price: int = 35

    # Track which markets have already signaled (first-signal-only by default)
    _signaled_markets: Dict[str, bool] = field(default_factory=dict)
    allow_multiple_signals: bool = False  # Set True for re-entry testing

    def on_trade(self, trade: 'Trade', state: 'MarketState') -> Optional['SignalEntry']:
        """
        Check if RLM signal fires after this trade.

        This method is called for each trade in chronological order.
        The state parameter contains the market state AFTER processing
        this trade (point-in-time).

        Args:
            trade: The trade that just occurred
            state: Market state after this trade

        Returns:
            SignalEntry if signal fires, None otherwise
        """
        # Check if we've already signaled for this market
        if not self.allow_multiple_signals:
            if state.market_ticker in self._signaled_markets:
                return None

        # Check minimum trades
        if state.total_trades < self.min_trades:
            return None

        # Check YES ratio
        ratio = self._yes_ratio(state)
        if ratio < self.yes_threshold:
            return None

        # Check price drop
        drop = self._price_drop(state)
        if drop < self.min_price_drop:
            return None

        # Check NO price (avoid cheap contracts)
        no_price = self._current_no_price(state)
        if no_price is None or no_price < self.min_no_price:
            return None

        # Signal fires!
        self._signaled_markets[state.market_ticker] = True

        return SignalEntry(
            market_ticker=state.market_ticker,
            signal_time=trade.timestamp,
            entry_price_cents=no_price,  # We're buying NO
            side='no',
            signal_strength=self._compute_strength(ratio, drop),
            metadata={
                'strategy': self.name,
                'yes_ratio': round(ratio, 4),
                'price_drop': drop,
                'no_price': no_price,
                'total_trades': state.total_trades,
                'yes_trades': state.yes_trades,
                'no_trades': state.no_trades,
                'open_price': state.open_price,
                'last_yes_price': state.last_yes_price,
            }
        )

    def _yes_ratio(self, state: 'MarketState') -> float:
        """Compute YES trade ratio."""
        if state.total_trades == 0:
            return 0.0
        return state.yes_trades / state.total_trades

    def _price_drop(self, state: 'MarketState') -> int:
        """Compute price drop from open (positive = price went down)."""
        if state.open_price is None or state.last_yes_price is None:
            return 0
        return state.open_price - state.last_yes_price

    def _current_no_price(self, state: 'MarketState') -> Optional[int]:
        """Get current NO price (100 - last YES price)."""
        if state.last_yes_price is None:
            return None
        return 100 - state.last_yes_price

    def _compute_strength(self, ratio: float, drop: int) -> float:
        """
        Compute signal strength (0-1).

        Higher YES ratio and larger drop = stronger signal.
        This is informational; the strategy uses binary signal detection.

        Strength formula:
        - YES ratio 0.70-1.0 maps to 0-0.5
        - Price drop 5-30 maps to 0-0.5
        - Total: 0-1.0
        """
        # Normalize ratio: 0.70 -> 0, 1.0 -> 0.5
        ratio_score = min((ratio - 0.70) / 0.30, 1.0) * 0.5

        # Normalize drop: 5 -> 0, 30 -> 0.5
        drop_score = min((drop - 5) / 25, 1.0) * 0.5

        return ratio_score + drop_score

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return current parameters.

        Useful for logging and reproducibility.
        """
        return {
            'name': self.name,
            'yes_threshold': self.yes_threshold,
            'min_trades': self.min_trades,
            'min_price_drop': self.min_price_drop,
            'min_no_price': self.min_no_price,
            'allow_multiple_signals': self.allow_multiple_signals,
        }

    def reset(self) -> None:
        """
        Reset internal state for a new backtest run.

        Clears the set of signaled markets.
        """
        self._signaled_markets.clear()

    def get_signaled_markets(self) -> Dict[str, bool]:
        """Return the set of markets that have signaled."""
        return dict(self._signaled_markets)


def create_rlm_strategy(
    yes_threshold: float = 0.70,
    min_trades: int = 25,
    min_price_drop: int = 5,
    min_no_price: int = 35,
    allow_multiple_signals: bool = False,
) -> RLMNoStrategy:
    """
    Factory function to create RLM strategy with custom parameters.

    Use this for parameter sweeps and sensitivity analysis.

    Args:
        yes_threshold: Minimum YES ratio to trigger (default 0.70)
        min_trades: Minimum trades before signal (default 25)
        min_price_drop: Minimum price drop in cents (default 5)
        min_no_price: Minimum NO entry price in cents (default 35)
        allow_multiple_signals: Allow multiple signals per market (default False)

    Returns:
        Configured RLMNoStrategy instance

    Example:
        # Test with stricter threshold
        strategy = create_rlm_strategy(yes_threshold=0.80, min_price_drop=10)
        results = backtester.run(trades, settlements)
    """
    return RLMNoStrategy(
        yes_threshold=yes_threshold,
        min_trades=min_trades,
        min_price_drop=min_price_drop,
        min_no_price=min_no_price,
        allow_multiple_signals=allow_multiple_signals,
    )
