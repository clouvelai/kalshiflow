"""
S-LATE-001 Strategy - Late-Arriving Large Money

Signal: Bet NO when large trades arrive late and favor NO direction.

Detected edge: +19.8% (331 markets, 95.47% win rate, 11/11 bucket pass)
~34% overlap with RLM

Point-in-Time Implementation:
- Cannot know "final 25% of trades" in real-time
- Use trade count threshold: after N trades, check last 25%
- Signal fires ONCE when conditions are first met

Parameters (from deep dive validation):
- min_trades: Minimum trades before checking (default: 50)
- large_threshold_cents: Dollar threshold for "large" trade (default: 5000 = $50)
- ratio_threshold: Late large ratio must be this much higher than early (default: 2.0)
- late_direction_threshold: YES ratio threshold to determine NO direction (default: 0.4)
- late_large_min_ratio: Minimum large trade ratio in late window (default: 0.2)
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from state import Trade, MarketState, SignalEntry


@dataclass
class SLateTrade:
    """Track a trade for S-LATE analysis."""
    timestamp: datetime
    value_cents: int
    taker_side: str
    yes_price: int


@dataclass
class SLateMarketState:
    """Extended state for S-LATE signal detection."""
    trades: List[SLateTrade] = field(default_factory=list)
    signaled: bool = False


@dataclass
class SLateTimingStrategy:
    """
    S-LATE-001: Late-Arriving Large Money Strategy.

    Signal fires when:
    1. Total trades >= min_trades
    2. Late window (last 25%) has >2x large trade ratio vs early window
    3. Late large trades favor NO (YES ratio < 0.4)
    4. Late large trade ratio > 0.2

    This is a POINT-IN-TIME implementation that fires once per market
    at the first moment conditions are met.
    """

    name: str = "s_late_timing"

    # Trade count threshold (real-time proxy for "final 25%")
    min_trades: int = 50

    # Large trade threshold (in cents, $50 = 5000)
    large_threshold_cents: int = 5000

    # Late must have this much more large trades than early
    ratio_threshold: float = 2.0

    # YES ratio below this = NO direction
    late_direction_threshold: float = 0.4

    # Minimum large trade ratio in late window
    late_large_min_ratio: float = 0.2

    # Track trades per market (for early/late split)
    _market_states: Dict[str, SLateMarketState] = field(default_factory=dict)

    def on_trade(self, trade: Trade, state: MarketState) -> Optional[SignalEntry]:
        """
        Process a trade and check for S-LATE signal.

        Returns SignalEntry if signal fires, None otherwise.
        """
        ticker = trade.market_ticker

        # Get or create our extended state
        if ticker not in self._market_states:
            self._market_states[ticker] = SLateMarketState()

        slate_state = self._market_states[ticker]

        # Already signaled this market
        if slate_state.signaled:
            return None

        # Track this trade
        trade_value = trade.count * trade.yes_price
        slate_state.trades.append(SLateTrade(
            timestamp=trade.timestamp,
            value_cents=trade_value,
            taker_side=trade.taker_side,
            yes_price=trade.yes_price
        ))

        n_trades = len(slate_state.trades)

        # Need minimum trades
        if n_trades < self.min_trades:
            return None

        # Split into early (first 75%) and late (last 25%)
        cutoff = int(n_trades * 0.75)
        early_trades = slate_state.trades[:cutoff]
        late_trades = slate_state.trades[cutoff:]

        if len(late_trades) < 3:
            return None

        # Calculate large trade ratios
        early_large = [t for t in early_trades if t.value_cents > self.large_threshold_cents]
        late_large = [t for t in late_trades if t.value_cents > self.large_threshold_cents]

        early_large_ratio = len(early_large) / len(early_trades) if early_trades else 0
        late_large_ratio = len(late_large) / len(late_trades) if late_trades else 0

        # Check if late has significantly more large trades
        if early_large_ratio > 0:
            ratio_multiple = late_large_ratio / early_large_ratio
        else:
            ratio_multiple = late_large_ratio * 10 if late_large_ratio > 0 else 0

        if ratio_multiple < self.ratio_threshold:
            return None

        if late_large_ratio < self.late_large_min_ratio:
            return None

        # Need at least 2 late large trades
        if len(late_large) < 2:
            return None

        # Determine direction of late large money
        late_large_yes = sum(1 for t in late_large if t.taker_side == 'yes')
        late_yes_ratio = late_large_yes / len(late_large)

        # We want NO direction (YES ratio < threshold)
        if late_yes_ratio >= self.late_direction_threshold:
            return None

        # Signal fires! Mark as signaled
        slate_state.signaled = True

        # Calculate entry price (NO price at this moment)
        current_yes_price = trade.yes_price
        no_price = 100 - current_yes_price

        return SignalEntry(
            market_ticker=ticker,
            signal_time=trade.timestamp,
            entry_price_cents=no_price,
            side='no',
            signal_strength=self._compute_strength(ratio_multiple, late_large_ratio, late_yes_ratio),
            metadata={
                'n_trades': n_trades,
                'early_large_ratio': early_large_ratio,
                'late_large_ratio': late_large_ratio,
                'ratio_multiple': ratio_multiple,
                'late_yes_ratio': late_yes_ratio,
                'late_large_count': len(late_large),
                'no_price': no_price
            }
        )

    def _compute_strength(
        self,
        ratio_multiple: float,
        late_large_ratio: float,
        late_yes_ratio: float
    ) -> float:
        """
        Compute signal strength (0-1).

        Higher ratio multiple, higher late large ratio, and lower YES ratio = stronger.
        """
        # Ratio multiple: 2-5 maps to 0-0.4
        ratio_score = min((ratio_multiple - 2.0) / 3.0, 1.0) * 0.4

        # Late large ratio: 0.2-0.5 maps to 0-0.3
        large_score = min((late_large_ratio - 0.2) / 0.3, 1.0) * 0.3

        # Late YES ratio: lower is better (0.4-0.0 maps to 0-0.3)
        direction_score = min((0.4 - late_yes_ratio) / 0.4, 1.0) * 0.3

        return max(0.0, min(1.0, ratio_score + large_score + direction_score))

    def get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        return {
            'min_trades': self.min_trades,
            'large_threshold_cents': self.large_threshold_cents,
            'ratio_threshold': self.ratio_threshold,
            'late_direction_threshold': self.late_direction_threshold,
            'late_large_min_ratio': self.late_large_min_ratio
        }

    def reset(self) -> None:
        """Reset internal state for new backtest."""
        self._market_states.clear()


def create_slate_strategy(
    min_trades: int = 50,
    large_threshold_dollars: int = 50,
    ratio_threshold: float = 2.0,
    late_direction_threshold: float = 0.4,
    late_large_min_ratio: float = 0.2
) -> SLateTimingStrategy:
    """
    Factory function to create S-LATE strategy with custom parameters.

    Args:
        min_trades: Minimum trades before checking signal
        large_threshold_dollars: Dollar threshold for "large" trade
        ratio_threshold: Late/early large ratio threshold
        late_direction_threshold: YES ratio below this = NO direction
        late_large_min_ratio: Minimum large trade ratio in late window

    Returns:
        Configured SLateTimingStrategy
    """
    return SLateTimingStrategy(
        min_trades=min_trades,
        large_threshold_cents=large_threshold_dollars * 100,
        ratio_threshold=ratio_threshold,
        late_direction_threshold=late_direction_threshold,
        late_large_min_ratio=late_large_min_ratio
    )
