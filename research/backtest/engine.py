"""
Point-in-Time Backtesting Engine.

This is the core engine that processes trades chronologically and fires
signals without look-ahead bias.

Key Design Principles:
    1. CHRONOLOGICAL PROCESSING: Trades are sorted by timestamp and processed
       in order. The engine never looks at future trades.

    2. STATE BEFORE SIGNAL: Market state is updated BEFORE the strategy
       evaluates the trade. This reflects reality - the trade has happened,
       you're deciding what to do next.

    3. FIRST SIGNAL WINS: By default, only one signal fires per market.
       This prevents strategies from "learning" from later trades.

    4. ENTRY PRICE AT SIGNAL TIME: The entry price recorded is the price
       at the moment the signal fired, not the final price.

    5. SETTLEMENT IS SEPARATE: P&L calculation uses settlement data that
       is provided separately, not derived from trades.

Usage:
    from research.backtest import PointInTimeBacktester, Trade
    from research.backtest.strategies import MyStrategy

    # Create strategy and backtester
    strategy = MyStrategy(price_threshold=85)
    backtester = PointInTimeBacktester(strategy)

    # Run backtest
    results = backtester.run(trades, settlements)

    # Analyze results
    print(f"Signals: {results.n_signals}")
    print(f"Win rate: {results.win_rate:.1%}")
    print(f"Edge: {results.raw_edge:.1%}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Optional, Callable
from datetime import datetime
import logging
import sys
import os

# Handle both package and direct imports
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

try:
    from .state import Trade, MarketState, SignalEntry
    from .strategies.protocol import Strategy
except ImportError:
    from state import Trade, MarketState, SignalEntry
    from strategies.protocol import Strategy

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """
    Results from a backtest run.

    This class holds all signals generated and computed metrics.
    Metrics are computed AFTER all signals are collected, using
    settlement data to determine wins/losses.

    Attributes:
        strategy_name: Name of the strategy tested
        parameters: Parameter dict from strategy.get_parameters()

        signals: List of all SignalEntry objects generated

        n_signals: Total signals fired
        n_wins: Signals that were correct (side == settlement)
        n_losses: Signals that were wrong
        n_unresolved: Markets without settlement data

        win_rate: n_wins / (n_wins + n_losses)
        raw_edge: Average P&L per signal as percentage
        total_pnl_cents: Sum of all P&L in cents

        entry_prices: List of entry prices for bucket analysis

    Methods:
        compute_metrics: Calculate metrics using settlement data
        get_bucket_stats: Analyze performance by entry price bucket
    """
    strategy_name: str
    parameters: Dict

    # Signals
    signals: List[SignalEntry] = field(default_factory=list)

    # Core metrics (computed from signals + settlements)
    n_signals: int = 0
    n_wins: int = 0
    n_losses: int = 0
    n_unresolved: int = 0

    win_rate: float = 0.0
    raw_edge: float = 0.0
    total_pnl_cents: int = 0

    # Entry price distribution for bucket analysis
    entry_prices: List[int] = field(default_factory=list)

    # Detailed per-signal results (populated by compute_metrics)
    signal_results: List[Dict] = field(default_factory=list)

    def compute_metrics(self, settlements: Dict[str, str]) -> None:
        """
        Compute metrics after signals collected.

        This uses settlement data (market_ticker -> 'yes' or 'no') to
        determine which signals won and calculate P&L.

        Args:
            settlements: Dict mapping market_ticker to 'yes' or 'no'

        P&L Calculation:
            For YES signals:
                Win (settle YES): profit = 100 - entry_price
                Lose (settle NO): profit = -entry_price

            For NO signals:
                Win (settle NO): profit = 100 - (100 - entry_price) = entry_price
                Lose (settle YES): profit = -(100 - entry_price)

            This simplifies to:
                YES signal win: +payout (100 - entry)
                YES signal loss: -entry
                NO signal win: +entry (we paid 100-entry for NO, won 100)
                NO signal loss: -(100 - entry)
        """
        self.n_signals = len(self.signals)
        self.signal_results = []

        for signal in self.signals:
            result = settlements.get(signal.market_ticker)

            signal_result = {
                'market_ticker': signal.market_ticker,
                'side': signal.side,
                'entry_price': signal.entry_price_cents,
                'settlement': result,
                'won': None,
                'pnl_cents': None
            }

            if result is None:
                self.n_unresolved += 1
                self.signal_results.append(signal_result)
                continue

            # Determine if we won
            won = (signal.side == result)
            signal_result['won'] = won

            # Calculate P&L
            if signal.side == 'yes':
                if won:  # YES signal, settled YES
                    pnl = 100 - signal.entry_price_cents
                else:  # YES signal, settled NO
                    pnl = -signal.entry_price_cents
            else:  # NO signal
                if won:  # NO signal, settled NO
                    pnl = signal.entry_price_cents  # We paid (100-entry), won 100
                else:  # NO signal, settled YES
                    pnl = -(100 - signal.entry_price_cents)

            signal_result['pnl_cents'] = pnl
            self.signal_results.append(signal_result)

            if won:
                self.n_wins += 1
            else:
                self.n_losses += 1

            self.total_pnl_cents += pnl
            self.entry_prices.append(signal.entry_price_cents)

        # Compute aggregate metrics
        resolved = self.n_wins + self.n_losses
        if resolved > 0:
            self.win_rate = self.n_wins / resolved
            # Edge = average P&L per contract as percentage
            self.raw_edge = self.total_pnl_cents / resolved / 100

    def get_bucket_stats(
        self,
        buckets: List[tuple] = None
    ) -> Dict[str, Dict]:
        """
        Analyze performance by entry price bucket.

        Args:
            buckets: List of (min, max) tuples defining price buckets.
                    Default: [(80, 85), (85, 90), (90, 95), (95, 100)]

        Returns:
            Dict mapping bucket names to stats:
            {
                '80-85': {
                    'n': 10,
                    'wins': 7,
                    'win_rate': 0.7,
                    'edge': 0.05,
                    'pnl': 50
                },
                ...
            }
        """
        if buckets is None:
            buckets = [(80, 85), (85, 90), (90, 95), (95, 100)]

        bucket_stats = {}

        for min_price, max_price in buckets:
            bucket_name = f"{min_price}-{max_price}"
            bucket_signals = [
                r for r in self.signal_results
                if r['settlement'] is not None
                and min_price <= r['entry_price'] < max_price
            ]

            n = len(bucket_signals)
            wins = sum(1 for r in bucket_signals if r['won'])
            pnl = sum(r['pnl_cents'] for r in bucket_signals)

            bucket_stats[bucket_name] = {
                'n': n,
                'wins': wins,
                'win_rate': wins / n if n > 0 else 0.0,
                'edge': pnl / n / 100 if n > 0 else 0.0,
                'pnl_cents': pnl
            }

        return bucket_stats

    def summary(self) -> str:
        """Return a formatted summary of results."""
        lines = [
            f"Strategy: {self.strategy_name}",
            f"Parameters: {self.parameters}",
            "",
            f"Signals: {self.n_signals}",
            f"  Resolved: {self.n_wins + self.n_losses}",
            f"  Unresolved: {self.n_unresolved}",
            "",
            f"Performance:",
            f"  Win Rate: {self.win_rate:.1%}",
            f"  Raw Edge: {self.raw_edge:+.2%}",
            f"  Total P&L: {self.total_pnl_cents:+,} cents"
        ]
        return "\n".join(lines)


class PointInTimeBacktester:
    """
    Replays trades chronologically, fires signals at first trigger.

    CRITICAL: No look-ahead bias - signals fire at the moment conditions
    are first met, using ONLY information available at that time.

    The engine processes trades in strict chronological order:
    1. Get next trade (sorted by timestamp)
    2. Update market state with this trade
    3. Ask strategy if signal fires (with updated state)
    4. If signal fires and market not already signaled, record it
    5. Repeat until all trades processed

    After processing, use settlement data to compute P&L.

    Attributes:
        strategy: The trading strategy being tested
        one_signal_per_market: If True, only one signal per market
        market_states: Dict of market_ticker -> MarketState
        signaled_markets: Set of markets that have already signaled
        signals: List of SignalEntry objects generated

    Usage:
        backtester = PointInTimeBacktester(strategy)
        results = backtester.run(trades, settlements)
    """

    def __init__(self, strategy: Strategy, one_signal_per_market: bool = True):
        """
        Initialize the backtester.

        Args:
            strategy: Strategy implementing the Protocol
            one_signal_per_market: If True, only fire one signal per market
                                  (typical for position entry strategies).
                                  Set False for strategies that can re-enter.
        """
        self.strategy = strategy
        self.one_signal_per_market = one_signal_per_market
        self.market_states: Dict[str, MarketState] = {}
        self.signaled_markets: set = set()
        self.signals: List[SignalEntry] = []

    def reset(self) -> None:
        """Reset for a new backtest run."""
        self.market_states.clear()
        self.signaled_markets.clear()
        self.signals.clear()
        self.strategy.reset()

    def _get_or_create_state(self, market_ticker: str) -> MarketState:
        """Get or create market state."""
        if market_ticker not in self.market_states:
            self.market_states[market_ticker] = MarketState(market_ticker=market_ticker)
        return self.market_states[market_ticker]

    def run(
        self,
        trades: Iterable[Trade],
        settlements: Dict[str, str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> BacktestResults:
        """
        Process all trades chronologically.

        This is the main entry point for running a backtest.

        Args:
            trades: Iterable of Trade objects (will be sorted by timestamp)
            settlements: Dict mapping market_ticker -> 'yes' or 'no'
            progress_callback: Optional callback(n_processed, n_total) for
                             progress reporting (called every 10k trades)

        Returns:
            BacktestResults with signals and computed metrics

        Example:
            # Load your trades and settlements
            trades = load_trades_from_csv('trades.csv')
            settlements = load_settlements('settlements.json')

            # Run backtest
            backtester = PointInTimeBacktester(MyStrategy())
            results = backtester.run(trades, settlements)

            # Check results
            print(results.summary())
        """
        self.reset()

        # Convert to list and sort chronologically
        # This ensures we process trades in the order they happened
        sorted_trades = sorted(trades, key=lambda t: t.timestamp)
        total_trades = len(sorted_trades)

        logger.info(f"Starting backtest with {total_trades:,} trades")

        for i, trade in enumerate(sorted_trades):
            # Get/create market state
            state = self._get_or_create_state(trade.market_ticker)

            # Update state with this trade FIRST
            # The strategy sees the world AFTER this trade happened
            state.update(trade)

            # Check if we should skip this market (already signaled)
            if self.one_signal_per_market and trade.market_ticker in self.signaled_markets:
                continue

            # Ask strategy if signal fires
            signal = self.strategy.on_trade(trade, state)

            if signal is not None:
                self.signals.append(signal)
                self.signaled_markets.add(trade.market_ticker)
                logger.debug(
                    f"Signal fired: {signal.market_ticker} "
                    f"side={signal.side} @ {signal.entry_price_cents}c"
                )

            # Progress callback (every 10k trades)
            if progress_callback and i % 10000 == 0:
                progress_callback(i, total_trades)

        if progress_callback:
            progress_callback(total_trades, total_trades)

        logger.info(f"Backtest complete: {len(self.signals)} signals from {len(self.signaled_markets)} markets")

        # Build results
        results = BacktestResults(
            strategy_name=self.strategy.name,
            parameters=self.strategy.get_parameters(),
            signals=self.signals.copy()
        )
        results.compute_metrics(settlements)

        return results

    def get_market_state(self, market_ticker: str) -> Optional[MarketState]:
        """
        Get the final state of a specific market.

        Useful for debugging or analysis after backtest completes.

        Args:
            market_ticker: The market to look up

        Returns:
            MarketState if the market was seen, None otherwise
        """
        return self.market_states.get(market_ticker)

    def get_all_states(self) -> Dict[str, MarketState]:
        """
        Get all market states after backtest.

        Returns:
            Dict mapping market_ticker to final MarketState
        """
        return self.market_states.copy()


def run_backtest(
    strategy: Strategy,
    trades: Iterable[Trade],
    settlements: Dict[str, str],
    one_signal_per_market: bool = True,
    progress: bool = False
) -> BacktestResults:
    """
    Convenience function to run a backtest.

    This is a simple wrapper around PointInTimeBacktester for
    quick backtests without managing the backtester instance.

    Args:
        strategy: Strategy to test
        trades: Trade data
        settlements: Settlement data
        one_signal_per_market: Only one signal per market
        progress: Print progress updates

    Returns:
        BacktestResults

    Example:
        results = run_backtest(
            MyStrategy(threshold=85),
            trades,
            settlements,
            progress=True
        )
        print(results.summary())
    """
    def progress_callback(n, total):
        if progress:
            pct = n / total * 100 if total > 0 else 0
            print(f"\rProcessing: {n:,}/{total:,} ({pct:.1f}%)", end="", flush=True)

    backtester = PointInTimeBacktester(strategy, one_signal_per_market)
    results = backtester.run(
        trades,
        settlements,
        progress_callback=progress_callback if progress else None
    )

    if progress:
        print()  # Newline after progress

    return results
