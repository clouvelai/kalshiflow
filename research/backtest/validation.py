"""
Validation Framework for Point-in-Time Backtesting.

Purpose:
    Provides comprehensive validation of backtest results to ensure
    strategy edge is real and not an artifact of data distribution.

Key Responsibilities:
    1. **Bucket-Matched Validation** - Controls for entry price distribution
       to prevent strategies that trade cheap contracts from appearing
       to have edge when they're just reflecting the underlying probabilities.

    2. **Statistical Significance** - Binomial test to ensure observed win
       rate is statistically different from expected win rate.

    3. **Concentration Check** - Ensures edge isn't driven by a single
       market (which could be luck or overfitting).

Architecture Position:
    Used after running PointInTimeBacktester to validate results:

        results = backtester.run(trades, settlements)
        report = validate_results(results, settlements)

        if report.passes_validation:
            print("Strategy has real edge!")
        else:
            print(f"Failed: {report.failure_reasons}")

Design Principles:
    - Conservative: Better to reject good strategies than accept bad ones
    - Transparent: All metrics and failure reasons are explicitly reported
    - Bucket-matched: Entry price distribution must be controlled
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from .engine import BacktestResults
    from .state import SignalEntry


@dataclass
class BucketStats:
    """
    Statistics for a price bucket.

    For NO bets at a given entry price:
    - Expected win rate = entry_price / 100
    - Edge = actual_win_rate - expected_win_rate

    Example:
        If entry price is 35 cents for NO:
        - Expected win rate = 35% (market implied probability)
        - If actual win rate is 45%, edge = +10%
    """
    bucket_range: Tuple[int, int]  # (min_price, max_price)
    n_signals: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    expected_win_rate: float = 0.0  # Based on entry price
    edge: float = 0.0

    def compute(self) -> None:
        """Compute win rate and edge statistics."""
        if self.n_signals > 0:
            self.win_rate = self.n_wins / self.n_signals
            # Expected win rate for NO bets = entry_price / 100
            # (since we're betting NO, we win if market settles NO)
            mid_price = (self.bucket_range[0] + self.bucket_range[1]) / 2
            self.expected_win_rate = mid_price / 100
            self.edge = self.win_rate - self.expected_win_rate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'bucket_range': self.bucket_range,
            'n_signals': self.n_signals,
            'n_wins': self.n_wins,
            'win_rate': self.win_rate,
            'expected_win_rate': self.expected_win_rate,
            'edge': self.edge
        }


@dataclass
class ValidationReport:
    """
    Comprehensive validation of backtest results.

    A strategy passes validation if:
    1. bucket_matched_edge > 0 (edge persists after controlling for price distribution)
    2. is_significant = True (p_value < significance_level)
    3. concentration_ok = True (no single market dominates)
    """

    # Input data
    strategy_name: str
    n_signals: int
    win_rate: float
    raw_edge: float
    total_pnl_cents: int

    # Bucket analysis
    bucket_stats: Dict[str, BucketStats] = field(default_factory=dict)
    bucket_matched_edge: float = 0.0

    # Statistical tests
    p_value: float = 1.0
    is_significant: bool = False

    # Concentration check
    max_market_concentration: float = 0.0
    max_market_ticker: str = ""
    concentration_ok: bool = True

    # Summary
    passes_validation: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'strategy_name': self.strategy_name,
            'n_signals': self.n_signals,
            'win_rate': self.win_rate,
            'raw_edge': self.raw_edge,
            'total_pnl_cents': self.total_pnl_cents,
            'bucket_matched_edge': self.bucket_matched_edge,
            'bucket_stats': {k: v.to_dict() for k, v in self.bucket_stats.items()},
            'p_value': self.p_value,
            'is_significant': self.is_significant,
            'max_market_concentration': self.max_market_concentration,
            'max_market_ticker': self.max_market_ticker,
            'concentration_ok': self.concentration_ok,
            'passes_validation': self.passes_validation,
            'failure_reasons': self.failure_reasons
        }

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"=== Validation Report: {self.strategy_name} ===",
            f"",
            f"Raw Results:",
            f"  Signals: {self.n_signals}",
            f"  Win Rate: {self.win_rate:.1%}",
            f"  Raw Edge: {self.raw_edge:+.1%}",
            f"  Total PnL: ${self.total_pnl_cents/100:.2f}",
            f"",
            f"Bucket-Matched Analysis:",
            f"  Bucket-Matched Edge: {self.bucket_matched_edge:+.1%}",
            f"",
            f"Statistical Test:",
            f"  P-Value: {self.p_value:.4f}",
            f"  Significant: {'Yes' if self.is_significant else 'No'}",
            f"",
            f"Concentration:",
            f"  Max Market: {self.max_market_ticker} ({self.max_market_concentration:.1%})",
            f"  Concentration OK: {'Yes' if self.concentration_ok else 'No'}",
            f"",
        ]

        if self.passes_validation:
            lines.append("RESULT: PASSES VALIDATION")
        else:
            lines.append("RESULT: FAILS VALIDATION")
            for reason in self.failure_reasons:
                lines.append(f"  - {reason}")

        return "\n".join(lines)


def bucket_matched_validation(
    entry_prices: List[int],
    wins: List[bool],
    side: str = 'no',
    bucket_size: int = 10
) -> Tuple[float, Dict[str, BucketStats]]:
    """
    Compute bucket-matched edge to control for entry price distribution.

    The Problem:
        A strategy that primarily trades cheap NO contracts (e.g., 30-35 cents)
        will naturally have a lower win rate than one trading expensive NO
        contracts (e.g., 65-70 cents). Raw edge doesn't account for this.

    The Solution:
        Group trades by entry price bucket and compute edge within each bucket.
        Expected win rate for NO bets = entry_price / 100.
        Edge = actual_win_rate - expected_win_rate.
        Final bucket-matched edge = weighted average across buckets.

    Args:
        entry_prices: List of entry prices in cents
        wins: List of whether each trade won (True/False)
        side: 'no' or 'yes' (determines expected win rate calculation)
        bucket_size: Size of price buckets (default 10 = 30-39, 40-49, etc.)

    Returns:
        Tuple of (bucket_matched_edge, dict of bucket_key -> BucketStats)

    Example:
        >>> prices = [35, 37, 42, 45, 68, 72]
        >>> wins = [True, True, True, False, True, True]
        >>> edge, stats = bucket_matched_validation(prices, wins)
        >>> print(f"Bucket-matched edge: {edge:.1%}")
    """
    if len(entry_prices) != len(wins):
        raise ValueError(f"entry_prices ({len(entry_prices)}) and wins ({len(wins)}) must have same length")

    # Group by bucket
    buckets: Dict[str, List[Tuple[int, bool]]] = {}

    for price, won in zip(entry_prices, wins):
        # Bucket: 30-39, 40-49, etc.
        bucket_min = (price // bucket_size) * bucket_size
        bucket_max = bucket_min + bucket_size - 1
        bucket_key = f"{bucket_min}-{bucket_max}"

        if bucket_key not in buckets:
            buckets[bucket_key] = []
        buckets[bucket_key].append((price, won))

    # Compute stats per bucket
    bucket_stats: Dict[str, BucketStats] = {}
    total_weighted_edge = 0.0
    total_weight = 0

    for bucket_key, data in sorted(buckets.items()):
        parts = bucket_key.split('-')
        bucket_range = (int(parts[0]), int(parts[1]))

        stats = BucketStats(bucket_range=bucket_range)
        stats.n_signals = len(data)
        stats.n_wins = sum(1 for _, won in data if won)
        stats.compute()

        bucket_stats[bucket_key] = stats

        # Weight by number of signals in bucket
        total_weighted_edge += stats.edge * stats.n_signals
        total_weight += stats.n_signals

    bucket_matched_edge = total_weighted_edge / total_weight if total_weight > 0 else 0.0

    return bucket_matched_edge, bucket_stats


def _normal_cdf(x: float) -> float:
    """Standard normal CDF using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def binomial_test(n_wins: int, n_trials: int, expected_prob: float) -> float:
    """
    Simple binomial test p-value using normal approximation.

    Tests whether observed win rate is significantly different from expected.
    Uses two-tailed test (detects both better AND worse than expected).

    Args:
        n_wins: Number of wins
        n_trials: Total number of trials
        expected_prob: Expected probability of winning (under null hypothesis)

    Returns:
        Two-tailed p-value. Lower = more significant.

    Note:
        For n_trials < 30, returns 1.0 (conservative estimate).
        In production, would use exact binomial test.
    """
    if n_trials == 0:
        return 1.0

    observed_prob = n_wins / n_trials

    # Use normal approximation for large n
    if n_trials >= 30:
        std_error = math.sqrt(expected_prob * (1 - expected_prob) / n_trials)
        if std_error == 0:
            return 1.0
        z_score = (observed_prob - expected_prob) / std_error
        # Two-tailed p-value from z-score
        p_value = 2 * (1 - _normal_cdf(abs(z_score)))
        return p_value

    # For small n, return conservative estimate
    # (would use scipy.stats.binom_test in production)
    return 1.0


def compute_concentration(
    signals: List['SignalEntry']
) -> Tuple[float, str, Dict[str, int]]:
    """
    Compute market concentration statistics.

    Args:
        signals: List of SignalEntry objects

    Returns:
        Tuple of (max_concentration, max_market_ticker, market_counts)
    """
    market_counts: Dict[str, int] = {}
    for signal in signals:
        ticker = signal.market_ticker
        market_counts[ticker] = market_counts.get(ticker, 0) + 1

    if not market_counts:
        return 0.0, "", {}

    max_ticker = max(market_counts, key=market_counts.get)
    max_concentration = market_counts[max_ticker] / len(signals)

    return max_concentration, max_ticker, market_counts


def validate_results(
    results: 'BacktestResults',
    settlements: Dict[str, str],
    significance_level: float = 0.05,
    max_concentration: float = 0.30
) -> ValidationReport:
    """
    Comprehensive validation of backtest results.

    Validation Checks:
        1. Bucket-matched edge > 0 (controls for entry price distribution)
        2. Statistical significance (p-value < significance_level)
        3. Market concentration <= max_concentration (prevents single-market dominance)

    Args:
        results: BacktestResults from PointInTimeBacktester
        settlements: Dict mapping market_ticker -> 'yes' or 'no'
        significance_level: P-value threshold (default 0.05)
        max_concentration: Max fraction of signals from single market (default 0.30)

    Returns:
        ValidationReport with all metrics and pass/fail determination

    Example:
        >>> results = backtester.run(trades, settlements)
        >>> report = validate_results(results, settlements)
        >>> print(report.summary())
    """
    report = ValidationReport(
        strategy_name=results.strategy_name,
        n_signals=results.n_signals,
        win_rate=results.win_rate,
        raw_edge=results.raw_edge,
        total_pnl_cents=results.total_pnl_cents
    )

    if results.n_signals == 0:
        report.failure_reasons.append("No signals generated")
        return report

    # Build wins list for bucket analysis
    wins: List[bool] = []
    entry_prices: List[int] = []

    for signal in results.signals:
        settlement = settlements.get(signal.market_ticker)
        if settlement:
            # For NO bets: win if market settled NO
            # For YES bets: win if market settled YES
            won = (signal.side == settlement)
            wins.append(won)
            entry_prices.append(signal.entry_price_cents)

    if len(wins) == 0:
        report.failure_reasons.append("No signals with settlement data")
        return report

    # 1. Bucket-matched validation
    bucket_edge, bucket_stats = bucket_matched_validation(
        entry_prices, wins, side='no', bucket_size=10
    )
    report.bucket_matched_edge = bucket_edge
    report.bucket_stats = bucket_stats

    # 2. Statistical significance
    # Expected win rate based on average entry price
    avg_entry = sum(entry_prices) / len(entry_prices)
    expected_win_rate = avg_entry / 100  # For NO bets

    report.p_value = binomial_test(
        sum(wins), len(wins), expected_win_rate
    )
    report.is_significant = report.p_value < significance_level

    # 3. Concentration check
    concentration, max_ticker, _ = compute_concentration(results.signals)
    report.max_market_concentration = concentration
    report.max_market_ticker = max_ticker
    report.concentration_ok = concentration <= max_concentration

    # Determine overall pass/fail
    if report.bucket_matched_edge <= 0:
        report.failure_reasons.append(
            f"Bucket-matched edge <= 0: {report.bucket_matched_edge:.2%}"
        )

    if not report.is_significant:
        report.failure_reasons.append(
            f"Not statistically significant: p={report.p_value:.4f} (threshold: {significance_level})"
        )

    if not report.concentration_ok:
        report.failure_reasons.append(
            f"Market concentration too high: {report.max_market_ticker} has {report.max_market_concentration:.1%} of signals (max: {max_concentration:.0%})"
        )

    report.passes_validation = len(report.failure_reasons) == 0

    return report


def quick_validate(
    entry_prices: List[int],
    wins: List[bool],
    strategy_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Quick validation for ad-hoc analysis without full BacktestResults.

    Args:
        entry_prices: List of entry prices in cents
        wins: List of win/loss results
        strategy_name: Name for reporting

    Returns:
        Dictionary with key metrics
    """
    if len(entry_prices) != len(wins):
        raise ValueError("entry_prices and wins must have same length")

    if len(entry_prices) == 0:
        return {
            'strategy_name': strategy_name,
            'n_signals': 0,
            'error': 'No signals'
        }

    bucket_edge, bucket_stats = bucket_matched_validation(entry_prices, wins)

    avg_entry = sum(entry_prices) / len(entry_prices)
    expected_win_rate = avg_entry / 100
    actual_win_rate = sum(wins) / len(wins)

    p_value = binomial_test(sum(wins), len(wins), expected_win_rate)

    return {
        'strategy_name': strategy_name,
        'n_signals': len(entry_prices),
        'win_rate': actual_win_rate,
        'expected_win_rate': expected_win_rate,
        'raw_edge': actual_win_rate - expected_win_rate,
        'bucket_matched_edge': bucket_edge,
        'p_value': p_value,
        'is_significant': p_value < 0.05,
        'bucket_stats': {k: v.to_dict() for k, v in bucket_stats.items()}
    }
