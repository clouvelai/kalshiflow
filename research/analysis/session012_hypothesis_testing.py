"""
Session 012: Exhaustive Hypothesis Testing
Objective: Find at least ONE more validated strategy beyond S007, S008, S009

Methodology:
1. CORRECT breakeven: trade_price / 100.0 (not inverted for NO)
2. Price proxy check: Compare to baseline at SAME price levels
3. Validation thresholds: N >= 100, p < 0.01, concentration < 30%, improvement > 1%

Hypotheses to test:
- H089: Interval Trading Pattern
- H091: Size Ratio Consistency (martingale)
- H095: Momentum Ignition (already tested, revisit)
- H096: Quote Stuffing Aftermath
- H099: Spread-Sensitive Bot
- H051: Trade Size Distribution Skew
- H054: Consecutive Same-Side Trade Runs
- H056: Contrarian at Extreme Prices Only
- H057: First Trade Direction Persistence
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
REPORT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/'

def load_data():
    """Load and prepare trade data"""
    print("Loading trade data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Calculate NO price for NO trades (trade_price IS the NO price for NO trades)
    # For YES trades, NO price = 100 - yes_price
    df['effective_no_price'] = np.where(
        df['taker_side'] == 'no',
        df['trade_price'],  # NO trades: trade_price is what they paid for NO
        100 - df['trade_price']  # YES trades: NO price is 100 - YES price
    )

    return df

def calculate_market_stats(df):
    """Calculate per-market statistics"""
    market_stats = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'id': 'count',
        'count': 'sum',
        'cost_dollars': 'sum',
        'trade_price': 'mean',
        'leverage_ratio': ['mean', 'std'],
        'taker_side': lambda x: (x == 'yes').mean(),  # YES ratio
        'timestamp': ['min', 'max'],
        'effective_no_price': 'mean'
    }).reset_index()

    # Flatten column names
    market_stats.columns = [
        'market_ticker', 'market_result', 'trade_count', 'contract_count',
        'total_dollars', 'avg_trade_price', 'avg_leverage', 'leverage_std',
        'yes_ratio', 'first_trade_ts', 'last_trade_ts', 'avg_no_price'
    ]

    # Calculate market duration
    market_stats['duration_seconds'] = (market_stats['last_trade_ts'] - market_stats['first_trade_ts']) / 1000

    return market_stats

def calculate_baseline_edge_by_no_price(df, bucket_size=5):
    """Calculate baseline win rate at each NO price level"""
    # Group by NO price bucket
    df_copy = df.copy()
    df_copy['no_price_bucket'] = (df_copy['effective_no_price'] // bucket_size) * bucket_size

    # Calculate baseline by market (one entry per market)
    market_baseline = df_copy.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price_bucket': 'first',  # Use first trade's price bucket
        'effective_no_price': 'mean'
    }).reset_index()

    # Group by price bucket
    baseline = market_baseline.groupby('no_price_bucket').agg({
        'market_result': lambda x: (x == 'no').mean(),  # NO win rate
        'market_ticker': 'count'
    }).reset_index()
    baseline.columns = ['no_price_bucket', 'baseline_no_win_rate', 'market_count']

    return baseline

def validate_strategy(signal_markets, all_markets, strategy_name, n_tests=20):
    """
    Validate a strategy with proper price proxy check

    Returns dict with validation results
    """
    print(f"\n{'='*60}")
    print(f"Validating: {strategy_name}")
    print(f"{'='*60}")

    # Basic stats
    n_signal = len(signal_markets)
    if n_signal < 100:
        print(f"REJECTED: Only {n_signal} markets (< 100 threshold)")
        return {'status': 'rejected', 'reason': 'insufficient_markets', 'n_markets': n_signal}

    # Calculate NO win rate for signal markets
    signal_no_wins = (signal_markets['market_result'] == 'no').sum()
    signal_no_win_rate = signal_no_wins / n_signal

    # Calculate breakeven (avg NO price)
    avg_no_price = signal_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100.0

    # Calculate edge
    edge = signal_no_win_rate - breakeven

    print(f"\nSignal Stats:")
    print(f"  Markets: {n_signal}")
    print(f"  NO Win Rate: {signal_no_win_rate:.1%}")
    print(f"  Avg NO Price: {avg_no_price:.1f}c")
    print(f"  Breakeven: {breakeven:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # Statistical significance (Bonferroni-corrected)
    # Use binomtest (newer scipy API) or calculate manually
    try:
        from scipy.stats import binomtest
        result = binomtest(signal_no_wins, n_signal, breakeven, alternative='greater')
        p_value = result.pvalue
    except (ImportError, AttributeError):
        # Manual calculation using normal approximation for large samples
        expected = n_signal * breakeven
        std = np.sqrt(n_signal * breakeven * (1 - breakeven))
        z = (signal_no_wins - expected) / std if std > 0 else 0
        p_value = 1 - stats.norm.cdf(z)

    bonferroni_threshold = 0.01 / n_tests

    print(f"  P-value: {p_value:.2e}")
    print(f"  Bonferroni threshold: {bonferroni_threshold:.4f}")

    if p_value >= bonferroni_threshold:
        print(f"REJECTED: P-value {p_value:.4f} >= {bonferroni_threshold:.4f}")
        return {'status': 'rejected', 'reason': 'not_significant', 'p_value': p_value, 'edge': edge}

    # Concentration check
    if 'profit_contribution' in signal_markets.columns:
        max_concentration = signal_markets['profit_contribution'].max()
    else:
        # Calculate profit contribution
        signal_markets_copy = signal_markets.copy()
        signal_markets_copy['no_won'] = (signal_markets_copy['market_result'] == 'no').astype(int)
        signal_markets_copy['profit'] = np.where(
            signal_markets_copy['no_won'] == 1,
            (100 - signal_markets_copy['avg_no_price']) * signal_markets_copy['contract_count'] / 100,
            -signal_markets_copy['avg_no_price'] * signal_markets_copy['contract_count'] / 100
        )
        total_profit = signal_markets_copy[signal_markets_copy['profit'] > 0]['profit'].sum()
        if total_profit > 0:
            signal_markets_copy['profit_contribution'] = np.where(
                signal_markets_copy['profit'] > 0,
                signal_markets_copy['profit'] / total_profit,
                0
            )
            max_concentration = signal_markets_copy['profit_contribution'].max()
        else:
            max_concentration = 0

    print(f"  Max Concentration: {max_concentration*100:.1f}%")

    if max_concentration > 0.30:
        print(f"REJECTED: Concentration {max_concentration:.1%} > 30%")
        return {'status': 'rejected', 'reason': 'concentration', 'concentration': max_concentration, 'edge': edge}

    # CRITICAL: Price proxy check
    # Compare signal win rate to baseline at SAME price levels
    print("\n  Price Proxy Check:")

    # Bucket signal markets by NO price
    signal_copy = signal_markets.copy()
    signal_copy['no_price_bucket'] = (signal_copy['avg_no_price'] // 5) * 5

    # Calculate signal win rate by bucket
    signal_by_bucket = signal_copy.groupby('no_price_bucket').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    signal_by_bucket.columns = ['no_price_bucket', 'signal_win_rate', 'signal_count']

    # Calculate baseline win rate by bucket (from ALL markets)
    all_markets_copy = all_markets.copy()
    all_markets_copy['no_price_bucket'] = (all_markets_copy['avg_no_price'] // 5) * 5

    baseline_by_bucket = all_markets_copy.groupby('no_price_bucket').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    baseline_by_bucket.columns = ['no_price_bucket', 'baseline_win_rate', 'baseline_count']

    # Merge and calculate weighted improvement
    comparison = signal_by_bucket.merge(baseline_by_bucket, on='no_price_bucket', how='left')
    comparison['improvement'] = comparison['signal_win_rate'] - comparison['baseline_win_rate']
    comparison['weight'] = comparison['signal_count'] / comparison['signal_count'].sum()

    weighted_improvement = (comparison['improvement'] * comparison['weight']).sum()

    print(f"  Weighted Improvement over Baseline: {weighted_improvement*100:.2f}%")

    # Show bucket breakdown
    print("\n  Bucket Breakdown:")
    print(f"  {'Bucket':<10} {'Signal WR':<12} {'Base WR':<12} {'Improve':<10} {'N':<8}")
    for _, row in comparison.iterrows():
        if pd.notna(row['baseline_win_rate']):
            print(f"  {row['no_price_bucket']:.0f}-{row['no_price_bucket']+5:.0f}c    "
                  f"{row['signal_win_rate']:.1%}        "
                  f"{row['baseline_win_rate']:.1%}        "
                  f"{row['improvement']*100:+.1f}%      "
                  f"{row['signal_count']:.0f}")

    if weighted_improvement <= 0.01:
        print(f"\nREJECTED: Improvement {weighted_improvement*100:.2f}% <= 1% threshold")
        return {
            'status': 'rejected',
            'reason': 'price_proxy',
            'improvement': weighted_improvement,
            'edge': edge,
            'n_markets': n_signal
        }

    # Temporal stability check
    print("\n  Temporal Stability:")
    signal_copy = signal_markets.copy()
    signal_copy = signal_copy.sort_values('first_trade_ts')
    n_half = len(signal_copy) // 2

    first_half = signal_copy.iloc[:n_half]
    second_half = signal_copy.iloc[n_half:]

    first_wr = (first_half['market_result'] == 'no').mean()
    second_wr = (second_half['market_result'] == 'no').mean()
    first_be = first_half['avg_no_price'].mean() / 100
    second_be = second_half['avg_no_price'].mean() / 100

    first_edge = first_wr - first_be
    second_edge = second_wr - second_be

    print(f"  First half:  Edge = {first_edge*100:+.1f}% (WR={first_wr:.1%}, BE={first_be:.1%})")
    print(f"  Second half: Edge = {second_edge*100:+.1f}% (WR={second_wr:.1%}, BE={second_be:.1%})")

    temporal_stable = first_edge > 0 and second_edge > 0

    if not temporal_stable:
        print(f"\nWARNING: Temporal instability detected")

    print(f"\n{'*'*60}")
    print(f"VALIDATED: {strategy_name}")
    print(f"  Edge: {edge*100:.2f}%")
    print(f"  Improvement: {weighted_improvement*100:.2f}%")
    print(f"  Markets: {n_signal}")
    print(f"  P-value: {p_value:.2e}")
    print(f"{'*'*60}")

    return {
        'status': 'validated',
        'edge': edge,
        'improvement': weighted_improvement,
        'n_markets': n_signal,
        'win_rate': signal_no_win_rate,
        'breakeven': breakeven,
        'p_value': p_value,
        'concentration': max_concentration,
        'temporal_stable': temporal_stable,
        'first_half_edge': first_edge,
        'second_half_edge': second_edge
    }


# ============================================================================
# HYPOTHESIS TESTING FUNCTIONS
# ============================================================================

def test_h089_interval_trading(df, market_stats):
    """
    H089: Interval Trading Pattern
    Detect bots by looking for regular time gaps between trades
    """
    print("\n" + "="*80)
    print("TESTING H089: Interval Trading Pattern")
    print("="*80)

    # Calculate time gaps between trades for each market
    df_sorted = df.sort_values(['market_ticker', 'timestamp'])
    df_sorted['time_gap'] = df_sorted.groupby('market_ticker')['timestamp'].diff()

    # For each market, calculate the coefficient of variation of time gaps
    # Low CV = regular intervals = bot behavior
    gap_stats = df_sorted.groupby('market_ticker').agg({
        'time_gap': ['mean', 'std', 'count']
    }).reset_index()
    gap_stats.columns = ['market_ticker', 'gap_mean', 'gap_std', 'trade_count']

    # Filter markets with enough trades
    gap_stats = gap_stats[gap_stats['trade_count'] >= 5]

    # Calculate coefficient of variation (std/mean)
    gap_stats['gap_cv'] = gap_stats['gap_std'] / gap_stats['gap_mean']
    gap_stats['gap_cv'] = gap_stats['gap_cv'].fillna(1)  # Handle edge cases

    # Low CV (< 0.3) suggests regular interval trading
    gap_stats['is_interval_trader'] = gap_stats['gap_cv'] < 0.3

    # Merge with market stats
    interval_markets = market_stats.merge(
        gap_stats[gap_stats['is_interval_trader']][['market_ticker', 'gap_cv']],
        on='market_ticker'
    )

    print(f"Found {len(interval_markets)} markets with regular interval trading (CV < 0.3)")

    if len(interval_markets) >= 100:
        return validate_strategy(interval_markets, market_stats, "H089: Interval Trading Pattern")
    else:
        # Try less strict threshold
        gap_stats['is_interval_trader'] = gap_stats['gap_cv'] < 0.5
        interval_markets = market_stats.merge(
            gap_stats[gap_stats['is_interval_trader']][['market_ticker', 'gap_cv']],
            on='market_ticker'
        )
        print(f"With CV < 0.5: {len(interval_markets)} markets")

        if len(interval_markets) >= 100:
            return validate_strategy(interval_markets, market_stats, "H089: Interval Trading (CV < 0.5)")

    return {'status': 'rejected', 'reason': 'insufficient_markets', 'n_markets': len(interval_markets)}


def test_h091_size_ratio_consistency(df, market_stats):
    """
    H091: Size Ratio Consistency (Martingale Detection)
    Look for patterns where trade sizes follow geometric progressions (2x, 2x, 2x)
    """
    print("\n" + "="*80)
    print("TESTING H091: Size Ratio Consistency (Martingale)")
    print("="*80)

    # Calculate size ratios between consecutive trades
    df_sorted = df.sort_values(['market_ticker', 'timestamp'])
    df_sorted['prev_count'] = df_sorted.groupby('market_ticker')['count'].shift(1)
    df_sorted['size_ratio'] = df_sorted['count'] / df_sorted['prev_count']
    df_sorted['size_ratio'] = df_sorted['size_ratio'].replace([np.inf, -np.inf], np.nan)

    # For each market, check if ratios are consistent (e.g., always ~2x)
    def check_martingale(group):
        ratios = group['size_ratio'].dropna()
        if len(ratios) < 3:
            return False
        # Check if ratios are within 20% of each other
        ratio_std = ratios.std()
        ratio_mean = ratios.mean()
        if ratio_mean == 0:
            return False
        cv = ratio_std / ratio_mean
        return cv < 0.3 and ratio_mean >= 1.5  # Consistent doubling-like pattern

    martingale_markets = df_sorted.groupby('market_ticker').apply(check_martingale).reset_index()
    martingale_markets.columns = ['market_ticker', 'is_martingale']
    martingale_markets = martingale_markets[martingale_markets['is_martingale']]

    # Merge with market stats
    signal_markets = market_stats[market_stats['market_ticker'].isin(martingale_markets['market_ticker'])]

    print(f"Found {len(signal_markets)} markets with martingale-like patterns")

    if len(signal_markets) >= 100:
        return validate_strategy(signal_markets, market_stats, "H091: Martingale Pattern")

    return {'status': 'rejected', 'reason': 'insufficient_markets', 'n_markets': len(signal_markets)}


def test_h096_quote_stuffing_aftermath(df, market_stats):
    """
    H096: Quote Stuffing Aftermath
    Look for bursts of activity followed by quiet periods, then follow the next trade
    """
    print("\n" + "="*80)
    print("TESTING H096: Quote Stuffing Aftermath")
    print("="*80)

    # Calculate trades per second for each market
    df['second'] = df['timestamp'] // 1000

    trades_per_second = df.groupby(['market_ticker', 'second']).size().reset_index(name='trades_count')

    # Find burst seconds (3+ trades per second)
    burst_seconds = trades_per_second[trades_per_second['trades_count'] >= 3]

    # For each market with bursts, find the trade AFTER the burst
    burst_markets = burst_seconds['market_ticker'].unique()

    # Check if post-burst activity has predictive value
    # We'll look at markets that had bursts and see if NO wins more often
    signal_markets = market_stats[market_stats['market_ticker'].isin(burst_markets)]

    print(f"Found {len(signal_markets)} markets with trade bursts")

    if len(signal_markets) >= 100:
        return validate_strategy(signal_markets, market_stats, "H096: Quote Stuffing Aftermath")

    return {'status': 'rejected', 'reason': 'insufficient_markets', 'n_markets': len(signal_markets)}


def test_h051_trade_size_skew(df, market_stats):
    """
    H051: Trade Size Distribution Skew
    Markets where trade sizes are highly skewed may indicate retail vs institutional
    """
    print("\n" + "="*80)
    print("TESTING H051: Trade Size Distribution Skew")
    print("="*80)

    # Calculate skewness of trade sizes for each market
    def calc_skewness(group):
        if len(group) < 5:
            return np.nan
        return stats.skew(group['count'])

    skewness = df.groupby('market_ticker').apply(calc_skewness).reset_index()
    skewness.columns = ['market_ticker', 'size_skewness']
    skewness = skewness.dropna()

    # Highly positive skew = mostly small trades with occasional large ones (retail dominated)
    # Highly negative skew = mostly large trades (institutional)

    # Test positive skew (retail) - fade these
    high_skew_threshold = skewness['size_skewness'].quantile(0.75)
    high_skew_markets = skewness[skewness['size_skewness'] > high_skew_threshold]['market_ticker']

    signal_markets = market_stats[market_stats['market_ticker'].isin(high_skew_markets)]

    print(f"Found {len(signal_markets)} markets with high positive skew (retail dominated)")

    if len(signal_markets) >= 100:
        result = validate_strategy(signal_markets, market_stats, "H051: High Skew (Retail)")
        if result['status'] == 'validated':
            return result

    # Also test negative skew (institutional)
    low_skew_threshold = skewness['size_skewness'].quantile(0.25)
    low_skew_markets = skewness[skewness['size_skewness'] < low_skew_threshold]['market_ticker']

    signal_markets = market_stats[market_stats['market_ticker'].isin(low_skew_markets)]

    print(f"\nFound {len(signal_markets)} markets with negative skew (institutional)")

    if len(signal_markets) >= 100:
        return validate_strategy(signal_markets, market_stats, "H051: Negative Skew (Institutional)")

    return {'status': 'rejected', 'reason': 'insufficient_improvement', 'n_markets': len(signal_markets)}


def test_h054_consecutive_runs(df, market_stats):
    """
    H054: Consecutive Same-Side Trade Runs
    Markets with long runs of same-side trades may indicate information accumulation
    """
    print("\n" + "="*80)
    print("TESTING H054: Consecutive Same-Side Trade Runs")
    print("="*80)

    # Calculate max run length for each market
    df_sorted = df.sort_values(['market_ticker', 'timestamp'])

    def max_run_length(group):
        sides = group['taker_side'].values
        if len(sides) < 3:
            return 0

        max_run = 1
        current_run = 1
        for i in range(1, len(sides)):
            if sides[i] == sides[i-1]:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 1
        return max_run

    runs = df_sorted.groupby('market_ticker').apply(max_run_length).reset_index()
    runs.columns = ['market_ticker', 'max_run']

    # Also get the dominant direction of the longest run
    def get_run_direction(group):
        sides = group['taker_side'].values
        if len(sides) < 3:
            return None

        max_run = 1
        current_run = 1
        max_run_side = sides[0]
        current_side = sides[0]

        for i in range(1, len(sides)):
            if sides[i] == sides[i-1]:
                current_run += 1
                if current_run > max_run:
                    max_run = current_run
                    max_run_side = sides[i]
            else:
                current_run = 1
                current_side = sides[i]
        return max_run_side

    run_directions = df_sorted.groupby('market_ticker').apply(get_run_direction).reset_index()
    run_directions.columns = ['market_ticker', 'run_direction']

    runs = runs.merge(run_directions, on='market_ticker')

    # Markets with long YES runs (5+) - fade them
    long_yes_runs = runs[(runs['max_run'] >= 5) & (runs['run_direction'] == 'yes')]['market_ticker']
    signal_markets = market_stats[market_stats['market_ticker'].isin(long_yes_runs)]

    print(f"Found {len(signal_markets)} markets with long YES runs (5+)")

    if len(signal_markets) >= 100:
        result = validate_strategy(signal_markets, market_stats, "H054: Long YES Runs (Fade)")
        if result['status'] == 'validated':
            return result

    # Markets with long NO runs - follow them
    long_no_runs = runs[(runs['max_run'] >= 5) & (runs['run_direction'] == 'no')]['market_ticker']
    signal_markets = market_stats[market_stats['market_ticker'].isin(long_no_runs)]

    print(f"\nFound {len(signal_markets)} markets with long NO runs (5+)")

    if len(signal_markets) >= 100:
        return validate_strategy(signal_markets, market_stats, "H054: Long NO Runs (Follow)")

    return {'status': 'rejected', 'reason': 'insufficient_markets'}


def test_h056_contrarian_extreme(df, market_stats):
    """
    H056: Contrarian at Extreme Prices Only
    Test if fading YES trades works specifically at extreme YES prices (>85c)
    """
    print("\n" + "="*80)
    print("TESTING H056: Contrarian at Extreme Prices")
    print("="*80)

    # Find markets where YES was trading at extreme prices (>85c)
    extreme_yes_markets = df[
        (df['taker_side'] == 'yes') &
        (df['trade_price'] > 85)
    ]['market_ticker'].unique()

    signal_markets = market_stats[market_stats['market_ticker'].isin(extreme_yes_markets)]

    print(f"Found {len(signal_markets)} markets with extreme YES trades (>85c)")

    if len(signal_markets) >= 100:
        result = validate_strategy(signal_markets, market_stats, "H056: Fade Extreme YES (>85c)")
        if result['status'] == 'validated':
            return result

    # Also try extreme NO prices (>85c)
    extreme_no_markets = df[
        (df['taker_side'] == 'no') &
        (df['trade_price'] > 85)  # This is NO price > 85c (expensive NO)
    ]['market_ticker'].unique()

    signal_markets = market_stats[market_stats['market_ticker'].isin(extreme_no_markets)]

    print(f"\nFound {len(signal_markets)} markets with extreme NO trades (>85c)")

    if len(signal_markets) >= 100:
        return validate_strategy(signal_markets, market_stats, "H056: Follow Extreme NO (>85c)")

    return {'status': 'rejected', 'reason': 'price_proxy_likely'}


def test_h057_first_trade_direction(df, market_stats):
    """
    H057: First Trade Direction Persistence
    Does the direction of the first trade predict the outcome?
    """
    print("\n" + "="*80)
    print("TESTING H057: First Trade Direction Persistence")
    print("="*80)

    # Get first trade for each market
    first_trades = df.sort_values(['market_ticker', 'timestamp']).groupby('market_ticker').first().reset_index()

    # Markets where first trade was YES
    first_yes = first_trades[first_trades['taker_side'] == 'yes']['market_ticker']
    signal_markets_yes = market_stats[market_stats['market_ticker'].isin(first_yes)]

    print(f"Found {len(signal_markets_yes)} markets with first trade = YES")

    # For these, we FADE the first trade (bet NO)
    if len(signal_markets_yes) >= 100:
        result = validate_strategy(signal_markets_yes, market_stats, "H057: First YES -> Fade")
        if result['status'] == 'validated':
            return result

    # Markets where first trade was NO
    first_no = first_trades[first_trades['taker_side'] == 'no']['market_ticker']
    signal_markets_no = market_stats[market_stats['market_ticker'].isin(first_no)]

    print(f"\nFound {len(signal_markets_no)} markets with first trade = NO")

    if len(signal_markets_no) >= 100:
        return validate_strategy(signal_markets_no, market_stats, "H057: First NO -> Follow")

    return {'status': 'rejected', 'reason': 'no_edge'}


def test_leverage_extremes(df, market_stats):
    """
    Test extreme leverage combinations not tested before
    """
    print("\n" + "="*80)
    print("TESTING: Extreme Leverage Patterns")
    print("="*80)

    # Markets with VERY high leverage trades (>5x)
    very_high_lev = df[df['leverage_ratio'] > 5]['market_ticker'].unique()
    signal_markets = market_stats[market_stats['market_ticker'].isin(very_high_lev)]

    print(f"Found {len(signal_markets)} markets with very high leverage (>5x)")

    if len(signal_markets) >= 100:
        result = validate_strategy(signal_markets, market_stats, "Very High Leverage (>5x)")
        if result['status'] == 'validated':
            return result

    # Markets with consistently HIGH leverage (mean > 3)
    high_lev_markets = market_stats[market_stats['avg_leverage'] > 3]

    print(f"\nFound {len(high_lev_markets)} markets with mean leverage > 3")

    if len(high_lev_markets) >= 100:
        result = validate_strategy(high_lev_markets, market_stats, "Consistently High Leverage (mean > 3)")
        if result['status'] == 'validated':
            return result

    return {'status': 'rejected', 'reason': 'no_new_edge'}


def test_leverage_with_time(df, market_stats):
    """
    Combine leverage signal with specific time windows
    """
    print("\n" + "="*80)
    print("TESTING: Leverage + Time Combinations")
    print("="*80)

    # Add hour to trades
    df_copy = df.copy()
    df_copy['hour'] = pd.to_datetime(df_copy['datetime']).dt.hour
    df_copy['day_of_week'] = pd.to_datetime(df_copy['datetime']).dt.dayofweek

    # High leverage during market open (9-11 AM ET)
    morning_high_lev = df_copy[
        (df_copy['hour'].isin([9, 10, 11])) &
        (df_copy['leverage_ratio'] > 2)
    ]['market_ticker'].unique()

    signal_markets = market_stats[market_stats['market_ticker'].isin(morning_high_lev)]

    print(f"Found {len(signal_markets)} markets with morning high leverage (9-11 AM, lev > 2)")

    if len(signal_markets) >= 100:
        result = validate_strategy(signal_markets, market_stats, "Morning High Leverage")
        if result['status'] == 'validated':
            return result

    # High leverage on weekdays only
    weekday_high_lev = df_copy[
        (df_copy['day_of_week'].isin([0, 1, 2, 3, 4])) &  # Mon-Fri
        (df_copy['leverage_ratio'] > 2.5)
    ]['market_ticker'].unique()

    signal_markets = market_stats[market_stats['market_ticker'].isin(weekday_high_lev)]

    print(f"\nFound {len(signal_markets)} markets with weekday high leverage (lev > 2.5)")

    if len(signal_markets) >= 100:
        result = validate_strategy(signal_markets, market_stats, "Weekday High Leverage")
        if result['status'] == 'validated':
            return result

    return {'status': 'rejected', 'reason': 'no_new_edge'}


def test_category_specific(df, market_stats):
    """
    Test category-specific leverage patterns
    """
    print("\n" + "="*80)
    print("TESTING: Category-Specific Leverage")
    print("="*80)

    # Extract category from ticker
    market_stats_copy = market_stats.copy()
    market_stats_copy['category'] = market_stats_copy['market_ticker'].str.extract(r'^(KX[A-Z]+)')[0]

    # Get categories with sufficient data
    category_counts = market_stats_copy['category'].value_counts()
    valid_categories = category_counts[category_counts >= 500].index

    print(f"Categories with 500+ markets: {list(valid_categories)}")

    results = []

    for category in valid_categories:
        cat_markets = market_stats_copy[market_stats_copy['category'] == category]

        # Test high leverage in this category
        high_lev_cat = cat_markets[cat_markets['avg_leverage'] > 2]

        if len(high_lev_cat) >= 100:
            print(f"\nTesting {category} with high leverage: {len(high_lev_cat)} markets")
            result = validate_strategy(high_lev_cat, market_stats, f"{category} High Leverage")
            if result['status'] == 'validated':
                results.append((category, result))

    if results:
        return results[0][1]  # Return first validated result

    return {'status': 'rejected', 'reason': 'no_category_edge'}


def test_trade_size_patterns(df, market_stats):
    """
    Test various trade size patterns
    """
    print("\n" + "="*80)
    print("TESTING: Trade Size Patterns")
    print("="*80)

    # Markets with large average trade size
    large_trade_markets = df.groupby('market_ticker')['count'].mean().reset_index()
    large_trade_markets.columns = ['market_ticker', 'avg_size']

    # Top quartile by trade size
    size_threshold = large_trade_markets['avg_size'].quantile(0.75)
    large_markets = large_trade_markets[large_trade_markets['avg_size'] > size_threshold]['market_ticker']

    signal_markets = market_stats[market_stats['market_ticker'].isin(large_markets)]

    print(f"Found {len(signal_markets)} markets with large avg trade size (>{size_threshold:.0f} contracts)")

    if len(signal_markets) >= 100:
        result = validate_strategy(signal_markets, market_stats, "Large Trade Size Markets")
        if result['status'] == 'validated':
            return result

    # Small trade markets (retail dominated)
    size_threshold = large_trade_markets['avg_size'].quantile(0.25)
    small_markets = large_trade_markets[large_trade_markets['avg_size'] < size_threshold]['market_ticker']

    signal_markets = market_stats[market_stats['market_ticker'].isin(small_markets)]

    print(f"\nFound {len(signal_markets)} markets with small avg trade size (<{size_threshold:.0f} contracts)")

    if len(signal_markets) >= 100:
        return validate_strategy(signal_markets, market_stats, "Small Trade Size Markets")

    return {'status': 'rejected', 'reason': 'no_size_edge'}


def main():
    """Run all hypothesis tests"""
    print("="*80)
    print("SESSION 012: EXHAUSTIVE HYPOTHESIS TESTING")
    print(f"Started: {datetime.now()}")
    print("="*80)

    # Load data
    df = load_data()
    market_stats = calculate_market_stats(df)

    print(f"\nMarket stats calculated: {len(market_stats)} markets")

    # Run all hypothesis tests
    results = {}

    tests = [
        ('H089', test_h089_interval_trading),
        ('H091', test_h091_size_ratio_consistency),
        ('H096', test_h096_quote_stuffing_aftermath),
        ('H051', test_h051_trade_size_skew),
        ('H054', test_h054_consecutive_runs),
        ('H056', test_h056_contrarian_extreme),
        ('H057', test_h057_first_trade_direction),
        ('Leverage_Extremes', test_leverage_extremes),
        ('Leverage_Time', test_leverage_with_time),
        ('Category_Specific', test_category_specific),
        ('Trade_Size', test_trade_size_patterns),
    ]

    validated_strategies = []

    for name, test_func in tests:
        try:
            result = test_func(df, market_stats)
            results[name] = result

            if result['status'] == 'validated':
                validated_strategies.append((name, result))
                print(f"\n*** VALIDATED: {name} ***")
        except Exception as e:
            print(f"\nError in {name}: {e}")
            results[name] = {'status': 'error', 'error': str(e)}

    # Summary
    print("\n" + "="*80)
    print("SESSION 012 SUMMARY")
    print("="*80)

    print(f"\nTotal tests run: {len(tests)}")
    print(f"Validated: {len(validated_strategies)}")
    print(f"Rejected: {len([r for r in results.values() if r.get('status') == 'rejected'])}")
    print(f"Errors: {len([r for r in results.values() if r.get('status') == 'error'])}")

    if validated_strategies:
        print("\n" + "*"*60)
        print("VALIDATED STRATEGIES:")
        print("*"*60)
        for name, result in validated_strategies:
            print(f"\n{name}:")
            print(f"  Edge: {result['edge']*100:.2f}%")
            print(f"  Improvement: {result['improvement']*100:.2f}%")
            print(f"  Markets: {result['n_markets']}")
            print(f"  P-value: {result['p_value']:.2e}")
    else:
        print("\nNo new strategies validated in this session.")

    # Save results
    output_path = os.path.join(REPORT_PATH, f'session012_results_{datetime.now().strftime("%Y%m%d_%H%M")}.json')

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results, validated_strategies


if __name__ == "__main__":
    results, validated = main()
