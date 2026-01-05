"""
Session 012: Deep Validation of H054 and H056
Verify the findings from initial testing with rigorous methodology
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
REPORT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/'

def load_data():
    """Load trade data"""
    print("Loading trade data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def calculate_max_run(sides):
    """Calculate max run length and direction for a sequence of sides"""
    if len(sides) < 3:
        return 0, None, None

    max_run = 1
    current_run = 1
    max_run_side = sides[0]
    max_run_end_idx = 0

    for i in range(1, len(sides)):
        if sides[i] == sides[i-1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
                max_run_side = sides[i]
                max_run_end_idx = i
        else:
            current_run = 1

    return max_run, max_run_side, max_run_end_idx


def validate_h054_consecutive_runs(df):
    """
    Deep validation of H054: Consecutive Same-Side Trade Runs
    Signal: Markets with 5+ consecutive YES trades
    Action: Bet NO (fade the YES momentum)
    """
    print("\n" + "="*80)
    print("DEEP VALIDATION: H054 - Consecutive Same-Side Trade Runs")
    print("="*80)

    df_sorted = df.sort_values(['market_ticker', 'timestamp'])

    print("Calculating run information...")
    results = []

    for ticker, group in df_sorted.groupby('market_ticker'):
        sides = group['taker_side'].values
        prices = group['trade_price'].values
        taker_sides = group['taker_side'].values
        market_result = group['market_result'].iloc[0]

        max_run, run_dir, run_end_idx = calculate_max_run(sides)

        # Calculate NO price at run end
        if run_end_idx is not None and run_end_idx < len(prices):
            if taker_sides[run_end_idx] == 'no':
                run_end_no_price = prices[run_end_idx]
            else:
                run_end_no_price = 100 - prices[run_end_idx]
        else:
            run_end_no_price = None

        results.append({
            'market_ticker': ticker,
            'max_run': max_run,
            'run_direction': run_dir,
            'run_end_no_price': run_end_no_price,
            'market_result': market_result
        })

    runs = pd.DataFrame(results)
    print(f"Processed {len(runs)} markets")

    # Filter markets with long YES runs (5+)
    long_yes_runs = runs[(runs['max_run'] >= 5) & (runs['run_direction'] == 'yes')]

    print(f"\nFound {len(long_yes_runs)} markets with 5+ consecutive YES trades")

    # Calculate stats
    n_markets = len(long_yes_runs)
    no_wins = (long_yes_runs['market_result'] == 'no').sum()
    no_win_rate = no_wins / n_markets

    avg_no_price = long_yes_runs['run_end_no_price'].mean()
    breakeven = avg_no_price / 100 if pd.notna(avg_no_price) else 0.5

    edge = no_win_rate - breakeven

    print(f"\nSignal Stats:")
    print(f"  Markets: {n_markets}")
    print(f"  NO Win Rate: {no_win_rate:.1%}")
    print(f"  Avg NO Price: {avg_no_price:.1f}c")
    print(f"  Breakeven: {breakeven:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # P-value (normal approximation)
    expected = n_markets * breakeven
    std = np.sqrt(n_markets * breakeven * (1 - breakeven))
    z = (no_wins - expected) / std if std > 0 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"  P-value: {p_value:.2e}")

    # Price proxy check by bucket
    print("\n  Price Proxy Check by NO Price Bucket:")
    long_yes_runs_copy = long_yes_runs.copy()
    long_yes_runs_copy['no_price_bucket'] = (long_yes_runs_copy['run_end_no_price'] // 10) * 10

    # Calculate baseline for all markets
    all_runs_copy = runs.copy()
    all_runs_copy['no_price_bucket'] = (all_runs_copy['run_end_no_price'] // 10) * 10

    baseline = all_runs_copy.groupby('no_price_bucket').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    baseline.columns = ['no_price_bucket', 'baseline_wr', 'baseline_count']

    signal = long_yes_runs_copy.groupby('no_price_bucket').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    signal.columns = ['no_price_bucket', 'signal_wr', 'signal_count']

    comparison = signal.merge(baseline, on='no_price_bucket', how='left')
    comparison['improvement'] = comparison['signal_wr'] - comparison['baseline_wr']
    comparison['weight'] = comparison['signal_count'] / comparison['signal_count'].sum()

    weighted_improvement = (comparison['improvement'] * comparison['weight']).sum()

    print(f"\n  {'Bucket':<10} {'Signal WR':<12} {'Base WR':<12} {'Improve':<10} {'N':<8}")
    for _, row in comparison.iterrows():
        if pd.notna(row['baseline_wr']):
            print(f"  {row['no_price_bucket']:.0f}-{row['no_price_bucket']+10:.0f}c   "
                  f"{row['signal_wr']:.1%}        "
                  f"{row['baseline_wr']:.1%}        "
                  f"{row['improvement']*100:+.1f}%      "
                  f"{row['signal_count']:.0f}")

    print(f"\n  Weighted Improvement: {weighted_improvement*100:.2f}%")

    # Temporal stability
    print("\n  Temporal Stability:")
    sorted_markets = long_yes_runs.sort_values('market_ticker')
    n_half = len(sorted_markets) // 2
    first_half = sorted_markets.iloc[:n_half]
    second_half = sorted_markets.iloc[n_half:]

    first_wr = (first_half['market_result'] == 'no').mean()
    second_wr = (second_half['market_result'] == 'no').mean()
    first_be = first_half['run_end_no_price'].mean() / 100
    second_be = second_half['run_end_no_price'].mean() / 100

    print(f"  First half:  WR={first_wr:.1%}, BE={first_be:.1%}, Edge={(first_wr-first_be)*100:.1f}%")
    print(f"  Second half: WR={second_wr:.1%}, BE={second_be:.1%}, Edge={(second_wr-second_be)*100:.1f}%")

    # Check for different run lengths
    print("\n  Edge by Run Length:")
    for run_len in [5, 6, 7, 8, 10]:
        subset = runs[(runs['max_run'] >= run_len) & (runs['run_direction'] == 'yes')]
        if len(subset) >= 50:
            subset_wr = (subset['market_result'] == 'no').mean()
            subset_be = subset['run_end_no_price'].mean() / 100
            print(f"  Run >= {run_len}: N={len(subset)}, WR={subset_wr:.1%}, Edge={(subset_wr-subset_be)*100:.1f}%")

    is_valid = weighted_improvement > 0.01 and p_value < 0.0005

    return {
        'strategy': 'H054: Long YES Runs (Fade)',
        'n_markets': n_markets,
        'no_win_rate': float(no_win_rate),
        'breakeven': float(breakeven),
        'edge': float(edge),
        'p_value': float(p_value),
        'weighted_improvement': float(weighted_improvement),
        'is_valid': bool(is_valid)
    }


def validate_h056_extreme_no(df):
    """
    Deep validation of H056: Contrarian at Extreme NO Prices
    Signal: Markets with NO trades at >85c
    Action: Bet NO (follow the expensive NO trades)
    """
    print("\n" + "="*80)
    print("DEEP VALIDATION: H056 - Extreme NO Prices (>85c)")
    print("="*80)

    # Find markets with extreme NO trades
    extreme_no = df[
        (df['taker_side'] == 'no') &
        (df['trade_price'] > 85)  # NO price > 85c
    ]

    extreme_no_markets = extreme_no.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',  # This IS the NO price for NO trades
        'count': 'sum'
    }).reset_index()
    extreme_no_markets.columns = ['market_ticker', 'market_result', 'avg_no_price', 'contract_count']

    print(f"\nFound {len(extreme_no_markets)} markets with NO trades at >85c")

    # Stats
    n_markets = len(extreme_no_markets)
    no_wins = (extreme_no_markets['market_result'] == 'no').sum()
    no_win_rate = no_wins / n_markets

    avg_no_price = extreme_no_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100

    edge = no_win_rate - breakeven

    print(f"\nSignal Stats:")
    print(f"  Markets: {n_markets}")
    print(f"  NO Win Rate: {no_win_rate:.1%}")
    print(f"  Avg NO Price: {avg_no_price:.1f}c")
    print(f"  Breakeven: {breakeven:.1%}")
    print(f"  Edge: {edge*100:.2f}%")

    # P-value
    expected = n_markets * breakeven
    std = np.sqrt(n_markets * breakeven * (1 - breakeven))
    z = (no_wins - expected) / std if std > 0 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"  P-value: {p_value:.2e}")

    # CRITICAL: Compare to baseline at same NO prices
    print("\n  Price Proxy Check:")

    # All markets with high NO price (not just those with NO trades)
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'no').mean()
    }).reset_index()
    all_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'no_ratio']

    # Calculate effective NO price for baseline
    # For markets with mostly NO trades, avg_trade_price IS roughly NO price
    # For markets with mostly YES trades, NO price = 100 - avg_trade_price
    all_markets['effective_no_price'] = np.where(
        all_markets['no_ratio'] > 0.5,
        all_markets['avg_trade_price'],  # Mostly NO trades
        100 - all_markets['avg_trade_price']  # Mostly YES trades
    )

    # Baseline at 85-95c NO price
    baseline_markets = all_markets[
        (all_markets['effective_no_price'] >= 85) &
        (all_markets['effective_no_price'] <= 95)
    ]

    baseline_no_wr = (baseline_markets['market_result'] == 'no').mean() if len(baseline_markets) > 0 else 0.5

    print(f"  Baseline NO WR at 85-95c: {baseline_no_wr:.1%} (N={len(baseline_markets)})")
    print(f"  Signal NO WR: {no_win_rate:.1%}")
    print(f"  Improvement: {(no_win_rate - baseline_no_wr)*100:+.2f}%")

    improvement = no_win_rate - baseline_no_wr

    # Concentration check
    extreme_no_markets['profit'] = np.where(
        extreme_no_markets['market_result'] == 'no',
        (100 - extreme_no_markets['avg_no_price']) * extreme_no_markets['contract_count'] / 100,
        -extreme_no_markets['avg_no_price'] * extreme_no_markets['contract_count'] / 100
    )

    total_profit = extreme_no_markets[extreme_no_markets['profit'] > 0]['profit'].sum()
    if total_profit > 0:
        extreme_no_markets['profit_contribution'] = np.where(
            extreme_no_markets['profit'] > 0,
            extreme_no_markets['profit'] / total_profit,
            0
        )
        max_concentration = extreme_no_markets['profit_contribution'].max()
    else:
        max_concentration = 0

    print(f"\n  Max Concentration: {max_concentration*100:.1f}%")

    # Temporal stability
    print("\n  Temporal Stability:")
    sorted_markets = extreme_no_markets.sort_values('market_ticker')
    n_half = len(sorted_markets) // 2
    first_half = sorted_markets.iloc[:n_half]
    second_half = sorted_markets.iloc[n_half:]

    first_wr = (first_half['market_result'] == 'no').mean()
    second_wr = (second_half['market_result'] == 'no').mean()
    first_be = first_half['avg_no_price'].mean() / 100
    second_be = second_half['avg_no_price'].mean() / 100

    print(f"  First half:  WR={first_wr:.1%}, BE={first_be:.1%}, Edge={(first_wr-first_be)*100:.1f}%")
    print(f"  Second half: WR={second_wr:.1%}, BE={second_be:.1%}, Edge={(second_wr-second_be)*100:.1f}%")

    is_valid = improvement > 0.01 and p_value < 0.0005 and max_concentration < 0.30

    return {
        'strategy': 'H056: Follow Extreme NO (>85c)',
        'n_markets': n_markets,
        'no_win_rate': float(no_win_rate),
        'breakeven': float(breakeven),
        'edge': float(edge),
        'p_value': float(p_value),
        'improvement': float(improvement),
        'max_concentration': float(max_concentration),
        'is_valid': bool(is_valid)
    }


def analyze_h054_mechanism(df):
    """
    Understand WHY H054 might work
    """
    print("\n" + "="*80)
    print("H054 MECHANISM ANALYSIS")
    print("="*80)

    df_sorted = df.sort_values(['market_ticker', 'timestamp'])

    # Calculate runs for all markets
    results = []
    for ticker, group in df_sorted.groupby('market_ticker'):
        sides = group['taker_side'].values
        market_result = group['market_result'].iloc[0]
        max_run, run_dir, _ = calculate_max_run(sides)
        results.append({
            'market_ticker': ticker,
            'max_run': max_run,
            'run_direction': run_dir,
            'market_result': market_result,
            'trade_count': len(sides)
        })

    runs = pd.DataFrame(results)

    print("\n1. Run Length Distribution:")
    for run_len in range(1, 11):
        n = len(runs[runs['max_run'] == run_len])
        print(f"  Run = {run_len}: {n} markets ({n/len(runs)*100:.1f}%)")

    print("\n2. YES vs NO Runs:")
    yes_runs = runs[runs['run_direction'] == 'yes']
    no_runs = runs[runs['run_direction'] == 'no']

    yes_no_wr = (yes_runs['market_result'] == 'no').mean()
    no_no_wr = (no_runs['market_result'] == 'no').mean()

    print(f"  Markets with max YES run: {len(yes_runs)}, NO WR: {yes_no_wr:.1%}")
    print(f"  Markets with max NO run: {len(no_runs)}, NO WR: {no_no_wr:.1%}")

    print("\n3. Long Runs vs Short Runs:")
    long_runs = runs[runs['max_run'] >= 5]
    short_runs = runs[runs['max_run'] < 5]

    long_no_wr = (long_runs['market_result'] == 'no').mean()
    short_no_wr = (short_runs['market_result'] == 'no').mean()

    print(f"  Long runs (5+): {len(long_runs)} markets, NO WR: {long_no_wr:.1%}")
    print(f"  Short runs (<5): {len(short_runs)} markets, NO WR: {short_no_wr:.1%}")

    print("\n4. Behavioral Hypothesis:")
    print("  Long YES runs may indicate:")
    print("  - Retail FOMO (fear of missing out) driving momentum")
    print("  - Overconfidence in favorites")
    print("  - When retail piles in YES, the smart money is already on NO")
    print("  - Mean reversion after overextension")


def main():
    print("="*80)
    print("SESSION 012: DEEP VALIDATION")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()

    # Validate H054
    result_h054 = validate_h054_consecutive_runs(df)

    # Validate H056
    result_h056 = validate_h056_extreme_no(df)

    # Mechanism analysis
    analyze_h054_mechanism(df)

    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    print(f"\nH054 (Long YES Runs):")
    print(f"  Valid: {result_h054['is_valid']}")
    print(f"  Edge: {result_h054['edge']*100:.2f}%")
    print(f"  Improvement: {result_h054['weighted_improvement']*100:.2f}%")

    print(f"\nH056 (Extreme NO >85c):")
    print(f"  Valid: {result_h056['is_valid']}")
    print(f"  Edge: {result_h056['edge']*100:.2f}%")
    print(f"  Improvement: {result_h056['improvement']*100:.2f}%")

    # Save results
    results = {
        'H054': result_h054,
        'H056': result_h056,
        'timestamp': datetime.now().isoformat()
    }

    output_path = os.path.join(REPORT_PATH, f'session012_deep_validation_{datetime.now().strftime("%Y%m%d_%H%M")}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
