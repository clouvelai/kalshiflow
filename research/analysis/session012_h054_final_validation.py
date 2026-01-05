"""
Session 012: Final Validation of H054 - Long YES Runs
Make sure the signal is actionable and not just statistical artifact
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
    print("Loading trade data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def analyze_h054_actionability(df):
    """
    Check if H054 signal is actionable in real-time trading
    Key question: Can we detect the run BEFORE market settles?
    """
    print("\n" + "="*80)
    print("H054 ACTIONABILITY ANALYSIS")
    print("="*80)

    df_sorted = df.sort_values(['market_ticker', 'timestamp'])

    results = []

    for ticker, group in df_sorted.groupby('market_ticker'):
        sides = group['taker_side'].values
        prices = group['trade_price'].values
        market_result = group['market_result'].iloc[0]
        n_trades = len(sides)

        if n_trades < 5:
            continue

        # Find the FIRST time we see 5+ consecutive YES trades
        first_yes_run_end = None
        current_run = 1
        current_side = sides[0]

        for i in range(1, len(sides)):
            if sides[i] == sides[i-1]:
                current_run += 1
                if current_run >= 5 and sides[i] == 'yes' and first_yes_run_end is None:
                    first_yes_run_end = i
            else:
                current_run = 1
                current_side = sides[i]

        if first_yes_run_end is not None:
            # Calculate the NO price at the moment we detect the signal
            if sides[first_yes_run_end] == 'yes':
                entry_no_price = 100 - prices[first_yes_run_end]
            else:
                entry_no_price = prices[first_yes_run_end]

            # How many trades remain after the signal?
            trades_remaining = n_trades - first_yes_run_end - 1

            # What percentage through the market's lifetime is this signal?
            signal_position = (first_yes_run_end + 1) / n_trades

            results.append({
                'market_ticker': ticker,
                'signal_detected_at_trade': first_yes_run_end + 1,
                'total_trades': n_trades,
                'trades_remaining': trades_remaining,
                'signal_position': signal_position,
                'entry_no_price': entry_no_price,
                'market_result': market_result
            })

    signal_markets = pd.DataFrame(results)
    print(f"\nFound {len(signal_markets)} markets where 5+ YES run was detected")

    # Analyze actionability
    print("\n1. Signal Timing:")
    print(f"  Average signal detected at trade #{signal_markets['signal_detected_at_trade'].mean():.1f}")
    print(f"  Median trades remaining after signal: {signal_markets['trades_remaining'].median():.0f}")
    print(f"  Mean signal position (0=start, 1=end): {signal_markets['signal_position'].mean():.2f}")

    # Edge by signal timing
    print("\n2. Edge by Signal Position:")
    early_signals = signal_markets[signal_markets['signal_position'] < 0.5]
    late_signals = signal_markets[signal_markets['signal_position'] >= 0.5]

    early_wr = (early_signals['market_result'] == 'no').mean() if len(early_signals) > 0 else 0
    late_wr = (late_signals['market_result'] == 'no').mean() if len(late_signals) > 0 else 0
    early_be = early_signals['entry_no_price'].mean() / 100 if len(early_signals) > 0 else 0.5
    late_be = late_signals['entry_no_price'].mean() / 100 if len(late_signals) > 0 else 0.5

    print(f"  Early signals (<50% through): N={len(early_signals)}, WR={early_wr:.1%}, BE={early_be:.1%}, Edge={(early_wr-early_be)*100:.1f}%")
    print(f"  Late signals (>=50% through): N={len(late_signals)}, WR={late_wr:.1%}, BE={late_be:.1%}, Edge={(late_wr-late_be)*100:.1f}%")

    # Edge by trades remaining
    print("\n3. Edge by Trades Remaining After Signal:")
    for threshold in [0, 5, 10, 20]:
        subset = signal_markets[signal_markets['trades_remaining'] >= threshold]
        if len(subset) >= 50:
            subset_wr = (subset['market_result'] == 'no').mean()
            subset_be = subset['entry_no_price'].mean() / 100
            print(f"  Trades remaining >= {threshold}: N={len(subset)}, WR={subset_wr:.1%}, BE={subset_be:.1%}, Edge={(subset_wr-subset_be)*100:.1f}%")

    # Actual entry price distribution
    print("\n4. Entry NO Price Distribution:")
    print(f"  Mean: {signal_markets['entry_no_price'].mean():.1f}c")
    print(f"  Median: {signal_markets['entry_no_price'].median():.1f}c")
    print(f"  Std: {signal_markets['entry_no_price'].std():.1f}c")

    # Entry price buckets
    print("\n5. Edge by Entry NO Price Bucket:")
    signal_markets['no_price_bucket'] = (signal_markets['entry_no_price'] // 10) * 10

    for bucket in sorted(signal_markets['no_price_bucket'].unique()):
        subset = signal_markets[signal_markets['no_price_bucket'] == bucket]
        if len(subset) >= 30:
            wr = (subset['market_result'] == 'no').mean()
            be = bucket / 100 + 0.05  # Midpoint of bucket
            print(f"  {bucket:.0f}-{bucket+10:.0f}c: N={len(subset)}, WR={wr:.1%}, BE~{be:.0%}, Edge~{(wr-be)*100:.0f}%")

    # Calculate overall edge with actual entry prices
    print("\n6. Final Edge Calculation with Actual Entry Prices:")
    total_wr = (signal_markets['market_result'] == 'no').mean()
    total_be = signal_markets['entry_no_price'].mean() / 100
    total_edge = total_wr - total_be

    print(f"  NO Win Rate: {total_wr:.1%}")
    print(f"  Average Entry NO Price: {signal_markets['entry_no_price'].mean():.1f}c")
    print(f"  Breakeven: {total_be:.1%}")
    print(f"  Expected Edge: {total_edge*100:.2f}%")

    # P-value
    n = len(signal_markets)
    k = (signal_markets['market_result'] == 'no').sum()
    z = (k - n * total_be) / np.sqrt(n * total_be * (1 - total_be))
    p_value = 1 - stats.norm.cdf(z)

    print(f"  P-value: {p_value:.2e}")

    # Concentration check
    signal_markets['profit'] = np.where(
        signal_markets['market_result'] == 'no',
        100 - signal_markets['entry_no_price'],
        -signal_markets['entry_no_price']
    )
    total_profit = signal_markets[signal_markets['profit'] > 0]['profit'].sum()
    if total_profit > 0:
        signal_markets['profit_contribution'] = np.where(
            signal_markets['profit'] > 0,
            signal_markets['profit'] / total_profit,
            0
        )
        max_concentration = signal_markets['profit_contribution'].max()
    else:
        max_concentration = 0

    print(f"  Max Concentration: {max_concentration*100:.1f}%")

    # CRITICAL: Price proxy check vs baseline
    print("\n7. Price Proxy Check (vs Baseline at Same Prices):")

    # Get all markets and their avg NO price
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'trade_price': 'mean',
        'taker_side': lambda x: (x == 'yes').mean()  # YES ratio
    }).reset_index()
    all_markets.columns = ['market_ticker', 'market_result', 'avg_trade_price', 'yes_ratio']

    # Calculate effective NO price
    # For high YES ratio markets, avg_trade_price is ~ YES price, so NO = 100 - YES
    # For low YES ratio markets, avg_trade_price is ~ NO price
    all_markets['effective_no_price'] = np.where(
        all_markets['yes_ratio'] > 0.5,
        100 - all_markets['avg_trade_price'],
        all_markets['avg_trade_price']
    )

    # Bucket comparison
    signal_markets_copy = signal_markets.copy()

    comparison_results = []
    for bucket in sorted(signal_markets_copy['no_price_bucket'].unique()):
        signal_subset = signal_markets_copy[signal_markets_copy['no_price_bucket'] == bucket]
        baseline_subset = all_markets[
            (all_markets['effective_no_price'] >= bucket) &
            (all_markets['effective_no_price'] < bucket + 10)
        ]

        if len(signal_subset) >= 10 and len(baseline_subset) >= 10:
            signal_wr = (signal_subset['market_result'] == 'no').mean()
            baseline_wr = (baseline_subset['market_result'] == 'no').mean()

            comparison_results.append({
                'bucket': bucket,
                'signal_n': len(signal_subset),
                'baseline_n': len(baseline_subset),
                'signal_wr': signal_wr,
                'baseline_wr': baseline_wr,
                'improvement': signal_wr - baseline_wr
            })

    comparison_df = pd.DataFrame(comparison_results)

    print(f"\n  {'Bucket':<10} {'Signal WR':<12} {'Base WR':<12} {'Improve':<10}")
    for _, row in comparison_df.iterrows():
        print(f"  {row['bucket']:.0f}-{row['bucket']+10:.0f}c   "
              f"{row['signal_wr']:.1%}        "
              f"{row['baseline_wr']:.1%}        "
              f"{row['improvement']*100:+.1f}%")

    # Weighted improvement
    comparison_df['weight'] = comparison_df['signal_n'] / comparison_df['signal_n'].sum()
    weighted_improvement = (comparison_df['improvement'] * comparison_df['weight']).sum()

    print(f"\n  Weighted Improvement over Baseline: {weighted_improvement*100:.2f}%")

    is_valid = (
        total_edge > 0.01 and
        p_value < 0.0005 and
        weighted_improvement > 0.01 and
        max_concentration < 0.30
    )

    print("\n" + "="*80)
    print("FINAL VALIDATION RESULT")
    print("="*80)
    print(f"  Edge: {total_edge*100:.2f}%")
    print(f"  Improvement over baseline: {weighted_improvement*100:.2f}%")
    print(f"  Markets: {len(signal_markets)}")
    print(f"  P-value: {p_value:.2e}")
    print(f"  Concentration: {max_concentration*100:.1f}%")
    print(f"  VALID: {is_valid}")

    return {
        'strategy': 'H054: Fade Long YES Runs (5+)',
        'n_markets': len(signal_markets),
        'no_win_rate': float(total_wr),
        'avg_entry_no_price': float(signal_markets['entry_no_price'].mean()),
        'breakeven': float(total_be),
        'edge': float(total_edge),
        'p_value': float(p_value),
        'weighted_improvement': float(weighted_improvement),
        'max_concentration': float(max_concentration),
        'is_valid': bool(is_valid),
        'early_signal_edge': float((early_wr - early_be) if len(early_signals) > 0 else 0),
        'late_signal_edge': float((late_wr - late_be) if len(late_signals) > 0 else 0)
    }


def main():
    print("="*80)
    print("SESSION 012: H054 FINAL VALIDATION")
    print(f"Started: {datetime.now()}")
    print("="*80)

    df = load_data()

    result = analyze_h054_actionability(df)

    # Save results
    output_path = os.path.join(REPORT_PATH, f'session012_h054_final_{datetime.now().strftime("%Y%m%d_%H%M")}.json')
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return result


if __name__ == "__main__":
    main()
