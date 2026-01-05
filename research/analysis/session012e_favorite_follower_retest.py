"""
Session 012e: Re-validate "Favorite Follower" (NO at 91-97c)
Using Session 012c strict methodology with bucket-by-bucket baseline comparison.

Original claim (experimental/VALIDATED_STRATEGIES.md):
- 311 markets
- 95.2% win rate
- 4.93% ROI

This analysis will determine if it's a genuine edge or just a price proxy.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/session012e_favorite_follower_retest.json'


def load_data():
    """Load the enriched trades data."""
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df


def calculate_baseline_win_rates(df, bucket_size=1):
    """
    Calculate baseline win rates for NO bets at each price bucket.
    This is the win rate you'd get by betting NO on ALL markets at that price.
    """
    # Get market-level data
    markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',  # Average NO price in that market
        'yes_price': 'mean'
    }).reset_index()

    # Create buckets
    markets['no_price_bucket'] = (markets['no_price'] // bucket_size) * bucket_size

    # Calculate baseline win rate per bucket
    baseline = markets.groupby('no_price_bucket').agg({
        'market_result': lambda x: (x == 'no').mean(),
        'market_ticker': 'count'
    }).reset_index()
    baseline.columns = ['bucket', 'baseline_win_rate', 'baseline_n']

    return baseline


def analyze_favorite_follower(df):
    """
    Analyze the Favorite Follower strategy (NO at 91-97c).
    Use bucket-by-bucket baseline comparison to check for price proxy.
    """
    print("\n" + "=" * 80)
    print("STRATEGY: Favorite Follower (NO at 91-97c)")
    print("=" * 80)

    # Define signal: NO price in 91-97c range
    # The original strategy says bet NO when NO price is 91-97c
    # This means YES price is 3-9c (100 - 91 to 100 - 97 = 9 to 3)

    # Get market-level aggregates
    markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'count': 'sum',
        'datetime': 'first'  # For temporal analysis
    }).reset_index()

    # Signal markets: Average NO price between 91-97c
    signal_markets = markets[(markets['no_price'] >= 91) & (markets['no_price'] <= 97)].copy()

    n = len(signal_markets)
    if n == 0:
        print("ERROR: No markets found with NO price 91-97c")
        return None

    no_wins = (signal_markets['market_result'] == 'no').sum()
    win_rate = no_wins / n
    avg_no_price = signal_markets['no_price'].mean()
    breakeven = avg_no_price / 100  # Breakeven win rate to profit at that price
    edge = win_rate - breakeven

    # Statistical significance
    z = (no_wins - n * breakeven) / np.sqrt(n * breakeven * (1 - breakeven)) if 0 < breakeven < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    print(f"\n--- RAW METRICS ---")
    print(f"N Markets: {n}")
    print(f"NO Win Rate: {win_rate:.1%}")
    print(f"Avg NO Price: {avg_no_price:.2f}c")
    print(f"Breakeven: {breakeven:.1%}")
    print(f"Edge: {edge * 100:+.2f}%")
    print(f"P-value: {p_value:.2e}")

    # Check concentration
    signal_markets['profit'] = signal_markets.apply(
        lambda r: (100 - r['no_price']) / 100 if r['market_result'] == 'no' else -1,
        axis=1
    )
    total_profit = signal_markets['profit'].sum()

    if total_profit > 0:
        signal_markets['profit_pct'] = signal_markets['profit'] / total_profit * 100
        top_10_pct = signal_markets.nlargest(10, 'profit')['profit_pct'].sum()
    else:
        top_10_pct = 0

    print(f"\n--- CONCENTRATION CHECK ---")
    print(f"Total Profit: {total_profit:.2f} units")
    print(f"Top 10 Markets: {top_10_pct:.1f}% of profit")
    print(f"Concentration OK: {'YES' if top_10_pct < 30 else 'NO'}")

    # CRITICAL: Bucket-by-bucket baseline comparison
    print(f"\n--- BUCKET-BY-BUCKET ANALYSIS ---")
    print("(Comparing signal win rate to baseline at same NO price)")

    # Build baseline from ALL markets
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()

    # Use 1c buckets for fine-grained comparison
    signal_markets['bucket'] = signal_markets['no_price'].round(0)
    all_markets['bucket'] = all_markets['no_price'].round(0)

    bucket_results = []

    print(f"\n{'Bucket':<10} {'Base WR':<12} {'Sig WR':<12} {'Improve':<12} {'N Signal':<10} {'N Base':<10}")
    print("-" * 66)

    for bucket in sorted(signal_markets['bucket'].unique()):
        sig = signal_markets[signal_markets['bucket'] == bucket]
        base = all_markets[all_markets['bucket'] == bucket]

        sig_n = len(sig)
        base_n = len(base)

        if sig_n >= 5 and base_n >= 5:  # Minimum sample
            sig_wr = (sig['market_result'] == 'no').mean()
            base_wr = (base['market_result'] == 'no').mean()
            improvement = sig_wr - base_wr

            bucket_results.append({
                'bucket': int(bucket),
                'baseline_wr': float(base_wr),
                'signal_wr': float(sig_wr),
                'improvement': float(improvement),
                'n_signal': int(sig_n),
                'n_baseline': int(base_n)
            })

            print(f"{bucket:.0f}c       "
                  f"{base_wr:.1%}        "
                  f"{sig_wr:.1%}        "
                  f"{improvement * 100:+.1f}%        "
                  f"{sig_n:<10} "
                  f"{base_n:<10}")

    # Calculate weighted improvement
    if bucket_results:
        total_n = sum(b['n_signal'] for b in bucket_results)
        weighted_improvement = sum(b['improvement'] * b['n_signal'] for b in bucket_results) / total_n
        positive_buckets = sum(1 for b in bucket_results if b['improvement'] > 0)
        total_buckets = len(bucket_results)
    else:
        weighted_improvement = 0
        positive_buckets = 0
        total_buckets = 0

    print(f"\n--- PRICE PROXY CHECK ---")
    print(f"Positive Buckets: {positive_buckets}/{total_buckets}")
    print(f"Weighted Improvement: {weighted_improvement * 100:+.2f}%")

    is_price_proxy = weighted_improvement <= 0 or positive_buckets < total_buckets / 2
    print(f"IS PRICE PROXY: {'YES' if is_price_proxy else 'NO'}")

    # Temporal stability check
    print(f"\n--- TEMPORAL STABILITY ---")
    signal_markets['date'] = pd.to_datetime(signal_markets['datetime']).dt.date
    dates = sorted(signal_markets['date'].unique())

    if len(dates) >= 4:
        quarters = np.array_split(dates, 4)
        quarter_results = []

        for i, q_dates in enumerate(quarters):
            q_data = signal_markets[signal_markets['date'].isin(q_dates)]
            if len(q_data) >= 10:
                q_wr = (q_data['market_result'] == 'no').mean()
                q_be = q_data['no_price'].mean() / 100
                q_edge = q_wr - q_be
                quarter_results.append({
                    'quarter': i + 1,
                    'n': len(q_data),
                    'win_rate': float(q_wr),
                    'breakeven': float(q_be),
                    'edge': float(q_edge)
                })
                print(f"Q{i+1}: {len(q_data)} markets, WR={q_wr:.1%}, BE={q_be:.1%}, Edge={q_edge*100:+.1f}%")

        positive_quarters = sum(1 for q in quarter_results if q['edge'] > 0)
        print(f"Positive Quarters: {positive_quarters}/4")
    else:
        quarter_results = []
        positive_quarters = 0

    # Final verdict
    print(f"\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    validation_checks = {
        'n_markets_pass': n >= 100,
        'edge_positive': edge > 0,
        'p_value_pass': p_value < 0.01,
        'concentration_pass': top_10_pct < 30,
        'not_price_proxy': not is_price_proxy,
        'temporal_stable': positive_quarters >= 2 if quarter_results else False
    }

    all_pass = all(validation_checks.values())

    print(f"N >= 100: {'PASS' if validation_checks['n_markets_pass'] else 'FAIL'} ({n})")
    print(f"Edge > 0: {'PASS' if validation_checks['edge_positive'] else 'FAIL'} ({edge*100:+.2f}%)")
    print(f"P < 0.01: {'PASS' if validation_checks['p_value_pass'] else 'FAIL'} (p={p_value:.2e})")
    print(f"Concentration < 30%: {'PASS' if validation_checks['concentration_pass'] else 'FAIL'} ({top_10_pct:.1f}%)")
    print(f"Not Price Proxy: {'PASS' if validation_checks['not_price_proxy'] else 'FAIL'} (improvement={weighted_improvement*100:+.2f}%)")
    print(f"Temporal Stability: {'PASS' if validation_checks['temporal_stable'] else 'FAIL'} ({positive_quarters}/4 quarters)")

    if all_pass:
        verdict = "VALIDATED"
        reason = "All checks pass"
    elif is_price_proxy:
        verdict = "REJECTED - PRICE PROXY"
        reason = f"Signal improvement of {weighted_improvement*100:+.2f}% vs baseline at same prices"
    elif not validation_checks['p_value_pass']:
        verdict = "REJECTED - NOT SIGNIFICANT"
        reason = f"p-value {p_value:.2e} > 0.01"
    else:
        failing = [k for k, v in validation_checks.items() if not v]
        verdict = "REJECTED"
        reason = f"Failed checks: {failing}"

    print(f"\nVERDICT: {verdict}")
    print(f"REASON: {reason}")

    # Compile results
    results = {
        'strategy': 'Favorite Follower (NO at 91-97c)',
        'original_claim': {
            'markets': 311,
            'win_rate': 0.952,
            'roi': 0.0493
        },
        'retest_results': {
            'n_markets': int(n),
            'win_rate': float(win_rate),
            'avg_no_price': float(avg_no_price),
            'breakeven': float(breakeven),
            'edge': float(edge),
            'p_value': float(p_value)
        },
        'concentration': {
            'top_10_pct': float(top_10_pct),
            'pass': bool(top_10_pct < 30)
        },
        'price_proxy_analysis': {
            'bucket_results': bucket_results,
            'positive_buckets': int(positive_buckets),
            'total_buckets': int(total_buckets),
            'weighted_improvement': float(weighted_improvement),
            'is_price_proxy': bool(is_price_proxy)
        },
        'temporal_stability': {
            'quarter_results': quarter_results,
            'positive_quarters': int(positive_quarters)
        },
        'validation_checks': {k: bool(v) for k, v in validation_checks.items()},
        'verdict': verdict,
        'reason': reason,
        'timestamp': datetime.now().isoformat()
    }

    return results


def compare_to_baseline_across_range(df):
    """
    Compare the 91-97c range to adjacent ranges to see if there's something special.
    """
    print("\n" + "=" * 80)
    print("COMPARISON: Signal Range vs Adjacent Ranges")
    print("=" * 80)

    markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean'
    }).reset_index()

    ranges = [
        ('85-90c', 85, 90),
        ('91-97c (Signal)', 91, 97),
        ('98-100c', 98, 100)
    ]

    print(f"\n{'Range':<20} {'N':<10} {'Win Rate':<12} {'Breakeven':<12} {'Edge':<12}")
    print("-" * 66)

    for name, lo, hi in ranges:
        subset = markets[(markets['no_price'] >= lo) & (markets['no_price'] <= hi)]
        n = len(subset)
        if n >= 10:
            wr = (subset['market_result'] == 'no').mean()
            be = subset['no_price'].mean() / 100
            edge = wr - be
            print(f"{name:<20} {n:<10} {wr:.1%}        {be:.1%}        {edge*100:+.2f}%")


def main():
    print("=" * 80)
    print("SESSION 012e: Re-validate Favorite Follower (NO at 91-97c)")
    print(f"Started: {datetime.now()}")
    print("Methodology: Session 012c strict bucket-by-bucket baseline comparison")
    print("=" * 80)

    df = load_data()
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Main analysis
    results = analyze_favorite_follower(df)

    if results:
        # Additional context
        compare_to_baseline_across_range(df)

        # Save results
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {OUTPUT_PATH}")

    print("\n" + "=" * 80)
    print("SESSION 012e COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
