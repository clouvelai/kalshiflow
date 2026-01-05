#!/usr/bin/env python3
"""
LSD MODE v2: Deep dive on promising MM strategies
=================================================
Check if the winners from v1 are real or price proxies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from scipy import stats

# Paths
BASE_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research")
TRADES_FILE = BASE_DIR / "data/trades/enriched_trades_resolved_ALL.csv"
REPORT_FILE = BASE_DIR / "reports/lsd_mm_strategies_v2.json"

def load_data():
    """Load trade data."""
    print("Loading trade data...")
    df = pd.read_csv(TRADES_FILE, parse_dates=['datetime'], low_memory=False)
    df = df.rename(columns={'market_ticker': 'ticker', 'datetime': 'created_time'})
    df['is_yes'] = df['taker_side'] == 'yes'
    print(f"Loaded {len(df):,} trades across {df['ticker'].nunique():,} markets")
    return df

def get_market_summary(df):
    """Aggregate trades to market level."""
    print("Aggregating to market level...")

    markets = df.groupby('ticker').agg({
        'count': 'sum',
        'yes_price': ['first', 'last', 'mean', 'std', 'min', 'max'],
        'is_yes': ['sum', 'mean'],
        'created_time': ['first', 'last'],
        'result': 'first'
    })

    # Flatten columns
    markets.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in markets.columns]
    markets = markets.rename(columns={
        'count_sum': 'n_trades',
        'yes_price_first': 'first_price',
        'yes_price_last': 'last_price',
        'yes_price_mean': 'avg_price',
        'yes_price_std': 'price_std',
        'yes_price_min': 'min_price',
        'yes_price_max': 'max_price',
        'is_yes_sum': 'yes_trades',
        'is_yes_mean': 'yes_ratio',
        'created_time_first': 'first_trade_time',
        'created_time_last': 'last_trade_time',
        'result_first': 'result'
    })

    # Derived fields
    markets['no_ratio'] = 1 - markets['yes_ratio']
    markets['price_move'] = markets['last_price'] - markets['first_price']
    markets['price_range'] = markets['max_price'] - markets['min_price']
    markets['no_won'] = markets['result'] == 'no'
    markets['duration_hours'] = (markets['last_trade_time'] - markets['first_trade_time']).dt.total_seconds() / 3600

    # Price buckets for baseline comparison
    markets['no_price'] = 100 - markets['avg_price']
    markets['no_bucket'] = (markets['no_price'] // 5) * 5

    # Filter resolved only
    markets = markets[markets['result'].isin(['yes', 'no'])].copy()

    print(f"Aggregated to {len(markets):,} resolved markets")
    return markets

def calc_baseline_by_bucket(markets):
    """Calculate baseline NO win rate by NO price bucket."""
    baseline = markets.groupby('no_bucket').agg({
        'no_won': ['mean', 'count']
    })
    baseline.columns = ['base_win_rate', 'base_count']
    return baseline

def validate_with_buckets(markets, signal_mask, baseline, name):
    """Validate strategy with bucket-by-bucket comparison."""
    subset = markets[signal_mask].copy()
    if len(subset) < 50:
        return None

    # Merge with baseline
    subset = subset.merge(baseline, left_on='no_bucket', right_index=True, how='left')

    # Per-bucket analysis
    bucket_stats = subset.groupby('no_bucket').agg({
        'no_won': ['mean', 'count'],
        'base_win_rate': 'first'
    })
    bucket_stats.columns = ['signal_win', 'count', 'base_win']
    bucket_stats['improvement'] = bucket_stats['signal_win'] - bucket_stats['base_win']
    bucket_stats = bucket_stats[bucket_stats['count'] >= 10]  # Min per bucket

    if len(bucket_stats) < 3:
        return None

    # Calculate weighted improvement
    weights = bucket_stats['count'] / bucket_stats['count'].sum()
    weighted_improvement = (bucket_stats['improvement'] * weights).sum()

    # How many buckets positive?
    positive_buckets = (bucket_stats['improvement'] > 0).sum()
    total_buckets = len(bucket_stats)

    # Overall stats
    overall_win = subset['no_won'].mean()
    expected_win = subset['base_win_rate'].mean()
    raw_edge = (overall_win - expected_win) * 100

    # Is it real?
    is_real = (
        weighted_improvement > 0.02 and  # 2% improvement over baseline
        positive_buckets / total_buckets > 0.5  # Majority of buckets positive
    )

    return {
        'name': name,
        'markets': len(subset),
        'raw_edge': round(raw_edge, 2),
        'weighted_improvement': round(weighted_improvement * 100, 2),
        'positive_buckets': positive_buckets,
        'total_buckets': total_buckets,
        'bucket_pass_rate': round(positive_buckets / total_buckets * 100, 1),
        'IS_REAL': is_real,
        'bucket_details': bucket_stats.to_dict('index')
    }

def main():
    """Validate the top MM strategies from v1."""
    print("=" * 60)
    print("LSD MODE v2: Bucket Validation of Winners")
    print("=" * 60)

    df = load_data()
    markets = get_market_summary(df)
    baseline = calc_baseline_by_bucket(markets)

    results = {}

    # H-MM010: Price Reversal Magnitude (edge +10.4%)
    print("\n[H-MM010] Price Reversal Magnitude...")
    signal = markets['price_move'] < -10
    results['H-MM010'] = validate_with_buckets(markets, signal, baseline, 'Price Reversal (>10c drop)')

    # H-MM011: Price Momentum (edge +8.1%)
    print("\n[H-MM011] Price Momentum...")
    signal = markets['price_move'] < 0
    results['H-MM011'] = validate_with_buckets(markets, signal, baseline, 'Price Momentum (toward NO)')

    # H-MM008: Toxic Flow Reversal (edge +7.9%)
    print("\n[H-MM008] Toxic Flow Reversal...")
    signal = (markets['yes_ratio'] > 0.7) & (markets['price_move'] < -5)
    results['H-MM008'] = validate_with_buckets(markets, signal, baseline, 'YES flow + price drop')

    # RLM: Retail Losing Money (edge +7.3%)
    print("\n[RLM] Retail Losing Money...")
    signal = (markets['yes_ratio'] > 0.65) & (markets['price_move'] < 0) & (markets['n_trades'] >= 15)
    results['RLM'] = validate_with_buckets(markets, signal, baseline, 'RLM (YES>65% + price drop + 15+ trades)')

    # H-MM002: Spread Oscillation (edge +6.4%)
    print("\n[H-MM002] Spread Oscillation...")
    signal = (markets['price_range'] > 20) & (markets['price_std'] < 10)
    results['H-MM002'] = validate_with_buckets(markets, signal, baseline, 'High range + low std')

    # H-MM013: Round Number Magnetism (edge +5.4%)
    print("\n[H-MM013] Round Number Magnetism...")
    markets['near_round'] = markets['last_price'].apply(
        lambda x: min(abs(x - 25), abs(x - 50), abs(x - 75)) < 3
    )
    signal = markets['near_round']
    results['H-MM013'] = validate_with_buckets(markets, signal, baseline, 'Near round numbers')

    # H-MM006: Dead Zone Trading (edge +5.0%)
    print("\n[H-MM006] Dead Zone Trading...")
    signal = (markets['n_trades'] >= 5) & (markets['n_trades'] <= 10)
    results['H-MM006'] = validate_with_buckets(markets, signal, baseline, 'Low activity (5-10 trades)')

    # Bonus: Try combining top signals
    print("\n[COMBO1] RLM + Price drop > 5c...")
    signal = (
        (markets['yes_ratio'] > 0.65) &
        (markets['price_move'] < -5) &
        (markets['n_trades'] >= 15)
    )
    results['COMBO1'] = validate_with_buckets(markets, signal, baseline, 'RLM + big price drop')

    print("\n[COMBO2] Price momentum + NO flow...")
    signal = (markets['price_move'] < -5) & (markets['no_ratio'] > 0.5)
    results['COMBO2'] = validate_with_buckets(markets, signal, baseline, 'Price drop + NO majority')

    print("\n[COMBO3] RLM + Round number magnetism...")
    signal = (
        (markets['yes_ratio'] > 0.65) &
        (markets['price_move'] < 0) &
        (markets['n_trades'] >= 15) &
        markets['near_round']
    )
    results['COMBO3'] = validate_with_buckets(markets, signal, baseline, 'RLM + near round')

    # Summary
    print("\n" + "=" * 60)
    print("BUCKET VALIDATION RESULTS")
    print("=" * 60)

    real_strategies = []
    proxies = []

    for h_id, r in results.items():
        if r is None:
            print(f"  {h_id}: SKIPPED (insufficient data)")
            continue

        status = "REAL EDGE" if r['IS_REAL'] else "PRICE PROXY"
        print(f"\n  {h_id}: {r['name']}")
        print(f"      Markets: {r['markets']}, Raw edge: {r['raw_edge']:+.1f}%")
        print(f"      Bucket improvement: {r['weighted_improvement']:+.1f}%, Pass rate: {r['bucket_pass_rate']:.0f}%")
        print(f"      --> {status}")

        if r['IS_REAL']:
            real_strategies.append((h_id, r))
        else:
            proxies.append((h_id, r))

    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)

    print(f"\nREAL STRATEGIES ({len(real_strategies)}):")
    for h_id, r in sorted(real_strategies, key=lambda x: -x[1]['weighted_improvement']):
        print(f"  {h_id}: {r['name']}")
        print(f"      Improvement: +{r['weighted_improvement']:.1f}% over baseline")
        print(f"      Buckets: {r['positive_buckets']}/{r['total_buckets']} positive")

    print(f"\nPRICE PROXIES ({len(proxies)}):")
    for h_id, r in proxies:
        print(f"  {h_id}: {r['weighted_improvement']:+.1f}% improvement (fail)")

    # Save
    output = {
        'session': 'LSD-MM-002',
        'timestamp': datetime.now().isoformat(),
        'mode': 'LSD bucket validation',
        'results': results
    }

    with open(REPORT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {REPORT_FILE}")
    return results

if __name__ == "__main__":
    main()
