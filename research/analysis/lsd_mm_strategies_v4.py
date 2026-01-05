#!/usr/bin/env python3
"""
LSD MODE v4: Non-Price-Movement Signals Only
=============================================
All previous winners use price_move as core signal.
Test ONLY signals that don't involve price movement.
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
REPORT_FILE = BASE_DIR / "reports/lsd_mm_strategies_v4.json"

def load_data():
    """Load trade data."""
    print("Loading trade data...")
    df = pd.read_csv(TRADES_FILE, parse_dates=['datetime'], low_memory=False)
    df = df.rename(columns={'market_ticker': 'ticker', 'datetime': 'created_time'})
    df['is_yes'] = df['taker_side'] == 'yes'
    df['hour'] = df['created_time'].dt.hour
    df['day_of_week'] = df['created_time'].dt.dayofweek
    print(f"Loaded {len(df):,} trades")
    return df

def get_market_summary(df):
    """Aggregate trades to market level with extra fields."""
    print("Aggregating to market level...")

    # First pass aggregation
    markets = df.groupby('ticker').agg({
        'count': ['sum', 'std', 'max', 'min'],
        'yes_price': ['first', 'last', 'mean', 'std', 'min', 'max'],
        'is_yes': ['sum', 'mean'],
        'created_time': ['first', 'last'],
        'result': 'first',
        'hour': ['mean', 'std'],
        'day_of_week': 'mean',
        'leverage_ratio': ['mean', 'std']
    })

    # Flatten columns
    markets.columns = ['_'.join(col) for col in markets.columns]
    markets = markets.rename(columns={
        'count_sum': 'total_contracts',
        'count_std': 'trade_size_std',
        'count_max': 'max_trade_size',
        'count_min': 'min_trade_size',
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
        'result_first': 'result',
        'hour_mean': 'avg_hour',
        'hour_std': 'hour_std',
        'day_of_week_mean': 'avg_dow',
        'leverage_ratio_mean': 'avg_leverage',
        'leverage_ratio_std': 'leverage_std'
    })

    # Count n_trades
    markets['n_trades'] = df.groupby('ticker').size()

    # Derived fields
    markets['no_ratio'] = 1 - markets['yes_ratio']
    markets['price_move'] = markets['last_price'] - markets['first_price']
    markets['price_range'] = markets['max_price'] - markets['min_price']
    markets['no_won'] = markets['result'] == 'no'
    markets['duration_hours'] = (markets['last_trade_time'] - markets['first_trade_time']).dt.total_seconds() / 3600
    markets['no_price'] = 100 - markets['avg_price']
    markets['no_bucket'] = (markets['no_price'] // 5) * 5

    # Derived metrics
    markets['trade_size_ratio'] = markets['max_trade_size'] / markets['min_trade_size'].replace(0, 1)
    markets['avg_trade_size'] = markets['total_contracts'] / markets['n_trades']

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

def validate_with_buckets(markets, signal_mask, baseline, name, min_markets=50):
    """Validate strategy with bucket-by-bucket comparison."""
    subset = markets[signal_mask].copy()
    if len(subset) < min_markets:
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
    bucket_stats = bucket_stats[bucket_stats['count'] >= 5]

    if len(bucket_stats) < 3:
        return None

    # Calculate weighted improvement
    weights = bucket_stats['count'] / bucket_stats['count'].sum()
    weighted_improvement = (bucket_stats['improvement'] * weights).sum()

    # How many buckets positive?
    positive_buckets = (bucket_stats['improvement'] > 0).sum()
    total_buckets = len(bucket_stats)

    is_real = (
        weighted_improvement > 0.02 and
        positive_buckets / total_buckets > 0.5
    )

    return {
        'name': name,
        'markets': len(subset),
        'weighted_improvement': round(weighted_improvement * 100, 2),
        'positive_buckets': positive_buckets,
        'total_buckets': total_buckets,
        'pass_rate': round(positive_buckets / total_buckets * 100, 1),
        'IS_REAL': is_real
    }

def main():
    """Test non-price-movement signals."""
    print("=" * 60)
    print("LSD MODE v4: Non-Price-Movement Signals")
    print("=" * 60)

    df = load_data()
    markets = get_market_summary(df)
    baseline = calc_baseline_by_bucket(markets)

    results = {}

    print("\n" + "-" * 60)
    print("Testing signals that DON'T use price movement")
    print("-" * 60)

    # CATEGORY 1: Flow-only signals
    print("\n=== FLOW-ONLY SIGNALS ===")

    print("\n[F01] Extreme NO flow (>80%)...")
    signal = markets['no_ratio'] > 0.8
    results['F01'] = validate_with_buckets(markets, signal, baseline, 'NO ratio >80%')
    if results['F01']:
        print(f"  {results['F01']['weighted_improvement']:+.1f}%, {results['F01']['markets']} mkts, {results['F01']['pass_rate']:.0f}% buckets")

    print("\n[F02] Extreme YES flow (>80%, bet NO)...")
    signal = markets['yes_ratio'] > 0.8
    results['F02'] = validate_with_buckets(markets, signal, baseline, 'YES ratio >80% bet NO')
    if results['F02']:
        print(f"  {results['F02']['weighted_improvement']:+.1f}%, {results['F02']['markets']} mkts, {results['F02']['pass_rate']:.0f}% buckets")

    print("\n[F03] Super extreme YES flow (>90%, bet NO)...")
    signal = markets['yes_ratio'] > 0.9
    results['F03'] = validate_with_buckets(markets, signal, baseline, 'YES ratio >90% bet NO')
    if results['F03']:
        print(f"  {results['F03']['weighted_improvement']:+.1f}%, {results['F03']['markets']} mkts, {results['F03']['pass_rate']:.0f}% buckets")

    # CATEGORY 2: Volume/size-only signals
    print("\n=== VOLUME/SIZE SIGNALS ===")

    print("\n[V01] High volume markets (top 10%)...")
    high_vol = markets['total_contracts'] > markets['total_contracts'].quantile(0.9)
    results['V01'] = validate_with_buckets(markets, high_vol, baseline, 'Top 10% volume')
    if results['V01']:
        print(f"  {results['V01']['weighted_improvement']:+.1f}%, {results['V01']['markets']} mkts, {results['V01']['pass_rate']:.0f}% buckets")

    print("\n[V02] Low volume markets (bottom 10%)...")
    low_vol = markets['total_contracts'] < markets['total_contracts'].quantile(0.1)
    results['V02'] = validate_with_buckets(markets, low_vol, baseline, 'Bottom 10% volume')
    if results['V02']:
        print(f"  {results['V02']['weighted_improvement']:+.1f}%, {results['V02']['markets']} mkts, {results['V02']['pass_rate']:.0f}% buckets")

    print("\n[V03] Many trades (>100)...")
    signal = markets['n_trades'] > 100
    results['V03'] = validate_with_buckets(markets, signal, baseline, '>100 trades')
    if results['V03']:
        print(f"  {results['V03']['weighted_improvement']:+.1f}%, {results['V03']['markets']} mkts, {results['V03']['pass_rate']:.0f}% buckets")

    print("\n[V04] Large avg trade size (>50 contracts)...")
    signal = markets['avg_trade_size'] > 50
    results['V04'] = validate_with_buckets(markets, signal, baseline, 'Avg trade >50 contracts')
    if results['V04']:
        print(f"  {results['V04']['weighted_improvement']:+.1f}%, {results['V04']['markets']} mkts, {results['V04']['pass_rate']:.0f}% buckets")

    # CATEGORY 3: Trade size distribution
    print("\n=== TRADE SIZE DISTRIBUTION ===")

    print("\n[S01] Uniform sizes (low std)...")
    signal = markets['trade_size_std'] < markets['trade_size_std'].quantile(0.1)
    results['S01'] = validate_with_buckets(markets, signal, baseline, 'Low trade size std')
    if results['S01']:
        print(f"  {results['S01']['weighted_improvement']:+.1f}%, {results['S01']['markets']} mkts, {results['S01']['pass_rate']:.0f}% buckets")

    print("\n[S02] Wild size variation (high std)...")
    signal = markets['trade_size_std'] > markets['trade_size_std'].quantile(0.9)
    results['S02'] = validate_with_buckets(markets, signal, baseline, 'High trade size std')
    if results['S02']:
        print(f"  {results['S02']['weighted_improvement']:+.1f}%, {results['S02']['markets']} mkts, {results['S02']['pass_rate']:.0f}% buckets")

    print("\n[S03] Max/min size ratio >10 (whale + minnows)...")
    signal = markets['trade_size_ratio'] > 10
    results['S03'] = validate_with_buckets(markets, signal, baseline, 'Max/min size ratio >10')
    if results['S03']:
        print(f"  {results['S03']['weighted_improvement']:+.1f}%, {results['S03']['markets']} mkts, {results['S03']['pass_rate']:.0f}% buckets")

    # CATEGORY 4: Time-based
    print("\n=== TIME-BASED SIGNALS ===")

    print("\n[T01] Trading spread over many hours (std > 6)...")
    signal = markets['hour_std'] > 6
    results['T01'] = validate_with_buckets(markets, signal, baseline, 'Hour std > 6')
    if results['T01']:
        print(f"  {results['T01']['weighted_improvement']:+.1f}%, {results['T01']['markets']} mkts, {results['T01']['pass_rate']:.0f}% buckets")

    print("\n[T02] Concentrated in short time (std < 1)...")
    signal = (markets['hour_std'] < 1) & (markets['n_trades'] >= 10)
    results['T02'] = validate_with_buckets(markets, signal, baseline, 'Hour std < 1')
    if results['T02']:
        print(f"  {results['T02']['weighted_improvement']:+.1f}%, {results['T02']['markets']} mkts, {results['T02']['pass_rate']:.0f}% buckets")

    print("\n[T03] Quick markets (<1 hour duration)...")
    signal = (markets['duration_hours'] < 1) & (markets['n_trades'] >= 10)
    results['T03'] = validate_with_buckets(markets, signal, baseline, 'Duration < 1hr')
    if results['T03']:
        print(f"  {results['T03']['weighted_improvement']:+.1f}%, {results['T03']['markets']} mkts, {results['T03']['pass_rate']:.0f}% buckets")

    print("\n[T04] Long markets (>24 hour duration)...")
    signal = markets['duration_hours'] > 24
    results['T04'] = validate_with_buckets(markets, signal, baseline, 'Duration > 24hr')
    if results['T04']:
        print(f"  {results['T04']['weighted_improvement']:+.1f}%, {results['T04']['markets']} mkts, {results['T04']['pass_rate']:.0f}% buckets")

    # CATEGORY 5: Leverage-based
    print("\n=== LEVERAGE SIGNALS ===")

    print("\n[L01] Low leverage std (consistent)...")
    signal = markets['leverage_std'] < 0.5
    results['L01'] = validate_with_buckets(markets, signal, baseline, 'Leverage std < 0.5')
    if results['L01']:
        print(f"  {results['L01']['weighted_improvement']:+.1f}%, {results['L01']['markets']} mkts, {results['L01']['pass_rate']:.0f}% buckets")

    print("\n[L02] High leverage std (volatile)...")
    signal = markets['leverage_std'] > 2.0
    results['L02'] = validate_with_buckets(markets, signal, baseline, 'Leverage std > 2.0')
    if results['L02']:
        print(f"  {results['L02']['weighted_improvement']:+.1f}%, {results['L02']['markets']} mkts, {results['L02']['pass_rate']:.0f}% buckets")

    # CATEGORY 6: Combination non-price signals
    print("\n=== COMBINATION SIGNALS (no price movement) ===")

    print("\n[C01] YES flow >70% + many trades (>30)...")
    signal = (markets['yes_ratio'] > 0.7) & (markets['n_trades'] >= 30)
    results['C01'] = validate_with_buckets(markets, signal, baseline, 'YES >70% + 30+ trades')
    if results['C01']:
        print(f"  {results['C01']['weighted_improvement']:+.1f}%, {results['C01']['markets']} mkts, {results['C01']['pass_rate']:.0f}% buckets")

    print("\n[C02] YES flow >80% + high volume...")
    signal = (markets['yes_ratio'] > 0.8) & (markets['total_contracts'] > 500)
    results['C02'] = validate_with_buckets(markets, signal, baseline, 'YES >80% + 500+ contracts')
    if results['C02']:
        print(f"  {results['C02']['weighted_improvement']:+.1f}%, {results['C02']['markets']} mkts, {results['C02']['pass_rate']:.0f}% buckets")

    print("\n[C03] YES flow >70% + uniform sizes (bot-like YES)...")
    signal = (markets['yes_ratio'] > 0.7) & (markets['trade_size_std'] < markets['trade_size_std'].quantile(0.2))
    results['C03'] = validate_with_buckets(markets, signal, baseline, 'YES >70% + low size std')
    if results['C03']:
        print(f"  {results['C03']['weighted_improvement']:+.1f}%, {results['C03']['markets']} mkts, {results['C03']['pass_rate']:.0f}% buckets")

    print("\n[C04] Many trades + high size ratio (whale among minnows)...")
    signal = (markets['n_trades'] >= 20) & (markets['trade_size_ratio'] > 20)
    results['C04'] = validate_with_buckets(markets, signal, baseline, '20+ trades + big size ratio')
    if results['C04']:
        print(f"  {results['C04']['weighted_improvement']:+.1f}%, {results['C04']['markets']} mkts, {results['C04']['pass_rate']:.0f}% buckets")

    print("\n[C05] Short duration + high trade count (burst)...")
    signal = (markets['duration_hours'] < 0.5) & (markets['n_trades'] >= 30)
    results['C05'] = validate_with_buckets(markets, signal, baseline, '<30min + 30+ trades')
    if results['C05']:
        print(f"  {results['C05']['weighted_improvement']:+.1f}%, {results['C05']['markets']} mkts, {results['C05']['pass_rate']:.0f}% buckets")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: Non-Price-Movement Signals")
    print("=" * 60)

    passed = []
    failed = []

    for h_id, r in results.items():
        if r is None:
            continue
        if r['IS_REAL']:
            passed.append((h_id, r))
        else:
            failed.append((h_id, r))

    print(f"\nPASSED ({len(passed)}):")
    if passed:
        for h_id, r in sorted(passed, key=lambda x: -x[1]['weighted_improvement']):
            print(f"  {h_id}: {r['name']}")
            print(f"      Improvement: {r['weighted_improvement']:+.1f}%, Markets: {r['markets']}, Buckets: {r['pass_rate']:.0f}%")
    else:
        print("  None passed!")

    print(f"\nBest FAILED (top 5):")
    for h_id, r in sorted(failed, key=lambda x: -x[1]['weighted_improvement'])[:5]:
        print(f"  {h_id}: {r['name']} -> {r['weighted_improvement']:+.1f}% ({r['pass_rate']:.0f}% buckets)")

    # Save
    output = {
        'session': 'LSD-MM-004',
        'timestamp': datetime.now().isoformat(),
        'mode': 'LSD non-price signals',
        'total_signals_tested': len([r for r in results.values() if r]),
        'signals_passed': len(passed),
        'results': {k: v for k, v in results.items() if v}
    }

    with open(REPORT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {REPORT_FILE}")
    return results

if __name__ == "__main__":
    main()
