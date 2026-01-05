#!/usr/bin/env python3
"""
LSD MODE v5: Final Analysis - RLM Independence + Additive Signals
==================================================================
1. Check if RLM adds edge BEYOND pure price movement
2. Find signals that ADD edge when combined with price movement
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
REPORT_FILE = BASE_DIR / "reports/lsd_mm_strategies_v5.json"

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
    """Aggregate trades to market level."""
    print("Aggregating to market level...")

    markets = df.groupby('ticker').agg({
        'count': ['sum', 'std', 'max', 'min'],
        'yes_price': ['first', 'last', 'mean', 'std', 'min', 'max'],
        'is_yes': ['sum', 'mean'],
        'created_time': ['first', 'last'],
        'result': 'first',
        'hour': ['mean', 'std'],
        'leverage_ratio': ['mean', 'std']
    })

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
        'leverage_ratio_mean': 'avg_leverage',
        'leverage_ratio_std': 'leverage_std'
    })

    markets['n_trades'] = df.groupby('ticker').size()
    markets['no_ratio'] = 1 - markets['yes_ratio']
    markets['price_move'] = markets['last_price'] - markets['first_price']
    markets['price_range'] = markets['max_price'] - markets['min_price']
    markets['no_won'] = markets['result'] == 'no'
    markets['duration_hours'] = (markets['last_trade_time'] - markets['first_trade_time']).dt.total_seconds() / 3600
    markets['no_price'] = 100 - markets['avg_price']
    markets['no_bucket'] = (markets['no_price'] // 5) * 5
    markets['trade_size_ratio'] = markets['max_trade_size'] / markets['min_trade_size'].replace(0, 1)
    markets['avg_trade_size'] = markets['total_contracts'] / markets['n_trades']

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

    subset = subset.merge(baseline, left_on='no_bucket', right_index=True, how='left')

    bucket_stats = subset.groupby('no_bucket').agg({
        'no_won': ['mean', 'count'],
        'base_win_rate': 'first'
    })
    bucket_stats.columns = ['signal_win', 'count', 'base_win']
    bucket_stats['improvement'] = bucket_stats['signal_win'] - bucket_stats['base_win']
    bucket_stats = bucket_stats[bucket_stats['count'] >= 5]

    if len(bucket_stats) < 3:
        return None

    weights = bucket_stats['count'] / bucket_stats['count'].sum()
    weighted_improvement = (bucket_stats['improvement'] * weights).sum()

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
    """Final analysis: RLM independence + additive signals."""
    print("=" * 60)
    print("LSD MODE v5: RLM Independence + Additive Signals")
    print("=" * 60)

    df = load_data()
    markets = get_market_summary(df)
    baseline = calc_baseline_by_bucket(markets)

    results = {}

    # ============================================================
    # PART 1: RLM INDEPENDENCE TEST
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1: Does RLM add edge BEYOND price movement?")
    print("=" * 60)

    # Base signal: price dropped
    price_dropped = markets['price_move'] < 0

    # RLM adds: yes_ratio > 0.65 + n_trades >= 15
    rlm_filter = (markets['yes_ratio'] > 0.65) & (markets['n_trades'] >= 15)

    print("\n[BASE] Pure price drop (price_move < 0)...")
    results['BASE'] = validate_with_buckets(markets, price_dropped, baseline, 'Price dropped only')
    print(f"  Result: {results['BASE']['weighted_improvement']:+.1f}%, {results['BASE']['markets']} mkts")

    print("\n[RLM] Price drop + YES>65% + 15+ trades...")
    rlm_signal = price_dropped & rlm_filter
    results['RLM'] = validate_with_buckets(markets, rlm_signal, baseline, 'RLM full signal')
    print(f"  Result: {results['RLM']['weighted_improvement']:+.1f}%, {results['RLM']['markets']} mkts")

    print("\n[INVERSE] Price drop + YES<65% or <15 trades (not RLM)...")
    not_rlm = price_dropped & ~rlm_filter
    results['NOT_RLM'] = validate_with_buckets(markets, not_rlm, baseline, 'Price drop but NOT RLM')
    print(f"  Result: {results['NOT_RLM']['weighted_improvement']:+.1f}%, {results['NOT_RLM']['markets']} mkts")

    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    base_edge = results['BASE']['weighted_improvement']
    rlm_edge = results['RLM']['weighted_improvement']
    not_rlm_edge = results['NOT_RLM']['weighted_improvement']
    rlm_delta = rlm_edge - base_edge
    print(f"  Base (price drop only): {base_edge:+.1f}%")
    print(f"  RLM (price drop + filters): {rlm_edge:+.1f}%")
    print(f"  Not RLM (price drop without filters): {not_rlm_edge:+.1f}%")
    print(f"  RLM ADDITIVE EDGE: {rlm_delta:+.1f}%")

    if rlm_delta > 0.5:
        print(f"\n  --> YES, RLM filters ADD {rlm_delta:.1f}% edge!")
    else:
        print(f"\n  --> NO, RLM filters only add {rlm_delta:.1f}% (marginal)")

    # ============================================================
    # PART 2: FIND ADDITIVE SIGNALS
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: Find signals that ADD edge to price drop")
    print("=" * 60)

    # Test various filters on top of price_dropped
    additive_tests = [
        ('A01', 'High YES ratio (>70%)', (markets['yes_ratio'] > 0.70)),
        ('A02', 'Super high YES ratio (>80%)', (markets['yes_ratio'] > 0.80)),
        ('A03', 'Many trades (>30)', (markets['n_trades'] >= 30)),
        ('A04', 'Many trades (>50)', (markets['n_trades'] >= 50)),
        ('A05', 'Big price drop (>5c)', (markets['price_move'] < -5)),
        ('A06', 'Huge price drop (>10c)', (markets['price_move'] < -10)),
        ('A07', 'Long duration (>24hr)', (markets['duration_hours'] > 24)),
        ('A08', 'Low leverage std (<0.5)', (markets['leverage_std'] < 0.5)),
        ('A09', 'High trade count + YES flow', (markets['n_trades'] >= 30) & (markets['yes_ratio'] > 0.65)),
        ('A10', 'Spread hours (std > 6)', (markets['hour_std'] > 6)),
        ('A11', 'Whale present (max trade > 100)', (markets['max_trade_size'] > 100)),
        ('A12', 'No whale (max trade <= 50)', (markets['max_trade_size'] <= 50)),
        ('A13', 'High volume (top 25%)', (markets['total_contracts'] > markets['total_contracts'].quantile(0.75))),
        ('A14', 'Low size std (uniform)', (markets['trade_size_std'] < markets['trade_size_std'].quantile(0.25))),
    ]

    print("\nTesting additive filters on top of price_move < 0:")
    print("-" * 60)

    additive_results = []

    for aid, name, filter_cond in additive_tests:
        combined_signal = price_dropped & filter_cond
        result = validate_with_buckets(markets, combined_signal, baseline, name)
        if result:
            delta = result['weighted_improvement'] - base_edge
            result['additive_edge'] = round(delta, 2)
            results[aid] = result
            additive_results.append((aid, name, result))
            print(f"  {aid}: {name}")
            print(f"      Improvement: {result['weighted_improvement']:+.1f}%, Additive: {delta:+.1f}%, Markets: {result['markets']}")

    # Sort by additive edge
    additive_results.sort(key=lambda x: -x[2]['additive_edge'])

    print("\n" + "=" * 60)
    print("TOP ADDITIVE SIGNALS (sorted by extra edge)")
    print("=" * 60)

    for aid, name, r in additive_results[:10]:
        print(f"  {aid}: {name}")
        print(f"      Base: {base_edge:+.1f}% + Extra: {r['additive_edge']:+.1f}% = Total: {r['weighted_improvement']:+.1f}%")
        print(f"      Markets: {r['markets']}, Buckets: {r['pass_rate']:.0f}%")

    # ============================================================
    # PART 3: OPTIMAL COMBINATION
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 3: Optimal Signal Combination")
    print("=" * 60)

    # Try combining top additive signals
    print("\n[OPT1] Price drop + big drop + YES flow...")
    opt1 = (markets['price_move'] < -5) & (markets['yes_ratio'] > 0.65) & (markets['n_trades'] >= 15)
    results['OPT1'] = validate_with_buckets(markets, opt1, baseline, 'Big drop + RLM filters')
    if results['OPT1']:
        print(f"  {results['OPT1']['weighted_improvement']:+.1f}%, {results['OPT1']['markets']} mkts")

    print("\n[OPT2] Huge drop (>10c) + high YES (>70%)...")
    opt2 = (markets['price_move'] < -10) & (markets['yes_ratio'] > 0.70)
    results['OPT2'] = validate_with_buckets(markets, opt2, baseline, 'Huge drop + high YES')
    if results['OPT2']:
        print(f"  {results['OPT2']['weighted_improvement']:+.1f}%, {results['OPT2']['markets']} mkts")

    print("\n[OPT3] Price drop + long duration + YES flow...")
    opt3 = (markets['price_move'] < 0) & (markets['duration_hours'] > 24) & (markets['yes_ratio'] > 0.65)
    results['OPT3'] = validate_with_buckets(markets, opt3, baseline, 'Drop + long + YES')
    if results['OPT3']:
        print(f"  {results['OPT3']['weighted_improvement']:+.1f}%, {results['OPT3']['markets']} mkts")

    print("\n[OPT4] Big drop + many trades + YES flow...")
    opt4 = (markets['price_move'] < -5) & (markets['n_trades'] >= 30) & (markets['yes_ratio'] > 0.65)
    results['OPT4'] = validate_with_buckets(markets, opt4, baseline, 'Big drop + many trades + YES')
    if results['OPT4']:
        print(f"  {results['OPT4']['weighted_improvement']:+.1f}%, {results['OPT4']['markets']} mkts")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\n1. CORE SIGNAL: Price moved toward NO (price_move < 0)")
    print(f"   Base improvement: {base_edge:+.1f}%")

    print("\n2. RLM FILTERS:")
    print(f"   - YES ratio > 65%: adds ~{rlm_delta:.1f}% edge")
    print(f"   - n_trades >= 15: filtering for significance")

    print("\n3. TOP ADDITIVE SIGNALS:")
    for aid, name, r in additive_results[:5]:
        if r['additive_edge'] > 0:
            print(f"   - {name}: +{r['additive_edge']:.1f}%")

    print("\n4. BEST TOTAL STRATEGIES:")
    opt_strategies = [
        (k, v) for k, v in results.items()
        if k.startswith('OPT') and v and v.get('weighted_improvement', 0) > 0
    ]
    for k, v in sorted(opt_strategies, key=lambda x: -x[1]['weighted_improvement']):
        print(f"   {k}: {v['name']}")
        print(f"       Improvement: {v['weighted_improvement']:+.1f}%, Markets: {v['markets']}")

    # Save
    output = {
        'session': 'LSD-MM-005',
        'timestamp': datetime.now().isoformat(),
        'mode': 'LSD RLM independence + additive',
        'base_edge': base_edge,
        'rlm_edge': rlm_edge,
        'rlm_additive': rlm_delta,
        'results': {k: v for k, v in results.items() if v}
    }

    with open(REPORT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {REPORT_FILE}")
    return results

if __name__ == "__main__":
    main()
