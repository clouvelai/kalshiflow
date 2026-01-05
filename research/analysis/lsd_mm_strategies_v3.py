#!/usr/bin/env python3
"""
LSD MODE v3: Wild Ideas + Independence Check
=============================================
Test absurd ideas and check which validated strategies are actually independent.
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
REPORT_FILE = BASE_DIR / "reports/lsd_mm_strategies_v3.json"

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

    markets = df.groupby('ticker').agg({
        'count': ['sum', 'std'],
        'yes_price': ['first', 'last', 'mean', 'std', 'min', 'max'],
        'is_yes': ['sum', 'mean'],
        'created_time': ['first', 'last'],
        'result': 'first',
        'hour': 'mean',
        'day_of_week': 'mean'
    })

    # Flatten columns
    markets.columns = ['_'.join(col) for col in markets.columns]
    markets = markets.rename(columns={
        'count_sum': 'n_trades',
        'count_std': 'trade_size_std',
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
        'day_of_week_mean': 'avg_dow'
    })

    # Derived fields
    markets['no_ratio'] = 1 - markets['yes_ratio']
    markets['price_move'] = markets['last_price'] - markets['first_price']
    markets['price_range'] = markets['max_price'] - markets['min_price']
    markets['no_won'] = markets['result'] == 'no'
    markets['duration_hours'] = (markets['last_trade_time'] - markets['first_trade_time']).dt.total_seconds() / 3600
    markets['no_price'] = 100 - markets['avg_price']
    markets['no_bucket'] = (markets['no_price'] // 5) * 5

    # Special flags
    markets['is_weekend'] = markets['avg_dow'] >= 4.5  # Mostly weekend trades
    markets['is_late_night'] = (markets['avg_hour'] >= 22) | (markets['avg_hour'] <= 6)
    markets['is_morning'] = (markets['avg_hour'] >= 6) & (markets['avg_hour'] <= 11)

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

    # Overall stats
    overall_win = subset['no_won'].mean()
    expected_win = subset['base_win_rate'].mean()

    # Is it real?
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
        'IS_REAL': is_real
    }

def main():
    """Test wild ideas and check independence."""
    print("=" * 60)
    print("LSD MODE v3: Wild Ideas + Independence Check")
    print("=" * 60)

    df = load_data()
    markets = get_market_summary(df)
    baseline = calc_baseline_by_bucket(markets)

    results = {}

    # ============================================================
    # PART 1: WILD IDEAS
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 1: WILD IDEAS")
    print("=" * 60)

    # W01: Prime number of trades
    print("\n[W01] Prime number of trades...")
    primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    signal = markets['n_trades'].isin(primes)
    results['W01'] = validate_with_buckets(markets, signal, baseline, 'Prime trade counts')
    if results['W01']:
        print(f"  Prime counts: {results['W01']['weighted_improvement']:+.1f}%, {results['W01']['markets']} markets")

    # W02: Golden ratio price
    print("\n[W02] Golden ratio prices (61.8)...")
    signal = (markets['avg_price'] >= 60) & (markets['avg_price'] <= 64)
    results['W02'] = validate_with_buckets(markets, signal, baseline, 'Golden ratio price zone')
    if results['W02']:
        print(f"  Golden ratio: {results['W02']['weighted_improvement']:+.1f}%, {results['W02']['markets']} markets")

    # W03: Perfect symmetry (price range = 2x std)
    print("\n[W03] Perfect price symmetry...")
    markets['price_symmetry'] = abs(markets['price_range'] - 2 * markets['price_std'].fillna(0))
    signal = markets['price_symmetry'] < 2
    results['W03'] = validate_with_buckets(markets, signal, baseline, 'Perfect price symmetry')
    if results['W03']:
        print(f"  Symmetry: {results['W03']['weighted_improvement']:+.1f}%, {results['W03']['markets']} markets")

    # W04: Exactly opposite flow (50% YES, 50% NO)
    print("\n[W04] Perfectly balanced flow...")
    signal = (markets['yes_ratio'] >= 0.48) & (markets['yes_ratio'] <= 0.52)
    results['W04'] = validate_with_buckets(markets, signal, baseline, 'Balanced flow (48-52%)')
    if results['W04']:
        print(f"  Balanced: {results['W04']['weighted_improvement']:+.1f}%, {results['W04']['markets']} markets")

    # W05: Minnow swarm (many small trades, no whales)
    print("\n[W05] Minnow swarm (all small trades)...")
    max_trades = df.groupby('ticker')['count'].max()
    minnow_markets = max_trades[max_trades < 20].index
    signal = markets.index.isin(minnow_markets)
    results['W05'] = validate_with_buckets(markets, signal, baseline, 'All small trades (<20 contracts)')
    if results['W05']:
        print(f"  Minnow swarm: {results['W05']['weighted_improvement']:+.1f}%, {results['W05']['markets']} markets")

    # W06: Size escalation (trades getting bigger)
    print("\n[W06] Trade size escalation...")
    # Compute first vs last trade size ratio per market
    trade_order = df.sort_values('created_time').groupby('ticker')
    first_size = trade_order['count'].first()
    last_size = trade_order['count'].last()
    escalation = last_size / first_size.replace(0, 1)
    escalating = escalation[escalation > 3].index  # Last trade 3x bigger
    signal = markets.index.isin(escalating)
    results['W06'] = validate_with_buckets(markets, signal, baseline, 'Trade size escalating 3x')
    if results['W06']:
        print(f"  Escalating: {results['W06']['weighted_improvement']:+.1f}%, {results['W06']['markets']} markets")

    # W07: Size de-escalation (trades getting smaller)
    print("\n[W07] Trade size de-escalation...")
    deescalating = escalation[escalation < 0.33].index  # Last trade 3x smaller
    signal = markets.index.isin(deescalating)
    results['W07'] = validate_with_buckets(markets, signal, baseline, 'Trade size de-escalating 3x')
    if results['W07']:
        print(f"  De-escalating: {results['W07']['weighted_improvement']:+.1f}%, {results['W07']['markets']} markets")

    # W08: Morning glory (morning trades only)
    print("\n[W08] Morning trading (6am-11am avg)...")
    signal = markets['is_morning']
    results['W08'] = validate_with_buckets(markets, signal, baseline, 'Morning trades')
    if results['W08']:
        print(f"  Morning: {results['W08']['weighted_improvement']:+.1f}%, {results['W08']['markets']} markets")

    # W09: Weekend warriors
    print("\n[W09] Weekend trading...")
    signal = markets['is_weekend']
    results['W09'] = validate_with_buckets(markets, signal, baseline, 'Weekend trades')
    if results['W09']:
        print(f"  Weekend: {results['W09']['weighted_improvement']:+.1f}%, {results['W09']['markets']} markets")

    # W10: Single whale dominance (one trade > 50% of volume)
    print("\n[W10] Single whale dominance...")
    total_vol = df.groupby('ticker')['count'].sum()
    max_trade = df.groupby('ticker')['count'].max()
    whale_share = max_trade / total_vol
    whale_dominated = whale_share[whale_share > 0.5].index
    signal = markets.index.isin(whale_dominated)
    results['W10'] = validate_with_buckets(markets, signal, baseline, 'Single trade >50% volume')
    if results['W10']:
        print(f"  Whale dominated: {results['W10']['weighted_improvement']:+.1f}%, {results['W10']['markets']} markets")

    # W11: Contrarian signal (YES majority but price UP = smart money on YES)
    print("\n[W11] Smart YES money (YES majority + price up)...")
    signal = (markets['yes_ratio'] > 0.6) & (markets['price_move'] > 5)
    results['W11'] = validate_with_buckets(markets, signal, baseline, 'YES flow + price UP')
    if results['W11']:
        print(f"  Smart YES: {results['W11']['weighted_improvement']:+.1f}%, {results['W11']['markets']} markets")
        # This should bet YES not NO!

    # W12: Extreme price volatility
    print("\n[W12] Extreme price volatility...")
    signal = markets['price_std'] > markets['price_std'].quantile(0.95)
    results['W12'] = validate_with_buckets(markets, signal, baseline, 'Top 5% price volatility')
    if results['W12']:
        print(f"  High vol: {results['W12']['weighted_improvement']:+.1f}%, {results['W12']['markets']} markets")

    # W13: Inverse of RLM (bet YES when NO flow + price rises)
    print("\n[W13] Inverse RLM (bet YES when NO flow + price rises)...")
    signal = (markets['no_ratio'] > 0.65) & (markets['price_move'] > 0) & (markets['n_trades'] >= 15)
    subset = markets[signal].copy()
    if len(subset) >= 50:
        yes_win = (~subset['no_won']).mean()
        breakeven = subset['avg_price'].mean() / 100
        edge = (yes_win - breakeven) * 100
        print(f"  Inverse RLM (bet YES): {edge:+.1f}% edge, {len(subset)} markets")
        results['W13'] = {'name': 'Inverse RLM (bet YES)', 'markets': len(subset), 'raw_edge': round(edge, 2)}
    else:
        results['W13'] = None

    # ============================================================
    # PART 2: INDEPENDENCE CHECK
    # ============================================================
    print("\n" + "=" * 60)
    print("PART 2: INDEPENDENCE CHECK")
    print("=" * 60)

    # Define the "best" strategies from v2
    strategies = {
        'RLM': (markets['yes_ratio'] > 0.65) & (markets['price_move'] < 0) & (markets['n_trades'] >= 15),
        'PRICE_DROP_10': markets['price_move'] < -10,
        'PRICE_MOMENTUM': markets['price_move'] < 0,
        'TOXIC_FLOW': (markets['yes_ratio'] > 0.7) & (markets['price_move'] < -5),
        'COMBO2': (markets['price_move'] < -5) & (markets['no_ratio'] > 0.5),
    }

    # Calculate overlap between strategies
    print("\nOverlap Matrix (% shared markets):")
    print("-" * 60)
    strat_names = list(strategies.keys())
    overlap_matrix = pd.DataFrame(index=strat_names, columns=strat_names)

    for s1 in strat_names:
        for s2 in strat_names:
            shared = (strategies[s1] & strategies[s2]).sum()
            s1_total = strategies[s1].sum()
            if s1_total > 0:
                overlap_matrix.loc[s1, s2] = round(shared / s1_total * 100, 1)
            else:
                overlap_matrix.loc[s1, s2] = 0

    print(overlap_matrix)

    # Find truly independent signals
    print("\n\nSearching for INDEPENDENT signals...")

    # Non-price-movement signals
    print("\n[IND1] NO ratio > 0.7 WITHOUT price drop...")
    signal = (markets['no_ratio'] > 0.7) & (markets['price_move'] >= 0)
    results['IND1'] = validate_with_buckets(markets, signal, baseline, 'NO flow but price stable')
    if results['IND1']:
        print(f"  NO flow stable: {results['IND1']['weighted_improvement']:+.1f}%, {results['IND1']['markets']} markets")

    print("\n[IND2] High trade count + balanced flow...")
    signal = (markets['n_trades'] >= 50) & (markets['yes_ratio'].between(0.4, 0.6))
    results['IND2'] = validate_with_buckets(markets, signal, baseline, 'High volume + balanced')
    if results['IND2']:
        print(f"  High vol balanced: {results['IND2']['weighted_improvement']:+.1f}%, {results['IND2']['markets']} markets")

    print("\n[IND3] Size consistency (low trade size std)...")
    bot_like = markets['trade_size_std'] < markets['trade_size_std'].quantile(0.2)
    signal = bot_like & (markets['n_trades'] >= 10)
    results['IND3'] = validate_with_buckets(markets, signal, baseline, 'Consistent trade sizes')
    if results['IND3']:
        print(f"  Bot-like sizing: {results['IND3']['weighted_improvement']:+.1f}%, {results['IND3']['markets']} markets")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("WILD IDEAS SUMMARY")
    print("=" * 60)

    wild_winners = []
    for h_id, r in results.items():
        if r and 'IS_REAL' in r and r['IS_REAL']:
            wild_winners.append((h_id, r))
        elif r and 'weighted_improvement' in r and r['weighted_improvement'] > 2:
            wild_winners.append((h_id, r))

    if wild_winners:
        print("\nPotentially interesting signals:")
        for h_id, r in sorted(wild_winners, key=lambda x: -x[1].get('weighted_improvement', 0)):
            print(f"  {h_id}: {r['name']}")
            print(f"      Markets: {r['markets']}, Improvement: {r.get('weighted_improvement', 'N/A'):+.1f}%")
    else:
        print("\nNo wild ideas passed validation threshold.")

    print("\n\nKEY INSIGHT:")
    print("-" * 60)
    print("All validated strategies from v2 are variations of the same theme:")
    print("  --> PRICE MOVED TOWARD NO + some flow filter")
    print("\nThe core signal is: price_move < 0 (or < -5 for stronger)")
    print("Everything else (YES ratio, trade count) is just refinement.")

    # Save
    output = {
        'session': 'LSD-MM-003',
        'timestamp': datetime.now().isoformat(),
        'mode': 'LSD wild ideas + independence',
        'overlap_matrix': overlap_matrix.to_dict(),
        'results': {k: v for k, v in results.items() if v is not None}
    }

    with open(REPORT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {REPORT_FILE}")
    return results

if __name__ == "__main__":
    main()
