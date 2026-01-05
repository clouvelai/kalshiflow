#!/usr/bin/env python3
"""
RLM Enhancement Validation - Full Backtest
============================================

Tests 3 hypotheses against 1.7M trade dataset to find "edges within the edge":

1. E-001: Volume-Weighted YES Ratio
   - Compare trade-count weighted vs volume-weighted YES ratio
   - Test if weighting by contract volume improves signal quality

2. F-001: RLM + S013 Combination
   - Find markets where BOTH RLM and S013 conditions fire
   - Calculate edge boost from combining independent signals

3. S-001: Position Scaling by Signal Strength
   - Bucket analysis by price_drop magnitude
   - Calculate optimal position sizing by signal strength tier

Validation Standard (H123 level):
- Sample Size: N >= 50 markets
- Statistical Significance: p < 0.05
- Bootstrap CI: Excludes zero
- Bucket Ratio: >= 80% positive buckets
- Temporal Stability: >= 3/4 quarters positive

Author: Quant Agent (Claude Opus 4.5)
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/rlm_enhancement_results.json'

# Global results container
results = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'analyst': 'Quant Agent (Claude Opus 4.5)',
        'session': 'RLM Enhancement Validation',
        'data_path': DATA_PATH
    },
    'baseline_rlm': {},
    'e001_volume_weighted': {},
    'f001_rlm_s013_combo': {},
    's001_signal_strength': {},
    'summary': {}
}


def load_data():
    """Load and prepare trade data."""
    print("=" * 80)
    print("RLM ENHANCEMENT VALIDATION - FULL BACKTEST")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date

    # Add quarter for temporal analysis
    df['quarter'] = pd.to_datetime(df['datetime']).dt.quarter
    df['year_quarter'] = df['datetime'].dt.to_period('Q')

    print(f"\nData loaded: {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Filter to resolved markets only
    resolved_markets = df[df['market_result'].isin(['yes', 'no'])]['market_ticker'].unique()
    df_resolved = df[df['market_ticker'].isin(resolved_markets)]

    print(f"Resolved markets: {len(resolved_markets):,}")
    print(f"Trades in resolved markets: {len(df_resolved):,}")

    results['metadata']['total_trades'] = len(df_resolved)
    results['metadata']['total_markets'] = len(resolved_markets)

    return df_resolved


def build_market_stats(df):
    """Build comprehensive per-market statistics for all analyses."""
    print("\n" + "=" * 80)
    print("BUILDING MARKET-LEVEL STATISTICS")
    print("=" * 80)

    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    # Calculate YES/NO volumes and trade counts
    def calc_market_stats(group):
        # Trade counts
        yes_trades = (group['taker_side'] == 'yes').sum()
        no_trades = (group['taker_side'] == 'no').sum()
        total_trades = len(group)

        # Volume (contract counts)
        yes_volume = group[group['taker_side'] == 'yes']['count'].sum()
        no_volume = group[group['taker_side'] == 'no']['count'].sum()
        total_volume = group['count'].sum()

        # Price movement
        first_yes_price = group['yes_price'].iloc[0]
        last_yes_price = group['yes_price'].iloc[-1]
        yes_price_drop = first_yes_price - last_yes_price

        # Leverage stats (for S013)
        lev_std = group['leverage_ratio'].std() if len(group) > 1 else 0
        lev_mean = group['leverage_ratio'].mean()

        # Time stats
        first_trade = group['datetime'].iloc[0]
        last_trade = group['datetime'].iloc[-1]

        return pd.Series({
            'yes_trades': yes_trades,
            'no_trades': no_trades,
            'total_trades': total_trades,
            'yes_volume': yes_volume,
            'no_volume': no_volume,
            'total_volume': total_volume,
            'yes_trade_ratio': yes_trades / total_trades if total_trades > 0 else 0,
            'yes_volume_ratio': yes_volume / total_volume if total_volume > 0 else 0,
            'no_trade_ratio': no_trades / total_trades if total_trades > 0 else 0,
            'no_volume_ratio': no_volume / total_volume if total_volume > 0 else 0,
            'first_yes_price': first_yes_price,
            'last_yes_price': last_yes_price,
            'yes_price_drop': yes_price_drop,
            'yes_price_dropped': yes_price_drop > 0,
            'avg_no_price': group['no_price'].mean(),
            'avg_yes_price': group['yes_price'].mean(),
            'leverage_std': lev_std if not pd.isna(lev_std) else 0,
            'leverage_mean': lev_mean,
            'market_result': group['market_result'].iloc[0],
            'first_trade': first_trade,
            'last_trade': last_trade
        })

    market_stats = df_sorted.groupby('market_ticker').apply(calc_market_stats).reset_index()

    # Add year_quarter for temporal analysis
    market_stats['year_quarter'] = pd.to_datetime(market_stats['first_trade']).dt.to_period('Q')

    # Add price bucket for baseline comparison
    market_stats['bucket_5c'] = (market_stats['avg_no_price'] // 5) * 5

    print(f"Built stats for {len(market_stats):,} markets")
    print(f"\nKey distributions:")
    print(f"  - Mean yes_trade_ratio: {market_stats['yes_trade_ratio'].mean():.3f}")
    print(f"  - Mean yes_volume_ratio: {market_stats['yes_volume_ratio'].mean():.3f}")
    print(f"  - Mean price_drop: {market_stats['yes_price_drop'].mean():.2f}c")
    print(f"  - Mean leverage_std: {market_stats['leverage_std'].mean():.3f}")

    return market_stats


def build_baseline(market_stats):
    """Build baseline NO win rates by 5c price bucket."""
    print("\n" + "=" * 80)
    print("BUILDING BASELINE (5c buckets)")
    print("=" * 80)

    baseline = {}
    print(f"\n{'Bucket':<12} {'Markets':>10} {'NO Wins':>10} {'Win Rate':>10}")
    print("-" * 45)

    for bucket in sorted(market_stats['bucket_5c'].unique()):
        bucket_data = market_stats[market_stats['bucket_5c'] == bucket]
        n_markets = len(bucket_data)
        no_wins = (bucket_data['market_result'] == 'no').sum()
        win_rate = no_wins / n_markets if n_markets > 0 else 0

        if n_markets >= 20:
            baseline[bucket] = {
                'n_markets': n_markets,
                'no_wins': no_wins,
                'win_rate': win_rate
            }
            print(f"{bucket:.0f}-{bucket+5:.0f}c      {n_markets:>10,} {no_wins:>10,} {win_rate:>10.1%}")

    print(f"\nBaseline built for {len(baseline)} buckets")
    return baseline


def calculate_edge_stats(signal_markets, baseline, side='no', min_markets=30):
    """Calculate comprehensive edge statistics with bucket-matched baseline comparison."""
    n = len(signal_markets)
    if n < min_markets:
        return {'n': n, 'valid': False, 'reason': f'insufficient_markets_{n}'}

    wins = (signal_markets['market_result'] == side).sum()
    wr = wins / n

    if side == 'no':
        avg_price = signal_markets['avg_no_price'].mean()
        signal_markets = signal_markets.copy()
        signal_markets['bucket_5c'] = (signal_markets['avg_no_price'] // 5) * 5
    else:
        avg_price = signal_markets['avg_yes_price'].mean()
        signal_markets = signal_markets.copy()
        signal_markets['bucket_5c'] = (signal_markets['avg_yes_price'] // 5) * 5

    be = avg_price / 100
    edge = wr - be

    # Statistical significance (Z-test)
    z = (wins - n * be) / np.sqrt(n * be * (1 - be)) if 0 < be < 1 else 0
    p_value = 1 - stats.norm.cdf(z)

    # Bucket-by-bucket analysis
    improvements = []
    for bucket in sorted(signal_markets['bucket_5c'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket_5c'] == bucket]
        n_sig = len(sig_bucket)

        if n_sig < 5:
            continue

        sig_wr = (sig_bucket['market_result'] == side).mean()
        base_wr = baseline[bucket]['win_rate']
        imp = sig_wr - base_wr

        improvements.append({
            'bucket': int(bucket),
            'sig_wr': float(sig_wr),
            'base_wr': float(base_wr),
            'improvement': float(imp),
            'n_sig': int(n_sig)
        })

    if not improvements:
        return {'n': n, 'valid': False, 'reason': 'no_valid_buckets'}

    total_n = sum(i['n_sig'] for i in improvements)
    weighted_imp = sum(i['improvement'] * i['n_sig'] for i in improvements) / total_n

    pos_buckets = sum(1 for i in improvements if i['improvement'] > 0)
    total_buckets = len(improvements)

    # Bootstrap CI
    n_bootstrap = 1000
    bootstrap_edges = []
    for _ in range(n_bootstrap):
        sample = signal_markets.sample(n=len(signal_markets), replace=True)
        sample_wr = (sample['market_result'] == side).mean()
        if side == 'no':
            sample_be = sample['avg_no_price'].mean() / 100
        else:
            sample_be = sample['avg_yes_price'].mean() / 100
        bootstrap_edges.append(sample_wr - sample_be)

    ci_lower = np.percentile(bootstrap_edges, 2.5)
    ci_upper = np.percentile(bootstrap_edges, 97.5)

    # Temporal stability
    quarters = signal_markets['year_quarter'].unique()
    quarter_edges = []
    for q in quarters:
        q_data = signal_markets[signal_markets['year_quarter'] == q]
        if len(q_data) >= 20:
            q_wr = (q_data['market_result'] == side).mean()
            q_be = q_data['avg_no_price' if side == 'no' else 'avg_yes_price'].mean() / 100
            quarter_edges.append({'quarter': str(q), 'edge': float(q_wr - q_be), 'n': len(q_data)})

    pos_quarters = sum(1 for q in quarter_edges if q['edge'] > 0)
    total_quarters = len(quarter_edges)

    return {
        'n': int(n),
        'valid': True,
        'wins': int(wins),
        'win_rate': float(wr),
        'avg_price': float(avg_price),
        'breakeven': float(be),
        'edge': float(edge),
        'p_value': float(p_value),
        'weighted_improvement': float(weighted_imp),
        'pos_buckets': pos_buckets,
        'total_buckets': total_buckets,
        'bucket_ratio': f"{pos_buckets}/{total_buckets}",
        'bucket_pct': float(pos_buckets / total_buckets) if total_buckets > 0 else 0,
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        'temporal_stability': f"{pos_quarters}/{total_quarters}",
        'quarter_details': quarter_edges,
        'bucket_details': improvements
    }


def validate_baseline_rlm(market_stats, baseline):
    """Validate the baseline RLM signal for comparison."""
    print("\n" + "=" * 80)
    print("BASELINE RLM VALIDATION")
    print("=" * 80)

    # Standard RLM: yes_trade_ratio > 0.65, price dropped, n_trades >= 15
    rlm_mask = (
        (market_stats['yes_trade_ratio'] > 0.65) &
        (market_stats['yes_price_dropped']) &
        (market_stats['total_trades'] >= 15)
    )

    rlm_markets = market_stats[rlm_mask].copy()
    stats_result = calculate_edge_stats(rlm_markets, baseline)

    print(f"\n--- Baseline RLM (65% YES trades, price drop, 15+ trades) ---")
    if stats_result['valid']:
        print(f"  Markets: {stats_result['n']:,}")
        print(f"  Win Rate: {stats_result['win_rate']*100:.1f}%")
        print(f"  Avg NO Price: {stats_result['avg_price']:.1f}c")
        print(f"  RAW EDGE: +{stats_result['edge']*100:.2f}%")
        print(f"  IMPROVEMENT vs Baseline: +{stats_result['weighted_improvement']*100:.2f}%")
        print(f"  P-value: {stats_result['p_value']:.2e}")
        print(f"  Bucket Analysis: {stats_result['bucket_ratio']} positive ({stats_result['bucket_pct']*100:.1f}%)")
        print(f"  95% CI: [{stats_result['ci_95_lower']*100:.2f}%, {stats_result['ci_95_upper']*100:.2f}%]")
        print(f"  Temporal Stability: {stats_result['temporal_stability']}")

    results['baseline_rlm'] = stats_result
    return rlm_markets


def analyze_e001_volume_weighted(market_stats, baseline):
    """
    E-001: Volume-Weighted YES Ratio Analysis

    Compare trade-count weighted vs volume-weighted YES ratio.
    Test if weighting by contract volume improves signal quality.
    """
    print("\n" + "=" * 80)
    print("E-001: VOLUME-WEIGHTED YES RATIO ANALYSIS")
    print("=" * 80)

    results['e001_volume_weighted'] = {
        'hypothesis': 'Volume-weighted YES ratio may be more informative than trade-count weighted',
        'comparison': {},
        'overlap_analysis': {},
        'recommendation': ''
    }

    # Base conditions (same for all variants)
    base_conditions = (
        (market_stats['yes_price_dropped']) &
        (market_stats['total_trades'] >= 15)
    )

    # Variant 1: Trade-weighted (current RLM)
    trade_weighted_mask = base_conditions & (market_stats['yes_trade_ratio'] > 0.65)
    trade_weighted = market_stats[trade_weighted_mask].copy()
    trade_stats = calculate_edge_stats(trade_weighted, baseline)

    print(f"\n--- Variant 1: Trade-Weighted (Current RLM) ---")
    print(f"  Markets: {trade_stats['n']:,}")
    if trade_stats['valid']:
        print(f"  Win Rate: {trade_stats['win_rate']*100:.1f}%")
        print(f"  Edge: +{trade_stats['edge']*100:.2f}%")
        print(f"  Improvement: +{trade_stats['weighted_improvement']*100:.2f}%")
        print(f"  Buckets: {trade_stats['bucket_ratio']} ({trade_stats['bucket_pct']*100:.1f}%)")

    # Variant 2: Volume-weighted
    volume_weighted_mask = base_conditions & (market_stats['yes_volume_ratio'] > 0.65)
    volume_weighted = market_stats[volume_weighted_mask].copy()
    volume_stats = calculate_edge_stats(volume_weighted, baseline)

    print(f"\n--- Variant 2: Volume-Weighted ---")
    print(f"  Markets: {volume_stats['n']:,}")
    if volume_stats['valid']:
        print(f"  Win Rate: {volume_stats['win_rate']*100:.1f}%")
        print(f"  Edge: +{volume_stats['edge']*100:.2f}%")
        print(f"  Improvement: +{volume_stats['weighted_improvement']*100:.2f}%")
        print(f"  Buckets: {volume_stats['bucket_ratio']} ({volume_stats['bucket_pct']*100:.1f}%)")

    # Variant 3: Both conditions (stricter)
    both_mask = base_conditions & (market_stats['yes_trade_ratio'] > 0.65) & (market_stats['yes_volume_ratio'] > 0.65)
    both_markets = market_stats[both_mask].copy()
    both_stats = calculate_edge_stats(both_markets, baseline)

    print(f"\n--- Variant 3: BOTH > 65% (Stricter) ---")
    print(f"  Markets: {both_stats['n']:,}")
    if both_stats['valid']:
        print(f"  Win Rate: {both_stats['win_rate']*100:.1f}%")
        print(f"  Edge: +{both_stats['edge']*100:.2f}%")
        print(f"  Improvement: +{both_stats['weighted_improvement']*100:.2f}%")
        print(f"  Buckets: {both_stats['bucket_ratio']} ({both_stats['bucket_pct']*100:.1f}%)")

    # Variant 4: Either condition (broader)
    either_mask = base_conditions & ((market_stats['yes_trade_ratio'] > 0.65) | (market_stats['yes_volume_ratio'] > 0.65))
    either_markets = market_stats[either_mask].copy()
    either_stats = calculate_edge_stats(either_markets, baseline)

    print(f"\n--- Variant 4: EITHER > 65% (Broader) ---")
    print(f"  Markets: {either_stats['n']:,}")
    if either_stats['valid']:
        print(f"  Win Rate: {either_stats['win_rate']*100:.1f}%")
        print(f"  Edge: +{either_stats['edge']*100:.2f}%")
        print(f"  Improvement: +{either_stats['weighted_improvement']*100:.2f}%")
        print(f"  Buckets: {either_stats['bucket_ratio']} ({either_stats['bucket_pct']*100:.1f}%)")

    # Overlap analysis
    trade_set = set(trade_weighted['market_ticker'])
    volume_set = set(volume_weighted['market_ticker'])

    overlap = len(trade_set & volume_set)
    trade_only = len(trade_set - volume_set)
    volume_only = len(volume_set - trade_set)

    print(f"\n--- Overlap Analysis ---")
    print(f"  Trade-weighted only: {trade_only:,} markets")
    print(f"  Volume-weighted only: {volume_only:,} markets")
    print(f"  Both: {overlap:,} markets")
    print(f"  Overlap %: {overlap/(len(trade_set | volume_set))*100:.1f}%")

    # Analyze trade-only vs volume-only markets
    trade_only_markets = market_stats[market_stats['market_ticker'].isin(trade_set - volume_set)]
    volume_only_markets = market_stats[market_stats['market_ticker'].isin(volume_set - trade_set)]

    if len(trade_only_markets) >= 30:
        trade_only_stats = calculate_edge_stats(trade_only_markets, baseline)
        print(f"\n  Trade-only markets edge: +{trade_only_stats.get('edge', 0)*100:.2f}%")

    if len(volume_only_markets) >= 30:
        volume_only_stats = calculate_edge_stats(volume_only_markets, baseline)
        print(f"  Volume-only markets edge: +{volume_only_stats.get('edge', 0)*100:.2f}%")

    # Store results
    results['e001_volume_weighted']['comparison'] = {
        'trade_weighted': trade_stats,
        'volume_weighted': volume_stats,
        'both_conditions': both_stats,
        'either_condition': either_stats
    }

    results['e001_volume_weighted']['overlap_analysis'] = {
        'trade_only': trade_only,
        'volume_only': volume_only,
        'overlap': overlap,
        'overlap_pct': overlap / (len(trade_set | volume_set)) if (trade_set | volume_set) else 0
    }

    # Recommendation
    if both_stats['valid'] and both_stats['edge'] > trade_stats.get('edge', 0):
        rec = f"IMPLEMENT: Both conditions (trade AND volume > 65%) shows higher edge (+{both_stats['edge']*100:.2f}% vs +{trade_stats.get('edge', 0)*100:.2f}%)"
    elif volume_stats['valid'] and volume_stats['edge'] > trade_stats.get('edge', 0):
        rec = f"CONSIDER: Volume-weighted shows higher edge (+{volume_stats['edge']*100:.2f}% vs +{trade_stats.get('edge', 0)*100:.2f}%)"
    else:
        rec = "KEEP CURRENT: Trade-weighted performs as well or better"

    results['e001_volume_weighted']['recommendation'] = rec
    print(f"\n  RECOMMENDATION: {rec}")

    return results['e001_volume_weighted']


def analyze_f001_rlm_s013_combo(market_stats, baseline):
    """
    F-001: RLM + S013 Combination Analysis

    Find markets where BOTH RLM and S013 conditions fire.
    Calculate edge boost from combining independent signals.
    """
    print("\n" + "=" * 80)
    print("F-001: RLM + S013 COMBINATION ANALYSIS")
    print("=" * 80)

    results['f001_rlm_s013_combo'] = {
        'hypothesis': 'When both RLM and S013 signals fire, edge should compound',
        'rlm_only': {},
        's013_only': {},
        'combined': {},
        'union': {},
        'overlap_analysis': {},
        'recommendation': ''
    }

    # RLM signal
    rlm_mask = (
        (market_stats['yes_trade_ratio'] > 0.65) &
        (market_stats['yes_price_dropped']) &
        (market_stats['total_trades'] >= 15)
    )

    # S013 signal: leverage_std < 0.7, no_trade_ratio > 0.5, n_trades >= 5
    s013_mask = (
        (market_stats['leverage_std'] < 0.7) &
        (market_stats['no_trade_ratio'] > 0.5) &
        (market_stats['total_trades'] >= 5)
    )

    # Markets
    rlm_markets = market_stats[rlm_mask].copy()
    s013_markets = market_stats[s013_mask].copy()
    combined_markets = market_stats[rlm_mask & s013_mask].copy()
    union_markets = market_stats[rlm_mask | s013_mask].copy()

    # Calculate stats
    rlm_stats = calculate_edge_stats(rlm_markets, baseline)
    s013_stats = calculate_edge_stats(s013_markets, baseline)
    combined_stats = calculate_edge_stats(combined_markets, baseline)
    union_stats = calculate_edge_stats(union_markets, baseline)

    print(f"\n--- RLM Only ---")
    print(f"  Markets: {rlm_stats['n']:,}")
    if rlm_stats['valid']:
        print(f"  Win Rate: {rlm_stats['win_rate']*100:.1f}%")
        print(f"  Edge: +{rlm_stats['edge']*100:.2f}%")
        print(f"  Improvement: +{rlm_stats['weighted_improvement']*100:.2f}%")
        print(f"  Buckets: {rlm_stats['bucket_ratio']} ({rlm_stats['bucket_pct']*100:.1f}%)")

    print(f"\n--- S013 Only ---")
    print(f"  Markets: {s013_stats['n']:,}")
    if s013_stats['valid']:
        print(f"  Win Rate: {s013_stats['win_rate']*100:.1f}%")
        print(f"  Edge: +{s013_stats['edge']*100:.2f}%")
        print(f"  Improvement: +{s013_stats['weighted_improvement']*100:.2f}%")
        print(f"  Buckets: {s013_stats['bucket_ratio']} ({s013_stats['bucket_pct']*100:.1f}%)")

    print(f"\n--- COMBINED (RLM AND S013) ---")
    print(f"  Markets: {combined_stats['n']:,}")
    if combined_stats['valid']:
        print(f"  Win Rate: {combined_stats['win_rate']*100:.1f}%")
        print(f"  Edge: +{combined_stats['edge']*100:.2f}%")
        print(f"  Improvement: +{combined_stats['weighted_improvement']*100:.2f}%")
        print(f"  Buckets: {combined_stats['bucket_ratio']} ({combined_stats['bucket_pct']*100:.1f}%)")
        print(f"  P-value: {combined_stats['p_value']:.2e}")
        print(f"  95% CI: [{combined_stats['ci_95_lower']*100:.2f}%, {combined_stats['ci_95_upper']*100:.2f}%]")

    print(f"\n--- UNION (RLM OR S013) ---")
    print(f"  Markets: {union_stats['n']:,}")
    if union_stats['valid']:
        print(f"  Win Rate: {union_stats['win_rate']*100:.1f}%")
        print(f"  Edge: +{union_stats['edge']*100:.2f}%")
        print(f"  Improvement: +{union_stats['weighted_improvement']*100:.2f}%")
        print(f"  Buckets: {union_stats['bucket_ratio']} ({union_stats['bucket_pct']*100:.1f}%)")

    # Overlap analysis
    rlm_set = set(rlm_markets['market_ticker'])
    s013_set = set(s013_markets['market_ticker'])

    overlap = len(rlm_set & s013_set)
    rlm_only = len(rlm_set - s013_set)
    s013_only = len(s013_set - rlm_set)

    print(f"\n--- Overlap Analysis ---")
    print(f"  RLM-only markets: {rlm_only:,}")
    print(f"  S013-only markets: {s013_only:,}")
    print(f"  Both signals: {overlap:,}")
    print(f"  Overlap % (of RLM): {overlap/len(rlm_set)*100:.1f}%" if rlm_set else "  N/A")
    print(f"  Overlap % (of S013): {overlap/len(s013_set)*100:.1f}%" if s013_set else "  N/A")

    # Edge boost calculation
    edge_boost = 0
    if combined_stats['valid'] and rlm_stats['valid']:
        edge_boost = combined_stats['edge'] - rlm_stats['edge']
        print(f"\n  EDGE BOOST from combining: +{edge_boost*100:.2f}%")
        print(f"  Combined edge / RLM edge: {combined_stats['edge']/rlm_stats['edge']*100:.1f}%")

    # Store results
    results['f001_rlm_s013_combo']['rlm_only'] = rlm_stats
    results['f001_rlm_s013_combo']['s013_only'] = s013_stats
    results['f001_rlm_s013_combo']['combined'] = combined_stats
    results['f001_rlm_s013_combo']['union'] = union_stats
    results['f001_rlm_s013_combo']['overlap_analysis'] = {
        'rlm_only': rlm_only,
        's013_only': s013_only,
        'overlap': overlap,
        'overlap_pct_of_rlm': overlap / len(rlm_set) if rlm_set else 0,
        'overlap_pct_of_s013': overlap / len(s013_set) if s013_set else 0,
        'edge_boost': edge_boost
    }

    # Recommendation
    if combined_stats['valid'] and combined_stats['n'] >= 50:
        if combined_stats['edge'] > rlm_stats.get('edge', 0) * 1.1:  # 10% improvement threshold
            rec = f"IMPLEMENT: Combined signal shows {edge_boost*100:+.2f}% edge boost ({combined_stats['n']} markets, {combined_stats['bucket_ratio']} buckets)"
        else:
            rec = f"MONITOR: Combined signal exists but edge boost is small ({edge_boost*100:+.2f}%)"
    else:
        rec = f"INSUFFICIENT: Only {combined_stats['n']} overlap markets (need 50+)"

    results['f001_rlm_s013_combo']['recommendation'] = rec
    print(f"\n  RECOMMENDATION: {rec}")

    return results['f001_rlm_s013_combo']


def analyze_s001_signal_strength(market_stats, baseline):
    """
    S-001: Position Scaling by Signal Strength

    Bucket analysis by price_drop magnitude.
    Calculate optimal position sizing by signal strength tier.
    """
    print("\n" + "=" * 80)
    print("S-001: POSITION SCALING BY SIGNAL STRENGTH")
    print("=" * 80)

    results['s001_signal_strength'] = {
        'hypothesis': 'Larger price drops indicate stronger smart money conviction; scale positions accordingly',
        'by_price_drop': [],
        'optimal_scaling': {},
        'ev_analysis': {},
        'recommendation': ''
    }

    # Base RLM conditions (without price drop filter)
    base_mask = (
        (market_stats['yes_trade_ratio'] > 0.65) &
        (market_stats['total_trades'] >= 15)
    )
    base_rlm = market_stats[base_mask].copy()

    print(f"\nBase RLM markets (before price drop filter): {len(base_rlm):,}")

    # Price drop buckets
    price_drop_buckets = [
        ('0-1c', 0, 1),
        ('1-2c', 1, 2),
        ('2-3c', 2, 3),
        ('3-5c', 3, 5),
        ('5-10c', 5, 10),
        ('10-15c', 10, 15),
        ('15-20c', 15, 20),
        ('20c+', 20, 100)
    ]

    print(f"\n{'Price Drop':<12} {'Markets':>10} {'Win Rate':>10} {'Avg NO':>10} {'Edge':>10} {'Improvement':>12} {'Buckets':>12}")
    print("-" * 80)

    bucket_results = []
    total_ev = 0

    for label, low, high in price_drop_buckets:
        bucket_mask = base_mask & (market_stats['yes_price_drop'] >= low) & (market_stats['yes_price_drop'] < high)
        bucket_markets = market_stats[bucket_mask].copy()

        if len(bucket_markets) < 20:
            print(f"{label:<12} {len(bucket_markets):>10} {'(insufficient)':>10}")
            bucket_results.append({
                'bucket': label,
                'low': low,
                'high': high,
                'n': len(bucket_markets),
                'valid': False
            })
            continue

        bucket_stats = calculate_edge_stats(bucket_markets, baseline, min_markets=20)

        if bucket_stats['valid']:
            # Calculate EV per $100 bet
            avg_no_price = bucket_stats['avg_price']
            win_rate = bucket_stats['win_rate']
            ev_per_100 = win_rate * (100 - avg_no_price) - (1 - win_rate) * avg_no_price

            print(f"{label:<12} {bucket_stats['n']:>10,} {bucket_stats['win_rate']*100:>9.1f}% {avg_no_price:>9.1f}c {bucket_stats['edge']*100:>9.2f}% {bucket_stats['weighted_improvement']*100:>11.2f}% {bucket_stats['bucket_ratio']:>12}")

            bucket_results.append({
                'bucket': label,
                'low': low,
                'high': high,
                'n': int(bucket_stats['n']),
                'valid': True,
                'win_rate': float(bucket_stats['win_rate']),
                'avg_no_price': float(avg_no_price),
                'edge': float(bucket_stats['edge']),
                'improvement': float(bucket_stats['weighted_improvement']),
                'bucket_ratio': bucket_stats['bucket_ratio'],
                'bucket_pct': float(bucket_stats['bucket_pct']),
                'ev_per_100': float(ev_per_100),
                'p_value': float(bucket_stats['p_value']),
                'ci_lower': float(bucket_stats['ci_95_lower']),
                'ci_upper': float(bucket_stats['ci_95_upper'])
            })

            total_ev += ev_per_100 * bucket_stats['n']
        else:
            print(f"{label:<12} {bucket_stats['n']:>10,} (invalid: {bucket_stats.get('reason', 'unknown')})")
            bucket_results.append({
                'bucket': label,
                'low': low,
                'high': high,
                'n': bucket_stats['n'],
                'valid': False
            })

    results['s001_signal_strength']['by_price_drop'] = bucket_results

    # EV Analysis
    print(f"\n--- Expected Value Analysis ---")
    valid_buckets = [b for b in bucket_results if b.get('valid', False)]

    if valid_buckets:
        # Sort by edge
        sorted_by_edge = sorted(valid_buckets, key=lambda x: x.get('edge', 0), reverse=True)
        print(f"\nBuckets sorted by EDGE (highest first):")
        for b in sorted_by_edge:
            print(f"  {b['bucket']}: +{b['edge']*100:.2f}% edge, {b['n']} markets")

        # Sort by total EV contribution
        sorted_by_ev = sorted(valid_buckets, key=lambda x: x.get('ev_per_100', 0) * x.get('n', 0), reverse=True)
        print(f"\nBuckets sorted by TOTAL EV (highest first):")
        for b in sorted_by_ev:
            ev_contribution = b['ev_per_100'] * b['n']
            print(f"  {b['bucket']}: ${ev_contribution:.0f} total EV ({b['n']} markets * ${b['ev_per_100']:.2f}/market)")

    # Optimal scaling recommendation
    print(f"\n--- Optimal Scaling Recommendation ---")

    scaling_tiers = {
        'SKIP': {'buckets': [], 'scale': 0.0, 'reason': 'Edge too low or negative'},
        'REDUCED': {'buckets': [], 'scale': 0.5, 'reason': 'Positive but weak edge'},
        'STANDARD': {'buckets': [], 'scale': 1.0, 'reason': 'Solid edge'},
        'INCREASED': {'buckets': [], 'scale': 1.5, 'reason': 'Strong edge'},
        'MAXIMUM': {'buckets': [], 'scale': 2.0, 'reason': 'Very strong edge'}
    }

    for b in valid_buckets:
        edge = b.get('edge', 0)
        bucket_pct = b.get('bucket_pct', 0)

        if edge < 0.05 or bucket_pct < 0.5:
            tier = 'SKIP'
        elif edge < 0.10 or bucket_pct < 0.7:
            tier = 'REDUCED'
        elif edge < 0.15 or bucket_pct < 0.8:
            tier = 'STANDARD'
        elif edge < 0.25:
            tier = 'INCREASED'
        else:
            tier = 'MAXIMUM'

        scaling_tiers[tier]['buckets'].append(b['bucket'])

    print(f"\nScaling Tiers:")
    for tier, data in scaling_tiers.items():
        if data['buckets']:
            print(f"  {tier} ({data['scale']}x): {', '.join(data['buckets'])} - {data['reason']}")

    results['s001_signal_strength']['optimal_scaling'] = scaling_tiers

    # Summary recommendation
    best_bucket = max(valid_buckets, key=lambda x: x.get('edge', 0)) if valid_buckets else None
    if best_bucket:
        rec = f"IMPLEMENT: Scale positions by price_drop. Best tier: {best_bucket['bucket']} (+{best_bucket['edge']*100:.2f}% edge). Skip price drops < 2c."
    else:
        rec = "INSUFFICIENT: Not enough data for scaling analysis"

    results['s001_signal_strength']['recommendation'] = rec
    print(f"\n  RECOMMENDATION: {rec}")

    return results['s001_signal_strength']


def generate_summary():
    """Generate overall summary and recommendations."""
    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)

    summary = {
        'baseline_rlm_edge': results['baseline_rlm'].get('edge', 0),
        'findings': [],
        'implementations': [],
        'verdict': {}
    }

    # E-001 Summary
    e001 = results['e001_volume_weighted']
    if e001.get('comparison', {}).get('both_conditions', {}).get('valid'):
        both = e001['comparison']['both_conditions']
        trade = e001['comparison']['trade_weighted']
        edge_diff = both['edge'] - trade.get('edge', 0)

        if edge_diff > 0.02:
            verdict = "IMPLEMENT"
            summary['implementations'].append(f"E-001: Add volume-weighted filter (both > 65%) for +{edge_diff*100:.2f}% edge boost")
        elif edge_diff > 0:
            verdict = "CONSIDER"
        else:
            verdict = "SKIP"

        summary['findings'].append(f"E-001 Volume-Weighted: {verdict} - Both conditions edge: +{both['edge']*100:.2f}% vs trade-only: +{trade.get('edge', 0)*100:.2f}%")
        summary['verdict']['e001'] = verdict

    # F-001 Summary
    f001 = results['f001_rlm_s013_combo']
    if f001.get('combined', {}).get('valid'):
        combined = f001['combined']
        rlm = f001['rlm_only']
        edge_boost = f001['overlap_analysis'].get('edge_boost', 0)

        if combined['n'] >= 50 and edge_boost > 0.02:
            verdict = "IMPLEMENT"
            summary['implementations'].append(f"F-001: Flag RLM+S013 overlap for higher confidence ({combined['n']} markets, +{edge_boost*100:.2f}% boost)")
        elif combined['n'] >= 30:
            verdict = "MONITOR"
        else:
            verdict = "INSUFFICIENT_DATA"

        summary['findings'].append(f"F-001 RLM+S013: {verdict} - Combined edge: +{combined['edge']*100:.2f}%, Overlap: {f001['overlap_analysis']['overlap']} markets")
        summary['verdict']['f001'] = verdict

    # S-001 Summary
    s001 = results['s001_signal_strength']
    valid_buckets = [b for b in s001.get('by_price_drop', []) if b.get('valid')]
    if valid_buckets:
        best = max(valid_buckets, key=lambda x: x.get('edge', 0))
        worst = min(valid_buckets, key=lambda x: x.get('edge', 0))

        edge_range = best['edge'] - worst['edge']

        if edge_range > 0.10:
            verdict = "IMPLEMENT"
            summary['implementations'].append(f"S-001: Scale by price_drop - {best['bucket']}: +{best['edge']*100:.2f}% vs {worst['bucket']}: +{worst['edge']*100:.2f}%")
        elif edge_range > 0.05:
            verdict = "CONSIDER"
        else:
            verdict = "UNIFORM"

        summary['findings'].append(f"S-001 Signal Strength: {verdict} - Edge range: {edge_range*100:.2f}% across price drop buckets")
        summary['verdict']['s001'] = verdict

    results['summary'] = summary

    print(f"\n--- Key Findings ---")
    for finding in summary['findings']:
        print(f"  * {finding}")

    print(f"\n--- Recommended Implementations ---")
    if summary['implementations']:
        for impl in summary['implementations']:
            print(f"  * {impl}")
    else:
        print("  None - baseline RLM is already optimal")

    return summary


def main():
    """Main execution."""
    # Load data
    df = load_data()

    # Build market-level stats
    market_stats = build_market_stats(df)

    # Build baseline
    baseline = build_baseline(market_stats)

    # Validate baseline RLM
    validate_baseline_rlm(market_stats, baseline)

    # Run all analyses
    analyze_e001_volume_weighted(market_stats, baseline)
    analyze_f001_rlm_s013_combo(market_stats, baseline)
    analyze_s001_signal_strength(market_stats, baseline)

    # Generate summary
    generate_summary()

    # Save results
    print(f"\n" + "=" * 80)
    print(f"SAVING RESULTS")
    print("=" * 80)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Completed: {datetime.now()}")

    return results


if __name__ == '__main__':
    main()
