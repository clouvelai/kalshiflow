"""
RLM Rolling Window vs Lifetime Accumulation Investigation
==========================================================

Research Question: Should RLM use rolling windows instead of lifetime accumulation?

Current Implementation (Lifetime):
- Trades accumulate indefinitely from market open
- first_yes_price anchors to FIRST trade ever
- Signal triggers when: >= min_trades, > yes_threshold, >= price_drop

This script tests:
1. Signal decay analysis (does edge decay as signal ages?)
2. Rolling window variants (5/10/15/30 min windows)
3. Price anchor staleness (first trade ever vs first trade in window)
4. Time-to-trigger distribution

Author: Quant Agent
Date: 2026-01-01
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from datetime import datetime, timedelta
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
MARKETS_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/markets/market_outcomes_ALL.csv'
OUTPUT_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/reports/rlm_rolling_window_investigation.json'

# RLM Parameters (current production config)
DEFAULT_YES_THRESHOLD = 0.70  # >70% YES trades
DEFAULT_MIN_TRADES = 25       # At least 25 trades
DEFAULT_MIN_PRICE_DROP = 2    # At least 2c drop

# Results container
results = {
    'metadata': {
        'analysis': 'RLM Rolling Window vs Lifetime Accumulation',
        'timestamp': datetime.now().isoformat(),
        'data_source': DATA_PATH,
    },
    'baseline_lifetime': {},
    'rolling_window_results': {},
    'signal_decay_analysis': {},
    'time_to_trigger_distribution': {},
    'price_anchor_analysis': {},
    'comparison_table': {},
    'recommendation': {}
}


def load_data():
    """Load and prepare trade data."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    df = pd.read_csv(DATA_PATH)
    markets = pd.read_csv(MARKETS_PATH)

    # Parse datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['timestamp_ms'] = df['timestamp']

    # Ensure sorted by market and time
    df = df.sort_values(['market_ticker', 'datetime'])

    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Check timestamp resolution
    time_diffs = df.groupby('market_ticker')['datetime'].diff().dropna()
    median_diff = time_diffs.median()
    print(f"Median time between trades: {median_diff}")

    results['metadata']['total_trades'] = len(df)
    results['metadata']['total_markets'] = df['market_ticker'].nunique()
    results['metadata']['date_range'] = f"{df['datetime'].min()} to {df['datetime'].max()}"

    return df, markets


def build_baseline_win_rates(df):
    """Build baseline win rates at 5c NO price buckets."""
    market_stats = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
    }).reset_index()

    market_stats['no_price_bucket'] = (market_stats['no_price'] // 5) * 5
    market_stats['no_win'] = (market_stats['market_result'] == 'no').astype(int)

    baseline = {}
    for bucket, group in market_stats.groupby('no_price_bucket'):
        if len(group) >= 20:
            baseline[bucket] = {
                'win_rate': group['no_win'].mean(),
                'n_markets': len(group)
            }

    return baseline


def compute_market_rlm_timeline(market_df, yes_threshold=DEFAULT_YES_THRESHOLD,
                                 min_trades=DEFAULT_MIN_TRADES,
                                 min_price_drop=DEFAULT_MIN_PRICE_DROP):
    """
    Compute RLM signal timeline for a single market.

    Returns dict with:
    - signal_triggered: bool
    - first_threshold_time: when trade count + YES% thresholds first met
    - signal_time: when price drop condition also met (full signal)
    - time_waiting_for_price_drop: difference between above
    - signal details at trigger time
    """
    if len(market_df) < min_trades:
        return {'signal_triggered': False, 'reason': 'insufficient_trades'}

    market_df = market_df.sort_values('datetime').reset_index(drop=True)

    # Track cumulative state trade-by-trade
    yes_count = 0
    no_count = 0
    first_yes_price = None
    first_threshold_met_idx = None
    first_threshold_met_time = None

    for idx, row in market_df.iterrows():
        # Update counts
        if row['taker_side'] == 'yes':
            yes_count += 1
        else:
            no_count += 1

        total = yes_count + no_count
        yes_ratio = yes_count / total if total > 0 else 0

        # Track first YES price
        if first_yes_price is None:
            first_yes_price = row['yes_price']

        current_yes_price = row['yes_price']
        price_drop = first_yes_price - current_yes_price

        # Check if trade count + YES% thresholds met (but not price drop yet)
        if first_threshold_met_idx is None:
            if total >= min_trades and yes_ratio > yes_threshold:
                first_threshold_met_idx = idx
                first_threshold_met_time = row['datetime']

        # Check if full signal (including price drop)
        if total >= min_trades and yes_ratio > yes_threshold and price_drop >= min_price_drop:
            # Full signal triggered
            signal_time = row['datetime']

            # Calculate waiting time
            if first_threshold_met_time is not None:
                waiting_time = (signal_time - first_threshold_met_time).total_seconds()
            else:
                waiting_time = 0

            return {
                'signal_triggered': True,
                'first_threshold_time': first_threshold_met_time,
                'signal_time': signal_time,
                'time_waiting_for_price_drop_seconds': waiting_time,
                'yes_ratio_at_signal': yes_ratio,
                'price_drop_at_signal': price_drop,
                'trade_count_at_signal': total,
                'no_price_entry': 100 - current_yes_price,
                'first_yes_price': first_yes_price,
                'signal_yes_price': current_yes_price,
                'market_result': market_df['market_result'].iloc[0]
            }

    # Thresholds met but price drop never happened
    if first_threshold_met_idx is not None:
        return {
            'signal_triggered': False,
            'reason': 'price_drop_never_met',
            'first_threshold_time': first_threshold_met_time,
            'yes_ratio_final': yes_ratio,
            'price_drop_final': price_drop
        }

    return {'signal_triggered': False, 'reason': 'thresholds_never_met'}


def compute_rolling_window_signal(market_df, window_minutes,
                                   yes_threshold=DEFAULT_YES_THRESHOLD,
                                   min_trades=DEFAULT_MIN_TRADES,
                                   min_price_drop=DEFAULT_MIN_PRICE_DROP):
    """
    Compute RLM signal using a rolling time window.

    Only counts trades within the last `window_minutes`.
    Price anchor is first trade in the window.
    """
    if len(market_df) < min_trades:
        return {'signal_triggered': False, 'reason': 'insufficient_trades'}

    market_df = market_df.sort_values('datetime').reset_index(drop=True)
    window_td = timedelta(minutes=window_minutes)

    # Check at each trade if signal conditions are met within window
    for end_idx, end_row in market_df.iterrows():
        window_start = end_row['datetime'] - window_td

        # Get trades in window
        window_df = market_df[
            (market_df['datetime'] >= window_start) &
            (market_df['datetime'] <= end_row['datetime'])
        ]

        if len(window_df) < min_trades:
            continue

        # Compute YES ratio in window
        yes_count = (window_df['taker_side'] == 'yes').sum()
        total = len(window_df)
        yes_ratio = yes_count / total

        if yes_ratio <= yes_threshold:
            continue

        # Compute price drop from first trade IN WINDOW
        window_first_yes_price = window_df['yes_price'].iloc[0]
        window_last_yes_price = window_df['yes_price'].iloc[-1]
        price_drop = window_first_yes_price - window_last_yes_price

        if price_drop < min_price_drop:
            continue

        # Signal triggered!
        return {
            'signal_triggered': True,
            'signal_time': end_row['datetime'],
            'yes_ratio': yes_ratio,
            'price_drop': price_drop,
            'trade_count_in_window': total,
            'no_price_entry': 100 - window_last_yes_price,
            'window_first_yes_price': window_first_yes_price,
            'window_last_yes_price': window_last_yes_price,
            'market_result': market_df['market_result'].iloc[0]
        }

    return {'signal_triggered': False, 'reason': 'signal_never_triggered'}


def calculate_edge_stats(signal_markets_df, baseline, side='no'):
    """Calculate comprehensive edge statistics with bucket-matched validation."""
    n = len(signal_markets_df)
    if n < 30:
        return {'n': n, 'valid': False, 'reason': 'insufficient_markets'}

    wins = (signal_markets_df['market_result'] == side).sum()
    wr = wins / n

    avg_price = signal_markets_df['no_price_entry'].mean()
    be = avg_price / 100
    edge = wr - be

    # Z-test for significance
    if 0 < be < 1:
        z = (wins - n * be) / np.sqrt(n * be * (1 - be))
        p_value = 1 - scipy_stats.norm.cdf(z)
    else:
        z = 0
        p_value = 1

    # Bucket-matched analysis
    signal_markets_df = signal_markets_df.copy()
    signal_markets_df['bucket_5c'] = (signal_markets_df['no_price_entry'] // 5) * 5

    improvements = []
    for bucket in sorted(signal_markets_df['bucket_5c'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets_df[signal_markets_df['bucket_5c'] == bucket]
        n_sig = len(sig_bucket)

        if n_sig < 5:
            continue

        sig_wr = (sig_bucket['market_result'] == side).mean()
        base_wr = baseline[bucket]['win_rate']
        imp = sig_wr - base_wr

        improvements.append({
            'bucket': bucket,
            'sig_wr': sig_wr,
            'base_wr': base_wr,
            'improvement': imp,
            'n_sig': n_sig
        })

    if not improvements:
        return {'n': n, 'valid': False, 'reason': 'no_valid_buckets'}

    total_n = sum(i['n_sig'] for i in improvements)
    weighted_imp = sum(i['improvement'] * i['n_sig'] for i in improvements) / total_n

    pos_buckets = sum(1 for i in improvements if i['improvement'] > 0)
    total_buckets = len(improvements)
    bucket_ratio = pos_buckets / total_buckets if total_buckets > 0 else 0

    # Bootstrap CI
    n_bootstrap = 500
    bootstrap_edges = []
    for _ in range(n_bootstrap):
        sample = signal_markets_df.sample(n=len(signal_markets_df), replace=True)
        sample_wr = (sample['market_result'] == side).mean()
        sample_be = sample['no_price_entry'].mean() / 100
        bootstrap_edges.append(sample_wr - sample_be)

    ci_lower = np.percentile(bootstrap_edges, 2.5)
    ci_upper = np.percentile(bootstrap_edges, 97.5)

    return {
        'n': n,
        'valid': True,
        'wins': int(wins),
        'win_rate': float(wr),
        'avg_no_price': float(avg_price),
        'breakeven': float(be),
        'edge': float(edge),
        'p_value': float(p_value),
        'weighted_improvement': float(weighted_imp),
        'pos_buckets': pos_buckets,
        'total_buckets': total_buckets,
        'bucket_ratio': float(bucket_ratio),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
    }


def analyze_lifetime_baseline(df, baseline):
    """Analyze lifetime accumulation (current implementation)."""
    print("\n" + "=" * 80)
    print("PHASE 1: LIFETIME ACCUMULATION BASELINE")
    print("=" * 80)

    lifetime_signals = []
    threshold_met_but_waiting = []

    for ticker, market_df in df.groupby('market_ticker'):
        result = compute_market_rlm_timeline(market_df)

        if result['signal_triggered']:
            lifetime_signals.append({
                'market_ticker': ticker,
                'market_result': result['market_result'],
                'no_price_entry': result['no_price_entry'],
                'yes_ratio': result['yes_ratio_at_signal'],
                'price_drop': result['price_drop_at_signal'],
                'trade_count': result['trade_count_at_signal'],
                'time_waiting': result['time_waiting_for_price_drop_seconds'],
                'first_threshold_time': result['first_threshold_time'],
                'signal_time': result['signal_time'],
            })
        elif result.get('reason') == 'price_drop_never_met':
            threshold_met_but_waiting.append({
                'market_ticker': ticker,
                'first_threshold_time': result['first_threshold_time'],
                'yes_ratio_final': result['yes_ratio_final'],
                'price_drop_final': result['price_drop_final'],
            })

    signals_df = pd.DataFrame(lifetime_signals)

    print(f"\nLifetime Accumulation Results:")
    print(f"  Markets with full signal: {len(signals_df):,}")
    print(f"  Markets with thresholds met but no price drop: {len(threshold_met_but_waiting):,}")

    if len(signals_df) >= 30:
        stats = calculate_edge_stats(signals_df, baseline)

        print(f"\n  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Avg NO Price: {stats['avg_no_price']:.1f}c")
        print(f"  Edge: {stats['edge']:.2%}")
        print(f"  Improvement vs Baseline: {stats['weighted_improvement']:.2%}")
        print(f"  P-value: {stats['p_value']:.2e}")
        print(f"  Bucket Ratio: {stats['pos_buckets']}/{stats['total_buckets']} ({stats['bucket_ratio']:.1%})")
        print(f"  95% CI: [{stats['ci_95_lower']:.2%}, {stats['ci_95_upper']:.2%}]")

        results['baseline_lifetime'] = {
            'stats': stats,
            'n_signals': len(signals_df),
            'n_threshold_met_no_price_drop': len(threshold_met_but_waiting),
        }

    return signals_df, threshold_met_but_waiting


def analyze_rolling_windows(df, baseline):
    """Analyze rolling window variants."""
    print("\n" + "=" * 80)
    print("PHASE 2: ROLLING WINDOW ANALYSIS")
    print("=" * 80)

    window_sizes = [5, 10, 15, 30, 60]  # minutes

    results['rolling_window_results'] = {}

    for window_min in window_sizes:
        print(f"\n--- {window_min}-Minute Rolling Window ---")

        rolling_signals = []

        for ticker, market_df in df.groupby('market_ticker'):
            result = compute_rolling_window_signal(market_df, window_min)

            if result['signal_triggered']:
                rolling_signals.append({
                    'market_ticker': ticker,
                    'market_result': result['market_result'],
                    'no_price_entry': result['no_price_entry'],
                    'yes_ratio': result['yes_ratio'],
                    'price_drop': result['price_drop'],
                    'trade_count': result['trade_count_in_window'],
                })

        signals_df = pd.DataFrame(rolling_signals)

        print(f"  Markets with signal: {len(signals_df):,}")

        if len(signals_df) >= 30:
            stats = calculate_edge_stats(signals_df, baseline)

            print(f"  Win Rate: {stats['win_rate']:.1%}")
            print(f"  Avg NO Price: {stats['avg_no_price']:.1f}c")
            print(f"  Edge: {stats['edge']:.2%}")
            print(f"  Improvement: {stats['weighted_improvement']:.2%}")
            print(f"  P-value: {stats['p_value']:.2e}")
            print(f"  Bucket Ratio: {stats['pos_buckets']}/{stats['total_buckets']} ({stats['bucket_ratio']:.1%})")

            results['rolling_window_results'][f'{window_min}min'] = {
                'stats': stats,
                'n_signals': len(signals_df),
            }
        else:
            print(f"  Insufficient signals for analysis")
            results['rolling_window_results'][f'{window_min}min'] = {
                'stats': {'valid': False, 'reason': 'insufficient_signals'},
                'n_signals': len(signals_df),
            }


def analyze_signal_decay(lifetime_signals_df, baseline):
    """Analyze if signal edge decays with time waiting for price drop."""
    print("\n" + "=" * 80)
    print("PHASE 3: SIGNAL DECAY ANALYSIS")
    print("=" * 80)

    if len(lifetime_signals_df) < 50:
        print("Insufficient signals for decay analysis")
        return

    # Bucket by waiting time
    wait_buckets = [
        ('0-1min', 0, 60),
        ('1-5min', 60, 300),
        ('5-15min', 300, 900),
        ('15-30min', 900, 1800),
        ('30min-1hr', 1800, 3600),
        ('1-4hr', 3600, 14400),
        ('>4hr', 14400, float('inf')),
    ]

    print("\nEdge by Time Waiting for Price Drop:")
    print("-" * 80)
    print(f"{'Bucket':<15} {'Markets':>10} {'Win Rate':>10} {'Edge':>10} {'Improvement':>12} {'P-value':>12}")
    print("-" * 80)

    decay_results = []

    for name, min_sec, max_sec in wait_buckets:
        bucket_df = lifetime_signals_df[
            (lifetime_signals_df['time_waiting'] >= min_sec) &
            (lifetime_signals_df['time_waiting'] < max_sec)
        ]

        if len(bucket_df) < 20:
            continue

        stats = calculate_edge_stats(bucket_df, baseline)

        if stats['valid']:
            print(f"{name:<15} {stats['n']:>10,} {stats['win_rate']:>9.1%} "
                  f"{stats['edge']:>9.2%} {stats['weighted_improvement']:>11.2%} "
                  f"{stats['p_value']:>11.2e}")

            decay_results.append({
                'bucket': name,
                'min_seconds': min_sec,
                'max_seconds': max_sec,
                **stats
            })

    results['signal_decay_analysis']['by_wait_time'] = decay_results

    # Statistical test for decay trend
    if len(decay_results) >= 3:
        # Linear regression on edge vs wait time (use bucket midpoints)
        wait_times = [(r['min_seconds'] + min(r['max_seconds'], 14400)) / 2 for r in decay_results]
        edges = [r['edge'] for r in decay_results]

        slope, intercept, r_value, p_value_trend, std_err = scipy_stats.linregress(wait_times, edges)

        print(f"\nDecay Trend Analysis:")
        print(f"  Slope: {slope*3600:.4f} per hour (negative = decay)")
        print(f"  R-squared: {r_value**2:.3f}")
        print(f"  P-value: {p_value_trend:.4f}")

        if slope < 0 and p_value_trend < 0.10:
            print(f"  CONCLUSION: Evidence of edge decay over time")
        else:
            print(f"  CONCLUSION: No significant edge decay detected")

        results['signal_decay_analysis']['trend'] = {
            'slope_per_hour': float(slope * 3600),
            'r_squared': float(r_value**2),
            'p_value': float(p_value_trend),
            'has_decay': slope < 0 and p_value_trend < 0.10
        }


def analyze_time_to_trigger(lifetime_signals_df):
    """Analyze distribution of time from thresholds met to full signal."""
    print("\n" + "=" * 80)
    print("PHASE 4: TIME-TO-TRIGGER DISTRIBUTION")
    print("=" * 80)

    wait_times = lifetime_signals_df['time_waiting'].dropna()

    print(f"\nTime Waiting for Price Drop (after trade count + YES% met):")
    print(f"  Min: {wait_times.min():.0f} seconds")
    print(f"  25th percentile: {wait_times.quantile(0.25):.0f} seconds ({wait_times.quantile(0.25)/60:.1f} min)")
    print(f"  Median: {wait_times.median():.0f} seconds ({wait_times.median()/60:.1f} min)")
    print(f"  75th percentile: {wait_times.quantile(0.75):.0f} seconds ({wait_times.quantile(0.75)/60:.1f} min)")
    print(f"  Max: {wait_times.max():.0f} seconds ({wait_times.max()/3600:.1f} hours)")
    print(f"  Mean: {wait_times.mean():.0f} seconds ({wait_times.mean()/60:.1f} min)")

    # Distribution breakdown
    instant = (wait_times == 0).sum()
    under_1min = ((wait_times > 0) & (wait_times < 60)).sum()
    under_5min = ((wait_times >= 60) & (wait_times < 300)).sum()
    under_30min = ((wait_times >= 300) & (wait_times < 1800)).sum()
    under_1hr = ((wait_times >= 1800) & (wait_times < 3600)).sum()
    over_1hr = (wait_times >= 3600).sum()

    total = len(wait_times)

    print(f"\nDistribution:")
    print(f"  Instant (0s): {instant:,} ({instant/total:.1%})")
    print(f"  <1 min: {under_1min:,} ({under_1min/total:.1%})")
    print(f"  1-5 min: {under_5min:,} ({under_5min/total:.1%})")
    print(f"  5-30 min: {under_30min:,} ({under_30min/total:.1%})")
    print(f"  30min-1hr: {under_1hr:,} ({under_1hr/total:.1%})")
    print(f"  >1 hour: {over_1hr:,} ({over_1hr/total:.1%})")

    results['time_to_trigger_distribution'] = {
        'min_seconds': float(wait_times.min()),
        'p25_seconds': float(wait_times.quantile(0.25)),
        'median_seconds': float(wait_times.median()),
        'p75_seconds': float(wait_times.quantile(0.75)),
        'max_seconds': float(wait_times.max()),
        'mean_seconds': float(wait_times.mean()),
        'distribution': {
            'instant': int(instant),
            'under_1min': int(under_1min),
            '1_to_5min': int(under_5min),
            '5_to_30min': int(under_30min),
            '30min_to_1hr': int(under_1hr),
            'over_1hr': int(over_1hr),
        },
        'total': int(total)
    }


def analyze_fresh_signals(lifetime_signals_df, baseline):
    """Analyze signals that triggered immediately (freshest signals)."""
    print("\n" + "=" * 80)
    print("PHASE 5: FRESH vs STALE SIGNAL COMPARISON")
    print("=" * 80)

    # Fresh = price drop happened within 5 minutes of thresholds being met
    fresh_df = lifetime_signals_df[lifetime_signals_df['time_waiting'] < 300]  # <5 min
    stale_df = lifetime_signals_df[lifetime_signals_df['time_waiting'] >= 1800]  # >30 min

    print(f"\nFresh Signals (<5 min wait): {len(fresh_df):,}")
    if len(fresh_df) >= 30:
        fresh_stats = calculate_edge_stats(fresh_df, baseline)
        if fresh_stats['valid']:
            print(f"  Win Rate: {fresh_stats['win_rate']:.1%}")
            print(f"  Edge: {fresh_stats['edge']:.2%}")
            print(f"  Improvement: {fresh_stats['weighted_improvement']:.2%}")
            print(f"  Bucket Ratio: {fresh_stats['bucket_ratio']:.1%}")
            results['signal_decay_analysis']['fresh_signals'] = fresh_stats

    print(f"\nStale Signals (>30 min wait): {len(stale_df):,}")
    if len(stale_df) >= 30:
        stale_stats = calculate_edge_stats(stale_df, baseline)
        if stale_stats['valid']:
            print(f"  Win Rate: {stale_stats['win_rate']:.1%}")
            print(f"  Edge: {stale_stats['edge']:.2%}")
            print(f"  Improvement: {stale_stats['weighted_improvement']:.2%}")
            print(f"  Bucket Ratio: {stale_stats['bucket_ratio']:.1%}")
            results['signal_decay_analysis']['stale_signals'] = stale_stats

    # Compare
    if len(fresh_df) >= 30 and len(stale_df) >= 30:
        fresh_stats = calculate_edge_stats(fresh_df, baseline)
        stale_stats = calculate_edge_stats(stale_df, baseline)

        if fresh_stats['valid'] and stale_stats['valid']:
            edge_diff = fresh_stats['edge'] - stale_stats['edge']
            imp_diff = fresh_stats['weighted_improvement'] - stale_stats['weighted_improvement']

            print(f"\nFresh vs Stale Comparison:")
            print(f"  Edge Difference: {edge_diff:+.2%} (fresh - stale)")
            print(f"  Improvement Difference: {imp_diff:+.2%} (fresh - stale)")

            if edge_diff > 0.02:
                print(f"  CONCLUSION: Fresh signals have meaningfully better edge")
            elif edge_diff < -0.02:
                print(f"  CONCLUSION: Stale signals actually have better edge (unexpected)")
            else:
                print(f"  CONCLUSION: No meaningful difference between fresh and stale")

            results['signal_decay_analysis']['fresh_vs_stale'] = {
                'edge_difference': float(edge_diff),
                'improvement_difference': float(imp_diff),
                'fresh_better': edge_diff > 0.02
            }


def generate_comparison_table():
    """Generate final comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    rows = []

    # Lifetime baseline
    if results['baseline_lifetime'].get('stats', {}).get('valid'):
        s = results['baseline_lifetime']['stats']
        rows.append({
            'Method': 'Lifetime',
            'Markets': s['n'],
            'Win Rate': f"{s['win_rate']:.1%}",
            'Avg Price': f"{s['avg_no_price']:.1f}c",
            'Edge': f"{s['edge']:.2%}",
            'Improvement': f"{s['weighted_improvement']:.2%}",
            'Bucket Ratio': f"{s['bucket_ratio']:.1%}",
            'P-value': f"{s['p_value']:.2e}",
        })

    # Rolling windows
    for window_name, data in results['rolling_window_results'].items():
        if data.get('stats', {}).get('valid'):
            s = data['stats']
            rows.append({
                'Method': f'Rolling {window_name}',
                'Markets': s['n'],
                'Win Rate': f"{s['win_rate']:.1%}",
                'Avg Price': f"{s['avg_no_price']:.1f}c",
                'Edge': f"{s['edge']:.2%}",
                'Improvement': f"{s['weighted_improvement']:.2%}",
                'Bucket Ratio': f"{s['bucket_ratio']:.1%}",
                'P-value': f"{s['p_value']:.2e}",
            })

    # Fresh signals only
    if results['signal_decay_analysis'].get('fresh_signals', {}).get('valid'):
        s = results['signal_decay_analysis']['fresh_signals']
        rows.append({
            'Method': 'Lifetime (Fresh <5m)',
            'Markets': s['n'],
            'Win Rate': f"{s['win_rate']:.1%}",
            'Avg Price': f"{s['avg_no_price']:.1f}c",
            'Edge': f"{s['edge']:.2%}",
            'Improvement': f"{s['weighted_improvement']:.2%}",
            'Bucket Ratio': f"{s['bucket_ratio']:.1%}",
            'P-value': f"{s['p_value']:.2e}",
        })

    # Print table
    if rows:
        print("\n" + "-" * 120)
        print(f"{'Method':<20} {'Markets':>10} {'Win Rate':>10} {'Avg Price':>10} {'Edge':>10} {'Improvement':>12} {'Bucket Ratio':>12} {'P-value':>12}")
        print("-" * 120)

        for row in rows:
            print(f"{row['Method']:<20} {row['Markets']:>10} {row['Win Rate']:>10} {row['Avg Price']:>10} "
                  f"{row['Edge']:>10} {row['Improvement']:>12} {row['Bucket Ratio']:>12} {row['P-value']:>12}")

        print("-" * 120)

    results['comparison_table'] = rows


def generate_recommendation():
    """Generate final recommendation based on analysis."""
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Get key metrics
    lifetime_stats = results['baseline_lifetime'].get('stats', {})
    decay_trend = results['signal_decay_analysis'].get('trend', {})
    fresh_vs_stale = results['signal_decay_analysis'].get('fresh_vs_stale', {})

    # Find best rolling window
    best_rolling = None
    best_rolling_improvement = -999

    for window_name, data in results['rolling_window_results'].items():
        stats = data.get('stats', {})
        if stats.get('valid') and stats.get('n', 0) >= 100:
            if stats['weighted_improvement'] > best_rolling_improvement:
                best_rolling_improvement = stats['weighted_improvement']
                best_rolling = {
                    'window': window_name,
                    'stats': stats
                }

    # Decision logic
    recommendation = {
        'action': None,
        'rationale': [],
        'parameters': {}
    }

    # Check if rolling window is meaningfully better
    if best_rolling:
        lifetime_imp = lifetime_stats.get('weighted_improvement', 0)
        rolling_imp = best_rolling['stats']['weighted_improvement']
        improvement_delta = rolling_imp - lifetime_imp

        print(f"\nKey Comparisons:")
        print(f"  Lifetime Improvement: {lifetime_imp:.2%}")
        print(f"  Best Rolling ({best_rolling['window']}) Improvement: {rolling_imp:.2%}")
        print(f"  Delta: {improvement_delta:+.2%}")

        if improvement_delta > 0.03:
            recommendation['action'] = 'SWITCH_TO_ROLLING'
            recommendation['rationale'].append(f"Rolling window shows {improvement_delta:.2%} better improvement")
            recommendation['parameters'] = {
                'window_minutes': int(best_rolling['window'].replace('min', '')),
                'yes_threshold': DEFAULT_YES_THRESHOLD,
                'min_trades': DEFAULT_MIN_TRADES,
                'min_price_drop': DEFAULT_MIN_PRICE_DROP
            }
        elif improvement_delta > 0.01:
            recommendation['action'] = 'CONSIDER_ROLLING'
            recommendation['rationale'].append(f"Rolling window shows marginal {improvement_delta:.2%} improvement")
        else:
            recommendation['action'] = 'KEEP_LIFETIME'
            recommendation['rationale'].append(f"Rolling window does not improve over lifetime ({improvement_delta:+.2%})")

    # Check decay analysis
    if decay_trend.get('has_decay'):
        recommendation['rationale'].append(f"Evidence of edge decay over time (slope: {decay_trend['slope_per_hour']:.4f}/hr)")
        if recommendation['action'] == 'KEEP_LIFETIME':
            recommendation['action'] = 'CONSIDER_ROLLING'
    else:
        recommendation['rationale'].append("No significant edge decay detected")

    # Check fresh vs stale
    if fresh_vs_stale.get('fresh_better'):
        recommendation['rationale'].append(f"Fresh signals have {fresh_vs_stale['edge_difference']:.2%} better edge than stale")
        if recommendation['action'] == 'KEEP_LIFETIME':
            recommendation['action'] = 'CONSIDER_FRESHNESS_FILTER'
    else:
        recommendation['rationale'].append("Fresh vs stale signal edge is similar")

    # Print recommendation
    print(f"\n{'='*60}")
    print(f"FINAL RECOMMENDATION: {recommendation['action']}")
    print(f"{'='*60}")

    for r in recommendation['rationale']:
        print(f"  - {r}")

    if recommendation['parameters']:
        print(f"\nRecommended Parameters:")
        for k, v in recommendation['parameters'].items():
            print(f"  {k}: {v}")

    # Additional considerations
    print(f"\nAdditional Considerations:")

    # Signal frequency tradeoff
    if best_rolling:
        lifetime_n = lifetime_stats.get('n', 0)
        rolling_n = best_rolling['stats']['n']
        freq_ratio = rolling_n / lifetime_n if lifetime_n > 0 else 0
        print(f"  - Signal frequency: Rolling has {freq_ratio:.1%} of lifetime signals ({rolling_n} vs {lifetime_n})")

        if freq_ratio < 0.5:
            print(f"    WARNING: Rolling window significantly reduces signal frequency")
            recommendation['rationale'].append(f"Rolling window reduces signals by {(1-freq_ratio):.0%}")

    # Bucket ratio check
    lifetime_bucket = lifetime_stats.get('bucket_ratio', 0)
    print(f"  - Lifetime bucket ratio: {lifetime_bucket:.1%} (>=80% needed to confirm not price proxy)")

    if best_rolling:
        rolling_bucket = best_rolling['stats'].get('bucket_ratio', 0)
        print(f"  - Best rolling bucket ratio: {rolling_bucket:.1%}")

    results['recommendation'] = recommendation

    return recommendation


def run_temporal_stability_check(df, baseline):
    """Check temporal stability across quarters for key strategies."""
    print("\n" + "=" * 80)
    print("TEMPORAL STABILITY CHECK")
    print("=" * 80)

    # Get all signals with timestamps
    all_signals = []

    for ticker, market_df in df.groupby('market_ticker'):
        result = compute_market_rlm_timeline(market_df)

        if result['signal_triggered']:
            all_signals.append({
                'market_ticker': ticker,
                'market_result': result['market_result'],
                'no_price_entry': result['no_price_entry'],
                'signal_time': result['signal_time'],
                'time_waiting': result['time_waiting_for_price_drop_seconds'],
            })

    signals_df = pd.DataFrame(all_signals)
    signals_df['signal_time'] = pd.to_datetime(signals_df['signal_time'])

    # Divide into quarters
    signals_df = signals_df.sort_values('signal_time')
    min_date = signals_df['signal_time'].min()
    max_date = signals_df['signal_time'].max()
    total_days = (max_date - min_date).days
    quarter_days = total_days // 4 if total_days > 0 else 1

    print(f"\nLifetime Accumulation - Quarterly Performance:")
    print("-" * 60)

    quarterly_results = []
    for q in range(4):
        q_start = min_date + timedelta(days=q * quarter_days)
        q_end = min_date + timedelta(days=(q + 1) * quarter_days)

        q_signals = signals_df[
            (signals_df['signal_time'] >= q_start) &
            (signals_df['signal_time'] < q_end)
        ]

        if len(q_signals) >= 20:
            stats = calculate_edge_stats(q_signals, baseline)
            if stats['valid']:
                print(f"  Q{q+1}: N={stats['n']}, Edge={stats['edge']:.2%}, Improvement={stats['weighted_improvement']:.2%}")
                quarterly_results.append({
                    'quarter': q + 1,
                    'n': stats['n'],
                    'edge': stats['edge'],
                    'improvement': stats['weighted_improvement']
                })

    positive_quarters = sum(1 for q in quarterly_results if q['improvement'] > 0)
    print(f"\nPositive quarters: {positive_quarters}/{len(quarterly_results)}")

    results['temporal_stability'] = {
        'quarters': quarterly_results,
        'positive_quarters': positive_quarters,
        'total_quarters': len(quarterly_results)
    }


def main():
    """Run full RLM rolling window investigation."""
    print("=" * 80)
    print("RLM ROLLING WINDOW vs LIFETIME ACCUMULATION INVESTIGATION")
    print(f"Started: {datetime.now()}")
    print("=" * 80)

    # Load data
    df, markets = load_data()

    # Build baseline
    baseline = build_baseline_win_rates(df)

    # Phase 1: Lifetime baseline
    lifetime_signals_df, threshold_met_waiting = analyze_lifetime_baseline(df, baseline)

    # Phase 2: Rolling windows
    analyze_rolling_windows(df, baseline)

    # Phase 3: Signal decay
    if len(lifetime_signals_df) > 0:
        analyze_signal_decay(lifetime_signals_df, baseline)

    # Phase 4: Time-to-trigger distribution
    if len(lifetime_signals_df) > 0:
        analyze_time_to_trigger(lifetime_signals_df)

    # Phase 5: Fresh vs stale
    if len(lifetime_signals_df) > 0:
        analyze_fresh_signals(lifetime_signals_df, baseline)

    # Temporal stability
    run_temporal_stability_check(df, baseline)

    # Comparison table
    generate_comparison_table()

    # Final recommendation
    generate_recommendation()

    # Save results
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {OUTPUT_PATH}")
    print(f"Analysis completed: {datetime.now()}")

    return results


if __name__ == "__main__":
    results = main()
