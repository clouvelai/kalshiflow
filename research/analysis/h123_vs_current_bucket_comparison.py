"""
H123 vs Current Parameters: Full Bucket-by-Bucket Comparison

Compares two RLM parameter sets:
1. H123: 5 trades minimum, any price drop (>0c), 70% YES threshold
2. Current: 25 trades minimum, 2c price drop minimum, 70% YES threshold

Outputs:
- Table 1: H123 per-bucket breakdown
- Table 2: Current per-bucket breakdown
- Table 3: Side-by-side comparison
- Analysis: Which buckets beat baseline for each?
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import math

import pandas as pd
import numpy as np
from scipy import stats

# Paths
DATA_PATH = '/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'
REPORTS_DIR = Path('/Users/samuelclark/Desktop/kalshiflow/research/reports')


def load_data():
    """Load trade data using pandas."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")

    # Filter to resolved markets only
    resolved_markets = df[df['market_result'].isin(['yes', 'no'])]['market_ticker'].unique()
    df_resolved = df[df['market_ticker'].isin(resolved_markets)]
    print(f"Resolved markets: {len(resolved_markets):,}")

    return df_resolved


def build_baseline(df):
    """Build baseline win rates at 5c price buckets for NO bets."""
    all_markets = df.groupby('market_ticker').agg({
        'market_result': 'first',
        'no_price': 'mean',
        'yes_price': 'mean',
        'datetime': 'first'
    }).reset_index()

    all_markets['bucket_5c'] = (all_markets['no_price'] // 5) * 5

    baseline = {}
    for bucket in sorted(all_markets['bucket_5c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_5c'] == bucket]
        n = len(bucket_markets)
        if n >= 20:
            baseline[bucket] = {
                'win_rate': (bucket_markets['market_result'] == 'no').mean(),
                'n_markets': n
            }

    print(f"Built baseline across {len(baseline)} price buckets")
    return all_markets, baseline


def get_rlm_markets(df, yes_trade_threshold=0.7, min_trades=5, require_price_move=True, price_move_threshold=0):
    """
    Identify RLM markets: Majority YES trades but price moved toward NO.
    """
    df_sorted = df.sort_values(['market_ticker', 'datetime'])

    market_stats = df_sorted.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'yes').mean(),
        'yes_price': ['first', 'last', 'mean'],
        'no_price': ['mean', 'first', 'last'],
        'market_result': 'first',
        'count': ['size', 'sum'],
    }).reset_index()

    market_stats.columns = [
        'market_ticker', 'yes_trade_ratio',
        'first_yes_price', 'last_yes_price', 'avg_yes_price',
        'avg_no_price', 'first_no_price', 'last_no_price',
        'market_result',
        'n_trades', 'total_contracts',
    ]

    # Calculate price movement
    market_stats['yes_price_moved_down'] = market_stats['last_yes_price'] < market_stats['first_yes_price']
    market_stats['yes_price_drop'] = market_stats['first_yes_price'] - market_stats['last_yes_price']

    # Apply RLM filters
    conditions = (
        (market_stats['yes_trade_ratio'] > yes_trade_threshold) &
        (market_stats['n_trades'] >= min_trades)
    )

    if require_price_move:
        conditions = conditions & (market_stats['yes_price_moved_down'])
        if price_move_threshold > 0:
            conditions = conditions & (market_stats['yes_price_drop'] >= price_move_threshold)

    rlm = market_stats[conditions].copy()

    return rlm, market_stats


def calculate_baseline_win_rate(no_price_cents: float) -> float:
    """Baseline = NO price / 100."""
    return no_price_cents / 100.0


def binomial_test_pvalue(wins: int, total: int, expected_rate: float) -> float:
    """One-sided p-value for win rate > expected rate."""
    if total == 0:
        return 1.0

    observed_rate = wins / total
    if observed_rate <= expected_rate:
        return 1.0

    std_err = math.sqrt(expected_rate * (1 - expected_rate) / total)
    if std_err == 0:
        return 0.0 if observed_rate > expected_rate else 1.0

    z = (observed_rate - expected_rate) / std_err
    p_value = 0.5 * math.erfc(z / math.sqrt(2))
    return p_value


def get_price_bucket(no_price: float) -> str:
    """Get 5c price bucket label."""
    lower = int((no_price // 5) * 5)
    return f"{lower}-{lower+5}c"


def get_bucket_midpoint(bucket: str) -> int:
    """Extract bucket midpoint for sorting."""
    lower = int(bucket.split('-')[0])
    return lower + 2


def analyze_rlm_strategy(
    rlm_markets: pd.DataFrame,
    baseline: Dict,
    strategy_name: str,
    parameters: Dict,
) -> Dict[str, Any]:
    """Analyze RLM strategy per-bucket."""

    # Add bucket column
    rlm = rlm_markets.copy()
    rlm['bucket_5c'] = (rlm['avg_no_price'] // 5) * 5
    rlm['bucket_label'] = rlm['avg_no_price'].apply(get_price_bucket)

    # Calculate per-bucket stats
    bucket_results = []

    for bucket in sorted(rlm['bucket_5c'].unique()):
        bucket_markets = rlm[rlm['bucket_5c'] == bucket]
        n_markets = len(bucket_markets)

        if n_markets == 0:
            continue

        wins = (bucket_markets['market_result'] == 'no').sum()
        win_rate = wins / n_markets
        avg_no_price = bucket_markets['avg_no_price'].mean()

        # Get baseline from bucket_matched baseline or use price-based
        if bucket in baseline:
            base_wr = baseline[bucket]['win_rate']
        else:
            base_wr = bucket / 100.0  # Use price as proxy

        improvement = win_rate - base_wr
        p_value = binomial_test_pvalue(int(wins), n_markets, base_wr)

        bucket_label = get_price_bucket(avg_no_price)

        bucket_results.append({
            'bucket': bucket_label,
            'bucket_midpoint': int(bucket) + 2,
            'n_markets': n_markets,
            'wins': int(wins),
            'losses': n_markets - int(wins),
            'win_rate': float(win_rate),
            'avg_no_price': float(avg_no_price),
            'baseline': float(base_wr),
            'improvement': float(improvement),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
        })

    # Sort by bucket
    bucket_results.sort(key=lambda x: x['bucket_midpoint'])

    # Calculate overall stats
    total_markets = len(rlm)
    total_wins = (rlm['market_result'] == 'no').sum()
    overall_win_rate = total_wins / total_markets if total_markets > 0 else 0

    # Weighted baseline
    weighted_baseline = sum(b['baseline'] * b['n_markets'] for b in bucket_results) / total_markets if total_markets > 0 else 0

    return {
        'strategy_name': strategy_name,
        'parameters': parameters,
        'bucket_results': bucket_results,
        'overall': {
            'total_markets': int(total_markets),
            'total_wins': int(total_wins),
            'win_rate': float(overall_win_rate),
            'weighted_baseline': float(weighted_baseline),
            'overall_improvement': float(overall_win_rate - weighted_baseline),
        }
    }


def format_table(results: Dict[str, Any], title: str) -> str:
    """Format results as a markdown table."""
    lines = [
        f"\n## {title}",
        f"**Parameters**: yes_threshold={results['parameters']['yes_threshold']:.0%}, "
        f"min_trades={results['parameters']['min_trades']}, "
        f"min_price_drop={results['parameters']['min_price_drop']}c",
        "",
        "| NO Price Bucket | N Markets | RLM Win Rate | Baseline | Improvement | P-value | Significant |",
        "|-----------------|-----------|--------------|----------|-------------|---------|-------------|",
    ]

    for b in results['bucket_results']:
        sig = "YES" if b['significant'] else "no"
        lines.append(
            f"| {b['bucket']:12} | {b['n_markets']:9} | {b['win_rate']:11.1%} | {b['baseline']:7.1%} | "
            f"{b['improvement']:+10.1%} | {b['p_value']:7.4f} | {sig:11} |"
        )

    # Add totals row
    o = results['overall']
    lines.append("|-----------------|-----------|--------------|----------|-------------|---------|-------------|")
    lines.append(
        f"| **TOTAL**       | {o['total_markets']:9} | {o['win_rate']:11.1%} | {o['weighted_baseline']:7.1%} | "
        f"{o['overall_improvement']:+10.1%} |         |             |"
    )

    return "\n".join(lines)


def format_comparison_table(h123: Dict, current: Dict) -> str:
    """Format side-by-side comparison table."""
    # Get all unique buckets
    all_buckets = set()
    h123_by_bucket = {}
    current_by_bucket = {}

    for b in h123['bucket_results']:
        all_buckets.add(b['bucket'])
        h123_by_bucket[b['bucket']] = b

    for b in current['bucket_results']:
        all_buckets.add(b['bucket'])
        current_by_bucket[b['bucket']] = b

    # Sort buckets
    sorted_buckets = sorted(all_buckets, key=lambda x: int(x.split('-')[0]))

    lines = [
        "\n## Table 3: Side-by-Side Comparison",
        "",
        "| Bucket | H123 N | H123 Win | H123 Improve | H123 Sig | Current N | Current Win | Current Improve | Current Sig |",
        "|--------|--------|----------|--------------|----------|-----------|-------------|-----------------|-------------|",
    ]

    for bucket in sorted_buckets:
        h = h123_by_bucket.get(bucket)
        c = current_by_bucket.get(bucket)

        if h:
            h_n = str(h['n_markets'])
            h_win = f"{h['win_rate']:.1%}"
            h_imp = f"{h['improvement']:+.1%}"
            h_sig = "YES" if h['significant'] else "no"
        else:
            h_n = h_win = h_imp = h_sig = "-"

        if c:
            c_n = str(c['n_markets'])
            c_win = f"{c['win_rate']:.1%}"
            c_imp = f"{c['improvement']:+.1%}"
            c_sig = "YES" if c['significant'] else "no"
        else:
            c_n = c_win = c_imp = c_sig = "-"

        lines.append(
            f"| {bucket:6} | {h_n:6} | {h_win:8} | {h_imp:12} | {h_sig:8} | "
            f"{c_n:9} | {c_win:11} | {c_imp:15} | {c_sig:11} |"
        )

    return "\n".join(lines)


def analyze_bucket_performance(results: Dict[str, Any], strategy_name: str) -> str:
    """Analyze which buckets beat baseline."""
    lines = [f"\n### {strategy_name} Bucket Analysis"]

    beats_baseline = []
    loses_to_baseline = []
    significant_wins = []

    for b in results['bucket_results']:
        if b['improvement'] > 0:
            beats_baseline.append(b)
            if b['significant']:
                significant_wins.append(b)
        else:
            loses_to_baseline.append(b)

    lines.append(f"\n**Buckets that beat baseline**: {len(beats_baseline)}/{len(results['bucket_results'])}")
    if beats_baseline:
        bucket_list = ", ".join(b['bucket'] for b in beats_baseline)
        lines.append(f"  - Buckets: {bucket_list}")

    lines.append(f"\n**Statistically significant improvements (p<0.05)**: {len(significant_wins)}/{len(results['bucket_results'])}")
    if significant_wins:
        bucket_list = ", ".join(b['bucket'] for b in significant_wins)
        lines.append(f"  - Buckets: {bucket_list}")

    lines.append(f"\n**Buckets that lose to baseline**: {len(loses_to_baseline)}/{len(results['bucket_results'])}")
    if loses_to_baseline:
        for b in loses_to_baseline:
            lines.append(f"  - {b['bucket']}: {b['improvement']:+.1%} (N={b['n_markets']})")

    # Find sweet spot (best improvement with N >= 20)
    valid_buckets = [b for b in results['bucket_results'] if b['n_markets'] >= 20]
    if valid_buckets:
        best = max(valid_buckets, key=lambda x: x['improvement'])
        lines.append(f"\n**Sweet spot (N>=20)**: {best['bucket']} with {best['improvement']:+.1%} improvement (N={best['n_markets']})")

    return "\n".join(lines)


def main():
    print("=" * 80)
    print("H123 vs Current Parameters: Full Bucket Comparison")
    print("=" * 80)

    # Load data
    df = load_data()

    # Build baseline
    all_markets, baseline = build_baseline(df)

    # Analyze H123 parameters (5 trades, any drop, 70% YES)
    print("\nAnalyzing H123 parameters (5 trades, any drop, 70% YES)...")
    h123_rlm, _ = get_rlm_markets(df, yes_trade_threshold=0.70, min_trades=5, price_move_threshold=0)
    print(f"  H123 signal markets: {len(h123_rlm):,}")

    h123_results = analyze_rlm_strategy(
        rlm_markets=h123_rlm,
        baseline=baseline,
        strategy_name="H123",
        parameters={'yes_threshold': 0.70, 'min_trades': 5, 'min_price_drop': 0},
    )

    # Analyze Current parameters (25 trades, 2c drop, 70% YES)
    print("\nAnalyzing Current parameters (25 trades, 2c drop, 70% YES)...")
    current_rlm, _ = get_rlm_markets(df, yes_trade_threshold=0.70, min_trades=25, price_move_threshold=2)
    print(f"  Current signal markets: {len(current_rlm):,}")

    current_results = analyze_rlm_strategy(
        rlm_markets=current_rlm,
        baseline=baseline,
        strategy_name="Current",
        parameters={'yes_threshold': 0.70, 'min_trades': 25, 'min_price_drop': 2},
    )

    # Build report
    report_lines = [
        "# H123 vs Current RLM Parameters: Full Bucket Comparison",
        f"\nGenerated: {datetime.now().isoformat()}",
        "",
        "## Executive Summary",
        "",
        f"### H123 (5 trades, any drop, 70% YES)",
        f"- Total Markets: {h123_results['overall']['total_markets']}",
        f"- Overall Win Rate: {h123_results['overall']['win_rate']:.1%}",
        f"- Weighted Baseline: {h123_results['overall']['weighted_baseline']:.1%}",
        f"- Overall Improvement: {h123_results['overall']['overall_improvement']:+.1%}",
        "",
        f"### Current (25 trades, 2c drop, 70% YES)",
        f"- Total Markets: {current_results['overall']['total_markets']}",
        f"- Overall Win Rate: {current_results['overall']['win_rate']:.1%}",
        f"- Weighted Baseline: {current_results['overall']['weighted_baseline']:.1%}",
        f"- Overall Improvement: {current_results['overall']['overall_improvement']:+.1%}",
    ]

    # Add Table 1
    report_lines.append(format_table(h123_results, "Table 1: H123 Parameters (5 trades, any drop)"))

    # Add Table 2
    report_lines.append(format_table(current_results, "Table 2: Current Parameters (25 trades, 2c drop)"))

    # Add Table 3 comparison
    report_lines.append(format_comparison_table(h123_results, current_results))

    # Add bucket analysis
    report_lines.append("\n## Bucket Performance Analysis")
    report_lines.append(analyze_bucket_performance(h123_results, "H123"))
    report_lines.append(analyze_bucket_performance(current_results, "Current"))

    # Add key insights
    report_lines.extend([
        "\n## Key Insights",
        "",
        "### 1. Trade-off: Volume vs Purity",
        f"- H123 generates {h123_results['overall']['total_markets']} signals vs Current's {current_results['overall']['total_markets']} ({h123_results['overall']['total_markets'] / current_results['overall']['total_markets']:.1f}x more)",
        f"- Current has higher win rate ({current_results['overall']['win_rate']:.1%} vs {h123_results['overall']['win_rate']:.1%})",
        f"- But H123 has higher improvement over baseline ({h123_results['overall']['overall_improvement']:+.1%} vs {current_results['overall']['overall_improvement']:+.1%})",
        "",
        "### 2. Price Range Comparison",
    ])

    # Calculate average NO price for each
    h123_avg_price = sum(b['avg_no_price'] * b['n_markets'] for b in h123_results['bucket_results']) / h123_results['overall']['total_markets']
    current_avg_price = sum(b['avg_no_price'] * b['n_markets'] for b in current_results['bucket_results']) / current_results['overall']['total_markets']

    report_lines.extend([
        f"- H123 average NO price: {h123_avg_price:.1f}c",
        f"- Current average NO price: {current_avg_price:.1f}c",
        f"- H123 trades at lower prices, capturing more alpha from mispriced markets",
        "",
        "### 3. Statistical Significance by Bucket",
    ])

    h123_sig = sum(1 for b in h123_results['bucket_results'] if b['significant'])
    current_sig = sum(1 for b in current_results['bucket_results'] if b['significant'])

    report_lines.extend([
        f"- H123: {h123_sig}/{len(h123_results['bucket_results'])} buckets significant",
        f"- Current: {current_sig}/{len(current_results['bucket_results'])} buckets significant",
    ])

    # Print report
    report = "\n".join(report_lines)
    print(report)

    # Save report
    report_path = REPORTS_DIR / "h123_vs_current_bucket_comparison.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n\nReport saved to: {report_path}")

    # Save JSON for programmatic access
    json_path = REPORTS_DIR / "h123_vs_current_bucket_comparison.json"
    with open(json_path, 'w') as f:
        json.dump({
            'h123': h123_results,
            'current': current_results,
            'generated_at': datetime.now().isoformat(),
        }, f, indent=2)
    print(f"JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
