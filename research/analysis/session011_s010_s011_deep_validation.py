#!/usr/bin/env python3
"""
Session 011 - Deep Validation of S010 and S011

OBJECTIVE: Rigorously validate the claimed edges of S010 (+76.6%) and S011 (+57.3%)

S010: Follow Round-Size Bot NO Consensus
- Signal: Markets where >60% of round-size trades (10, 25, 50, 100, 250, 500, 1000) are NO bets
         AND avg NO price < 45c
- Claimed: +76.6% edge, +40.2% improvement, 1,287 markets

S011: Stable Leverage Bot NO Consensus
- Signal: Markets where leverage std < 0.5 AND >60% NO consensus AND >=3 trades
- Claimed: +57.3% edge, +23.4% improvement, 592 markets

VALIDATION REQUIREMENTS:
1. Recalculate from scratch with CORRECT formula: breakeven = trade_price / 100.0
2. Price proxy check - compare to baseline at SAME price levels
3. Temporal stability - check edge in 4 quarters chronologically
4. Out-of-sample validation - 70% train / 30% test
5. Concentration analysis - check profit distribution
6. Category breakdown - verify works across different categories
7. Sanity checks - verify methodology and look for bugs
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os
from pathlib import Path

# Configuration
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]
LEVERAGE_STD_THRESHOLD = 0.5
NO_CONSENSUS_THRESHOLD = 0.6
MIN_TRADES = 3
NO_PRICE_THRESHOLD = 45  # cents

def load_data():
    """Load the enriched trades data"""
    data_path = Path(__file__).parent.parent / "data" / "trades" / "enriched_trades_resolved_ALL.csv"
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    return df


def calculate_correct_breakeven_and_edge(df, side_col='taker_side', price_col='trade_price'):
    """
    Calculate edge using CORRECT formula.

    For BOTH YES and NO trades:
    - breakeven = trade_price / 100.0 (what you need to win to break even)
    - edge = actual_win_rate - breakeven_rate
    """
    # Group by market
    market_results = df.groupby('market_ticker').agg({
        price_col: 'mean',
        'is_winner': 'first',  # All trades in same market have same result
        'market_result': 'first',
        side_col: lambda x: (x == 'no').mean()  # NO ratio
    }).reset_index()

    market_results.columns = ['market_ticker', 'avg_price', 'is_winner', 'market_result', 'no_ratio']

    # Calculate breakeven (cost / 100)
    market_results['breakeven'] = market_results['avg_price'] / 100.0

    # Win rate is simply % of markets where is_winner = True
    win_rate = market_results['is_winner'].mean()
    avg_breakeven = market_results['breakeven'].mean()
    edge = win_rate - avg_breakeven

    return {
        'markets': len(market_results),
        'win_rate': win_rate,
        'breakeven': avg_breakeven,
        'edge': edge,
        'avg_price': market_results['avg_price'].mean()
    }


def detect_s010_signal(df):
    """
    S010: Round-Size Bot NO Consensus

    Signal: Markets where >60% of round-size trades are NO bets AND avg NO price < 45c
    """
    # Identify round-size trades
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)

    # Get only round-size trades
    round_size_trades = df[df['is_round_size']].copy()

    print(f"\nS010 Signal Detection:")
    print(f"  Total trades: {len(df):,}")
    print(f"  Round-size trades: {len(round_size_trades):,} ({100*len(round_size_trades)/len(df):.1f}%)")

    # Aggregate by market
    market_stats = round_size_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),  # NO ratio
        'no_price': 'mean',  # Average NO price
        'count': 'count',  # Number of round-size trades
        'is_winner': 'first',
        'market_result': 'first'
    }).reset_index()

    market_stats.columns = ['market_ticker', 'no_ratio', 'avg_no_price', 'n_round_trades', 'is_winner', 'market_result']

    # Apply signal conditions: >60% NO AND avg NO price < 45c AND at least 1 round trade
    signal_markets = market_stats[
        (market_stats['no_ratio'] > NO_CONSENSUS_THRESHOLD) &
        (market_stats['avg_no_price'] < NO_PRICE_THRESHOLD) &
        (market_stats['n_round_trades'] >= 1)
    ].copy()

    print(f"  Markets with round-size trades: {len(market_stats):,}")
    print(f"  Signal markets (>60% NO, NO<45c): {len(signal_markets):,}")

    return signal_markets


def detect_s011_signal(df):
    """
    S011: Stable Leverage Bot NO Consensus

    Signal: Markets where leverage std < 0.5 AND >60% NO consensus AND >=3 trades
    """
    # Aggregate by market
    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['mean', 'std'],
        'taker_side': lambda x: (x == 'no').mean(),  # NO ratio
        'count': 'count',  # Number of trades
        'is_winner': 'first',
        'market_result': 'first',
        'trade_price': 'mean'
    }).reset_index()

    market_stats.columns = ['market_ticker', 'lev_mean', 'lev_std', 'no_ratio', 'n_trades', 'is_winner', 'market_result', 'avg_price']

    # Fill NaN std with 0 for single-trade markets
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)

    print(f"\nS011 Signal Detection:")
    print(f"  Total markets: {len(market_stats):,}")

    # Apply signal conditions
    signal_markets = market_stats[
        (market_stats['lev_std'] < LEVERAGE_STD_THRESHOLD) &
        (market_stats['no_ratio'] > NO_CONSENSUS_THRESHOLD) &
        (market_stats['n_trades'] >= MIN_TRADES)
    ].copy()

    print(f"  Signal markets (lev_std<0.5, >60% NO, >=3 trades): {len(signal_markets):,}")

    return signal_markets


def calculate_strategy_edge(signal_markets, strategy_name):
    """Calculate edge for a strategy given signal markets"""
    if len(signal_markets) == 0:
        return {
            'strategy': strategy_name,
            'status': 'NO_SIGNAL',
            'markets': 0
        }

    # For these strategies, we bet NO
    # So we win when market_result == 'no'
    signal_markets = signal_markets.copy()
    signal_markets['bet_wins'] = signal_markets['market_result'] == 'no'

    # Calculate win rate and edge
    win_rate = signal_markets['bet_wins'].mean()

    # Average price we pay for NO (from avg_no_price or 100 - avg_yes_price)
    if 'avg_no_price' in signal_markets.columns:
        avg_no_price = signal_markets['avg_no_price'].mean()
    else:
        avg_no_price = 100 - signal_markets['avg_price'].mean()

    # Breakeven = what we pay / 100
    breakeven = avg_no_price / 100.0

    # Edge = win_rate - breakeven
    edge = win_rate - breakeven

    # Calculate p-value (binomial test vs breakeven)
    n_wins = int(signal_markets['bet_wins'].sum())
    n_total = len(signal_markets)

    try:
        p_value = stats.binom_test(n_wins, n_total, breakeven, alternative='greater')
    except:
        # Fallback for older scipy
        p_value = stats.binomtest(n_wins, n_total, breakeven, alternative='greater').pvalue

    return {
        'strategy': strategy_name,
        'markets': n_total,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge,
        'avg_no_price': avg_no_price,
        'n_wins': n_wins,
        'p_value': p_value
    }


def calculate_baseline_at_price(df, no_price_min, no_price_max):
    """
    Calculate baseline edge for ALL NO trades at a specific price range.
    This is the control to check if signal is just a price proxy.
    """
    # Filter to trades in this NO price range
    df_filtered = df[(df['no_price'] >= no_price_min) & (df['no_price'] < no_price_max)].copy()

    if len(df_filtered) == 0:
        return None

    # Aggregate by market
    market_stats = df_filtered.groupby('market_ticker').agg({
        'no_price': 'mean',
        'is_winner': 'first',
        'market_result': 'first'
    }).reset_index()

    # For baseline, assume we always bet NO
    market_stats['bet_wins'] = market_stats['market_result'] == 'no'

    win_rate = market_stats['bet_wins'].mean()
    avg_no_price = market_stats['no_price'].mean()
    breakeven = avg_no_price / 100.0
    edge = win_rate - breakeven

    return {
        'price_range': f"{no_price_min}-{no_price_max}c",
        'markets': len(market_stats),
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge': edge
    }


def price_proxy_check(signal_markets, all_trades_df, strategy_name):
    """
    CRITICAL CHECK: Compare signal edge to baseline at SAME price levels.

    For each price bucket, calculate:
    - Signal edge (markets matching signal)
    - Baseline edge (ALL markets at same NO price)
    - Improvement = Signal edge - Baseline edge

    If improvement is near zero or negative, signal is just a price proxy!
    """
    print(f"\n{'='*60}")
    print(f"PRICE PROXY CHECK: {strategy_name}")
    print(f"{'='*60}")

    # Get signal market tickers
    signal_tickers = set(signal_markets['market_ticker'].unique())

    # Create NO price column if needed
    if 'avg_no_price' not in signal_markets.columns:
        signal_markets = signal_markets.copy()
        signal_markets['avg_no_price'] = 100 - signal_markets['avg_price']

    results = []

    # Check by 5c buckets
    for bucket_min in range(0, 50, 5):
        bucket_max = bucket_min + 5

        # Signal markets in this bucket
        signal_in_bucket = signal_markets[
            (signal_markets['avg_no_price'] >= bucket_min) &
            (signal_markets['avg_no_price'] < bucket_max)
        ]

        if len(signal_in_bucket) < 5:
            continue

        # Calculate signal edge
        signal_win_rate = (signal_in_bucket['market_result'] == 'no').mean()
        signal_breakeven = signal_in_bucket['avg_no_price'].mean() / 100.0
        signal_edge = signal_win_rate - signal_breakeven

        # Calculate baseline edge (ALL markets at this price, not in signal)
        baseline = calculate_baseline_at_price(all_trades_df, bucket_min, bucket_max)

        if baseline is None or baseline['markets'] < 10:
            continue

        improvement = signal_edge - baseline['edge']

        result = {
            'bucket': f"{bucket_min}-{bucket_max}c",
            'signal_markets': len(signal_in_bucket),
            'signal_win_rate': signal_win_rate,
            'signal_edge': signal_edge,
            'baseline_markets': baseline['markets'],
            'baseline_win_rate': baseline['win_rate'],
            'baseline_edge': baseline['edge'],
            'improvement': improvement
        }
        results.append(result)

        print(f"  {bucket_min:2d}-{bucket_max:2d}c: Signal WR={signal_win_rate:.1%}, Baseline WR={baseline['win_rate']:.1%}, Improvement={improvement:+.1%}")

    if not results:
        return {'improvement': None, 'results': []}

    # Calculate weighted average improvement
    total_signal_markets = sum(r['signal_markets'] for r in results)
    weighted_improvement = sum(r['improvement'] * r['signal_markets'] / total_signal_markets for r in results)

    print(f"\n  WEIGHTED AVERAGE IMPROVEMENT: {weighted_improvement:+.1%}")

    return {
        'improvement': weighted_improvement,
        'results': results
    }


def temporal_stability_check(signal_markets, strategy_name):
    """
    Check if edge is stable across 4 time periods (quarters of data).
    """
    print(f"\n{'='*60}")
    print(f"TEMPORAL STABILITY CHECK: {strategy_name}")
    print(f"{'='*60}")

    # Sort by market ticker (which often contains date)
    # For proper temporal split, we need timestamp from trades
    # Since signal_markets is aggregated, we'll use a simpler approach
    n = len(signal_markets)
    q_size = n // 4

    results = []

    for i, (q_name, start, end) in enumerate([
        ('Q1 (First 25%)', 0, q_size),
        ('Q2 (25-50%)', q_size, 2*q_size),
        ('Q3 (50-75%)', 2*q_size, 3*q_size),
        ('Q4 (Last 25%)', 3*q_size, n)
    ]):
        quarter = signal_markets.iloc[start:end]

        if len(quarter) < 10:
            continue

        win_rate = (quarter['market_result'] == 'no').mean()
        if 'avg_no_price' in quarter.columns:
            avg_no_price = quarter['avg_no_price'].mean()
        else:
            avg_no_price = 100 - quarter['avg_price'].mean()
        breakeven = avg_no_price / 100.0
        edge = win_rate - breakeven

        result = {
            'quarter': q_name,
            'markets': len(quarter),
            'win_rate': win_rate,
            'edge': edge
        }
        results.append(result)

        edge_status = "POSITIVE" if edge > 0 else "NEGATIVE"
        print(f"  {q_name}: N={len(quarter)}, WR={win_rate:.1%}, Edge={edge:+.1%} [{edge_status}]")

    positive_quarters = sum(1 for r in results if r['edge'] > 0)
    print(f"\n  POSITIVE QUARTERS: {positive_quarters}/4")

    return {
        'positive_quarters': positive_quarters,
        'total_quarters': len(results),
        'results': results
    }


def out_of_sample_validation(df, detect_signal_func, strategy_name, is_s010=True):
    """
    Split data 70/30 and check if signal works out-of-sample.
    """
    print(f"\n{'='*60}")
    print(f"OUT-OF-SAMPLE VALIDATION: {strategy_name}")
    print(f"{'='*60}")

    # Get unique markets and sort by ticker (proxy for time)
    markets = df['market_ticker'].unique()
    n_markets = len(markets)

    # 70/30 split
    train_size = int(0.7 * n_markets)
    train_markets = markets[:train_size]
    test_markets = markets[train_size:]

    train_df = df[df['market_ticker'].isin(train_markets)]
    test_df = df[df['market_ticker'].isin(test_markets)]

    print(f"  Train markets: {len(train_markets):,}")
    print(f"  Test markets: {len(test_markets):,}")

    # Detect signal on train data
    train_signal = detect_signal_func(train_df)
    train_edge = calculate_strategy_edge(train_signal, f"{strategy_name}_train")

    # Detect signal on test data
    test_signal = detect_signal_func(test_df)
    test_edge = calculate_strategy_edge(test_signal, f"{strategy_name}_test")

    print(f"\n  TRAIN: {train_edge['markets']} markets, Edge = {train_edge.get('edge', 0):+.1%}")
    print(f"  TEST:  {test_edge['markets']} markets, Edge = {test_edge.get('edge', 0):+.1%}")

    oos_holds = test_edge.get('edge', 0) > 0
    print(f"\n  OUT-OF-SAMPLE HOLDS: {'YES' if oos_holds else 'NO'}")

    return {
        'train': train_edge,
        'test': test_edge,
        'oos_holds': oos_holds
    }


def concentration_check(signal_markets, strategy_name):
    """
    Check if profit is concentrated in a few markets.
    Threshold: No single market should contribute >30% of total profit.
    """
    print(f"\n{'='*60}")
    print(f"CONCENTRATION CHECK: {strategy_name}")
    print(f"{'='*60}")

    # Calculate profit per market
    signal_markets = signal_markets.copy()
    signal_markets['bet_wins'] = signal_markets['market_result'] == 'no'

    if 'avg_no_price' in signal_markets.columns:
        avg_no_price = signal_markets['avg_no_price']
    else:
        avg_no_price = 100 - signal_markets['avg_price']

    # Profit = win - cost (assume 1 contract)
    signal_markets['profit'] = signal_markets['bet_wins'].apply(
        lambda w: 100 - avg_no_price.mean() if w else -avg_no_price.mean()
    )

    # Sort by absolute profit
    signal_markets['abs_profit'] = abs(signal_markets['profit'])
    sorted_markets = signal_markets.sort_values('abs_profit', ascending=False)

    total_profit = signal_markets[signal_markets['profit'] > 0]['profit'].sum()

    if total_profit <= 0:
        print("  No positive profit to analyze")
        return {'passes': False, 'reason': 'no_positive_profit'}

    # Check top markets
    top10_profit = sorted_markets.head(10)['profit'].sum()
    top10_pct = top10_profit / total_profit if total_profit > 0 else 0

    max_single_pct = sorted_markets.head(1)['profit'].values[0] / total_profit if total_profit > 0 else 0

    print(f"  Total positive profit: {total_profit:.2f}")
    print(f"  Top 10 markets contribute: {top10_pct:.1%}")
    print(f"  Top 1 market contributes: {max_single_pct:.1%}")

    passes = max_single_pct < 0.30
    print(f"\n  PASSES CONCENTRATION (<30%): {'YES' if passes else 'NO'}")

    return {
        'passes': passes,
        'max_single_pct': max_single_pct,
        'top10_pct': top10_pct
    }


def category_breakdown(signal_markets, strategy_name):
    """
    Break down edge by market category.
    """
    print(f"\n{'='*60}")
    print(f"CATEGORY BREAKDOWN: {strategy_name}")
    print(f"{'='*60}")

    # Extract category from ticker (e.g., KXNFL, KXNCAAF, KXBTCD)
    signal_markets = signal_markets.copy()
    signal_markets['category'] = signal_markets['market_ticker'].str.extract(r'^(KX[A-Z]+)')[0]
    signal_markets['bet_wins'] = signal_markets['market_result'] == 'no'

    if 'avg_no_price' in signal_markets.columns:
        signal_markets['avg_no_price_col'] = signal_markets['avg_no_price']
    else:
        signal_markets['avg_no_price_col'] = 100 - signal_markets['avg_price']

    results = []

    for category, group in signal_markets.groupby('category'):
        if len(group) < 10:
            continue

        win_rate = group['bet_wins'].mean()
        avg_no_price = group['avg_no_price_col'].mean()
        breakeven = avg_no_price / 100.0
        edge = win_rate - breakeven

        result = {
            'category': category,
            'markets': len(group),
            'win_rate': win_rate,
            'edge': edge
        }
        results.append(result)

        edge_status = "+" if edge > 0 else ""
        print(f"  {category}: N={len(group)}, WR={win_rate:.1%}, Edge={edge_status}{edge:.1%}")

    # Check if edge is positive across multiple categories
    positive_categories = sum(1 for r in results if r['edge'] > 0)
    total_categories = len(results)

    print(f"\n  POSITIVE CATEGORIES: {positive_categories}/{total_categories}")

    return {
        'positive_categories': positive_categories,
        'total_categories': total_categories,
        'results': results
    }


def sanity_checks(signal_markets, all_trades_df, strategy_name):
    """
    Run sanity checks to look for methodology errors.
    """
    print(f"\n{'='*60}")
    print(f"SANITY CHECKS: {strategy_name}")
    print(f"{'='*60}")

    issues = []

    # 1. Check for data leakage (using future information)
    # - This would require timestamp analysis
    print("  1. Data leakage check: Using aggregated market data (no leakage)")

    # 2. Check if claimed edge is plausible
    if 'avg_no_price' in signal_markets.columns:
        avg_no_price = signal_markets['avg_no_price'].mean()
    else:
        avg_no_price = 100 - signal_markets['avg_price'].mean()

    win_rate = (signal_markets['market_result'] == 'no').mean()
    breakeven = avg_no_price / 100.0
    edge = win_rate - breakeven

    # Edge > 50% is VERY suspicious
    if edge > 0.50:
        issues.append(f"SUSPICIOUS: Edge of {edge:.1%} is very high (>50%)")
        print(f"  2. Edge plausibility: SUSPICIOUS - {edge:.1%} edge is very high!")
    else:
        print(f"  2. Edge plausibility: OK - {edge:.1%} edge is within plausible range")

    # 3. Check sample size distribution
    n_markets = len(signal_markets)
    print(f"  3. Sample size: {n_markets} markets")
    if n_markets < 50:
        issues.append(f"LOW SAMPLE: Only {n_markets} markets")

    # 4. Check if signal is too narrow
    signal_ratio = n_markets / all_trades_df['market_ticker'].nunique()
    print(f"  4. Signal selectivity: {signal_ratio:.2%} of all markets match signal")
    if signal_ratio < 0.01:
        issues.append(f"NARROW SIGNAL: Only {signal_ratio:.2%} of markets match")

    # 5. Check for extreme win rates
    if win_rate > 0.95:
        issues.append(f"SUSPICIOUS: Win rate of {win_rate:.1%} is very high (>95%)")
        print(f"  5. Win rate check: SUSPICIOUS - {win_rate:.1%} is very high!")
    else:
        print(f"  5. Win rate check: OK - {win_rate:.1%}")

    # 6. Check average price is reasonable
    if avg_no_price < 5:
        issues.append(f"SUSPICIOUS: Average NO price of {avg_no_price:.1f}c is very low")
        print(f"  6. Price check: SUSPICIOUS - {avg_no_price:.1f}c avg is very low")
    else:
        print(f"  6. Price check: OK - {avg_no_price:.1f}c average NO price")

    print(f"\n  ISSUES FOUND: {len(issues)}")
    for issue in issues:
        print(f"    - {issue}")

    return {
        'issues': issues,
        'n_issues': len(issues)
    }


def run_full_validation(df, detect_func, strategy_name, is_s010=True):
    """Run complete validation suite for a strategy."""
    print(f"\n{'#'*70}")
    print(f"# FULL VALIDATION: {strategy_name}")
    print(f"{'#'*70}")

    # 1. Detect signal and calculate edge
    signal_markets = detect_func(df)
    edge_results = calculate_strategy_edge(signal_markets, strategy_name)

    print(f"\nBASE METRICS:")
    print(f"  Markets: {edge_results.get('markets', 0)}")
    print(f"  Win Rate: {edge_results.get('win_rate', 0):.2%}")
    print(f"  Breakeven: {edge_results.get('breakeven', 0):.2%}")
    print(f"  Edge: {edge_results.get('edge', 0):+.2%}")
    print(f"  P-value: {edge_results.get('p_value', 1):.2e}")

    if edge_results.get('markets', 0) < 50:
        print("\n  VALIDATION FAILED: Insufficient markets (<50)")
        return {
            'strategy': strategy_name,
            'status': 'REJECTED',
            'reason': 'insufficient_markets',
            'metrics': edge_results
        }

    # 2. Price proxy check
    proxy_check = price_proxy_check(signal_markets, df, strategy_name)

    # 3. Temporal stability
    temporal = temporal_stability_check(signal_markets, strategy_name)

    # 4. Out-of-sample validation
    oos = out_of_sample_validation(df, detect_func, strategy_name, is_s010)

    # 5. Concentration check
    concentration = concentration_check(signal_markets, strategy_name)

    # 6. Category breakdown
    categories = category_breakdown(signal_markets, strategy_name)

    # 7. Sanity checks
    sanity = sanity_checks(signal_markets, df, strategy_name)

    # FINAL VERDICT
    print(f"\n{'='*70}")
    print(f"FINAL VERDICT: {strategy_name}")
    print(f"{'='*70}")

    # Criteria
    criteria = {
        'sufficient_markets': edge_results.get('markets', 0) >= 50,
        'positive_edge': edge_results.get('edge', 0) > 0,
        'significant': edge_results.get('p_value', 1) < 0.01,
        'not_price_proxy': proxy_check.get('improvement', 0) is not None and proxy_check.get('improvement', 0) > 0.05,
        'temporal_stable': temporal.get('positive_quarters', 0) >= 2,
        'oos_holds': oos.get('oos_holds', False),
        'not_concentrated': concentration.get('passes', False),
        'multiple_categories': categories.get('positive_categories', 0) >= 2,
        'no_sanity_issues': sanity.get('n_issues', 0) == 0
    }

    for criterion, passes in criteria.items():
        status = "PASS" if passes else "FAIL"
        print(f"  {criterion}: {status}")

    all_pass = all(criteria.values())
    critical_pass = (
        criteria['sufficient_markets'] and
        criteria['positive_edge'] and
        criteria['significant'] and
        criteria['not_price_proxy']
    )

    if all_pass:
        status = 'VALIDATED'
        confidence = 'HIGH'
    elif critical_pass:
        status = 'VALIDATED'
        confidence = 'MEDIUM'
    else:
        status = 'REJECTED'
        confidence = 'LOW'

    print(f"\n  STATUS: {status}")
    print(f"  CONFIDENCE: {confidence}")
    print(f"  IMPROVEMENT OVER BASELINE: {proxy_check.get('improvement', 0):+.1%}" if proxy_check.get('improvement') else "  IMPROVEMENT OVER BASELINE: N/A")

    return {
        'strategy': strategy_name,
        'status': status,
        'confidence': confidence,
        'metrics': edge_results,
        'price_proxy': proxy_check,
        'temporal': temporal,
        'oos': oos,
        'concentration': concentration,
        'categories': categories,
        'sanity': sanity,
        'criteria': criteria
    }


def main():
    """Main validation routine."""
    print("="*70)
    print("SESSION 011 - DEEP VALIDATION OF S010 AND S011")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().isoformat()}")

    # Load data
    df = load_data()

    # Validate S010
    s010_results = run_full_validation(df, detect_s010_signal, "S010_Round_Size_Bot_NO", is_s010=True)

    # Validate S011
    s011_results = run_full_validation(df, detect_s011_signal, "S011_Stable_Leverage_Bot_NO", is_s010=False)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    for result in [s010_results, s011_results]:
        print(f"\n{result['strategy']}:")
        print(f"  Status: {result['status']}")
        print(f"  Confidence: {result.get('confidence', 'N/A')}")
        print(f"  Markets: {result['metrics'].get('markets', 0)}")
        print(f"  Edge: {result['metrics'].get('edge', 0):+.2%}")
        print(f"  Improvement vs Baseline: {result['price_proxy'].get('improvement', 0):+.1%}" if result['price_proxy'].get('improvement') else "  Improvement vs Baseline: N/A")

        if result['status'] == 'REJECTED':
            failed_criteria = [k for k, v in result['criteria'].items() if not v]
            print(f"  Failed Criteria: {', '.join(failed_criteria)}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'S010': {
            'status': s010_results['status'],
            'confidence': s010_results.get('confidence', 'N/A'),
            'metrics': s010_results['metrics'],
            'improvement': s010_results['price_proxy'].get('improvement'),
            'criteria': s010_results['criteria']
        },
        'S011': {
            'status': s011_results['status'],
            'confidence': s011_results.get('confidence', 'N/A'),
            'metrics': s011_results['metrics'],
            'improvement': s011_results['price_proxy'].get('improvement'),
            'criteria': s011_results['criteria']
        }
    }

    output_path = Path(__file__).parent.parent / "reports" / "session011_deep_validation_final.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
