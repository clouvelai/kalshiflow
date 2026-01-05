#!/usr/bin/env python3
"""
Session 011 - Deep Validation of S010 and S011 (VERSION 2)

CRITICAL FIX: The original S010 claim uses ROUND-SIZE TRADES to detect
markets with bot NO consensus. When >60% of round-size trades are NO,
we follow them and bet NO.

The win condition for betting NO is: market_result == 'no'

S010: Follow Round-Size Bot NO Consensus
- Signal: Markets where >60% of round-size trades (10, 25, 50, 100, 250, 500, 1000) are NO bets
         AND avg NO price of those trades < 45c
- Claimed: +76.6% edge, +40.2% improvement, 1,287 markets

S011: Stable Leverage Bot NO Consensus
- Signal: Markets where leverage std < 0.5 AND >60% NO consensus AND >=3 trades
- Claimed: +57.3% edge, +23.4% improvement, 592 markets
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]
LEVERAGE_STD_THRESHOLD = 0.5
NO_CONSENSUS_THRESHOLD = 0.6
MIN_TRADES = 3
NO_PRICE_THRESHOLD = 45

def load_data():
    """Load the enriched trades data"""
    data_path = Path(__file__).parent.parent / "data" / "trades" / "enriched_trades_resolved_ALL.csv"
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} trades across {df['market_ticker'].nunique():,} markets")
    return df


def validate_s010_detailed(df):
    """
    Validate S010: Round-Size Bot NO Consensus

    EXACT methodology from original analysis:
    1. Identify round-size trades (10, 25, 50, 100, 250, 500, 1000)
    2. For each market, calculate: % of round-size trades that are NO
    3. Signal: >60% are NO AND avg NO price of those trades < 45c
    4. Action: Bet NO
    5. Win: market_result == 'no'
    """
    print("\n" + "="*70)
    print("S010 DETAILED VALIDATION")
    print("="*70)

    # Step 1: Identify round-size trades
    df = df.copy()
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    round_size_trades = df[df['is_round_size']].copy()

    print(f"\nStep 1: Round-size trade identification")
    print(f"  Total trades: {len(df):,}")
    print(f"  Round-size trades: {len(round_size_trades):,} ({100*len(round_size_trades)/len(df):.1f}%)")

    # Step 2: Calculate NO ratio per market (among round-size trades only)
    round_market_stats = round_size_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),  # % NO trades
        'no_price': 'mean',  # Average NO price among round-size trades
        'count': 'count',  # Number of round-size trades
        'market_result': 'first',
    }).reset_index()
    round_market_stats.columns = ['market_ticker', 'no_ratio', 'avg_no_price', 'n_round_trades', 'market_result']

    print(f"\nStep 2: Market-level aggregation of round-size trades")
    print(f"  Markets with round-size trades: {len(round_market_stats):,}")

    # Step 3: Apply signal filters
    # >60% NO AND avg NO price < 45c
    signal_markets = round_market_stats[
        (round_market_stats['no_ratio'] > NO_CONSENSUS_THRESHOLD) &
        (round_market_stats['avg_no_price'] < NO_PRICE_THRESHOLD)
    ].copy()

    print(f"\nStep 3: Signal detection")
    print(f"  Markets with >60% NO in round-size trades: {len(round_market_stats[round_market_stats['no_ratio'] > NO_CONSENSUS_THRESHOLD]):,}")
    print(f"  Markets with avg NO price < 45c: {len(round_market_stats[round_market_stats['avg_no_price'] < NO_PRICE_THRESHOLD]):,}")
    print(f"  Signal markets (BOTH conditions): {len(signal_markets):,}")

    if len(signal_markets) == 0:
        return {'status': 'NO_SIGNAL', 'markets': 0}

    # Step 4: Calculate edge
    # We bet NO, so we win when market_result == 'no'
    signal_markets['bet_wins'] = (signal_markets['market_result'] == 'no').astype(int)

    win_rate = signal_markets['bet_wins'].mean()
    avg_no_price = signal_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100.0
    edge = (win_rate - breakeven) * 100  # As percentage points

    n_wins = signal_markets['bet_wins'].sum()
    n_total = len(signal_markets)

    # P-value
    binom_result = stats.binomtest(n_wins, n_total, breakeven, alternative='greater')
    p_value = binom_result.pvalue

    print(f"\nStep 4: Edge calculation")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Avg NO Price: {avg_no_price:.2f}c")
    print(f"  Breakeven: {breakeven:.2%}")
    print(f"  Edge: {edge:+.2f}%")
    print(f"  P-value: {p_value:.2e}")

    # Step 5: Price proxy check
    print(f"\nStep 5: Price Proxy Check")

    # Get ALL markets (not just round-size signal)
    all_market_agg = df.groupby('market_ticker').agg({
        'no_price': 'mean',
        'market_result': 'first'
    }).reset_index()
    all_market_agg['bet_wins'] = (all_market_agg['market_result'] == 'no').astype(int)

    # Price buckets
    signal_markets['price_bucket'] = (signal_markets['avg_no_price'] // 5) * 5
    all_market_agg['price_bucket'] = (all_market_agg['no_price'] // 5) * 5

    print(f"\n  {'Bucket':<10} {'Signal':<20} {'ALL Markets':<20} {'Improvement':<12}")
    print(f"  {'(NO price)':<10} {'WR (N)':<20} {'WR (N)':<20} {'(pp)':<12}")
    print("  " + "-"*65)

    improvements = []
    for bucket in sorted(signal_markets['price_bucket'].unique()):
        sig_bucket = signal_markets[signal_markets['price_bucket'] == bucket]
        all_bucket = all_market_agg[all_market_agg['price_bucket'] == bucket]

        if len(sig_bucket) >= 5 and len(all_bucket) >= 10:
            sig_wr = sig_bucket['bet_wins'].mean()
            all_wr = all_bucket['bet_wins'].mean()
            improvement = (sig_wr - all_wr) * 100

            print(f"  {int(bucket):2d}-{int(bucket)+5}c     {sig_wr:.1%} ({len(sig_bucket)})    {all_wr:.1%} ({len(all_bucket)})    {improvement:+.1f}%")

            improvements.append({'bucket': bucket, 'sig_n': len(sig_bucket), 'improvement': improvement})

    if improvements:
        total_n = sum(i['sig_n'] for i in improvements)
        weighted_improvement = sum(i['improvement'] * i['sig_n'] / total_n for i in improvements)
        print(f"\n  WEIGHTED AVERAGE IMPROVEMENT: {weighted_improvement:+.1f}%")
    else:
        weighted_improvement = None

    # Step 6: Temporal stability (4 quarters)
    print(f"\nStep 6: Temporal Stability")

    n = len(signal_markets)
    q_size = n // 4

    for i, (name, start, end) in enumerate([
        ('Q1', 0, q_size),
        ('Q2', q_size, 2*q_size),
        ('Q3', 2*q_size, 3*q_size),
        ('Q4', 3*q_size, n)
    ]):
        q_data = signal_markets.iloc[start:end]
        if len(q_data) > 0:
            q_wr = q_data['bet_wins'].mean()
            q_be = q_data['avg_no_price'].mean() / 100.0
            q_edge = (q_wr - q_be) * 100
            status = "+" if q_edge > 0 else "-"
            print(f"  {name}: N={len(q_data)}, WR={q_wr:.1%}, Edge={q_edge:+.1f}% [{status}]")

    # Step 7: Concentration
    print(f"\nStep 7: Concentration Check")

    # Simple concentration: max single market / total wins
    market_contrib = signal_markets.groupby('market_ticker')['bet_wins'].sum()
    if market_contrib.sum() > 0:
        max_contrib = market_contrib.max() / market_contrib.sum()
        print(f"  Max single market contribution: {max_contrib:.1%}")
    else:
        max_contrib = 0
        print("  No wins to analyze concentration")

    # Final verdict
    print(f"\n" + "="*70)
    print(f"S010 FINAL VERDICT")
    print("="*70)

    passes_markets = n_total >= 50
    passes_edge = edge > 0
    passes_pvalue = p_value < 0.01
    passes_proxy = weighted_improvement is not None and weighted_improvement > 5  # Need meaningful improvement
    passes_conc = max_contrib < 0.30

    print(f"  Markets >= 50: {'PASS' if passes_markets else 'FAIL'} ({n_total})")
    print(f"  Edge > 0: {'PASS' if passes_edge else 'FAIL'} ({edge:+.1f}%)")
    print(f"  P-value < 0.01: {'PASS' if passes_pvalue else 'FAIL'} ({p_value:.2e})")
    print(f"  Improvement > 5%: {'PASS' if passes_proxy else 'FAIL'} ({weighted_improvement:+.1f}% if weighted_improvement else 'N/A')")
    print(f"  Concentration < 30%: {'PASS' if passes_conc else 'FAIL'} ({max_contrib:.1%})")

    all_pass = passes_markets and passes_edge and passes_pvalue and passes_proxy and passes_conc

    status = "VALIDATED" if all_pass else "REJECTED"
    print(f"\n  STATUS: {status}")

    return {
        'strategy': 'S010',
        'status': status,
        'markets': n_total,
        'win_rate': float(win_rate),
        'breakeven': float(breakeven),
        'edge': float(edge),
        'p_value': float(p_value),
        'improvement': float(weighted_improvement) if weighted_improvement else None,
        'concentration': float(max_contrib),
        'criteria': {
            'passes_markets': passes_markets,
            'passes_edge': passes_edge,
            'passes_pvalue': passes_pvalue,
            'passes_proxy': passes_proxy,
            'passes_conc': passes_conc
        }
    }


def validate_s011_detailed(df):
    """
    Validate S011: Stable Leverage Bot NO Consensus

    Signal: Markets where leverage std < 0.5 AND >60% NO consensus AND >=3 trades
    Action: Bet NO
    Win: market_result == 'no'
    """
    print("\n" + "="*70)
    print("S011 DETAILED VALIDATION")
    print("="*70)

    df = df.copy()

    # Step 1: Aggregate by market
    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['mean', 'std'],
        'taker_side': lambda x: (x == 'no').mean(),  # NO ratio
        'no_price': 'mean',
        'count': 'count',  # Number of trades
        'market_result': 'first',
    }).reset_index()
    market_stats.columns = ['market_ticker', 'lev_mean', 'lev_std', 'no_ratio', 'avg_no_price', 'n_trades', 'market_result']

    # Fill NaN std with 0 (single-trade markets)
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)

    print(f"\nStep 1: Market aggregation")
    print(f"  Total markets: {len(market_stats):,}")

    # Step 2: Apply signal filters
    signal_markets = market_stats[
        (market_stats['lev_std'] < LEVERAGE_STD_THRESHOLD) &
        (market_stats['no_ratio'] > NO_CONSENSUS_THRESHOLD) &
        (market_stats['n_trades'] >= MIN_TRADES)
    ].copy()

    print(f"\nStep 2: Signal detection")
    print(f"  Markets with lev_std < 0.5: {len(market_stats[market_stats['lev_std'] < LEVERAGE_STD_THRESHOLD]):,}")
    print(f"  Markets with >60% NO: {len(market_stats[market_stats['no_ratio'] > NO_CONSENSUS_THRESHOLD]):,}")
    print(f"  Markets with >= 3 trades: {len(market_stats[market_stats['n_trades'] >= MIN_TRADES]):,}")
    print(f"  Signal markets (ALL conditions): {len(signal_markets):,}")

    if len(signal_markets) == 0:
        return {'status': 'NO_SIGNAL', 'markets': 0}

    # Step 3: Calculate edge
    signal_markets['bet_wins'] = (signal_markets['market_result'] == 'no').astype(int)

    win_rate = signal_markets['bet_wins'].mean()
    avg_no_price = signal_markets['avg_no_price'].mean()
    breakeven = avg_no_price / 100.0
    edge = (win_rate - breakeven) * 100

    n_wins = int(signal_markets['bet_wins'].sum())
    n_total = len(signal_markets)

    binom_result = stats.binomtest(n_wins, n_total, breakeven, alternative='greater')
    p_value = binom_result.pvalue

    print(f"\nStep 3: Edge calculation")
    print(f"  Win Rate: {win_rate:.2%}")
    print(f"  Avg NO Price: {avg_no_price:.2f}c")
    print(f"  Breakeven: {breakeven:.2%}")
    print(f"  Edge: {edge:+.2f}%")
    print(f"  P-value: {p_value:.2e}")

    # Step 4: Price proxy check
    print(f"\nStep 4: Price Proxy Check")

    all_market_agg = df.groupby('market_ticker').agg({
        'no_price': 'mean',
        'market_result': 'first'
    }).reset_index()
    all_market_agg['bet_wins'] = (all_market_agg['market_result'] == 'no').astype(int)

    signal_markets['price_bucket'] = (signal_markets['avg_no_price'] // 5) * 5
    all_market_agg['price_bucket'] = (all_market_agg['no_price'] // 5) * 5

    print(f"\n  {'Bucket':<10} {'Signal':<20} {'ALL Markets':<20} {'Improvement':<12}")
    print("  " + "-"*65)

    improvements = []
    for bucket in sorted(signal_markets['price_bucket'].unique()):
        sig_bucket = signal_markets[signal_markets['price_bucket'] == bucket]
        all_bucket = all_market_agg[all_market_agg['price_bucket'] == bucket]

        if len(sig_bucket) >= 5 and len(all_bucket) >= 10:
            sig_wr = sig_bucket['bet_wins'].mean()
            all_wr = all_bucket['bet_wins'].mean()
            improvement = (sig_wr - all_wr) * 100

            print(f"  {int(bucket):2d}-{int(bucket)+5}c     {sig_wr:.1%} ({len(sig_bucket)})    {all_wr:.1%} ({len(all_bucket)})    {improvement:+.1f}%")

            improvements.append({'bucket': bucket, 'sig_n': len(sig_bucket), 'improvement': improvement})

    if improvements:
        total_n = sum(i['sig_n'] for i in improvements)
        weighted_improvement = sum(i['improvement'] * i['sig_n'] / total_n for i in improvements)
        print(f"\n  WEIGHTED AVERAGE IMPROVEMENT: {weighted_improvement:+.1f}%")
    else:
        weighted_improvement = None

    # Step 5: Temporal stability
    print(f"\nStep 5: Temporal Stability")

    n = len(signal_markets)
    q_size = n // 4

    for i, (name, start, end) in enumerate([
        ('Q1', 0, q_size),
        ('Q2', q_size, 2*q_size),
        ('Q3', 2*q_size, 3*q_size),
        ('Q4', 3*q_size, n)
    ]):
        q_data = signal_markets.iloc[start:end]
        if len(q_data) > 0:
            q_wr = q_data['bet_wins'].mean()
            q_be = q_data['avg_no_price'].mean() / 100.0
            q_edge = (q_wr - q_be) * 100
            status = "+" if q_edge > 0 else "-"
            print(f"  {name}: N={len(q_data)}, WR={q_wr:.1%}, Edge={q_edge:+.1f}% [{status}]")

    # Step 6: Concentration
    print(f"\nStep 6: Concentration Check")

    market_contrib = signal_markets.groupby('market_ticker')['bet_wins'].sum()
    if market_contrib.sum() > 0:
        max_contrib = market_contrib.max() / market_contrib.sum()
        print(f"  Max single market contribution: {max_contrib:.1%}")
    else:
        max_contrib = 0

    # Final verdict
    print(f"\n" + "="*70)
    print(f"S011 FINAL VERDICT")
    print("="*70)

    passes_markets = n_total >= 50
    passes_edge = edge > 0
    passes_pvalue = p_value < 0.01
    passes_proxy = weighted_improvement is not None and weighted_improvement > 5
    passes_conc = max_contrib < 0.30

    print(f"  Markets >= 50: {'PASS' if passes_markets else 'FAIL'} ({n_total})")
    print(f"  Edge > 0: {'PASS' if passes_edge else 'FAIL'} ({edge:+.1f}%)")
    print(f"  P-value < 0.01: {'PASS' if passes_pvalue else 'FAIL'} ({p_value:.2e})")
    print(f"  Improvement > 5%: {'PASS' if passes_proxy else 'FAIL'} ({weighted_improvement:+.1f}%)" if weighted_improvement else f"  Improvement > 5%: FAIL (N/A)")
    print(f"  Concentration < 30%: {'PASS' if passes_conc else 'FAIL'} ({max_contrib:.1%})")

    all_pass = passes_markets and passes_edge and passes_pvalue and passes_proxy and passes_conc

    status = "VALIDATED" if all_pass else "REJECTED"
    print(f"\n  STATUS: {status}")

    return {
        'strategy': 'S011',
        'status': status,
        'markets': n_total,
        'win_rate': float(win_rate),
        'breakeven': float(breakeven),
        'edge': float(edge),
        'p_value': float(p_value),
        'improvement': float(weighted_improvement) if weighted_improvement else None,
        'concentration': float(max_contrib),
        'criteria': {
            'passes_markets': passes_markets,
            'passes_edge': passes_edge,
            'passes_pvalue': passes_pvalue,
            'passes_proxy': passes_proxy,
            'passes_conc': passes_conc
        }
    }


def main():
    print("="*70)
    print("SESSION 011 - DEEP VALIDATION V2")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")

    df = load_data()

    # Validate both strategies
    s010_results = validate_s010_detailed(df)
    s011_results = validate_s011_detailed(df)

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for result in [s010_results, s011_results]:
        print(f"\n{result['strategy']}:")
        print(f"  Status: {result['status']}")
        print(f"  Markets: {result.get('markets', 0)}")
        print(f"  Edge: {result.get('edge', 0):+.1f}%")
        print(f"  Improvement: {result.get('improvement', 'N/A')}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'S010': s010_results,
        'S011': s011_results
    }

    output_path = Path(__file__).parent.parent / "reports" / "session011_deep_validation_final_v2.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    return output


if __name__ == "__main__":
    main()
