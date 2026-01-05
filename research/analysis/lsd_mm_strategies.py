#!/usr/bin/env python3
"""
LSD MODE: Market Maker Strategy Exploration
===========================================
Rapid-fire testing of unconventional MM strategies.
Speed over rigor. Flag anything with raw edge >5%.

Tests 14 unconventional hypotheses in ~10 minutes.
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
REPORT_FILE = BASE_DIR / "reports/lsd_mm_strategies.json"

def load_data():
    """Load trade data."""
    print("Loading trade data...")
    df = pd.read_csv(TRADES_FILE, parse_dates=['datetime'])
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

    # Filter resolved only
    markets = markets[markets['result'].isin(['yes', 'no'])].copy()

    print(f"Aggregated to {len(markets):,} resolved markets")
    return markets

def calc_edge(markets, signal_mask, bet_no=True):
    """Calculate edge for a signal."""
    subset = markets[signal_mask]
    if len(subset) < 30:
        return None

    if bet_no:
        wins = subset['no_won'].sum()
        avg_no_price = 100 - subset['avg_price'].mean()
        breakeven = avg_no_price / 100
    else:
        wins = (~subset['no_won']).sum()
        breakeven = subset['avg_price'].mean() / 100

    win_rate = wins / len(subset)
    edge = (win_rate - breakeven) * 100

    # Quick significance
    z = (win_rate - breakeven) / np.sqrt(breakeven * (1-breakeven) / len(subset))
    p_value = 1 - stats.norm.cdf(abs(z))

    return {
        'markets': len(subset),
        'wins': int(wins),
        'win_rate': round(win_rate * 100, 1),
        'breakeven': round(breakeven * 100, 1),
        'edge': round(edge, 1),
        'p_value': round(p_value, 4),
        'pass': edge > 5.0
    }

def test_spread_compression(df, markets):
    """H-MM001: Spread compression - When spread narrows suddenly."""
    print("\n[H-MM001] Testing spread compression...")

    # Calculate per-trade spread proxy (price volatility in short window)
    # Use price std as proxy for spread behavior
    signal = markets['price_std'] < 3  # Very tight price range
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Tight spread (std<3): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_spread_oscillation(df, markets):
    """H-MM002: Markets that ping-pong between tight/wide spreads."""
    print("\n[H-MM002] Testing spread oscillation...")

    # High price range but low overall std = oscillating
    signal = (markets['price_range'] > 20) & (markets['price_std'] < 10)
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  High range + low std: {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_spread_asymmetry(df, markets):
    """H-MM003: Bid-ask imbalance as directional signal (proxy: yes/no ratio)."""
    print("\n[H-MM003] Testing spread asymmetry...")

    # Strong NO flow = potential edge for NO
    signal = markets['no_ratio'] > 0.7
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  High NO ratio (>70%): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_quote_stuffing(df, markets):
    """H-MM004: Rapid price changes without volume = manipulation?"""
    print("\n[H-MM004] Testing quote stuffing proxy...")

    # High price volatility relative to trade count
    markets['price_vol_per_trade'] = markets['price_std'] / np.maximum(markets['n_trades'], 1)
    signal = markets['price_vol_per_trade'] > markets['price_vol_per_trade'].quantile(0.9)
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  High price vol per trade (top 10%): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_time_weighted_imbalance(df, markets):
    """H-MM005: Imbalance that persists across multiple time windows."""
    print("\n[H-MM005] Testing persistent imbalance...")

    # Long duration markets with consistent NO flow
    signal = (markets['duration_hours'] > 24) & (markets['no_ratio'] > 0.6)
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Long market + NO flow: {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_dead_zone_trading(df, markets):
    """H-MM006: Edge in low-activity periods."""
    print("\n[H-MM006] Testing dead zone trading...")

    # Very few trades = potential mispricing
    signal = (markets['n_trades'] >= 5) & (markets['n_trades'] <= 10)
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Low activity (5-10 trades): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_informed_flow_clustering(df, markets):
    """H-MM007: Do informed trades cluster in time?"""
    print("\n[H-MM007] Testing informed flow clustering...")

    # Short duration markets with high trade count = burst activity
    short_duration = markets['duration_hours'] < 1
    high_trades = markets['n_trades'] >= 20
    signal = short_duration & high_trades
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Clustered trading (<1hr, 20+ trades): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_toxic_flow_reversal(df, markets):
    """H-MM008: Fade the flow after toxic bursts."""
    print("\n[H-MM008] Testing toxic flow reversal...")

    # Strong YES flow (retail) but price moved toward NO = fade YES
    signal = (markets['yes_ratio'] > 0.7) & (markets['price_move'] < -5)
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  YES flow + price drop: {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_size_price_divergence(df, markets):
    """H-MM009: Big trades at bad prices = uninformed?"""
    print("\n[H-MM009] Testing size-price divergence...")

    # Compute this from raw trades
    print("  Computing from raw trades...")

    # Get largest trades per market
    large_trades = df[df['count'] >= 50].copy()

    if len(large_trades) == 0:
        return None

    # Check if large trades happen at extreme prices
    large_trades['is_extreme'] = (large_trades['yes_price'] < 20) | (large_trades['yes_price'] > 80)

    # Markets with large trades at extreme prices
    extreme_large = large_trades.groupby('ticker')['is_extreme'].any()
    signal = markets.index.isin(extreme_large[extreme_large].index)
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Large trades at extreme prices: {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_fibonacci_trade_counts(df, markets):
    """H-MM012: Does the Nth fibonacci trade have edge?"""
    print("\n[H-MM012] Testing fibonacci trade counts...")

    fibs = [8, 13, 21, 34, 55, 89]
    signal = markets['n_trades'].isin(fibs)
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Fibonacci trade counts: {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_round_number_magnetism(df, markets):
    """H-MM013: Do prices gravitate to 25c, 50c, 75c?"""
    print("\n[H-MM013] Testing round number magnetism...")

    # Markets that close near round numbers
    markets['near_round'] = markets['last_price'].apply(
        lambda x: min(abs(x - 25), abs(x - 50), abs(x - 75)) < 3
    )
    signal = markets['near_round']
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Near round numbers: {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_contrarian_whale(df, markets):
    """H-MM014: Fade whale trades instead of follow."""
    print("\n[H-MM014] Testing contrarian whale...")

    # Get whale trades (>$100)
    whale_trades = df[df['count'] >= 100].copy()

    if len(whale_trades) == 0:
        return None

    # Markets where whales mostly bet YES
    whale_yes_ratio = whale_trades.groupby('ticker')['is_yes'].mean()
    whale_yes_markets = whale_yes_ratio[whale_yes_ratio > 0.7].index

    # Fade them - bet NO
    signal = markets.index.isin(whale_yes_markets)
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Fade whale YES: {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_price_reversal_magnitude(df, markets):
    """H-MM010: Large price moves that reverse."""
    print("\n[H-MM010] Testing price reversal magnitude...")

    # Big price drop = potential reversal opportunity
    signal = markets['price_move'] < -10
    result = calc_edge(markets, signal, bet_no=True)  # Bet NO on dropping prices

    if result:
        print(f"  Big price drop (>10c): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_price_momentum(df, markets):
    """H-MM011: Momentum following."""
    print("\n[H-MM011] Testing price momentum...")

    # Price moved toward NO = momentum says bet NO
    signal = markets['price_move'] < 0
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Price momentum (toward NO): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_rlm_combined(df, markets):
    """RLM: Retail Losing Money - combine YES ratio + price movement."""
    print("\n[RLM] Testing Retail Losing Money signal...")

    # YES ratio > 65% AND price moved toward NO AND enough trades
    signal = (
        (markets['yes_ratio'] > 0.65) &
        (markets['price_move'] < 0) &
        (markets['n_trades'] >= 15)
    )
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  RLM (YES>65% + price down + 15+ trades): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_extreme_yes_ratio(df, markets):
    """Test extreme YES ratio (>80%)."""
    print("\n[H-MM015] Testing extreme YES ratio...")

    signal = markets['yes_ratio'] > 0.80
    result = calc_edge(markets, signal, bet_no=True)

    if result:
        print(f"  Extreme YES ratio (>80%): {result['edge']:+.1f}% edge, {result['markets']} markets")
    return result

def test_price_bucket_filter(df, markets):
    """Test different price buckets."""
    print("\n[H-MM016] Testing price bucket filter...")

    results = {}
    for low, high in [(30, 50), (50, 70), (70, 90)]:
        bucket_name = f"NO {low}-{high}c"
        signal = (markets['avg_price'] >= low) & (markets['avg_price'] < high)
        result = calc_edge(markets, signal, bet_no=True)
        if result:
            results[bucket_name] = result
            print(f"  {bucket_name}: {result['edge']:+.1f}% edge, {result['markets']} markets")

    return results

def main():
    """Run all MM strategy tests."""
    print("=" * 60)
    print("LSD MODE: Market Maker Strategy Exploration")
    print("=" * 60)

    # Load data
    df = load_data()
    markets = get_market_summary(df)

    results = {}

    # Run all tests
    results['H-MM001'] = {'name': 'Spread Compression', 'result': test_spread_compression(df, markets)}
    results['H-MM002'] = {'name': 'Spread Oscillation', 'result': test_spread_oscillation(df, markets)}
    results['H-MM003'] = {'name': 'Spread Asymmetry', 'result': test_spread_asymmetry(df, markets)}
    results['H-MM004'] = {'name': 'Quote Stuffing', 'result': test_quote_stuffing(df, markets)}
    results['H-MM005'] = {'name': 'Time-Weighted Imbalance', 'result': test_time_weighted_imbalance(df, markets)}
    results['H-MM006'] = {'name': 'Dead Zone Trading', 'result': test_dead_zone_trading(df, markets)}
    results['H-MM007'] = {'name': 'Informed Flow Clustering', 'result': test_informed_flow_clustering(df, markets)}
    results['H-MM008'] = {'name': 'Toxic Flow Reversal', 'result': test_toxic_flow_reversal(df, markets)}
    results['H-MM009'] = {'name': 'Size-Price Divergence', 'result': test_size_price_divergence(df, markets)}
    results['H-MM010'] = {'name': 'Price Reversal Magnitude', 'result': test_price_reversal_magnitude(df, markets)}
    results['H-MM011'] = {'name': 'Price Momentum', 'result': test_price_momentum(df, markets)}
    results['H-MM012'] = {'name': 'Fibonacci Trade Counts', 'result': test_fibonacci_trade_counts(df, markets)}
    results['H-MM013'] = {'name': 'Round Number Magnetism', 'result': test_round_number_magnetism(df, markets)}
    results['H-MM014'] = {'name': 'Contrarian Whale', 'result': test_contrarian_whale(df, markets)}
    results['H-MM015'] = {'name': 'Extreme YES Ratio', 'result': test_extreme_yes_ratio(df, markets)}
    results['RLM'] = {'name': 'Retail Losing Money', 'result': test_rlm_combined(df, markets)}

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Quick Screening Results")
    print("=" * 60)

    passed = []
    failed = []

    for h_id, data in results.items():
        r = data['result']
        if r is None:
            continue

        if isinstance(r, dict) and 'edge' in r:
            if r['pass']:
                passed.append((h_id, data['name'], r))
            else:
                failed.append((h_id, data['name'], r))

    print("\n### PASSED (Edge > 5%)")
    if passed:
        for h_id, name, r in sorted(passed, key=lambda x: -x[2]['edge']):
            print(f"  {h_id}: {name}")
            print(f"      Edge: {r['edge']:+.1f}%, Markets: {r['markets']}, Win: {r['win_rate']:.1f}%, p={r['p_value']:.4f}")
    else:
        print("  None passed threshold")

    print("\n### FAILED (Edge <= 5%)")
    for h_id, name, r in sorted(failed, key=lambda x: -x[2]['edge'])[:5]:
        print(f"  {h_id}: {name} -> {r['edge']:+.1f}% edge")

    # Save results
    output = {
        'session': 'LSD-MM-001',
        'timestamp': datetime.now().isoformat(),
        'mode': 'LSD',
        'total_markets': len(markets),
        'hypotheses_tested': len([r for r in results.values() if r['result'] is not None]),
        'passed_threshold': len(passed),
        'results': results
    }

    with open(REPORT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to {REPORT_FILE}")

    return results

if __name__ == "__main__":
    main()
