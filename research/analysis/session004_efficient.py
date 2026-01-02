#!/usr/bin/env python3
"""
Session 004: Efficient Insider Trading Analysis

Optimized for performance - avoids expensive per-market loops.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import json
import re
import warnings
warnings.filterwarnings('ignore')


def extract_base_market(ticker):
    parts = ticker.rsplit('-', 1)
    if len(parts) == 2:
        if re.match(r'^[A-Z0-9]{1,10}$', parts[1]) and len(parts[1]) <= 10:
            return parts[0]
    return ticker


def extract_category(ticker):
    match = re.match(r'^(KX[A-Z]+)', ticker)
    return match.group(1) if match else 'UNKNOWN'


def calculate_max_market_share(profits):
    positive = profits[profits > 0]
    if len(positive) == 0 or positive.sum() == 0:
        return 1.0
    return float(positive.max() / positive.sum())


def binomial_test(wins, total, fair_rate):
    if total == 0:
        return 1.0
    result = stats.binomtest(wins, total, fair_rate, alternative='greater')
    return result.pvalue


def full_strategy_analysis(df, description, min_markets=50):
    """Complete strategy analysis with all validation checks."""
    if len(df) == 0:
        return None

    df = df.copy()
    if 'base_market' not in df.columns:
        df['base_market'] = df['market_ticker'].apply(extract_base_market)

    # Market-level aggregation
    market_stats = df.groupby('base_market').agg({
        'is_winner': 'first',
        'actual_profit_dollars': 'sum',
        'cost_dollars': 'sum',
        'trade_price': 'mean',
        'taker_side': 'first',
        'count': 'sum',
        'market_result': 'first'
    }).reset_index()

    unique_markets = len(market_stats)
    if unique_markets < min_markets:
        return {
            'description': description,
            'is_valid': False,
            'unique_markets': unique_markets,
            'reason': f'Not enough markets ({unique_markets} < {min_markets})'
        }

    wins = int(market_stats['is_winner'].sum())
    total = len(market_stats)
    win_rate = wins / total

    avg_price = market_stats['trade_price'].mean()
    side = market_stats['taker_side'].iloc[0]

    if side == 'yes':
        breakeven_rate = avg_price / 100.0
    else:
        breakeven_rate = (100 - avg_price) / 100.0

    edge = win_rate - breakeven_rate

    total_profit = float(market_stats['actual_profit_dollars'].sum())
    total_cost = float(market_stats['cost_dollars'].sum())
    roi = total_profit / total_cost if total_cost > 0 else 0

    max_share = calculate_max_market_share(market_stats['actual_profit_dollars'])
    p_value = binomial_test(wins, total, breakeven_rate)

    passes_markets = unique_markets >= min_markets
    passes_concentration = max_share < 0.3
    passes_significance = p_value < 0.05
    has_positive_edge = edge > 0

    is_valid = passes_markets and passes_concentration and passes_significance and has_positive_edge

    return {
        'description': description,
        'unique_markets': int(unique_markets),
        'total_trades': int(len(df)),
        'wins': wins,
        'win_rate': round(float(win_rate), 4),
        'breakeven_rate': round(float(breakeven_rate), 4),
        'edge': round(float(edge), 4),
        'total_profit': round(float(total_profit), 2),
        'total_cost': round(float(total_cost), 2),
        'roi': round(float(roi), 4),
        'max_market_share': round(float(max_share), 4),
        'p_value': round(float(p_value), 6),
        'avg_price': round(float(avg_price), 2),
        'side': side,
        'is_valid': is_valid,
        'passes_markets': passes_markets,
        'passes_concentration': passes_concentration,
        'passes_significance': passes_significance,
        'has_positive_edge': has_positive_edge
    }


def load_data():
    """Load all data."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    path = Path('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv')
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} resolved trades")

    if df['is_winner'].dtype == 'object':
        df['is_winner'] = df['is_winner'].map({'True': True, 'False': False, True: True, False: False})

    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['category'] = df['market_ticker'].apply(extract_category)

    print(f"Unique markets: {df['base_market'].nunique():,}")
    print(f"Categories: {df['category'].nunique()}")

    return df


def analyze_whale_vs_retail(df):
    """Compare whale vs retail performance at various price points."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Whale vs Retail Performance")
    print("=" * 70)

    WHALE_SIZE = 500

    results = []

    # Test various price ranges
    for side in ['yes', 'no']:
        for price_low, price_high in [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]:
            # Whale trades
            whale_subset = df[
                (df['count'] >= WHALE_SIZE) &
                (df['trade_price'] >= price_low) &
                (df['trade_price'] < price_high) &
                (df['taker_side'] == side)
            ]

            # Retail trades
            retail_subset = df[
                (df['count'] < WHALE_SIZE) &
                (df['trade_price'] >= price_low) &
                (df['trade_price'] < price_high) &
                (df['taker_side'] == side)
            ]

            whale_result = full_strategy_analysis(whale_subset, f"WHALE {side.upper()} at {price_low}-{price_high}c", min_markets=30)
            retail_result = full_strategy_analysis(retail_subset, f"RETAIL {side.upper()} at {price_low}-{price_high}c", min_markets=30)

            if whale_result and whale_result.get('unique_markets', 0) >= 30:
                results.append(whale_result)

            if retail_result and retail_result.get('unique_markets', 0) >= 30:
                results.append(retail_result)

            # Print comparison
            if whale_result and whale_result.get('unique_markets', 0) >= 30:
                w_edge = whale_result['edge']
                r_edge = retail_result['edge'] if retail_result and retail_result.get('unique_markets', 0) >= 30 else None

                if r_edge is not None:
                    diff = w_edge - r_edge
                    status = "***" if abs(diff) > 0.05 else ""
                    if abs(w_edge) > 0.03 or abs(diff) > 0.03:
                        print(f"{status}{side.upper()} at {price_low}-{price_high}c: "
                              f"WHALE edge={w_edge:+.1%}, RETAIL edge={r_edge:+.1%}, diff={diff:+.1%}")

    return results


def analyze_trade_sequence(df):
    """Analyze if being first vs last whale matters."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Trade Sequence (First vs Last Whale)")
    print("=" * 70)

    WHALE_SIZE = 500

    # Get first and last whale per market
    whale_df = df[df['count'] >= WHALE_SIZE].copy()

    # First whale per market
    first_whales = whale_df.loc[whale_df.groupby('base_market')['timestamp'].idxmin()]
    # Last whale per market
    last_whales = whale_df.loc[whale_df.groupby('base_market')['timestamp'].idxmax()]

    print(f"\nMarkets with first whale: {len(first_whales):,}")
    print(f"Markets with last whale: {len(last_whales):,}")

    # Analyze first whales by price
    print("\n--- First Whale by Price Range ---")
    for side in ['yes', 'no']:
        for price_low, price_high in [(60, 70), (70, 80), (80, 90)]:
            subset = first_whales[
                (first_whales['taker_side'] == side) &
                (first_whales['trade_price'] >= price_low) &
                (first_whales['trade_price'] < price_high)
            ]

            if len(subset) >= 50:
                wr = subset['is_winner'].mean()
                avg_price = subset['trade_price'].mean()

                if side == 'yes':
                    be = avg_price / 100
                else:
                    be = (100 - avg_price) / 100

                edge = wr - be
                profit = subset['actual_profit_dollars'].sum()

                if abs(edge) > 0.03:
                    status = "***" if edge > 0.1 else ""
                    print(f"  {status}First {side.upper()} at {price_low}-{price_high}c: "
                          f"{len(subset)} mkts, WR={wr:.1%}, edge={edge:+.1%}, ${profit:,.0f}")

    # Analyze last whales
    print("\n--- Last Whale by Price Range ---")
    for side in ['yes', 'no']:
        for price_low, price_high in [(60, 70), (70, 80), (80, 90)]:
            subset = last_whales[
                (last_whales['taker_side'] == side) &
                (last_whales['trade_price'] >= price_low) &
                (last_whales['trade_price'] < price_high)
            ]

            if len(subset) >= 50:
                wr = subset['is_winner'].mean()
                avg_price = subset['trade_price'].mean()

                if side == 'yes':
                    be = avg_price / 100
                else:
                    be = (100 - avg_price) / 100

                edge = wr - be
                profit = subset['actual_profit_dollars'].sum()

                if abs(edge) > 0.03:
                    status = "***" if edge > 0.1 else ""
                    print(f"  {status}Last {side.upper()} at {price_low}-{price_high}c: "
                          f"{len(subset)} mkts, WR={wr:.1%}, edge={edge:+.1%}, ${profit:,.0f}")


def analyze_conviction_matrix(df):
    """Analyze size x price conviction matrix."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Conviction Matrix (Size x Price)")
    print("=" * 70)

    print("\nMarket-level win rates by SIZE and PRICE (NO trades only):")
    print(f"{'Size':<15} {'50-60c':<12} {'60-70c':<12} {'70-80c':<12} {'80-90c':<12} {'90-100c':<12}")
    print("-" * 75)

    for size_label, min_size, max_size in [
        ('Retail (<50)', 1, 50),
        ('Small (50-200)', 50, 200),
        ('Medium (200-500)', 200, 500),
        ('Large (500-1k)', 500, 1000),
        ('Mega (1k+)', 1000, 100000)
    ]:
        row = f"{size_label:<15}"
        for price_low, price_high in [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]:
            subset = df[
                (df['count'] >= min_size) & (df['count'] < max_size) &
                (df['trade_price'] >= price_low) & (df['trade_price'] < price_high) &
                (df['taker_side'] == 'no')
            ]

            if len(subset) >= 100:
                # Aggregate by market
                market_stats = subset.groupby('base_market').agg({
                    'is_winner': 'first'
                })
                market_wr = market_stats['is_winner'].mean()
                row += f" {market_wr:.0%}".ljust(12)
            else:
                row += " -".ljust(12)

        print(row)


def explore_new_strategies(df):
    """Systematically explore new strategy ideas."""
    print("\n" + "=" * 70)
    print("ANALYSIS 4: New Strategy Exploration")
    print("=" * 70)

    all_results = []

    # YES strategies at various prices
    print("\n--- YES Strategies by Price ---")
    for price_low, price_high in [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]:
        subset = df[
            (df['trade_price'] >= price_low) &
            (df['trade_price'] < price_high) &
            (df['taker_side'] == 'yes')
        ]
        result = full_strategy_analysis(subset, f"YES at {price_low}-{price_high}c")

        if result and result.get('unique_markets', 0) >= 50:
            all_results.append(result)
            status = "VALID" if result.get('is_valid', False) else ""
            print(f"  {status} YES at {price_low}-{price_high}c: {result['unique_markets']} mkts, "
                  f"edge {result['edge']:+.1%}, ${result['total_profit']:,.0f}")

    # NO strategies at various prices
    print("\n--- NO Strategies by Price ---")
    for price_low, price_high in [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]:
        subset = df[
            (df['trade_price'] >= price_low) &
            (df['trade_price'] < price_high) &
            (df['taker_side'] == 'no')
        ]
        result = full_strategy_analysis(subset, f"NO at {price_low}-{price_high}c")

        if result and result.get('unique_markets', 0) >= 50:
            all_results.append(result)
            status = "VALID" if result.get('is_valid', False) else ""
            print(f"  {status} NO at {price_low}-{price_high}c: {result['unique_markets']} mkts, "
                  f"edge {result['edge']:+.1%}, ${result['total_profit']:,.0f}")

    # Size-filtered strategies
    print("\n--- Size-Filtered Strategies (NO at 70-80c) ---")
    for size_label, min_size, max_size in [
        ('Tiny (1-10)', 1, 10),
        ('Small (10-50)', 10, 50),
        ('Medium (50-200)', 50, 200),
        ('Large (200-500)', 200, 500),
        ('Whale (500+)', 500, 100000)
    ]:
        subset = df[
            (df['count'] >= min_size) & (df['count'] < max_size) &
            (df['trade_price'] >= 70) & (df['trade_price'] < 80) &
            (df['taker_side'] == 'no')
        ]
        result = full_strategy_analysis(subset, f"{size_label} NO at 70-80c", min_markets=30)

        if result and result.get('unique_markets', 0) >= 30:
            all_results.append(result)
            status = "VALID" if result.get('is_valid', False) else ""
            if result['edge'] > 0.2 or result.get('is_valid', False):
                print(f"  {status} {size_label} NO at 70-80c: {result['unique_markets']} mkts, "
                      f"edge {result['edge']:+.1%}")

    # Category-specific strategies
    print("\n--- Category-Specific (NO at 60-70c) ---")
    top_categories = df.groupby('category')['base_market'].nunique().nlargest(15).index.tolist()

    for cat in top_categories:
        subset = df[
            (df['category'] == cat) &
            (df['trade_price'] >= 60) & (df['trade_price'] < 70) &
            (df['taker_side'] == 'no')
        ]
        result = full_strategy_analysis(subset, f"{cat}: NO at 60-70c", min_markets=30)

        if result and result.get('is_valid', False):
            all_results.append(result)
            print(f"  VALID {cat}: NO at 60-70c: {result['unique_markets']} mkts, "
                  f"edge {result['edge']:+.1%}")

    return all_results


def validate_best_candidates(df):
    """Rigorously validate the best strategy candidates."""
    print("\n" + "=" * 70)
    print("FINAL VALIDATION: Best Strategy Candidates")
    print("=" * 70)

    validated = []

    # Test NO at 50-60c (new territory)
    candidates = [
        ('NO at 50-60c', lambda d: d[(d['trade_price'] >= 50) & (d['trade_price'] < 60) & (d['taker_side'] == 'no')]),
        ('NO at 55-65c', lambda d: d[(d['trade_price'] >= 55) & (d['trade_price'] < 65) & (d['taker_side'] == 'no')]),
        ('NO at 65-75c', lambda d: d[(d['trade_price'] >= 65) & (d['trade_price'] < 75) & (d['taker_side'] == 'no')]),
        ('YES at 10-20c', lambda d: d[(d['trade_price'] >= 10) & (d['trade_price'] < 20) & (d['taker_side'] == 'yes')]),
        ('YES at 85-95c', lambda d: d[(d['trade_price'] >= 85) & (d['trade_price'] < 95) & (d['taker_side'] == 'yes')]),
        ('Large NO at 70-80c (200+)', lambda d: d[(d['count'] >= 200) & (d['trade_price'] >= 70) & (d['trade_price'] < 80) & (d['taker_side'] == 'no')]),
    ]

    for name, filter_func in candidates:
        subset = filter_func(df)
        result = full_strategy_analysis(subset, name)

        if result and result.get('unique_markets', 0) >= 50:
            print(f"\n{name}:")
            print(f"  Markets: {result['unique_markets']}")
            print(f"  Win Rate: {result['win_rate']:.1%}")
            print(f"  Breakeven: {result['breakeven_rate']:.1%}")
            print(f"  Edge: {result['edge']:+.1%}")
            print(f"  Profit: ${result['total_profit']:,.0f}")
            print(f"  Max Concentration: {result['max_market_share']:.1%}")
            print(f"  P-value: {result['p_value']:.6f}")
            print(f"  VALID: {result['is_valid']}")

            if result['is_valid']:
                validated.append(result)

    return validated


def main():
    """Main analysis."""
    print("=" * 70)
    print("SESSION 004: INSIDER TRADING & NEW STRATEGY SEARCH")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data()

    # Run analyses
    whale_results = analyze_whale_vs_retail(df)
    analyze_trade_sequence(df)
    analyze_conviction_matrix(df)
    new_strategies = explore_new_strategies(df)
    validated = validate_best_candidates(df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if validated:
        print(f"\nNEW VALIDATED STRATEGIES: {len(validated)}")
        for v in sorted(validated, key=lambda x: x['edge'], reverse=True):
            print(f"  - {v['description']}: edge {v['edge']:+.1%}, {v['unique_markets']} mkts, ${v['total_profit']:,.0f}")
    else:
        print("\nNo new validated strategies found beyond existing ones.")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'whale_analysis': whale_results,
        'new_strategies': new_strategies,
        'validated': validated
    }

    output_path = Path('/Users/samuelclark/Desktop/kalshiflow/research/reports/session004_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
