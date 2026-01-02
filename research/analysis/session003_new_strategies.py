#!/usr/bin/env python3
"""
Session 003: New Strategy Investigation

Looking for strategies that are NOT just variations of the existing price-based ones.

KNOWN VALIDATED:
- YES at 80-90c: +5.1% edge (already in VALIDATED_STRATEGIES.md)
- NO at 80-90c: +3.3% edge (already in VALIDATED_STRATEGIES.md)
- NO at 90-100c: +1.2% edge (already in VALIDATED_STRATEGIES.md)

NEW CANDIDATES FROM DEEP DIVE:
1. NO at 60-70c: +30.5% edge - is this distinct from the above?
2. NO at 70-80c: +58.1% edge - different from 80-90c?
3. Whale NO trades: higher edge than retail?
4. First Trade NO: early trades have higher edge?

Let's investigate these more carefully.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def extract_base_market(ticker: str) -> str:
    """Extract the base market from a ticker."""
    parts = ticker.rsplit('-', 1)
    if len(parts) == 2:
        if re.match(r'^[A-Z0-9]{1,10}$', parts[1]) and len(parts[1]) <= 10:
            return parts[0]
    return ticker


def extract_category(ticker: str) -> str:
    """Extract market category from ticker."""
    match = re.match(r'^(KX[A-Z]+)', ticker)
    return match.group(1) if match else 'UNKNOWN'


def calculate_max_market_share(profits: pd.Series) -> float:
    """Calculate max concentration."""
    positive = profits[profits > 0]
    if len(positive) == 0 or positive.sum() == 0:
        return 1.0
    return positive.max() / positive.sum()


def binomial_test(wins: int, total: int, fair_rate: float) -> float:
    """One-sided binomial test."""
    if total == 0:
        return 1.0
    result = stats.binomtest(wins, total, fair_rate, alternative='greater')
    return result.pvalue


def full_strategy_analysis(df: pd.DataFrame, description: str, min_markets: int = 50) -> Dict:
    """Complete strategy analysis with all validation checks."""
    if len(df) == 0:
        return None

    df = df.copy()
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

    # Breakeven calculation
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

    # Validation
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
    """Load data."""
    path = Path('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv')
    print(f"Loading: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} trades")

    if df['is_winner'].dtype == 'object':
        df['is_winner'] = df['is_winner'].map({'True': True, 'False': False, True: True, False: False})

    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['category'] = df['market_ticker'].apply(extract_category)

    return df


def investigate_no_60_70(df: pd.DataFrame):
    """
    Deep investigation of NO at 60-70c strategy.

    Is this truly distinct from the 80-90c strategies, or is there overlap?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: NO at 60-70c")
    print("=" * 70)

    # Get all NO trades at 60-70c
    no_60_70 = df[
        (df['trade_price'] >= 60) &
        (df['trade_price'] < 70) &
        (df['taker_side'] == 'no')
    ].copy()

    print(f"\nTotal NO trades at 60-70c: {len(no_60_70):,}")

    result = full_strategy_analysis(no_60_70, "NO at 60-70c")

    print(f"\n--- Full Analysis ---")
    print(f"Unique markets: {result['unique_markets']}")
    print(f"Win rate: {result['win_rate']:.1%}")
    print(f"Breakeven: {result['breakeven_rate']:.1%}")
    print(f"Edge: {result['edge']:+.1%}")
    print(f"Total profit: ${result['total_profit']:,.0f}")
    print(f"Max concentration: {result['max_market_share']:.1%}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"VALID: {result['is_valid']}")

    # Category breakdown
    print("\n--- Category Breakdown ---")
    cat_stats = no_60_70.groupby('category').agg({
        'id': 'count',
        'base_market': 'nunique',
        'is_winner': 'mean',
        'actual_profit_dollars': 'sum'
    }).rename(columns={'id': 'trades', 'base_market': 'markets', 'is_winner': 'win_rate'})

    cat_stats = cat_stats.sort_values('trades', ascending=False)
    print(f"\n{'Category':<30} {'Trades':>8} {'Markets':>8} {'WinRate':>8} {'Profit':>12}")
    print("-" * 70)
    for cat, row in cat_stats.head(15).iterrows():
        print(f"{cat:<30} {row['trades']:>8,} {row['markets']:>8} {row['win_rate']:>8.1%} ${row['actual_profit_dollars']:>11,.0f}")

    return result


def investigate_no_70_80(df: pd.DataFrame):
    """
    Deep investigation of NO at 70-80c strategy.

    This is between 60-70c and 80-90c.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: NO at 70-80c")
    print("=" * 70)

    no_70_80 = df[
        (df['trade_price'] >= 70) &
        (df['trade_price'] < 80) &
        (df['taker_side'] == 'no')
    ].copy()

    print(f"\nTotal NO trades at 70-80c: {len(no_70_80):,}")

    result = full_strategy_analysis(no_70_80, "NO at 70-80c")

    print(f"\n--- Full Analysis ---")
    print(f"Unique markets: {result['unique_markets']}")
    print(f"Win rate: {result['win_rate']:.1%}")
    print(f"Breakeven: {result['breakeven_rate']:.1%}")
    print(f"Edge: {result['edge']:+.1%}")
    print(f"Total profit: ${result['total_profit']:,.0f}")
    print(f"Max concentration: {result['max_market_share']:.1%}")
    print(f"P-value: {result['p_value']:.6f}")
    print(f"VALID: {result['is_valid']}")

    return result


def investigate_whale_strategies(df: pd.DataFrame):
    """
    Do whale trades (500+ contracts) have more edge than retail?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: Whale vs Retail Edge Comparison")
    print("=" * 70)

    results = []

    for price_range, label in [((60, 70), "60-70c"), ((70, 80), "70-80c"), ((80, 90), "80-90c"), ((90, 100), "90-100c")]:
        print(f"\n--- {label} ---")

        for size_label, min_size, max_size in [('RETAIL', 1, 50), ('MEDIUM', 50, 200), ('WHALE', 500, 100000)]:
            subset = df[
                (df['trade_price'] >= price_range[0]) &
                (df['trade_price'] < price_range[1]) &
                (df['taker_side'] == 'no') &
                (df['count'] >= min_size) &
                (df['count'] < max_size)
            ]

            result = full_strategy_analysis(subset, f"{size_label} NO at {label}")

            if result and result.get('unique_markets', 0) >= 30:
                results.append(result)
                status = "VALID" if result['is_valid'] else ""
                print(f"  {status:6} {size_label:8}: {result['unique_markets']} mkts, "
                      f"WR {result['win_rate']:.1%}, edge {result['edge']:+.1%}, "
                      f"${result['total_profit']:,.0f}")

    return results


def investigate_category_specific(df: pd.DataFrame):
    """
    Are there category-specific strategies that are distinct?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: Category-Specific Strategies")
    print("=" * 70)

    results = []

    # Get categories with enough data
    cat_counts = df.groupby('category')['base_market'].nunique()
    big_categories = cat_counts[cat_counts >= 100].index.tolist()

    print(f"\nCategories with 100+ unique markets: {len(big_categories)}")

    # Test NO strategies at 60-80c for each category
    for price_range, label in [((60, 70), "60-70c"), ((70, 80), "70-80c")]:
        print(f"\n--- NO at {label} by Category ---")

        for cat in sorted(big_categories):
            subset = df[
                (df['category'] == cat) &
                (df['trade_price'] >= price_range[0]) &
                (df['trade_price'] < price_range[1]) &
                (df['taker_side'] == 'no')
            ]

            result = full_strategy_analysis(subset, f"{cat}: NO at {label}")

            if result and result.get('unique_markets', 0) >= 30:
                results.append(result)
                if result['is_valid']:
                    print(f"  VALID: {cat}: {result['unique_markets']} mkts, "
                          f"edge {result['edge']:+.1%}, profit ${result['total_profit']:,.0f}")

    return results


def investigate_sports_game_outcomes(df: pd.DataFrame):
    """
    Sports game markets might have unique patterns.

    Theory: In game markets, when one team is heavily favored (YES at 70-80c),
    is there value in betting NO (the underdog)?
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: Sports Game Markets")
    print("=" * 70)

    sports_categories = ['KXNFLGAME', 'KXNBAGAME', 'KXNHLGAME', 'KXNCAAFGAME',
                         'KXNCAAMBGAME', 'KXEPLGAME', 'KXUCLGAME']

    results = []

    for cat in sports_categories:
        cat_df = df[df['category'] == cat]

        if len(cat_df) == 0:
            continue

        print(f"\n--- {cat} ---")
        print(f"Total trades: {len(cat_df):,}")
        print(f"Unique markets: {cat_df['base_market'].nunique()}")

        # Test various price ranges
        for price_range, label in [((60, 70), "60-70c"), ((70, 80), "70-80c"), ((80, 90), "80-90c")]:
            for side in ['yes', 'no']:
                subset = cat_df[
                    (cat_df['trade_price'] >= price_range[0]) &
                    (cat_df['trade_price'] < price_range[1]) &
                    (cat_df['taker_side'] == side)
                ]

                result = full_strategy_analysis(subset, f"{cat}: {side.upper()} at {label}")

                if result and result.get('unique_markets', 0) >= 30 and result.get('edge') is not None:
                    results.append(result)
                    status = "VALID" if result.get('is_valid', False) else ""
                    if result.get('is_valid', False) or abs(result.get('edge', 0)) > 0.1:
                        print(f"  {status:6} {side.upper()} at {label}: {result['unique_markets']} mkts, "
                              f"edge {result['edge']:+.1%}, ${result['total_profit']:,.0f}")

    return results


def investigate_crypto_markets(df: pd.DataFrame):
    """
    Crypto markets (BTC, ETH) might have unique patterns.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATION: Crypto Markets (BTC, ETH)")
    print("=" * 70)

    crypto_categories = ['KXBTC', 'KXBTCD', 'KXETH', 'KXETHD']

    results = []

    for cat in crypto_categories:
        cat_df = df[df['category'] == cat]

        if len(cat_df) < 100:
            continue

        print(f"\n--- {cat} ---")
        print(f"Total trades: {len(cat_df):,}")
        print(f"Unique markets: {cat_df['base_market'].nunique()}")

        for price_range, label in [((60, 70), "60-70c"), ((70, 80), "70-80c"), ((80, 90), "80-90c"), ((90, 100), "90-100c")]:
            for side in ['yes', 'no']:
                subset = cat_df[
                    (cat_df['trade_price'] >= price_range[0]) &
                    (cat_df['trade_price'] < price_range[1]) &
                    (cat_df['taker_side'] == side)
                ]

                result = full_strategy_analysis(subset, f"{cat}: {side.upper()} at {label}")

                if result and result.get('unique_markets', 0) >= 30 and result.get('edge') is not None:
                    results.append(result)
                    if result.get('is_valid', False) or (result['unique_markets'] >= 50 and abs(result.get('edge', 0)) > 0.05):
                        status = "VALID" if result.get('is_valid', False) else ""
                        print(f"  {status:6} {side.upper()} at {label}: {result['unique_markets']} mkts, "
                              f"edge {result['edge']:+.1%}, ${result['total_profit']:,.0f}")

    return results


def compare_to_base_strategies(all_results: List[Dict]):
    """
    Compare new strategies to the base validated ones.
    """
    print("\n" + "=" * 70)
    print("COMPARISON TO BASE STRATEGIES")
    print("=" * 70)

    # Known validated strategies from VALIDATED_STRATEGIES.md
    base_strategies = {
        'YES at 80-90c': {'edge': 0.051, 'markets': 2110},
        'NO at 80-90c': {'edge': 0.033, 'markets': 2808},
        'NO at 90-100c': {'edge': 0.012, 'markets': 4741}
    }

    print("\n--- Base Validated Strategies ---")
    for name, stats in base_strategies.items():
        print(f"  {name}: edge {stats['edge']:+.1%}, {stats['markets']} markets")

    # New valid strategies from our investigation
    valid_new = [r for r in all_results if r and r.get('is_valid', False)]

    # Filter out strategies that are just subsets of base strategies
    truly_new = []
    for r in valid_new:
        desc = r['description']

        # Check if it's just a size subset of base
        is_size_subset = any(x in desc for x in ['RETAIL', 'MEDIUM', 'WHALE'])

        # Check if it's at a different price range
        is_new_price = any(x in desc for x in ['60-70c', '70-80c'])

        # Check if it's category specific
        is_category_specific = ':' in desc and not desc.startswith('All')

        if is_new_price or is_category_specific:
            truly_new.append(r)

    print("\n--- NEW Strategies (different from base) ---")
    if truly_new:
        for r in sorted(truly_new, key=lambda x: x['edge'], reverse=True)[:10]:
            print(f"  {r['description']}")
            print(f"    Edge: {r['edge']:+.1%}, Markets: {r['unique_markets']}, "
                  f"Profit: ${r['total_profit']:,.0f}")
    else:
        print("  No truly new strategies found.")


def main():
    """Main investigation."""
    print("=" * 70)
    print("SESSION 003: NEW STRATEGY INVESTIGATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data()

    all_results = []

    # Run investigations
    result = investigate_no_60_70(df)
    if result:
        all_results.append(result)

    result = investigate_no_70_80(df)
    if result:
        all_results.append(result)

    all_results.extend(investigate_whale_strategies(df))
    all_results.extend(investigate_category_specific(df))
    all_results.extend(investigate_sports_game_outcomes(df))
    all_results.extend(investigate_crypto_markets(df))

    # Summary comparison
    compare_to_base_strategies(all_results)

    # Final summary of truly new valid strategies
    print("\n" + "=" * 70)
    print("FINAL: NEW VALIDATED STRATEGIES FOR IMPLEMENTATION")
    print("=" * 70)

    valid_new = [r for r in all_results if r and r.get('is_valid', False)]

    # Identify truly distinct strategies
    distinct = []
    for r in valid_new:
        desc = r['description']
        # Skip subsets by size (RETAIL, MEDIUM, WHALE variations)
        if any(x in desc for x in ['RETAIL', 'MEDIUM', 'WHALE']):
            continue
        # Skip category subsets of known strategies
        if ':' in desc:
            price_part = desc.split(':')[-1].strip()
            if '80-90c' in price_part or '90-100c' in price_part:
                continue  # These are subsets of known strategies
        distinct.append(r)

    if distinct:
        print("\n*** DISTINCT NEW STRATEGIES ***\n")
        for r in sorted(distinct, key=lambda x: x['edge'], reverse=True):
            print(f"Strategy: {r['description']}")
            print(f"  Markets: {r['unique_markets']}")
            print(f"  Win Rate: {r['win_rate']:.1%} (Breakeven: {r['breakeven_rate']:.1%})")
            print(f"  Edge: {r['edge']:+.1%}")
            print(f"  Total Profit: ${r['total_profit']:,.0f}")
            print(f"  Max Concentration: {r['max_market_share']:.1%}")
            print(f"  P-value: {r['p_value']:.6f}")
            print()
    else:
        print("\nNo distinct new strategies found beyond price-based base strategies.")
        print("The base strategies (YES 80-90c, NO 80-90c, NO 90-100c) remain the core edge.")

    return all_results


if __name__ == '__main__':
    main()
