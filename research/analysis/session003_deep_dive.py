#!/usr/bin/env python3
"""
Session 003 Deep Dive: Investigate interesting patterns and validate properly.

Key observations from initial run:
1. Many strategies show negative edge but positive profit - need to investigate
2. NO strategies at high prices (70-100c) consistently show positive edge
3. Weekend and price-stable patterns showed high profits

Let's focus on finding NEW strategies that aren't just the base price-based ones.
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
    """Extract the base market (game/event) from a ticker, removing team suffix."""
    parts = ticker.rsplit('-', 1)
    if len(parts) == 2:
        if re.match(r'^[A-Z0-9]{1,10}$', parts[1]) and len(parts[1]) <= 10:
            return parts[0]
    return ticker


def get_price_bucket(price: float) -> str:
    """Get the 10-cent price bucket."""
    if pd.isna(price):
        return 'unknown'
    bucket = int(price // 10) * 10
    return f"{bucket}-{bucket+10}c"


def calculate_max_market_share(profits_by_market: pd.Series) -> float:
    """Calculate the maximum share of profit from any single market."""
    if len(profits_by_market) == 0:
        return 1.0
    positive_profits = profits_by_market[profits_by_market > 0]
    if len(positive_profits) == 0 or positive_profits.sum() == 0:
        return 1.0
    return positive_profits.max() / positive_profits.sum()


def binomial_test(wins: int, total: int, fair_rate: float) -> float:
    """Calculate p-value for win rate being different from breakeven."""
    if total == 0:
        return 1.0
    result = stats.binomtest(wins, total, fair_rate, alternative='greater')
    return result.pvalue


def calculate_strategy_stats(trades_df: pd.DataFrame, description: str) -> Dict:
    """
    Calculate proper statistics for a trading strategy.

    IMPORTANT: Edge should be calculated based on actual profit potential.
    For YES trades at price P: win pays (100-P), loss costs P
    For NO trades at price P: win pays P, loss costs (100-P)

    Breakeven rate = cost / (cost + win_potential)
    Edge = actual_win_rate - breakeven_rate
    """
    if len(trades_df) == 0:
        return None

    # Group by base market
    trades_df = trades_df.copy()
    trades_df['base_market'] = trades_df['market_ticker'].apply(extract_base_market)

    # Get unique markets and their outcomes
    market_stats = trades_df.groupby('base_market').agg({
        'is_winner': 'first',  # Same for all trades in market
        'actual_profit_dollars': 'sum',
        'cost_dollars': 'sum',
        'trade_price': 'mean',
        'taker_side': 'first',
        'count': 'sum',  # Total contracts
        'market_result': 'first'
    }).reset_index()

    unique_markets = len(market_stats)

    if unique_markets < 50:
        return {
            'description': description,
            'is_valid': False,
            'unique_markets': unique_markets,
            'reason': f'Not enough markets ({unique_markets} < 50)'
        }

    # Calculate aggregate stats
    wins = int(market_stats['is_winner'].sum())
    total = len(market_stats)
    win_rate = wins / total

    # Calculate average entry price and determine side
    avg_price = market_stats['trade_price'].mean()
    side = market_stats['taker_side'].iloc[0]

    # Calculate ACTUAL breakeven based on risk/reward
    # For YES at price P: you risk P to win (100-P)
    # breakeven = P / 100
    # For NO at price P (where P is what you paid for NO):
    # you risk P to win (100-P)
    # breakeven = P / 100
    # BUT in our data, 'trade_price' is the YES price, so:
    # For YES trades: risk = trade_price, reward = 100 - trade_price
    # For NO trades: risk = 100 - trade_price, reward = trade_price

    if side == 'yes':
        # Cost is trade_price, win pays 100 - trade_price
        avg_cost = avg_price
        avg_win = 100 - avg_price
    else:
        # Cost is 100 - trade_price, win pays trade_price
        avg_cost = 100 - avg_price
        avg_win = avg_price

    # Breakeven = cost / (cost + win) = cost / 100
    breakeven_rate = avg_cost / 100.0

    # Edge = win_rate - breakeven
    edge = win_rate - breakeven_rate

    # Calculate actual P&L
    total_profit = float(market_stats['actual_profit_dollars'].sum())
    total_cost = float(market_stats['cost_dollars'].sum())
    roi = total_profit / total_cost if total_cost > 0 else 0

    # Concentration check
    max_share = calculate_max_market_share(market_stats['actual_profit_dollars'])

    # Statistical significance (one-sided: is win rate > breakeven?)
    p_value = binomial_test(wins, total, breakeven_rate)

    # Validation
    passes_market_count = unique_markets >= 50
    passes_concentration = max_share < 0.3
    passes_significance = p_value < 0.05
    has_positive_edge = edge > 0

    is_valid = passes_market_count and passes_concentration and passes_significance and has_positive_edge

    return {
        'description': description,
        'unique_markets': int(unique_markets),
        'total_trades': int(len(trades_df)),
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
        'avg_contracts': round(float(market_stats['count'].mean()), 1),
        'passes_market_count': passes_market_count,
        'passes_concentration': passes_concentration,
        'passes_significance': passes_significance,
        'has_positive_edge': has_positive_edge,
        'is_valid': is_valid
    }


def load_data():
    """Load the enriched trades data."""
    path = Path('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv')
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} trades")

    # Convert is_winner to boolean
    if df['is_winner'].dtype == 'object':
        df['is_winner'] = df['is_winner'].map({'True': True, 'False': False, True: True, False: False})

    df['base_market'] = df['market_ticker'].apply(extract_base_market)

    return df


def test_extreme_price_strategies(df: pd.DataFrame) -> List[Dict]:
    """
    Test very extreme price strategies with 5-cent granularity.

    These are the known "favorite-longshot bias" strategies.
    """
    print("\n" + "=" * 70)
    print("EXTREME PRICE STRATEGIES (5c buckets)")
    print("=" * 70)

    results = []

    # 5-cent buckets at extremes
    price_ranges = [
        (1, 5), (5, 10), (10, 15), (15, 20), (20, 25),
        (75, 80), (80, 85), (85, 90), (90, 95), (95, 99)
    ]

    for side in ['yes', 'no']:
        for low, high in price_ranges:
            subset = df[
                (df['trade_price'] >= low) &
                (df['trade_price'] < high) &
                (df['taker_side'] == side)
            ]

            result = calculate_strategy_stats(subset, f"{side.upper()} at {low}-{high}c")

            if result and result.get('unique_markets', 0) >= 50:
                results.append(result)
                status = "VALID" if result['is_valid'] else ""
                edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                print(f"  {status:6} {result['description']}: {result['unique_markets']} mkts, "
                      f"WR {result['win_rate']:.1%} (BE: {result['breakeven_rate']:.1%}), "
                      f"edge {edge_str}, profit ${result['total_profit']:,.0f}")

    return results


def test_very_low_price_no_bets(df: pd.DataFrame) -> List[Dict]:
    """
    NEW HYPOTHESIS: Betting NO when YES price is VERY low (1-5c).

    Theory: When YES is priced at 1-5c, the market thinks it's ~97% likely to lose.
    If actual NO resolution rate is higher than 97%, there's edge.
    """
    print("\n" + "=" * 70)
    print("NEW HYPOTHESIS: NO bets at very low YES prices (1-10c)")
    print("=" * 70)

    results = []

    # When YES is priced 1-10c, NO is priced 90-99c
    # Buying NO at 90-99c means risking 90-99c to win 1-10c
    # This is expensive, but if win rate is high enough...

    price_ranges = [(1, 5), (5, 10), (1, 10)]

    for low, high in price_ranges:
        # NO trades when YES price is in range
        subset = df[
            (df['trade_price'] >= low) &
            (df['trade_price'] < high) &
            (df['taker_side'] == 'no')
        ]

        result = calculate_strategy_stats(subset, f"NO when YES is {low}-{high}c (NO costs {100-high}-{100-low}c)")

        if result and result.get('unique_markets', 0) >= 30:
            results.append(result)
            status = "VALID" if result['is_valid'] else ""
            edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
            print(f"  {status:6} {result['description']}")
            print(f"         Markets: {result['unique_markets']}, WR {result['win_rate']:.1%} (BE: {result['breakeven_rate']:.1%})")
            print(f"         Edge: {edge_str}, Profit: ${result['total_profit']:,.0f}")

    return results


def test_underdog_yes_bets(df: pd.DataFrame) -> List[Dict]:
    """
    Test YES bets at low prices (5-30c range).

    These are "underdog" bets. The question is: do underdogs win more or less
    than their price implies?
    """
    print("\n" + "=" * 70)
    print("UNDERDOG ANALYSIS: YES at low prices")
    print("=" * 70)

    results = []

    price_ranges = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30)]

    for low, high in price_ranges:
        subset = df[
            (df['trade_price'] >= low) &
            (df['trade_price'] < high) &
            (df['taker_side'] == 'yes')
        ]

        result = calculate_strategy_stats(subset, f"YES at {low}-{high}c (underdog)")

        if result and result.get('unique_markets', 0) >= 50:
            results.append(result)
            status = "VALID" if result['is_valid'] else ""
            edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
            print(f"  {status:6} {result['description']}: {result['unique_markets']} mkts, "
                  f"WR {result['win_rate']:.1%} (BE: {result['breakeven_rate']:.1%}), "
                  f"edge {edge_str}, profit ${result['total_profit']:,.0f}")

    return results


def test_whale_vs_retail(df: pd.DataFrame) -> List[Dict]:
    """
    Compare whale trades (500+ contracts) vs retail (<50 contracts).

    Maybe whales are more informed at certain prices?
    """
    print("\n" + "=" * 70)
    print("WHALE VS RETAIL ANALYSIS")
    print("=" * 70)

    results = []

    # Focus on price ranges where we already know there's edge
    for price_range, label in [((80, 90), "80-90c"), ((90, 100), "90-100c")]:
        for side in ['yes', 'no']:
            for size_label, min_size, max_size in [
                ('retail', 1, 50),
                ('medium', 50, 200),
                ('whale', 500, 100000)
            ]:
                subset = df[
                    (df['trade_price'] >= price_range[0]) &
                    (df['trade_price'] < price_range[1]) &
                    (df['taker_side'] == side) &
                    (df['count'] >= min_size) &
                    (df['count'] < max_size)
                ]

                result = calculate_strategy_stats(subset, f"{size_label.upper()} {side.upper()} at {label}")

                if result and result.get('unique_markets', 0) >= 50:
                    results.append(result)
                    status = "VALID" if result['is_valid'] else ""
                    edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                    print(f"  {status:6} {result['description']}: {result['unique_markets']} mkts, "
                          f"WR {result['win_rate']:.1%} (BE: {result['breakeven_rate']:.1%}), "
                          f"edge {edge_str}")

    return results


def test_first_trade_effect(df: pd.DataFrame) -> List[Dict]:
    """
    Does the FIRST trade in a market have more or less edge than later trades?

    Theory: First trades might capture initial mispricing before the market adjusts.
    """
    print("\n" + "=" * 70)
    print("FIRST TRADE EFFECT")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)

    # Sort and rank trades within each market
    df = df.sort_values(['base_market', 'timestamp'])
    df['trade_rank'] = df.groupby('base_market').cumcount() + 1

    for price_range, label in [((70, 80), "70-80c"), ((80, 90), "80-90c"), ((90, 100), "90-100c")]:
        for side in ['yes', 'no']:
            # First trade only
            first_trades = df[
                (df['trade_rank'] == 1) &
                (df['trade_price'] >= price_range[0]) &
                (df['trade_price'] < price_range[1]) &
                (df['taker_side'] == side)
            ]

            result = calculate_strategy_stats(first_trades, f"FIRST TRADE {side.upper()} at {label}")

            if result and result.get('unique_markets', 0) >= 50:
                results.append(result)
                status = "VALID" if result['is_valid'] else ""
                edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                print(f"  {status:6} {result['description']}: {result['unique_markets']} mkts, "
                      f"WR {result['win_rate']:.1%} (BE: {result['breakeven_rate']:.1%}), "
                      f"edge {edge_str}")

            # Later trades (rank > 10)
            later_trades = df[
                (df['trade_rank'] > 10) &
                (df['trade_price'] >= price_range[0]) &
                (df['trade_price'] < price_range[1]) &
                (df['taker_side'] == side)
            ]

            result = calculate_strategy_stats(later_trades, f"LATE TRADES (>10) {side.upper()} at {label}")

            if result and result.get('unique_markets', 0) >= 50:
                results.append(result)
                status = "VALID" if result['is_valid'] else ""
                edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                print(f"  {status:6} {result['description']}: {result['unique_markets']} mkts, "
                      f"WR {result['win_rate']:.1%} (BE: {result['breakeven_rate']:.1%}), "
                      f"edge {edge_str}")

    return results


def test_counter_trend_trades(df: pd.DataFrame) -> List[Dict]:
    """
    What about trades that go AGAINST the recent price direction?

    If price just moved UP, and someone buys NO - are they informed?
    If price just moved DOWN, and someone buys YES - are they catching value?
    """
    print("\n" + "=" * 70)
    print("COUNTER-TREND TRADES")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df = df.sort_values(['base_market', 'timestamp'])

    # Calculate price change from previous trade
    df['prev_price'] = df.groupby('base_market')['trade_price'].shift(1)
    df['price_change'] = df['trade_price'] - df['prev_price']

    # Counter-trend: buying YES after price dropped, or buying NO after price rose
    for price_range, label in [((70, 80), "70-80c"), ((80, 90), "80-90c")]:

        # YES after price drop (>5c)
        counter_yes = df[
            (df['price_change'] < -5) &
            (df['taker_side'] == 'yes') &
            (df['trade_price'] >= price_range[0]) &
            (df['trade_price'] < price_range[1])
        ]

        result = calculate_strategy_stats(counter_yes, f"YES after price drop (>5c) at {label}")

        if result and result.get('unique_markets', 0) >= 50:
            results.append(result)
            status = "VALID" if result['is_valid'] else ""
            edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
            print(f"  {status:6} {result['description']}: {result['unique_markets']} mkts, "
                  f"edge {edge_str}")

        # NO after price rise (>5c)
        counter_no = df[
            (df['price_change'] > 5) &
            (df['taker_side'] == 'no') &
            (df['trade_price'] >= price_range[0]) &
            (df['trade_price'] < price_range[1])
        ]

        result = calculate_strategy_stats(counter_no, f"NO after price rise (>5c) at {label}")

        if result and result.get('unique_markets', 0) >= 50:
            results.append(result)
            status = "VALID" if result['is_valid'] else ""
            edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
            print(f"  {status:6} {result['description']}: {result['unique_markets']} mkts, "
                  f"edge {edge_str}")

    return results


def test_yes_70_80(df: pd.DataFrame) -> Dict:
    """
    Test the YES at 70-80c strategy specifically.

    This is the "step down" from YES at 80-90c.
    """
    print("\n" + "=" * 70)
    print("TESTING: YES at 70-80c")
    print("=" * 70)

    subset = df[
        (df['trade_price'] >= 70) &
        (df['trade_price'] < 80) &
        (df['taker_side'] == 'yes')
    ]

    result = calculate_strategy_stats(subset, "YES at 70-80c")

    if result:
        print(f"\nResults for YES at 70-80c:")
        print(f"  Unique markets: {result['unique_markets']}")
        print(f"  Win rate: {result['win_rate']:.1%}")
        print(f"  Breakeven: {result['breakeven_rate']:.1%}")
        print(f"  Edge: {result['edge']:+.1%}")
        print(f"  Total profit: ${result['total_profit']:,.0f}")
        print(f"  Max concentration: {result['max_market_share']:.1%}")
        print(f"  P-value: {result['p_value']:.6f}")
        print(f"  Valid: {result['is_valid']}")

    return result


def test_no_60_70(df: pd.DataFrame) -> Dict:
    """
    Test the NO at 60-70c strategy.

    Fade moderate favorites.
    """
    print("\n" + "=" * 70)
    print("TESTING: NO at 60-70c")
    print("=" * 70)

    subset = df[
        (df['trade_price'] >= 60) &
        (df['trade_price'] < 70) &
        (df['taker_side'] == 'no')
    ]

    result = calculate_strategy_stats(subset, "NO at 60-70c")

    if result:
        print(f"\nResults for NO at 60-70c:")
        print(f"  Unique markets: {result['unique_markets']}")
        print(f"  Win rate: {result['win_rate']:.1%}")
        print(f"  Breakeven: {result['breakeven_rate']:.1%}")
        print(f"  Edge: {result['edge']:+.1%}")
        print(f"  Total profit: ${result['total_profit']:,.0f}")
        print(f"  Max concentration: {result['max_market_share']:.1%}")
        print(f"  P-value: {result['p_value']:.6f}")
        print(f"  Valid: {result['is_valid']}")

    return result


def main():
    """Run deep dive analysis."""
    print("=" * 70)
    print("SESSION 003 DEEP DIVE: Finding New Strategies")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data()

    all_results = []

    # Run focused tests
    all_results.extend(test_extreme_price_strategies(df))
    all_results.extend(test_very_low_price_no_bets(df))
    all_results.extend(test_underdog_yes_bets(df))
    all_results.extend(test_whale_vs_retail(df))
    all_results.extend(test_first_trade_effect(df))
    all_results.extend(test_counter_trend_trades(df))

    # Specific tests
    yes_70_80 = test_yes_70_80(df)
    if yes_70_80:
        all_results.append(yes_70_80)

    no_60_70 = test_no_60_70(df)
    if no_60_70:
        all_results.append(no_60_70)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: NEW VALID STRATEGIES")
    print("=" * 70)

    valid_strategies = [r for r in all_results if r and r.get('is_valid', False)]

    if valid_strategies:
        print(f"\nFound {len(valid_strategies)} valid strategies:\n")
        for r in sorted(valid_strategies, key=lambda x: x.get('edge', 0), reverse=True):
            print(f"  {r['description']}")
            print(f"    Markets: {r['unique_markets']}, WR: {r['win_rate']:.1%}, BE: {r['breakeven_rate']:.1%}")
            print(f"    Edge: {r['edge']:+.1%}, Profit: ${r['total_profit']:,.0f}")
            print(f"    Concentration: {r['max_market_share']:.1%}, P-value: {r['p_value']:.6f}")
            print()
    else:
        print("\nNo new valid strategies found beyond the known price-based ones.")

    return all_results


if __name__ == '__main__':
    main()
