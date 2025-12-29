#!/usr/bin/env python3
"""
Session 003: Fresh Hypothesis Testing
======================================

New areas to explore:
1. Time-to-expiry effects (do markets behave differently as settlement approaches?)
2. Trade intensity patterns (high vs low volume markets)
3. Price momentum within markets (continuation vs reversal)
4. Contract size patterns (large trades vs small trades)
5. Day-of-week effects
6. First trade vs later trades (market pricing evolution)
7. Consecutive trade patterns (momentum in trade direction)
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime, timedelta
import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def extract_market_category(ticker: str) -> str:
    """Extract the market category prefix from a ticker."""
    match = re.match(r'^(KX[A-Z]+)', ticker)
    return match.group(1) if match else 'UNKNOWN'


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


def binomial_test(wins: int, total: int, fair_rate: float = None) -> float:
    """Calculate p-value for win rate being different from breakeven."""
    if total == 0:
        return 1.0
    if fair_rate is None:
        fair_rate = 0.5
    result = stats.binomtest(wins, total, fair_rate, alternative='two-sided')
    return result.pvalue


def get_breakeven_rate(price_bucket: str, side: str) -> float:
    """Calculate the breakeven win rate for a price bucket and side."""
    try:
        parts = price_bucket.replace('c', '').split('-')
        low = int(parts[0])
        high = int(parts[1])
        midpoint = (low + high) / 2.0
        if side == 'yes':
            return midpoint / 100.0
        else:
            return (100 - midpoint) / 100.0
    except:
        return 0.5


def validate_strategy(
    market_df: pd.DataFrame,
    description: str,
    min_markets: int = 50
) -> Dict:
    """
    Validate a strategy using market-level analysis.

    market_df should have one row per market with:
    - base_market: market identifier
    - is_winner: whether the strategy won on this market
    - actual_profit_dollars: profit from this market
    - cost_dollars: cost of trades in this market
    - taker_side: 'yes' or 'no'
    - avg_price: average entry price
    """
    if len(market_df) < min_markets:
        return {
            'description': description,
            'is_valid': False,
            'unique_markets': len(market_df),
            'reason': f'Not enough markets ({len(market_df)} < {min_markets})'
        }

    # Calculate metrics
    wins = int(market_df['is_winner'].sum())
    total_markets = len(market_df)
    win_rate = wins / total_markets

    total_profit = float(market_df['actual_profit_dollars'].sum())
    total_cost = float(market_df['cost_dollars'].sum())
    roi = total_profit / total_cost if total_cost > 0 else 0

    # Concentration check
    max_share = calculate_max_market_share(market_df['actual_profit_dollars'])

    # Determine average price for breakeven calculation
    avg_price = market_df['avg_price'].mean() if 'avg_price' in market_df.columns else 50
    side = market_df['taker_side'].iloc[0] if 'taker_side' in market_df.columns else 'yes'

    # Calculate breakeven
    if side == 'yes':
        breakeven_rate = avg_price / 100.0
    else:
        breakeven_rate = (100 - avg_price) / 100.0

    # P-value test
    p_value = binomial_test(wins, total_markets, breakeven_rate)

    # Validation checks
    passes_market_count = total_markets >= min_markets
    passes_concentration = max_share < 0.3
    passes_significance = p_value < 0.05

    is_valid = passes_market_count and passes_concentration and passes_significance

    edge = win_rate - breakeven_rate

    return {
        'description': description,
        'unique_markets': int(total_markets),
        'wins': wins,
        'win_rate': round(float(win_rate), 4),
        'breakeven_rate': round(float(breakeven_rate), 4),
        'edge': round(float(edge), 4),
        'total_profit': round(float(total_profit), 2),
        'total_cost': round(float(total_cost), 2),
        'roi': round(float(roi), 4),
        'max_market_share': round(float(max_share), 4),
        'p_value': round(float(p_value), 6),
        'passes_market_count': passes_market_count,
        'passes_concentration': passes_concentration,
        'passes_significance': passes_significance,
        'is_valid': is_valid,
        'avg_price': round(float(avg_price), 2),
        'side': side
    }


def load_and_prepare_data():
    """Load and prepare the enriched trades data."""
    # Try multiple possible data locations
    data_paths = [
        Path('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv'),
        Path('/Users/samuelclark/Desktop/kalshiflow/backend/training/reports/enriched_trades_final.csv'),
    ]

    for path in data_paths:
        if path.exists():
            print(f"Loading data from: {path}")
            df = pd.read_csv(path)
            print(f"Loaded {len(df):,} trades")
            return df

    raise FileNotFoundError("Could not find enriched trades data")


def hypothesis_1_trade_intensity(df: pd.DataFrame) -> List[Dict]:
    """
    H1: Trade intensity patterns

    Do markets with high trade count behave differently?
    Maybe heavily-traded markets are more efficient, while low-traded markets have edge.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Trade Intensity Patterns")
    print("=" * 70)

    results = []

    # Add derived columns
    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['price_bucket'] = df['trade_price'].apply(get_price_bucket)

    # Only resolved trades
    resolved = df[df['market_result'].notna()].copy()

    # Calculate trades per market
    market_trade_counts = resolved.groupby('base_market').size()

    # Split into low vs high traded markets
    median_trades = market_trade_counts.median()
    q25 = market_trade_counts.quantile(0.25)
    q75 = market_trade_counts.quantile(0.75)

    print(f"\nTrade count distribution:")
    print(f"  Min: {market_trade_counts.min()}, Max: {market_trade_counts.max()}")
    print(f"  Q25: {q25:.0f}, Median: {median_trades:.0f}, Q75: {q75:.0f}")

    # Test different trade count thresholds
    thresholds = [
        ('very_low', 1, 5),       # 1-5 trades per market
        ('low', 5, 20),           # 5-20 trades per market
        ('medium', 20, 100),      # 20-100 trades per market
        ('high', 100, 500),       # 100-500 trades per market
        ('very_high', 500, 10000) # 500+ trades per market
    ]

    for price_bucket in ['70-80c', '80-90c', '90-100c']:
        for side in ['yes', 'no']:
            for label, min_trades, max_trades in thresholds:
                # Get markets in this trade count range
                markets_in_range = market_trade_counts[
                    (market_trade_counts >= min_trades) &
                    (market_trade_counts < max_trades)
                ].index

                # Filter trades to these markets and price/side
                subset = resolved[
                    (resolved['base_market'].isin(markets_in_range)) &
                    (resolved['price_bucket'] == price_bucket) &
                    (resolved['taker_side'] == side)
                ].copy()

                if len(subset) == 0:
                    continue

                # Aggregate to market level
                market_stats = subset.groupby('base_market').agg({
                    'is_winner': 'first',  # Same result for all trades in market
                    'actual_profit_dollars': 'sum',
                    'cost_dollars': 'sum',
                    'trade_price': 'mean',
                    'taker_side': 'first'
                }).reset_index()

                market_stats.columns = ['base_market', 'is_winner', 'actual_profit_dollars',
                                        'cost_dollars', 'avg_price', 'taker_side']

                description = f"Trade intensity {label} ({min_trades}-{max_trades}): {side.upper()} at {price_bucket}"
                result = validate_strategy(market_stats, description)

                if result['unique_markets'] >= 30:  # Report if substantial sample
                    results.append(result)
                    status = "VALID" if result['is_valid'] else ""
                    edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                    print(f"  {status:6} {description}: {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge {edge_str}, ${result['total_profit']:,.0f}")

    return results


def hypothesis_2_contract_size(df: pd.DataFrame) -> List[Dict]:
    """
    H2: Contract size patterns

    Do large trades (whales) predict differently than small trades (retail)?
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Contract Size Patterns")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['price_bucket'] = df['trade_price'].apply(get_price_bucket)

    resolved = df[df['market_result'].notna()].copy()

    # Trade size buckets (contracts per trade)
    size_thresholds = [
        ('tiny', 1, 10),        # 1-10 contracts
        ('small', 10, 50),      # 10-50 contracts
        ('medium', 50, 200),    # 50-200 contracts
        ('large', 200, 500),    # 200-500 contracts
        ('whale', 500, 10000)   # 500+ contracts
    ]

    print(f"\nContract size distribution:")
    print(f"  Min: {resolved['count'].min()}, Max: {resolved['count'].max()}")
    print(f"  Median: {resolved['count'].median():.0f}, Mean: {resolved['count'].mean():.0f}")

    for price_bucket in ['70-80c', '80-90c', '90-100c']:
        for side in ['yes', 'no']:
            for label, min_size, max_size in size_thresholds:
                subset = resolved[
                    (resolved['count'] >= min_size) &
                    (resolved['count'] < max_size) &
                    (resolved['price_bucket'] == price_bucket) &
                    (resolved['taker_side'] == side)
                ].copy()

                if len(subset) == 0:
                    continue

                # Aggregate to market level (take first trade outcome per market)
                market_stats = subset.groupby('base_market').agg({
                    'is_winner': 'first',
                    'actual_profit_dollars': 'sum',
                    'cost_dollars': 'sum',
                    'trade_price': 'mean',
                    'taker_side': 'first'
                }).reset_index()

                market_stats.columns = ['base_market', 'is_winner', 'actual_profit_dollars',
                                        'cost_dollars', 'avg_price', 'taker_side']

                description = f"Contract size {label} ({min_size}-{max_size}): {side.upper()} at {price_bucket}"
                result = validate_strategy(market_stats, description)

                if result['unique_markets'] >= 30:
                    results.append(result)
                    status = "VALID" if result['is_valid'] else ""
                    edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                    print(f"  {status:6} {description}: {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge {edge_str}, ${result['total_profit']:,.0f}")

    return results


def hypothesis_3_first_vs_later_trades(df: pd.DataFrame) -> List[Dict]:
    """
    H3: First trades vs later trades

    Are early trades in a market more or less informative?
    Maybe first trades capture initial mispricing.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: First Trades vs Later Trades")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['price_bucket'] = df['trade_price'].apply(get_price_bucket)

    resolved = df[df['market_result'].notna()].copy()

    # Sort by timestamp within each market and rank trades
    resolved = resolved.sort_values(['base_market', 'timestamp'])
    resolved['trade_rank'] = resolved.groupby('base_market').cumcount() + 1

    print(f"\nTrades per market distribution:")
    trades_per_market = resolved.groupby('base_market').size()
    print(f"  Median: {trades_per_market.median():.0f}, Max: {trades_per_market.max()}")

    # Test different trade positions
    position_tests = [
        ('first', 1, 1),       # Only first trade
        ('early', 1, 3),       # First 3 trades
        ('late', 10, 100),     # Trades 10+
        ('very_late', 50, 1000) # Trades 50+
    ]

    for price_bucket in ['70-80c', '80-90c', '90-100c']:
        for side in ['yes', 'no']:
            for label, min_rank, max_rank in position_tests:
                subset = resolved[
                    (resolved['trade_rank'] >= min_rank) &
                    (resolved['trade_rank'] <= max_rank) &
                    (resolved['price_bucket'] == price_bucket) &
                    (resolved['taker_side'] == side)
                ].copy()

                if len(subset) == 0:
                    continue

                # Aggregate to market level
                market_stats = subset.groupby('base_market').agg({
                    'is_winner': 'first',
                    'actual_profit_dollars': 'sum',
                    'cost_dollars': 'sum',
                    'trade_price': 'mean',
                    'taker_side': 'first'
                }).reset_index()

                market_stats.columns = ['base_market', 'is_winner', 'actual_profit_dollars',
                                        'cost_dollars', 'avg_price', 'taker_side']

                description = f"Trade position {label} (rank {min_rank}-{max_rank}): {side.upper()} at {price_bucket}"
                result = validate_strategy(market_stats, description)

                if result['unique_markets'] >= 30:
                    results.append(result)
                    status = "VALID" if result['is_valid'] else ""
                    edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                    print(f"  {status:6} {description}: {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge {edge_str}, ${result['total_profit']:,.0f}")

    return results


def hypothesis_4_consecutive_trades(df: pd.DataFrame) -> List[Dict]:
    """
    H4: Consecutive trade patterns

    When multiple consecutive trades are in the same direction,
    is that predictive? (Momentum vs mean reversion)
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Consecutive Trade Direction Patterns")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['price_bucket'] = df['trade_price'].apply(get_price_bucket)

    resolved = df[df['market_result'].notna()].copy()

    # Sort by timestamp within each market
    resolved = resolved.sort_values(['base_market', 'timestamp'])

    # For each trade, count how many consecutive same-direction trades preceded it
    def count_consecutive_same(group):
        sides = group['taker_side'].values
        counts = []
        for i in range(len(sides)):
            count = 0
            for j in range(i-1, -1, -1):
                if sides[j] == sides[i]:
                    count += 1
                else:
                    break
            counts.append(count)
        return pd.Series(counts, index=group.index)

    print("Calculating consecutive trade patterns...")
    resolved['consecutive_same'] = resolved.groupby('base_market', group_keys=False).apply(count_consecutive_same)

    # Test different consecutive trade thresholds
    consecutive_tests = [
        ('standalone', 0, 0),      # No consecutive trades before
        ('after_1_same', 1, 1),    # After 1 consecutive same direction
        ('after_2_same', 2, 2),    # After 2 consecutive same direction
        ('after_3plus', 3, 100),   # After 3+ consecutive same direction
    ]

    for price_bucket in ['70-80c', '80-90c', '90-100c']:
        for side in ['yes', 'no']:
            for label, min_consec, max_consec in consecutive_tests:
                subset = resolved[
                    (resolved['consecutive_same'] >= min_consec) &
                    (resolved['consecutive_same'] <= max_consec) &
                    (resolved['price_bucket'] == price_bucket) &
                    (resolved['taker_side'] == side)
                ].copy()

                if len(subset) == 0:
                    continue

                # Aggregate to market level
                market_stats = subset.groupby('base_market').agg({
                    'is_winner': 'first',
                    'actual_profit_dollars': 'sum',
                    'cost_dollars': 'sum',
                    'trade_price': 'mean',
                    'taker_side': 'first'
                }).reset_index()

                market_stats.columns = ['base_market', 'is_winner', 'actual_profit_dollars',
                                        'cost_dollars', 'avg_price', 'taker_side']

                description = f"Consecutive {label}: {side.upper()} at {price_bucket}"
                result = validate_strategy(market_stats, description)

                if result['unique_markets'] >= 30:
                    results.append(result)
                    status = "VALID" if result['is_valid'] else ""
                    edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                    print(f"  {status:6} {description}: {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge {edge_str}, ${result['total_profit']:,.0f}")

    return results


def hypothesis_5_day_of_week(df: pd.DataFrame) -> List[Dict]:
    """
    H5: Day of week effects

    Are certain days more exploitable? Weekend vs weekday patterns?
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: Day of Week Patterns")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['price_bucket'] = df['trade_price'].apply(get_price_bucket)

    # Parse datetime
    if 'datetime' in df.columns:
        df['trade_datetime'] = pd.to_datetime(df['datetime'])
    else:
        df['trade_datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    df['day_of_week'] = df['trade_datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['day_of_week'].isin([5, 6])

    resolved = df[df['market_result'].notna()].copy()

    print(f"\nTrade distribution by day:")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day in range(7):
        count = len(resolved[resolved['day_of_week'] == day])
        print(f"  {day_names[day]}: {count:,} trades")

    # Test by day type
    day_tests = [
        ('weekend', [5, 6]),
        ('weekday', [0, 1, 2, 3, 4]),
        ('monday', [0]),
        ('friday', [4]),
        ('sunday', [6]),
    ]

    for price_bucket in ['70-80c', '80-90c', '90-100c']:
        for side in ['yes', 'no']:
            for label, days in day_tests:
                subset = resolved[
                    (resolved['day_of_week'].isin(days)) &
                    (resolved['price_bucket'] == price_bucket) &
                    (resolved['taker_side'] == side)
                ].copy()

                if len(subset) == 0:
                    continue

                # Aggregate to market level
                market_stats = subset.groupby('base_market').agg({
                    'is_winner': 'first',
                    'actual_profit_dollars': 'sum',
                    'cost_dollars': 'sum',
                    'trade_price': 'mean',
                    'taker_side': 'first'
                }).reset_index()

                market_stats.columns = ['base_market', 'is_winner', 'actual_profit_dollars',
                                        'cost_dollars', 'avg_price', 'taker_side']

                description = f"Day {label}: {side.upper()} at {price_bucket}"
                result = validate_strategy(market_stats, description)

                if result['unique_markets'] >= 30:
                    results.append(result)
                    status = "VALID" if result['is_valid'] else ""
                    edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                    print(f"  {status:6} {description}: {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge {edge_str}, ${result['total_profit']:,.0f}")

    return results


def hypothesis_6_price_movement_within_market(df: pd.DataFrame) -> List[Dict]:
    """
    H6: Price movement patterns within a market

    After price moves up/down, is there momentum or reversion?
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 6: Price Movement Patterns Within Market")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)

    resolved = df[df['market_result'].notna()].copy()

    # Sort by timestamp and calculate price changes within market
    resolved = resolved.sort_values(['base_market', 'timestamp'])
    resolved['prev_price'] = resolved.groupby('base_market')['trade_price'].shift(1)
    resolved['price_change'] = resolved['trade_price'] - resolved['prev_price']

    print(f"\nPrice change distribution:")
    print(f"  Mean: {resolved['price_change'].mean():.2f}c")
    print(f"  Std: {resolved['price_change'].std():.2f}c")

    # Categorize price movements
    resolved['price_movement'] = pd.cut(
        resolved['price_change'],
        bins=[-100, -5, -2, 2, 5, 100],
        labels=['big_drop', 'small_drop', 'stable', 'small_rise', 'big_rise']
    )

    # Test trades after different price movements
    price_move_tests = [
        'big_drop', 'small_drop', 'stable', 'small_rise', 'big_rise'
    ]

    for price_bucket in ['70-80c', '80-90c', '90-100c']:
        for side in ['yes', 'no']:
            for movement in price_move_tests:
                subset = resolved[
                    (resolved['price_movement'] == movement) &
                    (resolved['trade_price'].apply(get_price_bucket) == price_bucket) &
                    (resolved['taker_side'] == side)
                ].copy()

                if len(subset) == 0:
                    continue

                # Aggregate to market level
                market_stats = subset.groupby('base_market').agg({
                    'is_winner': 'first',
                    'actual_profit_dollars': 'sum',
                    'cost_dollars': 'sum',
                    'trade_price': 'mean',
                    'taker_side': 'first'
                }).reset_index()

                market_stats.columns = ['base_market', 'is_winner', 'actual_profit_dollars',
                                        'cost_dollars', 'avg_price', 'taker_side']

                description = f"After {movement}: {side.upper()} at {price_bucket}"
                result = validate_strategy(market_stats, description)

                if result['unique_markets'] >= 30:
                    results.append(result)
                    status = "VALID" if result['is_valid'] else ""
                    edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                    print(f"  {status:6} {description}: {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge {edge_str}, ${result['total_profit']:,.0f}")

    return results


def hypothesis_7_extreme_prices(df: pd.DataFrame) -> List[Dict]:
    """
    H7: Extreme price analysis (more granular buckets)

    Test very specific price ranges for edge.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 7: Granular Price Analysis (5c buckets)")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)

    resolved = df[df['market_result'].notna()].copy()

    # More granular price buckets (5 cent)
    price_ranges = [
        (1, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30),
        (70, 75), (75, 80), (80, 85), (85, 90), (90, 95), (95, 99)
    ]

    for side in ['yes', 'no']:
        for low, high in price_ranges:
            subset = resolved[
                (resolved['trade_price'] >= low) &
                (resolved['trade_price'] < high) &
                (resolved['taker_side'] == side)
            ].copy()

            if len(subset) == 0:
                continue

            # Aggregate to market level
            market_stats = subset.groupby('base_market').agg({
                'is_winner': 'first',
                'actual_profit_dollars': 'sum',
                'cost_dollars': 'sum',
                'trade_price': 'mean',
                'taker_side': 'first'
            }).reset_index()

            market_stats.columns = ['base_market', 'is_winner', 'actual_profit_dollars',
                                    'cost_dollars', 'avg_price', 'taker_side']

            description = f"{side.upper()} at {low}-{high}c"
            result = validate_strategy(market_stats, description)

            if result['unique_markets'] >= 30:
                results.append(result)
                status = "VALID" if result['is_valid'] else ""
                edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                print(f"  {status:6} {description}: {result['unique_markets']} mkts, "
                      f"{result['win_rate']:.1%} WR (BE: {result['breakeven_rate']:.1%}), "
                      f"edge {edge_str}, ${result['total_profit']:,.0f}")

    return results


def hypothesis_8_dollar_volume_per_market(df: pd.DataFrame) -> List[Dict]:
    """
    H8: Dollar volume per market

    Markets with high dollar volume might be more efficient.
    Low dollar volume markets might have more edge.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS 8: Dollar Volume Per Market")
    print("=" * 70)

    results = []

    df = df.copy()
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['price_bucket'] = df['trade_price'].apply(get_price_bucket)

    resolved = df[df['market_result'].notna()].copy()

    # Calculate dollar volume per market
    market_volume = resolved.groupby('base_market')['cost_dollars'].sum()

    print(f"\nDollar volume per market:")
    print(f"  Min: ${market_volume.min():.0f}, Max: ${market_volume.max():,.0f}")
    print(f"  Median: ${market_volume.median():,.0f}, Mean: ${market_volume.mean():,.0f}")

    # Volume buckets
    volume_thresholds = [
        ('tiny', 0, 100),           # $0-100
        ('small', 100, 1000),       # $100-1k
        ('medium', 1000, 10000),    # $1k-10k
        ('large', 10000, 100000),   # $10k-100k
        ('huge', 100000, 10000000)  # $100k+
    ]

    for price_bucket in ['70-80c', '80-90c', '90-100c']:
        for side in ['yes', 'no']:
            for label, min_vol, max_vol in volume_thresholds:
                # Get markets in this volume range
                markets_in_range = market_volume[
                    (market_volume >= min_vol) &
                    (market_volume < max_vol)
                ].index

                subset = resolved[
                    (resolved['base_market'].isin(markets_in_range)) &
                    (resolved['price_bucket'] == price_bucket) &
                    (resolved['taker_side'] == side)
                ].copy()

                if len(subset) == 0:
                    continue

                # Aggregate to market level
                market_stats = subset.groupby('base_market').agg({
                    'is_winner': 'first',
                    'actual_profit_dollars': 'sum',
                    'cost_dollars': 'sum',
                    'trade_price': 'mean',
                    'taker_side': 'first'
                }).reset_index()

                market_stats.columns = ['base_market', 'is_winner', 'actual_profit_dollars',
                                        'cost_dollars', 'avg_price', 'taker_side']

                description = f"Volume {label} (${min_vol}-${max_vol}): {side.upper()} at {price_bucket}"
                result = validate_strategy(market_stats, description)

                if result['unique_markets'] >= 30:
                    results.append(result)
                    status = "VALID" if result['is_valid'] else ""
                    edge_str = f"+{result['edge']:.1%}" if result['edge'] > 0 else f"{result['edge']:.1%}"
                    print(f"  {status:6} {description}: {result['unique_markets']} mkts, "
                          f"{result['win_rate']:.1%} WR, edge {edge_str}, ${result['total_profit']:,.0f}")

    return results


def main():
    """Run all fresh hypothesis tests."""
    print("=" * 70)
    print("SESSION 003: FRESH HYPOTHESIS TESTING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_and_prepare_data()

    # Convert is_winner to boolean if needed
    if df['is_winner'].dtype == 'object':
        df['is_winner'] = df['is_winner'].map({'True': True, 'False': False, True: True, False: False})

    # All trades in this file are resolved (have market_result)
    resolved_count = len(df[df['market_result'].notna()])
    print(f"\nResolved trades: {resolved_count:,}")

    all_results = []

    # Run all hypothesis tests
    all_results.extend(hypothesis_1_trade_intensity(df))
    all_results.extend(hypothesis_2_contract_size(df))
    all_results.extend(hypothesis_3_first_vs_later_trades(df))
    all_results.extend(hypothesis_4_consecutive_trades(df))
    all_results.extend(hypothesis_5_day_of_week(df))
    all_results.extend(hypothesis_6_price_movement_within_market(df))
    all_results.extend(hypothesis_7_extreme_prices(df))
    all_results.extend(hypothesis_8_dollar_volume_per_market(df))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid_results = [r for r in all_results if r.get('is_valid', False)]
    profitable_valid = [r for r in valid_results if r.get('total_profit', 0) > 0]

    print(f"\nTotal hypotheses tested: {len(all_results)}")
    print(f"Valid strategies found: {len(valid_results)}")
    print(f"Profitable & valid: {len(profitable_valid)}")

    if profitable_valid:
        print("\n*** PROFITABLE VALID STRATEGIES ***")
        for r in sorted(profitable_valid, key=lambda x: x['total_profit'], reverse=True)[:15]:
            print(f"  {r['description']}")
            print(f"    Markets: {r['unique_markets']}, Win Rate: {r['win_rate']:.1%} (BE: {r['breakeven_rate']:.1%})")
            print(f"    Edge: {r['edge']:+.1%}, Profit: ${r['total_profit']:,.0f}, Max Share: {r['max_market_share']:.1%}")
            print(f"    P-value: {r['p_value']:.6f}")
            print()

    # Save results (convert numpy types to Python types)
    output_path = Path('/Users/samuelclark/Desktop/kalshiflow/research/reports/session003_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    serializable_results = convert_to_serializable(all_results)

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == '__main__':
    main()
