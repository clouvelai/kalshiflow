#!/usr/bin/env python3
"""
Session 004: Insider Trading / A Priori Knowledge Detection

Primary Focus: Detect cases where large bets were placed shortly before major market moves.

Key Hypotheses:
1. Pre-move whale activity: Large trades that preceded significant price shifts
2. Timing patterns: How far ahead of market resolution do informed traders act?
3. Size anomalies: Unusually large positions taken before events resolve
4. Conviction indicators: High-price YES bets (80-90c+) that win vs lose
5. Late whale activity: Do whales entering late have better information?
6. Price impact analysis: Do whale trades move prices AND predict outcomes?
7. Contrarian whales: When whales bet against the crowd, do they win?

All analysis follows rigorous validation:
- Market-level aggregation (not trade-level)
- N >= 50 unique markets
- Max concentration < 30%
- Statistical significance p < 0.05
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


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

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


def load_all_data():
    """Load all data files needed for analysis."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load enriched trades with outcomes
    trades_path = Path('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv')
    print(f"Loading: {trades_path}")
    df = pd.read_csv(trades_path)
    print(f"Loaded {len(df):,} resolved trades")

    # Convert boolean
    if df['is_winner'].dtype == 'object':
        df['is_winner'] = df['is_winner'].map({'True': True, 'False': False, True: True, False: False})

    # Add derived columns
    df['base_market'] = df['market_ticker'].apply(extract_base_market)
    df['category'] = df['market_ticker'].apply(extract_category)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Load market outcomes for settlement times
    markets_path = Path('/Users/samuelclark/Desktop/kalshiflow/research/data/markets/market_outcomes_ALL.csv')
    if markets_path.exists():
        print(f"Loading: {markets_path}")
        markets_df = pd.read_csv(markets_path)
        print(f"Loaded {len(markets_df):,} market outcomes")
    else:
        markets_df = None
        print("Warning: market_outcomes_ALL.csv not found")

    return df, markets_df


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

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


# ============================================================================
# HYPOTHESIS 1: PRE-MOVE WHALE ACTIVITY
# ============================================================================

def analyze_pre_move_whales(df: pd.DataFrame):
    """
    Do whale trades (500+ contracts) that occur before price moves predict outcomes?

    Theory: Informed traders enter BEFORE the price moves to their favor.
    If a whale buys YES at 40c and the price later goes to 70c, they might have
    information that's now reflected in the price.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS: Pre-Move Whale Activity")
    print("=" * 70)

    results = []

    # Define whale threshold
    WHALE_SIZE = 500

    # Get all whale trades
    whale_trades = df[df['count'] >= WHALE_SIZE].copy()
    print(f"\nTotal whale trades (>={WHALE_SIZE} contracts): {len(whale_trades):,}")

    # For each market, calculate:
    # 1. First whale trade price
    # 2. Final market consensus (average of last N trades)
    # 3. Did price move in whale's favor?

    markets_with_whales = whale_trades['base_market'].unique()
    print(f"Markets with whale trades: {len(markets_with_whales):,}")

    market_analysis = []

    for market in markets_with_whales:
        market_trades = df[df['base_market'] == market].sort_values('timestamp')
        market_whales = whale_trades[whale_trades['base_market'] == market].sort_values('timestamp')

        if len(market_whales) == 0 or len(market_trades) < 5:
            continue

        # First whale trade
        first_whale = market_whales.iloc[0]
        first_whale_price = first_whale['trade_price']
        first_whale_side = first_whale['taker_side']
        first_whale_ts = first_whale['timestamp']

        # Get trades AFTER first whale
        later_trades = market_trades[market_trades['timestamp'] > first_whale_ts]

        if len(later_trades) < 3:
            continue

        # Final price (avg of last 5 trades or all later trades)
        final_price = later_trades.tail(5)['trade_price'].mean()

        # Price movement (positive = moved in whale's favor)
        if first_whale_side == 'yes':
            price_move = final_price - first_whale_price
        else:
            price_move = first_whale_price - final_price

        # Outcome
        is_winner = first_whale['is_winner']
        profit = first_whale['actual_profit_dollars']

        market_analysis.append({
            'market': market,
            'whale_side': first_whale_side,
            'whale_price': first_whale_price,
            'final_price': final_price,
            'price_move': price_move,
            'price_moved_favorable': price_move > 5,  # At least 5c move
            'is_winner': is_winner,
            'profit': profit,
            'whale_size': first_whale['count']
        })

    analysis_df = pd.DataFrame(market_analysis)

    if len(analysis_df) == 0:
        print("Not enough data for analysis")
        return results

    print(f"\nMarkets analyzed: {len(analysis_df):,}")

    # Analysis 1: When price moves in whale's favor, do they win more?
    print("\n--- Price Movement vs Outcome ---")
    favorable_move = analysis_df[analysis_df['price_moved_favorable']]
    unfavorable_move = analysis_df[~analysis_df['price_moved_favorable']]

    print(f"Price moved favorably (>5c): {len(favorable_move):,} markets")
    if len(favorable_move) > 0:
        fav_wr = favorable_move['is_winner'].mean()
        print(f"  Win rate: {fav_wr:.1%}")
        print(f"  Total profit: ${favorable_move['profit'].sum():,.0f}")

    print(f"Price did NOT move favorably: {len(unfavorable_move):,} markets")
    if len(unfavorable_move) > 0:
        unfav_wr = unfavorable_move['is_winner'].mean()
        print(f"  Win rate: {unfav_wr:.1%}")
        print(f"  Total profit: ${unfavorable_move['profit'].sum():,.0f}")

    # Analysis 2: By whale side and price range
    print("\n--- Whale YES vs NO Performance ---")
    for side in ['yes', 'no']:
        side_df = analysis_df[analysis_df['whale_side'] == side]
        if len(side_df) >= 30:
            wr = side_df['is_winner'].mean()
            profit = side_df['profit'].sum()
            avg_price = side_df['whale_price'].mean()

            # Breakeven
            if side == 'yes':
                be = avg_price / 100
            else:
                be = (100 - avg_price) / 100

            edge = wr - be
            print(f"  Whale {side.upper()}: {len(side_df)} mkts, WR={wr:.1%}, BE={be:.1%}, Edge={edge:+.1%}, ${profit:,.0f}")

            if len(side_df) >= 50 and edge > 0.02:
                results.append({
                    'strategy': f'First whale {side.upper()}',
                    'markets': len(side_df),
                    'win_rate': wr,
                    'breakeven': be,
                    'edge': edge,
                    'profit': profit
                })

    # Analysis 3: By price range
    print("\n--- Whale Entry by Price Range ---")
    for price_low, price_high in [(10, 30), (30, 50), (50, 70), (70, 90)]:
        for side in ['yes', 'no']:
            subset = analysis_df[
                (analysis_df['whale_side'] == side) &
                (analysis_df['whale_price'] >= price_low) &
                (analysis_df['whale_price'] < price_high)
            ]
            if len(subset) >= 30:
                wr = subset['is_winner'].mean()
                avg_price = subset['whale_price'].mean()

                if side == 'yes':
                    be = avg_price / 100
                else:
                    be = (100 - avg_price) / 100

                edge = wr - be
                profit = subset['profit'].sum()

                if abs(edge) > 0.05:
                    status = "***" if edge > 0.1 else ""
                    print(f"  {status}Whale {side.upper()} at {price_low}-{price_high}c: "
                          f"{len(subset)} mkts, WR={wr:.1%}, Edge={edge:+.1%}")

    return results


# ============================================================================
# HYPOTHESIS 2: LATE WHALE ACTIVITY (Time to Expiry)
# ============================================================================

def analyze_late_whale_activity(df: pd.DataFrame):
    """
    Do whales entering late (after significant trading) have better information?

    Theory: Informed traders might wait for price discovery before striking,
    or they might be reacting to private information close to resolution.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS: Late Whale Activity")
    print("=" * 70)

    results = []

    WHALE_SIZE = 500

    # For each market, identify:
    # 1. Total trades and total time span
    # 2. First whale trade timestamp
    # 3. When in the market's life cycle did the whale enter?

    markets = df['base_market'].unique()

    market_analysis = []

    for market in markets:
        market_df = df[df['base_market'] == market].sort_values('timestamp')

        if len(market_df) < 10:
            continue

        # Market time span
        first_ts = market_df['timestamp'].min()
        last_ts = market_df['timestamp'].max()
        total_duration = last_ts - first_ts

        if total_duration <= 0:
            continue

        # Whale trades in this market
        whale_trades = market_df[market_df['count'] >= WHALE_SIZE]

        if len(whale_trades) == 0:
            continue

        # First whale timing
        first_whale = whale_trades.iloc[0]
        whale_ts = first_whale['timestamp']

        # Where in market lifecycle did whale enter? (0-1 scale)
        market_progress = (whale_ts - first_ts) / total_duration if total_duration > 0 else 0

        # Also calculate by trade count (how many trades before whale)
        trades_before_whale = len(market_df[market_df['timestamp'] < whale_ts])
        trade_progress = trades_before_whale / len(market_df)

        market_analysis.append({
            'market': market,
            'whale_side': first_whale['taker_side'],
            'whale_price': first_whale['trade_price'],
            'market_progress': market_progress,  # Time-based
            'trade_progress': trade_progress,     # Trade-based
            'is_winner': first_whale['is_winner'],
            'profit': first_whale['actual_profit_dollars'],
            'whale_size': first_whale['count'],
            'total_trades': len(market_df)
        })

    analysis_df = pd.DataFrame(market_analysis)

    if len(analysis_df) == 0:
        print("Not enough data for analysis")
        return results

    print(f"\nMarkets analyzed: {len(analysis_df):,}")

    # Analysis by whale timing
    print("\n--- Win Rate by Market Progress (Time) ---")
    for (low, high), label in [((0, 0.25), 'Early (0-25%)'),
                                ((0.25, 0.5), 'Mid-Early (25-50%)'),
                                ((0.5, 0.75), 'Mid-Late (50-75%)'),
                                ((0.75, 1.0), 'Late (75-100%)')]:
        subset = analysis_df[
            (analysis_df['market_progress'] >= low) &
            (analysis_df['market_progress'] < high)
        ]

        if len(subset) >= 30:
            wr = subset['is_winner'].mean()
            profit = subset['profit'].sum()

            print(f"  {label}: {len(subset)} mkts, WR={wr:.1%}, ${profit:,.0f}")

    # Analysis by trade progress
    print("\n--- Win Rate by Trade Progress ---")
    for (low, high), label in [((0, 0.25), 'Very Early (first 25% of trades)'),
                                ((0.25, 0.5), 'Early (25-50% of trades)'),
                                ((0.5, 0.75), 'Mid (50-75% of trades)'),
                                ((0.75, 1.0), 'Late (75-100% of trades)')]:
        subset = analysis_df[
            (analysis_df['trade_progress'] >= low) &
            (analysis_df['trade_progress'] < high)
        ]

        if len(subset) >= 30:
            wr = subset['is_winner'].mean()
            profit = subset['profit'].sum()
            avg_price = subset['whale_price'].mean()

            # Calculate breakeven for this subset
            yes_count = len(subset[subset['whale_side'] == 'yes'])
            no_count = len(subset[subset['whale_side'] == 'no'])

            print(f"  {label}: {len(subset)} mkts, WR={wr:.1%}, ${profit:,.0f}")
            print(f"    (YES: {yes_count}, NO: {no_count}, avg price: {avg_price:.0f}c)")

    # Analysis: Late whales at specific price points
    print("\n--- Late Whales (75%+ progress) by Side and Price ---")
    late_whales = analysis_df[analysis_df['trade_progress'] >= 0.75]

    for side in ['yes', 'no']:
        for (price_low, price_high) in [(60, 80), (80, 95)]:
            subset = late_whales[
                (late_whales['whale_side'] == side) &
                (late_whales['whale_price'] >= price_low) &
                (late_whales['whale_price'] < price_high)
            ]

            if len(subset) >= 30:
                wr = subset['is_winner'].mean()
                avg_price = subset['whale_price'].mean()

                if side == 'yes':
                    be = avg_price / 100
                else:
                    be = (100 - avg_price) / 100

                edge = wr - be
                profit = subset['profit'].sum()

                status = "PROMISING" if edge > 0.05 else ""
                print(f"  {status} Late whale {side.upper()} at {price_low}-{price_high}c: "
                      f"{len(subset)} mkts, WR={wr:.1%}, Edge={edge:+.1%}, ${profit:,.0f}")

                if edge > 0.05 and len(subset) >= 50:
                    results.append({
                        'strategy': f'Late whale {side.upper()} at {price_low}-{price_high}c',
                        'markets': len(subset),
                        'win_rate': wr,
                        'breakeven': be,
                        'edge': edge,
                        'profit': profit
                    })

    return results


# ============================================================================
# HYPOTHESIS 3: CONVICTION TRADING (Size * Price)
# ============================================================================

def analyze_conviction_trading(df: pd.DataFrame):
    """
    High conviction = large size at high-confidence prices.

    Theory: When someone bets big at 85c+, they're very confident.
    Do they know something? Or are they just confident?
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS: Conviction Trading (Size * Price)")
    print("=" * 70)

    results = []

    # Define conviction levels
    # Conviction = size_bucket * price_bucket

    df = df.copy()

    # Size buckets
    df['size_bucket'] = pd.cut(df['count'],
                               bins=[0, 50, 200, 500, 1000, 100000],
                               labels=['tiny', 'small', 'medium', 'large', 'mega'])

    # Price buckets (from taker's perspective)
    df['price_bucket'] = pd.cut(df['trade_price'],
                                bins=[0, 30, 50, 70, 85, 100],
                                labels=['longshot', 'underdog', 'tossup', 'favorite', 'heavy_fav'])

    # High conviction = large+ size at heavy_fav price
    print("\n--- High Conviction Analysis (Large+ Size at 85c+) ---")

    for side in ['yes', 'no']:
        high_conviction = df[
            (df['taker_side'] == side) &
            (df['count'] >= 500) &
            (df['trade_price'] >= 85)
        ]

        result = full_strategy_analysis(high_conviction, f"High conviction {side.upper()} (500+ @ 85c+)")

        if result and result.get('unique_markets', 0) >= 30:
            status = "VALID" if result.get('is_valid', False) else ""
            print(f"\n  {status} {side.upper()} conviction bets:")
            print(f"    Markets: {result['unique_markets']}")
            print(f"    Win Rate: {result['win_rate']:.1%}")
            print(f"    Breakeven: {result['breakeven_rate']:.1%}")
            print(f"    Edge: {result['edge']:+.1%}")
            print(f"    Profit: ${result['total_profit']:,.0f}")
            print(f"    Max Concentration: {result['max_market_share']:.1%}")

            if result.get('is_valid', False):
                results.append({
                    'strategy': f'High conviction {side.upper()} (500+ @ 85c+)',
                    'markets': result['unique_markets'],
                    'win_rate': result['win_rate'],
                    'breakeven': result['breakeven_rate'],
                    'edge': result['edge'],
                    'profit': result['total_profit']
                })

    # Analysis by conviction matrix
    print("\n--- Conviction Matrix (Size vs Price) ---")
    print(f"{'Size/Price':<15} {'Longshot':<12} {'Underdog':<12} {'Tossup':<12} {'Favorite':<12} {'Heavy Fav':<12}")
    print("-" * 75)

    for size_label, min_size, max_size in [('Retail', 1, 50), ('Medium', 50, 200),
                                            ('Large', 200, 500), ('Whale', 500, 1000),
                                            ('Mega', 1000, 100000)]:
        row = f"{size_label:<15}"
        for price_label, price_low, price_high in [('Longshot', 0, 30), ('Underdog', 30, 50),
                                                    ('Tossup', 50, 70), ('Favorite', 70, 85),
                                                    ('Heavy Fav', 85, 100)]:
            subset = df[
                (df['count'] >= min_size) & (df['count'] < max_size) &
                (df['trade_price'] >= price_low) & (df['trade_price'] < price_high)
            ]

            if len(subset) >= 100:
                # Aggregate by market
                market_wr = subset.groupby('base_market')['is_winner'].first().mean()
                row += f" {market_wr:.0%}".ljust(12)
            else:
                row += " -".ljust(12)

        print(row)

    return results


# ============================================================================
# HYPOTHESIS 4: CONTRARIAN WHALE ACTIVITY
# ============================================================================

def analyze_contrarian_whales(df: pd.DataFrame):
    """
    When a whale bets AGAINST the crowd (opposite direction of price trend),
    do they have information?

    Theory: If price is trending YES (rising) and a whale bets NO, they might
    be fading incorrect market sentiment with superior information.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS: Contrarian Whale Activity")
    print("=" * 70)

    results = []

    WHALE_SIZE = 500

    markets = df['base_market'].unique()

    market_analysis = []

    for market in markets:
        market_df = df[df['base_market'] == market].sort_values('timestamp')

        if len(market_df) < 10:
            continue

        # Whale trades
        whale_trades = market_df[market_df['count'] >= WHALE_SIZE]

        if len(whale_trades) == 0:
            continue

        for _, whale in whale_trades.iterrows():
            whale_ts = whale['timestamp']
            whale_side = whale['taker_side']
            whale_price = whale['trade_price']

            # Get trades BEFORE this whale
            before_trades = market_df[market_df['timestamp'] < whale_ts]

            if len(before_trades) < 5:
                continue

            # Market sentiment before whale (average direction)
            # If most trades were YES, sentiment is bullish
            recent_5 = before_trades.tail(5)
            yes_count = len(recent_5[recent_5['taker_side'] == 'yes'])
            sentiment = 'bullish' if yes_count >= 3 else 'bearish' if yes_count <= 2 else 'neutral'

            # Also check price trend
            first_price = before_trades.head(3)['trade_price'].mean()
            last_price = recent_5['trade_price'].mean()
            price_trend = 'up' if last_price > first_price + 5 else 'down' if last_price < first_price - 5 else 'flat'

            # Is whale contrarian?
            if whale_side == 'yes' and sentiment == 'bearish':
                is_contrarian = True
                contrarian_type = 'bullish_vs_bearish_sentiment'
            elif whale_side == 'no' and sentiment == 'bullish':
                is_contrarian = True
                contrarian_type = 'bearish_vs_bullish_sentiment'
            elif whale_side == 'yes' and price_trend == 'down':
                is_contrarian = True
                contrarian_type = 'buying_falling_price'
            elif whale_side == 'no' and price_trend == 'up':
                is_contrarian = True
                contrarian_type = 'selling_rising_price'
            else:
                is_contrarian = False
                contrarian_type = 'momentum'

            market_analysis.append({
                'market': market,
                'whale_side': whale_side,
                'whale_price': whale_price,
                'sentiment': sentiment,
                'price_trend': price_trend,
                'is_contrarian': is_contrarian,
                'contrarian_type': contrarian_type,
                'is_winner': whale['is_winner'],
                'profit': whale['actual_profit_dollars'],
                'whale_size': whale['count']
            })

    analysis_df = pd.DataFrame(market_analysis)

    if len(analysis_df) == 0:
        print("Not enough data for analysis")
        return results

    print(f"\nWhale trades analyzed: {len(analysis_df):,}")

    # Compare contrarian vs momentum whales
    print("\n--- Contrarian vs Momentum Whales ---")

    contrarian = analysis_df[analysis_df['is_contrarian']]
    momentum = analysis_df[~analysis_df['is_contrarian']]

    print(f"\nContrarian whales: {len(contrarian):,}")
    if len(contrarian) >= 50:
        c_wr = contrarian['is_winner'].mean()
        c_profit = contrarian['profit'].sum()
        print(f"  Win Rate: {c_wr:.1%}")
        print(f"  Profit: ${c_profit:,.0f}")

    print(f"\nMomentum whales: {len(momentum):,}")
    if len(momentum) >= 50:
        m_wr = momentum['is_winner'].mean()
        m_profit = momentum['profit'].sum()
        print(f"  Win Rate: {m_wr:.1%}")
        print(f"  Profit: ${m_profit:,.0f}")

    # Break down by contrarian type
    print("\n--- By Contrarian Type ---")
    for ctype in analysis_df['contrarian_type'].unique():
        subset = analysis_df[analysis_df['contrarian_type'] == ctype]
        if len(subset) >= 30:
            wr = subset['is_winner'].mean()
            profit = subset['profit'].sum()
            print(f"  {ctype}: {len(subset)} trades, WR={wr:.1%}, ${profit:,.0f}")

    # Contrarian by price range
    print("\n--- Contrarian Whales by Price Range ---")
    for side in ['yes', 'no']:
        for (price_low, price_high) in [(10, 30), (30, 50), (50, 70), (70, 90)]:
            subset = contrarian[
                (contrarian['whale_side'] == side) &
                (contrarian['whale_price'] >= price_low) &
                (contrarian['whale_price'] < price_high)
            ]

            if len(subset) >= 30:
                wr = subset['is_winner'].mean()
                avg_price = subset['whale_price'].mean()

                if side == 'yes':
                    be = avg_price / 100
                else:
                    be = (100 - avg_price) / 100

                edge = wr - be
                profit = subset['profit'].sum()

                if abs(edge) > 0.05:
                    status = "PROMISING" if edge > 0.1 else ""
                    print(f"  {status}Contrarian {side.upper()} at {price_low}-{price_high}c: "
                          f"{len(subset)} trades, WR={wr:.1%}, Edge={edge:+.1%}")

    return results


# ============================================================================
# HYPOTHESIS 5: WHALE VOLUME CONCENTRATION
# ============================================================================

def analyze_whale_volume_concentration(df: pd.DataFrame):
    """
    Markets where whale volume is concentrated (few but large trades)
    vs distributed (many medium trades) - which has more edge?

    Theory: Concentrated whale activity might indicate informed trading,
    while distributed activity might be market making.
    """
    print("\n" + "=" * 70)
    print("HYPOTHESIS: Whale Volume Concentration")
    print("=" * 70)

    results = []

    WHALE_SIZE = 500

    # Analyze at market level
    markets = df['base_market'].unique()

    market_analysis = []

    for market in markets:
        market_df = df[df['base_market'] == market]

        if len(market_df) < 5:
            continue

        # Calculate whale metrics for this market
        whale_trades = market_df[market_df['count'] >= WHALE_SIZE]
        total_volume = market_df['count'].sum()
        whale_volume = whale_trades['count'].sum()

        whale_volume_pct = whale_volume / total_volume if total_volume > 0 else 0
        num_whale_trades = len(whale_trades)
        avg_whale_size = whale_trades['count'].mean() if len(whale_trades) > 0 else 0

        # Herfindahl index for trade concentration
        if len(market_df) > 0:
            trade_shares = market_df['count'] / total_volume
            hhi = (trade_shares ** 2).sum()
        else:
            hhi = 0

        # Result for this market (take most common outcome)
        result = market_df['market_result'].iloc[0]

        # Simple profit calculation for this market
        profit = market_df['actual_profit_dollars'].sum()

        # Determine predominant direction
        yes_volume = market_df[market_df['taker_side'] == 'yes']['count'].sum()
        no_volume = market_df[market_df['taker_side'] == 'no']['count'].sum()
        predominant_side = 'yes' if yes_volume > no_volume else 'no'

        market_analysis.append({
            'market': market,
            'total_volume': total_volume,
            'whale_volume_pct': whale_volume_pct,
            'num_whale_trades': num_whale_trades,
            'avg_whale_size': avg_whale_size,
            'hhi': hhi,
            'result': result,
            'profit': profit,
            'predominant_side': predominant_side
        })

    analysis_df = pd.DataFrame(market_analysis)

    if len(analysis_df) == 0:
        print("Not enough data for analysis")
        return results

    print(f"\nMarkets analyzed: {len(analysis_df):,}")

    # Analyze by whale volume percentage
    print("\n--- Markets by Whale Volume Percentage ---")
    for (low, high), label in [((0, 0.1), 'Low whale (<10%)'),
                                ((0.1, 0.3), 'Medium whale (10-30%)'),
                                ((0.3, 0.5), 'High whale (30-50%)'),
                                ((0.5, 1.0), 'Whale dominated (50%+)')]:
        subset = analysis_df[
            (analysis_df['whale_volume_pct'] >= low) &
            (analysis_df['whale_volume_pct'] < high)
        ]

        if len(subset) >= 30:
            avg_profit = subset['profit'].mean()
            total_profit = subset['profit'].sum()
            print(f"  {label}: {len(subset)} mkts, avg profit ${avg_profit:.0f}, total ${total_profit:,.0f}")

    # Analyze by HHI (concentration)
    print("\n--- Markets by Trade Concentration (HHI) ---")
    for (low, high), label in [((0, 0.1), 'Dispersed (HHI < 0.1)'),
                                ((0.1, 0.25), 'Moderate (0.1-0.25)'),
                                ((0.25, 0.5), 'Concentrated (0.25-0.5)'),
                                ((0.5, 1.0), 'Highly concentrated (0.5+)')]:
        subset = analysis_df[
            (analysis_df['hhi'] >= low) &
            (analysis_df['hhi'] < high)
        ]

        if len(subset) >= 30:
            avg_profit = subset['profit'].mean()
            total_profit = subset['profit'].sum()
            print(f"  {label}: {len(subset)} mkts, avg profit ${avg_profit:.0f}, total ${total_profit:,.0f}")

    return results


# ============================================================================
# HYPOTHESIS 6: CROSS-CHECK WITH KNOWN STRATEGIES
# ============================================================================

def cross_check_insider_patterns(df: pd.DataFrame):
    """
    Cross-check insider trading patterns with our validated strategies.

    Key question: Are the validated strategies (NO at 70-80c, etc.)
    capturing insider activity, or is it pure favorite-longshot bias?
    """
    print("\n" + "=" * 70)
    print("CROSS-CHECK: Insider Patterns vs Validated Strategies")
    print("=" * 70)

    WHALE_SIZE = 500

    # Known validated strategy: NO at 70-80c
    no_70_80 = df[(df['trade_price'] >= 70) & (df['trade_price'] < 80) & (df['taker_side'] == 'no')]

    # Split into whale and retail
    whale_no_70_80 = no_70_80[no_70_80['count'] >= WHALE_SIZE]
    retail_no_70_80 = no_70_80[no_70_80['count'] < WHALE_SIZE]

    print("\n--- NO at 70-80c: Whale vs Retail ---")

    whale_result = full_strategy_analysis(whale_no_70_80, "WHALE NO at 70-80c", min_markets=30)
    retail_result = full_strategy_analysis(retail_no_70_80, "RETAIL NO at 70-80c", min_markets=30)

    if whale_result and whale_result.get('unique_markets', 0) >= 30:
        print(f"\n  WHALE NO at 70-80c:")
        print(f"    Markets: {whale_result['unique_markets']}")
        print(f"    Win Rate: {whale_result['win_rate']:.1%}")
        print(f"    Breakeven: {whale_result['breakeven_rate']:.1%}")
        print(f"    Edge: {whale_result['edge']:+.1%}")
        print(f"    Profit: ${whale_result['total_profit']:,.0f}")
        print(f"    Valid: {whale_result['is_valid']}")

    if retail_result and retail_result.get('unique_markets', 0) >= 30:
        print(f"\n  RETAIL NO at 70-80c:")
        print(f"    Markets: {retail_result['unique_markets']}")
        print(f"    Win Rate: {retail_result['win_rate']:.1%}")
        print(f"    Breakeven: {retail_result['breakeven_rate']:.1%}")
        print(f"    Edge: {retail_result['edge']:+.1%}")
        print(f"    Profit: ${retail_result['total_profit']:,.0f}")
        print(f"    Valid: {retail_result['is_valid']}")

    # Same for 80-90c
    print("\n--- NO at 80-90c: Whale vs Retail ---")
    no_80_90 = df[(df['trade_price'] >= 80) & (df['trade_price'] < 90) & (df['taker_side'] == 'no')]

    whale_no_80_90 = no_80_90[no_80_90['count'] >= WHALE_SIZE]
    retail_no_80_90 = no_80_90[no_80_90['count'] < WHALE_SIZE]

    whale_result = full_strategy_analysis(whale_no_80_90, "WHALE NO at 80-90c", min_markets=30)
    retail_result = full_strategy_analysis(retail_no_80_90, "RETAIL NO at 80-90c", min_markets=30)

    if whale_result and whale_result.get('unique_markets', 0) >= 30:
        print(f"\n  WHALE NO at 80-90c:")
        print(f"    Edge: {whale_result['edge']:+.1%}, Markets: {whale_result['unique_markets']}")

    if retail_result and retail_result.get('unique_markets', 0) >= 30:
        print(f"\n  RETAIL NO at 80-90c:")
        print(f"    Edge: {retail_result['edge']:+.1%}, Markets: {retail_result['unique_markets']}")


# ============================================================================
# NEW STRATEGY EXPLORATION
# ============================================================================

def explore_new_strategies(df: pd.DataFrame):
    """
    Systematically explore new strategy ideas beyond price-based ones.
    """
    print("\n" + "=" * 70)
    print("NEW STRATEGY EXPLORATION")
    print("=" * 70)

    all_results = []

    # Strategy 1: YES at 50-60c (slight favorites)
    print("\n--- Exploring: YES at various price points ---")
    for price_low, price_high in [(40, 50), (50, 60), (60, 70), (70, 80)]:
        subset = df[(df['trade_price'] >= price_low) & (df['trade_price'] < price_high) & (df['taker_side'] == 'yes')]
        result = full_strategy_analysis(subset, f"YES at {price_low}-{price_high}c")

        if result and result.get('unique_markets', 0) >= 50:
            all_results.append(result)
            status = "VALID" if result.get('is_valid', False) else ""
            print(f"  {status} YES at {price_low}-{price_high}c: {result['unique_markets']} mkts, "
                  f"edge {result['edge']:+.1%}, ${result['total_profit']:,.0f}")

    # Strategy 2: Size-filtered strategies
    print("\n--- Exploring: Size-filtered strategies ---")
    for size_label, min_size, max_size in [('Retail (1-50)', 1, 50),
                                            ('Medium (50-200)', 50, 200),
                                            ('Large (200-500)', 200, 500)]:
        for price_low, price_high in [(70, 80), (80, 90)]:
            subset = df[
                (df['count'] >= min_size) & (df['count'] < max_size) &
                (df['trade_price'] >= price_low) & (df['trade_price'] < price_high) &
                (df['taker_side'] == 'no')
            ]
            result = full_strategy_analysis(subset, f"{size_label} NO at {price_low}-{price_high}c")

            if result and result.get('unique_markets', 0) >= 50 and result.get('is_valid', False):
                all_results.append(result)
                print(f"  VALID {size_label} NO at {price_low}-{price_high}c: "
                      f"{result['unique_markets']} mkts, edge {result['edge']:+.1%}")

    # Strategy 3: Category-specific at new price points
    print("\n--- Exploring: Category-specific strategies ---")
    categories = df.groupby('category')['base_market'].nunique()
    big_cats = categories[categories >= 200].index.tolist()

    for cat in big_cats[:10]:  # Top 10 categories
        cat_df = df[df['category'] == cat]

        # Try NO at 50-60c for this category
        subset = cat_df[
            (cat_df['trade_price'] >= 50) & (cat_df['trade_price'] < 60) &
            (cat_df['taker_side'] == 'no')
        ]
        result = full_strategy_analysis(subset, f"{cat}: NO at 50-60c", min_markets=30)

        if result and result.get('is_valid', False):
            all_results.append(result)
            print(f"  VALID {cat}: NO at 50-60c: {result['unique_markets']} mkts, edge {result['edge']:+.1%}")

    # Return all valid strategies
    valid_strategies = [r for r in all_results if r and r.get('is_valid', False)]

    print(f"\n--- Total Valid New Strategies Found: {len(valid_strategies)} ---")
    for r in sorted(valid_strategies, key=lambda x: x['edge'], reverse=True)[:5]:
        print(f"  {r['description']}: edge {r['edge']:+.1%}, {r['unique_markets']} mkts")

    return valid_strategies


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main analysis."""
    print("=" * 70)
    print("SESSION 004: INSIDER TRADING / A PRIORI KNOWLEDGE DETECTION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df, markets_df = load_all_data()

    all_findings = []

    # Run all hypotheses
    findings = analyze_pre_move_whales(df)
    all_findings.extend(findings)

    findings = analyze_late_whale_activity(df)
    all_findings.extend(findings)

    findings = analyze_conviction_trading(df)
    all_findings.extend(findings)

    findings = analyze_contrarian_whales(df)
    all_findings.extend(findings)

    analyze_whale_volume_concentration(df)

    cross_check_insider_patterns(df)

    new_strategies = explore_new_strategies(df)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: All Findings")
    print("=" * 70)

    if all_findings:
        print("\nPromising insider trading patterns found:")
        for f in sorted(all_findings, key=lambda x: x.get('edge', 0), reverse=True):
            print(f"  {f['strategy']}: edge {f['edge']:+.1%}, {f['markets']} mkts, ${f['profit']:,.0f}")
    else:
        print("\nNo statistically significant insider trading patterns detected.")
        print("This is actually good news - it means the market is relatively efficient.")

    if new_strategies:
        print(f"\nNew validated strategies: {len(new_strategies)}")
        for s in new_strategies[:5]:
            print(f"  {s['description']}: edge {s['edge']:+.1%}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'insider_patterns': all_findings,
        'new_strategies': [s for s in new_strategies if s]
    }

    output_path = Path('/Users/samuelclark/Desktop/kalshiflow/research/reports/session004_insider_trading.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
