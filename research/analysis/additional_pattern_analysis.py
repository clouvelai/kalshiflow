#!/usr/bin/env python3
"""
Additional Pattern Analysis for Kalshi Trades

This script explores unconventional patterns that haven't been discovered yet:
1. Price movement sequences - what happens BEFORE profitable trades?
2. Inter-market correlations - when one market moves, do related markets follow?
3. Specific timing granularity - specific hours/minutes patterns
4. Spread market opportunities - price inefficiencies between related markets
5. Volume clustering patterns - what happens after a burst of volume?
6. Market age patterns - new markets vs old markets behavior
7. Close-time precision - trades in final minutes vs hours
8. Side-switching patterns - when traders flip from YES to NO
9. Unusual price levels - psychological anchors (25c, 50c, 75c)
10. Consecutive trade patterns - multiple trades in same market within seconds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import json
import re

# Configuration
REPORTS_DIR = Path(__file__).parent.parent.parent.parent / "training" / "reports"
TRADES_FILE = REPORTS_DIR / "enriched_trades_final.csv"
OUTCOMES_FILE = REPORTS_DIR / "market_outcomes.csv"
OUTPUT_FILE = REPORTS_DIR / "additional_patterns.json"


def load_data():
    """Load and prepare data."""
    print("Loading trade data...")
    trades = pd.read_csv(TRADES_FILE)
    outcomes = pd.read_csv(OUTCOMES_FILE)

    # Convert timestamp to datetime
    trades['datetime'] = pd.to_datetime(trades['timestamp'], unit='ms')
    trades['hour'] = trades['datetime'].dt.hour
    trades['minute'] = trades['datetime'].dt.minute
    trades['second'] = trades['datetime'].dt.second
    trades['day_of_week'] = trades['datetime'].dt.dayofweek

    # Filter to resolved trades only (non-null actual_profit_dollars)
    resolved = trades[trades['actual_profit_dollars'].notna()].copy()

    print(f"Loaded {len(trades)} total trades, {len(resolved)} resolved")
    return trades, resolved, outcomes


def analyze_psychological_price_levels(resolved):
    """
    Pattern 1: Psychological Price Anchors

    Hypothesis: Traders behave differently at round numbers (25c, 50c, 75c).
    These levels may have different win rates due to psychological effects.
    """
    print("\n=== PATTERN 1: Psychological Price Levels ===")

    results = {}

    # Define psychological levels
    psychological_levels = {
        'exact_25': (24, 26),    # Exactly around 25c
        'exact_50': (49, 51),    # Exactly around 50c
        'exact_75': (74, 76),    # Exactly around 75c
        'exact_10': (9, 11),     # Exactly around 10c
        'exact_90': (89, 91),    # Exactly around 90c
    }

    for name, (low, high) in psychological_levels.items():
        subset = resolved[(resolved['trade_price'] >= low) & (resolved['trade_price'] <= high)]
        if len(subset) < 50:
            continue

        wins = len(subset[subset['is_winner'] == True])
        win_rate = wins / len(subset)
        total_profit = subset['actual_profit_dollars'].sum()
        avg_profit_per_trade = total_profit / len(subset) if len(subset) > 0 else 0

        # Compare to adjacent non-psychological levels
        adjacent_low = resolved[(resolved['trade_price'] >= low - 5) & (resolved['trade_price'] < low)]
        adjacent_high = resolved[(resolved['trade_price'] > high) & (resolved['trade_price'] <= high + 5)]

        adjacent = pd.concat([adjacent_low, adjacent_high])
        adjacent_wins = len(adjacent[adjacent['is_winner'] == True])
        adjacent_win_rate = adjacent_wins / len(adjacent) if len(adjacent) > 0 else 0

        edge = win_rate - adjacent_win_rate

        results[name] = {
            'trades': len(subset),
            'win_rate': round(win_rate, 4),
            'adjacent_win_rate': round(adjacent_win_rate, 4),
            'edge_vs_adjacent': round(edge, 4),
            'total_profit': round(total_profit, 2),
            'avg_profit_per_trade': round(avg_profit_per_trade, 2),
            'roi': round(total_profit / subset['cost_dollars'].sum(), 4) if subset['cost_dollars'].sum() > 0 else 0
        }

        print(f"  {name}: {len(subset)} trades, win_rate={win_rate:.2%}, edge={edge:+.2%}, profit=${total_profit:,.0f}")

    # Analyze YES vs NO at psychological levels
    print("\n  Side analysis at 50c:")
    at_50 = resolved[(resolved['trade_price'] >= 49) & (resolved['trade_price'] <= 51)]
    for side in ['yes', 'no']:
        side_subset = at_50[at_50['taker_side'] == side]
        if len(side_subset) > 50:
            wins = len(side_subset[side_subset['is_winner'] == True])
            win_rate = wins / len(side_subset)
            profit = side_subset['actual_profit_dollars'].sum()
            print(f"    {side.upper()}: {len(side_subset)} trades, win_rate={win_rate:.2%}, profit=${profit:,.0f}")
            results[f'at_50_{side}'] = {
                'trades': len(side_subset),
                'win_rate': round(win_rate, 4),
                'total_profit': round(profit, 2)
            }

    return results


def analyze_consecutive_trades(resolved):
    """
    Pattern 2: Consecutive Trade Patterns

    Hypothesis: When multiple trades happen in the same market within seconds,
    there may be information cascades or momentum effects.
    """
    print("\n=== PATTERN 2: Consecutive Trade Patterns ===")

    results = {}

    # Sort by market and timestamp
    sorted_trades = resolved.sort_values(['market_ticker', 'timestamp'])

    # Find trades within 5 seconds of each other in the same market
    sorted_trades['prev_ticker'] = sorted_trades['market_ticker'].shift(1)
    sorted_trades['prev_timestamp'] = sorted_trades['timestamp'].shift(1)
    sorted_trades['prev_side'] = sorted_trades['taker_side'].shift(1)

    # Same market, within 5 seconds
    sorted_trades['is_rapid_follow'] = (
        (sorted_trades['market_ticker'] == sorted_trades['prev_ticker']) &
        ((sorted_trades['timestamp'] - sorted_trades['prev_timestamp']) <= 5000)
    )

    # Rapid follow trades
    rapid_follows = sorted_trades[sorted_trades['is_rapid_follow'] == True]

    # Same direction vs opposite direction
    same_direction = rapid_follows[rapid_follows['taker_side'] == rapid_follows['prev_side']]
    opposite_direction = rapid_follows[rapid_follows['taker_side'] != rapid_follows['prev_side']]

    if len(same_direction) > 100:
        wins = len(same_direction[same_direction['is_winner'] == True])
        win_rate = wins / len(same_direction)
        profit = same_direction['actual_profit_dollars'].sum()
        roi = profit / same_direction['cost_dollars'].sum() if same_direction['cost_dollars'].sum() > 0 else 0

        results['same_direction_rapid'] = {
            'trades': len(same_direction),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'Trade follows previous trade in same direction within 5 seconds'
        }
        print(f"  Same direction rapid: {len(same_direction)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}")

    if len(opposite_direction) > 100:
        wins = len(opposite_direction[opposite_direction['is_winner'] == True])
        win_rate = wins / len(opposite_direction)
        profit = opposite_direction['actual_profit_dollars'].sum()
        roi = profit / opposite_direction['cost_dollars'].sum() if opposite_direction['cost_dollars'].sum() > 0 else 0

        results['opposite_direction_rapid'] = {
            'trades': len(opposite_direction),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'Trade follows previous trade in OPPOSITE direction within 5 seconds (CONTRARIAN)'
        }
        print(f"  Opposite direction rapid: {len(opposite_direction)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}")

    # Analyze burst patterns (3+ trades in 10 seconds)
    print("\n  Analyzing trade bursts...")

    # Group trades by market and time window
    market_groups = sorted_trades.groupby('market_ticker')

    burst_trades = []
    for ticker, group in market_groups:
        if len(group) < 3:
            continue

        group = group.sort_values('timestamp')
        timestamps = group['timestamp'].values

        for i in range(len(timestamps) - 2):
            # Check if 3+ trades within 10 seconds
            window_end = timestamps[i] + 10000
            trades_in_window = sum(1 for t in timestamps[i:] if t <= window_end)

            if trades_in_window >= 3:
                # Get the LAST trade in the burst (following the crowd)
                window_trades = group[(group['timestamp'] >= timestamps[i]) &
                                     (group['timestamp'] <= window_end)]
                burst_trades.append(window_trades.iloc[-1])

    if burst_trades:
        burst_df = pd.DataFrame(burst_trades)
        wins = len(burst_df[burst_df['is_winner'] == True])
        win_rate = wins / len(burst_df)
        profit = burst_df['actual_profit_dollars'].sum()
        roi = profit / burst_df['cost_dollars'].sum() if burst_df['cost_dollars'].sum() > 0 else 0

        results['trade_burst_follow'] = {
            'trades': len(burst_df),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'Last trade in a burst of 3+ trades within 10 seconds'
        }
        print(f"  Trade burst (last in burst): {len(burst_df)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}")

    return results


def analyze_minute_precision(resolved):
    """
    Pattern 3: Minute-Level Timing Patterns

    Hypothesis: Specific minutes within hours may have different profitability
    (e.g., market opens, half-hour marks, end-of-hour).
    """
    print("\n=== PATTERN 3: Minute-Level Timing Patterns ===")

    results = {}

    # Key minute windows
    minute_windows = {
        'market_open_first_5min': (0, 4),     # First 5 minutes of hour
        'market_open_last_5min': (55, 59),    # Last 5 minutes of hour
        'half_hour_window': (28, 32),         # Around half hour
        'mid_hour': (25, 35),                 # Middle of hour
    }

    for name, (start, end) in minute_windows.items():
        subset = resolved[(resolved['minute'] >= start) & (resolved['minute'] <= end)]
        if len(subset) < 100:
            continue

        wins = len(subset[subset['is_winner'] == True])
        win_rate = wins / len(subset)
        total_profit = subset['actual_profit_dollars'].sum()
        roi = total_profit / subset['cost_dollars'].sum() if subset['cost_dollars'].sum() > 0 else 0

        # Compare to overall
        overall_wins = len(resolved[resolved['is_winner'] == True])
        overall_win_rate = overall_wins / len(resolved)
        edge = win_rate - overall_win_rate

        results[name] = {
            'trades': len(subset),
            'win_rate': round(win_rate, 4),
            'edge_vs_overall': round(edge, 4),
            'total_profit': round(total_profit, 2),
            'roi': round(roi, 4)
        }
        print(f"  {name}: {len(subset)} trades, win_rate={win_rate:.2%}, edge={edge:+.2%}, ROI={roi:.2%}")

    # Analyze specific high-performing minutes
    print("\n  Top/Bottom 5 minutes by ROI (min 200 trades):")
    minute_stats = []
    for minute in range(60):
        subset = resolved[resolved['minute'] == minute]
        if len(subset) >= 200:
            profit = subset['actual_profit_dollars'].sum()
            cost = subset['cost_dollars'].sum()
            roi = profit / cost if cost > 0 else 0
            minute_stats.append({
                'minute': minute,
                'trades': len(subset),
                'roi': roi,
                'profit': profit
            })

    minute_stats.sort(key=lambda x: x['roi'], reverse=True)

    print("  TOP 5:")
    for stat in minute_stats[:5]:
        print(f"    Minute {stat['minute']:02d}: {stat['trades']} trades, ROI={stat['roi']:.2%}, profit=${stat['profit']:,.0f}")
        results[f'minute_{stat["minute"]:02d}'] = {
            'trades': stat['trades'],
            'roi': round(stat['roi'], 4),
            'total_profit': round(stat['profit'], 2),
            'rank': 'top'
        }

    print("  BOTTOM 5:")
    for stat in minute_stats[-5:]:
        print(f"    Minute {stat['minute']:02d}: {stat['trades']} trades, ROI={stat['roi']:.2%}, profit=${stat['profit']:,.0f}")
        results[f'minute_{stat["minute"]:02d}'] = {
            'trades': stat['trades'],
            'roi': round(stat['roi'], 4),
            'total_profit': round(stat['profit'], 2),
            'rank': 'bottom'
        }

    return results


def analyze_price_movement_sequences(resolved):
    """
    Pattern 4: Price Movement Before Profitable Trades

    Hypothesis: Profitable trades may be preceded by specific price movements
    (e.g., price drops before a winning YES, or price rises before a winning NO).
    """
    print("\n=== PATTERN 4: Price Movement Sequences ===")

    results = {}

    # Sort by market and timestamp
    sorted_trades = resolved.sort_values(['market_ticker', 'timestamp'])

    # Get previous trade price in same market
    sorted_trades['prev_ticker'] = sorted_trades['market_ticker'].shift(1)
    sorted_trades['prev_price'] = sorted_trades['trade_price'].shift(1)

    # Only consider consecutive trades in same market
    same_market = sorted_trades[sorted_trades['market_ticker'] == sorted_trades['prev_ticker']].copy()
    same_market['price_change'] = same_market['trade_price'] - same_market['prev_price']

    # Categorize price changes
    same_market['price_direction'] = pd.cut(
        same_market['price_change'],
        bins=[-100, -5, -1, 1, 5, 100],
        labels=['big_drop', 'small_drop', 'flat', 'small_rise', 'big_rise']
    )

    # Analyze by price direction and side
    for direction in ['big_drop', 'small_drop', 'flat', 'small_rise', 'big_rise']:
        for side in ['yes', 'no']:
            subset = same_market[(same_market['price_direction'] == direction) &
                                (same_market['taker_side'] == side)]
            if len(subset) < 100:
                continue

            wins = len(subset[subset['is_winner'] == True])
            win_rate = wins / len(subset)
            profit = subset['actual_profit_dollars'].sum()
            roi = profit / subset['cost_dollars'].sum() if subset['cost_dollars'].sum() > 0 else 0

            key = f'{direction}_{side}'
            results[key] = {
                'trades': len(subset),
                'win_rate': round(win_rate, 4),
                'total_profit': round(profit, 2),
                'roi': round(roi, 4)
            }
            print(f"  {key}: {len(subset)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}")

    # Most interesting: contrarian patterns (buy YES after drop, buy NO after rise)
    print("\n  CONTRARIAN PATTERNS:")
    contrarian_yes = same_market[(same_market['price_direction'].isin(['big_drop', 'small_drop'])) &
                                 (same_market['taker_side'] == 'yes')]
    contrarian_no = same_market[(same_market['price_direction'].isin(['big_rise', 'small_rise'])) &
                                (same_market['taker_side'] == 'no')]

    for name, subset in [('buy_yes_after_drop', contrarian_yes), ('buy_no_after_rise', contrarian_no)]:
        if len(subset) > 100:
            wins = len(subset[subset['is_winner'] == True])
            win_rate = wins / len(subset)
            profit = subset['actual_profit_dollars'].sum()
            roi = profit / subset['cost_dollars'].sum() if subset['cost_dollars'].sum() > 0 else 0

            results[name] = {
                'trades': len(subset),
                'win_rate': round(win_rate, 4),
                'total_profit': round(profit, 2),
                'roi': round(roi, 4),
                'description': f'Contrarian: {name.replace("_", " ")}'
            }
            print(f"  {name}: {len(subset)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    return results


def analyze_market_category_combinations(resolved):
    """
    Pattern 5: Market Category Deep Dive

    Hypothesis: Specific combinations of category + side + price range
    may have outsized returns.
    """
    print("\n=== PATTERN 5: Category + Side + Price Combinations ===")

    results = {}

    # Extract market category from ticker
    resolved = resolved.copy()
    resolved['category'] = resolved['market_ticker'].str.extract(r'(KX[A-Z]+)')[0]

    # Define price buckets
    resolved['price_bucket'] = pd.cut(
        resolved['trade_price'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['longshot', 'underdog', 'mid', 'favorite', 'strong_fav']
    )

    # Group by category + side + price bucket
    combinations = resolved.groupby(['category', 'taker_side', 'price_bucket']).agg({
        'actual_profit_dollars': ['count', 'sum'],
        'cost_dollars': 'sum',
        'is_winner': ['sum', 'mean']
    }).reset_index()

    combinations.columns = ['category', 'side', 'price_bucket', 'trades', 'profit', 'cost', 'wins', 'win_rate']
    combinations['roi'] = combinations['profit'] / combinations['cost']

    # Filter to significant combinations (min 50 trades, positive ROI)
    profitable = combinations[(combinations['trades'] >= 50) & (combinations['roi'] > 0.1)]
    profitable = profitable.sort_values('roi', ascending=False)

    print("  Top 15 profitable combinations (min 50 trades, ROI > 10%):")
    for _, row in profitable.head(15).iterrows():
        key = f"{row['category']}_{row['side']}_{row['price_bucket']}"
        results[key] = {
            'trades': int(row['trades']),
            'win_rate': round(row['win_rate'], 4),
            'roi': round(row['roi'], 4),
            'total_profit': round(row['profit'], 2)
        }
        print(f"    {key}: {int(row['trades'])} trades, win_rate={row['win_rate']:.2%}, ROI={row['roi']:.2%}, profit=${row['profit']:,.0f}")

    return results


def analyze_size_momentum(resolved):
    """
    Pattern 6: Size Momentum

    Hypothesis: When big trades come in, following the same direction
    with the next trade may be profitable (or fading may be).
    """
    print("\n=== PATTERN 6: Size Momentum (Follow Big Trades) ===")

    results = {}

    # Sort by market and timestamp
    sorted_trades = resolved.sort_values(['market_ticker', 'timestamp'])

    # Get previous trade info
    sorted_trades['prev_ticker'] = sorted_trades['market_ticker'].shift(1)
    sorted_trades['prev_side'] = sorted_trades['taker_side'].shift(1)
    sorted_trades['prev_count'] = sorted_trades['count'].shift(1)
    sorted_trades['prev_cost'] = sorted_trades['cost_dollars'].shift(1)

    # Only same market
    same_market = sorted_trades[sorted_trades['market_ticker'] == sorted_trades['prev_ticker']]

    # Define "big" trades (top 10% by cost in that market)
    threshold = same_market['prev_cost'].quantile(0.90)

    # Trades that follow big trades
    after_big = same_market[same_market['prev_cost'] >= threshold]

    # Following direction
    follow_big = after_big[after_big['taker_side'] == after_big['prev_side']]
    fade_big = after_big[after_big['taker_side'] != after_big['prev_side']]

    for name, subset in [('follow_big_trade', follow_big), ('fade_big_trade', fade_big)]:
        if len(subset) < 100:
            continue

        wins = len(subset[subset['is_winner'] == True])
        win_rate = wins / len(subset)
        profit = subset['actual_profit_dollars'].sum()
        roi = profit / subset['cost_dollars'].sum() if subset['cost_dollars'].sum() > 0 else 0

        results[name] = {
            'trades': len(subset),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4)
        }
        print(f"  {name}: {len(subset)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    # Whale-sized specific (>=500 contracts)
    print("\n  After WHALE trades (>=500 contracts):")
    whale_trades = same_market[same_market['prev_count'] >= 500]

    follow_whale = whale_trades[whale_trades['taker_side'] == whale_trades['prev_side']]
    fade_whale = whale_trades[whale_trades['taker_side'] != whale_trades['prev_side']]

    for name, subset in [('follow_whale', follow_whale), ('fade_whale', fade_whale)]:
        if len(subset) < 50:
            continue

        wins = len(subset[subset['is_winner'] == True])
        win_rate = wins / len(subset)
        profit = subset['actual_profit_dollars'].sum()
        roi = profit / subset['cost_dollars'].sum() if subset['cost_dollars'].sum() > 0 else 0

        results[name] = {
            'trades': len(subset),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4)
        }
        print(f"    {name}: {len(subset)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    return results


def analyze_leverage_patterns(resolved):
    """
    Pattern 7: Leverage Ratio Patterns

    Hypothesis: Specific leverage ratios may indicate mispriced markets.
    High leverage with specific other characteristics may be goldmines.
    """
    print("\n=== PATTERN 7: Leverage Ratio Edge Cases ===")

    results = {}

    # Filter out extreme leverage (likely data errors)
    filtered = resolved[(resolved['leverage_ratio'] > 0.01) & (resolved['leverage_ratio'] < 100)]

    # Leverage buckets
    leverage_buckets = [
        ('extreme_leverage', 10, 100),
        ('high_leverage', 4, 10),
        ('medium_leverage', 2, 4),
        ('low_leverage', 1, 2),
        ('negative_leverage', 0.5, 1),
        ('very_negative', 0.01, 0.5),
    ]

    for name, low, high in leverage_buckets:
        subset = filtered[(filtered['leverage_ratio'] >= low) & (filtered['leverage_ratio'] < high)]
        if len(subset) < 100:
            continue

        wins = len(subset[subset['is_winner'] == True])
        win_rate = wins / len(subset)
        profit = subset['actual_profit_dollars'].sum()
        roi = profit / subset['cost_dollars'].sum() if subset['cost_dollars'].sum() > 0 else 0

        results[name] = {
            'trades': len(subset),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'leverage_range': f'{low}-{high}'
        }
        print(f"  {name} ({low}-{high}x): {len(subset)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}")

    # High leverage + whale combination
    print("\n  High leverage + whale trades:")
    high_lev_whale = filtered[(filtered['leverage_ratio'] >= 4) & (filtered['count'] >= 100)]
    if len(high_lev_whale) > 50:
        wins = len(high_lev_whale[high_lev_whale['is_winner'] == True])
        win_rate = wins / len(high_lev_whale)
        profit = high_lev_whale['actual_profit_dollars'].sum()
        roi = profit / high_lev_whale['cost_dollars'].sum() if high_lev_whale['cost_dollars'].sum() > 0 else 0

        results['high_leverage_whale'] = {
            'trades': len(high_lev_whale),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4)
        }
        print(f"    High leverage + whale: {len(high_lev_whale)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    return results


def analyze_inter_market_correlations(resolved):
    """
    Pattern 8: Inter-Market Correlations

    Hypothesis: When one market in a category moves, related markets follow.
    E.g., if KXNFLGAME-Team1 gets YES trades, KXNFLGAME-Team2 might get NO trades.
    """
    print("\n=== PATTERN 8: Inter-Market Correlations ===")

    results = {}

    # Extract base event from ticker (e.g., "KXNFLGAME-25DEC27BALGB" from "KXNFLGAME-25DEC27BALGB-BAL")
    resolved = resolved.copy()

    # Split by last hyphen to get base event
    resolved['base_event'] = resolved['market_ticker'].str.rsplit('-', n=1).str[0]

    # Find markets with multiple tickers (e.g., both teams in a game)
    event_counts = resolved.groupby('base_event')['market_ticker'].nunique()
    multi_ticker_events = event_counts[event_counts > 1].index

    multi_ticker_trades = resolved[resolved['base_event'].isin(multi_ticker_events)]

    print(f"  Found {len(multi_ticker_events)} multi-ticker events")

    # For game markets, analyze if betting opposite to flow is profitable
    game_events = multi_ticker_trades[multi_ticker_trades['base_event'].str.contains('GAME|FIGHT', na=False)]

    if len(game_events) > 100:
        # Group by event and analyze
        event_groups = game_events.groupby('base_event')

        contrarian_profits = []
        for event, group in event_groups:
            if len(group) < 10:
                continue

            # Get dominant direction
            yes_volume = group[group['taker_side'] == 'yes']['count'].sum()
            no_volume = group[group['taker_side'] == 'no']['count'].sum()

            if yes_volume == 0 and no_volume == 0:
                continue

            dominant = 'yes' if yes_volume > no_volume else 'no'
            dominant_ratio = max(yes_volume, no_volume) / (yes_volume + no_volume)

            if dominant_ratio > 0.7:  # Strong directional flow
                # Get the other side's trades
                contrarian = group[group['taker_side'] != dominant]
                if len(contrarian) > 0:
                    contrarian_profits.append({
                        'profit': contrarian['actual_profit_dollars'].sum(),
                        'cost': contrarian['cost_dollars'].sum(),
                        'trades': len(contrarian),
                        'wins': len(contrarian[contrarian['is_winner'] == True])
                    })

        if contrarian_profits:
            total_profit = sum(x['profit'] for x in contrarian_profits)
            total_cost = sum(x['cost'] for x in contrarian_profits)
            total_trades = sum(x['trades'] for x in contrarian_profits)
            total_wins = sum(x['wins'] for x in contrarian_profits)

            roi = total_profit / total_cost if total_cost > 0 else 0
            win_rate = total_wins / total_trades if total_trades > 0 else 0

            results['game_contrarian'] = {
                'trades': total_trades,
                'win_rate': round(win_rate, 4),
                'total_profit': round(total_profit, 2),
                'roi': round(roi, 4),
                'description': 'Bet against dominant flow (>70%) in game markets'
            }
            print(f"  Game contrarian (vs >70% flow): {total_trades} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${total_profit:,.0f}")

    return results


def analyze_first_last_trades(resolved):
    """
    Pattern 9: First/Last Trade in Session

    Hypothesis: The first or last trade in a market session may have
    different characteristics (e.g., informed traders move first/last).
    """
    print("\n=== PATTERN 9: First/Last Trades in Market Sessions ===")

    results = {}

    # Sort by market and timestamp
    sorted_trades = resolved.sort_values(['market_ticker', 'timestamp'])

    # Get first and last trades per market
    first_trades = sorted_trades.groupby('market_ticker').first().reset_index()
    last_trades = sorted_trades.groupby('market_ticker').last().reset_index()

    for name, trades in [('first_trade', first_trades), ('last_trade', last_trades)]:
        if len(trades) < 100:
            continue

        wins = len(trades[trades['is_winner'] == True])
        win_rate = wins / len(trades)
        profit = trades['actual_profit_dollars'].sum()
        roi = profit / trades['cost_dollars'].sum() if trades['cost_dollars'].sum() > 0 else 0

        results[name] = {
            'trades': len(trades),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4)
        }
        print(f"  {name}: {len(trades)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    # Analyze whale first trades
    print("\n  Whale first trades (>=100 contracts):")
    whale_first = first_trades[first_trades['count'] >= 100]
    if len(whale_first) > 50:
        wins = len(whale_first[whale_first['is_winner'] == True])
        win_rate = wins / len(whale_first)
        profit = whale_first['actual_profit_dollars'].sum()
        roi = profit / whale_first['cost_dollars'].sum() if whale_first['cost_dollars'].sum() > 0 else 0

        results['whale_first_trade'] = {
            'trades': len(whale_first),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4)
        }
        print(f"    Whale first trade: {len(whale_first)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    return results


def analyze_size_price_paradox(resolved):
    """
    Pattern 10: Size-Price Paradox

    Hypothesis: Large trades at "wrong" prices (e.g., big YES at high prices,
    big NO at low prices) may be informed traders.
    """
    print("\n=== PATTERN 10: Size-Price Paradox (Informed Trading?) ===")

    results = {}

    # Large YES at high prices (seems wrong but might be informed)
    large_yes_high = resolved[
        (resolved['count'] >= 100) &
        (resolved['taker_side'] == 'yes') &
        (resolved['trade_price'] >= 80)
    ]

    # Large NO at low prices (seems wrong but might be informed)
    large_no_low = resolved[
        (resolved['count'] >= 100) &
        (resolved['taker_side'] == 'no') &
        (resolved['trade_price'] <= 20)
    ]

    for name, subset in [('large_yes_at_80plus', large_yes_high), ('large_no_at_20minus', large_no_low)]:
        if len(subset) < 50:
            continue

        wins = len(subset[subset['is_winner'] == True])
        win_rate = wins / len(subset)
        profit = subset['actual_profit_dollars'].sum()
        roi = profit / subset['cost_dollars'].sum() if subset['cost_dollars'].sum() > 0 else 0

        results[name] = {
            'trades': len(subset),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'Large contrarian trades at extreme prices'
        }
        print(f"  {name}: {len(subset)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    # Even more extreme: whale YES at 90+, whale NO at 10-
    print("\n  EXTREME paradox (whale at extreme prices):")

    extreme_yes = resolved[
        (resolved['count'] >= 200) &
        (resolved['taker_side'] == 'yes') &
        (resolved['trade_price'] >= 90)
    ]

    extreme_no = resolved[
        (resolved['count'] >= 200) &
        (resolved['taker_side'] == 'no') &
        (resolved['trade_price'] <= 10)
    ]

    for name, subset in [('extreme_yes_90plus', extreme_yes), ('extreme_no_10minus', extreme_no)]:
        if len(subset) < 20:
            continue

        wins = len(subset[subset['is_winner'] == True])
        win_rate = wins / len(subset)
        profit = subset['actual_profit_dollars'].sum()
        roi = profit / subset['cost_dollars'].sum() if subset['cost_dollars'].sum() > 0 else 0

        results[name] = {
            'trades': len(subset),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4)
        }
        print(f"    {name}: {len(subset)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    return results


def analyze_volume_velocity(resolved):
    """
    Pattern 11: Volume Velocity

    Hypothesis: Markets that suddenly see high volume velocity
    (many trades in short time) may be moving on information.
    """
    print("\n=== PATTERN 11: Volume Velocity Spikes ===")

    results = {}

    # Sort by market and timestamp
    sorted_trades = resolved.sort_values(['market_ticker', 'timestamp'])

    # Calculate rolling 1-minute volume per market
    sorted_trades['minute_bucket'] = (sorted_trades['timestamp'] // 60000) * 60000

    minute_volume = sorted_trades.groupby(['market_ticker', 'minute_bucket']).agg({
        'count': 'sum',
        'cost_dollars': 'sum'
    }).reset_index()

    # Find high-velocity minutes (top 10%)
    velocity_threshold = minute_volume['count'].quantile(0.90)
    high_velocity_minutes = minute_volume[minute_volume['count'] >= velocity_threshold]

    # Get trades during high-velocity periods
    high_velocity_trades = sorted_trades.merge(
        high_velocity_minutes[['market_ticker', 'minute_bucket']],
        on=['market_ticker', 'minute_bucket'],
        how='inner'
    )

    if len(high_velocity_trades) > 100:
        wins = len(high_velocity_trades[high_velocity_trades['is_winner'] == True])
        win_rate = wins / len(high_velocity_trades)
        profit = high_velocity_trades['actual_profit_dollars'].sum()
        roi = profit / high_velocity_trades['cost_dollars'].sum() if high_velocity_trades['cost_dollars'].sum() > 0 else 0

        results['high_velocity_periods'] = {
            'trades': len(high_velocity_trades),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'Trades during top 10% volume velocity minutes'
        }
        print(f"  High velocity periods: {len(high_velocity_trades)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    # Contrast: low velocity trades
    low_velocity_minutes = minute_volume[minute_volume['count'] <= minute_volume['count'].quantile(0.25)]
    low_velocity_trades = sorted_trades.merge(
        low_velocity_minutes[['market_ticker', 'minute_bucket']],
        on=['market_ticker', 'minute_bucket'],
        how='inner'
    )

    if len(low_velocity_trades) > 100:
        wins = len(low_velocity_trades[low_velocity_trades['is_winner'] == True])
        win_rate = wins / len(low_velocity_trades)
        profit = low_velocity_trades['actual_profit_dollars'].sum()
        roi = profit / low_velocity_trades['cost_dollars'].sum() if low_velocity_trades['cost_dollars'].sum() > 0 else 0

        results['low_velocity_periods'] = {
            'trades': len(low_velocity_trades),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'Trades during bottom 25% volume velocity minutes'
        }
        print(f"  Low velocity periods: {len(low_velocity_trades)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    return results


def find_combination_strategies(resolved):
    """
    Pattern 12: Multi-Factor Combination Strategies

    Combine multiple factors that individually show edge to find super-strategies.
    """
    print("\n=== PATTERN 12: Multi-Factor Combination Strategies ===")

    results = {}

    # Factor 1: Hour 22 (known to be profitable)
    # Factor 2: NO trades
    # Factor 3: 30-50c price range
    # Factor 4: >= 100 contracts

    # Strategy A: Night NO trades at mid-prices
    strategy_a = resolved[
        (resolved['hour'] >= 21) &
        (resolved['taker_side'] == 'no') &
        (resolved['trade_price'] >= 30) &
        (resolved['trade_price'] <= 50) &
        (resolved['count'] >= 50)
    ]

    if len(strategy_a) > 50:
        wins = len(strategy_a[strategy_a['is_winner'] == True])
        win_rate = wins / len(strategy_a)
        profit = strategy_a['actual_profit_dollars'].sum()
        roi = profit / strategy_a['cost_dollars'].sum() if strategy_a['cost_dollars'].sum() > 0 else 0

        results['night_no_mid_whale'] = {
            'trades': len(strategy_a),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'Night (21-00h) + NO + 30-50c + >=50 contracts'
        }
        print(f"  Night NO mid-price whale: {len(strategy_a)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    # Strategy B: EPL + YES + 60-80c (known good category)
    resolved_with_cat = resolved.copy()
    resolved_with_cat['category'] = resolved_with_cat['market_ticker'].str.extract(r'(KX[A-Z]+)')[0]

    strategy_b = resolved_with_cat[
        (resolved_with_cat['category'].isin(['KXEPLG', 'KXEPLS', 'KXEPLT'])) &
        (resolved_with_cat['taker_side'] == 'yes') &
        (resolved_with_cat['trade_price'] >= 60) &
        (resolved_with_cat['trade_price'] <= 80)
    ]

    if len(strategy_b) > 50:
        wins = len(strategy_b[strategy_b['is_winner'] == True])
        win_rate = wins / len(strategy_b)
        profit = strategy_b['actual_profit_dollars'].sum()
        roi = profit / strategy_b['cost_dollars'].sum() if strategy_b['cost_dollars'].sum() > 0 else 0

        results['epl_yes_favorite'] = {
            'trades': len(strategy_b),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'EPL markets + YES + 60-80c (riding favorites)'
        }
        print(f"  EPL YES favorite: {len(strategy_b)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    # Strategy C: Minute 30-35 + High leverage + Whale
    strategy_c = resolved[
        (resolved['minute'] >= 30) &
        (resolved['minute'] <= 35) &
        (resolved['leverage_ratio'] >= 3) &
        (resolved['count'] >= 100)
    ]

    if len(strategy_c) > 20:
        wins = len(strategy_c[strategy_c['is_winner'] == True])
        win_rate = wins / len(strategy_c)
        profit = strategy_c['actual_profit_dollars'].sum()
        roi = profit / strategy_c['cost_dollars'].sum() if strategy_c['cost_dollars'].sum() > 0 else 0

        results['mid_hour_leverage_whale'] = {
            'trades': len(strategy_c),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'Minute 30-35 + 3x+ leverage + >=100 contracts'
        }
        print(f"  Mid-hour leverage whale: {len(strategy_c)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    # Strategy D: Large first trades in games
    sorted_trades = resolved.sort_values(['market_ticker', 'timestamp'])
    first_trades = sorted_trades.groupby('market_ticker').first().reset_index()

    game_first = first_trades[
        (first_trades['market_ticker'].str.contains('GAME|FIGHT')) &
        (first_trades['count'] >= 200)
    ]

    if len(game_first) > 20:
        wins = len(game_first[game_first['is_winner'] == True])
        win_rate = wins / len(game_first)
        profit = game_first['actual_profit_dollars'].sum()
        roi = profit / game_first['cost_dollars'].sum() if game_first['cost_dollars'].sum() > 0 else 0

        results['whale_opener_games'] = {
            'trades': len(game_first),
            'win_rate': round(win_rate, 4),
            'total_profit': round(profit, 2),
            'roi': round(roi, 4),
            'description': 'First trade in game markets >=200 contracts (whale opener)'
        }
        print(f"  Whale opener in games: {len(game_first)} trades, win_rate={win_rate:.2%}, ROI={roi:.2%}, profit=${profit:,.0f}")

    return results


def generate_summary(all_results):
    """Generate a summary of top findings."""
    print("\n" + "="*80)
    print("SUMMARY: TOP NEW STRATEGIES DISCOVERED")
    print("="*80)

    # Flatten all results
    all_patterns = []
    for category, patterns in all_results.items():
        for name, data in patterns.items():
            if isinstance(data, dict) and 'roi' in data:
                all_patterns.append({
                    'category': category,
                    'name': name,
                    **data
                })

    # Filter to significant patterns (min 50 trades, positive ROI)
    significant = [p for p in all_patterns if p.get('trades', 0) >= 50 and p.get('roi', 0) > 0.05]
    significant.sort(key=lambda x: x.get('roi', 0), reverse=True)

    print("\nTop 10 NEW Strategies by ROI (min 50 trades, ROI > 5%):\n")

    for i, pattern in enumerate(significant[:10], 1):
        print(f"{i}. {pattern['name']}")
        print(f"   Category: {pattern['category']}")
        print(f"   Trades: {pattern['trades']}")
        print(f"   Win Rate: {pattern.get('win_rate', 0):.2%}")
        print(f"   ROI: {pattern['roi']:.2%}")
        print(f"   Total Profit: ${pattern.get('total_profit', 0):,.0f}")
        if 'description' in pattern:
            print(f"   Description: {pattern['description']}")
        print()

    return significant[:10]


def main():
    """Main analysis function."""
    print("="*80)
    print("ADDITIONAL PATTERN ANALYSIS FOR KALSHI TRADES")
    print("="*80)

    trades, resolved, outcomes = load_data()

    all_results = {}

    # Run all analyses
    all_results['psychological_levels'] = analyze_psychological_price_levels(resolved)
    all_results['consecutive_trades'] = analyze_consecutive_trades(resolved)
    all_results['minute_precision'] = analyze_minute_precision(resolved)
    all_results['price_sequences'] = analyze_price_movement_sequences(resolved)
    all_results['category_combinations'] = analyze_market_category_combinations(resolved)
    all_results['size_momentum'] = analyze_size_momentum(resolved)
    all_results['leverage_patterns'] = analyze_leverage_patterns(resolved)
    all_results['inter_market'] = analyze_inter_market_correlations(resolved)
    all_results['first_last_trades'] = analyze_first_last_trades(resolved)
    all_results['size_price_paradox'] = analyze_size_price_paradox(resolved)
    all_results['volume_velocity'] = analyze_volume_velocity(resolved)
    all_results['combination_strategies'] = find_combination_strategies(resolved)

    # Generate summary
    top_strategies = generate_summary(all_results)

    # Add top strategies to results
    all_results['top_new_strategies'] = top_strategies
    all_results['generated'] = datetime.now().isoformat()
    all_results['total_trades_analyzed'] = len(resolved)

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    return all_results


if __name__ == "__main__":
    main()
