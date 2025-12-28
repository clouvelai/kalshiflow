#!/usr/bin/env python3
"""
NBA Deep Dive Analysis

Comprehensive analysis of NBA game markets to determine if NFL underdog patterns transfer.
Primary question: Does "Bet NO on underdogs priced at 20-40c" work for NBA like it does for NFL?

NFL Reference:
- KXNFLGAME + NO + underdog (20-40c) = 147.90% ROI, 66.18% win rate, $407,643 profit
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json


def load_data():
    """Load enriched trades data."""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "training" / "reports" / "enriched_trades_final.csv",
        Path(__file__).parent.parent.parent.parent.parent / "training" / "reports" / "enriched_trades_final.csv",
        Path("/Users/samuelclark/Desktop/kalshiflow/backend/training/reports/enriched_trades_final.csv"),
    ]

    for data_path in possible_paths:
        if data_path.exists():
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
            print(f"Total trades loaded: {len(df):,}")
            return df

    raise FileNotFoundError(f"Could not find enriched_trades_final.csv in any of: {possible_paths}")


def identify_market_types(df):
    """Identify all NBA-related market types."""
    # Get unique tickers
    nba_tickers = df[df['market_ticker'].str.contains('NBA', case=False, na=False)]['market_ticker'].unique()

    # Extract market type patterns
    patterns = defaultdict(int)
    for ticker in nba_tickers:
        # Extract the market type (e.g., KXNBAGAME, KXNBASPREAD, etc.)
        parts = ticker.split('-')
        if parts:
            base = parts[0]
            patterns[base] += 1

    print("\n=== NBA Market Type Distribution ===")
    for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count} unique tickers")

    return patterns


def filter_resolved_trades(df):
    """Filter to only resolved trades with outcomes."""
    resolved = df[df['is_winner'].notna()].copy()
    print(f"Resolved trades: {len(resolved):,} ({len(resolved)/len(df)*100:.1f}%)")
    return resolved


def analyze_nba_game_markets(df):
    """Analyze NBA game markets specifically."""
    # Filter to NBA game markets
    nba_game = df[df['market_ticker'].str.contains('KXNBAGAME', case=False, na=False)].copy()
    print(f"\n=== NBA GAME Markets Analysis ===")
    print(f"Total NBA game trades: {len(nba_game):,}")

    if len(nba_game) == 0:
        print("No NBA game trades found!")
        return {}

    # Basic stats
    resolved_nba = nba_game[nba_game['is_winner'].notna()].copy()
    print(f"Resolved NBA game trades: {len(resolved_nba):,}")

    if len(resolved_nba) == 0:
        print("No resolved NBA game trades!")
        return {}

    # Overall NBA game performance
    total_cost = resolved_nba['cost_dollars'].sum()
    total_profit = resolved_nba['actual_profit_dollars'].sum()
    win_rate = (resolved_nba['is_winner'] == True).mean() * 100
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

    print(f"\nOverall NBA Game Performance:")
    print(f"  Trades: {len(resolved_nba):,}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Total Cost: ${total_cost:,.2f}")
    print(f"  Total Profit: ${total_profit:,.2f}")
    print(f"  ROI: {roi:.2f}%")

    return resolved_nba


def analyze_by_side(df, category_name="NBA"):
    """Analyze YES vs NO side performance."""
    print(f"\n=== {category_name} Side Analysis (YES vs NO) ===")

    results = {}
    for side in ['yes', 'no']:
        side_df = df[df['taker_side'] == side]
        if len(side_df) == 0:
            continue

        total_cost = side_df['cost_dollars'].sum()
        total_profit = side_df['actual_profit_dollars'].sum()
        win_rate = (side_df['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        results[side] = {
            'trades': len(side_df),
            'win_rate': win_rate,
            'cost': total_cost,
            'profit': total_profit,
            'roi': roi
        }

        print(f"\n{side.upper()} Side:")
        print(f"  Trades: {len(side_df):,}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Total Cost: ${total_cost:,.2f}")
        print(f"  Profit: ${total_profit:,.2f}")
        print(f"  ROI: {roi:.2f}%")

    return results


def analyze_by_price_ranges(df, category_name="NBA"):
    """Analyze performance by price ranges."""
    print(f"\n=== {category_name} Price Range Analysis ===")

    # Define price ranges
    ranges = [
        ('Extreme Longshot (1-10c)', 1, 10),
        ('Longshot (10-20c)', 10, 20),
        ('Underdog (20-40c)', 20, 40),
        ('Slight Underdog (40-50c)', 40, 50),
        ('Coin Flip (49-51c)', 49, 51),
        ('Slight Favorite (50-60c)', 50, 60),
        ('Favorite (60-80c)', 60, 80),
        ('Heavy Favorite (80-90c)', 80, 90),
        ('Lock (90-99c)', 90, 99),
    ]

    results = []
    for range_name, low, high in ranges:
        range_df = df[(df['trade_price'] >= low) & (df['trade_price'] <= high)]
        if len(range_df) == 0:
            continue

        total_cost = range_df['cost_dollars'].sum()
        total_profit = range_df['actual_profit_dollars'].sum()
        win_rate = (range_df['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        results.append({
            'range': range_name,
            'low': low,
            'high': high,
            'trades': len(range_df),
            'win_rate': win_rate,
            'cost': total_cost,
            'profit': total_profit,
            'roi': roi
        })

        print(f"\n{range_name}:")
        print(f"  Trades: {len(range_df):,}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Profit: ${total_profit:,.2f}")
        print(f"  ROI: {roi:.2f}%")

    return results


def analyze_underdog_no_strategy(df, category_name="NBA"):
    """Analyze the specific underdog NO strategy that works for NFL."""
    print(f"\n=== {category_name} UNDERDOG NO Strategy Analysis ===")
    print("(Testing if NFL pattern transfers: NO + 20-40c = profitable)")

    # Filter to underdog range (20-40c)
    underdog_df = df[(df['trade_price'] >= 20) & (df['trade_price'] <= 40)]

    results = {}
    for side in ['yes', 'no']:
        side_df = underdog_df[underdog_df['taker_side'] == side]
        if len(side_df) == 0:
            continue

        total_cost = side_df['cost_dollars'].sum()
        total_profit = side_df['actual_profit_dollars'].sum()
        win_rate = (side_df['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        results[side] = {
            'trades': len(side_df),
            'win_rate': win_rate,
            'cost': total_cost,
            'profit': total_profit,
            'roi': roi
        }

        marker = "*** NFL PATTERN ***" if side == 'no' else ""
        print(f"\n{side.upper()} at 20-40c (Underdog Range): {marker}")
        print(f"  Trades: {len(side_df):,}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Total Cost: ${total_cost:,.2f}")
        print(f"  Profit: ${total_profit:,.2f}")
        print(f"  ROI: {roi:.2f}%")

    return results


def analyze_side_by_price_matrix(df, category_name="NBA"):
    """Create a full matrix of side x price range performance."""
    print(f"\n=== {category_name} Side x Price Matrix ===")

    price_ranges = [
        ('1-20c', 1, 20),
        ('20-40c', 20, 40),
        ('40-60c', 40, 60),
        ('60-80c', 60, 80),
        ('80-99c', 80, 99),
    ]

    matrix = []
    for range_name, low, high in price_ranges:
        range_df = df[(df['trade_price'] >= low) & (df['trade_price'] <= high)]

        for side in ['yes', 'no']:
            side_df = range_df[range_df['taker_side'] == side]
            if len(side_df) == 0:
                continue

            total_cost = side_df['cost_dollars'].sum()
            total_profit = side_df['actual_profit_dollars'].sum()
            win_rate = (side_df['is_winner'] == True).mean() * 100
            roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

            matrix.append({
                'range': range_name,
                'side': side,
                'trades': len(side_df),
                'win_rate': win_rate,
                'profit': total_profit,
                'roi': roi
            })

    # Print as table
    print("\n{:<10} {:<6} {:>8} {:>10} {:>12} {:>10}".format(
        "Price", "Side", "Trades", "Win Rate", "Profit", "ROI"))
    print("-" * 60)
    for row in sorted(matrix, key=lambda x: (-x['roi'], x['range'])):
        print("{:<10} {:<6} {:>8,} {:>9.1f}% {:>11,.0f} {:>9.1f}%".format(
            row['range'], row['side'].upper(), row['trades'],
            row['win_rate'], row['profit'], row['roi']))

    return matrix


def analyze_whale_vs_retail(df, category_name="NBA", whale_threshold=500):
    """Analyze whale (>=500 contracts) vs retail performance."""
    print(f"\n=== {category_name} Whale vs Retail Analysis ===")
    print(f"(Whale threshold: >= {whale_threshold} contracts)")

    whale_df = df[df['count'] >= whale_threshold]
    retail_df = df[df['count'] < whale_threshold]

    results = {}
    for name, subset in [('Whale (>=500)', whale_df), ('Retail (<500)', retail_df)]:
        if len(subset) == 0:
            continue

        total_cost = subset['cost_dollars'].sum()
        total_profit = subset['actual_profit_dollars'].sum()
        win_rate = (subset['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        results[name] = {
            'trades': len(subset),
            'win_rate': win_rate,
            'cost': total_cost,
            'profit': total_profit,
            'roi': roi
        }

        print(f"\n{name}:")
        print(f"  Trades: {len(subset):,}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Total Cost: ${total_cost:,.2f}")
        print(f"  Profit: ${total_profit:,.2f}")
        print(f"  ROI: {roi:.2f}%")

    return results


def analyze_whale_by_side(df, category_name="NBA", whale_threshold=500):
    """Analyze whale trades by side."""
    print(f"\n=== {category_name} Whale Trades by Side ===")

    whale_df = df[df['count'] >= whale_threshold]

    results = {}
    for side in ['yes', 'no']:
        side_df = whale_df[whale_df['taker_side'] == side]
        if len(side_df) == 0:
            continue

        total_cost = side_df['cost_dollars'].sum()
        total_profit = side_df['actual_profit_dollars'].sum()
        win_rate = (side_df['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        results[side] = {
            'trades': len(side_df),
            'win_rate': win_rate,
            'cost': total_cost,
            'profit': total_profit,
            'roi': roi
        }

        print(f"\nWhale {side.upper()}:")
        print(f"  Trades: {len(side_df):,}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Profit: ${total_profit:,.2f}")
        print(f"  ROI: {roi:.2f}%")

    return results


def analyze_time_patterns(df, category_name="NBA"):
    """Analyze performance by time of day and day of week."""
    print(f"\n=== {category_name} Time Pattern Analysis ===")

    # Convert timestamp to datetime
    df = df.copy()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['minute'] = df['datetime'].dt.minute

    # Hour analysis
    print("\nBy Hour:")
    hour_results = []
    for hour in sorted(df['hour'].unique()):
        hour_df = df[df['hour'] == hour]
        if len(hour_df) < 10:
            continue

        total_cost = hour_df['cost_dollars'].sum()
        total_profit = hour_df['actual_profit_dollars'].sum()
        win_rate = (hour_df['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        hour_results.append({
            'hour': hour,
            'trades': len(hour_df),
            'win_rate': win_rate,
            'profit': total_profit,
            'roi': roi
        })

    # Sort by ROI and show top/bottom
    hour_results_sorted = sorted(hour_results, key=lambda x: -x['roi'])
    print("\nTop 5 Hours by ROI:")
    for r in hour_results_sorted[:5]:
        print(f"  Hour {r['hour']:02d}: {r['trades']:,} trades, {r['win_rate']:.1f}% WR, ${r['profit']:,.0f} profit, {r['roi']:.1f}% ROI")

    print("\nBottom 5 Hours by ROI:")
    for r in hour_results_sorted[-5:]:
        print(f"  Hour {r['hour']:02d}: {r['trades']:,} trades, {r['win_rate']:.1f}% WR, ${r['profit']:,.0f} profit, {r['roi']:.1f}% ROI")

    # Day of week
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print("\nBy Day of Week:")
    for dow in sorted(df['dayofweek'].unique()):
        dow_df = df[df['dayofweek'] == dow]
        if len(dow_df) < 10:
            continue

        total_cost = dow_df['cost_dollars'].sum()
        total_profit = dow_df['actual_profit_dollars'].sum()
        win_rate = (dow_df['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        print(f"  {days[dow]}: {len(dow_df):,} trades, {win_rate:.1f}% WR, ${total_profit:,.0f} profit, {roi:.1f}% ROI")

    return hour_results


def compare_nba_vs_nfl(nba_df, nfl_df):
    """Direct comparison of NBA vs NFL performance."""
    print("\n" + "=" * 70)
    print("=== NBA vs NFL DIRECT COMPARISON ===")
    print("=" * 70)

    comparison = []

    # Define patterns to compare
    patterns = [
        ('Overall', lambda x: x),
        ('YES side', lambda x: x[x['taker_side'] == 'yes']),
        ('NO side', lambda x: x[x['taker_side'] == 'no']),
        ('Underdog (20-40c)', lambda x: x[(x['trade_price'] >= 20) & (x['trade_price'] <= 40)]),
        ('NO @ Underdog (20-40c)', lambda x: x[(x['trade_price'] >= 20) & (x['trade_price'] <= 40) & (x['taker_side'] == 'no')]),
        ('YES @ Underdog (20-40c)', lambda x: x[(x['trade_price'] >= 20) & (x['trade_price'] <= 40) & (x['taker_side'] == 'yes')]),
        ('Whale (>=500)', lambda x: x[x['count'] >= 500]),
        ('Favorite (60-80c)', lambda x: x[(x['trade_price'] >= 60) & (x['trade_price'] <= 80)]),
        ('NO @ Favorite (60-80c)', lambda x: x[(x['trade_price'] >= 60) & (x['trade_price'] <= 80) & (x['taker_side'] == 'no')]),
    ]

    print("\n{:<25} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}".format(
        "Pattern", "NBA", "NFL", "NBA WR", "NFL WR", "NBA ROI", "NFL ROI"))
    print("-" * 95)

    for name, filter_fn in patterns:
        nba_subset = filter_fn(nba_df)
        nfl_subset = filter_fn(nfl_df)

        nba_trades = len(nba_subset)
        nfl_trades = len(nfl_subset)

        if nba_trades > 0:
            nba_wr = (nba_subset['is_winner'] == True).mean() * 100
            nba_cost = nba_subset['cost_dollars'].sum()
            nba_profit = nba_subset['actual_profit_dollars'].sum()
            nba_roi = (nba_profit / nba_cost * 100) if nba_cost > 0 else 0
        else:
            nba_wr = nba_roi = 0

        if nfl_trades > 0:
            nfl_wr = (nfl_subset['is_winner'] == True).mean() * 100
            nfl_cost = nfl_subset['cost_dollars'].sum()
            nfl_profit = nfl_subset['actual_profit_dollars'].sum()
            nfl_roi = (nfl_profit / nfl_cost * 100) if nfl_cost > 0 else 0
        else:
            nfl_wr = nfl_roi = 0

        comparison.append({
            'pattern': name,
            'nba_trades': nba_trades,
            'nfl_trades': nfl_trades,
            'nba_wr': nba_wr,
            'nfl_wr': nfl_wr,
            'nba_roi': nba_roi,
            'nfl_roi': nfl_roi
        })

        print("{:<25} {:>8,} {:>8,} {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>9.1f}%".format(
            name, nba_trades, nfl_trades, nba_wr, nfl_wr, nba_roi, nfl_roi))

    return comparison


def find_nba_specific_edges(df):
    """Search for NBA-specific profitable patterns."""
    print("\n" + "=" * 70)
    print("=== SEARCHING FOR NBA-SPECIFIC EDGES ===")
    print("=" * 70)

    edges = []

    # Test all combinations of side x price range
    price_ranges = [
        ('1-15c', 1, 15),
        ('15-25c', 15, 25),
        ('25-35c', 25, 35),
        ('35-45c', 35, 45),
        ('45-55c', 45, 55),
        ('55-65c', 55, 65),
        ('65-75c', 65, 75),
        ('75-85c', 75, 85),
        ('85-99c', 85, 99),
    ]

    for range_name, low, high in price_ranges:
        range_df = df[(df['trade_price'] >= low) & (df['trade_price'] <= high)]

        for side in ['yes', 'no']:
            side_df = range_df[range_df['taker_side'] == side]
            if len(side_df) < 20:  # Minimum sample size
                continue

            total_cost = side_df['cost_dollars'].sum()
            total_profit = side_df['actual_profit_dollars'].sum()
            win_rate = (side_df['is_winner'] == True).mean() * 100
            roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

            if roi > 20:  # Significant positive edge
                edges.append({
                    'pattern': f"{side.upper()} @ {range_name}",
                    'trades': len(side_df),
                    'win_rate': win_rate,
                    'profit': total_profit,
                    'roi': roi
                })

    # Sort by ROI
    edges_sorted = sorted(edges, key=lambda x: -x['roi'])

    print("\nTop Profitable Patterns (ROI > 20%):")
    print("{:<25} {:>8} {:>10} {:>12} {:>10}".format(
        "Pattern", "Trades", "Win Rate", "Profit", "ROI"))
    print("-" * 70)
    for edge in edges_sorted[:15]:
        print("{:<25} {:>8,} {:>9.1f}% {:>11,.0f} {:>9.1f}%".format(
            edge['pattern'], edge['trades'], edge['win_rate'],
            edge['profit'], edge['roi']))

    return edges_sorted


def analyze_team_patterns(df, category_name="NBA"):
    """Extract team-specific patterns from tickers."""
    print(f"\n=== {category_name} Team Analysis ===")

    # Extract teams from tickers (format: KXNBAGAME-25DEC28SACLAL-SAC)
    df = df.copy()

    def extract_team(ticker):
        """Extract the specific team being bet on from ticker."""
        parts = ticker.split('-')
        if len(parts) >= 3:
            return parts[-1]  # Last part is usually the team
        return None

    df['team'] = df['market_ticker'].apply(extract_team)

    # Team performance
    team_results = []
    for team in df['team'].dropna().unique():
        team_df = df[df['team'] == team]
        if len(team_df) < 10:
            continue

        total_cost = team_df['cost_dollars'].sum()
        total_profit = team_df['actual_profit_dollars'].sum()
        win_rate = (team_df['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        team_results.append({
            'team': team,
            'trades': len(team_df),
            'win_rate': win_rate,
            'profit': total_profit,
            'roi': roi
        })

    # Sort by profit
    team_results_sorted = sorted(team_results, key=lambda x: -x['profit'])

    print("\nTop 10 Teams by Profit:")
    for r in team_results_sorted[:10]:
        print(f"  {r['team']}: {r['trades']:,} trades, {r['win_rate']:.1f}% WR, ${r['profit']:,.0f} profit, {r['roi']:.1f}% ROI")

    print("\nBottom 10 Teams by Profit:")
    for r in team_results_sorted[-10:]:
        print(f"  {r['team']}: {r['trades']:,} trades, {r['win_rate']:.1f}% WR, ${r['profit']:,.0f} profit, {r['roi']:.1f}% ROI")

    return team_results_sorted


def analyze_leverage_patterns(df, category_name="NBA"):
    """Analyze performance by leverage ratio."""
    print(f"\n=== {category_name} Leverage Analysis ===")

    # Define leverage buckets
    leverage_ranges = [
        ('Ultra Low (0-1x)', 0, 1),
        ('Low (1-2x)', 1, 2),
        ('Medium (2-4x)', 2, 4),
        ('High (4-10x)', 4, 10),
        ('Extreme (10x+)', 10, 100),
    ]

    results = []
    for range_name, low, high in leverage_ranges:
        range_df = df[(df['leverage_ratio'] >= low) & (df['leverage_ratio'] < high)]
        if len(range_df) < 10:
            continue

        total_cost = range_df['cost_dollars'].sum()
        total_profit = range_df['actual_profit_dollars'].sum()
        win_rate = (range_df['is_winner'] == True).mean() * 100
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        results.append({
            'range': range_name,
            'trades': len(range_df),
            'win_rate': win_rate,
            'profit': total_profit,
            'roi': roi
        })

        print(f"\n{range_name}:")
        print(f"  Trades: {len(range_df):,}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Profit: ${total_profit:,.2f}")
        print(f"  ROI: {roi:.2f}%")

    return results


def main():
    """Run comprehensive NBA analysis."""
    print("=" * 70)
    print("NBA DEEP DIVE ANALYSIS")
    print("Testing: Does NFL underdog NO strategy transfer to NBA?")
    print("=" * 70)

    # Load data
    df = load_data()

    # Filter to resolved trades only
    resolved_df = filter_resolved_trades(df)

    # Identify all NBA market types
    identify_market_types(df)

    # Get NBA GAME markets specifically
    nba_game_df = resolved_df[resolved_df['market_ticker'].str.contains('KXNBAGAME', case=False, na=False)].copy()

    # Get NFL GAME markets for comparison
    nfl_game_df = resolved_df[resolved_df['market_ticker'].str.contains('KXNFLGAME', case=False, na=False)].copy()

    print(f"\nNBA Game resolved trades: {len(nba_game_df):,}")
    print(f"NFL Game resolved trades: {len(nfl_game_df):,}")

    if len(nba_game_df) == 0:
        print("\nNo resolved NBA game trades found! Checking for any NBA trades...")
        nba_all = df[df['market_ticker'].str.contains('NBA', case=False, na=False)]
        print(f"Total NBA trades (including unresolved): {len(nba_all):,}")

        # Show sample of NBA tickers
        print("\nSample NBA tickers:")
        for ticker in nba_all['market_ticker'].unique()[:20]:
            print(f"  {ticker}")
        return

    # Run all analyses
    print("\n" + "=" * 70)
    print("SECTION 1: NBA GAME BASELINE ANALYSIS")
    print("=" * 70)

    analyze_by_side(nba_game_df, "NBA GAME")
    analyze_by_price_ranges(nba_game_df, "NBA GAME")
    analyze_side_by_price_matrix(nba_game_df, "NBA GAME")

    print("\n" + "=" * 70)
    print("SECTION 2: THE CRITICAL TEST - UNDERDOG NO STRATEGY")
    print("=" * 70)

    nba_underdog_results = analyze_underdog_no_strategy(nba_game_df, "NBA GAME")
    nfl_underdog_results = analyze_underdog_no_strategy(nfl_game_df, "NFL GAME")

    print("\n" + "=" * 70)
    print("SECTION 3: WHALE VS RETAIL")
    print("=" * 70)

    analyze_whale_vs_retail(nba_game_df, "NBA GAME")
    analyze_whale_by_side(nba_game_df, "NBA GAME")

    print("\n" + "=" * 70)
    print("SECTION 4: TIME PATTERNS")
    print("=" * 70)

    analyze_time_patterns(nba_game_df, "NBA GAME")

    print("\n" + "=" * 70)
    print("SECTION 5: LEVERAGE ANALYSIS")
    print("=" * 70)

    analyze_leverage_patterns(nba_game_df, "NBA GAME")

    print("\n" + "=" * 70)
    print("SECTION 6: NBA vs NFL HEAD-TO-HEAD")
    print("=" * 70)

    compare_nba_vs_nfl(nba_game_df, nfl_game_df)

    print("\n" + "=" * 70)
    print("SECTION 7: NBA-SPECIFIC EDGE DISCOVERY")
    print("=" * 70)

    find_nba_specific_edges(nba_game_df)

    print("\n" + "=" * 70)
    print("SECTION 8: TEAM-LEVEL ANALYSIS")
    print("=" * 70)

    analyze_team_patterns(nba_game_df, "NBA GAME")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. NFL UNDERDOG NO STRATEGY REFERENCE:")
    if 'no' in nfl_underdog_results:
        nfl_r = nfl_underdog_results['no']
        print(f"   KXNFLGAME + NO + 20-40c: {nfl_r['trades']:,} trades, {nfl_r['win_rate']:.1f}% WR, {nfl_r['roi']:.1f}% ROI, ${nfl_r['profit']:,.0f} profit")

    print("\n2. NBA UNDERDOG NO STRATEGY RESULT:")
    if 'no' in nba_underdog_results:
        nba_r = nba_underdog_results['no']
        print(f"   KXNBAGAME + NO + 20-40c: {nba_r['trades']:,} trades, {nba_r['win_rate']:.1f}% WR, {nba_r['roi']:.1f}% ROI, ${nba_r['profit']:,.0f} profit")

        if nba_r['roi'] > 20:
            print("\n   *** PATTERN TRANSFERS! NBA underdog NO is profitable ***")
        elif nba_r['roi'] > 0:
            print("\n   ** Pattern partially transfers - positive but weaker than NFL **")
        else:
            print("\n   Pattern does NOT transfer - NBA underdog NO is unprofitable")

    print("\n3. RECOMMENDATION:")
    # This will be filled based on the actual results
    print("   (See detailed analysis above)")


if __name__ == "__main__":
    main()
