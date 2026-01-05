#!/usr/bin/env python3
"""
Session 012 - Independence Check

Verify that the validated strategies are INDEPENDENT of each other.
If they all detect the same markets, they're not really different strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data" / "trades" / "enriched_trades_resolved_ALL.csv"
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def get_h087_signal_markets(df):
    """H087: Round Size Bot Detection - >60% NO, >= 5 round trades"""
    df = df.copy()
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    round_trades = df[df['is_round_size']].copy()

    round_market = round_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),
        'count': 'count'
    }).reset_index()
    round_market.columns = ['market_ticker', 'no_ratio', 'n_round_trades']

    signal = round_market[
        (round_market['no_ratio'] > 0.6) &
        (round_market['n_round_trades'] >= 5)
    ]
    return set(signal['market_ticker'])


def get_h088_signal_markets(df):
    """H088: Millisecond Burst Detection - >60% NO in bursts"""
    df = df.copy()
    df['second'] = df['timestamp'].dt.floor('S')

    burst_counts = df.groupby(['market_ticker', 'second']).size().reset_index(name='trades_per_second')
    bursts = burst_counts[burst_counts['trades_per_second'] >= 3]

    burst_df = df.merge(bursts[['market_ticker', 'second']], on=['market_ticker', 'second'])

    burst_market = burst_df.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean()
    }).reset_index()
    burst_market.columns = ['market_ticker', 'no_ratio']

    signal = burst_market[burst_market['no_ratio'] > 0.6]
    return set(signal['market_ticker'])


def get_h097_signal_markets(df):
    """H097: Bot Agreement Signal - >60% NO among bot-like trades, >= 5 trades"""
    df = df.copy()
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    df['second'] = df['timestamp'].dt.floor('S')
    second_counts = df.groupby(['market_ticker', 'second']).size().reset_index(name='trades_in_second')
    df = df.merge(second_counts, on=['market_ticker', 'second'], how='left')
    df['is_burst'] = df['trades_in_second'] >= 2
    df['is_bot_like'] = df['is_round_size'] | df['is_burst']

    bot_trades = df[df['is_bot_like']].copy()

    bot_market = bot_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),
        'count': 'count'
    }).reset_index()
    bot_market.columns = ['market_ticker', 'no_ratio', 'n_bot_trades']

    signal = bot_market[
        (bot_market['no_ratio'] > 0.6) &
        (bot_market['n_bot_trades'] >= 5)
    ]
    return set(signal['market_ticker'])


def get_h102_signal_markets(df):
    """H102: Leverage Stability - lev_std < 0.7, >50% NO, >= 5 trades"""
    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['mean', 'std'],
        'taker_side': lambda x: (x == 'no').mean(),
        'count': 'count'
    }).reset_index()
    market_stats.columns = ['market_ticker', 'lev_mean', 'lev_std', 'no_ratio', 'n_trades']
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)

    signal = market_stats[
        (market_stats['lev_std'] < 0.7) &
        (market_stats['no_ratio'] > 0.5) &
        (market_stats['n_trades'] >= 5)
    ]
    return set(signal['market_ticker'])


def main():
    print("="*70)
    print("SESSION 012 - STRATEGY INDEPENDENCE CHECK")
    print("="*70)

    df = load_and_prepare_data()
    total_markets = df['market_ticker'].nunique()
    print(f"\nTotal markets in dataset: {total_markets:,}")

    # Get signal markets for each strategy
    h087_markets = get_h087_signal_markets(df)
    h088_markets = get_h088_signal_markets(df)
    h097_markets = get_h097_signal_markets(df)
    h102_markets = get_h102_signal_markets(df)

    print(f"\n--- Signal Market Counts ---")
    print(f"H087 (Round Size Bot):     {len(h087_markets):,} markets")
    print(f"H088 (Millisecond Burst):  {len(h088_markets):,} markets")
    print(f"H097 (Bot Agreement):      {len(h097_markets):,} markets")
    print(f"H102 (Leverage Stability): {len(h102_markets):,} markets")

    # Calculate overlaps
    print(f"\n--- Pairwise Overlaps ---")

    pairs = [
        ('H087', 'H088', h087_markets, h088_markets),
        ('H087', 'H097', h087_markets, h097_markets),
        ('H087', 'H102', h087_markets, h102_markets),
        ('H088', 'H097', h088_markets, h097_markets),
        ('H088', 'H102', h088_markets, h102_markets),
        ('H097', 'H102', h097_markets, h102_markets),
    ]

    for name1, name2, set1, set2 in pairs:
        overlap = len(set1 & set2)
        overlap_pct1 = 100 * overlap / len(set1) if len(set1) > 0 else 0
        overlap_pct2 = 100 * overlap / len(set2) if len(set2) > 0 else 0
        print(f"{name1} & {name2}: {overlap} markets ({overlap_pct1:.1f}% of {name1}, {overlap_pct2:.1f}% of {name2})")

    # Check total overlap
    print(f"\n--- Combined Analysis ---")

    all_signal = h087_markets | h088_markets | h097_markets | h102_markets
    print(f"Union of all signals: {len(all_signal):,} markets")

    all_overlap = h087_markets & h088_markets & h097_markets & h102_markets
    print(f"Intersection of all signals: {len(all_overlap):,} markets")

    # Check if H097 is just a superset (since it's "bot-like" = round OR burst)
    print(f"\n--- H097 Superset Analysis ---")
    h087_in_h097 = len(h087_markets & h097_markets) / len(h087_markets) * 100 if len(h087_markets) > 0 else 0
    h088_in_h097 = len(h088_markets & h097_markets) / len(h088_markets) * 100 if len(h088_markets) > 0 else 0
    print(f"H087 markets also in H097: {h087_in_h097:.1f}%")
    print(f"H088 markets also in H097: {h088_in_h097:.1f}%")

    if h087_in_h097 > 80 or h088_in_h097 > 80:
        print("\nWARNING: H097 may be redundant with H087/H088")

    # Check independence scores
    print(f"\n--- Independence Analysis ---")
    print("\nFor strategies to be truly independent, they should have <50% overlap")

    independent_pairs = []
    for name1, name2, set1, set2 in pairs:
        overlap = len(set1 & set2)
        overlap_pct = 100 * overlap / min(len(set1), len(set2)) if min(len(set1), len(set2)) > 0 else 0
        independent = overlap_pct < 50
        status = "INDEPENDENT" if independent else "CORRELATED"
        print(f"{name1} vs {name2}: {overlap_pct:.1f}% overlap - {status}")
        if independent:
            independent_pairs.append((name1, name2))

    print(f"\n--- Recommendations ---")
    if len(independent_pairs) > 0:
        print(f"Independent strategy pairs found: {len(independent_pairs)}")
        for p in independent_pairs:
            print(f"  - {p[0]} and {p[1]}")
    else:
        print("WARNING: All strategies may be detecting similar markets")

    # Final unique market contribution
    print(f"\n--- Unique Market Contributions ---")
    h087_unique = h087_markets - (h088_markets | h097_markets | h102_markets)
    h088_unique = h088_markets - (h087_markets | h097_markets | h102_markets)
    h097_unique = h097_markets - (h087_markets | h088_markets | h102_markets)
    h102_unique = h102_markets - (h087_markets | h088_markets | h097_markets)

    print(f"H087 unique markets: {len(h087_unique):,}")
    print(f"H088 unique markets: {len(h088_unique):,}")
    print(f"H097 unique markets: {len(h097_unique):,}")
    print(f"H102 unique markets: {len(h102_unique):,}")


if __name__ == "__main__":
    main()
