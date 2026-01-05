#!/usr/bin/env python3
"""
Session 012 - CRITICAL VERIFICATION

Verify that our validated strategies are NOT the same bugs as Session 011c.

Session 011c Bugs:
1. S010 (H087): Used 100 - trade_price for NO price, which inverts when >60% trades are NO
2. S011 (H102): Didn't properly check price proxy - just selected expensive NO contracts

Our approach:
1. Use actual no_price from data (directly from the enriched dataset)
2. Comprehensive price proxy check at each price bucket

This script will verify the data is correct and the signals are real.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).parent.parent / "data" / "trades" / "enriched_trades_resolved_ALL.csv"


def verify_data_integrity():
    """Verify the data columns are what we think they are"""
    print("="*70)
    print("DATA INTEGRITY CHECK")
    print("="*70)

    df = pd.read_csv(DATA_PATH)

    print(f"\nColumns in dataset: {list(df.columns)}")

    # Check relationship between trade_price, yes_price, no_price
    print("\n--- Price Column Analysis ---")
    sample = df.sample(20)
    print("\nSample of price relationships:")
    print(sample[['market_ticker', 'taker_side', 'trade_price', 'yes_price', 'no_price']].to_string())

    # Key check: Is no_price = 100 - yes_price?
    df['calc_no_price'] = 100 - df['yes_price']
    df['no_price_matches'] = np.abs(df['no_price'] - df['calc_no_price']) < 0.1

    print(f"\nDoes no_price = 100 - yes_price?")
    print(f"  Match rate: {df['no_price_matches'].mean():.1%}")

    # Check: What is trade_price?
    print("\n--- Trade Price Analysis ---")

    # For YES trades, trade_price should be yes_price
    yes_trades = df[df['taker_side'] == 'yes']
    yes_match = np.abs(yes_trades['trade_price'] - yes_trades['yes_price']).mean()
    print(f"  For YES trades: avg |trade_price - yes_price| = {yes_match:.2f}")

    # For NO trades, trade_price should be no_price
    no_trades = df[df['taker_side'] == 'no']
    no_match = np.abs(no_trades['trade_price'] - no_trades['no_price']).mean()
    print(f"  For NO trades: avg |trade_price - no_price| = {no_match:.2f}")

    # CRITICAL: Is no_price always in the data correctly?
    print(f"\n--- NO Price Distribution ---")
    print(f"  Min: {df['no_price'].min():.1f}")
    print(f"  Median: {df['no_price'].median():.1f}")
    print(f"  Mean: {df['no_price'].mean():.1f}")
    print(f"  Max: {df['no_price'].max():.1f}")

    return df


def verify_h087_signal(df):
    """
    Verify H087: Round Size Bot Detection

    The Session 011c bug was: using 100 - trade_price for NO price
    Our approach: use actual no_price from data
    """
    print("\n" + "="*70)
    print("H087 VERIFICATION")
    print("="*70)

    ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]

    # Identify round-size trades
    df = df.copy()
    df['is_round_size'] = df['count'].isin(ROUND_SIZES)
    round_trades = df[df['is_round_size']].copy()

    print(f"\n1. Round-size trade stats:")
    print(f"   Total round trades: {len(round_trades):,}")

    # Aggregate by market
    round_market = round_trades.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),  # NO ratio
        'no_price': 'mean',  # ACTUAL NO price from data
        'trade_price': 'mean',  # For comparison
        'yes_price': 'mean',  # For comparison
        'count': 'count',
        'market_result': 'first'
    }).reset_index()
    round_market.columns = ['market_ticker', 'no_ratio', 'avg_no_price', 'avg_trade_price', 'avg_yes_price', 'n_round_trades', 'market_result']

    # Apply signal: >60% NO consensus, >= 5 round trades
    signal = round_market[
        (round_market['no_ratio'] > 0.6) &
        (round_market['n_round_trades'] >= 5)
    ].copy()

    print(f"\n2. Signal markets: {len(signal)}")

    # Key verification: What is the actual NO price distribution?
    print(f"\n3. Signal NO price distribution:")
    print(f"   Min: {signal['avg_no_price'].min():.1f}")
    print(f"   Median: {signal['avg_no_price'].median():.1f}")
    print(f"   Mean: {signal['avg_no_price'].mean():.1f}")
    print(f"   Max: {signal['avg_no_price'].max():.1f}")

    # Compare to what 100 - trade_price would give
    signal['bug_no_price'] = 100 - signal['avg_trade_price']
    print(f"\n4. What the buggy calculation would give (100 - trade_price):")
    print(f"   Min: {signal['bug_no_price'].min():.1f}")
    print(f"   Median: {signal['bug_no_price'].median():.1f}")
    print(f"   Mean: {signal['bug_no_price'].mean():.1f}")
    print(f"   Max: {signal['bug_no_price'].max():.1f}")

    # The bug in Session 011c: when >60% are NO trades, trade_price IS the NO price
    # So 100 - trade_price gives YES price (inverted!)

    # Calculate edge with CORRECT no_price
    signal['bet_wins'] = (signal['market_result'] == 'no').astype(int)
    win_rate = signal['bet_wins'].mean()
    avg_no_price = signal['avg_no_price'].mean()
    breakeven = avg_no_price / 100.0
    edge = (win_rate - breakeven) * 100

    print(f"\n5. CORRECT Edge Calculation:")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Avg NO Price: {avg_no_price:.1f}c")
    print(f"   Breakeven: {breakeven:.2%}")
    print(f"   Edge: {edge:+.2f}%")

    # What would bug give?
    bug_no_price = signal['bug_no_price'].mean()
    bug_breakeven = bug_no_price / 100.0
    bug_edge = (win_rate - bug_breakeven) * 100

    print(f"\n6. BUGGY Edge Calculation (Session 011c):")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Buggy NO Price: {bug_no_price:.1f}c")
    print(f"   Buggy Breakeven: {bug_breakeven:.2%}")
    print(f"   Buggy Edge: {bug_edge:+.2f}%")

    # Price proxy check
    print(f"\n7. Price Proxy Verification:")
    all_markets = df.groupby('market_ticker').agg({
        'no_price': 'mean',
        'market_result': 'first'
    }).reset_index()
    all_markets['bet_wins'] = (all_markets['market_result'] == 'no').astype(int)
    all_markets['price_bucket'] = (all_markets['no_price'] // 10) * 10
    signal['price_bucket'] = (signal['avg_no_price'] // 10) * 10

    print("\n   Price Bucket | Signal WR | Baseline WR | Improvement")
    print("   " + "-"*55)

    improvements = []
    for bucket in sorted(signal['price_bucket'].unique()):
        sig_b = signal[signal['price_bucket'] == bucket]
        all_b = all_markets[all_markets['price_bucket'] == bucket]

        if len(sig_b) >= 10 and len(all_b) >= 50:
            sig_wr = sig_b['bet_wins'].mean()
            all_wr = all_b['bet_wins'].mean()
            imp = (sig_wr - all_wr) * 100

            print(f"   {int(bucket):2d}-{int(bucket)+10}c      {sig_wr:.1%}       {all_wr:.1%}        {imp:+.1f}%")
            improvements.append({'bucket': bucket, 'n': len(sig_b), 'imp': imp})

    if improvements:
        total_n = sum(i['n'] for i in improvements)
        weighted_imp = sum(i['imp'] * i['n'] / total_n for i in improvements)
        print(f"\n   Weighted Average Improvement: {weighted_imp:+.2f}%")

    return signal


def verify_h102_signal(df):
    """
    Verify H102: Leverage Stability Bot Detection

    Session 011c bug: Didn't properly compare to baseline at same prices
    """
    print("\n" + "="*70)
    print("H102 VERIFICATION")
    print("="*70)

    # Aggregate by market
    market_stats = df.groupby('market_ticker').agg({
        'leverage_ratio': ['mean', 'std'],
        'taker_side': lambda x: (x == 'no').mean(),  # NO ratio
        'no_price': 'mean',
        'count': 'count',
        'market_result': 'first'
    }).reset_index()
    market_stats.columns = ['market_ticker', 'lev_mean', 'lev_std', 'no_ratio', 'avg_no_price', 'n_trades', 'market_result']
    market_stats['lev_std'] = market_stats['lev_std'].fillna(0)

    # Apply signal: lev_std < 0.7, >50% NO consensus, >= 5 trades
    signal = market_stats[
        (market_stats['lev_std'] < 0.7) &
        (market_stats['no_ratio'] > 0.5) &
        (market_stats['n_trades'] >= 5)
    ].copy()

    print(f"\n1. Signal markets: {len(signal)}")

    # NO price distribution
    print(f"\n2. Signal NO price distribution:")
    print(f"   Min: {signal['avg_no_price'].min():.1f}")
    print(f"   Median: {signal['avg_no_price'].median():.1f}")
    print(f"   Mean: {signal['avg_no_price'].mean():.1f}")
    print(f"   Max: {signal['avg_no_price'].max():.1f}")

    # Edge calculation
    signal['bet_wins'] = (signal['market_result'] == 'no').astype(int)
    win_rate = signal['bet_wins'].mean()
    avg_no_price = signal['avg_no_price'].mean()
    breakeven = avg_no_price / 100.0
    edge = (win_rate - breakeven) * 100

    print(f"\n3. Edge Calculation:")
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Avg NO Price: {avg_no_price:.1f}c")
    print(f"   Breakeven: {breakeven:.2%}")
    print(f"   Edge: {edge:+.2f}%")

    # The critical test: Compare to ALL markets at same price level
    print(f"\n4. CRITICAL Price Proxy Test:")

    all_markets = df.groupby('market_ticker').agg({
        'no_price': 'mean',
        'market_result': 'first'
    }).reset_index()
    all_markets['bet_wins'] = (all_markets['market_result'] == 'no').astype(int)
    all_markets['price_bucket'] = (all_markets['no_price'] // 10) * 10
    signal['price_bucket'] = (signal['avg_no_price'] // 10) * 10

    print("\n   Price Bucket | Signal WR | Baseline WR | Improvement | N_sig")
    print("   " + "-"*65)

    improvements = []
    for bucket in sorted(signal['price_bucket'].unique()):
        sig_b = signal[signal['price_bucket'] == bucket]
        all_b = all_markets[all_markets['price_bucket'] == bucket]

        if len(sig_b) >= 10 and len(all_b) >= 50:
            sig_wr = sig_b['bet_wins'].mean()
            all_wr = all_b['bet_wins'].mean()
            imp = (sig_wr - all_wr) * 100

            print(f"   {int(bucket):2d}-{int(bucket)+10}c      {sig_wr:.1%}       {all_wr:.1%}        {imp:+.1f}%       {len(sig_b)}")
            improvements.append({'bucket': bucket, 'n': len(sig_b), 'imp': imp})

    if improvements:
        total_n = sum(i['n'] for i in improvements)
        weighted_imp = sum(i['imp'] * i['n'] / total_n for i in improvements)
        print(f"\n   Weighted Average Improvement: {weighted_imp:+.2f}%")

        # Is this just selecting high NO prices?
        avg_bucket = sum(i['bucket'] * i['n'] / total_n for i in improvements)
        print(f"   Average Price Bucket: {avg_bucket:.1f}c")

        if weighted_imp > 0:
            print(f"\n   VERDICT: Signal shows {weighted_imp:+.2f}% improvement ABOVE baseline at same prices")
            print(f"   This is NOT a pure price proxy!")
        else:
            print(f"\n   VERDICT: Signal shows {weighted_imp:+.2f}% - this IS a price proxy")

    return signal


def verify_h088_signal(df):
    """
    Verify H088: Millisecond Burst Detection
    """
    print("\n" + "="*70)
    print("H088 VERIFICATION")
    print("="*70)

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['second'] = df['timestamp'].dt.floor('S')

    # Find bursts
    burst_counts = df.groupby(['market_ticker', 'second']).size().reset_index(name='trades_per_second')
    bursts = burst_counts[burst_counts['trades_per_second'] >= 3]

    burst_df = df.merge(bursts[['market_ticker', 'second']], on=['market_ticker', 'second'])

    # Aggregate by market
    burst_market = burst_df.groupby('market_ticker').agg({
        'taker_side': lambda x: (x == 'no').mean(),
        'no_price': 'mean',
        'market_result': 'first'
    }).reset_index()
    burst_market.columns = ['market_ticker', 'no_ratio', 'avg_no_price', 'market_result']

    # Signal: >60% NO in bursts
    signal = burst_market[burst_market['no_ratio'] > 0.6].copy()

    print(f"\n1. Signal markets: {len(signal)}")

    print(f"\n2. Signal NO price distribution:")
    print(f"   Mean: {signal['avg_no_price'].mean():.1f}")
    print(f"   Median: {signal['avg_no_price'].median():.1f}")

    signal['bet_wins'] = (signal['market_result'] == 'no').astype(int)
    win_rate = signal['bet_wins'].mean()
    avg_no_price = signal['avg_no_price'].mean()
    breakeven = avg_no_price / 100.0
    edge = (win_rate - breakeven) * 100

    print(f"\n3. Edge: {edge:+.2f}%")

    # Price proxy check
    all_markets = df.groupby('market_ticker').agg({
        'no_price': 'mean',
        'market_result': 'first'
    }).reset_index()
    all_markets['bet_wins'] = (all_markets['market_result'] == 'no').astype(int)
    all_markets['price_bucket'] = (all_markets['no_price'] // 10) * 10
    signal['price_bucket'] = (signal['avg_no_price'] // 10) * 10

    print(f"\n4. Price Proxy Check:")
    improvements = []
    for bucket in sorted(signal['price_bucket'].unique()):
        sig_b = signal[signal['price_bucket'] == bucket]
        all_b = all_markets[all_markets['price_bucket'] == bucket]
        if len(sig_b) >= 20 and len(all_b) >= 100:
            sig_wr = sig_b['bet_wins'].mean()
            all_wr = all_b['bet_wins'].mean()
            imp = (sig_wr - all_wr) * 100
            improvements.append({'bucket': bucket, 'n': len(sig_b), 'imp': imp})
            print(f"   {int(bucket)}-{int(bucket)+10}c: Signal {sig_wr:.1%} vs Baseline {all_wr:.1%} = {imp:+.1f}%")

    if improvements:
        total_n = sum(i['n'] for i in improvements)
        weighted_imp = sum(i['imp'] * i['n'] / total_n for i in improvements)
        print(f"\n   Weighted Improvement: {weighted_imp:+.2f}%")

    return signal


def main():
    print("="*70)
    print("SESSION 012 - CRITICAL VERIFICATION")
    print("="*70)
    print("\nThis script verifies our validated strategies are NOT the same bugs as Session 011c")

    df = verify_data_integrity()

    print("\n\n" + "="*70)
    print("VERIFYING EACH VALIDATED STRATEGY")
    print("="*70)

    verify_h087_signal(df)
    verify_h102_signal(df)
    verify_h088_signal(df)

    print("\n\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
Key differences from Session 011c bugs:

1. H087 (Round Size Bot):
   - Session 011c used: 100 - trade_price (buggy when >60% NO trades)
   - Our approach: Use actual no_price column from data
   - Verification: Our avg NO price is distributed across range, not inverted

2. H102 (Leverage Stability):
   - Session 011c: Only claimed edge, didn't check price proxy properly
   - Our approach: Bucket-by-bucket comparison to ALL markets at same price
   - Verification: Positive improvement at multiple price levels

3. H088 (Millisecond Burst):
   - This is a NEW hypothesis not tested in detail in Session 011c
   - Clean slate, no prior bugs to worry about

If the verification above shows:
- Positive improvements at multiple price buckets
- Price distributions that make sense
- Not just selecting extreme prices

Then the strategies are genuinely validated.
""")


if __name__ == "__main__":
    main()
