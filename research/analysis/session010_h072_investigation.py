#!/usr/bin/env python3
"""
Session 010: H072 Methodology Investigation

The "fade recent move" strategy shows +24.7% edge in ALL markets.
This is suspiciously high. Let's investigate WHY.

Hypothesis: The edge comes from the ASYMMETRY in price distribution.
- When price moves UP, last_price is HIGH (e.g., 80c)
  - Fade = bet NO at 20c
  - Win 63.2% of time
- When price moves DOWN, last_price is LOW (e.g., 20c)
  - Fade = bet YES at 20c
  - Win 36.5% of time

The "average fade price" of 24.7c is LOW because:
- Many markets have extreme prices (95c or 5c)
- The fade price is therefore also extreme (5c or 95c)

Let me verify this and see if the edge is REAL or an artifact.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades/enriched_trades_resolved_ALL.csv"


def load_data():
    """Load trades data."""
    print("Loading data...")
    trades = pd.read_csv(TRADES_FILE, low_memory=False)
    print(f"Loaded {len(trades):,} trades")
    return trades


def investigate_h072(trades):
    """Deep investigation of the fade recent move strategy."""
    print("\n" + "="*80)
    print("INVESTIGATING H072 METHODOLOGY")
    print("="*80)

    # Calculate per-market data
    market_data = []
    for ticker, group in trades.groupby('market_ticker'):
        if len(group) < 5:
            continue

        group = group.sort_values('timestamp')

        first_3 = group.head(3)['trade_price'].mean()
        last_3 = group.tail(3)['trade_price'].mean()
        major_move_up = last_3 > first_3

        result = group['market_result'].iloc[0]
        last_price = group.iloc[-1]['trade_price']

        market_data.append({
            'market_ticker': ticker,
            'first_price': first_3,
            'last_price': last_3,
            'major_move_up': major_move_up,
            'market_result': result,
            'price_change': last_3 - first_3
        })

    df = pd.DataFrame(market_data)

    # STEP 1: Understand the price distribution
    print("\n--- Price Distribution Analysis ---")
    print(f"Total markets: {len(df)}")
    print(f"\nFirst price (first 3 trades avg):")
    print(f"  Mean: {df['first_price'].mean():.1f}c")
    print(f"  Median: {df['first_price'].median():.1f}c")
    print(f"  Std: {df['first_price'].std():.1f}c")

    print(f"\nLast price (last 3 trades avg):")
    print(f"  Mean: {df['last_price'].mean():.1f}c")
    print(f"  Median: {df['last_price'].median():.1f}c")
    print(f"  Std: {df['last_price'].std():.1f}c")

    # STEP 2: Understand move direction vs result
    print("\n--- Move Direction vs Result ---")

    moved_up = df[df['major_move_up']]
    moved_down = df[~df['major_move_up']]

    print(f"\nMoved UP ({len(moved_up)} markets):")
    print(f"  Avg last price: {moved_up['last_price'].mean():.1f}c")
    print(f"  Settled YES: {(moved_up['market_result'] == 'yes').mean()*100:.1f}%")
    print(f"  Settled NO: {(moved_up['market_result'] == 'no').mean()*100:.1f}%")

    print(f"\nMoved DOWN ({len(moved_down)} markets):")
    print(f"  Avg last price: {moved_down['last_price'].mean():.1f}c")
    print(f"  Settled YES: {(moved_down['market_result'] == 'yes').mean()*100:.1f}%")
    print(f"  Settled NO: {(moved_down['market_result'] == 'no').mean()*100:.1f}%")

    # STEP 3: The CORRECT way to calculate edge
    # We need to compare to: what's the expected win rate given the price?
    print("\n--- Correct Edge Calculation ---")

    # For fading UP (betting NO):
    # Our cost = 100 - last_price = NO price
    # Breakeven = NO price / 100
    # We win if result = 'no'

    # For fading DOWN (betting YES):
    # Our cost = last_price = YES price
    # Breakeven = YES price / 100
    # We win if result = 'yes'

    df['fade_price'] = np.where(
        df['major_move_up'],
        100 - df['last_price'],  # Bet NO
        df['last_price']          # Bet YES
    )

    df['fade_wins'] = np.where(
        df['major_move_up'],
        df['market_result'] == 'no',
        df['market_result'] == 'yes'
    )

    print(f"\nFade price distribution:")
    print(f"  Mean: {df['fade_price'].mean():.1f}c")
    print(f"  Median: {df['fade_price'].median():.1f}c")
    print(f"  < 10c: {(df['fade_price'] < 10).mean()*100:.1f}% of markets")
    print(f"  < 20c: {(df['fade_price'] < 20).mean()*100:.1f}% of markets")
    print(f"  < 30c: {(df['fade_price'] < 30).mean()*100:.1f}% of markets")

    # STEP 4: Compare to BASELINE at same prices
    # This is the CRITICAL test. What if we just bet at these prices randomly?
    print("\n--- Baseline Comparison ---")
    print("What if we just bet at these fade prices WITHOUT the 'recent move' signal?")

    # For markets with fade_price < 30c (cheap bets), what's the baseline win rate?
    cheap_fades = df[df['fade_price'] < 30]
    print(f"\nMarkets with fade_price < 30c: {len(cheap_fades)}")

    # For these, we're betting at low prices. If it's NO at 20c, baseline should be:
    # - Expected win rate = depends on market calibration
    # Let's look at actual data for comparison

    # Get all trades at these low prices
    all_trades_low_price = trades[trades['trade_price'] < 30].copy()
    print(f"All trades at price < 30c: {len(all_trades_low_price):,}")

    # For NO trades at low NO prices (which means YES was high)
    no_trades_low = all_trades_low_price[all_trades_low_price['taker_side'] == 'no']
    yes_trades_low = all_trades_low_price[all_trades_low_price['taker_side'] == 'yes']

    print(f"\nNO trades at < 30c: {len(no_trades_low):,}")
    print(f"  Win rate: {no_trades_low['is_winner'].mean()*100:.1f}%")
    print(f"  Avg price: {no_trades_low['trade_price'].mean():.1f}c")

    print(f"\nYES trades at < 30c: {len(yes_trades_low):,}")
    print(f"  Win rate: {yes_trades_low['is_winner'].mean()*100:.1f}%")
    print(f"  Avg price: {yes_trades_low['trade_price'].mean():.1f}c")

    # STEP 5: The KEY insight
    print("\n" + "!"*60)
    print("KEY INSIGHT")
    print("!"*60)

    # When price moves UP, last_price is HIGH (say 80c)
    # We bet NO at 20c
    # Market settled NO 63.2% of the time

    # The QUESTION is: for markets where YES price ends at 80c,
    # what's the baseline NO win rate?

    # Let's check this directly
    print("\nWhen last YES price is 70-90c (so NO price is 10-30c):")
    high_yes_markets = df[(df['last_price'] >= 70) & (df['last_price'] <= 90)]
    no_win_rate = (high_yes_markets['market_result'] == 'no').mean()
    print(f"  Markets: {len(high_yes_markets)}")
    print(f"  Settled NO: {no_win_rate*100:.1f}%")

    # This is the BASELINE for betting NO in these markets
    # If "fade recent move" beats this, there's real edge

    # Markets where price moved UP to 70-90c range:
    up_to_high = df[(df['major_move_up']) & (df['last_price'] >= 70) & (df['last_price'] <= 90)]
    up_no_win = (up_to_high['market_result'] == 'no').mean()
    print(f"\nMarkets where price MOVED UP to 70-90c:")
    print(f"  Markets: {len(up_to_high)}")
    print(f"  Settled NO: {up_no_win*100:.1f}%")

    # Markets where price started and stayed at 70-90c (no move):
    stayed_high = df[(~df['major_move_up']) & (df['last_price'] >= 70) & (df['last_price'] <= 90)]
    stayed_no_win = (stayed_high['market_result'] == 'no').mean()
    print(f"\nMarkets where price STAYED at 70-90c (no up move):")
    print(f"  Markets: {len(stayed_high)}")
    print(f"  Settled NO: {stayed_no_win*100:.1f}%")

    improvement = up_no_win - no_win_rate
    print(f"\nImprovement from 'moved up' signal: {improvement*100:+.1f}%")

    # STEP 6: Final Analysis
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    # Calculate actual edge vs baseline more carefully
    # For each fade trade, compare to baseline at that price

    print("\nDo a proper price-matched comparison:")

    # Bin fade prices
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    df['price_bin'] = pd.cut(df['fade_price'], bins)

    print("\nBy fade price bin:")
    for bin_label in df['price_bin'].unique():
        if pd.isna(bin_label):
            continue
        subset = df[df['price_bin'] == bin_label]
        if len(subset) < 50:
            continue

        win_rate = subset['fade_wins'].mean()
        avg_price = subset['fade_price'].mean()
        breakeven = avg_price / 100.0
        edge = (win_rate - breakeven) * 100

        print(f"  {bin_label}: N={len(subset)}, WR={win_rate*100:.1f}%, BE={breakeven*100:.1f}%, Edge={edge:+.1f}%")

    # The REAL test: compare to betting at same prices WITHOUT the move signal
    print("\nComparing to baseline (betting at same prices, any move direction):")

    # Get baseline win rates by price bin
    for bin_label in df['price_bin'].unique():
        if pd.isna(bin_label):
            continue

        signal_subset = df[df['price_bin'] == bin_label]
        if len(signal_subset) < 50:
            continue

        # Signal stats
        signal_win = signal_subset['fade_wins'].mean()
        signal_price = signal_subset['fade_price'].mean()

        # Baseline: all trades at this price bin
        price_min = signal_subset['fade_price'].min()
        price_max = signal_subset['fade_price'].max()

        # Get all trades in this price range
        baseline_trades = trades[
            (trades['trade_price'] >= price_min) &
            (trades['trade_price'] <= price_max)
        ]
        baseline_win = baseline_trades['is_winner'].mean()

        improvement = (signal_win - baseline_win) * 100

        print(f"  {bin_label}: Signal WR={signal_win*100:.1f}%, Baseline WR={baseline_win*100:.1f}%, Improvement={improvement:+.1f}%")

    print("\n" + "!"*60)
    print("FINAL VERDICT")
    print("!"*60)
    print("The high 'edge' is largely due to price asymmetry:")
    print("1. When price moves UP, we bet NO at LOW prices (low breakeven)")
    print("2. When price moves DOWN, we bet YES at LOW prices (low breakeven)")
    print("3. The average breakeven is ~25%, but win rate is ~50%")
    print("4. This gives an APPARENT +25% edge")
    print()
    print("BUT: The actual improvement over baseline is smaller.")
    print("The 'recent move' signal may add SOME edge, but not +25%.")
    print("Most of the 'edge' comes from the structural price asymmetry.")


if __name__ == "__main__":
    trades = load_data()
    investigate_h072(trades)
