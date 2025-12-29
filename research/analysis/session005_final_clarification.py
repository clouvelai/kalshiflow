#!/usr/bin/env python3
"""
Session 005: FINAL CLARIFICATION

CRITICAL DISCOVERY: The trade_price column has DIFFERENT meanings
depending on taker_side!

For YES trades: trade_price = yes_price (what the taker paid)
For NO trades: trade_price = no_price (what the taker paid)

This is OPPOSITE of what I initially thought!

Let me verify and recalculate everything correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades" / "enriched_trades_resolved_ALL.csv"

print("=" * 80)
print("SESSION 005: FINAL CLARIFICATION")
print("=" * 80)
print()

# Load data
trades_df = pd.read_csv(TRADES_FILE)
if trades_df['is_winner'].dtype == 'object':
    trades_df['is_winner'] = trades_df['is_winner'].map({'True': True, 'False': False, True: True, False: False})

def extract_base_market(ticker):
    parts = ticker.rsplit('-', 1)
    if len(parts) == 2:
        if re.match(r'^[A-Z0-9]{1,10}$', parts[1]) and len(parts[1]) <= 10:
            return parts[0]
    return ticker

trades_df['base_market'] = trades_df['market_ticker'].apply(extract_base_market)

print(f"Total trades: {len(trades_df):,}")
print()

# =============================================================================
# VERIFY trade_price meaning
# =============================================================================
print("=" * 80)
print("STEP 1: Verify trade_price meaning")
print("=" * 80)

# Check for YES trades
yes_trades = trades_df[trades_df['taker_side'] == 'yes'].sample(10)
print("\nYES trades:")
for _, row in yes_trades.iterrows():
    calc_from_yes = row['yes_price'] * row['count'] / 100
    calc_from_trade = row['trade_price'] * row['count'] / 100
    print(f"  yes_price={row['yes_price']}, trade_price={row['trade_price']}, "
          f"cost=${row['cost_dollars']:.2f}, calc_yes=${calc_from_yes:.2f}, calc_trade=${calc_from_trade:.2f}")

# Check for NO trades
no_trades = trades_df[trades_df['taker_side'] == 'no'].sample(10)
print("\nNO trades:")
for _, row in no_trades.iterrows():
    calc_from_no = row['no_price'] * row['count'] / 100
    calc_from_trade = row['trade_price'] * row['count'] / 100
    print(f"  no_price={row['no_price']}, trade_price={row['trade_price']}, "
          f"cost=${row['cost_dollars']:.2f}, calc_no=${calc_from_no:.2f}, calc_trade=${calc_from_trade:.2f}")

print("""
CONCLUSION: trade_price = what the taker PAID
- For YES taker: trade_price = yes_price
- For NO taker: trade_price = no_price
""")

# =============================================================================
# NOW RECALCULATE EDGE CORRECTLY
# =============================================================================
print("=" * 80)
print("STEP 2: Recalculate Edge for NO at 80-90c")
print("=" * 80)

# NO trades where trade_price (= no_price) is 80-90c
# This means the taker paid 80-90c for NO
# Which means YES was priced at 10-20c (very unlikely to happen)

no_80_90 = trades_df[
    (trades_df['taker_side'] == 'no') &
    (trades_df['trade_price'] >= 80) &
    (trades_df['trade_price'] < 90)
].copy()

print(f"\nNO trades where taker paid 80-90c for NO:")
print(f"  Total trades: {len(no_80_90):,}")
print(f"  Unique markets: {no_80_90['base_market'].nunique():,}")

# What does this mean?
# If NO costs 85c, YES costs 15c
# The market thinks YES has only 15% chance
# Betting NO at 85c means: win 15c if NO wins, lose 85c if YES wins

# Market-level aggregation
market_stats = no_80_90.groupby('base_market').agg({
    'is_winner': 'first',
    'trade_price': 'mean',  # This is NO price (what they paid)
    'yes_price': 'mean',
    'no_price': 'mean'
}).reset_index()

n = len(market_stats)
wins = market_stats['is_winner'].sum()
win_rate = wins / n
avg_no_price = market_stats['no_price'].mean()  # 80-90c
avg_yes_price = market_stats['yes_price'].mean()  # Should be 10-20c

print(f"\nMarket-level stats:")
print(f"  Markets: {n}")
print(f"  NO wins: {wins} ({win_rate:.1%})")
print(f"  Average NO price paid: {avg_no_price:.1f}c")
print(f"  Average YES price: {avg_yes_price:.1f}c")

# Breakeven for NO bet at price X:
# If NO costs X cents, I need NO to win X% of the time to break even
breakeven = avg_no_price / 100
print(f"  Breakeven win rate: {breakeven:.1%}")

edge = win_rate - breakeven
print(f"  Edge = {win_rate:.1%} - {breakeven:.1%} = {edge:+.1%}")

print(f"""
INTERPRETATION:
- We're betting NO when YES is only 10-20c (the market thinks YES is unlikely)
- We pay 80-90c for NO
- NO wins {win_rate:.1%} of the time
- But we need it to win {breakeven:.1%} to break even
- So we're actually LOSING {-edge:.1%} on this strategy!
""")

# =============================================================================
# WAIT - Let me re-read the original analysis
# =============================================================================
print("=" * 80)
print("STEP 3: Understanding the ORIGINAL analysis filter")
print("=" * 80)
print("""
The ORIGINAL analysis in session004 used this logic:

```python
if side == 'no':
    breakeven_rate = (100 - avg_price) / 100.0
```

This suggests they were interpreting trade_price as YES_PRICE.
Let me check if that's what the filter was actually doing...

Actually wait - let me check if trade_price == no_price OR trade_price == yes_price
""")

# Check correlation
trades_df['tp_equals_yes'] = trades_df['trade_price'] == trades_df['yes_price']
trades_df['tp_equals_no'] = trades_df['trade_price'] == trades_df['no_price']

print(f"For ALL trades:")
print(f"  trade_price == yes_price: {trades_df['tp_equals_yes'].mean()*100:.1f}%")
print(f"  trade_price == no_price: {trades_df['tp_equals_no'].mean()*100:.1f}%")

yes_only = trades_df[trades_df['taker_side'] == 'yes']
print(f"\nFor YES trades:")
print(f"  trade_price == yes_price: {(yes_only['trade_price'] == yes_only['yes_price']).mean()*100:.1f}%")

no_only = trades_df[trades_df['taker_side'] == 'no']
print(f"\nFor NO trades:")
print(f"  trade_price == no_price: {(no_only['trade_price'] == no_only['no_price']).mean()*100:.1f}%")

# =============================================================================
# RECONSIDER: What strategy makes sense?
# =============================================================================
print()
print("=" * 80)
print("STEP 4: What strategy actually makes sense?")
print("=" * 80)
print("""
The CORRECT interpretation for a profitable NO strategy:
- Find markets where YES is expensive (80-90c) -> NO is cheap (10-20c)
- Bet NO at 10-20c
- If NO wins (even occasionally), you make money

This is betting AGAINST the favorite at extreme prices.

For this, we want: yes_price >= 80 AND yes_price < 90 with taker_side='no'
""")

# Find NO trades where YES was 80-90c (so NO was 10-20c)
no_at_expensive_yes = trades_df[
    (trades_df['taker_side'] == 'no') &
    (trades_df['yes_price'] >= 80) &
    (trades_df['yes_price'] < 90)
].copy()

print(f"NO trades where YES was 80-90c (so NO cost 10-20c):")
print(f"  Total trades: {len(no_at_expensive_yes):,}")
print(f"  Unique markets: {no_at_expensive_yes['base_market'].nunique():,}")

# Market-level stats
market_stats2 = no_at_expensive_yes.groupby('base_market').agg({
    'is_winner': 'first',
    'yes_price': 'mean',
    'no_price': 'mean'
}).reset_index()

n2 = len(market_stats2)
wins2 = market_stats2['is_winner'].sum()
win_rate2 = wins2 / n2
avg_no_price2 = market_stats2['no_price'].mean()

print(f"\nMarket-level stats:")
print(f"  Markets: {n2}")
print(f"  NO wins: {wins2} ({win_rate2:.1%})")
print(f"  Average NO price (our cost): {avg_no_price2:.1f}c")

breakeven2 = avg_no_price2 / 100
edge2 = win_rate2 - breakeven2

print(f"  Breakeven: {breakeven2:.1%}")
print(f"  Edge = {win_rate2:.1%} - {breakeven2:.1%} = {edge2:+.1%}")

# Now this is the same as original! Let me understand...

print()
print("=" * 80)
print("AH HA! THE CONFUSION RESOLVED")
print("=" * 80)
print("""
I was confused because:

1. For NO trades, trade_price = NO_PRICE (what taker pays for NO)
2. The original filter was: trade_price >= 80 AND trade_price < 90
3. This means: NO costs 80-90c (so YES costs 10-20c)

But the STRATEGY we want is the OPPOSITE:
- Bet NO when YES is expensive (80-90c)
- Which means NO is cheap (10-20c)

The original analysis was BACKWARDS:
- They found markets where NO was EXPENSIVE (80-90c)
- These are markets where YES was cheap (10-20c)
- The market already thought YES was unlikely
- Betting NO is the FAVORITE bet here, not the underdog

The breakeven calculation was also wrong. They used:
breakeven = (100 - trade_price) / 100 = (100 - 85) / 100 = 15%

But if NO costs 85c, the breakeven is 85%, not 15%!
""")

# =============================================================================
# CORRECT EDGE FOR "BET NO WHEN YES IS EXPENSIVE"
# =============================================================================
print()
print("=" * 80)
print("CORRECT STRATEGY: Bet NO when YES is 80-90c (expensive favorites)")
print("=" * 80)

# This is what makes sense as a strategy
correct_no = trades_df[
    (trades_df['taker_side'] == 'no') &
    (trades_df['yes_price'] >= 80) &
    (trades_df['yes_price'] < 90)
].copy()

print(f"\nNO trades where YES was 80-90c:")
print(f"  Total trades: {len(correct_no):,}")
print(f"  Unique markets: {correct_no['base_market'].nunique():,}")

# Sample to verify prices
sample = correct_no.sample(5)[['yes_price', 'no_price', 'trade_price', 'cost_dollars', 'count']]
print("\nSample trades:")
print(sample)

# Market-level
mkt = correct_no.groupby('base_market').agg({
    'is_winner': 'first',
    'yes_price': 'mean',
    'no_price': 'mean',
    'market_result': 'first'
}).reset_index()

n = len(mkt)
wins = mkt['is_winner'].sum()
wr = wins / n
avg_no = mkt['no_price'].mean()
be = avg_no / 100
edge = wr - be

print(f"\nMarket-level analysis:")
print(f"  Markets: {n:,}")
print(f"  NO wins: {wins:,} ({wr:.1%})")
print(f"  Average NO cost: {avg_no:.1f}c")
print(f"  Breakeven: {be:.1%}")
print(f"  EDGE = {wr:.1%} - {be:.1%} = {edge:+.1%}")

# Expected profit per bet
profit_win = (100 - avg_no) / 100
loss_lose = avg_no / 100
ev = wr * profit_win - (1 - wr) * loss_lose
roi = ev / (avg_no / 100)

print(f"\nExpected value per $1 risked:")
print(f"  Win: {wr:.1%} x ${profit_win:.2f} = ${wr * profit_win:.4f}")
print(f"  Lose: {1-wr:.1%} x ${loss_lose:.2f} = ${(1-wr) * loss_lose:.4f}")
print(f"  EV = ${ev:.4f} per bet of ${avg_no/100:.2f}")
print(f"  ROI = {roi:.1%}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)
print(f"""
THE CLAIMED +69% EDGE WAS A CALCULATION ERROR!

The original analysis made two mistakes:
1. Filtering on trade_price >= 80 finds NO trades where NO costs 80-90c
   (not where YES costs 80-90c)
2. Using breakeven = (100 - trade_price) / 100 gives the wrong breakeven

CORRECT INTERPRETATION:
- When filtering for trade_price >= 80 with taker_side='no':
  - We're finding bets where NO was EXPENSIVE (80-90c)
  - YES was CHEAP (10-20c)
  - The market thought YES was unlikely
  - This is NOT the underdog bet, it's the favorite bet!

THE REAL EDGE for "bet NO when YES is expensive (80-90c)":
  - Markets: {n:,}
  - Win rate: {wr:.1%}
  - Average NO cost: {avg_no:.1f}c
  - Breakeven: {be:.1%}
  - Edge: {edge:+.1%}

This is a {abs(edge)*100:.1f}% {"POSITIVE" if edge > 0 else "NEGATIVE"} edge.

If edge is POSITIVE, this is a genuine profitable strategy.
If edge is NEGATIVE, the previous claims were completely wrong.
""")
