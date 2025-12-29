#!/usr/bin/env python3
"""
Session 005: Deep Investigation of Edge Calculation

The claimed +69% edge seems impossibly high. Let's trace through EXACTLY
what the calculation is doing and verify if it's meaningful.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades" / "enriched_trades_resolved_ALL.csv"

print("=" * 80)
print("SESSION 005: DEEP INVESTIGATION")
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
# FOCUS: NO trades at 80-90c
# =============================================================================
print("=" * 80)
print("INVESTIGATING NO TRADES AT 80-90c")
print("=" * 80)

no_trades = trades_df[
    (trades_df['taker_side'] == 'no') &
    (trades_df['trade_price'] >= 80) &
    (trades_df['trade_price'] < 90)
].copy()

print(f"\nTotal NO trades at 80-90c: {len(no_trades):,}")
print(f"Unique markets: {no_trades['base_market'].nunique():,}")
print()

# What does 'trade_price' mean for a NO trade?
print("Understanding trade_price for NO trades:")
print(no_trades[['trade_price', 'yes_price', 'no_price', 'cost_dollars']].head(20))
print()

# It looks like trade_price might be the YES price, not the NO price!
# Let's verify
print("Checking if trade_price = yes_price:")
print(f"  Mean trade_price: {no_trades['trade_price'].mean():.1f}")
print(f"  Mean yes_price: {no_trades['yes_price'].mean():.1f}")
print(f"  Mean no_price: {no_trades['no_price'].mean():.1f}")
print()

# Check specific rows
print("Sample rows showing trade_price vs yes_price vs no_price:")
sample = no_trades.sample(10)[['taker_side', 'trade_price', 'yes_price', 'no_price', 'cost_dollars', 'count']]
print(sample)
print()

# =============================================================================
# THE KEY QUESTION: What does trade_price represent?
# =============================================================================
print("=" * 80)
print("KEY QUESTION: What is trade_price for NO trades?")
print("=" * 80)
print()

# If taker_side = 'no', the taker is buying NO contracts
# trade_price should be the price they paid for NO (i.e., no_price)
# But let's verify by checking the cost

# For NO trades:
# cost_dollars should = no_price * count / 100 (if no_price is in cents)
# OR cost_dollars = trade_price * count / 100 (if trade_price is what they paid)

no_sample = no_trades.sample(5)
for idx, row in no_sample.iterrows():
    calc_cost_from_no_price = row['no_price'] * row['count'] / 100
    calc_cost_from_trade_price = row['trade_price'] * row['count'] / 100
    print(f"Trade ID {row['id']}:")
    print(f"  count: {row['count']}, no_price: {row['no_price']}, trade_price: {row['trade_price']}")
    print(f"  Actual cost: ${row['cost_dollars']:.2f}")
    print(f"  If cost = no_price * count: ${calc_cost_from_no_price:.2f}")
    print(f"  If cost = trade_price * count: ${calc_cost_from_trade_price:.2f}")
    print()

# =============================================================================
# NOW THE CRITICAL CHECK: Win rate calculation
# =============================================================================
print("=" * 80)
print("CRITICAL CHECK: Win Rate Meaning")
print("=" * 80)

# For NO trades, is_winner = True means the NO bet won
# Let's verify by checking market_result
print("\nVerifying is_winner for NO trades:")
print("If is_winner=True, market_result should be 'no'")
print("If is_winner=False, market_result should be 'yes'")
print()

crosstab = pd.crosstab(no_trades['is_winner'], no_trades['market_result'])
print(crosstab)
print()

# Aggregate to market level
market_stats = no_trades.groupby('base_market').agg({
    'is_winner': 'first',
    'trade_price': 'mean',
    'no_price': 'mean',
    'market_result': 'first'
}).reset_index()

print(f"\nMarket-level aggregation ({len(market_stats)} unique markets):")
print(f"  Markets where NO wins (is_winner=True): {market_stats['is_winner'].sum()}")
print(f"  Markets where YES wins (is_winner=False): {(~market_stats['is_winner']).sum()}")
print(f"  Win rate: {market_stats['is_winner'].mean():.1%}")
print()

# =============================================================================
# EDGE CALCULATION - STEP BY STEP
# =============================================================================
print("=" * 80)
print("EDGE CALCULATION - STEP BY STEP")
print("=" * 80)

win_rate = market_stats['is_winner'].mean()
avg_trade_price = market_stats['trade_price'].mean()
avg_no_price = market_stats['no_price'].mean()

print(f"\nFacts:")
print(f"  Unique markets with NO trades at 80-90c: {len(market_stats)}")
print(f"  Markets where NO won: {market_stats['is_winner'].sum()}")
print(f"  Win rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
print(f"  Average trade_price: {avg_trade_price:.2f}c")
print(f"  Average no_price: {avg_no_price:.2f}c")
print()

# What's the breakeven?
# If I buy NO at X cents, I need NO to win X% of the time to break even
# Because: Win = $1, Lose = $0, Cost = $X/100
# Expected value = X% * ($1 - $X/100) - (1-X%) * $X/100 = X% - X%*X/100 - X/100 + X%*X/100 = X% - X/100
# Wait, that's not right. Let me recalculate.

# If NO costs X cents:
# - If NO wins: I get $1 (profit = $1 - $X/100 = $(1-X/100))
# - If NO loses: I get $0 (profit = -$X/100)
# Expected profit = WR * (1 - X/100) - (1-WR) * X/100
#                 = WR - WR*X/100 - X/100 + WR*X/100
#                 = WR - X/100
# Breakeven: WR - X/100 = 0 => WR = X/100

# So if NO costs 15c, breakeven WR is 15%

# BUT WAIT - the filter is trade_price >= 80, trade_price < 90
# This means trade_price is around 85c
# If trade_price for NO is 85c... that would mean NO costs 85c!
# But that doesn't match no_price

print("UNDERSTANDING THE FILTER:")
print(f"  Filter: trade_price >= 80 AND trade_price < 90")
print(f"  Average trade_price in filtered set: {avg_trade_price:.2f}c")
print(f"  Average no_price in filtered set: {avg_no_price:.2f}c")
print()

# AH! The issue is that trade_price might be the YES price, not the NO price!
# Let's check if trade_price = yes_price always
trades_df['trade_price_equals_yes'] = (trades_df['trade_price'] == trades_df['yes_price'])
print(f"trade_price == yes_price: {trades_df['trade_price_equals_yes'].mean()*100:.1f}%")

# If trade_price == yes_price, then for NO trades:
# When we filter trade_price >= 80, we're filtering for YES_PRICE >= 80
# This means NO_PRICE <= 20 (since yes_price + no_price = 100)

print()
print("=" * 80)
print("THE ERROR DISCOVERED!")
print("=" * 80)
print("""
The filter 'trade_price >= 80 AND trade_price < 90' with taker_side='no'
is filtering for NO trades where the YES_PRICE is 80-90c.

This means the NO_PRICE is 10-20c!

The previous analysis used breakeven = (100 - trade_price) / 100 = (100-85)/100 = 15%
This assumes that the trade_price is what the trader PAID.
But trade_price IS the yes_price, so the trader paid 100 - trade_price = 15c

So breakeven = 15% and actual win rate = 84.5%
Edge = 84.5% - 15% = +69.5% ???

Let me verify this makes sense...

If YES is priced at 85c, NO is priced at 15c.
If you buy NO at 15c:
- If NO wins (which happens {win_rate:.1%} of the time), you profit 85c
- If YES wins, you lose 15c

Expected value per $1 bet on NO:
= {win_rate:.1%} * $0.85 - {1-win_rate:.1%} * $0.15
""")

ev_per_no_bet = win_rate * 0.85 - (1 - win_rate) * 0.15
print(f"= {win_rate:.1%} * $0.85 - {(1-win_rate):.1%} * $0.15")
print(f"= ${win_rate * 0.85:.4f} - ${(1-win_rate) * 0.15:.4f}")
print(f"= ${ev_per_no_bet:.4f} per NO trade")
print()

# What's the ROI?
roi = ev_per_no_bet / 0.15
print(f"ROI = ${ev_per_no_bet:.4f} / $0.15 = {roi:.1%}")
print()

print("=" * 80)
print("WAIT - THIS EDGE IS WAY TOO HIGH")
print("=" * 80)
print("""
An 84.5% win rate when breakeven is 15% seems impossibly good.
Let me check if the is_winner field is correct.
""")

# Let's verify is_winner makes sense
print("\nCross-checking is_winner with market_result:")
print("For NO trades, is_winner should be True when market_result='no'")
print()

no_trades_check = no_trades[['market_result', 'is_winner']].copy()
check1 = (no_trades_check['market_result'] == 'no') & (no_trades_check['is_winner'] == True)
check2 = (no_trades_check['market_result'] == 'yes') & (no_trades_check['is_winner'] == False)
check3 = (no_trades_check['market_result'] == 'no') & (no_trades_check['is_winner'] == False)
check4 = (no_trades_check['market_result'] == 'yes') & (no_trades_check['is_winner'] == True)

print(f"market_result='no' AND is_winner=True: {check1.sum():,} (CORRECT)")
print(f"market_result='yes' AND is_winner=False: {check2.sum():,} (CORRECT)")
print(f"market_result='no' AND is_winner=False: {check3.sum():,} (WRONG if any)")
print(f"market_result='yes' AND is_winner=True: {check4.sum():,} (WRONG if any)")
print()

# =============================================================================
# THE REAL QUESTION: Is this selection bias?
# =============================================================================
print("=" * 80)
print("SELECTION BIAS INVESTIGATION")
print("=" * 80)
print("""
The edge seems real if we trust the data. But WHY would this edge exist?

Hypothesis: Selection bias - the markets where someone bets NO at high YES prices
are NOT random. They're markets where informed traders know something.

Let's check: What fraction of ALL markets have NO trades at 80-90c?
""")

total_markets = trades_df['base_market'].nunique()
markets_with_no_80_90 = no_trades['base_market'].nunique()
print(f"Total unique markets: {total_markets:,}")
print(f"Markets with NO trades at 80-90c: {markets_with_no_80_90:,}")
print(f"Percentage: {markets_with_no_80_90/total_markets*100:.2f}%")
print()

# What's the base rate of NO wins in ALL markets?
trades_df['datetime_parsed'] = pd.to_datetime(trades_df['datetime'])
last_trades = trades_df.loc[trades_df.groupby('base_market')['datetime_parsed'].idxmax()].copy()
no_win_rate_all = (last_trades['market_result'] == 'no').mean()
print(f"NO win rate across ALL markets: {no_win_rate_all:.1%}")
print()

# What's the NO win rate in markets that had NO trades at 80-90c?
markets_with_no_trades = set(no_trades['base_market'].unique())
last_trades_subset = last_trades[last_trades['base_market'].isin(markets_with_no_trades)]
no_win_rate_subset = (last_trades_subset['market_result'] == 'no').mean()
print(f"NO win rate in markets with NO trades at 80-90c: {no_win_rate_subset:.1%}")
print()

print("""
INTERPRETATION:
If the NO win rate in this subset is much higher than the base rate,
it means these are SPECIAL markets - ones where NO was more likely to win.
The NO traders at 80-90c were selecting markets where NO would win!

This is either:
a) Informed trading (they knew something)
b) Lucky coincidence
c) Data error

Given the large sample (1,676 markets), option (c) is unlikely.
Option (b) is possible but 84.5% vs 87.5% base rate doesn't fully explain it.

The edge might be REAL but it's not magical - it's detecting when
informed traders make contrarian bets at extreme prices.
""")

# =============================================================================
# REALITY CHECK: Expected Profit
# =============================================================================
print()
print("=" * 80)
print("REALITY CHECK: Expected Profit")
print("=" * 80)

# If we bet $0.15 (15c) on every NO trade at 80-90c, what's our total profit?
total_bets = len(market_stats)
wins = market_stats['is_winner'].sum()
losses = total_bets - wins

profit_per_win = 0.85  # 100c - 15c paid
loss_per_loss = 0.15   # The 15c we paid

total_profit = wins * profit_per_win - losses * loss_per_loss
total_wagered = total_bets * 0.15

print(f"Total markets bet on: {total_bets}")
print(f"Wins: {wins}")
print(f"Losses: {losses}")
print(f"Profit from wins: ${wins * profit_per_win:.2f}")
print(f"Loss from losses: ${losses * loss_per_loss:.2f}")
print(f"Net profit: ${total_profit:.2f}")
print(f"Total wagered: ${total_wagered:.2f}")
print(f"ROI: {total_profit/total_wagered*100:.1f}%")
print()

# Hmm, let's use actual dollars from the data
actual_profit = no_trades.groupby('base_market')['actual_profit_dollars'].sum().sum()
actual_cost = no_trades.groupby('base_market')['cost_dollars'].sum().sum()
print(f"From actual data:")
print(f"  Total actual profit: ${actual_profit:,.2f}")
print(f"  Total actual cost: ${actual_cost:,.2f}")
print(f"  ROI: {actual_profit/actual_cost*100:.1f}%")
