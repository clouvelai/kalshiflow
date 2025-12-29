#!/usr/bin/env python3
"""
Session 005: Rigorous Verification of Claimed Edges

URGENT: Previous research claimed edges of +69% for NO at 80-90c and +90% for NO at 90-100c.
These seem TOO HIGH to be correct. This script verifies from scratch.

Author: Quant Agent (Opus 4.5)
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades" / "enriched_trades_resolved_ALL.csv"
MARKETS_FILE = DATA_DIR / "markets" / "market_outcomes_ALL.csv"
RAW_TRADES_FILE = DATA_DIR / "trades" / "historical_trades_ALL.csv"

print("=" * 80)
print("SESSION 005: RIGOROUS EDGE VERIFICATION")
print("=" * 80)
print()

# =============================================================================
# STEP 1: DATA COMPLETENESS CHECK
# =============================================================================
print("STEP 1: DATA COMPLETENESS CHECK")
print("-" * 40)

# Load the enriched trades (has outcome data)
print(f"Loading: {TRADES_FILE}")
trades_df = pd.read_csv(TRADES_FILE)
print(f"  Total trades: {len(trades_df):,}")
print(f"  Columns: {list(trades_df.columns)}")
print(f"  Unique markets: {trades_df['market_ticker'].nunique():,}")
print(f"  Date range: {trades_df['datetime'].min()} to {trades_df['datetime'].max()}")
print()

# Load market outcomes
print(f"Loading: {MARKETS_FILE}")
markets_df = pd.read_csv(MARKETS_FILE)
print(f"  Total markets: {len(markets_df):,}")
print(f"  Markets with YES outcome: {(markets_df['result'] == 'yes').sum():,}")
print(f"  Markets with NO outcome: {(markets_df['result'] == 'no').sum():,}")
print()

# Cross-check: trades should match markets
trade_tickers = set(trades_df['market_ticker'].unique())
market_tickers = set(markets_df['ticker'].unique())
matched = trade_tickers & market_tickers
print(f"  Markets in trades data: {len(trade_tickers):,}")
print(f"  Markets in outcomes data: {len(market_tickers):,}")
print(f"  Matched markets: {len(matched):,}")
print()

# =============================================================================
# STEP 2: CALCULATE FINAL YES PRICE PER MARKET
# =============================================================================
print("STEP 2: CALCULATE FINAL YES PRICE PER MARKET")
print("-" * 40)

# Get the last trade per market
trades_df['created_time_dt'] = pd.to_datetime(trades_df['datetime'])
last_trades = trades_df.loc[trades_df.groupby('market_ticker')['created_time_dt'].idxmax()]
print(f"  Markets with last trade: {len(last_trades):,}")

# Final YES price is yes_price (already in cents 0-100)
# Make sure price is in 0-1 scale for probability
last_trades = last_trades.copy()
last_trades['final_yes_price'] = last_trades['yes_price'] / 100.0  # Convert to 0-1
print(f"  Sample final YES prices: {last_trades['final_yes_price'].head(10).tolist()}")
print()

# Merge with outcomes
last_trades_with_outcome = last_trades.merge(
    markets_df[['ticker', 'result']],
    left_on='market_ticker',
    right_on='ticker',
    how='inner'
)
print(f"  Markets with outcomes: {len(last_trades_with_outcome):,}")
print()

# =============================================================================
# STEP 3: DISTRIBUTION OF FINAL YES PRICES
# =============================================================================
print("STEP 3: DISTRIBUTION OF FINAL YES PRICES")
print("-" * 40)

# Create price buckets (10c each)
buckets = [
    (0.00, 0.10, "0-10c"),
    (0.10, 0.20, "10-20c"),
    (0.20, 0.30, "20-30c"),
    (0.30, 0.40, "30-40c"),
    (0.40, 0.50, "40-50c"),
    (0.50, 0.60, "50-60c"),
    (0.60, 0.70, "60-70c"),
    (0.70, 0.80, "70-80c"),
    (0.80, 0.90, "80-90c"),
    (0.90, 1.00, "90-100c"),
]

print("Price Distribution (Final YES Price):")
total_markets = len(last_trades_with_outcome)
for low, high, label in buckets:
    mask = (last_trades_with_outcome['final_yes_price'] >= low) & (last_trades_with_outcome['final_yes_price'] < high)
    count = mask.sum()
    pct = count / total_markets * 100
    print(f"  {label}: {count:,} markets ({pct:.1f}%)")

# Handle edge case of exactly 1.0
mask_100 = last_trades_with_outcome['final_yes_price'] == 1.0
print(f"  Exactly 100c: {mask_100.sum():,} markets")
print()

# =============================================================================
# STEP 4: EDGE CALCULATION - NO at 80-90c
# =============================================================================
print("STEP 4: EDGE CALCULATION - NO at 80-90c")
print("-" * 40)

# Find markets where final YES price was 80-90c
mask_80_90 = (last_trades_with_outcome['final_yes_price'] >= 0.80) & \
             (last_trades_with_outcome['final_yes_price'] < 0.90)
markets_80_90 = last_trades_with_outcome[mask_80_90].copy()

n_markets = len(markets_80_90)
print(f"  Sample size: {n_markets:,} markets")

# How many resolved to NO?
n_resolved_no = (markets_80_90['result'] == 'no').sum()
n_resolved_yes = (markets_80_90['result'] == 'yes').sum()
print(f"  Resolved to NO: {n_resolved_no:,}")
print(f"  Resolved to YES: {n_resolved_yes:,}")

# Win rate for betting NO
win_rate = n_resolved_no / n_markets if n_markets > 0 else 0
print(f"  Win rate for NO bet: {win_rate:.4f} ({win_rate*100:.2f}%)")

# Calculate average NO price (what we'd pay)
# NO price = 1 - YES price
markets_80_90['no_price'] = 1.0 - markets_80_90['final_yes_price']
avg_no_price = markets_80_90['no_price'].mean()
print(f"  Average NO price: {avg_no_price:.4f} ({avg_no_price*100:.2f}c)")

# Breakeven win rate
# If we pay X for NO, we need to win > X% of the time to profit
breakeven = avg_no_price
print(f"  Breakeven win rate: {breakeven:.4f} ({breakeven*100:.2f}%)")

# EDGE CALCULATION
# Edge = Win Rate - Breakeven Rate
edge = win_rate - breakeven
print(f"  EDGE = {edge:.4f} ({edge*100:.2f}%)")
print()

# Detailed breakdown
print("  === EDGE FORMULA VERIFICATION ===")
print(f"  When YES price = 85c, NO price = 15c")
print(f"  If we bet $0.15 on NO:")
print(f"    - If NO wins: We get $1.00 back (profit = $0.85)")
print(f"    - If YES wins: We lose $0.15")
print(f"  Breakeven: We need NO to win {0.15:.0%} of time to break even")
print(f"  Actual NO win rate in data: {win_rate:.1%}")
print(f"  Therefore Edge = {win_rate:.1%} - {breakeven:.1%} = {edge:.1%}")
print()

# =============================================================================
# STEP 5: EDGE CALCULATION - NO at 90-100c
# =============================================================================
print("STEP 5: EDGE CALCULATION - NO at 90-100c")
print("-" * 40)

# Find markets where final YES price was 90-100c
mask_90_100 = (last_trades_with_outcome['final_yes_price'] >= 0.90) & \
              (last_trades_with_outcome['final_yes_price'] < 1.00)
markets_90_100 = last_trades_with_outcome[mask_90_100].copy()

n_markets_90 = len(markets_90_100)
print(f"  Sample size: {n_markets_90:,} markets")

n_resolved_no_90 = (markets_90_100['result'] == 'no').sum()
n_resolved_yes_90 = (markets_90_100['result'] == 'yes').sum()
print(f"  Resolved to NO: {n_resolved_no_90:,}")
print(f"  Resolved to YES: {n_resolved_yes_90:,}")

win_rate_90 = n_resolved_no_90 / n_markets_90 if n_markets_90 > 0 else 0
print(f"  Win rate for NO bet: {win_rate_90:.4f} ({win_rate_90*100:.2f}%)")

markets_90_100['no_price'] = 1.0 - markets_90_100['final_yes_price']
avg_no_price_90 = markets_90_100['no_price'].mean()
print(f"  Average NO price: {avg_no_price_90:.4f} ({avg_no_price_90*100:.2f}c)")

breakeven_90 = avg_no_price_90
print(f"  Breakeven win rate: {breakeven_90:.4f} ({breakeven_90*100:.2f}%)")

edge_90 = win_rate_90 - breakeven_90
print(f"  EDGE = {edge_90:.4f} ({edge_90*100:.2f}%)")
print()

# =============================================================================
# STEP 6: SANITY CHECK - IS THIS SUBSET BIASED?
# =============================================================================
print("STEP 6: SANITY CHECK - IS THIS A BIASED SUBSET?")
print("-" * 40)

print(f"  Total markets with final trades: {total_markets:,}")
print(f"  Markets with YES 80-90c: {n_markets:,} ({n_markets/total_markets*100:.1f}%)")
print(f"  Markets with YES 90-100c: {n_markets_90:,} ({n_markets_90/total_markets*100:.1f}%)")
print()

# Check if final price correlates with outcome
print("  Outcome distribution by final YES price bucket:")
for low, high, label in buckets:
    mask = (last_trades_with_outcome['final_yes_price'] >= low) & (last_trades_with_outcome['final_yes_price'] < high)
    subset = last_trades_with_outcome[mask]
    if len(subset) > 0:
        yes_rate = (subset['result'] == 'yes').mean()
        no_rate = (subset['result'] == 'no').mean()
        print(f"    {label}: {len(subset):,} mkts | YES wins: {yes_rate:.1%} | NO wins: {no_rate:.1%}")
print()

# =============================================================================
# STEP 7: OUT-OF-SAMPLE VALIDATION
# =============================================================================
print("STEP 7: OUT-OF-SAMPLE VALIDATION (50/50 TIME SPLIT)")
print("-" * 40)

# Sort by time and split
last_trades_with_outcome_sorted = last_trades_with_outcome.sort_values('created_time_dt')
midpoint = len(last_trades_with_outcome_sorted) // 2
first_half = last_trades_with_outcome_sorted.iloc[:midpoint]
second_half = last_trades_with_outcome_sorted.iloc[midpoint:]

print(f"  First half: {len(first_half):,} markets")
print(f"    Time range: {first_half['created_time_dt'].min()} to {first_half['created_time_dt'].max()}")
print(f"  Second half: {len(second_half):,} markets")
print(f"    Time range: {second_half['created_time_dt'].min()} to {second_half['created_time_dt'].max()}")
print()

# Calculate edge on each half for 80-90c
def calc_edge(df, low, high):
    mask = (df['final_yes_price'] >= low) & (df['final_yes_price'] < high)
    subset = df[mask]
    if len(subset) == 0:
        return 0, 0, 0
    n = len(subset)
    win_rate = (subset['result'] == 'no').mean()
    avg_no_price = (1.0 - subset['final_yes_price']).mean()
    edge = win_rate - avg_no_price
    return n, win_rate, edge

print("  80-90c NO strategy:")
n1, wr1, e1 = calc_edge(first_half, 0.80, 0.90)
n2, wr2, e2 = calc_edge(second_half, 0.80, 0.90)
print(f"    First half:  N={n1:,}, Win Rate={wr1:.1%}, Edge={e1:.1%}")
print(f"    Second half: N={n2:,}, Win Rate={wr2:.1%}, Edge={e2:.1%}")
print(f"    Difference: {abs(e1-e2)*100:.1f} percentage points")
print()

print("  90-100c NO strategy:")
n1, wr1, e1 = calc_edge(first_half, 0.90, 1.00)
n2, wr2, e2 = calc_edge(second_half, 0.90, 1.00)
print(f"    First half:  N={n1:,}, Win Rate={wr1:.1%}, Edge={e1:.1%}")
print(f"    Second half: N={n2:,}, Win Rate={wr2:.1%}, Edge={e2:.1%}")
print(f"    Difference: {abs(e1-e2)*100:.1f} percentage points")
print()

# =============================================================================
# STEP 8: CONCENTRATION CHECK
# =============================================================================
print("STEP 8: CONCENTRATION CHECK (TOP MARKET PROFIT)")
print("-" * 40)

# Calculate profit per market for 80-90c NO strategy
wins_80_90 = markets_80_90['result'] == 'no'
markets_80_90 = markets_80_90.copy()
markets_80_90['profit'] = np.where(wins_80_90, 1.0 - markets_80_90['no_price'], -markets_80_90['no_price'])
total_profit = markets_80_90['profit'].sum()
top_market_profit = markets_80_90.groupby('market_ticker')['profit'].sum().max()
concentration = top_market_profit / total_profit if total_profit > 0 else 0

print(f"  Total profit (80-90c NO): ${total_profit:.2f} per $1 bet per market")
print(f"  Top market profit: ${top_market_profit:.2f}")
print(f"  Concentration: {concentration:.1%}")
print()

# =============================================================================
# STEP 9: REALITY CHECK - WHY DOESN'T EVERYONE DO THIS?
# =============================================================================
print("STEP 9: REALITY CHECK - WHAT'S THE CATCH?")
print("-" * 40)

print("""
POTENTIAL EXPLANATIONS FOR HIGH CLAIMED EDGE:

1. DEFINITION CONFUSION - "Edge" might be defined differently
   - Previous research may have calculated Win Rate - Breakeven incorrectly
   - Or used a different edge formula

2. PRICE UNITS CONFUSION
   - Prices might be in cents (0-100) vs probability (0-1)
   - This could inflate or deflate edge calculations

3. SELECTION BIAS
   - Final trade price IS the market's prediction
   - Betting at final price means betting at market-close prices
   - In practice, you'd bet earlier at different prices

4. EXECUTION IMPOSSIBILITY
   - You can't actually bet at the "final" price
   - By the time it's final, the market is closed
   - Real edge would be lower due to price movement

5. THE EDGE IS REAL BUT...
   - Markets are illiquid at extremes
   - Large bets would move prices
   - Transaction costs eat into edge
""")

# =============================================================================
# STEP 10: RECALCULATE WITH CLEAR METHODOLOGY
# =============================================================================
print()
print("STEP 10: FINAL VERIFIED EDGE CALCULATIONS")
print("=" * 60)

results = {}

for low, high, label in buckets:
    mask = (last_trades_with_outcome['final_yes_price'] >= low) & (last_trades_with_outcome['final_yes_price'] < high)
    subset = last_trades_with_outcome[mask].copy()

    if len(subset) < 50:
        continue

    n = len(subset)
    win_rate_no = (subset['result'] == 'no').mean()
    avg_no_price = (1.0 - subset['final_yes_price']).mean()
    breakeven = avg_no_price
    edge = win_rate_no - breakeven

    # Calculate profit for concentration check
    # Profit = (1 - no_price) if NO wins, else -no_price
    no_price_col = 1.0 - subset['final_yes_price']
    wins = subset['result'] == 'no'
    profit_per_market = np.where(wins, 1.0 - no_price_col, -no_price_col)
    total_profit = profit_per_market.sum()

    results[label] = {
        'n_markets': int(n),
        'no_win_rate': float(win_rate_no),
        'avg_no_price': float(avg_no_price),
        'breakeven': float(breakeven),
        'edge': float(edge),
        'total_profit': float(total_profit)
    }

    print(f"\n{label} NO Strategy:")
    print(f"  Markets: {n:,}")
    print(f"  NO wins: {(subset['result'] == 'no').sum():,} ({win_rate_no:.1%})")
    print(f"  YES wins: {(subset['result'] == 'yes').sum():,} ({1-win_rate_no:.1%})")
    print(f"  Avg NO price: {avg_no_price:.2%} ({avg_no_price*100:.1f}c)")
    print(f"  Breakeven: {breakeven:.2%}")
    print(f"  EDGE = Win Rate - Breakeven = {win_rate_no:.2%} - {breakeven:.2%} = {edge:.2%}")
    print(f"  Profit (per $1 per market): ${total_profit:.2f}")

print()
print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(f"{'Range':<12} {'Markets':<8} {'NO Wins':<10} {'Avg Cost':<10} {'Edge':<10}")
print("-" * 50)
for label, data in results.items():
    print(f"{label:<12} {data['n_markets']:<8,} {data['no_win_rate']*100:<10.1f}% {data['avg_no_price']*100:<10.1f}c {data['edge']*100:<10.1f}%")

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)

if 0.80 <= 0.85 < 0.90:  # Check 80-90c bucket
    r = results.get("80-90c", {})
    claimed_edge = 69.2
    actual_edge = r.get('edge', 0) * 100
    print(f"""
VERIFICATION RESULT FOR NO at 80-90c:
  Claimed Edge: +{claimed_edge:.1f}%
  Verified Edge: {actual_edge:+.1f}%
  Discrepancy: {abs(claimed_edge - actual_edge):.1f} percentage points
""")

if 0.90 <= 0.95 < 1.00:  # Check 90-100c bucket
    r = results.get("90-100c", {})
    claimed_edge = 90.3
    actual_edge = r.get('edge', 0) * 100
    print(f"""
VERIFICATION RESULT FOR NO at 90-100c:
  Claimed Edge: +{claimed_edge:.1f}%
  Verified Edge: {actual_edge:+.1f}%
  Discrepancy: {abs(claimed_edge - actual_edge):.1f} percentage points
""")

# Save results
output = {
    'data_stats': {
        'total_trades': int(len(trades_df)),
        'unique_markets': int(trades_df['market_ticker'].nunique()),
        'markets_with_outcomes': int(len(last_trades_with_outcome))
    },
    'verified_strategies': results
}

output_file = Path("/Users/samuelclark/Desktop/kalshiflow/research/reports/session005_verification.json")
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to: {output_file}")
