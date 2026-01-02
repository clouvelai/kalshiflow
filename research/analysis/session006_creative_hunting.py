#!/usr/bin/env python3
"""
Session 006: Creative Pattern Hunting

Mission: Find strategies quant firms won't touch.

Key Asset: For every trade, we know the ACTUAL outcome (is_winner, market_result)
This means we can calculate REAL P&L for any strategy.

Constraints (what we CAN'T compete on):
- Speed (HFT, latency arbitrage)
- Capital (market making)
- Data (alternative data)
- Sophistication (complex derivatives)

Where we MIGHT have edge:
- Small/illiquid markets (too small for funds)
- Behavioral patterns (human biases)
- Timing patterns (when retail makes mistakes)
- Market microstructure (Kalshi-specific patterns)
- Contrarian opportunities (crowd psychology)

HYPOTHESES TO TEST:
H031: Time-of-day patterns (morning vs evening vs overnight)
H032: Day-of-week patterns (weekend vs weekday behavior)
H033: Trade clustering (streaks of same-direction trades)
H034: Small vs large trade asymmetry (retail vs whale behavior)
H035: First/last trade of market (market open/close dynamics)
H036: Price distance from 50c (how far from uncertainty)
H037: Rapid price movement detection (momentum/reversal)
H038: Volume anomalies (unusual volume predicts outcome)
H039: Cross-market category effects (sports vs crypto vs weather)
H040: Market age/lifecycle (young vs mature markets)
H041: Time-to-expiry effects (final hours before settlement)
H042: Consecutive trade direction (streaks)
H043: Bid-ask spread patterns (spread as signal)
H044: Trade frequency patterns (busy vs quiet markets)
H045: Round number clustering (prices at 25c, 50c, 75c)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import re
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Paths
DATA_DIR = Path("/Users/samuelclark/Desktop/kalshiflow/research/data")
TRADES_FILE = DATA_DIR / "trades" / "enriched_trades_resolved_ALL.csv"
REPORTS_DIR = DATA_DIR.parent / "reports"

print("=" * 80)
print("SESSION 006: CREATIVE PATTERN HUNTING")
print("Finding Strategies Quant Firms Won't Touch")
print("=" * 80)
print()

# Load data
print("Loading data...")
df = pd.read_csv(TRADES_FILE)
if df['is_winner'].dtype == 'object':
    df['is_winner'] = df['is_winner'].map({'True': True, 'False': False, True: True, False: False})

df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day_name'] = df['datetime'].dt.day_name()
df['date'] = df['datetime'].dt.date

# Extract base market (remove suffix like -YES, -NO, or contract ID)
def extract_base_market(ticker):
    parts = ticker.rsplit('-', 1)
    if len(parts) == 2:
        if re.match(r'^[A-Z0-9]{1,10}$', parts[1]) and len(parts[1]) <= 10:
            return parts[0]
    return ticker

def extract_category(ticker):
    """Extract category prefix from ticker."""
    # Common patterns: KXBTCD, KXNFLGAME, KXNCAAFGAME, etc.
    if ticker.startswith('KX'):
        parts = ticker[2:].split('-')
        if parts:
            return 'KX' + parts[0][:10]  # First 10 chars of category
    return ticker.split('-')[0][:15]

df['base_market'] = df['market_ticker'].apply(extract_base_market)
df['category'] = df['market_ticker'].apply(extract_category)

print(f"Data Loaded:")
print(f"  Total trades: {len(df):,}")
print(f"  Unique markets: {df['base_market'].nunique():,}")
print(f"  Categories: {df['category'].nunique()}")
print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"  Total volume: ${df['cost_dollars'].sum():,.0f}")
print()

# =============================================================================
# HELPER: CORRECT STRATEGY VALIDATION
# =============================================================================

def validate_strategy(df, mask, description, min_markets=50, max_concentration=0.30):
    """
    Validate a strategy with CORRECT methodology.

    Returns dict with:
    - edge: win_rate - breakeven
    - markets: number of unique markets
    - profit: total actual profit
    - concentration: max profit % from single market
    - p_value: statistical significance
    """
    subset = df[mask].copy()
    if len(subset) == 0:
        return {'valid': False, 'reason': 'No trades match filter'}

    # Market-level aggregation
    market_stats = subset.groupby('base_market').agg({
        'is_winner': 'first',  # All trades in market have same outcome
        'trade_price': 'mean',
        'actual_profit_dollars': 'sum',
        'cost_dollars': 'sum',
        'count': 'sum'
    }).reset_index()

    n_markets = len(market_stats)
    if n_markets < min_markets:
        return {'valid': False, 'reason': f'Only {n_markets} markets (need {min_markets})'}

    wins = market_stats['is_winner'].sum()
    win_rate = wins / n_markets

    # CORRECT breakeven calculation
    avg_trade_price = market_stats['trade_price'].mean()
    breakeven = avg_trade_price / 100.0  # What you PAY determines breakeven

    edge = win_rate - breakeven

    # Total profit
    total_profit = market_stats['actual_profit_dollars'].sum()
    total_cost = market_stats['cost_dollars'].sum()
    roi = total_profit / total_cost if total_cost > 0 else 0

    # Concentration check
    if total_profit > 0:
        max_single_market_profit = market_stats['actual_profit_dollars'].max()
        concentration = max_single_market_profit / total_profit
    else:
        concentration = 1.0  # If no profit, concentration is 100%

    # Statistical significance (binomial test)
    if edge > 0:
        p_value = stats.binomtest(wins, n_markets, breakeven, alternative='greater').pvalue
    else:
        p_value = 1.0

    is_valid = (
        n_markets >= min_markets and
        concentration <= max_concentration and
        p_value < 0.05 and
        edge > 0.01  # At least 1% edge
    )

    return {
        'valid': is_valid,
        'description': description,
        'markets': n_markets,
        'wins': wins,
        'win_rate': round(win_rate, 4),
        'avg_price': round(avg_trade_price, 1),
        'breakeven': round(breakeven, 4),
        'edge': round(edge, 4),
        'total_profit': round(total_profit, 2),
        'total_cost': round(total_cost, 2),
        'roi': round(roi, 4),
        'concentration': round(concentration, 4),
        'p_value': round(p_value, 6),
        'trades': len(subset)
    }


def print_result(result):
    """Pretty print a strategy result."""
    if not result.get('valid', False):
        print(f"  INVALID: {result.get('reason', result.get('description', 'Unknown'))}")
        if result.get('markets'):
            print(f"    Markets: {result['markets']}, Edge: {result.get('edge', 0)*100:+.1f}%, P: {result.get('p_value', 1):.4f}")
        return

    print(f"  {result['description']}")
    print(f"    Markets: {result['markets']:,} | Win Rate: {result['win_rate']*100:.1f}% | Breakeven: {result['breakeven']*100:.1f}%")
    print(f"    Edge: {result['edge']*100:+.2f}% | ROI: {result['roi']*100:+.2f}% | P-value: {result['p_value']:.6f}")
    print(f"    Profit: ${result['total_profit']:,.0f} | Concentration: {result['concentration']*100:.1f}%")


# =============================================================================
# HYPOTHESIS 031: TIME-OF-DAY PATTERNS
# =============================================================================
print("\n" + "=" * 80)
print("H031: TIME-OF-DAY PATTERNS")
print("When do retail traders make the most mistakes?")
print("=" * 80)

# Group trades by hour
hourly_results = {}
for hour in range(24):
    for side in ['yes', 'no']:
        mask = (df['hour'] == hour) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} at hour {hour:02d}")
        hourly_results[(hour, side)] = result

# Find best hours
print("\nBest time-of-day patterns (edge > 2%, valid):")
valid_hourly = [(k, v) for k, v in hourly_results.items() if v.get('valid') and v.get('edge', 0) > 0.02]
valid_hourly.sort(key=lambda x: x[1]['edge'], reverse=True)
for (hour, side), result in valid_hourly[:5]:
    print_result(result)

# =============================================================================
# HYPOTHESIS 032: DAY-OF-WEEK PATTERNS
# =============================================================================
print("\n" + "=" * 80)
print("H032: DAY-OF-WEEK PATTERNS")
print("Do weekends behave differently?")
print("=" * 80)

day_results = {}
for day in range(7):
    for side in ['yes', 'no']:
        mask = (df['day_of_week'] == day) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day]}")
        day_results[(day, side)] = result
        if result.get('valid'):
            print_result(result)

# =============================================================================
# HYPOTHESIS 033: TRADE SIZE PATTERNS (RETAIL VS WHALE)
# =============================================================================
print("\n" + "=" * 80)
print("H033: TRADE SIZE PATTERNS")
print("Do small traders (retail) make different mistakes than large traders (whales)?")
print("=" * 80)

# Define size buckets
size_buckets = [
    (1, 10, 'Micro (1-10)'),
    (11, 50, 'Small (11-50)'),
    (51, 200, 'Medium (51-200)'),
    (201, 1000, 'Large (201-1000)'),
    (1001, float('inf'), 'Whale (1000+)')
]

size_results = {}
for low, high, name in size_buckets:
    for side in ['yes', 'no']:
        mask = (df['count'] >= low) & (df['count'] < high) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} trades, size {name}")
        size_results[(name, side)] = result

print("\nSize-based patterns:")
for (name, side), result in sorted(size_results.items(), key=lambda x: x[1].get('edge', -1), reverse=True):
    if result.get('markets', 0) >= 50:
        print_result(result)

# =============================================================================
# HYPOTHESIS 034: FIRST TRADE VS LAST TRADE OF MARKET
# =============================================================================
print("\n" + "=" * 80)
print("H034: FIRST/LAST TRADE SIGNALS")
print("Does the first or last trade predict the outcome?")
print("=" * 80)

# Add first/last trade flags
df['trade_order'] = df.groupby('base_market').cumcount()
df['trade_total'] = df.groupby('base_market')['id'].transform('count')
df['is_first_trade'] = df['trade_order'] == 0
df['is_last_trade'] = df['trade_order'] == df['trade_total'] - 1
df['is_first_5'] = df['trade_order'] < 5
df['is_last_5'] = df['trade_order'] >= df['trade_total'] - 5

for position, label in [('is_first_trade', 'First trade'), ('is_last_trade', 'Last trade'),
                         ('is_first_5', 'First 5 trades'), ('is_last_5', 'Last 5 trades')]:
    for side in ['yes', 'no']:
        mask = df[position] & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} in {label}")
        if result.get('markets', 0) >= 50:
            print_result(result)

# =============================================================================
# HYPOTHESIS 035: PRICE DISTANCE FROM 50c
# =============================================================================
print("\n" + "=" * 80)
print("H035: PRICE DISTANCE FROM 50c")
print("How far from uncertainty does the edge change?")
print("=" * 80)

df['dist_from_50'] = abs(df['yes_price'] - 50)

dist_results = {}
for dist_low, dist_high, label in [(0, 10, 'Near 50c (0-10)'),
                                    (10, 20, 'Moderate (10-20)'),
                                    (20, 30, 'Far (20-30)'),
                                    (30, 40, 'Very Far (30-40)'),
                                    (40, 50, 'Extreme (40-50)')]:
    for side in ['yes', 'no']:
        mask = (df['dist_from_50'] >= dist_low) & (df['dist_from_50'] < dist_high) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} when distance from 50c is {label}")
        dist_results[(label, side)] = result
        if result.get('markets', 0) >= 50:
            print_result(result)

# =============================================================================
# HYPOTHESIS 036: CATEGORY-SPECIFIC PATTERNS
# =============================================================================
print("\n" + "=" * 80)
print("H036: CATEGORY-SPECIFIC INEFFICIENCIES")
print("Which market categories are least efficient?")
print("=" * 80)

# Get top categories by volume
category_volume = df.groupby('category')['cost_dollars'].sum().sort_values(ascending=False)
top_categories = category_volume.head(20).index.tolist()

print(f"\nTop 20 categories by volume:")
cat_results = {}
for cat in top_categories:
    for side in ['yes', 'no']:
        mask = (df['category'] == cat) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} in {cat}")
        cat_results[(cat, side)] = result

# Print valid strategies by edge
print("\nCategory strategies with positive edge (sorted by edge):")
valid_cats = [(k, v) for k, v in cat_results.items() if v.get('valid') and v.get('edge', 0) > 0]
valid_cats.sort(key=lambda x: x[1]['edge'], reverse=True)
for (cat, side), result in valid_cats[:10]:
    print_result(result)

# =============================================================================
# HYPOTHESIS 037: TRADE CLUSTERING (STREAKS)
# =============================================================================
print("\n" + "=" * 80)
print("H037: TRADE CLUSTERING/STREAKS")
print("Does a streak of same-direction trades predict outcomes?")
print("=" * 80)

# Calculate streaks within each market
def calculate_streaks(group):
    group = group.sort_values('datetime')
    sides = group['taker_side'].values
    streaks = np.ones(len(sides), dtype=int)
    for i in range(1, len(sides)):
        if sides[i] == sides[i-1]:
            streaks[i] = streaks[i-1] + 1
        else:
            streaks[i] = 1
    return pd.Series(streaks, index=group.index)

print("Calculating trade streaks (this may take a moment)...")
df['streak'] = df.groupby('base_market', group_keys=False).apply(calculate_streaks)

streak_results = {}
for streak_len in [3, 5, 7, 10]:
    for side in ['yes', 'no']:
        # After a streak of YES trades, bet YES (momentum) or NO (reversal)?
        mask = (df['streak'] >= streak_len) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"After {streak_len}+ {side.upper()} streak, bet {side.upper()}")
        streak_results[(streak_len, side)] = result
        if result.get('markets', 0) >= 50 and result.get('edge', 0) > 0:
            print_result(result)

# =============================================================================
# HYPOTHESIS 038: VOLUME ANOMALIES
# =============================================================================
print("\n" + "=" * 80)
print("H038: VOLUME ANOMALIES")
print("Do markets with unusual volume have edge?")
print("=" * 80)

# Calculate market-level volume stats
market_volume = df.groupby('base_market')['cost_dollars'].sum()
vol_25 = market_volume.quantile(0.25)
vol_50 = market_volume.quantile(0.50)
vol_75 = market_volume.quantile(0.75)
vol_90 = market_volume.quantile(0.90)

df['market_volume'] = df['base_market'].map(market_volume)
df['is_low_volume'] = df['market_volume'] < vol_25
df['is_medium_volume'] = (df['market_volume'] >= vol_25) & (df['market_volume'] < vol_75)
df['is_high_volume'] = df['market_volume'] >= vol_75
df['is_very_high_volume'] = df['market_volume'] >= vol_90

volume_results = {}
for vol_flag, label in [('is_low_volume', 'Low Vol (<25%)'),
                         ('is_medium_volume', 'Medium Vol (25-75%)'),
                         ('is_high_volume', 'High Vol (>75%)'),
                         ('is_very_high_volume', 'Very High Vol (>90%)')]:
    for side in ['yes', 'no']:
        mask = df[vol_flag] & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} in {label} markets")
        volume_results[(label, side)] = result
        if result.get('markets', 0) >= 50:
            print_result(result)

# =============================================================================
# HYPOTHESIS 039: LEVERAGE PATTERNS
# =============================================================================
print("\n" + "=" * 80)
print("H039: LEVERAGE PATTERNS")
print("Do high-leverage trades (longshots) have different edges?")
print("=" * 80)

lev_results = {}
for lev_low, lev_high, label in [(0, 1, 'Low Leverage (0-1x)'),
                                  (1, 2, 'Medium Leverage (1-2x)'),
                                  (2, 5, 'High Leverage (2-5x)'),
                                  (5, 20, 'Very High Leverage (5-20x)'),
                                  (20, float('inf'), 'Extreme Leverage (20x+)')]:
    for side in ['yes', 'no']:
        mask = (df['leverage_ratio'] >= lev_low) & (df['leverage_ratio'] < lev_high) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} at {label}")
        lev_results[(label, side)] = result
        if result.get('markets', 0) >= 50:
            print_result(result)

# =============================================================================
# HYPOTHESIS 040: CONTRARIAN VS MOMENTUM (CROSS-TRADE PATTERNS)
# =============================================================================
print("\n" + "=" * 80)
print("H040: CONTRARIAN VS MOMENTUM")
print("If previous trade was YES, should next trade be YES or NO?")
print("=" * 80)

# Calculate previous trade direction
df_sorted = df.sort_values(['base_market', 'datetime'])
df_sorted['prev_side'] = df_sorted.groupby('base_market')['taker_side'].shift(1)
df_sorted['is_contrarian'] = df_sorted['taker_side'] != df_sorted['prev_side']

for is_contrarian, label in [(True, 'Contrarian (opposite prev)'), (False, 'Momentum (same as prev)')]:
    for side in ['yes', 'no']:
        mask = (df_sorted['is_contrarian'] == is_contrarian) & (df_sorted['taker_side'] == side) & df_sorted['prev_side'].notna()
        result = validate_strategy(df_sorted, mask, f"{side.upper()} {label}")
        if result.get('markets', 0) >= 50:
            print_result(result)

# =============================================================================
# HYPOTHESIS 041: TIME UNTIL SETTLEMENT
# =============================================================================
print("\n" + "=" * 80)
print("H041: TIME PATTERNS WITHIN TRADING DAY")
print("Are there patterns in early vs late trading hours?")
print("=" * 80)

# Group by trading session
df['is_early_morning'] = df['hour'] < 9
df['is_morning'] = (df['hour'] >= 9) & (df['hour'] < 12)
df['is_afternoon'] = (df['hour'] >= 12) & (df['hour'] < 17)
df['is_evening'] = (df['hour'] >= 17) & (df['hour'] < 21)
df['is_night'] = df['hour'] >= 21

session_results = {}
for session, label in [('is_early_morning', 'Early Morning (before 9am)'),
                        ('is_morning', 'Morning (9am-12pm)'),
                        ('is_afternoon', 'Afternoon (12pm-5pm)'),
                        ('is_evening', 'Evening (5pm-9pm)'),
                        ('is_night', 'Night (after 9pm)')]:
    for side in ['yes', 'no']:
        mask = df[session] & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} during {label}")
        session_results[(label, side)] = result
        if result.get('markets', 0) >= 50:
            print_result(result)

# =============================================================================
# HYPOTHESIS 042: ROUND NUMBER EFFECTS
# =============================================================================
print("\n" + "=" * 80)
print("H042: ROUND NUMBER EFFECTS")
print("Are prices at round numbers (25c, 50c, 75c) mispriced?")
print("=" * 80)

round_results = {}
for round_price in [10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90]:
    for side in ['yes', 'no']:
        # Within 2c of round number
        mask = (abs(df['yes_price'] - round_price) <= 2) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} near {round_price}c")
        round_results[(round_price, side)] = result

# Show interesting round number effects
print("\nRound number effects with edge > 2%:")
for (price, side), result in sorted(round_results.items(), key=lambda x: x[1].get('edge', -1), reverse=True):
    if result.get('valid') and result.get('edge', 0) > 0.02:
        print_result(result)

# =============================================================================
# HYPOTHESIS 043: TRADE COUNT PATTERNS
# =============================================================================
print("\n" + "=" * 80)
print("H043: TRADE COUNT IN MARKET")
print("Do markets with few trades behave differently?")
print("=" * 80)

# Count trades per market
market_trade_count = df.groupby('base_market').size()
df['market_trade_count'] = df['base_market'].map(market_trade_count)

tc_results = {}
for tc_low, tc_high, label in [(1, 5, 'Very Low (1-5 trades)'),
                                (5, 20, 'Low (5-20 trades)'),
                                (20, 100, 'Medium (20-100 trades)'),
                                (100, 500, 'High (100-500 trades)'),
                                (500, float('inf'), 'Very High (500+ trades)')]:
    for side in ['yes', 'no']:
        mask = (df['market_trade_count'] >= tc_low) & (df['market_trade_count'] < tc_high) & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} in {label} markets")
        tc_results[(label, side)] = result
        if result.get('markets', 0) >= 50:
            print_result(result)

# =============================================================================
# HYPOTHESIS 044: DOLLAR AMOUNT PATTERNS
# =============================================================================
print("\n" + "=" * 80)
print("H044: DOLLAR AMOUNT PATTERNS")
print("Do specific dollar amounts suggest retail vs institutional?")
print("=" * 80)

# Common retail amounts
df['is_round_dollar'] = (df['cost_dollars'] % 10 == 0) & (df['cost_dollars'] <= 100)
df['is_tiny_trade'] = df['cost_dollars'] < 5
df['is_big_trade'] = df['cost_dollars'] > 1000

dollar_results = {}
for flag, label in [('is_round_dollar', 'Round dollar amount ($10, $20, etc)'),
                     ('is_tiny_trade', 'Tiny trade (<$5)'),
                     ('is_big_trade', 'Big trade (>$1000)')]:
    for side in ['yes', 'no']:
        mask = df[flag] & (df['taker_side'] == side)
        result = validate_strategy(df, mask, f"{side.upper()} on {label}")
        dollar_results[(label, side)] = result
        if result.get('markets', 0) >= 50:
            print_result(result)

# =============================================================================
# HYPOTHESIS 045: COMBINED SIGNALS
# =============================================================================
print("\n" + "=" * 80)
print("H045: COMBINED SIGNALS")
print("Can we combine multiple weak signals for stronger edge?")
print("=" * 80)

# Try combinations
combos = [
    # Low volume + high leverage (retail longshots in illiquid markets)
    (df['is_low_volume'] & (df['leverage_ratio'] > 5) & (df['taker_side'] == 'no'),
     'NO on high-lev low-vol'),

    # Evening + round price + YES (retail betting favorites at night)
    (df['is_evening'] & (abs(df['yes_price'] - 75) <= 5) & (df['taker_side'] == 'no'),
     'NO at ~75c in evening'),

    # First trade + high leverage (opening longshot bets)
    (df['is_first_trade'] & (df['leverage_ratio'] > 5) & (df['taker_side'] == 'no'),
     'NO on first-trade longshots'),

    # Low trade count + extreme price
    ((df['market_trade_count'] < 20) & (df['yes_price'] < 20) & (df['taker_side'] == 'no'),
     'NO in illiquid + cheap YES'),

    # Tiny retail trades against favorites
    (df['is_tiny_trade'] & (df['yes_price'] > 70) & (df['taker_side'] == 'no'),
     'NO on tiny trades at high YES'),
]

for mask, label in combos:
    result = validate_strategy(df, mask, label)
    if result.get('markets', 0) >= 30:  # Lower threshold for combos
        print_result(result)

# =============================================================================
# FINAL SUMMARY: ALL PROMISING STRATEGIES
# =============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY: POTENTIALLY VIABLE STRATEGIES")
print("=" * 80)

# Collect all results
all_results = {}
all_results.update({f"hourly_{k}": v for k, v in hourly_results.items()})
all_results.update({f"day_{k}": v for k, v in day_results.items()})
all_results.update({f"size_{k}": v for k, v in size_results.items()})
all_results.update({f"dist_{k}": v for k, v in dist_results.items()})
all_results.update({f"cat_{k}": v for k, v in cat_results.items()})
all_results.update({f"vol_{k}": v for k, v in volume_results.items()})
all_results.update({f"lev_{k}": v for k, v in lev_results.items()})
all_results.update({f"session_{k}": v for k, v in session_results.items()})
all_results.update({f"round_{k}": v for k, v in round_results.items()})
all_results.update({f"tc_{k}": v for k, v in tc_results.items()})

# Filter to valid strategies with real edge
valid_strategies = {k: v for k, v in all_results.items()
                    if v.get('valid') and v.get('edge', 0) > 0.02}  # >2% edge

print(f"\nFound {len(valid_strategies)} strategies with >2% edge that pass validation:")
print("-" * 80)

for name, result in sorted(valid_strategies.items(), key=lambda x: x[1]['edge'], reverse=True):
    print_result(result)
    print()

# =============================================================================
# DEEP DIVE: ANY REAL EDGE?
# =============================================================================
print("\n" + "=" * 80)
print("REALITY CHECK: IS THERE ANY REAL EDGE?")
print("=" * 80)

# Check if ANY strategy beats random
print("\nBaseline: Random betting should have ~0% edge")
print("\nStrategies sorted by statistical significance:")

significant = [(k, v) for k, v in all_results.items()
               if v.get('p_value', 1) < 0.01 and v.get('edge', 0) > 0]
significant.sort(key=lambda x: x[1]['p_value'])

for name, result in significant[:10]:
    print(f"  {result['description']}: Edge={result['edge']*100:+.2f}%, P={result['p_value']:.6f}, N={result['markets']}")

# Save results
output = {
    'session': '006',
    'timestamp': datetime.now().isoformat(),
    'summary': {
        'total_trades': len(df),
        'total_markets': df['base_market'].nunique(),
        'strategies_tested': len(all_results),
        'strategies_valid': len(valid_strategies)
    },
    'valid_strategies': valid_strategies,
    'all_results': {str(k): v for k, v in all_results.items()}
}

output_file = REPORTS_DIR / "session006_creative_hunting.json"
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to: {output_file}")
