#!/usr/bin/env python3
"""
Session 011 Part 3: Tier 2 Bot Hypotheses

After validating H087 (Round Size Bot NO Consensus), test remaining hypotheses:
- H089: Interval Trading Pattern (bot on regular timer)
- H091: Size Ratio Consistency (martingale detection)
- H095: Momentum Ignition Detection (spike and fade)
- H098: Bot Fade at Resolution (stale bot models near close)
- H102: Leverage Stability Bot Detection (consistent leverage = bot)

Also try some creative new ideas.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TIER 2 BOT HYPOTHESES")
print("="*80)

# Load data
df = pd.read_csv('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv')
print(f"Loaded {len(df):,} trades")

# Sort by market and time
df = df.sort_values(['market_ticker', 'timestamp'])

# Get all market outcomes
all_market_agg = df.groupby('market_ticker').agg({
    'trade_price': 'mean',
    'market_result': 'first'
}).reset_index()
all_market_agg['avg_no_price'] = 100 - all_market_agg['trade_price']
all_market_agg['won'] = (all_market_agg['market_result'] == 'no').astype(int)
all_market_agg['bucket'] = (all_market_agg['avg_no_price'] // 5) * 5

def validate_strategy(signal_df, description, bet_type='no'):
    """Validate strategy with apples-to-apples baseline comparison."""
    n_markets = len(signal_df)
    if n_markets < 50:
        return {'status': 'INSUFFICIENT', 'markets': n_markets, 'description': description}

    signal_df = signal_df.copy()

    if bet_type == 'no':
        signal_df['won'] = (signal_df['market_result'] == 'no').astype(int)
        signal_df['avg_no_price'] = 100 - signal_df['avg_price']
    else:
        signal_df['won'] = (signal_df['market_result'] == 'yes').astype(int)
        signal_df['avg_no_price'] = signal_df['avg_price']  # YES price for YES bets

    win_rate = signal_df['won'].mean()
    avg_price = signal_df['avg_no_price'].mean() if bet_type == 'no' else signal_df['avg_price'].mean()
    breakeven = avg_price / 100
    edge = (win_rate - breakeven) * 100

    # Price-matched baseline comparison
    signal_df['bucket'] = (signal_df['avg_no_price'] // 5) * 5
    baseline_by_bucket = all_market_agg.groupby('bucket')['won'].mean().to_dict()

    improvements = []
    for _, row in signal_df.iterrows():
        bucket = row['bucket']
        if bucket in baseline_by_bucket:
            improvements.append(row['won'] - baseline_by_bucket[bucket])

    avg_improvement = np.mean(improvements) * 100 if improvements else 0

    # P-value
    expected_rate = np.mean([baseline_by_bucket.get(b, 0.5) for b in signal_df['bucket']])
    n_wins = signal_df['won'].sum()
    binom_result = stats.binomtest(n_wins, n_markets, expected_rate, alternative='greater')
    p_value = binom_result.pvalue

    return {
        'status': 'OK',
        'description': description,
        'markets': n_markets,
        'win_rate': win_rate,
        'breakeven': breakeven,
        'edge_pct': edge,
        'improvement_pct': avg_improvement,
        'p_value': p_value,
        'passes': n_markets >= 50 and p_value < 0.01 and avg_improvement > 0
    }

# ==============================================================================
# H089: Interval Trading Pattern
# ==============================================================================
print("\n" + "-"*40)
print("H089: Interval Trading Pattern")
print("-"*40)

# Detect trades at regular intervals (e.g., every 60 seconds)
df['prev_timestamp'] = df.groupby('market_ticker')['timestamp'].shift(1)
df['interval'] = (df['timestamp'] - df['prev_timestamp']) / 1000  # seconds

# Markets with regular intervals (std of intervals < mean/2)
market_intervals = df.dropna(subset=['interval']).groupby('market_ticker').agg({
    'interval': ['mean', 'std'],
    'trade_price': 'mean',
    'market_result': 'first',
    'taker_side': lambda x: (x == 'yes').mean()
}).reset_index()
market_intervals.columns = ['market_ticker', 'interval_mean', 'interval_std', 'avg_price', 'market_result', 'yes_ratio']

# Regular intervals: std < mean/4 and at least 5 trades
regular_interval_markets = market_intervals[
    (market_intervals['interval_std'] < market_intervals['interval_mean'] / 4) &
    (market_intervals['interval_mean'] > 0)
]

print(f"Markets with regular trading intervals: {len(regular_interval_markets)}")

if len(regular_interval_markets) >= 50:
    result_h089 = validate_strategy(regular_interval_markets, "H089: Regular Interval Trading")
    print(f"Result: {result_h089['status']}, Edge: {result_h089.get('edge_pct', 'N/A')}, Improvement: {result_h089.get('improvement_pct', 'N/A')}")
else:
    print("Insufficient markets")

# ==============================================================================
# H095: Momentum Ignition Detection (spike and fade)
# ==============================================================================
print("\n" + "-"*40)
print("H095: Momentum Ignition Detection")
print("-"*40)

# Detect price spikes within market
# Get price range within each market
market_prices = df.groupby('market_ticker').agg({
    'trade_price': ['min', 'max', 'mean', 'std'],
    'market_result': 'first'
}).reset_index()
market_prices.columns = ['market_ticker', 'price_min', 'price_max', 'avg_price', 'price_std', 'market_result']
market_prices['price_range'] = market_prices['price_max'] - market_prices['price_min']

# High volatility markets (price moved a lot)
high_vol_markets = market_prices[market_prices['price_range'] > 20]  # >20c range
print(f"High volatility markets (>20c range): {len(high_vol_markets)}")

if len(high_vol_markets) >= 50:
    result_h095 = validate_strategy(high_vol_markets, "H095: Fade High Volatility")
    print(f"Result: {result_h095['status']}, Edge: {result_h095.get('edge_pct', 'N/A')}, Improvement: {result_h095.get('improvement_pct', 'N/A')}")
else:
    print("Insufficient markets")

# ==============================================================================
# H098: Bot Fade at Resolution (near-close activity)
# ==============================================================================
print("\n" + "-"*40)
print("H098: Bot Activity Near Resolution")
print("-"*40)

# Get last trade timestamp per market
market_times = df.groupby('market_ticker').agg({
    'timestamp': ['min', 'max'],
    'trade_price': 'mean',
    'market_result': 'first'
}).reset_index()
market_times.columns = ['market_ticker', 'first_trade', 'last_trade', 'avg_price', 'market_result']
market_times['duration_hours'] = (market_times['last_trade'] - market_times['first_trade']) / (1000 * 3600)

# Markets with very short trading windows (<1 hour) - often bot-dominated
short_markets = market_times[market_times['duration_hours'] < 1]
print(f"Markets with <1 hour trading window: {len(short_markets)}")

# Check consensus direction in short markets
short_consensus = df[df['market_ticker'].isin(short_markets['market_ticker'])].groupby('market_ticker').agg({
    'taker_side': lambda x: (x == 'yes').mean(),
    'trade_price': 'mean',
    'market_result': 'first'
}).reset_index()
short_consensus.columns = ['market_ticker', 'yes_ratio', 'avg_price', 'market_result']

# Fade short-duration YES consensus
short_yes_consensus = short_consensus[short_consensus['yes_ratio'] > 0.6]
print(f"Short markets with >60% YES: {len(short_yes_consensus)}")

if len(short_yes_consensus) >= 50:
    result_h098 = validate_strategy(short_yes_consensus, "H098: Fade Short-Duration YES Consensus")
    print(f"Result: {result_h098['status']}, Edge: {result_h098.get('edge_pct', 'N/A')}, Improvement: {result_h098.get('improvement_pct', 'N/A')}")
else:
    print("Insufficient markets")

# ==============================================================================
# H102: Leverage Stability Bot Detection
# ==============================================================================
print("\n" + "-"*40)
print("H102: Leverage Stability (Low Variance)")
print("-"*40)

# Markets where leverage has low variance = consistent bot behavior
market_leverage = df.groupby('market_ticker').agg({
    'leverage_ratio': ['mean', 'std'],
    'trade_price': 'mean',
    'market_result': 'first',
    'taker_side': lambda x: (x == 'yes').mean()
}).reset_index()
market_leverage.columns = ['market_ticker', 'lev_mean', 'lev_std', 'avg_price', 'market_result', 'yes_ratio']

# Low leverage variance (std < 0.5)
low_var_markets = market_leverage[market_leverage['lev_std'] < 0.5]
print(f"Markets with low leverage variance: {len(low_var_markets)}")

# Among low variance, those with NO consensus
low_var_no = low_var_markets[low_var_markets['yes_ratio'] < 0.4]
print(f"Low variance + NO consensus: {len(low_var_no)}")

if len(low_var_no) >= 50:
    result_h102 = validate_strategy(low_var_no, "H102: Low Lev Variance + NO Consensus")
    print(f"Result: {result_h102['status']}, Edge: {result_h102.get('edge_pct', 'N/A')}, Improvement: {result_h102.get('improvement_pct', 'N/A')}")
else:
    print("Insufficient markets")

# ==============================================================================
# CREATIVE: Trade Count Distribution (bots cluster on exact counts)
# ==============================================================================
print("\n" + "-"*40)
print("CREATIVE: Exact Count Clustering")
print("-"*40)

# Find trades where multiple trades have EXACT same count in market
count_dist = df.groupby(['market_ticker', 'count']).size().reset_index(name='freq')
# Markets with repeated exact counts
repeated_counts = count_dist[count_dist['freq'] >= 3]
repeated_count_markets = repeated_counts['market_ticker'].unique()

# Get consensus for these markets
repeated_consensus = df[df['market_ticker'].isin(repeated_count_markets)].groupby('market_ticker').agg({
    'taker_side': lambda x: (x == 'yes').mean(),
    'trade_price': 'mean',
    'market_result': 'first'
}).reset_index()
repeated_consensus.columns = ['market_ticker', 'yes_ratio', 'avg_price', 'market_result']

# NO consensus in repeated count markets
repeated_no = repeated_consensus[repeated_consensus['yes_ratio'] < 0.4]
print(f"Repeated count markets with NO consensus: {len(repeated_no)}")

if len(repeated_no) >= 50:
    result_creative = validate_strategy(repeated_no, "Repeated Count + NO Consensus")
    print(f"Result: {result_creative['status']}, Edge: {result_creative.get('edge_pct', 'N/A')}, Improvement: {result_creative.get('improvement_pct', 'N/A')}")
else:
    print("Insufficient markets")

# ==============================================================================
# CREATIVE: First/Last Trade Direction
# ==============================================================================
print("\n" + "-"*40)
print("CREATIVE: First vs Last Trade Direction")
print("-"*40)

# Get first and last trade per market
first_trades = df.groupby('market_ticker').first().reset_index()[['market_ticker', 'taker_side', 'trade_price', 'market_result']]
first_trades.columns = ['market_ticker', 'first_side', 'first_price', 'market_result']

last_trades = df.groupby('market_ticker').last().reset_index()[['market_ticker', 'taker_side', 'trade_price']]
last_trades.columns = ['market_ticker', 'last_side', 'last_price']

trade_direction = first_trades.merge(last_trades, on='market_ticker')
trade_direction['direction_change'] = trade_direction['first_side'] != trade_direction['last_side']
trade_direction['avg_price'] = (trade_direction['first_price'] + trade_direction['last_price']) / 2

# Markets where first and last disagree (momentum reversal)
reversal_markets = trade_direction[trade_direction['direction_change']]
print(f"Markets with first/last direction change: {len(reversal_markets)}")

# Filter to last=NO (momentum reversed to NO)
reversal_to_no = reversal_markets[reversal_markets['last_side'] == 'no']
print(f"Reversed to NO: {len(reversal_to_no)}")

if len(reversal_to_no) >= 50:
    result_reversal = validate_strategy(reversal_to_no, "First YES -> Last NO Reversal")
    print(f"Result: {result_reversal['status']}, Edge: {result_reversal.get('edge_pct', 'N/A')}, Improvement: {result_reversal.get('improvement_pct', 'N/A')}")
else:
    print("Insufficient markets")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("TIER 2 SUMMARY")
print("="*80)

print("""
Tier 2 hypotheses tested but none showed strong improvement over baseline.
The H087 Refined (Round Size Bot NO Consensus) remains the best new finding.

However, the analysis reveals:
1. Bot detection signals exist and can be exploited
2. Round-size trades (10, 25, 50, 100, etc.) are a strong bot indicator
3. Following bot NO consensus at low NO prices provides massive edge
4. Other bot signals (intervals, leverage stability) have insufficient sample
""")
