#!/usr/bin/env python3
"""
Session 011: Deep Validation of H087_follow (Round Size Bot NO Consensus)

The initial test showed +30.7% edge with +30.4% improvement over baseline.
This is suspiciously high. We need to verify:
1. Methodology is correct
2. Not a data processing error
3. Temporal stability
4. Concentration check in detail
5. Price proxy check with correct matching
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEEP VALIDATION: H087 Follow Round Size Bot NO Consensus")
print("="*80)

# Load data
print("\nLoading trade data...")
df = pd.read_csv('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv')
print(f"Loaded {len(df):,} trades")

# Define round sizes
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]
df['is_round_size'] = df['count'].isin(ROUND_SIZES)

# ==============================================================================
# STEP 1: Recreate the signal
# ==============================================================================
print("\n" + "-"*40)
print("STEP 1: Recreate Signal")
print("-"*40)

round_size_trades = df[df['is_round_size']]
print(f"Round size trades: {len(round_size_trades):,}")

# Calculate consensus of round-size trades per market
round_consensus = round_size_trades.groupby('market_ticker').agg({
    'taker_side': lambda x: (x == 'yes').mean(),
    'trade_price': 'mean',
    'market_result': 'first',
    'count': 'sum'
}).reset_index()
round_consensus.columns = ['market_ticker', 'yes_ratio', 'avg_price', 'market_result', 'total_count']

# Signal: >60% round-size trades are NO (i.e., yes_ratio < 0.4)
signal_markets = round_consensus[round_consensus['yes_ratio'] < 0.4]['market_ticker'].tolist()
print(f"Markets with >60% round-size NO consensus: {len(signal_markets)}")

# Get all trades from signal markets
signal_df = round_consensus[round_consensus['yes_ratio'] < 0.4].copy()
signal_df['won'] = (signal_df['market_result'] == 'no').astype(int)

print(f"\nSignal market stats:")
print(f"  Markets: {len(signal_df)}")
print(f"  Win rate: {signal_df['won'].mean():.2%}")
print(f"  Avg YES price: {signal_df['avg_price'].mean():.1f}c")
print(f"  Avg NO price: {100 - signal_df['avg_price'].mean():.1f}c")

# ==============================================================================
# STEP 2: Verify edge calculation
# ==============================================================================
print("\n" + "-"*40)
print("STEP 2: Verify Edge Calculation")
print("-"*40)

# For NO bets:
# Cost = NO price = 100 - YES price
# Win = market settles NO
# Breakeven = NO price / 100

win_rate = signal_df['won'].mean()
avg_no_price = 100 - signal_df['avg_price'].mean()
breakeven = avg_no_price / 100

print(f"Win rate: {win_rate:.2%}")
print(f"Avg NO price: {avg_no_price:.1f}c")
print(f"Breakeven: {breakeven:.2%}")
print(f"Edge: {(win_rate - breakeven) * 100:.1f}%")

# Calculate profit per market
signal_df['no_price'] = 100 - signal_df['avg_price']
signal_df['profit_if_win'] = 100 - signal_df['no_price']  # win payout
signal_df['loss_if_lose'] = -signal_df['no_price']  # lose cost
signal_df['profit'] = np.where(signal_df['won'] == 1, signal_df['profit_if_win'], signal_df['loss_if_lose'])

print(f"\nProfit analysis (per contract):")
print(f"  Total profit: ${signal_df['profit'].sum() * signal_df['total_count'].mean():.0f}")
print(f"  Avg profit per market: ${signal_df['profit'].mean() * signal_df['total_count'].mean():.2f}")
print(f"  Win avg profit: ${signal_df[signal_df['won']==1]['profit_if_win'].mean():.2f}")
print(f"  Loss avg loss: ${-signal_df[signal_df['won']==0]['no_price'].mean():.2f}")

# ==============================================================================
# STEP 3: Price proxy check - correct methodology
# ==============================================================================
print("\n" + "-"*40)
print("STEP 3: Price Proxy Check")
print("-"*40)

# The key question: does the round-size NO consensus signal provide
# information BEYOND just the price level?

# Baseline: ALL markets at similar NO prices (not just round-size)
all_market_agg = df.groupby('market_ticker').agg({
    'trade_price': 'mean',
    'market_result': 'first'
}).reset_index()
all_market_agg['no_price'] = 100 - all_market_agg['trade_price']
all_market_agg['won'] = (all_market_agg['market_result'] == 'no').astype(int)

# Match on price range - use ACTUAL signal NO price range with tighter bounds
signal_no_prices = 100 - signal_df['avg_price']
# Use interquartile range to avoid outliers
q25, q75 = signal_no_prices.quantile([0.25, 0.75])
no_price_min = q25 - 5
no_price_max = q75 + 5

print(f"Signal NO price range (IQR): {q25:.1f}c - {q75:.1f}c")
print(f"Baseline matching range: {no_price_min:.1f}c - {no_price_max:.1f}c")

baseline_df = all_market_agg[
    (all_market_agg['no_price'] >= no_price_min) &
    (all_market_agg['no_price'] <= no_price_max)
]

print(f"Baseline markets at similar prices: {len(baseline_df)}")

# Calculate baseline edge
baseline_win_rate = baseline_df['won'].mean()
baseline_avg_no = baseline_df['no_price'].mean()
baseline_breakeven = baseline_avg_no / 100
baseline_edge = (baseline_win_rate - baseline_breakeven) * 100

print(f"\nBaseline at similar prices:")
print(f"  Markets: {len(baseline_df)}")
print(f"  Win rate: {baseline_win_rate:.2%}")
print(f"  Avg NO price: {baseline_avg_no:.1f}c")
print(f"  Breakeven: {baseline_breakeven:.2%}")
print(f"  Edge: {baseline_edge:.1f}%")

improvement = (win_rate - breakeven) * 100 - baseline_edge
print(f"\nImprovement over baseline: {improvement:.1f}%")

# ==============================================================================
# STEP 4: WHY such a huge improvement?
# ==============================================================================
print("\n" + "-"*40)
print("STEP 4: Investigate Why Such Large Improvement")
print("-"*40)

# Check if signal markets have different price distribution than baseline
print(f"\nPrice distribution comparison:")
print(f"Signal avg NO price: {(100 - signal_df['avg_price']).mean():.1f}c (std: {(100 - signal_df['avg_price']).std():.1f}c)")
print(f"Baseline avg NO price: {baseline_df['no_price'].mean():.1f}c (std: {baseline_df['no_price'].std():.1f}c)")

# Check market categories
signal_tickers = signal_df['market_ticker'].tolist()
signal_trades_df = df[df['market_ticker'].isin(signal_tickers)]

# Extract category from ticker
def extract_category(ticker):
    parts = ticker.split('-')
    if len(parts) > 0:
        # Remove date portion
        cat = parts[0]
        return cat
    return ticker

signal_df['category'] = signal_df['market_ticker'].apply(extract_category)
baseline_df['category'] = baseline_df['market_ticker'].apply(extract_category)

print("\nTop categories in SIGNAL markets:")
print(signal_df['category'].value_counts().head(10))

print("\nTop categories in BASELINE markets:")
print(baseline_df['category'].value_counts().head(10))

# ==============================================================================
# STEP 5: Temporal Stability
# ==============================================================================
print("\n" + "-"*40)
print("STEP 5: Temporal Stability")
print("-"*40)

# Add date to signal markets
signal_trades_df = df[df['market_ticker'].isin(signal_tickers)].copy()
signal_trades_df['date'] = pd.to_datetime(signal_trades_df['timestamp'], unit='ms').dt.date

# Get first trade date per market
market_dates = signal_trades_df.groupby('market_ticker')['date'].min().reset_index()
market_dates.columns = ['market_ticker', 'first_trade_date']

signal_with_date = signal_df.merge(market_dates, on='market_ticker')

# Split into first half and second half by date
# Sort by date and split in half
signal_with_date = signal_with_date.sort_values('first_trade_date')
split_idx = len(signal_with_date) // 2
median_date = signal_with_date.iloc[split_idx]['first_trade_date']
print(f"Median date: {median_date}")

first_half = signal_with_date[signal_with_date['first_trade_date'] < median_date]
second_half = signal_with_date[signal_with_date['first_trade_date'] >= median_date]

print(f"\nFirst half: {len(first_half)} markets, win rate: {first_half['won'].mean():.2%}")
print(f"Second half: {len(second_half)} markets, win rate: {second_half['won'].mean():.2%}")

# Edge by half
def calc_edge(df):
    win_rate = df['won'].mean()
    avg_no = 100 - df['avg_price'].mean()
    breakeven = avg_no / 100
    return (win_rate - breakeven) * 100

print(f"First half edge: {calc_edge(first_half):.1f}%")
print(f"Second half edge: {calc_edge(second_half):.1f}%")

# ==============================================================================
# STEP 6: Concentration Check
# ==============================================================================
print("\n" + "-"*40)
print("STEP 6: Concentration Check")
print("-"*40)

# Calculate profit per market more accurately
signal_df['profit_per_market'] = np.where(
    signal_df['won'] == 1,
    (100 - (100 - signal_df['avg_price'])) * signal_df['total_count'] / 100,  # Win payout
    -(100 - signal_df['avg_price']) * signal_df['total_count'] / 100  # Lose cost
)

total_profit = signal_df['profit_per_market'].abs().sum()
max_contribution = signal_df['profit_per_market'].abs().max()
concentration = max_contribution / total_profit if total_profit > 0 else 0

print(f"Total absolute profit: ${total_profit:.0f}")
print(f"Max single market contribution: ${max_contribution:.0f}")
print(f"Concentration: {concentration:.1%}")

# Top contributing markets
print("\nTop 10 markets by absolute profit:")
top_markets = signal_df.nlargest(10, 'profit_per_market')[['market_ticker', 'won', 'avg_price', 'total_count', 'profit_per_market']]
print(top_markets.to_string())

# ==============================================================================
# STEP 7: Statistical Significance
# ==============================================================================
print("\n" + "-"*40)
print("STEP 7: Statistical Significance")
print("-"*40)

n_markets = len(signal_df)
n_wins = signal_df['won'].sum()
expected_wins = int(breakeven * n_markets)

binom_result = stats.binomtest(n_wins, n_markets, breakeven, alternative='greater')
p_value = binom_result.pvalue

print(f"Markets: {n_markets}")
print(f"Wins: {n_wins}")
print(f"Expected wins at breakeven: {expected_wins}")
print(f"P-value: {p_value:.2e}")
print(f"Bonferroni threshold (0.05/5): {0.05/5}")
print(f"Passes Bonferroni: {p_value < 0.01}")

# ==============================================================================
# STEP 8: Verify signal makes sense
# ==============================================================================
print("\n" + "-"*40)
print("STEP 8: Signal Interpretation")
print("-"*40)

print("""
Signal: Markets where >60% of round-size trades are NO bets

Interpretation:
- Round sizes (10, 25, 50, 100, 250, 500, 1000) are likely bot/algorithmic trades
- When bots predominantly bet NO, they may have information
- We FOLLOW the bot NO consensus

The edge could be real if:
1. Bots are informed (have better models/data)
2. Bots are market makers providing liquidity (capturing spread on NO side)
3. Round-size NO bets correlate with some other signal

The edge could be spurious if:
1. Price distribution is different (we need exact matching)
2. Category selection bias (sports vs crypto etc)
3. Sample selection (markets with round trades are different)
""")

# ==============================================================================
# STEP 9: Stricter Price Matching
# ==============================================================================
print("\n" + "-"*40)
print("STEP 9: Stricter Price Matching")
print("-"*40)

# For each signal market, find baseline markets at EXACT same NO price (+/- 2c)
signal_df['no_price_bucket'] = ((100 - signal_df['avg_price']) // 5) * 5
all_market_agg['no_price_bucket'] = (all_market_agg['no_price'] // 5) * 5

# Calculate baseline edge per price bucket
baseline_by_bucket = all_market_agg.groupby('no_price_bucket').agg({
    'won': 'mean',
    'no_price': 'mean',
    'market_ticker': 'count'
}).reset_index()
baseline_by_bucket.columns = ['no_price_bucket', 'baseline_wr', 'baseline_no_price', 'baseline_n']
baseline_by_bucket['baseline_breakeven'] = baseline_by_bucket['baseline_no_price'] / 100
baseline_by_bucket['baseline_edge'] = (baseline_by_bucket['baseline_wr'] - baseline_by_bucket['baseline_breakeven']) * 100

# Join to signal
signal_with_baseline = signal_df.merge(baseline_by_bucket, on='no_price_bucket')

# Calculate improvement per market
signal_with_baseline['signal_edge'] = (signal_with_baseline['won'] - (100 - signal_with_baseline['avg_price']) / 100)
signal_with_baseline['improvement'] = signal_with_baseline['won'] - signal_with_baseline['baseline_wr']

# Aggregate
avg_improvement = signal_with_baseline['improvement'].mean() * 100
print(f"Average improvement over price-matched baseline: {avg_improvement:.1f}%")

# Weighted by inverse variance
print(f"\nBy price bucket:")
bucket_analysis = signal_with_baseline.groupby('no_price_bucket').agg({
    'won': ['mean', 'count'],
    'baseline_wr': 'first',
    'baseline_edge': 'first'
}).reset_index()
bucket_analysis.columns = ['bucket', 'signal_wr', 'signal_n', 'baseline_wr', 'baseline_edge']
bucket_analysis['signal_edge'] = (bucket_analysis['signal_wr'] - bucket_analysis['bucket']/100) * 100
bucket_analysis['improvement'] = bucket_analysis['signal_edge'] - bucket_analysis['baseline_edge']

print(bucket_analysis.to_string())

# Overall improvement weighted by sample size
weighted_improvement = (bucket_analysis['improvement'] * bucket_analysis['signal_n']).sum() / bucket_analysis['signal_n'].sum()
print(f"\nWeighted improvement over price-matched baseline: {weighted_improvement:.1f}%")

# ==============================================================================
# FINAL VALIDATION SUMMARY
# ==============================================================================
print("\n" + "="*80)
print("FINAL VALIDATION SUMMARY: H087_follow")
print("="*80)

validation_results = {
    'hypothesis': 'H087_follow',
    'description': 'Follow Round Size Bot NO Consensus (>60% NO)',
    'markets': len(signal_df),
    'win_rate': win_rate,
    'breakeven': breakeven,
    'edge_pct': (win_rate - breakeven) * 100,
    'p_value': p_value,
    'concentration': concentration,
    'improvement_vs_baseline': weighted_improvement,
    'temporal_stability': {
        'first_half_edge': calc_edge(first_half),
        'second_half_edge': calc_edge(second_half)
    },
    'validation_criteria': {
        'markets >= 50': len(signal_df) >= 50,
        'concentration < 30%': concentration < 0.30,
        'p_value < 0.01': p_value < 0.01,
        'improvement > 0': weighted_improvement > 0,
        'temporal_stability': calc_edge(first_half) > 0 and calc_edge(second_half) > 0
    }
}

passes_all = all(validation_results['validation_criteria'].values())
validation_results['is_validated'] = passes_all

print(f"\nMarkets: {validation_results['markets']}")
print(f"Win Rate: {validation_results['win_rate']:.2%}")
print(f"Breakeven: {validation_results['breakeven']:.2%}")
print(f"Edge: {validation_results['edge_pct']:.1f}%")
print(f"P-value: {validation_results['p_value']:.2e}")
print(f"Concentration: {validation_results['concentration']:.1%}")
print(f"Improvement vs baseline: {validation_results['improvement_vs_baseline']:.1f}%")
print(f"\nTemporal Stability:")
print(f"  First half edge: {validation_results['temporal_stability']['first_half_edge']:.1f}%")
print(f"  Second half edge: {validation_results['temporal_stability']['second_half_edge']:.1f}%")

print(f"\nValidation Criteria:")
for criterion, passed in validation_results['validation_criteria'].items():
    status = "PASS" if passed else "FAIL"
    print(f"  {criterion}: {status}")

print(f"\n{'='*40}")
print(f"FINAL VERDICT: {'VALIDATED' if passes_all else 'NOT VALIDATED'}")
print(f"{'='*40}")

# Save results
output_path = '/Users/samuelclark/Desktop/kalshiflow/research/reports/session011_h087_validation.json'
with open(output_path, 'w') as f:
    def convert_types(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        return obj

    json.dump(convert_types(validation_results), f, indent=2, default=str)
print(f"\nResults saved to: {output_path}")
