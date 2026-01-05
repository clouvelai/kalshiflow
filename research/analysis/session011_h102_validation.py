#!/usr/bin/env python3
"""
Session 011: Deep Validation of H102 - Low Leverage Variance + NO Consensus

Initial finding:
- Edge: +55.9%
- Improvement: +21.6%
- Markets: 830

Need to verify this is robust.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEEP VALIDATION: H102 - Low Leverage Variance + NO Consensus")
print("="*80)

# Load data
df = pd.read_csv('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv')
print(f"Loaded {len(df):,} trades")

# Get market-level stats
market_leverage = df.groupby('market_ticker').agg({
    'leverage_ratio': ['mean', 'std', 'count'],
    'trade_price': 'mean',
    'market_result': 'first',
    'taker_side': lambda x: (x == 'yes').mean()
}).reset_index()
market_leverage.columns = ['market_ticker', 'lev_mean', 'lev_std', 'n_trades', 'avg_price', 'market_result', 'yes_ratio']
market_leverage['avg_no_price'] = 100 - market_leverage['avg_price']

print(f"Total markets: {len(market_leverage)}")

# Baseline
all_market_agg = df.groupby('market_ticker').agg({
    'trade_price': 'mean',
    'market_result': 'first'
}).reset_index()
all_market_agg['avg_no_price'] = 100 - all_market_agg['trade_price']
all_market_agg['won'] = (all_market_agg['market_result'] == 'no').astype(int)
all_market_agg['bucket'] = (all_market_agg['avg_no_price'] // 5) * 5
baseline_by_bucket = all_market_agg.groupby('bucket')['won'].mean().to_dict()

# ==============================================================================
# Signal: Low leverage variance + NO consensus
# ==============================================================================
print("\n" + "-"*40)
print("Signal Definition")
print("-"*40)

# Low variance = std < 0.5 (bots have consistent leverage)
# NO consensus = yes_ratio < 0.4 (>60% NO trades)
# Need at least 3 trades to measure variance meaningfully
signal_df = market_leverage[
    (market_leverage['lev_std'] < 0.5) &
    (market_leverage['yes_ratio'] < 0.4) &
    (market_leverage['n_trades'] >= 3)
].copy()

print(f"Signal markets: {len(signal_df)}")
print(f"Leverage std distribution:")
print(signal_df['lev_std'].describe())

# ==============================================================================
# Edge Calculation
# ==============================================================================
print("\n" + "-"*40)
print("Edge Calculation")
print("-"*40)

signal_df['won'] = (signal_df['market_result'] == 'no').astype(int)
win_rate = signal_df['won'].mean()
avg_no_price = signal_df['avg_no_price'].mean()
breakeven = avg_no_price / 100
edge = (win_rate - breakeven) * 100

print(f"Win rate: {win_rate:.2%}")
print(f"Avg NO price: {avg_no_price:.1f}c")
print(f"Breakeven: {breakeven:.2%}")
print(f"Edge: {edge:.1f}%")

# ==============================================================================
# Price Proxy Check
# ==============================================================================
print("\n" + "-"*40)
print("Price Proxy Check (Apples-to-Apples)")
print("-"*40)

signal_df['bucket'] = (signal_df['avg_no_price'] // 5) * 5

print(f"{'Bucket':<12} {'Signal WR':<12} {'Baseline WR':<12} {'Improvement':<12}")
print("-"*48)

improvements = []
for bucket in sorted(signal_df['bucket'].unique()):
    sig_bucket = signal_df[signal_df['bucket'] == bucket]
    if bucket in baseline_by_bucket:
        sig_wr = sig_bucket['won'].mean()
        base_wr = baseline_by_bucket[bucket]
        improvement = (sig_wr - base_wr) * 100
        print(f"{bucket}-{bucket+5}c    {sig_wr:.1%}         {base_wr:.1%}         {improvement:+.1f}%")
        improvements.extend([improvement] * len(sig_bucket))

weighted_improvement = np.mean(improvements)
print(f"\nWeighted improvement: {weighted_improvement:.1f}%")

# ==============================================================================
# Statistical Significance
# ==============================================================================
print("\n" + "-"*40)
print("Statistical Significance")
print("-"*40)

# Compare to expected based on baseline at same prices
expected_wr = np.mean([baseline_by_bucket.get(b, 0.5) for b in signal_df['bucket']])
n_wins = signal_df['won'].sum()
n_markets = len(signal_df)

binom_result = stats.binomtest(n_wins, n_markets, expected_wr, alternative='greater')
p_value = binom_result.pvalue

print(f"Expected WR (from baseline): {expected_wr:.2%}")
print(f"Actual WR: {win_rate:.2%}")
print(f"P-value: {p_value:.2e}")
print(f"Bonferroni threshold: 0.01")
print(f"Passes: {p_value < 0.01}")

# ==============================================================================
# Temporal Stability
# ==============================================================================
print("\n" + "-"*40)
print("Temporal Stability")
print("-"*40)

df_with_date = df[df['market_ticker'].isin(signal_df['market_ticker'])].copy()
df_with_date['date'] = pd.to_datetime(df_with_date['timestamp'], unit='ms').dt.date
market_dates = df_with_date.groupby('market_ticker')['date'].min().reset_index()
signal_with_date = signal_df.merge(market_dates, on='market_ticker').sort_values('date')

split_idx = len(signal_with_date) // 2
first_half = signal_with_date.iloc[:split_idx]
second_half = signal_with_date.iloc[split_idx:]

first_wr = first_half['won'].mean()
second_wr = second_half['won'].mean()
first_edge = (first_wr - first_half['avg_no_price'].mean()/100) * 100
second_edge = (second_wr - second_half['avg_no_price'].mean()/100) * 100

print(f"First half: {len(first_half)} markets, WR={first_wr:.1%}, Edge={first_edge:.1f}%")
print(f"Second half: {len(second_half)} markets, WR={second_wr:.1%}, Edge={second_edge:.1f}%")

# ==============================================================================
# Concentration
# ==============================================================================
print("\n" + "-"*40)
print("Concentration Check")
print("-"*40)

signal_df['profit'] = np.where(
    signal_df['won'] == 1,
    (100 - signal_df['avg_no_price']) * signal_df['n_trades'] / 100,
    -signal_df['avg_no_price'] * signal_df['n_trades'] / 100
)

profit_abs = signal_df['profit'].abs()
concentration = profit_abs.max() / profit_abs.sum() if profit_abs.sum() > 0 else 0
print(f"Concentration: {concentration:.1%}")

# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

passes_all = (
    len(signal_df) >= 50 and
    concentration < 0.30 and
    p_value < 0.01 and
    weighted_improvement > 0 and
    first_edge > 0 and second_edge > 0
)

print(f"""
Strategy: H102 - Low Leverage Variance + NO Consensus

Signal:
- Markets with leverage_ratio std < 0.5 (consistent leverage = bot-like)
- AND >60% NO consensus
- AND at least 3 trades

Validation:
  Markets >= 50:        {'PASS' if len(signal_df) >= 50 else 'FAIL'} ({len(signal_df)} markets)
  Concentration < 30%:  {'PASS' if concentration < 0.30 else 'FAIL'} ({concentration:.1%})
  P-value < 0.01:       {'PASS' if p_value < 0.01 else 'FAIL'} ({p_value:.2e})
  Improvement > 0:      {'PASS' if weighted_improvement > 0 else 'FAIL'} ({weighted_improvement:.1f}%)
  Temporal stability:   {'PASS' if (first_edge > 0 and second_edge > 0) else 'FAIL'}

{'='*50}
VERDICT: {'VALIDATED' if passes_all else 'NOT VALIDATED'}
{'='*50}
""")

# Save results
output = {
    'hypothesis': 'H102',
    'description': 'Low Leverage Variance + NO Consensus',
    'signal': 'lev_std < 0.5 AND yes_ratio < 0.4 AND n_trades >= 3',
    'markets': int(len(signal_df)),
    'win_rate': float(win_rate),
    'breakeven': float(breakeven),
    'edge_pct': float(edge),
    'improvement_pct': float(weighted_improvement),
    'p_value': float(p_value),
    'concentration': float(concentration),
    'temporal_stability': {
        'first_half_edge': float(first_edge),
        'second_half_edge': float(second_edge)
    },
    'is_validated': passes_all
}

output_path = '/Users/samuelclark/Desktop/kalshiflow/research/reports/session011_h102_validation.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved to: {output_path}")
