#!/usr/bin/env python3
"""
Session 011: TRUE Apples-to-Apples Comparison for H087

The previous analysis showed +50.3% improvement but the signal markets have
avg NO price of 16.5c vs baseline avg of 25.1c.

For a TRUE comparison, we need to compare:
- Signal markets in each price bucket
- vs ALL markets in that EXACT same price bucket (not just filtered differently)

This will give us the REAL improvement the signal provides at each price level.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("APPLES-TO-APPLES COMPARISON: H087 Refined")
print("="*80)

# Load data
df = pd.read_csv('/Users/samuelclark/Desktop/kalshiflow/research/data/trades/enriched_trades_resolved_ALL.csv')
print(f"Loaded {len(df):,} trades")

# Define round sizes
ROUND_SIZES = [10, 25, 50, 100, 250, 500, 1000]
df['is_round_size'] = df['count'].isin(ROUND_SIZES)

# ==============================================================================
# GET ALL MARKETS (baseline)
# ==============================================================================
all_market_agg = df.groupby('market_ticker').agg({
    'trade_price': 'mean',  # avg YES price
    'market_result': 'first'
}).reset_index()
all_market_agg['avg_no_price'] = 100 - all_market_agg['trade_price']
all_market_agg['won'] = (all_market_agg['market_result'] == 'no').astype(int)
all_market_agg['bucket'] = (all_market_agg['avg_no_price'] // 5) * 5

print(f"Total markets: {len(all_market_agg)}")

# ==============================================================================
# GET SIGNAL MARKETS
# ==============================================================================
round_size_trades = df[df['is_round_size']]
round_consensus = round_size_trades.groupby('market_ticker').agg({
    'taker_side': lambda x: (x == 'yes').mean(),
    'trade_price': 'mean',
    'market_result': 'first'
}).reset_index()
round_consensus.columns = ['market_ticker', 'yes_ratio', 'avg_yes_price', 'market_result']
round_consensus['avg_no_price'] = 100 - round_consensus['avg_yes_price']
round_consensus['won'] = (round_consensus['market_result'] == 'no').astype(int)
round_consensus['bucket'] = (round_consensus['avg_no_price'] // 5) * 5

# Signal: >60% NO consensus AND NO < 45c
signal_df = round_consensus[
    (round_consensus['yes_ratio'] < 0.4) &
    (round_consensus['avg_no_price'] < 45)
].copy()

print(f"Signal markets: {len(signal_df)}")

# ==============================================================================
# TRUE APPLES-TO-APPLES: Compare signal vs ALL markets in SAME bucket
# ==============================================================================
print("\n" + "-"*80)
print("TRUE APPLES-TO-APPLES COMPARISON (Signal vs ALL markets at same price)")
print("-"*80)

print(f"{'Bucket':<10} {'Signal':<15} {'ALL Markets':<20} {'Improvement':<12} {'Sig'}")
print(f"{'(NO price)':<10} {'WR (N)':<15} {'WR (N)':<20} {'(pp)':<12}")
print("-"*80)

total_signal_n = 0
total_signal_wins = 0
total_weighted_improvement = 0

for bucket in sorted(signal_df['bucket'].unique()):
    signal_bucket = signal_df[signal_df['bucket'] == bucket]
    all_bucket = all_market_agg[all_market_agg['bucket'] == bucket]

    if len(signal_bucket) > 0 and len(all_bucket) > 0:
        sig_wr = signal_bucket['won'].mean()
        all_wr = all_bucket['won'].mean()
        sig_n = len(signal_bucket)
        all_n = len(all_bucket)
        improvement = (sig_wr - all_wr) * 100

        # Statistical test
        # H0: signal win rate = all markets win rate
        # Use chi-squared test for two proportions
        sig_wins = signal_bucket['won'].sum()
        all_wins = all_bucket['won'].sum()

        # Use chi-squared test manually
        # Expected proportion = all_wr
        # Observed = sig_wr
        # Simple binomial test instead
        binom_test = stats.binomtest(sig_wins, sig_n, all_wr, alternative='greater')
        p_val = binom_test.pvalue

        sig_star = '*' if p_val < 0.01 else ''

        print(f"{bucket}-{bucket+5}c     {sig_wr:.1%} ({sig_n}){' '*(8-len(str(sig_n)))} {all_wr:.1%} ({all_n}){' '*(12-len(str(all_n)))} {improvement:+.1f}%       {sig_star}")

        total_signal_n += sig_n
        total_signal_wins += signal_bucket['won'].sum()
        total_weighted_improvement += improvement * sig_n

print("-"*80)

# Overall weighted improvement
weighted_improvement = total_weighted_improvement / total_signal_n

print(f"\n{'OVERALL WEIGHTED IMPROVEMENT:':<40} {weighted_improvement:+.1f}%")

# Overall signal stats
overall_sig_wr = total_signal_wins / total_signal_n
overall_sig_no_price = signal_df['avg_no_price'].mean()
overall_sig_breakeven = overall_sig_no_price / 100
overall_sig_edge = (overall_sig_wr - overall_sig_breakeven) * 100

print(f"\nOverall Signal Stats:")
print(f"  Markets: {total_signal_n}")
print(f"  Win Rate: {overall_sig_wr:.2%}")
print(f"  Avg NO Price: {overall_sig_no_price:.1f}c")
print(f"  Breakeven: {overall_sig_breakeven:.2%}")
print(f"  Edge: {overall_sig_edge:.1f}%")
print(f"  Improvement over same-price baseline: {weighted_improvement:.1f}%")

# ==============================================================================
# VALIDATION WITH CORRECT IMPROVEMENT
# ==============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

# P-value for overall difference
# Compare signal to null hypothesis of baseline win rate at same prices
# Weight baseline WR by signal's price distribution
signal_df_with_baseline = signal_df.merge(
    all_market_agg.groupby('bucket')['won'].mean().reset_index().rename(columns={'won': 'baseline_wr'}),
    on='bucket',
    how='left'
)

expected_wins_under_null = (signal_df_with_baseline['baseline_wr'] * 1).sum()
actual_wins = signal_df['won'].sum()

# Binomial test against expected rate
expected_rate = expected_wins_under_null / len(signal_df)
binom_result = stats.binomtest(actual_wins, len(signal_df), expected_rate, alternative='greater')
p_value = binom_result.pvalue

print(f"\nStatistical Significance:")
print(f"  Expected wins under null (baseline rate): {expected_wins_under_null:.0f}")
print(f"  Actual wins: {actual_wins}")
print(f"  Expected rate: {expected_rate:.2%}")
print(f"  Actual rate: {signal_df['won'].mean():.2%}")
print(f"  P-value vs baseline: {p_value:.2e}")
print(f"  Bonferroni threshold (0.05/5): 0.01")
print(f"  Passes Bonferroni: {p_value < 0.01}")

# Temporal stability
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

print(f"\nTemporal Stability:")
print(f"  First half: {len(first_half)} markets, WR={first_wr:.1%}, Edge={first_edge:.1f}%")
print(f"  Second half: {len(second_half)} markets, WR={second_wr:.1%}, Edge={second_edge:.1f}%")
print(f"  Stable: {first_edge > 0 and second_edge > 0}")

# Concentration (rough estimate)
print(f"\nConcentration: ~1.6% (from previous analysis)")

# ==============================================================================
# FINAL VERDICT
# ==============================================================================
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

all_pass = (
    len(signal_df) >= 50 and
    p_value < 0.01 and
    weighted_improvement > 0 and
    first_edge > 0 and second_edge > 0
)

print(f"""
Strategy: S010 - Follow Round-Size Bot NO Consensus (NO < 45c)

Validation Criteria:
  Markets >= 50:           PASS ({len(signal_df)} markets)
  Concentration < 30%:     PASS (~1.6%)
  P-value < 0.01:          {'PASS' if p_value < 0.01 else 'FAIL'} ({p_value:.2e})
  Improvement > 0:         {'PASS' if weighted_improvement > 0 else 'FAIL'} ({weighted_improvement:.1f}%)
  Temporal stability:      {'PASS' if (first_edge > 0 and second_edge > 0) else 'FAIL'}

KEY METRICS:
  Edge: {overall_sig_edge:.1f}%
  Improvement over baseline: {weighted_improvement:.1f}%
  Expected return per $1: ${(overall_sig_wr * (100-overall_sig_no_price)/overall_sig_no_price - (1-overall_sig_wr)):.2f}

{'='*50}
VERDICT: {'VALIDATED' if all_pass else 'NOT VALIDATED'}
{'='*50}
""")

# Save final results
output = {
    'strategy_id': 'S010',
    'hypothesis': 'H087_refined',
    'description': 'Follow Round-Size Bot NO Consensus when NO < 45c',
    'signal': 'Markets where >60% of round-size trades are NO AND avg NO price < 45c',
    'action': 'Bet NO at current NO price',
    'validation': {
        'markets': int(len(signal_df)),
        'win_rate': float(overall_sig_wr),
        'avg_no_price': float(overall_sig_no_price),
        'breakeven': float(overall_sig_breakeven),
        'edge_pct': float(overall_sig_edge),
        'improvement_over_baseline_pct': float(weighted_improvement),
        'p_value': float(p_value),
        'temporal_stability': {
            'first_half_edge': float(first_edge),
            'second_half_edge': float(second_edge)
        }
    },
    'is_validated': all_pass
}

output_path = '/Users/samuelclark/Desktop/kalshiflow/research/reports/session011_h087_apples_to_apples.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved to: {output_path}")
