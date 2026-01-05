# Validation Framework Methodology Verification Report

**Created**: 2026-01-04
**Author**: Quant Agent
**Purpose**: Verify the validation framework implementation against proven methodology
**Reference Implementation**: `research/analysis/rlm_full_validation_20260104.py`
**Reference Results**: `research/reports/rlm_full_validation_20260104.json`

---

## Executive Summary

This document provides the authoritative specification for validation calculations. The trader specialist implementing `research/scripts/validation/` MUST follow these specifications exactly. Any deviation will produce incorrect results.

---

## CRITICAL METHODOLOGY REQUIREMENTS

### 1. Market-Level Aggregation (NOT Trade-Level)

**Status**: CRITICAL - Most common mistake

**WRONG Approach**:
```python
# DON'T count trades
n_trades = len(df[df['signal'] == True])
win_rate = (df[df['signal'] == True]['result'] == 'no').mean()
```

**CORRECT Approach**:
```python
# DO aggregate to market level FIRST
market_stats = df.groupby('market_ticker').agg({
    'result': 'first',           # Market outcome (same for all trades)
    'taker_side': lambda x: (x == 'yes').mean(),  # YES trade ratio
    'yes_price': ['first', 'last', 'mean'],
    'no_price': 'mean',
    'count': ['size', 'sum'],     # n_trades, total_contracts
    # ... other aggregations
}).reset_index()

# THEN filter markets by signal conditions
signal_markets = market_stats[
    (market_stats['yes_trade_ratio'] > 0.65) &
    (market_stats['n_trades'] >= 15) &
    (market_stats['yes_price_moved_down'])
]

# Calculate win rate at MARKET level
n_markets = len(signal_markets)
wins = (signal_markets['result'] == 'no').sum()
win_rate = wins / n_markets
```

**Why This Matters**:
- A market with 100 trades is ONE bet, not 100 bets
- Trade-level counting artificially inflates sample size
- Trade-level metrics are dominated by high-volume markets

**Verification Check**:
- `n_markets` should be in the hundreds to thousands, NOT millions
- H123 RLM has 4,148 markets (not ~7M trades)

---

### 2. Bucket-Matched Baseline Calculation

**Status**: CRITICAL - Required to detect price proxies

**Purpose**: Compare signal performance to the BASELINE win rate at the SAME price, not overall market.

**CORRECT Implementation**:

```python
def build_baseline(df):
    """Build baseline win rates at 5c price buckets for NO bets."""
    all_markets = df.groupby('market_ticker').agg({
        'result': 'first',
        'no_price': 'mean',  # Average NO price for the market
    }).reset_index()

    # Create 5-cent price buckets
    all_markets['bucket_5c'] = (all_markets['no_price'] // 5) * 5

    # Build baseline for each bucket
    baseline = {}
    for bucket in sorted(all_markets['bucket_5c'].unique()):
        bucket_markets = all_markets[all_markets['bucket_5c'] == bucket]
        n = len(bucket_markets)
        if n >= 20:  # Minimum for reliable baseline
            win_rate = (bucket_markets['result'] == 'no').mean()
            baseline[bucket] = {
                'win_rate': win_rate,
                'n_markets': n,
                'expected_edge': win_rate - (bucket + 2.5) / 100  # Midpoint
            }
    return baseline
```

**Bucket-Matched Comparison**:

```python
def calculate_bucket_improvement(signal_markets, baseline):
    """Calculate weighted improvement over bucket-matched baseline."""
    signal_markets['bucket_5c'] = (signal_markets['avg_no_price'] // 5) * 5

    bucket_results = []
    for bucket in sorted(signal_markets['bucket_5c'].unique()):
        if bucket not in baseline:
            continue

        sig_bucket = signal_markets[signal_markets['bucket_5c'] == bucket]
        n_sig = len(sig_bucket)
        if n_sig < 5:  # Minimum for comparison
            continue

        sig_win_rate = (sig_bucket['result'] == 'no').mean()
        base_win_rate = baseline[bucket]['win_rate']
        improvement = sig_win_rate - base_win_rate  # THIS IS THE KEY METRIC

        bucket_results.append({
            'bucket': bucket,
            'n_signal': n_sig,
            'n_baseline': baseline[bucket]['n_markets'],
            'signal_win_rate': sig_win_rate,
            'baseline_win_rate': base_win_rate,
            'improvement': improvement,
            'positive': improvement > 0
        })

    # Calculate WEIGHTED average improvement
    total_improvement = sum(br['improvement'] * br['n_signal'] for br in bucket_results)
    total_weight = sum(br['n_signal'] for br in bucket_results)
    avg_improvement = total_improvement / total_weight if total_weight > 0 else 0

    # Count positive buckets
    positive_buckets = sum(1 for br in bucket_results if br['positive'])
    bucket_ratio = positive_buckets / len(bucket_results) if bucket_results else 0

    return {
        'avg_improvement': avg_improvement,
        'bucket_ratio': bucket_ratio,
        'positive_buckets': positive_buckets,
        'n_buckets': len(bucket_results),
        'buckets': bucket_results
    }
```

**Price Proxy Detection**:

```python
def is_price_proxy(bucket_analysis):
    """
    Determine if a strategy is just a price proxy.

    A strategy is a PRICE PROXY if:
    - bucket_ratio < 0.6 (less than 60% of buckets show improvement)
    - OR avg_improvement < 0 (negative improvement when price-controlled)
    """
    return bucket_analysis['bucket_ratio'] < 0.6 or bucket_analysis['avg_improvement'] < 0
```

**H123 RLM Reference Values**:
- Positive buckets: 17/18 (94.4%)
- Avg improvement: +17.1%
- Is price proxy: FALSE

---

### 3. Win Rate and Edge Calculations

**CORRECT Formulas**:

```python
# Win rate (for NO bets)
wins = (signal_markets['result'] == 'no').sum()
n_markets = len(signal_markets)
win_rate = wins / n_markets

# Breakeven threshold (for NO bets)
avg_no_price = signal_markets['avg_no_price'].mean()  # In cents
breakeven = avg_no_price / 100  # Convert to fraction

# Raw edge
raw_edge = win_rate - breakeven

# Example: H123 RLM
# win_rate = 0.9421 (94.21%)
# avg_no_price = 72.37c
# breakeven = 0.7237 (72.37%)
# raw_edge = 0.9421 - 0.7237 = 0.2184 (21.84%)
```

**IMPORTANT**: The breakeven formula depends on which side you're betting:
- **Betting NO**: `breakeven = avg_no_price / 100`
- **Betting YES**: `breakeven = avg_yes_price / 100` (which is `1 - avg_no_price / 100`)

---

### 4. Statistical Significance (Z-Score and P-Value)

**CORRECT Implementation**:

```python
import numpy as np
from scipy import stats

def calculate_significance(wins, n_markets, breakeven):
    """
    Calculate z-score and p-value for the strategy.

    H0: True win rate = breakeven (no edge)
    H1: True win rate > breakeven (positive edge)

    This is a one-sided test (upper tail).
    """
    if n_markets == 0 or breakeven <= 0 or breakeven >= 1:
        return {'z_score': 0, 'p_value': 1.0, 'significant': False}

    # Expected wins under null hypothesis
    expected_wins = n_markets * breakeven

    # Standard deviation under null (binomial)
    std_dev = np.sqrt(n_markets * breakeven * (1 - breakeven))

    # Z-score
    z_score = (wins - expected_wins) / std_dev

    # P-value (one-sided upper tail)
    p_value = 1 - stats.norm.cdf(z_score)

    return {
        'z_score': z_score,
        'p_value': p_value,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'significant_001': p_value < 0.001
    }
```

**H123 RLM Reference Values**:
- Z-score: 31.46
- P-value: ~0 (extremely significant)

**Common Mistakes**:
- Using two-sided test (divide p-value by 2 for one-sided)
- Using t-test instead of normal approximation (fine for n >= 30)
- Not accounting for breakeven in null hypothesis

---

### 5. Confidence Intervals (95%)

**CORRECT Implementation**:

```python
def calculate_confidence_interval(win_rate, n_markets, breakeven):
    """
    Calculate 95% confidence interval for win rate and edge.

    Uses normal approximation (valid for n >= 30).
    """
    # Standard error
    std_err = np.sqrt(win_rate * (1 - win_rate) / n_markets)

    # 95% CI for win rate
    z_critical = 1.96  # 95% two-sided
    ci_lower = win_rate - z_critical * std_err
    ci_upper = win_rate + z_critical * std_err

    # 95% CI for edge
    edge_ci_lower = ci_lower - breakeven
    edge_ci_upper = ci_upper - breakeven

    return {
        'win_rate_lower': ci_lower,
        'win_rate_upper': ci_upper,
        'edge_lower': edge_ci_lower,
        'edge_upper': edge_ci_upper
    }
```

**H123 RLM Reference Values**:
- Win rate CI: [93.50%, 94.92%]
- Edge CI: [21.13%, 22.55%]

---

### 6. Bootstrap Confidence Intervals (Optional, More Robust)

**Implementation**:

```python
def bootstrap_confidence_interval(signal_markets, n_bootstrap=1000, ci=0.95):
    """
    Bootstrap confidence interval for edge.

    More robust than normal approximation, especially for small samples.
    """
    edges = []
    n = len(signal_markets)

    for _ in range(n_bootstrap):
        # Sample with replacement
        sample = signal_markets.sample(n=n, replace=True)

        wins = (sample['result'] == 'no').sum()
        win_rate = wins / n
        avg_price = sample['avg_no_price'].mean()
        breakeven = avg_price / 100
        edge = win_rate - breakeven
        edges.append(edge)

    # Calculate percentiles
    lower = np.percentile(edges, (1 - ci) / 2 * 100)
    upper = np.percentile(edges, (1 + ci) / 2 * 100)

    return {
        'bootstrap_mean': np.mean(edges),
        'bootstrap_std': np.std(edges),
        'ci_lower': lower,
        'ci_upper': upper,
        'edge_significant': lower > 0  # CI doesn't include zero
    }
```

---

### 7. Temporal Stability Check

**Purpose**: Ensure strategy works across different time periods, not just one lucky window.

**CORRECT Implementation**:

```python
def check_temporal_stability(signal_markets, time_column='first_trade_time'):
    """
    Check if strategy performs consistently across time periods.

    Threshold: >= 50% of periods should be profitable.
    """
    signal_markets = signal_markets.copy()
    signal_markets['week'] = signal_markets[time_column].dt.isocalendar().week

    weekly_results = []
    for week in sorted(signal_markets['week'].unique()):
        week_data = signal_markets[signal_markets['week'] == week]
        n = len(week_data)

        if n >= 10:  # Minimum for reliable weekly stat
            wins = (week_data['result'] == 'no').sum()
            win_rate = wins / n
            avg_price = week_data['avg_no_price'].mean()
            edge = win_rate - (avg_price / 100)

            weekly_results.append({
                'week': week,
                'n_markets': n,
                'win_rate': win_rate,
                'edge': edge,
                'profitable': edge > 0
            })

    positive_weeks = sum(1 for w in weekly_results if w['profitable'])
    stability_ratio = positive_weeks / len(weekly_results) if weekly_results else 0

    return {
        'weekly_results': weekly_results,
        'positive_weeks': positive_weeks,
        'total_weeks': len(weekly_results),
        'stability_ratio': stability_ratio,
        'passes_threshold': stability_ratio >= 0.5
    }
```

**H123 RLM Reference Values**:
- Positive weeks: 4/4 (100%)
- Passes threshold: TRUE

---

### 8. Concentration Check

**Purpose**: Ensure profit isn't dominated by a few lucky markets.

**CORRECT Implementation**:

```python
def check_concentration(signal_markets, max_single_market=0.30):
    """
    Check for concentration risk.

    Threshold: No single market should contribute >30% of total profit.
    """
    signal_markets = signal_markets.copy()
    signal_markets['won'] = (signal_markets['result'] == 'no').astype(int)
    signal_markets['profit'] = signal_markets['won'] - (signal_markets['avg_no_price'] / 100)

    total_profit = signal_markets['profit'].sum()

    if total_profit <= 0:
        return {
            'total_profit': total_profit,
            'concentration_ok': False,
            'reason': 'negative_total_profit'
        }

    # Max contribution from single market
    max_contribution = signal_markets['profit'].max() / total_profit

    # Top 5 contribution
    top5_contribution = signal_markets.nlargest(5, 'profit')['profit'].sum() / total_profit

    return {
        'total_profit': total_profit,
        'max_single_market_contribution': max_contribution,
        'top5_markets_contribution': top5_contribution,
        'concentration_ok': max_contribution < max_single_market,
        'threshold': max_single_market
    }
```

**H123 RLM Reference Values**:
- Max single market: 0.09% (well below 30%)
- Top 5 markets: 0.44%
- Concentration OK: TRUE

---

## Validation Criteria Summary

All strategies must pass ALL of these checks:

| Criterion | Threshold | H123 RLM Result |
|-----------|-----------|-----------------|
| Sample size | N >= 50 markets | 4,148 PASS |
| Statistical significance | p < 0.05 | p ~ 0 PASS |
| Not a price proxy | bucket_ratio >= 0.6 AND avg_improvement > 0 | 94.4%, +17.1% PASS |
| Concentration | max_single < 30% | 0.09% PASS |
| Temporal stability | >= 50% weeks profitable | 100% PASS |

---

## Common Pitfalls to Avoid

### 1. Trade-Level Analysis
**Wrong**: Counting trades instead of markets
**Fix**: Always aggregate to market level first

### 2. Missing Bucket Matching
**Wrong**: Comparing to overall baseline
**Fix**: Compare to same-price-bucket baseline

### 3. Incorrect Breakeven Formula
**Wrong**: Using YES price for NO bets
**Fix**: Match price to betting side

### 4. One-Sided vs Two-Sided Tests
**Wrong**: Using two-sided p-value for directional hypothesis
**Fix**: Use one-sided test (upper tail for positive edge)

### 5. Look-Ahead Bias
**Wrong**: Using post-resolution data in signal
**Fix**: Only use data available at trade time

### 6. Data Leakage in Temporal Splits
**Wrong**: Using overlapping time periods
**Fix**: Clean temporal separation

---

## Framework Verification Checklist

Before releasing the validation framework, verify:

- [ ] Market-level aggregation produces same count as reference (4,148 for H123)
- [ ] Baseline buckets match reference (18 buckets for H123)
- [ ] Win rate matches reference (94.21% for H123)
- [ ] Raw edge matches reference (+21.84% for H123)
- [ ] Z-score matches reference (31.46 for H123)
- [ ] P-value matches reference (~0 for H123)
- [ ] Bucket improvement matches reference (+17.1% for H123)
- [ ] Confidence interval matches reference ([21.13%, 22.55%] for H123)
- [ ] Temporal stability matches reference (4/4 weeks for H123)
- [ ] Concentration check matches reference (0.09% max for H123)
- [ ] Price proxy detection returns FALSE for H123

---

## Test Case: H123 RLM Validation

Use this configuration to test the framework:

```yaml
strategy:
  name: "H123 - Reverse Line Movement NO"
  hypothesis_id: "H123"
  action: "bet_no"

signal:
  conditions:
    - field: "yes_trade_ratio"
      operator: ">"
      value: 0.65
    - field: "yes_price_moved_down"
      operator: "=="
      value: true
    - field: "n_trades"
      operator: ">="
      value: 15
  entry_price_field: "avg_no_price"

validation:
  mode: "full"
  min_markets: 50
  p_threshold: 0.05
  bucket_size: 5
```

**Expected Results**:
```json
{
  "n_markets": 4148,
  "wins": 3908,
  "losses": 240,
  "win_rate": 0.9421,
  "avg_no_price": 72.37,
  "breakeven": 0.7237,
  "raw_edge": 0.2184,
  "z_score": 31.46,
  "p_value": 0.0,
  "bucket_analysis": {
    "n_buckets": 18,
    "positive_buckets": 17,
    "bucket_ratio": 0.944,
    "avg_improvement": 0.171
  },
  "is_price_proxy": false,
  "temporal_stability_ratio": 1.0,
  "concentration_ok": true,
  "verdict": "VALIDATED"
}
```

If the framework produces these results for H123, the methodology is correct.

---

## Appendix: Data Schema

**Expected Input Data** (enriched_trades_resolved_ALL.csv):

| Column | Type | Description |
|--------|------|-------------|
| market_ticker | string | Unique market identifier |
| datetime | timestamp | Trade timestamp |
| trade_price | int | Trade price in cents |
| yes_price | int | YES price (100 - NO price) |
| no_price | int | NO price in cents |
| count | int | Number of contracts |
| taker_side | string | "yes" or "no" |
| result | string | "yes" or "no" (market outcome) |
| leverage_ratio | float | Leverage at time of trade |

**Market-Level Aggregation Output**:

| Column | Type | Description |
|--------|------|-------------|
| market_ticker | string | Unique market identifier |
| result | string | Market outcome ("yes" or "no") |
| yes_trade_ratio | float | Fraction of YES trades |
| first_yes_price | int | Opening YES price |
| last_yes_price | int | Closing YES price |
| avg_yes_price | float | Average YES price |
| avg_no_price | float | Average NO price |
| n_trades | int | Number of trades |
| total_contracts | int | Sum of contracts |
| yes_price_moved_down | bool | Last YES price < First YES price |
| bucket_5c | int | 5-cent price bucket (0, 5, 10, ..., 95) |

---

*End of Verification Document*
