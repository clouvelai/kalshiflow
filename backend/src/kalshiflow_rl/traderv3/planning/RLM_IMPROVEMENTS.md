# RLM_NO Strategy Optimization Report

> **Research Date**: 2025-12-31
> **Mode**: LSD Mode (Rapid Lateral Exploration)
> **Data**: 1.62M trades, 72.8K resolved markets

---

## Executive Summary

### Top 3 Actionable Improvements (Updated 2025-12-31)

| Priority | Change | Impact | Effort |
|----------|--------|--------|--------|
| **P0** | Increase YES threshold from 65% to 70% | -86% false positive rate (15.1% -> 2.2%) | Low (env var) |
| **P1** | Increase min_trades from 15 to 25 | +0.9% win rate, +0.3% edge | Low (env var) |
| **P2** | Keep price drop at 2c (or increase to 3c) | Minimal change | None |

> **UPDATED ANALYSIS**: Full reliability grid search (Section 10) reveals that the current config
> (YES>65%, min_trades=15) has a **15.1% false positive rate** - 1 in 7 signals could be noise.
> Moving to YES>70%, min_trades=25 reduces false positive rate to **2.2%** (7x improvement)
> while maintaining similar edge (+3.0% vs +2.7%) and statistical significance (p=0.0014).
> See Section 10 for full Signal Reliability Matrix.

### Key Findings

1. **✅ IMPLEMENTED: High Reliability Configuration**:
   - Old: YES>65%, min_trades=15, drop>=2c → 15.1% false positive rate
   - **New: YES>70%, min_trades=25, drop>=2c → 2.2% false positive rate**
   - **7x improvement in signal reliability**

2. **Risk-adjusted analysis drove the final decision**:
   - min_trades=10 has 37.7% false positive rate (too high)
   - min_trades=15 has 15.1% false positive rate (was current)
   - **min_trades=25 has 2.2% false positive rate (now implemented)**

3. **All categories show positive edge** - no need to exclude any
   - Strongest: Media_Mentions (+24.4%), Sports_Pro (+23.7%)
   - Weakest but still positive: Crypto (+16.3%), Entertainment (+16.5%)

4. **Signal strength matters** (future enhancement):
   - Large price drops (20-50c): 28.1% edge
   - Very high trade velocity: 28.4% edge
   - Morning hours (8-11 AM, 4 PM, 9 PM): >25% edge

---

## 1. Signal Configuration Analysis

### 1.1 Price Drop Threshold (Critical Decision)

**Question**: Should we use 2c or 5c price drop?

| Threshold | Markets | Win Rate | Edge | EV/$100 | Total EV |
|-----------|---------|----------|------|---------|----------|
| 0c | 1,418 | 92.0% | 19.0% | $26.10 | $37,007 |
| 2c | 1,207 | 93.4% | 22.0% | $30.87 | $37,264 |
| **3c** | **1,141** | **94.5%** | **23.4%** | **$33.01** | **$37,663** |
| 5c | 1,042 | 95.4% | 24.9% | $35.29 | $36,772 |
| 7c | 967 | 96.0% | 25.8% | $36.75 | $35,535 |
| 10c | 879 | 96.6% | 27.3% | $39.37 | $34,609 |

**Recommendation**: Use **3c** price drop threshold
- Best trade-off between edge (23.4%) and signal frequency
- Peak Total EV ($37,663)
- 6% fewer markets than 2c but 1.4% higher edge

**Alternative**: If prioritizing fill rate over edge, keep 2c
- More signals = more chances to execute
- In practice, not all signals will fill anyway

### 1.2 YES Ratio Threshold

| Threshold | Markets | Win Rate | Edge | Total EV |
|-----------|---------|----------|------|----------|
| **60%** | **1,380** | **93.6%** | **22.5%** | **$43,766** |
| 65% | 1,207 | 93.4% | 22.0% | $37,264 |
| 70% | 1,020 | 92.8% | 21.2% | $30,146 |
| 75% | 813 | 92.7% | 20.7% | $23,380 |

**Recommendation**: Lower to **60%**
- +17% more signals
- Similar edge (0.5% difference)
- +17% Total EV improvement

**However**, interesting non-monotonic pattern in signal strength:
- 65-70% YES ratio: **26.6% edge** (best!)
- 90-100% YES ratio: 17.0% edge (worst)

This suggests:
- Very high YES ratios may be "too obvious" - price already adjusted
- Moderate YES ratios (65-75%) may catch the sweet spot

### 1.3 Minimum Trades

| Min Trades | Markets | Win Rate | Edge | Total EV | False Positive Rate |
|------------|---------|----------|------|----------|---------------------|
| 5 | 1,996 | 91.3% | 19.7% | $54,916 | ~45% |
| 10 | 1,443 | 92.8% | 21.4% | $43,314 | **37.7%** |
| **15** | **1,207** | **93.4%** | **22.0%** | **$37,264** | **30.4%** |
| 20 | 1,052 | 94.2% | 22.9% | $33,769 | 25.2% |
| 25 | 950 | 94.8% | 23.4% | $31,185 | 21.2% |

**Recommendation**: Keep at **15** trades (current setting)

**Why NOT lower to 10?**
- With min_trades=10 and YES_threshold=60%, triggering requires only 6/10 YES trades
- **37.7% false positive rate**: If trades were random noise, we'd falsely trigger 38% of the time!
- min_trades=15 reduces false positive rate to 30.4%
- The +16% Total EV from more signals is offset by lower per-signal reliability

**Risk-Adjusted Analysis**:
| Metric | min_trades=10 | min_trades=15 |
|--------|---------------|---------------|
| False Positive Rate | 37.7% | 30.4% |
| Win Rate | 92.8% | 93.4% |
| Trades until first loss | 16.4 | 18.9 |

**Rationale**: For production launch, prioritize signal reliability over volume.
Can revisit min_trades=10 after proving system stability.

---

## 2. Recommended Configuration

### Production Launch Configuration (RECOMMENDED)

```bash
RLM_YES_THRESHOLD=0.65     # Keep current (proven)
RLM_MIN_TRADES=15          # Keep current (lower false positive rate)
RLM_MIN_PRICE_DROP=3       # Increase from 2 to 3
```

| Metric | Current (2c) | Recommended (3c) | Change |
|--------|--------------|------------------|--------|
| Markets | 1,207 | 1,141 | -5% |
| Win Rate | 93.4% | 94.5% | +1.1% |
| Edge | 22.0% | 23.4% | **+1.4%** |
| False Positive Rate | 30.4% | 30.4% | Same |

**Why this configuration?**
1. **Only one parameter change** - minimal risk
2. **Higher conviction signals** - 3c price drop filters out noise
3. **Better edge** - +1.4% improvement per signal
4. **Same reliability** - keeps min_trades=15 for lower false positive rate

### Alternative: High-Edge Configuration (Post-Launch)

```bash
RLM_YES_THRESHOLD=0.65
RLM_MIN_TRADES=20          # Higher conviction
RLM_MIN_PRICE_DROP=5       # Stronger signal
```

| Metric | Value |
|--------|-------|
| Markets | 1,042 |
| Edge | 24.9% |
| Win Rate | 95.4% |
| False Positive Rate | 25.2% |

Consider this after proving system stability - fewer signals but higher quality.

### NOT Recommended: Aggressive Configuration

```bash
# DO NOT USE FOR LAUNCH
RLM_YES_THRESHOLD=0.60
RLM_MIN_TRADES=10
RLM_MIN_PRICE_DROP=3
```

While this shows +36% Total EV in backtests, the **37.7% false positive rate** at min_trades=10
means 6/10 YES trades could easily be random noise. Too risky for production.

---

## 3. Order Pricing Analysis

### Current Logic (rlm_service.py lines 690-721)

```python
def _calculate_no_entry_price(best_no_ask, best_no_bid, spread):
    if spread <= 2:   # Tight spread
        return best_no_ask - 1     # Aggressive, queue just below ask
    elif spread <= 4: # Normal spread
        return (best_no_ask + best_no_bid) // 2  # Midpoint
    else:             # Wide spread
        return best_no_bid + (spread * 3 // 4)   # 75% toward ask
```

### Assessment: **GOOD, Keep Current Logic**

**Rationale**:
1. RLM is a time-sensitive signal - we need fills, not price improvement
2. Current logic prioritizes fill probability appropriately
3. At tight spreads (<=2c), being 1c below ask is aggressive enough
4. At wide spreads, being 75% toward ask balances fill vs slippage

### Potential Enhancement (NOT RECOMMENDED for MVP)

Signal-strength adjusted pricing could be explored later:
- Stronger signals (larger price drop) -> more aggressive pricing
- Weaker signals -> more patient pricing

But this adds complexity without clear evidence of benefit. **Keep simple for now.**

---

## 4. Category Configuration

### All Categories Show Positive Edge

| Category | Markets | Edge | Verdict |
|----------|---------|------|---------|
| Media_Mentions | 23 | 24.4% | INCLUDE |
| Sports_Pro | 426 | 23.7% | INCLUDE |
| Other | 239 | 22.9% | INCLUDE |
| Sports_Other | 104 | 22.1% | INCLUDE |
| Politics | 19 | 21.5% | INCLUDE |
| Sports_College | 153 | 21.5% | INCLUDE |
| Sports_Soccer | 118 | 20.1% | INCLUDE |
| Weather | 19 | 18.0% | INCLUDE |
| Entertainment | 33 | 16.5% | INCLUDE |
| Crypto | 67 | 16.3% | INCLUDE |

**Recommendation**: Include ALL categories
- No category shows negative edge
- Excluding any would reduce signal frequency without improving overall edge
- The current category filter in lifecycle discovery is good

### Current Category Configuration (Keep As-Is)

```python
LIFECYCLE_CATEGORIES=sports,media_mentions,entertainment,crypto
```

This correctly includes the highest-edge categories.

**Note**: Politics and Weather also show positive edge but have few markets.
Consider adding to lifecycle categories if they become more active.

---

## 5. LSD Mode Discoveries

### 5.1 Time-of-Day Patterns

**High-Edge Hours** (>25% edge):
- Hour 8 (8 AM): 26.5% edge
- Hour 10 (10 AM): 29.2% edge
- Hour 11 (11 AM): 25.8% edge
- Hour 16 (4 PM): 29.0% edge
- Hour 21 (9 PM): 29.4% edge

**Low-Edge Hours** (<15% edge):
- Hour 0 (midnight): 16.5% edge
- Hour 14 (2 PM): 10.1% edge (AVOID)
- Hour 17 (5 PM): 12.0% edge

**NOT RECOMMENDED for MVP**: Time-based filtering adds complexity without
enough sample size to be confident. Monitor for future optimization.

### 5.2 Signal Strength Amplifiers

**Large Price Drops = Much Higher Edge**:
| Drop Size | Edge |
|-----------|------|
| 2-5c | 11.3% |
| 5-10c | 11.9% |
| 10-20c | 18.2% |
| 20-50c | **28.1%** |
| 50+c | **38.7%** |

**Implication**: Consider position sizing based on signal strength:
- Small drop (2-5c): Standard size
- Medium drop (10-20c): 1.5x size
- Large drop (20c+): 2x size

**NOT RECOMMENDED for MVP** - adds complexity. Revisit post-launch.

### 5.3 Trade Velocity

**More Trades = Higher Edge**:
| Velocity | Avg Trades | Edge |
|----------|------------|------|
| Low | 20 | 18.0% |
| Medium | 44 | 19.6% |
| High | 120 | 22.3% |
| Very High | 1,950 | **28.4%** |

**Implication**: Markets with high trading activity have stronger RLM signal.
This makes sense - more data = more reliable signal.

**Current min_trades=15 is appropriate** - see Section 8 for risk-adjusted analysis showing
min_trades=10 has 37.7% false positive rate.

### 5.4 Absurd Ideas Tested (Rejected)

- **Moon phases**: Not testable with available data
- **Fibonacci trade counts**: No pattern found
- **Double-down logic**: Increases risk without clear benefit
- **Signal decay**: Not enough data to analyze time-to-fill

---

## 6. Implementation Checklist

### ✅ IMPLEMENTED: High Reliability Configuration (2025-12-31)

The following configuration is now live in both `environment.py` defaults and `.env.paper`:

```bash
# High Reliability Config (2.2% false positive rate)
RLM_YES_THRESHOLD=0.70    # Changed from 0.65
RLM_MIN_TRADES=25         # Changed from 15
RLM_MIN_PRICE_DROP=2      # Kept at 2
```

**Files Updated:**
- `backend/src/kalshiflow_rl/traderv3/config/environment.py` - Updated defaults
- `backend/.env.paper` - Added explicit RLM settings

### Configuration Comparison

| Parameter | Old Value | New Value | Rationale |
|-----------|-----------|-----------|-----------|
| YES Threshold | 0.65 | **0.70** | Higher conviction signal |
| Min Trades | 15 | **25** | Lower false positive rate |
| Price Drop | 2c | 2c | No change |
| **False Positive Rate** | 15.1% | **2.2%** | **7x improvement** |

### No Changes Needed

- `RLM_CONTRACTS=100` - Keep current
- Order pricing logic - Keep current spread-aware logic
- Category filtering - Keep current (all categories positive)
- Rate limiting - Keep current 10/min

### Future Considerations (Post-Launch)

1. **Increase price_drop to 3c**: For +1.4% edge improvement
2. **Position sizing by signal strength**: Larger drops -> larger positions
3. **Time-based filtering**: Avoid hour 14 (2 PM)
4. **Velocity filtering**: Prioritize high-activity markets

### NOT Recommended

~~`RLM_MIN_TRADES=10`~~ - 37.7% false positive rate is too high for production

---

## 7. Expected Impact

### With Recommended Change (price_drop: 2c → 3c)

| Metric | Before (2c) | After (3c) | Change |
|--------|-------------|------------|--------|
| Markets | 1,207 | 1,141 | -5% |
| Edge | 22.0% | 23.4% | **+1.4%** |
| Win Rate | 93.4% | 94.5% | +1.1% |
| EV per $100 | $30.87 | $33.01 | +7% |

### Why Not Aggressive Changes?

The "aggressive" configuration (YES=60%, min_trades=10) showed +36% Total EV in backtests,
but this optimizes for volume over reliability:

| Config | Markets | Edge | False Positive Rate |
|--------|---------|------|---------------------|
| Conservative (recommended) | 1,141 | 23.4% | 30.4% |
| Aggressive (not recommended) | 1,547 | 23.2% | 37.7% |

The aggressive config has **7% higher false positive rate** for similar edge.
For production launch, we prioritize signal reliability.

---

## 8. Risk-Adjusted Analysis (Added Post-Review)

### The min_trades=10 Problem

User correctly identified that 6/10 YES trades (60% threshold at 10 trades) is a weak signal.

**Statistical Analysis**: If trades were truly random (50/50), how often would we falsely trigger?

| min_trades | Required YES | False Positive Rate |
|------------|--------------|---------------------|
| 10 | 6/10 (60%) | **37.7%** |
| 15 | 9/15 (60%) | **30.4%** |
| 20 | 12/20 (60%) | **25.2%** |

With 10 trades, we'd falsely trigger 38% of the time if trades were random noise!

### Full Risk Comparison

| Metric | min_trades=10 | min_trades=15 | min_trades=20 |
|--------|---------------|---------------|---------------|
| Markets | 1,443 | 1,207 | 1,052 |
| Win Rate | 92.8% | 93.4% | 94.2% |
| Edge | 21.4% | 22.0% | 22.9% |
| **False Positive Rate** | 37.7% | 30.4% | 25.2% |
| StdDev per market | 23.9c | 22.4c | 21.4c |
| Trades until first loss | 16.4 | 18.9 | 20.8 |

### Key Insight: Information Ratio is Similar

The Information Ratio (Edge / StdDev) is ~0.14 across all min_trades levels.
Higher min_trades reduces both edge AND variance proportionally.

**Conclusion**: min_trades=15 provides the best balance of:
- Reasonable signal frequency (1,207 markets)
- Lower false positive rate (30.4% vs 37.7%)
- Higher per-signal win rate (93.4% vs 92.8%)
- Fewer psychological losses (18.9 vs 16.4 trades until first loss)

---

## 9. Research Artifacts

- Analysis script: `research/analysis/rlm_optimization_lsd.py`
- Raw results: `research/reports/rlm_optimization_lsd.json`
- H123 validation: `research/reports/h123_production_validation.json`
- This document: `backend/src/kalshiflow_rl/traderv3/planning/RLM_IMPROVEMENTS.md`

---

## 10. Signal Reliability Matrix (Full Grid Search Analysis)

> **Analysis Date**: 2025-12-31
> **Objective**: Find the MOST RELIABLE signal configuration, NOT highest Total EV
> **Analysis Script**: `research/analysis/rlm_reliability_grid_search.py`
> **Results**: `research/reports/rlm_reliability_grid_search.json`

### 10.1 Key Insight: Reliability vs Volume Trade-off

The previous analysis optimized for Total EV, which pushed toward aggressive settings (YES=60%, min_trades=10).
This is risky because with only 6/10 YES trades required, the signal could easily be noise.

**The core question**: What configuration gives us highest CONFIDENCE the signal is real?

### 10.2 False Positive Rate Calculation

False Positive Rate = P(trigger | random 50/50 trades) using binomial distribution.

If trades were truly random, how often would we falsely trigger?

| YES Threshold | Required at 10 trades | Required at 15 trades | Required at 20 trades | Required at 25 trades |
|---------------|----------------------|----------------------|----------------------|----------------------|
| 55% | 6/10 = 37.7% FP | 9/15 = 30.4% FP | 12/20 = 25.2% FP | 14/25 = 34.5% FP |
| 60% | 7/10 = **17.2% FP** | 10/15 = 15.1% FP | 13/20 = 13.2% FP | 16/25 = 11.5% FP |
| 65% | 7/10 = 17.2% FP | 10/15 = 15.1% FP | 14/20 = **5.8% FP** | 17/25 = 5.4% FP |
| 70% | 8/10 = 5.5% FP | 11/15 = 5.9% FP | 15/20 = 2.1% FP | 18/25 = 2.2% FP |
| 75% | 8/10 = 5.5% FP | 12/15 = 1.8% FP | 16/20 = 0.6% FP | 19/25 = 0.7% FP |
| 80% | 9/10 = **1.1% FP** | 13/15 = 0.4% FP | 17/20 = 0.1% FP | 21/25 = 0.0% FP |

### 10.3 Full Reliability Matrix

Cell format: **False Positive Rate | Win Rate | Edge | p-value | Markets**

```
YES>    | min_trades=10      | min_trades=15      | min_trades=20      | min_trades=25      | min_trades=30      | min_trades=40      | min_trades=50
--------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------
55%     | !37.7% 93.2% +3.0% | 30.4% 93.8% +2.4%  | 25.2% 94.5% +2.5%  | 34.5% 95.1% +2.5%  | 29.2% 95.2% +2.2%  | *21.5% 95.8% +2.3% | *24.0% 95.8% +2.2%
        | p<0.001 N=1854     | p<0.001 N=1579     | p<0.001 N=1393     | p<0.001 N=1267     | p=0.002 N=1172     | p=0.001 N=1037     | p=0.003 N=909
--------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------
60%     | *17.2% 92.8% +3.0% | *15.1% 93.4% +2.5% | *13.2% 94.2% +2.5% | *11.5% 94.9% +2.5% | *10.0% 95.0% +2.3% | *7.7% 95.5% +2.5%  | *5.9% 95.6% +2.4%
        | p<0.001 N=1657     | p<0.001 N=1401     | p<0.001 N=1231     | p<0.001 N=1116     | p=0.003 N=1035     | p=0.002 N=916      | p=0.004 N=800
--------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------
65%     | *17.2% 92.7% +3.2% | *15.1% 93.2% +2.7% | *5.8% 94.1% +2.8%  | *5.4% 94.7% +2.8%  | *4.9% 94.7% +2.4%  | *1.9% 95.3% +2.6%  | *1.6% 95.4% +2.5%
        | p<0.001 N=1461     | p<0.001 N=1225     | p<0.001 N=1067     | p<0.001 N=963      | p=0.003 N=890      | p=0.002 N=783      | p=0.005 N=690
--------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------
70%     | *5.5% 92.3% +3.4%  | *5.9% 92.7% +2.8%  | *2.1% 93.5% +2.9%  | *2.2% 94.1% +3.0%  | *0.8% 94.1% +2.6%  | *0.3% 94.7% +2.9%  | *0.1% 94.9% +2.9%
        | p<0.001 N=1239     | p=0.001 N=1036     | p=0.002 N=889      | p=0.001 N=799      | p=0.006 N=740      | p=0.004 N=646      | p=0.006 N=567
--------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------
75%     | *5.5% 92.0% +3.6%  | *1.8% 92.6% +3.1%  | *0.6% 93.3% +3.1%  | *0.7% 93.9% +3.2%  | *0.3% 93.8% +2.7%  | *0.0% 94.2% +2.9%  | *0.0% 94.2% +2.8%
        | p<0.001 N=1003     | p=0.002 N=828      | p=0.003 N=706      | p=0.003 N=626      | p=0.012 N=576      | p=0.010 N=499      | p=0.020 N=433
--------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------
80%     | *1.1% 91.7% +4.1%  | *0.4% 92.1% +3.6%  | *0.1% 92.8% +3.7%  | *0.0% 93.1% +3.7%  | *0.0% 92.6% +3.0%  | *0.0% 92.9% +3.2%  | *0.0% 93.0% +3.1%
        | p<0.001 N=810      | p=0.002 N=659      | p=0.003 N=556      | p=0.004 N=492      | p=0.020 N=446      | p=0.020 N=380      | p=0.030 N=328
```

**Legend**: * = Good (FP<25%, WR>93%, p<0.01)  ! = Warning (FP>35%, p>0.05)

### 10.4 Tier Analysis: Finding the Sweet Spot

**Tier 1 (Strictest)**: FP<25%, WR>93%, p<0.01, N>500

| Rank | YES> | Min Trades | FP Rate | Win Rate | Edge | 95% CI | p-value | Markets |
|------|------|------------|---------|----------|------|--------|---------|---------|
| 1 | **75%** | **25** | **0.7%** | **93.9%** | **+3.2%** | [1.4%, 5.1%] | 0.0027 | 626 |
| 2 | 70% | 25 | 2.2% | 94.1% | +3.0% | [1.4%, 4.7%] | 0.0014 | 799 |
| 3 | 65% | 25 | 5.4% | 94.7% | +2.8% | [1.4%, 4.2%] | 0.0007 | 963 |
| 4 | 70% | 20 | 2.1% | 93.5% | +2.9% | [1.2%, 4.5%] | 0.0017 | 889 |
| 5 | 65% | 20 | 5.8% | 94.1% | +2.8% | [1.4%, 4.2%] | 0.0006 | 1067 |

**Tier 2 (Moderate)**: FP<30%, WR>92%, p<0.05, N>300

Best: **YES>80%, min_trades=20** (FP=0.1%, WR=92.8%, Edge=3.7%, N=556)

**Tier 3 (Relaxed)**: FP<35%, WR>90%, p<0.05, N>100

Best: **YES>80%, min_trades=10** (FP=1.1%, WR=91.7%, Edge=4.1%, N=810)

### 10.5 Composite Reliability Ranking

Score = (1 - FP_rate) x 0.4 + Win_rate x 0.3 + (1 - p_value) x 0.2 + CI_width_inverse x 0.1

| Rank | YES> | Min Trades | FP Rate | Win Rate | Edge | p-value | Reliability Score |
|------|------|------------|---------|----------|------|---------|-------------------|
| 1 | 65% | 40 | 1.9% | 95.3% | +2.6% | 0.0025 | **0.924** |
| 2 | 65% | 50 | 1.6% | 95.4% | +2.5% | 0.0050 | 0.922 |
| 3 | 70% | 40 | 0.3% | 94.7% | +2.9% | 0.0035 | 0.920 |
| 4 | 70% | 50 | 0.1% | 94.9% | +2.9% | 0.0059 | 0.917 |
| 5 | 70% | 30 | 0.8% | 94.1% | +2.6% | 0.0061 | 0.916 |
| 6 | 70% | 25 | 2.2% | 94.1% | +3.0% | 0.0014 | 0.914 |
| 7 | 70% | 20 | 2.1% | 93.5% | +2.9% | 0.0017 | 0.913 |
| 8 | 65% | 25 | 5.4% | 94.7% | +2.8% | 0.0007 | 0.911 |
| 9 | 60% | 50 | 5.9% | 95.6% | +2.4% | 0.0036 | 0.911 |
| 10 | 75% | 25 | 0.7% | 93.9% | +3.2% | 0.0027 | 0.911 |

### 10.6 Current Configuration Assessment

**Current (YES>65%, min_trades=15)**:
- False Positive Rate: **15.1%** (not great - 1 in 7 triggers could be noise)
- Win Rate: 93.2%
- Edge: 2.7% [1.3%, 4.1%]
- p-value: 0.0005
- Markets: 1,225

### 10.7 Reliability-First Recommendations

#### RECOMMENDED: High Reliability Configuration

```bash
RLM_YES_THRESHOLD=0.70     # Increase from 0.65
RLM_MIN_TRADES=25          # Increase from 15
RLM_MIN_PRICE_DROP=2       # Keep at 2c
```

| Metric | Current (65%/15) | Recommended (70%/25) | Change |
|--------|------------------|----------------------|--------|
| False Positive Rate | 15.1% | **2.2%** | **-86%** |
| Win Rate | 93.2% | 94.1% | +0.9% |
| Edge | +2.7% | +3.0% | +0.3% |
| p-value | 0.0005 | 0.0014 | Similar |
| Markets | 1,225 | 799 | -35% |

**Why this recommendation?**
1. **7x lower false positive rate** - 2.2% vs 15.1%
2. **Better win rate** - 94.1% vs 93.2%
3. **Higher edge** - +3.0% vs +2.7%
4. **Still statistically significant** - p=0.0014 < 0.01
5. **Reasonable sample size** - 799 markets (plenty of signals)

#### ALTERNATIVE: Ultra-Conservative Configuration

```bash
RLM_YES_THRESHOLD=0.75
RLM_MIN_TRADES=25
RLM_MIN_PRICE_DROP=2
```

| Metric | Value |
|--------|-------|
| False Positive Rate | **0.7%** |
| Win Rate | 93.9% |
| Edge | +3.2% |
| 95% CI | [1.4%, 5.1%] |
| p-value | 0.0027 |
| Markets | 626 |

This is the Tier 1 winner - extremely low false positive rate at 0.7%.
Trade-off: 49% fewer signals than current config.

### 10.8 Trade-off Summary

| Config | FP Rate | Markets | Edge | Recommendation |
|--------|---------|---------|------|----------------|
| YES>55%, min=10 | 37.7% | 1,854 | +3.0% | **AVOID** - Too noisy |
| YES>60%, min=10 | 17.2% | 1,657 | +3.0% | Aggressive - Production risk |
| YES>65%, min=15 (current) | 15.1% | 1,225 | +2.7% | Acceptable for MVP |
| **YES>70%, min=25** | **2.2%** | **799** | **+3.0%** | **RECOMMENDED** |
| YES>75%, min=25 | 0.7% | 626 | +3.2% | Ultra-conservative |
| YES>80%, min=20 | 0.1% | 556 | +3.7% | Highest edge, fewest signals |

### 10.9 Key Takeaways

1. **Current config (YES>65%, min=15) has 15% false positive rate**
   - 1 in 7 signals could be random noise
   - Acceptable for MVP, but not ideal

2. **Moving to YES>70%, min=25 reduces false positives by 86%**
   - Only 2.2% chance of false trigger
   - Still get 799 markets (plenty of signals)
   - Higher edge (+3.0% vs +2.7%)

3. **The "aggressive" config (YES>60%, min=10) is dangerous**
   - 17.2% false positive rate
   - More signals but lower reliability
   - NOT recommended for production

4. **There's no free lunch**
   - Lower false positive rate = fewer signals
   - Choose based on risk tolerance vs signal frequency
   - For production, reliability > volume

---

## Appendix: Signal Parameter Grid (Top 15 by Total EV)

| YES> | MinT | Drop | Markets | WinRate | Edge | EV/$100 | Total EV |
|------|------|------|---------|---------|------|---------|----------|
| 60% | 10 | 3c | 1,547 | 94.0% | 23.2% | $32.75 | $50,662 |
| 60% | 10 | 2c | 1,636 | 92.9% | 21.8% | $30.71 | $50,241 |
| 60% | 10 | 0c | 1,978 | 91.0% | 18.3% | $25.13 | $49,701 |
| 60% | 10 | 5c | 1,404 | 94.7% | 24.4% | $34.77 | $48,810 |
| 60% | 15 | 3c | 1,311 | 94.7% | 24.0% | $33.84 | $44,361 |
| 60% | 15 | 2c | 1,380 | 93.6% | 22.5% | $31.71 | $43,766 |
| 65% | 10 | 3c | 1,357 | 93.9% | 22.8% | $32.08 | $43,537 |
| 60% | 15 | 5c | 1,203 | 95.6% | 25.4% | $36.10 | $43,431 |
| 60% | 15 | 0c | 1,603 | 92.0% | 19.6% | $27.04 | $43,350 |
| 65% | 10 | 2c | 1,443 | 92.8% | 21.4% | $30.02 | $43,314 |
| 65% | 10 | 5c | 1,225 | 94.6% | 24.1% | $34.14 | $41,816 |
| 70% | 10 | 3c | 1,159 | 93.7% | 22.8% | $32.08 | $37,182 |
| **65%** | **15** | **2c** | **1,207** | **93.4%** | **22.0%** | **$30.87** | **$37,264** (current) |
| 65% | 15 | 3c | 1,141 | 94.5% | 23.4% | $33.01 | $37,663 |
| 65% | 15 | 5c | 1,042 | 95.4% | 24.9% | $35.29 | $36,772 |
