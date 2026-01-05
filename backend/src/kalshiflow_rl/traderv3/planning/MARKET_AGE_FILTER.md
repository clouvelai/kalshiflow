# Market Age Filter Research Findings

> **Status**: PENDING DECISION
> **Date**: 2026-01-05
> **Source**: Quant analysis on ~7.9M trades, ~316K markets
> **Artifacts**: `research/analysis/rlm_market_age_analysis.py`, `research/reports/rlm_market_age_analysis.json`

---

## Decision Required

**Where should this filter live?**

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **Strategy** | Filter in RLM signal detection | Clean separation, strategy-specific | May reject good markets |
| **Market Discovery** | Don't subscribe to old markets | Saves orderbook slots | Limits other strategies |

---

## Key Finding: Maximum Age Filter Improves RLM Edge

| Filter | Win Rate | Edge | Coverage |
|--------|----------|------|----------|
| No filter | 74.5% | +1.1% | 100% |
| **max_age < 168hr (1 week)** | **82.9%** | **+10.8%** | 83% |
| max_age < 72hr (3 days) | 83.0% | +11.0% | 69% |

**Edge improvement: +9.7 percentage points**

---

## Critical Discovery: Minimum Age Filters HURT Performance

| Min Age | Win Rate | Edge | Verdict |
|---------|----------|------|---------|
| No filter | 74.5% | +1.1% | Baseline |
| >= 6hr | 73.7% | -0.3% | **WORSE** |
| >= 12hr | 72.9% | -1.2% | **WORSE** |
| >= 24hr | 71.6% | -2.9% | **WORSE** |

**Do NOT implement minimum age filters.**

---

## Why Long-Dated Markets (>1 week) Fail RLM

Markets open for 1+ weeks have **33.8% NO win rate** when RLM says bet NO:

| Category | Markets | NO Win Rate | Issue |
|----------|---------|-------------|-------|
| KXTEAMSINSB (Super Bowl) | 16 | 0% | Long-dated events |
| KXOSCARNOMPIC (Oscars) | 13 | 0% | Long-dated events |
| KXIPO (IPO predictions) | 9 | 11% | Speculative |
| KXFEDMENTION (Fed speech) | 13 | 15% | Unpredictable |

**Why RLM fails on these**:
1. **Sustained YES enthusiasm**: Super Bowl, Oscars have loyal YES bettors
2. **Price drops ≠ retail losing**: Can be informed value buying
3. **Different dynamics**: These markets attract sophisticated YES bettors

---

## Age Bucket Analysis

| Age Bucket | Markets | Win Rate | Edge |
|------------|---------|----------|------|
| 0-6hr | 1,008 | 77.5% | +6.2% |
| 6-24hr | 982 | 79.8% | +7.3% |
| **24-48hr** | **1,016** | **89.7%** | **+17.3%** |
| 48hr-1wk | 1,008 | 84.6% | +12.0% |
| **1wk+** | **834** | **33.8%** | **-45.4%** |

The 1wk+ bucket destroys overall performance.

---

## Implementation Options

### Option A: Strategy Filter (RLM Service)

```python
# In rlm_service.py _detect_signal()
if tracked and tracked.open_ts > 0:
    market_age_hours = (now_ts - tracked.open_ts) / 3600
    if market_age_hours > self._max_market_age_hours:
        self._stats["signals_skipped_too_old"] += 1
        return None
```

**Pros**:
- Strategy-specific (other strategies may want old markets)
- Clean separation of concerns
- Easy to tune per-strategy

**Cons**:
- Still subscribing to orderbooks for markets we'll never trade

### Option B: Market Discovery Filter

```python
# In market discovery/subscription logic
if market_age_hours > MAX_MARKET_AGE_HOURS:
    # Don't subscribe to this market
    continue
```

**Pros**:
- Saves orderbook subscription slots
- System-wide efficiency

**Cons**:
- Affects all strategies uniformly
- Less flexible

### Option C: Hybrid (Category-Based)

```python
# Long-dated categories get special handling
LONG_DATED_CATEGORIES = ["KXTEAMSINSB", "KXOSCARNOMPIC", "KXIPO"]
if category in LONG_DATED_CATEGORIES:
    # Skip or use different strategy
```

**Pros**:
- Targeted filtering
- Preserves markets for other strategies

**Cons**:
- Hardcoded category list needs maintenance

---

## Recommendation

**Option A (Strategy Filter)** is recommended because:
1. RLM-specific issue - other strategies may work differently on old markets
2. Future strategies (S013, etc.) should have independent age analysis
3. Clean separation of concerns
4. Easy to rollback/tune

---

## Next Steps

1. [ ] Decision: Strategy vs Market Discovery vs Hybrid
2. [ ] Implementation based on decision
3. [ ] Add `market_age_hours` to signal logging for monitoring
4. [ ] Validate edge improvement in paper trading
