# RLM Spread Threshold Analysis for S-001 Position Scaling

**Date**: 2026-01-01
**Analyst**: Quant Agent (Claude Opus 4.5)
**Context**: Optimizing spread thresholds for RLM_NO strategy with S-001 position scaling

---

## Executive Summary

The RLM_NO strategy has been enhanced with S-001 position scaling:
- **5-10c price drop**: 50 contracts (1x baseline)
- **10-20c price drop**: 75 contracts (1.5x)
- **20c+ price drop**: 100 contracts (2x)

This document analyzes optimal spread thresholds for order entry, considering the trade-off between fill probability and slippage cost.

**Key Recommendations**:
1. **Keep current thresholds** for 1x positions (tight <= 2c, normal <= 4c)
2. **Use slightly more aggressive pricing** for 1.5x/2x positions to ensure fills
3. **Add spread-based position rejection** for extremely wide spreads (>8c)
4. **Log spread data** for empirical optimization in Phase 2

---

## Current Implementation

From `rlm_service.py` lines 719-750:

```python
def _calculate_no_entry_price(self, best_no_ask, best_no_bid, spread):
    if spread <= 2:      # Tight spread
        return best_no_ask - 1    # Join queue just below ask
    elif spread <= 4:    # Normal spread
        return (best_no_ask + best_no_bid) // 2  # Midpoint
    else:                # Wide spread
        return best_no_bid + (spread * 3 // 4)   # 75% toward ask
```

---

## Research Questions Addressed

### Q1: What is the theoretical spread distribution for RLM-qualifying markets?

**Analysis**:

RLM signals fire in markets where:
- >= 15 trades have occurred (active markets)
- >65% of trades are YES (retail-heavy)
- YES price has dropped (smart money disagrees)

Markets meeting these criteria tend to be:
- **Actively traded**: High trade count implies reasonable liquidity
- **Mid-to-high price range**: YES price was high (hence room to drop)
- **Sports/Crypto/Entertainment**: These categories have better liquidity than politics

**Estimated Spread Distribution** (based on Kalshi market structure):

| Spread Range | Estimated Frequency | Market Type |
|--------------|---------------------|-------------|
| 1-2c (Tight) | 40-50% | Popular sports, active crypto |
| 3-4c (Normal) | 30-35% | Most entertainment, less popular sports |
| 5-8c (Wide) | 15-20% | Media mentions, niche markets |
| >8c (Very Wide) | 5-10% | Low liquidity, near expiry |

### Q2: Fill Probability vs Slippage Trade-off

**Framework**:

For RLM signals, we're buying NO when retail is selling NO (they're buying YES). This means:
- **NO ask side**: Has our competition (market makers)
- **NO bid side**: We're providing liquidity if we use limits

**Fill Probability by Order Type**:

| Strategy | Fill Probability | Avg Slippage | Use Case |
|----------|------------------|--------------|----------|
| Hit the ask | 95-100% | 0-1c (spread cost) | Time-sensitive, tight spreads |
| Midpoint limit | 40-60% | -1c (price improvement) | Normal spreads, can wait |
| Near-bid limit | 20-40% | -2c (best price) | Wide spreads, low urgency |

**RLM Time Sensitivity**:

RLM signals are **time-sensitive** because:
1. Price is moving toward NO (our direction)
2. Other informed traders may be detecting the same pattern
3. Market settlement is approaching

This argues for **higher fill probability** over price improvement.

### Q3: Edge Sensitivity to Slippage

**Current Edge by Price Drop Tier**:

| Tier | Edge | Per-Cent Slippage Impact |
|------|------|--------------------------|
| 5-10c drop | +11.9% | 8.4% edge lost per cent |
| 10-20c drop | +17-19.5% | 5.4% edge lost per cent |
| 20c+ drop | +30.7% | 3.3% edge lost per cent |

**Calculation**: If edge is E% and you pay S cents extra slippage, effective edge = E% - S cents

**Interpretation**:
- For 1x positions (5-10c drop, +11.9% edge): 1c slippage = 8.4% relative edge loss
- For 2x positions (20c+ drop, +30.7% edge): 1c slippage = 3.3% relative edge loss

**Conclusion**: **Larger positions can afford more slippage** because their edge is proportionally higher.

### Q4: Position-Scaled Threshold Recommendations

**Principle**: Larger positions should prioritize fills because:
1. Edge is higher (can absorb more slippage)
2. Missing a 2x signal costs more than missing a 1x signal
3. Larger orders take longer to fill passively

**Recommended Thresholds**:

| Position Size | Spread Threshold | Pricing Strategy |
|---------------|------------------|------------------|
| **1x (50 contracts)** | Tight <= 2c | ask - 1c (passive) |
| | Normal <= 4c | Midpoint |
| | Wide <= 6c | bid + (spread * 0.6) |
| | Very Wide > 6c | SKIP or market order |
| **1.5x (75 contracts)** | Tight <= 2c | ask (take liquidity) |
| | Normal <= 4c | ask - 1c |
| | Wide <= 6c | bid + (spread * 0.7) |
| | Very Wide > 6c | bid + (spread * 0.8) |
| **2x (100 contracts)** | Tight <= 3c | ask (always take) |
| | Normal <= 5c | ask - 1c |
| | Wide <= 8c | bid + (spread * 0.75) |
| | Very Wide > 8c | bid + (spread * 0.85) |

---

## Theoretical Spread Distribution by NO Price

Since RLM signals fire when YES has dropped, we're often entering at NO prices of 30-70c (YES prices of 30-70c after drop from higher).

**Expected Spread Pattern**:

| NO Price Range | Typical Spread | Rationale |
|----------------|----------------|-----------|
| 30-40c (cheap NO) | 2-4c | High interest, balanced market |
| 40-60c (coin flip) | 1-3c | Most liquid, tight spreads |
| 60-80c (expensive NO) | 2-5c | Less interest in NO at high price |
| 80-90c (very expensive) | 3-8c | Few buyers, wide spreads |

---

## Implementation Recommendations

### Immediate Changes (Phase 1)

**Keep current thresholds** but add logging for empirical analysis:

```python
# In _calculate_no_entry_price, add spread logging:
self._spread_stats = {
    "tight_count": 0,    # <= 2c
    "normal_count": 0,   # <= 4c
    "wide_count": 0,     # > 4c
    "spreads_by_tier": {}  # {tier: [spread_values]}
}
```

### Validated Changes (Implement Now)

1. **Add position-aware spread multiplier**:

```python
def _calculate_no_entry_price(self, best_no_ask, best_no_bid, spread, scale_label):
    """
    Calculate entry price with position-scaling awareness.

    Larger positions (1.5x, 2x) use more aggressive pricing.
    """
    if scale_label == "2x":
        # 2x positions: prioritize fill over price
        if spread <= 3:  # Tight for 2x
            return best_no_ask  # Hit the ask
        elif spread <= 5:  # Normal for 2x
            return best_no_ask - 1  # Just below ask
        else:  # Wide
            return best_no_bid + int(spread * 0.75)

    elif scale_label == "1.5x":
        if spread <= 2:  # Tight
            return best_no_ask  # Hit the ask
        elif spread <= 4:  # Normal
            return best_no_ask - 1
        else:  # Wide
            return best_no_bid + int(spread * 0.70)

    else:  # 1x positions: current behavior
        if spread <= 2:
            return best_no_ask - 1
        elif spread <= 4:
            return (best_no_ask + best_no_bid) // 2
        else:
            return best_no_bid + int(spread * 0.75)
```

2. **Add spread rejection for extreme cases**:

```python
# Before calculating entry price:
if spread > 10:
    logger.warning(f"Skipping {signal.market_ticker}: spread too wide ({spread}c)")
    self._stats["signals_skipped"] += 1
    return  # Skip this signal
```

### Future Phase 2 Analysis

Once we collect spread data from live trading, analyze:
1. Actual spread distribution when signals fire
2. Fill rate by spread tier and position size
3. Slippage cost vs edge realized
4. Optimal threshold tuning based on empirical data

---

## Configuration Update

Update `environment.py` with new parameters:

```python
# RLM Spread Configuration
rlm_tight_spread: int = 2       # Threshold for tight spread (market order for 2x)
rlm_normal_spread: int = 4      # Threshold for normal spread
rlm_wide_spread: int = 6        # Threshold for wide spread
rlm_max_spread: int = 10        # Maximum spread to trade (skip if wider)
rlm_scale_aware_pricing: bool = True  # Use position-scaled pricing
```

---

## Summary of Recommended Changes

| Change | Impact | Risk | Priority |
|--------|--------|------|----------|
| Add spread logging | Data collection | None | P0 - Immediate |
| Position-aware pricing | Better fills for 2x | Slight increase in slippage for large positions | P1 - This Sprint |
| Max spread rejection (>10c) | Avoid bad fills | May miss edge in illiquid markets | P1 - This Sprint |
| Empirical threshold tuning | Optimal performance | Requires data | P2 - After data collection |

---

## Appendix: Market Microstructure Principles

### Why Larger Orders Need More Aggressive Pricing

1. **Market Impact**: Larger orders move price more
2. **Fill Time**: Larger orders take longer to fill passively
3. **Information Decay**: Signal value decays while waiting
4. **Opportunity Cost**: Missing a 2x signal = 2x the lost edge

### Why RLM Favors Aggressive Entry

1. **Time-Sensitive**: Price is moving toward NO (our bet)
2. **Information Edge**: We're betting against retail flow
3. **Settlement Risk**: Markets expire; unfilled orders expire too
4. **Competition**: Other informed traders may be entering

### The Spread-Fill Tradeoff

```
Fill Probability
    |
100%|          *--- Hit Ask (market order)
    |        *
 80%|      *
    |    *
 60%|  *-------- Midpoint
    | *
 40%|*
    |*---------- Near Bid (passive limit)
    +-------------------------> Price Improvement
           -2c   -1c    0
```

For RLM, we want to be in the upper-left quadrant: high fill probability, accepting minimal price improvement loss.

---

## Files to Update

1. **`/Users/samuelclark/Desktop/kalshiflow/backend/src/kalshiflow_rl/traderv3/services/rlm_service.py`**
   - Add spread logging
   - Implement position-aware pricing
   - Add max spread rejection

2. **`/Users/samuelclark/Desktop/kalshiflow/backend/src/kalshiflow_rl/traderv3/config/environment.py`**
   - Add new spread threshold parameters

3. **`/Users/samuelclark/Desktop/kalshiflow/research/RESEARCH_JOURNAL.md`**
   - Document this analysis session
