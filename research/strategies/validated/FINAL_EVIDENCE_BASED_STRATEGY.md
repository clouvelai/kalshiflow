# FINAL Evidence-Based Trading Strategy

**Analysis Date**: 2025-12-28
**Data**: 90,240 resolved trades (Dec 8-27, 2025)
**Markets Analyzed**: 7,931 unique markets
**Methodology**: Actual P/L using Kalshi API settlement data

---

## CRITICAL FINDING

**The winning signal is NOT what we initially thought.**

Partial data (27k trades) suggested: "Follow whale YES at 50-70c"
Full data (90k trades) proves: **Follow whale NO at 30-50c**

---

## The Winning Strategy

### Whale NO at 30-50c

**Definition**: Follow NO-side trades with >=100 contracts where price is 30-50 cents

| Metric | Value |
|--------|-------|
| Sample Size | 1,614 trades |
| Win Rate | 54.3% |
| Breakeven Win Rate | 40.2% |
| **Edge** | **+14.1%** |
| **ROI** | **+76.2%** |

### Why NO Side Outperforms YES

Full dataset side analysis:

| Side | Trades | Win Rate | ROI |
|------|--------|----------|-----|
| **NO** | 25,107 | 54.5% | **+20.5%** |
| YES | 65,133 | 40.5% | **-6.5%** |

**Hypothesis**: YES-side bettors are more likely to be retail/gamblers betting on outcomes they want. NO-side bettors are more likely to be informed/hedging.

---

## Secondary Signals

### Whale NO at 30-70c (Broader Range)

| Metric | Value |
|--------|-------|
| Sample Size | 2,827 trades |
| Edge | +11.6% |
| ROI | +43.0% |

### KXEP Category (Politics/Elections)

| Metric | Value |
|--------|-------|
| Sample Size | 914 trades |
| Edge | +21.5% |
| ROI | +47.6% |

### Whale Favorites (>=85c) - Conservative

| Metric | Value |
|--------|-------|
| Sample Size | 1,990 trades |
| Win Rate | 96.0% |
| Edge | +3.0% |
| ROI | +3.3% |
| Profit Factor | 1.81 |

---

## LOSING STRATEGIES (AVOID!)

### Whale YES at ANY Moderate Price

| Pattern | Trades | Edge | ROI |
|---------|--------|------|-----|
| Whale YES at 30-50c | 4,605 | -1.6% | -6.8% |
| Whale YES at 50-70c | 3,865 | -7.2% | -7.9% |
| Whale YES at 30-70c | 8,279 | -3.6% | -6.4% |

### Whale Longshots (ANY Side)

| Pattern | Trades | Edge | ROI |
|---------|--------|------|-----|
| Whale Longshots (<=15c) | 5,592 | N/A | -72.7% |
| Mega Longshots (<=20c) | 1,894 | N/A | -72.3% |

### Contrarian Bets

| Pattern | Trades | Edge |
|---------|--------|------|
| Against 85-99% consensus | 5,496 | -2.1% |

---

## Implementation for V3 Trader

### Core Signal Definition

```python
def is_high_conviction_whale(trade):
    """
    THE winning signal based on 90k resolved trades.
    """
    return (
        trade.count >= 100 and           # Meaningful size
        30 <= trade.price <= 50 and      # Optimal price range
        trade.taker_side == 'no'         # NO side has massive edge
    )
```

### Category Boost

```python
def is_politics_whale(trade):
    """
    Politics/election markets have extra edge.
    """
    return (
        trade.count >= 100 and
        trade.market_ticker.startswith('KXEP')
    )
```

### Conservative Alternative

```python
def is_safe_whale(trade):
    """
    High win rate, lower variance.
    """
    return (
        trade.count >= 100 and
        trade.price >= 85
    )
```

---

## Position Sizing

Based on edge and sample size:

| Strategy | Max Position | Expected Edge |
|----------|-------------|---------------|
| Whale NO 30-50c | 2% of bankroll | ~10-15% (after slippage) |
| Politics whale | 2% of bankroll | ~15-20% |
| Whale Favorites | 3% of bankroll | ~2-3% |

---

## Risk Management

1. **NEVER follow YES-side whale trades** - they lose money
2. **NEVER follow longshot whales** - they lose 70%+
3. **Daily loss limit**: 5% of bankroll
4. **Per-market limit**: 2% of bankroll

---

## Timing Considerations

| Window | Edge | Use |
|--------|------|-----|
| 3-12 hours before close | +1.4% | Slight timing bonus |
| 12+ hours before close | Negative | Avoid early trades |

---

## Data Quality Notes

1. **Sample period**: 3 weeks (Dec 8-27, 2025)
2. **Resolution rate**: 90.2% of trades resolved
3. **Market types**: Primarily sports, some politics
4. **Limitation**: Short time window, may not generalize

---

## Files Reference

| File | Description |
|------|-------------|
| `training/reports/historical_trades_full.csv` | 100k raw trades |
| `training/reports/market_outcomes.csv` | 7,931 market settlements |
| `training/reports/enriched_trades_final.csv` | Trades with outcomes |
| `training/reports/backtest_report.txt` | Strategy backtest |
| `training/reports/advanced_pattern_analysis.txt` | Pattern analysis |

---

## KEY TAKEAWAYS

1. **Follow whale NO bets at 30-50c** - This is the profitable signal
2. **Avoid whale YES bets** - They consistently lose money
3. **Avoid longshots entirely** - They are gambling, not informed
4. **Politics markets have extra edge** - KXEP category
5. **Timing helps slightly** - 3-12 hours before close is optimal

---

## Next Steps

1. Implement NO-side whale tracking in V3 trader
2. Add category-based signal boosting
3. Monitor ongoing performance
4. Expand data collection for longer validation period
