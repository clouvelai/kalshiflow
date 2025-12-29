# Session 002 Research Findings

**Date**: 2025-12-29
**Analyst**: Quant Agent (Opus 4.5)
**Dataset**: 1,619,902 trades across 72,791 markets

---

## Executive Summary

After exhaustive hypothesis testing of 8+ strategies using proper market-level validation, the key finding is:

**THE EXISTING PRICE-BASED STRATEGIES REMAIN THE BEST APPROACH**

No whale-based, time-based, or pattern-based enhancement provides statistically significant improvement over the validated price strategies.

---

## Hypotheses Tested and Results

### H007: Fade Whale Consensus (Contrarian)

**Initial Finding**: Fading 100% whale consensus shows 72.4% win rate, +22.4% edge vs 50%

**Deep Dive Revealed**:
- The edge is NOT from whale behavior - it's from PRICE
- At 85-100c: Fading whales = Betting NO = Same as existing NO strategy
- At 0-30c: Fading whales = Betting YES longshots = LOSES money
- Adding whale filter to price strategy: Only +1-4% marginal improvement
- BUT: Reduces sample size significantly and fails concentration tests

**Conclusion**: REJECTED. Whale consensus does not add meaningful edge. The apparent pattern is just a proxy for price-based strategies.

---

### H005: Time-of-Day Patterns

**Validated Patterns** (statistically significant but NEGATIVE edge):
| Strategy | Markets | Win Rate | Edge | Profit |
|----------|---------|----------|------|--------|
| NO at hour 07 | 612 | 57.0% | +15.7% | -$37,141 |
| YES at hour 04 | 2,257 | 24.7% | -15.7% | -$64,719 |
| YES during Late Morning | 12,439 | 18.2% | -24.3% | -$553,120 |

**Conclusion**: WEAK PATTERNS. Time of day shows some statistical significance but:
- Positive edge patterns have negative profit
- Large sample patterns show negative edge (YES morning trades lose money)
- No actionable time-based strategy identified

---

### H006: Category-Specific Efficiency

**Top Category Patterns** (validated with markets >= 30):
| Category | Strategy | Markets | Edge | Profit |
|----------|----------|---------|------|--------|
| KXNCAAFTOTAL | NO at 70-80c | 62 | +68.3% | $25,790 |
| KXNFLSPREAD | NO at 70-80c | 147 | +41.0% | $63,341 |
| KXNCAAFTOTAL | NO at 80-90c | 61 | +79.3% | $19,825 |
| KXBTCD | NO at 90-100c | 186 | +88.4% | $18,383 |

**Conclusion**: POTENTIALLY USEFUL. Some categories (NCAAF totals, NFL spreads, crypto) show higher edge than base strategies. However:
- Sample sizes are small (50-200 markets)
- These are SUBSETS of the general NO at 80-90c strategy
- Need more data to validate category-specific approaches

---

### H008: New Market Mispricing (Early Trades)

**Finding**: Early trades show significant patterns:
- First 1 trade YES: -2.8% edge (68,857 markets)
- First 30 min NO: +20.1% edge (5,394 markets)
- First 60 min NO: +19.7% edge (5,734 markets)

**Interpretation**:
- YES trades early = Retail rushing to bet favorites
- NO trades early = More sophisticated positioning
- Early NO trades have edge but it's the SAME edge as general NO strategy

**Conclusion**: INTERESTING BUT NOT ACTIONABLE. Early timing doesn't add edge over price-based strategies when properly controlled.

---

### H009: Price Velocity / Momentum

**Finding**:
- Momentum (follow price moves): FAILS validation, negative edge
- Mean reversion (fade price moves): Marginal edge (+1-2%) but NEGATIVE profit

**Conclusion**: REJECTED. Price momentum/reversion signals do not provide tradeable edge.

---

### H010: Trade Sequencing Patterns

**Finding**: After consecutive same-side trades:
- After 2+ YES trades, bet NO: +14.6% edge but FAILS concentration test
- After 3+ YES trades, bet NO: +13.9% edge but FAILS concentration test

**Conclusion**: REJECTED. Sequencing patterns fail concentration validation.

---

### H011: Volume Patterns

**Finding**: High-volume markets (top 25%) show similar edge to base strategies:
- High volume NO at 90-100c: +85.9% edge (3,083 markets) - VALIDATED
- High volume NO at 80-90c: +65.4% edge (2,250 markets) - VALIDATED

**Conclusion**: CONFIRMS BASE STRATEGY. Volume filtering doesn't improve edge; it just confirms that the base strategy works across volume levels.

---

### H012: Round Number Effects

**Finding**:
- Trades at exactly 25c, 50c, 75c show some clustering
- No tradeable edge identified at round numbers
- YES near 25c: -4.1% edge (negative)
- NO near 75c: +49.6% edge but fails validation

**Conclusion**: REJECTED. Round numbers don't provide actionable edge.

---

## Final Strategy Recommendations

### Tier 1: KEEP USING (Validated, High Confidence)

| Strategy | Markets | Win Rate | Edge | Profit | Status |
|----------|---------|----------|------|--------|--------|
| YES at 80-90c | 2,110 | 88.9% | +5.1% | $1.6M | VALIDATED |
| NO at 80-90c | 2,808 | 87.8% | +3.3% | $708k | VALIDATED |
| NO at 90-100c | 4,741 | 96.5% | +1.2% | $463k | VALIDATED |

### Tier 2: CONSIDER (Category Enhancements)

If seeking higher edge at cost of lower sample:
- KXNCAAFTOTAL NO at 70-80c: +68.3% edge, $25k profit
- KXNFLSPREAD NO at 70-80c: +41.0% edge, $63k profit
- KXBTCD NO at 80-90c: +72.4% edge, $13k profit

### Tier 3: DO NOT IMPLEMENT

| Pattern | Reason |
|---------|--------|
| Whale following | Fails concentration |
| Whale consensus fading | No improvement over price |
| Time-of-day strategies | Inconsistent edge |
| Price momentum/reversion | Negative profit |
| Trade sequencing | Fails concentration |
| Round number effects | No edge found |

---

## Key Learnings

### 1. Price is the Primary Signal
All validated strategies are price-based. Every "enhancement" (whale, time, volume) is really just a proxy for price.

### 2. Whale Behavior is Not Informative
- Whales are NOT informed traders
- Following them loses money
- Fading them = betting high-price favorites = same as base strategy

### 3. The Longshot Bias is the Edge
The consistent pattern across all analysis:
- YES bets at low prices (<50c) systematically lose
- NO bets at high prices (>80c) systematically win
- This is the "favorite-longshot bias" - retail overpays for underdogs

### 4. Complexity Does Not Help
Every attempt to add complexity (whale signals, timing, sequencing) either:
- Failed validation criteria, OR
- Provided no improvement over simple price rules

---

## Implementation Notes for V3 Trader

### Current YES_80_90 Strategy
**STATUS**: KEEP AS IS

The YES at 80-90c strategy (+5.1% edge) remains the best balance of:
- Edge (highest among validated)
- Sample size (2,110 markets)
- Implementation simplicity

### Potential Additions

1. **NO at 80-90c**: Add as complementary strategy
   - Diversifies across YES/NO sides
   - Same price monitoring
   - +3.3% edge validated

2. **Category Filter** (Optional Phase 2):
   - Focus on NCAAF totals, NFL spreads, crypto
   - Higher edge but lower frequency

### Do NOT Add
- Whale tracking/following
- Time-based entry rules
- Momentum indicators
- Trade sequence patterns

---

## Data and Scripts

### Analysis Scripts Created
- `/research/analysis/session002_deep_analysis.py` - Main hypothesis testing
- `/research/analysis/session002_whale_fade_deep_dive.py` - Whale consensus validation
- `/research/analysis/session002_combined_strategy.py` - Price vs whale comparison

### Data Used
- `research/data/trades/enriched_trades_resolved_ALL.csv` - 1.62M trades
- `research/data/markets/market_outcomes_ALL.csv` - 76,858 markets

---

## Next Steps

1. **Continue Data Collection**: Current data (Dec 5-27, 2025) covers ~3 weeks. Need 3+ months for robust category validation.

2. **Execution Optimization**: Focus research on:
   - Slippage measurement
   - Optimal order placement timing
   - Position sizing

3. **Paper Trading Validation**: Track YES_80_90 and NO_80_90 performance in live paper trading.

---

*Report generated by Session 002 research*
*Statistical methodology: Market-level validation with concentration risk checks*
*Validation criteria: N >= 50 markets, concentration < 30%, p-value < 0.05*
