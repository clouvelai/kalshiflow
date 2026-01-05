# Sports Betting Expert Hypothesis Briefs

**Generated**: 2026-01-01
**Source**: 20+ years sports betting expertise translated to prediction markets
**Agent**: Strategy Researcher (Claude Opus 4.5)
**Status**: INCOMING - Ready for LSD screening

---

## Executive Context: What Already Works

Before generating novel hypotheses, I reviewed all existing research:

**VALIDATED STRATEGIES (Do Not Duplicate):**
| Strategy | Edge | Mechanism |
|----------|------|-----------|
| RLM_NO (H123) | +17-24% | >65% trades YES but price drops -> bet NO |
| Dollar-Weighted (H-LSD-207) | +12% | Trades favor YES but dollars favor NO |
| Conviction Ratio (H-LSD-211) | +11.75% | NO trades 2x bigger than YES |
| Size Gradient (H-LSD-209) | +10% | Larger trades correlate with NO |
| Buyback Reversal (EXT-005) | +10.65% | First half YES, second half NO with larger size |

**CONFIRMED FAILURES (Do Not Retry):**
- Price-based signals (NO at 50-80c, etc.) - all near breakeven
- Time-of-day patterns (general) - weak/not actionable
- CLV (Closing Line Value) - Kalshi doesn't behave like sportsbooks
- Order flow direction alone (without size weighting)
- Individual bot detection (too noisy)

**KEY INSIGHT:** BEHAVIORAL and CAPITAL FLOW signals work. PRICE-LEVEL and TIME signals don't.

---

## DIMENSION 1: Momentum/Steam Patterns

### SPORTS-001: Steam Exhaustion Detection

**Source**: Sharp sports betting - steam moves eventually exhaust when all sharps have bet

**Signal**:
- Identify "steam" condition: 5+ consecutive same-direction trades in <60 seconds
- If steam has moved price >10c
- If 3+ minutes have passed with NO further same-direction trades
- The "steam" has exhausted

**Bet**: FADE the steam direction (if steam was YES trades, bet NO)

**Mechanism**:
In sports betting, when a line "steams" (moves sharply on sharp action), the first followers capture value but late followers are already getting worse prices. When the steam STOPS and price hasn't reverted, it often means:
1. All informed bettors have bet
2. Price has fully adjusted
3. Late retail FOMO creates fade opportunity

The key insight is detecting the STOP of momentum, not the momentum itself.

**Expected Edge**: +6-10%
**Independence from RLM**: HIGH - RLM looks at total YES ratio, this looks at TIMING/SEQUENCING of the steam
**Data Requirement**: timestamp_ms, taker_side, yes_price (to calculate inter-trade gaps and directional runs)
**Risk**: May require precise timing data; steam detection could be noisy

---

### SPORTS-002: Opening Move Reversal (Fade the Opener)

**Source**: Sports betting - first line releases often overcompensate, sharps fade the opener

**Signal**:
- Identify markets where YES price moved >10c in first 25% of trades
- Compare to second 25% of trades
- If direction REVERSED (first quarter YES-heavy, second quarter NO-heavy)
- This is an "opening move reversal"

**Bet**: Follow the REVERSAL direction (bet NO if opener was YES-heavy)

**Mechanism**:
Opening line releases in sports betting are often set conservatively. The first wave of bets creates overreaction. Sharp bettors then fade this overreaction, causing reversal. The REVERSAL is the informed trade, not the opening move.

This differs from "Buyback Reversal" (EXT-005) which uses halves - this specifically targets the OPENING QUARTER.

**Expected Edge**: +5-8%
**Independence from RLM**: MEDIUM - Both capture reversal, but this focuses on early vs later, not ratio vs price
**Data Requirement**: datetime, taker_side, yes_price (to segment by time quartiles)
**Risk**: May overlap with Buyback Reversal; need to test if specifically targeting opener adds value

---

### SPORTS-003: Momentum Velocity Stall

**Source**: Technical trading - momentum stall patterns precede reversals

**Signal**:
- Calculate "velocity" of YES trades: YES_trades_per_minute
- Track velocity over market life in thirds
- If velocity DROPPED by >50% from first third to second third
- But price stayed high (YES price >70c)

**Bet**: Bet NO (velocity stall at high price = bearish divergence)

**Mechanism**:
When YES momentum is DECELERATING but price hasn't adjusted, it's a divergence signal. The market is "running out of YES buyers" but sellers haven't arrived yet. This is a leading indicator of reversal.

In sports betting, when you see line movement slow but not reverse, it often precedes a sharp correction.

**Expected Edge**: +4-7%
**Independence from RLM**: HIGH - This measures ACCELERATION, RLM measures cumulative ratio
**Data Requirement**: timestamp_ms, taker_side (to calculate trade velocity)
**Risk**: Velocity calculation may be noisy with sparse trade data

---

## DIMENSION 2: Public/Retail Tells

### SPORTS-004: Extreme Public Sentiment Fade

**Source**: Sports betting - when 90%+ of bets are on one side, fade the public

**Signal**:
- YES trade ratio > 90% (extreme public sentiment on YES)
- AND market has > 10 trades (not thin market)
- AND price is NOT at extreme (YES price between 40-80c, not a "sure thing")

**Bet**: Bet NO (fade extreme public consensus at non-extreme prices)

**Mechanism**:
The "90/10 rule" in sports betting: when the public is 90%+ on one side but the line HASN'T moved to reflect this (price still moderate), it means the book is comfortable taking the action because they believe the public is wrong.

In prediction markets, this translates to: if 90%+ of trades are YES but YES price is still 40-80c, the market is disagreeing with the public through SIZE (fewer but larger NO trades).

**Expected Edge**: +8-15%
**Independence from RLM**: LOW-MEDIUM - This is a more extreme version of RLM concept
**Data Requirement**: taker_side, yes_price, count (market filter)
**Risk**: May simply be a stricter version of RLM; overlap likely

---

### SPORTS-005: Size Velocity Divergence (Retail Pile-On Detection)

**Source**: Market microstructure - when trade count surges but average size drops, it's retail

**Signal**:
- Trade frequency INCREASING (more trades per hour in recent period vs earlier)
- BUT average trade size DECREASING (smaller average count in recent period)
- This divergence indicates retail pile-on

**Bet**: FADE the direction of the increased frequency (bet opposite of what retail is doing)

**Mechanism**:
When trade velocity goes up but average size goes down, it means:
- Retail traders are arriving (small, frequent trades)
- Smart money has already positioned (their large trades are earlier)
- The current momentum is "noise" from unsophisticated money

This is similar to "Conviction Ratio" but measures CHANGE in patterns, not absolute levels.

**Expected Edge**: +5-9%
**Independence from RLM**: HIGH - This measures temporal CHANGE in patterns, not static ratios
**Data Requirement**: timestamp_ms, count (trade size), taker_side
**Risk**: Requires temporal analysis; may need significant trade count to detect

---

### SPORTS-006: Round Number Retail Clustering

**Source**: Behavioral economics - retail bets cluster at round prices

**Signal**:
- Calculate percentage of trades at "round" YES prices: 25c, 50c, 75c (within 2c)
- If round_number_ratio > 40% of trades
- This indicates retail-dominated market

**Bet**: If round-number ratio is HIGH and net direction is YES, bet NO (fade retail)

**Mechanism**:
Retail traders are attracted to psychologically significant prices. When a market has high concentration of trades at round numbers, it indicates:
- Retail dominance (sophisticated traders don't care about round numbers)
- Anchoring bias (traders betting because "50c feels right" not because of edge)
- Opportunity to fade the retail anchor

This differs from previous "round number" tests that looked at MARKET price, not TRADE clustering.

**Expected Edge**: +4-7%
**Independence from RLM**: HIGH - Measures price clustering, not directional ratio
**Data Requirement**: yes_price (to detect round number clustering)
**Risk**: May have insufficient sample if trades spread across many prices

---

## DIMENSION 3: Sharp Money Signals (Beyond RLM)

### SPORTS-007: Late-Arriving Large Money (Closing Line Hunt)

**Source**: Sports betting - sharp money arrives closest to event start

**Signal**:
- Identify the FINAL 25% of trades (by time, not count)
- Calculate ratio of large trades (>$50 value) in final 25% vs earlier periods
- If final_large_ratio > 2x earlier_large_ratio
- Large money is arriving late (sharp characteristic)

**Bet**: Follow the DIRECTION of the late large trades

**Mechanism**:
In sports betting, the "sharpest" money comes in closest to kickoff because:
1. Information crystallizes (injuries, weather, lineup confirmations)
2. Sharps want to minimize exposure time
3. Early retail money has already moved the line for them to exploit

If late-arriving money is disproportionately large AND going one direction, follow it.

**Expected Edge**: +6-10%
**Independence from RLM**: MEDIUM - Both detect smart money, but this uses TIMING as key dimension
**Data Requirement**: timestamp_ms, trade_value_cents, taker_side
**Risk**: Needs expiration time data to know "how close to event"; may not have this

---

### SPORTS-008: Size Distribution Shape Change

**Source**: Market microstructure - uniform vs bimodal size distributions indicate different regimes

**Signal**:
- Calculate size distribution of trades in first half vs second half
- If first half was UNIFORM (varied sizes) but second half is CLUSTERED (similar sizes)
- OR if size distribution went from unimodal to bimodal
- Regime change detected

**Bet**: Follow the direction of the NEW regime (if second half trades cluster on NO, bet NO)

**Mechanism**:
When trade size distribution changes shape during a market's life, it indicates a "regime shift":
- Uniform -> Clustered: A dominant player (bot or whale) has entered
- Unimodal -> Bimodal: Both retail (small) and institutions (large) now active

The NEW regime represents the "informed" state, especially if sizes cluster.

**Expected Edge**: +5-8%
**Independence from RLM**: HIGH - Measures distribution SHAPE change, not direction ratios
**Data Requirement**: count (trade size), timestamp_ms (for temporal split)
**Risk**: Distribution analysis may require more trades than available

---

### SPORTS-009: Spread Widening Before Sharp Entry

**Source**: Market microstructure - market makers widen spreads when they anticipate informed flow

**Signal**:
- Track the "effective spread" (price change per trade)
- If spread WIDENED (larger price moves per trade) in a specific period
- Followed by a large trade in ONE direction
- The large trade after spread widening is likely informed

**Bet**: Follow the direction of the large trade that came after spread widening

**Mechanism**:
Market makers in traditional finance widen spreads when they expect informed traders. In prediction markets, we can detect "spread widening" by looking at price volatility per trade. If a period of wider moves is followed by a large directional bet, that bet is likely informed.

This is detecting the "preparation" for smart money, not just the smart money itself.

**Expected Edge**: +5-9%
**Independence from RLM**: HIGH - Measures spread dynamics, completely different dimension
**Data Requirement**: yes_price, count, timestamp_ms (to calculate per-trade price moves)
**Risk**: Kalshi doesn't have true orderbook, so "spread" is inferred from price changes

---

## DIMENSION 4: Cross-Market/Structural

### SPORTS-010: Multi-Outcome Pricing Inconsistency

**Source**: Academic arbitrage research - related markets should sum to 100%

**Signal**:
- For markets on SAME event (e.g., "Will team score over/under X")
- If complementary markets don't sum to ~100% (more than 5% deviation)
- One market is mispriced

**Bet**: Bet the CHEAPER side (if sum > 105%, bet the underpriced outcome)

**Mechanism**:
Complementary markets (YES/NO on same outcome from different framings) should sum to 100%. When they don't, there's either:
1. Temporary mispricing (arb opportunity)
2. Liquidity imbalance (one side has more action)

This was "rejected" in Session 008 as "multi-leg by design" - but we should retest with PURE complements, not legs of different events.

**Expected Edge**: +3-6%
**Independence from RLM**: COMPLETE - This is arbitrage, not directional
**Data Requirement**: market_ticker (to identify related markets), yes_price
**Risk**: Need to identify truly complementary markets; may be rare

---

### SPORTS-011: Category Momentum Contagion

**Source**: Behavioral finance - category-level momentum creates spillover effects

**Signal**:
- Track recent resolution outcomes by CATEGORY (e.g., NFL)
- If last 3+ resolutions in category went AGAINST the favorite (underdogs won)
- Bettors in NEXT market in category will OVERCORRECT (overbet underdogs)

**Bet**: Bet the FAVORITE in next market after underdog streak (fade the category recency bias)

**Mechanism**:
After a streak of "upsets" in a category, bettors suffer from RECENCY BIAS:
- "NFL favorites are losing this week" -> Bet more underdogs
- This creates overcrowding on the newly-fashionable side
- The CORRECTION is to fade the recency bias

**Expected Edge**: +4-8%
**Independence from RLM**: COMPLETE - Cross-market category-level signal
**Data Requirement**: category, market_result, close_time (to sequence resolutions)
**Risk**: Requires sequential resolution data; may need market metadata

---

### SPORTS-012: NCAAF Totals Specialist (Category Drilling)

**Source**: Session 009 finding - NCAAFTOTAL showed +22.5% edge but insufficient sample

**Signal**:
- NCAAF totals (over/under) markets specifically
- Apply base strategy (bet NO when appropriate)
- OPTIONAL: Combine with Dollar-Weighted or Conviction Ratio for higher confidence

**Bet**: Prioritize NCAAFTOTAL markets when other signals fire

**Mechanism**:
NCAAFTOTAL markets showed the highest raw edge in category analysis (+22.5%). This suggests:
- NCAAF totals attract less sophisticated money
- The "totals" framing may create behavioral biases (over/under feels like gambling)
- Worth specifically tracking and scaling up when other signals align

**Expected Edge**: +15-25% (when combined with other signals)
**Independence from RLM**: PARTIAL - Category-specific application of existing signals
**Data Requirement**: category (specifically NCAAFTOTAL markets)
**Risk**: Small sample size (only 94 markets); may be data-mining artifact

---

## DIMENSION 5: Absurd/LSD-Style Ideas

### SPORTS-013: Trade Count Milestone Fading

**Source**: Behavioral/numerological - round number trade counts attract attention

**Signal**:
- When market hits exactly 100 trades, 500 trades, or 1000 trades
- Track the DIRECTION of the milestone trade
- And track the NEXT 5 trades after milestone

**Bet**: If milestone trade was YES and next 5 trend NO, bet NO (fade the milestone momentum)

**Mechanism**:
This is genuinely speculative: round-number trade counts may attract attention/commentary, leading to retail pile-ons. The trades AFTER a milestone may represent more informed repositioning.

Or it could be complete noise. That's LSD mode!

**Expected Edge**: +3-7% (speculative)
**Independence from RLM**: COMPLETE - Numerological/structural signal
**Data Requirement**: count (to track number of trades)
**Risk**: This is probably noise; test quickly and move on

---

### SPORTS-014: Bot Signature Fade (Clock-Like Trading)

**Source**: Session 011 bot research - clock-like inter-arrival patterns indicate bots

**Signal**:
- Calculate coefficient of variation (CV) of inter-trade times
- If CV < 0.3 (very consistent timing), market is bot-dominated
- Track the DIRECTION bots are trading

**Bet**: FADE the bot direction (bet opposite)

**Mechanism**:
Previous research found bots but couldn't extract edge from FOLLOWING them. What about FADING them?

Bots may be:
- Market makers (neutral, creating noise)
- Arbitrageurs (capturing small edges, may be wrong about direction)
- Over-optimizing for fills, not edge

If bots are NOISE, fading them may extract signal from what remains.

**Expected Edge**: +4-8% (speculative)
**Independence from RLM**: HIGH - Uses timing patterns, not size/direction
**Data Requirement**: timestamp_ms (to calculate inter-arrival CV)
**Risk**: Bot detection was noisy before; may not work

---

### SPORTS-015: Fibonacci Price Attractors (Magic Levels)

**Source**: Technical analysis/numerology - Fib levels act as support/resistance

**Signal**:
- Calculate if YES price is near Fibonacci ratios: 23.6c, 38.2c, 50c, 61.8c, 76.4c
- Within 2c of these levels
- Track if price BOUNCES off these levels or BREAKS through

**Bet**:
- If price bounced off level, bet toward the bounce direction
- If price broke through level after multiple touches, follow the breakout

**Mechanism**:
This is technical analysis applied to prediction markets. Fibonacci levels are self-fulfilling prophecies in equity markets because traders EXPECT them to matter. If Kalshi traders use similar mental models, these levels may act as attractors.

The validated "Triple Weird Stack" (WILD-010) used Fibonacci TRADE COUNTS. This uses Fibonacci PRICE LEVELS.

**Expected Edge**: +2-5% (very speculative)
**Independence from RLM**: COMPLETE - Price level technical analysis
**Data Requirement**: yes_price (to detect proximity to Fib levels)
**Risk**: Probably complete noise; quick test only

---

## Summary Table

| ID | Name | Source Concept | Bet Direction | Expected Edge | RLM Independence |
|----|------|---------------|---------------|---------------|------------------|
| SPORTS-001 | Steam Exhaustion | Momentum exhaustion | Fade steam | +6-10% | HIGH |
| SPORTS-002 | Opening Move Reversal | Fade the opener | Follow reversal | +5-8% | MEDIUM |
| SPORTS-003 | Momentum Velocity Stall | Bearish divergence | Bet NO | +4-7% | HIGH |
| SPORTS-004 | Extreme Public Fade | 90/10 rule | Bet NO | +8-15% | LOW-MEDIUM |
| SPORTS-005 | Size Velocity Divergence | Retail pile-on | Fade retail | +5-9% | HIGH |
| SPORTS-006 | Round Number Clustering | Anchoring bias | Fade retail | +4-7% | HIGH |
| SPORTS-007 | Late-Arriving Large | Closing line hunt | Follow late money | +6-10% | MEDIUM |
| SPORTS-008 | Size Distribution Change | Regime shift | Follow new regime | +5-8% | HIGH |
| SPORTS-009 | Spread Widening Before Sharp | MM anticipation | Follow post-widen | +5-9% | HIGH |
| SPORTS-010 | Multi-Outcome Inconsistency | Arbitrage | Bet cheaper side | +3-6% | COMPLETE |
| SPORTS-011 | Category Momentum Contagion | Recency bias fade | Bet favorite | +4-8% | COMPLETE |
| SPORTS-012 | NCAAF Totals Specialist | Category drilling | Prioritize NCAAFTOTAL | +15-25% | PARTIAL |
| SPORTS-013 | Trade Count Milestone | Numerological | Fade milestone | +3-7% | COMPLETE |
| SPORTS-014 | Bot Signature Fade | Fade bots | Opposite of bots | +4-8% | HIGH |
| SPORTS-015 | Fibonacci Price Attractors | Technical analysis | Level-based | +2-5% | COMPLETE |

---

## Prioritization for LSD Screening

### Tier 1: Test First (Highest Potential, Most Independent)
| ID | Why Priority |
|----|-------------|
| SPORTS-001 | Steam exhaustion is a proven sports concept, timing-based signal |
| SPORTS-005 | Size velocity divergence is novel temporal pattern |
| SPORTS-008 | Size distribution shape is completely new dimension |
| SPORTS-009 | Spread widening captures MM behavior, unexplored |
| SPORTS-011 | Cross-market category signal is unique angle |

### Tier 2: Test Second (Good Potential)
| ID | Why Priority |
|----|-------------|
| SPORTS-002 | Opening reversal may overlap with Buyback, test for independence |
| SPORTS-003 | Momentum stall is simple to test |
| SPORTS-007 | Late large money requires timing data |
| SPORTS-012 | NCAAFTOTAL drilling extends promising finding |

### Tier 3: LSD Absurd (Quick Test)
| ID | Why Priority |
|----|-------------|
| SPORTS-004 | May overlap with RLM, quick check |
| SPORTS-006 | Round number clustering is novel angle |
| SPORTS-013 | Trade count milestone is pure LSD |
| SPORTS-014 | Bot fading is speculative but cheap to test |
| SPORTS-015 | Fibonacci prices are maximum LSD |

---

## Implementation Notes

### Data Fields Needed
- `timestamp_ms` - For all timing-based hypotheses
- `taker_side` - For all directional signals
- `yes_price` / `no_price` - For price analysis
- `count` - Trade size for size analysis
- `trade_value_cents` - Dollar weighting
- `market_ticker` - For cross-market analysis
- `category` - For category drilling
- `is_winner` / `market_result` - For validation

### Calculation Patterns

**Steam Detection:**
```python
def detect_steam(trades):
    # Group trades by 60-second windows
    # Find runs of 5+ same-direction
    # Check if price moved >10c during run
    pass
```

**Velocity Divergence:**
```python
def calc_velocity_divergence(trades):
    # Split into time periods
    # trades_per_hour / avg_size in each period
    # Detect increasing frequency + decreasing size
    pass
```

**Size Distribution:**
```python
from scipy.stats import entropy, ks_2samp
def size_distribution_change(trades_first_half, trades_second_half):
    # Calculate histograms
    # Compare entropy or KS statistic
    pass
```

---

*Generated by Strategy Researcher Agent - 2026-01-01*
*These hypotheses are INCOMING - awaiting LSD screening by Quant Agent*
