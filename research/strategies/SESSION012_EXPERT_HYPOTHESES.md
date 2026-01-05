# Session 012: Expert Hypothesis Generation

**Date**: 2025-12-29
**Analyst**: The Quant (Opus 4.5)
**Objective**: Generate 15+ genuinely novel hypotheses that haven't been tested
**Starting Hypothesis ID**: H103 (continuing from H102)
**Session Type**: Expert Creative Brainstorm

---

## What We Know Works

### Validated Strategies (Passed All Tests)
| Strategy | Edge | Improvement vs Baseline | Mechanism |
|----------|------|------------------------|-----------|
| S007: Fade High-Leverage YES | +3.5% | +6.8% | Retail longshot bias |
| S008: Fade Drunk Sports Betting | +3.5% | +1.1% | Late-night impulsive betting |
| S009: Extended Drunk Betting | +4.6% | +1.3% | Evening weekend sports |

### Key Insight: WHY These Work
All validated strategies capture **BEHAVIORAL** information:
- S007: High leverage = emotional bet on longshot = systematically wrong
- S008/S009: Late-night weekend = impulsive/intoxicated = systematically wrong

The market IS efficient for price-based strategies. We need BEHAVIORAL or STRUCTURAL signals.

---

## What Has DEFINITIVELY Failed

### Price-Based Strategies (All Calculation Errors or Near-Zero Edge)
- NO at 50-60c, 60-70c, 70-80c, 80-90c, 90-100c: Near breakeven after correction
- YES at any price range: Negative edge

### Bot Detection (Methodology Errors)
- H087: Round-size bot detection - **CRITICAL BUG** (inverted NO prices)
- H102: Leverage stability - **PRICE PROXY** (-1.2% improvement)
- H088, H090, H094, H095, H097, H098: All either price proxies or weak

### Information-Based Approaches
- CLV (Closing Line Value): Kalshi doesn't behave like sports betting
- Order flow imbalance rate-of-change: Pure price proxy
- Whale following/fading: Just price proxy
- Insider trading patterns: None detected, edge is behavioral

### Time-Based Approaches
- Time-of-day patterns (general): Weak, not actionable
- Day-of-week patterns: No reliable edge
- Resolution proximity: Price proxy

### Category-Based Approaches
- Category-specific strategies: **MIRAGE** - different market subsets
- NCAAFTOTAL: Promising but only 94 markets, needs more data

---

## The Novel Hypotheses

### DIMENSION 1: Price Path Shape Analysis

These hypotheses analyze the TRAJECTORY of prices, not just levels.

---

### H103: Price Path Asymmetry - "Slow Drift vs Fast Spike"

**Hypothesis**: Markets where price SLOWLY drifted to an extreme (many small moves) predict differently than markets where price SPIKED rapidly (few large moves) to the same extreme.

**Behavioral Rationale**:
- Slow drift = gradual information incorporation = efficient
- Fast spike = overreaction/momentum = potential reversion
- Same endpoint, different path = different reliability
- Human traders cause spikes; information causes drift

**Why It Might Work**:
- Price path encodes information about HOW the market reached its conclusion
- Spike patterns may indicate emotional trading that's exploitable
- This is NOT a price proxy - it measures SHAPE at any price level

**How to Test**:
```python
# For each market:
# 1. Calculate total price change (start to end)
# 2. Count number of trades (n_trades)
# 3. Calculate "moves_per_trade" = total_change / n_trades
# 4. High moves_per_trade = spike (few large moves)
# 5. Low moves_per_trade = drift (many small moves)
#
# Signal: At extreme prices (>85c or <15c), fade SPIKE patterns
```

**Signal Definition**:
- When: Price > 85c (or < 15c) AND moves_per_trade > median
- Action: Fade (bet NO if YES price spiked high)

**Expected Edge**: +3-8% if spike patterns are exploitable
**Risk of Price Proxy**: LOW - measures PATH, not level

---

### H104: Volatility Regime Shift Detection

**Hypothesis**: When a market transitions from LOW to HIGH volatility (or vice versa) mid-life, the direction of trades DURING the transition predicts outcomes better than trades before or after.

**Behavioral Rationale**:
- Volatility shift = new information entering the market
- Traders during transition are RESPONDING to new information
- Pre-shift traders may be stale; post-shift traders are chasing
- The transition window captures the "informed response"

**Why It Might Work**:
- Information cascade theory: first responders have the real info
- Volatility shift is a structural signal, not a price signal
- Captures the moment of information discovery

**How to Test**:
```python
# For each market:
# 1. Split trades into thirds by time
# 2. Calculate volatility (price std) in each third
# 3. If vol_third2 > 2 * vol_third1 OR vol_third2 > 2 * vol_third3:
#    -> Volatility shift detected in middle period
# 4. Track: Direction of trades DURING shift vs outcome
```

**Signal Definition**:
- When: Volatility in middle third > 2x volatility in adjacent thirds
- Action: Follow the net direction of middle-third trades

**Expected Edge**: +4-8% if transition trades are informed
**Risk of Price Proxy**: LOW - measures volatility structure

---

### H105: Price Stickiness at Levels - "Resistance/Support Failure"

**Hypothesis**: When price repeatedly TRIES to break through a level (tests it 3+ times) but eventually breaks through, the market has revealed conviction. Following the breakout has edge.

**Behavioral Rationale**:
- Multiple tests at a level = resistance/support (like technical analysis)
- Breakout after multiple tests = strong conviction finally winning
- This is market microstructure, not just price
- Technical analysis patterns may work in prediction markets

**Why It Might Work**:
- Markets "remember" important price levels (round numbers, previous highs)
- Multiple tests mean multiple traders defending that level
- When defenders are exhausted, breakout is real

**How to Test**:
```python
# For each market:
# 1. Identify "test levels" - prices touched 3+ times
# 2. Track if price eventually broke through
# 3. After breakout, track outcome
# 4. Compare: breakouts after 1 test vs 3+ tests
```

**Signal Definition**:
- When: Price breaks level that was tested 3+ times
- Action: Follow the breakout direction

**Expected Edge**: +3-6% if technical patterns translate
**Risk of Price Proxy**: MEDIUM - involves price levels, but measures structure

---

### DIMENSION 2: Trade Size Distribution Analysis

These hypotheses analyze the DISTRIBUTION of trade sizes, not just individual trades.

---

### H106: Bimodal Size Distribution - "Two Camps"

**Hypothesis**: When a market has a bimodal trade size distribution (many small trades AND many large trades, but few medium), it indicates polarization. Polarized markets may have different predictability.

**Behavioral Rationale**:
- Bimodal = retail (small) + institutional (large) both active
- Missing medium trades = no "in-between" players
- Polarization may indicate controversial/uncertain outcome
- OR: institutional large trades are informed

**Why It Might Work**:
- Trade size distribution encodes market structure
- Bimodal markets have different dynamics than unimodal
- May reveal when smart money and dumb money disagree

**How to Test**:
```python
# For each market:
# 1. Calculate trade size distribution
# 2. Measure bimodality (Hartigan's dip statistic or similar)
# 3. In bimodal markets: compare direction of small vs large trades
# 4. If they disagree: who is right?
```

**Signal Definition**:
- When: Market has bimodal size distribution AND large/small disagree
- Action: Follow the large trades (institutions informed)

**Expected Edge**: +3-7% if size structure is informative
**Risk of Price Proxy**: LOW - measures distribution shape

---

### H107: Trade Size Entropy - "Organized vs Chaotic"

**Hypothesis**: Markets with LOW trade size entropy (few distinct sizes, predictable pattern) are bot-dominated. Markets with HIGH entropy (many distinct sizes, random) are human-dominated. Edge differs.

**Behavioral Rationale**:
- Bots use fixed sizes (100, 100, 100) - low entropy
- Humans vary sizes (47, 23, 189) - high entropy
- Entropy measures "randomness" of the size distribution
- Could segment markets better than individual trade detection

**Why It Might Work**:
- Previous bot detection failed by looking at individual trades
- Entropy measures the OVERALL pattern of a market
- May capture bot-dominance more robustly

**How to Test**:
```python
# For each market:
# 1. Calculate Shannon entropy of trade sizes
# 2. Low entropy (<2.0) = bot-dominated
# 3. High entropy (>3.5) = human-dominated
# 4. Test: Edge of base strategy differs by entropy segment?
```

**Signal Definition**:
- When: Market has low size entropy (bot-dominated)
- Action: Apply different strategy (maybe fade, maybe follow)

**Expected Edge**: +2-5% if segmentation reveals different dynamics
**Risk of Price Proxy**: LOW - measures entropy, not price

---

### DIMENSION 3: Time Series Patterns Within Markets

These hypotheses look at sequential patterns in trade streams.

---

### H108: Momentum Exhaustion Point - "The Fifth Trade"

**Hypothesis**: After N consecutive trades in the same direction, the next trade in the OPPOSITE direction is particularly informative - it may signal a turning point.

**Behavioral Rationale**:
- Momentum runs eventually exhaust
- The first trader to bet against the run is either dumb or informed
- 4+ consecutive same-direction trades = strong momentum
- First counter-trend trade after long run = potential reversal signal

**Why It Might Work**:
- Mean reversion is real in many markets
- The timing of reversal matters
- Consecutive trade counts are structural, not price-based

**How to Test**:
```python
# For each market:
# 1. Find runs of 4+ consecutive same-direction trades
# 2. Identify the first opposite-direction trade after run
# 3. Track: Does following the counter-trend trade work?
# 4. Compare to random opposite-direction trades
```

**Signal Definition**:
- When: After 4+ consecutive YES trades, first NO trade appears (or vice versa)
- Action: Follow the counter-trend trade

**Expected Edge**: +3-7% if momentum exhaustion is detectable
**Risk of Price Proxy**: LOW - measures sequence, not price

---

### H109: Trade Interval Acceleration/Deceleration

**Hypothesis**: When time between trades is ACCELERATING (trades coming faster and faster), it indicates increasing attention/urgency. Decelerating trades indicate waning interest. Direction of accelerating trades may be more predictive.

**Behavioral Rationale**:
- Increasing trade frequency = market heating up
- Decreasing frequency = market cooling off
- Hot markets attract more attention, may have more edge
- Cold markets are forgotten, may be more efficient

**Why It Might Work**:
- Trade interval patterns capture market "temperature"
- Urgency may indicate information arrival
- Cooling off may indicate false alarm

**How to Test**:
```python
# For each market:
# 1. Calculate time gaps between consecutive trades
# 2. Fit linear regression to gaps over time
# 3. Negative slope = accelerating (intervals shrinking)
# 4. Positive slope = decelerating (intervals growing)
# 5. In accelerating markets: follow majority direction
```

**Signal Definition**:
- When: Trade intervals show significant acceleration (slope < -0.1)
- Action: Follow the majority direction of accelerating phase

**Expected Edge**: +3-6% if acceleration indicates real information
**Risk of Price Proxy**: LOW - measures timing structure

---

### H110: First/Last Trade Direction Persistence

**Hypothesis**: The direction of the FIRST trade in a market sets an "anchor" that the LAST trade often agrees with. When first and last DISAGREE, one is more predictive than the other.

**Behavioral Rationale**:
- First trade = researcher/informed person found the market first
- Last trade = final conviction before resolution
- Agreement = consistent signal; disagreement = uncertainty
- One may dominate the other systematically

**Why It Might Work**:
- First-mover advantage is real in many domains
- Last-mover has most information but may also be noise
- The relationship between first and last encodes market evolution

**How to Test**:
```python
# For each market:
# 1. Get first trade direction and last trade direction
# 2. Calculate: agree vs disagree
# 3. When disagree: track which one predicts better
# 4. Signal based on winner
```

**Signal Definition**:
- When: First and last trade disagree
- Action: Follow [first or last, whichever tests better]

**Expected Edge**: +2-5% if first/last relationship is informative
**Risk of Price Proxy**: LOW - measures position in sequence

---

### DIMENSION 4: Cross-Market and Category Analysis

These hypotheses look at relationships BETWEEN markets.

---

### H111: Same-Event Multi-Market Correlation

**Hypothesis**: For events with multiple markets (e.g., "Will X happen" AND "Will Y happen" where X and Y are related), trades in one market that CONFLICT with fair pricing of the other indicate mispricing.

**Behavioral Rationale**:
- Related markets should have correlated prices
- When traders bet X will happen but also bet not-Y (where X implies Y), there's inconsistency
- Cross-market inconsistency = at least one market is mispriced
- Exploit by betting on the "correct" pricing

**Why It Might Work**:
- Arbitrage relationships exist across related markets
- Retail may not see cross-market relationships
- Information from one market can inform another

**How to Test**:
```python
# Identify related market pairs (same event, different questions)
# E.g., "Will Biden win TX?" and "Will Biden win the election?"
#
# If Biden at 5% in TX but 52% nationally: inconsistent
# Track which market is "right" more often
```

**Signal Definition**:
- When: Related markets have inconsistent pricing
- Action: Bet on the market that should "converge"

**Expected Edge**: +5-10% if cross-market inefficiency exists
**Risk of Price Proxy**: MEDIUM - involves price comparison, but across markets

---

### H112: Category Momentum Spillover (Refined)

**Hypothesis**: When a CATEGORY (e.g., NFL) has multiple resolutions in a short period that DEFY the favorite-longshot bias (longshots winning), the NEXT market in that category may see overcorrection.

**Behavioral Rationale**:
- "NFL has had 3 upsets this weekend" -> "Maybe the favorite bias doesn't apply today"
- Traders may overcorrect by betting MORE on longshots
- This overcorrection is exploitable
- Opposite: streak of favorites winning -> underbet longshots

**Why It Might Work**:
- Recency bias is a documented behavioral phenomenon
- Category-level streaks influence individual market betting
- The spillover creates temporary mispricing

**How to Test**:
```python
# For each category with multiple markets per day:
# 1. Track resolution streak (favorites or longshots winning)
# 2. After 3+ longshot wins: is next market's favorite overpriced?
# 3. After 3+ favorite wins: is next market's longshot overpriced?
```

**Signal Definition**:
- When: Category had 3+ consecutive longshot wins today
- Action: Bet FAVORITE in next market (overcorrection fade)

**Expected Edge**: +3-7% if category spillover exists
**Risk of Price Proxy**: LOW - measures cross-market sequence

---

### DIMENSION 5: Novel Behavioral Patterns

These hypotheses target specific, untested behavioral biases.

---

### H113: "Round Number Magnet" Effect

**Hypothesis**: Prices that approach but don't reach a round number (49c, 51c near 50c) create psychological tension. Markets that "almost" hit a round number may behave differently.

**Behavioral Rationale**:
- Round numbers are psychologically significant (50c = "even odds")
- Price at 49c feels like "almost 50-50"
- Traders may anchor to round numbers
- The difference between 49c and 51c may not match the probability difference

**Why It Might Work**:
- Anchoring is a well-documented bias
- Round number effects exist in stock markets
- Prediction markets may have similar psychology

**How to Test**:
```python
# For trades near round numbers:
# 1. Define "near" as within 2c of 25c, 50c, 75c
# 2. Calculate: does edge differ for near-round vs far-from-round?
# 3. Specifically: is 48-52c range different from 43-47c or 53-57c?
```

**Signal Definition**:
- When: Price is within 2c of a round number (50c, 75c, etc.)
- Action: Bet toward the round number (price is attracted to it)

**Expected Edge**: +2-5% if round number magnet effect exists
**Risk of Price Proxy**: MEDIUM - involves price but tests specific psychological effect

---

### H114: "Certainty Premium" at Near-100% Prices

**Hypothesis**: At extreme prices (95c+), traders pay a "certainty premium" - they overpay for the psychological comfort of a "sure thing". This creates exploitable edge in betting against near-certainties.

**Behavioral Rationale**:
- Humans prefer certainty over uncertainty
- A "95% chance" feels almost certain, but 5% happens 1 in 20 times
- Behavioral economics shows people overpay for certainty
- This may create mispricing at extreme high prices

**Why It Might Work**:
- Certainty bias is documented in prospect theory
- "Sure things" feel qualitatively different from "almost sure"
- The 95c-99c range may have systematic mispricing

**How to Test**:
```python
# For markets with final prices 95c-99c:
# 1. Calculate: actual win rate vs implied probability
# 2. If actual < implied consistently, there's edge
# 3. BUT: this was tested before - key is time-conditional
# 4. Test: Is certainty premium stronger in evening/weekend?
```

**Signal Definition**:
- When: YES price at 95c+ AND time is evening/weekend (retail heavy)
- Action: Bet NO (fade the certainty premium)

**Expected Edge**: +2-4% if time-conditional certainty premium exists
**Risk of Price Proxy**: HIGH - but combined with time filter may work

---

### H115: Trade Size "Commitment" Signal

**Hypothesis**: The FIRST large trade (100+ contracts) in a market signals genuine commitment. If this committed trader later trades in the OPPOSITE direction (reduces position), it's a strong reversal signal.

**Behavioral Rationale**:
- Large initial bet = high conviction
- Reversal of large bet = conviction changed based on new info
- This is different from random large trades
- The SEQUENCE matters: large -> small opposite = information

**Why It Might Work**:
- Position reversal requires overcoming sunk cost bias
- Traders who reverse must have strong reason
- This captures conviction dynamics, not just size

**How to Test**:
```python
# For each market:
# 1. Find first trade with count >= 100
# 2. Track subsequent trades by same size (proxy for same trader)
# 3. If same-size trade appears in OPPOSITE direction later:
#    -> Flag as "reversal"
# 4. Track: Does reversal predict final outcome?
```

**Signal Definition**:
- When: Large trade followed by same-size opposite-direction trade
- Action: Follow the reversal direction

**Expected Edge**: +4-8% if position reversal is informative
**Risk of Price Proxy**: LOW - measures size sequence pattern

---

### H116: "Event Proximity Confidence" Asymmetry

**Hypothesis**: As an event approaches, confident trades (high size, low leverage) become MORE predictive, while uncertain trades (small size, high leverage) become LESS predictive. The gap widens near resolution.

**Behavioral Rationale**:
- Near resolution, some traders have real information
- Others are speculating/gambling
- The difference in trade characteristics reveals which is which
- Earlier in market life, this distinction matters less

**Why It Might Work**:
- Information asymmetry increases near resolution
- Confident trades near resolution = someone knows something
- This is time-conditional AND behavior-conditional

**How to Test**:
```python
# For markets where we know close_time:
# 1. Segment trades into "early" (>1 day out) and "late" (<1 hour out)
# 2. Calculate edge of high-conviction trades (large, low leverage) in each period
# 3. Calculate edge of low-conviction trades (small, high leverage) in each period
# 4. If gap widens near resolution, signal exists
```

**Signal Definition**:
- When: High-conviction trade (count >= 50, leverage < 2) in final hour
- Action: Follow that trade

**Expected Edge**: +3-6% if proximity amplifies signal quality
**Risk of Price Proxy**: LOW - measures time x behavior interaction

---

### H117: "Contrarian at the 90% Line"

**Hypothesis**: When YES price rises ABOVE 90c (or drops below 10c) for the first time, contrarian bets placed EXACTLY at that moment have edge - the market may be overextending.

**Behavioral Rationale**:
- 90c is a psychological threshold ("90% certain")
- First touch of 90c may trigger FOMO buying
- Smart contrarians bet against the FOMO
- The KEY is timing: exactly when threshold is first crossed

**Why It Might Work**:
- Threshold crossing is a discrete event
- FOMO creates temporary mispricing
- First cross may be overshoot; later 90c levels are different

**How to Test**:
```python
# For each market:
# 1. Identify first trade that crossed 90c (or 10c)
# 2. Bet opposite at that exact moment
# 3. Track: Does this contrarian timing work?
# 4. Compare to random contrarian at 90c
```

**Signal Definition**:
- When: Price first crosses 90c (has been <90c, now >=90c)
- Action: Bet NO immediately after

**Expected Edge**: +5-12% if threshold FOMO is exploitable
**Risk of Price Proxy**: MEDIUM - involves price threshold but timing-specific

---

## Summary Table: Novel Hypotheses

| ID | Name | Dimension | Price Proxy Risk | Novelty Score |
|----|------|-----------|-----------------|---------------|
| H103 | Price Path Asymmetry | Path Shape | LOW | HIGH |
| H104 | Volatility Regime Shift | Path Shape | LOW | HIGH |
| H105 | Price Level Breakout | Path Shape | MEDIUM | MEDIUM |
| H106 | Bimodal Size Distribution | Size Dist | LOW | HIGH |
| H107 | Trade Size Entropy | Size Dist | LOW | HIGH |
| H108 | Momentum Exhaustion Point | Sequence | LOW | MEDIUM |
| H109 | Trade Interval Acceleration | Sequence | LOW | HIGH |
| H110 | First/Last Direction | Sequence | LOW | LOW |
| H111 | Same-Event Multi-Market | Cross-Market | MEDIUM | HIGH |
| H112 | Category Momentum Spillover | Cross-Market | LOW | MEDIUM |
| H113 | Round Number Magnet | Behavioral | MEDIUM | MEDIUM |
| H114 | Certainty Premium (Time-Cond) | Behavioral | HIGH | LOW |
| H115 | Trade Size Commitment | Behavioral | LOW | HIGH |
| H116 | Event Proximity Confidence | Behavioral | LOW | HIGH |
| H117 | Contrarian at 90% Line | Behavioral | MEDIUM | MEDIUM |

---

## Priority Testing Recommendations

### Tier 1: Test First (Novel + Low Price Proxy Risk)

| ID | Hypothesis | Why Prioritize |
|----|------------|----------------|
| H103 | Price Path Asymmetry | Novel, measures path not level |
| H106 | Bimodal Size Distribution | Novel size distribution analysis |
| H109 | Trade Interval Acceleration | Time structure, never tested |
| H115 | Trade Size Commitment | Conviction dynamics, low proxy risk |
| H116 | Event Proximity Confidence | Time x behavior interaction |

### Tier 2: Test Second (Novel but Higher Risk)

| ID | Hypothesis | Why Promising |
|----|------------|---------------|
| H104 | Volatility Regime Shift | Information cascade theory |
| H107 | Trade Size Entropy | Bot segmentation angle |
| H108 | Momentum Exhaustion | Mean reversion timing |
| H111 | Same-Event Multi-Market | Arbitrage inefficiency |
| H117 | Contrarian at 90% Line | FOMO exploitation |

### Tier 3: Test If Tier 1-2 Fail

| ID | Hypothesis | Notes |
|----|------------|-------|
| H105 | Price Level Breakout | Technical analysis angle |
| H110 | First/Last Direction | Partially tested before |
| H112 | Category Momentum | Related to failed category tests |
| H113 | Round Number Magnet | Similar to failed round number tests |
| H114 | Certainty Premium | High price proxy risk |

---

## Key Innovations vs Previous Sessions

### What Makes These Different

1. **Price PATH, not price LEVEL**
   - H103, H104, H105 analyze trajectory
   - Can't be dismissed as "just betting at high prices"

2. **Distribution Analysis**
   - H106, H107 look at aggregate patterns
   - More robust than individual trade detection

3. **Sequence/Timing Patterns**
   - H108, H109, H110 examine trade ordering
   - Structural patterns, not price correlation

4. **Cross-Market Relationships**
   - H111, H112 look between markets
   - Unexploited correlation structure

5. **Conditional Behavioral Signals**
   - H115, H116, H117 are time/context conditional
   - More specific than general behavioral theories

---

## Testing Methodology Requirements

### For All Hypotheses

1. **Price Proxy Check is MANDATORY**
   - Compare to baseline at same price levels
   - Improvement must be > +2%

2. **Sample Size Requirements**
   - Minimum 50 unique markets
   - Pattern-based: minimum 200 trades with pattern

3. **Temporal Stability**
   - Split by time period
   - Edge must exist in both halves

4. **Concentration Check**
   - No single market > 30% of profit

5. **Bonferroni Correction**
   - With 15 hypotheses: p < 0.003 required

---

## Data Requirements

### Available
- `timestamp` / `datetime` - for timing analysis
- `count` - trade size
- `leverage_ratio` - behavioral signal
- `taker_side` - YES or NO
- `yes_price` / `no_price` - price at trade time
- `market_ticker` - for category/cross-market analysis
- `is_winner` / `market_result` - outcome

### Need to Derive
- Price path statistics (volatility, trend)
- Trade size distributions (entropy, bimodality)
- Sequential patterns (runs, reversals)
- Cross-market relationships
- Close time proximity (where available)

---

## Next Steps

1. **Update RESEARCH_JOURNAL.md** with Session 012 entry
2. **Create analysis script**: `session012_expert_hypotheses.py`
3. **Test Tier 1 hypotheses** in priority order
4. **Start with H103** (Price Path Asymmetry) - most novel
5. **Apply rigorous methodology** including price proxy checks

---

*Generated by The Quant (Opus 4.5) - Session 012*
*These hypotheses are the result of expert brainstorming on prediction market microstructure.*
*Designed to find patterns that previous sessions missed.*
*Focus: Price PATH, size DISTRIBUTION, trade SEQUENCE, cross-market RELATIONSHIPS, conditional BEHAVIOR.*
