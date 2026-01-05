# Session 010: Novel Hypothesis Generation

**Date**: 2025-12-29
**Analyst**: The Quant (Opus 4.5)
**Objective**: Generate 15+ creative hypotheses that previous sessions have NOT explored
**Starting Hypothesis ID**: H070 (continuing from H066 in RESEARCH_JOURNAL)

---

## Context: What We Know

### The One Validated Strategy
- **S007: Fade High-Leverage YES** (+3.5% edge, 53,938 markets, NOT a price proxy)
- This exploits behavioral mispricing when retail bets longshots

### What Has Been Exhaustively Tested (Don't Repeat)
- Simple price-level strategies (NO at X-Yc) - calculation errors made them look better than they are
- Whale following/fading - just a price proxy
- Time-of-day/day-of-week patterns - weak to nonexistent
- Trade size filtering - same direction as base
- Category efficiency (KXBTCD, etc.) - was a mirage (different market subsets)
- Closing Line Value - Kalshi doesn't behave like sports betting
- Order flow imbalance RoC - price proxy
- Price oscillation - price proxy
- Gambler's fallacy - weak effect

### The Key Insight from S007
The leverage ratio signal works because it captures BEHAVIORAL information beyond price. The key is finding other behavioral/structural signals.

---

## The Novel Hypotheses

### H070: "Drunk Sports Betting" - Late Night Weekend Retail

**Hypothesis**: Late-night (11PM-3AM local time) weekend sports bets exhibit extreme favorite-longshot bias due to impulsive, emotionally-charged betting behavior.

**Behavioral Rationale**:
- People drinking on Friday/Saturday nights make impulsive bets
- Emotional attachment to "their team" overrides rational pricing
- Round-number bet sizes ($10, $25, $50) on favorites
- High-leverage longshot bets are more common (FOMO/drunk optimism)
- Sports markets are most active during this window

**How to Test**:
```python
# Filter criteria:
# 1. Time: 23:00-03:00 local time (need to infer timezone)
# 2. Day: Friday/Saturday nights (Friday 23:00 -> Saturday 03:00)
# 3. Category: Sports markets (KXNFL, KXNCAAF, KXNBA, etc.)
# 4. Trade characteristics: High leverage OR round count (10, 25, 50)
# 5. Compare: Edge of fading these trades vs baseline
```

**Signal Definition**:
- When: Weekend late-night sports trade with leverage > 3 or count in [10, 25, 50, 100]
- Action: Fade the trade (if YES, bet NO; if NO, bet YES)

**Expected Edge**: +5-10% if behavioral pattern is real
**Risk of Price Proxy**: MEDIUM - need to control for price level

---

### H071: Trade Clustering Velocity

**Hypothesis**: Rapid sequences of trades (multiple trades within 5-10 seconds) in the SAME direction indicate informed trading, while rapid trades in ALTERNATING directions indicate market maker activity/noise.

**Behavioral Rationale**:
- Informed traders accumulate positions quickly
- Market makers quote both sides
- A burst of same-direction trades = someone knows something
- A burst of alternating trades = normal market making

**How to Test**:
```python
# For each market:
# 1. Identify "bursts" - 3+ trades within 10 seconds
# 2. Calculate burst "directionality" - % of trades in same direction
# 3. High directionality burst (>80% same side) = potential signal
# 4. Track outcome: Do high-directionality bursts predict correctly?
```

**Signal Definition**:
- When: 3+ trades in 10 seconds, >80% same direction
- Action: Follow the direction (bet with the burst)

**Expected Edge**: +3-8% if informed trading is detectable
**Risk of Price Proxy**: LOW - measures trade PATTERN, not price

---

### H072: Price Path Volatility Regimes

**Hypothesis**: Markets that experience calm price evolution (low volatility) before resolution settle according to the final price, but markets with chaotic price paths (high volatility) are more likely to settle AGAINST the final price.

**Behavioral Rationale**:
- High volatility = uncertainty, information flow conflict
- When "both sides" are trading aggressively, one must be wrong
- The side that's losing the volatility battle may still win the outcome
- Calm markets = genuine probability; chaotic markets = noise

**How to Test**:
```python
# For each market:
# 1. Calculate price volatility (std of price changes)
# 2. Segment markets into volatility quartiles
# 3. For high-volatility markets: Does the "final direction" predict outcome?
# 4. Compare: Win rate of following final price in high vs low vol
```

**Signal Definition**:
- When: Market has high price volatility (top quartile)
- Action: Consider fading the final price direction

**Expected Edge**: +5-15% if information cascade theory holds
**Risk of Price Proxy**: LOW - measures volatility, not price level

---

### H073: Contrarian at the "Point of Maximum Pain"

**Hypothesis**: When price reaches an extreme (95c+ or 5c-) AFTER having been moderate (40-60c), contrarian bets have edge because the move was emotional overreaction.

**Behavioral Rationale**:
- Markets that start extreme and stay extreme = genuine favorites
- Markets that BECOME extreme = potential overreaction
- The "point of maximum pain" for the other side triggers capitulation
- Contrarian wins when the crowd panics into a bad price

**How to Test**:
```python
# For each market:
# 1. Track price evolution
# 2. Identify markets that START at 40-60c but END at 90c+ or 10c-
# 3. These are "momentum victims" - price moved dramatically
# 4. Bet against the final extreme price
```

**Signal Definition**:
- When: Market started 40-60c but final price is 90c+ (or 10c-)
- Action: Bet NO (or YES) - fade the extreme

**Expected Edge**: +10-20% if overreaction theory holds
**Risk of Price Proxy**: HIGH - need careful methodology to avoid price trap

---

### H074: First-Trade Informed Advantage

**Hypothesis**: The VERY FIRST trade in a market is placed by someone with information advantage (they found the market first, researched it, etc.), and following that direction has edge.

**Behavioral Rationale**:
- First movers have done research
- New markets are inefficiently priced
- The first trader "sets the anchor"
- Later traders may be responding to the first trade

**How to Test**:
```python
# For each market:
# 1. Identify the first trade
# 2. Record its direction (YES or NO)
# 3. Track: Does first trade direction predict outcome?
# 4. Control for: Price level of first trade
```

**Signal Definition**:
- When: First trade in a new market
- Action: Follow the direction

**Expected Edge**: +2-5% if first-mover theory holds
**Risk of Price Proxy**: MEDIUM - first trade price is also a signal

---

### H075: "Retail vs Pro" Time Windows

**Hypothesis**: Different times of day have different trader compositions. Morning (9AM-12PM) trades are more professional; evening (6PM-10PM) trades are more retail. Edge varies by time window.

**Behavioral Rationale**:
- Professional traders work business hours
- Retail traders bet after work/school
- Weekday mornings = institutions closing positions
- Weekend evenings = drunk retail (see H070)

**How to Test**:
```python
# Segment trades by hour of day:
# 1. Morning: 6-9AM (pre-market)
# 2. Business Hours: 9AM-5PM
# 3. Evening: 5PM-10PM
# 4. Late Night: 10PM-6AM
# Compare: Leverage ratio distribution, edge of fading, by time window
```

**Signal Definition**:
- When: Trade occurs in "retail heavy" window (evening/late night)
- Action: Apply stronger fade on high-leverage trades

**Expected Edge**: +2-4% improvement on S007 by time targeting
**Risk of Price Proxy**: LOW - measures time, not price

---

### H076: "Smart Money Alert" - Large Trades at Low Leverage

**Hypothesis**: Large trades (100+ contracts) at LOW leverage (<1.5) indicate sophisticated/institutional money betting on likely outcomes. These are high-conviction, low-risk-reward bets that WIN.

**Behavioral Rationale**:
- Smart money doesn't chase 10x returns
- Institutions bet big on "almost certain" outcomes
- Low leverage + large size = high conviction in a likely winner
- Opposite of retail longshot betting

**How to Test**:
```python
# Filter:
# 1. Count >= 100 (large trade)
# 2. Leverage ratio < 1.5 (low return potential)
# 3. These are "smart money" bets
# Action: FOLLOW (not fade) these trades
```

**Signal Definition**:
- When: Trade has count >= 100 AND leverage < 1.5
- Action: Follow the trade direction

**Expected Edge**: +2-5% if institutional signal is real
**Risk of Price Proxy**: MEDIUM - low leverage correlates with high price

---

### H077: Post-Settlement Reversion in Recurring Markets

**Hypothesis**: In recurring daily markets (KXBTCD, KXETHD), if yesterday's outcome was extreme (very unlikely outcome happened), today's market overreacts and creates edge.

**Behavioral Rationale**:
- Recency bias: "Bitcoin was down big yesterday, it'll be down again"
- Gambler's fallacy: "Bitcoin can't be down 3 days in a row"
- Both biases create exploitable mispricing
- Mean reversion is real in some markets

**How to Test**:
```python
# For recurring market series (KXBTCD):
# 1. Track yesterday's outcome (and how extreme it was)
# 2. Track today's opening price
# 3. Does yesterday's extreme outcome predict today's mispricing?
# 4. Example: If BTC was down 3 days in a row, is today's "down" overpriced?
```

**Signal Definition**:
- When: Recurring market opens after extreme prior outcome
- Action: Fade the recency bias direction

**Expected Edge**: +5-10% if behavioral pattern exists in recurring markets
**Risk of Price Proxy**: LOW - measures cross-day pattern, not price

---

### H078: Leverage Ratio Divergence from Price

**Hypothesis**: When leverage ratio and price give CONFLICTING signals, there's edge. Specifically, when price implies low probability but leverage is moderate (people aren't betting it like a longshot), the market may be mispriced.

**Behavioral Rationale**:
- Price = market consensus
- Leverage = how traders are TREATING the bet
- Divergence = tension between market and trader behavior
- If traders treat a "longshot" like a normal bet, maybe it's not that long

**How to Test**:
```python
# Calculate expected leverage from price:
# - YES at 20c should have leverage ~5 (100/20)
# - If actual leverage is ~2, traders are betting smaller (cautious)
#
# Flag markets where:
# - Price implies longshot (20-30c)
# - Leverage is LOWER than expected (traders aren't treating it as longshot)
```

**Signal Definition**:
- When: Price implies leverage X, but actual leverage is <0.5X
- Action: Bet in the direction traders are cautiously betting

**Expected Edge**: +3-8% if divergence is informative
**Risk of Price Proxy**: LOW - explicitly measures DIVERGENCE from price

---

### H079: Multi-Trade Whale Accumulation Pattern

**Hypothesis**: When a whale (count >= 500) spreads their bet across multiple trades in the same market (instead of one big trade), they're trying to hide information and are likely correct.

**Behavioral Rationale**:
- Smart whales split orders to avoid moving the market
- Multiple trades from same "actor" (inferred) = conviction
- Single big trade = could be impulsive
- Trade fragmentation is a stealth tactic

**How to Test**:
```python
# For each market:
# 1. Look for "suspicious sequences" - multiple large trades (100+)
#    in same direction within 5 minutes
# 2. Sum the sequence volume
# 3. If sequence total > 500, flag as "stealth whale"
# 4. Track: Do stealth whale directions predict better than single trades?
```

**Signal Definition**:
- When: 3+ trades of 100+ contracts, same direction, within 5 minutes
- Action: Follow the stealth whale direction

**Expected Edge**: +5-10% if whale detection works
**Risk of Price Proxy**: LOW - measures PATTERN, not price

---

### H080: "Expiry Proximity Squeeze" - Last Hour Trading

**Hypothesis**: In the final hour before market resolution, prices can detach from fundamentals as traders desperately exit or double down. Fading extreme moves in this window has edge.

**Behavioral Rationale**:
- Last hour = panic/desperation
- Prices can temporarily "overshoot"
- Market makers may withdraw, reducing liquidity
- Retail makes emotional closing bets

**How to Test**:
```python
# For each market (where we know close_time):
# 1. Identify trades in final 60 minutes before close
# 2. Calculate price direction in final hour
# 3. If price moved 10c+ in final hour, flag as "squeeze"
# 4. Fade the squeeze direction
```

**Signal Definition**:
- When: Price moved 10c+ in final hour before resolution
- Action: Fade the direction of the move

**Expected Edge**: +5-15% if expiry squeeze is real
**Risk of Price Proxy**: MEDIUM - final price still matters

---

### H081: Cross-Category Sentiment Spillover

**Hypothesis**: Strong outcomes in one category (e.g., several NFL favorites winning) create sentiment spillover into related categories (NFL player props, next week's NFL games), causing mispricing.

**Behavioral Rationale**:
- "NFL favorites are crushing it this week" -> overbetting favorites next week
- Winning streaks in a category create overconfidence
- Cross-market correlation in BETTING behavior (not outcomes)
- Exploitable if we can detect the spillover

**How to Test**:
```python
# 1. Track category-level outcomes (% favorites winning per day)
# 2. Next day, check: Is favorite-longshot bias STRONGER after good favorite day?
# 3. Fade the amplified bias
```

**Signal Definition**:
- When: Previous day saw >70% favorites win in a category
- Action: Bet on longshots in that category (fade the overconfidence)

**Expected Edge**: +3-8% if spillover effect exists
**Risk of Price Proxy**: LOW - measures cross-day sentiment, not price

---

### H082: Trade Count Cluster Analysis

**Hypothesis**: Markets with unusual trade count distributions (many trades of exactly the same size) indicate automated/bot activity, which may be informed or dumb depending on the pattern.

**Behavioral Rationale**:
- Retail traders bet random amounts ($13.47, $27.00)
- Bots bet exact amounts repeatedly (100, 100, 100)
- Informed bots = follow them
- Market making bots = neutral
- Arbitrage bots = they know something

**How to Test**:
```python
# For each market:
# 1. Calculate trade count distribution (histogram of contract sizes)
# 2. Flag markets with high "count concentration" (many trades same size)
# 3. Separate: Round counts (100, 500) vs exact repeats (137, 137, 137)
# 4. Track: Do bot-heavy markets behave differently?
```

**Signal Definition**:
- When: Market has >50% of trades at same count
- Action: Follow the majority trade direction (bots may be informed)

**Expected Edge**: +2-5% if bot detection works
**Risk of Price Proxy**: LOW - measures distribution, not price

---

### H083: "Minnow Swarm" Retail Consensus

**Hypothesis**: When MANY small trades (count < 10) all bet the same direction, retail consensus is forming. Retail consensus is usually WRONG, so fade it.

**Behavioral Rationale**:
- Many small bets = retail swarm
- Retail follows headlines, social media, gut feelings
- When "everyone" on Reddit thinks X, X is probably wrong
- The crowd is often the "dumb money"

**How to Test**:
```python
# For each market:
# 1. Count small trades (count < 10) in each direction
# 2. Calculate "minnow imbalance" = (YES_minnow - NO_minnow) / total_minnow
# 3. When minnow imbalance > 70% in one direction, flag as "retail consensus"
# 4. Fade the retail consensus
```

**Signal Definition**:
- When: >70% of small trades (<10 contracts) are in same direction
- Action: Fade the retail direction

**Expected Edge**: +3-8% if retail is predictably wrong
**Risk of Price Proxy**: MEDIUM - retail behavior correlates with price

---

### H084: Leverage Ratio Trend Within Market

**Hypothesis**: If leverage ratio is INCREASING over time within a market (later trades have higher leverage than earlier trades), it indicates increasing speculation/desperation. Fade the direction of increasing leverage.

**Behavioral Rationale**:
- Early trades = considered, researched
- Late high-leverage trades = FOMO, desperation
- Increasing leverage trend = market attracting longshot bettors
- These late longshot bettors are probably wrong

**How to Test**:
```python
# For each market:
# 1. Calculate average leverage in first 25% vs last 25% of trades
# 2. If leverage_late > leverage_early * 1.5, flag as "increasing speculation"
# 3. Fade the majority direction of late high-leverage trades
```

**Signal Definition**:
- When: Late trades have 1.5x+ higher average leverage than early trades
- Action: Fade the direction of late high-leverage trades

**Expected Edge**: +3-7% if desperation is detectable
**Risk of Price Proxy**: LOW - measures leverage TREND, not absolute price

---

### H085: "Closing Bell" Institutional Pattern

**Hypothesis**: Large trades (100+ contracts) placed in the final 10 minutes of US market hours (3:50-4:00 PM ET) are institutional closing trades. These represent "informed money" exiting/entering based on end-of-day information.

**Behavioral Rationale**:
- Institutions have trading windows
- End-of-day = rebalancing, closing positions
- Large EOD trades = institution with day's information
- These traders have full day's data

**How to Test**:
```python
# Filter:
# 1. Time: 3:50-4:00 PM ET (19:50-20:00 UTC)
# 2. Size: count >= 100
# 3. Track: Do these trades predict better than random large trades?
```

**Signal Definition**:
- When: Large trade (100+) in final 10 minutes of US market hours
- Action: Follow the trade direction

**Expected Edge**: +2-5% if institutional pattern exists
**Risk of Price Proxy**: LOW - measures timing and size, not price

---

## Priority Testing Order

### Tier 1: Test First (Most Novel, Highest Potential)

| ID | Hypothesis | Why Prioritize |
|----|------------|----------------|
| H070 | Drunk Sports Betting | User requested, strong behavioral rationale |
| H071 | Trade Clustering Velocity | Novel pattern, not price-based |
| H072 | Price Path Volatility | Information cascade theory support |
| H078 | Leverage Divergence | Builds on validated S007 signal |
| H084 | Leverage Ratio Trend | Detects desperation, time-based |

### Tier 2: Test Second (Strong Rationale)

| ID | Hypothesis | Why Promising |
|----|------------|---------------|
| H073 | Maximum Pain Contrarian | Novel angle on failed contrarian |
| H076 | Smart Money Alert | Opposite of S007, may complement |
| H079 | Stealth Whale | Pattern detection, not price |
| H080 | Expiry Squeeze | Known market phenomenon |
| H083 | Minnow Swarm | Retail consensus fade |

### Tier 3: Test If Tier 1-2 Fail (Speculative)

| ID | Hypothesis | Notes |
|----|------------|-------|
| H074 | First Trade | Partially tested before |
| H075 | Retail vs Pro Time | May overlap with H070 |
| H077 | Recurring Market Reversion | Requires cross-day analysis |
| H081 | Sentiment Spillover | Complex, cross-category |
| H082 | Bot Detection | May be too noisy |
| H085 | Closing Bell | Narrow time window |

---

## Key Innovation vs Previous Sessions

### What's Different About These Hypotheses

1. **Pattern-Based, Not Price-Based**
   - H071 (clustering), H079 (stealth whale), H082 (bot detection) measure PATTERNS
   - Cannot be dismissed as price proxies

2. **Time-Conditional**
   - H070 (late night), H075 (time windows), H080 (expiry), H085 (closing bell)
   - These are testable with different time controls

3. **Building on S007**
   - H078 (leverage divergence), H084 (leverage trend) extend the validated leverage signal
   - More likely to find edge in same family

4. **Cross-Trade Analysis**
   - H071, H079, H083 look at SEQUENCES of trades, not individual trades
   - Novel angle we haven't explored

5. **Behavioral Specificity**
   - H070 (drunk betting), H073 (maximum pain), H077 (recency bias)
   - Target specific behavioral biases with clear mechanisms

---

## Testing Methodology Notes

### For All Hypotheses

1. **Price Proxy Check is MANDATORY**
   - After finding raw edge, MUST compare to baseline at same price levels
   - If improvement over baseline < +2%, reject as price proxy

2. **Sample Size Requirements**
   - Minimum 50 unique markets (not trades)
   - Minimum 500 trades for pattern-based hypotheses

3. **Temporal Stability**
   - Split data into halves, verify edge exists in both
   - Day-by-day consistency check

4. **Concentration Check**
   - No single market can contribute >30% of profit
   - Pattern must be diversified

5. **Bonferroni Correction**
   - With 16 new hypotheses, p-value threshold = 0.05/16 = 0.003
   - Be rigorous about statistical significance

---

## Data Requirements

### Available in Current Data
- `timestamp` / `datetime` - for time-based hypotheses
- `count` - trade size
- `leverage_ratio` - for leverage-based hypotheses
- `taker_side` - YES or NO
- `yes_price` / `no_price` - price at trade time
- `market_ticker` - can parse for category
- `is_winner` / `market_result` - outcome

### Need to Derive
- Time zone adjustment (convert to local time for H070)
- Trade sequences within market (for H071, H079)
- Price volatility within market (for H072)
- Leverage trend over time (for H084)
- Cross-day market linking (for H077, H081)
- Close time proximity (for H080) - need to join with market data

---

## Next Steps

1. Update RESEARCH_JOURNAL.md with Session 010 entry
2. Begin testing Tier 1 hypotheses in priority order
3. Start with H070 (Drunk Sports Betting) per user request
4. Create analysis script: `session010_novel_hypotheses.py`
5. Document results rigorously, including all failures

---

*Generated by The Quant (Opus 4.5) - Session 010*
*These hypotheses have NOT been tested. They are creative brainstorm output.*
*Testing required before any can be considered validated.*
