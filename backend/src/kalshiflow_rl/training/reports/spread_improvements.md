# Market Spread Analysis and Improvements

*Date: December 15, 2024*

## Executive Summary

After analyzing Session 32's orderbook data, we discovered that **80% of Kalshi markets have spreads >20¢**, making them economically untradeable. This is not a data error - it accurately reflects Kalshi's market structure where most markets are illiquid, obscure predictions with minimal trading interest.

## The Spread Problem

### What We Found

**Session 32 Statistics** (1,000 markets, 17.8 hours):
- **3.1%** of markets: <5¢ spreads (liquid, tradeable)
- **17.0%** of markets: 5-20¢ spreads (semi-liquid, challenging)
- **79.9%** of markets: >20¢ spreads (illiquid, untradeable)

### Why Wide Spreads Kill Profitability

With a 20¢ spread:
1. **Entry Cost**: Buy YES at 65¢ (ask price)
2. **Immediate Value**: Position worth 45¢ (bid price)
3. **Instant Loss**: -20¢ or -30.8% of investment
4. **Break-even Requirement**: Market must move 44% in your favor

### Examples from Session 32

**Liquid Markets (<5¢ spreads):**
- Presidential party markets: 3.4¢
- Major political figures: 0-2¢
- Fed Chair nominees: 0.3-1.5¢

**Illiquid Markets (>50¢ spreads):**
- CO2 level predictions: 87-92¢
- Turkish parliament seats: 86¢
- James Bond actor markets: 83¢
- Pope selection markets: 83¢

## Root Cause Analysis

### Why So Many Wide Spreads?

1. **Market Universe**: Session 32 captured ALL 1,000 Kalshi markets, including hundreds of niche predictions
2. **Low Interest**: Most markets have minimal trading volume
3. **Information Asymmetry**: Obscure events attract few informed traders
4. **Market Maker Risk**: Wide spreads compensate for inventory risk in thin markets

### Impact on RL Training

The model correctly learns that:
- Trading wide-spread markets = guaranteed loss
- Best action = HOLD (do nothing)
- No profitable opportunities exist

This is the RIGHT conclusion for the WRONG data!

## Solution: Market Liquidity Filtering

### Implementation Strategy

#### Phase 1: Immediate Fixes
1. **Filter Training Data**
   - Only train on markets with <10¢ average spread
   - Expected to reduce from 1,000 to ~50-100 markets
   - Focus on presidential, congressional, and major economic events

2. **Add Spread Features to Observations**
   ```python
   new_features = {
       'spread_cents': current_spread,
       'spread_pct': spread / mid_price,
       'avg_spread_10min': rolling_average,
       'spread_volatility': spread_std
   }
   ```

3. **Implement Market Selection Logic**
   ```python
   def filter_liquid_markets(session_data, max_spread_cents=10):
       """Only include markets with reasonable spreads"""
       return [m for m in session_data.markets 
               if m.avg_spread <= max_spread_cents]
   ```

#### Phase 2: Advanced Improvements
1. **Tiered Training**
   - Separate models for different liquidity tiers
   - Tier 1: <5¢ spreads (aggressive trading)
   - Tier 2: 5-10¢ spreads (selective trading)
   - Tier 3: >10¢ spreads (market making only)

2. **Dynamic Spread Thresholds**
   - Adjust based on expected edge
   - Trade wide spreads only with high conviction
   - Incorporate time to expiry (spreads narrow near resolution)

3. **Market Making Strategy**
   - For semi-liquid markets (10-20¢ spreads)
   - Post limit orders inside the spread
   - Capture spread while providing liquidity

## Expected Outcomes After Implementation

### Before (Current State)
- Training on 1,000 markets, 80% illiquid
- Model learns trading = loss
- Negative portfolio returns
- 20% win rate

### After (With Filtering)
- Training on 50-100 liquid markets
- Model learns profitable patterns
- Positive expected value
- 45-55% win rate target

## Training Recommendations

### Best Sessions for Liquid Markets
- **Session 41**: More recent data, potentially better liquidity
- **Session 70**: Shorter duration, focused collection
- **Avoid Session 32**: Too many illiquid markets

### Optimal Training Parameters
```bash
uv run python src/kalshiflow_rl/training/train_sb3.py \
  --sessions 41,70 \
  --algorithm ppo \
  --total-timesteps 500000 \
  --max-spread 10 \           # NEW: Filter for liquid markets
  --spread-features true \     # NEW: Include spread in observations
  --min-episode-length 100 \
  --model-save-path models/kalshi_rl_liquid.zip
```

## Validation Metrics

Track these metrics to verify improvement:
1. **Average spread of training markets**: Should be <10¢
2. **Percentage of profitable episodes**: Target >40%
3. **Average trade P&L**: Should exceed spread cost
4. **Sharpe ratio**: Target >0.5
5. **Win rate**: Target 45-55%

## Key Insights

1. **The data is correct**: We're accurately capturing Kalshi's market structure
2. **Most markets are untradeable**: This is reality, not a bug
3. **Quality over quantity**: Better to train on 50 liquid markets than 1,000 mixed
4. **Spread awareness is critical**: Model needs to know the cost of trading
5. **Market selection is a skill**: Identifying tradeable markets is part of the strategy

## Conclusion

The wide spread problem is not a data collection or interpretation error - it's the reality of Kalshi's market structure. The solution is to filter training data to focus on liquid, tradeable markets where profitable strategies can actually be learned. This requires adding spread-based filtering, including spread features in observations, and potentially developing separate strategies for different liquidity tiers.

By training exclusively on liquid markets (<10¢ spreads), we expect to see positive portfolio returns and meaningful learning of profitable trading patterns.