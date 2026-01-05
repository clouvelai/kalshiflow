# LSD Mode: Lateral Strategy Discovery

> "The good acid" - User, 2025-12-29

## What is LSD Mode?

LSD (Lateral Strategy Discovery) is an exploration mode for the quant agent that prioritizes **speed and creativity over rigor**. The goal is to rapidly screen many hypotheses to find promising candidates for full validation.

## Enabling LSD Mode

**To enter LSD mode:** User says "LSD mode" or "enter LSD mode"
**To exit LSD mode:** User says "normal mode" or "exit LSD mode"

## LSD Mode Rules

### 1. Speed Over Rigor
- Quick 10-minute edge checks per hypothesis
- Skip bucket-by-bucket baseline comparison for initial screening
- Use simple edge calculation: `(win_rate - breakeven_rate)`
- Save full validation for promising candidates

### 2. Lower Threshold
Flag anything with:
- Raw edge > 5%, OR
- Improvement over naive baseline > 3%

(Normal mode requires bucket-matched +8% improvement)

### 3. Absurdity Encouraged
Test combinations that seem "wrong":
- "What if leverage variance + time of day + trade size?"
- "What if we combine 3 weak signals?"
- "What would the OPPOSITE of our validated strategies look like?"
- Moon phases, fibonacci numbers, prime trade counts

### 4. Volume Over Quality
- Test 10 ideas in LSD mode vs 2 in normal mode
- Finding 1 promising candidate out of 10 is SUCCESS
- Quick reject and move on - no attachment to hypotheses

### 5. Document Everything
Even failures are data. Note WHY something failed:
- "Price proxy - edge disappeared at same price buckets"
- "Insufficient sample - only 23 markets matched"
- "Signal too rare - <1% of trades qualify"

---

## LSD Mode Techniques

### Technique 1: Signal Stacking
Combine 2-3 weak signals that failed individually:
```python
# Example: All individually failed, but combined?
condition = (
    leverage > 2 AND
    is_weekend AND
    hour >= 18 AND hour <= 23 AND
    trade_count > 5 AND
    no_ratio > 0.5
)
```

### Technique 2: Negation Testing
What if we do the OPPOSITE of validated strategies?
- S013 bets NO on low leverage variance → What about HIGH variance?
- S007 fades high leverage → What about FOLLOWING high leverage?
- If whales are wrong, are minnows right?

### Technique 3: Extreme Parameter Sweep
Push thresholds to extremes:
```
Tested: leverage > 2
Try: leverage > 5, > 10, > 20
Tested: no_ratio > 0.5
Try: no_ratio > 0.8, > 0.9, == 1.0
```

### Technique 4: Cross-Dimension Mixing
Combine dimensions never tested together:
```
Time (hour) × Size (count) × Price (leverage) × Sequence (run length)
Category × Day of week × Trade velocity × Whale presence
```

### Technique 5: Absurd Combinations (AGGRESSIVE)
Test things that "shouldn't" work:

**Temporal Absurdities:**
- Moon phase (new moon vs full moon)
- Day of month (1st, 15th, end of month)
- Market "age" (exactly 24 hours old)

**Numerological Patterns:**
- Fibonacci trade counts (5, 8, 13, 21 trades)
- Markets with prime number of trades
- Trade sizes that are powers of 2

**Anti-Patterns:**
- What's the WORST strategy? Avoid those markets
- Inverse of every validated strategy
- "Cursed" markets: what do losing trades have in common?

**Meta-Strategies:**
- "When would a bot NOT trade?" → Trade there
- "What would a drunk person do?" → Do the opposite
- Markets with no whale activity (retail-only paradise?)

---

## LSD Mode Output Format

### Quick Screening Table
| Hypothesis | Signal | Quick Edge | Threshold? | Notes | Full Validate? |
|------------|--------|------------|------------|-------|----------------|
| H150 | Fib trade count | +7.2% | YES | Interesting | YES |
| H151 | Moon phase | +1.1% | NO | No pattern | NO |
| H152 | 4-signal stack | +12.3% | YES | Check methodology | MAYBE |

### Promising Candidate Brief
```markdown
## H150: Fibonacci Trade Count Signal

**Quick Edge:** +7.2%
**Markets:** 312
**Signal:** Markets with exactly 8, 13, or 21 trades

**Why Promising:**
- Edge > 5% threshold
- Reasonable sample size
- Not obviously a price proxy (trade count independent of price)

**Concerns:**
- Need to verify temporal stability
- May be overfitting to specific trade count values

**Recommendation:** FULL VALIDATION
```

---

## LSD Session Workflow

### Phase 1: Hypothesis Intake
1. Read incoming hypothesis briefs from `research/hypotheses/incoming/`
2. Review any untested hypotheses from previous sessions
3. Generate new absurd combinations using LSD techniques

### Phase 2: Quick Screening
For each hypothesis (10 min max):
1. Write quick analysis script
2. Calculate raw edge
3. Check sample size (need > 50 markets)
4. Flag if edge > 5%
5. Document result in screening table

### Phase 3: Session Summary
1. Create session results file
2. List all tested hypotheses with results
3. Highlight promising candidates
4. Recommend which to fully validate

---

## Key Differences: LSD vs Normal Mode

| Aspect | LSD Mode | Normal Mode |
|--------|----------|-------------|
| Edge threshold | > 5% raw | > 8% bucket-matched improvement |
| Time per hypothesis | ~10 min | ~30-60 min |
| Bucket analysis | Skip | Required |
| Temporal stability | Skip | Required |
| Bonferroni correction | Skip | Required |
| Absurd ideas | Encouraged | Rare |
| Goal | Find candidates | Validate strategies |

---

## Success Metrics for LSD Sessions

A successful LSD session:
- [ ] Tests 8-12 hypotheses
- [ ] Documents all results (even failures)
- [ ] Finds 1-3 promising candidates
- [ ] Completes in 2-3 hours
- [ ] Generates new hypothesis ideas for next session

---

## Example LSD Session Log

### Session LSD-001 - 2025-12-29

**Mode:** LSD (Aggressive)
**Duration:** 2.5 hours
**Hypotheses Tested:** 11
**Promising Hits:** 2

#### Screening Results

| ID | Hypothesis | Edge | Pass? | Notes |
|----|------------|------|-------|-------|
| L001 | Moon phase (full moon) | +0.3% | NO | No lunar correlation |
| L002 | Fibonacci trade counts | +6.8% | YES | Curious pattern |
| L003 | Prime trade counts | +1.2% | NO | Near random |
| L004 | Powers of 2 size | +2.1% | NO | Weak signal |
| L005 | 4-signal stack (lev+time+size+cat) | +8.9% | YES | Strong! |
| L006 | Inverse of S013 (high lev var) | -2.4% | NO | Confirms S013 correct |
| L007 | Minnow swarm | +3.2% | NO | Below threshold |
| L008 | No whale markets | +4.1% | NO | Just below |
| L009 | Day of month (1st) | +0.8% | NO | No pattern |
| L010 | Market age = 24h | +5.2% | YES | Investigate |
| L011 | VIX correlation | N/A | NO | Need external data |

#### Promising Candidates for Full Validation
1. **L002: Fibonacci Trade Counts** - Edge: +6.8%, 312 markets
2. **L005: 4-Signal Stack** - Edge: +8.9%, 187 markets
3. **L010: Market Age = 24h** - Edge: +5.2%, 203 markets (marginal)

---

*LSD Mode: Where absurd ideas become validated strategies.*

---

## Validation Framework (REQUIRED)

### Overview

We have built an automated validation framework to replace ad-hoc analysis scripts. **Always use this framework** instead of writing one-off scripts.

**Location:** `research/scripts/validation/`

### Usage

```bash
cd backend

# Quick LSD screening (~6 seconds with cache)
uv run python ../research/scripts/validate_strategy.py --config h123_rlm.yaml --mode lsd

# Full rigorous validation (~6 seconds with cache)
uv run python ../research/scripts/validate_strategy.py --config h123_rlm.yaml --mode full

# Save results to JSON
uv run python ../research/scripts/validate_strategy.py --config h123_rlm.yaml --output ../research/reports/my_result.json

# List all available strategy configs
uv run python ../research/scripts/validate_strategy.py --list
```

### Creating a New Strategy Config

Add a YAML file to `research/strategies/configs/`:

```yaml
# research/strategies/configs/my_new_strategy.yaml
strategy:
  name: "My New Hypothesis"
  hypothesis_id: "H999"
  action: "bet_no"  # or "bet_yes"
  description: "Description of the signal logic"

signal:
  conditions:
    - field: "yes_trade_ratio"
      operator: ">"
      value: 0.70
    - field: "n_trades"
      operator: ">="
      value: 15
    # Add more conditions as needed

validation:
  min_markets: 50
  significance_level: 0.05
```

### Complete Signal Reference

This is the **complete inventory** of all signals available for strategy validation. The signals are computed from ~7.9M trades across ~316K markets.

---

#### TIER 1: Core Directional Signals (Most Predictive)

These signals have shown the most predictive power across validated strategies.

| Field | Type | Description | Range | Example | Predictive Power |
|-------|------|-------------|-------|---------|------------------|
| `yes_trade_ratio` | float | Fraction of trades that are YES bets | 0.0-1.0 | `> 0.65` | **HIGH** - Core of RLM strategy |
| `no_trade_ratio` | float | Fraction of trades that are NO bets | 0.0-1.0 | `> 0.50` | **HIGH** - Inverse of yes_trade_ratio |
| `price_moved_toward_no` | bool | YES price dropped from first to last trade | True/False | `== true` | **HIGH** - Core of RLM strategy |
| `yes_price_dropped` | bool | Same as price_moved_toward_no (alias) | True/False | `== true` | **HIGH** |
| `yes_price_drop` | int | Amount YES price dropped (cents) | -100 to +100 | `>= 5` | **HIGH** - Position sizing signal |
| `price_move_toward_no` | int | Same as yes_price_drop (alias) | -100 to +100 | `> 10` | **HIGH** |

**Best Use**: Combine `yes_trade_ratio > 0.65` with `price_moved_toward_no == true` for RLM strategy.

---

#### TIER 2: Volume and Size Signals

| Field | Type | Description | Range | Example | Predictive Power |
|-------|------|-------------|-------|---------|------------------|
| `n_trades` | int | Number of trades in market | 1-10000+ | `>= 15` | **MEDIUM** - Filters noise |
| `total_contracts` | int | Total contracts traded | 1-100000+ | `> 500` | **MEDIUM** |
| `total_value_cents` | int | Total trade value in cents | 100-10M+ | `> 100000` | **MEDIUM** |
| `avg_trade_size` | float | Average contracts per trade | 1.0-1000+ | `> 50` | **MEDIUM** |
| `avg_trade_value` | float | Average trade value in cents | 1-10000+ | `> 1000` | **MEDIUM** |
| `yes_trade_count` | int | Number of YES trades | 0-10000+ | `> 10` | **LOW** |
| `no_trade_count` | int | Number of NO trades | 0-10000+ | `> 10` | **LOW** |

**Best Use**: `n_trades >= 15` as a filter to ensure sufficient market activity.

---

#### TIER 3: Price Level Signals

| Field | Type | Description | Range | Example | Predictive Power |
|-------|------|-------------|-------|---------|------------------|
| `avg_yes_price` | float | Average YES price across trades (cents) | 1-99 | `< 50` | **HIGH** - Defines breakeven |
| `avg_no_price` | float | Average NO price across trades (cents) | 1-99 | `> 70` | **HIGH** - Defines breakeven |
| `first_yes_price` | int | Opening YES price (cents) | 1-99 | `< 60` | **LOW** |
| `last_yes_price` | int | Closing YES price (cents) | 1-99 | `< 50` | **LOW** |
| `first_no_price` | int | Opening NO price (cents) | 1-99 | `> 40` | **LOW** |
| `last_no_price` | int | Closing NO price (cents) | 1-99 | `> 50` | **LOW** |
| `yes_price_std` | float | Volatility of YES price (std dev) | 0-50+ | `< 5` | **LOW** |
| `yes_price_bucket` | float | 5-cent bucket for YES price | 0,5,10...95 | `== 65` | Used for baseline |
| `no_price_bucket` | float | 5-cent bucket for NO price | 0,5,10...95 | `== 30` | Used for baseline |

**Warning**: Price-level signals often create "price proxies" - edge that's just the baseline for that price.

---

#### TIER 4: Whale and Size Distribution Signals

| Field | Type | Description | Range | Example | Predictive Power |
|-------|------|-------------|-------|---------|------------------|
| `has_whale` | bool | Market has any trade >= $100 | True/False | `== true` | **MEDIUM** |
| `whale_trade_count` | int | Number of trades >= $100 | 0-1000+ | `>= 3` | **MEDIUM** |
| `round_size_count` | int | Trades with round sizes (10,25,50,100,250,500,1000) | 0-1000+ | `> 5` | **LOW** - Bot detection |
| `max_trade_size` | int | Largest single trade (contracts) | 1-10000+ | `> 500` | **MEDIUM** |
| `min_trade_size` | int | Smallest single trade (contracts) | 1-10000+ | `>= 10` | **LOW** |
| `trade_size_range` | int | max - min trade size | 0-10000+ | `> 100` | **LOW** |

**Best Use**: `has_whale == true` combined with other signals. Whale signals alone are often price proxies.

---

#### TIER 5: Leverage Signals

| Field | Type | Description | Range | Example | Predictive Power |
|-------|------|-------------|-------|---------|------------------|
| `avg_leverage` | float | Average leverage ratio (potential_profit/cost) | 0.01-100+ | `> 2.0` | **LOW** - Often price proxy |
| `leverage_std` | float | Volatility of leverage ratios | 0-50+ | `< 0.7` | **MEDIUM** - Bot detection |

**Warning**: Leverage is highly correlated with price. Low leverage = high NO price. High leverage = low NO price.

**Best Use**: `leverage_std < 0.7` for bot-like consistent trading patterns.

---

#### TIER 6: Temporal Signals

| Field | Type | Description | Range | Example | Predictive Power |
|-------|------|-------------|-------|---------|------------------|
| `first_trade_time` | datetime | When first trade occurred | Datetime | N/A (filtering) | **LOW** |
| `last_trade_time` | datetime | When last trade occurred | Datetime | N/A (filtering) | **LOW** |
| `market_duration_hours` | float | Time from first to last trade | 0-1000+ | `> 24` | **LOW** |
| `week` | int | ISO week number | 1-53 | `in [48, 49, 50]` | Used for temporal stability |
| `avg_hour` | float | Average hour of trades (0-23) | 0.0-23.99 | `>= 22` | **LOW** - Time of day |
| `weekend_trade_ratio` | float | Fraction of trades on weekend | 0.0-1.0 | `> 0.5` | **LOW** |
| `is_mostly_weekend` | bool | >50% of trades on weekend | True/False | `== true` | **LOW** |
| `is_late_night` | bool | Avg hour 22:00-06:00 | True/False | `== true` | **LOW** - "Drunk betting" |
| `is_morning` | bool | Avg hour 06:00-12:00 | True/False | `== true` | **LOW** |
| `is_afternoon` | bool | Avg hour 12:00-18:00 | True/False | `== true` | **LOW** |
| `is_evening` | bool | Avg hour 18:00-22:00 | True/False | `== true` | **LOW** |

**Best Use**: Temporal signals are mostly useful for filtering or temporal stability checks. Note: Time-based signals (weekend, late night) have NOT shown consistent edge in validation.

---

#### TIER 7: Outcome Signals (For Analysis Only)

These are computed from settlement data and cannot be used for live trading signals.

| Field | Type | Description | Range | Example | Notes |
|-------|------|-------------|-------|---------|-------|
| `market_result` | str | Actual outcome | "yes", "no", "" | N/A | Ground truth |
| `is_resolved` | bool | Market has settled | True/False | `== true` | Filter for analysis |
| `no_won` | bool | NO outcome won | True/False | N/A | Convenience field |
| `yes_won` | bool | YES outcome won | True/False | N/A | Convenience field |

---

#### TIER 8: Category Signal

| Field | Type | Description | Example Values | Predictive Power |
|-------|------|-------------|----------------|------------------|
| `category` | str | Market category prefix | "KXNFL", "KXNBA", "KXBTCD", "KXNCAAF" | **MEDIUM** |

**Top Categories by Volume**:
- `KXNFL*` - NFL football (spreads, totals, props)
- `KXNBA*` - NBA basketball
- `KXNCAAF*` - College football
- `KXBTCD*` - Bitcoin daily
- `KXETH*` - Ethereum daily

**Best Use**: Category is useful for understanding edge distribution, but category-specific strategies often fail the "not a price proxy" test.

---

#### TIER 9: Price Movement Signals (Less Common)

| Field | Type | Description | Range | Example | Predictive Power |
|-------|------|-------------|-------|---------|------------------|
| `no_price_dropped` | bool | NO price dropped from first to last | True/False | `== true` | **LOW** |
| `no_price_drop` | int | Amount NO price dropped (cents) | -100 to +100 | `> 5` | **LOW** |

---

### Signals NOT Currently Exposed (Potential Future Work)

These signals would need to be computed in `cache.py` to use:

| Potential Signal | Description | Why Not Exposed |
|------------------|-------------|-----------------|
| `trade_velocity` | Trades per hour | Requires market duration calculation |
| `size_distribution_skew` | Skewness of trade sizes | Requires scipy in cache computation |
| `trade_size_entropy` | Information entropy of sizes | Requires scipy |
| `first_half_ratio` | YES ratio in first 50% of trades | Requires trade ordering |
| `second_half_ratio` | YES ratio in last 50% of trades | Requires trade ordering |
| `price_oscillation` | Count of price direction changes | Requires trade-by-trade analysis |
| `run_length_max` | Longest same-direction trade streak | Requires trade-by-trade analysis |
| `price_volatility_early` | Std dev of price in first half | Requires trade ordering |
| `price_volatility_late` | Std dev of price in second half | Requires trade ordering |
| `whale_direction` | Dominant whale trade direction | Requires whale-filtered aggregation |
| `whale_consensus` | % of whale trades in same direction | Requires whale-filtered aggregation |

**To add a new signal**: Modify `research/scripts/validation/cache.py` in the `_compute_markets()` method.

**Recently Added (v1.1.0)**: `avg_hour`, `weekend_trade_ratio`, `is_mostly_weekend`, `is_late_night`, `is_morning`, `is_afternoon`, `is_evening`, `max_trade_size`, `min_trade_size`, `trade_size_range`

---

### Operators Reference

| Operator | Description | Example |
|----------|-------------|---------|
| `>` | Greater than | `n_trades > 10` |
| `>=` | Greater than or equal | `n_trades >= 15` |
| `<` | Less than | `avg_yes_price < 50` |
| `<=` | Less than or equal | `avg_yes_price <= 30` |
| `==` | Equals | `has_whale == true` |
| `!=` | Not equals | `market_result != ""` |
| `in` | Value in list | `category in ["KXNFL", "KXNBA"]` |
| `not_in` | Value not in list | `category not_in ["KXBTCD"]` |

---

### Field Aliases

These fields are mapped to their canonical names in the validator:

| Alias | Maps To |
|-------|---------|
| `trade_count` | `n_trades` |
| `price_dropped` | `yes_price_dropped` |
| `price_move_toward_no` (int version) | `yes_price_drop` |

---

### Signal Combination Patterns (Validated)

Based on the Research Journal, these combinations have proven edge:

**RLM (H123) - Best Strategy**:
```yaml
conditions:
  - field: "yes_trade_ratio"
    operator: ">"
    value: 0.65
  - field: "price_moved_toward_no"
    operator: "=="
    value: true
  - field: "n_trades"
    operator: ">="
    value: 15
```
Edge: +21.8%, 4,148 markets, 17/18 buckets positive

**Late Money (SPORTS-007)**:
```yaml
conditions:
  - field: "has_whale"
    operator: "=="
    value: true
  - field: "market_duration_hours"
    operator: ">"
    value: 1  # Trades in final 20% of market
  - field: "whale_trade_count"
    operator: ">="
    value: 2
```
Edge: +19.8%, 331 markets

**Bot Leverage Stability (H102)**:
```yaml
conditions:
  - field: "leverage_std"
    operator: "<"
    value: 0.7
  - field: "round_size_count"
    operator: ">"
    value: 3
  - field: "n_trades"
    operator: ">="
    value: 10
```
Edge: +11.3%, 485 markets

---

### Common Pitfalls

1. **Price Proxy Trap**: Many signals (leverage, category, whale) are just selecting high-NO-price markets where baseline already favors NO.

2. **Insufficient Sample**: Need >= 50 markets for reliable validation. <100 markets is suspicious.

3. **Single Bucket Dominance**: If >90% of signals fall in one price bucket, you're probably just betting the bucket.

4. **Temporal Instability**: A strategy that works in one quarter but fails in others is likely noise.

5. **Look-Ahead Bias**: Never use fields computed from final state (final_price, settlement_time) as signals.

### Validation Modes

| Mode | Time | Use Case |
|------|------|----------|
| `lsd` | ~2-6s | Quick screening, find promising candidates |
| `full` | ~6-30s | Rigorous validation with all statistical checks |

### What Full Validation Checks

1. **Sample size** - Minimum 50 markets
2. **Statistical significance** - p-value < 0.05
3. **Not a price proxy** - >60% of buckets show improvement
4. **Concentration** - No single market >30% of profit
5. **Temporal stability** - >50% of time periods profitable
6. **Bucket-matched improvement** - Edge vs same-price baseline

---

## Tooling Self-Improvement (CRITICAL)

### Philosophy

**Every time you do repetitive work, ask: "Can I automate this?"**

The validation framework exists because we noticed we were writing 400-line scripts over and over. Now it's 20 lines of YAML + one command.

### Self-Improvement Principles

1. **DRY (Don't Repeat Yourself)**
   - If you write similar code twice, abstract it
   - If you run similar commands twice, script them
   - If you copy-paste, something is wrong

2. **Identify Pain Points**
   - What takes the longest?
   - What do you dread doing?
   - What causes the most errors?

3. **Invest in Tooling**
   - Spending 2 hours to save 30 minutes per day = 1 week ROI
   - Good tooling compounds - others benefit too
   - Document everything so future-you can use it

4. **Continuous Improvement Loop**
   ```
   Do work manually → Notice friction → Automate → Document → Repeat
   ```

### Current Tooling Inventory

| Tool | Location | Purpose |
|------|----------|---------|
| **Validation Framework** | `research/scripts/validation/` | Strategy validation |
| **Data Update Script** | `research/scripts/update_research_data.py` | Refresh historical data |
| **Session Cleanup** | `backend/src/kalshiflow_rl/scripts/cleanup_sessions.py` | Clean RL sessions |
| **Model Cleanup** | `backend/src/kalshiflow_rl/scripts/cleanup_trained_models.py` | Clean trained models |

### When to Build New Tools

**BUILD a tool when:**
- You've done the same task 3+ times
- The task takes >10 minutes manually
- Errors are common (automation = consistency)
- Others need to do the same task

**DON'T build a tool when:**
- It's truly a one-off task
- The abstraction would be more complex than the task
- Requirements are still unclear (prototype manually first)

### Tool Quality Standards

Every tool should have:
1. **CLI interface** with `--help`
2. **Dry-run mode** (`--dry-run`) for safe testing
3. **Logging** so you can debug issues
4. **Documentation** in the file header or README
5. **Error handling** with clear messages

### Improvement Backlog

Track automation ideas in `research/TOOLING_BACKLOG.md`:

```markdown
## Tooling Backlog

### High Priority
- [ ] Batch validation runner (run all strategies overnight)
- [ ] Automated data freshness alerts

### Medium Priority
- [ ] Strategy comparison report generator
- [ ] Hypothesis queue management CLI

### Ideas
- [ ] Slack/Discord notifications for completed validations
- [ ] Web UI for validation results
```

### After Each Session

Ask yourself:
1. What manual work did I do that could be automated?
2. Did I copy-paste any code?
3. Did I run similar commands multiple times?
4. What would make the NEXT session faster?

**If you identify an improvement opportunity, either:**
- Implement it immediately (if <30 min)
- Add it to the backlog with priority
- Mention it to the user as a suggestion

---

## Data Management

### Updating Historical Data

The research data should be updated regularly (weekly recommended):

```bash
cd backend

# Update trades and outcomes from production Supabase
uv run python ../research/scripts/update_research_data.py

# Preview what would be updated (no changes)
uv run python ../research/scripts/update_research_data.py --dry-run

# Quick update - trades only, skip slow outcome fetch
uv run python ../research/scripts/update_research_data.py --skip-outcomes
```

### Data Files

| File | Description | Update Frequency |
|------|-------------|------------------|
| `research/data/trades/historical_trades_ALL.csv` | All public trades | Weekly |
| `research/data/markets/market_outcomes_ALL.csv` | Settlement results | Weekly |
| `research/data/trades/enriched_trades_resolved_ALL.csv` | Trades + outcomes | After outcome update |

### Cache Management

The validation framework caches market aggregations in parquet format:

```bash
# Cache location
research/data/cache/

# Clear cache if data was updated externally
rm -rf research/data/cache/*.parquet
```

The cache auto-invalidates when source files change (based on file hash).
