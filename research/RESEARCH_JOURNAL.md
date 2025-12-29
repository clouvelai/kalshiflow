# Quant Research Journal

This is the living journal of all research sessions. The quant agent MUST update this file at the start and end of every research session.

---

## How to Use This Journal

### At Session Start
1. Read the "Active Research Questions" and "Recent Sessions" sections
2. Review any incomplete work from previous sessions
3. Add a new session entry with date and objectives

### At Session End
1. Document what was tested and results
2. Update "Active Research Questions" with new questions discovered
3. Move completed questions to "Resolved Questions"
4. Update the "Hypothesis Tracker" table
5. **Update VALIDATED_STRATEGIES.md** if any strategies were validated or rejected (see workflow below)

### Strategy Validation Workflow
When you validate or reject a strategy, you MUST also update:
`backend/src/kalshiflow_rl/traderv3/planning/VALIDATED_STRATEGIES.md`

This document is the bridge between research and implementation - the trader-specialist agent uses it to know what strategies to code.

---

## Active Research Questions

*Questions we're actively investigating. Pick one to continue or add new ones.*

1. **[RESOLVED]** Are there time-of-day patterns in Kalshi trading? -> Weak patterns, not actionable
2. **[RESOLVED]** Do category-specific strategies exist? -> Yes, but subsets of base strategy
3. **[RESOLVED]** Can we detect informed traders before price moves? -> No, whale trades are just price
4. **[RESOLVED]** Is there edge in fading whale consensus (contrarian)? -> No, it's just price
5. **[RESOLVED]** Do new markets show initial mispricing? -> No actionable edge
6. **[RESOLVED-Session003]** Is there edge at 70-80c range? -> YES! NO at 70-80c has +51.3% edge
7. **[OPEN]** Can execution timing (slippage) be optimized?
8. **[OPEN]** Does edge change as market approaches expiry?
9. **[OPEN]** Are there cross-market correlations to exploit?
10. **[OPEN]** Can category-specific strategies (KXBTCD, KXNCAAMBGAME) scale?

---

## Resolved Questions

*Questions we've answered with data. Include the answer and link to detailed analysis.*

| Question | Answer | Evidence | Date |
|----------|--------|----------|------|
| Is whale-following profitable at 30-70c? | NO - concentration risk | >30% profit from single markets | 2024-12-28 |
| Does YES at 80-90c have edge? | YES - +5.1% | 2,110 markets, validated | 2024-12-28 |
| Does following 100% whale consensus work? | NO - 27% win rate | Contrarian might work | 2024-12-28 |
| Does FADING whale consensus work? | NO - just a price proxy | +1-4% marginal over base | 2025-12-29 |
| Are there time-of-day patterns? | WEAK - not actionable | Hour 07 NO has edge but negative profit | 2025-12-29 |
| Do category-specific strategies exist? | YES - subsets of base | NCAAF totals, NFL spreads higher edge | 2025-12-29 |
| Do early trades have edge? | NO - same as base strategy | Early NO = general NO at high prices | 2025-12-29 |
| Does price momentum/reversion work? | NO - negative profit | +1-2% edge but loses money | 2025-12-29 |
| Does trade sequencing predict outcomes? | NO - fails concentration | Sequential patterns not reliable | 2025-12-29 |
| Does NO at 70-80c have edge? | **YES - +51.3%** | 1,437 mkts, $1M profit, 23.7% conc | 2025-12-29 |
| Does NO at 60-70c have edge? | YES - +30.5% | 1,321 mkts, passes validation | 2025-12-29 |

---

## Hypothesis Tracker

*Master list of all hypotheses tested. Status: Validated | Rejected | In Progress | Queued*

| ID | Hypothesis | Status | Edge | Markets | Notes |
|----|------------|--------|------|---------|-------|
| H001 | YES at 80-90c beats market | Validated | +5.1% | 2,110 | Favorite-longshot bias |
| H002 | NO at 80-90c beats market | Validated | +3.3% | 2,808 | Same mechanism |
| H003 | Follow whales at 30-70c | Rejected | - | - | Concentration >30% |
| H004 | Follow 100% whale consensus | Rejected | -22% | - | Actually contrarian signal |
| H005 | Time-of-day patterns | Rejected | weak | 612 | Hour 07 NO has +15.7% but loses money |
| H006 | Category-specific edge | Validated | varies | varies | NCAAF/NFL subsets show higher edge |
| H007 | Fade whale consensus | Rejected | +1-4% | ~5k | Just a price proxy, no real improvement |
| H008 | New market mispricing | Rejected | +20% | 5,394 | Early NO = same as general NO |
| H009 | Price velocity/momentum | Rejected | +1-2% | ~5k | Positive edge but negative profit |
| H010 | Trade sequencing | Rejected | +13% | ~5k | Fails concentration test |
| H011 | Volume-weighted signals | Validated | same | ~3k | Confirms base strategy works at all volumes |
| H012 | Round number effects | Rejected | - | - | No actionable edge found |
| H013 | Trade intensity patterns | Rejected | - | - | No unique edge over base |
| H014 | Contract size (whale vs retail) | Rejected | same | ~1k | Same direction, whales slightly better |
| H015 | First trade effect | Rejected | - | - | No edge over later trades |
| H016 | Consecutive trade direction | Rejected | - | - | Not predictive |
| H017 | Day of week patterns | Rejected | - | - | No reliable edge |
| H018 | Price movement patterns | Rejected | - | - | Just price proxies |
| H019 | NO at 70-80c range | **Validated** | +51.3% | 1,437 | NEW! Highest edge strategy |
| H020 | Dollar volume per market | Rejected | - | - | No unique edge |
| H021 | NO at 60-70c range | Promising | +30.5% | 1,321 | Valid but lower edge |
| H022 | Category-specific (KXBTCD, KXNCAAMBGAME) | Promising | varies | 60-120 | Need more data |

---

## Session Log

### Session 003 - 2025-12-29
**Objective**: Find NEW validated strategies beyond the existing price-based ones
**Continuing from**: Session 002 (price is the primary signal)
**Analyst**: Quant Agent (Opus 4.5)
**Duration**: ~2 hours
**Session Status**: COMPLETED - NEW STRATEGY VALIDATED

**Mission**: Find at least one additional strategy to add to VALIDATED_STRATEGIES.md

**Hypotheses Tested**:
1. H013: Trade intensity patterns (high vs low volume markets) -> No unique edge
2. H014: Contract size patterns (whale vs retail) -> Whales have slightly higher edge but same direction
3. H015: First trade effect -> No significant difference from later trades
4. H016: Consecutive trade direction -> No actionable edge
5. H017: Day of week patterns -> No reliable edge
6. H018: Price movement patterns (after rise/fall) -> Just price proxies
7. H019: Granular price buckets (5c) -> Found NO at 70-80c!
8. H020: Dollar volume per market -> No unique edge
9. H021: NO at 60-70c range -> VALID but lower edge
10. H022: Category-specific (KXNCAAMBGAME, KXBTCD) -> Promising but small samples

**Key Findings**:

1. **NEW VALIDATED STRATEGY: NO at 70-80c**
   - Markets: 1,437
   - Win Rate: 76.5%
   - Breakeven: 25.3%
   - Edge: +51.3% (HIGHEST of all strategies!)
   - Historical Profit: $1,016,046
   - Max Concentration: 23.7%
   - P-value: < 0.0001
   - **Added to VALIDATED_STRATEGIES.md as S004**

2. **Promising Candidate: NO at 60-70c**
   - Markets: 1,321
   - Win Rate: 66.1%
   - Breakeven: 35.6%
   - Edge: +30.5%
   - Passes validation but lower edge than 70-80c

3. **The Edge Pattern Is Clear**:
   - All validated strategies are NO bets at high YES prices
   - The lower the YES price threshold, the higher the edge but lower win rate
   - NO at 90-100c: +1.2% edge, 96.5% WR
   - NO at 80-90c: +3.3% edge, 87.8% WR
   - NO at 70-80c: +51.3% edge, 76.5% WR (NEW!)
   - NO at 60-70c: +30.5% edge, 66.1% WR (promising)

4. **Category-Specific Insights**:
   - KXNCAAMBGAME (college basketball): +63.8% edge at 70-80c
   - KXBTCD (Bitcoin daily): Consistent edge across all ranges
   - KXMVESPORTSMULTIGAMEEXTENDED (eSports): +57.6% edge at 70-80c

**Strategies REJECTED**:
- Trade intensity filtering (no improvement)
- First trade timing (no edge over later trades)
- Consecutive trade momentum (not predictive)
- Day of week patterns (no reliable edge)
- Whale-specific filtering (same direction as base)

**Files Created**:
- `research/analysis/session003_fresh_hypotheses.py` - Initial hypothesis testing
- `research/analysis/session003_deep_dive.py` - Corrected edge calculations
- `research/analysis/session003_new_strategies.py` - Final strategy validation

**Documents Updated**:
- `backend/src/kalshiflow_rl/traderv3/planning/VALIDATED_STRATEGIES.md` - Added S004

**Recommendations for V3 Trader**:
1. Implement NO at 70-80c (S004) as next priority after S002
2. Consider combined strategy running S001-S004 together
3. Monitor NO at 60-70c for potential future addition
4. Category-specific strategies (KXBTCD, KXNCAAMBGAME) need more data

---

### Session 002 - 2025-12-29
**Objective**: Deep pattern hunting - explore untested hypotheses and find new trading edges
**Continuing from**: Session 001 findings (whale consensus is anti-predictive)
**Analyst**: Quant Agent (Opus 4.5)
**Duration**: ~3 hours
**Session Status**: COMPLETED

**Hypotheses Tested**:
1. H007: Fade whale consensus -> REJECTED (just a price proxy)
2. H005: Time-of-day patterns -> REJECTED (weak, not actionable)
3. H006: Category-specific efficiency -> VALIDATED (subsets show higher edge)
4. H008: New market mispricing -> REJECTED (early NO = general NO)
5. H009: Price velocity/momentum -> REJECTED (negative profit)
6. H010: Trade sequencing patterns -> REJECTED (fails concentration)
7. H011: Volume-weighted signals -> VALIDATED (confirms base strategy)
8. H012: Round number effects -> REJECTED (no edge found)

**Key Findings**:
1. **PRICE IS THE PRIMARY SIGNAL**: All validated strategies are price-based. Every "enhancement" is really just a proxy for price.
2. **WHALE BEHAVIOR IS NOT INFORMATIVE**: The apparent "fade whale consensus" pattern is just betting high-price favorites (= base strategy).
3. **THE LONGSHOT BIAS IS THE EDGE**: YES bets at low prices systematically lose; NO bets at high prices systematically win.
4. **COMPLEXITY DOES NOT HELP**: Every attempt to add complexity failed validation or provided no improvement.

**Validated Strategies (Tier 1)**:
- YES at 80-90c: +5.1% edge, $1.6M profit (KEEP)
- NO at 80-90c: +3.3% edge, $708k profit (KEEP)
- NO at 90-100c: +1.2% edge, $463k profit (KEEP)

**Category Enhancements (Tier 2)**:
- KXNCAAFTOTAL NO at 70-80c: +68.3% edge, $25k profit
- KXNFLSPREAD NO at 70-80c: +41.0% edge, $63k profit
- KXBTCD NO at 80-90c: +72.4% edge, $13k profit

**Strategies Rejected**:
- Whale following/fading (no improvement over price)
- Time-of-day strategies (inconsistent edge)
- Price momentum/reversion (negative profit)
- Trade sequencing (fails concentration)
- Round number effects (no edge)

**Recommendations for V3 Trader**:
1. KEEP YES_80_90 strategy as primary
2. Consider adding NO_80_90 for diversification
3. DO NOT add whale-based complexity
4. Focus execution research on slippage optimization

**Files Created**:
- `research/analysis/session002_deep_analysis.py` - Main hypothesis testing
- `research/analysis/session002_whale_fade_deep_dive.py` - Whale consensus validation
- `research/analysis/session002_combined_strategy.py` - Price vs whale comparison
- `research/strategies/validated/SESSION002_FINDINGS.md` - Full findings report

---

### Session 001 - 2024-12-28
**Objective**: Analyze whale-following strategies from public trade feed
**Duration**: ~2 hours
**Analyst**: Quant Agent

**Hypotheses Tested**:
1. H003: Whale-following at moderate prices → REJECTED (concentration)
2. H004: 100% whale consensus → REJECTED (27% win rate, contrarian signal)

**Key Findings**:
- Whale-following fails validation due to profit concentration in single markets
- When 100% of whales agree, following them LOSES money
- This suggests contrarian whale-fading might work (H007 queued)

**New Questions Generated**:
- Should we FADE whale consensus instead of following?
- Are there specific whale SIZE thresholds that matter?
- Do whales behave differently in different categories?

**Next Steps**:
- Test H007 (fade whale consensus)
- Investigate time-of-day patterns (H005)
- Look at category-specific behaviors

**Files Created/Modified**:
- `research/strategies/MVP_STRATEGY_IDEAS.md` - Main analysis doc
- `research/strategies/rejected/whale-following-analysis.md`
- `research/analysis/public_trade_feed_analysis.py`

---

*Add new sessions above this line, keeping most recent at top*

---

## Data Inventory

*What data do we have available for analysis?*

| Dataset | Records | Date Range | Location |
|---------|---------|------------|----------|
| Public trades | ~1.7M | All time | `research/data/trades/` |
| Settled markets | ~78k | All time | `research/data/markets/` |
| Market outcomes | ~78k | All time | `research/data/markets/market_outcomes_ALL.csv` |
| Enriched trades | ~1.7M | All time | `research/data/trades/enriched_trades_resolved_ALL.csv` |

---

## Promising Leads

*Patterns or anomalies noticed but not yet fully investigated*

1. **Category-specific higher edge**: NCAAF totals, NFL spreads, crypto show higher edge than base (need more data)
2. **Time to expiry patterns**: Edge might change as market approaches settlement (not yet tested)
3. **Cross-market correlations**: Related markets might provide signals (not yet tested)
4. **Execution optimization**: Slippage and timing improvements could increase realized edge

---

## Dead Ends (Don't Revisit)

*Approaches we've thoroughly tested and confirmed don't work*

1. **Simple whale-following at any price**: Fails concentration test
2. **Following unanimous whale consensus**: Negative edge (-22%)
3. **Fading whale consensus**: Just a price proxy, no real improvement over base strategy
4. **Time-of-day patterns**: Weak edge, negative profit on validated patterns
5. **Early trade mispricing**: Same as general strategy, no unique edge
6. **Price momentum/reversion**: Positive edge but NEGATIVE profit
7. **Trade sequencing patterns**: Fails concentration test
8. **Round number effects**: No actionable edge found

---
