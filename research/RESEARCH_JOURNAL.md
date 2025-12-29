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

---

## Active Research Questions

*Questions we're actively investigating. Pick one to continue or add new ones.*

1. **[OPEN]** Are there time-of-day patterns in Kalshi trading? (morning vs evening edge)
2. **[OPEN]** Do category-specific strategies exist? (sports vs politics vs crypto)
3. **[OPEN]** Can we detect informed traders before price moves?
4. **[OPEN]** Is there edge in fading whale consensus (contrarian)?
5. **[OPEN]** Do new markets show initial mispricing?

---

## Resolved Questions

*Questions we've answered with data. Include the answer and link to detailed analysis.*

| Question | Answer | Evidence | Date |
|----------|--------|----------|------|
| Is whale-following profitable at 30-70c? | NO - concentration risk | >30% profit from single markets | 2024-12-28 |
| Does YES at 80-90c have edge? | YES - +5.1% | 2,110 markets, validated | 2024-12-28 |
| Does following 100% whale consensus work? | NO - 27% win rate | Contrarian might work | 2024-12-28 |

---

## Hypothesis Tracker

*Master list of all hypotheses tested. Status: ✅ Validated | ❌ Rejected | 🔄 In Progress | 📋 Queued*

| ID | Hypothesis | Status | Edge | Markets | Notes |
|----|------------|--------|------|---------|-------|
| H001 | YES at 80-90c beats market | ✅ | +5.1% | 2,110 | Favorite-longshot bias |
| H002 | NO at 80-90c beats market | ✅ | +3.3% | 2,808 | Same mechanism |
| H003 | Follow whales at 30-70c | ❌ | - | - | Concentration >30% |
| H004 | Follow 100% whale consensus | ❌ | -22% | - | Actually contrarian signal |
| H005 | Time-of-day patterns | 📋 | ? | ? | Not yet tested |
| H006 | Category-specific edge | 📋 | ? | ? | Not yet tested |
| H007 | Fade whale consensus | 📋 | ? | ? | Suggested by H004 failure |
| H008 | New market mispricing | 📋 | ? | ? | Not yet tested |

---

## Session Log

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

1. **Whale consensus contrarian**: When 100% of whales agree, fading them might work (27% follow = 73% fade?)
2. **Category efficiency variance**: Some categories might be less efficient than others
3. **Time decay patterns**: Edge might change as market approaches expiry
4. **Round number effects**: Prices might cluster at 25c, 50c, 75c

---

## Dead Ends (Don't Revisit)

*Approaches we've thoroughly tested and confirmed don't work*

1. **Simple whale-following at any price**: Fails concentration test
2. **Following unanimous whale consensus**: Negative edge

---
