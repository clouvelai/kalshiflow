# Hypothesis Queue

This file tracks incoming hypothesis briefs from the Strategy Researcher agent and their status through the LSD/validation pipeline.

## Queue Status

| Status | Count | Description |
|--------|-------|-------------|
| Incoming | 29 | Fresh from research agent, not yet screened |
| LSD Screening | 0 | Currently being quick-tested in LSD mode |
| Promising | 0 | Passed LSD threshold (>5% edge), awaiting full validation |
| In Validation | 0 | Currently undergoing full bucket-matched validation |
| Validated | 0 | Passed all tests, added to VALIDATED_STRATEGIES.md |
| Rejected | 0 | Failed screening or validation |

---

## Incoming Hypotheses

*Hypothesis briefs from Strategy Researcher agent, awaiting LSD screening*

### External Research (Sports Betting / Academic / Crypto)

| ID | Title | Source | Priority | Date Added |
|----|-------|--------|----------|------------|
| EXT-001 | Early Trade Premium (CLV) | Sports Betting CLV Research | HIGH | 2025-12-29 |
| EXT-002 | Steam Move / Cascade Detection | Sports Betting Steam Research | HIGH | 2025-12-29 |
| EXT-003 | Reverse Line Movement (RLM) | Sports Betting RLM Research | HIGH | 2025-12-29 |
| EXT-004 | VPIN-Style Flow Toxicity | Academic Market Microstructure | MEDIUM | 2025-12-29 |
| EXT-005 | Buy-Back Trap / Late Reversal | Sports Betting Syndicate Tactics | HIGH | 2025-12-29 |
| EXT-006 | Surprise Event Mispricing | Academic In-Play Research | MEDIUM | 2025-12-29 |
| EXT-007 | Correlated Asset Lag | Polymarket Strategy Research | MEDIUM | 2025-12-29 |
| EXT-008 | Endgame Sweep (Near-Certainty) | Polymarket Strategy Research | LOW | 2025-12-29 |
| EXT-009 | Sum-to-One Violation | Vanderbilt Academic Research | MEDIUM | 2025-12-29 |

### LSD Mode Absurd Hypotheses

| ID | Title | Type | Priority | Date Added |
|----|-------|------|----------|------------|
| LSD-001 | Fibonacci Trade Count | Numerological | LSD-ABSURD | 2025-12-29 |
| LSD-002 | Prime Number Trade Count | Numerological | LSD-ABSURD | 2025-12-29 |
| LSD-003 | Inverse Worst Strategy (Curse Avoidance) | Anti-Pattern | LSD-META | 2025-12-29 |
| LSD-004 | 4-Signal Mega Stack | Signal Stacking | LSD-STACK | 2025-12-29 |
| LSD-005 | What Would A Drunk Do? | Meta Strategy | LSD-META | 2025-12-29 |

### Sports Betting Expert Hypotheses (NEW - 2026-01-01)

*Source: `/research/hypotheses/incoming/SPORTS_EXPERT_HYPOTHESES_20260101.md`*

| ID | Title | Source Concept | Priority | Date Added |
|----|-------|---------------|----------|------------|
| SPORTS-001 | Steam Exhaustion Detection | Momentum exhaustion after sharp moves | TIER-1 | 2026-01-01 |
| SPORTS-002 | Opening Move Reversal | Fade the opener (first 25% of trades) | TIER-2 | 2026-01-01 |
| SPORTS-003 | Momentum Velocity Stall | Bearish divergence (velocity drops, price stays) | TIER-1 | 2026-01-01 |
| SPORTS-004 | Extreme Public Sentiment Fade | 90/10 rule - fade 90%+ consensus | TIER-3 | 2026-01-01 |
| SPORTS-005 | Size Velocity Divergence | Trade count up + avg size down = retail | TIER-1 | 2026-01-01 |
| SPORTS-006 | Round Number Retail Clustering | Trades cluster at 25c/50c/75c = retail | TIER-3 | 2026-01-01 |
| SPORTS-007 | Late-Arriving Large Money | Sharp money arrives last | TIER-2 | 2026-01-01 |
| SPORTS-008 | Size Distribution Shape Change | Uniform->Clustered = regime shift | TIER-1 | 2026-01-01 |
| SPORTS-009 | Spread Widening Before Sharp | MMs widen before informed trades | TIER-1 | 2026-01-01 |
| SPORTS-010 | Multi-Outcome Pricing Inconsistency | Related markets should sum to 100% | TIER-2 | 2026-01-01 |
| SPORTS-011 | Category Momentum Contagion | Fade recency bias after upset streak | TIER-1 | 2026-01-01 |
| SPORTS-012 | NCAAF Totals Specialist | Category drilling on high-edge segment | TIER-2 | 2026-01-01 |
| SPORTS-013 | Trade Count Milestone Fading | Numerological - 100th/500th/1000th trade | LSD-ABSURD | 2026-01-01 |
| SPORTS-014 | Bot Signature Fade | Fade clock-like trading patterns | TIER-3 | 2026-01-01 |
| SPORTS-015 | Fibonacci Price Attractors | Technical levels as support/resistance | LSD-ABSURD | 2026-01-01 |

---

## LSD Screening Queue

*Hypotheses currently being quick-tested in LSD mode*

| ID | Title | Quick Edge | Status | Notes |
|----|-------|------------|--------|-------|
| - | Queue empty | - | - | - |

---

## Promising Candidates

*Passed LSD threshold (>5% edge), awaiting full validation*

| ID | Title | Quick Edge | Markets | Date Flagged |
|----|-------|------------|---------|--------------|
| - | No promising candidates yet | - | - | - |

---

## Recently Validated

| ID | Title | Final Edge | Improvement | Date Validated |
|----|-------|------------|-------------|----------------|
| H102/S013 | Low Leverage Variance NO | +11.3% | +8.0% | 2025-12-29 |
| H121 | Whale Low Leverage Fade | +5.8% | +6.8% | 2025-12-29 |
| H122 | Whale + S013 Combined | +15.0% | +11.3% | 2025-12-29 |

---

## Recently Rejected

| ID | Title | Reason | Date |
|----|-------|--------|------|
| - | See RESEARCH_JOURNAL.md for full rejection history | - | - |

---

## Hypothesis Brief Format

When the Strategy Researcher agent adds a hypothesis, it should follow this format:

```json
{
  "hypothesis_id": "EXT-001",
  "source": "SSRN Paper: 'Favorite-Longshot Bias in Prediction Markets'",
  "source_url": "https://...",
  "title": "Conditional Favorite-Longshot by Event Type",
  "mechanism": "Description of WHY this might work",
  "proposed_signal": {
    "condition": "What triggers the signal",
    "action": "What to trade",
    "expected_edge": "Estimated edge if it works"
  },
  "data_requirements": ["fields needed from trade data"],
  "price_proxy_risk": "LOW/MEDIUM/HIGH",
  "novelty_vs_tested": "How is this different from what we've tested?",
  "priority": "HIGH/MEDIUM/LOW"
}
```

---

## Workflow

```
Strategy Researcher → Incoming → LSD Screening → Promising → Full Validation → Validated
                                      ↓                           ↓
                                   Rejected                    Rejected
```

---

*Last updated: 2026-01-01*
