# Prompt Engineering Integration Handoff

**Author**: prompt-engineer agent
**Date**: 2026-01-09
**Branch**: `sam/orderbook-signal-improvements`

This document summarizes the prompt infrastructure changes and what the fullstack-websocket-engineer and kalshi-flow-trader-specialist need to verify for correct integration.

---

## Summary of Changes

### 1. New Prompts Infrastructure

Created a versioned prompt management system at:
```
/backend/src/kalshiflow_rl/traderv3/prompts/
    __init__.py          # Module exports
    current.py           # Active version configuration
    versions/
        event_context_v1.py   # Phase 2+3 baseline
        event_context_v2.py   # Phase 2+3 improved (grounding + calibration)
        market_eval_v1.py     # Phase 5 baseline
        market_eval_v2.py     # Phase 5 improved (calibration self-check)
    schemas/
        full_context_schema.py      # FullContextOutput Pydantic model
        market_assessment_schema.py # SingleMarketAssessment Pydantic model
```

### 2. New Schema Fields (v2 - Backward Compatible)

All new fields have defaults, so existing data will not break.

#### SingleMarketAssessment (Phase 5 output)
```python
# NEW fields in SingleMarketAssessment:
evidence_cited: List[str] = []           # 1-3 evidence points supporting estimate
what_would_change_mind: str = ""         # Single measurement that would shift estimate
assumption_flags: List[str] = []         # Assumptions made due to missing info
calibration_notes: str = ""              # Notes on confidence calibration
evidence_quality: str = "medium"         # high/medium/low
```

#### BatchMarketAssessmentOutput (wrapper)
```python
# NEW field:
cross_market_consistency_check: str = "" # For mutually exclusive markets
```

#### FullContextOutput (Phase 2+3 output)
```python
# NEW grounding fields:
grounding_notes: str = ""                # What is from context vs assumed
base_rate_source: str = "estimated"      # Source of base rate

# NEW calibration fields:
uncertainty_factors: List[str] = []      # Key uncertainty factors
information_gaps: List[str] = []         # What info would improve analysis
```

### 3. Modified EventResearchService

Location: `/backend/src/kalshiflow_rl/traderv3/services/event_research_service.py`

Key additions:
- **Source quality heuristics** (lines 81-161): `assess_source_quality()` function categorizes URLs as high/medium/low quality
- **Trusted domain lists**: `TRUSTED_DOMAINS` (official/financial) and `MEDIUM_TRUST_DOMAINS`
- **Semantic frame queries** (lines 163-229): `generate_semantic_frame_queries()` generates targeted search queries based on frame type
- **Evidence quality tracking** (lines 1424-1454): Source quality breakdown in reliability reasoning

### 4. Modified MarketAssessment Dataclass

Location: `/backend/src/kalshiflow_rl/traderv3/state/event_research_context.py`

The `MarketAssessment` dataclass (lines 474-543) now includes the v2 calibration fields with defaults.

### 5. WebSocket Broadcast Format

Location: `/backend/src/kalshiflow_rl/traderv3/core/websocket_manager.py`

The `broadcast_event_research()` method (lines 912-1009) now includes the new fields in the WebSocket payload with `getattr()` fallbacks for backward compatibility.

---

## For fullstack-websocket-engineer

### WebSocket Message Format Changes

The `event_research` message type now includes additional fields per market assessment:

```javascript
// Frontend receives via "event_research" message type
{
  "event_ticker": "KXFEDCHAIRNOM",
  "markets": [
    {
      "ticker": "KXFEDCHAIRNOM-WARSH",
      "evidence_probability": 0.35,
      "market_probability": 0.28,
      "mispricing_magnitude": 0.07,
      "recommendation": "BUY_YES",
      "confidence": "medium",
      "edge_explanation": "...",

      // === NEW v2 FIELDS (may be empty/default on old data) ===
      "evidence_cited": ["AP News reports shortlist...", "Fed insider sources..."],
      "what_would_change_mind": "Official White House announcement of nominee",
      "assumption_flags": ["Assumed no major news since last search"],
      "calibration_notes": "Medium confidence due to single-source reporting",
      "evidence_quality": "medium",  // high, medium, or low

      // Additional context fields (already existed)
      "specific_question": "Will Kevin Warsh be the next Fed Chair?",
      "driver_application": "Trump's preference is the key driver..."
    }
  ],
  // ... rest of event context
}
```

### Verification Checklist

1. **Frontend resilience**: Verify frontend handles missing/empty new fields gracefully
   - `evidence_cited` may be empty array `[]`
   - `what_would_change_mind` may be empty string `""`
   - `assumption_flags` may be empty array `[]`
   - `evidence_quality` defaults to `"medium"` if missing

2. **Events tab display**: If Events tab shows market assessments, consider displaying:
   - Evidence quality badge (high/medium/low)
   - Evidence citations (collapsible list)
   - "What would change mind" as tooltip or expandable section
   - Assumption flags as warning indicators

3. **No breaking changes**: The WebSocket format is additive only. Existing frontend code should continue working.

4. **Type hints**: Frontend TypeScript types (if any) may need updating for the new optional fields.

### Files to Check

- `/frontend/src/components/` - Any component consuming `event_research` messages
- Check for TypeScript interfaces that type the event_research payload

---

## For kalshi-flow-trader-specialist

### Trading Decision Flow Changes

The agentic research pipeline now produces richer assessment data that can inform trading decisions.

### Key Integration Points

1. **MarketAssessment dataclass** has new fields that flow through to trade decisions:
   ```python
   # In MarketAssessment (event_research_context.py)
   evidence_cited: List[str]        # Can log what evidence drove the decision
   what_would_change_mind: str      # Can add to decision trace
   assumption_flags: List[str]      # Reduce confidence if assumptions made
   evidence_quality: str            # Factor into decision thresholds
   ```

2. **Confidence modulation**: The new fields enable smarter confidence handling:
   ```python
   # Example: Reduce effective confidence if assumptions were made
   if assessment.assumption_flags:
       # Could downgrade HIGH -> MEDIUM, MEDIUM -> LOW
       pass

   # Example: Factor evidence quality into edge threshold
   if assessment.evidence_quality == "low":
       # Could require larger edge to trade
       pass
   ```

3. **Decision logging**: The new fields should flow to `research_decisions` table for calibration analysis:
   - `evidence_cited` helps audit what drove the decision
   - `assumption_flags` helps identify decisions made under uncertainty
   - `evidence_quality` enables filtering/grouping in analysis

### Verification Checklist

1. **No regression**: Verify existing trade flow still works with v2 prompts active
   - Set `PROMPT_MARKET_EVAL_VERSION=v1` to test v1 fallback
   - Set `PROMPT_MARKET_EVAL_VERSION=v2` to test v2 (default)

2. **Field propagation**: Verify new fields flow to persistence layer
   - Check `research_decisions` table schema includes new columns (or stores as JSON)
   - Verify fields appear in trade logs/traces

3. **Edge calculation**: No changes to edge calculation logic (evidence_probability, market_probability, mispricing_magnitude unchanged)

4. **Skip reason tracking**: Consider adding skip reasons for:
   - `low_evidence_quality`: Skipped due to evidence_quality = "low"
   - `high_assumptions`: Skipped due to many assumption_flags

### Files to Check

- `/backend/src/kalshiflow_rl/traderv3/strategies/plugins/agentic_research.py` - Main strategy plugin
- `/backend/src/kalshiflow_rl/traderv3/services/trading_decision_service.py` - Decision execution

---

## A/B Testing Configuration

The prompt versions can be switched via environment variables:

```bash
# Default (v2 prompts)
export PROMPT_EVENT_CONTEXT_VERSION=v2
export PROMPT_MARKET_EVAL_VERSION=v2

# Rollback to v1
export PROMPT_EVENT_CONTEXT_VERSION=v1
export PROMPT_MARKET_EVAL_VERSION=v1
```

No code changes required for version switching.

---

## Metrics to Watch

After deployment, monitor these metrics:

1. **Parse fail rate**: Should not increase (v2 schemas have defaults)
2. **Calibration error**: Compare v2 vs v1 probability estimates vs outcomes
3. **Evidence quality distribution**: What fraction of assessments have high/medium/low quality
4. **Assumption frequency**: How often are assumption_flags populated
5. **Trade skip reasons**: Are new skip reasons (if added) firing appropriately

---

## Known Limitations

1. **Backward compatibility only**: Old cached research results will have empty v2 fields
2. **No frontend changes required**: But UI could be enhanced to display new fields
3. **No threshold changes**: Evidence quality does not yet modulate trading thresholds (future work)

---

## Files Modified (Quick Reference)

| File | Change Type | Description |
|------|-------------|-------------|
| `prompts/__init__.py` | NEW | Module exports |
| `prompts/current.py` | NEW | Version configuration |
| `prompts/versions/event_context_v1.py` | NEW | Phase 2+3 baseline prompt |
| `prompts/versions/event_context_v2.py` | NEW | Phase 2+3 improved prompt |
| `prompts/versions/market_eval_v1.py` | NEW | Phase 5 baseline prompt |
| `prompts/versions/market_eval_v2.py` | NEW | Phase 5 improved prompt |
| `prompts/schemas/full_context_schema.py` | NEW | FullContextOutput schema |
| `prompts/schemas/market_assessment_schema.py` | NEW | SingleMarketAssessment schema |
| `services/event_research_service.py` | MODIFIED | Source quality, semantic queries |
| `state/event_research_context.py` | MODIFIED | MarketAssessment v2 fields |
| `core/websocket_manager.py` | MODIFIED | broadcast_event_research v2 fields |

---

## Contact

For prompt-specific questions, tag the prompt-engineer agent.
For integration issues, coordinate between fullstack-websocket-engineer and kalshi-flow-trader-specialist.
