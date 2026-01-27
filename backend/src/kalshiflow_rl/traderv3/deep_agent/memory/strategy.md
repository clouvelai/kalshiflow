# Reddit Entity Trading Strategy

## Signal Filters
- Only trade signals with confidence > 0.7
- Only trade when |price_impact_score| > 30
- Require spread < 5c for entry

## Understanding Price Impact
- **OUT markets**: Negative sentiment → POSITIVE price impact (scandal makes OUT likely)
- **WIN/CONFIRM**: Sentiment preserved as price impact
- TRUST the transformation, don't second-guess

## Position Sizing
- Max 25 contracts per trade
- Max 3 positions per entity
- Max $100 total exposure

## Entry Rules
- price_impact_score > 50: Strong YES signal
- price_impact_score < -50: Strong NO signal
- Match signal direction with trade side (use suggested_side)

## Exit Rules
- Take profit at 20% gain
- Cut losses at 15% decline
- Exit if new contradicting signal appears for same entity

## Signal Quality Hierarchy
1. High confidence (>0.8) + High impact (>70) = STRONG
2. Medium confidence (0.6-0.8) + High impact = MODERATE
3. High confidence + Medium impact (30-70) = MODERATE
4. Low confidence or low impact = SKIP

---

## Strategy Updates

_The agent will update this section as it learns from entity signal outcomes._

