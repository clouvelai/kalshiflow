# Mistakes to Avoid

## Entity Signal Mistakes

1. **STALE SIGNALS**: Don't trade signals > 2 hours old - they're priced in
2. **LOW CONFIDENCE**: Signals with confidence < 0.5 are unreliable
3. **SMALL IMPACT**: |price_impact_score| < 20 = noise, not signal
4. **THIN MARKETS**: Spread > 10c indicates illiquidity - skip
5. **OVERTRADING**: Max 5 trades per hour

## Transformation Mistakes

6. **IGNORING MARKET TYPE**: OUT markets INVERT sentiment - don't forget!
7. **DOUBLE-INVERTING**: The price_impact_score is already transformed - use it directly
8. **SUGGESTED SIDE**: Trust the suggested_side field, it accounts for transformation

## Execution Mistakes

9. **CHASING**: If price moved 10%+ since signal, the edge is gone
10. **NO EXIT PLAN**: Always have exit criteria before entering
11. **POSITION STACKING**: Don't keep adding to losing positions

## Signal Source Mistakes

12. **ENTITY MISMATCH**: Ensure entity in signal matches market subject
13. **SUBREDDIT BIAS**: r/politics may have political bias - discount extreme sentiment
14. **AUTHOR CHECK**: Anonymous authors deserve lower confidence

---

## Mistakes Log

_Record specific mistakes here for learning._

