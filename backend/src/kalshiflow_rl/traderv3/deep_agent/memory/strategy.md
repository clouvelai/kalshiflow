# Trading Strategy

## Entry Rules
- Use get_extraction_signals() to find markets with aggregated extraction activity
- **Minimum signal bar to trade:**
  - At least 2 unique sources (1 source = noise, ignore)
  - Consensus strength > 0.5 (directional agreement)
  - Max engagement > 100 upvotes (proves the content resonated)
- **Strong signal indicators (higher conviction):**
  - 5+ unique sources with consensus > 0.7
  - Entity sentiment aligned with trade direction (negative entity = bearish)
  - GDELT confirms Reddit narrative (5+ articles, aligned tone)
- Confirm with preflight_check() before every trade
- Every trade is data: track what works, discard what doesn't
- Refine these rules after every session based on actual outcomes

## Position Sizing
- **$1,000 max exposure per event** (cost = contracts x price_in_cents / 100)
- Default trade size: **$25-100**. Only go larger with strong conviction.
- Scale by confidence:
  - Speculative ($25-50): 2-3 sources, consensus 0.5-0.7
  - Moderate ($50-100): 3-5 sources, consensus > 0.7, engagement > 200
  - High conviction ($100-250): 5+ sources, consensus > 0.8, GDELT confirms
  - Maximum ($250-500): Rare. Overwhelming multi-source evidence, obvious mispricing
- Always calculate dollar cost before placing a trade

## Risk Rules
- Check event context before correlated trades (preflight_check or get_event_context)
- Never exceed $1,000 total cost basis on any single event
- Avoid spreads > 8c (low liquidity = high execution risk)
- Watch for mutual exclusivity risk in multi-candidate events

## Exit Rules
- **Cut losses**: If new extraction signals contradict your position direction, consider selling
- **Take profits**: If position is up significantly and signals have decayed (no new sources), consider selling
- **Signal decay**: If the extraction signals that drove entry are now stale (>12h, no new sources), reassess
- Develop more specific exit rules through experience and record them here

## Execution Strategy Selection
- Strong conviction (5+ sources, high consensus) -> aggressive (don't miss the fill)
- Moderate conviction (2-4 sources) -> moderate (save spread cost)
- Speculative / weak conviction -> passive (only enter at a great price)
