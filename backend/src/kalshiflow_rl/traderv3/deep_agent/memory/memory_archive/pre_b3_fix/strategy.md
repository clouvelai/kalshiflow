# Trading Strategy

## Entry Rules
- Use get_extraction_signals() to find markets with aggregated extraction activity
- Evaluate signal strength: source count, engagement, directional consensus, magnitude
- Develop specific entry criteria through experience and record them here
- Every trade is data: track what works, discard what doesn't
- No market category is off-limits — evaluate every signal on its merits

## Position Sizing
- **$1,000 max exposure per event** (cost = contracts x price_in_cents / 100)
- Default trade size: **$25-100**. Only go larger with strong conviction.
- Scale by confidence:
  - Speculative ($25-50): 1-2 sources, weak consensus
  - Moderate ($50-100): 2-4 sources, decent consensus
  - High conviction ($100-250): 5+ sources, strong consensus, clear edge
  - Maximum ($250-500): Rare. Overwhelming evidence, obvious mispricing
- Always calculate dollar cost before placing a trade

## Risk Rules
- Check event context before correlated trades (get_event_context)
- Never exceed $1,000 total cost basis on any single event
- Watch for mutual exclusivity risk in multi-candidate events

## Exit Rules
- (Develop through experience)

