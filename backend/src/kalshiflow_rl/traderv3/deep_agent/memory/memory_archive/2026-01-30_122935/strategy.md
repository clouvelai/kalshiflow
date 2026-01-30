# Trading Strategy

## Entry Rules
- Use assess_trade_opportunity() as your quantitative backbone — it returns STRONG/MODERATE/WEAK/AVOID
- Develop specific entry criteria through experience and record them here
- Every trade is data: track what works, discard what doesn't
- No market category is off-limits — evaluate every signal on its merits

## Position Sizing
- **$100 max exposure per event** (cost = contracts x price_in_cents / 100)
- Scale contracts based on conviction: stronger quality rating + higher confidence = larger position
- Always calculate dollar cost before placing a trade

## Risk Rules
- Check event context before correlated trades (get_event_context)
- Never exceed $100 total cost basis on any single event
- Watch for mutual exclusivity risk in multi-candidate events

## Exit Rules
- (Develop through experience)

