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

## Signal Persistence Rule (LEARNED FROM DATA)
- **SIGNALS DIE WITHIN 2-3 CYCLES** — not 5+ cycles as previously thought
- If same signal_id persists >3 cycles with no price movement -> DEAD SIGNAL
- Focus on NEW signal_ids and fresh timestamps (<2 hours old)
- 13-cycle persistence study proves: Signal age > Confidence scores
- Perfect confidence (1.0) means NOTHING if signal is stale

## Signal Quality Hierarchy (LEARNED FROM DATA)
1. **NEW signal_id** with fresh timestamp (<1 hour) = PRIORITY
2. **Recurring signal_id** within 2-3 cycles = CAUTION
3. **Persistent signal_id** >3 cycles = DEAD (ignore completely)

## Execution Speed (LEARNED FROM DATA)
- When criteria are met, execute within 1-2 function calls: assess -> think -> trade
- Markets process Reddit sentiment within 1-2 cycles or reject it entirely
- Clear edges require immediate execution, not extended verification

## Arbitrage Opportunities
- When mutually exclusive event has YES prices sum >105c -> NO arbitrage
- Focus on events with 5+ markets for maximum arbitrage potential

## Anti-Patterns (LEARNED FROM DATA)
- Trading on signals >3 cycles old (waste of capital)
- Believing high confidence scores override signal staleness
- Analyzing the same dead signals repeatedly instead of waiting for fresh data
- Analysis paralysis on clear edges — when criteria are met, execute immediately

## Exit Rules
- (Develop through experience)
