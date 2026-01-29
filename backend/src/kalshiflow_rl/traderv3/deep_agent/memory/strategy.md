# Trading Strategy - UPDATED

## Entry Rules
- Trade when |impact| >= 40, confidence >= 0.7, gap > 5c
- **Market Selection Criteria**
  - Avoid high-volume institutional markets (>500k volume)
  - Avoid entertainment/celebrity markets 
  - Avoid niche terminology markets
  - Focus on breaking news events with retail participation

## Position Sizing
- Strong signals (+75): 25 contracts
- Moderate signals (+40): 15 contracts

## Risk Rules
- Max 5 positions
- Check event context before correlated trades
- **Signal Persistence Rule (CRITICAL - UPDATED)**
  - **SIGNALS DIE WITHIN 2-3 CYCLES** - not 5+ cycles as previously thought
  - If same signal_id persists >3 cycles with no price movement → DEAD SIGNAL
  - Focus on NEW signal_ids and fresh timestamps (<2 hours old)
  - **13-cycle persistence study proves**: Signal age > Confidence scores
  - Perfect confidence (1.0) means NOTHING if signal is stale

## Signal Quality Hierarchy (UPDATED)
1. **NEW signal_id** with fresh timestamp (<1 hour) = PRIORITY
2. **Recurring signal_id** within 2-3 cycles = CAUTION
3. **Persistent signal_id** >3 cycles = DEAD (ignore completely)

## Arbitrage Opportunities
- When mutually exclusive event has YES prices sum >105c → NO arbitrage
- Focus on events with 5+ markets for maximum arbitrage potential
- Guaranteed profit from N-1 positions (where N = total markets in event)

## Target Events
- Breaking news with retail participation
- Political events with multiple outcomes
- Avoid entertainment/celebrity speculation

## Anti-Patterns (PROVEN)
- Trading on signals >3 cycles old (waste of capital and mental energy)
- Believing high confidence scores override signal staleness
- Analyzing the same dead signals repeatedly instead of waiting for fresh data

## Key Learning: Signal Freshness Is Everything
**13-cycle persistence study definitively proved:**
- Markets either process Reddit sentiment within 1-2 cycles or reject it entirely
- Signal persistence beyond 3 cycles = 100% failure rate
- Professional traders have superior information sources that override stale Reddit sentiment
- Discipline to ignore stale signals (even with perfect confidence) is critical for success