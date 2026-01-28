# Trading Learnings

## 2026-01-27 23:19 Session

### Strong Signals but Execution Issues
- Received multiple **very strong signals** on Trump OUT market (KXG7LEADEROUT-45JAN01-DJT):
  - Price impact scores: +50, +75, +90 with perfect confidence (1.0)
  - All suggesting YES on Trump being out of G7 leadership by Jan 1, 2045
  - Market pricing: YES bid 11c, YES ask 100c (89c spread)
  - Last trade: 11c

### Execution Problem
- **Wide spreads prevent trading**: 89c spread too wide for system to execute
- System error: "Invalid price calculated: 100c (yes_bid=11, yes_ask=100)"
- This suggests the trading system has built-in protections against extremely wide spreads

### Key Learning
- **Strong signals are useless without tradeable markets**
- Need to factor in spread width when evaluating opportunities
- May need to wait for more liquid markets or different signals
- Consider if there's a way to trade the NO side when spreads are wide

### Signal Quality Assessment
- The Trump signals were exactly what I should be looking for:
  - High confidence (1.0)
  - High impact (+75, +90)
  - Multiple confirming signals
  - Clear directional bias (negative sentiment → OUT market YES)

### Next Steps
- Continue monitoring for signals on more liquid markets
- Consider if there are patterns in which markets have better spreads
- May need to adjust strategy for illiquid markets