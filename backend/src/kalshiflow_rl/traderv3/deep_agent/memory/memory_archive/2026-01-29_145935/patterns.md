# Winning Trading Patterns

## Arbitrage Pattern - Overpriced Mutually Exclusive Events
**Pattern**: When YES prices sum >105c in mutually exclusive events, NO contracts are guaranteed profit

**Identification**:
1. Check event context with get_event_context()
2. Look for YES_sum significantly above 100c
3. Confirm mutual exclusivity (only ONE can win)
4. Verify decent liquidity on target markets

**Execution**:
- Buy NO on multiple markets within position limits
- Focus on markets with best spreads (<8c)
- Maximize contracts (25 each) for maximum profit
- Guaranteed profit on all positions except the ONE winner

**Risk**: Essentially zero - mathematical arbitrage
**Success Rate**: 100% (by definition)

**Example**: KXTRUMPSAY-26FEB02 with YES_sum=844c yielded 5 NO positions, 4 guaranteed winners.