# Reddit Entity Signal Learnings

## Trading Mechanics (2026-01-27)

### Order Execution Pattern
- **All Kalshi orders are limit orders** - no market orders exist
- When buying, use the ASK price to cross the spread and fill immediately
- YES contracts: use `yes_ask` as limit price
- NO contracts: use `100 - yes_bid` as limit price (NO ask = 100 - YES bid)
- Price must be 1-99 cents (never 0 or 100)

## Recent Strong Signals (2026-01-27)

### Trump OUT Market (KXG7LEADEROUT-45JAN01-DJT)
- **Multiple signals**: -75 to -90 sentiment → +75 to +90 price impact
- **Perfect confidence**: 1.0 across all signals
- **Market pricing**: Last trade 11¢, current bid 11¢
- **Signal logic**: OUT markets benefit from negative sentiment (scandal makes OUT more likely)
- **Edge opportunity**: Strong signals suggest upward pressure on YES price

### Iran Leader Market (KXNEXTIRANLEADER-45JAN01-ABO)
- **Signal**: "Position abolished" entity with -75 price impact, confidence 1.0
- **Market issue**: No pricing data available (0 bid/ask)
- **Status**: Likely illiquid or inactive market

## Signal Quality Assessment
- Confidence 1.0 signals are consistently appearing
- Price impact scores are strong (+/-75 to +/-90)
- Sentiment transformation logic appears to be working correctly for OUT markets
- Multiple corroborating signals on same entity increase confidence

## Next Actions
1. Execute trades on markets with valid bid/ask spreads
2. Monitor for additional high-confidence signals
3. Track fill rates and slippage on limit orders
