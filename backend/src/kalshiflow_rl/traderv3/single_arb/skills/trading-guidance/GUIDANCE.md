# Trading Guidance

## Sizing
- Edge >= 10c: up to 25 contracts (verify depth > 20 at price level)
- Edge 5-10c: 5-15 contracts
- Edge 2-5c: 1-5 contracts
- Max 20% capital per event. Max 50 contracts per market.

## Exits
- Review positions each cycle. Use pnl_ct to assess per-contract profitability.
- PnL < -12c/contract: cut the loss.
- TIME_PRESSURE flag means < 1h to settlement: exit unless high conviction from recent news.
- Partial arb fills create directional exposure: check and exit unhedged legs.
- Exit a YES position: place_order(side="yes", action="sell")
- Exit a NO position: place_order(side="no", action="sell")
- NEVER buy the opposite side to hedge.

## Execution
- Limit orders only. Use TTL.

## Sniper
- Configure edge threshold, capital limits, cooldown.
- Don't duplicate Sniper's arb execution — focus on position management and directional conviction.

## Regime Signals
- VPIN > 0.85: reduce size by half.
- VPIN > 0.95 with prior loss evidence in memory: stand aside.
- Sweep/whale activity: data to store, not a reason to freeze.

## Hard Stops
- Negative edge (arb is unprofitable)
- Spread > 15c (can't fill)
- Position limit reached
- Non-ME event for short arb (probability sum not guaranteed)
