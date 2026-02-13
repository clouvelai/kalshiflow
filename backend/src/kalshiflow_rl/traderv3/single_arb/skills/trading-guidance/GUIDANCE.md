# Trading Guidance

## Early Bird Details

### Why Early Bird Works
- 0% maker fees on resting limits
- First orders anchor the price (no opening auction on Kalshi)
- Wide spreads at open (10-20c) narrow as market makers enter
- System detects activation 10-50x faster than retail UI

### Complement Strategy Execution
fair_value is deterministic: 100 - sum(other market YES prices in ME event).
1. Place YES limit at fair_value - 2c
2. Place NO limit at (100 - fair_value) - 2c
3. Both are maker orders = 0% fees
4. If one side fills, cancel_order the other immediately
5. If both fill, you captured the spread (net profit = 4c - directional risk)
6. If spread < 5c when you check: opportunity has passed, skip

### Captain-Decide Execution
1. search_news(event title + key entity)
2. get_market_state(event_ticker) for current book
3. Estimate fair YES price from news + context
4. Place resting limit at your estimate
5. store_insight with reasoning for future calibration

### Early Bird Exits
- +5c profit in 30min: sell half (partial profit)
- -8c against: cut the full position
- Spread narrows to < 3c: market matured, tighten stops to -3c

## Position Exits
- Exit YES: place_order(side="yes", action="sell")
- Exit NO: place_order(side="no", action="sell")
- NEVER buy opposite side. This locks in losses.
- Auto-actions handle: stop_loss (-12c/ct), time_exit (<30min to close)
- Override auto-actions via configure_automation ONLY with recent news that justifies holding

## Hard Stops (non-negotiable)
- Negative edge on ME arb
- Spread > 15c (unfillable)
- Position limit reached
- Short arb on non-ME event (probability sums not constrained to 100)

## Key Market Rules
- Non-ME events: YES sums CAN exceed 100c. NORMAL, not arb.
- ME arb edge across many legs may net negative after fees. Let Sniper compute this.
- YES@95c = 95% implied probability. The 5% downside costs 95c/contract.
- YES_bid + NO_bid > 100c is market maker spread, not free money.

## Search Query Quality
Good queries (specific entity + event + timeframe):
  "Tesla Q4 2025 earnings delivery numbers"
  "Fed FOMC March 2026 rate decision expectations"
  "Super Bowl 2026 winner odds Chiefs"
Bad queries (vague, no entity):
  "market news today"
  "what's happening in politics"
