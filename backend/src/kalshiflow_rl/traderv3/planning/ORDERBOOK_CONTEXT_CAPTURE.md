# Order Context Capture for Quant Analysis

## Decision Summary
**Date:** 2025-01-03
**Status:** Implemented & Tested

Capture comprehensive order context (signal, orderbook, position) at order placement time for post-hoc quant analysis of strategy performance, fill quality, and slippage.

## What We Capture (Per Filled Order)

| Category | Fields |
|----------|--------|
| **Identity** | `order_id`, `market_ticker`, `session_id` |
| **Signal** | `strategy`, `signal_id`, `signal_detected_at`, `signal_params` (JSONB) |
| **Market** | `market_category`, `market_close_ts`, `hours_to_settlement`, `trades_in_market` |
| **Price** | `no_price_at_signal`, `bucket_5c` (45,50,...,95) |
| **Orderbook** | `best_bid_cents`, `best_ask_cents`, `bid_ask_spread_cents`, `spread_tier`, `bid_size_contracts`, `ask_size_contracts` |
| **Position** | `existing_position_count`, `existing_position_side`, `is_reentry`, `entry_number`, `balance_cents` |
| **Order** | `action`, `side`, `order_price_cents`, `order_quantity`, `order_type` |
| **Timing** | `placed_at`, `hour_of_day_utc`, `day_of_week`, `calendar_week` |
| **Fill** | `fill_count`, `fill_avg_price_cents`, `filled_at`, `time_to_fill_ms`, `slippage_cents` |
| **Settlement** | `market_result`, `settled_at`, `realized_pnl_cents` |

### Orderbook Fields (Detail)

| Field | Type | Purpose |
|-------|------|---------|
| `bid_ask_spread_cents` | INT | Primary variable for threshold tuning |
| `spread_tier` | VARCHAR(10) | Quick filter: "tight"/"normal"/"wide" |
| `best_bid_cents` | INT | Reference for slippage calculation |
| `best_ask_cents` | INT | Reference for slippage calculation |
| `bid_size_contracts` | INT | Liquidity indicator (thin book detection) |
| `ask_size_contracts` | INT | Liquidity indicator (thin book detection) |

## Why Keep All Fields (Quant Rationale)

### Spread Data (Critical)
- **Threshold Optimization**: RLM uses spread tiers (tight ≤2c, normal ≤4c, wide >4c) for pricing. Empirical data validates/tunes these thresholds.
- **Slippage Attribution**: Separate signal quality from execution quality. Did we lose edge to bad signals or bad fills?
- **Fill Rate Analysis**: Passive vs aggressive pricing by spread tier.

### Size Data (Kept for Future Use)
- **Thin Book Detection**: Skip signals when liquidity is low (bid_size < X).
- **Impact Estimation**: Large orders relative to book size may cause slippage.
- Storage cost is trivial (~4 bytes per field, 1000s orders/month).

## Alternative Considered: Slim Capture

Could drop `best_bid/ask_cents` and `bid/ask_size_contracts`, keeping only:
- `bid_ask_spread_cents`
- `spread_tier`

**Rejected because:**
1. Bid/ask needed for accurate slippage calculation
2. Size data enables future "thin book" filtering hypothesis
3. Storage cost is negligible

## Data Sources

| Field | Source | When Captured |
|-------|--------|---------------|
| `bid_ask_spread_cents` | `get_shared_orderbook_state(ticker).get_snapshot()` | At order placement |
| `spread_tier` | Computed from spread using RLM thresholds | At order placement |
| `best_bid_cents` | Orderbook snapshot `no_bids` | At order placement |
| `best_ask_cents` | Orderbook snapshot `no_asks` | At order placement |
| `bid_size_contracts` | Sum of `no_bids` at best price | At order placement |
| `ask_size_contracts` | Sum of `no_asks` at best price | At order placement |

## Why Capture at Order Placement (Not Settlement)

Some data is **ephemeral** and only exists at the moment of the trading decision:

| Data Type | Available at Decision | Available at Settlement | Reconstructable? |
|-----------|----------------------|------------------------|------------------|
| Signal params (yes_ratio, price_drop) | ✅ | ❌ | NO - ephemeral trade stream state |
| Orderbook snapshot | ✅ | ❌ | NO - changes every second |
| Position before trade | ✅ | ⚠️ Already changed | Partial |
| Balance before trade | ✅ | ⚠️ Already changed | Partial |
| Fill details | ❌ | ✅ | YES |
| Market result | ❌ | ✅ | YES |

The signal context (`signal_params`) is the most valuable data - without it, we can't validate whether the strategy is working as designed.

## Implementation Location

- **Data Model**: `traderv3/state/order_context.py` - `OrderbookSnapshot` dataclass
- **Capture Logic**: `traderv3/services/trading_decision_service.py` - `_stage_order_context()`
- **Persistence**: `traderv3/services/order_context_service.py` - `persist_on_fill()`

## Example Analysis Queries

### Spread Distribution at Signal Time
```sql
SELECT spread_tier, COUNT(*) as trades,
       AVG(bid_ask_spread_cents) as avg_spread
FROM order_contexts
WHERE strategy = 'rlm_no' AND settled_at IS NOT NULL
GROUP BY spread_tier;
```

### Slippage by Spread Tier
```sql
SELECT spread_tier,
       AVG(slippage_cents) as avg_slippage,
       AVG(CASE WHEN market_result = 'no' THEN 1.0 ELSE 0.0 END) as win_rate
FROM order_contexts
WHERE strategy = 'rlm_no' AND settled_at IS NOT NULL
GROUP BY spread_tier;
```

### Thin Book Detection Candidates
```sql
SELECT market_ticker, bid_size_contracts, ask_size_contracts,
       market_result, realized_pnl_cents
FROM order_contexts
WHERE (bid_size_contracts < 50 OR ask_size_contracts < 50)
  AND settled_at IS NOT NULL
ORDER BY filled_at DESC;
```

### Edge by Price Bucket (Bucket-Matched Validation)
```sql
SELECT bucket_5c,
       COUNT(*) as n,
       AVG(CASE WHEN market_result = 'no' THEN 1.0 ELSE 0.0 END) as win_rate,
       bucket_5c / 100.0 as implied_win_rate,
       AVG(CASE WHEN market_result = 'no' THEN 1.0 ELSE 0.0 END) - (bucket_5c / 100.0) as edge
FROM order_contexts
WHERE settled_at IS NOT NULL
  AND strategy = 'rlm_no'
  AND side = 'no'
GROUP BY bucket_5c
HAVING COUNT(*) >= 5
ORDER BY bucket_5c;
```

## Architecture & Lifecycle

### Data Flow
```
1. STAGE   → TradingDecisionService._stage_order_context() captures context in memory
2. PERSIST → StateContainer._sync_trading_attachments() detects fill → persist_on_fill()
3. LINK    → StateContainer._capture_settlement_for_attachment() → link_settlement()
4. DISCARD → Cancelled orders: discard_staged_context() clears memory
```

### Files
| File | Purpose |
|------|---------|
| `state/order_context.py` | Data models (`StagedOrderContext`, `OrderbookSnapshot`) |
| `services/order_context_service.py` | Service (stage, persist, link, export) |
| `services/trading_decision_service.py` | Capture logic (`_stage_order_context()`) |
| `core/state_container.py` | Fill detection & settlement linking |
| `supabase/migrations/20260103_order_contexts.sql` | Database schema |

### Memory Management
- Staged contexts held in memory until fill/cancel
- `discard_staged_context()` called on:
  - TTL expiry (TradingFlowOrchestrator)
  - Order cancelled/rejected (StateContainer)
- Fire-and-forget tasks use `_log_task_exception()` callback for error visibility

## Export Endpoint

```bash
# CSV export (default)
curl "http://localhost:8005/v3/export/order-contexts?strategy=rlm_no&settled_only=true"

# JSON export
curl "http://localhost:8005/v3/export/order-contexts?format=json&settled_only=false"

# Filter by date
curl "http://localhost:8005/v3/export/order-contexts?from_date=2025-01-01&to_date=2025-01-31"
```

## Related Documentation
- `RLM_SPREAD_THRESHOLD_ANALYSIS.md` - Theoretical spread analysis
- `VALIDATED_STRATEGIES.md` - Strategy validation methodology
- `RLM_IMPROVEMENTS.md` - RLM strategy enhancements
