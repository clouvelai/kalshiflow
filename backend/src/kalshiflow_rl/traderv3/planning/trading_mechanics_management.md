# Trading Mechanics Management - V3 Trader

**Document Status**: Ready for Review
**Prepared For**: Flore (Systems Architect Review)
**Created**: 2025-12-27
**Author**: Claude (V3 Trader Specialist)

---

## Executive Summary

This document specifies the trading mechanics for the V3 trader's "Follow the Whale" execution layer. The design relies entirely on syncing state from Kalshi's API - **no local database layer is required**. Kalshi is the single source of truth for orders, positions, and fills.

**Core Principle**: Nothing is faked, nothing is janky. Orders are real Kalshi API calls. State comes directly from Kalshi. If it doesn't work, we fail clearly.

---

## Part 1: Kalshi API Reference

### 1.1 Order Lifecycle

```
                                    ┌──────────┐
                                    │  CREATE  │
                                    │  ORDER   │
                                    └────┬─────┘
                                         │
                                         v
                                    ┌──────────┐
                    ┌───────────────│ RESTING  │───────────────┐
                    │               │ (Active) │               │
                    │               └────┬─────┘               │
                    │                    │                     │
                    v                    v                     v
              ┌──────────┐        ┌──────────┐          ┌──────────┐
              │ CANCELED │        │ PARTIAL  │          │ EXECUTED │
              │          │        │  FILL    │          │  (Full)  │
              └──────────┘        └────┬─────┘          └──────────┘
                                       │
                                       v
                                 ┌──────────┐
                                 │ EXECUTED │
                                 │  (Full)  │
                                 └──────────┘
```

**Order Statuses**:
- `resting`: Active, awaiting matches in the orderbook
- `executed`: Fully filled, all contracts matched
- `canceled`: Cancelled by user or system

### 1.2 Create Order

**Endpoint**: `POST /portfolio/orders`

**Request Body**:
```json
{
  "ticker": "INXD-25JAN03",
  "action": "buy",
  "side": "yes",
  "count": 5,
  "type": "limit",
  "yes_price": 65,
  "order_group_id": "uuid-optional"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `ticker` | string | Yes | Market identifier |
| `action` | enum | Yes | `"buy"` or `"sell"` |
| `side` | enum | Yes | `"yes"` or `"no"` |
| `count` | int | Yes | Number of contracts (min: 1) |
| `type` | enum | No | `"limit"` (default) or `"market"` |
| `yes_price` | int | * | Price in cents (1-99) for YES contracts |
| `no_price` | int | * | Price in cents (1-99) for NO contracts |
| `order_group_id` | string | No | UUID for portfolio limits |
| `time_in_force` | enum | No | `"fill_or_kill"`, `"good_till_canceled"`, `"immediate_or_cancel"` |

*Either `yes_price` or `no_price` required for limit orders*

**Response (201 Created)**:
```json
{
  "order": {
    "order_id": "abc123-uuid",
    "ticker": "INXD-25JAN03",
    "side": "yes",
    "action": "buy",
    "type": "limit",
    "status": "resting",
    "yes_price": 65,
    "initial_count": 5,
    "fill_count": 0,
    "remaining_count": 5,
    "created_time": "2025-12-27T10:30:00Z",
    "last_update_time": "2025-12-27T10:30:00Z",
    "taker_fees": 0,
    "maker_fees": 0
  }
}
```

### 1.3 Get Orders

**Endpoint**: `GET /portfolio/orders`

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter: `"resting"`, `"canceled"`, `"executed"` |
| `ticker` | string | Filter by market ticker |
| `limit` | int | Results per page (max: 200) |
| `cursor` | string | Pagination cursor |

**Response**:
```json
{
  "orders": [
    {
      "order_id": "abc123",
      "ticker": "INXD-25JAN03",
      "status": "resting",
      "initial_count": 5,
      "fill_count": 2,
      "remaining_count": 3,
      ...
    }
  ],
  "cursor": "next-page-token"
}
```

### 1.4 Get Positions

**Endpoint**: `GET /portfolio/positions`

**Response**:
```json
{
  "market_positions": [
    {
      "ticker": "INXD-25JAN03",
      "position": 10,
      "total_traded": 6500,
      "market_exposure": 6500,
      "realized_pnl": 0,
      "fees_paid": 35,
      "last_updated_ts": "2025-12-27T10:35:00Z"
    }
  ]
}
```

**Position Field**:
- Positive value = YES contracts owned
- Negative value = NO contracts owned

### 1.5 Get Fills

**Endpoint**: `GET /portfolio/fills`

**Response**:
```json
{
  "fills": [
    {
      "fill_id": "fill-uuid",
      "order_id": "abc123",
      "ticker": "INXD-25JAN03",
      "side": "yes",
      "action": "buy",
      "count": 2,
      "yes_price": 65,
      "is_taker": true,
      "created_time": "2025-12-27T10:32:00Z"
    }
  ]
}
```

### 1.6 Order Groups (Portfolio Limits)

Order groups provide portfolio-level limits for risk management.

**Create**: `POST /portfolio/order_groups/create`
```json
{
  "contracts_limit": 10000
}
```

**Response**:
```json
{
  "order_group_id": "group-uuid",
  "max_absolute_position": 10000,
  "current_absolute_position": 0,
  "status": "active"
}
```

**Get Status**: `GET /portfolio/order_groups/{id}`

**Reset**: `POST /portfolio/order_groups/{id}/reset`

---

## Part 2: State Management Architecture

### 2.1 No Database Required

**Key Insight**: Kalshi is the database. We sync, don't store.

```
┌─────────────────────────────────────────────────────────┐
│                    KALSHI API                           │
│  (Single Source of Truth for orders/positions/fills)   │
└────────────────────────────┬────────────────────────────┘
                             │
                             │ REST API calls
                             │ (sync_with_kalshi)
                             v
┌─────────────────────────────────────────────────────────┐
│                  KalshiDataSync                         │
│                                                         │
│  - Fetches balance, positions, orders, settlements     │
│  - Builds TraderState from raw API data                │
│  - Tracks changes between syncs                        │
└────────────────────────────┬────────────────────────────┘
                             │
                             v
┌─────────────────────────────────────────────────────────┐
│                   TraderState                           │
│              (In-Memory Snapshot)                       │
│                                                         │
│  balance: int (cents)                                  │
│  portfolio_value: int (cents)                          │
│  positions: Dict[ticker, position_data]                │
│  orders: Dict[order_id, order_data]                    │
│  order_group: Optional[OrderGroupState]                │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Sync Cycle

Every trading cycle:
1. **Sync from Kalshi** → Get current truth
2. **Make trading decisions** → Based on current state
3. **Execute orders** → Place via API
4. **Sync again** → Verify execution

```python
# Sync pattern (already implemented in coordinator.py)
state, changes = await self._trading_client.sync_with_kalshi()

# State now contains:
# - state.balance (cents)
# - state.positions (dict by ticker)
# - state.orders (dict by order_id)
# - state.order_group (if active)
```

### 2.3 Order Tracking (In-Memory Only)

For whale following, we need to track which whales we've followed. This is **ephemeral** - we don't need persistence:

```python
@dataclass
class FollowedWhale:
    """Tracks a whale bet we've attempted to follow."""
    whale_id: str           # market:timestamp:price:side
    our_order_id: str       # Kalshi order_id from response
    placed_at: float        # When we placed our order
    market_ticker: str
    side: str               # yes/no
    our_count: int          # How many contracts we ordered
    whale_size_cents: int   # Original whale size

# In-memory tracking (cleared on restart, which is fine)
_followed_whales: Dict[str, FollowedWhale] = {}
```

**Why no persistence?**
- Kalshi has the actual order state
- On restart, we sync and see our current positions
- We don't re-follow whales because we already have positions
- Simplicity > complexity for MVP

---

## Part 3: Execution Flow

### 3.1 Order Placement Flow

```
                 ┌────────────────────┐
                 │  Whale Detected    │
                 │  (BigBet from      │
                 │   WhaleTracker)    │
                 └─────────┬──────────┘
                           │
                           v
                 ┌────────────────────┐
                 │  Check: Already    │──── Yes ────> Skip
                 │  Followed?         │
                 └─────────┬──────────┘
                           │ No
                           v
                 ┌────────────────────┐
                 │  Check: Have       │──── Yes ────> Skip
                 │  Position in       │
                 │  this market?      │
                 └─────────┬──────────┘
                           │ No
                           v
                 ┌────────────────────┐
                 │  Check: Whale      │──── Yes ────> Skip
                 │  Too Old?          │
                 │  (>120 seconds)    │
                 └─────────┬──────────┘
                           │ No
                           v
                 ┌────────────────────┐
                 │  Get Entry Price   │
                 │  from Orderbook    │
                 └─────────┬──────────┘
                           │
                           v
                 ┌────────────────────┐
                 │  Place Order via   │
                 │  trading_client.   │
                 │  place_order()     │
                 └─────────┬──────────┘
                           │
                           v
                 ┌────────────────────┐
                 │  Track in          │
                 │  _followed_whales  │
                 │  (order_id link)   │
                 └────────────────────┘
```

### 3.2 Place Order Code Path

```python
# In V3TradingClientIntegration.place_order()
# File: clients/trading_client_integration.py

async def place_order(
    self,
    ticker: str,
    action: str,      # "buy" or "sell"
    side: str,        # "yes" or "no"
    count: int,       # Number of contracts
    price: int,       # Price in cents (1-99)
    order_type: str = "limit",
    order_group_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Place an order through the trading client.

    Returns the full Kalshi API response including order_id.
    """
    # Validate connection
    if not self._connected:
        raise RuntimeError("Cannot place order - trading client not connected")

    # Validate limits
    if len(self._metrics.open_orders) >= self._max_orders:
        raise ValueError(f"Maximum orders ({self._max_orders}) exceeded")

    # Use order group for portfolio limits
    group_id = order_group_id or self._order_group_id

    # Delegate to demo client
    response = await self._client.create_order(
        ticker=ticker,
        action=action,
        side=side,
        count=count,
        price=price,
        type=order_type,
        order_group_id=group_id
    )

    # Track in metrics
    if "order" in response:
        order_id = response["order"]["order_id"]
        self._metrics.open_orders[order_id] = response["order"]

    return response
```

### 3.3 Sync to Verify

After placing orders, sync to see actual state:

```python
# Sync to get Kalshi's view of our orders
state, changes = await self._trading_client.sync_with_kalshi()

# Check our order status
for order_id, order_data in state.orders.items():
    status = order_data.get("status")
    fill_count = order_data.get("fill_count", 0)
    remaining = order_data.get("remaining_count", 0)

    if status == "executed":
        # Order fully filled
        logger.info(f"Order {order_id} fully executed: {fill_count} contracts")
    elif status == "resting" and fill_count > 0:
        # Partial fill
        logger.info(f"Order {order_id} partially filled: {fill_count}/{fill_count + remaining}")
    elif status == "canceled":
        # Order was canceled (maybe by market close)
        logger.warning(f"Order {order_id} was canceled")
```

---

## Part 4: Component Integration

### 4.1 Current Implementation Status

| Component | File | Status |
|-----------|------|--------|
| KalshiDemoTradingClient | `clients/demo_client.py` | Complete |
| V3TradingClientIntegration | `clients/trading_client_integration.py` | Complete |
| KalshiDataSync | `sync/kalshi_data_sync.py` | Complete |
| TraderState | `state/trader_state.py` | Complete |
| WhaleTracker | `services/whale_tracker.py` | Complete |
| TradingDecisionService | `services/trading_decision_service.py` | **Stubs Only** |
| TradingFlowOrchestrator | `core/trading_flow_orchestrator.py` | **Not Wired** |

### 4.2 What Needs Implementation

#### A. Fix TradingDecisionService Stubs

**Current** (lines 319-329):
```python
async def _execute_buy(self, decision: TradingDecision) -> bool:
    """Execute a buy order."""
    # TODO: Implement actual order placement
    logger.info(f"STUB: Would buy {decision.quantity} {decision.side} @ {decision.price}")
    return True  # Fake success
```

**Required**:
```python
async def _execute_buy(self, decision: TradingDecision) -> bool:
    """Execute a buy order through the trading client."""
    if not self._trading_client:
        logger.error("No trading client configured")
        return False

    try:
        response = await self._trading_client.place_order(
            ticker=decision.market,
            action="buy",
            side=decision.side,
            count=decision.quantity,
            price=decision.price,
            order_type="limit"
        )

        if "order" in response:
            order_id = response["order"]["order_id"]
            logger.info(f"Order placed: {order_id}")
            return True

        logger.error(f"Unexpected response: {response}")
        return False

    except Exception as e:
        logger.error(f"Order placement failed: {e}")
        return False
```

#### B. Add WHALE_FOLLOWER Strategy

```python
class TradingStrategy(Enum):
    HOLD = "hold"
    PAPER_TEST = "paper_test"
    RL_MODEL = "rl_model"
    CUSTOM = "custom"
    WHALE_FOLLOWER = "whale_follower"  # NEW
```

#### C. Wire WhaleTracker to TradingDecisionService

```python
# In TradingDecisionService or TradingFlowOrchestrator

async def _whale_follower_strategy(
    self,
    whale_queue: List[BigBet],
    orderbook_snapshot: Optional[dict] = None
) -> Optional[TradingDecision]:
    """
    Naive whale following: take top unfollowed whale.

    Args:
        whale_queue: Current whale queue from WhaleTracker
        orderbook_snapshot: Optional orderbook for price lookup

    Returns:
        TradingDecision or None if no whale to follow
    """
    for whale in whale_queue:
        whale_id = f"{whale.market_ticker}:{whale.timestamp_ms}:{whale.price_cents}:{whale.side}"

        # Skip if already followed
        if whale_id in self._followed_whales:
            continue

        # Skip if whale is too old (configurable, default 120s)
        age_seconds = (time.time() * 1000 - whale.timestamp_ms) / 1000
        if age_seconds > self._max_whale_age:
            continue

        # Skip if we already have a position in this market
        if whale.market_ticker in self._state_container.trading_state.positions:
            continue

        # Determine entry price
        if orderbook_snapshot:
            entry_price = self._calculate_entry_price(
                orderbook_snapshot,
                whale.side
            )
        else:
            # Use whale's price as reference if no orderbook
            entry_price = whale.price_cents

        return TradingDecision(
            action="buy",
            market=whale.market_ticker,
            side=whale.side,
            quantity=self._position_size,  # Configurable, default 5
            price=entry_price,
            reason=f"Following ${whale.whale_size / 100:.0f} whale",
            strategy=TradingStrategy.WHALE_FOLLOWER
        )

    return None
```

---

## Part 5: Error Handling

### 5.1 No Fallback Logic

Per requirements, we don't have fallback logic. Errors are surfaced clearly:

```python
# BAD - Don't do this
try:
    response = await place_order(...)
except Exception:
    logger.warning("Order failed, using cached value")
    return fake_response  # NO!

# GOOD - Fail clearly
try:
    response = await place_order(...)
except Exception as e:
    logger.error(f"Order placement failed: {e}")
    raise  # Propagate the error
```

### 5.2 Error Categories

| Error Type | Handling |
|------------|----------|
| Connection lost | Retry connection, fail if timeout |
| Order rejected | Log reason, don't retry same order |
| Insufficient funds | Log balance, halt trading |
| Market closed | Log market status, skip order |
| Rate limited | Backoff, retry with delay |
| API error (4xx/5xx) | Log full error, surface to coordinator |

### 5.3 Health Implications

Errors affect component health:

```python
# In V3TradingClientIntegration.is_healthy()
if self._consecutive_api_errors >= self._max_api_errors:
    return False  # Mark unhealthy after N consecutive failures
```

---

## Part 6: Configuration

### 6.1 Existing Configuration

```python
# In environment.py
V3_TRADING_MAX_ORDERS = int(os.getenv("V3_TRADING_MAX_ORDERS", "10"))
V3_TRADING_MAX_POSITION_SIZE = int(os.getenv("V3_TRADING_MAX_POSITION_SIZE", "100"))
```

### 6.2 New Configuration Needed

```bash
# Whale following configuration
WHALE_FOLLOW_POSITION_SIZE=5           # Contracts per whale follow
WHALE_FOLLOW_MAX_AGE_SECONDS=120       # Max whale age to consider
WHALE_FOLLOW_MAX_POSITIONS=3           # Max concurrent positions
WHALE_FOLLOW_PRICE_OFFSET_CENTS=2      # Bid above best bid
```

---

## Part 7: Demo Environment Cleanup

### 7.1 Current State
- 100 open orders
- 180 positions

### 7.2 Cleanup Strategy

**Orders** - Cancel via API:
```python
# Already implemented in trading_client_integration.py
await self._trading_client.cancel_all_orders()
```

**Positions** - Let them settle:
- Positions represent exposure to market outcomes
- Cannot "close" prediction market positions like stocks
- They settle when the market settles (natural expiration)
- Focus forward on new whale-following trades

### 7.3 Cleanup Endpoint

Add `/v3/cleanup` endpoint:
```python
@router.post("/v3/cleanup")
async def cleanup_demo_environment():
    """Cancel all open orders for fresh start."""
    result = await coordinator.trading_client.cancel_all_orders()
    return {
        "cancelled": len(result.get("cancelled", [])),
        "errors": len(result.get("errors", [])),
        "note": "Positions will settle naturally when markets close"
    }
```

---

## Part 8: Testing Strategy

### 8.1 Unit Tests
- Order creation request formatting
- Response parsing
- Error handling

### 8.2 Integration Tests
- Place order on demo account
- Verify order appears in get_orders
- Verify position updates after fill
- Order cancellation

### 8.3 E2E Test
1. Start V3 trader in paper mode
2. Wait for whale detection
3. Verify order placed
4. Sync and verify position created
5. Check frontend shows order/position

---

## Part 9: Implementation Order

### Phase 1: Foundation (Immediate)
1. Add `/v3/cleanup` endpoint for demo orders
2. Add debug logging to whale queue frontend
3. Verify whale queue shows in console

### Phase 2: Execution Core (Next)
1. Add WHALE_FOLLOWER strategy enum
2. Implement `_whale_follower_strategy()` in TradingDecisionService
3. Fix `_execute_buy()` and `_execute_sell()` stubs
4. Add `_followed_whales` tracking dict

### Phase 3: Integration (After Core)
1. Wire TradingFlowOrchestrator to use whale strategy
2. Add whale follow configuration to environment.py
3. Wire coordinator to pass whale queue to decision service

### Phase 4: Validation
1. Run V3 trader in paper mode
2. Monitor whale detection
3. Verify orders placed on whale signals
4. Sync and verify positions created

---

## Part 10: Open Questions for Review

1. **Position sizing**: Should we use fixed contracts (5) or percentage of whale size?

2. **Entry pricing**:
   - Use whale's exact price?
   - Use best bid/ask from orderbook?
   - Offset by N cents for better fill probability?

3. **Exit strategy**:
   - Hold until settlement?
   - Exit on profit target?
   - For MVP, recommend: hold until settlement (simplest)

4. **Concurrent whale follows**:
   - Max 3 positions at once?
   - Or unlimited within order group limits?

5. **Whale age threshold**:
   - 120 seconds seems reasonable
   - Too short = miss opportunities
   - Too long = stale signal

---

## Appendix A: Existing File References

| Purpose | File Path |
|---------|-----------|
| Demo API client | `clients/demo_client.py` |
| Trading integration | `clients/trading_client_integration.py` |
| State sync | `sync/kalshi_data_sync.py` |
| Trader state | `state/trader_state.py` |
| Whale tracker | `services/whale_tracker.py` |
| Trading decisions | `services/trading_decision_service.py` |
| Flow orchestrator | `core/trading_flow_orchestrator.py` |
| Coordinator | `core/coordinator.py` |
| Configuration | `config/environment.py` |

---

## Appendix B: API Response Examples

### Create Order Response
```json
{
  "order": {
    "order_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
    "user_id": "user123",
    "ticker": "INXD-25JAN03",
    "side": "yes",
    "action": "buy",
    "type": "limit",
    "status": "resting",
    "yes_price": 65,
    "no_price": 35,
    "initial_count": 5,
    "fill_count": 0,
    "remaining_count": 5,
    "created_time": "2025-12-27T10:30:00.000Z",
    "last_update_time": "2025-12-27T10:30:00.000Z",
    "order_group_id": "group-uuid-here"
  }
}
```

### Get Positions Response
```json
{
  "market_positions": [
    {
      "ticker": "INXD-25JAN03",
      "position": 5,
      "total_traded": 3250,
      "total_traded_dollars": "32.50",
      "market_exposure": 3250,
      "market_exposure_dollars": "32.50",
      "realized_pnl": 0,
      "realized_pnl_dollars": "0.00",
      "fees_paid": 18,
      "fees_paid_dollars": "0.18",
      "last_updated_ts": "2025-12-27T10:32:00.000Z"
    }
  ],
  "event_positions": []
}
```

---

*Document ready for Flore's review.*
