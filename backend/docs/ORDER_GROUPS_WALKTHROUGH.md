# Order Groups: How They Work & Integration Guide

## Current Order Placement Flow (Without Order Groups)

### Step-by-Step Flow

```
1. ActorService receives orderbook delta
   └─> Calls execute_limit_order_action(action, market_ticker, orderbook_snapshot)

2. KalshiMultiMarketOrderManager.execute_limit_order_action()
   ├─> Maps action (0-4) to side/contract_side
   ├─> Calculates limit price from orderbook
   ├─> Checks cash reserves
   └─> Calls _place_order_via_kalshi()

3. _place_order_via_kalshi()
   └─> Calls trading_client.create_order() with:
       {
         "ticker": "INXD-25JAN03",
         "action": "buy",
         "side": "yes",
         "count": 5,
         "yes_price": 50,
         "type": "limit"
       }

4. KalshiDemoTradingClient.create_order()
   └─> POST /portfolio/orders
       └─> Returns: { "order": { "order_id": "abc123", ... } }

5. Order tracking
   └─> OrderManager tracks order in self.open_orders
   └─> Waits for fill via FillListener WebSocket
```

### Current Code Locations

**Order Creation:**
- `KalshiMultiMarketOrderManager.execute_limit_order_action()` (line ~953)
- `KalshiMultiMarketOrderManager._place_order_via_kalshi()` (line ~1050)
- `KalshiDemoTradingClient.create_order()` (line ~508)

**Order Tracking:**
- `KalshiMultiMarketOrderManager.open_orders` (dict of OrderInfo)
- `KalshiMultiMarketOrderManager._kalshi_to_internal` (ID mapping)

---

## How Order Groups Work

### Concept

Order groups are a **contract limit enforcement mechanism** provided by Kalshi:

1. **Create a group** with a `contracts_limit` (e.g., 100 contracts)
2. **Associate orders** with the group by including `order_group_id`
3. **Kalshi tracks** total contracts matched across all orders in the group
4. **When limit hit**: Kalshi automatically cancels ALL orders in the group
5. **Reset group**: Clear the limit counter to allow new orders

### Key Mechanics

**Contracts Limit = Total Contracts Matched**
- When you place an order for 5 contracts and it fills → 5 contracts matched
- When you place another order for 10 contracts and it fills → 15 total matched
- When total reaches `contracts_limit` → all remaining orders cancelled

**Automatic Cancellation**
- Kalshi cancels orders **server-side** (no API call needed)
- Happens immediately when limit is reached
- All orders in the group are cancelled, not just new ones

**Reset Mechanism**
- After limit hit, you must reset the group to place new orders
- Reset clears the contracts matched counter
- Group can be reused (no need to create new group)

---

## Order Groups Integration: Step-by-Step

### Phase 1: Startup - Create Order Group

**Location:** `KalshiMultiMarketOrderManager.initialize()`

```python
async def initialize(self, initialization_tracker=None):
    # ... existing initialization ...
    
    # Create order group (if enabled)
    if config.RL_USE_ORDER_GROUPS:
        try:
            group_response = await self.trading_client.create_order_group(
                contracts_limit=config.RL_MAX_TOTAL_CONTRACTS  # e.g., 500
            )
            self.order_group_id = group_response["order_group_id"]
            logger.info(f"Order group created: {self.order_group_id} (limit: {config.RL_MAX_TOTAL_CONTRACTS})")
        except Exception as e:
            logger.error(f"Failed to create order group: {e}")
            # Could disable order groups or use fallback
            self.order_group_id = None
```

**What Changes:**
- ✅ Add `self.order_group_id` attribute
- ✅ Create group on startup (one-time)
- ✅ Store group ID for use in all orders

**No Breaking Changes:**
- If order groups disabled, `self.order_group_id = None`
- Existing code continues to work

---

### Phase 2: Order Placement - Include Group ID

**Location:** `KalshiDemoTradingClient.create_order()`

**Current Code:**
```python
async def create_order(
    self,
    ticker: str,
    action: str,
    side: str,
    count: int,
    price: Optional[int] = None,
    type: str = "limit"
) -> Dict[str, Any]:
    order_data = {
        "ticker": ticker,
        "action": action,
        "side": side,
        "count": count,
        "type": type
    }
    # ... add price fields ...
    response = await self._make_request("POST", "/portfolio/orders", order_data)
    return response
```

**With Order Groups:**
```python
async def create_order(
    self,
    ticker: str,
    action: str,
    side: str,
    count: int,
    price: Optional[int] = None,
    type: str = "limit",
    order_group_id: Optional[str] = None  # NEW PARAMETER
) -> Dict[str, Any]:
    order_data = {
        "ticker": ticker,
        "action": action,
        "side": side,
        "count": count,
        "type": type
    }
    
    # Include order_group_id if provided
    if order_group_id:
        order_data["order_group_id"] = order_group_id
    
    # ... add price fields ...
    response = await self._make_request("POST", "/portfolio/orders", order_data)
    return response
```

**Location:** `KalshiMultiMarketOrderManager._place_order_via_kalshi()`

**Current Code:**
```python
async def _place_order_via_kalshi(
    self,
    ticker: str,
    side: OrderSide,
    contract_side: ContractSide,
    quantity: int,
    limit_price: int
) -> Optional[str]:
    kalshi_action = "buy" if side == OrderSide.BUY else "sell"
    kalshi_side = "yes" if contract_side == ContractSide.YES else "no"
    
    response = await self.trading_client.create_order(
        ticker=ticker,
        action=kalshi_action,
        side=kalshi_side,
        count=quantity,
        price=limit_price,
        type="limit"
    )
    # ... extract order_id ...
```

**With Order Groups:**
```python
async def _place_order_via_kalshi(
    self,
    ticker: str,
    side: OrderSide,
    contract_side: ContractSide,
    quantity: int,
    limit_price: int
) -> Optional[str]:
    kalshi_action = "buy" if side == OrderSide.BUY else "sell"
    kalshi_side = "yes" if contract_side == ContractSide.YES else "no"
    
    response = await self.trading_client.create_order(
        ticker=ticker,
        action=kalshi_action,
        side=kalshi_side,
        count=quantity,
        price=limit_price,
        type="limit",
        order_group_id=self.order_group_id  # NEW: Include group ID
    )
    # ... extract order_id ...
```

**What Changes:**
- ✅ Add optional `order_group_id` parameter to `create_order()`
- ✅ Pass `self.order_group_id` when placing orders
- ✅ If `order_group_id` is None, orders work normally (backward compatible)

**No Breaking Changes:**
- Parameter is optional
- Existing calls without `order_group_id` still work
- Orders without group ID are not limited by groups

---

### Phase 3: Monitoring - Check Group State

**Location:** New method in `KalshiMultiMarketOrderManager`

```python
async def _check_order_group_state(self) -> Dict[str, Any]:
    """
    Check order group state to see if limit was hit.
    
    Returns:
        Dict with group state info
    """
    if not self.order_group_id:
        return {"enabled": False}
    
    try:
        group_info = await self.trading_client.get_order_group(self.order_group_id)
        
        # Group info structure (from Kalshi API):
        # {
        #   "is_auto_cancel_enabled": true,
        #   "orders": ["order_id_1", "order_id_2", ...]  # List of order IDs in group
        # }
        
        # If group has no orders and we've been trading, limit might have been hit
        # (Kalshi cancels all orders when limit hit, so orders list becomes empty)
        
        return {
            "enabled": True,
            "order_group_id": self.order_group_id,
            "active_orders": len(group_info.get("orders", [])),
            "auto_cancel_enabled": group_info.get("is_auto_cancel_enabled", False)
        }
    except Exception as e:
        logger.error(f"Failed to check order group state: {e}")
        return {"enabled": True, "error": str(e)}
```

**Integration Point:** Periodic sync or recalibration loop

```python
async def _periodic_sync(self):
    """Periodic sync with Kalshi (every 30s)."""
    while self.is_active:
        await asyncio.sleep(30)
        
        # Existing sync operations
        await self.sync_orders_with_kalshi()
        await self._sync_positions_with_kalshi()
        
        # NEW: Check order group state
        if self.order_group_id:
            group_state = await self._check_order_group_state()
            if group_state.get("active_orders") == 0 and self._has_recent_trading_activity():
                # Limit likely hit - all orders cancelled
                logger.warning("Order group limit may have been hit - checking...")
                await self._handle_order_group_limit_hit()
```

**What Changes:**
- ✅ Add `_check_order_group_state()` method
- ✅ Integrate into periodic sync or recalibration loop
- ✅ Detect when limit hit (all orders cancelled)

---

### Phase 4: Recovery - Reset Group When Limit Hit

**Location:** New method in `KalshiMultiMarketOrderManager`

```python
async def _handle_order_group_limit_hit(self) -> None:
    """
    Handle order group limit being hit.
    
    When limit is hit, Kalshi cancels all orders. We need to:
    1. Verify limit was actually hit (not just orders filled normally)
    2. Reset the group to allow new orders
    3. Log the event
    """
    if not self.order_group_id:
        return
    
    try:
        # Reset the group (clears contracts matched counter)
        await self.trading_client.reset_order_group(self.order_group_id)
        logger.info(f"Order group {self.order_group_id} reset - can place new orders")
        
        # Optionally: Broadcast event to UI
        if self._websocket_manager:
            await self._websocket_manager.broadcast_order_group_reset({
                "order_group_id": self.order_group_id,
                "reset_at": time.time(),
                "reason": "contracts_limit_hit"
            })
    except Exception as e:
        logger.error(f"Failed to reset order group: {e}")
```

**Integration Point:** Called from `_check_order_group_state()` or periodic sync

**What Changes:**
- ✅ Add `_handle_order_group_limit_hit()` method
- ✅ Reset group to allow new orders
- ✅ Log and broadcast event

---

## Complete Flow: With Order Groups

### Startup Sequence

```
1. KalshiMultiMarketOrderManager.initialize()
   ├─> Connect to Kalshi
   ├─> Start fill/position listeners
   ├─> Sync orders/positions
   └─> Create order group (if enabled)
       └─> POST /portfolio/order_groups/create
           └─> Store order_group_id
```

### Order Placement Sequence

```
1. ActorService.execute_limit_order_action()
   └─> KalshiMultiMarketOrderManager.execute_limit_order_action()
       └─> _place_order_via_kalshi()
           └─> trading_client.create_order(
                   ...,
                   order_group_id=self.order_group_id  # NEW
               )
               └─> POST /portfolio/orders
                   {
                     "ticker": "...",
                     "action": "buy",
                     "side": "yes",
                     "count": 5,
                     "yes_price": 50,
                     "order_group_id": "abc123"  # NEW FIELD
                   }
```

### Limit Hit Sequence

```
1. Orders fill → contracts matched increases
   └─> When total contracts matched >= contracts_limit:
       └─> Kalshi automatically cancels ALL orders in group
           └─> FillListener receives cancellation notifications
               └─> OrderManager removes orders from tracking

2. Periodic sync detects limit hit
   └─> _check_order_group_state()
       └─> Sees active_orders = 0 (all cancelled)
           └─> _handle_order_group_limit_hit()
               └─> PUT /portfolio/order_groups/{order_group_id}/reset
                   └─> Group reset, can place new orders
```

---

## Code Changes Summary

### Files to Modify

1. **`backend/src/kalshiflow_rl/trading/demo_client.py`**
   - Add 5 order group API methods:
     - `create_order_group(contracts_limit)`
     - `get_order_groups()`
     - `get_order_group(order_group_id)`
     - `reset_order_group(order_group_id)`
     - `delete_order_group(order_group_id)`
   - Modify `create_order()` to accept optional `order_group_id` parameter

2. **`backend/src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py`**
   - Add `self.order_group_id` attribute
   - Create order group in `initialize()`
   - Pass `order_group_id` in `_place_order_via_kalshi()`
   - Add `_check_order_group_state()` method
   - Add `_handle_order_group_limit_hit()` method
   - Integrate group checking into periodic sync

3. **`backend/src/kalshiflow_rl/config.py`**
   - Add `RL_USE_ORDER_GROUPS` config flag
   - Add `RL_MAX_TOTAL_CONTRACTS` config value

### Backward Compatibility

✅ **All changes are backward compatible:**
- Order group creation is optional (config flag)
- `order_group_id` parameter is optional
- If `order_group_id` is None, orders work exactly as before
- No breaking changes to existing APIs

---

## Benefits & Impact

### What Order Groups Give Us

1. **Automatic Limit Enforcement**
   - No need to manually check limits before each order
   - Kalshi enforces limits server-side (fail-safe)

2. **Simplified Position Management**
   - Set total contract limit once (e.g., 500 contracts)
   - All orders automatically respect the limit
   - No complex logic to track total contracts manually

3. **Clean State Management**
   - When limit hit, all orders cancelled automatically
   - Clear state: no orders, can reset and continue
   - Easier to reason about system state

4. **Recovery Mechanism**
   - Reset group to start fresh after limit hit
   - No need to manually cancel orders
   - Can be part of self-healing strategy

### What Doesn't Change

- ✅ Order placement logic (just add one parameter)
- ✅ Fill processing (works exactly the same)
- ✅ Position tracking (unchanged)
- ✅ Cash management (unchanged)
- ✅ ActorService flow (unchanged)

### What We Still Need to Do

- ❌ Position closing logic (independent of order groups)
- ❌ Market viability checks (independent)
- ❌ Portfolio health assessment (independent)
- ❌ Cash flow monitoring (independent)

**Order groups are complementary, not a replacement for these features.**

---

## Example: Order Group Lifecycle

### Scenario: 100 Contract Limit

```
Startup:
  └─> Create order group with contracts_limit=100
      └─> order_group_id = "group_123"

Trading:
  Order 1: Buy 10 YES @ 50¢
    └─> Includes order_group_id="group_123"
    └─> Fills → 10 contracts matched
  
  Order 2: Buy 20 YES @ 51¢
    └─> Includes order_group_id="group_123"
    └─> Fills → 30 total contracts matched
  
  Order 3: Buy 30 YES @ 52¢
    └─> Includes order_group_id="group_123"
    └─> Fills → 60 total contracts matched
  
  Order 4: Buy 50 YES @ 53¢
    └─> Includes order_group_id="group_123"
    └─> Fills → 110 total contracts matched
    └─> ⚠️ LIMIT HIT (110 > 100)
    └─> Kalshi automatically cancels ALL remaining orders in group

Recovery:
  └─> Periodic sync detects all orders cancelled
      └─> Reset order group
          └─> Contracts matched counter cleared
          └─> Can place new orders (starts at 0 again)
```

---

## Decision Points

### 1. Single Group vs Multiple Groups

**Option A: Single Group (Recommended)**
- One group for all markets
- Total portfolio limit (e.g., 500 contracts)
- Simpler to manage
- ✅ **Recommended for MVP**

**Option B: Per-Market Groups**
- One group per market
- Per-market limits (e.g., 100 contracts per market)
- More complex to manage
- ⚠️ Consider for future if needed

### 2. When to Reset Group

**Option A: Automatic Reset (Recommended)**
- Reset immediately when limit hit detected
- Allows trading to continue automatically
- ✅ **Recommended for MVP**

**Option B: Manual Reset**
- Require manual intervention to reset
- More control, but less automated
- ⚠️ Consider if you want manual oversight

### 3. Limit Strategy

**Option A: Conservative (Recommended)**
- Set limit = max you ever want to trade
- Reset periodically (e.g., daily)
- ✅ **Recommended for MVP**

**Option B: Aggressive**
- Set limit = session limit
- Reset after each session
- ⚠️ More complex lifecycle management

---

## Testing Strategy

### Unit Tests

1. **Order Group Creation**
   - Test group creation on startup
   - Test error handling if creation fails

2. **Order Placement with Group**
   - Test order placement includes `order_group_id`
   - Test order placement without group (backward compat)

3. **Group State Checking**
   - Test `_check_order_group_state()` method
   - Test limit hit detection

4. **Group Reset**
   - Test `_handle_order_group_limit_hit()` method
   - Test reset after limit hit

### Integration Tests

1. **Full Lifecycle**
   - Create group → Place orders → Hit limit → Reset → Place more orders

2. **Edge Cases**
   - Group creation fails → Fallback to no groups
   - Reset fails → Retry logic
   - Multiple rapid resets

### Paper Trading Tests

1. **Real Limit Hit**
   - Place orders until limit hit
   - Verify all orders cancelled
   - Verify reset allows new orders

2. **Recovery**
   - Hit limit → Reset → Continue trading
   - Verify state is clean after reset

---

## Summary

**Order groups are a simple but powerful addition:**

1. **Minimal Code Changes**: Just add group ID to order placement
2. **Backward Compatible**: All changes are optional
3. **Fail-Safe**: Kalshi enforces limits server-side
4. **Clean State**: Automatic cancellation when limit hit
5. **Recovery**: Reset mechanism for continued trading

**They complement position closing and calibration, but don't replace them.**

The investment is small (2-3 days), the risk is low (optional feature), and the benefits are significant (automatic limit enforcement, fail-safe mechanism).

