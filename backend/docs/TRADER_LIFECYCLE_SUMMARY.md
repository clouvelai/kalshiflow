# Trader Lifecycle: Current State & Improvement Plan

## Executive Summary

**Current Problem:** The trader opens positions until cash is exhausted, then has no mechanism to close positions and recoup cash. It cannot self-calibrate to the environment or recover from bad states.

**Key Insight:** Inspired by video game bots, the trader needs to:
1. **Calibrate** on startup (check where it is, what state it's in)
2. **Recalibrate** continuously (verify it's still in sync)
3. **Recover** when it goes off track (automatically fix drift)

**Solution:** Implement position management, environment calibration, and self-healing mechanisms.

---

## Current Actor Loop (Simplified)

```
STARTUP
├─ Infrastructure (DB, EventBus, OrderbookClient)
├─ OrderManager Init
│  ├─ Connect to Kalshi
│  ├─ Start fill/position listeners
│  ├─ Sync orders/positions/cash
│  └─ Start periodic sync (30s)
└─ ActorService Init
   ├─ Load RL model
   ├─ Subscribe to orderbook events
   └─ Start event processing loop

TRADING LOOP (Continuous)
└─ For each orderbook delta:
   ├─ Build observation (orderbook + portfolio)
   ├─ Select action (RL model or hardcoded)
   ├─ Execute action (place order if not HOLD/throttled)
   └─ Update positions (read after 100ms delay)

PARALLEL PROCESSES
├─ FillListener: Real-time fill notifications → update positions
├─ PositionListener: Real-time position updates
└─ Periodic Sync: Every 30s, sync with Kalshi API
```

---

## Critical Gaps

### 1. **No Position Closing** ⚠️ CRITICAL

**What's Missing:**
- Logic to close positions when:
  - Cash is low (need to free up capital)
  - Position P&L hits thresholds (take profit/stop loss)
  - Market is closing (event ending)
  - Position is stuck (no fills for too long)

**Impact:**
- Trader becomes non-functional after cash exhaustion
- Cannot adapt to changing conditions
- Positions remain open until event settlement

**Order Groups Impact:**
- **Position closing can be implemented independently** - it's just placing opposite orders
- Order groups are about limiting total contracts matched, not about closing positions
- Order groups could ensure we don't exceed limits when closing positions (complementary, not blocking)
- **Recommendation:** Implement position closing first, then enhance with order groups if needed

**Fix Needed:**
```python
# Add to KalshiMultiMarketOrderManager
async def close_position(self, ticker: str, reason: str):
    """Close position by placing opposite order."""
    position = self.positions.get(ticker)
    if not position or position.is_flat:
        return
    
    # Determine opposite side
    if position.contracts > 0:  # Long YES
        side = OrderSide.SELL
        contract_side = ContractSide.YES
    else:  # Long NO
        side = OrderSide.SELL
        contract_side = ContractSide.NO
    
    # Place closing order (order_group_id included if available)
    await self.execute_limit_order_action(
        action=...,  # Map to appropriate action
        market_ticker=ticker,
        orderbook_snapshot=...,
        reason=f"close_position:{reason}"
    )
```

### 2. **Incomplete Environment Calibration** ⚠️ HIGH

**Current State:**
- ✅ We DO have system launch sequence and systems tab that runs at init
- ✅ We DO sync with Kalshi (balance, positions, orders) via `InitializationTracker`
- ✅ We DO check environment state during initialization
- ✅ Systems tab shows initialization progress

**What's Missing:**
- ❌ Display "last portfolio data sync X" timestamp in systems tab
- ❌ Additional calibration checks beyond state sync:
  - Markets are viable (still active) - not checked
  - Portfolio is healthy (no stuck positions) - not checked
  - Cash flow is sufficient - not checked

**Impact:**
- Trader may start with non-viable markets
- No visibility into portfolio health issues
- No cash flow assessment before trading begins

**Calibration Strategy:**
Calibration should be two-part:
1. **(a) Sync state with Kalshi** - ✅ Already done via `InitializationTracker`
   - Sync balance, positions, orders, settlements
   - Show "last portfolio data sync X" in systems tab
2. **(b) Additional trader calibration checks** - ❌ Missing
   - Check markets are viable (still active)
   - Assess portfolio health (any stuck positions?)
   - Verify cash flow is sufficient

**Fix Needed:**
```python
# Add to KalshiMultiMarketOrderManager.initialize()
async def calibrate_environment(self, initialization_tracker=None):
    """Calibrate to current environment state."""
    # Part (a): Sync state with Kalshi - already done via InitializationTracker
    # Just need to track last sync timestamp for systems tab display
    
    # Part (b): Additional trader calibration checks
    if initialization_tracker:
        await initialization_tracker.mark_step_in_progress("trader_calibration")
    
    # 1. Check market viability
    viable_markets = await self._check_markets_viable()
    if not viable_markets["all_viable"]:
        logger.warning(f"Some markets not viable: {viable_markets['non_viable']}")
    
    # 2. Assess portfolio health
    portfolio_health = await self._assess_portfolio_health()
    if portfolio_health["needs_recovery"]:
        logger.warning(f"Portfolio health issues: {portfolio_health['issues']}")
        # Could trigger recovery actions here
    
    # 3. Verify cash flow
    cash_flow_status = await self._verify_cash_flow()
    if not cash_flow_status["sufficient"]:
        logger.warning(f"Cash flow insufficient: {cash_flow_status['reason']}")
        # Could trigger position closing here
    
    if initialization_tracker:
        await initialization_tracker.mark_step_complete("trader_calibration", {
            "markets_viable": viable_markets["all_viable"],
            "portfolio_healthy": portfolio_health["healthy"],
            "cash_flow_sufficient": cash_flow_status["sufficient"],
            "last_sync_timestamp": time.time()  # For systems tab display
        })
```

### 3. **No Continuous Recalibration** ⚠️ HIGH

**What's Missing:**
- Continuous monitoring of:
  - Position health (should any be closed?)
  - Market states (are markets closing?)
  - Cash flow (approaching reserve threshold?)
  - State drift (calculated vs synced values)

**Impact:**
- Trader doesn't adapt to changing conditions
- Drift accumulates over time
- Missed opportunities to close positions

**Decision Point:**
- Option A: Replace current periodic sync (30s) with full recalibration loop
- Option B: Add recalibration loop in addition to periodic sync
- Option C: Introduce actor loop: `calibrate → trade/close/cleanup → calibrate`

**Recommendation:** Start with Option B (add recalibration), then consider Option C if we want more structured lifecycle management.

**Fix Needed:**
```python
# Add recalibration loop (could replace or complement periodic sync)
async def recalibrate_loop(self):
    """Continuously recalibrate to environment."""
    while self.is_active:
        await asyncio.sleep(60)  # Every minute (or replace periodic sync)
        
        # Full calibration cycle
        # 1. Sync state with Kalshi (like periodic sync does now)
        await self._sync_positions_with_kalshi()
        await self.sync_orders_with_kalshi()
        
        # 2. Additional trader calibration checks
        await self._check_markets_viable()
        await self._assess_portfolio_health()
        await self._verify_cash_flow()
        
        # 3. Check position health (should any be closed?)
        await self._monitor_position_health()
        
        # 4. Check market states (are markets closing?)
        await self._monitor_market_states()
        
        # 5. Check state drift
        drift = await self._check_state_drift()
        if drift > threshold:
            await self._correct_drift()
        
        # Update systems tab with last sync timestamp
        await self._update_systems_tab_sync_timestamp()
```

### 4. **No Order Group Management** ⚠️ MEDIUM

**What's Missing:**
- Order groups for coordinated order management
- Automatic cancellation when limits hit
- Simplified position limit enforcement

**Impact:**
- More complex position limit logic
- No fail-safe if our logic has bugs
- Harder to reason about system state

**Order Groups API Review:**

Based on [Kalshi Order Groups API](https://docs.kalshi.com/api-reference/order-groups/get-order-group), the full API surface is:

1. **Create Order Group** (`POST /portfolio/order_groups/create`)
   - Creates group with `contracts_limit`
   - Returns `order_group_id`
   - When limit is hit, all orders in group are automatically cancelled

2. **Get Order Groups** (`GET /portfolio/order_groups`)
   - Lists all order groups for user
   - Useful for discovering existing groups

3. **Get Order Group** (`GET /portfolio/order_groups/{order_group_id}`)
   - Returns group details: `is_auto_cancel_enabled`, `orders[]` (list of order IDs)
   - Useful for monitoring group state

4. **Reset Order Group** (`PUT /portfolio/order_groups/{order_group_id}/reset`)
   - Resets group to allow new orders after limit was hit
   - Clears the contracts matched counter

5. **Delete Order Group** (`DELETE /portfolio/order_groups/{order_group_id}`)
   - Deletes group (cancels all orders in group)
   - Useful for cleanup

**How Order Groups Work in Trading Mechanics:**

1. **Limit Enforcement**: When `contracts_limit` is reached, Kalshi automatically:
   - Cancels ALL orders in the group
   - Prevents new orders from being placed (until reset)

2. **Order Association**: Include `order_group_id` in order creation:
   ```python
   POST /portfolio/orders
   {
     "ticker": "...",
     "action": "buy",
     "side": "yes",
     "count": 5,
     "order_group_id": "abc123",  # Associate with group
     "yes_price": 50
   }
   ```

3. **State Monitoring**: Check group state to see if limit hit:
   ```python
   group = await get_order_group(order_group_id)
   if len(group["orders"]) == 0 and contracts_matched >= limit:
       # Limit hit, all orders cancelled
       await reset_order_group(order_group_id)  # Allow new orders
   ```

**Implementation Investment Assessment:**

**Effort Level:** Medium (2-3 days)
- Add 5 API methods to `KalshiDemoTradingClient` (~1 day)
- Integrate order group creation in `initialize()` (~0.5 day)
- Modify order placement to include `order_group_id` (~0.5 day)
- Add group state monitoring (~0.5 day)
- Testing and edge cases (~0.5 day)

**Benefits:**
- ✅ Automatic limit enforcement (fail-safe)
- ✅ Simplified position limit logic
- ✅ Clean state management (auto-cancellation)
- ✅ Recovery mechanism (reset to continue)

**Risks:**
- ⚠️ Need to handle group state correctly (check before placing orders)
- ⚠️ Need to reset groups appropriately (when to reset?)
- ⚠️ Multiple groups vs single group strategy (which is better?)

**Recommendation:** Implement order groups in Phase 3. They're complementary to position closing (not blocking), and provide valuable fail-safe mechanisms.

**Fix Needed:**
```python
# Add to demo_client.py
async def create_order_group(self, contracts_limit: int) -> Dict[str, Any]:
    """Create order group with contract limit."""
    return await self._make_request(
        "POST", 
        "/portfolio/order_groups/create",
        {"contracts_limit": contracts_limit}
    )

async def get_order_groups(self) -> Dict[str, Any]:
    """Get all order groups."""
    return await self._make_request("GET", "/portfolio/order_groups")

async def get_order_group(self, order_group_id: str) -> Dict[str, Any]:
    """Get specific order group details."""
    return await self._make_request(
        "GET",
        f"/portfolio/order_groups/{order_group_id}"
    )

async def reset_order_group(self, order_group_id: str) -> Dict[str, Any]:
    """Reset order group to allow new orders."""
    return await self._make_request(
        "PUT",
        f"/portfolio/order_groups/{order_group_id}/reset"
    )

async def delete_order_group(self, order_group_id: str) -> Dict[str, Any]:
    """Delete order group (cancels all orders)."""
    return await self._make_request(
        "DELETE",
        f"/portfolio/order_groups/{order_group_id}"
    )

# Add to KalshiMultiMarketOrderManager
async def initialize(self):
    # ... existing init ...
    
    # Create order group (optional, based on config)
    if config.RL_USE_ORDER_GROUPS:
        group = await self.trading_client.create_order_group(
            contracts_limit=config.RL_MAX_TOTAL_CONTRACTS
        )
        self.order_group_id = group["order_group_id"]
        logger.info(f"Order group created: {self.order_group_id}")
    
    # Include in all orders (modify create_order to include order_group_id)
    # Monitor group state periodically
```

---

## Recommended Implementation Plan

### Phase 1: Position Management (Week 1) - CRITICAL

**Goal:** Enable trader to close positions and recoup cash

**Decision:** Implement position closing **before** order groups. They're independent - position closing is about placing opposite orders, order groups are about limiting total contracts. We can enhance position closing with order groups later if needed.

**Tasks:**
1. ✅ Add `close_position()` method to `KalshiMultiMarketOrderManager`
2. ✅ Add position health monitoring (P&L, time in position)
3. ✅ Add cash recovery strategy (close positions when cash low)
4. ✅ Add market state monitoring (detect closing markets)
5. ✅ Test position closing in paper trading

**Files to Modify:**
- `backend/src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py`
- `backend/src/kalshiflow_rl/trading/actor_service.py` (add position monitoring)

**Note:** Order groups won't require changes to position closing logic - they're complementary features.

### Phase 2: Environment Calibration (Week 2) - HIGH

**Goal:** Complete trader calibration on startup and continuously

**Current State:**
- ✅ We have `InitializationTracker` that syncs state with Kalshi (balance, positions, orders)
- ✅ Systems tab shows initialization progress
- ❌ Missing: Additional trader calibration checks (markets viable, portfolio health, cash flow)
- ❌ Missing: "Last portfolio data sync X" timestamp in systems tab

**Tasks:**
1. ✅ Add "last portfolio data sync X" timestamp display to systems tab
2. ✅ Add trader calibration step to `initialize()` (markets viable, portfolio health, cash flow)
3. ✅ Integrate trader calibration into `InitializationTracker` workflow
4. ✅ Add continuous recalibration loop (replace or complement periodic sync)
5. ✅ Add state drift detection
6. ✅ Test calibration in various states

**Files to Modify:**
- `backend/src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py` (add calibration methods)
- `backend/src/kalshiflow_rl/trading/initialization_tracker.py` (add trader calibration step)
- `frontend/src/components/SystemHealth.jsx` (add last sync timestamp display)

**Decision Point:** Should recalibration replace periodic sync or run in addition? Start with addition, then consider replacement if it provides better coverage.

### Phase 3: Order Groups (Week 3) - MEDIUM

**Goal:** Use order groups for simplified position management and fail-safe limits

**Investment Assessment:**
- **Effort:** Medium (2-3 days)
- **Complexity:** Low-Medium (straightforward API integration)
- **Risk:** Low (optional feature, can be disabled via config)

**Tasks:**
1. ✅ Add 5 order group API methods to `KalshiDemoTradingClient`
   - `create_order_group()`
   - `get_order_groups()`
   - `get_order_group()`
   - `reset_order_group()`
   - `delete_order_group()`
2. ✅ Create order group on startup (if `RL_USE_ORDER_GROUPS=true`)
3. ✅ Include `order_group_id` in all order placements
4. ✅ Monitor group state periodically (check if limit hit)
5. ✅ Reset group when limit hit (to allow new orders)
6. ✅ Test order group limits and edge cases

**Files to Modify:**
- `backend/src/kalshiflow_rl/trading/demo_client.py` (add 5 API methods)
- `backend/src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py` (integrate groups)
- `backend/src/kalshiflow_rl/config.py` (add `RL_USE_ORDER_GROUPS` config)

**Note:** Order groups are complementary to position closing - they won't require changes to position closing logic.

### Phase 4: Self-Healing (Week 4) - MEDIUM

**Goal:** Automatically recover from bad states

**Tasks:**
1. ✅ Add state recovery (correct drift)
2. ✅ Add position recovery (close stuck positions)
3. ✅ Add comprehensive error handling
4. ✅ Test recovery scenarios

**Files to Modify:**
- `backend/src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py`

---

## Order Groups: Deep Dive

### What They Are

Order groups allow you to:
- Set a **contracts limit** for a group of orders
- **Automatically cancel all orders** in the group when limit is hit
- **Prevent new orders** until group is reset

### API Endpoints

```python
# Create order group
POST /portfolio/order_groups/create
{
  "contracts_limit": 100  # Max contracts that can be matched
}

# Response
{
  "order_group_id": "abc123"
}

# Include in orders
POST /portfolio/orders
{
  "ticker": "...",
  "action": "buy",
  "side": "yes",
  "count": 5,
  "order_group_id": "abc123",  # Include this
  "yes_price": 50
}

# Reset group (allows new orders)
PUT /portfolio/order_groups/{order_group_id}/reset

# Get group info
GET /portfolio/order_groups/{order_group_id}
```

### Use Cases for Our System

1. **Total Portfolio Limit**
   - Create one order group for all markets
   - Set limit = max total contracts (e.g., 500)
   - Automatically prevents over-leveraging

2. **Per-Market Limits**
   - Create order group per market
   - Set limit = max position per market (e.g., 100)
   - Automatically prevents over-trading in single market

3. **Session Limits**
   - Create order group per trading session
   - Set limit = max contracts for session
   - Automatically stops trading when limit hit

### Benefits

1. **Simplified Logic**: No need to manually check limits on every order
2. **Fail-Safe**: Kalshi enforces limits even if our logic has bugs
3. **Clean State**: When limit hit, all orders cancelled automatically
4. **Recovery**: Reset group to start fresh after limit hit

### Implementation Example

```python
class OrderGroupManager:
    def __init__(self, trading_client, max_contracts: int):
        self.trading_client = trading_client
        self.max_contracts = max_contracts
        self.order_group_id = None
    
    async def initialize(self):
        """Create order group on startup."""
        response = await self.trading_client.create_order_group(
            contracts_limit=self.max_contracts
        )
        self.order_group_id = response["order_group_id"]
        logger.info(f"Order group created: {self.order_group_id}")
    
    async def check_and_reset_if_needed(self):
        """Check if limit hit, reset if needed."""
        if not self.order_group_id:
            return
        
        group_info = await self.trading_client.get_order_group(
            self.order_group_id
        )
        
        if group_info["contracts_matched"] >= group_info["contracts_limit"]:
            logger.warning("Order group limit hit, resetting...")
            await self.trading_client.reset_order_group(self.order_group_id)
            logger.info("Order group reset, can place new orders")
```

---

## Key Metrics to Track

### Position Health
- Unrealized P&L per position
- Time in position
- Distance from take profit/stop loss
- Market closing time (if available)

### Cash Flow
- Available cash vs reserve threshold
- Cash invested vs recouped
- Projected cash needs (for closing positions)

### State Drift
- Calculated cash vs synced cash
- Calculated portfolio vs synced portfolio
- Position discrepancies

### Market States
- Market status (open/closed/ending)
- Time until market closes
- Event end time

---

## Success Criteria

### Phase 1 (Position Management)
- ✅ Trader can close positions when cash is low
- ✅ Trader can close positions based on P&L thresholds
- ✅ Trader can close positions when markets are closing
- ✅ Cash is recouped and available for new trades

### Phase 2 (Calibration)
- ✅ Trader verifies state on startup
- ✅ Trader detects drift and corrects it
- ✅ Trader adapts to market state changes
- ✅ Trader maintains sync with Kalshi

### Phase 3 (Order Groups)
- ✅ Order groups created on startup
- ✅ All orders include order_group_id
- ✅ Limits enforced automatically
- ✅ Groups reset when needed

### Phase 4 (Self-Healing)
- ✅ Trader recovers from bad states automatically
- ✅ Drift is corrected before it becomes a problem
- ✅ Stuck positions are closed automatically
- ✅ System maintains healthy state

---

## Next Steps

1. **Review this document** with the team
2. **Prioritize phases** based on business needs
3. **Start Phase 1** (position management) - most critical
4. **Test incrementally** in paper trading
5. **Iterate** based on results

---

## Questions to Answer

1. **Position Closing Strategy**: What triggers should close positions?
   - Cash threshold? (e.g., < $1000)
   - P&L thresholds? (e.g., +20% or -10%)
   - Time in position? (e.g., > 1 hour)
   - Market closing? (e.g., < 5 min until close)

2. **Order Group Limits**: What limits should we set?
   - Total contracts? (e.g., 500)
   - Per market? (e.g., 100)
   - Per session? (e.g., 1000)
   - **Strategy decision:** Single group for all markets, or per-market groups?

3. **Calibration Frequency**: How often should we recalibrate?
   - Every minute? (aggressive)
   - Every 5 minutes? (balanced)
   - Every 15 minutes? (conservative)
   - **Decision:** Replace periodic sync (30s) or run in addition?

4. **Recovery Strategy**: How aggressive should recovery be?
   - Close all positions if cash low? (aggressive)
   - Close worst positions first? (balanced)
   - Wait for manual intervention? (conservative)

5. **Actor Loop Structure**: Should we introduce structured lifecycle?
   - Option A: Current (reactive) - respond to orderbook deltas
   - Option B: Structured loop - `calibrate → trade/close/cleanup → calibrate`
   - **Decision:** Start with Option A, consider Option B if we need more control

---

## References

- [Kalshi Order Groups API - Create](https://docs.kalshi.com/api-reference/order-groups/create-order-group)
- [Kalshi Order Groups API - Get](https://docs.kalshi.com/api-reference/order-groups/get-order-group)
- [Kalshi Order Groups API - Get All](https://docs.kalshi.com/api-reference/order-groups/get-order-groups)
- [Kalshi Order Groups API - Reset](https://docs.kalshi.com/api-reference/order-groups/reset-order-group)
- [Kalshi Order Groups API - Delete](https://docs.kalshi.com/api-reference/order-groups/delete-order-group)
- [Current Actor Loop Documentation](./orderbook_delta_flow.md)
- [Startup Flow Documentation](./STARTUP_FLOW_MAPPING.md)
- [Initialization Tracker Implementation](../src/kalshiflow_rl/trading/initialization_tracker.py)

