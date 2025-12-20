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

TRADING LOOP (Reactive - Continuous)
└─ For each orderbook delta:
   ├─ Build observation (orderbook + portfolio)
   ├─ Select action (RL model or hardcoded)
   ├─ Execute action (place order if not HOLD/throttled)
   └─ Update positions (read after 100ms delay)

RECALIBRATION LOOP (Proactive - Every 60s)
└─ Periodic sync and recalibration:
   ├─ Sync state with Kalshi (positions, orders, cash)
   ├─ Monitor position health (P&L thresholds, max hold time)
   ├─ Monitor market states (closing markets)
   ├─ Close positions that meet criteria (take profit, stop loss, etc.)
   └─ Recover cash if needed (close worst-performing positions)

PARALLEL PROCESSES
├─ FillListener: Real-time fill notifications → update positions
├─ PositionListener: Real-time position updates
└─ Periodic Sync + Recalibration: Every 60s, sync with Kalshi API and monitor positions
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

**Architecture:**
- **Position closing is part of the RECALIBRATION LOOP**, not the trading loop
- **Trading Loop** (Reactive): Responds to orderbook deltas → Opens positions
- **Recalibration Loop** (Proactive): Runs every 60s → Closes positions, syncs state, manages risk
- This separation ensures:
  - Trading decisions remain fast and reactive
  - Risk management is proactive and systematic
  - No conflicts between opening and closing positions

**Order Groups Impact:**
- **Position closing can be implemented independently** - it's just placing opposite orders
- Order groups are about limiting total contracts matched, not about closing positions
- Order groups could ensure we don't exceed limits when closing positions (complementary, not blocking)
- **Recommendation:** Implement position closing first, then enhance with order groups if needed

**Implementation Status:** ✅ COMPLETED
- `close_position()` method implemented in `KalshiMultiMarketOrderManager`
- Position health monitoring (`_monitor_position_health()`) checks P&L thresholds and max hold time
- Cash recovery (`_recover_cash_by_closing_positions()`) closes worst-performing positions when cash is low
- Market state monitoring (`_monitor_market_states()`) closes positions in markets that are closing
- Recalibration loop integrated into `_start_periodic_sync()` (runs every 60s)
- WebSocket broadcasts include closing reason for real-time UI visibility

**WebSocket Message Format:**

When a position is closed, the `trader_action` message includes:
```json
{
  "type": "trader_action",
  "data": {
    "timestamp": 1702334567.89,
    "market_ticker": "INXD-25JAN03",
    "action": {
      "action_id": 2,
      "action_name": "SELL_YES_LIMIT",
      "position_size": 10,
      "quantity": 10,
      "limit_price": 55,
      "reason": "close_position:take_profit"  // Indicates closing action
    },
    "execution_result": {
      "executed": true,
      "order_id": "order_124",
      "status": "placed",
      "reason": "close_position:take_profit"  // Closing reason
    }
  }
}
```

When a position is being closed, the `position_update` message includes:
```json
{
  "type": "position_update",
  "data": {
    "ticker": "INXD-25JAN03",
    "position": 10,
    "closing_reason": "take_profit",  // Reason for closing
    "changed_fields": ["position"],
    "previous_values": {"position": 10},
    "timestamp": 1702334567.89
  }
}
```

**Closing Reasons:**
- `take_profit`: Position hit take profit threshold (default: +20%)
- `stop_loss`: Position hit stop loss threshold (default: -10%)
- `cash_recovery`: Position closed to recover cash (when balance < reserve)
- `market_closing`: Market is closing/ending soon
- `max_hold_time`: Position exceeded max hold time (default: 1 hour)

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

### Phase 1: Position Management (Week 1) - ✅ COMPLETED

**Goal:** Enable trader to close positions and recoup cash

**Status:** ✅ **COMPLETED** - All tasks delivered and tested

**Decision:** Implement position closing **before** order groups. They're independent - position closing is about placing opposite orders, order groups are about limiting total contracts. We can enhance position closing with order groups later if needed.

**Completed Tasks:**
1. ✅ Add `close_position()` method to `KalshiMultiMarketOrderManager`
2. ✅ Add position health monitoring (P&L, time in position)
3. ✅ Add cash recovery strategy (close positions when cash low)
4. ✅ Add market state monitoring (detect closing markets)
5. ✅ Integrate position closing into recalibration loop (every 60s)
6. ✅ Add trader status tracking with detailed closing breakdown
7. ✅ Add WebSocket visibility for position closing actions
8. ✅ Add click-to-copy functionality for status history log

**Delivered:**
- Position closing based on P&L thresholds (take profit: +20%, stop loss: -10%)
- Position closing based on max hold time (default: 1 hour)
- Cash recovery when balance falls below reserve threshold
- Market closing detection (closes positions before markets end)
- Real-time status updates via WebSocket with detailed closing reasons
- Trader status footer in UI showing current status and history log
- Copy-to-clipboard functionality for status history

**Files Modified:**
- `backend/src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py` - Position closing logic
- `backend/src/kalshiflow_rl/trading/actor_service.py` - Status updates
- `backend/src/kalshiflow_rl/websocket_manager.py` - Status broadcasting
- `frontend/src/components/TraderStatePanel.jsx` - Status display and copy feature
- `frontend/src/components/RLTraderDashboard.jsx` - Status message handling

**Note:** Order groups won't require changes to position closing logic - they're complementary features.

### Phase 2: Core Actor Loop Improvements (M2) - HIGH

**Goal:** Improve the core actor/trader loop for better decision-making, coordination, and reliability

**Focus:** The core trading loop that processes orderbook events and makes trading decisions. This is the heart of the trader - where observations are built, actions are selected, and orders are executed.

**Current Actor Loop (from `ActorService._process_market_update`):**

The actor loop is a 4-step pipeline that processes each orderbook delta:

```
1. build_observation → Convert orderbook state to model input (52-feature array)
2. select_action → Use RL model or hardcoded selector to choose action (0-4)
3. execute_action → Place order via OrderManager (if not HOLD/throttled)
4. update_positions → Track portfolio state after execution
```

**Current Architecture:**
- **Event-driven:** OrderbookClient → EventBus → ActorService queue
- **Serial processing:** Single queue for all markets (prevents race conditions)
- **4-step pipeline:** Each event goes through observation → action → execution → update
- **Parallel loops:** Trading loop (reactive) and recalibration loop (proactive every 60s) run independently
- **State sharing:** Both loops share same state (positions, cash, orders) via OrderManager

**M2 Focus Areas:**

1. **Actor Loop Coordination**
   - Ensure trading loop and recalibration loop work together properly
   - Prevent conflicts when both loops access shared state
   - Coordinate position opening (trading loop) with position closing (recalibration loop)
   - **Status:** Deferred from M1, needs implementation

2. **State Management**
   - Improve how actor loop accesses and updates trader state
   - Ensure state consistency between trading and recalibration loops
   - Better handling of state snapshots during recalibration

3. **Decision Making**
   - Enhance action selection logic (when to trade, when to hold)
   - Smarter market condition checks before action selection
   - Better integration with cash reserve thresholds
   - Consider position limits and market viability

4. **Error Handling**
   - Better recovery from failures in the actor loop
   - Graceful degradation when components fail
   - Circuit breaker improvements

5. **Performance**
   - Optimize the 4-step pipeline
   - Reduce latency in observation building
   - Improve action selection speed

**What to Defer:**
- ❌ Hierarchical status display with sub-steps (thinking mode) → **Defer to post-MVP**
- ⚠️ Enhanced calibration checks (markets viable, portfolio health) → Can be added incrementally
- ⚠️ "Last portfolio data sync X" timestamp → Low priority, can add later

**Key Files:**
- `backend/src/kalshiflow_rl/trading/actor_service.py` - Core actor loop implementation
- `backend/src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py` - Order execution and state management
- `backend/src/kalshiflow_rl/trading/action_selector.py` - Action selection logic
- `backend/src/kalshiflow_rl/environments/feature_extractors.py` - Observation building

**Integration Points:**
- ActorService uses OrderManager for execution (`execute_limit_order_action`)
- Recalibration loop in OrderManager runs in parallel (every 60s)
- Both loops share same state (positions, cash, orders) - potential coordination needed
- EventBus routes orderbook deltas to ActorService queue

---

## Post-M2 Trader Loop Architecture

### Trader States (Mutually Exclusive)

The trader operates in one state at a time, with clear transitions and full visibility in `trader_status`:

1. **`initializing`** - Startup state
   - Syncing initial state with Kalshi (balance, positions, orders)
   - Loading RL model
   - Setting up infrastructure
   - **Transition to:** `trading` when ready

2. **`trading`** - Active trading state
   - Processing orderbook deltas through 4-step pipeline
   - Making trading decisions (opening positions)
   - **Can transition to:** `calibrating` (periodic), `paused` (errors), `stopping` (shutdown)
   - **Status details:** Shows last processed event, current queue depth, action counts

3. **`calibrating`** - Recalibration state
   - Synchronizing state with Kalshi
   - Monitoring position health
   - Closing positions that meet criteria
   - Monitoring market states
   - Recovering cash if needed
   - **Sub-states tracked in status:**
     - `calibrating -> syncing state` - Syncing with Kalshi API
     - `calibrating -> closing positions` - Closing positions based on health
     - `calibrating -> monitoring markets` - Checking market states
     - `calibrating -> cash recovery` - Recovering cash if needed
   - **Transition to:** `trading` when complete (or `paused` if errors)
   - **Duration:** Target < 5 seconds for brute-force halt approach

4. **`paused`** - Error/Recovery state
   - Trader encountered an error that requires attention
   - Trading loop halted
   - Can be manually resumed or auto-resume after recovery
   - **Transition to:** `trading` (resume) or `stopping` (shutdown)

5. **`stopping`** - Shutdown state
   - Graceful shutdown in progress
   - Closing remaining positions (optional)
   - Saving state
   - **Transition to:** None (terminal state)

### Available Actions

The trader has two types of actions:

**Trading Actions (Actor Loop - 5-action space):**
- **0: HOLD** - No action, wait for next event
- **1: BUY_YES_LIMIT** - Place buy limit order for YES contracts (5 contracts)
- **2: SELL_YES_LIMIT** - Place sell limit order for YES contracts (5 contracts)
- **3: BUY_NO_LIMIT** - Place buy limit order for NO contracts (5 contracts)
- **4: SELL_NO_LIMIT** - Place sell limit order for NO contracts (5 contracts)

**Position Management Actions (Recalibration Loop):**
- **`close_position(reason)`** - Close a position by placing opposite order
  - Reasons: `take_profit`, `stop_loss`, `cash_recovery`, `market_closing`, `max_hold_time`
  - Automatically determines correct opposite action (SELL YES if long YES, etc.)

### State Machine Flow

```
STARTUP
  ↓
[initializing] ──────────────────┐
  ↓                              │
[trading] ←──────────────────────┼── (every 60s)
  ↓      │                       │
  │      │ (error)               │
  │      ↓                       │
  │  [paused] ─── (resume) ────┘
  │      │
  │      │ (shutdown)
  │      ↓
  └──→ [stopping]
```

**Key Behaviors:**
- **Mutually Exclusive States:** Only one state active at a time
- **Trading Halted During Calibration:** Trading loop paused during `calibrating` state
- **State Visibility:** Current state always visible in `trader_status.current_status`
- **Status History:** All state changes logged sequentially with timestamps, results, and time_in_status
- **Simplified Logging:** State changes are clear from sequential log entries (no separate transition entries)

### Actor Loop (Trading State)

When in `trading` state, the loop processes events through 4-step pipeline:

```
Orderbook Delta Event
  ↓
1. build_observation()
   - Get orderbook snapshot
   - Extract 52 features
   - Include portfolio state (cash, positions)
  ↓
2. select_action()
   - Check market viability
   - Check cash reserves
   - Check position limits
   - Use RL model or hardcoded selector
   - Returns: 0-4 (HOLD or trading action)
  ↓
3. execute_action()
   - Validate action (cash, limits, throttling)
   - Place order via OrderManager
   - Track execution result
  ↓
4. update_positions()
   - Update portfolio state tracking
   - Log metrics
  ↓
Next Event
```

**Pre-Action Checks (Enhanced in M2):**
- Market viability (market still active/open)
- Cash reserve threshold (force HOLD if below minimum)
- Position limits (prevent over-leveraging)
- Throttling (respect rate limits per market)
- Circuit breakers (skip disabled markets)

### Recalibration Loop (Calibrating State)

When transitioning to `calibrating` state:

```
Every 60s (configurable)
  ↓
[State: calibrating]
  ↓
1. calibrating -> syncing state
   - Sync orders with Kalshi
   - Sync positions with Kalshi
   - Sync cash balance
   - Track state changes
  ↓
2. calibrating -> closing positions
   - Monitor position health (P&L, time)
   - Close positions meeting criteria
   - Track closing results
  ↓
3. calibrating -> monitoring markets
   - Check market states
   - Close positions in closing markets
  ↓
4. calibrating -> cash recovery (if needed)
   - Check cash balance vs reserve
   - Close worst-performing positions if low
  ↓
[State: trading] (resume trading loop)
```

**Performance Targets:**
- Total calibration duration: **< 5 seconds** (enables brute-force halt)
- Individual steps: **< 2 seconds each** (where possible)
- Atomic operations: Each step is self-contained and recoverable

---

## Event Handling During Calibration

### Problem Statement

When the trader enters `calibrating` state, the trading loop is halted. During this time:
- Orderbook deltas continue arriving via WebSocket
- Events queue up in `ActorService._event_queue`
- If calibration takes too long, queue can overflow or events become stale
- Lost events mean missed trading opportunities

**Current Situation:**
- Calibration runs every 60s
- Target calibration duration: < 5 seconds
- Queue size: 1000 events (configurable)
- Average event rate: Variable (depends on market activity)

### Strategy Options

#### Option 1: Brute-Force Halt (M2 Approach)

**Description:** Simply halt trading loop during calibration, accept event loss if calibration < 5s.

**Implementation:**
- Trading loop checks state before processing events
- If state == `calibrating`, skip event processing
- Queue up to capacity (1000 events)
- Resume processing when state returns to `trading`

**Pros:**
- ✅ Simple to implement
- ✅ Minimal coordination complexity
- ✅ Acceptable if calibration < 5s (at 10 events/sec, ~50 events lost max)
- ✅ No queue manipulation needed

**Cons:**
- ❌ Event loss during calibration
- ❌ Stale events may be processed after calibration
- ❌ Queue overflow if calibration takes longer than expected

**Target:** Keep calibration under 5 seconds to make this acceptable for M2.

#### Option 2: Pause and Dequeue (Future Enhancement)

**Description:** Pause trading loop, clear processed events, resume with fresh events.

**Implementation:**
- When entering `calibrating`:
  - Stop dequeuing new events
  - Clear already-processed events from queue
  - Keep unprocessed events (with timestamp checks)
- When exiting `calibrating`:
  - Resume dequeuing
  - Process events with freshness checks (skip stale events)

**Pros:**
- ✅ No processed event waste
- ✅ Fresh events processed immediately after calibration
- ✅ Better handling of event freshness

**Cons:**
- ⚠️ More complex queue management
- ⚠️ Need to track event freshness/validity
- ⚠️ May still have some event loss

**When to Use:** If calibration takes > 5 seconds regularly, or if event freshness is critical.

#### Option 3: Atomic Calibration with Event Buffering (Long-Term)

**Description:** Make calibration truly atomic and non-blocking, process events during calibration.

**Implementation:**
- Calibration runs as atomic operations with state locks
- Trading loop continues processing events
- State updates locked during calibration sync points
- Calibration uses snapshots for consistency

**Pros:**
- ✅ Zero event loss
- ✅ Continuous trading
- ✅ Seamless operation

**Cons:**
- ⚠️ Complex state synchronization
- ⚠️ Requires careful locking mechanisms
- ⚠️ Potential race conditions
- ⚠️ Significant architectural changes

**When to Use:** Post-MVP when we need zero-downtime trading.

#### Option 4: Incremental Calibration (Hybrid)

**Description:** Break calibration into smaller, faster chunks that don't require full halt.

**Implementation:**
- Sync state: < 1s (can run alongside trading)
- Monitor positions: < 1s (read-only, non-blocking)
- Close positions: Queue closing actions (non-blocking)
- Only halt for critical operations (if any)

**Pros:**
- ✅ Minimal trading disruption
- ✅ Faster perceived calibration
- ✅ Better user experience

**Cons:**
- ⚠️ More complex coordination
- ⚠️ Some operations may still need coordination

**When to Use:** If we can make calibration operations truly non-blocking.

### Recommended Approach for M2

**Primary Strategy: Option 1 (Brute-Force Halt)**

For M2, we'll use brute-force halt with strict performance targets:
- **Calibration duration target: < 5 seconds**
- **Monitoring:** Track calibration duration in metrics
- **Alerts:** Warn if calibration exceeds 5 seconds
- **Optimization focus:** Make calibration operations atomic and fast

**Implementation Checklist:**
- [ ] Add state check in `_process_market_update()` - skip if `calibrating`
- [ ] Add calibration duration tracking
- [ ] Add metrics/alerts for slow calibration
- [ ] Optimize calibration operations (see below)

**Performance Optimization for Calibration:**

1. **Parallel Operations:**
   - Sync orders, positions, cash in parallel (if possible)
   - Batch API calls where supported

2. **Caching:**
   - Cache market info between calibrations
   - Only fetch changed data

3. **Incremental Processing:**
   - Only check positions that changed since last calibration
   - Skip markets with no positions

4. **Timeout Protection:**
   - Set timeouts on all API calls
   - Abort calibration if any step exceeds threshold

### Future Enhancement Path

**Short-Term (Post-M2):**
- Implement Option 2 (Pause and Dequeue) if calibration consistently > 5s
- Add event freshness checks (skip events older than 10s)

**Long-Term (Post-MVP):**
- Implement Option 3 (Atomic Calibration) for zero-downtime trading
- Full state synchronization with minimal locks

**Metrics to Track:**
- Calibration duration (target: < 5s)
- Events queued during calibration
- Events lost/stale events processed
- Queue overflow frequency

---

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

## M2 Accelerators & Context

### Key Files to Review

**Core Actor Loop:**
- `backend/src/kalshiflow_rl/trading/actor_service.py` - Main actor service with 4-step pipeline
  - `_process_market_update()` - Core loop implementation (line ~532)
  - `_build_observation()` - Converts orderbook to model input
  - `_select_action()` - Action selection logic
  - `_safe_execute_action()` - Order execution
  - `_update_positions()` - Portfolio state tracking

**Order Execution:**
- `backend/src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py` - Order manager
  - `execute_limit_order_action()` - Executes actions from actor loop
  - `_start_periodic_sync()` - Recalibration loop (line ~3677)
  - Position closing methods (from M1)

**Action Selection:**
- `backend/src/kalshiflow_rl/trading/action_selector.py` - Action selection strategies
  - `RLModelSelector` - Uses trained PPO model
  - `HardcodedSelector` - Simple hardcoded strategy

**Observation Building:**
- `backend/src/kalshiflow_rl/environments/feature_extractors.py` - Feature extraction
  - `LiveObservationAdapter` - Converts orderbook to 52-feature array

### Current Architecture

**Event Flow:**
```
OrderbookClient (WebSocket) 
  → EventBus (internal routing)
    → ActorService._event_queue (async queue)
      → _event_processing_loop() (serial processing)
        → _process_market_update() (4-step pipeline)
```

**Parallel Processes:**
- **Trading Loop:** Reactive, processes orderbook deltas as they arrive
- **Recalibration Loop:** Proactive, runs every 60s in `_start_periodic_sync()`
- **FillListener:** Real-time fill notifications via WebSocket
- **PositionListener:** Real-time position updates via WebSocket

**State Management:**
- OrderManager maintains authoritative state (cash, positions, orders)
- ActorService reads state from OrderManager (no local copies)
- Both loops access same OrderManager instance
- State synced with Kalshi API periodically (every 60s)

### Known Issues & Gaps

1. **Trading Loop & Recalibration Loop Coordination**
   - **Issue:** Both loops can access shared state simultaneously
   - **Risk:** Trading loop might open positions while recalibration loop is closing them
   - **Status:** Deferred from M1, needs coordination mechanism
   - **Potential Solution:** Add lock/flag to prevent position opening during active recalibration

2. **Action Selection Logic**
   - **Issue:** Action selection doesn't consider market conditions deeply
   - **Gap:** No market viability checks before action selection
   - **Opportunity:** Add market state awareness to action selector

3. **Error Recovery**
   - **Issue:** Actor loop errors can cause queue backup
   - **Gap:** Limited recovery mechanisms for failed observations/actions
   - **Opportunity:** Better error handling and circuit breakers

4. **Performance Optimization**
   - **Issue:** Observation building happens on every event
   - **Gap:** Could cache observations or optimize feature extraction
   - **Opportunity:** Profile and optimize hot paths

### Integration Points

**ActorService ↔ OrderManager:**
- ActorService calls `order_manager.execute_limit_order_action()`
- OrderManager handles all Kalshi API interactions
- OrderManager maintains cash, positions, orders state

**ActorService ↔ Recalibration Loop:**
- Both run in parallel (no direct communication)
- Share same OrderManager instance
- Recalibration loop can close positions while trading loop opens new ones

**EventBus ↔ ActorService:**
- EventBus routes orderbook deltas to ActorService queue
- Non-blocking: EventBus doesn't wait for processing
- Queue-based: Events queued if ActorService is busy

## Next Steps

1. ✅ **Phase 1 (M1) Complete** - Position management delivered
2. ✅ **Phase 2 (M2) Status Logging Complete** - Simplified status logging, removed transition entries
3. **Continue Phase 2 (M2)** - Core actor loop improvements
4. **Focus Areas:**
   - Actor loop coordination (trading ↔ recalibration)
   - Enhanced decision making
   - Error recovery improvements
5. **Test incrementally** in paper trading
6. **Iterate** based on results

## Recent Changes (Status Logging Simplification)

**Completed:** Simplified trader status logging by removing transition entry complexity
- Removed separate transition entries (e.g., "trading -> calibrating") from status history
- Simplified `_update_trader_status` to use single code path for all status updates
- State changes are now clear from sequential log entries without cluttering the history
- Maintained all essential tracking: `previous_state`, `time_in_status`, trading stats reset
- Status history shows clean sequential entries: `trading` → `calibrating` → `calibrating -> syncing state` → `trading`

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
- [M2 Trader Status Console Example](./M2_TRADER_STATUS_EXAMPLE.md) - Example output of trader status at end of M2

