# Actor Lifecycle Analysis & Improvement Plan

## Current Actor Loop: Initialization to Trading

### Phase 1: System Initialization (`app.py:lifespan`)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Infrastructure Setup                                      │
├─────────────────────────────────────────────────────────────┤
│ • Database (PostgreSQL) initialization                      │
│ • WriteQueue (background DB writes)                          │
│ • EventBus (internal event routing)                         │
│ • Market selection (discovery or config)                     │
│ • OrderbookClient (WebSocket to Kalshi)                     │
│ • StatsCollector (metrics tracking)                         │
│ • WebSocketManager (frontend connections)                   │
└─────────────────────────────────────────────────────────────┘
```

### Phase 2: Trading Components Initialization

```
┌─────────────────────────────────────────────────────────────┐
│ 2. OrderManager Initialization                              │
│    (KalshiMultiMarketOrderManager.initialize())             │
├─────────────────────────────────────────────────────────────┤
│ • Trading client connection (KalshiDemoTradingClient)       │
│ • Fill processor task start (_process_fills queue)         │
│ • FillListener WebSocket (real-time fill notifications)     │
│ • PositionListener WebSocket (real-time position updates)    │
│ • Cleanup (optional - cancels open orders)                  │
│ • Order sync (syncs orders from Kalshi API)                 │
│ • Position sync (syncs positions + cash balance)            │
│ • Settlement sync (fetches past 24h settlements)             │
│ • Periodic sync task start (every 30s)                      │
│ • Session start values capture (initial cash/portfolio)     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 3. ActorService Initialization                              │
├─────────────────────────────────────────────────────────────┤
│ • Create ActorService with market tickers, model path      │
│ • Set action selector (hardcoded or RL model)              │
│ • Set observation adapter (LiveObservationAdapter)           │
│ • Set order manager (KalshiMultiMarketOrderManager)        │
│ • Set websocket manager                                     │
│ • actor_service.initialize():                              │
│   - Validate dependencies                                    │
│   - Load RL model (if needed)                               │
│   - Subscribe to EventBus (orderbook snapshots/deltas)      │
│   - Start event processing loop (_event_processing_loop)    │
└─────────────────────────────────────────────────────────────┘
```

### Phase 3: Trading Loop (Continuous)

```
┌─────────────────────────────────────────────────────────────┐
│ 4. Event Processing Pipeline                                │
│    (ActorService._process_market_update)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Step 1: Build Observation                          │   │
│  │ • Get orderbook snapshot from SharedOrderbookState │   │
│  │ • LiveObservationAdapter.build_observation()       │   │
│  │ • Include portfolio features (cash, positions)      │   │
│  │ • Returns: 52-feature numpy array                  │   │
│  └────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Step 2: Select Action                               │   │
│  │ • Check cash reserve threshold (force HOLD if low)  │   │
│  │ • ActionSelector.select_action(observation)        │   │
│  │ • RLModelSelector: model.predict() (deterministic)  │   │
│  │ • Returns: action 0-4 (HOLD or trading action)       │   │
│  └────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Step 3: Execute Action                             │   │
│  │ • If HOLD: return immediately                      │   │
│  │ • Check throttling (250ms per market)              │   │
│  │ • Get orderbook snapshot                           │   │
│  │ • OrderManager.execute_limit_order_action()        │   │
│  │   - Calculate limit price (aggressive/passive/mid) │   │
│  │   - Check cash availability (for BUY orders)       │   │
│  │   - Place order via Kalshi API                     │   │
│  │ • Update throttle timestamp                        │   │
│  └────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Step 4: Update Positions                           │   │
│  │ • Wait 100ms (position_read_delay_ms)              │   │
│  │ • OrderManager.get_positions()                     │   │
│  │ • OrderManager.get_portfolio_value()               │   │
│  │ • Log position changes                             │   │
│  │ • Broadcast to WebSocket (if configured)            │   │
│  └────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 5. Parallel Fill Processing                                 │
│    (KalshiMultiMarketOrderManager._process_fills)           │
├─────────────────────────────────────────────────────────────┤
│ • FillListener receives fill notifications via WebSocket   │
│ • Queues fill events to fills_queue                        │
│ • _process_fills() processes fills asynchronously           │
│ • Updates positions immediately on fill                    │
│ • Updates cash balance (calculated, not synced)             │
│ • Tracks cashflow (invested vs recouped)                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 6. Periodic Sync (Every 30s)                                │
│    (KalshiMultiMarketOrderManager._periodic_sync)           │
├─────────────────────────────────────────────────────────────┤
│ • Sync orders with Kalshi API                              │
│ • Sync positions with Kalshi API                           │
│ • Sync cash balance (authoritative from Kalshi)             │
│ • Sync portfolio value                                     │
│ • Calculate drift (calculated vs synced)                   │
│ • Sync settlements (past 24h)                               │
└─────────────────────────────────────────────────────────────┘
```

## Critical Gaps & Issues

### 1. **No Position Closing/Recouping Strategy** ⚠️ CRITICAL

**Current State:**
- Trader opens positions using cash balance
- Positions remain open until:
  - Event ends (automatic settlement by Kalshi)
  - Manual intervention (user adds funds to close positions)
- No automatic position management or closing logic

**Problem:**
- Once cash is exhausted, trader cannot:
  - Close existing positions to free up cash
  - Recoup invested capital
  - Continue trading in new markets
- Positions are "stuck" until event settlement

**Impact:**
- Trader becomes non-functional after initial cash depletion
- No way to recover from poor position sizing
- Cannot adapt to changing market conditions

### 2. **No Environment Calibration on Startup** ⚠️ HIGH

**Current State:**
- Startup syncs orders/positions from Kalshi
- No active calibration of:
  - Market availability/viability
  - Current portfolio state vs expected state
  - Cash availability vs required reserves
  - Active markets vs configured markets

**Missing:**
- **State Reconciliation**: Verify we're in sync with Kalshi
- **Market Viability Check**: Are configured markets still active?
- **Portfolio Health Check**: Are positions healthy? Any stuck positions?
- **Cash Flow Analysis**: Can we continue trading? Do we need to close positions?

### 3. **No Continuous Recalibration** ⚠️ HIGH

**Current State:**
- Periodic sync every 30s (passive)
- No active monitoring of:
  - Position health (unrealized P&L, time in position)
  - Market state changes (market closing, event ending)
  - Cash flow constraints (approaching reserve threshold)
  - Trading opportunities (new markets, better prices)

**Missing:**
- **Position Health Monitor**: Track positions that should be closed
- **Market State Monitor**: Detect when markets are closing/ending
- **Cash Flow Monitor**: Proactively manage cash to avoid reserve threshold
- **Opportunity Monitor**: Identify when to close positions for better opportunities

### 4. **No Self-Healing/Recovery Mechanisms** ⚠️ MEDIUM

**Current State:**
- Circuit breakers disable markets on errors
- No recovery strategies:
  - Position stuck? No automatic closing attempt
  - Cash exhausted? No position liquidation
  - Market closed? No cleanup of related orders/positions
  - Sync drift? No automatic correction

**Missing:**
- **Position Recovery**: Automatically close stuck positions
- **Cash Recovery**: Liquidate positions when cash is low
- **State Recovery**: Correct drift between calculated and synced values
- **Market Recovery**: Clean up when markets close

### 5. **No Order Group Management** ⚠️ MEDIUM

**Current State:**
- Orders placed individually
- No grouping or coordination
- No automatic cancellation when limits hit

**Missing:**
- **Order Groups**: Use Kalshi order groups for coordinated order management
- **Group Limits**: Set contract limits per group to prevent over-trading
- **Automatic Cancellation**: Cancel all orders in group when limit hit

## Order Groups Investigation

### What Are Order Groups?

From Kalshi API docs: Order groups allow you to:
- **Set a contracts limit** for a group of orders
- **Automatically cancel all orders** in the group when the limit is hit
- **Prevent new orders** from being placed until the group is reset

### API Endpoints

```
POST /portfolio/order_groups/create
{
  "contracts_limit": 2  // Max contracts that can be matched in this group
}

Response:
{
  "order_group_id": "<string>"
}
```

### Potential Use Cases

1. **Per-Market Position Limits**
   - Create order group per market
   - Set limit = max position size for that market
   - Automatically prevents over-trading in a single market

2. **Total Portfolio Limits**
   - Create single order group for all markets
   - Set limit = max total contracts across portfolio
   - Automatically prevents over-leveraging

3. **Session Limits**
   - Create order group per trading session
   - Set limit = max contracts for session
   - Automatically stops trading when limit hit

4. **Risk Management**
   - Create order groups for different risk levels
   - Set limits based on risk tolerance
   - Automatically enforces risk limits

### Benefits for Our System

1. **Simplified Position Management**
   - No need to manually track and enforce position limits
   - Kalshi automatically cancels orders when limit hit
   - Prevents over-trading without complex logic

2. **Automatic Risk Control**
   - Set limits at group creation
   - No need to check limits on every order
   - Fail-safe mechanism if our logic has bugs

3. **Clean State Management**
   - When limit hit, all orders cancelled automatically
   - Clear state: no orders, can reset and continue
   - Easier to reason about system state

4. **Recovery Mechanism**
   - If we get into bad state, reset order group
   - Start fresh without manual cleanup
   - Can be part of self-healing strategy

### Implementation Strategy

1. **Create Order Group on Startup**
   ```python
   # In KalshiMultiMarketOrderManager.initialize()
   order_group = await trading_client.create_order_group(
       contracts_limit=config.RL_MAX_TOTAL_CONTRACTS
   )
   self.order_group_id = order_group["order_group_id"]
   ```

2. **Include Order Group in All Orders**
   ```python
   # In place_order()
   order_data["order_group_id"] = self.order_group_id
   ```

3. **Monitor Group State**
   ```python
   # Check if group limit hit
   group_info = await trading_client.get_order_group(self.order_group_id)
   if group_info["contracts_matched"] >= group_info["contracts_limit"]:
       # All orders cancelled, reset group
       await trading_client.reset_order_group(self.order_group_id)
   ```

4. **Reset on Recovery**
   ```python
   # In cleanup/recovery logic
   if self.order_group_id:
       await trading_client.reset_order_group(self.order_group_id)
   ```

## Recommended Improvements

### Priority 1: Position Management & Cash Recovery

**Goal:** Enable trader to close positions and recoup cash

**Implementation:**

1. **Position Health Monitor**
   ```python
   class PositionHealthMonitor:
       async def check_positions(self):
           for ticker, position in self.positions.items():
               # Check if position should be closed
               if self._should_close_position(position):
                   await self._close_position(ticker, position)
   
       def _should_close_position(self, position):
           # Close if:
           # - Unrealized P&L > threshold (take profit)
           # - Unrealized P&L < -threshold (stop loss)
           # - Time in position > max_hold_time
           # - Market closing soon
           return (
               position.unrealized_pnl > self.take_profit_threshold or
               position.unrealized_pnl < -self.stop_loss_threshold or
               position.time_in_position > self.max_hold_time or
               self._market_closing_soon(position.ticker)
           )
   ```

2. **Cash Recovery Strategy**
   ```python
   async def recover_cash(self):
       # If cash < reserve threshold, close positions
       if self.cash_balance < self.min_cash_reserve:
           # Close positions with worst P&L first
           positions_by_pnl = sorted(
               self.positions.items(),
               key=lambda x: x[1].unrealized_pnl
           )
           for ticker, position in positions_by_pnl:
               if self.cash_balance >= self.min_cash_reserve:
                   break
               await self._close_position(ticker, position)
   ```

3. **Market State Monitor**
   ```python
   async def monitor_market_states(self):
       # Check if markets are closing/ending
       for ticker in self.market_tickers:
           market_info = await self.trading_client.get_market(ticker)
           if market_info["status"] == "closed":
               # Close all positions in this market
               await self._close_all_positions_in_market(ticker)
   ```

### Priority 2: Environment Calibration

**Goal:** Calibrate to environment on startup and continuously

**Implementation:**

1. **Startup Calibration**
   ```python
   async def calibrate_environment(self):
       # 1. Verify Kalshi connection
       await self._verify_kalshi_connection()
       
       # 2. Sync and verify state
       await self._sync_and_verify_state()
       
       # 3. Check market viability
       viable_markets = await self._check_market_viability()
       
       # 4. Assess portfolio health
       portfolio_health = await self._assess_portfolio_health()
       
       # 5. Determine trading readiness
       if not self._is_ready_to_trade(portfolio_health):
           await self._recover_to_ready_state()
   ```

2. **Continuous Recalibration**
   ```python
   async def recalibrate_periodically(self):
       while self.is_active:
           await asyncio.sleep(60)  # Every minute
           
           # Check for drift
           drift = await self._check_state_drift()
           if drift > threshold:
               await self._correct_drift()
           
           # Check market states
           await self._monitor_market_states()
           
           # Check position health
           await self._monitor_position_health()
           
           # Check cash flow
           await self._monitor_cash_flow()
   ```

### Priority 3: Order Group Integration

**Goal:** Use order groups for simplified position management

**Implementation:**

1. **Order Group Manager**
   ```python
   class OrderGroupManager:
       async def initialize(self):
           # Create order group with total contract limit
           group = await self.trading_client.create_order_group(
               contracts_limit=config.RL_MAX_TOTAL_CONTRACTS
           )
           self.order_group_id = group["order_group_id"]
       
       async def check_group_state(self):
           # Check if limit hit
           group_info = await self.trading_client.get_order_group(
               self.order_group_id
           )
           if group_info["contracts_matched"] >= group_info["contracts_limit"]:
               # Reset group to continue trading
               await self.trading_client.reset_order_group(
                   self.order_group_id
               )
   ```

2. **Include in Order Placement**
   ```python
   # In execute_limit_order_action()
   order_data["order_group_id"] = self.order_group_id
   ```

### Priority 4: Self-Healing Mechanisms

**Goal:** Automatically recover from bad states

**Implementation:**

1. **State Recovery**
   ```python
   async def recover_state(self):
       # If drift too large, force sync
       if abs(self._last_sync_drift_cash) > self.max_drift:
           await self._force_sync_with_kalshi()
       
       # If positions out of sync, reconcile
       if self._positions_out_of_sync():
           await self._reconcile_positions()
   ```

2. **Position Recovery**
   ```python
   async def recover_positions(self):
       # Close stuck positions (no fills for too long)
       for ticker, position in self.positions.items():
           if self._is_position_stuck(position):
               await self._force_close_position(ticker)
   ```

## Implementation Plan

### Phase 1: Critical Fixes (Week 1)
1. ✅ Implement position closing logic
2. ✅ Add cash recovery strategy
3. ✅ Add market state monitoring
4. ✅ Test position closing in paper trading

### Phase 2: Calibration (Week 2)
1. ✅ Implement startup calibration
2. ✅ Add continuous recalibration loop
3. ✅ Add state drift detection
4. ✅ Test calibration in various states

### Phase 3: Order Groups (Week 3)
1. ✅ Research order group API fully
2. ✅ Implement order group manager
3. ✅ Integrate with order placement
4. ✅ Test order group limits

### Phase 4: Self-Healing (Week 4)
1. ✅ Implement state recovery
2. ✅ Add position recovery
3. ✅ Add comprehensive error handling
4. ✅ Test recovery scenarios

## Summary

**Current State:**
- Trader initializes and starts trading
- Opens positions until cash exhausted
- No way to close positions or recoup cash
- No calibration or self-healing

**Key Improvements Needed:**
1. **Position Management**: Close positions to free cash
2. **Environment Calibration**: Sync with environment on startup and continuously
3. **Order Groups**: Use for simplified position limits
4. **Self-Healing**: Automatically recover from bad states

**Inspired by Video Game Bots:**
- **Calibration**: Check environment state on startup (like checking game state)
- **Recalibration**: Continuously verify we're in sync (like checking if we're still in the right place)
- **Recovery**: Automatically fix when we go off track (like pathfinding back to correct location)
- **State Management**: Always know where we are and what we should do next (like maintaining game state)

The key insight: **The trader should always know its state relative to the environment (Kalshi + DB) and be able to recalibrate when it drifts.**

