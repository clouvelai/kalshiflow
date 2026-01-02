# Current RL Trader Startup Flow Mapping

This document maps the current startup sequence when running `scripts/run-rl-trader.sh` with `RL_ACTOR_ENABLED=true`.

## Startup Sequence (Current State)

### Phase 1: Core Infrastructure (app.py:lifespan)
1. **Auth Validation** (conditional)
   - If `RL_ACTOR_ENABLED=false`: Validate auth early
   - If `RL_ACTOR_ENABLED=true`: Skip early validation (OrderManager validates later)

2. **Database Initialization**
   - `rl_db.initialize()` - PostgreSQL connection pool

3. **Write Queue Start**
   - `get_write_queue().start()` - Background queue for DB writes

4. **EventBus Start**
   - `get_event_bus().start()` - Internal event bus for orderbook updates

5. **Market Selection**
   - `select_market_tickers()` - Discovery mode or config mode
   - Returns list of market tickers to monitor

6. **OrderbookClient Initialization**
   - Create `OrderbookClient(market_tickers, stats_collector)`
   - Set connection/disconnection/error handlers
   - **Start as background task** (async, non-blocking)

7. **Stats Collector Start**
   - `stats_collector.start()` - Metrics tracking

8. **WebSocket Manager Start**
   - `websocket_manager.start()` - Frontend WebSocket connections
   - Subscribes to orderbook state updates
   - Starts stats broadcast loop

### Phase 2: Trading Components (if RL_ACTOR_ENABLED=true)
9. **Observation Adapter Creation**
   - `LiveObservationAdapter(...)` - Converts orderbook to observation space

10. **OrderManager Initialization** (`KalshiMultiMarketOrderManager.initialize()`)
    - **Trading Client Connection**
      - `KalshiDemoTradingClient.connect()` - Connects to Kalshi API
      - Validates credentials
    - **Fill Processor Start**
      - `_process_fills()` task started - Processes fill events from queue
    - **Fill Listener Start** (WebSocket)
      - `FillListener.start()` - WebSocket listener for Kalshi fill events
      - Falls back to periodic sync if fails
    - **Cleanup (if enabled)**
      - `cleanup_orders_and_positions()` - Cancels open orders (if `RL_CLEANUP_ON_START=true`)
      - Broadcasts cleanup summary via WebSocket
    - **Order Sync**
      - `sync_orders_with_kalshi(is_startup=True)` - Syncs orders from Kalshi API
      - `_sync_positions_with_kalshi(is_startup=True)` - Syncs positions and cash balance
      - Only if `RL_ORDER_SYNC_ENABLED` and `RL_ORDER_SYNC_ON_STARTUP` are true
    - **Periodic Sync Start**
      - `_start_periodic_sync()` task - Periodic sync loop
    - **Session Start Values Capture**
      - Records initial cash and portfolio value

11. **ActorService Initialization**
    - Create `ActorService(...)` with market tickers, model path
    - Set action selector (hardcoded or RL model)
    - Register selector with order manager
    - Set order manager and websocket manager
    - Set state change callback for WebSocket broadcasts
    - `actor_service.initialize()`:
      - Validates dependencies
      - Loads RL model (if needed)
      - Subscribes to EventBus for orderbook snapshots/deltas
      - Starts event processing loop

### Phase 3: Services Running
- OrderbookClient: Continuously receiving orderbook updates (background task)
- WebSocketManager: Broadcasting updates to frontend
- EventBus: Routing orderbook events
- ActorService: Processing market events, making trading decisions
- OrderManager: Managing orders, positions, fills
- FillListener: Listening to Kalshi WebSocket for fills

## Current Issues / Gaps

1. **No Formalized Initialization Checklist**
   - Startup happens but no step-by-step verification
   - No visibility into which steps completed/failed

2. **No Health Check Broadcasting During Startup**
   - Frontend doesn't know initialization progress
   - No WebSocket messages for initialization steps

3. **Cleanup/Sync Not Guaranteed to Complete**
   - Cleanup and sync happen but frontend isn't notified of completion
   - No error handling visibility for frontend

4. **No Component Health Status**
   - Health checks exist in `/rl/health` endpoint but not broadcast
   - Frontend can't see component health in real-time

5. **Orderbook Sync Not Explicitly Verified**
   - OrderbookClient starts but no explicit check that it's receiving data
   - No verification that markets are subscribed correctly

6. **No Trader State Verification**
   - Positions/orders synced but no confirmation message
   - No verification that listeners are properly subscribed

## Current WebSocket Messages (Frontend Receives)
- `connection` - Initial connection with API URLs
- `stats` - Orderbook statistics
- `trader_state` - Combined trader state
- `trader_action` - Individual trading actions
- `orders_update` - Order updates
- `positions_update` - Position updates
- `portfolio_update` - Portfolio/balance updates
- `fill_event` - Fill events

## Missing WebSocket Messages for Initialization
- `initialization_start` - Initialization sequence started
- `initialization_step` - Individual step progress
- `initialization_complete` - All steps completed
- `initialization_error` - Step failed
- `component_health` - Component health status updates

