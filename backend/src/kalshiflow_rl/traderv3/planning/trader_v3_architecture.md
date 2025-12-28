# TRADER V3 Architecture Documentation

> Machine-readable architecture reference for coding agents.
> Last updated: 2024-12-28

## 1. System Overview

TRADER V3 is an event-driven paper trading system for Kalshi prediction markets. It uses WebSocket connections to receive real-time orderbook data, maintains state through a centralized container with version tracking, and coordinates trading decisions through a clean component architecture. All trading data is in CENTS (Kalshi's native unit).

**Current Status**: MVP complete with orderbook integration, state management, WebSocket broadcasting, and event-driven whale following ("Follow the Whale"). The WHALE_FOLLOWER strategy executes trades in real-time based on detected whale activity.

## 2. Architecture Diagram

```
                                    +------------------+
                                    |   Frontend UI    |
                                    | (localhost:5173) |
                                    +--------+---------+
                                             |
                                    WebSocket (ws://localhost:8005/v3/ws)
                                             |
+====================================================================================================+
|                                    V3 TRADER (Port 8005)                                           |
+====================================================================================================+
|                                                                                                    |
|    +------------------+          +------------------+          +------------------+                |
|    |  Starlette App   |--------->|   V3Coordinator  |<---------|    V3Config      |                |
|    |     (app.py)     |          | (orchestration)  |          | (environment.py) |                |
|    +------------------+          +--------+---------+          +------------------+                |
|                                           |                                                        |
|              +----------------------------+----------------------------+                           |
|              |                            |                            |                           |
|              v                            v                            v                           |
|    +------------------+          +------------------+          +------------------+                |
|    | V3StateMachine   |          |    EventBus      |          | V3WebSocketMgr   |                |
|    | (state_machine)  |<-------->| (pub/sub events) |<-------->| (client comms)   |                |
|    +------------------+          +------------------+          +------------------+                |
|              |                            ^                            ^                           |
|              |                            |                            |                           |
|              v                            |                            |                           |
|    +------------------+          +--------+--------+                   |                           |
|    | V3StateContainer |          |                 |                   |                           |
|    | (centralized)    |<---------+ V3HealthMonitor +-------------------+                           |
|    +------------------+          |                 |                                               |
|              ^                   +-----------------+                                               |
|              |                            ^                                                        |
|              |                            |                                                        |
|    +---------+---------+---------+--------+--------+                                               |
|    |                   |         |                 |                                               |
|    v                   v         v                 v                                               |
|  +-------------------+ +-------------------+  +----+---------------+  +-------------------+        |
|  | V3OrderbookInteg  | |V3TradingClientInt|  | V3StatusReporter   |  | WhaleTracker      |        |
|  | (market data)     | | (order mgmt)     |  | (status broadcast) |  | (big bet detect)  |        |
|  +--------+----------+ +--------+---------+  +--------------------+  +--------+----------+        |
|           |                     |                                             ^                   |
|           |                     v                                             |                   |
|           |            +-------------------+          +-------------------+   |                   |
|           |            | KalshiDataSync    |--------->| TraderState       |   |                   |
|           |            | (sync service)    |          | (state/trader_    |   |                   |
|           |            +-------------------+          |  state.py)        |   |                   |
|           |                     |                     +-------------------+   |                   |
|           |                     v                                             |                   |
|           |            +-------------------+                                  |                   |
|           |            |TradingFlowOrch    |                                  |                   |
|           |            | (cycle mgmt)      |                                  |                   |
|           |            +-------------------+                                  |                   |
|           |                     |                                             |                   |
|           v                     v                                             |                   |
|  +-------------------+ +-------------------+                 +----------------+--+                 |
|  | OrderbookClient   | |KalshiDemoTrading  |                 | V3TradesIntegration|               |
|  | (data/orderbook)  | | Client            |                 | (public trades)    |               |
|  +--------+----------+ +--------+---------+                  +--------+-----------+               |
|           |                     |                                     |                           |
|           |                     |                                     v                           |
|           |                     |                            +-------------------+                |
|           |                     |                            | TradesClient      |                |
|           |                     |                            | (trades WS)       |                |
|           |                     |                            +--------+----------+                |
|           |                     |                                     |                           |
|           |                     |                            +--------+----------+                |
|           |                     |                            | PositionListener  |                |
|           |                     |                            | (positions WS)    |                |
|           |                     |                            +--------+----------+                |
|           |                     |                                     |                           |
+===========|=====================|=====================================|===========================+
            |                     |                                     |
            v                     v                                     v
    +---------------+     +---------------+                    +----------------+
    | Kalshi WS API |     | Kalshi REST   |                    | Kalshi WS API  |
    | (orderbook)   |     | (demo-api)    |                    | (user channel) |
    +---------------+     +---------------+                    +----------------+
```

## 3. Component Index

### 3.1 Core Components (`traderv3/core/`)

#### V3Coordinator
- **File**: `core/coordinator.py`
- **Purpose**: Central orchestrator managing component lifecycle, event loop, and system coordination
- **Key Methods**:
  - `start()` - Initialize and start all components
  - `stop()` - Graceful shutdown of all components
  - `get_status()` - Get comprehensive system status
  - `get_health()` - Quick health check
  - `is_healthy()` - Boolean health status
- **Emits Events**: None directly (delegates to StatusReporter)
- **Subscribes To**: None directly
- **Dependencies**: V3StateMachine, EventBus, V3WebSocketManager, V3OrderbookIntegration, V3TradingClientIntegration (optional), V3StateContainer, V3HealthMonitor, V3StatusReporter, TradingFlowOrchestrator

#### V3StateMachine
- **File**: `core/state_machine.py`
- **Purpose**: Manages operational state with validated transitions, timeout protection, and recovery
- **Key Methods**:
  - `transition_to(new_state, context, metadata)` - Validated state transition
  - `enter_error_state(error_context, error)` - Transition to ERROR with tracking
  - `check_state_timeout()` - Check if current state exceeded timeout
  - `get_metrics()` - Get StateMetrics dataclass
  - `get_status_summary()` - Get detailed status dictionary
  - `register_state_callback(state, callback, on_enter)` - Register transition callbacks
- **Emits Events**: `STATE_TRANSITION`, `SYSTEM_ACTIVITY`
- **Subscribes To**: None
- **Dependencies**: EventBus (optional)

#### EventBus
- **File**: `core/event_bus.py`
- **Purpose**: Central pub/sub event distribution with async processing and error isolation
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `subscribe(event_type, callback)` - Register event handler
  - `emit_state_transition()` - Emit state change event
  - `emit_trader_status()` - Emit status update event
  - `emit_system_activity()` - Emit unified console message
  - `emit_orderbook_snapshot()` / `emit_orderbook_delta()` - Market data events
- **Emits Events**: Distributes all event types
- **Subscribes To**: N/A (is the event distributor)
- **Dependencies**: None

#### V3StateContainer
- **File**: `core/state_container.py`
- **Purpose**: Centralized state storage with versioning, change detection, and atomic updates
- **Key Methods**:
  - `update_trading_state(state, changes)` - Update trading data (returns bool if changed)
  - `update_component_health(name, healthy, details)` - Update component health
  - `update_machine_state(state, context, metadata)` - Update state machine reference
  - `get_full_state()` - Get complete state snapshot
  - `get_trading_summary()` - Get trading state for WebSocket broadcast
  - `atomic_update(update_func)` - Thread-safe atomic state update
  - `compare_and_swap(expected_version, update_func)` - CAS operation
- **Emits Events**: None
- **Subscribes To**: None
- **Dependencies**: TraderState, StateChange

#### V3HealthMonitor
- **File**: `core/health_monitor.py`
- **Purpose**: Monitors component health and triggers recovery from ERROR states
- **Key Methods**:
  - `start()` / `stop()` - Start/stop monitoring loop
  - `get_status()` - Get monitor status
  - `is_healthy()` - Check if monitor itself is healthy
- **Emits Events**: `SYSTEM_ACTIVITY` (health_check type)
- **Subscribes To**: None (polls components directly)
- **Dependencies**: V3Config, V3StateMachine, EventBus, V3WebSocketManager, V3StateContainer, V3OrderbookIntegration, V3TradingClientIntegration (optional), V3TradesIntegration (optional), WhaleTracker (optional)

#### V3StatusReporter
- **File**: `core/status_reporter.py`
- **Purpose**: Broadcasts status updates and trading state changes to WebSocket clients
- **Key Methods**:
  - `start()` / `stop()` - Start/stop reporting loops
  - `emit_status_update(context)` - Emit immediate status update
  - `emit_trading_state()` - Broadcast trading state via WebSocket
  - `set_started_at(timestamp)` - Set system start time for uptime calc
- **Emits Events**: `TRADER_STATUS` (via EventBus)
- **Subscribes To**: None (polls state directly)
- **Dependencies**: V3Config, V3StateMachine, EventBus, V3WebSocketManager, V3StateContainer, V3OrderbookIntegration, V3TradingClientIntegration

#### V3WebSocketManager
- **File**: `core/websocket_manager.py`
- **Purpose**: Manages WebSocket connections to frontend clients, broadcasts events
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `handle_websocket(websocket)` - Handle new client connection
  - `broadcast_message(message_type, data)` - Send to all clients
  - `broadcast_console_message(level, message, context)` - Console-style message
  - `get_stats()` - Get connection statistics
  - `get_health_details()` - Get detailed health information
- **Emits Events**: None
- **Subscribes To**: `SYSTEM_ACTIVITY`, `TRADER_STATUS`
- **Dependencies**: EventBus, V3StateMachine (optional), Starlette WebSocket

#### TradingFlowOrchestrator
- **File**: `core/trading_flow_orchestrator.py`
- **Purpose**: Manages complete trading cycles with sync, evaluate, decide, execute phases
- **Key Methods**:
  - `check_and_run_cycle()` - Check if cycle should run and execute
  - `run_trading_cycle()` - Execute full trading cycle
  - `get_stats()` - Get orchestrator statistics
  - `set_cycle_interval(interval)` - Configure cycle timing
- **Emits Events**: `SYSTEM_ACTIVITY` (trading_cycle type)
- **Subscribes To**: None
- **Dependencies**: V3Config, V3TradingClientIntegration, V3OrderbookIntegration, TradingDecisionService, V3StateContainer, EventBus, V3StateMachine

### 3.2 Client Integrations (`traderv3/clients/`)

#### V3OrderbookIntegration
- **File**: `clients/orderbook_integration.py`
- **Purpose**: Wraps OrderbookClient for V3, tracks metrics and connection state
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `wait_for_connection(timeout)` - Wait for WebSocket connection
  - `wait_for_first_snapshot(timeout)` - Wait for initial data
  - `get_metrics()` - Get orderbook metrics
  - `get_health_details()` - Get detailed health info
  - `ensure_session_for_recovery()` - Prepare for ERROR recovery
- **Emits Events**: None (listens to OrderbookClient events)
- **Subscribes To**: `ORDERBOOK_SNAPSHOT`, `ORDERBOOK_DELTA`
- **Dependencies**: OrderbookClient, EventBus

#### V3TradingClientIntegration
- **File**: `clients/trading_client_integration.py`
- **Purpose**: Wraps KalshiDemoTradingClient for order/position management
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `wait_for_connection(timeout)` - Connect to trading API
  - `sync_with_kalshi()` - Sync state (returns TraderState, StateChange)
  - `place_order(ticker, action, side, count, price)` - Place order
  - `cancel_order(order_id)` - Cancel single order
  - `cancel_all_orders()` - Cancel all open orders
  - `cancel_orphaned_orders()` - Cancel orders without order_group_id
  - `create_or_get_order_group(contracts_limit)` - Setup portfolio limits
  - `reset_order_group()` - Reset order group session
- **Emits Events**: `SYSTEM_ACTIVITY` (operation, cleanup types)
- **Subscribes To**: None
- **Dependencies**: KalshiDemoTradingClient, EventBus, KalshiDataSync

#### KalshiDemoTradingClient
- **File**: `clients/demo_client.py`
- **Purpose**: Direct API client for Kalshi demo (paper) trading environment
- **Key Methods**:
  - `connect()` / `disconnect()` - API connection
  - `get_account_info()` - Get balance and portfolio value
  - `get_positions()` - Get all positions
  - `get_orders()` - Get open orders
  - `create_order()` - Place new order
  - `cancel_order()` - Cancel order
  - `create_order_group()` - Create portfolio limit group
  - `get_order_group()` - Get order group status
- **Emits Events**: None
- **Subscribes To**: None
- **Dependencies**: None (uses httpx for REST calls)

#### TradesClient
- **File**: `clients/trades_client.py`
- **Purpose**: WebSocket client for Kalshi public trades stream (all markets)
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `wait_for_connection(timeout)` - Wait for WebSocket connection
  - `is_healthy()` - Check connection and message activity health
  - `get_stats()` - Get client statistics
  - `get_health_details()` - Get detailed health information
  - `from_env()` - Factory method to create from environment variables
- **Features**:
  - Subscribes to Kalshi `trade` channel for all public trades
  - Automatic reconnection with exponential backoff
  - Callback pattern for trade processing
  - Health monitoring based on message activity
- **Emits Events**: None (uses callback pattern)
- **Subscribes To**: None
- **Dependencies**: KalshiAuth, websockets

#### V3TradesIntegration
- **File**: `clients/trades_integration.py`
- **Purpose**: Integration layer wrapping TradesClient for V3 EventBus
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `wait_for_connection(timeout)` - Wait for trades WebSocket connection
  - `wait_for_first_trade(timeout)` - Wait for data flow confirmation
  - `get_metrics()` - Get integration metrics
  - `is_healthy()` - Check if integration is healthy
  - `get_health_details()` - Get detailed health information
- **Emits Events**: `PUBLIC_TRADE_RECEIVED` (via EventBus.emit_public_trade)
- **Subscribes To**: None (listens to TradesClient callbacks)
- **Dependencies**: TradesClient, EventBus

#### PositionListener
- **File**: `clients/position_listener.py`
- **Purpose**: WebSocket listener for real-time position updates from Kalshi
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `is_healthy()` - Check if listener is running and WebSocket connected
  - `get_metrics()` - Get listener statistics (positions received/processed)
  - `get_status()` - Get full status for monitoring
  - `get_health_details()` - Get detailed health information
- **Features**:
  - Subscribes to Kalshi `market_positions` WebSocket channel
  - Automatic reconnection with configurable delay
  - Heartbeat monitoring for connection health
  - Centi-cents to cents conversion (Kalshi API uses centi-cents)
- **Emits Events**: `MARKET_POSITION` (via EventBus.emit_market_position_update)
- **Subscribes To**: None (listens to Kalshi WebSocket)
- **Dependencies**: EventBus, KalshiAuth

### 3.3 Services (`traderv3/services/`)

#### TradingDecisionService
- **File**: `services/trading_decision_service.py`
- **Purpose**: Implements trading strategy logic and order execution. Strategies: HOLD, PAPER_TEST, RL_MODEL, CUSTOM. Note: WHALE_FOLLOWER strategy detection/timing is handled by WhaleExecutionService; this service executes the actual orders.
- **Key Methods**:
  - `evaluate_market(market, orderbook)` - Generate trading decision
  - `execute_decision(decision)` - Execute a trading decision (used by WhaleExecutionService)
  - `set_strategy(strategy)` - Change trading strategy
  - `get_stats()` - Get service statistics
  - `get_followed_whale_ids()` - Get IDs of successfully followed whales
  - `get_followed_whales()` - Get full data for followed whales
- **Emits Events**: `SYSTEM_ACTIVITY` (trading_decision, trading_error, whale_follow types)
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, V3StateContainer, EventBus

#### WhaleTracker
- **File**: `services/whale_tracker.py`
- **Purpose**: Tracks biggest bets (whales) from public trades for "Follow the Whale" strategy
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `is_healthy()` - Check if tracker is healthy (running + prune task active)
  - `get_queue_state()` - Get current whale queue and statistics
  - `get_health_details()` - Get detailed health information
  - `remove_whale(whale_id)` - Remove whale from queue after processing
- **Key Classes**:
  - `BigBet`: Dataclass representing a significant trade with whale_size calculation
    - `whale_size`: max(cost, payout) where cost=count*price_cents, payout=count*100
    - Captures both high-conviction (expensive) and high-leverage (cheap) bets
- **Configuration** (environment variables):
  - `WHALE_QUEUE_SIZE`: Maximum bets in queue (default: 10)
  - `WHALE_WINDOW_MINUTES`: Sliding window duration (default: 5)
  - `WHALE_MIN_SIZE_CENTS`: Minimum whale_size to track (default: 10000 = $100)
- **Emits Events**: `WHALE_QUEUE_UPDATED` (via EventBus.emit_whale_queue)
- **Subscribes To**: `PUBLIC_TRADE_RECEIVED`
- **Dependencies**: EventBus

#### WhaleExecutionService
- **File**: `services/whale_execution_service.py`
- **Purpose**: Event-driven whale execution with rate limiting, deduplication, and decision tracking
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `is_healthy()` - Check if service is healthy
  - `get_decision_history(limit)` - Get recent whale decisions (most recent first)
  - `get_stats()` - Get comprehensive statistics with skip reason breakdown
  - `get_health_details()` - Get detailed health information
- **Key Classes**:
  - `WhaleDecision`: Tracks decision about a whale (followed, skipped_age, skipped_position, etc.)
    - `whale_id`: Unique identifier (market_ticker:timestamp_ms)
    - `action`: Decision action (followed, skipped_age, skipped_position, skipped_orders, rate_limited, failed)
    - `reason`: Human-readable explanation
    - `order_id`: Kalshi order ID if whale was followed
- **Key Features**:
  - **Token Bucket Rate Limiting**: Prevents trading too fast (configurable trades/minute)
  - **Deduplication**: Each whale evaluated only once across all queue updates
  - **Immediate Execution**: Processes whales on event receipt, no 30s cycle delay
  - **Decision Audit**: Records all decisions for frontend visibility
  - **Position Check**: Skips whales in markets where user already has a position (new markets only strategy)
- **Configuration** (environment variables):
  - `WHALE_MAX_TRADES_PER_MINUTE`: Rate limit capacity (default: 3)
  - `WHALE_TOKEN_REFILL_SECONDS`: Token refill interval (default: 20s = ~3/minute)
  - `WHALE_MAX_AGE_SECONDS`: Maximum whale age to follow (default: 120s)
- **Emits Events**: `SYSTEM_ACTIVITY` (whale_processing type)
- **Subscribes To**: `WHALE_QUEUE_UPDATED`
- **Dependencies**: EventBus, TradingDecisionService, V3StateContainer, WhaleTracker

### 3.4 State (`traderv3/state/`)

#### TraderState
- **File**: `state/trader_state.py`
- **Purpose**: Data class representing complete trader state (ALL VALUES IN CENTS)
- **Key Fields**:
  - `balance: int` - Available cash in cents
  - `portfolio_value: int` - Portfolio value in cents
  - `positions: Dict[str, Any]` - Position data by ticker
  - `orders: Dict[str, Any]` - Order data by order_id
  - `order_group: Optional[OrderGroupState]` - Portfolio limit group
  - `sync_timestamp: float` - When state was synced
- **Factory Method**: `from_kalshi_data(balance_data, positions_data, orders_data, settlements_data)`

#### StateChange
- **File**: `state/trader_state.py`
- **Purpose**: Tracks deltas between syncs
- **Key Fields**:
  - `balance_change: int` - Change in cents
  - `portfolio_value_change: int` - Change in cents
  - `position_count_change: int` - Change in position count
  - `order_count_change: int` - Change in order count

### 3.5 Sync (`traderv3/sync/`)

#### KalshiDataSync
- **File**: `sync/kalshi_data_sync.py`
- **Purpose**: Fetches state from Kalshi API and tracks changes between syncs
- **Key Methods**:
  - `sync_with_kalshi()` - Full sync, returns (TraderState, StateChange)
  - `refresh_state()` - Refresh without tracking changes
  - `set_order_group_id(id)` - Set order group to track
  - `has_state()` - Check if synced state exists
- **Emits Events**: None
- **Subscribes To**: None
- **Dependencies**: KalshiDemoTradingClient, TraderState, StateChange

### 3.6 Configuration (`traderv3/config/`)

#### V3Config
- **File**: `config/environment.py`
- **Purpose**: Environment-based configuration with validation
- **Key Fields**:
  - `api_url`, `ws_url` - Kalshi API endpoints
  - `api_key_id`, `private_key_content` - Authentication
  - `market_tickers` - Markets to subscribe
  - `max_markets` - Market limit
  - `enable_trading_client` - Enable trading integration
  - `trading_mode` - "paper" or "production"
  - `port` - Server port (default 8005)
- **Factory Method**: `from_env()` - Load from environment variables
- **Key Methods**: `is_demo_environment()`, `get_environment_name()`, `validate()`

## 4. State Machine

### 4.1 States

| State | Description | Timeout |
|-------|-------------|---------|
| `STARTUP` | Initial state, loading configuration | 30s |
| `INITIALIZING` | Starting EventBus, WebSocket manager | 60s |
| `ORDERBOOK_CONNECT` | Connecting to Kalshi orderbook WebSocket | 120s |
| `TRADING_CLIENT_CONNECT` | Connecting to trading API (optional) | 60s |
| `KALSHI_DATA_SYNC` | Syncing positions/orders with exchange | 60s |
| `READY` | Fully operational, monitoring markets | Infinite |
| `ACTING` | Executing trades (temporary) | 60s |
| `ERROR` | Error state with recovery capability | 300s |
| `SHUTDOWN` | Terminal state for graceful shutdown | 30s |

### 4.2 State Diagram

```
                     +----------+
                     | STARTUP  |
                     +----+-----+
                          |
                          v
                  +---------------+
                  | INITIALIZING  |
                  +-------+-------+
                          |
                          v
               +-------------------+
               | ORDERBOOK_CONNECT |
               +--------+----------+
                        |
          +-------------+-------------+
          |                           |
          v                           v
+---------------------+          +--------+
|TRADING_CLIENT_CONNECT|         | READY  |<----+
+----------+----------+          +---+----+     |
           |                         |          |
           v                         |          |
   +---------------+                 |          |
   |KALSHI_DATA_SYNC|                |          |
   +-------+-------+                 |          |
           |                         |          |
           +----------->+------------+          |
                        |                       |
                        v                       |
                   +--------+                   |
                   | ACTING |-------------------+
                   +---+----+
                       |
                       |     (on error from any state)
           +-----------+---------------------------+
           |                                       |
           v                                       v
       +-------+                            +----------+
       | ERROR |<-------------------------->| SHUTDOWN |
       +---+---+                            +----------+
           |
           +-----> STARTUP (recovery)
```

### 4.3 Valid Transitions

```python
VALID_TRANSITIONS = {
    STARTUP: {INITIALIZING, ERROR, SHUTDOWN},
    INITIALIZING: {ORDERBOOK_CONNECT, ERROR, SHUTDOWN},
    ORDERBOOK_CONNECT: {TRADING_CLIENT_CONNECT, READY, ERROR, SHUTDOWN},
    TRADING_CLIENT_CONNECT: {KALSHI_DATA_SYNC, ERROR, SHUTDOWN},
    KALSHI_DATA_SYNC: {READY, ERROR, SHUTDOWN},
    READY: {ACTING, ORDERBOOK_CONNECT, ERROR, SHUTDOWN},
    ACTING: {READY, ERROR, SHUTDOWN},
    ERROR: {STARTUP, SHUTDOWN},
    SHUTDOWN: {}  # Terminal
}
```

## 5. Event Catalog

| Event Type | Publisher | Subscribers | Payload |
|------------|-----------|-------------|---------|
| `ORDERBOOK_SNAPSHOT` | OrderbookClient | V3OrderbookIntegration | `MarketEvent{market_ticker, sequence_number, timestamp_ms, metadata}` |
| `ORDERBOOK_DELTA` | OrderbookClient | V3OrderbookIntegration | `MarketEvent{market_ticker, sequence_number, timestamp_ms, metadata}` |
| `STATE_TRANSITION` | V3StateMachine | (legacy, kept for compat) | `StateTransitionEvent{from_state, to_state, context, metadata}` |
| `TRADER_STATUS` | V3StatusReporter | V3WebSocketManager | `TraderStatusEvent{state, metrics, health, timestamp}` |
| `SYSTEM_ACTIVITY` | V3StateMachine, V3HealthMonitor, TradingFlowOrchestrator, TradingDecisionService, WhaleExecutionService | V3WebSocketManager | `SystemActivityEvent{activity_type, message, metadata}` |
| `PUBLIC_TRADE_RECEIVED` | V3TradesIntegration | WhaleTracker | `PublicTradeEvent{market_ticker, timestamp_ms, side, price_cents, count}` |
| `WHALE_QUEUE_UPDATED` | WhaleTracker | V3WebSocketManager, WhaleExecutionService | `WhaleQueueEvent{queue, stats, timestamp}` |
| `MARKET_POSITION` | PositionListener | V3StateContainer | `MarketPositionEvent{ticker, position_data, timestamp}` |
| `SETTLEMENT` | (future) | (future) | Market settlement events |

### 5.1 SystemActivityEvent Types

| activity_type | Source | Description |
|---------------|--------|-------------|
| `state_transition` | V3StateMachine | State machine state changes |
| `sync` | V3Coordinator | Kalshi API sync operations |
| `health_check` | V3HealthMonitor | Component health status |
| `trading_cycle` | TradingFlowOrchestrator | Trading cycle start/complete |
| `trading_decision` | TradingDecisionService | Trade execution |
| `trading_error` | TradingDecisionService | Trade execution failures |
| `operation` | V3TradingClientIntegration | Order group operations |
| `cleanup` | V3TradingClientIntegration | Orphaned order cleanup |
| `connection` | V3Coordinator | WebSocket connection events |
| `whale_processing` | WhaleExecutionService | Whale queue processing status (for frontend animation) |
| `whale_follow` | TradingDecisionService | Successful whale follow execution |

## 6. Data Flow Traces

### 6.1 System Startup Sequence

```
1. app.py: lifespan() starts
2. V3Config.from_env() loads configuration
3. rl_db.initialize() initializes database
4. EventBus created
5. V3StateMachine created with EventBus
6. V3WebSocketManager created with EventBus
7. Market tickers selected (discovery or config mode)
8. OrderbookClient created with market tickers
9. V3OrderbookIntegration wraps OrderbookClient
10. [if trading enabled] KalshiDemoTradingClient created
11. [if trading enabled] V3TradingClientIntegration wraps trading client
12. V3Coordinator created with all components
13. coordinator.start() called:
    a. _initialize_components():
       - event_bus.start()
       - websocket_manager.start()
       - state_machine.start() -> STARTUP -> INITIALIZING
    b. _establish_connections():
       - _connect_orderbook():
         * orderbook_integration.start()
         * wait_for_connection(30s)
         * wait_for_first_snapshot(10s)
         * state_machine -> ORDERBOOK_CONNECT
       - [if trading] _connect_trading_client():
         * trading_client.wait_for_connection()
         * create_order_group()
         * state_machine -> TRADING_CLIENT_CONNECT
       - [if trading] _sync_trading_state():
         * sync_with_kalshi()
         * state_container.update_trading_state()
         * state_machine -> KALSHI_DATA_SYNC
       - [if trading + cleanup_on_startup] _cleanup_orphaned_orders():
         * cancel_orphaned_orders() for orders without order_group_id
       - _transition_to_ready():
         * state_machine -> READY
    c. _run_event_loop() starts as background task
    d. health_monitor.start()
    e. status_reporter.start()
14. System is READY
```

### 6.2 Trading Cycle Flow

```
1. _run_event_loop() checks if in READY state
2. trading_orchestrator.check_and_run_cycle()
3. If interval elapsed (30s default):
   a. TradingCycle created with unique cycle_id
   b. SYNC phase:
      - sync_with_kalshi() fetches balance, positions, orders
      - state_container.update_trading_state()
   c. EVALUATE phase:
      - Select markets to evaluate (first 3)
      - Get orderbook for each market
   d. DECIDE phase:
      - trading_service.evaluate_market() for each
      - Returns TradingDecision (HOLD by default)
   e. EXECUTE phase (if non-HOLD decisions):
      - state_machine -> ACTING
      - trading_service.execute_decision()
      - Post-execution sync
      - state_machine -> READY
   f. COMPLETE phase:
      - Cycle stored in history
      - SYSTEM_ACTIVITY event emitted
```

### 6.3 Shutdown Sequence

```
1. SIGINT/SIGTERM received
2. lifespan() finally block executes
3. coordinator.stop() called:
   a. _running = False
   b. _event_loop_task cancelled
   c. health_monitor.stop()
   d. status_reporter.stop()
   e. [if trading] trading_client_integration.stop()
      - reset_order_group()
      - disconnect()
   f. orderbook_integration.stop()
      - orderbook_client.stop()
   g. state_machine -> SHUTDOWN
   h. state_machine.stop()
   i. websocket_manager.stop()
      - Disconnect all clients
   j. event_bus.stop()
      - Clear all subscribers
4. write_queue.stop()
5. rl_db.close()
6. Application exits
```

### 6.4 Whale Execution Flow (Event-Driven)

```
1. TradesClient receives public trade from Kalshi WebSocket
2. V3TradesIntegration processes trade:
   - Emits PUBLIC_TRADE_RECEIVED event via EventBus
3. WhaleTracker receives PUBLIC_TRADE_RECEIVED:
   - Calculates whale_size = max(cost, payout)
   - If whale_size >= WHALE_MIN_SIZE_CENTS:
     * Creates BigBet, adds to priority queue
     * Emits WHALE_QUEUE_UPDATED event
4. WhaleExecutionService receives WHALE_QUEUE_UPDATED:
   a. Refill rate limit tokens
   b. For each whale in queue (sorted by whale_size desc):
      i.   Deduplication check: Skip if whale_id in _evaluated_whale_ids
      ii.  Age check: Skip if age > WHALE_MAX_AGE_SECONDS (record skipped_age)
      iii. Position check: Skip if market_ticker in positions (record skipped_position)
      iv.  Orders check: Skip if market_ticker has open orders (record skipped_orders)
      v.   Rate limit check: Skip if no tokens available (do NOT record - retry later)
      vi.  Execute: Call TradingDecisionService.execute_decision()
           - TradingDecisionService._execute_buy() places order
           - On success: Record "followed", remove from queue
           - On failure: Record "failed", remove from queue
   c. Emit whale_processing system activity for frontend animation
5. Frontend receives whale_processing message and animates UI
```

**Key Design Decisions:**
- **Immediate Execution**: Whales processed on event receipt, not 30s cycle
- **Deduplication**: Each whale evaluated ONCE across all queue updates
- **Rate Limiting**: Token bucket prevents trading too fast
- **New Markets Only**: Skips whales in markets where user has ANY position
- **Removal on Terminal Decision**: Followed/skipped/failed whales removed from queue
- **Retry on Rate Limit**: Rate-limited whales stay in queue for retry

## 7. Configuration

### 7.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KALSHI_API_URL` | Yes | - | REST API base URL |
| `KALSHI_WS_URL` | Yes | - | WebSocket URL |
| `KALSHI_API_KEY_ID` | Yes | - | API key identifier |
| `KALSHI_PRIVATE_KEY_CONTENT` | Yes | - | RSA private key |
| `ENVIRONMENT` | No | `local` | Environment name (paper/local/production) |
| `RL_MODE` | No | `discovery` | Market selection mode |
| `RL_MARKET_TICKERS` | No | `INXD-25JAN03` | Comma-separated tickers (config mode) |
| `RL_ORDERBOOK_MARKET_LIMIT` | No | `10` | Max markets to subscribe |
| `V3_ENABLE_TRADING_CLIENT` | No | `false` | Enable trading integration |
| `V3_TRADING_MODE` | No | `paper` | Trading mode |
| `V3_PORT` | No | `8005` | Server port |
| `V3_LOG_LEVEL` | No | `INFO` | Logging level |
| `V3_TRADING_STRATEGY` | No | `hold` | Trading strategy (hold, whale_follower, paper_test, rl_model) |
| `V3_ENABLE_WHALE_DETECTION` | No | `false` | Enable whale detection feature |
| `WHALE_QUEUE_SIZE` | No | `10` | Max whale bets to track in queue |
| `WHALE_WINDOW_MINUTES` | No | `5` | Sliding window for whale tracking (minutes) |
| `WHALE_MIN_SIZE_CENTS` | No | `10000` | Minimum whale_size threshold ($100 default) |
| `WHALE_MAX_TRADES_PER_MINUTE` | No | `3` | Token bucket capacity for whale following rate limit |
| `WHALE_TOKEN_REFILL_SECONDS` | No | `20` | Token refill interval (20s = ~3 trades/minute) |
| `WHALE_MAX_AGE_SECONDS` | No | `120` | Maximum whale age to follow (2 minutes) |
| `V3_CLEANUP_ON_STARTUP` | No | `true` | Cancel orphaned orders on startup |

### 7.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v3/health` | GET | Quick health check |
| `/v3/status` | GET | Detailed system status |
| `/v3/cleanup` | POST | Cancel orphaned orders (`?orphaned_only=true/false`) |
| `/v3/ws` | WebSocket | Real-time updates |

### 7.3 WebSocket Message Types (to frontend)

| Type | Description |
|------|-------------|
| `connection` | Initial connection acknowledgment |
| `history_replay` | Historical state transitions for late joiners |
| `system_activity` | Unified console messages |
| `trader_status` | Periodic status/metrics update |
| `trading_state` | Trading data (balance, positions, orders) |
| `whale_queue` | Whale detection queue update (when enabled) |
| `whale_processing` | Whale processing status for frontend animation |
| `ping` | Keep-alive ping |

#### whale_processing Message Format

```json
{
  "type": "whale_processing",
  "data": {
    "whale_id": "KXBTCMINY-25-2-DEC31-70000:1703700000000",
    "status": "processing",
    "action": null,
    "timestamp": 1703700001.5
  }
}
```

After processing:
```json
{
  "type": "whale_processing",
  "data": {
    "whale_id": "KXBTCMINY-25-2-DEC31-70000:1703700000000",
    "status": "complete",
    "action": "followed",
    "timestamp": 1703700002.3
  }
}
```

Possible `action` values: `followed`, `skipped_age`, `skipped_position`, `skipped_orders`, `rate_limited`, `failed`

#### whale_queue Message Format

```json
{
  "type": "whale_queue",
  "data": {
    "queue": [
      {
        "market_ticker": "INXD-25JAN03",
        "side": "yes",
        "price_cents": 65,
        "count": 200,
        "cost_dollars": 130.00,
        "payout_dollars": 200.00,
        "whale_size_dollars": 200.00,
        "age_seconds": 45.2,
        "timestamp_ms": 1703700000000
      }
    ],
    "stats": {
      "trades_seen": 1500,
      "trades_discarded": 1450,
      "discard_rate_percent": 96.7,
      "queue_size": 5,
      "window_minutes": 5,
      "min_size_dollars": 100.0
    },
    "timestamp": 1703700045.2
  }
}
```

## 8. Degraded Mode

The system supports **degraded mode** when the orderbook WebSocket is unavailable but trading API is functional:

1. `_connect_orderbook()` fails or times out
2. System continues with `degraded: True` in state metadata
3. `V3HealthMonitor` detects degraded mode and does NOT transition to ERROR
4. Trading syncs continue every 30 seconds
5. Frontend displays degraded status warning
6. Trading decisions are HOLD only (no orderbook data)

## 9. Key Design Patterns

1. **Event-Driven Architecture**: All inter-component communication through EventBus
2. **State Machine**: Predictable state transitions with timeout protection
3. **Versioned State Container**: Change detection for efficient broadcasts
4. **Degraded Mode**: System continues operating without orderbook
5. **Health Monitoring**: Non-blocking health checks that report but don't control
6. **Atomic Updates**: Thread-safe state mutations with version tracking

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2024-12-27 | Initial architecture document created | Claude |
| 2024-12-27 | Foundation cleanup: removed ~600 lines dead code, fixed async patterns | Claude |
| 2024-12-27 | Final simplification: removed unused _coordinator ref, publish() wrapper (-57 lines) | Claude |
| 2024-12-27 | Added "Follow the Whale" feature: TradesClient, V3TradesIntegration, WhaleTracker | Claude |
| 2024-12-27 | Phase 1 complete: /v3/cleanup endpoint, whale queue snapshot, orphaned order cleanup on startup | Claude |
| 2024-12-28 | Added WhaleExecutionService: event-driven whale execution with rate limiting, deduplication | Claude |
| 2024-12-28 | Added whale_processing WebSocket message type, whale execution flow trace, environment vars | Claude |
| 2024-12-28 | Removed dead code: evaluate_whale_queue() in TradingDecisionService (~120 lines) | Claude |
| 2024-12-28 | Architecture review: Added PositionListener docs, MARKET_POSITION event, V3_TRADING_STRATEGY env var | Claude |
| 2024-12-28 | Dead code cleanup: Removed empty _handle_orderbook_event(), stub get_orderbook(), commented code | Claude |
