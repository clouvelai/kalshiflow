# TRADER V3 Architecture Documentation

> Machine-readable architecture reference for coding agents.
> Last updated: 2025-01-01 (comprehensive update: Lifecycle Discovery System, TrackedMarketsState, RLM Service, Trading Attachments)

## 1. System Overview

TRADER V3 is an event-driven paper trading system for Kalshi prediction markets. It uses WebSocket connections to receive real-time orderbook data and lifecycle events, maintains state through a centralized container with version tracking, and coordinates trading decisions through a clean component architecture. All trading data is in CENTS (Kalshi's native unit).

**Current Status**: Production-ready with lifecycle-based market discovery, multiple trading strategies, real-time WebSocket broadcasting, and comprehensive state management.

**Trading Strategies**:
- **WHALE_FOLLOWER**: Fade low-leverage YES whales (bet NO when whale bets YES with leverage < 2)
- **YES_80_90**: Buy YES at 80-90c (+5.1% validated edge)
- **RLM_NO**: Reverse Line Movement - bet NO when public bets YES but price drops (+17.38% validated edge, highest)

**Market Discovery Modes**:
- **Config Mode**: Static market list from `RL_MARKET_TICKERS` environment variable
- **Discovery Mode**: Dynamic market discovery via Lifecycle Discovery System with category filtering and capacity management

**Key Architectural Features**:
- **Lifecycle Discovery**: Real-time market discovery via Kalshi `market_lifecycle` WebSocket channel
- **TrackedMarketsState**: Centralized state for discovered markets with DB persistence
- **Trading Attachments**: Per-market trading state linking positions/orders to tracked markets
- **Real-time Prices**: MarketTickerListener (WebSocket) + MarketPriceSyncer (REST fallback)

## 2. Architecture Diagram

```
                                    +------------------+
                                    |   Frontend UI    |
                                    | (localhost:5173) |
                                    +--------+---------+
                                             |
                                    WebSocket (ws://localhost:8005/v3/ws)
                                             |
+====================================================================================================================+
|                                    V3 TRADER (Port 8005)                                                            |
+====================================================================================================================+
|                                                                                                                     |
|    +------------------+          +------------------+          +------------------+                                 |
|    |  Starlette App   |--------->|   V3Coordinator  |<---------|    V3Config      |                                 |
|    |     (app.py)     |          | (orchestration)  |          | (environment.py) |                                 |
|    +------------------+          +--------+---------+          +------------------+                                 |
|                                           |                                                                         |
|              +----------------------------+----------------------------+                                            |
|              |                            |                            |                                            |
|              v                            v                            v                                            |
|    +------------------+          +------------------+          +------------------+                                 |
|    | V3StateMachine   |          |    EventBus      |          | V3WebSocketMgr   |                                 |
|    | (state_machine)  |<-------->| (pub/sub events) |<-------->| (client comms)   |                                 |
|    +------------------+          +------------------+          +------------------+                                 |
|              |                            ^                            ^                                            |
|              |                            |                            |                                            |
|              v                            |                            |                                            |
|    +------------------+          +--------+--------+                   |                                            |
|    | V3StateContainer |          |                 |                   |                                            |
|    | (centralized)    |<---------+ V3HealthMonitor +-------------------+                                            |
|    +------------------+          |                 |                                                                |
|              ^                   +-----------------+                                                                |
|              |                            ^                                                                         |
|              |                            |                                                                         |
|    +---------+---------+---------+--------+--------+---------+---------+---------+                                  |
|    |                   |         |                 |         |         |         |                                  |
|    v                   v         v                 v         v         v         v                                  |
|  +-------------------+ +-------------------+ +----+---------------+ +-------------------+ +-------------------+     |
|  | V3OrderbookInteg  | |V3TradingClientInt| | V3StatusReporter   | | RLMService        | |EventLifecycleSvc  |     |
|  | (market data)     | | (order mgmt)     | | (status broadcast) | | (RLM strategy)    | | (lifecycle proc)  |     |
|  +--------+----------+ +--------+---------+ +--------------------+ +--------+----------+ +--------+----------+     |
|           |                     |                                          |                      |                |
|           |                     v                                          |                      v                |
|           |            +-------------------+          +-------------------+ |   +-------------------+              |
|           |            | KalshiDataSync    |--------->| TraderState       | |   |TrackedMarketsState|              |
|           |            | (sync service)    |          | (state/trader_    | |   | (discovery state) |              |
|           |            +-------------------+          |  state.py)        | |   +-------------------+              |
|           |                     |                     +-------------------+ |            ^                         |
|           |                     v                                           |            |                         |
|           |            +-------------------+          +-------------------+ |   +--------+----------+              |
|           |            |TradingFlowOrch    |          |TradingDecision    |-+   |TrackedMarketSync  |              |
|           |            | (cycle mgmt)      |          | Service           |     | (REST sync 30s)   |              |
|           |            +-------------------+          +-------------------+     +-------------------+              |
|           |                     |                                                        |                         |
|           v                     v                                                        v                         |
|  +-------------------+ +-------------------+     +--------------------+  +-------------------+ +------------------+|
|  | OrderbookClient   | |KalshiDemoTrading  |     |V3TradesIntegration |  |ApiDiscoverySyncer | |UpcomingMktSyncer ||
|  | (data/orderbook)  | | Client            |     | (public trades)    |  | (bootstrap mkts)  | | (upcoming mkts)  ||
|  +--------+----------+ +--------+---------+      +--------+-----------+  +-------------------+ +------------------+|
|           |                     |                         |                                                        |
|           |                     |                         v                                                        |
|           |                     |                +-------------------+      +--------------------+                 |
|           |                     |                | TradesClient      |      |V3LifecycleInteg   |                  |
|           |                     |                | (trades WS)       |      | (lifecycle events)|                  |
|           |                     |                +--------+----------+      +--------+-----------+                 |
|           |                     |                         |                          |                             |
|           |                     |                +--------+----------+      +--------+-----------+                 |
|           |                     |                | PositionListener  |      | LifecycleClient   |                  |
|           |                     |                | (positions WS)    |      | (lifecycle WS)    |                  |
|           |                     |                +--------+----------+      +--------+-----------+                 |
|           |                     |                         |                          |                             |
|           |                     |                +--------+----------+      +--------+-----------+                 |
|           |                     |                | FillListener      |      |MarketTickerListener|                 |
|           |                     |                | (fills WS)        |      | (ticker WS)        |                 |
|           |                     |                +--------+----------+      +--------+-----------+                 |
|           |                     |                         |                          |                             |
+===========|=====================|=========================|==========================|=============================+
            |                     |                         |                          |
            v                     v                         v                          v
    +---------------+     +---------------+        +----------------+         +----------------+
    | Kalshi WS API |     | Kalshi REST   |        | Kalshi WS API  |         | Kalshi WS API  |
    | (orderbook)   |     | (demo-api)    |        | (user channel) |         |(lifecycle/tick)|
    +---------------+     +---------------+        +----------------+         +----------------+
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
- **Dependencies**: V3StateMachine, EventBus, V3WebSocketManager, V3OrderbookIntegration, V3TradingClientIntegration (optional), V3StateContainer, V3HealthMonitor, V3StatusReporter, TradingFlowOrchestrator, MarketTickerListener (optional), MarketPriceSyncer (optional), TradingStateSyncer (optional), FillListener (optional), Yes8090Service (optional)

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
  - `update_trading_state(state, changes)` - Update trading data from Kalshi sync (returns bool if changed)
  - `update_single_position(ticker, position_data)` - Update single position from real-time WebSocket push
  - `update_component_health(name, healthy, details)` - Update component health
  - `update_machine_state(state, context, metadata)` - Update state machine reference
  - `update_whale_queue(state)` - Update whale queue state from tracker
  - `get_full_state()` - Get complete state snapshot
  - `get_trading_summary(order_group_id)` - Get trading state for WebSocket broadcast (includes settlements, P&L)
  - `get_whale_summary()` - Get whale queue summary for WebSocket broadcast
  - `initialize_session_pnl(balance, portfolio_value)` - Initialize session P&L tracking on first sync
  - `set_component_degraded(component, is_degraded, reason)` - Track degraded components
  - `atomic_update(update_func)` - Thread-safe atomic state update
  - `compare_and_swap(expected_version, update_func)` - CAS operation
- **Key State Types**:
  - `_trading_state`: TraderState with positions, orders, balance, settlements
  - `_session_pnl_state`: SessionPnLState for P&L tracking from session start
  - `_whale_state`: WhaleQueueState for whale detection queue
  - `_settled_positions`: Deque of last 50 settlements (synced from REST API)
  - `_session_updated_tickers`: Set of tickers updated via WebSocket this session
  - `_market_prices`: Dict[str, MarketPriceData] for real-time market prices
  - `_market_prices_version`: Version tracking for market price updates
- **MarketPriceData Fields**:
  - `ticker`: Market ticker string
  - `last_price`: Last traded price (cents)
  - `yes_bid`, `yes_ask`: YES side bid/ask prices (cents)
  - `no_bid`, `no_ask`: NO side bid/ask prices (cents)
  - `volume`: Total volume traded
  - `open_interest`: Active contracts
  - `close_time`: Market close time (ISO timestamp, from REST only)
  - `timestamp`: Update timestamp
- **Market Price Methods**:
  - `update_market_price(ticker, price_data)` - Update single ticker price
  - `get_market_price(ticker)` - Get price for ticker
  - `get_all_market_prices()` - Get all market prices
  - `clear_market_price(ticker)` - Remove ticker from prices
  - `get_market_prices_summary()` - Get summary for WebSocket broadcast
- **Emits Events**: None
- **Subscribes To**: `MARKET_POSITION_UPDATE` (via coordinator subscription)
- **Dependencies**: TraderState, StateChange, SessionPnLState, WhaleQueueState

#### V3HealthMonitor
- **File**: `core/health_monitor.py`
- **Purpose**: Monitors component health and triggers recovery from ERROR states
- **Key Methods**:
  - `start()` / `stop()` - Start/stop monitoring loop
  - `get_status()` - Get monitor status
  - `is_healthy()` - Check if monitor itself is healthy
- **Emits Events**: `SYSTEM_ACTIVITY` (health_check type)
- **Subscribes To**: None (polls components directly)
- **Dependencies**: V3Config, V3StateMachine, EventBus, V3WebSocketManager, V3StateContainer, V3OrderbookIntegration, V3TradingClientIntegration (optional), V3TradesIntegration (optional), WhaleTracker (optional), FillListener (optional)

#### V3StatusReporter
- **File**: `core/status_reporter.py`
- **Purpose**: Broadcasts status updates and trading state changes to WebSocket clients
- **Key Methods**:
  - `start()` / `stop()` - Start/stop reporting loops
  - `emit_status_update(context)` - Emit immediate status update
  - `emit_trading_state()` - Broadcast trading state via WebSocket (includes P&L, settlements, position listener health)
  - `set_started_at(timestamp)` - Set system start time for uptime calc
  - `set_position_listener(position_listener)` - Set position listener for health reporting
- **Broadcast Data** (via `emit_trading_state()`):
  - `balance`, `portfolio_value`, `position_count`, `order_count`
  - `positions_details`: Per-position P&L with unrealized_pnl, realized_pnl, fees_paid
  - `pnl`: Session P&L breakdown (realized, unrealized, total)
  - `settlements`: Last 50 settlements with net_pnl calculation
  - `position_listener`: Health status of real-time position WebSocket
- **Emits Events**: `TRADER_STATUS` (via EventBus)
- **Subscribes To**: None (polls state directly)
- **Dependencies**: V3Config, V3StateMachine, EventBus, V3WebSocketManager, V3StateContainer, V3OrderbookIntegration, V3TradingClientIntegration, PositionListener (optional)

#### V3WebSocketManager
- **File**: `core/websocket_manager.py`
- **Purpose**: Manages WebSocket connections to frontend clients, broadcasts events with message coalescing
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `handle_websocket(websocket)` - Handle new client connection
  - `broadcast_message(message_type, data)` - Send to all clients (with coalescing for frequent types)
  - `_broadcast_immediate(message_type, data)` - Bypass coalescing for critical messages
  - `_flush_pending()` - Flush coalesced messages to clients
  - `broadcast_console_message(level, message, context)` - Console-style message
  - `get_stats()` - Get connection statistics
  - `get_health_details()` - Get detailed health information
- **Message Coalescing**:
  - 100ms batching window for frequent message types
  - **Immediate (critical)**: `state_transition`, `whale_processing`, `connection`, `system_activity`, `history_replay`
  - **Coalesced (frequent)**: `trading_state`, `trader_status`, `whale_queue`
  - Coalescing keeps only the latest message of each type, reducing frontend re-renders
- **Key Fields**: `_pending_messages`, `_coalesce_task`, `_coalesce_interval`
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
  - `get_account_info()` - Get balance and portfolio value (cents)
  - `get_positions()` - Get all positions
  - `get_orders(ticker, status)` - Get orders (default: resting/open only)
  - `get_settlements()` - Get all settlements (fee_cost is string in dollars, others in cents)
  - `create_order(ticker, action, side, count, price, type, order_group_id)` - Place new order
  - `cancel_order(order_id)` - Cancel single order
  - `batch_cancel_orders(order_ids)` - Cancel multiple orders with fallback
  - `get_fills(ticker)` - Get trade fills
  - `get_markets(limit)` - Get available markets
  - `create_order_group(contracts_limit)` - Create portfolio limit group
  - `get_order_group(order_group_id)` - Get order group status
  - `update_order_group(order_group_id, contracts_limit)` - Update order group limits
  - `reset_order_group(order_group_id)` - Reset order group (clears positions/orders)
  - `delete_order_group(order_group_id)` - Delete order group
  - `list_order_groups(status)` - List all order groups
- **Safety Validations**:
  - Validates URLs point to demo-api.kalshi.co (not production)
  - Raises KalshiDemoAuthError if production URLs detected
- **Emits Events**: None
- **Subscribes To**: None
- **Dependencies**: aiohttp, websockets, KalshiAuth

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

#### MarketTickerListener
- **File**: `clients/market_ticker_listener.py`
- **Purpose**: WebSocket listener for real-time market price updates from Kalshi ticker channel
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `update_subscriptions(tickers)` - Dynamic subscription management (add/remove tickers)
  - `get_subscribed_tickers()` - Get list of currently subscribed tickers
  - `get_metrics()` - Get listener statistics
  - `get_status()` - Get full status for monitoring
  - `is_healthy()` - Check if running and WebSocket connected
  - `get_health_details()` - Get detailed health information
- **Key Features**:
  - Filtered subscription (only position tickers, not firehose)
  - Throttled updates (configurable, default 500ms per ticker)
  - Dynamic subscription management as positions change
  - Automatic reconnection on disconnect
- **Ticker Message Format** (from Kalshi):
  ```json
  {
    "type": "ticker",
    "msg": {
      "market_ticker": "INXD-25JAN03",
      "price": 52,           // last traded price (cents)
      "yes_bid": 50,         // best yes bid (cents)
      "yes_ask": 54,         // best yes ask (cents)
      "no_bid": 46,          // best no bid (cents)
      "no_ask": 50,          // best no ask (cents)
      "volume": 1500,        // total volume traded
      "open_interest": 12000,// active contracts
      "ts": 1703808000       // unix timestamp (seconds)
    }
  }
  ```
- **Emits Events**: `MARKET_TICKER_UPDATE` (via EventBus.emit_market_ticker_update)
- **Subscribes To**: Kalshi WebSocket `ticker` channel
- **Dependencies**: EventBus, KalshiAuth

#### PositionListener
- **File**: `clients/position_listener.py`
- **Purpose**: WebSocket listener for real-time position updates from Kalshi
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `is_healthy()` - Check if listener is running and WebSocket connected (handles .closed attribute safely)
  - `get_metrics()` - Get listener statistics (positions received/processed, connection count)
  - `get_status()` - Get full status for monitoring
  - `get_health_details()` - Get detailed health information for initialization tracker
- **Features**:
  - Subscribes to Kalshi `market_positions` WebSocket channel
  - Automatic reconnection with configurable delay
  - Heartbeat monitoring for connection health
  - Centi-cents to cents conversion (Kalshi API uses centi-cents)
  - WebSocket closed state detection with fallback (handles different WS library attributes)
- **Position Data Fields** (from WebSocket, after conversion):
  - `position`: Contract count (+ long, - short)
  - `market_exposure`: Current market value in cents (was position_cost in centi-cents)
  - `realized_pnl`: Realized P&L in cents
  - `fees_paid`: Fees in cents
  - `volume`: Total contracts traded
  - Note: `total_traded` (cost basis) NOT included - preserved from REST sync via merge
- **Emits Events**: `MARKET_POSITION_UPDATE` (via EventBus.emit_market_position_update)
- **Subscribes To**: None (listens to Kalshi WebSocket)
- **Dependencies**: EventBus, KalshiAuth

#### FillListener
- **File**: `clients/fill_listener.py`
- **Purpose**: WebSocket listener for real-time order fill notifications from Kalshi
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `is_healthy()` - Check if listener is running and WebSocket connected
  - `get_metrics()` - Get listener statistics (fills received/processed, connection count)
  - `get_status()` - Get full status for monitoring
  - `get_health_details()` - Get detailed health information for initialization tracker
- **Features**:
  - Subscribes to Kalshi `fill` WebSocket channel (authenticated)
  - Receives all fills for the account (ignores market_tickers filter)
  - Automatic reconnection with configurable delay
  - Heartbeat monitoring for connection health
  - Price values in cents (native Kalshi format)
- **Fill Message Format** (from Kalshi):
  ```json
  {
    "type": "fill",
    "sid": 13,
    "msg": {
      "trade_id": "d91bc706-ee49-470d-82d8-11418bda6fed",
      "order_id": "ee587a1c-8b87-4dcf-b721-9f6f790619fa",
      "market_ticker": "HIGHNY-22DEC23-B53.5",
      "is_taker": true,
      "side": "yes",
      "yes_price": 75,
      "count": 278,
      "action": "buy",
      "ts": 1671899397,
      "post_position": 500
    }
  }
  ```
- **Emits Events**: `ORDER_FILL` (via EventBus.emit_order_fill)
- **Subscribes To**: None (listens to Kalshi WebSocket)
- **Dependencies**: EventBus, KalshiAuth
- **Architecture Note**: Provides instant feedback when orders are filled, complementing REST API sync for immediate console UX

### 3.3 Services (`traderv3/services/`)

#### TradingDecisionService
- **File**: `services/trading_decision_service.py`
- **Purpose**: Implements trading strategy logic and order execution. Strategies: HOLD, WHALE_FOLLOWER, PAPER_TEST, RL_MODEL, YES_80_90, CUSTOM. Note: WHALE_FOLLOWER strategy detection/timing is handled by WhaleExecutionService; this service executes the actual orders.
- **Available Strategies** (TradingStrategy enum):
  - `HOLD`: Never trade (safe default)
  - `WHALE_FOLLOWER`: Follow big bets detected by WhaleTracker
  - `PAPER_TEST`: Simple test trades for paper mode
  - `RL_MODEL`: Use trained RL model (not yet integrated)
  - `YES_80_90`: Buy YES at 80-90c (validated +5.1% edge strategy)
  - `CUSTOM`: Custom strategy implementation
- **Key Methods**:
  - `evaluate_market(market, orderbook)` - Generate trading decision
  - `execute_decision(decision)` - Execute a trading decision (used by WhaleExecutionService)
  - `set_strategy(strategy)` - Change trading strategy
  - `get_stats()` - Get service statistics
  - `get_followed_whale_ids()` - Get IDs of successfully followed whales
  - `get_followed_whales()` - Get full data for followed whales
  - `get_decision_history(limit)` - Get recent whale decision history for audit display
  - `get_decision_stats()` - Get whale decision statistics
- **Key Classes**:
  - `FollowedWhale`: Tracks whale trades we have followed (whale_id, order_id, market, side, count, price)
  - `WhaleDecision`: Tracks decisions about whale trades (followed, skipped_age, skipped_position, etc.)
  - `TradingDecision`: Represents a trading decision (action, market, side, quantity, price)
- **Constants**:
  - `WHALE_FOLLOW_CONTRACTS = 5`: Fixed contracts per whale follow
  - `WHALE_MAX_AGE_SECONDS = 120`: Skip whales older than 2 minutes
- **Emits Events**: `SYSTEM_ACTIVITY` (trading_decision, trading_error, whale_follow types)
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, V3StateContainer, EventBus, WhaleTracker (optional)

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

#### MarketPriceSyncer
- **File**: `services/market_price_syncer.py`
- **Purpose**: REST API sync for market prices on startup and periodic refresh (every 30s)
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management (initial sync on start)
  - `is_healthy()` - Check if running and last sync not stale
  - `get_health_details()` - Get detailed health information
- **Key Properties**:
  - `sync_count` - Number of syncs completed
  - `tickers_synced` - Number of tickers in last sync
  - `last_sync_time` - Timestamp of last successful sync
- **Behavior**:
  - Performs initial sync immediately on `start()`
  - Runs periodic sync loop every 30 seconds (configurable)
  - Updates StateContainer's `_market_prices` directly (batch update pattern)
  - Fetches prices for all position tickers via REST API (batched in groups of 100)
- **Emits Events**: `SYSTEM_ACTIVITY` (sync type with sync_type="market_prices")
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, V3StateContainer, EventBus
- **Architecture Note**: Works alongside MarketTickerListener WebSocket for redundancy

#### TradingStateSyncer
- **File**: `services/trading_state_syncer.py`
- **Purpose**: Dedicated service for periodic trading state sync (balance, positions, orders, settlements) from Kalshi REST API
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management (initial sync on start)
  - `is_healthy()` - Check if running and last sync not stale
  - `get_health_details()` - Get detailed health information
- **Key Properties**:
  - `sync_count` - Number of syncs completed
  - `last_sync_time` - Timestamp of last successful sync
- **Behavior**:
  - Performs initial sync immediately on `start()`
  - Runs periodic sync loop every 20 seconds (configurable)
  - Updates StateContainer with trading state and change detection
  - Broadcasts trading_state via StatusReporter after each sync
  - Emits console-friendly system activity messages
- **Console Message Format**: "Trading session synced: X positions, $Y balance, Z settlements"
- **Emits Events**: `SYSTEM_ACTIVITY` (sync type with sync_type="trading_state")
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, V3StateContainer, EventBus, StatusReporter
- **Architecture Note**: Runs in dedicated asyncio task for reliability (same pattern as MarketPriceSyncer)

#### Yes8090Service
- **File**: `services/yes_80_90_service.py`
- **Purpose**: Event-driven trading service implementing the validated +5.1% edge YES at 80-90c strategy
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `is_healthy()` - Check if service is running
  - `get_stats()` - Get comprehensive service statistics
  - `get_decision_history()` - Get recent signal decisions
  - `get_health_details()` - Get detailed health information
  - `reset_processed_markets()` - Clear processed markets set for re-entry
- **Key Classes**:
  - `Yes8090Signal`: Detected signal with market_ticker, prices, tier, timestamp
  - `Yes8090Decision`: Records decisions (executed, skipped, failed) for audit
- **Signal Detection Criteria** (ALL must be true):
  1. Best YES ask price >= 80c AND <= 90c
  2. Best YES ask size >= min_liquidity (default: 10)
  3. Bid-ask spread <= max_spread (default: 5c)
  4. Market NOT already processed this session
  5. No existing position in this market
- **Tier System**:
  - Tier A (83-87c): Sweet spot, higher edge, 150 contracts default
  - Tier B (80-83c or 87-90c): Edges, 100 contracts default
- **Features**:
  - Token bucket rate limiting (default: 10 trades/minute)
  - Deduplication (one entry per market per session)
  - Decision history tracking (last 100 decisions)
- **Configuration** (environment variables):
  - `YES8090_MIN_PRICE`: Minimum YES ask price (default: 80)
  - `YES8090_MAX_PRICE`: Maximum YES ask price (default: 90)
  - `YES8090_MIN_LIQUIDITY`: Minimum contracts at best ask (default: 10)
  - `YES8090_MAX_SPREAD`: Maximum bid-ask spread (default: 5)
  - `YES8090_CONTRACTS`: Default contracts per trade (default: 100)
  - `YES8090_TIER_A_CONTRACTS`: Contracts for Tier A signals (default: 150)
  - `YES8090_MAX_CONCURRENT`: Maximum concurrent positions (default: 100)
- **Emits Events**: `SYSTEM_ACTIVITY` (strategy_start, strategy_stop, yes_80_90_signal, yes_80_90_execute types)
- **Subscribes To**: `ORDERBOOK_SNAPSHOT`, `ORDERBOOK_DELTA`
- **Dependencies**: EventBus, TradingDecisionService, V3StateContainer

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
  - `order_group_id` (property) - Get current order group ID being tracked
- **Sync Data Fetched**:
  1. Balance: `get_account_info()` - balance and portfolio_value in cents
  2. Positions: `get_positions()` - market_positions array
  3. Orders: `get_orders()` - orders array (open/resting only)
  4. Settlements: `get_settlements()` - historical settlements (optional, may fail)
  5. Order Group: `get_order_group(id)` - if order_group_id is set
- **Emits Events**: None
- **Subscribes To**: None
- **Dependencies**: KalshiDemoTradingClient, TraderState, StateChange, OrderGroupState

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

### 3.7 Lifecycle Discovery System (`traderv3/services/` and `traderv3/state/`)

The Lifecycle Discovery System enables dynamic market discovery via Kalshi's real-time lifecycle events, with category filtering and capacity management.

#### LifecycleClient
- **File**: `clients/lifecycle_client.py`
- **Purpose**: WebSocket client for Kalshi market lifecycle events stream
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `wait_for_connection(timeout)` - Wait for WebSocket connection
  - `is_healthy()` - Check connection health
  - `get_stats()` - Get client statistics
  - `get_health_details()` - Get detailed health information
- **Lifecycle Event Types** (from Kalshi):
  - `created`: New market opened
  - `activated`: Market trading enabled
  - `deactivated`: Market trading paused
  - `determined`: Market outcome resolved
  - `settled`: Positions liquidated
  - `close_date_updated`: Close time changed
- **Emits Events**: None (uses callback pattern)
- **Subscribes To**: Kalshi WebSocket `market_lifecycle` channel
- **Dependencies**: KalshiAuth, websockets

#### V3LifecycleIntegration
- **File**: `clients/lifecycle_integration.py`
- **Purpose**: Integration layer wrapping LifecycleClient for V3 EventBus
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `wait_for_connection(timeout)` - Wait for lifecycle WebSocket connection
  - `is_healthy()` - Check if integration is healthy
  - `get_metrics()` - Get integration metrics
  - `get_health_details()` - Get detailed health information
- **Emits Events**: `MARKET_LIFECYCLE_EVENT` (via EventBus.emit_market_lifecycle)
- **Subscribes To**: None (listens to LifecycleClient callbacks)
- **Dependencies**: LifecycleClient, EventBus

#### EventLifecycleService
- **File**: `services/event_lifecycle_service.py`
- **Purpose**: Central processing unit for lifecycle discovery - receives raw lifecycle events, enriches via REST API, filters by category, and manages tracked markets state
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management and EventBus subscription
  - `set_subscribe_callback(callback)` - Register orderbook subscription callback
  - `set_unsubscribe_callback(callback)` - Register orderbook unsubscription callback
  - `track_market_from_api_data(market_info)` - Track market from API discovery (bypasses REST lookup)
  - `get_stats()` - Get service statistics
  - `get_health_details()` - Get detailed health information
- **Processing Flow (on 'created' event)**:
  1. Check capacity (fast fail)
  2. REST lookup GET /markets/{ticker}
  3. Category filter (configurable via `LIFECYCLE_CATEGORIES`)
  4. Persist to TrackedMarketsState and DB
  5. Emit MARKET_TRACKED event
  6. Call orderbook subscribe callback
- **Processing Flow (on 'determined' event)**:
  1. Update TrackedMarketsState status
  2. Update DB status
  3. Emit MARKET_DETERMINED event
  4. Call orderbook unsubscribe callback
- **Configuration** (environment variables):
  - `LIFECYCLE_CATEGORIES`: Comma-separated list (default: "sports,media_mentions,entertainment,crypto")
- **Emits Events**: `MARKET_TRACKED`, `MARKET_DETERMINED`, `SYSTEM_ACTIVITY` (lifecycle_event type)
- **Subscribes To**: `MARKET_LIFECYCLE_EVENT`
- **Dependencies**: EventBus, TrackedMarketsState, V3TradingClientIntegration, RLDatabase

#### TrackedMarketsState
- **File**: `state/tracked_markets.py`
- **Purpose**: Single source of truth for lifecycle-discovered markets with capacity management, version tracking, and DB persistence
- **Key Classes**:
  - `MarketStatus`: Enum with `ACTIVE`, `DETERMINED`, `SETTLED` states
  - `TrackedMarket`: Dataclass with full market metadata and real-time data
- **TrackedMarket Fields**:
  - `ticker`, `event_ticker`, `title`, `category`: Market identification
  - `status`: Current lifecycle status (MarketStatus enum)
  - `created_ts`, `open_ts`, `close_ts`: Kalshi timestamps
  - `determined_ts`, `settled_ts`: Optional outcome timestamps
  - `tracked_at`: When we started tracking (epoch seconds)
  - `price`, `volume`, `volume_24h`, `open_interest`: Real-time market data
  - `yes_bid`, `yes_ask`: Best bid/ask prices (cents)
  - `discovery_source`: "lifecycle_ws" | "api" | "db_recovery"
- **Key Methods**:
  - `add_market(market)` - Add market with capacity check
  - `update_market(ticker, **kwargs)` - Update market fields
  - `update_status(ticker, status)` - Status transition
  - `remove_market(ticker)` - Remove after settlement cleanup
  - `get_active()`, `get_active_tickers()` - Get tradeable markets
  - `at_capacity()`, `capacity_remaining()` - Capacity management
  - `get_stats()` - Statistics for frontend display
  - `get_snapshot()` - Full state snapshot for WebSocket broadcast
  - `load_from_db(db_markets)` - Recovery on startup
- **Key Properties**:
  - `capacity`: Maximum markets (default 50)
  - `active_count`, `total_count`: Current counts
  - `version`: Change detection tracking
- **Emits Events**: None
- **Subscribes To**: None
- **Dependencies**: None (pure state container)

#### TrackedMarketsSyncer
- **File**: `services/tracked_markets_syncer.py`
- **Purpose**: REST API sync for tracked market info with periodic refresh (30s)
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management (initial sync on start)
  - `sync_market_info()` - Manual sync trigger
  - `is_healthy()` - Check if running and last sync not stale
  - `get_health_details()` - Get detailed health information
- **Behavior**:
  - Fetches prices for active tracked markets via REST API
  - Detects closed/settled markets via API status
  - Triggers cleanup callback on market close
  - Emits lifecycle events for Activity Feed
- **Emits Events**: `SYSTEM_ACTIVITY` (sync and lifecycle_event types)
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, TrackedMarketsState, EventBus

#### ApiDiscoverySyncer
- **File**: `services/api_discovery_syncer.py`
- **Purpose**: REST API-based market discovery for already-open markets on startup
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management (initial discovery on start)
  - `is_healthy()` - Check if running
  - `get_health_details()` - Get detailed health information
- **Behavior**:
  - Bootstraps lifecycle mode by discovering already-open markets
  - Filters by configured categories (same as EventLifecycleService)
  - Respects capacity limits from TrackedMarketsState
  - Uses EventLifecycleService.track_market_from_api_data() for tracking
  - Periodic refresh every 5 minutes (configurable)
- **Configuration** (constructor parameters):
  - `sync_interval`: Seconds between periodic syncs (default 300)
  - `batch_size`: Maximum markets per API call (default 200)
  - `close_min_minutes`: Skip markets closing within N minutes (default 10)
- **Emits Events**: `SYSTEM_ACTIVITY` (discovery type)
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, EventLifecycleService, TrackedMarketsState, EventBus

#### UpcomingMarketsSyncer
- **File**: `services/upcoming_markets_syncer.py`
- **Purpose**: Syncs markets that will open within configured window (default 4 hours)
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `get_snapshot_message()` - Get upcoming markets for WebSocket broadcast
  - `get_health_details()` - Get detailed health information
- **Emits Events**: `SYSTEM_ACTIVITY` (upcoming_markets type)
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, EventBus

### 3.8 Trading Attachment System (`traderv3/state/`)

The Trading Attachment System links trading state (orders, positions, P&L) to lifecycle-discovered tracked markets for unified UI display.

#### TradingAttachment
- **File**: `state/trading_attachment.py`
- **Purpose**: Attach trading state to lifecycle-discovered tracked markets
- **Key Classes**:
  - `TradingState`: Enum with `MONITORING`, `SIGNAL_READY`, `ORDER_PENDING`, `ORDER_RESTING`, `POSITION_OPEN`, `AWAITING_SETTLEMENT`, `SETTLED`
  - `TrackedMarketOrder`: Order placed in a tracked market
  - `TrackedMarketPosition`: Position in a tracked market
  - `TrackedMarketSettlement`: Final outcome when market settles
  - `TradingAttachment`: Unified view of trading activity per market
- **TradingAttachment Fields**:
  - `ticker`: Market identifier
  - `trading_state`: Current trading lifecycle state
  - `orders`: Dict of TrackedMarketOrder by order_id
  - `position`: Optional TrackedMarketPosition
  - `settlement`: Optional TrackedMarketSettlement
  - `version`: Change detection
- **Key Properties**:
  - `has_exposure`: True if active orders or position
  - `total_pnl`: Realized + unrealized P&L
  - `active_orders`: Orders still pending/resting
- **Data Flow**:
  1. TradingStateSyncer.sync_with_kalshi() -> StateContainer.update_trading_state()
  2. update_trading_state() -> _sync_trading_attachments() (hooks into existing sync)
  3. For each position/order, if market is tracked, update attachment
  4. FillListener.ORDER_FILL -> mark_order_filled_in_attachment() (real-time)
- **Emits Events**: None
- **Subscribes To**: None
- **Dependencies**: None (pure data structure)

### 3.9 RLM Trading Strategy (`traderv3/services/`)

#### RLMService
- **File**: `services/rlm_service.py`
- **Purpose**: Event-driven Reverse Line Movement strategy - bets NO when public bets YES but price drops
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `is_healthy()` - Check if service is running
  - `get_stats()` - Get comprehensive service statistics
  - `get_market_states()` - Get all RLM market states for WebSocket broadcast
  - `get_decision_history(limit)` - Get recent signal decisions
  - `get_health_details()` - Get detailed health information
- **Key Classes**:
  - `MarketTradeState`: Tracks per-market trade accumulation (yes_trades, no_trades, first/last yes price)
  - `RLMSignal`: Detected signal with market_ticker, price_drop, yes_ratio, timestamp
  - `RLMDecision`: Records decisions (executed, skipped, failed) for audit
- **Signal Detection Criteria** (ALL must be true):
  1. Market is in TrackedMarketsState (lifecycle-discovered)
  2. `yes_ratio >= threshold` (default: 0.65 = 65% of trades are YES)
  3. `price_drop >= min_price_drop` (default: 2c = first YES price - last YES price)
  4. `trade_count >= min_trades` (default: 15 trades accumulated)
  5. No existing position in this market
- **Features**:
  - Token bucket rate limiting (configurable trades/minute)
  - Deduplication (one entry per market per session)
  - Trade-by-trade state accumulation from public trades stream
  - Real-time WebSocket broadcasts for frontend visualization
- **Configuration** (environment variables):
  - `RLM_YES_RATIO_THRESHOLD`: Minimum YES trade ratio (default: 0.65)
  - `RLM_MIN_PRICE_DROP`: Minimum price drop in cents (default: 2)
  - `RLM_MIN_TRADES`: Minimum trades before signal (default: 15)
  - `RLM_CONTRACTS`: Contracts per trade (default: 5)
  - `RLM_MAX_TRADES_PER_MINUTE`: Rate limit (default: 10)
- **Emits Events**: `SYSTEM_ACTIVITY` (rlm_signal, rlm_execute types), `RLM_MARKET_UPDATE`, `RLM_TRADE_ARRIVED`
- **Subscribes To**: `PUBLIC_TRADE_RECEIVED`, `MARKET_TRACKED`, `MARKET_DETERMINED`
- **Dependencies**: EventBus, TradingDecisionService, V3StateContainer, TrackedMarketsState

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
| `ORDERBOOK_SNAPSHOT` | OrderbookClient | V3OrderbookIntegration, Yes8090Service | `MarketEvent{market_ticker, sequence_number, timestamp_ms, metadata}` |
| `ORDERBOOK_DELTA` | OrderbookClient | V3OrderbookIntegration, Yes8090Service | `MarketEvent{market_ticker, sequence_number, timestamp_ms, metadata}` |
| `STATE_TRANSITION` | V3StateMachine | (legacy, kept for compat) | `StateTransitionEvent{from_state, to_state, context, metadata}` |
| `TRADER_STATUS` | V3StatusReporter | V3WebSocketManager | `TraderStatusEvent{state, metrics, health, timestamp}` |
| `SYSTEM_ACTIVITY` | V3StateMachine, V3HealthMonitor, TradingFlowOrchestrator, TradingDecisionService, WhaleExecutionService, MarketPriceSyncer, TradingStateSyncer, Yes8090Service, EventLifecycleService, RLMService, ApiDiscoverySyncer | V3WebSocketManager | `SystemActivityEvent{activity_type, message, metadata}` |
| `PUBLIC_TRADE_RECEIVED` | V3TradesIntegration | WhaleTracker, RLMService | `PublicTradeEvent{market_ticker, timestamp_ms, side, price_cents, count}` |
| `WHALE_QUEUE_UPDATED` | WhaleTracker | V3WebSocketManager, WhaleExecutionService | `WhaleQueueEvent{queue, stats, timestamp}` |
| `MARKET_POSITION_UPDATE` | PositionListener | V3StateContainer (via coordinator subscription) | `MarketPositionEvent{market_ticker, position_data, timestamp}` |
| `MARKET_TICKER_UPDATE` | MarketTickerListener | V3Coordinator (updates StateContainer) | `MarketTickerEvent{market_ticker, price_data, timestamp}` |
| `ORDER_FILL` | FillListener | (console UX, future: StateContainer) | `OrderFillEvent{trade_id, order_id, market_ticker, is_taker, side, action, price_cents, count, post_position, fill_timestamp}` |
| `SETTLEMENT` | (via REST sync) | - | Settlements synced from Kalshi REST API, stored in StateContainer |
| `MARKET_LIFECYCLE_EVENT` | V3LifecycleIntegration | EventLifecycleService | `MarketLifecycleEvent{event_type, market_ticker, timestamp}` |
| `MARKET_TRACKED` | EventLifecycleService | RLMService, V3WebSocketManager | `MarketTrackedEvent{market_ticker, category, discovery_source, timestamp}` |
| `MARKET_DETERMINED` | EventLifecycleService | RLMService, V3WebSocketManager | `MarketDeterminedEvent{market_ticker, result, timestamp}` |
| `RLM_MARKET_UPDATE` | RLMService | V3WebSocketManager | `RLMMarketUpdateEvent{market_ticker, yes_trades, no_trades, price_drop, yes_ratio}` |
| `RLM_TRADE_ARRIVED` | RLMService | V3WebSocketManager | `RLMTradeArrivedEvent{market_ticker, side, count, price_cents}` |

### 5.1 SystemActivityEvent Types

| activity_type | Source | Description |
|---------------|--------|-------------|
| `state_transition` | V3StateMachine | State machine state changes |
| `sync` | TradingStateSyncer, MarketPriceSyncer, TrackedMarketsSyncer | Kalshi API sync operations (sync_type: trading_state, market_prices, tracked_markets) |
| `health_check` | V3HealthMonitor | Component health status |
| `trading_cycle` | TradingFlowOrchestrator | Trading cycle start/complete |
| `trading_decision` | TradingDecisionService | Trade execution |
| `trading_error` | TradingDecisionService | Trade execution failures |
| `operation` | V3TradingClientIntegration | Order group operations |
| `cleanup` | V3TradingClientIntegration | Orphaned order cleanup |
| `connection` | V3Coordinator | WebSocket connection events |
| `whale_processing` | WhaleExecutionService | Whale queue processing status (for frontend animation) |
| `whale_follow` | TradingDecisionService | Successful whale follow execution |
| `strategy_start` | Yes8090Service | YES 80-90c strategy started |
| `strategy_stop` | Yes8090Service | YES 80-90c strategy stopped |
| `yes_80_90_signal` | Yes8090Service | YES 80-90c signal detected in orderbook |
| `yes_80_90_execute` | Yes8090Service | YES 80-90c signal execution result (success/failed) |
| `lifecycle_event` | EventLifecycleService | Market lifecycle event (tracked, determined, settled) |
| `discovery` | ApiDiscoverySyncer | API-based market discovery results |
| `rlm_signal` | RLMService | RLM signal detected (price drop + YES bias) |
| `rlm_execute` | RLMService | RLM signal execution result (success/failed) |
| `upcoming_markets` | UpcomingMarketsSyncer | Upcoming markets sync results |

### 5.2 MarketTickerEvent Payload

Price data received from Kalshi WebSocket `ticker` channel:

```json
{
  "event_type": "MARKET_TICKER_UPDATE",
  "market_ticker": "INXD-25JAN03",
  "price_data": {
    "ticker": "INXD-25JAN03",
    "last_price": 52,        // last traded price (cents)
    "yes_bid": 50,           // best yes bid (cents)
    "yes_ask": 54,           // best yes ask (cents)
    "no_bid": 46,            // best no bid (cents)
    "no_ask": 50,            // best no ask (cents)
    "volume": 1500,          // total volume traded
    "open_interest": 12000,  // active contracts
    "timestamp": 1703808000  // unix timestamp (seconds)
  },
  "timestamp": 1703808000.5
}
```

**Note**: MarketTickerListener applies throttling (default 500ms) per ticker to reduce event frequency.

### 5.3 MarketPositionEvent Payload

Position data received from Kalshi WebSocket `market_positions` channel, converted from centi-cents to cents:

```json
{
  "event_type": "MARKET_POSITION_UPDATE",
  "market_ticker": "INXD-25JAN03",
  "position_data": {
    "ticker": "INXD-25JAN03",
    "position": 100,           // contracts (+ YES, - NO)
    "market_exposure": 6500,   // current value in cents (from position_cost)
    "realized_pnl": 0,         // realized P&L in cents
    "fees_paid": 20,           // fees in cents
    "volume": 100,             // total contracts traded
    "last_updated": 1703700000.0
  },
  "timestamp": 1703700000.5
}
```

**Note**: `total_traded` (cost basis) is NOT included in WebSocket updates. It's preserved from REST sync via merge in StateContainer.update_single_position().

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
    f. [if trading] market_ticker_listener.start():
       * Subscribes to position tickers via Kalshi ticker channel
       * Emits MARKET_TICKER_UPDATE events on price changes
    g. [if trading] market_price_syncer.start():
       * Performs initial REST sync of market prices
       * Starts 30s periodic sync loop
    h. [if trading] trading_state_syncer.start():
       * Performs initial trading state sync (balance, positions, orders, settlements)
       * Starts 20s periodic sync loop in dedicated asyncio task
       * Emits console messages: "Trading session synced: X positions, $Y balance, Z settlements"
    i. [if strategy=YES_80_90] yes_80_90_service.start():
       * Subscribes to ORDERBOOK_SNAPSHOT/DELTA events
       * Begins monitoring for 80-90c signals
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
   c. [if yes_80_90] yes_80_90_service.stop()
      - Emits strategy_stop system activity
   d. [if trading] trading_state_syncer.stop()
      - Cancels periodic sync task
   e. [if trading] market_price_syncer.stop()
      - Cancels sync loop task
   f. [if trading] market_ticker_listener.stop()
      - Closes WebSocket, cancels listener task
   g. health_monitor.stop()
   h. status_reporter.stop()
   i. [if trading] trading_client_integration.stop()
      - reset_order_group()
      - disconnect()
   j. orderbook_integration.stop()
      - orderbook_client.stop()
   k. state_machine -> SHUTDOWN
   l. state_machine.stop()
   m. websocket_manager.stop()
      - Disconnect all clients
   n. event_bus.stop()
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

### 6.5 Settlements Data Flow

```
1. KalshiDataSync.sync_with_kalshi() called (every 30s or on startup)
2. Fetches settlements from Kalshi REST API:
   - GET /portfolio/settlements
   - Response: {settlements: [{ticker, yes_count, no_count, yes_total_cost,
                              no_total_cost, revenue, fee_cost, market_result,
                              settled_time}, ...]}
   - Note: fee_cost is STRING in DOLLARS (e.g., "0.34"), others in CENTS
3. TraderState.from_kalshi_data() stores raw settlements array
4. StateContainer.update_trading_state() receives TraderState:
   a. Clears existing _settled_positions deque
   b. For each settlement (last 50):
      i.   Determine side: is_yes_position = yes_count > 0
      ii.  Get count and total_cost for the relevant side
      iii. Parse settled_time ISO string to epoch seconds
      iv.  Convert fee_cost string (dollars) to cents: int(float(fee) * 100)
      v.   Calculate net_pnl = revenue - total_cost - fees_cents
      vi.  Create settlement dict with: ticker, position, side, market_result,
           total_cost, revenue, fees, net_pnl, closed_at
      vii. Append to _settled_positions deque
   c. Update _total_settlements_count
5. StatusReporter.emit_trading_state() broadcasts via WebSocket:
   - Includes: settlements: list(_settled_positions)
   - Includes: settlements_count: total count
6. Frontend receives trading_state message with settlements array
```

**Key Design Decisions:**
- **REST API is Source of Truth**: Settlements synced from REST, not WebSocket events
- **Fee Parsing Fix**: fee_cost is string in dollars, converted to cents
- **Net P&L Calculation**: revenue - cost - fees (not just revenue)
- **Deque Storage**: Last 50 settlements stored for UI display
- **ISO Timestamp Parsing**: settled_time parsed from ISO 8601 to epoch seconds

### 6.6 Real-Time Position Updates Flow

```
1. PositionListener connected to Kalshi WebSocket
2. Subscribed to "market_positions" channel
3. Kalshi sends position update message:
   {type: "market_position", msg: {market_ticker, position, position_cost,
                                   realized_pnl, fees_paid, volume}}
   Note: All monetary values in CENTI-CENTS (1/10000 of dollar)
4. PositionListener._handle_position_message():
   a. Increment positions_received counter
   b. Convert centi-cents to cents (divide by 100)
   c. Create formatted_position dict with:
      - market_exposure (NOT total_traded - this is current value)
      - position, realized_pnl, fees_paid, volume, last_updated
5. Emit MARKET_POSITION_UPDATE via EventBus
6. Coordinator's event handler receives event:
   a. Calls StateContainer.update_single_position(ticker, position_data)
7. StateContainer.update_single_position():
   a. If position == 0: Capture settlement, remove from positions dict
   b. Else: MERGE with existing position (preserves total_traded from REST)
   c. Track ticker in _session_updated_tickers
   d. Recalculate position_count and portfolio_value
   e. Increment trading_state_version
8. StatusReporter detects version change, broadcasts trading_state
9. Frontend receives updated positions with session_updated flag
```

**Key Design Decisions:**
- **Merge, Not Replace**: WebSocket updates merged with existing position to preserve total_traded
- **Session Tracking**: _session_updated_tickers tracks which positions updated via WebSocket
- **Position Closure**: When position == 0, capture settlement data before deletion
- **Immediate Updates**: No cycle delay - position updates broadcast within 1 second

### 6.7 YES 80-90c Strategy Flow (Event-Driven)

```
1. OrderbookClient receives snapshot/delta from Kalshi WebSocket
2. EventBus emits ORDERBOOK_SNAPSHOT or ORDERBOOK_DELTA event
3. Yes8090Service._handle_orderbook() receives event:
   a. Acquire processing lock (prevent concurrent processing)
   b. Skip if already processed this market
   c. Skip if position count >= max_concurrent (100 default)
   d. Skip if market has existing position or open orders
4. Yes8090Service._detect_signal() evaluates orderbook:
   a. Get best YES ask price and size from orderbook
   b. Check: 80c <= yes_ask <= 90c
   c. Check: yes_ask_size >= min_liquidity (10 default)
   d. Check: bid-ask spread <= max_spread (5c default)
   e. Determine tier: A (83-87c) or B (edges)
   f. If all criteria met: create Yes8090Signal
5. Yes8090Service._execute_signal() processes signal:
   a. Check rate limit tokens (10 trades/min default)
   b. If rate limited: return without marking processed (retry later)
   c. Mark market as processed (deduplication)
   d. Calculate entry price based on spread:
      - Tight (<=2c): best_ask - 1c
      - Normal (<=4c): midpoint
      - Wide (>4c): best_bid + 1c
   e. Determine position size: Tier A = 150, Tier B = 100
   f. Create TradingDecision with YES_80_90 strategy
   g. Call TradingDecisionService.execute_decision()
6. TradingDecisionService places order via trading client
7. Yes8090Service records decision in history (executed/failed)
8. System activity events broadcast for frontend visibility
```

**Key Design Decisions:**
- **Event-Driven**: Immediate signal processing on orderbook updates
- **Deduplication**: Each market processed only once per session
- **Rate Limiting**: Token bucket prevents overtrading (10 trades/min)
- **Tier-Based Sizing**: Higher confidence signals (83-87c) get larger positions
- **New Markets Only**: Skips markets with existing positions
- **Hold to Settlement**: No early exit logic in MVP

### 6.8 Market Ticker Price Updates Flow

```
1. MarketTickerListener connected to Kalshi WebSocket
2. Subscribed to "ticker" channel for position tickers
3. Kalshi sends ticker update:
   {type: "ticker", msg: {market_ticker, price, yes_bid, yes_ask,
                          no_bid, no_ask, volume, open_interest, ts}}
4. MarketTickerListener._handle_ticker_message():
   a. Increment updates_received counter
   b. Check throttling (default 500ms per ticker)
   c. If throttled: skip update, increment throttled counter
   d. Extract price data from message
5. Emit MARKET_TICKER_UPDATE via EventBus
6. Coordinator._handle_market_ticker_update() receives event:
   a. Calls StateContainer.update_market_price(ticker, price_data)
7. StateContainer.update_market_price():
   a. Create/update MarketPriceData for ticker
   b. Increment _market_prices_version
8. StatusReporter includes market prices in trading_state broadcast
```

**Key Design Decisions:**
- **Filtered Subscription**: Only subscribes to position tickers (not all markets)
- **Dynamic Subscriptions**: Tickers added/removed as positions change
- **Throttled Updates**: Default 500ms per ticker reduces event volume
- **Redundancy with REST**: MarketPriceSyncer provides fallback pricing

### 6.9 Lifecycle Market Discovery Flow

```
1. [Kalshi WS] market_lifecycle event (created)
        ↓
2. [LifecycleClient] Receives event, calls callback
        ↓
3. [V3LifecycleIntegration] emit_market_lifecycle() to EventBus
        ↓
4. [EventLifecycleService] _handle_lifecycle_event()
        ↓ Check capacity (fast fail if at limit)
        ↓ REST lookup GET /markets/{ticker}
        ↓ Category filter (LIFECYCLE_CATEGORIES)
        ↓
5. [TrackedMarketsState] add_market() if passes filters
        ↓
6. [RLDatabase] insert_tracked_market() for persistence
        ↓
7. [EventBus] emit_market_tracked()
        ↓
8. [EventLifecycleService] Call orderbook subscribe callback
        ↓
9. [V3OrderbookIntegration] Subscribe to market orderbook
        ↓
10. [V3WebSocketManager] broadcast_tracked_markets() + broadcast_lifecycle_event()
         ↓
11. [Frontend] useLifecycleWebSocket receives tracked_markets and lifecycle_event
```

**Key Design Decisions:**
- **Category Filtering**: Only track markets in configured categories (sports, crypto, etc.)
- **Capacity Management**: TrackedMarketsState enforces max market limit (default 50)
- **DB Persistence**: Markets persisted for session recovery
- **Orderbook Subscription**: Dynamic orderbook subscription on market tracking
- **Dual Broadcast**: Both full snapshot and activity event for frontend

### 6.10 RLM Signal Detection Flow

```
1. [Kalshi WS] Public trade arrives
        ↓
2. [TradesClient] trade callback
        ↓
3. [V3TradesIntegration] emit_public_trade() to EventBus
        ↓
4. [RLMService] _handle_public_trade()
        ↓ Check if market is in TrackedMarketsState
        ↓ (Skip if not tracked - only lifecycle markets)
        ↓
5. [MarketTradeState] Accumulate trade
        - Increment yes_trades or no_trades
        - Track first_yes_price and last_yes_price
        - Calculate price_drop = first - last
        - Calculate yes_ratio = yes_trades / total_trades
        ↓
6. [EventBus] emit_rlm_trade_arrived() (for animation pulse)
        ↓
7. [EventBus] emit_rlm_market_update() (state change)
        ↓
8. [RLMService] Check signal criteria:
        - yes_ratio >= threshold (default 0.65)
        - price_drop >= min_price_drop (default 2c)
        - trade_count >= min_trades (default 15)
        - No existing position in market
        ↓
9. If signal detected: Create TradingDecision (action=BUY, side=NO)
        ↓
10. [TradingDecisionService] execute_decision()
        ↓
11. [V3WebSocketManager] broadcast rlm_market_state and rlm_trade_arrived
         ↓
12. [Frontend] Updates rlmStates and tradePulses in useLifecycleWebSocket
```

**Key Design Decisions:**
- **Lifecycle-Only Markets**: RLM only monitors tracked markets (not all markets)
- **Trade-by-Trade Accumulation**: State built from individual trades, not snapshots
- **Contrarian Signal**: YES bias + price drop = bet NO (fade the crowd)
- **Immediate Trade Pulses**: rlm_trade_arrived bypasses coalescing for animation
- **Deduplication**: One signal per market per session

### 6.11 API Discovery Bootstrap Flow

```
1. [System Startup] ApiDiscoverySyncer.start()
        ↓
2. [ApiDiscoverySyncer] Fetch open markets from Kalshi REST API
        - GET /markets with status=active filter
        - Batch size: 200 markets per call
        ↓
3. For each market in response:
        ↓ Category filter (LIFECYCLE_CATEGORIES)
        ↓ Skip if close_ts < now + close_min_minutes
        ↓ Check TrackedMarketsState capacity
        ↓
4. [EventLifecycleService] track_market_from_api_data()
        - Bypasses REST lookup (already have market data)
        ↓
5. [TrackedMarketsState] add_market()
        ↓
6. [RLDatabase] insert_tracked_market()
        ↓
7. Continue lifecycle WebSocket monitoring for new markets
```

**Key Design Decisions:**
- **Bootstrap Already-Open Markets**: Discovers markets that opened before system started
- **Same Category Filter**: Uses LIFECYCLE_CATEGORIES for consistency
- **Capacity Respect**: Stops adding when TrackedMarketsState at capacity
- **Periodic Refresh**: Re-runs every 5 minutes to catch missed markets
- **Skip Expiring Soon**: Ignores markets closing within close_min_minutes

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
| `YES8090_MIN_PRICE` | No | `80` | Minimum YES ask price for signals (cents) |
| `YES8090_MAX_PRICE` | No | `90` | Maximum YES ask price for signals (cents) |
| `YES8090_MIN_LIQUIDITY` | No | `10` | Minimum contracts at best ask |
| `YES8090_MAX_SPREAD` | No | `5` | Maximum bid-ask spread (cents) |
| `YES8090_CONTRACTS` | No | `100` | Default contracts per trade |
| `YES8090_TIER_A_CONTRACTS` | No | `150` | Contracts for Tier A signals (83-87c) |
| `YES8090_MAX_CONCURRENT` | No | `100` | Maximum concurrent positions |
| `LIFECYCLE_CATEGORIES` | No | `sports,crypto,entertainment,media_mentions` | Comma-separated category filter for discovery |
| `LIFECYCLE_CAPACITY` | No | `50` | Maximum tracked markets |
| `RLM_YES_RATIO_THRESHOLD` | No | `0.65` | Minimum YES trade ratio for RLM signal |
| `RLM_MIN_PRICE_DROP` | No | `2` | Minimum price drop in cents for RLM signal |
| `RLM_MIN_TRADES` | No | `15` | Minimum trades before RLM signal evaluation |
| `RLM_CONTRACTS` | No | `5` | Contracts per RLM trade |
| `RLM_MAX_TRADES_PER_MINUTE` | No | `10` | RLM rate limit (trades/minute) |

### 7.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v3/health` | GET | Quick health check |
| `/v3/status` | GET | Detailed system status |
| `/v3/cleanup` | POST | Cancel orphaned orders (`?orphaned_only=true/false`) |
| `/v3/ws` | WebSocket | Real-time updates |

### 7.3 WebSocket Message Types (to frontend)

| Type | Description | Coalescing |
|------|-------------|------------|
| `connection` | Initial connection acknowledgment | Immediate |
| `history_replay` | Historical state transitions for late joiners | Immediate |
| `system_activity` | Unified console messages | Immediate |
| `state_transition` | State machine state changes | Immediate |
| `whale_processing` | Whale processing status for frontend animation | Immediate |
| `lifecycle_event` | Real-time lifecycle event (tracked, determined, etc.) | Immediate |
| `trader_status` | Periodic status/metrics update | Coalesced (100ms) |
| `trading_state` | Trading data (balance, positions, orders, P&L, settlements) | Coalesced (100ms) |
| `whale_queue` | Whale detection queue update (when enabled) | Coalesced (100ms) |
| `tracked_markets` | Full snapshot of tracked markets with stats | Coalesced (100ms) |
| `market_info_update` | Incremental market price/volume update | Coalesced (100ms) |
| `rlm_states_snapshot` | Initial snapshot of all RLM market states (on connect) | Immediate |
| `rlm_market_state` | Real-time RLM market state update | Coalesced (100ms) |
| `rlm_trade_arrived` | Trade pulse for RLM animation | Immediate |
| `trade_processing` | Trade processing stats (1.5s heartbeat) | Coalesced (100ms) |
| `upcoming_markets` | Markets opening within 4 hours | Coalesced (100ms) |
| `ping` | Keep-alive ping | Immediate |

**Coalescing Behavior**: Messages marked "Coalesced" are batched in a 100ms window. Only the latest message of each type is sent, reducing frontend re-renders during high-frequency updates. Critical messages (Immediate) bypass coalescing for real-time responsiveness.

#### trading_state Message Format

Sent immediately when a client connects via `handle_websocket()` (before waiting for periodic broadcasts) and on every trading sync cycle via `emit_trading_state()`. The immediate send on connect uses `get_trading_summary()` from StateContainer to provide clients with current state without delay.

```json
{
  "type": "trading_state",
  "data": {
    "timestamp": 1703700000.0,
    "version": 42,
    "balance": 100000,                    // cents
    "portfolio_value": 25000,             // cents
    "position_count": 3,
    "order_count": 2,
    "positions": ["INXD-25JAN03", "KXBTC-25JAN03"],  // ticker list
    "open_orders": 2,
    "order_list": [                       // formatted for display
      {"order_id": "abc12345", "ticker": "INXD-25JAN03", "side": "yes",
       "action": "buy", "price": 65, "count": 5, "status": "resting"}
    ],
    "sync_timestamp": 1703699970.5,
    "changes": {                          // optional, from last sync
      "balance": 0, "portfolio_value": 100, "positions": 0, "orders": 0
    },
    "order_group": {                      // optional
      "id": "group123", "status": "active", "order_count": 2, "order_ids": ["abc1", "def2"]
    },
    "pnl": {                              // session P&L breakdown
      "session_pnl": 1500,                // total session P&L (cents)
      "realized_pnl": 500,                // closed positions (cents)
      "unrealized_pnl": 1000,             // open positions (cents)
      "starting_equity": 123500,          // session start equity (cents)
      "current_equity": 125000,           // current equity (cents)
      "session_duration_minutes": 45.5
    },
    "positions_details": [                // per-position P&L
      {"ticker": "INXD-25JAN03", "position": 10, "side": "yes",
       "total_traded": 6000, "market_exposure": 6500,
       "realized_pnl": 0, "unrealized_pnl": 500, "fees_paid": 20,
       "session_updated": true, "last_updated": 1703700000.0}
    ],
    "position_listener": {                // real-time updates health
      "connected": true, "positions_received": 15, "positions_processed": 15,
      "last_update": 1703700000.0, "connection_count": 1
    },
    "settlements": [                      // last 50 settlements
      {"ticker": "INXD-25DEC31", "position": 5, "side": "yes",
       "market_result": "yes", "total_cost": 3000, "revenue": 500,
       "fees": 10, "net_pnl": -2510, "closed_at": 1703600000.0}
    ],
    "settlements_count": 12               // total settlements
  }
}
```

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

#### tracked_markets Message Format

Sent on client connect and when tracked markets state changes.

```json
{
  "type": "tracked_markets",
  "data": {
    "markets": [
      {
        "ticker": "KXNFL-25JAN05-DET",
        "event_ticker": "KXNFL-25JAN05",
        "title": "Will Detroit Lions win?",
        "category": "Sports",
        "status": "active",
        "created_ts": 1704067200,
        "open_ts": 1704067200,
        "close_ts": 1704153600,
        "tracked_at": 1704067500.5,
        "price": 65,
        "volume": 12500,
        "volume_24h": 3200,
        "open_interest": 5400,
        "yes_bid": 64,
        "yes_ask": 66,
        "discovery_source": "lifecycle_ws",
        "time_to_close_seconds": 86400,
        "trading": {
          "ticker": "KXNFL-25JAN05-DET",
          "trading_state": "position_open",
          "orders": [],
          "position": {"contracts": 5, "side": "no", "entry_price": 35},
          "settlement": null,
          "version": 3
        }
      }
    ],
    "stats": {
      "tracked": 15,
      "capacity": 50,
      "total": 18,
      "by_category": {"Sports": 10, "Crypto": 5},
      "by_status": {"active": 15, "determined": 2, "settled": 1},
      "determined_today": 3,
      "tracked_total": 25,
      "rejected_capacity": 0,
      "rejected_category": 12,
      "version": 42
    },
    "version": 42,
    "timestamp": 1704067500.5
  }
}
```

#### lifecycle_event Message Format

```json
{
  "type": "lifecycle_event",
  "data": {
    "event_type": "created",
    "market_ticker": "KXNFL-25JAN05-DET",
    "action": "tracked",
    "reason": "api_discovery",
    "metadata": {
      "category": "Sports",
      "discovery_source": "api"
    },
    "timestamp": "12:34:56"
  }
}
```

#### rlm_market_state Message Format

```json
{
  "type": "rlm_market_state",
  "data": {
    "market_ticker": "KXNFL-25JAN05-DET",
    "yes_trades": 45,
    "no_trades": 12,
    "total_trades": 57,
    "yes_ratio": 0.789,
    "first_yes_price": 68,
    "last_yes_price": 62,
    "price_drop": 6,
    "position_contracts": 0,
    "signal_trigger_count": 0,
    "timestamp": "12:34:56"
  }
}
```

#### rlm_trade_arrived Message Format

```json
{
  "type": "rlm_trade_arrived",
  "data": {
    "market_ticker": "KXNFL-25JAN05-DET",
    "side": "yes",
    "count": 15,
    "price_cents": 62,
    "timestamp": "12:34:56"
  }
}
```

#### trade_processing Message Format (1.5s heartbeat)

```json
{
  "type": "trade_processing",
  "data": {
    "recent_trades": [
      {
        "trade_id": "abc123",
        "market_ticker": "KXNFL-25JAN05-DET",
        "side": "yes",
        "price_cents": 62,
        "count": 15,
        "timestamp": 1704067500.5,
        "age_seconds": 5
      }
    ],
    "stats": {
      "trades_seen": 1500,
      "trades_filtered": 1400,
      "trades_tracked": 100,
      "filter_rate_percent": 93.3
    },
    "decisions": {
      "detected": 5,
      "executed": 3,
      "rate_limited": 1,
      "skipped": 1,
      "reentries": 0,
      "low_balance": 0
    },
    "decision_history": [],
    "last_updated": 1704067500.5,
    "timestamp": "12:34:56"
  }
}
```

### 7.4 Frontend Panels (V3TraderConsole.jsx)

The frontend receives `trading_state` messages and displays data in specialized panels:

#### PositionListPanel
Displays open positions with per-contract P&L calculations.

**Columns:**
| Column | Source Field | Description |
|--------|--------------|-------------|
| Ticker | `positions_details[].ticker` | Market ticker |
| Side | `positions_details[].side` | "YES" or "NO" badge |
| Qty | `positions_details[].position` | Contract count (absolute) |
| Cost/C | `total_traded / position` | Cost per contract in cents |
| Value/C | `market_exposure / position` | Current value per contract |
| Unreal/C | `unrealized_pnl / position` | Unrealized P&L per contract |
| P&L | `positions_details[].unrealized_pnl` | Total unrealized P&L |
| Updated | `positions_details[].last_updated` | Last update timestamp |

**Features:**
- Real-time update indicators (green dot for recently changed)
- YES/NO section headers with aggregate totals
- Position listener status badge (Live/Polling)
- Session updates count

#### SettlementsPanel
Displays closed positions with final P&L economics.

**Columns:**
| Column | Source Field | Description |
|--------|--------------|-------------|
| Ticker | `settlements[].ticker` | Market ticker |
| Side | `settlements[].side` | "YES" or "NO" badge |
| Qty | `settlements[].position` | Contract count (absolute) |
| Cost | `settlements[].total_cost` | Total cost in cents |
| Payout | `settlements[].revenue` | Revenue received in cents |
| Fees | `settlements[].fees` | Fees paid in cents |
| Net P&L | `settlements[].net_pnl` | Net P&L (revenue - cost - fees) |
| Closed | `settlements[].closed_at` | Close timestamp |

**Features:**
- Collapsible with expand/collapse toggle
- Total Net P&L header showing aggregate across all settlements
- Green/red styling for profit/loss
- Toast notifications for new settlements

### 7.5 Frontend WebSocket Hook (useLifecycleWebSocket.js)

#### Purpose
React hook for managing Lifecycle Discovery WebSocket connection, handling lifecycle-specific message types and RLM state updates.

#### Location
`frontend/src/hooks/useLifecycleWebSocket.js`

#### Returned State
```javascript
{
  // Connection state
  wsStatus: 'disconnected' | 'connected' | 'error',
  isConnected: boolean,
  lastUpdateTime: number,

  // Tracked markets
  trackedMarkets: {
    markets: TrackedMarket[],
    stats: TrackedMarketsStats,
    version: number
  },
  markets: TrackedMarket[],           // Convenience alias
  stats: TrackedMarketsStats,          // Convenience alias
  isAtCapacity: boolean,

  // Lifecycle events (Activity Feed)
  recentEvents: LifecycleEvent[],
  clearEvents: () => void,

  // RLM (Reverse Line Movement) state
  rlmStates: Record<string, RLMMarketState>,
  tradePulses: Record<string, { side: 'yes'|'no', ts: number }>,

  // Upcoming markets
  upcomingMarkets: UpcomingMarket[],

  // Trading state
  tradingState: { balance: number, min_trader_cash: number }
}
```

#### Message Handlers
| Message Type | Handler | State Updated |
|--------------|---------|---------------|
| `tracked_markets` | Full snapshot | `trackedMarkets`, creates startup event |
| `lifecycle_event` | Add to feed | `recentEvents` (with deduplication) |
| `market_info_update` | Incremental | `trackedMarkets.markets[].price/volume` |
| `rlm_states_snapshot` | Initial load | `rlmStates` |
| `rlm_market_state` | Real-time | `rlmStates[ticker]` |
| `rlm_trade_arrived` | Animation | `tradePulses[ticker]` (auto-clears 1.5s) |
| `upcoming_markets` | Schedule | `upcomingMarkets` |
| `trading_state` | Balance | `tradingState` |
| `ping` | Heartbeat | Responds with `pong` |

#### Zombie Connection Detection
- Server sends pings every 30s
- Frontend monitors 60s timeout
- Heartbeat check every 15s
- Forces reconnection on stale connections

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
7. **Message Coalescing**: Backend batches frequent WebSocket messages (100ms window) to reduce frontend re-renders. Critical messages (state transitions, whale processing) bypass coalescing for responsiveness
8. **Zombie Connection Detection**: Frontend monitors WebSocket heartbeats (60s timeout, 15s check interval) and forces reconnection on stale connections. Reconnect timeout cleanup prevents timer accumulation

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
| 2024-12-28 | Settlements panel fixes: fee_cost parsing (string dollars to cents), net P&L calculation (revenue-cost-fees) | Claude |
| 2024-12-28 | PositionListener fix: Handle .closed attribute safely with fallback for different WS library versions | Claude |
| 2024-12-28 | Immediate trading state on client connect: trading_state broadcast within 1 second via version monitoring | Claude |
| 2024-12-28 | Comprehensive docs update: StateContainer (settlements, P&L, whale queue), StatusReporter (broadcast data), KalshiDataSync (sync data), demo_client (all methods), trading_state message format, settlements/position data flows | Claude |
| 2024-12-28 | Added YES_80_90 strategy to TradingStrategy enum (validated +5.1% edge) | Claude |
| 2024-12-28 | Added Section 7.4: Frontend Panels documentation (PositionListPanel, SettlementsPanel columns and features) | Claude |
| 2024-12-29 | Comprehensive architecture update: Added MarketTickerListener, MarketPriceSyncer, Yes8090Service | Claude |
| 2024-12-29 | Added MARKET_TICKER_UPDATE event, MarketTickerEvent payload (Section 5.2), YES8090_* env vars | Claude |
| 2024-12-29 | Updated architecture diagram with new components (ticker WS, price syncer, 80-90c strategy) | Claude |
| 2024-12-29 | Added StateContainer market prices state (MarketPriceData fields and methods) | Claude |
| 2024-12-29 | Added data flow traces: Section 6.7 YES 80-90c Strategy, Section 6.8 Market Ticker Updates | Claude |
| 2024-12-29 | Updated startup/shutdown sequences with new component lifecycle | Claude |
| 2024-12-29 | Added TradingStateSyncer service for reliable periodic trading state sync (20s interval) | Claude |
| 2024-12-29 | WebSocket performance: Added message coalescing (100ms batching), zombie connection detection (60s timeout) | Claude |
| 2024-12-29 | Added FillListener component for real-time order fill notifications via Kalshi fill WebSocket channel | Claude |
| 2025-01-01 | **Comprehensive lifecycle discovery update**: Added Section 3.7 (Lifecycle Discovery System), Section 3.8 (Trading Attachment System), Section 3.9 (RLM Service) | Claude |
| 2025-01-01 | Added LifecycleClient, V3LifecycleIntegration, EventLifecycleService, TrackedMarketsState, TrackedMarketsSyncer, ApiDiscoverySyncer, UpcomingMarketsSyncer | Claude |
| 2025-01-01 | Added TradingAttachment state for per-market trading data (orders, positions, settlements) | Claude |
| 2025-01-01 | Added RLMService for Reverse Line Movement strategy (+17.38% validated edge) | Claude |
| 2025-01-01 | Added 5 new events: MARKET_LIFECYCLE_EVENT, MARKET_TRACKED, MARKET_DETERMINED, RLM_MARKET_UPDATE, RLM_TRADE_ARRIVED | Claude |
| 2025-01-01 | Added 8 new WebSocket message types: tracked_markets, lifecycle_event, market_info_update, rlm_states_snapshot, rlm_market_state, rlm_trade_arrived, trade_processing, upcoming_markets | Claude |
| 2025-01-01 | Added data flow traces: Section 6.9 (Lifecycle Discovery), Section 6.10 (RLM Signal Detection), Section 6.11 (API Discovery Bootstrap) | Claude |
| 2025-01-01 | Added Section 7.5: Frontend WebSocket Hook (useLifecycleWebSocket.js) documentation | Claude |
| 2025-01-01 | Updated architecture diagram with lifecycle components (RLMService, EventLifecycleSvc, TrackedMarketsState, etc.) | Claude |
| 2025-01-01 | Added LIFECYCLE_* and RLM_* environment variables | Claude |

## 10. Cleanup Recommendations

Identified architectural issues and cleanup opportunities from Dec 2024 review:

### 10.1 High Priority

1. **MarketPriceSyncer Direct State Access** (`services/market_price_syncer.py:185-191`)
   - **Issue**: Directly accesses `_market_prices` and `_market_prices_version` private attributes
   - **Fix**: Use `update_market_price()` method or add `batch_update_market_prices()` method

2. **WhaleExecutionService Memory Growth** (`services/whale_execution_service.py`)
   - **Issue**: `_evaluated_whale_ids` set grows unbounded over session lifetime
   - **Fix**: Implement periodic cleanup or use TTL-based cache (e.g., clear entries older than 1 hour)

### 10.2 Medium Priority

1. **Type Hint Inconsistency** (`core/websocket_manager.py:28`)
   - **Issue**: Imports `StateContainer` but actual class is `V3StateContainer`
   - **Fix**: Update import to `from .state_container import V3StateContainer`

2. **Inconsistent Health Check Patterns**
   - **Issue**: Services have different health reporting interfaces
   - **Fix**: Standardize on both `is_healthy()` and `get_health_details()` for all components

3. ~~**Test Files in Main Package**~~ ✅ RESOLVED 2025-01-01
   - **Files**: Moved to `backend/tests/traderv3/`

### 10.3 Low Priority

1. **Empty `__init__.py` Files**
   - All package `__init__.py` files are empty
   - **Fix**: Consider adding `__all__` exports for better IDE support

2. **Frontend Duplicate Components**
   - **Issue**: Multiple WhaleQueuePanel implementations in frontend
   - **Fix**: Consolidate to single implementation

### 10.4 Documentation Notes

1. **SYSTEM_ACTIVITY Event Sources**: Updated to include MarketPriceSyncer and Yes8090Service
2. **Event Catalog Accuracy**: SETTLEMENT is synced via REST, not real-time events (clarified in docs)
