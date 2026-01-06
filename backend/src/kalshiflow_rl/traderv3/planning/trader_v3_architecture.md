# TRADER V3 Architecture Documentation

> Machine-readable architecture reference for coding agents.
> Last updated: 2025-01-06 (added strategy attribution system, updated state container caches)

---

## 1. System Overview

TRADER V3 is an event-driven paper trading system for Kalshi prediction markets. It uses WebSocket connections to receive real-time orderbook data and lifecycle events, maintains state through a centralized container with version tracking, and coordinates trading decisions through a clean component architecture. All trading data is in CENTS (Kalshi's native unit).

**Current Status**: Production-ready with lifecycle-based market discovery, RLM trading strategy, real-time WebSocket broadcasting, comprehensive state management, and strategy attribution for multi-strategy P&L tracking.

**Trading Strategies**:
- **HOLD**: Never trade (safe default)
- **RLM_NO**: Reverse Line Movement - bet NO when public bets YES but price drops (+17.38% validated edge)

**Market Discovery Modes**:
- **Config Mode**: Static market list from `RL_MARKET_TICKERS` environment variable
- **Discovery Mode**: Dynamic market discovery via Lifecycle Discovery System with category filtering and capacity management

**Key Architectural Features**:
- **Lifecycle Discovery**: Real-time market discovery via Kalshi `market_lifecycle` WebSocket channel
- **TrackedMarketsState**: Centralized in-memory state for discovered markets (no DB persistence)
- **Trading Attachments**: Per-market trading state linking positions/orders to tracked markets
- **Real-time Prices**: MarketTickerListener (WebSocket) + MarketPriceSyncer (REST fallback)
- **Strategy Attribution**: Per-order/position strategy tracking for multi-strategy P&L analysis
- **Order Context Service**: Captures trade-time context for quant analysis

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
|    +---------+---------+---------+--------+--------+---------+---------+                                            |
|    |                   |         |                 |         |         |                                            |
|    v                   v         v                 v         v         v                                            |
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
|           |                     |                              |                         |                         |
|           |                     |                     +--------+----------+              |                         |
|           |                     |                     |OrderContextService|              |                         |
|           |                     |                     | (quant capture)   |              |                         |
|           |                     |                     +-------------------+              |                         |
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
- **Managed Components**:
  - `_state_machine`: V3StateMachine
  - `_event_bus`: EventBus
  - `_websocket_manager`: V3WebSocketManager
  - `_orderbook_integration`: V3OrderbookIntegration
  - `_trading_client_integration`: Optional V3TradingClientIntegration
  - `_state_container`: V3StateContainer
  - `_health_monitor`: V3HealthMonitor
  - `_status_reporter`: V3StatusReporter
  - `_listener_bootstrap`: ListenerBootstrapService
  - `_trading_service`: Optional TradingDecisionService
  - `_trading_orchestrator`: Optional TradingFlowOrchestrator
  - `_strategy_coordinator`: Optional StrategyCoordinator
  - `_rlm_service`: Optional RLMService (deprecated, use strategy_coordinator)
  - `_tmo_fetcher`: Optional TrueMarketOpenFetcher
  - `_position_listener`: Optional PositionListener
  - `_market_ticker_listener`: Optional MarketTickerListener
  - `_market_price_syncer`: Optional MarketPriceSyncer
  - `_trading_state_syncer`: Optional TradingStateSyncer
  - `_fill_listener`: Optional FillListener
  - `_lifecycle_client`: Optional LifecycleClient
  - `_lifecycle_integration`: Optional V3LifecycleIntegration
  - `_tracked_markets_state`: Optional TrackedMarketsState
  - `_event_lifecycle_service`: Optional EventLifecycleService
  - `_lifecycle_syncer`: Optional TrackedMarketsSyncer
  - `_upcoming_markets_syncer`: Optional UpcomingMarketsSyncer
  - `_api_discovery_syncer`: Optional ApiDiscoverySyncer
- **Emits Events**: None directly (delegates to StatusReporter)
- **Subscribes To**: None directly

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
  - `emit_public_trade()` - Public trade received
  - `emit_market_lifecycle()` - Lifecycle event
  - `emit_market_tracked()` / `emit_market_determined()` - Discovery events
  - `emit_rlm_market_update()` / `emit_rlm_trade_arrived()` - RLM events
  - `emit_order_fill()` - Order fill notification
  - `emit_market_position_update()` / `emit_market_ticker_update()` - Position/price events
- **Emits Events**: Distributes all event types
- **Subscribes To**: N/A (is the event distributor)
- **Dependencies**: None

#### V3StateContainer
- **File**: `core/state_container.py`
- **Purpose**: Centralized state storage with versioning, change detection, atomic updates, and strategy attribution
- **Key Methods**:
  - `update_trading_state(state, changes)` - Update trading data from Kalshi sync (returns bool if changed)
  - `update_single_position(ticker, position_data)` - Update single position from real-time WebSocket push
  - `update_component_health(name, healthy, details)` - Update component health
  - `update_machine_state(state, context, metadata)` - Update state machine reference
  - `get_full_state()` - Get complete state snapshot
  - `get_trading_summary(order_group_id)` - Get trading state for WebSocket broadcast (includes settlements, P&L, strategy attribution)
  - `initialize_session_pnl(balance, portfolio_value)` - Initialize session P&L tracking on first sync
  - `record_order_fill(cost_cents, contracts)` - Record cash spent on fill
  - `record_ttl_cancellation(count)` - Record TTL-cancelled orders
  - `set_component_degraded(component, is_degraded, reason)` - Track degraded components
  - `atomic_update(update_func)` - Thread-safe atomic state update
  - `compare_and_swap(expected_version, update_func)` - CAS operation
  - `get_or_create_trading_attachment(ticker)` - Get/create trading attachment for market
  - `update_order_in_attachment(ticker, order_id, order_data)` - Track order in attachment
  - `mark_order_filled_in_attachment(ticker, order_id, fill_count, fill_price)` - Mark fill in attachment
  - `remove_order(order_id)` - Remove filled order from state immediately
  - `cleanup_market(ticker)` - Cleanup state for determined/settled market
- **Key State Types**:
  - `_trading_state`: TraderState with positions, orders, balance, settlements
  - `_session_pnl_tracker`: SessionPnLTracker for P&L tracking from session start
  - `_settled_positions`: Deque of last 500 settlements (synced from REST API)
  - `_session_updated_tickers`: Set of tickers updated via WebSocket this session
  - `_market_prices`: Dict[str, MarketPriceData] for real-time market prices
  - `_trading_attachments`: Dict[str, TradingAttachment] per-market trading state
  - `_component_health`: Dict[str, ComponentHealth] for health tracking
- **Strategy Attribution Caches** (new):
  - `_settlement_strategy_cache`: Dict[str, str | None] - Maps ticker to strategy_id from order_contexts DB for settlements
  - `_position_strategy_cache`: Dict[str, str | None] - Maps ticker to strategy_id from order_contexts DB for positions
  - These caches prevent N+1 DB queries when formatting positions/settlements for display
  - Cache persists across sync cycles; populated via `_batch_lookup_strategies_from_db()`
- **Strategy Lookup Flow**:
  1. Check cache first (includes DB results from previous lookups)
  2. Check in-memory TradingAttachment (has strategy_id from filled orders)
  3. Batch query order_contexts table for remaining unknown tickers
  4. Cache results (including None for tickers not found)
- **MarketPriceData Fields**:
  - `ticker`: Market ticker string
  - `last_price`: Last traded price (cents)
  - `yes_bid`, `yes_ask`: YES side bid/ask prices (cents)
  - `no_bid`, `no_ask`: NO side bid/ask prices (cents)
  - `volume`: Total volume traded
  - `open_interest`: Active contracts
  - `close_time`: Market close time (ISO timestamp, from REST only)
  - `timestamp`: Update timestamp
  - `price_source`: "ws_ticker" or "rest_sync"
  - `last_ws_update_time`: Unix timestamp of last WS update
- **Market Price Methods**:
  - `update_market_price(ticker, price_data)` - Update single ticker price
  - `get_market_price(ticker)` - Get price for ticker
  - `get_all_market_prices()` - Get all market prices
  - `clear_market_price(ticker)` - Remove ticker from prices
  - `get_market_prices_summary()` - Get summary for WebSocket broadcast
- **Emits Events**: None
- **Subscribes To**: `MARKET_POSITION_UPDATE` (via coordinator subscription)
- **Dependencies**: TraderState, StateChange, SessionPnLTracker, TradingAttachment, OrderContextService

#### V3HealthMonitor
- **File**: `core/health_monitor.py`
- **Purpose**: Monitors component health and triggers recovery from ERROR states
- **Key Methods**:
  - `start()` / `stop()` - Start/stop monitoring loop
  - `get_status()` - Get monitor status
  - `is_healthy()` - Check if monitor itself is healthy
- **Emits Events**: `SYSTEM_ACTIVITY` (health_check type)
- **Subscribes To**: None (polls components directly)
- **Dependencies**: V3Config, V3StateMachine, EventBus, V3WebSocketManager, V3StateContainer, V3OrderbookIntegration, V3TradingClientIntegration (optional), V3TradesIntegration (optional), FillListener (optional)

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
  - `positions_details`: Per-position P&L with unrealized_pnl, realized_pnl, fees_paid, strategy_id
  - `pnl`: Session P&L breakdown (realized, unrealized, total)
  - `settlements`: Last 500 settlements with net_pnl calculation and strategy_id
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
  - `set_trading_service(service)` - Set trading service for order actions
  - `set_state_container(container)` - Set state container for immediate state on connect
- **Message Coalescing**:
  - 100ms batching window for frequent message types
  - **Immediate (critical)**: `state_transition`, `connection`, `system_activity`, `history_replay`
  - **Coalesced (frequent)**: `trading_state`, `trader_status`, `tracked_markets`, `rlm_market_state`
  - Coalescing keeps only the latest message of each type, reducing frontend re-renders
- **Key Fields**: `_pending_messages`, `_coalesce_task`, `_coalesce_interval`
- **Emits Events**: None
- **Subscribes To**: `SYSTEM_ACTIVITY`, `TRADER_STATUS`
- **Dependencies**: EventBus, V3StateMachine (optional), TradingDecisionService (optional), V3StateContainer (optional), Starlette WebSocket

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
- **Purpose**: Wraps OrderbookClient for V3, tracks metrics, connection state, and signal aggregation
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `wait_for_connection(timeout)` - Wait for WebSocket connection
  - `wait_for_first_snapshot(timeout)` - Wait for initial data
  - `get_metrics()` - Get orderbook metrics (includes signal aggregator stats)
  - `get_health_details()` - Get detailed health info
  - `ensure_session_for_recovery()` - Prepare for ERROR recovery
  - `get_orderbook_signals(ticker)` - Get live 10-second bucket signal data for a market
  - `set_trading_client(client)` - Inject trading client for REST fallback
  - `is_connection_healthy(threshold)` - Check WS connection health (message-based)
  - `refresh_orderbook_if_stale(ticker)` - REST fallback only on connection-level degradation
- **Signal Aggregation**: Auto-creates `OrderbookSignalAggregator` on first snapshot (lazy init)
  - Aggregates snapshots into 10-second buckets
  - Tracks spread OHLC, volume imbalance, BBO depth, delta count, large orders
  - Persists to `orderbook_signals` table (~1.7KB/market/day)
- **Emits Events**: None (listens to OrderbookClient events)
- **Subscribes To**: `ORDERBOOK_SNAPSHOT`, `ORDERBOOK_DELTA`
- **Dependencies**: OrderbookClient, EventBus, OrderbookSignalAggregator

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
  - `get_market(ticker)` - Fetch market info via REST
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
  - `get_market(ticker)` - Get single market info
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
- **Emits Events**: `MARKET_TICKER_UPDATE` (via EventBus.emit_market_ticker_update)
- **Subscribes To**: Kalshi WebSocket `ticker` channel
- **Dependencies**: EventBus, KalshiAuth

#### PositionListener
- **File**: `clients/position_listener.py`
- **Purpose**: WebSocket listener for real-time position updates from Kalshi
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `is_healthy()` - Check if listener is running and WebSocket connected
  - `get_metrics()` - Get listener statistics
  - `get_status()` - Get full status for monitoring
  - `get_health_details()` - Get detailed health information
- **Features**:
  - Subscribes to Kalshi `market_positions` WebSocket channel
  - Automatic reconnection with configurable delay
  - Heartbeat monitoring for connection health
  - Centi-cents to cents conversion (Kalshi API uses centi-cents)
- **Position Data Fields** (from WebSocket, after conversion):
  - `position`: Contract count (+ long, - short)
  - `market_exposure`: Current market value in cents
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
  - `get_metrics()` - Get listener statistics
  - `get_status()` - Get full status for monitoring
  - `get_health_details()` - Get detailed health information
- **Features**:
  - Subscribes to Kalshi `fill` WebSocket channel (authenticated)
  - Receives all fills for the account
  - Automatic reconnection with configurable delay
  - Price values in cents (native Kalshi format)
- **Emits Events**: `ORDER_FILL` (via EventBus.emit_order_fill)
- **Subscribes To**: None (listens to Kalshi WebSocket)
- **Dependencies**: EventBus, KalshiAuth

### 3.3 Services (`traderv3/services/`)

#### TradingDecisionService
- **File**: `services/trading_decision_service.py`
- **Purpose**: Implements trading strategy logic and order execution
- **Available Strategies** (TradingStrategy enum):
  - `HOLD`: Never trade (safe default)
  - `RLM_NO`: Reverse Line Movement - bet NO when public bets YES but price drops
- **Key Methods**:
  - `evaluate_market(market, orderbook)` - Generate trading decision
  - `execute_decision(decision)` - Execute a trading decision
  - `set_strategy(strategy)` - Change trading strategy
  - `get_stats()` - Get service statistics
- **Key Classes**:
  - `TradingDecision`: Represents a trading decision with strategy_id for attribution
    - `action`: "buy", "sell", "hold"
    - `market`: Market ticker
    - `side`: "yes" or "no"
    - `quantity`: Number of contracts
    - `price`: Limit price in cents
    - `strategy_id`: String identifier (e.g., "rlm_no", "s013") - preferred over deprecated `strategy` enum
    - `signal_params`: Strategy-specific parameters for quant analysis
- **Order Context Integration**:
  - Creates `StagedOrderContext` with full trade-time context before placing order
  - Stages context via `OrderContextService.stage_context()`
  - Context persisted to DB on fill confirmation
- **Emits Events**: `SYSTEM_ACTIVITY` (trading_decision, trading_error types)
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, V3StateContainer, EventBus, V3OrderbookIntegration, OrderContextService

#### OrderContextService
- **File**: `services/order_context_service.py`
- **Purpose**: Manages order context capture and persistence for quant analysis
- **Lifecycle**:
  1. **STAGE**: Capture context in memory when order is placed (`stage_context()`)
  2. **PERSIST**: Write to DB when fill is confirmed (`persist_on_fill()`)
  3. **LINK**: Update settlement data when market settles (`link_settlement()`)
  4. **EXPORT**: Provide CSV export for quant analysis (`get_contexts_for_export()`, `generate_csv()`)
- **Key Methods**:
  - `initialize(db_pool)` - Initialize with database connection pool
  - `stage_context(context)` - Stage order context in memory
  - `has_staged_context(order_id)` - Check if staged context exists
  - `get_staged_context(order_id)` - Get staged context
  - `persist_on_fill(order_id, fill_count, fill_avg_price_cents, filled_at)` - Persist to DB on fill
  - `link_settlement(order_id, market_result, realized_pnl_cents, settled_at)` - Link settlement outcome
  - `discard_staged_context(order_id)` - Discard for cancelled orders
  - `get_contexts_for_export(...)` - Query contexts for export
  - `generate_csv(contexts)` - Generate CSV string
  - `get_metrics()` - Get service metrics
- **Key Properties**:
  - `db_pool` - Public accessor for database connection pool (used by StateContainer for batch strategy lookups)
- **Global Access**: `get_order_context_service()` returns singleton instance
- **Emits Events**: None
- **Subscribes To**: None
- **Dependencies**: RLDatabase (connection pool)

#### MarketPriceSyncer
- **File**: `services/market_price_syncer.py`
- **Purpose**: REST API sync for market prices on startup and periodic refresh (every 30s)
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management (initial sync on start)
  - `is_healthy()` - Check if running and last sync not stale
  - `get_health_details()` - Get detailed health information
- **Behavior**:
  - Performs initial sync immediately on `start()`
  - Runs periodic sync loop every 30 seconds (configurable)
  - Updates StateContainer's `_market_prices` directly
  - Fetches prices for all position tickers via REST API
- **Emits Events**: `SYSTEM_ACTIVITY` (sync type with sync_type="market_prices")
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, V3StateContainer, EventBus

#### TradingStateSyncer
- **File**: `services/trading_state_syncer.py`
- **Purpose**: Dedicated service for periodic trading state sync from Kalshi REST API
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management (initial sync on start)
  - `is_healthy()` - Check if running and last sync not stale
  - `get_health_details()` - Get detailed health information
- **Synced Data**: Balance, positions, orders, settlements
- **Behavior**:
  - Performs initial sync immediately on `start()`
  - Runs periodic sync loop every 20 seconds (configurable)
  - Updates StateContainer with trading state and change detection
  - Broadcasts trading_state via StatusReporter after each sync
- **Emits Events**: `SYSTEM_ACTIVITY` (sync type with sync_type="trading_state")
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, V3StateContainer, EventBus, StatusReporter

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
  - `set_orderbook_integration(integration)` - Inject orderbook integration for signal access
- **Key Classes**:
  - `MarketTradeState`: Tracks per-market trade accumulation
  - `RLMSignal`: Detected signal with market_ticker, price_drop, yes_ratio
  - `RLMDecision`: Records decisions (executed, skipped, failed)
  - `TrackedTrade`: Trade record for display
- **Signal Detection Criteria** (ALL must be true):
  1. Market is in TrackedMarketsState (lifecycle-discovered)
  2. `yes_ratio >= threshold` (default: 0.65)
  3. `price_drop >= min_price_drop` (default: 2c)
  4. `trade_count >= min_trades` (default: 15)
  5. No existing position in this market
- **Signal Broadcaster**: Background task that broadcasts orderbook signals for ALL tracked markets every 10 seconds
- **Emits Events**: `SYSTEM_ACTIVITY`, `RLM_MARKET_UPDATE`, `RLM_TRADE_ARRIVED`
- **Subscribes To**: `PUBLIC_TRADE_RECEIVED`, `MARKET_TRACKED`, `MARKET_DETERMINED`, `TMO_FETCHED`
- **Dependencies**: EventBus, TradingDecisionService, V3StateContainer, TrackedMarketsState, V3OrderbookIntegration

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
  4. Add to TrackedMarketsState (in-memory only)
  5. Emit MARKET_TRACKED event
  6. Call orderbook subscribe callback
- **Processing Flow (on 'determined' event)**:
  1. Update TrackedMarketsState status (in-memory)
  2. Emit MARKET_DETERMINED event
  3. Call orderbook unsubscribe callback
- **Emits Events**: `MARKET_TRACKED`, `MARKET_DETERMINED`, `SYSTEM_ACTIVITY`
- **Subscribes To**: `MARKET_LIFECYCLE_EVENT`
- **Dependencies**: EventBus, TrackedMarketsState, V3TradingClientIntegration

#### TrackedMarketsSyncer
- **File**: `services/tracked_markets_syncer.py`
- **Purpose**: REST API sync for tracked market info with periodic refresh (30s) and dormant market cleanup
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `sync_market_info()` - Manual sync trigger
  - `is_healthy()` - Check if running
  - `get_health_details()` - Get detailed health information
- **Dormant Detection**: Unsubscribes markets with zero 24h volume after grace period
  - Position Protection: Skips markets with open positions or resting orders
- **Emits Events**: `SYSTEM_ACTIVITY`
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, TrackedMarketsState, EventBus, V3StateContainer (optional)

#### ApiDiscoverySyncer
- **File**: `services/api_discovery_syncer.py`
- **Purpose**: REST API-based market discovery for already-open markets on startup
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management (initial discovery on start)
  - `is_healthy()` - Check if running
  - `get_health_details()` - Get detailed health information
- **Behavior**:
  - Bootstraps lifecycle mode by discovering already-open markets
  - Filters by configured categories
  - Respects capacity limits from TrackedMarketsState
  - Periodic refresh every 5 minutes (configurable)
- **Emits Events**: `SYSTEM_ACTIVITY` (discovery type)
- **Subscribes To**: None
- **Dependencies**: V3TradingClientIntegration, EventLifecycleService, TrackedMarketsState, EventBus

#### OrderCleanupService
- **File**: `services/order_cleanup_service.py`
- **Purpose**: Manages order TTL cleanup and orphaned order cancellation
- **TTL Cleanup**: Automatically cancels resting orders after configurable timeout (default 5 minutes)
- **Startup Cleanup**: Resets old order groups on restart
- **Dependencies**: V3TradingClientIntegration, V3StateContainer

#### ListenerBootstrapService
- **File**: `services/listener_bootstrap_service.py`
- **Purpose**: Bootstraps real-time WebSocket listeners (PositionListener, FillListener, MarketTickerListener)
- **Dependencies**: V3Config, EventBus, V3StateContainer, V3HealthMonitor, V3StatusReporter

### 3.4 State (`traderv3/state/`)

#### TraderState
- **File**: `state/trader_state.py`
- **Purpose**: Data class representing complete trader state (ALL VALUES IN CENTS)
- **Key Fields**:
  - `balance: int` - Available cash in cents
  - `portfolio_value: int` - Portfolio value in cents
  - `positions: Dict[str, Any]` - Position data by ticker
  - `orders: Dict[str, Any]` - Order data by order_id
  - `settlements: List[Dict]` - Settlement history
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

#### SessionPnLTracker
- **File**: `state/session_pnl_tracker.py`
- **Purpose**: Tracks session-level P&L and cash flow metrics
- **Key Methods**:
  - `initialize(balance, portfolio_value)` - Capture starting state
  - `record_order_fill(cost_cents, contracts)` - Track cash spent
  - `record_settlement(cash_received, fees)` - Track cash received
  - `record_ttl_cancellation(count)` - Track TTL cancellations
  - `get_pnl_summary(current_balance, current_portfolio_value, positions_details)` - Compute P&L
  - `reset()` - Reset all state
- **Tracked Metrics**:
  - `_session_cash_invested` - Cents spent on orders this session
  - `_session_cash_received` - Cents received from settlements
  - `_session_orders_count` - Orders placed
  - `_session_settlements_count` - Positions settled
  - `_session_total_fees_paid` - Total fees in cents
  - `_session_orders_cancelled_ttl` - TTL-cancelled orders
- **Dependencies**: SessionPnLState

#### TradingAttachment
- **File**: `state/trading_attachment.py`
- **Purpose**: Attach trading state to lifecycle-discovered tracked markets
- **Key Classes**:
  - `TradingState`: Enum with `MONITORING`, `SIGNAL_READY`, `ORDER_PENDING`, `ORDER_RESTING`, `POSITION_OPEN`, `AWAITING_SETTLEMENT`, `SETTLED`
  - `TrackedMarketOrder`: Order placed in a tracked market (includes `strategy_id`)
  - `TrackedMarketPosition`: Position in a tracked market
  - `TrackedMarketSettlement`: Final outcome when market settles (includes `strategy_id`, `per_order_pnl`)
  - `TradingAttachment`: Unified view of trading activity per market
- **TradingAttachment Fields**:
  - `ticker`: Market identifier
  - `trading_state`: Current trading lifecycle state
  - `orders`: Dict of TrackedMarketOrder by order_id
  - `position`: Optional TrackedMarketPosition
  - `settlement`: Optional TrackedMarketSettlement
  - `version`: Change detection
- **TrackedMarketOrder Fields**:
  - `order_id`, `signal_id`, `action`, `side`, `count`, `price`, `status`
  - `placed_at`, `fill_count`, `fill_avg_price`, `filled_at`, `cancelled_at`
  - `strategy_id`: String identifier for the strategy that placed this order
- **TrackedMarketSettlement Fields**:
  - `result`, `determined_at`, `settled_at`
  - `final_position`, `final_pnl`, `revenue`, `cost_basis`, `fees`
  - `strategy_id`: Strategy that opened the position (or "mixed" if multiple)
  - `per_order_pnl`: Dict mapping order_id to proportional P&L allocation
- **Key Properties**:
  - `has_exposure`: True if active orders or position
  - `total_pnl`: Realized + unrealized P&L
  - `active_orders`: Orders still pending/resting
- **Data Flow**:
  1. TradingStateSyncer -> StateContainer.update_trading_state()
  2. update_trading_state() -> _sync_trading_attachments() (hooks into existing sync)
  3. For each position/order, if market is tracked, update attachment
  4. FillListener.ORDER_FILL -> mark_order_filled_in_attachment() (real-time)
- **Dependencies**: None (pure data structure)

#### StagedOrderContext
- **File**: `state/order_context.py`
- **Purpose**: Comprehensive order context captured at trade time for quant analysis
- **Key Fields**:
  - Market context: `market_ticker`, `market_category`, `market_close_ts`, `hours_to_settlement`
  - Signal context: `signal_id`, `signal_detected_at`, `signal_params`
  - Price context: `no_price_at_signal`, `bucket_5c`
  - Orderbook context: `best_bid_cents`, `best_ask_cents`, `bid_ask_spread_cents`, `spread_tier`
  - Position context: `existing_position_count`, `existing_position_side`, `is_reentry`
  - Order details: `action`, `side`, `order_price_cents`, `order_quantity`
  - Strategy: `strategy`, `strategy_version`
- **Key Methods**:
  - `compute_derived_fields()` - Calculate bucket, hour_of_day, etc.
  - `to_db_dict(fill_count, fill_avg_price_cents, filled_at)` - Convert for DB insert

#### OrderbookContext
- **File**: `state/order_context.py`
- **Purpose**: Comprehensive orderbook context for trade-time pricing decisions
- **Key Fields**:
  - NO Side: `no_best_bid`, `no_best_ask`, `no_spread`, `no_bid_size_at_bbo`, `no_ask_size_at_bbo`
  - YES Side: `yes_best_bid`, `yes_best_ask`, `yes_spread`
  - Derived: `spread_tier` (SpreadTier enum), `bid_imbalance`, `bbo_depth_tier` (BBODepthTier enum)
  - Freshness: `last_update_ms`, `captured_at`, `is_stale`
- **Enums**:
  - `SpreadTier`: TIGHT (<=2c), NORMAL (<=4c), WIDE (>4c), UNKNOWN
  - `BBODepthTier`: THICK (>=100), NORMAL (20-99), THIN (<20), UNKNOWN
- **Key Methods**:
  - `from_orderbook_snapshot(snapshot)` - Factory from SharedOrderbookState
  - `get_recommended_entry_price(aggressive, max_spread)` - Calculate optimal NO entry price
  - `should_skip_due_to_staleness()` - Check if data too old for trading

#### TrackedMarketsState
- **File**: `state/tracked_markets.py`
- **Purpose**: Single source of truth for lifecycle-discovered markets (in-memory, no DB persistence)
- **Key Classes**:
  - `MarketStatus`: Enum with `ACTIVE`, `DETERMINED`, `SETTLED` states
  - `TrackedMarket`: Dataclass with full market metadata
- **Key Methods**:
  - `add_market(market)` - Add market with capacity check
  - `update_market(ticker, **kwargs)` - Update market fields
  - `update_status(ticker, status)` - Status transition
  - `remove_market(ticker)` - Remove after settlement cleanup
  - `get_active()`, `get_active_tickers()` - Get tradeable markets
  - `is_tracked(ticker)` - Check if market is tracked
  - `get_market(ticker)` - Get market by ticker
  - `at_capacity()`, `capacity_remaining()` - Capacity management
  - `get_stats()` - Statistics for frontend display
  - `get_snapshot()` - Full state snapshot for WebSocket broadcast
- **Key Properties**:
  - `capacity`: Maximum markets (default 50)
  - `active_count`, `total_count`: Current counts
  - `version`: Change detection tracking
- **Dependencies**: None (pure state container)

### 3.5 Sync (`traderv3/sync/`)

#### KalshiDataSync
- **File**: `sync/kalshi_data_sync.py`
- **Purpose**: Fetches state from Kalshi API and tracks changes between syncs
- **Key Methods**:
  - `sync_with_kalshi()` - Full sync, returns (TraderState, StateChange)
  - `refresh_state()` - Refresh without tracking changes
  - `set_order_group_id(id)` - Set order group to track
  - `has_state()` - Check if synced state exists
  - `order_group_id` (property) - Get current order group ID
- **Sync Data Fetched**:
  1. Balance: `get_account_info()` - balance and portfolio_value in cents
  2. Positions: `get_positions()` - market_positions array
  3. Orders: `get_orders()` - orders array (open/resting only)
  4. Settlements: `get_settlements()` - historical settlements
  5. Order Group: `get_order_group(id)` - if order_group_id is set
- **Dependencies**: KalshiDemoTradingClient, TraderState, StateChange, OrderGroupState

### 3.6 Strategy System (`traderv3/strategies/`)

#### StrategyCoordinator
- **File**: `strategies/coordinator.py`
- **Purpose**: Manages multiple concurrent trading strategies with shared rate limiting
- **Key Methods**:
  - `start()` / `stop()` - Lifecycle management
  - `load_config(config_dir)` - Load strategy configs from YAML
  - `rate_limit_acquire(tokens)` - Shared rate limiting across strategies
  - `get_stats()` - Aggregate statistics from all strategies
  - `get_health_details()` - Health information
- **Key Classes**:
  - `StrategyConfig`: Configuration loaded from YAML
  - `TokenBucket`: Shared rate limiter for all strategies
- **Dependencies**: StrategyRegistry, Strategy protocol

#### StrategyRegistry
- **File**: `strategies/registry.py`
- **Purpose**: Registry of available strategy implementations
- **Registered Strategies**:
  - `hold`: HoldStrategy (never trade)
  - `rlm_no`: RLMNoStrategy (reverse line movement)

#### Strategy Protocol
- **File**: `strategies/protocol.py`
- **Purpose**: Defines the strategy interface
- **Key Methods**:
  - `start(context)` / `stop()` - Lifecycle
  - `evaluate(market_data)` - Generate trading signal
  - `get_stats()` - Strategy statistics

### 3.7 Configuration (`traderv3/config/`)

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
  - `trading_strategy_str` - Strategy: "hold" or "rlm_no"
  - `trading_strategy` - TradingStrategy enum (parsed from string)
  - `port` - Server port (default 8005)
  - `min_trader_cash` - Balance protection threshold (cents)
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

### 4.2 Valid Transitions

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
| `ORDERBOOK_SNAPSHOT` | OrderbookClient | V3OrderbookIntegration | `MarketEvent` |
| `ORDERBOOK_DELTA` | OrderbookClient | V3OrderbookIntegration | `MarketEvent` |
| `STATE_TRANSITION` | V3StateMachine | (legacy) | `StateTransitionEvent` |
| `TRADER_STATUS` | V3StatusReporter | V3WebSocketManager | `TraderStatusEvent` |
| `SYSTEM_ACTIVITY` | Multiple | V3WebSocketManager | `SystemActivityEvent` |
| `PUBLIC_TRADE_RECEIVED` | V3TradesIntegration | RLMService | `PublicTradeEvent` |
| `MARKET_POSITION_UPDATE` | PositionListener | V3StateContainer | `MarketPositionEvent` |
| `MARKET_TICKER_UPDATE` | MarketTickerListener | V3Coordinator | `MarketTickerEvent` |
| `ORDER_FILL` | FillListener | V3StateContainer | `OrderFillEvent` |
| `MARKET_LIFECYCLE_EVENT` | V3LifecycleIntegration | EventLifecycleService | `MarketLifecycleEvent` |
| `MARKET_TRACKED` | EventLifecycleService | RLMService, V3WebSocketManager | `MarketTrackedEvent` |
| `MARKET_DETERMINED` | EventLifecycleService | RLMService, V3WebSocketManager | `MarketDeterminedEvent` |
| `RLM_MARKET_UPDATE` | RLMService | V3WebSocketManager | `RLMMarketUpdateEvent` |
| `RLM_TRADE_ARRIVED` | RLMService | V3WebSocketManager | `RLMTradeArrivedEvent` |
| `TMO_FETCHED` | TrueMarketOpenFetcher | RLMService | `TMOFetchedEvent` |

## 6. Strategy Attribution System

The Strategy Attribution System provides per-strategy P&L tracking across positions and settlements, enabling multi-strategy analysis.

### 6.1 Overview

When multiple strategies trade in the same account, we need to attribute each trade and its resulting P&L to the originating strategy. This is achieved through:

1. **Order-time capture**: TradingDecision includes `strategy_id`
2. **Order tracking**: TrackedMarketOrder stores `strategy_id`
3. **DB persistence**: OrderContextService persists strategy to `order_contexts` table
4. **Settlement attribution**: V3StateContainer looks up strategy for each settlement
5. **Cache optimization**: Batch DB queries with in-memory caching

### 6.2 Data Flow

```
1. [TradingDecisionService] Creates TradingDecision with strategy_id
        |
2. [TradingDecisionService] Creates StagedOrderContext with strategy
        |
3. [OrderContextService] stage_context() stores in memory
        |
4. [KalshiDemoTradingClient] Order placed, returns order_id
        |
5. [V3StateContainer] update_order_in_attachment() with strategy_id
        |
6. [FillListener] ORDER_FILL event received
        |
7. [OrderContextService] persist_on_fill() writes to DB with strategy
        |
8. [V3StateContainer] mark_order_filled_in_attachment()
        |
9. [Position Closes - REST sync detects position gone]
        |
10. [V3StateContainer] _capture_settlement_for_attachment()
        | Extracts strategy_id from filled orders
        | Single strategy -> use it; Multiple -> "mixed"
        |
11. [V3StateContainer] _link_settlement_to_order_contexts()
        | Allocates P&L proportionally to each order
        | Writes per-order realized_pnl to DB
```

### 6.3 Cache Architecture

The V3StateContainer maintains two strategy caches to avoid N+1 DB queries:

```python
# Settlement strategy cache
_settlement_strategy_cache: dict[str, str | None]
# Maps ticker -> strategy_id for settlements display

# Position strategy cache
_position_strategy_cache: dict[str, str | None]
# Maps ticker -> strategy_id for positions display
```

**Cache Population Flow**:

1. On sync, collect tickers not in cache
2. Check in-memory TradingAttachment first
3. For remaining unknowns, call `_batch_lookup_strategies_from_db()`
4. Cache all results (including None for not-found)
5. Use cached values in `_format_position_details()` and settlement formatting

**Batch DB Query**:
```sql
SELECT DISTINCT ON (market_ticker) market_ticker, strategy
FROM order_contexts
WHERE market_ticker = ANY($1) AND strategy IS NOT NULL
ORDER BY market_ticker, filled_at DESC
```

### 6.4 Strategy ID Values

| Strategy ID | Description |
|-------------|-------------|
| `hold` | HOLD strategy (no trades) |
| `rlm_no` | RLM NO strategy |
| `s001` - `s999` | Custom plugin strategies |
| `mixed` | Position has orders from multiple strategies |
| `null` | Unknown (no order context found) |

### 6.5 Frontend Display

The `positions_details` and `settlements` arrays in `trading_state` messages include `strategy_id`:

```json
{
  "positions_details": [
    {
      "ticker": "INXD-25JAN03",
      "position": 10,
      "side": "yes",
      "strategy_id": "rlm_no",
      ...
    }
  ],
  "settlements": [
    {
      "ticker": "INXD-25DEC31",
      "net_pnl": 500,
      "strategy_id": "rlm_no",
      ...
    }
  ]
}
```

## 7. Configuration

### 7.1 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KALSHI_API_URL` | Yes | - | REST API base URL |
| `KALSHI_WS_URL` | Yes | - | WebSocket URL |
| `KALSHI_API_KEY_ID` | Yes | - | API key identifier |
| `KALSHI_PRIVATE_KEY_CONTENT` | Yes | - | RSA private key |
| `ENVIRONMENT` | No | `local` | Environment name |
| `RL_MODE` | No | `discovery` | Market selection mode |
| `RL_MARKET_TICKERS` | No | `INXD-25JAN03` | Comma-separated tickers |
| `RL_ORDERBOOK_MARKET_LIMIT` | No | `10` | Max markets to subscribe |
| `V3_ENABLE_TRADING_CLIENT` | No | `false` | Enable trading integration |
| `V3_TRADING_MODE` | No | `paper` | Trading mode |
| `V3_PORT` | No | `8005` | Server port |
| `V3_LOG_LEVEL` | No | `INFO` | Logging level |
| `V3_TRADING_STRATEGY` | No | `hold` | Trading strategy |
| `V3_CLEANUP_ON_STARTUP` | No | `true` | Cancel orphaned orders |
| `V3_ORDER_TTL_ENABLED` | No | `true` | Enable TTL cleanup |
| `V3_ORDER_TTL_SECONDS` | No | `300` | TTL threshold (5 min) |
| `V3_MIN_TRADER_CASH` | No | `10000` | Balance protection (cents) |
| `LIFECYCLE_CATEGORIES` | No | `sports,crypto,...` | Category filter |
| `LIFECYCLE_CAPACITY` | No | `50` | Maximum tracked markets |
| `DORMANT_DETECTION_ENABLED` | No | `true` | Enable dormant cleanup |
| `DORMANT_VOLUME_THRESHOLD` | No | `0` | Dormant volume threshold |
| `DORMANT_GRACE_PERIOD_HOURS` | No | `1.0` | Grace period |
| `RLM_YES_RATIO_THRESHOLD` | No | `0.65` | YES trade ratio threshold |
| `RLM_MIN_PRICE_DROP` | No | `2` | Minimum price drop (cents) |
| `RLM_MIN_TRADES` | No | `15` | Minimum trades for signal |
| `RLM_CONTRACTS` | No | `5` | Contracts per trade |
| `RLM_MAX_TRADES_PER_MINUTE` | No | `10` | Rate limit |

### 7.2 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v3/health` | GET | Quick health check |
| `/v3/status` | GET | Detailed system status |
| `/v3/cleanup` | POST | Cancel orphaned orders |
| `/v3/ws` | WebSocket | Real-time updates |

### 7.3 WebSocket Message Types

| Type | Description | Coalescing |
|------|-------------|------------|
| `connection` | Connection acknowledgment | Immediate |
| `history_replay` | Historical state transitions | Immediate |
| `system_activity` | Console messages | Immediate |
| `state_transition` | State machine changes | Immediate |
| `lifecycle_event` | Real-time lifecycle event | Immediate |
| `trader_status` | Periodic status/metrics | Coalesced (100ms) |
| `trading_state` | Trading data with P&L | Coalesced (100ms) |
| `tracked_markets` | Tracked markets snapshot | Coalesced (100ms) |
| `rlm_states_snapshot` | Initial RLM states | Immediate |
| `rlm_market_state` | RLM state update | Coalesced (100ms) |
| `rlm_trade_arrived` | Trade pulse animation | Immediate |
| `trade_processing` | Processing stats | Coalesced (100ms) |
| `upcoming_markets` | Markets opening soon | Coalesced (100ms) |
| `ping` | Keep-alive | Immediate |

## 8. Data Flow Traces

### 8.1 System Startup Sequence

```
1. app.py: lifespan() starts
2. V3Config.from_env() loads configuration
3. rl_db.initialize() initializes database
4. OrderContextService initialized with DB pool
5. EventBus created
6. V3StateMachine created with EventBus
7. V3WebSocketManager created with EventBus
8. Market tickers selected (discovery or config mode)
9. OrderbookClient created
10. V3OrderbookIntegration wraps OrderbookClient
11. [if trading] KalshiDemoTradingClient created
12. [if trading] V3TradingClientIntegration wraps trading client
13. V3Coordinator created with all components
14. coordinator.start() called
15. System transitions: STARTUP -> INITIALIZING -> ORDERBOOK_CONNECT -> READY
16. Background services started (health monitor, status reporter, syncers)
17. [if RLM_NO] RLM service started
```

### 8.2 Order Placement with Context Capture

```
1. [RLMService] Signal detected, creates TradingDecision with strategy_id="rlm_no"
        |
2. [TradingDecisionService] execute_decision() called
        | Creates StagedOrderContext with full trade-time context
        | Calls OrderContextService.stage_context()
        |
3. [V3TradingClientIntegration] place_order()
        |
4. [KalshiDemoTradingClient] create_order() -> returns order_id
        |
5. [V3StateContainer] update_order_in_attachment()
        | Stores TrackedMarketOrder with strategy_id
        |
6. [FillListener] ORDER_FILL event
        |
7. [V3StateContainer] mark_order_filled_in_attachment()
        |
8. [OrderContextService] persist_on_fill()
        | Writes to order_contexts table with strategy
```

### 8.3 Settlement with Strategy Attribution

```
1. [TradingStateSyncer] sync_with_kalshi() fetches settlements
        |
2. [V3StateContainer] update_trading_state()
        | For each settlement:
        |   a. Check _settlement_strategy_cache
        |   b. Check TradingAttachment.orders for strategy_id
        |   c. If not found, add to batch lookup list
        |
3. [V3StateContainer] _batch_lookup_strategies_from_db()
        | Single query for all unknown tickers
        | Updates _settlement_strategy_cache
        |
4. [V3StateContainer] Format settlement with strategy_id
        |
5. [V3StatusReporter] emit_trading_state()
        | Broadcasts settlements with strategy attribution
```

### 8.4 Graceful Shutdown

```
1. SIGINT/SIGTERM received
2. coordinator.stop() called
3. RLM service stopped
4. Syncers stopped (trading_state, market_price, lifecycle)
5. Listeners stopped (position, fill, ticker)
6. Health monitor and status reporter stopped
7. Trading client integration stopped (resets order group)
8. Orderbook integration stopped
9. State machine -> SHUTDOWN
10. WebSocket manager stopped (disconnects clients)
11. Event bus stopped
12. OrderContextService cleanup (discards staged contexts)
13. Database pool closed
14. Application exits
```
