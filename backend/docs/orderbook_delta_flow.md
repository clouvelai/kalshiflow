# Orderbook Delta Message Flow

## Sequence Diagram

```mermaid
sequenceDiagram
    participant WS as Kalshi WebSocket (Orderbook)
    participant OC as OrderbookClient
    participant SOS as SharedOrderbookState
    participant WQ as WriteQueue
    participant EB as EventBus
    participant AS as ActorService
    participant LOA as LiveObservationAdapter
    participant ASel as ActionSelector
    participant OM as OrderManager
    participant FL as FillListener
    participant WSFL as Kalshi WebSocket (Fills)
    participant API as Kalshi API

    Note over FL,WSFL: Parallel WebSocket for real-time fills
    FL->>WSFL: Subscribe to "fill" channel
    WSFL-->>FL: Connected & subscribed

    WS->>OC: Delta message
    Note over OC: Parse & validate
    
    OC->>SOS: apply_delta()
    SOS-->>OC: Updated
    
    OC->>WQ: enqueue_delta()
    WQ-->>OC: Queued
    
    OC->>EB: emit_orderbook_delta()
    EB->>AS: MarketEvent
    
    Note over AS: Queue event
    
    Note over AS,LOA: Step 1: Build Observation
    AS->>SOS: get_snapshot()
    SOS-->>AS: Snapshot
    AS->>LOA: build_observation()
    LOA-->>AS: 52-feature array
    
    Note over AS,ASel: Step 2: Select Action
    AS->>ASel: select_action()
    alt HardcodedSelector
        ASel-->>AS: action=0 (HOLD)
    else RLModelSelector
        ASel->>ASel: model.predict()
        ASel-->>AS: action=0-4
    end
    
    Note over AS,API: Step 3: Execute Action
    alt HOLD
        AS-->>AS: Skip execution
    else Trading Action
        alt Throttled
            AS-->>AS: Skip (throttled)
        else Not Throttled
            AS->>SOS: get_snapshot()
            AS->>OM: execute_limit_order_action()
            OM->>API: place_order()
            API-->>OM: Order placed
            OM-->>AS: Result
        end
    end
    
    Note over WSFL,OM: Real-time Fill Processing (Parallel)
    alt Order Filled
        WSFL->>FL: Fill notification
        FL->>OM: queue_fill()
        OM->>OM: _process_fills()
        OM->>OM: Update positions immediately
    end
    
    Note over AS,OM: Step 4: Update Positions (Dual Approach)
    alt HOLD or Throttled
        AS-->>AS: Skip
    else Executed
        AS->>AS: Wait 100ms (configurable)
        Note over AS: Fallback sync for missed fills
        AS->>OM: get_positions()
        OM-->>AS: Positions (reconciled)
    end
    
    AS->>AS: Update metrics
```

## Simple Text Flow

```
1. Kalshi WebSocket → OrderbookClient
   └─> Receives delta message {delta: +5, price: 50, side: "yes"}

2. OrderbookClient._process_delta()
   ├─> Parse message, calculate new_size
   ├─> SharedOrderbookState.apply_delta() → Update in-memory orderbook
   ├─> WriteQueue.enqueue_delta() → Queue for DB write (non-blocking)
   └─> EventBus.emit_orderbook_delta() → Publish event (non-blocking)

3. EventBus → ActorService
   └─> ActorService._handle_event_bus_event() receives MarketEvent
       └─> trigger_event() → Queue ActorEvent (non-blocking)

3.5. FillListener (Parallel Process - Runs Independently)
   └─> FillListener WebSocket → Kalshi User Fills Channel
       ├─> Started by OrderManager at initialization
       ├─> Subscribe to "fill" channel for tracked markets
       ├─> Receive real-time fill notifications
       └─> OrderManager.queue_fill() → Update positions immediately
           └─> _process_fills() queue → Async position reconciliation

4. ActorService Background Loop
   └─> _process_market_update() processes queued event

5. Step 1: Build Observation
   ├─> ActorService → SharedOrderbookState.get_snapshot()
   ├─> ActorService → LiveObservationAdapter.build_observation()
   │   ├─> Convert to SessionDataPoint format
   │   └─> Compute temporal features (activity_score, momentum, time_gap)
   └─> Returns: 52-feature numpy array

6. Step 2: Select Action
   ├─> ActorService → ActionSelector.select_action(observation, market_ticker)
   │
   ├─> If HardcodedSelector:
   │   └─> Returns: action = 0 (HOLD)
   │
   └─> If RLModelSelector:
       ├─> Uses cached PPO model (loaded once at startup)
       ├─> model.predict(observation, deterministic=True)
       └─> Returns: action = 0-4 (HOLD or trading action)

7. Step 3: Execute Action
   ├─> ActorService._safe_execute_action(action, market_ticker)
   │
   ├─> If action == 0 (HOLD):
   │   └─> Return immediately (skip orderbook fetch, OrderManager call)
   │
   └─> If action != 0 (Trading action):
       ├─> Check throttling (250ms since last action for this market)
       │
       ├─> If throttled:
       │   └─> Return {"status": "throttled"} (skip execution)
       │
       └─> If not throttled:
           ├─> Fetch orderbook snapshot
           ├─> OrderManager.execute_limit_order_action(action, market_ticker, snapshot)
           │   ├─> Calculate limit price from snapshot
           │   ├─> Check cash availability (for BUY orders)
           │   └─> Trading client place_order() → Kalshi API
           │       └─> Returns: order_id
           └─> Update throttle timestamp

8. Step 4: Update Positions (Dual Approach)
   ├─> If HOLD or throttled:
   │   └─> Skip (no position changes)
   │
   └─> If order executed:
       ├─> Real-time path (via FillListener):
       │   ├─> FillListener receives fill notification immediately
       │   ├─> Queues fill for processing
       │   └─> OrderManager updates positions in real-time
       │
       └─> Fallback sync path (for reliability):
           ├─> Wait 100ms (configurable via position_read_delay_ms)
           ├─> OrderManager.get_positions() → Read current positions
           ├─> Reconcile with real-time fill updates
           ├─> OrderManager.get_portfolio_value() → Read portfolio value
           └─> Log position changes

9. Update Metrics
   └─> Track processing_time, events_processed, errors, etc.
```

## Key Points

1. **Non-blocking**: OrderbookClient doesn't wait for ActorService processing
2. **Serial Processing**: Single queue ensures no race conditions
3. **Throttling**: Enforced per-market (250ms minimum between actions)
4. **HOLD Optimization**: HOLD actions skip orderbook fetch, OrderManager call, and position updates
5. **Dual Position Tracking**: 
   - **Real-time fills**: FillListener provides immediate fill notifications via separate WebSocket
   - **Fallback sync**: 100ms delayed position read ensures consistency for missed fills
6. **Parallel WebSocket Connections**:
   - **Orderbook WebSocket**: Receives market delta updates
   - **Fill WebSocket**: Receives real-time fill notifications (managed by FillListener)
7. **Position Reconciliation**: OrderManager reconciles real-time fills with periodic position syncs
8. **Configurable Delays**: `position_read_delay_ms` parameter allows tuning of fallback sync timing

## Architecture Notes

### FillListener Service

The FillListener is a critical component that provides real-time order fill notifications:

- **Initialization**: Started automatically by `KalshiMultiMarketOrderManager` during setup
- **WebSocket Connection**: Maintains separate WebSocket connection to Kalshi's user fill channel
- **Subscription**: Subscribes to "fill" channel for all tracked markets
- **Fill Processing**: Forwards fills to OrderManager via `queue_fill()` method
- **Queue Processing**: OrderManager processes fills asynchronously via `_process_fills()` 
- **Position Updates**: Immediate position reconciliation upon fill receipt

### Error Handling & Circuit Breakers

The system includes comprehensive error handling:

- **Market-level circuit breakers**: Disable markets after repeated errors
- **Automatic re-enabling**: Markets re-enabled after cooldown period
- **Error tracking**: Per-market error counters and timestamps
- **Graceful degradation**: System continues operating with remaining healthy markets

### Performance Optimizations

- **Lazy service initialization**: Services initialized only when needed
- **Dependency injection**: ServiceContainer pattern for efficient service management
- **Queue-based processing**: Prevents blocking on I/O operations
- **Batch database writes**: WriteQueue batches updates for efficiency
- **Observation caching**: Recent observations cached to avoid redundant computation

### Portfolio Integration

The observation building process includes portfolio features:

- **Cash balance**: Available cash for trading
- **Position sizes**: Current position in each market
- **Order features**: Pending orders and their characteristics
- **Portfolio value**: Total portfolio value tracking
- **PnL tracking**: Real-time profit/loss calculations

