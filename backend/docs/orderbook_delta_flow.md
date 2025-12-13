# Orderbook Delta Message Flow

## Sequence Diagram

```mermaid
sequenceDiagram
    participant KalshiWS as Kalshi WebSocket
    participant OC as OrderbookClient
    participant SOS as SharedOrderbookState
    participant WQ as WriteQueue
    participant EB as EventBus
    participant AS as ActorService
    participant LOA as LiveObservationAdapter
    participant ASel as ActionSelector
    participant OM as OrderManager
    participant KAPI as KalshiDemoTradingClient

    KalshiWS->>OC: Delta message<br/>{delta: +5, price: 50, side: "yes"}
    
    Note over OC: _process_delta()
    OC->>OC: Parse message<br/>Calculate new_size<br/>Determine action
    
    OC->>SOS: apply_delta(delta_data)
    SOS-->>OC: Updated (in-memory)
    
    OC->>WQ: enqueue_delta(delta_data)
    WQ-->>OC: Success (queued for DB)
    
    OC->>EB: emit_orderbook_delta()<br/>(non-blocking)
    EB-->>OC: Event published
    
    EB->>AS: MarketEvent<br/>(ORDERBOOK_DELTA)
    
    Note over AS: _handle_event_bus_event()
    AS->>AS: trigger_event()<br/>Queue ActorEvent
    
    Note over AS: Background processing loop
    AS->>AS: _process_market_update()
    
    rect rgb(200, 220, 255)
        Note over AS,LOA: Step 1: Build Observation
        AS->>SOS: get_snapshot()
        SOS-->>AS: Orderbook snapshot
        AS->>LOA: build_observation(market_ticker)
        LOA->>LOA: Convert to SessionDataPoint<br/>Compute temporal features
        LOA-->>AS: 52-feature observation array
    end
    
    rect rgb(220, 255, 200)
        Note over AS,ASel: Step 2: Select Action
        AS->>ASel: select_action(observation, market_ticker)
        alt HardcodedSelector
            ASel-->>AS: action = 0 (HOLD)
        else RLModelSelector
            ASel->>ASel: model.predict(observation)
            ASel-->>AS: action = 0-4
        end
    end
    
    rect rgb(255, 220, 200)
        Note over AS,OM: Step 3: Execute Action
        AS->>AS: _safe_execute_action(action)
        
        alt action == 0 (HOLD)
            AS-->>AS: Return immediately<br/>(skip execution)
        else action != 0
            AS->>AS: Check throttling<br/>(250ms per market)
            alt Throttled
                AS-->>AS: Return throttled<br/>(skip execution)
            else Not Throttled
                AS->>SOS: get_snapshot()
                SOS-->>AS: Orderbook snapshot
                AS->>OM: execute_limit_order_action(action, market_ticker, snapshot)
                OM->>OM: Calculate limit price<br/>Check cash availability
                OM->>KAPI: place_order()
                KAPI-->>OM: Order placed (order_id)
                OM-->>AS: Execution result<br/>{status: "placed", executed: true}
                AS->>AS: Update throttle timestamp
            end
        end
    end
    
    rect rgb(255, 255, 200)
        Note over AS,OM: Step 4: Update Positions
        alt HOLD or Throttled
            AS-->>AS: Skip (no position changes)
        else Order Executed
            AS->>AS: Wait 100ms<br/>(for async fill processing)
            AS->>OM: get_positions()<br/>get_portfolio_value()
            OM-->>AS: Updated positions<br/>Portfolio value
            AS->>AS: Log position changes
        end
    end
    
    AS->>AS: Update metrics<br/>(processing_time, events_processed)
```

## Key Points

1. **Non-blocking**: OrderbookClient doesn't wait for ActorService processing
2. **Serial Processing**: Single queue ensures no race conditions
3. **Throttling**: Enforced per-market (250ms minimum between actions)
4. **HOLD Optimization**: HOLD actions skip orderbook fetch, OrderManager call, and position updates
5. **Eventual Consistency**: Position updates wait 100ms for async fill processing

