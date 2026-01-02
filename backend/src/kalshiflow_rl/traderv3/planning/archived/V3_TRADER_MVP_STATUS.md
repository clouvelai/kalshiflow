# V3 Trader MVP Status & Architecture

## Executive Summary

The V3 Trader is a production-ready trading system for Kalshi markets with clean separation of concerns, robust error handling, and support for degraded mode operation. The system successfully connects to orderbooks, syncs trading state, and provides real-time WebSocket updates to frontend clients.

**Current Status: ✅ OPERATIONAL**
- Foundation: Complete and stable
- Trading: Ready for strategy implementation
- Monitoring: Full observability via console and APIs

## Architecture Overview

```
                           V3 TRADER ARCHITECTURE (ACTUAL)
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Frontend (http://localhost:5173/v3-trader)           │
│                              WebSocket Client Console                        │
└────────────────────────────────▲────────────────────────────────────────────┘
                                 │ WebSocket (ws://localhost:8005/v3/ws)
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                            app.py (FastAPI Application)                      │
│  Endpoints:                                                                  │
│  - GET /v3/health → Health status                                           │
│  - GET /v3/status → Detailed system status                                  │
│  - WS  /v3/ws    → WebSocket for real-time updates                         │
└────────────────────────────────▲────────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                        V3Coordinator (810 lines)                            │
│                     src/kalshiflow_rl/traderv3/core/coordinator.py          │
│                                                                              │
│  Responsibilities:                                                           │
│  • State machine orchestration (startup → ready → shutdown)                 │
│  • Service lifecycle management                                             │
│  • Event loop coordination                                                  │
│  • Trading sync scheduling (30-second intervals)                            │
│  • Degraded mode handling                                                   │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│EventBus  │ │State     │ │WebSocket │ │Health    │ │Status    │ │Trading   │
│          │ │Machine   │ │Manager   │ │Monitor   │ │Reporter  │ │Flow      │
│event_bus │ │state_    │ │websocket_│ │health_   │ │status_   │ │trading_  │
│.py       │ │machine.py│ │manager.py│ │monitor.py│ │reporter. │ │flow.py   │
│          │ │          │ │          │ │(277 lines)│ │py        │ │          │
└────▲─────┘ └──────────┘ └────▲─────┘ └──────────┘ └──────────┘ └────▲─────┘
     │                         │                                        │
     │ Events                  │ Broadcasts                            │ Decisions
     │                         │                                        │
┌────┴───────────────────┬─────┴────────────────┬──────────────────────┴─────┐
│                        │                       │                            │
▼                        ▼                       ▼                            ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────┐
│OrderbookIntegr.  │ │TradingClientInt. │ │KalshiDataSync    │ │TradingDecision│
│                  │ │                  │ │                  │ │Service       │
│orderbook_        │ │trading_client_   │ │kalshi_data_      │ │trading_      │
│integration.py    │ │integration.py    │ │sync.py           │ │decision.py   │
│                  │ │(1,158 lines)     │ │(508 lines)       │ │              │
│Wraps orderbook   │ │Manages orders    │ │Syncs exchange    │ │Strategy impl │
│client for V3     │ │and positions     │ │state             │ │(HOLD/RL)     │
└────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘ └──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│OrderbookClient   │ │KalshiDemoTrading │ │StateContainer    │
│(985 lines)       │ │Client            │ │(363 lines)       │
│                  │ │                  │ │                  │
│WebSocket conn    │ │REST API calls    │ │Version tracking  │
│to Kalshi         │ │to demo-api       │ │State storage     │
└──────────────────┘ └──────────────────┘ └──────────────────┘
         │                    │
         ▼                    ▼
    Kalshi WebSocket     Kalshi REST API
    (orderbook data)     (trades/positions)
```

## Component Lifecycle & Data Flow

### 1. Startup Sequence
```python
STARTUP → INITIALIZING → ORDERBOOK_CONNECT → TRADING_CLIENT_CONNECT → KALSHI_DATA_SYNC → READY
```

1. **STARTUP**: Application initialization
   - Load configuration from environment
   - Initialize database and write queue
   - Create core components (EventBus, StateMachine, WebSocketManager)

2. **INITIALIZING**: Component setup
   - Start EventBus processing loop
   - Configure WebSocket manager subscriptions
   - Initialize health monitoring

3. **ORDERBOOK_CONNECT**: Orderbook WebSocket connection
   - Start OrderbookClient with market list
   - Wait for WebSocket connection (30s timeout)
   - If fails: Enter degraded mode (continue without orderbook)
   - If succeeds: Wait for first snapshot (10s timeout)

4. **TRADING_CLIENT_CONNECT**: Trading API connection
   - Initialize KalshiDemoTradingClient
   - Authenticate with demo API
   - Verify connectivity

5. **KALSHI_DATA_SYNC**: Initial state synchronization
   - Fetch all positions from exchange
   - Fetch all open orders
   - Populate StateContainer
   - Calculate initial metrics

6. **READY**: Operational state
   - Start periodic sync loop (30-second intervals)
   - Process trading decisions
   - Monitor component health
   - Handle WebSocket broadcasts

### 2. Event Flow

```
User Action → Frontend → WebSocket → Coordinator → Service → Event → Broadcast
```

Example: Trade Execution
1. Frontend sends trade request via WebSocket
2. WebSocketManager receives and forwards to Coordinator
3. Coordinator checks state machine allows trading
4. TradingFlow orchestrates decision process
5. TradingDecisionService evaluates opportunity
6. TradingClientIntegration places order via API
7. EventBus emits order_placed event
8. WebSocketManager broadcasts to all clients

### 3. Health Monitoring

```python
# Health check runs every 5 seconds
components_health = {
    "event_bus": EventBus.is_healthy(),
    "state_machine": StateMachine.is_healthy(),
    "orderbook_integration": OrderbookIntegration.is_healthy(),
    "trading_client": TradingClientIntegration.is_healthy()
}

# Degraded mode detection
if state == READY and not all_healthy:
    if is_degraded_mode:
        # Expected - continue operating
        log.info("Operating in degraded mode")
    else:
        # Unexpected - transition to ERROR
        transition_to(ERROR)
```

## Key Files & Responsibilities

### Core Orchestration
- `coordinator.py` (810 lines) - Main orchestrator, event loop, state transitions
- `event_bus.py` (500+ lines) - Pub/sub for all system events
- `state_machine.py` (200+ lines) - State transition logic and validation

### Service Integrations
- `orderbook_integration.py` (300+ lines) - Wraps OrderbookClient for V3
- `trading_client_integration.py` (1,158 lines) - Order/position management
- `kalshi_data_sync.py` (508 lines) - Exchange state synchronization

### Monitoring & Reporting
- `health_monitor.py` (277 lines) - Component health checks
- `status_reporter.py` (150+ lines) - Status aggregation and broadcasting
- `websocket_manager.py` (400+ lines) - Client connection management

### Trading Logic
- `trading_flow.py` (300+ lines) - Trading decision orchestration
- `trading_decision.py` (200+ lines) - Strategy implementation (HOLD/RL)

### State Management
- `state_container.py` (363 lines) - Centralized state with versioning
- `models.py` - Pydantic models for type safety

## Configuration & Environment

### Required Environment Variables
```bash
# API Configuration (paper trading)
KALSHI_API_URL=https://demo-api.kalshi.co/trade-api/v2
KALSHI_WS_URL=wss://demo-api.kalshi.co/trade-api/ws/v2
KALSHI_API_KEY_ID=<your_demo_key>
KALSHI_PRIVATE_KEY_CONTENT=<your_rsa_key>

# Trading Configuration
V3_MARKET_TICKERS=INXD-25JAN03,NASDAQ-25JAN03  # or "DISCOVERY" for auto
V3_MAX_MARKETS=10                               # Max markets in discovery mode
V3_MAX_ORDERS=10                                # Max concurrent orders
V3_MAX_POSITION_SIZE=100                        # Max position per market
V3_SYNC_INTERVAL_SECONDS=30                     # Trading sync interval
V3_HEALTH_CHECK_INTERVAL=5                      # Health check frequency

# Server Configuration
V3_HOST=0.0.0.0
V3_PORT=8005
V3_LOG_LEVEL=INFO
```

## API Endpoints

### Health Check
```bash
GET /v3/health
Response: {
    "healthy": true,
    "status": "running", 
    "state": "ready",
    "uptime": 35.316
}
```

### Detailed Status
```bash
GET /v3/status
Response: {
    "state": "ready",
    "markets": 10,
    "snapshots_received": 145,
    "deltas_received": 892,
    "positions": 15,
    "orders": 2,
    "session_id": 1511,
    "uptime": 120.5,
    "components": {
        "event_bus": true,
        "orderbook": true,
        "trading": true
    }
}
```

### WebSocket Connection
```javascript
ws://localhost:8005/v3/ws

// Message types received:
{
    "type": "system_activity",
    "data": {
        "activity_type": "state_transition",
        "message": "ready → acting",
        "metadata": {"severity": "info"}
    }
}

{
    "type": "trader_status",
    "data": {
        "state": "ready",
        "markets": 10,
        "positions": 15
    }
}
```

## Degraded Mode Operation

The system supports degraded mode when the orderbook WebSocket is unavailable:

1. **Detection**: Orderbook connection fails or 503 errors
2. **Behavior**: 
   - Trading continues using REST API only
   - No real-time orderbook data
   - Sync continues every 30 seconds
   - Health checks don't trigger ERROR state
3. **Recovery**: Automatic when orderbook becomes available

## Error Handling

### State Transitions on Error
```
READY → ERROR → (recovery) → INITIALIZING → ... → READY
```

### Error Recovery
- Automatic recovery attempts when all components healthy
- Graceful degradation for non-critical failures
- Session cleanup on critical errors
- Full restart capability via shutdown/startup

## Testing & Validation

### Start the System
```bash
# Using the launcher script (recommended)
./scripts/run-rl-trader.sh --port 8005 --markets 10

# Or directly with uvicorn
ENVIRONMENT=paper uv run uvicorn src.kalshiflow_rl.traderv3.app:app --port 8005
```

### Verify Operation
1. Check health: `curl http://localhost:8005/v3/health`
2. View console: `http://localhost:5173/v3-trader`
3. Monitor logs: Check for clean state transitions
4. Test trading: System should maintain READY state

## Known Issues & Limitations

1. **Orderbook 503 Errors**: Handled via degraded mode
2. **State Duplication**: Some state exists in multiple places (technical debt)
3. **Large Coordinator**: 810 lines doing multiple responsibilities
4. **No Rate Limiting**: Could hit API limits with many markets

## Recent Fixes (2024-12-27)

1. **Health Monitor Fix**: Properly checks degraded mode before ERROR transition
2. **Log Spam Reduction**: Throttled health messages to once per minute
3. **Sync Message Severity**: Shows as [info] not [error] in degraded mode
4. **Frontend Console**: Respects severity field for proper message styling

## Future Improvements (Not Critical)

1. Extract WebSocketEventHandler service (reduce coordinator complexity)
2. Implement LeanStateContainer (remove redundant state)
3. Add rate limiting for API calls
4. Implement circuit breaker pattern for service failures
5. Add metrics collection (Prometheus/Grafana)

## For Coding Agents

### Quick Debugging
- State machine stuck? Check `health_monitor.py` line 183-230
- WebSocket issues? Check `orderbook_integration.py` connection logic
- Sync failures? Check `kalshi_data_sync.py` error handling
- Console spam? Check `coordinator.py` event emission frequency

### Key Patterns
- All services communicate through EventBus (no direct service-to-service)
- StateContainer has versioning for rollback capability
- Degraded mode is a feature, not a bug
- Health checks should report, not control state transitions

### Testing Changes
1. Always check browser console at `http://localhost:5173/v3-trader`
2. Monitor state transitions in logs
3. Verify health endpoint remains healthy
4. Test with orderbook disconnection (kill port 8002)
5. Ensure sync continues in degraded mode

## Conclusion

The V3 Trader foundation is **production-ready** with robust error handling, clean separation of concerns, and full observability. The system successfully handles degraded mode operation and provides a stable platform for implementing trading strategies.

**Status**: ✅ Ready for trading logic implementation
**Stability**: ✅ No state thrashing, minimal log spam
**Monitoring**: ✅ Full visibility via console and APIs
**Next Step**: Implement actual trading strategies in `trading_decision.py`