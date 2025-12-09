# Ship RL Orderbook Collector - Implementation Plan

## Executive Summary

This document outlines the implementation plan for the RL Orderbook Collector, an intermediary milestone that establishes the foundation for the Kalshi RL Trading Subsystem. This service will collect real orderbook data from multiple markets, store it in PostgreSQL, and surface it to the frontend via WebSocket - all before implementing any RL training or inference components.

**Rationale:**
1. Validate orderbook processing and storage is working exactly as intended
2. Start collecting training data immediately (the sooner we start, the better)
3. Prevent bugs/errors from assumptions and hypothetical data
4. Establish a solid upstream pipeline for all downstream RL components

## Architecture Overview

### Deployment Model

The RL Orderbook Collector will be deployed as a **separate service** alongside the existing Kalshiflow backend:

```
Railway Services:
├── kalshi-flowboard-backend     # Existing trade flow (unchanged)
│   └── src/kalshiflow/app.py
│
└── kalshiflow-rl-backend       # NEW: RL orderbook collector
    └── src/kalshiflow_rl/app.py
```

Both services will:
- Run independently on Railway
- Have separate WebSocket endpoints
- Share the same PostgreSQL database (different tables)
- Be accessed by the same frontend application

### System Components

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Kalshi Orderbook│────▶│   RL Backend     │────▶│    PostgreSQL    │
│    WebSocket     │     │    Service       │     │   (Supabase)     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                               │                           │
                               ▼                           ▼
                         ┌──────────────────┐     ┌──────────────────┐
                         │  Frontend WS     │     │  RL Orderbook    │
                         │   Connection     │────▶│   Dashboard      │
                         └──────────────────┘     └──────────────────┘
```

## Implementation Details

### Phase 1: Backend Service Implementation

#### 1.1 Application Entry Point (`src/kalshiflow_rl/app.py`)

Create a standalone Starlette application that:
- Initializes OrderbookClient with markets from `RL_MARKET_TICKERS`
- Sets up WebSocket endpoint for frontend connections
- Manages service lifecycle (startup/shutdown)
- Coordinates all async components

**Key responsibilities:**
```python
- Load configuration from environment
- Initialize database connection pool
- Start orderbook client for multiple markets
- Start write queue for async DB persistence
- Start WebSocket manager for frontend broadcasts
- Track and expose statistics
```

#### 1.2 WebSocket Manager (`src/kalshiflow_rl/websocket_manager.py`)

New component to handle frontend connections:
- Accept WebSocket connections from frontend
- Maintain connection registry
- Broadcast orderbook updates to all connected clients
- Send periodic statistics updates
- Handle reconnection gracefully

**Message Protocol:**
```json
// Connection established
{
  "type": "connection",
  "data": {
    "markets": ["KXCABOUT-29", "KXFEDDECISION-25DEC", "KXLLM1-25DEC31"],
    "status": "connected",
    "version": "1.0.0"
  }
}

// Orderbook snapshot (initial state)
{
  "type": "orderbook_snapshot",
  "data": {
    "market_ticker": "KXCABOUT-29",
    "timestamp_ms": 1734567890123,
    "sequence_number": 1001,
    "yes_bids": {"45": 100, "44": 250},
    "yes_asks": {"55": 150, "56": 300},
    "no_bids": {"40": 200},
    "no_asks": {"60": 175},
    "yes_mid_price": 50.0,
    "no_mid_price": 50.0
  }
}

// Orderbook delta (incremental update)
{
  "type": "orderbook_delta",
  "data": {
    "market_ticker": "KXCABOUT-29",
    "timestamp_ms": 1734567891234,
    "sequence_number": 1002,
    "side": "yes",
    "action": "update",
    "price": 45,
    "old_size": 100,
    "new_size": 150
  }
}

// Statistics update (every second)
{
  "type": "stats",
  "data": {
    "markets_active": 3,
    "snapshots_processed": 15,
    "deltas_processed": 1523,
    "messages_per_second": 12.5,
    "db_queue_size": 45,
    "uptime_seconds": 3600,
    "last_update_ms": 1734567890123
  }
}
```

#### 1.3 Statistics Collector (`src/kalshiflow_rl/stats_collector.py`)

Lightweight statistics tracking:
```python
class StatsCollector:
    - track_snapshot(market_ticker: str)
    - track_delta(market_ticker: str)
    - track_db_write(count: int)
    - get_stats() -> Dict[str, Any]
    - get_market_stats(market_ticker: str) -> Dict[str, Any]
```

Metrics to track:
- Total snapshots/deltas processed
- Per-market update counts
- Database write statistics
- Queue sizes and health
- WebSocket connection count
- Service uptime

### Phase 2: Frontend Implementation

#### 2.1 New Route (`/rl-orderbooks`)

Create a standalone page with no navigation from main app (direct URL access only):

**File Structure:**
```
frontend/src/
├── pages/
│   └── RLOrderbooks.tsx       # New orderbook dashboard
├── hooks/
│   └── useRLOrderbook.ts      # WebSocket hook for RL backend
└── components/
    └── rl/
        ├── OrderbookStats.tsx  # Hero statistics section
        ├── OrderbookGrid.tsx   # Market grid container
        └── OrderbookCard.tsx   # Individual market card
```

#### 2.2 Hero Statistics Section

Display real-time system metrics:
```typescript
interface OrderbookStats {
  marketsActive: number;
  snapshotsProcessed: number;
  deltasProcessed: number;
  messagesPerSecond: number;
  uptimeSeconds: number;
  connectionStatus: 'connecting' | 'connected' | 'disconnected';
  lastUpdateMs: number;
}
```

Visual elements:
- Large metric cards with icons
- Real-time update indicators
- Connection status badge
- Uptime counter (formatted as HH:MM:SS)

#### 2.3 Market Grid View

Grid of cards showing each market's orderbook:

**Per-Market Card:**
```typescript
interface MarketOrderbook {
  marketTicker: string;
  lastSequence: number;
  updateCount: number;
  lastUpdateMs: number;
  bestBid: { yes: number; no: number };
  bestAsk: { yes: number; no: number };
  spread: { yes: number; no: number };
  depth: { bids: number; asks: number };
}
```

Visual features:
- Market ticker as card title
- Best bid/ask prices prominently displayed
- Spread indicator
- Update counter
- Time since last update
- Visual pulse on update

#### 2.4 WebSocket Hook (`useRLOrderbook`)

Manage connection to RL backend:
```typescript
const useRLOrderbook = () => {
  const [stats, setStats] = useState<OrderbookStats>();
  const [orderbooks, setOrderbooks] = useState<Map<string, MarketOrderbook>>();
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>();
  
  // Connect to RL backend WebSocket (different port/endpoint)
  // Handle reconnection with exponential backoff
  // Parse and dispatch messages
  // Maintain local state
  
  return { stats, orderbooks, connectionStatus };
};
```

### Phase 3: Deployment Configuration

#### 3.1 Railway Service Setup

**New Service: `kalshiflow-rl-backend`**
```toml
# railway.toml for RL backend
[deploy]
startCommand = "cd backend && uv run uvicorn kalshiflow_rl.app:app --host 0.0.0.0 --port $PORT"

[healthcheck]
path = "/health"
timeout = 30

[restart]
policy = "on-failure"
maxRetries = 3
```

**Environment Variables:**
```env
# Reuse existing Kalshi credentials
KALSHI_API_KEY_ID=${KALSHI_API_KEY_ID}
KALSHI_PRIVATE_KEY_CONTENT=${KALSHI_PRIVATE_KEY_CONTENT}

# RL-specific configuration
RL_MARKET_TICKERS=KXCABOUT-29,KXFEDDECISION-25DEC,KXLLM1-25DEC31
RL_BATCH_SIZE=100
RL_SAMPLE_RATE=1
RL_FLUSH_INTERVAL=1.0

# Database (shared with main service)
DATABASE_URL=${DATABASE_URL}
DATABASE_URL_POOLED=${DATABASE_URL_POOLED}

# Port will be assigned by Railway
PORT=<assigned>
```

#### 3.2 Frontend Configuration

Update frontend to connect to both backends:
```typescript
// config.ts
export const BACKENDS = {
  trades: import.meta.env.VITE_TRADES_WS_URL || 'ws://localhost:8000/ws',
  orderbooks: import.meta.env.VITE_RL_WS_URL || 'ws://localhost:8001/ws'
};
```

## Data Flow

### Orderbook Data Pipeline

1. **Subscription Phase**
   - OrderbookClient connects to Kalshi WebSocket
   - Subscribes to markets from `RL_MARKET_TICKERS`
   - Receives initial snapshots for each market

2. **Processing Phase**
   - Snapshots → Update SharedOrderbookState → Queue for DB → Broadcast to frontend
   - Deltas → Apply to SharedOrderbookState → Queue for DB → Broadcast to frontend

3. **Persistence Phase**
   - WriteQueue batches messages
   - Async write to PostgreSQL tables:
     - `rl_orderbook_snapshots` - Full orderbook states
     - `rl_orderbook_deltas` - Incremental updates

4. **Frontend Updates**
   - WebSocket manager broadcasts to all connected clients
   - Frontend updates UI in real-time
   - Statistics refresh every second

## Testing Strategy

### Local Development Testing

1. **Backend Validation**
   ```bash
   # Start RL backend
   cd backend
   uv run uvicorn kalshiflow_rl.app:app --reload --port 8001
   
   # Check health
   curl http://localhost:8001/health
   
   # Monitor logs for orderbook messages
   ```

2. **Database Verification**
   ```sql
   -- Check snapshots
   SELECT market_ticker, COUNT(*) 
   FROM rl_orderbook_snapshots 
   GROUP BY market_ticker;
   
   -- Check deltas
   SELECT market_ticker, COUNT(*) 
   FROM rl_orderbook_deltas 
   WHERE timestamp_ms > (EXTRACT(EPOCH FROM NOW() - INTERVAL '1 hour') * 1000)
   GROUP BY market_ticker;
   ```

3. **Frontend Testing**
   ```bash
   # Access RL dashboard directly
   http://localhost:5173/rl-orderbooks
   
   # Verify:
   - WebSocket connects successfully
   - Statistics update in real-time
   - Market cards show orderbook data
   - Updates flow continuously
   ```

### Production Validation

1. **Deployment Checklist**
   - [ ] RL backend service created on Railway
   - [ ] Environment variables configured
   - [ ] Health check passing
   - [ ] Logs show successful Kalshi connection
   - [ ] Database tables receiving data

2. **Monitoring Points**
   - WebSocket connection stability
   - Message processing rate
   - Database write performance
   - Memory usage trends
   - Error rates in logs

## Success Criteria

### Must Have (Launch Blocking)
- ✅ OrderbookClient connects to all configured markets
- ✅ Snapshots and deltas stored correctly in PostgreSQL
- ✅ Frontend receives real-time updates via WebSocket
- ✅ Statistics accurately reflect processing
- ✅ System remains stable for 1+ hour of continuous operation

### Should Have (Post-Launch)
- Reconnection handling for Kalshi WebSocket
- Graceful degradation on partial market failures
- Performance metrics dashboard
- Alert on anomalies (no data for X minutes)

### Nice to Have (Future)
- Historical data replay capability
- Data export functionality
- Market health indicators
- Orderbook visualization charts

## Migration Notes

### From Existing RL Code

**What We Keep:**
- `OrderbookClient` - Already handles multi-market subscriptions
- `database.py` - RL tables already defined
- `write_queue.py` - Async write infrastructure ready
- `orderbook_state.py` - SharedOrderbookState for in-memory updates

**What We Add:**
- `app.py` - Service orchestration
- `websocket_manager.py` - Frontend broadcasting
- `stats_collector.py` - Metrics tracking

**What We Defer:**
- All training components (environments, models, etc.)
- Actor/inference systems
- Paper trading logic
- Reward calculations

### Database Considerations

The RL tables are already created but isolated from trade flow:
- `rl_orderbook_snapshots` - Full orderbook states
- `rl_orderbook_deltas` - Incremental updates
- `rl_models` - Not used in this phase
- `rl_trading_episodes` - Not used in this phase
- `rl_trading_actions` - Not used in this phase

## Risk Mitigation

### Potential Issues & Solutions

1. **High message volume overwhelming system**
   - Solution: Delta sampling already implemented (1 in N)
   - Monitoring: Queue size metrics in stats

2. **WebSocket connection instability**
   - Solution: Automatic reconnection with exponential backoff
   - Monitoring: Connection status in frontend UI

3. **Database write latency**
   - Solution: Async write queue with batching
   - Monitoring: Queue depth and flush metrics

4. **Memory growth from orderbook state**
   - Solution: Bounded in-memory state per market
   - Monitoring: Memory usage in Railway metrics

## Timeline

### Week 1: Backend Implementation
- Day 1-2: Create app.py and wire up existing components
- Day 3-4: Implement WebSocket manager and stats collector
- Day 5: Local testing and debugging

### Week 2: Frontend & Deployment
- Day 1-2: Build frontend components and route
- Day 3: WebSocket hook and state management
- Day 4: Deploy to Railway and configure
- Day 5: Production validation and monitoring

## Next Steps After This Milestone

Once the orderbook collector is successfully deployed and collecting data:

1. **Immediate Next (Week 3-4):**
   - Analyze collected data quality
   - Implement data validation checks
   - Add monitoring dashboards

2. **Following Sprint:**
   - Begin implementing Gymnasium environment using collected data
   - Design reward functions based on real orderbook dynamics
   - Start training pipeline development

3. **Future Sprints:**
   - Actor implementation with hot-reload
   - Paper trading integration
   - Model registry and versioning
   - Performance evaluation framework

## Conclusion

This orderbook collector milestone provides a solid foundation for the RL Trading Subsystem by:
1. Establishing real data flow from Kalshi to our system
2. Validating our orderbook processing logic with actual market data
3. Building the upstream pipeline that all RL components will depend on
4. Starting data collection immediately for future training
5. Creating a visible, production-ready service that demonstrates progress

By shipping this first, we ensure that all subsequent RL development is grounded in real, validated data rather than assumptions or simulations.