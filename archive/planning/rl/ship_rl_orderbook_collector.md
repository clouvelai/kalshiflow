# Ship RL Orderbook Collector - Implementation Plan

## Executive Summary

This document outlines the implementation plan for the RL Orderbook Collector SHIPPING milestone. The data collection system is READY TO SHIP with a simple WebSocket stats dashboard. Complex orderbook UI (300+ market grid) has been moved to a future milestone to prioritize ML model development.

**Updated Priorities:**
1. ✅ SHIP NOW: Data collection system with simple stats view (session ID, market count, live metrics)
2. 🧠 NEXT: Focus on advanced RL model development and training
3. 📈 FUTURE: Complex multi-market orderbook grid UI (milestone 4.2)
4. ⚡ GOAL: Start collecting data immediately while building the actual ML models

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

#### 2.2 Simple Stats Dashboard (SHIP VERSION)

Display essential system metrics for operational visibility:
```typescript
interface RLStats {
  sessionId: string;
  sessionStatus: 'active' | 'connecting' | 'disconnected';
  marketsActive: number;
  snapshotsProcessed: number;
  deltasProcessed: number;
  messagesPerSecond: number;
  uptimeSeconds: number;
  lastUpdateMs: number;
}
```

Minimal UI elements for shipping:
- Current session ID and status
- Number of markets being tracked  
- Live counts of snapshots/deltas collected
- Basic health metrics (uptime, message rate)
- Clean, focused design for operational visibility

#### 2.3 Market Grid View (MOVED TO FUTURE MILESTONE 4.2)

**DEFERRED**: Complex multi-market orderbook grid UI with 300+ market cards has been moved to future milestone 4.2. This includes:
- Real-time orderbook depth visualization
- Market activity heatmaps
- Advanced filtering and sorting
- Interactive orderbook drill-down views
- Performance optimization for 300+ cards

**RATIONALE**: Simple stats view is sufficient to validate data collection pipeline. Priority shifted to ML model development after shipping basic data collection system.

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

5. **Session Tracking Phase**
   - Each WebSocket connection creates a new session in `rl_orderbook_sessions`
   - All snapshots and deltas tagged with current `session_id`
   - Reconnections create new sessions, handling sequence number resets
   - Session boundaries enable proper data reconstruction and analysis

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

## Implementation Status

### ✅ PHASE 1 COMPLETE (All 8 Work Units)

#### Completed Work Units:
1. **WU001** ✅ Modified Application Entry Point for Multi-Market Support
   - Updated `src/kalshiflow_rl/app.py` with multi-market initialization
   - Integrated WebSocket manager and statistics collector
   - Health endpoint includes multi-market status

2. **WU002** ✅ Created WebSocket Manager Component
   - Implemented `src/kalshiflow_rl/websocket_manager.py`
   - Handles multiple concurrent frontend connections
   - Non-blocking broadcasting architecture

3. **WU003** ✅ Implemented Statistics Collector
   - Created `src/kalshiflow_rl/stats_collector.py`
   - Thread-safe metrics tracking
   - Per-market and aggregate statistics

4. **WU004** ✅ Added WebSocket Route to Application
   - WebSocketRoute at `/rl/ws` endpoint integrated
   - Proper Starlette routing configuration

5. **WU005** ✅ Updated Configuration for Multi-Market Support
   - Parses `RL_MARKET_TICKERS` environment variable
   - Backward compatible with single ticker
   - Test verified with 3 markets

6. **WU006** ✅ Created Deployment Configuration
   - `railway-rl.toml` for Railway deployment
   - Startup scripts and test utilities
   - Health check and restart policies

7. **WU007** ✅ Backend E2E Test Implementation
   - Comprehensive test in `tests/test_rl_orderbook_e2e.py`
   - Non-flaky design focusing on snapshots
   - Clear validation with ✅/❌ indicators

8. **WU008** ✅ Added E2E Test to CI Pipeline
   - Test scripts for easy execution
   - Documentation updated in CLAUDE.md
   - Integrated into development workflow

### ✅ PHASE 2 COMPLETE - SESSION TRACKING SYSTEM (2025-12-10)
### 🚚 PHASE 3 READY - SIMPLE STATS UI FOR SHIPPING (2025-12-10)

#### Session Tracking Implementation:
9. **WU009** ✅ Database Schema Enhancement for Session Tracking
10. **WU010** ✅ OrderbookClient Session Management Integration  
11. **WU011** ✅ Write Queue Session Integration
12. **WU012** ✅ Historical Analysis Tools Update
13. **WU013** ✅ Production Deployment and Validation

#### SHIPPING MILESTONE - Simple Stats UI:
14. **WU014** 🚚 Simple RL Stats Dashboard for Immediate Shipping
    - Create `/rl-stats` route with basic React component
    - Display current session ID and connection status
    - Show number of markets being tracked (300+)
    - Display live counts of snapshots/deltas collected
    - Show basic health metrics (uptime, message rate)
    - Clean, focused UI for operational visibility
    - **STATUS**: SHIP BLOCKING - Required for Milestone 1.1 release

### Running the Service

```bash
# Start the RL orderbook collector service
cd backend
uv run uvicorn kalshiflow_rl.app:app --reload --port 8001

# Run the E2E test
uv run pytest tests/test_rl_orderbook_e2e.py -v

# Check health endpoint
curl http://localhost:8001/rl/health
```

## Success Criteria

### ✅ READY TO SHIP (Launch Blocking)
- ✅ OrderbookClient connects to all configured markets
- ✅ Snapshots and deltas stored correctly in PostgreSQL  
- ✅ System remains stable for 1+ hour of continuous operation
- ✅ Session tracking handles WebSocket reconnections properly
- ✅ Data collection operates at scale with 300+ markets
- 🚚 Simple stats WebSocket dashboard shows live metrics

### 🧠 NEXT PRIORITY (Post-Shipping)
- Advanced RL model development with sophisticated architectures
- Multi-objective reward functions and risk management
- Comprehensive model evaluation and backtesting framework
- Trading actor development with hot-reload capabilities

### 📈 FUTURE MILESTONES (Deferred)
- Complex multi-market orderbook grid UI (milestone 4.2)
- Advanced orderbook depth visualization
- Market heatmaps and analytics
- Interactive orderbook drill-down views

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

## Next Steps After Shipping

Once the orderbook collector with simple stats view is shipped:

1. **🧠 IMMEDIATE FOCUS (Priority 1):**
   - Design advanced multi-market RL model architectures
   - Implement sophisticated reward functions with risk management
   - Create comprehensive model evaluation and backtesting framework
   - Build trading actor with hot-reload capabilities

2. **🎯 ML MODEL DEVELOPMENT:**
   - Attention mechanisms for market correlation modeling
   - LSTM/GRU components for temporal modeling
   - Ensemble methods for model robustness
   - Risk-aware architecture components

3. **📈 FUTURE UI DEVELOPMENT:**
   - Complex multi-market orderbook grid (milestone 4.2)
   - Real-time orderbook depth visualization
   - Market heatmaps and advanced analytics
   - Interactive features for 300+ markets

## Conclusion

**PRIORITY UPDATE**: The orderbook collector is READY TO SHIP with a simple stats dashboard. Complex UI has been strategically deferred to focus on what matters most:

1. ✅ **SHIP NOW**: Real data collection with operational visibility
2. 🧠 **BUILD NEXT**: Advanced ML models that can actually trade
3. 📈 **ENHANCE LATER**: Complex visualization for 300+ markets

**Strategic Benefits:**
- Start collecting training data immediately
- Validate data pipeline with production deployment
- Focus development effort on ML model sophistication
- Ship visible progress while building core trading capabilities
- Defer complex UI until after trading actor is functional

This approach ensures we have a working data collection system in production while focusing on the ML models that will actually generate trading value.