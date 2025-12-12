# Kalshi Actor/Trader Service Implementation Plan (v1)

---

## 🔍 REVIEW NOTES (Pre-Implementation Validation)

> **Reviewed against current codebase on 2024-12-11**
> 
> This section documents compatibility validation, identified gaps, and required clarifications before implementation.

### ✅ Validated Components (Confirmed Compatible)

| Component | Status | Location |
|-----------|--------|----------|
| `OrderbookClient` | ✅ Exists | `backend/src/kalshiflow_rl/data/orderbook_client.py` |
| `SharedOrderbookState` | ✅ Exists with `get_snapshot()` | `backend/src/kalshiflow_rl/data/orderbook_state.py` |
| `write_queue` | ✅ Exists with `enqueue_snapshot/delta` | `backend/src/kalshiflow_rl/data/write_queue.py` |
| `LimitOrderActionSpace` | ✅ Exists with async `execute_action()` | `backend/src/kalshiflow_rl/environments/limit_order_action_space.py` |
| `KalshiOrderManager` | ✅ Exists | `backend/src/kalshiflow_rl/trading/order_manager.py` |
| `KalshiDemoTradingClient` | ✅ Exists | `backend/src/kalshiflow_rl/trading/demo_client.py` |
| `WebSocketManager` | ✅ Exists at `/rl/ws` | `backend/src/kalshiflow_rl/websocket_manager.py` |
| `RL_MARKET_MODE=discovery` | ✅ Config exists | `backend/src/kalshiflow_rl/config.py:44` |
| `ORDERBOOK_MARKET_LIMIT` | ✅ Config exists | `backend/src/kalshiflow_rl/config.py:45` |

### ⚠️ Critical Integration Issues to Address

#### 1. **Observation Format Mismatch** (HIGH PRIORITY)

**Problem**: Training uses `SessionDataPoint` objects with pre-computed temporal features. Live data provides raw `SharedOrderbookState.get_snapshot()` dicts.

**Training pipeline expects (`build_observation_from_session_data`):**
```python
SessionDataPoint:
  - markets_data: Dict[str, Dict]  # ticker -> orderbook data
  - mid_prices: Dict[str, Tuple]   # ticker -> (yes_mid, no_mid)
  - time_gap: float                # Seconds since previous update
  - activity_score: float          # Pre-computed [0,1]
  - momentum: float                # Pre-computed [-1,1]
```

**Live pipeline provides (`SharedOrderbookState.get_snapshot()`):**
```python
{
  'market_ticker': str,
  'yes_bids': Dict[int, int],
  'yes_asks': Dict[int, int],
  'no_bids': Dict[int, int],
  'no_asks': Dict[int, int],
  'yes_mid_price': float,
  'no_mid_price': float,
  ...
}
```

**Required in `LiveObservationAdapter`:**
1. Convert `get_snapshot()` → `markets_data` format (straightforward - just wrap in dict)
2. **Compute `activity_score` on-the-fly** from volume changes (reference: `session_data_loader.py:646-648`)
3. **Compute `momentum` on-the-fly** from mid-price changes (reference: `session_data_loader.py:660-675`)
4. Track `time_gap` between updates (use `last_update_time` from snapshot)

#### 2. **SharedOrderbookState.get_snapshot() is Async** (MEDIUM)

**Problem**: Plan says "read full current orderbook from `SharedOrderbookState.get_snapshot()`" without noting it's async.

**Signature in `orderbook_state.py:381-384`:**
```python
async def get_snapshot(self) -> Dict[str, Any]:
    """Get atomic snapshot of current orderbook state."""
    async with self._lock:
        return deepcopy(self._state.to_dict())
```

**Impact**: ActorService must be async-aware when reading orderbook state.

#### 3. **Actor Trigger Needs Market Context** (MEDIUM)

**Problem**: Plan says "trigger after `write_queue.enqueue_*()` returns True" but doesn't specify how to pass market context.

**Current `enqueue_snapshot` signature:**
```python
async def enqueue_snapshot(self, snapshot_data: Dict[str, Any]) -> bool
```

**Required in trigger callback:**
```python
actor_trigger(market_ticker: str, update_type: str, sequence_number: int)
```

**Solution**: The `snapshot_data` passed to `enqueue_*` contains `market_ticker` - the callback in `OrderbookClient` has access to this.

#### 4. **Multi-Market Actor Selection Logic Missing** (LOW)

**Problem**: Plan mentions `RL_ACTOR_TARGET_MARKET_MODE=hot` for trading "top volume" market, but no selection logic is specified.

**Current state**: `OrderbookClient` subscribes to multiple markets from `config.RL_MARKET_TICKERS`, but actor needs to decide WHICH market to trade on each step.

**Needs specification**:
- How is "hottest" market determined? (volume? spread? activity?)
- Can actor switch markets mid-session?
- What happens if selected market becomes illiquid?

### 📝 Minor Clarifications Needed

1. **Config values don't exist yet**: `RL_ACTOR_ENABLED`, `RL_ACTOR_MODEL_ID`, `RL_ACTOR_TICK_THROTTLE_MS`, `RL_ACTOR_TARGET_MARKET_MODE`, `RL_ACTOR_CONTRACT_SIZE` - these need to be added to `config.py`

2. **OrderManager.get_order_features()** exists and returns `OrderFeatures` with `.to_array()` method (5 features) - compatible with observation builder.

3. **OrderManager.get_position_info()** exists and returns format compatible with `extract_portfolio_features()`.

### 🔧 Recommended Pre-Implementation Tasks

1. [ ] Add actor-specific config values to `config.py`
2. [ ] Design `LiveObservationAdapter` with explicit temporal feature computation
3. [ ] Specify market selection strategy for multi-market mode
4. [ ] Add actor event message types to `WebSocketManager`
5. [ ] Create integration test that validates live observation matches training observation format

---

## Overview

We already have a **live orderbook collector** in `kalshiflow_rl` that:

1) Receives orderbook snapshot/delta messages from Kalshi (`OrderbookClient`)
2) Updates the in-memory `SharedOrderbookState`
3) Pushes the *same normalized message* into `write_queue` for DB persistence (non-blocking)
4) Broadcasts orderbook snapshots to the frontend (`WebSocketManager` at `/rl/ws`)

**Key refinement (your idea):** instead of building a parallel “actor ingestion” system, the actor should **tap the existing collector pipeline** and run inference/trading **only after the message has been accepted by the persistence queue** (i.e., after `write_queue.enqueue_snapshot/enqueue_delta` returns `True`).

This keeps the actor aligned with the exact stream we persist for training, while still executing trades off the up-to-date in-memory orderbook state.

## System Architecture Understanding

### Training Pipeline (Existing)
- **DB session data** → `SessionDataLoader` loads historical snapshots/deltas
- **Env step** → `MarketAgnosticKalshiEnv` converts session point → `OrderbookState`
- **Feature extraction** → `build_observation_from_session_data()` (52-dim)
- **Execution** → `LimitOrderActionSpace` + `SimulatedOrderManager`

### Live Collector Pipeline (Existing)
- **Kalshi WS** → `OrderbookClient` (`backend/src/kalshiflow_rl/data/orderbook_client.py`)
  - Updates `SharedOrderbookState.apply_snapshot/apply_delta`
  - Enqueues *normalized* snapshot/delta dicts into `write_queue` (`backend/src/kalshiflow_rl/data/write_queue.py`)
- **Frontend WS** → `WebSocketManager` broadcasts orderbook snapshots (`/rl/ws`)

### Live Actor Pipeline (Revised Plan)
- **Trigger**: run actor step when a snapshot/delta is successfully enqueued to `write_queue`
- **State source**: read full current orderbook from `SharedOrderbookState.get_snapshot()` (not from DB)
- **Observation**: build training-compatible observation from live orderbook snapshot + actor’s current portfolio state
- **Action**: `model.predict(obs)` → discrete action 0–4
- **Execution**: `LimitOrderActionSpace.execute_action()` → `KalshiOrderManager.place_order()` → `KalshiDemoTradingClient`
- **Broadcast**: publish actor events (action/order/fill/position) via the **existing** `/rl/ws` WebSocket (no new WS endpoint needed for v1)

## Answer to “do we use the demo client directly?” (clarified)

For the actor service:
- **Trades should flow through the OrderManager abstraction**:
  - `LimitOrderActionSpace` → `KalshiOrderManager` → `KalshiDemoTradingClient`
- **Position tracking should come from the OrderManager**:
  - `order_manager.get_positions()` / `get_position_info()` reads the local tracked positions
  - Optional: `sync_positions_with_kalshi()` can reconcile against Kalshi demo API if needed
- **Avoid calling the demo client directly in the actor** (keep one abstraction boundary). The existing `TradingSession` code calls demo client directly in places, but for the new actor we should not.

## Core Components

### Existing Components to Reuse (no new ingestion system)

1. **OrderbookClient** (`backend/src/kalshiflow_rl/data/orderbook_client.py`)
   - Discovery mode already supported via `RL_MARKET_MODE=discovery` and `ORDERBOOK_MARKET_LIMIT=100`
   - Produces normalized `snapshot_data` and `delta_data` and updates `SharedOrderbookState`

2. **SharedOrderbookState** (`backend/src/kalshiflow_rl/data/orderbook_state.py`)
   - In-memory authoritative book; supports `get_snapshot()` to read the full book

3. **write_queue** (`backend/src/kalshiflow_rl/data/write_queue.py`)
   - Non-blocking persistence queue (`enqueue_snapshot`, `enqueue_delta`)
   - We will “tap” this moment (post-enqueue) to drive actor steps

4. **Execution stack**
   - `LimitOrderActionSpace` (`backend/src/kalshiflow_rl/environments/limit_order_action_space.py`)
   - `KalshiOrderManager` (`backend/src/kalshiflow_rl/trading/order_manager.py`)
   - `KalshiDemoTradingClient` (`backend/src/kalshiflow_rl/trading/demo_client.py`)

5. **WebSocketManager** (`backend/src/kalshiflow_rl/websocket_manager.py`)
   - Existing `/rl/ws` endpoint; we can extend message types to include actor events

### New Components to Build (minimal, focused)

1. **ActorService / ActorLoop** (NEW)
   - Orchestrates: (event trigger) → (snapshot read) → (obs) → (predict) → (execute) → (broadcast)
   - **Location**: `backend/src/kalshiflow_rl/actor/actor_service.py`

2. **Live Observation Adapter** (NEW)
   - Converts a live `SharedOrderbookState.get_snapshot()` dict into the same feature schema used in training
   - Maintains short history (e.g., last 10 snapshots) for temporal features
   - **Location**: `backend/src/kalshiflow_rl/actor/live_observation.py`
   
   > **⚠️ IMPLEMENTATION DETAIL**: Must compute temporal features on-the-fly that training pre-computes:
   > ```python
   > # Reference: session_data_loader.py:646-675
   > 
   > # activity_score computation:
   > num_markets = len(markets_data)
   > total_volume = sum(market.get('total_volume', 0) for market in markets_data.values())
   > activity_score = min(1.0, (num_markets * np.log(1 + total_volume)) / 1000.0)
   > 
   > # momentum computation (requires 3+ historical mid-prices):
   > if len(prices) >= 3:
   >     recent_change = prices[-1] - prices[-2]
   >     prev_change = prices[-2] - prices[-3]
   >     momentum = np.tanh((recent_change - prev_change) / abs(prev_change))
   > 
   > # time_gap: delta between current and previous snapshot timestamp_ms
   > ```

3. **Model Inference Wrapper** (NEW)
   - Loads a trained SB3 model and returns discrete action (0–4)
   - Optional hot reload (later)
   - **Location**: `backend/src/kalshiflow_rl/actor/inference.py`

4. **Actor Event Schema + Broadcast Helper** (NEW)
   - Defines actor event types and payloads
   - Publishes over existing `WebSocketManager` connection set
   - **Location**: `backend/src/kalshiflow_rl/actor/events.py`

### Small changes to existing components (not “new components”, but required wiring)

A. **Tap point for actor trigger (choose one of these; both are small):**

- **Option A (recommended): add a post-enqueue callback in `OrderbookClient`**
  - After `await write_queue.enqueue_*()` returns `True`, call `actor_trigger(market_ticker, update_type, sequence_number)`.
  - This gives us “actor runs only on persisted-stream-accepted messages”, without modifying `write_queue`.

- **Option B: add optional callbacks to `write_queue` itself**
  - e.g. `write_queue.add_subscriber(fn)` called on successful `enqueue_*`.

B. **Extend `WebSocketManager` to broadcast actor events**
  - Keep `/rl/ws` as the single WS endpoint for RL dashboarding in v1.

## Actor Loop Flow (Revised)

```
Kalshi WS message
  ↓
OrderbookClient._process_snapshot/_process_delta
  ↓
SharedOrderbookState updated (authoritative in-memory book)
  ↓
write_queue.enqueue_snapshot/enqueue_delta
  ↓ (only if enqueue returns True)
Actor trigger enqueues an internal ActorEvent into actor_service
  ↓
ActorService reads SharedOrderbookState.get_snapshot()
  ↓
LiveObservationAdapter builds 52-dim observation (training-compatible)
  ↓
Model predicts discrete action (0–4)
  ↓
LimitOrderActionSpace.execute_action(action, ticker, orderbook_state)
  ↓
KalshiOrderManager → KalshiDemoTradingClient executes
  ↓
Actor events broadcast via existing /rl/ws
  ↓
Frontend shows “actor trading” view
```

> **⚠️ ASYNC CONSIDERATIONS**:
> - `SharedOrderbookState.get_snapshot()` is async (uses asyncio.Lock)
> - `LimitOrderActionSpace.execute_action()` is async (calls async OrderManager methods)
> - ActorService must run in async context or use dedicated event loop

> **⚠️ DATA CONVERSION REQUIRED** in `LiveObservationAdapter`:
> ```python
> # get_snapshot() returns single-market dict
> snapshot = await shared_state.get_snapshot()
> 
> # Must convert to SessionDataPoint-like format for build_observation_from_session_data()
> markets_data = {snapshot['market_ticker']: snapshot}
> mid_prices = {snapshot['market_ticker']: (snapshot['yes_mid_price'], snapshot['no_mid_price'])}
> 
> # Must compute temporal features on-the-fly (not pre-computed like training data):
> time_gap = (snapshot['last_update_time'] - prev_timestamp_ms) / 1000.0
> activity_score = compute_activity_score(markets_data)  # See session_data_loader.py:646-648
> momentum = compute_momentum(mid_price_history)  # See session_data_loader.py:660-675
> ```

## Frontend Plan (simplified)

- Reuse the existing RL websocket connection (`/rl/ws`) and add support for new message types:
  - `actor_action`, `actor_order`, `actor_fill`, `actor_position`, `actor_pnl`
- Implement a simple `ActorTradingView` that displays:
  - last N actions
  - last N orders/fills
  - current positions + cash

**New UI**:
- `frontend/src/components/ActorTradingView.jsx` (NEW)
- Optionally a hook `frontend/src/hooks/useRlWs.js` or `useActorEvents.js` (NEW) that filters messages from the shared `/rl/ws` stream.

## Configuration

The collector already handles market discovery and subscription. Actor config should focus on:

```bash
RL_ACTOR_ENABLED=true
RL_ACTOR_MODEL_ID=<optional>
RL_ACTOR_TICK_THROTTLE_MS=250   # prevent overtrading on bursty deltas
RL_ACTOR_TARGET_MARKET_MODE=hot # e.g. trade only “top volume” market from discovery set
RL_ACTOR_CONTRACT_SIZE=10

# Demo trading client (already)
KALSHI_PAPER_TRADING_API_KEY_ID=...
KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT=...
```

> **⚠️ CONFIG NOT YET IMPLEMENTED**: The following variables need to be added to `backend/src/kalshiflow_rl/config.py`:
> - `RL_ACTOR_ENABLED` (bool)
> - `RL_ACTOR_MODEL_ID` (optional str)
> - `RL_ACTOR_TICK_THROTTLE_MS` (int, default 250)
> - `RL_ACTOR_TARGET_MARKET_MODE` (str: "hot" | "config" | "all")
> - `RL_ACTOR_CONTRACT_SIZE` (int, default 10)
>
> Demo trading config already exists and is validated:
> - ✅ `KALSHI_PAPER_TRADING_API_KEY_ID` (config.py:27)
> - ✅ `KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT` (config.py:28)
> - ✅ `KALSHI_PAPER_TRADING_WS_URL` (config.py:29)
> - ✅ `KALSHI_PAPER_TRADING_API_URL` (config.py:30)

## Implementation Order (Revised)

1. **Actor trigger wiring** (tap post-enqueue in `OrderbookClient` or `write_queue`)
   > ✅ Straightforward - add callback after `enqueue_*()` returns True in `OrderbookClient._process_snapshot/_process_delta`

2. **LiveObservationAdapter** (build 52-dim obs from `SharedOrderbookState.get_snapshot()` + portfolio state)
   > ⚠️ **Critical component** - must compute temporal features (`activity_score`, `momentum`, `time_gap`) on-the-fly. See `session_data_loader.py:610-677` for reference implementation.

3. **ModelInferenceWrapper** (load model, predict action)
   > ✅ Straightforward - wrap SB3 model with `model.predict(obs)` call

4. **ActorService** (queueing, throttling, execution through OrderManager)
   > ⚠️ Must handle async context properly - `get_snapshot()` and `execute_action()` are both async

5. **WebSocketManager extension** (broadcast actor events over `/rl/ws`)
   > ✅ Straightforward - add new message dataclasses (e.g., `ActorActionMessage`, `ActorOrderMessage`)

6. **Frontend ActorTradingView** (display actor activity)
   > ✅ Standard React component consuming filtered `/rl/ws` messages

## Why this approach makes sense

- **No duplicate ingestion**: one WS client, one normalization path, one in-memory state machine.
- **Alignment with training data**: actor decisions are tied to the same messages we persist.
- **Lower operational complexity**: no new WS endpoints required for MVP.
- **Backpressure-aware**: if the write queue is full and drops messages, the actor won’t trade on unpersisted updates (by design).

## Future Enhancements (out of scope for v1)

- DB persistence of actor actions/trades (separate tables)
- Separate `/rl/actor/ws` endpoint if we outgrow `/rl/ws`
- Multi-market coordination / portfolio-level strategy
- Risk controls (max position, max order rate, kill switch)

---

## 📋 REVIEW SUMMARY

### Ready for Implementation ✅

| Component | Status | Notes |
|-----------|--------|-------|
| Actor trigger mechanism | ✅ Ready | Add callback in `OrderbookClient` after `enqueue_*()` |
| Execution stack | ✅ Ready | `LimitOrderActionSpace` → `KalshiOrderManager` → `KalshiDemoTradingClient` chain is complete |
| WebSocket broadcast | ✅ Ready | Extend existing `/rl/ws` with new message types |
| Position tracking | ✅ Ready | `OrderManager.get_position_info()` returns compatible format |
| Order features | ✅ Ready | `OrderManager.get_order_features()` returns compatible format |
| Config (discovery mode) | ✅ Ready | `RL_MARKET_MODE=discovery` already in config.py |

### Requires Careful Implementation ⚠️

| Component | Priority | Issue |
|-----------|----------|-------|
| **LiveObservationAdapter** | HIGH | Must compute `activity_score`, `momentum`, `time_gap` on-the-fly to match training format |
| **Async handling** | MEDIUM | `get_snapshot()` and `execute_action()` are async - ActorService must be async-aware |
| **Market selection** | MEDIUM | Multi-market mode needs explicit selection strategy (which market to trade?) |

### Before Implementation

1. [ ] Add actor-specific config values to `config.py`
2. [ ] Design `LiveObservationAdapter` with explicit temporal feature computation formulas
3. [ ] Define market selection strategy for `RL_ACTOR_TARGET_MARKET_MODE=hot`
4. [ ] Create integration test: verify live observation exactly matches training observation format
5. [ ] Document async execution flow for ActorService

### Key Integration Test

```python
# CRITICAL: Before deployment, verify observation format compatibility
def test_live_observation_matches_training():
    """Ensure live observation has same shape and semantics as training."""
    # Load a known training observation from SessionDataLoader
    training_obs = get_training_observation(session_id=1, step=0)
    
    # Create equivalent live observation from SharedOrderbookState
    live_obs = live_adapter.build_observation(snapshot_data, portfolio_state)
    
    assert training_obs.shape == live_obs.shape == (52,)
    # Feature-by-feature validation for temporal features especially
```

---

## 🔧 Unified Trading Client & Environment Tracking

### Problem Statement

Currently, `KalshiDemoTradingClient` is hardcoded for demo/paper trading only:

```python
# Current: demo_client.py
class KalshiDemoTradingClient:
    def __init__(self, mode: str = "paper"):
        if mode != "paper":
            raise ValueError(...)  # Forces paper-only
        
        # Hardcoded demo credentials
        self.api_key_id = config.KALSHI_PAPER_TRADING_API_KEY_ID
        self.rest_base_url = config.KALSHI_PAPER_TRADING_API_URL  # demo-api.kalshi.co
```

This prevents using the same client for production trading without code changes.

### Solution: Unified `KalshiTradingClient`

Refactor to a single `KalshiTradingClient` that selects endpoints based on environment config:

```python
# Proposed: trading_client.py (replaces demo_client.py)

class KalshiTradingClient:
    """
    Unified Kalshi trading client for both demo and production environments.
    
    Environment selection is controlled by:
    1. KALSHI_TRADING_ENVIRONMENT env var ("demo" | "production")
    2. Or explicit constructor parameter
    
    Credentials loaded from:
    - Demo: KALSHI_PAPER_TRADING_API_KEY_ID, KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT
    - Prod: KALSHI_TRADING_API_KEY_ID, KALSHI_TRADING_PRIVATE_KEY_CONTENT
    """
    
    # Environment-specific endpoints
    ENDPOINTS = {
        "demo": {
            "rest_url": "https://demo-api.kalshi.co/trade-api/v2",
            "ws_url": "wss://demo-api.kalshi.co/trade-api/ws/v2"
        },
        "production": {
            "rest_url": "https://trading-api.kalshi.com/trade-api/v2",
            "ws_url": "wss://trading-api.kalshi.com/trade-api/ws/v2"
        }
    }
    
    def __init__(self, environment: str = None):
        """
        Initialize trading client.
        
        Args:
            environment: "demo" or "production" (defaults to KALSHI_TRADING_ENVIRONMENT)
        """
        self.environment = environment or os.getenv("KALSHI_TRADING_ENVIRONMENT", "demo")
        
        if self.environment not in self.ENDPOINTS:
            raise ValueError(f"Invalid environment: {self.environment}. Must be 'demo' or 'production'")
        
        # Select credentials based on environment
        if self.environment == "demo":
            self.api_key_id = config.KALSHI_PAPER_TRADING_API_KEY_ID
            self.private_key_content = config.KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT
        else:
            self.api_key_id = config.KALSHI_TRADING_API_KEY_ID
            self.private_key_content = config.KALSHI_TRADING_PRIVATE_KEY_CONTENT
        
        # Set endpoints
        self.rest_base_url = self.ENDPOINTS[self.environment]["rest_url"]
        self.ws_url = self.ENDPOINTS[self.environment]["ws_url"]
        
        # ... rest of initialization
        
        logger.info(f"KalshiTradingClient initialized for {self.environment} environment")
```

### Config Changes Required

Add to `config.py`:

```python
# Trading Environment Selection
self.KALSHI_TRADING_ENVIRONMENT: str = os.getenv("KALSHI_TRADING_ENVIRONMENT", "demo")

# Production Trading Credentials (separate from paper trading)
self.KALSHI_TRADING_API_KEY_ID: Optional[str] = os.getenv("KALSHI_TRADING_API_KEY_ID")
self.KALSHI_TRADING_PRIVATE_KEY_CONTENT: Optional[str] = os.getenv("KALSHI_TRADING_PRIVATE_KEY_CONTENT")
```

### Environment Files

**.env.demo** (paper trading):
```bash
KALSHI_TRADING_ENVIRONMENT=demo
KALSHI_PAPER_TRADING_API_KEY_ID=your_demo_key
KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT=your_demo_private_key
```

**.env.production** (real trading):
```bash
KALSHI_TRADING_ENVIRONMENT=production
KALSHI_TRADING_API_KEY_ID=your_production_key
KALSHI_TRADING_PRIVATE_KEY_CONTENT=your_production_private_key
```

### Safety Guards

```python
class KalshiTradingClient:
    def __init__(self, environment: str = None):
        # ...
        
        # SAFETY: Require explicit confirmation for production
        if self.environment == "production":
            if not os.getenv("KALSHI_PRODUCTION_CONFIRMED", "").lower() == "true":
                raise ValueError(
                    "Production trading requires KALSHI_PRODUCTION_CONFIRMED=true. "
                    "This is a safety check to prevent accidental production trades."
                )
            logger.warning("⚠️ PRODUCTION TRADING ENABLED - Real money at risk!")
```

---

## 📊 Environment Tracking for Orderbook Sessions

### Problem Statement

Currently, `rl_orderbook_sessions` table doesn't track which Kalshi environment the data was collected from. This matters because:

1. Demo and production orderbooks may have different liquidity/spread characteristics
2. Training on demo data vs production data could affect model performance
3. We need to know data provenance for debugging and analysis

### Current Session Schema

```sql
CREATE TABLE rl_orderbook_sessions (
    session_id BIGSERIAL PRIMARY KEY,
    market_tickers TEXT[] NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL,
    websocket_url TEXT,  -- Currently stores the URL but not parsed
    connection_metadata JSONB,
    -- ...
);
```

### Solution: Add Environment Column

**Schema Migration:**

```sql
-- Add environment column to existing sessions table
ALTER TABLE rl_orderbook_sessions 
    ADD COLUMN IF NOT EXISTS environment VARCHAR(20) DEFAULT 'demo';

-- Add constraint for valid environments
ALTER TABLE rl_orderbook_sessions 
    ADD CONSTRAINT chk_sessions_environment 
    CHECK (environment IN ('demo', 'production', 'unknown'));

-- Backfill existing sessions based on websocket_url
UPDATE rl_orderbook_sessions 
SET environment = CASE 
    WHEN websocket_url LIKE '%demo-api%' THEN 'demo'
    WHEN websocket_url LIKE '%trading-api.kalshi.com%' THEN 'production'
    ELSE 'unknown'
END
WHERE environment IS NULL OR environment = 'demo';

-- Add index for environment filtering
CREATE INDEX IF NOT EXISTS idx_sessions_environment 
    ON rl_orderbook_sessions(environment);
```

**Database Code Update (`database.py`):**

```python
async def create_session(
    self, 
    market_tickers: List[str], 
    websocket_url: str = None,
    environment: str = "demo"  # NEW PARAMETER
) -> int:
    """Create a new orderbook session and return its ID."""
    async with self.get_connection() as conn:
        session_id = await conn.fetchval('''
            INSERT INTO rl_orderbook_sessions (
                market_tickers, websocket_url, environment, status, started_at
            ) VALUES ($1, $2, $3, 'active', CURRENT_TIMESTAMP)
            RETURNING session_id
        ''', market_tickers, websocket_url, environment)
        
        logger.info(f"Created orderbook session {session_id} ({environment}) for {len(market_tickers)} markets")
        return session_id
```

**OrderbookClient Update:**

```python
class OrderbookClient:
    def __init__(self, market_tickers: Optional[List[str]] = None):
        # ...
        self.ws_url = config.KALSHI_WS_URL
        
        # Determine environment from URL
        self.environment = self._detect_environment(self.ws_url)
    
    def _detect_environment(self, ws_url: str) -> str:
        """Detect environment from WebSocket URL."""
        if "demo-api" in ws_url:
            return "demo"
        elif "trading-api.kalshi.com" in ws_url or "api.elections.kalshi.com" in ws_url:
            return "production"
        else:
            return "unknown"
    
    async def _connect_and_subscribe(self) -> None:
        # ...
        
        # Create new session with environment
        self._session_id = await rl_db.create_session(
            market_tickers=self.market_tickers,
            websocket_url=self.ws_url,
            environment=self.environment  # Pass environment
        )
```

### Querying by Environment

```python
# Get only production sessions for analysis
async def get_production_sessions() -> List[Dict]:
    async with rl_db.get_connection() as conn:
        rows = await conn.fetch("""
            SELECT * FROM rl_orderbook_sessions 
            WHERE environment = 'production' AND status = 'closed'
            ORDER BY started_at DESC
        """)
        return [dict(row) for row in rows]

# Filter training data by environment
async def load_session(self, session_id: int) -> Optional[SessionData]:
    session_info = await self._db.get_session(session_id)
    
    # Log environment for debugging
    logger.info(f"Loading session {session_id} from {session_info.get('environment', 'unknown')} environment")
    # ...
```

### Implementation Order

1. [ ] Run schema migration to add `environment` column
2. [ ] Backfill existing sessions based on `websocket_url`
3. [ ] Update `create_session()` to accept environment parameter
4. [ ] Update `OrderbookClient` to detect and pass environment
5. [ ] Update session listing scripts to show environment
6. [ ] Refactor `KalshiDemoTradingClient` → `KalshiTradingClient`
7. [ ] Add safety guards for production trading
8. [ ] Update `.env.example` with new config variables
