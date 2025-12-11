# Kalshi Actor/Trader Service Implementation Plan

## Overview

This plan outlines the implementation of a live Kalshi actor/trader service that operates as the production counterpart to the training environment. The actor subscribes to live orderbook updates, uses a trained RL model for decision-making, executes trades via the OrderManager, and broadcasts trading events to the frontend.

## System Architecture Understanding

### Training Pipeline (Existing)
- **Session Data** → `SessionDataLoader` loads historical orderbook data from database
- **Environment** → `MarketAgnosticKalshiEnv` uses pre-loaded session data
- **Feature Extraction** → `build_observation_from_session_data()` creates 52-feature observations
- **Action Execution** → `LimitOrderActionSpace` + `SimulatedOrderManager` for training
- **Model Training** → RL agent learns from episodes generated from session data

### Live Pipeline (To Be Built)
- **Orderbook Updates** → `OrderbookClient` subscribes to live Kalshi WebSocket (discovery mode, limit 100)
- **Orderbook State** → `SharedOrderbookState` maintains current market state
- **Feature Extraction** → Same `build_observation_from_session_data()` logic (reused)
- **Model Inference** → Load trained model and get actions
- **Action Execution** → `LimitOrderActionSpace` + `KalshiOrderManager` with `KalshiDemoTradingClient`
- **Event Broadcasting** → WebSocket to frontend for visualization

## Core Components

### Existing Components to Reuse

1. **OrderbookClient** (`backend/src/kalshiflow_rl/data/orderbook_client.py`)
   - Already supports discovery mode via `RL_MARKET_MODE=discovery` and `ORDERBOOK_MARKET_LIMIT=100`
   - Subscribes to orderbook_delta channel
   - Updates `SharedOrderbookState` for each market
   - Can be reused directly

2. **OrderManager** (`backend/src/kalshiflow_rl/trading/order_manager.py`)
   - `KalshiOrderManager` already implemented for live trading
   - Integrates with `KalshiDemoTradingClient` for demo API
   - Handles order placement, cancellation, fill tracking
   - Position tracking and P&L calculation
   - **Status**: Ready to use

3. **KalshiDemoTradingClient** (`backend/src/kalshiflow_rl/trading/demo_client.py`)
   - Full demo API integration
   - Order creation, cancellation, position tracking
   - WebSocket support for fill tracking
   - **Status**: Ready to use

4. **LimitOrderActionSpace** (`backend/src/kalshiflow_rl/environments/limit_order_action_space.py`)
   - Converts discrete actions (0-4) to order intents
   - Works with both SimulatedOrderManager and KalshiOrderManager
   - **Status**: Ready to use

5. **Feature Extractors** (`backend/src/kalshiflow_rl/environments/feature_extractors.py`)
   - `build_observation_from_session_data()` - used in training
   - Need to create `build_observation_from_orderbook_state()` for live data
   - **Status**: Logic exists, needs live data adapter

6. **WebSocketManager** (`backend/src/kalshiflow_rl/websocket_manager.py`)
   - Already broadcasts orderbook updates to frontend
   - Can be extended to broadcast trading events
   - **Status**: Can be extended

### New Components to Build

1. **KalshiActorService** (NEW)
   - Main service that orchestrates the actor loop
   - Subscribes to orderbook updates
   - Manages model inference
   - Coordinates order execution
   - Broadcasts trading events
   - **Location**: `backend/src/kalshiflow_rl/actor/kalshi_actor_service.py`

2. **Live Observation Builder** (NEW)
   - Adapts `build_observation_from_session_data()` for live orderbook data
   - Converts `SharedOrderbookState` → observation format
   - Maintains observation history for temporal features
   - **Location**: `backend/src/kalshiflow_rl/actor/live_observation_builder.py`

3. **Model Inference Handler** (NEW)
   - Loads trained model from ModelRegistry
   - Handles model inference with proper observation format
   - Manages model hot-reloading
   - **Location**: `backend/src/kalshiflow_rl/actor/model_inference.py`

4. **Trading Event Broadcaster** (NEW)
   - Extends WebSocketManager to broadcast trading events
   - Formats trading events (actions, orders, fills, positions)
   - Separate channel for actor trading events
   - **Location**: `backend/src/kalshiflow_rl/actor/trading_event_broadcaster.py`

5. **Actor WebSocket Endpoint** (NEW)
   - New WebSocket endpoint `/rl/actor/ws` for frontend
   - Broadcasts actor trading events
   - Separate from orderbook WebSocket
   - **Location**: `backend/src/kalshiflow_rl/actor/websocket_endpoint.py`

6. **Frontend Actor View** (NEW)
   - React component to display actor trading activity
   - Shows current positions, recent trades, P&L
   - Real-time updates via WebSocket
   - **Location**: `frontend/src/components/ActorTradingView.jsx`

## Implementation Details

### 1. KalshiActorService

**File**: `backend/src/kalshiflow_rl/actor/kalshi_actor_service.py`

**Responsibilities**:
- Initialize OrderbookClient with discovery mode (limit 100)
- Initialize KalshiOrderManager with KalshiDemoTradingClient
- Load trained model via ModelRegistry
- Main actor loop:
  1. Receive orderbook delta/update
  2. Build observation from current orderbook state
  3. Get action from model
  4. Execute action via LimitOrderActionSpace + KalshiOrderManager
  5. Broadcast trading event to frontend
- Handle model hot-reloading
- Graceful shutdown

**Key Methods**:
```python
class KalshiActorService:
    async def start(self) -> None
    async def stop(self) -> None
    async def _actor_loop(self) -> None
    async def _process_orderbook_update(self, market_ticker: str, orderbook: OrderbookState) -> None
    async def _execute_trading_step(self, market_ticker: str) -> None
```

### 2. Live Observation Builder

**File**: `backend/src/kalshiflow_rl/actor/live_observation_builder.py`

**Responsibilities**:
- Convert `SharedOrderbookState` to observation format compatible with training
- Maintain observation history for temporal features
- Extract portfolio features from OrderManager
- Reuse existing feature extraction logic where possible

**Key Methods**:
```python
def build_observation_from_live_orderbook(
    orderbook_state: OrderbookState,
    historical_states: List[OrderbookState],
    position_data: Dict[str, Any],
    portfolio_value: int,
    cash_balance: int
) -> np.ndarray
```

**Implementation Strategy**:
- Reuse `extract_market_agnostic_features()` from feature_extractors.py
- Convert OrderbookState to the dict format expected by feature extractors
- Maintain last 10 orderbook states for temporal features
- Extract portfolio features from OrderManager.get_position_info()

### 3. Model Inference Handler

**File**: `backend/src/kalshiflow_rl/actor/model_inference.py`

**Responsibilities**:
- Load model from ModelRegistry
- Handle inference with proper observation format
- Support model hot-reloading
- Cache inference results for performance

**Key Methods**:
```python
class ModelInferenceHandler:
    async def load_model(self, model_id: Optional[int] = None) -> bool
    async def predict(self, observation: np.ndarray) -> int
    async def check_for_model_updates(self) -> None
```

**Implementation Notes**:
- Use ModelRegistry.load_model() to load trained model
- Ensure observation format matches training (52 features)
- Use deterministic=True for production inference
- Support hot-reloading without stopping actor loop

### 4. Trading Event Broadcaster

**File**: `backend/src/kalshiflow_rl/actor/trading_event_broadcaster.py`

**Responsibilities**:
- Extend WebSocketManager or create separate broadcaster
- Format trading events (actions, orders, fills, positions)
- Broadcast to all connected frontend clients

**Event Types**:
- `actor_action` - Action taken by model
- `actor_order` - Order placed/cancelled
- `actor_fill` - Order filled
- `actor_position` - Position update
- `actor_pnl` - P&L update

**Message Format**:
```json
{
  "type": "actor_action",
  "data": {
    "market_ticker": "MARKET-123",
    "action": 1,
    "action_name": "BUY_YES_LIMIT",
    "timestamp": "2025-01-10T15:30:00Z"
  }
}
```

### 5. Actor WebSocket Endpoint

**File**: `backend/src/kalshiflow_rl/actor/websocket_endpoint.py`

**Responsibilities**:
- New WebSocket endpoint `/rl/actor/ws`
- Accept frontend connections
- Broadcast trading events from TradingEventBroadcaster
- Handle connection lifecycle

**Integration**:
- Add to `backend/src/kalshiflow_rl/app.py` routing
- Use Starlette WebSocketEndpoint pattern
- Similar to existing `/rl/ws` endpoint

### 6. Frontend Actor View

**File**: `frontend/src/components/ActorTradingView.jsx`

**Responsibilities**:
- Display actor trading activity
- Show current positions, recent trades, P&L
- Real-time updates via WebSocket
- Simple, focused UI for monitoring

**Features**:
- Current positions table
- Recent trades/actions log
- P&L chart
- Connection status
- Model info

**WebSocket Hook**:
- Create `useActorWebSocket.js` hook similar to `useWebSocket.js`
- Connect to `ws://localhost:8000/rl/actor/ws`
- Handle actor event types

## Actor Loop Flow

```
1. OrderbookClient receives delta/update
   ↓
2. SharedOrderbookState updated
   ↓
3. KalshiActorService._process_orderbook_update() triggered
   ↓
4. LiveObservationBuilder builds observation from orderbook state
   ↓
5. ModelInferenceHandler.predict() gets action (0-4)
   ↓
6. LimitOrderActionSpace.execute_action() converts to order intent
   ↓
7. KalshiOrderManager.place_order() executes via demo API
   ↓
8. Order fills tracked via WebSocket or polling
   ↓
9. TradingEventBroadcaster broadcasts event to frontend
   ↓
10. Frontend ActorTradingView displays update
```

## Configuration

**Environment Variables**:
```bash
# Actor Service Configuration
RL_ACTOR_ENABLED=true
RL_ACTOR_MODEL_ID=<optional_model_id>  # None = use latest
RL_ACTOR_DISCOVERY_MODE=true
RL_ACTOR_MARKET_LIMIT=100
RL_ACTOR_TICK_INTERVAL=1.0  # Seconds between trading steps
RL_ACTOR_INITIAL_CASH=10000  # Starting cash in cents

# Demo Trading Client (already configured)
KALSHI_PAPER_TRADING_API_KEY_ID=<key>
KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT=<key_content>
KALSHI_PAPER_TRADING_API_URL=https://demo-api.kalshi.co/trade-api/v2
```

## Integration Points

### Backend Integration

1. **App Startup** (`backend/src/kalshiflow_rl/app.py`)
   - Initialize KalshiActorService in lifespan
   - Start actor service if `RL_ACTOR_ENABLED=true`
   - Register WebSocket endpoint `/rl/actor/ws`

2. **OrderbookClient Integration**
   - Actor subscribes to orderbook updates via SharedOrderbookState callbacks
   - No need to duplicate OrderbookClient - reuse existing instance

3. **WebSocketManager Extension**
   - Extend existing WebSocketManager or create separate broadcaster
   - Both can coexist (orderbook updates + trading events)

### Frontend Integration

1. **New Route**
   - Add `/actor` route to frontend router
   - Render ActorTradingView component

2. **WebSocket Connection**
   - New hook `useActorWebSocket.js`
   - Connect to `/rl/actor/ws` endpoint
   - Handle actor event types

## Testing Strategy

1. **Unit Tests**
   - LiveObservationBuilder observation format validation
   - ModelInferenceHandler model loading and inference
   - TradingEventBroadcaster event formatting

2. **Integration Tests**
   - End-to-end actor loop with mock orderbook updates
   - Order execution via KalshiDemoTradingClient
   - WebSocket event broadcasting

3. **E2E Tests**
   - Full actor loop with real orderbook data (no trades)
   - Frontend connection and event display
   - Model hot-reloading

## Deployment Considerations

1. **Service Startup**
   - Actor service starts automatically if enabled
   - Graceful shutdown on app termination
   - Health check endpoint `/rl/actor/health`

2. **Model Management**
   - Models stored in ModelRegistry
   - Hot-reloading without service restart
   - Fallback to previous model on load failure

3. **Monitoring**
   - Log all actions, orders, fills
   - Track inference latency
   - Monitor P&L and positions
   - Alert on errors or disconnections

## Future Enhancements (Out of Scope)

- Database persistence of actions/trades (mentioned as future)
- Multi-market trading coordination
- Risk management and position limits
- Performance optimization (caching, batching)
- Advanced model ensemble strategies

## File Structure

```
backend/src/kalshiflow_rl/
├── actor/                          # NEW
│   ├── __init__.py
│   ├── kalshi_actor_service.py     # Main actor service
│   ├── live_observation_builder.py # Observation building for live data
│   ├── model_inference.py          # Model loading and inference
│   ├── trading_event_broadcaster.py # Event broadcasting
│   └── websocket_endpoint.py       # WebSocket endpoint
├── data/
│   └── orderbook_client.py         # REUSE - orderbook subscription
├── trading/
│   ├── order_manager.py            # REUSE - KalshiOrderManager
│   └── demo_client.py              # REUSE - demo API client
├── environments/
│   ├── limit_order_action_space.py # REUSE - action execution
│   └── feature_extractors.py       # REUSE - feature logic
└── websocket_manager.py            # EXTEND - add trading events

frontend/src/
├── components/
│   └── ActorTradingView.jsx       # NEW - actor trading display
└── hooks/
    └── useActorWebSocket.js        # NEW - WebSocket hook for actor
```

## Implementation Order

1. **Live Observation Builder** - Foundation for observation building
2. **Model Inference Handler** - Model loading and inference
3. **Trading Event Broadcaster** - Event formatting and broadcasting
4. **KalshiActorService** - Main actor loop orchestration
5. **Actor WebSocket Endpoint** - Backend WebSocket integration
6. **Frontend Actor View** - UI for monitoring actor activity

## Key Design Decisions

1. **Reuse Existing Components**: Maximize reuse of OrderbookClient, OrderManager, ActionSpace
2. **Separate WebSocket Channel**: Trading events on separate channel from orderbook updates
3. **Same Observation Format**: Use same 52-feature observation as training for consistency
4. **Demo API Only**: Start with demo API (KalshiDemoTradingClient) for safety
5. **No Database Persistence**: Skip action/trade history persistence for MVP
6. **Single Market Focus**: Start with single-market trading, extend to multi-market later
