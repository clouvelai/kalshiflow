# Kalshi Actor/Trader Service Implementation Plan (v1)

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

## Implementation Order (Revised)

1. **Actor trigger wiring** (tap post-enqueue in `OrderbookClient` or `write_queue`)
2. **LiveObservationAdapter** (build 52-dim obs from `SharedOrderbookState.get_snapshot()` + portfolio state)
3. **ModelInferenceWrapper** (load model, predict action)
4. **ActorService** (queueing, throttling, execution through OrderManager)
5. **WebSocketManager extension** (broadcast actor events over `/rl/ws`)
6. **Frontend ActorTradingView** (display actor activity)

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
