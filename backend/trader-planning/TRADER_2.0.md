# TRADER 2.0 - Focused Extraction Plan

## Goal
Extract the WORKING pieces from the 5,665-line OrderManager into a clean, maintainable system with functional parity. No new features. Just clean up what works.

## Core Design Philosophy
Think like a videogame bot:
- Simple state machine: IDLE → CALIBRATING → READY ↔ ACTING
- Clear calibration: system_check → sync_data → act
- Self-recovery on errors
- Always know what state we're in

## What We're Extracting (Working Features Only)

### From OrderManager (5,665 lines) we extract:
1. **Order Management** (lines 3749-4155, 1424-1475)
   - place_order() 
   - cancel_order()
   - Order tracking dict
   
2. **Position Tracking** (lines 1746-1879, 1509-1745)
   - Update from fills
   - Track P&L
   - Position state dict

3. **State Sync** (lines 2546-3158)
   - Sync positions from API
   - Sync orders from API
   - Sync balance from API

4. **Status Logging** (lines 2096-2182)
   - The trader status log that's critical for debugging
   - Status history
   - Copy-paste format

5. **WebSocket Listeners** (already working)
   - Fill listener
   - Position listener
   - Connection management

### Existing Working Components to Keep:
6. **OrderbookClient** (`src/kalshiflow_rl/data/orderbook_client.py`)
   - WebSocket orderbook listener (when not 503ing)
   - Market data state management
   - Event emission

7. **ActorService** (`src/kalshiflow_rl/trading/actor_service.py`)
   - Decision loop
   - Action processing queue
   - Event subscription

8. **LiveObservationAdapter** (`src/kalshiflow_rl/trading/live_observation_adapter.py`)
   - Converts orderbook → observation space
   - Provides get_observation()

9. **EventBus** (`src/kalshiflow_rl/trading/event_bus.py`)
   - Connects components
   - Pub/sub pattern

## New Architecture (5 Extracted Services + 4 Existing)

### Extracted from OrderManager:
```python
# 1. OrderService (~500 lines)
class OrderService:
    """Handles order lifecycle only"""
    - place_order()
    - cancel_order() 
    - get_open_orders()
    - handle_fill_event()
    
# 2. PositionTracker (~400 lines)  
class PositionTracker:
    """Tracks positions from fills"""
    - update_from_fill()
    - get_position()
    - calculate_pnl()
    
# 3. StateSync (~300 lines)
class StateSync:
    """Reconciles with Kalshi API - NO hypothetical cash management"""
    - sync_positions()
    - sync_orders()
    - sync_balance()  # Just read from API, no add/withdraw nonsense
    
# 4. StatusLogger (~200 lines)
class StatusLogger:
    """The critical debugging tool - MUST PRESERVE"""
    - log_status()
    - get_status_history()
    - format_for_copy_paste()

# 5. TraderCoordinator (~200 lines)
class TraderCoordinator:
    """Thin orchestration layer"""
    - calibrate()
    - process_action()
    - get_state()
```

### Keep Existing (Already Working):
- **OrderbookClient** - Market data WebSocket
- **ActorService** - Decision making
- **LiveObservationAdapter** - Observation building
- **EventBus** - Message routing

**Total:** ~3,250 lines (vs 5,665 current monolith)

## Data Flow

```
OrderbookClient → EventBus → ActorService → LiveObservationAdapter → RL Model
                                    ↓
                              Action Decision
                                    ↓
                            TraderCoordinator
                                    ↓
                             OrderService
                                    ↓
                            Kalshi API/WebSocket
                                    ↓
                        Fill/Position Listeners
                                    ↓
                          PositionTracker
                                    ↓
                            StatusLogger
```

## State Machine (Game Bot Style)

```
States:
- IDLE: Not started
- CALIBRATING: Running system_check → sync_data
- READY: Calibrated and waiting for actions
- ACTING: Processing an action
- ERROR: Self-recovery mode

Transitions:
IDLE → CALIBRATING (on start)
CALIBRATING → READY (on success)
CALIBRATING → ERROR (on fail, retry)
READY ↔ ACTING (on actions)
ANY → ERROR → CALIBRATING (self-recovery)
```

## Calibration Flow (Simple)

```python
async def calibrate():
    # 1. System Check
    await check_exchange_status()
    await verify_websocket_connections()
    
    # 2. Sync Data (from Kalshi API)
    positions = await sync_positions_from_api()
    orders = await sync_orders_from_api()
    balance = await sync_balance_from_api()
    
    # 3. Ready to Act
    set_state(READY)
    log_status("Calibration complete")
```

## What We're NOT Building
- ❌ Hypothetical cash management systems
- ❌ Add/withdraw funds methods  
- ❌ Complex reservation tokens
- ❌ 10+ service architectures
- ❌ Migration strategies
- ❌ Feature flags
- ❌ Theoretical monitoring
- ❌ 5-week timelines

## Implementation Plan (1 Week)

### Day 1-2: Extract Core Services
- Pull OrderService from OrderManager (place_order, cancel_order)
- Pull PositionTracker from OrderManager (position updates from fills)
- Pull StatusLogger (preserve debugging capability)

### Day 3-4: State Management
- Implement simple state machine (5 states)
- Extract StateSync service (sync with API)
- Build TraderCoordinator (thin orchestration)

### Day 5-6: Integration
- Wire services together
- Connect to existing ActorService
- Connect to existing OrderbookClient
- Test with existing WebSocket listeners

### Day 7: Testing & Cleanup
- End-to-end testing
- Remove old OrderManager
- Verify functional parity

## Success Criteria
- ✅ Same capabilities as current system
- ✅ Orders place/cancel correctly
- ✅ Positions track from fills
- ✅ State syncs with Kalshi API
- ✅ Status log works for debugging
- ✅ Under 3,500 lines total (vs 5,665)
- ✅ Works with existing ActorService
- ✅ Handles orderbook 503 errors gracefully

## File Structure
```
trading/
├── services/
│   ├── order_service.py (~500 lines) [NEW - extracted]
│   ├── position_tracker.py (~400 lines) [NEW - extracted]
│   ├── state_sync.py (~300 lines) [NEW - extracted]
│   └── status_logger.py (~200 lines) [NEW - extracted]
├── coordinator.py (~200 lines) [NEW - extracted]
├── state_machine.py (~100 lines) [NEW]
├── actor_service.py [EXISTING - keep]
├── live_observation_adapter.py [EXISTING - keep]
├── event_bus.py [EXISTING - keep]
└── kalshi_multi_market_order_manager.py [DELETE after extraction]
```

## Key Extractions from OrderManager

### Order Placement (extract lines 3749-3833)
```python
# Current: OrderManager.place_order()
# Becomes: OrderService.place_order()
# Remove all the cash reservation complexity
# Just place order and track it
```

### Position Update from Fill (extract lines 1746-1879)
```python
# Current: OrderManager._update_position_from_fill()
# Becomes: PositionTracker.update_from_fill()
# Keep the working position tracking logic
```

### Status Logging (extract lines 2096-2182) - CRITICAL
```python
# Current: OrderManager._update_trader_status()
# Becomes: StatusLogger.log_status()
# MUST PRESERVE - this is our debugging lifeline
```

### State Sync (extract lines 2546-3158)
```python
# Current: sync_positions_from_api + sync_orders_from_api
# Becomes: StateSync.sync_positions(), sync_orders(), sync_balance()
# Just read from API - no fancy cash management
```

## Handling the 503 Orderbook Issue

Since orderbook WebSocket returns 503 on demo environment:
1. OrderbookClient continues trying to connect (existing behavior)
2. System can still trade without orderbook (use last known prices)
3. ActorService can use fallback observations
4. Don't block trading on orderbook availability

## No Overthinking

This is an EXTRACTION, not a redesign. We're:
1. Taking the working code from the god object
2. Organizing it into logical services  
3. Keeping the debugging tools
4. Maintaining functional parity
5. Shipping in 1 week

Focus on getting the working pieces out and organized. That's it.