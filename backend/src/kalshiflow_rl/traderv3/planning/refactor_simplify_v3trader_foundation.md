# V3 Trader Refactoring: Simplify Foundation (REVISED)

## Executive Summary

This document provides a step-by-step plan to safely refactor the V3 trader from its current 5,000+ line monolithic structure into a clean, maintainable architecture. The refactoring preserves all functionality while making the system easier to understand, test, and extend.

**Key Principle: Extract and wrap, don't rewrite.**

## Current System Architecture (The Problem)

```
                           CURRENT V3 ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Frontend WebSocket Clients                                            │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    V3Coordinator (968 lines!)                          │
│  - State machine logic                                                 │
│  - Orderbook state management                                          │
│  - Trading state management                                            │
│  - Health monitoring loops                                             │
│  - Metrics collection                                                  │
│  - WebSocket broadcasting                                              │
│  - Error recovery                                                      │
│  - Sync orchestration                                                  │
└──────┬────────┬────────┬────────┬────────┬────────┬──────────────────┘
       │        │        │        │        │        │
       ▼        ▼        ▼        ▼        ▼        ▼
┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌──────────────┐ ┌─────────┐
│Orderbook │ │Trading Client│ │KalshiData │ │State         │ │Event    │
│Client    │ │Integration   │ │Sync       │ │Container     │ │Bus      │
│(985 lines)│ │(1,158 lines) │ │(508 lines)│ │(363 lines)   │ │7+ types │
│          │ │              │ │           │ │              │ │         │
│WORKS!    │ │- Orders      │ │- Sync API │ │- Dup state!  │ │         │
└──────────┘ │- Positions   │ │- Markets  │ │- More state! │ └─────────┘
             │- Also state! │ │- Orders   │ │- Why??       │
             └──────┬───────┘ └─────┬─────┘ └──────────────┘
                    │                │
                    ▼                ▼
             ┌──────────────────────────┐
             │  KalshiDemoTradingClient │
             │    (Actual API calls)    │
             └──────────────────────────┘

PROBLEMS:
1. State duplicated in 4+ places:
   Orderbooks: Coordinator + StateContainer + OrderbookClient
   Orders: Coordinator + TradingClientIntegration + StateContainer  
   Positions: Coordinator + TradingClientIntegration + KalshiDataSync

2. Services talk directly to each other:
   TradingClientIntegration ←---→ KalshiDataSync
   KalshiDataSync ←---→ StateContainer
   
3. Coordinator doing EVERYTHING (968 lines of mixed concerns)

4. No clear WebSocket event handling (scattered across services)
```

## The Core Issues

1. **State is everywhere** - Same data in StateContainer, Coordinator, TradingClientIntegration
2. **Coordinator is 968 lines** - Does orchestration, health, metrics, sync, everything  
3. **Services talk to each other** - Creates spaghetti dependencies
4. **OrderbookClient works but is buried** - 985 lines that work well but wrapped in complexity
5. **No WebSocket event handler** - Events scattered across multiple services

## Proposed Clean Architecture (The Solution)

```
                         CLEAN V3 ARCHITECTURE
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Frontend WebSocket Clients                                            │
│                                                                         │
└────────────────────────────┬────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────────────┐
        │         V3Coordinator (~250 lines)            │
        │                                                │
        │  ONLY DOES:                                    │
        │  - State machine transitions                   │
        │  - Call services based on state                │
        │  - Route messages to WebSocket                   │
        │  - Coordinate service locks                     │
        └──────────┬──────┬──────┬──────┬──────────────┘
                   │      │      │      │
                   ▼      ▼      ▼      ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐
        │ Orderbook    │ │ Trading      │ │ WebSocket    │ │ Health   │
        │ Service      │ │ Service      │ │ EventHandler │ │ Monitor  │
        │ (~150 lines) │ │ (~400 lines) │ │ (~200 lines) │ │(~100 lines)
        │              │ │              │ │              │ │          │
        │ OWNS:        │ │ OWNS:        │ │ OWNS:        │ │ OWNS:    │
        │ - Wrapper    │ │ - Orders     │ │ - WS events  │ │ - Health │
        │   for client │ │ - Positions  │ │ - Broadcasts │ │ - Metrics│
        └──────┬───────┘ └──────┬───────┘ └──────────────┘ └──────────┘
               │                 │
               ▼                 ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │ Orderbook    │ │ Kalshi       │ │ Lean State   │
        │ Client       │ │ Trading      │ │ Container    │
        │ (985 lines)  │ │ Client       │ │ (~100 lines) │
        │ KEEP AS-IS!  │ │              │ │ Version-safe │
        └──────────────┘ └──────────────┘ └──────────────┘

KEY IMPROVEMENTS:
1. Each service owns ONE thing completely
2. Services NEVER talk to each other (only to coordinator)
3. Coordinator ONLY orchestrates (no implementation)
4. OrderbookClient KEPT AS-IS (works well, just wrapped)
5. StateContainer TRANSFORMED to lean version (not deleted)
6. WebSocketEventHandler centralizes all WS events
7. HealthMonitor extracted for clean monitoring
```

## What Stays vs What Changes

### What Stays (Working Code)
- **OrderbookClient (985 lines)** - KEEP AS-IS, already has V3 integration hooks
- **KalshiDataSync (508 lines)** - KEEP, works for sync operations
- **State machine** - All states preserved for trading bot behavior
- **WebSocket messages** - Exact same format to frontend
- **Functionality** - Everything works identically

### What Changes (Simplifications)
- **StateContainer** - TRANSFORM to lean version (~100 lines from 363)
- **Coordinator** - EXTRACT concerns (~250 lines from 968)  
- **TradingClientIntegration** - CLEAN UP (~400 lines from 1,158)
- **NEW: WebSocketEventHandler** - Centralize WS events (~200 lines)
- **NEW: OrderbookService** - Thin wrapper for OrderbookClient (~150 lines)
- **NEW: HealthMonitor** - Extract health loops (~100 lines)

## State Versioning Protocol (Safety First)

Before ANY refactoring, implement state versioning to ensure safe rollback:

```python
# State versioning example - add to LeanStateContainer
class LeanStateContainer:
    def __init__(self):
        self._version = 0
        self._state = {
            "version": 0,
            "positions": {},
            "orders": {},
            "last_update": None
        }
        self._version_history = deque(maxlen=10)  # Keep last 10 versions
    
    def update_positions(self, positions: Dict):
        """Version-safe position update"""
        old_version = self._version
        self._version += 1
        
        # Save snapshot before update
        self._version_history.append({
            "version": old_version,
            "timestamp": datetime.now(),
            "state": deepcopy(self._state)
        })
        
        # Update state
        self._state["positions"] = positions
        self._state["version"] = self._version
        self._state["last_update"] = datetime.now()
        
        # Broadcast version change
        await self._broadcast_state_change(old_version, self._version)
    
    def rollback(self, to_version: int):
        """Emergency rollback to previous state version"""
        for snapshot in reversed(self._version_history):
            if snapshot["version"] == to_version:
                self._state = deepcopy(snapshot["state"])
                self._version = to_version
                logger.warning(f"ROLLBACK: State rolled back to version {to_version}")
                return True
        return False
```

## Implementation Roadmap (Safe, Incremental)

### Step 0: Add State Versioning (Safety First)
**Risk: Low | Impact: High | Lines: ~50**

```python
# Add to existing StateContainer first
class V3StateContainer:
    def __init__(self):
        self._version = 0  # ADD THIS
        # ... existing code ...
    
    def update_any_state(self):
        self._version += 1  # INCREMENT ON ANY UPDATE
        # ... existing update code ...
```

**Validation:** State version increments on updates, visible in status endpoint

---

### Step 1: Extract HealthMonitor (Safest Start)
**Risk: Very Low | Impact: Medium | Lines: ~100**

Extract health monitoring from Coordinator to dedicated service:

```python
# health_monitor.py (NEW FILE)
class HealthMonitor:
    def __init__(self):
        self._health_status = {}
        self._last_heartbeat = {}
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """Start health monitoring loops"""
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
    
    async def _monitor_loop(self):
        while True:
            try:
                # Check orderbook health
                orderbook_health = await self._check_orderbook_health()
                # Check trading health  
                trading_health = await self._check_trading_health()
                # Update status
                self._health_status = {
                    "orderbook": orderbook_health,
                    "trading": trading_health,
                    "timestamp": datetime.now()
                }
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            await asyncio.sleep(5)
    
    def get_status(self) -> Dict:
        return self._health_status
```

**Code to move from coordinator.py:**
- Lines 234-289: Health check loops
- Lines 712-756: Heartbeat monitoring
- Lines 823-867: Status compilation

**Validation:** Health endpoint still works, monitoring continues

---

### Step 2: Create WebSocketEventHandler (Critical Service)
**Risk: Low | Impact: High | Lines: ~200**

Centralize all WebSocket event handling:

```python
# websocket_event_handler.py (NEW FILE)
class WebSocketEventHandler:
    def __init__(self, broadcaster):
        self._broadcaster = broadcaster
        self._event_queue = asyncio.Queue()
        self._processing_task = None
    
    async def start(self):
        """Start event processing"""
        self._processing_task = asyncio.create_task(self._process_events())
    
    async def emit_event(self, event_type: str, data: Any):
        """Queue event for processing"""
        await self._event_queue.put({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now()
        })
    
    async def _process_events(self):
        """Process events and broadcast to clients"""
        while True:
            try:
                event = await self._event_queue.get()
                
                # Format for frontend
                message = self._format_message(event)
                
                # Broadcast to all clients
                await self._broadcaster.broadcast(message)
                
            except Exception as e:
                logger.error(f"Event processing error: {e}")
    
    def _format_message(self, event: Dict) -> Dict:
        """Format event for frontend consumption"""
        if event["type"] == "orderbook_update":
            return {
                "type": "orderbook",
                "data": event["data"],
                "timestamp": event["timestamp"].isoformat()
            }
        # ... other event types ...
```

**Validation:** WebSocket messages still reach frontend

---

### Step 3: Transform StateContainer to Lean Version
**Risk: Medium | Impact: High | Lines: ~100 from 363**

Transform, don't delete - keep essential state management:

```python
# lean_state_container.py (TRANSFORM from state_container.py)
class LeanStateContainer:
    """Minimal state container - just facts, no logic"""
    
    def __init__(self):
        self._version = 0
        self._lock = asyncio.Lock()
        
        # Essential state only
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._last_sync: Optional[datetime] = None
    
    async def get_positions(self) -> Dict[str, Position]:
        """Thread-safe position access"""
        async with self._lock:
            return deepcopy(self._positions)
    
    async def update_positions(self, positions: Dict[str, Position]):
        """Version-safe position update"""
        async with self._lock:
            self._version += 1
            self._positions = positions
            self._last_sync = datetime.now()
    
    async def get_orders(self) -> Dict[str, Order]:
        """Thread-safe order access"""
        async with self._lock:
            return deepcopy(self._orders)
    
    async def update_orders(self, orders: Dict[str, Order]):
        """Version-safe order update"""
        async with self._lock:
            self._version += 1
            self._orders = orders
    
    def get_version(self) -> int:
        return self._version
    
    def get_last_sync(self) -> Optional[datetime]:
        return self._last_sync
```

**What to remove from current StateContainer:**
- Complex event handling
- Direct service communication
- Business logic
- Redundant state copies

**Validation:** State still accessible, version increments work

---

### Step 4: Create OrderbookService Wrapper
**Risk: Low | Impact: Medium | Lines: ~150**

Wrap existing OrderbookClient without changing it:

```python
# orderbook_service.py (NEW FILE)
class OrderbookService:
    """Thin wrapper around existing OrderbookClient"""
    
    def __init__(self, orderbook_client: OrderbookClient, event_handler: WebSocketEventHandler):
        self._client = orderbook_client  # Use existing 985-line client AS-IS
        self._event_handler = event_handler
        self._orderbooks: Dict[str, OrderbookSnapshot] = {}
    
    async def start(self, market_tickers: List[str]):
        """Start orderbook client with V3 integration"""
        # OrderbookClient already has event_bus parameter for V3
        await self._client.connect()
        await self._client.subscribe_to_markets(market_tickers)
        
        # Set up event listener
        self._client.on_orderbook_update = self._handle_orderbook_update
    
    async def _handle_orderbook_update(self, market: str, orderbook: OrderbookSnapshot):
        """Process orderbook updates"""
        # Store locally
        self._orderbooks[market] = orderbook
        
        # Emit event for WebSocket broadcast
        await self._event_handler.emit_event("orderbook_update", {
            "market": market,
            "orderbook": orderbook.to_dict()
        })
    
    def get_orderbook(self, market: str) -> Optional[OrderbookSnapshot]:
        """Get current orderbook for market"""
        return self._orderbooks.get(market)
    
    def get_all_orderbooks(self) -> Dict[str, OrderbookSnapshot]:
        """Get all orderbooks"""
        return self._orderbooks.copy()
```

**Key Point:** OrderbookClient is NOT refactored, just wrapped!

**Validation:** Orderbook updates still flow to frontend

---

### Step 5: Clean TradingClientIntegration
**Risk: Medium | Impact: High | Lines: ~400 from 1,158**

Remove redundant state management, keep trading logic:

```python
# trading_service.py (REFACTORED from trading_client_integration.py)
class TradingService:
    """Clean trading service - owns orders and positions"""
    
    def __init__(self, trading_client: KalshiDemoTradingClient, 
                 state_container: LeanStateContainer,
                 event_handler: WebSocketEventHandler):
        self._client = trading_client
        self._state = state_container
        self._event_handler = event_handler
        
        # Local caches for performance
        self._active_orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
    
    async def place_order(self, request: OrderRequest) -> Order:
        """Place order with state tracking"""
        # Place via client
        order = await self._client.place_order(request)
        
        # Update local cache
        self._active_orders[order.id] = order
        
        # Update state container
        await self._state.update_orders(self._active_orders)
        
        # Emit event
        await self._event_handler.emit_event("order_placed", order.to_dict())
        
        return order
    
    async def sync_with_exchange(self):
        """Sync positions and orders with Kalshi"""
        # Get from exchange
        positions = await self._client.get_positions()
        orders = await self._client.get_orders()
        
        # Update local caches
        self._positions = {p.market: p for p in positions}
        self._active_orders = {o.id: o for o in orders if o.status == "active"}
        
        # Update state container
        await self._state.update_positions(self._positions)
        await self._state.update_orders(self._active_orders)
        
        # Emit sync complete event
        await self._event_handler.emit_event("sync_complete", {
            "positions": len(self._positions),
            "orders": len(self._active_orders)
        })
```

**What to remove from current TradingClientIntegration:**
- Direct communication with KalshiDataSync
- Duplicate state management
- Complex event routing

**Validation:** Orders place, positions track, sync works

---

### Step 6: Final Coordinator Cleanup
**Risk: Low | Impact: High | Lines: ~250 from 968**

Clean coordinator to pure orchestration:

```python
# coordinator.py (CLEANED)
class V3Coordinator:
    """Pure orchestrator - no implementation, just coordination"""
    
    def __init__(self, 
                 orderbook_service: OrderbookService,
                 trading_service: TradingService,
                 event_handler: WebSocketEventHandler,
                 health_monitor: HealthMonitor,
                 state_container: LeanStateContainer):
        # Services only, no state
        self.orderbook_service = orderbook_service
        self.trading_service = trading_service
        self.event_handler = event_handler
        self.health_monitor = health_monitor
        self.state = state_container
        
        # State machine
        self.state_machine = TradingStateMachine()
        
        # Coordination locks
        self._sync_lock = asyncio.Lock()
        self._trading_lock = asyncio.Lock()
    
    async def start(self):
        """Start all services in correct order"""
        # Start health monitoring
        await self.health_monitor.start_monitoring()
        
        # Start event handler
        await self.event_handler.start()
        
        # Start orderbook service
        await self.orderbook_service.start(self.market_tickers)
        
        # Initial sync
        await self.sync_with_exchange()
        
        # Transition to ready
        self.state_machine.transition_to(State.READY)
    
    async def handle_trade_signal(self, signal: TradeSignal):
        """Coordinate trading based on signal"""
        async with self._trading_lock:
            # Check state allows trading
            if not self.state_machine.can_trade():
                return
            
            # Transition to acting
            self.state_machine.transition_to(State.ACTING)
            
            try:
                # Get orderbook from service
                orderbook = self.orderbook_service.get_orderbook(signal.market)
                
                # Place order via service
                order = await self.trading_service.place_order(signal.to_order_request())
                
                # Emit success event
                await self.event_handler.emit_event("trade_executed", order.to_dict())
                
            except Exception as e:
                # Handle error
                logger.error(f"Trade failed: {e}")
                self.state_machine.transition_to(State.ERROR)
            
            finally:
                # Back to ready
                self.state_machine.transition_to(State.READY)
    
    async def sync_with_exchange(self):
        """Coordinate full sync"""
        async with self._sync_lock:
            await self.trading_service.sync_with_exchange()
            
    def get_status(self) -> Dict:
        """Aggregate status from all services"""
        return {
            "state": self.state_machine.current_state,
            "version": self.state.get_version(),
            "health": self.health_monitor.get_status(),
            "positions": len(self.state._positions),
            "orders": len(self.state._orders)
        }
```

**What to remove from current Coordinator:**
- All state management
- All implementation details  
- Health monitoring loops
- Direct WebSocket handling

**Validation:** Full system works end-to-end

---

## Risk Mitigation Strategies

### For Each Step:

1. **Version Everything**
   - Add version numbers to all state changes
   - Keep version history for rollback
   - Log version transitions

2. **Parallel Running**
   - Run new service alongside old code initially
   - Compare outputs to ensure parity
   - Switch over only when confirmed working

3. **Feature Flags**
   ```python
   USE_NEW_ORDERBOOK_SERVICE = os.getenv("USE_NEW_ORDERBOOK_SERVICE", "false") == "true"
   
   if USE_NEW_ORDERBOOK_SERVICE:
       service = OrderbookService(client)
   else:
       # Use old approach
   ```

4. **Comprehensive Logging**
   ```python
   logger.info(f"REFACTOR_CHECKPOINT: Step 1 - HealthMonitor extracted")
   logger.info(f"REFACTOR_VALIDATION: Health endpoint responding: {health_status}")
   ```

5. **Rollback Plan**
   - Each step is reversible via git
   - State versioning allows runtime rollback
   - Feature flags allow instant switchback

## Success Metrics

### Quantitative:
- **Line count reduction**: 5,000+ → ~2,000 lines total
- **Coordinator reduction**: 968 → ~250 lines
- **StateContainer reduction**: 363 → ~100 lines
- **Service isolation**: 0 service-to-service calls

### Qualitative:
- **Clear ownership**: Each service owns one domain
- **Simple coordination**: Coordinator only orchestrates
- **Easy debugging**: Linear flow, clear logs
- **Safe trading**: State versioning prevents corruption

## Common Pitfalls to Avoid

1. **Don't refactor OrderbookClient internals** - It works, just wrap it
2. **Don't delete StateContainer entirely** - Transform to lean version
3. **Don't skip state versioning** - Add it FIRST for safety
4. **Don't do all steps at once** - Each step should be validated
5. **Don't break WebSocket format** - Frontend expects exact format

## Concrete Code Examples

### Example 1: Coordinator Lock Pattern
```python
class V3Coordinator:
    async def critical_operation(self):
        """Example of safe coordination with locks"""
        async with self._operation_lock:
            # Check preconditions
            if not self.state_machine.can_operate():
                logger.warning("Operation blocked by state machine")
                return
            
            # Log operation start
            logger.info(f"Starting operation at version {self.state.get_version()}")
            
            try:
                # Perform operation
                result = await self.service.do_operation()
                
                # Update state
                await self.state.update_something(result)
                
                # Emit event
                await self.event_handler.emit_event("operation_complete", result)
                
            except Exception as e:
                # Log failure
                logger.error(f"Operation failed: {e}")
                
                # Attempt rollback
                await self.state.rollback(previous_version)
                
                # Transition to error state
                self.state_machine.transition_to(State.ERROR)
```

### Example 2: Service Never Talks to Service
```python
# WRONG - Services talking directly
class TradingService:
    def __init__(self, orderbook_service):  # ❌ Service dependency
        self.orderbook_service = orderbook_service
    
    async def place_order(self):
        orderbook = await self.orderbook_service.get_orderbook()  # ❌ Direct call

# RIGHT - Coordinator mediates
class TradingService:
    def __init__(self):  # ✓ No service dependencies
        pass
    
    async def place_order(self, orderbook: OrderbookSnapshot):  # ✓ Data passed in
        # Use provided orderbook
        
class V3Coordinator:
    async def coordinate_trade(self):
        # Coordinator gets from one service
        orderbook = self.orderbook_service.get_orderbook()
        
        # Coordinator passes to another service
        order = await self.trading_service.place_order(orderbook)
```

### Example 3: WebSocket Event Flow
```python
# Clean event flow through system
class OrderbookService:
    async def _handle_update(self, data):
        # 1. Process update
        orderbook = self._process(data)
        
        # 2. Emit to event handler (not directly to WS!)
        await self.event_handler.emit_event("orderbook_update", orderbook)

class WebSocketEventHandler:
    async def _process_events(self):
        event = await self._queue.get()
        
        # 3. Format for frontend
        message = self._format_for_frontend(event)
        
        # 4. Broadcast to clients
        await self.broadcaster.broadcast(message)

# Result: OrderbookService → EventHandler → Broadcaster → Clients
# Clean, traceable, debuggable flow
```

## Testing Strategy

### Unit Tests for Each Service
```python
# test_orderbook_service.py
async def test_orderbook_service_wraps_client():
    """Verify OrderbookService correctly wraps OrderbookClient"""
    mock_client = Mock(spec=OrderbookClient)
    service = OrderbookService(mock_client, mock_event_handler)
    
    # Test wrapper functionality
    await service.start(["MARKET1"])
    mock_client.connect.assert_called_once()
    mock_client.subscribe_to_markets.assert_called_with(["MARKET1"])

# test_health_monitor.py  
async def test_health_monitor_independence():
    """Verify HealthMonitor works independently"""
    monitor = HealthMonitor()
    await monitor.start_monitoring()
    
    # Should produce health status without dependencies
    status = monitor.get_status()
    assert "orderbook" in status
    assert "trading" in status
```

### Integration Tests for Coordination
```python
# test_coordinator_integration.py
async def test_coordinator_orchestration():
    """Verify Coordinator correctly orchestrates services"""
    # Create all services
    services = create_test_services()
    coordinator = V3Coordinator(**services)
    
    # Start system
    await coordinator.start()
    
    # Verify orchestration
    assert coordinator.state_machine.current_state == State.READY
    assert coordinator.state.get_version() > 0
    
    # Test trading flow
    signal = create_test_signal()
    await coordinator.handle_trade_signal(signal)
    
    # Verify coordination worked
    assert services["trading_service"].place_order.called
    assert services["event_handler"].emit_event.called
```

## Migration Checklist

### Pre-Migration
- [ ] Create feature branch `refactor/v3-simplification`
- [ ] Add state versioning to existing StateContainer
- [ ] Set up comprehensive logging
- [ ] Create rollback plan

### Step-by-Step Migration
- [ ] **Step 0**: State versioning implemented and tested
- [ ] **Step 1**: HealthMonitor extracted and validated
- [ ] **Step 2**: WebSocketEventHandler created and integrated
- [ ] **Step 3**: StateContainer transformed to lean version
- [ ] **Step 4**: OrderbookService wrapper created
- [ ] **Step 5**: TradingService cleaned up
- [ ] **Step 6**: Coordinator reduced to orchestration only

### Post-Migration
- [ ] All services under 400 lines
- [ ] No service-to-service communication
- [ ] State versioning working
- [ ] Full E2E test suite passes
- [ ] Performance metrics comparable or better
- [ ] Deployment successful

## Conclusion

This refactoring plan transforms a complex, tangled system into a clean, maintainable architecture while preserving all functionality. The key insights:

1. **OrderbookClient doesn't need refactoring** - Just wrap it
2. **StateContainer gets transformed, not deleted** - Keep essential state
3. **WebSocketEventHandler is critical** - Centralize all WS events  
4. **Services never talk to each other** - Only through coordinator
5. **State versioning provides safety** - Always able to rollback

By following this plan step-by-step with careful validation at each stage, we can safely evolve the V3 trader into a system that's easy to understand, test, and extend with new trading logic.

**Remember: Extract and wrap, don't rewrite. The code works - it just needs better organization.**