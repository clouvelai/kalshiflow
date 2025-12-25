# V3 Event Loop Foundation Refactoring Plan

## Executive Summary
This plan refactors the V3 Coordinator's monolithic 361-line `start()` function into a clean event loop architecture. The goal is to maintain ALL existing functionality while creating a foundation that can later support trading states. No new features will be added - this is purely structural improvement.

## Current State Analysis

### The Problem: Monolithic start() Function
The current `start()` function (lines 83-361 in coordinator.py) handles:
1. **Component initialization** (Event Bus, WebSocket Manager, State Machine, Orderbook, Trading Client)
2. **Connection establishment** with actual waits and retries
3. **State transitions** with real metrics collection
4. **Data syncing** with Kalshi API
5. **Task spawning** for periodic monitoring

This 278-line function mixes startup logic with operational concerns, making it difficult to:
- Add trading states later (ACTING, CALIBRATING periodic syncs)
- Handle errors gracefully without full restart
- Test individual operations in isolation
- Understand the flow at a glance

### Existing Periodic Tasks (Already Running)
Three background tasks are spawned once READY:
1. **_monitor_health()** (lines 435-516): Checks component health every 30s, handles ERROR recovery
2. **_report_status()** (lines 645-656): Broadcasts status updates every 10s
3. **_monitor_trading_state()** (lines 657-716): Syncs trading state every 30s when trading enabled

## Design Principles

### 1. Separation of Concerns
- **Startup sequence**: One-time initialization (STARTUP → READY)
- **Event handlers**: Periodic operations after READY
- **Error recovery**: Isolated error handling without restart

### 2. Event Loop Pattern
```python
async def run(self):
    """Main event loop after startup."""
    while self._running:
        # Check for state-specific work
        if self._state_machine.current_state == V3State.READY:
            await self._handle_ready_state()
        elif self._state_machine.current_state == V3State.ERROR:
            await self._handle_error_state()
        
        # Small sleep to prevent CPU spinning
        await asyncio.sleep(0.1)
```

### 3. Handler Architecture
Each handler is a focused method that:
- Checks preconditions
- Performs one logical operation
- Updates state container
- Emits appropriate events
- Returns quickly (non-blocking)

## Refactoring Plan

### Phase 1: Extract Startup Sequence

#### Current Structure (lines 83-361)
```python
async def start(self) -> None:
    # 278 lines of mixed concerns:
    # - Component startup
    # - Connection waiting
    # - State transitions
    # - Metric collection
    # - Task spawning
```

#### New Structure
```python
async def start(self) -> None:
    """Start the V3 trader system - just initialization."""
    if self._running:
        logger.warning("V3 Coordinator is already running")
        return
    
    try:
        # Initialize components
        await self._initialize_components()
        
        # Establish connections
        await self._establish_connections()
        
        # Start event loop
        self._running = True
        self._event_loop_task = asyncio.create_task(self._run_event_loop())
        
        logger.info("✅ TRADER V3 STARTED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"Failed to start V3 Coordinator: {e}")
        await self.stop()
        raise
```

### Phase 2: Extract Connection Logic

#### New Method: _initialize_components()
```python
async def _initialize_components(self) -> None:
    """Initialize all core components in order."""
    logger.info("=" * 60)
    logger.info("STARTING TRADER V3")
    logger.info(f"Environment: {self._config.get_environment_name()}")
    logger.info("=" * 60)
    
    self._started_at = time.time()
    
    # Start components (no connections yet)
    logger.info("1/3 Starting Event Bus...")
    await self._event_bus.start()
    
    logger.info("2/3 Starting WebSocket Manager...")
    self._websocket_manager.set_coordinator(self)
    await self._websocket_manager.start()
    
    logger.info("3/3 Starting State Machine...")
    await self._state_machine.start()
    
    await self._emit_status_update("System initializing")
```

#### New Method: _establish_connections()
```python
async def _establish_connections(self) -> None:
    """Establish all external connections."""
    # Orderbook connection
    await self._connect_orderbook()
    
    # Trading client connection (if configured)
    if self._trading_client_integration:
        await self._connect_trading_client()
        await self._sync_trading_state()
    
    # Transition to READY with actual metrics
    await self._transition_to_ready()
```

### Phase 3: Create Event Loop

#### New Method: _run_event_loop()
```python
async def _run_event_loop(self) -> None:
    """
    Main event loop - handles all periodic operations.
    This is the heart of the V3 trader after startup.
    """
    # Start monitoring tasks
    self._start_monitoring_tasks()
    
    last_sync_time = time.time()
    sync_interval = 30.0  # Sync every 30 seconds
    
    while self._running:
        try:
            current_state = self._state_machine.current_state
            
            # State-specific handlers
            if current_state == V3State.READY:
                # Check if sync needed
                if self._trading_client_integration and \
                   time.time() - last_sync_time > sync_interval:
                    await self._handle_trading_sync()
                    last_sync_time = time.time()
                
                # Future: This is where ACTING state logic would go
                # if self._has_pending_actions():
                #     await self._handle_acting_state()
                
            elif current_state == V3State.ERROR:
                # Error recovery is handled by _monitor_health()
                # Just sleep longer in error state
                await asyncio.sleep(1.0)
                continue
            
            elif current_state == V3State.SHUTDOWN:
                # Exit loop on shutdown
                break
            
            # Small sleep to prevent CPU spinning
            await asyncio.sleep(0.1)
            
        except asyncio.CancelledError:
            logger.info("Event loop cancelled")
            break
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
            await self._state_machine.enter_error_state(
                "Event loop error", e
            )
```

### Phase 4: Extract Sync Handler

#### New Method: _handle_trading_sync()
```python
async def _handle_trading_sync(self) -> None:
    """
    Handle periodic trading state synchronization.
    Extracted from _monitor_trading_state() lines 671-702.
    """
    if not self._trading_client_integration:
        return
    
    if self._state_machine.current_state != V3State.READY:
        return
    
    logger.debug("Performing periodic trading state sync...")
    try:
        # Use the sync service to get fresh data
        state, changes = await self._trading_client_integration.sync_with_kalshi()
        
        # Always update sync timestamp
        state.sync_timestamp = time.time()
        
        # Update state container
        state_changed = self._state_container.update_trading_state(state, changes)
        
        # Log significant changes
        if changes and (abs(changes.balance_change) > 0 or 
                      changes.position_count_change != 0 or 
                      changes.order_count_change != 0):
            logger.info(
                f"Trading state updated - "
                f"Balance: ${state.balance/100:.2f} ({changes.balance_change:+d} cents), "
                f"Positions: {state.position_count} ({changes.position_count_change:+d}), "
                f"Orders: {state.order_count} ({changes.order_count_change:+d})"
            )
        
        # Broadcast state (even if unchanged, to update sync timestamp)
        if state_changed or True:  # Always broadcast on sync
            await self._emit_trading_state()
            
    except Exception as e:
        logger.error(f"Periodic trading sync failed: {e}")
        # Don't transition to ERROR for sync failures - just log and continue
```

### Phase 5: Simplify Monitoring Tasks

#### Current: Three Separate Monitoring Tasks
```python
# Current implementation spawns 3 tasks:
self._health_task = asyncio.create_task(self._monitor_health())
self._status_task = asyncio.create_task(self._report_status())  
self._trading_state_task = asyncio.create_task(self._monitor_trading_state())
```

#### New: Keep Health and Status, Integrate Trading Sync
```python
def _start_monitoring_tasks(self) -> None:
    """Start background monitoring tasks."""
    # Health monitoring - critical for error recovery
    self._health_task = asyncio.create_task(self._monitor_health())
    
    # Status reporting - for UI updates
    self._status_task = asyncio.create_task(self._report_status())
    
    # Trading state monitoring is now handled in main event loop
    # This gives us better control for future ACTING state
```

### Phase 6: Clean Helper Methods

#### Extract Connection Helpers
```python
async def _connect_orderbook(self) -> None:
    """Connect to orderbook WebSocket and wait for data."""
    logger.info("Connecting to orderbook...")
    
    # Start integration
    await self._orderbook_integration.start()
    
    # Wait for connection
    logger.info("🔄 Waiting for orderbook connection...")
    connection_success = await self._orderbook_integration.wait_for_connection(timeout=30.0)
    
    if not connection_success:
        raise RuntimeError("Failed to connect to orderbook WebSocket")
    
    # Wait for first snapshot
    logger.info("Waiting for initial orderbook snapshot...")
    data_flowing = await self._orderbook_integration.wait_for_first_snapshot(timeout=10.0)
    
    if not data_flowing:
        logger.warning("No orderbook data received - continuing anyway")
    
    # Collect metrics and transition
    metrics = self._orderbook_integration.get_metrics()
    await self._state_machine.transition_to(
        V3State.ORDERBOOK_CONNECT,
        context=f"Connected to {metrics['markets_connected']} markets",
        metadata={
            "markets_connected": metrics["markets_connected"],
            "snapshots_received": metrics["snapshots_received"],
            "ws_url": self._config.ws_url
        }
    )

async def _connect_trading_client(self) -> None:
    """Connect to trading API."""
    logger.info("Connecting to trading API...")
    
    await self._state_machine.transition_to(
        V3State.TRADING_CLIENT_CONNECT,
        context="Connecting to trading API",
        metadata={
            "mode": self._trading_client_integration._client.mode,
            "api_url": self._trading_client_integration.api_url
        }
    )
    
    connected = await self._trading_client_integration.wait_for_connection(timeout=30.0)
    
    if not connected:
        raise RuntimeError("Failed to connect to trading API")

async def _sync_trading_state(self) -> None:
    """Perform initial trading state sync."""
    logger.info("🔄 Syncing with Kalshi...")
    
    state, changes = await self._trading_client_integration.sync_with_kalshi()
    
    # Store in container
    self._state_container.update_trading_state(state, changes)
    
    await self._state_machine.transition_to(
        V3State.KALSHI_DATA_SYNC,
        context=f"Synced: {state.position_count} positions, {state.order_count} orders",
        metadata={
            "balance": state.balance,
            "positions": state.position_count,
            "orders": state.order_count
        }
    )
```

## Implementation Impact

### What Changes
1. **start() method**: Reduced from 278 lines to ~30 lines
2. **New methods**: 8 focused methods, each under 50 lines
3. **Event loop**: Central place for all periodic operations
4. **Clear separation**: Startup vs operational concerns

### What Stays the Same
1. **All functionality preserved**: Every operation remains
2. **State transitions unchanged**: Same sequence and timing
3. **Monitoring tasks continue**: Health, status, sync all work
4. **Error handling intact**: Same recovery mechanisms
5. **Metrics collection**: All metrics still gathered

### File Changes Required
Only one file needs modification:
- `backend/src/kalshiflow_rl/traderv3/core/coordinator.py`

No changes needed to:
- State machine
- Event bus
- State container
- Trading client integration
- Any other components

## Testing Strategy

### 1. Functionality Tests
Verify all existing operations still work:
```python
# Test startup sequence
assert coordinator.start() completes successfully
assert all components initialized in order
assert connections established with retries

# Test periodic operations
assert health monitoring continues every 30s
assert status reporting continues every 10s
assert trading sync happens every 30s when enabled

# Test error recovery
assert ERROR state triggers recovery attempts
assert health checks resume after recovery
```

### 2. Integration Tests
Run existing test suite to ensure no regression:
```bash
# Run V3 trader with paper account
./scripts/run-v3.sh --env paper

# Verify state progression
STARTUP → INITIALIZING → ORDERBOOK_CONNECT → TRADING_CLIENT_CONNECT → KALSHI_DATA_SYNC → READY

# Check logs for periodic operations
grep "Performing periodic trading state sync" 
grep "STATUS:" # Should see every 10s
```

### 3. Performance Validation
Ensure refactoring doesn't impact performance:
- CPU usage should remain low (event loop sleeps appropriately)
- Memory usage unchanged (no new data structures)
- WebSocket latency unaffected (same processing path)

## Migration Path

### Step 1: Create New Methods (Non-Breaking)
Add all new methods without changing start():
- _initialize_components()
- _establish_connections()
- _connect_orderbook()
- _connect_trading_client()
- _sync_trading_state()
- _transition_to_ready()
- _run_event_loop()
- _handle_trading_sync()

### Step 2: Test New Methods
Call new methods from test harness to validate:
```python
# Test each method in isolation
await coordinator._initialize_components()
assert components started

await coordinator._connect_orderbook()
assert orderbook connected
```

### Step 3: Refactor start()
Replace monolithic start() with calls to new methods:
```python
async def start(self) -> None:
    await self._initialize_components()
    await self._establish_connections()
    self._running = True
    self._event_loop_task = asyncio.create_task(self._run_event_loop())
```

### Step 4: Update _monitor_trading_state()
Simplify to just track version changes:
```python
async def _monitor_trading_state(self) -> None:
    """Monitor trading state version for broadcasts."""
    last_version = -1
    
    while self._running:
        try:
            await asyncio.sleep(1.0)
            
            # Just check for version changes
            # Syncing now happens in main event loop
            current_version = self._state_container.trading_state_version
            
            if current_version > last_version:
                await self._emit_trading_state()
                last_version = current_version
                
        except asyncio.CancelledError:
            break
```

### Step 5: Validate and Deploy
Run comprehensive tests to ensure:
- All state transitions work
- Periodic operations continue
- Error recovery functions
- No functionality lost

## Future Benefits

This refactoring enables:

### 1. Clean ACTING State Addition
```python
# Future: Easy to add trading logic
if current_state == V3State.READY:
    if self._has_pending_actions():
        await self._state_machine.transition_to(V3State.ACTING)
    
elif current_state == V3State.ACTING:
    await self._handle_acting_state()
    if self._actions_complete():
        await self._state_machine.transition_to(V3State.READY)
```

### 2. Testable Operations
Each handler can be tested independently:
```python
# Test sync handler without full startup
await coordinator._handle_trading_sync()
assert state updated
```

### 3. Observable State Transitions
Event loop makes state changes visible:
```python
# Easy to log/monitor state transitions
logger.info(f"Event loop iteration: state={current_state}, pending_actions={pending}")
```

### 4. Error Isolation
Errors in one handler don't affect others:
```python
# Sync failure doesn't stop health monitoring
try:
    await self._handle_trading_sync()
except Exception as e:
    logger.error(f"Sync failed: {e}")
    # Continue running, don't transition to ERROR
```

## Success Criteria

The refactoring is successful when:

1. ✅ **start() under 50 lines**: Focused on initialization only
2. ✅ **Event loop running**: Central control flow after startup  
3. ✅ **All operations preserved**: Nothing removed or broken
4. ✅ **Tests passing**: Existing test suite still works
5. ✅ **Logs unchanged**: Same log output (order may vary slightly)
6. ✅ **Performance stable**: No CPU/memory regression
7. ✅ **Code clarity improved**: Each method has single responsibility

## Code Snippets

### Complete New start() Method
```python
async def start(self) -> None:
    """Start the V3 trader system."""
    if self._running:
        logger.warning("V3 Coordinator is already running")
        return
    
    logger.info("=" * 60)
    logger.info("STARTING TRADER V3")
    logger.info(f"Environment: {self._config.get_environment_name()}")
    logger.info(f"Markets: {', '.join(self._config.market_tickers[:3])}...")
    logger.info("=" * 60)
    
    self._started_at = time.time()
    
    try:
        # Phase 1: Initialize components
        await self._initialize_components()
        
        # Phase 2: Establish connections
        await self._establish_connections()
        
        # Phase 3: Start event loop
        self._running = True
        self._event_loop_task = asyncio.create_task(self._run_event_loop())
        
        logger.info("=" * 60)
        logger.info("✅ TRADER V3 STARTED SUCCESSFULLY")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to start V3 Coordinator: {e}")
        await self.stop()
        raise
```

### Complete Event Loop
```python
async def _run_event_loop(self) -> None:
    """Main event loop for V3 trader operations."""
    # Start monitoring tasks
    self._health_task = asyncio.create_task(self._monitor_health())
    self._status_task = asyncio.create_task(self._report_status())
    
    last_sync_time = time.time()
    sync_interval = 30.0
    
    logger.info("Event loop started")
    
    while self._running:
        try:
            current_state = self._state_machine.current_state
            
            # Handle READY state operations
            if current_state == V3State.READY:
                # Periodic trading sync
                if (self._trading_client_integration and 
                    time.time() - last_sync_time > sync_interval):
                    await self._handle_trading_sync()
                    last_sync_time = time.time()
                
                # Future: Action handling would go here
                
            elif current_state == V3State.ERROR:
                # Error state - wait for recovery
                await asyncio.sleep(1.0)
                
            elif current_state == V3State.SHUTDOWN:
                # Clean shutdown requested
                break
            
            # Prevent CPU spinning
            await asyncio.sleep(0.1)
            
        except asyncio.CancelledError:
            logger.info("Event loop cancelled")
            break
        except Exception as e:
            logger.error(f"Event loop error: {e}")
            # Don't crash - try to recover
            await asyncio.sleep(1.0)
    
    logger.info("Event loop stopped")
```

## Conclusion

This refactoring plan transforms the monolithic `start()` function into a clean event loop architecture without changing any functionality. The new structure provides a solid foundation for future trading features while maintaining all existing operations. The implementation can be done incrementally with full testing at each step, ensuring no regression or functionality loss.

The key insight is that the current code already has the right components - it just needs better organization. By extracting focused methods and creating a central event loop, we make the code more maintainable, testable, and ready for future enhancements.