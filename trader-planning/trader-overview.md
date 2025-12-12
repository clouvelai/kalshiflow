# Kalshi Trading Actor MVP Design

## Executive Summary

This document outlines the MVP architecture for the Kalshi trading actor system, focusing on rapid deployment to the Kalshi demo environment for real-world iteration on edge cases. The design integrates with existing orderbook collector infrastructure while maintaining modular action selection capabilities.

## CRITICAL PERFORMANCE REQUIREMENTS

**Model Caching (MANDATORY)**: The RL model MUST be loaded once at startup and cached for all subsequent predictions. Loading the model on each orderbook update would be a performance disaster.

**Data Flow**: `orderbook update → construct observation → CACHED model.predict() → action → trade`

**Training Consistency**: Actor MUST use the same 10-contract size as training to ensure strategy consistency.

## Critical Architectural Decision: INTEGRATE WITH ORDERBOOK COLLECTOR

**RECOMMENDATION: Integrate with existing orderbook collector pipeline**

**Reasoning:**
1. **Faster MVP deployment** - Leverages proven WebSocket connection, data normalization, and state management
2. **Data consistency** - Uses exact same stream that feeds training data (no training/inference discrepancies)
3. **Proven reliability** - OrderbookClient + SharedOrderbookState + WriteQueue architecture already validated
4. **Resource efficiency** - One WebSocket connection, one authentication, one state management system
5. **Operational simplicity** - Single service to monitor, debug, and maintain

**Integration Point:** After `write_queue.enqueue_*()` returns True, trigger actor step

**Processing Model:** Serial queue-based processing for 1k markets using market-aware single OrderManager

## MVP Architecture Overview

```
                    LIVE TRADING ACTOR INTEGRATION DESIGN
                              
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EXISTING ORDERBOOK COLLECTOR                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Kalshi WS ──→ OrderbookClient ──→ SharedOrderbookState ──→ WriteQueue         │
│                                          │                       │              │
│                                          │                       │              │
│                                   [authoritative                 │              │
│                                    real-time state]              │              │
│                                          │                       │              │
│                                          │                       ↓              │
│                                          │               [only AFTER            │
│                                          │                successful            │
│                                          │                enqueue]              │
│                                          │                       │              │
└─────────────────────────────────────────┼───────────────────────┼──────────────┘
                                          │                       │
┌─────────────────────────────────────────┼───────────────────────┼──────────────┐
│                          NEW TRADING ACTOR COMPONENTS                          │
├─────────────────────────────────────────┼───────────────────────┼──────────────┤
│                                          │                       │              │
│                                          │         ┌─────────────▼──────────────┐
│                                          │         │     ActorEventTrigger      │
│                                          │         │  - Market ticker           │
│                                          │         │  - Update type            │
│                                          │         │  - Sequence number        │
│                                          │         │  - Trigger timestamp      │
│                                          │         └─────────────┬──────────────┘
│                                          │                       │
│        ┌─────────────────────────────────▼──┐                  │
│        │        ActorService                │                  │
│        │  - Queue actor events              │◄─────────────────┘
│        │  - Throttle to prevent overtrading │
│        │  - Coordinate full trading cycle   │
│        │  - Error handling & logging       │
│        └─────────────────┬──────────────────┘
│                          │
│                          ▼
│        ┌─────────────────────────────────────┐
│        │     LiveObservationAdapter          │
│        │  - Read from SharedOrderbookState   │
│        │  - Convert to SessionDataPoint fmt  │  ◄── CRITICAL: Must compute
│        │  - Compute temporal features         │      activity_score, momentum,
│        │  - Match training observation        │      time_gap on-the-fly
│        │  - Include portfolio state           │
│        └─────────────────┬───────────────────┘
│                          │
│                          ▼
│        ┌─────────────────────────────────────┐
│        │      ActionSelector                 │  ◄── MODULAR: RL Model OR
│        │  - Model inference OR               │      Hardcoded Strategy
│        │  - Hardcoded strategy               │
│        │  - Hot-reloadable models            │
│        │  - Strategy A/B testing             │
│        └─────────────────┬───────────────────┘
│                          │
│                          ▼
│        ┌─────────────────────────────────────┐
│        │  Serial Execution Pipeline          │
│        │  1. obs = build_observation()       │
│        │  2. action = select_action()        │
│        │  3. trade = safe_execute_action()   │  ◄── Market-aware
│        │  4. pos = update_positions(trade)   │      OrderManager
│        └─────────────────┬───────────────────┘
│                          │
│                          ▼
│        ┌─────────────────────────────────────┐
│        │   ActorEventBroadcaster            │
│        │  - Action taken                    │
│        │  - Order details                   │
│        │  - Position updates                │
│        │  - P&L changes                     │
│        │  - Error events                    │
│        └─────────────────┬───────────────────┘
│                          │
│                          ▼
│                WebSocketManager (/rl/ws)
│                          │
│                          ▼
│                   [Frontend Display]
│
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Core Integration Points

### 1. Trigger Integration (OrderbookClient)

**Where:** `OrderbookClient._process_snapshot()` and `OrderbookClient._process_delta()`
**When:** After `write_queue.enqueue_*()` returns True
**Implementation:**

```python
# In OrderbookClient._process_snapshot()
success = await write_queue.enqueue_snapshot(snapshot_data)
if success and self._actor_trigger:
    await self._actor_trigger(
        market_ticker=snapshot_data['market_ticker'],
        update_type='snapshot',
        sequence_number=snapshot_data.get('sequence', 0)
    )
```

## Serial Queue-Based Processing Architecture

### Clean 4-Step Pipeline
```python
class ActorService:
    def __init__(self, order_manager: KalshiOrderManager):
        self.order_manager = order_manager  # Single market-aware OrderManager
        self.update_queue = asyncio.Queue(maxsize=10000)  # Handle 1k markets
        self.obs_adapter = LiveObservationAdapter(order_manager)
        self.action_selector = ActionSelector()
        self.running = False
        
    async def process_market_update(self, market_ticker: str, snapshot: Dict, seq_id: int):
        """Clean 4-step serial pipeline - no complexity."""
        try:
            # Step 1: Build observation (async, non-blocking)
            obs = await self.build_observation_async(market_ticker, snapshot)
            
            # Step 2: Get action from model (cached model, fast)
            action = await self.action_selector.select_action(obs, market_ticker)
            
            # Step 3: Execute action safely (includes HOLD handling + error management)
            trade = await self.safe_execute_action(action, market_ticker, snapshot, seq_id)
            
            # Step 4: Update positions (clean separation of concerns)
            updated_pos = await self.update_positions(trade)
            
        except Exception as e:
            logger.error(f"Failed to process {market_ticker}: {e}")
            await self.log_actor_event("PIPELINE_ERROR", market_ticker, seq_id, error=str(e))

    async def safe_execute_action(self, action: int, market_ticker: str, 
                                 snapshot: Dict, seq_id: int) -> Optional[TradeResult]:
        """Execute action with comprehensive error handling and logging."""
        
        # Always log the action decision (including HOLD)
        await self.log_actor_event("ACTION", market_ticker, seq_id, 
                                  action=action, 
                                  portfolio_value=self.order_manager.get_total_portfolio_value())
        
        if action == 0:  # HOLD
            logger.debug(f"HOLD decision for {market_ticker}")
            return TradeResult(action=0, market_ticker=market_ticker, executed=False)
        
        try:
            # Execute via market-aware OrderManager
            execution_result = await self.order_manager.execute_limit_order_action(
                action=action,
                market_ticker=market_ticker,  # OrderManager handles per-market state
                orderbook_snapshot=snapshot
            )
            
            if execution_result:
                await self.log_actor_event("ORDER_PLACED", market_ticker, seq_id,
                                          order_id=execution_result.order_id,
                                          side=execution_result.side,
                                          quantity=execution_result.quantity,
                                          price=execution_result.limit_price)
                
                return TradeResult(
                    action=action,
                    market_ticker=market_ticker,
                    executed=True,
                    order_info=execution_result
                )
            else:
                await self.log_actor_event("ORDER_FAILED", market_ticker, seq_id,
                                          action=action, error="Execution returned None")
                return TradeResult(action=action, market_ticker=market_ticker, executed=False)
                
        except Exception as e:
            await self.log_actor_event("EXECUTION_ERROR", market_ticker, seq_id,
                                      action=action, error=str(e))
            logger.error(f"Execution failed for {market_ticker} action {action}: {e}")
            return TradeResult(action=action, market_ticker=market_ticker, executed=False, error=str(e))

    async def update_positions(self, trade: TradeResult) -> Optional[PositionUpdate]:
        """Update position tracking and portfolio state."""
        if not trade.executed:
            return None
            
        try:
            # OrderManager already updated positions during execution
            # This method handles additional logging/broadcasting
            
            updated_position = self.order_manager.get_position_for_market(trade.market_ticker)
            portfolio_value = self.order_manager.get_total_portfolio_value()
            
            position_update = PositionUpdate(
                market_ticker=trade.market_ticker,
                position=updated_position,
                portfolio_value=portfolio_value,
                timestamp=time.time()
            )
            
            # Broadcast position update via WebSocket
            await self.broadcast_position_update(position_update)
            
            return position_update
            
        except Exception as e:
            logger.error(f"Failed to update positions for {trade.market_ticker}: {e}")
            return None
```

## Market-Aware OrderManager Architecture

### Single OrderManager with Market-Specific Methods
```python
class KalshiOrderManager(OrderManager):
    """Enhanced OrderManager for multi-market live trading."""
    
    def __init__(self, demo_client, initial_cash=1000.0):
        super().__init__(initial_cash)
        self.client = demo_client
        # Market-aware state tracking
        self.per_market_throttle = {}  # market -> last_action_time
        self.active_markets = set()
        
    # NEW: Market-aware methods (not in training OrderManager)
    def get_position_for_market(self, market_ticker: str) -> Position:
        """Get position for specific market."""
        return self.positions.get(market_ticker, Position.empty())
    
    def get_orders_for_market(self, market_ticker: str) -> List[OrderInfo]:
        """Get orders for specific market."""
        return [o for o in self.open_orders.values() if o.ticker == market_ticker]
    
    def can_trade_market(self, market_ticker: str) -> bool:
        """Check if market can be traded (throttling, risk limits)."""
        if market_ticker in self.per_market_throttle:
            return time.time() > self.per_market_throttle[market_ticker] + 0.25
        return True
    
    async def execute_limit_order_action(self, action: int, market_ticker: str, 
                                        orderbook_snapshot: Dict) -> Optional[OrderInfo]:
        """Execute action on specific market with global portfolio constraints."""
        # Check market-specific throttling
        if not self.can_trade_market(market_ticker):
            logger.debug(f"Market {market_ticker} throttled")
            return None
            
        # Check global cash constraints
        if self.cash_balance < 100:  # Minimum for 10 contracts
            logger.warning(f"Insufficient cash for {market_ticker}")
            return None
            
        # Execute via existing action space
        result = await self.place_limit_order(action, market_ticker, orderbook_snapshot)
        
        # Update throttling
        if result:
            self.per_market_throttle[market_ticker] = time.time()
            
        return result
```

### Training Compatibility (No Changes)
```python
# SimulatedOrderManager stays EXACTLY as-is
class SimulatedOrderManager(OrderManager):
    # No market-awareness needed
    # Single market per episode
    # No additional complexity
```

### 2. Observation Building (Critical Component)

**Challenge:** Training uses pre-computed temporal features, live data needs on-the-fly computation

**Required LiveObservationAdapter implementation:**
```python
class LiveObservationAdapter:
    def __init__(self):
        self._history = deque(maxlen=10)  # Keep recent snapshots
        self._last_prices = {}
        # CRITICAL: Model loaded once and cached
        self._cached_model = None
        
    def load_model(self, model_path: str):
        """Load model once at startup - NEVER in hot path."""
        from stable_baselines3 import PPO
        self._cached_model = PPO.load(model_path)
        logger.info(f"Model loaded and cached: {model_path}")
        
    async def build_observation(
        self, 
        market_ticker: str, 
        shared_state: SharedOrderbookState,
        order_manager: OrderManager
    ) -> np.ndarray:
        # 1. Get current snapshot
        snapshot = await shared_state.get_snapshot()
        
        # 2. Convert to SessionDataPoint format
        markets_data = {market_ticker: snapshot}
        mid_prices = {
            market_ticker: (
                snapshot['yes_mid_price'], 
                snapshot['no_mid_price']
            )
        }
        
        # 3. Compute temporal features ON-THE-FLY
        time_gap = self._calculate_time_gap(snapshot)
        activity_score = self._compute_activity_score(markets_data)
        momentum = self._compute_momentum(market_ticker, mid_prices)
        
        # 4. Create SessionDataPoint
        data_point = SessionDataPoint(
            markets_data=markets_data,
            mid_prices=mid_prices,
            time_gap=time_gap,
            activity_score=activity_score,
            momentum=momentum,
            timestamp_ms=snapshot['last_update_time']
        )
        
        # 5. Use existing feature extraction
        return build_observation_from_session_data(
            data_point, 
            order_manager=order_manager
        )
```

### 3. Modular Action Selection

**Interface Design:**
```python
from abc import ABC, abstractmethod

class ActionSelector(ABC):
    @abstractmethod
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        """Select action (0-4) based on observation.
        
        Args:
            observation: 52-feature observation vector from LiveObservationAdapter
            market_ticker: Market context (used for logging/debugging, not model input)
                          since model is market-agnostic
        
        Note: market_ticker is NOT used as model input (model is market-agnostic)
              but may be useful for execution context, logging, or future strategy routing
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging."""
        pass

class RLModelSelector(ActionSelector):
    def __init__(self, model_path: str):
        # CRITICAL: Load model ONCE at initialization, not per prediction
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)
        self.model_path = model_path
        logger.info(f"RLModelSelector: Model loaded and cached from {model_path}")
        
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        # Use CACHED model - no loading here
        action, _states = self.model.predict(observation, deterministic=True)
        return int(action)
        
    def get_strategy_name(self) -> str:
        return f"RL_Model({self.model_path})"

class HardcodedSelector(ActionSelector):
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        # Simple strategy: buy when mid-price < 0.4, sell when > 0.6
        mid_price = observation[6]  # yes_mid_price_norm feature
        if mid_price < 0.4:
            return 1  # BUY_YES_LIMIT
        elif mid_price > 0.6:
            return 2  # SELL_YES_LIMIT
        return 0  # HOLD
        
    def get_strategy_name(self) -> str:
        return "Hardcoded_MeanReversion"
```

## Scalability & Performance Targets

### MVP Target: 1k Markets
- **Current Load**: 2-3 orderbook events/sec across 2k markets  
- **Processing Time**: ~9ms per market update (obs + predict + execute + position)
- **Expected Load**: 1.25 events/sec for 1k markets
- **CPU Usage**: ~1.125% (well within limits)

### Serial Processing Benefits
- **Simple Debugging**: Linear execution flow
- **No Race Conditions**: Single thread handles all portfolio state
- **Easy Monitoring**: Queue depth shows system health  
- **Predictable Performance**: 9ms × queue_size = processing lag

### Performance Circuit Breakers
- **Queue Overflow**: Drop updates if queue exceeds 10k (prevent memory explosion)
- **Slow Processing**: Alert if individual updates exceed 50ms
- **Model Performance**: Circuit breaker if prediction >5ms consistently
- **Cash Exhaustion**: Stop trading when insufficient cash globally

```python
# Performance monitoring in ActorService
async def process_with_monitoring(self, market_ticker: str, snapshot: Dict, seq_id: int):
    start_time = time.time()
    
    await self.process_market_update(market_ticker, snapshot, seq_id)
    
    processing_time = time.time() - start_time
    
    # Alert on slow processing
    if processing_time > 0.050:  # 50ms
        logger.warning(f"Slow processing: {market_ticker} took {processing_time:.3f}s")
    
    # Update performance metrics
    self.metrics['avg_processing_time'] = (
        self.metrics['avg_processing_time'] * 0.99 + processing_time * 0.01
    )

## Database Safety Protocol

**MANDATORY BEFORE ANY SCHEMA CHANGES:**

1. **Full Database Backup**:
   ```bash
   # Create timestamped backup of entire database
   pg_dump $DATABASE_URL > kalshiflow_backup_$(date +%Y%m%d_%H%M%S).sql
   
   # Verify backup integrity
   pg_restore --list kalshiflow_backup_$(date +%Y%m%d_%H%M%S).sql
   ```

2. **Critical Data Verification**:
   ```sql
   -- Verify our 200-300k+ training datapoints are safe
   SELECT COUNT(*) FROM rl_orderbook_snapshots; 
   SELECT COUNT(*) FROM rl_orderbook_deltas;
   SELECT COUNT(*) FROM rl_orderbook_sessions;
   ```

3. **Backup Storage**: Store backup in multiple locations (local + cloud)

4. **Recovery Test**: Verify backup can be restored to test database

**Never proceed with schema changes without completed backup verification.**

## MVP Implementation Plan

### Milestone 1: Database Safety & Actor Service Framework (Week 1)
**Time Estimate: 4-5 days (including safety procedures)**

**Components to implement:**
1. **Database Safety Protocol** (FIRST PRIORITY)
   - Complete database backup with verification
   - Test restore to staging environment
   - Document recovery procedures

2. **ActorService core loop**
   - Event queuing and throttling
   - Async execution coordination
   - Error handling and recovery
   - Configuration management
   - **Model caching initialization** (load once at startup)

2. **Integration with OrderbookClient**
   - Add actor trigger callback mechanism
   - Pass market context (ticker, type, sequence)
   - Non-blocking trigger execution

3. **Basic testing infrastructure**
   - Mock SharedOrderbookState for testing
   - Actor event validation
   - Trigger mechanism verification

**Success criteria:**
- Actor triggers on orderbook updates
- Events queued and processed without blocking WebSocket
- Basic logging and error handling working

### Milestone 2: Live Observation Builder (Week 2)
**Time Estimate: 4-5 days**

**Components to implement:**
1. **LiveObservationAdapter class**
   - Async SharedOrderbookState.get_snapshot() integration
   - SessionDataPoint format conversion
   - Temporal feature computation (activity_score, momentum, time_gap)
   - **Model loading and caching infrastructure** (load once, predict many)

2. **Temporal feature computation**
   - Port logic from session_data_loader.py:646-675
   - Maintain sliding window of historical data
   - Handle edge cases (insufficient data, missing features)

3. **Integration testing**
   - Validate observation matches training format exactly
   - Test with real SharedOrderbookState data
   - Performance benchmarking (<1ms per observation)

**Success criteria:**
- Live observations match training format (52 features)
- Temporal features computed correctly
- Performance target met (<1ms)

### Milestone 3: Modular Action Selection (Week 3)
**Time Estimate: 2-3 days**

**Components to implement:**
1. **ActionSelector interface and implementations**
   - Abstract base class
   - RLModelSelector with **cached model** (no hot-reload in MVP)
   - HardcodedSelector for rule-based strategies

2. **Cached model inference**
   - **Load model once at startup** (not hot-reload in MVP)
   - Model validation and error handling
   - Prediction timing monitoring (<1ms per prediction)
   - **Contract size validation** (ensure 10 matches training)

3. **Strategy configuration**
   - Environment-based strategy selection
   - A/B testing framework (future)
   - Strategy performance tracking

**Success criteria:**
- Both RL and hardcoded strategies working
- Model hot-reload functional
- Strategy switching without restart

### Milestone 4: ActorEvent Persistence (Week 4)
**Time Estimate: 2-3 days**

**Components to implement:**
1. **ActorEvent logging table**
   ```sql
   CREATE TABLE rl_actor_events (
       id BIGSERIAL PRIMARY KEY,
       session_id INTEGER,
       market_ticker VARCHAR(100) NOT NULL,
       sequence_id BIGINT NOT NULL,  -- Maps back to orderbook snapshots/deltas
       action_taken INTEGER NOT NULL, -- 0-4 action space
       observation_hash VARCHAR(64), -- For debugging/correlation
       portfolio_value DECIMAL(10,2),
       error_message TEXT,
       created_at TIMESTAMPTZ DEFAULT NOW()
   );
   ```

2. **Event persistence integration**
   - Non-blocking event logging (async queue)
   - Link events to orderbook updates via sequence_id
   - Portfolio state tracking
   - Error event capture

### Milestone 5: Execution Integration (Week 5)
**Time Estimate: 3-4 days**

**Components to implement:**
1. **Integration with existing execution stack**
   - LimitOrderActionSpace integration (with **10 contract size**)
   - OrderManager dependency injection
   - KalshiDemoTradingClient configuration

2. **Event broadcasting**
   - ActorEventBroadcaster implementation
   - WebSocketManager message types
   - Frontend event consumption

3. **End-to-end testing**
   - Complete actor cycle testing
   - Demo environment deployment
   - Real market data validation

**Success criteria:**
- Actions execute via KalshiDemoTradingClient
- Events broadcast to frontend
- Complete pipeline functional

## Deployment Configuration

### Environment Variables
```bash
# Actor Control
RL_ACTOR_ENABLED=true
RL_ACTOR_STRATEGY=rl_model  # 'rl_model' | 'hardcoded' | 'disabled'
RL_ACTOR_MODEL_PATH=/path/to/trained_model.zip
RL_ACTOR_THROTTLE_MS=250  # Minimum time between actions

# Market Selection
RL_ACTOR_TARGET_MARKET_MODE=hot  # 'hot' | 'config' | 'first'
RL_ACTOR_CONTRACT_SIZE=10  # MUST match training (currently hardcoded to 10)

# Demo Trading (already exists)
KALSHI_PAPER_TRADING_API_KEY_ID=...
KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT=...
```

### Service Configuration
```python
# Add to config.py
@dataclass
class ActorConfig:
    enabled: bool = os.getenv("RL_ACTOR_ENABLED", "false").lower() == "true"
    strategy: str = os.getenv("RL_ACTOR_STRATEGY", "disabled")
    model_path: Optional[str] = os.getenv("RL_ACTOR_MODEL_PATH")
    throttle_ms: int = int(os.getenv("RL_ACTOR_THROTTLE_MS", "250"))
    target_market_mode: str = os.getenv("RL_ACTOR_TARGET_MARKET_MODE", "hot")
    contract_size: int = int(os.getenv("RL_ACTOR_CONTRACT_SIZE", "10"))  # Must match training
    
    def __post_init__(self):
        # Validate contract size matches training expectation
        if self.contract_size != 10:
            raise ValueError(f"Contract size {self.contract_size} != 10 (training value)")
```

## Critical Performance Architecture

### Model Caching Data Flow (MANDATORY)

```
                    PERFORMANCE-CRITICAL DATA FLOW
                              
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            STARTUP (ONE TIME)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ActorService.__init__() ──→ RLModelSelector.load_model()                     │
│                                        │                                        │
│                                        ▼                                        │
│                               self.model = PPO.load()    ◄── LOAD ONCE         │
│                                        │                                        │
│                                        ▼                                        │
│                               [Model cached in memory]                         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────┼───────────────────────────────────────┐
│                           RUNTIME (HOT PATH)                                   │
├─────────────────────────────────────────┼───────────────────────────────────────┤
│                                          │                                     │
│  Orderbook Update ──→ Actor Event ──→ LiveObservationAdapter                  │
│                                          │                                     │
│                                          ▼                                     │
│                               observation = build_obs()   ◄── <1ms target     │
│                                          │                                     │
│                                          ▼                                     │
│                           action = CACHED_model.predict()  ◄── <1ms target     │
│                                          │                                     │
│                                          ▼                                     │
│                                [Execute trade action]                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

**CRITICAL**: No model loading in hot path. Total actor loop target: <50ms
```

## Data Flow Validation

### Training-Live Consistency Validation
```python
def test_observation_consistency():
    """Critical test: Ensure live obs matches training obs exactly."""
    # Load known training observation
    session_data = await SessionDataLoader().load_session(9)
    training_obs = build_observation_from_session_data(
        session_data.data_points[100], 
        order_manager
    )
    
    # Create equivalent live observation
    adapter = LiveObservationAdapter()
    live_obs = await adapter.build_observation(
        market_ticker, shared_state, order_manager
    )
    
    # Validate exact match
    assert training_obs.shape == live_obs.shape == (52,)
    np.testing.assert_allclose(training_obs, live_obs, rtol=1e-6)
```

## Risk Controls & Safety

### MVP Risk Controls
1. **Paper Trading Only** - No real money at risk
2. **Position Limits** - Max 100 contracts per market  
3. **Rate Limiting** - Max 1 action per 250ms per market
4. **Kill Switch** - Immediate disable via environment variable
5. **Error Circuit Breaker** - Disable after N consecutive errors
6. **Contract Size Lock** - Hardcoded to 10 (matches training)
7. **Database Protection** - Full backup before any schema changes
8. **Performance Circuit Breaker** - Disable if prediction >5ms consistently

### Monitoring & Observability
```python
# Actor metrics to track
- actions_per_minute
- position_count_by_market
- orders_placed_count
- orders_filled_count
- current_portfolio_value
- last_action_timestamp
- errors_count_by_type
- strategy_active
- model_prediction_time_ms  # CRITICAL: Must stay <1ms
- observation_build_time_ms  # CRITICAL: Must stay <1ms
- total_actor_loop_time_ms   # CRITICAL: Must stay <50ms
- model_cache_hit_rate       # Should be 100%
```

## Alternative Rejected Architectures

### ❌ Independent Service Architecture
**Why Rejected:**
- Duplicate WebSocket connections (resource waste)
- Data sync complexity between services
- Training/inference data discrepancies
- Higher operational overhead
- Slower MVP deployment

### ❌ Database-Driven Architecture
**Why Rejected:**
- Introduces latency in trading loop
- Adds database dependency for real-time operations
- Complicates deployment and scaling
- Not aligned with existing non-blocking patterns

## Success Metrics

### MVP Success Criteria
1. **Deployment Speed** - MVP deployed to demo environment within 5 weeks (including safety)
2. **Data Consistency** - Live observations match training format exactly
3. **Strategy Flexibility** - Both RL and hardcoded strategies working
4. **Execution Reliability** - Orders execute successfully via KalshiDemoTradingClient
5. **Real-time Performance** - Actor loop completes within 50ms of trigger
6. **Error Resilience** - Actor continues operating despite individual action failures
7. **Performance Standards**:
   - Model prediction: <1ms (using cached model)
   - Observation building: <1ms
   - Total actor loop: <50ms
8. **Data Safety** - All training data protected with verified backups
9. **Contract Consistency** - 10-contract size matches training exactly

### Edge Case Discovery Goals
- Test with various market conditions (high/low volatility, wide/tight spreads)
- Validate behavior during market halts or connection issues  
- Stress test with rapid orderbook updates
- Validate position tracking across market reopens
- Test model hot-reload during live trading

## Conclusion

This MVP design prioritizes rapid deployment by integrating with proven infrastructure while maintaining modularity for future enhancements. The architecture enables immediate iteration on real market data in the Kalshi demo environment while preserving the ability to swap between RL models and hardcoded strategies.

The critical path is implementing LiveObservationAdapter correctly to ensure training/inference consistency. All other components can be implemented incrementally around this core requirement.

**Recommended immediate next step:** 

1. **MANDATORY FIRST**: Execute database safety protocol with full backup
2. Begin Milestone 1 (Database Safety & Actor Service Framework) 
3. Implement model caching architecture in LiveObservationAdapter
4. Validate contract size consistency (10) between training and execution

**CRITICAL REMINDERS:**
- Model loaded ONCE at startup, cached for all predictions
- market_ticker parameter in select_action is for context, not model input
- ActorEvent persistence maps to orderbook updates via sequence_id
- Contract size must remain 10 to match training
- Database backup required before ANY schema changes