# Useful Patterns from Old Trading Architecture

This document preserves valuable patterns from the deprecated trading codebase that will be useful for future TRADER 2.0 implementation.

## 1. Orderbook Snapshot → Observation Transformation

### Live Observation Adapter Pattern
From `trading/live_observation_adapter.py`:

```python
class LiveObservationAdapter:
    """
    Converts SharedOrderbookState to training-consistent 52-feature observations.
    
    Key patterns:
    - Maintains sliding window history for temporal features
    - Per-market sliding windows with deque(maxlen=window_size)
    - Cache observations for performance (100ms TTL)
    - Dependency injection for orderbook state registry
    """
    
    async def build_observation(
        self,
        market_ticker: str,
        position_data: Optional[Dict[str, Any]] = None,
        portfolio_value: float = 10000.0,
        cash_balance: float = 10000.0,
        order_features: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        # 1. Check cache first for performance
        cache_key = f"{market_ticker}_{portfolio_value}_{cash_balance}"
        
        # 2. Get live orderbook state via injected registry
        if self._orderbook_state_registry:
            shared_state = await self._orderbook_state_registry.get_shared_orderbook_state(market_ticker)
        
        # 3. Convert to SessionDataPoint format for consistency with training
        current_snapshot = await self._get_live_snapshot(shared_state, market_ticker)
        
        # 4. Build 52-feature observation using training feature extractors
        # Uses same pipeline: extract_market_agnostic_features + extract_temporal_features
```

### Feature Engineering Patterns
- **Temporal Features**: Sliding window of last N snapshots for volume/price trends
- **Market Agnostic Features**: Normalized features that work across different markets
- **Portfolio Features**: Position size, P&L, cash balance as part of observation
- **Order Features**: Active order state (5 features) included in observation

## 2. Observation → Action Selection Flow

### Actor Service Pattern
From `trading/actor_service.py`:

```python
class ActorService:
    """
    4-step pipeline for action selection:
    1. build_observation → 2. select_action → 3. execute_action → 4. update_positions
    """
    
    async def _process_event(self, event: ActorEvent):
        # 1. Build observation
        observation = await self._build_observation(event.market_ticker)
        
        # 2. Select action using model
        action = await self._select_action(observation, event.market_ticker)
        
        # 3. Execute action (send to order manager)
        success = await self._execute_action(action, event.market_ticker)
        
        # 4. Update positions after delay
        await self._update_positions_after_delay()
```

### Model Inference Pattern
```python
async def _select_action(self, observation: np.ndarray, market_ticker: str) -> int:
    """Model inference with safety checks."""
    # Reshape observation for model
    obs_tensor = observation.reshape(1, -1)
    
    # Get model prediction
    action, _states = self._cached_model.predict(
        obs_tensor,
        deterministic=True  # No exploration in production
    )
    
    # Validate action is in valid range
    if action < 0 or action >= self.action_space_size:
        return 0  # Default to HOLD
    
    return int(action)
```

### Action Space Mapping
```python
# 5-action space (classic)
ACTION_MAPPING = {
    0: "HOLD",
    1: "BUY_YES", 
    2: "SELL_YES",
    3: "BUY_NO",
    4: "SELL_NO"
}

# 21-action space (with position sizing)
# Actions 0-4: Same as above with 5 contracts
# Actions 5-9: Same pattern with 10 contracts
# Actions 10-14: Same pattern with 20 contracts
# Actions 15-19: Same pattern with 50 contracts
# Action 20: HOLD
```

## 3. Action → Order Execution

### Order Manager Pattern
From `trading/order_manager.py` and `trading/kalshi_multi_market_order_manager.py`:

```python
class OrderManager:
    """
    Handles order lifecycle and position tracking.
    
    Key patterns:
    - Option B cash tracking: Deduct on place, restore on cancel
    - Fill processing via async queue
    - Position tracking with Kalshi convention (+YES/-NO)
    """
    
    async def execute_action(
        self,
        action: int,
        ticker: str,
        orderbook_state: OrderbookState
    ) -> bool:
        # 1. Map action to order parameters
        if action == 0:  # HOLD
            return True
        elif action == 1:  # BUY_YES
            side = OrderSide.BUY
            contract_side = ContractSide.YES
        # ... etc
        
        # 2. Calculate order size based on risk limits
        quantity = self._calculate_order_quantity(ticker, side, contract_side)
        
        # 3. Calculate limit price based on strategy
        limit_price = self._calculate_limit_price(
            side, contract_side, orderbook, strategy="aggressive"
        )
        
        # 4. Check cash availability (Option B)
        required_cash = (limit_price / 100.0) * quantity
        if side == OrderSide.BUY and self.available_cash < required_cash:
            return False  # Insufficient funds
        
        # 5. Place order via API
        order_id = await self._place_order_api(ticker, side, contract_side, quantity, limit_price)
        
        # 6. Update cash tracking (deduct promised cash)
        if side == OrderSide.BUY:
            self.promised_cash += required_cash
            self.available_cash -= required_cash
```

### Limit Price Calculation Pattern
```python
def _calculate_limit_price(
    self,
    side: OrderSide,
    contract_side: ContractSide,
    orderbook: OrderbookState,
    strategy: str = "aggressive"
) -> int:
    """Smart pricing based on orderbook state."""
    
    # Get best bid/ask from orderbook
    if contract_side == ContractSide.YES:
        best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
        best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
    else:
        # NO contracts use inverted YES prices
        yes_best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
        yes_best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
        best_bid = 99 - yes_best_ask  # NO bid = 99 - YES ask
        best_ask = 99 - yes_best_bid  # NO ask = 99 - YES bid
    
    if strategy == "aggressive":
        # Cross the spread for immediate execution
        price = best_ask if side == OrderSide.BUY else best_bid
    elif strategy == "passive":
        # Join the inside market
        price = best_bid if side == OrderSide.BUY else best_ask
    else:  # "mid"
        price = int((best_bid + best_ask) / 2)
    
    return max(1, min(99, price))
```

### Position Tracking Pattern
```python
@dataclass
class Position:
    """Kalshi convention: +contracts for YES, -contracts for NO."""
    ticker: str
    contracts: int       # +YES/-NO
    cost_basis: float    # Total cost in cents
    realized_pnl: float  # Cumulative realized P&L
    
    def get_unrealized_pnl(self, current_yes_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.contracts > 0:
            # Long YES: profit when YES price rises
            current_value = self.contracts * current_yes_price
        else:
            # Long NO: profit when YES price falls
            current_value = abs(self.contracts) * (1.0 - current_yes_price)
        
        return current_value - self.cost_basis
```

### Fill Processing Pattern
```python
async def _process_fill_queue(self):
    """Process fills from WebSocket in order."""
    while not self._shutdown:
        try:
            fill_event = await self._fills_queue.get()
            
            # 1. Find corresponding order
            order = self._find_order_by_kalshi_id(fill_event.order_id)
            
            # 2. Update order status and quantities
            if fill_event.is_partial:
                order.quantity -= fill_event.count
            else:
                order.status = OrderStatus.FILLED
            
            # 3. Restore promised cash for unfilled portion
            if order.side == OrderSide.BUY:
                unfilled_value = (order.quantity * order.limit_price) / 100.0
                self.promised_cash -= unfilled_value
                
            # 4. Update position
            self._update_position_from_fill(fill_event)
            
        except Exception as e:
            logger.error(f"Error processing fill: {e}")
```

## 4. Risk Management Patterns

### Order Sizing
```python
def _calculate_order_quantity(self, ticker: str, side: OrderSide) -> int:
    """Risk-based order sizing."""
    # Base size from configuration
    base_size = self.order_quantity
    
    # Scale based on available capital
    if side == OrderSide.BUY:
        max_affordable = int(self.available_cash / 0.5)  # Assume 50 cent price
        base_size = min(base_size, max_affordable)
    
    # Position limits
    current_position = self.positions.get(ticker, Position()).contracts
    if abs(current_position) + base_size > self.max_position_size:
        base_size = max(0, self.max_position_size - abs(current_position))
    
    return base_size
```

### Cash Reserve Management
```python
# Always maintain minimum cash reserve
CASH_RESERVE_RATIO = 0.2  # Keep 20% cash minimum

if self.available_cash < self.total_capital * CASH_RESERVE_RATIO:
    return 0  # Force HOLD action
```

## 5. Event-Driven Architecture Patterns

### Event Bus Pattern
```python
class EventBus:
    """Pub/sub for decoupled components."""
    
    async def publish(self, event_type: EventType, data: Any):
        """Publish event to all subscribers."""
        for callback in self._subscribers[event_type]:
            asyncio.create_task(callback(data))
    
    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to event type."""
        self._subscribers[event_type].append(callback)
```

### WebSocket Integration
```python
# Non-blocking trigger pattern
async def on_orderbook_update(market_ticker: str):
    """Trigger actor without blocking orderbook client."""
    event = ActorEvent(
        market_ticker=market_ticker,
        update_type="snapshot",
        timestamp_ms=int(time.time() * 1000)
    )
    
    # Non-blocking queue put
    try:
        self._event_queue.put_nowait(event)
    except asyncio.QueueFull:
        # Drop event if queue full (backpressure)
        pass
```

## 6. State Synchronization Patterns

### Position Reconciliation
```python
async def sync_positions(self):
    """Reconcile local state with Kalshi API."""
    # 1. Fetch positions from API
    api_positions = await self.client.get_positions()
    
    # 2. Update local state
    for pos_data in api_positions:
        ticker = pos_data["ticker"]
        contracts = pos_data["position"]  # Kalshi convention
        cost_basis = pos_data["position_cost"] / 10000  # Convert centi-cents
        
        self.positions[ticker] = Position(
            ticker=ticker,
            contracts=contracts,
            cost_basis=cost_basis
        )
    
    # 3. Clear stale positions
    for ticker in list(self.positions.keys()):
        if ticker not in api_positions:
            del self.positions[ticker]
```

### Order Reconciliation
```python
async def sync_orders(self):
    """Sync active orders with API."""
    # 1. Get open orders from API
    api_orders = await self.client.get_orders(status="open")
    
    # 2. Cancel local orders not in API
    local_order_ids = {o.kalshi_order_id for o in self._active_orders.values()}
    api_order_ids = {o["order_id"] for o in api_orders}
    
    for order_id in local_order_ids - api_order_ids:
        # Order was filled or cancelled externally
        self._handle_external_order_change(order_id)
    
    # 3. Add API orders not tracked locally
    for order_id in api_order_ids - local_order_ids:
        # Order placed externally
        self._add_external_order(order_id)
```

## Key Takeaways

1. **Observation Building**: Use sliding windows for temporal features, cache for performance
2. **Action Selection**: Model inference with safety checks, deterministic in production
3. **Order Execution**: Smart pricing based on orderbook, Option B cash tracking
4. **Position Management**: Kalshi convention (+YES/-NO), track cost basis and P&L
5. **Event-Driven**: Non-blocking queues, pub/sub for decoupling
6. **State Sync**: Regular reconciliation with API as source of truth

These patterns provide a solid foundation for implementing TRADER 2.0's clean service architecture while maintaining functional parity with the existing system.