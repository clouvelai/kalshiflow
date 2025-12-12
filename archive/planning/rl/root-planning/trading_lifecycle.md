# Trading Lifecycle Documentation

## Overview

This document describes the complete trading lifecycle for the Kalshi Flow RL system, from training episodes through paper trading validation. The architecture uses a unified OrderManager abstraction with two implementations: SimulatedOrderManager for training and KalshiOrderManager for paper/production trading.

## Architecture: Two OrderManager Implementations

### 1. SimulatedOrderManager (Training)
- Pure Python simulation, no API calls
- Instant order fills based on orderbook crossing
- Deterministic for reproducible training

### 2. KalshiOrderManager (Paper Trading & Future Production)
- Wraps KalshiDemoTradingClient for paper trading
- Will wrap production client in future (same interface)
- Real API calls, async fill processing via WebSocket

## Training Episode Lifecycle

### Episode Initialization
```python
# Step 1: Environment Reset
env = MarketAgnosticKalshiEnv(
    order_manager=SimulatedOrderManager(),
    session_loader=SessionDataLoader()
)

obs = env.reset()
    ├── Select session_id from pool
    ├── Load SessionData (historical orderbooks)
    ├── Initialize SimulatedOrderManager
    │     ├── open_orders = {}
    │     ├── positions = {}
    │     └── cash = 1000.0
    └── Build initial observation
```

### Training Loop (Per Step)
```python
for step in range(max_steps):
    # Step 2: Model Decision
    action = model.predict(obs)  # Returns 0-4
    # 0: HOLD
    # 1: BUY_YES_LIMIT
    # 2: SELL_YES_LIMIT  
    # 3: BUY_NO_LIMIT
    # 4: SELL_NO_LIMIT
    
    # Step 3: Environment Step
    obs, reward, done, info = env.step(action)
```

### Environment Step Breakdown
```
env.step(action):
    ├── 3.1: Decode Action
    │   └── action → order_intent (side, action, price_strategy)
    │
    ├── 3.2: Execute via SimulatedOrderManager
    │   order_manager.execute(order_intent, orderbook):
    │       ├── Cancel conflicting orders
    │       ├── Calculate limit price (bid/ask based)
    │       ├── Check immediate fill
    │       │   ├── If crosses spread: execute fill
    │       │   └── Else: add to open_orders
    │       └── Return execution result
    │
    ├── 3.3: Advance Session Time
    │   session_data.advance():
    │       ├── Move to next timestamp
    │       ├── Load new orderbook state
    │       └── Calculate time_gap
    │
    ├── 3.4: Process Pending Orders
    │   order_manager.check_fills(new_orderbook):
    │       └── For each open order:
    │           ├── Check if would fill at new prices
    │           ├── Execute fills
    │           └── Update positions
    │
    ├── 3.5: Calculate Reward
    │   reward = new_portfolio_value - old_portfolio_value
    │
    └── 3.6: Build New Observation
        ├── Market features (21): spreads, volumes, imbalances
        ├── Order features (5): has_open_buy, buy_distance_from_mid
        ├── Portfolio features (12): positions, cash_ratio
        └── Temporal features (14): time_gaps, momentum
```

## Paper Trading Actor Lifecycle

### Actor Initialization
```python
# Step 1: Initialize Actor with KalshiDemoTradingClient
demo_client = KalshiDemoTradingClient()
await demo_client.connect()

actor = TradingActor(
    model=trained_model,
    order_manager=KalshiOrderManager(demo_client),
    markets=['INXD-23DEC29-B4652']
)
    ├── Load trained PPO model
    ├── Connect to demo-api.kalshi.co
    ├── Subscribe to orderbook WebSocket
    └── Subscribe to user-fills WebSocket
```

### Actor Loop (1 Second Intervals)
```python
async def actor_loop():
    while True:
        for market in markets:
            # Step 2: Get Current State
            orderbook = await orderbook_ws.get_latest(market)
            order_state = order_manager.get_order_state(market)
            positions = order_manager.get_positions()
            
            # Step 3: Build Observation
            obs = build_observation(
                orderbook=orderbook,
                order_features=order_manager.get_order_features(market),
                portfolio_features=extract_portfolio_features(positions)
            )
            
            # Step 4: Model Inference
            action = model.predict(obs)  # 0-4
            
            # Step 5: Execute via KalshiOrderManager
            if action != HOLD:
                await order_manager.execute(action, market)
```

### KalshiOrderManager Execution Logic
```python
async def execute(self, action: int, market: str):
    current_orders = self.get_open_orders(market)
    
    if action == BUY_YES_LIMIT:
        # Cancel any conflicting sell orders
        for order in current_orders:
            if order.side == 'no' or order.action == 'sell':
                await self.client.cancel_order(order.id)
        
        # Place or adjust buy order
        if has_buy_order:
            # Amend existing if price changed
            if should_adjust_price:
                await self.client.amend_order(order.id, new_price)
        else:
            # Place new order at bid
            result = await self.client.create_order(
                ticker=market,
                side='yes',
                action='buy',
                count=10,
                type='limit',
                yes_price=orderbook.best_bid
            )
            self.track_order(result.order_id)
    
    # Similar logic for other actions...
```

### Async Fill Processing
```python
# Running in background via WebSocket
async def on_fill(fill_msg):
    """Process fill notifications from user-fills WebSocket."""
    order_manager.process_fill(fill_msg):
        ├── Update positions[market]
        ├── Remove from open_orders
        ├── Update cash_balance
        └── Log execution
```

## Observation Space Structure

The observation is consistent across training and paper trading:

```python
observation = np.array([
    # Market Features (21)
    best_yes_bid_norm,      # 0.01-0.99
    best_yes_ask_norm,      # 0.01-0.99
    yes_spread_norm,        # 0.001-0.99
    yes_volume_norm,        # 0.0-1.0 (log scaled)
    yes_book_depth_norm,    # 0.0-1.0
    yes_side_imbalance,     # -1.0-1.0
    # ... (similar for NO side)
    arbitrage_opportunity,  # 0.0-1.0
    market_efficiency,      # 0.0-1.0
    
    # Order Features (5) - FROM OrderManager
    has_open_buy,           # 0 or 1
    has_open_sell,          # 0 or 1
    buy_distance_from_mid,  # 0.0-1.0
    sell_distance_from_mid, # 0.0-1.0
    time_since_order,       # 0.0-1.0
    
    # Portfolio Features (12)
    cash_ratio,             # 0.0-1.0
    position_ratio,         # 0.0-1.0
    position_count_norm,    # 0.0-1.0
    long_position_ratio,    # 0.0-1.0
    net_position_bias,      # -1.0-1.0
    unrealized_pnl_ratio,   # -1.0-1.0
    # ...
    
    # Temporal Features (14)
    time_since_last_update, # 0.0-1.0
    activity_score,         # 0.0-1.0
    price_momentum,         # -1.0-1.0
    volatility_regime,      # 0.0-1.0
    # ...
])
```

## Action Space (5 Discrete Actions)

Simple action space with OrderManager handling complexity:

```python
actions = {
    0: "HOLD",           # No action
    1: "BUY_YES_LIMIT",  # OrderManager places buy at bid
    2: "SELL_YES_LIMIT", # OrderManager places sell at ask
    3: "BUY_NO_LIMIT",   # OrderManager places buy at no_bid
    4: "SELL_NO_LIMIT"   # OrderManager places sell at no_ask
}
```

The OrderManager translates these high-level intents into:
- Proper limit orders with pricing
- Order cancellation when switching sides
- Order amendment when adjusting prices
- Position and cash tracking

## Key Differences: Training vs Paper Trading

| Aspect | Training | Paper Trading |
|--------|----------|---------------|
| **Data Source** | SessionData (historical) | Live WebSocket |
| **OrderManager** | SimulatedOrderManager | KalshiOrderManager |
| **Order Execution** | Instant simulation | Real API calls |
| **Fill Processing** | Synchronous check | Async WebSocket |
| **Time Advancement** | Per step (controlled) | Real-time (1 sec) |
| **Episodes** | Fixed length sessions | Continuous |
| **Latency** | None | Real network latency |

## Benefits of This Architecture

1. **Unified Interface**: Same OrderManager API for training and trading
2. **Simple Actions**: Model only decides WHAT, not HOW
3. **Clean Separation**: Order complexity isolated in OrderManager
4. **Easy Testing**: Swap OrderManager to test different environments
5. **Production Ready**: Paper trading uses real API (demo account)
6. **Future Proof**: Production trading just swaps client credentials

## Next Steps for Implementation

1. **M4b**: Implement OrderManager abstraction and two implementations
2. **M5**: UnifiedPositionTracker integrated with OrderManager
3. **M6**: Finalize 5-action space relying on OrderManager
4. **M7**: Complete environment with OrderManager dependency injection

## V1.0 Scope

- Focus on paper trading with KalshiDemoTradingClient
- No production trading yet (future client swap)
- Fixed 10 contract sizing
- Single market at a time
- 5 simple actions with OrderManager handling complexity