# Probabilistic Fill Model Documentation

## Overview

The probabilistic fill model simulates realistic order execution dynamics in the Kalshi Flow RL Trading System. Unlike simplistic models where orders fill with 100% certainty when prices cross, this implementation considers multiple market factors to determine fill probability.

## Architecture

### Core Components

#### 1. MarketActivityTracker
Tracks recent market activity in a sliding time window to gauge market liquidity and activity levels.

```python
class MarketActivityTracker:
    def __init__(self, window_seconds: int = 300):
        """Track trades in a 5-minute sliding window"""
        
    def add_trade(self, size: int) -> None:
        """Record a trade for activity tracking"""
        
    def get_activity_level(self) -> float:
        """Return normalized activity level 0-1"""
```

Activity Level Mapping:
- **0.0-0.5**: Dead market (<5 trades/min)
- **0.5-0.7**: Normal market (5-20 trades/min)
- **0.7-1.0**: Active market (>20 trades/min)

#### 2. Fill Probability Calculation

The `calculate_fill_probability()` method computes fill likelihood based on:

##### Base Probability (Price Aggression)
- **Crossing spread (aggressive)**: 95-99%
- **At best bid/ask (passive)**: 40%
- **Inside spread**: 40-90% (position-dependent)
- **Away from market**: <30% (decaying with distance)

##### Modifying Factors
1. **Time Priority** (±5%)
   - New orders: -10% (back of queue)
   - 10 seconds: 0% (middle of queue)
   - 30+ seconds: +5% (front of queue)

2. **Size Impact** (±10%)
   - <10 contracts: +5% (small orders fill easier)
   - 10-50 contracts: 0% (normal size)
   - 50-100 contracts: -3% (slightly harder)
   - >100 contracts: -10% (large orders fill slower)

3. **Market Activity** (±10%)
   - Maps activity level [0,1] to modifier [-0.1, +0.1]
   - Dead markets: -10% fill probability
   - Active markets: +10% fill probability

4. **Edge Case Adjustments**
   - Wide spreads (>5¢): Up to -10% penalty
   - Empty orderbooks: -30% penalty

Final probability is clamped to [0.01, 0.99].

#### 3. Order Classification

The `_is_aggressive_order()` method determines order type:
- **Aggressive**: Crosses the spread (marketable)
- **Passive**: Joins the bid/ask queue

This distinction is crucial for fill logic:
- Passive orders fill at their limit price when probability hits
- Aggressive orders use depth consumption for VWAP calculation

## Integration with Depth Consumption

The probabilistic model works alongside the existing depth consumption system:

```python
async def check_fills(self, orderbook: OrderbookState) -> List[OrderInfo]:
    for order in self.open_orders.values():
        fill_probability = self.calculate_fill_probability(order, orderbook)
        
        if random.random() < fill_probability:
            if self._is_aggressive_order(order, orderbook):
                # Aggressive: Use depth consumption for VWAP
                fill_result = self.calculate_fill_with_depth(order, orderbook)
                if fill_result['can_fill']:
                    self._process_fill_with_vwap(...)
            else:
                # Passive: Fill at limit price
                self._process_fill(order, order.limit_price)
```

## Usage in Training

### SimulatedOrderManager Configuration

```python
# Initialize with default settings
manager = SimulatedOrderManager(
    initial_cash=100000,  # Starting capital in cents
    small_order_threshold=20  # Orders <20 skip depth consumption
)

# The manager automatically:
# - Tracks market activity
# - Calculates fill probabilities
# - Handles partial fills
# - Maintains position tracking
```

### Environment Integration

The probabilistic fill model is automatically used when `SimulatedOrderManager` is selected:

```python
env = MarketAgnosticKalshiEnv(
    market_ticker="TICKER",
    order_manager=SimulatedOrderManager(initial_cash=100000),
    # ... other config
)
```

## Training Impact

### Realistic Market Dynamics
1. **No Guaranteed Fills**: Agents can't assume orders always execute
2. **Queue Position Matters**: Earlier orders have higher fill probability
3. **Size-Aware Execution**: Large orders face execution risk
4. **Activity Sensitivity**: Dead markets reduce fill rates

### Strategic Learning
Agents learn to:
- Balance aggressive vs passive order placement
- Consider order size impact on execution
- Time orders based on market activity
- Manage partial fill risk

## Configuration Parameters

### MarketActivityTracker
- `window_seconds`: Time window for activity tracking (default: 300)

### SimulatedOrderManager
- `initial_cash`: Starting capital in cents (default: 100000)
- `small_order_threshold`: Size below which orders skip depth consumption (default: 20)

### Probability Bounds
- Minimum fill probability: 0.01 (1%)
- Maximum fill probability: 0.99 (99%)

## Testing

The model includes comprehensive test coverage:

```bash
# Run probabilistic fill tests
uv run pytest tests/test_rl/trading/test_probabilistic_fills.py -v

# Test coverage includes:
# - Market activity tracking
# - Probability calculations
# - Time priority effects
# - Size impact modifiers
# - Statistical fill rates
# - Integration with depth consumption
```

All tests use `random.seed(42)` for deterministic results.

## Performance Characteristics

- **Overhead**: Minimal (<1% training speed impact)
- **Memory**: O(n) where n = trades in time window
- **Computation**: O(1) for probability calculation

## Validation Results

Successfully validated with PPO training on session 32:
- 10,000 timesteps at 2,410 steps/second
- 116 episodes completed without errors
- Realistic fill rates observed:
  - Aggressive orders: >95%
  - Passive orders: 30-40%
  - Inside spread: 50-70%

## Future Enhancements

Potential improvements:
1. **Queue Position Tracking**: Explicit FIFO modeling
2. **Market Impact**: Large orders temporarily move prices
3. **Hidden Liquidity**: Iceberg order simulation
4. **Time-of-Day Effects**: Activity patterns
5. **Maker/Taker Fees**: Fee-aware fill probability

## References

- Implementation: `src/kalshiflow_rl/trading/order_manager.py`
- Tests: `tests/test_rl/trading/test_probabilistic_fills.py`
- Validation: `src/kalshiflow_rl/training/validate_probabilistic_fills.py`