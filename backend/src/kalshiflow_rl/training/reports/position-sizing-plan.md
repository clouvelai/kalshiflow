# Position Sizing Implementation Plan - Surgical Edition

## Executive Summary
Implement variable position sizing (5, 10, 20, 50, 100 contracts) by expanding action space to **21 discrete actions** (not 25). Complete rewrite of action execution with no backward compatibility. This is a breaking change that removes the fixed 10-contract system entirely.

## 1. Core Architecture Design

### Action Space Definition
- **21 total actions** (0-20) - no wasted neural network capacity
- Action 0: HOLD (no trade)
- Actions 1-20: Trading actions with position sizes encoded

### Position Sizes
```python
POSITION_SIZES = [5, 10, 20, 50, 100]  # Small to Giant
```

### Action Encoding/Decoding
```python
def decode_action(action: int) -> Tuple[int, int]:
    """Decode single action into base action and size index.
    
    The model outputs a single action (0-20) which encodes both
    the trading intent AND the position size.
    
    Returns:
        base_action: 0=HOLD, 1=BUY_YES, 2=SELL_YES, 3=BUY_NO, 4=SELL_NO
        size_index: 0-4 mapping to [5, 10, 20, 50, 100] contracts
    """
    if action == 0:
        return 0, 0  # HOLD, no size
    
    adjusted = action - 1  # Now 0-19 for trading actions
    base_action = (adjusted // 5) + 1  # Which trading intent (1-4)
    size_index = adjusted % 5  # Which size (0-4)
    return base_action, size_index

def encode_action(base_action: int, size_index: int) -> int:
    """Encode base action and size into single action."""
    if base_action == 0:
        return 0  # HOLD
    return (base_action - 1) * 5 + size_index + 1
```

### Action Mapping
```
Action 0: HOLD
Actions 1-5: BUY_YES (5, 10, 20, 50, 100 contracts)
Actions 6-10: SELL_YES (5, 10, 20, 50, 100 contracts)
Actions 11-15: BUY_NO (5, 10, 20, 50, 100 contracts)
Actions 16-20: SELL_NO (5, 10, 20, 50, 100 contracts)
```

## 2. Critical Implementation Requirements

### 2.1 DELETE FIRST Strategy
**Complete removal required before implementation:**
- `execute_action_sync()` method entirely
- All async/sync detection logic
- Fixed `contract_size` parameter handling
- Any backward compatibility code

### 2.2 Thread Safety & Concurrency
```python
class PositionAwareActionSpace:
    def __init__(self):
        self.position_lock = threading.Lock()
        self.position_sizes = [5, 10, 20, 50, 100]
        
    def execute_action(self, action: int, ticker: str, orderbook) -> Any:
        """Single execution path - no async/sync bifurcation."""
        base_action, size_index = self.decode_action(action)
        size = self.position_sizes[size_index] if base_action != 0 else 0
        
        with self.position_lock:
            return self.order_manager.execute_with_size(
                base_action, size, ticker, orderbook
            )
```

### 2.3 Position Size Validation
```python
@dataclass
class PositionConfig:
    """Central configuration for position sizing."""
    sizes: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    max_position_per_market: int = 500
    max_position_value: int = 50000  # $500 in cents
    max_portfolio_concentration: float = 0.20  # 20% max
    min_cash_buffer: int = 5000  # $50 minimum reserve

class PositionSizeValidator:
    def __init__(self, config: PositionConfig):
        self.config = config
        
    def validate_action(
        self, 
        action: int, 
        cash: int, 
        current_position: int,
        orderbook: Dict
    ) -> bool:
        """Validate if action is executable given constraints."""
        if action == 0:
            return True  # HOLD always valid
            
        base_action, size_index = decode_action(action)
        size = self.config.sizes[size_index]
        
        # Get relevant price from orderbook
        price = self._get_execution_price(base_action, orderbook)
        position_value = size * price
        
        # Check all constraints
        checks = [
            abs(current_position + size) <= self.config.max_position_per_market,
            position_value <= self.config.max_position_value,
            position_value <= cash * self.config.max_portfolio_concentration,
            cash - position_value >= self.config.min_cash_buffer
        ]
        
        return all(checks)
```

## 3. Risk Management & Market Impact

### 3.1 Liquidity-Aware Sizing
```python
class AdaptivePositionSizing:
    """Adjust position sizes based on market liquidity."""
    
    LIQUIDITY_MULTIPLIERS = {
        "high": 1.0,    # >$10k volume in 10min
        "medium": 0.5,  # $1k-$10k volume
        "low": 0.2      # <$1k volume
    }
    MAX_POSITION_PCT_OF_VOLUME = 0.05  # Max 5% of recent volume
    
    def get_max_size(
        self, 
        base_size: int, 
        market_volume_10min: float,
        orderbook_depth: float
    ) -> int:
        """Calculate maximum allowed size given market conditions."""
        # Classify liquidity
        if market_volume_10min > 1000000:  # $10k in cents
            liquidity = "high"
        elif market_volume_10min > 100000:  # $1k in cents
            liquidity = "medium"
        else:
            liquidity = "low"
        
        # Apply multiplier
        adjusted_size = int(base_size * self.LIQUIDITY_MULTIPLIERS[liquidity])
        
        # Cap by volume percentage
        volume_limit = int(market_volume_10min * self.MAX_POSITION_PCT_OF_VOLUME / 50)  # Assume avg price 50¢
        
        return min(adjusted_size, volume_limit, base_size)
```

### 3.2 Market Impact Estimation
```python
def estimate_market_impact(
    size: int, 
    orderbook: Dict, 
    side: str
) -> Tuple[float, float]:
    """Estimate price impact and spread cost of order.
    
    Returns:
        avg_fill_price: Expected average fill price
        impact_cost: Cost in cents due to market impact
    """
    levels = orderbook['yes_asks'] if side == 'buy' else orderbook['yes_bids']
    
    cumulative_volume = 0
    weighted_price = 0
    
    for price, volume in levels:
        remaining = size - cumulative_volume
        level_fill = min(remaining, volume)
        weighted_price += price * level_fill
        cumulative_volume += level_fill
        
        if cumulative_volume >= size:
            break
    
    if cumulative_volume < size:
        # Not enough liquidity
        return float('inf'), float('inf')
    
    avg_fill_price = weighted_price / size
    mid_price = (orderbook['yes_bid'][0][0] + orderbook['yes_ask'][0][0]) / 2
    impact_cost = abs(avg_fill_price - mid_price) * size
    
    return avg_fill_price, impact_cost
```

## 4. Observation Space Enhancements

### 4.1 Position Sizing Features
```python
def get_position_sizing_features(self) -> np.ndarray:
    """Additional features to help model learn sizing."""
    return np.array([
        # Current position state
        self.current_position / 500,  # Normalized by max
        self.current_position_value / self.portfolio_value,  # Position concentration
        
        # Capacity indicators
        self.available_cash / 100000,  # Normalized cash ($1000)
        self._calculate_max_affordable_size() / 100,  # Max size we can afford
        
        # Market liquidity
        self.market_volume_10min / 1000000,  # Normalized by $10k
        self.orderbook_depth_5pct / 1000,  # Contracts within 5% of mid
        
        # Spread cost indicators
        self.current_spread,  # Bid-ask spread
        self.avg_spread_10min,  # Recent average spread
    ])
```

### 4.2 Integration with Existing Observation
```python
# In market_agnostic_env.py
def _get_observation(self) -> np.ndarray:
    """Extended observation with position sizing features."""
    base_obs = self._get_base_observation()  # Existing features
    sizing_features = self.get_position_sizing_features()
    
    return np.concatenate([base_obs, sizing_features])
```

## 5. Implementation Steps - Day by Day

### Day 1: Core Architecture Surgery

#### File: `limit_order_action_space.py`
```python
# COMPLETE REWRITE REQUIRED
class LimitOrderActionSpace:
    def __init__(self, order_manager, position_config: Optional[PositionConfig] = None):
        self.order_manager = order_manager
        self.config = position_config or PositionConfig()
        self.validator = PositionSizeValidator(self.config)
        self.position_sizes = self.config.sizes
        self.action_space = spaces.Discrete(21)  # NOT 25!
        
    def decode_action(self, action: int) -> Tuple[int, int]:
        # Implementation from above
        
    def execute_action(self, action: int, ticker: str, orderbook: Dict) -> Any:
        """Single execution path with size awareness."""
        if not self.validator.validate_action(action, self.cash, self.position, orderbook):
            return None  # Invalid action
            
        base_action, size_index = self.decode_action(action)
        
        if base_action == 0:
            return None  # HOLD
            
        size = self.position_sizes[size_index]
        
        # Map to order manager methods
        action_map = {
            1: lambda: self.order_manager.place_buy_yes_limit(ticker, size, orderbook),
            2: lambda: self.order_manager.place_sell_yes_limit(ticker, size, orderbook),
            3: lambda: self.order_manager.place_buy_no_limit(ticker, size, orderbook),
            4: lambda: self.order_manager.place_sell_no_limit(ticker, size, orderbook),
        }
        
        return action_map[base_action]()
```

#### File: `market_agnostic_env.py`
```python
# Update action space
self.action_space = spaces.Discrete(21)  # Changed from 5

# Update observation space to include sizing features
sizing_feature_dim = 8  # New features
self.observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(existing_dim + sizing_feature_dim,),
    dtype=np.float32
)
```

### Day 2: Safety & Edge Cases

#### Critical Edge Cases to Handle
1. **Partial Fills**
   ```python
   def handle_partial_fill(self, order_id: str, filled: int, remaining: int):
       """Track partial fills correctly."""
       self.position += filled
       self.pending_orders[order_id] = remaining
   ```

2. **Cash Depletion**
   ```python
   def get_action_mask(self) -> np.ndarray:
       """Mask actions that would deplete cash buffer."""
       mask = np.ones(21, dtype=bool)
       for action in range(1, 21):
           if not self.validator.validate_action(action, self.cash, self.position, self.orderbook):
               mask[action] = False
       return mask
   ```

3. **Concurrent Orders**
   ```python
   def can_place_order(self, size: int) -> bool:
       """Check if we can place order given pending orders."""
       pending_value = sum(o.size * o.price for o in self.pending_orders.values())
       return self.cash - pending_value >= self.config.min_cash_buffer
   ```

### Day 3: Testing & Validation

#### Comprehensive Test Suite
```python
# test_position_sizing.py
class TestPositionSizing:
    def test_all_21_actions_execute(self):
        """Verify each action executes correctly."""
        
    def test_action_encoding_decoding(self):
        """Test encode/decode symmetry."""
        
    def test_position_limits_enforced(self):
        """Verify all risk limits work."""
        
    def test_liquidity_adjustment(self):
        """Test market liquidity constraints."""
        
    def test_concurrent_orders(self):
        """Test thread safety with multiple sizes."""
        
    def test_partial_fill_handling(self):
        """Verify partial fills tracked correctly."""
        
    def test_cash_depletion_prevention(self):
        """Ensure min cash buffer maintained."""
```

## 6. Files to Modify

### Core Changes (DELETE & REWRITE)
1. `limit_order_action_space.py` - Complete rewrite, remove all sync logic
2. `market_agnostic_env.py` - Update to 21 actions, add sizing features

### Updates Required
3. `sb3_wrapper.py` - Ensure 21-action passthrough
4. `train_sb3.py` - Handle 21-action space
5. `actor_service.py` - Decode 21 actions for live trading
6. `order_manager.py` - Already supports variable quantity (verify)

### New Files
7. `position_sizing.py` - Centralized sizing logic and validation
8. `test_position_sizing.py` - Comprehensive test suite

### Test Updates
9. `test_limit_order_action_space.py` - Update for 21 actions
10. `test_market_agnostic_env.py` - Test with new action space
11. `test_limit_order_integration.py` - Integration tests with sizes

## 7. Performance Metrics

### Technical Metrics
```python
class PositionSizingMetrics:
    # Efficiency metrics
    action_space_utilization: float  # % of 21 actions used
    memory_consumption: int  # Bytes for policy network
    execution_latency: Dict[int, float]  # Latency by position size
    
    # Trading metrics  
    position_size_distribution: Dict[int, int]  # Usage count by size
    avg_winning_position_size: float
    avg_losing_position_size: float
    size_to_liquidity_ratio: float  # How well we match market depth
    
    # Risk metrics
    max_position_reached: int
    portfolio_concentration_events: int  # Times we hit concentration limit
    spread_cost_by_size: Dict[int, float]  # Spread paid by position size
```

## 8. Critical Success Factors

### Must Pass Before Deployment
- ✅ All 21 actions execute without errors
- ✅ Position limits enforced (volume %, portfolio %, absolute)
- ✅ Thread-safe concurrent execution
- ✅ Spread costs included in decisions
- ✅ No memory leaks in policy network
- ✅ Partial fills handled correctly
- ✅ Cash buffer never violated

### Performance Targets
- 📊 >10% profitability improvement vs fixed sizing
- 📊 <100ms execution latency for any position size
- 📊 >80% action space utilization (using most actions)
- 📊 Appropriate size/liquidity matching

## 9. Implementation Notes for RL Agent

### Critical Reminders
1. **DELETE FIRST** - Remove ALL sync/async bifurcation before starting
2. **Type Everything** - Use type hints for every function: `-> Tuple[int, int]`
3. **Test Continuously** - Write test for each component before implementing
4. **No Legacy Support** - This is a clean break, no migration needed
5. **Validate Boundaries** - Check limits at every decision point
6. **Document Market Impact** - Add docstrings explaining sizing logic

### Common Pitfalls to Avoid
- ❌ Don't keep any execute_action_sync code
- ❌ Don't use 25 actions (wastes neural network capacity)
- ❌ Don't forget thread safety for position tracking
- ❌ Don't assume unlimited market liquidity
- ❌ Don't skip partial fill handling
- ❌ Don't violate cash buffer constraints

### Architecture Principles
- Single execution path (no async/sync split)
- Explicit validation before execution
- Position sizes are part of the action, not a separate decision
- Market impact considered before placing orders
- Fail safely when constraints violated

## 10. Rollback Plan

If critical issues arise:
1. Git revert to previous commit
2. Redeploy old model (incompatible with new code)
3. Document failure modes for next attempt
4. No data migration needed (clean break)

## Appendix: Complete Action Mapping Reference

```python
ACTION_REFERENCE = {
    0: ("HOLD", 0),
    1: ("BUY_YES", 5),
    2: ("BUY_YES", 10),
    3: ("BUY_YES", 20),
    4: ("BUY_YES", 50),
    5: ("BUY_YES", 100),
    6: ("SELL_YES", 5),
    7: ("SELL_YES", 10),
    8: ("SELL_YES", 20),
    9: ("SELL_YES", 50),
    10: ("SELL_YES", 100),
    11: ("BUY_NO", 5),
    12: ("BUY_NO", 10),
    13: ("BUY_NO", 20),
    14: ("BUY_NO", 50),
    15: ("BUY_NO", 100),
    16: ("SELL_NO", 5),
    17: ("SELL_NO", 10),
    18: ("SELL_NO", 20),
    19: ("SELL_NO", 50),
    20: ("SELL_NO", 100),
}
```

This surgical implementation eliminates all complexity, provides clear boundaries, and ensures the RL agent can execute flawlessly.