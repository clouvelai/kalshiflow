# Position Sizing Implementation Plan - Surgical Edition V2

## Executive Summary
Implement variable position sizing (5, 10, 20, 50, 100 contracts) by expanding action space to **21 discrete actions** (not 25). **PRESERVE the async/sync architecture** which is intentional for separating training (SimulatedOrderManager) from live trading (KalshiMultiMarketOrderManager).

## Critical Architecture Discovery
The async/sync bifurcation is **intentional and necessary**:
- **Training**: Uses SimulatedOrderManager → sync execution path
- **Live Trading**: Uses KalshiMultiMarketOrderManager → async execution path
- **Detection Logic**: Checks order manager type to route appropriately
- **Both paths must be preserved and extended with position sizing**

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

### 2.1 PRESERVE Async/Sync Architecture
**DO NOT DELETE execute_action_sync or the routing logic!** Instead:
- Keep the SimulatedOrderManager detection logic
- Keep both execute_action_sync and execute_action methods
- Extend both paths with position sizing
- Remove fixed contract_size parameter

### 2.2 Update Both Execution Paths
```python
class LimitOrderActionSpace:
    def __init__(self, order_manager, position_config: Optional[PositionConfig] = None):
        self.order_manager = order_manager
        self.config = position_config or PositionConfig()
        self.position_sizes = self.config.sizes
        # Remove: self.contract_size = contract_size
        
    def execute_action_sync(self, action: int, ticker: str, orderbook: OrderbookState):
        """Synchronous execution for training environments."""
        # Keep detection logic for SimulatedOrderManager
        if hasattr(self.order_manager, '__class__') and 'SimulatedOrderManager' in str(self.order_manager.__class__):
            return self._execute_action_sync_simulated(action, ticker, orderbook)
        else:
            # Handle real order manager (keep existing logic)
            ...
    
    async def execute_action(self, action: int, ticker: str, orderbook: OrderbookState):
        """Async execution for live trading."""
        base_action, size_index = self.decode_action(action)
        size = self.position_sizes[size_index] if base_action != 0 else 0
        # Pass size to order placement methods
        ...
    
    def _execute_action_sync_simulated(self, action: int, ticker: str, orderbook: OrderbookState):
        """Sync execution for SimulatedOrderManager."""
        base_action, size_index = self.decode_action(action)
        size = self.position_sizes[size_index] if base_action != 0 else 0
        # Use size instead of self.contract_size
        ...
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
    def validate_action(self, action: int, cash: int, current_position: int, orderbook: Dict) -> bool:
        """Validate if action is executable given constraints."""
        if action == 0:
            return True  # HOLD always valid
            
        base_action, size_index = decode_action(action)
        size = self.config.sizes[size_index]
        
        # Validate size constraints
        price = self._get_execution_price(base_action, orderbook)
        position_value = size * price
        
        checks = [
            abs(current_position + size) <= self.config.max_position_per_market,
            position_value <= self.config.max_position_value,
            position_value <= cash * self.config.max_portfolio_concentration,
            cash - position_value >= self.config.min_cash_buffer
        ]
        
        return all(checks)
```

## 3. Implementation Steps - Day by Day

### Day 1: Core Architecture Extension

#### File: `limit_order_action_space.py`
**EXTEND, don't delete!**
1. Add PositionConfig and PositionSizeValidator classes
2. Add decode_action method
3. Update __init__ to remove contract_size, add position_config
4. Update get_gym_space() to return Discrete(21)
5. **Extend execute_action_sync**: Add position sizing while keeping SimulatedOrderManager detection
6. **Extend execute_action**: Add position sizing to async path
7. **Update _execute_action_sync_simulated**: Use decoded size instead of fixed contract_size
8. **Update _execute_buy_action_sync and _execute_sell_action_sync**: Use variable size

#### File: `market_agnostic_env.py`
1. Update action space to Discrete(21)
2. Add position sizing features to observation space (optional for Day 1)
3. No changes needed to step() - it already calls execute_action_sync

### Day 2: Order Manager Updates

#### Verify OrderManager Support
Both SimulatedOrderManager and KalshiMultiMarketOrderManager already accept variable quantity:
- SimulatedOrderManager.place_order(... quantity: int ...)
- KalshiMultiMarketOrderManager.execute_order(...) uses quantity

#### Update Trading Integration
1. **actor_service.py**: Already passes action directly, no changes needed
2. **kalshi_multi_market_order_manager.py**: Already handles variable quantity

### Day 3: Testing & Validation

#### Test Both Paths
```python
# test_position_sizing.py
class TestPositionSizing:
    def test_sync_path_with_simulated_manager(self):
        """Test position sizing in training environment."""
        
    def test_async_path_with_kalshi_manager(self):
        """Test position sizing in live trading."""
        
    def test_action_encoding_decoding(self):
        """Test all 21 actions encode/decode correctly."""
        
    def test_position_limits_enforced(self):
        """Verify risk limits work in both paths."""
```

## 4. Risk Management & Market Impact

### Liquidity-Aware Sizing
```python
class AdaptivePositionSizing:
    """Adjust position sizes based on market liquidity."""
    
    LIQUIDITY_MULTIPLIERS = {
        "high": 1.0,    # >$10k volume in 10min
        "medium": 0.5,  # $1k-$10k volume
        "low": 0.2      # <$1k volume
    }
    MAX_POSITION_PCT_OF_VOLUME = 0.05  # Max 5% of recent volume
```

### Market Impact Estimation
```python
def estimate_market_impact(size: int, orderbook: Dict, side: str) -> Tuple[float, float]:
    """Estimate price impact and spread cost of order."""
    # Implementation from previous plan
```

## 5. Files to Modify

### Core Changes (EXTEND, not delete)
1. `limit_order_action_space.py` - Extend both sync/async paths
2. `market_agnostic_env.py` - Update to 21 actions

### Verification Only (already support variable quantity)
3. `order_manager.py` - SimulatedOrderManager already supports quantity
4. `kalshi_multi_market_order_manager.py` - Already supports quantity
5. `actor_service.py` - Already passes action through correctly

### New Files
6. `position_sizing.py` - Centralized sizing logic
7. `test_position_sizing.py` - Test both execution paths

## 6. Critical Success Factors

### Must Pass Before Deployment
- ✅ All 21 actions execute in BOTH sync and async paths
- ✅ SimulatedOrderManager detection still works
- ✅ Training environment uses sync path
- ✅ Live trading uses async path
- ✅ Position limits enforced in both paths
- ✅ No breaking changes to existing architecture

### Architecture Preservation
- ✅ execute_action_sync still detects SimulatedOrderManager
- ✅ Async execution still works for KalshiMultiMarketOrderManager
- ✅ Training/inference pipeline isolation maintained

## 7. Implementation Notes for RL Agent

### Critical Reminders
1. **EXTEND, DON'T DELETE** - Keep the async/sync routing logic
2. **Test Both Paths** - Ensure training and live both work
3. **Preserve Detection Logic** - Keep SimulatedOrderManager type checking
4. **Update Both Paths** - Add position sizing to sync AND async
5. **Verify Order Managers** - Confirm they accept variable quantity

### What NOT to Do
- ❌ Don't delete execute_action_sync
- ❌ Don't remove SimulatedOrderManager detection
- ❌ Don't break the training/live separation
- ❌ Don't assume one execution path

### What TO Do
- ✅ Add decode_action to both paths
- ✅ Pass decoded size to order managers
- ✅ Keep existing architecture intact
- ✅ Test both SimulatedOrderManager and KalshiMultiMarketOrderManager

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

This revised plan preserves the critical architectural separation while adding position sizing to both training and live trading paths.