# Position Sizing Implementation Plan

## Executive Summary
Add variable position sizing to the Kalshi RL trading system by expanding the action space from 5 to 25 actions (5 trading intents × 5 position sizes). This is a breaking change that removes the fixed 10-contract system.

## 1. Position Size Definitions

### Contract Sizes
- **Small**: 5 contracts
- **Medium**: 10 contracts (current default)
- **Large**: 20 contracts
- **Huge**: 50 contracts
- **Giant**: 100 contracts

### Risk Management Rationale
- Small (5): Testing waters, low conviction trades
- Medium (10): Standard size, balanced risk/reward
- Large (20): Higher conviction without excessive exposure
- Huge (50): Strong conviction, significant capital allocation
- Giant (100): Maximum conviction, aggressive positioning

## 2. Action Space Expansion

### Current System (5 actions)
```
0: HOLD
1: BUY_YES_LIMIT
2: SELL_YES_LIMIT
3: BUY_NO_LIMIT
4: SELL_NO_LIMIT
```

### New System (25 actions)
```
Action = base_action * 5 + size_index

0: HOLD (no size needed)
1-5: BUY_YES_LIMIT (Small, Medium, Large, Huge, Giant)
6-10: SELL_YES_LIMIT (Small, Medium, Large, Huge, Giant)
11-15: BUY_NO_LIMIT (Small, Medium, Large, Huge, Giant)
16-20: SELL_NO_LIMIT (Small, Medium, Large, Huge, Giant)
21-24: Reserved for future use
```

### Action Encoding/Decoding
```python
def decode_action(action: int) -> Tuple[int, int]:
    """Decode action into base action and size index."""
    if action == 0:
        return 0, 0  # HOLD, no size
    
    adjusted = action - 1
    base_action = (adjusted // 5) + 1  # 1-4
    size_index = adjusted % 5  # 0-4
    return base_action, size_index

def encode_action(base_action: int, size_index: int) -> int:
    """Encode base action and size into single action."""
    if base_action == 0:
        return 0  # HOLD
    return (base_action - 1) * 5 + size_index + 1
```

## 3. Implementation Steps

### Phase 1: Core Action Space Changes

#### 3.1 Update LimitOrderActionSpace (`limit_order_action_space.py`)
```python
# Key changes:
1. Add POSITION_SIZES constant:
   POSITION_SIZES = [5, 10, 20, 50, 100]  # Small to Giant

2. Update LimitOrderActions enum to include size variants:
   - Keep base actions (0-4) for backward reference
   - Add size-specific actions (1-20)

3. Modify __init__ to accept variable sizes:
   - Remove fixed contract_size parameter
   - Add position_sizes parameter with defaults

4. Update get_gym_space():
   - Change from Discrete(5) to Discrete(25)

5. Modify execute_action() and execute_action_sync():
   - Decode action to get base_action and size_index
   - Pass appropriate contract size to order placement

6. Update action validation and masking:
   - Consider available cash for different sizes
   - Add position limits per size tier
```

#### 3.2 Update MarketAgnosticKalshiEnv (`market_agnostic_env.py`)
```python
# Key changes:
1. Update action space:
   - Change from spaces.Discrete(5) to spaces.Discrete(25)

2. No observation space changes needed initially
   - Position size is implicit in the action taken
   - Model learns to choose size based on market conditions

3. Update step() method:
   - Pass full action (0-24) to action_space_handler
   - Handler decodes and executes with appropriate size
```

### Phase 2: Training Infrastructure Updates

#### 3.3 Update Training Scripts (`train_sb3.py`)
```python
# Key changes:
1. Update model initialization:
   - Ensure policy network handles 25 outputs
   - May need to adjust network architecture

2. Update action masking if used:
   - Mask invalid size choices based on cash
   - Consider position limits

3. Update logging:
   - Track position size distribution
   - Monitor average position size per episode
```

#### 3.4 Update SB3 Wrapper (`sb3_wrapper.py`)
```python
# Key changes:
1. Ensure action space is properly passed through
2. Update any action validation or preprocessing
```

### Phase 3: Live Trading Updates

#### 3.5 Update ActorService (`actor_service.py`)
```python
# Key changes:
1. Handle 25-action output from model
2. Pass decoded action to order manager
3. Update action logging to include size
```

#### 3.6 Update OrderManagers (`order_manager.py`, `kalshi_multi_market_order_manager.py`)
```python
# Key changes:
1. SimulatedOrderManager:
   - Already accepts variable quantity
   - No changes needed

2. KalshiMultiMarketOrderManager:
   - Already accepts variable quantity
   - Ensure proper handling of larger orders
   - Add position limit checks
```

### Phase 4: Testing Updates

#### 3.7 Update Tests
```python
# Files to update:
1. test_limit_order_action_space.py:
   - Test all 25 actions
   - Test action encoding/decoding
   - Test size-specific validation

2. test_limit_order_integration.py:
   - Test different position sizes
   - Test cash constraints per size

3. test_market_agnostic_env.py:
   - Test with 25-action space
   - Verify reward calculation with different sizes
```

## 4. Observation Space Considerations

### Current Approach (No Changes Needed)
- Keep observation space unchanged initially
- Model learns to infer appropriate size from market conditions
- Position size is implicit in portfolio features

### Future Enhancement (Optional)
```python
# Could add position size distribution features:
- current_position_size: Normalized current position
- avg_position_size: Recent average size
- max_position_capacity: Based on available cash
```

## 5. Reward Function Considerations

### No Changes Needed
- Current reward = portfolio value change already handles variable sizes correctly
- Larger positions naturally create larger rewards/penalties
- Risk/reward tradeoff emerges naturally

### Implicit Incentives
- Large positions in good opportunities → Higher rewards
- Large positions in bad trades → Higher penalties
- Natural pressure to size appropriately

## 6. Risk Management

### Position Limits
```python
class PositionLimits:
    MAX_POSITION_PER_MARKET = 500  # contracts
    MAX_POSITION_VALUE = 50000  # $500 in cents
    MAX_PERCENT_OF_PORTFOLIO = 0.25  # 25% max per position
    
    def validate_order(self, size, current_position, cash_balance, price):
        # Check total position after order
        new_position = current_position + size
        if abs(new_position) > self.MAX_POSITION_PER_MARKET:
            return False
            
        # Check value limits
        position_value = size * price
        if position_value > self.MAX_POSITION_VALUE:
            return False
            
        # Check portfolio percentage
        if position_value > cash_balance * self.MAX_PERCENT_OF_PORTFOLIO:
            return False
            
        return True
```

### Cash Management
- Ensure adequate cash for position sizes
- Reserve minimum cash buffer (e.g., $50)
- Prevent overleveraging

## 7. Migration Strategy

### Training Migration
1. Start fresh training with new 25-action space
2. No need to migrate old models (incompatible)
3. Compare performance metrics between fixed and variable sizing

### Paper Trading Migration
1. Deploy new model to paper trading
2. Monitor position size distribution
3. Verify risk limits are respected
4. Compare profitability vs fixed sizing

## 8. Monitoring & Metrics

### New Metrics to Track
```python
class PositionSizeMetrics:
    - position_size_distribution: Dict[int, int]  # {5: 10, 10: 25, ...}
    - avg_winning_position_size: float
    - avg_losing_position_size: float
    - size_efficiency: float  # correlation between size and profit
    - max_position_reached: int
    - size_changes_per_episode: int
```

### Dashboard Updates
- Add position size histogram
- Show size vs profit correlation
- Track sizing patterns over time

## 9. Testing Strategy

### Unit Tests
1. Test each of 25 actions executes correctly
2. Test action encoding/decoding logic
3. Test position limit enforcement
4. Test cash constraint validation

### Integration Tests
1. Full episode with various position sizes
2. Verify reward calculation accuracy
3. Test order manager with all sizes
4. Verify portfolio tracking

### E2E Tests
1. Train small model with new action space
2. Run paper trading simulation
3. Verify live order execution
4. Check risk limit compliance

## 10. Implementation Order

### Day 1: Core Changes
1. Update `limit_order_action_space.py` with 25 actions
2. Update `market_agnostic_env.py` action space
3. Create action encoding/decoding utilities
4. Update basic tests

### Day 2: Training Integration
1. Update training scripts
2. Test training pipeline
3. Verify metrics collection
4. Run short training session

### Day 3: Live Trading & Testing
1. Update ActorService
2. Comprehensive testing
3. Paper trading validation
4. Documentation updates

## 11. Rollback Plan

If issues arise:
1. Models are incompatible - must use new models
2. Can quickly revert code to 5-action system
3. Keep old model checkpoints for comparison
4. Document performance differences

## 12. Success Criteria

### Functional Success
- [x] All 25 actions execute correctly
- [x] Training completes without errors
- [x] Paper trading executes orders properly
- [x] Risk limits enforced

### Performance Success
- [ ] Variable sizing improves profitability by >10%
- [ ] Sharpe ratio improves
- [ ] Model learns meaningful sizing patterns
- [ ] Appropriate size selection for market conditions

## Appendix: Code Examples

### Action Decoding Example
```python
def process_action(self, action: int) -> Tuple[str, int]:
    """Process raw action into intent and size."""
    if action == 0:
        return "HOLD", 0
    
    # Decode action
    base_action, size_index = decode_action(action)
    size = POSITION_SIZES[size_index]
    
    intent_map = {
        1: "BUY_YES",
        2: "SELL_YES", 
        3: "BUY_NO",
        4: "SELL_NO"
    }
    
    return intent_map[base_action], size
```

### Testing Example
```python
def test_all_position_sizes():
    """Test each position size works correctly."""
    env = create_test_env()
    
    for base_action in range(1, 5):
        for size_idx, size in enumerate(POSITION_SIZES):
            action = encode_action(base_action, size_idx)
            obs, reward, done, truncated, info = env.step(action)
            
            # Verify correct size was used
            assert info['last_order_size'] == size
```

## Notes

- This is a breaking change - no backward compatibility
- Focus on simplicity - avoid over-engineering
- Let the model learn sizing strategies
- Monitor carefully during paper trading
- Consider adding size decay over time (future enhancement)