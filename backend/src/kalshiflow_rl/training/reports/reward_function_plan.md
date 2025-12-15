# Reward Function Misalignment Analysis and Fix Plan

## UPDATE: December 15, 2024 - Implementation Complete

### What We Actually Changed

After much investigation, we discovered and fixed several issues:

1. **Portfolio Calculation Misalignment (FIXED)**:
   - **Problem**: Portfolio values used mid-prices, hiding spread costs
   - **Solution**: Modified `Position.get_unrealized_pnl()` to accept bid/ask prices
   - Long positions now use bid price (what you can sell at)
   - Short positions use ask price for NO contracts
   - Files changed: `trading/order_manager.py`, `environments/market_agnostic_env.py`

2. **Missing Methods in KalshiMultiMarketOrderManager (FIXED)**:
   - **Problem**: Accidentally deleted `get_portfolio_value()` during refactoring
   - **Solution**: Restored 4 missing methods needed by ActorService
   - Added cash balance sync from Kalshi API
   - File changed: `trading/kalshi_multi_market_order_manager.py`

3. **Wide Spread Discovery (NOT A BUG)**:
   - **Finding**: 80% of markets have >20¢ spreads (genuinely illiquid)
   - **Impact**: Model correctly learns that trading these = guaranteed loss
   - **Solution**: Need to filter training data for liquid markets (<10¢ spreads)
   - This is a data selection issue, not a reward function bug

### The Journey ("There and Back Again")

1. Started with reward misalignment hypothesis
2. Implemented bid/ask spread awareness
3. Broke E2E tests (13 failures)
4. Discovered we deleted critical methods
5. Restored missing functionality
6. E2E tests pass again
7. Discovered the real problem: training on illiquid markets

### Current Status

✅ **Reward function alignment**: Complete and verified
✅ **E2E functionality**: Restored and passing tests
✅ **Portfolio syncing**: Working with Kalshi API
⚠️ **Training data**: Still dominated by illiquid markets (next fix needed)

---

# Original Analysis (Preserved for Context)

## I. Training-Trading Synchronization Strategy

### Critical Requirement
**The training environment and live trading system MUST use identical portfolio calculation and reward logic.** Any divergence between these systems will lead to:
- Models trained on incorrect signals that fail in live trading
- Unexpected behavior when deployed models encounter real market conditions
- Financial losses from misaligned incentives between training and execution

### Current Architecture Analysis

#### Shared Components
Both training and live trading currently share some common abstractions:
1. **OrderManager Base Class** (`trading/order_manager.py`)
   - Defines portfolio value calculation interface
   - Implements position tracking and P&L calculations
   - Provides `get_total_portfolio_value()` method

2. **Position Class** (`trading/order_manager.py`)
   - Tracks contract counts, cost basis, realized P&L
   - Calculates unrealized P&L given current prices
   - Shared between SimulatedOrderManager and KalshiMultiMarketOrderManager

#### Divergence Points

**Training Environment** (`environments/market_agnostic_env.py`):
- Uses `SimulatedOrderManager` for order execution
- Portfolio calculation: `order_manager.get_portfolio_value_cents()`
- Fill simulation: Immediate fills when crossing spread
- No actual spread costs deducted from portfolio

**Live Trading** (`trading/kalshi_multi_market_order_manager.py`):
- Uses `KalshiMultiMarketOrderManager` for real orders
- Portfolio calculation: `get_portfolio_value()` - simplified version
- Fill processing: Real fills from Kalshi API
- Spread costs implicit in actual execution prices

**Key Differences Identified:**
1. **Portfolio Calculation**:
   - Training: Uses mark-to-market with mid prices
   - Live: Currently uses only cost basis (line 815: `total_value += position.cost_basis`)
   - **CRITICAL BUG**: Live trading doesn't include unrealized P&L!

2. **Spread Cost Accounting**:
   - Training: Orders fill at best bid/ask but no spread cost deducted
   - Live: Real fills include spread cost implicitly
   - **MISALIGNMENT**: Training doesn't penalize crossing spreads

3. **Order Features**:
   - Both use similar feature extraction but different data sources
   - Potential for feature distribution shift

### Synchronization Implementation Plan

#### Phase 1: Create Unified Portfolio Calculator
Create a shared module that both systems MUST use:

**File**: `src/kalshiflow_rl/trading/portfolio_calculator.py`
```python
class UnifiedPortfolioCalculator:
    """
    Single source of truth for portfolio value calculations.
    Used by both training and live trading to ensure consistency.
    """
    
    @staticmethod
    def calculate_portfolio_value(
        cash_balance: float,
        positions: Dict[str, Position],
        current_prices: Dict[str, float],
        include_spread_cost: bool = True
    ) -> float:
        """
        Calculate total portfolio value with consistent logic.
        
        Args:
            cash_balance: Available cash in dollars
            positions: Position objects by ticker
            current_prices: Current mid prices by ticker
            include_spread_cost: Whether to deduct spread costs
        
        Returns:
            Total portfolio value in dollars
        """
        total_value = cash_balance
        
        for ticker, position in positions.items():
            if not position.is_flat and ticker in current_prices:
                # Always include unrealized P&L (critical fix)
                unrealized_pnl = position.get_unrealized_pnl(current_prices[ticker])
                position_value = position.cost_basis + unrealized_pnl
                
                # Optionally deduct spread cost for liquidation value
                if include_spread_cost:
                    # Assume 2 cent spread as conservative estimate
                    spread_cost = abs(position.contracts) * 0.02
                    position_value -= spread_cost
                
                total_value += position_value
        
        return total_value
```

#### Phase 2: Update Both Systems to Use Unified Calculator

**Training Environment Updates**:
```python
# market_agnostic_env.py
from ..trading.portfolio_calculator import UnifiedPortfolioCalculator

def step(self, action):
    # Before action
    prev_value = UnifiedPortfolioCalculator.calculate_portfolio_value(
        self.order_manager.cash_balance,
        self.order_manager.positions,
        self._get_current_prices(),
        include_spread_cost=True  # Include spreads in training
    )
    
    # ... execute action ...
    
    # After action
    new_value = UnifiedPortfolioCalculator.calculate_portfolio_value(
        self.order_manager.cash_balance,
        self.order_manager.positions,
        self._get_current_prices(),
        include_spread_cost=True
    )
    
    reward = new_value - prev_value
```

**Live Trading Updates**:
```python
# kalshi_multi_market_order_manager.py
from ..trading.portfolio_calculator import UnifiedPortfolioCalculator

def get_portfolio_value(self) -> float:
    """Calculate portfolio value using unified logic."""
    # Get current market prices from orderbook states
    current_prices = self._get_current_market_prices()
    
    return UnifiedPortfolioCalculator.calculate_portfolio_value(
        self.cash_balance,
        self.positions,
        current_prices,
        include_spread_cost=True  # Same as training
    )
```

#### Phase 3: Add Spread Cost to Fill Processing

**SimulatedOrderManager Updates**:
```python
def _process_fill(self, order, fill_price, fill_timestamp=None):
    # Existing fill logic...
    
    # NEW: Deduct spread cost from cash
    spread_cost = self._calculate_spread_cost(order, fill_price)
    self.cash_balance -= spread_cost
    
    # Track for metrics
    self.total_spread_costs += spread_cost
```

#### Phase 4: Create Validation Tests

**File**: `tests/test_portfolio_sync.py`
```python
def test_portfolio_calculation_consistency():
    """Verify training and live systems calculate identical values."""
    # Setup identical positions and prices
    positions = create_test_positions()
    prices = create_test_prices()
    
    # Calculate using training method
    training_value = training_env.calculate_portfolio_value(positions, prices)
    
    # Calculate using live method
    live_value = live_manager.get_portfolio_value(positions, prices)
    
    # Must be EXACTLY equal
    assert training_value == live_value
```

### Files Requiring Coordinated Updates

When implementing Phase 1 fix, these files MUST be updated together:

1. **Core Calculation Logic**:
   - `src/kalshiflow_rl/trading/portfolio_calculator.py` (NEW)
   - `src/kalshiflow_rl/trading/order_manager.py` (update base class)

2. **Training Environment**:
   - `src/kalshiflow_rl/environments/market_agnostic_env.py`
   - `src/kalshiflow_rl/environments/limit_order_action_space.py`

3. **Live Trading**:
   - `src/kalshiflow_rl/trading/kalshi_multi_market_order_manager.py`
   - `src/kalshiflow_rl/trading/actor_service.py`

4. **Tests**:
   - `tests/test_portfolio_sync.py` (NEW)
   - `tests/test_rl/environments/test_market_agnostic_env.py`

### Validation Approach

1. **Unit Tests**: Test UnifiedPortfolioCalculator with known inputs/outputs
2. **Integration Tests**: Verify both systems use the calculator correctly
3. **Regression Tests**: Ensure existing functionality not broken
4. **A/B Comparison**: Run parallel calculations to verify consistency
5. **Live Monitoring**: Add metrics to detect calculation divergence

### Risk Mitigation

1. **Gradual Rollout**: Test in paper trading before live deployment
2. **Feature Flags**: Allow switching between old/new calculation
3. **Monitoring**: Alert on portfolio value discrepancies
4. **Rollback Plan**: Keep old code paths available for quick revert

### Success Criteria

- [ ] Single portfolio calculation function used by both systems
- [ ] Spread costs included in both training and live trading
- [ ] Test coverage proving calculation consistency
- [ ] No divergence in portfolio values between systems
- [ ] Live trading includes unrealized P&L (critical bug fix)

## A. Executive Summary

### Problem Statement
The current reward function in the Kalshi RL trading environment exhibits a critical misalignment: agents receive positive rewards (avg 47.78 to 50.48 per step) while portfolio values remain completely flat at 10,000 cents. This disconnect means the model is optimizing for a signal that doesn't correlate with actual profitability, leading to learned behaviors that don't generate returns.

### Impact on Model Performance
- **False Learning Signal**: Model learns to maximize rewards that don't represent P&L
- **Wasted Training**: 100,000+ timesteps optimizing wrong objective
- **No Trading Behavior**: Portfolio remains static despite "positive" rewards
- **Misleading Metrics**: High rewards mask zero actual returns

### Proposed Solution Overview
Replace the current reward calculation with a proper P&L-based reward that:
1. Accurately tracks portfolio value changes including unrealized P&L
2. Properly accounts for bid-ask spreads in order execution
3. Normalizes rewards appropriately for stable learning
4. Provides clear signal for profitable vs unprofitable actions

## B. Current Implementation Analysis

### Code Walkthrough

The current reward calculation occurs in `market_agnostic_env.py` lines 183-242:

```python
# Line 183-186: Get portfolio value BEFORE action
prev_portfolio_value = self.order_manager.get_portfolio_value_cents(
    self._get_current_market_prices()
)

# Line 205-227: Execute action (order placement)
action_result = self.action_space_handler.execute_action_sync(
    action, self.current_market, orderbook
)

# Line 237-242: Calculate reward AFTER action
new_portfolio_value = self.order_manager.get_portfolio_value_cents(
    self._get_current_market_prices()
)
# Simple reward: portfolio value change in cents
reward = float(new_portfolio_value - prev_portfolio_value)
```

### Mathematical Formulation of Current Rewards

Current reward function:
```
R(t) = PV(t) - PV(t-1)
where:
  PV(t) = cash_balance + Σ(position_value_at_mid_price)
```

### Specific Examples Showing Misalignment

**Example 1: Order Placement Without Fill**
```
Step 1: Cash=10000¢, Position=0, PV=10000¢
Action: BUY_YES_LIMIT
Step 2: Cash=10000¢, Position=0, PV=10000¢ (order pending, not filled)
Reward: 10000 - 10000 = 0
```

**Example 2: Order Fills But No Price Movement**
```
Step 1: Cash=10000¢, Position=0, PV=10000¢
Action: BUY_YES_LIMIT at 50¢
Step 2: Cash=9500¢, Position=10 YES, PV=9500 + 10*50 = 10000¢
Reward: 10000 - 10000 = 0
```

The portfolio value only changes if:
1. An order fills AND
2. The market price moves after the fill

This creates a massive disconnect where the agent gets no reward for successful trades unless prices move immediately.

## C. Root Cause Analysis

### Why the Misalignment Occurs

1. **Portfolio Calculation Ignores Execution Costs**
   - When buying at ask or selling at bid, the immediate P&L loss from spread isn't captured
   - Portfolio value uses mid-price, masking the cost of crossing the spread

2. **Reward Timing Issue**
   - Rewards are calculated immediately after action
   - Most orders don't fill instantly (limit orders)
   - Even filled orders need price movement to show P&L

3. **Missing Transaction Costs**
   - No explicit modeling of spread costs in reward
   - Agent doesn't learn the cost of trading

4. **Unrealized vs Realized P&L Confusion**
   - Current system mixes concepts without clear accounting
   - Portfolio value calculation doesn't properly track entry prices vs current prices

### Common Pitfalls in RL Reward Design for Trading

1. **Sparse Reward Problem**: Only rewarding on position close leads to sparse signals
2. **Credit Assignment**: Hard to attribute P&L to specific actions
3. **Delayed Consequences**: Actions have effects that manifest over many steps
4. **Market Noise**: Random price movements can dominate reward signal

### Specific Issues in Our Implementation

1. **Order Manager's Portfolio Calculation** (`order_manager.py` lines 366-383):
   ```python
   def get_portfolio_value_cents(self, current_prices):
       # Uses mid-price for position valuation
       simple_prices[ticker] = yes_mid / 100.0
       total_value_dollars = self.get_total_portfolio_value(simple_prices)
   ```
   This always values positions at mid, hiding spread costs.

2. **Position Tracking** (`order_manager.py` lines 485-573):
   - Properly tracks cost basis and realized P&L
   - But portfolio value calculation doesn't use this information correctly

3. **No Order Fill Rewards**:
   - Agent gets no signal when orders successfully fill
   - Can't learn market timing or execution quality

## D. Proposed Solution

### New Reward Function Formula

```
R(t) = α * ΔPV_adjusted(t) + β * execution_quality(t) + γ * position_penalty(t)

where:
  ΔPV_adjusted(t) = (PV_marked(t) - PV_marked(t-1)) / scale_factor
  
  PV_marked(t) = cash + Σ(position * mark_price) - Σ(|position| * half_spread)
  
  execution_quality(t) = {
    if order filled:
      (mid_price - fill_price) * quantity * side_sign / scale_factor
    else:
      0
  }
  
  position_penalty(t) = -δ * |position|² / max_position²

Parameters:
  α = 1.0 (P&L weight)
  β = 0.1 (execution quality weight) 
  γ = 0.01 (position regularization weight)
  δ = 0.001 (position penalty coefficient)
  scale_factor = 100 (convert cents to reasonable range)
```

### Pseudocode Implementation

```python
def calculate_reward(self, prev_state, action, new_state, order_result):
    # 1. Calculate adjusted portfolio values
    prev_pv = self._calculate_marked_portfolio_value(prev_state)
    new_pv = self._calculate_marked_portfolio_value(new_state)
    
    # 2. P&L component
    pnl_reward = (new_pv - prev_pv) / 100.0  # Scale from cents
    
    # 3. Execution quality component
    exec_reward = 0.0
    if order_result and order_result.filled:
        mid_price = (orderbook.best_bid + orderbook.best_ask) / 2
        if order_result.side == 'buy':
            exec_reward = (mid_price - order_result.fill_price) / 100.0
        else:
            exec_reward = (order_result.fill_price - mid_price) / 100.0
        exec_reward *= order_result.quantity / 10.0  # Normalize by typical size
    
    # 4. Position penalty (encourage manageable positions)
    position_penalty = -0.001 * (abs(new_state.position) / 100) ** 2
    
    # 5. Combine components
    total_reward = pnl_reward + 0.1 * exec_reward + 0.01 * position_penalty
    
    return total_reward

def _calculate_marked_portfolio_value(self, state):
    """Calculate portfolio value with spread adjustment."""
    cash = state.cash_balance
    
    # Mark positions with spread cost
    position_value = 0
    for ticker, position_info in state.positions.items():
        position = position_info['position']
        if position != 0:
            # Get current orderbook
            orderbook = state.orderbooks[ticker]
            mid_price = (orderbook.best_bid + orderbook.best_ask) / 2
            half_spread = (orderbook.best_ask - orderbook.best_bid) / 2
            
            # Value position at mid minus half spread cost
            position_value += abs(position) * (mid_price - half_spread * 0.5)
    
    return cash + position_value
```

### How It Addresses Each Issue

1. **Spread Costs**: Explicitly deducts spread from position valuation
2. **Execution Quality**: Rewards good fills vs poor fills
3. **Immediate Feedback**: Execution quality gives instant signal
4. **Position Management**: Penalty prevents excessive positions
5. **Proper Scaling**: Normalizes cents to ~[-1, 1] range for stable learning

## E. Implementation Details

### Step-by-Step Code Changes

#### 1. Modify `market_agnostic_env.py`

**Location**: Lines 183-242 in `step()` method

**Current Code**:
```python
# Line 237-242
new_portfolio_value = self.order_manager.get_portfolio_value_cents(
    self._get_current_market_prices()
)
reward = float(new_portfolio_value - prev_portfolio_value)
```

**New Code**:
```python
# Line 237-242 replacement
new_portfolio_value = self.order_manager.get_portfolio_value_cents(
    self._get_current_market_prices()
)

# Calculate marked portfolio values with spread adjustment
prev_marked_value = self._calculate_marked_portfolio_value(
    prev_portfolio_value, 
    self.order_manager.positions,
    self._get_prev_orderbook()
)
new_marked_value = self._calculate_marked_portfolio_value(
    new_portfolio_value,
    self.order_manager.positions,
    orderbook
)

# P&L reward component (scaled)
pnl_reward = (new_marked_value - prev_marked_value) / 100.0

# Execution quality reward
exec_reward = self._calculate_execution_reward(action_result, orderbook)

# Position penalty
position = self.order_manager.positions.get(self.current_market, Position()).contracts
position_penalty = -0.001 * (abs(position) / 100) ** 2

# Combine rewards
reward = pnl_reward + 0.1 * exec_reward + 0.01 * position_penalty
```

#### 2. Add Helper Methods to `market_agnostic_env.py`

**Add after line 374**:

```python
def _calculate_marked_portfolio_value(
    self, 
    base_value: int,
    positions: Dict[str, Position],
    orderbook: Optional[OrderbookState]
) -> int:
    """
    Calculate portfolio value with spread cost adjustment.
    
    Args:
        base_value: Base portfolio value in cents
        positions: Current positions
        orderbook: Current orderbook for spread calculation
        
    Returns:
        Adjusted portfolio value in cents
    """
    if not orderbook or not positions:
        return base_value
    
    # Calculate spread adjustment
    spread_cost = 0
    for ticker, position in positions.items():
        if position.contracts != 0 and ticker == self.current_market:
            # Get spread
            best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
            best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
            
            if best_bid and best_ask:
                half_spread = (best_ask - best_bid) / 2.0
                # Deduct half spread per contract as liquidation cost
                spread_cost += abs(position.contracts) * half_spread * 0.5
    
    return base_value - int(spread_cost)

def _calculate_execution_reward(
    self,
    action_result: Optional[ActionExecutionResult],
    orderbook: OrderbookState
) -> float:
    """
    Calculate execution quality reward.
    
    Rewards good fills (better than mid) and penalizes bad fills.
    
    Args:
        action_result: Result from action execution
        orderbook: Current orderbook
        
    Returns:
        Execution quality reward (scaled)
    """
    if not action_result or not hasattr(action_result, 'order'):
        return 0.0
    
    order = action_result.order
    if not order or order.status != OrderStatus.FILLED:
        return 0.0
    
    # Calculate mid price
    best_bid = orderbook._get_best_price(orderbook.yes_bids, is_bid=True)
    best_ask = orderbook._get_best_price(orderbook.yes_asks, is_bid=False)
    
    if not best_bid or not best_ask:
        return 0.0
    
    mid_price = (best_bid + best_ask) / 2.0
    
    # Calculate execution quality
    if order.side == OrderSide.BUY:
        # Good execution: bought below mid
        quality = (mid_price - order.fill_price) / 100.0
    else:
        # Good execution: sold above mid
        quality = (order.fill_price - mid_price) / 100.0
    
    # Scale by quantity (normalize to 10 contracts)
    quality *= order.quantity / 10.0
    
    return quality

def _get_prev_orderbook(self) -> Optional[OrderbookState]:
    """Get orderbook from previous timestep for comparison."""
    if self.current_step > 0:
        prev_data = self.market_view.get_timestep_data(self.current_step - 1)
        if prev_data and self.current_market in prev_data.markets_data:
            return convert_session_data_to_orderbook(
                prev_data.markets_data[self.current_market],
                self.current_market
            )
    return None
```

#### 3. Modify `SimulatedOrderManager` to Track Fill Information

**Location**: `order_manager.py`, enhance `_process_fill()` method

**Add fill tracking to OrderInfo** (line 498):
```python
# Update order status
order.status = OrderStatus.FILLED
order.filled_at = fill_timestamp
order.fill_price = fill_price
# ADD: Store in last fill for reward calculation
self.last_fill = {
    'order': order,
    'timestamp': fill_timestamp,
    'spread_at_fill': getattr(orderbook, 'spread', 0) if orderbook else 0
}
```

### Files to Modify

1. `src/kalshiflow_rl/environments/market_agnostic_env.py`
   - Main reward calculation logic
   - Helper methods for portfolio marking
   - Execution quality calculation

2. `src/kalshiflow_rl/trading/order_manager.py`
   - Enhanced fill tracking
   - Spread-aware portfolio calculation

3. `src/kalshiflow_rl/environments/limit_order_action_space.py`
   - Return fill information in ActionExecutionResult

### Testing Approach

1. **Unit Tests**:
   ```python
   def test_reward_with_spread_cost():
       """Test that spread costs are reflected in rewards."""
       env = create_test_env()
       
       # Place buy order at ask
       prev_value = env.order_manager.get_portfolio_value_cents()
       action = LimitOrderActions.BUY_YES_LIMIT
       obs, reward, done, truncated, info = env.step(action)
       
       # Reward should be negative (spread cost)
       assert reward < 0, "Buying at ask should incur spread cost"
   
   def test_execution_quality_reward():
       """Test execution quality component."""
       env = create_test_env()
       
       # Place limit order below mid
       # Should get positive execution reward if filled
   ```

2. **Integration Tests**:
   - Run short training episodes
   - Verify rewards correlate with P&L
   - Check that portfolio value changes match reward accumulation

3. **Regression Tests**:
   - Ensure rewards are in reasonable range [-1, 1]
   - Verify no NaN or inf values
   - Check reward components sum correctly

## F. Expected Impact

### How This Will Improve Training

1. **Aligned Incentives**: Agent learns to maximize actual P&L, not phantom rewards
2. **Faster Learning**: Immediate feedback from execution quality
3. **Better Exploration**: Position penalty prevents stuck states
4. **Realistic Behavior**: Agent learns cost of trading and spread management

### Metrics to Monitor

1. **Reward-P&L Correlation**: Should be > 0.9
2. **Average Episode Return**: Should match portfolio change
3. **Fill Rate**: Percentage of orders that execute
4. **Spread Capture**: Average execution vs mid-price
5. **Position Turnover**: Frequency of position changes
6. **Sharpe Ratio**: Risk-adjusted returns

### Success Criteria

- [ ] Portfolio value changes when trades execute
- [ ] Rewards correlate with P&L (correlation > 0.9)
- [ ] Agent learns to manage positions (not just accumulate)
- [ ] Positive returns achieved within 100k timesteps
- [ ] Spread costs reflected in decision making

## G. Alternative Approaches

### 1. Pure P&L Reward (Simplest)
**Formula**: `R(t) = (PV(t) - PV(t-1)) / scale`

**Pros**:
- Dead simple
- Directly optimizes what we care about
- No hyperparameter tuning

**Cons**:
- Sparse signal (only on fills)
- No execution quality feedback
- Can lead to position accumulation

### 2. Sharpe Ratio Optimization
**Formula**: `R(t) = sharpe_ratio(returns[-N:])` 

**Pros**:
- Optimizes risk-adjusted returns
- Naturally controls position size
- Industry standard metric

**Cons**:
- Requires return history buffer
- Numerical instability with small denominators
- Delayed feedback

### 3. Multi-Objective Reward
**Formula**: `R(t) = Σ(wi * ri)` for multiple objectives

**Pros**:
- Can optimize multiple goals
- Flexible framework
- Explicit control over behavior

**Cons**:
- Many hyperparameters to tune
- Objectives may conflict
- Hard to interpret

### Why the Proposed Solution is Best

Our proposed solution (marked P&L + execution quality + position penalty) is optimal because:

1. **Immediate Feedback**: Execution quality provides instant signal
2. **True P&L Tracking**: Spread-adjusted portfolio value reflects reality
3. **Balanced Objectives**: Manages profitability vs risk naturally
4. **Tunable but Simple**: Few parameters, clear interpretation
5. **Battle-Tested**: Similar approaches work in production trading systems

## H. Code Examples

### Current Problematic Code

```python
# market_agnostic_env.py, lines 237-242
new_portfolio_value = self.order_manager.get_portfolio_value_cents(
    self._get_current_market_prices()
)
# Simple reward: portfolio value change in cents
reward = float(new_portfolio_value - prev_portfolio_value)
```

**Problems**:
- No spread adjustment
- No execution feedback
- Raw cents not scaled
- No position management

### Proposed Fixed Code

```python
# market_agnostic_env.py, lines 237-260 (expanded)
# Get portfolio values
new_portfolio_value = self.order_manager.get_portfolio_value_cents(
    self._get_current_market_prices()
)

# Calculate P&L with spread adjustment
spread_cost = self._calculate_spread_cost(orderbook, self.order_manager.positions)
adjusted_pv_change = (new_portfolio_value - prev_portfolio_value - spread_cost) / 100.0

# Execution quality bonus/penalty
exec_quality = 0.0
if action_result and action_result.order_placed and action_result.fill_info:
    fill_info = action_result.fill_info
    mid_price = (orderbook.best_bid + orderbook.best_ask) / 2
    if fill_info['side'] == 'buy':
        exec_quality = (mid_price - fill_info['price']) * fill_info['quantity'] / 1000.0
    else:
        exec_quality = (fill_info['price'] - mid_price) * fill_info['quantity'] / 1000.0

# Position management penalty
position = self.order_manager.positions.get(self.current_market, Position()).contracts
position_penalty = -0.0001 * (position / 10) ** 2  # Normalized by typical size

# Combine reward components
reward = adjusted_pv_change + 0.1 * exec_quality + position_penalty

# Log components for debugging
logger.debug(f"Reward components: P&L={adjusted_pv_change:.4f}, Exec={exec_quality:.4f}, Pos={position_penalty:.4f}")
```

### Test Cases

```python
def test_spread_cost_in_reward():
    """Verify spread costs reduce rewards appropriately."""
    env = MarketAgnosticKalshiEnv(market_view, config)
    env.reset()
    
    # Setup: Market with 2 cent spread
    orderbook = OrderbookState("TEST")
    orderbook.yes_bids = {49: 100}
    orderbook.yes_asks = {51: 100}
    
    # Action: Buy at ask
    initial_cash = env.order_manager.cash_balance
    obs, reward, _, _, info = env.step(LimitOrderActions.BUY_YES_LIMIT)
    
    # Verify: Negative reward from spread cost
    expected_spread_cost = 10 * 1  # 10 contracts * 1 cent half-spread
    expected_reward = -expected_spread_cost / 100.0
    assert abs(reward - expected_reward) < 0.01, f"Expected reward {expected_reward}, got {reward}"

def test_execution_quality_bonus():
    """Test that good fills get execution bonus."""
    env = MarketAgnosticKalshiEnv(market_view, config)
    env.reset()
    
    # Setup: Place passive limit order
    orderbook = OrderbookState("TEST")
    orderbook.yes_bids = {49: 100}
    orderbook.yes_asks = {51: 100}
    
    # Action: Buy at 49 (join bid)
    env.order_manager.pricing_strategy = "passive"
    obs, reward, _, _, info = env.step(LimitOrderActions.BUY_YES_LIMIT)
    
    # Simulate fill at our price (good execution)
    # Would save 1 cent vs crossing spread
    execution_bonus = 1.0 * 10 / 1000.0  # 1 cent * 10 contracts / scale
    assert reward > 0, "Good execution should yield positive execution reward component"

def test_position_penalty():
    """Test position accumulation penalty."""
    env = MarketAgnosticKalshiEnv(market_view, config)
    env.reset()
    
    # Build large position
    for _ in range(10):
        env.step(LimitOrderActions.BUY_YES_LIMIT)
    
    # Next trade should have position penalty
    obs, reward, _, _, info = env.step(LimitOrderActions.BUY_YES_LIMIT)
    
    # Position = 110 contracts, penalty should be significant
    position_penalty = -0.0001 * (110 / 10) ** 2
    assert position_penalty < -0.01, "Large positions should incur penalty"
```

## Implementation Priority

### Phase 1: Core Fix (Immediate)

**CRITICAL: Must follow Section I synchronization strategy to ensure training-trading consistency**

1. **Create UnifiedPortfolioCalculator** (see Section I)
   - Single source of truth for portfolio calculations
   - Used by both training and live trading
   - Includes spread costs in portfolio value

2. **Update portfolio value calculations**:
   - SimulatedOrderManager: Use UnifiedPortfolioCalculator
   - KalshiMultiMarketOrderManager: Use UnifiedPortfolioCalculator
   - Fix live trading bug (not including unrealized P&L)

3. **Fix reward calculation**:
   - Include spread costs in portfolio value changes
   - Scale rewards properly (cents to ~[-1, 1])
   - Add basic P&L tracking logs

4. **Validate synchronization**:
   - Create test_portfolio_sync.py
   - Verify identical calculations between systems
   - Add monitoring for divergence detection

### Phase 2: Execution Quality (Next Sprint)  
1. Add execution quality reward component
2. Track fill quality metrics
3. Enhance logging with spread capture stats

### Phase 3: Position Management (Future)
1. Add position penalty term
2. Implement risk limits
3. Add more sophisticated position sizing

## Conclusion

The current reward function fundamentally fails to align with trading profitability. The proposed solution directly addresses this by:

1. **Properly accounting for trading costs** via spread-adjusted portfolio values
2. **Providing immediate feedback** through execution quality rewards
3. **Encouraging sustainable trading** via position penalties
4. **Scaling appropriately** for stable neural network training

This fix is critical and should be implemented immediately before further training, as the current system is teaching the agent behaviors that don't correlate with actual trading success.