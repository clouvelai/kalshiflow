# Reward Function Implementation Assessment

*Date: December 15, 2024*
*Assessor: RL Expert System*

## Executive Summary

The Phase 1 reward function improvements were **partially implemented successfully** but took a simplified approach compared to the original plan. The core issue of reward-P&L alignment was addressed, but the implementation revealed that the fundamental problem was **not the reward calculation** but rather **training on illiquid markets**.

## What Was Actually Changed vs The Plan

### ✅ Successfully Implemented

1. **Bid-Ask Spread Awareness in Portfolio Calculation**
   - `Position.get_unrealized_pnl()` now accepts bid/ask prices
   - Long positions use bid price (what you can sell at)
   - Short positions use ask price for NO contracts
   - Backward compatibility maintained with mid-price fallback

2. **Environment Using Bid/Ask Prices**
   - `_get_current_market_prices()` extracts bid/ask from orderbooks
   - Portfolio value calculations pass bid/ask when available
   - Falls back to mid prices with minimal spread when bid/ask unavailable

3. **Missing Methods Restored**
   - Added 4 critical methods to `KalshiMultiMarketOrderManager`
   - Fixed E2E test failures (13 → 0 failures)
   - Cash balance syncs from Kalshi API

### ❌ NOT Implemented (From Original Plan)

1. **UnifiedPortfolioCalculator Class**
   - Plan called for separate calculator module
   - Instead: Extended existing `OrderManager` base class methods
   - **Assessment**: Simpler approach achieved same goal

2. **Spread Cost Deduction in Rewards**
   - Plan included explicit spread penalty in reward calculation
   - Instead: Implicit through bid/ask portfolio valuation
   - **Assessment**: Current approach is cleaner and more realistic

3. **Execution Quality Reward Component**
   - Plan had multi-component reward (P&L + execution + position penalty)
   - Instead: Simple portfolio value change only
   - **Assessment**: KISS principle - good decision to keep simple

4. **Live Trading Unrealized P&L Fix**
   - `KalshiMultiMarketOrderManager.get_portfolio_value()` still doesn't use bid/ask
   - Uses `cost_basis + realized_pnl` (missing unrealized P&L)
   - **Critical Bug**: Live trading portfolio values are incorrect!

## Assessment: Did We Fix the Core Issue?

### ✅ Reward-P&L Alignment: MOSTLY FIXED

**Training Environment:**
- Portfolio calculations now use realistic bid/ask prices
- Rewards properly reflect spread costs when entering positions
- Backward compatibility maintained for existing code

**Live Trading:**
- ⚠️ **Critical Issue**: `get_portfolio_value()` doesn't include unrealized P&L
- Portfolio value = cash + cost_basis + realized_pnl (WRONG)
- Should be = cash + cost_basis + unrealized_pnl (using bid/ask)
- **Impact**: Live portfolio tracking will be incorrect

### 🤔 Did We Keep It Simple or Overcomplicate?

**Kept Simple (Good):**
- No multi-component reward function
- No separate calculator class
- Used existing position/order manager infrastructure
- Maintained backward compatibility

**Overcomplicated (Minor):**
- Bid/ask extraction logic could be cleaner
- Some duplicate code between training and live systems

**Assessment**: Overall kept appropriately simple. The team avoided the temptation to over-engineer.

## Is the System Still Functional?

### ✅ Training System: FULLY FUNCTIONAL
- E2E tests pass
- Training completes successfully
- Models can be saved and loaded
- Curriculum learning works

### ⚠️ Trading System: PARTIALLY FUNCTIONAL
- Order placement works
- Position tracking works
- Portfolio value calculation is incorrect (missing unrealized P&L)
- Will cause discrepancy between displayed and actual portfolio value

### ✅ Data Collection: FULLY FUNCTIONAL
- Orderbook collection works
- WebSocket streaming works
- Database storage works

## What Should We Keep, Remove, or Fix?

### 🟢 KEEP (Working Well)

1. **Bid/Ask Aware Position P&L**
   - Clean implementation in `Position.get_unrealized_pnl()`
   - Good backward compatibility
   - Correct logic for long/short positions

2. **Environment Price Extraction**
   - `_get_current_market_prices()` properly extracts bid/ask
   - Good fallback to mid prices
   - Clean integration with portfolio calculations

3. **Simple Reward Function**
   - Portfolio value change only
   - No complex multi-component rewards
   - Easy to debug and understand

### 🔴 FIX IMMEDIATELY

1. **Live Trading Portfolio Calculation**
```python
# In KalshiMultiMarketOrderManager.get_portfolio_value()
# Current (WRONG):
def get_portfolio_value(self) -> float:
    total = self.cash_balance
    for position in self.positions.values():
        if not position.is_flat:
            total += position.cost_basis + position.realized_pnl  # WRONG!
    return total

# Should be:
def get_portfolio_value(self) -> float:
    total = self.cash_balance
    for ticker, position in self.positions.items():
        if not position.is_flat and ticker in self.orderbook_states:
            orderbook = self.orderbook_states[ticker]
            # Extract bid/ask from orderbook
            yes_bid = self._get_best_bid(orderbook)
            yes_ask = self._get_best_ask(orderbook)
            # Use bid/ask for unrealized P&L
            unrealized_pnl = position.get_unrealized_pnl(
                yes_bid=yes_bid/100.0,  # Convert cents to probability
                yes_ask=yes_ask/100.0
            )
            total += position.cost_basis + unrealized_pnl
    return total
```

2. **Add Spread Features to Observation Space**
   - Currently model can't see spreads explicitly
   - Add spread_cents, spread_pct features
   - Critical for market selection

### 🟡 CONSIDER REMOVING (Not Needed)

1. **Complex Reward Components**
   - Execution quality reward not needed yet
   - Position penalty not needed with proper risk management
   - Keep simple until proven need for complexity

2. **test_spread_fix.py**
   - Temporary test file with incorrect API usage
   - Tests are covered in test_portfolio_sync.py

## The Real Problem: Market Liquidity

The investigation revealed the actual issue wasn't the reward function but:
- **80% of markets have >20¢ spreads** (economically untradeable)
- Model correctly learns these are unprofitable
- Solution is data filtering, not reward engineering

## Recommendations

### Immediate Actions (Priority Order)

1. **Fix Live Trading Portfolio Calculation**
   - Add unrealized P&L using bid/ask prices
   - Test with paper trading account
   - Verify portfolio values match expected

2. **Implement Market Liquidity Filtering**
   - Filter training to markets with <10¢ spreads
   - Use sessions 41 or 70 with more liquid markets
   - Add spread features to observations

3. **Clean Up Test Files**
   - Remove test_spread_fix.py
   - Update test_portfolio_sync.py with full coverage
   - Add integration test for live trading portfolio

### Validation Steps

```bash
# 1. Test portfolio calculation consistency
uv run pytest tests/test_portfolio_sync.py -v

# 2. Verify E2E still passes
uv run pytest tests/test_rl_backend_e2e_regression.py -v

# 3. Test paper trading with fixed portfolio
ENVIRONMENT=paper uv run python src/kalshiflow_rl/scripts/validate_model.py \
  --model models/kalshi_rl_phase1_initial.zip

# 4. Train new model with liquid markets only
uv run python src/kalshiflow_rl/training/train_sb3.py \
  --session 41 --max-spread 10 --total-timesteps 100000
```

## Conclusion

The reward function improvements successfully addressed the technical alignment between rewards and portfolio values in the training environment. The implementation was appropriately simple, avoiding over-engineering. However, a critical bug remains in live trading portfolio calculation that must be fixed immediately.

The discovery that 80% of markets are economically untradeable due to wide spreads is the more fundamental issue. The model's behavior (avoiding trades) is actually correct given these market conditions. The path forward is clear:

1. Fix live trading portfolio calculation
2. Filter training data to liquid markets
3. Add spread awareness features
4. Retrain and validate profitability

The system architecture is sound, the training pipeline works, and the core issues are identified with clear solutions. The team is well-positioned to achieve profitability with these targeted fixes.