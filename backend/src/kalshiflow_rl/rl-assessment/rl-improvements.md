# RL Training Improvements - Priority Ranked

Last Updated: 2025-12-17 (CRITICAL UPDATE: Session 12 Model Policy Collapse)

## CRITICAL: Session 12 Model Has Collapsed - Immediate Action Required

### 0.1 IMMEDIATE: Replace Broken Session 12 Model
**Status**: 🚨 **URGENT - Model is executing 96% SELL_NO only**
**Priority**: CRITICAL - Financial risk if deployed
**Evidence**: Training shows 962,049 SELL_NO out of 1,003,520 actions (95.87%)

**Immediate Actions**:
1. **Stop using session12_ppo_final.zip immediately**
2. **Revert to session32_final.zip or use hardcoded HOLD**
3. **Do NOT trade with current model - it will lose money**

**Root Cause Analysis**:
- Entropy coefficient too low (0.01) → exploration collapsed
- Reward function biased toward SELL_NO (+1.32 avg reward vs negative for others)
- 7 features with zero variance in training data
- No action diversity requirements or masking

### 0.2 Retrain with Anti-Collapse Measures
**Priority**: CRITICAL - Must fix before any production use
**Implementation**: 
```python
# Key changes for retraining
model = PPO(
    "MlpPolicy", env,
    ent_coef=0.1,    # 10x increase from 0.01
    clip_range=0.1,   # More conservative updates
    target_kl=0.01,   # Early stopping on divergence
    n_steps=256       # Smaller batches
)

# Add to reward function
def add_diversity_bonus(action_history):
    action_entropy = calculate_entropy(action_history[-100:])
    return action_entropy * 0.1  # Reward action diversity

# Implement action masking
if consecutive_same_actions > 3:
    mask_repeated_action()
```

**Validation Requirements**:
- Entropy must stay > 1.0 throughout training
- No action should exceed 40% of total actions
- Test on multiple sessions (not just session 12)
- Verify balanced action distribution in paper trading

Last Updated: 2025-12-17 (CRITICAL UPDATE: Session 12 Model Policy Collapse)

## URGENT: Simulation Fidelity Issues (✅ RESOLVED - Dec 15, 2024)

### 0. Fix Order Simulation Realism 
**Status**: ✅ **COMPLETED - Full Pipeline Validated (Dec 15, 2024)**
**Impact**: CRITICAL - Successfully validated end-to-end training with realistic fills
**Evidence**: PPO training on session 32 (10k timesteps) completed without errors

**Completed Features** ✅:
- **Orderbook depth consumption** - Large orders walk the book with realistic slippage
- **VWAP pricing** - Volume-weighted average prices for multi-level fills  
- **Partial fills** - Orders partially fill when liquidity insufficient
- **Consumed liquidity tracking** - 5-second decay prevents double-filling
- **SimulatedOrderManager integration** - Fully integrated with MarketAgnosticKalshiEnv
- **21-action space** - Variable position sizing (5 levels × 2 sides + HOLD)

**Validation Results (Session 32, 10k timesteps)**:
- ✅ Training completed: 116 episodes, 4.15 seconds runtime
- ✅ No NaN/inf errors detected  
- ✅ Consistent order fills logged throughout training
- ✅ Portfolio value tracking works correctly ($100 starting cash)
- ✅ Win rate: 28.45% (33/116 episodes profitable - realistic for early training)
- ✅ Speed: 2,410 timesteps/second (excellent performance)

**Remaining Enhancements** (Lower Priority):
- Probabilistic fill model based on price/size/time (Nice to have)
- Market impact modeling (spread widening from orders) (Future work)
- Time-in-queue priority for limit orders (Future work)

**Implementation**: See `trading/order_manager.py:calculate_fill_with_depth()`
**Tests**: See `tests/test_depth_consumption.py` (17 comprehensive test cases)
**See**: `rl-assessment/order-simulation-fidelity-analysis.md` for full analysis

## Critical Priority (Immediate Impact on Profitability)

### 1. Add Spread-Aware Features to Observation Space ✅ (IMPLEMENTED & VERIFIED)
**Status**: ✅ IMPLEMENTED (Dec 16, 2024) - Observation space verified at 54 dimensions
**Impact**: CRITICAL - Model now aware of explicit trading costs
**Evidence**: 80% of markets have >20¢ spreads making them untradeable
**Completed**:
- ✅ Added direct spread cost features (spread_cents, spread_pct)
- ✅ Added spread regime classification (liquid/illiquid indicator)
- ✅ Added profitability threshold features (breakeven move required)
- ✅ Removed redundant features (8 features removed)
- ✅ Added transaction fee penalty in reward function (10% of spread)
- ✅ Verified observation dimension = 54 (28 market + 10 temporal + 11 portfolio + 5 order)
**Next Step**: Increase transaction fee to 50% of spread for conservative trading
**Expected Result**: 30-50% reduction in trading frequency once fee increased

### 2. Fix Excessive Trading Behavior (95%+ Activity) - NEW TOP PRIORITY
**Impact**: CRITICAL - Agent trades too frequently for paper trading
**Evidence**: Training shows 95.2% trading vs 3.7% HOLD despite ent_coef=0.07
**Root Causes**:
- Action space imbalance (20 trading vs 1 HOLD = 95.2% prior)
- Action 20 (100-contract SELL_NO) dominates with 6.62 avg reward
- Transaction fees (10% of spread) insufficient deterrent
**Solutions**:
1. Increase transaction fee to 50% of spread (0.5 multiplier)
2. Add HOLD stability bonus (+0.0001 reward)
3. Reduce entropy coefficient to 0.01
4. Consider action space redesign (add more HOLD-like actions)
**Expected Result**: Reduce trading to target 30-50% activity

### 3. Implement Proper Episode Boundaries
**Impact**: HIGH - Current setup may leak future information
**Evidence**: Session 9 has 47,851 timesteps but episodes don't respect market boundaries
**Solution**:
- Use MarketSessionView for clean episode boundaries per market
- Ensure each episode is one complete market session
- Reset order state between episodes properly
- Test with --curriculum mode which already does this correctly

### 4. Scale Training Duration
**Impact**: HIGH - 10K timesteps insufficient for learning
**Evidence**: Session 9 has 282 viable markets with 50+ timesteps each
**Solution**:
- Run minimum 100K timesteps for initial convergence
- Use full session data (47,851 timesteps) for comprehensive coverage
- Monitor learning curves for plateau detection

## High Priority (Direct Trading Performance)

### 5. Reward Function Enhancement
**Impact**: HIGH - Simple portfolio value change may be too sparse
**Evidence**: Large gaps between trades, weak learning signal
**Solution**:
- Add intermediate rewards for good order placement
- Reward spread capture explicitly
- Penalize excessive position concentration
- Include drawdown penalties

### 6. Hyperparameter Optimization
**Impact**: HIGH - Default PPO parameters not tuned for trading
**Evidence**: Default learning rate 3e-4 may be too high for financial data
**Solution**:
- Reduce learning rate to 1e-4 or 5e-5
- Increase batch size from 64 to 256 for more stable updates
- Tune GAE lambda for better credit assignment
- Use longer n_steps (8192) for better advantage estimation

### 7. Feature Engineering Improvements (UPDATED - See #1 for Spread Features)
**Impact**: MEDIUM-HIGH - Better features enable better decisions
**Evidence**: Current 52 features miss spread awareness and have redundancies
**Solution**:
- ✅ Add spread-aware features (see #1 and observation-space-review.md)
- Add order imbalance features at multiple price levels
- Include momentum indicators over different time windows
- Enhance temporal features with market microstructure signals

## Medium Priority (Training Efficiency)

### 8. Curriculum Learning Strategy
**Impact**: MEDIUM - Better market ordering improves learning
**Evidence**: 282 viable markets vary greatly in difficulty
**Solution**:
- Start with high-liquidity, low-volatility markets
- Progress to more challenging markets
- Use market characteristics for difficulty scoring
- Implement adaptive curriculum based on performance

### 9. Multi-Session Training
**Impact**: MEDIUM - More diverse data improves generalization
**Evidence**: Sessions 5-9 available, only using session 9
**Solution**:
- Combine multiple sessions for training
- Use session rotation for better coverage
- Test generalization across unseen sessions

### 10. Action Space Refinement ✅ (COMPLETED)
**Impact**: MEDIUM - Current 5-action space was limiting
**Evidence**: Fixed contract size (10) limited position sizing flexibility
**Solution**: ✅ IMPLEMENTED 21-action space with 5 position sizes
- ✅ Variable position sizing: [5, 10, 20, 50, 100] contracts
- ✅ 21 total actions: HOLD + 4 actions × 5 sizes
- Future: Consider aggressive vs passive pricing modes

## Lower Priority (Future Enhancements)

### 11. Model Architecture Updates
**Impact**: LOW-MEDIUM - MlpPolicy may be sufficient initially
**Evidence**: Default 2-layer MLP with 64 units each
**Solution**:
- Test deeper networks (3-4 layers)
- Add LSTM layers for temporal dependencies
- Consider attention mechanisms for market selection

### 12. Evaluation Metrics Enhancement
**Impact**: LOW - Better metrics for development iteration
**Evidence**: Current metrics focus on episode rewards
**Solution**:
- Add Sharpe ratio tracking
- Implement maximum drawdown monitoring
- Track win rate and profit factor
- Add per-market performance breakdown

### 13. Live Data Integration
**Impact**: LOW - Not needed until strategy profitable in simulation
**Evidence**: Currently using historical session data only
**Solution**:
- Prepare WebSocket integration for real-time data
- Implement online learning capabilities
- Add position synchronization with exchange

## Experimental Ideas

### 14. Ensemble Methods
- Train multiple agents with different hyperparameters
- Use voting or averaging for action selection
- Implement policy distillation from ensemble

### 15. Market Regime Detection
- Classify markets by volatility/liquidity regimes
- Train separate policies per regime
- Dynamic policy selection based on conditions

### 16. Risk Management Layer
- Add maximum position limits
- Implement stop-loss logic
- Portfolio-level risk constraints

---

## Recommended Next Experiment

**Experiment**: Add spread-aware features and fix HOLD-only behavior
**Session**: 32 or newer sessions with liquid markets
**Algorithm**: PPO with modified hyperparameters
**Changes**:
1. ✅ **PRIORITY 1**: Implement spread-aware features in feature_extractors.py
   - Add spread_cents, spread_pct, spread_regime features
   - Remove redundant features to maintain 52-feature space
2. Filter training data to markets with <10¢ spreads
3. Increase entropy coefficient to 0.1
4. Add exploration bonus to reward: +0.001 for non-HOLD actions
5. Reduce learning rate to 1e-4
6. Increase training to 100K timesteps
7. Use curriculum mode for clean episode boundaries

**Command** (after implementing spread features):
```bash
python train_sb3.py --session 32 --curriculum --algorithm ppo \
  --learning-rate 0.0001 --total-timesteps 100000 \
  --max-spread 10  # Filter for liquid markets only
```

Note: See `rl-assessment/observation-space-review.md` for detailed implementation guide.