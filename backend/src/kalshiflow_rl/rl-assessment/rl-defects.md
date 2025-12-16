# RL System Defects - Severity Ranked

Last Updated: 2025-12-16 07:00

## Critical Defects (Training Blockers)

### 1. Observation Dimension Mismatch - FIXED
**Severity**: ~~CRITICAL~~ RESOLVED
**Component**: MarketAgnosticKalshiEnv
**Description**: Environment expects 54 features and provides 54 (previously had mismatch)
**Status**: FIXED - Observation dimension correctly set to 54
**Verification**: Confirmed via test environment creation

### 2. Excessive Trading Activity (95%+) Despite Conservative Settings
**Severity**: CRITICAL
**Component**: Action Space Design / Reward Structure
**Description**: Agent trades 95%+ of time even with entropy coefficient 0.07
**Reproduction**:
```bash
python train_sb3.py --session 32 --algorithm ppo --total-timesteps 100000
# Observe action distribution: 95%+ trading actions, only 3-4% HOLD
```
**Impact**: Unrealistic trading frequency unsuitable for paper trading
**Root Cause**: 
- **Action space imbalance**: 20 trading actions vs 1 HOLD (95.2% prior for trading)
- **Large position rewards dominate**: Action 20 (100 contracts SELL_NO) has highest reward
- **Transaction fees too weak**: 10% of spread insufficient deterrent
**Suggested Fix**:
1. **Increase transaction fee penalty** in `unified_metrics.py`:
   ```python
   # Line 97: Increase from 0.1 to 0.5 (50% of spread)
   transaction_fee = 0.5 * spread_cents * (quantity / 10.0)
   ```
2. **Add HOLD reward bonus** in `market_agnostic_env.py`:
   ```python
   # After calculating reward
   if action == 0:  # HOLD action
       reward += 0.0001  # Small stability bonus
   ```
3. **Reduce entropy coefficient further**:
   ```python
   "ent_coef": 0.01,  # From 0.07 to 0.01 for less exploration
   ```

### 2. Episode Boundaries Not Respecting Market Sessions
**Severity**: HIGH
**Component**: SessionBasedEnvironment
**Description**: Episodes may span multiple markets, leaking future information
**Reproduction**:
```bash
python train_sb3.py --session 9 --algorithm ppo --total-timesteps 5000
# Check logs for market switches mid-episode
```
**Impact**: Agent learns from future data it shouldn't have access to
**Root Cause**: Episode length based on timesteps, not market boundaries
**Suggested Fix**:
- Already fixed in curriculum mode - use that as default
- Or modify SessionBasedEnvironment to force reset on market change

## High Severity (Performance Issues)

### 3. Transaction Fee Too Weak
**Severity**: HIGH
**Component**: UnifiedRewardCalculator in unified_metrics.py
**Description**: Transaction fee of 0.01 * spread (1% for standard trades) is unrealistically low
**Reproduction**:
```bash
# Check logs during training - fee penalties are negligible
python train_sb3.py --session 32 --algorithm ppo --total-timesteps 1000
```
**Impact**: Model learns unrealistic high-frequency trading behavior
**Root Cause**: Fee multiplier set too low (0.01 instead of realistic 0.1+)
**Suggested Fix**:
```python
# In unified_metrics.py line 95:
transaction_fee = 0.1 * spread_cents * (quantity / 10.0)  # 10x increase
```
**Analysis**: Real costs include spread + impact + fees (~10% minimum)

### 4. Insufficient Training Duration Default
**Severity**: HIGH
**Component**: train_sb3.py defaults
**Description**: Default training timesteps too low for convergence
**Reproduction**: Run training with default settings
**Impact**: Agent doesn't learn meaningful patterns
**Root Cause**: Conservative defaults for testing
**Suggested Fix**:
- Change default --total-timesteps from None to 100000
- Add warning if timesteps < 50000

### 4. Fixed Hyperparameters Not Tuned for Trading
**Severity**: HIGH
**Component**: get_default_model_params()
**Description**: PPO hyperparameters use generic defaults not optimized for trading
**Reproduction**: Compare training performance with different hyperparameters
**Impact**: Slow or failed convergence
**Root Cause**: Using Stable Baselines3 defaults
**Suggested Fix**:
```python
def get_default_model_params(algorithm: str) -> Dict[str, Any]:
    if algorithm.lower() == "ppo":
        return {
            "learning_rate": 1e-4,  # Was 3e-4
            "n_steps": 8192,  # Was 2048
            "batch_size": 256,  # Was 64
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.1,  # Was 0.01
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 1
        }
```

## Medium Severity (Feature Gaps)

### 5. No Order Imbalance Features
**Severity**: MEDIUM
**Component**: feature_extractors.py
**Description**: Missing orderbook imbalance signals crucial for price prediction
**Reproduction**: Check observation features - no imbalance metrics
**Impact**: Agent misses key microstructure signals
**Root Cause**: Not implemented in initial feature set
**Suggested Fix**:
```python
def extract_market_features():
    # Add after existing features
    total_bid_volume = sum(orderbook.yes_bids.values()) + sum(orderbook.no_bids.values())
    total_ask_volume = sum(orderbook.yes_asks.values()) + sum(orderbook.no_asks.values())
    imbalance = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume + 1e-8)
    features.append(np.clip(imbalance, -1, 1))
```

### 6. Fixed Contract Size Limitation
**Severity**: MEDIUM
**Component**: LimitOrderActionSpace
**Description**: Hardcoded 10 contract size limits position flexibility
**Reproduction**: Check order placement - always 10 contracts
**Impact**: Can't scale positions based on confidence
**Root Cause**: Simplification for initial implementation
**Suggested Fix**:
- Add position_size parameter to action space
- Or implement multiple size levels (5, 10, 20 contracts)

### 7. No Momentum Indicators
**Severity**: MEDIUM
**Component**: feature_extractors.py
**Description**: Missing price momentum features for trend detection
**Reproduction**: Check temporal features - no momentum metrics
**Impact**: Agent can't detect trends effectively
**Root Cause**: Not in initial feature design
**Suggested Fix**:
- Add rolling price changes over multiple windows
- Include RSI-like indicators

## Low Severity (Nice to Have)

### 8. M10 Diagnostics Not Enabled by Default
**Severity**: LOW
**Component**: train_sb3.py
**Description**: Diagnostic callbacks disabled by default
**Reproduction**: Run training without --disable-m10-diagnostics flag
**Impact**: Harder to debug training issues
**Root Cause**: Performance consideration
**Suggested Fix**: Enable by default, add --disable flag for production

### 9. No Learning Rate Scheduling
**Severity**: LOW
**Component**: Model creation in train_sb3.py
**Description**: Fixed learning rate throughout training
**Reproduction**: Check model parameters during training
**Impact**: Suboptimal convergence
**Root Cause**: Not implemented
**Suggested Fix**: Add learning rate scheduler callback

### 10. Missing Sharpe Ratio in Episode Metrics
**Severity**: LOW
**Component**: PortfolioMetricsCallback
**Description**: No Sharpe ratio calculation during training
**Reproduction**: Check episode statistics output
**Impact**: Harder to evaluate risk-adjusted returns
**Root Cause**: Not implemented in initial version
**Suggested Fix**: Add Sharpe calculation to episode statistics

## Known Issues (Won't Fix)

### 11. Async Warning in Action Execution
**Severity**: MINIMAL
**Component**: LimitOrderActionSpace
**Description**: "asyncio.run() cannot be called from running event loop" warning
**Reproduction**: Run any training
**Impact**: Cosmetic warning only, functionality works
**Root Cause**: Sync wrapper for async code
**Note**: Doesn't affect training, safe to ignore

---

## Priority Fix Order

1. **Fix observation dimension to 54** (IMMEDIATE - training blocker)
2. **Increase transaction fee to 0.1** (Quick fix, major impact on realism)
3. **Fix HOLD-only behavior** (Critical - prevents any learning)
4. **Increase default training duration** (Easy fix, high impact)
5. **Tune hyperparameters** (Simple config change, major improvement)
6. **Enable M10 diagnostics by default** (Help debugging)

## Testing After Fixes

After implementing fixes, validate with:
```bash
# Test exploration fix
python train_sb3.py --session 9 --curriculum --algorithm ppo \
  --learning-rate 0.0001 --total-timesteps 100000 \
  --m10-console-freq 100

# Check diagnostics for:
# - Action distribution (should show non-HOLD actions)
# - Reward signal strength
# - Portfolio value changes
# - Episode boundaries
```