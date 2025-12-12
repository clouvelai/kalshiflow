# RL Training Improvements - Priority Ranked

Last Updated: 2025-12-11 21:35

## Critical Priority (Immediate Impact on Profitability)

### 1. Fix HOLD-Only Behavior Pattern
**Impact**: CRITICAL - Agent stuck in HOLD prevents any trading
**Evidence**: Curriculum training on 473 markets shows agent defaulting to HOLD action
**Solution**: 
- Implement exploration bonuses in reward function
- Add entropy regularization to PPO (increase ent_coef from 0.01 to 0.1)
- Use action masking to force non-HOLD actions periodically
- Add minimum trading frequency requirement in reward

### 2. Implement Proper Episode Boundaries
**Impact**: HIGH - Current setup may leak future information
**Evidence**: Session 9 has 47,851 timesteps but episodes don't respect market boundaries
**Solution**:
- Use MarketSessionView for clean episode boundaries per market
- Ensure each episode is one complete market session
- Reset order state between episodes properly
- Test with --curriculum mode which already does this correctly

### 3. Scale Training Duration
**Impact**: HIGH - 10K timesteps insufficient for learning
**Evidence**: Session 9 has 282 viable markets with 50+ timesteps each
**Solution**:
- Run minimum 100K timesteps for initial convergence
- Use full session data (47,851 timesteps) for comprehensive coverage
- Monitor learning curves for plateau detection

## High Priority (Direct Trading Performance)

### 4. Reward Function Enhancement
**Impact**: HIGH - Simple portfolio value change may be too sparse
**Evidence**: Large gaps between trades, weak learning signal
**Solution**:
- Add intermediate rewards for good order placement
- Reward spread capture explicitly
- Penalize excessive position concentration
- Include drawdown penalties

### 5. Hyperparameter Optimization
**Impact**: HIGH - Default PPO parameters not tuned for trading
**Evidence**: Default learning rate 3e-4 may be too high for financial data
**Solution**:
- Reduce learning rate to 1e-4 or 5e-5
- Increase batch size from 64 to 256 for more stable updates
- Tune GAE lambda for better credit assignment
- Use longer n_steps (8192) for better advantage estimation

### 6. Feature Engineering Improvements
**Impact**: MEDIUM-HIGH - Better features enable better decisions
**Evidence**: Current 52 features may miss key patterns
**Solution**:
- Add order imbalance features
- Include momentum indicators
- Add spread regime classification
- Enhance temporal features with market microstructure signals

## Medium Priority (Training Efficiency)

### 7. Curriculum Learning Strategy
**Impact**: MEDIUM - Better market ordering improves learning
**Evidence**: 282 viable markets vary greatly in difficulty
**Solution**:
- Start with high-liquidity, low-volatility markets
- Progress to more challenging markets
- Use market characteristics for difficulty scoring
- Implement adaptive curriculum based on performance

### 8. Multi-Session Training
**Impact**: MEDIUM - More diverse data improves generalization
**Evidence**: Sessions 5-9 available, only using session 9
**Solution**:
- Combine multiple sessions for training
- Use session rotation for better coverage
- Test generalization across unseen sessions

### 9. Action Space Refinement
**Impact**: MEDIUM - Current 5-action space may be limiting
**Evidence**: Fixed contract size (10) limits position sizing flexibility
**Solution**:
- Add variable position sizing
- Implement aggressive vs passive pricing modes
- Consider market orders for high-confidence signals

## Lower Priority (Future Enhancements)

### 10. Model Architecture Updates
**Impact**: LOW-MEDIUM - MlpPolicy may be sufficient initially
**Evidence**: Default 2-layer MLP with 64 units each
**Solution**:
- Test deeper networks (3-4 layers)
- Add LSTM layers for temporal dependencies
- Consider attention mechanisms for market selection

### 11. Evaluation Metrics Enhancement
**Impact**: LOW - Better metrics for development iteration
**Evidence**: Current metrics focus on episode rewards
**Solution**:
- Add Sharpe ratio tracking
- Implement maximum drawdown monitoring
- Track win rate and profit factor
- Add per-market performance breakdown

### 12. Live Data Integration
**Impact**: LOW - Not needed until strategy profitable in simulation
**Evidence**: Currently using historical session data only
**Solution**:
- Prepare WebSocket integration for real-time data
- Implement online learning capabilities
- Add position synchronization with exchange

## Experimental Ideas

### 13. Ensemble Methods
- Train multiple agents with different hyperparameters
- Use voting or averaging for action selection
- Implement policy distillation from ensemble

### 14. Market Regime Detection
- Classify markets by volatility/liquidity regimes
- Train separate policies per regime
- Dynamic policy selection based on conditions

### 15. Risk Management Layer
- Add maximum position limits
- Implement stop-loss logic
- Portfolio-level risk constraints

---

## Recommended Next Experiment

**Experiment**: Fix HOLD-only behavior with enhanced exploration
**Session**: 9 (47,851 timesteps, 282 viable markets)
**Algorithm**: PPO with modified hyperparameters
**Changes**:
1. Increase entropy coefficient to 0.1
2. Add exploration bonus to reward: +0.001 for non-HOLD actions
3. Reduce learning rate to 1e-4
4. Increase training to 100K timesteps
5. Use curriculum mode for clean episode boundaries

**Command**:
```bash
python train_sb3.py --session 9 --curriculum --algorithm ppo \
  --learning-rate 0.0001 --total-timesteps 100000
```

Note: Entropy coefficient needs to be modified in get_default_model_params() first.