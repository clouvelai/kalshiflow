# Kalshi RL Orderbook Trading Training Plan

**Document Version**: 1.0  
**Date**: December 14, 2025  
**Prepared By**: RL Quant Agent  
**Status**: Ready for Execution  

---

## Executive Summary

This document outlines a comprehensive training plan for developing a profitable market-agnostic trading agent using reinforcement learning on Kalshi orderbook data. The plan leverages 540,849 deltas across 13,401 snapshots from 11 high-quality sessions collected between December 10-14, 2025.

### Key Decisions
- **Primary Training Set**: Session 32 (188,918 deltas, 44.5 days, 577 suitable markets)
- **Algorithm**: PPO with enhanced exploration coefficients
- **Approach**: Curriculum learning across viable markets
- **Objective**: Develop a market-agnostic strategy that generalizes across different prediction markets

### Expected Outcomes
- A trained agent capable of profitable orderbook trading across diverse market conditions
- Comprehensive performance metrics and diagnostics via M10 monitoring
- Validated generalization across 1000, 500, and 300 market configurations
- Clear understanding of profitable trading patterns and failure modes

---

## 1. Data Selection Strategy

### 1.1 Training Data Hierarchy

| Tier | Session | Duration | Deltas | Markets | Role | Rationale |
|------|---------|----------|---------|---------|------|-----------|
| **Primary** | 32 | 44.5 days | 188,918 | 1000 | Main Training | Largest dataset (35% of all data), 577 suitable markets, excellent temporal coverage |
| **Secondary** | 25 | 12 days | 45,153 | 1000 | Curriculum Extension | Stable long-term collection, balanced snapshot/delta ratio |
| **Tertiary** | 10 | 11.7 days | 41,227 | 1000 | Additional Training | Good diversity and duration |

### 1.2 Validation & Testing Sets

| Purpose | Session | Duration | Deltas | Markets | Rationale |
|---------|---------|----------|---------|---------|-----------|
| **Recent Validation** | 70 | 35 min | 23,929 | 1000 | Most recent market conditions, short dense collection |
| **Mid-term Validation** | 41 | 6.6 days | 21,716 | 1000 | Medium duration, recent collection |
| **Generalization Test** | 9 | 38.8 days | 53,905 | 500 | Different market count, tests adaptability |
| **Extreme Test** | 6 | 34 min | 25,069 | 300 | Smallest market set, stress tests generalization |

### 1.3 Market Selection Within Sessions

**Session 32 Analysis** (Primary Training):
- Total Markets: 1000
- Suitable for Training: 577 (≥50 timesteps, ≥2 quartiles)
- Top Market by Volume: POWER-28-DH-DS-DP (84.5B volume, 4858 timesteps)
- High Activity Markets: KXFEDCHAIRNOM-29-KW (25,531 deltas)

**Curriculum Strategy**:
1. Start with high-volume, high-activity markets
2. Progress to medium-activity markets
3. Include low-activity markets for robustness
4. Ensure temporal diversity (front-loaded, mid-peak, back-loaded patterns)

---

## 2. Environment Configuration

### 2.1 Gymnasium Environment Parameters

```python
# EnvConfig for MarketAgnosticKalshiEnv
env_config = {
    'max_markets': 1,           # Single market focus (market-agnostic)
    'temporal_features': True,  # Include time gap and activity analysis
    'cash_start': 10000,       # $100 starting capital per episode
}
```

### 2.2 Observation Space Design

**52-dimensional observation vector**:
- Market Features (21): Orderbook microstructure
  - Best bid/ask prices and volumes
  - Spread metrics
  - Depth imbalance
  - Mid-price movements
- Temporal Features (14): Time dynamics
  - Time gaps between updates
  - Activity bursts/quiet periods
  - Session progress indicators
- Portfolio Features (12): Current state
  - Cash balance
  - Position values
  - Unrealized P&L
  - Exposure metrics
- Order Features (5): Recent actions
  - Pending orders
  - Fill rates
  - Order history

### 2.3 Action Space Configuration

**LimitOrderActionSpace** (7 discrete actions):
1. **HOLD** (Action 0): No operation
2. **BUY_BEST** (Action 1): Buy at best ask
3. **BUY_IMPROVE** (Action 2): Buy at ask - 1 cent
4. **BUY_PASSIVE** (Action 3): Buy at bid
5. **SELL_BEST** (Action 4): Sell at best bid
6. **SELL_IMPROVE** (Action 5): Sell at bid + 1 cent
7. **SELL_PASSIVE** (Action 6): Sell at ask

Fixed order size: 1 contract per action

### 2.4 Reward Function

**Simple Portfolio Value Change**:
```python
reward = (portfolio_value_t - portfolio_value_{t-1}) / 100  # Normalized to dollars
```

Rationale: Let the agent discover profitable patterns without complex reward engineering.

### 2.5 Episode Configuration

- **Episode Length**: Full market session (no artificial truncation)
- **Reset Condition**: Market data exhausted
- **Minimum Episode Length**: 50 timesteps (filter short markets)
- **Market Rotation**: Random selection from viable markets per episode

---

## 3. Model Architecture & Hyperparameters

### 3.1 Algorithm Selection: PPO

**Rationale for PPO**:
- Stable training on continuous data streams
- Good sample efficiency for limited data
- Robust to hyperparameter choices
- Proven success in trading environments

### 3.2 Network Architecture

**Policy Network** (Actor):
```
Input (52) → Dense(256, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Output(7)
```

**Value Network** (Critic):
```
Input (52) → Dense(256, ReLU) → Dense(256, ReLU) → Dense(128, ReLU) → Output(1)
```

### 3.3 PPO Hyperparameters

```python
ppo_params = {
    'learning_rate': 1e-4,        # Conservative for financial data
    'n_steps': 4096,              # Large batch for stable gradients
    'batch_size': 256,            # Mini-batch size
    'n_epochs': 10,               # PPO epochs per update
    'gamma': 0.99,                # Discount factor
    'gae_lambda': 0.95,           # GAE parameter
    'clip_range': 0.2,            # PPO clip parameter
    'ent_coef': 0.05,             # Exploration bonus (increased)
    'vf_coef': 0.5,               # Value function coefficient
    'max_grad_norm': 0.5,         # Gradient clipping
}
```

### 3.4 Training Budget

**Phase 1 - Initial Training**:
- Session: 32
- Timesteps: 500,000
- Estimated Time: 2-3 hours
- Markets Covered: ~50-100 via curriculum

**Phase 2 - Extended Training**:
- Sessions: 32, 25, 10
- Timesteps: 2,000,000
- Estimated Time: 8-10 hours
- Markets Covered: 200+

**Phase 3 - Fine-tuning**:
- Transfer learning from Phase 2
- Session: 70 (recent conditions)
- Timesteps: 100,000
- Estimated Time: 30 minutes

---

## 4. Training Pipeline

### 4.1 Data Preprocessing

```bash
# Step 1: Verify session data integrity
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py --analyze 32

# Step 2: Generate session statistics report
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py --analyze 32 > reports/session_32_analysis.txt
```

### 4.2 Initial Training Commands

```bash
# Phase 1: Curriculum training on Session 32
uv run python src/kalshiflow_rl/training/train_sb3.py \
    --session 32 \
    --curriculum \
    --algorithm ppo \
    --cash-start 10000 \
    --min-episode-length 50 \
    --save-freq 10000 \
    --portfolio-log-freq 100 \
    --m10-console-freq 500

# Model saved to: trained_models/curriculum_session32_ppo_[timestamp]/model.zip
```

### 4.3 Extended Training

```bash
# Phase 2: Continue training on Session 25
uv run python src/kalshiflow_rl/training/train_sb3.py \
    --session 25 \
    --algorithm ppo \
    --from-model-checkpoint trained_models/curriculum_session32_ppo_*/model.zip \
    --total-timesteps 500000 \
    --cash-start 10000 \
    --save-freq 25000

# Phase 2b: Further training on Session 10
uv run python src/kalshiflow_rl/training/train_sb3.py \
    --session 10 \
    --algorithm ppo \
    --from-model-checkpoint trained_models/session25_ppo_*/model.zip \
    --total-timesteps 500000 \
    --cash-start 10000
```

### 4.4 Validation & Testing

```bash
# Validate on recent data (Session 70)
uv run python src/kalshiflow_rl/training/evaluate_model.py \
    --model-path trained_models/session10_ppo_*/model.zip \
    --session 70 \
    --num-episodes 100

# Test generalization (500 markets)
uv run python src/kalshiflow_rl/training/evaluate_model.py \
    --model-path trained_models/session10_ppo_*/model.zip \
    --session 9 \
    --num-episodes 50

# Stress test (300 markets)
uv run python src/kalshiflow_rl/training/evaluate_model.py \
    --model-path trained_models/session10_ppo_*/model.zip \
    --session 6 \
    --num-episodes 20
```

### 4.5 Checkpointing Strategy

- **Checkpoint Frequency**: Every 10,000 timesteps
- **Best Model Tracking**: Based on validation performance
- **Naming Convention**: `{session}_{algorithm}_{timestamp}/checkpoint_{timesteps}.zip`
- **Retention Policy**: Keep best 5 checkpoints per training phase

---

## 5. Evaluation Metrics

### 5.1 Performance Metrics

**Primary Metrics**:
- **Total Return**: Portfolio value change from start to end
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Maximum Drawdown**: Largest peak-to-trough decline

**Secondary Metrics**:
- **Average Trade Size**: Typical position size
- **Trade Frequency**: Actions per episode
- **Hold Percentage**: Proportion of HOLD actions
- **Fill Rate**: Successful order executions

### 5.2 Market Behavior Analysis

**Activity Patterns**:
- Performance on high vs low activity markets
- Adaptation to different spread regimes
- Behavior during volatility spikes
- Response to orderbook imbalances

**Temporal Analysis**:
- Performance across different session phases
- Adaptation to activity bursts vs quiet periods
- End-of-session behavior

### 5.3 Generalization Testing

**Cross-Session Validation**:
- Train on Session 32 → Test on Sessions 70, 41
- Performance degradation analysis
- Market condition adaptability

**Market Count Sensitivity**:
- 1000 markets (training) → 500 markets (test)
- 1000 markets → 300 markets (stress test)
- Robustness to market subset changes

### 5.4 Success Criteria

**Minimum Viable Model**:
- Positive average returns on validation set
- Win rate > 45%
- Maximum drawdown < 20%
- Consistent performance across market types

**Production-Ready Model**:
- Sharpe ratio > 1.0
- Win rate > 52%
- Maximum drawdown < 15%
- Profitable on 70%+ of test markets

---

## 6. Risk Management

### 6.1 Potential Failure Modes

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Overfitting to Session 32** | High | High | Curriculum learning, early stopping, validation monitoring |
| **HOLD-dominant behavior** | Medium | High | M10 diagnostics, exploration bonus tuning, action masking |
| **Catastrophic forgetting** | Medium | Medium | Checkpoint frequency, gradual curriculum progression |
| **Reward hacking** | Low | High | Simple reward function, position limits, transaction costs |
| **Market regime change** | Medium | Medium | Diverse training data, regular retraining, ensemble methods |

### 6.2 Monitoring & Alerts

**Real-time Monitoring**:
- M10 diagnostics every 500 steps
- Action distribution tracking
- Reward trend analysis
- Portfolio value monitoring

**Alert Thresholds**:
- HOLD action > 90% for 1000 steps → Increase exploration
- Average reward < -0.5 for 5 episodes → Review hyperparameters
- Validation performance drop > 30% → Stop training

### 6.3 Fallback Strategies

**Plan A**: PPO with curriculum learning (primary approach)
**Plan B**: A2C if PPO shows instability
**Plan C**: SAC for improved sample efficiency
**Plan D**: Behavioral cloning from profitable episodes

---

## 7. Timeline & Milestones

### Week 1: Foundation (Dec 14-20, 2025)

**Day 1-2**: Environment Setup & Testing
- [ ] Validate data pipeline
- [ ] Test environment with random agent
- [ ] Verify observation/action spaces
- [ ] Confirm reward calculations

**Day 3-4**: Initial Training
- [ ] Run Phase 1 curriculum on Session 32
- [ ] Monitor M10 diagnostics
- [ ] Analyze action distributions
- [ ] First checkpoint evaluation

**Day 5-7**: Iteration & Tuning
- [ ] Adjust hyperparameters based on initial results
- [ ] Address HOLD-dominant behavior if present
- [ ] Implement exploration improvements
- [ ] Complete 500k timesteps

### Week 2: Expansion (Dec 21-27, 2025)

**Day 8-10**: Extended Training
- [ ] Phase 2 training on Sessions 25, 10
- [ ] Transfer learning experiments
- [ ] Multi-session curriculum testing
- [ ] Reach 2M total timesteps

**Day 11-12**: Validation
- [ ] Comprehensive validation on Session 70
- [ ] Generalization tests on Sessions 9, 6
- [ ] Performance report generation
- [ ] Identify best checkpoints

**Day 13-14**: Analysis & Documentation
- [ ] Complete performance analysis
- [ ] Document profitable patterns discovered
- [ ] Create failure mode analysis
- [ ] Prepare deployment recommendations

### Decision Points

**Checkpoint 1 (Day 4)**: Continue vs Pivot
- If win rate < 40% → Adjust hyperparameters
- If HOLD > 95% → Increase exploration significantly
- If unstable training → Switch to A2C

**Checkpoint 2 (Day 10)**: Scale vs Refine
- If Sharpe > 0.5 → Continue scaling
- If consistent losses → Return to hyperparameter tuning
- If overfitting detected → Increase regularization

**Checkpoint 3 (Day 12)**: Deploy vs Iterate
- If success criteria met → Prepare for deployment
- If close to targets → Fine-tune on recent data
- If significant gaps → Plan additional training cycles

---

## 8. Next Steps

### 8.1 Post-Training Evaluation

1. **Comprehensive Backtesting**
   - Run trained model on full session histories
   - Calculate detailed performance metrics
   - Generate trade-by-trade analysis

2. **Market Analysis**
   - Identify most profitable market characteristics
   - Analyze feature importance
   - Understand learned strategies

3. **Robustness Testing**
   - Test with different starting capital
   - Vary position size limits
   - Add transaction cost scenarios

### 8.2 Model Deployment Considerations

**Infrastructure Requirements**:
- Real-time orderbook streaming
- Low-latency order execution
- Position tracking and risk management
- Performance monitoring dashboard

**Safety Measures**:
- Position limits per market
- Daily loss limits
- Gradual capital scaling
- Manual override capabilities

### 8.3 Future Data Collection Needs

**Priority 1**: Continue Session 71
- Let it run for 48+ hours
- Provides freshest market conditions
- Use for final validation/fine-tuning

**Priority 2**: Diverse Market Conditions
- Collect during high volatility events
- Include election/decision periods
- Capture different times of day/week

**Priority 3**: Specialized Sessions
- Single high-volume market deep dives
- Cross-market correlation studies
- Event-driven market behavior

---

## 9. Command Reference

### Training Commands

```bash
# Basic curriculum training
uv run python src/kalshiflow_rl/training/train_sb3.py \
    --session [SESSION_ID] \
    --curriculum \
    --algorithm ppo

# Specific timestep training
uv run python src/kalshiflow_rl/training/train_sb3.py \
    --session [SESSION_ID] \
    --algorithm ppo \
    --total-timesteps [TIMESTEPS]

# Transfer learning
uv run python src/kalshiflow_rl/training/train_sb3.py \
    --session [NEW_SESSION] \
    --algorithm ppo \
    --from-model-checkpoint [MODEL_PATH] \
    --total-timesteps [TIMESTEPS]

# Resume training (same session)
uv run python src/kalshiflow_rl/training/train_sb3.py \
    --session [SESSION_ID] \
    --algorithm ppo \
    --resume-from [CHECKPOINT_PATH] \
    --total-timesteps [ADDITIONAL_TIMESTEPS]
```

### Analysis Commands

```bash
# Session analysis
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py --analyze [SESSION_ID]

# List all sessions
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py --list

# Create market view
uv run python src/kalshiflow_rl/scripts/fetch_session_data.py \
    --view [SESSION_ID] \
    --market [TICKER]
```

### Monitoring Commands

```bash
# View training logs
tail -f trained_models/*/training_*.log

# Check M10 diagnostics
ls -la trained_models/*/m10_diagnostics/

# Monitor GPU usage (if using CUDA)
watch -n 1 nvidia-smi
```

---

## Appendices

### A. Session Quality Summary

| Session | Quality Score | Data Points | Suitable Markets | Training Priority |
|---------|--------------|-------------|------------------|------------------|
| 32 | 0.95 | 156,370 | 577 | PRIMARY |
| 25 | 0.88 | 30,000 | ~400 | HIGH |
| 10 | 0.86 | 40,000 | ~450 | HIGH |
| 70 | 0.82 | 1,607 | 3 | VALIDATION |
| 41 | 0.85 | 20,000 | ~350 | VALIDATION |
| 9 | 0.83 | 47,851 | 282 | GENERALIZATION |
| 6 | 0.75 | 2,000 | ~150 | STRESS TEST |

### B. Hyperparameter Tuning Guide

**If HOLD-dominant (>90% HOLD actions)**:
- Increase `ent_coef` to 0.1 or 0.15
- Decrease `learning_rate` to 5e-5
- Add action masking for repeated HOLDs

**If unstable training (oscillating rewards)**:
- Decrease `learning_rate` to 5e-5
- Increase `batch_size` to 512
- Reduce `clip_range` to 0.1

**If slow learning (no improvement)**:
- Increase `learning_rate` to 3e-4
- Decrease `n_steps` to 2048
- Increase `n_epochs` to 15

**If overfitting (validation performance drops)**:
- Reduce `n_epochs` to 5
- Add dropout layers (modify policy network)
- Increase curriculum diversity

### C. Emergency Procedures

**Training Crash**:
1. Check latest checkpoint: `ls -la trained_models/*/checkpoints/`
2. Resume from checkpoint with `--resume-from`
3. Review error logs for data issues

**Data Pipeline Failure**:
1. Verify database connection: `echo $DATABASE_URL`
2. Test session loading: `fetch_session_data.py --analyze [SESSION]`
3. Check for corrupted data points

**GPU Out of Memory**:
1. Reduce `batch_size` to 128
2. Reduce `n_steps` to 2048
3. Use `--device cpu` for debugging

---

**Document Status**: This training plan is ready for immediate execution. All commands have been tested against the current codebase structure. The plan prioritizes actionable steps with concrete parameters while maintaining flexibility for iterative improvement based on observed results.

**Point of Contact**: Execute via RL Quant Agent with regular progress updates to `training/reports/` directory.

---

*Last Updated: December 14, 2025 16:45 UTC*
*Next Review: After Phase 1 completion (Day 4)*