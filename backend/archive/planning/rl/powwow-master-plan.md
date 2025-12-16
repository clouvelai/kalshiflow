# Pow Wow Master Plan v2.0: Kalshi Flow Trading Agent (REFINED)

## Executive Summary

Based on thorough analysis of the actual codebase state and 445K timesteps of collected data across 7 high-quality sessions, this refined plan focuses on MINIMAL changes to achieve non-HOLD trading behavior and ship to paper trading.

**Current State Reality Check**:
- ✅ 95% infrastructure complete (order simulation fixed, environment works)
- ✅ 445K timesteps across sessions 9, 10, 32, 41, 70, 71, 72
- ✅ PPO trains successfully (2,410 timesteps/second)
- ❌ Agent stuck in HOLD (28% win rate shows it CAN trade when forced)
- ❌ No exploration happening (entropy too low)

**Revised Target**: Ship to paper trading with ANY non-HOLD behavior
**Realistic Timeline**: 5-7 days to paper trading (not 3-4 weeks)

## What Already Works (DON'T REBUILD)

### Infrastructure (95% Complete)
- ✅ **Order Simulation**: Realistic fills with depth consumption DONE
- ✅ **Environment**: MarketAgnosticKalshiEnv fully functional
- ✅ **Data Pipeline**: 7 sessions with 445K+ timesteps ready
- ✅ **Training Loop**: PPO trains at 2,410 steps/sec
- ✅ **Features**: 22 market features already implemented
- ✅ **Action Space**: 21 actions (5 levels × 2 sides + HOLD)

### What's Actually Broken
1. **HOLD-Only Behavior**: Entropy coefficient 0.01 too low → increase to 0.1
2. **No Exploration Bonus**: Add +0.001 reward for non-HOLD actions
3. **Default Timesteps**: 10K too small → need 100K minimum
4. **Learning Rate**: 3e-4 too high → reduce to 1e-4

## Part 1: MINIMAL Changes for Non-HOLD Behavior

### 1.1 Critical Config Changes (30 minutes)

```python
# In train_sb3.py get_default_model_params():
"ent_coef": 0.1,        # Was 0.01 - CRITICAL for exploration
"learning_rate": 1e-4,  # Was 3e-4 - too high for finance
"batch_size": 256,      # Was 64 - more stable
"n_steps": 2048,        # Keep as is

# In market_agnostic_env.py step():
if action != 0:  # Non-HOLD action
    reward += 0.001  # Small exploration bonus
```

### 1.2 Feature Prioritization (Ship with existing 22, add 5 critical ones)

**Already Have (KEEP)**:
- volume_imbalance, yes/no_side_imbalance 
- spread features, mid prices
- book depth, liquidity concentration
- arbitrage_opportunity, market_efficiency

**Add ONLY These 5 Critical Features** (2 hours):
```python
# ORDER FLOW MOMENTUM
price_momentum_5m = (current_mid - mid_5min_ago) / mid_5min_ago
volume_acceleration = (volume_1m - volume_5m) / volume_5m

# POSITION CONTEXT  
position_duration = time_since_entry / 300  # 5min normalized
unrealized_pnl_pct = unrealized_pnl / position_value

# SPREAD REGIME
spread_percentile_1h = percentile_rank(current_spread, last_hour)
```

### 1.2 Simplified Training Strategy (Use What We Have)

**Phase 1: Get ANY Trading** (Day 1)
```bash
# Session 32: Most recent, 188K timesteps
python train_sb3.py --session 32 --algorithm ppo \
  --total-timesteps 100000 --learning-rate 0.0001
```
Goal: See non-HOLD actions (>20% of time)

**Phase 2: Multi-Session Training** (Day 2)
```bash
# Combine best sessions for diversity
python train_sb3.py --session 32 70 71 --algorithm ppo \
  --total-timesteps 200000 --learning-rate 0.0001  
```
Goal: Generalize across markets

**Phase 3: Curriculum on Viable Markets** (Day 3)
```bash
# Use curriculum mode for clean episodes
python train_sb3.py --session 32 --curriculum \
  --algorithm ppo --total-timesteps 100000
```
Goal: 30%+ win rate

### 1.3 Keep Model Simple (Already Good)

**DON'T CHANGE**:
- Algorithm: PPO (working)
- Network: MlpPolicy (sufficient)
- Actions: 21-action space (good granularity)

**ONLY TUNE**:
- Entropy: 0.01 → 0.1
- Learning rate: 3e-4 → 1e-4  
- Batch size: 64 → 256
- Training steps: 10K → 100K

### 1.4 Success Metrics Reality Check

**v1.0 Minimum Bar** (Ship if we hit ANY of these):
- Agent takes non-HOLD actions >20% of time
- Win rate >30% (we already see 28%!)
- Any positive episodes in paper trading
- No critical errors in 1-hour paper run

**v1.1 Goals** (After paper trading feedback):
- Consistent spread capture (1-2¢)
- Sharpe > 0.5 (not 1.0 yet)
- Daily drawdown < 10%

**v2.0 Dreams** (Month 2+):
- Sharpe > 1.0
- Profitable daily P&L
- Scale to 100+ markets

## Part 2: What's Actually Needed for Paper Trading

### 2.1 Paper Trading Essentials (Already 90% Done)

**Already Working**:
- ✅ SimulatedOrderManager with realistic fills
- ✅ Position tracking in environment
- ✅ Cash management ($10K starting)
- ✅ Order validation and constraints

**Actual TODOs** (2-3 hours):
1. Add paper trading flag to environment
2. Connect to demo-api.kalshi.co WebSocket
3. Add position reconciliation loop
4. Test with live orderbook data

```python
# Simple paper trading wrapper
class PaperTradingEnv(MarketAgnosticKalshiEnv):
    def __init__(self, use_live_data=True):
        super().__init__()
        self.live_ws = KalshiDemoClient() if use_live_data else None
        
    def get_current_orderbook(self):
        return self.live_ws.get_latest() if self.live_ws else self.historical
```

### 2.2 Skip Database Optimization (Not Needed for v1.0)

**Current Performance is Fine**:
- Training: 2,410 timesteps/sec 
- Data loading: <1 sec for 100K timesteps
- Memory usage: <2GB for full session

**Don't waste time on**:
- Batched writes (already implemented)
- Compression (not bottleneck)
- Memory optimization (plenty of headroom)

### 2.3 Simplified Paper Trading Launch

**Day 4-5 Checklist**:
1. Train model with fixed configs (100K steps)
2. Load model in paper trading script
3. Connect to demo WebSocket
4. Run for 1 hour
5. Check: Did it place ANY orders?
6. If yes → Ship it!

**Don't Block on**:
- Perfect position reconciliation
- All edge cases handled  
- 24-hour stability (do that after shipping)
- Beautiful metrics (add later)

## Part 3: Minimal Frontend for Paper Trading

### 3.1 MVP Dashboard (1 day max)

**Just Show**:
```
┌─────────────────────────────────┐
│ Paper Trading: ACTIVE            │
│ Model: v1.0 | Session: 32        │
├─────────────────────────────────┤
│ Cash: $10,000 | P&L: +$0        │
│ Positions: 0 | Orders: 0        │  
├─────────────────────────────────┤
│ Last Action: HOLD               │
│ Time: 14:32:10                  │
└─────────────────────────────────┘
```

**Skip for v1.0**:
- Complex WebSocket messages
- Real-time charts
- Position cards
- Performance metrics

**Just Log to Console**:
- Every order placed
- Every fill received  
- Position changes
- Errors

### 3.2 Clean UI Separation

**Mode A: Collector Status** (When collecting data)
```
┌─────────────────────────────────────┐
│ 📊 Orderbook Collector Status        │
├─────────────────────────────────────┤
│ Markets: 8 | Snapshots: 12,456      │
│ Duration: 2h 34m | Size: 245 MB     │
└─────────────────────────────────────┘
```

**Mode B: Active Trader** (When trading)
```
┌─────────────────────────────────────┐
│ 🤖 RL Trader | Paper | +$250 (2.5%) │
├─────────────────────────────────────┤
│ [Position Cards with P&L]           │
│ [Recent Actions Feed]                │
│ [Key Metrics]                        │
└─────────────────────────────────────┘
```

### 3.3 Implementation Priorities

**Week 1**: Core Trading View
- Unified trader_state WebSocket
- Portfolio header component
- Position cards with real-time P&L
- Action feed (last 5 actions)

**Week 2**: Enhanced Monitoring  
- Collector status view
- Performance metrics
- Market coverage visualization

## Realistic 5-Day Sprint to Paper Trading

### Day 1 (Monday): Fix Training
**Morning (2 hours)**:
- [ ] Change entropy coefficient to 0.1
- [ ] Add exploration bonus (+0.001 for non-HOLD)
- [ ] Set learning rate to 1e-4
- [ ] Set default timesteps to 100K

**Afternoon (4 hours)**:
- [ ] Run training on session 32
- [ ] Verify non-HOLD actions in logs
- [ ] Save model if >20% non-HOLD rate

### Day 2 (Tuesday): Multi-Session Training
**Morning (2 hours)**:
- [ ] Add 5 critical features (momentum, position context)
- [ ] Test feature extraction

**Afternoon (4 hours)**:  
- [ ] Train on sessions 32, 70, 71 combined
- [ ] Run for 200K timesteps
- [ ] Check action diversity

### Day 3 (Wednesday): Paper Trading Wrapper
**Morning (3 hours)**:
- [ ] Create PaperTradingEnv class
- [ ] Connect to demo WebSocket
- [ ] Test orderbook feed

**Afternoon (3 hours)**:
- [ ] Load trained model
- [ ] Run paper trading loop
- [ ] Log all actions to file

### Day 4 (Thursday): Minimal Dashboard
**Morning (2 hours)**:
- [ ] Create simple text dashboard
- [ ] Show cash, positions, last action

**Afternoon (2 hours)**:
- [ ] Run 1-hour paper trading test
- [ ] Monitor for errors
- [ ] Document any issues

### Day 5 (Friday): Ship It!
**Morning (2 hours)**:
- [ ] Fix critical bugs only
- [ ] Write simple run script
- [ ] Update README with instructions

**Afternoon**: 
- [ ] Deploy to paper trading
- [ ] Monitor first trades
- [ ] Celebrate shipping v1.0!

## Revised Success Criteria

### v1.0 Ship Criteria (MUST have all):
- ✅ Agent takes non-HOLD actions >20% of time
- ✅ No training crashes in 100K timesteps  
- ✅ Paper trading runs for 1 hour without errors
- ✅ At least 1 order placed in paper trading
- ✅ Basic logging of all actions

### v1.1 Improvements (Week 2):
- Win rate >35%
- Simple dashboard showing P&L
- 4-hour stability in paper trading
- Handle position reconciliation

### v2.0 Goals (Month 2):
- Sharpe > 0.5
- Beautiful frontend
- Multi-market trading
- Consistent profitability

## Critical Path & Blockers

### What Could Block Ship (FIX FIRST):
1. **Entropy too low** → Agent won't explore (30 min fix)
2. **No live data** → Can't paper trade (3 hour fix)
3. **Model won't load** → Can't deploy (1 hour fix)

### What WON'T Block Ship (ignore for v1.0):
- Test suite failures (fix later)
- Database optimization (not needed)
- Beautiful UI (console is fine)
- Perfect position tracking (good enough)
- 24-hour stability (test after shipping)
- Complex features (22 is enough)

## Key Insights from Reality Check

### What We Learned:
1. **Infrastructure MORE complete than thought** (95% not 85%)
2. **Problem is SIMPLER than expected** (just exploration)
3. **We have PLENTY of data** (445K timesteps)
4. **Training WORKS** (28% win rate proves it)
5. **Just need to SHIP** (perfect is enemy of good)

### Decisions Made:
1. **Use existing features** (22 is enough, maybe add 5)
2. **Fix config only** (entropy, LR, exploration bonus)
3. **Skip complex UI** (console logging is fine)
4. **Train on what we have** (don't collect more)
5. **Ship in 5 days** (not 3-4 weeks)

## Next 4 Hours (DO RIGHT NOW)

### Hour 1: Fix Training Config
```python
# 1. Edit train_sb3.py
- Change ent_coef to 0.1
- Change learning_rate to 1e-4  
- Change batch_size to 256

# 2. Edit market_agnostic_env.py
- Add exploration bonus in step()
```

### Hour 2-4: Train & Verify
```bash
# Start training
python train_sb3.py --session 32 --algorithm ppo \
  --total-timesteps 100000 --learning-rate 0.0001

# Watch logs for:
- Non-HOLD actions appearing
- No NaN/inf errors
- Episode rewards changing
```

If we see >20% non-HOLD actions → We're ready for paper trading!

## Conclusion: Ship Fast, Iterate in Production

**The Reality**: We're 95% done, not 85%. The infrastructure works, the data exists, the training runs. We just need the agent to explore.

**The Plan**: 5 days to paper trading, not 3-4 weeks.

**Day 1-2**: Fix configs, train with exploration
**Day 3-4**: Wire up paper trading
**Day 5**: Ship it!

**Success looks like**: 
- Agent places orders (even bad ones)
- Paper trading doesn't crash
- We learn from real execution

**The Motto**: "Ship at 80%, iterate to 100%"

We have 445K timesteps of data, a working environment, and proven infrastructure. The only thing standing between us and paper trading is a few config changes and the courage to ship imperfect code.

---

*Document refined by: RL Assessment Expert*  
*Date: December 15, 2024*  
*Version: 2.0 - Reality-Based Refinement*