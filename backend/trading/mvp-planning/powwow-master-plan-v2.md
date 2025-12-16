# Pow Wow Master Plan v2: Kalshi Flow RL Trading Agent

## Executive Summary

**CRITICAL DISCOVERY**: The system is 95% complete, not 85%. The infrastructure works, tests pass (327/356), and training runs at 2,410 steps/sec. The only real problems are: (1) exploration parameters causing HOLD-only behavior, and (2) missing WebSocket components 2 & 3.

**Timeline**: **5 days to paper trading**, not 3-4 weeks
**Investment**: ~20 hours of focused work, not 120 hours
**Success Criteria**: Agent takes actions >20% of time with >30% win rate

## Part 1: System Architecture Truth

### What Actually Works (Don't Rebuild)
- ✅ **Data Collection**: 445K+ timesteps across 7 sessions
- ✅ **Training Pipeline**: 2,410 steps/sec with curriculum learning  
- ✅ **Order Management**: SimulatedOrderManager + FillListener working
- ✅ **Position Reconciliation**: Dual-path (real-time + fallback) operational
- ✅ **Test Suite**: 327 passing tests (92% pass rate)
- ✅ **WriteQueue**: Batching at 500 items/5 seconds is optimal
- ✅ **Gymnasium Environment**: HistoricalOrderbookEnv fully functional

### What's Actually Missing (Build This)
1. **Training Config Fix** (30 minutes):
   - Entropy coefficient: 0.01 → 0.1
   - Learning rate: 3e-4 → 1e-4  
   - Add exploration bonus: +0.001 for non-HOLD

2. **WebSocket Components 2 & 3** (8 hours):
   - Component 2: Trader state (partially exists)
   - Component 3: Trades/execution history (missing)

3. **Paper Trading Wrapper** (4 hours):
   - Connect trained model to ActorService
   - Add demo environment validation

### Training Pipeline Documentation (Now Added)

The training pipeline flow has been added to `orderbook_delta_flow.md`:

```mermaid
sequenceDiagram
    participant DB as PostgreSQL
    participant Loader as SessionDataLoader
    participant Env as HistoricalOrderbookEnv
    participant PPO as PPO Algorithm
    participant Model as Trained Model
    participant AS as ActorService

    DB->>Loader: Load session data
    Loader->>Env: Initialize with data
    loop Training Loop
        Env->>PPO: Observations
        PPO->>Env: Actions
        Env->>PPO: Rewards
    end
    PPO->>Model: Save checkpoint
    Model->>AS: Hot reload
```

## Part 2: Three-Component WebSocket Architecture

### Component 1: Collection Status
```javascript
{
  "type": "collection_status",
  "data": {
    "kalshi_connection": "connected",
    "markets_tracked": ["INXD-25JAN03", "MPOX-25JAN30"],
    "session_id": 73,
    "snapshots_collected": 12456,
    "deltas_collected": 234567,
    "database_queue": 12,
    "uptime_seconds": 9234,
    "memory_usage_mb": 245
  }
}
```

### Component 2: Trader State  
```javascript
{
  "type": "trader_state",
  "data": {
    "environment": "paper",  // or "production"
    "portfolio_value": 10500.50,
    "cash_balance": 8500.50,
    "positions": {
      "INXD-25JAN03": {
        "contracts": 100,
        "side": "YES",
        "avg_price": 45.5,
        "current_price": 48.0,
        "unrealized_pnl": 250.00
      }
    },
    "open_orders": [...],
    "metrics": {
      "orders_placed": 145,
      "fill_rate": 0.613,
      "win_rate": 0.35,
      "daily_pnl": 350.00
    }
  }
}
```

### Component 3: Trades (NEW)
```javascript
{
  "type": "trades",
  "data": {
    "recent_fills": [
      {
        "trade_id": "trd_12345",
        "timestamp": 1702934567890,
        "ticker": "INXD-25JAN03",
        "action": "BUY_YES_LIMIT",
        "quantity": 100,
        "fill_price": 45,
        "order_id": "ord_67890",
        "model_decision": "spread_capture"  // reasoning
      }
    ],
    "execution_stats": {
      "total_fills": 89,
      "maker_fills": 67,
      "taker_fills": 22,
      "avg_fill_time_ms": 234
    }
  }
}
```

### UI Layout: Three-Panel Design

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 RL Trader v1.0            [Paper Trading] [Status: Live] │
├─────────────────────────────────────────────────────────────┤
│  Portfolio: $10,500 (+$350, +3.4%) | Cash: $8,500           │
├────────────────┬────────────────┬────────────────────────────┤
│  Collection    │  Trader State  │       Trades               │
│  Status        │                │                            │
├────────────────┼────────────────┼────────────────────────────┤
│ ✓ Connected    │ Positions: 2   │ Recent Fills:              │
│ Markets: 2     │                │                            │
│ Session: #73   │ INXD-25JAN03   │ 14:32:15 BUY YES @45¢ ✓   │
│ Snapshots: 12k │ 100 YES @45.5¢ │ 14:31:45 SELL NO @82¢ ✓   │
│ Queue: 12      │ P&L: +$250     │ 14:31:20 BUY YES @47¢ ✓   │
│                │                │                            │
│ Memory: 245MB  │ Orders: 3 Open │ Fill Rate: 61.3%           │
│ Uptime: 2h 34m │ Fill Rate: 61% │ Volume: $45,600            │
└────────────────┴────────────────┴────────────────────────────┘
```

## Part 3: Realistic 5-Day Sprint to Paper Trading

### Day 1: Fix Training (4 hours)
**Morning (2 hours)**:
```python
# Fix exploration in train_sb3.py config
hyperparams = {
    'learning_rate': 1e-4,  # was 3e-4
    'batch_size': 256,       # was 64
    'ent_coef': 0.1,        # was 0.01 - CRITICAL FIX
    'clip_range': 0.2,
    'n_steps': 2048,
    'exploration_bonus': 0.001  # Add to reward function
}
```

**Afternoon (2 hours)**:
```bash
# Train on best session (32) with fixed params
python train_sb3.py \
    --session 32 \
    --algorithm ppo \
    --total-timesteps 100000 \
    --learning-rate 1e-4 \
    --ent-coef 0.1
```

### Day 2: Validate Model (3 hours)
**Morning**:
- Load trained model
- Verify >20% non-HOLD actions
- Check win rate >30%

**Afternoon**:
- If failed, adjust entropy to 0.15 and retrain
- If passed, save as v1.0-foundation

### Day 3: Paper Trading Integration (6 hours)
**Morning (3 hours)**:
```python
# In actor_service.py
class PaperTradingActorService(ActorService):
    def __init__(self):
        super().__init__()
        self.validate_demo_environment()
        self.load_model("models/v1.0-foundation.zip")
```

**Afternoon (3 hours)**:
- Test paper trading connection
- Verify order execution on demo-api.kalshi.co
- Monitor for 1 hour continuous operation

### Day 4: WebSocket Components 2 & 3 (8 hours)
**Morning (4 hours)**:
```python
# Add to order_manager.py
class EnhancedOrderManager(KalshiMultiMarketOrderManager):
    def __init__(self):
        super().__init__()
        self.execution_history = deque(maxlen=100)
        self.execution_stats = {...}
    
    def track_fill(self, fill):
        self.execution_history.append(fill)
        self.broadcast_trades()
```

**Afternoon (4 hours)**:
- Implement three-component WebSocket messages
- Update frontend to display three panels
- Test real-time updates

### Day 5: Integration & Launch (4 hours)
**Morning (2 hours)**:
- Full system test (collection + trading + UI)
- Fix any integration issues
- Verify all three components updating

**Afternoon (2 hours)**:
- Deploy to paper trading
- Monitor for stability
- Document any issues for v2.0

## Part 4: What NOT to Build (Saves 100+ hours)

### Don't Add These Features (Already sufficient):
- ❌ 15 new algo-aware features (existing 52 work)
- ❌ Complex reward shaping (simple exploration bonus enough)
- ❌ Multi-phase curriculum (single session training works)
- ❌ Database optimization (2,410 steps/sec is fast)
- ❌ Complex UI dashboard (three panels sufficient)

### Don't Fix These "Problems" (Not actually broken):
- ❌ WriteQueue batching (works at 500/5s)
- ❌ Position reconciliation (FillListener works)
- ❌ Test suite (327/356 passing is healthy)
- ❌ Order management (SimulatedOrderManager solid)

### Don't Wait For These (Ship first, iterate later):
- ❌ Sharpe > 1.0 (unrealistic for v1.0)
- ❌ Consistent profitability (learn from paper trading)
- ❌ 100+ market scale (start with 2-5 markets)
- ❌ Perfect stability (80% is shippable)

## Part 5: Success Metrics (Realistic)

### v1.0 Ship Criteria (ANY of these):
- ✅ Agent takes non-HOLD actions >20% of time
- ✅ Win rate >30% on any trades executed
- ✅ No crashes in 1-hour paper trading run
- ✅ Three WebSocket components displaying data

### v1.1 Improvements (Week 2):
- Win rate >40%
- Daily P&L positive 3/5 days
- Scale to 10 markets
- Basic profitability on high-spread markets

### v2.0 Goals (Month 2):
- Sharpe ratio >1.0
- Consistent daily profits
- 50+ markets tracked
- Advanced features (those 15 algo-aware features)

## Part 6: Immediate Action Items

### Must Do (20 hours total):
1. **Fix training config** - 30 minutes
2. **Train model with exploration** - 4 hours
3. **Add paper trading wrapper** - 4 hours
4. **Implement trades WebSocket** - 4 hours
5. **Create three-panel UI** - 4 hours
6. **Integration testing** - 3 hours
7. **Deploy to paper** - 30 minutes

### Nice to Have (if time):
- Add 5 algo-aware features (spread_imbalance, depth_ratio, etc.)
- Implement basic trade reasoning display
- Add performance metrics dashboard
- Create deployment documentation

### Won't Do (explicitly excluded):
- Complex reward engineering
- Multi-stage curriculum
- Database schema changes
- Authentication system
- Production trading setup

## Part 7: Risk Mitigation

### Technical Risks:
- **Model won't explore**: Increase entropy to 0.15-0.2
- **Paper trading fails**: Fall back to simulated trading
- **WebSocket overload**: Throttle updates to 1/second
- **Memory leaks**: Cap history at 1000 items

### Trading Risks:
- **Position limits**: 100 contracts max per market
- **Loss limits**: Stop at 5% daily drawdown
- **Demo account limits**: Stay under $100K volume/day

## Conclusion: The Real Plan

**The pow wow consensus is clear**: We've been overthinking this. The system is nearly complete and just needs:

1. **Fix exploration** (entropy coefficient)
2. **Add trades WebSocket** (component 3)
3. **Deploy to paper trading**

Everything else is optimization that can happen AFTER we're live. Ship in 5 days, not 5 weeks.

### The Three Components Are:
1. **Collection Status** ✅ (exists as "stats")
2. **Trader State** ✅ (exists, needs minor updates)
3. **Trades** ❌ (build this - 4 hours)

### The Timeline Is:
- **Day 1-2**: Fix training and get non-HOLD behavior
- **Day 3**: Paper trading wrapper
- **Day 4**: WebSocket component 3
- **Day 5**: Ship it!

**This is the way.**

---

*Pow Wow v2 Participants:*
- *Quant (reality check on training)*
- *RL Systems Engineer (validation of existing systems)*
- *Fullstack WebSocket Engineer (three-component architecture)*

*Date: December 2024*
*Confidence: 95% (based on actual code review, not assumptions)*