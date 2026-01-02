# Ship to Paper Trading - Quick Action Checklist

## Hour 1: Fix Critical Configs ⏰

### 1. Update Training Hyperparameters
**File**: `/backend/src/kalshiflow_rl/training/train_sb3.py`

```python
# In get_default_model_params():
"ent_coef": 0.1,        # Was 0.01 - CRITICAL 
"learning_rate": 1e-4,  # Was 3e-4
"batch_size": 256,      # Was 64
```

### 2. Add Exploration Bonus  
**File**: `/backend/src/kalshiflow_rl/environments/market_agnostic_env.py`

```python
# In step() method, after calculating base reward:
if action != 0:  # Non-HOLD action
    reward += 0.001  # Small exploration bonus
```

## Hour 2-4: Train Model 🚀

```bash
cd backend
uv run python src/kalshiflow_rl/training/train_sb3.py \
  --session 32 \
  --algorithm ppo \
  --total-timesteps 100000 \
  --learning-rate 0.0001
```

### Watch for Success Indicators:
- ✅ Non-HOLD actions >20% of time
- ✅ No NaN/inf errors
- ✅ Episodes completing
- ✅ Win rate >25%

## Day 2: Add 5 Critical Features (Optional)

If agent still stuck in HOLD, add these to feature extractor:

```python
# Price momentum
price_momentum_5m = (current_mid - mid_5min_ago) / mid_5min_ago

# Volume acceleration  
volume_acceleration = (volume_1m - volume_5m) / volume_5m

# Position context
position_duration = time_since_entry / 300
unrealized_pnl_pct = unrealized_pnl / position_value

# Spread regime
spread_percentile_1h = percentile_rank(current_spread, last_hour)
```

## Day 3: Paper Trading Wrapper

```python
class PaperTradingEnv(MarketAgnosticKalshiEnv):
    def __init__(self):
        super().__init__()
        self.ws_client = KalshiDemoClient()
        
    def get_current_orderbook(self):
        return self.ws_client.get_latest_orderbook()
```

## Day 4: Minimal Monitoring

Just log to console:
```
[14:32:10] Action: BUY_YES_AGGRESSIVE | Market: KXPRES-28 | Contracts: 10
[14:32:11] Fill: YES @ 45¢ | Position: +10 | Cash: $9,955
[14:32:25] Action: HOLD | Market: KXPRES-28
[14:32:40] Action: SELL_YES_PASSIVE | Market: KXPRES-28 | Contracts: 10  
[14:32:42] Fill: YES @ 48¢ | Position: 0 | P&L: +$30
```

## Day 5: Ship Criteria ✅

Ship if ALL true:
- [ ] Agent takes non-HOLD actions >20% of time
- [ ] Training completes 100K timesteps without crash
- [ ] Paper trading connects to demo WebSocket
- [ ] At least 1 order placed in test run
- [ ] No critical errors in 1-hour test

## Common Issues & Quick Fixes

### Agent Still Only HOLDs?
- Increase exploration bonus to 0.01
- Increase entropy to 0.2
- Add action masking to force trades every N steps

### Training Too Slow?
- Reduce timesteps to 50K for initial test
- Use smaller batch_size (128)
- Disable logging/callbacks

### Paper Trading Won't Connect?
- Check ENVIRONMENT=paper in .env
- Verify demo API credentials
- Test with curl first

## Success Metrics

### v1.0 (Ship Now):
- Any non-HOLD behavior
- No crashes

### v1.1 (Week 2):
- Win rate >30%
- Basic P&L tracking

### v2.0 (Month 2):
- Sharpe >0.5
- Consistent profits

---

**Remember**: Ship at 80%, iterate to 100%. Perfect is the enemy of good.

**The Goal**: Get ANYTHING trading in paper by end of week. We can improve from there.