# M10: RL System Instrumentation & Diagnostics Plan

## Executive Summary

The RL agent trained on Kalshi orderbook data exhibits a critical behavior: it never executes trades, always choosing HOLD actions. This plan outlines a systematic instrumentation approach to collect diagnostic data that will reveal:
- Whether the agent sees profitable opportunities
- If the action space is properly explored during training
- Whether rewards provide sufficient learning gradient
- If technical issues prevent order execution

The instrumentation is designed for rapid implementation with minimal code changes, providing immediate insights into the agent's decision-making process.

## Phase 1: Core Instrumentation (Data Collection)

### 1.1 Action Distribution Tracking

**What to log:**
- Count of each action type (BUY_YES, SELL_YES, BUY_NO, SELL_NO, HOLD)
- Action probabilities from the policy network
- Exploration vs exploitation ratio (when action sampled vs deterministic)
- Action entropy over time

**Implementation location:**
- `environments/market_agnostic_env.py` - Add in `step()` method after action processing
- `training/train_sb3.py` - Add custom callback to track action statistics

**Code snippet:**
```python
# In MarketAgnosticTradingEnv.step()
self.action_history.append({
    'step': self.current_step,
    'action': action,
    'action_type': self._decode_action(action),
    'market_state': {
        'best_yes_bid': self.current_orderbook.get('best_yes_bid'),
        'best_yes_ask': self.current_orderbook.get('best_yes_ask'),
        'spread': self.current_orderbook.get('spread')
    }
})
```

**Expected insights:**
- Is the agent exploring different actions early in training?
- Does action diversity decrease over episodes?
- Are certain actions never attempted?

### 1.2 Reward Signal Analysis

**What to log:**
- Individual reward components (PnL, position penalty, hold penalty)
- Reward magnitude distribution
- Reward sparsity (% of steps with non-zero reward)
- Cumulative reward trajectory per episode

**Implementation location:**
- `environments/market_agnostic_env.py` - Modify `_calculate_reward()` method
- Create reward decomposition dict before returning final reward

**Code snippet:**
```python
# In _calculate_reward()
reward_components = {
    'pnl_change': pnl_change,
    'position_penalty': position_penalty,
    'hold_penalty': hold_penalty if action == 4 else 0,
    'total_reward': total_reward,
    'current_position': self.position,
    'current_pnl': current_pnl
}
self.reward_history.append(reward_components)
```

**Expected insights:**
- Are rewards too sparse to learn from?
- Is position penalty dominating the reward signal?
- Does PnL change provide sufficient gradient?

### 1.3 Observation Space Monitoring

**What to log:**
- Min/max/mean/std of each observation feature
- NaN or infinity checks
- Feature correlation matrix every N episodes
- Feature importance (if using tree-based models for analysis)

**Implementation location:**
- `environments/market_agnostic_env.py` - In `_get_observation()` method
- Add observation validation and statistics tracking

**Code snippet:**
```python
# In _get_observation()
obs_stats = {
    'step': self.current_step,
    'obs_min': float(obs.min()),
    'obs_max': float(obs.max()),
    'obs_mean': float(obs.mean()),
    'obs_std': float(obs.std()),
    'has_nan': bool(np.isnan(obs).any()),
    'has_inf': bool(np.isinf(obs).any())
}
if self.current_step % 100 == 0:  # Log every 100 steps
    self.observation_stats.append(obs_stats)
```

**Expected insights:**
- Are features properly normalized?
- Do some features dominate the observation?
- Are there numerical instabilities?

### 1.4 Order Execution Pipeline

**What to log:**
- Order attempt details (price, quantity, side)
- Order validation results (why rejected if applicable)
- Actual fills vs attempted orders
- Slippage and market impact

**Implementation location:**
- `environments/market_agnostic_env.py` - In `_execute_order()` method
- Track both successful and failed order attempts

**Code snippet:**
```python
# In _execute_order()
order_log = {
    'step': self.current_step,
    'action': action_type,
    'attempted_price': order_price,
    'attempted_qty': quantity,
    'available_liquidity': available_liquidity,
    'execution_success': success,
    'rejection_reason': reason if not success else None,
    'final_position': self.position
}
self.order_history.append(order_log)
```

**Expected insights:**
- Are orders being rejected due to insufficient liquidity?
- Is the agent attempting trades but failing execution?
- What percentage of trade attempts succeed?

### 1.5 Market Dynamics

**What to log:**
- Price volatility over rolling windows
- Spread changes and patterns
- Volume/liquidity availability
- Number of price levels in orderbook

**Implementation location:**
- `environments/market_agnostic_env.py` - Track in `step()` method
- Calculate rolling statistics every N steps

**Code snippet:**
```python
# In step() method
if len(self.price_history) >= 20:
    market_stats = {
        'step': self.current_step,
        'price_volatility': np.std(self.price_history[-20:]),
        'avg_spread': np.mean(self.spread_history[-20:]),
        'liquidity_yes': sum(self.current_orderbook.get('yes_bids', {}).values()),
        'liquidity_no': sum(self.current_orderbook.get('no_bids', {}).values()),
        'orderbook_depth': len(self.current_orderbook.get('yes_bids', {}))
    }
    self.market_dynamics.append(market_stats)
```

**Expected insights:**
- Is there enough volatility to profit from?
- Are spreads too wide to trade profitably?
- Is liquidity sufficient for the agent's position limits?

## Phase 2: Enhanced Logging

### 2.1 Episode-Level Metrics

**Metrics to track at episode end:**
```python
episode_summary = {
    'episode': self.episode_count,
    'total_steps': self.current_step,
    'action_distribution': Counter(self.episode_actions),
    'total_reward': sum(self.episode_rewards),
    'final_pnl': self.final_pnl,
    'final_position': self.position,
    'num_trades': self.trade_count,
    'avg_spread': np.mean(self.spread_history),
    'price_range': (min(self.price_history), max(self.price_history)),
    'profitable_opportunities': self.count_profitable_opportunities(),
    'exploration_rate': self.exploration_rate
}
```

**Implementation:**
- Add to `environments/market_agnostic_env.py` in `reset()` method
- Log before resetting episode variables

### 2.2 Step-Level Metrics

**Critical metrics per step:**
```python
step_metrics = {
    'step': self.current_step,
    'action': action,
    'obs_summary': {
        'price_features': obs[0:10].tolist(),
        'volume_features': obs[10:20].tolist(),
        'position_feature': obs[-1]
    },
    'reward': reward,
    'q_values': self.get_q_values() if available,
    'market_opportunity': self.check_market_opportunity()
}
```

**Implementation:**
- Log every 10-50 steps to avoid overhead
- Include in training callback

### 2.3 Training Progress Indicators

**Key learning metrics:**
```python
training_metrics = {
    'training_step': self.num_timesteps,
    'learning_rate': self.learning_rate,
    'policy_loss': self.policy_loss,
    'value_loss': self.value_loss,
    'entropy': self.entropy,
    'explained_variance': self.explained_variance,
    'kl_divergence': self.kl_divergence,
    'clip_fraction': self.clip_fraction
}
```

**Implementation:**
- Use custom callback in `training/train_sb3.py`
- Log every N training steps

## Phase 3: Diagnostic Tools

### 3.1 Action Heatmap

**Implementation:**
```python
# Create 2D heatmap: time (x-axis) vs action (y-axis)
def create_action_heatmap(action_history, episode_length):
    heatmap = np.zeros((5, episode_length))  # 5 actions
    for step_data in action_history:
        heatmap[step_data['action'], step_data['step']] += 1
    return heatmap
```

**Location:** New file `diagnostics/action_analysis.py`

**Visualization:** Use matplotlib to show action patterns over time

### 3.2 Reward Breakdown

**Implementation:**
```python
def analyze_reward_components(reward_history):
    components = defaultdict(list)
    for reward_data in reward_history:
        for key, value in reward_data.items():
            components[key].append(value)
    
    # Calculate statistics for each component
    stats = {}
    for key, values in components.items():
        stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'zero_fraction': sum(1 for v in values if v == 0) / len(values)
        }
    return stats
```

**Location:** New file `diagnostics/reward_analysis.py`

### 3.3 Market Opportunity Analysis

**Implementation:**
```python
def identify_missed_opportunities(market_data, action_history):
    opportunities = []
    for i in range(len(market_data) - 1):
        current = market_data[i]
        future = market_data[i + 1]
        
        # Check if buying would have been profitable
        if future['mid_price'] > current['best_ask'] * 1.002:  # 0.2% profit threshold
            if action_history[i] != 'BUY':
                opportunities.append({
                    'step': i,
                    'type': 'missed_buy',
                    'potential_profit': future['mid_price'] - current['best_ask']
                })
    return opportunities
```

**Location:** New file `diagnostics/opportunity_analysis.py`

## Implementation Checklist

### Priority 1 (Immediate - Day 1)
□ Add action counter to environment (`market_agnostic_env.py`)
□ Add reward component tracking (`_calculate_reward()`)
□ Create basic episode summary logging
□ Add observation validation checks
□ Implement simple console output for action distribution

### Priority 2 (Quick Wins - Day 2)
□ Enhance PortfolioMetricsCallback with action tracking
□ Add market dynamics logging (spreads, volatility)
□ Implement order execution tracking
□ Create diagnostic summary script
□ Add training progress metrics to callbacks

### Priority 3 (Deep Analysis - Day 3)
□ Build action heatmap visualization
□ Create reward breakdown analysis
□ Implement market opportunity detector
□ Add feature importance analysis
□ Create comprehensive diagnostic report generator

## Expected Outputs

After implementing this instrumentation, we will have:

1. **Action Distribution Report**: Clear view of action selection patterns showing if HOLD dominance is immediate or develops over time
2. **Reward Signal Analysis**: Understanding if rewards are too sparse, penalties too high, or gradient too weak
3. **Observation Quality Report**: Validation that features are properly normalized and informative
4. **Order Execution Log**: Detailed record of why trades fail (if attempted)
5. **Market Opportunity Map**: Identification of profitable trades the agent missed
6. **Training Progress Dashboard**: Real-time view of learning metrics and convergence
7. **Episode Summaries**: Concise reports showing behavior evolution across episodes

## Success Criteria

We will know the instrumentation is complete when we can answer:

### Primary Questions
- **Why does the agent always choose HOLD?**
  - Measured by: action distribution logs, Q-value analysis
  - Success: Can identify specific cause (exploration, reward, or technical issue)

- **Are there profitable opportunities being missed?**
  - Measured by: market opportunity analysis, hindsight profit calculation
  - Success: Quantify potential profit from optimal trading

- **Is the observation space properly normalized?**
  - Measured by: observation statistics, feature distributions
  - Success: All features in reasonable ranges, no numerical issues

- **Does the reward signal provide learning gradient?**
  - Measured by: reward component breakdown, sparsity analysis
  - Success: Identify if rewards are too sparse or penalties dominate

- **Is exploration happening during training?**
  - Measured by: action entropy, exploration rate tracking
  - Success: Confirm exploration schedule is working as intended

### Secondary Questions
- What is the typical spread when agent should trade?
- How does position size affect decision making?
- Are there specific market conditions that trigger non-HOLD actions?
- Does the agent ever attempt trades that fail execution?

## Next Steps

Once instrumentation is complete and data collected:

1. **Run instrumented training for 10-50 episodes**
2. **Generate diagnostic report**
3. **Identify top 3 issues causing HOLD-only behavior**
4. **Create targeted fixes based on data insights**
5. **Validate fixes with A/B testing**

## Notes

- All logging should be toggleable via `debug_mode` flag to avoid overhead in production
- Store diagnostic data in structured format (JSON/pickle) for later analysis
- Consider using TensorBoard for real-time metric visualization
- Keep logging overhead under 5% of training time