# Simple Trading Strategy Implementation Plan

## Overview
Leverage the existing `ActionSelector` infrastructure to implement profitable hardcoded trading strategies while continuing RL development in parallel.

## Current Situation
- **Infrastructure**: Fully built RL system with orderbook depth consumption and probabilistic fills
- **Problem**: RL approach too complex for the simple spread capture opportunity  
- **Solution**: Use pluggable `ActionSelector` interface for simple, profitable strategies
- **Timeline**: Get profitable in 2-4 weeks instead of 6-12 months

## Implementation Plan

### Phase 1: Market Making Selector

Create `market_maker_selector.py`:

```python
class MarketMakerSelector(ActionSelector):
    """Simple but profitable market-making strategy."""
    
    def __init__(self, config: dict):
        self.min_spread = config.get('min_spread', 2)
        self.max_position = config.get('max_position', 100)
        self.order_size_index = config.get('order_size_index', 1)  # 10 contracts default
        
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        # Extract key features from observation
        spread = self._get_spread(observation)
        position = self._get_position(observation)
        mid_price = self._get_mid_price(observation)
        
        # Only trade when spread is profitable
        if spread < self.min_spread:
            return 0  # HOLD
        
        # Risk management: reduce activity with large positions
        if abs(position) > self.max_position:
            return 0  # HOLD to reduce risk
        
        # Inventory management
        if position > 20:
            # Skew towards selling when long
            return 6 + self.order_size_index  # SELL_YES
        elif position < -20:
            # Skew towards buying when short  
            return 1 + self.order_size_index  # BUY_YES
        else:
            # Balanced market making
            if spread > 3:
                # Wide spread - alternate sides
                return (1 + self.order_size_index) if np.random.random() > 0.5 else (6 + self.order_size_index)
            else:
                return 0  # Wait for better spread
    
    def _get_spread(self, obs: np.ndarray) -> float:
        # Extract spread from observation vector
        # Features 6-7 are best bid/ask
        return obs[7] - obs[6] if len(obs) > 7 else 0
    
    def _get_position(self, obs: np.ndarray) -> float:
        # Extract position from observation vector  
        # Feature 42 is position
        return obs[42] if len(obs) > 42 else 0
    
    def _get_mid_price(self, obs: np.ndarray) -> float:
        # Calculate mid from bid/ask
        return (obs[6] + obs[7]) / 2 if len(obs) > 7 else 50
```

### Phase 2: Spread Capture Selector

Create `spread_capture_selector.py`:

```python
class SpreadCaptureSelector(ActionSelector):
    """Focus purely on capturing wide spreads."""
    
    def __init__(self, config: dict):
        self.min_spread = config.get('min_spread', 5)
        self.order_size_index = config.get('order_size_index', 0)  # 5 contracts (small)
        self.last_action = 0
        
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        spread = self._get_spread(observation)
        
        # Only act on very wide spreads
        if spread >= self.min_spread:
            # Alternate between buy and sell to capture spread
            if self.last_action <= 5:  # Was HOLD or BUY
                self.last_action = 6 + self.order_size_index  # SELL_YES
            else:
                self.last_action = 1 + self.order_size_index  # BUY_YES
            return self.last_action
        
        self.last_action = 0
        return 0  # HOLD
```

### Phase 3: Arbitrage Selector

Create `arbitrage_selector.py`:

```python
class ArbitrageSelector(ActionSelector):
    """Look for YES+NO != 100 opportunities."""
    
    def __init__(self, config: dict):
        self.min_edge = config.get('min_edge', 2)  # Minimum cents of edge
        self.order_size_index = config.get('order_size_index', 2)  # 20 contracts
        
    async def select_action(self, observation: np.ndarray, market_ticker: str) -> int:
        yes_ask = self._get_yes_ask(observation)
        no_ask = self._get_no_ask(observation)
        
        # Check if we can buy both for less than 100
        total_cost = yes_ask + no_ask
        if total_cost <= (100 - self.min_edge):
            # Arbitrage opportunity - buy both
            # In single market actor, just buy YES
            # Would need multi-market actor for full arb
            return 1 + self.order_size_index  # BUY_YES
        
        yes_bid = self._get_yes_bid(observation)
        no_bid = self._get_no_bid(observation)
        
        # Check if we can sell both for more than 100
        total_sale = yes_bid + no_bid
        if total_sale >= (100 + self.min_edge):
            # Sell both for profit
            return 6 + self.order_size_index  # SELL_YES
        
        return 0  # HOLD
```

### Phase 4: Configuration Updates

Update `actor_config.yaml`:

```yaml
actor:
  # Strategy selection
  strategy: "market_maker"  # Options: market_maker, spread_capture, arbitrage, rl_model, hold
  
  # RL Model settings (when strategy: "rl_model")
  model_path: "trained_models/latest_model.zip"
  
  # Market Maker settings
  market_maker:
    min_spread: 2
    max_position: 100
    order_size_index: 1  # 0=5, 1=10, 2=20, 3=50, 4=100 contracts
  
  # Spread Capture settings
  spread_capture:
    min_spread: 5
    order_size_index: 0  # Small orders for testing
  
  # Arbitrage settings
  arbitrage:
    min_edge: 2
    order_size_index: 2  # Medium size
```

### Phase 5: Factory Function Update

Update `create_action_selector()` in `action_selector.py`:

```python
def create_action_selector(config: dict) -> ActionSelector:
    """
    Factory function to create appropriate ActionSelector based on config.
    
    Args:
        config: Actor configuration dict
        
    Returns:
        ActionSelector instance
    """
    strategy = config.get("strategy", "hold")
    
    if strategy == "market_maker":
        from .market_maker_selector import MarketMakerSelector
        return MarketMakerSelector(config.get("market_maker", {}))
    
    elif strategy == "spread_capture":
        from .spread_capture_selector import SpreadCaptureSelector
        return SpreadCaptureSelector(config.get("spread_capture", {}))
    
    elif strategy == "arbitrage":
        from .arbitrage_selector import ArbitrageSelector
        return ArbitrageSelector(config.get("arbitrage", {}))
    
    elif strategy == "rl_model":
        model_path = config.get("model_path")
        if not model_path:
            logger.error("No model_path specified for rl_model strategy")
            return HardcodedSelector("hold")
        return RLModelSelector(model_path)
    
    else:
        # Default to hold strategy
        return HardcodedSelector("hold")
```

## Testing Plan

### 1. Unit Tests
Create `test_simple_strategies.py`:
- Test each selector with various market conditions
- Verify position management logic
- Ensure risk limits are respected

### 2. Backtesting
- Run each strategy on historical orderbook data
- Measure: spread capture rate, fill rate, P&L
- Compare strategies across different market conditions

### 3. Paper Trading
- Start with 5-10 markets
- Small positions (5-10 contracts)
- Monitor for 24-48 hours
- Track metrics: fills, P&L, max drawdown

### 4. Production Rollout
- Start with 1 market, minimal size
- Gradually increase markets and size
- A/B test different strategies
- Keep RL training in parallel

## Success Metrics

### Immediate (Week 1)
- [ ] Positive P&L in paper trading
- [ ] Fill rate > 30% for passive orders
- [ ] Spread capture on >50% of trades

### Short-term (Week 2-4)
- [ ] Consistent daily profits
- [ ] Sharpe ratio > 1.0
- [ ] Max drawdown < 10%

### Long-term (Month 2+)
- [ ] RL model beats simple strategies
- [ ] Automated strategy selection
- [ ] Multi-market coordination

## Risk Management

### Position Limits
- Max position per market: 100 contracts
- Max portfolio value at risk: 20%
- Minimum cash reserve: $50

### Circuit Breakers
- Stop trading if:
  - Loss > $50 in 1 hour
  - Loss > $100 in 1 day
  - Technical errors > 5 in 10 minutes

### Monitoring
- Real-time P&L dashboard
- Fill rate tracking
- Position exposure heatmap
- Alert on anomalies

## Implementation Timeline

### Week 1
- [ ] Implement MarketMakerSelector
- [ ] Create unit tests
- [ ] Backtest on 1 week of data
- [ ] Deploy to paper trading

### Week 2  
- [ ] Implement SpreadCaptureSelector
- [ ] Implement ArbitrageSelector
- [ ] A/B test all strategies
- [ ] Select best performer

### Week 3
- [ ] Production deployment (small)
- [ ] Build monitoring dashboard
- [ ] Iterate on parameters

### Week 4
- [ ] Scale to more markets
- [ ] Increase position sizes
- [ ] Document learnings

## Parallel RL Development

While simple strategies run:

1. **Improve reward function**: Add spread capture rewards
2. **Better features**: Add microstructure signals
3. **More training**: 1M+ timesteps across sessions
4. **Hyperparameter tuning**: Optimize PPO settings

Goal: RL beats simple strategies within 2 months

## Key Insights

1. **Simple strategies can be profitable NOW** - Don't need complex RL for basic spread capture
2. **Infrastructure is ready** - ActionSelector interface makes strategies pluggable
3. **Risk is managed** - Hardcoded rules prevent disasters
4. **RL continues improving** - Not abandoning the sophisticated approach
5. **Fast iteration** - Can test new strategies in hours, not weeks

## Next Steps

1. Review this plan
2. Implement MarketMakerSelector first
3. Test locally with simulated orders
4. Deploy to paper trading
5. Monitor and iterate

The path to profitability is through **starting simple and iterating fast**.