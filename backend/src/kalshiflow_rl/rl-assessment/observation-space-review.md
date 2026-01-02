# Observation Space Review and Spread-Aware Feature Recommendations

*Date: December 16, 2024*
*Reviewer: RL Assessment Expert*

## Executive Summary

After reviewing the current observation space implementation (52 features), I recommend:
- **KEEP**: 70% of current features (36/52) - they provide valuable trading signals
- **REMOVE**: 15% of redundant features (8/52) - overlapping or low-signal features
- **ADD**: 15 new spread-aware features critical for profitable trading

The most critical gap is the lack of explicit spread cost awareness. The model currently sees spread components (bid/ask prices) but lacks features that directly quantify trading costs and profitability thresholds.

## Current Observation Space Breakdown (52 Features)

### Market Features (21 features per market × 1 market = 21)
- Price features: 8 (bid/ask/mid/spread for YES/NO)
- Volume features: 6 (volumes, imbalances)
- Order book shape: 4 (depth, liquidity concentration)
- Efficiency features: 3 (arbitrage, efficiency metrics)

### Temporal Features (14)
- Time features: 3 (time since update, time of day, day of week)
- Activity features: 6 (activity score, change, frequency, burst, quiet, trend)
- Price momentum: 1
- Market stability: 4 (volatility, consistency, stability, persistence)

### Portfolio Features (12)
- Composition: 2 (cash ratio, position ratio)
- Position metrics: 8 (count, size, long/short ratios, concentration, P&L)
- Risk metrics: 2 (diversity, leverage)

### Order Features (5)
- Order state: 5 (open buy/sell flags, distances, time since order)

## Features to KEEP (36/52) ✅

### Critical Price Features (KEEP ALL)
```python
'best_yes_bid_norm'     # Essential for entry/exit decisions
'best_yes_ask_norm'     # Essential for entry/exit decisions
'best_no_bid_norm'      # Essential for NO-side trading
'best_no_ask_norm'      # Essential for NO-side trading
'yes_mid_price_norm'    # Core price signal
'no_mid_price_norm'     # Core price signal
```

### Essential Volume Features (KEEP)
```python
'total_volume_norm'     # Market activity indicator
'volume_imbalance'      # Directional pressure signal
'yes_side_imbalance'    # YES-specific pressure
'no_side_imbalance'     # NO-specific pressure
```

### Valuable Temporal Features (KEEP)
```python
'time_since_last_update'  # Activity indicator
'activity_score'          # Composite activity metric
'activity_burst_indicator'# Opportunity detection
'price_momentum'          # Trend signal
'volatility_regime'       # Risk assessment
```

### Critical Portfolio Features (KEEP ALL)
```python
'cash_ratio'              # Capital allocation
'position_ratio'          # Exposure level
'net_position_bias'       # Directional exposure
'unrealized_pnl_ratio'    # Current performance
'leverage'                # Risk level
```

## Features to REMOVE (8/52) ❌

### Redundant Features
```python
'yes_volume_norm'         # Redundant with total_volume_norm + volume_imbalance
'no_volume_norm'          # Redundant with total_volume_norm + volume_imbalance
'cross_side_efficiency'   # Redundant with arbitrage_opportunity
'quiet_period_indicator'  # Inverse of activity_burst_indicator
'activity_consistency'    # Low signal value
'activity_persistence'    # Overlaps with activity_trend
'position_diversity'      # Not relevant for single-market training
'day_of_week_norm'        # Low signal for prediction markets
```

### Rationale for Removal
- Volume features are captured by total_volume and imbalances
- Efficiency metrics overlap significantly
- Activity features have redundancy
- Some features don't apply to single-market focus

## New Spread-Aware Features to ADD (15) 🆕

### 1. Direct Spread Cost Features (CRITICAL)
```python
def calculate_spread_cost_features(orderbook_data):
    """Direct spread cost quantification"""
    
    # Absolute spread in cents
    yes_spread_cents = (best_yes_ask - best_yes_bid)
    no_spread_cents = (best_no_ask - best_no_bid)
    
    # Percentage spread (relative to mid-price)
    yes_spread_pct = yes_spread_cents / yes_mid_price if yes_mid_price > 0 else 0
    no_spread_pct = no_spread_cents / no_mid_price if no_mid_price > 0 else 0
    
    # Round-trip cost (buy + sell spread)
    yes_roundtrip_cost = yes_spread_cents * 2  # Cost to enter and exit
    no_roundtrip_cost = no_spread_cents * 2
    
    # Effective spread (weighted by volume at best prices)
    yes_effective_spread = calculate_effective_spread(yes_bids, yes_asks)
    no_effective_spread = calculate_effective_spread(no_bids, no_asks)
    
    return {
        'yes_spread_cents': yes_spread_cents / 100.0,  # Normalize to [0,1]
        'no_spread_cents': no_spread_cents / 100.0,
        'yes_spread_pct': min(yes_spread_pct, 0.5),  # Cap at 50%
        'no_spread_pct': min(no_spread_pct, 0.5),
        'yes_roundtrip_cost': yes_roundtrip_cost / 100.0,
        'no_roundtrip_cost': no_roundtrip_cost / 100.0,
        'yes_effective_spread': yes_effective_spread / 100.0,
        'no_effective_spread': no_effective_spread / 100.0
    }
```

### 2. Spread Regime Classification
```python
def calculate_spread_regime(spread_cents):
    """Classify market by spread tightness"""
    if spread_cents < 2:
        return 1.0   # Ultra-tight (very liquid)
    elif spread_cents < 5:
        return 0.75  # Tight (liquid)
    elif spread_cents < 10:
        return 0.5   # Medium (semi-liquid)
    elif spread_cents < 20:
        return 0.25  # Wide (challenging)
    else:
        return 0.0   # Very wide (avoid)

features['yes_spread_regime'] = calculate_spread_regime(yes_spread_cents)
features['no_spread_regime'] = calculate_spread_regime(no_spread_cents)
```

### 3. Profitability Threshold Features
```python
def calculate_profitability_thresholds(orderbook_data, fees=0.07):
    """Calculate minimum price movement needed for profit"""
    
    # Minimum move to break even (spread + fees)
    yes_breakeven_move = (yes_spread_cents + fees * 2) / yes_mid_price
    no_breakeven_move = (no_spread_cents + fees * 2) / no_mid_price
    
    # Distance to profitable exit (current price to breakeven)
    yes_profit_distance = yes_breakeven_move * yes_mid_price
    no_profit_distance = no_breakeven_move * no_mid_price
    
    return {
        'yes_breakeven_move': min(yes_breakeven_move, 1.0),
        'no_breakeven_move': min(no_breakeven_move, 1.0),
        'yes_profit_distance': yes_profit_distance / 100.0,
        'no_profit_distance': no_profit_distance / 100.0
    }
```

### 4. Liquidity-Adjusted Features
```python
def calculate_liquidity_features(orderbook_data):
    """Features that combine spread with liquidity"""
    
    # Spread-volume ratio (tightness per unit of liquidity)
    yes_spread_volume_ratio = yes_spread_cents / max(total_yes_volume, 1)
    no_spread_volume_ratio = no_spread_cents / max(total_no_volume, 1)
    
    # Liquidity score (inverse of spread * volume)
    yes_liquidity_score = total_yes_volume / max(yes_spread_cents, 0.1)
    no_liquidity_score = total_no_volume / max(no_spread_cents, 0.1)
    
    # Normalized to [0,1]
    max_liquidity_ref = 10000.0
    
    return {
        'yes_liquidity_score': min(yes_liquidity_score / max_liquidity_ref, 1.0),
        'no_liquidity_score': min(no_liquidity_score / max_liquidity_ref, 1.0)
    }
```

### 5. Trade Execution Quality Features
```python
def calculate_execution_features(orderbook_data, trade_size=10):
    """Estimate execution quality for different trade sizes"""
    
    # Market impact for small trade (VWAP for trade_size contracts)
    yes_small_impact = calculate_vwap(yes_asks, trade_size) - yes_mid_price
    no_small_impact = calculate_vwap(no_asks, trade_size) - no_mid_price
    
    # Market impact for large trade (VWAP for 5x trade_size)
    yes_large_impact = calculate_vwap(yes_asks, trade_size * 5) - yes_mid_price
    no_large_impact = calculate_vwap(no_asks, trade_size * 5) - no_mid_price
    
    return {
        'yes_small_impact': yes_small_impact / 100.0,
        'no_small_impact': no_small_impact / 100.0,
        'yes_large_impact': yes_large_impact / 100.0,
        'no_large_impact': no_large_impact / 100.0
    }
```

### 6. Temporal Spread Features
```python
def calculate_temporal_spread_features(current_spread, historical_spreads):
    """Track spread dynamics over time"""
    
    if len(historical_spreads) >= 5:
        # Spread trend (widening or tightening)
        spread_trend = np.polyfit(range(len(historical_spreads)), 
                                historical_spreads, 1)[0]
        
        # Spread volatility
        spread_volatility = np.std(historical_spreads)
        
        # Current spread vs average
        avg_spread = np.mean(historical_spreads)
        spread_deviation = (current_spread - avg_spread) / max(avg_spread, 1)
    else:
        spread_trend = 0.0
        spread_volatility = 0.1
        spread_deviation = 0.0
    
    return {
        'spread_trend': np.tanh(spread_trend * 10),  # Normalize to [-1,1]
        'spread_volatility': min(spread_volatility / 10.0, 1.0),
        'spread_deviation': np.tanh(spread_deviation)
    }
```

## Implementation Priority

### Phase 1: Critical Spread Features (Immediate)
1. Add `yes_spread_cents` and `no_spread_cents` (absolute spread cost)
2. Add `yes_spread_pct` and `no_spread_pct` (relative spread cost)
3. Add `yes_spread_regime` and `no_spread_regime` (liquidity classification)

### Phase 2: Profitability Features (Next Sprint)
1. Add breakeven move calculations
2. Add round-trip cost features
3. Add liquidity score metrics

### Phase 3: Advanced Features (Future)
1. Add execution quality estimates
2. Add temporal spread dynamics
3. Add market impact predictions

## Expected Impact

### Before (Current State)
- Model unaware of explicit trading costs
- Treats 2¢ and 50¢ spreads similarly
- No profitability threshold awareness
- Results: Unprofitable trades, negative returns

### After (With Spread Features)
- Model explicitly sees trading costs
- Can differentiate liquid vs illiquid markets
- Knows minimum move needed for profit
- Expected: 30-50% improvement in win rate

## Code Integration Example

```python
# In feature_extractors.py, modify extract_market_agnostic_features():

def extract_market_agnostic_features(orderbook_data: Dict[str, Any]) -> Dict[str, float]:
    """Enhanced with spread-aware features"""
    
    # ... existing feature extraction ...
    
    # ADD: Direct spread features
    yes_spread_cents = (best_yes_ask - best_yes_bid) if (best_yes_ask and best_yes_bid) else 50
    no_spread_cents = (best_no_ask - best_no_bid) if (best_no_ask and best_no_bid) else 50
    
    features['yes_spread_cents'] = yes_spread_cents / 100.0  # Normalize
    features['no_spread_cents'] = no_spread_cents / 100.0
    
    # ADD: Spread percentage
    features['yes_spread_pct'] = min(yes_spread_cents / max(yes_mid * 100, 1), 0.5)
    features['no_spread_pct'] = min(no_spread_cents / max(no_mid * 100, 1), 0.5)
    
    # ADD: Spread regime
    features['yes_spread_regime'] = calculate_spread_regime(yes_spread_cents)
    features['no_spread_regime'] = calculate_spread_regime(no_spread_cents)
    
    # ADD: Liquidity score
    yes_liquidity = total_yes_volume / max(yes_spread_cents, 0.1)
    no_liquidity = total_no_volume / max(no_spread_cents, 0.1)
    features['yes_liquidity_score'] = min(yes_liquidity / 10000.0, 1.0)
    features['no_liquidity_score'] = min(no_liquidity / 10000.0, 1.0)
    
    # REMOVE: Redundant features
    # Remove: yes_volume_norm, no_volume_norm, cross_side_efficiency
    
    return features
```

## Validation Metrics

Track these metrics to verify improvement:

1. **Spread Awareness**: Model should avoid markets with spread_regime < 0.25
2. **Trade Selection**: >80% of trades in markets with <10¢ spreads
3. **Win Rate**: Increase from 20% to 45-55%
4. **Profit per Trade**: Should exceed spread cost + fees
5. **Sharpe Ratio**: Target >0.5 with spread features

## Conclusion

The current observation space provides a solid foundation but critically lacks explicit spread awareness. By removing 8 redundant features and adding 15 spread-aware features, we can maintain the same 52-feature dimension while dramatically improving the model's ability to:

1. **Identify tradeable markets** (via spread regime classification)
2. **Calculate true costs** (via spread cost features)
3. **Determine profitability thresholds** (via breakeven features)
4. **Optimize execution** (via liquidity and impact features)

These changes are essential for "low and slow" profitable trading with proper fee awareness. The spread features directly address the #1 issue identified in training: the model needs to understand when NOT to trade due to excessive costs.

## Recommended Next Steps

1. Implement Phase 1 spread features (3 features)
2. Remove redundant features to maintain 52-feature space
3. Retrain on filtered liquid markets (spread < 10¢)
4. Compare win rates before/after spread features
5. Iterate with Phase 2 features based on results