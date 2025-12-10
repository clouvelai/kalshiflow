# Kalshi Environment Foundation: Assessment and 1.0 Rewrite Strategy

## Executive Summary

Based on comprehensive assessment of the current Kalshi RL environment and alignment with the market-agnostic 1.0 model goals, we are implementing a **complete architectural rewrite**. The existing environment, while functional, has fundamental limitations that prevent achieving the sophisticated pattern discovery and market-agnostic trading capabilities outlined in our theoretical framework.

### Key Decision: Complete Rewrite with Session-Based Architecture

**Rationale**: The introduction of `session_ids` in our collected data provides **data continuity guarantees** that enable a dramatically simplified and more powerful training approach. Instead of complex multi-market coordination, we can specify a single session_id for training episodes, gaining algorithmic simplicity while maintaining real-world validity.

### Strategic Goals for 1.0 Environment
1. **Market-Agnostic Training**: Single model that generalizes across all Kalshi markets
2. **Session-Based Episodes**: Use session_id for guaranteed data continuity and simplicity  
3. **Sophisticated Pattern Discovery**: Architecture optimized for Tier 3/4 orderbook strategies
4. **Production-Ready Design**: Clean, fast, and maintainable codebase aligned with theory

## Assessment of Current Environment

### Current Architecture Strengths
✅ **Non-blocking data pipeline**: Efficient orderbook ingestion via async queues  
✅ **Database integration**: Complete PostgreSQL schema with proper indexing  
✅ **Multi-market support**: Variable market count (1-10) with scaling  
✅ **Gymnasium compatibility**: Passes all SB3 integration tests  
✅ **Performance**: Sub-millisecond step/reset operations  

### Critical Limitations Requiring Rewrite

#### 1. **Incompatible with Market-Agnostic Theory**
**Current**: Market-specific environments with ticker-aware configurations
```python
# Current approach - market-specific
market_tickers = ["TRUMP-2024", "FED-HIKE-DEC"]
env = KalshiTradingEnv(market_tickers=market_tickers)
```

**Required**: Market-agnostic training where model never sees market identity
```python
# Target approach - market-agnostic
session_id = "session_12345"  # Contains mixed markets, no ticker exposure
env = KalshiTradingEnv(session_id=session_id, max_markets=5)
```

#### 2. **Complex Multi-Market Coordination**
**Current**: Attempts to coordinate multiple markets simultaneously
- 545-feature observation space trying to handle 10 markets
- Complex cross-market synchronization 
- Artificial market padding and normalization

**Required**: Simple single-session replay with natural market mixing
- Session contains organic multi-market data
- No artificial coordination needed
- Natural temporal flow preserved

#### 3. **No Temporal Pattern Discovery Support**
**Current**: Static feature extraction without temporal learning
```python
# Current - basic features only
obs = [price_levels, spreads, volumes]  # Static snapshot
```

**Required**: Temporal-aware architecture supporting sophisticated patterns
```python
# Target - temporal pattern discovery
obs = {
    'market_features': [...],      # Orderbook state
    'temporal_features': [...],    # Time-gap analysis, burst detection
    'sequence_features': [...]     # Pattern sequences, regime detection
}
```

#### 4. **Dummy Data Dependencies**
**Current**: Relies on synthetic/dummy data generation
- Hard-coded sinusoidal price movements
- Artificial volume and spread generation
- No real market dynamics

**Required**: Pure historical data with real market microstructure
- Only authentic Kalshi orderbook data
- Real temporal gaps and patterns
- Authentic market regime changes

#### 5. **Reward Function Oversimplification**
**Current**: Basic P&L calculation without sophisticated strategy support
```python
# Current - simple returns
reward = position_change * price_change
```

**Required**: Multi-tier strategy-aware rewards supporting pattern discovery
```python
# Target - strategy-aware rewards
reward = {
    'basic_pnl': position_pnl,
    'pattern_discovery': novelty_bonus,
    'temporal_exploitation': timing_bonus,
    'risk_adjusted': sharpe_contribution
}
```

### Session Data Advantages

With session_ids, we gain:
- **Data Continuity**: Guaranteed contiguous market data with no gaps
- **Natural Market Mixing**: Sessions contain organic multi-market activity
- **Temporal Authenticity**: Real timing patterns and sequence relationships
- **Simplicity**: No complex market coordination or padding required
- **Scalability**: Easy to generate diverse training scenarios

## New Environment Architecture

### Core Design Principles

#### 1. **Market-Agnostic by Design**
```python
class MarketAgnosticKalshiEnv(gym.Env):
    """
    Model never sees market tickers or market-specific metadata.
    All features are normalized to universal probability space [0,1].
    Supports 1-N markets per episode via session_id specification.
    """
    
    def __init__(self, session_config: SessionConfig):
        # NO market_tickers parameter
        # NO market-specific configuration
        # Session contains all necessary market data
        pass
```

#### 2. **Session-Based Episode Generation**
```python
class SessionConfig:
    session_id: str                    # Primary: Use specific session
    session_pool: List[str]           # Alternative: Random from pool
    max_markets: int = 5              # Limit concurrent markets
    episode_length: int = 1000        # Steps per episode
    start_offset: int = 0             # Skip initial steps
    temporal_features: bool = True    # Enable temporal analysis
```

#### 3. **Sophisticated Feature Engineering**
```python
class MarketAgnosticObservationSpace:
    """
    Three-tier feature hierarchy optimized for pattern discovery.
    """
    
    # Tier 1: Basic orderbook (ALWAYS)
    basic_features: np.ndarray        # [12] - spreads, mid-prices, volumes
    
    # Tier 2: Temporal patterns (CONDITIONAL)
    temporal_features: np.ndarray     # [16] - time gaps, activity rates, momentum
    
    # Tier 3: Advanced patterns (RESEARCH)
    sequence_features: np.ndarray     # [24] - pattern sequences, regime detection
```

### Session-Based Architecture Implementation

#### Session Data Structure
```python
@dataclass
class SessionData:
    """
    Complete session data with all markets and temporal information.
    """
    session_id: str
    markets: List[str]                    # Markets active in this session
    start_time: datetime
    end_time: datetime
    data_points: List[SessionDataPoint]   # Ordered by timestamp
    
@dataclass 
class SessionDataPoint:
    """
    Single temporal data point across all active markets.
    """
    timestamp: int                        # Unix ms
    sequence_number: int                  # Global sequence
    market_states: Dict[str, OrderbookSnapshot]  # Market ticker -> state
    deltas: List[OrderbookDelta]         # Changes since last point
    time_gap: int                        # Ms since previous point
```

#### Session-Based Data Loading
```python
class SessionDataLoader:
    """
    Loads complete session data with temporal consistency guarantees.
    """
    
    def load_session(self, session_id: str) -> SessionData:
        """
        Load complete session data from database.
        
        Key advantages over current approach:
        - Single database query for entire session
        - Natural multi-market coordination
        - Preserved temporal relationships
        - No market-specific configuration needed
        """
        
        # Load all session data in single query
        session_data = await self.db.fetch_session_data(session_id)
        
        # Group by timestamp for multi-market coordination
        temporal_points = self._group_by_timestamp(session_data)
        
        # Calculate temporal features (gaps, momentum, etc.)
        enriched_points = self._add_temporal_features(temporal_points)
        
        return SessionData(
            session_id=session_id,
            data_points=enriched_points
        )
    
    def _group_by_timestamp(self, raw_data) -> List[SessionDataPoint]:
        """
        Natural multi-market coordination via timestamp grouping.
        No artificial market padding or complex synchronization.
        """
        pass
```

### Market-Agnostic Feature Engineering Strategy

#### Universal Feature Extraction
```python
def extract_market_agnostic_features(market_state: OrderbookSnapshot) -> np.ndarray:
    """
    Extract features that work identically across ALL Kalshi markets.
    
    Key insight: Prediction markets are naturally normalized to [0,100] cents.
    This enables universal feature engineering without market-specific scaling.
    """
    
    features = []
    
    # 1. Probability Space Features (already normalized!)
    features.extend([
        market_state.yes_mid_price / 100.0,  # Direct probability [0,1]
        market_state.no_mid_price / 100.0,   # Complement probability [0,1]
        abs(market_state.yes_mid_price - 50) / 50.0,  # Distance from 50/50 [0,1]
    ])
    
    # 2. Spread Features (Market Confidence)
    features.extend([
        market_state.yes_spread / max(market_state.yes_mid_price, 1),  # Relative spread
        market_state.no_spread / max(market_state.no_mid_price, 1),    # Relative spread
    ])
    
    # 3. Order Book Imbalance (Universal Microstructure)
    yes_bid_volume = sum(level.quantity for level in market_state.yes_bids[:5])
    yes_ask_volume = sum(level.quantity for level in market_state.yes_asks[:5])
    yes_imbalance = (yes_bid_volume - yes_ask_volume) / max(yes_bid_volume + yes_ask_volume, 1)
    features.append(yes_imbalance)
    
    # 4. Extremity Features (Distance from certainty)
    min_uncertainty = min(market_state.yes_mid_price, 100 - market_state.yes_mid_price)
    features.append(min_uncertainty / 50.0)  # [0,1] where 1 = most uncertain
    
    # 5. Arbitrage Opportunity Detection
    total_mid = market_state.yes_mid_price + market_state.no_mid_price
    arbitrage_signal = abs(total_mid - 100) / 100.0
    features.append(arbitrage_signal)
    
    return np.array(features, dtype=np.float32)
```

#### Temporal Feature Engineering
```python
def extract_temporal_features(
    current_point: SessionDataPoint, 
    history: List[SessionDataPoint]
) -> np.ndarray:
    """
    Extract temporal features that capture market dynamics.
    
    These features enable Tier 3/4 strategy discovery:
    - Temporal clustering patterns
    - Activity regime detection  
    - Momentum and mean reversion signals
    """
    
    features = []
    
    # 1. Time Gap Analysis (Log Scale)
    if len(history) > 0:
        delta_t_ms = current_point.timestamp - history[-1].timestamp
        log_delta_t = np.log1p(delta_t_ms / 1000.0)  # Log seconds
        features.append(log_delta_t)
        
        # Gap classification
        features.extend([
            1.0 if delta_t_ms < 100 else 0.0,      # Burst (<100ms)
            1.0 if delta_t_ms < 5000 else 0.0,     # Active (<5s) 
            1.0 if delta_t_ms > 30000 else 0.0,    # Quiet (>30s)
        ])
    else:
        features.extend([0.0, 0.0, 0.0, 0.0])
    
    # 2. Activity Rate Features
    if len(history) >= 10:
        now = current_point.timestamp
        recent_10s = [p for p in history[-10:] if now - p.timestamp < 10000]
        recent_60s = [p for p in history[-50:] if now - p.timestamp < 60000]
        
        features.extend([
            len(recent_10s) / 10.0,    # Events per second (10s window)
            len(recent_60s) / 60.0,    # Events per second (60s window)
        ])
    else:
        features.extend([0.0, 0.0])
    
    # 3. Activity Momentum
    if len(history) >= 20:
        recent_gaps = [history[i].timestamp - history[i-1].timestamp 
                      for i in range(-10, 0) if i < len(history)]
        older_gaps = [history[i].timestamp - history[i-1].timestamp 
                     for i in range(-20, -10) if i < len(history)]
        
        if recent_gaps and older_gaps:
            recent_avg = np.mean(recent_gaps)
            older_avg = np.mean(older_gaps) 
            acceleration = (recent_avg - older_avg) / max(older_avg, 1.0)
            features.append(np.tanh(acceleration))  # Bounded [-1,1]
        else:
            features.append(0.0)
    else:
        features.append(0.0)
        
    return np.array(features, dtype=np.float32)
```

### Training Strategy Evolution

#### Curriculum Learning with Sessions
```python
class SessionBasedCurriculum:
    """
    Progressive training complexity using session selection.
    """
    
    def select_training_session(self, episode_num: int) -> str:
        """
        Curriculum learning through intelligent session selection.
        """
        progress = min(episode_num / 50000, 1.0)  # 0->1 over 50k episodes
        
        if progress < 0.2:
            # Phase 1: Stable single-market sessions
            return random.choice(self.stable_single_market_sessions)
        elif progress < 0.5:
            # Phase 2: Multi-market moderate volatility  
            return random.choice(self.moderate_multi_market_sessions)
        else:
            # Phase 3: Full diversity including volatile/edge-case sessions
            return random.choice(self.all_sessions)
```

#### Episode Generation from Sessions
```python
class SessionBasedEpisodes:
    """
    Generate training episodes from session data.
    """
    
    def generate_episode(self, session_id: str, config: EpisodeConfig) -> Iterator[Experience]:
        """
        Generate single training episode from session data.
        
        Key advantages:
        - No complex multi-market coordination
        - Natural temporal flow preserved
        - Real market dynamics and patterns
        - Variable episode lengths and market counts
        """
        
        # Load session data (all markets, all time points)
        session_data = self.session_loader.load_session(session_id)
        
        # Select temporal window for this episode
        start_idx = random.randint(0, max(0, len(session_data.data_points) - config.max_length))
        end_idx = min(start_idx + config.max_length, len(session_data.data_points))
        episode_data = session_data.data_points[start_idx:end_idx]
        
        # Initialize episode state
        positions = {}  # Market -> position
        episode_history = []
        
        for i, data_point in enumerate(episode_data):
            # Build market-agnostic observation
            obs = self._build_observation(data_point, episode_history, positions)
            
            # Yield current state for agent action
            if i > 0:  # Skip first observation
                yield Experience(
                    observation=obs,
                    available_actions=self._get_available_actions(data_point.market_states),
                    timestamp=data_point.timestamp,
                    metadata={'session_id': session_id, 'step': i}
                )
            
            episode_history.append(data_point)
    
    def _build_observation(
        self, 
        data_point: SessionDataPoint, 
        history: List[SessionDataPoint],
        positions: Dict[str, float]
    ) -> np.ndarray:
        """
        Build market-agnostic observation using shared observation builder.
        
        CRITICAL: Uses SAME function as inference actor for consistency.
        """
        
        # Extract market states (up to max_markets, random selection if more)
        active_markets = list(data_point.market_states.keys())
        if len(active_markets) > self.max_markets:
            active_markets = random.sample(active_markets, self.max_markets)
        
        market_features = []
        temporal_features = []
        
        for market in active_markets[:self.max_markets]:
            market_state = data_point.market_states[market]
            
            # Market-agnostic features (NO ticker identity exposed)
            market_feat = extract_market_agnostic_features(market_state)
            market_features.append(market_feat)
            
            # Temporal features for this market
            temp_feat = extract_temporal_features(data_point, history)
            temporal_features.append(temp_feat)
        
        # Pad to max_markets if needed
        while len(market_features) < self.max_markets:
            market_features.append(np.zeros_like(market_features[0]))
            temporal_features.append(np.zeros_like(temporal_features[0]))
        
        # Combine into single observation
        obs = np.concatenate([
            np.array(market_features).flatten(),
            np.array(temporal_features).flatten(),
            self._encode_portfolio_state(positions)
        ])
        
        return obs
```

### Primitive Action Space

#### Market-Agnostic Primitive Actions
```python
class PrimitiveActionSpace:
    """
    Minimal action space with primitive operations that allows the agent to
    discover complex strategies (like arbitrage) through learning, rather
    than having them hardcoded.
    
    Key principle: The agent should learn that buying both YES and NO when
    their sum < 95 is profitable through experience, not through pre-programmed
    "ARBITRAGE" actions.
    """
    
    def __init__(self, max_markets: int = 5):
        # Simplified primitive actions - agent learns to combine these
        self.base_actions = [
            # No action
            'HOLD',           # 0: No action - no orders placed, no commitment
            
            # Market orders (NOW) - immediate execution, take liquidity
            'BUY_YES_NOW',    # 1: Market buy YES at current ask (pay spread for immediacy)
            'SELL_YES_NOW',   # 2: Market sell YES at current bid (accept lower price for immediacy)
            'BUY_NO_NOW',     # 3: Market buy NO at current ask
            'SELL_NO_NOW',    # 4: Market sell NO at current bid
            
            # Limit orders (WAIT) - provide liquidity, wait for fill
            'BUY_YES_WAIT',   # 5: Limit buy YES at current best bid (join the queue)
            'SELL_YES_WAIT',  # 6: Limit sell YES at current best ask (join the queue)
            'BUY_NO_WAIT',    # 7: Limit buy NO at current best bid
            'SELL_NO_WAIT',   # 8: Limit sell NO at current best ask
        ]
        
        # Key distinction:
        # - HOLD: No market exposure, no orders in book, truly passive
        # - *_WAIT: Active limit orders in book, committed to trade if filled
        # - *_NOW: Immediate execution at current market prices
        
        # Multi-discrete allows simultaneous actions across markets
        # This enables the agent to discover arbitrage by taking
        # BUY_YES and BUY_NO actions on the same market
        self.action_space = spaces.MultiDiscrete([len(self.base_actions)] * max_markets)
    
    def decode_action(self, action: np.ndarray, market_states: List[Dict]) -> List[Order]:
        """
        Decode RL actions into Kalshi orders.
        
        The agent learns through rewards which combinations work:
        - Simultaneous BUY_YES_NOW + BUY_NO_NOW when sum < 95 → positive reward (arbitrage)
        - WAIT orders when spread is wide → positive reward from spread capture
        - NOW orders during momentum → positive reward from directional moves
        - HOLD when uncertain → avoid losses from bad trades
        """
        orders = []
        
        for i, market_state in enumerate(market_states[:len(action)]):
            action_idx = action[i]
            action_type = self.base_actions[action_idx]
            
            if action_type == 'HOLD':
                # No order placed - truly passive, no market commitment
                continue
                
            ticker = market_state['ticker']
            
            # Fixed size for 1.0 model (can be learned in future versions)
            DEFAULT_SIZE = 10  # contracts
            
            # NOW actions - immediate execution at market prices
            if action_type == 'BUY_YES_NOW':
                # Market order - pay the spread for immediate execution
                orders.append({
                    'ticker': ticker,
                    'action': 'buy',
                    'side': 'yes',
                    'count': DEFAULT_SIZE,
                    'type': 'market'  # Executes at current ask
                })
                
            elif action_type == 'SELL_YES_NOW':
                # Market order - accept bid for immediate execution
                orders.append({
                    'ticker': ticker,
                    'action': 'sell',
                    'side': 'yes',
                    'count': DEFAULT_SIZE,
                    'type': 'market'  # Executes at current bid
                })
                
            elif action_type == 'BUY_NO_NOW':
                orders.append({
                    'ticker': ticker,
                    'action': 'buy',
                    'side': 'no',
                    'count': DEFAULT_SIZE,
                    'type': 'market'
                })
                
            elif action_type == 'SELL_NO_NOW':
                orders.append({
                    'ticker': ticker,
                    'action': 'sell',
                    'side': 'no',
                    'count': DEFAULT_SIZE,
                    'type': 'market'
                })
                
            # WAIT actions - limit orders that provide liquidity
            elif action_type == 'BUY_YES_WAIT':
                # Limit order at current best bid - join the queue
                best_bid = market_state['yes_bids'][0][0]
                
                orders.append({
                    'ticker': ticker,
                    'action': 'buy',
                    'side': 'yes',
                    'count': DEFAULT_SIZE,
                    'price': best_bid,  # Join existing bid
                    'type': 'limit'
                })
                
            elif action_type == 'SELL_YES_WAIT':
                # Limit order at current best ask - join the queue
                best_ask = market_state['yes_asks'][0][0]
                
                orders.append({
                    'ticker': ticker,
                    'action': 'sell',
                    'side': 'yes',
                    'count': DEFAULT_SIZE,
                    'price': best_ask,  # Join existing ask
                    'type': 'limit'
                })
                
            elif action_type == 'BUY_NO_WAIT':
                best_bid = market_state['no_bids'][0][0]
                
                orders.append({
                    'ticker': ticker,
                    'action': 'buy',
                    'side': 'no',
                    'count': DEFAULT_SIZE,
                    'price': best_bid,
                    'type': 'limit'
                })
                
            elif action_type == 'SELL_NO_WAIT':
                best_ask = market_state['no_asks'][0][0]
                
                orders.append({
                    'ticker': ticker,
                    'action': 'sell',
                    'side': 'no',
                    'count': DEFAULT_SIZE,
                    'price': best_ask,
                    'type': 'limit'
                })
        
        return orders
```

#### Why Primitive Actions Enable Strategy Discovery

```python
class StrategyEmergenceExample:
    """
    Examples of how the agent discovers complex strategies from primitive actions.
    
    Key insight: HOLD vs WAIT actions have very different commitments:
    - HOLD: No orders placed, no market exposure
    - WAIT: Active limit orders in book, committed to trade if filled
    - NOW: Immediate execution, pay/receive spread
    """
    
    def arbitrage_discovery_example(self):
        """
        The agent learns arbitrage without it being hardcoded.
        """
        # Episode 1000: Random exploration
        # State: yes_ask=45, no_ask=48 (sum=93, arbitrage opportunity!)
        # Actions: [HOLD, BUY_YES_NOW, SELL_NO_WAIT, HOLD, HOLD]
        # Reward: -1.5 (random uncoordinated trades)
        
        # Episode 5000: Starts noticing pattern
        # State: yes_ask=45, no_ask=48
        # Actions: [BUY_YES_NOW, HOLD, HOLD, HOLD, HOLD]
        # Reward: +2.0 (small profit)
        
        # Episode 10000: Discovers full arbitrage
        # State: yes_ask=45, no_ask=48  
        # Actions: [BUY_YES_NOW, BUY_NO_NOW, HOLD, HOLD, HOLD]  # Same market!
        # Reward: +7.0 (guaranteed profit: 100 - 93 = 7¢ per contract)
        # 
        # Agent learns: When yes_ask + no_ask < 95, use NOW actions for immediate capture!
        
    def spread_provision_discovery(self):
        """
        The agent learns to provide liquidity in wide spreads using WAIT actions.
        """
        # State: yes_bid=40, yes_ask=45 (5¢ spread)
        # 
        # Option 1: HOLD
        # Actions: [HOLD, HOLD, HOLD, HOLD, HOLD]
        # Result: No orders placed, no profit opportunity
        # Reward: 0.0
        #
        # Option 2: BUY_YES_WAIT (join bid at 40¢)
        # Actions: [BUY_YES_WAIT, HOLD, HOLD, HOLD, HOLD]
        # Result: Places limit buy at 40¢, waits in queue
        # Reward: +2.0 if filled and price moves up
        #
        # Option 3: SELL_YES_WAIT (join ask at 45¢)  
        # Actions: [SELL_YES_WAIT, HOLD, HOLD, HOLD, HOLD]
        # Result: Places limit sell at 45¢, commits to sell if buyer appears
        # Reward: +3.0 if filled at 45¢ (assuming cost basis < 45¢)
        #
        # Agent learns: WAIT actions = active liquidity provision ≠ HOLD
        
    def momentum_vs_patience_discovery(self):
        """
        The agent learns when to use NOW (immediate) vs WAIT (patient) vs HOLD.
        """
        # Scenario 1: Fast-moving market with momentum
        # State: Price rapidly rising, YES moved from 40→45→48 in seconds
        # Actions: [BUY_YES_NOW, HOLD, HOLD, HOLD, HOLD]
        # Result: Immediate execution at 48¢, rides momentum to 52¢
        # Reward: +4.0
        # Learning: Use NOW during momentum to guarantee execution
        
        # Scenario 2: Stable market with wide spread
        # State: YES_bid=45, YES_ask=48, low volatility
        # Actions: [BUY_YES_WAIT, HOLD, HOLD, HOLD, HOLD]
        # Result: Limit order at 45¢, saves 3¢ vs market order
        # Reward: +3.0 (saved spread)
        # Learning: Use WAIT in stable markets to get better prices
        
        # Scenario 3: Uncertain/choppy market
        # State: Conflicting signals, unclear direction
        # Actions: [HOLD, HOLD, HOLD, HOLD, HOLD]
        # Result: No orders, avoids losses from false signals
        # Reward: 0.0 (better than -2.0 from bad trade)
        # Learning: HOLD when uncertain prevents losses
```

### Unified P&L and Reward System

#### Simplified Position Tracking
```python
class UnifiedPositionTracker:
    """
    Unified position tracking that works for both training and inference.
    
    Design principles:
    1. Match Kalshi's actual position structure
    2. Simple enough for training simulation
    3. Accurate enough for production trading
    4. Single source of truth for P&L calculations
    """
    
    def __init__(self):
        # Core position data matching Kalshi API structure
        self.positions = {}  # ticker -> position_data
        self.cash_balance = 10000.0  # Starting capital
        
    def update_position(self, ticker: str, fill_data: Dict) -> Dict:
        """
        Update position from a trade fill (training or production).
        
        Args:
            ticker: Market ticker
            fill_data: {
                'side': 'yes' or 'no',
                'action': 'buy' or 'sell',
                'quantity': number of contracts,
                'price': execution price in cents,
                'fee': transaction fee in dollars
            }
            
        Returns:
            Updated position state
        """
        if ticker not in self.positions:
            self.positions[ticker] = {
                'position': 0,  # +YES/-NO contracts (Kalshi convention)
                'cost_basis': 0.0,  # Total cost in dollars
                'realized_pnl': 0.0,  # Cumulative realized P&L
                'fees_paid': 0.0  # Total fees
            }
        
        pos = self.positions[ticker]
        quantity = fill_data['quantity']
        price_dollars = fill_data['price'] / 100.0
        fee = fill_data.get('fee', quantity * 0.007)  # Default 0.7¢ per contract
        
        # Update position based on Kalshi's convention:
        # Positive position = YES contracts
        # Negative position = NO contracts
        
        if fill_data['action'] == 'buy':
            if fill_data['side'] == 'yes':
                # Buying YES contracts
                new_position = pos['position'] + quantity
                
                if pos['position'] >= 0:  # Adding to YES position
                    pos['cost_basis'] += quantity * price_dollars
                elif pos['position'] + quantity <= 0:  # Closing NO position
                    # Realize P&L on closed NO contracts
                    closed_qty = min(quantity, abs(pos['position']))
                    avg_cost = pos['cost_basis'] / abs(pos['position']) if pos['position'] != 0 else 0
                    pos['realized_pnl'] += closed_qty * (avg_cost - price_dollars)
                    pos['cost_basis'] = abs(new_position) * price_dollars if new_position < 0 else 0
                else:  # Flipping from NO to YES
                    # Close all NO, open new YES
                    avg_cost = pos['cost_basis'] / abs(pos['position'])
                    pos['realized_pnl'] += abs(pos['position']) * (avg_cost - price_dollars)
                    pos['cost_basis'] = (quantity - abs(pos['position'])) * price_dollars
                    
                pos['position'] = new_position
                
            else:  # Buying NO contracts
                new_position = pos['position'] - quantity
                
                if pos['position'] <= 0:  # Adding to NO position
                    pos['cost_basis'] += quantity * price_dollars
                elif pos['position'] - quantity >= 0:  # Closing YES position
                    # Realize P&L on closed YES contracts
                    closed_qty = min(quantity, pos['position'])
                    avg_cost = pos['cost_basis'] / pos['position'] if pos['position'] != 0 else 0
                    pos['realized_pnl'] += closed_qty * (price_dollars - avg_cost)
                    pos['cost_basis'] = new_position * (pos['cost_basis'] / pos['position']) if new_position > 0 else 0
                else:  # Flipping from YES to NO
                    # Close all YES, open new NO
                    avg_cost = pos['cost_basis'] / pos['position']
                    pos['realized_pnl'] += pos['position'] * (price_dollars - avg_cost)
                    pos['cost_basis'] = (quantity - pos['position']) * price_dollars
                    
                pos['position'] = new_position
                
            # Deduct cost from cash
            self.cash_balance -= (quantity * price_dollars + fee)
            
        else:  # sell
            if fill_data['side'] == 'yes' and pos['position'] > 0:
                # Selling YES contracts
                sell_qty = min(quantity, pos['position'])
                avg_cost = pos['cost_basis'] / pos['position']
                pos['realized_pnl'] += sell_qty * (price_dollars - avg_cost)
                pos['position'] -= sell_qty
                pos['cost_basis'] = pos['position'] * avg_cost if pos['position'] > 0 else 0
                self.cash_balance += (sell_qty * price_dollars - fee)
                
            elif fill_data['side'] == 'no' and pos['position'] < 0:
                # Selling NO contracts (closing position)
                sell_qty = min(quantity, abs(pos['position']))
                avg_cost = pos['cost_basis'] / abs(pos['position'])
                pos['realized_pnl'] += sell_qty * (avg_cost - price_dollars)
                pos['position'] += sell_qty
                pos['cost_basis'] = abs(pos['position']) * avg_cost if pos['position'] < 0 else 0
                self.cash_balance += (sell_qty * price_dollars - fee)
        
        pos['fees_paid'] += fee
        return pos
    
    def calculate_unrealized_pnl(self, ticker: str, yes_mid: float, no_mid: float) -> float:
        """
        Calculate unrealized P&L for a position.
        
        Args:
            ticker: Market ticker
            yes_mid: Current YES mid price in cents
            no_mid: Current NO mid price in cents
            
        Returns:
            Unrealized P&L in dollars
        """
        if ticker not in self.positions or self.positions[ticker]['position'] == 0:
            return 0.0
            
        pos = self.positions[ticker]
        contracts = pos['position']
        
        if contracts != 0:
            avg_cost_per_contract = pos['cost_basis'] / abs(contracts)
            
            if contracts > 0:  # YES position
                current_value = contracts * (yes_mid / 100.0)
                unrealized = current_value - pos['cost_basis']
            else:  # NO position
                current_value = abs(contracts) * (no_mid / 100.0)
                unrealized = pos['cost_basis'] - current_value  # NO profits when price falls
        else:
            unrealized = 0.0
            
        return unrealized
    
    def get_total_portfolio_value(self, market_prices: Dict[str, Dict]) -> float:
        """
        Calculate total portfolio value including cash and all positions.
        
        Args:
            market_prices: {ticker: {'yes_mid': cents, 'no_mid': cents}}
            
        Returns:
            Total portfolio value in dollars
        """
        total_unrealized = 0.0
        
        for ticker, pos in self.positions.items():
            if ticker in market_prices and pos['position'] != 0:
                prices = market_prices[ticker]
                total_unrealized += self.calculate_unrealized_pnl(
                    ticker, 
                    prices['yes_mid'], 
                    prices['no_mid']
                )
        
        # Portfolio = cash + cost basis of all positions + unrealized P&L
        position_value = sum(p['cost_basis'] for p in self.positions.values())
        return self.cash_balance + position_value + total_unrealized
    
    def update_from_kalshi_api(self, api_positions: List[Dict]) -> None:
        """
        Update positions from Kalshi API response (inference only).
        
        Args:
            api_positions: List of positions from GET /portfolio/positions
        """
        for api_pos in api_positions:
            ticker = api_pos['ticker']
            self.positions[ticker] = {
                'position': api_pos['position'],  # Already in Kalshi format
                'cost_basis': api_pos['market_exposure'] / 100.0,  # cents to dollars
                'realized_pnl': api_pos['realized_pnl'] / 100.0,
                'fees_paid': api_pos['fees_paid'] / 100.0
            }
    
    def update_from_websocket(self, ws_msg: Dict) -> None:
        """
        Update from Kalshi WebSocket position update (inference only).
        
        IMPORTANT: WebSocket uses centi-cents (1/10,000 dollar)!
        """
        ticker = ws_msg['market_ticker']
        self.positions[ticker] = {
            'position': ws_msg['position'],
            'cost_basis': ws_msg['position_cost'] / 10000.0,  # centi-cents to dollars!
            'realized_pnl': ws_msg['realized_pnl'] / 10000.0,
            'fees_paid': ws_msg['fees_paid'] / 10000.0
        }
```

#### Simplified Reward Calculator
```python
class UnifiedRewardCalculator:
    """
    Simple, clear reward calculation for both training and evaluation.
    
    Core principle: Reward = change in portfolio value
    This naturally captures everything that matters:
    - Realized P&L from trades
    - Unrealized P&L changes  
    - Transaction costs (fees)
    - Opportunity costs (cash not deployed)
    """
    
    def __init__(self, reward_scale: float = 0.01):
        """
        Initialize reward calculator.
        
        Args:
            reward_scale: Scale factor to map dollar changes to RL-friendly range
                         Default 0.01 means $100 change = 1.0 reward
        """
        self.reward_scale = reward_scale
        self.last_portfolio_value = None
        
    def calculate_step_reward(
        self,
        portfolio_value: float,
        trades_executed: int = 0
    ) -> float:
        """
        Calculate reward for one environment step.
        
        Args:
            portfolio_value: Current total portfolio value
            trades_executed: Number of trades in this step (optional penalty)
            
        Returns:
            Scaled reward value
        """
        if self.last_portfolio_value is None:
            self.last_portfolio_value = portfolio_value
            return 0.0
        
        # Core reward: change in portfolio value
        value_change = portfolio_value - self.last_portfolio_value
        reward = value_change * self.reward_scale
        
        # Optional: Tiny penalty for excessive trading
        # (but fees already provide this signal naturally)
        if trades_executed > 5:
            reward -= 0.001 * (trades_executed - 5)
        
        # Store for next step
        self.last_portfolio_value = portfolio_value
        
        # Clip to prevent instability
        return np.clip(reward, -2.0, 2.0)
    
    def calculate_episode_metrics(
        self,
        initial_value: float,
        final_value: float,
        total_trades: int,
        episode_length: int
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for episode analysis.
        
        Returns metrics useful for understanding agent performance.
        """
        total_return = final_value - initial_value
        return_pct = (total_return / initial_value) * 100
        
        # Annualized Sharpe approximation (simplified)
        # Assumes episode represents one day of trading
        daily_return = total_return / initial_value
        sharpe_approx = daily_return * np.sqrt(252) / 0.02  # Assume 2% daily vol
        
        return {
            'total_return': total_return,
            'return_pct': return_pct,
            'sharpe_approx': sharpe_approx,
            'trades_per_step': total_trades / episode_length,
            'final_value': final_value
        }
```

#### Integration Example
```python
class UnifiedTradingSystem:
    """
    Example showing how the unified system works in both contexts.
    """
    
    def __init__(self, mode: str = 'training'):
        self.mode = mode
        self.position_tracker = UnifiedPositionTracker()
        self.reward_calculator = UnifiedRewardCalculator()
        
    def process_trade(self, trade_data: Dict) -> float:
        """
        Process a trade and return reward (works for both training and inference).
        """
        # Update position
        self.position_tracker.update_position(
            trade_data['ticker'],
            trade_data
        )
        
        # Get current portfolio value
        market_prices = self._get_current_prices()  # From env or API
        portfolio_value = self.position_tracker.get_total_portfolio_value(market_prices)
        
        # Calculate reward
        reward = self.reward_calculator.calculate_step_reward(
            portfolio_value,
            trades_executed=1
        )
        
        return reward
    
    def sync_with_kalshi(self) -> None:
        """
        Sync with real Kalshi positions (inference only).
        """
        if self.mode == 'inference':
            # Get positions from API
            positions = self.kalshi_client.get_positions()
            self.position_tracker.update_from_kalshi_api(positions)
            
            # Or from WebSocket
            ws_update = await self.kalshi_ws.receive()
            self.position_tracker.update_from_websocket(ws_update)
```

## Implementation Strategy: Clean Slate Approach

### Recommended Approach: Delete First, Then Build Fresh

After careful analysis, the **optimal strategy** is to DELETE the old system first, then build fresh from scratch. This avoids:
1. Wasting time fixing obsolete tests
2. Temptation to reference broken old code
3. Getting bogged down in legacy code refactoring
4. Confusion about which implementation is active
5. Accidentally importing old broken modules

### Step-by-Step Implementation Plan

#### Step 1: DELETE FIRST - Clean Slate (Day 1)

**SURGICAL deletion - only remove old environment code, NOT orderbook collection:**

```bash
# Delete ONLY old RL environment code (NOT orderbook collector!)
rm -rf backend/src/kalshiflow_rl/environments/
rm backend/src/kalshiflow_rl/trading/trading_metrics.py

# Delete ONLY old environment tests (preserve working orderbook tests!)
# BE SPECIFIC - only delete environment-related tests
rm backend/tests/test_rl/test_kalshi_env.py
rm backend/tests/test_rl/test_observation_space.py
rm backend/tests/test_rl/test_reward_functions.py
rm backend/tests/test_rl/test_multi_market.py
rm backend/tests/test_rl/test_feature_extraction.py
rm backend/tests/test_rl/test_performance.py

# KEEP THESE WORKING TESTS:
# ✅ backend/tests/test_rl/test_rl_orderbook_e2e.py - orderbook collection works!
# ✅ backend/tests/test_rl/test_orderbook_*.py - any orderbook tests
# ✅ backend/tests/test_rl/test_historical_data_loader.py - data loading works!

# Create organized test directories for new implementation
mkdir -p backend/tests/test_rl/environment
mkdir -p backend/tests/test_rl/training

# Delete old training/eval scripts (these are definitely obsolete)
rm backend/scripts/train_rl_agent.py
rm backend/scripts/evaluate_agent.py

# Commit this SURGICAL deletion
git add -A
git commit -m "chore: Delete old RL environment (keep working orderbook collection)

- Removed market-specific environment (wrong approach)
- Removed complex reward system (overcomplicated)
- Removed environment tests that validated incorrect behavior
- KEPT orderbook collection (working correctly)
- KEPT orderbook tests (test_rl_orderbook_e2e.py, etc.)
- Clean slate for market-agnostic environment"
```

**Why delete FIRST?**
1. **No temptation** to look at old code
2. **No confusion** about which version to use
3. **No accidental imports** of old modules
4. **Forces fresh thinking** about the problem
5. **Tests can't accidentally pass** for wrong reasons

#### Step 2: Build Fresh Structure (Day 2-3)
```bash
backend/src/kalshiflow_rl/
├── environments/              # NEW - Build fresh here
│   ├── __init__.py
│   ├── market_agnostic_env.py
│   ├── session_data_loader.py
│   ├── feature_extractors.py
│   └── unified_metrics.py
└── trading/
    └── unified_metrics.py     # NEW - Build fresh here
```

**Now building in the MAIN location (no version suffix):**
- No parallel structures to manage
- No migration step needed later
- Clear that this IS the implementation

#### Step 3: Build Core Components (Day 4-5)

**Build fresh with no legacy thinking:**

```python
# environments/market_agnostic_env.py
class MarketAgnosticKalshiEnv(gym.Env):
    """Fresh implementation - no legacy code to reference."""
    pass

# trading/unified_metrics.py  
class UnifiedPositionTracker:
    """Kalshi-aligned position tracking from scratch."""
    pass

class UnifiedRewardCalculator:
    """Simple reward = portfolio value change."""
    pass

# environments/session_data_loader.py
class SessionDataLoader:
    """Load from rl_orderbook tables by session_id."""
    pass
```

**Benefits of deleting first:**
- ✅ Can't accidentally import old modules (they don't exist!)
- ✅ Can't be tempted to "just look" at old implementation
- ✅ Fresh mental model without legacy assumptions
- ✅ No confusion about which version is active

#### Step 4: Write Minimal New Tests (Day 6)

**Build tests in a properly organized structure:**

```bash
backend/tests/test_rl/
├── environment/                    # NEW environment tests
│   ├── __init__.py
│   ├── test_market_agnostic.py    # Test new environment behavior
│   ├── test_unified_metrics.py    # Test unified metrics
│   └── test_session_loader.py     # Test session data loading
├── training/                       # Training infrastructure tests
│   ├── __init__.py
│   └── test_training_pipeline.py
└── orderbook/                      # KEEP existing working tests
    └── test_rl_orderbook_e2e.py
```

**Configure pytest to run new tests:**

```bash
# backend/tests/test_rl/environment/conftest.py
"""
Environment Test Configuration

These tests validate the NEW correct behavior.
Old code has been deleted - there's nothing to compare against.
"""

# Run with:
pytest tests/test_rl/environment/  # Environment tests
pytest tests/test_rl/  # All RL tests
```

**New Test Philosophy - Start from ZERO assumptions:**

```python
# test_rl/environment/test_market_agnostic.py
"""
Market-Agnostic Environment Tests

IMPORTANT: These tests assume NOTHING from old implementation.
Old environment never worked correctly. Its tests are irrelevant.
We're testing the NEW correct behavior only.
"""

def test_market_agnostic_env_basic():
    """Can we create env and run one episode?"""
    env = MarketAgnosticKalshiEnv(session_id="test_session")
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert obs is not None

def test_position_convention():
    """Verify Kalshi position convention: +YES, -NO"""
    tracker = UnifiedPositionTracker()
    
    # This is the CORRECT behavior we want
    # Old implementation had it wrong with separate yes/no positions
    tracker.update_position("TEST", {
        'side': 'yes', 'action': 'buy', 
        'quantity': 10, 'price': 50
    })
    assert tracker.positions["TEST"]['position'] == 10  # YES = positive
    
    tracker.update_position("TEST", {
        'side': 'no', 'action': 'buy',
        'quantity': 5, 'price': 50  
    })
    assert tracker.positions["TEST"]['position'] == 5  # Reduced YES position

def test_reward_is_simple():
    """Reward = portfolio value change ONLY."""
    calc = UnifiedRewardCalculator()
    
    # Old implementation had complex penalties - we don't want those
    # This tests our NEW simple approach
    reward = calc.calculate_step_reward(10100, trades_executed=0)  # $100 gain
    assert 0.9 < reward < 1.1  # Roughly scaled by 0.01
    
    # No complex diversification bonuses
    # No pattern discovery bonuses
    # Just pure value change
```

**Don't waste time on:**
- ❌ Comparing new behavior to old tests
- ❌ Making new env pass any old test
- ❌ Looking at old tests for "inspiration"
- ❌ Comprehensive test coverage initially

#### Step 5: Validation Checkpoint (Day 7)

**Quick validation of the fresh implementation:**

```python
# scripts/validate_new_environment.py
"""
Validate the new implementation works end-to-end.
No comparison to old code needed - it's already deleted!
"""
def validate_new_env():
    # 1. Can we create the environment?
    env = MarketAgnosticKalshiEnv(session_id="test")
    
    # 2. Can we run 100 episodes?
    for _ in range(100):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
    
    # 3. Can we train a simple agent?
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)
    
    print("✅ New environment validated!")

if __name__ == "__main__":
    validate_new_env()
```

#### Step 6: Build Forward (Day 8+)

**Now with clean codebase:**
- Write new tests for actual 1.0 requirements
- Add features incrementally
- No legacy baggage

### Critical Success Factors

#### DO ✅
- Delete old system completely before building
- Use completely new class names initially
- Test only core functionality
- Delete old code aggressively once validated
- Start training with simple baseline

#### DON'T ❌
- Try to maintain backward compatibility
- Update old tests for new behavior  
- Refactor old code incrementally
- Keep old code "just in case"
- Write comprehensive tests before proving it works

### Risk Mitigation

**If something goes wrong:**
```bash
# Everything is in git, so worst case:
git checkout main -- backend/src/kalshiflow_rl/environments/
git checkout main -- backend/src/kalshiflow_rl/trading/
git checkout main -- tests/
```

But commit the deletion as a clear break:
```bash
git add -A
git commit -m "feat: Replace market-specific env with market-agnostic environment

BREAKING CHANGE: Complete environment rewrite for 1.0 model
- Deleted old multi-market environment
- Deleted complex reward functions  
- New session-based architecture
- Unified position tracking matching Kalshi API"
```

### Timeline Summary

- **Day 1**: DELETE all old RL code and tests (clean slate)
- **Days 2-3**: Create fresh structure in main location
- **Days 4-5**: Build new components from scratch
- **Day 6**: Write minimal tests in tests/test_rl/environment/
- **Day 7**: Validate new environment works end-to-end
- **Day 8+**: Iterate on clean codebase

**Total: 7 days to clean implementation vs weeks of refactoring**

### Why "Delete First" is Superior

1. **Psychological Clean Break**: Can't reference old code that doesn't exist
2. **No Parallel Confusion**: Only one implementation at any time
3. **Forced Innovation**: Must solve problems fresh, not copy old patterns
4. **No Migration Step**: Building directly in final location
5. **Clear Git History**: Deletion commit marks clean break from old approach

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
**Goal**: Build session-based data infrastructure and market-agnostic observation space

#### Milestone 1.1: Session Data Infrastructure
- [ ] `SessionDataLoader` class for database integration
- [ ] `SessionDataPoint` structure with multi-market support
- [ ] Database queries optimized for session-based loading
- [ ] Session metadata management and filtering

#### Milestone 1.2: Market-Agnostic Observation Space
- [ ] `extract_market_agnostic_features()` universal feature engineering
- [ ] `extract_temporal_features()` for temporal pattern discovery  
- [ ] Updated observation space supporting 1-N markets per episode
- [ ] Feature consistency validation between training and inference

### Phase 2: Environment Core (Week 2)
**Goal**: Complete environment rewrite with session-based architecture

#### Milestone 2.1: New Environment Architecture
- [ ] `MarketAgnosticKalshiEnv` class replacing current implementation
- [ ] Session-based episode generation without market coordination complexity
- [ ] Market-agnostic action space with dynamic market scaling
- [ ] Integration with existing SB3 training infrastructure

#### Milestone 2.2: Temporal Pattern Discovery
- [ ] Advanced temporal feature engineering (time gaps, activity clustering)
- [ ] Sequence pattern detection capabilities
- [ ] Regime detection and classification
- [ ] Pattern discovery reward components

### Phase 3: Training Strategy (Week 3)
**Goal**: Implement market-agnostic training pipeline with curriculum learning

#### Milestone 3.1: Curriculum Learning
- [ ] Session categorization by complexity and market diversity
- [ ] Progressive curriculum from single-market to multi-market sessions
- [ ] Training session selection algorithms
- [ ] Validation strategy with held-out markets

#### Milestone 3.2: Advanced Reward Engineering  
- [ ] Multi-tier reward structure (Tiers 1-4)
- [ ] Pattern discovery detection and bonuses
- [ ] Risk-adjusted return calculations
- [ ] Performance baseline establishment

### Phase 4: Integration and Testing (Week 4)
**Goal**: Integrate with existing infrastructure and validate performance

#### Milestone 4.1: Infrastructure Integration
- [ ] Integration with existing actor/inference pipeline
- [ ] Hot-reload protocol updates for new model architecture
- [ ] Database schema updates for session-based metadata
- [ ] Performance optimization and memory management

#### Milestone 4.2: Validation and Testing
- [ ] Comprehensive test suite for new environment
- [ ] Training performance validation (Sharpe ratio targets)
- [ ] Cross-market generalization testing
- [ ] Production readiness validation

## Code Examples

### Session-Based Environment Initialization
```python
# Old approach - market-specific
env = KalshiTradingEnv(
    market_tickers=["TRUMP-2024", "FED-HIKE-DEC"],
    observation_config=config,
    episode_length=1000
)

# New approach - session-based, market-agnostic
env = MarketAgnosticKalshiEnv(
    session_config=SessionConfig(
        session_pool=["session_001", "session_002", "session_003"],
        max_markets=5,
        episode_length=1000,
        temporal_features=True
    )
)
```

### Market-Agnostic Training Loop
```python
def train_market_agnostic_model():
    """
    Training loop supporting market-agnostic learning.
    """
    
    # Environment with session-based episodes
    env = MarketAgnosticKalshiEnv(session_config=config)
    
    # Model never sees market identities
    model = PPO('MlpPolicy', env)
    
    # Curriculum learning with session progression
    curriculum = SessionBasedCurriculum()
    
    for episode in range(100000):
        # Select session based on curriculum
        session_id = curriculum.select_training_session(episode)
        
        # Update environment for this episode
        env.set_session(session_id)
        
        # Train on this session's data
        model.learn(total_timesteps=1000)
        
        # Validate on held-out sessions periodically
        if episode % 1000 == 0:
            validate_on_unseen_sessions(model)
```

### Feature Extraction Example
```python
# Market-agnostic features (works on ANY Kalshi market)
def build_observation_from_session_data(
    data_point: SessionDataPoint,
    history: List[SessionDataPoint],
    positions: Dict[str, float]
) -> np.ndarray:
    """
    Build observation that works identically across all markets.
    """
    
    features = []
    
    # Process each active market (up to max_markets)
    active_markets = list(data_point.market_states.keys())[:MAX_MARKETS]
    
    for market_ticker in active_markets:
        market_state = data_point.market_states[market_ticker]
        
        # Universal features (NO market-specific logic)
        market_features = extract_market_agnostic_features(market_state)
        temporal_features = extract_temporal_features(data_point, history)
        
        features.extend(market_features)
        features.extend(temporal_features)
    
    # Pad to fixed size if fewer than max_markets active
    while len(features) < EXPECTED_FEATURE_SIZE:
        features.extend([0.0] * FEATURES_PER_MARKET)
    
    # Add portfolio state (market-agnostic position info)
    portfolio_features = encode_portfolio_state(positions, active_markets)
    features.extend(portfolio_features)
    
    return np.array(features, dtype=np.float32)
```

## Testing Strategy

### Unit Testing
```python
class TestMarketAgnosticEnvironment:
    """
    Comprehensive testing for new environment architecture.
    """
    
    def test_session_data_loading(self):
        """Test session data loading and temporal consistency."""
        loader = SessionDataLoader()
        session_data = loader.load_session("test_session_001")
        
        assert len(session_data.data_points) > 0
        assert all(dp.timestamp for dp in session_data.data_points)
        assert session_data.data_points == sorted(session_data.data_points, key=lambda x: x.timestamp)
    
    def test_market_agnostic_features(self):
        """Test that features work identically across different markets."""
        # Test with political market
        political_market = create_test_orderbook("TRUMP-2024", yes_mid=65)
        political_features = extract_market_agnostic_features(political_market)
        
        # Test with economic market at same probability 
        economic_market = create_test_orderbook("FED-HIKE-DEC", yes_mid=65)
        economic_features = extract_market_agnostic_features(economic_market)
        
        # Features should be identical for same probability levels
        np.testing.assert_array_equal(political_features, economic_features)
    
    def test_observation_consistency(self):
        """Test observation consistency between training and inference."""
        # Build observation using environment method
        env = MarketAgnosticKalshiEnv(session_config=test_config)
        env_obs = env._build_observation(test_data_point, test_history, test_positions)
        
        # Build observation using shared function (used by inference actor)
        shared_obs = build_observation_from_session_data(test_data_point, test_history, test_positions)
        
        # Must be identical
        np.testing.assert_array_equal(env_obs, shared_obs)
```

### Integration Testing
```python
def test_end_to_end_training():
    """Test complete training pipeline with market-agnostic approach."""
    
    # Create environment with multiple diverse sessions
    env = MarketAgnosticKalshiEnv(
        session_config=SessionConfig(
            session_pool=["political_session", "economic_session", "sports_session"],
            max_markets=3
        )
    )
    
    # Train model on mixed session data
    model = PPO('MlpPolicy', env)
    model.learn(total_timesteps=10000)
    
    # Test generalization on completely unseen session types  
    test_sessions = ["weather_session", "crypto_session"]
    for session_id in test_sessions:
        env.set_session(session_id) 
        test_rewards = test_model_performance(model, env, episodes=10)
        assert np.mean(test_rewards) > baseline_performance
```

## Success Metrics

### Training Performance Targets
1. **Cross-Market Sharpe Ratio**: >1.0 on unseen markets
2. **Training Efficiency**: <2 hours for 50,000 episodes  
3. **Memory Usage**: <4GB for complete session data loading
4. **Observation Consistency**: 100% identical features between training/inference

### Pattern Discovery Validation
1. **Tier 2 Strategy Discovery**: Demonstrate temporal spread mean reversion
2. **Tier 3 Pattern Recognition**: Multi-level order imbalance exploitation  
3. **Market Generalization**: >70% performance retention on unseen markets
4. **Temporal Pattern Exploitation**: Evidence of burst/quiet period strategies

### Architecture Quality Metrics  
1. **Code Simplicity**: <5,000 lines for complete environment (vs 15,000+ current)
2. **Test Coverage**: >95% code coverage with comprehensive integration tests
3. **Performance**: Sub-millisecond step/reset operations maintained
4. **Maintainability**: Clear separation of concerns and minimal dependencies

## Implementation Strategy

### Step 1: Complete Deletion (Day 1)
- Delete ALL old environment code (`backend/src/kalshiflow_rl/environments/`)
- Delete ALL old environment tests (but KEEP working orderbook tests)
- Delete old training scripts that reference old environment
- Commit the deletion as a clear break from the past
- No temptation to look at or reference broken code

### Step 2: Build Fresh (Days 2-7)
- Build new environment directly in main location
- Write new tests in properly organized structure (`test_rl/environment/`)
- No parallel implementations to manage
- No migration step needed - this IS the implementation
- Fresh thinking without legacy assumptions

### Step 3: Validate and Ship (Day 8+)
- Validate new implementation works end-to-end
- No comparison to old code needed (it's deleted!)
- Train simple baseline model to verify functionality
- Ship when core functionality is proven
- Iterate on clean codebase going forward

### Risk Mitigation
- Everything is in git if rollback needed: `git checkout main -- backend/src/kalshiflow_rl/environments/`
- But commit deletion as clear break to prevent confusion
- Start with minimal functionality and build up
- Test core behaviors only initially
- Add complexity incrementally on solid foundation

## Conclusion

The session-based, market-agnostic environment rewrite represents a fundamental architectural improvement that aligns our implementation with the sophisticated theoretical framework outlined in `kalshi-theory.md`. By leveraging session_ids for data continuity and eliminating market-specific complexity, we create a cleaner, more powerful foundation for achieving Sharpe ratios >1.0 through pattern discovery rather than simple arbitrage.

The new architecture supports:
- **True market-agnostic learning** with single models trading all markets
- **Sophisticated pattern discovery** through temporal-aware feature engineering  
- **Production-ready simplicity** with session-based episode generation
- **Extensible design** supporting advanced RL strategies and research directions

**Implementation Timeline**: 4 weeks to complete rewrite with comprehensive testing and validation.

**Expected Outcome**: Production-ready environment capable of discovering Tier 3/4 orderbook strategies and achieving consistent cross-market performance through universal pattern recognition.