"""
Example showing what a training episode and step looks like in the new RL environment.

This demonstrates the flow from session data → episode → individual steps.
"""

# ============================================================
# EPISODE STRUCTURE: One Session = One Episode
# ============================================================

def example_episode_flow():
    """
    Shows what happens during a training episode.
    
    A single session (e.g., session_6) becomes one complete episode
    with 621 timesteps of data across 300 markets.
    """
    
    # 1. EPISODE INITIALIZATION (env.reset())
    episode_start = {
        "session_id": 6,
        "markets": 300,  # 300 different Kalshi markets
        "timesteps": 621,  # 621 sequential observations
        "duration": "33+ minutes of real trading data",
        "initial_cash": 1000.0,
        "initial_positions": {}  # No positions at start
    }
    
    # 2. INITIAL OBSERVATION (what agent sees at timestep 0)
    initial_observation = {
        # Market-agnostic features (model never sees ticker names)
        "market_0": {
            "yes_bid": 0.45,    # Normalized price [0,1]
            "yes_ask": 0.47,    # Normalized price [0,1]
            "no_bid": 0.52,     # Normalized price [0,1]
            "no_ask": 0.54,     # Normalized price [0,1]
            "yes_spread": 0.02,  # Normalized spread
            "no_spread": 0.02,   # Normalized spread
            "yes_depth": 0.8,    # Normalized depth indicator
            "no_depth": 0.7,     # Normalized depth indicator
            "imbalance": 0.15,   # Buy/sell imbalance [-1,1]
        },
        "market_1": {
            # Similar features for market 1
            "yes_bid": 0.72,
            "yes_ask": 0.74,
            "no_bid": 0.25,
            "no_ask": 0.27,
            # ... etc
        },
        # ... up to 300 markets
        
        # Temporal features
        "time_gap": 0.0,  # First timestep
        "activity_score": 0.3,  # Current activity level [0,1]
        "momentum": 0.05,  # Price momentum [-1,1]
        
        # Portfolio state
        "cash_remaining": 1.0,  # Normalized (1000/1000)
        "positions": [0, 0, 0, 0, 0],  # No positions yet
    }
    
    # 3. AGENT TAKES ACTION
    action = [1, 2, 0, 1, 0]  # MultiDiscrete action
    # Meaning:
    # - Market 0: BUY_YES (action 1)
    # - Market 1: BUY_NO (action 2)  
    # - Market 2: HOLD (action 0)
    # - Market 3: BUY_YES (action 1)
    # - Market 4: HOLD (action 0)
    
    # 4. ENVIRONMENT STEP (env.step(action))
    next_timestep = 1  # Move to next data point in session
    
    # 5. NEXT OBSERVATION (timestep 1)
    next_observation = {
        # Market states at next timestamp (1.5 seconds later)
        "market_0": {
            "yes_bid": 0.46,    # Price moved up
            "yes_ask": 0.48,
            "no_bid": 0.51,     # Price moved down
            "no_ask": 0.53,
            # ... features updated from new orderbook state
        },
        
        # Temporal features show time progression
        "time_gap": 1.5,  # 1.5 seconds since last update
        "activity_score": 0.45,  # Activity increased
        "momentum": 0.08,  # Momentum building
        
        # Portfolio updated after trades
        "cash_remaining": 0.82,  # Spent 180 on trades
        "positions": [10, -15, 0, 5, 0],  # New positions
        "unrealized_pnl": 2.5,  # Small profit already
    }
    
    # 6. REWARD CALCULATION
    reward = 2.5  # Portfolio value change (1002.5 - 1000.0)
    # Simple reward = portfolio value change only
    # No complex reward engineering needed
    
    # 7. EPISODE CONTINUATION
    done = False  # Episode continues for 621 timesteps
    info = {
        "step": 1,
        "steps_remaining": 620,
        "portfolio_value": 1002.5,
        "trades_executed": 4,
        "markets_with_positions": 3
    }
    
    return next_observation, reward, done, info


def example_full_episode():
    """
    Shows complete episode flow from start to finish.
    """
    
    episode_timeline = """
    EPISODE FLOW (Session 6 → Training Episode):
    
    Step 0:   [Start] Cash=$1000, No positions
              Markets: 300 active (elections, sports, economics, etc.)
              Action: Agent explores with small trades
              
    Step 50:  [Early] Cash=$750, 15 positions across 8 markets
              Detecting arbitrage opportunities
              Activity burst detected in election markets
              
    Step 200: [Mid] Cash=$400, 45 positions across 25 markets
              Agent learning spread provision strategies
              Quiet period in sports markets
              
    Step 400: [Late] Cash=$200, 72 positions across 40 markets
              Complex multi-market strategies emerging
              Portfolio value: $1,150 (15% profit)
              
    Step 621: [End] Final portfolio value: $1,087
              Total reward: +87 (8.7% return)
              Episode complete
    """
    
    return episode_timeline


def example_observation_features():
    """
    Details the market-agnostic features in observations.
    """
    
    features_per_market = {
        # Price features (normalized to [0,1])
        "yes_bid": "Best YES bid price (0.45 = 45¢)",
        "yes_ask": "Best YES ask price (0.47 = 47¢)",
        "no_bid": "Best NO bid price (0.52 = 52¢)",
        "no_ask": "Best NO ask price (0.54 = 54¢)",
        
        # Spread features
        "yes_spread": "YES bid-ask spread (normalized)",
        "no_spread": "NO bid-ask spread (normalized)",
        "arbitrage_gap": "YES_ask + NO_ask - 1.0 (arb opportunity)",
        
        # Depth features
        "yes_bid_depth": "YES bid side liquidity [0,1]",
        "yes_ask_depth": "YES ask side liquidity [0,1]",
        "no_bid_depth": "NO bid side liquidity [0,1]",
        "no_ask_depth": "NO ask side liquidity [0,1]",
        
        # Imbalance features
        "yes_imbalance": "YES buy/sell pressure [-1,1]",
        "no_imbalance": "NO buy/sell pressure [-1,1]",
        "cross_imbalance": "YES vs NO imbalance [-1,1]",
    }
    
    temporal_features = {
        "time_gap": "Seconds since last update",
        "activity_score": "Recent trading activity [0,1]",
        "momentum": "Price movement direction [-1,1]",
        "burst_indicator": "In activity burst? [0,1]",
        "quiet_indicator": "In quiet period? [0,1]",
    }
    
    portfolio_features = {
        "cash_remaining": "Normalized cash (current/initial)",
        "position_yes": "YES position for this market",
        "position_no": "NO position for this market",
        "position_value": "Current value of position",
        "unrealized_pnl": "Unrealized P&L for position",
    }
    
    return features_per_market, temporal_features, portfolio_features


def example_action_space():
    """
    Shows the primitive action space design.
    """
    
    actions_per_market = {
        0: "HOLD - Do nothing",
        1: "BUY_YES - Place YES order at current ask",
        2: "BUY_NO - Place NO order at current ask",
        
        # Agent discovers strategies through these primitives:
        # - Arbitrage: BUY_YES + BUY_NO when sum < $1
        # - Spread provision: Alternate between markets
        # - Momentum: Follow activity bursts
    }
    
    multi_market_action = """
    Action shape: [3, 3, 3, 3, 3] for 5 markets
    
    Example action: [1, 2, 0, 1, 0]
    - Market 0: BUY_YES
    - Market 1: BUY_NO (maybe arbitrage with market 0?)
    - Market 2: HOLD
    - Market 3: BUY_YES  
    - Market 4: HOLD
    
    The agent learns to coordinate actions across markets!
    """
    
    return actions_per_market, multi_market_action


if __name__ == "__main__":
    # Show the episode flow
    print("=" * 60)
    print("RL TRAINING EPISODE STRUCTURE")
    print("=" * 60)
    
    # Basic flow
    obs, reward, done, info = example_episode_flow()
    print("\n1. BASIC STEP FLOW:")
    print(f"   Observation shape: ~{15 * 300} features (15 per market × 300 markets)")
    print(f"   Action: MultiDiscrete with 3 choices per market")
    print(f"   Reward: {reward} (portfolio value change)")
    print(f"   Done: {done} (continues for 621 steps)")
    
    # Full episode
    timeline = example_full_episode()
    print("\n2. FULL EPISODE TIMELINE:")
    print(timeline)
    
    # Feature details
    market_feats, temp_feats, port_feats = example_observation_features()
    print("\n3. OBSERVATION FEATURES:")
    print(f"   Per-market features: {len(market_feats)}")
    print(f"   Temporal features: {len(temp_feats)}")
    print(f"   Portfolio features: {len(port_feats)}")
    print(f"   Total features: ~{(len(market_feats) + len(port_feats)) * 300 + len(temp_feats)}")
    
    # Action space
    actions, multi_action = example_action_space()
    print("\n4. ACTION SPACE:")
    print(f"   Actions per market: {len(actions)} choices")
    print(f"   Total action space: 3^300 possible combinations")
    print("   Agent learns to discover arbitrage, spread provision, etc.")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT: One session = One complete episode")
    print("The agent experiences 621 timesteps of real market evolution")
    print("Learning from actual market dynamics, not synthetic data!")
    print("=" * 60)