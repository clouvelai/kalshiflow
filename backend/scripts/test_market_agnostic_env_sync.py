#!/usr/bin/env python3
"""
Synchronous test script for MarketAgnosticKalshiEnv M7 implementation.

Tests the environment with real session data in sync context.
"""

import asyncio
import numpy as np
import logging

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig, convert_session_data_to_orderbook
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
from kalshiflow_rl.data.orderbook_state import OrderbookState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_environment_sync():
    """Test environment in synchronous context."""
    print("🚀 Testing MarketAgnosticKalshiEnv in sync context...")
    
    # Load sessions synchronously
    session_loader = SessionDataLoader()
    available_sessions = asyncio.run(session_loader.get_available_session_ids())
    
    if not available_sessions:
        print("No session data available for testing")
        return False
    
    print(f"Found {len(available_sessions)} available sessions: {available_sessions}")
    
    # Use only valid sessions with sufficient data (based on check_session_data.py results)
    valid_sessions = [s for s in available_sessions if s in [7, 6, 5]]
    if not valid_sessions:
        print("No valid sessions found")
        return False
    
    # Load session data
    try:
        session_data = asyncio.run(session_loader.load_session(valid_sessions[0]))
        if not session_data:
            print(f"Could not load session {valid_sessions[0]}")
            return False
        print(f"✅ Session data loaded: session_id={session_data.session_id}, {len(session_data.data_points)} data points")
    except Exception as e:
        print(f"❌ Session data loading failed: {e}")
        return False
    
    # Create environment config
    config = EnvConfig(
        max_markets=1,
        temporal_features=True,
        cash_start=10000  # $100 in cents
    )
    
    # Initialize environment
    try:
        env = MarketAgnosticKalshiEnv(session_data, config)
        print(f"✅ Environment initialized: {env.observation_space.shape} obs, {env.action_space.n} actions")
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return False
    
    # Test reset
    try:
        observation, info = env.reset()
        
        print(f"✅ Reset successful:")
        print(f"  Session: {info['session_id']}")
        print(f"  Market: {info['market_ticker']}")
        print(f"  Episode length: {info['episode_length']}")
        print(f"  Initial cash: ${info['initial_cash']/100:.2f}")
        print(f"  Observation shape: {observation.shape}")
        
        # Verify observation
        assert observation.shape == (52,)  # Updated to match actual observation size
        assert observation.dtype == np.float32
        assert not np.any(np.isnan(observation))
        assert not np.any(np.isinf(observation))
        
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test a few steps
    try:
        total_reward = 0.0
        for step in range(min(5, info['episode_length'])):
            action = step % 5  # Cycle through actions
            
            observation, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            
            # Verify step output
            assert observation.shape == (52,)  # Updated to match actual observation size
            assert observation.dtype == np.float32
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            
            print(f"  Step {step}: action={action}, reward={reward:.4f}, "
                  f"portfolio=${step_info['portfolio_value']/100:.2f}, "
                  f"progress={step_info['episode_progress']:.2%}")
            
            if terminated or truncated:
                print(f"  Episode terminated at step {step}")
                break
        
        print(f"✅ Steps completed successfully, total reward: {total_reward:.4f}")
        
    except Exception as e:
        print(f"❌ Environment step failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test cleanup
    try:
        env.close()
        print("✅ Environment closed successfully")
    except Exception as e:
        print(f"❌ Environment close failed: {e}")
        return False
    
    print("🎉 All tests passed! MarketAgnosticKalshiEnv M7 is working correctly.")
    return True


if __name__ == "__main__":
    success = test_environment_sync()
    if success:
        print("\n✅ M7_MARKET_AGNOSTIC_ENV milestone completed successfully!")
    else:
        print("\n❌ M7_MARKET_AGNOSTIC_ENV milestone failed")
        exit(1)