#!/usr/bin/env python3
"""
Simplified test of MarketAgnosticKalshiEnv to avoid event loop issues.
This version tests the environment without nested asyncio calls.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add the backend/src directory to the Python path
backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
import asyncio


def test_environment_creation():
    """Test basic environment creation and properties."""
    print("🔧 Testing Environment Creation")
    print("=" * 50)
    
    # Load session data
    try:
        loader = SessionDataLoader()
        available_sessions = asyncio.run(loader.get_available_session_ids())
        
        if not available_sessions:
            print("❌ No session data available")
            return None
        
        # Try to load session 5 first, fallback to first available
        session_id = 5 if 5 in available_sessions else available_sessions[0]
        session_data = asyncio.run(loader.load_session(session_id))
        
        if not session_data:
            print(f"❌ Could not load session {session_id}")
            return None
            
        print(f"✅ Session data loaded: session_id={session_data.session_id}, {len(session_data.data_points)} data points")
        
    except Exception as e:
        print(f"❌ Failed to load session data: {e}")
        return None
    
    # Create environment config
    config = EnvConfig(
        max_markets=1,
        temporal_features=True,
        cash_start=100000  # $1000 in cents
    )
    
    # Create environment
    try:
        env = MarketAgnosticKalshiEnv(session_data, config)
        print("✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Session ID: {session_data.session_id}")
        print(f"   Config: max_markets={config.max_markets}, temporal_features={config.temporal_features}")
        return env
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return None


def test_single_reset(env):
    """Test a single reset operation."""
    print("\n🔄 Testing Single Reset")
    print("=" * 50)
    
    try:
        obs, info = env.reset(seed=42)
        print("✅ Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation type: {obs.dtype}")
        print(f"   Observation range: [{np.min(obs):.4f}, {np.max(obs):.4f}]")
        print(f"   Non-zero features: {np.count_nonzero(obs)}/{len(obs)}")
        
        if info:
            print(f"   Reset info:")
            for key, value in info.items():
                print(f"     {key}: {value}")
        
        return obs, info
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_action_execution(env):
    """Test executing actions."""
    print("\n⚡ Testing Action Execution")
    print("=" * 50)
    
    action_names = {0: 'HOLD', 1: 'BUY_YES_LIMIT', 2: 'SELL_YES_LIMIT', 3: 'BUY_NO_LIMIT', 4: 'SELL_NO_LIMIT'}
    
    for action in range(min(3, env.action_space.n)):  # Test first 3 actions
        try:
            print(f"\n🎯 Testing action {action} ({action_names.get(action, 'UNKNOWN')})")
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   ✅ Action executed successfully")
            print(f"   Reward: {reward:.6f}")
            print(f"   Terminated: {terminated}")
            print(f"   Truncated: {truncated}")
            print(f"   New observation shape: {obs.shape}")
            
            if info:
                print(f"   Step info:")
                for key, value in info.items():
                    if isinstance(value, (int, float)):
                        print(f"     {key}: {value}")
                    else:
                        print(f"     {key}: {value}")
                        
            if terminated or truncated:
                print(f"   ⚠️  Episode ended after action {action}")
                break
                
        except Exception as e:
            print(f"   ❌ Action {action} failed: {e}")
            import traceback
            traceback.print_exc()
            break


def test_observation_consistency(env):
    """Test multiple resets for observation consistency."""
    print("\n🔍 Testing Observation Consistency")
    print("=" * 50)
    
    observations = []
    
    for i in range(3):
        try:
            obs, info = env.reset(seed=42 + i)
            observations.append(obs)
            print(f"   Reset {i+1}: shape={obs.shape}, range=[{np.min(obs):.4f}, {np.max(obs):.4f}]")
        except Exception as e:
            print(f"   ❌ Reset {i+1} failed: {e}")
            break
    
    if len(observations) > 1:
        print(f"   📊 Consistency check:")
        print(f"     All same shape: {all(obs.shape == observations[0].shape for obs in observations)}")
        print(f"     All same dtype: {all(obs.dtype == observations[0].dtype for obs in observations)}")
        
        # Check if observations are different (should be if using different seeds/sessions)
        same_values = all(np.allclose(obs, observations[0]) for obs in observations[1:])
        print(f"     All identical values: {same_values} (should be False for different episodes)")


def main():
    """Run all tests."""
    print("🧪 MarketAgnosticKalshiEnv - Simplified Test Suite")
    print("=" * 80)
    
    # Test environment creation
    env = test_environment_creation()
    if env is None:
        print("\n❌ Cannot continue - environment creation failed")
        return
    
    # Test single reset
    obs, info = test_single_reset(env)
    if obs is None:
        print("\n❌ Cannot continue - reset failed")
        return
    
    # Test action execution
    test_action_execution(env)
    
    # Test observation consistency
    test_observation_consistency(env)
    
    # Clean up
    try:
        env.close()
        print("\n✅ Environment closed successfully")
    except Exception as e:
        print(f"\n⚠️  Environment close failed: {e}")
    
    print("\n🎉 Simplified test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()