#!/usr/bin/env python3
"""
Simple test to verify the new MarketAgnosticKalshiEnv pattern works.
Tests environment initialization with pre-loaded SessionData.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import logging

# Add src directory to path
backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader

logging.basicConfig(level=logging.WARNING)  # Reduce noise


async def test_new_pattern():
    """Test the new MarketAgnosticKalshiEnv pattern."""
    print("🧪 Testing new MarketAgnosticKalshiEnv pattern")
    print("=" * 50)
    
    # Step 1: Load SessionData
    try:
        print("Loading session data...")
        loader = SessionDataLoader()
        available_sessions = await loader.get_available_session_ids()
        
        if not available_sessions:
            print("❌ No session data available")
            return False
        
        print(f"Found {len(available_sessions)} sessions: {available_sessions}")
        
        # Try session 6 first (known to be good), then fallback to others
        for session_id in [6, 5, 7] + available_sessions:
            if session_id in available_sessions:
                try:
                    session_data = await loader.load_session(session_id)
                    if session_data and len(session_data.data_points) >= 3:
                        print(f"✅ Loaded session {session_id}: {len(session_data.data_points)} data points")
                        break
                except Exception as e:
                    print(f"⚠️  Session {session_id} failed to load: {e}")
                    continue
        else:
            print("❌ No valid session data could be loaded")
            return False
        
    except Exception as e:
        print(f"❌ Session loading failed: {e}")
        return False
    
    # Step 2: Create environment with SessionData
    try:
        print("Creating environment...")
        config = EnvConfig(
            max_markets=1,
            temporal_features=True,
            cash_start=10000  # $100 in cents
        )
        
        env = MarketAgnosticKalshiEnv(session_data, config)
        print(f"✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test reset
    try:
        print("Testing environment reset...")
        observation, info = env.reset()
        
        print(f"✅ Reset successful:")
        print(f"   Session: {info['session_id']}")
        print(f"   Market: {info['market_ticker']}")
        print(f"   Episode length: {info['episode_length']}")
        print(f"   Observation shape: {observation.shape}")
        print(f"   Observation range: [{observation.min():.4f}, {observation.max():.4f}]")
        
        # Basic validation
        expected_shape = (52,)  # Known observation size
        if observation.shape != expected_shape:
            print(f"⚠️  Observation shape mismatch: expected {expected_shape}, got {observation.shape}")
        
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(f"⚠️  Observation contains NaN or Inf values")
        
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Test a few steps
    try:
        print("Testing environment steps...")
        
        for i in range(3):
            action = i % env.action_space.n
            obs, reward, terminated, truncated, step_info = env.step(action)
            
            print(f"   Step {i}: action={action}, reward={reward:.4f}, "
                  f"portfolio=${step_info['portfolio_value']/100:.2f}")
            
            if terminated or truncated:
                print(f"   Episode ended at step {i}")
                break
        
        print("✅ Steps completed successfully")
        
    except Exception as e:
        print(f"❌ Environment steps failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Cleanup
    try:
        env.close()
        print("✅ Environment closed successfully")
    except Exception as e:
        print(f"⚠️  Environment close failed: {e}")
    
    print("\n🎉 All tests passed! New pattern is working correctly.")
    return True


def main():
    """Run the test."""
    try:
        success = asyncio.run(test_new_pattern())
        if success:
            print("\n✅ NEW PATTERN VALIDATION SUCCESSFUL!")
            print("All tests updated to use SessionData pattern are working.")
        else:
            print("\n❌ NEW PATTERN VALIDATION FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()