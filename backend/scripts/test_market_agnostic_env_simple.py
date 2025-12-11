#!/usr/bin/env python3
"""
Simple test script for MarketAgnosticKalshiEnv M7 implementation.

Tests the environment with real session data to verify all components work.
"""

import asyncio
import numpy as np
import logging

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig, convert_session_data_to_orderbook
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
from kalshiflow_rl.data.orderbook_state import OrderbookState

# Configure logging with DEBUG for order execution
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# Enable debug logging for key modules
logging.getLogger("kalshiflow_rl.environments.limit_order_action_space").setLevel(logging.DEBUG)
logging.getLogger("kalshiflow_rl.trading.order_manager").setLevel(logging.DEBUG)
logging.getLogger("kalshiflow_rl.environments.market_agnostic_env").setLevel(logging.DEBUG)


async def test_conversion_function():
    """Test the session data to OrderbookState conversion function."""
    print("=== Testing conversion function ===")
    
    session_loader = SessionDataLoader()
    available_sessions = await session_loader.get_available_session_ids()
    
    if not available_sessions:
        print("No session data available for testing")
        return False
    
    # Load first session
    session_id = available_sessions[0]
    session_data = await session_loader.load_session(session_id)
    
    if not session_data or not session_data.data_points:
        print(f"Session {session_id} has no data points")
        return False
    
    # Get first data point with market data
    first_data = session_data.data_points[0]
    if not first_data.markets_data:
        print("No market data in first data point")
        return False
    
    # Test conversion for first market
    market_ticker = list(first_data.markets_data.keys())[0]
    market_data = first_data.markets_data[market_ticker]
    
    print(f"Testing conversion for market: {market_ticker}")
    
    # Convert to OrderbookState
    orderbook = convert_session_data_to_orderbook(market_data, market_ticker)
    
    # Verify conversion worked
    assert isinstance(orderbook, OrderbookState)
    assert orderbook.market_ticker == market_ticker
    
    total_levels = len(orderbook.yes_bids) + len(orderbook.yes_asks) + len(orderbook.no_bids) + len(orderbook.no_asks)
    print(f"✅ Conversion successful: {total_levels} total price levels")
    
    return True


async def test_environment_basic():
    """Test basic environment functionality."""
    print("\n=== Testing environment initialization ===")
    
    session_loader = SessionDataLoader()
    available_sessions = await session_loader.get_available_session_ids()
    
    if not available_sessions:
        print("No session data available for testing")
        return False
    
    # Load first session and create market view
    session_id = available_sessions[0]
    session_data = await session_loader.load_session(session_id)
    
    # Get first market with sufficient data
    market_ticker = list(session_data.markets_involved)[0]
    market_view = session_data.create_market_view(market_ticker)
    
    # Create config
    config = EnvConfig(
        max_markets=1,
        temporal_features=True,
        cash_start=10000  # $100 in cents
    )
    
    # Initialize environment with market view
    env = MarketAgnosticKalshiEnv(market_view, config)
    
    # Check basic properties (52 features based on MarketAgnosticKalshiEnv.OBSERVATION_DIM)
    assert env.observation_space.shape == (52,)
    assert env.action_space.n == 5
    
    print(f"✅ Environment initialized: {env.observation_space.shape} obs, {env.action_space.n} actions")
    
    return env


async def test_environment_episode():
    """Test running a complete environment episode."""
    print("\n=== Testing environment episode ===")
    
    # Get environment from basic test
    env = await test_environment_basic()
    if not env:
        return False
    
    try:
        # Reset environment
        observation, info = env.reset()
        
        # Check reset output
        assert isinstance(observation, np.ndarray)
        assert observation.shape == (52,)
        assert observation.dtype == np.float32
        
        print(f"✅ Reset successful:")
        print(f"  Session: {info['session_id']}")
        print(f"  Market: {info['market_ticker']}")
        print(f"  Episode length: {info['episode_length']}")
        print(f"  Initial cash: ${info['initial_cash']/100:.2f}")
        
        # Take a few steps
        total_reward = 0.0
        for step in range(min(10, info['episode_length'])):
            action = step % 5  # Cycle through actions
            
            observation, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            
            # Check step output  
            assert isinstance(observation, np.ndarray)
            assert observation.shape == (52,)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            
            if step % 3 == 0:  # Log every 3rd step
                print(f"  Step {step}: action={action}, reward={reward:.4f}, "
                      f"portfolio=${step_info['portfolio_value']/100:.2f}")
            
            if terminated or truncated:
                print(f"  Episode terminated at step {step}")
                break
        
        print(f"✅ Episode completed successfully, total reward: {total_reward:.4f}")
        
        # Clean up
        env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_all_components():
    """Test all components together."""
    print("🚀 Starting MarketAgnosticKalshiEnv M7 test suite...")
    
    try:
        # Test conversion function
        if not await test_conversion_function():
            print("❌ Conversion function test failed")
            return False
        
        # Test environment episode
        if not await test_environment_episode():
            print("❌ Environment episode test failed")
            return False
        
        print("\n🎉 All tests passed! MarketAgnosticKalshiEnv M7 is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_all_components())
    if success:
        print("\n✅ M7_MARKET_AGNOSTIC_ENV milestone completed successfully!")
    else:
        print("\n❌ M7_MARKET_AGNOSTIC_ENV milestone failed")
        exit(1)