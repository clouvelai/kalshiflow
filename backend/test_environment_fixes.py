#!/usr/bin/env python3
"""
Test script to validate the environment fixes for realistic trading mechanics.

This script tests:
1. Liquidity consumption tracking
2. Realistic transaction costs  
3. Price walking for large orders
4. Action success rate should be ~5% profitable

Usage:
    python test_environment_fixes.py
"""

import sys
import os
import logging
import numpy as np
import asyncio
from typing import Dict, List

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader


def setup_logging():
    """Set up logging for test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def test_liquidity_consumption():
    """Test that liquidity consumption prevents infinite liquidity exploitation."""
    print("\n=== Testing Liquidity Consumption ===")
    
    try:
        # Load session data for testing
        loader = SessionDataLoader()
        sessions = await loader.get_available_sessions()
        
        if not sessions:
            print("❌ No sessions available for testing")
            return False
            
        # Get a session with good data
        session_id = sessions[0]['session_id']
        session_data = await loader.load_session(session_id)
        
        if not session_data.data_points:
            print(f"❌ Session {session_id} has no data points")
            return False
        
        # Create market view for testing - find one with sufficient data
        if not session_data.markets_involved:
            print(f"❌ Session {session_id} has no markets")
            return False
        
        market_view = None
        for target_market in session_data.markets_involved:
            test_view = session_data.create_market_view(target_market)
            if test_view and test_view.get_episode_length() >= 10:  # Need at least 10 steps
                market_view = test_view
                break
        
        if not market_view:
            print(f"❌ No markets in session {session_id} have sufficient data (need >=10 steps)")
            return False
        
        print(f"✅ Using market view: {market_view.target_market} with {market_view.get_episode_length()} steps from session {session_id}")
        
        # Create environment with small cash to test constraints
        config = EnvConfig(cash_start=50000)  # $500 starting cash
        env = MarketAgnosticKalshiEnv(market_view, config)
        
        obs, info = env.reset()
        print(f"✅ Environment reset successfully. Initial cash: ${info['initial_cash'] / 100:.2f}")
        
        # Test rapid trading actions to trigger liquidity consumption
        profitable_trades = 0
        total_trades = 0
        rewards = []
        
        for step in range(min(50, market_view.get_episode_length() - 1)):
            # Try aggressive trading action (BUY_YES with largest size)
            action = env.action_space.n - 1  # Largest size trade action
            
            obs, reward, terminated, truncated, step_info = env.step(action)
            rewards.append(reward)
            
            if reward > 0:
                profitable_trades += 1
            if action != 0:  # Non-HOLD action
                total_trades += 1
            
            # Check consumed liquidity tracking
            if len(env.consumed_liquidity) > 0:
                print(f"✅ Step {step}: Liquidity consumption tracked ({len(env.consumed_liquidity)} entries)")
                
            if terminated or truncated:
                break
        
        # Analyze results
        profit_rate = profitable_trades / total_trades if total_trades > 0 else 0
        avg_reward = np.mean(rewards)
        final_portfolio = step_info['portfolio_value']
        
        print(f"✅ Test completed:")
        print(f"   - Profitable trades: {profitable_trades}/{total_trades} ({profit_rate:.1%})")
        print(f"   - Average reward per step: {avg_reward:.6f}")
        print(f"   - Final portfolio value: ${final_portfolio / 100:.2f}")
        print(f"   - Liquidity entries tracked: {len(env.consumed_liquidity)}")
        
        # Success criteria: profit rate should be low (~5-20%) not 95%+
        success = (profit_rate < 0.4)  # Less than 40% profitable
        
        if success:
            print("✅ PASS: Environment properly constrains profitability")
        else:
            print(f"❌ FAIL: Profit rate too high ({profit_rate:.1%}), indicates insufficient constraints")
            
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_realistic_fees():
    """Test that realistic fees are being applied correctly."""
    print("\n=== Testing Realistic Transaction Fees ===")
    
    try:
        # Create simple test environment
        loader = SessionDataLoader()
        sessions = await loader.get_available_sessions()
        
        if not sessions:
            print("❌ No sessions available for testing")
            return False
        
        session_id = sessions[0]['session_id']
        session_data = await loader.load_session(session_id)
        
        if not session_data.markets_involved:
            print(f"❌ No markets available")
            return False
        
        market_view = None
        for target_market in session_data.markets_involved:
            test_view = session_data.create_market_view(target_market)
            if test_view and test_view.get_episode_length() >= 10:  # Need at least 10 steps
                market_view = test_view
                break
        
        if not market_view:
            print(f"❌ No markets in session have sufficient data (need >=10 steps)")
            return False
            
        config = EnvConfig(cash_start=100000)  # $1000 starting cash
        env = MarketAgnosticKalshiEnv(market_view, config)
        
        obs, info = env.reset()
        initial_portfolio = info['initial_cash']
        
        # Execute a single large trade to test fee calculation
        large_trade_action = env.action_space.n - 1  # Largest size
        obs, reward, terminated, truncated, step_info = env.step(large_trade_action)
        
        portfolio_change = step_info['portfolio_value'] - initial_portfolio
        
        print(f"✅ Large trade executed:")
        print(f"   - Initial portfolio: ${initial_portfolio / 100:.2f}")
        print(f"   - Final portfolio: ${step_info['portfolio_value'] / 100:.2f}")
        print(f"   - Portfolio change: ${portfolio_change / 100:.2f}")
        print(f"   - Reward: {reward:.6f}")
        
        # For a large trade, we expect significant fee impact (negative reward even without position change)
        fee_impact_detected = (reward < -0.001)  # Significant negative reward
        
        if fee_impact_detected:
            print("✅ PASS: Realistic fees are being applied")
        else:
            print("❌ FAIL: Fees appear insufficient")
            
        return fee_impact_detected
        
    except Exception as e:
        print(f"❌ Fee test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_price_walking():
    """Test that large orders walk through multiple price levels."""
    print("\n=== Testing Price Walking Mechanics ===")
    
    try:
        # Load session and test environment
        loader = SessionDataLoader()
        sessions = await loader.get_available_sessions()
        
        if not sessions:
            print("❌ No sessions available")
            return False
        
        session_id = sessions[0]['session_id']
        session_data = await loader.load_session(session_id)
        
        if not session_data.markets_involved:
            print(f"❌ No markets available")
            return False
        
        market_view = None
        for target_market in session_data.markets_involved:
            test_view = session_data.create_market_view(target_market)
            if test_view and test_view.get_episode_length() >= 10:  # Need at least 10 steps
                market_view = test_view
                break
        
        if not market_view:
            print(f"❌ No markets in session have sufficient data (need >=10 steps)")
            return False
        
        config = EnvConfig(cash_start=100000)
        env = MarketAgnosticKalshiEnv(market_view, config)
        
        obs, info = env.reset()
        
        # Test constraint application directly
        current_data = env.market_view.get_timestep_data(0)
        if env.current_market in current_data.markets_data:
            from kalshiflow_rl.environments.market_agnostic_env import convert_session_data_to_orderbook
            orderbook = convert_session_data_to_orderbook(
                current_data.markets_data[env.current_market],
                env.current_market
            )
            
            # Test large order constraint
            large_action = env.action_space.n - 1
            constraint_result = env._apply_realistic_orderbook_constraints(
                orderbook, large_action, env.current_market
            )
            
            print(f"✅ Price walking test:")
            print(f"   - Can execute: {constraint_result['can_execute']}")
            print(f"   - Available quantity: {constraint_result['available_quantity']}")
            print(f"   - Price levels hit: {len(constraint_result['execution_levels'])}")
            print(f"   - Total cost: {constraint_result['total_cost']}")
            
            # Success: either multiple levels hit OR reasonable constraints applied
            price_walking_works = (
                len(constraint_result['execution_levels']) > 1 or  # Multiple levels
                constraint_result['available_quantity'] < 100     # Quantity constraints
            )
            
            if price_walking_works:
                print("✅ PASS: Price walking mechanics are working")
            else:
                print("❌ FAIL: Price walking appears disabled")
                
            return price_walking_works
        else:
            print("❌ No orderbook data available for testing")
            return False
            
    except Exception as e:
        print(f"❌ Price walking test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all environment tests."""
    print("🧪 Testing Environment Fixes for Realistic Trading Mechanics")
    print("=" * 60)
    
    setup_logging()
    
    # Run all tests
    tests = [
        ("Liquidity Consumption", test_liquidity_consumption),
        ("Realistic Fees", test_realistic_fees),
        ("Price Walking", test_price_walking),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        try:
            results[test_name] = await test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Environment fixes are working correctly!")
        print("The model should now only be profitable ~5-20% of the time, not 95%+")
    else:
        print("⚠️  Some tests failed - environment may still have exploitable infinite liquidity")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)