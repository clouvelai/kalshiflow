#!/usr/bin/env python3
"""
Test the fixed M7 MarketAgnosticKalshiEnv implementation.
This tests the fixes for market selection and cash flow issues.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add the backend/src directory to the Python path
backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader


async def test_fixed_m7():
    """Test the fixed M7 implementation."""
    
    print("=" * 80)
    print("🔧 TEST: FIXED M7 MarketAgnosticKalshiEnv")
    print("=" * 80)
    
    # Load session data
    print("\n1️⃣ Loading session data...")
    loader = SessionDataLoader()
    session_data = await loader.load_session(6)  # Use session 6
    
    if not session_data:
        print("❌ Failed to load session data")
        return False
    
    print(f"✅ Session loaded: {len(session_data.data_points)} data points")
    print(f"   Average spread: ${session_data.avg_spread:.4f}")
    
    # Create environment
    print("\n2️⃣ Creating environment...")
    config = EnvConfig(
        max_markets=1,
        temporal_features=True,
        cash_start=100000  # $1000 in cents
    )
    
    env = MarketAgnosticKalshiEnv(session_data, config)
    
    # Test multiple resets to ensure market selection is working
    print("\n3️⃣ Testing market selection fix...")
    
    for reset_num in range(3):
        obs, info = env.reset()
        selected_market = env.current_market
        
        print(f"\nReset {reset_num + 1}: Selected {selected_market}")
        
        # Check if selected market has data in early steps
        early_data_steps = []
        for step in range(min(10, len(session_data.data_points))):
            data_point = session_data.data_points[step]
            if selected_market in data_point.markets_data:
                early_data_steps.append(step)
        
        print(f"  Early data available at steps: {early_data_steps[:5]}..." if len(early_data_steps) > 5 else f"  Early data available at steps: {early_data_steps}")
        
        if early_data_steps:
            print(f"  ✅ Market has early data!")
            
            # Test first step with data
            first_step = early_data_steps[0]
            market_data = session_data.data_points[first_step].markets_data[selected_market]
            
            yes_bids = market_data.get('yes_bids', {})
            yes_asks = market_data.get('yes_asks', {})
            
            if yes_bids and yes_asks:
                best_bid = max(yes_bids.keys())
                best_ask = min(yes_asks.keys())
                spread = best_ask - best_bid
                print(f"  Step {first_step}: bid={best_bid}¢, ask={best_ask}¢, spread={spread}¢")
            
            break
        else:
            print(f"  ❌ No early data - selection algorithm still broken")
    
    # Test cash flow with actual trading
    print("\n4️⃣ Testing cash flow with trading actions...")
    
    obs, info = env.reset()
    
    print(f"\n💰 Initial state:")
    print(f"   Cash balance: ${env.position_tracker.cash_balance / 100:.2f}")
    print(f"   Order manager cash: ${env.order_manager.cash_balance:.2f}")
    
    # Run 10 steps with active trading
    total_reward = 0.0
    cash_changes = []
    
    for step in range(min(10, env.episode_length)):
        # Get current cash
        prev_cash = env.position_tracker.cash_balance
        prev_om_cash = env.order_manager.cash_balance
        
        # Try different actions
        if step % 4 == 0:
            action = 1  # BUY_YES_LIMIT
        elif step % 4 == 1:
            action = 2  # SELL_YES_LIMIT
        elif step % 4 == 2:
            action = 3  # BUY_NO_LIMIT
        else:
            action = 4  # SELL_NO_LIMIT
        
        # Execute step
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        
        # Check cash changes
        new_cash = env.position_tracker.cash_balance
        new_om_cash = env.order_manager.cash_balance
        
        cash_change = new_cash - prev_cash
        om_cash_change = new_om_cash - prev_om_cash
        
        cash_changes.append((cash_change, om_cash_change))
        
        print(f"\n   Step {step + 1}: action={action}, reward={reward:.4f}")
        print(f"     Cash change: tracker=${cash_change/100:.2f}, om=${om_cash_change:.2f}")
        print(f"     New balances: tracker=${new_cash/100:.2f}, om=${new_om_cash:.2f}")
        
        # Check if any orders were placed/filled
        open_orders = env.order_manager.get_open_orders(env.current_market)
        positions = env.position_tracker.positions
        
        print(f"     Open orders: {len(open_orders)}")
        if open_orders:
            for order in open_orders[:2]:  # Show first 2
                print(f"       - {order.side.name} {order.contract_side.name} {order.quantity}@{order.limit_price}¢")
        
        print(f"     Positions: {positions}")
        
        if terminated or truncated:
            print(f"     Episode ended at step {step + 1}")
            break
    
    # Summary
    print(f"\n5️⃣ Summary:")
    print(f"   Total reward: {total_reward:.4f}")
    print(f"   Final cash: ${env.position_tracker.cash_balance / 100:.2f}")
    print(f"   Final OM cash: ${env.order_manager.cash_balance:.2f}")
    
    # Check if any cash actually moved
    total_cash_changes = sum(abs(change[0]) for change in cash_changes)
    total_om_changes = sum(abs(change[1]) for change in cash_changes)
    
    print(f"   Total cash movement: tracker=${total_cash_changes/100:.2f}, om=${total_om_changes:.2f}")
    
    if total_cash_changes > 0 or total_om_changes > 0:
        print(f"   ✅ CASH FLOW WORKING! Money actually moved during trading.")
        return True
    else:
        print(f"   ❌ Cash flow still broken - no money movement detected.")
        return False


async def main():
    """Run the test."""
    try:
        success = await test_fixed_m7()
        if success:
            print(f"\n🎉 M7 CASH FLOW FIX SUCCESSFUL!")
        else:
            print(f"\n❌ M7 cash flow fix failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())