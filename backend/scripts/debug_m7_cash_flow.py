#!/usr/bin/env python3
"""
Debug script to analyze why cash doesn't change in M7 MarketAgnosticKalshiEnv.
Traces order execution, spread conditions, and fills.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import logging

# Configure logging for detailed debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s - %(name)s - %(message)s'
)

# Add the backend/src directory to the Python path
backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader


async def debug_cash_flow():
    """Debug the cash flow issue in M7 environment."""
    
    print("=" * 80)
    print("🔍 DEBUG: M7 CASH FLOW ANALYSIS")
    print("=" * 80)
    
    # Load session data
    print("\n1️⃣ Loading session data...")
    loader = SessionDataLoader()
    session_data = await loader.load_session(6)  # Use session 6 with good data
    
    if not session_data:
        print("❌ Failed to load session data")
        return
    
    print(f"✅ Session loaded: {len(session_data.data_points)} data points")
    print(f"   Average spread: ${session_data.avg_spread:.4f}")
    print(f"   Markets involved: {len(session_data.markets_involved)}")
    
    # Create environment
    print("\n2️⃣ Creating environment...")
    config = EnvConfig(
        max_markets=1,
        temporal_features=True,
        cash_start=100000  # $1000 in cents
    )
    
    env = MarketAgnosticKalshiEnv(session_data, config)
    
    # Reset environment
    print("\n3️⃣ Resetting environment...")
    obs, info = env.reset()
    
    print(f"✅ Environment reset successful")
    print(f"   Selected market: {env.current_market}")
    print(f"   Initial cash: ${env.position_tracker.cash_balance / 100:.2f}")
    print(f"   Order manager cash: ${env.order_manager.cash_balance:.2f}")
    
    # Analyze first few data points for spread conditions
    print("\n4️⃣ Analyzing spread conditions...")
    
    for step in range(min(5, env.episode_length)):
        print(f"\n--- STEP {step} ---")
        
        # Get current data point
        current_data = env.session_data.get_timestep_data(step)
        if not current_data:
            print(f"❌ No data for step {step}")
            break
        
        if env.current_market not in current_data.markets_data:
            print(f"❌ No data for {env.current_market} at step {step}")
            continue
        
        market_data = current_data.markets_data[env.current_market]
        
        # Extract orderbook information
        yes_bids = market_data.get('yes_bids', {})
        yes_asks = market_data.get('yes_asks', {})
        
        if yes_bids and yes_asks:
            best_bid = max(yes_bids.keys()) if yes_bids else None
            best_ask = min(yes_asks.keys()) if yes_asks else None
            
            if best_bid is not None and best_ask is not None:
                spread = best_ask - best_bid
                print(f"📊 Orderbook: Bid={best_bid}¢, Ask={best_ask}¢, Spread={spread}¢")
                
                # Check what aggressive pricing would be
                aggressive_buy_price = best_ask  # Buy at ask for immediate fill
                aggressive_sell_price = best_bid  # Sell at bid for immediate fill
                
                print(f"🎯 Aggressive pricing: Buy@{aggressive_buy_price}¢, Sell@{aggressive_sell_price}¢")
                
                # Estimate cost for 10 contracts
                buy_cost = 10 * aggressive_buy_price / 100.0
                sell_cost = 10 * aggressive_sell_price / 100.0
                
                print(f"💰 Cost estimates: Buy=${buy_cost:.2f}, Sell=${sell_cost:.2f}")
                
                # Test fill logic
                from kalshiflow_rl.trading.order_manager import OrderInfo, OrderSide, ContractSide, OrderStatus
                import time
                
                # Create a test buy order
                test_order = OrderInfo(
                    order_id="test_order",
                    ticker=env.current_market,
                    side=OrderSide.BUY,
                    contract_side=ContractSide.YES,
                    quantity=10,
                    limit_price=aggressive_buy_price,
                    status=OrderStatus.PENDING,
                    placed_at=time.time()
                )
                
                # Create orderbook state for testing
                from kalshiflow_rl.environments.market_agnostic_env import convert_session_data_to_orderbook
                orderbook = convert_session_data_to_orderbook(market_data, env.current_market)
                
                # Test if order can fill immediately
                can_fill = env.order_manager._can_fill_immediately(test_order, orderbook)
                print(f"🔥 Can fill immediately: {can_fill}")
                
                if can_fill:
                    fill_price = env.order_manager._get_fill_price(test_order, orderbook)
                    print(f"🎉 Would fill at: {fill_price}¢")
                else:
                    print(f"❌ Order would NOT fill immediately")
                    print(f"   Order limit price: {test_order.limit_price}¢")
                    print(f"   Market best ask: {best_ask}¢")
                    print(f"   Condition: {test_order.limit_price} >= {best_ask} = {test_order.limit_price >= best_ask}")
            else:
                print(f"❌ No valid bid/ask prices")
        else:
            print(f"❌ Empty orderbook: bids={len(yes_bids)}, asks={len(yes_asks)}")
        
        # Test actual action execution
        print("\n🎬 Testing action execution...")
        
        # Try a BUY_YES action (action 1)
        action = 1
        prev_cash = env.position_tracker.cash_balance
        prev_om_cash = env.order_manager.cash_balance
        
        print(f"💵 Before action: tracker_cash=${prev_cash/100:.2f}, om_cash=${prev_om_cash:.2f}")
        
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        new_cash = env.position_tracker.cash_balance
        new_om_cash = env.order_manager.cash_balance
        
        print(f"💵 After action: tracker_cash=${new_cash/100:.2f}, om_cash=${new_om_cash:.2f}")
        print(f"🔄 Cash change: tracker=${(new_cash-prev_cash)/100:.2f}, om=${new_om_cash-prev_om_cash:.2f}")
        print(f"🎁 Reward: {reward:.4f}")
        
        # Check if any orders were placed
        open_orders = env.order_manager.get_open_orders(env.current_market)
        print(f"📋 Open orders: {len(open_orders)}")
        
        for order in open_orders:
            print(f"   Order: {order.side.name} {order.contract_side.name} {order.quantity}@{order.limit_price}¢")
        
        # Check positions
        positions = env.position_tracker.positions
        print(f"📊 Positions: {positions}")
        
        if terminated or truncated:
            print(f"🔚 Episode ended at step {step}")
            break
    
    print("\n5️⃣ Summary of findings...")
    
    print(f"💡 Key insights:")
    print(f"   - Average spread: ${session_data.avg_spread:.4f} (very wide)")
    print(f"   - Large spreads prevent aggressive orders from filling")
    print(f"   - Aggressive pricing strategy crosses spread but spreads are too wide")
    print(f"   - Need to either use tighter spreads or adjust pricing strategy")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    print(f"   1. Use session with tighter spreads (< $5)")
    print(f"   2. Implement 'mid' pricing strategy for training")
    print(f"   3. Add spread validation in action space")
    print(f"   4. Consider synthetic data with realistic spreads")


async def main():
    """Run the debug analysis."""
    try:
        await debug_cash_flow()
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())