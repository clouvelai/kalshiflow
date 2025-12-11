#!/usr/bin/env python3
"""
Debug market selection and data availability in M7 environment.
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


async def debug_market_selection():
    """Debug market selection and data availability."""
    
    print("=" * 80)
    print("🔍 DEBUG: MARKET SELECTION AND DATA AVAILABILITY")
    print("=" * 80)
    
    # Load session data
    print("\n1️⃣ Loading session data...")
    loader = SessionDataLoader()
    session_data = await loader.load_session(6)  # Use session 6
    
    if not session_data:
        print("❌ Failed to load session data")
        return
    
    print(f"✅ Session loaded: {len(session_data.data_points)} data points")
    
    # Analyze market activity across first 20 data points
    print("\n2️⃣ Analyzing market activity across timesteps...")
    
    market_activity = {}
    market_data_points = {}
    
    # Check first 20 data points
    for step in range(min(20, len(session_data.data_points))):
        data_point = session_data.data_points[step]
        
        print(f"\nStep {step}:")
        print(f"  Available markets: {len(data_point.markets_data)}")
        
        for market_ticker in data_point.markets_data.keys():
            if market_ticker not in market_activity:
                market_activity[market_ticker] = 0
                market_data_points[market_ticker] = []
                
            market_activity[market_ticker] += 1
            market_data_points[market_ticker].append(step)
            
        # Show top 5 markets in this step
        if data_point.markets_data:
            sample_markets = list(data_point.markets_data.keys())[:5]
            print(f"  Sample markets: {sample_markets}")
    
    # Show most active markets
    print("\n3️⃣ Most active markets (appearing in most timesteps):")
    sorted_markets = sorted(market_activity.items(), key=lambda x: x[1], reverse=True)
    
    for i, (market, count) in enumerate(sorted_markets[:10]):
        print(f"  {i+1:2d}. {market:<30} appears in {count:2d} timesteps")
        print(f"      Steps: {market_data_points[market][:10]}...")
    
    # Test environment creation and market selection
    print("\n4️⃣ Testing environment creation...")
    config = EnvConfig(
        max_markets=1,
        temporal_features=True,
        cash_start=100000
    )
    
    env = MarketAgnosticKalshiEnv(session_data, config)
    
    # Test multiple resets to see market selection
    print("\n5️⃣ Testing market selection across multiple resets...")
    
    selected_markets = []
    for reset_num in range(5):
        obs, info = env.reset()
        selected_market = env.current_market
        selected_markets.append(selected_market)
        
        print(f"\nReset {reset_num + 1}: Selected {selected_market}")
        
        # Check data availability for this market in first 10 steps
        steps_with_data = []
        for step in range(min(10, len(session_data.data_points))):
            data_point = session_data.data_points[step]
            if selected_market in data_point.markets_data:
                steps_with_data.append(step)
        
        print(f"  Data available at steps: {steps_with_data}")
        
        if not steps_with_data:
            print(f"  ❌ No data available for this market in first 10 steps!")
        else:
            # Test spread for this market
            first_step_with_data = steps_with_data[0]
            market_data = session_data.data_points[first_step_with_data].markets_data[selected_market]
            
            yes_bids = market_data.get('yes_bids', {})
            yes_asks = market_data.get('yes_asks', {})
            
            if yes_bids and yes_asks:
                best_bid = max(yes_bids.keys())
                best_ask = min(yes_asks.keys())
                spread = best_ask - best_bid
                print(f"  Step {first_step_with_data}: bid={best_bid}¢, ask={best_ask}¢, spread={spread}¢")
            else:
                print(f"  Step {first_step_with_data}: Empty orderbook")
    
    # Analyze the market selection algorithm
    print(f"\n6️⃣ Analyzing market selection algorithm...")
    
    # Show what the _select_most_active_market function is doing
    all_market_activity = {}
    for data_point in session_data.data_points:
        for market_ticker, market_data in data_point.markets_data.items():
            if market_ticker not in all_market_activity:
                all_market_activity[market_ticker] = 0
                
            # Calculate activity the same way as the environment does
            total_depth = 0
            if 'yes_bids' in market_data:
                total_depth += sum(market_data['yes_bids'].values())
            if 'yes_asks' in market_data:
                total_depth += sum(market_data['yes_asks'].values())
            if 'no_bids' in market_data:
                total_depth += sum(market_data['no_bids'].values())
            if 'no_asks' in market_data:
                total_depth += sum(market_data['no_asks'].values())
            all_market_activity[market_ticker] += total_depth
    
    print(f"\nTop markets by total orderbook depth (selection algorithm):")
    sorted_by_depth = sorted(all_market_activity.items(), key=lambda x: x[1], reverse=True)
    
    for i, (market, depth) in enumerate(sorted_by_depth[:10]):
        data_point_count = market_activity.get(market, 0)
        print(f"  {i+1:2d}. {market:<30} depth={depth:8.0f}, in {data_point_count:2d} timesteps")
    
    # Test if the most active market actually has early data
    if sorted_by_depth:
        most_active_market = sorted_by_depth[0][0]
        print(f"\n7️⃣ Testing most active market: {most_active_market}")
        
        early_steps_with_data = []
        for step in range(min(20, len(session_data.data_points))):
            data_point = session_data.data_points[step]
            if most_active_market in data_point.markets_data:
                early_steps_with_data.append(step)
        
        print(f"  Early data availability (first 20 steps): {early_steps_with_data}")
        
        if early_steps_with_data:
            # Test orderbook quality
            step = early_steps_with_data[0]
            market_data = session_data.data_points[step].markets_data[most_active_market]
            
            yes_bids = market_data.get('yes_bids', {})
            yes_asks = market_data.get('yes_asks', {})
            
            if yes_bids and yes_asks:
                best_bid = max(yes_bids.keys())
                best_ask = min(yes_asks.keys())
                spread = best_ask - best_bid
                print(f"  First data at step {step}: bid={best_bid}¢, ask={best_ask}¢, spread={spread}¢")
                
                # Test order fill logic
                print(f"  Testing order fills at spread={spread}¢:")
                
                # Test aggressive buy (should cross spread)
                aggressive_buy_price = best_ask
                print(f"    Aggressive buy at {aggressive_buy_price}¢ (ask price)")
                print(f"    Would fill: {aggressive_buy_price >= best_ask}")
                
                # Test aggressive sell (should cross spread) 
                aggressive_sell_price = best_bid
                print(f"    Aggressive sell at {aggressive_sell_price}¢ (bid price)")
                print(f"    Would fill: {aggressive_sell_price <= best_bid}")
            else:
                print(f"  Empty orderbook at step {step}")
        else:
            print(f"  ❌ Most active market has NO early data!")


async def main():
    """Run the debug analysis."""
    try:
        await debug_market_selection()
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())