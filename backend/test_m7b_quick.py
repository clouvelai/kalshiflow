#!/usr/bin/env python3
"""Quick test to check if orders are executing."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
from kalshiflow_rl.environments.market_agnostic_env import MarketAgnosticKalshiEnv, EnvConfig


async def quick_test():
    """Quick test of order execution."""
    
    print("Loading session...")
    database_url = os.getenv("DATABASE_URL")
    loader = SessionDataLoader(database_url=database_url)
    
    # Get most recent session
    sessions = await loader.get_available_sessions()
    session_id = sessions[0]['session_id']
    
    # Load session
    session_data = await loader.load_session(session_id)
    
    # Find a market
    market_activity = {}
    for dp in session_data.data_points[:100]:
        for market in dp.markets_data:
            market_activity[market] = market_activity.get(market, 0) + 1
    
    target_market = max(market_activity.items(), key=lambda x: x[1])[0]
    print(f"Using market: {target_market}")
    
    # Create view
    market_view = session_data.create_market_view(target_market)
    
    # Initialize environment
    config = EnvConfig(cash_start=10000, max_markets=1)
    env = MarketAgnosticKalshiEnv(market_view=market_view, config=config)
    
    # Reset
    obs, info = env.reset()
    print(f"Reset done, cash: {env.position_tracker.cash_balance}")
    
    # Try HOLD action (0)
    print("\nTrying HOLD action...")
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"After HOLD: reward={reward}, cash={env.position_tracker.cash_balance}")
    print(f"Positions: {env.position_tracker.positions}")
    
    # Try BUY action (1)
    print("\nTrying BUY_YES_LIMIT action...")
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"After BUY: reward={reward}, cash={env.position_tracker.cash_balance}")
    print(f"Positions: {env.position_tracker.positions}")
    
    # Check order manager
    print(f"\nOrder manager cash: ${env.order_manager.cash_balance:.2f}")
    print(f"Open orders: {len(env.order_manager.open_orders)}")
    
    print("\n✅ Test completed!")
    return True


if __name__ == "__main__":
    asyncio.run(quick_test())