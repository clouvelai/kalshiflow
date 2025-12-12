#!/usr/bin/env python3
"""
Check available session data to find valid sessions for testing.
"""

import asyncio
import logging
from kalshiflow_rl.environments.session_data_loader import SessionDataLoader

logging.basicConfig(level=logging.INFO)


async def check_sessions():
    """Check available sessions and their data quality."""
    session_loader = SessionDataLoader()
    available_sessions = await session_loader.get_available_session_ids()
    
    print(f"Found {len(available_sessions)} available sessions: {available_sessions}")
    
    valid_sessions = []
    
    for session_id in available_sessions:
        try:
            session_data = await session_loader.load_session(session_id)
            if session_data:
                length = session_data.get_episode_length()
                print(f"Session {session_id}: {length} data points")
                
                if length >= 3:
                    valid_sessions.append(session_id)
                    print(f"  ✅ Valid session: {session_id}")
                    
                    # Check first data point
                    if session_data.data_points:
                        first_data = session_data.data_points[0]
                        market_count = len(first_data.markets_data)
                        print(f"  - {market_count} markets in first data point")
                        if first_data.markets_data:
                            first_market = list(first_data.markets_data.keys())[0]
                            print(f"  - First market: {first_market}")
                else:
                    print(f"  ❌ Session {session_id}: insufficient data ({length} < 3)")
            else:
                print(f"  ❌ Session {session_id}: failed to load")
                
        except Exception as e:
            print(f"  ❌ Session {session_id}: error loading - {e}")
    
    print(f"\nValid sessions for testing: {valid_sessions}")
    return valid_sessions


if __name__ == "__main__":
    asyncio.run(check_sessions())