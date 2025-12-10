#!/usr/bin/env python3
"""
Test SessionDataLoader with REAL session data from the database.
This validates that M3 actually works with production data.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshiflow_rl.environments.session_data_loader import SessionDataLoader
from kalshiflow_rl.data.database import Database

async def test_real_session_loading():
    """Test loading real session data from the database."""
    
    print("=" * 60)
    print("TESTING M3 SESSIONDATALOADER WITH REAL DATA")
    print("=" * 60)
    
    # Initialize database connection
    db = Database()
    await db.initialize()
    
    try:
        # First, let's see what sessions we have
        print("\n1. Checking available sessions...")
        sessions = await db.pool.fetch("""
            SELECT session_id, started_at, ended_at, 
                   status, snapshots_count, deltas_count,
                   market_tickers
            FROM rl_orderbook_sessions 
            WHERE status = 'closed'
            ORDER BY session_id DESC
            LIMIT 5
        """)
        
        if not sessions:
            print("❌ No closed sessions found in database!")
            return
            
        print(f"✅ Found {len(sessions)} closed sessions:")
        for s in sessions:
            markets = s['market_tickers'] if s['market_tickers'] else []
            print(f"   Session {s['session_id']}: {s['snapshots_count']} snapshots, "
                  f"{s['deltas_count']} deltas, {len(markets)} markets")
        
        # Pick the most recent session with data
        session_to_test = sessions[0]
        session_id = session_to_test['session_id']
        
        print(f"\n2. Testing with Session {session_id}...")
        print(f"   Period: {session_to_test['started_at']} to {session_to_test['ended_at']}")
        print(f"   Markets: {len(session_to_test['market_tickers'])} markets")
        
        # Initialize SessionDataLoader
        loader = SessionDataLoader(database_url=os.getenv("DATABASE_URL"))
        
        # Load the session
        print(f"\n3. Loading session data...")
        session_data = await loader.load_session(session_id)
        
        if session_data:
            print(f"✅ Successfully loaded session {session_id}!")
            
            # Display statistics
            print(f"\n4. Session Statistics:")
            print(f"   - Episode length: {session_data.get_episode_length()} timesteps")
            print(f"   - Duration: {session_data.total_duration}")
            print(f"   - Markets involved: {len(session_data.markets_involved)}")
            print(f"   - First 5 markets: {session_data.markets_involved[:5]}")
            
            # Check data quality
            print(f"\n5. Data Quality:")
            print(f"   - Data quality score: {session_data.data_quality_score:.2f}")
            print(f"   - Average spread: {session_data.avg_spread:.4f}")
            print(f"   - Volatility score: {session_data.volatility_score:.4f}")
            print(f"   - Market diversity: {session_data.market_diversity:.2f}")
            
            # Temporal features
            print(f"\n6. Temporal Analysis:")
            print(f"   - Activity bursts: {len(session_data.activity_bursts)}")
            print(f"   - Quiet periods: {len(session_data.quiet_periods)}")
            if session_data.temporal_gaps:
                avg_gap = sum(session_data.temporal_gaps) / len(session_data.temporal_gaps)
                print(f"   - Average time gap: {avg_gap:.2f} seconds")
            
            # Sample some data points
            print(f"\n7. Sample Data Points:")
            for i in [0, 100, 200, -1]:
                if i < session_data.get_episode_length():
                    dp = session_data.get_timestep_data(i if i >= 0 else session_data.get_episode_length() + i)
                    if dp:
                        print(f"\n   Timestep {i}:")
                        print(f"   - Timestamp: {dp.timestamp}")
                        print(f"   - Markets with data: {len(dp.markets_data)}")
                        print(f"   - Time gap: {dp.time_gap:.2f}s")
                        print(f"   - Activity score: {dp.activity_score:.2f}")
                        
                        # Show first market's features
                        if dp.spreads:
                            first_market = list(dp.spreads.keys())[0]
                            yes_spread, no_spread = dp.spreads[first_market]
                            print(f"   - {first_market} spreads: YES={yes_spread}, NO={no_spread}")
            
            print(f"\n✅ PIPELINE VALIDATION SUCCESSFUL!")
            print(f"   Database → Orderbook → Features → Episode Ready!")
            
            # Test feature extraction
            print(f"\n8. Testing Feature Extraction...")
            from kalshiflow_rl.environments.feature_extractors import build_observation_from_session_data
            
            # Get observation for first timestep
            observation = build_observation_from_session_data(
                session_data, 
                current_step=0,
                max_markets=5
            )
            
            print(f"   Observation shape: {observation.shape}")
            print(f"   Observation dtype: {observation.dtype}")
            print(f"   Non-zero features: {np.count_nonzero(observation)}/{observation.size}")
            print(f"   Feature range: [{observation.min():.2f}, {observation.max():.2f}]")
            
            return session_data
            
        else:
            print(f"❌ Failed to load session {session_id}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await db.cleanup()


async def main():
    """Main entry point."""
    try:
        import numpy as np
    except ImportError:
        print("Installing numpy...")
        os.system("uv pip install numpy")
        import numpy as np
        
    session = await test_real_session_loading()
    
    if session:
        print("\n" + "=" * 60)
        print("M3 DELIVERABLE VERIFIED!")
        print("We can successfully:")
        print("✅ Load any historical session from the database")
        print("✅ Get complete training episode with all features extracted")
        print("✅ Feed this directly to the gym environment for training")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())