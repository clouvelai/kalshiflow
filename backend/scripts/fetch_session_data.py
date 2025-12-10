#!/usr/bin/env python3
"""
Fetch and display session data from the RL orderbook database.

This script provides utilities to list available sessions or load a specific
session through the M3 pipeline (Database → Orderbook → Features → Episode).

Usage:
    # List all available sessions
    python fetch_session_data.py --list
    
    # Load specific session
    python fetch_session_data.py 6
    
    # Load most recent session
    python fetch_session_data.py
    
Examples:
    $ python fetch_session_data.py --list
    Shows all sessions with their metadata
    
    $ python fetch_session_data.py 6
    Loads session 6 and displays episode summary
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshiflow_rl.environments.session_data_loader import SessionDataLoader


async def list_available_sessions():
    """List all available sessions in the database."""
    
    print("=" * 80)
    print("AVAILABLE SESSIONS IN DATABASE")
    print("=" * 80)
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return
    
    # Use SessionDataLoader to get sessions with metadata
    loader = SessionDataLoader(database_url=database_url)
    sessions = await loader.get_available_sessions()
    
    if not sessions:
        print("No sessions found in database.")
        return
    
    print(f"\nFound {len(sessions)} closed sessions:\n")
    
    # Header
    print(f"{'ID':>6} {'Status':<10} {'Duration':<12} {'Markets':>8} {'Snapshots':>10} {'Deltas':>8} {'Start Time':<20}")
    print("-" * 80)
    
    # Sessions
    for session in sessions:
        duration = session.get('duration')
        if duration:
            # Format duration as MM:SS
            total_seconds = int(duration.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            duration_str = f"{minutes:02d}:{seconds:02d}"
        else:
            duration_str = "N/A"
            
        markets = session.get('num_markets', 0)
        if markets == 0 and session.get('market_tickers'):
            markets = len(session['market_tickers'])
            
        print(f"{session['session_id']:>6} "
              f"{session.get('status', 'unknown'):<10} "
              f"{duration_str:<12} "
              f"{markets:>8} "
              f"{session.get('snapshots_count', 0):>10} "
              f"{session.get('deltas_count', 0):>8} "
              f"{str(session.get('started_at', 'N/A'))[:19]:<20}")
    
    print("\n" + "=" * 80)
    print("Use 'python fetch_session_data.py <session_id>' to load a specific session")
    print("=" * 80)


async def load_session_data(session_id: int = None):
    """Load and display a specific session's data through the M3 pipeline."""
    
    print("=" * 60)
    print("LOADING SESSION DATA: DB → ORDERBOOK → FEATURES → EPISODE")
    print("=" * 60)
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return None
    
    # Initialize SessionDataLoader
    print("\n1. Initializing SessionDataLoader...")
    loader = SessionDataLoader(database_url=database_url)
    
    # If no session_id provided, find the most recent one
    if session_id is None:
        print("   No session_id provided, finding most recent...")
        # Quick query to get latest session
        import asyncpg
        conn = await asyncpg.connect(database_url)
        try:
            row = await conn.fetchrow("""
                SELECT session_id, snapshots_count, deltas_count 
                FROM rl_orderbook_sessions 
                WHERE status = 'closed' 
                ORDER BY session_id DESC 
                LIMIT 1
            """)
            if row:
                session_id = row['session_id']
                print(f"   Found session {session_id} ({row['snapshots_count']} snapshots, {row['deltas_count']} deltas)")
            else:
                print("❌ No closed sessions found!")
                return None
        finally:
            await conn.close()
    
    # Load the session
    print(f"\n2. Loading session {session_id}...")
    session_data = await loader.load_session(session_id)
    
    if not session_data:
        print(f"❌ Failed to load session {session_id}")
        return None
    
    print("✅ Session loaded successfully!")
    
    # Display concise summary
    print(f"\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 EPISODE METRICS:")
    print(f"   Session ID:        {session_data.session_id}")
    print(f"   Episode Length:    {session_data.get_episode_length()} timesteps")
    print(f"   Duration:          {session_data.total_duration}")
    print(f"   Markets:           {len(session_data.markets_involved)} active")
    
    print(f"\n📈 DATA QUALITY:")
    print(f"   Quality Score:     {session_data.data_quality_score:.2%}")
    print(f"   Avg Spread:        ${session_data.avg_spread:.4f}")
    print(f"   Volatility:        {session_data.volatility_score:.3f}")
    print(f"   Market Diversity:  {session_data.market_diversity:.2f}")
    
    print(f"\n⚡ TEMPORAL FEATURES:")
    print(f"   Activity Bursts:   {len(session_data.activity_bursts)} periods")
    print(f"   Quiet Periods:     {len(session_data.quiet_periods)} periods")
    if session_data.temporal_gaps:
        avg_gap = sum(session_data.temporal_gaps) / len(session_data.temporal_gaps)
        print(f"   Avg Time Gap:      {avg_gap:.2f} seconds")
    
    print(f"\n🎯 SAMPLE DATA POINTS:")
    sample_indices = [0, session_data.get_episode_length() // 2, -1]
    for idx in sample_indices:
        actual_idx = idx if idx >= 0 else session_data.get_episode_length() + idx
        dp = session_data.get_timestep_data(actual_idx)
        if dp:
            print(f"   Step {actual_idx:4d}: {len(dp.markets_data):3d} markets, "
                  f"gap={dp.time_gap:5.1f}s, activity={dp.activity_score:.2f}")
    
    print(f"\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE: Ready for training!")
    print("=" * 60)
    
    return session_data


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch and display RL session data from database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_session_data.py --list       # List all available sessions
  python fetch_session_data.py 6            # Load session 6
  python fetch_session_data.py              # Load most recent session
        """
    )
    
    parser.add_argument('session_id', type=int, nargs='?', 
                       help='Session ID to load (uses latest if not provided)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available sessions instead of loading data')
    
    args = parser.parse_args()
    
    try:
        if args.list:
            # List available sessions
            await list_available_sessions()
        else:
            # Load session data
            session_data = await load_session_data(args.session_id)
        
            if session_data:
                print(f"\n✨ Success! Session {session_data.session_id} is ready for training.")
                print(f"   This episode contains {session_data.get_episode_length()} timesteps")
                print(f"   across {len(session_data.markets_involved)} markets.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())