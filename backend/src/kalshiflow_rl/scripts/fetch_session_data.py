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
    
    # Analyze session activity
    python fetch_session_data.py --analyze 9
    
Examples:
    $ python fetch_session_data.py --list
    Shows all sessions with their metadata
    
    $ python fetch_session_data.py 6
    Loads session 6 and displays episode summary
    
    $ python fetch_session_data.py --analyze 9
    Analyzes market activity distribution for session 9
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import asyncpg
from collections import defaultdict
import numpy as np

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
    print(f"{'ID':>6} {'Env':<10} {'Status':<10} {'Duration':<12} {'Markets':>8} {'Snapshots':>10} {'Deltas':>8} {'Start Time':<20}")
    print("-" * 110)
    
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
            
        env = session.get('environment', 'unknown')[:10]  # Truncate to fit column
        print(f"{session['session_id']:>6} "
              f"{env:<10} "
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
    print(f"   Environment:       {session_data.environment or 'unknown'}")
    print(f"   WebSocket URL:     {session_data.websocket_url or 'not recorded'}")
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


async def analyze_session_activity(session_id: int = None):
    """
    Analyze market activity distribution and selection strategies for a session.
    
    Provides detailed analysis of:
    - Market activity distribution across timesteps
    - Comparison of early-bias vs overall selection strategies
    - Temporal coverage patterns
    - Orderbook change intensity
    """
    print("=" * 80)
    print("SESSION ACTIVITY ANALYSIS")
    print("=" * 80)
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return None
    
    # Initialize SessionDataLoader
    loader = SessionDataLoader(database_url=database_url)
    
    # If no session_id provided, find the most recent one
    if session_id is None:
        conn = await asyncpg.connect(database_url)
        try:
            row = await conn.fetchrow("""
                SELECT session_id FROM rl_orderbook_sessions 
                WHERE status = 'closed' 
                ORDER BY session_id DESC 
                LIMIT 1
            """)
            if row:
                session_id = row['session_id']
            else:
                print("❌ No closed sessions found!")
                return None
        finally:
            await conn.close()
    
    print(f"\nAnalyzing session {session_id}...")
    session_data = await loader.load_session(session_id)
    
    if not session_data:
        print(f"❌ Failed to load session {session_id}")
        return None
    
    print(f"✅ Session loaded: {session_data.get_episode_length()} timesteps, {len(session_data.markets_involved)} markets")
    
    # 1. Overall market activity statistics
    print("\n" + "=" * 80)
    print("1. MARKET ACTIVITY DISTRIBUTION")
    print("-" * 80)
    
    market_activity = defaultdict(lambda: {'count': 0, 'volume': 0, 'changes': []})
    market_presence_by_quartile = defaultdict(lambda: [0, 0, 0, 0])
    prev_market_volumes = {}
    q_size = len(session_data.data_points) // 4
    
    # Analyze each timestep
    timesteps_with_multiple = 0
    max_markets_per_timestep = 0
    
    for i, dp in enumerate(session_data.data_points):
        num_markets = len(dp.markets_data)
        if num_markets > 1:
            timesteps_with_multiple += 1
        max_markets_per_timestep = max(max_markets_per_timestep, num_markets)
        
        q_idx = min(i // q_size, 3) if q_size > 0 else 0
        
        for market_ticker, market_data in dp.markets_data.items():
            market_activity[market_ticker]['count'] += 1
            market_presence_by_quartile[market_ticker][q_idx] += 1
            
            # Calculate orderbook volume
            current_volume = 0
            for side in ['yes_bids', 'yes_asks', 'no_bids', 'no_asks']:
                if side in market_data:
                    current_volume += sum(market_data[side].values())
            
            market_activity[market_ticker]['volume'] += current_volume
            
            # Track changes (deltas)
            if market_ticker in prev_market_volumes:
                volume_change = abs(current_volume - prev_market_volumes[market_ticker])
                if volume_change > 0:
                    market_activity[market_ticker]['changes'].append(volume_change)
            
            prev_market_volumes[market_ticker] = current_volume
    
    print(f"\nSession Statistics:")
    print(f"  Total timesteps:              {len(session_data.data_points)}")
    print(f"  Unique markets:               {len(market_activity)}")
    print(f"  Max markets per timestep:     {max_markets_per_timestep}")
    print(f"  Timesteps with >1 market:     {timesteps_with_multiple} ({100*timesteps_with_multiple/len(session_data.data_points):.1f}%)")
    
    # 2. Top markets by different metrics
    print("\n" + "=" * 80)
    print("2. TOP MARKETS BY ACTIVITY METRICS")
    print("-" * 80)
    
    # Sort by volume
    top_by_volume = sorted(market_activity.items(), 
                          key=lambda x: x[1]['volume'], 
                          reverse=True)[:10]
    
    print("\nTop 10 by Total Orderbook Volume:")
    print(f"{'Market':<40} {'Volume':>15} {'Timesteps':>10} {'Changes':>10}")
    print("-" * 80)
    for market, stats in top_by_volume:
        print(f"{market[:38]:<40} {stats['volume']:>15,.0f} {stats['count']:>10} {len(stats['changes']):>10}")
    
    # Sort by changes (delta activity)
    top_by_changes = sorted(market_activity.items(), 
                           key=lambda x: len(x[1]['changes']), 
                           reverse=True)[:10]
    
    print("\nTop 10 by Orderbook Changes (Deltas):")
    print(f"{'Market':<40} {'Changes':>10} {'Avg Change':>15} {'Timesteps':>10}")
    print("-" * 80)
    for market, stats in top_by_changes:
        avg_change = np.mean(stats['changes']) if stats['changes'] else 0
        print(f"{market[:38]:<40} {len(stats['changes']):>10} {avg_change:>15,.0f} {stats['count']:>10}")
    
    # 3. Market selection comparison
    print("\n" + "=" * 80)
    print("3. MARKET SELECTION STRATEGY COMPARISON")
    print("-" * 80)
    
    # Early-bias approach (first 20%)
    early_timesteps = max(1, int(len(session_data.data_points) * 0.2))
    early_markets = set()
    
    for step in range(min(early_timesteps, len(session_data.data_points))):
        data_point = session_data.data_points[step]
        early_markets.update(data_point.markets_data.keys())
    
    # Calculate activity for early markets only
    early_activity = {}
    for market, stats in market_activity.items():
        if market in early_markets:
            early_activity[market] = stats['volume']
    
    early_winner = max(early_activity.items(), key=lambda x: x[1])[0] if early_activity else None
    
    # Overall approach (all timesteps)
    overall_winner = top_by_volume[0][0] if top_by_volume else None
    
    print(f"\nEarly-Bias Approach (first {early_timesteps} timesteps = 20%):")
    print(f"  Candidate markets:    {len(early_markets)}")
    print(f"  Selected market:      {early_winner}")
    if early_winner:
        print(f"  Volume:              {market_activity[early_winner]['volume']:,.0f}")
        print(f"  Total timesteps:     {market_activity[early_winner]['count']}")
    
    print(f"\nOverall Approach (all timesteps):")
    print(f"  Selected market:      {overall_winner}")
    if overall_winner:
        print(f"  Volume:              {market_activity[overall_winner]['volume']:,.0f}")
        print(f"  Total timesteps:     {market_activity[overall_winner]['count']}")
    
    if early_winner != overall_winner:
        print(f"\n⚠️  Different selection! Early-bias would miss the most active market.")
    else:
        print(f"\n✅ Both approaches select the same market.")
    
    # 4. Temporal coverage analysis
    print("\n" + "=" * 80)
    print("4. TEMPORAL COVERAGE ANALYSIS")
    print("-" * 80)
    
    markets_to_analyze = []
    if early_winner:
        markets_to_analyze.append(("Early-bias winner", early_winner))
    if overall_winner and overall_winner != early_winner:
        markets_to_analyze.append(("Overall winner", overall_winner))
    
    # Add top delta-active market if different
    delta_winner = top_by_changes[0][0] if top_by_changes else None
    if delta_winner and delta_winner not in [early_winner, overall_winner]:
        markets_to_analyze.append(("Most changes", delta_winner))
    
    for label, market in markets_to_analyze:
        quartiles = market_presence_by_quartile[market]
        total_presence = sum(quartiles)
        
        print(f"\n{label}: {market}")
        print(f"  Total presence:      {total_presence} timesteps ({100*total_presence/len(session_data.data_points):.1f}%)")
        print(f"  Q1 (0-25%):         {quartiles[0]:5d} timesteps ({100*quartiles[0]/max(total_presence,1):.1f}% of market's activity)")
        print(f"  Q2 (25-50%):        {quartiles[1]:5d} timesteps ({100*quartiles[1]/max(total_presence,1):.1f}% of market's activity)")
        print(f"  Q3 (50-75%):        {quartiles[2]:5d} timesteps ({100*quartiles[2]/max(total_presence,1):.1f}% of market's activity)")
        print(f"  Q4 (75-100%):       {quartiles[3]:5d} timesteps ({100*quartiles[3]/max(total_presence,1):.1f}% of market's activity)")
        
        # Determine activity pattern
        if quartiles[0] > sum(quartiles[1:]):
            pattern = "Front-loaded (most activity early)"
        elif quartiles[3] > sum(quartiles[:3]):
            pattern = "Back-loaded (most activity late)"
        elif quartiles[1] + quartiles[2] > quartiles[0] + quartiles[3]:
            pattern = "Mid-session peak"
        else:
            pattern = "Evenly distributed"
        print(f"  Pattern:            {pattern}")
    
    # 5. Recommendations
    print("\n" + "=" * 80)
    print("5. TRAINING RECOMMENDATIONS")
    print("-" * 80)
    
    # Find markets suitable for training (>50 timesteps with good distribution)
    suitable_markets = []
    for market, stats in market_activity.items():
        if stats['count'] >= 50:
            quartiles = market_presence_by_quartile[market]
            # Check if market has presence in at least 2 quartiles
            active_quartiles = sum(1 for q in quartiles if q > 0)
            if active_quartiles >= 2:
                suitable_markets.append((market, stats['volume'], stats['count'], active_quartiles))
    
    suitable_markets.sort(key=lambda x: x[1], reverse=True)  # Sort by volume
    
    print(f"\nMarkets suitable for training (≥50 timesteps, ≥2 quartiles):")
    print(f"  Found {len(suitable_markets)} suitable markets out of {len(market_activity)} total")
    
    if suitable_markets:
        print(f"\n  Top 5 recommended markets:")
        print(f"  {'Market':<40} {'Volume':>15} {'Timesteps':>10} {'Quartiles':>10}")
        print("  " + "-" * 78)
        for market, volume, count, quartiles in suitable_markets[:5]:
            print(f"  {market[:38]:<40} {volume:>15,.0f} {count:>10} {quartiles:>10}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)


async def get_market_session_view(session_id: int = None, market_ticker: str = None):
    """
    Load a specific market view from a session and display its statistics.
    
    Creates a MarketSessionView for efficient single-market access and shows
    detailed statistics about the market's presence and characteristics.
    """
    print("=" * 80)
    print("MARKET SESSION VIEW ANALYSIS")
    print("=" * 80)
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        return None
    
    # Initialize SessionDataLoader
    loader = SessionDataLoader(database_url=database_url)
    
    # If no session_id provided, find the most recent one
    if session_id is None:
        conn = await asyncpg.connect(database_url)
        try:
            row = await conn.fetchrow("""
                SELECT session_id FROM rl_orderbook_sessions 
                WHERE status = 'closed' 
                ORDER BY session_id DESC 
                LIMIT 1
            """)
            if row:
                session_id = row['session_id']
                print(f"Using most recent session: {session_id}")
            else:
                print("❌ No closed sessions found!")
                return None
        finally:
            await conn.close()
    
    print(f"\nLoading session {session_id}...")
    session_data = await loader.load_session(session_id)
    
    if not session_data:
        print(f"❌ Failed to load session {session_id}")
        return None
    
    print(f"✅ Session loaded: {session_data.get_episode_length()} timesteps, {len(session_data.markets_involved)} markets")
    
    # If no market specified, show available markets and let user choose
    if market_ticker is None:
        print("\n" + "=" * 80)
        print("AVAILABLE MARKETS IN SESSION")
        print("-" * 80)
        
        # Analyze markets to help with selection
        market_stats = {}
        for dp in session_data.data_points:
            for market in dp.markets_data:
                if market not in market_stats:
                    market_stats[market] = {'count': 0, 'volume': 0}
                market_stats[market]['count'] += 1
                
                # Calculate volume
                market_data = dp.markets_data[market]
                for side in ['yes_bids', 'yes_asks', 'no_bids', 'no_asks']:
                    if side in market_data:
                        market_stats[market]['volume'] += sum(market_data[side].values())
        
        # Show top markets
        top_markets = sorted(market_stats.items(), key=lambda x: x[1]['volume'], reverse=True)[:10]
        
        print(f"\nTop 10 markets by volume (choose one for --market option):")
        print(f"{'Market':<40} {'Timesteps':>10} {'Coverage':>10} {'Volume':>15}")
        print("-" * 80)
        for market, stats in top_markets:
            coverage = 100 * stats['count'] / session_data.get_episode_length()
            print(f"{market[:38]:<40} {stats['count']:>10} {coverage:>9.1f}% {stats['volume']:>15,.0f}")
        
        print(f"\n💡 Tip: Run with --market <ticker> to create a view for a specific market")
        return None
    
    # Create market view
    print(f"\n" + "=" * 80)
    print(f"CREATING MARKET VIEW: {market_ticker}")
    print("-" * 80)
    
    try:
        market_view = session_data.create_market_view(market_ticker)
        
        if market_view is None:
            print(f"❌ Market {market_ticker} not found in session")
            return None
        
        print(f"✅ Market view created successfully!")
        
        # Display market view statistics
        print(f"\n" + "=" * 80)
        print("MARKET VIEW STATISTICS")
        print("-" * 80)
        
        print(f"\n📊 BASIC METRICS:")
        print(f"   Session ID:        {market_view.session_id}")
        print(f"   Market:            {market_view.target_market}")
        print(f"   Data Points:       {market_view.get_episode_length()} timesteps")
        print(f"   Coverage:          {100 * market_view.get_episode_length() / session_data.get_episode_length():.1f}% of session")
        print(f"   Duration:          {market_view.total_duration}")
        
        print(f"\n📈 DATA QUALITY:")
        print(f"   Quality Score:     {market_view.data_quality_score:.2%}")
        print(f"   Avg Spread:        ${market_view.avg_spread:.4f}")
        print(f"   Volatility:        {market_view.volatility_score:.3f}")
        
        print(f"\n⚡ TEMPORAL CHARACTERISTICS:")
        print(f"   Activity Bursts:   {len(market_view.activity_bursts)} periods")
        print(f"   Quiet Periods:     {len(market_view.quiet_periods)} periods")
        if market_view.temporal_gaps:
            avg_gap = np.mean(market_view.temporal_gaps)
            max_gap = np.max(market_view.temporal_gaps)
            print(f"   Avg Time Gap:      {avg_gap:.2f} seconds")
            print(f"   Max Time Gap:      {max_gap:.2f} seconds")
        
        # Analyze temporal distribution
        print(f"\n📅 TEMPORAL DISTRIBUTION:")
        if market_view.get_episode_length() >= 4:
            q_size = market_view.get_episode_length() // 4
            quartiles = [0, 0, 0, 0]
            
            for i in range(market_view.get_episode_length()):
                q_idx = min(i // q_size, 3)
                quartiles[q_idx] += 1
            
            total = sum(quartiles)
            print(f"   Q1 (0-25%):        {quartiles[0]:4d} timesteps ({100*quartiles[0]/total:.1f}%)")
            print(f"   Q2 (25-50%):       {quartiles[1]:4d} timesteps ({100*quartiles[1]/total:.1f}%)")
            print(f"   Q3 (50-75%):       {quartiles[2]:4d} timesteps ({100*quartiles[2]/total:.1f}%)")
            print(f"   Q4 (75-100%):      {quartiles[3]:4d} timesteps ({100*quartiles[3]/total:.1f}%)")
            
            # Determine pattern
            if quartiles[0] > sum(quartiles[1:]):
                pattern = "Front-loaded (most activity early)"
            elif quartiles[3] > sum(quartiles[:3]):
                pattern = "Back-loaded (most activity late)"
            elif quartiles[1] + quartiles[2] > quartiles[0] + quartiles[3]:
                pattern = "Mid-session peak"
            else:
                pattern = "Evenly distributed"
            print(f"   Pattern:           {pattern}")
        
        print(f"\n🎯 SAMPLE DATA POINTS:")
        sample_indices = [0, market_view.get_episode_length() // 2, -1]
        for idx in sample_indices:
            actual_idx = idx if idx >= 0 else market_view.get_episode_length() + idx
            if 0 <= actual_idx < market_view.get_episode_length():
                dp = market_view.get_timestep_data(actual_idx)
                if dp and market_view.target_market in dp.markets_data:
                    market_data = dp.markets_data[market_view.target_market]
                    volume = 0
                    for side in ['yes_bids', 'yes_asks', 'no_bids', 'no_asks']:
                        if side in market_data:
                            volume += sum(market_data[side].values())
                    print(f"   Step {actual_idx:4d}: gap={dp.time_gap:5.1f}s, activity={dp.activity_score:.2f}, volume={volume:8.0f}")
        
        print(f"\n" + "=" * 80)
        print("✅ MARKET VIEW READY FOR TRAINING")
        print("=" * 80)
        
        return market_view
        
    except Exception as e:
        print(f"❌ Error creating market view: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch and display RL session data from database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_session_data.py --list                    # List all available sessions
  python fetch_session_data.py 6                         # Load session 6
  python fetch_session_data.py                           # Load most recent session
  python fetch_session_data.py --analyze 9               # Analyze activity for session 9
  python fetch_session_data.py --analyze                 # Analyze most recent session
  python fetch_session_data.py --view 9                  # Show available markets for session 9
  python fetch_session_data.py --view 9 --market TICKER  # Create view for specific market
        """
    )
    
    parser.add_argument('session_id', type=int, nargs='?', 
                       help='Session ID to load (uses latest if not provided)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List all available sessions instead of loading data')
    parser.add_argument('--analyze', '-a', action='store_true',
                       help='Analyze market activity distribution and selection strategies')
    parser.add_argument('--view', '-v', action='store_true',
                       help='Create and analyze a market-specific session view')
    parser.add_argument('--market', '-m', type=str,
                       help='Market ticker for creating a specific market view (use with --view)')
    
    args = parser.parse_args()
    
    try:
        if args.list:
            # List available sessions
            await list_available_sessions()
        elif args.analyze:
            # Analyze session activity
            await analyze_session_activity(args.session_id)
        elif args.view:
            # Create and analyze market view
            await get_market_session_view(args.session_id, args.market)
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