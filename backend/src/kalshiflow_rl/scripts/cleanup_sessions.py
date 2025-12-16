#!/usr/bin/env python3
"""
Session cleanup tool for RL orderbook database.

Provides utilities to identify and remove empty, test, or old sessions
from the database with safety features and reporting.

Usage:
    # List empty sessions (dry run)
    python cleanup_sessions.py --list-empty
    
    # Delete specific sessions
    python cleanup_sessions.py --delete 2,3,8,18-22
    
    # Delete all empty sessions (with confirmation)
    python cleanup_sessions.py --delete-empty
    
    # Delete test sessions (< 5 min, < 5 markets)
    python cleanup_sessions.py --delete-test
    
    # Generate cleanup report
    python cleanup_sessions.py --report
    
    # Show overall statistics
    python cleanup_sessions.py --stats
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import argparse
from typing import List, Set
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kalshiflow_rl.data.database import RLDatabase


def parse_session_ids(session_str: str) -> List[int]:
    """Parse session ID string like '2,3,8,18-22' into list of IDs."""
    session_ids = set()
    
    for part in session_str.split(','):
        part = part.strip()
        if '-' in part:
            # Handle range
            start, end = part.split('-')
            start, end = int(start), int(end)
            session_ids.update(range(start, end + 1))
        else:
            # Single ID
            session_ids.add(int(part))
    
    return sorted(list(session_ids))


def format_duration(duration):
    """Format timedelta as human-readable string."""
    if duration is None:
        return "N/A"
    
    total_seconds = int(duration.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    
    if days > 0:
        return f"{days}d {hours:02d}h"
    elif hours > 0:
        return f"{hours:02d}:{minutes:02d}"
    else:
        return f"{minutes:02d}:{total_seconds % 60:02d}"


def print_session_table(sessions: List[dict], title: str):
    """Print formatted table of sessions."""
    if not sessions:
        print("No sessions found.")
        return
    
    print(f"\n{title}")
    print("=" * 100)
    print(f"{'ID':>6} {'Status':<10} {'Duration':<12} {'Markets':>8} {'Snapshots':>10} {'Deltas':>8} {'Start Time':<20}")
    print("-" * 100)
    
    for session in sessions:
        duration_str = format_duration(session.get('duration'))
        markets = len(session.get('market_tickers', [])) if session.get('market_tickers') else 0
        
        print(f"{session['session_id']:>6} "
              f"{session.get('status', 'unknown'):<10} "
              f"{duration_str:<12} "
              f"{markets:>8} "
              f"{session.get('snapshots_count', 0):>10} "
              f"{session.get('deltas_count', 0):>8} "
              f"{str(session.get('started_at', 'N/A'))[:19]:<20}")
    
    print(f"\nTotal: {len(sessions)} sessions")


async def confirm_action(prompt: str) -> bool:
    """Ask user for confirmation."""
    response = input(f"\n{prompt} (yes/no): ").lower().strip()
    return response in ['yes', 'y']


async def list_empty_sessions(db: RLDatabase):
    """List all empty sessions."""
    sessions = await db.get_empty_sessions()
    print_session_table(sessions, "EMPTY SESSIONS (No Data)")
    
    if sessions:
        # Calculate space that would be freed
        print(f"\n⚠️  These {len(sessions)} sessions contain no data and can be safely deleted.")
        session_ids = [s['session_id'] for s in sessions]
        print(f"   Session IDs: {','.join(map(str, session_ids[:10]))}" + 
              (f"... ({len(session_ids)-10} more)" if len(session_ids) > 10 else ""))
    
    return sessions


async def list_test_sessions(db: RLDatabase):
    """List all test sessions."""
    sessions = await db.get_test_sessions()
    print_session_table(sessions, "TEST SESSIONS (< 5 min, <= 5 markets)")
    
    if sessions:
        print(f"\n⚠️  These {len(sessions)} sessions appear to be test runs.")
        session_ids = [s['session_id'] for s in sessions]
        print(f"   Session IDs: {','.join(map(str, session_ids[:10]))}" + 
              (f"... ({len(session_ids)-10} more)" if len(session_ids) > 10 else ""))
    
    return sessions


async def delete_sessions_with_confirmation(db: RLDatabase, session_ids: List[int], description: str):
    """Delete sessions with user confirmation."""
    if not session_ids:
        print("No sessions to delete.")
        return
    
    print(f"\n🗑️  PREPARING TO DELETE {len(session_ids)} {description}")
    print(f"   Session IDs: {','.join(map(str, session_ids[:20]))}" + 
          (f"... ({len(session_ids)-20} more)" if len(session_ids) > 20 else ""))
    
    if not await confirm_action(f"Are you sure you want to delete {len(session_ids)} sessions?"):
        print("❌ Deletion cancelled.")
        return
    
    print(f"\n🔄 Deleting {len(session_ids)} sessions...")
    
    # Perform deletion with progress tracking
    results = await db.delete_sessions(session_ids)
    
    # Count successes and failures
    succeeded = [r for r in results if r.get('status') == 'deleted']
    failed = [r for r in results if r.get('status') != 'deleted']
    
    # Calculate totals
    total_snapshots = sum(r.get('snapshots_deleted', 0) for r in succeeded)
    total_deltas = sum(r.get('deltas_deleted', 0) for r in succeeded)
    
    print(f"\n✅ DELETION COMPLETE")
    print(f"   Sessions deleted: {len(succeeded)}")
    print(f"   Snapshots removed: {total_snapshots:,}")
    print(f"   Deltas removed: {total_deltas:,}")
    
    if failed:
        print(f"\n❌ Failed to delete {len(failed)} sessions:")
        for r in failed[:5]:
            print(f"   Session {r['session_id']}: {r.get('error', 'Unknown error')}")
        if len(failed) > 5:
            print(f"   ... and {len(failed)-5} more")
    
    # Write deletion log
    log_file = f"cleanup_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'total_sessions': len(session_ids),
            'succeeded': len(succeeded),
            'failed': len(failed),
            'snapshots_deleted': total_snapshots,
            'deltas_deleted': total_deltas,
            'results': results
        }, f, indent=2, default=str)
    
    print(f"\n📝 Deletion log saved to: {log_file}")


async def generate_report(db: RLDatabase):
    """Generate comprehensive cleanup report."""
    print("\n" + "=" * 80)
    print("SESSION CLEANUP REPORT")
    print("=" * 80)
    
    # Get overall statistics
    stats = await db.get_session_statistics()
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total sessions:        {stats['total_sessions']}")
    print(f"   Sessions with data:    {stats['sessions_with_data']} ({100*stats['sessions_with_data']/max(stats['total_sessions'],1):.1f}%)")
    print(f"   Empty sessions:        {stats['empty_sessions']} ({100*stats['empty_sessions']/max(stats['total_sessions'],1):.1f}%)")
    print(f"   Test sessions:         {stats['test_sessions']} ({100*stats['test_sessions']/max(stats['total_sessions'],1):.1f}%)")
    print(f"   Total snapshots:       {stats['total_snapshots']:,}")
    print(f"   Total deltas:          {stats['total_deltas']:,}")
    
    # Get environment breakdown
    env_stats = await db.get_environment_statistics()
    if env_stats:
        print(f"\n📍 ENVIRONMENT BREAKDOWN:")
        for env, count in env_stats.items():
            if env:  # Skip NULL environments
                print(f"   {env:<15}    {count} sessions")
        if None in env_stats:
            print(f"   {'unknown':<15}    {env_stats[None]} sessions (no environment recorded)")
    
    # Get empty sessions
    empty = await db.get_empty_sessions()
    test = await db.get_test_sessions()
    
    # Find overlapping sessions
    empty_ids = set(s['session_id'] for s in empty)
    test_ids = set(s['session_id'] for s in test)
    overlap = empty_ids & test_ids
    
    print(f"\n🗑️  CLEANUP CANDIDATES:")
    print(f"   Empty sessions:        {len(empty)} sessions")
    print(f"   Test sessions:         {len(test)} sessions")
    print(f"   Overlap:               {len(overlap)} sessions (both empty and test)")
    print(f"   Unique deletable:      {len(empty_ids | test_ids)} sessions total")
    
    # Group by deletion priority
    all_deletable = empty_ids | test_ids
    
    print(f"\n📋 RECOMMENDED DELETION BATCHES:")
    
    # Batch 1: Pure test sessions (1-5 markets, < 5 minutes)
    batch1 = [s['session_id'] for s in test if s['session_id'] in test_ids and len(s.get('market_tickers', [])) <= 3]
    if batch1:
        print(f"\n   Batch 1 - Test sessions (≤3 markets):")
        print(f"   {','.join(map(str, sorted(batch1)[:20]))}" + 
              (f"... ({len(batch1)-20} more)" if len(batch1) > 20 else ""))
        print(f"   Total: {len(batch1)} sessions")
    
    # Batch 2: Empty short runs
    batch2 = [s['session_id'] for s in empty 
              if s['session_id'] not in batch1 and 
              s.get('duration') and s['duration'].total_seconds() < 3600]
    if batch2:
        print(f"\n   Batch 2 - Empty short runs (<1 hour):")
        print(f"   {','.join(map(str, sorted(batch2)[:20]))}" + 
              (f"... ({len(batch2)-20} more)" if len(batch2) > 20 else ""))
        print(f"   Total: {len(batch2)} sessions")
    
    # Batch 3: Empty long runs (potential issues)
    batch3 = [s['session_id'] for s in empty 
              if s['session_id'] not in batch1 and s['session_id'] not in batch2]
    if batch3:
        print(f"\n   Batch 3 - Empty long runs (>1 hour, potential issues):")
        print(f"   {','.join(map(str, sorted(batch3)[:20]))}" + 
              (f"... ({len(batch3)-20} more)" if len(batch3) > 20 else ""))
        print(f"   Total: {len(batch3)} sessions")
    
    print(f"\n💡 CLEANUP COMMANDS:")
    if batch1:
        print(f"   # Delete test sessions")
        print(f"   python cleanup_sessions.py --delete {','.join(map(str, sorted(batch1)[:10]))}")
    if batch2:
        print(f"   # Delete empty short runs")
        print(f"   python cleanup_sessions.py --delete {','.join(map(str, sorted(batch2)[:10]))}")
    print(f"   # Delete all empty sessions at once")
    print(f"   python cleanup_sessions.py --delete-empty")
    
    print("\n" + "=" * 80)
    print("END OF REPORT")
    print("=" * 80)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='RL session cleanup tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup_sessions.py --list-empty              # List empty sessions
  python cleanup_sessions.py --list-test               # List test sessions
  python cleanup_sessions.py --delete 2,3,8,18-22      # Delete specific sessions
  python cleanup_sessions.py --delete-empty            # Delete all empty sessions
  python cleanup_sessions.py --delete-test             # Delete all test sessions
  python cleanup_sessions.py --report                  # Generate cleanup report
  python cleanup_sessions.py --stats                   # Show database statistics
        """
    )
    
    # List operations
    parser.add_argument('--list-empty', action='store_true',
                       help='List all empty sessions (no data)')
    parser.add_argument('--list-test', action='store_true',
                       help='List all test sessions (short duration, few markets)')
    
    # Delete operations
    parser.add_argument('--delete', type=str,
                       help='Delete specific session IDs (e.g., "2,3,8,18-22")')
    parser.add_argument('--delete-empty', action='store_true',
                       help='Delete all empty sessions')
    parser.add_argument('--delete-test', action='store_true',
                       help='Delete all test sessions')
    
    # Report operations
    parser.add_argument('--report', action='store_true',
                       help='Generate comprehensive cleanup report')
    parser.add_argument('--stats', action='store_true',
                       help='Show overall database statistics')
    
    # Options
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompts (use with caution!)')
    
    args = parser.parse_args()
    
    # Initialize database
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ DATABASE_URL not set in environment")
        sys.exit(1)
    
    db = RLDatabase(database_url=database_url)
    
    try:
        # Handle different operations
        if args.list_empty:
            await list_empty_sessions(db)
        
        elif args.list_test:
            await list_test_sessions(db)
        
        elif args.delete:
            session_ids = parse_session_ids(args.delete)
            await delete_sessions_with_confirmation(db, session_ids, "specified sessions")
        
        elif args.delete_empty:
            empty = await db.get_empty_sessions()
            session_ids = [s['session_id'] for s in empty]
            await delete_sessions_with_confirmation(db, session_ids, "empty sessions")
        
        elif args.delete_test:
            test = await db.get_test_sessions()
            session_ids = [s['session_id'] for s in test]
            await delete_sessions_with_confirmation(db, session_ids, "test sessions")
        
        elif args.report:
            await generate_report(db)
        
        elif args.stats:
            stats = await db.get_session_statistics()
            print("\n📊 DATABASE STATISTICS:")
            for key, value in stats.items():
                if value is not None:
                    if isinstance(value, int) and value > 1000:
                        print(f"   {key:20s}: {value:,}")
                    else:
                        print(f"   {key:20s}: {value}")
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())