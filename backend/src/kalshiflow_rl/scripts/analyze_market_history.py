#!/usr/bin/env python3
"""
Analyze the complete orderbook history for a specific market and session.

Usage:
    python analyze_market_history.py <market_ticker> [--session-id <id>]
    python analyze_market_history.py KXROBOTMARS-35 --session-id 1
    python analyze_market_history.py --list  # List markets with data
    python analyze_market_history.py --sessions  # List all sessions
"""

import asyncio
import asyncpg
import argparse
import sys
from datetime import datetime
from typing import Optional

from ..data.database import rl_db
from ..environments.historical_data_loader import (
    HistoricalDataLoader, 
    DataLoadConfig
)


async def list_sessions():
    """List all orderbook sessions."""
    await rl_db.initialize()
    
    try:
        sessions = await rl_db.get_active_sessions()
        
        # Also get closed sessions
        async with rl_db.get_connection() as conn:
            all_sessions = await conn.fetch("""
                SELECT 
                    session_id,
                    market_tickers,
                    started_at,
                    ended_at,
                    status,
                    messages_received,
                    snapshots_count,
                    deltas_count
                FROM rl_orderbook_sessions
                ORDER BY session_id DESC
                LIMIT 50
            """)
        
        if not all_sessions:
            print("No sessions found")
            return
        
        print("=" * 100)
        print("ORDERBOOK SESSIONS")
        print("=" * 100)
        print(f"{'ID':<6} {'Status':<10} {'Markets':<30} {'Started':<20} {'Duration':<15} {'Messages':<10}")
        print("-" * 100)
        
        for row in all_sessions:
            markets_str = ', '.join(row['market_tickers'][:2])  # Show first 2
            if len(row['market_tickers']) > 2:
                markets_str += f" (+{len(row['market_tickers'])-2})"
            
            started = row['started_at'].strftime('%Y-%m-%d %H:%M:%S')
            
            if row['ended_at']:
                duration = (row['ended_at'] - row['started_at']).total_seconds()
                duration_str = f"{duration:.1f}s"
            else:
                duration_str = "Active"
            
            messages = row.get('messages_received', 0)
            
            print(f"{row['session_id']:<6} {row['status']:<10} {markets_str:<30} {started:<20} {duration_str:<15} {messages:<10}")
        
        print(f"\nTotal: {len(all_sessions)} sessions (showing latest 50)")
        
    finally:
        await rl_db.close()


async def list_markets_with_data(db_url: str):
    """List all markets that have orderbook data."""
    conn = await asyncpg.connect(db_url)
    try:
        markets = await conn.fetch("""
            WITH market_stats AS (
                SELECT 
                    s.market_ticker,
                    COUNT(DISTINCT s.session_id) as session_count,
                    COUNT(*) as total_events,
                    MIN(s.timestamp_ms) as first_seen,
                    MAX(s.timestamp_ms) as last_seen,
                    MAX(s.session_id) as latest_session
                FROM (
                    SELECT market_ticker, timestamp_ms, session_id FROM rl_orderbook_snapshots
                    UNION ALL
                    SELECT market_ticker, timestamp_ms, session_id FROM rl_orderbook_deltas
                ) s
                GROUP BY s.market_ticker
            )
            SELECT * FROM market_stats
            ORDER BY total_events DESC
        """)
        
        if not markets:
            print("No markets found with orderbook data")
            return
        
        print("=" * 100)
        print("MARKETS WITH ORDERBOOK DATA")
        print("=" * 100)
        print(f"{'Market Ticker':<30} {'Sessions':<10} {'Events':<10} {'First Seen':<20} {'Last Seen':<20} {'Latest Session':<10}")
        print("-" * 100)
        
        for row in markets:
            first = datetime.fromtimestamp(row['first_seen']/1000).strftime('%Y-%m-%d %H:%M:%S')
            last = datetime.fromtimestamp(row['last_seen']/1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{row['market_ticker']:<30} {row['session_count']:<10} {row['total_events']:<10} {first:<20} {last:<20} {row['latest_session']:<10}")
        
        print(f"\nTotal: {len(markets)} markets")
        
    finally:
        await conn.close()


async def analyze_market_history(market_ticker: str, db_url: str, session_id: Optional[int] = None):
    """Analyze orderbook history for a specific market and optional session."""
    
    await rl_db.initialize()
    
    try:
        # If session_id provided, use session-specific queries
        if session_id:
            print("=" * 70)
            print(f"SESSION {session_id} ORDERBOOK ANALYSIS: {market_ticker}")
            print("=" * 70)
            
            # Get session info
            session = await rl_db.get_session(session_id)
            if not session:
                print(f"❌ Session {session_id} not found")
                return
            
            if market_ticker not in session['market_tickers']:
                print(f"⚠️ Warning: {market_ticker} not in session's configured markets: {session['market_tickers']}")
            
            print(f"\n📋 Session Info:")
            print(f"  Status: {session['status']}")
            print(f"  Started: {session['started_at']}")
            if session['ended_at']:
                print(f"  Ended: {session['ended_at']}")
                duration = (session['ended_at'] - session['started_at']).total_seconds()
                print(f"  Duration: {duration:.1f} seconds")
            print(f"  Markets: {', '.join(session['market_tickers'])}")
            
            # Get session-specific data
            snapshots = await rl_db.get_session_snapshots(session_id, market_ticker)
            deltas = await rl_db.get_session_deltas(session_id, market_ticker)
            
            print(f"\n📊 Session Data:")
            print(f"  Snapshots: {len(snapshots)}")
            print(f"  Deltas: {len(deltas)}")
            
            if not snapshots and not deltas:
                print(f"\n❌ No data found for {market_ticker} in session {session_id}")
                return
            
            # Analyze the session data
            all_events = []
            for snap in snapshots:
                all_events.append({
                    'type': 'snapshot',
                    'timestamp_ms': snap['timestamp_ms'],
                    'sequence_number': snap['sequence_number'],
                    'data': snap
                })
            
            for delta in deltas:
                all_events.append({
                    'type': 'delta',
                    'timestamp_ms': delta['timestamp_ms'],
                    'sequence_number': delta['sequence_number'],
                    'data': delta
                })
            
            # Sort by sequence number
            all_events.sort(key=lambda x: x['sequence_number'])
            
            if all_events:
                min_time = min(e['timestamp_ms'] for e in all_events)
                max_time = max(e['timestamp_ms'] for e in all_events)
                min_seq = min(e['sequence_number'] for e in all_events)
                max_seq = max(e['sequence_number'] for e in all_events)
                
                print(f"\n📈 Sequence Analysis:")
                print(f"  Time Range: {datetime.fromtimestamp(min_time/1000).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.fromtimestamp(max_time/1000).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Duration: {(max_time - min_time) / 1000:.1f} seconds")
                print(f"  Sequence Range: {min_seq} - {max_seq}")
                print(f"  Total Events: {len(all_events)}")
                
                # Check for sequence gaps
                expected_sequences = max_seq - min_seq + 1
                if expected_sequences != len(all_events):
                    print(f"  ⚠️ Sequence Gaps: {expected_sequences - len(all_events)} missing sequences")
                
                # Show first few and last few events
                print(f"\n📋 Event Timeline:")
                print("  First 5 events:")
                for event in all_events[:5]:
                    timestamp = datetime.fromtimestamp(event['timestamp_ms']/1000).strftime('%H:%M:%S')
                    print(f"    [{timestamp}] Seq {event['sequence_number']:6} - {event['type'].upper()}")
                
                if len(all_events) > 10:
                    print("  ...")
                
                print("  Last 5 events:")
                for event in all_events[-5:]:
                    timestamp = datetime.fromtimestamp(event['timestamp_ms']/1000).strftime('%H:%M:%S')
                    print(f"    [{timestamp}] Seq {event['sequence_number']:6} - {event['type'].upper()}")
            
            await rl_db.close()
            return
        
        # Original logic for analyzing all data (no specific session)
        print("=" * 70)
        print(f"ORDERBOOK HISTORY ANALYSIS: {market_ticker} (ALL SESSIONS)")
        print("=" * 70)
    
    conn = await asyncpg.connect(db_url)
    try:
        # Get time range and event counts
        stats = await conn.fetchrow("""
            WITH all_events AS (
                SELECT timestamp_ms, sequence_number, session_id, 'snapshot' as event_type
                FROM rl_orderbook_snapshots 
                WHERE market_ticker = $1
                UNION ALL
                SELECT timestamp_ms, sequence_number, session_id, 'delta' as event_type
                FROM rl_orderbook_deltas 
                WHERE market_ticker = $1
            )
            SELECT 
                MIN(timestamp_ms) as min_time,
                MAX(timestamp_ms) as max_time,
                COUNT(*) as total_events,
                COUNT(CASE WHEN event_type = 'snapshot' THEN 1 END) as snapshot_count,
                COUNT(CASE WHEN event_type = 'delta' THEN 1 END) as delta_count,
                MIN(sequence_number) as min_seq,
                MAX(sequence_number) as max_seq,
                COUNT(DISTINCT session_id) as session_count
            FROM all_events
        """, market_ticker)
        
        if not stats or stats['total_events'] == 0:
            print(f"❌ No orderbook data found for {market_ticker}")
            return
        
        print(f"\n📊 Data Overview:")
        print(f"  Time Range: {datetime.fromtimestamp(stats['min_time']/1000).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.fromtimestamp(stats['max_time']/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {(stats['max_time'] - stats['min_time']) / 1000:.1f} seconds")
        print(f"  Total Events: {stats['total_events']} ({stats['snapshot_count']} snapshots, {stats['delta_count']} deltas)")
        print(f"  Sessions: {stats.get('session_count', 1)}")
        print(f"  Sequence Range: {stats['min_seq']} - {stats['max_seq']}")
        
    finally:
        await conn.close()
    
    # Load and reconstruct full history
    loader = HistoricalDataLoader(db_url)
    
    try:
        config = DataLoadConfig(
            market_tickers=[market_ticker],
            start_time=datetime.fromtimestamp(stats['min_time']/1000),
            end_time=datetime.fromtimestamp(stats['max_time']/1000),
            preload_strategy="reconstruct",
            min_activity_threshold=0,
            include_inactive_periods=True,
            validate_sequences=True
        )
        
        print("\n⚙️ Reconstructing orderbook states...")
        data = await loader.load_historical_data(config)
        
        if market_ticker not in data:
            print(f"❌ Failed to reconstruct data for {market_ticker}")
            return
            
        states = data[market_ticker]
        print(f"✅ Reconstructed {len(states)} orderbook states\n")
        
        # Analyze the history
        print("📈 ORDERBOOK EVOLUTION:")
        print("-" * 70)
        
        # Track metrics
        bid_changes = []
        ask_changes = []
        volume_changes = []
        significant_events = []
        
        prev_state = None
        for i, state in enumerate(states):
            ob = state.orderbook_state
            timestamp = datetime.fromtimestamp(state.timestamp_ms/1000)
            
            # Calculate best bid/ask
            yes_best_bid = None
            yes_best_ask = None
            
            if ob.get('yes_bids'):
                bid_prices = [int(p) for p in ob['yes_bids'].keys()]
                if bid_prices:
                    yes_best_bid = max(bid_prices)
            
            if ob.get('yes_asks'):
                ask_prices = [int(p) for p in ob['yes_asks'].keys()]
                if ask_prices:
                    yes_best_ask = min(ask_prices)
            
            volume = ob.get('total_volume', 0)
            
            # Track changes
            if prev_state:
                prev_ob = prev_state.orderbook_state
                
                # Bid changes
                prev_bid = None
                if prev_ob.get('yes_bids'):
                    prev_bid_prices = [int(p) for p in prev_ob['yes_bids'].keys()]
                    if prev_bid_prices:
                        prev_bid = max(prev_bid_prices)
                
                if prev_bid != yes_best_bid and prev_bid and yes_best_bid:
                    bid_changes.append(yes_best_bid - prev_bid)
                
                # Volume changes
                prev_volume = prev_ob.get('total_volume', 0)
                vol_change = volume - prev_volume
                if abs(vol_change) > 100:
                    volume_changes.append(vol_change)
                    significant_events.append({
                        'time': timestamp,
                        'type': 'volume',
                        'change': vol_change,
                        'seq': state.sequence_number
                    })
            
            # Show key events
            if i == 0 or state.is_snapshot or i == len(states) - 1:
                print(f"\n[{timestamp.strftime('%H:%M:%S')}] Seq {state.sequence_number} ({'SNAPSHOT' if state.is_snapshot else 'DELTA'})")
                if yes_best_bid:
                    print(f"  Bid: {yes_best_bid}¢", end="")
                if yes_best_ask:
                    print(f"  Ask: {yes_best_ask}¢", end="")
                print(f"  Volume: {volume:,}")
            
            prev_state = state
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY:")
        print("-" * 70)
        
        if bid_changes:
            print(f"• Bid moved {len(bid_changes)} times (avg: {sum(bid_changes)/len(bid_changes):+.2f}¢)")
        else:
            print(f"• Bid price remained stable")
        
        if volume_changes:
            print(f"• Volume changed {len(volume_changes)} times (net: {sum(volume_changes):+,})")
        
        print(f"• Events per minute: {len(states) / ((states[-1].timestamp_ms - states[0].timestamp_ms) / 60000):.1f}")
        
        # Market state
        first_vol = states[0].orderbook_state.get('total_volume', 0)
        last_vol = states[-1].orderbook_state.get('total_volume', 0)
        
        print(f"\n🎯 Market Activity: {'High' if len(states) > 20 else 'Moderate' if len(states) > 10 else 'Low'}")
        print(f"📉 Volume Trend: {last_vol - first_vol:+,} ({'↑' if last_vol > first_vol else '↓' if last_vol < first_vol else '→'})")
        
        print("\n" + "=" * 70)
        
    finally:
        await loader.disconnect()
        await rl_db.close()


async def main():
    parser = argparse.ArgumentParser(description='Analyze orderbook history for Kalshi markets')
    parser.add_argument('market_ticker', nargs='?', help='Market ticker to analyze (e.g., KXROBOTMARS-35)')
    parser.add_argument('--session-id', type=int, help='Specific session ID to analyze')
    parser.add_argument('--list', action='store_true', help='List all markets with orderbook data')
    parser.add_argument('--sessions', action='store_true', help='List all orderbook sessions')
    parser.add_argument('--db-url', help='Database URL (uses DATABASE_URL env var if not provided)')
    
    args = parser.parse_args()
    
    # Get database URL
    import os
    db_url = args.db_url or os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:54322/postgres')
    
    if args.sessions:
        await list_sessions()
    elif args.list:
        await list_markets_with_data(db_url)
    elif args.market_ticker:
        await analyze_market_history(args.market_ticker.upper(), db_url, args.session_id)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())