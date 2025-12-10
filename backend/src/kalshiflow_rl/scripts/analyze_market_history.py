#!/usr/bin/env python3
"""
Analyze the complete orderbook history for a specific market.

Usage:
    python analyze_market_history.py <market_ticker>
    python analyze_market_history.py KXROBOTMARS-35
    python analyze_market_history.py --list  # List markets with data
"""

import asyncio
import asyncpg
import argparse
import sys
from datetime import datetime
from typing import Optional

from ..environments.historical_data_loader import (
    HistoricalDataLoader, 
    DataLoadConfig
)


async def list_markets_with_data(db_url: str):
    """List all markets that have orderbook data."""
    conn = await asyncpg.connect(db_url)
    try:
        markets = await conn.fetch("""
            WITH market_stats AS (
                SELECT 
                    market_ticker,
                    COUNT(*) as total_events,
                    MIN(timestamp_ms) as first_seen,
                    MAX(timestamp_ms) as last_seen
                FROM (
                    SELECT market_ticker, timestamp_ms FROM rl_orderbook_snapshots
                    UNION ALL
                    SELECT market_ticker, timestamp_ms FROM rl_orderbook_deltas
                ) combined
                GROUP BY market_ticker
            )
            SELECT * FROM market_stats
            ORDER BY total_events DESC
        """)
        
        if not markets:
            print("No markets found with orderbook data")
            return
        
        print("=" * 70)
        print("MARKETS WITH ORDERBOOK DATA")
        print("=" * 70)
        print(f"{'Market Ticker':<30} {'Events':<10} {'First Seen':<20} {'Last Seen':<20}")
        print("-" * 70)
        
        for row in markets:
            first = datetime.fromtimestamp(row['first_seen']/1000).strftime('%Y-%m-%d %H:%M:%S')
            last = datetime.fromtimestamp(row['last_seen']/1000).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{row['market_ticker']:<30} {row['total_events']:<10} {first:<20} {last:<20}")
        
        print(f"\nTotal: {len(markets)} markets")
        
    finally:
        await conn.close()


async def analyze_market_history(market_ticker: str, db_url: str):
    """Analyze complete orderbook history for a specific market."""
    
    print("=" * 70)
    print(f"ORDERBOOK HISTORY ANALYSIS: {market_ticker}")
    print("=" * 70)
    
    conn = await asyncpg.connect(db_url)
    try:
        # Get time range and event counts
        stats = await conn.fetchrow("""
            WITH all_events AS (
                SELECT timestamp_ms, sequence_number, 'snapshot' as event_type
                FROM rl_orderbook_snapshots 
                WHERE market_ticker = $1
                UNION ALL
                SELECT timestamp_ms, sequence_number, 'delta' as event_type
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
                MAX(sequence_number) as max_seq
            FROM all_events
        """, market_ticker)
        
        if not stats or stats['total_events'] == 0:
            print(f"❌ No orderbook data found for {market_ticker}")
            return
        
        print(f"\n📊 Data Overview:")
        print(f"  Time Range: {datetime.fromtimestamp(stats['min_time']/1000).strftime('%Y-%m-%d %H:%M:%S')} - {datetime.fromtimestamp(stats['max_time']/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {(stats['max_time'] - stats['min_time']) / 1000:.1f} seconds")
        print(f"  Total Events: {stats['total_events']} ({stats['snapshot_count']} snapshots, {stats['delta_count']} deltas)")
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


async def main():
    parser = argparse.ArgumentParser(description='Analyze orderbook history for Kalshi markets')
    parser.add_argument('market_ticker', nargs='?', help='Market ticker to analyze (e.g., KXROBOTMARS-35)')
    parser.add_argument('--list', action='store_true', help='List all markets with orderbook data')
    parser.add_argument('--db-url', help='Database URL (uses DATABASE_URL env var if not provided)')
    
    args = parser.parse_args()
    
    # Get database URL
    import os
    db_url = args.db_url or os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:54322/postgres')
    
    if args.list:
        await list_markets_with_data(db_url)
    elif args.market_ticker:
        await analyze_market_history(args.market_ticker.upper(), db_url)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())