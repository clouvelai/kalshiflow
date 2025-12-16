#!/usr/bin/env python
"""Query the RL orderbook tables to show example data."""

import asyncio
import asyncpg
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment
load_dotenv('.env.local')
DATABASE_URL = os.getenv('DATABASE_URL')


async def show_orderbook_data():
    """Display orderbook data from the database."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        
        # First check if tables exist
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'orderbook_snapshots'
            )
        """)
        
        if not table_exists:
            print("\n❌ RL orderbook tables don't exist yet.")
            print("Run the E2E test first: uv run pytest tests/test_rl_backend_e2e_regression.py")
            await conn.close()
            return
        
        # Get the most recent orderbook snapshot
        print('\n=== MOST RECENT ORDERBOOK SNAPSHOT ===\n')
        snapshot = await conn.fetchrow('''
            SELECT * FROM orderbook_snapshots 
            WHERE market_ticker = 'KXCABOUT-29'
            ORDER BY timestamp DESC 
            LIMIT 1
        ''')
        
        if snapshot:
            print(f'Market: {snapshot["market_ticker"]}')
            print(f'Timestamp: {snapshot["timestamp"]}')
            print(f'Sequence: {snapshot["sequence_number"]}')
            
            # Parse and display YES orderbook
            print(f'\n📈 YES Side Orderbook:')
            yes_book = json.loads(snapshot['yes_book']) if snapshot['yes_book'] else {}
            if yes_book.get('bids'):
                print('  Bids (Buy Orders):')
                for i, bid in enumerate(yes_book['bids'][:5], 1):
                    print(f'    #{i}: {bid[0]}¢ x {bid[1]} contracts')
            else:
                print('  No bids')
                
            if yes_book.get('asks'):
                print('  Asks (Sell Orders):')
                for i, ask in enumerate(yes_book['asks'][:5], 1):
                    print(f'    #{i}: {ask[0]}¢ x {ask[1]} contracts')
            else:
                print('  No asks')
            
            # Parse and display NO orderbook
            print(f'\n📉 NO Side Orderbook:')
            no_book = json.loads(snapshot['no_book']) if snapshot['no_book'] else {}
            if no_book.get('bids'):
                print('  Bids (Buy Orders):')
                for i, bid in enumerate(no_book['bids'][:5], 1):
                    print(f'    #{i}: {bid[0]}¢ x {bid[1]} contracts')
            else:
                print('  No bids')
                
            if no_book.get('asks'):
                print('  Asks (Sell Orders):')
                for i, ask in enumerate(no_book['asks'][:5], 1):
                    print(f'    #{i}: {ask[0]}¢ x {ask[1]} contracts')
            else:
                print('  No asks')
                
            # Calculate spreads
            yes_spread = "N/A"
            no_spread = "N/A"
            if yes_book.get('bids') and yes_book.get('asks'):
                yes_spread = yes_book['asks'][0][0] - yes_book['bids'][0][0]
            if no_book.get('bids') and no_book.get('asks'):
                no_spread = no_book['asks'][0][0] - no_book['bids'][0][0]
            
            print(f'\n📊 Spreads:')
            print(f'  YES spread: {yes_spread}¢')
            print(f'  NO spread: {no_spread}¢')
        else:
            print('No snapshots found for KXCABOUT-29')
        
        # Get summary stats
        print('\n=== DATABASE SUMMARY ===\n')
        snapshot_count = await conn.fetchval(
            "SELECT COUNT(*) FROM orderbook_snapshots WHERE market_ticker = 'KXCABOUT-29'"
        )
        delta_count = await conn.fetchval(
            "SELECT COUNT(*) FROM orderbook_deltas WHERE market_ticker = 'KXCABOUT-29'"
        )
        
        print(f'📚 Total Snapshots: {snapshot_count}')
        print(f'📝 Total Deltas: {delta_count}')
        
        # Show a sample delta
        if delta_count > 0:
            print('\n=== SAMPLE DELTA UPDATE ===\n')
            delta = await conn.fetchrow('''
                SELECT * FROM orderbook_deltas 
                WHERE market_ticker = 'KXCABOUT-29'
                ORDER BY timestamp DESC 
                LIMIT 1
            ''')
            if delta:
                print(f'Timestamp: {delta["timestamp"]}')
                print(f'Sequence: {delta["sequence_number"]}')
                print(f'Delta Type: {delta["delta_type"]}')
                delta_data = json.loads(delta['delta_data']) if delta['delta_data'] else {}
                print(f'Delta Content Preview:')
                print(json.dumps(delta_data, indent=2)[:500])
        
        await conn.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(show_orderbook_data())