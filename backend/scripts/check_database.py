#!/usr/bin/env python3
"""
Simple script to check RL orderbook database contents.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kalshiflow_rl.database import rl_db

async def check_database():
    """Check database contents."""
    await rl_db.initialize()
    
    try:
        async with rl_db.get_connection() as conn:
            # Count total records
            total_count = await conn.fetchval("SELECT COUNT(*) FROM rl_orderbook_data")
            print(f"📊 Total records: {total_count}")
            
            if total_count > 0:
                # Group by message type and market
                stats = await conn.fetch("""
                    SELECT 
                        message_type,
                        market_ticker,
                        COUNT(*) as count,
                        MIN(timestamp) as first_record,
                        MAX(timestamp) as last_record
                    FROM rl_orderbook_data 
                    GROUP BY message_type, market_ticker 
                    ORDER BY market_ticker, message_type
                """)
                
                print("\n📈 Records by market and type:")
                for row in stats:
                    first_time = row['first_record'].strftime('%H:%M:%S') if row['first_record'] else 'N/A'
                    last_time = row['last_record'].strftime('%H:%M:%S') if row['last_record'] else 'N/A'
                    print(f"  {row['market_ticker']:<20} {row['message_type']:<10} {row['count']:>5} records  ({first_time} - {last_time})")
                
                # Show sample snapshot data
                sample = await conn.fetchrow("""
                    SELECT market_ticker, sequence_number, snapshot_data::text
                    FROM rl_orderbook_data 
                    WHERE message_type = 'snapshot'
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                
                if sample:
                    print(f"\n📋 Sample snapshot for {sample['market_ticker']} (seq: {sample['sequence_number']}):")
                    # Truncate long JSON for readability
                    snapshot_str = sample['snapshot_data'][:200] + "..." if len(sample['snapshot_data']) > 200 else sample['snapshot_data']
                    print(f"  {snapshot_str}")
                
            else:
                print("❌ No records found in database")
                
    finally:
        await rl_db.close()

if __name__ == "__main__":
    asyncio.run(check_database())