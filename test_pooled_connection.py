#!/usr/bin/env python3
"""
Test script to verify pooled database connection using the exact same logic as production.
This will help isolate whether the connection issue is Render-specific or broader.
"""

import asyncio
import sys
import os
import logging

# Add backend to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from kalshiflow.database import Database

# Configure logging to match production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_pooled_connection():
    """Test the exact same connection logic that's failing in production"""
    
    # Use the exact pooled URL from production
    pooled_url = "postgresql://postgres:PPToqgqad8eb4drB@db.fnsbruyvocdefnhzjiyk.supabase.co:6543/postgres"
    
    print("=" * 80)
    print("TESTING POOLED DATABASE CONNECTION")
    print("=" * 80)
    print(f"Testing connection to: {pooled_url}")
    print("This mimics the exact same Database.initialize() call that's failing on Render")
    print()
    
    # Create database instance with explicit pooled URL
    database = Database(database_url=pooled_url)
    
    try:
        print("🔄 Calling database.initialize()...")
        print("   This creates the connection pool with the same settings as production:")
        print("   - min_size=2, max_size=10")
        print("   - statement_cache_size=0 (for pgbouncer compatibility)")
        print("   - command_timeout=30")
        print()
        
        # This calls the exact same code that's failing on Render
        await database.initialize()
        
        print("✅ SUCCESS: Database connection pool created successfully!")
        print("   The pooled connection (port 6543) is working from this environment")
        print()
        
        # Test a simple query to confirm it's fully functional
        print("🔄 Testing database query...")
        async with database.get_connection() as conn:
            version = await conn.fetchval('SELECT version()')
            print(f"✅ PostgreSQL version: {version}")
            
            # Test if we can see our tables
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            print(f"✅ Found {len(tables)} tables: {[t['table_name'] for t in tables]}")
        
        print()
        print("🔄 Cleaning up connection pool...")
        await database.close()
        print("✅ Connection pool closed successfully")
        
        print()
        print("=" * 80)
        print("CONCLUSION: Pooled connection works fine locally!")
        print("The issue is likely Render-specific networking or IPv4/IPv6 compatibility.")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print()
        print("Full error traceback:")
        import traceback
        traceback.print_exc()
        
        print()
        print("=" * 80)
        print("CONCLUSION: The pooled connection itself has issues!")
        print("This suggests the problem is not just Render-specific.")
        print("=" * 80)

if __name__ == "__main__":
    print("Starting pooled connection test...")
    asyncio.run(test_pooled_connection())