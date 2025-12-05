#!/usr/bin/env python3
"""
Test script to verify the new Supavisor URL format works locally.
This tests the newer aws-0-us-west-1.pooler.supabase.com format.
"""

import asyncio
import sys
import os

# Add backend to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from kalshiflow.database import Database

async def test_supavisor_url():
    """Test the new Supavisor URL format that should work better with Render"""
    
    # New Supavisor URL format from search results
    supavisor_url = "postgresql://postgres.fnsbruyvocdefnhzjiyk:PPToqgqad8eb4drB@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
    
    print("=" * 80)
    print("TESTING NEW SUPAVISOR URL FORMAT")
    print("=" * 80)
    print(f"Testing connection to: {supavisor_url}")
    print("This is the newer format that should be more Render-compatible")
    print()
    
    # Create database instance with new URL format
    database = Database(database_url=supavisor_url)
    
    try:
        print("🔄 Calling database.initialize() with new Supavisor URL...")
        print("   Format: postgresql://postgres.PROJECT_ID:PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres")
        print("   This should be more compatible with IPv4-only environments like Render")
        print()
        
        # Test the connection
        await database.initialize()
        
        print("✅ SUCCESS: New Supavisor URL works!")
        print("   This format should work better on Render")
        print()
        
        # Test database operations
        print("🔄 Testing database operations...")
        async with database.get_connection() as conn:
            version = await conn.fetchval('SELECT version()')
            print(f"✅ PostgreSQL version: {version}")
            
            # Test table access
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """)
            print(f"✅ Found {len(tables)} tables: {[t['table_name'] for t in tables]}")
            
            # Test a simple query
            count = await conn.fetchval("SELECT COUNT(*) FROM trades")
            print(f"✅ Trades table has {count} records")
        
        print()
        print("🔄 Cleaning up...")
        await database.close()
        print("✅ Connection closed successfully")
        
        print()
        print("=" * 80)
        print("✅ CONCLUSION: New Supavisor URL works locally!")
        print("This should be deployed to Render as the new DATABASE_URL")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print()
        print("Full error traceback:")
        import traceback
        traceback.print_exc()
        
        print()
        print("=" * 80)
        print("❌ CONCLUSION: New Supavisor URL has issues!")
        print("We may need to try a different approach or URL format")
        print("=" * 80)
        
        return False

if __name__ == "__main__":
    print("Testing new Supavisor URL format...")
    success = asyncio.run(test_supavisor_url())
    exit(0 if success else 1)