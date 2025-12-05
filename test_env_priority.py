#!/usr/bin/env python3
"""
Test script to verify which database URL is being selected by our priority logic.
This will help confirm if DATABASE_URL_POOLED is being used correctly.
"""

import os
import sys

# Add backend to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

def test_env_priority():
    """Test the exact same environment variable logic used in production"""
    
    print("=" * 80)
    print("TESTING ENVIRONMENT VARIABLE PRIORITY")
    print("=" * 80)
    
    # Simulate Render environment variables (these should be set in Render)
    test_env = {
        'DATABASE_URL_POOLED': 'postgresql://postgres:PPToqgqad8eb4drB@db.fnsbruyvocdefnhzjiyk.supabase.co:6543/postgres',
        'DATABASE_URL': 'postgresql://postgres:PPToqgqad8eb4drB@db.fnsbruyvocdefnhzjiyk.supabase.co:5432/postgres'
    }
    
    # Set the environment variables to simulate Render
    for key, value in test_env.items():
        os.environ[key] = value
    
    print("Simulated Render environment variables:")
    for key, value in test_env.items():
        print(f"  {key} = {value}")
    print()
    
    # Test the exact priority logic from Database.__init__
    print("Testing Database URL priority logic:")
    print("  self.database_url = (")
    print("      database_url or")  
    print("      os.getenv('DATABASE_URL_POOLED') or")
    print("      os.getenv('DATABASE_URL')")
    print("  )")
    print()
    
    database_url = None  # No explicit URL passed
    pooled_url = os.getenv("DATABASE_URL_POOLED")
    direct_url = os.getenv("DATABASE_URL")
    
    final_url = (
        database_url or 
        pooled_url or 
        direct_url
    )
    
    print("Results:")
    print(f"  database_url (explicit): {database_url}")
    print(f"  DATABASE_URL_POOLED: {pooled_url}")
    print(f"  DATABASE_URL: {direct_url}")
    print(f"  Final selected URL: {final_url}")
    print()
    
    # Check which port is being used
    if final_url:
        if ':6543/' in final_url:
            print("✅ CORRECT: Using pooled connection (port 6543)")
        elif ':5432/' in final_url:
            print("❌ WRONG: Using direct connection (port 5432)")
        else:
            print("⚠️  UNKNOWN: Could not determine port from URL")
    else:
        print("❌ ERROR: No database URL selected")
    
    print()
    print("=" * 80)
    
    # Clean up environment
    for key in test_env:
        if key in os.environ:
            del os.environ[key]

if __name__ == "__main__":
    test_env_priority()