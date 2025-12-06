#!/usr/bin/env python3
"""
Test database connectivity from Railway environment
"""
import os
import psycopg2
from urllib.parse import urlparse

def test_connection(db_url, connection_name):
    """Test a database connection"""
    print(f"\n🔧 Testing {connection_name}:")
    print(f"URL: {db_url[:50]}...")
    
    try:
        # Parse the URL to get components
        parsed = urlparse(db_url)
        print(f"Host: {parsed.hostname}")
        print(f"Port: {parsed.port}")
        
        # Test connection
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT version();")
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        print(f"✅ {connection_name} SUCCESS")
        print(f"PostgreSQL version: {result[0][:50]}...")
        return True
        
    except Exception as e:
        print(f"❌ {connection_name} FAILED")
        print(f"Error: {str(e)}")
        return False

def main():
    print("🚀 Railway Database Connection Test")
    print("=" * 50)
    
    # Test both connection strings
    direct_url = os.getenv('DATABASE_URL')
    pooled_url = os.getenv('DATABASE_URL_POOLED')
    
    # Test the CORRECT pooled format for Supabase
    # Supabase pooled connections use [project-ref].pooler.supabase.com format
    project_ref = "fnsbruyvocdefnhzjiyk" 
    password = "PPToqgqad8eb4drB"
    correct_pooled = f"postgresql://postgres:{password}@{project_ref}.pooler.supabase.com:5432/postgres"
    
    results = []
    
    if direct_url:
        results.append(test_connection(direct_url, "Direct Connection"))
    
    if pooled_url:
        results.append(test_connection(pooled_url, "Current Pooled (WRONG)"))
    
    results.append(test_connection(correct_pooled, "Correct Pooled Connection"))
    
    print("\n" + "=" * 50)
    print("📊 SUMMARY:")
    working_count = sum(results)
    total_count = len(results)
    print(f"Working connections: {working_count}/{total_count}")
    
    if working_count > 0:
        print("✅ At least one connection works!")
    else:
        print("❌ No connections working - check network/IPv6 support")

if __name__ == "__main__":
    main()