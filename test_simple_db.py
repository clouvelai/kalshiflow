#!/usr/bin/env python3
"""
Simple test to check if Railway can connect to Supabase database
"""
import os
import psycopg2

def test_railway_connection():
    """Test the current Railway DATABASE_URL"""
    db_url = os.getenv('DATABASE_URL')
    
    print(f"🔧 Testing Railway's DATABASE_URL:")
    print(f"URL: {db_url[:60]}...")
    
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        
        # Simple test query
        cursor.execute("SELECT 1 as test;")
        result = cursor.fetchone()
        
        print(f"✅ SUCCESS: {result}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    if test_railway_connection():
        print("🎉 Railway can connect to Supabase!")
    else:
        print("💥 Railway cannot connect to Supabase")