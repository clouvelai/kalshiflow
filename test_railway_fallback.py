#!/usr/bin/env python3
"""
Test the Railway DATABASE_URL_FALLBACK connection string locally
"""
import psycopg2

def test_fallback_connection():
    """Test the Railway fallback connection string"""
    # This is what I can see in the Railway screenshot
    fallback_url = "postgresql://postgres:PPToqgqad8eb4drB@db.fnsbruyvocdefnhzjiyk.supabase.co:5432/postgres"
    
    print(f"🔧 Testing Railway DATABASE_URL_FALLBACK:")
    print(f"URL: {fallback_url[:70]}...")
    
    try:
        conn = psycopg2.connect(fallback_url)
        cursor = conn.cursor()
        
        # Simple test query
        cursor.execute("SELECT 1 as test, version();")
        result = cursor.fetchone()
        
        print(f"✅ SUCCESS!")
        print(f"Test result: {result[0]}")
        print(f"PostgreSQL version: {result[1][:50]}...")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Railway DATABASE_URL_FALLBACK Connection")
    print("=" * 60)
    
    if test_fallback_connection():
        print("\n🎉 Railway fallback connection should work!")
        print("Your deployment should succeed now.")
    else:
        print("\n💥 Fallback connection failed locally")
        print("There may be an issue with the connection string.")