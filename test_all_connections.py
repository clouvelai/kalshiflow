#!/usr/bin/env python3
"""
Test all possible Supabase connection string formats from Railway
"""
import psycopg2

def test_connection(db_url, name):
    """Test a single database connection"""
    print(f"\n🔧 Testing {name}")
    print(f"URL: {db_url[:70]}...")
    
    try:
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test;")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        print(f"✅ {name} SUCCESS!")
        return True
        
    except Exception as e:
        print(f"❌ {name} FAILED: {str(e)}")
        return False

def main():
    print("🚀 Testing All Supabase Connection Formats from Railway")
    print("=" * 60)
    
    # Your project details
    project_ref = "fnsbruyvocdefnhzjiyk"
    password = "PPToqgqad8eb4drB"
    
    # Test all possible connection formats
    test_cases = [
        # Direct connections (IPv6 - likely to fail on Railway)
        (f"postgresql://postgres:{password}@db.{project_ref}.supabase.co:5432/postgres", "Direct Connection (port 5432)"),
        (f"postgresql://postgres:{password}@db.{project_ref}.supabase.co:6543/postgres", "Direct Connection (port 6543)"),
        
        # Pooled connections - different username formats
        (f"postgresql://postgres:{password}@aws-0-us-east-1.pooler.supabase.com:5432/postgres", "Pooled - postgres user"),
        (f"postgresql://postgres.{project_ref}:{password}@aws-0-us-east-1.pooler.supabase.com:5432/postgres", "Pooled - postgres.project user"),
        
        # Pooled with different regions
        (f"postgresql://postgres:{password}@aws-0-us-west-1.pooler.supabase.com:5432/postgres", "Pooled - US West"),
        (f"postgresql://postgres:{password}@aws-0-eu-west-1.pooler.supabase.com:5432/postgres", "Pooled - EU West"),
        
        # Session mode pooled connections
        (f"postgresql://postgres:{password}@{project_ref}.pooler.supabase.com:5432/postgres", "Session Pooler"),
        
        # Transaction mode with different ports
        (f"postgresql://postgres:{password}@aws-0-us-east-1.pooler.supabase.com:6543/postgres", "Transaction Pooler (6543)"),
        
        # Try with database name specified
        (f"postgresql://postgres:{password}@aws-0-us-east-1.pooler.supabase.com:5432/{project_ref}", "Pooled with project DB name"),
        
        # Alternative hostname formats
        (f"postgresql://postgres:{password}@db.{project_ref}.pooler.supabase.com:5432/postgres", "Alternative pooler format"),
    ]
    
    working_connections = []
    
    for db_url, name in test_cases:
        if test_connection(db_url, name):
            working_connections.append((name, db_url))
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY: {len(working_connections)}/{len(test_cases)} connections work")
    
    if working_connections:
        print("\n🎉 WORKING CONNECTIONS:")
        for i, (name, url) in enumerate(working_connections, 1):
            print(f"{i}. {name}")
            print(f"   {url}")
    else:
        print("\n💥 No connections worked from Railway environment")

if __name__ == "__main__":
    main()