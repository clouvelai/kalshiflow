#!/usr/bin/env python3
"""Debug Railway -> Supabase connection"""
import os
import asyncio
import asyncpg
import socket
from urllib.parse import urlparse

async def test_supabase_connection():
    """Test various connection methods to Supabase"""
    
    # Test different connection URLs
    urls = [
        "postgresql://postgres:PPToqgqad8eb4drB@db.fnsbruyvocdefnhzjiyk.supabase.co:5432/postgres",
        "postgresql://postgres.fnsbruyvocdefnhzjiyk:PPToqgqad8eb4drB@aws-0-us-east-1.pooler.supabase.com:5432/postgres",
        "postgresql://postgres.fnsbruyvocdefnhzjiyk:PPToqgqad8eb4drB@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    ]
    
    for i, url in enumerate(urls, 1):
        print(f"\n=== Test {i}: {url.split('@')[1]} ===")
        
        try:
            # Parse URL to test basic connectivity
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port
            
            print(f"Host: {host}, Port: {port}")
            
            # Test DNS resolution
            try:
                ip = socket.gethostbyname(host)
                print(f"✅ DNS Resolution: {host} -> {ip}")
            except Exception as dns_error:
                print(f"❌ DNS Resolution failed: {dns_error}")
                continue
                
            # Test TCP connectivity
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result == 0:
                    print(f"✅ TCP Connection: {host}:{port} reachable")
                else:
                    print(f"❌ TCP Connection: {host}:{port} unreachable (error {result})")
                    continue
            except Exception as tcp_error:
                print(f"❌ TCP Connection failed: {tcp_error}")
                continue
                
            # Test PostgreSQL connection
            try:
                conn = await asyncpg.connect(url, timeout=10)
                result = await conn.fetchval("SELECT 1")
                await conn.close()
                print(f"✅ PostgreSQL Connection: Success (returned {result})")
                return url  # Return successful URL
            except Exception as pg_error:
                print(f"❌ PostgreSQL Connection failed: {pg_error}")
                
        except Exception as e:
            print(f"❌ General error: {e}")
    
    print("\n❌ All connection attempts failed")
    return None

if __name__ == "__main__":
    print("🔍 Testing Railway -> Supabase connectivity...")
    print(f"Environment: {os.getenv('RAILWAY_ENVIRONMENT', 'local')}")
    
    try:
        working_url = asyncio.run(test_supabase_connection())
        if working_url:
            print(f"\n🎉 SUCCESS! Use this URL: {working_url}")
        else:
            print("\n💥 All connection methods failed")
    except Exception as e:
        print(f"\n💥 Script failed: {e}")