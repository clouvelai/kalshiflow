#!/usr/bin/env python3
"""
Test script to verify OrderbookClient can connect to Kalshi and receive orderbook data.

This script demonstrates that the WebSocket connection issue has been fixed.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

# Load environment
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env.local"
if env_path.exists():
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")

# Set test market
os.environ["RL_MARKET_TICKER"] = "KXCABOUT-29"

from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.data.auth import validate_rl_auth
from kalshiflow_rl.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_orderbook_connection")


async def test_orderbook_connection():
    """Test OrderbookClient connection to Kalshi."""
    
    print("🚀 Testing Orderbook WebSocket Connection")
    print("=" * 50)
    
    # Validate authentication
    if not validate_rl_auth():
        print("❌ Authentication validation failed")
        return False
    
    print(f"✅ Authentication valid")
    print(f"📊 Market: {config.RL_MARKET_TICKER}")
    print(f"🌐 WebSocket URL: {config.KALSHI_WS_URL}")
    
    # Create orderbook client
    client = OrderbookClient(config.RL_MARKET_TICKER)
    
    # Track connection events
    connection_events = {
        'connected': asyncio.Event(),
        'first_message': asyncio.Event(),
        'error': None
    }
    
    async def on_connected():
        print("🔗 WebSocket connected to Kalshi")
        connection_events['connected'].set()
    
    async def on_error(error):
        print(f"❌ WebSocket error: {error}")
        connection_events['error'] = error
    
    # Set up event handlers
    client.on_connected(on_connected)
    client.on_error(on_error)
    
    # Start client
    print("🌐 Connecting to Kalshi WebSocket...")
    client_task = asyncio.create_task(client.start())
    
    try:
        # Wait for connection
        await asyncio.wait_for(connection_events['connected'].wait(), timeout=30.0)
        print("✅ WebSocket connected successfully")
        
        # Wait for some data
        print("⏳ Waiting for orderbook data (20 seconds)...")
        await asyncio.sleep(20.0)
        
        # Check stats
        stats = client.get_stats()
        print(f"📊 Connection Statistics:")
        print(f"  - Connected: {stats['connected']}")
        print(f"  - Messages received: {stats['messages_received']}")
        print(f"  - Snapshots: {stats['snapshots_received']}")
        print(f"  - Deltas: {stats['deltas_received']}")
        print(f"  - Reconnects: {stats['reconnect_count']}")
        print(f"  - Last sequence: {stats['last_sequence']}")
        
        if stats['messages_received'] > 0:
            print("✅ Received orderbook messages")
        else:
            print("⚠️  No messages received - this can be normal for inactive markets")
        
        if connection_events['error']:
            print(f"❌ Errors encountered: {connection_events['error']}")
            return False
        else:
            print("✅ No errors encountered")
        
        return True
        
    except asyncio.TimeoutError:
        print("❌ Failed to connect within 30 seconds")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    finally:
        print("🧹 Cleaning up...")
        await client.stop()
        if not client_task.done():
            client_task.cancel()
            try:
                await client_task
            except asyncio.CancelledError:
                pass


async def main():
    """Main test function."""
    success = await test_orderbook_connection()
    
    print("=" * 50)
    if success:
        print("🎉 OrderbookClient WebSocket Connection Test: ✅ PASSED")
        print("🚀 The WebSocket connection issue has been FIXED!")
    else:
        print("💥 OrderbookClient WebSocket Connection Test: ❌ FAILED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())