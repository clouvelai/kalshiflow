#!/usr/bin/env python
"""Debug script to see raw orderbook messages from Kalshi across multiple markets."""

import asyncio
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os
import websockets
from websockets.exceptions import ConnectionClosed

# Load environment
load_dotenv('.env.local')

# Now we can import (after env is loaded)
import sys
sys.path.insert(0, '/Users/samuelclark/Desktop/kalshiflow/backend/src')

from kalshiflow_rl.data.auth import get_rl_auth
from kalshiflow_rl.config import config


async def debug_orderbook():
    """Connect and show raw messages from Kalshi for multiple markets."""
    print("🚀 Connecting to Kalshi orderbook WebSocket...")
    
    # Configure multiple markets to test
    # Set environment variable to test the multi-market RL configuration
    import os
    test_markets = [
        "KXCABOUT-29",              # Next cabinet member out
        "KXEPLGAME-25DEC08WOLMUN",  # Premier League game
        "KXFEDDECISION-25DEC"       # Fed meeting decision
    ]
    
    # Set RL_MARKET_TICKERS for testing
    os.environ["RL_MARKET_TICKERS"] = ",".join(test_markets)
    
    print(f"🎯 Testing multi-market configuration: {', '.join(test_markets)}")
    
    # Reload config to pick up new environment
    import importlib
    import kalshiflow_rl.config
    importlib.reload(kalshiflow_rl.config)
    from kalshiflow_rl.config import config
    
    print(f"✅ Config loaded - monitoring {len(config.RL_MARKET_TICKERS)} markets: {config.RL_MARKET_TICKERS}")
    
    # Get auth headers
    auth = get_rl_auth()
    headers = auth.create_websocket_headers()
    
    # Connect to WebSocket
    ws_url = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    print(f"📡 Connecting to: {ws_url}")
    
    async with websockets.connect(
        ws_url,
        additional_headers=headers,
        ping_interval=30
    ) as websocket:
        print("✅ Connected!")
        
        # Subscribe to orderbook for the configured markets
        channels = [f"orderbook_delta.{ticker}" for ticker in config.RL_MARKET_TICKERS]
        subscribe_msg = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": channels
            }
        }
        
        await websocket.send(json.dumps(subscribe_msg))
        print(f"📨 Sent subscription for {len(channels)} channels:")
        for i, channel in enumerate(channels, 1):
            print(f"   {i}. {channel}")
        
        # Listen for messages
        print("\n⏳ Listening for messages (60 seconds)...")
        print("-" * 80)
        
        timeout_time = time.time() + 60
        message_count = 0
        snapshot_count = 0
        delta_count = 0
        
        # Track messages per market
        market_message_counts = {ticker: 0 for ticker in config.RL_MARKET_TICKERS}
        market_snapshot_counts = {ticker: 0 for ticker in config.RL_MARKET_TICKERS}
        market_delta_counts = {ticker: 0 for ticker in config.RL_MARKET_TICKERS}
        
        def extract_market_from_channel(channel):
            """Extract market ticker from channel string."""
            if "orderbook_delta." in channel:
                return channel.split("orderbook_delta.", 1)[1]
            return "unknown"
        
        while time.time() < timeout_time:
            try:
                # Wait for message with short timeout
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                message_count += 1
                
                # Parse and display
                msg_data = json.loads(message)
                channel = msg_data.get('channel', '')
                market_ticker = extract_market_from_channel(channel)
                
                print(f"\n📬 Message #{message_count} at {datetime.utcnow().isoformat()}:")
                print(f"Channel: {channel}")
                print(f"Market: {market_ticker}")
                print(f"Type: {msg_data.get('type', 'unknown')}")
                
                # Update per-market counters
                if market_ticker in market_message_counts:
                    market_message_counts[market_ticker] += 1
                
                if msg_data.get('type') == 'snapshot':
                    print("📸 SNAPSHOT received!")
                    snapshot_count += 1
                    if market_ticker in market_snapshot_counts:
                        market_snapshot_counts[market_ticker] += 1
                    
                    data = msg_data.get('data', {})
                    print(f"  Sequence: {data.get('seq')}")
                    
                    # Show YES book
                    yes_book = data.get('yes', {})
                    yes_bids = yes_book.get('b', {})
                    yes_asks = yes_book.get('a', {})
                    print(f"  YES: {len(yes_bids)} bids, {len(yes_asks)} asks")
                    
                    # Show NO book
                    no_book = data.get('no', {})
                    no_bids = no_book.get('b', {})
                    no_asks = no_book.get('a', {})
                    print(f"  NO: {len(no_bids)} bids, {len(no_asks)} asks")
                        
                elif msg_data.get('type') == 'delta':
                    print("📝 DELTA received!")
                    delta_count += 1
                    if market_ticker in market_delta_counts:
                        market_delta_counts[market_ticker] += 1
                    
                    data = msg_data.get('data', {})
                    print(f"  Sequence: {data.get('seq')}")
                    print(f"  Price: {data.get('price')}¢")
                    print(f"  Side: {data.get('side')}")
                    print(f"  Old size: {data.get('old_size')}")
                    print(f"  New size: {data.get('new_size', data.get('size'))}")
                    
                elif msg_data.get('msg') == 'ack':
                    print("✅ SUBSCRIPTION CONFIRMED")
                else:
                    # Show first part of message for other types
                    print(json.dumps(msg_data, indent=2)[:300] + "...")
                
            except asyncio.TimeoutError:
                # No message received in 1 second, continue
                continue
            except Exception as e:
                print(f"❌ Error: {e}")
                break
        
        print("-" * 80)
        print(f"\n📊 OVERALL Summary: Received {message_count} messages in 60 seconds")
        print(f"   📸 Total Snapshots: {snapshot_count}")
        print(f"   📝 Total Deltas: {delta_count}")
        
        print("\n📊 PER-MARKET Summary:")
        for market_ticker in config.RL_MARKET_TICKERS:
            msg_count = market_message_counts.get(market_ticker, 0)
            snap_count = market_snapshot_counts.get(market_ticker, 0)
            delta_count_market = market_delta_counts.get(market_ticker, 0)
            
            if msg_count > 0:
                print(f"   ✅ {market_ticker}: {msg_count} messages ({snap_count} snapshots, {delta_count_market} deltas)")
            else:
                print(f"   ⚠️  {market_ticker}: No messages (potentially inactive)")
        
        active_markets = sum(1 for count in market_message_counts.values() if count > 0)
        print(f"\n📈 Active markets: {active_markets}/{len(config.RL_MARKET_TICKERS)}")
        
        if message_count == 0:
            print("\n⚠️  No messages received for any market")
            print("💡 This could mean:")
            print("   - Markets are currently inactive (no trading)")
            print("   - Wrong market tickers")
            print("   - Network/authentication issues")
        else:
            print(f"\n✅ Multi-market orderbook ingestion working! Processed data from {active_markets} markets")

if __name__ == "__main__":
    asyncio.run(debug_orderbook())