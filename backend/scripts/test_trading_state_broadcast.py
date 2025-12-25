#!/usr/bin/env python3
"""Test script to verify trading_state WebSocket broadcasts."""

import asyncio
import json
import time
from typing import Optional

import websockets
from websockets.client import WebSocketClientProtocol


async def listen_for_trading_state():
    """Connect to V3 WebSocket and listen for trading_state messages."""
    uri = "ws://localhost:8005/v3/ws"
    
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to V3 WebSocket")
            print("📊 Listening for trading_state messages...")
            print("-" * 60)
            
            message_count = 0
            trading_state_count = 0
            last_trading_state = None
            
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=60.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    # Track all message types
                    if data.get("type") == "trading_state":
                        trading_state_count += 1
                        trading_data = data.get("data", {})
                        
                        # Format currency values
                        balance = trading_data.get("balance", 0) / 100
                        portfolio_value = trading_data.get("portfolio_value", 0) / 100
                        
                        print(f"\n📊 TRADING STATE UPDATE #{trading_state_count}")
                        print(f"  💰 Balance: ${balance:,.2f}")
                        print(f"  📈 Portfolio: ${portfolio_value:,.2f}")
                        print(f"  📦 Positions: {trading_data.get('position_count', 0)}")
                        print(f"  📝 Orders: {trading_data.get('order_count', 0)}")
                        print(f"  🕐 Sync Time: {time.strftime('%H:%M:%S', time.localtime(trading_data.get('sync_timestamp', 0)))}")
                        print(f"  🔢 Version: {trading_data.get('version', 0)}")
                        
                        # Show changes if present
                        changes = trading_data.get("changes")
                        if changes:
                            print(f"\n  📈 CHANGES:")
                            if changes.get("balance_change"):
                                change = changes["balance_change"] / 100
                                print(f"    • Balance: ${change:+,.2f}")
                            if changes.get("portfolio_change"):
                                change = changes["portfolio_change"] / 100
                                print(f"    • Portfolio: ${change:+,.2f}")
                            if changes.get("position_count_change"):
                                print(f"    • Positions: {changes['position_count_change']:+d}")
                            if changes.get("order_count_change"):
                                print(f"    • Orders: {changes['order_count_change']:+d}")
                        
                        # Show positions if any
                        positions = trading_data.get("positions", [])
                        if positions:
                            print(f"\n  🎯 TOP POSITIONS:")
                            for pos in positions[:3]:
                                market_value = pos.get("market_value", 0) / 100
                                side = pos.get("side", "?").upper()
                                quantity = pos.get("quantity", 0)
                                ticker = pos.get("market_ticker", "UNKNOWN")
                                print(f"    • {ticker}: {side} x{quantity} = ${market_value:,.2f}")
                        
                        last_trading_state = trading_data
                        print("-" * 60)
                    
                    elif data.get("type") == "state_transition":
                        state = data.get("data", {}).get("to_state", "?")
                        print(f"  ⚡ State transition: {state}")
                    
                    elif data.get("type") == "trader_status":
                        # Just note we got status, don't print details
                        if message_count % 10 == 0:
                            print(f"  📡 Status update received (total messages: {message_count})")
                
                except asyncio.TimeoutError:
                    print("\n⏱️  No messages received for 60 seconds")
                    if last_trading_state:
                        print("📊 Last trading state:")
                        print(f"  • Balance: ${last_trading_state.get('balance', 0) / 100:,.2f}")
                        print(f"  • Portfolio: ${last_trading_state.get('portfolio_value', 0) / 100:,.2f}")
                    break
                
    except websockets.exceptions.ConnectionRefused:
        print("❌ Connection refused. Make sure V3 trader is running on port 8005")
    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")
        if trading_state_count > 0:
            print(f"📊 Received {trading_state_count} trading state updates")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("V3 TRADING STATE WEBSOCKET TEST")
    print("=" * 60)
    
    asyncio.run(listen_for_trading_state())