#!/usr/bin/env python
"""
Test script to verify V3 trader order group data flow.
"""
import asyncio
import json
import websockets
from datetime import datetime

async def test_v3_order_group():
    """Connect to V3 WebSocket and monitor order group data."""
    uri = "ws://localhost:8005/v3/ws"
    
    print(f"\n{'='*60}")
    print("V3 TRADER ORDER GROUP TEST")
    print(f"{'='*60}")
    print(f"Connecting to: {uri}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to V3 Trader WebSocket")
            print("\nListening for messages...\n")
            
            message_count = 0
            trading_state_count = 0
            has_order_group = False
            
            while message_count < 100:  # Listen for up to 100 messages
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    # Check for trading_state messages with order group data
                    if data.get("type") == "trading_state":
                        trading_state_count += 1
                        state_data = data.get("data", {})
                        
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Trading State #{trading_state_count}")
                        print(f"  Version: {state_data.get('version', 'N/A')}")
                        print(f"  Balance: ${state_data.get('balance', 0) / 100:.2f}")
                        print(f"  Portfolio: ${state_data.get('portfolio_value', 0) / 100:.2f}")
                        print(f"  Positions: {state_data.get('position_count', 0)}")
                        print(f"  Orders: {state_data.get('order_count', 0)}")
                        
                        # Check for order group data
                        order_group = state_data.get("order_group")
                        if order_group:
                            has_order_group = True
                            print(f"\n  📊 ORDER GROUP DATA DETECTED:")
                            print(f"     ID: {order_group.get('id', 'N/A')}")
                            print(f"     Status: {order_group.get('status', 'N/A')}")
                            print(f"     Contract Limit: {order_group.get('contracts_limit', 'N/A')}")
                            
                            usage = order_group.get('usage', {})
                            if usage:
                                print(f"     Usage:")
                                print(f"       - Contracts: {usage.get('contracts', 0)}/{order_group.get('contracts_limit', 0)}")
                                print(f"       - Position: ${usage.get('position', 0) / 100:.2f}")
                                print(f"       - Open Orders: {usage.get('orders', 0)}")
                                
                                # Calculate usage percentage
                                if order_group.get('contracts_limit'):
                                    usage_pct = (usage.get('contracts', 0) / order_group['contracts_limit']) * 100
                                    print(f"       - Usage %: {usage_pct:.1f}%")
                        else:
                            print(f"  ⚠️  No order group data in this message")
                    
                    # Also track other message types
                    elif data.get("type") in ["system_activity", "state_transition"]:
                        activity_data = data.get("data", {})
                        message_text = activity_data.get("message", "")
                        if "order_group" in message_text.lower():
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Order Group Activity:")
                            print(f"  {message_text}")
                
                except asyncio.TimeoutError:
                    print("\n⏱️  Timeout waiting for messages (30s)")
                    break
                except Exception as e:
                    print(f"\n❌ Error processing message: {e}")
                    continue
            
            # Summary
            print(f"\n{'='*60}")
            print("TEST SUMMARY")
            print(f"{'='*60}")
            print(f"Total messages received: {message_count}")
            print(f"Trading state messages: {trading_state_count}")
            print(f"Order group data found: {'✅ YES' if has_order_group else '❌ NO'}")
            
            if has_order_group:
                print("\n✅ Order group data is being properly transmitted!")
            else:
                print("\n⚠️  No order group data detected in trading state messages.")
                print("This is expected if:")
                print("  1. Using demo API (order groups not supported)")
                print("  2. Order group not yet created")
                print("  3. Trading client not configured")
            
    except websockets.exceptions.ConnectionRefused:
        print("❌ Could not connect to V3 Trader WebSocket")
        print("   Make sure the V3 trader is running on port 8005")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("\n🔍 Starting V3 Trader Order Group Test...")
    asyncio.run(test_v3_order_group())