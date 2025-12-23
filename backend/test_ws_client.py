#!/usr/bin/env python3
import asyncio
import json
import websockets
import sys
from datetime import datetime

async def test_ws():
    uri = "ws://localhost:8003/rl/ws"
    print(f"🔌 Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected! Listening for messages...\n")
            
            msg_count = 0
            start_time = datetime.now()
            
            while True:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    msg = json.loads(message)
                    msg_count += 1
                    
                    # Filter for trader-related messages
                    if msg['type'] in ['trader_state', 'state_transition', 'trader_status']:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        print(f"[{timestamp}] {msg['type']}:")
                        
                        if 'data' in msg:
                            data = msg['data']
                            if msg['type'] == 'trader_state':
                                print(f"  State: {data.get('state', 'N/A')}")
                                print(f"  Cash: ${data.get('cash_balance', 0):.2f}")
                                print(f"  Portfolio: ${data.get('portfolio_value', 0):.2f}")
                                print(f"  Total: ${data.get('total_value', 0):.2f}")
                                print(f"  Positions: {data.get('positions_count', 0)}")
                            elif msg['type'] == 'state_transition':
                                print(f"  {data.get('from_state', '?')} → {data.get('to_state', '?')}")
                                print(f"  Reason: {data.get('reason', 'N/A')}")
                            print()
                            
                except asyncio.TimeoutError:
                    # Print periodic status
                    if msg_count > 0 and (datetime.now() - start_time).seconds % 5 == 0:
                        print(f"  [{datetime.now().strftime('%H:%M:%S')}] Received {msg_count} messages total...")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(test_ws()))
