#!/usr/bin/env python3
import asyncio
import json
import websockets
from datetime import datetime

async def monitor_ws():
    uri = "ws://localhost:8003/rl/ws"
    print(f"Connecting to {uri}...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Monitoring messages...\n")
        
        # Listen for first 10 messages
        for i in range(10):
            message = await websocket.recv()
            msg = json.loads(message)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] Message {i+1} - Type: {msg.get('type', 'unknown')}")
            
            if msg.get('type') == 'trader_state':
                data = msg.get('data', {})
                print(f"  State: {data.get('state', 'MISSING')}")
                print(f"  Cash: ${data.get('cash_balance', 'MISSING')}")
                print(f"  Portfolio: ${data.get('portfolio_value', 'MISSING')}")
                print(f"  Total: ${data.get('total_value', 'MISSING')}")
                print(f"  Raw data keys: {list(data.keys())}")
            elif msg.get('type') == 'error':
                print(f"  Error: {msg.get('data', {}).get('message', 'Unknown error')}")
            
            print()

asyncio.run(monitor_ws())
