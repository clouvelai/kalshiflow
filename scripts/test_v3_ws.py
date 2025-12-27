#!/usr/bin/env python
"""Test V3 WebSocket to verify duplicate message fix."""

import asyncio
import json
import websockets
import sys
from datetime import datetime

async def test_v3_websocket():
    """Connect to V3 WebSocket and monitor for duplicate messages."""
    uri = "ws://localhost:8005/v3/ws"
    
    print(f"Connecting to {uri}...")
    
    message_cache = {}  # Track recent messages to detect duplicates
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected! Monitoring for messages...")
            print("-" * 60)
            
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Check for system_activity messages
                if data.get("type") == "system_activity":
                    activity = data["data"]
                    timestamp = activity.get("timestamp", "")
                    state = activity.get("state", "UNKNOWN")
                    msg_text = activity.get("message", "")
                    
                    # Create a unique key for this message (ignoring timestamp)
                    msg_key = f"{activity.get('activity_type')}:{msg_text}"
                    
                    # Check if we've seen this exact message very recently (within 0.5 seconds)
                    now = datetime.now()
                    if msg_key in message_cache:
                        last_seen = message_cache[msg_key]
                        if (now - last_seen).total_seconds() < 0.5:
                            print(f"⚠️  DUPLICATE: [{state}] {msg_text}")
                        else:
                            print(f"✓  [{state}] {msg_text}")
                    else:
                        print(f"✓  [{state}] {msg_text}")
                    
                    message_cache[msg_key] = now
                    
                    # Clean old entries from cache
                    for k in list(message_cache.keys()):
                        if (now - message_cache[k]).total_seconds() > 2:
                            del message_cache[k]
                            
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(test_v3_websocket()))