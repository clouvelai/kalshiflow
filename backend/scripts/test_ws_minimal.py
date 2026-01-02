#!/usr/bin/env python3
"""
Minimal WebSocket connection test for Kalshi demo API.
Tests connection with minimal parameters.
"""

import asyncio
import websockets
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshiflow_rl.data.auth import get_rl_auth


async def test_connection():
    """Test WebSocket connection with minimal parameters."""
    
    print("Testing Kalshi WebSocket connection with minimal parameters...")
    
    # Get auth headers
    auth = get_rl_auth()
    headers = auth.create_websocket_headers()
    
    ws_url = "wss://demo-api.kalshi.co/trade-api/ws/v2"
    
    print(f"Connecting to: {ws_url}")
    print("Using minimal connection parameters (defaults only)")
    
    try:
        # Try connection with minimal parameters
        async with websockets.connect(
            ws_url,
            additional_headers=headers
        ) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Try to subscribe to a test market
            test_sub = {
                "id": 1,
                "cmd": "subscribe",
                "params": {
                    "channels": ["orderbook_delta"],
                    "market_tickers": ["KXNFLREC-25DEC28TBMIA-MIADWALLER83-6"]
                }
            }
            
            await websocket.send(str(test_sub).replace("'", '"'))
            print("Sent test subscription...")
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            print(f"Received response: {response[:200]}")
            
            print("\n✅ WebSocket connection and subscription successful!")
            return True
            
    except websockets.exceptions.InvalidStatus as e:
        print(f"\n❌ Connection rejected: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Connection failed: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)