#!/usr/bin/env python
"""Simple script to connect to Kalshi and show orderbook data."""

import asyncio
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment
load_dotenv('.env.local')

# Now we can import (after env is loaded)
import sys
sys.path.insert(0, '/Users/samuelclark/Desktop/kalshiflow/backend/src')

from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.data.orderbook_state import get_shared_orderbook_state


async def show_orderbook():
    """Connect and show orderbook data."""
    print("🚀 Connecting to Kalshi orderbook WebSocket for KXCABOUT-29...")
    
    # Create client for our market
    client = OrderbookClient(market_ticker="KXCABOUT-29")
    
    # Get the shared state
    shared_state = await get_shared_orderbook_state("KXCABOUT-29")
    
    # Connect to WebSocket
    connect_task = asyncio.create_task(client.start())
    
    # Wait for connection and initial data
    print("⏳ Waiting for orderbook data...")
    await asyncio.sleep(10)  # Give more time for initial snapshot
    
    # Get current state
    state = await shared_state.get_snapshot()
    
    print("\n" + "=" * 80)
    print("📊 KXCABOUT-29 ORDERBOOK SNAPSHOT")
    print("=" * 80)
    
    if state:
        print(f"\nMarket: {state['market_ticker']}")
        print(f"Last Sequence: {state['last_sequence']}")
        print(f"Timestamp: {datetime.utcnow().isoformat()}")
        
        # YES orderbook
        print("\n📈 YES Side Orderbook:")
        yes_bids = state.get('yes_bids', {})
        yes_asks = state.get('yes_asks', {})
        
        if yes_bids:
            print("  Bids (Buy Orders):")
            sorted_bids = sorted(yes_bids.items(), key=lambda x: -x[0])
            for i, (price, size) in enumerate(sorted_bids[:5], 1):
                print(f"    #{i}: {price}¢ × {size} contracts")
        else:
            print("  No bids")
        
        if yes_asks:
            print("  Asks (Sell Orders):")
            sorted_asks = sorted(yes_asks.items(), key=lambda x: x[0])
            for i, (price, size) in enumerate(sorted_asks[:5], 1):
                print(f"    #{i}: {price}¢ × {size} contracts")
        else:
            print("  No asks")
        
        # Show YES spread and mid from state
        if state.get('yes_spread') is not None:
            print(f"  Spread: {state['yes_spread']}¢, Mid: {state.get('yes_mid_price', 'N/A'):.1f}¢" if state.get('yes_mid_price') else f"  Spread: {state['yes_spread']}¢")
        
        # NO orderbook
        print("\n📉 NO Side Orderbook:")
        no_bids = state.get('no_bids', {})
        no_asks = state.get('no_asks', {})
        
        if no_bids:
            print("  Bids (Buy Orders):")
            sorted_bids = sorted(no_bids.items(), key=lambda x: -x[0])
            for i, (price, size) in enumerate(sorted_bids[:5], 1):
                print(f"    #{i}: {price}¢ × {size} contracts")
        else:
            print("  No bids")
        
        if no_asks:
            print("  Asks (Sell Orders):")
            sorted_asks = sorted(no_asks.items(), key=lambda x: x[0])
            for i, (price, size) in enumerate(sorted_asks[:5], 1):
                print(f"    #{i}: {price}¢ × {size} contracts")
        else:
            print("  No asks")
        
        # Show NO spread and mid from state
        if state.get('no_spread') is not None:
            print(f"  Spread: {state['no_spread']}¢, Mid: {state.get('no_mid_price', 'N/A'):.1f}¢" if state.get('no_mid_price') else f"  Spread: {state['no_spread']}¢")
        
        # Market insights
        print("\n💡 Market Insights:")
        
        # Check if there's any volume
        total_yes_bids = sum(yes_bids.values())
        total_yes_asks = sum(yes_asks.values())
        total_no_bids = sum(no_bids.values())
        total_no_asks = sum(no_asks.values())
        
        print(f"  Total YES liquidity: {total_yes_bids} bids, {total_yes_asks} asks")
        print(f"  Total NO liquidity: {total_no_bids} bids, {total_no_asks} asks")
        
        # Implied probability from mid prices
        if state.get('yes_mid_price'):
            print(f"  Implied YES probability: {state['yes_mid_price']:.1f}%")
        
        if state.get('no_mid_price'):
            print(f"  Implied NO probability: {state['no_mid_price']:.1f}%")
        
        # Total volume
        if state.get('total_volume'):
            print(f"  Total orderbook volume: {state['total_volume']} contracts")
    else:
        print("❌ No orderbook data received yet")
    
    print("\n" + "=" * 80)
    
    # Cancel connection
    connect_task.cancel()
    try:
        await connect_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(show_orderbook())