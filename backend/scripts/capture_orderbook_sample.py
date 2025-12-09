#!/usr/bin/env python
"""Capture a sample orderbook directly from Kalshi WebSocket."""

import asyncio
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import sys

# Load environment FIRST before any imports
load_dotenv('.env.local')

# Add src to path
sys.path.insert(0, '/Users/samuelclark/Desktop/kalshiflow/backend/src')

# Import RL modules (after env is loaded)
from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.data.orderbook_state import SharedOrderbookState
from kalshiflow_rl.config import RLConfig

async def capture_orderbook_sample():
    """Capture a sample orderbook from Kalshi."""
    print("🚀 Connecting to Kalshi orderbook WebSocket...")
    
    # Create shared state
    shared_state = SharedOrderbookState("KXCABOUT-29")
    
    # Track messages
    messages_received = []
    snapshot_received = None
    
    # Create client with message capture
    client = OrderbookClient(
        market_ticker="KXCABOUT-29",
        shared_state=shared_state,
        write_queue=None  # Don't write to DB
    )
    
    # Override the message handler to capture messages
    original_handler = client._process_message
    
    async def capture_handler(message):
        """Capture messages while processing."""
        messages_received.append(message)
        
        # Check if it's a snapshot
        if message.get('type') == 'snapshot':
            nonlocal snapshot_received
            snapshot_received = message
            print("📸 Snapshot received!")
        elif message.get('type') == 'delta':
            print(f"📝 Delta #{len(messages_received)} received")
        
        # Call original handler
        await original_handler(message)
    
    client._process_message = capture_handler
    
    # Run for 30 seconds to capture data
    print("⏳ Capturing orderbook data for 30 seconds...")
    
    try:
        # Start the client
        connect_task = asyncio.create_task(client.connect())
        
        # Wait for some messages
        await asyncio.sleep(30)
        
        # Get the current state
        state_snapshot = await shared_state.get_snapshot()
        
        print("\n" + "=" * 80)
        print("📊 CAPTURED ORDERBOOK DATA")
        print("=" * 80)
        
        if snapshot_received:
            print("\n🎯 INITIAL SNAPSHOT:")
            snapshot_data = snapshot_received.get('msg', {})
            
            # Parse YES orderbook
            yes_book = snapshot_data.get('yes', {})
            print("\n📈 YES Side:")
            if yes_book.get('bids'):
                print("  Top 5 Bids:")
                for i, bid in enumerate(yes_book['bids'][:5], 1):
                    print(f"    #{i}: {bid[0]}¢ × {bid[1]} contracts")
            if yes_book.get('asks'):
                print("  Top 5 Asks:")
                for i, ask in enumerate(yes_book['asks'][:5], 1):
                    print(f"    #{i}: {ask[0]}¢ × {ask[1]} contracts")
            
            # Parse NO orderbook
            no_book = snapshot_data.get('no', {})
            print("\n📉 NO Side:")
            if no_book.get('bids'):
                print("  Top 5 Bids:")
                for i, bid in enumerate(no_book['bids'][:5], 1):
                    print(f"    #{i}: {bid[0]}¢ × {bid[1]} contracts")
            if no_book.get('asks'):
                print("  Top 5 Asks:")
                for i, ask in enumerate(no_book['asks'][:5], 1):
                    print(f"    #{i}: {ask[0]}¢ × {ask[1]} contracts")
        
        print(f"\n📬 Total messages received: {len(messages_received)}")
        print(f"   - Snapshots: {sum(1 for m in messages_received if m.get('type') == 'snapshot')}")
        print(f"   - Deltas: {sum(1 for m in messages_received if m.get('type') == 'delta')}")
        
        # Show current state
        if state_snapshot:
            print("\n🔄 CURRENT IN-MEMORY STATE:")
            print(f"   Market: {state_snapshot['market_ticker']}")
            print(f"   Last Sequence: {state_snapshot['last_sequence']}")
            
            # YES state
            yes_state = state_snapshot['yes_book']
            if yes_state['bids']:
                best_bid = max(yes_state['bids'].items(), key=lambda x: x[0])
                print(f"   YES Best Bid: {best_bid[0]}¢ × {best_bid[1]}")
            if yes_state['asks']:
                best_ask = min(yes_state['asks'].items(), key=lambda x: x[0])
                print(f"   YES Best Ask: {best_ask[0]}¢ × {best_ask[1]}")
            
            # NO state
            no_state = state_snapshot['no_book']
            if no_state['bids']:
                best_bid = max(no_state['bids'].items(), key=lambda x: x[0])
                print(f"   NO Best Bid: {best_bid[0]}¢ × {best_bid[1]}")
            if no_state['asks']:
                best_ask = min(no_state['asks'].items(), key=lambda x: x[0])
                print(f"   NO Best Ask: {best_ask[0]}¢ × {best_ask[1]}")
        
        # Save sample to file
        sample_file = '/tmp/orderbook_sample.json'
        with open(sample_file, 'w') as f:
            json.dump({
                'captured_at': datetime.utcnow().isoformat(),
                'market': 'KXCABOUT-29',
                'snapshot': snapshot_received,
                'total_messages': len(messages_received),
                'current_state': state_snapshot
            }, f, indent=2, default=str)
        
        print(f"\n💾 Sample saved to: {sample_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cancel the connection
        connect_task.cancel()
        try:
            await connect_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    asyncio.run(capture_orderbook_sample())