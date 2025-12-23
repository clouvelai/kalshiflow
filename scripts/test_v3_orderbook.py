#!/usr/bin/env python3
"""Test V3 Orderbook Integration."""

import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend', 'src'))

from kalshiflow_rl.traderv3.core.event_bus import EventBus
from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.traderv3.clients.orderbook_integration import V3OrderbookIntegration


async def test_v3_orderbook():
    """Test V3 orderbook integration with direct event bus."""
    print("Testing V3 Orderbook Integration...")
    
    # Create V3 event bus
    event_bus = EventBus()
    await event_bus.start()
    print("✅ V3 Event bus started")
    
    # Test market
    test_market = ["INXD-25JAN03"]
    
    # Create orderbook client with V3 event bus
    orderbook_client = OrderbookClient(
        market_tickers=test_market,
        event_bus=event_bus  # Pass V3 event bus directly
    )
    print(f"✅ OrderbookClient created with V3 event bus for {test_market}")
    
    # Create V3 orderbook integration
    orderbook_integration = V3OrderbookIntegration(
        orderbook_client=orderbook_client,
        event_bus=event_bus,
        market_tickers=test_market
    )
    print("✅ V3OrderbookIntegration created")
    
    # Subscribe to events to verify they're flowing
    snapshot_count = [0]
    delta_count = [0]
    
    async def on_snapshot(market_ticker: str, metadata: dict):
        snapshot_count[0] += 1
        print(f"📊 Snapshot {snapshot_count[0]}: {market_ticker}")
    
    async def on_delta(market_ticker: str, metadata: dict):
        delta_count[0] += 1
        if delta_count[0] % 10 == 0:  # Log every 10th delta
            print(f"📈 Delta {delta_count[0]}: {market_ticker}")
    
    await event_bus.subscribe_to_orderbook_snapshot(on_snapshot)
    await event_bus.subscribe_to_orderbook_delta(on_delta)
    print("✅ Subscribed to V3 events")
    
    # Start the integration
    await orderbook_integration.start()
    print("✅ Orderbook integration started")
    
    # Wait for connection
    connected = await orderbook_integration.wait_for_connection(timeout=30)
    if not connected:
        print("❌ Failed to connect to orderbook WebSocket")
        return False
    print("✅ Connected to orderbook WebSocket")
    
    # Wait for first snapshot
    snapshot_received = await orderbook_integration.wait_for_first_snapshot(timeout=10)
    if not snapshot_received:
        print("❌ No snapshot received")
        return False
    print(f"✅ First snapshot received!")
    
    # Let it run for a bit to collect some data
    print("\n⏳ Collecting data for 10 seconds...")
    await asyncio.sleep(10)
    
    # Get metrics
    metrics = orderbook_integration.get_metrics()
    print(f"\n📊 Final Metrics:")
    print(f"  Snapshots: {metrics['snapshots_received']}")
    print(f"  Deltas: {metrics['deltas_received']}")
    print(f"  Markets connected: {metrics['markets_connected']}")
    print(f"  V3 Event Bus Snapshots: {snapshot_count[0]}")
    print(f"  V3 Event Bus Deltas: {delta_count[0]}")
    
    # Verify data flowed through V3 event bus
    # Success if we got at least one snapshot (deltas may not come if market is inactive)
    success = snapshot_count[0] > 0
    
    # Stop everything
    await orderbook_integration.stop()
    await event_bus.stop()
    
    if success:
        print("\n✅ Test PASSED - V3 orderbook integration working correctly!")
    else:
        print("\n❌ Test FAILED - No data received through V3 event bus")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(test_v3_orderbook())
    sys.exit(0 if success else 1)