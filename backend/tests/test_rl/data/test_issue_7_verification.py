"""
Verification test for Issue #7: OrderbookClient global registry integration.

This test demonstrates that the fix allows other components to access orderbook state
that has been updated by OrderbookClient via the global get_shared_orderbook_state() function.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.data.orderbook_state import (
    get_shared_orderbook_state,
    cleanup_global_orderbook_states
)


@pytest.mark.asyncio
async def test_issue_7_orderbook_client_populates_global_registry():
    """
    Test that demonstrates Issue #7 is resolved.
    
    BEFORE THE FIX: OrderbookClient only updated its local _orderbook_states,
    but other components using get_shared_orderbook_state() couldn't access the data.
    
    AFTER THE FIX: OrderbookClient updates both local state and global registry,
    allowing other components to access the same data via get_shared_orderbook_state().
    """
    # Clean up global registry
    await cleanup_global_orderbook_states()
    
    # Create OrderbookClient
    client = OrderbookClient(market_tickers=["ISSUE-7-TEST"])
    
    # Simulate what happens during normal operation: client processes a snapshot
    snapshot_message = {
        "type": "orderbook_snapshot",
        "seq": 99999,
        "msg": {
            "market_ticker": "ISSUE-7-TEST",
            "yes": [[45, 100], [46, 200]],
            "no": [[54, 150], [55, 300]]
        }
    }
    
    # Mock dependencies to avoid actual DB writes
    with patch('kalshiflow_rl.data.orderbook_client.write_queue') as mock_write_queue:
        with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_snapshot'):
            mock_write_queue.enqueue_snapshot = AsyncMock(return_value=True)
            
            # Process the snapshot (this should update both local and global registries)
            await client._process_snapshot(snapshot_message)
    
    # NOW THE CRITICAL TEST: Can other components access the data via get_shared_orderbook_state?
    # This is what was BROKEN before Issue #7 fix
    
    # Simulate another component (like LiveObservationAdapter or MultiMarketOrderManager)
    # accessing the global registry
    async def simulate_other_component_access():
        """This is what other components do to access orderbook data."""
        shared_state = await get_shared_orderbook_state("ISSUE-7-TEST")
        snapshot = await shared_state.get_snapshot()
        return snapshot
    
    # Test that other component can access the data
    accessed_data = await simulate_other_component_access()
    
    # Verify the data is accessible and correct
    assert accessed_data is not None
    assert accessed_data['market_ticker'] == "ISSUE-7-TEST"
    assert accessed_data['last_sequence'] == 99999
    
    # Verify the orderbook data is present
    assert 45 in accessed_data['yes_bids']
    assert accessed_data['yes_bids'][45] == 100
    assert 46 in accessed_data['yes_bids']
    assert accessed_data['yes_bids'][46] == 200
    
    # Verify derived asks are present (Kalshi reciprocal relationships)
    # YES_BID at 45 → NO_ASK at (99-45) = 54
    # YES_BID at 46 → NO_ASK at (99-46) = 53
    assert 54 in accessed_data['no_asks']
    assert 53 in accessed_data['no_asks']
    
    print("✅ Issue #7 RESOLVED: Other components can access OrderbookClient data via global registry")


@pytest.mark.asyncio
async def test_issue_7_multiple_components_access_same_data():
    """
    Test that multiple components can access the same orderbook data concurrently.
    
    This demonstrates that the global registry approach works for the intended use case
    where multiple components (LiveObservationAdapter, MultiMarketOrderManager, etc.)
    need to access the same orderbook data.
    """
    # Clean up global registry
    await cleanup_global_orderbook_states()
    
    client = OrderbookClient(market_tickers=["MULTI-ACCESS-TEST"])
    
    # Process a snapshot
    snapshot_message = {
        "type": "orderbook_snapshot",
        "seq": 12345,
        "msg": {
            "market_ticker": "MULTI-ACCESS-TEST",
            "yes": [[50, 500]],
            "no": [[49, 400]]
        }
    }
    
    with patch('kalshiflow_rl.data.orderbook_client.write_queue') as mock_write_queue:
        with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_snapshot'):
            mock_write_queue.enqueue_snapshot = AsyncMock(return_value=True)
            await client._process_snapshot(snapshot_message)
    
    # Simulate multiple components accessing the data concurrently
    async def component_1_access():
        """Simulate LiveObservationAdapter accessing orderbook data."""
        state = await get_shared_orderbook_state("MULTI-ACCESS-TEST")
        snapshot = await state.get_snapshot()
        return snapshot['yes_bids'][50]  # Should be 500
    
    async def component_2_access():
        """Simulate MultiMarketOrderManager accessing orderbook data."""
        state = await get_shared_orderbook_state("MULTI-ACCESS-TEST")
        top_levels = await state.get_top_levels(1)
        return top_levels['yes_bids'][50]  # Should be 500
    
    async def component_3_access():
        """Simulate another component accessing spread data."""
        state = await get_shared_orderbook_state("MULTI-ACCESS-TEST")
        spreads = await state.get_spreads_and_mids()
        return spreads['yes_spread']  # Should be some calculated spread
    
    # Access data concurrently from multiple components
    results = await asyncio.gather(
        component_1_access(),
        component_2_access(), 
        component_3_access()
    )
    
    # All components should get the same underlying data
    assert results[0] == 500  # LiveObservationAdapter got correct bid size
    assert results[1] == 500  # MultiMarketOrderManager got same bid size
    assert results[2] is not None  # Spread calculation worked
    
    print("✅ Multiple components can access same orderbook data via global registry")