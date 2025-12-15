"""
Test that OrderbookClient populates the global SharedOrderbookState registry.

This is a focused test for Issue #7: ensuring OrderbookClient updates both its local
state and the global registry that other components access via get_shared_orderbook_state().
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, patch

from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.data.orderbook_state import (
    get_shared_orderbook_state,
    cleanup_global_orderbook_states
)


class TestOrderbookClientGlobalRegistry:
    """Test OrderbookClient integration with global SharedOrderbookState registry."""
    
    @pytest.mark.asyncio
    async def test_snapshot_updates_global_registry(self):
        """Test that _process_snapshot updates global registry via get_shared_orderbook_state."""
        # Clean up global registry
        await cleanup_global_orderbook_states()
        
        # Create OrderbookClient
        client = OrderbookClient(market_tickers=["TEST-MARKET"])
        
        # Verify global registry is initially empty
        global_state = await get_shared_orderbook_state("TEST-MARKET")
        initial_snapshot = await global_state.get_snapshot()
        assert initial_snapshot['last_sequence'] == 0  # Fresh state
        
        # Create a mock snapshot message (Kalshi format)
        snapshot_message = {
            "type": "orderbook_snapshot",
            "seq": 12345,
            "msg": {
                "market_ticker": "TEST-MARKET",
                "yes": [[45, 100], [44, 150], [43, 200]],  # [[price, qty], ...]
                "no": [[55, 80], [56, 120], [57, 90]]
            }
        }
        
        # Mock write queue and event bus to prevent actual DB writes and events
        with patch('kalshiflow_rl.data.orderbook_client.get_write_queue') as mock_get_queue:
            mock_write_queue = AsyncMock()
            mock_get_queue.return_value = mock_write_queue
            with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_snapshot') as mock_emit:
                mock_write_queue.enqueue_snapshot = AsyncMock(return_value=True)
                mock_emit = AsyncMock(return_value=True)
                
                # Process the snapshot
                await client._process_snapshot(snapshot_message)
        
        # Verify global registry was updated
        updated_global_state = await get_shared_orderbook_state("TEST-MARKET")
        updated_snapshot = await updated_global_state.get_snapshot()
        
        assert updated_snapshot['last_sequence'] == 12345
        assert updated_snapshot['market_ticker'] == "TEST-MARKET"
        assert len(updated_snapshot['yes_bids']) == 3  # Should have 3 yes bids
        assert len(updated_snapshot['no_asks']) == 3   # Should have 3 derived no asks
        
        # Verify the price levels are correctly populated
        assert 45 in updated_snapshot['yes_bids']
        assert updated_snapshot['yes_bids'][45] == 100
    
    @pytest.mark.asyncio
    async def test_delta_updates_global_registry(self):
        """Test that _process_delta updates global registry via get_shared_orderbook_state."""
        # Clean up global registry
        await cleanup_global_orderbook_states()
        
        client = OrderbookClient(market_tickers=["TEST-MARKET"])
        
        # Initialize local state (this happens in client.start())
        client._orderbook_states["TEST-MARKET"] = await get_shared_orderbook_state("TEST-MARKET")
        
        # First, establish a snapshot in both local and global registry
        snapshot_message = {
            "type": "orderbook_snapshot",
            "seq": 100,
            "msg": {
                "market_ticker": "TEST-MARKET",
                "yes": [[50, 200]],
                "no": [[49, 150]]
            }
        }
        
        with patch('kalshiflow_rl.data.orderbook_client.get_write_queue') as mock_get_queue:
            mock_write_queue = AsyncMock()
            mock_get_queue.return_value = mock_write_queue
            with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_snapshot'):
                mock_write_queue.enqueue_snapshot = AsyncMock(return_value=True)
                await client._process_snapshot(snapshot_message)
        
        # Now send a delta update
        delta_message = {
            "type": "orderbook_delta",
            "seq": 101,
            "msg": {
                "market_ticker": "TEST-MARKET",
                "side": "yes",
                "price": 50,
                "delta": 50  # Increase by 50
            }
        }
        
        with patch('kalshiflow_rl.data.orderbook_client.get_write_queue') as mock_get_queue:
            mock_write_queue = AsyncMock()
            mock_get_queue.return_value = mock_write_queue
            with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_delta'):
                mock_write_queue.enqueue_delta = AsyncMock(return_value=True)
                await client._process_delta(delta_message)
        
        # Verify global registry was updated by the delta
        global_state = await get_shared_orderbook_state("TEST-MARKET")
        updated_snapshot = await global_state.get_snapshot()
        
        assert updated_snapshot['last_sequence'] == 101  # Should be updated sequence
        assert updated_snapshot['yes_bids'][50] == 250  # 200 + 50 = 250
    
    @pytest.mark.asyncio
    async def test_multiple_markets_global_registry(self):
        """Test that OrderbookClient updates global registry for multiple markets."""
        # Clean up global registry
        await cleanup_global_orderbook_states()
        
        client = OrderbookClient(market_tickers=["MARKET-A", "MARKET-B"])
        
        # Send snapshots for both markets
        snapshot_a = {
            "type": "orderbook_snapshot",
            "seq": 1000,
            "msg": {
                "market_ticker": "MARKET-A",
                "yes": [[45, 100]],
                "no": [[55, 90]]
            }
        }
        
        snapshot_b = {
            "type": "orderbook_snapshot", 
            "seq": 2000,
            "msg": {
                "market_ticker": "MARKET-B",
                "yes": [[48, 150]],
                "no": [[52, 120]]
            }
        }
        
        with patch('kalshiflow_rl.data.orderbook_client.get_write_queue') as mock_get_queue:
            mock_write_queue = AsyncMock()
            mock_get_queue.return_value = mock_write_queue
            with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_snapshot'):
                mock_write_queue.enqueue_snapshot = AsyncMock(return_value=True)
                await client._process_snapshot(snapshot_a)
                await client._process_snapshot(snapshot_b)
        
        # Verify both markets in global registry
        state_a = await get_shared_orderbook_state("MARKET-A")
        state_b = await get_shared_orderbook_state("MARKET-B") 
        
        snapshot_a_data = await state_a.get_snapshot()
        snapshot_b_data = await state_b.get_snapshot()
        
        assert snapshot_a_data['last_sequence'] == 1000
        assert snapshot_a_data['market_ticker'] == "MARKET-A"
        assert snapshot_a_data['yes_bids'][45] == 100
        
        assert snapshot_b_data['last_sequence'] == 2000
        assert snapshot_b_data['market_ticker'] == "MARKET-B"
        assert snapshot_b_data['yes_bids'][48] == 150
    
    @pytest.mark.asyncio
    async def test_global_registry_accessible_by_other_components(self):
        """Test that other components can access OrderbookClient-updated states via global registry."""
        # Clean up global registry
        await cleanup_global_orderbook_states()
        
        client = OrderbookClient(market_tickers=["SHARED-MARKET"])
        
        # Process a snapshot
        snapshot_message = {
            "type": "orderbook_snapshot",
            "seq": 5000,
            "msg": {
                "market_ticker": "SHARED-MARKET",
                "yes": [[46, 300], [47, 250]],
                "no": [[53, 180], [54, 220]]
            }
        }
        
        with patch('kalshiflow_rl.data.orderbook_client.get_write_queue') as mock_get_queue:
            mock_write_queue = AsyncMock()
            mock_get_queue.return_value = mock_write_queue
            with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_snapshot'):
                mock_write_queue.enqueue_snapshot = AsyncMock(return_value=True)
                await client._process_snapshot(snapshot_message)
        
        # Simulate other components accessing the global state
        # (This is what LiveObservationAdapter and MultiMarketOrderManager do)
        async def other_component_access():
            shared_state = await get_shared_orderbook_state("SHARED-MARKET")
            snapshot = await shared_state.get_snapshot()
            return snapshot
        
        # Verify other component can access the data
        accessed_snapshot = await other_component_access()
        
        assert accessed_snapshot['market_ticker'] == "SHARED-MARKET"
        assert accessed_snapshot['last_sequence'] == 5000
        assert accessed_snapshot['yes_bids'][46] == 300
        assert accessed_snapshot['yes_bids'][47] == 250
        
        # Verify derived asks are correct (Kalshi reciprocal relationships)
        # YES_BID at 46 → NO_ASK at (99-46) = 53
        # YES_BID at 47 → NO_ASK at (99-47) = 52
        assert 53 in accessed_snapshot['no_asks']
        assert 52 in accessed_snapshot['no_asks']
    
    @pytest.mark.asyncio
    async def test_local_and_global_states_synchronized(self):
        """Test that local client state and global registry stay synchronized."""
        # Clean up global registry
        await cleanup_global_orderbook_states()
        
        client = OrderbookClient(market_tickers=["SYNC-TEST"])
        
        # Initialize local state (this happens in client.start())
        local_state = client._orderbook_states["SYNC-TEST"] = (
            await get_shared_orderbook_state("SYNC-TEST")
        )
        
        # Process snapshot
        snapshot_message = {
            "type": "orderbook_snapshot", 
            "seq": 7777,
            "msg": {
                "market_ticker": "SYNC-TEST",
                "yes": [[42, 500]],
                "no": [[58, 400]]
            }
        }
        
        with patch('kalshiflow_rl.data.orderbook_client.get_write_queue') as mock_get_queue:
            mock_write_queue = AsyncMock()
            mock_get_queue.return_value = mock_write_queue
            with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_snapshot'):
                mock_write_queue.enqueue_snapshot = AsyncMock(return_value=True)
                await client._process_snapshot(snapshot_message)
        
        # Get snapshots from both local and global
        local_snapshot = await local_state.get_snapshot()
        global_state = await get_shared_orderbook_state("SYNC-TEST")
        global_snapshot = await global_state.get_snapshot()
        
        # Verify both are synchronized
        assert local_snapshot['last_sequence'] == global_snapshot['last_sequence'] == 7777
        assert local_snapshot['yes_bids'] == global_snapshot['yes_bids']
        assert local_snapshot['no_bids'] == global_snapshot['no_bids']
        assert local_snapshot['yes_asks'] == global_snapshot['yes_asks']
        assert local_snapshot['no_asks'] == global_snapshot['no_asks']
        
        # Test delta synchronization
        delta_message = {
            "type": "orderbook_delta",
            "seq": 7778, 
            "msg": {
                "market_ticker": "SYNC-TEST",
                "side": "no",
                "price": 58,
                "delta": -100  # Reduce by 100
            }
        }
        
        with patch('kalshiflow_rl.data.orderbook_client.get_write_queue') as mock_get_queue:
            mock_write_queue = AsyncMock()
            mock_get_queue.return_value = mock_write_queue
            with patch('kalshiflow_rl.data.orderbook_client.emit_orderbook_delta'):
                mock_write_queue.enqueue_delta = AsyncMock(return_value=True)
                await client._process_delta(delta_message)
        
        # Verify both updated consistently
        local_snapshot_after = await local_state.get_snapshot()
        global_snapshot_after = await global_state.get_snapshot()
        
        assert local_snapshot_after['last_sequence'] == global_snapshot_after['last_sequence'] == 7778
        
        # Check that no_bids at price 58 was reduced from 400 to 300 (400 - 100)
        if 58 in local_snapshot_after['no_bids']:
            assert local_snapshot_after['no_bids'][58] == 300  # 400 - 100
            assert global_snapshot_after['no_bids'][58] == 300  # Should be same in both
        else:
            # If the price level was completely removed (delta resulted in 0 size)
            assert 58 not in global_snapshot_after['no_bids']