"""
Integration tests for RL Trading Subsystem.

Tests the complete pipeline from WebSocket message processing
through orderbook state updates to database persistence.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import AsyncMock, MagicMock, patch

from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.data.orderbook_state import get_shared_orderbook_state
from kalshiflow_rl.data.write_queue import OrderbookWriteQueue, get_write_queue
from kalshiflow_rl.data.database import rl_db


@pytest.fixture(scope="function")
def mock_websocket():
    """Mock WebSocket for testing."""
    websocket = MagicMock()
    websocket.__aenter__ = AsyncMock(return_value=websocket)
    websocket.__aexit__ = AsyncMock(return_value=None)
    websocket.send = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


@pytest.fixture
def sample_websocket_snapshot():
    """Sample WebSocket snapshot message."""
    return {
        "type": "orderbook_snapshot",
        "seq": 1000,
        "msg": {
            "market_ticker": "TEST-MARKET",
            "yes": [
                [45, 1000],  # Bid at 45
                [44, 500],   # Bid at 44
                [55, 800],   # Ask at 55
                [56, 400]    # Ask at 56
            ],
            "no": [
                [45, 600],   # Bid at 45
                [44, 300],   # Bid at 44
                [55, 700],   # Ask at 55
                [56, 200]    # Ask at 56
            ]
        }
    }


@pytest.fixture
def sample_websocket_delta():
    """Sample WebSocket delta message."""
    return {
        "type": "orderbook_delta",
        "seq": 1001,  # Sequence number in outer message like snapshots
        "msg": {
            "market_ticker": "TEST-MARKET",
            "side": "yes",
            "price": 45,
            "delta": 200  # Positive delta means size increased by 200
        }
    }


class TestCompleteDataPipeline:
    """Test the complete data flow pipeline."""
    
    @pytest.mark.asyncio
    @patch('kalshiflow_rl.data.write_queue.rl_db')
    async def test_end_to_end_message_flow(
        self,
        mock_rl_db,
        sample_websocket_snapshot,
        sample_websocket_delta
    ):
        """Test complete message flow from message processing to orderbook state updates."""
        
        # Setup database mock
        mock_rl_db.batch_insert_snapshots = AsyncMock(return_value=1)
        mock_rl_db.batch_insert_deltas = AsyncMock(return_value=1)
        
        # Create and start a test write queue
        test_write_queue = OrderbookWriteQueue(
            batch_size=10,
            flush_interval=0.1,
            max_queue_size=100
        )
        # Set session ID so writes are actually persisted
        test_write_queue.set_session_id(123)
        await test_write_queue.start()
        
        try:
            # Patch the global write_queue import in orderbook_client
            with patch('kalshiflow_rl.data.orderbook_client.get_write_queue', return_value=test_write_queue):
                # Create client (no WebSocket connection needed for this test)
                client = OrderbookClient("TEST-MARKET")
                
                # Initialize orderbook states for all markets the client handles
                for market_ticker in client.market_tickers:
                    client._orderbook_states[market_ticker] = await get_shared_orderbook_state(market_ticker)
                
                # Process snapshot message directly
                await client._process_message(json.dumps(sample_websocket_snapshot))
                
                # Process delta message directly  
                await client._process_message(json.dumps(sample_websocket_delta))
                
                # Wait for any async processing
                await asyncio.sleep(0.2)
                
                # Verify orderbook state was updated
                shared_state = await get_shared_orderbook_state("TEST-MARKET")
                snapshot = await shared_state.get_snapshot()
                
                assert snapshot["last_sequence"] == 1001  # Delta updates sequence
                assert snapshot["market_ticker"] == "TEST-MARKET"
                
                # Verify snapshot has expected structure
                assert "yes_bids" in snapshot
                assert "yes_asks" in snapshot
                assert "no_bids" in snapshot
                assert "no_asks" in snapshot
                
                # Verify specific bid/ask data from test messages
                # Snapshot sets yes_bid at 45 to 1000
                # Delta adds 200 to the existing size: 1000 + 200 = 1200
                assert 45 in snapshot["yes_bids"]  # From snapshot
                assert snapshot["yes_bids"][45] == 1200  # Updated by delta (delta=200 adds to existing)
                
                # Verify that write queue received messages
                stats = test_write_queue.get_stats()
                assert stats["messages_enqueued"] >= 2  # At least snapshot and delta
            
        finally:
            # Stop the write queue
            await test_write_queue.stop()
    
    @pytest.mark.asyncio
    async def test_non_blocking_behavior(self):
        """Test that message processing doesn't block on slow operations."""
        
        # Create write queue with slow flush
        write_queue = OrderbookWriteQueue(
            batch_size=1000,  # Large batch to prevent immediate flush
            flush_interval=10.0  # Long interval
        )
        
        await write_queue.start()
        
        try:
            # Enqueue many messages and measure time
            num_messages = 1000
            start_time = time.time()
            
            for i in range(num_messages):
                snapshot_data = {
                    "market_ticker": "TEST-MARKET",
                    "timestamp_ms": int(time.time() * 1000),
                    "sequence_number": i,
                    "yes_bids": {"45": 1000},
                    "yes_asks": {"55": 800},
                    "no_bids": {"45": 600},
                    "no_asks": {"55": 700}
                }
                
                await write_queue.enqueue_snapshot(snapshot_data)
            
            enqueue_time = time.time() - start_time
            
            # Should be very fast (non-blocking)
            assert enqueue_time < 1.0
            
            # All messages should be queued
            assert write_queue._messages_enqueued == num_messages
            
        finally:
            await write_queue.stop()
    
    @pytest.mark.asyncio
    @patch('kalshiflow_rl.data.write_queue.rl_db')
    async def test_message_persistence_accuracy(self, mock_rl_db):
        """Test that messages are persisted accurately."""
        
        mock_rl_db.batch_insert_snapshots = AsyncMock()
        mock_rl_db.batch_insert_deltas = AsyncMock()
        
        write_queue = OrderbookWriteQueue(
            batch_size=2,
            flush_interval=0.1,
        )
        
        # Set session ID so writes are actually persisted
        write_queue.set_session_id(123)
        
        await write_queue.start()
        
        try:
            # Enqueue specific test data
            snapshot1 = {
                "market_ticker": "TEST-MARKET",
                "timestamp_ms": 1000,
                "sequence_number": 100,
                "yes_bids": {"45": 1000},
                "yes_asks": {"55": 800},
                "no_bids": {"45": 600},
                "no_asks": {"55": 700}
            }
            
            snapshot2 = {
                "market_ticker": "TEST-MARKET",
                "timestamp_ms": 2000,
                "sequence_number": 200,
                "yes_bids": {"46": 1100},
                "yes_asks": {"54": 900},
                "no_bids": {"46": 650},
                "no_asks": {"54": 750}
            }
            
            await write_queue.enqueue_snapshot(snapshot1)
            await write_queue.enqueue_snapshot(snapshot2)
            
            # Wait for batch processing
            await asyncio.sleep(0.2)
            
            # Verify correct data was passed to database
            mock_rl_db.batch_insert_snapshots.assert_called_once()
            
            call_args = mock_rl_db.batch_insert_snapshots.call_args[0][0]
            assert len(call_args) == 2
            
            # Check first snapshot
            assert call_args[0]["market_ticker"] == "TEST-MARKET"
            assert call_args[0]["sequence_number"] == 100
            assert call_args[0]["timestamp_ms"] == 1000
            
            # Check second snapshot
            assert call_args[1]["market_ticker"] == "TEST-MARKET"
            assert call_args[1]["sequence_number"] == 200
            assert call_args[1]["timestamp_ms"] == 2000
            
        finally:
            await write_queue.stop()
    
    @pytest.mark.asyncio
    @patch('kalshiflow_rl.data.write_queue.rl_db')
    async def test_multi_market_isolation(self, mock_rl_db):
        """Test that multi-market client properly isolates market data."""
        
        # Setup database mock
        mock_rl_db.batch_insert_snapshots = AsyncMock(return_value=1)
        mock_rl_db.batch_insert_deltas = AsyncMock(return_value=1)
        
        # Create and start a test write queue
        test_write_queue = OrderbookWriteQueue(
            batch_size=10,
            flush_interval=0.1,
            max_queue_size=100
        )
        # Set session ID so writes are actually persisted
        test_write_queue.set_session_id(123)
        await test_write_queue.start()
        
        try:
            # Patch the global write_queue import
            with patch('kalshiflow_rl.data.orderbook_client.get_write_queue', return_value=test_write_queue):
                # Create multi-market client
                client = OrderbookClient(["MARKET-A", "MARKET-B"])
                
                # Initialize orderbook states for all markets
                for market_ticker in client.market_tickers:
                    client._orderbook_states[market_ticker] = await get_shared_orderbook_state(market_ticker)
                
                # Create snapshots for different markets
                # Use Kalshi array format [[price, qty], ...]
                snapshot_a = {
                    "type": "orderbook_snapshot",
                    "seq": 100,
                    "msg": {
                        "market_ticker": "MARKET-A",
                        "yes": [[50, 1000], [55, 800]],  # 50 is bid, 55 is ask
                        "no": [[50, 600], [55, 700]]
                    }
                }
                
                snapshot_b = {
                    "type": "orderbook_snapshot",
                    "seq": 200,
                    "msg": {
                        "market_ticker": "MARKET-B",
                        "yes": [[45, 2000], [60, 1200]],  # 45 is bid, 60 is ask
                        "no": [[45, 1000], [60, 800]]
                    }
                }
                
                # Process snapshots for both markets
                await client._process_message(json.dumps(snapshot_a))
                await client._process_message(json.dumps(snapshot_b))
                
                # Wait for processing
                await asyncio.sleep(0.1)
                
                # Verify isolation: each market has its own state
                state_a = await get_shared_orderbook_state("MARKET-A")
                state_b = await get_shared_orderbook_state("MARKET-B")
                
                snapshot_a_result = await state_a.get_snapshot()
                snapshot_b_result = await state_b.get_snapshot()
                
                # Market A should only have Market A data
                assert snapshot_a_result["market_ticker"] == "MARKET-A"
                assert snapshot_a_result["last_sequence"] == 100
                assert 50 in snapshot_a_result["yes_bids"]
                assert snapshot_a_result["yes_bids"][50] == 1000
                
                # Market B should only have Market B data
                assert snapshot_b_result["market_ticker"] == "MARKET-B"
                assert snapshot_b_result["last_sequence"] == 200
                assert 45 in snapshot_b_result["yes_bids"]
                assert snapshot_b_result["yes_bids"][45] == 2000
                
                # Cross-contamination check: Market A should not have Market B's price levels
                assert 45 not in snapshot_a_result["yes_bids"]
                assert 50 not in snapshot_b_result["yes_bids"]
                
                # Process delta for Market A only
                delta_a = {
                    "type": "orderbook_delta",
                    "seq": 101,  # Sequence number for MARKET-A
                    "msg": {
                        "market_ticker": "MARKET-A",
                        "side": "yes",
                        "price": 50,
                        "delta": 500  # Add 500 to bid at 50
                    }
                }
                
                await client._process_message(json.dumps(delta_a))
                await asyncio.sleep(0.1)
                
                # Verify only Market A was updated
                updated_state_a = await get_shared_orderbook_state("MARKET-A")
                updated_snapshot_a = await updated_state_a.get_snapshot()
                unchanged_state_b = await get_shared_orderbook_state("MARKET-B")
                unchanged_snapshot_b = await unchanged_state_b.get_snapshot()
                
                assert updated_snapshot_a["last_sequence"] == 101  # Seq updated from delta
                # Delta of 500 adds to existing 1000 → 1500
                assert updated_snapshot_a["yes_bids"][50] == 1500
                
                assert unchanged_snapshot_b["last_sequence"] == 200  # Unchanged
                assert unchanged_snapshot_b["yes_bids"][45] == 2000  # Unchanged
                
                # Verify write queue received messages for both markets
                stats = test_write_queue.get_stats()
                assert stats["messages_enqueued"] >= 3  # 2 snapshots + 1 delta
            
        finally:
            await test_write_queue.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_updates_consistency(self):
        """Test that concurrent updates maintain consistency."""
        
        shared_state = await get_shared_orderbook_state("TEST-MARKET")
        
        # Apply initial snapshot
        initial_snapshot = {
            "market_ticker": "TEST-MARKET",
            "timestamp_ms": int(time.time() * 1000),
            "sequence_number": 100,
            "yes_bids": {"45": 1000},
            "yes_asks": {"55": 800},
            "no_bids": {"45": 600},
            "no_asks": {"55": 700}
        }
        
        await shared_state.apply_snapshot(initial_snapshot)
        
        # Create concurrent delta applications
        async def apply_delta(seq):
            delta = {
                "market_ticker": "TEST-MARKET",
                "timestamp_ms": int(time.time() * 1000),
                "sequence_number": 100 + seq,
                "side": "yes",
                "action": "update",
                "price": 45,
                "old_size": 1000,
                "new_size": 1000 + seq * 100
            }
            return await shared_state.apply_delta(delta)
        
        # Run concurrent updates
        tasks = [apply_delta(i) for i in range(1, 11)]  # seq 101-110
        results = await asyncio.gather(*tasks)
        
        # All should succeed (though order might vary due to concurrency)
        assert all(results)
        
        # Final state should be consistent
        final_snapshot = await shared_state.get_snapshot()
        assert final_snapshot["last_sequence"] >= 101  # At least one update applied
        assert "yes_bids" in final_snapshot
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under realistic load."""
        
        # Create components with realistic settings
        write_queue = OrderbookWriteQueue(
            batch_size=100,
            flush_interval=1.0,
            max_queue_size=10000
        )
        
        shared_state = await get_shared_orderbook_state("LOAD-TEST")
        
        await write_queue.start()
        
        try:
            # Apply initial snapshot
            initial_snapshot = {
                "market_ticker": "LOAD-TEST",
                "timestamp_ms": int(time.time() * 1000),
                "sequence_number": 0,
                "yes_bids": {"45": 1000},
                "yes_asks": {"55": 800},
                "no_bids": {"45": 600},
                "no_asks": {"55": 700}
            }
            
            await shared_state.apply_snapshot(initial_snapshot)
            await write_queue.enqueue_snapshot(initial_snapshot)
            
            # Generate high-frequency deltas (simulate real trading)
            num_deltas = 5000
            start_time = time.time()
            
            for i in range(num_deltas):
                delta_data = {
                    "market_ticker": "LOAD-TEST",
                    "timestamp_ms": int(time.time() * 1000),
                    "sequence_number": i + 1,
                    "side": "yes" if i % 2 == 0 else "no",
                    "action": "update",
                    "price": 45 if i % 3 == 0 else 55,
                    "old_size": 1000,
                    "new_size": 1000 + (i % 500)
                }
                
                # Update state and queue for persistence
                await shared_state.apply_delta(delta_data)
                await write_queue.enqueue_delta(delta_data)
            
            processing_time = time.time() - start_time
            
            # Should handle 5000 updates in reasonable time
            assert processing_time < 10.0  # Less than 10 seconds
            
            # System should remain responsive
            final_snapshot = await shared_state.get_snapshot()
            assert final_snapshot["last_sequence"] > 0
            
            # Write queue should be healthy
            assert write_queue.is_healthy()
            
            # Check throughput
            throughput = num_deltas / processing_time
            assert throughput > 500  # At least 500 updates/sec
            
        finally:
            await write_queue.stop()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test system recovery from various error conditions."""
        
        # Test recovery from orderbook state corruption
        shared_state = await get_shared_orderbook_state("ERROR-TEST")
        
        # Apply valid snapshot
        valid_snapshot = {
            "market_ticker": "ERROR-TEST",
            "timestamp_ms": int(time.time() * 1000),
            "sequence_number": 100,
            "yes_bids": {"45": 1000},
            "yes_asks": {"55": 800},
            "no_bids": {"45": 600},
            "no_asks": {"55": 700}
        }
        
        await shared_state.apply_snapshot(valid_snapshot)
        
        # Try to apply invalid deltas
        invalid_deltas = [
            {
                "sequence_number": 50,  # Out of order
                "side": "yes",
                "action": "update",
                "price": 45,
                "new_size": 1200
            },
            {
                "sequence_number": 101,
                "side": "invalid",  # Invalid side
                "action": "update",
                "price": 45,
                "new_size": 1200
            },
            {
                "sequence_number": 102,
                "side": "yes",
                "action": "invalid",  # Invalid action
                "price": 45,
                "new_size": 1200
            }
        ]
        
        # Apply invalid deltas (should be rejected)
        for delta in invalid_deltas:
            delta["market_ticker"] = "ERROR-TEST"
            delta["timestamp_ms"] = int(time.time() * 1000)
            result = await shared_state.apply_delta(delta)
            assert result is False
        
        # State should remain valid
        final_snapshot = await shared_state.get_snapshot()
        assert final_snapshot["last_sequence"] == 100  # Unchanged
        assert final_snapshot["market_ticker"] == "ERROR-TEST"
        
        # Apply valid delta to confirm recovery
        valid_delta = {
            "market_ticker": "ERROR-TEST",
            "timestamp_ms": int(time.time() * 1000),
            "sequence_number": 101,
            "side": "yes",
            "action": "update",
            "price": 45,
            "old_size": 1000,
            "new_size": 1200
        }
        
        result = await shared_state.apply_delta(valid_delta)
        assert result is True
        
        final_snapshot = await shared_state.get_snapshot()
        assert final_snapshot["last_sequence"] == 101