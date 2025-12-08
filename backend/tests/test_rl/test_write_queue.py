"""
Unit tests for OrderbookWriteQueue.

Tests non-blocking behavior, batching, sampling, and performance
under various load conditions.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from kalshiflow_rl.data.write_queue import OrderbookWriteQueue, MessageType


@pytest_asyncio.fixture(scope="function")
async def write_queue():
    """Create a test write queue with small configuration."""
    queue = OrderbookWriteQueue(
        batch_size=5,
        flush_interval=0.1,
        delta_sample_rate=2,  # Keep 1 out of 2 deltas
        max_queue_size=100
    )
    yield queue
    if queue._running:
        await queue.stop()


@pytest.fixture
def sample_snapshot():
    """Sample snapshot data for testing."""
    return {
        "market_ticker": "TEST-MARKET",
        "timestamp_ms": int(time.time() * 1000),
        "sequence_number": 100,
        "yes_bids": {"45": 1000, "44": 500},
        "yes_asks": {"55": 800, "56": 400},
        "no_bids": {"45": 600, "44": 300},
        "no_asks": {"55": 700, "56": 200}
    }


@pytest.fixture
def sample_delta():
    """Sample delta data for testing."""
    return {
        "market_ticker": "TEST-MARKET",
        "timestamp_ms": int(time.time() * 1000),
        "sequence_number": 101,
        "side": "yes",
        "action": "update",
        "price": 45,
        "old_size": 1000,
        "new_size": 1200
    }


class TestOrderbookWriteQueue:
    """Test cases for OrderbookWriteQueue."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test queue initialization with default config."""
        queue = OrderbookWriteQueue()
        
        assert queue.batch_size > 0
        assert queue.flush_interval > 0
        assert queue.delta_sample_rate >= 1
        assert queue.max_queue_size > 0
        assert not queue._running
    
    @pytest.mark.asyncio
    async def test_start_stop(self, write_queue):
        """Test queue start and stop operations."""
        assert not write_queue._running
        
        await write_queue.start()
        assert write_queue._running
        assert write_queue._flush_task is not None
        
        await write_queue.stop()
        assert not write_queue._running
    
    @pytest.mark.asyncio
    async def test_enqueue_snapshot_non_blocking(self, write_queue, sample_snapshot):
        """Test that snapshot enqueue is non-blocking."""
        await write_queue.start()
        
        # Measure enqueue time
        start_time = time.time()
        result = await write_queue.enqueue_snapshot(sample_snapshot)
        enqueue_time = time.time() - start_time
        
        assert result is True
        assert enqueue_time < 0.001  # Should be sub-millisecond
        assert write_queue._messages_enqueued == 1
        assert write_queue._snapshot_queue.qsize() == 1
    
    @pytest.mark.asyncio
    async def test_enqueue_delta_non_blocking(self, write_queue, sample_delta):
        """Test that delta enqueue is non-blocking."""
        await write_queue.start()
        
        # Measure enqueue time
        start_time = time.time()
        result = await write_queue.enqueue_delta(sample_delta)
        enqueue_time = time.time() - start_time
        
        assert result is True
        assert enqueue_time < 0.001  # Should be sub-millisecond
        # First delta with sample_rate=2 gets sampled out, second delta gets queued
        first_result = await write_queue.enqueue_delta(sample_delta)
        assert first_result is True
        # With sample_rate=2, only every 2nd delta is queued
        assert write_queue._messages_enqueued == 1
    
    @pytest.mark.asyncio
    async def test_delta_sampling(self, write_queue, sample_delta):
        """Test delta sampling reduces message volume."""
        await write_queue.start()
        
        # Enqueue multiple deltas
        for i in range(10):
            delta = sample_delta.copy()
            delta["sequence_number"] = 100 + i
            result = await write_queue.enqueue_delta(delta)
            assert result is True  # All calls return True (success)
        
        # With sample_rate=2, should have 5 messages enqueued and 5 sampled
        assert write_queue._messages_enqueued == 5  # Every 2nd message is queued
        assert write_queue._deltas_sampled == 5     # Every other message is sampled out
        assert write_queue._messages_enqueued + write_queue._deltas_sampled == 10
    
    @pytest.mark.asyncio
    @patch('kalshiflow_rl.data.write_queue.rl_db')
    async def test_batch_processing(self, mock_db, write_queue, sample_snapshot):
        """Test batching functionality."""
        mock_db.batch_insert_snapshots = AsyncMock(return_value=5)
        await write_queue.start()
        
        # Enqueue exactly batch_size messages
        for i in range(write_queue.batch_size):
            snapshot = sample_snapshot.copy()
            snapshot["sequence_number"] = 100 + i
            await write_queue.enqueue_snapshot(snapshot)
        
        # Wait for flush interval
        await asyncio.sleep(write_queue.flush_interval + 0.05)
        
        # Should have called batch insert with all messages
        mock_db.batch_insert_snapshots.assert_called_once()
        call_args = mock_db.batch_insert_snapshots.call_args[0][0]
        assert len(call_args) == write_queue.batch_size
    
    @pytest.mark.asyncio
    async def test_queue_backpressure(self, sample_snapshot):
        """Test queue backpressure handling."""
        # Create queue with very small size
        queue = OrderbookWriteQueue(max_queue_size=5)
        await queue.start()
        
        try:
            # Fill up the queue
            success_count = 0
            for i in range(20):
                snapshot = sample_snapshot.copy()
                snapshot["sequence_number"] = 100 + i
                result = await queue.enqueue_snapshot(snapshot)
                if result:
                    success_count += 1
            
            # Should have backpressure after queue fills up
            assert success_count < 20
            assert queue._queue_full_errors > 0
            
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    @patch('kalshiflow_rl.data.write_queue.rl_db')
    async def test_graceful_shutdown_preserves_messages(self, mock_db, sample_snapshot):
        """Test that shutdown flushes remaining messages."""
        mock_db.batch_insert_snapshots = AsyncMock(return_value=3)
        
        queue = OrderbookWriteQueue(flush_interval=10.0)  # Long flush interval
        await queue.start()
        
        # Enqueue some messages
        for i in range(3):
            snapshot = sample_snapshot.copy()
            snapshot["sequence_number"] = 100 + i
            await queue.enqueue_snapshot(snapshot)
        
        # Stop queue (should flush remaining messages)
        await queue.stop()
        
        # Should have flushed the remaining messages
        mock_db.batch_insert_snapshots.assert_called_once()
        call_args = mock_db.batch_insert_snapshots.call_args[0][0]
        assert len(call_args) == 3
    
    @pytest.mark.asyncio
    async def test_performance_high_load(self, sample_delta):
        """Test performance under high message load."""
        queue = OrderbookWriteQueue(
            batch_size=100,
            flush_interval=0.1,
            delta_sample_rate=1,  # No sampling
            max_queue_size=10000
        )
        await queue.start()
        
        try:
            # Generate high load
            num_messages = 1000
            start_time = time.time()
            
            for i in range(num_messages):
                delta = sample_delta.copy()
                delta["sequence_number"] = i
                await queue.enqueue_delta(delta)
            
            enqueue_time = time.time() - start_time
            
            # Should handle 1000 messages in reasonable time
            assert enqueue_time < 1.0  # Less than 1 second
            
            # Check that messages were enqueued
            assert queue._messages_enqueued == num_messages
            
            # Should still be responsive to new messages
            extra_delta = sample_delta.copy()
            extra_delta["sequence_number"] = num_messages + 1
            result = await queue.enqueue_delta(extra_delta)
            assert result is True
            
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_health_check(self, write_queue, sample_snapshot):
        """Test health checking functionality."""
        await write_queue.start()
        
        # Should be healthy when running
        assert write_queue.is_healthy() is True
        
        # Add some messages
        await write_queue.enqueue_snapshot(sample_snapshot)
        assert write_queue.is_healthy() is True
        
        await write_queue.stop()
        
        # Should be unhealthy when stopped
        assert write_queue.is_healthy() is False
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, write_queue, sample_snapshot, sample_delta):
        """Test statistics collection."""
        await write_queue.start()
        
        # Initial stats
        stats = write_queue.get_stats()
        assert stats["messages_enqueued"] == 0
        assert stats["running"] is True
        
        # Enqueue some messages
        await write_queue.enqueue_snapshot(sample_snapshot)
        await write_queue.enqueue_delta(sample_delta)
        
        stats = write_queue.get_stats()
        assert stats["messages_enqueued"] >= 1  # At least snapshot (delta might be sampled)
        assert "config" in stats
        assert "queue_full_errors" in stats
    
    @pytest.mark.asyncio
    @patch('kalshiflow_rl.data.write_queue.rl_db')
    async def test_database_error_handling(self, mock_db, write_queue, sample_snapshot):
        """Test handling of database errors during flush."""
        # Mock database to raise an error
        mock_db.batch_insert_snapshots = AsyncMock(side_effect=Exception("Database error"))
        
        await write_queue.start()
        
        # Enqueue a message
        await write_queue.enqueue_snapshot(sample_snapshot)
        
        # Wait for flush attempt
        await asyncio.sleep(write_queue.flush_interval + 0.05)
        
        # Should have attempted database write despite error
        mock_db.batch_insert_snapshots.assert_called_once()
        
        # Queue should still be running and healthy
        assert write_queue._running is True