"""
Tests for ActorService - Core actor service with model caching and queue processing.
"""

import asyncio
import pytest
import pytest_asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from kalshiflow_rl.trading.actor_service import (
    ActorService,
    ActorEvent,
    ActorMetrics
)
from kalshiflow_rl.trading.service_container import create_test_container


@pytest_asyncio.fixture
async def mock_actor_service():
    """Create a mock actor service using dependency injection for testing."""
    async with create_test_container("actor_test") as container:
        # Create mock dependencies
        mock_event_bus = Mock()
        mock_observation_adapter = Mock()
        mock_observation_adapter.build_observation = AsyncMock(return_value=np.zeros(52))
        
        # Create service with injected dependencies
        service = ActorService(
            market_tickers=["TEST-MARKET-1", "TEST-MARKET-2"],
            model_path=None,  # No model for basic tests
            queue_size=100,
            throttle_ms=50,  # Faster for testing
            event_bus=mock_event_bus,
            observation_adapter=mock_observation_adapter
        )
        await service.initialize()
        
        yield service
        
        await service.shutdown()


@pytest.fixture
def sample_actor_event():
    """Create a sample actor event for testing."""
    return ActorEvent(
        market_ticker="TEST-MARKET-1",
        update_type="snapshot",
        sequence_number=12345,
        timestamp_ms=int(time.time() * 1000),
        received_at=time.time()
    )


class TestActorService:
    """Test cases for ActorService functionality."""
    
    @pytest.mark.asyncio
    async def test_actor_service_initialization(self):
        """Test actor service initializes correctly."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            queue_size=50
        )
        
        assert service.market_tickers == ["TEST-MARKET"]
        assert service._event_queue.maxsize == 50
        assert not service._processing
        assert service.metrics.events_queued == 0
        
        await service.initialize()
        
        assert service._processing
        assert service.metrics.started_at is not None
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_event_queuing(self, mock_actor_service):
        """Test event queuing mechanism."""
        service = mock_actor_service
        
        # Test successful event queuing
        success = await service.trigger_event(
            market_ticker="TEST-MARKET-1",
            update_type="snapshot",
            sequence_number=100,
            timestamp_ms=int(time.time() * 1000)
        )
        
        assert success
        assert service.metrics.events_queued == 1
        assert service.metrics.queue_depth == 1
        
        # Test that events can be queued immediately (throttling happens at execution time)
        immediate_success = await service.trigger_event(
            market_ticker="TEST-MARKET-1",
            update_type="delta",
            sequence_number=101,
            timestamp_ms=int(time.time() * 1000)
        )
        
        # Events can be queued immediately - throttling happens at execution time
        assert immediate_success
        assert service.metrics.events_queued == 2  # Both events queued
    
    @pytest.mark.asyncio
    async def test_throttling_behavior(self, mock_actor_service):
        """Test per-market throttling behavior at execution time."""
        service = mock_actor_service
        
        market1 = "TEST-MARKET-1"
        market2 = "TEST-MARKET-2"
        
        # Events can be queued immediately - throttling happens at execution time
        success1 = await service.trigger_event(market1, "snapshot", 100, int(time.time() * 1000))
        assert success1
        
        # Immediate second event for same market can also be queued
        success2 = await service.trigger_event(market1, "delta", 101, int(time.time() * 1000))
        assert success2
        
        # Different market can also be queued
        success3 = await service.trigger_event(market2, "snapshot", 102, int(time.time() * 1000))
        assert success3
        
        # Note: Actual throttling happens during execution in _safe_execute_action()
        # This test verifies that events can be queued without throttling at queue time
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_actor_service):
        """Test error handling in event processing."""
        service = mock_actor_service
        
        # Mock injected observation adapter that raises an error
        service._injected_observation_adapter.build_observation = AsyncMock(side_effect=Exception("Test error"))
        
        # Queue an event
        await service.trigger_event(
            market_ticker="TEST-MARKET-1",
            update_type="snapshot",
            sequence_number=300,
            timestamp_ms=int(time.time() * 1000)
        )
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Verify error was handled
        assert service.metrics.errors >= 1
        assert service.metrics.last_error is not None
        assert "TEST-MARKET-1" in service._error_counts
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, mock_actor_service):
        """Test circuit breaker for excessive errors."""
        service = mock_actor_service
        service._max_errors_per_market = 2  # Lower threshold for testing
        
        # Mock injected adapter that always fails
        service._injected_observation_adapter.build_observation = AsyncMock(side_effect=Exception("Persistent error"))
        
        initial_markets = len(service.market_tickers)
        
        # Trigger multiple errors for the same market
        for i in range(5):
            await service.trigger_event(
                market_ticker="TEST-MARKET-1",
                update_type="snapshot",
                sequence_number=400 + i,
                timestamp_ms=int(time.time() * 1000)
            )
            await asyncio.sleep(0.06)  # Wait for throttle
        
        # Wait for all processing
        await asyncio.sleep(0.5)
        
        # Market should be removed due to circuit breaker
        assert len(service.market_tickers) < initial_markets
        assert "TEST-MARKET-1" not in service.market_tickers
    
    @pytest.mark.asyncio  
    async def test_metrics_structure_only(self, mock_actor_service):
        """Test metrics structure - not processing performance (unimplemented)."""
        service = mock_actor_service
        
        # Test that metrics structure exists and has expected keys
        metrics = service.get_metrics()
        
        # Verify metrics structure (what's actually implemented)
        expected_keys = [
            "events_queued", "events_processed", "model_predictions", 
            "orders_executed", "avg_processing_time_ms", "model_loaded",
            "active_markets", "processing"
        ]
        
        for key in expected_keys:
            assert key in metrics
            # All metrics should be reasonable types
            assert isinstance(metrics[key], (int, float, bool))
        
        # Basic sanity checks
        assert metrics["active_markets"] == 2  # TEST-MARKET-1 and TEST-MARKET-2
        assert metrics["model_loaded"] is False  # No model loaded in M1-M2
        assert metrics["processing"] is True  # Service is running
    


class TestActorMetrics:
    """Test cases for actor metrics and status reporting."""
    
    def test_actor_metrics_initialization(self):
        """Test ActorMetrics dataclass initialization."""
        metrics = ActorMetrics()
        
        assert metrics.events_queued == 0
        assert metrics.events_processed == 0
        assert metrics.model_predictions == 0
        assert metrics.orders_executed == 0
        assert metrics.total_processing_time == 0.0
        assert metrics.avg_processing_time == 0.0
        assert metrics.queue_depth == 0
        assert metrics.max_queue_depth == 0
        assert metrics.errors == 0
        assert metrics.last_error is None
        assert metrics.started_at is None
        assert metrics.last_processed_at is None
    
    @pytest.mark.asyncio
    async def test_status_reporting(self, mock_actor_service):
        """Test status reporting functionality."""
        service = mock_actor_service
        
        status = service.get_status()
        
        assert status["service"] == "ActorService"
        assert status["status"] == "running"
        assert "TEST-MARKET-1" in status["markets"]
        assert "TEST-MARKET-2" in status["markets"]
        assert status["model_available"] is False
        assert "metrics" in status
        assert "performance" in status
        
        # Test metrics structure
        metrics = status["metrics"]
        assert "events_queued" in metrics
        assert "events_processed" in metrics
        assert "avg_processing_time_ms" in metrics
        assert "model_loaded" in metrics
        assert "active_markets" in metrics
        assert "processing" in metrics
        
        # Test performance structure
        performance = status["performance"]
        assert "avg_processing_time_ms" in performance
        assert "total_events" in performance
        assert "error_rate" in performance
        assert "throughput_events_per_sec" in performance