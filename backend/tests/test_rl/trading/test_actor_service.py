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
from kalshiflow_rl.trading.action_selector import HardcodedSelector


@pytest_asyncio.fixture
async def mock_actor_service():
    """Create a mock actor service using dependency injection for testing."""
    async with create_test_container("actor_test") as container:
        # Create mock dependencies
        mock_event_bus = Mock()
        mock_observation_adapter = Mock()
        mock_observation_adapter.build_observation = AsyncMock(return_value=np.zeros(52))
        mock_order_manager = Mock()
        mock_order_manager.get_positions = Mock(return_value={})
        mock_order_manager.get_portfolio_value = Mock(return_value=10000.0)
        mock_order_manager.get_cash_balance = Mock(return_value=10000.0)
        mock_order_manager.get_order_features = Mock(return_value={})
        
        # Create service with injected dependencies
        service = ActorService(
            market_tickers=["TEST-MARKET-1", "TEST-MARKET-2"],
            model_path=None,  # No model for basic tests
            queue_size=100,
            throttle_ms=50,  # Faster for testing
            event_bus=mock_event_bus,
            observation_adapter=mock_observation_adapter,
            strict_validation=False  # Disable for tests that don't set all dependencies
        )
        
        # Set required dependencies
        service.set_action_selector(HardcodedSelector())
        service.set_order_manager(mock_order_manager)
        
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
            queue_size=50,
            strict_validation=False  # Disable validation for basic init test
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
        """Test circuit breaker for excessive errors - markets disabled, not removed."""
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
        
        # Market should be disabled (in disabled set), not removed from list
        assert len(service.market_tickers) == initial_markets  # Still in list
        assert "TEST-MARKET-1" in service.market_tickers  # Still in list
        assert "TEST-MARKET-1" in service._disabled_markets  # But disabled
        assert not service._should_process_market("TEST-MARKET-1")  # Should not process
    
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


class TestFoundationIssuesFixes:
    """Test cases for foundation issues fixes."""
    
    @pytest.mark.asyncio
    async def test_dependency_validation_strict_mode(self):
        """Test dependency validation fails fast in strict mode."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            strict_validation=True
        )
        
        # Should raise ValueError when initializing without dependencies
        with pytest.raises(ValueError, match="missing required dependencies"):
            await service.initialize()
    
    @pytest.mark.asyncio
    async def test_dependency_validation_lenient_mode(self):
        """Test dependency validation allows missing dependencies in lenient mode."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            strict_validation=False
        )
        
        # Should not raise error
        await service.initialize()
        assert service._processing
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_model_not_loaded_with_stub_selector(self):
        """Test model is not loaded when stub selector is configured."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            model_path="/fake/path/to/model.zip",
            strict_validation=False
        )
        
        # Set stub selector
        service.set_action_selector(HardcodedSelector())
        
        # Mock observation adapter and order manager
        mock_obs = Mock()
        mock_obs.build_observation = AsyncMock(return_value=np.zeros(52))
        service._injected_observation_adapter = mock_obs
        
        mock_om = Mock()
        service.set_order_manager(mock_om)
        
        await service.initialize()
        
        # Model should not be loaded
        assert not service._model_loaded
        assert service._cached_model is None
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_model_loading_warning_with_stub(self):
        """Test warning is logged when model_path provided but stub selector used."""
        import logging
        from io import StringIO
        
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("kalshiflow_rl.actor_service")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        try:
            service = ActorService(
                market_tickers=["TEST-MARKET"],
                model_path="/fake/path/to/model.zip",
                strict_validation=False
            )
            
            service.set_action_selector(HardcodedSelector())
            
            mock_obs = Mock()
            mock_obs.build_observation = AsyncMock(return_value=np.zeros(52))
            service._injected_observation_adapter = mock_obs
            
            mock_om = Mock()
            service.set_order_manager(mock_om)
            
            await service.initialize()
            
            # Check warning was logged
            log_output = log_stream.getvalue()
            assert "Model path provided" in log_output or "stub selector" in log_output.lower()
            
            await service.shutdown()
        finally:
            logger.removeHandler(handler)
    
    @pytest.mark.asyncio
    async def test_portfolio_defaults_from_order_manager(self):
        """Test portfolio defaults come from OrderManager."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            strict_validation=False
        )
        
        # Create mock order manager with initial_cash
        mock_om = Mock()
        mock_om.initial_cash = 5000.0
        mock_om.get_positions = Mock(return_value={})
        mock_om.get_portfolio_value = Mock(return_value=5000.0)
        mock_om.get_cash_balance = Mock(return_value=5000.0)
        service.set_order_manager(mock_om)
        
        # Get defaults
        portfolio_value, cash_balance = service._get_default_portfolio_values()
        
        assert portfolio_value == 5000.0
        assert cash_balance == 5000.0
    
    @pytest.mark.asyncio
    async def test_portfolio_defaults_fallback(self):
        """Test portfolio defaults fall back to config if OrderManager unavailable."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            strict_validation=False
        )
        
        # No order manager set
        portfolio_value, cash_balance = service._get_default_portfolio_values()
        
        # Should use default (10000.0 or config value)
        assert portfolio_value > 0
        assert cash_balance > 0
    
    @pytest.mark.asyncio
    async def test_injected_adapter_preferred(self):
        """Test injected adapter is preferred over callback adapter."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            strict_validation=False
        )
        
        # Create both adapters
        mock_injected = Mock()
        mock_injected.build_observation = AsyncMock(return_value=np.array([1.0, 2.0, 3.0]))
        service._injected_observation_adapter = mock_injected
        
        mock_callback = AsyncMock(return_value=np.array([4.0, 5.0, 6.0]))
        service.set_observation_adapter(mock_callback)
        
        service.set_action_selector(HardcodedSelector())
        mock_om = Mock()
        mock_om.get_positions = Mock(return_value={})
        mock_om.get_portfolio_value = Mock(return_value=10000.0)
        mock_om.get_cash_balance = Mock(return_value=10000.0)
        service.set_order_manager(mock_om)
        
        await service.initialize()
        
        # Trigger event
        await service.trigger_event("TEST-MARKET", "snapshot", 1, int(time.time() * 1000))
        await asyncio.sleep(0.1)
        
        # Injected adapter should be called, callback should not
        assert mock_injected.build_observation.called
        assert not mock_callback.called
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_callback_adapter_deprecation_warning(self):
        """Test deprecation warning when callback adapter is used."""
        import logging
        from io import StringIO
        
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        logger = logging.getLogger("kalshiflow_rl.actor_service")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        try:
            service = ActorService(
                market_tickers=["TEST-MARKET"],
                strict_validation=False
            )
            
            # Only callback adapter, no injected
            mock_callback = AsyncMock(return_value=np.zeros(52))
            service.set_observation_adapter(mock_callback)
            
            service.set_action_selector(HardcodedSelector())
            mock_om = Mock()
            mock_om.get_positions = Mock(return_value={})
            mock_om.get_portfolio_value = Mock(return_value=10000.0)
            mock_om.get_cash_balance = Mock(return_value=10000.0)
            service.set_order_manager(mock_om)
            
            await service.initialize()
            
            # Trigger event
            await service.trigger_event("TEST-MARKET", "snapshot", 1, int(time.time() * 1000))
            await asyncio.sleep(0.1)
            
            # Check deprecation warning was logged
            log_output = log_stream.getvalue()
            assert "deprecated" in log_output.lower() or "callback" in log_output.lower()
            
            await service.shutdown()
        finally:
            logger.removeHandler(handler)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_disables_not_removes(self):
        """Test circuit breaker disables markets but doesn't remove from list."""
        service = ActorService(
            market_tickers=["TEST-MARKET-1", "TEST-MARKET-2"],
            strict_validation=False
        )
        
        mock_obs = Mock()
        mock_obs.build_observation = AsyncMock(return_value=np.zeros(52))
        service._injected_observation_adapter = mock_obs
        
        service.set_action_selector(HardcodedSelector())
        mock_om = Mock()
        mock_om.get_positions = Mock(return_value={})
        mock_om.get_portfolio_value = Mock(return_value=10000.0)
        mock_om.get_cash_balance = Mock(return_value=10000.0)
        service.set_order_manager(mock_om)
        
        service._max_errors_per_market = 2
        
        await service.initialize()
        
        # Cause errors
        mock_obs.build_observation = AsyncMock(side_effect=Exception("Error"))
        
        for i in range(3):
            await service.trigger_event("TEST-MARKET-1", "snapshot", i, int(time.time() * 1000))
            await asyncio.sleep(0.1)
        
        # Market should be disabled but still in list
        assert "TEST-MARKET-1" in service.market_tickers
        assert "TEST-MARKET-1" in service._disabled_markets
        assert not service._should_process_market("TEST-MARKET-1")
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_auto_re_enable(self):
        """Test circuit breaker auto re-enables markets after delay."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            strict_validation=False
        )
        
        mock_obs = Mock()
        mock_obs.build_observation = AsyncMock(return_value=np.zeros(52))
        service._injected_observation_adapter = mock_obs
        
        service.set_action_selector(HardcodedSelector())
        mock_om = Mock()
        mock_om.get_positions = Mock(return_value={})
        mock_om.get_portfolio_value = Mock(return_value=10000.0)
        mock_om.get_cash_balance = Mock(return_value=10000.0)
        service.set_order_manager(mock_om)
        
        # Set short re-enable delay for testing
        service._market_re_enable_delay_seconds = 0.1
        
        await service.initialize()
        
        # Disable market manually
        service._disabled_markets["TEST-MARKET"] = time.time()
        assert not service._should_process_market("TEST-MARKET")
        
        # Wait for re-enable delay
        await asyncio.sleep(0.15)
        
        # Market should be auto re-enabled
        assert service._should_process_market("TEST-MARKET")
        assert "TEST-MARKET" not in service._disabled_markets
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_manual_re_enable_market(self):
        """Test manual re-enable of disabled market."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            strict_validation=False
        )
        
        # Manually disable market
        service._disabled_markets["TEST-MARKET"] = time.time()
        assert not service._should_process_market("TEST-MARKET")
        
        # Re-enable
        result = service.re_enable_market("TEST-MARKET")
        assert result is True
        assert "TEST-MARKET" not in service._disabled_markets
        assert service._should_process_market("TEST-MARKET")
        
        # Re-enable non-disabled market
        result = service.re_enable_market("TEST-MARKET")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_position_read_delay(self):
        """Test position read delay accounts for async fill processing."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            position_read_delay_ms=50,  # 50ms delay
            strict_validation=False
        )
        
        mock_obs = Mock()
        mock_obs.build_observation = AsyncMock(return_value=np.zeros(52))
        service._injected_observation_adapter = mock_obs
        
        service.set_action_selector(HardcodedSelector())
        
        # Mock order manager with position tracking
        mock_om = Mock()
        mock_om.get_positions = Mock(return_value={"TEST-MARKET": {"position": 10}})
        mock_om.get_portfolio_value = Mock(return_value=10000.0)
        mock_om.get_cash_balance = Mock(return_value=10000.0)
        service.set_order_manager(mock_om)
        
        await service.initialize()
        
        # Track time
        start_time = time.time()
        
        # Trigger event with execution result
        await service.trigger_event("TEST-MARKET", "snapshot", 1, int(time.time() * 1000))
        
        # Wait for processing (including position update delay)
        await asyncio.sleep(0.2)
        
        # Position update should have been called after delay
        # (We can't easily test exact timing, but we can verify it was called)
        assert mock_om.get_positions.called
        
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_stub_selector_detection(self):
        """Test stub selector detection works correctly."""
        service = ActorService(
            market_tickers=["TEST-MARKET"],
            strict_validation=False
        )
        
        # Test with stub function
        service.set_action_selector(HardcodedSelector())
        assert service._is_stub_selector() is True
        
        # Test with HardcodedSelector instance
        hardcoded_instance = HardcodedSelector()
        service.set_action_selector(hardcoded_instance)
        assert service._is_stub_selector() is True
        
        # Test with non-stub function
        async def real_selector(obs, market):
            return 1
        
        service.set_action_selector(real_selector)
        # May or may not detect as stub depending on heuristics, but should not crash
        service._is_stub_selector()  # Should not raise
        
        # Test with no selector
        service._action_selector = None
        assert service._is_stub_selector() is False