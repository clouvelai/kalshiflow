"""
Integration tests for Trader MVP - Milestones 1 & 2.

Tests the integration between ActorService, LiveObservationAdapter, and OrderbookClient
to ensure the complete pipeline works end-to-end.
"""

import asyncio
import pytest
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from kalshiflow_rl.trading.actor_service import (
    initialize_actor_service,
    get_actor_service,
    shutdown_actor_service
)
from kalshiflow_rl.trading.live_observation_adapter import (
    initialize_live_observation_adapter,
    get_live_observation_adapter,
    build_live_observation
)
from kalshiflow_rl.trading.event_bus import (
    emit_orderbook_snapshot,
    emit_orderbook_delta,
    get_event_bus,
    shutdown_event_bus
)
from kalshiflow_rl.trading.action_selector import create_action_selector_stub


class TestMilestone1Integration:
    """Test integration of Milestone 1 components."""
    
    @pytest.mark.asyncio
    async def test_actor_service_initialization_complete(self):
        """Test complete ActorService initialization with all components."""
        # Initialize actor service
        actor_service = await initialize_actor_service(
            market_tickers=["TEST-MARKET-1", "TEST-MARKET-2"],
            queue_size=100,
            throttle_ms=50
        )
        
        try:
            # Verify initialization
            assert actor_service is not None
            assert actor_service._processing
            assert len(actor_service.market_tickers) == 2
            
            # Test global getter
            global_service = await get_actor_service()
            assert global_service is actor_service
            
            # Test status reporting
            status = actor_service.get_status()
            assert status["status"] == "running"
            assert status["model_available"] is False  # No model in test
            assert len(status["markets"]) == 2
            
        finally:
            await shutdown_actor_service()
            # Don't shutdown event bus if it wasn't started by us (may be shared)
            try:
                event_bus = await get_event_bus()
                if event_bus and event_bus._running:
                    await shutdown_event_bus()
            except Exception:
                pass  # Event bus may already be shut down
    
    @pytest.mark.asyncio
    async def test_orderbook_client_actor_integration(self):
        """Test integration between OrderbookClient and ActorService via triggers."""
        # Ensure EventBus is started (get_event_bus auto-starts it)
        from kalshiflow_rl.trading.event_bus import get_event_bus
        event_bus = await get_event_bus()
        
        # Start EventBus if not running (may have been shut down by previous test)
        if not event_bus._running:
            await event_bus.start()
        
        # Initialize actor service (subscribes to EventBus)
        actor_service = await initialize_actor_service(
            market_tickers=["TEST-MARKET"],
            queue_size=50
        )
        
        try:
            # Test event bus trigger (simulating OrderbookClient call via event bus)
            initial_queue_count = actor_service.metrics.events_queued
            
            success = await emit_orderbook_snapshot(
                market_ticker="TEST-MARKET",
                sequence_number=12345,
                timestamp_ms=int(time.time() * 1000)
            )
            
            assert success, "EventBus should accept snapshot event"
            
            # Wait for event to be processed through event bus
            await asyncio.sleep(0.2)
            assert actor_service.metrics.events_queued > initial_queue_count, "ActorService should have queued the event"
            
            # Test throttling behavior (event bus should accept, but actor service throttles)
            immediate_trigger = await emit_orderbook_delta(
                market_ticker="TEST-MARKET",
                sequence_number=12346,
                timestamp_ms=int(time.time() * 1000)
            )
            
            # Event bus should accept event (throttling happens at actor service level)
            assert immediate_trigger, "EventBus should accept delta event"
            
        finally:
            await shutdown_actor_service()
            # Don't shutdown event bus - other tests may need it
    
    # NOTE: Full pipeline test removed - tests unimplemented order execution
    # This test gives false confidence about non-working order execution
    # According to M1-M2 issues doc: OrderManager integration is incomplete
    # 
    # TODO: Re-enable when order execution is properly connected in M3+
    # @pytest.mark.asyncio
    # async def test_actor_pipeline_with_real_order_execution(self):
    #     """Test complete actor pipeline with real order execution when implemented."""
    #     pass
    
    @pytest.mark.asyncio
    async def test_event_bus_to_action_selector_integration(self):
        """Test event bus integration with real ActionSelector stub - M1-M2 scope."""
        # Test the ActionSelector stub independently (what's actually implemented)
        action_selector_stub = create_action_selector_stub(debug=True)
        
        # Test that ActionSelector stub works correctly
        test_observation = np.random.rand(52)
        action = await action_selector_stub(test_observation, "TEST-MARKET")
        
        # Verify ActionSelector behavior (real functionality)
        assert action == 0  # HOLD action
        assert action_selector_stub.call_count == 1
        assert action_selector_stub.last_called_at is not None
        
        # Check stats (real functionality that works)
        stats = action_selector_stub.get_stats()
        assert stats["is_stub"] is True
        assert stats["call_count"] == 1
        assert stats["stub_action"] == "HOLD"


class TestMilestone2Integration:
    """Test integration of Milestone 2 components."""
    
    @pytest.mark.asyncio
    async def test_live_observation_adapter_initialization(self):
        """Test LiveObservationAdapter initialization and integration."""
        # Initialize adapter
        adapter = await initialize_live_observation_adapter(
            window_size=8,
            max_markets=1,
            temporal_context_minutes=15
        )
        
        # Verify initialization
        assert adapter is not None
        assert adapter.window_size == 8
        assert adapter.max_markets == 1
        assert adapter.temporal_context_minutes == 15
        
        # Test global getter
        global_adapter = await get_live_observation_adapter()
        assert global_adapter is adapter
        
        # Test status reporting
        status = adapter.get_status()
        assert status["adapter"] == "LiveObservationAdapter"
        assert status["window_size"] == 8
        assert status["max_markets"] == 1
    
    # NOTE: Observation building integration test removed - mocks SharedOrderbookState
    # According to M1-M2 issues doc: tests should use real SharedOrderbookState
    # This test gave false confidence about mocked orderbook data
    # 
    # TODO: Re-enable with real SharedOrderbookState integration in M3+
    
    # NOTE: Temporal feature integration test removed - also mocks SharedOrderbookState
    # Same issues as previous test - uses mocked orderbook data
    # 
    # TODO: Re-enable with real SharedOrderbookState integration in M3+

