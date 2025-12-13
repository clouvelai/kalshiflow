"""
Tests for ActionSelector Stub - M1-M2 Integration Component.

Tests the minimal ActionSelector stub that enables M1-M2 pipeline integration.
This stub always returns HOLD action for safe testing of ActorService integration.
"""

import pytest
import pytest_asyncio
import numpy as np
import time

from kalshiflow_rl.trading.action_selector import (
    ActionSelectorStub,
    create_action_selector_stub,
    select_action_stub
)
from kalshiflow_rl.environments.limit_order_action_space import LimitOrderActions


class TestActionSelectorStub:
    """Test cases for ActionSelector stub functionality."""
    
    def test_stub_initialization(self):
        """Test ActionSelector stub initializes correctly."""
        stub = ActionSelectorStub(debug=True)
        
        assert stub.debug is True
        assert stub.call_count == 0
        assert stub.last_called_at is None
        
        stats = stub.get_stats()
        assert stats["is_stub"] is True
        assert stats["stub_action"] == "HOLD"
        assert stats["action_value"] == LimitOrderActions.HOLD.value
        assert stats["call_count"] == 0
    
    @pytest.mark.asyncio
    async def test_stub_always_returns_hold(self):
        """Test stub always returns HOLD action regardless of input."""
        stub = ActionSelectorStub(debug=False)
        
        # Test with various observation shapes and market tickers
        test_cases = [
            (np.zeros(52), "EXAMPLE-MARKET-1"),
            (np.random.random(100), "TEST-MARKET-ABC"),
            (np.ones(10), "ANOTHER-MARKET"),
            (np.array([0.5, 0.3, 0.7]), "FINAL-TEST")
        ]
        
        for observation, market_ticker in test_cases:
            action = await stub(observation, market_ticker)
            
            # Always returns HOLD action
            assert action == LimitOrderActions.HOLD.value
            assert action == 0
    
    @pytest.mark.asyncio
    async def test_stub_call_tracking(self):
        """Test stub tracks calls correctly."""
        stub = ActionSelectorStub(debug=True)
        
        observation = np.zeros(52)
        
        # First call
        before_time = time.time()
        action1 = await stub(observation, "TEST-MARKET-1")
        after_time = time.time()
        
        assert action1 == LimitOrderActions.HOLD.value
        assert stub.call_count == 1
        assert stub.last_called_at is not None
        assert before_time <= stub.last_called_at <= after_time
        
        # Second call
        action2 = await stub(observation, "TEST-MARKET-2")
        
        assert action2 == LimitOrderActions.HOLD.value
        assert stub.call_count == 2
        
        # Third call
        action3 = await stub(observation, "TEST-MARKET-1") 
        
        assert action3 == LimitOrderActions.HOLD.value
        assert stub.call_count == 3
        
        # Verify stats
        stats = stub.get_stats()
        assert stats["call_count"] == 3
        assert stats["last_called_at"] == stub.last_called_at
    
    @pytest.mark.asyncio
    async def test_factory_function(self):
        """Test factory function creates correct stub."""
        stub = create_action_selector_stub(debug=False)
        
        assert isinstance(stub, ActionSelectorStub)
        assert stub.debug is False
        
        # Test it works
        action = await stub(np.zeros(10), "FACTORY-TEST")
        assert action == LimitOrderActions.HOLD.value
        assert stub.call_count == 1
    
    @pytest.mark.asyncio
    async def test_function_wrapper(self):
        """Test direct function wrapper for ActorService integration."""
        # Test the simple function wrapper
        action = await select_action_stub(np.zeros(52), "FUNCTION-TEST")
        
        assert action == LimitOrderActions.HOLD.value
        assert action == 0
        
        # Should work multiple times
        action2 = await select_action_stub(np.ones(100), "ANOTHER-FUNCTION-TEST")
        assert action2 == LimitOrderActions.HOLD.value


class TestActionSelectorIntegration:
    """Test ActionSelector stub integration patterns."""
    
    @pytest.mark.asyncio
    async def test_actor_service_integration_pattern(self):
        """Test integration pattern expected by ActorService."""
        stub = create_action_selector_stub(debug=False)
        
        # Simulate ActorService call pattern
        observation = np.random.random(52)  # Typical observation size
        market_ticker = "INTEGRATION-TEST"
        
        # Call as if from ActorService._select_action()
        action = await stub(observation, market_ticker)
        
        # Verify correct type and value
        assert isinstance(action, int)
        assert action == 0  # HOLD
        assert action == LimitOrderActions.HOLD.value
        
        # Verify no exceptions with edge cases
        edge_cases = [
            np.array([]),  # Empty array
            np.zeros(1),   # Single value
            np.full(1000, 0.5),  # Large array
        ]
        
        for edge_obs in edge_cases:
            edge_action = await stub(edge_obs, "EDGE-CASE")
            assert edge_action == LimitOrderActions.HOLD.value
    
    @pytest.mark.asyncio
    async def test_concurrent_calls(self):
        """Test stub handles concurrent calls correctly."""
        import asyncio
        
        stub = ActionSelectorStub(debug=False)
        
        # Create multiple concurrent calls
        async def make_call(i):
            obs = np.zeros(52)
            return await stub(obs, f"CONCURRENT-{i}")
        
        # Run 10 concurrent calls
        tasks = [make_call(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should return HOLD
        assert all(action == LimitOrderActions.HOLD.value for action in results)
        assert len(results) == 10
        
        # Call count should be accurate
        assert stub.call_count == 10


class TestActionSelectorDebugLogging:
    """Test debug logging functionality."""
    
    @pytest.mark.asyncio
    async def test_debug_logging_enabled(self, caplog):
        """Test debug logging when enabled."""
        import logging
        
        # Ensure debug level is captured
        caplog.set_level(logging.DEBUG)
        
        stub = ActionSelectorStub(debug=True)
        
        observation = np.zeros(52)
        await stub(observation, "DEBUG-TEST")
        
        # Should have debug log entries
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        assert len(debug_logs) > 0
        
        # Check log content
        log_messages = [record.message for record in debug_logs]
        debug_message = next((msg for msg in log_messages if "ActionSelector STUB called" in msg), None)
        assert debug_message is not None
        assert "DEBUG-TEST" in debug_message
        assert "HOLD" in debug_message
    
    @pytest.mark.asyncio
    async def test_debug_logging_disabled(self, caplog):
        """Test no debug logging when disabled."""
        import logging
        
        caplog.set_level(logging.DEBUG)
        
        stub = ActionSelectorStub(debug=False)
        
        observation = np.zeros(52)
        await stub(observation, "NO-DEBUG-TEST")
        
        # Should not have ActionSelector debug logs (may have other debug logs)
        debug_logs = [record for record in caplog.records if record.levelname == "DEBUG"]
        stub_debug_logs = [log for log in debug_logs if "ActionSelector STUB called" in log.message]
        assert len(stub_debug_logs) == 0


# ===============================================================================
# M1-M2 Integration Test Notes
# ===============================================================================
#
# These tests validate the ActionSelector stub provides:
#
# 1. ✅ Correct async interface for ActorService integration
# 2. ✅ Always returns safe HOLD action (LimitOrderActions.HOLD = 0)
# 3. ✅ Proper call tracking for debugging M1-M2 pipeline
# 4. ✅ Factory functions for easy integration
# 5. ✅ Concurrent call handling for async environment
# 6. ✅ Debug logging control for development vs production
#
# The stub enables ActorService pipeline testing without requiring:
# - Full RL model implementation (M3 scope)
# - Complex observation processing (M3 scope)  
# - Real trading strategy logic (M3 scope)
#
# This satisfies M1-M2 requirements for end-to-end pipeline validation.