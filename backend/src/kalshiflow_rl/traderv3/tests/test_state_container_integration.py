"""
Test V3StateContainer integration with coordinator and KalshiDataSync.

This test verifies:
- State container properly stores trading state after sync
- Trading state broadcasting works
- Component health tracking works
- State machine updates are reflected
"""

import asyncio
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def test_state_container_integration():
    """Test the V3StateContainer integration."""
    
    try:
        # Import after logging is configured
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        
        from kalshiflow_rl.traderv3.core.state_container import V3StateContainer
        from kalshiflow_rl.traderv3.state.trader_state import TraderState, StateChange
        from kalshiflow_rl.traderv3.core.state_machine import TraderState as V3State
        
        logger.info("=" * 60)
        logger.info("Testing V3StateContainer Integration")
        logger.info("=" * 60)
        
        # 1. Create state container
        container = V3StateContainer()
        logger.info("✅ Created V3StateContainer")
        
        # 2. Test trading state storage
        logger.info("\n--- Testing Trading State Storage ---")
        
        # Create mock trading state
        mock_state = TraderState(
            balance=100000,  # $1000 in cents
            portfolio_value=50000,  # $500 in cents
            position_count=3,
            order_count=2,
            sync_timestamp=1234567890
        )
        
        # Update with no changes (first update)
        changed = await container.update_trading_state(mock_state)
        assert changed == True, "First state update should return True"
        assert container.trading_state_version == 1, "Version should be 1"
        logger.info(f"✅ Stored initial trading state (version {container.trading_state_version})")

        # Try updating with same state
        changed = await container.update_trading_state(mock_state)
        assert changed == False, "Identical state should return False"
        assert container.trading_state_version == 1, "Version should not change"
        logger.info("✅ Correctly detected no change in state")
        
        # Update with changes
        new_state = TraderState(
            balance=110000,  # +$100
            portfolio_value=55000,  # +$50
            position_count=4,  # +1 position
            order_count=1,  # -1 order
            sync_timestamp=1234567900
        )
        
        changes = StateChange(
            balance_change=10000,
            portfolio_value_change=5000,
            position_count_change=1,
            order_count_change=-1
        )
        
        changed = await container.update_trading_state(new_state, changes)
        assert changed == True, "Changed state should return True"
        assert container.trading_state_version == 2, "Version should be 2"
        assert container.last_state_change == changes, "Changes should be stored"
        logger.info(f"✅ Stored updated trading state with changes (version {container.trading_state_version})")
        
        # Test trading summary
        summary = container.get_trading_summary()
        assert summary["has_state"] == True
        assert summary["version"] == 2
        assert summary["balance"] == 110000
        assert summary["changes"]["balance"] == 10000
        logger.info("✅ Trading summary correctly generated")
        
        # 3. Test component health tracking
        logger.info("\n--- Testing Component Health ---")
        
        container.update_component_health("orderbook", True, {"connected": True})
        container.update_component_health("trading_client", True, {"mode": "paper"})
        container.update_component_health("state_machine", True)
        
        assert container.is_system_healthy() == True
        logger.info("✅ All components healthy")
        
        # Simulate component failure
        container.update_component_health("orderbook", False, error="Connection lost")
        assert container.is_system_healthy() == False
        
        health_summary = container.get_health_summary()
        assert health_summary["system_healthy"] == False
        assert health_summary["unhealthy_count"] == 1
        assert container.get_component_health("orderbook").error_count == 1
        logger.info("✅ Component health failure correctly tracked")
        
        # Recover component
        container.update_component_health("orderbook", True, {"connected": True})
        assert container.is_system_healthy() == True
        assert container.get_component_health("orderbook").error_count == 0
        logger.info("✅ Component recovery correctly tracked")
        
        # 4. Test state machine reference
        logger.info("\n--- Testing State Machine Reference ---")
        
        container.update_machine_state(
            V3State.READY,
            "System ready",
            {"markets": 10}
        )
        
        assert container.machine_state == V3State.READY
        assert container.machine_state_context == "System ready"
        assert container.machine_state_metadata["markets"] == 10
        logger.info("✅ State machine reference stored correctly")
        
        # 5. Test full state snapshot
        logger.info("\n--- Testing Full State Snapshot ---")
        
        full_state = container.get_full_state()
        assert "trading" in full_state
        assert "health" in full_state
        assert "machine" in full_state
        assert "container" in full_state
        
        assert full_state["trading"]["version"] == 2
        assert full_state["health"]["system_healthy"] == True
        assert full_state["machine"]["state"] == "ready"  # State enum value is lowercase
        assert full_state["container"]["trading_version"] == 2
        
        logger.info("✅ Full state snapshot generated correctly")
        
        # 6. Test reset
        logger.info("\n--- Testing Reset ---")
        
        container.reset()
        assert container.trading_state is None
        assert container.trading_state_version == 0
        assert len(container.get_all_component_health()) == 0
        assert container.machine_state is None
        logger.info("✅ Container reset successfully")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return False


def main():
    """Run the integration test."""
    success = asyncio.run(test_state_container_integration())
    exit(0 if success else 1)


if __name__ == "__main__":
    main()