"""
Test Cash Recovery State Machine - TRADER 2.0

Simple test to validate the cash recovery flow:
IDLE → CALIBRATING → LOW_CASH → RECOVER_CASH → CALIBRATING → READY
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

from trading.state_machine import TraderStateMachine, TraderState
from trading.coordinator import TraderCoordinator
from trading.services.position_tracker import PositionTracker, Position, FillInfo, OrderSide, ContractSide
from trading.services.status_logger import StatusLogger

# Configure logging for testing
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockKalshiClient:
    """Mock Kalshi client for testing."""
    
    def __init__(self):
        self.account_balance = 50.0  # Low balance to trigger cash recovery
        
    async def get_account_info(self):
        """Mock account info with low balance."""
        return {
            "user_id": "test_user",
            "balance": self.account_balance * 100,  # Return in cents
            "buying_power": self.account_balance
        }
    
    async def get_positions(self):
        """Mock positions."""
        return []
    
    async def get_orders(self):
        """Mock orders."""
        return []


async def test_cash_recovery_flow():
    """
    Test the complete cash recovery state machine flow.
    """
    logger.info("=== Starting Cash Recovery State Machine Test ===")
    
    # Create mock client
    mock_client = MockKalshiClient()
    
    # Create coordinator with low cash threshold
    coordinator = TraderCoordinator(
        client=mock_client,
        initial_cash_balance=50.0,  # Start with low cash
        minimum_cash_threshold=100.0  # Require $100 minimum
    )
    
    # Add some test positions to enable recovery
    test_positions = {
        "MARKET1": Position(
            ticker="MARKET1",
            contracts=10,  # Long YES position
            cost_basis=30.0,
            realized_pnl=0.0
        ),
        "MARKET2": Position(
            ticker="MARKET2", 
            contracts=-5,  # Long NO position
            cost_basis=20.0,
            realized_pnl=0.0
        )
    }
    coordinator.position_tracker.positions = test_positions
    
    logger.info(f"Initial state: {coordinator.state_machine.current_state.value}")
    logger.info(f"Cash balance: ${coordinator.position_tracker.cash_balance:.2f}")
    logger.info(f"Minimum threshold: ${coordinator.minimum_cash_threshold:.2f}")
    logger.info(f"Active positions: {len(coordinator.position_tracker.get_active_positions())}")
    
    # Test 1: Check cash threshold
    logger.info("\n--- Test 1: Cash Threshold Check ---")
    cash_status = await coordinator._check_cash_threshold()
    logger.info(f"Cash sufficient: {cash_status['sufficient']}")
    logger.info(f"Cash deficit: ${cash_status['deficit']:.2f}")
    logger.info(f"Positions available: {cash_status['positions_available']}")
    
    assert not cash_status["sufficient"], "Cash should be insufficient"
    assert cash_status["deficit"] > 0, "Should have cash deficit"
    assert cash_status["positions_available"], "Should have positions for liquidation"
    
    # Test 2: State machine transitions
    logger.info("\n--- Test 2: State Machine Transitions ---")
    
    # Start with IDLE state
    assert coordinator.state_machine.current_state == TraderState.IDLE
    
    # Test transition to LOW_CASH
    success = await coordinator.state_machine.trigger_cash_recovery("test_low_cash")
    assert not success, "Cannot trigger cash recovery from IDLE state"
    
    # Move to CALIBRATING first
    await coordinator.state_machine.transition_to(TraderState.CALIBRATING, "test")
    
    # Now trigger cash recovery
    success = await coordinator.state_machine.trigger_cash_recovery("test_low_cash")
    assert success, "Should be able to trigger cash recovery from CALIBRATING"
    assert coordinator.state_machine.current_state == TraderState.LOW_CASH
    
    logger.info(f"State after cash recovery trigger: {coordinator.state_machine.current_state.value}")
    
    # Test transition to RECOVER_CASH
    success = await coordinator.state_machine.start_position_liquidation()
    assert success, "Should be able to start liquidation from LOW_CASH"
    assert coordinator.state_machine.current_state == TraderState.RECOVER_CASH
    
    logger.info(f"State after liquidation start: {coordinator.state_machine.current_state.value}")
    
    # Test completion of cash recovery
    success = await coordinator.state_machine.complete_cash_recovery()
    assert success, "Should be able to complete cash recovery"
    assert coordinator.state_machine.current_state == TraderState.CALIBRATING
    
    logger.info(f"State after recovery complete: {coordinator.state_machine.current_state.value}")
    
    # Test 3: Bulk close functionality
    logger.info("\n--- Test 3: Bulk Close Positions ---")
    
    # Reset positions for bulk close test
    coordinator.position_tracker.positions = test_positions.copy()
    
    initial_positions = len(coordinator.position_tracker.get_active_positions())
    logger.info(f"Positions before bulk close: {initial_positions}")
    
    bulk_result = await coordinator.bulk_close_all_positions("test_recovery")
    
    logger.info(f"Bulk close result: {bulk_result['success']}")
    logger.info(f"Positions attempted: {bulk_result['positions_attempted']}")
    logger.info(f"Positions closed: {bulk_result['positions_closed']}")
    
    assert bulk_result["success"], "Bulk close should succeed"
    assert bulk_result["positions_attempted"] == initial_positions, "Should attempt to close all positions"
    
    # Test 4: Complete calibration flow with cash recovery
    logger.info("\n--- Test 4: Full Calibration Flow ---")
    
    # Reset state and add positions again
    coordinator.state_machine.reset_to_idle()
    coordinator.position_tracker.positions = test_positions.copy()
    coordinator.position_tracker.cash_balance = 50.0  # Reset to low cash
    
    # Try calibration - should trigger cash recovery
    calibration_result = await coordinator.calibrate()
    
    logger.info(f"Calibration success: {calibration_result['success']}")
    if not calibration_result["success"]:
        logger.info(f"Calibration error: {calibration_result['error']}")
        logger.info(f"Failed step: {calibration_result.get('step_failed', 'unknown')}")
    
    # The calibration should fail due to low cash and trigger recovery flow
    assert not calibration_result["success"], "Calibration should fail due to low cash"
    assert "cash" in calibration_result["error"].lower(), "Error should mention cash"
    assert coordinator.state_machine.current_state == TraderState.LOW_CASH, "Should be in LOW_CASH state"
    
    logger.info("=== Cash Recovery State Machine Test Completed ===")
    
    # Test 5: State machine recovery behavior
    logger.info("\n--- Test 5: Automated State Recovery ---")
    
    # The state machine should automatically handle the cash recovery
    # Give it a moment to process the callbacks
    await asyncio.sleep(0.5)
    
    current_state = coordinator.state_machine.current_state
    logger.info(f"Final state: {current_state.value}")
    
    # State should have automatically progressed through recovery
    assert current_state in [TraderState.RECOVER_CASH, TraderState.CALIBRATING], \
        f"Expected RECOVER_CASH or CALIBRATING, got {current_state.value}"
    
    return True


async def test_position_tracker_cash_methods():
    """
    Test the new cash-related methods in PositionTracker.
    """
    logger.info("\n=== Testing PositionTracker Cash Methods ===")
    
    # Create position tracker with low cash
    tracker = PositionTracker(initial_cash_balance=75.0)
    
    # Add some positions
    tracker.positions = {
        "TEST1": Position("TEST1", 10, 50.0, 0.0),
        "TEST2": Position("TEST2", -5, 25.0, 0.0)
    }
    
    # Test cash threshold check
    threshold_result = tracker.check_cash_threshold(100.0)
    logger.info(f"Cash sufficient: {threshold_result['sufficient']}")
    logger.info(f"Current balance: ${threshold_result['current_balance']:.2f}")
    logger.info(f"Deficit: ${threshold_result['deficit']:.2f}")
    logger.info(f"Threshold ratio: {threshold_result['threshold_ratio']:.2f}")
    
    assert not threshold_result["sufficient"], "Cash should be insufficient"
    assert threshold_result["deficit"] == 25.0, "Deficit should be $25"
    assert threshold_result["threshold_ratio"] == 0.75, "Ratio should be 0.75"
    
    # Test liquidation estimate
    liquidation_result = tracker.estimate_position_liquidation_value()
    logger.info(f"Estimated recovery: ${liquidation_result['estimated_recovery']:.2f}")
    logger.info(f"Positions count: {liquidation_result['positions_count']}")
    
    assert liquidation_result["positions_count"] == 2, "Should find 2 positions"
    assert liquidation_result["estimated_recovery"] > 0, "Should estimate some recovery value"
    
    # Test comprehensive cash status
    cash_status = tracker.get_cash_status(100.0)
    logger.info(f"Can recover via liquidation: {cash_status['can_recover_via_liquidation']}")
    logger.info(f"Recovery needed: ${cash_status['recovery_needed']:.2f}")
    
    assert "threshold_check" in cash_status, "Should include threshold check"
    assert "liquidation_estimate" in cash_status, "Should include liquidation estimate"
    
    # Test cash balance update
    success = tracker.update_cash_balance_from_api(150.0)
    assert success, "Should successfully update cash balance"
    assert tracker.cash_balance == 150.0, "Cash balance should be updated"
    
    # Check threshold again after update
    new_threshold_result = tracker.check_cash_threshold(100.0)
    assert new_threshold_result["sufficient"], "Cash should now be sufficient"
    
    logger.info("PositionTracker cash methods test completed successfully")
    
    return True


async def main():
    """Run all cash recovery tests."""
    try:
        logger.info("Starting TRADER 2.0 Cash Recovery Tests")
        
        # Test position tracker methods
        await test_position_tracker_cash_methods()
        
        # Test full cash recovery flow
        await test_cash_recovery_flow()
        
        logger.info("\n🎉 All cash recovery tests PASSED! 🎉")
        logger.info("Cash recovery state machine is ready for integration")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())