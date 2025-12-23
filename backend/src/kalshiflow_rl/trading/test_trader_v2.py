"""
Test TraderV2 functional parity with OrderManager.

Simple validation script to ensure the extracted services maintain
the same capabilities as the monolithic OrderManager.
"""

import asyncio
import logging
import sys
from typing import Dict, Any

# Test imports
from .trader_v2 import TraderV2
from .demo_client import KalshiDemoTradingClient

logger = logging.getLogger("kalshiflow_rl.trading.test_trader_v2")


async def test_basic_functionality() -> Dict[str, Any]:
    """Test basic TraderV2 functionality."""
    test_results = {
        "initialization": False,
        "service_creation": False,
        "compatibility_interface": False,
        "error_handling": False,
        "status_reporting": False
    }
    
    try:
        # Test initialization
        # Note: This uses dummy client for testing
        class DummyClient(KalshiDemoTradingClient):
            def __init__(self):
                # Minimal initialization for testing
                self.api_key_id = "test_key"
                self.private_key_content = "test_private_key"
                self.base_url = "https://demo-api.kalshi.co/trade-api/v2"
                
            async def get_account_info(self):
                return {"user_id": "test_user", "balance": 10000}
            
            async def get_positions(self):
                return {"positions": []}
            
            async def get_orders(self):
                return {"orders": []}
        
        dummy_client = DummyClient()
        
        trader = TraderV2(
            client=dummy_client,
            initial_cash_balance=1000.0,
            market_tickers=["TEST-MARKET"]
        )
        
        test_results["initialization"] = True
        logger.info("✅ Initialization test passed")
        
        # Test service creation
        assert trader.coordinator is not None
        assert trader.state_machine is not None
        assert trader.order_service is not None
        assert trader.position_tracker is not None
        assert trader.status_logger is not None
        test_results["service_creation"] = True
        logger.info("✅ Service creation test passed")
        
        # Test compatibility interface
        assert hasattr(trader, 'place_order')
        assert hasattr(trader, 'cancel_order')
        assert hasattr(trader, 'get_positions')
        assert hasattr(trader, 'get_cash_balance_cents')
        assert hasattr(trader, 'get_open_orders')
        test_results["compatibility_interface"] = True
        logger.info("✅ Compatibility interface test passed")
        
        # Test error handling
        assert hasattr(trader, 'emergency_stop')
        assert hasattr(trader, 'handle_orderbook_failure')
        assert hasattr(trader, 'attempt_recovery')
        test_results["error_handling"] = True
        logger.info("✅ Error handling test passed")
        
        # Test status reporting
        status = trader.get_comprehensive_status()
        assert isinstance(status, dict)
        assert "state_machine" in status
        assert "services" in status
        assert "portfolio" in status
        
        debug_summary = trader.get_debug_summary()
        assert isinstance(debug_summary, str)
        assert "TRADER STATUS" in debug_summary
        test_results["status_reporting"] = True
        logger.info("✅ Status reporting test passed")
        
        # Test functional parity validation
        parity_result = await trader.validate_functional_parity()
        logger.info(f"Functional parity validation: {parity_result}")
        
        return {
            "success": True,
            "tests": test_results,
            "functional_parity": parity_result
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "tests": test_results
        }


async def test_state_machine() -> Dict[str, Any]:
    """Test state machine functionality."""
    try:
        from .state_machine import TraderStateMachine, TraderState
        from .services.status_logger import StatusLogger
        
        # Test state machine
        status_logger = StatusLogger()
        state_machine = TraderStateMachine(status_logger)
        
        # Test initial state
        assert state_machine.current_state == TraderState.IDLE
        
        # Test valid transitions
        success = await state_machine.transition_to(TraderState.CALIBRATING, "test")
        assert success
        assert state_machine.current_state == TraderState.CALIBRATING
        
        success = await state_machine.transition_to(TraderState.READY, "test")
        assert success
        assert state_machine.current_state == TraderState.READY
        
        success = await state_machine.transition_to(TraderState.ACTING, "test")
        assert success
        assert state_machine.current_state == TraderState.ACTING
        
        success = await state_machine.transition_to(TraderState.READY, "test")
        assert success
        assert state_machine.current_state == TraderState.READY
        
        # Test operational checks
        assert state_machine.is_operational()
        assert state_machine.is_trading_allowed()
        assert state_machine.can_process_action()
        
        logger.info("✅ State machine test passed")
        
        return {"success": True}
        
    except Exception as e:
        logger.error(f"State machine test failed: {e}")
        return {"success": False, "error": str(e)}


async def run_tests() -> Dict[str, Any]:
    """Run all tests."""
    logger.info("Running TraderV2 functional parity tests...")
    
    test_results = {}
    
    # Test basic functionality
    test_results["basic_functionality"] = await test_basic_functionality()
    
    # Test state machine
    test_results["state_machine"] = await test_state_machine()
    
    # Overall success
    overall_success = all(result.get("success", False) for result in test_results.values())
    
    return {
        "success": overall_success,
        "tests": test_results,
        "summary": f"{'✅ All tests passed' if overall_success else '❌ Some tests failed'}"
    }


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    async def main():
        results = await run_tests()
        
        print("\n" + "="*50)
        print("TRADER 2.0 FUNCTIONAL PARITY TEST RESULTS")
        print("="*50)
        print(f"Overall Success: {results['success']}")
        print(f"Summary: {results['summary']}")
        
        for test_name, result in results['tests'].items():
            print(f"\n{test_name.upper()}:")
            print(f"  Success: {result.get('success', False)}")
            if not result.get('success', False):
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*50)
        
        # Exit with appropriate code
        sys.exit(0 if results['success'] else 1)
    
    asyncio.run(main())