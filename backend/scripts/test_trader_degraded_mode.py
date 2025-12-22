#!/usr/bin/env python3
"""
Test script for validating the RL trader's degraded mode and graceful degradation.

This script simulates various failure scenarios to ensure the trader:
1. Properly classifies components as critical vs optional
2. Continues trading when critical components work (even without WebSockets)
3. Activates fallback strategies for unhealthy optional components
4. Clearly communicates health status and limitations
"""

import asyncio
import logging
import json
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def simulate_503_error():
    """Simulate a 503 Service Unavailable error from Kalshi."""
    raise Exception("503 Service Unavailable")


async def simulate_healthy_exchange():
    """Simulate a healthy exchange response."""
    return {
        "exchange_active": True,
        "trading_active": True
    }


async def test_health_check_classification():
    """Test that components are properly classified as critical vs optional."""
    logger.info("\n=== Testing Health Check Classification ===")
    
    # Import the order manager
    from kalshiflow_rl.trading.kalshi_multi_market_order_manager import KalshiMultiMarketOrderManager
    
    # Create a mock client with AsyncMock
    mock_client = AsyncMock()
    
    # Create order manager
    order_manager = KalshiMultiMarketOrderManager(initial_cash=10000.0)
    # Inject mock client
    order_manager.trading_client = mock_client
    order_manager._client = mock_client
    
    # Test 1: All components healthy
    logger.info("\nTest 1: All components healthy")
    mock_client.get_exchange_status.return_value = {
        "exchange_active": True,
        "trading_active": True
    }
    
    with patch.object(order_manager, '_check_orderbook_health_gracefully') as mock_ob_health:
        mock_ob_health.return_value = {
            "healthy": True,
            "connected": True,
            "snapshots": 100,
            "deltas": 500,
            "reason": "healthy"
        }
        
        with patch.object(order_manager, '_check_fill_listener_health') as mock_fill_health:
            mock_fill_health.return_value = {
                "healthy": True,
                "active": True,
                "fills_received": 10
            }
            
            with patch.object(order_manager, '_check_order_listener_health') as mock_order_health:
                mock_order_health.return_value = {
                    "healthy": True,
                    "active": True,
                    "orders_tracked": 5
                }
                
                # Mock position listener
                order_manager._position_listener = MagicMock()
                order_manager._position_listener.is_healthy.return_value = True
                
                health_results = await order_manager._calibration_health_checks()
                
                # Debug output
                logger.debug(f"Health results: {json.dumps(health_results, indent=2, default=str)}")
                
                assert health_results["can_trade"] == True
                assert health_results["overall_status"] == "fully_operational"
                logger.info(f"✅ All healthy: can_trade={health_results['can_trade']}, status={health_results['overall_status']}")
    
    # Test 2: Optional components unhealthy (WebSockets down)
    logger.info("\nTest 2: Optional components unhealthy (WebSockets down)")
    
    # Mock position listener as unhealthy
    order_manager._position_listener = None
    
    with patch.object(order_manager, '_check_orderbook_health_gracefully') as mock_ob_health:
        mock_ob_health.return_value = {
            "healthy": False,
            "connected": False,
            "snapshots": 0,
            "deltas": 0,
            "reason": "connection_issue",
            "error": "WebSocket connection failed"
        }
        
        with patch.object(order_manager, '_check_fill_listener_health') as mock_fill_health:
            mock_fill_health.return_value = {
                "healthy": False,
                "active": False,
                "fills_received": 0,
                "error": "WebSocket not connected"
            }
            
            with patch.object(order_manager, '_check_order_listener_health') as mock_order_health:
                mock_order_health.return_value = {
                    "healthy": True,
                    "active": True,
                    "orders_tracked": 5
                }
                
                health_results = await order_manager._calibration_health_checks()
                
                assert health_results["can_trade"] == True  # Still can trade!
                assert health_results["overall_status"] == "degraded"
                assert len(health_results["fallback_strategies"]) > 0
                logger.info(f"✅ Degraded mode: can_trade={health_results['can_trade']}, status={health_results['overall_status']}")
                logger.info(f"   Fallback strategies: {health_results['fallback_strategies']}")
    
    # Test 3: Critical component failure (Exchange 503)
    logger.info("\nTest 3: Critical component failure (Exchange 503)")
    mock_client.get_exchange_status.side_effect = Exception("503 Service Unavailable")
    
    # Reset position listener for this test
    order_manager._position_listener = MagicMock()
    order_manager._position_listener.is_healthy.return_value = True
    
    with patch.object(order_manager, '_check_orderbook_health_gracefully') as mock_ob_health:
        mock_ob_health.return_value = {
            "healthy": True,
            "connected": True,
            "snapshots": 100,
            "deltas": 500,
            "reason": "healthy"
        }
        
        with patch.object(order_manager, '_check_fill_listener_health') as mock_fill_health:
            mock_fill_health.return_value = {
                "healthy": True,
                "active": True,
                "fills_received": 10
            }
            
            with patch.object(order_manager, '_check_order_listener_health') as mock_order_health:
                mock_order_health.return_value = {
                    "healthy": True,
                    "active": True,
                    "orders_tracked": 5
                }
                
                health_results = await order_manager._calibration_health_checks()
                
                assert health_results["can_trade"] == False  # Cannot trade without exchange!
                assert health_results["overall_status"] == "paused"
                assert health_results["components"]["exchange"]["is_503"] == True
                logger.info(f"✅ Paused due to 503: can_trade={health_results['can_trade']}, status={health_results['overall_status']}")
                logger.info(f"   Exchange 503 detected: {health_results['components']['exchange']['is_503']}")


async def test_fallback_activation():
    """Test that fallback strategies are properly activated."""
    logger.info("\n=== Testing Fallback Strategy Activation ===")
    
    from kalshiflow_rl.trading.kalshi_multi_market_order_manager import KalshiMultiMarketOrderManager
    
    # Create a mock client
    mock_client = AsyncMock()
    
    # Create order manager
    order_manager = KalshiMultiMarketOrderManager(initial_cash=10000.0)
    # Inject mock client
    order_manager.trading_client = mock_client
    
    # Create health results with unhealthy optional components
    health_results = {
        "components": {
            "orderbook": {
                "healthy": False,
                "category": "optional",
                "fallback": {
                    "strategy": "rest_api_polling"
                }
            },
            "fill_listener": {
                "healthy": False,
                "category": "optional",
                "fallback": {
                    "strategy": "order_status_polling"
                }
            },
            "position_listener": {
                "healthy": False,
                "category": "optional",
                "fallback": {
                    "strategy": "periodic_sync"
                }
            }
        }
    }
    
    # Activate fallback strategies
    await order_manager._activate_fallback_strategies(health_results)
    
    # Check that fallback flags are set
    assert getattr(order_manager, '_fallback_orderbook_enabled', False) == True
    assert getattr(order_manager, '_fallback_fill_polling_enabled', False) == True
    assert getattr(order_manager, '_fallback_position_sync_enabled', False) == True
    
    logger.info("✅ All fallback strategies activated successfully")


async def test_degraded_mode_adjustments():
    """Test that trading behavior is adjusted in degraded mode."""
    logger.info("\n=== Testing Degraded Mode Adjustments ===")
    
    from kalshiflow_rl.trading.kalshi_multi_market_order_manager import KalshiMultiMarketOrderManager
    
    # Create a mock client
    mock_client = AsyncMock()
    
    # Create order manager
    order_manager = KalshiMultiMarketOrderManager(initial_cash=10000.0)
    # Inject mock client
    order_manager.trading_client = mock_client
    
    # Set initial values
    order_manager.max_position_size = 100
    order_manager.min_cash_reserve = 500
    order_manager._degraded_mode = True
    
    # Apply degraded mode adjustments
    await order_manager._handle_degraded_trading()
    
    # Check adjustments
    assert order_manager.max_position_size == 50  # Reduced from 100
    assert order_manager.min_cash_reserve == 1000  # Increased from 500
    assert order_manager._prefer_closing_positions == True
    
    logger.info("✅ Degraded mode adjustments applied correctly")
    logger.info(f"   Max position size: 100 → 50")
    logger.info(f"   Min cash reserve: 500 → 1000")
    logger.info(f"   Prefer closing positions: True")


async def test_recovery_cycle():
    """Test the recovery cycle from paused state."""
    logger.info("\n=== Testing Recovery Cycle ===")
    
    from kalshiflow_rl.trading.kalshi_multi_market_order_manager import KalshiMultiMarketOrderManager
    
    # Create a mock client
    mock_client = AsyncMock()
    
    # Create order manager
    order_manager = KalshiMultiMarketOrderManager(initial_cash=10000.0)
    # Inject mock client
    order_manager.trading_client = mock_client
    order_manager._client = mock_client
    order_manager._running = True
    order_manager._calibration_in_progress = False
    order_manager._trader_status = "paused -> waiting for recovery"
    order_manager._last_calibration_time = 0
    
    # Mock the run_calibration method
    calibration_calls = []
    async def mock_calibration(reason):
        calibration_calls.append(reason)
        # Simulate successful recovery on second attempt
        if len(calibration_calls) >= 2:
            order_manager._trader_status = "trading"
    
    order_manager.run_calibration = mock_calibration
    
    # Run state machine loop for a short time
    loop_task = asyncio.create_task(order_manager._state_machine_loop())
    
    # Wait for recovery attempts
    await asyncio.sleep(2.5)
    
    # Stop the loop
    order_manager._running = False
    await loop_task
    
    # Check that recovery was attempted
    assert len(calibration_calls) > 0
    assert "periodic_recovery_check" in calibration_calls[0]
    
    logger.info("✅ Recovery cycle working correctly")
    logger.info(f"   Recovery attempts: {len(calibration_calls)}")
    logger.info(f"   Recovery reasons: {calibration_calls}")


async def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing RL Trader Degraded Mode and Graceful Degradation")
    logger.info("=" * 60)
    
    try:
        # Test health check classification
        await test_health_check_classification()
        
        # Test fallback activation
        await test_fallback_activation()
        
        # Test degraded mode adjustments
        await test_degraded_mode_adjustments()
        
        # Test recovery cycle
        await test_recovery_cycle()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL TESTS PASSED")
        logger.info("=" * 60)
        logger.info("\nSummary:")
        logger.info("- Components properly classified as critical vs optional")
        logger.info("- System continues trading in degraded mode when critical components work")
        logger.info("- Fallback strategies activate for unhealthy optional components")
        logger.info("- Trading behavior adjusts appropriately in degraded mode")
        logger.info("- Recovery cycle attempts to restore full functionality")
        logger.info("\nThe trader is now resilient to WebSocket failures and 503 errors!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())