#!/usr/bin/env python3
"""
Test script for TRADER 2.0 WebSocket Integration

Verifies that TraderV2 properly broadcasts messages via the global WebSocketManager.
"""

import asyncio
import logging
import json
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TraderV2 and required components
from trading.trader_v2 import TraderV2
from trading.demo_client import KalshiDemoTradingClient
from websocket_manager import websocket_manager
from config import config


class WebSocketTestMonitor:
    """Monitor WebSocket broadcasts for testing."""
    
    def __init__(self):
        self.messages = []
        self.message_counts = {
            "trader_state": 0,
            "orders_update": 0,
            "position_update": 0,
            "portfolio_update": 0,
            "fill_event": 0,
            "initialization_step": 0,
            "initialization_complete": 0
        }
    
    async def intercept_broadcast(self, method_name: str, data: Dict[str, Any]):
        """Intercept broadcast calls for testing."""
        logger.info(f"📡 Broadcast: {method_name}")
        logger.info(f"   Data: {json.dumps(data, indent=2)[:200]}...")
        
        self.messages.append({
            "method": method_name,
            "data": data,
            "timestamp": time.time()
        })
        
        if method_name in self.message_counts:
            self.message_counts[method_name] += 1
    
    def get_summary(self) -> str:
        """Get test summary."""
        return f"""
        WebSocket Broadcast Summary:
        ============================
        Total Messages: {len(self.messages)}
        
        By Type:
        - Trader State: {self.message_counts['trader_state']}
        - Orders Update: {self.message_counts['orders_update']}
        - Position Update: {self.message_counts['position_update']}
        - Portfolio Update: {self.message_counts['portfolio_update']}
        - Fill Events: {self.message_counts['fill_event']}
        - Initialization Steps: {self.message_counts['initialization_step']}
        - Initialization Complete: {self.message_counts['initialization_complete']}
        """


async def test_trader_v2_websocket():
    """Test TraderV2 WebSocket integration."""
    logger.info("=" * 60)
    logger.info("TRADER 2.0 WebSocket Integration Test")
    logger.info("=" * 60)
    
    # Create test monitor
    monitor = WebSocketTestMonitor()
    
    # Monkey-patch websocket_manager methods to intercept broadcasts
    original_methods = {}
    for method_name in ["broadcast_trader_state", "broadcast_orders_update", 
                        "broadcast_position_update", "broadcast_portfolio_update",
                        "broadcast_fill_event", "broadcast_initialization_step",
                        "broadcast_initialization_complete"]:
        if hasattr(websocket_manager, method_name):
            original = getattr(websocket_manager, method_name)
            original_methods[method_name] = original
            
            async def make_interceptor(name):
                async def interceptor(data, *args, **kwargs):
                    await monitor.intercept_broadcast(name, data)
                    # Still call original if it exists
                    return await original_methods[name](data, *args, **kwargs)
                return interceptor
            
            setattr(websocket_manager, method_name, await make_interceptor(method_name))
    
    try:
        # Initialize demo client
        logger.info("\n1. Initializing Demo Client...")
        demo_client = KalshiDemoTradingClient(
            api_key_id=config.KALSHI_API_KEY_ID,
            private_key_content=config.KALSHI_PRIVATE_KEY_CONTENT,
            api_url="https://demo-api.kalshi.co/trade-api/v2"
        )
        
        # Create TraderV2 with global websocket manager
        logger.info("\n2. Creating TraderV2 with WebSocket Manager...")
        trader = TraderV2(
            client=demo_client,
            websocket_manager=websocket_manager,
            initial_cash_balance=10000.0,
            market_tickers=["INXD-25JAN03", "TRUMPNOMOT-25JAN06"]
        )
        
        # Start TraderV2 (should trigger calibration broadcasts)
        logger.info("\n3. Starting TraderV2 (calibration should broadcast)...")
        startup_result = await trader.start(
            enable_websockets=True,
            enable_orderbook=False
        )
        
        if not startup_result["success"]:
            logger.error(f"❌ Startup failed: {startup_result}")
            return False
        
        logger.info(f"✅ TraderV2 started successfully")
        logger.info(f"   State: {trader.state_machine.current_state.value}")
        logger.info(f"   Services: {trader.services_status}")
        
        # Wait a bit for any async broadcasts
        await asyncio.sleep(2)
        
        # Test placing an order (should trigger order broadcast)
        logger.info("\n4. Testing order placement (should broadcast)...")
        from data.orderbook_state import OrderbookState
        
        # Create mock orderbook
        mock_orderbook = OrderbookState("TEST-MARKET")
        mock_orderbook.best_bid = {"price": 45, "quantity": 100}
        mock_orderbook.best_ask = {"price": 55, "quantity": 100}
        
        # Place test order
        order_result = await trader.place_order(
            ticker="TEST-MARKET",
            side="buy",
            contract_side="yes",
            quantity=10,
            orderbook=mock_orderbook,
            pricing_strategy="passive"
        )
        
        if order_result:
            logger.info(f"✅ Order placed: {order_result['order_id']}")
        else:
            logger.info("⚠️ Order placement skipped (expected in test mode)")
        
        # Wait for broadcasts
        await asyncio.sleep(2)
        
        # Get comprehensive status
        logger.info("\n5. Getting comprehensive status...")
        status = trader.get_comprehensive_status()
        logger.info(f"System Health: {status['system_health']}")
        logger.info(f"Trading Status: {status.get('state', 'N/A')}")
        
        # Stop trader
        logger.info("\n6. Stopping TraderV2...")
        await trader.stop()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info(monitor.get_summary())
        logger.info("=" * 60)
        
        # Verify broadcasts occurred
        success = len(monitor.messages) > 0
        if success:
            logger.info("\n✅ WebSocket Integration Test PASSED")
            logger.info(f"   Received {len(monitor.messages)} broadcast messages")
        else:
            logger.error("\n❌ WebSocket Integration Test FAILED")
            logger.error("   No broadcast messages detected")
        
        return success
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restore original methods
        for method_name, original in original_methods.items():
            setattr(websocket_manager, method_name, original)


async def main():
    """Main test entry point."""
    success = await test_trader_v2_websocket()
    
    if success:
        logger.info("\n🎉 All WebSocket integration tests passed!")
    else:
        logger.error("\n💥 WebSocket integration tests failed")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)