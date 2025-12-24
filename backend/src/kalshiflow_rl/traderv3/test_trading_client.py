#!/usr/bin/env python3
"""
Test script for TRADER V3 Trading Client Integration.

Tests the new trading client integration with the V3 architecture.
Verifies state transitions: ORDERBOOK_CONNECT -> TRADING_CLIENT_CONNECT -> KALSHI_DATA_SYNC -> READY
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_trading_client")


async def test_trading_client_integration():
    """Test the V3 trading client integration."""
    
    # Load environment - should be in paper mode
    load_dotenv()
    
    # Check environment
    environment = os.environ.get("ENVIRONMENT", "local")
    logger.info(f"Environment: {environment}")
    
    if environment != "paper":
        logger.warning("Not in paper environment - switching to paper mode for safety")
        os.environ["ENVIRONMENT"] = "paper"
        # Reload dotenv with paper environment
        load_dotenv(dotenv_path=".env.paper", override=True)
    
    # Enable trading client for this test
    os.environ["V3_ENABLE_TRADING_CLIENT"] = "true"
    os.environ["V3_TRADING_MAX_ORDERS"] = "5"
    os.environ["V3_TRADING_MAX_POSITION_SIZE"] = "50"
    os.environ["RL_MODE"] = "config"  # Use specific markets, not discovery
    os.environ["RL_MARKET_TICKERS"] = "INXD-25JAN03"  # Single market for testing
    
    logger.info("=" * 60)
    logger.info("TESTING V3 TRADING CLIENT INTEGRATION")
    logger.info("=" * 60)
    
    try:
        # Import V3 components
        from kalshiflow_rl.traderv3.core.state_machine import TraderStateMachine as V3StateMachine
        from kalshiflow_rl.traderv3.core.event_bus import EventBus, EventType
        from kalshiflow_rl.traderv3.core.websocket_manager import V3WebSocketManager
        from kalshiflow_rl.traderv3.core.coordinator import V3Coordinator
        from kalshiflow_rl.traderv3.clients.orderbook_integration import V3OrderbookIntegration
        from kalshiflow_rl.traderv3.clients.trading_client_integration import V3TradingClientIntegration
        from kalshiflow_rl.traderv3.config.environment import load_config
        from kalshiflow_rl.data.orderbook_client import OrderbookClient
        from kalshiflow_rl.trading.demo_client import KalshiDemoTradingClient
        
        # Load configuration
        config = load_config()
        assert config.enable_trading_client, "Trading client should be enabled"
        assert config.trading_mode == "paper", "Should be in paper mode"
        
        logger.info(f"✅ Config loaded - Trading enabled: {config.enable_trading_client}, Mode: {config.trading_mode}")
        
        # Create components
        logger.info("Creating V3 components...")
        
        event_bus = EventBus()
        await event_bus.start()
        
        state_machine = V3StateMachine(event_bus=event_bus)
        await state_machine.start()
        
        websocket_manager = V3WebSocketManager(event_bus=event_bus, state_machine=state_machine)
        await websocket_manager.start()
        
        # Create orderbook client
        orderbook_client = OrderbookClient(
            market_tickers=["INXD-25JAN03"],
            event_bus=event_bus
        )
        
        orderbook_integration = V3OrderbookIntegration(
            orderbook_client=orderbook_client,
            event_bus=event_bus,
            market_tickers=["INXD-25JAN03"]
        )
        await orderbook_integration.start()
        
        # Create trading client
        logger.info("Creating trading client integration...")
        trading_client = KalshiDemoTradingClient(mode="paper")
        
        trading_client_integration = V3TradingClientIntegration(
            trading_client=trading_client,
            event_bus=event_bus,
            max_orders=5,
            max_position_size=50
        )
        await trading_client_integration.start()
        
        # Create coordinator with trading client
        coordinator = V3Coordinator(
            config=config,
            state_machine=state_machine,
            event_bus=event_bus,
            websocket_manager=websocket_manager,
            orderbook_integration=orderbook_integration,
            trading_client_integration=trading_client_integration
        )
        
        # Track state transitions
        states_seen = []
        
        async def track_state_transition(event_type, data):
            """Track state transitions."""
            if event_type == EventType.STATE_TRANSITION:
                from_state = data.get("from_state")
                to_state = data.get("to_state")
                states_seen.append(to_state)
                logger.info(f"State transition: {from_state} -> {to_state}")
        
        event_bus.subscribe(EventType.STATE_TRANSITION, track_state_transition)
        
        # Start the coordinator
        logger.info("Starting coordinator...")
        await coordinator.start()
        
        # Wait a bit for system to stabilize
        await asyncio.sleep(5)
        
        # Check final state
        current_state = state_machine.current_state.value
        logger.info(f"Current state: {current_state}")
        
        # Verify state transitions
        logger.info(f"States seen: {states_seen}")
        
        # Check expected states were visited
        expected_states = ["initializing", "orderbook_connect"]
        if config.enable_trading_client:
            expected_states.extend(["trading_client_connect", "kalshi_data_sync"])
        expected_states.append("ready")
        
        for state in expected_states:
            if state in states_seen:
                logger.info(f"✅ State '{state}' was reached")
            else:
                logger.warning(f"❌ State '{state}' was NOT reached")
        
        # Check component health
        health = coordinator.get_health()
        logger.info(f"System health: {health}")
        
        # Get detailed status
        status = coordinator.get_status()
        logger.info(f"System status:")
        logger.info(f"  - State: {status['state']}")
        logger.info(f"  - Environment: {status['environment']}")
        if 'trading_mode' in status:
            logger.info(f"  - Trading mode: {status['trading_mode']}")
        
        # Check trading client metrics
        if trading_client_integration:
            trading_metrics = trading_client_integration.get_metrics()
            logger.info(f"Trading client metrics:")
            logger.info(f"  - Connected: {trading_metrics['connected']}")
            logger.info(f"  - Mode: {trading_metrics['mode']}")
            logger.info(f"  - Calibrated: {trading_metrics['calibrated']}")
            logger.info(f"  - Balance: {trading_metrics['balance']}")
            logger.info(f"  - Positions: {trading_metrics['positions_count']}")
            logger.info(f"  - Open orders: {trading_metrics['open_orders_count']}")
        
        # Test successful if we reached READY state
        if current_state == "ready":
            logger.info("=" * 60)
            logger.info("✅ TEST PASSED - Trading client integration successful!")
            logger.info("=" * 60)
        else:
            logger.error(f"❌ TEST FAILED - Expected READY state, got {current_state}")
        
        # Stop the system
        logger.info("Stopping coordinator...")
        await coordinator.stop()
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        raise
    
    finally:
        logger.info("Test complete")


if __name__ == "__main__":
    asyncio.run(test_trading_client_integration())