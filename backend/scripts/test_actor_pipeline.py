#!/usr/bin/env python3
"""
Test script for Actor/Trader Pipeline M1-M2 verification.

Initializes the ActorService components and waits for at least one
orderbook update to verify the end-to-end pipeline works.
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshiflow_rl.config import config, logger
from kalshiflow_rl.data.database import rl_db
from kalshiflow_rl.data.write_queue import write_queue
from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.data.orderbook_state import get_shared_orderbook_state
from kalshiflow_rl.data.auth import validate_rl_auth
from kalshiflow_rl.trading.event_bus import get_event_bus
from kalshiflow_rl.trading.actor_service import ActorService
from kalshiflow_rl.trading.kalshi_multi_market_order_manager import KalshiMultiMarketOrderManager
from kalshiflow_rl.trading.live_observation_adapter import LiveObservationAdapter
from kalshiflow_rl.trading.hardcoded_policies import HardcodedSelector

# Track if we've seen an event
event_received = False
shutdown_event = asyncio.Event()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger.info("=" * 60)
logger.info("🧪 Actor/Trader Pipeline Test - M1-M2 Verification")
logger.info("=" * 60)


async def main():
    """Main test function."""
    global event_received
    
    try:
        # Validate authentication
        if not validate_rl_auth():
            logger.error("❌ Authentication validation failed")
            return 1
        
        logger.info("✅ Authentication validated")
        
        # Initialize database
        await rl_db.initialize()
        logger.info("✅ Database initialized")
        
        # Start write queue
        await write_queue.start()
        logger.info("✅ Write queue started")
        
        # Start EventBus
        event_bus = await get_event_bus()
        await event_bus.start()
        logger.info("✅ EventBus started")
        
        # Get market tickers (use config, limit to 1-2 for testing)
        market_tickers = config.RL_MARKET_TICKERS[:2] if len(config.RL_MARKET_TICKERS) > 0 else ["INXD-25JAN03"]
        logger.info(f"📊 Testing with markets: {', '.join(market_tickers)}")
        
        # Create OrderbookStateRegistry wrapper
        class OrderbookStateRegistryWrapper:
            async def get_shared_orderbook_state(self, market_ticker: str):
                return await get_shared_orderbook_state(market_ticker)
        
        registry = OrderbookStateRegistryWrapper()
        
        # Create LiveObservationAdapter
        observation_adapter = LiveObservationAdapter(
            window_size=10,
            max_markets=1,
            temporal_context_minutes=30,
            orderbook_state_registry=registry
        )
        logger.info("✅ LiveObservationAdapter created")
        
        # Create KalshiMultiMarketOrderManager (skip demo client init)
        order_manager = KalshiMultiMarketOrderManager(initial_cash=10000.0)
        logger.info("✅ KalshiMultiMarketOrderManager created (demo client skipped)")
        
        # Create ActorService
        actor_service = ActorService(
            market_tickers=market_tickers,
            model_path=None,
            queue_size=1000,
            throttle_ms=250,
            event_bus=event_bus,
            observation_adapter=observation_adapter
        )
        
        # Set action selector (hardcoded - always hold)
        actor_service.set_action_selector(HardcodedSelector())
        logger.info("✅ Action selector configured (HardcodedSelector)")
        
        # Set order manager
        actor_service.set_order_manager(order_manager)
        logger.info("✅ Order manager configured")
        
        # Initialize ActorService
        await actor_service.initialize()
        logger.info("✅ ActorService initialized and ready")
        
        # Initialize OrderbookClient
        orderbook_client = OrderbookClient(market_tickers=market_tickers)
        logger.info("✅ OrderbookClient created")
        
        # Start orderbook client as background task
        orderbook_task = asyncio.create_task(orderbook_client.start())
        logger.info("✅ OrderbookClient started (connecting to Kalshi...)")
        
        # Wait for at least one event to be processed
        logger.info("⏳ Waiting for orderbook update and ActorService processing...")
        logger.info("   (This may take 10-30 seconds for first snapshot)")
        
        # Monitor ActorService metrics
        start_time = asyncio.get_event_loop().time()
        timeout = 60  # 60 second timeout
        
        while not shutdown_event.is_set():
            await asyncio.sleep(2)
            
            # Check ActorService metrics
            metrics = actor_service.get_metrics()
            elapsed = asyncio.get_event_loop().time() - start_time
            
            if metrics['events_processed'] > 0:
                logger.info("=" * 60)
                logger.info("✅ SUCCESS: ActorService processed at least one event!")
                logger.info("=" * 60)
                logger.info(f"📊 Metrics:")
                logger.info(f"   - Events queued: {metrics['events_queued']}")
                logger.info(f"   - Events processed: {metrics['events_processed']}")
                logger.info(f"   - Model predictions: {metrics['model_predictions']}")
                logger.info(f"   - Orders executed: {metrics['orders_executed']}")
                logger.info(f"   - Avg processing time: {metrics['avg_processing_time_ms']:.2f}ms")
                logger.info(f"   - Queue depth: {metrics['queue_depth']}")
                logger.info(f"   - Errors: {metrics['errors']}")
                event_received = True
                break
            
            if elapsed > timeout:
                logger.warning(f"⏱️  Timeout after {timeout}s - no events processed yet")
                logger.info(f"   Current metrics: {metrics}")
                break
            
            logger.debug(f"   Waiting... ({elapsed:.1f}s elapsed, {metrics['events_queued']} queued)")
        
        # Give it a moment to finish processing
        await asyncio.sleep(2)
        
        # Final metrics
        final_metrics = actor_service.get_metrics()
        logger.info("=" * 60)
        logger.info("📊 Final ActorService Metrics:")
        logger.info(f"   - Events queued: {final_metrics['events_queued']}")
        logger.info(f"   - Events processed: {final_metrics['events_processed']}")
        logger.info(f"   - Model predictions: {final_metrics['model_predictions']}")
        logger.info(f"   - Orders executed: {final_metrics['orders_executed']}")
        logger.info(f"   - Avg processing time: {final_metrics['avg_processing_time_ms']:.2f}ms")
        logger.info(f"   - Errors: {final_metrics['errors']}")
        logger.info(f"   - Last error: {final_metrics.get('last_error', 'None')}")
        logger.info("=" * 60)
        
        # Shutdown
        logger.info("🛑 Shutting down...")
        
        # Cancel orderbook task
        orderbook_task.cancel()
        try:
            await orderbook_task
        except asyncio.CancelledError:
            pass
        
        # Stop orderbook client
        try:
            await orderbook_client.stop()
        except Exception as e:
            logger.error(f"Error stopping orderbook client: {e}")
        
        # Shutdown ActorService
        await actor_service.shutdown()
        
        # Shutdown OrderManager
        await order_manager.shutdown()
        
        # Stop EventBus
        await event_bus.stop()
        
        # Stop write queue
        await write_queue.stop()
        
        # Close database
        await rl_db.close()
        
        logger.info("✅ Shutdown complete")
        
        if event_received:
            logger.info("=" * 60)
            logger.info("✅ TEST PASSED: Pipeline is working!")
            logger.info("=" * 60)
            return 0
        else:
            logger.warning("=" * 60)
            logger.warning("⚠️  TEST INCOMPLETE: No events processed")
            logger.warning("=" * 60)
            return 1
        
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        shutdown_event.set()
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_event.set()


if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the test
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

