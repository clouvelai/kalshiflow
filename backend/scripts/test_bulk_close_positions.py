#!/usr/bin/env python3
"""
Test script for bulk position closing functionality.

Uses real positions from the account and tests the bulk close positions functionality.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshiflow_rl.trading.kalshi_multi_market_order_manager import KalshiMultiMarketOrderManager
from kalshiflow_rl.trading.demo_client import KalshiDemoTradingClient
from kalshiflow_rl.config import logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger.info("=" * 60)
logger.info("🧪 Bulk Position Closing Test")
logger.info("=" * 60)


async def main():
    """Main test function."""
    try:
        # Initialize order manager
        order_manager = KalshiMultiMarketOrderManager(initial_cash=10000.0)
        logger.info("✅ Order manager created")
        
        # Initialize trading client and connect
        trading_client = KalshiDemoTradingClient(mode="paper")
        await trading_client.connect()
        order_manager.trading_client = trading_client
        logger.info("✅ Trading client connected")
        
        # Sync positions from Kalshi
        sync_result = await order_manager._sync_state_with_kalshi()
        logger.info(f"✅ Synced {len(order_manager.positions)} positions from Kalshi")
        
        # Display current positions
        active_positions = {
            ticker: pos
            for ticker, pos in order_manager.positions.items()
            if not pos.is_flat
        }
        logger.info(f"\n📋 Active positions to close: {len(active_positions)}")
        for ticker, position in active_positions.items():
            side = "YES" if position.contracts > 0 else "NO"
            logger.info(f"  - {ticker}: {abs(position.contracts)} {side} contracts")
        
        if not active_positions:
            logger.warning("⚠️  No active positions found. Nothing to close.")
            return 0
        
        # Test bulk close positions
        logger.info("\n🚀 Starting bulk position close...")
        result = await order_manager.close_bulk_positions()
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("📊 Bulk Close Results")
        logger.info("=" * 60)
        logger.info(f"Total positions: {result.get('total_positions', 0)}")
        logger.info(f"Orders submitted: {result.get('orders_submitted', 0)}")
        logger.info(f"Batches submitted: {result.get('batches_submitted', 0)}")
        
        client_order_ids = result.get('client_order_ids', [])
        if client_order_ids:
            logger.info(f"\n✅ Client Order IDs ({len(client_order_ids)}):")
            for i, order_id in enumerate(client_order_ids, 1):
                logger.info(f"  {i}. {order_id}")
        
        
        logger.info("=" * 60)
        
        # Disconnect
        await trading_client.disconnect()
        logger.info("✅ Trading client disconnected")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

