#!/usr/bin/env python3
"""
Debug script to test analytics service real-time updates.
"""

import asyncio
import logging
import websockets
import json
from datetime import datetime

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def test_websocket_updates():
    """Connect to backend WebSocket and monitor analytics updates."""
    
    try:
        # Connect to local backend WebSocket
        uri = "ws://localhost:8000/ws/stream"
        logger.info(f"Connecting to {uri}")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected to WebSocket")
            
            # Track message types and timing
            message_counts = {
                "snapshot": 0,
                "trade": 0,
                "analytics_update": 0,
                "other": 0
            }
            
            analytics_updates = []
            last_analytics_time = None
            
            # Listen for messages for 30 seconds
            start_time = datetime.now()
            timeout = 30  # seconds
            
            logger.info(f"Listening for messages for {timeout} seconds...")
            
            try:
                while (datetime.now() - start_time).seconds < timeout:
                    # Set a short timeout for each message
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "unknown")
                        
                        if msg_type in message_counts:
                            message_counts[msg_type] += 1
                        else:
                            message_counts["other"] += 1
                        
                        current_time = datetime.now()
                        
                        if msg_type == "analytics_update":
                            analytics_data = data.get("data", {})
                            mode = analytics_data.get("mode", "unknown")
                            current_period = analytics_data.get("current_period", {})
                            volume = current_period.get("volume_usd", 0)
                            trades = current_period.get("trade_count", 0)
                            
                            analytics_updates.append({
                                "time": current_time,
                                "mode": mode,
                                "volume": volume,
                                "trades": trades
                            })
                            
                            time_since_last = ""
                            if last_analytics_time:
                                delta = (current_time - last_analytics_time).total_seconds()
                                time_since_last = f" (+{delta:.1f}s)"
                            
                            logger.info(f"📊 ANALYTICS UPDATE ({mode}): ${volume:.2f} volume, {trades} trades{time_since_last}")
                            last_analytics_time = current_time
                            
                        elif msg_type == "trade":
                            trade_data = data.get("data", {}).get("trade", {})
                            ticker = trade_data.get("market_ticker", "unknown")
                            logger.info(f"💰 TRADE: {ticker}")
                            
                        elif msg_type == "snapshot":
                            logger.info("📸 SNAPSHOT received")
                        
                    except json.JSONDecodeError:
                        logger.error("Failed to parse message as JSON")
                        continue
                        
            except asyncio.TimeoutError:
                # Normal timeout, continue
                pass
                
            # Print summary
            total_messages = sum(message_counts.values())
            logger.info("\n" + "="*50)
            logger.info("📈 MESSAGE SUMMARY")
            logger.info("="*50)
            logger.info(f"Total messages: {total_messages}")
            for msg_type, count in message_counts.items():
                if count > 0:
                    logger.info(f"  {msg_type}: {count}")
            
            logger.info(f"\n📊 ANALYTICS UPDATES: {len(analytics_updates)}")
            if analytics_updates:
                logger.info("Recent updates:")
                for update in analytics_updates[-5:]:  # Last 5 updates
                    logger.info(f"  {update['time'].strftime('%H:%M:%S')} - {update['mode']}: ${update['volume']:.2f}, {update['trades']} trades")
                
                # Check if analytics are updating with trades
                if len(analytics_updates) > 1:
                    hour_updates = [u for u in analytics_updates if u['mode'] == 'hour']
                    day_updates = [u for u in analytics_updates if u['mode'] == 'day']
                    
                    if hour_updates:
                        first_hour = hour_updates[0]
                        last_hour = hour_updates[-1]
                        volume_change = last_hour['volume'] - first_hour['volume']
                        trades_change = last_hour['trades'] - first_hour['trades']
                        logger.info(f"\n🔄 HOUR MODE CHANGES: ${volume_change:.2f} volume, {trades_change} trades")
                    
                    if day_updates:
                        first_day = day_updates[0]
                        last_day = day_updates[-1]
                        volume_change = last_day['volume'] - first_day['volume']
                        trades_change = last_day['trades'] - first_day['trades']
                        logger.info(f"🔄 DAY MODE CHANGES: ${volume_change:.2f} volume, {trades_change} trades")
            
            logger.info("="*50)
                
    except Exception as e:
        logger.error(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket_updates())