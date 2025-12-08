#!/usr/bin/env python3
"""
Test script to demonstrate the duplicate detection in action and show statistics.
"""

import asyncio
import json
from datetime import datetime

from kalshiflow.duplicate_detector import TradeDuplicateDetector
from kalshiflow.aggregator import TradeAggregator
from kalshiflow.models import Trade


async def test_duplicate_detection_with_stats():
    """Test duplicate detection and show comprehensive statistics."""
    print("🔍 Testing In-Memory Duplicate Detection")
    print("=" * 50)
    
    # Initialize components
    detector = TradeDuplicateDetector(window_minutes=5)
    aggregator = TradeAggregator()
    
    # Start services
    await detector.start()
    await aggregator.start()
    
    try:
        # Create a series of trades with some duplicates
        base_time = int(datetime.now().timestamp() * 1000)
        
        test_trades = [
            # Trade 1 - Unique
            Trade(
                market_ticker="PRES2024",
                yes_price=55, no_price=45,
                yes_price_dollars=0.55, no_price_dollars=0.45,
                count=100, taker_side="yes", ts=base_time
            ),
            # Trade 2 - Different trade (unique)
            Trade(
                market_ticker="PRES2024", 
                yes_price=56, no_price=44,
                yes_price_dollars=0.56, no_price_dollars=0.44,
                count=150, taker_side="no", ts=base_time + 1000
            ),
            # Trade 3 - Duplicate of Trade 1
            Trade(
                market_ticker="PRES2024",
                yes_price=55, no_price=45,
                yes_price_dollars=0.55, no_price_dollars=0.45,
                count=100, taker_side="yes", ts=base_time
            ),
            # Trade 4 - Different market (unique)
            Trade(
                market_ticker="SPORTS2024",
                yes_price=30, no_price=70,
                yes_price_dollars=0.30, no_price_dollars=0.70,
                count=200, taker_side="yes", ts=base_time + 2000
            ),
            # Trade 5 - Duplicate of Trade 2
            Trade(
                market_ticker="PRES2024",
                yes_price=56, no_price=44,
                yes_price_dollars=0.56, no_price_dollars=0.44,
                count=150, taker_side="no", ts=base_time + 1000
            ),
            # Trade 6 - Another duplicate of Trade 1
            Trade(
                market_ticker="PRES2024",
                yes_price=55, no_price=45,
                yes_price_dollars=0.55, no_price_dollars=0.45,
                count=100, taker_side="yes", ts=base_time
            ),
        ]
        
        print("📊 Processing test trades through aggregator...")
        print()
        
        processed_count = 0
        duplicate_count = 0
        
        for i, trade in enumerate(test_trades, 1):
            print(f"🔄 Trade {i}: {trade.market_ticker} ${trade.yes_price_dollars:.2f}/${trade.no_price_dollars:.2f} x{trade.count} ({trade.taker_side})")
            
            # Process through aggregator (which uses duplicate detector)
            result = aggregator.process_trade(trade)
            
            if result is None:
                print(f"   ❌ DUPLICATE DETECTED - Trade filtered")
                duplicate_count += 1
            else:
                print(f"   ✅ UNIQUE - Processed successfully")
                processed_count += 1
            
            print()
        
        print("📈 FINAL STATISTICS")
        print("=" * 50)
        print(f"Total trades: {len(test_trades)}")
        print(f"Processed (unique): {processed_count}")
        print(f"Duplicates filtered: {duplicate_count}")
        print(f"Duplicate rate: {(duplicate_count / len(test_trades)) * 100:.1f}%")
        print()
        
        # Show aggregator statistics including duplicate detection
        agg_stats = aggregator.get_stats()
        dd_stats = agg_stats["duplicate_detection"]
        
        print("🎯 DUPLICATE DETECTION STATISTICS")
        print("=" * 50)
        print(f"Total checks: {dd_stats['total_checks']}")
        print(f"Duplicates detected: {dd_stats['duplicates_detected']}")
        print(f"Unique trades: {dd_stats['unique_trades']}")
        print(f"Detection rate: {dd_stats['detection_rate_percent']:.1f}%")
        print(f"Cache size: {dd_stats['cache_size']} trade hashes")
        print(f"Window size: {dd_stats['window_minutes']} minutes")
        print()
        
        print("⚙️  AGGREGATOR STATISTICS") 
        print("=" * 50)
        print(f"Active tickers: {agg_stats['active_tickers']}")
        print(f"Recent trades count: {agg_stats['recent_trades_count']}")
        print(f"Total trades in memory: {agg_stats['total_trades_in_memory']}")
        print()
        
        # Show detector representation
        print(f"🔧 Detector state: {detector}")
        print()
        
        # Test hash uniqueness
        print("🧪 HASH UNIQUENESS TEST")
        print("=" * 50)
        
        trade1 = test_trades[0]
        trade2 = test_trades[1]  # Different trade
        trade3 = test_trades[2]  # Duplicate of trade1
        
        hash1 = detector._generate_trade_hash(trade1)
        hash2 = detector._generate_trade_hash(trade2)
        hash3 = detector._generate_trade_hash(trade3)
        
        print(f"Trade 1 hash: {hash1[:16]}...")
        print(f"Trade 2 hash: {hash2[:16]}...")
        print(f"Trade 3 hash: {hash3[:16]}... (should match Trade 1)")
        print(f"Hash 1 == Hash 3: {hash1 == hash3}")
        print(f"Hash 1 != Hash 2: {hash1 != hash2}")
        print()
        
        print("✅ DUPLICATE DETECTION TEST COMPLETED SUCCESSFULLY!")
        
    finally:
        await detector.stop()
        await aggregator.stop()


if __name__ == "__main__":
    asyncio.run(test_duplicate_detection_with_stats())