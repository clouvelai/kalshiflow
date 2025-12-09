#!/usr/bin/env python3
"""
Test script for verifying multi-market RL orderbook collector functionality.

Tests:
- Multi-market configuration parsing
- OrderbookClient initialization with multiple markets
- WebSocket manager functionality
- Statistics collector integration
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend src to path
backend_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_path))

# Set test environment variables
os.environ["RL_MARKET_TICKERS"] = "KXCABOUT-29,KXFEDDECISION-25DEC,KXLLM1-25DEC31"
os.environ["DATABASE_URL"] = os.getenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")

from kalshiflow_rl.config import config
from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.websocket_manager import websocket_manager
from kalshiflow_rl.stats_collector import stats_collector


async def test_config():
    """Test multi-market configuration parsing."""
    print("\n=== Testing Configuration ===")
    print(f"RL_MARKET_TICKERS environment: {os.getenv('RL_MARKET_TICKERS')}")
    print(f"Parsed market tickers: {config.RL_MARKET_TICKERS}")
    print(f"Number of markets: {len(config.RL_MARKET_TICKERS)}")
    
    assert len(config.RL_MARKET_TICKERS) == 3, "Should have 3 markets configured"
    assert "KXCABOUT-29" in config.RL_MARKET_TICKERS
    assert "KXFEDDECISION-25DEC" in config.RL_MARKET_TICKERS
    assert "KXLLM1-25DEC31" in config.RL_MARKET_TICKERS
    print("✅ Configuration test passed")


async def test_orderbook_client():
    """Test OrderbookClient multi-market initialization."""
    print("\n=== Testing OrderbookClient ===")
    
    # Test with explicit market list
    client = OrderbookClient(market_tickers=["TEST1", "TEST2", "TEST3"])
    assert len(client.market_tickers) == 3
    assert "TEST1" in client.market_tickers
    print("✅ Explicit market list works")
    
    # Test with config default
    client2 = OrderbookClient()
    assert len(client2.market_tickers) == 3
    assert client2.market_tickers == config.RL_MARKET_TICKERS
    print("✅ Config default works")
    
    # Test backward compatibility with single ticker
    client3 = OrderbookClient(market_tickers="SINGLE_TICKER")
    assert len(client3.market_tickers) == 1
    assert client3.market_tickers == ["SINGLE_TICKER"]
    print("✅ Backward compatibility works")
    
    print("✅ OrderbookClient test passed")


async def test_websocket_manager():
    """Test WebSocket manager initialization."""
    print("\n=== Testing WebSocketManager ===")
    
    # Check initialization
    assert websocket_manager._market_tickers == config.RL_MARKET_TICKERS
    print(f"Markets configured: {websocket_manager._market_tickers}")
    
    # Test stats integration
    websocket_manager.stats_collector = stats_collector
    stats = await websocket_manager._gather_stats()
    
    assert "markets_active" in stats
    # Markets active will be 0 initially since no snapshots have been processed
    # Just verify the field exists
    print(f"Stats gathered: {stats}")
    
    print("✅ WebSocketManager test passed")


async def test_stats_collector():
    """Test statistics collector functionality."""
    print("\n=== Testing StatsCollector ===")
    
    # Track some test data
    stats_collector.track_snapshot("TEST_MARKET1")
    stats_collector.track_snapshot("TEST_MARKET2")
    stats_collector.track_delta("TEST_MARKET1")
    
    stats = stats_collector.get_stats()
    
    assert stats["snapshots_processed"] == 2
    assert stats["deltas_processed"] == 1
    assert stats["markets_active"] == 2
    assert "TEST_MARKET1" in stats["per_market"]
    assert "TEST_MARKET2" in stats["per_market"]
    
    print(f"Stats: {stats}")
    print("✅ StatsCollector test passed")


async def main():
    """Run all tests."""
    print("Starting multi-market RL orderbook collector tests...")
    
    try:
        await test_config()
        await test_orderbook_client()
        await test_websocket_manager()
        await test_stats_collector()
        
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())