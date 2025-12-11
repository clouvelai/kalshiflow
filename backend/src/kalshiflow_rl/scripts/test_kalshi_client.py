#!/usr/bin/env python3
"""
Standalone script to test Kalshi WebSocket client functionality.

This script demonstrates the Kalshi client by connecting, authenticating,
subscribing to public trades, and printing trade messages to stdout.

Usage:
    uv run backend/scripts/test_kalshi_client.py

Environment variables required:
    KALSHI_API_KEY_ID: Your Kalshi API key ID
    KALSHI_PRIVATE_KEY_PATH: Path to your RSA private key file
    KALSHI_WS_URL: WebSocket URL (optional, defaults to production URL)
"""

import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path so we can import kalshiflow
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalshiflow.kalshi_client import KalshiWebSocketClient
from kalshiflow.models import Trade, ConnectionStatus


def setup_logging():
    """Configure logging for the test script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific loggers
    logging.getLogger('websockets').setLevel(logging.WARNING)
    logging.getLogger('kalshiflow.auth').setLevel(logging.INFO)
    logging.getLogger('kalshiflow.kalshi_client').setLevel(logging.INFO)


def validate_environment():
    """Validate that required environment variables are set."""
    required_vars = ['KALSHI_API_KEY_ID', 'KALSHI_PRIVATE_KEY_PATH']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nRequired environment variables:")
        print("  KALSHI_API_KEY_ID: Your Kalshi API key ID")
        print("  KALSHI_PRIVATE_KEY_PATH: Path to your RSA private key file (.pem)")
        print("  KALSHI_WS_URL: WebSocket URL (optional)")
        print("\nExample:")
        print("  export KALSHI_API_KEY_ID=your_api_key_here")
        print("  export KALSHI_PRIVATE_KEY_PATH=/path/to/your/private_key.pem")
        return False
    
    # Check if private key file exists
    private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')
    if not Path(private_key_path).exists():
        print(f"Error: Private key file not found: {private_key_path}")
        return False
    
    return True


class TradeCounter:
    """Simple trade counter and display helper."""
    
    def __init__(self):
        self.trade_count = 0
        self.markets = set()
        self.start_time = datetime.now()
    
    def add_trade(self, trade: Trade):
        """Add a trade to the counter."""
        self.trade_count += 1
        self.markets.add(trade.market_ticker)
    
    def get_stats(self) -> str:
        """Get current statistics as a string."""
        elapsed = datetime.now() - self.start_time
        elapsed_seconds = elapsed.total_seconds()
        
        if elapsed_seconds > 0:
            rate = self.trade_count / elapsed_seconds * 60  # trades per minute
            return f"Trades: {self.trade_count}, Markets: {len(self.markets)}, Rate: {rate:.1f}/min"
        else:
            return f"Trades: {self.trade_count}, Markets: {len(self.markets)}"


def on_trade_received(trade: Trade, counter: TradeCounter):
    """Handle incoming trade messages."""
    counter.add_trade(trade)
    
    # Format trade information
    timestamp = trade.timestamp.strftime("%H:%M:%S")
    ticker = trade.market_ticker[:30]  # Truncate long tickers
    side = trade.taker_side.upper()
    side_color = "🟢" if side == "YES" else "🔴"
    price = trade.price_display
    count = trade.count
    
    print(f"[{timestamp}] {side_color} {ticker:<30} {side:<3} {price:<6} x{count:<4} | {counter.get_stats()}")


def on_connection_status_changed(status: ConnectionStatus):
    """Handle connection status changes."""
    if status.connected:
        print("✅ Connected to Kalshi WebSocket")
    else:
        if status.error_message:
            print(f"❌ Disconnected: {status.error_message}")
        else:
            print("❌ Disconnected from Kalshi WebSocket")
        
        if status.reconnect_attempts > 0:
            print(f"🔄 Reconnect attempts: {status.reconnect_attempts}")


async def run_test_client():
    """Run the test client."""
    print("🚀 Starting Kalshi WebSocket Client Test")
    print("=" * 80)
    
    # Validate environment
    if not validate_environment():
        return 1
    
    # Setup trade counter
    counter = TradeCounter()
    
    # Create client
    try:
        client = KalshiWebSocketClient.from_env(
            on_trade_callback=lambda trade: on_trade_received(trade, counter),
            on_connection_change=on_connection_status_changed,
        )
        
        print(f"📡 Connecting to: {client.websocket_url}")
        print(f"🔑 Using API key: {client.auth.api_key_id}")
        print("📊 Trade data will appear below when received...")
        print("-" * 80)
        
        # Setup graceful shutdown
        shutdown_event = asyncio.Event()
        
        def signal_handler():
            print("\n🛑 Shutdown signal received, stopping client...")
            shutdown_event.set()
        
        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
        
        # Start client
        client_task = asyncio.create_task(client.start())
        
        try:
            # Wait for shutdown signal
            await shutdown_event.wait()
        except KeyboardInterrupt:
            print("\n🛑 Keyboard interrupt received, stopping client...")
        
        # Stop client gracefully
        await client.stop()
        
        # Wait for client task to complete
        try:
            await asyncio.wait_for(client_task, timeout=5.0)
        except asyncio.TimeoutError:
            print("⚠️ Client shutdown timed out")
            client_task.cancel()
        
        print("\n" + "=" * 80)
        print(f"📈 Final Statistics: {counter.get_stats()}")
        print("✅ Test completed successfully")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error running test client: {e}")
        logging.exception("Error in test client")
        return 1


def main():
    """Main entry point."""
    setup_logging()
    
    try:
        exit_code = asyncio.run(run_test_client())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.exception("Unexpected error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()