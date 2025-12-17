#!/usr/bin/env python3
"""
Test script to directly query the Kalshi demo account balance.
This will help diagnose why the system sees $0 while the web shows $980.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kalshiflow_rl.trading.demo_client import KalshiDemoTradingClient
from kalshiflow_rl.config import config
from dotenv import load_dotenv

async def test_demo_balance():
    """Test querying the demo account balance directly."""
    
    # Load paper trading environment
    env_file = Path(__file__).parent / ".env.paper"
    if env_file.exists():
        load_dotenv(env_file, override=True)
        print(f"✅ Loaded environment from {env_file}")
    else:
        print(f"❌ Environment file not found: {env_file}")
        return
    
    # Show API URL being used
    api_url = os.getenv("KALSHI_API_URL", "")
    print(f"\nAPI URL: {api_url}")
    
    if "demo-api" not in api_url:
        print("❌ ERROR: Not using demo API URL. Please ensure ENVIRONMENT=paper")
        return
    
    # Create demo client
    print("\n📡 Connecting to Kalshi demo API...")
    client = KalshiDemoTradingClient()
    
    try:
        # Connect to demo API
        await client.connect()
        print("✅ Connected to demo API")
        
        # Get account info
        print("\n💰 Fetching account balance...")
        account_info = await client.get_account_info()
        
        print("\nRaw API Response:")
        print("-" * 50)
        import json
        print(json.dumps(account_info, indent=2))
        print("-" * 50)
        
        # Parse balance
        if "balance" in account_info:
            balance_cents = account_info["balance"]
            balance_dollars = balance_cents / 100.0
            print(f"\n💵 Balance: ${balance_dollars:.2f} (raw: {balance_cents} cents)")
        else:
            print("\n❌ No 'balance' field in response")
            print(f"Available fields: {list(account_info.keys())}")
        
        # Also try to get positions
        print("\n📊 Fetching positions...")
        positions_response = await client.get_positions()
        positions = positions_response.get("positions", [])
        
        if positions:
            print(f"Found {len(positions)} position(s):")
            for pos in positions:
                ticker = pos.get("ticker", "UNKNOWN")
                contracts = pos.get("position", 0)
                print(f"  - {ticker}: {contracts} contracts")
        else:
            print("No positions found")
        
        # Get recent fills
        print("\n📈 Fetching recent fills...")
        fills_response = await client.get_fills()
        fills = fills_response.get("fills", [])
        
        if fills:
            print(f"Found {len(fills)} recent fill(s):")
            for fill in fills[:5]:  # Show first 5
                ticker = fill.get("ticker", "UNKNOWN")
                side = fill.get("side", "")
                qty = fill.get("count", 0)
                price = fill.get("yes_price", fill.get("no_price", 0))
                print(f"  - {ticker}: {side} {qty} @ {price}¢")
        else:
            print("No recent fills found")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hasattr(client, 'disconnect'):
            await client.disconnect()
        print("\n✅ Connection closed")

if __name__ == "__main__":
    asyncio.run(test_demo_balance())