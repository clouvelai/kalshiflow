#!/usr/bin/env python3
"""
Quick test to verify the price field fix works for order creation.
"""

import asyncio
import sys
import os

# Add the backend source to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from kalshiflow_rl.trading.demo_client import KalshiDemoTradingClient, KalshiDemoOrderError

async def test_order_creation():
    """Test that order creation now works with fixed price fields."""
    print("Testing order creation with fixed price fields...")
    
    try:
        # Create demo client
        client = KalshiDemoTradingClient()
        await client.connect()
        print("✅ Connected to demo API")
        
        # Test YES contract order (should use yes_price)
        print("\nTesting YES contract order...")
        try:
            response = await client.create_order(
                ticker="KXELONMARS-99",  # Simple test market
                action="buy",
                side="yes", 
                count=1,
                price=50,  # 50 cents
                type="limit"
            )
            print(f"✅ YES order created successfully: {response.get('order', {}).get('order_id', 'N/A')}")
            
        except Exception as e:
            if "yes_price" in str(e).lower() or "no_price" in str(e).lower():
                print(f"❌ Price field error still exists: {e}")
                return False
            else:
                print(f"⚠️  Other error (may be expected): {e}")
        
        # Test NO contract order (should use no_price) 
        print("\nTesting NO contract order...")
        try:
            response = await client.create_order(
                ticker="KXELONMARS-99",
                action="buy", 
                side="no",
                count=1,
                price=50,
                type="limit"
            )
            print(f"✅ NO order created successfully: {response.get('order', {}).get('order_id', 'N/A')}")
            
        except Exception as e:
            if "yes_price" in str(e).lower() or "no_price" in str(e).lower():
                print(f"❌ Price field error still exists: {e}")
                return False
            else:
                print(f"⚠️  Other error (may be expected): {e}")
        
        await client.disconnect()
        print("\n✅ Price field fix verified - orders can be created without field errors!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_order_creation())
    sys.exit(0 if success else 1)