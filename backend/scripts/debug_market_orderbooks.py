#!/usr/bin/env python3
"""
Debug script to fetch market orderbooks via Kalshi REST API.
Verifies if markets have active orderbooks or are empty.

Usage:
    python scripts/debug_market_orderbooks.py [market_ticker1] [market_ticker2] ...
    
If no tickers provided, uses RL_MARKET_TICKERS from config.
"""

import asyncio
import json
import sys
import time
from typing import List, Dict, Any, Optional
import httpx

# Add the src directory to Python path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from kalshiflow_rl.data.auth import get_rl_auth
    from kalshiflow_rl.config import config
except ImportError:
    # Alternative import for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.kalshiflow_rl.data.auth import get_rl_auth
    from src.kalshiflow_rl.config import config


class KalshiOrderbookDebugger:
    """Debug tool for fetching Kalshi market orderbooks via REST API."""
    
    def __init__(self):
        self.rl_auth = get_rl_auth()
        # Get the underlying KalshiAuth object for REST API calls
        self.auth = self.rl_auth.auth
        self.base_url = "https://api.elections.kalshi.com/trade-api/v2"
        
    async def fetch_market_orderbook(self, market_ticker: str) -> Dict[str, Any]:
        """
        Fetch orderbook for a specific market via REST API.
        
        Args:
            market_ticker: The market ticker (e.g., 'KXCABOUT-29')
            
        Returns:
            Dictionary containing orderbook data or error info
        """
        url = f"{self.base_url}/markets/{market_ticker}/orderbook"
        
        try:
            # Create authenticated headers
            headers = self.auth.create_auth_headers("GET", f"/trade-api/v2/markets/{market_ticker}/orderbook")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10.0)
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "market_ticker": market_ticker,
                        "data": response.json(),
                        "status_code": response.status_code
                    }
                else:
                    return {
                        "success": False,
                        "market_ticker": market_ticker,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "status_code": response.status_code
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "market_ticker": market_ticker,
                "error": f"Request failed: {str(e)}",
                "status_code": None
            }
    
    def analyze_orderbook(self, orderbook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze orderbook data to determine activity level.
        
        Args:
            orderbook_data: Raw orderbook response from API
            
        Returns:
            Analysis summary with activity metrics
        """
        if not orderbook_data.get("success"):
            return {
                "is_empty": True,
                "analysis": "Failed to fetch orderbook",
                "error": orderbook_data.get("error")
            }
        
        data = orderbook_data.get("data", {})
        orderbook = data.get("orderbook", {})
        
        # Extract bid/ask data
        yes_bids = orderbook.get("yes", []) or []
        no_bids = orderbook.get("no", []) or []
        
        # Count levels and volume
        yes_levels = len(yes_bids) if isinstance(yes_bids, list) else 0
        no_levels = len(no_bids) if isinstance(no_bids, list) else 0
        total_levels = yes_levels + no_levels
        
        # Calculate total volume
        yes_volume = sum(level[1] if len(level) > 1 else 0 for level in yes_bids) if yes_bids else 0
        no_volume = sum(level[1] if len(level) > 1 else 0 for level in no_bids) if no_bids else 0
        total_volume = yes_volume + no_volume
        
        # Determine spread info
        best_yes_bid = max([level[0] for level in yes_bids], default=None) if yes_bids else None
        best_yes_ask = min([level[0] for level in yes_bids], default=None) if yes_bids else None
        best_no_bid = max([level[0] for level in no_bids], default=None) if no_bids else None
        best_no_ask = min([level[0] for level in no_bids], default=None) if no_bids else None
        
        return {
            "is_empty": total_levels == 0,
            "total_levels": total_levels,
            "yes_levels": yes_levels,
            "no_levels": no_levels,
            "total_volume": total_volume,
            "yes_volume": yes_volume,
            "no_volume": no_volume,
            "best_yes_bid": best_yes_bid,
            "best_yes_ask": best_yes_ask,
            "best_no_bid": best_no_bid,
            "best_no_ask": best_no_ask,
            "has_spread": best_yes_bid is not None or best_no_bid is not None,
            "analysis": "Empty orderbook" if total_levels == 0 else f"Active market: {total_levels} levels, {total_volume} volume"
        }
    
    def print_market_summary(self, market_ticker: str, analysis: Dict[str, Any], orderbook_data: Dict[str, Any]):
        """Print formatted summary for a market."""
        print(f"\n{'='*60}")
        print(f"MARKET: {market_ticker}")
        print(f"{'='*60}")
        
        if not orderbook_data.get("success"):
            print(f"❌ ERROR: {analysis.get('error', 'Unknown error')}")
            return
            
        if analysis["is_empty"]:
            print("📭 EMPTY ORDERBOOK")
            print("   No active bids or asks")
        else:
            print("📈 ACTIVE ORDERBOOK")
            print(f"   Total Levels: {analysis['total_levels']}")
            print(f"   Total Volume: {analysis['total_volume']:,}")
            print(f"   YES side: {analysis['yes_levels']} levels, {analysis['yes_volume']:,} volume")
            print(f"   NO side: {analysis['no_levels']} levels, {analysis['no_volume']:,} volume")
            
            if analysis['best_yes_bid']:
                print(f"   Best YES bid: {analysis['best_yes_bid']}¢")
            if analysis['best_no_bid']:
                print(f"   Best NO bid: {analysis['best_no_bid']}¢")
        
        # Show raw data sample
        data = orderbook_data.get("data", {})
        orderbook = data.get("orderbook", {})
        print(f"\nRaw orderbook keys: {list(orderbook.keys())}")
        
        yes_levels = orderbook.get("yes", []) or []
        no_levels = orderbook.get("no", []) or []
        yes_sample = yes_levels[:3]  # First 3 levels
        no_sample = no_levels[:3]   # First 3 levels
        if yes_sample:
            print(f"YES sample levels: {yes_sample}")
        if no_sample:
            print(f"NO sample levels: {no_sample}")
    
    async def debug_markets(self, market_tickers: List[str]):
        """
        Debug multiple markets and print comprehensive analysis.
        
        Args:
            market_tickers: List of market tickers to analyze
        """
        print(f"🔍 KALSHI ORDERBOOK DEBUGGER")
        print(f"Analyzing {len(market_tickers)} markets...")
        print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = []
        
        # Fetch orderbooks for all markets
        for ticker in market_tickers:
            print(f"\nFetching {ticker}...", end=" ", flush=True)
            orderbook_data = await self.fetch_market_orderbook(ticker)
            analysis = self.analyze_orderbook(orderbook_data)
            
            if orderbook_data.get("success"):
                status = "✅ ACTIVE" if not analysis["is_empty"] else "📭 EMPTY"
            else:
                status = "❌ ERROR"
            
            print(status)
            
            results.append({
                "ticker": ticker,
                "orderbook_data": orderbook_data,
                "analysis": analysis
            })
        
        # Print detailed analysis for each market
        for result in results:
            self.print_market_summary(
                result["ticker"], 
                result["analysis"], 
                result["orderbook_data"]
            )
        
        # Print summary table
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Market':<20} {'Status':<10} {'Levels':<8} {'Volume':<12} {'Analysis'}")
        print("-" * 80)
        
        for result in results:
            analysis = result["analysis"]
            ticker = result["ticker"]
            
            if not result["orderbook_data"].get("success"):
                status = "ERROR"
                levels = "N/A"
                volume = "N/A"
                summary = analysis.get("error", "Unknown error")[:30]
            else:
                status = "EMPTY" if analysis["is_empty"] else "ACTIVE"
                levels = str(analysis["total_levels"])
                volume = f"{analysis['total_volume']:,}" if analysis['total_volume'] else "0"
                summary = analysis["analysis"][:30]
            
            print(f"{ticker:<20} {status:<10} {levels:<8} {volume:<12} {summary}")
        
        # Recommendations
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print(f"{'='*80}")
        
        active_markets = [r for r in results if r["orderbook_data"].get("success") and not r["analysis"]["is_empty"]]
        empty_markets = [r for r in results if r["orderbook_data"].get("success") and r["analysis"]["is_empty"]]
        error_markets = [r for r in results if not r["orderbook_data"].get("success")]
        
        if active_markets:
            print(f"✅ {len(active_markets)} markets have active orderbooks - suitable for RL training")
            for result in active_markets:
                print(f"   • {result['ticker']}: {result['analysis']['total_levels']} levels")
        
        if empty_markets:
            print(f"📭 {len(empty_markets)} markets have empty orderbooks - unsuitable for RL training")
            for result in empty_markets:
                print(f"   • {result['ticker']}: No active orders")
        
        if error_markets:
            print(f"❌ {len(error_markets)} markets had fetch errors")
            for result in error_markets:
                print(f"   • {result['ticker']}: {result['analysis'].get('error', 'Unknown error')}")
        
        if not active_markets:
            print("\n🎯 NEXT STEPS:")
            print("   1. Find markets with active trading via Kalshi web interface")
            print("   2. Update RL_MARKET_TICKERS with active market tickers")
            print("   3. Re-run this script to verify orderbook activity")
            print("   4. Consider election markets, Fed decisions, or other high-volume categories")


async def main():
    """Main entry point."""
    # Get market tickers from command line or config
    if len(sys.argv) > 1:
        market_tickers = sys.argv[1:]
        print(f"Using command line tickers: {market_tickers}")
    else:
        market_tickers = config.RL_MARKET_TICKERS
        print(f"Using RL_MARKET_TICKERS from config: {market_tickers}")
    
    if not market_tickers:
        print("❌ No market tickers provided. Either:")
        print("   1. Set RL_MARKET_TICKERS environment variable")
        print("   2. Pass tickers as command line arguments")
        print("   Example: python debug_market_orderbooks.py KXCABOUT-29 KXFEDDECISION-25DEC")
        sys.exit(1)
    
    debugger = KalshiOrderbookDebugger()
    await debugger.debug_markets(market_tickers)


if __name__ == "__main__":
    asyncio.run(main())