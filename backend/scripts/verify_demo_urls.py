#!/usr/bin/env python3
"""
Kalshi Demo URL Verification Script.

Tests both REST API and WebSocket connections to verify the demo account URLs are correct.
Includes fallback logic for domain mismatches between .com and .co.
"""

import asyncio
import logging
import sys
import os
import time
from typing import Dict, Any, Optional

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import aiohttp
import websockets

from kalshiflow_rl.config import config
from kalshiflow_rl.trading.demo_client import KalshiDemoTradingClient, KalshiDemoTradingClientError


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DemoURLVerifier:
    """Verifies Kalshi demo account URLs with fallback logic."""
    
    def __init__(self):
        self.rest_url = config.KALSHI_PAPER_TRADING_API_URL
        self.ws_url = config.KALSHI_PAPER_TRADING_WS_URL
        self.api_key_id = config.KALSHI_PAPER_TRADING_API_KEY_ID
        self.private_key_content = config.KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT
        
        # Alternative URLs for fallback testing
        self.rest_url_alt = self.rest_url.replace('.co/', '.com/') if '.co/' in self.rest_url else self.rest_url.replace('.com/', '.co/')
        self.ws_url_alt = self.ws_url.replace('.com/', '.co/') if '.com/' in self.ws_url else self.ws_url.replace('.co/', '.com/')
    
    async def verify_rest_api(self, url: str) -> Dict[str, Any]:
        """
        Test REST API connection to the given URL.
        
        Args:
            url: Base URL for REST API
            
        Returns:
            Verification result dictionary
        """
        logger.info(f"Testing REST API: {url}")
        
        result = {
            "url": url,
            "success": False,
            "status_code": None,
            "error": None,
            "response_time_ms": None,
            "endpoint_tested": "/markets"
        }
        
        try:
            # Test with unauthenticated markets endpoint first
            markets_url = f"{url}/markets?limit=1"
            
            start_time = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(markets_url) as response:
                    end_time = time.time()
                    result["response_time_ms"] = int((end_time - start_time) * 1000)
                    result["status_code"] = response.status
                    
                    if response.status == 200:
                        response_data = await response.json()
                        if "markets" in response_data:
                            result["success"] = True
                            markets_count = len(response_data.get("markets", []))
                            logger.info(f"✅ REST API working: {url} (got {markets_count} markets in {result['response_time_ms']}ms)")
                        else:
                            result["error"] = "Invalid response format - no markets data"
                    else:
                        response_text = await response.text()
                        result["error"] = f"HTTP {response.status}: {response_text[:200]}"
                        
        except aiohttp.ClientError as e:
            result["error"] = f"Connection error: {str(e)}"
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
        
        if not result["success"]:
            logger.error(f"❌ REST API failed: {url} - {result['error']}")
        
        return result
    
    async def verify_websocket(self, url: str) -> Dict[str, Any]:
        """
        Test WebSocket connection to the given URL.
        
        Args:
            url: WebSocket URL
            
        Returns:
            Verification result dictionary
        """
        logger.info(f"Testing WebSocket: {url}")
        
        result = {
            "url": url,
            "success": False,
            "error": None,
            "connection_time_ms": None,
            "ping_supported": False
        }
        
        try:
            start_time = time.time()
            
            # Try to connect without authentication first (should still establish connection)
            async with websockets.connect(url, ping_interval=None) as websocket:
                end_time = time.time()
                result["connection_time_ms"] = int((end_time - start_time) * 1000)
                
                # Test if ping/pong works
                try:
                    await websocket.ping()
                    result["ping_supported"] = True
                except:
                    pass  # Ping not supported, but connection works
                
                result["success"] = True
                logger.info(f"✅ WebSocket working: {url} (connected in {result['connection_time_ms']}ms)")
                
        except websockets.InvalidURI as e:
            result["error"] = f"Invalid WebSocket URI: {str(e)}"
        except websockets.ConnectionClosed as e:
            result["error"] = f"Connection closed: {str(e)}"
        except OSError as e:
            result["error"] = f"Network error: {str(e)}"
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
        
        if not result["success"]:
            logger.error(f"❌ WebSocket failed: {url} - {result['error']}")
        
        return result
    
    async def verify_authenticated_demo_client(self) -> Dict[str, Any]:
        """
        Test the full demo client with authentication.
        
        Returns:
            Verification result dictionary
        """
        logger.info("Testing authenticated demo client")
        
        result = {
            "success": False,
            "error": None,
            "capabilities": {
                "authentication": False,
                "markets_access": False,
                "portfolio_access": False
            },
            "account_balance": None,
            "markets_count": 0
        }
        
        try:
            # Test with current configuration
            async with KalshiDemoTradingClient(mode="paper") as client:
                # Test market access
                try:
                    markets_response = await client.get_markets(limit=5)
                    result["capabilities"]["markets_access"] = True
                    result["markets_count"] = len(markets_response.get("markets", []))
                    logger.info(f"✅ Markets access working (got {result['markets_count']} markets)")
                except Exception as e:
                    logger.warning(f"⚠️ Markets access failed: {e}")
                
                # Test portfolio access (may be limited in demo)
                try:
                    account_info = await client.get_account_info()
                    result["capabilities"]["portfolio_access"] = True
                    result["account_balance"] = str(client.balance)
                    logger.info(f"✅ Portfolio access working (balance: ${client.balance})")
                except Exception as e:
                    logger.warning(f"⚠️ Portfolio access limited (expected for demo): {e}")
                
                # Authentication worked if we got this far
                result["capabilities"]["authentication"] = True
                result["success"] = True
                logger.info("✅ Demo client authentication successful")
                
        except KalshiDemoTradingClientError as e:
            result["error"] = f"Demo client error: {str(e)}"
            logger.error(f"❌ Demo client failed: {e}")
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"❌ Unexpected demo client error: {e}")
        
        return result
    
    async def run_verification(self) -> Dict[str, Any]:
        """
        Run complete verification of demo URLs with fallback testing.
        
        Returns:
            Complete verification results
        """
        logger.info("🚀 Starting Kalshi demo URL verification...")
        logger.info(f"REST API URL: {self.rest_url}")
        logger.info(f"WebSocket URL: {self.ws_url}")
        logger.info(f"API Key ID: {self.api_key_id}")
        
        if not self.api_key_id or not self.private_key_content:
            logger.error("❌ Demo credentials not configured - check environment variables")
            return {"error": "Missing demo credentials"}
        
        results = {
            "timestamp": time.time(),
            "configuration": {
                "rest_url": self.rest_url,
                "ws_url": self.ws_url,
                "api_key_id": self.api_key_id,
                "has_private_key": bool(self.private_key_content)
            },
            "rest_api": {},
            "websocket": {},
            "demo_client": {},
            "recommendations": []
        }
        
        # Test primary REST API URL
        results["rest_api"]["primary"] = await self.verify_rest_api(self.rest_url)
        
        # Test alternative REST API URL if primary fails
        if not results["rest_api"]["primary"]["success"]:
            logger.info(f"Primary REST URL failed, testing alternative: {self.rest_url_alt}")
            results["rest_api"]["alternative"] = await self.verify_rest_api(self.rest_url_alt)
            
            if results["rest_api"]["alternative"]["success"]:
                results["recommendations"].append(
                    f"Use {self.rest_url_alt} instead of {self.rest_url} for REST API"
                )
        
        # Test primary WebSocket URL
        results["websocket"]["primary"] = await self.verify_websocket(self.ws_url)
        
        # Test alternative WebSocket URL if primary fails
        if not results["websocket"]["primary"]["success"]:
            logger.info(f"Primary WebSocket URL failed, testing alternative: {self.ws_url_alt}")
            results["websocket"]["alternative"] = await self.verify_websocket(self.ws_url_alt)
            
            if results["websocket"]["alternative"]["success"]:
                results["recommendations"].append(
                    f"Use {self.ws_url_alt} instead of {self.ws_url} for WebSocket"
                )
        
        # Test full demo client with authentication
        results["demo_client"] = await self.verify_authenticated_demo_client()
        
        # Generate summary
        rest_working = results["rest_api"]["primary"]["success"] or results["rest_api"].get("alternative", {}).get("success", False)
        ws_working = results["websocket"]["primary"]["success"] or results["websocket"].get("alternative", {}).get("success", False)
        auth_working = results["demo_client"]["success"]
        
        logger.info("\n" + "="*60)
        logger.info("📋 VERIFICATION SUMMARY")
        logger.info("="*60)
        logger.info(f"REST API:       {'✅ WORKING' if rest_working else '❌ FAILED'}")
        logger.info(f"WebSocket:      {'✅ WORKING' if ws_working else '❌ FAILED'}")
        logger.info(f"Authentication: {'✅ WORKING' if auth_working else '❌ FAILED'}")
        
        if results["recommendations"]:
            logger.info("\n🔧 RECOMMENDATIONS:")
            for rec in results["recommendations"]:
                logger.info(f"  • {rec}")
        
        if rest_working and ws_working and auth_working:
            logger.info("\n🎉 All systems working! Demo trading client is ready to use.")
        else:
            logger.info("\n⚠️ Some issues detected. Check the details above.")
        
        return results


async def main():
    """Main verification function."""
    verifier = DemoURLVerifier()
    results = await verifier.run_verification()
    
    # Exit with appropriate code
    if results.get("error"):
        sys.exit(1)
    
    rest_ok = results["rest_api"]["primary"]["success"] or results["rest_api"].get("alternative", {}).get("success", False)
    ws_ok = results["websocket"]["primary"]["success"] or results["websocket"].get("alternative", {}).get("success", False)
    
    if not (rest_ok and ws_ok):
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())