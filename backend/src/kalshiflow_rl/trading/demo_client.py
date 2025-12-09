"""
Kalshi Demo Trading Client for Paper Trading.

Provides a clean interface to the Kalshi demo account API for realistic
paper trading without affecting production systems. Uses the official
demo-api.kalshi.co endpoints for authentic order execution experience.
"""

import asyncio
import logging
import time
import base64
import json
import tempfile
import os
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal

import aiohttp
import websockets

# Import the proven authentication from main kalshiflow package
from kalshiflow.auth import KalshiAuth

from ..config import config

logger = logging.getLogger("kalshiflow_rl.trading.demo_client")


class KalshiDemoTradingClientError(Exception):
    """Base exception for Kalshi demo trading client errors."""
    pass


class KalshiDemoAuthError(KalshiDemoTradingClientError):
    """Authentication error with demo account."""
    pass


class KalshiDemoOrderError(KalshiDemoTradingClientError):
    """Order execution error on demo account."""
    pass


class KalshiDemoTradingClient:
    """
    Demo trading client for paper trading using Kalshi's official demo account.
    
    This client provides realistic order execution and position tracking using
    the official demo-api.kalshi.co endpoints. It's designed to be API-compatible
    with a production trading client to enable seamless paper→live transitions.
    
    Key Features:
    - Official demo API integration
    - Realistic order fills and latency
    - Complete position and balance tracking
    - Full error handling with informative messages
    - Credential isolation from production
    """
    
    def __init__(self, mode: str = "paper"):
        """
        Initialize demo trading client.
        
        Args:
            mode: Trading mode - must be "paper" for demo client
            
        Raises:
            ValueError: If mode is not "paper" 
            KalshiDemoAuthError: If demo credentials are missing
        """
        if mode != "paper":
            raise ValueError(f"KalshiDemoTradingClient only supports 'paper' mode, got: {mode}")
        
        # Validate demo credentials are configured
        if not config.KALSHI_PAPER_TRADING_API_KEY_ID:
            raise KalshiDemoAuthError("KALSHI_PAPER_TRADING_API_KEY_ID not configured")
        
        if not config.KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT:
            raise KalshiDemoAuthError("KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT not configured")
        
        self.mode = mode
        self.api_key_id = config.KALSHI_PAPER_TRADING_API_KEY_ID
        self.rest_base_url = config.KALSHI_PAPER_TRADING_API_URL
        self.ws_url = config.KALSHI_PAPER_TRADING_WS_URL
        
        # Initialize authentication using proven KalshiAuth class
        try:
            # Create a temporary environment with demo credentials for KalshiAuth
            self._setup_demo_auth_env()
            
            # Create KalshiAuth instance with demo credentials
            self.auth = KalshiAuth(
                api_key_id=self.api_key_id,
                private_key_path=self._temp_key_file
            )
            logger.info("Demo account authentication initialized successfully")
        except Exception as e:
            raise KalshiDemoAuthError(f"Failed to initialize demo auth: {e}")
        
        # Client state
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self.is_connected = False
        
        # Demo account state tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.balance: Decimal = Decimal("0.00")
        
        logger.info(f"KalshiDemoTradingClient initialized in {mode} mode")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    def _setup_demo_auth_env(self) -> None:
        """
        Set up temporary authentication environment for demo credentials.
        
        Creates a temporary private key file that KalshiAuth can use.
        """
        try:
            # Create temporary file for private key
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pem', prefix='kalshi_demo_key_')
            self._temp_key_file = temp_path
            
            with os.fdopen(temp_fd, 'w') as temp_file:
                # Ensure proper key format with line breaks
                private_key_content = config.KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT
                if not private_key_content.startswith('-----BEGIN'):
                    # Add PKCS8 headers if missing
                    formatted_key = f"-----BEGIN PRIVATE KEY-----\n{private_key_content}\n-----END PRIVATE KEY-----"
                else:
                    formatted_key = private_key_content
                    
                temp_file.write(formatted_key)
                
        except Exception as e:
            # Clean up temp file if creation fails
            if hasattr(self, '_temp_key_file'):
                try:
                    os.unlink(self._temp_key_file)
                except:
                    pass
            raise KalshiDemoAuthError(f"Failed to create temporary demo key file: {e}")
    
    def _cleanup_demo_auth_env(self) -> None:
        """Clean up temporary authentication files."""
        if hasattr(self, '_temp_key_file'):
            try:
                os.unlink(self._temp_key_file)
                logger.debug("Cleaned up temporary demo key file")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary demo key file: {e}")
    
    def _create_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Create authentication headers for demo API requests using proven KalshiAuth.
        
        Args:
            method: HTTP method
            path: API path (e.g., '/portfolio/balance')
            
        Returns:
            Dictionary of authentication headers
        """
        try:
            # Kalshi API requires the FULL path including /trade-api/v2 for signature generation
            # The endpoint path alone (e.g., '/portfolio/balance') is not sufficient
            full_signature_path = f"/trade-api/v2{path}"
            
            # Use the proven KalshiAuth method with the full path for signature
            headers = self.auth.create_auth_headers(method, full_signature_path)
            
            # Add content type for API requests
            headers['Content-Type'] = 'application/json'
            
            return headers
            
        except Exception as e:
            raise KalshiDemoAuthError(f"Failed to create auth headers: {e}")
    
    async def connect(self) -> None:
        """
        Connect to demo account API.
        
        Tests connection with public markets endpoint and validates
        demo account portfolio access.
        
        Raises:
            KalshiDemoAuthError: If authentication or connection fails
        """
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()
            
            # Test connection with markets endpoint (public access)
            # Demo accounts may have limited portfolio access
            await self.get_markets(limit=1)
            
            # Get account info - if this fails, the demo account is not properly configured
            await self.get_account_info()
            logger.info("Demo account with full portfolio access")
            
            self.is_connected = True
            logger.info("Demo account connection established")
            
        except Exception as e:
            if self.session:
                await self.session.close()
                self.session = None
            raise KalshiDemoAuthError(f"Failed to connect to demo account: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from demo account API."""
        if self.ws_connection:
            await self.ws_connection.close()
            self.ws_connection = None
        
        if self.session:
            await self.session.close()
            self.session = None
        
        # Clean up temporary authentication files
        self._cleanup_demo_auth_env()
        
        self.is_connected = False
        logger.info("Demo account disconnected")
    
    async def _make_request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated request to demo API.
        
        Args:
            method: HTTP method
            path: API path (without base URL)
            data: Request body data (for POST/PUT requests)
            
        Returns:
            Response JSON data
            
        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        if not self.session:
            raise KalshiDemoTradingClientError("Not connected to demo account")
        
        url = f"{self.rest_base_url}{path}"
        headers = self._create_auth_headers(method, path)
        
        try:
            async with self.session.request(method, url, headers=headers, json=data) as response:
                response_text = await response.text()
                
                if response.status >= 400:
                    error_msg = f"Demo API error {response.status}: {response_text}"
                    logger.error(f"Request failed: {error_msg}")
                    raise KalshiDemoTradingClientError(error_msg)
                
                # Parse JSON response
                try:
                    return json.loads(response_text) if response_text else {}
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON response: {response_text}")
                    raise KalshiDemoTradingClientError(f"Invalid JSON response from demo API")
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise KalshiDemoTradingClientError(f"HTTP request failed: {e}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get demo account information including balance.
        
        Returns:
            Account information including balance
        """
        try:
            response = await self._make_request("GET", "/portfolio/balance")
            
            # Update balance tracking - Kalshi returns balance in cents
            if "balance" in response:
                self.balance = Decimal(str(response["balance"])) / 100  # Convert cents to dollars
            elif "available_balance" in response:
                self.balance = Decimal(str(response["available_balance"])) / 100
            
            logger.debug(f"Demo account balance: ${self.balance}")
            return response
            
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get account info: {e}")
    
    async def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions on demo account.
        
        Returns:
            Dictionary of positions by market ticker
            
        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            response = await self._make_request("GET", "/portfolio/positions")
            
            # Update positions tracking
            self.positions = {}
            if "positions" in response:
                for position in response["positions"]:
                    ticker = position.get("ticker", "")
                    if ticker:
                        self.positions[ticker] = position
            
            logger.debug(f"Demo account has {len(self.positions)} positions")
            return response
            
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get positions: {e}")
    
    async def get_orders(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get open orders on demo account.
        
        Args:
            ticker: Optional market ticker to filter orders
            
        Returns:
            Dictionary of orders
            
        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            path = "/portfolio/orders"
            if ticker:
                path += f"?ticker={ticker}"
            
            response = await self._make_request("GET", path)
            
            # Update orders tracking
            if "orders" in response:
                for order in response["orders"]:
                    order_id = order.get("order_id", "")
                    if order_id:
                        self.orders[order_id] = order
            
            logger.debug(f"Demo account has {len(self.orders)} orders")
            return response
            
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get orders: {e}")
    
    async def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price: Optional[int] = None,
        type: str = "limit"
    ) -> Dict[str, Any]:
        """
        Create order on demo account.
        
        Args:
            ticker: Market ticker (e.g., "INXD-25JAN03")
            action: "buy" or "sell"
            side: "yes" or "no"
            count: Number of contracts
            price: Limit price in cents (1-99), None for market orders
            type: Order type ("limit" or "market")
            
        Returns:
            Order creation response
            
        Raises:
            KalshiDemoOrderError: If order creation fails
        """
        try:
            order_data = {
                "ticker": ticker,
                "action": action,
                "side": side,
                "count": count,
                "type": type
            }
            
            if price is not None:
                order_data["price"] = price
            
            logger.info(f"Creating demo order: {action} {count} {side} contracts of {ticker} @ {price}¢")
            
            response = await self._make_request("POST", "/portfolio/orders", order_data)
            
            # Track the order
            if "order" in response:
                order = response["order"]
                order_id = order.get("order_id", "")
                if order_id:
                    self.orders[order_id] = order
                    logger.info(f"Demo order created: {order_id}")
            
            return response
            
        except Exception as e:
            raise KalshiDemoOrderError(f"Failed to create order: {e}")
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel order on demo account.
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Cancellation response
            
        Raises:
            KalshiDemoOrderError: If cancellation fails
        """
        try:
            logger.info(f"Cancelling demo order: {order_id}")
            
            response = await self._make_request("DELETE", f"/portfolio/orders/{order_id}")
            
            # Remove from tracking
            if order_id in self.orders:
                del self.orders[order_id]
                logger.info(f"Demo order cancelled: {order_id}")
            
            return response
            
        except Exception as e:
            raise KalshiDemoOrderError(f"Failed to cancel order {order_id}: {e}")
    
    async def get_fills(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get trade fills on demo account.
        
        Args:
            ticker: Optional market ticker to filter fills
            
        Returns:
            Dictionary of fills
        """
        try:
            path = "/portfolio/fills"
            if ticker:
                path += f"?ticker={ticker}"
            
            response = await self._make_request("GET", path)
            
            logger.debug(f"Retrieved fills for demo account")
            return response
            
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get fills: {e}")
    
    async def get_markets(self, limit: int = 100) -> Dict[str, Any]:
        """
        Get available markets on demo account.
        
        Args:
            limit: Maximum number of markets to return
            
        Returns:
            Markets data
        """
        try:
            response = await self._make_request("GET", f"/markets?limit={limit}")
            
            markets_count = len(response.get("markets", []))
            logger.debug(f"Retrieved {markets_count} markets from demo account")
            return response
            
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get markets: {e}")
    
    async def connect_websocket(self) -> None:
        """
        Connect to demo account WebSocket for real-time updates.
        
        This provides user-specific streams like order fills, position updates, etc.
        Separate from the public orderbook WebSocket used for market data.
        """
        # Use proven KalshiAuth for WebSocket authentication
        ws_headers = self.auth.create_auth_headers("GET", "/trade-api/ws/v2")
        
        # Remove Content-Type header for WebSocket connection
        if 'Content-Type' in ws_headers:
            del ws_headers['Content-Type']
        
        try:
            logger.info(f"Connecting to demo WebSocket: {self.ws_url}")
            self.ws_connection = await websockets.connect(self.ws_url, extra_headers=ws_headers)
            logger.info("Demo account WebSocket connected successfully")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise KalshiDemoTradingClientError(f"Failed to connect WebSocket: {e}")
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """
        Get summary of demo trading state.
        
        Returns:
            Summary of positions, orders, and balance
        """
        return {
            "mode": self.mode,
            "connected": self.is_connected,
            "balance": str(self.balance),
            "positions_count": len(self.positions),
            "orders_count": len(self.orders),
            "positions": dict(self.positions),
            "orders": dict(self.orders)
        }
    
    def get_demo_limitations(self) -> Dict[str, Any]:
        """
        Get documentation of demo account capabilities.
        
        Returns:
            Dictionary describing demo account features and capabilities
        """
        return {
            "demo_account_capabilities": {
                "portfolio_balance": "✅ Full access to account balance",
                "portfolio_positions": "✅ Full access to position tracking",
                "portfolio_orders": "✅ Full access to order management",
                "order_creation": "✅ Full order creation capability",
                "order_cancellation": "✅ Full order cancellation capability",
                "order_fills": "✅ Full access to fill history",
                "markets_api": "✅ Full access to market data",
                "authentication": "✅ RSA signature authentication working",
                "websocket_support": "✅ Both orderbook and user streams"
            },
            "demo_features": {
                "api_compatibility": "100% compatible with production API",
                "realistic_execution": "Orders execute with realistic fills",
                "paper_balance": f"${self.balance} - Demo account balance",
                "position_tracking": "Full position and P&L tracking",
                "no_financial_risk": "Safe testing with virtual funds"
            },
            "recommended_usage": {
                "strategy_testing": "✅ Complete strategy validation before production",
                "order_flow_testing": "✅ Test all order types and edge cases",
                "risk_management": "✅ Validate position limits and controls",
                "production_ready": "✅ Seamless transition to live trading"
            }
        }
    
    def validate_mode(self, requested_mode: str) -> None:
        """
        Validate that the requested trading mode is allowed.
        
        Args:
            requested_mode: Mode to validate
            
        Raises:
            ValueError: If mode is not allowed
        """
        if requested_mode not in config.ALLOWED_TRADING_MODES:
            raise ValueError(f"Trading mode '{requested_mode}' not allowed. Allowed modes: {config.ALLOWED_TRADING_MODES}")
        
        if requested_mode != "paper":
            raise ValueError(f"KalshiDemoTradingClient only supports 'paper' mode, got: {requested_mode}")


# Factory function for creating demo trading client
async def create_demo_trading_client(mode: str = "paper") -> KalshiDemoTradingClient:
    """
    Create and connect a demo trading client.
    
    Args:
        mode: Trading mode (must be "paper")
        
    Returns:
        Connected KalshiDemoTradingClient instance
    """
    client = KalshiDemoTradingClient(mode=mode)
    await client.connect()
    return client