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

from ...config import config as rl_config
config = rl_config

logger = logging.getLogger("kalshiflow_rl.traderv3.clients.demo_client")


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
        if not config.KALSHI_API_KEY_ID:
            raise KalshiDemoAuthError("KALSHI_API_KEY_ID not configured. Use ENVIRONMENT=paper with .env.paper file")
        
        if not config.KALSHI_PRIVATE_KEY_CONTENT:
            raise KalshiDemoAuthError("KALSHI_PRIVATE_KEY_CONTENT not configured. Use ENVIRONMENT=paper with .env.paper file")
        
        self.mode = mode
        self.api_key_id = config.KALSHI_API_KEY_ID
        self.rest_base_url = config.KALSHI_API_URL
        self.ws_url = config.KALSHI_WS_URL
        
        # Validate that URLs point to demo API, not production
        if "api.elections.kalshi.com" in self.rest_base_url:
            raise KalshiDemoAuthError(
                f"Demo client cannot use production API URL: {self.rest_base_url}. "
                "Use ENVIRONMENT=paper with .env.paper file containing demo-api.kalshi.co URLs"
            )
        
        if "api.elections.kalshi.com" in self.ws_url:
            raise KalshiDemoAuthError(
                f"Demo client cannot use production WebSocket URL: {self.ws_url}. "
                "Use ENVIRONMENT=paper with .env.paper file containing demo-api.kalshi.co URLs"
            )
        
        # Ensure URLs point to demo API
        if "demo-api.kalshi.co" not in self.rest_base_url:
            raise KalshiDemoAuthError(
                f"Demo client must use demo-api.kalshi.co API URL, got: {self.rest_base_url}. "
                "Use ENVIRONMENT=paper with .env.paper file"
            )
        
        if "demo-api.kalshi.co" not in self.ws_url:
            raise KalshiDemoAuthError(
                f"Demo client must use demo-api.kalshi.co WebSocket URL, got: {self.ws_url}. "
                "Use ENVIRONMENT=paper with .env.paper file"
            )
        
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
                private_key_content = config.KALSHI_PRIVATE_KEY_CONTENT
                
                if not private_key_content:
                    raise KalshiDemoAuthError("KALSHI_PRIVATE_KEY_CONTENT is empty or None")
                
                if not private_key_content.startswith('-----BEGIN'):
                    # Add PKCS8 headers if missing
                    formatted_key = f"-----BEGIN PRIVATE KEY-----\n{private_key_content}\n-----END PRIVATE KEY-----"
                else:
                    # Key already has headers - ensure proper PEM format with newlines
                    # Handle case where newlines might be lost by dotenv
                    formatted_key = private_key_content
                    
                    # Replace escaped newlines with actual newlines (dotenv may escape them)
                    formatted_key = formatted_key.replace('\\n', '\n')
                    
                    # If key doesn't have newlines but has headers, normalize the format
                    # PEM format requires: BEGIN line\ncontent\nEND line
                    if '\n' not in formatted_key and '-----BEGIN' in formatted_key:
                        # Find BEGIN and END markers
                        begin_marker = '-----BEGIN'
                        end_marker = '-----END'
                        begin_idx = formatted_key.find(begin_marker)
                        end_idx = formatted_key.find(end_marker)
                        
                        if begin_idx != -1 and end_idx != -1:
                            # Find where BEGIN line ends (after the second '-----')
                            begin_end = formatted_key.find('-----', begin_idx + len(begin_marker))
                            if begin_end != -1:
                                begin_end += 5  # Length of '-----'
                                # Extract content between headers
                                content = formatted_key[begin_end:end_idx].strip()
                                # Reconstruct with proper newlines
                                formatted_key = (
                                    formatted_key[:begin_end] + '\n' +
                                    content + '\n' +
                                    formatted_key[end_idx:]
                                )
                    
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
    
    def _validate_balance_response(self, response: Dict[str, Any]) -> None:
        """
        Validate balance/portfolio API response structure.
        
        Args:
            response: Response from /portfolio/balance endpoint
            
        Raises:
            ValueError: If response structure is invalid or missing required fields
        """
        if not isinstance(response, dict):
            raise ValueError(f"Balance response must be a dictionary, got {type(response)}")
        
        # Must have both balance and portfolio_value (per Kalshi API docs)
        if "balance" not in response:
            raise ValueError("Balance response missing required 'balance' field")
        
        if "portfolio_value" not in response:
            raise ValueError("Balance response missing required 'portfolio_value' field")
        
        # Validate types - both must be integers (cents)
        if not isinstance(response["balance"], (int, float)):
            raise ValueError(f"Balance must be numeric, got {type(response['balance'])}: {response['balance']}")
        
        if not isinstance(response["portfolio_value"], (int, float)):
            raise ValueError(f"Portfolio value must be numeric, got {type(response['portfolio_value'])}: {response['portfolio_value']}")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get demo account information including balance and portfolio_value.
        
        Returns:
            Account information including balance and portfolio_value
            
        Raises:
            KalshiDemoTradingClientError: If request fails
            ValueError: If response structure is invalid
        """
        try:
            response = await self._make_request("GET", "/portfolio/balance")
            
            # Validate response structure strictly
            self._validate_balance_response(response)
            
            # Update balance tracking - Kalshi returns balance in cents
            self.balance = Decimal(str(response["balance"])) / 100  # Convert cents to dollars
            
            logger.debug(f"Demo account balance: ${self.balance}, portfolio_value: {response.get('portfolio_value', 'N/A')}")
            return response
            
        except ValueError as e:
            raise KalshiDemoTradingClientError(f"Invalid balance response structure: {e}")
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get account info: {e}")
    
    def _validate_positions_response(self, response: Dict[str, Any]) -> None:
        """
        Validate positions API response structure.
        
        Args:
            response: Response from /portfolio/positions endpoint
            
        Raises:
            ValueError: If response structure is invalid
        """
        if not isinstance(response, dict):
            raise ValueError(f"Positions response must be a dictionary, got {type(response)}")
        
        # Must have either "positions" or "market_positions" array (can be empty)
        positions_key = None
        if "positions" in response:
            positions_key = "positions"
        elif "market_positions" in response:
            positions_key = "market_positions"
        else:
            raise ValueError("Positions response missing required 'positions' or 'market_positions' field")
        
        # Validate it's a list/array
        if not isinstance(response[positions_key], list):
            raise ValueError(f"Positions field must be a list, got {type(response[positions_key])}")
    
    async def get_positions(self) -> Dict[str, Any]:
        """
        Get current positions on demo account.
        
        Returns:
            Dictionary of positions by market ticker
            
        Raises:
            KalshiDemoTradingClientError: If request fails
            ValueError: If response structure is invalid
        """
        try:
            response = await self._make_request("GET", "/portfolio/positions")
            
            # Validate response structure
            self._validate_positions_response(response)
            
            # Update positions tracking
            self.positions = {}
            positions_list = response.get("positions", response.get("market_positions", []))
            for position in positions_list:
                ticker = position.get("ticker", "")
                if ticker:
                    self.positions[ticker] = position
            
            logger.debug(f"Demo account has {len(self.positions)} positions")
            return response
            
        except ValueError as e:
            raise KalshiDemoTradingClientError(f"Invalid positions response structure: {e}")
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get positions: {e}")
    
    def _validate_orders_response(self, response: Dict[str, Any]) -> None:
        """
        Validate orders API response structure.
        
        Args:
            response: Response from /portfolio/orders endpoint
            
        Raises:
            ValueError: If response structure is invalid
        """
        if not isinstance(response, dict):
            raise ValueError(f"Orders response must be a dictionary, got {type(response)}")
        
        # Must have "orders" array (can be empty)
        if "orders" not in response:
            raise ValueError("Orders response missing required 'orders' field")
        
        # Validate it's a list/array
        if not isinstance(response["orders"], list):
            raise ValueError(f"Orders field must be a list, got {type(response['orders'])}")
    
    async def get_orders(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """
        Get open orders on demo account.
        
        Args:
            ticker: Optional market ticker to filter orders
            
        Returns:
            Dictionary of orders
            
        Raises:
            KalshiDemoTradingClientError: If request fails
            ValueError: If response structure is invalid
        """
        try:
            path = "/portfolio/orders"
            if ticker:
                path += f"?ticker={ticker}"
            
            response = await self._make_request("GET", path)
            
            # Validate response structure
            self._validate_orders_response(response)
            
            # Update orders tracking
            for order in response["orders"]:
                order_id = order.get("order_id", "")
                if order_id:
                    self.orders[order_id] = order
            
            logger.debug(f"Demo account has {len(self.orders)} orders")
            return response
            
        except ValueError as e:
            raise KalshiDemoTradingClientError(f"Invalid orders response structure: {e}")
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get orders: {e}")
    
    async def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price: Optional[int] = None,
        type: str = "limit",
        order_group_id: Optional[str] = None
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
            order_group_id: Optional order group ID for portfolio limits
            
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
            
            # Add order group if provided
            if order_group_id:
                order_data["order_group_id"] = order_group_id
            
            # Kalshi API requires specific price field names based on contract side
            if price is not None:
                if side.lower() == "yes":
                    order_data["yes_price"] = price
                elif side.lower() == "no":
                    order_data["no_price"] = price
                else:
                    raise KalshiDemoOrderError(f"Invalid contract side: {side}. Must be 'yes' or 'no'")
            
            if order_group_id:
                logger.info(f"Creating demo order: {action} {count} {side} contracts of {ticker} @ {price}¢ (group: {order_group_id[:8]}...)")
            else:
                logger.info(f"Creating demo order: {action} {count} {side} contracts of {ticker} @ {price}¢ (no portfolio limits)")
            
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
    
    async def batch_cancel_orders(self, order_ids: List[str]) -> Dict[str, Any]:
        """
        Cancel multiple orders in batch.
        
        According to Kalshi API docs, this should use POST /portfolio/orders/batched
        with a list of order IDs to cancel.
        
        Args:
            order_ids: List of order IDs to cancel
            
        Returns:
            Dictionary with cancellation results
            
        Raises:
            KalshiDemoOrderError: If batch cancellation fails
        """
        if not order_ids:
            return {"cancelled": [], "errors": [], "total": 0}
            
        try:
            logger.info(f"Batch cancelling {len(order_ids)} demo orders")
            
            # Try batch cancel endpoint first
            try:
                request_data = {
                    "order_ids": order_ids
                }
                
                response = await self._make_request(
                    "POST", 
                    "/portfolio/orders/batched",
                    json=request_data
                )
                
                # Process response
                cancelled = response.get("cancelled", [])
                errors = response.get("errors", [])
                
                # Update local tracking
                for cancelled_id in cancelled:
                    if cancelled_id in self.orders:
                        del self.orders[cancelled_id]
                
                logger.info(f"Batch cancel complete: {len(cancelled)} cancelled, {len(errors)} errors")
                
                return {
                    "cancelled": cancelled,
                    "errors": errors,
                    "total": len(order_ids),
                    "success_count": len(cancelled),
                    "error_count": len(errors)
                }
                
            except Exception as batch_error:
                # If batch endpoint fails, fall back to individual cancellations
                logger.warning(f"Batch cancel endpoint failed: {batch_error}. Falling back to individual cancellations")

                cancelled = []
                already_gone = []
                errors = []

                for order_id in order_ids:
                    try:
                        await self.cancel_order(order_id)
                        cancelled.append(order_id)
                    except Exception as e:
                        error_str = str(e).lower()
                        # 404 means order is already gone - count as success
                        if "404" in error_str or "not found" in error_str:
                            already_gone.append(order_id)
                            # Remove from local tracking if present
                            if order_id in self.orders:
                                del self.orders[order_id]
                        else:
                            errors.append({"order_id": order_id, "error": str(e)})

                # Combine cancelled and already_gone as successful outcomes
                all_cleared = cancelled + already_gone
                logger.info(
                    f"Individual cancel fallback complete: {len(cancelled)} cancelled, "
                    f"{len(already_gone)} already gone, {len(errors)} errors"
                )
                
                return {
                    "cancelled": all_cleared,  # Include both cancelled and already_gone
                    "errors": errors,
                    "total": len(order_ids),
                    "success_count": len(all_cleared),
                    "error_count": len(errors),
                    "fallback_used": True,
                    "already_gone": len(already_gone)
                }
                
        except Exception as e:
            raise KalshiDemoOrderError(f"Failed to batch cancel orders: {e}")
    
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
    
    async def get_settlements(self) -> Dict[str, Any]:
        """
        Get all settlements from demo account.
        
        According to Kalshi API docs: https://docs.kalshi.com/api-reference/portfolio/get-settlements
        All fields except fee_cost are in cents. fee_cost is a string in dollars.
        
        Returns:
            Dictionary with settlements array and cursor for pagination
            
        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            # Simple path without query parameters - fetch all settlements
            path = "/portfolio/settlements"
            
            logger.info(f"Fetching all settlements from demo account (same client as balance/positions)")
            response = await self._make_request("GET", path)
            
            # Validate response structure
            if not isinstance(response, dict):
                raise ValueError(f"Settlements response must be a dictionary, got {type(response)}")
            
            if "settlements" not in response:
                raise ValueError("Settlements response missing required 'settlements' field")
            
            if not isinstance(response["settlements"], list):
                raise ValueError(f"Settlements field must be a list, got {type(response['settlements'])}")
            
            settlements_count = len(response.get("settlements", []))
            logger.debug(f"Retrieved {settlements_count} settlements from demo account")
            
            return response
            
        except ValueError as e:
            raise KalshiDemoTradingClientError(f"Invalid settlements response structure: {e}")
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get settlements: {e}")
    
    # ===================
    # Order Group Methods
    # ===================
    
    async def create_order_group(self, contracts_limit: int = 10000) -> Dict[str, Any]:
        """
        Create an order group for portfolio limit management.
        
        API only accepts contracts_limit parameter per docs.
        
        Args:
            contracts_limit: Maximum number of contracts (default 10000)
            
        Returns:
            Order group response with order_group_id
            
        Raises:
            KalshiDemoTradingClientError: If creation fails
        """
        try:
            data = {
                "contracts_limit": contracts_limit
            }
            
            response = await self._make_request("POST", "/portfolio/order_groups/create", data)
            
            if not response.get("order_group_id"):
                raise ValueError(f"No order_group_id in response: {response}")
            
            logger.info(f"Created order group: {response['order_group_id'][:8]}... "
                       f"(contracts_limit: {contracts_limit})")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to create order group: {e}")
            raise KalshiDemoTradingClientError(f"Failed to create order group: {e}")
    
    async def get_order_group(self, order_group_id: str) -> Dict[str, Any]:
        """
        Get order group status and usage.
        
        Args:
            order_group_id: UUID of the order group
            
        Returns:
            Order group details including current usage
            
        Raises:
            KalshiDemoTradingClientError: If fetch fails
        """
        try:
            response = await self._make_request("GET", f"/portfolio/order_groups/{order_group_id}")
            
            logger.debug(f"Order group {order_group_id[:8]}... status: "
                        f"position={response.get('current_absolute_position', 0)}/{response.get('max_absolute_position', 0)}, "
                        f"orders={response.get('current_open_orders', 0)}/{response.get('max_open_orders', 0)}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get order group {order_group_id[:8]}...: {e}")
            raise KalshiDemoTradingClientError(f"Failed to get order group: {e}")
    
    async def update_order_group(self, order_group_id: str, contracts_limit: int) -> Dict[str, Any]:
        """
        Update order group limits.
        
        Args:
            order_group_id: UUID of the order group
            contracts_limit: New maximum number of contracts
            
        Returns:
            Updated order group details
            
        Raises:
            KalshiDemoTradingClientError: If update fails
        """
        try:
            data = {
                "contracts_limit": contracts_limit
            }
            
            response = await self._make_request("PATCH", f"/portfolio/order_groups/{order_group_id}", data)
            
            logger.info(f"Updated order group {order_group_id[:8]}... to: "
                       f"contracts_limit={contracts_limit}")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to update order group {order_group_id[:8]}...: {e}")
            raise KalshiDemoTradingClientError(f"Failed to update order group: {e}")
    
    async def reset_order_group(self, order_group_id: str) -> Dict[str, Any]:
        """
        Reset an order group (clears positions and cancels orders).
        
        According to API docs, use POST /portfolio/order_groups/{id}/reset.
        
        Args:
            order_group_id: UUID of the order group
            
        Returns:
            Reset confirmation
            
        Raises:
            KalshiDemoTradingClientError: If reset fails
        """
        try:
            response = await self._make_request("POST", f"/portfolio/order_groups/{order_group_id}/reset")
            
            logger.info(f"Reset order group {order_group_id[:8]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to reset order group {order_group_id[:8]}...: {e}")
            raise KalshiDemoTradingClientError(f"Failed to reset order group: {e}")
    
    # Backward compatibility alias
    async def close_order_group(self, order_group_id: str) -> Dict[str, Any]:
        """Alias for reset_order_group for backward compatibility."""
        return await self.reset_order_group(order_group_id)
    
    async def list_order_groups(self, status: Optional[str] = None) -> Dict[str, Any]:
        """
        List all order groups for the account.
        
        Args:
            status: Optional filter by status (active, closed)
            
        Returns:
            List of order groups
            
        Raises:
            KalshiDemoTradingClientError: If listing fails
        """
        try:
            params = {}
            if status:
                params["status"] = status
            
            # Build query string if params exist
            query_string = "&".join([f"{k}={v}" for k, v in params.items()]) if params else ""
            path = f"/portfolio/order_groups{'?' + query_string if query_string else ''}"
            
            response = await self._make_request("GET", path)
            
            # Demo API returns {} when no groups, normalize to expected format
            if response == {}:
                response = {"order_groups": []}
            
            groups_count = len(response.get("order_groups", []))
            logger.debug(f"Retrieved {groups_count} order groups" + (f" with status={status}" if status else ""))
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to list order groups: {e}")
            raise KalshiDemoTradingClientError(f"Failed to list order groups: {e}")
    
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
            self.ws_connection = await websockets.connect(self.ws_url, additional_headers=ws_headers)
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