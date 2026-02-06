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
            self._setup_demo_auth_env()
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
        """Set up temporary authentication environment for demo credentials."""
        from .auth_utils import setup_kalshi_auth

        try:
            self.auth, self._temp_key_file = setup_kalshi_auth(prefix="kalshi_demo_key_")
        except Exception as e:
            raise KalshiDemoAuthError(f"Failed to create temporary demo key file: {e}")

    def _cleanup_demo_auth_env(self) -> None:
        """Clean up temporary authentication files."""
        from .auth_utils import cleanup_kalshi_auth

        cleanup_kalshi_auth(getattr(self, '_temp_key_file', None))
        self._temp_key_file = None
    
    def _create_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Create authentication headers for demo API requests using proven KalshiAuth.

        Args:
            method: HTTP method
            path: API path (e.g., '/portfolio/balance' or '/portfolio/settlements?limit=200')

        Returns:
            Dictionary of authentication headers
        """
        try:
            # Strip query parameters for signature - Kalshi signature is computed on base path only
            # The URL will still include query params, but signature excludes them
            base_path = path.split('?')[0]

            # Kalshi API requires the FULL path including /trade-api/v2 for signature generation
            # The endpoint path alone (e.g., '/portfolio/balance') is not sufficient
            full_signature_path = f"/trade-api/v2{base_path}"

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

        Tests connection with exchange status endpoint and validates
        demo account portfolio access.

        Raises:
            KalshiDemoAuthError: If authentication or connection fails
        """
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession()

            # Test connection with exchange status (lightweight, always available)
            status = await self.get_exchange_status()
            if not status.get("exchange_active"):
                logger.warning("Exchange is not active - trading may be limited")

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

        Includes a single retry with 1-second backoff for 502 errors, which
        are common transient failures on the demo API.

        Args:
            method: HTTP method
            path: API path (without base URL)
            data: Request body data (for POST/PUT/DELETE requests)

        Returns:
            Response JSON data

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        if not self.session:
            raise KalshiDemoTradingClientError("Not connected to demo account")

        url = f"{self.rest_base_url}{path}"

        max_attempts = 2  # 1 initial + 1 retry for 502
        for attempt in range(1, max_attempts + 1):
            headers = self._create_auth_headers(method, path)

            try:
                async with self.session.request(method, url, headers=headers, json=data) as response:
                    response_text = await response.text()

                    # Retry once on 502 Bad Gateway (common transient demo API error)
                    if response.status == 502 and attempt < max_attempts:
                        logger.warning(f"Demo API 502 on {method} {path} (attempt {attempt}/{max_attempts}), retrying in 1s")
                        await asyncio.sleep(1)
                        continue

                    if response.status >= 400:
                        error_msg = f"Demo API error {response.status}: {response_text}"
                        if response.status == 502:
                            logger.warning(f"Demo API 502 on {method} {path} after {attempt} attempt(s): {response_text[:200]}")
                        else:
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

        # Should not be reached, but safety net
        raise KalshiDemoTradingClientError(f"Request failed after {max_attempts} attempts")
    
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
    
    async def get_exchange_status(self) -> Dict[str, Any]:
        """
        Get exchange status (lightweight connectivity check).

        GET /trade-api/v2/exchange/status

        Returns:
            Dict with exchange_active, trading_active, exchange_estimated_resume_time
        """
        try:
            response = await self._make_request("GET", "/exchange/status")
            logger.debug(
                f"Exchange status: active={response.get('exchange_active')}, "
                f"trading={response.get('trading_active')}"
            )
            return response
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get exchange status: {e}")

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
    
    async def get_orders(self, ticker: Optional[str] = None, status: str = "resting") -> Dict[str, Any]:
        """
        Get orders on demo account.

        Args:
            ticker: Optional market ticker to filter orders
            status: Order status filter - "resting" (open), "canceled", or "executed"
                    Default is "resting" to only get open orders

        Returns:
            Dictionary of orders (filtered by status)

        Raises:
            KalshiDemoTradingClientError: If request fails
            ValueError: If response structure is invalid
        """
        try:
            # Build path - only use ticker filter in API call
            # Status filter is applied locally because demo API has signature issues with it
            path = "/portfolio/orders"
            if ticker:
                path += f"?ticker={ticker}"

            response = await self._make_request("GET", path)

            # Validate response structure
            self._validate_orders_response(response)

            # Filter by status locally (API returns all orders, we want only specified status)
            all_orders = response.get("orders", [])
            if status:
                filtered_orders = [
                    order for order in all_orders
                    if order.get("status") == status
                ]
            else:
                filtered_orders = all_orders

            # Clear and rebuild orders tracking from filtered orders (API is source of truth)
            self.orders.clear()
            for order in filtered_orders:
                order_id = order.get("order_id", "")
                if order_id:
                    self.orders[order_id] = order

            logger.debug(f"Demo account has {len(self.orders)} {status} orders (filtered from {len(all_orders)} total)")

            # Return response with filtered orders for consistency
            return {"orders": filtered_orders}
            
        except ValueError as e:
            raise KalshiDemoTradingClientError(f"Invalid orders response structure: {e}")
        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get orders: {e}")

    async def get_queue_positions(
        self,
        market_tickers: Optional[str] = None,
        event_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get queue positions for all resting orders.

        Queue position represents the number of contracts ahead of yours
        that need to be matched before your order gets filled.

        Args:
            market_tickers: Comma-separated list of market tickers to filter by
            event_ticker: Event ticker to filter by

        Returns:
            Dictionary with queue_positions list

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            path = "/portfolio/orders/queue_positions"
            params = []
            if market_tickers:
                params.append(f"market_tickers={market_tickers}")
            if event_ticker:
                params.append(f"event_ticker={event_ticker}")
            if params:
                path += "?" + "&".join(params)

            response = await self._make_request("GET", path)
            return response

        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get queue positions: {e}")

    async def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price: Optional[int] = None,
        type: str = "limit",
        order_group_id: Optional[str] = None,
        expiration_ts: Optional[int] = None,
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
            expiration_ts: Optional Unix timestamp in seconds for auto-cancellation.
                          Kalshi cancels the order when this timestamp passes.

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

            # Add expiration timestamp for auto-cancel (Kalshi native TTL)
            if expiration_ts is not None:
                order_data["expiration_ts"] = expiration_ts
            
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
        Cancel multiple orders individually.

        The demo API (demo-api.kalshi.co) does not support the batch cancel
        endpoint (DELETE /portfolio/orders returns 404). This method skips the
        batch attempt entirely and cancels each order individually.

        Args:
            order_ids: List of order IDs to cancel

        Returns:
            Dictionary with cancellation results

        Raises:
            KalshiDemoOrderError: If all cancellations fail
        """
        if not order_ids:
            return {"cancelled": [], "errors": [], "total": 0}

        try:
            logger.info(f"Cancelling {len(order_ids)} demo orders individually (batch not supported on demo API)")

            cancelled = []
            already_gone = []
            errors = []

            for order_id in order_ids:
                try:
                    await self.cancel_order(order_id)
                    cancelled.append(order_id)
                except Exception as e:
                    error_str = str(e).lower()
                    # 404 means order is already gone (filled or cancelled) - count as success
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
                f"Individual cancel complete: {len(cancelled)} cancelled, "
                f"{len(already_gone)} already gone, {len(errors)} errors"
            )

            return {
                "cancelled": all_cleared,
                "errors": errors,
                "total": len(order_ids),
                "success_count": len(all_cleared),
                "error_count": len(errors),
                "already_gone": len(already_gone),
            }

        except Exception as e:
            raise KalshiDemoOrderError(f"Failed to cancel orders: {e}")
    
    async def get_fills(
        self,
        ticker: Optional[str] = None,
        order_group_id: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get trade fills on demo account.

        Args:
            ticker: Optional market ticker to filter fills
            order_group_id: Optional order group ID to filter fills
            limit: Maximum number of fills to return (default 100)

        Returns:
            Dictionary of fills
        """
        try:
            params = []
            if ticker:
                params.append(f"ticker={ticker}")
            if order_group_id:
                params.append(f"order_group_id={order_group_id}")
            if limit != 100:
                params.append(f"limit={limit}")

            path = "/portfolio/fills"
            if params:
                path += "?" + "&".join(params)

            response = await self._make_request("GET", path)

            logger.debug(f"Retrieved fills for demo account")
            return response

        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get fills: {e}")
    
    async def get_markets(
        self,
        limit: int = 100,
        tickers: Optional[List[str]] = None,
        status: Optional[str] = None,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get available markets on demo account.

        Args:
            limit: Maximum number of markets to return
            tickers: Optional list of specific tickers to fetch
            status: Optional market status filter ('unopened', 'open', 'closed', 'settled')
            event_ticker: Optional event ticker to filter by
            series_ticker: Optional series ticker to filter by

        Returns:
            Markets data with market details including bid/ask prices and close_time
        """
        try:
            # Build query parameters
            params = [f"limit={limit}"]
            if tickers:
                params.append(f"tickers={','.join(tickers)}")
            if status:
                params.append(f"status={status}")
            if event_ticker:
                params.append(f"event_ticker={event_ticker}")
            if series_ticker:
                params.append(f"series_ticker={series_ticker}")

            query_string = "&".join(params)
            response = await self._make_request("GET", f"/markets?{query_string}")

            markets_count = len(response.get("markets", []))
            logger.debug(f"Retrieved {markets_count} markets from demo account")
            return response

        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get markets: {e}")

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        """
        Get a single market by ticker.

        GET /trade-api/v2/markets/{ticker}

        Args:
            ticker: Market ticker (e.g., "INXD-25JAN03")

        Returns:
            Market data including ticker, title, status, etc.
            Note: category field is often empty - use get_event() to get category

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            response = await self._make_request("GET", f"/markets/{ticker}")
            logger.debug(f"Retrieved market {ticker}")
            return response

        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get market {ticker}: {e}")

    async def get_orderbook(self, ticker: str, depth: int = 5) -> Dict[str, Any]:
        """
        Fetch orderbook via REST API (fallback for stale WebSocket data).

        GET /trade-api/v2/markets/{ticker}/orderbook

        This is used as a fallback when the WebSocket orderbook data is stale
        (>5 seconds old) at signal execution time. Rate limit: 10/s.

        Args:
            ticker: Market ticker (e.g., "INXD-25JAN03")
            depth: Number of price levels to return (1-100, 0 for all). Default 5.

        Returns:
            Dict with "orderbook" containing:
            - yes: Array of [price, count] pairs for YES bids
            - no: Array of [price, count] pairs for NO bids
            Note: A bid for YES at price X equals an ask for NO at price (100-X)

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            path = f"/markets/{ticker}/orderbook"
            if depth > 0:
                path += f"?depth={depth}"

            response = await self._make_request("GET", path)
            logger.debug(f"Retrieved orderbook for {ticker} (depth={depth})")
            return response

        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get orderbook for {ticker}: {e}")

    async def get_market_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> Dict[str, Any]:
        """
        Get candlestick OHLC data for a market.

        GET /trade-api/v2/series/{series_ticker}/markets/{ticker}/candlesticks

        Use this to get the true market open price from the first candlestick's
        yes_bid.open value.

        Args:
            series_ticker: Series ticker (e.g., "INXD" - extracted from market ticker)
            ticker: Market ticker (e.g., "INXD-25JAN03")
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            period_interval: Candle period in minutes - 1 (1-min), 60 (1-hour), 1440 (1-day)

        Returns:
            Dict with "ticker" and "candlesticks" array. Each candlestick has:
            - end_period_ts: Unix timestamp for period end
            - yes_bid: OHLC for YES buy offers (open, high, low, close)
            - yes_ask: OHLC for YES sell offers
            - price: Trade price OHLC
            - volume: Contracts traded
            - open_interest: Total contracts by period end

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            params = [
                f"start_ts={start_ts}",
                f"end_ts={end_ts}",
                f"period_interval={period_interval}",
            ]
            query_string = "&".join(params)
            path = f"/series/{series_ticker}/markets/{ticker}/candlesticks?{query_string}"

            response = await self._make_request("GET", path)

            candlesticks_count = len(response.get("candlesticks", []))
            logger.debug(f"Retrieved {candlesticks_count} candlesticks for {ticker}")
            return response

        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get candlesticks for {ticker}: {e}")

    async def get_event(self, event_ticker: str) -> Dict[str, Any]:
        """
        Get event details by event_ticker, including nested markets.

        GET /trade-api/v2/events/{event_ticker}

        Events contain the category field that is often empty in market responses.
        The response also includes a "markets" array with full market data including
        yes_sub_title, no_sub_title, subtitle, and rules_primary fields.

        Note: The market "title" field is DEPRECATED in Kalshi API. Use yes_sub_title
        as the primary source for market/candidate names.

        Args:
            event_ticker: Event ticker (e.g., "KXNFL-25JAN05")

        Returns:
            Event data including category field and markets array attached

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            response = await self._make_request("GET", f"/events/{event_ticker}")
            logger.debug(f"Retrieved event {event_ticker}")

            # Return event with markets array attached
            # The API returns "event" object + "markets" array at top level
            event = response.get("event", {})
            event["markets"] = response.get("markets", [])
            return event

        except Exception as e:
            logger.error(f"Failed to get event {event_ticker}: {e}")
            return {}

    async def get_event_candlesticks(
        self,
        series_ticker: str,
        event_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> Dict[str, Any]:
        """
        Get candlestick OHLC data for ALL markets in an event at once.

        GET /trade-api/v2/series/{series_ticker}/events/{event_ticker}/candlesticks

        This is more efficient than fetching candlesticks per-market because it
        returns data for all markets in a single API call.

        Args:
            series_ticker: Series ticker (e.g., "KXPRESNOMD")
            event_ticker: Event ticker (e.g., "KXPRESNOMD-28")
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)
            period_interval: Candle period in minutes - 1 (1-min), 60 (1-hour), 1440 (1-day)

        Returns:
            Dict with:
            - market_tickers: List of market tickers
            - market_candlesticks: List of candlestick arrays (one per market)
            Each candlestick has:
            - end_period_ts: Unix timestamp for period end
            - yes_bid: OHLC for YES buy offers
            - yes_ask: OHLC for YES sell offers
            - price: Trade price OHLC
            - volume: Contracts traded
            - open_interest: Total contracts by period end

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            params = [
                f"start_ts={start_ts}",
                f"end_ts={end_ts}",
                f"period_interval={period_interval}",
            ]
            query_string = "&".join(params)
            path = f"/series/{series_ticker}/events/{event_ticker}/candlesticks?{query_string}"

            response = await self._make_request("GET", path)

            market_tickers = response.get("market_tickers", [])
            market_candlesticks = response.get("market_candlesticks", [])
            total_candles = sum(len(c) for c in market_candlesticks if c)

            logger.debug(
                f"Retrieved {total_candles} candlesticks across "
                f"{len(market_tickers)} markets for event {event_ticker}"
            )
            return response

        except Exception as e:
            raise KalshiDemoTradingClientError(
                f"Failed to get event candlesticks for {event_ticker}: {e}"
            )

    async def get_series(
        self,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get series list with optional category filter.

        GET /trade-api/v2/series

        The series endpoint supports category as a first-class query parameter,
        making it the correct way to discover markets by category.

        Args:
            category: Filter series by category (e.g., "Politics", "Economics")

        Returns:
            List of series dicts with series_ticker, title, category, etc.

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            params = []
            if category:
                params.append(f"category={category}")

            query_string = "&".join(params) if params else ""
            path = f"/series?{query_string}" if query_string else "/series"
            response = await self._make_request("GET", path)

            series_list = response.get("series", [])
            logger.debug(f"Retrieved {len(series_list)} series (category={category})")
            return series_list

        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get series: {e}")

    async def get_events(
        self,
        status: Optional[str] = None,
        with_nested_markets: bool = False,
        limit: int = 200,
        cursor: Optional[str] = None,
        min_close_ts: Optional[int] = None,
        series_ticker: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get events with optional filtering and pagination.

        GET /trade-api/v2/events

        This is the efficient batch endpoint for fetching multiple events
        with their markets in a single call.

        Args:
            status: Filter by 'open', 'closed', or 'settled'
            with_nested_markets: Include markets nested in each event
            limit: Max results per page (1-200, default 200)
            cursor: Pagination cursor from previous response
            min_close_ts: Filter events with at least one market closing after this Unix timestamp
            series_ticker: Filter events by series ticker (API-level filter)

        Returns:
            {"events": [...], "cursor": "..."} where cursor is empty if no more pages

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            params = [f"limit={limit}"]
            if status:
                params.append(f"status={status}")
            if with_nested_markets:
                params.append("with_nested_markets=true")
            if cursor:
                params.append(f"cursor={cursor}")
            if min_close_ts:
                params.append(f"min_close_ts={min_close_ts}")
            if series_ticker:
                params.append(f"series_ticker={series_ticker}")

            query_string = "&".join(params)
            response = await self._make_request("GET", f"/events?{query_string}")

            events_count = len(response.get("events", []))
            has_more = bool(response.get("cursor"))
            logger.debug(
                f"Retrieved {events_count} events "
                f"(status={status}, nested_markets={with_nested_markets}, "
                f"series_ticker={series_ticker}, has_more={has_more})"
            )
            return response

        except Exception as e:
            raise KalshiDemoTradingClientError(f"Failed to get events: {e}")

    async def get_settlements(self, max_settlements: int = 500) -> Dict[str, Any]:
        """
        Get all settlements from demo account with pagination.

        According to Kalshi API docs: https://docs.kalshi.com/api-reference/portfolio/get-settlements
        - limit: max 200 per request (default 100)
        - cursor: pagination cursor for next page
        - All fields except fee_cost are in cents. fee_cost is a string in dollars.

        Args:
            max_settlements: Maximum total settlements to fetch (default 500)

        Returns:
            Dictionary with settlements array (paginated results combined)

        Raises:
            KalshiDemoTradingClientError: If request fails
        """
        try:
            all_settlements = []
            cursor = None
            page = 1

            logger.info(f"Fetching settlements from demo account (max {max_settlements})")

            while len(all_settlements) < max_settlements:
                # Build path with query parameters
                path = "/portfolio/settlements?limit=200"
                if cursor:
                    path += f"&cursor={cursor}"

                response = await self._make_request("GET", path)

                # Validate response structure
                if not isinstance(response, dict):
                    raise ValueError(f"Settlements response must be a dictionary, got {type(response)}")

                if "settlements" not in response:
                    raise ValueError("Settlements response missing required 'settlements' field")

                if not isinstance(response["settlements"], list):
                    raise ValueError(f"Settlements field must be a list, got {type(response['settlements'])}")

                page_settlements = response.get("settlements", [])
                all_settlements.extend(page_settlements)

                logger.debug(f"Page {page}: fetched {len(page_settlements)} settlements (total: {len(all_settlements)})")

                # Check for more pages
                cursor = response.get("cursor")
                if not cursor or len(page_settlements) == 0:
                    break  # No more pages

                page += 1

            # Trim to max if we exceeded
            if len(all_settlements) > max_settlements:
                all_settlements = all_settlements[:max_settlements]

            logger.info(f"Retrieved {len(all_settlements)} total settlements from demo account ({page} pages)")

            return {"settlements": all_settlements, "cursor": cursor}

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

        According to Kalshi API docs, use PUT /portfolio/order_groups/{id}/reset.

        Args:
            order_group_id: UUID of the order group

        Returns:
            Reset confirmation

        Raises:
            KalshiDemoTradingClientError: If reset fails
        """
        try:
            response = await self._make_request("PUT", f"/portfolio/order_groups/{order_group_id}/reset")

            logger.info(f"Reset order group {order_group_id[:8]}...")

            return response

        except Exception as e:
            logger.error(f"Failed to reset order group {order_group_id[:8]}...: {e}")
            raise KalshiDemoTradingClientError(f"Failed to reset order group: {e}")

    async def delete_order_group(self, order_group_id: str) -> Dict[str, Any]:
        """Delete an order group (Kalshi API: DELETE /portfolio/order_groups/{id})."""
        try:
            response = await self._make_request("DELETE", f"/portfolio/order_groups/{order_group_id}")
            logger.info(f"Deleted order group {order_group_id[:8]}...")
            return response
        except Exception as e:
            logger.error(f"Failed to delete order group {order_group_id[:8]}...: {e}")
            raise KalshiDemoTradingClientError(f"Failed to delete order group: {e}")
    
    # Backward compatibility alias
    async def close_order_group(self, order_group_id: str) -> Dict[str, Any]:
        """Alias for reset_order_group for backward compatibility."""
        return await self.reset_order_group(order_group_id)
    
    async def list_order_groups(self, status: Optional[str] = None) -> Dict[str, Any]:
        """
        List all order groups for the account.

        Note: The Kalshi API does not support status filtering. The status
        parameter is accepted for API compatibility but ignored.

        Returns:
            List of order groups with id and is_auto_cancel_enabled

        Raises:
            KalshiDemoTradingClientError: If listing fails
        """
        try:
            # Note: Kalshi API has no query parameters for this endpoint
            # We fetch all groups and caller can filter if needed
            if status:
                logger.debug(f"Note: status={status} filter requested but API does not support filtering")

            response = await self._make_request("GET", "/portfolio/order_groups")

            # Demo API returns {} when no groups, normalize to expected format
            if response == {}:
                response = {"order_groups": []}

            groups_count = len(response.get("order_groups", []))
            logger.debug(f"Retrieved {groups_count} order groups")

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