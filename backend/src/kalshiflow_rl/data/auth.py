"""
Authentication wrapper for RL Trading Subsystem.

Provides RL-specific authentication utilities that leverage the existing
Kalshi authentication infrastructure from the main kalshiflow package.
"""

import logging
from typing import Dict

# Import existing authentication from main package
from kalshiflow.auth import KalshiAuth, KalshiAuthError

logger = logging.getLogger("kalshiflow_rl.auth")


class RLKalshiAuth:
    """
    RL-specific wrapper for Kalshi authentication.
    
    Uses the existing KalshiAuth implementation but provides
    RL-specific configuration and WebSocket authentication
    methods optimized for orderbook subscriptions.
    """
    
    def __init__(self):
        """Initialize RL auth using configuration."""
        try:
            # Use the same from_env method as the main app for consistency
            self.auth = KalshiAuth.from_env()
            logger.info("RL Kalshi authentication initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RL Kalshi auth: {e}")
            raise
    
    def create_websocket_headers(self) -> Dict[str, str]:
        """
        Create authentication headers for WebSocket handshake.
        
        Returns:
            Dict with authentication headers for WebSocket connection
        """
        try:
            # Use existing auth method for WebSocket authentication
            headers = self.auth.create_auth_headers("GET", "/trade-api/ws/v2")
            
            logger.debug("Created WebSocket authentication headers")
            return headers
            
        except Exception as e:
            logger.error(f"Failed to create WebSocket auth headers: {e}")
            raise KalshiAuthError(f"WebSocket auth header creation failed: {e}")
    
    def create_subscription_message(self, market_ticker: str) -> Dict[str, any]:
        """
        Create orderbook subscription message with authentication.
        
        Args:
            market_ticker: Market ticker to subscribe to (e.g., "INXD-25JAN03")
            
        Returns:
            Dict with subscription message for WebSocket
        """
        try:
            # Create the subscription message format expected by Kalshi
            subscription_message = {
                "id": f"orderbook_sub_{market_ticker}",
                "cmd": "subscribe",
                "params": {
                    "channels": [f"orderbook_delta.{market_ticker}"],
                    # Include authentication in subscription if needed
                    **self.auth.create_websocket_auth_message()
                }
            }
            
            logger.debug(f"Created subscription message for market: {market_ticker}")
            return subscription_message
            
        except Exception as e:
            logger.error(f"Failed to create subscription message for {market_ticker}: {e}")
            raise KalshiAuthError(f"Subscription message creation failed: {e}")
    
    def validate_auth(self) -> bool:
        """
        Validate that authentication is properly configured.
        
        Returns:
            bool: True if auth is valid, False otherwise
        """
        try:
            # Test signature generation
            test_headers = self.auth.create_auth_headers("GET", "/test")
            
            # Basic validation - check that required headers are present
            required_headers = ["KALSHI-ACCESS-KEY", "KALSHI-ACCESS-SIGNATURE", "KALSHI-ACCESS-TIMESTAMP"]
            for header in required_headers:
                if header not in test_headers:
                    logger.error(f"Missing required header: {header}")
                    return False
            
            # Check that values are not empty
            for header, value in test_headers.items():
                if not value or not isinstance(value, str):
                    logger.error(f"Invalid value for header {header}: {value}")
                    return False
            
            logger.info("Authentication validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Authentication validation failed: {e}")
            return False


# Global RL auth instance - initialized lazily
rl_auth = None


def get_rl_auth() -> RLKalshiAuth:
    """
    Get the global RL authentication instance.
    
    Returns:
        RLKalshiAuth instance
        
    Raises:
        KalshiAuthError: If auth is not properly initialized
    """
    global rl_auth
    if rl_auth is None:
        try:
            rl_auth = RLKalshiAuth()
        except Exception as e:
            logger.error(f"Failed to create RL auth instance: {e}")
            raise KalshiAuthError(f"RL authentication initialization failed: {e}")
    
    return rl_auth


def validate_rl_auth() -> bool:
    """
    Validate that RL authentication is working.
    
    Returns:
        bool: True if auth is valid and working
    """
    try:
        auth = get_rl_auth()
        return auth.validate_auth()
    except Exception as e:
        logger.error(f"RL auth validation failed: {e}")
        return False