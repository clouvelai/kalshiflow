"""
RSA signature authentication for Kalshi API using private key file.

Based on the reference implementation:
https://github.com/clouvelai/prophete/blob/main/backend/app/core/auth.py
"""

import os
import time
import base64
import logging
from pathlib import Path
from typing import Dict, Optional

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)


class KalshiAuthError(Exception):
    """Custom exception for authentication errors."""
    pass


class RSASigner:
    """RSA signature generator using private key file."""
    
    def __init__(self, private_key_path: str):
        """
        Initialize RSA signer with private key file.
        
        Args:
            private_key_path: Path to the RSA private key file (.pem)
        """
        self.private_key_path = Path(private_key_path)
        self._private_key: Optional[rsa.RSAPrivateKey] = None
        self._load_private_key()
    
    def _load_private_key(self) -> None:
        """Load and validate the RSA private key from file."""
        if not self.private_key_path.exists():
            raise KalshiAuthError(f"Private key file not found: {self.private_key_path}")
        
        try:
            with open(self.private_key_path, "rb") as key_file:
                private_key_data = key_file.read()
                
            # Try to load the key (with or without passphrase)
            self._private_key = serialization.load_pem_private_key(
                private_key_data,
                password=None,  # Assuming no passphrase for now
            )
            
            # Verify it's an RSA key
            if not isinstance(self._private_key, rsa.RSAPrivateKey):
                raise KalshiAuthError("Private key must be an RSA key")
                
            logger.info(f"Successfully loaded RSA private key from {self.private_key_path}")
            
        except Exception as e:
            raise KalshiAuthError(f"Failed to load private key: {e}")
    
    def sign(self, message: str) -> str:
        """
        Sign a message using RSA with PSS padding and SHA256.
        
        Args:
            message: String message to sign
            
        Returns:
            Base64-encoded signature string
        """
        if self._private_key is None:
            raise KalshiAuthError("Private key not loaded")
        
        try:
            # Convert message to bytes
            message_bytes = message.encode('utf-8')
            
            # Sign with RSA-PSS padding and SHA256
            signature = self._private_key.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Encode as base64
            signature_b64 = base64.b64encode(signature).decode('utf-8')
            logger.debug(f"Generated signature for message: {message[:50]}...")
            
            return signature_b64
            
        except Exception as e:
            raise KalshiAuthError(f"Failed to sign message: {e}")


class KalshiAuth:
    """Kalshi API authentication handler using RSA signatures."""
    
    def __init__(self, api_key_id: str, private_key_path: str):
        """
        Initialize Kalshi authentication.
        
        Args:
            api_key_id: The Kalshi API key ID
            private_key_path: Path to the RSA private key file
        """
        self.api_key_id = api_key_id
        self.signer = RSASigner(private_key_path)
        logger.info(f"Initialized Kalshi auth with API key: {api_key_id}")
    
    @classmethod
    def from_env(cls) -> 'KalshiAuth':
        """
        Create KalshiAuth instance from environment variables.
        
        Required environment variables:
        - KALSHI_API_KEY_ID: The API key ID
        - KALSHI_PRIVATE_KEY_PATH: Path to RSA private key file
        
        Returns:
            KalshiAuth instance
        """
        api_key_id = os.getenv("KALSHI_API_KEY_ID")
        private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")
        
        if not api_key_id:
            raise KalshiAuthError("KALSHI_API_KEY_ID environment variable is required")
        if not private_key_path:
            raise KalshiAuthError("KALSHI_PRIVATE_KEY_PATH environment variable is required")
            
        return cls(api_key_id, private_key_path)
    
    def create_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """
        Create authentication headers for Kalshi API request.
        
        The signature is generated from: timestamp_ms + method + path
        
        Args:
            method: HTTP method (e.g., 'GET', 'POST')
            path: Request path (e.g., '/trade-api/ws/v2')
            
        Returns:
            Dictionary of authentication headers
        """
        # Generate timestamp in milliseconds
        timestamp_ms = str(int(time.time() * 1000))
        
        # Create signature payload: timestamp + method + path
        sig_payload = timestamp_ms + method + path
        
        # Sign the payload
        signature = self.signer.sign(sig_payload)
        
        # Create headers
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }
        
        logger.debug(f"Created auth headers for {method} {path}")
        return headers
    
    def create_websocket_auth_message(self) -> Dict[str, str]:
        """
        Create WebSocket authentication message.
        
        Returns:
            Dictionary containing authentication data for WebSocket
        """
        # For WebSocket, we use an empty path typically
        return self.create_auth_headers("GET", "/")