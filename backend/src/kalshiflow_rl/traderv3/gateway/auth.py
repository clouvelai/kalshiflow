"""Unified auth wrapper for KalshiGateway.

Thin wrapper around the existing KalshiAuth class. Handles temporary
PEM file lifecycle and provides headers for both REST and WS auth.
"""

import logging
import os
import tempfile
from typing import Dict, Optional

from ..clients.auth_utils import setup_kalshi_auth, cleanup_kalshi_auth, format_private_key

logger = logging.getLogger("kalshiflow_rl.traderv3.gateway.auth")


class GatewayAuth:
    """Manages Kalshi authentication for the gateway.

    Owns the temporary PEM file lifecycle and delegates signature
    generation to the proven KalshiAuth class.
    """

    def __init__(
        self,
        api_key_id: Optional[str] = None,
        private_key_content: Optional[str] = None,
    ):
        self._override_api_key_id = api_key_id
        self._override_private_key_content = private_key_content
        self._auth = None
        self._temp_key_file: Optional[str] = None

    def setup(self) -> None:
        """Initialize auth credentials. Creates temp PEM file.

        When override credentials were provided at construction time, uses
        those instead of the global config — this is how the hybrid-mode
        prod gateway authenticates with production Kalshi.
        """
        if self._override_api_key_id and self._override_private_key_content:
            from kalshiflow.auth import KalshiAuth

            temp_fd, temp_path = tempfile.mkstemp(suffix=".pem", prefix="kalshi_gw_prod_")
            try:
                with os.fdopen(temp_fd, "w") as f:
                    f.write(format_private_key(self._override_private_key_content))
                self._auth = KalshiAuth(
                    api_key_id=self._override_api_key_id,
                    private_key_path=temp_path,
                )
                self._temp_key_file = temp_path
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
            logger.info("Gateway auth initialized (override credentials)")
        else:
            self._auth, self._temp_key_file = setup_kalshi_auth(prefix="kalshi_gw_")
            logger.info("Gateway auth initialized")

    def cleanup(self) -> None:
        """Remove temporary PEM file."""
        cleanup_kalshi_auth(self._temp_key_file)
        self._temp_key_file = None
        self._auth = None

    def rest_headers(self, method: str, path: str) -> Dict[str, str]:
        """Create auth headers for a REST request.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            path: API path WITHOUT /trade-api/v2 prefix and WITHOUT query params.
                  e.g. "/portfolio/balance" or "/portfolio/orders"

        Returns:
            Dict of auth headers including Content-Type.
        """
        if not self._auth:
            raise RuntimeError("GatewayAuth not initialized - call setup() first")

        full_path = f"/trade-api/v2{path}"
        headers = self._auth.create_auth_headers(method, full_path)
        headers["Content-Type"] = "application/json"
        return headers

    def ws_auth_message(self) -> Dict:
        """Build the authentication command for Kalshi WebSocket.

        Returns:
            Dict suitable for json.dumps and sending over WS.
        """
        if not self._auth:
            raise RuntimeError("GatewayAuth not initialized - call setup() first")

        # Kalshi WS auth uses the same signature mechanism
        headers = self._auth.create_auth_headers("GET", "/trade-api/ws/v2")
        return {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": [],  # filled by caller
                "market_tickers": [],  # filled by caller
            },
        } | {
            # Auth fields extracted from headers
            "api_key": headers.get("KALSHI-ACCESS-KEY", ""),
            "timestamp": headers.get("KALSHI-ACCESS-TIMESTAMP", ""),
            "signature": headers.get("KALSHI-ACCESS-SIGNATURE", ""),
        }
