"""Structured error hierarchy for KalshiGateway.

All gateway errors inherit from KalshiError, enabling uniform catch-all
handling while allowing granular recovery for specific failure modes.
"""


class KalshiError(Exception):
    """Base exception for all Kalshi gateway errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class KalshiAuthError(KalshiError):
    """Authentication or credential error."""
    pass


class KalshiRateLimitError(KalshiError):
    """Rate limit exceeded (429)."""
    pass


class KalshiOrderError(KalshiError):
    """Order placement, cancellation, or validation error."""
    pass


class KalshiConnectionError(KalshiError):
    """Network or WebSocket connection error."""
    pass


class KalshiNotFoundError(KalshiError):
    """Resource not found (404) - e.g. order already filled/cancelled."""
    pass
