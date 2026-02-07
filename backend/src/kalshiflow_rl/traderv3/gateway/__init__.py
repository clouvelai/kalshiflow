"""KalshiGateway - Unified API/WS client for the V3 trader.

Public exports:
    KalshiGateway: Main client class (REST + WS)
    GatewayEventBridge: Translates WS events to EventBus
    GatewayAuth: Auth credential management
    WSMultiplexer: Single multiplexed WS connection
    GatewayRateLimiter: Token bucket rate limiter

Error hierarchy:
    KalshiError (base)
    KalshiAuthError
    KalshiOrderError
    KalshiRateLimitError
    KalshiConnectionError
    KalshiNotFoundError

Models (Pydantic v2):
    Balance, Position, Fill, Order, OrderResponse, etc.
"""

from .client import KalshiGateway
from .event_bridge import GatewayEventBridge
from .auth import GatewayAuth
from .ws_multiplexer import WSMultiplexer
from .rate_limiter import GatewayRateLimiter
from .errors import (
    KalshiError,
    KalshiAuthError,
    KalshiOrderError,
    KalshiRateLimitError,
    KalshiConnectionError,
    KalshiNotFoundError,
)

__all__ = [
    "KalshiGateway",
    "GatewayEventBridge",
    "GatewayAuth",
    "WSMultiplexer",
    "GatewayRateLimiter",
    "KalshiError",
    "KalshiAuthError",
    "KalshiOrderError",
    "KalshiRateLimitError",
    "KalshiConnectionError",
    "KalshiNotFoundError",
]
