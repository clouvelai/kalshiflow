"""Token bucket rate limiter for KalshiGateway.

Uses aiolimiter for async-safe rate limiting with burst support.
Default: 10 requests/second sustained, 20 burst for arb leg placement.
"""

import logging
from aiolimiter import AsyncLimiter

logger = logging.getLogger("kalshiflow_rl.traderv3.gateway.rate_limiter")


class GatewayRateLimiter:
    """Async token-bucket rate limiter.

    Wraps aiolimiter.AsyncLimiter to provide a simple acquire() interface.
    The burst capacity allows rapid placement of arb legs without blocking.
    """

    def __init__(self, rate: float = 10.0, burst: int = 20):
        """
        Args:
            rate: Sustained requests per second.
            burst: Maximum burst capacity (tokens available at start).
        """
        self._limiter = AsyncLimiter(max_rate=burst, time_period=burst / rate)
        self._rate = rate
        self._burst = burst
        logger.info(f"Rate limiter: {rate} req/s sustained, {burst} burst")

    async def acquire(self) -> None:
        """Wait until a token is available. Non-blocking when under limit."""
        await self._limiter.acquire()
