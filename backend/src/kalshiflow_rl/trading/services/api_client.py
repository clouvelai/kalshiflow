"""
ApiClient - Created for TRADER 2.0

Wrapper around KalshiDemoTradingClient with enhanced error handling and retry logic.
Provides a clean abstraction layer for API calls with monitoring and statistics.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..demo_client import KalshiDemoTradingClient, KalshiDemoTradingClientError

logger = logging.getLogger("kalshiflow_rl.trading.services.api_client")


@dataclass
class ApiCall:
    """Represents an API call for tracking and retries."""
    method_name: str
    args: tuple
    kwargs: dict
    timestamp: float
    attempts: int = 0
    success: bool = False
    error: Optional[str] = None
    duration: float = 0.0


class ApiClient:
    """
    Enhanced wrapper around KalshiDemoTradingClient.
    
    Provides retry logic, error handling, rate limiting, and statistics
    for all API interactions with Kalshi.
    """
    
    def __init__(
        self,
        client: KalshiDemoTradingClient,
        status_logger: Optional['StatusLogger'] = None
    ):
        """
        Initialize ApiClient.
        
        Args:
            client: KalshiDemoTradingClient instance
            status_logger: Optional StatusLogger for activity tracking
        """
        self.client = client
        self.status_logger = status_logger
        
        # Retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        self.backoff_multiplier = 2.0
        
        # Rate limiting
        self.rate_limit_delay = 0.1  # Minimum delay between API calls
        self.last_call_time: Optional[float] = None
        
        # Statistics
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retries_total": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "last_call_time": None,
            "last_error": None,
            "calls_by_method": {},
            "errors_by_type": {}
        }
        
        # Call history (for debugging)
        self.call_history: List[ApiCall] = []
        self.max_history_size = 100
        
        logger.info("ApiClient initialized")
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between API calls."""
        if self.last_call_time:
            elapsed = time.time() - self.last_call_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)
    
    async def _execute_with_retry(self, method_name: str, *args, **kwargs) -> Any:
        """
        Execute API method with retry logic.
        
        Args:
            method_name: Name of the client method to call
            *args: Arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            API response
            
        Raises:
            KalshiDemoTradingClientError: If all retry attempts fail
        """
        api_call = ApiCall(
            method_name=method_name,
            args=args,
            kwargs=kwargs,
            timestamp=time.time()
        )
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                api_call.attempts = attempt + 1
                
                # Apply rate limiting
                await self._rate_limit()
                
                # Get the method from the client
                method = getattr(self.client, method_name)
                
                # Execute the API call
                call_start = time.time()
                result = await method(*args, **kwargs)
                call_duration = time.time() - call_start
                
                # Success
                api_call.success = True
                api_call.duration = call_duration
                
                # Update statistics
                self._update_stats(api_call, success=True)
                
                # Log success
                if self.status_logger and attempt > 0:
                    await self.status_logger.log_action_result(
                        "api_retry_success",
                        f"{method_name} succeeded after {attempt} retries",
                        call_duration
                    )
                
                self.last_call_time = time.time()
                
                logger.debug(f"API call succeeded: {method_name} ({call_duration:.3f}s, attempt {attempt + 1})")
                
                return result
                
            except Exception as e:
                last_error = e
                api_call.error = str(e)
                
                # Log retry attempt
                if attempt < self.max_retries:
                    retry_delay = self.retry_delay * (self.backoff_multiplier ** attempt)
                    
                    logger.warning(f"API call failed, retrying in {retry_delay}s: {method_name} - {str(e)}")
                    
                    if self.status_logger:
                        await self.status_logger.log_action_result(
                            "api_retry",
                            f"{method_name} retry {attempt + 1}/{self.max_retries}",
                            0.0
                        )
                    
                    await asyncio.sleep(retry_delay)
                else:
                    # Final failure
                    api_call.duration = time.time() - start_time
                    self._update_stats(api_call, success=False)
                    
                    logger.error(f"API call failed after {self.max_retries} retries: {method_name} - {str(e)}")
                    
                    if self.status_logger:
                        await self.status_logger.log_action_result(
                            "api_failure",
                            f"{method_name} failed after {self.max_retries} retries",
                            api_call.duration
                        )
        
        # All retries exhausted
        raise last_error or KalshiDemoTradingClientError(f"API call failed: {method_name}")
    
    def _update_stats(self, api_call: ApiCall, success: bool) -> None:
        """Update API statistics."""
        self.stats["total_calls"] += 1
        self.stats["last_call_time"] = api_call.timestamp
        self.stats["total_duration"] += api_call.duration
        self.stats["avg_duration"] = self.stats["total_duration"] / self.stats["total_calls"]
        
        # Method-specific stats
        method_stats = self.stats["calls_by_method"].get(api_call.method_name, {
            "calls": 0, "successes": 0, "failures": 0, "retries": 0
        })
        method_stats["calls"] += 1
        method_stats["retries"] += api_call.attempts - 1
        
        if success:
            self.stats["successful_calls"] += 1
            method_stats["successes"] += 1
        else:
            self.stats["failed_calls"] += 1
            method_stats["failures"] += 1
            self.stats["last_error"] = api_call.error
            
            # Error type stats
            error_type = type(api_call.error).__name__ if api_call.error else "unknown"
            self.stats["errors_by_type"][error_type] = self.stats["errors_by_type"].get(error_type, 0) + 1
        
        self.stats["calls_by_method"][api_call.method_name] = method_stats
        
        if api_call.attempts > 1:
            self.stats["retries_total"] += api_call.attempts - 1
        
        # Add to history
        self.call_history.append(api_call)
        if len(self.call_history) > self.max_history_size:
            self.call_history.pop(0)
    
    # API method wrappers with enhanced error handling
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        return await self._execute_with_retry("get_account_info")
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get all positions."""
        return await self._execute_with_retry("get_positions")
    
    async def get_orders(self, ticker: Optional[str] = None) -> Dict[str, Any]:
        """Get orders, optionally filtered by ticker."""
        if ticker:
            return await self._execute_with_retry("get_orders", ticker=ticker)
        else:
            return await self._execute_with_retry("get_orders")
    
    async def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        price: int,
        type: str = "limit"
    ) -> Dict[str, Any]:
        """Create an order."""
        return await self._execute_with_retry(
            "create_order",
            ticker=ticker,
            action=action,
            side=side,
            count=count,
            price=price,
            type=type
        )
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order."""
        return await self._execute_with_retry("cancel_order", order_id)
    
    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get order details."""
        return await self._execute_with_retry("get_order", order_id)
    
    async def get_market(self, ticker: str) -> Dict[str, Any]:
        """Get market information."""
        return await self._execute_with_retry("get_market", ticker)
    
    async def get_markets(self) -> Dict[str, Any]:
        """Get all markets."""
        return await self._execute_with_retry("get_markets")
    
    async def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        """Get orderbook for a market."""
        return await self._execute_with_retry("get_orderbook", ticker)
    
    # Health and monitoring methods
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check by making a lightweight API call.
        
        Returns:
            Dict with health check results
        """
        try:
            start_time = time.time()
            
            # Try to get account info as health check
            await self.get_account_info()
            
            duration = time.time() - start_time
            
            return {
                "healthy": True,
                "response_time": duration,
                "timestamp": time.time(),
                "last_error": None
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "response_time": None,
                "timestamp": time.time(),
                "last_error": str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API client statistics."""
        current_time = time.time()
        
        # Calculate success rate
        success_rate = self.stats["successful_calls"] / max(1, self.stats["total_calls"])
        retry_rate = self.stats["retries_total"] / max(1, self.stats["total_calls"])
        
        return {
            "total_calls": self.stats["total_calls"],
            "successful_calls": self.stats["successful_calls"],
            "failed_calls": self.stats["failed_calls"],
            "success_rate": success_rate,
            "retries_total": self.stats["retries_total"],
            "retry_rate": retry_rate,
            "avg_duration": self.stats["avg_duration"],
            "last_call_time": self.stats["last_call_time"],
            "time_since_last_call": current_time - self.stats["last_call_time"] if self.stats["last_call_time"] else None,
            "last_error": self.stats["last_error"],
            "calls_by_method": self.stats["calls_by_method"].copy(),
            "errors_by_type": self.stats["errors_by_type"].copy(),
            "configuration": {
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "backoff_multiplier": self.backoff_multiplier,
                "rate_limit_delay": self.rate_limit_delay
            }
        }
    
    def get_recent_calls(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent API calls for debugging.
        
        Args:
            limit: Maximum number of calls to return
            
        Returns:
            List of recent API calls
        """
        recent_calls = self.call_history[-limit:] if limit else self.call_history
        
        return [
            {
                "method_name": call.method_name,
                "timestamp": call.timestamp,
                "attempts": call.attempts,
                "success": call.success,
                "duration": call.duration,
                "error": call.error
            }
            for call in recent_calls
        ]
    
    def is_healthy(self) -> bool:
        """
        Check if API client is healthy based on recent performance.
        
        Returns:
            True if client appears healthy
        """
        # Check if we have recent successful calls
        if self.stats["total_calls"] == 0:
            return True  # No calls yet, assume healthy
        
        # Check success rate
        success_rate = self.stats["successful_calls"] / self.stats["total_calls"]
        if success_rate < 0.8:  # Less than 80% success rate
            return False
        
        # Check if last call was recent and successful
        if self.stats["last_call_time"]:
            time_since_last = time.time() - self.stats["last_call_time"]
            if time_since_last < 300:  # Less than 5 minutes ago
                # Check recent call history
                recent_calls = self.call_history[-5:] if len(self.call_history) >= 5 else self.call_history
                recent_successes = sum(1 for call in recent_calls if call.success)
                if recent_successes == 0:  # No recent successes
                    return False
        
        return True
    
    def reset_statistics(self) -> None:
        """Reset all statistics (for testing or maintenance)."""
        logger.info("Resetting API client statistics")
        
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retries_total": 0,
            "total_duration": 0.0,
            "avg_duration": 0.0,
            "last_call_time": None,
            "last_error": None,
            "calls_by_method": {},
            "errors_by_type": {}
        }
        
        self.call_history.clear()
    
    def configure_retries(
        self,
        max_retries: int = None,
        retry_delay: float = None,
        backoff_multiplier: float = None,
        rate_limit_delay: float = None
    ) -> None:
        """
        Configure retry behavior.
        
        Args:
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            backoff_multiplier: Exponential backoff multiplier
            rate_limit_delay: Minimum delay between calls
        """
        if max_retries is not None:
            self.max_retries = max_retries
        if retry_delay is not None:
            self.retry_delay = retry_delay
        if backoff_multiplier is not None:
            self.backoff_multiplier = backoff_multiplier
        if rate_limit_delay is not None:
            self.rate_limit_delay = rate_limit_delay
        
        logger.info(f"API client retry configuration updated: max_retries={self.max_retries}, "
                   f"retry_delay={self.retry_delay}, backoff_multiplier={self.backoff_multiplier}, "
                   f"rate_limit_delay={self.rate_limit_delay}")