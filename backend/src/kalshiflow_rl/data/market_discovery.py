"""
Market discovery service for Kalshi RL Trading Subsystem.

Provides functionality to dynamically discover active markets using Kalshi's
REST API, filtering for open markets with trading activity.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import httpx

from .auth import get_rl_auth
from ..config import config

logger = logging.getLogger("kalshiflow_rl.market_discovery")


class MarketDiscoveryService:
    """Service for discovering active Kalshi markets for RL training."""
    
    def __init__(self):
        """Initialize market discovery service."""
        self.rl_auth = get_rl_auth()
        self.auth = self.rl_auth.auth
        # Use config URL which respects ENVIRONMENT (paper → demo-api.kalshi.co)
        self.base_url = config.KALSHI_API_URL
        
    async def fetch_markets_list(self, 
                                limit: int = 100, 
                                status: str = "open") -> Dict[str, Any]:
        """
        Fetch markets list from Kalshi API.
        
        Args:
            limit: Maximum number of markets to fetch (default: 100)
            status: Market status filter - 'open' for active markets
            
        Returns:
            Dictionary containing API response or error info
        """
        url = f"{self.base_url}/markets"
        params = {
            "limit": limit,
            "status": status
        }
        
        try:
            # Create authenticated headers
            headers = self.auth.create_auth_headers("GET", "/trade-api/v2/markets")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "success": True,
                        "data": data,
                        "total_markets": len(data.get("markets", [])),
                        "status_code": response.status_code
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "status_code": response.status_code
                    }
                    
        except Exception as e:
            logger.error(f"Failed to fetch markets list: {e}")
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "status_code": None
            }
    
    def prioritize_markets(self, markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prioritize markets based on activity indicators.
        
        Args:
            markets: List of market dictionaries from API
            
        Returns:
            Sorted list of markets (most active first)
        """
        def get_activity_score(market: Dict[str, Any]) -> float:
            """Calculate activity score for market prioritization."""
            score = 0.0
            
            # Prioritize markets with volume information if available
            volume = market.get("volume", 0) or 0
            if volume > 0:
                score += float(volume) * 1000  # High weight for volume
            
            # Prioritize markets with recent activity
            last_trade = market.get("last_trade_price")
            if last_trade is not None:
                score += 500  # Bonus for having trade history
            
            # Prioritize markets with tighter spreads (if bid/ask available)
            # This would require additional orderbook calls, skip for now
            
            # Prioritize certain categories (elections, Fed decisions, etc.)
            title = market.get("title", "").lower()
            if any(keyword in title for keyword in ["election", "fed", "rate", "decision", "vote"]):
                score += 100
            
            # Prioritize markets closing sooner (more urgency = more activity)
            # This would require parsing close_time, skip for now
            
            return score
        
        # Sort by activity score (highest first)
        prioritized = sorted(markets, key=get_activity_score, reverse=True)
        
        # Log prioritization info only if there are markets
        if prioritized and len(prioritized) > 0:
            top_5 = prioritized[:5]
            top_tickers = [m.get('ticker', 'N/A') for m in top_5]
            logger.info(f"Top markets by activity: {', '.join(top_tickers)}")
        
        return prioritized
    
    async def fetch_active_markets(self, limit: int = None) -> List[str]:
        """
        Fetch list of active market tickers suitable for RL training.
        
        Args:
            limit: Maximum number of markets to return (defaults to config.ORDERBOOK_MARKET_LIMIT)
            
        Returns:
            List of market ticker strings, prioritized by activity
        """
        if limit is None:
            limit = config.ORDERBOOK_MARKET_LIMIT
            
        logger.info(f"Fetching up to {limit} active markets from Kalshi API...")
        
        # Fetch markets from API
        result = await self.fetch_markets_list(limit=limit, status="open")
        
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Failed to fetch markets: {error_msg}")
            
            # Fallback to configured tickers
            logger.warning("Falling back to configured RL_MARKET_TICKERS")
            return config.RL_MARKET_TICKERS
        
        markets = result["data"].get("markets", [])
        total_fetched = len(markets)
        
        if not markets:
            logger.warning("No open markets found, falling back to configured tickers")
            return config.RL_MARKET_TICKERS
        
        # Prioritize markets by activity
        prioritized_markets = self.prioritize_markets(markets)
        
        # Extract tickers
        market_tickers = []
        for market in prioritized_markets:
            ticker = market.get("ticker")
            if ticker:
                market_tickers.append(ticker)
        
        # Limit to requested number
        final_tickers = market_tickers[:limit]
        
        logger.info(f"Selected {len(final_tickers)} markets from {total_fetched} available open markets")
        
        # Log sample of selected markets
        sample_size = min(10, len(final_tickers))
        logger.info(f"Top {sample_size} selected markets: {final_tickers[:sample_size]}")
        
        if len(final_tickers) < 10:
            logger.warning(f"Only found {len(final_tickers)} markets - consider lowering limit or checking market availability")
        
        return final_tickers
    
    async def get_market_info(self, market_ticker: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific market.
        
        Args:
            market_ticker: The market ticker to query
            
        Returns:
            Dictionary containing market information
        """
        url = f"{self.base_url}/markets/{market_ticker}"
        
        try:
            headers = self.auth.create_auth_headers("GET", f"/trade-api/v2/markets/{market_ticker}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=10.0)
                
                if response.status_code == 200:
                    return {
                        "success": True,
                        "data": response.json(),
                        "status_code": response.status_code
                    }
                else:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "status_code": response.status_code
                    }
                    
        except Exception as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "status_code": None
            }


# Global service instance
_market_discovery_service = None


def get_market_discovery_service() -> MarketDiscoveryService:
    """Get the global market discovery service instance."""
    global _market_discovery_service
    if _market_discovery_service is None:
        _market_discovery_service = MarketDiscoveryService()
    return _market_discovery_service


async def fetch_active_markets(limit: int = None) -> List[str]:
    """
    Convenience function to fetch active markets.
    
    Args:
        limit: Maximum number of markets to return
        
    Returns:
        List of active market tickers
    """
    service = get_market_discovery_service()
    return await service.fetch_active_markets(limit)


async def validate_market_selection(market_tickers: List[str], 
                                   max_check: int = 5) -> Dict[str, Any]:
    """
    Validate a selection of markets by checking their orderbook activity.
    
    Args:
        market_tickers: List of market tickers to validate
        max_check: Maximum number of markets to check (for performance)
        
    Returns:
        Dictionary with validation results
    """
    service = get_market_discovery_service()
    
    # Import here to avoid circular dependencies
    from ..debug_market_orderbooks import KalshiOrderbookDebugger
    
    debugger = KalshiOrderbookDebugger()
    
    # Check first few markets
    check_tickers = market_tickers[:max_check]
    results = []
    
    for ticker in check_tickers:
        orderbook_data = await debugger.fetch_market_orderbook(ticker)
        analysis = debugger.analyze_orderbook(orderbook_data)
        results.append({
            "ticker": ticker,
            "is_active": not analysis["is_empty"],
            "levels": analysis.get("total_levels", 0),
            "volume": analysis.get("total_volume", 0)
        })
    
    active_count = sum(1 for r in results if r["is_active"])
    total_checked = len(results)
    
    return {
        "total_markets": len(market_tickers),
        "checked_markets": total_checked,
        "active_markets": active_count,
        "activity_rate": active_count / total_checked if total_checked > 0 else 0,
        "sample_results": results
    }