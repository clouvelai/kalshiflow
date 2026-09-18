"""
In-memory market metadata cache backed by Kalshi REST.

Used to enrich top trades with titles/categories without a persistent database.
"""

import os
import asyncio
import logging
from typing import Dict, Optional, Any, List
import aiohttp
from .auth import KalshiAuth, KalshiAuthError

logger = logging.getLogger(__name__)


class KalshiMarketAPI:
    """REST API client for fetching market data from Kalshi."""
    
    def __init__(self, auth: KalshiAuth, base_url: str = None):
        self.auth = auth
        self.base_url = base_url or os.getenv("KALSHI_REST_API_URL", "https://api.elections.kalshi.com")
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return self._session
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_market_details(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch market details from Kalshi REST API."""
        try:
            path = f"/trade-api/v2/markets/{ticker}"
            headers = self.auth.create_auth_headers("GET", path)
            headers.update({
                "Accept": "application/json",
                "Content-Type": "application/json"
            })
            
            session = await self._get_session()
            url = f"{self.base_url}{path}"
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                if response.status == 404:
                    logger.warning(f"Market {ticker} not found")
                    return None
                if response.status == 429:
                    logger.warning(f"Rate limited when fetching market {ticker}")
                    return None
                logger.error(f"Failed to fetch market {ticker}: {response.status}")
                return None
                    
        except Exception as e:
            logger.error(f"Error fetching market details for {ticker}: {e}")
            return None


class MarketMetadataService:
    """In-memory metadata cache with optional Kalshi REST fetches."""
    
    def __init__(self, auth: KalshiAuth = None):
        try:
            self.auth = auth or KalshiAuth.from_env()
            self.api_client = KalshiMarketAPI(self.auth)
        except (KalshiAuthError, Exception) as e:
            logger.warning(f"Metadata REST client unavailable: {e}")
            self.auth = None
            self.api_client = None
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._fetch_attempted: set[str] = set()
        self.fetch_enabled = os.getenv("METADATA_FETCH_ENABLED", "true").lower() == "true"
        logger.info(f"Initialized MarketMetadataService (fetch_enabled={self.fetch_enabled})")
    
    async def start(self):
        if not self.fetch_enabled:
            logger.info("Metadata fetching disabled by configuration")
            return
        logger.info("Metadata service started")
    
    async def stop(self):
        if self.api_client:
            await self.api_client.close()
        logger.info("Stopped metadata service")
    
    async def get_market_metadata(self, ticker: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(ticker)
    
    async def get_markets_metadata(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        return {ticker: self._cache[ticker] for ticker in tickers if ticker in self._cache}
    
    async def fetch_metadata_now(self, ticker: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        if not self.fetch_enabled or not self.api_client:
            return self._cache.get(ticker)
            
        try:
            cached = self._cache.get(ticker)
            if cached:
                return cached
            
            self._fetch_attempted.add(ticker)
            market_data = await asyncio.wait_for(
                self.api_client.fetch_market_details(ticker),
                timeout=timeout
            )
            if not market_data:
                return None
            
            market = market_data.get("market", {})
            if not market:
                return None
            
            liquidity_raw = market.get("liquidity_dollars", 0)
            try:
                liquidity_dollars = float(liquidity_raw) if liquidity_raw is not None else 0
            except (ValueError, TypeError):
                liquidity_dollars = 0
            
            metadata = {
                "title": market.get("title", ticker),
                "category": market.get("category", "Unknown"),
                "liquidity_dollars": liquidity_dollars,
                "open_interest": market.get("open_interest", 0),
                "latest_expiration_time": market.get("close_time"),
            }
            self._cache[ticker] = metadata
            return metadata
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error in immediate metadata fetch for {ticker}: {e}")
            return None
    
    async def get_service_status(self) -> Dict[str, Any]:
        return {
            "fetch_enabled": self.fetch_enabled,
            "attempted_fetches": len(self._fetch_attempted),
            "cached_markets": len(self._cache),
        }


_metadata_service_instance = None

def get_metadata_service() -> Optional[MarketMetadataService]:
    return _metadata_service_instance

def initialize_metadata_service(auth: KalshiAuth = None) -> MarketMetadataService:
    global _metadata_service_instance
    _metadata_service_instance = MarketMetadataService(auth)
    return _metadata_service_instance
