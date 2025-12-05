"""
Market Metadata Service for fetching and caching Kalshi market information via REST API.

This service provides smart fetching logic to enrich market data with human-readable
information while maintaining performance of the core trade processing pipeline.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any, List
import aiohttp
from .auth import KalshiAuth, KalshiAuthError
from .database import Database

logger = logging.getLogger(__name__)


class KalshiMarketAPI:
    """REST API client for fetching market data from Kalshi."""
    
    def __init__(self, auth: KalshiAuth, base_url: str = None):
        """
        Initialize the Kalshi market API client.
        
        Args:
            auth: KalshiAuth instance for API authentication
            base_url: Base URL for Kalshi API (defaults to env var)
        """
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
        """
        Fetch market details from Kalshi REST API.
        
        Args:
            ticker: Market ticker to fetch
            
        Returns:
            Market data dictionary or None if failed
        """
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
                    data = await response.json()
                    return data
                elif response.status == 404:
                    logger.warning(f"Market {ticker} not found")
                    return None
                elif response.status == 429:
                    logger.warning(f"Rate limited when fetching market {ticker}")
                    return None
                else:
                    logger.error(f"Failed to fetch market {ticker}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching market details for {ticker}: {e}")
            return None


class MarketMetadataService:
    """Service for managing market metadata with smart caching and fetching."""
    
    def __init__(self, database: Database, auth: KalshiAuth = None):
        """
        Initialize the metadata service.
        
        Args:
            database: Database instance for caching
            auth: KalshiAuth instance for API calls (optional, will create from env if None)
        """
        self.database = database
        self.auth = auth or KalshiAuth.from_env()
        self.api_client = KalshiMarketAPI(self.auth)
        
        # Track which markets we've attempted to fetch (to avoid repeated failures)
        self._fetch_attempted: Set[str] = set()
        
        # Configuration
        self.fetch_enabled = os.getenv("METADATA_FETCH_ENABLED", "true").lower() == "true"
        
        logger.info(f"Initialized MarketMetadataService (fetch_enabled={self.fetch_enabled})")
    
    async def start(self):
        """Start the metadata service."""
        if not self.fetch_enabled:
            logger.info("Metadata fetching disabled by configuration")
            return
        logger.info("Metadata service started")
    
    async def stop(self):
        """Stop the metadata service and cleanup resources."""
        await self.api_client.close()
        logger.info("Stopped metadata service")
    
    
    
    async def get_market_metadata(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get market metadata from cache.
        
        Args:
            ticker: Market ticker
            
        Returns:
            Market metadata dictionary or None if not cached
        """
        return await self.database.get_market_metadata(ticker)
    
    async def get_markets_metadata(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for multiple markets from cache.
        
        Args:
            tickers: List of market tickers
            
        Returns:
            Dictionary mapping ticker to metadata
        """
        return await self.database.get_markets_metadata(tickers)
    
    async def fetch_metadata_now(self, ticker: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Immediately fetch metadata for a market with timeout protection.
        
        Args:
            ticker: Market ticker to fetch
            timeout: Maximum time to wait for fetch (seconds)
            
        Returns:
            Market metadata dictionary or None if failed/timeout
        """
        if not self.fetch_enabled:
            return None
            
        try:
            # Check cache first
            cached = await self.database.get_market_metadata(ticker)
            if cached:
                return cached
            
            # Fetch with timeout
            market_data = await asyncio.wait_for(
                self.api_client.fetch_market_details(ticker),
                timeout=timeout
            )
            
            if not market_data:
                return None
            
            # Extract relevant fields from the API response
            market = market_data.get("market", {})
            if not market:
                return None
            
            # Extract metadata fields
            title = market.get("title", ticker)
            category = market.get("category", "Unknown")
            
            # Handle liquidity_dollars (may be string or number)
            liquidity_raw = market.get("liquidity_dollars", 0)
            try:
                liquidity_dollars = float(liquidity_raw) if liquidity_raw is not None else 0
            except (ValueError, TypeError):
                liquidity_dollars = 0
                
            open_interest = market.get("open_interest", 0)
            
            # Handle expiration time
            expiration_time = None
            if market.get("close_time"):
                try:
                    expiration_time = market["close_time"]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid close_time format for {ticker}: {market.get('close_time')}")
            
            # Store the complete API response as raw data
            raw_market_data = json.dumps(market_data)
            
            # Save to database (async, don't wait)
            asyncio.create_task(
                self.database.insert_or_update_market(
                    ticker=ticker,
                    title=title,
                    category=category,
                    liquidity_dollars=liquidity_dollars,
                    open_interest=open_interest,
                    latest_expiration_time=expiration_time,
                    raw_market_data=raw_market_data
                )
            )
            
            # Return metadata immediately
            metadata = {
                "title": title,
                "category": category,
                "liquidity_dollars": liquidity_dollars,
                "open_interest": open_interest,
                "latest_expiration_time": expiration_time
            }
            
            return metadata
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error in immediate metadata fetch for {ticker}: {e}")
            return None
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status for monitoring."""
        return {
            "fetch_enabled": self.fetch_enabled,
            "attempted_fetches": len(self._fetch_attempted)
        }


# Global service instance
_metadata_service_instance = None

def get_metadata_service() -> Optional[MarketMetadataService]:
    """Get the global metadata service instance."""
    return _metadata_service_instance

def initialize_metadata_service(database: Database, auth: KalshiAuth = None) -> MarketMetadataService:
    """Initialize the global metadata service instance."""
    global _metadata_service_instance
    _metadata_service_instance = MarketMetadataService(database, auth)
    return _metadata_service_instance