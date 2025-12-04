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
            
            logger.debug(f"Fetching market details for {ticker} from {url}")
            
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully fetched market details for {ticker}")
                    return data
                elif response.status == 404:
                    logger.warning(f"Market {ticker} not found (404)")
                    return None
                elif response.status == 429:
                    logger.warning(f"Rate limited when fetching market {ticker}")
                    return None
                else:
                    logger.error(f"Failed to fetch market {ticker}: {response.status} - {await response.text()}")
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
        self._fetch_queue = asyncio.Queue()
        self._fetch_worker_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Configuration
        self.fetch_enabled = os.getenv("METADATA_FETCH_ENABLED", "true").lower() == "true"
        self.check_interval = int(os.getenv("METADATA_CHECK_INTERVAL", "30"))
        
        logger.info(f"Initialized MarketMetadataService (fetch_enabled={self.fetch_enabled})")
    
    async def start(self):
        """Start the metadata service background tasks."""
        if not self.fetch_enabled:
            logger.info("Metadata fetching disabled by configuration")
            return
            
        self._running = True
        self._fetch_worker_task = asyncio.create_task(self._fetch_worker())
        logger.info("Started metadata service background worker")
    
    async def stop(self):
        """Stop the metadata service and cleanup resources."""
        self._running = False
        
        if self._fetch_worker_task and not self._fetch_worker_task.done():
            self._fetch_worker_task.cancel()
            try:
                await self._fetch_worker_task
            except asyncio.CancelledError:
                pass
        
        await self.api_client.close()
        logger.info("Stopped metadata service")
    
    async def _fetch_worker(self):
        """Background worker for processing metadata fetch requests."""
        while self._running:
            try:
                # Wait for a ticker to fetch with timeout
                ticker = await asyncio.wait_for(
                    self._fetch_queue.get(), 
                    timeout=1.0
                )
                
                await self._fetch_and_cache_market(ticker)
                self._fetch_queue.task_done()
                
                # Rate limiting - wait between requests
                await asyncio.sleep(0.5)
                
            except asyncio.TimeoutError:
                # No items to fetch, continue loop
                continue
            except Exception as e:
                logger.error(f"Error in fetch worker: {e}")
                await asyncio.sleep(1.0)
    
    async def _fetch_and_cache_market(self, ticker: str):
        """
        Fetch market metadata from API and cache in database.
        
        Args:
            ticker: Market ticker to fetch
        """
        try:
            # Mark as attempted regardless of success to avoid repeated failures
            self._fetch_attempted.add(ticker)
            
            market_data = await self.api_client.fetch_market_details(ticker)
            if not market_data:
                logger.warning(f"No market data returned for {ticker}")
                return
            
            # Extract relevant fields from the API response
            market = market_data.get("market", {})
            if not market:
                logger.warning(f"No market object in response for {ticker}")
                return
            
            # Extract metadata fields
            title = market.get("title", ticker)
            category = market.get("category", "Unknown")
            liquidity_dollars = market.get("liquidity", 0)
            open_interest = market.get("open_interest", 0)
            
            # Handle expiration time
            expiration_time = None
            if market.get("close_time"):
                try:
                    # Convert to ISO format if needed
                    expiration_time = market["close_time"]
                except (ValueError, TypeError):
                    logger.warning(f"Invalid close_time format for {ticker}: {market.get('close_time')}")
            
            # Store the complete API response as raw data
            raw_market_data = json.dumps(market_data)
            
            # Save to database
            success = await self.database.insert_or_update_market(
                ticker=ticker,
                title=title,
                category=category,
                liquidity_dollars=liquidity_dollars,
                open_interest=open_interest,
                latest_expiration_time=expiration_time,
                raw_market_data=raw_market_data
            )
            
            if success:
                logger.info(f"Cached metadata for {ticker}: {title}")
            else:
                logger.error(f"Failed to cache metadata for {ticker}")
                
        except Exception as e:
            logger.error(f"Error fetching and caching market {ticker}: {e}")
    
    async def queue_metadata_fetch(self, ticker: str) -> bool:
        """
        Queue a market ticker for metadata fetching.
        
        Args:
            ticker: Market ticker to fetch
            
        Returns:
            True if queued, False if already attempted or disabled
        """
        if not self.fetch_enabled:
            return False
        
        if ticker in self._fetch_attempted:
            return False
        
        # Check if already in cache
        if await self.database.market_exists(ticker):
            return False
        
        try:
            self._fetch_queue.put_nowait(ticker)
            logger.debug(f"Queued metadata fetch for {ticker}")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Fetch queue is full, dropping request for {ticker}")
            return False
    
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
    
    def should_fetch_metadata(self, ticker: str) -> bool:
        """
        Determine if we should attempt to fetch metadata for a ticker.
        
        Args:
            ticker: Market ticker
            
        Returns:
            True if should fetch, False otherwise
        """
        if not self.fetch_enabled:
            return False
        
        # Don't fetch if already attempted
        if ticker in self._fetch_attempted:
            return False
        
        # Only fetch for markets that appear in hot markets
        return True
    
    async def monitor_hot_markets(self, hot_markets: List[Dict[str, Any]]):
        """
        Monitor hot markets list and queue metadata fetching for new markets.
        
        Args:
            hot_markets: List of hot market dictionaries with 'ticker' field
        """
        if not self.fetch_enabled:
            return
        
        for market in hot_markets:
            ticker = market.get("ticker")
            if ticker and self.should_fetch_metadata(ticker):
                await self.queue_metadata_fetch(ticker)
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get service status for monitoring."""
        return {
            "running": self._running,
            "fetch_enabled": self.fetch_enabled,
            "attempted_fetches": len(self._fetch_attempted),
            "queue_size": self._fetch_queue.qsize(),
            "worker_running": self._fetch_worker_task and not self._fetch_worker_task.done(),
            "check_interval": self.check_interval
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