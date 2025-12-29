#!/usr/bin/env python3
"""
Full Settled Markets Fetcher - Save complete market data for all settled markets.

Uses the Kalshi GET /markets endpoint with status=settled filter to efficiently
fetch all settled markets and save the FULL response to disk.

This gives us complete historical market data for backtesting and analysis.

API: https://docs.kalshi.com/api-reference/market/get-markets

Usage:
    # Fetch all settled markets and save to JSON
    python fetch_full_settled_markets.py --output settled_markets.json

    # Resume from previous run
    python fetch_full_settled_markets.py --output settled_markets.json --resume

    # Also save CSV summary
    python fetch_full_settled_markets.py --output settled_markets.json --csv-summary settled_summary.csv
"""

import asyncio
import aiohttp
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
import tempfile

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kalshiflow.auth import KalshiAuth

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SettledMarketsFetcher:
    """
    Fetches all settled markets from Kalshi API with pagination.

    Uses GET /markets?status=settled with cursor-based pagination.
    Saves complete market data to JSON for full historical analysis.
    """

    def __init__(
        self,
        api_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        rate_limit: float = 2.0,  # Conservative for paginated endpoint
    ):
        self.api_url = api_url
        self.min_interval = 1.0 / rate_limit
        self.last_request_time = 0.0

        # Initialize authentication
        self.auth = self._init_auth()

        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None

        # All fetched markets
        self.markets: Dict[str, Dict[str, Any]] = {}

    def _init_auth(self) -> KalshiAuth:
        """Initialize Kalshi authentication from environment."""
        from dotenv import load_dotenv

        backend_dir = Path(__file__).parent.parent.parent.parent
        env_files = [
            backend_dir / ".env",
            backend_dir / ".env.local",
        ]

        for env_file in env_files:
            if env_file.exists():
                logger.info(f"Loading credentials from {env_file.name}")
                load_dotenv(env_file, override=True)

        api_key_id = os.getenv("KALSHI_API_KEY_ID")
        private_key_content = os.getenv("KALSHI_PRIVATE_KEY_CONTENT")

        if not api_key_id or not private_key_content:
            raise ValueError("KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_CONTENT required")

        # Strip quotes and handle escaping
        if private_key_content.startswith('"') and private_key_content.endswith('"'):
            private_key_content = private_key_content[1:-1]

        # Create temp key file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pem', prefix='kalshi_key_')
        self._temp_key_file = temp_path

        with os.fdopen(temp_fd, 'w') as f:
            f.write(private_key_content.replace('\\n', '\n'))

        return KalshiAuth(api_key_id, temp_path)

    def _cleanup(self):
        if hasattr(self, '_temp_key_file'):
            try:
                os.unlink(self._temp_key_file)
            except:
                pass

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()
        self._cleanup()

    async def _rate_limit_wait(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def _create_headers(self, method: str, path: str) -> Dict[str, str]:
        full_path = f"/trade-api/v2{path}"
        headers = self.auth.create_auth_headers(method, full_path)
        headers["Content-Type"] = "application/json"
        return headers

    async def fetch_settled_page(self, cursor: str = None, limit: int = 1000) -> tuple:
        """
        Fetch one page of settled markets.

        Returns:
            (markets_list, next_cursor) - markets and cursor for next page (None if done)
        """
        path = f"/markets?status=settled&limit={limit}"
        if cursor:
            path += f"&cursor={cursor}"

        url = f"{self.api_url}{path}"

        await self._rate_limit_wait()

        try:
            headers = self._create_headers("GET", path.split('?')[0])

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    markets = data.get("markets", [])
                    next_cursor = data.get("cursor", "")
                    return markets, next_cursor if next_cursor else None

                elif response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 10))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self.fetch_settled_page(cursor, limit)

                else:
                    text = await response.text()
                    logger.error(f"API error: {response.status} - {text[:500]}")
                    return [], None

        except Exception as e:
            logger.error(f"Request error: {e}")
            return [], None

    async def fetch_all_settled(
        self,
        max_pages: int = None,
        save_callback: callable = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch ALL settled markets with pagination.

        Args:
            max_pages: Optional limit on pages (for testing)
            save_callback: Optional callback(markets_dict) to save incrementally

        Returns:
            Dict mapping ticker -> full market data
        """
        cursor = None
        page = 0
        total_fetched = 0

        while True:
            page += 1

            if max_pages and page > max_pages:
                logger.info(f"Reached max pages limit ({max_pages})")
                break

            markets, next_cursor = await self.fetch_settled_page(cursor)

            if not markets:
                if cursor is None:
                    logger.warning("No markets returned on first page!")
                break

            # Store markets by ticker
            page_markets = {}
            for market in markets:
                ticker = market.get("ticker")
                if ticker:
                    self.markets[ticker] = market
                    page_markets[ticker] = market

            total_fetched += len(markets)

            logger.info(f"Page {page}: fetched {len(markets)} markets (total: {total_fetched})")

            # Incremental save if callback provided
            if save_callback and page_markets:
                save_callback(page_markets, page)

            # Check if we're done
            if not next_cursor:
                logger.info("Reached end of pagination")
                break

            cursor = next_cursor

        logger.info(f"Completed: fetched {len(self.markets)} total settled markets")
        return self.markets


def save_markets_json(markets: Dict[str, Dict], output_path: str):
    """Save complete market data to JSON."""
    with open(output_path, 'w') as f:
        json.dump(markets, f, indent=2, default=str)
    logger.info(f"Saved {len(markets)} markets to {output_path}")


def save_markets_csv_summary(markets: Dict[str, Dict], output_path: str):
    """Save CSV summary of key fields."""
    import csv

    fields = [
        'ticker', 'title', 'result', 'status',
        'settlement_value', 'settlement_ts',
        'close_time', 'expiration_time',
        'volume', 'open_interest',
        'event_ticker', 'category',
        'yes_bid', 'yes_ask', 'no_bid', 'no_ask',
        'last_price', 'previous_yes_bid', 'previous_yes_ask',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()

        for ticker, market in markets.items():
            row = {'ticker': ticker}
            for field in fields[1:]:
                row[field] = market.get(field, '')
            writer.writerow(row)

    logger.info(f"Saved CSV summary to {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description='Fetch all settled markets with full data'
    )

    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for full market data')
    parser.add_argument('--csv-summary', type=str,
                       help='Optional CSV summary file')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from existing JSON file')
    parser.add_argument('--max-pages', type=int, default=None,
                       help='Limit pages (for testing)')
    parser.add_argument('--rate-limit', type=float, default=2.0,
                       help='Requests per second (default: 2)')

    args = parser.parse_args()

    # Resume handling
    existing_markets = {}
    if args.resume and os.path.exists(args.output):
        logger.info(f"Resuming from {args.output}")
        with open(args.output, 'r') as f:
            existing_markets = json.load(f)
        logger.info(f"Loaded {len(existing_markets)} existing markets")

    start_time = time.time()

    # Incremental save callback
    all_markets = dict(existing_markets)

    def incremental_save(page_markets, page_num):
        all_markets.update(page_markets)
        if page_num % 10 == 0:  # Save every 10 pages
            save_markets_json(all_markets, args.output)

    async with SettledMarketsFetcher(rate_limit=args.rate_limit) as fetcher:
        # If resuming, start from existing
        fetcher.markets = dict(existing_markets)

        markets = await fetcher.fetch_all_settled(
            max_pages=args.max_pages,
            save_callback=incremental_save
        )

    # Final save
    save_markets_json(markets, args.output)

    if args.csv_summary:
        save_markets_csv_summary(markets, args.csv_summary)

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed:.1f} seconds")

    # Summary stats
    results = {'yes': 0, 'no': 0, 'other': 0}
    for m in markets.values():
        result = m.get('result', '')
        if result == 'yes':
            results['yes'] += 1
        elif result == 'no':
            results['no'] += 1
        else:
            results['other'] += 1

    print(f"\n{'='*60}")
    print(f"SETTLED MARKETS SUMMARY")
    print(f"{'='*60}")
    print(f"  Total markets:    {len(markets):,}")
    print(f"  YES resolved:     {results['yes']:,}")
    print(f"  NO resolved:      {results['no']:,}")
    print(f"  Other/Unknown:    {results['other']:,}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
