#!/usr/bin/env python3
"""
Market Outcomes Fetcher - Fetch settlement results for historical trades.

This script fetches market outcome data from Kalshi REST API to determine
whether trades in our historical database were profitable (winners vs losers).

Key Kalshi API fields:
- status: "determined" means market has resolved
- result: "yes" or "no" indicates winning side

Usage:
    # Extract unique tickers from trades CSV and fetch outcomes
    python fetch_market_outcomes.py --from-csv trades.csv --output outcomes.csv

    # Fetch outcomes for specific tickers
    python fetch_market_outcomes.py --tickers TICKER1,TICKER2 --output outcomes.csv

    # Resume from previous run (skip already fetched tickers)
    python fetch_market_outcomes.py --from-csv trades.csv --output outcomes.csv --resume

    # Fetch with rate limit control
    python fetch_market_outcomes.py --from-csv trades.csv --rate-limit 10

Author: Claude Code
"""

import asyncio
import aiohttp
import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kalshiflow.auth import KalshiAuth, RSASigner
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, requests_per_second: float = 5.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

    async def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            await asyncio.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


class MarketOutcomeFetcher:
    """
    Fetches market outcome data from Kalshi REST API.

    Handles:
    - RSA authentication
    - Rate limiting
    - Error handling and retries
    - Progress tracking
    - Resume capability
    """

    def __init__(
        self,
        api_url: str = "https://api.elections.kalshi.com/trade-api/v2",
        rate_limit: float = 5.0,  # requests per second
        max_retries: int = 3,
    ):
        self.api_url = api_url
        self.rate_limiter = RateLimiter(rate_limit)
        self.max_retries = max_retries

        # Initialize authentication
        self.auth = self._init_auth()

        # Statistics
        self.stats = {
            "total_fetched": 0,
            "success": 0,
            "not_found": 0,
            "errors": 0,
            "determined_yes": 0,
            "determined_no": 0,
            "not_determined": 0,
        }

        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None

    def _init_auth(self) -> KalshiAuth:
        """Initialize Kalshi authentication from environment."""
        from dotenv import load_dotenv

        # Load environment files in order (last loaded overrides)
        backend_dir = Path(__file__).parent.parent.parent.parent
        env_files = [
            backend_dir / ".env",
            backend_dir / ".env.local",
        ]

        for env_file in env_files:
            if env_file.exists():
                logger.info(f"Loading credentials from {env_file.name}")
                load_dotenv(env_file, override=True)

        try:
            # Use the same pattern as KalshiDemoTradingClient for proper key handling
            api_key_id = os.getenv("KALSHI_API_KEY_ID")
            private_key_content = os.getenv("KALSHI_PRIVATE_KEY_CONTENT")

            if not api_key_id:
                raise ValueError("KALSHI_API_KEY_ID not set")
            if not private_key_content:
                raise ValueError("KALSHI_PRIVATE_KEY_CONTENT not set")

            # Strip quotes if present (from shell quoting)
            if private_key_content.startswith('"') and private_key_content.endswith('"'):
                private_key_content = private_key_content[1:-1]

            # Create temporary file for private key with proper format handling
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pem', prefix='kalshi_key_')
            self._temp_key_file = temp_path

            with os.fdopen(temp_fd, 'w') as temp_file:
                # Replace escaped newlines with actual newlines (dotenv may escape them)
                formatted_key = private_key_content.replace('\\n', '\n')

                # The key should already have proper BEGIN/END markers
                # Just ensure it has proper line breaks
                temp_file.write(formatted_key)

            auth = KalshiAuth(api_key_id, temp_path)
            logger.info("Authentication initialized successfully")
            return auth
        except Exception as e:
            logger.error(f"Failed to initialize authentication: {e}")
            raise

    def _cleanup_temp_key(self):
        """Clean up temporary key file."""
        if hasattr(self, '_temp_key_file'):
            try:
                os.unlink(self._temp_key_file)
            except:
                pass

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self._cleanup_temp_key()

    def _create_auth_headers(self, method: str, path: str) -> Dict[str, str]:
        """Create authentication headers for API request."""
        full_path = f"/trade-api/v2{path}"
        headers = self.auth.create_auth_headers(method, full_path)
        headers["Content-Type"] = "application/json"
        return headers

    async def fetch_market(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch market data for a single ticker.

        Args:
            ticker: Market ticker (e.g., "KXNFLGAME-25DEC27BALGB-BAL")

        Returns:
            Market data dict or None if not found/error
        """
        if not self.session:
            raise RuntimeError("Fetcher not initialized. Use async context manager.")

        path = f"/markets/{ticker}"
        url = f"{self.api_url}{path}"

        for attempt in range(self.max_retries):
            try:
                await self.rate_limiter.wait()

                headers = self._create_auth_headers("GET", path)

                async with self.session.get(url, headers=headers) as response:
                    self.stats["total_fetched"] += 1

                    if response.status == 200:
                        data = await response.json()
                        market = data.get("market", data)
                        self.stats["success"] += 1

                        # Track outcome stats
                        if market.get("status") == "determined":
                            if market.get("result") == "yes":
                                self.stats["determined_yes"] += 1
                            elif market.get("result") == "no":
                                self.stats["determined_no"] += 1
                            else:
                                self.stats["not_determined"] += 1
                        else:
                            self.stats["not_determined"] += 1

                        return market

                    elif response.status == 404:
                        self.stats["not_found"] += 1
                        logger.debug(f"Market not found: {ticker}")
                        return None

                    elif response.status == 429:
                        # Rate limited - back off and retry
                        retry_after = int(response.headers.get("Retry-After", 5))
                        logger.warning(f"Rate limited. Waiting {retry_after}s...")
                        await asyncio.sleep(retry_after)
                        continue

                    else:
                        response_text = await response.text()
                        logger.warning(f"API error for {ticker}: {response.status} - {response_text[:200]}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                        self.stats["errors"] += 1
                        return None

            except aiohttp.ClientError as e:
                logger.warning(f"Client error fetching {ticker}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                self.stats["errors"] += 1
                return None

            except Exception as e:
                logger.error(f"Unexpected error fetching {ticker}: {e}")
                self.stats["errors"] += 1
                return None

        return None

    async def fetch_markets_batch(
        self,
        tickers: List[str],
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch market data for multiple tickers.

        Args:
            tickers: List of market tickers
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Dict mapping ticker -> market data (excludes failed fetches)
        """
        results = {}
        total = len(tickers)

        for i, ticker in enumerate(tickers):
            market_data = await self.fetch_market(ticker)
            if market_data:
                results[ticker] = market_data

            if progress_callback:
                progress_callback(i + 1, total)

            # Progress logging every 100 tickers
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{total} ({100 * (i + 1) / total:.1f}%)")

        return results

    def get_stats_summary(self) -> str:
        """Get formatted statistics summary."""
        s = self.stats
        total = s["success"] + s["not_found"] + s["errors"]

        lines = [
            "",
            "=" * 60,
            "MARKET OUTCOME FETCH SUMMARY",
            "=" * 60,
            f"  Total API calls:     {s['total_fetched']:,}",
            f"  Successful:          {s['success']:,}",
            f"  Not found:           {s['not_found']:,}",
            f"  Errors:              {s['errors']:,}",
            "",
            "  RESOLVED MARKETS:",
            f"    YES wins:          {s['determined_yes']:,}",
            f"    NO wins:           {s['determined_no']:,}",
            f"    Not determined:    {s['not_determined']:,}",
            "=" * 60,
        ]

        return "\n".join(lines)


def extract_unique_tickers_from_csv(csv_path: str) -> List[str]:
    """Extract unique market tickers from trades CSV."""
    tickers = set()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row.get('market_ticker', '')
            if ticker:
                tickers.add(ticker)

    logger.info(f"Extracted {len(tickers):,} unique tickers from {csv_path}")
    return sorted(list(tickers))


def load_existing_outcomes(output_path: str) -> Set[str]:
    """Load tickers that have already been fetched (for resume)."""
    existing = set()

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('ticker', '')
                if ticker:
                    existing.add(ticker)
        logger.info(f"Found {len(existing):,} existing outcomes in {output_path}")

    return existing


def save_outcomes_to_csv(outcomes: Dict[str, Dict[str, Any]], output_path: str, append: bool = False):
    """Save market outcomes to CSV."""
    mode = 'a' if append else 'w'

    # Define columns to export
    columns = [
        'ticker', 'status', 'result', 'close_time', 'expiration_time',
        'settlement_value', 'settlement_ts', 'title', 'subtitle',
        'event_ticker', 'category', 'yes_sub_title', 'no_sub_title',
        'open_time', 'open_interest', 'volume', 'volume_24h'
    ]

    with open(output_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        if not append or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            writer.writeheader()

        for ticker, market in outcomes.items():
            row = {
                'ticker': ticker,
                'status': market.get('status', ''),
                'result': market.get('result', ''),
                'close_time': market.get('close_time', ''),
                'expiration_time': market.get('expiration_time', ''),
                'settlement_value': market.get('settlement_value', ''),
                'settlement_ts': market.get('settlement_ts', ''),
                'title': market.get('title', ''),
                'subtitle': market.get('subtitle', ''),
                'event_ticker': market.get('event_ticker', ''),
                'category': market.get('category', ''),
                'yes_sub_title': market.get('yes_sub_title', ''),
                'no_sub_title': market.get('no_sub_title', ''),
                'open_time': market.get('open_time', ''),
                'open_interest': market.get('open_interest', ''),
                'volume': market.get('volume', ''),
                'volume_24h': market.get('volume_24h', ''),
            }
            writer.writerow(row)

    logger.info(f"Saved {len(outcomes):,} outcomes to {output_path}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fetch market outcomes from Kalshi API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch outcomes for all trades in CSV
  python fetch_market_outcomes.py --from-csv trades.csv --output outcomes.csv

  # Resume a previous run
  python fetch_market_outcomes.py --from-csv trades.csv --output outcomes.csv --resume

  # Fetch specific tickers
  python fetch_market_outcomes.py --tickers TICKER1,TICKER2 --output outcomes.csv

  # Control rate limit (default 5 req/sec)
  python fetch_market_outcomes.py --from-csv trades.csv --rate-limit 10
        """
    )

    parser.add_argument('--from-csv', type=str, metavar='FILE',
                       help='CSV file with trades (extracts unique market_ticker column)')
    parser.add_argument('--tickers', type=str,
                       help='Comma-separated list of tickers to fetch')
    parser.add_argument('--output', type=str, required=True, metavar='FILE',
                       help='Output CSV file for market outcomes')
    parser.add_argument('--resume', action='store_true',
                       help='Skip tickers already in output file')
    parser.add_argument('--rate-limit', type=float, default=5.0,
                       help='Requests per second (default: 5)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of tickers to fetch (for testing)')
    parser.add_argument('--batch-save', type=int, default=100,
                       help='Save to file every N tickers (default: 100)')

    args = parser.parse_args()

    # Determine tickers to fetch
    if args.from_csv:
        tickers = extract_unique_tickers_from_csv(args.from_csv)
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        parser.error("Must specify either --from-csv or --tickers")
        return

    # Handle resume
    if args.resume:
        existing = load_existing_outcomes(args.output)
        original_count = len(tickers)
        tickers = [t for t in tickers if t not in existing]
        logger.info(f"Resuming: {original_count - len(tickers):,} already fetched, {len(tickers):,} remaining")

    # Apply limit if specified
    if args.limit:
        tickers = tickers[:args.limit]
        logger.info(f"Limited to {len(tickers):,} tickers")

    if not tickers:
        logger.info("No tickers to fetch")
        return

    logger.info(f"Fetching outcomes for {len(tickers):,} tickers")
    logger.info(f"Rate limit: {args.rate_limit} requests/second")
    logger.info(f"Output: {args.output}")

    start_time = time.time()

    # Fetch outcomes
    async with MarketOutcomeFetcher(rate_limit=args.rate_limit) as fetcher:
        all_outcomes = {}

        # Process in batches for incremental saving
        batch_size = args.batch_save
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            outcomes = await fetcher.fetch_markets_batch(batch)
            all_outcomes.update(outcomes)

            # Incremental save
            if outcomes:
                save_outcomes_to_csv(outcomes, args.output, append=(i > 0 or args.resume))

            # Progress update
            fetched = min(i + batch_size, len(tickers))
            elapsed = time.time() - start_time
            rate = fetched / elapsed if elapsed > 0 else 0
            eta = (len(tickers) - fetched) / rate if rate > 0 else 0

            logger.info(
                f"Progress: {fetched:,}/{len(tickers):,} "
                f"({100 * fetched / len(tickers):.1f}%) | "
                f"Rate: {rate:.1f}/sec | "
                f"ETA: {eta/60:.1f} min"
            )

        # Final stats
        print(fetcher.get_stats_summary())

    elapsed = time.time() - start_time
    logger.info(f"Completed in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    asyncio.run(main())
