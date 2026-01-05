#!/usr/bin/env python3
"""
Research Data Updater - Idempotent script to update historical trade data.

This script:
1. Connects to production Supabase PostgreSQL
2. Fetches trades newer than the existing CSV data
3. Deduplicates by (market_ticker, timestamp, taker_side, count)
4. Appends new trades to historical_trades_ALL.csv
5. Fetches market outcomes for new tickers
6. Regenerates enriched CSV files

Usage:
    python update_research_data.py                    # Normal update from prod
    python update_research_data.py --dry-run          # Preview what would be updated
    python update_research_data.py --force            # Reprocess everything
    python update_research_data.py --env local        # Use local DB instead

Author: Claude Code
"""

import asyncio
import argparse
import csv
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict

# Add parent directories to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
RESEARCH_DIR = PROJECT_ROOT / "research"
DATA_DIR = RESEARCH_DIR / "data"
TRADES_DIR = DATA_DIR / "trades"
MARKETS_DIR = DATA_DIR / "markets"

sys.path.insert(0, str(BACKEND_DIR / "src"))

import asyncpg
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ResearchDataUpdater:
    """Updates research data files from production database."""

    # CSV file paths
    HISTORICAL_TRADES_CSV = TRADES_DIR / "historical_trades_ALL.csv"
    ENRICHED_TRADES_CSV = TRADES_DIR / "enriched_trades_ALL.csv"
    ENRICHED_RESOLVED_CSV = TRADES_DIR / "enriched_trades_resolved_ALL.csv"
    MARKET_OUTCOMES_CSV = MARKETS_DIR / "market_outcomes_ALL.csv"

    # Trade CSV columns (matching historical format)
    TRADE_COLUMNS = [
        'id', 'market_ticker', 'taker_side', 'count', 'yes_price', 'no_price',
        'timestamp', 'datetime', 'trade_price', 'cost_dollars', 'max_payout_dollars',
        'potential_profit_dollars', 'leverage_ratio'
    ]

    # Enriched CSV columns (with market metadata)
    ENRICHED_COLUMNS = TRADE_COLUMNS + [
        'event_ticker', 'category', 'title', 'subtitle', 'yes_sub_title', 'no_sub_title'
    ]

    # Resolved CSV columns (with settlement outcomes)
    RESOLVED_COLUMNS = ENRICHED_COLUMNS + [
        'status', 'result', 'outcome_known', 'trade_won', 'profit_loss_dollars'
    ]

    def __init__(
        self,
        env: str = "production",
        dry_run: bool = False,
        force: bool = False,
        skip_outcomes: bool = False,
        trades_only: bool = False,
        enriched_only: bool = False
    ):
        self.env = env
        self.dry_run = dry_run
        self.force = force
        self.skip_outcomes = skip_outcomes
        self.trades_only = trades_only
        self.enriched_only = enriched_only
        self.db_url = None
        self.conn = None

        # Statistics
        self.stats = {
            "existing_trades": 0,
            "new_trades_fetched": 0,
            "duplicates_skipped": 0,
            "new_trades_added": 0,
            "new_tickers": 0,
            "outcomes_fetched": 0,
        }

    def _load_env(self):
        """Load environment-specific .env file."""
        if self.env == "production":
            env_file = BACKEND_DIR / ".env.production"
        elif self.env == "local":
            env_file = BACKEND_DIR / ".env.local"
        else:
            env_file = BACKEND_DIR / f".env.{self.env}"

        if not env_file.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")

        load_dotenv(env_file, override=True)
        self.db_url = os.getenv("DATABASE_URL")

        if not self.db_url:
            raise ValueError(f"DATABASE_URL not set in {env_file}")

        logger.info(f"Loaded environment from {env_file.name}")

    async def connect(self):
        """Connect to database."""
        self._load_env()
        self.conn = await asyncpg.connect(self.db_url, command_timeout=120)
        logger.info("Connected to database")

    async def close(self):
        """Close database connection."""
        if self.conn:
            await self.conn.close()
            logger.info("Database connection closed")

    def _load_existing_trades(self) -> Tuple[Set[Tuple], int, int]:
        """
        Load existing trades from CSV and return deduplication keys.

        Returns:
            Tuple of (set of (ticker, ts, side, count) tuples, max_timestamp, max_id)
        """
        existing_keys = set()
        max_ts = 0
        max_id = 0

        if not self.HISTORICAL_TRADES_CSV.exists():
            logger.warning(f"Historical trades CSV not found: {self.HISTORICAL_TRADES_CSV}")
            return existing_keys, max_ts, max_id

        logger.info(f"Loading existing trades from {self.HISTORICAL_TRADES_CSV}")

        with open(self.HISTORICAL_TRADES_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('market_ticker', '')
                ts = int(row.get('timestamp', 0))
                side = row.get('taker_side', '')
                count = int(row.get('count', 0))
                trade_id = int(row.get('id', 0))

                # Create deduplication key
                key = (ticker, ts, side, count)
                existing_keys.add(key)

                max_ts = max(max_ts, ts)
                max_id = max(max_id, trade_id)

                self.stats["existing_trades"] += 1

        logger.info(f"Loaded {self.stats['existing_trades']:,} existing trades")
        logger.info(f"Max timestamp: {datetime.fromtimestamp(max_ts/1000) if max_ts else 'N/A'}")
        logger.info(f"Max ID: {max_id:,}")

        return existing_keys, max_ts, max_id

    async def _fetch_new_trades(self, after_ts: int) -> List[Dict]:
        """
        Fetch trades from database newer than given timestamp.

        Args:
            after_ts: Timestamp in milliseconds to fetch trades after

        Returns:
            List of trade dictionaries
        """
        logger.info(f"Fetching trades after {datetime.fromtimestamp(after_ts/1000)}")

        # Count first
        count = await self.conn.fetchval(
            "SELECT COUNT(*) FROM trades WHERE ts > $1",
            after_ts
        )
        logger.info(f"Found {count:,} trades in database after cutoff")

        if self.dry_run:
            logger.info("[DRY RUN] Would fetch these trades")
            return []

        # Fetch in batches to avoid memory issues
        batch_size = 100000
        offset = 0
        all_trades = []

        while True:
            logger.info(f"Fetching batch: offset={offset:,}, limit={batch_size:,}")

            rows = await self.conn.fetch("""
                SELECT
                    id, market_ticker, taker_side, count,
                    yes_price, no_price, ts
                FROM trades
                WHERE ts > $1
                ORDER BY ts ASC
                LIMIT $2 OFFSET $3
            """, after_ts, batch_size, offset)

            if not rows:
                break

            for row in rows:
                trade = dict(row)
                all_trades.append(trade)

            self.stats["new_trades_fetched"] += len(rows)
            offset += batch_size

            if len(rows) < batch_size:
                break

        logger.info(f"Fetched {len(all_trades):,} trades total")
        return all_trades

    def _calculate_trade_fields(self, trade: Dict) -> Dict:
        """
        Calculate derived fields for a trade row.

        Args:
            trade: Raw trade dict from database

        Returns:
            Trade dict with all calculated fields
        """
        ts = trade['ts']
        side = trade['taker_side']
        count = trade['count']
        yes_price = trade['yes_price']
        no_price = trade['no_price']

        # Trade price based on side
        trade_price = yes_price if side == 'yes' else no_price

        # Cost in dollars (price is in cents)
        cost_dollars = count * trade_price / 100.0

        # Max payout is count * $1 per contract
        max_payout_dollars = float(count)

        # Potential profit
        potential_profit_dollars = max_payout_dollars - cost_dollars

        # Leverage ratio
        leverage_ratio = potential_profit_dollars / cost_dollars if cost_dollars > 0 else 0

        return {
            'id': trade['id'],
            'market_ticker': trade['market_ticker'],
            'taker_side': side,
            'count': count,
            'yes_price': yes_price,
            'no_price': no_price,
            'timestamp': ts,
            'datetime': datetime.fromtimestamp(ts/1000).isoformat(),
            'trade_price': trade_price,
            'cost_dollars': round(cost_dollars, 2),
            'max_payout_dollars': max_payout_dollars,
            'potential_profit_dollars': round(potential_profit_dollars, 2),
            'leverage_ratio': leverage_ratio,
        }

    def _backup_file(self, filepath: Path):
        """Create backup of existing file."""
        if filepath.exists():
            backup_path = filepath.with_suffix(f".bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            shutil.copy2(filepath, backup_path)
            logger.info(f"Created backup: {backup_path.name}")

    def _append_trades_to_csv(self, trades: List[Dict], existing_keys: Set[Tuple], next_id: int):
        """
        Append new trades to CSV, deduplicating against existing data.

        Args:
            trades: List of trade dicts from database
            existing_keys: Set of (ticker, ts, side, count) tuples for dedup
            next_id: Next ID to assign to new trades
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would append trades to CSV")
            return

        # Backup existing file
        self._backup_file(self.HISTORICAL_TRADES_CSV)

        # Open CSV for appending
        file_exists = self.HISTORICAL_TRADES_CSV.exists()

        with open(self.HISTORICAL_TRADES_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.TRADE_COLUMNS)

            if not file_exists:
                writer.writeheader()

            for trade in trades:
                # Create dedup key
                key = (trade['market_ticker'], trade['ts'], trade['taker_side'], trade['count'])

                if key in existing_keys:
                    self.stats["duplicates_skipped"] += 1
                    continue

                # Add to existing keys to handle duplicates within batch
                existing_keys.add(key)

                # Calculate fields and assign new sequential ID
                row = self._calculate_trade_fields(trade)
                row['id'] = next_id
                next_id += 1

                writer.writerow(row)
                self.stats["new_trades_added"] += 1

        logger.info(f"Added {self.stats['new_trades_added']:,} new trades to CSV")
        logger.info(f"Skipped {self.stats['duplicates_skipped']:,} duplicates")

    def _load_market_outcomes(self) -> Dict[str, Dict]:
        """Load existing market outcomes from CSV."""
        outcomes = {}

        if not self.MARKET_OUTCOMES_CSV.exists():
            return outcomes

        with open(self.MARKET_OUTCOMES_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('ticker', '')
                if ticker:
                    outcomes[ticker] = row

        logger.info(f"Loaded {len(outcomes):,} existing market outcomes")
        return outcomes

    def _get_new_tickers(self, existing_outcomes: Dict[str, Dict]) -> List[str]:
        """Identify tickers in trades CSV that don't have outcomes yet."""
        all_tickers = set()

        with open(self.HISTORICAL_TRADES_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row.get('market_ticker', '')
                if ticker:
                    all_tickers.add(ticker)

        new_tickers = [t for t in all_tickers if t not in existing_outcomes]
        logger.info(f"Found {len(new_tickers):,} tickers without outcomes")

        self.stats["new_tickers"] = len(new_tickers)
        return sorted(new_tickers)

    async def _fetch_market_outcomes(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Fetch market outcomes from Kalshi API.

        This delegates to the existing fetch_market_outcomes.py script's logic.
        """
        if not tickers:
            return {}

        if self.dry_run:
            logger.info(f"[DRY RUN] Would fetch outcomes for {len(tickers):,} tickers")
            return {}

        # Import the fetcher
        sys.path.insert(0, str(RESEARCH_DIR / "analysis"))
        from fetch_market_outcomes import MarketOutcomeFetcher, save_outcomes_to_csv

        logger.info(f"Fetching outcomes for {len(tickers):,} tickers...")

        async with MarketOutcomeFetcher(rate_limit=10.0) as fetcher:
            outcomes = await fetcher.fetch_markets_batch(tickers)
            self.stats["outcomes_fetched"] = len(outcomes)

            # Save to CSV (append mode)
            if outcomes:
                save_outcomes_to_csv(outcomes, str(self.MARKET_OUTCOMES_CSV), append=True)
                logger.info(f"Appended {len(outcomes):,} outcomes to CSV")

        return outcomes

    def _regenerate_enriched_csv(self, outcomes: Dict[str, Dict]):
        """Regenerate enriched_trades_ALL.csv with market metadata."""
        if self.dry_run:
            logger.info("[DRY RUN] Would regenerate enriched trades CSV")
            return

        logger.info("Regenerating enriched trades CSV...")

        # Load all outcomes
        all_outcomes = self._load_market_outcomes()

        # Backup existing
        self._backup_file(self.ENRICHED_TRADES_CSV)

        # Process trades
        count = 0
        with open(self.HISTORICAL_TRADES_CSV, 'r') as infile, \
             open(self.ENRICHED_TRADES_CSV, 'w', newline='') as outfile:

            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=self.ENRICHED_COLUMNS)
            writer.writeheader()

            for row in reader:
                ticker = row.get('market_ticker', '')
                outcome = all_outcomes.get(ticker, {})

                # Add market metadata
                row['event_ticker'] = outcome.get('event_ticker', '')
                row['category'] = outcome.get('category', '')
                row['title'] = outcome.get('title', '')
                row['subtitle'] = outcome.get('subtitle', '')
                row['yes_sub_title'] = outcome.get('yes_sub_title', '')
                row['no_sub_title'] = outcome.get('no_sub_title', '')

                writer.writerow(row)
                count += 1

        logger.info(f"Wrote {count:,} enriched trades")

    def _regenerate_resolved_csv(self):
        """Regenerate enriched_trades_resolved_ALL.csv with settlement outcomes."""
        if self.dry_run:
            logger.info("[DRY RUN] Would regenerate resolved trades CSV")
            return

        logger.info("Regenerating resolved trades CSV...")

        # Load all outcomes
        all_outcomes = self._load_market_outcomes()

        # Backup existing
        self._backup_file(self.ENRICHED_RESOLVED_CSV)

        # Process enriched trades
        count = 0
        determined_count = 0

        with open(self.ENRICHED_TRADES_CSV, 'r') as infile, \
             open(self.ENRICHED_RESOLVED_CSV, 'w', newline='') as outfile:

            reader = csv.DictReader(infile)
            writer = csv.DictWriter(outfile, fieldnames=self.RESOLVED_COLUMNS)
            writer.writeheader()

            for row in reader:
                ticker = row.get('market_ticker', '')
                outcome = all_outcomes.get(ticker, {})

                status = outcome.get('status', '')
                result = outcome.get('result', '')

                # Determine if outcome is known
                outcome_known = status == 'determined' and result in ('yes', 'no')

                row['status'] = status
                row['result'] = result
                row['outcome_known'] = outcome_known

                if outcome_known:
                    determined_count += 1
                    side = row.get('taker_side', '')
                    cost_dollars = float(row.get('cost_dollars', 0))
                    max_payout = float(row.get('max_payout_dollars', 0))

                    # Trade wins if side matches result
                    trade_won = (side == result)
                    profit_loss = max_payout - cost_dollars if trade_won else -cost_dollars

                    row['trade_won'] = trade_won
                    row['profit_loss_dollars'] = round(profit_loss, 2)
                else:
                    row['trade_won'] = ''
                    row['profit_loss_dollars'] = ''

                writer.writerow(row)
                count += 1

        logger.info(f"Wrote {count:,} resolved trades ({determined_count:,} with known outcomes)")

    async def run(self):
        """Run the full update process."""
        logger.info("=" * 60)
        logger.info(f"RESEARCH DATA UPDATE - {datetime.now()}")
        logger.info(f"Environment: {self.env}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info(f"Force: {self.force}")
        logger.info(f"Skip outcomes: {self.skip_outcomes}")
        logger.info(f"Trades only: {self.trades_only}")
        logger.info(f"Enriched only: {self.enriched_only}")
        logger.info("=" * 60)

        try:
            # Enriched-only mode: just regenerate CSVs from existing data
            if self.enriched_only:
                logger.info("Enriched-only mode: regenerating enriched CSVs from existing data")
                existing_outcomes = self._load_market_outcomes()
                self._regenerate_enriched_csv(existing_outcomes)
                self._regenerate_resolved_csv()
                self._print_summary()
                return

            await self.connect()

            # Step 1: Load existing trades and get cutoff
            existing_keys, max_ts, max_id = self._load_existing_trades()

            if self.force:
                # Force mode: start from beginning
                max_ts = 0
                max_id = 0
                existing_keys = set()
                logger.info("Force mode: ignoring existing data")

            # Step 2: Fetch new trades from database
            new_trades = await self._fetch_new_trades(max_ts)

            if new_trades:
                # Step 3: Append new trades to CSV
                self._append_trades_to_csv(new_trades, existing_keys, max_id + 1)

            # If trades-only mode, stop here
            if self.trades_only:
                logger.info("Trades-only mode: skipping outcomes and enrichment")
                self._print_summary()
                return

            # Step 4: Identify new tickers needing outcomes
            existing_outcomes = self._load_market_outcomes()
            new_tickers = self._get_new_tickers(existing_outcomes)

            if new_tickers and not self.skip_outcomes:
                # Step 5: Fetch market outcomes for new tickers
                await self._fetch_market_outcomes(new_tickers)
            elif self.skip_outcomes:
                logger.info(f"Skipping outcome fetch ({len(new_tickers):,} new tickers)")

            # Step 6: Regenerate enriched CSVs
            if new_trades or self.force:
                self._regenerate_enriched_csv(existing_outcomes)
                self._regenerate_resolved_csv()

            # Print summary
            self._print_summary()

        finally:
            await self.close()

    def _print_summary(self):
        """Print update summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("UPDATE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Existing trades:        {self.stats['existing_trades']:,}")
        logger.info(f"  New trades fetched:     {self.stats['new_trades_fetched']:,}")
        logger.info(f"  Duplicates skipped:     {self.stats['duplicates_skipped']:,}")
        logger.info(f"  New trades added:       {self.stats['new_trades_added']:,}")
        logger.info(f"  New tickers found:      {self.stats['new_tickers']:,}")
        logger.info(f"  Outcomes fetched:       {self.stats['outcomes_fetched']:,}")
        logger.info("=" * 60)

        if self.dry_run:
            logger.info("")
            logger.info("[DRY RUN] No files were modified")


def main():
    parser = argparse.ArgumentParser(
        description='Update research data from production database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Normal update from production database
  python update_research_data.py

  # Preview what would be updated (no changes)
  python update_research_data.py --dry-run

  # Force full reprocessing
  python update_research_data.py --force

  # Use local database instead
  python update_research_data.py --env local

  # Fetch trades only (skip outcomes and enrichment) - fast update
  python update_research_data.py --trades-only

  # Update but skip slow outcome fetch (for quick daily updates)
  python update_research_data.py --skip-outcomes

  # Just regenerate enriched CSVs from existing data (no DB connection)
  python update_research_data.py --enriched-only
        """
    )

    parser.add_argument(
        '--env', type=str, default='production',
        choices=['production', 'local', 'paper'],
        help='Environment to use (default: production)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force full reprocessing, ignoring existing data'
    )
    parser.add_argument(
        '--skip-outcomes', action='store_true',
        help='Skip fetching market outcomes (fast update, outcomes fetched separately)'
    )
    parser.add_argument(
        '--trades-only', action='store_true',
        help='Only fetch and append trades (no outcomes, no enriched CSVs)'
    )
    parser.add_argument(
        '--enriched-only', action='store_true',
        help='Only regenerate enriched CSVs from existing data (no DB connection)'
    )

    args = parser.parse_args()

    updater = ResearchDataUpdater(
        env=args.env,
        dry_run=args.dry_run,
        force=args.force,
        skip_outcomes=args.skip_outcomes,
        trades_only=args.trades_only,
        enriched_only=args.enriched_only
    )

    asyncio.run(updater.run())


if __name__ == "__main__":
    main()
