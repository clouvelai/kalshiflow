"""
PostgreSQL database setup and operations for storing Kalshi trade data.
Unified database implementation using PostgreSQL for all environments.
"""

import os
import asyncio
import asyncpg
import socket
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union
from decimal import Decimal
from .models import Trade
import json
import logging

logger = logging.getLogger(__name__)


class Database:
    """PostgreSQL database manager for trade data storage."""
    
    def __init__(self, database_url: str = None, pool_size: int = 10):
        """Initialize database with connection pool."""
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.pool_size = pool_size
        self._pool = None
        self._init_lock = asyncio.Lock()
        self._initialized = False
    
    
    def _convert_decimals_to_float(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Decimal values to float for JSON serialization."""
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = float(value)
        return data
    
    async def initialize(self):
        """Initialize database connection pool and schema."""
        async with self._init_lock:
            if self._initialized:
                return
                
            if not self.database_url:
                raise ValueError("DATABASE_URL environment variable is required")
        
        try:
            # Create connection pool
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=20,
                statement_cache_size=0,  # Required for pgbouncer compatibility
                server_settings={
                    'application_name': 'kalshiflow',
                    'timezone': 'UTC'
                }
            )
            
            # Test the connection
            async with self._pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
                
            # Run migrations only if not using local Supabase
            # Local Supabase handles migrations via supabase/migrations/ 
            supabase_url = os.getenv("SUPABASE_URL", "")
            if not supabase_url.startswith("http://localhost"):
                await self._run_migrations()
            else:
                logger.info("Skipping migrations for local Supabase (handled by Supabase CLI)")
            
            logger.info(f"PostgreSQL database initialized with pool size {self.pool_size}")
            self._initialized = True
                
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise
    
    async def setup_database(self):
        """Alias for initialize() for consistency with previous interface."""
        await self.initialize()
    
    async def store_trade(self, trade_data: Dict[str, Any]) -> int:
        """Store trade data in PostgreSQL database."""
        received_at = int(datetime.now().timestamp() * 1000)
        
        async with self.get_connection() as conn:
            row_id = await conn.fetchval('''
                INSERT INTO trades (
                    market_ticker, yes_price, no_price, yes_price_dollars, 
                    no_price_dollars, count, taker_side, ts, received_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            ''', 
                trade_data['ticker'],
                trade_data.get('yes_bid', trade_data.get('price', 50)),
                trade_data.get('no_bid', 100 - trade_data.get('price', 50)),
                trade_data.get('yes_ask', trade_data.get('price', 50)) / 100.0,
                trade_data.get('no_ask', (100 - trade_data.get('price', 50))) / 100.0,
                trade_data['volume'],
                trade_data['side'],
                trade_data['timestamp'],
                received_at
            )
            return row_id
    
    async def close(self):
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("PostgreSQL database pool closed")
    
    async def _run_migrations(self):
        """Run database migrations."""
        migrations_dir = os.path.join(os.path.dirname(__file__), '../../migrations')
        
        async with self._pool.acquire() as conn:
            # Create migrations table if not exists
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS migrations (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) NOT NULL UNIQUE,
                    applied_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Run migration files in order
            migration_files = [
                '001_initial_schema.sql', 
                '002_indexes.sql', 
                '003_fix_constraints.sql',
                '004_optimize_production_indexes.sql',
                '005_add_production_constraints.sql',
                '006_production_optimizations.sql'
            ]
            
            for filename in migration_files:
                # Check if migration already applied
                existing = await conn.fetchrow(
                    "SELECT id FROM migrations WHERE filename = $1",
                    filename
                )
                
                if existing:
                    logger.debug(f"Migration {filename} already applied")
                    continue
                
                # Read and execute migration
                migration_path = os.path.join(migrations_dir, filename)
                if os.path.exists(migration_path):
                    with open(migration_path, 'r') as f:
                        migration_sql = f.read()
                    
                    try:
                        # Handle concurrent index creation separately 
                        if 'CREATE INDEX CONCURRENTLY' in migration_sql:
                            # Split statements and execute non-concurrent ones in transaction
                            statements = [s.strip() for s in migration_sql.split(';') if s.strip()]
                            
                            # Execute non-concurrent statements in transaction
                            non_concurrent_stmts = [s for s in statements 
                                                  if 'CREATE INDEX CONCURRENTLY' not in s and s]
                            concurrent_stmts = [s for s in statements 
                                              if 'CREATE INDEX CONCURRENTLY' in s]
                            
                            # Execute non-concurrent statements in transaction
                            if non_concurrent_stmts:
                                async with conn.transaction():
                                    for stmt in non_concurrent_stmts:
                                        if stmt:
                                            await conn.execute(stmt)
                            
                            # Execute concurrent statements outside transaction
                            for stmt in concurrent_stmts:
                                if stmt:
                                    try:
                                        await conn.execute(stmt)
                                    except Exception as idx_err:
                                        # Log but don't fail on index creation errors
                                        logger.warning(f"Index creation warning in {filename}: {idx_err}")
                            
                            # Mark migration as applied
                            await conn.execute(
                                "INSERT INTO migrations (filename) VALUES ($1)",
                                filename
                            )
                        else:
                            # Execute regular migration in transaction
                            async with conn.transaction():
                                await conn.execute(migration_sql)
                                await conn.execute(
                                    "INSERT INTO migrations (filename) VALUES ($1)",
                                    filename
                                )
                        
                        logger.info(f"Applied migration: {filename}")
                        
                    except Exception as e:
                        logger.error(f"Failed to apply migration {filename}: {e}")
                        raise
                else:
                    logger.warning(f"Migration file not found: {migration_path}")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self._initialized:
            await self.initialize()
        
        async with self._pool.acquire() as conn:
            yield conn
    
    async def insert_trade(self, trade: Trade) -> int:
        """Insert a trade record and return the row ID."""
        received_at = int(datetime.now().timestamp() * 1000)
        
        async with self.get_connection() as conn:
            row_id = await conn.fetchval('''
                INSERT INTO trades (
                    market_ticker, yes_price, no_price, yes_price_dollars, 
                    no_price_dollars, count, taker_side, ts, received_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            ''', 
                trade.market_ticker,
                trade.yes_price,
                trade.no_price,
                trade.yes_price_dollars,
                trade.no_price_dollars,
                trade.count,
                trade.taker_side,
                trade.ts,
                received_at
            )
            return row_id
    
    async def get_recent_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Get recent trades ordered by timestamp descending."""
        async with self.get_connection() as conn:
            rows = await conn.fetch('''
                SELECT * FROM trades 
                ORDER BY ts DESC 
                LIMIT $1
            ''', limit)
            # Convert Decimal values to float for JSON serialization
            return [self._convert_decimals_to_float(dict(row)) for row in rows]
    
    async def get_trades_for_ticker(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a specific ticker."""
        async with self.get_connection() as conn:
            rows = await conn.fetch('''
                SELECT * FROM trades 
                WHERE market_ticker = $1
                ORDER BY ts DESC 
                LIMIT $2
            ''', ticker, limit)
            return [self._convert_decimals_to_float(dict(row)) for row in rows]
    
    async def get_trades_in_window(self, window_minutes: int = 10) -> List[Dict[str, Any]]:
        """Get trades within a time window (minutes ago to now)."""
        cutoff_ts = int((datetime.now().timestamp() - window_minutes * 60) * 1000)
        
        async with self.get_connection() as conn:
            rows = await conn.fetch('''
                SELECT * FROM trades 
                WHERE ts >= $1
                ORDER BY ts DESC
            ''', cutoff_ts)
            return [self._convert_decimals_to_float(dict(row)) for row in rows]
    
    async def get_ticker_stats(self, ticker: str, window_minutes: int = 10) -> Dict[str, Any]:
        """Get aggregated statistics for a ticker within a time window."""
        cutoff_ts = int((datetime.now().timestamp() - window_minutes * 60) * 1000)
        
        async with self.get_connection() as conn:
            row = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as trade_count,
                    SUM(count) as total_volume,
                    SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END) as yes_volume,
                    SUM(CASE WHEN taker_side = 'no' THEN count ELSE 0 END) as no_volume,
                    AVG(yes_price_dollars) as avg_yes_price,
                    AVG(no_price_dollars) as avg_no_price,
                    MAX(ts) as last_trade_ts
                FROM trades 
                WHERE market_ticker = $1 AND ts >= $2
            ''', ticker, cutoff_ts)
            return self._convert_decimals_to_float(dict(row)) if row else {}
    
    async def cleanup_old_trades(self, days_to_keep: int = 30):
        """Remove trades older than specified days."""
        cutoff_ts = int((datetime.now().timestamp() - days_to_keep * 24 * 60 * 60) * 1000)
        
        async with self.get_connection() as conn:
            result = await conn.execute('''
                DELETE FROM trades WHERE ts < $1
            ''', cutoff_ts)
            # Parse result like "DELETE 1234"
            return int(result.split()[1]) if result.split()[1].isdigit() else 0
    
    async def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics for debugging."""
        async with self.get_connection() as conn:
            # Get all stats in one query for efficiency
            stats = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_trades,
                    MIN(ts) as oldest_trade,
                    MAX(ts) as newest_trade,
                    COUNT(DISTINCT market_ticker) as unique_tickers
                FROM trades
            ''')
            
            return {
                "total_trades": stats["total_trades"] if stats else 0,
                "oldest_trade": stats["oldest_trade"] if stats else None,
                "newest_trade": stats["newest_trade"] if stats else None,
                "unique_tickers": stats["unique_tickers"] if stats else 0,
                "database_type": "PostgreSQL",
                "pool_size": self.pool_size
            }
    
    # Market metadata methods
    async def insert_or_update_market(self, ticker: str, title: str, category: str = None,
                                    liquidity_dollars: float = None, open_interest: int = None,
                                    latest_expiration_time: str = None, raw_market_data: Union[str, dict] = None) -> bool:
        """Insert or update market metadata using PostgreSQL UPSERT."""
        # Handle raw_market_data serialization for PostgreSQL JSONB
        json_data = None
        if raw_market_data:
            # If raw_market_data is already a dict, use it directly
            if isinstance(raw_market_data, dict):
                json_data = raw_market_data
            # If raw_market_data is a string, parse it to validate JSON
            elif isinstance(raw_market_data, str):
                try:
                    json_data = json.loads(raw_market_data)
                except json.JSONDecodeError:
                    json_data = {"raw": raw_market_data}
            else:
                json_data = {"raw": str(raw_market_data)}
        
        # Parse expiration time
        expiration_dt = None
        if latest_expiration_time:
            try:
                expiration_dt = datetime.fromisoformat(latest_expiration_time.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                expiration_dt = None
        
        async with self.get_connection() as conn:
            # Convert json_data to JSON string if it's a dict
            json_param = None
            if json_data is not None:
                if isinstance(json_data, dict):
                    json_param = json.dumps(json_data)
                else:
                    json_param = json_data
            
            await conn.execute('''
                INSERT INTO markets (
                    ticker, title, category, liquidity_dollars, open_interest, 
                    latest_expiration_time, raw_market_data, created_at, updated_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT (ticker) DO UPDATE SET 
                    title = EXCLUDED.title,
                    category = EXCLUDED.category,
                    liquidity_dollars = EXCLUDED.liquidity_dollars,
                    open_interest = EXCLUDED.open_interest,
                    latest_expiration_time = EXCLUDED.latest_expiration_time,
                    raw_market_data = EXCLUDED.raw_market_data,
                    updated_at = CURRENT_TIMESTAMP
            ''', ticker, title, category, liquidity_dollars, open_interest, 
                 expiration_dt, json_param)
            
            return True
    
    async def get_market_metadata(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get market metadata for a ticker."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM markets WHERE ticker = $1
            ''', ticker)
            return dict(row) if row else None
    
    async def get_markets_metadata(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market metadata for multiple tickers."""
        if not tickers:
            return {}
        
        async with self.get_connection() as conn:
            rows = await conn.fetch('''
                SELECT * FROM markets WHERE ticker = ANY($1)
            ''', tickers)
            return {row['ticker']: dict(row) for row in rows}
    
    async def market_exists(self, ticker: str) -> bool:
        """Check if market metadata exists for a ticker."""
        async with self.get_connection() as conn:
            exists = await conn.fetchval('''
                SELECT EXISTS(SELECT 1 FROM markets WHERE ticker = $1)
            ''', ticker)
            return exists
    
    async def get_all_cached_markets(self) -> List[Dict[str, Any]]:
        """Get all cached market metadata."""
        async with self.get_connection() as conn:
            rows = await conn.fetch('SELECT * FROM markets ORDER BY updated_at DESC')
            return [dict(row) for row in rows]
    
    # Recovery methods for warm restart functionality
    async def get_trades_for_recovery(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get trades for warm restart recovery within specified hours."""
        cutoff_ts = int((datetime.now().timestamp() - hours * 3600) * 1000)
        # Filter out corrupted timestamps from 1970 (before 2020-01-01)
        min_valid_ts = int(datetime(2020, 1, 1).timestamp() * 1000)
        
        async with self.get_connection() as conn:
            rows = await conn.fetch('''
                SELECT * FROM trades 
                WHERE ts >= $1 AND ts >= $2
                ORDER BY ts ASC
            ''', cutoff_ts, min_valid_ts)
            return [self._convert_decimals_to_float(dict(row)) for row in rows]
    
    async def get_trades_for_minute_recovery(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get trades for minute-level recovery within specified minutes."""
        cutoff_ts = int((datetime.now().timestamp() - minutes * 60) * 1000)
        # Filter out corrupted timestamps from 1970 (before 2020-01-01)  
        min_valid_ts = int(datetime(2020, 1, 1).timestamp() * 1000)
        
        async with self.get_connection() as conn:
            rows = await conn.fetch('''
                SELECT * FROM trades 
                WHERE ts >= $1 AND ts >= $2
                ORDER BY ts ASC
            ''', cutoff_ts, min_valid_ts)
            return [self._convert_decimals_to_float(dict(row)) for row in rows]
    
    async def get_recovery_trade_count(self, hours: int = 24) -> int:
        """Get count of trades available for recovery to estimate processing time."""
        cutoff_ts = int((datetime.now().timestamp() - hours * 3600) * 1000)
        # Filter out corrupted timestamps from 1970 (before 2020-01-01)
        min_valid_ts = int(datetime(2020, 1, 1).timestamp() * 1000)
        
        async with self.get_connection() as conn:
            count = await conn.fetchval('''
                SELECT COUNT(*) FROM trades 
                WHERE ts >= $1 AND ts >= $2
            ''', cutoff_ts, min_valid_ts)
            return count or 0


# Global database instance
_db_instance = None

def get_database() -> Database:
    """Get the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance