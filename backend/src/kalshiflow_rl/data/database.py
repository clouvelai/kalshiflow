"""
PostgreSQL database schema and operations for RL Trading Subsystem.

Manages all RL-specific tables including orderbook snapshots, deltas,
model registry, trading episodes, and trading actions. Provides
async connection management and batch write operations.
"""

import asyncio
import asyncpg
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any, Union
from decimal import Decimal

from ..config import config

logger = logging.getLogger("kalshiflow_rl.database")


class RLDatabase:
    """PostgreSQL database manager for RL Trading Subsystem."""
    
    def __init__(self, database_url: str = None):
        """Initialize database with connection pool."""
        self.database_url = database_url or config.DATABASE_URL
        self._pool = None
        self._init_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize database connection pool and create RL schema."""
        async with self._init_lock:
            if self._initialized:
                return
                
            if not self.database_url:
                raise ValueError("DATABASE_URL is required for RL database")
        
            try:
                # Create connection pool
                self._pool = await asyncpg.create_pool(
                    self.database_url,
                    min_size=config.DB_POOL_MIN_SIZE,
                    max_size=config.DB_POOL_MAX_SIZE,
                    command_timeout=config.DB_POOL_TIMEOUT,
                    statement_cache_size=0,  # Required for pgbouncer compatibility
                    server_settings={
                        'application_name': 'kalshiflow_rl',
                        'timezone': 'UTC'
                    }
                )
                
                # Test the connection
                async with self._pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                
                # Create RL tables
                await self._create_rl_schema()
                
                logger.info(f"RL database initialized with pool size {config.DB_POOL_MAX_SIZE}")
                self._initialized = True
                
            except Exception as e:
                logger.error(f"Failed to initialize RL database: {e}")
                raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool."""
        if not self._pool:
            await self.initialize()
        
        async with self._pool.acquire() as conn:
            yield conn
    
    async def _create_rl_schema(self):
        """Create all RL-specific database tables and indexes."""
        async with self.get_connection() as conn:
            # Create tables in dependency order
            await self._create_orderbook_snapshots_table(conn)
            await self._create_orderbook_deltas_table(conn)
            await self._create_models_table(conn)
            await self._create_trading_episodes_table(conn)
            await self._create_trading_actions_table(conn)
            
            # Analyze tables for optimal query planning
            await conn.execute("ANALYZE rl_orderbook_snapshots")
            await conn.execute("ANALYZE rl_orderbook_deltas")
            await conn.execute("ANALYZE rl_models")
            await conn.execute("ANALYZE rl_trading_episodes")
            await conn.execute("ANALYZE rl_trading_actions")
            
            logger.info("RL database schema created successfully")
    
    async def _create_orderbook_snapshots_table(self, conn: asyncpg.Connection):
        """Create orderbook_snapshots table with full state snapshots."""
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS rl_orderbook_snapshots (
                id BIGSERIAL PRIMARY KEY,
                market_ticker VARCHAR(100) NOT NULL,
                timestamp_ms BIGINT NOT NULL,
                sequence_number BIGINT NOT NULL,
                yes_bids JSONB NOT NULL,  -- {price: size} mapping
                yes_asks JSONB NOT NULL,  -- {price: size} mapping
                no_bids JSONB NOT NULL,   -- {price: size} mapping
                no_asks JSONB NOT NULL,   -- {price: size} mapping
                yes_spread INTEGER,       -- spread in cents
                no_spread INTEGER,        -- spread in cents
                yes_mid_price DECIMAL(10,4),
                no_mid_price DECIMAL(10,4),
                total_volume BIGINT,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Create indexes for performance
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_market_time 
                ON rl_orderbook_snapshots(market_ticker, timestamp_ms DESC);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_sequence 
                ON rl_orderbook_snapshots(market_ticker, sequence_number DESC);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp 
                ON rl_orderbook_snapshots(timestamp_ms DESC);
        ''')
        
        # Add constraints
        await conn.execute('''
            ALTER TABLE rl_orderbook_snapshots 
            ADD CONSTRAINT IF NOT EXISTS chk_snapshots_timestamp_valid 
            CHECK (timestamp_ms > 1577836800000);  -- After 2020-01-01
        ''')
        
        await conn.execute('''
            ALTER TABLE rl_orderbook_snapshots 
            ADD CONSTRAINT IF NOT EXISTS chk_snapshots_sequence_positive 
            CHECK (sequence_number >= 0);
        ''')
        
        await conn.execute('''
            COMMENT ON TABLE rl_orderbook_snapshots IS 'Full orderbook state snapshots for RL training data';
        ''')
    
    async def _create_orderbook_deltas_table(self, conn: asyncpg.Connection):
        """Create orderbook_deltas table for incremental updates."""
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS rl_orderbook_deltas (
                id BIGSERIAL PRIMARY KEY,
                market_ticker VARCHAR(100) NOT NULL,
                timestamp_ms BIGINT NOT NULL,
                sequence_number BIGINT NOT NULL,
                side VARCHAR(10) NOT NULL,     -- 'yes' or 'no'
                action VARCHAR(10) NOT NULL,   -- 'add', 'remove', 'update'
                price INTEGER NOT NULL,       -- price in cents
                old_size BIGINT,              -- previous size (for updates/removes)
                new_size BIGINT,              -- new size (for adds/updates)
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Create indexes for performance
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_deltas_market_seq 
                ON rl_orderbook_deltas(market_ticker, sequence_number DESC);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_deltas_timestamp 
                ON rl_orderbook_deltas(timestamp_ms DESC);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_deltas_market_time_side 
                ON rl_orderbook_deltas(market_ticker, timestamp_ms DESC, side);
        ''')
        
        # Add constraints
        await conn.execute('''
            ALTER TABLE rl_orderbook_deltas 
            ADD CONSTRAINT IF NOT EXISTS chk_deltas_side 
            CHECK (side IN ('yes', 'no'));
        ''')
        
        await conn.execute('''
            ALTER TABLE rl_orderbook_deltas 
            ADD CONSTRAINT IF NOT EXISTS chk_deltas_action 
            CHECK (action IN ('add', 'remove', 'update'));
        ''')
        
        await conn.execute('''
            ALTER TABLE rl_orderbook_deltas 
            ADD CONSTRAINT IF NOT EXISTS chk_deltas_price_positive 
            CHECK (price > 0);
        ''')
        
        await conn.execute('''
            COMMENT ON TABLE rl_orderbook_deltas IS 'Incremental orderbook updates for efficient state reconstruction';
        ''')
    
    async def _create_models_table(self, conn: asyncpg.Connection):
        """Create models table for RL model registry."""
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS rl_models (
                id BIGSERIAL PRIMARY KEY,
                model_name VARCHAR(200) NOT NULL,
                version VARCHAR(50) NOT NULL,
                algorithm VARCHAR(50) NOT NULL,     -- 'PPO', 'A2C', etc.
                market_ticker VARCHAR(100) NOT NULL,
                file_path TEXT NOT NULL,           -- path to model file
                hyperparameters JSONB,             -- training hyperparameters
                training_metrics JSONB,            -- training performance metrics
                validation_metrics JSONB,          -- validation performance metrics
                status VARCHAR(20) NOT NULL DEFAULT 'training',  -- 'training', 'active', 'retired', 'failed'
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name, version)
            );
        ''')
        
        # Create indexes
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_models_status_created 
                ON rl_models(status, created_at DESC);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_models_market_status 
                ON rl_models(market_ticker, status);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_models_name_version 
                ON rl_models(model_name, version);
        ''')
        
        # Add constraints
        await conn.execute('''
            ALTER TABLE rl_models 
            ADD CONSTRAINT IF NOT EXISTS chk_models_status 
            CHECK (status IN ('training', 'active', 'retired', 'failed'));
        ''')
        
        # Add updated_at trigger
        await conn.execute('''
            CREATE OR REPLACE FUNCTION update_rl_models_updated_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        ''')
        
        await conn.execute('''
            CREATE TRIGGER update_rl_models_updated_at 
                BEFORE UPDATE ON rl_models 
                FOR EACH ROW 
                EXECUTE FUNCTION update_rl_models_updated_at();
        ''')
        
        await conn.execute('''
            COMMENT ON TABLE rl_models IS 'Registry of trained RL models with metadata and performance metrics';
        ''')
    
    async def _create_trading_episodes_table(self, conn: asyncpg.Connection):
        """Create trading_episodes table for training session tracking."""
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS rl_trading_episodes (
                id BIGSERIAL PRIMARY KEY,
                model_id BIGINT REFERENCES rl_models(id),
                episode_number INTEGER NOT NULL,
                market_ticker VARCHAR(100) NOT NULL,
                start_timestamp_ms BIGINT NOT NULL,
                end_timestamp_ms BIGINT NOT NULL,
                start_balance DECIMAL(15,4) NOT NULL,
                end_balance DECIMAL(15,4) NOT NULL,
                total_return DECIMAL(10,4) NOT NULL,      -- percentage return
                max_drawdown DECIMAL(10,4),               -- maximum loss percentage
                sharpe_ratio DECIMAL(10,4),               -- risk-adjusted return
                num_actions INTEGER NOT NULL DEFAULT 0,   -- number of actions taken
                num_trades INTEGER NOT NULL DEFAULT 0,    -- number of completed trades
                episode_reward DECIMAL(15,8),             -- total episode reward
                episode_length INTEGER NOT NULL,          -- steps in episode
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Create indexes
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_episodes_model_episode 
                ON rl_trading_episodes(model_id, episode_number);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_episodes_market_time 
                ON rl_trading_episodes(market_ticker, start_timestamp_ms DESC);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_episodes_performance 
                ON rl_trading_episodes(total_return DESC, sharpe_ratio DESC);
        ''')
        
        # Add constraints
        await conn.execute('''
            ALTER TABLE rl_trading_episodes 
            ADD CONSTRAINT IF NOT EXISTS chk_episodes_timeframe 
            CHECK (end_timestamp_ms > start_timestamp_ms);
        ''')
        
        await conn.execute('''
            ALTER TABLE rl_trading_episodes 
            ADD CONSTRAINT IF NOT EXISTS chk_episodes_episode_positive 
            CHECK (episode_number >= 0);
        ''')
        
        await conn.execute('''
            COMMENT ON TABLE rl_trading_episodes IS 'Training episode tracking with performance metrics';
        ''')
    
    async def _create_trading_actions_table(self, conn: asyncpg.Connection):
        """Create trading_actions table for detailed action logging."""
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS rl_trading_actions (
                id BIGSERIAL PRIMARY KEY,
                episode_id BIGINT REFERENCES rl_trading_episodes(id),
                action_timestamp_ms BIGINT NOT NULL,
                step_number INTEGER NOT NULL,
                action_type VARCHAR(20) NOT NULL,         -- 'buy_yes', 'buy_no', 'sell_yes', 'sell_no', 'hold'
                price INTEGER,                            -- price in cents (null for hold)
                quantity BIGINT,                          -- quantity (null for hold)
                position_before JSONB,                    -- position before action {yes_shares: X, no_shares: Y, balance: Z}
                position_after JSONB,                     -- position after action
                reward DECIMAL(15,8),                     -- immediate reward for this action
                observation JSONB,                        -- market observation at action time
                model_confidence DECIMAL(5,4),            -- model confidence [0,1] if available
                executed BOOLEAN DEFAULT FALSE,          -- whether action was executed (for paper trading)
                execution_price INTEGER,                 -- actual execution price if different from intended
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        
        # Create indexes
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_actions_episode_step 
                ON rl_trading_actions(episode_id, step_number);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_actions_timestamp 
                ON rl_trading_actions(action_timestamp_ms DESC);
        ''')
        
        await conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_actions_type_timestamp 
                ON rl_trading_actions(action_type, action_timestamp_ms DESC);
        ''')
        
        # Add constraints
        await conn.execute('''
            ALTER TABLE rl_trading_actions 
            ADD CONSTRAINT IF NOT EXISTS chk_actions_type 
            CHECK (action_type IN ('buy_yes', 'buy_no', 'sell_yes', 'sell_no', 'hold'));
        ''')
        
        await conn.execute('''
            ALTER TABLE rl_trading_actions 
            ADD CONSTRAINT IF NOT EXISTS chk_actions_step_positive 
            CHECK (step_number >= 0);
        ''')
        
        await conn.execute('''
            COMMENT ON TABLE rl_trading_actions IS 'Detailed logging of all trading actions and decisions';
        ''')
    
    async def close(self):
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()
            self._initialized = False
            logger.info("RL database connection pool closed")
    
    # Batch write operations for high-performance ingestion
    
    async def batch_insert_snapshots(self, snapshots: List[Dict[str, Any]]) -> int:
        """Batch insert orderbook snapshots."""
        if not snapshots:
            return 0
        
        async with self.get_connection() as conn:
            records = []
            for snap in snapshots:
                records.append((
                    snap['market_ticker'],
                    snap['timestamp_ms'],
                    snap['sequence_number'],
                    snap['yes_bids'],
                    snap['yes_asks'],
                    snap['no_bids'],
                    snap['no_asks'],
                    snap.get('yes_spread'),
                    snap.get('no_spread'),
                    snap.get('yes_mid_price'),
                    snap.get('no_mid_price'),
                    snap.get('total_volume', 0)
                ))
            
            query = '''
                INSERT INTO rl_orderbook_snapshots (
                    market_ticker, timestamp_ms, sequence_number,
                    yes_bids, yes_asks, no_bids, no_asks,
                    yes_spread, no_spread, yes_mid_price, no_mid_price, total_volume
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            '''
            
            await conn.executemany(query, records)
            return len(records)
    
    async def batch_insert_deltas(self, deltas: List[Dict[str, Any]]) -> int:
        """Batch insert orderbook deltas."""
        if not deltas:
            return 0
        
        async with self.get_connection() as conn:
            records = []
            for delta in deltas:
                records.append((
                    delta['market_ticker'],
                    delta['timestamp_ms'],
                    delta['sequence_number'],
                    delta['side'],
                    delta['action'],
                    delta['price'],
                    delta.get('old_size'),
                    delta.get('new_size')
                ))
            
            query = '''
                INSERT INTO rl_orderbook_deltas (
                    market_ticker, timestamp_ms, sequence_number,
                    side, action, price, old_size, new_size
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            '''
            
            await conn.executemany(query, records)
            return len(records)
    
    async def get_latest_snapshot(self, market_ticker: str) -> Optional[Dict[str, Any]]:
        """Get the latest orderbook snapshot for a market."""
        async with self.get_connection() as conn:
            row = await conn.fetchrow('''
                SELECT * FROM rl_orderbook_snapshots 
                WHERE market_ticker = $1 
                ORDER BY sequence_number DESC 
                LIMIT 1
            ''', market_ticker)
            
            if row:
                return dict(row)
            return None
    
    async def get_deltas_since_sequence(self, market_ticker: str, sequence_number: int) -> List[Dict[str, Any]]:
        """Get all deltas since a given sequence number."""
        async with self.get_connection() as conn:
            rows = await conn.fetch('''
                SELECT * FROM rl_orderbook_deltas 
                WHERE market_ticker = $1 AND sequence_number > $2 
                ORDER BY sequence_number ASC
            ''', market_ticker, sequence_number)
            
            return [dict(row) for row in rows]


# Global RL database instance
rl_db = RLDatabase()