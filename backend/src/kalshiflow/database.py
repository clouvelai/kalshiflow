"""
SQLite database setup and operations for storing Kalshi trade data.
"""

import os
import sqlite3
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any
from .models import Trade
import aiosqlite


class Database:
    """SQLite database manager for trade data storage."""
    
    def __init__(self, db_path: str = None):
        """Initialize database with given path."""
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "./kalshi_trades.db")
        self._init_lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize database schema if not exists."""
        async with self._init_lock:
            if self._initialized:
                return
                
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        market_ticker TEXT NOT NULL,
                        yes_price INTEGER NOT NULL,
                        no_price INTEGER NOT NULL,
                        yes_price_dollars REAL NOT NULL,
                        no_price_dollars REAL NOT NULL,
                        count INTEGER NOT NULL,
                        taker_side TEXT NOT NULL,
                        ts INTEGER NOT NULL,
                        received_at INTEGER NOT NULL,
                        FOREIGN KEY(market_ticker) REFERENCES markets(ticker) ON DELETE CASCADE
                    )
                ''')
                
                # Create indexes for performance
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_ts 
                    ON trades(ts)
                ''')
                
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_ticker_ts 
                    ON trades(market_ticker, ts)
                ''')
                
                await db.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_received_at 
                    ON trades(received_at)
                ''')
                
                await db.commit()
                
            self._initialized = True
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection with proper initialization."""
        if not self._initialized:
            await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            yield db
    
    async def insert_trade(self, trade: Trade) -> int:
        """Insert a trade record and return the row ID."""
        received_at = int(datetime.now().timestamp() * 1000)
        
        async with self.get_connection() as db:
            cursor = await db.execute('''
                INSERT INTO trades (
                    market_ticker, yes_price, no_price, yes_price_dollars, 
                    no_price_dollars, count, taker_side, ts, received_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.market_ticker,
                trade.yes_price,
                trade.no_price,
                trade.yes_price_dollars,
                trade.no_price_dollars,
                trade.count,
                trade.taker_side,
                trade.ts,
                received_at
            ))
            await db.commit()
            return cursor.lastrowid
    
    async def get_recent_trades(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Get recent trades ordered by timestamp descending."""
        async with self.get_connection() as db:
            async with db.execute('''
                SELECT * FROM trades 
                ORDER BY ts DESC 
                LIMIT ?
            ''', (limit,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_trades_for_ticker(self, ticker: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades for a specific ticker."""
        async with self.get_connection() as db:
            async with db.execute('''
                SELECT * FROM trades 
                WHERE market_ticker = ?
                ORDER BY ts DESC 
                LIMIT ?
            ''', (ticker, limit)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_trades_in_window(self, window_minutes: int = 10) -> List[Dict[str, Any]]:
        """Get trades within a time window (minutes ago to now)."""
        cutoff_ts = int((datetime.now().timestamp() - window_minutes * 60) * 1000)
        
        async with self.get_connection() as db:
            async with db.execute('''
                SELECT * FROM trades 
                WHERE ts >= ?
                ORDER BY ts DESC
            ''', (cutoff_ts,)) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
    
    async def get_ticker_stats(self, ticker: str, window_minutes: int = 10) -> Dict[str, Any]:
        """Get aggregated statistics for a ticker within a time window."""
        cutoff_ts = int((datetime.now().timestamp() - window_minutes * 60) * 1000)
        
        async with self.get_connection() as db:
            async with db.execute('''
                SELECT 
                    COUNT(*) as trade_count,
                    SUM(count) as total_volume,
                    SUM(CASE WHEN taker_side = 'yes' THEN count ELSE 0 END) as yes_volume,
                    SUM(CASE WHEN taker_side = 'no' THEN count ELSE 0 END) as no_volume,
                    AVG(yes_price_dollars) as avg_yes_price,
                    AVG(no_price_dollars) as avg_no_price,
                    MAX(ts) as last_trade_ts
                FROM trades 
                WHERE market_ticker = ? AND ts >= ?
            ''', (ticker, cutoff_ts)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else {}
    
    async def cleanup_old_trades(self, days_to_keep: int = 30):
        """Remove trades older than specified days."""
        cutoff_ts = int((datetime.now().timestamp() - days_to_keep * 24 * 60 * 60) * 1000)
        
        async with self.get_connection() as db:
            cursor = await db.execute('''
                DELETE FROM trades WHERE ts < ?
            ''', (cutoff_ts,))
            await db.commit()
            return cursor.rowcount
    
    async def get_db_stats(self) -> Dict[str, Any]:
        """Get database statistics for debugging."""
        async with self.get_connection() as db:
            # Get total trade count
            async with db.execute('SELECT COUNT(*) as total_trades FROM trades') as cursor:
                total_row = await cursor.fetchone()
            
            # Get oldest and newest trades
            async with db.execute('''
                SELECT MIN(ts) as oldest_trade, MAX(ts) as newest_trade FROM trades
            ''') as cursor:
                range_row = await cursor.fetchone()
            
            # Get unique tickers count
            async with db.execute('SELECT COUNT(DISTINCT market_ticker) as unique_tickers FROM trades') as cursor:
                ticker_row = await cursor.fetchone()
            
            return {
                "total_trades": total_row["total_trades"] if total_row else 0,
                "oldest_trade": range_row["oldest_trade"] if range_row else None,
                "newest_trade": range_row["newest_trade"] if range_row else None,
                "unique_tickers": ticker_row["unique_tickers"] if ticker_row else 0,
                "db_path": self.db_path
            }


# Global database instance
_db_instance = None

def get_database() -> Database:
    """Get the global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance