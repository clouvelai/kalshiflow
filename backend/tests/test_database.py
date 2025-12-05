"""
Unit tests for the Database class.

This module provides comprehensive testing of the database layer, focusing on:
- Database connection and initialization
- Trade data insertion and retrieval
- Market metadata operations
- Data validation and constraints
- Aggregation functions
- Recovery operations
- Error handling and edge cases

Tests are designed to catch real bugs and data corruption issues.
"""

import pytest
import os
import asyncio
import asyncpg
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import patch, AsyncMock, MagicMock

from kalshiflow.database import Database
from kalshiflow.models import Trade


class MockAsyncContextManager:
    """Helper class for mocking async context managers."""
    
    def __init__(self, return_value):
        self.return_value = return_value
        
    async def __aenter__(self):
        return self.return_value
        
    async def __aexit__(self, *args):
        pass


class TestDatabaseInitialization:
    """Test database initialization, connection management, and configuration."""
    
    def test_database_init_default_config(self):
        """Test database initialization with default configuration."""
        db = Database()
        
        # Should not be initialized yet
        assert not db._initialized
        assert db._pool is None
        assert db.pool_size == 10
        
        # Database URL should come from environment
        expected_url = (
            os.getenv("DATABASE_URL_POOLED") or 
            os.getenv("DATABASE_URL")
        )
        assert db.database_url == expected_url
    
    def test_database_init_custom_config(self):
        """Test database initialization with custom configuration."""
        custom_url = "postgresql://user:pass@localhost:5432/testdb"
        db = Database(database_url=custom_url, pool_size=5)
        
        assert db.database_url == custom_url
        assert db.pool_size == 5
        assert not db._initialized
    
    @pytest.mark.asyncio
    async def test_database_init_no_url_raises_error(self):
        """Test that initialization fails without database URL."""
        with patch.dict(os.environ, {}, clear=True):
            db = Database(database_url=None)
            
            with pytest.raises(ValueError, match="DATABASE_URL environment variable is required"):
                await db.initialize()
    
    @pytest.mark.asyncio
    async def test_database_init_multiple_calls_idempotent(self):
        """Test that multiple initialization calls are safe (idempotent)."""
        db = Database(database_url="postgresql://test")
        
        with patch('kalshiflow.database.asyncpg.create_pool', new_callable=AsyncMock) as mock_pool:
            mock_pool_instance = AsyncMock()
            mock_pool.return_value = mock_pool_instance
            
            with patch.object(db, '_run_migrations', new_callable=AsyncMock):
                # First call
                await db.initialize()
                assert db._initialized
                assert mock_pool.call_count == 1
                
                # Second call should not create another pool
                await db.initialize()
                assert mock_pool.call_count == 1  # Still 1
    
    @pytest.mark.asyncio
    async def test_database_close(self):
        """Test database connection pool cleanup."""
        db = Database()
        
        # Mock pool
        mock_pool = AsyncMock()
        db._pool = mock_pool
        db._initialized = True
        
        await db.close()
        
        mock_pool.close.assert_called_once()
        assert db._pool is None
        assert not db._initialized


class TestTradeOperations:
    """Test trade data insertion, retrieval, and validation."""
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database with connection pool."""
        db = Database(database_url="postgresql://test")
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        # Mock acquire() to return the context manager directly
        mock_pool.acquire = MagicMock(return_value=MockAsyncContextManager(mock_conn))
        
        db._pool = mock_pool
        db._initialized = True
        
        return db, mock_conn
    
    @pytest.fixture
    def valid_trade_data(self):
        """Valid trade data for testing."""
        return {
            'ticker': 'PRESWIN25',
            'price': 65,
            'volume': 100,
            'side': 'yes',
            'timestamp': 1701648000000
        }
    
    @pytest.fixture
    def valid_trade_model(self):
        """Valid Trade model for testing."""
        return Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1701648000000
        )
    
    @pytest.mark.asyncio
    async def test_store_trade_success(self, mock_database, valid_trade_data):
        """Test successful trade storage."""
        db, mock_conn = mock_database
        mock_conn.fetchval.return_value = 123  # Mock row ID
        
        with patch('kalshiflow.database.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.timestamp.return_value = 1701648100.0
            mock_datetime.now.return_value = mock_now
            
            row_id = await db.store_trade(valid_trade_data)
        
        assert row_id == 123
        mock_conn.fetchval.assert_called_once()
        
        # Verify the SQL call
        call_args = mock_conn.fetchval.call_args
        sql = call_args[0][0]
        params = call_args[0][1:]
        
        assert "INSERT INTO trades" in sql
        assert len(params) == 9  # Should have 9 parameters
        assert params[0] == 'PRESWIN25'  # ticker
        assert params[5] == 100  # count
        assert params[6] == 'yes'  # taker_side
    
    @pytest.mark.asyncio
    async def test_insert_trade_model(self, mock_database, valid_trade_model):
        """Test inserting a Trade model."""
        db, mock_conn = mock_database
        mock_conn.fetchval.return_value = 456
        
        with patch('kalshiflow.database.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.timestamp.return_value = 1701648100.0
            mock_datetime.now.return_value = mock_now
            
            row_id = await db.insert_trade(valid_trade_model)
        
        assert row_id == 456
        
        call_args = mock_conn.fetchval.call_args
        params = call_args[0][1:]
        
        assert params[0] == "PRESWIN25"
        assert params[1] == 65  # yes_price
        assert params[2] == 35  # no_price
        assert params[3] == 0.65  # yes_price_dollars
        assert params[4] == 0.35  # no_price_dollars
    
    @pytest.mark.asyncio
    async def test_get_recent_trades(self, mock_database):
        """Test retrieving recent trades."""
        db, mock_conn = mock_database
        
        # Mock database rows with Decimal values
        mock_rows = [
            {
                'id': 1,
                'market_ticker': 'TEST1',
                'yes_price_dollars': Decimal('0.65'),
                'no_price_dollars': Decimal('0.35'),
                'count': 100,
                'ts': 1701648000000
            },
            {
                'id': 2,
                'market_ticker': 'TEST2',
                'yes_price_dollars': Decimal('0.45'),
                'no_price_dollars': Decimal('0.55'),
                'count': 50,
                'ts': 1701648060000
            }
        ]
        
        mock_conn.fetch.return_value = mock_rows
        
        trades = await db.get_recent_trades(limit=2)
        
        assert len(trades) == 2
        assert trades[0]['market_ticker'] == 'TEST1'
        assert trades[0]['yes_price_dollars'] == 0.65  # Converted from Decimal
        assert trades[1]['market_ticker'] == 'TEST2'
        assert trades[1]['no_price_dollars'] == 0.55  # Converted from Decimal
        
        # Verify SQL call
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        limit_param = call_args[0][1]
        
        assert "ORDER BY ts DESC" in sql
        assert "LIMIT" in sql
        assert limit_param == 2
    
    @pytest.mark.asyncio
    async def test_get_trades_for_ticker(self, mock_database):
        """Test retrieving trades for specific ticker."""
        db, mock_conn = mock_database
        
        mock_rows = [
            {'id': 1, 'market_ticker': 'PRESWIN25', 'count': 100},
            {'id': 2, 'market_ticker': 'PRESWIN25', 'count': 50}
        ]
        mock_conn.fetch.return_value = mock_rows
        
        trades = await db.get_trades_for_ticker('PRESWIN25', limit=10)
        
        assert len(trades) == 2
        assert all(trade['market_ticker'] == 'PRESWIN25' for trade in trades)
        
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        ticker_param = call_args[0][1]
        limit_param = call_args[0][2]
        
        assert "WHERE market_ticker = $1" in sql
        assert ticker_param == 'PRESWIN25'
        assert limit_param == 10


class TestMarketOperations:
    """Test market metadata operations and JSON handling."""
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        db = Database(database_url="postgresql://test")
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        mock_pool.acquire = MagicMock(return_value=MockAsyncContextManager(mock_conn))
        
        db._pool = mock_pool
        db._initialized = True
        
        return db, mock_conn
    
    @pytest.mark.asyncio
    async def test_insert_or_update_market_new(self, mock_database):
        """Test inserting new market metadata."""
        db, mock_conn = mock_database
        
        raw_market_data = {
            "id": "PRESWIN25",
            "title": "Presidential Win 2025",
            "status": "open"
        }
        
        result = await db.insert_or_update_market(
            ticker="PRESWIN25",
            title="Presidential Win 2025",
            category="Politics",
            liquidity_dollars=150000.50,
            open_interest=5000,
            latest_expiration_time="2025-01-20T12:00:00Z",
            raw_market_data=raw_market_data
        )
        
        assert result is True
        mock_conn.execute.assert_called_once()
        
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1:]
        
        assert "INSERT INTO markets" in sql
        assert "ON CONFLICT (ticker) DO UPDATE" in sql
        assert params[0] == "PRESWIN25"
        assert params[1] == "Presidential Win 2025"
        assert params[2] == "Politics"
        assert params[3] == 150000.50
        assert params[4] == 5000
        
        # Check that JSON data was serialized
        json_param = params[6]
        assert isinstance(json_param, str)
        import json
        parsed_json = json.loads(json_param)
        assert parsed_json["id"] == "PRESWIN25"
    
    @pytest.mark.asyncio
    async def test_insert_market_invalid_json(self, mock_database):
        """Test market insertion with invalid JSON data."""
        db, mock_conn = mock_database
        
        # Test with string that's not JSON
        result = await db.insert_or_update_market(
            ticker="TEST",
            title="Test Market",
            raw_market_data="not valid json"
        )
        
        assert result is True
        
        call_args = mock_conn.execute.call_args
        params = call_args[0][1:]
        json_param = params[6]
        
        # Should wrap invalid JSON in a "raw" field
        import json
        parsed_json = json.loads(json_param)
        assert parsed_json["raw"] == "not valid json"
    
    @pytest.mark.asyncio
    async def test_insert_market_invalid_date(self, mock_database):
        """Test market insertion with invalid expiration date."""
        db, mock_conn = mock_database
        
        # Invalid date format should not crash, just be None
        result = await db.insert_or_update_market(
            ticker="TEST",
            title="Test Market",
            latest_expiration_time="invalid date"
        )
        
        assert result is True
        
        call_args = mock_conn.execute.call_args
        params = call_args[0][1:]
        expiration_param = params[5]
        
        assert expiration_param is None
    
    @pytest.mark.asyncio
    async def test_get_market_metadata(self, mock_database):
        """Test retrieving market metadata."""
        db, mock_conn = mock_database
        
        mock_row = {
            'ticker': 'PRESWIN25',
            'title': 'Presidential Win 2025',
            'category': 'Politics',
            'liquidity_dollars': Decimal('150000.50'),
            'raw_market_data': '{"test": "data"}'
        }
        mock_conn.fetchrow.return_value = mock_row
        
        metadata = await db.get_market_metadata('PRESWIN25')
        
        assert metadata is not None
        assert metadata['ticker'] == 'PRESWIN25'
        assert metadata['title'] == 'Presidential Win 2025'
        
        call_args = mock_conn.fetchrow.call_args
        sql = call_args[0][0]
        ticker_param = call_args[0][1]
        
        assert "SELECT * FROM markets WHERE ticker = $1" in sql
        assert ticker_param == 'PRESWIN25'
    
    @pytest.mark.asyncio
    async def test_get_market_metadata_not_found(self, mock_database):
        """Test retrieving non-existent market metadata."""
        db, mock_conn = mock_database
        mock_conn.fetchrow.return_value = None
        
        metadata = await db.get_market_metadata('NONEXISTENT')
        
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_market_exists(self, mock_database):
        """Test checking if market exists."""
        db, mock_conn = mock_database
        
        # Test existing market
        mock_conn.fetchval.return_value = True
        exists = await db.market_exists('PRESWIN25')
        assert exists is True
        
        # Test non-existing market
        mock_conn.fetchval.return_value = False
        exists = await db.market_exists('NONEXISTENT')
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_get_markets_metadata_multiple(self, mock_database):
        """Test retrieving metadata for multiple markets."""
        db, mock_conn = mock_database
        
        mock_rows = [
            {'ticker': 'MARKET1', 'title': 'Market 1'},
            {'ticker': 'MARKET2', 'title': 'Market 2'}
        ]
        mock_conn.fetch.return_value = mock_rows
        
        tickers = ['MARKET1', 'MARKET2', 'NONEXISTENT']
        metadata = await db.get_markets_metadata(tickers)
        
        assert len(metadata) == 2
        assert 'MARKET1' in metadata
        assert 'MARKET2' in metadata
        assert 'NONEXISTENT' not in metadata
        
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        tickers_param = call_args[0][1]
        
        assert "WHERE ticker = ANY($1)" in sql
        assert tickers_param == tickers
    
    @pytest.mark.asyncio
    async def test_get_markets_metadata_empty_list(self, mock_database):
        """Test retrieving metadata with empty ticker list."""
        db, mock_conn = mock_database
        
        metadata = await db.get_markets_metadata([])
        
        assert metadata == {}
        mock_conn.fetch.assert_not_called()


class TestAggregationOperations:
    """Test aggregation functions and time-window queries."""
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        db = Database(database_url="postgresql://test")
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        mock_pool.acquire = MagicMock(return_value=MockAsyncContextManager(mock_conn))
        
        db._pool = mock_pool
        db._initialized = True
        
        return db, mock_conn
    
    @pytest.mark.asyncio
    async def test_get_trades_in_window(self, mock_database):
        """Test retrieving trades within time window."""
        db, mock_conn = mock_database
        
        mock_rows = [
            {'id': 1, 'ts': 1701648000000},
            {'id': 2, 'ts': 1701648300000}
        ]
        mock_conn.fetch.return_value = mock_rows
        
        with patch('kalshiflow.database.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.timestamp.return_value = 1701648600.0  # Current time
            mock_datetime.now.return_value = mock_now
            
            trades = await db.get_trades_in_window(window_minutes=10)
        
        assert len(trades) == 2
        
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        cutoff_param = call_args[0][1]
        
        assert "WHERE ts >= $1" in sql
        # Should be 10 minutes ago: 1701648600 - 600 = 1701648000
        expected_cutoff = int((1701648600.0 - 10 * 60) * 1000)
        assert cutoff_param == expected_cutoff
    
    @pytest.mark.asyncio
    async def test_get_ticker_stats(self, mock_database):
        """Test getting aggregated ticker statistics."""
        db, mock_conn = mock_database
        
        mock_stats = {
            'trade_count': 10,
            'total_volume': 1000,
            'yes_volume': 600,
            'no_volume': 400,
            'avg_yes_price': Decimal('0.65'),
            'avg_no_price': Decimal('0.35'),
            'last_trade_ts': 1701648000000
        }
        mock_conn.fetchrow.return_value = mock_stats
        
        with patch('kalshiflow.database.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.timestamp.return_value = 1701648600.0
            mock_datetime.now.return_value = mock_now
            
            stats = await db.get_ticker_stats('PRESWIN25', window_minutes=10)
        
        assert stats['trade_count'] == 10
        assert stats['total_volume'] == 1000
        assert stats['yes_volume'] == 600
        assert stats['no_volume'] == 400
        assert stats['avg_yes_price'] == 0.65  # Converted from Decimal
        assert stats['avg_no_price'] == 0.35
        
        call_args = mock_conn.fetchrow.call_args
        sql = call_args[0][0]
        ticker_param = call_args[0][1]
        cutoff_param = call_args[0][2]
        
        assert "WHERE market_ticker = $1 AND ts >= $2" in sql
        assert ticker_param == 'PRESWIN25'
        assert "SUM(CASE WHEN taker_side = 'yes'" in sql
        assert "SUM(CASE WHEN taker_side = 'no'" in sql
    
    @pytest.mark.asyncio
    async def test_get_ticker_stats_no_data(self, mock_database):
        """Test ticker stats when no data available."""
        db, mock_conn = mock_database
        mock_conn.fetchrow.return_value = None
        
        stats = await db.get_ticker_stats('NONEXISTENT')
        
        assert stats == {}
    
    @pytest.mark.asyncio
    async def test_get_db_stats(self, mock_database):
        """Test getting database statistics."""
        db, mock_conn = mock_database
        
        mock_stats = {
            'total_trades': 5000,
            'oldest_trade': 1701648000000,
            'newest_trade': 1701748000000,
            'unique_tickers': 25
        }
        mock_conn.fetchrow.return_value = mock_stats
        
        stats = await db.get_db_stats()
        
        assert stats['total_trades'] == 5000
        assert stats['oldest_trade'] == 1701648000000
        assert stats['newest_trade'] == 1701748000000
        assert stats['unique_tickers'] == 25
        assert stats['database_type'] == 'PostgreSQL'
        assert 'pool_size' in stats
        
        call_args = mock_conn.fetchrow.call_args
        sql = call_args[0][0]
        
        assert "COUNT(*) as total_trades" in sql
        assert "MIN(ts) as oldest_trade" in sql
        assert "MAX(ts) as newest_trade" in sql
        assert "COUNT(DISTINCT market_ticker)" in sql
    
    @pytest.mark.asyncio
    async def test_cleanup_old_trades(self, mock_database):
        """Test cleaning up old trade data."""
        db, mock_conn = mock_database
        mock_conn.execute.return_value = "DELETE 123"
        
        with patch('kalshiflow.database.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.timestamp.return_value = 1701648000.0
            mock_datetime.now.return_value = mock_now
            
            deleted_count = await db.cleanup_old_trades(days_to_keep=7)
        
        assert deleted_count == 123
        
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        cutoff_param = call_args[0][1]
        
        assert "DELETE FROM trades WHERE ts < $1" in sql
        # Should be 7 days ago in milliseconds
        expected_cutoff = int((1701648000.0 - 7 * 24 * 60 * 60) * 1000)
        assert cutoff_param == expected_cutoff


class TestRecoveryOperations:
    """Test recovery methods and data corruption handling."""
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        db = Database(database_url="postgresql://test")
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        mock_pool.acquire = MagicMock(return_value=MockAsyncContextManager(mock_conn))
        
        db._pool = mock_pool
        db._initialized = True
        
        return db, mock_conn
    
    @pytest.mark.asyncio
    async def test_get_trades_for_recovery(self, mock_database):
        """Test retrieving trades for recovery operations."""
        db, mock_conn = mock_database
        
        mock_trades = [
            {'id': 1, 'ts': 1701648000000, 'market_ticker': 'TEST1'},
            {'id': 2, 'ts': 1701649000000, 'market_ticker': 'TEST2'}
        ]
        mock_conn.fetch.return_value = mock_trades
        
        # Test without mocking time - just verify the method works and check SQL structure
        trades = await db.get_trades_for_recovery(hours=24)
        
        assert len(trades) == 2
        
        call_args = mock_conn.fetch.call_args
        sql = call_args[0][0]
        cutoff_param = call_args[0][1]
        min_valid_param = call_args[0][2]
        
        assert "WHERE ts >= $1 AND ts >= $2" in sql
        assert "ORDER BY ts ASC" in sql  # Recovery should be chronological
        
        # Verify parameters are reasonable timestamps
        assert isinstance(cutoff_param, int)
        assert cutoff_param > 0  # Should be a valid timestamp
        assert isinstance(min_valid_param, int)
        assert min_valid_param == int(datetime(2020, 1, 1).timestamp() * 1000)  # Corruption filter
    
    @pytest.mark.asyncio
    async def test_get_trades_for_minute_recovery(self, mock_database):
        """Test minute-level recovery operations."""
        db, mock_conn = mock_database
        
        mock_trades = [
            {'id': 1, 'ts': 1701648000000},
            {'id': 2, 'ts': 1701648060000}
        ]
        mock_conn.fetch.return_value = mock_trades
        
        with patch('kalshiflow.database.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.timestamp.return_value = 1701648600.0
            mock_datetime.now.return_value = mock_now
            
            trades = await db.get_trades_for_minute_recovery(minutes=10)
        
        assert len(trades) == 2
        
        call_args = mock_conn.fetch.call_args
        cutoff_param = call_args[0][1]
        
        # Should be 10 minutes ago in milliseconds
        expected_cutoff = int((1701648600.0 - 10 * 60) * 1000)
        assert cutoff_param == expected_cutoff
    
    @pytest.mark.asyncio
    async def test_get_recovery_trade_count(self, mock_database):
        """Test counting trades available for recovery."""
        db, mock_conn = mock_database
        mock_conn.fetchval.return_value = 500
        
        with patch('kalshiflow.database.datetime') as mock_datetime:
            mock_now = MagicMock()
            mock_now.timestamp.return_value = 1701648000.0
            mock_datetime.now.return_value = mock_now
            
            count = await db.get_recovery_trade_count(hours=12)
        
        assert count == 500
        
        call_args = mock_conn.fetchval.call_args
        sql = call_args[0][0]
        cutoff_param = call_args[0][1]
        min_valid_param = call_args[0][2]
        
        assert "SELECT COUNT(*)" in sql
        assert "WHERE ts >= $1 AND ts >= $2" in sql
        
        expected_cutoff = int((1701648000.0 - 12 * 3600) * 1000)
        assert cutoff_param == expected_cutoff
    
    @pytest.mark.asyncio
    async def test_get_recovery_trade_count_no_data(self, mock_database):
        """Test recovery count when no data available."""
        db, mock_conn = mock_database
        mock_conn.fetchval.return_value = None
        
        count = await db.get_recovery_trade_count(hours=1)
        
        assert count == 0


class TestConstraintValidation:
    """Test data validation and constraint handling."""
    
    @pytest.fixture
    def mock_database(self):
        """Create a mock database for testing."""
        db = Database(database_url="postgresql://test")
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        mock_pool.acquire = MagicMock(return_value=MockAsyncContextManager(mock_conn))
        
        db._pool = mock_pool
        db._initialized = True
        
        return db, mock_conn
    
    def test_decimal_conversion(self):
        """Test Decimal to float conversion for JSON serialization."""
        db = Database()
        
        data = {
            'price': Decimal('0.65'),
            'volume': 100,
            'nested': {
                'amount': Decimal('1500.25')
            },
            'text': 'test'
        }
        
        # Note: _convert_decimals_to_float only converts top-level Decimal values
        converted = db._convert_decimals_to_float(data.copy())
        
        assert converted['price'] == 0.65
        assert converted['volume'] == 100
        assert isinstance(converted['nested']['amount'], Decimal)  # Nested not converted
        assert converted['text'] == 'test'
    
    @pytest.mark.asyncio
    async def test_constraint_violation_handling(self, mock_database):
        """Test handling of database constraint violations."""
        db, mock_conn = mock_database
        
        # Simulate constraint violation
        constraint_error = asyncpg.exceptions.CheckViolationError(
            "new row for relation \"trades\" violates check constraint \"chk_count_positive\""
        )
        mock_conn.fetchval.side_effect = constraint_error
        
        invalid_trade = {
            'ticker': 'TEST',
            'price': 50,
            'volume': -10,  # Invalid: negative volume
            'side': 'yes',
            'timestamp': 1701648000000
        }
        
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await db.store_trade(invalid_trade)
    
    @pytest.mark.asyncio
    async def test_foreign_key_constraint(self, mock_database):
        """Test foreign key constraint handling."""
        db, mock_conn = mock_database
        
        # Simulate foreign key violation
        fk_error = asyncpg.exceptions.ForeignKeyViolationError(
            "insert or update on table \"trades\" violates foreign key constraint"
        )
        mock_conn.fetchval.side_effect = fk_error
        
        trade_with_invalid_ticker = {
            'ticker': 'NONEXISTENT_MARKET',
            'price': 50,
            'volume': 100,
            'side': 'yes',
            'timestamp': 1701648000000
        }
        
        with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
            await db.store_trade(trade_with_invalid_ticker)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test handling of database connection failures."""
        db = Database(database_url="postgresql://invalid:5432/nonexistent")
        
        with patch('kalshiflow.database.asyncpg.create_pool', new_callable=AsyncMock) as mock_pool:
            mock_pool.side_effect = asyncpg.exceptions.ConnectionDoesNotExistError(
                "connection does not exist"
            )
            
            with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
                await db.initialize()
    
    @pytest.mark.asyncio
    async def test_query_timeout(self):
        """Test handling of query timeouts."""
        db = Database(database_url="postgresql://test")
        mock_pool = AsyncMock()
        mock_conn = AsyncMock()
        
        # Simulate timeout
        mock_conn.fetch.side_effect = asyncio.TimeoutError("Query timeout")
        
        mock_pool.acquire = MagicMock(return_value=MockAsyncContextManager(mock_conn))
        
        db._pool = mock_pool
        db._initialized = True
        
        with pytest.raises(asyncio.TimeoutError):
            await db.get_recent_trades(limit=100)
    
    @pytest.mark.asyncio
    async def test_connection_context_manager_error(self):
        """Test error handling in connection context manager."""
        db = Database(database_url="postgresql://test")
        
        # Database not initialized
        db._initialized = False
        
        with patch.object(db, 'initialize', side_effect=Exception("Init failed")):
            with pytest.raises(Exception, match="Init failed"):
                async with db.get_connection():
                    pass


class TestMigrations:
    """Test database migration functionality."""
    
    @pytest.mark.asyncio
    async def test_run_migrations_skip_local_supabase(self):
        """Test that migrations are skipped for local Supabase."""
        db = Database(database_url="postgresql://test")
        mock_pool = AsyncMock()
        db._pool = mock_pool
        
        with patch.dict(os.environ, {'SUPABASE_URL': 'http://localhost:54321'}):
            with patch('kalshiflow.database.asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
                mock_create_pool.return_value = mock_pool
                with patch.object(db, '_run_migrations', new_callable=AsyncMock) as mock_migrations:
                    await db.initialize()
                    mock_migrations.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_run_migrations_production(self):
        """Test that migrations run for production environment."""
        db = Database(database_url="postgresql://test")
        mock_pool = AsyncMock()
        db._pool = mock_pool
        
        with patch.dict(os.environ, {'SUPABASE_URL': 'https://prod.supabase.co'}):
            with patch('kalshiflow.database.asyncpg.create_pool', new_callable=AsyncMock) as mock_create_pool:
                mock_create_pool.return_value = mock_pool
                with patch.object(db, '_run_migrations', new_callable=AsyncMock) as mock_migrations:
                    await db.initialize()
                    mock_migrations.assert_called_once()


if __name__ == "__main__":
    # Run specific test classes for debugging
    import sys
    
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main([f"-v", f"tests/test_database.py::{test_class}"])
    else:
        pytest.main(["-v", "tests/test_database.py"])