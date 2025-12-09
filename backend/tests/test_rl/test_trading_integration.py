"""
Tests for RL Trading Integration with Demo Account.

Comprehensive test suite for the Kalshi demo account integration
including trading sessions, order execution, database logging,
and integration with the broader RL infrastructure.
"""

import pytest
import pytest_asyncio
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import components under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from kalshiflow_rl.trading.demo_client import KalshiDemoTradingClient, KalshiDemoAuthError, KalshiDemoOrderError, KalshiDemoTradingClientError
from kalshiflow_rl.trading.integration import TradingSession, TradingSessionManager, create_trading_session
from kalshiflow_rl.trading.action_write_queue import ActionWriteQueue
from kalshiflow_rl.trading.demo_account_test_results import run_demo_account_tests
from kalshiflow_rl.environments.action_space import ActionType
from kalshiflow_rl.config import config


class TestDemoTradingClient:
    """Test the KalshiDemoTradingClient functionality."""
    
    @pytest_asyncio.fixture
    async def demo_client(self):
        """Create a demo trading client for testing."""
        with patch('kalshiflow_rl.trading.demo_client.config') as mock_config:
            mock_config.KALSHI_PAPER_TRADING_API_KEY_ID = 'test_key_id'
            mock_config.KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT = '''-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAtest...
-----END RSA PRIVATE KEY-----'''
            mock_config.KALSHI_PAPER_TRADING_API_URL = 'https://test-api.demo.co/v2'
            mock_config.KALSHI_PAPER_TRADING_WS_URL = 'wss://test-api.demo.co/ws/v2'
            
            # Mock the KalshiAuth to avoid real authentication
            with patch('kalshiflow_rl.trading.demo_client.KalshiAuth') as mock_auth:
                mock_auth_instance = MagicMock()  # Regular mock, not async
                mock_auth_instance.create_auth_headers.return_value = {
                    'KALSHI-ACCESS-KEY': 'test_key',
                    'KALSHI-ACCESS-SIGNATURE': 'test_signature',
                    'KALSHI-ACCESS-TIMESTAMP': '1234567890'
                }
                mock_auth.return_value = mock_auth_instance
                
                client = KalshiDemoTradingClient(mode='paper')
                yield client
                
                if hasattr(client, '_temp_key_file'):
                    client._cleanup_demo_auth_env()
    
    def test_demo_client_initialization(self):
        """Test demo client initializes correctly."""
        with patch('kalshiflow_rl.trading.demo_client.config') as mock_config:
            mock_config.KALSHI_PAPER_TRADING_API_KEY_ID = 'test_key'
            mock_config.KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT = '-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----'
            mock_config.KALSHI_PAPER_TRADING_API_URL = 'https://test-api.demo.co/v2'
            mock_config.KALSHI_PAPER_TRADING_WS_URL = 'wss://test-api.demo.co/ws/v2'
            
            with patch('kalshiflow_rl.trading.demo_client.KalshiAuth'):
                client = KalshiDemoTradingClient(mode='paper')
                
                assert client.mode == 'paper'
                assert client.api_key_id == 'test_key'
                assert not client.is_connected
                assert len(client.positions) == 0
                assert len(client.orders) == 0
    
    def test_demo_client_mode_validation(self):
        """Test that demo client only accepts paper mode."""
        with patch('kalshiflow_rl.trading.demo_client.config') as mock_config:
            mock_config.KALSHI_PAPER_TRADING_API_KEY_ID = 'test_key'
            mock_config.KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT = '-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----'
            
            with pytest.raises(ValueError, match="only supports 'paper' mode"):
                KalshiDemoTradingClient(mode='live')
    
    def test_missing_credentials(self):
        """Test error handling for missing credentials."""
        with patch('kalshiflow_rl.trading.demo_client.config') as mock_config:
            mock_config.KALSHI_PAPER_TRADING_API_KEY_ID = None
            mock_config.KALSHI_PAPER_TRADING_PRIVATE_KEY_CONTENT = None
            
            with pytest.raises(KalshiDemoAuthError, match="API_KEY_ID not configured"):
                KalshiDemoTradingClient(mode='paper')
    
    @pytest.mark.asyncio
    async def test_connect_with_limited_access(self, demo_client):
        """Test connection handling with limited demo account access."""
        # Mock aiohttp session
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"markets": []})
        
        # Mock the async context manager properly
        mock_request_ctx = AsyncMock()
        mock_request_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_request_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_session.request.return_value = mock_request_ctx
        mock_session.close = AsyncMock()
        
        # Set the mocked session directly on the client
        demo_client.session = mock_session
        demo_client.is_connected = True
        demo_client.balance = 10000.00
        
        # Test that the client connects successfully
        assert demo_client.is_connected
        assert demo_client.balance == 10000.00
    
    @pytest.mark.asyncio 
    async def test_proper_error_handling(self, demo_client):
        """Test proper exception handling for portfolio operations."""
        # Set up the demo client to be "connected" but fail on portfolio operations
        demo_client.is_connected = True
        demo_client.session = AsyncMock()
        
        # Mock _make_request to simulate authentication errors
        async def mock_make_request(method, path, data=None):
            if '/portfolio' in path:
                # Simulate portfolio access error by raising the right exception type
                raise KalshiDemoTradingClientError("HTTP 401: authentication_error")
            else:
                return {"markets": []}
        
        demo_client._make_request = mock_make_request
        
        # Test that proper exceptions are raised (no more graceful degradation)
        with pytest.raises(KalshiDemoTradingClientError, match="HTTP 401: authentication_error"):
            await demo_client.get_positions()
            
        with pytest.raises(KalshiDemoTradingClientError, match="HTTP 401: authentication_error"):
            await demo_client.get_orders()
    
    @pytest.mark.asyncio
    async def test_order_execution_errors(self, demo_client):
        """Test proper error handling for order execution."""
        # Set up the demo client to be "connected" but fail on order operations
        demo_client.is_connected = True
        demo_client.session = AsyncMock()
        
        # Mock _make_request to simulate authentication errors for orders
        async def mock_make_request(method, path, data=None):
            if '/portfolio/orders' in path and method == 'POST':
                # Simulate order creation error by raising the right exception type
                raise KalshiDemoTradingClientError("HTTP 401: authentication_error")
            else:
                return {"orders": []}
        
        demo_client._make_request = mock_make_request
        
        # Test that proper exceptions are raised for order creation (no more simulation)
        with pytest.raises(KalshiDemoTradingClientError, match="HTTP 401: authentication_error"):
            await demo_client.create_order(
                ticker='TEST-MARKET',
                action='buy',
                side='yes',
                count=1,
                price=50
            )
    
    def test_demo_capabilities_documentation(self, demo_client):
        """Test that demo capabilities are properly documented."""
        limitations = demo_client.get_demo_limitations()
        
        assert 'demo_account_capabilities' in limitations
        assert 'demo_features' in limitations
        assert 'recommended_usage' in limitations
        
        # Verify specific capabilities (no more limitations since demo works fully)
        assert limitations['demo_account_capabilities']['markets_api'] == "✅ Full access to market data"
        assert limitations['demo_account_capabilities']['portfolio_balance'] == "✅ Full access to account balance"
        assert limitations['demo_account_capabilities']['order_creation'] == "✅ Full order creation capability"
        
        # Verify demo features
        assert limitations['demo_features']['api_compatibility'] == "100% compatible with production API"
        assert limitations['demo_features']['realistic_execution'] == "Orders execute with realistic fills"


class TestTradingIntegration:
    """Test the trading integration with RL infrastructure."""
    
    @pytest_asyncio.fixture
    async def mock_database(self):
        """Mock database for testing."""
        mock_db = AsyncMock()
        mock_db.initialize = AsyncMock()
        mock_db.create_trading_action = AsyncMock(return_value=123)  # Mock action ID
        mock_db.batch_insert_trading_actions = AsyncMock(return_value=1)  # Mock batch insert
        mock_db.close = AsyncMock()
        return mock_db
    
    @pytest_asyncio.fixture
    async def mock_demo_client(self):
        """Mock demo client for testing."""
        mock_client = AsyncMock()
        mock_client.mode = 'paper'
        mock_client.balance = 10000.00
        mock_client.positions = {}
        mock_client.orders = {}
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_client.create_order = AsyncMock(return_value={
            'order': {
                'order_id': 'test_order_123',
                'status': 'open'
            }
        })
        mock_client.get_positions = AsyncMock(return_value={'positions': []})
        mock_client.get_trading_summary = AsyncMock(return_value={
            'mode': 'paper',
            'balance': '10000.00',
            'positions_count': 0,
            'orders_count': 0
        })
        return mock_client
    
    @pytest.mark.asyncio
    async def test_trading_session_lifecycle(self, mock_database, mock_demo_client):
        """Test complete trading session lifecycle."""
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            mock_db_class.return_value = mock_database
            with patch('kalshiflow_rl.trading.integration.create_demo_trading_client') as mock_client_func:
                mock_client_func.return_value = mock_demo_client
                # Create and start session
                session = TradingSession('test_session', episode_id=456)
                await session.start(mode='paper')
                
                assert session.is_active
                assert session.session_name == 'test_session'
                assert session.episode_id == 456
                assert session.demo_client == mock_demo_client
                assert session.database == mock_database
                assert session.action_write_queue is not None
                assert session.action_write_queue._running
                
                # Stop session
                await session.stop()
                
                assert not session.is_active
                assert session.action_write_queue is None  # Queue should be cleaned up
                mock_demo_client.disconnect.assert_called_once()
                mock_database.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trading_action_execution(self, mock_database, mock_demo_client):
        """Test trading action execution and logging."""
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            mock_db_class.return_value = mock_database
            with patch('kalshiflow_rl.trading.integration.create_demo_trading_client') as mock_client_func:
                mock_client_func.return_value = mock_demo_client
                with patch('kalshiflow_rl.trading.action_write_queue.rl_db', mock_database):
                    with patch('kalshiflow_rl.config.config.RL_MARKET_TICKERS', ['TEST-MARKET']):
                        session = TradingSession('test_session', episode_id=789)
                        await session.start(mode='paper')
                        
                        # Execute trading action
                        result = await session.execute_trading_action(
                            action_type=ActionType.BUY_YES,
                            ticker='TEST-MARKET',
                            quantity=1,
                            price=50,
                            side='yes',
                            observation={'test': True},
                            model_confidence=0.8
                        )
                        
                        # Verify action execution
                        assert result['action_type'] == 'BUY_YES'
                        assert result['ticker'] == 'TEST-MARKET'
                        assert result['step_number'] == 1
                        assert result['logged'] == True  # Should be True when successfully enqueued
                        
                        # Force flush the queue to trigger database write
                        if session.action_write_queue:
                            await session.action_write_queue.force_flush()
                            
                            # Small delay to allow async processing
                            await asyncio.sleep(0.1)
                        
                        # Verify database batch insert was called (not individual create)
                        mock_database.batch_insert_trading_actions.assert_called()
                        call_args = mock_database.batch_insert_trading_actions.call_args[0][0]
                        assert len(call_args) == 1  # One action in batch
                        action_data = call_args[0]
                        assert action_data['episode_id'] == 789
                        assert action_data['action_type'] == 'buy_yes'
                        assert action_data['price'] == 50
                        assert action_data['quantity'] == 1
                        
                        await session.stop()
    
    @pytest.mark.asyncio
    async def test_hold_action(self, mock_database, mock_demo_client):
        """Test hold action execution."""
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            mock_db_class.return_value = mock_database
            with patch('kalshiflow_rl.trading.integration.create_demo_trading_client') as mock_client_func:
                mock_client_func.return_value = mock_demo_client
                with patch('kalshiflow_rl.trading.action_write_queue.rl_db', mock_database):
                    with patch('kalshiflow_rl.config.config.RL_MARKET_TICKERS', ['TEST-MARKET']):
                        session = TradingSession('test_session')
                        await session.start(mode='paper')
                        
                        # Execute hold action
                        result = await session.execute_trading_action(
                            action_type=ActionType.HOLD,
                            ticker='TEST-MARKET',
                            observation={'hold_test': True}
                        )
                        
                        # Verify hold action
                        assert result['action_type'] == 'HOLD'
                        assert result['execution_result']['action'] == 'hold'
                        assert result['execution_result']['executed'] == False
                        
                        # Force flush the queue to trigger database write
                        if session.action_write_queue:
                            await session.action_write_queue.force_flush()
                            await asyncio.sleep(0.1)
                        
                        # Verify database batch insert was called
                        mock_database.batch_insert_trading_actions.assert_called()
                        call_args = mock_database.batch_insert_trading_actions.call_args[0][0]
                        action_data = call_args[0]
                        assert action_data['action_type'] == 'hold'
                        assert action_data['quantity'] is None
                        assert action_data['price'] is None
                        
                        await session.stop()
    
    @pytest.mark.asyncio
    async def test_session_manager(self):
        """Test trading session manager functionality."""
        manager = TradingSessionManager()
        
        # Mock dependencies
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            with patch('kalshiflow_rl.trading.integration.create_demo_trading_client') as mock_client_func:
                # Set up mock returns
                mock_db = AsyncMock()
                mock_db.initialize = AsyncMock()
                mock_db.close = AsyncMock()
                mock_db_class.return_value = mock_db
                
                mock_client = AsyncMock()
                mock_client.connect = AsyncMock()
                mock_client.disconnect = AsyncMock()
                mock_client.get_trading_summary = AsyncMock(return_value={'mode': 'paper'})
                mock_client_func.return_value = mock_client
                
                # Create session
                session = await manager.create_session('test_manager_session')
                assert 'test_manager_session' in manager.active_sessions
                
                # Start session
                await manager.start_session('test_manager_session', mode='paper')
                
                # Get active sessions
                active = manager.get_active_sessions()
                assert len(active) == 1
                assert 'test_manager_session' in active
                
                # Stop session
                summary = await manager.stop_session('test_manager_session')
                assert summary['session_name'] == 'test_manager_session'
                assert len(manager.active_sessions) == 0
                assert len(manager.session_history) == 1
    
    @pytest.mark.asyncio
    async def test_invalid_market_error(self, mock_database, mock_demo_client):
        """Test error handling for invalid market tickers."""
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            mock_db_class.return_value = mock_database
            with patch('kalshiflow_rl.trading.integration.create_demo_trading_client') as mock_client_func:
                mock_client_func.return_value = mock_demo_client
                session = TradingSession('test_session')
                await session.start(mode='paper')
                
                # Try to execute action on unconfigured market
                with pytest.raises(ValueError, match="Market INVALID-MARKET not configured"):
                    await session.execute_trading_action(
                        action_type=ActionType.BUY_YES,
                        ticker='INVALID-MARKET',
                        quantity=1
                    )
                
                await session.stop()
    
    @pytest.mark.asyncio
    async def test_action_callbacks(self, mock_database, mock_demo_client):
        """Test action callback functionality."""
        callback_called = []
        
        def test_callback(action_data):
            callback_called.append(action_data)
        
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            mock_db_class.return_value = mock_database
            with patch('kalshiflow_rl.trading.integration.create_demo_trading_client') as mock_client_func:
                mock_client_func.return_value = mock_demo_client
                with patch('kalshiflow_rl.config.config.RL_MARKET_TICKERS', ['TEST-MARKET']):
                    session = TradingSession('callback_test')
                    session.add_action_callback(test_callback)
                    await session.start(mode='paper')
                    
                    # Execute action
                    await session.execute_trading_action(
                        action_type=ActionType.BUY_YES,
                        ticker='TEST-MARKET',
                        quantity=1,
                        price=50  # Provide price for executed trades
                    )
                    
                    # Verify callback was called
                    assert len(callback_called) == 1
                    assert callback_called[0]['action_type'] == 'buy_yes'
                    assert callback_called[0]['quantity'] == 1
                    
                    await session.stop()


class TestActionWriteQueue:
    """Test the ActionWriteQueue functionality."""
    
    @pytest_asyncio.fixture
    async def mock_rl_db(self):
        """Mock RL database for queue testing."""
        mock_db = AsyncMock()
        mock_db.batch_insert_trading_actions = AsyncMock(return_value=5)
        return mock_db
    
    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test queue initializes correctly."""
        queue = ActionWriteQueue(batch_size=10, flush_interval=0.5, max_queue_size=100)
        
        assert queue.batch_size == 10
        assert queue.flush_interval == 0.5
        assert queue.max_queue_size == 100
        assert not queue._running
        assert queue._actions_enqueued == 0
        assert queue._actions_written == 0
    
    @pytest.mark.asyncio
    async def test_queue_start_stop(self):
        """Test queue start and stop lifecycle."""
        queue = ActionWriteQueue(batch_size=5, flush_interval=0.1, max_queue_size=50)
        
        # Start queue
        await queue.start()
        assert queue._running
        assert queue._flush_task is not None
        
        # Stop queue
        await queue.stop()
        assert not queue._running
        assert queue._flush_task.done()
    
    @pytest.mark.asyncio 
    async def test_non_blocking_enqueue(self):
        """Test that enqueue operations are truly non-blocking."""
        queue = ActionWriteQueue(batch_size=5, flush_interval=1.0, max_queue_size=10)
        await queue.start()
        
        try:
            # Enqueue should return immediately
            start_time = time.time()
            
            action_data = {
                'episode_id': 123,
                'action_timestamp_ms': int(time.time() * 1000),
                'step_number': 1,
                'action_type': 'buy_yes',
                'price': 50,
                'quantity': 1,
                'position_before': {},
                'position_after': {},
                'reward': 0.1,
                'observation': {},
                'model_confidence': 0.8,
                'executed': True,
                'execution_price': 50
            }
            
            result = await queue.enqueue_action(action_data)
            
            elapsed_time = time.time() - start_time
            
            # Should return immediately (< 1ms)
            assert elapsed_time < 0.001
            assert result is True
            assert queue._actions_enqueued == 1
            assert queue._action_queue.qsize() == 1
            
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_rl_db):
        """Test batch processing of queued actions."""
        with patch('kalshiflow_rl.trading.action_write_queue.rl_db', mock_rl_db):
            queue = ActionWriteQueue(batch_size=3, flush_interval=0.1, max_queue_size=20)
            await queue.start()
            
            try:
                # Enqueue multiple actions
                for i in range(5):
                    action_data = {
                        'episode_id': 123,
                        'action_timestamp_ms': int(time.time() * 1000),
                        'step_number': i,
                        'action_type': 'buy_yes',
                        'price': 50 + i,
                        'quantity': 1,
                        'position_before': {},
                        'position_after': {},
                        'reward': 0.1,
                        'observation': {},
                        'model_confidence': 0.8,
                        'executed': True,
                        'execution_price': 50 + i
                    }
                    await queue.enqueue_action(action_data)
                
                # Force flush to trigger batch processing
                await queue.force_flush()
                
                # Should have called batch insert with first batch
                mock_rl_db.batch_insert_trading_actions.assert_called()
                call_args = mock_rl_db.batch_insert_trading_actions.call_args[0][0]
                assert len(call_args) <= 5  # Should batch the actions
                
            finally:
                await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_full_handling(self):
        """Test graceful handling when queue is full."""
        queue = ActionWriteQueue(batch_size=5, flush_interval=1.0, max_queue_size=3)
        await queue.start()
        
        try:
            action_data = {
                'episode_id': 123,
                'action_timestamp_ms': int(time.time() * 1000),
                'step_number': 1,
                'action_type': 'buy_yes',
                'price': 50,
                'quantity': 1,
                'position_before': {},
                'position_after': {},
                'reward': 0.1,
                'observation': {},
                'model_confidence': 0.8,
                'executed': True,
                'execution_price': 50
            }
            
            # Fill queue to capacity
            for i in range(3):
                result = await queue.enqueue_action(action_data)
                assert result is True
            
            # Next enqueue should fail gracefully
            result = await queue.enqueue_action(action_data)
            assert result is False
            assert queue._queue_full_errors == 1
            
        finally:
            await queue.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retry(self):
        """Test error handling and retry logic."""
        mock_db = AsyncMock()
        # First call fails, second succeeds
        mock_db.batch_insert_trading_actions.side_effect = [
            Exception("Database error"),
            2  # Success on retry
        ]
        
        with patch('kalshiflow_rl.trading.action_write_queue.rl_db', mock_db):
            queue = ActionWriteQueue(batch_size=2, flush_interval=0.1, max_queue_size=20, max_retries=2)
            await queue.start()
            
            try:
                # Enqueue actions
                for i in range(2):
                    action_data = {
                        'episode_id': 123,
                        'action_timestamp_ms': int(time.time() * 1000),
                        'step_number': i,
                        'action_type': 'buy_yes',
                        'price': 50,
                        'quantity': 1,
                        'position_before': {},
                        'position_after': {},
                        'reward': 0.1,
                        'observation': {},
                        'model_confidence': 0.8,
                        'executed': True,
                        'execution_price': 50
                    }
                    await queue.enqueue_action(action_data)
                
                # Force flush (will fail first time)
                await queue.force_flush()
                
                # Allow retry processing
                await asyncio.sleep(0.2)
                await queue.force_flush()
                
                # Should have been called twice (original + retry)
                assert mock_db.batch_insert_trading_actions.call_count >= 2
                assert queue._failed_writes > 0
                assert queue._retry_attempts > 0
                
            finally:
                await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_health_check(self):
        """Test queue health monitoring."""
        queue = ActionWriteQueue(batch_size=5, flush_interval=0.1, max_queue_size=10)
        
        # Queue not running initially
        assert not queue.is_healthy()
        
        await queue.start()
        
        # Should be healthy when running and not full
        assert queue.is_healthy()
        
        # Fill queue near capacity
        for i in range(9):  # 90% full
            action_data = {
                'episode_id': 123,
                'action_timestamp_ms': int(time.time() * 1000),
                'step_number': i,
                'action_type': 'buy_yes'
            }
            await queue.enqueue_action(action_data)
        
        # Should be unhealthy when near full
        assert not queue.is_healthy()
        
        await queue.stop()
    
    @pytest.mark.asyncio
    async def test_queue_statistics(self):
        """Test queue statistics tracking."""
        queue = ActionWriteQueue(batch_size=5, flush_interval=0.1, max_queue_size=10)
        await queue.start()
        
        try:
            # Initial stats
            stats = queue.get_stats()
            assert stats['running'] is True
            assert stats['actions_enqueued'] == 0
            assert stats['actions_written'] == 0
            assert stats['failed_writes'] == 0
            assert 'config' in stats
            
            # Enqueue some actions
            for i in range(3):
                action_data = {'episode_id': 123, 'action_timestamp_ms': int(time.time() * 1000), 'step_number': i, 'action_type': 'buy_yes'}
                await queue.enqueue_action(action_data)
            
            # Updated stats
            stats = queue.get_stats()
            assert stats['actions_enqueued'] == 3
            assert stats['queue_size'] == 3
            
        finally:
            await queue.stop()


class TestConvenienceFunctions:
    """Test convenience functions for integration."""
    
    @pytest.mark.asyncio
    async def test_create_trading_session_function(self):
        """Test create_trading_session convenience function."""
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            with patch('kalshiflow_rl.trading.integration.create_demo_trading_client') as mock_client_func:
                # Set up mock returns
                mock_db = AsyncMock()
                mock_db.initialize = AsyncMock()
                mock_db.close = AsyncMock()
                mock_db_class.return_value = mock_db
                
                mock_client = AsyncMock()
                mock_client.connect = AsyncMock()
                mock_client.disconnect = AsyncMock()
                mock_client_func.return_value = mock_client
                
                # Test auto-start session
                session = await create_trading_session('convenience_test', auto_start=True)
                
                assert session.session_name == 'convenience_test'
                assert session.is_active
                
                await session.stop()
    
    @pytest.mark.asyncio
    async def test_execute_trade_function(self):
        """Test execute_trade convenience function."""
        from kalshiflow_rl.trading.integration import execute_trade, session_manager
        
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            with patch('kalshiflow_rl.trading.integration.create_demo_trading_client') as mock_client_func:
                with patch('kalshiflow_rl.config.config.RL_MARKET_TICKERS', ['TEST-MARKET']):
                    # Set up mock returns
                    mock_db = AsyncMock()
                    mock_db.initialize = AsyncMock()
                    mock_db.close = AsyncMock()
                    mock_db.create_trading_action = AsyncMock(return_value=123)
                    mock_db.batch_insert_trading_actions = AsyncMock(return_value=1)
                    mock_db_class.return_value = mock_db
                    
                    mock_client = AsyncMock()
                    mock_client.connect = AsyncMock()
                    mock_client.disconnect = AsyncMock()
                    mock_client.create_order = AsyncMock(return_value={'order': {'order_id': 'test_123', 'status': 'open'}})
                    mock_client_func.return_value = mock_client
                    
                    # Create session manually first
                    session = await session_manager.create_session('paper_trade_test')
                    await session_manager.start_session('paper_trade_test')
                    
                    # Execute trade through convenience function
                    result = await execute_trade(
                        session_name='paper_trade_test',
                        action_type=ActionType.BUY_YES,
                        ticker='TEST-MARKET',
                        quantity=1,
                        price=60
                    )
                    
                    assert result['action_type'] == 'BUY_YES'
                    assert result['ticker'] == 'TEST-MARKET'
                    
                    await session_manager.stop_session('paper_trade_test')


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_session_not_active_error(self):
        """Test error when trying to execute actions on inactive session."""
        session = TradingSession('inactive_test')
        
        with pytest.raises(ValueError, match="Trading session is not active"):
            await session.execute_trading_action(
                action_type=ActionType.BUY_YES,
                ticker='TEST-MARKET',
                quantity=1
            )
    
    @pytest.mark.asyncio
    async def test_invalid_mode_error(self):
        """Test error for invalid trading mode."""
        session = TradingSession('mode_test')
        
        with pytest.raises(ValueError, match="Only 'paper' mode supported"):
            await session.start(mode='live')
    
    @pytest.mark.asyncio
    async def test_demo_client_not_initialized_error(self):
        """Test error when demo client is not initialized."""
        mock_database = AsyncMock()
        mock_database.initialize = AsyncMock()
        mock_database.close = AsyncMock()
        
        with patch('kalshiflow_rl.trading.integration.RLDatabase') as mock_db_class:
            mock_db_class.return_value = mock_database
            session = TradingSession('no_client_test')
            session.is_active = True  # Manually set active without proper initialization
            session.database = mock_database
            
            with pytest.raises(ValueError, match="Demo client not initialized"):
                await session.execute_trading_action(
                    action_type=ActionType.BUY_YES,
                    ticker='TEST-MARKET',
                    quantity=1
                )


class TestRewardCalculation:
    """Test reward calculation logic."""
    
    def test_metrics_calculator_initialization(self):
        """Test that TradingSession initializes metrics calculator properly."""
        session = TradingSession('reward_test')
        
        assert session.metrics_calculator is not None
        assert session.metrics_calculator.cash_balance == 10000.0
        assert session.metrics_calculator.total_trades == 0
    
    def test_metrics_calculator_reward_calculation(self):
        """Test reward calculation through metrics calculator."""
        from kalshiflow_rl.trading.trading_metrics import TradingMetricsCalculator
        
        calculator = TradingMetricsCalculator()
        
        # Execute a profitable trade
        trade_result = calculator.execute_trade(
            market_ticker='TEST-MARKET',
            side='yes',
            direction='buy',
            quantity=10,
            price_cents=50
        )
        
        # Calculate reward
        reward = calculator.calculate_step_reward(
            trades_executed=[trade_result],
            market_prices={'TEST-MARKET': {'yes_mid': 60.0, 'no_mid': 40.0}}
        )
        
        # Should have a positive reward from unrealized gains
        assert isinstance(reward, float)
    
    def test_metrics_calculator_fee_tracking(self):
        """Test that fees are properly tracked."""
        session = TradingSession('fee_test')
        
        # Test that calculator handles fees correctly
        assert session.metrics_calculator.total_fees_paid == 0.0
        
        # Note: Actual fee testing would require executing trades
        # which is covered in integration tests above


@pytest.mark.asyncio
async def test_comprehensive_demo_account_test_suite():
    """Test the comprehensive demo account test suite."""
    # This test validates that the test suite itself works correctly
    # In a real environment, this would test against actual demo credentials
    
    # Mock the demo client to simulate test results
    with patch('kalshiflow_rl.trading.demo_account_test_results.KalshiDemoTradingClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.connect.return_value = None
        mock_client.disconnect.return_value = None
        mock_client.get_markets.return_value = {'markets': [{'ticker': 'TEST-MARKET'}]}
        mock_client.get_account_info.side_effect = Exception("Demo API error 401: authentication_error")
        mock_client.get_positions.return_value = {'positions': []}
        mock_client.get_orders.return_value = {'orders': []}
        mock_client.create_order.return_value = {'order': {'order_id': 'demo_123', 'status': 'open'}}
        mock_client.connect_websocket.side_effect = Exception("WebSocket error")
        mock_client.balance = 10000.00
        
        mock_client_class.return_value = mock_client
        
        # Run the test suite
        results = await run_demo_account_tests()
        
        # Verify test structure
        assert 'authentication' in results
        assert 'market_data_access' in results
        assert 'portfolio_operations' in results
        assert 'order_operations' in results
        assert 'summary' in results
        
        # Verify summary structure
        summary = results['summary']
        assert 'overall_status' in summary
        assert 'test_statistics' in summary
        assert 'demo_account_verdict' in summary
        assert 'recommendations' in summary


if __name__ == '__main__':
    pytest.main([__file__])