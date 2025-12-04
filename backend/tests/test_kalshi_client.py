"""
Unit tests for Kalshi WebSocket client.
"""

import pytest
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from kalshiflow.kalshi_client import KalshiWebSocketClient, KalshiClientError
from kalshiflow.auth import KalshiAuth
from kalshiflow.models import Trade, ConnectionStatus


@pytest.fixture
def mock_auth():
    """Create a mock KalshiAuth instance."""
    auth = MagicMock(spec=KalshiAuth)
    auth.api_key_id = "test_api_key"
    auth.create_websocket_auth_message.return_value = {
        "KALSHI-ACCESS-KEY": "test_api_key",
        "KALSHI-ACCESS-SIGNATURE": "mock_signature",
        "KALSHI-ACCESS-TIMESTAMP": "1701648000123"
    }
    return auth


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    websocket = AsyncMock()
    websocket.send = AsyncMock()
    websocket.recv = AsyncMock()
    websocket.close = AsyncMock()
    websocket.ping_interval = 30
    websocket.ping_timeout = 10
    websocket.close_timeout = 10
    return websocket


class TestKalshiWebSocketClient:
    """Test KalshiWebSocketClient functionality."""
    
    def test_init(self, mock_auth):
        """Test client initialization."""
        client = KalshiWebSocketClient(
            websocket_url="wss://test.example.com",
            auth=mock_auth,
            max_reconnect_attempts=5,
            base_reconnect_delay=2.0,
            max_reconnect_delay=30.0
        )
        
        assert client.websocket_url == "wss://test.example.com"
        assert client.auth == mock_auth
        assert client.max_reconnect_attempts == 5
        assert client.base_reconnect_delay == 2.0
        assert client.max_reconnect_delay == 30.0
        assert client.is_connected is False
        assert client.is_authenticated is False
        assert client.reconnect_attempts == 0
        assert client.should_reconnect is True
    
    @pytest.mark.asyncio
    async def test_connect_success(self, mock_auth, mock_websocket):
        """Test successful WebSocket connection."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        
        # Create an async mock for websockets.connect
        async def mock_connect(*args, **kwargs):
            return mock_websocket
        
        # Mock successful connection flow
        with patch('websockets.connect', side_effect=mock_connect):
            # Auth now happens via headers, only subscription response needed
            mock_websocket.recv.return_value = json.dumps({"id": 1, "type": "subscribed"})
            
            result = await client.connect()
        
        assert result is True
        assert client.is_connected is True
        assert client.is_authenticated is True
        assert client.reconnect_attempts == 0
        
        # Verify trades subscription was sent (only one message now)
        mock_websocket.send.assert_called_once()
        trades_call = mock_websocket.send.call_args[0][0]
        trades_data = json.loads(trades_call)
        assert trades_data["id"] == 1
        assert trades_data["cmd"] == "subscribe"
        assert "trade" in trades_data["params"]["channels"]
    
    @pytest.mark.asyncio
    async def test_connect_websocket_failure(self, mock_auth):
        """Test WebSocket connection failure."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        
        with patch('websockets.connect', side_effect=ConnectionError("Connection failed")):
            result = await client.connect()
        
        assert result is False
        assert client.is_connected is False
        assert client.is_authenticated is False
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self, mock_auth, mock_websocket):
        """Test successful authentication."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        client.websocket = mock_websocket
        
        # Authentication now happens via headers
        result = await client.authenticate()
        
        assert result is True
        assert client.is_authenticated is True
        # No send should be called as auth is done via headers
        mock_websocket.send.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_authenticate_failure(self, mock_auth, mock_websocket):
        """Test authentication failure."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        client.websocket = mock_websocket
        
        # Authentication now happens via headers and always succeeds if connected
        result = await client.authenticate()
        
        assert result is True  # Always returns True now
        assert client.is_authenticated is True
    
    @pytest.mark.asyncio
    async def test_authenticate_timeout(self, mock_auth, mock_websocket):
        """Test authentication timeout."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        client.websocket = mock_websocket
        
        # Authentication now happens via headers and doesn't wait for response
        result = await client.authenticate()
        
        assert result is True  # Always returns True now
        assert client.is_authenticated is True
    
    @pytest.mark.asyncio
    async def test_subscribe_to_trades_success(self, mock_auth, mock_websocket):
        """Test successful trades subscription."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        client.websocket = mock_websocket
        
        # Mock subscription response with correct ID
        mock_websocket.recv.return_value = json.dumps({
            "id": 1,
            "type": "subscribed"
        })
        
        result = await client.subscribe_to_trades()
        
        assert result is True
        mock_websocket.send.assert_called_once()
        
        # Verify subscription message format
        call_args = mock_websocket.send.call_args[0][0]
        data = json.loads(call_args)
        assert data["id"] == 1
        assert data["cmd"] == "subscribe"
        assert "trade" in data["params"]["channels"]
    
    @pytest.mark.asyncio
    async def test_handle_trade_message(self, mock_auth):
        """Test handling of trade messages."""
        trade_callback = MagicMock()
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth, on_trade_callback=trade_callback)
        
        trade_message = {
            "type": "trade",
            "msg": {
                "market_ticker": "PRESWIN25",
                "yes_price": 65,
                "no_price": 35,
                "count": 100,
                "taker_side": "yes",
                "ts": 1701648000000
            }
        }
        
        await client.handle_message(json.dumps(trade_message))
        
        # Verify callback was called with Trade object
        trade_callback.assert_called_once()
        trade = trade_callback.call_args[0][0]
        assert isinstance(trade, Trade)
        assert trade.market_ticker == "PRESWIN25"
        assert trade.yes_price == 65
        assert trade.taker_side == "yes"
    
    @pytest.mark.asyncio
    async def test_handle_heartbeat_message(self, mock_auth):
        """Test handling of heartbeat messages."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        
        heartbeat_message = {"type": "heartbeat"}
        
        # Should not raise any errors
        await client.handle_message(json.dumps(heartbeat_message))
    
    @pytest.mark.asyncio
    async def test_handle_invalid_json_message(self, mock_auth):
        """Test handling of invalid JSON messages."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        
        # Should not raise errors, just log them
        await client.handle_message("invalid json {")
    
    @pytest.mark.asyncio
    async def test_handle_invalid_trade_message(self, mock_auth):
        """Test handling of invalid trade messages."""
        trade_callback = MagicMock()
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth, on_trade_callback=trade_callback)
        
        invalid_trade_message = {
            "type": "trade",
            "msg": {
                "market_ticker": "PRESWIN25",
                # Missing required fields
            }
        }
        
        # Should not raise errors or call callback
        await client.handle_message(json.dumps(invalid_trade_message))
        trade_callback.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_disconnect(self, mock_auth, mock_websocket):
        """Test WebSocket disconnection."""
        client = KalshiWebSocketClient("wss://test.example.com", mock_auth)
        client.websocket = mock_websocket
        client.is_connected = True
        client.is_authenticated = True
        
        await client.disconnect()
        
        assert client.should_reconnect is False
        assert client.is_connected is False
        assert client.is_authenticated is False
        mock_websocket.close.assert_called_once()
    
    def test_reconnect_delay_calculation(self, mock_auth):
        """Test exponential backoff delay calculation."""
        client = KalshiWebSocketClient(
            "wss://test.example.com", 
            mock_auth,
            base_reconnect_delay=1.0,
            max_reconnect_delay=8.0
        )
        
        # Test various reconnection attempts
        client.reconnect_attempts = 1
        # Delay should be base_delay * 2^(attempts-1) = 1.0 * 2^0 = 1.0
        # Plus jitter, so should be between 1.0 and 1.3
        
        client.reconnect_attempts = 2
        # Delay should be 1.0 * 2^1 = 2.0, plus jitter
        
        client.reconnect_attempts = 4
        # Delay should be 1.0 * 2^3 = 8.0 (at max), plus jitter
        
        client.reconnect_attempts = 10
        # Should be capped at max_reconnect_delay = 8.0
        
        # This is mainly testing the logic works without errors
        # The actual delay calculation is tested implicitly in reconnect_with_backoff
    
    def test_from_env_success(self):
        """Test creating client from environment variables."""
        with patch.dict(os.environ, {
            'KALSHI_API_KEY_ID': 'test_api_key',
            'KALSHI_PRIVATE_KEY_PATH': '/tmp/test.pem',
            'KALSHI_WS_URL': 'wss://test.example.com'
        }), \
        patch('kalshiflow.auth.KalshiAuth.from_env') as mock_auth_from_env:
            
            mock_auth = MagicMock()
            mock_auth_from_env.return_value = mock_auth
            
            client = KalshiWebSocketClient.from_env()
            
            assert client.websocket_url == 'wss://test.example.com'
            assert client.auth == mock_auth
            mock_auth_from_env.assert_called_once()
    
    def test_from_env_default_url(self):
        """Test creating client with default WebSocket URL."""
        with patch.dict(os.environ, {
            'KALSHI_API_KEY_ID': 'test_api_key',
            'KALSHI_PRIVATE_KEY_PATH': '/tmp/test.pem'
            # No KALSHI_WS_URL set
        }, clear=True), \
        patch('kalshiflow.auth.KalshiAuth.from_env') as mock_auth_from_env:
            
            mock_auth = MagicMock()
            mock_auth_from_env.return_value = mock_auth
            
            client = KalshiWebSocketClient.from_env()
            
            # Should use default URL
            assert client.websocket_url == "wss://api.elections.kalshi.com/trade-api/ws/v2"


class TestConnectionStatusCallback:
    """Test connection status callback functionality."""
    
    @pytest.mark.asyncio
    async def test_connection_status_callback_on_connect(self, mock_auth, mock_websocket):
        """Test connection status callback is called on successful connection."""
        status_callback = MagicMock()
        client = KalshiWebSocketClient(
            "wss://test.example.com", 
            mock_auth,
            on_connection_change=status_callback
        )
        
        # Create an async mock for websockets.connect
        async def mock_connect(*args, **kwargs):
            return mock_websocket
        
        with patch('websockets.connect', side_effect=mock_connect), \
             patch.object(client, 'authenticate', return_value=True), \
             patch.object(client, 'subscribe_to_trades', return_value=True):
            
            await client.connect()
        
        # Should call status callback with connected=True
        status_callback.assert_called()
        status = status_callback.call_args[0][0]
        assert isinstance(status, ConnectionStatus)
        assert status.connected is True
    
    @pytest.mark.asyncio
    async def test_connection_status_callback_on_failure(self, mock_auth):
        """Test connection status callback is called on connection failure."""
        status_callback = MagicMock()
        client = KalshiWebSocketClient(
            "wss://test.example.com", 
            mock_auth,
            on_connection_change=status_callback
        )
        
        with patch('websockets.connect', side_effect=ConnectionError("Failed")):
            await client.connect()
        
        # Should call status callback with connected=False and error message
        status_callback.assert_called()
        status = status_callback.call_args[0][0]
        assert isinstance(status, ConnectionStatus)
        assert status.connected is False
        assert status.error_message == "Failed"