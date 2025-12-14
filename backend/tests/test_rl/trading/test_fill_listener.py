"""
Tests for FillListener service.

Tests the WebSocket-based fill notification listener.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from kalshiflow_rl.trading.fill_listener import FillListener, FillListenerError, FillListenerAuthError
from kalshiflow_rl.trading.kalshi_multi_market_order_manager import (
    KalshiMultiMarketOrderManager,
    FillEvent,
)


class TestFillEvent:
    """Tests for FillEvent parsing."""
    
    def test_from_kalshi_message_full(self):
        """Test FillEvent parsing with all fields."""
        message = {
            "type": "fill",
            "sid": 13,
            "msg": {
                "trade_id": "d91bc706-ee49-470d-82d8-11418bda6fed",
                "order_id": "ee587a1c-8b87-4dcf-b721-9f6f790619fa",
                "market_ticker": "HIGHNY-22DEC23-B53.5",
                "is_taker": True,
                "side": "yes",
                "yes_price": 75,
                "count": 278,
                "action": "buy",
                "ts": 1671899397,
                "post_position": 500
            }
        }
        
        fill_event = FillEvent.from_kalshi_message(message)
        
        assert fill_event is not None
        assert fill_event.kalshi_order_id == "ee587a1c-8b87-4dcf-b721-9f6f790619fa"
        assert fill_event.fill_price == 75
        assert fill_event.fill_quantity == 278
        assert fill_event.market_ticker == "HIGHNY-22DEC23-B53.5"
        assert fill_event.post_position == 500
        assert fill_event.action == "buy"
        assert fill_event.side == "yes"
        assert fill_event.fill_timestamp == 1671899397.0
    
    def test_from_kalshi_message_minimal(self):
        """Test FillEvent parsing with minimal required fields."""
        message = {
            "type": "fill",
            "msg": {
                "order_id": "test-order-123",
                "yes_price": 50,
                "count": 10
            }
        }
        
        fill_event = FillEvent.from_kalshi_message(message)
        
        assert fill_event is not None
        assert fill_event.kalshi_order_id == "test-order-123"
        assert fill_event.fill_price == 50
        assert fill_event.fill_quantity == 10
        assert fill_event.market_ticker == ""  # Default when not provided
        assert fill_event.post_position is None  # Optional field
    
    def test_from_kalshi_message_empty(self):
        """Test FillEvent parsing with empty message."""
        message = {}
        
        fill_event = FillEvent.from_kalshi_message(message)
        
        assert fill_event is None  # Should return None for empty msg
    
    def test_from_kalshi_message_wrong_key(self):
        """Test FillEvent parsing with 'data' key (old format) returns None."""
        message = {
            "type": "fill",
            "data": {  # Wrong key - should be 'msg'
                "order_id": "test-123",
                "yes_price": 50,
                "count": 10
            }
        }
        
        fill_event = FillEvent.from_kalshi_message(message)
        
        # Should return None because 'msg' key is empty
        assert fill_event is None


class TestFillListener:
    """Tests for FillListener service."""
    
    @pytest.fixture
    def mock_order_manager(self):
        """Create mock order manager."""
        manager = Mock(spec=KalshiMultiMarketOrderManager)
        manager.queue_fill = AsyncMock()
        return manager
    
    def test_initialization(self, mock_order_manager):
        """Test FillListener initialization."""
        listener = FillListener(
            order_manager=mock_order_manager,
            ws_url="wss://test.example.com/ws",
            reconnect_delay_seconds=3.0,
            heartbeat_timeout_seconds=15.0
        )
        
        assert listener.order_manager == mock_order_manager
        assert listener.ws_url == "wss://test.example.com/ws"
        assert listener.reconnect_delay == 3.0
        assert listener.heartbeat_timeout == 15.0
        assert not listener._running
        assert listener._fills_received == 0
        assert listener._fills_processed == 0
    
    def test_get_metrics(self, mock_order_manager):
        """Test FillListener metrics reporting."""
        listener = FillListener(order_manager=mock_order_manager)
        
        metrics = listener.get_metrics()
        
        assert "running" in metrics
        assert "connected" in metrics
        assert "fills_received" in metrics
        assert "fills_processed" in metrics
        assert "connection_count" in metrics
        assert metrics["running"] is False
        assert metrics["fills_received"] == 0
    
    def test_get_status(self, mock_order_manager):
        """Test FillListener status reporting."""
        listener = FillListener(order_manager=mock_order_manager)
        
        status = listener.get_status()
        
        assert status["service"] == "FillListener"
        assert status["status"] == "stopped"
        assert "metrics" in status
        assert "ws_url" in status


class TestPartialFillHandling:
    """Tests for partial fill handling in OrderManager."""
    
    @pytest.fixture
    def order_manager(self):
        """Create order manager for testing."""
        return KalshiMultiMarketOrderManager(initial_cash=10000.0)
    
    @pytest.fixture
    def mock_trading_client(self):
        """Create mock trading client."""
        client = Mock()
        client.create_order = AsyncMock()
        client.cancel_order = AsyncMock()
        return client
    
    @pytest.fixture
    def sample_orderbook_snapshot(self):
        """Sample orderbook for testing."""
        return {
            'yes_bids': {'49': 200, '50': 100},
            'yes_asks': {'51': 150, '52': 250},
            'no_bids': {'47': 200, '48': 100},
            'no_asks': {'49': 150, '50': 250}
        }
    
    @pytest.mark.asyncio
    async def test_partial_fill_proportional_cash_release(
        self, order_manager, mock_trading_client, sample_orderbook_snapshot
    ):
        """Test that partial fills release proportional cash."""
        order_manager.trading_client = mock_trading_client
        
        # Start fill processor
        order_manager._fill_processor_task = asyncio.create_task(order_manager._process_fills())
        
        try:
            # Place order for 10 contracts
            mock_trading_client.create_order.return_value = {
                "order": {"order_id": "kalshi_partial_123"}
            }
            
            initial_cash = order_manager.cash_balance
            
            result = await order_manager.execute_limit_order_action(
                action=1,  # BUY_YES_LIMIT
                market_ticker="TEST-PARTIAL",
                orderbook_snapshot=sample_orderbook_snapshot
            )
            
            order_id = result["order_id"]
            order = order_manager.open_orders[order_id]
            initial_promised = order.promised_cash
            cash_after_order = order_manager.cash_balance
            
            # Simulate partial fill (5 of 10 contracts)
            await order_manager.queue_fill({
                "type": "fill",
                "msg": {
                    "order_id": order.kalshi_order_id,
                    "market_ticker": "TEST-PARTIAL",
                    "yes_price": 51,
                    "count": 5,  # Half of the order
                    "action": "buy",
                    "side": "yes",
                    "ts": 1671899397,
                    "post_position": 5
                }
            })
            
            await asyncio.sleep(0.2)
            
            # Order should still be open with 5 remaining
            assert order_id in order_manager.open_orders
            remaining_order = order_manager.open_orders[order_id]
            assert remaining_order.quantity == 5  # Half remaining
            
            # Promised cash should be halved
            expected_remaining_promised = initial_promised * 0.5
            assert abs(remaining_order.promised_cash - expected_remaining_promised) < 0.01
            
            # Position should be created with 5 contracts
            positions = order_manager.get_positions()
            assert "TEST-PARTIAL" in positions
            assert positions["TEST-PARTIAL"]["position"] == 5
            
        finally:
            if order_manager._fill_processor_task:
                order_manager._fill_processor_task.cancel()
                try:
                    await order_manager._fill_processor_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    async def test_complete_fill_removes_order(
        self, order_manager, mock_trading_client, sample_orderbook_snapshot
    ):
        """Test that complete fills remove order from tracking."""
        order_manager.trading_client = mock_trading_client
        
        # Start fill processor
        order_manager._fill_processor_task = asyncio.create_task(order_manager._process_fills())
        
        try:
            # Place order for 10 contracts
            mock_trading_client.create_order.return_value = {
                "order": {"order_id": "kalshi_complete_123"}
            }
            
            result = await order_manager.execute_limit_order_action(
                action=1,  # BUY_YES_LIMIT
                market_ticker="TEST-COMPLETE",
                orderbook_snapshot=sample_orderbook_snapshot
            )
            
            order_id = result["order_id"]
            order = order_manager.open_orders[order_id]
            
            # Simulate complete fill (10 of 10 contracts)
            await order_manager.queue_fill({
                "type": "fill",
                "msg": {
                    "order_id": order.kalshi_order_id,
                    "market_ticker": "TEST-COMPLETE",
                    "yes_price": 51,
                    "count": 10,  # Full order
                    "action": "buy",
                    "side": "yes",
                    "ts": 1671899397,
                    "post_position": 10
                }
            })
            
            await asyncio.sleep(0.2)
            
            # Order should be removed from tracking
            assert order_id not in order_manager.open_orders
            
            # Position should be created with 10 contracts
            positions = order_manager.get_positions()
            assert "TEST-COMPLETE" in positions
            assert positions["TEST-COMPLETE"]["position"] == 10
            
        finally:
            if order_manager._fill_processor_task:
                order_manager._fill_processor_task.cancel()
                try:
                    await order_manager._fill_processor_task
                except asyncio.CancelledError:
                    pass
    
    @pytest.mark.asyncio
    async def test_post_position_updates_position_directly(
        self, order_manager, mock_trading_client, sample_orderbook_snapshot
    ):
        """Test that post_position from Kalshi updates position accurately."""
        order_manager.trading_client = mock_trading_client
        
        # Start fill processor
        order_manager._fill_processor_task = asyncio.create_task(order_manager._process_fills())
        
        try:
            # Place order
            mock_trading_client.create_order.return_value = {
                "order": {"order_id": "kalshi_postpos_123"}
            }
            
            result = await order_manager.execute_limit_order_action(
                action=1,  # BUY_YES_LIMIT
                market_ticker="TEST-POSTPOS",
                orderbook_snapshot=sample_orderbook_snapshot
            )
            
            order_id = result["order_id"]
            order = order_manager.open_orders[order_id]
            
            # Simulate fill with post_position that differs from calculated
            # (simulating scenario where Kalshi's position is authoritative)
            await order_manager.queue_fill({
                "type": "fill",
                "msg": {
                    "order_id": order.kalshi_order_id,
                    "market_ticker": "TEST-POSTPOS",
                    "yes_price": 51,
                    "count": 10,
                    "action": "buy",
                    "side": "yes",
                    "ts": 1671899397,
                    "post_position": 25  # Indicates existing position + fill
                }
            })
            
            await asyncio.sleep(0.2)
            
            # Position should match post_position, not calculated value
            positions = order_manager.get_positions()
            assert "TEST-POSTPOS" in positions
            assert positions["TEST-POSTPOS"]["position"] == 25  # From post_position
            
        finally:
            if order_manager._fill_processor_task:
                order_manager._fill_processor_task.cancel()
                try:
                    await order_manager._fill_processor_task
                except asyncio.CancelledError:
                    pass
