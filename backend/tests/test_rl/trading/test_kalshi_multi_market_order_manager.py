"""
Tests for KalshiMultiMarketOrderManager - Actor/Trader Order Manager.

Tests the order manager used in the M1-M2 actor/trader pipeline.
Focuses on implemented functionality: order execution, position tracking, cash management.
"""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import time

from kalshiflow_rl.trading.kalshi_multi_market_order_manager import (
    KalshiMultiMarketOrderManager,
    OrderStatus,
    OrderSide,
    ContractSide,
    FillEvent
)


@pytest.fixture
def mock_trading_client():
    """Create a mock KalshiDemoTradingClient."""
    client = Mock()
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.create_order = AsyncMock()
    client.cancel_order = AsyncMock()
    return client


@pytest.fixture
def order_manager():
    """Create a KalshiMultiMarketOrderManager for testing."""
    return KalshiMultiMarketOrderManager(initial_cash=10000.0)


@pytest.fixture
def sample_orderbook_snapshot():
    """Create a sample orderbook snapshot for testing."""
    return {
        "yes_bids": {"50": 100, "49": 200},
        "yes_asks": {"51": 150, "52": 250},
        "no_bids": {"48": 100, "47": 200},
        "no_asks": {"49": 150, "50": 250}
    }


class TestKalshiMultiMarketOrderManager:
    """Test cases for KalshiMultiMarketOrderManager functionality."""
    
    def test_initialization(self, order_manager):
        """Test order manager initializes correctly."""
        assert order_manager.cash_balance == 10000.0
        assert order_manager.promised_cash == 0.0
        assert order_manager.initial_cash == 10000.0
        assert len(order_manager.open_orders) == 0
        assert len(order_manager.positions) == 0
        assert order_manager.trading_client is None
    
    @pytest.mark.asyncio
    async def test_execute_limit_order_action_hold(self, order_manager):
        """Test execute_limit_order_action with HOLD action."""
        result = await order_manager.execute_limit_order_action(
            action=0,  # HOLD
            market_ticker="TEST-MARKET"
        )
        
        assert result is not None
        assert result["status"] == "hold"
        assert result["executed"] is False
        assert result["action"] == 0
    
    @pytest.mark.asyncio
    async def test_execute_limit_order_action_contract_size(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test that contract size is fixed at 10 (matches training)."""
        order_manager.trading_client = mock_trading_client
        
        # Mock successful order creation
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_test_123"}
        }
        
        result = await order_manager.execute_limit_order_action(
            action=1,  # BUY_YES_LIMIT
            market_ticker="TEST-MARKET",
            orderbook_snapshot=sample_orderbook_snapshot
        )
        
        # Verify contract size is 10
        assert result is not None
        assert result["quantity"] == 10
        
        # Verify API was called with quantity=10
        mock_trading_client.create_order.assert_called_once()
        call_kwargs = mock_trading_client.create_order.call_args[1]
        assert call_kwargs["count"] == 10
    
    @pytest.mark.asyncio
    async def test_limit_price_calculation_from_orderbook(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test limit price calculation from orderbook snapshot."""
        order_manager.trading_client = mock_trading_client
        
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_test_123"}
        }
        
        # Test BUY YES (should use best ask = 51)
        result = await order_manager.execute_limit_order_action(
            action=1,  # BUY_YES_LIMIT
            market_ticker="TEST-MARKET",
            orderbook_snapshot=sample_orderbook_snapshot
        )
        
        assert result is not None
        assert result["limit_price"] == 51  # Best ask
        
        # Test SELL YES (should use best bid = 50)
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_test_124"}
        }
        
        result = await order_manager.execute_limit_order_action(
            action=2,  # SELL_YES_LIMIT
            market_ticker="TEST-MARKET",
            orderbook_snapshot=sample_orderbook_snapshot
        )
        
        assert result is not None
        assert result["limit_price"] == 50  # Best bid
    
    @pytest.mark.asyncio
    async def test_limit_price_fallback_without_orderbook(self, order_manager, mock_trading_client):
        """Test limit price falls back to mid-market (50) when no orderbook provided."""
        order_manager.trading_client = mock_trading_client
        
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_test_123"}
        }
        
        result = await order_manager.execute_limit_order_action(
            action=1,  # BUY_YES_LIMIT
            market_ticker="TEST-MARKET",
            orderbook_snapshot=None
        )
        
        assert result is not None
        assert result["limit_price"] == 50  # Default fallback
    
    @pytest.mark.asyncio
    async def test_cash_management_option_b(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test Option B cash management (deduct on place, restore on cancel)."""
        order_manager.trading_client = mock_trading_client
        
        initial_cash = order_manager.cash_balance
        
        # Mock successful order creation
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_test_123"}
        }
        
        # Place BUY order (should deduct cash)
        result = await order_manager.execute_limit_order_action(
            action=1,  # BUY_YES_LIMIT
            market_ticker="TEST-MARKET",
            orderbook_snapshot=sample_orderbook_snapshot
        )
        
        assert result is not None
        assert result["status"] == "placed"
        
        # Cash should be deducted (limit_price=51, quantity=10, cost=5.10)
        expected_cost = (51 / 100.0) * 10
        assert order_manager.cash_balance == initial_cash - expected_cost
        assert order_manager.promised_cash == expected_cost
        
        # Cancel order (should restore cash)
        order_id = result["order_id"]
        mock_trading_client.cancel_order.return_value = {"success": True}
        
        success = await order_manager.cancel_order(order_id)
        assert success
        
        # Cash should be restored
        assert order_manager.cash_balance == initial_cash
        assert order_manager.promised_cash == 0.0
    
    @pytest.mark.asyncio
    async def test_insufficient_cash_handling(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test handling of insufficient cash for BUY orders."""
        order_manager.trading_client = mock_trading_client
        order_manager.cash_balance = 0.10  # Very low cash
        
        result = await order_manager.execute_limit_order_action(
            action=1,  # BUY_YES_LIMIT
            market_ticker="TEST-MARKET",
            orderbook_snapshot=sample_orderbook_snapshot
        )
        
        assert result is not None
        assert result["status"] == "insufficient_cash"
        assert result["executed"] is False
        assert "required" in result
        assert "available" in result
    
    @pytest.mark.asyncio
    async def test_position_tracking(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test position tracking after fill."""
        order_manager.trading_client = mock_trading_client
        
        # Start fill processor task (required for fill processing)
        order_manager._fill_processor_task = asyncio.create_task(order_manager._process_fills())
        
        try:
            # Place order
            mock_trading_client.create_order.return_value = {
                "order": {"order_id": "kalshi_test_123"}
            }
            
            result = await order_manager.execute_limit_order_action(
                action=1,  # BUY_YES_LIMIT
                market_ticker="TEST-MARKET",
                orderbook_snapshot=sample_orderbook_snapshot
            )
            
            order_id = result["order_id"]
            kalshi_order_id = order_manager.open_orders[order_id].kalshi_order_id
            
            # Simulate fill
            await order_manager.queue_fill({
                "data": {
                    "order_id": kalshi_order_id,
                    "yes_price": 51,
                    "count": 10
                }
            })
            
            # Wait for fill processing
            await asyncio.sleep(0.2)
            
            # Check position was created
            positions = order_manager.get_positions()
            assert "TEST-MARKET" in positions
            position = positions["TEST-MARKET"]
            assert position["position"] == 10  # Long 10 YES contracts
        finally:
            # Clean up fill processor
            if order_manager._fill_processor_task:
                order_manager._fill_processor_task.cancel()
                try:
                    await order_manager._fill_processor_task
                except asyncio.CancelledError:
                    pass
    
    def test_get_order_features(self, order_manager):
        """Test order features extraction."""
        features = order_manager.get_order_features("TEST-MARKET")
        
        assert "has_open_buy" in features
        assert "has_open_sell" in features
        assert "time_since_order" in features
        assert features["has_open_buy"] == 0.0
        assert features["has_open_sell"] == 0.0
    
    def test_get_metrics(self, order_manager):
        """Test metrics reporting."""
        metrics = order_manager.get_metrics()
        
        assert "cash_balance" in metrics
        assert "promised_cash" in metrics
        assert "portfolio_value" in metrics
        assert "orders_placed" in metrics
        assert "orders_filled" in metrics
        assert "orders_cancelled" in metrics
        assert "open_orders_count" in metrics
        assert "positions_count" in metrics
        
        assert metrics["cash_balance"] == 10000.0
        assert metrics["orders_placed"] == 0
        assert metrics["orders_filled"] == 0
    
    @pytest.mark.asyncio
    async def test_action_space_mapping(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test action space mapping matches LimitOrderActions enum."""
        order_manager.trading_client = mock_trading_client
        
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_test_123"}
        }
        
        # Test all action mappings
        action_mappings = [
            (1, OrderSide.BUY, ContractSide.YES),   # BUY_YES_LIMIT
            (2, OrderSide.SELL, ContractSide.YES),   # SELL_YES_LIMIT
            (3, OrderSide.BUY, ContractSide.NO),     # BUY_NO_LIMIT
            (4, OrderSide.SELL, ContractSide.NO),    # SELL_NO_LIMIT
        ]
        
        for action, expected_side, expected_contract_side in action_mappings:
            result = await order_manager.execute_limit_order_action(
                action=action,
                market_ticker="TEST-MARKET",
                orderbook_snapshot=sample_orderbook_snapshot
            )
            
            assert result is not None
            assert result["side"] == expected_side.name
            assert result["contract_side"] == expected_contract_side.name
            
            # Reset mock for next iteration
            mock_trading_client.create_order.return_value = {
                "order": {"order_id": f"kalshi_test_{action}"}
            }

