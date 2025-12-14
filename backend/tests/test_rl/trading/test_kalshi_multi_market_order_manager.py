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
    FillEvent,
    OrderInfo,
    Position
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
    
    @pytest.mark.asyncio
    async def test_sync_orders_restart_scenario(self, order_manager):
        """Test sync adds orders found in Kalshi but not in local memory (restart scenario)."""
        order_manager.trading_client = Mock()
        
        initial_cash = order_manager.cash_balance
        
        # Simulate Kalshi has an order we don't know about
        kalshi_order = {
            "order_id": "kalshi_restart_123",
            "ticker": "TEST-MARKET",
            "side": "yes",
            "action": "buy",
            "status": "resting",
            "yes_price": 55,
            "remaining_count": 10,
            "initial_count": 10,
            "fill_count": 0
        }
        
        order_manager.trading_client.get_orders = AsyncMock(return_value={
            "orders": [kalshi_order]
        })
        
        # Sync should add the order (restart scenario)
        stats = await order_manager.sync_orders_with_kalshi(is_startup=True)
        
        assert stats["added"] == 1
        assert stats["found_in_kalshi"] == 1
        assert len(order_manager.open_orders) == 1
        
        # Verify order was added correctly
        our_order_id = list(order_manager.open_orders.keys())[0]
        order = order_manager.open_orders[our_order_id]
        assert order.kalshi_order_id == "kalshi_restart_123"
        assert order.ticker == "TEST-MARKET"
        assert order.status == OrderStatus.PENDING
        assert order.quantity == 10
        # Cash should be reserved for BUY order
        expected_cost = (55 / 100.0) * 10
        assert order_manager.cash_balance == initial_cash - expected_cost
        assert order_manager.promised_cash == expected_cost
    
    @pytest.mark.asyncio
    async def test_sync_orders_status_mismatch(self, order_manager):
        """Test sync updates local order status to match Kalshi."""
        order_manager.trading_client = Mock()
        
        # Set up cash balance to match order state
        promised_cash = 5.5
        order_manager.cash_balance -= promised_cash
        order_manager.promised_cash = promised_cash
        
        # Create local order with PENDING status
        our_order_id = order_manager._generate_order_id()
        local_order = OrderInfo(
            order_id=our_order_id,
            kalshi_order_id="kalshi_status_123",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time(),
            promised_cash=promised_cash
        )
        order_manager.open_orders[our_order_id] = local_order
        order_manager._kalshi_to_internal["kalshi_status_123"] = our_order_id
        
        # Kalshi says order is executed
        kalshi_order = {
            "order_id": "kalshi_status_123",
            "ticker": "TEST-MARKET",
            "side": "yes",
            "action": "buy",
            "status": "executed",
            "yes_price": 55,
            "remaining_count": 0,
            "initial_count": 10,
            "fill_count": 10
        }
        
        order_manager.trading_client.get_orders = AsyncMock(return_value={
            "orders": [kalshi_order]
        })
        
        initial_cash = order_manager.cash_balance
        initial_promised = order_manager.promised_cash
        
        # Sync should update status and process fill
        stats = await order_manager.sync_orders_with_kalshi()
        
        assert stats["updated"] >= 1
        assert stats["discrepancies"] >= 1
        # Order should be removed (fully filled)
        assert "kalshi_status_123" not in order_manager._kalshi_to_internal
        # Promised cash should be released
        assert order_manager.promised_cash < initial_promised
        assert order_manager.promised_cash == 0.0
    
    @pytest.mark.asyncio
    async def test_sync_orders_partial_fill(self, order_manager):
        """Test sync processes partial fills from Kalshi."""
        order_manager.trading_client = Mock()
        
        # Set up cash balance to match order state
        promised_cash = 5.5
        order_manager.cash_balance -= promised_cash
        order_manager.promised_cash = promised_cash
        
        # Create local order with full quantity
        our_order_id = order_manager._generate_order_id()
        local_order = OrderInfo(
            order_id=our_order_id,
            kalshi_order_id="kalshi_partial_123",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,  # Full quantity
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time(),
            promised_cash=promised_cash
        )
        order_manager.open_orders[our_order_id] = local_order
        order_manager._kalshi_to_internal["kalshi_partial_123"] = our_order_id
        
        # Kalshi says 5 contracts filled, 5 remaining
        kalshi_order = {
            "order_id": "kalshi_partial_123",
            "ticker": "TEST-MARKET",
            "side": "yes",
            "action": "buy",
            "status": "resting",
            "yes_price": 55,
            "remaining_count": 5,
            "initial_count": 10,
            "fill_count": 5
        }
        
        order_manager.trading_client.get_orders = AsyncMock(return_value={
            "orders": [kalshi_order]
        })
        
        initial_promised = order_manager.promised_cash
        
        # Sync should process partial fill
        stats = await order_manager.sync_orders_with_kalshi()
        
        assert stats["partial_fills"] >= 1
        assert stats["updated"] >= 1
        # Order should still exist but with reduced quantity
        assert "kalshi_partial_123" in order_manager._kalshi_to_internal
        updated_order = order_manager.open_orders[our_order_id]
        assert updated_order.quantity == 5
        # Promised cash should be reduced proportionally
        assert order_manager.promised_cash < initial_promised
    
    @pytest.mark.asyncio
    async def test_sync_orders_external_cancellation(self, order_manager):
        """Test sync removes orders that exist locally but not in Kalshi."""
        order_manager.trading_client = Mock()
        
        # Set up cash balance to match order state
        promised_cash = 5.5
        initial_cash = order_manager.cash_balance
        order_manager.cash_balance -= promised_cash
        order_manager.promised_cash = promised_cash
        
        # Create local order
        our_order_id = order_manager._generate_order_id()
        local_order = OrderInfo(
            order_id=our_order_id,
            kalshi_order_id="kalshi_cancelled_123",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time(),
            promised_cash=promised_cash
        )
        order_manager.open_orders[our_order_id] = local_order
        order_manager._kalshi_to_internal["kalshi_cancelled_123"] = our_order_id
        
        # Kalshi has no orders (order was cancelled externally)
        order_manager.trading_client.get_orders = AsyncMock(return_value={
            "orders": []
        })
        
        # Sync should remove the order and restore cash
        stats = await order_manager.sync_orders_with_kalshi()
        
        assert stats["removed"] == 1
        assert len(order_manager.open_orders) == 0
        assert "kalshi_cancelled_123" not in order_manager._kalshi_to_internal
        # Cash should be restored to original (we subtracted 5.5, then restored 5.5)
        assert order_manager.cash_balance == initial_cash
        assert order_manager.promised_cash == 0.0
    
    @pytest.mark.asyncio
    async def test_sync_orders_no_discrepancies(self, order_manager):
        """Test sync when local state matches Kalshi (no discrepancies)."""
        order_manager.trading_client = Mock()
        
        # Set up cash balance to match order state
        promised_cash = 5.5
        order_manager.cash_balance -= promised_cash
        order_manager.promised_cash = promised_cash
        
        # Create local order
        our_order_id = order_manager._generate_order_id()
        local_order = OrderInfo(
            order_id=our_order_id,
            kalshi_order_id="kalshi_sync_123",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time(),
            promised_cash=promised_cash
        )
        order_manager.open_orders[our_order_id] = local_order
        order_manager._kalshi_to_internal["kalshi_sync_123"] = our_order_id
        
        # Kalshi has matching order
        kalshi_order = {
            "order_id": "kalshi_sync_123",
            "ticker": "TEST-MARKET",
            "side": "yes",
            "action": "buy",
            "status": "resting",
            "yes_price": 55,
            "remaining_count": 10,
            "initial_count": 10,
            "fill_count": 0
        }
        
        order_manager.trading_client.get_orders = AsyncMock(return_value={
            "orders": [kalshi_order]
        })
        
        # Sync should find no discrepancies
        stats = await order_manager.sync_orders_with_kalshi()
        
        assert stats["discrepancies"] == 0
        assert stats["updated"] == 0
        assert stats["added"] == 0
        assert stats["removed"] == 0
        # Order should still exist
        assert "kalshi_sync_123" in order_manager._kalshi_to_internal
    
    @pytest.mark.asyncio
    async def test_sync_positions(self, order_manager):
        """Test position synchronization with Kalshi."""
        order_manager.trading_client = Mock()
        
        # Create local position
        order_manager.positions["TEST-MARKET"] = Position(
            ticker="TEST-MARKET",
            contracts=5,
            cost_basis=100.0,
            realized_pnl=10.0
        )
        
        # Kalshi has different position
        kalshi_positions = [
            {
                "ticker": "TEST-MARKET",
                "position": 10  # Different from local
            },
            {
                "ticker": "OTHER-MARKET",
                "position": 3  # New position
            }
        ]
        
        order_manager.trading_client.get_positions = AsyncMock(return_value={
            "positions": kalshi_positions
        })
        
        # Sync should update positions
        await order_manager._sync_positions_with_kalshi()
        
        # Local position should match Kalshi
        assert order_manager.positions["TEST-MARKET"].contracts == 10
        # New position should be added
        assert "OTHER-MARKET" in order_manager.positions
        assert order_manager.positions["OTHER-MARKET"].contracts == 3
    
    @pytest.mark.asyncio
    async def test_sync_orders_filled_order_not_added(self, order_manager):
        """Test that fully filled orders from Kalshi are not added to open_orders."""
        order_manager.trading_client = Mock()
        
        # Kalshi has a filled order
        kalshi_order = {
            "order_id": "kalshi_filled_123",
            "ticker": "TEST-MARKET",
            "side": "yes",
            "action": "buy",
            "status": "executed",
            "yes_price": 55,
            "remaining_count": 0,
            "initial_count": 10,
            "fill_count": 10
        }
        
        order_manager.trading_client.get_orders = AsyncMock(return_value={
            "orders": [kalshi_order]
        })
        
        initial_cash = order_manager.cash_balance
        
        # Sync should process fills but not add to open_orders
        stats = await order_manager.sync_orders_with_kalshi()
        
        # Order should not be in open_orders (it's filled)
        assert len(order_manager.open_orders) == 0
        # But position should be updated
        assert "TEST-MARKET" in order_manager.positions
        assert order_manager.positions["TEST-MARKET"].contracts == 10
    

