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
            
            # Simulate fill (using Kalshi WebSocket format with 'msg' key)
            await order_manager.queue_fill({
                "type": "fill",
                "sid": 1,
                "msg": {
                    "order_id": kalshi_order_id,
                    "market_ticker": "TEST-MARKET",
                    "yes_price": 51,
                    "count": 10,
                    "action": "buy",
                    "side": "yes",
                    "ts": 1671899397,
                    "post_position": 10
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
    
    @pytest.mark.asyncio
    async def test_close_position_long_yes(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test close_position() closes a long YES position by placing SELL YES order."""
        order_manager.trading_client = mock_trading_client
        
        # Create a long YES position
        order_manager.positions["TEST-MARKET"] = Position(
            ticker="TEST-MARKET",
            contracts=10,  # Long YES
            cost_basis=500.0,  # $5.00 in cents
            realized_pnl=0.0,
            opened_at=time.time() - 100  # Opened 100 seconds ago
        )
        
        # Mock successful order creation
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_close_123"}
        }
        
        # Mock orderbook snapshot retrieval
        with patch.object(order_manager, '_get_orderbook_snapshot', new_callable=AsyncMock) as mock_snapshot:
            mock_snapshot.return_value = sample_orderbook_snapshot
            
            result = await order_manager.close_position("TEST-MARKET", "take_profit")
            
            assert result is not None
            assert result.get("executed") is True
            # Should place SELL YES order (action 2)
            mock_trading_client.create_order.assert_called_once()
            call_kwargs = mock_trading_client.create_order.call_args[1]
            assert call_kwargs["action"] == "sell"
            assert call_kwargs["side"] == "yes"
            # Should track closing reason
            assert "TEST-MARKET" in order_manager._active_closing_reasons
            assert order_manager._active_closing_reasons["TEST-MARKET"] == "take_profit"
    
    @pytest.mark.asyncio
    async def test_close_position_long_no(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test close_position() closes a long NO position by placing SELL NO order."""
        order_manager.trading_client = mock_trading_client
        
        # Create a long NO position
        order_manager.positions["TEST-MARKET"] = Position(
            ticker="TEST-MARKET",
            contracts=-10,  # Long NO (negative contracts)
            cost_basis=500.0,
            realized_pnl=0.0,
            opened_at=time.time() - 100
        )
        
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_close_456"}
        }
        
        with patch.object(order_manager, '_get_orderbook_snapshot', new_callable=AsyncMock) as mock_snapshot:
            mock_snapshot.return_value = sample_orderbook_snapshot
            
            result = await order_manager.close_position("TEST-MARKET", "stop_loss")
            
            assert result is not None
            assert result.get("executed") is True
            # Should place SELL NO order (action 4)
            call_kwargs = mock_trading_client.create_order.call_args[1]
            assert call_kwargs["action"] == "sell"
            assert call_kwargs["side"] == "no"
            assert order_manager._active_closing_reasons["TEST-MARKET"] == "stop_loss"
    
    @pytest.mark.asyncio
    async def test_close_position_flat_position(self, order_manager):
        """Test close_position() returns None for flat positions."""
        # Create flat position
        order_manager.positions["TEST-MARKET"] = Position(
            ticker="TEST-MARKET",
            contracts=0,  # Flat
            cost_basis=0.0,
            realized_pnl=0.0
        )
        
        result = await order_manager.close_position("TEST-MARKET", "take_profit")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_close_position_no_position(self, order_manager):
        """Test close_position() returns None when position doesn't exist."""
        result = await order_manager.close_position("NONEXISTENT", "take_profit")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_close_position_reason_propagation(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test that reason is propagated through execution result."""
        order_manager.trading_client = mock_trading_client
        
        order_manager.positions["TEST-MARKET"] = Position(
            ticker="TEST-MARKET",
            contracts=10,
            cost_basis=500.0,
            realized_pnl=0.0
        )
        
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_reason_123"}
        }
        
        with patch.object(order_manager, '_get_orderbook_snapshot', new_callable=AsyncMock) as mock_snapshot:
            mock_snapshot.return_value = sample_orderbook_snapshot
            
            result = await order_manager.close_position("TEST-MARKET", "cash_recovery")
            
            assert result is not None
            assert result.get("reason") == "close_position:cash_recovery"
    
    @pytest.mark.asyncio
    async def test_monitor_position_health_take_profit(self, order_manager, sample_orderbook_snapshot):
        """Test _monitor_position_health() detects take profit threshold."""
        # Set take profit threshold to 20%
        with patch('kalshiflow_rl.trading.kalshi_multi_market_order_manager.config') as mock_config:
            mock_config.RL_POSITION_TAKE_PROFIT_THRESHOLD = 0.20
            mock_config.RL_POSITION_STOP_LOSS_THRESHOLD = -0.10
            mock_config.RL_POSITION_MAX_HOLD_TIME_SECONDS = 3600
            
            # Create position with 25% profit (above 20% threshold)
            # Cost basis: 500 cents, current value: 625 cents (25% profit)
            order_manager.positions["TEST-MARKET"] = Position(
                ticker="TEST-MARKET",
                contracts=10,  # Long YES
                cost_basis=500.0,  # $5.00
                realized_pnl=0.0,
                opened_at=time.time() - 100
            )
            
            # Mock orderbook with YES price at 0.625 (62.5 cents) = 25% profit
            orderbook = {
                "yes_bid": 62,
                "yes_ask": 63,
                "no_bid": 37,
                "no_asks": 38
            }
            
            with patch.object(order_manager, '_get_orderbook_snapshot', new_callable=AsyncMock) as mock_snapshot:
                mock_snapshot.return_value = orderbook
                
                positions_to_close = await order_manager._monitor_position_health()
                
                assert len(positions_to_close) == 1
                assert positions_to_close[0][0] == "TEST-MARKET"
                assert positions_to_close[0][1] == "take_profit"
    
    @pytest.mark.asyncio
    async def test_monitor_position_health_stop_loss(self, order_manager):
        """Test _monitor_position_health() detects stop loss threshold."""
        with patch('kalshiflow_rl.trading.kalshi_multi_market_order_manager.config') as mock_config:
            mock_config.RL_POSITION_TAKE_PROFIT_THRESHOLD = 0.20
            mock_config.RL_POSITION_STOP_LOSS_THRESHOLD = -0.10
            mock_config.RL_POSITION_MAX_HOLD_TIME_SECONDS = 3600
            
            # Create position with -15% loss (below -10% threshold)
            order_manager.positions["TEST-MARKET"] = Position(
                ticker="TEST-MARKET",
                contracts=10,
                cost_basis=500.0,  # $5.00
                realized_pnl=0.0,
                opened_at=time.time() - 100
            )
            
            # Mock orderbook with YES price at 0.425 (42.5 cents) = -15% loss
            orderbook = {
                "yes_bid": 42,
                "yes_ask": 43,
                "no_bid": 57,
                "no_asks": 58
            }
            
            with patch.object(order_manager, '_get_orderbook_snapshot', new_callable=AsyncMock) as mock_snapshot:
                mock_snapshot.return_value = orderbook
                
                positions_to_close = await order_manager._monitor_position_health()
                
                assert len(positions_to_close) == 1
                assert positions_to_close[0][0] == "TEST-MARKET"
                assert positions_to_close[0][1] == "stop_loss"
    
    @pytest.mark.asyncio
    async def test_monitor_position_health_max_hold_time(self, order_manager, sample_orderbook_snapshot):
        """Test _monitor_position_health() detects max hold time threshold."""
        with patch('kalshiflow_rl.trading.kalshi_multi_market_order_manager.config') as mock_config:
            mock_config.RL_POSITION_TAKE_PROFIT_THRESHOLD = 0.20
            mock_config.RL_POSITION_STOP_LOSS_THRESHOLD = -0.10
            mock_config.RL_POSITION_MAX_HOLD_TIME_SECONDS = 100  # 100 seconds
            
            # Create position opened 150 seconds ago (exceeds 100s threshold)
            order_manager.positions["TEST-MARKET"] = Position(
                ticker="TEST-MARKET",
                contracts=10,
                cost_basis=500.0,
                realized_pnl=0.0,
                opened_at=time.time() - 150  # 150 seconds ago
            )
            
            with patch.object(order_manager, '_get_orderbook_snapshot', new_callable=AsyncMock) as mock_snapshot:
                mock_snapshot.return_value = sample_orderbook_snapshot
                
                positions_to_close = await order_manager._monitor_position_health()
                
                assert len(positions_to_close) == 1
                assert positions_to_close[0][0] == "TEST-MARKET"
                assert positions_to_close[0][1] == "max_hold_time"
    
    @pytest.mark.asyncio
    async def test_recover_cash_by_closing_positions(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test _recover_cash_by_closing_positions() closes worst positions when cash is low."""
        order_manager.trading_client = mock_trading_client
        order_manager.min_cash_reserve = 100.0
        order_manager.cash_balance = 50.0  # Below reserve
        
        # Create two positions with different P&L
        order_manager.positions["LOSER"] = Position(
            ticker="LOSER",
            contracts=10,
            cost_basis=500.0,  # $5.00
            realized_pnl=0.0,
            opened_at=time.time() - 100
        )
        
        order_manager.positions["WINNER"] = Position(
            ticker="WINNER",
            contracts=10,
            cost_basis=500.0,
            realized_pnl=0.0,
            opened_at=time.time() - 100
        )
        
        # Mock orderbook snapshots with different prices
        loser_orderbook = {"yes_bid": 40, "yes_ask": 41, "no_bid": 59, "no_asks": 60}  # -20% loss
        winner_orderbook = {"yes_bid": 60, "yes_ask": 61, "no_bid": 39, "no_asks": 40}  # +20% profit
        
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_recover_123"}
        }
        
        async def get_snapshot(ticker):
            if ticker == "LOSER":
                return loser_orderbook
            return winner_orderbook
        
        with patch.object(order_manager, '_get_orderbook_snapshot', side_effect=get_snapshot):
            # Mock close_position to track calls
            with patch.object(order_manager, 'close_position', new_callable=AsyncMock) as mock_close:
                mock_close.return_value = {"executed": True, "order_id": "test_123"}
                
                await order_manager._recover_cash_by_closing_positions()
                
                # Should have attempted to close positions
                assert mock_close.called
    
    @pytest.mark.asyncio
    async def test_recover_cash_skips_when_sufficient(self, order_manager):
        """Test _recover_cash_by_closing_positions() does nothing when cash is sufficient."""
        order_manager.min_cash_reserve = 100.0
        order_manager.cash_balance = 200.0  # Above reserve
        
        # Create a position
        order_manager.positions["TEST-MARKET"] = Position(
            ticker="TEST-MARKET",
            contracts=10,
            cost_basis=500.0,
            realized_pnl=0.0
        )
        
        with patch.object(order_manager, 'close_position', new_callable=AsyncMock) as mock_close:
            await order_manager._recover_cash_by_closing_positions()
            
            # Should not attempt to close positions
            assert not mock_close.called
    
    @pytest.mark.asyncio
    async def test_monitor_market_states_closing_market(self, order_manager, mock_trading_client):
        """Test _monitor_market_states() closes positions in closing markets."""
        order_manager.trading_client = mock_trading_client
        
        order_manager.positions["CLOSING-MARKET"] = Position(
            ticker="CLOSING-MARKET",
            contracts=10,
            cost_basis=500.0,
            realized_pnl=0.0
        )
        
        # Mock market info with closing status
        mock_trading_client.get_markets = AsyncMock(return_value={
            "markets": [{
                "ticker": "CLOSING-MARKET",
                "status": "ending"
            }]
        })
        
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_market_close_123"}
        }
        
        with patch.object(order_manager, '_get_orderbook_snapshot', new_callable=AsyncMock) as mock_snapshot:
            mock_snapshot.return_value = {"yes_bid": 50, "yes_ask": 51, "no_bid": 49, "no_asks": 50}
            
            with patch.object(order_manager, 'close_position', new_callable=AsyncMock) as mock_close:
                mock_close.return_value = {"executed": True}
                
                await order_manager._monitor_market_states()
                
                # Should attempt to close position
                assert mock_close.called
                call_args = mock_close.call_args
                assert call_args[0][0] == "CLOSING-MARKET"
                assert call_args[0][1] == "market_closing"
    
    @pytest.mark.asyncio
    async def test_execute_order_with_reason(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test execute_order() includes reason in execution result."""
        order_manager.trading_client = mock_trading_client
        
        mock_trading_client.create_order.return_value = {
            "order": {"order_id": "kalshi_reason_123"}
        }
        
        result = await order_manager.execute_order(
            market_ticker="TEST-MARKET",
            action=1,  # BUY_YES_LIMIT
            orderbook_snapshot=sample_orderbook_snapshot,
            reason="close_position:take_profit"
        )
        
        assert result is not None
        assert result.get("reason") == "close_position:take_profit"
        assert result.get("status") == "placed"
    
    @pytest.mark.asyncio
    async def test_broadcast_position_update_with_closing_reason(self, order_manager):
        """Test _broadcast_position_update_with_changes() includes closing_reason."""
        # Create position with active closing reason
        order_manager.positions["TEST-MARKET"] = Position(
            ticker="TEST-MARKET",
            contracts=10,
            cost_basis=500.0,
            realized_pnl=0.0
        )
        order_manager._active_closing_reasons["TEST-MARKET"] = "take_profit"
        
        # Mock websocket manager
        mock_ws_manager = Mock()
        mock_ws_manager.broadcast_position_update = AsyncMock()
        order_manager._websocket_manager = mock_ws_manager
        
        # Broadcast position update
        await order_manager._broadcast_position_update_with_changes(
            market_ticker="TEST-MARKET",
            changed_fields=["position"],
            previous_values={"position": 10},
            was_settled=False
        )
        
        # Verify closing_reason was included
        assert mock_ws_manager.broadcast_position_update.called
        call_args = mock_ws_manager.broadcast_position_update.call_args[0][0]
        assert call_args.get("closing_reason") == "take_profit"
    
    @pytest.mark.asyncio
    async def test_monitor_and_close_positions(self, order_manager, mock_trading_client, sample_orderbook_snapshot):
        """Test _monitor_and_close_positions() orchestrates health monitoring and closing."""
        order_manager.trading_client = mock_trading_client
        
        with patch('kalshiflow_rl.trading.kalshi_multi_market_order_manager.config') as mock_config:
            mock_config.RL_POSITION_TAKE_PROFIT_THRESHOLD = 0.20
            mock_config.RL_POSITION_STOP_LOSS_THRESHOLD = -0.10
            mock_config.RL_POSITION_MAX_HOLD_TIME_SECONDS = 3600
            
            # Create position that should be closed
            order_manager.positions["TEST-MARKET"] = Position(
                ticker="TEST-MARKET",
                contracts=10,
                cost_basis=500.0,
                realized_pnl=0.0,
                opened_at=time.time() - 100
            )
            
            # Mock orderbook with profitable price
            orderbook = {"yes_bid": 62, "yes_ask": 63, "no_bid": 37, "no_asks": 38}
            
            mock_trading_client.create_order.return_value = {
                "order": {"order_id": "kalshi_monitor_123"}
            }
            
            with patch.object(order_manager, '_get_orderbook_snapshot', new_callable=AsyncMock) as mock_snapshot:
                mock_snapshot.return_value = orderbook
                
                with patch.object(order_manager, 'close_position', new_callable=AsyncMock) as mock_close:
                    mock_close.return_value = {"executed": True, "order_id": "test_123"}
                    
                    await order_manager._monitor_and_close_positions()
                    
                    # Should have attempted to close position
                    assert mock_close.called

