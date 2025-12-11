"""
Tests for OrderManager implementations.

This test suite validates both SimulatedOrderManager and KalshiOrderManager
implementations to ensure they provide consistent interfaces and behavior
for the RL environment.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal

import numpy as np

from kalshiflow_rl.trading.order_manager import (
    OrderManager,
    SimulatedOrderManager, 
    KalshiOrderManager,
    OrderInfo,
    Position,
    OrderFeatures,
    OrderStatus,
    OrderSide,
    ContractSide
)
from kalshiflow_rl.data.orderbook_state import OrderbookState


class TestOrderManagerBase:
    """Test the base OrderManager abstract class."""
    
    def test_order_manager_initialization(self):
        """Test basic initialization of OrderManager base class."""
        # Can't instantiate abstract class directly, but can test via subclass
        manager = SimulatedOrderManager(initial_cash=500.0)
        
        assert manager.initial_cash == 500.0
        assert manager.cash_balance == 500.0
        assert manager.positions == {}
        assert manager.open_orders == {}
    
    def test_position_creation(self):
        """Test Position dataclass functionality."""
        position = Position(
            ticker="TEST-MARKET",
            contracts=10,
            cost_basis=50.0,
            realized_pnl=5.0
        )
        
        assert position.ticker == "TEST-MARKET"
        assert position.contracts == 10
        assert position.is_long_yes
        assert not position.is_long_no
        assert not position.is_flat
        
        # Test unrealized PnL calculation
        unrealized = position.get_unrealized_pnl(current_yes_price=0.6)
        # 10 contracts * 0.6 price = 6.0 value, minus 50.0 cost basis = -44.0
        assert unrealized == 6.0 - 50.0
    
    def test_order_info_creation(self):
        """Test OrderInfo dataclass functionality."""
        order = OrderInfo(
            order_id="test_order_1",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        assert order.order_id == "test_order_1"
        assert order.side == OrderSide.BUY
        assert order.contract_side == ContractSide.YES
        assert order.is_active()
        assert order.time_since_placed >= 0
    
    def test_order_features_conversion(self):
        """Test OrderFeatures to array conversion."""
        features = OrderFeatures(
            has_open_buy=1.0,
            has_open_sell=0.0,
            buy_distance_from_mid=0.1,
            sell_distance_from_mid=0.0,
            time_since_order=0.5
        )
        
        array = features.to_array()
        expected = np.array([1.0, 0.0, 0.1, 0.0, 0.5], dtype=np.float32)
        np.testing.assert_array_equal(array, expected)


class TestSimulatedOrderManager:
    """Test the SimulatedOrderManager implementation."""
    
    @pytest.fixture
    def orderbook(self):
        """Create a sample orderbook for testing."""
        orderbook = OrderbookState("TEST-MARKET")
        
        # Apply snapshot data to populate the orderbook
        snapshot_data = {
            "yes_bids": {
                "50": 100,  # 50¢ bid with size 100
                "49": 200   # 49¢ bid with size 200
            },
            "yes_asks": {
                "51": 150,  # 51¢ ask with size 150
                "52": 250   # 52¢ ask with size 250
            },
            "no_bids": {
                "48": 100,  # 48¢ no bid  
                "47": 200   # 47¢ no bid
            },
            "no_asks": {
                "49": 150,  # 49¢ no ask
                "50": 250   # 50¢ no ask
            }
        }
        
        orderbook.apply_snapshot(snapshot_data)
        return orderbook
    
    @pytest.fixture
    def manager(self):
        """Create a SimulatedOrderManager for testing."""
        return SimulatedOrderManager(initial_cash=1000.0)
    
    @pytest.mark.asyncio
    async def test_aggressive_buy_immediate_fill(self, manager, orderbook):
        """Test aggressive buy order that fills immediately."""
        # Aggressive buy should cross the spread and fill at ask (51¢)
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="aggressive"
        )
        
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == 51  # Filled at ask
        assert order.quantity == 10
        
        # Check position was created
        assert "TEST-MARKET" in manager.positions
        position = manager.positions["TEST-MARKET"]
        assert position.contracts == 10  # Long 10 YES contracts
        assert position.cost_basis == 5.10  # 10 * 0.51
        
        # Check cash was deducted
        assert manager.cash_balance == 1000.0 - 5.10
    
    @pytest.mark.asyncio
    async def test_passive_buy_limit_order(self, manager, orderbook):
        """Test passive buy order that becomes a limit order."""
        # Passive buy should join the bid (50¢) and not fill immediately
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        assert order is not None
        assert order.status == OrderStatus.PENDING
        assert order.limit_price == 50  # At current bid
        
        # Should be in open orders
        assert order.order_id in manager.open_orders
        
        # No position created yet
        assert "TEST-MARKET" not in manager.positions
        
        # No cash deducted yet
        assert manager.cash_balance == 1000.0
    
    @pytest.mark.asyncio
    async def test_order_fill_on_price_movement(self, manager, orderbook):
        """Test that pending orders fill when prices move in their favor."""
        # Place passive buy order at 50¢
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        assert order.status == OrderStatus.PENDING
        
        # Create new orderbook with lower ask price (order should fill)
        new_orderbook = OrderbookState("TEST-MARKET")
        
        # Apply snapshot where ask drops to 50¢ or below so our 50¢ bid can fill
        new_snapshot_data = {
            "yes_bids": {
                "49": 200,   # 49¢ bid 
                "48": 300    # 48¢ bid
            },
            "yes_asks": {
                "50": 150,   # 50¢ ask (matches our bid, should fill)
                "51": 250    # 51¢ ask
            },
            "no_bids": {
                "49": 100,   # NO bid
                "48": 200    # NO bid
            },
            "no_asks": {
                "50": 150,   # NO ask
                "51": 250    # NO ask
            }
        }
        
        new_orderbook.apply_snapshot(new_snapshot_data)
        
        # Check for fills
        filled_orders = await manager.check_fills(new_orderbook)
        
        assert len(filled_orders) == 1
        assert filled_orders[0].order_id == order.order_id
        assert filled_orders[0].status == OrderStatus.FILLED
        assert filled_orders[0].fill_price == 50  # Filled at new ask
        
        # Order should be removed from open orders
        assert order.order_id not in manager.open_orders
    
    @pytest.mark.asyncio
    async def test_order_cancellation(self, manager, orderbook):
        """Test order cancellation."""
        # Place order
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        assert order.order_id in manager.open_orders
        
        # Cancel order
        success = await manager.cancel_order(order.order_id)
        
        assert success
        assert order.status == OrderStatus.CANCELLED
        assert order.order_id not in manager.open_orders
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, manager, orderbook):
        """Test cancelling all orders."""
        # Place multiple orders
        order1 = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        order2 = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.SELL,
            contract_side=ContractSide.YES,
            quantity=5,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        assert len(manager.open_orders) == 2
        
        # Cancel all orders
        cancelled_count = await manager.cancel_all_orders()
        
        assert cancelled_count == 2
        assert len(manager.open_orders) == 0
    
    @pytest.mark.asyncio
    async def test_order_amendment(self, manager, orderbook):
        """Test order price amendment."""
        # Place passive order
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        assert order.limit_price == 50
        
        # Amend to aggressive price that should fill
        success = await manager.amend_order(order.order_id, 51, orderbook)
        
        assert success
        # Order should have filled at the new aggressive price
        assert order.status == OrderStatus.FILLED
        assert order.order_id not in manager.open_orders
    
    @pytest.mark.asyncio
    async def test_no_contract_position_tracking(self, manager, orderbook):
        """Test NO contract position tracking with Kalshi convention."""
        # Buy NO contracts (should result in negative YES position)
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.NO,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="aggressive"
        )
        
        assert order.status == OrderStatus.FILLED
        
        # Check position (should be -10 for long NO position)
        position = manager.positions["TEST-MARKET"]
        assert position.contracts == -10  # Long NO = negative YES
        assert position.is_long_no
        assert not position.is_long_yes
    
    @pytest.mark.asyncio
    async def test_position_pnl_calculation(self, manager, orderbook):
        """Test position P&L calculation with price changes."""
        # Buy YES contracts at 51¢
        await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="aggressive"
        )
        
        position = manager.positions["TEST-MARKET"]
        assert position.contracts == 10
        assert position.cost_basis == 5.10  # 10 * 0.51
        
        # Test unrealized P&L at different prices
        pnl_at_60 = position.get_unrealized_pnl(0.60)  # Price rises to 60¢
        assert pnl_at_60 == (10 * 0.60) - 5.10  # 6.00 - 5.10 = 0.90
        
        pnl_at_40 = position.get_unrealized_pnl(0.40)  # Price falls to 40¢
        assert pnl_at_40 == (10 * 0.40) - 5.10  # 4.00 - 5.10 = -1.10
    
    def test_order_features_extraction(self, manager, orderbook):
        """Test extraction of order features for RL observation."""
        # No orders initially
        features = manager.get_order_features("TEST-MARKET", orderbook)
        
        assert features.has_open_buy == 0.0
        assert features.has_open_sell == 0.0
        assert features.buy_distance_from_mid == 0.0
        assert features.sell_distance_from_mid == 0.0
        assert features.time_since_order == 0.0
    
    @pytest.mark.asyncio
    async def test_order_features_with_open_orders(self, manager, orderbook):
        """Test order features with open orders."""
        # Place a buy order
        buy_order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        features = manager.get_order_features("TEST-MARKET", orderbook)
        
        assert features.has_open_buy == 1.0
        assert features.has_open_sell == 0.0
        
        # Buy distance should be calculated from mid price
        mid_price = (50 + 51) / 2.0  # 50.5¢
        expected_distance = abs(mid_price - buy_order.limit_price) / 100.0
        assert abs(features.buy_distance_from_mid - expected_distance) < 0.01
    
    def test_portfolio_value_calculation(self, manager, orderbook):
        """Test total portfolio value calculation."""
        # Initial value should be just cash
        initial_value = manager.get_total_portfolio_value({"TEST-MARKET": 0.50})
        assert initial_value == 1000.0
        
        # After getting a position, should include unrealized P&L
        manager.positions["TEST-MARKET"] = Position(
            ticker="TEST-MARKET",
            contracts=10,
            cost_basis=5.0,
            realized_pnl=0.0
        )
        
        # At 60¢ price: 10 * 0.60 = 6.0 value, 5.0 cost basis + 1.0 unrealized = 6.0
        value_at_60 = manager.get_total_portfolio_value({"TEST-MARKET": 0.60})
        expected = 1000.0 + 5.0 + 1.0  # cash + cost_basis + unrealized_pnl
        assert value_at_60 == expected


class TestKalshiOrderManager:
    """Test the KalshiOrderManager implementation."""
    
    @pytest.fixture
    def mock_demo_client(self):
        """Create a mock KalshiDemoTradingClient."""
        client = Mock()
        client.create_order = AsyncMock()
        client.cancel_order = AsyncMock()
        client.get_orders = AsyncMock()
        client.get_positions = AsyncMock()
        client.get_account_info = AsyncMock()
        return client
    
    @pytest.fixture
    def orderbook(self):
        """Create a sample orderbook for testing."""
        orderbook = OrderbookState("TEST-MARKET")
        
        # Apply snapshot data to populate the orderbook
        snapshot_data = {
            "yes_bids": {
                "50": 100,
                "49": 200
            },
            "yes_asks": {
                "51": 150,
                "52": 250
            },
            "no_bids": {
                "48": 100,
                "47": 200
            },
            "no_asks": {
                "49": 150,
                "50": 250
            }
        }
        
        orderbook.apply_snapshot(snapshot_data)
        return orderbook
    
    @pytest.fixture
    def manager(self, mock_demo_client):
        """Create a KalshiOrderManager for testing."""
        return KalshiOrderManager(mock_demo_client, initial_cash=1000.0)
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, manager, mock_demo_client, orderbook):
        """Test successful order placement via Kalshi API."""
        # Mock successful order creation response
        mock_demo_client.create_order.return_value = {
            "order": {"order_id": "kalshi_123", "status": "resting"}
        }
        
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="aggressive"
        )
        
        assert order is not None
        assert order.status == OrderStatus.PENDING
        assert order.limit_price == 51  # Aggressive buy at ask
        
        # Check that API was called correctly
        mock_demo_client.create_order.assert_called_once_with(
            ticker="TEST-MARKET",
            action="buy",
            side="yes",
            count=10,
            price=51,
            type="limit"
        )
        
        # Check order tracking
        assert order.order_id in manager.open_orders
        assert order.order_id in manager._kalshi_order_mapping
        assert manager._kalshi_order_mapping[order.order_id] == "kalshi_123"
    
    @pytest.mark.asyncio
    async def test_place_order_failure(self, manager, mock_demo_client, orderbook):
        """Test order placement failure."""
        # Mock failed order creation
        mock_demo_client.create_order.return_value = {"error": "Insufficient balance"}
        
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="aggressive"
        )
        
        assert order is None
    
    @pytest.mark.asyncio
    async def test_cancel_order_success(self, manager, mock_demo_client, orderbook):
        """Test successful order cancellation."""
        # First place an order
        mock_demo_client.create_order.return_value = {
            "order": {"order_id": "kalshi_123", "status": "resting"}
        }
        
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="aggressive"
        )
        
        # Mock successful cancellation
        mock_demo_client.cancel_order.return_value = {"success": True}
        
        # Cancel the order
        success = await manager.cancel_order(order.order_id)
        
        assert success
        assert order.status == OrderStatus.CANCELLED
        assert order.order_id not in manager.open_orders
        assert order.order_id not in manager._kalshi_order_mapping
        
        # Check that API was called correctly
        mock_demo_client.cancel_order.assert_called_once_with("kalshi_123")
    
    @pytest.mark.asyncio
    async def test_check_fills_success(self, manager, mock_demo_client, orderbook):
        """Test checking for order fills."""
        # Place an order first
        mock_demo_client.create_order.return_value = {
            "order": {"order_id": "kalshi_123", "status": "resting"}
        }
        
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="aggressive"
        )
        
        # Mock filled order response
        mock_demo_client.get_orders.return_value = {
            "orders": [{
                "order_id": "kalshi_123",
                "status": "filled",
                "yes_price": 51,
                "ticker": "TEST-MARKET"
            }]
        }
        
        # Check for fills
        filled_orders = await manager.check_fills(orderbook)
        
        assert len(filled_orders) == 1
        assert filled_orders[0].order_id == order.order_id
        assert filled_orders[0].status == OrderStatus.FILLED
        assert filled_orders[0].fill_price == 51
        
        # Order should be removed from tracking
        assert order.order_id not in manager.open_orders
        assert order.order_id not in manager._kalshi_order_mapping
    
    @pytest.mark.asyncio
    async def test_position_sync(self, manager, mock_demo_client):
        """Test syncing positions with Kalshi API."""
        mock_demo_client.get_positions.return_value = {
            "positions": [{
                "ticker": "TEST-MARKET",
                "position": 10
            }]
        }
        
        mock_demo_client.get_account_info.return_value = {
            "balance": 95000  # $950 in cents
        }
        
        await manager.sync_positions_with_kalshi()
        
        # Check positions were synced
        assert "TEST-MARKET" in manager.positions
        assert manager.positions["TEST-MARKET"].contracts == 10
        
        # Check balance was synced
        assert manager.cash_balance == 950.0
    
    @pytest.mark.asyncio
    async def test_order_amendment_via_cancel_replace(self, manager, mock_demo_client, orderbook):
        """Test order amendment via cancel + replace strategy."""
        # Place initial order
        mock_demo_client.create_order.return_value = {
            "order": {"order_id": "kalshi_123", "status": "resting"}
        }
        
        order = await manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        # Mock successful cancellation and new order
        mock_demo_client.cancel_order.return_value = {"success": True}
        mock_demo_client.create_order.return_value = {
            "order": {"order_id": "kalshi_124", "status": "resting"}
        }
        
        # Amend order
        success = await manager.amend_order(order.order_id, 52, orderbook)
        
        assert success
        
        # Should have cancelled old and placed new order
        assert mock_demo_client.cancel_order.call_count == 1
        assert mock_demo_client.create_order.call_count == 2  # Original + replacement


class TestOrderManagerIntegration:
    """Integration tests for OrderManager with real scenarios."""
    
    @pytest.mark.asyncio
    async def test_trading_scenario_profit_and_loss(self):
        """Test a complete trading scenario with profit and loss."""
        manager = SimulatedOrderManager(initial_cash=1000.0)
        
        # Initial orderbook
        orderbook1 = OrderbookState("PROFIT-TEST")
        
        snapshot_data_1 = {
            "yes_bids": {
                "40": 100   # YES bid at 40¢
            },
            "yes_asks": {
                "41": 100   # YES ask at 41¢
            },
            "no_bids": {
                "58": 100   # NO bid at 58¢
            },
            "no_asks": {
                "59": 100   # NO ask at 59¢
            }
        }
        
        orderbook1.apply_snapshot(snapshot_data_1)
        
        # Buy YES at 41¢
        buy_order = await manager.place_order(
            ticker="PROFIT-TEST",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=100,
            orderbook=orderbook1,
            pricing_strategy="aggressive"
        )
        
        assert buy_order.status == OrderStatus.FILLED
        assert manager.cash_balance == 1000.0 - 41.0  # Paid $41 for 100 contracts
        
        # Price moves favorably - new orderbook
        orderbook2 = OrderbookState("PROFIT-TEST")
        
        snapshot_data_2 = {
            "yes_bids": {
                "60": 100   # YES bid at 60¢ 
            },
            "yes_asks": {
                "61": 100   # YES ask at 61¢
            },
            "no_bids": {
                "38": 100   # NO bid at 38¢
            },
            "no_asks": {
                "39": 100   # NO ask at 39¢
            }
        }
        
        orderbook2.apply_snapshot(snapshot_data_2)
        
        # Sell YES at 60¢ for profit
        sell_order = await manager.place_order(
            ticker="PROFIT-TEST",
            side=OrderSide.SELL,
            contract_side=ContractSide.YES,
            quantity=100,
            orderbook=orderbook2,
            pricing_strategy="aggressive"
        )
        
        assert sell_order.status == OrderStatus.FILLED
        
        # Check final position and P&L
        position = manager.positions["PROFIT-TEST"]
        assert position.contracts == 0  # Flat position
        assert position.realized_pnl == 19.0  # $60 - $41 = $19 profit
        assert manager.cash_balance == 1000.0 - 41.0 + 60.0  # Original + profit
    
    @pytest.mark.asyncio
    async def test_order_feature_evolution(self):
        """Test how order features evolve during trading."""
        manager = SimulatedOrderManager(initial_cash=1000.0)
        
        orderbook = OrderbookState("FEATURES-TEST")
        
        snapshot_data = {
            "yes_bids": {
                "50": 100   # YES bid
            },
            "yes_asks": {
                "51": 100   # YES ask
            },
            "no_bids": {
                "48": 100   # NO bid
            },
            "no_asks": {
                "49": 100   # NO ask
            }
        }
        
        orderbook.apply_snapshot(snapshot_data)
        
        # Initially no orders
        features_0 = manager.get_order_features("FEATURES-TEST", orderbook)
        assert features_0.has_open_buy == 0.0
        assert features_0.has_open_sell == 0.0
        
        # Place buy order
        await manager.place_order(
            ticker="FEATURES-TEST",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        features_1 = manager.get_order_features("FEATURES-TEST", orderbook)
        assert features_1.has_open_buy == 1.0
        assert features_1.has_open_sell == 0.0
        assert features_1.buy_distance_from_mid > 0.0
        
        # Place sell order too
        await manager.place_order(
            ticker="FEATURES-TEST",
            side=OrderSide.SELL,
            contract_side=ContractSide.YES,
            quantity=5,
            orderbook=orderbook,
            pricing_strategy="passive"
        )
        
        features_2 = manager.get_order_features("FEATURES-TEST", orderbook)
        assert features_2.has_open_buy == 1.0
        assert features_2.has_open_sell == 1.0
        assert features_2.sell_distance_from_mid > 0.0
        
        # Cancel all orders
        await manager.cancel_all_orders()
        
        features_3 = manager.get_order_features("FEATURES-TEST", orderbook)
        assert features_3.has_open_buy == 0.0
        assert features_3.has_open_sell == 0.0
    
    def test_price_calculation_strategies(self):
        """Test different pricing strategy calculations."""
        manager = SimulatedOrderManager()
        
        orderbook = OrderbookState("PRICING-TEST")
        
        # Apply snapshot data to create wide spread
        snapshot_data = {
            "yes_bids": {
                "45": 100   # YES bid
            },
            "yes_asks": {
                "55": 100   # YES ask  
            },
            "no_bids": {
                "44": 100   # NO bid
            },
            "no_asks": {
                "54": 100   # NO ask
            }
        }
        
        orderbook.apply_snapshot(snapshot_data)
        
        # Test YES contract pricing
        aggressive_buy_yes = manager._calculate_limit_price(
            OrderSide.BUY, ContractSide.YES, orderbook, "aggressive"
        )
        assert aggressive_buy_yes == 55  # Buy at ask
        
        passive_buy_yes = manager._calculate_limit_price(
            OrderSide.BUY, ContractSide.YES, orderbook, "passive"
        )
        assert passive_buy_yes == 45  # Buy at bid
        
        mid_buy_yes = manager._calculate_limit_price(
            OrderSide.BUY, ContractSide.YES, orderbook, "mid"
        )
        assert mid_buy_yes == 50  # Buy at mid
        
        # Test NO contract pricing (derived from YES prices)
        aggressive_buy_no = manager._calculate_limit_price(
            OrderSide.BUY, ContractSide.NO, orderbook, "aggressive"
        )
        assert aggressive_buy_no == 54  # NO ask = 99 - YES bid = 99 - 45 = 54
        
        passive_buy_no = manager._calculate_limit_price(
            OrderSide.BUY, ContractSide.NO, orderbook, "passive"
        )
        assert passive_buy_no == 44  # NO bid = 99 - YES ask = 99 - 55 = 44