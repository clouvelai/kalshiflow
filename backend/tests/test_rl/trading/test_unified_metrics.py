"""
Tests for OrderManager data classes and enums (Training components).

Tests data classes and enums used in training (SimulatedOrderManager).
DO NOT MODIFY - These tests are critical for training pipeline validation.

For actor/trader order manager tests, see test_kalshi_multi_market_order_manager.py
"""

import pytest
import time
from unittest.mock import Mock, AsyncMock

# Import actual implemented classes (used in training)
from kalshiflow_rl.trading.order_manager import OrderStatus, OrderSide, ContractSide, Position, OrderInfo, OrderFeatures


class TestDataClasses:
    """Test the data classes that are actually implemented."""
    
    def test_position_creation(self):
        """Test Position data class."""
        position = Position(
            ticker="TEST-MARKET",
            contracts=10,
            cost_basis=450.0,
            realized_pnl=25.0
        )
        
        assert position.ticker == "TEST-MARKET"
        assert position.contracts == 10
        assert position.is_long_yes is True
        assert position.is_long_no is False
        assert position.is_flat is False
        
    def test_position_no_contracts(self):
        """Test Position with NO contracts (negative)."""
        position = Position(
            ticker="TEST-MARKET-NO",
            contracts=-5,
            cost_basis=200.0,
            realized_pnl=0.0
        )
        
        assert position.contracts == -5
        assert position.is_long_yes is False
        assert position.is_long_no is True
        assert position.is_flat is False
        
    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        # YES position
        yes_position = Position("TEST", 10, 400.0, 0.0)  # Bought 10 YES at $40 total
        
        # Price goes up to 0.5 (50 cents)
        pnl = yes_position.get_unrealized_pnl(0.5)
        expected = 10 * 0.5 - 400.0  # Current value - cost basis
        assert pnl == expected
        
    def test_order_info_creation(self):
        """Test OrderInfo data class."""
        import time
        order = OrderInfo(
            order_id="test-123",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=45,  # In cents
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        assert order.order_id == "test-123"
        assert order.ticker == "TEST-MARKET"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.PENDING
        assert order.is_active() is True
        
    def test_order_features(self):
        """Test OrderFeatures data class."""
        features = OrderFeatures(
            has_open_buy=1.0,
            has_open_sell=0.0,
            buy_distance_from_mid=0.3,
            sell_distance_from_mid=0.0,
            time_since_order=0.5
        )
        
        array = features.to_array()
        assert array.shape == (5,)
        assert array[0] == 1.0  # has_open_buy
        assert array[1] == 0.0  # has_open_sell


class TestOrderEnums:
    """Test order-related enums and constants."""
    
    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING == 0
        assert OrderStatus.FILLED == 1
        assert OrderStatus.CANCELLED == 2
        assert OrderStatus.REJECTED == 3
    
    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY == 0
        assert OrderSide.SELL == 1
    
    def test_contract_side_enum(self):
        """Test ContractSide enum values."""
        assert ContractSide.YES == 0
        assert ContractSide.NO == 1




if __name__ == "__main__":
    pytest.main([__file__, "-v"])