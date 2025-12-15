"""
Test orderbook depth consumption for SimulatedOrderManager.

Tests the new depth consumption feature that makes order filling realistic by:
1. Walking the orderbook for large orders
2. Calculating volume-weighted average price (VWAP)
3. Tracking consumed liquidity to prevent double-filling
4. Ensuring small orders still fill at best bid/ask
"""

import pytest
import time
from unittest.mock import Mock

from kalshiflow_rl.trading.order_manager import (
    SimulatedOrderManager, OrderInfo, OrderSide, ContractSide, OrderStatus, ConsumedLiquidity
)
from kalshiflow_rl.data.orderbook_state import OrderbookState
from sortedcontainers import SortedDict


@pytest.fixture
def order_manager():
    """Create a SimulatedOrderManager with default settings."""
    return SimulatedOrderManager(initial_cash=100000, small_order_threshold=20)


@pytest.fixture
def sample_orderbook():
    """Create a sample orderbook with realistic depth."""
    orderbook = OrderbookState("TEST-MARKET")
    
    # Set up YES side: bids and derived asks
    # YES bids: 48¢ (100), 47¢ (150), 46¢ (200)
    # Derived NO asks: 52¢ (100), 53¢ (150), 54¢ (200)
    orderbook.yes_bids[48] = 100
    orderbook.yes_bids[47] = 150
    orderbook.yes_bids[46] = 200
    
    # YES asks: 50¢ (80), 51¢ (120), 52¢ (180)
    # Derived from NO bids: 50¢ (80), 49¢ (120), 48¢ (180)
    orderbook.yes_asks[50] = 80
    orderbook.yes_asks[51] = 120
    orderbook.yes_asks[52] = 180
    
    # Set up NO side (derived from YES)
    # NO bids: 50¢ (80), 49¢ (120), 48¢ (180) [99-YES_ASK]
    orderbook.no_bids[50] = 80
    orderbook.no_bids[49] = 120
    orderbook.no_bids[48] = 180
    
    # NO asks: 52¢ (100), 53¢ (150), 54¢ (200) [99-YES_BID]
    orderbook.no_asks[52] = 100
    orderbook.no_asks[53] = 150
    orderbook.no_asks[54] = 200
    
    orderbook.last_sequence = 1
    orderbook.last_update_time = int(time.time() * 1000)
    
    return orderbook


class TestSmallOrderOptimization:
    """Test that small orders (<20 contracts) still fill at best price."""
    
    def test_small_buy_order_fills_at_best_ask(self, order_manager, sample_orderbook):
        """Small buy orders should fill at best ask without slippage."""
        order = OrderInfo(
            order_id="test_001",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,  # Small order
            limit_price=55,  # High limit to ensure fill
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 10
        assert fill_result['vwap_price'] == 50  # Best ask price
        assert len(fill_result['consumed_levels']) == 1
        assert fill_result['consumed_levels'][0]['price'] == 50
        assert fill_result['consumed_levels'][0]['quantity'] == 10
    
    def test_small_sell_order_fills_at_best_bid(self, order_manager, sample_orderbook):
        """Small sell orders should fill at best bid without slippage."""
        order = OrderInfo(
            order_id="test_002",
            ticker="TEST-MARKET",
            side=OrderSide.SELL,
            contract_side=ContractSide.YES,
            quantity=15,  # Small order
            limit_price=40,  # Low limit to ensure fill
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 15
        assert fill_result['vwap_price'] == 48  # Best bid price
        assert len(fill_result['consumed_levels']) == 1
    
    def test_small_order_respects_limit_price(self, order_manager, sample_orderbook):
        """Small orders should still respect limit price."""
        order = OrderInfo(
            order_id="test_003",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=49,  # Below best ask of 50
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is False
        assert fill_result['filled_quantity'] == 0


class TestLargeOrderDepthConsumption:
    """Test that large orders walk the orderbook and experience slippage."""
    
    def test_large_buy_order_walks_orderbook(self, order_manager, sample_orderbook):
        """Large buy orders should consume multiple price levels."""
        order = OrderInfo(
            order_id="test_004",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=250,  # Large order requiring multiple levels
            limit_price=55,  # High limit to allow walking
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 250
        
        # Should consume: 80@50, 120@51, 50@52
        assert len(fill_result['consumed_levels']) == 3
        
        # Check consumed levels
        levels = fill_result['consumed_levels']
        assert levels[0] == {'price': 50, 'quantity': 80}
        assert levels[1] == {'price': 51, 'quantity': 120}
        assert levels[2] == {'price': 52, 'quantity': 50}
        
        # Calculate expected VWAP: (80*50 + 120*51 + 50*52) / 250 = 50.88, rounds to 51
        expected_vwap = round((80*50 + 120*51 + 50*52) / 250)
        assert fill_result['vwap_price'] == expected_vwap
    
    def test_large_sell_order_walks_orderbook(self, order_manager, sample_orderbook):
        """Large sell orders should consume multiple bid levels."""
        order = OrderInfo(
            order_id="test_005",
            ticker="TEST-MARKET",
            side=OrderSide.SELL,
            contract_side=ContractSide.YES,
            quantity=300,
            limit_price=45,  # Low limit to allow walking
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 300
        
        # Should consume: 100@48, 150@47, 50@46
        assert len(fill_result['consumed_levels']) == 3
        
        # Calculate expected VWAP: (100*48 + 150*47 + 50*46) / 300
        expected_vwap = int((100*48 + 150*47 + 50*46) / 300)
        assert fill_result['vwap_price'] == expected_vwap
    
    def test_order_exhausts_liquidity_partial_fill(self, order_manager, sample_orderbook):
        """Orders larger than available liquidity should partially fill."""
        order = OrderInfo(
            order_id="test_006",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=500,  # More than available liquidity (380)
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 380  # All available liquidity
        
        # Should consume all ask levels
        total_consumed = sum(level['quantity'] for level in fill_result['consumed_levels'])
        assert total_consumed == 380


class TestNOContractHandling:
    """Test depth consumption for NO contracts."""
    
    def test_buy_no_contracts_walks_derived_asks(self, order_manager, sample_orderbook):
        """Buying NO should walk derived NO asks (from YES bids)."""
        order = OrderInfo(
            order_id="test_007",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.NO,
            quantity=200,
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 200
        
        # Should consume NO asks: 52¢ (100), 53¢ (100 out of 150)
        # These are derived from YES bids: 47¢ (100), 46¢ (100 out of 150)
        assert len(fill_result['consumed_levels']) == 2
    
    def test_sell_no_contracts_walks_derived_bids(self, order_manager, sample_orderbook):
        """Selling NO should walk derived NO bids (from YES asks)."""
        order = OrderInfo(
            order_id="test_008",
            ticker="TEST-MARKET",
            side=OrderSide.SELL,
            contract_side=ContractSide.NO,
            quantity=150,
            limit_price=45,
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 150
        
        # Should consume NO bids: 50¢ (80), 49¢ (70 out of 120)
        # These are derived from YES asks: 49¢ (80), 50¢ (70 out of 120)
        assert len(fill_result['consumed_levels']) == 2


class TestConsumedLiquidityTracking:
    """Test consumed liquidity tracking and decay."""
    
    def test_consumed_liquidity_prevents_double_filling(self, order_manager, sample_orderbook):
        """Consumed liquidity should prevent immediate refilling."""
        # First order consumes liquidity
        order1 = OrderInfo(
            order_id="test_009",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=50,
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result1 = order_manager.calculate_fill_with_depth(order1, sample_orderbook)
        order_manager._track_consumed_liquidity(order1, fill_result1['consumed_levels'])
        
        # Second order should see reduced liquidity
        order2 = OrderInfo(
            order_id="test_010",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=50,
            limit_price=55,
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result2 = order_manager.calculate_fill_with_depth(order2, sample_orderbook)
        
        assert fill_result2['can_fill'] is True
        # Should get remaining 30 @ 50¢, then need to go to 51¢
        assert fill_result2['filled_quantity'] == 50
        # Should consume multiple price levels due to consumed liquidity
        assert len(fill_result2['consumed_levels']) > 1
        # Should include both 50¢ and 51¢ levels
        prices_consumed = [level['price'] for level in fill_result2['consumed_levels']]
        assert 50 in prices_consumed
        assert 51 in prices_consumed
        # VWAP should be at least 50 (due to rounding of 50.4)
        assert fill_result2['vwap_price'] >= 50
    
    def test_consumed_liquidity_expires(self, order_manager, sample_orderbook):
        """Consumed liquidity should expire after decay time."""
        # Create consumed liquidity with short decay time for testing
        consumed = ConsumedLiquidity(
            ticker="TEST-MARKET",
            side="yes_ask",
            price=50,
            consumed_quantity=50,
            timestamp=time.time() - 6.0,  # 6 seconds ago (> 5 second decay)
            decay_time=5.0
        )
        
        assert consumed.is_expired() is True
        assert consumed.get_available_quantity(80) == 80  # Full quantity available
    
    def test_cleanup_expired_liquidity(self, order_manager):
        """Expired liquidity should be cleaned up automatically."""
        # Add expired liquidity manually
        key = "TEST-MARKET_yes_ask_50"
        order_manager.consumed_liquidity[key] = ConsumedLiquidity(
            ticker="TEST-MARKET",
            side="yes_ask",
            price=50,
            consumed_quantity=50,
            timestamp=time.time() - 10.0,  # 10 seconds ago
            decay_time=5.0
        )
        
        # Cleanup should remove it
        order_manager._cleanup_expired_liquidity()
        assert key not in order_manager.consumed_liquidity


class TestLimitPriceRespect:
    """Test that limit prices are properly respected during depth consumption."""
    
    def test_buy_order_stops_at_limit_price(self, order_manager, sample_orderbook):
        """Buy orders should stop walking when reaching limit price."""
        order = OrderInfo(
            order_id="test_011",
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=300,
            limit_price=51,  # Should stop after 51¢ level
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 200  # 80@50 + 120@51
        
        # Should not consume 52¢ level
        prices = [level['price'] for level in fill_result['consumed_levels']]
        assert 52 not in prices
        assert max(prices) == 51
    
    def test_sell_order_stops_at_limit_price(self, order_manager, sample_orderbook):
        """Sell orders should stop walking when reaching limit price."""
        order = OrderInfo(
            order_id="test_012",
            ticker="TEST-MARKET",
            side=OrderSide.SELL,
            contract_side=ContractSide.YES,
            quantity=300,
            limit_price=47,  # Should stop after 47¢ level
            status=OrderStatus.PENDING,
            placed_at=time.time()
        )
        
        fill_result = order_manager.calculate_fill_with_depth(order, sample_orderbook)
        
        assert fill_result['can_fill'] is True
        assert fill_result['filled_quantity'] == 250  # 100@48 + 150@47
        
        # Should not consume 46¢ level
        prices = [level['price'] for level in fill_result['consumed_levels']]
        assert 46 not in prices
        assert min(prices) == 47


class TestIntegrationWithOrderExecution:
    """Test integration with actual order placement and execution."""
    
    @pytest.mark.asyncio
    async def test_small_order_immediate_fill(self, order_manager, sample_orderbook):
        """Small orders should fill immediately at best price."""
        order = await order_manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=15,
            orderbook=sample_orderbook,
            pricing_strategy="aggressive"
        )
        
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.fill_price == 50  # Best ask
        assert order.filled_quantity == 15
        assert order.remaining_quantity == 0
    
    @pytest.mark.asyncio
    async def test_large_order_immediate_fill_with_slippage(self, order_manager, sample_orderbook):
        """Large orders should fill immediately with VWAP pricing."""
        initial_cash = order_manager.cash_balance
        
        order = await order_manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=200,
            orderbook=sample_orderbook,
            pricing_strategy="aggressive"
        )
        
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.fill_price > 50  # VWAP should be higher than best ask
        assert order.filled_quantity == 200
        assert order.remaining_quantity == 0
        
        # Cash should be reduced by VWAP cost
        expected_cost = (order.fill_price / 100.0) * 200
        assert abs(order_manager.cash_balance - (initial_cash - expected_cost)) < 0.01
    
    @pytest.mark.asyncio
    async def test_large_order_partial_fill(self, order_manager, sample_orderbook):
        """Orders exceeding liquidity should partially fill."""
        order = await order_manager.place_order(
            ticker="TEST-MARKET",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=500,  # Exceeds available liquidity
            orderbook=sample_orderbook,
            pricing_strategy="aggressive"
        )
        
        assert order is not None
        assert order.status == OrderStatus.PENDING  # Still has remaining quantity
        assert order.filled_quantity == 380  # All available liquidity
        assert order.remaining_quantity == 120
        assert order.order_id in order_manager.open_orders


class TestConsumedLiquidityStats:
    """Test monitoring and debugging features."""
    
    def test_consumed_liquidity_stats(self, order_manager):
        """Test consumed liquidity statistics."""
        # Add some test consumed liquidity
        order_manager.consumed_liquidity["TEST1_yes_ask_50"] = ConsumedLiquidity(
            ticker="TEST1", side="yes_ask", price=50, consumed_quantity=100, timestamp=time.time()
        )
        order_manager.consumed_liquidity["TEST1_yes_bid_48"] = ConsumedLiquidity(
            ticker="TEST1", side="yes_bid", price=48, consumed_quantity=50, timestamp=time.time()
        )
        order_manager.consumed_liquidity["TEST2_no_ask_52"] = ConsumedLiquidity(
            ticker="TEST2", side="no_ask", price=52, consumed_quantity=75, timestamp=time.time()
        )
        
        stats = order_manager.get_consumed_liquidity_stats()
        
        assert stats['total_consumed_levels'] == 3
        assert "TEST1" in stats['by_ticker']
        assert "TEST2" in stats['by_ticker']
        assert stats['by_ticker']['TEST1']['levels'] == 2
        assert stats['by_ticker']['TEST1']['total_quantity'] == 150
        assert stats['by_ticker']['TEST2']['levels'] == 1
        assert stats['by_ticker']['TEST2']['total_quantity'] == 75
        
        assert "yes_ask" in stats['by_side']
        assert "yes_bid" in stats['by_side']
        assert "no_ask" in stats['by_side']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])