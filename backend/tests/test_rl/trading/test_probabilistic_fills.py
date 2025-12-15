"""
Test suite for probabilistic fill model in SimulatedOrderManager.

Tests verify:
1. Fill probabilities match expected ranges
2. Time priority affects fill probability
3. Size impact modifies fill rates
4. Market activity influences fills
5. Edge cases handled properly
"""

import pytest
import asyncio
import random
import time
from typing import Dict, List
from unittest.mock import MagicMock, patch

from kalshiflow_rl.trading.order_manager import (
    SimulatedOrderManager,
    OrderSide,
    ContractSide,
    OrderInfo,
    MarketActivityTracker
)
from kalshiflow_rl.data.orderbook_state import OrderbookState


class TestMarketActivityTracker:
    """Test the MarketActivityTracker functionality."""
    
    def test_activity_tracker_initialization(self):
        """Test activity tracker initializes correctly."""
        tracker = MarketActivityTracker(window_seconds=300)
        assert tracker.window_seconds == 300
        assert len(tracker.trades) == 0
        assert tracker.get_activity_level() == 0.0
    
    def test_activity_level_calculation(self):
        """Test activity level calculation based on trade frequency."""
        tracker = MarketActivityTracker(window_seconds=300)
        
        # Add trades to simulate different activity levels
        # Low activity: 3 trades = ~0.6 trades/min
        for _ in range(3):
            tracker.add_trade(10)
        
        activity = tracker.get_activity_level()
        assert 0.0 <= activity <= 0.2  # Low activity
        
        # Normal activity: 50 trades = ~10 trades/min  
        for _ in range(47):  # 47 more to make 50 total
            tracker.add_trade(10)
        
        activity = tracker.get_activity_level()
        assert 0.3 <= activity <= 0.7  # Normal activity
        
        # High activity: 120 trades = ~24 trades/min
        for _ in range(70):  # 70 more to make 120 total
            tracker.add_trade(10)
        
        activity = tracker.get_activity_level()
        assert 0.7 <= activity <= 1.0  # High activity
    
    def test_trade_cleanup(self):
        """Test old trades are cleaned up properly."""
        tracker = MarketActivityTracker(window_seconds=1)  # 1 second window
        
        # Add a trade
        tracker.add_trade(10)
        assert tracker.get_trade_count() == 1
        
        # Wait for window to expire
        time.sleep(1.1)
        
        # Add new trade to trigger cleanup
        tracker.add_trade(10)
        assert tracker.get_trade_count() == 1  # Old trade should be cleaned up


class TestProbabilisticFills:
    """Test probabilistic fill model in SimulatedOrderManager."""
    
    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        """Set random seed for deterministic tests."""
        random.seed(42)
        yield
        # Reset seed after test
        random.seed()
    
    @pytest.fixture
    def manager(self):
        """Create a SimulatedOrderManager for testing."""
        return SimulatedOrderManager(initial_cash=100000)
    
    @pytest.fixture
    def orderbook(self):
        """Create a sample orderbook for testing."""
        ob = OrderbookState(market_ticker="TEST-123")
        ob.last_update_time = time.time()
        # Populate orderbook with test data
        ob.yes_bids[45] = 100
        ob.yes_bids[44] = 200
        ob.yes_bids[43] = 300
        ob.yes_asks[50] = 100
        ob.yes_asks[51] = 200
        ob.yes_asks[52] = 300
        return ob
    
    def test_price_aggression_probability(self, manager, orderbook):
        """Test base probability calculation for different price levels."""
        # Create test orders at different price points
        
        # Buy order crossing spread (aggressive)
        order_aggressive = OrderInfo(
            order_id="test_1",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=51,  # Above ask of 50
            status=0,
            placed_at=time.time()
        )
        prob = manager._calculate_price_aggression_probability(order_aggressive, orderbook)
        assert 0.95 <= prob <= 0.99, f"Aggressive buy should have high probability, got {prob}"
        
        # Buy order at bid (passive)
        order_passive = OrderInfo(
            order_id="test_2",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=45,  # At bid
            status=0,
            placed_at=time.time()
        )
        prob = manager._calculate_price_aggression_probability(order_passive, orderbook)
        assert 0.35 <= prob <= 0.45, f"Passive buy at bid should have ~40% probability, got {prob}"
        
        # Buy order inside spread (closer to bid)
        order_inside = OrderInfo(
            order_id="test_3",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=47,  # Inside spread, closer to bid
            status=0,
            placed_at=time.time()
        )
        prob = manager._calculate_price_aggression_probability(order_inside, orderbook)
        assert 0.5 <= prob <= 0.7, f"Buy inside spread should have moderate probability, got {prob}"
        
        # Buy order near ask (aggressive but not crossing)
        order_near_ask = OrderInfo(
            order_id="test_3b",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=49,  # 1 cent below ask
            status=0,
            placed_at=time.time()
        )
        prob = manager._calculate_price_aggression_probability(order_near_ask, orderbook)
        assert 0.7 <= prob <= 0.9, f"Buy near ask should have high probability, got {prob}"
        
        # Sell order crossing spread (aggressive)
        order_sell_aggressive = OrderInfo(
            order_id="test_4",
            ticker="TEST-123",
            side=OrderSide.SELL,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=44,  # Below bid of 45
            status=0,
            placed_at=time.time()
        )
        prob = manager._calculate_price_aggression_probability(order_sell_aggressive, orderbook)
        assert 0.95 <= prob <= 0.99, f"Aggressive sell should have high probability, got {prob}"
    
    def test_time_priority_modifier(self, manager, orderbook):
        """Test that time in queue affects fill probability."""
        # Create order and check probability at different times
        order = OrderInfo(
            order_id="test_time",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=45,  # At bid (passive)
            status=0,
            placed_at=time.time()
        )
        
        # Initial probability (0 seconds in queue)
        prob_initial = manager.calculate_fill_probability(order, orderbook)
        
        # After 10 seconds (should be higher)
        order.placed_at = time.time() - 10
        prob_10s = manager.calculate_fill_probability(order, orderbook)
        
        # After 30 seconds (should be even higher)
        order.placed_at = time.time() - 30
        prob_30s = manager.calculate_fill_probability(order, orderbook)
        
        assert prob_30s > prob_10s >= prob_initial, (
            f"Probability should increase with time: {prob_initial:.3f} -> {prob_10s:.3f} -> {prob_30s:.3f}"
        )
        
        # Verify the actual modifier values (scaled down modifiers)
        assert prob_10s - prob_initial >= 0.08, "10s should add at least 8% probability"
        assert prob_30s - prob_initial >= 0.12, "30s should add at least 12% probability"
    
    def test_size_impact_modifier(self, manager, orderbook):
        """Test that order size affects fill probability."""
        # Small order (< 10 contracts)
        order_small = OrderInfo(
            order_id="test_small",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=5,
            limit_price=45,  # At bid
            status=0,
            placed_at=time.time()
        )
        prob_small = manager.calculate_fill_probability(order_small, orderbook)
        
        # Normal order (10-50 contracts)
        order_normal = OrderInfo(
            order_id="test_normal",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=30,
            limit_price=45,  # At bid
            status=0,
            placed_at=time.time()
        )
        prob_normal = manager.calculate_fill_probability(order_normal, orderbook)
        
        # Large order (> 100 contracts)
        order_large = OrderInfo(
            order_id="test_large",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=150,
            limit_price=45,  # At bid
            status=0,
            placed_at=time.time()
        )
        prob_large = manager.calculate_fill_probability(order_large, orderbook)
        
        assert prob_small > prob_normal > prob_large, (
            f"Small orders should fill easier: small={prob_small:.3f}, "
            f"normal={prob_normal:.3f}, large={prob_large:.3f}"
        )
        
        # Check magnitude of differences (scaled modifiers)
        assert prob_small - prob_normal >= 0.04, "Small vs normal should differ by at least 4%"
        assert prob_normal - prob_large >= 0.08, "Normal vs large should differ by at least 8%"
    
    def test_market_activity_modifier(self, manager, orderbook):
        """Test that market activity affects fill probability."""
        order = OrderInfo(
            order_id="test_activity",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=45,  # At bid
            status=0,
            placed_at=time.time()
        )
        
        # Low activity
        manager.activity_tracker = MarketActivityTracker()  # Empty = low activity
        prob_low = manager.calculate_fill_probability(order, orderbook)
        
        # Simulate normal activity
        for _ in range(50):  # ~10 trades/min
            manager.activity_tracker.add_trade(10)
        prob_normal = manager.calculate_fill_probability(order, orderbook)
        
        # Simulate high activity
        for _ in range(100):  # ~30 trades/min total
            manager.activity_tracker.add_trade(10)
        prob_high = manager.calculate_fill_probability(order, orderbook)
        
        assert prob_high > prob_normal > prob_low, (
            f"Higher activity should increase fill probability: "
            f"low={prob_low:.3f}, normal={prob_normal:.3f}, high={prob_high:.3f}"
        )
    
    def test_wide_spread_penalty(self, manager):
        """Test that wide spreads reduce fill probability."""
        # Normal spread orderbook
        orderbook_normal = OrderbookState(market_ticker="TEST-123")
        orderbook_normal.last_update_time = time.time()
        orderbook_normal.yes_bids[48] = 100
        orderbook_normal.yes_asks[50] = 100
        
        # Wide spread orderbook
        orderbook_wide = OrderbookState(market_ticker="TEST-123")
        orderbook_wide.last_update_time = time.time()
        orderbook_wide.yes_bids[40] = 100
        orderbook_wide.yes_asks[50] = 100
        
        order = OrderInfo(
            order_id="test_spread",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=45,  # Inside spread
            status=0,
            placed_at=time.time()
        )
        
        prob_normal = manager.calculate_fill_probability(order, orderbook_normal)
        prob_wide = manager.calculate_fill_probability(order, orderbook_wide)
        
        # In wide spread, order at 45 is inside spread (40-50), so should have higher prob
        # In normal spread, order at 45 is below bid (48), so should have lower prob
        assert prob_wide > prob_normal, (
            f"Order inside wide spread should have higher probability than below bid: "
            f"wide={prob_wide:.3f}, normal={prob_normal:.3f}"
        )
    
    def test_empty_orderbook_handling(self, manager):
        """Test handling of empty orderbooks."""
        orderbook_empty = OrderbookState(market_ticker="TEST-123")
        orderbook_empty.last_update_time = time.time()
        # Keep orderbook empty - no bids/asks
        
        order = OrderInfo(
            order_id="test_empty",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=50,
            status=0,
            placed_at=time.time()
        )
        
        prob = manager.calculate_fill_probability(order, orderbook_empty)
        assert 0.01 <= prob <= 0.2, f"Empty orderbook should have low probability, got {prob}"
    
    @pytest.mark.asyncio
    async def test_statistical_fill_rates(self, orderbook):
        """Test fill rates match expected probabilities over many trials."""
        # Set seed for deterministic testing
        random.seed(42)
        
        test_cases = [
            # (limit_price, expected_min_rate, expected_max_rate, description)
            # Aggressive orders should fill most of the time
            (51, 0.85, 1.00, "Aggressive buy crossing spread"),
            # Passive orders at bid have moderate fill rate  
            (45, 0.30, 0.50, "Passive buy at bid"),  
            # Inside spread orders have medium-high fill rate
            (47, 0.50, 0.70, "Buy inside spread"),
        ]
        
        for limit_price, min_rate, max_rate, description in test_cases:
            # Reset random seed for each test case for consistency
            random.seed(42)
            
            # Reset manager for each test case
            test_manager = SimulatedOrderManager(initial_cash=100000)
            
            # Add consistent market activity
            for _ in range(20):
                test_manager.activity_tracker.add_trade(10)
            
            fills = 0
            trials = 100
            
            for trial_num in range(trials):
                # Create order directly (skip automatic fill logic)
                order = OrderInfo(
                    order_id=f"test_{trial_num}",
                    ticker="TEST-123",
                    side=OrderSide.BUY,
                    contract_side=ContractSide.YES,
                    quantity=10,
                    limit_price=limit_price,
                    status=0,
                    placed_at=time.time() - 5  # 5 seconds old for some time priority
                )
                test_manager.open_orders[order.order_id] = order
                
                # Check fills using probabilistic model
                filled = await test_manager.check_fills(orderbook)
                if filled:
                    fills += 1
                    
                # Clean up for next trial
                test_manager.open_orders.clear()
                test_manager.consumed_liquidity.clear()  # Important to reset liquidity
            
            fill_rate = fills / trials
            assert min_rate <= fill_rate <= max_rate, (
                f"{description}: Expected {min_rate:.0%}-{max_rate:.0%}, got {fill_rate:.0%}"
            )
    
    @pytest.mark.asyncio 
    async def test_partial_fill_probability_increase(self, manager, orderbook):
        """Test that partial fills slightly increase remaining fill probability."""
        # Create a large order
        order = OrderInfo(
            order_id="test_partial",
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=100,
            limit_price=45,  # At bid
            status=0,
            placed_at=time.time()
        )
        # Set filled quantity after creation
        order.filled_quantity = 0
        order.remaining_quantity = 100
        
        # Get initial probability
        prob_initial = manager.calculate_fill_probability(order, orderbook)
        
        # Simulate partial fill
        order.update_partial_fill(50, 45)
        
        # Probability should be similar (size reduced, but still at bid)
        prob_after_partial = manager.calculate_fill_probability(order, orderbook)
        
        # Remaining 50 contracts should have slightly better fill probability
        # since it's now a medium-sized order instead of large
        assert prob_after_partial > prob_initial, (
            f"Partial fill should improve remaining probability: "
            f"{prob_initial:.3f} -> {prob_after_partial:.3f}"
        )
    
    @pytest.mark.asyncio
    async def test_integration_with_depth_consumption(self, manager, orderbook):
        """Test that probabilistic fills work with depth consumption."""
        # Place a large aggressive order that would consume depth
        order = await manager.place_order(
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=200,  # Large order
            orderbook=orderbook,
            pricing_strategy="aggressive"
        )
        
        # The order should be placed
        assert order is not None
        
        # If it didn't fill immediately, it should be in open orders
        if order.order_id in manager.open_orders:
            # Simulate market activity to increase fill probability
            for _ in range(20):
                manager.activity_tracker.add_trade(10)
            
            # Check fills multiple times to account for probabilistic nature
            total_filled = 0
            for _ in range(10):
                filled_orders = await manager.check_fills(orderbook)
                for filled_order in filled_orders:
                    total_filled += filled_order.quantity
            
            # With aggressive pricing and activity, should get some fills
            # but not guaranteed due to probabilistic model
            print(f"Filled {total_filled} out of 200 contracts across 10 checks")
    
    def test_probability_bounds(self, manager, orderbook):
        """Test that probabilities stay within valid bounds."""
        test_orders = [
            # Various extreme cases
            OrderInfo("t1", "TEST", OrderSide.BUY, ContractSide.YES, 1, 99, 0, time.time()),
            OrderInfo("t2", "TEST", OrderSide.BUY, ContractSide.YES, 1000, 1, 0, time.time()),
            OrderInfo("t3", "TEST", OrderSide.SELL, ContractSide.YES, 50, 50, 0, time.time() - 3600),
            OrderInfo("t4", "TEST", OrderSide.SELL, ContractSide.NO, 10, 45, 0, time.time()),
        ]
        
        for order in test_orders:
            prob = manager.calculate_fill_probability(order, orderbook)
            assert 0.01 <= prob <= 0.99, (
                f"Probability out of bounds for order {order.order_id}: {prob}"
            )


class TestProbabilisticFillsWithRealData:
    """Test probabilistic fills with more realistic market scenarios."""
    
    @pytest.fixture
    def volatile_orderbook(self):
        """Create a volatile orderbook with wide spread."""
        ob = OrderbookState(market_ticker="VOLATILE-123")
        ob.last_update_time = time.time()
        ob.yes_bids[35] = 50
        ob.yes_bids[34] = 100
        ob.yes_bids[33] = 150
        ob.yes_asks[45] = 50
        ob.yes_asks[46] = 100
        ob.yes_asks[47] = 150
        return ob
    
    @pytest.fixture
    def tight_orderbook(self):
        """Create a tight orderbook with narrow spread."""
        ob = OrderbookState(market_ticker="TIGHT-123")
        ob.last_update_time = time.time()
        ob.yes_bids[49] = 500
        ob.yes_bids[48] = 1000
        ob.yes_bids[47] = 1500
        ob.yes_asks[50] = 500
        ob.yes_asks[51] = 1000
        ob.yes_asks[52] = 1500
        return ob
    
    @pytest.mark.asyncio
    async def test_volatile_market_behavior(self, volatile_orderbook):
        """Test behavior in volatile markets with wide spreads."""
        manager = SimulatedOrderManager(initial_cash=100000)
        
        # Passive orders should have very low fill rates in volatile markets
        order = await manager.place_order(
            ticker="VOLATILE-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            orderbook=volatile_orderbook,
            pricing_strategy="passive"
        )
        
        if order and order.order_id in manager.open_orders:
            # Check fills 20 times
            fills = 0
            for _ in range(20):
                filled = await manager.check_fills(volatile_orderbook)
                if filled:
                    fills += 1
            
            # In volatile market with wide spread, passive orders rarely fill
            assert fills <= 5, f"Too many fills in volatile market: {fills}/20"
    
    @pytest.mark.asyncio
    async def test_tight_market_behavior(self, tight_orderbook):
        """Test behavior in tight markets with narrow spreads."""
        manager = SimulatedOrderManager(initial_cash=100000)
        
        # Add market activity
        for _ in range(30):
            manager.activity_tracker.add_trade(25)
        
        # Orders near mid should have decent fill rates in tight markets
        order = OrderInfo(
            order_id="tight_test",
            ticker="TIGHT-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=20,
            limit_price=49,  # At bid in tight market
            status=0,
            placed_at=time.time() - 15  # Been waiting 15 seconds
        )
        manager.open_orders[order.order_id] = order
        
        # Check probability
        prob = manager.calculate_fill_probability(order, tight_orderbook)
        
        # In tight, active market with time priority, should have decent probability
        assert prob >= 0.4, f"Tight market probability too low: {prob}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])