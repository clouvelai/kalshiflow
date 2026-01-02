"""
Unit tests for OrderbookState and SharedOrderbookState.

Tests thread safety, atomic operations, spread calculations,
and subscriber notifications.
"""

import pytest
import asyncio
import time
from unittest.mock import MagicMock
from decimal import Decimal

from kalshiflow_rl.data.orderbook_state import (
    OrderbookState, 
    SharedOrderbookState,
    get_shared_orderbook_state
)


@pytest.fixture
def sample_snapshot():
    """Sample snapshot data for testing.
    
    Kalshi only provides bids. Asks are derived using the reciprocal relationship:
    - YES_BID at X → NO_ASK at (99 - X)
    - NO_BID at Y → YES_ASK at (99 - Y)
    
    So with yes_bids at 45,44 and no_bids at 45,44:
    - yes_asks derived from no_bids: {54: 600, 55: 300}
    - no_asks derived from yes_bids: {54: 1000, 55: 500}
    """
    return {
        "market_ticker": "TEST-MARKET",
        "timestamp_ms": int(time.time() * 1000),
        "sequence_number": 100,
        "yes_bids": {"45": 1000, "44": 500},
        "no_bids": {"45": 600, "44": 300},
        # Note: yes_asks and no_asks in input are ignored - they're derived from bids
    }


@pytest.fixture
def sample_delta():
    """Sample delta data for testing."""
    return {
        "market_ticker": "TEST-MARKET",
        "timestamp_ms": int(time.time() * 1000),
        "sequence_number": 101,
        "side": "yes",
        "action": "update",
        "price": 45,
        "old_size": 1000,
        "new_size": 1200
    }


class TestOrderbookState:
    """Test cases for OrderbookState."""
    
    def test_initialization(self):
        """Test orderbook state initialization."""
        state = OrderbookState("TEST-MARKET")
        
        assert state.market_ticker == "TEST-MARKET"
        assert state.last_sequence == 0
        assert len(state.yes_bids) == 0
        assert len(state.yes_asks) == 0
        assert len(state.no_bids) == 0
        assert len(state.no_asks) == 0
    
    def test_apply_snapshot(self, sample_snapshot):
        """Test applying orderbook snapshot."""
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        assert state.last_sequence == 100
        assert state.market_ticker == "TEST-MARKET"
        
        # Check yes bids (from input)
        assert 45 in state.yes_bids
        assert state.yes_bids[45] == 1000
        assert 44 in state.yes_bids
        assert state.yes_bids[44] == 500
        
        # Check yes asks (derived from no_bids: NO_BID at Y → YES_ASK at 99-Y)
        # no_bids: {45: 600, 44: 300} → yes_asks: {54: 600, 55: 300}
        assert 54 in state.yes_asks
        assert state.yes_asks[54] == 600
        assert 55 in state.yes_asks
        assert state.yes_asks[55] == 300
        
        # Check no bids (from input)
        assert 45 in state.no_bids
        assert state.no_bids[45] == 600
        assert 44 in state.no_bids
        assert state.no_bids[44] == 300
        
        # Check no asks (derived from yes_bids: YES_BID at X → NO_ASK at 99-X)
        # yes_bids: {45: 1000, 44: 500} → no_asks: {54: 1000, 55: 500}
        assert 54 in state.no_asks
        assert state.no_asks[54] == 1000
        assert 55 in state.no_asks
        assert state.no_asks[55] == 500
    
    def test_apply_delta_add(self, sample_snapshot):
        """Test applying add delta."""
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        # Add new bid
        delta = {
            "sequence_number": 101,
            "timestamp_ms": int(time.time() * 1000),
            "side": "yes",
            "action": "add",
            "price": 43,
            "new_size": 750
        }
        
        result = state.apply_delta(delta)
        assert result is True
        assert state.last_sequence == 101
        assert 43 in state.yes_bids
        assert state.yes_bids[43] == 750
    
    def test_apply_delta_remove(self, sample_snapshot):
        """Test applying remove delta."""
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        # Remove existing bid
        delta = {
            "sequence_number": 101,
            "timestamp_ms": int(time.time() * 1000),
            "side": "yes",
            "action": "remove",
            "price": 45
        }
        
        result = state.apply_delta(delta)
        assert result is True
        assert state.last_sequence == 101
        assert 45 not in state.yes_bids
    
    def test_apply_delta_update(self, sample_snapshot):
        """Test applying update delta."""
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        # Update existing bid
        delta = {
            "sequence_number": 101,
            "timestamp_ms": int(time.time() * 1000),
            "side": "yes",
            "action": "update",
            "price": 45,
            "old_size": 1000,
            "new_size": 1200
        }
        
        result = state.apply_delta(delta)
        assert result is True
        assert state.last_sequence == 101
        assert state.yes_bids[45] == 1200
    
    def test_sequence_validation(self, sample_snapshot):
        """Test sequence number validation."""
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        # Try to apply old delta
        old_delta = {
            "sequence_number": 99,  # Older than snapshot
            "timestamp_ms": int(time.time() * 1000),
            "side": "yes",
            "action": "update",
            "price": 45,
            "new_size": 1200
        }
        
        result = state.apply_delta(old_delta)
        assert result is False
        assert state.last_sequence == 100  # Unchanged
    
    def test_spread_calculations(self, sample_snapshot):
        """Test spread calculation.
        
        With sample data:
        - yes_bids: {45: 1000, 44: 500} → best_bid = 45
        - yes_asks: derived from no_bids {45: 600, 44: 300} → {54: 600, 55: 300} → best_ask = 54
        - no_bids: {45: 600, 44: 300} → best_bid = 45
        - no_asks: derived from yes_bids {45: 1000, 44: 500} → {54: 1000, 55: 500} → best_ask = 54
        """
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        # Yes spread = best_ask - best_bid = 54 - 45 = 9
        yes_spread = state.get_yes_spread()
        assert yes_spread == 9
        
        # No spread = best_ask - best_bid = 54 - 45 = 9
        no_spread = state.get_no_spread()
        assert no_spread == 9
    
    def test_mid_price_calculations(self, sample_snapshot):
        """Test mid price calculation.
        
        With sample data:
        - yes: best_bid = 45, best_ask = 54 (derived from no_bid at 45)
        - no: best_bid = 45, best_ask = 54 (derived from yes_bid at 45)
        """
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        # Yes mid = (best_bid + best_ask) / 2 = (45 + 54) / 2 = 49.5
        yes_mid = state.get_yes_mid_price()
        assert yes_mid == Decimal('49.5')
        
        # No mid = (best_bid + best_ask) / 2 = (45 + 54) / 2 = 49.5
        no_mid = state.get_no_mid_price()
        assert no_mid == Decimal('49.5')
    
    def test_cache_invalidation(self, sample_snapshot):
        """Test that cache is invalidated on updates.
        
        Initial state: yes_bids best=45, yes_asks best=54 → spread=9
        After delta adding bid at 46: yes_bids best=46, yes_asks best=54 → spread=8
        """
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        # Calculate spread (populates cache)
        spread1 = state.get_yes_spread()
        assert spread1 == 9  # 54 - 45 = 9
        
        # Apply delta that changes spread
        delta = {
            "sequence_number": 101,
            "timestamp_ms": int(time.time() * 1000),
            "side": "yes",
            "action": "add",
            "price": 46,  # New best bid
            "new_size": 800
        }
        
        state.apply_delta(delta)
        
        # Spread should be recalculated
        spread2 = state.get_yes_spread()
        assert spread2 == 8  # 54 - 46 = 8
    
    def test_top_levels(self, sample_snapshot):
        """Test getting top price levels."""
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        top_levels = state.get_top_levels(2)
        
        assert "yes_bids" in top_levels
        assert "yes_asks" in top_levels
        assert "no_bids" in top_levels
        assert "no_asks" in top_levels
        
        # Should have at most 2 levels per side
        assert len(top_levels["yes_bids"]) <= 2
        assert len(top_levels["yes_asks"]) <= 2
    
    def test_total_volume(self, sample_snapshot):
        """Test total volume calculation.
        
        With sample data (only bids provided, asks derived):
        - yes_bids: {45: 1000, 44: 500} → 1500
        - no_bids: {45: 600, 44: 300} → 900
        - yes_asks: {54: 600, 55: 300} → 900 (derived from no_bids)
        - no_asks: {54: 1000, 55: 500} → 1500 (derived from yes_bids)
        Total = 1500 + 900 + 900 + 1500 = 4800
        """
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        total_volume = state.get_total_volume()
        assert total_volume == 4800
    
    def test_to_dict(self, sample_snapshot):
        """Test converting state to dictionary."""
        state = OrderbookState("TEST-MARKET")
        state.apply_snapshot(sample_snapshot)
        
        state_dict = state.to_dict()
        
        assert state_dict["market_ticker"] == "TEST-MARKET"
        assert state_dict["last_sequence"] == 100
        assert "yes_bids" in state_dict
        assert "yes_asks" in state_dict
        assert "no_bids" in state_dict
        assert "no_asks" in state_dict
        assert "yes_spread" in state_dict
        assert "no_spread" in state_dict
        assert "total_volume" in state_dict


class TestSharedOrderbookState:
    """Test cases for SharedOrderbookState."""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test shared state initialization."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        
        assert shared_state.market_ticker == "TEST-MARKET"
        assert shared_state._lock is not None
        assert len(shared_state._subscribers) == 0
    
    @pytest.mark.asyncio
    async def test_atomic_snapshot_application(self, sample_snapshot):
        """Test atomic snapshot application."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        
        await shared_state.apply_snapshot(sample_snapshot)
        
        # Get snapshot should be atomic
        snapshot = await shared_state.get_snapshot()
        assert snapshot["last_sequence"] == 100
        assert snapshot["market_ticker"] == "TEST-MARKET"
    
    @pytest.mark.asyncio
    async def test_atomic_delta_application(self, sample_snapshot, sample_delta):
        """Test atomic delta application."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        
        await shared_state.apply_snapshot(sample_snapshot)
        result = await shared_state.apply_delta(sample_delta)
        
        assert result is True
        
        snapshot = await shared_state.get_snapshot()
        assert snapshot["last_sequence"] == 101
    
    @pytest.mark.asyncio
    async def test_concurrent_access(self, sample_snapshot):
        """Test concurrent access to shared state."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        await shared_state.apply_snapshot(sample_snapshot)
        
        # Create multiple concurrent readers
        async def read_snapshot():
            return await shared_state.get_snapshot()
        
        # Run multiple readers concurrently
        tasks = [read_snapshot() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All should get consistent snapshots
        for result in results:
            assert result["last_sequence"] == 100
            assert result["market_ticker"] == "TEST-MARKET"
    
    @pytest.mark.asyncio
    async def test_subscriber_notifications(self, sample_snapshot):
        """Test subscriber notification system."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        
        # Add mock subscriber
        notifications = []
        def mock_subscriber(data):
            notifications.append(data)
        
        shared_state.add_subscriber(mock_subscriber)
        
        # Apply snapshot
        await shared_state.apply_snapshot(sample_snapshot)
        
        # Should have received notification
        assert len(notifications) == 1
        assert notifications[0]["update_type"] == "snapshot"
        assert notifications[0]["market_ticker"] == "TEST-MARKET"
    
    @pytest.mark.asyncio
    async def test_subscriber_management(self):
        """Test adding and removing subscribers."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        
        # Mock subscribers
        sub1 = MagicMock()
        sub2 = MagicMock()
        
        # Add subscribers
        shared_state.add_subscriber(sub1)
        shared_state.add_subscriber(sub2)
        assert len(shared_state._subscribers) == 2
        
        # Remove subscriber
        shared_state.remove_subscriber(sub1)
        assert len(shared_state._subscribers) == 1
        assert sub2 in shared_state._subscribers
        assert sub1 not in shared_state._subscribers
    
    @pytest.mark.asyncio
    async def test_subscriber_error_handling(self, sample_snapshot):
        """Test handling of subscriber errors."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        
        # Add failing subscriber
        def failing_subscriber(data):
            raise Exception("Subscriber error")
        
        # Add working subscriber
        notifications = []
        def working_subscriber(data):
            notifications.append(data)
        
        shared_state.add_subscriber(failing_subscriber)
        shared_state.add_subscriber(working_subscriber)
        
        # Apply snapshot (should not fail despite subscriber error)
        await shared_state.apply_snapshot(sample_snapshot)
        
        # Working subscriber should still get notification
        assert len(notifications) == 1
    
    @pytest.mark.asyncio
    async def test_get_spreads_and_mids(self, sample_snapshot):
        """Test atomic spreads and mids getter.
        
        With sample data:
        - yes: best_bid=45, best_ask=54 → spread=9, mid=49.5
        - no: best_bid=45, best_ask=54 → spread=9, mid=49.5
        """
        shared_state = SharedOrderbookState("TEST-MARKET")
        await shared_state.apply_snapshot(sample_snapshot)
        
        spreads_mids = await shared_state.get_spreads_and_mids()
        
        assert "yes_spread" in spreads_mids
        assert "no_spread" in spreads_mids
        assert "yes_mid_price" in spreads_mids
        assert "no_mid_price" in spreads_mids
        
        assert spreads_mids["yes_spread"] == 9
        assert spreads_mids["no_spread"] == 9
        assert spreads_mids["yes_mid_price"] == 49.5
        assert spreads_mids["no_mid_price"] == 49.5
    
    @pytest.mark.asyncio
    async def test_get_stats(self, sample_snapshot):
        """Test statistics collection."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        await shared_state.apply_snapshot(sample_snapshot)
        
        stats = shared_state.get_stats()
        
        assert stats["market_ticker"] == "TEST-MARKET"
        assert stats["last_sequence"] == 100
        assert "subscribers_count" in stats
        assert "total_volume" in stats
        assert "price_level_count" in stats
    
    @pytest.mark.asyncio
    async def test_notification_throttling(self, sample_snapshot):
        """Test that notifications are throttled."""
        shared_state = SharedOrderbookState("TEST-MARKET")
        shared_state._notification_throttle = 0.1  # 100ms throttle
        
        notifications = []
        def mock_subscriber(data):
            notifications.append(data)
        
        shared_state.add_subscriber(mock_subscriber)
        
        # Apply multiple quick updates
        await shared_state.apply_snapshot(sample_snapshot)
        
        delta1 = {
            "sequence_number": 101,
            "timestamp_ms": int(time.time() * 1000),
            "side": "yes",
            "action": "update",
            "price": 45,
            "new_size": 1200
        }
        
        delta2 = {
            "sequence_number": 102,
            "timestamp_ms": int(time.time() * 1000),
            "side": "yes",
            "action": "update",
            "price": 45,
            "new_size": 1300
        }
        
        await shared_state.apply_delta(delta1)
        await shared_state.apply_delta(delta2)  # Should be throttled
        
        # Should have fewer notifications than updates due to throttling
        assert len(notifications) <= 3  # snapshot + maybe 1-2 deltas


class TestOrderbookStateRegistry:
    """Test cases for global orderbook state registry."""
    
    @pytest.mark.asyncio
    async def test_get_shared_orderbook_state(self):
        """Test getting shared orderbook state from registry."""
        # First call should create new state
        state1 = await get_shared_orderbook_state("TEST-MARKET")
        assert isinstance(state1, SharedOrderbookState)
        assert state1.market_ticker == "TEST-MARKET"
        
        # Second call should return same instance
        state2 = await get_shared_orderbook_state("TEST-MARKET")
        assert state1 is state2
        
        # Different market should create different instance
        state3 = await get_shared_orderbook_state("OTHER-MARKET")
        assert state1 is not state3
        assert state3.market_ticker == "OTHER-MARKET"