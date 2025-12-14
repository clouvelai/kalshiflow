"""
Tests for orderbook parsing from Kalshi WebSocket format.

Ensures that:
1. Snapshots are parsed correctly from array format [[price, qty], ...]
2. Deltas are parsed with correct integer types
3. Bid/ask separation works based on price threshold (≤50 = bid, >50 = ask)
4. Integer keys are used consistently throughout
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any

from kalshiflow_rl.data.orderbook_client import OrderbookClient
from kalshiflow_rl.data.orderbook_state import SharedOrderbookState


class TestOrderbookParsing:
    """Test orderbook message parsing from Kalshi WebSocket."""
    
    @pytest.fixture
    def kalshi_snapshot_message(self) -> Dict[str, Any]:
        """Example snapshot message as sent by Kalshi WebSocket."""
        return {
            "type": "orderbook_snapshot",
            "seq": 12345,
            "msg": {
                "market_ticker": "TEST-MARKET",
                "yes": [
                    [45, 100],  # Bid at 45 cents
                    [44, 200],  # Bid at 44 cents
                    [55, 150],  # Ask at 55 cents
                    [56, 300]   # Ask at 56 cents
                ],
                "no": [
                    [40, 250],  # Bid at 40 cents
                    [39, 175],  # Bid at 39 cents
                    [60, 225],  # Ask at 60 cents
                    [61, 400]   # Ask at 61 cents
                ]
            }
        }
    
    @pytest.fixture
    def kalshi_delta_message(self) -> Dict[str, Any]:
        """Example delta message as sent by Kalshi WebSocket."""
        return {
            "type": "orderbook_delta",
            "seq": 12346,
            "msg": {
                "market_ticker": "TEST-MARKET",
                "side": "yes",
                "price": 46,
                "delta": 50
            }
        }
    
    @pytest.fixture
    def orderbook_client(self):
        """Create an orderbook client for testing."""
        client = OrderbookClient(market_tickers=["TEST-MARKET"])
        # Mock the WebSocket connection
        client._websocket = MagicMock()
        client._running = True
        return client
    
    @pytest.mark.asyncio
    async def test_snapshot_parsing(self, orderbook_client, kalshi_snapshot_message):
        """Test that snapshot messages are parsed correctly."""
        # Mock the orderbook state
        mock_state = AsyncMock(spec=SharedOrderbookState)
        mock_state.apply_snapshot = AsyncMock(return_value=None)
        orderbook_client._orderbook_states["TEST-MARKET"] = mock_state
        
        # Mock the write queue
        with patch('kalshiflow_rl.data.orderbook_client.write_queue') as mock_queue:
            mock_queue.enqueue_snapshot = AsyncMock()
            
            # Process the snapshot
            await orderbook_client._process_snapshot(kalshi_snapshot_message)
            
            # Verify the snapshot was parsed correctly
            assert mock_queue.enqueue_snapshot.called
            snapshot_data = mock_queue.enqueue_snapshot.call_args[0][0]
            
            # Check that data was parsed correctly with integer keys
            assert snapshot_data["market_ticker"] == "TEST-MARKET"
            assert snapshot_data["sequence_number"] == 12345
            
            # Verify bid/ask separation for yes side
            assert snapshot_data["yes_bids"] == {45: 100, 44: 200}  # Prices ≤ 50
            assert snapshot_data["yes_asks"] == {55: 150, 56: 300}  # Prices > 50
            
            # Verify bid/ask separation for no side
            assert snapshot_data["no_bids"] == {40: 250, 39: 175}  # Prices ≤ 50
            assert snapshot_data["no_asks"] == {60: 225, 61: 400}  # Prices > 50
            
            # Verify all keys are integers, not strings
            for book in ["yes_bids", "yes_asks", "no_bids", "no_asks"]:
                for key in snapshot_data[book].keys():
                    assert isinstance(key, int), f"Key {key} in {book} should be int"
    
    @pytest.mark.asyncio
    async def test_delta_parsing(self, orderbook_client, kalshi_delta_message):
        """Test that delta messages are parsed correctly."""
        from kalshiflow_rl.data.orderbook_state import OrderbookState, SharedOrderbookState
        
        # Create real shared state (not a mock) because _process_delta accesses _state
        shared_state = SharedOrderbookState("TEST-MARKET")
        # Initialize with a snapshot so we have initial state
        await shared_state.apply_snapshot({
            "market_ticker": "TEST-MARKET",
            "timestamp_ms": 1234567890,
            "sequence_number": 12340,
            "yes_bids": {45: 100},
            "no_bids": {40: 200},
        })
        
        orderbook_client._orderbook_states["TEST-MARKET"] = shared_state
        
        # Mock the write queue
        with patch('kalshiflow_rl.data.orderbook_client.write_queue') as mock_queue:
            mock_queue.enqueue_delta = AsyncMock(return_value=True)
            
            # Process the delta
            await orderbook_client._process_delta(kalshi_delta_message)
            
            # Verify the delta was parsed correctly
            assert mock_queue.enqueue_delta.called
            delta_data = mock_queue.enqueue_delta.call_args[0][0]
            
            assert delta_data["market_ticker"] == "TEST-MARKET"
            assert delta_data["sequence_number"] == 12346  # From outer message seq field
            assert delta_data["side"] == "yes"
            assert delta_data["price"] == 46  # Should be int
            # old_size is 0 because price 46 didn't exist before
            assert delta_data["old_size"] == 0
            # new_size is delta added to old_size: 0 + 50 = 50
            assert delta_data["new_size"] == 50
            assert delta_data["action"] == "add"  # old_size=0, new_size>0
            
            # Verify price is integer
            assert isinstance(delta_data["price"], int)
            assert isinstance(delta_data["old_size"], int)
            assert isinstance(delta_data["new_size"], int)
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, orderbook_client):
        """Test edge cases in orderbook parsing."""
        # Test snapshot with price exactly at 50 (should be bid)
        edge_snapshot = {
            "type": "orderbook_snapshot",
            "seq": 99999,
            "msg": {
                "market_ticker": "TEST-MARKET",
                "yes": [[50, 100], [51, 200]],
                "no": []
            }
        }
        
        mock_state = AsyncMock(spec=SharedOrderbookState)
        orderbook_client._orderbook_states["TEST-MARKET"] = mock_state
        
        with patch('kalshiflow_rl.data.orderbook_client.write_queue') as mock_queue:
            mock_queue.enqueue_snapshot = AsyncMock()
            
            await orderbook_client._process_snapshot(edge_snapshot)
            
            snapshot_data = mock_queue.enqueue_snapshot.call_args[0][0]
            assert 50 in snapshot_data["yes_bids"]  # 50 should be a bid
            assert 51 in snapshot_data["yes_asks"]  # 51 should be an ask
    
    @pytest.mark.asyncio
    async def test_orderbook_state_integration(self):
        """Test that OrderbookState works with integer keys.
        
        Note: Kalshi only provides bids. Asks are derived using reciprocal relationship:
        - YES_BID at X → NO_ASK at (99 - X)
        - NO_BID at Y → YES_ASK at (99 - Y)
        """
        from kalshiflow_rl.data.orderbook_state import OrderbookState
        
        state = OrderbookState("TEST-MARKET")
        
        # Apply a snapshot with integer keys (only bids provided, asks derived)
        snapshot_data = {
            "market_ticker": "TEST-MARKET",
            "timestamp_ms": 1234567890,
            "sequence_number": 100,
            "yes_bids": {45: 100, 44: 200},  # Integer keys
            "no_bids": {40: 250, 39: 175},
        }
        
        state.apply_snapshot(snapshot_data)
        
        # Verify bids were stored correctly
        assert state.yes_bids[45] == 100
        assert state.yes_bids[44] == 200
        assert state.no_bids[40] == 250
        assert state.no_bids[39] == 175
        
        # Verify asks were derived correctly from reciprocal relationship
        # NO_BID at 40 → YES_ASK at 59 (99-40)
        # NO_BID at 39 → YES_ASK at 60 (99-39)
        assert state.yes_asks[59] == 250
        assert state.yes_asks[60] == 175
        
        # YES_BID at 45 → NO_ASK at 54 (99-45)
        # YES_BID at 44 → NO_ASK at 55 (99-44)
        assert state.no_asks[54] == 100
        assert state.no_asks[55] == 200
        
        # Verify spreads are calculated correctly
        # yes: best_bid=45, best_ask=59 → spread=14
        yes_spread = state.get_yes_spread()
        assert yes_spread == 59 - 45
        
        # no: best_bid=40, best_ask=54 → spread=14
        no_spread = state.get_no_spread()
        assert no_spread == 54 - 40
    
    def test_message_type_detection(self, orderbook_client):
        """Test correct detection of message types."""
        # Snapshot message
        snapshot_msg = {
            "type": "orderbook_snapshot",
            "msg": {"market_ticker": "TEST"}
        }
        assert orderbook_client._get_message_type(snapshot_msg) == "snapshot"
        
        # Delta message
        delta_msg = {
            "type": "orderbook_delta",
            "msg": {"market_ticker": "TEST"}
        }
        assert orderbook_client._get_message_type(delta_msg) == "delta"
        
        # Subscription acknowledgment
        ack_msg = {"type": "subscribed"}
        assert orderbook_client._get_message_type(ack_msg) == "subscription_ack"
        
        # Heartbeat
        heartbeat_msg = {"ping": 1234}
        assert orderbook_client._get_message_type(heartbeat_msg) == "heartbeat"