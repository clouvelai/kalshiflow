"""
Unit tests for Kalshi trade message models.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from kalshiflow.models import (
    Trade, TradeMessage, TickerState, 
    SnapshotMessage, TradeUpdateMessage, ConnectionStatus
)


class TestTrade:
    """Test Trade model validation and functionality."""
    
    def test_valid_trade(self):
        """Test creating a valid trade."""
        trade = Trade(
            market_ticker="PRESWIN25",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1701648000000  # Mock timestamp
        )
        
        assert trade.market_ticker == "PRESWIN25"
        assert trade.yes_price == 65
        assert trade.no_price == 35
        assert trade.yes_price_dollars == 0.65
        assert trade.no_price_dollars == 0.35
        assert trade.count == 100
        assert trade.taker_side == "yes"
        assert trade.ts == 1701648000000
    
    def test_price_validation(self):
        """Test price range validation."""
        # Valid price range
        Trade(
            market_ticker="TEST",
            yes_price=0,
            no_price=99,
            yes_price_dollars=0.0,
            no_price_dollars=0.99,
            count=1,
            taker_side="yes",
            ts=1701648000000
        )
        
        # Invalid price range
        with pytest.raises(ValidationError):
            Trade(
                market_ticker="TEST",
                yes_price=100,  # Too high
                no_price=35,
                yes_price_dollars=0.65,
                no_price_dollars=0.35,
                count=1,
                taker_side="yes",
                ts=1701648000000
            )
        
        with pytest.raises(ValidationError):
            Trade(
                market_ticker="TEST",
                yes_price=65,
                no_price=-1,  # Too low
                yes_price_dollars=0.65,
                no_price_dollars=0.35,
                count=1,
                taker_side="yes",
                ts=1701648000000
            )
    
    def test_dollar_price_validation(self):
        """Test dollar price range validation."""
        # Invalid dollar price range
        with pytest.raises(ValidationError):
            Trade(
                market_ticker="TEST",
                yes_price=65,
                no_price=35,
                yes_price_dollars=1.0,  # Too high
                no_price_dollars=0.35,
                count=1,
                taker_side="yes",
                ts=1701648000000
            )
        
        with pytest.raises(ValidationError):
            Trade(
                market_ticker="TEST",
                yes_price=65,
                no_price=35,
                yes_price_dollars=0.65,
                no_price_dollars=-0.01,  # Too low
                count=1,
                taker_side="yes",
                ts=1701648000000
            )
    
    def test_positive_count_validation(self):
        """Test that trade count must be positive."""
        with pytest.raises(ValidationError):
            Trade(
                market_ticker="TEST",
                yes_price=65,
                no_price=35,
                yes_price_dollars=0.65,
                no_price_dollars=0.35,
                count=0,  # Invalid
                taker_side="yes",
                ts=1701648000000
            )
        
        with pytest.raises(ValidationError):
            Trade(
                market_ticker="TEST",
                yes_price=65,
                no_price=35,
                yes_price_dollars=0.65,
                no_price_dollars=0.35,
                count=-5,  # Invalid
                taker_side="yes",
                ts=1701648000000
            )
    
    def test_timestamp_property(self):
        """Test timestamp conversion to datetime."""
        trade = Trade(
            market_ticker="TEST",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1701648000000
        )
        
        timestamp = trade.timestamp
        assert isinstance(timestamp, datetime)
        assert timestamp.timestamp() == 1701648000.0
    
    def test_price_display_property(self):
        """Test price display for different taker sides."""
        trade_yes = Trade(
            market_ticker="TEST",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="yes",
            ts=1701648000000
        )
        
        trade_no = Trade(
            market_ticker="TEST",
            yes_price=65,
            no_price=35,
            yes_price_dollars=0.65,
            no_price_dollars=0.35,
            count=100,
            taker_side="no",
            ts=1701648000000
        )
        
        assert trade_yes.price_display == "$0.65"
        assert trade_no.price_display == "$0.35"


class TestTradeMessage:
    """Test TradeMessage model and conversion."""
    
    def test_trade_message_conversion(self):
        """Test converting TradeMessage to Trade."""
        trade_msg = TradeMessage(msg={
            "market_ticker": "PRESWIN25",
            "yes_price": 65,
            "no_price": 35,
            "count": 100,
            "taker_side": "yes",
            "ts": 1701648000000
        })
        
        trade = trade_msg.to_trade()
        
        assert isinstance(trade, Trade)
        assert trade.market_ticker == "PRESWIN25"
        assert trade.yes_price == 65
        assert trade.no_price == 35
        assert trade.yes_price_dollars == 0.65
        assert trade.no_price_dollars == 0.35
        assert trade.count == 100
        assert trade.taker_side == "yes"
        assert trade.ts == 1701648000000


class TestTickerState:
    """Test TickerState model and properties."""
    
    def test_ticker_state_properties(self):
        """Test TickerState computed properties."""
        state = TickerState(
            ticker="PRESWIN25",
            last_yes_price=65,
            last_no_price=35,
            last_trade_time=1701648000000,
            volume_window=1000,
            trade_count_window=10,
            yes_flow=600,
            no_flow=400,
            price_points=[0.60, 0.62, 0.65]
        )
        
        assert state.net_flow == 200  # 600 - 400
        assert state.last_yes_price_dollars == 0.65
        assert state.last_no_price_dollars == 0.35
        assert state.flow_direction == "bullish"
    
    def test_flow_direction(self):
        """Test flow direction calculation."""
        # Bullish
        state = TickerState(
            ticker="TEST",
            last_yes_price=50,
            last_no_price=50,
            last_trade_time=0,
            yes_flow=600,
            no_flow=400
        )
        assert state.flow_direction == "bullish"
        
        # Bearish
        state.yes_flow = 300
        state.no_flow = 500
        assert state.flow_direction == "bearish"
        
        # Neutral
        state.yes_flow = 400
        state.no_flow = 400
        assert state.flow_direction == "neutral"


class TestWebSocketMessages:
    """Test WebSocket message models."""
    
    def test_snapshot_message(self):
        """Test SnapshotMessage validation."""
        snapshot = SnapshotMessage(data={
            "recent_trades": [],
            "hot_markets": []
        })
        
        assert snapshot.type == "snapshot"
        assert "recent_trades" in snapshot.data
        assert "hot_markets" in snapshot.data
        
        # Test validation failure
        with pytest.raises(ValidationError):
            SnapshotMessage(data={
                "recent_trades": []
                # Missing hot_markets
            })
    
    def test_trade_update_message(self):
        """Test TradeUpdateMessage validation."""
        update = TradeUpdateMessage(data={
            "trade": {
                "market_ticker": "TEST",
                "yes_price": 50,
                "no_price": 50,
                "count": 1,
                "taker_side": "yes",
                "ts": 0
            },
            "ticker_state": {
                "ticker": "TEST",
                "last_yes_price": 50,
                "last_no_price": 50,
                "last_trade_time": 0
            }
        })
        
        assert update.type == "trade"
        assert "trade" in update.data
        assert "ticker_state" in update.data
        
        # Test validation failure
        with pytest.raises(ValidationError):
            TradeUpdateMessage(data={
                "trade": {}
                # Missing ticker_state
            })


class TestConnectionStatus:
    """Test ConnectionStatus model."""
    
    def test_connection_status(self):
        """Test ConnectionStatus model."""
        status = ConnectionStatus(
            connected=True,
            last_connected=datetime.now(),
            reconnect_attempts=0,
            error_message=None
        )
        
        assert status.connected is True
        assert status.last_connected is not None
        assert status.reconnect_attempts == 0
        assert status.error_message is None