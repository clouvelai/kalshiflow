"""
Tests for the PrimitiveActionSpace implementation.

This test suite validates the 5-action primitive action space designed for 
truly stateless operation in the Kalshi Flow RL Trading Subsystem.
"""

import pytest
import numpy as np
from gymnasium import spaces

from kalshiflow_rl.environments.action_space import (
    PrimitiveActionSpace,
    PrimitiveActions,
    DecodedAction,
    primitive_action_space
)


class TestPrimitiveActions:
    """Test the PrimitiveActions enum."""
    
    def test_action_values(self):
        """Test that actions have correct integer values."""
        assert PrimitiveActions.HOLD == 0
        assert PrimitiveActions.BUY_YES_NOW == 1
        assert PrimitiveActions.SELL_YES_NOW == 2
        assert PrimitiveActions.BUY_NO_NOW == 3
        assert PrimitiveActions.SELL_NO_NOW == 4
    
    def test_action_count(self):
        """Test that we have exactly 5 actions."""
        actions = list(PrimitiveActions)
        assert len(actions) == 5


class TestDecodedAction:
    """Test the DecodedAction dataclass."""
    
    def test_to_dict(self):
        """Test conversion to dictionary format."""
        action = DecodedAction(
            ticker="MARKET-123",
            action="buy",
            side="yes",
            count=10,
            type="market"
        )
        
        expected = {
            'ticker': "MARKET-123",
            'action': "buy", 
            'side': "yes",
            'count': 10,
            'type': "market"
        }
        
        assert action.to_dict() == expected


class TestPrimitiveActionSpace:
    """Test the PrimitiveActionSpace class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = PrimitiveActionSpace()
        self.test_ticker = "TEST-MARKET-123"
    
    def test_initialization(self):
        """Test proper initialization of action space."""
        assert self.action_space.FIXED_CONTRACT_SIZE == 10
        assert len(self.action_space.action_descriptions) == 5
        
        # Check that all actions have descriptions
        for action in PrimitiveActions:
            assert action in self.action_space.action_descriptions
    
    def test_get_gym_space(self):
        """Test gymnasium space creation."""
        gym_space = self.action_space.get_gym_space()
        
        assert isinstance(gym_space, spaces.Discrete)
        assert gym_space.n == 5
        
        # Verify it's the same as directly creating spaces.Discrete(5)
        expected_space = spaces.Discrete(5)
        assert gym_space.n == expected_space.n
    
    def test_decode_action_hold(self):
        """Test decoding HOLD action."""
        result = self.action_space.decode_action(0, self.test_ticker)
        assert result is None
    
    def test_decode_action_buy_yes_now(self):
        """Test decoding BUY_YES_NOW action."""
        result = self.action_space.decode_action(1, self.test_ticker)
        
        assert isinstance(result, DecodedAction)
        assert result.ticker == self.test_ticker
        assert result.action == "buy"
        assert result.side == "yes"
        assert result.count == 10
        assert result.type == "market"
    
    def test_decode_action_sell_yes_now(self):
        """Test decoding SELL_YES_NOW action."""
        result = self.action_space.decode_action(2, self.test_ticker)
        
        assert isinstance(result, DecodedAction)
        assert result.ticker == self.test_ticker
        assert result.action == "sell"
        assert result.side == "yes"
        assert result.count == 10
        assert result.type == "market"
    
    def test_decode_action_buy_no_now(self):
        """Test decoding BUY_NO_NOW action.""" 
        result = self.action_space.decode_action(3, self.test_ticker)
        
        assert isinstance(result, DecodedAction)
        assert result.ticker == self.test_ticker
        assert result.action == "buy"
        assert result.side == "no"
        assert result.count == 10
        assert result.type == "market"
    
    def test_decode_action_sell_no_now(self):
        """Test decoding SELL_NO_NOW action."""
        result = self.action_space.decode_action(4, self.test_ticker)
        
        assert isinstance(result, DecodedAction)
        assert result.ticker == self.test_ticker
        assert result.action == "sell"
        assert result.side == "no" 
        assert result.count == 10
        assert result.type == "market"
    
    def test_decode_action_all_actions(self):
        """Test decoding all valid actions."""
        expected_results = [
            None,  # HOLD
            {"action": "buy", "side": "yes"},   # BUY_YES_NOW
            {"action": "sell", "side": "yes"},  # SELL_YES_NOW
            {"action": "buy", "side": "no"},    # BUY_NO_NOW
            {"action": "sell", "side": "no"},   # SELL_NO_NOW
        ]
        
        for action_id, expected in enumerate(expected_results):
            result = self.action_space.decode_action(action_id, self.test_ticker)
            
            if expected is None:
                assert result is None
            else:
                assert result is not None
                assert result.action == expected["action"]
                assert result.side == expected["side"]
                assert result.ticker == self.test_ticker
                assert result.count == 10
                assert result.type == "market"
    
    def test_decode_action_invalid_range(self):
        """Test error handling for invalid action range."""
        with pytest.raises(ValueError, match="Action must be in range"):
            self.action_space.decode_action(-1, self.test_ticker)
        
        with pytest.raises(ValueError, match="Action must be in range"):
            self.action_space.decode_action(5, self.test_ticker)
        
        with pytest.raises(ValueError, match="Action must be in range"):
            self.action_space.decode_action(100, self.test_ticker)
    
    def test_decode_action_invalid_type(self):
        """Test error handling for invalid action type."""
        with pytest.raises(ValueError, match="Action must be an integer"):
            self.action_space.decode_action("invalid", self.test_ticker)
        
        with pytest.raises(ValueError, match="Action must be an integer"):
            self.action_space.decode_action(3.5, self.test_ticker)
        
        with pytest.raises(ValueError, match="Action must be an integer"):
            self.action_space.decode_action(None, self.test_ticker)
    
    def test_decode_action_numpy_integer(self):
        """Test decoding with numpy integer types."""
        # Test various numpy integer types
        for numpy_int in [np.int32(2), np.int64(3), np.uint8(1)]:
            result = self.action_space.decode_action(numpy_int, self.test_ticker)
            assert result is not None
            assert result.count == 10
            assert result.type == "market"
    
    def test_validate_action_all_valid(self):
        """Test that all 5 actions validate as valid."""
        for action_id in range(5):
            is_valid, reason = self.action_space.validate_action(action_id)
            assert is_valid, f"Action {action_id} should be valid, got reason: {reason}"
    
    def test_validate_action_invalid_range(self):
        """Test validation of invalid action range."""
        is_valid, reason = self.action_space.validate_action(-1)
        assert not is_valid
        assert "not in valid range" in reason
        
        is_valid, reason = self.action_space.validate_action(5)
        assert not is_valid  
        assert "not in valid range" in reason
    
    def test_validate_action_hold_always_valid(self):
        """Test that HOLD action is always valid."""
        is_valid, reason = self.action_space.validate_action(PrimitiveActions.HOLD)
        assert is_valid
        assert "HOLD action is always valid" in reason
    
    def test_get_action_info_hold(self):
        """Test action info for HOLD action."""
        info = self.action_space.get_action_info(PrimitiveActions.HOLD)
        
        assert info['action_id'] == 0
        assert info['action_name'] == 'HOLD'
        assert 'No action' in info['description']
        assert info['is_trade'] is False
        assert info['contract_size'] == 0
        assert info['order_type'] is None
    
    def test_get_action_info_trade_actions(self):
        """Test action info for trading actions."""
        for action_id in range(1, 5):
            info = self.action_space.get_action_info(action_id)
            
            assert info['action_id'] == action_id
            assert info['action_name'] in ['BUY_YES_NOW', 'SELL_YES_NOW', 'BUY_NO_NOW', 'SELL_NO_NOW']
            assert 'Market' in info['description']
            assert info['is_trade'] is True
            assert info['contract_size'] == 10
            assert info['order_type'] == 'market'
    
    def test_get_action_info_invalid_range(self):
        """Test error handling for invalid action info request."""
        with pytest.raises(ValueError, match="Action must be in range"):
            self.action_space.get_action_info(-1)
        
        with pytest.raises(ValueError, match="Action must be in range"):
            self.action_space.get_action_info(5)
    
    def test_get_all_actions_info(self):
        """Test getting info for all actions."""
        all_info = self.action_space.get_all_actions_info()
        
        assert len(all_info) == 5
        assert set(all_info.keys()) == {0, 1, 2, 3, 4}
        
        # Verify each action info is correct
        for action_id, info in all_info.items():
            assert info['action_id'] == action_id
            assert isinstance(info['action_name'], str)
            assert isinstance(info['description'], str)
            assert isinstance(info['is_trade'], bool)


class TestStatelessOperation:
    """Test that the action space is truly stateless."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = PrimitiveActionSpace()
        self.test_ticker = "STATELESS-TEST-123"
    
    def test_no_state_between_calls(self):
        """Test that action decoding doesn't maintain state."""
        # Decode multiple actions
        action1 = self.action_space.decode_action(1, self.test_ticker)
        action2 = self.action_space.decode_action(2, self.test_ticker)
        action3 = self.action_space.decode_action(1, self.test_ticker)
        
        # Each call should produce independent results
        assert action1.action == "buy"
        assert action1.side == "yes"
        
        assert action2.action == "sell"  
        assert action2.side == "yes"
        
        # Third call should be identical to first (no state)
        assert action3.action == action1.action
        assert action3.side == action1.side
        assert action3.ticker == action1.ticker
        assert action3.count == action1.count
    
    def test_fixed_contract_size(self):
        """Test that contract size is always fixed."""
        for action_id in range(1, 5):  # Skip HOLD
            result = self.action_space.decode_action(action_id, self.test_ticker)
            assert result.count == 10, f"Action {action_id} should have fixed count of 10"
    
    def test_only_market_orders(self):
        """Test that all orders are market orders (no limit orders)."""
        for action_id in range(1, 5):  # Skip HOLD
            result = self.action_space.decode_action(action_id, self.test_ticker)
            assert result.type == "market", f"Action {action_id} should be market order"
    
    def test_no_wait_actions(self):
        """Test that there are no WAIT/limit order actions."""
        # All actions should decode to either None or market orders
        for action_id in range(5):
            result = self.action_space.decode_action(action_id, self.test_ticker)
            
            if result is not None:
                assert result.type == "market", "All non-HOLD actions should be market orders"
        
        # Verify we only have exactly 5 actions (no WAIT actions)
        assert self.action_space.get_gym_space().n == 5


class TestGlobalInstance:
    """Test the global primitive_action_space instance."""
    
    def test_global_instance_exists(self):
        """Test that global instance is available."""
        assert primitive_action_space is not None
        assert isinstance(primitive_action_space, PrimitiveActionSpace)
    
    def test_global_instance_functionality(self):
        """Test that global instance works correctly."""
        gym_space = primitive_action_space.get_gym_space()
        assert isinstance(gym_space, spaces.Discrete)
        assert gym_space.n == 5
        
        # Test action decoding
        result = primitive_action_space.decode_action(1, "GLOBAL-TEST")
        assert result is not None
        assert result.action == "buy"
        assert result.side == "yes"


class TestKalshiOrderFormat:
    """Test compatibility with Kalshi API order format."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.action_space = PrimitiveActionSpace()
        self.test_ticker = "KALSHI-FORMAT-TEST"
    
    def test_order_format_structure(self):
        """Test that decoded actions match expected Kalshi format."""
        # Test a buy YES order
        result = self.action_space.decode_action(1, self.test_ticker)
        order_dict = result.to_dict()
        
        # Expected Kalshi order format fields
        required_fields = ['ticker', 'action', 'side', 'count', 'type']
        for field in required_fields:
            assert field in order_dict, f"Missing required field: {field}"
        
        # Check value constraints
        assert order_dict['action'] in ['buy', 'sell']
        assert order_dict['side'] in ['yes', 'no']
        assert isinstance(order_dict['count'], int)
        assert order_dict['count'] > 0
        assert order_dict['type'] == 'market'
    
    def test_all_actions_produce_valid_orders(self):
        """Test that all non-HOLD actions produce valid Kalshi orders."""
        for action_id in range(1, 5):  # Skip HOLD
            result = self.action_space.decode_action(action_id, self.test_ticker)
            order_dict = result.to_dict()
            
            # Verify structure
            assert order_dict['ticker'] == self.test_ticker
            assert order_dict['action'] in ['buy', 'sell']
            assert order_dict['side'] in ['yes', 'no']
            assert order_dict['count'] == 10
            assert order_dict['type'] == 'market'
    
    def test_action_side_combinations(self):
        """Test that we have all 4 action/side combinations."""
        expected_combinations = {
            1: ('buy', 'yes'),
            2: ('sell', 'yes'),
            3: ('buy', 'no'), 
            4: ('sell', 'no')
        }
        
        for action_id, (expected_action, expected_side) in expected_combinations.items():
            result = self.action_space.decode_action(action_id, self.test_ticker)
            assert result.action == expected_action
            assert result.side == expected_side