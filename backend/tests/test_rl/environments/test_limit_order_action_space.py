"""
Tests for the LimitOrderActionSpace implementation.

Tests cover action validation, masking, execution, and integration
with the OrderManager abstraction.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
import numpy as np

from kalshiflow_rl.environments.limit_order_action_space import (
    LimitOrderActionSpace,
    LimitOrderActions,
    ActionType,
    ActionExecutionResult
)
from kalshiflow_rl.trading.order_manager import OrderManager, OrderSide, ContractSide
from kalshiflow_rl.data.orderbook_state import OrderbookState


@pytest.fixture
def mock_order_manager():
    """Create a mock OrderManager for testing."""
    mock = AsyncMock(spec=OrderManager)
    mock.get_open_orders.return_value = []
    mock.place_order.return_value = MagicMock(order_id="test_order_123")
    mock.cancel_order.return_value = True
    mock.amend_order.return_value = True
    mock.cancel_all_orders.return_value = 0
    mock._calculate_limit_price.return_value = 50
    return mock


@pytest.fixture
def sample_orderbook():
    """Create a sample orderbook for testing."""
    orderbook = OrderbookState("TEST_MARKET")
    orderbook.yes_bids = {45: 100, 44: 200, 43: 150}
    orderbook.yes_asks = {47: 100, 48: 200, 49: 150}
    orderbook.no_bids = {53: 100, 52: 200}
    orderbook.no_asks = {55: 100, 56: 150}
    orderbook.last_update = 1234567890
    return orderbook


@pytest.fixture
def action_space(mock_order_manager):
    """Create a LimitOrderActionSpace for testing."""
    return LimitOrderActionSpace(
        order_manager=mock_order_manager,
        contract_size=10,
        pricing_strategy="aggressive"
    )


class TestLimitOrderActions:
    """Test the LimitOrderActions enum."""
    
    def test_action_values(self):
        """Test that action enum values are correct."""
        assert LimitOrderActions.HOLD.value == 0
        assert LimitOrderActions.BUY_YES_LIMIT.value == 1
        assert LimitOrderActions.SELL_YES_LIMIT.value == 2
        assert LimitOrderActions.BUY_NO_LIMIT.value == 3
        assert LimitOrderActions.SELL_NO_LIMIT.value == 4
    
    def test_action_count(self):
        """Test that we have exactly 5 actions."""
        assert len(LimitOrderActions) == 5


class TestActionType:
    """Test the ActionType compatibility enum."""
    
    def test_action_type_mapping(self):
        """Test ActionType maps correctly for integration."""
        assert ActionType.HOLD.value == 0
        assert ActionType.BUY_YES.value == 1
        assert ActionType.SELL_YES.value == 2
        assert ActionType.BUY_NO.value == 3
        assert ActionType.SELL_NO.value == 4
        assert ActionType.CLOSE_POSITION.value == 5


class TestActionExecutionResult:
    """Test the ActionExecutionResult dataclass."""
    
    def test_successful_result(self):
        """Test successful action execution result."""
        result = ActionExecutionResult(
            action_taken=LimitOrderActions.BUY_YES_LIMIT,
            order_placed=True,
            order_cancelled=False,
            order_amended=False,
            order_id="test_123"
        )
        
        assert result.was_successful() is True
        assert result.action_taken == LimitOrderActions.BUY_YES_LIMIT
        assert result.order_id == "test_123"
    
    def test_failed_result(self):
        """Test failed action execution result."""
        result = ActionExecutionResult(
            action_taken=LimitOrderActions.BUY_YES_LIMIT,
            order_placed=False,
            order_cancelled=False,
            order_amended=False,
            order_id=None,
            error_message="Insufficient funds"
        )
        
        assert result.was_successful() is False
        assert result.error_message == "Insufficient funds"


class TestLimitOrderActionSpace:
    """Test the main LimitOrderActionSpace class."""
    
    def test_initialization(self, mock_order_manager):
        """Test action space initialization."""
        action_space = LimitOrderActionSpace(
            order_manager=mock_order_manager,
            contract_size=15,
            pricing_strategy="passive"
        )
        
        assert action_space.order_manager == mock_order_manager
        assert action_space.contract_size == 15
        assert action_space.pricing_strategy == "passive"
        assert len(action_space.action_descriptions) == 5
    
    def test_gym_space(self, action_space):
        """Test Gym space representation."""
        gym_space = action_space.get_gym_space()
        
        assert gym_space.n == 21
        assert hasattr(gym_space, 'sample')  # It's a proper Gym space
    
    def test_action_info(self, action_space):
        """Test getting information about specific actions."""
        # Test HOLD action
        hold_info = action_space.get_action_info(0)
        assert hold_info['action_id'] == 0
        assert hold_info['action_name'] == 'HOLD'
        assert hold_info['is_trade'] is False
        assert hold_info['contract_size'] == 0
        
        # Test trading action
        buy_yes_info = action_space.get_action_info(1)
        assert buy_yes_info['action_id'] == 1
        assert buy_yes_info['action_name'] == 'BUY_YES_LIMIT'
        assert buy_yes_info['is_trade'] is True
        assert buy_yes_info['contract_size'] == 10
        assert buy_yes_info['order_type'] == 'limit'
    
    def test_all_actions_info(self, action_space):
        """Test getting information about all actions."""
        all_info = action_space.get_all_actions_info()
        
        assert len(all_info) == 5
        assert all(isinstance(k, int) for k in all_info.keys())
        assert all(0 <= k <= 4 for k in all_info.keys())
    
    def test_action_validation_valid(self, action_space, sample_orderbook):
        """Test action validation for valid actions."""
        # HOLD should always be valid
        is_valid, reason = action_space.validate_action(0, "TEST", sample_orderbook)
        assert is_valid is True
        assert "always valid" in reason.lower()
        
        # Trading actions should be valid with good orderbook
        is_valid, reason = action_space.validate_action(1, "TEST", sample_orderbook)
        assert is_valid is True
        assert "valid" in reason.lower()
    
    def test_action_validation_invalid_action_id(self, action_space, sample_orderbook):
        """Test action validation for invalid action IDs."""
        is_valid, reason = action_space.validate_action(5, "TEST", sample_orderbook)
        assert is_valid is False
        assert "not in valid range" in reason
        
        is_valid, reason = action_space.validate_action(-1, "TEST", sample_orderbook)
        assert is_valid is False
        assert "not in valid range" in reason
    
    def test_action_validation_empty_orderbook(self, action_space):
        """Test action validation with empty orderbook."""
        empty_orderbook = OrderbookState("EMPTY")
        
        # HOLD should still be valid
        is_valid, reason = action_space.validate_action(0, "EMPTY", empty_orderbook)
        assert is_valid is True
        
        # Trading actions should be invalid
        is_valid, reason = action_space.validate_action(1, "EMPTY", empty_orderbook)
        assert is_valid is False
        assert "no valid prices" in reason
    
    def test_action_validation_wide_spread(self, action_space):
        """Test action validation with very wide spread."""
        wide_spread_orderbook = OrderbookState("WIDE")
        wide_spread_orderbook.yes_bids = {10: 100}  # Very low bid
        wide_spread_orderbook.yes_asks = {90: 100}  # Very high ask
        
        is_valid, reason = action_space.validate_action(1, "WIDE", wide_spread_orderbook)
        assert is_valid is False
        assert "spread too wide" in reason.lower()
    
    def test_action_masking(self, action_space, sample_orderbook):
        """Test action masking functionality."""
        mask = action_space.get_action_mask("TEST", sample_orderbook, cash_balance=1000.0)
        
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (5,)
        assert mask.dtype == bool
        assert mask[0] == True  # HOLD should be valid
    
    def test_action_masking_insufficient_cash(self, action_space, sample_orderbook):
        """Test action masking with insufficient cash."""
        mask = action_space.get_action_mask("TEST", sample_orderbook, cash_balance=0.01)
        
        assert mask[0] == True  # HOLD should still be valid
        # Trading actions may be invalid due to insufficient cash
        # The exact behavior depends on the cash check logic
    
    def test_pricing_strategy_management(self, action_space):
        """Test pricing strategy get/set."""
        assert action_space.get_pricing_strategy() == "aggressive"
        
        action_space.set_pricing_strategy("passive")
        assert action_space.get_pricing_strategy() == "passive"
        
        action_space.set_pricing_strategy("mid")
        assert action_space.get_pricing_strategy() == "mid"
        
        # Test invalid strategy
        with pytest.raises(ValueError):
            action_space.set_pricing_strategy("invalid")
    
    def test_contract_size_access(self, action_space):
        """Test contract size access."""
        assert action_space.get_contract_size() == 10
    
    def test_suggest_best_action(self, action_space, sample_orderbook):
        """Test action suggestion functionality."""
        # Test no target
        action, reason = action_space.suggest_best_action("TEST", sample_orderbook)
        assert action == 0
        assert "no target" in reason.lower()
        
        # Test target requiring YES position
        action, reason = action_space.suggest_best_action("TEST", sample_orderbook, target_position=10)
        assert action == 1  # BUY_YES_LIMIT
        assert "buy yes" in reason.lower()
        
        # Test target requiring NO position
        action, reason = action_space.suggest_best_action("TEST", sample_orderbook, target_position=-10)
        assert action == 3  # BUY_NO_LIMIT
        assert "buy no" in reason.lower()
        
        # Test target requiring position reduction
        action, reason = action_space.suggest_best_action("TEST", sample_orderbook, target_position=0, current_position=10)
        assert action == 2  # SELL_YES_LIMIT
        assert "sell yes" in reason.lower()
    
    def test_action_space_info(self, action_space):
        """Test comprehensive action space information."""
        info = action_space.get_action_space_info()
        
        assert info["action_space_type"] == "LimitOrderActionSpace"
        assert info["action_count"] == 5
        assert info["contract_size"] == 10
        assert info["pricing_strategy"] == "aggressive"
        
        # Check constraints
        assert "order_management" in info["constraints"]
        assert "order_type" in info["constraints"]
        
        # Check features
        assert info["features"]["action_masking"] is True
        assert info["features"]["action_validation"] is True
        
        # Check Kalshi compliance
        assert info["kalshi_compliance"]["limit_orders_only"] is True
        assert info["kalshi_compliance"]["position_convention"] == "+YES/-NO contracts"


class TestActionExecution:
    """Test action execution functionality."""
    
    @pytest.mark.asyncio
    async def test_hold_action_execution(self, action_space, sample_orderbook):
        """Test HOLD action execution."""
        result = await action_space.execute_action(0, "TEST", sample_orderbook)
        
        assert result.action_taken == LimitOrderActions.HOLD
        assert result.was_successful()
        assert result.order_placed is False
    
    @pytest.mark.asyncio
    async def test_invalid_action_execution(self, action_space, sample_orderbook):
        """Test invalid action execution."""
        result = await action_space.execute_action(10, "TEST", sample_orderbook)
        
        assert result.was_successful() is False
        assert "not a valid" in result.error_message
    
    @pytest.mark.asyncio
    async def test_buy_action_execution(self, action_space, sample_orderbook, mock_order_manager):
        """Test buy action execution."""
        # Mock successful order placement
        mock_order = MagicMock()
        mock_order.order_id = "test_buy_123"
        mock_order_manager.place_order.return_value = mock_order
        
        result = await action_space.execute_action(1, "TEST", sample_orderbook)
        
        assert result.action_taken == LimitOrderActions.BUY_YES_LIMIT
        assert result.was_successful()
        assert result.order_placed is True
        assert result.order_id == "test_buy_123"
        
        # Verify OrderManager was called correctly
        mock_order_manager.place_order.assert_called_once()
        call_args = mock_order_manager.place_order.call_args
        assert call_args.kwargs["ticker"] == "TEST"
        assert call_args.kwargs["side"] == OrderSide.BUY
        assert call_args.kwargs["contract_side"] == ContractSide.YES
        assert call_args.kwargs["quantity"] == 10
    
    @pytest.mark.asyncio
    async def test_sell_action_execution(self, action_space, sample_orderbook, mock_order_manager):
        """Test sell action execution."""
        mock_order = MagicMock()
        mock_order.order_id = "test_sell_123"
        mock_order_manager.place_order.return_value = mock_order
        
        result = await action_space.execute_action(2, "TEST", sample_orderbook)
        
        assert result.action_taken == LimitOrderActions.SELL_YES_LIMIT
        assert result.was_successful()
        assert result.order_placed is True
        assert result.order_id == "test_sell_123"
        
        # Verify OrderManager was called correctly
        mock_order_manager.place_order.assert_called_once()
        call_args = mock_order_manager.place_order.call_args
        assert call_args.kwargs["side"] == OrderSide.SELL
        assert call_args.kwargs["contract_side"] == ContractSide.YES
    
    @pytest.mark.asyncio
    async def test_order_placement_failure(self, action_space, sample_orderbook, mock_order_manager):
        """Test handling of order placement failure."""
        # Mock failed order placement
        mock_order_manager.place_order.return_value = None
        
        result = await action_space.execute_action(1, "TEST", sample_orderbook)
        
        assert result.action_taken == LimitOrderActions.BUY_YES_LIMIT
        assert result.was_successful() is False
        assert "Failed to place" in result.error_message
        assert result.order_id is None
    
    @pytest.mark.asyncio
    async def test_exception_handling(self, action_space, sample_orderbook, mock_order_manager):
        """Test exception handling during action execution."""
        # Mock exception during order placement
        mock_order_manager.place_order.side_effect = Exception("Network error")
        
        result = await action_space.execute_action(1, "TEST", sample_orderbook)
        
        assert result.was_successful() is False
        assert "Network error" in result.error_message


class TestOrderConflictManagement:
    """Test order conflict management and cancellation logic."""
    
    @pytest.mark.asyncio
    async def test_conflicting_order_cancellation(self, action_space, sample_orderbook, mock_order_manager):
        """Test that conflicting orders are cancelled."""
        # Mock existing conflicting orders
        existing_orders = [
            MagicMock(order_id="sell_order", side=OrderSide.SELL, contract_side=ContractSide.YES),
            MagicMock(order_id="no_order", side=OrderSide.BUY, contract_side=ContractSide.NO)
        ]
        mock_order_manager.get_open_orders.return_value = existing_orders
        mock_order_manager.cancel_order.return_value = True
        
        # Mock successful new order
        mock_order = MagicMock(order_id="new_buy_yes")
        mock_order_manager.place_order.return_value = mock_order
        
        # Execute BUY_YES action - should cancel conflicting orders
        result = await action_space.execute_action(1, "TEST", sample_orderbook)
        
        assert result.was_successful()
        assert result.order_cancelled is True
        assert mock_order_manager.cancel_order.call_count == 2  # Both conflicting orders cancelled
    
    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, action_space, mock_order_manager):
        """Test cancel all orders functionality."""
        mock_order_manager.cancel_all_orders.return_value = 3
        
        cancelled_count = await action_space.cancel_all_orders("TEST")
        
        assert cancelled_count == 3
        mock_order_manager.cancel_all_orders.assert_called_once_with("TEST")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])