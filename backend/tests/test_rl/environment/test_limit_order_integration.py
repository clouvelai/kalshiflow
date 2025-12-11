"""
Integration tests for limit order action space and OrderManager system.

Tests the full pipeline: Environment → LimitOrderActionSpace → OrderManager → Execution
Validates both SimulatedOrderManager and mocked KalshiOrderManager behavior.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from typing import Dict, Any

from src.kalshiflow_rl.environments.limit_order_action_space import (
    LimitOrderActionSpace, 
    LimitOrderActions, 
    ActionExecutionResult,
    ActionType
)
from src.kalshiflow_rl.trading.order_manager import (
    SimulatedOrderManager, 
    KalshiOrderManager,
    OrderManager,
    OrderInfo,
    OrderSide,
    ContractSide,
    OrderStatus,
    Position
)
from src.kalshiflow_rl.data.orderbook_state import OrderbookState


@pytest.fixture
def mock_orderbook_state():
    """Create a mock orderbook state for testing."""
    state = OrderbookState(market_ticker="TEST-123")
    
    # Set up realistic orderbook data using proper snapshot format
    snapshot_data = {
        "yes_bids": {
            "64": 100,  # 64¢ bid with size 100
            "63": 50,   # 63¢ bid with size 50
            "62": 75    # 62¢ bid with size 75
        },
        "yes_asks": {
            "66": 80,   # 66¢ ask with size 80
            "67": 60,   # 67¢ ask with size 60
            "68": 90    # 68¢ ask with size 90
        },
        "no_bids": {
            "32": 80,   # NO bid derived from YES ask
            "31": 60,
            "30": 90
        },
        "no_asks": {
            "34": 100,  # NO ask derived from YES bid
            "35": 50,
            "36": 75
        }
    }
    
    state.apply_snapshot(snapshot_data)
    return state


@pytest.fixture
def simulated_order_manager():
    """Create SimulatedOrderManager for testing."""
    return SimulatedOrderManager(initial_cash=1000.0)


@pytest.fixture  
def mock_kalshi_order_manager():
    """Create mocked KalshiOrderManager for testing."""
    # Create a mock demo client
    mock_client = AsyncMock()
    mock_client.create_order.return_value = {
        "order": {"order_id": "kalshi_123", "status": "pending"}
    }
    mock_client.cancel_order.return_value = {"success": True}
    mock_client.get_orders.return_value = {"orders": []}
    
    # Create the manager with mocked client
    manager = KalshiOrderManager(mock_client, initial_cash=1000.0)
    return manager


@pytest.mark.asyncio
class TestLimitOrderIntegration:
    """Test integration between action space and order managers."""
    
    async def test_hold_action_with_simulated_manager(
        self, 
        simulated_order_manager, 
        mock_orderbook_state
    ):
        """Test HOLD action execution with SimulatedOrderManager."""
        action_space = LimitOrderActionSpace(
            order_manager=simulated_order_manager,
            contract_size=10
        )
        
        # Execute HOLD action
        result = await action_space.execute_action(
            action=LimitOrderActions.HOLD,
            ticker="TEST-123",
            orderbook=mock_orderbook_state
        )
        
        # Verify results
        assert result.was_successful()
        assert result.action_taken == LimitOrderActions.HOLD
        assert not result.order_placed
        assert not result.order_cancelled
        assert not result.order_amended
        assert result.order_id is None
    
    async def test_buy_yes_action_with_simulated_manager(
        self, 
        simulated_order_manager, 
        mock_orderbook_state
    ):
        """Test BUY_YES_LIMIT action execution with SimulatedOrderManager."""
        action_space = LimitOrderActionSpace(
            order_manager=simulated_order_manager,
            contract_size=10,
            pricing_strategy="passive"
        )
        
        # Execute BUY_YES_LIMIT action
        result = await action_space.execute_action(
            action=LimitOrderActions.BUY_YES_LIMIT,
            ticker="TEST-123", 
            orderbook=mock_orderbook_state
        )
        
        # Verify results
        assert result.was_successful()
        assert result.action_taken == LimitOrderActions.BUY_YES_LIMIT
        # Should place order since no existing conflicting orders
        assert result.order_placed
        assert not result.order_cancelled
        assert result.order_id is not None
        
        # Verify order was added to manager
        open_orders = simulated_order_manager.get_open_orders("TEST-123")
        assert len(open_orders) == 1
        assert open_orders[0].side == OrderSide.BUY
        assert open_orders[0].contract_side == ContractSide.YES
        assert open_orders[0].quantity == 10
    
    async def test_sell_no_action_with_simulated_manager(
        self, 
        simulated_order_manager, 
        mock_orderbook_state
    ):
        """Test SELL_NO_LIMIT action execution with SimulatedOrderManager."""
        action_space = LimitOrderActionSpace(
            order_manager=simulated_order_manager,
            contract_size=10,
            pricing_strategy="passive"
        )
        
        # Execute SELL_NO_LIMIT action
        result = await action_space.execute_action(
            action=LimitOrderActions.SELL_NO_LIMIT,
            ticker="TEST-123",
            orderbook=mock_orderbook_state
        )
        
        # Verify results
        assert result.was_successful()
        assert result.action_taken == LimitOrderActions.SELL_NO_LIMIT
        assert result.order_placed
        assert result.order_id is not None
        
        # Verify order was added to manager
        open_orders = simulated_order_manager.get_open_orders("TEST-123")
        assert len(open_orders) == 1
        assert open_orders[0].side == OrderSide.SELL
        assert open_orders[0].contract_side == ContractSide.NO
        assert open_orders[0].quantity == 10
    
    async def test_order_conflict_resolution(
        self, 
        simulated_order_manager, 
        mock_orderbook_state
    ):
        """Test that conflicting orders are cancelled when switching sides."""
        action_space = LimitOrderActionSpace(
            order_manager=simulated_order_manager,
            contract_size=10
        )
        
        # First place a BUY YES order
        result1 = await action_space.execute_action(
            action=LimitOrderActions.BUY_YES_LIMIT,
            ticker="TEST-123",
            orderbook=mock_orderbook_state
        )
        assert result1.was_successful()
        assert result1.order_placed
        
        # Verify we have one open order
        open_orders = simulated_order_manager.get_open_orders("TEST-123")
        assert len(open_orders) == 1
        assert open_orders[0].side == OrderSide.BUY
        
        # Now place a SELL YES order (conflicting)
        result2 = await action_space.execute_action(
            action=LimitOrderActions.SELL_YES_LIMIT,
            ticker="TEST-123",
            orderbook=mock_orderbook_state
        )
        assert result2.was_successful()
        assert result2.order_placed
        assert result2.order_cancelled  # Previous order should be cancelled
        
        # Verify we still have one order but it's now a sell order
        open_orders = simulated_order_manager.get_open_orders("TEST-123")
        assert len(open_orders) == 1
        assert open_orders[0].side == OrderSide.SELL
    
    async def test_all_five_actions(
        self, 
        simulated_order_manager, 
        mock_orderbook_state
    ):
        """Test all five limit order actions work correctly."""
        action_space = LimitOrderActionSpace(
            order_manager=simulated_order_manager,
            contract_size=10
        )
        
        actions_to_test = [
            (LimitOrderActions.HOLD, None, None),
            (LimitOrderActions.BUY_YES_LIMIT, OrderSide.BUY, ContractSide.YES),
            (LimitOrderActions.SELL_YES_LIMIT, OrderSide.SELL, ContractSide.YES),
            (LimitOrderActions.BUY_NO_LIMIT, OrderSide.BUY, ContractSide.NO),
            (LimitOrderActions.SELL_NO_LIMIT, OrderSide.SELL, ContractSide.NO)
        ]
        
        for action, expected_side, expected_contract_side in actions_to_test:
            # Clear any existing orders
            await simulated_order_manager.cancel_all_orders("TEST-123")
            
            # Execute action
            result = await action_space.execute_action(
                action=action,
                ticker="TEST-123",
                orderbook=mock_orderbook_state
            )
            
            # Verify basic success
            assert result.was_successful(), f"Action {action} failed"
            assert result.action_taken == action
            
            if action == LimitOrderActions.HOLD:
                # HOLD should not place orders
                assert not result.order_placed
                open_orders = simulated_order_manager.get_open_orders("TEST-123")
                assert len(open_orders) == 0
            else:
                # Trading actions should place orders
                assert result.order_placed
                assert result.order_id is not None
                
                # Verify order details
                open_orders = simulated_order_manager.get_open_orders("TEST-123")
                assert len(open_orders) == 1
                assert open_orders[0].side == expected_side
                assert open_orders[0].contract_side == expected_contract_side
                assert open_orders[0].quantity == 10
    
    async def test_action_validation(
        self, 
        simulated_order_manager, 
        mock_orderbook_state
    ):
        """Test action validation with various inputs."""
        action_space = LimitOrderActionSpace(
            order_manager=simulated_order_manager,
            contract_size=10
        )
        
        # Test valid actions
        for action in range(5):
            is_valid, reason = action_space.validate_action(
                action=action,
                ticker="TEST-123",
                orderbook=mock_orderbook_state
            )
            assert is_valid, f"Valid action {action} failed validation: {reason}"
        
        # Test invalid actions
        invalid_actions = [-1, 5, 10, 100]
        for action in invalid_actions:
            is_valid, reason = action_space.validate_action(
                action=action,
                ticker="TEST-123",
                orderbook=mock_orderbook_state
            )
            assert not is_valid, f"Invalid action {action} passed validation"
    
    async def test_pricing_strategies(
        self, 
        simulated_order_manager, 
        mock_orderbook_state
    ):
        """Test different pricing strategies produce different prices."""
        prices_by_strategy = {}
        
        for strategy in ["aggressive", "passive", "mid"]:
            action_space = LimitOrderActionSpace(
                order_manager=simulated_order_manager,
                contract_size=10,
                pricing_strategy=strategy
            )
            
            # Clear any existing orders
            await simulated_order_manager.cancel_all_orders("TEST-123")
            
            # Execute BUY YES action
            result = await action_space.execute_action(
                action=LimitOrderActions.BUY_YES_LIMIT,
                ticker="TEST-123",
                orderbook=mock_orderbook_state
            )
            
            assert result.was_successful()
            
            # Get the order price
            open_orders = simulated_order_manager.get_open_orders("TEST-123")
            assert len(open_orders) == 1
            prices_by_strategy[strategy] = open_orders[0].limit_price
        
        # Verify we got different prices (at minimum, aggressive should be highest)
        # Aggressive should be willing to pay more (hit the ask)
        # Passive should bid lower 
        # Mid should be in between
        assert len(set(prices_by_strategy.values())) > 1, "Different strategies should produce different prices"

    @patch('src.kalshiflow_rl.trading.demo_client.KalshiDemoTradingClient')
    async def test_kalshi_order_manager_integration(
        self, 
        mock_demo_client_class,
        mock_orderbook_state
    ):
        """Test integration with mocked KalshiOrderManager."""
        # Set up mock client
        mock_client = AsyncMock()
        mock_client.create_order.return_value = {
            "order": {"order_id": "kalshi_123", "status": "pending"}
        }
        mock_demo_client_class.return_value = mock_client
        
        # Create KalshiOrderManager
        kalshi_manager = KalshiOrderManager(mock_client, initial_cash=1000.0)
        
        action_space = LimitOrderActionSpace(
            order_manager=kalshi_manager,
            contract_size=10
        )
        
        # Execute BUY YES action
        result = await action_space.execute_action(
            action=LimitOrderActions.BUY_YES_LIMIT,
            ticker="TEST-123",
            orderbook=mock_orderbook_state
        )
        
        # Verify action executed successfully
        assert result.was_successful()
        assert result.action_taken == LimitOrderActions.BUY_YES_LIMIT
        assert result.order_placed
        assert result.order_id is not None
        
        # Verify the Kalshi client was called
        mock_client.create_order.assert_called_once()
        call_args = mock_client.create_order.call_args
        assert call_args[1]['ticker'] == "TEST-123"
        assert call_args[1]['action'] == "buy"
        assert call_args[1]['side'] == "yes"
        assert call_args[1]['count'] == 10
        assert call_args[1]['type'] == "limit"
    
    async def test_websocket_fill_tracking_setup(self):
        """Test WebSocket fill tracking methods on KalshiOrderManager."""
        # Create mocked client
        mock_client = AsyncMock()
        kalshi_manager = KalshiOrderManager(mock_client, initial_cash=1000.0)
        
        # Test WebSocket connection setup
        await kalshi_manager._connect_websocket()
        assert hasattr(kalshi_manager, '_ws_connected')
        assert hasattr(kalshi_manager, '_ws_url')
        
        # Test fill tracking activation
        await kalshi_manager.start_fill_tracking()
        assert kalshi_manager.is_fill_tracking_active()
        
        # Test fill tracking deactivation
        await kalshi_manager.stop_fill_tracking()
        assert not kalshi_manager.is_fill_tracking_active()
    
    async def test_fill_message_processing(self):
        """Test processing of fill messages from WebSocket."""
        mock_client = AsyncMock()
        kalshi_manager = KalshiOrderManager(mock_client, initial_cash=1000.0)
        
        # Set up a tracked order
        order_id = "our_order_123"
        kalshi_order_id = "kalshi_order_456"
        
        order = OrderInfo(
            order_id=order_id,
            ticker="TEST-123",
            side=OrderSide.BUY,
            contract_side=ContractSide.YES,
            quantity=10,
            limit_price=65,
            status=OrderStatus.PENDING,
            placed_at=1000000
        )
        
        # Add to tracking
        kalshi_manager.open_orders[order_id] = order
        kalshi_manager._kalshi_order_mapping[order_id] = kalshi_order_id
        kalshi_manager._reverse_mapping[kalshi_order_id] = order_id
        
        # Simulate fill message
        fill_message = {
            "type": "fill",
            "data": {
                "order_id": kalshi_order_id,
                "ticker": "TEST-123",
                "side": "yes", 
                "action": "buy",
                "count": 10,
                "yes_price": 65,
                "fill_time": "2023-12-10T15:30:00Z"
            }
        }
        
        # Process fill message
        await kalshi_manager._process_fill_message(fill_message)
        
        # Verify order was removed from tracking (fully filled)
        assert order_id not in kalshi_manager.open_orders
        assert order_id not in kalshi_manager._kalshi_order_mapping
        assert kalshi_order_id not in kalshi_manager._reverse_mapping
        
        # Verify position was updated
        position = kalshi_manager.positions.get("TEST-123")
        assert position is not None
        assert position.contracts == 10  # +YES position


@pytest.mark.asyncio
class TestActionTypeCompatibility:
    """Test backward compatibility with ActionType enum for integration."""
    
    async def test_action_type_mapping(self):
        """Test that ActionType enum maps correctly to LimitOrderActions."""
        # Verify enum values match expectations
        assert ActionType.HOLD == 0
        assert ActionType.BUY_YES == 1  
        assert ActionType.SELL_YES == 2
        assert ActionType.BUY_NO == 3
        assert ActionType.SELL_NO == 4
        assert ActionType.CLOSE_POSITION == 5
        
        # Verify integration compatibility
        assert ActionType.HOLD == LimitOrderActions.HOLD
        assert ActionType.BUY_YES == LimitOrderActions.BUY_YES_LIMIT
        assert ActionType.SELL_YES == LimitOrderActions.SELL_YES_LIMIT
        assert ActionType.BUY_NO == LimitOrderActions.BUY_NO_LIMIT
        assert ActionType.SELL_NO == LimitOrderActions.SELL_NO_LIMIT


@pytest.mark.asyncio
class TestPerformanceAndReliability:
    """Test performance and reliability of the limit order system."""
    
    async def test_order_manager_state_consistency(
        self, 
        simulated_order_manager, 
        mock_orderbook_state
    ):
        """Test that order manager maintains consistent state across operations."""
        action_space = LimitOrderActionSpace(
            order_manager=simulated_order_manager,
            contract_size=10
        )
        
        # Execute multiple actions and verify state consistency
        actions = [
            LimitOrderActions.BUY_YES_LIMIT,
            LimitOrderActions.SELL_YES_LIMIT,
            LimitOrderActions.BUY_NO_LIMIT,
            LimitOrderActions.HOLD,
            LimitOrderActions.SELL_NO_LIMIT
        ]
        
        for i, action in enumerate(actions):
            result = await action_space.execute_action(
                action=action,
                ticker=f"TEST-{i}",  # Use different tickers
                orderbook=mock_orderbook_state
            )
            
            assert result.was_successful()
            
            # Verify cash balance consistency
            assert simulated_order_manager.cash_balance >= 0
            
            # Verify position tracking consistency
            for ticker, position in simulated_order_manager.positions.items():
                assert isinstance(position.contracts, (int, float))
                assert isinstance(position.cost_basis, (int, float))
                assert isinstance(position.realized_pnl, (int, float))
    
    async def test_concurrent_order_operations(
        self, 
        simulated_order_manager,
        mock_orderbook_state
    ):
        """Test that order operations work correctly under concurrent execution."""
        action_space = LimitOrderActionSpace(
            order_manager=simulated_order_manager,
            contract_size=10
        )
        
        # Create multiple concurrent operations
        tasks = []
        for i in range(5):
            task = action_space.execute_action(
                action=LimitOrderActions.BUY_YES_LIMIT,
                ticker=f"TEST-{i}",
                orderbook=mock_orderbook_state
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all operations succeeded
        for i, result in enumerate(results):
            assert result.was_successful(), f"Task {i} failed"
            assert result.order_placed
        
        # Verify we have the expected number of orders
        total_orders = sum(
            len(simulated_order_manager.get_open_orders(f"TEST-{i}"))
            for i in range(5)
        )
        assert total_orders == 5