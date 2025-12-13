"""
Critical Data Flow Integration Test - Issue #8, Test 1 of 5

Tests the core data flow: OrderbookClient → SharedOrderbookState → LiveObservationAdapter

This integration test validates:
- Real OrderbookState creation and updates
- SharedOrderbookState subscription and notifications 
- LiveObservationAdapter conversion to training-consistent features
- End-to-end data flow without mocking critical components

Focuses on M1-M2 functionality with minimal mocking for quick execution (<30 seconds).
"""

import asyncio
import pytest
import time
import numpy as np
from decimal import Decimal
from datetime import datetime
from unittest.mock import AsyncMock, patch

from kalshiflow_rl.data.orderbook_state import OrderbookState, SharedOrderbookState
from kalshiflow_rl.trading.live_observation_adapter import (
    LiveObservationAdapter, 
    initialize_live_observation_adapter,
    get_live_observation_adapter,
    build_live_observation
)
from kalshiflow_rl.data.orderbook_state import get_shared_orderbook_state


class TestCriticalDataFlowIntegration:
    """Test critical data flow integration with real components."""
    
    @pytest.mark.asyncio
    async def test_orderbook_to_shared_state_flow(self):
        """Test OrderbookState updates flow to SharedOrderbookState subscribers."""
        
        # Initialize real components
        market_ticker = "TEST-MARKET-01"
        
        # Create real OrderbookState
        orderbook_state = OrderbookState(market_ticker)
        
        # Create real SharedOrderbookState for this market
        shared_state = SharedOrderbookState(market_ticker)
        
        # Set up subscription tracking
        notifications_received = []
        
        def notification_callback(notification_data: dict):
            notifications_received.append({
                'ticker': notification_data.get('market_ticker'),
                'timestamp': notification_data.get('timestamp'),
                'state_data': notification_data
            })
        
        # Subscribe to updates
        shared_state.add_subscriber(notification_callback)
        
        # Simulate orderbook updates
        sample_snapshot = {
            'type': 'snapshot',
            'market': market_ticker,
            'sequence': 1,
            'ts': int(time.time() * 1000),
            'yes': {
                'bids': [{'price': 55, 'size': 100}, {'price': 54, 'size': 200}],
                'asks': [{'price': 56, 'size': 150}, {'price': 57, 'size': 250}]
            },
            'no': {
                'bids': [{'price': 44, 'size': 120}, {'price': 43, 'size': 180}],
                'asks': [{'price': 45, 'size': 130}, {'price': 46, 'size': 220}]
            }
        }
        
        # Update orderbook state
        orderbook_state.apply_snapshot(sample_snapshot)
        
        # Update shared state
        await shared_state.apply_snapshot(orderbook_state.to_dict())
        
        # Wait for async propagation
        await asyncio.sleep(0.1)
        
        # Verify subscription notification
        assert len(notifications_received) == 1
        notification = notifications_received[0]
        assert notification['ticker'] == market_ticker
        assert 'state_data' in notification
        
        # Verify state data structure
        state_data = notification['state_data']
        assert 'market_ticker' in state_data
        assert 'update_type' in state_data
        assert 'state_summary' in state_data
        
        summary = state_data['state_summary']
        assert 'last_sequence' in summary
        assert 'last_update_time' in summary
        assert 'total_volume' in summary
        
        print("✅ OrderbookState → SharedOrderbookState flow verified")

    @pytest.mark.asyncio 
    async def test_shared_state_to_observation_adapter_flow(self):
        """Test SharedOrderbookState data flows to LiveObservationAdapter correctly."""
        
        market_ticker = "TEST-MARKET-02"
        
        # Initialize observation adapter (uses global registry)
        adapter = LiveObservationAdapter(
            window_size=10,
            max_markets=1
        )
        
        # Create sample orderbook data
        orderbook_data = {
            'market_ticker': market_ticker,
            'last_sequence': 5,
            'last_update_time': time.time(),
            'yes_bids': {60: 100, 59: 150, 58: 200},
            'yes_asks': {61: 120, 62: 180, 63: 250},
            'no_bids': {40: 110, 39: 160, 38: 190},
            'no_asks': {41: 130, 42: 170, 43: 240}
        }
        
        # Get SharedOrderbookState for this market and update it
        market_shared_state = await get_shared_orderbook_state(market_ticker)
        await market_shared_state.apply_snapshot(orderbook_data)
        
        # Build observation using adapter
        observation = await adapter.build_observation(market_ticker)
        
        # Verify observation structure
        assert observation is not None
        assert isinstance(observation, np.ndarray)
        assert len(observation) == 52, f"Expected 52 features, got {len(observation)}"
        
        # Verify observation contains valid data (not all zeros/NaN)
        assert not np.isnan(observation).all(), "Observation contains all NaN values"
        assert not (observation == 0).all(), "Observation contains all zero values"
