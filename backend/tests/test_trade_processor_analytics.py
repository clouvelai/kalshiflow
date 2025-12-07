"""
Unit tests for TradeProcessor analytics broadcasting functionality.

This test suite validates the analytics broadcasting system in TradeProcessor,
focusing on the configuration, statistics tracking, and basic broadcast functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kalshiflow.trade_processor import TradeProcessor


class TestAnalyticsBroadcastConfiguration:
    """Test analytics broadcast configuration and timing."""
    
    @pytest.mark.asyncio
    async def test_analytics_interval_configuration(self):
        """Test analytics broadcast interval is configured correctly."""
        with patch('kalshiflow.trade_processor.get_database') as mock_get_db, \
             patch('kalshiflow.trade_processor.get_aggregator') as mock_get_agg, \
             patch('kalshiflow.trade_processor.get_analytics_service') as mock_get_analytics:
            
            # Mock all dependencies
            mock_db = AsyncMock()
            mock_db.initialize = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_agg = AsyncMock()
            mock_agg.start = AsyncMock()
            mock_agg.stop = AsyncMock()
            mock_get_agg.return_value = mock_agg
            
            mock_analytics = AsyncMock()
            mock_analytics.start = AsyncMock()
            mock_analytics.stop = AsyncMock()
            mock_get_analytics.return_value = mock_analytics
            
            processor = TradeProcessor()
            
            # Verify default interval is 1 second for smooth updates
            assert processor._analytics_interval == 1.0
            
            # Test that interval is reasonable for production
            assert processor._analytics_interval >= 0.5
            assert processor._analytics_interval <= 2.0
    
    @pytest.mark.asyncio
    async def test_hot_markets_interval_configuration(self):
        """Test hot markets broadcast interval is configured correctly."""
        with patch('kalshiflow.trade_processor.get_database') as mock_get_db, \
             patch('kalshiflow.trade_processor.get_aggregator') as mock_get_agg, \
             patch('kalshiflow.trade_processor.get_analytics_service') as mock_get_analytics:
            
            # Mock all dependencies
            mock_db = AsyncMock()
            mock_db.initialize = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_agg = AsyncMock()
            mock_agg.start = AsyncMock()
            mock_agg.stop = AsyncMock()
            mock_get_agg.return_value = mock_agg
            
            mock_analytics = AsyncMock()
            mock_analytics.start = AsyncMock()
            mock_analytics.stop = AsyncMock()
            mock_get_analytics.return_value = mock_analytics
            
            processor = TradeProcessor()
            
            # Verify default interval is 5 seconds
            assert processor._hot_markets_interval == 5
            
            # Test that interval is reasonable for production
            assert processor._hot_markets_interval >= 3.0
            assert processor._hot_markets_interval <= 10.0


class TestBroadcastStatistics:
    """Test analytics broadcast statistics tracking."""
    
    @pytest.mark.asyncio
    async def test_statistics_initialization(self):
        """Test that analytics broadcast statistics are properly initialized."""
        with patch('kalshiflow.trade_processor.get_database') as mock_get_db, \
             patch('kalshiflow.trade_processor.get_aggregator') as mock_get_agg, \
             patch('kalshiflow.trade_processor.get_analytics_service') as mock_get_analytics:
            
            # Mock all dependencies
            mock_db = AsyncMock()
            mock_db.initialize = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_agg = AsyncMock()
            mock_agg.start = AsyncMock()
            mock_agg.stop = AsyncMock()
            mock_agg.get_stats = MagicMock(return_value={})
            mock_get_agg.return_value = mock_agg
            
            mock_analytics = AsyncMock()
            mock_analytics.start = AsyncMock()
            mock_analytics.stop = AsyncMock()
            mock_analytics.get_stats = MagicMock(return_value={})
            mock_get_analytics.return_value = mock_analytics
            
            processor = TradeProcessor()
            
            # Check initial state
            stats = processor.get_stats()
            assert 'analytics_broadcasts_sent' in stats
            assert 'analytics_broadcast_errors' in stats
            assert stats['analytics_broadcasts_sent'] == 0
            assert stats['analytics_broadcast_errors'] == 0
            
            # Simulate some statistics updates
            processor.stats['analytics_broadcasts_sent'] = 10
            processor.stats['analytics_broadcast_errors'] = 2
            
            # Verify they're included in stats
            stats = processor.get_stats()
            assert stats['analytics_broadcasts_sent'] == 10
            assert stats['analytics_broadcast_errors'] == 2
    
    @pytest.mark.asyncio
    async def test_websocket_broadcaster_integration(self):
        """Test proper integration with WebSocket broadcaster."""
        with patch('kalshiflow.trade_processor.get_database') as mock_get_db, \
             patch('kalshiflow.trade_processor.get_aggregator') as mock_get_agg, \
             patch('kalshiflow.trade_processor.get_analytics_service') as mock_get_analytics:
            
            # Mock all dependencies
            mock_db = AsyncMock()
            mock_db.initialize = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_agg = AsyncMock()
            mock_agg.start = AsyncMock()
            mock_agg.stop = AsyncMock()
            mock_get_agg.return_value = mock_agg
            
            mock_analytics = AsyncMock()
            mock_analytics.start = AsyncMock()
            mock_analytics.stop = AsyncMock()
            mock_get_analytics.return_value = mock_analytics
            
            processor = TradeProcessor()
            
            # Create mock WebSocket broadcaster
            mock_broadcaster = AsyncMock()
            mock_broadcaster.broadcast = AsyncMock()
            mock_broadcaster.broadcast_analytics_update = AsyncMock()
            
            # Set broadcaster
            processor.set_websocket_broadcaster(mock_broadcaster)
            assert processor.websocket_broadcaster is mock_broadcaster
            
            # Verify it's accessible to broadcast loops
            assert processor.websocket_broadcaster is not None
    
    @pytest.mark.asyncio
    async def test_processor_lifecycle_with_broadcast_tasks(self):
        """Test that start/stop properly manages broadcast task lifecycle."""
        with patch('kalshiflow.trade_processor.get_database') as mock_get_db, \
             patch('kalshiflow.trade_processor.get_aggregator') as mock_get_agg, \
             patch('kalshiflow.trade_processor.get_analytics_service') as mock_get_analytics:
            
            # Mock all dependencies
            mock_db = AsyncMock()
            mock_db.initialize = AsyncMock()
            mock_get_db.return_value = mock_db
            
            mock_agg = AsyncMock()
            mock_agg.start = AsyncMock()
            mock_agg.stop = AsyncMock()
            mock_get_agg.return_value = mock_agg
            
            mock_analytics = AsyncMock()
            mock_analytics.start = AsyncMock()
            mock_analytics.stop = AsyncMock()
            mock_get_analytics.return_value = mock_analytics
            
            processor = TradeProcessor()
            
            # Verify tasks are not running initially
            assert processor._analytics_task is None
            assert processor._hot_markets_task is None
            
            # Start processor
            await processor.start()
            
            # Verify tasks are created and running
            assert processor._analytics_task is not None
            assert processor._hot_markets_task is not None
            assert not processor._analytics_task.done()
            assert not processor._hot_markets_task.done()
            
            # Store task references before stopping
            analytics_task = processor._analytics_task
            hot_markets_task = processor._hot_markets_task
            
            # Stop processor
            await processor.stop()
            
            # Verify tasks are done (either cancelled or completed) and references are cleared
            assert analytics_task.done()
            assert hot_markets_task.done()
            assert processor._analytics_task is None
            assert processor._hot_markets_task is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])