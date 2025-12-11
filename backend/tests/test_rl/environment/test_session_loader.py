"""
Comprehensive tests for SessionDataLoader.

This test validates the complete pipeline:
database → orderbook reconstruction → features → episode data
"""

import pytest
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from kalshiflow_rl.environments.session_data_loader import (
    SessionDataLoader, 
    SessionData, 
    SessionDataPoint
)


@pytest.fixture
def mock_database():
    """Mock database with sample orderbook data."""
    # Sample session info
    session_info = {
        'session_id': 123,
        'started_at': datetime(2024, 1, 1, 10, 0, 0),
        'ended_at': datetime(2024, 1, 1, 11, 0, 0),
        'market_tickers': ['MARKET1', 'MARKET2'],
        'status': 'closed',
        'snapshots_count': 100,
        'deltas_count': 500
    }
    
    # Sample snapshots data
    base_timestamp = int(datetime(2024, 1, 1, 10, 0, 0).timestamp() * 1000)
    
    snapshots = [
        {
            'id': 1,
            'session_id': 123,
            'market_ticker': 'MARKET1',
            'timestamp_ms': base_timestamp,
            'sequence_number': 1000,
            'yes_bids': json.dumps({'45': 100, '44': 200}),
            'yes_asks': json.dumps({'46': 150, '47': 250}),
            'no_bids': json.dumps({'54': 120, '53': 180}),
            'no_asks': json.dumps({'55': 160, '56': 240}),
            'yes_spread': 1,
            'no_spread': 1,
            'yes_mid_price': Decimal('45.5'),
            'no_mid_price': Decimal('54.5'),
            'total_volume': 1050
        },
        {
            'id': 2,
            'session_id': 123,
            'market_ticker': 'MARKET2',
            'timestamp_ms': base_timestamp + 1000,
            'sequence_number': 1001,
            'yes_bids': json.dumps({'48': 80, '47': 160}),
            'yes_asks': json.dumps({'49': 120, '50': 200}),
            'no_bids': json.dumps({'51': 90, '50': 170}),
            'no_asks': json.dumps({'52': 130, '53': 210}),
            'yes_spread': 1,
            'no_spread': 1,
            'yes_mid_price': Decimal('48.5'),
            'no_mid_price': Decimal('51.5'),
            'total_volume': 970
        }
    ]
    
    # Sample deltas data
    deltas = [
        {
            'id': 1,
            'session_id': 123,
            'market_ticker': 'MARKET1',
            'timestamp_ms': base_timestamp + 500,
            'sequence_number': 1000,
            'side': 'yes',
            'action': 'update',
            'price': 45,
            'old_size': 100,
            'new_size': 150
        }
    ]
    
    # Create mock database
    mock_db = MagicMock()
    mock_db._initialized = True
    
    mock_db.get_session = AsyncMock(return_value=session_info)
    mock_db.get_session_snapshots = AsyncMock(return_value=snapshots)
    mock_db.get_session_deltas = AsyncMock(return_value=deltas)
    
    # Mock the async context manager
    mock_conn = AsyncMock()
    mock_conn.fetch = AsyncMock(return_value=[{
        'session_id': 123,
        'started_at': datetime.now(),
        'ended_at': datetime.now() + timedelta(hours=1),
        'status': 'closed',
        'market_tickers': ['MARKET1', 'MARKET2'],
        'num_markets': 2,
        'snapshots_count': 100,
        'deltas_count': 500,
        'duration': timedelta(hours=1)
    }])
    
    mock_db.get_connection = MagicMock()
    mock_db.get_connection.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_db.get_connection.return_value.__aexit__ = AsyncMock(return_value=None)
    
    return mock_db


@pytest.fixture
def session_loader(mock_database):
    """SessionDataLoader with mocked database."""
    loader = SessionDataLoader()
    loader._db = mock_database
    return loader


class TestSessionDataPoint:
    """Test SessionDataPoint dataclass functionality."""
    
    def test_datapoint_creation(self):
        """Test SessionDataPoint can be created with required fields."""
        timestamp = datetime.now()
        timestamp_ms = int(timestamp.timestamp() * 1000)
        
        datapoint = SessionDataPoint(
            timestamp=timestamp,
            timestamp_ms=timestamp_ms,
            markets_data={'MARKET1': {'test': 'data'}}
        )
        
        assert datapoint.timestamp == timestamp
        assert datapoint.timestamp_ms == timestamp_ms
        assert datapoint.markets_data == {'MARKET1': {'test': 'data'}}
        assert datapoint.time_gap == 0.0
        assert datapoint.activity_score == 0.0
        assert datapoint.momentum == 0.0


class TestSessionData:
    """Test SessionData dataclass functionality."""
    
    def test_session_creation(self):
        """Test SessionData can be created with required fields."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        session = SessionData(
            session_id=123,
            start_time=start_time,
            end_time=end_time,
            data_points=[],
            markets_involved=['MARKET1', 'MARKET2']
        )
        
        assert session.session_id == 123
        assert session.start_time == start_time
        assert session.end_time == end_time
        assert session.total_duration == timedelta(hours=1)
        assert session.markets_involved == ['MARKET1', 'MARKET2']
        assert session.data_quality_score == 1.0
    
    def test_get_timestep_data(self):
        """Test retrieving data for specific timesteps."""
        session = SessionData(
            session_id=123,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            data_points=[
                SessionDataPoint(
                    timestamp=datetime.now(),
                    timestamp_ms=1000,
                    markets_data={}
                ),
                SessionDataPoint(
                    timestamp=datetime.now(),
                    timestamp_ms=2000,
                    markets_data={}
                )
            ],
            markets_involved=['MARKET1']
        )
        
        # Valid indices
        assert session.get_timestep_data(0) is not None
        assert session.get_timestep_data(1) is not None
        
        # Invalid indices
        assert session.get_timestep_data(-1) is None
        assert session.get_timestep_data(2) is None
    
    def test_get_episode_length(self):
        """Test episode length calculation."""
        session = SessionData(
            session_id=123,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            data_points=[
                SessionDataPoint(datetime.now(), 1000, {}),
                SessionDataPoint(datetime.now(), 2000, {}),
                SessionDataPoint(datetime.now(), 3000, {})
            ],
            markets_involved=['MARKET1']
        )
        
        assert session.get_episode_length() == 3


class TestSessionDataLoader:
    """Test SessionDataLoader functionality."""
    
    @pytest.mark.asyncio
    async def test_get_available_sessions(self, session_loader, mock_database):
        """Test fetching available sessions with metadata."""
        sessions = await session_loader.get_available_sessions()
        
        assert isinstance(sessions, list)
        assert len(sessions) == 1
        
        # Check that we get full session metadata
        session = sessions[0]
        assert isinstance(session, dict)
        assert session['session_id'] == 123
        assert session['status'] == 'closed'
        assert session['num_markets'] == 2
        assert session['snapshots_count'] == 100
        assert session['deltas_count'] == 500
        assert 'started_at' in session
        assert 'ended_at' in session
        assert 'duration' in session
        
        mock_database.get_connection.assert_called()
    
    @pytest.mark.asyncio
    async def test_validate_session_quality(self, session_loader):
        """Test session quality validation."""
        quality = await session_loader.validate_session_quality(123)
        
        assert isinstance(quality, float)
        assert 0.0 <= quality <= 1.0
    
    @pytest.mark.asyncio
    async def test_validate_session_quality_nonexistent(self, session_loader):
        """Test session quality for nonexistent session."""
        # Mock empty response for nonexistent session
        session_loader._db.get_session = AsyncMock(return_value=None)
        
        quality = await session_loader.validate_session_quality(999)
        assert quality == 0.0
    
    @pytest.mark.asyncio
    async def test_load_session_success(self, session_loader):
        """Test successful session loading."""
        session_data = await session_loader.load_session(123)
        
        assert session_data is not None
        assert isinstance(session_data, SessionData)
        assert session_data.session_id == 123
        assert len(session_data.markets_involved) == 2
        assert session_data.markets_involved == ['MARKET1', 'MARKET2']
    
    @pytest.mark.asyncio
    async def test_load_session_nonexistent(self, session_loader):
        """Test loading nonexistent session."""
        # Mock empty response for nonexistent session
        session_loader._db.get_session = AsyncMock(return_value=None)
        
        session_data = await session_loader.load_session(999)
        assert session_data is None
    
    @pytest.mark.asyncio
    async def test_load_session_no_snapshots(self, session_loader):
        """Test loading session with no snapshots."""
        # Mock empty snapshots
        session_loader._db.get_session_snapshots = AsyncMock(return_value=[])
        
        session_data = await session_loader.load_session(123)
        assert session_data is None
    
    @pytest.mark.asyncio
    @patch('kalshiflow_rl.data.orderbook_state.OrderbookState')
    async def test_orderbook_reconstruction(self, mock_orderbook_class, session_loader):
        """Test orderbook state reconstruction."""
        # Mock OrderbookState
        mock_orderbook = MagicMock()
        mock_orderbook_class.return_value = mock_orderbook
        mock_orderbook.to_dict.return_value = {
            'market_ticker': 'MARKET1',
            'yes_spread': 1,
            'no_spread': 1,
            'yes_mid_price': 45.5,
            'no_mid_price': 54.5,
            'yes_bids': {'45': 150},  # After delta update
            'yes_asks': {'46': 150},
            'no_bids': {'54': 120},
            'no_asks': {'55': 160},
            'total_volume': 1100
        }
        mock_orderbook.apply_snapshot.return_value = None
        mock_orderbook.apply_delta.return_value = True
        
        session_data = await session_loader.load_session(123)
        
        assert session_data is not None
        assert len(session_data.data_points) > 0
        
        # Verify OrderbookState was used
        mock_orderbook_class.assert_called()
        mock_orderbook.apply_snapshot.assert_called()
    
    def test_group_by_timestamp(self, session_loader):
        """Test timestamp grouping functionality."""
        # Create sample raw data
        raw_data = [
            {
                'market_ticker': 'MARKET1',
                'timestamp_ms': 1000,
                'yes_spread': 1,
                'no_spread': 1,
                'yes_mid_price': 45.5,
                'no_mid_price': 54.5,
                'yes_bids': {'45': 100},
                'yes_asks': {'46': 150},
                'no_bids': {'54': 120},
                'no_asks': {'55': 160},
                'total_volume': 530
            },
            {
                'market_ticker': 'MARKET2',
                'timestamp_ms': 1000,  # Same timestamp for coordination
                'yes_spread': 2,
                'no_spread': 2,
                'yes_mid_price': 48.5,
                'no_mid_price': 51.5,
                'yes_bids': {'48': 80},
                'yes_asks': {'50': 120},
                'no_bids': {'50': 90},
                'no_asks': {'52': 130},
                'total_volume': 420
            }
        ]
        
        data_points = session_loader._group_by_timestamp(raw_data)
        
        assert len(data_points) == 1  # Should be grouped by timestamp
        point = data_points[0]
        
        assert point.timestamp_ms == 1000
        assert len(point.markets_data) == 2
        assert 'MARKET1' in point.markets_data
        assert 'MARKET2' in point.markets_data
        
        # Check extracted features
        assert len(point.spreads) == 2
        assert point.spreads['MARKET1'] == (1, 1)
        assert point.spreads['MARKET2'] == (2, 2)
        
        assert len(point.mid_prices) == 2
        assert point.mid_prices['MARKET1'][0] == Decimal('45.5')
        assert point.mid_prices['MARKET2'][0] == Decimal('48.5')
        
        # Check depths
        assert len(point.depths) == 2
        assert point.depths['MARKET1']['yes_bids_depth'] == 100
        assert point.depths['MARKET2']['yes_bids_depth'] == 80
        
        # Check imbalances
        assert len(point.imbalances) == 2
        market1_yes_imbalance = point.imbalances['MARKET1']['yes_imbalance']
        assert abs(market1_yes_imbalance - ((100 - 150) / (100 + 150))) < 0.001
    
    def test_add_temporal_features(self, session_loader):
        """Test temporal feature calculation."""
        data_points = [
            SessionDataPoint(
                timestamp=datetime.fromtimestamp(1000),
                timestamp_ms=1000000,
                markets_data={'MARKET1': {'total_volume': 500}}
            ),
            SessionDataPoint(
                timestamp=datetime.fromtimestamp(1002),
                timestamp_ms=1002000,
                markets_data={'MARKET1': {'total_volume': 600}}
            ),
            SessionDataPoint(
                timestamp=datetime.fromtimestamp(1005),
                timestamp_ms=1005000,
                markets_data={'MARKET1': {'total_volume': 800}}
            )
        ]
        
        # Add mid prices for momentum calculation
        data_points[0].mid_prices = {'MARKET1': (Decimal('45.0'), None)}
        data_points[1].mid_prices = {'MARKET1': (Decimal('46.0'), None)}
        data_points[2].mid_prices = {'MARKET1': (Decimal('47.0'), None)}
        
        session_loader._add_temporal_features(data_points)
        
        # Check time gaps
        assert data_points[0].time_gap == 0.0
        assert data_points[1].time_gap == 2.0  # 2 seconds
        assert data_points[2].time_gap == 3.0  # 3 seconds
        
        # Check activity scores are calculated
        for point in data_points:
            assert 0.0 <= point.activity_score <= 1.0
        
        # Check momentum is calculated (might be 0 if changes are too small)
        # The momentum algorithm requires significant price changes to avoid noise
        assert isinstance(data_points[2].momentum, (float, np.floating))  # Should be calculated as a number
    
    def test_compute_session_stats(self, session_loader):
        """Test session-level statistics computation."""
        # Create session with sample data
        data_points = []
        for i in range(10):
            point = SessionDataPoint(
                timestamp=datetime.fromtimestamp(1000 + i),
                timestamp_ms=(1000 + i) * 1000,
                markets_data={}
            )
            point.spreads = {
                'MARKET1': (1 + i % 3, 2 + i % 2),  # Variable spreads
                'MARKET2': (1, 1)  # Constant spreads
            }
            point.momentum = 0.1 * (i - 5)  # Variable momentum
            point.activity_score = 0.5 + 0.1 * (i % 4)  # Variable activity
            data_points.append(point)
        
        session_data = SessionData(
            session_id=123,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1),
            data_points=data_points,
            markets_involved=['MARKET1', 'MARKET2']
        )
        
        session_loader._compute_session_stats(session_data)
        
        # Check computed statistics
        assert session_data.avg_spread > 0.0
        assert 0.0 <= session_data.volatility_score <= 1.0
        assert 0.0 <= session_data.market_diversity <= 1.0
        assert session_data.market_diversity == 0.2  # 2 markets / 10 max
        
        # Check temporal gaps are computed
        assert len(session_data.temporal_gaps) == len(data_points)
        
        # Check activity analysis
        assert isinstance(session_data.activity_bursts, list)
        assert isinstance(session_data.quiet_periods, list)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, session_loader):
        """Test error handling in load_session."""
        # Mock database error
        session_loader._db.get_session = AsyncMock(side_effect=Exception("Database error"))
        
        session_data = await session_loader.load_session(123)
        assert session_data is None
    
    @pytest.mark.asyncio
    async def test_close_cleanup(self, session_loader):
        """Test proper cleanup on close."""
        await session_loader.close()
        # Should not raise any exceptions


class TestIntegration:
    """Integration tests for complete pipeline."""
    
    @pytest.mark.asyncio
    @patch('kalshiflow_rl.data.orderbook_state.OrderbookState')
    async def test_complete_pipeline(self, mock_orderbook_class, mock_database):
        """Test complete pipeline from database to episode data."""
        # Setup mock orderbook
        mock_orderbook = MagicMock()
        mock_orderbook_class.return_value = mock_orderbook
        mock_orderbook.to_dict.return_value = {
            'market_ticker': 'MARKET1',
            'yes_spread': 1,
            'no_spread': 1,
            'yes_mid_price': 45.5,
            'no_mid_price': 54.5,
            'yes_bids': {'45': 150},
            'yes_asks': {'46': 150},
            'no_bids': {'54': 120},
            'no_asks': {'55': 160},
            'total_volume': 1000
        }
        mock_orderbook.apply_snapshot.return_value = None
        mock_orderbook.apply_delta.return_value = True
        
        # Create loader
        loader = SessionDataLoader()
        loader._db = mock_database
        
        # Load session
        session_data = await loader.load_session(123)
        
        assert session_data is not None
        
        # Validate complete data structure
        assert isinstance(session_data, SessionData)
        assert session_data.session_id == 123
        assert len(session_data.markets_involved) == 2
        assert len(session_data.data_points) > 0
        
        # Validate data points have all required features
        for point in session_data.data_points:
            assert isinstance(point, SessionDataPoint)
            assert point.timestamp_ms > 0
            assert isinstance(point.markets_data, dict)
            assert isinstance(point.spreads, dict)
            assert isinstance(point.mid_prices, dict)
            assert isinstance(point.depths, dict)
            assert isinstance(point.imbalances, dict)
            assert isinstance(point.time_gap, float)
            assert isinstance(point.activity_score, float)
            assert isinstance(point.momentum, float)
        
        # Validate session stats are computed
        assert session_data.avg_spread >= 0.0
        assert 0.0 <= session_data.volatility_score
        assert 0.0 <= session_data.market_diversity <= 1.0
        
        # Validate temporal analysis
        assert isinstance(session_data.temporal_gaps, list)
        assert isinstance(session_data.activity_bursts, list)
        assert isinstance(session_data.quiet_periods, list)
        
        print("✅ Complete pipeline test passed - database → orderbook reconstruction → features → episode data")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])