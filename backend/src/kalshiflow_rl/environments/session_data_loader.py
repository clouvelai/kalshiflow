"""
Session-based data loading for market-agnostic RL environments.

This module handles loading historical orderbook data by session_id to create
episodes with guaranteed data continuity. Sessions provide natural multi-market
coordination through timestamp grouping.

PRICE FORMAT CONVENTION:
- Database: Prices stored as integer cents (1-99)  
- Features: Prices normalized to probability space (0.01-0.99)
- Conversion: probability = cents / 100.0
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
import logging
import json
import asyncpg
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SessionDataPoint:
    """
    Single timestamped data point within a session.
    
    This represents market state at a specific timestamp, potentially
    covering multiple markets that were active at that time.
    """
    timestamp: datetime
    timestamp_ms: int
    markets_data: Dict[str, Dict[str, Any]]  # market_ticker -> reconstructed OrderbookState data
    
    # Market-agnostic features extracted from orderbook states (RAW FORMAT - database cents)
    spreads: Dict[str, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)  # ticker -> (yes_spread_cents, no_spread_cents)
    mid_prices: Dict[str, Tuple[Optional[Decimal], Optional[Decimal]]] = field(default_factory=dict)  # ticker -> (yes_mid_cents, no_mid_cents)
    depths: Dict[str, Dict[str, int]] = field(default_factory=dict)  # ticker -> {yes_bids_depth, yes_asks_depth, no_bids_depth, no_asks_depth}
    imbalances: Dict[str, Dict[str, float]] = field(default_factory=dict)  # ticker -> {yes_imbalance, no_imbalance}
    
    # Temporal features (computed by _add_temporal_features)
    time_gap: float = 0.0  # Seconds since previous data point
    activity_score: float = 0.0  # Market activity intensity [0,1]
    momentum: float = 0.0  # Price momentum indicator [-1,1]
    

@dataclass
class SessionData:
    """
    Complete session data for episode generation.
    
    Contains all timestamped data points for a session with metadata
    and pre-computed features for efficient episode execution.
    """
    session_id: int  # Use int to match database
    start_time: datetime
    end_time: datetime
    data_points: List[SessionDataPoint]
    
    # Session metadata
    markets_involved: List[str]
    total_duration: timedelta = field(init=False)
    data_quality_score: float = 1.0
    
    # Pre-computed temporal analysis
    temporal_gaps: List[float] = field(default_factory=list)  # Time gaps between data points
    activity_bursts: List[Tuple[int, int]] = field(default_factory=list)  # (start_idx, end_idx) of high activity
    quiet_periods: List[Tuple[int, int]] = field(default_factory=list)  # (start_idx, end_idx) of low activity
    
    # Episode statistics for curriculum learning
    avg_spread: float = 0.0  # Average spread across all markets/sides
    volatility_score: float = 0.0  # Price volatility measure
    market_diversity: float = 0.0  # Number of active markets normalized
    
    def __post_init__(self):
        """Calculate derived fields after initialization."""
        self.total_duration = self.end_time - self.start_time
        
    def get_timestep_data(self, step: int) -> Optional[SessionDataPoint]:
        """Get data for specific timestep in episode."""
        if 0 <= step < len(self.data_points):
            return self.data_points[step]
        return None
        
    def get_episode_length(self) -> int:
        """Total number of steps available in this session."""
        return len(self.data_points)


class SessionDataLoader:
    """
    Loads session-based orderbook data from database for RL training.
    
    This class handles the database interface for loading historical data
    organized by session_id. It provides efficient single-query loading
    and pre-processing of temporal features.
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize session data loader.
        
        Args:
            database_url: Optional database connection URL
        """
        self.database_url = database_url
        self._pool = None
        
        # Import the RLDatabase for connection
        from ..data.database import rl_db
        self._db = rl_db
        
    async def load_session(self, session_id: int) -> Optional[SessionData]:
        """
        Load complete session data with single database query.
        
        This method loads all orderbook data for the given session_id,
        groups it by timestamp for natural multi-market coordination,
        and pre-computes temporal features for efficiency.
        
        PRICE FORMAT: Returns raw database format (cents) - conversion to 
        probability space happens in feature extraction layer.
        
        Args:
            session_id: Session identifier to load
            
        Returns:
            SessionData with all timestamped data points containing:
            - Raw price data in cents (1-99) from database
            - Pre-computed temporal features
            - Market-agnostic aggregations
            Returns None if session not found
        """
        logger.info(f"Loading session data for session_id: {session_id}")
        
        try:
            # Initialize database connection if needed
            if not self._db._initialized:
                await self._db.initialize()
            
            # Get session metadata
            session_info = await self._db.get_session(session_id)
            if not session_info:
                logger.warning(f"Session {session_id} not found")
                return None
            
            # Load snapshots and deltas for this session
            snapshots = await self._db.get_session_snapshots(session_id)
            deltas = await self._db.get_session_deltas(session_id)
            
            if not snapshots:
                logger.warning(f"No snapshots found for session {session_id}")
                return None
            
            logger.info(f"Loaded {len(snapshots)} snapshots and {len(deltas)} deltas for session {session_id}")
            
            # Reconstruct orderbook states using existing OrderbookState functionality
            raw_data = self._reconstruct_orderbook_states(snapshots, deltas)
            
            # Group by timestamp for multi-market coordination
            data_points = self._group_by_timestamp(raw_data)
            
            if not data_points:
                logger.warning(f"No valid data points created for session {session_id}")
                return None
            
            # Create session data object
            session_data = SessionData(
                session_id=session_id,
                start_time=session_info['started_at'],
                end_time=session_info['ended_at'] or datetime.now(),
                data_points=data_points,
                markets_involved=session_info['market_tickers']
            )
            
            # Add temporal features
            self._add_temporal_features(session_data.data_points)
            
            # Compute session-level statistics for curriculum learning
            self._compute_session_stats(session_data)
            
            logger.info(f"Successfully loaded session {session_id} with {len(data_points)} data points")
            return session_data
            
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return None
    
    async def get_available_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of all available sessions with metadata.
        
        Returns:
            List of dictionaries containing session metadata:
                - session_id: Unique session identifier
                - started_at: Session start timestamp
                - ended_at: Session end timestamp
                - status: Session status (closed/active)
                - market_tickers: List of market tickers
                - num_markets: Number of markets in session
                - snapshots_count: Number of orderbook snapshots
                - deltas_count: Number of orderbook deltas
                - duration: Session duration (timedelta)
        """
        try:
            if not self._db._initialized:
                await self._db.initialize()
            
            # Get all closed sessions with full metadata (no filtering)
            async with self._db.get_connection() as conn:
                rows = await conn.fetch("""
                    SELECT session_id, started_at, ended_at, status, 
                           market_tickers,
                           array_length(market_tickers, 1) as num_markets,
                           snapshots_count, deltas_count,
                           ended_at - started_at as duration
                    FROM rl_orderbook_sessions 
                    WHERE ended_at IS NOT NULL 
                    AND status = 'closed'
                    ORDER BY started_at DESC
                """)
                
                # Convert rows to dictionaries with all metadata
                available_sessions = [dict(row) for row in rows]
                logger.info(f"Found {len(available_sessions)} available sessions")
                return available_sessions
                
        except Exception as e:
            logger.error(f"Error getting available sessions: {e}")
            return []
    
    async def get_available_session_ids(self) -> List[int]:
        """
        Get list of available session IDs (convenience method).
        
        Returns:
            List of session IDs available for loading
        """
        sessions = await self.get_available_sessions()
        return [s['session_id'] for s in sessions]
    
    async def validate_session_quality(self, session_id: int) -> float:
        """
        Validate data quality for a session.
        
        Checks for missing data, gaps, and other quality issues.
        
        Args:
            session_id: Session to validate
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            if not self._db._initialized:
                await self._db.initialize()
            
            session_info = await self._db.get_session(session_id)
            if not session_info:
                return 0.0
            
            # Check basic data availability
            snapshots_count = session_info.get('snapshots_count', 0)
            deltas_count = session_info.get('deltas_count', 0)
            
            # Calculate quality score based on data completeness
            quality_score = 1.0
            
            # Penalize sessions with too little data
            if snapshots_count < 10:
                quality_score *= 0.3
            elif snapshots_count < 50:
                quality_score *= 0.7
            
            # Check for reasonable delta/snapshot ratio
            if snapshots_count > 0:
                delta_ratio = deltas_count / snapshots_count
                if delta_ratio < 1:  # Should have more deltas than snapshots typically
                    quality_score *= 0.8
            
            # Check session duration
            if session_info['ended_at'] and session_info['started_at']:
                duration = session_info['ended_at'] - session_info['started_at']
                if duration.total_seconds() < 300:  # Less than 5 minutes
                    quality_score *= 0.5
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error validating session {session_id}: {e}")
            return 0.0
    
    def _group_by_timestamp(self, raw_data: List[Dict[str, Any]]) -> List[SessionDataPoint]:
        """
        Group orderbook snapshots by timestamp for multi-market coordination.
        
        This creates natural coordination points where multiple markets
        can be observed and acted upon simultaneously.
        
        PRICE FORMAT: Preserves raw database cents format (1-99). 
        Feature extraction will convert to probability space later.
        
        Args:
            raw_data: Raw reconstructed orderbook state data (prices in cents)
            
        Returns:
            List of SessionDataPoint grouped by timestamp with:
            - spreads/mid_prices in raw cents format
            - Volume/depth calculations as absolute values
            - Imbalances as normalized ratios [-1, 1]
        """
        # Group data by timestamp
        timestamp_groups = defaultdict(dict)
        
        for entry in raw_data:
            timestamp_ms = entry['timestamp_ms']
            market_ticker = entry['market_ticker']
            timestamp_groups[timestamp_ms][market_ticker] = entry
        
        # Create SessionDataPoint objects
        data_points = []
        
        for timestamp_ms in sorted(timestamp_groups.keys()):
            markets_data = timestamp_groups[timestamp_ms]
            
            # Convert timestamp to datetime
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)
            
            # Extract market-agnostic features from orderbook states
            spreads = {}
            mid_prices = {}
            depths = {}
            imbalances = {}
            
            for market_ticker, market_data in markets_data.items():
                # Extract features from reconstructed orderbook state
                yes_spread = market_data.get('yes_spread')
                no_spread = market_data.get('no_spread')
                spreads[market_ticker] = (yes_spread, no_spread)
                
                yes_mid = market_data.get('yes_mid_price')
                no_mid = market_data.get('no_mid_price')
                if yes_mid is not None:
                    yes_mid = Decimal(str(yes_mid))
                if no_mid is not None:
                    no_mid = Decimal(str(no_mid))
                mid_prices[market_ticker] = (yes_mid, no_mid)
                
                # Calculate depths (sum of top 5 levels)
                yes_bids = market_data.get('yes_bids', {})
                yes_asks = market_data.get('yes_asks', {})
                no_bids = market_data.get('no_bids', {})
                no_asks = market_data.get('no_asks', {})
                
                depths[market_ticker] = {
                    'yes_bids_depth': sum(list(yes_bids.values())[:5]) if yes_bids else 0,
                    'yes_asks_depth': sum(list(yes_asks.values())[:5]) if yes_asks else 0,
                    'no_bids_depth': sum(list(no_bids.values())[:5]) if no_bids else 0,
                    'no_asks_depth': sum(list(no_asks.values())[:5]) if no_asks else 0
                }
                
                # Calculate order book imbalances
                yes_bid_vol = sum(yes_bids.values()) if yes_bids else 0
                yes_ask_vol = sum(yes_asks.values()) if yes_asks else 0
                no_bid_vol = sum(no_bids.values()) if no_bids else 0
                no_ask_vol = sum(no_asks.values()) if no_asks else 0
                
                yes_total = yes_bid_vol + yes_ask_vol
                no_total = no_bid_vol + no_ask_vol
                
                yes_imbalance = (yes_bid_vol - yes_ask_vol) / yes_total if yes_total > 0 else 0.0
                no_imbalance = (no_bid_vol - no_ask_vol) / no_total if no_total > 0 else 0.0
                
                imbalances[market_ticker] = {
                    'yes_imbalance': yes_imbalance,
                    'no_imbalance': no_imbalance
                }
            
            # Create data point
            data_point = SessionDataPoint(
                timestamp=timestamp,
                timestamp_ms=timestamp_ms,
                markets_data=markets_data,
                spreads=spreads,
                mid_prices=mid_prices,
                depths=depths,
                imbalances=imbalances
            )
            
            data_points.append(data_point)
        
        logger.info(f"Created {len(data_points)} data points from {len(raw_data)} raw entries")
        return data_points
    
    def _add_temporal_features(self, data_points: List[SessionDataPoint]) -> None:
        """
        Add temporal features like time gaps and activity analysis.
        
        Calculates:
        - Time gaps between data points
        - Activity burst detection
        - Quiet period identification
        - Momentum indicators
        
        Args:
            data_points: Session data points to enhance with temporal features
        """
        if len(data_points) < 2:
            return
        
        # Calculate time gaps and activity metrics
        prev_timestamp = None
        mid_prices_history = defaultdict(list)  # For momentum calculation
        
        for i, point in enumerate(data_points):
            # Calculate time gap
            if prev_timestamp is not None:
                time_gap = (point.timestamp_ms - prev_timestamp) / 1000.0  # Convert to seconds
                point.time_gap = time_gap
            else:
                point.time_gap = 0.0
            
            prev_timestamp = point.timestamp_ms
            
            # Calculate activity score based on number of markets and total volume
            num_markets = len(point.markets_data)
            total_volume = sum(
                market_data.get('total_volume', 0) 
                for market_data in point.markets_data.values()
            )
            
            # Normalize activity score [0,1]
            point.activity_score = min(1.0, (num_markets * np.log(1 + total_volume)) / 1000.0)
            
            # Track mid prices for momentum calculation
            for market_ticker in point.mid_prices:
                yes_mid, no_mid = point.mid_prices[market_ticker]
                if yes_mid is not None:
                    mid_prices_history[market_ticker].append(float(yes_mid))
            
            # Calculate momentum (price change direction)
            momentum_sum = 0.0
            momentum_count = 0
            
            for market_ticker in mid_prices_history:
                prices = mid_prices_history[market_ticker]
                if len(prices) >= 3:  # Need at least 3 points for trend
                    recent_change = prices[-1] - prices[-2]
                    prev_change = prices[-2] - prices[-3]
                    
                    # Momentum is acceleration of price changes
                    if abs(prev_change) > 0.001:  # Avoid division by small numbers
                        momentum = (recent_change - prev_change) / abs(prev_change)
                        momentum_sum += np.tanh(momentum)  # Bound to [-1,1]
                        momentum_count += 1
            
            if momentum_count > 0:
                point.momentum = momentum_sum / momentum_count
            else:
                point.momentum = 0.0
        
        logger.info(f"Added temporal features to {len(data_points)} data points")
    
    def _reconstruct_orderbook_states(self, snapshots: List[Dict[str, Any]], deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reconstruct orderbook states using existing OrderbookState functionality.
        
        Args:
            snapshots: List of orderbook snapshots from database
            deltas: List of orderbook deltas from database
            
        Returns:
            List of reconstructed orderbook states with market-agnostic features
        """
        from ..data.orderbook_state import OrderbookState
        
        # Group by market ticker
        market_snapshots = defaultdict(list)
        market_deltas = defaultdict(list)
        
        for snapshot in snapshots:
            market_snapshots[snapshot['market_ticker']].append(snapshot)
        
        for delta in deltas:
            market_deltas[delta['market_ticker']].append(delta)
        
        reconstructed_states = []
        
        # Process each market separately
        for market_ticker in market_snapshots.keys():
            market_snaps = sorted(market_snapshots[market_ticker], key=lambda x: x['sequence_number'])
            market_dels = sorted(market_deltas[market_ticker], key=lambda x: x['sequence_number'])
            
            # Create orderbook state
            orderbook = OrderbookState(market_ticker)
            
            # Apply snapshots and deltas in sequence order
            snap_idx = 0
            delta_idx = 0
            
            while snap_idx < len(market_snaps) or delta_idx < len(market_dels):
                # Determine which to process next based on sequence number
                process_snapshot = False
                
                if snap_idx >= len(market_snaps):
                    # Only deltas left
                    process_snapshot = False
                elif delta_idx >= len(market_dels):
                    # Only snapshots left
                    process_snapshot = True
                else:
                    # Both available, choose by sequence number
                    snap_seq = market_snaps[snap_idx]['sequence_number']
                    delta_seq = market_dels[delta_idx]['sequence_number']
                    process_snapshot = snap_seq <= delta_seq
                
                if process_snapshot and snap_idx < len(market_snaps):
                    snapshot = market_snaps[snap_idx]
                    
                    # Convert database format to OrderbookState format
                    snapshot_data = {
                        'sequence_number': snapshot['sequence_number'],
                        'timestamp_ms': snapshot['timestamp_ms'],
                        'yes_bids': json.loads(snapshot['yes_bids']) if isinstance(snapshot['yes_bids'], str) else snapshot['yes_bids'],
                        'yes_asks': json.loads(snapshot['yes_asks']) if isinstance(snapshot['yes_asks'], str) else snapshot['yes_asks'],
                        'no_bids': json.loads(snapshot['no_bids']) if isinstance(snapshot['no_bids'], str) else snapshot['no_bids'],
                        'no_asks': json.loads(snapshot['no_asks']) if isinstance(snapshot['no_asks'], str) else snapshot['no_asks']
                    }
                    
                    orderbook.apply_snapshot(snapshot_data)
                    
                    # Store reconstructed state
                    state_data = orderbook.to_dict()
                    state_data['timestamp_ms'] = snapshot['timestamp_ms']
                    reconstructed_states.append(state_data)
                    
                    snap_idx += 1
                    
                elif delta_idx < len(market_dels):
                    delta = market_dels[delta_idx]
                    
                    # Apply delta
                    success = orderbook.apply_delta(delta)
                    
                    if success:
                        # Store reconstructed state after delta
                        state_data = orderbook.to_dict()
                        state_data['timestamp_ms'] = delta['timestamp_ms']
                        reconstructed_states.append(state_data)
                    
                    delta_idx += 1
        
        # Sort by timestamp for temporal consistency
        reconstructed_states.sort(key=lambda x: x['timestamp_ms'])
        
        logger.info(f"Reconstructed {len(reconstructed_states)} orderbook states")
        return reconstructed_states
    
    def _compute_session_stats(self, session_data: SessionData) -> None:
        """
        Compute session-level statistics for curriculum learning.
        
        Args:
            session_data: Session data to analyze
        """
        if not session_data.data_points:
            return
        
        # Calculate average spread across all markets and sides
        spreads = []
        volatilities = []
        market_set = set()
        
        for point in session_data.data_points:
            for market_ticker, (yes_spread, no_spread) in point.spreads.items():
                market_set.add(market_ticker)
                if yes_spread is not None:
                    spreads.append(yes_spread)
                if no_spread is not None:
                    spreads.append(no_spread)
            
            # Track price volatility via momentum
            volatilities.append(abs(point.momentum))
        
        session_data.avg_spread = np.mean(spreads) if spreads else 0.0
        session_data.volatility_score = np.mean(volatilities) if volatilities else 0.0
        session_data.market_diversity = len(market_set) / 10.0  # Normalize assuming max 10 markets
        
        # Identify activity bursts and quiet periods
        activity_scores = [point.activity_score for point in session_data.data_points]
        activity_threshold = np.percentile(activity_scores, 75) if activity_scores else 0.5
        quiet_threshold = np.percentile(activity_scores, 25) if activity_scores else 0.2
        
        # Find bursts (consecutive high activity periods)
        in_burst = False
        burst_start = 0
        
        for i, score in enumerate(activity_scores):
            if score >= activity_threshold and not in_burst:
                in_burst = True
                burst_start = i
            elif score < activity_threshold and in_burst:
                in_burst = False
                session_data.activity_bursts.append((burst_start, i-1))
        
        if in_burst:  # Close final burst
            session_data.activity_bursts.append((burst_start, len(activity_scores)-1))
        
        # Find quiet periods
        in_quiet = False
        quiet_start = 0
        
        for i, score in enumerate(activity_scores):
            if score <= quiet_threshold and not in_quiet:
                in_quiet = True
                quiet_start = i
            elif score > quiet_threshold and in_quiet:
                in_quiet = False
                session_data.quiet_periods.append((quiet_start, i-1))
        
        if in_quiet:  # Close final quiet period
            session_data.quiet_periods.append((quiet_start, len(activity_scores)-1))
        
        # Compute temporal gaps
        session_data.temporal_gaps = [point.time_gap for point in session_data.data_points]
        
        logger.info(
            f"Session stats - Avg spread: {session_data.avg_spread:.2f}, "
            f"Volatility: {session_data.volatility_score:.3f}, "
            f"Markets: {len(market_set)}, "
            f"Bursts: {len(session_data.activity_bursts)}, "
            f"Quiet periods: {len(session_data.quiet_periods)}"
        )
    
    async def close(self) -> None:
        """Clean up database connections."""
        # The RLDatabase handles connection pooling, no explicit cleanup needed
        pass